import numpy as np
import time, os, types
import pickle
from tqdm.auto import tqdm
from tqdm import trange
from ..loader import patterns as pt
from ..utils.plot_func import create_filepath, save_raw
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from aotools.functions import phaseFromZernikes, zernike_noll



class IterationAlgos():
    """ This is the base class to be used for various iteration based algorithms for optimization
        including Stepwise Sequential (SSA), Continuous Sequential (CSA), Random Partition (RPA) 
        and Hadamard Partition (HPA) algorithms. 
    """
        
    def __init__(self, 
                 slm, camera,
                 pattern_loader,
                 slm_macropixel=112,
                 slm_resolution=(800, 600),
                 calib_px=112, 
                 total_iterations=1,
                 phase_steps=8, 
                 remote=True,
                 type=None, 
                 save_path=None):
        
        # slm settings
        # self.N = int(slm_segments ** 0.5)
        self.slm_macropixel = slm_macropixel
        self.calib_px = calib_px
        self.m = phase_steps

        resX, resY = slm_resolution
        self.patternSLM = pt.PatternsBacic(resX, resY)
        
        self.total_iterations = total_iterations

        # hardware objects
        self.slm = slm
        self.camera = camera
        
        # loader
        self.pattern_loader = pattern_loader
        
        # is the slm remotely connected to a rasp pi ?
        self.remote = remote
        
        # save raw data path
        self.save_path = save_path


    def register_callback(self, callback):
        """ This callback function is used to pass a custom cost function
            to the optimization object

        Parameters
        ----------
        callback
            the cost function
        """
        self.callback = callback
        
    # def create_filepath(self, **kwargs): staticmethod(create_filepath(self, **kwargs))
        
    
    def _phi_k(self, k):
        """ Returns the slm-calibrated phase sweep values, the discretization of which is defined by the
            user (int m)

        Parameters
        ----------
        k (int)
        Returns
        -------
        phi (int) calibrated for the SLM
        """
        return (2 * (k + 1) / self.m) * self.calib_px / 2
    
    def _create_pattern(self, k, mask, pattern):
        """ Prepares a pattern ready for the SLM: enlarges it according to the 
            macropixel size and adds it in the middle of the SLM screen
        """
        phi = self.phi_k(k)
        pattern[mask] = phi
        # temp = self.patternSLM.pattern2SLM(pattern, self.slm_macropixel)
        temp = self.patternSLM._enlarge_pattern(pattern, self.slm_macropixel)
        temp = self.patternSLM.add_subpattern(temp)
        return temp

    def _upload_pattern(self, pattern, slm_delay=0.1):
        """ Uploads a pattern to the SLM either in remote or local mode. Adds a user-defined
            time delay to make sure that the pattern is uploaded. 
        """
        if self.remote:
            self.slm.sendArray(pattern)
        else:
            self.slm.updateArray(pattern)
        time.sleep(slm_delay)
        
    
    def _get_frame(self):
        """ Get frame from zelux thorlabs camera
        """
        frame = self.camera.get_pending_frame_or_null()
        image_buffer = np.copy(frame.image_buffer)
        return image_buffer
    
    def _phase2SLM(self, mask):
        """ Converts a phase mask to the SLM readable  and phase-calibrated format
        """
        arg = np.angle(mask, deg=False) # gives angle betwwen -pi and pi
        # scale phase between 0 and 2pi
        # arg2pi = (arg + 2 * np.pi) % (2 * np.pi)
        arg2pi = arg + np.pi
        # normalize to SLM 2pi calibration value
        arg2SLM = arg2pi * self.calib_px / (2 * np.pi) 
        
        return arg2SLM.astype('uint8')
    
    def _create_filepath(self, **kwargs):
        """ creates a filepath to save data
        """

        date_str = time.strftime("%Y%m%d")
        date_time_str = time.strftime("%Y%m%d-%H:%M")
        
        new_path = os.path.join(self.save_path, date_str)
        
        # check if dir exists
        isExist = os.path.exists(new_path)
        # and create it
        if not isExist:
            os.makedirs(new_path)
        
        # create the filename based on kwargs    
        date_time_str = time.strftime("%Y%m%d-%H:%M")
        filename = str(date_time_str)
        for key, value in kwargs.items():
            filename += '_'
            filename += str(key) +  str(value)
        
        if self.save_path:
            self.filepath = os.path.join(new_path, filename)
        else:
            self.filepath = filename
            
        return self.filepath
    
    def save_raw(self):      

        with open(self.filepath + '.pkl', 'wb') as fp:
            pickle.dump((self.data_out), fp)
            
    def plot(self, frame, idx):
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        mask = axs[0, 0].imshow(self.final_pattern)
        axs[0, 0].set_title("Focus mask")
        axs[0, 0].set_xlabel("SLM x px #")
        axs[0, 0].set_ylabel("SLM y px #")
        divider = make_axes_locatable(axs[0, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(mask, cax=cax)

        focus_img = axs[0, 1].imshow(frame)
        axs[0, 1].set_title("Focusing")
        axs[0, 1].set_xlabel("Camera x px #")
        axs[0, 1].set_ylabel("Camera y px #")
        divider = make_axes_locatable(axs[0, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(focus_img, cax=cax)

        axs[1, 0].plot(self.cost)
        axs[1, 0].set_box_aspect(1)
        axs[1, 0].set_title("Cost function optimization")
        axs[1, 0].set_ylabel("Cost function")
        axs[1, 0].set_xlabel("Iterations #")

        axs[1, 1].plot(frame[idx])
        axs[1, 1].set_box_aspect(1)
        axs[1, 1].set_title("Focusing profile")

        axs[1, 1].set_xlabel("Camera x px #")
        axs[1, 1].set_ylabel("Intensity (a.u.)")

        fig.tight_layout()
        
        plt.savefig(self.filepath, dpi=400, transparent=True)
        
        
        
        """ The following classes can be condensed into only one in principle. To do. 
        """
class ContinuousSequential(IterationAlgos):
    
        
    def run(self):
        self.N = len(self.pattern_loader[0])

        gray = 0
        self.final_pattern = np.array([[gray for _ in range(self.N)] for _ in range(self.N)]).astype('uint8')
        
        self.cost = []
        counter = 0
        self.frames = {}

        for iteration in range(1, self.total_iterations+1):   
             
            # sweep each slm pixel
            with tqdm(self.pattern_loader) as pat_epoch:
                for _, mask, _ in pat_epoch:
                    temp_pattern = self.final_pattern.copy()
                    corr = []
                    # sweep phase at each pixel
                    for k in np.arange(0, self.m):
                        # create pattern, i.e one pattern for each phase value
                        temp = self.create_pattern(k, mask, temp_pattern)

                        # upload pattern to slm
                        self.upload_pattern(temp)

                        # get interferogram from camera
                        frame = self.get_frame()

                        # calculate correlation here
                        corr_k = self.callback(frame)
                        corr.append(corr_k)

                    counter += 1 
                    self.frames[counter] = frame

                    # update pattern with max corr
                    self.cost.append(np.max(corr))
                    self.final_pattern[mask] = self.phi_k(np.argmax(corr))
                                        
                    # print out status message
                    descr = [f"Iteration #: {iteration}",
                               f"Pattern #: {counter}"]
                    pat_epoch.set_description(' | '.join(descr))
                    pat_epoch.set_postfix(Cost=np.max(corr))
                    pat_epoch.refresh()


        return self.final_pattern, self.cost, self.frames
    

class StepwiseSequential(IterationAlgos):
    
    def run(self):
        gray = 0
        self.N = len(self.pattern_loader[0])

        self.final_pattern = np.array([[gray for _ in range(self.N)] for _ in range(self.N)]).astype('uint8')
        self.final_pattern_k = self.final_pattern.copy()
        
        self.cost = []
        counter = 0
        self.frames = {}

        for iteration in range(0, self.total_iterations):   
            temp_pattern = self.final_pattern.copy() 
            # sweep each slm pixel
            for _, mask, _ in tqdm((self.pattern_loader)):               
                corr = []
                # sweep phase at each pixel
                for k in np.arange(0, self.m):
                    # create pattern, i.e one pattern for each phase value
                    temp = self.create_pattern(k, mask, temp_pattern)
                    # upload pattern to slm
                    self.upload_pattern(temp, 0.1)
                    
                    # get interferogram from camera
                    frame = self.get_frame()

                    # calculate correlation here
                    corr_k = self.callback(frame)
                    corr.append(corr_k)
                
                counter += 1
                self.frames[counter] = frame
                temp_pattern = self.final_pattern.copy()
                
                # update pattern with max corr
                self.cost.append(np.max(corr))
                # pattern[idx[0], idx[1]] = self.phi_k(np.argmax(corr))
                self.final_pattern_k[mask] = self.phi_k(np.argmax(corr))
            self.final_pattern = self.final_pattern_k.copy()

        return self.final_pattern, self.cost, self.frames
    
class RandomPartition(IterationAlgos):
    
        
    def run(self):
        
        self.N = len(self.pattern_loader[0])

        gray = 0
        self.final_pattern = np.array([[gray for _ in range(self.N)] for _ in range(self.N)]).astype('uint8')
        
        self.cost = []
        counter = 0
        self.frames = {}

        for iteration in range(1, self.total_iterations+1):   
             
            # sweep each slm pixel
            with tqdm(self.pattern_loader) as pat_epoch:
                for mask, _ in pat_epoch:
                    temp_pattern = self.final_pattern.copy()
                    plt.figure()
                    plt.imshow(temp_pattern)
                    plt.colorbar()
                    corr = []
                    # sweep phase at each pixel
                    for k in np.arange(0, self.m):
                        # create pattern, i.e one pattern for each phase value
                        temp = self.create_pattern(k, mask, temp_pattern)

                        # upload pattern to slm
                        self.upload_pattern(temp)

                        # get interferogram from camera
                        frame = self.get_frame()

                        # calculate correlation here
                        corr_k = self.callback(frame)
                        corr.append(corr_k)

                    counter += 1 
                    self.frames[counter] = frame

                    # update pattern with max corr
                    self.cost.append(np.max(corr))
                    self.final_pattern[mask] = self.phi_k(np.argmax(corr))
                    print(self.phi_k(np.argmax(corr)))

                    # print out status message
                    descr = [f"Iteration #: {iteration}",
                               f"Pattern #: {counter}"]
                    pat_epoch.set_description(' | '.join(descr))
                    pat_epoch.set_postfix(Cost=np.max(corr))
                    pat_epoch.refresh()


        return self.final_pattern, self.cost, self.frames
    
class HadamardPartition(IterationAlgos):
    
    def run():
        pass
    

""" This class is implementing an idea found here https://www.wavefrontshaping.net/post/id/23
    that uses Zenike Polynomials to optimize optical aberrations on a focused laser beam.
    This implementation permits however the use of other basis as well, via a pattern loader.
"""
class CoefficientsOptimization(IterationAlgos):
        
    def __init__(self, 
                slm, camera,
                slm_resolution=(600, 800),
                calib_px=112, 
                num_of_coeffs=15,
                radius=300, 
                center=[600 // 2, 800 //2], 
                remote=True, 
                save_path=None):
        
        type = 'CoeffOptim'
        super().__init__(slm, camera, pattern_loader=None, 
                         slm_macropixel=None, slm_resolution=slm_resolution,  calib_px=calib_px, 
                         total_iterations=None, phase_steps=None, remote=remote, type=type, save_path=save_path)

        
        self.slm = slm
        self.camera = camera
        
        # SLM
        resX, resY = slm_resolution
        self.patternSLM = pt.PatternsBacic(resX, resY)
        self.calib_px = calib_px
        self.shape = slm_resolution
        
        # Zernike
        self.num_of_coeffs = num_of_coeffs
        self.radius = radius
        self.center = center
        
        # is the slm remotely connected to a rasp pi ?
        self.remote = remote
        
        # save raw data path
        # self.type
        # self.save_path = save_path
        # self.filepath = self._create_filepath()
        self.filepath = self._create_filepath(type='CoeffsOptim', coeffs_num=str(num_of_coeffs))

    
    def register_cost_callback(self, callback):
        """ This callback function is used to pass a custom cost function
            to the optimization object

        Parameters
        ----------
        callback
            the cost function
        """
        self.cost_callback = callback
        
    def register_pattern_callback(self, callback):
        self.pattern_callback = callback
    
    def _get_disk_mask(self, center = None):
        '''
        Taken from S. Popoff blog
        Generate a binary mask with value 1 inside a disk, 0 elsewhere
        :param shape: list of integer, shape of the returned array
        :radius: integer, radius of the disk
        :center: list of integers, position of the center
        :return: numpy array, the resulting binary mask
        '''
        shape = [2 * self.radius] * 2
        if not center:
            center = (shape[0] // 2, shape[1] // 2)
        X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        mask = (Y - center[0]) ** 2 + (X - center[1]) ** 2 < self.radius ** 2
        
        return mask.astype('bool')
        
    def _complex_mask_from_coeff(self, vec):
        '''
        Taken from S. Popoff blog
        Generate a complex phase mask from a vector containting the coefficient of the first Zernike polynoms.
        :param DMD_resolution: list of integers, contains the resolution of the DMD, e.g. [1920,1200]
        :param: integer, radius of the illumination disk on the DMD
        :center: list of integers, contains the position of the center of the illumination disk
        :center: list of float, the coefficient of the first Zernike polynoms
        '''
        # Generate a complex phase mask from the coefficients
        # zern_mask = np.exp(1j * phaseFromZernikes(vec, 2 * self.radius))
        coeff_mask = self.pattern_callback(vec)
        # We want the amplitude to be 0 outside the disk, we fist generate a binary disk mask
        amp_mask = self._get_disk_mask()
        
        # put the Zernike mask at the right position and multiply by the disk mask
        mask = np.zeros(shape = self.shape, dtype='complex')
        mask[self.center[0] - self.radius:self.center[0] + self.radius,
             self.center[1] - self.radius:self.center[1] + self.radius] = coeff_mask * amp_mask
        
        return mask
    
    def run(self, coeff_range=(-2, 2, 0.5)):
        
        counter = 0
        self.data_out = {}
        frames = {}
        masks = {}
        cost = []
        
        # initialize the coefficients to optimize
        coeffs = np.zeros(self.num_of_coeffs)
        # 
        coeff_idx = np.arange(coeff_range[0], coeff_range[1], coeff_range[2])
        
        iterator = trange(self.num_of_coeffs)
        for idx in iterator:
            cost_temp = []
            for coeff in coeff_idx:
                coeffs[idx] = coeff
                zmask = self._complex_mask_from_coeff(coeffs)
                zmask = self._phase2SLM(zmask)
                self._upload_pattern(zmask, 0.1)
                
                # get interferogram from camera
                frame = self._get_frame()

                # calculate cost function and save it
                cost_k = self.cost_callback(frame)
                cost_temp.append(cost_k)

            # update pattern with max corr
            cost.append(np.max(cost_temp))            
            coeffs[idx] = coeff_idx[np.argmax(cost_temp)]
            
            # just reload the optimal mask for this iteration and save 
            # the corresponding frame
            counter += 1 
            zmask = self._complex_mask_from_coeff(coeffs)
            zmask = self._phase2SLM(zmask)
            self._upload_pattern(zmask, 0.1)
            frame = self._get_frame()
            frames[counter] = frame
            masks[counter] = zmask
            
            # print out status message
            descr = [f"Iteration #: {idx}"]
            iterator.set_description(' | '.join(descr))
            iterator.set_postfix(Cost=cost[idx])
            iterator.refresh()

            # save all into a big dict

            self.data_out["coeffs"] = coeffs
            self.data_out["cost"] = cost
            self.data_out["frames"] = frames
            self.data_out["masks"] = masks

        return self.data_out