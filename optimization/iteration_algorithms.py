import numpy as np
import time, os
import pickle
from tqdm.auto import tqdm
from slmOptim.patternSLM import patterns as pt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from aotools.functions import phaseFromZernikes



class IterationAlgos():
    """ This is the base class to be used for various iteration based algorithms for optimization
        including Stepwise Sequential (SSA), Continuous Sequential (CSA), Random Partition (RPA) 
        and Hadamard Partition (HPA) algorithms. 
    """
    
    def __init__(self, 
                 slm, 
                 camera,
                 pattern_loader,
                 total_iterations=1,
                 slm_resolution=(800, 600),
                 slm_segments=256,
                 slm_macropixel=5, 
                 slm_calibration_pixel=112,
                 phase_steps=8,
                 remote=True,
                 save_path=None):
        
        self.slm = slm
        self.camera = camera
        
        self.total_iterations = total_iterations

        # slm settings
        self.N = int(slm_segments ** 0.5)
        self.slm_macropixel = slm_macropixel
        self.calib_px = slm_calibration_pixel
        
        self.pattern_loader = pattern_loader
                
        self.m = phase_steps
        
        # SLM
        resX, resY = slm_resolution
        self.patternSLM = pt.Pattern(resX, resY)
        
        # is the slm remotely connected to a rasp pi ?
        self.remote = remote

        # the type of the iteration algorithm        
        self.type = type
        
        # save raw data path
        self.save_path = save_path
        self.filepath = self._create_filepath()
     
    def register_callback(self, callback):
        """ This callback function is used to pass a custom cost function
            to the optimization object

        Parameters
        ----------
        callback
            the cost function
        """
        self.callback = callback
    
    def phi_k(self, k):
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
    
    def create_pattern(self, k, mask, pattern):
        """ Prepares a pattern ready for the SLM: enlarges it according to the 
            macropixel size and adds it in the middle of the SLM screen
        """
        phi = self.phi_k(k)
        pattern[mask] = phi
        # temp = self.patternSLM.pattern2SLM(pattern, self.slm_macropixel)
        temp = self.patternSLM._enlarge_pattern(pattern, self.slm_macropixel)
        temp = self.patternSLM.add_subpattern(temp)
        return temp

    def upload_pattern(self, pattern, slm_delay=0.1):
        """ Uploads a pattern to the SLM either in remote or local mode. Adds a user-defined
            time delay to make sure that the pattern is uploaded. 
        """
        if self.remote:
            self.slm.sendArray(pattern)
        else:
            self.slm.updateArray(pattern)
        time.sleep(slm_delay)
        
    
    def get_frame(self):
        """ Get frame from zelux thorlabs camera
        """
        frame = self.camera.get_pending_frame_or_null()
        image_buffer = np.copy(frame.image_buffer)
        return image_buffer
    
    def _create_filepath(self):
        """ Creates a directory and a filename to save raw data
            and figures
        """

        timestr = time.strftime("%Y%m%d")
        new_path = os.path.join(self.save_path, timestr)
        
        # check if dir exists
        isExist = os.path.exists(new_path)
        # and create it
        if not isExist:
            os.makedirs(new_path)
        
        # define a filename
        filename = '{}_optim_raw_data_{}_slm_segs{}_slm_macro{}.'.format(timestr,
                                                                            self.type,
                                                                            self.N ** 2, 
                                                                            self.slm_macropixel)
        if self.save_path:
            self.filepath = os.path.join(new_path, filename)
        else:
            self.filepath = self.filename
            
        return self.filepath
    
    
    def save_raw(self):      

        with open(self.filepath + 'pkl', 'wb') as fp:
            pickle.dump((self.frames, self.cost, self.final_pattern), fp)
            
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
        
        plt.savefig(self.filepath + "png", dpi=400, transparent=True)
        
        
        
        """ The following classes can be condensed into only one in principle. To do. 
        """
class ContinuousSequential(IterationAlgos):
    
        
    def run(self):
        
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
    
    def run():
        pass
    
class HadamardPartition(IterationAlgos):
    
    def run():
        pass
    
    
class ZernikesPolynomials(IterationAlgos):
    
    def __init__(self, 
                slm, 
                camera,
                slm_resolution=(800, 600),
                slm_calibration_pixel=112,
                num_of_zernike_coeffs=8,
                radius=200, 
                center = [600 // 2, 800 // 2],
                remote=True,
                save_path=None):
        
        self.slm = slm
        self.camera = camera
        
        # SLM
        resX, resY = slm_resolution
        self.patternSLM = pt.Pattern(resX, resY)
        self.calib_px = slm_calibration_pixel

        # Zernike
        self.num_of_zernike_coeffs = num_of_zernike_coeffs
        self.radius = radius
        self.center = center
        
        # is the slm remotely connected to a rasp pi ?
        self.remote = remote
        
        # save raw data path
        # self.save_path = save_path
        # self.filepath = self._create_filepath()
    
    
    
    def _get_disk_mask(self, center = None):
        '''
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
        return mask.astype('int')
    
    def _complex_mask_from_zernike_coeff(self, vec):
        '''
        Generate a complex phase mask from a vector containting the coefficient of the first Zernike polynoms.
        :param DMD_resolution: list of integers, contains the resolution of the DMD, e.g. [1920,1200]
        :param: integer, radius of the illumination disk on the DMD
        :center: list of integers, contains the position of the center of the illumination disk
        :center: list of float, the coefficient of the first Zernike polynoms
        '''
        # Generate a complex phase mask from the coefficients
        zern_mask = np.exp(1j * phaseFromZernikes(vec, 2 * self.radius))
        
        # We want the amplitude to be 0 outside the disk, we fist generate a binary disk mask
        amp_mask = self._get_disk_mask()
        
        # put the Zernik mask at the right position and multiply by the disk mask
        mask = np.zeros(shape = self.shape, dtype='complex')
        mask[self.center[0] - self.radius:self.center[0] + self.radius,
            self.center[1] - self.radius:self.center[1] + self.radius] = zern_mask * amp_mask
        
        return mask
    
    def run(self):
        
        counter = 0
        frames = {}
        cost = []
        
        coeffs = np.zeros(self.num_of_zernike_coeffs)
        phi_k = np.arange(-2, 2, 0.5)
        
        for idx in range(self.num_of_zernike_coeffs):
            cost_temp = []
            for phi in phi_k:
                coeffs[idx] = phi
                zmask = self._complex_mask_from_zernike_coeff(coeffs)
                self.upload_pattern(zmask, 0.1)
                
                    # get interferogram from camera
                frame = self.get_frame()

                # calculate correlation here
                cost_k = self.callback(frame)
                cost_temp.append(cost_k)
                
            counter += 1 
            frames[counter] = frame

            # update pattern with max corr
            self.cost.append(np.max(cost_temp))
            coeffs[idx] = phi

        return zmask, coeffs, cost
                
                    
                    

