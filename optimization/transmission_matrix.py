from ..patternSLM import patterns as pt
from ..patternSLM import upload as up
from ..zeluxPy import helper_functions as cam
from slmPy import slmpy
from scipy.linalg import hadamard
import threading
from tqdm.auto import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import os

"""
"""


class measTM:

    def __init__(self, 
                 slm, camera, 
                 pattern_loader,
                 num_in, 
                 slm_macropixel_size,
                 calib_px=112,
                 remote=True, 
                 corr_path=None,
                 save_path=None):
        """

        Parameters
        ----------
        roi
        bins
        exposure_time
        gain
        timeout
        order
        mag
        monitor
        """
        
        # camera
        self.camera = camera
        
        # slm settings
        self.slm = slm
        self.num_in = num_in
        self.slm_macropixel_size = slm_macropixel_size
        self.calib_px = calib_px
        self.remote = remote # is the slm connected to a raspberry Pi ?
        
        # load basis patterns
        self.pattern_loader = pattern_loader
        
        # correction pattern
        self.corr_path = corr_path
        
        # save raw data path
        self.save_path = save_path
        
    def get(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        
        # Create flag events
        download_frame_event = threading.Event()
        upload_pattern_event = threading.Event()
        stop_all_event = threading.Event()

        # Create the threads
        upload_thread = up.SlmUploadPatternsThread(self.slm,
                                            download_frame_event,
                                            upload_pattern_event,
                                            stop_all_event,
                                            calib_px = self.calib_px,
                                            num_in=self.num_in,
                                            slm_macropixel_size=self.slm_macropixel_size,
                                            path=self.corr_path)

        download_thread = cam.FrameAcquisitionThread(self.camera,
                                                     download_frame_event, 
                                                     upload_pattern_event, 
                                                     stop_all_event,
                                                     num_of_frames=1)

        # Start the threads
        upload_thread.start()
        download_thread.start()

        # Wait for the threads to finish
        download_thread.join()
        upload_thread.join()

        # The main thread will wait for both threads to finish before continuing
        
        # Finally, kill all
        # self.slm.close()
        # self.init_camera.destroy()
        # print("Program execution completed - camera and slm killed! ")
        
        # get and return data 
        self.patterns = upload_thread.patterns
        self.frames = download_thread.frames, 
        
        return self.patterns, self.frames
    
    def get2(self, slm_delay=0.1):
        """_summary_

        Parameters
        ----------
        slm_delay : float, optional
            _description_, by default 0.1

        Returns
        -------
        _type_
            _description_
        """
        # basis = pt.Pattern._get_hadamard_basis(self.num_in)
        pi = int(self.calib_px / 2)
        four_phases = [0, pi / 2, pi, 3 * pi / 2]
        
        # four_intensities = {}
        # self.patterns = []
        self.frames = []
        
        # resX, resY = self.slm.getSize()
        # resX, resY = (800, 600)
        # slm_patterns = pt.Pattern(resX, resY)


        # loop through each 2d vector of the loaded basis
        # for vector in tqdm(basis, desc='Uploading Hadamard patterns', leave=True):
        for vector in tqdm(self.pattern_loader, desc="Uploading pattern vectors", leave=True):
            # and for each vector load the four reference phases
            for phase in four_phases:
                # _, pattern = slm_patterns.hadamard_pattern_bis(vector, n=self.slm_macropixel_size, gray=phase)
                if self.remote:
                    self.slm.sendArray(vector)
                else:
                    self.slm.updateArray(vector) # load each vector to slm
                time.sleep(slm_delay)
                # get frame for each phase
                frame = self.camera.get_pending_frame_or_null()
                image_buffer_copy = np.copy(frame.image_buffer)

                # four_intensities[idx, phase] = image_buffer_copy
                # self.patterns.append(vector)
                self.frames.append(image_buffer_copy)
                

        print("ΤΜ acquisition completed ! ")
            
        # return self.patterns, self.frames
        return self.frames
        
    def save(self):

        timestr = time.strftime("%Y%m%d")
        new_path = os.path.join(self.save_path, timestr)
        
        # check if dir exists
        isExist = os.path.exists(new_path)
        # and create it
        if not isExist:
            os.makedirs(new_path)
        
        filename = '{}_tm_raw_data_num_in{}_slm_macro{}.pkl'.format(timestr,  
                                                                    self.num_in, 
                                                                    self.slm_macropixel_size)
        
        if self.save_path:
            filepath = os.path.join(new_path, filename)
        else:
            filepath = filename
        
        print(filepath)

        with open(filepath, 'wb') as fp:
            pickle.dump((self.frames), fp)
        
        

class calcTM:

    def __init__(self, data):
        self.data = data
        
    
    @staticmethod
    def four_phases_method(intensities):
        I1 = float(intensities[0])
        I2 = float(intensities[1])
        I3 = float(intensities[2])
        I4 = float(intensities[3])
        complex_field = complex((I1 - I4) / 4, (I3 - I2) / 4)
        return complex_field
    
    def _calc_dim(self):
        shape = np.array(self.data).shape
        
        total_num = shape[0]
        frame_shape = (shape[1], shape[2])
        slm_px_len = int(shape[0] / 4)
        cam_px_len = shape[1] * shape[2]
        
        return total_num, frame_shape, slm_px_len, cam_px_len

    def _calc_obs(self):

        # get dimensions and length that will be useful for the for loops
        total_num, frame_shape, slm_px_len, cam_px_len = self._calc_dim()
        
        # organize iterator over all frames into group of 4. Each group corresponds to 
        # the one 4-phase measurement, i.e. one slm vector with 4 grayscale levels
        iterator = np.arange(0, total_num)
        iterator = [iterator[n:n+4] for n in range(0, len(iterator), 4)]

        # initialize a 2d complex matrix: the observed transmission matrix
        tm_obs = np.full(shape=(cam_px_len, slm_px_len), fill_value=0).astype('complex128')
        
        # loop through every pixel of a camera frame (2d array)
        cam_px_idx = 0
        for iy, ix in np.ndindex(frame_shape):
            slm_px_idx = 0
            # loop through every 4-phase measurement - equivalently every slm pixel
            for subiterator in iterator:
                four_intensities_temp = []
                # create a temp list with four intensities
                for subsub in subiterator:
                    four_intensities_temp.append(self.data[subsub][iy, ix])
                # calculate the complex field value
                four_phases = self.four_phases_method(four_intensities_temp)
                # save it into the transmission matrix
                tm_obs[cam_px_idx, slm_px_idx] = four_phases
                # increment pixel indices
                slm_px_idx += 1
            cam_px_idx += 1
        
        return tm_obs


    def _normalization_factor(self):

        # get dimensions and length that will be useful for the for loops
        total_num, frame_shape, slm_px_len, cam_px_len = self._calc_dim()
        # organize iterator over all frames into group of 4. Each group corresponds to 
        # the one 4-phase measurement, i.e. one slm vector with 4 grayscale levels
        iterator = np.arange(0, total_num)
        iterator = [iterator[n:n+4] for n in range(0, len(iterator), 4)]

        # initialize a 2d matrix: 
        norm_ij = np.full(shape=(cam_px_len, slm_px_len), fill_value=0).astype('float64')
        
        # loop through every pixel of a camera frame (2d array)
        cam_px_idx = 0
        for iy, ix in np.ndindex(frame_shape):
            slm_px_idx = 0
            cam_px = []
            # loop through every 4-phase measurement - equivalently every slm pixel
            for subiterator in iterator:
                cam_amp_temp = []
                # create a temp list with four intensities
                for subsub in subiterator:
                    cam_amp_temp.append(self.data[subsub][iy, ix])
                # calculate the complex field value
                cam_px.append(cam_amp_temp[3])
                # increment pixel indices
                slm_px_idx += 1

            std = np.array(cam_px).std()
            norm_ij[cam_px_idx, :] = std
            cam_px_idx += 1

        return norm_ij
    
    def _had2canonical(self, matrix):
        """ Perform a basis change: from the hadamard to the canonical one
            by calculating the dot product between the measured TM and the hadamard
            matrix on which the basis is created
        """
        _, _, slm_px_len, _ = self._calc_dim()
        h = hadamard(slm_px_len)
        tm_can = np.dot(matrix, h)
        return tm_can

    def calc_plot_tm(self):
        
        tm_obs = self._calc_obs()
        norm = self._normalization_factor()
        tm_fil = tm_obs / norm
        tm = self._had2canonical(tm_fil)
        
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(7, 7))
        axs[0, 0].imshow(abs(tm_obs), aspect='auto')
        axs[1, 0].imshow(norm, aspect='auto')
        axs[0, 1].imshow(abs(tm_fil), aspect='auto')
        axs[1, 1].imshow(abs(tm), aspect='auto')

        axs[0, 0].set_title("Hadamard TM")
        axs[1, 0].set_title("Normalization")
        axs[0, 1].set_title("Filtered TM")
        axs[1, 1].set_title("Canonical TM")

        fig.text(0.5, -0.01, 'slm pixels #', ha='center')
        fig.text(-0.01, 0.5, 'camera pixels #', va='center', rotation='vertical')
        fig.tight_layout()
        
        return tm_obs, norm, tm_fil, tm   
             