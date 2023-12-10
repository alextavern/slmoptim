from ..loader import patterns as pt
from ..utils import upload as up
from ..utils import download as down
from mpl_toolkits.axes_grid1 import make_axes_locatable
from slmPy import slmpy
from scipy.linalg import hadamard
import threading
from tqdm.auto import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import os
import warnings

"""
"""


class measTM:

    def __init__(self, 
                 slm, camera, 
                 pattern_loader, 
                 slm_macropixel_size,
                 slm_resolution=(800, 600),
                 calib_px=112,
                 remote=True, 
                 corr_path=None,
                 save_path=None):
        """

        Parameters
        ----------
        slm
        camera
        pattern loader
        slm macropixel
        slm resolution
        slm calibration pixel
        remote operation
        correction pattern path
        save data path
        """
        
        # camera
        self.camera = camera
        
        # slm settings
        self.slm = slm
        self.num_in = len(pattern_loader)
        self.slm_macropixel_size = slm_macropixel_size
        self.calib_px = calib_px
        self.resX, self.resY = slm_resolution
        self.remote = remote # is the slm connected to a raspberry Pi ?
        
        # load basis patterns
        self.pattern_loader = pattern_loader
        
        # correction pattern
        self.corr_path = corr_path
        
        # save raw data path
        self.save_path = save_path
        self.filepath = self._create_filepath()
        
    def get(self):
        """ Opens two parallel threads one to upload patterns to the SLM and one to download frames
            from the camera
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

        download_thread = down.FrameAcquisitionThread(self.camera,
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
        
        # get and return data 
        self.patterns = upload_thread.patterns
        self.frames = download_thread.frames, 
        
        return self.patterns, self.frames
    
    def get2(self, slm_delay=0.1):
        """ A simpler implemetantion of the TM acquisition where only a time delay is introduced between 
            uploading and downloading in order to make sure that the pattern is uploaded before acquiring
            a frame

        Parameters
        ----------
        slm_delay : float, optional
            time delay between uploading and downloading, by default 0.1

        Returns
        -------
        frames : list
            a list of interferogram frames            
        """
        
        pi = int(self.calib_px / 2)
        four_phases = [0, pi / 2, pi, 3 * pi / 2]
        
        self.frames = []
        
        resX, resY = (800, 600)
        slm_patterns = pt.PatternsBacic(resX, resY)


        # loop through each 2d vector of the loaded basis
        for vector in tqdm(self.pattern_loader, desc="Uploading pattern vectors", leave=True):
            # and for each vector load the four reference phases
            for phase in four_phases:
                pattern = slm_patterns.pattern_to_SLM(vector, self.slm_macropixel_size, phase)
                if self.remote:
                    self.slm.sendArray(pattern)
                else:
                    self.slm.updateArray(pattern) # load each vector to slm
                time.sleep(slm_delay) # wait to make sure the vector is loaded
                
                # get frame for each phase
                frame = self.camera.get_pending_frame_or_null()
                image_buffer_copy = np.copy(frame.image_buffer)
                
                # check saturation
                max_level = np.amax(image_buffer_copy)
                if max_level > 1000:
                    warnings.warn("Pixel saturation: {}.".format(max_level), UserWarning)

                self.frames.append(image_buffer_copy)
                

        print("ΤΜ acquisition completed ! ")
            
        return self.frames
    
    def _create_filepath(self):
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
        
        filename = '{}_tm_raw_data_num_in{}_slm_macro{}'.format(date_time_str,  
                                                                self.num_in, 
                                                                self.slm_macropixel_size)
        
        if self.save_path:
            self.filepath = os.path.join(new_path, filename)
        else:
            self.filepath = filename
            
        return self.filepath
        
    def save(self):
        """ saves raw data to a pickle format
        """

        with open(self.filepath + '.pkl', 'wb') as fp:
            pickle.dump((self.frames), fp)
        
class calcTM(measTM):

    def __init__(self, data, loader=None):
        self.data = data
        self.loader = loader
    
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
        self.tm_obs = np.full(shape=(cam_px_len, slm_px_len), fill_value=0).astype('complex128')
        
        # loop through every pixel of a camera frame (2d array)
        cam_px_idx = 0
        for iy, ix in tqdm(np.ndindex(frame_shape), desc='Iterating through camera pixels', leave=True):
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
                self.tm_obs[cam_px_idx, slm_px_idx] = four_phases
                # increment pixel indices
                slm_px_idx += 1
            cam_px_idx += 1
        
        return self.tm_obs
    
    def _normalization_factors(self):
        """ Calculates the standard deviation of each row of the observed 
            hadamard-based transmission matrix and returns a diagonal matrix
            whose elements give the std

        Returns
        -------
            norm_factors: diagonal 2d array - each element is the std of the TM
        """
        tm = abs(self.tm_obs)
        
        # Get means for each row
        means = [row.mean() for row in tm]
        # Calculate squared errors
        squared_errors = [(row - mean) ** 2 for row, mean in zip(tm, means)]
        # Calculate the inverse square root of mean for each row of squared errors 
        # this is the inverse standard deviation of each row
        norm_factors = np.diag([(1 / row.mean()) ** 0.5 for row in squared_errors])
        
        return norm_factors
        
    def _normalize(self):
        """ Normalizes the observed TM by computing its dot product with the normalization
            factors diagonal matrix
        """
        norm_factors = self._normalization_factors()
        tm_fil = norm_factors@self.tm_obs
        # tm_fil = np.dot(norm_factors, self.tm_obs)
        return tm_fil
    
    def _had2canonical(self):
        """ Perform a basis change: from the hadamard to the canonical one
            by calculating the dot product between the measured TM and the hadamard
            matrix on which the basis is created
        """
        _, _, slm_px_len, _ = self._calc_dim()
        h = hadamard(slm_px_len)
        # tm_can = np.dot(matrix, h)
        return h

    def _change_to_canonical_basis(self, matrix, loader):
        
        if loader:
            passage = []
            # here we iterate through every vector in the loader, 
            # each vector is flattened and added as a row in the 
            # passage matrix
            for vector in loader:
                passage.append(vector.flatten())
            passage = np.array(passage)
        else:
            passage = self._had2canonical()
            
        # perform the basis change
        tm_can = np.dot(matrix, passage)
        
        return tm_can
    
    def calc(self):
    
        self.tm_obs = self._calc_obs()
        self.tm_fil = self._normalize()
        # tm = self._had2canonical(tm_fil)
        self.tm = self._change_to_canonical_basis(self.tm_fil, self.loader)
        
        return self.tm_obs, self.tm_fil, self.tm   
    
    def plot(self, figsize=(10, 5)):
        
        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=figsize)
            
        obs = axs[0].imshow(abs(self.tm_obs), aspect='auto')
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(obs, cax=cax)
        
        
        fil = axs[1].imshow(abs(self.tm_fil), aspect='auto')
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(fil, cax=cax)
        
        can = axs[2].imshow(abs(tm), aspect='auto')
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(can, cax=cax)

        axs[0].set_title("Hadamard TM")
        axs[1].set_title("Filtered TM")
        axs[2].set_title("Canonical TM")

        fig.text(0.5, -0.01, 'slm pixels #', ha='center')
        fig.text(-0.01, 0.5, 'camera pixels #', va='center', rotation='vertical')
        fig.tight_layout()
        
        return fig
    
    def calc_plot_tm(self, figsize=(10, 5)):
    
        tm_obs = self._calc_obs()
        tm_fil = self._normalize()
        # tm = self._had2canonical(tm_fil)
        tm = self._change_to_canonical_basis(tm_fil, self.loader)
        
        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=figsize)
        
        obs = axs[0].imshow(abs(tm_obs), aspect='auto')
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(obs, cax=cax)
        
        
        fil = axs[1].imshow(abs(tm_fil), aspect='auto')
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(fil, cax=cax)
        
        can = axs[2].imshow(abs(tm), aspect='auto')
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(can, cax=cax)

        axs[0].set_title("Hadamard TM")
        axs[1].set_title("Filtered TM")
        axs[2].set_title("Canonical TM")

        fig.text(0.5, -0.01, 'slm pixels #', ha='center')
        fig.text(-0.01, 0.5, 'camera pixels #', va='center', rotation='vertical')
        fig.tight_layout()
        
        return tm_obs, tm_fil, tm   
             