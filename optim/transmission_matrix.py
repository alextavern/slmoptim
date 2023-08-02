from ..patternSLM import patterns as pt
from ..zeluxPy import helper_functions as cam
from slmPy import slmpy
from scipy.linalg import hadamard
import threading
import numpy as np
import pickle
import matplotlib.pyplot as plt
"""
"""


class meas_TM:

    def __init__(self, roi=(556, 476, 684 - 1, 604 - 1), bins=8, exposure_time=100, gain=1, timeout=100,
                 order=4, mag=5,
                 monitor=1):
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
        # camera settings
        self.roi = roi
        self.bins = bins
        self.exposure_time = exposure_time
        self.gain = gain
        self.timeout = timeout
        # hadamard settings
        self.order = order
        self.mag = mag
        # slm monitor setting
        self.slm = slmpy.SLMdisplay(monitor=monitor)
        
    def get_tm(self):
        """

        Returns
        -------

        """
        # Create flag events
        download_frame_event = threading.Event()
        upload_pattern_event = threading.Event()
        stop_all_event = threading.Event()

        # Create the threads
        upload_thread = pt.SlmUploadPatternsThread(self.slm,
                                            download_frame_event,
                                            upload_pattern_event,
                                            stop_all_event,
                                            order=self.order,
                                            mag=self.mag)
        
        download_thread = cam.CameraThread(download_frame_event,
                                           upload_pattern_event,
                                           stop_all_event,
                                           roi=self.roi,
                                           bins=(self.bins, self.bins),
                                           exposure_time=self.exposure_time,
                                           gain=self.gain,
                                           timeout=self.timeout)

        # Start the threads
        upload_thread.start()
        download_thread.start()

        # Wait for the threads to finish
        download_thread.join()
        upload_thread.join()

        # The main thread will wait for both threads to finish before continuing
        # Finally, close slm
        self.slm.close()
        print("Program execution completed.")
        
        return upload_thread.patterns, download_thread.frames
    
    def save_tm(self):
        filename = 'tm_raw_data_roi:{}_bins:{}_order:{}_mag:{}.pkl'.format([self.roi, self.bins, self.order, self.mag])
        with open(filename, 'wb') as fp:
            pickle.dump(self.download_thread.frames, fp)
        return
        

class calc_TM:

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
    
    def _calc_tm_dim(self):
        shape = np.array(self.data).shape
        
        total_num = shape[0]
        frame_shape = (shape[1], shape[2])
        slm_px_len = int(shape[0] / 4)
        cam_px_len = shape[1] * shape[2]
        
        return total_num, frame_shape, slm_px_len, cam_px_len

    def _calc_tm_obs(self):

        # get dimensions and length that will be useful for the for loops
        total_num, frame_shape, slm_px_len, cam_px_len = self._calc_tm_dim()
        
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
            cam_px = []
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
        total_num, frame_shape, slm_px_len, cam_px_len = self._calc_tm_dim()
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
    
    @staticmethod
    def _change2canonical(matrix, order=8, mag=3):
        h = hadamard(2 ** order)
        # h = pt.Pattern._enlarge_pattern(h, mag)
        tm = np.dot(matrix, h)
        return tm

    def calc_plot_tm(self):
        
        tm_obs = abs(self._calc_tm_obs())
        norm = self._normalization_factor()
        tm_fil = abs(tm_obs / norm)
        tm = self._change2canonical(tm_fil)
        
        
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(7, 7))
        axs[0, 0].imshow(abs(tm_obs), aspect='auto')
        axs[1, 0].imshow(norm, aspect='auto')
        axs[0, 1].imshow(tm_fil, aspect='auto')
        axs[1, 1].imshow(tm, aspect='auto')

        axs[0, 0].set_title("Hadamard TM")
        axs[1, 0].set_title("Normalization")
        axs[0, 1].set_title("Filtered TM")
        axs[1, 1].set_title("Canonical TM")

        fig.text(0.5, -0.01, 'camera pixels #', ha='center')
        fig.text(-0.01, 0.5, 'slm pixels #', va='center', rotation='vertical')
        fig.tight_layout()
        
        return tm_obs, norm, tm_fil, tm
        
class InverseLight:
    
    def __init__(self, target) -> None:
        pass
    
    
    def _conj_trans(matrix):
        return  matrix.transpose().conjugate()
    
    def _inverse_prop(target):
        # first flatten
        target_frame_flattened = []
        for iy, ix in np.ndindex(frame_shape):
            target_frame_flattened.append(target_frame[iy, ix])

        target_frame_flattened = np.array(target_frame_flattened)

        # apply inversion
        inverse = np.dot(tm_T_star, target_frame_flattened.T)

        # get phase
        arg = np.angle(inverse, deg=False)
        arg2pi = (arg + 2 * np.pi) % (2 * np.pi)
        arg2SLM = arg2pi * 112 / (2 * np.pi) 

        # and unflatten again
        phase_mask = np.full(shape=(16, 16), fill_value=0).astype('float64')
        for idx, ij in enumerate(np.ndindex(16, 16)):
            phase_mask[ij[0], ij[1]] = arg2SLM[idx]
            
        # enlarge for the SLM macropixels
        phase_mask_enlarged = _enlarge_pattern(phase_mask, 4)