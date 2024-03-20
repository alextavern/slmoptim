from collections.abc import Callable, Iterable, Mapping
from typing import Any
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
import threading
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .redpitaya_scpi import scpi
from slmPy import slmpy
import sys 
sys.path.append("/usr/lib/python3/dist-packages/")
from metavision_core.event_io.raw_reader import initiate_device, RawReader
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from metavision_hal import I_ROI, I_LL_Biases
import time

class PropheseeCamera():
    
    def __init__(self, **kwargs):
        print("Prophesee camera initialized - use init_cam() to arm the camera and get() to get frames")
        # camera settings
        self.roi_size_x = kwargs.get('roi_size_x', 0)
        self.roi_size_y = kwargs.get('roi_size_y', 0)
        self.roi_off = kwargs.get('roi_off', (0, 0))
        self.accumulation_time = kwargs.get('accumulation_time', 1000)

    def init_cam(self):
        """ Initializes and sets camera parameters
        """
        # camera instance
        self.camera = initiate_device("")
        self.init_time = time.time() # ref time to know the time stamp of current events which are generated with respect to the time we initiate the camera
        self.height = self.camera.get_i_geometry().get_height() # camera height
        self.width = self.camera.get_i_geometry().get_width() # camera width

        # configure
        self.cam_stream = RawReader.from_device(device=self.camera)
        self.frameGen = OnDemandFrameGenerationAlgorithm(width = self.width, height = self.height)
        self.frameGen.set_accumulation_time_us(self.accumulation_time)
        if self.roi_size_x != 0:
            self.roi_size_off_x, self.roi_size_off_y = self.roi_off
            self.camera.get_i_roi().set_window(self.camera.get_i_roi().Window(self.roi_size_off_x, self.roi_size_off_y, self.roi_size, self.roi_size)) 
            self.camera.get_i_roi().enable(True)
        
        return self.camera
    
    def close_cam(self):
        del self.camera

    def get(self):
        """ Get frame from Prophesee camera
        """
        if not self.cam_stream.is_done():
            # check that enough time has past since last generation
            if self.cam_stream.current_time + self.accumulation_time > (time.time()-self.init_time)*1e6:
                time.sleep(self.accumulation_time*1e-6)
            
            # move the cursor to the current time 
            self.cam_stream.seek_time((time.time()-self.init_time)*1e6-self.accumulation_time)
            
            # load and process the events of the last accumulation_time period
            events = self.cam_stream.load_delta_t(self.accumulation_time)
            # while events.size == 0:
            #    events = self.cam_stream.load_delta_t(self.accumulation_time)
            self.frameGen.process_events(events)
            frame = np.zeros((self.height, self.width, 3), np.uint8)
            self.frameGen.generate(events['t'][-1], frame)
            img = np.mean(frame, axis = 2)
        
        return img

class ZeluxCamera():
    
    def __init__(self, **kwargs):
        print("Zelux camera initiliazed - use init_cam() to arm and trigger it and get() to get frames")
        # camera settings
        self.roi_size = kwargs.get('roi_size', 100)
        self.roi_off = kwargs.get('roi_off', (0, 0))
        self.bins = kwargs.get('bins', 1)
        self.exposure_time = kwargs.get('exposure_time', 100)
        self.gain = kwargs.get('gain', 1)
        self.timeout = kwargs.get('timeout', 100)
    
    def set_roi(self):
        """ Calculates the Region of interest. The user gives a window size and x, y offsets from
            the sensor center

        Returns
        -------
        roi (tuple or int)
        """
        if type(self.roi_size) is tuple:
            width, height = self.roi_size
        else:
            width = self.roi_size
            height = width
            
        offset_x, offset_y = self.roi_off
        middle_x = int(1440 / 2) + offset_x
        middle_y = int(1080 / 2) - offset_y

        roi = (middle_x - int(width/ 2), 
            middle_y - int(height / 2), 
            middle_x + int(width / 2), 
            middle_y + int(height / 2))
        
        return roi
    
    def init_cam(self):
        """ Initializes and sets camera parameters
        """
        # camera instance
        self.sdk = TLCameraSDK()
        available_cameras = self.sdk.discover_available_cameras()
        self.camera = self.sdk.open_camera(available_cameras[0])

        # configure
        self.camera.exposure_time_us = self.exposure_time  # set exposure to 11 ms
        self.camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
        self.camera.image_poll_timeout_ms = self.timeout  # 1 second polling timeout
        self.camera.roi = self.set_roi()

        # set binning for camera macropixels
        (self.camera.binx, self.camera.biny) = (self.bins, self.bins)

        if self.camera.gain_range.max > self.gain:
            db_gain = self.gain
            gain_index = self.camera.convert_decibels_to_gain(db_gain)
            self.camera.gain = gain_index

        # arm - trigger
        self.camera.arm(2)
        self.camera.issue_software_trigger()
        
        return self.camera
    
    def close_cam(self):
        self.camera.disarm()
        self.camera.dispose()
        self.sdk.dispose()  
        
    def get(self):
        """ Get frame from zelux thorlabs camera
        """
        frame = self.camera.get_pending_frame_or_null()
        image_buffer = np.copy(frame.image_buffer)
        
        return image_buffer


class RedPitaya:

    def __init__(self, **kwargs):
        # pass arguments
        self.IP = kwargs.get('IP', '172.24.40.69')
        self.num_of_samples = kwargs.get('number_of_samples', 16384)
        self.clock = kwargs.get('clock', 125 * 1e6)
        self.channel = kwargs.get('source', 2)
        
        # launch scpi server - make sure it is manually activated from the redpi interface
        self.rp = self._launch_scpi_server()
        
        # decimation factors of 1, 8, 64, 1024, 8192, 65536 are accepted with the
        # original redpitaya software, otherwise an error is raised
        self.decimation = kwargs.get('decimation', 8192)
        self.rp.tx_txt('ACQ:DEC ' + str(self.decimation))
        
        # make sure the decimation was set
        self.rp.tx_txt('ACQ:DEC?')
        # display('Decimation set to: ' + str(self.rp.rx_txt()))
        print('Red Pitaya daq loaded - decimation set to: ' + str(self.rp.rx_txt()))

        # prepare buffer dataframes
        self.buff_ffts = pd.DataFrame()
        self.buffs = pd.DataFrame()
        
        self.num_of_avg = kwargs.get('number_of_avg', 10)
        self.offset = kwargs.get('offset', 2) # the offset takes a certain number of spectra in the beginning. 
                             # Sometimes the Red Pitaya produces trash in the first spectra
                             
    def _launch_scpi_server(self):
        server = scpi(self.IP)
        return server

    def acquire(self, fourier=False):
        # do the acquisitions and save it in the computers memory (not on the Red Pitaya).
        for i in range(1, self.num_of_avg + self.offset):
            # if i % 50 == 0:
            #     print(i)
                # display(i)
            self.rp.tx_txt('ACQ:START')
            self.rp.tx_txt('ACQ:TRIG NOW')
            self.rp.tx_txt('ACQ:SOUR' + str(self.channel) + ':DATA?')
            buff_string = ''
            buff_string = self.rp.rx_txt()
            buff_string = buff_string.strip('{}\n\r').replace("  ", "").split(',')
            #display(buff_string)
            self.buffs[i] = list(map(float, buff_string))
            if fourier:
                self.buff_ffts[i] = (np.fft.fft(self.buffs[i]) / self.num_of_samples)**2 # it is squared to convert to power
            
        # return self.buffs, self.buff_ffts
        return self.buffs

        
    def plot_timetrace(self, idx=1):
        
        x = np.arange(0, self.decimation / self.clock * self.num_of_samples, self.decimation / self.clock)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(x * 1000, self.buffs[idx] * 1000, label='', lw=1)
        ax.set_title('raw time trace');
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')

        return fig
            
    def _calc_fftfreq(self):
        # determine the timestep to calculate the frequency axis
        timestep = self.decimation / self.clock
        
        # get the frequencies
        self.freq = pd.Series(np.fft.fftfreq(self.num_of_samples, d=timestep))
        self.freq2plot = self.freq[0:int(self.freq.size / 2)]
        
        return self.freq, self.freq2plot
    
    @staticmethod
    def _psd2dBm(power):
        dBm = 10 * np.log10(power / (0.001 * 50))
        return dBm
    
    def calc_fft(self, freq_range=(100, 1500), log=True):
        
        freq_min, freq_max = freq_range
        self.freq, self.freq2plot = self._calc_fftfreq()
        
        # get the first usable spectrum into the result variable
        fft_avgd = 2 * np.abs(self.buff_ffts[1 + self.offset][0:int(self.freq.size / 2)]) / self.num_of_avg
        for i in range(2 + self.offset, self.num_of_avg + self.offset):
            fft_avgd = fft_avgd + 2 * np.abs(self.buff_ffts[i][0:int(self.freq.size / 2)]) / self.num_of_avg
        
        fft_avgd_dBm = self._psd2dBm(fft_avgd) 
        
        # put it into a dataframe in order to easier select the frequency range
        self.fft_df = pd.DataFrame({'Freq': self.freq2plot, 'FFT': fft_avgd_dBm})
        # select frequency range
        self.fft_df = self.fft_df[(self.fft_df['Freq'] >= freq_min) & (self.fft_df['Freq'] <= freq_max)]

 
        return self.fft_df['FFT'], self.fft_df
    
    def plot_fft(self, freq_range=(1, 100e6), logx=False):
        freq_min, freq_max = freq_range

        fig, ax = plt.subplots(figsize=(4, 2))
        self.fft_df.plot(x='Freq', y='FFT', ax=ax)
        # ax.plot(self.freq2plot, np.sqrt(fft_avgd), label='Current Circuit')

        ax.set_title('Power Density Spectrum (' + str(self.num_of_avg) + ' average)');
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Displacement (dBm)')
        # ax.set_yscale('log')
        if logx: ax.set_xscale('log')
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'size': 8})
    
        return fig
        
class FrameAcquisitionThread(threading.Thread):
    
    def __init__(self, camera, download_frame_event, upload_pattern_event, stop_all_event, num_of_frames=1):
        """ 
        This is a thread that once a zelux camera is configured, armed and triggered, downloads one (or more) frame from
        the camera. The associated thread events syncronises this thread with the parallel ones. 
        Parameters
        ----------
        camera : class object
            zelux thorlabs SDK
        download_frame_event : thread event
        upload_pattern_event : thread event
        stop_all_event : thread event
        num_of_frames : int, optional
        """
        super(FrameAcquisitionThread, self).__init__()

        self.num_of_frames = num_of_frames
        self.camera = camera

        self.upload = upload_pattern_event
        self.download = download_frame_event
        self.stop = stop_all_event

        self.frames = []
        
    def run(self):
        
        while not self.stop.is_set():
            self.download.wait()
            # if self.download.is_set():
            #     print("downloading frame now")
            # Download the frame from the camera
            frame = self.camera.get_pending_frame_or_null()
            image_buffer_copy = np.copy(frame.image_buffer)
            self.frames.append(image_buffer_copy)
            
            # set flags to threads
            self.download.clear()
            self.upload.set()
            
            
class CameraThread(threading.Thread):

    def __init__(self, download_frame_event, upload_pattern_event, stop_all_event, roi=(0, 0, 1440, 1080), bins=(1, 1), exposure_time=11000, gain=6, timeout=1000):
        """This class opens a thread that run in parallel to the SLM thread (see patternSLM/patterns.py)
        It sets a bunch of thread events that act complementary fo the SLM thread. 
        First, it opens a thread that configures a zelux thorlabs camera. Then, a nested thread downloads frames from the camera.

        Parameters
        ----------
        download_frame_event : thread event
        upload_pattern_event : thead event
        stop_all_event : thread event
        roi : tuple, optional
            set region of intereset of the camer, by default (0, 0, 1440, 1080)
        bins : tuple, optional
            set the camera macropixel size, by default (1, 1)
        exposure_time : int, optional
            set camera exposure time, by default 11000
        gain : int, optional
            set camera gain, by default 6
        timeout : int, optional
            set camera timeout, by default 1000
        """
        super(CameraThread, self).__init__()
        self.roi = roi
        self.bins = bins
        self.gain = float(gain)
        self.exposure_time = exposure_time
        self.timeout = timeout
        
        self.upload = upload_pattern_event
        self.download = download_frame_event
        self.stop = stop_all_event

        self.frames = None
    
    def run(self):
        """ Inspired by thorlabs sdk example: detect available cameras, set camera properties, arm and trigger. 
        Then open a nested thread to download one frame from the camera once a pattern is uploaded to the SLM
        (the slm thread sets the corresponsind thread event).
        When everything is done, the camera is disarmed and disposed.
        """
        
        with TLCameraSDK() as sdk:
            available_cameras = sdk.discover_available_cameras()
            if len(available_cameras) < 1:
                print("no cameras detected")
            # Configure the camera settings
            print("USB camera configured, armed and triggered.")

            with sdk.open_camera(available_cameras[0]) as camera:
                camera.exposure_time_us = self.exposure_time  # set exposure to 11 ms
                camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode (if 0)
                camera.image_poll_timeout_ms = self.timeout  # polling timeout
                camera.roi = self.roi # set roi
                # set binning for camera macropixels
                (camera.binx, camera.biny) = self.bins

                if camera.gain_range.max > 0:
                    db_gain = self.gain
                    gain_index = camera.convert_decibels_to_gain(db_gain)
                    camera.gain = gain_index
                    # print(f"Gain: {camera.convert_gain_to_decibels(camera.gain)}")
                    # print(camera.gain_range, camera.gain)

                camera.arm(2)
                camera.issue_software_trigger()


                download_thread = FrameAcquisitionThread(camera,
                                                         self.download, 
                                                         self.upload, 
                                                         self.stop,
                                                         num_of_frames=1)
                download_thread.start()
                download_thread.join()

                camera.disarm()
                camera.dispose()

                self.frames = download_thread.frames



