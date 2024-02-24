from collections.abc import Callable, Iterable, Mapping
from typing import Any
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
import threading
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class RedPi:
    
    def __init__(self, rp_server, num_of_samples=16384, clock=125e6, prescaler=2**11, num_of_avg=10, offset=5):
            
        self.rp = rp_server 
        self.num_of_samples = num_of_samples
        self.clock = clock
        
        self.prescaler = prescaler
        
        self.rp.tx_txt('ACQ:DEC ' + str(self.prescaler))
        
        # make sure the prescaler was set
        self.rp.tx_txt('ACQ:DEC?')
        display('Prescaler set to: ' + str(self.rp.rx_txt()))
        
        # prepare buffer dataframes
        self.buff_ffts = pd.DataFrame()
        self.buffs = pd.DataFrame()
        
        self.num_of_avg = num_of_avg
        self.offset = offset # the offset takes a certain number of spectra in the beginning. 
                             # Sometimes the Red Pitaya produces trash in the first spectra

    def acquire(self):
        # do the acquisitions and save it in the computers memory (not on the Red Pitaya).
        for i in range(1, self.num_of_avg + self.offset):
            if i % 50 == 0:
                display(i)
            self.rp.tx_txt('ACQ:START')
            self.rp.tx_txt('ACQ:TRIG NOW')
            self.rp.tx_txt('ACQ:SOUR1:DATA?')

            buff_string = ''
            buff_string = self.rp.rx_txt()
            buff_string = buff_string.strip('{}\n\r').replace("  ", "").split(',')
            #display(buff_string)
            self.buffs[i] = list(map(float, buff_string))
            self.buff_ffts[i] = (np.fft.fft(self.buffs[i]) / self.num_of_samples)**2
            
        return self.buffs, self.buff_ffts
        
    def plot_timetrace(self, idx=1):
        
        x = np.arange(0, self.prescaler / self.clock * self.num_of_samples, self.prescaler / self.clock)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(x * 1000, self.buffs[idx] * 1000, label='', lw=1)
        ax.set_title('raw time trace');
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')

        return fig
            
    def _calc_fftfreq(self):
        # determine the timestep to calculate the frequency axis
        timestep = self.prescaler / self.clock
        
        # get the frequencies
        self.freq = pd.Series(np.fft.fftfreq(self.num_of_samples, d=timestep))
        self.freq2plot = self.freq[0:int(self.freq.size / 2)]
        
        return self.freq, self.freq2plot
        
            
    def calc_fft(self, freq_range=(100, 1500), log=True):
        
        freq_min, freq_max = freq_range
        self.freq, self.freq2plot = self._calc_fftfreq()
        
        # get the first usable spectrum into the result variable
        fft_avgd = 2 * np.abs(self.buff_ffts[1 + self.offset][0:int(self.freq.size / 2)]) / self.num_of_avg
        for i in range(2 + self.offset, self.num_of_avg + self.offset):
            fft_avgd = fft_avgd + 2 * np.abs(self.buff_ffts[i][0:int(self.freq.size / 2)]) / self.num_of_avg
        

        # put it into a dataframe in order to easier select the frequency range
        fft_df = pd.DataFrame({'Freq': self.freq2plot, 'FFT': np.sqrt(fft_avgd)})
        # select frequency range
        fft_df = fft_df[(fft_df['Freq'] >= freq_min) & (fft_df['Freq'] <= freq_max)]

        # plot
        fig, ax = plt.subplots(figsize=(4, 2))
        fft_df.plot(x='Freq', y='FFT', ax=ax)
        # ax.plot(self.freq2plot, np.sqrt(fft_avgd), label='Current Circuit')

        ax.set_title('Noise Amplitude Spectrum (' + str(self.num_of_avg) + ' average)');
        ax.set_xlabel('Freqency (Hz)')
        ax.set_ylabel('Noise amplitude. (V/$\sqrt{Hz}$)')
        ax.set_yscale('log')
        if log: ax.set_xscale('log')
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'size': 8})

        return fft_df, fig
    
        
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



