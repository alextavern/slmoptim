import matplotlib.pyplot as plt
import threading
import numpy as np
import pandas as pd
import ctypes

from picosdk.ps4000a import ps4000a as ps
from picosdk.functions import assert_pico_ok
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

from .redpitaya_scpi import scpi
from .misc import get_params

class PropheseeCamera():
    
    def __init__(self) -> None:
        pass


class PicoScope():
    """ A class using the picosdk to fetch timeseries data from picoscope. Heavily inspired
        by Loic Rondin's implementation. Merci Loic !
        
        https://github.com/picotech/picosdk-python-wrappers/blob/master/ps4000aExamples/ps4000aRapidBlockExample.py
    """
    
    def __init__(self, **config):
            # hardware constants
            self.SAMPLE_RATE = 80e6
            self.NUM_OF_CHANNELS = 2
        
            # Create chandle and status ready for use
            self.chandle = ctypes.c_int16()
            self.status = {}
            self.active_ch = []

            # parameters and attributes
            self.config = config
            self._get_params(**config)
            self.params = self._config_constructor()

    def _get_params(self, **config):
        for key, value in config.items():
            setattr(self, key, value) # sets the instanitated class as an attrinute of this class

    def _config_constructor(self):
        
        # create here new dict that the picoscope can understand
        params = {}

        # create all channels
        params['ch'] = []
        for i in range(self.NUM_OF_CHANNELS):
            params['ch'].append([0,1,1,0])

        # activate user-defined channel
        params['ch'][self.channel] = [1, self.coupling, self.voltage_range, self.offset] 

        # set trigger
        params['trigger'] = [self.trigger, 0, 1024, 2, 0, 1000]

        # num of samples
        params['num_of_samples'] = self.num_of_samples
        
        # sampling rate
        params['sampling_rate'] = self.sampling_rate
        
        return params
    
    def start(self):
        """Initialize the pico scope"""
        # Open PicoScope 4000 Series device
        # Returns handle to chandle for use in future API functions
        self.status["openunit"] = ps.ps4000aOpenUnit(ctypes.byref(self.chandle), None)
        try:
            assert_pico_ok(self.status["openunit"])
        except:	

            powerStatus = self.status["openunit"]	

            if powerStatus == 286:
                self.status["changePowerSource"] = ps.ps4000aChangePowerSource(self.chandle, powerStatus)
            else:
                raise	

            assert_pico_ok(self.status["changePowerSource"])
        return self.status
    
    def set_config(self):
        """
        Configure the aquisition of the picoscope (channel and trigger)

        Parameters
        ----------
        params : dictionary hosting the config
           params['ch'][i_channel] = [enabled,coupling,range,analogOffset]
           params['trigger'] = [enabled , source,threshold, direction, delay, auto Trigger time (ms)] 

           - enabled = 0 (disabled) or 1
           - coupling = 0 (AC) or 1 (DC)
           - range = 0 (±10 mV), 1 (±20 mV), 2 (±50), 3 (±100), 4 (±200), 5 (±500), 6 (±1 V), 7 (± 2V)
           - analogOffset in V

        TODO? gestion param correctC
        """
        # Set up channel 
        # handle = chandle
        # channel = PS4000a_CHANNEL_A = 0
        # enabled = 1
        # coupling type = PS4000a_DC = 1
        # range = PS4000a_2V = 7,verticalContainer
        # analogOffset = 0 V
        # self.params = params
        pch = self.params['ch']
        for i in range(self.NUM_OF_CHANNELS):
            if(pch[i][0]==1):
                self.active_ch.append(i)
            self.status["setCh"] = ps.ps4000aSetChannel(self.chandle, i, pch[i][0], pch[i][1], pch[i][2], pch[i][3])
            assert_pico_ok(self.status["setCh"])
        self.Nch = np.size(self.active_ch)

        # Set up single trigger
        # handle = chandle
        # enabled = 1
        # source = PS4000a_CHANNEL_A = 0
        # threshold = 1024 ADC counts
        # direction = PS4000a_RISING = 2
        # delay = 0 s
        # auto Trigger = 1000 ms
        ptrig = self.params['trigger']
        self.status["trigger"] = ps.ps4000aSetSimpleTrigger(self.chandle, ptrig[0], ptrig[1], 
                                                       ptrig[2], ptrig[3], ptrig[4], ptrig[5])
        assert_pico_ok(self.status["trigger"])
        
        #set up acquisition
        #pSR= param['setSR']
        #self.sampleInterval = ctypes.c_int32(pSR['SR'])
        #self.sampleUnits = ps.PS4000A_TIME_UNITS['PS4000A_US'] #TODO
        
    def get_config(self):
        """return the config file that is in use"""
        return self.params
    
    def get(self):
        """
        Record a set of data using the block acqusition mode from the picoscope
        Parameters
        ----------
        sample_number: int
            number of sample to collect (default= 100000)
        number_capture: int
            TODO (default=1)
        sampling_rate: float
            sampling rate in Hz. (default =1)
            Minimal timebase is 12.5 ns, thus if the sampling is not a multiple of 12.5 ns,
            the closest value below the requested sampling rate will be used 
            this should be called re-sampling.

         Returns
        -------
        buffers: array
            A 2d arrays with the data for each channel 
        """
        # Get timebase information
        # WARNING: When using this example it may not be possible to access all Timebases as all channels are enabled by default when opening the scope.  
        # To access these Timebases, set any unused analogue channels to off.
        # handle = chandle
        # timebase = 8 => 8x(12,5 ns) or sampling rate 80 MHz/(8+1)
        # noSamples = maxSamples
        # pointer to timeIntervalNanoseconds = ctypes.byref(timeIntervalns)
        # pointer to maxSamples = ctypes.byref(returnedMaxSamples)
        # segment index = 0
        
        self.timebase = int(self.SAMPLE_RATE / self.sampling_rate - 1 ) # 1 MS/s
        timeIntervalns = ctypes.c_float()
        returnedMaxSamples = ctypes.c_int32()
        oversample = ctypes.c_int16(1)
        self.status["getTimebase2"] = ps.ps4000aGetTimebase2(
            self.chandle, 
            self.timebase, 
            self.num_of_samples,
            ctypes.byref(timeIntervalns), 
            ctypes.byref(returnedMaxSamples), 
            0
            )
        
        assert_pico_ok(self.status["getTimebase2"])
        
        nMaxSamples = ctypes.c_int32(0)
        
        self.status["setMemorySegments"] = ps.ps4000aMemorySegments(
            self.chandle, 
            self.num_of_avg, 
            ctypes.byref(nMaxSamples))
        
        assert_pico_ok(self.status["setMemorySegments"])
        
        # Set number of captures
        # handle = chandle
        # nCaptures = 
        self.status["SetNoOfCaptures"] = ps.ps4000aSetNoOfCaptures(self.chandle, self.num_of_avg)
        
        assert_pico_ok(self.status["SetNoOfCaptures"])
        
        # set up buffers
        self.buffers = {}
        for capture_idx in range(self.num_of_avg):
            self.buffers[capture_idx] = np.zeros(shape=(self.Nch, self.num_of_samples), dtype=np.int16)
            for i in range(self.Nch):
                ch = self.active_ch[i]
                self.status["setDataBuffers" + str(capture_idx)] = ps.ps4000aSetDataBuffers(
                    self.chandle, 
                    ch,
                    self.buffers[capture_idx][i].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                    None, 
                    self.num_of_samples,
                    capture_idx,
                    0
                    )
                
            assert_pico_ok(self.status["setDataBuffers" + str(capture_idx)])

    
        # run block capture
        # handle = chandle
        # number of pre-trigger samples = preTriggerSamples
        # number of post-trigger samples = PostTriggerSamples
        # timebase = 3 = 80 ns = timebase (see Programmer's guide for mre information on timebases)
        # time indisposed ms = None (not needed in the example)
        # segment index = 0
        # lpReady = None (using ps4000aIsReady rather than ps4000aBlockReady)
        # pParameter = None
        
        self.status["runBlock"] = ps.ps4000aRunBlock(
            self.chandle, 
            0, 
            self.num_of_samples, 
            self.timebase, 
            None, 
            0,  
            None, 
            None)
        
        assert_pico_ok(self.status["runBlock"])
        # check for end of capture
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            self.status["isReady"] = ps.ps4000aIsReady(self.chandle, ctypes.byref(ready))
        
        # Creates a overlow location for data
        overflow = (ctypes.c_int16 * self.num_of_avg)()
        # Creates converted types maxsamples
        cmaxSamples = ctypes.c_int32(self.num_of_samples)
        
        # collect data 
        # handle = chandle
        # noOfSamples = cmaxSamples
        # fromSegmentIndex = 0
        # toSegmentIndex = 9
        # downSampleRatio = 1
        # downSampleRatioMode = PS4000A_RATIO_MODE_NONE
        self.status["getValuesBulk"] = ps.ps4000aGetValuesBulk(
            self.chandle, 
            ctypes.byref(cmaxSamples), 
            0, 
            self.num_of_avg - 1, 
            1, 
            0, 
            ctypes.byref(overflow)
            )
        
        assert_pico_ok(self.status["getValuesBulk"])
        
        if self.fourier:
            for key, value in self.buffers.items():
                self.buffers[key] = np.fft.fft(value[0] / self.num_of_samples)**2 # it is squared to convert to power

            self.buffers = self._average_buff_fftdBm() 

        return self.buffers[self.frequency_range[0]:self.frequency_range[1]]
    
    def plot_fft(self):
        if self.fourier:
            freq = self._calc_fftfreq()
            fig = plt.figure()
            plt.plot(freq, self.buffers)
            return fig
        else:
            print("no spectrum to plot")
    
    @staticmethod
    def _psd2dBm(power):
        dBm = 10 * np.log10(power / (0.001 * 50))
        return dBm

    def _calc_fftfreq(self):
        timestep = 1 / self.sampling_rate
        freq = pd.Series(np.fft.fftfreq(self.num_of_samples, d=timestep))
        freq2plot = freq[0:int(freq.size / 2)]
        return freq2plot

    def _average_buff_fftdBm(self):
        # freq = self._calc_fftfreq()
        fft_avgd = np.zeros(int(self.num_of_samples / 2))
        for value in self.buffers.values():
            temp = 2 * np.abs(value[0:int(self.num_of_samples / 2)]) / self.num_of_avg
            fft_avgd += temp
        fft_avgd_dBm = self._psd2dBm(fft_avgd)
        return fft_avgd_dBm
        #
    def stop(self):
        """Stop and close the picoscope"""
        # Stop the scope
        # handle = chandle
        self.status["stop"] = ps.ps4000aStop(self.chandle)
        assert_pico_ok(self.status["stop"])	

        # Disconnect the scope
        # handle = chandle
        self.status["close"] = ps.ps4000aCloseUnit(self.chandle)
        assert_pico_ok(self.status["close"])	

        # Display self.status returns
        return self.status
    
class ZeluxCamera():
    
    def __init__(self, **config):
        print("Zelux camera initiliazed - use init_cam() to arm and trigger it and get() to get frames")
        get_params(self, **config)
    
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
            
        offset_x, offset_y = self.offset
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

    def __init__(self, **config):
        
        # get parameters
        get_params(self, **config)
        
        # launch scpi server - make sure it is manually activated from the redpi interface
        self.rp = self._launch_scpi_server()
        

        self.rp.tx_txt('ACQ:DEC ' + str(self.decimation))
        
        # make sure the decimation was set
        self.rp.tx_txt('ACQ:DEC?')
        self.dec = self.rp.rx_txt()
        # display('Decimation set to: ' + str(self.rp.rx_txt()))
        # print('Red Pitaya daq loaded - decimation set to: ' + str(self.rp.rx_txt()))

        # prepare buffer dataframes
        self.buff_ffts = pd.DataFrame()
        self.buffs = pd.DataFrame()
                             
    def _launch_scpi_server(self):
        server = scpi(self.IP)
        return server

    def get(self, fourier=True):
        """ get time trace from redpitaya
        """
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
        return self.buff_ffts if fourier else self.buffs
        # return self.buffs

        
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
    
    def calc_fft(self, freq_range=(1000, 10000), log=True):
        
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

        # ax.set_title('Power Density Spectrum (' + str(self.num_of_avg) + ' average)');
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



