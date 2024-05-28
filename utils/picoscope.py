import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ctypes

from picosdk.ps4000a import ps4000a as ps
from picosdk.functions import assert_pico_ok

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
        Records a set of data using the block acqusition mode from the picoscope
        Parameters
        ----------
        sample_number: int
            number of sample to collect (default= 100000)
        number_capture: int
        sampling_rate: float
            sampling rate in Hz. (default = 1)
            Minimal timebase is 12.5 ns, thus if the sampling is not a multiple of 12.5 ns,
            the closest value below the requested sampling rate will be used 
            this should be called re-sampling.

        Returns
        -------
        buffers: array
            A 2d arrays with the data for each channel 
            or a 1d array of the averaged spectrum (if fourier is True)
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
        
        # set up buffers - one buffer list for each average needs to be locally alocated
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
        
        # iteration through values of buffers dict to calculate the fft
        if self.fourier:
            for key, value in self.buffers.items():
                self.buffers[key] = np.fft.fft(value[0] / self.num_of_samples)**2 # it is squared to convert to power
        # average all buffers
            self.buffers = self._average_buff_fftdBm() 

        return self.buffers[self.frequency_range[0]:self.frequency_range[1]]
    
    def plot_fft(self):
        """ when get method returns an averaged spectrum this method plots it
        """
        if self.fourier:
            freq = self._calc_fftfreq()
            fig = plt.figure()
            plt.plot(freq, self.buffers)
            return fig
        else:
            print("no spectrum to plot")
    
    @staticmethod
    def _psd2dBm(power):
        """ converts power to dBm with a load of 50 Ohms
        """
        return 10 * np.log10(power / (0.001 * 50))

    def _calc_fftfreq(self):
        """ calculate frequency axis of an fft for a given sampling rate
            and a given number of samples
        """
        timestep = 1 / self.sampling_rate
        freq = pd.Series(np.fft.fftfreq(self.num_of_samples, d=timestep))
        return freq[0:int(freq.size / 2)]

    def _average_buff_fftdBm(self):
        """ takes a dictionary whose values are fft lists and averages them
        """
        # freq = self._calc_fftfreq()
        fft_avgd = np.zeros(int(self.num_of_samples / 2))
        for value in self.buffers.values():
            temp = 2 * np.abs(value[0:int(self.num_of_samples / 2)]) / self.num_of_avg
            fft_avgd += temp
        fft_avgd_dBm = self._psd2dBm(fft_avgd)
        return fft_avgd_dBm
        
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