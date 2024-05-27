import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from .redpitaya_scpi import scpi
from ..utils.misc import CommonMethods


class RedPitaya(CommonMethods):

    def __init__(self, **config):
        
        # get parameters
        CommonMethods.get_params(self, **config)
        # get_params(self, **config)
        
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
    
    def sine(self, waveform='sine', freq=1000, amp=1):
        
        self.rp.tx_txt('GEN:RST')

        self.rp.tx_txt('SOUR1:FUNC ' + str(waveform).upper())
        self.rp.tx_txt('SOUR1:FREQ:FIX ' + str(freq))
        self.rp.tx_txt('SOUR1:VOLT ' + str(amp))
    
        self.rp.tx_txt('OUTPUT:STATE ON')
        self.rp.tx_txt('SOUR:TRig:INT')
        
        self.rp.close()
        
    def sine_on_off(self, freq=3280, amp=0.1, gate=0.1, cycles=10):
        cycle = 0
        while cycle < cycles:
            self.sine(freq=freq, amp=amp)
            time.sleep(gate)
            self.off()
            cycle += 1
            
    def off(self):
        self.rp.tx_txt('OUTPUT:STATE OFF')

    def close(self):
        self.rp.close()
