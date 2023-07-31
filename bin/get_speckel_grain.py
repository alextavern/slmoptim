import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as sfft
from scipy import optimize
from slmOptim.zeluxPy.helper_functions import get_interferogram

def hist(array):
   counts, bins = np.histogram(array, bins=100)
   return counts, bins

def calc_acf(array):
   # fft of the speckle image
   fft = sfft.fft2(array)
   # power spectrum of fft
   power_spec = np.abs(fft) ** 2
   # inverse fft of power specturm + center freq
   acf = sfft.ifft2(power_spec).real
   acf_shift = sfft.fftshift(acf)
   
   return acf_shift
   
def gaussian_func(x, a, x0, sigma, b):
   return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + b


def fit_gaussian(array, init_values=[1e9, 100, 20, 6.5e8]):
   y_data = array
   x_data = range(0,len(y_data))
   
   params, params_covariance = optimize.curve_fit(gaussian_func, x_data, y_data,
                                           p0=init_values)
   
   return params

def plot(data):
   # plot a 2x2 figure
   fig, axs = plt.subplots(2, 2)
   
   #figure 1
   axs[0, 0].imshow(data['speckle'])
   axs[0, 0].set_xlabel('x (px)')
   axs[0, 0].set_ylabel('y (px)')

   # figure 2
   counts, bins = data['hist']
   axs[0, 1].hist(bins[:-1], bins, weights=counts)
   axs[0, 1].set_xlabel('intensity (a.u.)')
   axs[0, 1].set_ylabel('counts #')  
   # figure 3
   axs[1, 0].imshow(data['acf'])
   axs[1, 0].set_xlabel('x (px)')
   axs[1, 0].set_ylabel('y (px)')
   
   # figure 4
   y_data = data['acf'][100]
   x_data = range(0, len(y_data))
   axs[1, 1].plot(x_data, y_data, x_data, data['fit'])
   axs[1, 1].set_xlabel('x (px)')
   axs[1, 1].set_ylabel('intensity (a.u.)')
   axs[1, 1].legend(['raw', 'fit'])
   axs[1, 1].set_title('grain size: {:0.2f}'.format(data['grain_size']))
   fig.tight_layout()
   fig.savefig("speckle_grain.png", dpi=400, transparent=False)
   return fig
   


if __name__ == "__main__":
   
   data_out = {}
   
   # get speckle pattern
   speckle = get_interferogram(roi=(620, 440, 820, 640),
                           num_of_frames=1,
                           exposure_time=1000, 
                           gain=5, 
                           timeout=100)
   
   # calculate speckle histogram
   histogram = hist(speckle[1])
   
   # calculate speckle autocorrelation function
   acf = calc_acf(speckle[1])
   
   # fit and measure speckle grain size
   y_data = acf[100]
   x_data = range(0, len(y_data))
   fit_params = fit_gaussian(acf[100])
   fit = gaussian_func(x_data, 
                       fit_params[0], 
                       fit_params[1], 
                       fit_params[2], 
                       fit_params[3])
   
   # get fwhm of gaussian, this is the grain size
   fwhm = 2.355 * fit_params[2]
   print('The speckle grain size is equal to {} pixels'.format(fwhm))
   
   # save all to dictionary
   data_out['speckle'] = speckle[1]
   data_out['hist'] = histogram
   data_out['acf'] = acf
   data_out['fit'] = fit
   data_out['grain_size'] = fwhm
   
   # plot and save results
   figure = plot(data_out)
