import numpy as np

def correlation_coefficient(X, Y):
    """ takes two 2d arrays, flattens them calculates the 
        corresponding covariance matrix. the correlation coefficient
        is calculated by r = cov(x, y) / sqrt(var(x) * var(y))
    """
    # flatten the input matrices
    x = X.flatten()
    y = Y.flatten()
    # stack them
    stacked = np.stack((x, y), axis=0)

    # calculate the covariance matrix
    covar = np.cov(stacked)
    # use the cov matrix elements to calculate correlation coefficient
    corr_coeff = covar[0, 1] / np.sqrt(covar[0, 0] * covar[1, 1])
    return corr_coeff


def signal_to_noise_ratio1d(data, mask_radius=1):

    center = np.argmax(data)

    X = np.arange(len(data))
    # We generate a mask representing the disk we want to intensity to be concentrated in
    mask = (X - center) ** 2 < mask_radius ** 2
    
    signal = np.sum((data) * mask) / np.sum(mask)
    noise = np.sum((data) * (1. - mask)) / np.sum(1. - mask)
    cost = signal - noise  # substraction because input signal is in dB
    
    return cost

def signal_to_noise_ratio2d(frame, mask_radius=8, mask_offset=(0, 0), intesity_only=False):
    """ Thank you S. Popoff
        Creates mask with a disk in the center and calculates the ratio of the
        pixel intensity in the disk to the pixel intensity outside the disk.
    """

    res = frame.shape
    off = mask_offset
    mask_center = [res[0] // 2 + off[0], res[1] // 2 + off[1]]
    
    X, Y = np.meshgrid(np.arange(res[0]),np.arange(res[1]))

    # We generate a mask representing the disk we want to intensity to be concentrated in
    mask = (X - mask_center[0]) ** 2 + (Y - mask_center[1]) ** 2 < mask_radius ** 2

    if intesity_only:
        cost = np.sum((frame) * mask)
    else:
        signal = np.sum((frame) * mask) / np.sum(mask)
        noise = np.sum((frame) * (1. - mask)) / np.sum(1. - mask)
        cost = signal / noise

    return cost    

def max_of_spectrum(spectrum):
    """ Takes an FFT series and returns its maximum"""
    return np.max(spectrum)

def peak_to_peak(time_series):
    """ Take a times series and returns the peak to peak voltage"""
    return np.ptp(time_series)

def optomechanical_waist(frames):
    """ Implements an optomechanical waist parameter-based cost function
        (see VB these p. 93). To use with an iterative optimization method.
    """
        
    num_of_frames = len(frames)
    
    # create a zero 2d array on which the measurement mode will be averaged
    shape = frames[0].shape
    meas_mode_avg =  np.zeros((shape[0], shape[1]))

    for idx in range(1, num_of_frames):
        # difference between the square roots of two frames
        frame_diff = (frames[idx-1] - frames[idx]) / (frames[idx] ** 0.5)

        meas_mode_avg += frame_diff / num_of_frames

    return meas_mode_avg
    
def measurement_mode(frames):
    """ calculating measurement mode from a series of frames
    """
    
    num_of_frames = len(frames)
    
    # create a zero 2d array on which the measurement mode will be averaged
    shape = frames[0].shape
    meas_mode_avg =  np.zeros((shape[0], shape[1]))

    for idx in range(1, num_of_frames):
        # difference between the square roots of two frames
        frame_diff = (frames[idx-1] ** 0.5 - frames[idx] ** 0.5)

        # normalization factor of this particular differnce
        frame_diff_squared = frame_diff ** 2
        norm_factor = frame_diff_squared.sum() ** 0.5

        # normalize
        frame_diff_norm = frame_diff / norm_factor
        # correct sign otherwise everything will be averaged to zero
        frame_diff_norm *= np.sign(frame_diff_norm)
        
        # average 
        meas_mode_avg += frame_diff / num_of_frames

    return meas_mode_avg
    
    