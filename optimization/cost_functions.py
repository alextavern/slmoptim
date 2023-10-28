import numpy as np

def corr_coef(X, Y):
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


def snr(frame, mask_radius=8, mask_offset=(0, 0), intesity_only=False):
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

    return cost, mask    