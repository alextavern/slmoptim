from mpl_toolkits.axes_grid1 import make_axes_locatable
from slmPy import slmpy
from ..zeluxPy import helper_functions as cam
import matplotlib.pyplot as plt
from ..patternSLM import patterns as  pt

def check(ij, order, mag, roi, bins, exposure_time, gain, timeout):
# do some checks

    slm = slmpy.SLMdisplay(monitor=1)
    resX, resY = slm.getSize()

    slm_patterns = pt.Pattern(resX, resY)
    _, pattern = slm_patterns.hadamard_pattern(order, ij, n=mag, gray=0)

    slm.updateArray(pattern)
    frame = cam.get_frame_binned(roi, bins, gain, exposure_time, gain, timeout)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(pattern)
    axs[0].set_xlabel('SLM x pixels #')
    axs[0].set_xlabel('SLM y pixels #')
    axs[0].set_title('SLM pattern ')

    fr = axs[1].imshow(frame[1])
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(fr, cax=cax)
    axs[1].set_xlabel('Camera x pixels #')
    axs[1].set_xlabel('Camera y pixels #')
    axs[1].set_title('Camera speckle')
    
    slm.close()
    
    return pattern, frame

def plot_focus(pattern, frame):

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    mask = axs[0].imshow(pattern, aspect=1)
    # fig.colorbar(mask, ax=axs[0])
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mask, cax=cax)

    fo = axs[1].imshow(frame[1])
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(fo, cax=cax)