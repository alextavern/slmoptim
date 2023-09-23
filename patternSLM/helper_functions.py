from mpl_toolkits.axes_grid1 import make_axes_locatable
from slmPy import slmpy
from ..zeluxPy import helper_functions as cam
import matplotlib.pyplot as plt
from ..patternSLM import patterns as  pt

def check(ij, dim, slm_macropixel_size, roi, bins, exposure_time, gain, timeout, norm=True):
# do some checks

    slm = slmpy.SLMdisplay(monitor=1)
    resX, resY = slm.getSize()

    slm_patterns = pt.Pattern(resX, resY)
    _, pattern = slm_patterns.hadamard_pattern(dim, ij, n=slm_macropixel_size, gray=0)

    slm.updateArray(pattern)
    frame = cam.get_frame_binned(roi, bins, gain, exposure_time, gain, timeout)

    if norm:
        frame = cam.normalize_frame(frame)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(pattern)
    axs[0].set_xlabel('SLM x pixels #')
    axs[0].set_ylabel('SLM y pixels #')
    axs[0].set_title('SLM pattern ')

    fr = axs[1].imshow(frame)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(fr, cax=cax)
    axs[1].set_xlabel('Camera x pixels #')
    axs[1].set_ylabel('Camera y pixels #')
    axs[1].set_title('Camera speckle')
    
    fig.tight_layout()
    
    slm.close()
    
    return pattern, frame    

def plot_focus(frame1, frame2, norm=True):

    
    fig, axs = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
    if norm:
        frame1 = cam.normalize_frame(frame1)
        frame2 = cam.normalize_frame(frame2)

    mask = axs[0].imshow(frame1, aspect=1)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mask, cax=cax)

    fo = axs[1].imshow(frame2)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(fo, cax=cax)
    
    axs[0].set_title("w/ mask")
    axs[1].set_title("w/o mask")

    fig.text(0.5, 0.25, 'camera pixels x #', ha='center')

    fig.text(-0.01, 0.5, 'camera pixels y #', va='center', rotation='vertical')
    fig.tight_layout()
    return fig