from mpl_toolkits.axes_grid1 import make_axes_locatable
from slmPy import slmpy
from ..utils import camera_func as cam
import matplotlib.pyplot as plt
from ..loader import patterns as  pt
import numpy as np
import time
from scipy import ndimage
import numpy as np
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK




def pattern2interferogram(slm, camera, pattern, slm_macropixel, slm_resolution=(800, 600), 
          remote=True, 
          norm=False):
    
    resX, resY = slm_resolution
    slm_patterns = pt.PatternsBacic(resX, resY)
    pattern = slm_patterns.pattern_to_SLM(pattern, n=slm_macropixel)
    # _, pattern = slm_patterns.hadamard_pattern(N, ij, n=slm_macropix, gray=0)
    
    if remote:
        slm.sendArray(pattern)
    else:
        slm.updateArray(pattern)
    
    time.sleep(0.5)
    frame = camera.get_pending_frame_or_null()
    frame = np.copy(frame.image_buffer)
    
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