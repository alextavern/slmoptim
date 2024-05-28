from ..initializer.init_experiment import InitExperiment
from ..loader import patterns as pt

from ..optimizer import transmission_matrix, phase_conjugation
from ..utils import misc

from slmPy import slmpy

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time, os
from matplotlib.colors import LogNorm
import time

def reset_camera(init, bins, roi_size, time=200):
    
    init.close_cam()
    exposure_time = time
    init = InitExperiment(roi_size, off, bins, exposure_time, gain, timeout)
    camera = init.init_cam()
    return camera, init

def measure_tm(camera, num_in):
    
    # create patterns to load
    pattern_loader = pt.HadamardPatternGenerator(num_in ** 2, slm_calib_px)
    
    # instantiate tm
    tm_raw = transmission_matrix.measTM(slm,
                                    camera,
                                    pattern_loader=pattern_loader,
                                    slm_macropixel_size=slm_macropixel,
                                    calib_px=slm_calib_px, 
                                    save_path=path,
                                    remote=True)

    # run acquisition
    time_delay = 0.1
    frames = tm_raw.get2(time_delay)
    tm_raw.save()
    
    # calculater TM
    tr = transmission_matrix.calcTM(frames)
    tm_had, tm_fil, tm = tr.calc_plot_tm(figsize=(10, 3))   
    
    return tm

def conjugate(tm):
    
    # define target
    target_shape = (int(tm.shape[0] ** 0.5), int(tm.shape[0] ** 0.5))

    tgt = phase_conjugation.Target(target_shape)
    target_frame = tgt.gauss(num=16, order=0, w0=1e-4, slm_calibration_px=112)

    # phase conjugation - create mask
    msk = phase_conjugation.InverseLight(target_frame, tm, slm_macropixel=slm_macropixel, calib_px=112)
    phase_mask = msk.inverse_prop(conj=True)
    
    # get inversion operator and calculate snr
    inversion_operator_focus, inversion_operator_detection = msk.calc_inv_operator()
    snr = msk.snr()   
    
    return phase_mask, snr, target_frame, inversion_operator_focus, inversion_operator_detection
    
def focus(phase_mask):
    # merge phase mask into an slm pattern
    patternSLM = pt.Pattern(resX, resY)
    focusing_mask = patternSLM.add_subpattern(phase_mask, gray = 10)
    slm.sendArray(focusing_mask)
    return focusing_mask 

def set_mirror():
    patSLM = pt.Pattern(resX, resY)
    mirror = patSLM.mirror()
    slm.sendArray(mirror)
    time.sleep(.2)
    
def get_frame(camera):
    frame = camera.get_pending_frame_or_null()
    frame = np.copy(frame.image_buffer)
    
    return frame
    
def plot(phase_mask, frame_speck, frame_focus, off=0):
    fig, axs = plt.subplots(2, 2, figsize=(7,7))

    speck = axs[0, 0].imshow(frame_speck)
    axs[0, 0].set_title("Diffusing pattern")
    axs[0, 0].set_xlabel("Camera x px #")
    axs[0, 0].set_ylabel("Camera y px #")
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(speck, cax=cax)   

    mask = axs[1, 0].imshow(phase_mask)
    axs[1, 0].set_title("Focus mask")
    axs[1, 0].set_xlabel("SLM x px #")
    axs[1, 0].set_ylabel("SLM y px #")
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mask, cax=cax)   

    frame = axs[0, 1].imshow(frame_focus)
    # frame = axs[0, 1].imshow(frame_focus)
    # frame = axs[1].imshow(frame_focus)

    axs[0, 1].set_title("Focusing")
    axs[0, 1].set_xlabel("Camera x px #")
    axs[0, 1].set_ylabel("Camera y px #")
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(frame, cax=cax)   

    profile_line = len(frame_focus) // 2 + off
    axs[1, 1].plot(frame_focus[profile_line][:])
    axs[1, 1].set_box_aspect(1)
    axs[1, 1].set_title("Focus profile")
    axs[1, 1].set_xlabel("Camera x px #")
    axs[1, 1].set_ylabel("Intensity (a.u.) #")

    fig.tight_layout()
    
    timestr = time.strftime("%Y%m%d-%H%M")
    filename = "{}_focus_num_in{}_slm_macro{}.png".format(timestr, num_in, slm_macropixel)
    filepath = os.path.join(path, filename)
    fig.savefig(filepath, dpi=200, transparent=True)

    
def plot2(target_frame, inversion_operator_focus, inversion_operator_detection):
    fig2, axs = plt.subplots(1, 3, figsize=(10,10))

    tar = axs[0].imshow(abs(target_frame))
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig2.colorbar(tar, cax=cax)   

    inv_foc = axs[1].imshow(abs(inversion_operator_focus))
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig2.colorbar(inv_foc, cax=cax)   

    inv_det = axs[2].imshow(abs(inversion_operator_detection))
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig2.colorbar(inv_det, cax=cax)   

    fig2.tight_layout()
    
    timestr = time.strftime("%Y%m%d-%H%M")
    filename = "{}_operators_num_in{}_slm_macro{}.png".format(timestr, num_in, slm_macropixel)
    filepath = os.path.join(path, filename)
    fig2.savefig(filepath, dpi=200, transparent=True)
    
def signal2noise(frame, dark=1.033, mask_radius=8):
    """ Thank you S. Popoff
        Creates mask with a disk in the center and calculates the ratio of the
        pixel intensity in the disk to the pixel intensity outside the disk.
    """

    res = frame.shape
    
    max_idx = ndimage.maximum_position(frame)
    mask_center = [max_idx[0], max_idx[1]]
    
    
    Y, X = np.meshgrid(np.arange(res[0]),np.arange(res[1]))

    # We generate a mask representing the disk we want to intensity to be concentrated in
    mask = (X - mask_center[0]) ** 2 + (Y - mask_center[1]) ** 2 < mask_radius ** 2

    signal = np.sum((frame) * mask) / np.sum(mask)
    noise = np.sum((frame) * (1. - mask)) / np.sum(1. - mask) - dark
    snr = signal / noise

    return snr, mask
    
if __name__ ==  "__main__":
    
    n = [4, 8, 16, 32, 64]
    m = [7, 6, 5, 4, 3]
    
    tms=[]
    snrs = []
    masks = []
    frames = []
    
    for num_in, slm_macropixel in zip(n, m):
        
        camera, init = reset_camera(init, 5, 100, 200)
        tm = measure_tm(camera, num_in)
        mask, snr, target_frame, inversion_operator_focus, inversion_operator_detection  = conjugate(tm)
        focusing_mask = focus(mask)
        camera, init = reset_camera(init, 1, 200, 100)
        frame_focus = get_frame(camera)
        set_mirror()
        frame_speck = get_frame(camera)
        
        plot(mask, frame_speck, frame_focus, -1)
        plot2(target_frame, inversion_operator_focus, inversion_operator_detection)
        
        snr, maks = signal2noise(frame_focus, dark=1.033, mask_radius=1)
        tms.append(tm)
        masks.append(mask)
        frames.append(frame_focus)   
        snrs.append(snr)