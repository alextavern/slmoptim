from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import camera_func as cam
import matplotlib.pyplot as plt
from ..loader import patterns as  pt
import numpy as np
import time, pickle, os





def get_params(self, **config):

    for key, value in config.items():
        setattr(self, key, value) # sets the instanitated class as an attrinute of this class            

def pattern_frame(slm, camera, pattern, slm_macropixel, slm_resolution=(800, 600),
                  off=(0, 0), 
                  remote=True, 
                  norm=False):
    
    resX, resY = slm_resolution
    slm_patterns = pt.PatternsBacic(resX, resY)
    pattern = slm_patterns.pattern_to_SLM(pattern, n=slm_macropixel, off=off)
    
    if remote:
        slm.sendArray(pattern)
    else:
        slm.updateArray(pattern)
    
    time.sleep(0.5)
    frame = camera.get()
    # frame = np.copy(frame.image_buffer)
    
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


def two_frames(frame1, frame2, norm=True):

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

def create_filepath(self, **kwargs):
    """ creates a filepath to save data
    """

    date_str = time.strftime("%Y%m%d")
    date_time_str = time.strftime("%Y%m%d-%H:%M")
    
    new_path = os.path.join(self.save_path, date_str)
    
    # check if dir exists
    isExist = os.path.exists(new_path)
    # and create it
    if not isExist:
        os.makedirs(new_path)
    
    # create the filename based on kwargs    
    date_time_str = time.strftime("%Y%m%d-%H:%M")
    filename = str(date_time_str)
    for key, value in kwargs.items():
        filename += '_'
        filename += str(key) +  str(value)
    
    # filename = '{}_tm_raw_data_num_in{}_slm_macro{}'.format(date_time_str,  
    #                                                         self.num_in, 
    #                                                         self.slm_macropixel_size)
    #filename = '{}_'.format(date_time_str) + '{}_raw_data_num_in{}_slm_macro{}'.format(self.type, 
    #                                                                                   self.num_in, 
    #                                                                                   self.slm_macropixel_size)
    
    if self.save_path:
        self.filepath = os.path.join(new_path, filename)
    else:
        self.filepath = filename
        
    return self.filepath
    
# def create_filepath(self):
#     """ creates a filepath to save data
#     """

#     date_str = time.strftime("%Y%m%d")
#     date_time_str = time.strftime("%Y%m%d-%H:%M")
    
#     new_path = os.path.join(self.save_path, date_str)
    
#     # check if dir exists
#     isExist = os.path.exists(new_path)
#     # and create it
#     if not isExist:
#         os.makedirs(new_path)
    
#     # filename = '{}_tm_raw_data_num_in{}_slm_macro{}'.format(date_time_str,  
#     #                                                         self.num_in, 
#     #                                                         self.slm_macropixel_size)
#     filename = '{}_'.format(date_time_str) + '{}_raw_data_num_in{}_slm_macro{}'.format(self.type, 
#                                                                                        self.num_in, 
#                                                                                        self.slm_macropixel_size)
    
#     if self.save_path:
#         self.filepath = os.path.join(new_path, filename)
#     else:
#         self.filepath = filename
        
#     return self.filepath
        
def save_raw(self):
    """ saves raw data to a pickle format
    """

    with open(self.filepath + '.pkl', 'wb') as fp:
        pickle.dump((self.data_out), fp)