from ..utils.download import ZeluxCamera, RedPitaya, PropheseeCamera
from ..utils.upload import SpatialLightModulator


# inspired by this simple illustration of the factory method design pattern in python
#https://medium.com/@vadimpushtaev/python-choosing-subclass-cf5b1b67c696

""" This class is used to initialize the hardware parts ot the experiment.
    In order to make adjustable for different hardware components in the future, 
    a simple class (InitExperiment) is used to instantiate classes with the hardware
    parameters. The user needs only to provide the class and add it manually to the
    mapping. 
    Here, the SLM - using slmpy - the Zelux Thorcam - using the thorlabs SDK and 
    a RedPitaya card are implemented.
    
    This class can be used either using an external configuration yaml file or by 
    using the classmethod decorated create_hardware to create a component individually
    for testing purposes.
    
    e.g. to create a hardware component:
    
    init_experiment = InitExperiment()
    camera = init_experiment.create_hardware('ccd')
    camera.init_cam(**kwargs)
"""

    
class InitExperiment():
    
    def __init__(self, **config):
        
        # this for-loop has the dictionary keys hard-coded
        # if there is a change in the config.yaml file
        # these changes must be made here too
        for component in config['hardware']:
            class_name = config['hardware'][component]['name'] # extract the class names
            hardware_params = config['hardware'][component]['params'] # extract hardware params

            new_hardware = self.create_hardware(class_name, **hardware_params)

            setattr(self, component, new_hardware) # set the class as an attrinute of this class

    @classmethod
    def create_hardware(cls, hardware_type, **params):
        """ Creates a hardware object by instantiating its corresponding class.
            The possible objects are included into a manually given map. If a new component
            needs to be implemented the user must include it into the map and
            provide its class (it should be located into utils.download/upload) 

        Parameters
        ----------
        hardware_type : string

        Returns
        -------
        instantiated class

        Raises
        ------
        ValueError
            if the hardware name is not included into the mapping
        """
        
        # here is the mapping
        HARDWARE_TYPE_TO_CLASS_MAP = {
            'slmpy': SpatialLightModulator,
            'ccd':  ZeluxCamera,
            'event-based': PropheseeCamera,
            'redpi': RedPitaya,
        }
        
        if hardware_type not in HARDWARE_TYPE_TO_CLASS_MAP:
            raise ValueError('Bad hardware type {}'.format(hardware_type))
    
        return HARDWARE_TYPE_TO_CLASS_MAP[hardware_type](**params)
    



# class InitExperiment_old():
    
#     def __init__(self, 
#                  roi_size=100, 
#                  roi_off=(0, 0), 
#                  bins=1, 
#                  exposure_time=100, 
#                  gain=1, 
#                  timeout=100,
#                  remote=True,
#                  SERVER = '10.42.0.234', 
#                  monitor=1): 
        
#         # camera settings
#         self.roi_size = roi_size
#         self.roi_off = roi_off
#         self.bins = bins
#         self.exposure_time = exposure_time
#         self.gain = gain
#         self.timeout = timeout
        
#         # slm settings
#         self.remote = remote
#         self.SERVER = SERVER
#         self.monitor = monitor
        
#     def init_slm(self):
#         """ initializes slmpy SLM
#         """
#         if self.remote:
#             self.slm = slmpy.Client()
#             self.slm.start(self.SERVER)
#         else:    
#             self.slm = slmpy.SLMdisplay(self.monitor)
#         return self.slm
    
#     def close_slm(self):
#         self.slm.close()
    
#     def set_roi(self):
#         """ Calculates the Region of interest. The user gives a window size and x, y offsets from
#             the sensor center

#         Returns
#         -------
#         roi (tuple or int)
#         """
#         if type(self.roi_size) is tuple:
#             width, height = self.roi_size
#         else:
#             width = self.roi_size
#             height = width
            
#         offset_x, offset_y = self.roi_off
#         middle_x = int(1440 / 2) + offset_x
#         middle_y = int(1080 / 2) - offset_y

#         roi = (middle_x - int(width/ 2), 
#             middle_y - int(height / 2), 
#             middle_x + int(width / 2), 
#             middle_y + int(height / 2))
        
#         return roi
        
#     def init_cam(self):
#         """ Initializes and sets camera parameters
#         """
#         # camera instance
#         self.sdk = TLCameraSDK()
#         available_cameras = self.sdk.discover_available_cameras()
#         self.camera = self.sdk.open_camera(available_cameras[0])

#         # configure
#         self.camera.exposure_time_us = self.exposure_time  # set exposure to 11 ms
#         self.camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
#         self.camera.image_poll_timeout_ms = self.timeout  # 1 second polling timeout
#         self.camera.roi = self.set_roi()

#         # set binning for camera macropixels
#         (self.camera.binx, self.camera.biny) = (self.bins, self.bins)

#         if self.camera.gain_range.max > self.gain:
#             db_gain = self.gain
#             gain_index = self.camera.convert_decibels_to_gain(db_gain)
#             self.camera.gain = gain_index

#         # arm - trigger
#         self.camera.arm(2)
#         self.camera.issue_software_trigger()
        
#         return self.camera
    
#     def close_cam(self):
#         self.camera.disarm()
#         self.camera.dispose()
#         self.sdk.dispose()        
        
#     def config(self):
#         cam = self.init_cam()
#         slm = self.init_slm()
        
#         return cam, slm
        
#     def destroy(self):
#         self.close_slm()
#         self.close_cam()

