from slmPy import slmpy
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

class InitExperiment():

    def __init__(self, 
                 roi_size=100, 
                 roi_off=(0, 0), 
                 bins=1, 
                 exposure_time=100, 
                 gain=1, 
                 timeout=100,
                 remote=True,
                 SERVER = '10.42.0.234', 
                 monitor=1): 
        
        # camera settings
        self.roi_size = roi_size
        self.roi_off = roi_off
        self.bins = bins
        self.exposure_time = exposure_time
        self.gain = gain
        self.timeout = timeout
        
        # slm settings
        self.remote = remote
        self.SERVER = SERVER
        self.monitor = monitor
        
    def init_slm(self):
        if self.remote:
            self.slm = slmpy.Client()
            self.slm.start(self.SERVER)
        else:    
            self.slm = slmpy.SLMdisplay(self.monitor)
        return self.slm
    
    def close_slm(self):
        self.slm.close()
    
    def set_roi(self):
        """ Calculates the Region of interest. The user gives a window size and x, y offsets from
            the sensor center

        Returns
        -------
        roi (tuple or int)
        """
        print()
        if type(self.roi_size) is tuple:
            width, height = self.roi_size
        else:
            width = self.roi_size
            height = width
            
        offset_x, offset_y = self.roi_off
        middle_x = int(1440 / 2) + offset_x
        middle_y = int(1080 / 2) - offset_y

        roi = (middle_x - int(width/ 2), 
            middle_y - int(height / 2), 
            middle_x + int(width / 2), 
            middle_y + int(height / 2))
        
        return roi
        
    def init_cam(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        # camera instance
        self.sdk = TLCameraSDK()
        available_cameras = self.sdk.discover_available_cameras()
        self.camera = self.sdk.open_camera(available_cameras[0])

        # configure
        self.camera.exposure_time_us = self.exposure_time  # set exposure to 11 ms
        self.camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
        self.camera.image_poll_timeout_ms = self.timeout  # 1 second polling timeout
        self.camera.roi = self.set_roi()

        # set binning for camera macropixels
        (self.camera.binx, self.camera.biny) = (self.bins, self.bins)

        if self.camera.gain_range.max > self.gain:
            db_gain = self.gain
            gain_index = self.camera.convert_decibels_to_gain(db_gain)
            self.camera.gain = gain_index

        # arm - trigger
        self.camera.arm(2)
        self.camera.issue_software_trigger()
        
        return self.camera
    
    def close_cam(self):
        self.camera.disarm()
        self.camera.dispose()
        self.sdk.dispose()        
        
    def config(self):
        cam = self.init_cam()
        slm = self.init_slm()
        
        return cam, slm
        
    def destroy(self):
        self.close_slm()
        self.close_cam()

