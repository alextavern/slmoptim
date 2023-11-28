import numpy as np
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK


def set_roi(roi_size, roi_off):
    offset_x, offset_y = roi_off
    middle_x = int(1440 / 2) + offset_x
    middle_y = int(1080 / 2) - offset_y

    roi = (middle_x - int(roi_size / 2), 
            middle_y - int(roi_size / 2), 
            middle_x + int(roi_size / 2), 
            middle_y + int(roi_size / 2))
    
    return roi

def get_interferogram(roi=(0, 0, 1440, 1080), 
                      bins=(1,1), 
                      num_of_frames=1, 
                      exposure_time=5000, 
                      gain=1, timeout=10000, 
                      return_roi=False):
    camera = Cam(roi, bins, num_of_frames, exposure_time, gain, timeout, return_roi)
    frames = camera.get_frames()
    return frames


def get_frame_binned(roi, bins, num_of_frames=1, exposure_time=5000, gain=1, timeout=1000, return_roi=False):

    frame = get_interferogram(roi=roi,
                              bins=(bins, bins),
                              num_of_frames=num_of_frames,
                              exposure_time=exposure_time,
                              gain=gain,
                              timeout=timeout, 
                              return_roi=return_roi)

    return frame

def rotate_frame(frame, angle):
    frame_rot = ndimage.rotate(frame, angle)
    return frame_rot


def normalize_frame(frame):
    frame_norm = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
    return frame_norm


class InitCamera():
    def __init__(self, roi_size=100, roi_off=(0, 0), bins=1, exposure_time=50, gain=1, timeout=100):    
        # camera settings
        self.roi_size = roi_size
        self.offset_x, self.offset_y = roi_off # roi offset from center
        self.bins = bins
        self.exposure_time = exposure_time
        self.gain = gain
        self.timeout = timeout
    
    def set_roi(self):
        """ Calculates the Region of interest. The user gives a window size and x, y offsets from
            the sensor center

        Returns
        -------
        roi (tuple)
        """

        middle_x = int(1440 / 2) + self.offset_x
        middle_y = int(1080 / 2) - self.offset_y

        roi = (middle_x - int(self.roi_size / 2), 
               middle_y - int(self.roi_size / 2), 
               middle_x + int(self.roi_size / 2), 
               middle_y + int(self.roi_size / 2))
        
        return roi
        
    def config(self):
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
    
    def destroy(self):
        self.camera.disarm()
        self.camera.dispose()
        self.sdk.dispose()
        

class Cam:

    def __init__(self, roi=(0, 0, 1440, 1080), 
                 bins=(1, 1), 
                 num_of_frames=10, 
                 exposure_time=11000, 
                 gain=6, 
                 timeout=1000, 
                 return_roi=False):
        self.roi = roi
        self.bins = bins
        self.num_of_frames = num_of_frames
        self.gain = float(gain)
        self.exposure_time = exposure_time
        self.timeout = timeout
        self.return_roi = return_roi

    def get_frames(self):

        frames = {}
        with TLCameraSDK() as sdk:
            available_cameras = sdk.discover_available_cameras()
            if len(available_cameras) < 1:
                print("no cameras detected")

            with sdk.open_camera(available_cameras[0]) as camera:
                camera.exposure_time_us = self.exposure_time  # set exposure to 11 ms
                camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
                camera.image_poll_timeout_ms = self.timeout  # 1 second polling timeout
                camera.roi = self.roi
                # set binning for macropixels
                (camera.binx, camera.biny) = self.bins

                if camera.gain_range.max > 0:
                    db_gain = self.gain
                    gain_index = camera.convert_decibels_to_gain(db_gain)
                    camera.gain = gain_index
                    # print(f"Gain: {camera.convert_gain_to_decibels(camera.gain)}")
                    # print(camera.gain_range, camera.gain)

                if self.return_roi:
                    print("The real camera ROI is: {}".format(camera.roi))
                camera.arm(2)

                camera.issue_software_trigger()

                frame = camera.get_pending_frame_or_null()
                buffer_copy = np.copy(frame.image_buffer)
                frame = buffer_copy


                # for i in range(self.num_of_frames):
                #     frame = camera.get_pending_frame_or_null()
                #     if frame is not None:

                #         image_buffer_copy = np.copy(frame.image_buffer)
                #         frames[frame.frame_count] = image_buffer_copy
                #     else:
                #         print("timeout reached during polling, program exiting...")
                #         break

                camera.disarm()
            return frame