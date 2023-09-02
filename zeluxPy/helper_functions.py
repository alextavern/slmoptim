from scipy import ndimage
import numpy as np
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
import threading

def get_interferogram(roi=(0, 0, 1440, 1080), bins=(1,1), num_of_frames=1, exposure_time=5000, gain=1, timeout=10000, return_roi=True):
    camera = Cam(roi, bins, num_of_frames, exposure_time, gain, timeout, return_roi)
    frames = camera.get_frames()
    return frames


def get_frame_binned(roi, bins, num_of_frames=1, exposure_time=5000, gain=1, timeout=1000, return_roi=True):

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

        try:
            # if on Windows, use the provided setup script to add the DLLs folder to the PATH
            from windows_setup import configure_path
            configure_path()
        except ImportError:
            configure_path = None

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
                # print(camera.roi)

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

                for i in range(self.num_of_frames):
                    frame = camera.get_pending_frame_or_null()
                    if frame is not None:

                        image_buffer_copy = np.copy(frame.image_buffer)
                        frames[frame.frame_count] = image_buffer_copy
                    else:
                        print("timeout reached during polling, program exiting...")
                        break

                camera.disarm()
            return frames



class CameraThread(threading.Thread):

    def __init__(self, start_all_event, stop_all_event, roi=(0, 0, 1440, 1080), bins=(1, 1), exposure_time=11000, gain=6, timeout=1000):
        """This class opens a thread that run in parallel to the SLM thread (see patternSLM/patterns.py)
        It sets a bunch of thread events that act complementary fo the SLM thread. 
        First, it opens a thread that configures a zelux thorlabs camera. Then, a nested thread downloads frames from the camera.

        Parameters
        ----------
        download_frame_event : thread event
        upload_pattern_event : thead event
        stop_all_event : thread event
        roi : tuple, optional
            set region of intereset of the camer, by default (0, 0, 1440, 1080)
        bins : tuple, optional
            set the camera macropixel size, by default (1, 1)
        exposure_time : int, optional
            set camera exposure time, by default 11000
        gain : int, optional
            set camera gain, by default 6
        timeout : int, optional
            set camera timeout, by default 1000
        """
        super(CameraThread, self).__init__()
        self.roi = roi
        self.bins = bins
        self.gain = float(gain)
        self.exposure_time = exposure_time
        self.timeout = timeout
        
        self.start = start_all_event
        self.stop = stop_all_event

        self.camera = None
    
    def run(self):
        """ Inspired by thorlabs sdk example: detect available cameras, set camera properties, arm and trigger. 
        Then open a nested thread to download one frame from the camera once a pattern is uploaded to the SLM
        (the slm thread sets the corresponsind thread event).
        When everything is done, the camera is disarmed and disposed.
        """
        
        while not self.stop.is_set():
            # create camera instance
            sdk = TLCameraSDK()
            # search for available cameras
            available_cameras = sdk.discover_available_cameras()
            self.camera = sdk.open_camera(available_cameras[0])
            
            # configuration
            self.camera.exposure_time_us = self.exposure_time  # set exposure to 11 ms
            self.camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode (if 0)
            self.camera.image_poll_timeout_ms = self.timeout  # polling timeout
            self.camera.roi = self.roi # set roi
            # set binning for camera macropixels
            (self.camera.binx, self.camera.biny) = self.bins

            if self.camera.gain_range.max > 0:
                db_gain = self.gain
                gain_index = self.camera.convert_decibels_to_gain(db_gain)
                self.camera.gain = gain_index
            # arm and trigger camera
            self.camera.arm(2)
            self.camera.issue_software_trigger()
            self.start.set()
        
        self.camera.disarm()
        self.camera.dispose()
