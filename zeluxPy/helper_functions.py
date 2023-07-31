from scipy import ndimage
from .polling import Cam
import numpy as np
import threading
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame


def get_interferogram(roi=(0, 0, 1440, 1080), bins=(1,1), num_of_frames=1, exposure_time=5000, gain=1, timeout=1000):
    camera = Cam(roi, bins, num_of_frames, exposure_time, gain, timeout)
    frames = camera.get_frames()
    return frames


def get_frame_binned(roi, bins, num_of_frames=1, exposure_time=5000, gain=1, timeout=1000):

    frame = get_interferogram(roi=roi,
                              bins=(bins, bins),
                              num_of_frames=num_of_frames,
                              exposure_time=exposure_time,
                              gain=gain,
                              timeout=timeout)

    return frame


def rotate_frame(frame, angle):
    frame_rot = ndimage.rotate(frame, angle)
    return frame_rot


class CameraThread(threading.Thread):

    def __init__(self, download_frame_event, upload_pattern_event, stop_all_event, roi=(0, 0, 1440, 1080), bins=(1, 1), exposure_time=11000, gain=6, timeout=1000):
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
        
        self.download = download_frame_event
        self.stop = stop_all_event
        self.upload = upload_pattern_event

        self.frames = None
    
    def run(self):
        """ Inspired by thorlabs sdk example: detect available cameras, set camera properties, arm and trigger. 
        Then open a nested thread to download one frame from the camera once a pattern is uploaded to the SLM
        (the slm thread sets the corresponsind thread event).
        When everything is done, the camera is disarmed and disposed.
        """
        
        with TLCameraSDK() as sdk:
            available_cameras = sdk.discover_available_cameras()
            if len(available_cameras) < 1:
                print("no cameras detected")
            # Configure the camera settings

            with sdk.open_camera(available_cameras[0]) as camera:
                camera.exposure_time_us = self.exposure_time  # set exposure to 11 ms
                camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode (if 0)
                camera.image_poll_timeout_ms = self.timeout  # polling timeout
                camera.roi = self.roi # set roi
                # set binning for camera macropixels
                (camera.binx, camera.biny) = self.bins

                if camera.gain_range.max > 0:
                    db_gain = self.gain
                    gain_index = camera.convert_decibels_to_gain(db_gain)
                    camera.gain = gain_index
                    # print(f"Gain: {camera.convert_gain_to_decibels(camera.gain)}")
                    # print(camera.gain_range, camera.gain)

                camera.arm(2)

                camera.issue_software_trigger()
                
                print("USB camera configured and armed.")

                download_thread = FrameAcquisitionThread(camera,
                                                         self.download, 
                                                         self.upload, 
                                                         self.stop,
                                                         num_of_frames=1)
                download_thread.start()
                download_thread.join()

                camera.disarm()
                camera.dispose()

                self.frames = download_thread.frames


class FrameAcquisitionThread(threading.Thread):
    
    def __init__(self, camera, download_frame_event, upload_pattern_event, stop_all_event, num_of_frames=1):
        """ 
        This is a thread that once a zelux camera is configured, armed and triggered, downloads one (or more) frame from
        the camera. The associated thread events syncronises this thread with the parallel ones. 
        Parameters
        ----------
        camera : class object
            zelux thorlabs SDK
        download_frame_event : thread event
        upload_pattern_event : thread event
        stop_all_event : thread event
        num_of_frames : int, optional
        """
        super(FrameAcquisitionThread, self).__init__()

        self.num_of_frames = num_of_frames
        self.camera = camera

        self.download = download_frame_event
        self.stop = stop_all_event
        self.upload = upload_pattern_event

        self.frames = []
        
    def run(self):
        
        while not self.stop.is_set():
            self.download.wait()
            
            # Download the frames from the camera
            for i in range(self.num_of_frames):
                frame = self.camera.get_pending_frame_or_null()
                if frame is not None:
                    image_buffer_copy = np.copy(frame.image_buffer)
                    self.frames.append(image_buffer_copy)
                else:
                    print("timeout reached during polling, program exiting...")
                    break
            self.download.clear()
            self.upload.set()
