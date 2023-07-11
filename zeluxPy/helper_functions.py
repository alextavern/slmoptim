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


class CameraConfigThread(threading.Thread):

    def __init__(self, trigger_event, stop_event, upload_pattern_event, roi=(0, 0, 1440, 1080), bins=(1, 1), exposure_time=11000, gain=6, timeout=1000):
        super(CameraConfigThread, self).__init__()
        self.roi = roi
        self.bins = bins
        self.gain = float(gain)
        self.exposure_time = exposure_time
        self.timeout = timeout

        self.trigger_event = trigger_event
        self.stop_event = stop_event
        self.upload_pattern_event = upload_pattern_event
        self.frames = None

    def run(self):

        print("Configuring the USB camera...")
        with TLCameraSDK() as sdk:
            available_cameras = sdk.discover_available_cameras()
            if len(available_cameras) < 1:
                print("no cameras detected")
            # Configure the camera settings

            with sdk.open_camera(available_cameras[0]) as camera:
                camera.exposure_time_us = self.exposure_time  # set exposure to 11 ms
                camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
                camera.image_poll_timeout_ms = self.timeout  # 1 second polling timeout
                camera.roi = self.roi # set roi
                # set binning for macropixels
                (camera.binx, camera.biny) = self.bins

                if camera.gain_range.max > 0:
                    db_gain = self.gain
                    gain_index = camera.convert_decibels_to_gain(db_gain)
                    camera.gain = gain_index
                    # print(f"Gain: {camera.convert_gain_to_decibels(camera.gain)}")
                    # print(camera.gain_range, camera.gain)

                camera.arm(2)

                camera.issue_software_trigger()

                # hear put a flag to trigger download
                download_thread = FrameAcquisitionThread(camera,
                                                         self.trigger_event,
                                                         self.stop_event,
                                                         self.upload_pattern_event,
                                                         num_of_frames=1)
                download_thread.start()
                download_thread.join()

                camera.disarm()
                camera.dispose()
                self.frames = download_thread.frames


class FrameAcquisitionThread(threading.Thread):

    def __init__(self, camera, trigger_event, stop_event, upload_pattern_event, num_of_frames=1):
        super(FrameAcquisitionThread, self).__init__()
        self.num_of_frames = num_of_frames
        self.camera = camera
        self.trigger_event = trigger_event
        self.stop_event = stop_event
        self.upload_pattern_event = upload_pattern_event
        self.frames = []

#     def stop(self):
#         self._stop_event.set()

    def run(self):

        while not self.stop_event.is_set():
            self.trigger_event.wait()
#             print("Downloading frames from the USB camera...")
            # Download the frames from the camera
            for i in range(self.num_of_frames):
                frame = self.camera.get_pending_frame_or_null()
                if frame is not None:
                    image_buffer_copy = np.copy(frame.image_buffer)
                    self.frames.append(image_buffer_copy)
                else:
                    print("timeout reached during polling, program exiting...")
                    break
            self.upload_pattern_event.set()
#             if self.upload_pattern_event.is_set():
#                 print("upload now bitch!")
            self.trigger_event.clear()