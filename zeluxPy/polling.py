"""
Polling Example

This example shows how to open a camera, adjust some settings, and poll for images. It also shows how 'with' statements
can be used to automatically clean up camera and SDK resources.

"""

import numpy as np
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK


class Cam:

    def __init__(self, roi=(0, 0, 1440, 1080), bins=(1, 1), num_of_frames=10, exposure_time=11000, gain=6, timeout=1000):
        self.roi = roi
        self.bins = bins
        self.num_of_frames = num_of_frames
        self.gain = float(gain)
        self.exposure_time = exposure_time
        self.timeout = timeout

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
                old_roi = camera.roi  # store the current roi

                # print("Exposure time: {}".format(self.exposure_time))
                # print("Timeout: {}".format(self.timeout))

                # camera.roi = (100, 100, 600, 600)
                # set roi to be at origin point (100, 100) with a width & height of 500
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

                camera.arm(2)

                camera.issue_software_trigger()

                for i in range(self.num_of_frames):
                    frame = camera.get_pending_frame_or_null()
                    if frame is not None:
                        # print("frame #{} received!".format(frame.frame_count))

                        # frame.image_buffer  # .../ perform operations using the data from image_buffer

                        #  NOTE: frame.image_buffer is a temporary memory buffer that may be overwritten during
                        #  the next call to get_pending_frame_or_null. The following line makes a deep copy of the
                        #  image data:
                        image_buffer_copy = np.copy(frame.image_buffer)
                        frames[frame.frame_count] = image_buffer_copy
                    else:
                        print("timeout reached during polling, program exiting...")
                        break

                camera.disarm()
                camera.roi = old_roi  # reset the roi back to the original roi
            return frames

        #  Because we are using the 'with' statement context-manager, disposal has been taken care of.

# print("program completed")