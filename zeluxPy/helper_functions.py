from scipy import ndimage
from .polling import Cam


def get_interferogram(roi=(0, 0, 1440, 1080), num_of_frames=1, exposure_time=5000, gain=1, timeout=1000):
    camera = Cam(roi, num_of_frames, exposure_time, gain, timeout)
    frames = camera.get_frames()
    return frames


def rotate_frame(frame, angle):
    frame_rot = ndimage.rotate(frame, angle)
    return frame_rot
