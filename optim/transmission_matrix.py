from ..patternSLM import patterns as pt
from ..zeluxPy import helper_functions as cam
from slmPy import slmpy
import threading

"""
"""


class TM:

    def __init__(self, roi=(556, 476, 684 - 1, 604 - 1), bins=8, exposure_time=100, gain=1, timeout=100,
                 order=4, mag=5,
                 monitor=1):
        """

        Parameters
        ----------
        roi
        bins
        exposure_time
        gain
        timeout
        order
        mag
        monitor
        """
        # camera settings
        self.roi = roi
        self.bins = bins
        self.exposure_time = exposure_time
        self.gain = gain
        self.timeout = timeout
        # hadamard settings
        self.order = order
        self.mag = mag
        # slm monitor setting
        self.slm = slmpy.SLMdisplay(monitor=monitor)

    def get_tm(self):
        """

        Returns
        -------

        """
        # Create flag events
        download_frame_event = threading.Event()
        upload_pattern_event = threading.Event()
        stop_all_event = threading.Event()

        # Create the threads
        download_thread = cam.CameraThread(download_frame_event,
                                           upload_pattern_event,
                                           stop_all_event,
                                           roi=self.roi,
                                           bins=(self.bins, self.bins),
                                           exposure_time=self.exposure_time,
                                           gain=self.gain,
                                           timeout=self.timeout)

        upload_thread = pt.SlmUploadPatternsThread(self.slm,
                                                   download_frame_event,
                                                   upload_pattern_event,
                                                   stop_all_event,
                                                   order=self.order,
                                                   mag=self.mag)

        # Start the threads
        download_thread.start()
        upload_thread.start()

        # Wait for the threads to finish
        download_thread.join()
        upload_thread.join()

        # The main thread will wait for both threads to finish before continuing
        print("Program execution completed.")
        self.slm.close()
        return download_thread.frames

    @staticmethod
    def four_phases_method(I1, I2, I3, I4):
        return complex((I1 - I4) / 4, (I3 - I2) / 4)

    def calc_tm(self):
        return

    def plot(self):
        return

    def invert(self):
        return
