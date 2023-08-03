import threading
from .patterns import Pattern
from tqdm.auto import tqdm
import time

class SlmUploadPatternsThread(threading.Thread):
    
    def __init__(self, slm, download_frame_event, upload_pattern_event, stop_all_event, calib_px=112, order=4, mag=5):
        """ This thread is designed to run in paraller with another thread that download frames from a camera. In particular
        this class uploads a hadamard vector on the SLM, sets a thread event that triggers the other thread to download a frame. 
        Then, it waits for a thread event to be set from the camera thread to upload the next hadamard vector. 
        Finally, once all patterns are uploaded to the SLM a thread event is set that stops and closes all threads. 

        It needs an SLM object along with the SLM calibration constant and the hadamard basis parameters.

        Parameters
        ----------
        slm : class object
            slmpy - popoff
        download_frame_event : thread event
        upload_pattern_event : thread event
        stop_all_event : thread event
        calib_px : int
            the grayscale value that corresponds to a 2pi modulation, by default 112
        order : int, optional
            hadamard matrix order, by default 4
        mag : int, optional
            magnification factor of had vector in the SLM active area
            indirectly it set the SLM macropixel size, by default 5
        """
        super(SlmUploadPatternsThread, self).__init__()

        self.slm = slm
        self.resX, self.resY = slm.getSize()
        self.calib_px = calib_px
        self.order = order
        self.mag = mag
        self.length = 2 ** order
        
        self.slm_patterns = Pattern(self.resX, self.resY, calib_px)
        self.basis = self.slm_patterns._get_hadamard_basis(self.order)
        
        # the grayscale level that yields a pi phase modulation on the slm
        pi = int(calib_px / 2)
        # for the 4-step phase shift interferometry
        self.four_phases = [0, pi / 2, pi, 3 * pi / 2]
        
        # event triggers for syncing
        self.download = download_frame_event
        self.upload = upload_pattern_event
        self.stop = stop_all_event

        self.patterns = []
        
    def run(self):

            # loop through each 2d vector of the hadamard basis - basis is already generated here
            self.upload.set()
            for vector in tqdm(self.basis, desc='Uploading Hadamard patterns', leave=True):
                # and for each vector load the four reference phases
                for phase in self.four_phases:
                    _, pattern = self.slm_patterns.hadamard_pattern_bis(vector, n=self.mag, gray=phase)
                    self.upload.wait()
                    self.slm.updateArray(pattern) # load each vector to slm

                    # bug? :the first frame/pattern pair is badly synced - I haven't figured it out..
                    # it works by allowing it to sleep a bit:
                    time.sleep(.1) 
                    self.patterns.append(pattern)
                    # send flag to other threads here
                    self.download.set()
                    self.upload.clear()

            return  self.stop.set()
        