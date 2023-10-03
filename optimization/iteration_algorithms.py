import numpy as np
import time
from tqdm.auto import tqdm
from slmOptim.patternSLM import patterns as pt
from slmOptim.optimization import phase_conjugation


class IterationAlgos():
    """ This is the base class to be used for various iteration based algorithms for optimization
        including Stepwise Sequential (SSA), Continuous Sequential (CSA), Random Partition (RPA) 
        and Hadamard Partition (HPA) algorithms. 
    """
    
    def __init__(self, 
                 slm, 
                 camera,
                 pattern_loader,
                 total_iterations=1,
                 slm_resolution=(800, 600),
                 slm_segments=256,
                 slm_macropixel=5, 
                 slm_calibration_pixel=112,
                 phase_steps=8,
                 remote=True):
        
        self.slm = slm
        self.camera = camera
        
        self.total_iterations = total_iterations

        # slm settings
        self.N = int(slm_segments ** 0.5)
        self.slm_macropixel = slm_macropixel
        self.calib_px = slm_calibration_pixel
        
        self.pattern_loader = pattern_loader
                
        self.m = phase_steps
        
        # SLM
        resX, resY = slm_resolution
        self.patternSLM = pt.Pattern(resX, resY)
        
        # is the slm remotely connected to a rasp pi ?
        self.remote = remote
        
     
    def register_callback(self, callback):
        """_summary_

        Parameters
        ----------
        callback
            _description_
        """
        self.callback = callback
    
    def phi_k(self, k):
        return (2 * (k + 1) / self.m) * self.calib_px
    
    def create_pattern(self, k, mask, pattern):
        phi = self.phi_k(k)
        pattern[mask] = phi
        temp = self.patternSLM._enlarge_pattern(pattern, self.slm_macropixel)
        temp = self.patternSLM.add_subpattern(temp)
        return temp

    def upload_pattern(self, pattern, slm_delay=0.1):
        if self.remote:
            self.slm.sendArray(pattern)
        else:
            self.slm.updateArray(pattern)
        time.sleep(slm_delay)
        
    def update_pattern(self, pattern):
        new_pattern = pattern
        return new_pattern
    
    def get_frame(self):
        frame = self.camera.get_pending_frame_or_null()
        image_buffer = np.copy(frame.image_buffer)
        return image_buffer
    
class ContinuousSequential(IterationAlgos):
        
    def run(self):
        gray = 0
        final_pattern = np.array([[gray for _ in range(self.N)] for _ in range(self.N)]).astype('uint8')
        
        cost = []
        counter = 0
        frames = {}

        for iteration in range(0, self.total_iterations):   
             
            # sweep each slm pixel
            for _, mask, _ in tqdm((self.pattern_loader)):
                temp_pattern = final_pattern.copy()
                corr = []
                # sweep phase at each pixel
                for k in np.arange(0, self.m):
                    # create pattern, i.e one pattern for each phase value
                    temp = self.create_pattern(k, mask, temp_pattern)

                    # upload pattern to slm
                    self.upload_pattern(temp)

                    # get interferogram from camera
                    frame = self.get_frame()

                    # calculate correlation here
                    corr_k = self.callback(frame)
                    corr.append(corr_k)

                counter += 1 
                frames[counter] = frame

                # update pattern with max corr
                cost.append(np.max(corr))
                final_pattern[mask] = self.phi_k(np.argmax(corr))

        return final_pattern, cost, frames
    

class StepwiseSequential(IterationAlgos):
    
    def run(self):
        gray = 0
        final_pattern = np.array([[gray for _ in range(self.N)] for _ in range(self.N)]).astype('uint8')
        final_pattern_k = final_pattern.copy()
        
        cost = []
        counter = 0
        frames = {}

        for iteration in range(0, self.total_iterations):   
            temp_pattern = final_pattern.copy() 
            # sweep each slm pixel
            for _, mask, _ in tqdm((self.pattern_loader)):               
                corr = []
                # sweep phase at each pixel
                for k in np.arange(0, self.m):
                    # create pattern, i.e one pattern for each phase value
                    temp = self.create_pattern(k, mask, temp_pattern)
                    # upload pattern to slm
                    self.upload_pattern(temp, 0.1)
                    
                    # get interferogram from camera
                    frame = self.get_frame()

                    # calculate correlation here
                    corr_k = self.callback(frame)
                    corr.append(corr_k)
                
                counter += 1
                frames[counter] = frame
                temp_pattern = final_pattern.copy()
                
                # update pattern with max corr
                cost.append(np.max(corr))
                # pattern[idx[0], idx[1]] = self.phi_k(np.argmax(corr))
                final_pattern_k[mask] = self.phi_k(np.argmax(corr))
            final_pattern = final_pattern_k.copy()

        return final_pattern, cost, frames
    
    class RandomPartition(IterationAlgos):
        
        def run():
            pass
        
    class HadamardPartition(IterationAlgos):
        
        def run():
            pass