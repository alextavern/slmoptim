import numpy as np
import time
from tqdm.auto import tqdm
from slmOptim.patternSLM import patterns as pt
from slmOptim.optimization import phase_conjugation




class SimpleOptimize():
    def __init__(self, 
                 slm, 
                 camera,
                 total_iterations=1,
                 slm_segments=256,
                 slm_macropixel=1, 
                 slm_calibration_pixel=112, 
                 corr_path=None, 
                 save_path=None):
                
        self.slm = slm
        self.camera = camera
        
        self.total_iterations = total_iterations


        # slm settings
        self.N = int(slm_segments ** 0.5)
        self.slm_macropixel = slm_macropixel
        self.calib_px = slm_calibration_pixel
        
        # correction pattern
        self.corr_path = corr_path
        
        # save raw data path
        self.save_path = save_path
        
        resX, resY = slm.getSize()
        self.patternSLM = pt.Pattern(resX, resY)
        
        self.m = 8
        
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

    @staticmethod
    def set_target(shape=(104, 104)):
        # define target
        target_shape = shape
        tgt = phase_conjugation.Target(target_shape)
        target = tgt.square((4, 4), offset_x=-20, offset_y=0, intensity=100)
        return target

    def phi_k(self, k):
        return (2 * (k + 1) / self.m) * self.calib_px
   

    def get_random_pixels(self):
        """ creates all indices of a 2d matrix at a random order
            in order to later sample randomly the pixels of a given mask
        """
        # this will be a list of tuples
        indices = []
        for i in np.arange(self.N):
            for j in np.arange(self.N):
                indices.append((i, j)) # append a tuple to list
        # to array        
        indices = np.array(indices)

        # randomize
        rng = np.random.default_rng()
        rng.shuffle(indices)

        return indices
        
    def get_frame(self):
        frame = self.camera.get_pending_frame_or_null()
        frame_saved = np.copy(frame.image_buffer)
        return frame_saved
        
    def create_pattern(self, pattern, idx):
        phi =self.phi_k(k)32
        pattern[idx[0], idx[1]] = phi
        pattern = pattern.astype('uint8')
        temp = self.patternSLM._enlarge_pattern(pattern, self.slm_macropixel)
        temp = self.patternSLM.add_subpattern(temp)
        return temp    
    
    def upload_pattern(self, pattern):
        self.slm.updateArray(pattern)
        time.sleep(.1)

        
    
    @staticmethod
    def cost(X, Y):
        """ takes two 2d arrays, flattens them calculates the 
            corresponding covariance matrix. the correlation coefficient
            is calculated by r = cov(x, y) / sqrt(var(x) * var(y))
        """
        # flatten the input matrices
        x = X.flatten()
        y = Y.flatten()
        # stack them
        stacked = np.stack((x, y), axis=0)

        # calculate the covariance matrix
        covar = np.cov(stacked)
        # use the cov matrix elements to calculate correlation coefficient
        corr_coeff = covar[0, 1] / np.sqrt(covar[0, 0] * covar[1, 1])
        return corr_coeff
    
    def update_pattern(self):
        pass
    
    def run(self):
        # define target
        target_shape = (104, 104)
        tgt = phase_conjugation.Target(target_shape)
        target = tgt.square((4, 4), offset_x=-20, offset_y=0, intensity=100)
        
        # params
        indices = self.get_random_pixels()
            
        gray = 8
        pattern = np.array([[gray for _ in range(self.N)] for _ in range(self.N)]).astype('uint8')

        cost = []
        counter = 0
        frames = {}

        
        for iter in range(0, 1):
            # sweep through slm pixels
            for idx in tqdm((indices)):
                corr = []
                # sweep through pixel phase
                for k in np.arange(0,self.m):
                    # create pattern
                    phi =self.phi_k(k)
                    temp = self.create_pattern(pattern, idx, phi)
                    # pattern[idx[0], idx[1]] = phi
                    # pattern = pattern.astype('uint8')
                    # temp = patternSLM._enlarge_pattern(pattern, self.slm_macropixel)
                    # temp = patternSLM.add_subpattern(temp)

                    # upload pattern to slm
                    self.upload_pattern(temp)

                    # get interferogram from camera
                    image_buffer_copy = self.get_frame()
                                        
                    # calculate correlation here
                    corr_k = self.cost(image_buffer_copy, target)
                    corr.append(corr_k)
                    
                counter += 1 
                frames[counter] = image_buffer_copy

                # update pattern with max corr
                cost.append(np.max(corr))
                pattern[idx[0], idx[1]] = self.phi_k(np.argmax(corr))
            
        return pattern, cost
