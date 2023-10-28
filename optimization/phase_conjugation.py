import numpy as np
from ..patternSLM import patterns as pt

class Target:
    def __init__(self, shape) -> None:
        self.shape = shape
        
    def square(self, focus_shape, offset_x=0, offset_y=0, intensity=1000):
        # create a focus point
        target_frame = np.full(shape=self.shape, fill_value=0).astype('float64')
        target_focus = np.full(shape=focus_shape, fill_value=intensity).astype('float64')

        # put it in the middle of the slm screen
        # first calculate offsets from the image center
        subpattern_dim = target_focus.shape
        center_x = int(self.shape[1] / 2 - subpattern_dim[0] / 2) + offset_x
        center_y = int(self.shape[1] / 2 - subpattern_dim[1] / 2) - offset_y

        # and then add the vector in the center of the initialized pattern
        target_frame[center_y:center_y + subpattern_dim[0], center_x:center_x + subpattern_dim[1]] = target_focus
        
        return target_frame

    def gauss(self, 
              order=0,
              num=1, 
              w0=4e-4, 
              wavelength=632e-9, 
              size=15e-3, 
              slm_calibration_px=112):
                 
        N = int(self.shape[0])
        gauss = pt.GaussPatternGenerator(N=N, num=num, LG=True,
                                        w0=w0, 
                                        wavelength=wavelength, 
                                        size=size, 
                                        slm_calibration_px=slm_calibration_px) 
        amp, phase = gauss[order]
        complex_field = amp * np.exp(1j * phase)
        
        return complex_field
    
    
class InverseLight:
    
    def __init__(self, target, tm, calib_px=56, slm_macropixel=4):

        self.target = target
        self.shape = target.shape
        self.tm = tm
        self.calib_px = calib_px
        
        self.tm_shape = tm.shape
        self.phase_mask_shape = (int(np.sqrt(self.tm_shape[1])), int(np.sqrt(self.tm_shape[1])))
        self.phase_mask_mag = slm_macropixel
        
    def _conj_trans(self):
        """ Calculates conjugate transpose matrix of input transmission matrix

        Returns:
            inv_operator: _description_
        """
        self.tm_T_star = self.tm.transpose().conjugate()
        return self.tm_T_star
    
    def _inverse(self):
        self.tm_inv = np.linalg.inv(self.tm)
        return self.tm_inv

        
    def calc_inv_operator(self):
        inv_operator = self.tm@self.tm_T_star
        return inv_operator

    
    def inverse_prop(self, conj=True):
        """ Calculates the inverse light propagation and produces a phase mask.
            User must define inversion method: phase conjugation or matrix inversion.
        """
        
        # first flatten frame
        target_frame_flattened = []
        for iy, ix in np.ndindex(self.shape):
            target_frame_flattened.append(self.target[iy, ix])

        target_frame_flattened = np.array(target_frame_flattened)

        # apply inversion
        if conj:
            # tm_T_star = self._conj_trans()
            tm_inv = self._conj_trans()
        else:
            tm_inv = self._inverse()
            
        # inverse = np.dot(tm_T_star, target_frame_flattened.T)
        inverse = tm_inv@target_frame_flattened.T

        # get phase (in -pi to pi)
        arg = np.angle(inverse, deg=False)
        # scale phase between 0 and 2pi
        arg2pi = (arg + 2 * np.pi) % (2 * np.pi)
        # normalize to SLM 2pi calibration value
        arg2SLM = arg2pi * self.calib_px / (2 * np.pi) 

        # and unflatten to get a 2d SLM pattern
        phase_mask = np.full(shape=self.phase_mask_shape, fill_value=0).astype('float64')
        for idx, ij in enumerate(np.ndindex(self.phase_mask_shape[0], self.phase_mask_shape[1])):
            phase_mask[ij[0], ij[1]] = arg2SLM[idx]
            
        # enlarge pattern for the SLM macropixels
        self.phase_mask_enlarged = pt.Pattern._enlarge_pattern(phase_mask, self.phase_mask_mag)
        
        return self.phase_mask_enlarged