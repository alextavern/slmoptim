import numpy as np
from ..patternSLM import patterns as  pt

class Target:
    def __init__(self, shape) -> None:
        self.shape = shape
        
    def square(self, focus_shape):
        # create a focus point
        target_frame = np.full(shape=self.shape, fill_value=0).astype('float64')
        target_focus = np.full(shape=focus_shape, fill_value=1).astype('float64')

        # put it in the middle of the slm screen
        # first calculate offsets from the image center
        subpattern_dim = target_focus.shape
        offset_x = int(self.shape[1] / 2 - subpattern_dim[0] / 2)
        offset_y = int(self.shape[1] / 2 - subpattern_dim[1] / 2)

        # and then add the vector in the center of the initialized pattern
        target_frame[offset_y:offset_y + subpattern_dim[0], offset_x:offset_x + subpattern_dim[1]] = target_focus
        
        return target_frame

    def some_other_pattern(self):
        pass
    
    
class InverseLight:
    
    def __init__(self, target, tm, calib_px=112, mag=4):

        self.target = target
        self.shape = target.shape
        self.tm = tm
        self.calib_px = calib_px
        
        self.tm_shape = tm.shape
        self.phase_mask_shape = (int(np.sqrt(self.tm_shape[1])), int(np.sqrt(self.tm_shape[1])))
        self.phase_mask_mag = mag
        
    def _conj_trans(self):
        """ Calculates conjugate transpose matrix of input transmission matrix

        Returns:
            _type_: _description_
        """
        tm_T_star = self.tm.transpose().conjugate()
        return tm_T_star
    
    def inverse_prop(self):
        """ Calculates the inverse light propagation and produces a phase mask

        Returns
        -------
        _type_
            _description_
        """
        
        # first flatten frame
        target_frame_flattened = []
        for iy, ix in np.ndindex(self.shape):
            target_frame_flattened.append(self.target[iy, ix])

        target_frame_flattened = np.array(target_frame_flattened)

        # apply inversion
        # tm_T_star = self._conj_trans()
        tm_T_star = self.tm.transpose().conjugate()
        inverse = np.dot(tm_T_star, target_frame_flattened.T)

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
    
        