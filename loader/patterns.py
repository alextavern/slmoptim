import numpy as np
import cv2
from scipy import signal
from scipy.linalg import hadamard
from LightPipes import *
from aotools.functions import zernike_noll
from diffractio import degrees, mm, nm, um
from diffractio.scalar_sources_XY import Scalar_source_XY
import random

"""
A class that creates various patterns to be uploaded to an SLM:
1/ a simple mirror (for slm "deactivation")
2/ a bi-mirror (for popoff calibration)
3/ a diffraction grating (for a simple calibration)
4/ a mirror + diffraction grating (for the thorlabs calibration)
5/ a series of methods that create a hadamard vector pattern - this one has become useless because of the HadamardPatternLoader class
   (to be removed)

Some additional methods are also include to adjust a given pattern to the SLM screen:
a/ enlarge_pattern, it only magnifies the pattern
b/ pattern_to_SLM, adds in a center of the SLM screen
"""

class PatternsBacic:

    def __init__(self, res_x, res_y, grayphase=112):
        """
        constructs the pattern class which permits to generate a bunch of patterns ready to upload to
        a SLM. Here, each method generates one pattern mask to be directly used with the SLM.
        It needs the resolution of the SLM screen and its calibration grayscale value. 

        Parameters
        ----------
        res_x
        res_y
        grayphase
        """
        self.res_x = res_x
        self.res_y = res_y
        self.grayphase = grayphase
        x = np.linspace(0, res_x, res_x)
        y = np.linspace(0, res_y, res_y)
        self.X, self.Y = np.meshgrid(x, y)

    def mirror(self, gray=0):
        """
        creates a 2d mask with a constant value for the grayscale on the SLM

        Parameters
        ----------
        gray: int

        Returns
        -------
        pattern: 2d array

        """
        # create a 2d array
        rows = self.res_x
        cols = self.res_y

        # make sure that the image is composed by 8bit integers between 0 and 255
        pattern = np.array([[gray for _ in range(rows)] for _ in range(cols)]).astype('uint8')
        # pattern = np.full(shape=(s, self.resY),fill_value=gray).astype('uint8')

        return pattern

    def bi_mirror(self, gray1=0, gray2=100, a=2):
        """
        creates a two-part 2d mask. One part has constant value for the grayscale on the SLM
        while the second part has a different one

        Parameters
        ----------
        gray1: int
        gray2: int
        a: int

        Returns
        -------
        pattern: 2d array
        """
        # create a 2d array
        rows = self.res_x
        cols = self.res_y
        # make sure that the image is composed by 8bit integers between 0 and 255
        pattern = np.full(shape=(cols, rows), fill_value=0).astype('uint8')

        # replace half of the image with a constant value and the other half with a different constant value
        pattern[:cols, : int(rows / a)] = int(gray1)
        pattern[:cols, int(rows / a):] = int(gray2)

        return pattern.astype('uint8')
    
    def grad_mirror(self, axis=0):
        """
        creates a gradient mirror. This can be useful in order to filter out
        diffraction spots from the slm pixels. Another popoff suggestion. To implement experimentaly.

        Parameters
        ----------
        axis: int

        Returns
        -------
        pattern: 2d array
        """
        gray = self.grayphase
        x = np.linspace(0, gray, 800)
        y = np.linspace(0, gray, 600)
        X, Y = np.meshgrid(x, y)
        
        if axis == 0: pattern = X
        else: pattern = Y

        return pattern.astype('uint8')

    def grating(self, sq_amp=255, sq_period=100):
        """
        creates a square grating with a given contrast and period
        Parameters
        ----------
        sq_amp: int
        sq_period: int

        Returns
        -------
        pattern: 2d array
        """

        # make sure that the image is composed by 8bit integers between 0 and 255
        pattern = np.round((sq_amp * (0.5 + 0.5 * signal.square(2 * np.pi * self.X / sq_period))).astype('uint8'))

        return pattern.astype('uint8')

    def mirror_grating(self, sq_amp=255, sq_period=100, a=2, gray=100):
        """
        creates a two-part mask. One part is a given grayscale mirror and the second part is a square grating with
        given contrast and period
        Parameters
        ----------
        sq_amp: int
        sq_period: int
        a: int
        gray: int

        Returns
        -------
        pattern: 2d array

        """
        # make sure that the image is composed by 8bit integers between 0 and 255
        pattern = np.round((sq_amp * (0.5 + 0.5 * signal.square(2 * np.pi * self.X / sq_period))).astype('uint8'))
    
        # replace half of the image with a constant value
        pattern[:self.res_y, : int(self.res_x / a)] = gray

        return pattern.astype('uint8')

    # @staticmethod
    # def _get_hadamard_basis(dim):
    #     """
    #     calculates the outer product of all combination of the rows of a hadamard matrix of a given order to
    #     generate 2d patterns that constitute a hadamard basis.
    #     Parameters
    #     ----------
    #     dim: the dimensions of each basis vector dim x dim (int)

    #     Returns
    #     -------
    #     matrices: all the 2d patterns (array)

    #     """
        
    #     order = int((np.log2(dim)))

    #     h = hadamard(2 ** order)
    #     matrices = [np.outer(h[i], h[j]) for i in range(0, len(h)) for j in range(0, len(h))]
    #     return matrices

    @staticmethod
    def _get_hadamard_vector(order, i, j):
        """
        calculates the (i,j)th vector of a hadamard basis of a given order
        Parameters
        ----------
        order: int
        i: int
        j: int

        Returns
        -------
        2d array

        """
        h = hadamard(2 ** order)
        matrix = np.outer(h[i], h[j])
        return matrix

    def _hadamard_int2phase(self, vector):
        """
        replaces the elements of an hadamard vector (-1, 1) with the useful slm values (0, pi)
        one needs to know the grayscale value of the slm that gives a phase shift of pi
        Parameters
        ----------
        vector: 2d array

        Returns
        -------
        vector: 2d array

        """
        vector[vector == -1] = 0     # phi = 0
        vector[vector == 1] = self.grayphase / 2  # phi = pi
        return vector

    def hadamard_pattern(self, dim, hadamard_vector_idx, n=1, gray=0):
        """
        creates a hadamard vector and puts it in the middle of the slm screen
        Parameters
        ----------
        dim: hadamard matrix dimension dim x dim (int)
        hadamard_vector_idx: input of the index of the hadamard vector (tuple with 2 int)
        n: hadamard vector magnification factor
        gray: grayscale level of the unaffected slm screen (int)

        Returns
        -------
        pattern: 2d array

        """
        # create a 2d array
        rows = self.res_x
        cols = self.res_y
        # make sure that the image is composed by 8bit integers between 0 and 255
        pattern = np.full(shape=(cols, rows), fill_value=gray).astype('uint8')

        # get a 2d hadamard vector
        order = int((np.log2(dim)))

        i, j = hadamard_vector_idx[0], hadamard_vector_idx[1]
        hadamard_vector = self._get_hadamard_vector(order, i, j)
        # enlarge vector
        hadamard_vector = self._enlarge_pattern(hadamard_vector, n)
        # replace values to grayscale-phase values
        hadamard_vector = self._hadamard_int2phase(hadamard_vector)

        # put it in the middle of the slm screen
        # first calculate offsets from the image center
        subpattern_dim = hadamard_vector.shape
        offset_x = int(rows / 2 - subpattern_dim[0] / 2)
        offset_y = int(cols / 2 - subpattern_dim[1] / 2)
        # and then add the vector in the center of the initialized pattern
        pattern[offset_y:offset_y + subpattern_dim[0], offset_x:offset_x + subpattern_dim[1]] = hadamard_vector

        return hadamard_vector, pattern.astype('uint8')

    def pattern_to_SLM(self, vector, n=1, gray=0, off=(0, 0)):
        """
        puts an enlarged vector in the middle of the slm screen
        
        Parameters
        ----------
        vector: input vector
        n: vector magnification factor
        gray: grayscale level of the unaffected slm screen (int)

        Returns
        -------
        pattern: 2d array

        """
        # create a 2d array
        rows = self.res_x
        cols = self.res_y
        # make sure that the image is composed by 8bit integers between 0 and 255
        pattern = np.full(shape=(cols, rows), fill_value=gray).astype('uint8')

        # enlarge vector by "macropixeling"
        vector = self._enlarge_pattern2(vector, n)
        # replace values to grayscale-phase values
        # vector = self._hadamard_int2phase(vector)

        # put it in the middle of the slm screen
        # first calculate offsets from the image center
        offx, offy = off
        subpattern_dim = vector.shape
        offset_x = int(rows / 2 - subpattern_dim[0] / 2) + offx
        offset_y = int(cols / 2 - subpattern_dim[1] / 2) + offy
        # and then add the vector in the center of the initialized pattern
        pattern[offset_y:offset_y + subpattern_dim[0], offset_x:offset_x + subpattern_dim[1]] = vector

        return pattern.astype('uint8')
    
    def random(self, dim):

        phase_list = np.arange(0, 2 * self.grayphase + 1, self.grayphase / 4)
        pattern = np.array([[random.choice(phase_list) for i in range(dim)] for j in range(dim)])
        return pattern.astype('uint8')


    # def add_subpattern(self, subpattern, gray=0):
    #     """_summary_

    #     Args:
    #         subpattern (_type_): _description_
    #         gray (int, optional): _description_. Defaults to 0.

    #     Returns:
    #         _type_: _description_
    #     """
    #     # create a 2d array
    #     rows = self.res_x
    #     cols = self.res_y
    #     # make sure that the image is composed by 8bit integers between 0 and 255
    #     pattern = np.full(shape=(cols, rows), fill_value=gray).astype('uint8')
        
    #     # put it in the middle of the slm screen
    #     # first calculate offsets from the image center
    #     subpattern_dim = subpattern.shape
    #     offset_x = int(rows / 2 - subpattern_dim[0] / 2)
    #     offset_y = int(cols / 2 - subpattern_dim[1] / 2)
    #     # and then add the vector in the center of the initialized pattern
    #     pattern[offset_y:offset_y + subpattern_dim[0], offset_x:offset_x + subpattern_dim[1]] = subpattern

    #     return pattern.astype('uint8')
        
    @staticmethod
    def _enlarge_pattern(matrix, n):
        """
        It takes as input a 2d matrix and augments its dimensions by 2^(n-1) by conserving the same pattern
        To be removed. The 2^(n-1) factor is not well adapted as it leaed to underfilling the useful slm screen.
        Parameters
        ----------
        matrix: input 2d array
        n: magnification factor

        Returns
        -------
        matrix: enlarged 2d array
        """

        if n < 1:
            raise Exception("sorry, magnification factor should not be zero")
        for i in range(1, n):
            new_dim = matrix.shape[0] * 2
            matrix = np.stack((matrix, matrix), axis=1).flatten().reshape(new_dim, int(new_dim / 2))
            matrix = np.dstack((matrix, matrix)).flatten().reshape(new_dim, new_dim)

        return matrix
    
    @staticmethod
    def _enlarge_pattern2(matrix, n):
        """
        a better and simpler version of enlarge_pattern where any magnification can be done
        ----------
        matrix: input 2d array
        n: magnification factor

        Returns
        -------
        matrix: enlarged 2d array
        """

        if n < 1:
            raise Exception("sorry, magnification factor should not be zero")
        matrix = np.repeat(np.repeat(matrix, n, axis=1), n, axis=0)
        
        return matrix

    def correct_aberrations(self, correction, pattern, alpha=0.5):
        """ A method to superpose a correction pattern to any mask to be uploaded on the SLM
        Parameters
        ----------
        correction : 2-d array, 
            the correction pattern
        pattern : 2-d array
        alpha : float,
            the transparency-mixing parameter, by default 0.5

        Returns
        -------
        2-d array
            the final "corrected" pattern
        """
        
        # blend images
        beta = (1.0 - alpha)
        pattern = cv2.addWeighted(pattern, alpha, correction, beta, 0.0)
        # pattern = pattern + corr_patt2
        
        return pattern.astype('uint8')
 

""" A series of "pattern-loader" classes

    BasePatternGenerator: the basis class that provides methods to be inherited

    a/ OnePixelPatternGenerator
    b/ RandomPatternGenerator
    c/ HadamardPatternGenerator
    d/ GaussPatternGenerator
    e/ LaguerrePatternGenerator
    f/ ZernikePatternGenerator
    g/ PlaneWavePatternGenerator
    e/ ... 
"""

class BasePatternGenerator:
    def __init__(self, num_of_segments, num_of_patterns):

        self.N = num_of_patterns
        self.M = num_of_segments
        self.disk_diameter = int(num_of_segments ** 0.5)
        self.radius = self.disk_diameter // 2
    
    def _get_disk_mask(self):

        res = (self.disk_diameter, self.disk_diameter)
        
        mask_center = [res[0] // 2,res[1] // 2]
        X, Y = np.meshgrid(np.arange(res[0]), np.arange(res[1]))

        # We generate a mask representing the disk we want to intensity to be concentrated in
        mask = (X - mask_center[0]) ** 2 + (Y - mask_center[1]) ** 2 < self.radius ** 2
        
        return mask
    
    @staticmethod
    def _normalize(pattern, vmax=255):
        pattern = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern)) * vmax
        return pattern
    
    def _get_phase(self, wave, calib_px):
        arg = np.angle(wave, deg=False)
        arg2pi = arg + np.pi
        arg2SLM = arg2pi * calib_px / (2 * np.pi)
        arg2SLM = arg

        return arg2SLM
    
    def __getitem__(self, idx):
        pattern = self.patterns[idx]    
        return pattern
    
    def __len__(self):
        return len(self.patterns)
    
    
    
class OnePixelPatternGenerator(BasePatternGenerator):
    
    def __init__(self, num_of_segments=256):
        super().__init__(num_of_segments, num_of_patterns=None)
    
        self.random_idx = self._get_random_pixels()
        self.patterns = self._create_patterns()
    
    def _get_random_pixels(self):
        """ creates all indices of a 2d matrix at a random order
            in order to later sample randomly the pixels of a given mask
        """
        disk = self._get_disk_mask()
        # this will be a list of tuples
        indices = []
        for i in np.arange(self.disk_diameter):
            for j in np.arange(self.disk_diameter):
                if disk[i, j]:
                    indices.append((i, j)) # append a tuple to list

        # to array        
        indices = np.array(indices)

        # randomize
        rng = np.random.default_rng()
        rng.shuffle(indices)

        return indices
    
    def _create_patterns(self):
        """ creates a series of one-pixel 2d pattern by using the random indices from 
            _get_random_pixels
        """
        gray = 0
        phi = 1
        patterns = []
        indices = []
        masks = []
        for i, j in self.random_idx:
            mask = np.zeros(self.disk_diameter ** 2, dtype=bool )
            new_dim = int(self.disk_diameter)
            mask = mask.reshape(new_dim, new_dim)
            
            zero_pattern = np.array([[gray for _ in range(self.disk_diameter)] for _ in range(self.disk_diameter)]).astype('uint8')
            temp = zero_pattern
            temp[i, j] = phi
            mask[i, j] = 1
            patterns.append(temp)
            indices.append((i, j))
            masks.append(mask)
            
        return masks
    

class RandomPatternGenerator(BasePatternGenerator):

    def __init__(self, num_of_segments, num_of_patterns, phase_range):
        super().__init__(num_of_segments, num_of_patterns)

        self.phase_range = phase_range
        self.patterns = self._create_patterns()

    def _random_partition(self):
        """
        """
        mask = np.zeros(self.M, dtype=bool)
        mask[:int((self.M) / 2)] = 1
        np.random.shuffle(mask)
        
        new_dim = int(self.M ** 0.5)
        mask = mask.reshape(new_dim, new_dim)
        
        disk_mask = self._get_disk_mask()
        
        return mask * disk_mask
    
    def _create_pattern(self):
        gray=0
        dim = int(self.M ** 0.5)
        pattern = np.array([[gray for _ in range(dim)] for _ in range(dim)]).astype('uint8')
        mask = self._random_partition()
        pattern[mask] = 1
        return mask, pattern

    def _create_patterns(self):
        patterns = []
        masks = []
        for i in range(self.N):
            mask, pattern = self._create_pattern()
            patterns.append(pattern)
            masks.append(mask)
        return masks
    

class HadamardPatternGenerator(BasePatternGenerator):
    
    def __init__(self, num_of_segments, calib_px):
            super().__init__(num_of_segments, num_of_patterns=None)

            self.calib_px = calib_px
            self.patterns = self._create_patterns()

    def __getitem__(self, idx):
        item = self.patterns[idx]
        item = self._int2phase(item)        
        return item.astype('uint8')
    
    
    def _int2phase(self, vector):
        """
        replaces the elements of an hadamard vector (-1, 1) with the useful slm values (0, pi)
        one needs to know the grayscale value of the slm that gives a phase shift of pi
        Parameters
        ----------
        vector: 2d array

        Returns
        -------
        vector: 2d array

        """
        vector[vector == -1] = 0     # phi = 0
        vector[vector == 1] = self.calib_px / 2  # phi = pi
        return vector
    
    def _create_patterns(self):
        """
        calculates the outer product of all combination of the rows of a hadamard matrix of a given order to
        generate 2d patterns that constitute a hadamard basis.
        Parameters
        ----------
        dim: the dimensions of each basis vector dim x dim (int)

        Returns
        -------
        patterns: all the 2d patterns (array)

        """
        dim = int(self.M ** 0.5)
        order = int((np.log2(dim)))

        h = hadamard(2 ** order)
        patterns = [np.outer(h[i], h[j]) for i in range(0, len(h)) for j in range(0, len(h))]
        
        return patterns
        

class GaussPatternGenerator(BasePatternGenerator):
    
    def __init__(self, 
                 num_of_segments=256, 
                 num_of_patterns=16, 
                 LG=False,
                 w0=3e-3, 
                 wavelength=632e-9, 
                 size=15e-3,
                 phase=False,
                 calib_px=112):
        
        super().__init__(num_of_segments, 
                         num_of_patterns)

        self.w0 = w0 # waist
        self.wavelength = wavelength # wavelenght
        self.size = size 
        self.LG = LG # Hermite-Gauss or Laguerre Gauss
        self.phase = phase
        self.calib_px = calib_px
        
        self.patterns = self._create_patterns()

    def _create_sorted_indices(self):
        """ Create a a list of sorted 2d indices in order to generate
            vectors with increased spatial frequency.

        Returns
        -------
        a list of lists
        """
        indices = []
        for n in range(self.N):
            for m in range(self.N):
                indices.append([n, m])
        return sorted(indices, key=sum)                                               
        
    def _create_patterns(self):
        """ Uses LightPipe library to create HG or LG patterns

        Returns
        -------
            a list with all created patterns
        """
        disk = self._get_disk_mask()

        patterns = []
        
        F = Begin(self.size, self.wavelength, int(self.M ** 0.5))
        indices = self._create_sorted_indices()
        for n, m in indices:
                F = GaussBeam(F, self.w0, LG=self.LG, n=m, m=n)
        
                if self.phase:
                    pattern = Phase(F)
                    pattern = pattern * self.calib_px / (np.pi)
                else:
                    pattern = Intensity(0, F)
#                     pattern = self._normalize(pattern, 112)
        
                patterns.append(pattern)
                
        return patterns
    

class LaguerrePatternGenerator(BasePatternGenerator):
    
    def __init__(self, 
                num_of_segments=256, 
                num_of_patterns=16,
                waist=100,
                phase=True,
                slm_calibration_px=112):    
                
        super().__init__(num_of_segments, 
                        num_of_patterns)

        self.phase = phase
        self.w0 = waist * um
        self.patterns = self._create_patterns()

    def _create_patterns(self):
        """ Uses diffractio package """
        
        patterns = []
        
        x0 = np.linspace(-1 * mm, 1 * mm, int(self.M ** 0.5))
        y0 = np.linspace(-1 * mm, 1 * mm, int(self.M ** 0.5))
        wavelength = 0.6238 * um
        lg = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        
        for p in range(int(self.N ** 0.5) + 1):
            for l in range(int(self.N ** 0.5) + 1):

                lg.laguerre_beam(A=1,
                                 n=p,
                                 l=l,
                                 r0=(0 * um, 0 * um),
                                 w0=self.w0,
                                 z0=0,
                                 z=0.01 * um)
                
                patterns.append(abs(lg.u))

        return patterns
        

class ZernikePatternGenerator(BasePatternGenerator):
    def __init__(self, 
                num_of_segments=256, 
                num_of_patterns=16,
                phase=True,
                slm_calibration_px=112):
                
        super().__init__(num_of_segments, 
                        num_of_patterns)

        self.phase = phase
        self.patterns = self._create_patterns()

    def _create_patterns(self):
        """ Uses AOtools
        """

        # disk = self._get_disk_mask()
        patterns = []

        for idx in range(1, self.N + 2):
            pattern = zernike_noll(idx, int(self.M ** 0.5))
            if self.phase:
                pattern = self._get_phase(pattern, 112)
            else:
#                 pattern = self._normalize(pattern, 112)
                pass

            patterns.append(pattern)

        return patterns


class PlaneWaveGenerator(BasePatternGenerator):
    
    def __init__(self, num_of_segments, calib_px=112, krange=(1, 20, 1), phistep=20):
            super().__init__(num_of_segments, num_of_patterns=None)

            self.calib_px = calib_px
            
            self.degrees = np.pi / 180

            x0 = np.linspace(0, self.M, self.M)
            y0 = np.linspace(0, self.M, self.M)
            self.X, self.Y = np.meshgrid(x0, y0)

            self.krange1, self.krange2, self.kstep = krange
            self.phistep = phistep
                 
            self.patterns = self._create_patterns()

    
    def _get_wave(self, k, r=1, theta=1 , phi=90 , z0=0):
        """ Generates a 2d plane wave using spherical coordinates.

        Parameters:
            k (float): wavenumber
            r (float): maximum amplitude
            theta (float): angle in radians
            phi (float): angle in radians
            z0 (float): constant value for phase shift

        Returns:
            wave (complex): complex field
        """
    #     k = 2 * pi / wavelength
        theta = theta * self.degrees
        phi = phi * self.degrees
        wave = r * np.exp(1.j * k *
                         (self.X * np.sin(theta) * np.cos(phi) +
                          self.Y * np.sin(theta) * np.sin(phi) + z0 * np.cos(theta)))
        return wave

    def _create_patterns(self):
        patterns = []
        
        k1 = self.krange1
        k2 = self.krange2
        step = self.kstep
                 
        for k in range(k1, k2, step):
            for angle in np.arange(0, 360, self.phistep):
                pattern = self._get_wave(k=k, phi=angle)
                pattern = self._get_phase(pattern, self.calib_px)
                patterns.append(pattern)
        return patterns
    
    
    
def superpose(loader, coeffs):
    """ A simple function that creates a linear combination of given vectors
        It is used with coefficient optimizer.

    Args:
        loader: pattern loader object
        coeffs: a list with coefficients

    Returns:
        mask: a 2d array
    """

    shape = loader[1].shape
    
    # create array with vectors
    vectorList = []
    for vector in loader:
        vectorList.append(vector)
    vectorArray = np.array(vectorList)

    # superpose
    mask = np.zeros(shape)
    for idx in range(len(coeffs)):
        mask += vectorArray[idx] * coeffs[idx]

    # convert to complex mask
    mask = np.exp(1j * mask)
    # calculate arg
    # mask = _phase2SLM(mask)

    return mask
