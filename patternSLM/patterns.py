import numpy as np
import cv2
from scipy import signal
from scipy.linalg import hadamard
from LightPipes import *
import random

"""
A class that creates various patterns to be uploaded to an SLM:
1/ a simple mirror 
2/ a bi-mirror (for popoff calibration)
3/ a diffraction grating (for a simple calibration)
4/ a mirror + diffraction grating (for the thorlabs calibration)
5/ a series of methods that create a hadamard vector pattern
"""

class Pattern:

    def __init__(self, res_x, res_y, grayphase=112):
        """
        constructs the pattern class which permits to generate a bunch of patterns ready to upload to
        a SLM. It needs the resolution of the SLM screen and its calibration grayscale value. 
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
        creates a gradient mirror

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

    @staticmethod
    def _get_hadamard_basis(dim):
        """
        calculates the outer product of all combination of the rows of a hadamard matrix of a given order to
        generate 2d patterns that constitute a hadamard basis.
        Parameters
        ----------
        dim: the dimensions of each basis vector dim x dim (int)

        Returns
        -------
        matrices: all the 2d patterns (array)

        """
        
        order = int((np.log2(dim)))

        h = hadamard(2 ** order)
        matrices = [np.outer(h[i], h[j]) for i in range(0, len(h)) for j in range(0, len(h))]
        return matrices

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

    def enlarge_to_SLM(self, vector, n=1, gray=0):
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
        vector = self._enlarge_pattern(vector, n)
        # replace values to grayscale-phase values
        vector = self._hadamard_int2phase(vector)

        # put it in the middle of the slm screen
        # first calculate offsets from the image center
        subpattern_dim = vector.shape
        offset_x = int(rows / 2 - subpattern_dim[0] / 2)
        offset_y = int(cols / 2 - subpattern_dim[1] / 2)
        # and then add the vector in the center of the initialized pattern
        pattern[offset_y:offset_y + subpattern_dim[0], offset_x:offset_x + subpattern_dim[1]] = vector

        return pattern.astype('uint8')
    
    def random(self, dim):
        """_summary_

        Args:
            dim (_type_): _description_

        Returns:
            _type_: _description_
        """
        phase_list = np.arange(0, 2 * self.grayphase + 1, self.grayphase / 4)
        pattern = np.array([[random.choice(phase_list) for i in range(dim)] for j in range(dim)])
        return pattern.astype('uint8')


    def add_subpattern(self, subpattern, gray=0):
        """_summary_

        Args:
            subpattern (_type_): _description_
            gray (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        # create a 2d array
        rows = self.res_x
        cols = self.res_y
        # make sure that the image is composed by 8bit integers between 0 and 255
        pattern = np.full(shape=(cols, rows), fill_value=gray).astype('uint8')
        
        # put it in the middle of the slm screen
        # first calculate offsets from the image center
        subpattern_dim = subpattern.shape
        offset_x = int(rows / 2 - subpattern_dim[0] / 2)
        offset_y = int(cols / 2 - subpattern_dim[1] / 2)
        # and then add the vector in the center of the initialized pattern
        pattern[offset_y:offset_y + subpattern_dim[0], offset_x:offset_x + subpattern_dim[1]] = subpattern

        return pattern.astype('uint8')
        
    @staticmethod
    def _enlarge_pattern(matrix, n):
        """
        it takes as input a 2d matrix and augments its dimensions by 2^(n-1) by conserving the same pattern
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
    
    def pattern2SLM(self, pattern, n):
        temp = self._enlarge_pattern(pattern, n)
        temp = self.add_subpattern(temp)
        return temp
    
    def correct_aberrations(self, correction, pattern, alpha=0.5):

        
        # blend images
        beta = (1.0 - alpha)
        pattern = cv2.addWeighted(pattern, alpha, correction, beta, 0.0)
        # pattern = pattern + corr_patt2
        
        return pattern.astype('uint8')
    
        

    

    
    
class OnePixelPatternGenerator:
    
    def __init__(self, slm_segments):
        
        disk_diameter = int(slm_segments ** 0.5)
        self.radius = disk_diameter // 2
        self.N = disk_diameter
    
        self.random_idx = self._get_random_pixels()
        self.indices, self.masks, self.patterns = self._create_patterns()

    def __getitem__(self, idx):
        pattern = self.patterns[idx] 
        index = self.indices[idx] 
        mask = self.masks[idx]      
        return index, mask, pattern
    
    def __len__(self):
        return len(self.patterns)
    
    def _get_disk_mask(self):

        res = (self.N, self.N)
        
        mask_center = [res[0] // 2,res[1] // 2]
        X, Y = np.meshgrid(np.arange(res[0]),np.arange(res[1]))

        # We generate a mask representing the disk we want to intensity to be concentrated in
        mask = (X - mask_center[0]) ** 2 + (Y - mask_center[1]) ** 2 < self.radius ** 2
        
        return mask
    
    def _get_random_pixels(self):
        """ creates all indices of a 2d matrix at a random order
            in order to later sample randomly the pixels of a given mask
        """
        disk = self._get_disk_mask()
        # this will be a list of tuples
        indices = []
        for i in np.arange(self.N):
            for j in np.arange(self.N):
                if disk[i, j]:
                    indices.append((i, j)) # append a tuple to list
#         indices = np.vstack([disk.argmax(axis=0), np.arange(len(disk[0]))]).T[disk.sum(0) > 0]

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
            mask = np.zeros(self.N ** 2, dtype=bool )
            new_dim = int(self.N)
            mask = mask.reshape(new_dim, new_dim)
            
            zero_pattern = np.array([[gray for _ in range(self.N)] for _ in range(self.N)]).astype('uint8')
            temp = zero_pattern
            temp[i, j] = phi
            mask[i, j] = 1
            patterns.append(temp)
            indices.append((i, j))
            masks.append(mask)
            
        return indices, masks, patterns
    
    
class RandomPatternGenerator:
    
    def __init__(self, num_of_patterns, slm_segments):
        
        disk_diameter = int(slm_segments ** 0.5)
        self.radius = disk_diameter // 2
        self.M = disk_diameter

        self.N = num_of_patterns
    
        self.masks, self.patterns = self._create_patterns()

    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        mask = self.masks[idx]        
        return mask, pattern
    
    def __len__(self):
        return len(self.patterns)
    
    def _get_disk_mask(self):

        res = (self.M, self.M)
        mask_center = [res[0] // 2,res[1] // 2]
        X, Y = np.meshgrid(np.arange(res[0]),np.arange(res[1]))

        # We generate a mask representing the disk we want to intensity to be concentrated in
        mask = (X - mask_center[0]) ** 2 + (Y - mask_center[1]) ** 2 < self.radius ** 2

        return mask
    
    def _random_partition(self):
        """
        """
        mask = np.zeros(self.M ** 2, dtype=bool)
        mask[:int((self.M ** 2) / 2)] = 1
        np.random.shuffle(mask)
        
        new_dim = int(self.M)
        mask = mask.reshape(new_dim, new_dim)
        
        return mask
    
    def _create_pattern(self):
        disk = self._get_disk_mask()
        gray = 0
        dim = int(self.M)
        pattern = np.array([[gray for _ in range(dim)] for _ in range(dim)]).astype('uint8')
        mask = self._random_partition()
        pattern[mask] = 1

        return mask * disk, pattern * disk
    

    def _create_patterns(self):
        patterns = []
        masks = []
        for i in range(self.N):
            mask, pattern = self._create_pattern()
            patterns.append(pattern)
            masks.append(mask)
        return masks, patterns
    

class HadamardPatternGenerator:
    
    def __init__(self, num_of_pixels, calib_px):

            self.num_of_px = num_of_pixels
            self.calib_px = calib_px
            self.patterns = self._create_patterns()

    def __getitem__(self, idx):
        item = self.patterns[idx]
        item = self._hadamard_int2phase(item)        
        return item
    
    def __len__(self):
        return len(self.patterns)
    
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
        dim = int(self.num_of_px ** 0.5)
        order = int((np.log2(dim)))

        h = hadamard(2 ** order)
        patterns = [np.outer(h[i], h[j]) for i in range(0, len(h)) for j in range(0, len(h))]
        
        return patterns
        
class GaussPatternGenerator:
    
    def __init__(self, N, num=16, LG=True,
                 w0=3e-3, 
                 wavelength=632e-9, 
                 size=15e-3, 
                 slm_calibration_px=112):
        
        self.N = N  # number of pixles
        self.w0 = w0 # waist
        self.wavelength = wavelength # wavelenght
        self.size = size 
        self.LG = LG # Hermite-Gauss or Laguerre Gauss
        self.num = num # number of vectors
        self.calib_px = slm_calibration_px
        
        self.patterns = self._create_patterns()
    
    def __getitem__(self, idx):
        phi = self.patterns[idx]    
        phi = self._gauss_int2phase(phi)
        return phi
    
    def __len__(self):
        return len(self.patterns)
    
    def _gauss_int2phase(self, vector):
        """
        replaces the elements of the gaussian phase (0, 3.14) with the useful slm values (0, grayscale)
        one needs to know the grayscale value of the slm that gives a phase shift of pi
        Parameters
        ----------
        vector: 2d array

        Returns
        -------
        vector: 2d array

        """

        return vector * self.calib_px / np.pi

    
    def _create_sorted_indices(self):
        """ Create a a list of sorted 2d indices in order to generate
            vectors with increased spatial frequency.

        Returns
        -------
        a list of lists
        """
        indices = []
        for n in range(self.num):
            for m in range(self.num):
                indices.append([n, m])
        return sorted(indices, key=sum)
    
    def _create_patterns(self):
        """ Uses LightPipe library to create HG or LG patterns

        Returns
        -------
            a list with all created patterns
        """
        patterns = []
        
        F = Begin(self.size, self.wavelength, self.N)
        indices = self._create_sorted_indices()
        for n, m in indices:
                F = GaussBeam(F, self.w0, LG=self.LG, n=m, m=n)
                # Amp = Intensity(0, F)
                Phi = Phase(F)
                patterns.append(Phi)
        return patterns