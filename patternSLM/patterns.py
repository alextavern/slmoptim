import numpy as np
from scipy import signal
from scipy.linalg import hadamard
import threading
from tqdm.auto import tqdm


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
    def _get_hadamard_basis(order):
        """
        calculates the outer product of all combination of the rows of a hadamard matrix of a given order to
        generate 2d patterns that constitute a hadamard basis.
        Parameters
        ----------
        order: the order of the hadamard matrix (int)

        Returns
        -------
        matrices: all the 2d patterns (array)

        """
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

    def hadamard_pattern(self, order, hadamard_vector_idx, n=1, gray=0):
        """
        creates a hadamard vector and puts it in the middle of the slm screen
        Parameters
        ----------
        order: hadamard matrix order (int)
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

        return pattern.astype('uint8')

    def hadamard_pattern_bis(self, vector, n=1, gray=0):
        """
        creates a hadamard vector and puts it in the middle of the slm screen
        Parameters
        ----------
        vector: one hadamard vector
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
        hadamard_vector = vector
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

        return pattern.astype('uint8')

    @staticmethod
    def _enlarge_pattern(matrix, n):
        """
        it takes as input a 2d matrix and augments its dimensions by 2nx2n by conserving the same pattern
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

        return matrix.astype('uint8')


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
        
        pi = int(calib_px / 2)
        self.four_phases = [0, pi / 2, pi, 3 * pi / 2]
        
        self.download = download_frame_event
        self.stop = stop_all_event
        self.upload = upload_pattern_event
        
    def run(self):

            # loop through each 2d vector of the hadamard basis - basis is already generated here
            self.upload.set()
            for idx, vector in enumerate(tqdm(self.basis)):
                # and for each vector load the four reference phases
                for phase in tqdm(self.four_phases, leave=False):
                    pattern = self.slm_patterns.hadamard_pattern_bis(vector, n=self.mag, gray=phase)
                    self.upload.wait()
                    self.slm.updateArray(pattern) # load each vector to slm
                    # send flag to other threads here
                    self.download.set()
                    self.upload.clear()
#                     if self.trigger_event.is_set():
#                         print('download pattern')
           
#             if self.stop_event.is_set():
#                 print('stop all')
            return  self.stop.set()