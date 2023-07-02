import numpy as np
from scipy import signal
from scipy.linalg import hadamard


class Pattern:

    def __init__(self, res_x, res_y, grayphase=112):
        """

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

    def hadamard_pattern(self, order, hadamard_vector_idx, gray=0):
        """
        creates a hadamard vector and puts it in the middle of the slm screen
        Parameters
        ----------
        order: hadamard matrix order (int)
        hadamard_vector_idx: input of the index of the hadamard vector (tuple with 2 int)
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
