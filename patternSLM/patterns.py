import numpy as np
from scipy import signal
from scipy.linalg import hadamard


class Pattern:

    def __init__(self, resX, resY):
        self.resX = resX
        self.resY = resY
        x = np.linspace(0, resX, resX)
        y = np.linspace(0, resY, resY)
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
        rows = self.resX
        cols = self.resY

        # make sure that the image is composed by 8bit integers between 0 and 255
        pattern = np.array([[gray for i in range(rows)] for j in range(cols)]).astype('uint8')
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
        rows = self.resX
        cols = self.resY
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
        pattern[:self.resY, : int(self.resX / a)] = gray

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

        h = hadamard(order)
        matrices = [np.outer(h[i], h[j]) for i in range(0, len(h)) for j in range(0, len(h))]
        return matrices

    @staticmethod
    def _get_hadamard_vector(order, i, j):
        h = hadamard(order)
        matrix = np.outer(h[i], h[j])
        return matrix

    def hadamard_pattern(self, order, hadamard_vector_idx, gray=0):

        # create a 2d array
        rows = self.resX
        cols = self.resY
        # make sure that the image is composed by 8bit integers between 0 and 255
        pattern = np.full(shape=(cols, rows), fill_value=gray).astype('uint8')

        # put in the middle of the SLM screen the hadamard 2d vector
        i, j = hadamard_vector_idx[0], hadamard_vector_idx[1]
        hadamard_vector = self._get_hadamard_vector(order, i, j)
        subpattern_dim = hadamard_vector.shape

