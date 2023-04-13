""" Symbolic Aggregated approXimation not using heuristic from literature"""


import numpy as np
from scipy.spatial.distance import pdist
from itertools import combinations

__author__ = "Klara Gutekunst"
__copyright__ = "Copyright 2023, Intelligent Embedded Systems, Universität Kassel/Finanzamt Kassel"

class HOTSAX:
    """  Implement HOT alternative; HOT-SAX discord discovery algorithm """

    def __init__(self, sax):
        ''' Initialize a new discords discovery objection.

        Parameters:
        ===========
        sax - trained sax model
        '''
        self.sax = sax

    def compare_pairwise(self, window_size, X):
        ''' Calulates the squared euclidean distance between data frames.

        Parameters:
        ===========
        window_size - number of data points which belong to a frame
        X - array containing time series data

        Return:
        =======
        list which stores the squared euclidean distance of the frames.
        An entry is summed result of the squared euclidean distances of the respective entries of two frames.
        First all combinations of frame distances to the first frame (excluding itself) are stored, then with the next
        frame (no repetition) etc.
        '''
        if (window_size != 1) and (window_size < len(X)):
            # windows entries contain data points, which belong to the same window
            windows = np.array_split(X, len(X)//window_size)

            # frames between which the squared euclidean distance will be calculated
            combis = list(combinations(np.arange(0, len(windows), 1), 2))

            # sum up squared euclidean distance per respective entry of two windows
            return [[sum((x[0] - x[1]) ** 2 for x in zip(windows[window1], windows[window2]))] for window1, window2 in combis]
        else:
            # squared euclidean distance between entries
            return pdist(X=np.expand_dims(X, axis=1), metric='sqeuclidean')