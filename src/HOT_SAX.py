""" Symbolic Aggregated approXimation not using heuristic from literature"""


import numpy as np
from scipy.spatial.distance import pdist
from itertools import combinations

__author__ = "Klara Gutekunst"
__copyright__ = "Copyright 2023, Intelligent Embedded Systems, Universität Kassel/Finanzamt Kassel"

class HOTSAX:
    """  Implement HOT alternative; HOT-SAX discord discovery algorithm """

    def __init__(self, sax, window_size, number_of_discords):
        ''' Initialize a new discords discovery objection.

        Parameters:
        ===========
        sax - trained sax model
        window_size - number of data points which belong to a frame
        number_of_discords - number of discords to identify
        '''
        self.sax = sax
        self.window_size = window_size
        self.number_of_discords = number_of_discords

    def compare_pairwise(self, X):
        ''' Calulates the squared euclidean distance between data frames.

        Parameters:
        ===========
        X - array containing time series data

        Return:
        =======
        list which stores the squared euclidean distance of the frames.
        An entry is summed result of the squared euclidean distances of the respective entries of two frames.
        First all combinations of frame distances to the first frame (excluding itself) are stored, then with the next
        frame (no repetition) etc.
        '''
        if (self.window_size != 1) and (self.window_size < len(X)):
            # windows entries contain data points, which belong to the same window
            windows = np.array_split(X, len(X)//self.window_size)

            # frames between which the squared euclidean distance will be calculated
            combis = list(combinations(np.arange(0, len(windows), 1), 2))

            # sum up squared euclidean distance per respective entry of two windows
            return np.array([[sum((x[0] - x[1]) ** 2 for x in zip(windows[window1], windows[window2]))] for window1, window2 in combis]).flatten()
        else:
            # squared euclidean distance between entries
            return pdist(X=np.expand_dims(X, axis=1), metric='sqeuclidean')

    def identify_discord(self, X):
        '''

        '''
        # obtain list containing squared euclidean distances
        distances = self.compare_pairwise(X)
        print(distances)

        # find index i of the greatest n values in list containing squared euclidean distances
        # first index corresponds to the largest distance
        sorted_distances_indices = np.argsort(distances)[- self.number_of_discords:]
        i = np.flip(sorted_distances_indices[- self.number_of_discords:])
        print(i)

        # return list of frames which correspond to i