""" Symbolic Aggregated approXimation not using heuristic from literature"""


import numpy as np
from scipy.stats import norm
from functools import reduce
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
        '''
        '''

        print(window_size, X)
        if (window_size != 1) and (window_size < len(X)):
            # windows entries contain data points, which belong to the same window
            windows = np.array_split(X, len(X)//window_size)
            print('windows:', windows)

            # sum up die squared euclidean distance per respective entry of two windows
            combis = list(combinations(np.arange(0, len(windows), 1), 2))
            print('Combis', combis)
            Y = [[sum((x[0] - x[1]) ** 2 for x in zip(windows[window1], windows[window2]))] for window1, window2 in combis]
        else:
            Y = pdist(X=np.expand_dims(X, axis=1), metric='sqeuclidean')
        return Y
