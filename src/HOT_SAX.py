""" Symbolic Aggregated approXimation not using heuristic from literature"""


import numpy as np
from scipy.stats import norm
from functools import reduce
from scipy.spatial.distance import pdist

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
        #sub_lists = np.array_split(X, len(X)//window_size) #[X[i:i+window_size] for i in range(0, len(X)), window:size)]
        #dist = [(x - y)**2 for x,y in zip(window1, window2)]
        #np.apply_along_axis(lambda x,y: chr(x[0] + ord('a')), 0, np.expand_dims(S, axis=0))
        #print(1, np.expand_dims(X, axis=1))
        # compare paa version (only one value for paa_length many entries
        if window_size != 1:
            print([X[i:i+window_size] for i in range(0, len(X), window_size)])
            Y = [sum(pdist(X=np.expand_dims(X[i:i+window_size], axis=1), metric='sqeuclidean')) for i in range(0, len(X), window_size)]
        else:
            Y = pdist(X=np.expand_dims(X, axis=1), metric='sqeuclidean')
        return Y
