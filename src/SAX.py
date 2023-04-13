""" Symbolic Aggregated approXimation """

import numpy as np
from scipy.stats import norm
from functools import reduce

__author__ = "Christian Gruhl"
__copyright__ = "Copyright 2019, Intelligent Embedded Systems, Universität Kassel"

class SAX:
    """  Implement SAX """

    def __init__(self, symbols=5, paa_length=3):
        """ Initialize a new approximation objection.

        Parameters:
        ===========
        symbols - number of different symbols to be used for encoding
        
        paa_length - number of sample points to aggregate.
        """
        self.symbols = symbols
        self.paa_length = paa_length
        
    def paa(self, X):
        """ Perform piecewise aggreagated approximation. """
        return np.asarray([ np.mean(X[start:min(start+self.paa_length, len(X))]) for start in np.arange(0, len(X), self.paa_length) ])
    
    def fit(self, X):
        """ Fit boundaries.

        Calculates mean and std from training data to be used in
        transform(X)
        
        Parameters:
        ===========
        X - time series do be used as source

        Return:
        =======
        reference to self
        """
        self.X_paa = self.paa(X)
        self.mean_ = np.mean(self.X_paa)
        self.std_  = np.std(self.X_paa)
        self.percentiles_ = np.linspace(0, 1, self.symbols + 1)
        self.boundaries_ = norm.ppf(self.percentiles_)[1:-1]
        return self

    def transform(self, X, return_as='symbol'):
        """
        Transform the given time series to a symbolic representation.

        Standardizes X with the parameters learned from training data.

        Arguments:
        ==========
        X - time series do be converted.

        as_symbol - type of returned representation 'symbol' ndarray with chr, 'str' a string, 'idx' ndarray with integers
        """
        Xt = self.paa(X)
        Xt = (Xt - self.mean_) / self.std_
        S = np.searchsorted(self.boundaries_, Xt)
        if return_as == 'str' or return_as == 'symbol':
            # change (Klara), x[0] to get int, not array
            symbols = np.apply_along_axis(lambda x: chr(x[0] + ord('a')), 0, np.expand_dims(S, axis=0))
            # change (Klara)
            #symbols = np.array(list(map(lambda x: chr(x + ord('a')), S)))
            if return_as == 'str':
                return reduce(str.__add__, symbols, '')
            else:
                return symbols
        else:
            return S
    
    def reconstruct(self, S, reconstruct_from='symbol'):
        """
        Reconstruct the time series from its Symbolic representation.

        Arguments:
        ==========
        S - symbolic representation of a time series

        reconstruct_from - type of input 'symbol', 'str', 'idx'

        Returns:
        ========
        Reconstructed signal from S 
        """

        if reconstruct_from == 'symbol' or reconstruct_from == 'str': # change (Klara)
            S = np.apply_along_axis(lambda x : ord(x[0]) - ord('a'), 0, np.expand_dims(S, axis=0))

        rec_percentiles = np.linspace(0, 1, self.symbols + 2)
        rec_boundaries = norm.ppf(rec_percentiles)[1:-1]
        return np.repeat((rec_boundaries[S] * self.std_) + self.mean_, self.paa_length)
