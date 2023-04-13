""" Symbolic Aggregated approXimation not using heuristic from literature"""


import numpy as np
from scipy.spatial.distance import pdist
from itertools import combinations

__author__ = "Klara Gutekunst"
__copyright__ = "Copyright 2023, Intelligent Embedded Systems, Universität Kassel/Finanzamt Kassel"

class HOTSAX:
    """  Implement HOT alternative; HOT-SAX discord discovery algorithm """

    def __init__(self, window_size, number_of_discords):
        ''' Initialize a new discords discovery objection.

        Parameters:
        ===========
        window_size - number of data points which belong to a frame
        number_of_discords - number of discords to identify
        '''
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

    def smallest_distance_frame(self, X):
        ''' for every frame: finds the minimum distance of entries which measure distance between that certain and another frame

        Parameters:
        ===========
        X - array containing time series data

        Return:
        =======
        result_value - list containing the minimum distance for every frame
        result_index - list containing the index in the distances list of the minimum distance for every frame
        '''
        # obtain list containing squared euclidean distances
        distances = self.compare_pairwise(X)

        # prepare result list containing minimum distance for every frame
        result_index = []
        result_value = []

        # store smallest distance to another frame for every frame
        for frame in range(0, len(X)//self.window_size):
            # indices where distances between certain frame and others is stored
            start = max(frame * len(X) - sum(range(frame + 1)), 0)
            end = start + (len(X) - (frame + 1))
            own_indices = np.arange(start, end, 1)
            other_indices = np.array([(len(X) * i - sum(range(i + 1)) + frame - (i + 1)) for i in range(0, frame)])
            # other_indices is empty for frame==0; concatenating [] with own_indices causes indices type to be float
            # to avoid error caused by trying to index using floats, concatenating only happens for indices bigger than 0
            indices = np.concatenate((own_indices, other_indices)) if frame > 0 else own_indices

            # index of minimum distance value of certain frame
            # argmin returns index of minimum of sublist of distances, which is index of indices list equivalent containing all distances of that certain frame
            minimum = indices[np.argmin(distances[indices])]
            result_index.append(minimum)
            result_value.append(distances[minimum])

        return result_value, result_index

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