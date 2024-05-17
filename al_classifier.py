import numpy as np
from learner import Classifier

class ALClassifierBasic(Classifier):
    def __init__(self, space_shape:tuple, reactants_dim:int):
        super().__init__(space_shape)
        self.reactants_dim = reactants_dim

    def suggest_next_n_points(self, n:int, measured_indices:np.ndarray)->list:
        '''next_points is a list of indices of the next points to be measured'''
        if len(self.uncertainty) == 0:
            return []

        distance_from_unknown = abs(self.uncertainty.T[0] - .5)
        uncertainty_order = np.argsort(distance_from_unknown)

        next_points = []
        next_point = uncertainty_order[0]
        point_uncertainties = []
        i = 0
        while len(next_points) < n and (i+1) < len(uncertainty_order):
            while (next_point in measured_indices) and (i+1) < len(uncertainty_order):
                i += 1
                next_point = uncertainty_order[i]
            next_points.append(next_point)
            point_uncertainties.append((distance_from_unknown[uncertainty_order[i]])/.5)
            i += 1
            if i < len(uncertainty_order):
                next_point = uncertainty_order[i]
        avg_point_certainty = np.average(point_uncertainties)

        self.done = avg_point_certainty > .5
        return next_points, point_uncertainties
    