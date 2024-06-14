import numpy as np
from learners.learner import Classifier

class ALClassifierBasic(Classifier):
    '''Active learning classifier that suggests the top n points with the highest uncertainty that haven't been measured yet to be measured next'''
    def __init__(self, space_shape:tuple, cutoff_certainty=.9):
        super().__init__(space_shape)
        self.certainty_cutoff = (cutoff_certainty - .5)/.5

    def suggest_next_n_points(self, X:np.ndarray, n:int, measured_indices:np.ndarray)->list:
        '''next_points is a list of indices of the next points to be measured'''
        # TODO: pick different locations based on uncertainty, ensure batch doesn't cover only one area

        uncertainty = self.predict(X)

        if len(uncertainty) == 0:
            return []

        distance_from_unknown = abs(uncertainty.T[0] - .5)
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

        self.done = avg_point_certainty > self.certainty_cutoff
        return uncertainty, self.predicted_surface, next_points, point_uncertainties
    