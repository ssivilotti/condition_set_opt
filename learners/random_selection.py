import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from space_mat import SpaceMatrix
from learners.learner import Classifier

class ALClassifierRandomSelection(Classifier):
    '''Randomly selects the next points to be measured, benchmarking against active learning classifiers'''
    def __init__(self, space_shape:int):
        super().__init__(space_shape)

    def suggest_next_n_points(self, X:np.ndarray, n:int, measured_indices:np.ndarray)->list:
        '''next_points is a list of indices of the next points to be measured'''
        uncertainty = self.predict(X)

        if len(uncertainty) == 0:
            return []

        num_options = np.prod(self.shape)

        options = np.array([i for i in range(num_options) if not (i in measured_indices)])

        next_points = np.random.choice(options, n, replace=False).tolist()
        
        point_uncertainties = (abs(np.array([uncertainty[i][0] for i in next_points]) - .5) / .5).tolist()
        avg_point_certainty = np.average(point_uncertainties)

        # self.done = avg_point_certainty > .5
        self.done = False
        return uncertainty, self.predicted_surface, next_points, point_uncertainties