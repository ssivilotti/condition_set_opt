import numpy as np
from learners.learner import Classifier, YieldPred

class ALClassifierRandomSelection(Classifier):
    '''Randomly selects the next points to be measured, benchmarking against active learning classifiers'''
    def __init__(self, space_shape:int, cpus=None, model_type='GP'):
        super().__init__(space_shape, model_type, cpus)

    def suggest_next_n_points(self, X:np.ndarray, n:int, measured_indices:set)->list:
        '''next_points is a list of indices of the next points to be measured'''
        uncertainty = self.predict(X)

        assert len(uncertainty) != 0, 'Failed to predict uncertainty'

        num_options = np.prod(self.shape)

        options = np.array([i for i in range(num_options) if not (i in measured_indices)])

        next_points = np.random.choice(options, n, replace=False).tolist()
        
        point_uncertainties = (abs(np.array([uncertainty[i][0] for i in next_points]) - .5) / .5).tolist()

        return uncertainty, self.predicted_surface, next_points, point_uncertainties
    
class ALYieldPredRandomSelection(YieldPred):
    '''Randomly selects the next points to be measured, benchmarking against active learning classifiers'''
    def __init__(self, space_shape:int):
        super().__init__(space_shape)

    def suggest_next_n_points(self, X:np.ndarray, n:int, measured_indices:set)->list:
        '''next_points is a list of indices of the next points to be measured'''
        yield_std = self.predict(X)

        if len(yield_std) == 0:
            return []

        num_options = np.prod(self.shape)

        options = np.array([i for i in range(num_options) if not (i in measured_indices)])

        next_points = np.random.choice(options, n, replace=False).tolist()
        
        point_uncertainties = np.array([yield_std[i] for i in next_points])

        return yield_std, self.predicted_surface, next_points, point_uncertainties