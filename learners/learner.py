import numpy as np
from abc import ABC, abstractmethod
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from space_mat import SpaceMatrix
import time

class Learner(ABC):
    def __init__(self, shape):
        self.done = False
        self.shape = shape

    @abstractmethod
    def fit(self, X:np.ndarray, y:np.ndarray)->None:
        pass

    @abstractmethod
    def predict(self, X:np.ndarray)->tuple:
        pass

    @abstractmethod
    def suggest_next_n_points(self, X:np.ndarray, n:int, measured_indices:np.ndarray)->tuple:
        pass

    @abstractmethod
    def reset(self)->None:
        pass

    @abstractmethod
    def initialize_model(self)->None:
        pass


class Classifier(Learner):
    '''abstract class for binary classifier model, used in Controller
    subclasses must implement suggest_next_n_points to conplete the active learning agent'''
    def __init__(self, space_shape:int, cpus=None):
        super().__init__(space_shape)
        self.cpus = cpus
        self.reset()
    
    def fit(self, X:np.ndarray, y:np.ndarray)->None:
        if self.model == None:
            self.initialize_model()
        self.model.fit(X, y)

    def initialize_model(self):
        kernel = 1.0 * RBF(1.0)
        self.model = GaussianProcessClassifier(kernel=kernel, copy_X_train=False, n_jobs=self.cpus)

    def predict(self, X:np.ndarray)->tuple:
        assert len(X) > 0, 'No points to predict'
        uncertainty = []
        # fix memory error for large datasets
        start = time.time()
        while len(X) > 25000:
            uncertainty.extend(self.model.predict_proba(X[:25000]))
            X = X[25000:]
        uncertainty.extend(self.model.predict_proba(X))
        print(f"Time to predict: {time.time() - start}")
        # Predicted surface is a matrix of the predicted probability of the positive class (above .5 is predicted to be true)
        # self.predicted_surface = SpaceMatrix(np.array([uncertainty[i][1] > .5 for i in range(len(uncertainty))]).reshape(self.shape, order='C'))
        self.predicted_surface = SpaceMatrix(np.array([uncertainty[i][1] for i in range(len(uncertainty))]).reshape(self.shape, order='C'))
        return np.array(uncertainty)

    def reset(self)->None:
        self.done = False
        self.predicted_surface = None
        self.initialize_model()

class YieldPred(Learner):
    # TODO: set y between 0-1, not 0-100
    '''abstract class for binary classifier model, used in Controller
    subclasses must implement suggest_next_n_points to conplete the active learning agent'''
    def __init__(self, space_shape:int):
        super().__init__(space_shape)
        self.reset()
    
    def fit(self, X:np.ndarray, y:np.ndarray)->None:
        if self.model == None:
            self.initialize_model()
        self.model.fit(X, y)

    def initialize_model(self):
        kernel = 1.0 * RBF(1.0)
        self.model = GaussianProcessRegressor(kernel=kernel, random_state=0)

    def predict(self, X:np.ndarray)->tuple:
        yield_mean, yield_std = self.model.predict(X, return_std=True)
        self.predicted_surface = SpaceMatrix(np.array(yield_mean).reshape(self.shape, order='C'))
        return yield_std

    def reset(self)->None:
        self.done = False
        self.predicted_surface = None
        self.initialize_model()
    