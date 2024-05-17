import numpy as np
from abc import ABC, abstractmethod
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from space_mat import SpaceMatrix

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
    def suggest_next_n_points(self, n:int, measured_indices:np.ndarray)->np.ndarray:
        pass

    @abstractmethod
    def reset(self)->None:
        pass

    @abstractmethod
    def initialize_model(self)->None:
        pass


class Classifier(Learner):
    def __init__(self, space_shape:int):
        super().__init__(space_shape)
        self.reset()
    
    def fit(self, X:np.ndarray, y:np.ndarray)->None:
        if self.model == None:
            self.initialize_model()
        self.model.fit(X, y)

    def initialize_model(self):
        kernel = 1.0 * RBF(1.0)
        self.model = GaussianProcessClassifier(kernel=kernel,random_state=0)

    def predict(self, X:np.ndarray)->tuple:
        self.uncertainty = self.model.predict_proba(X)
        predicted_surface = SpaceMatrix(np.array([self.uncertainty[i][1] > .5 for i in range(len(self.uncertainty))]).reshape(self.shape, order='C'))
        return self.uncertainty, predicted_surface

    def reset(self)->None:
        self.done = False
        self.uncertainty = []
        self.initialize_model()
    