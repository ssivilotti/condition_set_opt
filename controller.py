import os
import pickle
import numpy as np
import datetime as dt
from scipy.stats import qmc
from chemical_space import ChemicalSpace
from space_mat import SpaceMatrix
from space_mat import BINARY_COUNT
from learner import Learner
from al_classifier import ALClassifierBasic
from random_selection import ALClassifierRandomSelection

# Learners
# Active Learning Classifier
ALC = 0
# Active Learning Classifier with Tuned Aquistion Function
ALCGEN2 = 1
# Random Selection of Next Points
RAND = -1

# Initial Seed Sampling Methods
KPP = 0
LHS = 1

# handles featurization, chosing optimal sets and communication between the active learning algorithm and the chemical space
class Controller:
    def __init__(self, chemical_space:ChemicalSpace, learner_type:int=ALC, batch_size:int=10, max_experiments:int=1000, known_points=None) :
        self.chemical_space = chemical_space
        if learner_type == ALC:
            self.learner = ALClassifierBasic(chemical_space.shape, chemical_space.reactants_dim)
        if learner_type == RAND:
            self.learner = ALClassifierRandomSelection(chemical_space.shape)
        self.scoring_function = BINARY_COUNT
        self.batch_size = batch_size
        self.max_experiments = max_experiments
        self.metrics = {'max_experiments': self.max_experiments, 'batch_size': self.batch_size,}
        if chemical_space.descriptors == None:
            self.all_points_featurized = [self.convert_to_onehot(point) for point in chemical_space.all_points]
        if known_points == None:
            self.seed_len = 49
            self.known_points = []
        else:
            self.seed_len = 0
            self.known_points = known_points
        self.optimization_runs = 0

    def convert_to_onehot(self, point):
        onehot = np.zeros(np.sum(self.chemical_space.shape))
        for i, p in enumerate(point):
            onehot[int(p + np.sum(self.chemical_space.shape[:i]))] = 1
        return onehot
    
    def convert_from_onehot(self, onehot):
        point = []
        num = 0
        shape_counter = 0
        for i in range(len(onehot)):
            if onehot[i] == 1:
                point.append(num)
            num += 1
            if num >= self.chemical_space.shape[shape_counter]:
                num = 0
                shape_counter += 1
        return point
    
    def initial_seed(self, n, sampling_method:int = LHS) -> list:
        if sampling_method == KPP:
            return []
        elif sampling_method == LHS:
            # ensure n is a perfect quare of prime numbers
            sampler = qmc.LatinHypercube(d=len(self.chemical_space.shape), strength=2)
            seed = sampler.integers(l_bounds=np.zeros(len(self.chemical_space.shape), dtype=int), u_bounds=list(self.chemical_space.shape), n=n)
            return seed
        return []
    
    def get_initial_seed(self, initial_seed:int):
        # TODO: implement known points
        if initial_seed == 0:
            return self.known_points, self.known_points, 0
        seed = []
        seed_attempts = 0
        seed_vals = []
        seed_vals_sum = 0
        # ensures that seed has at least one successful reaction and at least one unsuccessful reaction
        while seed_attempts < 10 and (seed_vals_sum < 1 or seed_vals_sum >= len(seed)):
            seed = self.initial_seed(initial_seed)
            seed_vals = np.array([self.chemical_space.measure_reaction_yield(seed[i]) for i in range(len(seed))])
            seed_vals_sum = np.sum(seed_vals)
            seed_attempts += 1
        return seed, seed_vals, seed_attempts
    
    def convert_point_to_idx(self, point):
        idx = 0
        for i, n in enumerate(point):
            idx += n
            if i < len(self.chemical_space.shape) - 1:
                idx *= self.chemical_space.shape[i+1]            
            # np.prod(self.chemical_space.shape[i+1:])
        return idx
    
    def optimize(self, cutoff:float, save_to_file=False)->dict:
        # measured_yields = {}
        num_experiments_run:int = 0

        seed, seed_vals, seed_attempts = self.get_initial_seed(self.seed_len)

        metrics = {'accuracy': [], 'precision': [], 'recall': [], 
                   'best_set': [], 'coverage': [], 
                   'points_suggested': [seed], 'certainties':[[.5 for i in range(len(seed))]]}

        x = None
        y = None

        next_points = np.array(seed)
        known_idxs = [self.convert_point_to_idx(point) for point in next_points]

        predicted_surface:SpaceMatrix
        best_set = []
        last_change = 0
        coverage = 0

        self.learner.initialize_model()

        while (not(self.learner.done) or (num_experiments_run < len(seed) + self.batch_size)) and (num_experiments_run < self.max_experiments):
            # measure yields for uncertain points
            measurement = np.array([self.chemical_space.measure_reaction_yield(next_points[i]) > cutoff for i in range(len(next_points))])
            # measured_yields.update({tuple(next_points[i]): measurement[i] for i in range(len(next_points))})

            if (x is None) and (y is None):
                x = [self.convert_to_onehot(point) for point in next_points]
                y = measurement
            else:
                x = np.append(x, [self.convert_to_onehot(point) for point in next_points], axis=0)
                y =  np.append(y, measurement, axis=0)

            num_experiments_run += len(measurement)
            self.learner.fit(x, y)
            
            uncertainty, predicted_surface = self.learner.predict(np.array(self.all_points_featurized))

            accuracy, precicion, recall = self.chemical_space.score_classifier_prediction(uncertainty, cutoff)

            predicted_set, coverage = predicted_surface.best_condition_sets(self.chemical_space.all_conditions, self.scoring_function, 3, 1)
            predicted_set = predicted_set[0]
            coverage = coverage[0]
            if predicted_set != best_set:
                best_set = predicted_set
                last_change = 0
            else:
                last_change += 1

            next_points, certainties = self.learner.suggest_next_n_points(self.batch_size, known_idxs)
            known_idxs = known_idxs + next_points
            next_points = [self.chemical_space.all_points[i] for i in next_points]
            
            metrics['accuracy'].append(accuracy)
            metrics['precision'].append(precicion)
            metrics['recall'].append(recall)
            metrics['best_set'].append(best_set)
            metrics['coverage'].append(coverage)
            metrics['points_suggested'].append(next_points)
            metrics['certainties'].append(certainties)

        metrics['num_experiments_run'] = num_experiments_run   

        self.metrics[self.optimization_runs] = metrics

        if save_to_file:
            self.save_optimization_run_metrics()
        
        self.optimization_runs += 1

        self.learner.reset()

        return best_set, coverage
    
    def do_repeats(self, n_repeats:int, cutoff:int):
        for i in range(n_repeats):
            self.optimize(cutoff)
        self.save_metrics_to_pkl()
    
    def save_metrics_to_pkl(self)->None:
        date_str = f"{dt.datetime.now()}"
        date_str = date_str[:10] +"_"+ date_str[11:19]
        os.makedirs(f'metrics/{self.chemical_space.dataset_name}', exist_ok = True) 
        with open(f"metrics/{self.chemical_space.dataset_name}/metrics_{date_str}.pkl", "wb+") as f:
            pickle.dump(self.metrics, f)
        
    def save_optimization_run_metrics(self, key=None)->None:
        date_str = f"{dt.datetime.now()}"
        date_str = date_str[:10] +"_"+ date_str[11:19]
        if key == None:
            key = self.optimization_runs
        item = self.metrics[key]
        os.makedirs(f'metrics/{self.chemical_space.dataset_name}', exist_ok = True) 
        if not(type(item) is dict):
            with open(f"metrics/{self.chemical_space.dataset_name}/metrics_{date_str}_{key}.txt", "w+") as f:
                f.write(f"{key}: {item}\n")
            return
        with open(f"metrics/{self.chemical_space.dataset_name}/metrics_{date_str}_{key}.txt", "w+") as f:
            for key, val in self.metrics[key].items():
                f.write(f"{key}: {val}\n")
 