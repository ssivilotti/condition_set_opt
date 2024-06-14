import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt
from scipy.stats import qmc
from chemical_space import ChemicalSpace
from space_mat import SpaceMatrix
from space_mat import THRESHOLDED_COUNT
from learners.learner import Learner
from learners.al_classifier import ALClassifierBasic
from learners.random_selection import ALClassifierRandomSelection
from tools.functions import convert_to_onehot, convert_point_to_idx

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
    def __init__(self, chemical_space:ChemicalSpace, batch_size:int=10, max_experiments:int=1000, max_set_size:int=3, learner_type:int=ALC, known_points=None) :
        self.chemical_space = chemical_space
        if learner_type == ALC:
            self.learner = ALClassifierBasic(chemical_space.shape)
        if learner_type == RAND:
            self.learner = ALClassifierRandomSelection(chemical_space.shape)
        self.scoring_function =  THRESHOLDED_COUNT(np.prod(chemical_space.shape[chemical_space.conditions_dim:]))(.5)
        self.batch_size = batch_size
        self.max_experiments = max_experiments
        self.metrics = {'max_experiments': self.max_experiments, 'batch_size': self.batch_size,}
        if chemical_space.descriptors == None:
            self.all_points_featurized = [convert_to_onehot(self.chemical_space.shape, point) for point in chemical_space.all_points]
        if known_points == None:
            self.seed_len = 49
            self.known_points = []
        else:
            self.seed_len = 0
            self.known_points = known_points
        self.optimization_runs = 0
        date_str = f"{dt.datetime.now()}"
        self.date_str = date_str[:10] +"_"+ date_str[11:19]
        self.max_set_size = max_set_size
        self.cond_to_rank_map = chemical_space.yield_surface.rank_conditions(chemical_space.all_conditions, max_set_size)
    
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
    
    def optimize(self, cutoff:float, save_to_file=False)->dict:
        # measured_yields = {}
        self.learner.reset()

        num_experiments_run:int = 0

        seed, seed_vals, seed_attempts = self.get_initial_seed(self.seed_len)

        metrics = {'cutoff': cutoff, 'accuracy': [], 'precision': [], 'recall': [], 
                   'best_sets': [], 'coverages': [], 
                   'points_suggested': [seed], 'certainties':[[.5 for i in range(len(seed))]]}

        x = None
        y = None

        next_points = np.array(seed)
        known_idxs = [convert_point_to_idx(self.chemical_space.shape, point) for point in next_points]

        # for a classifier, the predicted surface is the probability of the positive class
        predicted_surface:SpaceMatrix
        best_set = []
        last_change = 0
        coverage = 0

        self.learner.initialize_model()

        while (not(self.learner.done) or (num_experiments_run < len(seed) + 3*(self.batch_size))) and (num_experiments_run < self.max_experiments):
            # measure yields for uncertain points
            measurement = np.array([self.chemical_space.measure_reaction_yield(next_points[i]) > cutoff for i in range(len(next_points))])
            # measured_yields.update({tuple(next_points[i]): measurement[i] for i in range(len(next_points))})

            if (x is None) and (y is None):
                x = [convert_to_onehot(self.chemical_space.shape, point) for point in next_points]
                y = measurement
            else:
                x = np.append(x, [convert_to_onehot(self.chemical_space.shape, point) for point in next_points], axis=0)
                y =  np.append(y, measurement, axis=0)

            num_experiments_run += len(measurement)
            self.learner.fit(x, y)

            all_points_uncertainty, predicted_surface, next_points, certainties = self.learner.suggest_next_n_points(np.array(self.all_points_featurized), self.batch_size, known_idxs)
            known_idxs = known_idxs + next_points
            next_points = [self.chemical_space.all_points[i] for i in next_points]

            accuracy, precicion, recall = self.chemical_space.score_classifier_prediction(all_points_uncertainty, cutoff)

            predicted_sets_1, coverages_1 = predicted_surface.best_condition_sets(self.chemical_space.all_conditions, self.scoring_function, 1, 3)
            predicted_sets_2, coverages_2 = predicted_surface.best_condition_sets(self.chemical_space.all_conditions, self.scoring_function, 2, 3)
            predicted_sets, coverages = predicted_surface.best_condition_sets(self.chemical_space.all_conditions, self.scoring_function, self.max_set_size, 3)
            predicted_set = predicted_sets[0]
            coverage = coverages[0]
            if predicted_set != best_set:
                best_set = predicted_set
                last_change = 0
            else:
                last_change += 1
            
            metrics['accuracy'].append(accuracy)
            metrics['precision'].append(precicion)
            metrics['recall'].append(recall)
            metrics['best_sets'].append([predicted_sets_1, predicted_sets_2, predicted_sets])
            metrics['coverages'].append([coverages_1, coverages_2, coverages])
            metrics['points_suggested'].append(next_points)
            metrics['certainties'].append(certainties)

        metrics['num_experiments_run'] = num_experiments_run   

        self.metrics[self.optimization_runs] = metrics

        if save_to_file:
            self.save_optimization_run_metrics()
            self.plot_metrics(self.optimization_runs)
            self.plot_coverage(self.optimization_runs)
        
        self.optimization_runs += 1

        return best_set, coverage
    
    def do_repeats(self, n_repeats:int, cutoff:float):
        for i in range(n_repeats):
            self.optimize(cutoff)
        self.save_metrics_to_pkl()
    
    def save_metrics_to_pkl(self)->None:
        # date_str = f"{dt.datetime.now()}"
        # date_str = date_str[:10] +"_"+ date_str[11:19]
        os.makedirs(f'metrics/{self.chemical_space.dataset_name}/{self.date_str}', exist_ok = True) 
        with open(f"metrics/{self.chemical_space.dataset_name}/{self.date_str}/metrics_{self.date_str}.pkl", "wb+") as f:
            pickle.dump(self.metrics, f)
        
    def save_optimization_run_metrics(self, key=None)->None:
        if key == None:
            key = self.optimization_runs
        item = self.metrics[key]
        os.makedirs(f'metrics/{self.chemical_space.dataset_name}/{self.date_str}', exist_ok = True) 
        if not(type(item) is dict):
            with open(f"metrics/{self.chemical_space.dataset_name}/{self.date_str}/metrics_{key}.txt", "w+") as f:
                f.write(f"{key}: {item}\n")
            return
        with open(f"metrics/{self.chemical_space.dataset_name}/{self.date_str}/metrics_{key}.txt", "w+") as f:
            for key, val in self.metrics[key].items():
                f.write(f"{key}: {val}\n")

    def plot_metrics(self, repeat_no:int=None)->None:
        if repeat_no == None:
            # TODO: implement plotting of all repeats, show average and std/confidence intervals
            return
        metrics = self.metrics[repeat_no]
        points_measured = [self.seed_len]+[self.batch_size*i for i in range(len(metrics['accuracy']))]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
        ax1.plot(metrics['accuracy'])
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Batch')
        ax2.plot(metrics['precision'])
        ax2.set_title('Precision')
        ax2.set_xlabel('Batch')
        ax3.plot(metrics['recall'])
        ax3.set_title('Recall')
        ax3.set_xlabel('Batch')
        plt.savefig(f'metrics/{self.chemical_space.dataset_name}/{self.date_str}/metrics_{repeat_no}.png')
    
    def plot_set_preds(self, repeat_no:int=None)->None:
        ax1:plt.Axes
        ax2:plt.Axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        if repeat_no == None:
            # TODO
            return
        coverages = np.array(self.metrics[repeat_no]['coverages'])
        best_sets = self.metrics[repeat_no]['best_sets']
        # best_1 = coverages[:,0][:,0]
        # best_2 = coverages[:,1][:,0]
        best_3 = coverages[:,2][:,0]
        # best1_actual = [self.chemical_space.yield_surface.count_coverage(set) for set in best_sets[:,0][:,1]]
        # best2_actual = [self.chemical_space.yield_surface.count_coverage(set) for set in best_sets[:,1][:,1]]
        best_pred_sets = [s[0] for s in [sets[2] for sets in best_sets]]
        best3_actual = [self.chemical_space.yield_surface.count_coverage(set, self.metrics[repeat_no]['cutoff']) for set in best_pred_sets]
        xs = np.arange(len(best_3))
        ax1.plot(xs, best_3, xs, best3_actual)
        # plt.plot(xs, best_1, xs, best_2, xs, best_3, xs, best1_actual, xs, best2_actual, xs, best3_actual)
        ax1.set_title('Coverage of Best Predicted Sets')
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Coverage')
        ax1.legend(['Predicted Coverage', 'Actual Coverage'])
        set_ranks = [self.cond_to_rank_map[set] for set in best_pred_sets]
        ax2.plot(xs, set_ranks)
        ax2.set_title('Rank of Best Predicted Sets')
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('Rank')
        ax2.set_yscale('log')
        # plt.legend(['Best Single Condtion', 'Best Set of at most 2 Condtions', 'Best Set of at most 3 Conditions'])
        plt.savefig(f'metrics/{self.chemical_space.dataset_name}/{self.date_str}/coverages_{repeat_no}.png')