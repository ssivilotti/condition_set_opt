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
from learners.al_classifier_david import ALClassifierD
from tools.functions import convert_to_onehot, convert_point_to_idx
from learners.al_classifier_modified import ALClassifier
import time

# Learners
# Active Learning Classifier
EXPLORE = 0
# Active Learning Classifier with Tuned Aquistion Function
EXPEXP = 1
EXPLOIT = 2
# Random Selection of Next Points
RAND = -1

# handles featurization, chosing optimal sets and communication between the active learning algorithm and the chemical space
class Controller:
    def __init__(self, chemical_space:ChemicalSpace, yield_cutoff:float=None, batch_size:int=10, max_experiments:int=1000, max_set_size:int=3, learner_type:int=EXPEXP, early_stopping=True, output_dir='.', num_cpus=None) :
        self.chemical_space = chemical_space
        if yield_cutoff == None:
            return
        self.output_dir = output_dir
        self.cutoff = yield_cutoff
        self.batch_size = batch_size
        self.max_experiments = max_experiments
        self.date_str = dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S_%f')
        self.optimization_runs = 0
        self.max_set_size = max_set_size
        self.early_stopping = early_stopping
        self.config = {'max_experiments': self.max_experiments, 'batch_size': self.batch_size, 'cutoff': self.cutoff, 'learner_type': learner_type, 'date': self.date_str, 'max_set_size': self.max_set_size, 'early_stopping': early_stopping}
        self.scoring_function =  THRESHOLDED_COUNT(np.prod(chemical_space.shape[chemical_space.conditions_dim:]))(.5)
        self.metrics = {}
        if chemical_space.descriptors == None:
            self.all_points_featurized = [convert_to_onehot(self.chemical_space.shape, point) for point in chemical_space.all_points]
        self.cond_to_rank_map = chemical_space.yield_surface.rank_conditions(chemical_space.all_conditions, max_set_size, yield_cutoff)
        self.init_learner(learner_type)
    
    def init_learner(self, learner_type)->None:
        if learner_type == EXPLORE:
            self.learner = ALClassifierBasic(self.chemical_space.shape, self.num_cpus)
        if learner_type == RAND:
            self.learner = ALClassifierRandomSelection(self.chemical_space.shape, self.num_cpus)
        if learner_type == EXPEXP:
            self.learner = ALClassifier(self.chemical_space.shape, self.chemical_space.all_conditions, self.max_set_size, cpus=self.num_cpus)
        if learner_type == EXPLOIT:
            self.learner = ALClassifier(self.chemical_space.shape, self.chemical_space.all_conditions, self.max_set_size, alpha_init_fun=(lambda x: np.zeros(x)), cpus=self.num_cpus)
        else:
            raise ValueError("Invalid learner type input")

    def load_from_pkl(self, pickle_config_filepath, pickle_metrics_filepath=None):
        '''
        configures the controller from a pickle file and loads in the associated metrics
        @params:
        pickle_config_filepath: str, the path to the pickle file containing the configuration of the optimizer
        pickle_metrics_filepath: str, the path to the pickle file containing the metrics of the optimizer
        '''
        with open(pickle_config_filepath, 'rb') as f:
            self.config = pickle.load(f)
        if pickle_metrics_filepath != None:
            with open(pickle_metrics_filepath, 'rb') as f:
                self.metrics = pickle.load(f)
        else:
            self.metrics = {}
        self.cutoff = float(self.config['cutoff'])
        self.batch_size = int(self.config['batch_size'])
        self.max_experiments = int(self.config['max_experiments'])
        self.date_str = str(self.config['date'])
        self.max_set_size = int(self.config['max_set_size'])
        self.optimization_runs = len(self.metrics)
        self.early_stopping = self.config['early_stopping']
        self.scoring_function =  THRESHOLDED_COUNT(np.prod(self.chemical_space.shape[self.chemical_space.conditions_dim:]))(.5)
        if self.chemical_space.descriptors == None:
            self.all_points_featurized = [convert_to_onehot(self.chemical_space.shape, point) for point in self.chemical_space.all_points]
        self.cond_to_rank_map = self.chemical_space.yield_surface.rank_conditions(self.chemical_space.all_conditions, self.max_set_size, self.cutoff)
        self.init_learner(self.config['learner_type'])
    
    def load_pkl_metrics (self, pickle_metrics_filepath):
        '''
        sets the metrics of the controller to the metrics from the pickle file
        '''
        with open(pickle_metrics_filepath, 'rb') as f:
            self.metrics = pickle.load(f)
        self.optimization_runs = len(self.metrics)
    
    def get_initial_seed(self, sampling_method:str = 'LHS') -> list:
        '''
        returns a list of n points to be used as the initial seed for the optimization, with at least one successful and one unsuccessful reaction
        @params:
        n: int, the number of points to return, for LHS n must be a perfect square of a prime number
        sampling_method: str, the method to use to sample the initial seed, either \"KPP\" or \"LHS\"
        '''
        # TODO: implement known points
        # if initial_seed == 0:
        #     return self.known_points, self.known_points, 0
        seed = []
        seed_attempts = 0
        seed_vals = []
        seed_vals_sum = 0
        # ensures that seed has at least one successful reaction and at least one unsuccessful reaction
        while seed_attempts < 10 and (seed_vals_sum < 1 or seed_vals_sum >= len(seed)):
            if sampling_method == 'LHS':
                sampler = qmc.LatinHypercube(d=len(self.chemical_space.shape))
                seed = sampler.integers(l_bounds=np.zeros(len(self.chemical_space.shape), dtype=int), u_bounds=list(self.chemical_space.shape), n=self.batch_size)
            else:
                return []
            seed_vals = np.array([self.chemical_space.measure_reaction_yield(seed[i]) for i in range(len(seed))])
            seed_vals_sum = np.sum(seed_vals)
            seed_attempts += 1
        return seed
    
    def optimize(self, save_to_file=False)->tuple:
        start_time = time.time()
        self.learner.reset()
        self.learner.initialize_model()

        num_experiments_run:int = 0

        seed = self.get_initial_seed()

        metrics = {'accuracy': [], 'precision': [], 'recall': [], 
                   'best_sets': [], 'coverages': [], 
                   'points_suggested': [seed], 'uncertainties':[[1.0 for i in range(len(seed))]]}

        x = None
        y = None

        next_points = np.array(seed)
        known_idxs = set([convert_point_to_idx(self.chemical_space.shape, point) for point in next_points])

        # for a classifier, the predicted surface is the probability of the positive class
        predicted_surface:SpaceMatrix
        best_set = []
        last_change = 0
        coverage = 0

        print(f"Startup time: {time.time() - start_time} seconds")

        last_measured_time = time.time()
        loop_time = time.time()

        while (num_experiments_run < self.max_experiments) and (not self.early_stopping or (not(self.learner.done) or (num_experiments_run < 4*(self.batch_size)))):
            # measure yields for uncertain points
            measurement = np.array([self.chemical_space.measure_reaction_yield(next_points[i]) >= self.cutoff for i in range(len(next_points))])
            # measured_yields.update({tuple(next_points[i]): measurement[i] for i in range(len(next_points))})

            if (x is None) and (y is None):
                x = [convert_to_onehot(self.chemical_space.shape, point) for point in next_points]
                y = measurement
            else:
                x = np.append(x, [convert_to_onehot(self.chemical_space.shape, point) for point in next_points], axis=0)
                y =  np.append(y, measurement, axis=0)

            num_experiments_run += len(measurement)
            last_measured_time = time.time()
            self.learner.fit(x, y)
            print(f"fit: {time.time() - last_measured_time} seconds")
            last_measured_time = time.time()

            all_points_uncertainty, predicted_surface, next_points, certainties = self.learner.suggest_next_n_points(np.array(self.all_points_featurized), self.batch_size, known_idxs)
            known_idxs.update(next_points)
            next_points = [self.chemical_space.all_points[i] for i in next_points]
            print(f"suggest next points: {time.time() - last_measured_time} seconds")
            last_measured_time = time.time()

            accuracy, precicion, recall = self.chemical_space.score_classifier_prediction(all_points_uncertainty, self.cutoff)
            print(f"score model: {time.time() - last_measured_time} seconds")
            last_measured_time = time.time()
            sets = predicted_surface.best_condition_sets(self.chemical_space.all_conditions, self.scoring_function, self.max_set_size, 10)
            print(f"best sets: {time.time() - last_measured_time} seconds")
            last_measured_time = time.time()
            predicted_set = sets[0]['set']
            coverage = sets[0]['coverage']
            if predicted_set != best_set:
                best_set = predicted_set
                last_change = 0
            else:
                last_change += 1
            
            
            metrics['accuracy'].append(accuracy)
            metrics['precision'].append(precicion)
            metrics['recall'].append(recall)
            metrics['best_sets'].append(sets['set'])
            metrics['coverages'].append(sets['coverage'])
            metrics['points_suggested'].append(next_points)
            metrics['uncertainties'].append(certainties)
            print(f"{num_experiments_run} experiments run")
            print(f"Loop Time: {time.time() - loop_time} seconds")
            loop_time = time.time()
            last_measured_time = time.time()

        metrics['num_experiments_run'] = num_experiments_run   

        self.metrics[self.optimization_runs] = metrics

        if save_to_file:
            self.save_metrics_to_pkl()
            self.plot_metrics(self.optimization_runs)
            self.plot_set_preds(self.optimization_runs)
            print(num_experiments_run)
        
        self.optimization_runs += 1

        return best_set, coverage
    
    def do_repeats(self, n_repeats:int):
        for i in range(n_repeats):
            print(f"starting {i}")
            self.optimize()
            print(f"Finished {i}")
            self.save_metrics_to_pkl()
            self.metrics.clear()
    
    def save_metrics_to_pkl(self)->None:
        os.makedirs(f'{self.output_dir}/metrics/{self.chemical_space.dataset_name}/{self.date_str}', exist_ok = True) 
        with open(f"{self.output_dir}/metrics/{self.chemical_space.dataset_name}/{self.date_str}/config_{self.date_str}.pkl", "wb+") as f:
            pickle.dump(self.config, f) 
        try:
            with open(f"{self.output_dir}/metrics/{self.chemical_space.dataset_name}/{self.date_str}/metrics_{self.date_str}.pkl", "rb") as f:
                metrics = pickle.load(f)
                self.metrics.update(metrics)
        except:
            pass
        with open(f"{self.output_dir}/metrics/{self.chemical_space.dataset_name}/{self.date_str}/metrics_{self.date_str}.pkl", "wb+") as f:
            pickle.dump(self.metrics, f)

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
        plt.savefig(f'{self.output_dir}/metrics/{self.chemical_space.dataset_name}/{self.date_str}/metrics_{repeat_no}.png')
    
    def plot_set_preds(self, repeat_no:int=None)->None:
        ax1:plt.Axes
        ax2:plt.Axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        if repeat_no == None:
            # TODO
            return
        else:
            coverages = np.array(self.metrics[repeat_no]['coverages'])
            best_sets = self.metrics[repeat_no]['best_sets']
        # best_1 = coverages[:,0][:,0]
        # best_2 = coverages[:,1][:,0]
        # best_3 = coverages[:,2][:,0]
        best_cov = coverages[:,0]
        # best1_actual = [self.chemical_space.yield_surface.count_coverage(set) for set in best_sets[:,0][:,1]]
        # best2_actual = [self.chemical_space.yield_surface.count_coverage(set) for set in best_sets[:,1][:,1]]
        best_pred_sets = [s[0] for s in best_sets]
        best3_actual = [self.chemical_space.yield_surface.count_coverage(set, self.cutoff) for set in best_pred_sets]
        xs = np.arange(len(best_cov))
        ax1.plot(xs, best_cov, xs, best3_actual)
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
        plt.savefig(f'{self.output_dir}/metrics/{self.chemical_space.dataset_name}/{self.date_str}/coverages_{repeat_no}.png') 

    