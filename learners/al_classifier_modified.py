import itertools
import numpy as np
from learners.learner import Classifier, YieldPred
from space_mat import THRESHOLDED_COUNT
from tools.functions import convert_point_to_idx, convert_idx_to_point
from joblib import Parallel, delayed
import time

class ALClassifier(Classifier):
    '''Active learning classifier that suggests the top n points with the highest uncertainty that haven't been measured yet to be measured next'''
    def __init__(self, space_shape:tuple, all_conditions, max_set_size, alpha_init_fun=(lambda x: np.linspace(0, 1, x, endpoint=True)), cpus=None):
        super().__init__(space_shape, cpus)
        self.all_conditions = all_conditions
        self.max_set_size = max_set_size
        self.alpha_init_fun = alpha_init_fun

    def exploit_helper(self, idx, sets):
        result = 0
        point = convert_idx_to_point(self.shape, idx)
        c = tuple(point[:len(sets['set'][0])])
        r = tuple(point[len(sets['set'][0]):])
        for s in sets:
            if c in s:
                if len(s) == 1:
                    result += s['coverage'] # maybe add a weight here
                else:
                    result += s['coverage'] * ((np.sum([1- self.predicted_surface[(cond + r)] for cond in s if c != cond]))/(len(s) - 1))
        return result
        # result = [0] * np.prod(self.shape)
        # if len(s) == 1:
        #     idx = 0
        #     for i, n in enumerate(s['set'][0]):
        #         idx += n
        #         if i < len(self.shape) - 1:   
        #             idx *= self.shape[i+1]
        #     return s['coverage'] # maybe add a weight here
        # else:
        #         all_reactants = list(itertools.product(*[range(s) for s in self.shape[len(s['set'][0]):]]))
        #         result += s['coverage'] * ((np.sum([1- self.predicted_surface[(cond + r)] for cond in s if c != cond]))/(len(s) - 1))
        # return result
        
    def suggest_next_n_points(self, X:np.ndarray, n:int, measured_indices:set)->list:
        '''next_points is a list of indices of the next points to be measured'''
        # TODO: pick different locations based on uncertainty, ensure batch doesn't cover only one area
        uncertainty = self.predict(X)
        
        explore_a = self.alpha_init_fun(n)
        exploit_a = 1 - explore_a

        # compute explore val for all points
        explore = 1 - (2*abs(uncertainty.T[0] - .5))
        
        # compute exploit val for all points
        exploit = [0] * len(uncertainty)
        t0 = time.time()

        # compute all coverages for all sets
        sets = self.predicted_surface.get_all_set_coverages(self.all_conditions, THRESHOLDED_COUNT(np.prod(self.shape[len(self.all_conditions[0]):]))(.5), self.max_set_size, num_cpus=self.cpus)

        print(f"Time to get all set coverages: {time.time() - t0} seconds")
        t0 = time.time()
        # compute exploit val for all points
        # all_reactants = list(itertools.product(*[range(s) for s in self.shape[len(self.all_conditions[0]):]]))
        # for i, s in enumerate(sets['set']):
        #     for reactant in all_reactants:
        #         if len(s) == 1:
        #             exploit[convert_point_to_idx(self.shape, s[0] + reactant)] += sets[i]['coverage'] # maybe add a weight here
        #         else:
        #             for cond in s:
        #                 exploit[convert_point_to_idx(self.shape, cond + reactant)] += sets[i]['coverage'] * ((np.sum([1- self.predicted_surface[(c + reactant)] for c in s if c != cond]))/(len(s) - 1))
        exploit = Parallel(n_jobs=self.cpus)(delayed(self.exploit_helper)(idx, sets) for idx in range(len(uncertainty)))

        exploit = exploit/(np.sum([len(self.all_conditions)**n for n in range(0, self.max_set_size)]))

        print(f"time to create exploit: {time.time() - t0}")
        t0 = time.time()

        #multiply by probablity of success for each point
        exploit = exploit * uncertainty.T[1]

        unmeasured_points = np.array([(i, explore[i], exploit[i]) for i in range(len(uncertainty)) if i not in measured_indices], dtype=[('idx', int), ('explore', float), ('exploit', float)])

        print(f"Time to create unmeasured points: {time.time() - t0} seconds")
        t0 = time.time()
        # multiply explore and exploit by alphas, find max idx along that axis (as long as two are not the same)
        opt_vals = np.array([unmeasured_points['explore']]).T@np.array([explore_a]) + np.array([unmeasured_points['exploit']]).T@np.array([exploit_a])

        print(f"Time to create opt_vals: {time.time() - t0} seconds")
        t0 = time.time()
        top_k_idxs = np.argpartition(opt_vals, -n, axis=0)[-n:]
        print(f"Time to get top k idxs (partition): {time.time() - t0} seconds")
        t0 = time.time()

        top_k_idxs = np.diagonal(top_k_idxs[np.argsort([[opt_vals[idxs[i]][i] for i in range(len(explore_a))] for idxs in top_k_idxs], axis=0, kind='stable')], axis1=1, axis2=2)
        print(f"Time to get top k idxs (sorted): {time.time() - t0} seconds")
        t0 = time.time()

        # get unique points for all alpha values
        idxs = top_k_idxs[-1]
        best_idxs = np.full(n, -1, dtype=int)
        for i in range(len(top_k_idxs)):
            idx = idxs[i]
            if idxs[i] in best_idxs:
                j = 2
                while top_k_idxs[-1*j][i] in best_idxs:
                    j += 1
                idx = top_k_idxs[-j][i]
            best_idxs[i] = idx

        next_idxs = unmeasured_points[best_idxs]['idx']
        point_uncertainties = unmeasured_points[best_idxs]['explore']

        print(f"Time to get best idxs: {time.time() - t0} seconds")

        return uncertainty, self.predicted_surface, next_idxs, point_uncertainties
    

