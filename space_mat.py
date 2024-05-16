import numpy as np
import itertools

class SpaceMatrix:
    def __init__(self, matrix:np.ndarray) -> None:
        self.mat = matrix
        self.shape = self.mat.shape

    def __getitem__(self, key:tuple) -> np.ndarray:
        return self.mat[key]
    
    def __setitem__(self, key:tuple, value:np.ndarray) -> None:
        self.mat[key] = value

    def __iter__(self):
        return iter(self.mat)

    def get_condition_surface(self, condition:tuple) -> np.ndarray:
        s = self.mat[condition[0]]
        for c in condition[1:]:
            s = s[c]
        return s

    def count_coverage(self, condtion_set:tuple, yield_threshold) -> int:
        yield_coverage_surface = np.zeros(self.shape[len(condtion_set[0]):])
        for cond in condtion_set:
            s = self.get_condition_surface(cond)
            yield_coverage_surface = np.maximum(yield_coverage_surface, s)
        return np.sum(yield_coverage_surface >= yield_threshold)

    def best_condition_sets(self, condition_options:list, yield_threshold, max_set_size:int=1, num_sets:int=10) -> tuple:
        possible_combos = list(itertools.combinations(condition_options, max_set_size))
        coverages = [self.count_coverage(set, yield_threshold) for set in possible_combos]
        if max_set_size > 1:
            best_condition_sets_smaller, best_coverages_smaller = self.best_condition_sets(condition_options, yield_threshold, max_set_size-1, num_sets)
            possible_combos = possible_combos + best_condition_sets_smaller
            coverages = coverages + best_coverages_smaller
        print(coverages)
        print(sum(self.mat))
        best_set_idxs = np.argsort(coverages)[-1*num_sets:]
        best_sets = [possible_combos[i] for i in best_set_idxs]
        best_coverages = [coverages[i] for i in best_set_idxs]
        return best_sets, best_coverages