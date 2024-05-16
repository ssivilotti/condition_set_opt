import numpy as np
import itertools

# scoring functions for condition set coverage of a chemical space
# more complex scoring functions can be added
# the last argument is an np.ndarray of the maximum yield surface of the conditions
THRESHOLDED_COUNT = lambda a: lambda x: np.sum(x > a)
BINARY_COUNT = lambda x: np.sum(x)
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

    def get_condition_coverage(self, condition_set:tuple) -> np.ndarray:
        yield_coverage_surface = np.zeros(self.shape[len(condition_set[0]):])
        for cond in condition_set:
            s = self.get_condition_surface(cond)
            yield_coverage_surface = np.maximum(yield_coverage_surface, s)
        return yield_coverage_surface

    def count_coverage(self, condition_set:tuple, yield_threshold) -> int:
        return self.score_coverage(condition_set, yield_threshold, THRESHOLDED_COUNT(yield_threshold))
    
    def score_coverage(self, condition_set:tuple, scoring_function) -> float:
        yield_coverage_surface = self.get_condition_coverage(condition_set)
        return scoring_function(yield_coverage_surface)

    def best_condition_sets(self, condition_options:list, scoring_function, max_set_size:int=1, num_sets:int=10) -> tuple:
        possible_combos = list(itertools.combinations(condition_options, max_set_size))
        coverages = [self.score_coverage(set, scoring_function) for set in possible_combos]
        if max_set_size > 1:
            best_condition_sets_smaller, best_coverages_smaller = self.best_condition_sets(condition_options, scoring_function, max_set_size-1, num_sets)
            possible_combos = possible_combos + best_condition_sets_smaller
            coverages = coverages + best_coverages_smaller
        print(coverages)
        print(sum(self.mat))
        best_set_idxs = np.argsort(coverages)[-1*num_sets:]
        best_sets = [possible_combos[i] for i in best_set_idxs]
        best_coverages = [coverages[i] for i in best_set_idxs]
        return best_sets, best_coverages