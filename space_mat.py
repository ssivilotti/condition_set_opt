import numpy as np
import itertools

# scoring functions for condition set coverage of a chemical space
# more complex scoring functions can be added
# the last argument is an np.ndarray of the maximum yield surface of the conditions
THRESHOLDED_COUNT = lambda a: lambda x: np.sum(x > a)
BINARY_COUNT = lambda x: np.sum(x)
class SpaceMatrix:
    def __init__(self, matrix:np.ndarray) -> None:
        '''
        @params
        matrix: np.ndarray, the matrix of the chemical space can be either yields or binary values, the shape should be the conditions followed by the reactants such that if n is the number of dimensions in a condition, shape[:n] should all relate to condition options and shape[n:] should all relate to reactant options

        attributes:
        mat: np.ndarray, the matrix of the chemical space
        shape: tuple, the shape of the matrix
        '''
        self.mat = matrix
        self.shape = self.mat.shape
        #TODO: implement caching of condition coverage

    '''Allow for indexing and slicing of the matrix'''
    def __getitem__(self, key:tuple) -> np.ndarray:
        return self.mat[key]
    
    def __setitem__(self, key:tuple, value:np.ndarray) -> None:
        self.mat[key] = value

    def __iter__(self):
        return iter(self.mat)

    def get_condition_surface(self, condition:tuple) -> np.ndarray:
        '''
        @params:
        condition: tuple of integers that represents a single condition in the chemical space
        
        returns:
        np.ndarray, the yield surface for the given condition over all reactant combinations
        '''
        s = self.mat[condition[0]]
        for c in condition[1:]:
            s = s[c]
        return s

    def get_condition_coverage(self, condition_set:tuple) -> np.ndarray:
        '''
        @params:
        condition_set: tuple of tuples of integers that represents a set of conditions in the chemical space

        returns:
        np.ndarray, the maximum yield surface for the given set of conditions over all reactant combinations
        '''
        yield_coverage_surface = np.zeros(self.shape[len(condition_set[0]):])
        for cond in condition_set:
            s = self.get_condition_surface(cond)
            yield_coverage_surface = np.maximum(yield_coverage_surface, s)
        return yield_coverage_surface

    def count_coverage(self, condition_set:tuple, yield_threshold:float) -> int:
        '''
        @params:
        condition_set: tuple of tuples of integers that represents a set of conditions in the chemical space
        yield_threshold: float, the minimum yield to count as a successful condition

        returns:
        int, the number of reactant combinations that meet the yield threshold for the given set of conditions
        '''
        return self.score_coverage(condition_set, THRESHOLDED_COUNT(yield_threshold))
    
    def score_coverage(self, condition_set:tuple, scoring_function) -> float:
        '''
        @params:
        condition_set: tuple of tuples of integers that represents a set of conditions in the chemical space
        scoring_function: function np.ndarray->float, a function that takes in the maximum yield surface of the conditions and returns a score

        returns:
        float, the score of the given set of conditions based on the scoring function
        '''
        yield_coverage_surface = self.get_condition_coverage(condition_set)
        return scoring_function(yield_coverage_surface)

    def best_condition_sets(self, condition_options:list, scoring_function, max_set_size:int=1, num_sets:int=10) -> tuple:
        '''
        @params:
        condition_options: list of tuples of integers, representing all possible or a subset of conditions in the chemical space
        scoring_function: function np.ndarray->float, a function that takes in the maximum yield surface of the conditions and returns a score
        max_set_size: int, the maximum number of conditions to include in a set
        num_sets: int, the number of sets to return

        returns: list of condition tuples, list of floats, the best condition sets and their corresponding scores in descending order of coverage
        ignores sets that contain a subset with the same coverage
        '''
        possible_combos = list(itertools.combinations(condition_options, max_set_size))
        coverages = [self.score_coverage(set, scoring_function) for set in possible_combos]
        if max_set_size > 1:
            best_condition_sets_smaller, best_coverages_smaller = self.best_condition_sets(condition_options, scoring_function, max_set_size-1, num_sets)
            possible_combos = possible_combos + best_condition_sets_smaller[::-1]
            coverages = coverages + best_coverages_smaller[::-1]
        set_idxs = np.array(coverages).argsort(kind = 'stable')
        best_set_idxs = np.zeros(num_sets, dtype=int)
        i = 0
        set_idx = len(set_idxs)-1
        sets_to_remove = []
        while i < num_sets:
            # if set does not contain a set already seen, add it
            unique_set = np.all([not (s <= set(possible_combos[set_idxs[set_idx]])) for s in sets_to_remove])
            if unique_set:
                # if set is smaller than max_set_size, add it to sets to remove
                if len(possible_combos[set_idxs[set_idx]]) < max_set_size:
                    sets_to_remove.append(set(possible_combos[set_idxs[set_idx]]))
                best_set_idxs[i] = set_idxs[set_idx]
                i += 1
            set_idx -= 1
            if set_idx > 0 and coverages[set_idxs[set_idx]] != coverages[set_idxs[set_idx + 1]]:
                sets_to_remove = []
        best_sets = [possible_combos[i] for i in best_set_idxs]
        best_coverages = [coverages[i] for i in best_set_idxs]
        return best_sets, best_coverages