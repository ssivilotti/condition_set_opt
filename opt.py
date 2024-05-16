# import libraries
import itertools
import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import qmc
# import matplotlib.pyplot as plt
import datetime as dt
from chemical_space import ChemicalSpace
from space_mat import SpaceMatrix

def optimize(chemical_space:ChemicalSpace, cutoff=.9, batch_size=8, max_experiments=1000):
    def convert_to_onehot(point):
        onehot = np.zeros(np.sum(chemical_space.shape))
        for i, p in enumerate(point):
            onehot[int(p + np.sum(chemical_space.shape[:i]))] = 1
        return onehot
    
    def convert_from_onehot(onehot):
        point = []
        num = 0
        shape_counter = 0
        for i in range(len(onehot)):
            if onehot[i] == 1:
                point.append(num)
            num += 1
            if num >= chemical_space.shape[shape_counter]:
                num = 0
                shape_counter += 1
        return point

    all_points = chemical_space.all_points
    all_points_one_hot = [convert_to_onehot(point) for point in all_points]

    # must be square of prime number
    initial_seed = 49

    # -1 if not measured, 0<=x<=1 if measured
    measured_yields = {}
    
    # save important information for the scoring function
    # reaction_difficulty = np.zeros()

    # metrics to save for benchmarking
    num_experiments_run:int = 0

    predicted_surface = SpaceMatrix(np.zeros(chemical_space.shape))

    # 'run the reaction and measure the yield'
    
    # Define Scoring functions
    # All scoring functions must:
        # 1. When any two rows are similar = bad
        # 2. When any two rows are complementary = good
        # 3. When any one row is low yield = bad
        # 4. When any one row is high yield = good
        # 5. Adding more reaction conditions = bad

    def reactionDifficultyScore(reaction:int) -> float:
        # for each reaction in reactions, look at how hard it is and reward harder reactions
        return 0.0

    # more successful reactions from the condition is good
    def conditionIndividualScore(condition:int) -> float:
        return 0.0

    # more diverse coverage from the set of conditions is good
    def conditionSetScore(conditions:list) -> float:
        return 0.0

    # Set up Active Learning Classifier over 2D space (reactants x conditions)
        # start with One-hot encoding + move to featurized later
    kernel = 1.0 * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kernel,random_state=0)

    # Main optimization algo
    # initial seed
        # kpp or latin hyper cube sampling
    KPP = 0
    LHS = 1
    def initialSeed(n, sampling_method:int = LHS) -> list:
        if sampling_method == KPP:
            return []
        elif sampling_method == LHS:
            # ensure n is a perfect quare of prime numbers
            sampler = qmc.LatinHypercube(d=len(chemical_space.shape), strength=2)
            seed = sampler.integers(l_bounds=np.zeros(len(chemical_space.shape), dtype=int), u_bounds=list(chemical_space.shape), n=n)
            return seed
        return []

    seed = []
    seed_attempts = 0
    seed_vals = 0
    seed_vals_sum = 0
    # ensures that seed has at least one successful reaction and at least one unsuccessful reaction
    while seed_attempts < 10 and (seed_vals_sum < 1 or seed_vals_sum >= len(seed)):
        seed = initialSeed(initial_seed)
        seed_vals = np.array([chemical_space.measure_reaction_yield(seed[i]) for i in range(len(seed))])
        seed_vals_sum = np.sum(seed_vals)
        seed_attempts += 1
        # print(seed_attempts)
    print(seed)

    x = None
    y = None

    point_certainty = .5
    next_points = np.array(seed)

    # (len(all_points) * .1)

    metrics = {'accuracy': [], 'precision': [], 'recall': []}

    best_set = []
    last_change = 0
    coverage = 0

    while (max(point_certainty, 1 - point_certainty) < .7 or (num_experiments_run < initial_seed + batch_size)) and (num_experiments_run < max_experiments) and (last_change < 10):
        # measure yields for uncertain points
        measurement = np.array([chemical_space.measure_reaction_yield(next_points[i]) > cutoff for i in range(len(next_points))])
        measured_yields.update({tuple(next_points[i]): measurement[i] for i in range(len(next_points))})
        # print(f"measurement: {measurement}")
        if (x is None) and (y is None):
            x = [convert_to_onehot(point) for point in next_points]
            y = measurement
        else:
            x = np.append(x, [convert_to_onehot(point) for point in next_points], axis=0)
            y =  np.append(y, measurement, axis=0)

        num_experiments_run += len(measurement)
        gpc.fit(x, y)
        # Active learning
            # add measurements to Active Learning Classifier
        # Bayesian optimization
            # get set of conditions suggestion from classifier knowledge
        # Score
        # Get next most uncertain points
            # get certainty for all points and suggest the most uncertain (something for not doing similar reactions?)
        uncertainty = gpc.predict_proba(np.array(all_points_one_hot))

        predicted_surface = SpaceMatrix(np.array([uncertainty[i][1] > .5 for i in range(len(uncertainty))]).reshape(chemical_space.shape, order='C'))
        
        accuracy, precicion, recall = chemical_space.score_classifier_prediction(uncertainty, cutoff)
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precicion)
        metrics['recall'].append(recall)

        predicted_set, coverage = predicted_surface.best_condition_sets(chemical_space.all_condtions, True, 3, 1)
        predicted_set = predicted_set[0]
        coverage = coverage[0]
        if predicted_set != best_set:
            best_set = predicted_set
            last_change = 0
        else:
            last_change += 1

        uncertainty_order = np.argsort(abs(uncertainty.T[0] - .5))

        next_points = []
        next_point = all_points[uncertainty_order[0]]
        point_uncertainties = []
        i = 0
        while len(next_points) < batch_size and (i+1) < len(uncertainty_order):
            while (tuple(next_point) in measured_yields) and (i+1) < len(uncertainty_order):
                i += 1
                next_point = all_points[uncertainty_order[i]]
            next_points.append(next_point)
            point_uncertainties.append(uncertainty[uncertainty_order[i]][0])
            i += 1
            if i < len(uncertainty_order):
                next_point = all_points[uncertainty_order[i]]
        # print(point_uncertainties)
        # print(next_points)
        point_certainty = np.average(point_uncertainties)
        print(f"uncertainty of {next_points}: {point_certainty}")
        print(num_experiments_run)

    # best_set = bestReactionSet(2)
    
    date_str = f"{dt.datetime.now()}"
    date_str = date_str[:10] +"_"+ date_str[11:19]
    with open(f"metrics/metrics_{date_str}.txt", "w") as f:
        f.write(f"accuracy: {metrics['accuracy']}\n")
        f.write(f"precision: {metrics['precision']}\n")
        f.write(f"recall: {metrics['recall']}\n")
        f.write(f"best_set: {best_set}\n")
        # f.write(f"predicted_surface: {predicted_surface}\n")

    # print(x)
    # print(uncertainty)
    # print(metrics)

    print(f"best_set: {best_set}, num_experiments_run: {num_experiments_run}, coverage: {coverage}")

# run the optimization
# optimize((30, 20, 10), dataset_file='datasets/correlated_toy_30x20x10.csv', cutoff=.5, batch_size=3)
aryl_scope = ChemicalSpace(['electrophile_id', 'nucleophile_id'], ['ligand_name'], 'datasets/Good_Datasets/aryl-scope-ligand.csv')
optimize(aryl_scope, cutoff=40, batch_size=15, max_experiments=100)