# import libraries
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import qmc
import matplotlib.pyplot as plt
import datetime as dt
# from chemical_space import ChemicalSpace


# define inputs
# num_conditions:int = 10
# num_reactant_combos:int = 5

def optimize(shape, dataset_file=None, data_mat=None, cutoff=.9, batch_size=8):
    def convert_to_onehot(point):
        onehot = np.zeros(np.sum(shape))
        for i, p in enumerate(point):
            onehot[int(p + np.sum(shape[:i]))] = 1
        return onehot
    
    def convert_from_onehot(onehot):
        point = []
        num = 0
        shape_counter = 0
        for i in range(len(onehot)):
            if onehot[i] == 1:
                point.append(num)
            num += 1
            if num >= shape[shape_counter]:
                num = 0
                shape_counter += 1
        return point
    
    # read in dataset
    yield_surface = np.zeros(shape)

    all_points_iter = itertools.product(*(range(i) for i in shape))
    all_points = [list(point) for point in all_points_iter]
    all_points_one_hot = [convert_to_onehot(point) for point in all_points]
    
    if dataset_file:
        try:
            # TODO: header of file = r1, r2, c1, c2, ..., yield
            with open(dataset_file, 'r') as f:
                    f.readline()
                    line = f.readline()
                    while line:
                        rxn = line.split(',') # rxn = [condition, reactions, yield]
                        indices = [int(i) for i in rxn[:-1]]
                        yield_surface[tuple(indices)] = float(rxn[-1]) 
                        line = f.readline()
        except:
            raise Exception('Error reading datsaset file')
    elif data_mat is not None:
        yield_surface = data_mat
    else:
        raise Exception('No dataset provided')

    yield_classifier_surface = yield_surface > cutoff

    # print(yield_classifier_surface)

    # -1 if not measured, 0<=x<=1 if measured
    measured_yields = {}
    
    # save important information for the scoring function
    reaction_difficulty = np.zeros(shape)

    # metrics to save for benchmarking
    num_experiments_run:int = 0

    y_true_all_points = np.array([yield_surface[tuple(point)] > cutoff for point in all_points])

    predicted_surface = np.zeros(shape)

    # 'run the reaction and measure the yield'
    def measureReactionYield(point:list) -> float:
        # print(f"condition: {condition}, reactions: {reactions}")
        # indices = conditions[:].extend(reactions[:])
        # reactions[:]
        # indices.insert(0, condition)
        # print(f"indices: {indices}")
        measurement = yield_surface[tuple(point)]
        # print(f"measurement: {measurement}")
        # for reaction in reactions:
        #     measurement = measurement[reaction]
        # imbue some randomness?
        measured_yields[tuple(point)] = measurement
        # num_experiments_run = num_experiments_run + 1
        return measurement
    
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

    def count_coverage_for_set(set):
        covered = predicted_surface[set[0]]
        for i in range(1,len(set)):
            covered = np.logical_or(covered, predicted_surface[set[i]])
        return np.sum(covered, axis=(0,1))

    def bestReactionSet(n = 2):
        possible_combos = list(itertools.combinations(range(shape[0]), n))
        coverages = [count_coverage_for_set(set) for set in possible_combos]
        print(coverages)
        best_set = possible_combos[np.argmax(coverages)]
        return best_set


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
            sampler = qmc.LatinHypercube(d=len(shape), strength=2)
            seed = sampler.integers(l_bounds=np.zeros(len(shape), dtype=int), u_bounds=list(shape), n=n)
            return seed
        return []
    # measure

    seed = []
    seed_attempts = 0
    seed_vals = 0
    seed_vals_sum = 0
    while seed_attempts < 10 and (seed_vals_sum < 1 or seed_vals_sum >= len(seed)):
        seed = initialSeed(49)
        seed_vals = np.array([yield_classifier_surface[tuple(seed[i])] for i in range(len(seed))])
        seed_vals_sum = np.sum(seed_vals)
        seed_attempts += 1
        print(seed_attempts)
    print(seed)

    # for all reactions in seed, measure yield
    # seed_X = np.array(seed)
    # seed_y = np.array([measureReactionYield(seed[i][0], seed[i][1]) > .5 for i in range(len(seed))])
    # num_experiments_run += len(seed_y)
    # print(seed_y)

    # fit classifier to measured yields
    # gpc.fit(seed_X, seed_y)

    # print(seed_y)
    # print(gpc.predict(seed_X))

    x = None
    y = None

    point_certainty = .5
    next_points = np.array(seed)

    # (len(all_points) * .1)

    metrics = {'accuracy': [], 'precision': [], 'recall': []}

    best_set = []
    last_change = 0

    def score_AL(uncertainty_pred, y_true=y_true_all_points):
        y_pred = uncertainty_pred.T[1] > .5
        metrics['accuracy'].append(accuracy_score(y_true, y_pred))
        metrics['precision'].append(precision_score(y_true, y_pred))
        metrics['recall'].append(recall_score(y_true, y_pred))

    while ((max(point_certainty, 1 - point_certainty) < .7) or num_experiments_run < 75) and (num_experiments_run < 900) and (last_change < 10):
        # measure yields for uncertain points
        measurement = np.array([measureReactionYield(next_points[i]) > cutoff for i in range(len(next_points))])
        print(f"measurement: {measurement}")
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

        predicted_surface = np.array([uncertainty[i][1] > .5 for i in range(len(uncertainty))]).reshape(shape, order='C')
        
        score_AL(uncertainty)

        predicted_set = bestReactionSet(2)
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
            # uncertainty_order = np.delete(uncertainty_order, 0)
            # TODO: make a dictionary of measurements for faster lookup
            # np.isin(np.array(all_points[uncertainty_order[i]]), x)
            while (tuple(next_point) in measured_yields) and (i+1) < len(uncertainty_order):
                i += 1
                next_point = all_points[uncertainty_order[i]]
            next_points.append(next_point)
            point_uncertainties.append(uncertainty[uncertainty_order[i]][0])
            i += 1
            if i < len(uncertainty_order):
                next_point = all_points[uncertainty_order[i]]
            # else:
            #     break
        print(point_uncertainties)
        print(next_points)
        point_certainty = np.average(point_uncertainties)
        # next_point = [all_points[uncertainty_order[0]]]
        # point_certainty = uncertainty[uncertainty_order[0]][0]
        print(f"uncertainty of {next_points}: {point_certainty}")
        print(num_experiments_run)

    # best_set = bestReactionSet(2)
    
    date_str = f"{dt.datetime.now()}"
    date_str = date_str[:10] +"_"+ date_str[11:19]
    with open(f"metrics_{date_str}.txt", "w") as f:
        f.write(f"accuracy: {metrics['accuracy']}\n")
        f.write(f"precision: {metrics['precision']}\n")
        f.write(f"recall: {metrics['recall']}\n")
        f.write(f"best_set: {best_set}\n")
        f.write(f"predicted_surface: {predicted_surface}\n")

    print(x)
    # print(uncertainty)
    print(metrics)

    print(f"best_set: {best_set}, num_experiments_run: {num_experiments_run}")

def create_mat(data,cond_title, r1_title, r2_title,  target_title):
    # Create a matrix of the data
    unique_r1 = data[r1_title].unique()
    unique_r2 = data[r2_title].unique()
    unique_cond = data[cond_title].unique()
    data_mat = np.zeros((len(unique_cond), len(unique_r1), len(unique_r2)))
    for i, r1 in enumerate(unique_r1):
        for j, r2 in enumerate(unique_r2):
            for k, cond in enumerate(unique_cond):
                data_mat[k, i, j] = data[(data[r1_title] == r1) & (data[r2_title] == r2) & (data[cond_title] == cond)][target_title].values[0]
    # for i in range(len(unique_r1)):
    #     for j in range(len(unique_r2)):
    #         data_mat[i, j] = data[(data[r1_title] == r1[i]) & (data[r2_title] == r2[j])]['yield'].values[0]
    return data_mat

# run the optimization
# optimize((30, 20, 10), dataset_file='datasets/correlated_toy_30x20x10.csv', cutoff=.5, batch_size=3)
aryl_scope = pd.read_csv('datasets/Good_Datasets/aryl-scope-ligand.csv')
aryl_scope_mat = create_mat(aryl_scope, 'ligand_name', 'electrophile_id', 'nucleophile_id', 'yield')
optimize(aryl_scope_mat.shape, data_mat=aryl_scope_mat, cutoff=40, batch_size=15)