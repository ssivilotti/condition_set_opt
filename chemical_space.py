import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ChemicalSpace:
    def __init__(self, reactant_titles:list, condition_titles:list, data_file, target_title='yield') -> None:
        self.reactants_dim = len(reactant_titles)
        self.condtions_dim = len(condition_titles)
        self.yield_surface = self.create_mat(data_file, condition_titles, reactant_titles, target_title)
        self.shape = self.yield_surface.shape
        self.all_points = list(itertools.product(*[range(s) for s in self.shape]))
        self._y_true = np.array([self.yield_surface[point] for point in self.all_points])

    def create_mat(self, data_file, cond_titles:list, reactant_titles:list,  target_title)->np.ndarray:
        # Create a matrix of the data
        data = pd.read_csv(data_file)
        unique_rs = [data[r_title].unique() for r_title in reactant_titles]
        unique_conds = [data[cond_title].unique() for cond_title in cond_titles]
        titles = cond_titles + reactant_titles
        self.labels = unique_conds + unique_rs
        shape = [len(item) for item in self.labels]
        data_mat = np.zeros(tuple(shape))
        # for idx in itertools.product(*[range(s) for s in shape]):
            # data_mat[idx] = data[(data[reactant_titles[0]] == unique_rs[0][idx[0]]) & (data[reactant_titles[1]] == unique_rs[1][idx[1]]) & (data[cond_titles[0]] == unique_conds[0][idx[2])][target_title].values[0]
        for idx, series in data.iterrows():
            point = [np.where(self.labels[i] == (series[title])) for i, title in enumerate(titles)]
            data_mat[tuple(point)] = series[target_title]
        # for i, r1 in enumerate(unique_r1):
        #     for j, r2 in enumerate(unique_r2):
        #         for k, cond in enumerate(unique_cond):
        #             data_mat[k, i, j] = data[(data[r1_title] == r1) & (data[r2_title] == r2) & (data[cond_title] == cond)][target_title].values[0]
        # for i in range(len(unique_r1)):
        #     for j in range(len(unique_r2)):
        #         data_mat[i, j] = data[(data[r1_title] == r1[i]) & (data[r2_title] == r2[j])]['yield'].values[0]
        return data_mat

    def measure_reaction_yield(self, point:list) -> float:
        return self.yield_surface[tuple(point)]
    
    def score_classifier_prediction(self, y_pred, cutoff) -> tuple:
        y_pred = y_pred.T[1] > .5
        y_true = self._y_true > cutoff
        return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred)


