import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from space_mat import SpaceMatrix

class ChemicalSpace:
    def __init__(self, reactant_titles:list, condition_titles:list, data_file, target_title='yield') -> None:
        self.reactants_dim = len(reactant_titles)
        self.condtions_dim = len(condition_titles)
        self.yield_surface = SpaceMatrix(self._create_mat(data_file, condition_titles, reactant_titles, target_title))
        self.shape = self.yield_surface.shape
        self.all_points = list(itertools.product(*[range(s) for s in self.shape]))
        self._y_true = np.array([self.yield_surface[point] for point in self.all_points])
        self.all_condtions = list(itertools.product(*[range(s) for s in self.shape[:self.condtions_dim]]))

    def _create_mat(self, data_file, cond_titles:list, reactant_titles:list,  target_title)->np.ndarray:
        # Create a matrix of the data
        data = pd.read_csv(data_file)
        unique_rs = [data[r_title].unique() for r_title in reactant_titles]
        unique_conds = [data[cond_title].unique() for cond_title in cond_titles]
        self.titles = cond_titles + reactant_titles
        self.labels = unique_conds + unique_rs
        shape = [len(item) for item in self.labels]
        data_mat = np.zeros(tuple(shape))
        for idx, series in data.iterrows():
            point = [np.where(self.labels[i] == (series[title])) for i, title in enumerate(self.titles)]
            data_mat[tuple(point)] = series[target_title]
        return data_mat

    def measure_reaction_yield(self, point:list) -> float:
        return self.yield_surface[tuple(point)]
    
    def score_classifier_prediction(self, y_pred, cutoff) -> tuple:
        y_pred = y_pred.T[1] > .5
        y_true = self._y_true > cutoff
        return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred)

    def best_condition_sets(self, yield_threshold:float, max_set_size:int=1, num_sets:int=10) -> tuple:
        return self.yield_surface.best_condition_sets(self.all_condtions, yield_threshold, max_set_size, num_sets)
    
    def format_condition(self, condition:tuple) -> str:
        return ', '.join([f'{self.labels[i][c]}' for i, c in enumerate(condition)])

    def plot_conditions(self, file_name, conditions:tuple, yield_threshold=0) -> None:
        '''
        conditions: tuple of condtion tuples
        '''
        ylabel = self.titles[self.condtions_dim]
        xlabel = self.titles[self.condtions_dim + 1]
        yields = np.zeros(self.shape[self.condtions_dim:])
        for condition in conditions:
            yields = np.maximum(yields, self.yield_surface.get_condition_surface(condition))
        below_threshold = yields < yield_threshold
        yields[below_threshold] = 0
        plt.imshow(yields, vmin=0, vmax=100)
        plt.colorbar(label='% Yield')
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        condition_str = '; '.join([self.format_condition(c) for c in conditions])
        if len(conditions) == 1:
            condition_str = f"Condition {conditions[0]} Yield"
        else:
            condition_str = f"Conditions {condition_str} Max Yield"
        plt.title(f'{condition_str}', fontsize=20)
        plt.savefig(file_name)

    def plot_surface(self, dataset_name, zlabel = '% Yield', title='Highest Yield Across All Condtions') -> None:    
        ylabel = self.titles[self.condtions_dim]
        xlabel = self.titles[self.condtions_dim + 1]
        max_yield_surface = np.amax(self.yield_surface.T, axis=-1*(self.condtions_dim)).T
        plt.imshow(max_yield_surface)
        plt.colorbar(label=zlabel)
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        plt.title(title, fontsize=20)
        plt.savefig(f'{dataset_name}.png')
