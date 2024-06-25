import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from chemical_space import ChemicalSpace
import seaborn as sns

def convert_to_onehot(shape: tuple, point:list)-> list:
    onehot = np.zeros(np.sum(shape))
    for i, p in enumerate(point):
        onehot[int(p + np.sum(shape[:i]))] = 1
    return onehot

def convert_from_onehot(shape: tuple, onehot: int)-> list:
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

def convert_point_to_idx(shape:tuple, point:list)-> int:
    idx = 0
    for i, n in enumerate(point):
        idx += n
        if i < len(shape) - 1:
            idx *= shape[i+1]
    return idx

def convert_idx_to_point(shape:tuple, idx:int)-> list:
    point = [0]*len(shape)
    for i in range(len(shape)-1, -1, -1):
        point[i] = (idx % shape[i])
        idx = idx // shape[i]
    return point

def compare_spaces(spaces:list[ChemicalSpace])->None:
    fig, (ax3, ax0, ax1, ax2) = plt.subplots(1, 4, figsize=(20, 5))
    for cs in spaces:
        yields, coverage = cs.get_yield_coverage()
        all_yields, successful_rxns = cs.get_yield_success()
        cond_yield, cond_coverage = cs.get_individual_conditions_coverage()
        diff = [0]*(len(yields)*2)
        diff_yield = [0]*(len(yields)*2)
        diff_total_cov = [0]*(len(yields)*2)
        j, k = 0, 0
        for i in range(len(diff)):
            if j >= len(cond_yield) or (k < len(yields) and yields[k] > cond_yield[j]):
                diff_yield[i] = yields[k]
                diff[i] = coverage[k] - cond_coverage[j-1]
                diff_total_cov[i] = coverage[k]
                k += 1
            else:
                diff_yield[i] = cond_yield[j]
                if k >= len(yields):
                    diff[i] = 1 - cond_coverage[j]
                    diff_total_cov[i] = 1
                else:
                    diff[i] = coverage[k] - cond_coverage[j]
                    diff_total_cov[i] = coverage[k]
                j += 1
        ax0.plot(diff_yield, diff, label=cs.dataset_name)
        ax1.plot(yields, coverage, label=cs.dataset_name)
        ax2.plot(all_yields, successful_rxns, label=cs.dataset_name)
        ax3.plot(diff_total_cov, diff, label=cs.dataset_name)
    ax3.set_xlabel('Total Coverage')
    ax3.set_ylabel('Difference in Total and Individual Condition Coverage')
    ax3.legend()
    ax3.set_title('Total Coverage vs Difference')
    ax0.set_xlabel('Yield')
    ax0.set_ylabel('Difference in Total and Individual Condition Coverage')
    ax0.legend()
    ax0.set_title('Yield vs Gap between Individual and Total Coverage')
    ax1.set_xlabel('Yield')
    ax1.set_ylabel('Coverage')
    ax1.legend()
    ax1.set_title('Yield vs Coverage')
    ax2.set_xlabel('Yield')
    ax2.set_ylabel('Successful Reactions')
    ax2.legend()
    ax2.set_title('Yield vs Successful Reactions')
    plt.show()

def plot_learner_metrics(chemical_space:ChemicalSpace, metric_filepaths:list, names:list, abbreviations:list=None, output_file_path:str=None)->None:
        assert len(metric_filepaths) == len(names), 'Number of filepaths and names must be the same'
        # TODO: plot with confidence intervals (with sns)
        if abbreviations is None:
            abbreviations = names
        pred_covs = [f'{abbreviations[i]} Predicted Coverage' for i in range(len(names))]
        actual_covs = [f'{abbreviations[i]} Actual Coverage' for i in range(len(names))]
        config_file = metric_filepaths[0].replace('metrics_', 'config_')
        with open(config_file, 'rb') as f:
            config = pickle.load(f)
            max_set_size = config['max_set_size']
            yield_cutoff = config['cutoff']
        cond_to_rank_map = chemical_space.yield_surface.rank_conditions(chemical_space.all_conditions, max_set_size, yield_cutoff)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        df = pd.DataFrame(columns=['Batch', 'Predicted Coverage', 'Actual Coverage', 'Rank', 'Accuracy', 'Precision', 'Recall', 'name', 'pred_name', 'actual_name'])
        accuracy, precision, recall = [], [], []
        m = {}
        for i, metric_file in enumerate(metric_filepaths):
            with open(metric_file, 'rb') as f:
                m = pickle.load(f)
            for j, metric in m.items():
                accuracy.append(metric['accuracy'])
                precision.append(metric['precision'])
                recall.append(metric['recall'])
                coverages = np.array(metric['coverages'])
                best_sets = metric['best_sets']
                best_3 = coverages[:,0]
                best_pred_sets = [s[0] for s in best_sets]
                best3_actual = [chemical_space.yield_surface.count_coverage(set, yield_cutoff) for set in best_pred_sets]
                xs = np.arange(len(best_3))
                set_ranks = [cond_to_rank_map[set] for set in best_pred_sets]
                # append to df
                df = pd.concat([df, pd.DataFrame({'Batch':xs, 'Predicted Coverage':best_3, 'Actual Coverage': best3_actual, 'Rank':set_ranks, 'Accuracy':metric['accuracy'], 'Precision':metric['precision'], 'Recall':metric['recall'], 'name':[names[i] for _ in range(len(xs))], 'pred_name':[pred_covs[i] for _ in range(len(xs))], 'actual_name':[actual_covs[i] for _ in range(len(xs))]})])
                # ax1.plot(xs, best_3, label=f'{abbreviations[i]} Predicted Coverage')
                # ax1.plot(xs, best3_actual, label=f'{abbreviations[i]} Actual Coverage')
                # ax2.plot(xs, set_ranks)
        
        cov_labels = [""]*(len(pred_covs)*2)
        for i in range(len(pred_covs)):
            cov_labels[i*2] = pred_covs[i]
            cov_labels[i*2+1] = actual_covs[i]

        g = sns.lineplot(data=df, x='Batch', y='Predicted Coverage', hue='pred_name', hue_order=pred_covs, ax=ax1)
        sns.lineplot(data=df, x='Batch', y='Actual Coverage', hue='actual_name', hue_order=actual_covs, ax=g, palette='dark')
        sns.lineplot(data=df, x='Batch', y='Rank', hue='name', ax=ax2)

        ax1.set_title('Coverage of Best Predicted Sets')
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Coverage')
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=handles, labels=labels)
        ax2.set_title('Rank of Best Predicted Sets')
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('Rank')
        ax2.set_yscale('log')
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles=handles, labels=labels)
        if output_file_path is not None:
            os.makedirs(output_file_path, exist_ok=True)
            plt.savefig(f'{output_file_path}/set_analysis.png') 
        
        fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(15,5))
        sns.lineplot(data=df, x='Batch', y='Accuracy',
            hue='name',
            hue_order=names,
            ax=ax3)
        sns.lineplot(data=df, x='Batch', y='Precision', 
            hue='name',
            hue_order=names,
            ax=ax4)
        sns.lineplot(data=df, x='Batch', y='Recall', 
            hue='name',
            hue_order=names,
            ax=ax5)
        # for i in range(len(accuracy)):
        #     ax3.plot(accuracy[i])
        #     ax4.plot(precision[i])
        #     ax5.plot(recall[i])
        ax3.set_title('Accuracy')
        ax3.set_xlabel('Batch')
        ax4.set_title('Precision')
        ax4.set_xlabel('Batch')
        ax5.set_title('Recall')
        ax5.set_xlabel('Batch')
        for ax in [ax3, ax4, ax5]:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=labels)
        # ax5.legend(names)
        if output_file_path is not None:
            plt.savefig(f'{output_file_path}/metrics.png')