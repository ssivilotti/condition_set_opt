from controller import Controller
from chemical_space import ChemicalSpace
import sys
from pathlib import Path
import os

dataset = sys.argv[1]
num_cpus = sys.argv[2]
batch_size = int(sys.argv[3])
opt_init = int(sys.argv[4])
n_repeats = int(sys.argv[5])
learner_type = int(sys.argv[6])
model_type = sys.argv[7]
stochastic_batch_num = None if sys.argv[8] == -1 else int(sys.argv[8])
cond_subset = int(sys.argv[9])


optimizer:Controller

original_path = Path('/home/sofials3/cond_opt')

os.chdir(original_path)


# load in data
if dataset == 'aryl_scope':
    aryl_scope = ChemicalSpace(['ligand_name'], ['electrophile_id', 'nucleophile_id'], f'{original_path}/datasets/real_datasets/aryl-scope-ligand.csv')
    aryl_scope.titles = ['Ligand', 'Electrophile', 'Nucleophile']
    optimizer = Controller(aryl_scope,21.66, batch_size=batch_size, max_experiments=1000, early_stopping=False, learner_type=learner_type, output_dir=original_path, num_cpus=num_cpus, stochastic_cond_num=stochastic_batch_num, model_type=model_type)
elif dataset == 'borylation':
    borylation = ChemicalSpace(['ligand_name', 'solvent'], ['electrophile_id'], f'{original_path}/datasets/real_datasets/borylation.csv')
    borylation.titles = ['Ligand', 'Solvent', 'Electrophile']
    optimizer = Controller(borylation, 77.24, batch_size=batch_size, max_experiments=1000, early_stopping=False, learner_type=learner_type, output_dir=original_path, num_cpus=num_cpus, stochastic_cond_num=stochastic_batch_num, model_type=model_type, max_set_size=4)
elif dataset == 'deoxy':
    deoxy = ChemicalSpace(['base_name', 'fluoride_name'], ['substrate_name'], f'{original_path}/datasets/real_datasets/deoxyf.csv')
    deoxy.titles = ['Base', 'Fluoride', 'Substrate']
    optimizer = Controller(deoxy, 50, batch_size=batch_size, max_experiments=500, early_stopping=False, learner_type=learner_type, output_dir=original_path, num_cpus=num_cpus, stochastic_cond_num=stochastic_batch_num, model_type=model_type)
elif dataset == 'buchwald_hartwig':
    cond_map = [['Catalyst'], ['Solvent'], ['Base'], ['Solvent','Base'], ['Catalyst','Base'], ['Catalyst','Solvent']]
    buchwald_hartwig = ChemicalSpace(cond_map[cond_subset], ['Amine','Bromide'], f'{original_path}/datasets/real_datasets/buchwald-hartwig.csv')
    buchwald_hartwig.dataset_name += '_' + '_'.join(cond_map[cond_subset])
    optimizer = Controller(buchwald_hartwig, 26.89, batch_size=batch_size, max_experiments=4500, early_stopping=False, learner_type=learner_type, output_dir=original_path, num_cpus=num_cpus, stochastic_cond_num=stochastic_batch_num, model_type=model_type)
else:
    raise ValueError('Invalid dataset')

optimizer.optimization_runs=opt_init
    
optimizer.do_repeats(n_repeats)

