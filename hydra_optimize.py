from chemical_space import ChemicalSpace
from controller import Controller, RAND, ALC
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    optimizer:Controller

    original_path = Path(get_original_cwd())

    # load in data
    if cfg.dataset == 'aryl_scope':
        aryl_scope = ChemicalSpace(['ligand_name'], ['electrophile_id', 'nucleophile_id'], f'{original_path}/datasets/real_datasets/aryl-scope-ligand.csv')
        aryl_scope.titles = ['Ligand', 'Electrophile', 'Nucleophile']
        optimizer = Controller(aryl_scope,21.66, batch_size=cfg.batch_size, max_experiments=1000, early_stopping=False, learner_type=cfg.learner_type, output_dir=original_path)
    elif cfg.dataset == 'borylation':
        borylation = ChemicalSpace(['ligand_name', 'solvent'], ['electrophile_id'], f'{original_path}/datasets/real_datasets/borylation.csv')
        borylation.titles = ['Ligand', 'Solvent', 'Electrophile']
        optimizer = Controller(borylation, 77.24, batch_size=cfg.batch_size, max_experiments=1000, early_stopping=False, learner_type=cfg.learner_type, output_dir=original_path)
    elif cfg.dataset == 'deoxy':
        deoxy = ChemicalSpace(['base_name', 'fluoride_name'], ['substrate_name'], f'{original_path}/datasets/real_datasets/deoxyf.csv')
        deoxy.titles = ['Base', 'Fluoride', 'Substrate']
        optimizer = Controller(deoxy, 50, batch_size=cfg.batch_size, max_experiments=500, early_stopping=False, learner_type=cfg.learner_type, output_dir=original_path)
    elif cfg.dataset == 'buchwald_hartwig':
        buchwald_hartwig = ChemicalSpace(['Catalyst','Solvent','Base'], ['Amine','Bromide'], f'{original_path}/datasets/real_datasets/buchwald-hartwig.csv')
        optimizer = Controller(buchwald_hartwig, 26.89, batch_size=cfg.batch_size, max_experiments=1000, early_stopping=False, learner_type=cfg.learner_type, output_dir=original_path)
    else:
        raise ValueError('Invalid dataset')
    
    optimizer.do_repeats(cfg.n_repeats)

if __name__ == "__main__":
    run()

