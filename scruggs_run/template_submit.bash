#!/bin/bash
#$ -N {name}
#$ -cwd
#$ -o {name}.out
#$ -e {name}.err
#$ -pe orte {num_cpus}
#$ -q all.q
#$ -l hostname=compute-0-3.local 
# module load openmpi/3.1.4
module load anaconda/3-2022.05

cd {dir}

cd ..

conda run -n cond_opt_env python scruggs_downselect.py {dataset} {num_cpus} {batch_size} {opt_init} {n_repeats} {learner_type} {model_type} {stochastic_batch_num} {cond_subset}