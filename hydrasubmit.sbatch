#!/bin/bash
#
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodelist=ccc0309
#SBATCH --job-name=opt_main
#SBATCH --partition=mmli
#SBATCH --output=myjob%j.out
#SBATCH --error=myjob%j.err
##SBATCH --mail-user=sofials3@illinois.edu
##SBATCH --mail-type=END#
# End of embedded SBATCH options
#

module load anaconda3/2024.02-Feb

conda run -p /projects/mmli/sofials3/conda_envs/cond_opt python hydra_optimize.py --multirun
