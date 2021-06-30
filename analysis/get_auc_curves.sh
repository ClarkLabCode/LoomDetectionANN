#!/bin/bash
#SBATCH --job-name=get_auc_curves
#SBATCH --partition=day
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5G
#SBATCH --time=1-
#SBATCH --mail-type=ALL
#SBATCH --mail-user=baohua.zhou@yale.edu

module load miniconda
source activate py37_dev

/usr/bin/time python3 get_auc_curves.py