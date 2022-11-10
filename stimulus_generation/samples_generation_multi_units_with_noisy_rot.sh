#!/bin/bash
#SBATCH --job-name=revision_samples_generation
#SBATCH --partition=day
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=5G
#SBATCH --time=1-
#SBATCH --mail-type=ALL
#SBATCH --mail-user=baohua.zhou@yale.edu

module load miniconda
source activate py37_dev

/usr/bin/time python3 samples_generation_multi_units_run_with_noisy_rot.py