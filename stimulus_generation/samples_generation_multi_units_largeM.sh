#!/bin/bash
#SBATCH --job-name=largeM_samples_generation
#SBATCH --partition=day
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=3G
#SBATCH --time=1-
#SBATCH --mail-type=ALL
#SBATCH --mail-user=baohua.zhou@yale.edu

module load miniconda
source activate py37_dev

/usr/bin/time python3 samples_generation_multi_units_run_largeM.py