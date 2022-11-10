#!/bin/bash
#SBATCH --job-name=samples_generation_grid
#SBATCH --partition=day
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=10G
#SBATCH --time=1-
#SBATCH --mail-type=ALL
#SBATCH --mail-user=baohua.zhou@yale.edu

module load miniconda
source activate py37_dev

/usr/bin/time python3 samples_generation_grid.py