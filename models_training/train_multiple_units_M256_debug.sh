#!/bin/bash
#SBATCH --job-name=debug_M256_training
#SBATCH --partition=day
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --time=1-
#SBATCH --mail-type=ALL
#SBATCH --mail-user=baohua.zhou@yale.edu

module load miniconda

conda activate py37_dev

export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
/usr/bin/time python train_multiple_units_M256_debug.py