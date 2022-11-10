#!/bin/bash
#SBATCH --job-name=get_loss_landscape
#SBATCH --partition=day
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=19G
#SBATCH --time=1-
#SBATCH --mail-type=ALL
#SBATCH --mail-user=baohua.zhou@yale.edu

module load miniconda

conda activate py37_dev

export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
/usr/bin/time python get_loss_landscape_run.py