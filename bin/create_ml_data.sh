#!/bin/bash 
#SBATCH --job-name=ml-data
#SBATCH --output=logs/create-ml-data-%A.txt 
#SBATCH --partition=shared 
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00 
#SBATCH --mem-per-cpu=100gb


source ~/.bashrc
conda activate graphcast

srun python -m scripts.create_ml_data --data-config-path /home/a/antonio/repos/autoregressive-ml/config/test_data_config.yaml