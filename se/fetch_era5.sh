#!/bin/bash 
#SBATCH --job-name=test
#SBATCH --output=logs/era5-%A.txt 
#SBATCH --partition=shared 
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00 
#SBATCH --mem-per-cpu=100gb

source ~/.bashrc
conda activate base

# srun python /home/a/antonio/repos/autoregressive-ml/automl/fetch_era5.py --year 2016 --output-dir /home/a/antonio/nobackups/era5 --surface --months 1 2 3;
srun python /home/a/antonio/repos/autoregressive-ml/automl/fetch_era5.py --year 2016 --output-dir /home/a/antonio/nobackups/era5 --plevels --months 1 --days 1;
# srun python /home/a/antonio/repos/autoregressive-ml/automl/fetch_era5.py --year 2016 --output-dir /home/a/antonio/nobackups/era5 --plevels --pressure-level 1000 --months 1 2 3 --vars specific_humidity;