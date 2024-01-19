#!/bin/bash 
#SBATCH --job-name=fetch-era5
#SBATCH --output=logs/era5-%A.txt 
#SBATCH --partition=shared 
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00 
#SBATCH --mem-per-cpu=100gb

source ~/.bashrc
conda activate base

# srun python /home/a/antonio/repos/autoregressive-ml/automl/fetch_era5.py --years 2014 2013 2012 --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --surface;
srun python /home/a/antonio/repos/autoregressive-ml/automl/fetch_era5.py --year 2014 2013 --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --plevels --days 1 15 --months 1 2 7 8;
# srun python /home/a/antonio/repos/autoregressive-ml/automl/fetch_era5.py --year 2016 --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --pressure-level 1000 --plevels;
# srun python /home/a/antonio/repos/autoregressive-ml/automl/fetch_era5.py --year 2016 --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --plevels --days -1;
