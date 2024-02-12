#!/bin/bash 
#SBATCH --job-name=fetch-era5
#SBATCH --output=logs/era5-%A.txt 
#SBATCH --partition=shared 
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00 
#SBATCH --mem-per-cpu=100gb

source ~/.bashrc
conda activate graphcast

# srun python /home/a/antonio/repos/autoregressive-ml/automl/fetch_era5.py --years 2011 --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --surface;
# srun python /home/a/antonio/repos/autoregressive-ml/automl/fetch_era5.py --year 2016 --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --plevels --days 1 15 --months 3 4 5 6 7 8;
# srun python /home/a/antonio/repos/autoregressive-ml/automl/fetch_era5.py --year 2016 --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --vars sea_surface_temperature;
srun python /home/a/antonio/repos/autoregressive-ml/automl/fetch_era5.py --year 2016 --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --vars temperature --pressure-levels 1000 975 950 --force-overwrite;
