#!/bin/bash 
#SBATCH --job-name=fetch-era5
#SBATCH --output=logs/era5-%A.txt 
#SBATCH --partition=shared 
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00 
#SBATCH --mem-per-cpu=100gb

source ~/.bashrc
conda activate graphcast

# srun python /home/a/antonio/repos/autoregressive-ml/automl/fetch_era5.py --years 2011 2012 2013 2014 2015 --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --surface;
srun python /home/a/antonio/repos/autoregressive-ml/automl/fetch_era5.py --years 2011 2012 2013 2014 2015 --pressure-levels 1000 900 850 750 500 100 1 --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --plevels --force-overwrite;
# srun python /home/a/antonio/repos/autoregressive-ml/automl/fetch_era5.py --year 2016 --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --vars sea_surface_temperature;
# srun python /home/a/antonio/repos/autoregressive-ml/automl/fetch_era5.py --year 2016 --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --vars geopotential u_component_of_wind v_component_of_wind vertical_velocity specific_humidity --pressure-levels 1000 975 950 500 50 --months 1 7 --days 1 2 3 4 5 6 7 8 9 10;
