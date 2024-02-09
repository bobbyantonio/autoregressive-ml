#!/bin/bash 
#SBATCH --job-name=regrid-era5
#SBATCH --output=logs/regrid-%A-%a.txt 
#SBATCH --partition=shared 
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00 
#SBATCH --mem-per-cpu=100gb
#SBATCH --array=5

source ~/.bashrc
conda activate graphcast

srun python /home/a/antonio/repos/autoregressive-ml/scripts/regrid_data.py --years 2016 --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5_1deg --days 1 --months ${SLURM_ARRAY_TASK_ID} --resolution 1;
