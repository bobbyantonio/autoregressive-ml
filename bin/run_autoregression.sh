#!/bin/bash 

source ~/.bashrc
conda activate graphcast

year=2016

# for month in 1 2 3 4 5 6 7 8 9 10 11 12
# do  
#     echo "Evaluating month $month"
#     python -m scripts.autoregression --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions --num-steps 100 --year $year --month $month --day 1 --hour-start 18
# done

# python -m scripts.autoregression --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions --num-steps 320 --year $year --month 1 --day 1 --hour-start 18

for var in 2m_temperature mean_sea_level_pressure 10m_v_component_of_wind 10m_u_component_of_wind total_precipitation_6hr
do
    echo "Evaluating var $var"
    python -m scripts.autoregression --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions --num-steps 100 --year $year --month 1 --day 1 --hour-start 18 --var-to-replace $var
done