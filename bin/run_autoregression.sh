#!/bin/bash 

source ~/.bashrc
conda activate graphcast

# python -m automl.autoregression --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/low_res_plevel_hi_res_surface_conservative--num-steps 320 --year 2016 --month 1 --day 1 --hour-start 18 
# for year in 2016
# do  
#     echo "Evaluating year $year"
#     for month in 1 2 7 8
#     do  
#         echo "Evaluating month $month"
#         for day in 1 15
#         do  
            
#             python -m automl.autoregression --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions --num-steps 320 --year $year --month $month --day $day --hour-start 18
#         done
#     done
# done

# Now same again but with temperature replacement
# for year in 2011 2012 2013 2014 2015 2016
# do  
#     echo "Evaluating year $year"
#     for month in 1 2 7 8
#     do  
#         echo "Evaluating month $month"
#         for day in 1 15
#         do  
            
#             python -m automl.autoregression --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions --num-steps 320 --year $year --month $month --day $day --hour-start 18
#         done
#     done
# done

# python -m automl.autoregression --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions --num-steps 320 --year $year --month 1 --day 1 --hour-start 18
#  mean_sea_level_pressure 10m_v_component_of_wind 10m_u_component_of_wind total_precipitation_6hr


# for var in sea_surface_temperature
# do
#     echo "Evaluating var $var"
#     python -m automl.autoregression --input-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --num-steps 320 --year 2016 --month 1 --day 1 --hour-start 18 --var-to-replace $var --replace-uses-lsm
# done

# Sensitity testing for low res variables

for lowresvar in temperature geopotential u_component_of_wind v_component_of_wind vertical_velocity specific_humidity
do
    echo "Evaluating $lowresvar as low res"
    for month in 1 7
    do

        python -m automl.autoregression --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions/low_res_${lowresvar} --num-steps 40 --year 2016 --month ${month} --day 1 --hour-start 18 
    done
done