#!/bin/bash 

cd /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5;

for var in divergence geopotential potential_vorticity specific_cloud_ice_water_content specific_snow_water_content u_component_of_wind vertical_velocity fraction_of_cloud_cover ozone_mass_mixing_ratio relative_humidity specific_cloud_liquid_water_content specific_rain_water_content temperature v_component_of_wind vorticity
do
    mv \[None\]hPa/$var/2011/ plevels/$var/
    mv \[None\]hPa/$var/2012/ plevels/$var/
done