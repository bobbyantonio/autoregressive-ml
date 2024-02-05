# Note that this requires setup of an API key and installing cdsapi:
# Go to: https://cds.climate.copernicus.eu/api-how-to
import os, sys
import subprocess
import cdsapi
import xarray as xr
from pathlib import Path
import numpy as np
import tempfile
from calendar import monthrange
from typing import Iterable
from argparse import ArgumentParser

HOME = Path(__file__).parents[1]

sys.path.append( str(HOME / 'automl'))

from automl import data

PRESSURE_LEVELS_ERA5_37 = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900,
    925, 950, 975, 1000)


SURFACE_VARS =  (
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                'geopotential', 'land_sea_mask', 'mean_sea_level_pressure',
                'toa_incident_solar_radiation', 'total_precipitation', 'sea_surface_temperature'
)

PRESSURE_LEVEL_VARS = (
                'potential_vorticity',
                'specific_rain_water_content',
                'specific_snow_water_content',
                'geopotential',
                'temperature',
                'u_component_of_wind',
                'v_component_of_wind',
                'specific_humidity',
                'relative_humidity',
                'vertical_velocity',
                'vorticity',
                'divergence',
                'ozone_mass_mixing_ratio',
                'specific_cloud_liquid_water_content',
                'specific_cloud_ice_water_content',
                'fraction_of_cloud_cover')

cds_api_client = cdsapi.Client()

def format_days(year: str, month:str, days:str):
    final_month_day = monthrange(int(year), int(month))[1]
    
    output_days = []
    for day in days:
        if int(day) < 0:
            final_month_day = monthrange(int(year), int(month))[1]
            if int(day) < -1*final_month_day:
                raise ValueError(f'Invalid value day={day}')
            day= final_month_day + 1 + int(day)

        if not int(day) > final_month_day and not int(day) < 1:
            output_days.append(f'{int(day):02d}')
        
    return output_days



def retrieve_data(year:int, 
                    output_prefix:str,
                    var:str,
                    months:Iterable=range(1,13),
                    days:Iterable=range(1,32),
                    pressure_level=None,
                    output_resolution: float=None
                    ):
    if var=='total_precipitation':
        # Collect full history for precip since it needs to be aggregated (the others are subsamples)
        time = [f'{n:02d}:00' for n in range(24)]
    else:
        time =  [f'{n:02d}:00' for n in (0,6,12,18)]
    

    
    request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': var,
            'year': str(year),
            'month': months,
            'day': [f'{int(day):02d}' for day in days],
            'time': time,
        }

    
    if pressure_level is not None:
        if not isinstance(pressure_level, tuple) and not isinstance(pressure_level, list):
            pressure_level = [pressure_level]
        request['pressure_level'] = [str(lvl) for lvl in pressure_level]
    with tempfile.NamedTemporaryFile() as fp:
        
        cds_api_client.retrieve(
            'reanalysis-era5-single-levels' if pressure_level is None else 'reanalysis-era5-pressure-levels', 
            request, fp.name)
        
        if output_resolution is not None and output_resolution != 0.25:
            ds = xr.load_dataset(fp.name)
            ds = data.interpolate_dataset_on_lat_lon(ds, 
                                   latitude_vals=np.arange(-90, 90, output_resolution) , 
                                   longitude_vals=np.arange(0,360,output_resolution),
                                   interp_method ='conservative')
            ds.to_netcdf(fp.name + 'regridded')
            output_prefix += f'_{output_resolution}deg'
    
        ### 
        # Split into days to make it easier to look up values at daily level
        obase = output_prefix
        res = subprocess.run(['cdo', 'splitday', fp.name, obase])
        
        if res.returncode != 0:
            raise IOError(f'Error splitting files into days for var = {var}, year= {year}, months={months}, days={days}')
        

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Folder to save data to")
    parser.add_argument('--years', nargs='+', required=True,
                        help='Year(s) to collect data for')
    parser.add_argument('--surface', action='store_true',
                        help='Collect surface variables')
    parser.add_argument('--plevels', action='store_true',
                        help='Collect pressure level data')
    parser.add_argument('--pressure-level',  nargs='+', default=[None],
                    help='Specific pressure levels to collect for')
    parser.add_argument('--vars', nargs='+', default=None,
                    help='Specific variables to collect data for')  
    parser.add_argument('--months', nargs='+', default=range(1,13),
                        help='Months to collect data for')
    parser.add_argument('--days', nargs='+', default=range(1,32),
                    help='Days to collect data for') 
    parser.add_argument('--resolution', type=float, default=None,
                    help='Resolution to save to (will regrid if != 0.25)')  
    args = parser.parse_args()
    
    if args.vars:
        for var in args.vars:
            
            print(f'** Fetching var={var}', flush=True)
            
            for year in args.years:
                
                print(f'** Fetching year={year}', flush=True)
                
                for month in args.months:
                    
                    print(f'** Fetching month={month}', flush=True)
                    
                    padded_month =f'{int(month):02d}'
                    
                    days = format_days(year, month, args.days)
                      
                    if var in SURFACE_VARS:
                        
                        var_dir = os.path.join(args.output_dir, 'surface', var, str(year))
                        pressure_level=None
                        
                    elif var in PRESSURE_LEVEL_VARS:
                        
                        if args.pressure_level is None:
                            pressure_level = PRESSURE_LEVELS_ERA5_37
                        else:
                            pressure_level = args.pressure_level

                        var_dir = os.path.join(args.output_dir, 'plevels', var, str(year))
                    
                    output_prefix = os.path.join(var_dir, f'era5_{var}_{year}{padded_month}')
                        
                    os.makedirs(var_dir, exist_ok=True)
                        
                    # Don't overwrite existing data 
                    days = [d for d in days if not os.path.exists(output_prefix + f'{d}.nc')]
                    
                    if len(days)> 0:
                        retrieve_data(year=year,
                                    months=[padded_month],
                                    days=days,
                                    var=var,
                                    pressure_level=pressure_level,
                                    output_resolution=args.resolution,
                                    output_prefix=os.path.join(var_dir, f'era5_{var}_{year}{padded_month}'))
                        
    else: 
        if args.surface:
            subfolder_name = 'surface'
            vars = SURFACE_VARS
            pressure_level=None

        elif args.plevels: 
            subfolder_name = 'plevels'
            vars = PRESSURE_LEVEL_VARS

            pressure_level=PRESSURE_LEVELS_ERA5_37
        else:
            raise ValueError('Input arguments invalid') 
            
        for var in vars:
            print(f'**Fetching var={var}', flush=True)
            
            
            for year in args.years:
                
                print(f'** Fetching year={year}', flush=True)
                
                var_dir = os.path.join(args.output_dir,  subfolder_name, var, str(year))
                os.makedirs(var_dir, exist_ok=True)
                        
                for month in args.months:
                    
                    print(f'** Fetching month={month}', flush=True)
                    
                    padded_month =f'{int(month):02d}'
                    days = format_days(year, month, args.days)
                    
                    output_prefix=os.path.join(var_dir, f'era5_{var}_{year}{padded_month}')
                    
                    # Don't overwrite existing data 
                    # days = [d for d in days if not os.path.exists(output_prefix + f'{d}.nc')]
                    
                    # if len(days)> 0:
                
                    retrieve_data(year=year,
                                months=[padded_month],
                                days=days,
                                var=var,
                                pressure_level=pressure_level,
                                output_resolution=args.resolution,
                                output_prefix=os.path.join(var_dir, f'era5_{var}_{year}{padded_month}'))
        