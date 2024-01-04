# Note that this requires setup of an API key and installing cdsapi:
# Go to: https://cds.climate.copernicus.eu/api-how-to
import os, sys
import subprocess
import cdsapi
from tqdm import tqdm
import tempfile
from typing import Iterable
from argparse import ArgumentParser


PRESSURE_LEVELS_ERA5_37 = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900,
    925, 950, 975, 1000)


SURFACE_VARS =  (
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                'geopotential', 'land_sea_mask', 'mean_sea_level_pressure',
                'toa_incident_solar_radiation', 'total_precipitation',
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

c = cdsapi.Client()


def retrieve_data(year:int, 
                    output_prefix:str,
                    var:str,
                    months:Iterable=range(1,13),
                    days:Iterable=range(1,32),
                    pressure_level=None
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
        
        c.retrieve(
            'reanalysis-era5-single-levels' if pressure_level is None else 'reanalysis-era5-pressure-levels', 
            request, fp.name)
    
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
    parser.add_argument('--year', type=int, required=True,
                        help='Year to collect data for')
    parser.add_argument('--surface', action='store_true',
                        help='Collect surface variables')
    parser.add_argument('--plevels', action='store_true',
                        help='Collect pressure level data')
    parser.add_argument('--pressure-level', default=None,
                    help='Specific pressure level to collect for')
    parser.add_argument('--vars', nargs='+', default=None,
                    help='Specific variable to collect data for')  
    parser.add_argument('--months', nargs='+', default=range(1,13),
                        help='Months to collect data for')
    parser.add_argument('--days', nargs='+', default=range(1,32),
                    help='Days to collect data for')  
    args = parser.parse_args()
    
    if args.vars:
        for var in args.vars:
            for month in args.months:
                padded_month =f'{int(month):02d}'
                
            
                if var in SURFACE_VARS:
                    
                    var_dir = os.path.join(args.output_dir, 'surface', var, str(args.year))
                    os.makedirs(var_dir, exist_ok=True)
                    
                    retrieve_data(year=args.year,
                                months=[padded_month],
                                days=args.days,
                                var=var,
                                output_prefix=os.path.join(var_dir, f'era5_{var}_{args.year}{padded_month}'))
                elif var in PRESSURE_LEVEL_VARS and var != 'geopotential':
                    var_dir = os.path.join(args.output_dir, 'plevels', var, str(args.year))
                    os.makedirs(var_dir, exist_ok=True)
                        
                    if args.pressure_level is not None:
                        
                        output_prefix = os.path.join(var_dir, f'era5_{var}_{args.pressure_level}hPa_{args.year}{padded_month}')
                        pressure_level=args.pressure_level
                    else:
                        
                        output_prefix = os.path.join(var_dir, f'era5_{var}_{args.year}{padded_month}')
                        pressure_level = PRESSURE_LEVELS_ERA5_37
                    retrieve_data(year=args.year,
                                months=[padded_month],
                                days=args.days,
                                var=var,
                                pressure_level=pressure_level,
                                output_prefix=output_prefix)
    
    if args.surface:
        for var in SURFACE_VARS:
            
            var_dir = os.path.join(args.output_dir, 'surface', var, str(args.year))
            os.makedirs(var_dir, exist_ok=True)
                    
            for month in args.months:
                padded_month =f'{int(month):02d}'
                retrieve_data(year=args.year,
                            months=[padded_month],
                            days=args.days,
                            var=var,
                            output_prefix=os.path.join(var_dir, f'era5_{var}_{args.year}{padded_month}'))
            
    if args.plevels and args.vars is None: 
        for var in PRESSURE_LEVEL_VARS:
            
            var_dir = os.path.join(args.output_dir, 'plevels', var, str(args.year))
            os.makedirs(var_dir, exist_ok=True)
                    
            for month in args.months:
                padded_month =f'{int(month):02d}'
                if args.pressure_level is not None:
                    output_prefix = os.path.join(var_dir, f'era5_{var}_{args.pressure_level}hPa_{args.year}{padded_month}')
                else:
                    output_prefix = os.path.join(var_dir, f'era5_{var}_{args.year}{padded_month}')
                retrieve_data(year=args.year,
                                    months=[padded_month],
                                    var=var,
                                    days=args.days,
                                    pressure_level=PRESSURE_LEVELS_ERA5_37,
                                    output_prefix=output_prefix)
    
    