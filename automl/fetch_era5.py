# Note that this requires setup of an API key and installing cdsapi:
# Go to: https://cds.climate.copernicus.eu/api-how-to
import os, sys
import subprocess
import cdsapi
import datetime
import pandas as pd
import xarray as xr
from pathlib import Path
import numpy as np
import tempfile
from calendar import monthrange
from typing import Iterable
from argparse import ArgumentParser

HOME = Path(__file__).parents[1]

sys.path.append( str(HOME))

from automl import data
from automl.utils import utils

PRESSURE_LEVELS_ERA5_37 = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900,
    925, 950, 975, 1000)


SURFACE_VARS =  (
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                'geopotential', 'land_sea_mask', 'mean_sea_level_pressure',
                'toa_incident_solar_radiation', 'total_precipitation', 'sea_surface_temperature'
)

AOPP_ERA5_DIR = '/network/group/aopp/met_data/MET001_ERA5/data'


ERA5_SHORT_NAME_LOOKUP = {'surface': {'2m_temperature': 'tas',
                                      '10m_u_component_of_wind': 'uas', 
                                      '10m_v_component_of_wind': 'vas',
                                      'land_sea_mask': 'lsm',
                                      'mean_sea_level_pressure': 'psl',
                                      'toa_incident_solar_radiation': 'rsdt', 
                                      'total_precipitation': 'pr'},
                         'plevels': {'geopotential': 'zg',
                                     'specific_humidity': 'hus',
                                     'temperature': 'ta',
                                     'u_component_of_wind': 'ua',
                                     'v_component_of_wind': 'va'}}

PRESSURE_LEVEL_VARS = data.ERA5_PLEVEL_VARS

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

def save_array_to_separate_hours(output_dir, da, var_name):

    for tmp_da in da.transpose('time', ...):
        tmp_time = pd.Timestamp(tmp_da['time'].item()).to_pydatetime()
        tmp_da = data.format_dataarray(tmp_da).drop('time')

        output_fp = os.path.join(output_dir, f"era5_{var_name}_{tmp_time.strftime('%Y%m%d_%H')}.nc")
        tmp_da.to_netcdf(output_fp)

        # lat_vals = tmp_da['lat'].values
        # lon_vals = tmp_da['lon'].values
        # # Save metadata alongside
        # metadata_dict = {'min_lat': str(min(lat_vals)),
        #                 'max_lat': str(max(lat_vals)),
        #                 'min_lon': str(min(lon_vals)),
        #                 'max_lon': str(max(lon_vals)),
        #                 'lat_step_size': str(lat_vals[1] - lat_vals[0]),
        #                 'lon_step_size': str(lon_vals[1] - lon_vals[0]),
        #                 'pressure_levels': [] if 'level' not in tmp_da.coords else [str(int(l)) for l in tmp_da.level.values]}

        # utils.write_to_yaml(fpath=output_fp.replace('.nc', '.yml'), data=metadata_dict)

def save_array_to_separate_days(output_dir, da, var_name):

    all_datetimes = [pd.Timestamp(item).to_pydatetime() for item in da['time'].values]
    all_yrmonthdays = sorted(set([(dt.year, dt.month, dt.day) for dt in all_datetimes]))

    for ymd in all_yrmonthdays:
        relevant_dts = [dt for dt in all_datetimes if dt.year==ymd[0] and dt.month==ymd[1] and dt.day==ymd[2]]
        tmp_da = da.sel(time=relevant_dts)
        tmp_da = data.format_dataarray(tmp_da)
        
        output_fp = os.path.join(output_dir, f"era5_{var_name}_{ymd[0]}{ymd[1]:02d}{ymd[2]:02d}.nc")
        tmp_da.to_netcdf(output_fp)

        # lat_vals = tmp_da['lat'].values
        # lon_vals = tmp_da['lon'].values
        # # Save metadata alongside
        # metadata_dict = {'min_lat': str(min(lat_vals)),
        #                 'max_lat': str(max(lat_vals)),
        #                 'min_lon': str(min(lon_vals)),
        #                 'max_lon': str(max(lon_vals)),
        #                 'lat_step_size': str(lat_vals[1] - lat_vals[0]),
        #                 'lon_step_size': str(lon_vals[1] - lon_vals[0]),
        #                 'pressure_levels': [] if 'level' not in tmp_da.coords else [str(int(l)) for l in tmp_da.level.values]}

        # utils.write_to_yaml(fpath=output_fp.replace('.nc', '.yml'), data=metadata_dict)

def retrieve_data(year:int, 
                    output_dir:str,
                    var:str,
                    month:int,
                    days:Iterable=range(1,32),
                    pressure_level=None,
                    output_resolution: float=None
                    ):
    
    output_prefix=os.path.join(output_dir, f'era5_{var}_{year}{int(month):02d}')

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
            'month': str(month),
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

        ds = xr.open_dataset(fp.name)
    return ds

        

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
    parser.add_argument('--pressure-levels',  nargs='+', default=[None],
                    help='Specific pressure levels to collect for')
    parser.add_argument('--vars', nargs='+', default=None,
                    help='Specific variables to collect data for')  
    parser.add_argument('--months', nargs='+', default=range(1,13),
                        help='Months to collect data for')
    parser.add_argument('--days', nargs='+', default=range(1,32),
                    help='Days to collect data for') 
    parser.add_argument('--resolution', type=float, default=0.25,
                    help='Resolution to save to (will regrid if != 0.25)') 
    parser.add_argument('--force-overwrite', action='store_true',
                        help='Force overwrite of existing data.')
    args = parser.parse_args()

    if args.surface:
        vars = SURFACE_VARS
    elif args.plevels: 
        vars = PRESSURE_LEVEL_VARS
    elif args.vars:
        vars = args.vars
    else:
        raise ValueError('Input arguments invalid') 


    for var in vars:
        
        print(f'** Fetching var={var}', flush=True)

        if var in SURFACE_VARS:
                    
            data_category = 'surface'
            pressure_levels=None

        elif var in PRESSURE_LEVEL_VARS:

            data_category = 'plevels'
            if args.pressure_levels is None:
                pressure_levels = PRESSURE_LEVELS_ERA5_37
            else:
                pressure_levels = args.pressure_levels
        
        else: 
            raise ValueError(f'Unrecognised variable {var}')
        
        era5_var_name = data.ERA5_VARNAME_LOOKUP.get(var, var)
        
        for year in args.years:
            print(f'** Fetching year={year}', flush=True)

            var_dir = os.path.join(args.output_dir, data_category, var, str(year))

            # First check that this data isn't already in the AOPP data; if so then just copy from there
            short_era5_name = ERA5_SHORT_NAME_LOOKUP[data_category].get(era5_var_name)

            if short_era5_name is not None: # Short names only provided where there is some data for that variable

                existing_fp = os.path.join(AOPP_ERA5_DIR, f'{short_era5_name}/1hr/{short_era5_name}_1hr_ERA5_{args.resolution}x{args.resolution}_{year}01-{year}12.nc')
                if os.path.exists(existing_fp):

                    if var=='total_precipitation':
                        # Collect full history for precip since it needs to be aggregated (the others are subsamples)
                        hours = range(24)
                    else:
                        hours = [0,6,12,18]

                    datetimes_to_save = []
                    for month in args.months:
                        days = format_days(year, month, args.days)
                        datetimes_to_save += [datetime.datetime(year=int(year), month=int(month), day=int(day), hour=h) for day in days for h in hours]
                    datetimes_to_save = sorted(set(datetimes_to_save))

                    if not args.force_overwrite:
                        datetimes_to_save = [dt for dt in datetimes_to_save if not os.path.exists(os.path.join(var_dir, f'era5_{var}_{year}{dt.month:02d}{dt.day:02d}.nc'))]

                    ds = xr.open_dataset(existing_fp)
                    ds = ds.sel(time=datetimes_to_save)

                    if pressure_levels is not None:
                        ds = ds.sel(level=pressure_levels)

                    da = ds[list(ds.data_vars)[0]]
                    save_array_to_separate_days(output_dir=var_dir, da=da, var_name=var)

            else:

                for month in args.months:
                    
                    print(f'** Fetching month={month}', flush=True)
                    
                    padded_month =f'{int(month):02d}'
                    
                    days = format_days(year, month, args.days)
                                        
                    os.makedirs(var_dir, exist_ok=True)
                    
                    if not args.force_overwrite:    
                        # Don't overwrite existing data 
                        
                        days = [d for d in days if not os.path.exists(os.path.join(var_dir, f'era5_{var}_{year}{padded_month}{d}.nc'))]
                    
                    if len(days)> 0:
                        ds = retrieve_data(year=year,
                                    month=padded_month,
                                    days=days,
                                    var=var,
                                    pressure_level=pressure_levels,
                                    output_resolution=args.resolution,
                                    output_dir=var_dir)

                        da = ds[list(ds.data_vars)[0]]
                        save_array_to_separate_days(output_dir=var_dir, da=da, var_name=var)

        