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

def retrieve_data(years: list, 
                    output_dir:str,
                    var:str,
                    months:list,
                    days:Iterable=range(1,32),
                    pressure_level=None,
                    output_resolution: float=0.25
                    ):
    
    output_prefix=os.path.join(output_dir, f'era5_{var}_{year}{int(month):02d}')
    months = [f'{int(month):02d}' for month in months]
    days = format_days(year, month, args.days)
    years = [str(year) for year in years]

    if var=='total_precipitation':
        # Collect full history for precip since it needs to be aggregated (the others are subsamples)
        time = [f'{n:02d}:00' for n in range(24)]
    else:
        time =  [f'{n:02d}:00' for n in (0,6,12,18)]
    
    request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': var,
            'year': years,
            'month': months,
            'day': days,
            'time': time,
            'grid': [output_resolution, output_resolution]
        }
    
    if pressure_level is not None:
        if not isinstance(pressure_level, tuple) and not isinstance(pressure_level, list):
            pressure_level = [pressure_level]
        request['pressure_level'] = [str(lvl) for lvl in pressure_level]

    with tempfile.NamedTemporaryFile() as fp:
        
        cds_api_client.retrieve(
            'reanalysis-era5-single-levels' if pressure_level is None else 'reanalysis-era5-pressure-levels', 
            request, fp.name)

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
                pressure_levels = data.PRESSURE_LEVELS_ERA5_37
            else:
                pressure_levels = args.pressure_levels
        
        else: 
            raise ValueError(f'Unrecognised variable {var}')
        
        era5_var_name = data.ERA5_VARNAME_LOOKUP.get(var, var)

        for month in args.months:
            # loop by month first since Copernicus doesn't allow you to submit a request with 31st Feb, and this way
            # means less waiting time.
            
            print(f'** Fetching month={month}', flush=True)

            var_dir = os.path.join(args.output_dir, data_category, var)

            if var=='total_precipitation':
                # Collect full history for precip since it needs to be aggregated (the others are subsamples)
                hours = range(24)
            else:
                hours = [0,6,12,18]

            datetimes_to_save = []
            for year in args.years:
                days = format_days(year=year, month=month, days=args.days)
                
                datetimes_to_save += [datetime.datetime(year=int(year), month=int(month), day=int(day), hour=h) for day in days for h in hours]
            datetimes_to_save = sorted(set(datetimes_to_save))

            if not args.force_overwrite:
                datetimes_to_save = [dt for dt in datetimes_to_save if not os.path.exists(os.path.join(var_dir, f'era5_{var}_{year}{dt.month:02d}{dt.day:02d}.nc'))]

            years_to_save = sorted(set([dt.year for dt in datetimes_to_save]))
            days_to_save = sorted(set(dt.day for dt in datetimes_to_save))

            if len(datetimes_to_save) > 0:
                for year in years_to_save:
                    # First check that this data isn't already in the AOPP data; if so then just copy from there
                    short_era5_name = ERA5_SHORT_NAME_LOOKUP[data_category].get(era5_var_name)

                    existing_fp = os.path.join(AOPP_ERA5_DIR, f'{short_era5_name}/1hr/{short_era5_name}_1hr_ERA5_{args.resolution}x{args.resolution}_{year}01-{year}12.nc')
                    
                    if os.path.exists(existing_fp):

                        ds = xr.open_dataset(existing_fp)
                        ds = ds.sel(time=[dt for dt in datetimes_to_save if dt.year ==year])

                        if pressure_levels is not None:
                            ds = ds.sel(level=pressure_levels)
                        da = ds[list(ds.data_vars)[0]]
                        save_array_to_separate_days(output_dir=var_dir, da=da, var_name=var)

                else:

                    # if var in SURFACE_VARS:
                    # Quicker to download it all at once
                    ds = retrieve_data(years=years_to_save,
                                    months=[month],
                                    days=days_to_save,
                                    var=var,
                                    pressure_level=pressure_levels,
                                    output_resolution=args.resolution,
                                    output_dir=var_dir)

                    da = ds[list(ds.data_vars)[0]]
                    save_array_to_separate_days(output_dir=var_dir, da=da, var_name=var)
