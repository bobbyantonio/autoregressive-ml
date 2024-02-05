import os, sys
import datetime
import dask
import xarray as xr
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from functools import partial
from graphcast import graphcast as gc

HOME = Path(__file__).parents[1]
DATASET_FOLDER = '/network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5'

ERA5_SURFACE_VARS = list(gc.TARGET_SURFACE_VARS) + list(gc.EXTERNAL_FORCING_VARS) 
ERA5_PLEVEL_VARS = list(gc.TARGET_ATMOSPHERIC_VARS)
ERA5_STATIC_VARS = list(gc.STATIC_VARS)
ERA5_SEA_VARS = ['sea_surface_temperature']

ERA5_VARNAME_LOOKUP = {'total_precipitation_6hr': 'total_precipitation',
                       'geopotential_at_surface': 'geopotential'}

def format_dataarray(da):
    
    rename_dict = {'latitude': 'lat', 'longitude': 'lon' }

    da = da.rename(rename_dict)
    
    da = da.sortby('lat', ascending=True)
    da = da.sortby('lon', ascending=True)
    
    return da

def load_clean_dataarray(fp, add_batch_dim=False):
    
    da = xr.load_dataarray(fp)
    da = format_dataarray(da)
    
    if add_batch_dim:
        raise NotImplementedError()
        da = da.expand_dims({'batch': 1})

    return da

def add_datetime(ds, start: str,
                 periods: int,
                 freq: str='6h',
                 ):
    
    dt_arr = np.expand_dims(pd.date_range(start=start, periods=periods, freq=freq),0)
    ds = ds.assign_coords(datetime=(('batch', 'time'),  dt_arr))
    return ds

def infer_lat_lon_names(ds: xr.Dataset):
    """
    Infer names of latitude / longitude coordinates from the dataset

    Args:
        ds (xr.Dataset): dataset (containing one latitude coordinate 
        and one longitude coordinate)

    Returns:
        tuple: (lat name, lon name)
    """
    
    coord_names = list(ds.coords)
    lat_var_name = [item for item in coord_names if item.startswith('lat')]
    lon_var_name = [item for item in coord_names if item.startswith('lon')]
    
    assert (len(lat_var_name) == 1) and (len(lon_var_name) == 1), IndexError('Cannot infer latitude and longitude names from this dataset')

    return lat_var_name[0], lon_var_name[0]

def convert_to_relative_time(ds, zero_time: np.datetime64):
    
    ds['time'] = ds['time'] - zero_time
    return ds

def ns_to_hr(ns_val: float):
    
    return ns_val* 1e-9 / (60*60)


def _preprocess_sel_levels(x: xr.Dataset, levels: list):
    return x.sel(level=levels)


def load_era5(var: str,
            datetimes: list,
            era_data_dir: str,
            pressure_levels: list=None
            ):
    """Load ERA5 data, focused towards data that the Graphcast model expects (6 hourly)

    Args:
        var (str): variable name
        year (int): year
        month (int): month
        day (int): day
        hour (int): hour
        era_data_dir (str): folder containing era5 data
        pressure_levels (list, optional): List of pressure levels to fetch, if relevant. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    

    if pressure_levels is None:
        preprocess_func = None
        if var not in ERA5_SURFACE_VARS + ERA5_STATIC_VARS + ERA5_SEA_VARS:
            raise ValueError(f'Variable {var} not found in possible surface variable names')
        data_category = 'surface'
    else:
        preprocess_func = None
        if isinstance(pressure_levels , tuple):
            pressure_levels = list(pressure_levels)
        preprocess_func = partial(_preprocess_sel_levels, levels=pressure_levels)
        
        if var not in ERA5_PLEVEL_VARS:
            raise ValueError(f'Variable {var} not found in possible atmospheric variable names')
        data_category = 'plevels'
        
    if isinstance(pressure_levels , tuple):
        pressure_levels = list(pressure_levels)
 
    if var == 'total_precipitation_6hr':
        extra_datetimes = []
        for dt in datetimes:
            extra_datetimes += [dt - datetime.timedelta(hours=n) for n in range(1,6)]
            
        if len(set(datetimes).intersection(set(extra_datetimes))) > 0:
            raise ValueError('Datetimes must be at least 6hr apart, as the current code cannot currently resample precipitation otherwise.')
        
        datetimes = sorted(set(list(datetimes) + extra_datetimes))
    
    era5_var_name = ERA5_VARNAME_LOOKUP.get(var, var)
        
    fps = sorted(set([os.path.join(era_data_dir, data_category, era5_var_name, f"{item.strftime('%Y')}/era5_{era5_var_name}_{item.strftime('%Y%m%d')}.nc") for item in datetimes]))
    
    # Need to preprocess to select levels, otherwise this function has trouble combining them
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        da = xr.open_mfdataset(fps, preprocess=preprocess_func)
        da = da.sel(time=datetimes)[list(da.data_vars)[0]]
        
    if pressure_levels is not None:
        da = da.sel(level=pressure_levels)
        
    da = format_dataarray(da)
    da.name = var
      
    if var == 'total_precipitation_6hr':
                
        # Have to do some funny stuff with offsets to ensure the right hours are being aggregated,
        # and labelled in the right way
        da = da.resample(time='6h', 
                            label='right', 
                            offset=datetime.timedelta(hours=1), # Offset of grouping
                            ).sum()
        offset = pd.tseries.frequencies.to_offset("1h")
        da['time'] = da.get_index("time") - offset
    
    return da

def load_era5_static(year: int, month: int, day: int, hour: int=1, era5_data_dir: str=DATASET_FOLDER):
    
    static_das = []

    for var in tqdm(gc.STATIC_VARS):

        da = load_era5(var=var,
                       datetimes=[datetime.datetime(year=year, month=month, day=day, hour=hour)],
                       era_data_dir=era5_data_dir).load()
        static_das.append(da)

    static_ds = xr.merge(static_das)
    static_ds = static_ds.isel(time=0)
    static_ds = static_ds.drop_vars('time')
    
    return static_ds

def load_era5_surface(year: int, 
                      month: int, 
                      day: int, 
                      hour: int, 
                      gather_input_datetimes: bool=True,
                      era5_data_dir: str=DATASET_FOLDER, 
                      vars: list=ERA5_SURFACE_VARS):
    
    if not isinstance(vars, list) and not isinstance(vars, tuple):
        vars = [vars]
        
    surf_das = {}
    time_sel = pd.date_range(start=datetime.datetime(year,month,day,hour) - datetime.timedelta(hours=12), 
                                    periods=3, 
                                    freq='6h')
    if not gather_input_datetimes:
        time_sel = time_sel[-1:]
    
    for var in tqdm(vars):

        tmp_da = load_era5(var, time_sel, era_data_dir=era5_data_dir).load()

        tmp_da = tmp_da.expand_dims({'batch': 1})

        surf_das[var] = tmp_da
            
    surface_ds = xr.merge(surf_das.values())
    
    # Add datetime coordinate
    if gather_input_datetimes:
        surface_ds = add_datetime(surface_ds,
                                start=time_sel[0],
                                periods=3, freq='6h')
    else:
        surface_ds = add_datetime(surface_ds,
                                start=time_sel[0],
                                periods=1, freq='6h')
    
    return surface_ds

def load_era5_plevel(year: int,
                     month: int, 
                     day: int,
                     hour: int,
                     gather_input_datetimes: bool=True,
                     pressure_levels: list=gc.PRESSURE_LEVELS_ERA5_37,
                     era5_data_dir: str=DATASET_FOLDER, 
                     vars: list=ERA5_PLEVEL_VARS):
    
    if not isinstance(vars, list) and not isinstance(vars, tuple):
        vars = [vars]
    
    plevel_das = {}
    
    time_sel = pd.date_range(start=datetime.datetime(year,month,day,hour) - datetime.timedelta(hours=12), 
                                 periods=3, 
                                 freq='6h')
    if not gather_input_datetimes:
        time_sel = time_sel[-1:]

    for var in tqdm(vars):
        
        tmp_da = load_era5(var, time_sel, era_data_dir=era5_data_dir,
                        pressure_levels=pressure_levels
                        ).load()
                    
        tmp_da = tmp_da.expand_dims({'batch': 1})

        plevel_das[var] = tmp_da

    plevel_ds = xr.merge(plevel_das.values())
    
    # Add datetime coordinate
    if gather_input_datetimes:
        plevel_ds = add_datetime(plevel_ds,
                                start=time_sel[0],
                                periods=3, freq='6h')
    else:
        plevel_ds = add_datetime(plevel_ds,
                                start=time_sel[0],
                                periods=1, freq='6h')
    
    return plevel_ds