import os, sys
import datetime
import dask
import calendar

import xarray as xr
os.environ['ESMFMKFILE'] = '/home/a/antonio/nobackups/miniforge3/envs/graphcast/lib/esmf.mk'

import xesmf as xe
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from functools import partial
from graphcast import graphcast as gc

HOME = Path(__file__).parents[1]
HI_RES_ERA5_DIR = '/network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5'
LOW_RES_ERA5_DIR = '/network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5_1deg/bilinear'

SECONDS_IN_DAY = 24*60*60

ERA5_SURFACE_VARS = TARGET_SURFACE_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "total_precipitation_6hr",
    "toa_incident_solar_radiation")

ERA5_PLEVEL_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)

GENERATED_FORCING_VARS = (
    "year_progress_sin",
    "year_progress_cos",
    "day_progress_sin",
    "day_progress_cos",
    "latitude_sin",
    "longitude_cos",
    "longitude_sin"
)

ERA5_STATIC_VARS = (
    "geopotential_at_surface",
    "land_sea_mask",
)
ERA5_SEA_VARS = ('sea_surface_temperature',)

ERA5_VARNAME_LOOKUP = {'total_precipitation_6hr': 'total_precipitation',
                       'geopotential_at_surface': 'geopotential'}




PRESSURE_LEVELS_ERA5_37 = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900,
    925, 950, 975, 1000)

REGRIDDING_STRATEGY = {'total_precipitation': 'conservative',
                       'total_precipitation_6hr': 'bilinear'}

def format_dataarray(da):

    lat_var_name, _ = get_lat_lon_names(da)
    
    if lat_var_name == 'latitude':
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

def get_lat_lon_names(ds: xr.Dataset):
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

def interpolate_dataset_on_lat_lon(ds: xr.Dataset, 
                                   latitude_vals: list, 
                                   longitude_vals: list,
                                   interp_method:str ='bilinear'):
    """
    Interpolate dataset to new lat/lon values

    Args:
        ds (xr.Dataset): Datast to interpolate
        latitude_vals (list): list of latitude values to interpolate to
        longitude_vals (list): list of longitude values to interpolate to
        interp_method (str, optional): name of interpolation method. Defaults to 'bilinear'._

    Returns:
        xr,Dataset: interpolated dataset
    """
    lat_var_name, lon_var_name = get_lat_lon_names(ds)
    
    ds_out = xr.Dataset(
        {
            lat_var_name: ([lat_var_name], latitude_vals),
            lon_var_name: ([lon_var_name], longitude_vals),
        }
    )

    regridder = xe.Regridder(ds, ds_out, interp_method)
    regridded_ds = ds.copy()
    
    # Make float vars C-contiguous (to avoid warning message and potentially improve performance)
    if isinstance(regridded_ds, xr.Dataset):
        for var in list(regridded_ds.data_vars):
            if regridded_ds[var].values.dtype.kind == 'f':
                regridded_ds[var].values = np.ascontiguousarray(regridded_ds[var].values)
    else:
        if regridded_ds.values.dtype.kind == 'f':
            regridded_ds.values = np.ascontiguousarray(regridded_ds.values)
            
    regridded_ds = regridder(regridded_ds)

    return regridded_ds


def convert_to_relative_time(ds, zero_time: np.datetime64):
    
    ds['time'] = ds['time'] - zero_time
    return ds

def ns_to_hr(ns_val: float):
    
    return ns_val* 1e-9 / (60*60)


def _preprocess_sel_levels(x: xr.Dataset, levels: list):
    return x.sel(level=levels)

def open_large_dataset(fps: list[str],
                       datetimes: list,
                       pressure_levels: list[int]=None
                       ):
    if pressure_levels is not None:
        if isinstance(pressure_levels , tuple):
                pressure_levels = list(pressure_levels)
        preprocess_func = partial(_preprocess_sel_levels, levels=pressure_levels)
    else:
        preprocess_func = None

    # Need to preprocess to select levels, otherwise this function has trouble combining them
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        ds = xr.open_mfdataset(fps, preprocess=preprocess_func)
        ds = ds.sel(time=datetimes)
    
    return ds
    
def get_year_progress(dt: datetime.datetime):

    dt_0 = datetime.datetime(dt.year,1,1,0)
    t_delta = (dt - dt_0)

    total_year_seconds = (365 + calendar.isleap(dt.year))*SECONDS_IN_DAY
    year_progress = t_delta.total_seconds() / total_year_seconds

    return year_progress

def get_day_progress(dt: datetime.datetime):

    dt_0 = datetime.datetime(dt.year,1,1,0)
    t_delta = (dt - dt_0)

    day_progress = t_delta.seconds / SECONDS_IN_DAY

    return day_progress
    
    
def load_era5(var: str,
            datetimes: list,
            era_data_dir: str,
            pressure_levels: list=None,
            output_resolution: float=0.25,
            interpolation_method: str='bilinear'
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
    if var in ERA5_SURFACE_VARS + ERA5_STATIC_VARS + ERA5_SEA_VARS:
        data_category = 'surface'
    elif var in ERA5_PLEVEL_VARS:
        data_category = 'plevels'
    else:
        raise ValueError(f'Variable {var} not found in possible variable names')


    if pressure_levels is not None and data_category == 'surface':
        pressure_levels = None        
        
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
    
    da = open_large_dataset(fps=fps, pressure_levels=pressure_levels, datetimes=datetimes)
    da = da[list(da.data_vars)[0]]

    # If not in correct output resolution, then regrid
    # Here we assume that resolution is uniform, and that resolution doesn't go below the 4th decimal place
    lat_var_name, _ = get_lat_lon_names(da)
    current_resolution = np.round(np.abs(da[lat_var_name][1] - da[lat_var_name][0]), 4)
    if current_resolution != output_resolution:
        
        # Currently assumes global data
        new_lat_vals = np.arange(-90, 90 + output_resolution, output_resolution)
        new_lon_vals = np.arange(0, 360, output_resolution)

        da = interpolate_dataset_on_lat_lon(da, 
                                   latitude_vals=new_lat_vals, 
                                   longitude_vals=new_lon_vals,
                                   interp_method = interpolation_method)

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

def load_era5_static(year: int, month: int, day: int, hour: int=1, era5_data_dir: str=HI_RES_ERA5_DIR):
    
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
                      era5_data_dir: str=HI_RES_ERA5_DIR, 
                      vars: list=ERA5_SURFACE_VARS,
                      low_res_vars: list=None):
    
    if not isinstance(vars, list) and not isinstance(vars, tuple):
        vars = [vars]

    if not isinstance(vars, list) and not isinstance(vars, tuple):
        low_res_vars = [low_res_vars]
        
    surf_das = {}
    time_sel = pd.date_range(start=datetime.datetime(year,month,day,hour) - datetime.timedelta(hours=12), 
                                    periods=3, 
                                    freq='6h')
    if not gather_input_datetimes:
        time_sel = time_sel[-1:]
    
    for var in tqdm(vars):
        tmp_data_dir = HI_RES_ERA5_DIR
        if low_res_vars is not None:
            if var in low_res_vars:
                tmp_data_dir = LOW_RES_ERA5_DIR               

        tmp_da = load_era5(var, time_sel, era_data_dir=tmp_data_dir).load()

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
                     era5_data_dir: str=HI_RES_ERA5_DIR, 
                     vars: list=ERA5_PLEVEL_VARS,
                     low_res_vars: list=None):
    
    if not isinstance(vars, list) and not isinstance(vars, tuple):
        vars = [vars]
    
    plevel_das = {}
    
    time_sel = pd.date_range(start=datetime.datetime(year,month,day,hour) - datetime.timedelta(hours=12), 
                                 periods=3, 
                                 freq='6h')
    if not gather_input_datetimes:
        time_sel = time_sel[-1:]

    for var in tqdm(vars):
        tmp_data_dir = HI_RES_ERA5_DIR
        if low_res_vars is not None:
            if var in low_res_vars:
                tmp_data_dir = LOW_RES_ERA5_DIR            
        
        tmp_da = load_era5(var, time_sel, era_data_dir=tmp_data_dir,
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