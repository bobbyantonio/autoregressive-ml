import os, sys
import datetime
import xarray as xr
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from graphcast import graphcast as gc

HOME = Path(__file__).parents[1]
DATASET_FOLDER = str(HOME / 'dataset')

ERA5_SURFACE_VARS = list(gc.TARGET_SURFACE_VARS) + list(gc.EXTERNAL_FORCING_VARS)
ERA5_PLEVEL_VARS = list(gc.TARGET_ATMOSPHERIC_VARS)


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

def load_era5(var: str,
            year: int,
            month: int,
            day: int,
            hour: int,
            era_data_dir: str,
            pressure_level=None,
            ):
    
    if pressure_level is None:
        if var not in ERA5_SURFACE_VARS:
            raise ValueError(f'Variable {var} not found in possible surface variable names')
        data_type = 'surface'
    else:
        if var not in ERA5_PLEVEL_VARS:
            raise ValueError(f'Variable {var} not found in possible atmospheric variable names')
        data_type = 'plevels'

    if var != 'total_precipitation_6hr':
        time_sel = [datetime.datetime(year,month, day, hour)]
        era5_var_name = var
    else:
        time_sel = pd.date_range(start=datetime.datetime(year,month,day,hour) - datetime.timedelta(hours=5), periods=6, freq='1h')
        era5_var_name = 'total_precipitation'
        
    fps = set([f"era5_{era5_var_name}_{item.strftime('%Y%m')}.nc" for item in time_sel])
    
    das = []
    
    for fp in fps:
        da = xr.load_dataarray(os.path.join(era_data_dir, data_type, fp))
        da = format_dataarray(da)
        das.append(da)
    da = xr.concat(das, dim='time')
    da = da.sel(time=time_sel)
      
    if var == 'total_precipitation_6hr':
                
        # Have to do some funny stuff with offsets to ensure the right hours are being aggregated,
        # and labelled in the right way
        da[var] = da.resample(time='6h', 
                            label='right', 
                            offset=datetime.timedelta(hours=1), # Offset of grouping
                            loffset =datetime.timedelta(hours=-1) # Label offset
                            ).sum()
        
    if pressure_level is not None:
        da = da.sel(level=pressure_level)
    
    return da.to_dataset()

def load_era5_static(year, month):
    
    static_das = {}
    rename_dict = {}

    for var in tqdm(gc.STATIC_VARS):

        if var == 'geopotential_at_surface':
            var = 'geopotential'
            
        folder_name = 'surface'
        tmp_da = load_clean_dataarray(os.path.join(DATASET_FOLDER, folder_name, f'era5_{var}_{year}{month:02d}.nc'), 
                                        add_batch_dim=False)
        tmp_da = tmp_da.isel(time=0)
        static_das[var] = tmp_da
        rename_dict[tmp_da.name] = var
    
    rename_dict['z'] = 'geopotential_at_surface'

    static_ds = xr.merge(static_das.values())
    static_ds = static_ds.rename(rename_dict)
    static_ds = static_ds.drop_vars('time')

    # Check lat values are correctly ordered
    assert static_ds.lat[0] < 0 
    assert static_ds.lat[-1] > 0
    
    return static_ds

def load_era5_surface(year, month):
    
    surf_das = {}
    rename_dict = {}

    for var in tqdm(gc.TARGET_SURFACE_VARS + gc.EXTERNAL_FORCING_VARS):

        if var == 'total_precipitation_6hr':
            var = 'total_precipitation'
            
        folder_name = 'surface'
        tmp_da = load_clean_dataarray(os.path.join(DATASET_FOLDER, folder_name, f'era5_{var}_{year}{1:02d}.nc'),
                                        add_batch_dim=True,
                                        )
        if var != 'total_precipitation':
            time_sel = pd.date_range(start=datetime.datetime(year,month, 1, 6), periods=3, freq='6h')
        else:
            time_sel = pd.date_range(start=datetime.datetime(year,month, 1, 1), periods=18, freq='1h')
        tmp_da = tmp_da.sel(time=time_sel)

        surf_das[var] = tmp_da
        rename_dict[tmp_da.name] = var
        
        if var == 'total_precipitation':
            # Have to do some funny stuff with offsets to ensure the right hours are being aggregated,
            # and labelled in the right way
            surf_das[var] = tmp_da.resample(time='6h', 
                                            label='right', 
                                            offset=datetime.timedelta(hours=1), # Offset of grouping
                                            loffset =datetime.timedelta(hours=-1) # Label offset
                                            ).sum()
            rename_dict[tmp_da.name] = 'total_precipitation_6hr'
            
    surface_ds = xr.merge(surf_das.values())
    surface_ds = surface_ds.rename(rename_dict)

    assert sorted(surface_ds.data_vars) == sorted(gc.EXTERNAL_FORCING_VARS + gc.TARGET_SURFACE_VARS )

    # Add datetime coordinate
    surface_ds = add_datetime(surface_ds,
                              start=datetime.datetime(year,month, 1, 6),
                              periods=3, freq='6h')
    
    return surface_ds

def load_era5_plevel(year, month):
    
    plevel_das = {}
    rename_dict = {}
    
    time_sel = pd.date_range(start=datetime.datetime(year, month, 1, 6), periods=3, freq='6h')

    for var in tqdm(gc.TARGET_ATMOSPHERIC_VARS):

        folder_name = 'plevels'
        tmp_da = load_clean_dataarray(os.path.join(DATASET_FOLDER, folder_name, f'era5_{var}_{year}{month:02d}.nc'),
                                        add_batch_dim=True)

        tmp_da = tmp_da.sel(time=time_sel)
        plevel_das[var] = tmp_da
        rename_dict[tmp_da.name] = var

    plevel_ds = xr.merge(plevel_das.values())
    plevel_ds = plevel_ds.rename(rename_dict)

    assert sorted(plevel_ds.data_vars) == sorted(gc.TARGET_ATMOSPHERIC_VARS)
    
    # Add datetime coordinate
    plevel_ds = add_datetime(plevel_ds,
                            start=datetime.datetime(year,month, 1, 6),
                            periods=3, freq='6h')
    
    return plevel_ds