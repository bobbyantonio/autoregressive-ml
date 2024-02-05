import sys, os
import unittest
from tqdm import tqdm
import datetime
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

from unittest.mock import patch
from unittest import mock

HOME = Path(__file__).parents[1]
DATA_FOLDER = '/network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5'

sys.path.append(str(HOME))

def mock_cds_api_client_retrieve(type, request, fp):
    
    ds = xr.load_dataset(os.path.join(DATA_FOLDER, 'surface/total_precipitation/2016/era5_total_precipitation_20160101.nc'))
    
    ds.to_netcdf(fp, compute=True)
    return

from automl import data, fetch_era5
    
class TestLoad(unittest.TestCase):
    
    def test_load_era5(self):

        year = 2016
        month = 1
        day = 1
        hour = 12

        lat_coords = []
        lon_coords = []
        
        t0 = datetime.datetime(year=year, month=month, day=day, hour=hour)
        
        # Check error is raised if datetimes are too close together
        with self.assertRaises(ValueError):
            da1 = data.load_era5(var='total_precipitation_6hr', datetimes=[t0, t0 - datetime.timedelta(hours=1)],
                                    era_data_dir=DATA_FOLDER)

        # Check it works ok if they are spaced by 6hrs
        da1 = data.load_era5(var='total_precipitation_6hr', datetimes=[t0, t0 - datetime.timedelta(hours=6)],
                                    era_data_dir=DATA_FOLDER)

        vars = data.ERA5_STATIC_VARS + data.ERA5_SURFACE_VARS
        for v in tqdm(vars):
    
            da1 = data.load_era5(var=v, datetimes=[datetime.datetime(year=year, month=month, day=day, hour=hour)],
                                    era_data_dir=DATA_FOLDER)

            self.assertIsInstance(da1, xr.DataArray)
            self.assertEqual(da1.name, v)
            
            lat_var_name, lon_var_name = data.infer_lat_lon_names(da1)
            
            # Check no NaNs
            self.assertFalse(np.any(np.isnan(da1.values)))
            
            # check that lat lon are ascending
            self.assertListEqual(list(da1[lat_var_name].values), sorted(da1[lat_var_name].values))
            self.assertListEqual(list(da1[lon_var_name].values), sorted(da1[lon_var_name].values))
       
            lat_coords.append(tuple(sorted(da1.coords[lat_var_name].values)))
            lon_coords.append(tuple(sorted(da1.coords[lon_var_name].values)))
            
            time_vals = [pd.to_datetime(item) for item in da1['time'].values]
            self.assertListEqual(time_vals, [datetime.datetime(year=year, month=month, day=day, hour=hour)])
            
            # TODO: test precip sums over the right hours
            if v == 'total_precipitation_6hr':
                da = xr.load_dataarray(os.path.join(DATA_FOLDER, 'surface', 'total_precipitation', str(year), f"era5_total_precipitation_{year}{month:02d}{day:02d}.nc"))
                relevant_dates = pd.date_range(start=datetime.datetime(year,month,day,hour) - datetime.timedelta(hours=5), periods=6, freq='1h')
                
                self.assertEqual(max(relevant_dates), datetime.datetime(year,month,day,hour))
                da = da.sel(time=relevant_dates)
                
                self.assertEqual(np.round(da.values.sum(), 2), np.round(da1.values.sum(), 2))
            # else:
                

                
        # Check lat and long coordinates are all the same
        self.assertEqual(len(set(lat_coords)), 1)
        self.assertEqual(len(set(lon_coords)), 1)
        
        ### Pressure level vars
        lat_coords = []
        lon_coords = []

        vars = data.ERA5_PLEVEL_VARS
        for v in tqdm(vars):
            
            da1 = data.load_era5(var=v, datetimes=[datetime.datetime(year=year, month=month, day=day, hour=hour)],
                                    era_data_dir=DATA_FOLDER, pressure_levels=[1000, 850])

            self.assertIsInstance(da1, xr.DataArray)
            self.assertEqual(da1.name, v)
            
            lat_var_name, lon_var_name = data.infer_lat_lon_names(da1)
            
            # Check no NaNs
            self.assertFalse(np.any(np.isnan(da1.values)))
            
            # check that lat lon are ascending
            self.assertListEqual(list(da1[lat_var_name].values), sorted(da1[lat_var_name].values))
            self.assertListEqual(list(da1[lon_var_name].values), sorted(da1[lon_var_name].values))
    
            lat_coords.append(tuple(sorted(da1.coords[lat_var_name].values)))
            lon_coords.append(tuple(sorted(da1.coords[lon_var_name].values)))
            
            time_vals = [pd.to_datetime(item) for item in da1['time'].values]
            self.assertListEqual(time_vals, [datetime.datetime(year=year, month=month, day=day, hour=hour)])
                
        # Check lat and long coordinates are all the same
        self.assertEqual(len(set(lat_coords)), 1)
        self.assertEqual(len(set(lon_coords)), 1)


    def test_load_era5_static(self):

        year = 2016
        month = 1
        day = 1
        hour = 12

        ds = data.load_era5_static(year, month, day, hour, era5_data_dir=DATA_FOLDER)
        
        self.assertIsInstance(ds, xr.Dataset)
        
        self.assertListEqual(sorted(ds.coords), ['lat', 'lon'])
        
    def test_load_era5_surface(self):

        year = 2016
        month = 1
        day = 1
        hour = 18

        ds = data.load_era5_surface(year, month, day, hour, era5_data_dir=DATA_FOLDER)
        
        self.assertIsInstance(ds, xr.Dataset)
        
        self.assertListEqual(sorted(ds.coords), ['datetime', 'lat', 'lon', 'time'])
        self.assertDictEqual({'lon': 1440, 'lat': 721, 'time': 3, 'batch': 1}, dict(ds.dims))

        for v in ds.data_vars:
            self.assertFalse(np.any(np.isnan(ds[v].values)))
            
    def test_load_era5_plevels(self):

        year = 2016
        month = 1
        day = 1
        hour = 18

        ds = data.load_era5_plevel(year, month, day, hour, era5_data_dir=DATA_FOLDER)
        
        self.assertIsInstance(ds, xr.Dataset)
        
        self.assertListEqual(sorted(ds.coords), ['datetime', 'lat', 'level', 'lon', 'time'])
        self.assertDictEqual({'lon': 1440, 'lat': 721, 'level': 37, 'time': 3, 'batch': 1}, dict(ds.dims))

        for v in ds.data_vars:
            self.assertFalse(np.any(np.isnan(ds[v].values)))
    
    @patch.object(fetch_era5.cds_api_client, 'retrieve', mock_cds_api_client_retrieve) 
    def test_fetch_era5(self):
        
        fetch_era5.retrieve_data(year=2016, 
                    output_prefix='surface',
                    var='2m_temperature',
                    months=[1],
                    days=[1],
                    pressure_level=None,
                    output_resolution=1.0
                    )
        
        

if __name__ == '__main__':
    unittest.main()