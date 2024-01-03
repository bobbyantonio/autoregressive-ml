import sys, os
import unittest
from tqdm import tqdm
import datetime
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

HOME = Path(__file__).parents[1]
data_folder = str(HOME / 'dataset')

sys.path.append(str(HOME))


from automl import data
    
class TestLoad(unittest.TestCase):
    
    def test_load_era5(self):

        year = 2016
        month = 1
        day = 1
        hour = 12

        lat_coords = []
        lon_coords = []

        # vars = data.ERA5_SURFACE_VARS
        vars = data.ERA5_STATIC_VARS + data.ERA5_SURFACE_VARS
        # for v in tqdm(['total_precipitation_6hr']):
        for v in tqdm(vars):
    
            da1 = data.load_era5(var=v, year=year, month=month, day=day, hour=hour,
                                    era_data_dir=data_folder)

            self.assertIsInstance(da1, xr.DataArray)
            
            lat_var_name, lon_var_name = data.infer_lat_lon_names(da1)
            
            # Check no NaNs
            self.assertFalse(np.any(np.isnan(da1.values)))
            
            # check that lat lon are ascending
            self.assertListEqual(list(da1[lat_var_name].values), sorted(da1[lat_var_name].values))
            self.assertListEqual(list(da1[lon_var_name].values), sorted(da1[lon_var_name].values))
       
            lat_coords.append(tuple(sorted(da1.coords[lat_var_name].values)))
            lon_coords.append(tuple(sorted(da1.coords[lon_var_name].values)))
            
            # TODO: test precip sums over the right hours
            if v == 'total_precipitation_6hr':
                da = xr.load_dataarray(os.path.join(data_folder, 'surface', f"era5_total_precipitation_{year}{month:02d}.nc"))
                relevant_dates = pd.date_range(start=datetime.datetime(year,month,day,hour) - datetime.timedelta(hours=5), periods=6, freq='1h')
                
                self.assertEqual(max(relevant_dates), datetime.datetime(year,month,day,hour))
                da = da.sel(time=relevant_dates)
                
                self.assertEqual(np.round(da.values.sum(), 2), np.round(da1.values.sum(), 2))
                
        # Check lat and long coordinates are all the same
        self.assertEqual(len(set(lat_coords)), 1)
        self.assertEqual(len(set(lon_coords)), 1)

        
if __name__ == '__main__':
    unittest.main()