import sys, os
import unittest
from tqdm import tqdm
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from numpy import testing

HOME = Path(__file__).parents[1]
data_folder = str(HOME / 'dataset')

sys.path.append(str(HOME))


from automl import data
    
class TestLoad(unittest.TestCase):
    
    def test_load_era5_surface(self):

        year = 2016
        month = 1
        day = 1
        hour = 12

        lat_coords = []
        lon_coords = []

        vars = data.ERA5_SURFACE_VARS

        for v in tqdm(vars):
    
            ds1 = data.load_era5(var=v, year=year, month=month, day=day, hour=hour,
                                    era_data_dir=data_folder)

            self.assertIsInstance(ds1, xr.Dataset)
            
            lat_var_name, lon_var_name = data.infer_lat_lon_names(ds1)
            
            # check that lat lon are ascending
            self.assertListEqual(list(ds1[lat_var_name].values), sorted(ds1[lat_var_name].values))
            self.assertListEqual(list(ds1[lon_var_name].values), sorted(ds1[lon_var_name].values))
       
            lat_coords.append(tuple(sorted(ds1.coords[lat_var_name].values)))
            lon_coords.append(tuple(sorted(ds1.coords[lon_var_name].values)))

        # Check lat and long coordinates are all the same
        self.assertEqual(len(set(lat_coords)), 1)
        self.assertEqual(len(set(lon_coords)), 1)

        
if __name__ == '__main__':
    unittest.main()