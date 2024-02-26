import sys, os
import unittest
import yaml
from tqdm import tqdm
import datetime
import tempfile
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
from pathlib import Path

from unittest.mock import patch
from unittest import mock

HOME = Path(__file__).parents[1]
STATS_DIR = '/network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5_stats'
DATA_FOLDER = '/network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5'
DATA_FOLDER_LOWRES = '/network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5_1deg'

sys.path.append(str(HOME))

from automl import data, dataset, fetch_era5
from scripts.create_ml_data import load_data_by_datetime
from automl.utils import read_config, utils

data_config = read_config.read_data_config(config_filename='test_data_config.yaml',
                                        config_folder=str(HOME / 'config'))

class TestDataset(unittest.TestCase):

    def test_dataset(self):
        datetimes = [
                    datetime.datetime(2015,1,1,12),
                    datetime.datetime(2015,1,1,18),
                    datetime.datetime(2015,1,2,0)
                    ]
        
        ml_ds = dataset.ERA5_Dataset(dates = datetimes,
                 data_config=data_config,
                 shuffle=False, 
                 repeat_data=False, 
                 train=True)
        
        diffs_stddev_by_level = xr.load_dataset(os.path.join(STATS_DIR, "diffs_stddev_by_level.nc")).sel(level=data_config.pressure_levels).compute()
        mean_by_level = xr.load_dataset(os.path.join(STATS_DIR, "mean_by_level.nc")).sel(level=data_config.pressure_levels).compute()
        stddev_by_level = xr.load_dataset(os.path.join(STATS_DIR, "stddev_by_level.nc")).sel(level=data_config.pressure_levels).compute()

        for n, dt in enumerate(datetimes):
                    
            data_item = ml_ds[n]
            inputs = data_item[0]
            target_residuals = data_item[1]

            self.assertEqual(inputs[0].shape, inputs[1].shape)
            self.assertEqual(inputs[0].shape, (len(data_config.input_fields), 721, 1440))
            self.assertEqual(target_residuals.shape, (len(data_config.target_fields), 721, 1440))

            for n, f in enumerate(data_config.input_fields):
                da = data.load_era5(var=f, datetimes=[dt], era_data_dir=data_config.paths['ERA5'],
                                    pressure_levels=data_config.pressure_levels).compute()
                
                if f in data.ERA5_PLEVEL_VARS:
                    for pl in data_config.pressure_levels:
                        expected_val = (da.sel(level=pl).values[0,...] - mean_by_level[f].sel(level=pl).item()) / stddev_by_level[f].sel(level=pl).item()
                        self.assertTrue(np.allclose(expected_val, inputs[0][n,...]))
                else:
                    expected_val = (da.values[0,...] - mean_by_level[f].item()) / stddev_by_level[f].item()
                    self.assertTrue(np.allclose(expected_val, inputs[0][n,...]))

                da_plus6 = data.load_era5(var=f, datetimes=[dt - datetime.timedelta(hours=6)], era_data_dir=data_config.paths['ERA5'],
                                    pressure_levels=data_config.pressure_levels).compute()

                if f in data.ERA5_PLEVEL_VARS:
                    for pl in data_config.pressure_levels:
                        expected_val = (da_plus6.sel(level=pl).values[0,...] - mean_by_level[f].sel(level=pl).item()) / stddev_by_level[f].sel(level=pl).item()
                        self.assertTrue(np.allclose(expected_val, inputs[1][n,...]))
                else:
                    expected_val = (da_plus6.values[0,...] - mean_by_level[f].item()) / stddev_by_level[f].item()
                    self.assertTrue(np.allclose(expected_val, inputs[1][n,...]))
            
            # Check static derived fields
            
            
            for n, f in enumerate(data_config.target_fields):
                da = data.load_era5(var=f, datetimes=[dt + datetime.timedelta(hours=6)], era_data_dir=data_config.paths['ERA5'],
                                    pressure_levels=data_config.pressure_levels).compute()
                da_0 = data.load_era5(var=f, datetimes=[dt], era_data_dir=data_config.paths['ERA5'],
                                    pressure_levels=data_config.pressure_levels).compute()
                expected_val = (da.values[0,...] - da_0.values[0,...]) / diffs_stddev_by_level[f].item()
                self.assertTrue(np.allclose(expected_val, target_residuals[n,...]))

    def test_load_data_by_datetime(self):

        dt = datetime.datetime(2016,1,1,0)
        inputs, target_residuals = load_data_by_datetime(dt,
                                                data_config=data_config)

        # Check that data is as expected
        for n, f in enumerate(data_config.input_fields):
            da = data.load_era5(var=f, datetimes=[dt], era_data_dir=data_config.paths['ERA5'],
                                pressure_levels=data_config.pressure_levels).compute()
            self.assertTrue(np.allclose(da.values[0,...], inputs['input_t0'][n,...]))

            da_plus6 = data.load_era5(var=f, datetimes=[dt - datetime.timedelta(hours=6)], era_data_dir=data_config.paths['ERA5'],
                                pressure_levels=data_config.pressure_levels).compute()
            self.assertTrue(np.allclose(da_plus6.values[0,...], inputs['input_tm6h'][n,...]))
        
        for n, f in enumerate(data_config.target_fields):
            da = data.load_era5(var=f, datetimes=[dt + datetime.timedelta(hours=6)], era_data_dir=data_config.paths['ERA5'],
                                pressure_levels=data_config.pressure_levels).compute()
            da_0 = data.load_era5(var=f, datetimes=[dt], era_data_dir=data_config.paths['ERA5'],
                                pressure_levels=data_config.pressure_levels).compute()
            self.assertTrue(np.allclose(da.values[0,...] - da_0.values[0,...], target_residuals[n,...]))
