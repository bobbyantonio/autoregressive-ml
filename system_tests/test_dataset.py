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
DATA_FOLDER = '/network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5'
DATA_FOLDER_LOWRES = '/network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5_1deg'

sys.path.append(str(HOME))

from automl import data, dataset, fetch_era5
from scripts.create_ml_data import load_data_by_datetime
from automl.utils import read_config, utils

class TestDataset(unittest.TestCase):

    def test_dataset(self):
        pass

    def test_load_data_by_datetime(self):
        data_config = read_config.read_data_config(config_filename='test_data_config.yaml',
                                                config_folder=str(HOME / 'config'))
        dt = datetime.datetime(2016,1,1,0)
        inputs, targets = load_data_by_datetime(dt,
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
            self.assertTrue(np.allclose(da.values[0,...], targets[n,...]))
