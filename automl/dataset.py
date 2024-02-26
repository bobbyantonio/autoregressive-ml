import os
import torch
import datetime
import calendar
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
from typing import Union, Iterable
from types import SimpleNamespace

from torchvision.transforms import v2

from automl import data

SECONDS_IN_DAY = 24*60*60

class ERA5_Dataset(Dataset):
    def __init__(self, 
                 dates: list, 
                 data_config: SimpleNamespace,
                 shuffle: bool=True, 
                 repeat_data: bool=False, 
                 train: bool = True,
                 seed: int=None):
        # TODO: add cosine of the latitude, and the sine and cosine of the longitude
        # TODO: allow .nc input (for when testing and doing inference)
        self.train = train

        # TODO: implement this via Torch
        self.shuffle = shuffle
        
        if self.shuffle:
            rng = np.random.default_rng(seed=seed)
            dates = rng.shuffle(dates)
        self.dates = dates

        self.repeat_data = repeat_data

        self.input_fields = data_config.input_fields
        self.target_fields = data_config.target_fields
        self.pressure_levels = data_config.pressure_levels
        self.derived_static_fields = ['']

        self.data_dir = data_config.paths['ml_data']
        self.stats_dir = data_config.paths['ERA5_stats']

        self.latitude_vals = np.arange(-90, 90 + data_config.latitude_step_size, data_config.latitude_step_size)
        self.longitude_vals = np.arange(0, 360, data_config.longitude_step_size)

        # Load stats for normalising
        self._diffs_stddev_by_level = xr.load_dataset(os.path.join(self.stats_dir, "diffs_stddev_by_level.nc")).sel(level=data_config.pressure_levels).compute()
        self._mean_by_level = xr.load_dataset(os.path.join(self.stats_dir, "mean_by_level.nc")).sel(level=data_config.pressure_levels).compute()
        self._stddev_by_level = xr.load_dataset(os.path.join(self.stats_dir, "stddev_by_level.nc")).sel(level=data_config.pressure_levels).compute()

        # Select the right input normalising values in order
        field_list = []
        self.input_normalising_means = []
        self.input_normalising_stddev = []
        
        for f in data_config.input_fields:
            if f in data.ERA5_PLEVEL_VARS:
                field_list += [(f, pl) for pl in sorted(data_config.pressure_levels)]
                self.input_normalising_means += [self._mean_by_level[f].sel(level=pl).item() for pl in sorted(data_config.pressure_levels)]
                self.input_normalising_stddev += [self._stddev_by_level[f].sel(level=pl).item() for pl in sorted(data_config.pressure_levels)]
            else:
                field_list += [(f, None)]
                self.input_normalising_means += [self._mean_by_level[f].item()]
                self.input_normalising_stddev += [self._stddev_by_level[f].item()]

        # Derived fields assumed to be Sin and Cos functions
        self.input_normalising_means += [0] * len(data.GENERATED_FORCING_VARS)
        self.input_normalising_stddev += [1 / np.sqrt(2)] * len(data.GENERATED_FORCING_VARS)
        
        self.target_normalising_diffs = []
        for f in data_config.target_fields:
            if f in data.ERA5_PLEVEL_VARS:
                self.target_normalising_diffs += [self._diffs_stddev_by_level[f].sel(level=pl).item() for pl in sorted(data_config.pressure_levels)]
            else:
                self.target_normalising_diffs += [self._diffs_stddev_by_level[f].item()]

        self.input_transform = v2.Compose([
            v2.Normalize(mean=self.input_normalising_means, std=self.input_normalising_stddev),
            ])
        
        # TODO: use pixel specific mean and variance
        self.target_transform = v2.Compose([
            v2.Normalize(mean=[0]*len(self.target_normalising_diffs), std=self.target_normalising_diffs),
            ])  

    def __len__(self):
        return len(self.dates)
    

    def create_derived_static_fields(self,
                                     dt: datetime.date):

        derived_fields = {}
        # Inputs derived from lat/lon
        lon_broadcast = np.broadcast_to(self.longitude_vals, (len(self.latitude_vals), len(self.longitude_vals)))
        derived_fields['longitude_sin'] = np.sin(lon_broadcast * 2*np.pi / 360)
        derived_fields['longitude_cos'] = np.cos(lon_broadcast * 2*np.pi / 360)

        derived_fields['latitude_sin'] = np.sin(np.broadcast_to(self.latitude_vals, ( len(self.longitude_vals), len(self.latitude_vals))).transpose() * 2*np.pi / 360)

        year_progress = data.get_year_progress(dt) * np.ones((len(self.latitude_vals), len(self.longitude_vals)))
        derived_fields['year_progress_sin'] = np.sin(year_progress * 2*np.pi)
        derived_fields['year_progress_cos'] = np.cos(year_progress * 2*np.pi)

        day_progress = data.get_day_progress(dt)

        lon_broadcast_phase = 2*np.pi * lon_broadcast / 360
        day_progress_phase = day_progress * 2*np.pi

        derived_fields['day_progress_sin'] = np.sin(day_progress_phase + lon_broadcast_phase)
        derived_fields['day_progress_cos'] = np.cos(day_progress_phase + lon_broadcast_phase)

        # Concatenate arrays and return. Use dict to preserve order
        return torch.stack([torch.tensor(derived_fields[k]) for k in data.GENERATED_FORCING_VARS], 0)    

    def __getitem__(self, idx):

        dt = self.dates[idx]
        dt_m6 = dt - datetime.timedelta(hours=6)

        inputs_fp_t0 = os.path.join(self.data_dir, f"{'train' if self.train else 'validation'}", f"inputs_era5_{dt.strftime('%Y%m%d_%H')}.pt")
        inputs_fp_tm6 = os.path.join(self.data_dir, f"{'train' if self.train else 'validation'}", f"inputs_era5_{dt_m6.strftime('%Y%m%d_%H')}.pt")
        targets_fp = os.path.join(self.data_dir, f"{'train' if self.train else 'validation'}", f"targets_era5_{dt.strftime('%Y%m%d_%H')}.pt")

        inputs_t0 = torch.load(inputs_fp_t0)
        inputs_tm6 = torch.load(inputs_fp_tm6)
        target_residuals = torch.load(targets_fp)

        static_derived_inputs_t0 = self.create_derived_static_fields(dt)
        static_derived_inputs_tm6 = self.create_derived_static_fields(dt - datetime.timedelta(hours=6))

        inputs_t0 = self.input_transform(torch.cat([inputs_t0, static_derived_inputs_t0])).type(torch.FloatTensor)
        inputs_tm6 = self.input_transform(torch.cat([inputs_tm6, static_derived_inputs_tm6])).type(torch.FloatTensor)

        inputs = torch.stack([inputs_t0, inputs_tm6])

        return inputs, self.target_transform(target_residuals)
