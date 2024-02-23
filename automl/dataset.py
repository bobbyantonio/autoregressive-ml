import os
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
from typing import Union, Iterable
from types import SimpleNamespace

from torchvision.transforms import v2

from automl import data

class ERA5_Dataset(Dataset):
    def __init__(self, 
                 dates: list, 
                 data_config: SimpleNamespace,
                 shuffle: bool=True, 
                 repeat_data: bool=False, 
                 train: bool = True,
                 seed: int=None):
        # TODO: lat long and time inputs.
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

        self.data_dir = data_config.paths['ml_data']
        self.stats_dir = data_config.paths['ERA5_stats']

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

        self.target_normalising_diffs = []
        for f in data_config.target_fields:
            if f in data.ERA5_PLEVEL_VARS:
                self.target_normalising_diffs += [self._diffs_stddev_by_level[f].sel(level=pl).item() for pl in sorted(data_config.pressure_levels)]
            else:
                self.target_normalising_diffs += [self._diffs_stddev_by_level[f].item()]


    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, idx):

        dt = self.dates[idx]
        inputs_fp = os.path.join(self.data_dir, f"{'train' if self.train else 'validation'}", f"inputs_era5_{dt.strftime('%Y%m%d_%H')}.pt")
        targets_fp = os.path.join(self.data_dir, f"{'train' if self.train else 'validation'}", f"targets_era5_{dt.strftime('%Y%m%d_%H')}.pt")

        inputs = torch.load(inputs_fp)
        target_residuals = torch.load(targets_fp)

        # TODO: transforms
        input_transform = v2.Compose([
            v2.Normalize(mean=self.input_normalising_means, std=self.input_normalising_stddev),
            ])
        
        # TODO: use pixel specific mean and variance
        target_transform = v2.Compose([
            v2.Normalize(mean=[0]*len(self.target_normalising_diffs), std=self.target_normalising_diffs),
            ])
        

        return input_transform(inputs), target_transform(target_residuals)
