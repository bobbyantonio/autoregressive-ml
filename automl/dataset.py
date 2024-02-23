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

        # Create list of variables
        field_list = []
        for f in data_config.input_fields:
            if f in data.ERA5_PLEVEL_VARS:
                field_list += [f'{f}_{pl}' for pl in data_config.pressure_levels]
            else:
                field_list += [f]

        self.data_dir = data_config.paths['ml_data']
        self.stats_dir = data_config.paths['ERA5_stats']

        # Load stats for normalising
        self.diffs_stddev_by_level = xr.load_dataset(os.path.join(self.stats_dir, "diffs_stddev_by_level.nc")).sel(level=data_config.pressure_levels).compute()
        self.mean_by_level = xr.load_dataset(os.path.join(self.stats_dir, "mean_by_level.nc")).sel(level=data_config.pressure_levels).compute()
        self.stddev_by_level = xr.load_dataset(os.path.join(self.stats_dir, "stddev_by_level.nc")).sel(level=data_config.pressure_levels).compute()

    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, idx):

        dt = self.dates[idx]
        inputs_fp = os.path.join(self.data_dir, f"{'train' if self.train else 'validation'}", f"inputs_era5_{dt.strftime('%Y%m%d_%H')}.pt")
        targets_fp = os.path.join(self.data_dir, f"{'train' if self.train else 'validation'}", f"targets_era5_{dt.strftime('%Y%m%d_%H')}.pt")

        inputs = torch.load(inputs_fp)
        targets = torch.load(targets_fp)

        # TODO: transforms
        transforms = v2.Compose([
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        return inputs, targets
