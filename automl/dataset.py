import os
import numpy as np
from torch.utils.data import Dataset
from typing import Union, Iterable
from types import SimpleNamespace


class ERA5_Dataset(Dataset):
    def __init__(self, 
                 dates: list, 
                 batch_size: int, 
                 data_config: SimpleNamespace,
                 shuffle: bool=True, 
                 repeat_data: bool=False, 
                 train: bool = True,
                 seed: int=None):

        self.train = train

        # TODO: implement this via Torch
        self.shuffle = shuffle
        
        if self.shuffle:
            rng = np.random.default_rng(seed=seed)
            dates = rng.shuffle(dates)
        self.dates = dates

        self.batch_size = batch_size
        self.repeat_data = repeat_data

        self.input_fields = data_config.input_surface_fields
        self.target_fields = data_config.target_fields
        self.pressure_levels = data_config.pressure_levels

        self.data_dir = data_config.paths['ml_data']

    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, idx):

        dt = self.dates[idx]
        fp = os.path.join(self.data_dir, f"{'train' if self.train else 'validation'}", f"era5_{dt.strftime('%Y%m%d_%H')}.npz")
        
        data_dict = np.load(fp)

        # {'input_t0': dt, 'input_tm6h'}

        # return data, targets
