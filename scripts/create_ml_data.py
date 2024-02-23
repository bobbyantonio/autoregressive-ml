# Aim of this script: to save easily loaded copies of the training/test data
import os, sys
import git
import datetime
import torch
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from calendar import monthrange
from argparse import ArgumentParser
import warnings

import numpy as np
from automl.utils import read_config, utils

HOME = Path(__file__).parents[1]

sys.path.append( str(HOME ))

from automl import data

float_type_lookup = {16: np.float16, 32: np.float32}

def load_data_by_datetime(dt: datetime.datetime, 
                          data_config: SimpleNamespace,
                          change_types: bool=False):

    # Gather data at t=0 and t=-6h for inputs, +6h for targets
    arrs = {}
    input_dts = {'input_t0': {'datetime': dt, 'fields': data_config.input_fields}, 
                'input_tm6h': {'datetime': dt - datetime.timedelta(hours=6), 'fields': data_config.input_fields},
                'targets': {'datetime': dt + datetime.timedelta(hours=6), 'fields': data_config.target_fields}}

    for datatype, tmp_dict in input_dts.items():
        
        arrs[datatype] = []
        tmp_dt = tmp_dict['datetime']

        for field in tmp_dict['fields']:
            
            if change_types:
                output_type = np.float16 if field in data_config.float16_fields else np.float32
            else:
                output_type = np.float32

            da = data.load_era5(var=field, datetimes=[tmp_dt], era_data_dir=data_config.paths['ERA5'],
                                pressure_levels=data_config.pressure_levels).compute()
            da = data.format_dataarray(da).drop_vars('time').astype(output_type)

            # Check for numerical over/under flow
            if np.isinf(da.values).any():
                raise TypeError(f'Precision mismatch for var {field}')

            if field in data.ERA5_PLEVEL_VARS:
                for pl in sorted(data_config.pressure_levels):
                    
                    arrs[datatype].append(da.sel(level=pl).drop_vars('level').values)
            else:
                arrs[datatype].append(da.values)

        arrs[datatype] = np.concatenate(arrs[datatype])

    inputs = {k: v for k, v in arrs.items() if k.startswith('input')}
    targets = arrs['targets']

    return inputs, targets

if __name__ == '__main__':
    
    parser = ArgumentParser(description='Write normalised data to npy files.')

    parser = ArgumentParser()
    parser.add_argument('--data-config-path', type=str, required=True)
    parser.add_argument('--debug',action='store_true')
    args = parser.parse_args()

    path_split = os.path.split(args.data_config_path)
    data_config = read_config.read_data_config(config_filename=path_split[-1],
                                                config_folder=path_split[0])

    years_dict = {'train': data_config.train_years,
                  'validation': data_config.validation_years}
    
    if hasattr(data_config, 'test_years'):
        years_dict['test'] = data_config.test_years

    for data_label, years in years_dict.items():
        print(f'Writing {data_label} data')


        for y in years:
            print(f'year = {y}')
            for month in tqdm(range(1,13)):
                print(f'Month = {month}')

                all_input_datetimes = list(pd.date_range(start=datetime.datetime(int(y), month, 1), 
                                                    end=datetime.datetime(int(y), month, monthrange(int(y), month)[1]), freq='6h'))

                for dt in tqdm(all_input_datetimes):
                    
                    # Exclude datapoints at extreme ends of year, to avoid mixing with other datasets
                    # (Assumes that data is split by year)
                    if (dt - datetime.timedelta(hours=6)).year != y or (dt + datetime.timedelta(hours=6)).year != y:
                        continue

                    inputs, targets = load_data_by_datetime(dt=dt, data_config=data_config)
                        
                    #####################
                    # Save data and metadata
                    output_dir = os.path.join(data_config.paths['ml_data'], data_label)
                    inputs_fp = os.path.join(output_dir, f"inputs_era5_{dt.strftime('%Y%m%d_%H')}.pt")
                    targets_fp = os.path.join(output_dir, f"targets_era5_{dt.strftime('%Y%m%d_%H')}.pt")
                    os.makedirs(output_dir, exist_ok=True)

      
                    # TODO: saving a concatenated np array to a .npy file is faster than this
                    # but I expect there to be advantages of the torch loading function for ML applications.
                    # Something to experiment with if IO is slow, but order of ms so probably not a big deal
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning) # To filter out PyTorch user warning
                        input_tensor = torch.tensor([inputs['input_t0'], inputs['input_tm6h']])
                        target_tensor = torch.tensor(targets)

                    torch.save(input_tensor, inputs_fp)
                    torch.save(target_tensor, targets_fp)

                    # Save metadata alongside
                    metadata_dict = {'vars': vars,
                                    'pressure_levels': sorted(data_config.pressure_levels)}

                    utils.write_to_yaml(fpath=inputs_fp.replace('.pt', '.yml').replace('inputs', 'metadata'), data=metadata_dict)
                
    # Finally write data config and git commit to folder
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with open(os.path.join(args.output_dir, 'git_commit.txt'), 'w+') as ofh:
        ofh.write(sha)

    utils.write_to_yaml(os.path.join(args.output_dir, 'data_config.yaml'), utils.convert_namespace_to_dict(data_config))