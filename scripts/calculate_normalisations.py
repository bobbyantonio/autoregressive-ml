import os, sys
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

HOME = Path(__file__).parents[1]

sys.path.append( str(HOME))

from automl import data
from automl.utils import read_config, utils


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-config-path', type=str, required=True)
    parser.add_argument('--debug',action='store_true')
    args = parser.parse_args()

    path_split = os.path.split(args.data_config_path)
    data_config = read_config.read_data_config(config_filename=path_split[-1],
                                                config_folder=path_split[0])
    norm_years = data_config.normalisation_years

    all_datetimes = []
    for y in norm_years:
        all_datetimes += list(pd.date_range(start=datetime.datetime(y, 1, 1), 
                                            end=datetime.datetime(y, 12, 31), freq='6h'))
    all_datetimes = sorted(set(all_datetimes))
    all_datetimes_plus_6hr = [item + datetime.timedelta(hours=6) for item in all_datetimes]

    for var in data_config.input_surface_fields:
        
        vals = data.load_era5(var=var, datetimes=all_datetimes, era_data_dir=data_config.paths['ERA5']).values

        mean = np.mean(vals)
        std = np.std(vals)

    for var in data_config.input_atmospheric_fields:
        for pl in data_config.pressure_levels:
            pass

    for var in data_config.input_static_fields:

        pass

    
