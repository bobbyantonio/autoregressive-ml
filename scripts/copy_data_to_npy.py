# Aim of this script: to save easily loaded copies of the training/test data
import os, sys
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from calendar import monthrange
from argparse import ArgumentParser

import numpy as np
from automl.utils import read_config, utils

HOME = Path(__file__).parents[1]

sys.path.append( str(HOME ))

from automl import data

float_type_lookup = {16: np.float16, 32: np.float32}

def save_array_to_separate_hours(output_dir, da, var_name, output_type):

    lat_var_name, lon_var_name = data.get_lat_lon_names(da)
    for tmp_da in da.transpose('time', ...):
        tmp_time = pd.Timestamp(tmp_da['time'].item()).to_pydatetime()
        tmp_da = data.format_dataarray(tmp_da).drop_vars('time')

        output_fp = os.path.join(output_dir, f"era5_{var_name}_{tmp_time.strftime('%Y%m%d_%H')}.npy")
        np.save(output_fp, tmp_da.values.astype(output_type))

        lat_vals = tmp_da[lat_var_name].values
        lon_vals = tmp_da[lon_var_name].values
        # Save metadata alongside
        metadata_dict = {'min_lat': str(min(lat_vals)),
                        'max_lat': str(max(lat_vals)),
                        'min_lon': str(min(lon_vals)),
                        'max_lon': str(max(lon_vals)),
                        'lat_step_size': str(lat_vals[1] - lat_vals[0]),
                        'lon_step_size': str(lon_vals[1] - lon_vals[0]),
                        'pressure_levels': [] if 'level' not in tmp_da.coords else [str(int(l)) for l in tmp_da.level.values]}

        utils.write_to_yaml(fpath=output_fp.replace('.npy', '.yml'), data=metadata_dict)

if __name__ == '__main__':
    
    parser = ArgumentParser(description='Write normalised data to npy files.')

    parser = ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Folder to save data to")
    parser.add_argument('--data-config-path', type=str, required=True)
    parser.add_argument('--debug',action='store_true')
    args = parser.parse_args()

    path_split = os.path.split(args.data_config_path)
    data_config = read_config.read_data_config(config_filename=path_split[-1],
                                                config_folder=path_split[0])

    years_dict = {'train': data_config.train_years,
                  'validation': data_config.validation_years,
                  'test': data_config.test_years}
    
    data_type_dict = {'sur'}

    for data_label, years in years_dict.items():
        print(f'Writing {data_label} data')


        for y in years:
            for month in tqdm(range(1,13)):

                all_datetimes = list(pd.date_range(start=datetime.datetime(int(y), month, 1), 
                                                    end=datetime.datetime(int(y), month, monthrange(int(y), month)[1]), freq='6h'))
                
                for var in data_config.input_atmospheric_fields:

                    output_type = np.float16 if var in data_config.float16_fields else np.float32
                    print(var)
                    da = data.load_era5(var=var, datetimes=all_datetimes, era_data_dir=data_config.paths['ERA5'],
                                        pressure_levels=data_config.pressure_levels).compute()

                    output_dir = os.path.join(args.output_dir, f'{data_label}/plevels/{var}/{y}/{month}')

                    os.makedirs(output_dir, exist_ok=True)

                    save_array_to_separate_hours(var_name=var, output_dir=output_dir, da=da, output_type=output_type)
                
                # for var in data_config.input_surface_fields:

                #     output_type = np.float16 if var in data_config.float16_fields else np.float32
                #     print(var)
                #     da = data.load_era5(var=var, datetimes=all_datetimes, era_data_dir=data_config.paths['ERA5']).compute()

                #     output_dir = os.path.join(args.output_dir, data_label, 'surface', var, y, str(month))

                #     os.makedirs(output_dir, exist_ok=True)

                #     save_array_to_separate_hours(var_name=var, output_dir=output_dir, da=da, output_type=output_type)