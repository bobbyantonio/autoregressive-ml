# Note that this requires setup of an API key and installing cdsapi:
# Go to: https://cds.climate.copernicus.eu/api-how-to
import os, sys
import xarray as xr
from tqdm import tqdm
import numpy as np
from pathlib import Path
from calendar import monthrange
from argparse import ArgumentParser

HOME = Path(__file__).parents[1]

sys.path.append( str(HOME ))

from automl import data, fetch_era5


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Folder to save data to")
    parser.add_argument('--resolution', type=float, required=True,
                help='Resolution to save to (will regrid if != 0.25)')  
    parser.add_argument('--years', nargs='+', required=True,
                        help='Year(s) to collect data for')
    parser.add_argument('--months', nargs='+', default=range(1,13),
                        help='Months to collect data for')
    parser.add_argument('--days', nargs='+', default=range(1,32),
                    help='Days to collect data for') 

    args = parser.parse_args()
    
    if args.resolution == 0.25:
        raise ValueError('ERA5 dataset already stored in 0.25 resolution')
    
    if data.DATASET_FOLDER == args.output_dir:
        raise ValueError('Output folder mush be different from input')
    
    var_collections = {'surface': data.ERA5_SURFACE_VARS + data.ERA5_STATIC_VARS, 'plevels': data.ERA5_PLEVEL_VARS}
    
    for k, vars in var_collections.items():
        print(f'Regridding {k} variables')
        for var in tqdm(vars):

            era5_var = data.ERA5_VARNAME_LOOKUP.get(var, var)
            for year in args.years:
                for month in args.months:
                    days = fetch_era5.format_days(year, month, args.days)

                    for day in days:
                    
                        date_str = f'{year}{int(month):02d}{int(day):02d}'
                        suffix = f'era5_{era5_var}_{date_str}.nc'

                        fp_in = os.path.join(data.DATASET_FOLDER, k, era5_var, year,  suffix)
                        
                        folder_out = os.path.join(args.output_dir, k, era5_var, year)
                        os.makedirs(folder_out, exist_ok=True)
                        fp_out = os.path.join(folder_out, suffix)

                        if not os.path.exists(fp_out):

                            ds = xr.load_dataset(fp_in)
                            interp_ds = data.interpolate_dataset_on_lat_lon(ds, 
                                                    latitude_vals=np.arange(-90, 90, args.resolution) , 
                                                    longitude_vals=np.arange(0,360, args.resolution),
                                                    interp_method ='conservative')
                            interp_ds.to_netcdf(fp_out)                