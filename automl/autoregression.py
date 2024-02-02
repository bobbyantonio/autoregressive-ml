'''
Script for rolling out graphcast predictions

On Oxford's A100, it currently runs out of memory when trying to load in enough data for between 80-100 time steps (80 works, 100 doesn't)

'''

import dataclasses
import functools
from types import SimpleNamespace
from argparse import ArgumentParser

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast as gc
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree

import os, sys
import datetime
import haiku as hk
import jax
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
import pandas as pd
import logging
from pathlib import Path
from jax.lib import xla_bridge
from calendar import monthrange

HOME = Path(__file__).parents[1]
sys.path.append(str(HOME))

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

sys.path.append('/home/a/antonio/repos/graphcast-ox/data_prep')

from automl import data

DATASET_FOLDER = '/network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5'
GRAPHCAST_DIR = '/home/a/antonio/repos/graphcast-ox'
OUTPUT_VARS = [
    '2m_temperature', 'total_precipitation_6hr', '10m_v_component_of_wind', '10m_u_component_of_wind', 'specific_humidity', 'temperature'
]
NONEGATIVE_VARS = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
    "temperature",
    "specific_humidity",
]

params_file = SimpleNamespace(value=os.path.join(GRAPHCAST_DIR, 'params/params_GraphCast-ERA5_1979-2017-resolution_0.25-pressure_levels_37-mesh_2to6-precipitation_input_and_output.npz'))

    # @title Load the model
with open(f"{params_file.value}", 'rb') as f:
    ckpt = checkpoint.load(f, gc.CheckPoint)
params = ckpt.params

model_config = ckpt.model_config
task_config = ckpt.task_config

with open(os.path.join(GRAPHCAST_DIR, "stats/diffs_stddev_by_level.nc"),"rb") as f:
    diffs_stddev_by_level = xr.load_dataset(f).compute()
with open(os.path.join(GRAPHCAST_DIR, "stats/mean_by_level.nc"), "rb") as f:
    mean_by_level = xr.load_dataset(f).compute()
with open(os.path.join(GRAPHCAST_DIR, "stats/stddev_by_level.nc"),"rb") as f:
    stddev_by_level = xr.load_dataset(f).compute()

def get_all_visible_methods(obj):
    return [item for item in dir(obj) if not item.startswith('_')]

def unpack_np_datetime(dt):
    
    converted_dt = pd.to_datetime(dt)
    
    return converted_dt.year, converted_dt.month, converted_dt.day, converted_dt.hour

def format_dataarray(da):
    
    rename_dict = {'latitude': 'lat', 'longitude': 'lon' }

    da = da.rename(rename_dict)
    
    da = da.sortby('lat', ascending=True)
    da = da.sortby('lon', ascending=True)
    
    return da

def load_clean_dataarray(fp, add_batch_dim=False):
    
    da = xr.load_dataarray(fp)
    da = format_dataarray(da)
    
    if add_batch_dim:
        da = da.expand_dims({'batch': 1})

    return da

def add_datetime(ds, start: str=None,
                 periods: int=None,
                 dt_arr: np.array=None,
                 freq: str='6h',
                 ):
    if dt_arr is None:
        dt_arr = np.expand_dims(pd.date_range(start=start, periods=periods, freq=freq),0)
    else:
        dt_arr = np.expand_dims(dt_arr,0)
        
    ds = ds.assign_coords(datetime=(('batch', 'time'),  dt_arr))
    return ds

def convert_to_relative_time(ds, zero_time: np.datetime64):
    
    ds['time'] = ds['time'] - zero_time
    return ds

def ns_to_hr(ns_val: float):
    
    return ns_val* 1e-9 / (60*60)


################################
## Functions from graphcast notebook
def construct_wrapped_graphcast(
    model_config: gc.ModelConfig,
    task_config: gc.TaskConfig):
    """Constructs and wraps the GraphCast Predictor."""
    
    # Deeper one-step predictor.
    predictor = gc.GraphCast(model_config, task_config)

    # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
    # from/to float32 to/from BFloat16.
    predictor = casting.Bfloat16Cast(predictor)

    # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
    # BFloat16 happens after applying normalization to the inputs/targets.
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level)

    # Wraps everything so the one-step model can produce trajectories.
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    
    return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    
    predictor = construct_wrapped_graphcast(model_config, task_config)
    
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
    
    predictor = construct_wrapped_graphcast(model_config, task_config)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    
    return xarray_tree.map_structure(
        lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
    def _aux(params, state, i, t, f):
        
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, jax.random.PRNGKey(0), model_config, task_config,
            i, t, f)
        return loss, (diagnostics, next_state)
    
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
        _aux, has_aux=True)(params, state, inputs, targets, forcings)
    return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
    return functools.partial(
        fn, model_config=model_config, task_config=task_config)

state = {}
# Always pass params and state, so the usage below are simpler
def with_params(fn):
    return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
    return lambda **kw: fn(**kw)[0]


run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
        run_forward.apply))))



if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Folder to save data to")
    parser.add_argument('--num-steps', type=int, required=True,
                        help='Number of autoregressive steps to run for')
    parser.add_argument('--year', type=int, required=True,
                        help='Year to collect data for')
    parser.add_argument('--month', type=int, default=1,
                        help='Month to start on')
    parser.add_argument('--day', type=int, default=1,
                    help='Day of month to start on')
    parser.add_argument('--hour-start', default='random',
                    help='First target time to predict for') 
    parser.add_argument('--load-era5', action='store_true',
                        help='Load from ERA5') 
    parser.add_argument('--var-to-replace', type=str, default=None,
                        help="Variable to replace with ERA5 input during autoregression",
                        choices=list(gc.TARGET_SURFACE_VARS) + ['specific_humidity', 'temperature'] # For now limit to the surface vars
                        )
    parser.add_argument('--cache-inputs', action='store_true',
                        help='If active, then inputs will be cached to allow fast iteration')
    parser.add_argument('--replace-uses-lsm', action='store_true',
                    help='If active, then variable replacement only occurs over sea/ocean points')
    parser.add_argument('--interpolate-to-1000hPa', action='store_true',
                help='If active, will interpolate temperature with 1000hPa value')
    args = parser.parse_args()
    year = args.year
    month = args.month
    day = args.day

    
    if day == -1:
        day = monthrange(year=year, month=month)[-1]
        
    if args.hour_start == 'random':
        if day == 1 and month ==1:
            # Put here to prevent data issues when the first day in the data is selected
            hour_start = 18
        else:
            hour_start = np.random.choice([0,6,12,18])
    else:
        hour_start = int(args.hour_start)
    
    print(f'Running for {year}-{month:02d}-{day:02d} {hour_start}')
    
    logger.info(f'Platform: {xla_bridge.get_backend().platform}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    all_datetimes = [datetime.datetime(year, month, day, hour_start) + 
                            datetime.timedelta(hours=6*n - 12) for n in range(args.num_steps+2)]
    time_lookup = { int((d - all_datetimes[1]).total_seconds()*1e9) : d for d in all_datetimes}

    if not args.cache_inputs or not os.path.exists('cached_inputs.nc'):
        ########
        # Static variables
        static_ds = data.load_era5_static(year=year, month=month, day=day, hour=hour_start, era5_data_dir=DATASET_FOLDER)

        ######
        # Surface
        surface_ds = data.load_era5_surface(year=year, month=month, day=day, hour=hour_start, era5_data_dir=DATASET_FOLDER)

        #############
        # Pressure levels 
        plevel_ds = data.load_era5_plevel(year=year, month=month, day=day, hour=hour_start, era5_data_dir=DATASET_FOLDER)
        prepared_ds = xr.merge([static_ds, surface_ds, plevel_ds])
        prepared_ds = convert_to_relative_time(prepared_ds, prepared_ds['time'][1])

        ##############
        # Load forcing data for targets
        print('--- Loading forcings ', flush=True)
        
        t0 = prepared_ds['datetime'][0][1].values
        
        ds_slice = prepared_ds.isel(time=slice(-1, None))
        ds_final_datetime = ds_slice['datetime'][0][0].values
        
        dts_to_fill = [ds_final_datetime + np.timedelta64(6*n, 'h') for n in range(1, args.num_steps)]

        solar_rad_das = []
        for dt in dts_to_fill:
            y, m, d, h = unpack_np_datetime(dt)
            tmp_da = data.load_era5(var='toa_incident_solar_radiation',
                                                datetimes=[datetime.datetime(year=y,
                                                                            month=m,
                                                                            day=d,
                                                                            hour=h)],
                                                era_data_dir=DATASET_FOLDER).load()
            tmp_da = tmp_da.expand_dims({'batch': 1})
            solar_rad_das.append(tmp_da)

        future_forcings = xr.concat(solar_rad_das, dim='time').to_dataset()
        future_forcings = add_datetime(future_forcings, dt_arr=dts_to_fill)
        future_forcings = convert_to_relative_time(future_forcings, zero_time=t0)

        ############################
        
        if args.cache_inputs:
            with open('cached_inputs.nc', 'wb+') as ofh:
                pickle.dump({'prepared_ds': prepared_ds,
                            'future_forcings': future_forcings, 't0': t0}, ofh)
    else:
        with open('cached_inputs.nc', 'rb') as ifh:
            data_dict = pickle.load(ifh)
            
        prepared_ds = data_dict['prepared_ds']
        future_forcings = data_dict['future_forcings']
        t0 = data_dict['t0']
            
        
    ############################
    
    if args.var_to_replace is not None:
        # Replace one of the vars with the ERA5 version

        if not args.cache_inputs or not os.path.exists('cached_replacement_vars.nc'):
            target_datetimes = all_datetimes[2:]
            
            replacement_das = []
            for dt in target_datetimes:
                if args.var_to_replace in data.ERA5_SURFACE_VARS:
                    tmp_da = data.load_era5_surface(dt.year, dt.month, dt.day, dt.hour, era5_data_dir=DATASET_FOLDER, 
                                                    vars=[args.var_to_replace], gather_input_datetimes=False)
                elif args.var_to_replace in data.ERA5_PLEVEL_VARS:
                    tmp_da = data.load_era5_plevel(dt.year, dt.month, dt.day, 
                                                   dt.hour, era5_data_dir=DATASET_FOLDER,
                                                   pressure_levels=[1000],
                                                   vars=[args.var_to_replace], gather_input_datetimes=False)
                else:
                    raise ValueError(f'Variable {args.var_to_replace} not found in surface or pressure level lists')
                replacement_das.append(tmp_da)
                
            era5_target_da = xr.concat(replacement_das, dim='time')
            era5_target_da = convert_to_relative_time(era5_target_da, zero_time=t0)[args.var_to_replace]
            
            
            if args.interpolate_to_1000hPa and args.var_to_replace == '2m_temperature':
                # For experimenting with replacing t2m with interpolated field between SST and T1000hPa
                replacement_das = []
                for dt in target_datetimes:
                    tmp_da = data.load_era5_plevel(dt.year, dt.month, dt.day, 
                                                    dt.hour, era5_data_dir=DATASET_FOLDER,
                                                    pressure_levels=[1000],
                                                    vars=['temperature'], gather_input_datetimes=False)

                    replacement_das.append(tmp_da)
                era5_target_1000hPa_da = xr.concat(replacement_das, dim='time')
                era5_target_1000hPa_da = convert_to_relative_time(era5_target_1000hPa_da, zero_time=t0)[args.var_to_replace]
            
            if args.cache_inputs:
                with open('cached_replacement_vars.nc', 'wb+') as ofh:
                    pickle.dump({'era5_target_da': era5_target_da}, ofh)
        else:
            with open('cached_replacement_vars.nc', 'rb') as ifh:
                data_dict = pickle.load(ifh)
                
            era5_target_da = data_dict['era5_target_da']
    else:
        print('Skipping variable replacement since none specified', flush=True)
    
    ############################

    task_config_dict = dataclasses.asdict(task_config)

    current_inputs, targets, current_forcings = data_utils.extract_inputs_targets_forcings(
                prepared_ds, target_lead_times=slice("6h", "6h"),
                **task_config_dict)
    
    predictor_fn = run_forward_jitted
    rng=jax.random.PRNGKey(0)
    targets_template=targets * np.nan
    verbose=True
    num_steps_per_chunk = 1
    ############################

    sorted_input_coords_and_vars = sorted(current_inputs.coords) + sorted(current_inputs.data_vars)
    sorted_forcing_coords_and_vars = sorted(current_forcings.coords) + sorted(current_forcings.data_vars)
    current_inputs = xr.Dataset(current_inputs)[sorted_input_coords_and_vars]
    targets_template = xr.Dataset(targets_template)[sorted(current_inputs.coords) + sorted(targets_template.data_vars)]
    current_forcings = xr.Dataset(current_forcings)[sorted_forcing_coords_and_vars]

    if "datetime" in current_inputs.coords:
        del current_inputs.coords["datetime"]

    if "datetime" in targets_template.coords:
        output_datetime = targets_template.coords["datetime"]
        del targets_template.coords["datetime"]
    else:
        output_datetime = None

    if "datetime" in current_forcings.coords:
        del current_forcings.coords["datetime"]

    num_target_steps = targets_template.dims["time"]
    num_chunks, remainder = divmod(num_target_steps, num_steps_per_chunk)
    if remainder != 0:
        raise ValueError(
            f"The number of steps per chunk {num_steps_per_chunk} must "
            f"evenly divide the number of target steps {num_target_steps} ")

    if len(np.unique(np.diff(targets_template.coords["time"].data))) > 1:
        raise ValueError("The targets time coordinates must be evenly spaced")

    # Our template targets will always have a time axis corresponding for the
    # timedeltas for the first chunk.
    targets_chunk_time = targets_template.time.isel(
        time=slice(0, num_steps_per_chunk))
    input_times = current_inputs.time.values

    current_inputs.attrs = {}
    # Target template is fixed
    current_targets_template = targets_template.isel(time=slice(0,1))

    predictions = []
    for chunk_index in tqdm(range(args.num_steps)):
        
        if chunk_index > 0:
            current_forcings = future_forcings.isel(time=slice(chunk_index-1, chunk_index))
            
            data_utils.add_derived_vars(current_forcings)
            current_forcings = current_forcings[list(task_config_dict['forcing_variables'])].drop_vars("datetime")
            current_forcings = current_forcings

        actual_target_relative_time = current_forcings.coords["time"]  
        current_forcings = current_forcings.assign_coords(time=targets_chunk_time)
        current_forcings = current_forcings.compute()[sorted_forcing_coords_and_vars]
        
        if args.var_to_replace is not None and chunk_index > 0:
            ## Replace vars if appropriate
            if num_steps_per_chunk > 1:
                raise ValueError('This code assumes chunks are always of size 1')
            
            actual_input_times = current_inputs['time'].values + np.timedelta64(6*chunk_index,'h')
            
            # Have to do it this way as other ways like assigning via
            # current_inputs[var_to_replace].loc[dict(time=input_times[t_ix])].values = new_vals
            # Doesn't seem to work
            # But perhaps there is a better way that I have missed
            new_vals = []
            for t_ix, t in enumerate(actual_input_times):
                if t > 0:
                    if args.var_to_replace in gc.ALL_ATMOSPHERIC_VARS:
                        # Only replace pressure levels that are provided
                        level_vals = era5_target_da.sel(time=t).level.values
                        remaining_level_vals = sorted(set(gc.PRESSURE_LEVELS_ERA5_37).difference(set(level_vals)))
                        
                        da1 = current_inputs[args.var_to_replace].sel(time=input_times[t_ix]).sel(level=remaining_level_vals)
                        da1['time'] = input_times[t_ix]
                        da2 = era5_target_da.sel(time=t).drop_vars('datetime')
                        da2['time'] = input_times[t_ix]
                        
                        new_val = xr.concat([da1, da2], dim='level')
                    else:
                        
                        new_val = era5_target_da.sel(time=t).drop_vars('datetime')
                        new_val['time'] = input_times[t_ix]
                        
                        # Using land sea mask only currently supported with surface variables
                        if args.replace_uses_lsm:

                            land_points = np.expand_dims(static_ds['land_sea_mask'].values >= 0.5, axis=0)
                            new_val.values[land_points] = current_inputs[args.var_to_replace].sel(time=input_times[t_ix]).values[land_points]
                        
                    new_vals.append(new_val)
                    
                else:
                    new_vals.append(current_inputs[args.var_to_replace].sel(time=input_times[t_ix]))
            
            new_da = xr.concat(new_vals, dim='time')
            new_inputs = xr.merge([new_da, current_inputs[[v for v in current_inputs.data_vars if v != args.var_to_replace]]])
            
            new_inputs = new_inputs[list(current_inputs.data_vars)]
            new_inputs = new_inputs[sorted_input_coords_and_vars]
            current_inputs = new_inputs
            current_inputs.attrs = {}
    
            
        # Make sure nonnegative vars are non negative
        for nn_var in NONEGATIVE_VARS:
            
            tmp_data = current_inputs[nn_var].values.copy()
            tmp_data[tmp_data<0] = 0
            current_inputs[nn_var].values = tmp_data
        
        # Make predictions for the chunk.
        rng, this_rng = jax.random.split(rng)
        prediction = predictor_fn(
            rng=this_rng,
            inputs=current_inputs,
            targets_template=current_targets_template,
            forcings=current_forcings)

        next_frame = xr.merge([prediction, current_forcings])

        current_inputs = rollout._get_next_inputs(current_inputs, next_frame)
        prediction = prediction.assign_coords(time=actual_target_relative_time+t0)
        
        if args.var_to_replace is not None:
            
            suffix= '_lsm' if args.replace_uses_lsm else ''
            save_dir = os.path.join(args.output_dir, f'replace_{args.var_to_replace}{suffix}')
        
        else:
            save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        fp = os.path.join(save_dir, 
                          f'pred_{year}{month:02d}{day:02d}_n{chunk_index}.nc')

        prediction[OUTPUT_VARS].sel(level=[1000,850]).to_netcdf(fp)
        del prediction
        
    logger.info('Complete')
