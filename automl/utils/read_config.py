import os
import sys
import copy
import tensorflow as tf
import numpy as np
import types
import math
from pathlib import Path

HOME = Path(__file__).parents[2]
CONFIG_FOLDER = HOME / 'config'

from automl.utils.utils import load_yaml_file

def read_config(config_filename: str=None, config_folder: str=CONFIG_FOLDER) -> dict:

    config_path = os.path.join(config_folder, config_filename)
    try:
        config_dict = load_yaml_file(config_path)
    except FileNotFoundError as e:
        print(e)
        print(f"You must set {config_filename} in the config folder.")
        sys.exit(1)       
    
    return config_dict

def read_model_config(config_filename: str='model_config.yaml', config_folder: str=CONFIG_FOLDER,
                      model_config_dict: dict=None) -> dict:
    
    if model_config_dict is None:
        model_config_dict = read_config(config_filename=config_filename, config_folder=config_folder)
      
    model_config = copy.deepcopy(model_config_dict)
    for k, v in model_config.items():
        if isinstance(v, dict):
            model_config[k] = types.SimpleNamespace(**v)
    model_config = types.SimpleNamespace(**model_config)

    return model_config

def read_data_config(config_filename: str='data_config.yaml', config_folder: str=CONFIG_FOLDER,
                     data_config_dict: dict=None) -> dict:
    if data_config_dict is None:
        
        data_config_dict = read_config(config_filename=config_filename, config_folder=config_folder)
    
    data_config_ns = types.SimpleNamespace(**data_config_dict)

    return data_config_ns