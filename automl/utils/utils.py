import os
from pathlib import Path
import hashlib
import json
import yaml
import re
import copy
from glob import glob
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from calendar import monthrange
from typing import Iterable, Tuple, Callable, List
from timezonefinder import TimezoneFinder
from dateutil import tz
from types import SimpleNamespace

tz_finder = TimezoneFinder()
from_zone = tz.gettz('UTC')


def hash_dict(params: dict):
    h = hashlib.shake_256()
    h.update(json.dumps(params, sort_keys=True).encode('utf-8'))
    return h.hexdigest(8)


def load_yaml_file(fpath: str):  
    with open(fpath, 'r') as f:
        data = yaml.safe_load(f)
        
    return data

def write_to_yaml(fpath: str, data: dict):
    
    with open(fpath, 'w+') as ofh:
        yaml.dump(data, ofh, default_flow_style=False)
        
def date_range_from_year_month_range(year_month_ranges):
    
    if not isinstance(year_month_ranges[0], list):
        year_month_ranges = [year_month_ranges]
    
    output_dates = []
    for ym_range in year_month_ranges:
        start_date = datetime.datetime(year=int(ym_range[0][:4]), month=int(ym_range[0][4:6]), day=1)
        
        end_year = int(ym_range[-1][:4])
        end_month = int(ym_range[-1][4:6])
        end_date = datetime.datetime(year=end_year, 
                                    month=end_month, 
                                    day=monthrange(end_year, end_month)[1])
        output_dates += [item.date() for item in pd.date_range(start=start_date, end=end_date)]
    return sorted(set(output_dates))


def get_local_datetime(utc_datetime: datetime.datetime, longitude: float, latitude: float) -> datetime.datetime:
    """Get datetime at locality defined by lat,long, from UTC datetime

    Args:
        utc_datetime (datetime.datetime): UTC datetime
        longitude (float): longitude
        latitude (float): latitude

    Returns:
        datetime.datetime: Datetime in local time
    """
    utc_datetime.replace(tzinfo=from_zone)
    
    timezone = tz_finder.timezone_at(lng=longitude, lat=latitude)
    to_zone = tz.gettz(timezone)

    local_datetime = utc_datetime.astimezone(to_zone)
    
    return local_datetime

def get_local_hour(hour: int, longitude: float, latitude: float):
    """Convert hour to hour in locality.
    
    This is not as precise as get_local_datetime, as it won't contain information about e.g. BST. 
    But it is useful when the date is not known but the hour is

    Args:
        hour (int): UTC hour
        longitude (float): longitude
        latitude (float): latitude
    Returns:
        int: approximate hour in local time
    """
    utc_datetime = datetime.datetime(year=2000, month=1, day=1, hour=hour)
    utc_datetime.replace(tzinfo=from_zone)
    
    timezone = tz_finder.timezone_at(lng=longitude, lat=latitude)
    to_zone = tz.gettz(timezone)

    local_hour = utc_datetime.astimezone(to_zone).hour
    
    return local_hour

def convert_namespace_to_dict(ns_obj: SimpleNamespace) -> dict:
    """Convert nested namespace object to dict

    Args:
        ns_obj (SimpleNamespace): nested namespace

    Returns:
        dict: namespace converted into dict
    """
    output_dict = copy.deepcopy(ns_obj).__dict__
    for k, v in output_dict.items():
        if isinstance(v, SimpleNamespace):
            output_dict[k] = v.__dict__
            
    return output_dict
