import os
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime

# Import ML Packages
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# For Visualization
from scipy.stats import gaussian_kde


def train_RF(train_df, buffer_avg=True):
    if buffer_avg:
        feature_list = ['aod_047', 'aod_055', 'aod_buffer_047', 'aod_buffer_055',  # 'avg_pm25',
                        'day_cos', 'day_sin', 'daymet_dayl', 'daymet_lat', 'daymet_lon',
                        'daymet_prcp', 'daymet_srad', 'daymet_tmax', 'daymet_tmin', 'daymet_vp',
                        'dem', 'gridmet_th', 'gridmet_vs',  # 'knnidw_distance', 'knnidw_pm25', 'knnidw_pm25_val',
                        'month_cos', 'month_sin', 'ndvi', 'wildfire_smoke',
                        'year']

    else:
        feature_list = ['aod_047', 'aod_055',  # 'aod_buffer_047', 'aod_buffer_055',  # 'avg_pm25',
                        'day_cos', 'day_sin', 'daymet_dayl', 'daymet_lat', 'daymet_lon',
                        'daymet_prcp', 'daymet_srad', 'daymet_tmax', 'daymet_tmin', 'daymet_vp',
                        'dem', 'gridmet_th', 'gridmet_vs',  # 'knnidw_distance', 'knnidw_pm25', 'knnidw_pm25_val',
                        'month_cos', 'month_sin', 'ndvi', 'wildfire_smoke',
                        'year']

    return None


if __name__ == "__main__":
    file_list = glob("../data/input_1D/*.csv")

    pm_df = [pd.read_csv(file, low_memory=False) for file in file_list]
    pm_df = pd.concat(pm_df, ignore_index=True)

    print("Dota loading finished!")

    print("Start Training Model with Buffer Average...")
