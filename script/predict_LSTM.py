from datetime import datetime, timedelta
import time
from glob import glob
import os
import joblib

from joblib import parallel_backend

import rioxarray  # for the extension to load
from rioxarray.merge import merge_arrays
import xarray

import rasterio
from rasterio.plot import show
from pyproj import CRS

import geopandas as gpd
import numpy as np

import tensorflow as tf
import keras.layers
from attention import Attention

# from scipy.signal import convolve2d
# from scipy import ndimage
# from multiprocessing import Pool
# from functools import partial
#
# import matplotlib.pyplot as plt
#
# import earthpy as et
# import earthpy.spatial as es
import earthpy.plot as ep

from numba import cuda
import gc
from numba import cuda

import warnings

warnings.filterwarnings("ignore")


def load_region_mask(mask_path):
    """
    :param mask_path:
    :return:
    """
    region_mask = xarray.open_dataarray(mask_path)

    region_mask = region_mask.values

    # region mapping
    # Western = 1
    # MidWestern = 2
    # Eastern = 3
    # Southern = 4
    us_mask = (region_mask != 0)
    western_mask = (region_mask == 1)
    midwestern_mask = (region_mask == 2)
    eastern_mask = (region_mask == 3)
    southern_mask = region_mask == 4

    return region_mask, us_mask, western_mask, midwestern_mask, eastern_mask, southern_mask


def load_input(date_list, n_lag, region_mask):
    """
    :param date_list:
    :param n_lag:
    :return:
    """
    assert len(date_list) == n_lag

    for i in range(len(date_list)):
        date = date_list[i]
        print(f'Loading data of date {i + 1}: {date}')
        # merged_xr_path = f'../data/merged_inputs/{date}_with_imputation.nc'
        merged_xr_path = f'/media/zhongying/WD_BLACK/PM2.5_CONUS_LSTM/data/merged_inputs/{date}_with_imputation.nc'
        merged_xr = xarray.open_dataset(merged_xr_path)

        # should be 21 vars
        merged_3darray = np.array([
            merged_xr['aod_047'],  # 0
            merged_xr['aod_055'],  # 1
            merged_xr['day_cos'],  # 2
            merged_xr['day_sin'],  # 3
            merged_xr['daymet_dayl'],  # 4
            merged_xr['daymet_lat'],  # 5
            merged_xr['daymet_lon'],  # 6
            merged_xr['daymet_prcp'],  # 7
            merged_xr['daymet_srad'],  # 8
            merged_xr['daymet_tmax'],  # 9
            merged_xr['daymet_tmin'],  # 10
            merged_xr['daymet_vp'],  # 11
            merged_xr['dem'][0],  # 12
            merged_xr['gridmet_th'],  # 13
            merged_xr['gridmet_vs'],  # 14
            # merged_xr['knnidw_distance'],
            merged_xr['knnidw_pm25'],  # 15
            # merged_xr['knnidw_pm25_val'],
            merged_xr['month_cos'],  # 17
            merged_xr['month_sin'],  # 18
            merged_xr['ndvi'],  # 19
            merged_xr['wildfire_smoke'],  # 20
            merged_xr['year']  # 21
        ])

        if i == 0:
            input_arr = merged_3darray[:, region_mask].T
        else:
            input_arr = np.column_stack((input_arr, merged_3darray[:, region_mask].T))

    input_arr = input_arr.reshape(-1, n_lag, n_features)

    print(f'Input Array shape: {input_arr.shape}')

    return input_arr


def scale_input(input_arr, X_scaler):
    num_data, time_lag, num_features = input_arr.shape
    input_arr = input_arr.reshape(num_data * time_lag, num_features)

    with parallel_backend('threading', n_jobs=12):
        scaled_arr = X_scaler.transform(input_arr)
        # Mask NaN for (-1)
        scaled_arr[np.isnan(scaled_arr)] = -1

    scaled_arr = scaled_arr.reshape(num_data, time_lag, num_features)

    return scaled_arr


def model_predict(input_arr, num_model,
                  n_lag, n_features,
                  X_scaler, y_scaler):
    """
    :param input_arr:
    :param region:
    :param num_model:
    :param n_lag:
    :param n_features:
    :return:
    """
    # Scale input array
    scaled_arr = scale_input(input_arr, X_scaler)

    tensor = tf.convert_to_tensor(scaled_arr,
                                  dtype=tf.float32)

    for i, num_M in enumerate(num_model):
        model_path = f'../model/LSTM_model/knn/LSTM_{num_M}.h5'
        # Load model
        model = keras.models.load_model(model_path,
                                        custom_objects={'Attention': Attention},
                                        compile=True)

        # Make Prediction
        predict = model.predict(tensor, batch_size=128)

        # Clear Session to Free memory
        del model
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        _ = gc.collect()

        with parallel_backend('threading', n_jobs=10):
            inv_predict = y_scaler.inverse_transform(predict)

        # print(f'Min value of prediction: {np.nanmin(inv_predict)}')
        # print(f'Max value of prediction: {np.nanmax(inv_predict)}')
        print(f'Num of values < 0: {inv_predict[inv_predict < 0].shape}')
        # print(f'Num of values > 50: {inv_predict[inv_predict > 50].shape}')
        inv_predict[inv_predict < 0] = 0.5

        if i == 0:
            pred_models = inv_predict
        else:
            pred_models = np.hstack((pred_models, inv_predict))

        # Release GPU Memory
        # device = cuda.get_current_device()
        # device.reset()

    return pred_models


def shift_input_arr(input_arr, estimate_date, region_mask):
    print(f'Loading new data of date: {estimate_date}')
    # merged_xr_path = f'../data/merged_inputs/{estimate_date}_with_imputation.nc'
    merged_xr_path = f'/media/zhongying/WD_BLACK/PM2.5_CONUS_LSTM/data/merged_inputs/{estimate_date}_with_imputation.nc'
    merged_xr = xarray.open_dataset(merged_xr_path)

    # should be 21 vars
    merged_3darray = np.array([
        merged_xr['aod_047'],  # 0
        merged_xr['aod_055'],  # 1
        merged_xr['day_cos'],  # 2
        merged_xr['day_sin'],  # 3
        merged_xr['daymet_dayl'],  # 4
        merged_xr['daymet_lat'],  # 5
        merged_xr['daymet_lon'],  # 6
        merged_xr['daymet_prcp'],  # 7
        merged_xr['daymet_srad'],  # 8
        merged_xr['daymet_tmax'],  # 9
        merged_xr['daymet_tmin'],  # 10
        merged_xr['daymet_vp'],  # 11
        merged_xr['dem'][0],  # 12
        merged_xr['gridmet_th'],  # 13
        merged_xr['gridmet_vs'],  # 14
        # merged_xr['knnidw_distance'],
        merged_xr['knnidw_pm25'],  # 15
        # merged_xr['knnidw_pm25_val'],
        merged_xr['month_cos'],  # 17
        merged_xr['month_sin'],  # 18
        merged_xr['ndvi'],  # 19
        merged_xr['wildfire_smoke'],  # 20
        merged_xr['year']  # 21
    ])

    # Load new array
    new_input_arr = merged_3darray[:, region_mask].T
    new_input_arr = np.expand_dims(new_input_arr, axis=1)
    # Drop the oldest day's data
    input_arr = input_arr[:, 1:, :]
    print(f'Input Shape after drop: {input_arr.shape}')
    # Append new day's dta
    input_arr = np.column_stack((input_arr, new_input_arr))
    print(f'Input Shape after stack: {input_arr.shape}')

    return input_arr


if __name__ == "__main__":
    # Fer Regionalization Mask
    mask_path = '../data/state_boundary/new_regionalization.nc'
    region_xr = xarray.open_dataarray(mask_path, decode_coords="all")
    region_xr = region_xr.astype('float32')

    region_mask, us_mask, western_mask, midwestern_mask, eastern_mask, southern_mask = load_region_mask(mask_path)
    # Convert region_mask dtype
    region_mask = region_mask.astype('float32')

    mask_dict = {
        'Western': western_mask,
        'MidWestern': midwestern_mask,
        'Eastern': eastern_mask,
        'Southern': southern_mask
    }

    # Define Params.
    n_lag = 21
    n_features = 21
    num_model = [1, 5, 9]
    # Expand Dim for model outputs
    region_mask = np.repeat(region_mask[:, :, np.newaxis], len(num_model), axis=2)
    # Initialize input_arrZho
    input_arr = {
        'Western': None,
        'MidWestern': None,
        'Eastern': None,
        'Southern': None,
    }

    # Calculate Input dates
    # estimate_date = datetime(2005, 8, 26).date()
    estimate_date = datetime(2010, 12, 31).date()

    # Define Date
    end_date = datetime(2012, 1, 1).date()
    delta = timedelta(days=1)

    # Load Scaler
    X_scaler = joblib.load("../model/LSTM_Scaler/X_scaler.pkl")
    y_scaler = joblib.load("../model/LSTM_Scaler/y_scaler.pkl")

    # Set GPU Memory
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)

    while estimate_date < end_date:
        # Release GPU Memory
        # print('Release GPU Memory')
        # device = cuda.get_current_device()
        # device.reset()

        start_time = time.time()

        if os.path.isfile(f'../output/ensemble/{estimate_date}.nc'):
            print("File already exists!")
        else:
            # Make estimation
            if input_arr['Western'] is None:
                date_list = [estimate_date - timedelta(days=x) for x in range(n_lag)]
                date_list = sorted(date_list)

                print(f'Start date: {date_list[0]} \n'
                      f'End date: {date_list[-1]}')

                print(f'Loading input array for all regions.')
                input_arr['Western'] = load_input(date_list=date_list,
                                                  n_lag=n_lag,
                                                  region_mask=western_mask)

                input_arr['MidWestern'] = load_input(date_list=date_list,
                                                     n_lag=n_lag,
                                                     region_mask=midwestern_mask)
                input_arr['Eastern'] = load_input(date_list=date_list,
                                                  n_lag=n_lag,
                                                  region_mask=eastern_mask)
                input_arr['Southern'] = load_input(date_list=date_list,
                                                   n_lag=n_lag,
                                                   region_mask=southern_mask)

                print('Start Predicting for all regions.')

                region_mask[western_mask] = model_predict(input_arr=input_arr['Western'],
                                                          num_model=num_model,
                                                          n_lag=n_lag,
                                                          n_features=n_features,
                                                          X_scaler=X_scaler, y_scaler=y_scaler)
                region_mask[midwestern_mask] = model_predict(input_arr=input_arr['MidWestern'],
                                                             num_model=num_model,
                                                             n_lag=n_lag,
                                                             n_features=n_features,
                                                             X_scaler=X_scaler, y_scaler=y_scaler)
                region_mask[eastern_mask] = model_predict(input_arr=input_arr['Eastern'],
                                                          num_model=num_model,
                                                          n_lag=n_lag,
                                                          n_features=n_features,
                                                          X_scaler=X_scaler, y_scaler=y_scaler)
                region_mask[southern_mask] = model_predict(input_arr=input_arr['Southern'],
                                                           num_model=num_model,
                                                           n_lag=n_lag,
                                                           n_features=n_features,
                                                           X_scaler=X_scaler, y_scaler=y_scaler)

                print('Prediction finished.')
                # Ensemble by Avg
                ensemble_mask = np.nanmean(region_mask, axis=2)

                print('Starting Exporting.')

                output_path = f'../output/ensemble/{estimate_date}.nc'
                raw_path = f'../output/raw/{estimate_date}.nc'
                region_xr.values = ensemble_mask
                region_xr = region_xr.to_dataset(name='PM2.5')
                region_xr.attrs['description'] = f"Ensemble of {len(num_model)} Bi-LSTM Attn Models"

                raw_xr = region_xr.copy()
                raw_xr = raw_xr.rename({"PM2.5": f"Model_{num_model[0]}"})
                for idx, num_M in enumerate(num_model):
                    raw_xr[f"Model_{num_M}"] = raw_xr[f"Model_{num_model[0]}"]
                    raw_xr[f"Model_{num_M}"].values = region_mask[:, :, idx]

                raw_xr.attrs['description'] = f"Output of {len(num_model)} Bi-LSTM Attn Models"

                # Add projection
                proj = CRS.from_wkt('PROJCS["MODIS Sinusoidal",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",'
                                    '6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],'
                                    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG",'
                                    '"9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Sinusoidal"],PARAMETER['
                                    '"longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER['
                                    '"false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",'
                                    'EAST],AXIS["Northing",NORTH]]')
                region_xr = region_xr.rio.write_crs(proj)
                raw_xr = raw_xr.rio.write_crs(proj)

                region_xr.to_netcdf(path=output_path, mode='w', )
                raw_xr.to_netcdf(path=raw_path, mode='w', )


            else:
                print('Input is not empty. Start Shifting Inputs.')
                input_arr['Western'] = shift_input_arr(input_arr=input_arr['Western'],
                                                       estimate_date=estimate_date,
                                                       region_mask=western_mask)
                input_arr['MidWestern'] = shift_input_arr(input_arr=input_arr['MidWestern'],
                                                          estimate_date=estimate_date,
                                                          region_mask=midwestern_mask)
                input_arr['Eastern'] = shift_input_arr(input_arr=input_arr['Eastern'],
                                                       estimate_date=estimate_date,
                                                       region_mask=eastern_mask)
                input_arr['Southern'] = shift_input_arr(input_arr=input_arr['Southern'],
                                                        estimate_date=estimate_date,
                                                        region_mask=southern_mask)

                print('Start predicting for all regions')
                region_mask[western_mask] = model_predict(input_arr=input_arr['Western'],
                                                          num_model=num_model,
                                                          n_lag=n_lag,
                                                          n_features=n_features,
                                                          X_scaler=X_scaler, y_scaler=y_scaler)
                region_mask[midwestern_mask] = model_predict(input_arr=input_arr['MidWestern'],
                                                             num_model=num_model,
                                                             n_lag=n_lag,
                                                             n_features=n_features,
                                                             X_scaler=X_scaler, y_scaler=y_scaler)
                region_mask[eastern_mask] = model_predict(input_arr=input_arr['Eastern'],
                                                          num_model=num_model,
                                                          n_lag=n_lag,
                                                          n_features=n_features,
                                                          X_scaler=X_scaler, y_scaler=y_scaler)
                region_mask[southern_mask] = model_predict(input_arr=input_arr['Southern'],
                                                           num_model=num_model,
                                                           n_lag=n_lag,
                                                           n_features=n_features,
                                                           X_scaler=X_scaler, y_scaler=y_scaler)

                print('Prediction finished.')
                # Ensemble by Avg
                ensemble_mask = np.nanmean(region_mask, axis=2)

                print('Starting Exporting.')

                output_path = f'../output/ensemble/{estimate_date}.nc'
                raw_path = f'../output/raw/{estimate_date}.nc'
                region_xr['PM2.5'].values = ensemble_mask

                for idx, num_M in enumerate(num_model):
                    raw_xr[f"Model_{num_M}"].values = region_mask[:, :, idx]

                # Add projection
                proj = CRS.from_wkt('PROJCS["MODIS Sinusoidal",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",'
                                    '6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],'
                                    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG",'
                                    '"9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Sinusoidal"],PARAMETER['
                                    '"longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER['
                                    '"false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",'
                                    'EAST],AXIS["Northing",NORTH]]')
                region_xr = region_xr.rio.write_crs(proj)
                raw_xr = raw_xr.rio.write_crs(proj)

                region_xr.to_netcdf(path=output_path, mode='w', )
                raw_xr.to_netcdf(path=raw_path, mode='w', )

        estimate_date += delta

        print("--- %s seconds ---" % (time.time() - start_time))
