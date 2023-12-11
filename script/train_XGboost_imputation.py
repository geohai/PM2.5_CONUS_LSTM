import os
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import joblib

# Import ML Packages
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

import xgboost

# For Visualization
from scipy.stats import gaussian_kde


def train_RF(train_df, buffer_avg=True, target="047", validation=False):
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

    train_df = train_df[feature_list]
    # Filter out records have grount truth
    train_df = train_df[train_df['aod_047'].notnull()]
    if buffer_avg:
        train_df = train_df[train_df['aod_buffer_047'].notnull()]
        print(f"Total Samples | Buffer: {buffer_avg}: {train_df.shape}")

    # Fill NaNs or RF cannot work
    train_df = train_df.fillna(0)

    # Setting Hyperparameters. Please refer to the SI for more information
    xgb_params = dict(learning_rate=np.arange(0.05, 0.3, 0.05),
                      n_estimators=np.arange(100, 500, 100),
                      gamma=np.arange(1, 10, 1),
                      subsample=np.arange(0.1, 0.5, 0.1),
                      max_depth=[int(i) for i in np.arange(1, 10, 1)])

    # inititalization
    xgb_regressor = xgboost.XGBRegressor(booster='gbtree', verbosity=0, tree_method='gpu_hist')

    # find optimal parameters for random forest regressor using  RandomizedSearchCV.
    # Set random_state=42 and be careful about scoring type
    xgb_regressor_cv = RandomizedSearchCV(xgb_regressor, xgb_params, cv=5,
                                          scoring='neg_root_mean_squared_error',
                                          n_jobs=14)

    if validation:
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

        X_train = train_df.drop(columns=['aod_047', 'aod_055'])
        y_train = train_df[[f'aod_{target}']]

        X_val = val_df.drop(columns=['aod_047', 'aod_055'])
        y_val = val_df[[f'aod_{target}']]
    else:
        val_df = None
        X_train = train_df.drop(columns=['aod_047', 'aod_055'])
        y_train = train_df[[f'aod_{target}']]

    xgb_regressor_cv.fit(X_train, y_train.values.ravel())
    best_params = xgb_regressor_cv.best_params_

    # create best_rf_regressor sunig the parameters above and fit it to training data
    best_xgb_regressor = xgb_regressor_cv.best_estimator_
    best_xgb_regressor.fit(X_train, y_train.values.ravel())
    # model evaluation for training set
    train_r2_xgb = round(best_xgb_regressor.score(X_train, y_train), 2)
    print('Training R2 score of XGBoost is {}'.format(train_r2_xgb))
    y_train_predicted_xgb = best_xgb_regressor.predict(X_train)
    rmse_train_xgb = (np.sqrt(mean_squared_error(y_train, y_train_predicted_xgb)))
    print('RMSE on the training set for the XGBoost model is: {}'.format(rmse_train_xgb))
    mbe_train_xgb = np.mean(y_train_predicted_xgb - y_train.values.squeeze())
    print("MBE on training set is for the XGBoost model is: {}".format(mbe_train_xgb))

    if validation:
        # model evaluation for test set
        y_test_predicted_xgb = best_xgb_regressor.predict(X_val)
        rmse_test_xgb = (np.sqrt(mean_squared_error(y_val, y_test_predicted_xgb)))
        print("RMSE on testing set is for the XGBoost model is: {}".format(rmse_test_xgb))

        mbe_test_xgb = np.mean(y_test_predicted_xgb - y_val.values.squeeze())
        print("MBE on testing set is for the XGBoost model is: {}".format(mbe_test_xgb))

    print(f"Exporting RF_{target}_{'with' if buffer_avg else 'w/o'} model...")
    joblib.dump(best_xgb_regressor, f"../model/XGB_Imputation/XGB_AOD{target}_"
                                    f"{'with' if buffer_avg else 'without'}_buffer.joblib")

    print("================================================================================")


if __name__ == "__main__":
    file_list = glob("../data/input_1D/*.csv")

    pm_df = [pd.read_csv(file, low_memory=False) for file in file_list]
    pm_df = pd.concat(pm_df, ignore_index=True)

    print("Dota loading finished!")

    # Start Training Model with Buffer Average
    print("Start Training Model with Buffer Average...")
    train_RF(train_df=pm_df, buffer_avg=True, target="047", validation=True)
    train_RF(train_df=pm_df, buffer_avg=True, target="055", validation=True)

    # Start Training Model with Buffer Average
    print("Start Training Model w/o Buffer Average...")
    train_RF(train_df=pm_df, buffer_avg=False, target="047", validation=True)
    train_RF(train_df=pm_df, buffer_avg=False, target="055", validation=True)
