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

    rf_regressor = RandomForestRegressor(random_state=42)
    # rf_regressor = RandomForestRegressor()

    rf_params = {'n_estimators': np.arange(30, 200, 10),
                 'max_depth': np.arange(1, 15, 1),
                 'min_samples_split': np.arange(2, 50, 1),
                 'min_samples_leaf': np.arange(2, 50, 1),
                 'max_features': ['sqrt', 'log2']}  # could also add 'criterion':['mse', 'mae'],

    # find optimal parameters for random forest regressor using  RandomizedSearchCV.
    # Set random_state=42 and be careful about scoring type
    rf_regressor_cv = RandomizedSearchCV(rf_regressor, rf_params, cv=5,
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

    rf_regressor_cv.fit(X_train, y_train.values.ravel())
    best_params = rf_regressor_cv.best_params_

    # create best_rf_regressor sunig the parameters above and fit it to training data
    best_rf_regressor = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                              max_depth=best_params['max_depth'],
                                              max_features=best_params['max_features'],
                                              min_samples_leaf=best_params['min_samples_leaf'],
                                              min_samples_split=best_params['min_samples_split'],
                                              n_jobs=14)
    best_rf_regressor.fit(X_train, y_train.values.ravel())
    # model evaluation for training set
    train_r2_rf = round(best_rf_regressor.score(X_train, y_train), 2)
    print('Training R2 score of Random Forest is {}'.format(train_r2_rf))
    y_train_predicted_rf = best_rf_regressor.predict(X_train)
    rmse_train_rf = (np.sqrt(mean_squared_error(y_train, y_train_predicted_rf)))
    print('RMSE on the training set for the Random Forest model is: {}'.format(rmse_train_rf))
    mbe_train_rf = np.mean(y_train_predicted_rf - y_train.values.squeeze())
    print("MBE on training set is for the Random Forest model is: {}".format(mbe_train_rf))

    if validation:
        # model evaluation for test set
        y_test_predicted_rf = best_rf_regressor.predict(X_val)
        rmse_test_rf = (np.sqrt(mean_squared_error(y_val, y_test_predicted_rf)))
        print("RMSE on testing set is for the Random Forest model is: {}".format(rmse_test_rf))

        mbe_test_rf = np.mean(y_test_predicted_rf - y_val.values.squeeze())
        print("MBE on testing set is for the Random Forest model is: {}".format(mbe_test_rf))

    print(f"Exporting RF_{target}_{'with' if buffer_avg else 'w/o'} model...")
    joblib.dump(best_rf_regressor, f"../model/RF_imputation/RF_AOD{target}_"
                                   f"{'with' if buffer_avg else 'without'}_buffer.joblib")

    print("================================================================================")


if __name__ == "__main__":
    file_list = glob("../data/input_1D/*.csv")

    pm_df = [pd.read_csv(file, low_memory=False) for file in file_list]
    pm_df = pd.concat(pm_df, ignore_index=True)

    print("Dota loading finished!")

    # Start Training Model with Buffer Average
    print("Start Training Model with Buffer Average...")
    train_RF(train_df=pm_df, buffer_avg=True, target="047", validation=False)
    train_RF(train_df=pm_df, buffer_avg=True, target="055", validation=False)

    # Start Training Model with Buffer Average
    print("Start Training Model w/o Buffer Average...")
    train_RF(train_df=pm_df, buffer_avg=False, target="047", validation=False)
    train_RF(train_df=pm_df, buffer_avg=False, target="055", validation=False)
