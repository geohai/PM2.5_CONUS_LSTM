import os
import time
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import joblib
from functools import partial

# Import ML Packages
import sklearn
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Integer

import xgboost

# For Visualization
from scipy.stats import gaussian_kde


# Reporting util for different optimizers
def report_perf(optimizer, X, y, title="model", callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers

    optimizer = a sklearn or a skopt optimizer
    X = the training set
    y = our target
    title = a string label for the experiment
    """
    start = time.time()

    if callbacks is not None:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)

    d = pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_

    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           + u"\u00B1" + " %.3f") % (time.time() - start,
                                     len(optimizer.cv_results_['params']),
                                     best_score,
                                     best_score_std))
    print('Best parameters:')
    print(best_params)
    print()
    return best_params


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
    # if buffer_avg:
    #     xgb_params = dict(learning_rate=np.arange(0.05, 0.3, 0.05),
    #                       eta = np.arange(0.1, 0.5, 0.1),
    #                       n_estimators=np.arange(100, 800, 100),
    #                       gamma=np.arange(1, 10, 1),
    #                       subsample=np.arange(0.1, 0.5, 0.1),
    #                       max_depth=[int(i) for i in np.arange(3, 12, 1)],
    #                       colsample_bytree=np.arange(0.1, 1.0, 0.2))
    # else:
    #     xgb_params = dict(learning_rate=np.arange(0.05, 0.3, 0.05),
    #                       eta=np.arange(0.1, 0.5, 0.1),
    #                       n_estimators=np.arange(10, 500, 30),
    #                       gamma=np.arange(1, 10, 1),
    #                       subsample=np.arange(0.1, 0.5, 0.1),
    #                       max_depth=[int(i) for i in np.arange(1, 10, 1)],
    #                       colsample_bytree=np.arange(0.1, 1.0, 0.2))

    xgb_params = dict(learning_rate=(0.01, 1.0, 'uniform'),
                      n_estimators=(50, 2000),
                      gamma=(1, 10),
                      subsample=(0.1, 1.0, 'uniform'),
                      max_depth=(2, 12),
                      colsample_bytree=(0.1, 1.0, 'uniform'),
                      reg_lambda=(1e-9, 100., 'uniform'))

    # inititalization
    xgb_regressor = xgboost.XGBRegressor(booster='gbtree',
                                         objective='reg:squarederror',
                                         verbosity=0,
                                         tree_method='gpu_hist')

    # find optimal parameters for random forest regressor using  RandomizedSearchCV.
    # Set random_state=42 and be careful about scoring type
    # xgb_regressor_cv = RandomizedSearchCV(xgb_regressor, xgb_params, cv=5,
    #                                       scoring='neg_root_mean_squared_error',
    #                                       n_jobs=14)

    # Setting the scoring function
    scoring = make_scorer(partial(mean_squared_error, squared=False),
                          greater_is_better=False)

    xgb_regressor_cv = BayesSearchCV(estimator=xgb_regressor,
                                     search_spaces=xgb_params,
                                     scoring=scoring,
                                     cv=5,
                                     n_iter=120,
                                     n_points=1,
                                     n_jobs=16,
                                     return_train_score=True,
                                     refit=False,
                                     optimizer_kwargs={'base_estimator': 'GP'})

    # Running the optimizer
    overdone_control = DeltaYStopper(delta=0.0001)  # We stop if the gain of the optimization becomes too small
    time_limit_control = DeadlineStopper(total_time=60 * 60 * 4)  # We impose a time limit (7 hours)

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

    # xgb_regressor_cv.fit(X_train, y_train.values.ravel())
    np.int = int
    best_params = report_perf(xgb_regressor_cv, X_train, y_train, 'XGBoost_regression',
                              callbacks=[overdone_control, time_limit_control])

    # best_params = xgb_regressor_cv.best_params_
    # print(best_params)

    # create best_rf_regressor sunig the parameters above and fit it to training data
    # best_xgb_regressor = xgb_regressor_cv.best_estimator_
    best_xgb_regressor = xgboost.XGBRegressor(booster='gbtree',
                                              objective='reg:squarederror',
                                              verbosity=0,
                                              tree_method='gpu_hist',
                                              **best_params)
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
    train_RF(train_df=pm_df, buffer_avg=True, target="047", validation=False)
    train_RF(train_df=pm_df, buffer_avg=True, target="055", validation=False)

    # Start Training Model with Buffer Average
    print("Start Training Model w/o Buffer Average...")
    train_RF(train_df=pm_df, buffer_avg=False, target="047", validation=False)
    train_RF(train_df=pm_df, buffer_avg=False, target="055", validation=False)
