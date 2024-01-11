from datetime import datetime, timedelta
import time
from glob import glob
import os
import joblib

import xarray
import xgboost

import numpy as np

import warnings

warnings.filterwarnings("ignore")


def aod_imputation_RF(date,
                      rf_aod_047_wbuffer,
                      rf_aod_055_wbuffer,
                      rf_aod_047_wobuffer,
                      rf_aod_055_wobuffer):
    """

    :param date: The date of file need to Impute
    :param rf_aod_047_wbuffer: RF model for AOD_047 with buffer avg.
    :param rf_aod_055_wbuffer: RF model for AOD_055 with buffer avg.
    :param rf_aod_047_wobuffer: RF model for AOD_047 w/o buffer avg.
    :param rf_aod_055_wobuffer: RF model for AOD_055 w/o buffer avg.
    :return: None
    :output: Export imputed NetCDF File
    """
    # Load the saved stacked data
    unimputed_xr = xarray.open_dataset(f'../data/merged_inputs/{str(date)}_wo_imputation.nc')

    unimputed_xr['ndvi'] = unimputed_xr['ndvi'].fillna(0)

    unimputed_3darray = np.array([
        unimputed_xr['aod_buffer_047'],  # 0
        unimputed_xr['aod_buffer_055'],  # 1
        unimputed_xr['day_cos'],  # 2
        unimputed_xr['day_sin'],  # 3
        unimputed_xr['daymet_dayl'],  # 4
        unimputed_xr['daymet_lat'],  # 5
        unimputed_xr['daymet_lon'],  # 6
        unimputed_xr['daymet_prcp'],  # 7
        unimputed_xr['daymet_srad'],  # 8
        unimputed_xr['daymet_tmax'],  # 9
        unimputed_xr['daymet_tmin'],  # 10
        unimputed_xr['daymet_vp'],  # 11
        unimputed_xr['dem'][0],  # 12
        unimputed_xr['gridmet_th'],  # 13
        unimputed_xr['gridmet_vs'],  # 14
        unimputed_xr['month_cos'],  # 15
        unimputed_xr['month_sin'],  # 16
        unimputed_xr['ndvi'],  # 17
        unimputed_xr['wildfire_smoke'],  # 18
        unimputed_xr['year'],  # 19
    ])

    aod_3darray = np.array([
        unimputed_xr['aod_047'],  # 0
        unimputed_xr['aod_055'],  # 1
    ])

    # Get idx of coordinates with buffer
    model_047_wbuffer_idx = (~np.isnan(unimputed_3darray[0])) & (~np.isnan(unimputed_3darray[1])) \
                            & (~np.isnan(unimputed_3darray[4])) \
                            & (~np.isnan(unimputed_3darray[13])) & (np.isnan(aod_3darray[0]))

    model_055_wbuffer_idx = (~np.isnan(unimputed_3darray[0])) & (~np.isnan(unimputed_3darray[1])) \
                            & (~np.isnan(unimputed_3darray[4])) \
                            & (~np.isnan(unimputed_3darray[13])) & (np.isnan(aod_3darray[1]))

    # Get idx of coordinates without buffer
    model_047_wobuffer_idx = (np.isnan(unimputed_3darray[0])) & (np.isnan(unimputed_3darray[1])) \
                             & (~np.isnan(unimputed_3darray[4])) \
                             & (~np.isnan(unimputed_3darray[13])) & (np.isnan(aod_3darray[0]))

    model_055_wobuffer_idx = (np.isnan(unimputed_3darray[0])) & (np.isnan(unimputed_3darray[1])) \
                             & (~np.isnan(unimputed_3darray[4])) \
                             & (~np.isnan(unimputed_3darray[13])) & (np.isnan(aod_3darray[1]))

    # Imputation with Buffer
    print('RF models to impute NaNs')
    # rf_aod_047_wbuffer = joblib.load('../model/saved_model/aod_imputation_model/aod_047_buffer.joblib')
    aod_047_wbuffer_pred = rf_aod_047_wbuffer.predict(unimputed_3darray[:, model_047_wbuffer_idx].T)

    # rf_aod_055_wbuffer = joblib.load('../model/saved_model/aod_imputation_model/aod_055_buffer.joblib')
    aod_055_wbuffer_pred = rf_aod_055_wbuffer.predict(unimputed_3darray[:, model_055_wbuffer_idx].T)

    # Imputation without Buffer
    nobuffer_col = np.delete(np.arange(unimputed_3darray.shape[0]), [0, 1])
    # rf_aod_047_wobuffer = joblib.load('../model/saved_model/aod_imputation_model/aod_047_nobuffer.joblib')
    aod_047_wobuffer_pred = rf_aod_047_wobuffer.predict(unimputed_3darray[:, model_047_wobuffer_idx][nobuffer_col, :].T)

    # rf_aod_055_wobuffer = joblib.load('../model/saved_model/aod_imputation_model/aod_055_nobuffer.joblib')
    aod_055_wobuffer_pred = rf_aod_055_wobuffer.predict(unimputed_3darray[:, model_055_wobuffer_idx][nobuffer_col, :].T)

    # Put all prediction together
    aod_3darray[0][model_047_wbuffer_idx] = aod_047_wbuffer_pred
    aod_3darray[0][model_047_wobuffer_idx] = aod_047_wobuffer_pred

    aod_3darray[1][model_055_wbuffer_idx] = aod_055_wbuffer_pred
    aod_3darray[1][model_055_wobuffer_idx] = aod_055_wobuffer_pred

    # Add back to original xarray
    unimputed_xr['aod_047'] = (['y', 'x'], aod_3darray[0])
    unimputed_xr['aod_055'] = (['y', 'x'], aod_3darray[1])

    print('Start Exporting')
    unimputed_xr.to_netcdf(path=f'../data/merged_inputs/{str(date)}_with_imputation.nc',
                           mode='w',)
    print('Exporting Finished')
    os.remove(f'../data/merged_inputs/{str(date)}_wo_imputation.nc')
    print('Unimputed NetCDF is deleted')


if __name__ == "__main__":
    start_date = datetime(2019, 1,1).date()
    end_date = datetime(2022, 1, 1).date()
    delta = timedelta(days=1)

    print('Loading four models')
    '''
    ######### This is for RF ##########
    rf_aod_047_wbuffer = joblib.load('../model/RF_imputation/RF_AOD047_with_buffer.joblib')

    rf_aod_055_wbuffer = joblib.load('../model/RF_imputation/RF_AOD055_with_buffer.joblib')

    # Imputation without Buffer
    rf_aod_047_wobuffer = joblib.load('../model/RF_imputation/RF_AOD047_without_buffer.joblib')

    rf_aod_055_wobuffer = joblib.load('../model/RF_imputation/RF_AOD055_without_buffer.joblib')
    '''
    rf_aod_047_wbuffer = joblib.load('../model/XGB_Imputation/XGB_AOD047_with_buffer.joblib')

    rf_aod_055_wbuffer = joblib.load('../model/XGB_Imputation/XGB_AOD055_with_buffer.joblib')

    # Imputation without Buffer
    rf_aod_047_wobuffer = joblib.load('../model/XGB_Imputation/XGB_AOD047_without_buffer.joblib')

    rf_aod_055_wobuffer = joblib.load('../model/XGB_Imputation/XGB_AOD055_without_buffer.joblib')

    while start_date < end_date:

        print(f"Current Date: {start_date}")
        start_time = time.time()

        if os.path.isfile(f'../data/merged_inputs/{start_date}_with_imputation.nc'):
            print("File already Imputed!")
        else:
            aod_imputation_RF(date=start_date,
                              rf_aod_047_wbuffer=rf_aod_047_wbuffer,
                              rf_aod_055_wbuffer=rf_aod_055_wbuffer,
                              rf_aod_047_wobuffer=rf_aod_047_wobuffer,
                              rf_aod_055_wobuffer=rf_aod_055_wobuffer)
        start_date = start_date + delta

        print("--- %s seconds ---" % (time.time() - start_time))
