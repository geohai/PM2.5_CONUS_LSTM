import os
from glob import glob
import time

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray


def generate_input1D(estimate_date, time_lag=21):
    date_list = (pd.date_range(estimate_date - timedelta(days=time_lag - 1), estimate_date, freq="d")
                 .strftime("%Y-%m-%d")
                 .tolist())

    ref_xr = xarray.open_dataset(f"../data/merged_inputs/{estimate_date}_with_imputation.nc").squeeze()
    idx_arr = np.argwhere(~np.isnan(ref_xr['avg_pm25'].values))
    print(f"Num of Points: {idx_arr.shape[0]}")

    # This is the easiest solution, but takes longer.
    '''
    for i in range(len(date_list)):
        current_xr = xarray.open_dataset(f"../data/merged_inputs/{date_list[i]}_with_imputation.nc").squeeze()
        current_xr = current_xr.drop(["aod_buffer_047", "aod_buffer_055", "knnidw_pm25", "avg_pm25"])

        one_input = current_xr.isel(y=idx_arr[:, 0], x=idx_arr[:, 1]).to_array().values
        one_input = np.diagonal(one_input, axis1=1, axis2=2)
        one_input = np.einsum("xy->yx", one_input)

        if i == 0:
            n_input = [one_input]
        else:
            n_input.append(one_input)

        print(f"Current date: {i+1} / {time_lag}")

    n_input = np.stack(n_input, axis=1)
    n_truth = ref_xr['avg_pm25'].isel(y=idx_arr[:, 0], x=idx_arr[:, 1]).values
    n_truth = np.diagonal(n_truth)
    n_truth = n_truth.reshape(-1, 1)
    # Check Shape
    print(n_input.shape)
    print(n_truth.shape)
    '''
    for i in range(len(date_list)):
        current_xr = xarray.open_dataset(f"../data/merged_inputs/{date_list[i]}_with_imputation.nc").squeeze()
        current_xr = current_xr.drop(["aod_buffer_047", "aod_buffer_055", "avg_pm25"])
        print(f"Current date: {i + 1} / {time_lag}")

        for j in range(idx_arr.shape[0]):
            y = idx_arr[j, 0]
            x = idx_arr[j, 1]

            one_input = current_xr.isel(y=y, x=x).to_array().values
            if j == 0:
                n_daily = [one_input]
            else:
                n_daily.append(one_input)

            # Extract Ground Truth for Last Date
            if i == time_lag-1:
                one_truth = ref_xr['avg_pm25'].isel(y=y, x=x).values
                if j == 0:
                    n_truth = [one_truth]
                else:
                    n_truth.append(one_truth)

        n_daily = np.stack(n_daily, axis=0)
        if i == 0:
            n_input = [n_daily]
        else:
            n_input.append(n_daily)


    n_input = np.stack(n_input, axis=1)
    n_truth = np.stack(n_truth, axis=0)
    n_truth = n_truth.reshape(-1, 1)

    # print(n_input.shape)
    # print(n_truth.shape)

    if os.path.exists(f'../data/input_TS'):
        np.save(f'../data/input_TS/TS_X_{estimate_date}.npy', n_input)
        np.save(f'../data/input_TS/TS_y_{estimate_date}.npy', n_truth)
    else:
        os.makedirs(f'../data/input_TS')
        np.save(f'../data/input_TS/TS_X_{estimate_date}.npy', n_input)
        np.save(f'../data/input_TS/TS_y_{estimate_date}.npy', n_truth)

    print(f"Time-series Inputs for {estimate_date} generated!")


if __name__ == "__main__":
    # start_date = datetime(2005, 8, 25).date()
    start_date = datetime(2005, 8, 25).date()
    end_date = datetime(2022, 1, 1).date()
    delta = timedelta(days=1)

    while start_date < end_date:

        print(f"Current Date: {start_date}")
        start_time = time.time()

        if os.path.isfile(f'../data/input_TS/TS_X_{start_date}.npy'):
            print("File already existed!")
        else:
            generate_input1D(estimate_date=start_date, time_lag=21)

        start_date = start_date + delta

        print("--- %s seconds ---" % (time.time() - start_time))
