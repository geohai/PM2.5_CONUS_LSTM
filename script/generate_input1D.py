import os
from glob import glob
import time

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray


def generate_input1D(start_date, end_date):
    delta = timedelta(days=1)

    i = 0

    while start_date <= end_date:

        print(f"Current Date: {start_date}")
        start_time = time.time()

        merged_xr = xarray.open_dataset(f"../data/merged_inputs/{start_date}_wo_imputation.nc").squeeze()
        print("Extracting points with PM2.5 Measurements.")
        input_arr = merged_xr.to_array().values[:, merged_xr['avg_pm25'].notnull()].T
        #     input_arr = merged_xr.where(merged_xr['avg_ozone'].notnull(), drop=True)
        input_arr = np.c_[input_arr, np.full((input_arr.shape[0], 1), start_date)]

        print("Doing VStack.")
        if i == 0:
            all_arr = input_arr
        else:
            all_arr = np.vstack((all_arr, input_arr))

        start_date = start_date + delta

        i += 1

        print("--- %s seconds ---" % (time.time() - start_time))

    print("All points extracted!")

    input_df = pd.DataFrame(all_arr, columns=list(merged_xr.keys()) + ['date'])
    input_df.to_csv(f'../data/input_1D/{start_date.year-1}_points_wdates.csv', index=False)


if __name__ == "__main__":
    start_date = datetime(2015, 1, 1).date()
    end_date = datetime(2018, 12, 31).date()

    generate_input1D(start_date=start_date, end_date=end_date)
