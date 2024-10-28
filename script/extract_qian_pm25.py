import numpy as np
import pandas as pd
import geopandas as gpd
import xarray
import rioxarray

# Rasterize Package
from functools import partial
from shapely.geometry import box, mapping

from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata, rasterize_image

from datetime import datetime, timedelta
from glob import glob
import time

from sklearn.metrics import mean_squared_error


def load_EPA_PM25(pm_df, start_date, daymet_image):
    # Filter date if necessary
    day_pm_df = pm_df[(pm_df['date_local'] == str(start_date))]
    print(f"Origin: Num of non-NaN values: {day_pm_df.shape[0]}")
    # Reduce Columns
    day_pm_df = day_pm_df[['latitude', 'longitude', 'date_local',
                           'arithmetic_mean', 'local_site_name']]
    day_pm_df = day_pm_df.rename(columns={'arithmetic_mean': 'avg_pm25',
                                          'date_local': 'date'})
    # Convert to Rasters
    day_pm_gpd = gpd.GeoDataFrame(day_pm_df,
                                  geometry=gpd.points_from_xy(day_pm_df.longitude,
                                                              day_pm_df.latitude,
                                                              crs="EPSG:4326"))

    day_pm_grid = make_geocube(vector_data=day_pm_gpd,
                               measurements=["avg_pm25"],
                               #                                       resolution=(-1000, 1000),
                               rasterize_function=partial(rasterize_image, fill=-1),
                               #                                       output_crs=daymet_image.rio.crs,
                               #                                       align=(0,0),
                               like=daymet_image
                               )
    print(f"Rasterized: Num of non-NaN values: {np.count_nonzero(~np.isnan(day_pm_grid['avg_pm25'].values))}")

    return day_pm_grid


def load_qian_pm25(start_date, daymet_image):
    qian_xr = rioxarray.open_rasterio(f"/home/zhongying/Documents/PM2.5_estimation/qian_pm25"
                                      f"/{str(start_date).replace('-', '')}.tif").squeeze()

    qian_xr = qian_xr.rio.reproject(daymet_image.rio.crs)
    qian_xr = qian_xr.reindex(y=daymet_image.y,
                              x=daymet_image.x,
                              method='nearest')

    qian_xr = qian_xr.where(qian_xr > 0)

    return qian_xr


if __name__ == "__main__":
    print("Loading EPA's PM2.5 Measurements!")
    pm_list = glob("../data/PM25/*.csv")

    pm_df = [pd.read_csv(file, low_memory=False) for file in pm_list]
    pm_df = pd.concat(pm_df, ignore_index=True)

    # Filter out negative measurements
    pm_df = pm_df[pm_df['arithmetic_mean'] >= 0]

    # Define Evaluation Period
    # start_date = datetime(2005, 8, 5).date()
    start_date = datetime(2016, 1, 1).date()
    end_date = datetime(2017, 1, 1).date()
    delta = timedelta(days=1)

    daymet_image = xarray.open_dataset("../output/ensemble/2005-08-25.nc",
                                       decode_coords="all")
    eval_df = None

    while start_date < end_date:
        print(f"Current Date: {start_date}")
        start_time = time.time()

        # Load Data
        truth_xr = load_EPA_PM25(pm_df=pm_df, start_date=start_date, daymet_image=daymet_image)

        qian_xr = load_qian_pm25(start_date=start_date, daymet_image=daymet_image)

        # Start Evaluating
        idx_arr = np.argwhere(~np.isnan(truth_xr['avg_pm25'].values))

        truth_arr = np.diagonal(truth_xr['avg_pm25'].isel(y=idx_arr[:, 0],
                                                          x=idx_arr[:, 1]))

        coords_arr =np.array(
            (truth_xr['avg_pm25'].isel(y=idx_arr[:, 0],x=idx_arr[:, 1]).coords["y"].values,
             truth_xr['avg_pm25'].isel(y=idx_arr[:, 0],x=idx_arr[:, 1]).coords["x"].values
             )
        ).T

        qian_arr = np.diagonal(qian_xr.isel(y=idx_arr[:, 0],
                                            x=idx_arr[:, 1]))

        # Get rid of NaNs
        nonan_idx = np.argwhere(~np.isnan(qian_arr))
        truth_arr = truth_arr[nonan_idx].squeeze()
        qian_arr = qian_arr[nonan_idx].squeeze()
        coords_arr = coords_arr[nonan_idx].squeeze()

        qian_rmse = mean_squared_error(truth_arr,
                                       qian_arr,
                                       squared=False)

        qian_mbe = np.mean(qian_arr - truth_arr)

        print(f"Qian Model: RMSE: {qian_rmse} | MBE: {qian_mbe}")

        daily_eval_df = pd.DataFrame(coords_arr, columns=['y', 'x'])
        daily_eval_df['date'] = start_date
        daily_eval_df['truth_pm25'] = truth_arr
        daily_eval_df['qian_pm25'] = qian_arr

        if eval_df is None:
            eval_df = daily_eval_df
        else:
            eval_df = pd.concat([eval_df, daily_eval_df])

        # Export Annually when last day of the year
        if start_date.month == 12 and start_date.day == 31:
            eval_df.to_csv(f"../eval/Qian_Eval_{start_date.year}.csv", index=False)

        start_date = start_date + delta

        print("--- %s seconds ---" % (time.time() - start_time))
