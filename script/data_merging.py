import os.path
from datetime import datetime, timedelta
import calendar
import time
from glob import glob

import rioxarray  # for the extension to load
from rioxarray.merge import merge_arrays
import xarray

import rasterio
from rasterio.plot import show
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata, rasterize_image

import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# from scipy.signal import convolve2d
from scipy import ndimage
from multiprocessing import Pool
from functools import partial


# Viz Packages
# import matplotlib.pyplot as plt
#
# import earthpy as et
# import earthpy.spatial as es
# import earthpy.plot as ep


def print_raster(raster):
    print(
        f"shape: {raster.rio.shape}\n"
        f"resolution: {raster.rio.resolution()}\n"
        f"bounds: {raster.rio.bounds()}\n"
        f"sum: {raster.sum().item()}\n"
        f"CRS: {raster.rio.crs}\n"
    )


def load_aod_5km(start_date):
    """
    :param start_date: datetime.date()
    :return: aod_buffer: xarray.DataArray
    """
    print("Loading AOD 5km Buffer!")

    aod_buffer = rioxarray.open_rasterio(f'../data/aod_buffer/{str(start_date)}.tif')

    print("AOD 5km Buffer Finished")
    return aod_buffer


def load_ndvi(start_date, aod_buffer):
    """
    :param aod_buffer:
    :param start_date: datetime.date()
    :return: ndvi_image: xarray.DataArray
    """
    print('Loading NDVI!')
    ndvi_list = glob('../data/ndvi_processed/*.tif')

    # If WinOS, glob will return path with "\\"
    ndvi_list = list(map(lambda x: x.replace('\\', '/'), ndvi_list))

    ndvi_date_list = [datetime.strptime(date_str.split('/')[-1][:-4],
                                        '%Y-%m-%d').date() for date_str in ndvi_list]

    # find the closest date to the experiment date
    cls_ndvi = min(ndvi_date_list, key=lambda sub: abs(sub - start_date))

    ndvi_image = rioxarray.open_rasterio(f"../data/ndvi_processed/{str(cls_ndvi)}.tif")
    ndvi_image = ndvi_image.where(ndvi_image >= -1.0)

    # Manual: https://gina.alaska.edu/wp-content/uploads/2021/02/eMODIS_derived_NDVI_metrics_ver1.0.pdf
    # ndvi_image = ndvi_image * 0.0001

    # ndvi_image = ndvi_image.interpolate_na(dim='x', method='nearest', max_gap=100000)

    # ndvi_image = ndvi_image.interpolate_na(dim='x', method='nearest')

    ndvi_image = ndvi_image.reindex(y=aod_buffer.y, x=aod_buffer.x, method='nearest')

    print("NDVI Loading Finished")

    return ndvi_image


def load_aod(start_date):
    """
    :param start_date: start_date: datetime.date()
    :return: aod_raw: xarray.DataArray
    """

    print("Loading AOD RAW!")
    aod_raw = rioxarray.open_rasterio(f'../data/aod_processed/{str(start_date)}.tif')

    print("AOD RAW Finished!")
    return aod_raw


def load_dem(ndvi_image):
    """
    :return: dem_image: xarray.DataArray
    """
    print("Loading DEM!")

    dem_image = rioxarray.open_rasterio(f'../data/elevation/elevation_1KMmd_GMTEDmd.tif')
    dem_image = dem_image.rio.reproject_match(ndvi_image)
    # Another way to downscaling
    # dem_image = dem_image.coarsen(y=2, x=2, boundary='pad').mean()
    # Use Nearest Reindex to match coordinates
    # dem_image = dem_image.reindex(y=ndvi_image.y, x=ndvi_image.x, method='nearest')

    print("DEM Finished!")

    return dem_image


def load_daymet(start_date, ndvi_image):
    """

    :param start_date:
    :return:
    """
    print("Loading Daymet!")

    year = start_date.year
    day = start_date.timetuple().tm_yday - 1

    # Check if leap year
    '''
    The Daymet calendar is based on a standard calendar year. All Daymet years have 1 - 365 days, 
    including leap years. For leap years, the Daymet database includes leap day. Values for December 
    31 are discarded from leap years to maintain a 365-day year.
    Source: https://daymet.ornl.gov/overview
    '''
    if calendar.isleap(year):
        leap_year = True
    else:
        leap_year = False

    if (start_date.month == 12) and (start_date.day) == 31:
        day = day - 1

    # ===================== dayl ==============================
    daymet_dayl = xarray.open_dataset(f'../data/meteo/daymet_v4_daily_na_dayl_{year}.nc')

    # # Select by current date
    # # 1. Convert date to Julian date
    daymet_dayl = daymet_dayl.isel(time=day)['dayl']
    daymet_dayl = daymet_dayl.drop(['time', 'lat', 'lon'])

    # # # Initiate CRS to daymet data
    daymet_proj = ("+proj=lcc +ellps=WGS84 +a=6378137 +b=6356752.314245 +lat_1=25 +lat_2=60 +lon_0=-100 +lat_0=42.5 "
                   "+x_0=0 +y_0=0 +units=m +no_defs")
    daymet_dayl = daymet_dayl.rio.write_crs(daymet_proj)
    daymet_dayl = daymet_dayl.rio.reproject_match(ndvi_image)

    # ========================== prcp ================================
    daymet_prcp = xarray.open_dataset(f'../data/meteo/daymet_v4_daily_na_prcp_{year}.nc')

    # # Select by current date
    # # 1. Convert date to Julian date
    daymet_prcp = daymet_prcp.isel(time=day)['prcp']
    daymet_prcp = daymet_prcp.drop(['time', 'lat', 'lon'])

    # # # # Initiate CRS to daymet data
    daymet_prcp = daymet_prcp.rio.write_crs(daymet_proj)
    daymet_prcp = daymet_prcp.rio.reproject_match(ndvi_image)

    # ============================ srad ===============================
    daymet_srad = xarray.open_dataset(f'../data/meteo/daymet_v4_daily_na_srad_{year}.nc')

    # # Select by current date
    # # 1. Convert date to Julian date
    daymet_srad = daymet_srad.isel(time=day)['srad']
    daymet_srad = daymet_srad.drop(['time', 'lat', 'lon'])

    # # # # Initiate CRS to daymet data
    daymet_srad = daymet_srad.rio.write_crs(daymet_proj)
    daymet_srad = daymet_srad.rio.reproject_match(ndvi_image)

    # ============================== swe ====================================
    # daymet_swe = xarray.open_dataset(f'./meteo/daymet_v4_daily_na_swe_{year}.nc')
    #
    # # # Select by current date
    # # # 1. Convert date to Julian date
    # daymet_swe = daymet_swe.isel(time=day)['swe']
    # daymet_swe = daymet_swe.drop(['time', 'lat', 'lon'])
    #
    # # # # # Initiate CRS to daymet data
    # daymet_swe = daymet_swe.rio.write_crs(daymet_proj)
    # daymet_swe = daymet_swe.rio.reproject_match(ndvi_image)

    # ============================== tmax ===================================
    daymet_tmax = xarray.open_dataset(f'../data/meteo/daymet_v4_daily_na_tmax_{year}.nc')

    # # Select by current date
    # # 1. Convert date to Julian date
    daymet_tmax = daymet_tmax.isel(time=day)['tmax']
    daymet_tmax = daymet_tmax.drop(['time', 'lat', 'lon'])

    # # # # Initiate CRS to daymet data
    daymet_tmax = daymet_tmax.rio.write_crs(daymet_proj)
    daymet_tmax = daymet_tmax.rio.reproject_match(ndvi_image)

    # ================================ tmin =================================
    daymet_tmin = xarray.open_dataset(f'../data/meteo/daymet_v4_daily_na_tmin_{year}.nc')

    # # Select by current date
    # # 1. Convert date to Julian date
    daymet_tmin = daymet_tmin.isel(time=day)['tmin']
    daymet_tmin = daymet_tmin.drop(['time', 'lat', 'lon'])

    # # # # Initiate CRS to daymet data
    daymet_tmin = daymet_tmin.rio.write_crs(daymet_proj)
    daymet_tmin = daymet_tmin.rio.reproject_match(ndvi_image)

    # ================================ vp ======================================
    daymet_vp = xarray.open_dataset(f'../data/meteo/daymet_v4_daily_na_vp_{year}.nc')

    # # Select by current date
    # # 1. Convert date to Julian date
    daymet_vp = daymet_vp.isel(time=day)['vp']
    daymet_vp = daymet_vp.drop(['time', 'lat', 'lon'])

    # # # # Initiate CRS to daymet data
    daymet_vp = daymet_vp.rio.write_crs(daymet_proj)
    daymet_vp = daymet_vp.rio.reproject_match(ndvi_image)

    print("Daymet Finished!")

    return daymet_dayl, daymet_prcp, daymet_srad, daymet_tmax, daymet_tmin, daymet_vp, daymet_proj


def load_gridmet(start_date, ndvi_image):
    """

    :param start_date:
    :return:
    """

    print("Loading Gridmet")

    # =========================== wind direction ============================
    year = start_date.year
    gridmet_th = xarray.open_dataset(f'../data/gridmet/th_{year}.nc')

    # # Select by current date
    # # 1. Convert date to Julian date
    gridmet_th = gridmet_th.isel(day=start_date.timetuple().tm_yday - 1)['wind_from_direction']

    # # # # Initiate CRS to daymet data
    gridmet_proj = 'epsg:4326'
    gridmet_th = gridmet_th.rio.write_crs(gridmet_proj)
    gridmet_th = gridmet_th.rio.reproject_match(ndvi_image)

    # ========================== wind velocity ===================================
    gridmet_vs = xarray.open_dataset(f'../data/gridmet/vs_{year}.nc')

    # # Select by current date
    # # 1. Convert date to Julian date
    gridmet_vs = gridmet_vs.isel(day=start_date.timetuple().tm_yday - 1)['wind_speed']

    # # # # Initiate CRS to daymet data
    gridmet_vs = gridmet_vs.rio.write_crs(gridmet_proj)
    gridmet_vs = gridmet_vs.rio.reproject_match(ndvi_image)

    print("Gridmet Finished!")

    return gridmet_th, gridmet_vs


def generate_coords(ndvi_image):
    """
    :param ndvi_image:
    :return:
    """
    print("Generating lat/lon")

    daymet_proj = ("+proj=lcc +ellps=WGS84 +a=6378137 +b=6356752.314245 +lat_1=25 +lat_2=60 +lon_0=-100 +lat_0=42.5 "
                   "+x_0=0 +y_0=0 +units=m +no_defs")

    daymet_vp_copy = xarray.open_dataset('../data/meteo/daymet_v4_daily_na_vp_2005.nc')

    daymet_vp_copy['latitude'] = daymet_vp_copy['lat']
    daymet_vp_copy['longitude'] = daymet_vp_copy['lon']

    daymet_lat_lon = daymet_vp_copy[['latitude', 'longitude']]
    daymet_lat_lon = daymet_lat_lon.drop(['lat', 'lon'])

    # # # # # Initiate CRS to daymet data
    daymet_lat_lon = daymet_lat_lon.rio.write_crs(daymet_proj)
    daymet_lat_lon = daymet_lat_lon.rio.reproject_match(ndvi_image)

    daymet_lat_lon = daymet_lat_lon.where((daymet_lat_lon <= 180))

    print("lat/lon generated!")

    return daymet_lat_lon


def load_wfsmoke(start_date, ndvi_image):
    """

    :param start_date:
    :return:
    """
    print("loading wildfire smoke!")

    # Open example raster
    # gridmet_th.rio.to_raster('./gridmet/th_temp.nc')
    # raster = rasterio.open(r"./gridmet/th_temp.nc")
    gridmet_th_copy = rioxarray.open_rasterio('../data/gridmet/th_2005.nc')
    gridmet_proj = 'epsg:4326'
    raster = rasterio.open(r"../data/gridmet/th_2005.nc")
    # daymet_dayl = xarray.open_rasterio('./meteo/daymet_v4_daily_na_dayl_2005.nc')
    # raster = rasterio.open(r"./meteo/daymet_v4_daily_na_vp_2005.nc")
    density_dict = {
        'NA': -1,
        'Light': 1,
        'Medium': 2,
        'Heavy': 3
    }
    try:
        zipurl = f"https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile" \
                 f"/{start_date.year}/{start_date.month:02d}/hms_smoke{start_date.strftime('%Y%m%d')}.zip"

        # Read in vector
        vector = gpd.read_file(zipurl)

        vector['Density'] = vector['Density'].map(density_dict)

        # Get list of geometries for all features in vector file
        geom = [shapes for shapes in vector.geometry]

        # create tuples of geometry, value pairs, where value is the attribute value you want to burn
        geom_value = ((geom, value) for geom, value in zip(vector.geometry, vector['Density']))

        # Rasterize vector using the shape and transform of the raster
        rasterized_smoke = rasterio.features.rasterize(geom_value,
                                                       out_shape=raster.shape,
                                                       transform=raster.transform,
                                                       all_touched=True,
                                                       fill=0, )  # background value

        smoke_mask = xarray.DataArray(rasterized_smoke,
                                      dims=['y', 'x'],
                                      coords=dict(
                                          x=gridmet_th_copy['x'],
                                          y=gridmet_th_copy['y']),
                                      attrs=dict(
                                          description='Wildfire Smoke Density',
                                      ))

        smoke_mask = smoke_mask.rio.write_crs(gridmet_proj)
        smoke_mask = smoke_mask.rio.reproject_match(ndvi_image)

    except Exception as e:
        print(f'========== Error Information: {e} ==========')

        smoke_mask = gridmet_th_copy[0]
        smoke_mask.values = np.zeros(smoke_mask.shape)

        # smoke_mask = smoke_mask.reindex(y=ndvi_image.y, x=ndvi_image.x, method='nearest')
        smoke_mask = smoke_mask.rio.reproject_match(ndvi_image)

        smoke_mask = smoke_mask.drop(['crs', 'day'])

    print("Wildfire Smoke Finished!")

    return smoke_mask


def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
    return data


def load_date_var(start_date, ndvi_image):
    """

    :param start_date:
    :return:
    """
    print("Load date-related var!")

    daymet_proj = ("+proj=lcc +ellps=WGS84 +a=6378137 +b=6356752.314245 +lat_1=25 +lat_2=60 +lon_0=-100 +lat_0=42.5 "
                   "+x_0=0 +y_0=0 +units=m +no_defs")
    daymet_vp_copy = xarray.open_dataset('../data/meteo/daymet_v4_daily_na_vp_2005.nc')

    daymet_date = xarray.Dataset(data_vars=dict(
        year=(['y', 'x'], np.full(daymet_vp_copy['vp'].shape[1:], float(start_date.year))),
        day=(['y', 'x'], np.full(daymet_vp_copy['vp'].shape[1:], start_date.timetuple().tm_yday)),
        month=(['y', 'x'], np.full(daymet_vp_copy['vp'].shape[1:], start_date.month))
    ),
        coords=dict(
            x=daymet_vp_copy['x'],
            y=daymet_vp_copy['y']),
        attrs=dict(
            description='Date related variables',
        ))

    daymet_date = encode(daymet_date, 'day', 365)
    daymet_date = encode(daymet_date, 'month', 12)

    daymet_date = daymet_date.drop(['day', 'month'])

    # # # # # Initiate CRS to daymet data
    daymet_date = daymet_date.rio.write_crs(daymet_proj)
    daymet_date = daymet_date.rio.reproject_match(ndvi_image)

    daymet_date = daymet_date.where((daymet_date <= 3000))

    print("Date-related vars finished!")

    return daymet_date


def load_EPA_PM(pm_df, start_date, ndvi_image):
    """
    :param pm_df:
    :param start_date:
    :param ndvi_image:
    :return:
    """
    print("Load EPA PM2.5!")
    # Filter date if necessary
    daily_pm_df = pm_df[(pm_df['date_local'] == str(start_date))]

    print(f"Origin: Num of non-NaN values: {daily_pm_df.shape[0]}")
    # Reduce Columns
    daily_pm_df = daily_pm_df[['latitude', 'longitude', 'date_local',
                               'arithmetic_mean', 'local_site_name']]
    daily_pm_df = daily_pm_df.rename(columns={'arithmetic_mean': 'avg_pm25',
                                              'date_local': 'date'})

    daily_pm_gpd = gpd.GeoDataFrame(daily_pm_df,
                                    geometry=gpd.points_from_xy(daily_pm_df.longitude,
                                                                daily_pm_df.latitude,
                                                                crs="EPSG:4326"))

    daily_pm_grid = make_geocube(vector_data=daily_pm_gpd,
                                 measurements=["avg_pm25"],
                                 #                                       resolution=(-1000, 1000),
                                 rasterize_function=partial(rasterize_image, fill=-1),
                                 #                                       output_crs=daymet_image.rio.crs,
                                 #                                       align=(0,0),
                                 like=ndvi_image
                                 )
    print(f"Rasterized: Num of non-NaN values: {np.count_nonzero(~np.isnan(daily_pm_grid['avg_pm25'].values))}")

    #     daily_pm_grid = daily_pm_grid.reindex(y=daymet_image.y, x=daymet_image.x, method="nearest")

    print(f"Rasterized: Num of non-NaN values: {np.count_nonzero(~np.isnan(daily_pm_grid['avg_pm25'].values))}")

    return daily_pm_grid


def idw_weights(distances):
    """
    Function to create the inverse distance weighting
    :param distances:
    :return:
    """
    # Set the power parameter for IDW (e.g., 2)
    power = 1
    # Calculate the IDW weights based on distances
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = 1.0 / distances ** power

    weights[weights == np.inf] = 999

    return weights


def idw_weights_val(distances):
    """
    Function to create the inverse distance weighting
    :param distances:
    :return:
    """
    # Set the power parameter for IDW (e.g., 2)
    power = 1
    # Exclude self-measurements
    distances[distances <= (1000 * 30)] = 0

    # Calculate the IDW weights based on distances
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = 1.0 / distances ** power

    # Excluding Self-measurements when Validating
    weights[weights == np.inf] = 0

    return weights


def knn_idw_pm(daily_pm_xr):
    """
    :param daily_pm_xr:
    :param power:
    :param validation:
    :return:
    """
    base_crs = daily_pm_xr.rio.crs
    daily_pm_xr = daily_pm_xr.rio.reproject("EPSG:3857")

    grid_within_regions = daily_pm_xr.to_dataframe().reset_index()
    grid_within_regions = gpd.GeoDataFrame(
        grid_within_regions["avg_pm25"],
        geometry=gpd.points_from_xy(grid_within_regions.x, grid_within_regions.y)
    )

    daily_pm_gpd = grid_within_regions[grid_within_regions['avg_pm25'].notna()]

    print("Grids Generated!")

    # Get property coordinates
    re_x = daily_pm_gpd.geometry.x
    re_y = daily_pm_gpd.geometry.y

    # Make a list of the coordinates
    points = list(zip(re_y, re_x))

    # Make a list of the weights
    weights = daily_pm_gpd['avg_pm25']

    # Extract the grid coordinates from our GeoDataFrame
    x_coords = grid_within_regions.geometry.x  # [pt.x for pt in grid_within_regions.geometry]
    y_coords = grid_within_regions.geometry.y  # [pt.y for pt in grid_within_regions.geometry]

    # Make a list of the grid coordinates
    grid_points = list(zip(y_coords, x_coords))

    print("Start KNN-IDW Training...")

    # Create a KNeighborsRegressor instance
    knn_regressor_val = KNeighborsRegressor(n_neighbors=9, weights=idw_weights_val, n_jobs=1)

    # Fit the KNeighborsRegressor to the input points and weights
    knn_regressor_val.fit(points, weights)

    # Perform KNN interpolation
    print("Start KNN-IDW Predicting...")
    interpolated_weights_val = knn_regressor_val.predict(grid_points)
    interpolated_weights_val = interpolated_weights_val.reshape(np.flip(daily_pm_xr['avg_pm25'].shape)).T

    # Create a KNeighborsRegressor instance
    knn_regressor = KNeighborsRegressor(n_neighbors=9, weights=idw_weights, n_jobs=1)
    knn_regressor.fit(points, weights)

    print("Start KNN-IDW Predicting...")

    interpolated_weights = knn_regressor.predict(grid_points)
    interpolated_weights = interpolated_weights.reshape(np.flip(daily_pm_xr['avg_pm25'].shape)).T

    print("Calculating Distance...")
    nearest_distance = np.min(knn_regressor_val.kneighbors(grid_points, 3)[0], axis=1)
    nearest_distance = nearest_distance.reshape(daily_pm_xr['avg_pm25'].shape)

    daily_pm_xr['knnidw_pm25_val'] = (('y', 'x'), interpolated_weights_val)
    daily_pm_xr['knnidw_pm25'] = (('y', 'x'), interpolated_weights)
    daily_pm_xr['knnidw_distance'] = (('y', 'x'), nearest_distance)
    daily_pm_xr = daily_pm_xr.drop(['avg_pm25'])

    return daily_pm_xr['knnidw_pm25_val'], daily_pm_xr['knnidw_pm25'], daily_pm_xr['knnidw_distance']


def merge_datasets(start_date, dem_image, daymet_lat_lon, pm_df):
    aod_buffer = load_aod_5km(start_date=start_date)
    ndvi_image = load_ndvi(start_date=start_date, aod_buffer=aod_buffer)
    aod_raw = load_aod(start_date=start_date)
    # Loading dem is too slow
    # dem_image = load_dem(ndvi_image=ndvi_image)
    daymet_dayl, daymet_prcp, daymet_srad, daymet_tmax, daymet_tmin, daymet_vp, daymet_proj = load_daymet(
        start_date=start_date, ndvi_image=ndvi_image)
    gridmet_th, gridmet_vs = load_gridmet(start_date=start_date, ndvi_image=ndvi_image)
    # daymet_lat_lon = genrate_coords(ndvi_image=ndvi_image)
    smoke_mask = load_wfsmoke(start_date=start_date, ndvi_image=ndvi_image)
    daymet_date = load_date_var(start_date=start_date, ndvi_image=ndvi_image)
    # Load EPA PM Data
    pm_image = load_EPA_PM(pm_df=pm_df, start_date=start_date, ndvi_image=ndvi_image)
    pm_image = pm_image.rio.reproject_match(ndvi_image)
    knnidw_pm_val, knnidw_pm, knnidw_distance = knn_idw_pm(daily_pm_xr=pm_image)
    knnidw_pm_val = knnidw_pm_val.rio.reproject_match(ndvi_image)
    knnidw_pm = knnidw_pm.rio.reproject_match(ndvi_image)
    knnidw_distance = knnidw_distance.rio.reproject_match(ndvi_image)

    merge_xr = xarray.combine_by_coords([
        ndvi_image[0].to_dataset(name='ndvi'),
        aod_buffer[0].to_dataset(name='aod_buffer_047'),
        aod_buffer[1].to_dataset(name='aod_buffer_055'),
        aod_raw[0].to_dataset(name='aod_047'),
        aod_raw[1].to_dataset(name='aod_055'),
        dem_image.to_dataset(name='dem'),
        daymet_dayl.to_dataset(name='daymet_dayl'),
        daymet_prcp.to_dataset(name='daymet_prcp'),
        daymet_srad.to_dataset(name='daymet_srad'),
        gridmet_th.to_dataset(name='gridmet_th'),
        daymet_tmax.to_dataset(name='daymet_tmax'),
        daymet_tmin.to_dataset(name='daymet_tmin'),
        daymet_vp.to_dataset(name='daymet_vp'),
        gridmet_vs.to_dataset(name='gridmet_vs'),
        daymet_lat_lon['latitude'].to_dataset(name='daymet_lat'),
        daymet_lat_lon['longitude'].to_dataset(name='daymet_lon'),
        smoke_mask.to_dataset(name='wildfire_smoke'),
        daymet_date['year'].to_dataset(name='year'),
        daymet_date['day_sin'].to_dataset(name='day_sin'),
        daymet_date['day_cos'].to_dataset(name='day_cos'),
        daymet_date['month_sin'].to_dataset(name='month_sin'),
        daymet_date['month_cos'].to_dataset(name='month_cos'),
        knnidw_pm_val.to_dataset(name="knnidw_pm25_val"),
        knnidw_pm.to_dataset(name="knnidw_pm25"),
        knnidw_distance.to_dataset(name="knnidw_distance"),
        pm_image['avg_pm25'].to_dataset(name="avg_pm25")
    ],
        join='left', combine_attrs='drop_conflicts')

    merge_xr = merge_xr.drop(['lambert_conformal_conic', 'crs', 'day', 'spatial_ref'])

    print("Merging finished, start exporting!")
    merge_xr.to_netcdf(path=f'../data/merged_inputs/{start_date}_wo_imputation.nc',
                       mode='w',
                       encoding={'aod_buffer_047': {'dtype': 'float32'},
                                 'aod_buffer_055': {'dtype': 'float32'},
                                 'aod_047': {'dtype': 'float32'},
                                 'aod_055': {'dtype': 'float32'},
                                 'day_cos': {'dtype': 'float32'},
                                 'day_sin': {'dtype': 'float32'},
                                 'daymet_dayl': {'dtype': 'float32'},
                                 'daymet_lat': {'dtype': 'float32'},
                                 'daymet_lon': {'dtype': 'float32'},
                                 'daymet_prcp': {'dtype': 'float32'},
                                 'daymet_srad': {'dtype': 'float32'},
                                 'daymet_tmax': {'dtype': 'float32'},
                                 'daymet_tmin': {'dtype': 'float32'},
                                 'daymet_vp': {'dtype': 'float32'},
                                 'dem': {'dtype': 'float32'},
                                 'gridmet_th': {'dtype': 'float32'},
                                 'gridmet_vs': {'dtype': 'float32'},
                                 'month_cos': {'dtype': 'float32'},
                                 'month_sin': {'dtype': 'float32'},
                                 'ndvi': {'dtype': 'float32'},
                                 'wildfire_smoke': {'dtype': 'int16'},
                                 'year': {'dtype': 'float32'},
                                 'knnidw_pm25_val': {'dtype': 'float32'},
                                 'knnidw_pm25': {'dtype': 'float32'},
                                 'knnidw_distance': {'dtype': 'float32'},
                                 'avg_pm25': {'dtype': 'float32'},
                                 }
                       )

    # xarray.open_dataset(f'./outputs/merged_{start_date}_wo_imputation.nc')

    print("Exporting finished!")


if __name__ == "__main__":
    # start_date = datetime(2005, 8, 5).date()
    start_date = datetime(2007, 12, 10).date()
    end_date = datetime(2010, 1, 1).date()
    delta = timedelta(days=1)

    ndvi_image = load_aod_5km(start_date=start_date)
    dem_image = load_dem(ndvi_image=ndvi_image)
    daymet_lat_lon = generate_coords(ndvi_image=ndvi_image)

    print("Loading EPA's PM2.5 Measurements!")
    pm_list = glob("../data/PM25/*.csv")

    pm_df = [pd.read_csv(file, low_memory=False) for file in pm_list]
    pm_df = pd.concat(pm_df, ignore_index=True)

    # Filter out negative measurements
    pm_df = pm_df[pm_df['arithmetic_mean'] >= 0]

    while start_date < end_date:

        print(f"Current Date: {start_date}")
        start_time = time.time()

        if os.path.isfile(f'../data/merged_inputs/{start_date}_wo_imputation.nc'):
            print("File already existed!")
        else:
            merge_datasets(start_date=start_date,
                           dem_image=dem_image,
                           daymet_lat_lon=daymet_lat_lon,
                           pm_df=pm_df)

        start_date = start_date + delta

        print("--- %s seconds ---" % (time.time() - start_time))
