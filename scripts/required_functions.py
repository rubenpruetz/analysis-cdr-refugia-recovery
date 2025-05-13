
# import required libraries
import rasterio as rs
from rasterio.warp import Resampling
import pandas as pd
import numpy as np
import rioxarray
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns
from rasterio.mask import mask
from shapely.geometry import shape, mapping
from pathlib import Path
path_uea = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/uea_maps/UEA_20km')

# function to resample geotiffs
def tiff_resampler(input_tif,  # input tiff (string)
                   target_resolution,  # target x and y cell resolution (tuple)
                   resampling_method,  # choose rs resampling method (string)
                   output_name):  # output tiff (string)
    with rs.open(input_tif) as src:

        transform, width, height = rs.warp.aligned_target(
            src.transform,
            src.width,
            src.height,
            target_resolution)

        data, transform = rs.warp.reproject(source=src.read(masked=True),
                                            destination=np.zeros(
                                                (src.count, height, width)),
                                            src_transform=src.transform,
                                            dst_transform=transform,
                                            src_crs=src.crs,
                                            dst_crs=src.crs,
                                            dst_nodata=src.nodata,
                                            resampling=Resampling[resampling_method])
        profile = src.profile
        profile.update(transform=transform, driver='GTiff',
                       height=data.shape[1], width=data.shape[2])

        with rs.open(output_name, 'w', **profile) as dst:
            dst.write(data)

# function to create binary raster based on refugia threshold
def binary_converter(input_tif,  # input tif (string)
                     filepath,  # string + /
                     threshold,  # minimum value (integer)
                     output_name):  # specify output name (string)
    with rs.open(filepath / input_tif) as src:
        data = src.read()  # Read the GeoTIFF
        profile = src.profile  # Get metadata of GeoTiff

    output_data = np.where(data >= threshold, 1,
                           np.where(data < threshold, 0, data))

    profile.update(dtype=rs.float32)  # update metadata
    with rs.open(filepath / output_name, 'w', **profile) as dst:  # create and write output file
        dst.write(output_data.astype(profile['dtype']))

# area calculation per raster cell in WGS84
def land_area_calculation(filepath, input_name, output_name=None):
    """
    Function to calc land area for each raster cell in WGS84 without reprojecting
    Adapted from:
    https://gis.stackexchange.com/questions/317392/determine-area-of-cell-in-raster-qgis
    """
    with rs.open(filepath / input_name) as src:
        input_raster = src.read(1)
        input_raster = np.nan_to_num(input_raster, nan=-3.40282e+38)
        profile = src.profile
        gt = src.transform
        pix_width = gt[0]
        ulY = gt[5]  # upper left y
        rows = src.height
        cols = src.width
        lrY = ulY + gt[4] * rows  # lower right y

    lats = np.linspace(ulY, lrY, rows+1)

    a = 6378137  # semi-major axis for WGS84
    b = 6356752.314245179  # semi-minor axis WGS84
    lats = lats * np.pi/180  # degrees to radians
    e = np.sqrt(1-(b/a)**2)
    sinlats = np.sin(lats)
    zm = 1 - e * sinlats
    zp = 1 + e * sinlats
    q = pix_width/360  # distance between meridians

    # compute areas for each latitude
    areas_to_equator = np.pi * b**2 * \
        ((2*np.arctanh(e*sinlats) / (2*e) + sinlats / (zp*zm))) / 10**6
    areas_between_lats = np.diff(areas_to_equator)
    areas_cells = np.abs(areas_between_lats) * q  # unit is km2 (x100 for ha)
    areagrid = np.transpose(np.matlib.repmat(areas_cells, cols, 1))

    # set all values to nan that <= 0 in input (if < area will be calculated for zero values too)
    areagrid[input_raster <= 0] = np.nan

    if output_name:
        with rs.open(filepath / output_name, 'w', **profile) as dst:
            dst.write(areagrid, 1)
    else:
        return areagrid

# sum cells in array that are positive (squeeze removes non-required dims)
def pos_val_summer(arr, squeeze=True):
    if squeeze:
        arr = np.squeeze(arr)

    arr = np.clip(arr, 0, None)  # Set values below zero to 0
    return np.nansum(arr)  # Sum only non-NaN values

# function to concat multiple dfs across models
def load_and_concat(suffix, paths):
    dfs = [pd.read_csv(_path / f'{i}_{suffix}.csv') for i, _path in paths.items()]
    return pd.concat(dfs, ignore_index=True)

# function to overlay raster and admin boundary shapefile
def admin_bound_calculator(key, admin_sf, intersect_src):
    sf = admin_sf
    shapes = sf.shapes()
    records = sf.records()

    country_vals = {}
    for record, shp in zip(records, shapes):  # calc raster vals in polygons
        country_name = record['iso3']
        geom = shape(shp.__geo_interface__)
        # mask the raster with the reprojected geometry
        out_image, _ = mask(intersect_src, [mapping(geom)], crop=True)
        out_image = out_image[0]  # extract the first band

        nodata_value = intersect_src.nodata
        if nodata_value is not None:
            out_image = np.where(out_image == nodata_value, np.nan, out_image)

        total_value = np.nansum(out_image)  # calc sum without nan values
        country_vals[country_name] = total_value
    df = pd.DataFrame(list(country_vals.items()), columns=['iso3', 'km2'])
    df['key'] = key
    return df[['key', 'iso3', 'km2']].copy()

# function to interpolate available land cover years to a certain target year
def land_cover_interpolator(model,
                            filepath,
                            land_cover,
                            scenario,
                            yr_low,  # closest available year before target
                            yr_up,  # closest available year after target
                            yr_target):  # year that shall be returned

    lower_tiff = f'{model}_{land_cover}_{scenario}_{yr_low}.tif'
    upper_tiff = f'{model}_{land_cover}_{scenario}_{yr_up}.tif'
    output_name = f'{model}_{land_cover}_{scenario}_{yr_target}.tif'

    with rs.open(filepath / lower_tiff) as src_low:
        with rs.open(filepath / upper_tiff) as src_up:
            # read raster data and geospatial information
            lower_tiff = src_low.read(1)
            upper_tiff = src_up.read(1)
            profile_lower = src_low.profile

            yr_diff = yr_up - yr_low  # diff of known years
            tiff_diff = upper_tiff - lower_tiff  # diff of known tiffs

            # lower tiff plus the fraction of tiff_diff for a given target yr
            tiff_target = lower_tiff + (tiff_diff * ((yr_target - yr_low) / yr_diff))

            profile_updated = profile_lower.copy()
            profile_updated.update(dtype=rs.float32)

            with rs.open(filepath / output_name, "w", **profile_updated) as dst:
                dst.write(tiff_target.astype(rs.float32), 1)

# function to compare impact on refugia before and after overshoot of 1.5C
def os_land_in_refugia_calculator(model,
                                  filepath,
                                  scenario,
                                  pre_os_yr,  # specify file for start of overshoot
                                  post_os_yr):  # specify file for end of overshoot

    pre_os_ar = f'{model}_Afforestation_{scenario}_{pre_os_yr}.tif'
    post_os_ar = f'{model}_Afforestation_{scenario}_{post_os_yr}.tif'
    pre_os_be = f'{model}_Bioenergy_{scenario}_{pre_os_yr}.tif'
    post_os_be = f'{model}_Bioenergy_{scenario}_{post_os_yr}.tif'

    pre_os_ar = rioxarray.open_rasterio(filepath / pre_os_ar, masked=True)
    post_os_ar = rioxarray.open_rasterio(filepath / post_os_ar, masked=True)
    pre_os_be = rioxarray.open_rasterio(filepath / pre_os_be, masked=True)
    post_os_be = rioxarray.open_rasterio(filepath / post_os_be, masked=True)
    refugia = rioxarray.open_rasterio(path_uea / 'bio1.5_bin.tif',
                                      masked=True)  # adjust file if needed

    # align files
    pre_os_ar = pre_os_ar.rio.reproject_match(refugia)
    post_os_ar = post_os_ar.rio.reproject_match(refugia)
    pre_os_be = pre_os_be.rio.reproject_match(refugia)
    post_os_be = post_os_be.rio.reproject_match(refugia)

    # calculate combined mitigation within refugia for pre and post os
    pre_os = (pre_os_ar + pre_os_be) * refugia
    post_os = (post_os_ar + post_os_be) * refugia

    # calculate absolute change between start end end of overshoot
    os_diff = post_os - pre_os   # positive numbers mean larger post os effect

    # calculate potential maximum area for diff file as input for relative change
    os_diff_bin = os_diff.where(os_diff.isnull(), 1)  # all=1 if not nodata
    os_diff_bin.rio.to_raster(filepath / 'os_diff_bin.tif', driver='GTiff')
    land_area_calculation(filepath, 'os_diff_bin.tif', 'os_diff_bin_max.tif')
    os_diff_max = rioxarray.open_rasterio(filepath / 'os_diff_bin_max.tif',
                                          masked=True)
    os_rel_change = (os_diff / os_diff_max) * 100

    # save file
    os_rel_change.rio.to_raster(filepath / f'{model}_{scenario}_pre_vs_post_os.tif')
