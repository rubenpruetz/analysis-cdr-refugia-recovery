
# import required libraries
import rasterio as rs
from rasterio.warp import Resampling
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
import os
import numpy.matlib
from time import time
from pathlib import Path

from required_functions import *

path_aim = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps')
path_globiom = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps')
path_image = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/image_maps')
path_uea = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/uea_maps/UEA_20km')

# load lookup table containing nc file information
lookup_resample = pd.read_csv(
    path_uea / 'lookup_table_uea_resample_20km.csv')

lookup_interpol = pd.read_csv(
    path_uea / 'lookup_table_uea_interpol_20km.csv')

lookup_globiom_nc_df = pd.read_csv(path_globiom / 'lookup_table_globiom_nc_files.csv')
lookup_globiom_nc_df['year'] = lookup_globiom_nc_df['year'].astype(str)

lookup_aim_nc_df = pd.read_csv(
    path_aim / 'lookup_table_aim_nc_files.csv')

lookup_image_nc_df = pd.read_csv(
    path_image / 'lookup_table_image_nc_files.csv')

lookup_image_nc_pre = pd.read_csv(path_image /
                                  'lookup_table_image_nc_files_preprocessing.csv')

# %% adjust names of biodiv files
for index, row in lookup_resample.iterrows():  # use lookup to resample uea files
    input_tif = row['filename']
    output_name = row['output_name']

    with rs.open(path_uea / input_tif, 'r') as input_tiff1:
        tiff = input_tiff1.read()
        profile = input_tiff1.profile

    with rs.open(path_uea / output_name, 'w', **profile) as dst:
        dst.write(tiff.astype(profile['dtype']))

# linearily interpolate warmig level between rasters
inter_steps = 4  # number of desired interpolation steps

for index, row in lookup_interpol.iterrows():  # use lookup to interpolate uea files
    lower_file = row['lower_file']
    upper_file = row['upper_file']
    step_a = row['step_a']
    step_b = row['step_b']
    step_c = row['step_c']
    step_d = row['step_d']

    with rs.open(path_uea / lower_file, 'r') as input_tiff1:
        lower_tiff = input_tiff1.read()
        profile = input_tiff1.profile

    with rs.open(path_uea / upper_file, 'r') as input_tiff2:
        upper_tiff = input_tiff2.read()

    tiff_diff = upper_tiff - lower_tiff
    step_1 = tiff_diff * (1/(inter_steps+1)) + lower_tiff
    step_2 = tiff_diff * (2/(inter_steps+1)) + lower_tiff
    step_3 = tiff_diff * (3/(inter_steps+1)) + lower_tiff
    step_4 = tiff_diff * (4/(inter_steps+1)) + lower_tiff

    filenames = [step_a, step_b, step_c, step_d]
    files = [step_1, step_2, step_3, step_4]

    for filename, file in zip(filenames, files):
        with rs.open(path_uea / filename, 'w', **profile) as dst:
            dst.write(file.astype(profile['dtype']))

# create binary raster based on refugia threshold (0.75) using binary_converter
input_files = ['bio1.0_near.tif', 'bio1.1_near.tif', 'bio1.2_near.tif',
               'bio1.3_near.tif', 'bio1.4_near.tif', 'bio1.5_near.tif',
               'bio1.6_near.tif', 'bio1.7_near.tif', 'bio1.8_near.tif',
               'bio1.9_near.tif', 'bio2.0_near.tif', 'bio2.1_near.tif',
               'bio2.2_near.tif', 'bio2.3_near.tif', 'bio2.4_near.tif',
               'bio2.5_near.tif', 'bio2.6_near.tif', 'bio2.7_near.tif',
               'bio2.8_near.tif', 'bio2.9_near.tif', 'bio3.0_near.tif',
               'bio3.1_near.tif', 'bio3.2_near.tif', 'bio3.3_near.tif',
               'bio3.4_near.tif', 'bio3.5_near.tif', 'bio3.6_near.tif',
               'bio3.7_near.tif', 'bio3.8_near.tif', 'bio3.9_near.tif',
               'bio4.0_near.tif', 'bio4.1_near.tif', 'bio4.2_near.tif',
               'bio4.3_near.tif', 'bio4.4_near.tif', 'bio4.5_near.tif']

for input_file in input_files:
    output_file = input_file.replace('near.tif', 'bin.tif')
    binary_converter(input_file, path_uea, 0.75, output_file)

# %% AIM, GLOBIOM, and IMAGE land use data processing:

# preprocess GLOBIOM data to order dimensions and to select the data variable
for i in lookup_globiom_nc_df['nc_file'].unique().tolist():

    nc_file_xr = xr.open_dataset(path_globiom / i, decode_times=False)
    nc_file_xr = nc_file_xr[['longitude', 'latitude', 'time', 'lc_class',
                             'GLOBIOM land use projections']]
    os.remove(path_globiom / i)  # delete original file before saving new
    nc_file_xr.to_netcdf(path_globiom / i)

# preprocess IMAGE data to order dimensions and to select the data variable
for index, row in lookup_image_nc_pre.iterrows():
    scenario = row['scenario']
    file = row['file']

    if file == 'GLANDCOVER_30MIN.nc':
        var = 'GLANDCOVER_30MIN'
    else:
        var = 'GLANDCOVERDETAIL_30MIN'

    output_name = scenario + '_' + var + '.nc'
    nc_file_xr = xr.open_dataset(path_image / scenario / file)
    nc_file_xr = nc_file_xr[['longitude', 'latitude', 'time', 'N', var]]
    nc_file_xr = nc_file_xr.stack(time_lc=('time', 'N'))
    nc_file_xr = nc_file_xr.transpose('time_lc', 'latitude', 'longitude')
    nc_file_xr = nc_file_xr.drop_vars(['time_lc', 'time', 'N'])
    nc_file_xr.coords['time_lc'] = (('time_lc'), np.arange(len(nc_file_xr.coords['time_lc'])))
    nc_file_xr.to_netcdf(path_image / output_name)

# %% write crs, convert to tif, and create individual tifs per year and variable
target_res = (0.1666666666670000019, 0.1666666666670000019)  # uea resolution
land_infos = np.array(['Afforestation', 'Bioenergy', 'cropland_other',
                       'forest_total', 'Cropland_total'])  # define for later

start = time()

models = ['AIM', 'GLOBIOM', 'IMAGE']  # AIM, GLOBIOM, and IMAGE

for model in models:

    if model == 'GLOBIOM':
        path = path_globiom
        lookup_table = lookup_globiom_nc_df
    elif model == 'AIM':
        path = path_aim
        lookup_table = lookup_aim_nc_df
    elif model == 'IMAGE':
        path = path_image
        lookup_table = lookup_image_nc_df

    for index, row in lookup_table.iterrows():  # use lookup to resample uea files
        input_file = row['nc_file']
        band = row['band']
        output_name = row['output_name']

        nc_file = rioxarray.open_rasterio(path / input_file,
                                          decode_times=False,
                                          band_as_variable=True)
        data_array_proj = nc_file.rio.write_crs('EPSG:4326')
        data_array_proj = data_array_proj['band_' + str(band)]
        data_array_proj.rio.to_raster(path / 'temp_large_file.tif',
                                      driver='GTiff')

        with rs.open(path / 'temp_large_file.tif') as src:
            data = src.read(1)
            profile = src.profile.copy()
            profile.update(count=1)
        with rs.open(path / output_name, 'w', **profile) as dst:
            dst.write(data, 1)

        # resample land use data to resolution of biodiv data
        tiff_resampler(path / output_name, target_res, 'nearest',
                       path / output_name)

        # ensure that files are capped at 100%
        cap_file = rioxarray.open_rasterio(path / output_name,
                                           decode_times=False,
                                           band_as_variable=True)
        cap_file = cap_file.where((cap_file <= 1) | cap_file.isnull(), 1)  ####################### maybe not required
        cap_file.rio.to_raster(path / output_name, driver='GTiff')

    # compute total bioenergy and forest per scenario and year
    scenarios = lookup_table['scenario'].unique()
    scenarios = scenarios.astype(str)
    years = lookup_table['year'].unique()
    years = years.astype(str)

    for scenario in scenarios:
        for year in years:
            # only relevant for IMAGE
            try:
                energy_crops_ir = f'{model}_energy_crops_ir_{scenario}_{year}.tif'
                energy_crops_rf = f'{model}_energy_crops_rf_{scenario}_{year}.tif'
                output_name = f'{model}_Bioenergy_{scenario}_{year}.tif'

                energy_crops_ir = rioxarray.open_rasterio(path / energy_crops_ir,
                                                          masked=True)
                energy_crops_rf = rioxarray.open_rasterio(path / energy_crops_rf,
                                                          masked=True)

                bioenergy = energy_crops_ir + energy_crops_rf

                bioenergy.rio.to_raster(path / output_name,
                                        driver='GTiff')
            except Exception as e:
                print(f'Error processing: {e}')
                continue

    for scenario in scenarios:
        for year in years:
            # only relevant for AIM and GLOBIOM
            try:
                unmanaged_forest = f'{model}_forest_unmanaged_{scenario}_{year}.tif'
                managed_forest = f'{model}_forest_managed_{scenario}_{year}.tif'
                output_name = f'{model}_forest_total_{scenario}_{year}.tif'

                unmanaged_forest = rioxarray.open_rasterio(path / unmanaged_forest,
                                                           masked=True)
                managed_forest = rioxarray.open_rasterio(path / managed_forest,
                                                         masked=True)

                total_forest = unmanaged_forest + managed_forest

                total_forest.rio.to_raster(path / output_name,
                                           driver='GTiff')
            except Exception as e:
                print(f'Error processing: {e}')
                continue

    # compute afforestation for all years vs 2010
    for scenario in scenarios:
        file_2010 = f'{model}_forest_total_{scenario}_2010.tif'

        for year in years:
            forest_file = f'{model}_forest_total_{scenario}_{year}.tif'
            ar_file_yr = f'{model}_Afforestation_{scenario}_{year}.tif'

            forest_2010 = rioxarray.open_rasterio(
                path / file_2010, masked=True)
            forest_yr = rioxarray.open_rasterio(
                path / forest_file, masked=True)

            forest_change = (forest_yr - forest_2010)  # -ve=loss; +ve=gain

            gain_yr = forest_change.where(
                (forest_change > 0) | forest_change.isnull(), 0)

            gain_yr.rio.to_raster(path / ar_file_yr, driver='GTiff')

    # calculate grid area based on arbitrarily chosen input file
    arbit_input = rioxarray.open_rasterio(
        path / f'{model}_Afforestation_SSP1-19_2050.tif', masked=True)

    bin_land = arbit_input.where(arbit_input.isnull(), 1)  # all=1 if not nodata
    bin_land.rio.to_raster(path / 'bin_land.tif', driver='GTiff')

    land_area_calculation(path, 'bin_land.tif', f'{model}_max_land_area_km2.tif')
    max_land_area = rioxarray.open_rasterio(path /
                                            f'{model}_max_land_area_km2.tif',
                                            masked=True)

    # calculate land use areas based on total surface and land use fractions
    for land_info in land_infos:
        for scenario in scenarios:
            for year in years:

                try:
                    processing = f'{model}_{land_info}_{scenario}_{year}.tif'
                    land_fract = rioxarray.open_rasterio(
                        path / processing, masked=True)

                    land_fract = land_fract.rio.reproject_match(max_land_area)
                    land_area = land_fract * max_land_area

                    land_area.rio.to_raster(path / processing,
                                            driver='GTiff')
                except Exception as e:
                    print(f'Error processing: {e}')
                    continue

end = time()
print(f'Runtime {(end - start) / 60} min')

# %% test alignment between spatial data and AR6 data concerning land cover

scenario_set = ['SSP1-19', 'SSP2-26', 'SSP3-34']
year_set = [2020, 2050, 2100]

for model in models:
    if model == 'GLOBIOM':
        path = path_globiom
    elif model == 'AIM':
        path = path_aim

    for scenario in scenario_set:
        for year in year_set:
            # only relevant for AIM and GLOBIOM
            try:
                cropland_bioeng = f'{model}_Bioenergy_{scenario}_{year}.tif'
                cropland_other = f'{model}_cropland_other_{scenario}_{year}.tif'
                output_name = f'{model}_Cropland_total_{scenario}_{year}.tif'

                cropland_bioeng = rioxarray.open_rasterio(path / cropland_bioeng,
                                                          masked=True)
                cropland_other = rioxarray.open_rasterio(path / cropland_other,
                                                         masked=True)
                total_cropland = cropland_bioeng + cropland_other

                total_cropland.rio.to_raster(path / output_name,
                                             driver='GTiff')
            except Exception as e:
                print(f'Error processing: {e}')
                continue

for model in models:
    if model == 'GLOBIOM':
        path = path_globiom
    elif model == 'AIM':
        path = path_aim
    elif model == 'IMAGE':
        path = path_image

    for scenario in scenario_set:
        for year in year_set:

            forest_total = f'{model}_forest_total_{scenario}_{year}.tif'
            cropland_total = f'{model}_Cropland_total_{scenario}_{year}.tif'

            forest = rioxarray.open_rasterio(path / forest_total,
                                             masked=True)
            cropland = rioxarray.open_rasterio(path / cropland_total,
                                               masked=True)
            forest = pos_val_summer(forest, squeeze=True)
            cropland = pos_val_summer(cropland, squeeze=True)

            forest = forest * 100 / 1000000  # from km2 to Mha
            cropland = cropland * 100 / 1000000  # from km2 to Mha

            print(f'{model} {scenario} {year} Forest: {forest} Mha')
            print(f'{model} {scenario} {year} Cropland: {cropland} Mha')
