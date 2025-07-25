
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
path_gcam = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/gcam_maps')
path_globiom = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps')
path_image = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/image_maps')
path_uea = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/uea_maps/UEA_20km')

# load lookup table containing nc file information
lookup_resample = pd.read_csv(path_uea / 'lookup_table_uea_resample_20km.csv')
lookup_interpol = pd.read_csv(path_uea / 'lookup_table_uea_interpol_20km.csv')

lookup_aim_nc_df = pd.read_csv(path_aim / 'lookup_table_aim_nc_files.csv')

lookup_gcam_nc_df = pd.read_csv(path_gcam / 'lookup_table_gcam_nc_files.csv')

lookup_globiom_nc_df = pd.read_csv(path_globiom / 'lookup_table_globiom_nc_files.csv')
lookup_globiom_nc_df['year'] = lookup_globiom_nc_df['year'].astype(str)

lookup_image_nc_df = pd.read_csv(path_image / 'lookup_table_image_nc_files.csv')

lookup_image_nc_pre = pd.read_csv(path_image /
                                  'lookup_table_image_nc_files_preprocessing.csv')

# specify project resolution
target_res = (0.1666666666670000019, 0.1666666666670000019)  # uea resolution
land_infos = np.array(['Afforestation', 'Bioenergy'])  # define for later

# %% adjust names of biodiv files
for index, row in lookup_resample.iterrows():  # use lookup to resample uea files
    input_tif = row['filename']
    output_name = row['output_name']

    with rs.open(path_uea / input_tif, 'r') as input_tiff1:
        tiff = input_tiff1.read()
        profile = input_tiff1.profile

    with rs.open(path_uea / output_name, 'w', **profile) as dst:
        dst.write(tiff.astype(profile['dtype']))

# linearily interpolate warming level between rasters
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

# %% AIM, GCAM, GLOBIOM, and IMAGE land use data processing:

# preprocess GCAM data to resample and combine PFT data for forest and bioenergy
unique_files = lookup_gcam_nc_df['input_file'].unique()
for input_nc in unique_files:  # stack PFT variables into one DataArray
    ds_gcam = xr.open_dataset(path_gcam / input_nc, decode_times=False)
    pft_vars = [f'PFT{i}' for i in range(33)]
    pft_data = xr.concat([ds_gcam[var] for var in pft_vars], dim="pft")
    pft_data = pft_data.assign_coords(pft=np.arange(len(pft_vars)))

    pft_data = pft_data.transpose('pft', 'latitude', 'longitude')
    ds_new = xr.Dataset({"LC_area_share": pft_data})

    ds_new = ds_new.sel(latitude=slice(None, None, -1))  # flip map
    ds_new = ds_new.expand_dims(time=[0])  # add dummy time dimension
    ds_new.to_netcdf(path_gcam / f'_{input_nc}')

for index, row in lookup_gcam_nc_df.iterrows():
    input_nc = row['input_file']
    band = row['band']
    output_name = row['output_name']

    nc_file = rioxarray.open_rasterio(path_gcam / f'_{input_nc}',
                                      decode_times=False,
                                      band_as_variable=True)
    data_array_proj = nc_file.rio.write_crs('EPSG:4326')
    data_array_proj = data_array_proj['band_' + str(band)]  # bands are pft+1
    data_array_proj.rio.to_raster(path_gcam / 'temp_large_file.tif',
                                  driver='GTiff')

    tiff_resampler(path_gcam / 'temp_large_file.tif', target_res, 'nearest',
                   path_gcam / output_name)

scenarios = lookup_gcam_nc_df['scenario'].unique()
scenarios = scenarios.astype(str)
years = lookup_gcam_nc_df['year'].unique()
years = years.astype(str)

for scenario in scenarios:
    for year in years:

        forest_dict = {}
        for i in range(1, 9):  # PFT 1-8 as in Chen et al. 2020
            f_file = f'GCAM_PFT{i}_Forest_{scenario}_{year}.tif'
            forest_dict[f'f_{i}'] = rioxarray.open_rasterio(path_gcam / f_file,
                                                            masked=True)
        total_forest = sum(forest_dict.values())
        total_forest = total_forest * 0.01  # 0-100 --> 0-1
        total_forest = total_forest.clip(max=1) # cap at 1
        total_f_name = f'GCAM_forest_total_{scenario}_{year}.tif'
        total_forest.rio.to_raster(path_gcam / total_f_name, driver='GTiff')

        bioenergy_dict = {}
        for i in range(29, 31):
            b_file = f'GCAM_PFT{i}_Bioenergy plantation_{scenario}_{year}.tif'
            bioenergy_dict[f'f_{i}'] = rioxarray.open_rasterio(path_gcam / b_file,
                                                               masked=True)
        total_bioenergy = sum(bioenergy_dict.values())
        total_bioenergy = total_bioenergy * 0.01  # 0-100 --> 0-1
        total_bioenergy = total_bioenergy.clip(max=1) # cap at 1
        total_b_name = f'GCAM_Bioenergy_{scenario}_{year}.tif'
        total_bioenergy.rio.to_raster(path_gcam / total_b_name, driver='GTiff')

for scenario in scenarios:  # compute afforestation for all years vs base year
    file_baseyr = f'GCAM_forest_total_{scenario}_2020.tif'

    for year in years:
        forest_file = f'GCAM_forest_total_{scenario}_{year}.tif'
        ar_file_yr = f'GCAM_Afforestation_{scenario}_{year}.tif'

        forest_base_yr = rioxarray.open_rasterio(path_gcam / file_baseyr, masked=True)
        forest_yr = rioxarray.open_rasterio(path_gcam / forest_file, masked=True)

        forest_change = (forest_yr - forest_base_yr)  # -ve=loss; +ve=gain

        gain_yr = forest_change.where(
            (forest_change > 0) | forest_change.isnull(), 0)

        gain_yr.rio.to_raster(path_gcam / ar_file_yr, driver='GTiff')

arbit_input = rioxarray.open_rasterio(path_gcam /  # calculate grid area
                                      'GCAM_Afforestation_SSP2-26_2050.tif',
                                      masked=True)

bin_land = arbit_input.where(arbit_input.isnull(), 1)  # all=1 if not nodata
bin_land.rio.to_raster(path_gcam / 'bin_land.tif', driver='GTiff')

land_area_calculation(path_gcam, 'bin_land.tif', 'GCAM_max_land_area_km2.tif')
max_land_area = rioxarray.open_rasterio(path_gcam / 'GCAM_max_land_area_km2.tif',
                                        masked=True)

for land_info in land_infos:  # calculate areas based on surface and land use fractions
    for scenario in scenarios:
        for year in years:

            processing = f'GCAM_{land_info}_{scenario}_{year}.tif'
            land_fract = rioxarray.open_rasterio(path_gcam / processing, masked=True)
            land_fract = land_fract.rio.reproject_match(max_land_area)
            land_area = land_fract * max_land_area
            land_area.rio.to_raster(path_gcam / processing, driver='GTiff')

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
start = time()

models = ['AIM', 'GLOBIOM', 'IMAGE']

for model in models:

    if model == 'AIM':
        path = path_aim
        lookup_table = lookup_aim_nc_df
    elif model == 'GLOBIOM':
        path = path_globiom
        lookup_table = lookup_globiom_nc_df
    elif model == 'IMAGE':
        path = path_image
        lookup_table = lookup_image_nc_df

    for index, row in lookup_table.iterrows():  # use lookup to resample files
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

    # compute afforestation for all years vs base year
    for scenario in scenarios:
        file_baseyr = f'{model}_forest_total_{scenario}_2010.tif'

        for year in years:
            forest_file = f'{model}_forest_total_{scenario}_{year}.tif'
            ar_file_yr = f'{model}_Afforestation_{scenario}_{year}.tif'

            forest_base_yr = rioxarray.open_rasterio(
                path / file_baseyr, masked=True)
            forest_yr = rioxarray.open_rasterio(
                path / forest_file, masked=True)

            forest_change = (forest_yr - forest_base_yr)  # -ve=loss; +ve=gain

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
