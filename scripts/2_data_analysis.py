#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:58:18 2025

Draft to be use for the second biodiversity paper when I want to include GCAM and also look at bioenergy (without CCS)

@author: rubenprutz
"""


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import rioxarray
import rasterio as rs
from time import time
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
import cmasher as cmr
from required_functions import *
import shapefile
from pathlib import Path
plt.rcParams.update({'figure.dpi': 600})

path_project = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity')
path_uea = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/uea_maps/UEA_20km')
path_ipl = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/ipl_maps/01_Data')
path_globiom = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps')
path_aim = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps')
path_image = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/image_maps')
path_gcam = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/gcam_maps')
path_cz = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/koppen_geiger_maps/1991_2020')
path_ar6_data = Path('/Users/rpruetz/Documents/phd/datasets')

ar6_db = pd.read_csv(path_ar6_data / 'AR6_Scenarios_Database_World_v1.1.csv')
lookup_mi_cdr_df = pd.read_csv(path_project / 'lookup_table_cdr_files_all_models.csv')
lookup_mi_cdr_df['year'] = lookup_mi_cdr_df['year'].astype(str)

# %% get temperatures for SSP-RCP combinations
all_years = [str(year) for year in range(2020, 2101)]

models = ['MESSAGE-GLOBIOM 1.0', 'AIM/CGE 2.0', 'GCAM 4.2', 'IMAGE 3.0.1']
scenarios = ['SSP1-Baseline', 'SSP1-19', 'SSP1-26', 'SSP1-34', 'SSP1-45',
             'SSP2-Baseline', 'SSP2-19', 'SSP2-26', 'SSP2-34', 'SSP2-45',
             'SSP2-60', 'SSP3-Baseline', 'SSP3-34', 'SSP3-45', 'SSP3-60']
variable = ['AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile']

ar6_data = ar6_db.loc[ar6_db['Variable'].isin(variable)]
ar6_data = ar6_data.loc[ar6_data['Model'].isin(models)]
ar6_data = ar6_data.loc[ar6_data['Scenario'].isin(scenarios)]
ar6_data = ar6_data.round(1)  # round temperatures

# allow no temperature decline by calculating peak warming up until each year
for year in range(2021, 2101):
    cols_til_year = ar6_data.loc[:, '2020':str(year)]
    ar6_data[f'{year}_max'] = cols_til_year.max(axis=1)

cols = ['Model', 'Scenario', '2020'] + [f'{year}_max' for year in range(2021, 2101)]
ar6_data_stab = ar6_data[cols]
ar6_data_stab = ar6_data_stab.rename(columns={f'{year}_max': str(year) for year in all_years})

ar6_data = ar6_data[['Model', 'Scenario'] + all_years].copy()

# %% calculate AR and BECCS removals for SSP-RCP combinations
numeric_cols20 = [str(year) for year in range(2020, 2110, 10)]

if model == 'GLOBIOM':
    ar_variable = 'Carbon Sequestration|Land Use|Afforestation'
elif model == 'AIM':
    ar_variable = 'Carbon Sequestration|Land Use|Afforestation'
elif model == 'IMAGE':  # for IMAGE afforestation is not available
    ar_variable = 'Carbon Sequestration|Land Use'

ar6_db = ar6_db.loc[ar6_db['Model'].isin([model_setup]) & ar6_db['Scenario'].isin(scenarios)]
cdr = ar6_db[['Scenario', 'Variable'] + numeric_cols20].copy()
cdr_array = ['Carbon Sequestration|CCS|Biomass',
             ar_variable]
cdr = cdr[cdr['Variable'].isin(cdr_array)]
cdr[numeric_cols20] = cdr[numeric_cols20].clip(lower=0)  # set negative values to zero
cdr = cdr.melt(id_vars=['Scenario', 'Variable'], var_name='Year', value_name='Removal')
cdr['Removal'] = cdr['Removal'] * 0.001  # Mt to Gt
cdr['Year'] = pd.to_numeric(cdr['Year'])

ar_removal = cdr[cdr['Variable'] == ar_variable]
ar_removal['Variable'] = 'AR removal'

beccs_removal = cdr[cdr['Variable'] == 'Carbon Sequestration|CCS|Biomass']
beccs_removal['Variable'] = 'BECCS removal'

# calculate BECCS removal through energy crops only (no residues)
ec_share = energy_crop_share.loc[energy_crop_share['Model'].isin([model])]
beccs_removal = pd.merge(beccs_removal,
                         ec_share[['Scenario', 'Year', 'Share_energy_crops']],
                         on=['Scenario', 'Year'])
beccs_removal['Removal'] = beccs_removal['Removal'] * beccs_removal['Share_energy_crops']

# %% choose between biodiv recovery or no recovery after peak warming

temperature_decline = 'not_allowed'  # options: 'allowed' or 'not_allowed'

if temperature_decline == 'allowed':
    warm_file = ar6_data.copy()
elif temperature_decline == 'not_allowed':
    warm_file = ar6_data_stab.copy()

bio_select = warm_file.set_index(['Model', 'Scenario'])
bio_select = 'bio' + \
    bio_select.select_dtypes(include=np.number).astype(str) + '_bin.tif'
bio_select.reset_index(inplace=True)

# rename models for the subsequent step
bio_select.replace({'Model': {'AIM/CGE 2.0': 'AIM',
                              'MESSAGE-GLOBIOM 1.0': 'GLOBIOM',
                              'GCAM 4.2': 'GCAM',
                              'IMAGE 3.0.1': 'IMAGE'}}, inplace=True)

# %% choose model to run the script with
model = 'AIM'  # options: 'GLOBIOM' or 'AIM' or 'IMAGE'

if model == 'GLOBIOM':
    path = path_globiom
    model_setup = 'MESSAGE-GLOBIOM 1.0'
    removal_lvl = 2
elif model == 'AIM':
    path = path_aim
    model_setup = 'AIM/CGE 2.0'
    removal_lvl = 2
elif model == 'IMAGE':
    path = path_image
    model_setup = 'IMAGE 3.0.1'
    removal_lvl = 2
elif model == 'GCAM':
    path = path_gcam
    model_setup = 'GCAM 4.2'
    removal_lvl = 2

# %% calculate CDR land impact over time
years = ['2020', '2040', '2060', '2080', '2100']
lookup_sub_yrs = lookup_mi_cdr_df.copy()
lookup_sub_yrs = lookup_sub_yrs.loc[lookup_sub_yrs['year'].isin(years)]

# load climate zone files
cz1 = rioxarray.open_rasterio(path_cz / 'clim_zon_class1.tif', masked=True)
cz2 = rioxarray.open_rasterio(path_cz / 'clim_zon_class2.tif', masked=True)
cz3 = rioxarray.open_rasterio(path_cz / 'clim_zon_class3.tif', masked=True)
cz4 = rioxarray.open_rasterio(path_cz / 'clim_zon_class4.tif', masked=True)
cz5 = rioxarray.open_rasterio(path_cz / 'clim_zon_class5.tif', masked=True)

start = time()  # runtime monitoring

def overlay_calculator(input_tif,  # land use model input file (string)
                       filepath,  # filepath input file + / (string)
                       file_year,  # year of input file (string)
                       file_scenario,  # input file SSP-RCP scenario (string)
                       mitigation_option,  # 'Afforestation' or 'Bioenergy'
                       lu_model):  # GLOBIOM or AIM or GCAM or IMAGE

    # load files for CDR and refugia
    land_use = rioxarray.open_rasterio(filepath / f'{lu_model}_{input_tif}',
                                       masked=True)

    bio_file = ''.join(bio_select[(bio_select['Model'] == lu_model) &
                                  (bio_select['Scenario'] == file_scenario)][file_year])

    refugia = rioxarray.open_rasterio(path_uea / bio_file, masked=True)
    refugia = refugia.rio.reproject_match(land_use)  # align files

    # calculate land impact on refugia (global)
    cdr_in_bio = land_use * refugia

    # calculate refugia extent
    bio_area = land_area_calculation(path_uea, bio_file)

    # calculate aggregated area "losses" and refugia
    cdr_in_bio_agg = pos_val_summer(cdr_in_bio, squeeze=True)
    bio_area_agg = pos_val_summer(bio_area, squeeze=True)

    # calculate warming and land impact on refugia (regional)
    cdr_in_bio_regs = []
    bio_area_regs = []

    cz_values = [cz1, cz2, cz3, cz4, cz5]

    for cz in cz_values:
        cz_m = cz.rio.reproject_match(land_use)  # ensure consistent match
        cdr_in_bio_cz = (land_use * refugia) * cz_m

        refugia_cz = refugia * cz_m
        refugia_cz.rio.to_raster(path_uea / 'refugia_cz_temp.tif', driver='GTiff')
        refugia_cz = land_area_calculation(path_uea, 'refugia_cz_temp.tif')

        cdr_in_bio_cz = pos_val_summer(cdr_in_bio_cz, squeeze=True)
        refugia_cz = pos_val_summer(refugia_cz, squeeze=True)

        # calculate regional area "losses" and refugia
        cdr_in_bio_regs.append(cdr_in_bio_cz)
        bio_area_regs.append(refugia_cz)

    return cdr_in_bio_agg, bio_area_agg, cdr_in_bio_regs, bio_area_regs

# use overlay_calculator
def process_row(row):
    input_tif = row['file_name']
    file_year = row['year']
    file_scenario = row['scenario']
    mitigation_option = row['mitigation_option']

    try:
        # run overlay_calculator for all scenarios to retrieve areas as outputs
        cdr_in_bio_agg, bio_area_agg, cdr_in_bio_regs, \
            bio_area_regs = overlay_calculator(input_tif,
                                               path,
                                               file_year,
                                               file_scenario,
                                               mitigation_option,
                                               model)

        # create a dictionary with the calculated values
        result_dict = {
            'scenario': file_scenario,
            'mitigation_option': mitigation_option,
            'year': file_year,
            'bio_area': bio_area_agg,
            'cdr_in_bio': cdr_in_bio_agg,
            'bio_area_reg': bio_area_regs,
            'cdr_in_bio_reg': cdr_in_bio_regs}

        return result_dict

    except:
        print(f'Unsuccessful for file: {input_tif}')
        return {
            'scenario': file_scenario,
            'mitigation_option': mitigation_option,
            'year': file_year,
            'bio_area': float('nan'),
            'cdr_in_bio': float('nan'),
            'bio_area_reg': [float('nan')] * 5,
            'cdr_in_bio_reg': [float('nan')] * 5}


area_df = pd.DataFrame.from_records(lookup_sub_yrs.apply(process_row,
                                                         axis=1).values)
area_df = area_df.reset_index(drop=True)

cz_columns = pd.DataFrame(area_df['bio_area_reg'].to_list(),
                          columns=['bio_area_cz1',
                                   'bio_area_cz2',
                                   'bio_area_cz3',
                                   'bio_area_cz4',
                                   'bio_area_cz5'])
area_df = pd.concat([area_df.drop(columns='bio_area_reg'), cz_columns], axis=1)

cz_columns = pd.DataFrame(area_df['cdr_in_bio_reg'].to_list(),
                          columns=['cdr_in_bio_cz1',
                                   'cdr_in_bio_cz2',
                                   'cdr_in_bio_cz3',
                                   'cdr_in_bio_cz4',
                                   'cdr_in_bio_cz5'])
area_df = pd.concat([area_df.drop(columns='cdr_in_bio_reg'), cz_columns], axis=1)

end = time()
print(f'Runtime {(end - start) / 60} min')

area_df['alloc_perc'] = area_df['cdr_in_bio'] / area_df['bio_area'] * 100
for i in range(1, 6):  # calculate land loss percentages for all climate zones
    area_df[f'alloc_perc_cz{i}'] = area_df[f'cdr_in_bio_cz{i}'] / area_df[f'bio_area_cz{i}'] * 100

area_df['SSP'] = area_df['scenario'].str.split('-').str[0]
area_df['RCP'] = area_df['scenario'].str.split('-').str[1]
area_df.rename(columns={'year': 'Year'}, inplace=True)
area_df['Model'] = f'{model}'
area_df.to_csv(path / f'{model}_area_df_clim_zone_temp_decline_{temperature_decline}.csv', index=False)

# %%

paths = {'GLOBIOM': path_globiom, 'AIM': path_aim, 
         'IMAGE': path_image, 'GCAM': path_gcam}
area_df = load_and_concat('area_df_clim_zone', paths)

cdr_option = 'Afforestation'
area_df = area_df.query('mitigation_option == @cdr_option')

rcp_pal = {'19': '#00adcf', '26': '#173c66', '34': '#f79320',
           '45': '#e71d24', '60': '#951b1d', 'Baseline': 'dimgrey'}
all_rcps = sorted(area_df['RCP'].unique())

fig, axes = plt.subplots(4, 6, figsize=(12, 8), sharex=True, sharey=True)
sns.lineplot(data=area_df.query('Model == "AIM"'), x='Year', y='alloc_perc',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[0, 0])
sns.lineplot(data=area_df.query('Model == "GLOBIOM"'), x='Year', y='alloc_perc',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), hue_order=all_rcps, legend=True, ax=axes[1, 0])
sns.lineplot(data=area_df.query('Model == "GCAM"'), x='Year', y='alloc_perc',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[2, 0])
sns.lineplot(data=area_df.query('Model == "IMAGE"'), x='Year', y='alloc_perc',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[3, 0])

sns.lineplot(data=area_df.query('Model == "AIM"'), x='Year', y='alloc_perc_cz1',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[0, 1])
sns.lineplot(data=area_df.query('Model == "GLOBIOM"'), x='Year', y='alloc_perc_cz1',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[1, 1])
sns.lineplot(data=area_df.query('Model == "GCAM"'), x='Year', y='alloc_perc_cz1',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[2, 1])
sns.lineplot(data=area_df.query('Model == "IMAGE"'), x='Year', y='alloc_perc_cz1',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[3, 1])

sns.lineplot(data=area_df.query('Model == "AIM"'), x='Year', y='alloc_perc_cz2',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[0, 2])
sns.lineplot(data=area_df.query('Model == "GLOBIOM"'), x='Year', y='alloc_perc_cz2',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[1, 2])
sns.lineplot(data=area_df.query('Model == "GCAM"'), x='Year', y='alloc_perc_cz2',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[2, 2])
sns.lineplot(data=area_df.query('Model == "IMAGE"'), x='Year', y='alloc_perc_cz2',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[3, 2])

sns.lineplot(data=area_df.query('Model == "AIM"'), x='Year', y='alloc_perc_cz3',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[0, 3])
sns.lineplot(data=area_df.query('Model == "GLOBIOM"'), x='Year', y='alloc_perc_cz3',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[1, 3])
sns.lineplot(data=area_df.query('Model == "GCAM"'), x='Year', y='alloc_perc_cz3',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[2, 3])
sns.lineplot(data=area_df.query('Model == "IMAGE"'), x='Year', y='alloc_perc_cz3',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[3, 3])

sns.lineplot(data=area_df.query('Model == "AIM"'), x='Year', y='alloc_perc_cz4',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[0, 4])
sns.lineplot(data=area_df.query('Model == "GLOBIOM"'), x='Year', y='alloc_perc_cz4',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[1, 4])
sns.lineplot(data=area_df.query('Model == "GCAM"'), x='Year', y='alloc_perc_cz4',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[2, 4])
sns.lineplot(data=area_df.query('Model == "IMAGE"'), x='Year', y='alloc_perc_cz4',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[3, 4])

sns.lineplot(data=area_df.query('Model == "AIM"'), x='Year', y='alloc_perc_cz5',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[0, 5])
sns.lineplot(data=area_df.query('Model == "GLOBIOM"'), x='Year', y='alloc_perc_cz5',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[1, 5])
sns.lineplot(data=area_df.query('Model == "GCAM"'), x='Year', y='alloc_perc_cz5',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[2, 5])
sns.lineplot(data=area_df.query('Model == "IMAGE"'), x='Year', y='alloc_perc_cz5',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[3, 5])

axes[1, 0].legend(bbox_to_anchor=(-0.05, 2.6), loc='upper left', ncols=12,
                  columnspacing=0.8, handletextpad=0.1)

axes[0, 0].set_title('Global')
axes[0, 1].set_title('Tropical')
axes[0, 2].set_title('Arid')
axes[0, 3].set_title('Temperate')
axes[0, 4].set_title('Cold')
axes[0, 5].set_title('Polar')

axes[3, 0].set_xlabel('')
axes[3, 1].set_xlabel('')
axes[3, 2].set_xlabel('')
axes[3, 3].set_xlabel('')
axes[3, 4].set_xlabel('')
axes[3, 5].set_xlabel('')

axes[0, 0].set_ylabel('AIM')
axes[1, 0].set_ylabel('GLOBIOM')
axes[2, 0].set_ylabel('GCAM')
axes[3, 0].set_ylabel('IMAGE')

fig.supylabel(f'Remaining refugia allocated for {cdr_option} [%] (SSP1-SSP3 range)', 
              x=0.05, va='center')

for ax in axes.flat:
    ax.set_xlim(2020, 2100)
    ax.set_xticks([2020, 2060, 2100])

plt.subplots_adjust(hspace=0.15)
plt.subplots_adjust(wspace=0.4)
sns.despine()
plt.show()