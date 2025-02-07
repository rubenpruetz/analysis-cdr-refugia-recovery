
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
path_globiom = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps')
path_aim = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps')
path_image = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/image_maps')
path_gcam = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/gcam_maps')
path_cz = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/koppen_geiger_maps/1991_2020')
path_ar6_data = Path('/Users/rpruetz/Documents/phd/datasets')

ar6_db = pd.read_csv(path_ar6_data / 'AR6_Scenarios_Database_World_v1.1.csv')
lookup_mi_luc_df = pd.read_csv(path_project / 'lookup_table_ar_bioenergy_files_all_models.csv')
lookup_mi_luc_df['year'] = lookup_mi_luc_df['year'].astype(str)

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

# %% specify years for the analysis
years = ['2030', '2050', '2080', '2100']
lookup_sub_yrs = lookup_mi_luc_df.copy()
lookup_sub_yrs = lookup_sub_yrs.loc[lookup_sub_yrs['year'].isin(years)]

# load climate zone files
cz1 = rioxarray.open_rasterio(path_cz / 'clim_zon_class1.tif', masked=True)
cz2 = rioxarray.open_rasterio(path_cz / 'clim_zon_class2.tif', masked=True)
cz3 = rioxarray.open_rasterio(path_cz / 'clim_zon_class3.tif', masked=True)
cz4 = rioxarray.open_rasterio(path_cz / 'clim_zon_class4.tif', masked=True)
cz5 = rioxarray.open_rasterio(path_cz / 'clim_zon_class5.tif', masked=True)

# %% choose model to run the script with
models = ['AIM', 'GCAM', 'GLOBIOM', 'IMAGE']

for model in models:
    if model == 'GLOBIOM':
        path = path_globiom
    elif model == 'AIM':
        path = path_aim
    elif model == 'IMAGE':
        path = path_image
    elif model == 'GCAM':
        path = path_gcam

    start = time()  # runtime monitoring

    def overlay_calculator(input_tif,  # land use model input file (string)
                           filepath,  # filepath input file
                           file_year,  # year of input file (string)
                           file_scenario,  # input file SSP-RCP scenario (string)
                           mitigation_option,  # 'Afforestation' or 'Bioenergy'
                           biodiv_ref_warm_file,  # Bio file for ref warm (1.3C)
                           lu_model):  # AIM, GCAM, GLOBIOM or IMAGE

        # STEP1: load files for CDR, refugia, and baseline refugia
        land_use = rioxarray.open_rasterio(filepath / f'{lu_model}_{input_tif}',
                                           masked=True)  # mask nan values for calc

        bio_file_wCDR = ''.join(bio_select[(bio_select['Model'] == lu_model) &
                                           (bio_select['Scenario'] == file_scenario)][file_year])

        refugia = rioxarray.open_rasterio(path_uea / bio_file_wCDR,
                                          masked=True)  # mask nan values for calc

        refugia_ref = rioxarray.open_rasterio(path_uea / biodiv_ref_warm_file,
                                              masked=True)

        # align files
        refugia = refugia.rio.reproject_match(land_use)
        refugia_ref = refugia_ref.rio.reproject_match(land_use)

        # calculate warming and land impact on reference refugia (global)
        cdr_in_bio_ref = land_use * refugia_ref
        ref_bio_warm_loss = refugia_ref - refugia
        ref_bio_warm_loss.rio.to_raster(path_uea / 'ref_bio_warm_loss_temp.tif',
                                        driver='GTiff')

        # calculate refugia extents
        ref_bio_warm_loss_a = land_area_calculation(path_uea,
                                                    'ref_bio_warm_loss_temp.tif')
        refugia_ref_a = land_area_calculation(path_uea, biodiv_ref_warm_file)

        # calculate aggregated area "losses" and baseline refugia
        cdr_in_bio_ref_agg = pos_val_summer(cdr_in_bio_ref, squeeze=True)
        ref_bio_warm_loss_agg = pos_val_summer(ref_bio_warm_loss_a, squeeze=True)
        refugia_ref_agg = pos_val_summer(refugia_ref_a, squeeze=True)

        # calculate warming and land impact on reference refugia (regional)
        cdr_in_bio_ref_regs = []
        ref_bio_warm_loss_regs = []
        refugia_ref_regs = []

        cz_values = [cz1, cz2, cz3, cz4, cz5]

        for cz in cz_values:
            cz_m = cz.rio.reproject_match(land_use)  # ensure consistent match
            cdr_in_bio_ref_cz = (land_use * refugia_ref) * cz_m

            ref_bio_warm_loss_cz = (refugia_ref - refugia) * cz_m
            ref_bio_warm_loss_cz.rio.to_raster(path_uea /
                                               'ref_bio_warm_loss_cz_temp.tif',
                                               driver='GTiff')
            ref_bio_warm_loss_cz = land_area_calculation(path_uea,
                                                         'ref_bio_warm_loss_cz_temp.tif')

            refugia_ref_cz = refugia_ref * cz_m
            refugia_ref_cz.rio.to_raster(path_uea / 'refugia_ref_cz_temp.tif',
                                         driver='GTiff')
            refugia_ref_cz = land_area_calculation(path_uea,
                                                   'refugia_ref_cz_temp.tif')

            cdr_in_bio_ref_cz = pos_val_summer(cdr_in_bio_ref_cz, squeeze=True)
            ref_bio_warm_loss_cz = pos_val_summer(ref_bio_warm_loss_cz, squeeze=True)
            refugia_ref_cz = pos_val_summer(refugia_ref_cz, squeeze=True)

            # calculate regional area "losses" and baseline refugia
            cdr_in_bio_ref_regs.append(cdr_in_bio_ref_cz)
            ref_bio_warm_loss_regs.append(ref_bio_warm_loss_cz)
            refugia_ref_regs.append(refugia_ref_cz)

        return cdr_in_bio_ref_agg, ref_bio_warm_loss_agg, refugia_ref_agg, \
            cdr_in_bio_ref_regs, ref_bio_warm_loss_regs, refugia_ref_regs

    # use overlay_calculator
    def process_row(row):
        input_tif = row['file_name']
        file_year = row['year']
        file_scenario = row['scenario']
        mitigation_option = row['mitigation_option']

        try:
            # run overlay_calculator for all scenarios to retrieve areas as outputs
            cdr_in_bio_ref_agg, ref_bio_warm_loss_agg, refugia_ref_agg, \
                cdr_in_bio_ref_regs, ref_bio_warm_loss_regs, refugia_ref_regs = \
                    overlay_calculator(input_tif,
                                       path,
                                       file_year,
                                       file_scenario,
                                       mitigation_option,
                                       'bio1.3_bin.tif',
                                       model)

            # create a dictionary with the calculated values
            result_dict = {
                'scenario': file_scenario,
                'mitigation_option': mitigation_option,
                'year': file_year,
                'refug_ref': refugia_ref_agg,
                'cdr_in_refug_ref': cdr_in_bio_ref_agg,
                'refug_ref_warm_loss': ref_bio_warm_loss_agg,
                'refug_ref_reg': refugia_ref_regs,
                'cdr_in_refug_ref_reg': cdr_in_bio_ref_regs,
                'refug_ref_warm_loss_reg': ref_bio_warm_loss_regs}

            return result_dict

        except Exception as e:
            print(f'Unsuccessful for file {input_tif}: {e}')
            return {
                'scenario': file_scenario,
                'mitigation_option': mitigation_option,
                'year': file_year,
                'refug_ref': float('nan'),
                'cdr_in_refug_ref': float('nan'),
                'refug_ref_warm_loss': float('nan'),
                'refug_ref_reg': [float('nan')] * 5,
                'cdr_in_refug_ref_reg': [float('nan')] * 5,
                'refug_ref_warm_loss_reg': [float('nan')] * 5}

    area_df = pd.DataFrame.from_records(lookup_sub_yrs.apply(process_row,
                                                             axis=1).values)
    area_df = area_df.reset_index(drop=True)

    cz_columns = pd.DataFrame(area_df['refug_ref_reg'].to_list(),
                              columns=['refug_ref_cz1',
                                       'refug_ref_cz2',
                                       'refug_ref_cz3',
                                       'refug_ref_cz4',
                                       'refug_ref_cz5'])
    area_df = pd.concat([area_df.drop(columns='refug_ref_reg'),
                         cz_columns], axis=1)

    cz_columns = pd.DataFrame(area_df['cdr_in_refug_ref_reg'].to_list(),
                              columns=['cdr_in_refug_ref_cz1',
                                       'cdr_in_refug_ref_cz2',
                                       'cdr_in_refug_ref_cz3',
                                       'cdr_in_refug_ref_cz4',
                                       'cdr_in_refug_ref_cz5'])
    area_df = pd.concat([area_df.drop(columns='cdr_in_refug_ref_reg'),
                         cz_columns], axis=1)

    cz_columns = pd.DataFrame(area_df['refug_ref_warm_loss_reg'].to_list(),
                              columns=['refug_ref_warm_loss_cz1',
                                       'refug_ref_warm_loss_cz2',
                                       'refug_ref_warm_loss_cz3',
                                       'refug_ref_warm_loss_cz4',
                                       'refug_ref_warm_loss_cz5'])
    area_df = pd.concat([area_df.drop(columns='refug_ref_warm_loss_reg'),
                         cz_columns], axis=1)

    # preprocess df for plotting
    area_df = area_df.groupby(['scenario',
                               'year',
                               'refug_ref',
                               'refug_ref_cz1',
                               'refug_ref_cz2',
                               'refug_ref_cz3',
                               'refug_ref_cz4',
                               'refug_ref_cz5',
                               'refug_ref_warm_loss',
                               'refug_ref_warm_loss_cz1',
                               'refug_ref_warm_loss_cz2',
                               'refug_ref_warm_loss_cz3',
                               'refug_ref_warm_loss_cz4',
                               'refug_ref_warm_loss_cz5'])[['cdr_in_refug_ref',
                                                            'cdr_in_refug_ref_cz1',
                                                            'cdr_in_refug_ref_cz2',
                                                            'cdr_in_refug_ref_cz3',
                                                            'cdr_in_refug_ref_cz4',
                                                            'cdr_in_refug_ref_cz5']].sum()
    area_df.reset_index(inplace=True)

    area_df['warm_loss_perc'] = area_df['refug_ref_warm_loss'] / area_df['refug_ref'] * 100
    for i in range(1, 6):  # calculate warm loss percentages for all climate zones
        area_df[f'warm_loss_perc_cz{i}'] = area_df[f'refug_ref_warm_loss_cz{i}'] / area_df[f'refug_ref_cz{i}'] * 100

    area_df['land_loss_perc'] = area_df['cdr_in_refug_ref'] / area_df['refug_ref'] * 100
    for i in range(1, 6):  # calculate land loss percentages for all climate zones
        area_df[f'land_loss_perc_cz{i}'] = area_df[f'cdr_in_refug_ref_cz{i}'] / area_df[f'refug_ref_cz{i}'] * 100

    area_df['SSP'] = area_df['scenario'].str.split('-').str[0]
    area_df['RCP'] = area_df['scenario'].str.split('-').str[1]
    area_df.rename(columns={'year': 'Year'}, inplace=True)
    area_df['Model'] = model

    # save for later use
    area_df.to_csv(path / f'{model}_area_df_clim_zone_temp_decline_{temperature_decline}.csv', index=False)

    end = time()
    print(f'Runtime {(end - start) /60} min')

# %% plot warming versus land use change impact on refugia

paths = {'AIM': path_aim, 'GCAM': path_gcam, 'GLOBIOM': path_globiom, 'IMAGE': path_image}
area_df = load_and_concat('area_df_clim_zone_not_allowed', paths)  # choose between allowed and not allowed

area_df = area_df.query('SSP == "SSP2"').reset_index(drop=True)

area_df['RCP'] = area_df['RCP'].astype(str)
rcp_palette = {'19': '#00adcf', '26': '#173c66', '34': '#f79320',
               '45': '#e71d24', '60': '#951b1d', 'Baseline': 'dimgrey'}

fig, axes = plt.subplots(4, 6, figsize=(12, 6), sharex=True, sharey=True)

models = ['AIM', 'GCAM', 'GLOBIOM', 'IMAGE']
cz_columns = [('land_loss_perc', 'warm_loss_perc')] + [
    (f'land_loss_perc_cz{i}', f'warm_loss_perc_cz{i}') for i in range(1, 6)]

for col_idx, (x_col, y_col) in enumerate(cz_columns):
    for row_idx, model in enumerate(models):
        ax = axes[row_idx, col_idx]

        sns.lineplot(data=area_df.query(f'Model == "{model}"'), x=x_col, y=y_col,
                     hue='RCP', palette=rcp_palette, legend=False, ax=ax)

        sns.scatterplot(data=area_df.query(f'Model == "{model}"'), x=x_col,
                        y=y_col, hue='RCP', palette=rcp_palette, style='Year',
                        s=80, legend=(row_idx == 0 and col_idx == 0), ax=ax)

for i in range(4):
    for j in range(6):
        axes[i, j].plot([0, 30], [0, 30], linestyle='--', color='grey')

axes[0, 0].legend(bbox_to_anchor=(-0.05, 2.45), loc='upper left', ncols=12,
                  columnspacing=0.8, handletextpad=0.1)

axes[0, 0].set_title('Global')
axes[0, 1].set_title('Tropical')
axes[0, 2].set_title('Arid')
axes[0, 3].set_title('Temperate')
axes[0, 4].set_title('Cold')
axes[0, 5].set_title('Polar')

#plt.xlim(0, 30)
plt.ylim(-5, 100)

axes[3, 0].set_xlabel('')
axes[3, 1].set_xlabel('')
axes[3, 2].set_xlabel('')
axes[3, 3].set_xlabel('')
axes[3, 4].set_xlabel('')
axes[3, 5].set_xlabel('')

axes[0, 0].set_ylabel('AIM')
axes[1, 0].set_ylabel('GCAM')
axes[2, 0].set_ylabel('GLOBIOM')
axes[3, 0].set_ylabel('IMAGE')

fig.supxlabel("Today's refugia lost to afforestation & bioenergy plantations (combined effect assuming all negative) [%]",
              x=0.51, y=0.05)
fig.supylabel("Today's refugia lost to global warming [%]", x=0.05)

plt.subplots_adjust(hspace=0.15)
plt.subplots_adjust(wspace=0.15)
sns.despine()
plt.show()
