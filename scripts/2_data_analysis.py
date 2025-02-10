
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
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
temperature_decline = 'allowed'  # options: 'allowed' or 'not_allowed'

if temperature_decline == 'allowed':
    warm_file = ar6_data.copy()
    recovery = '(Full recovery)'
elif temperature_decline == 'not_allowed':
    warm_file = ar6_data_stab.copy()
    recovery = '(No recovery)'

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

cz4 = cz4.rio.reproject_match(cz5)  # ensure consistent match
cz45 = cz4 + cz5  # combine cold and polar zones to one
cz45.rio.to_raster(path_cz / 'clim_zon_class45.tif', driver='GTiff')
cz4 = rioxarray.open_rasterio(path_cz / 'clim_zon_class45.tif', masked=True)

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

        # STEP1: load files for LUC, refugia, and baseline refugia
        land_use = rioxarray.open_rasterio(filepath / f'{lu_model}_{input_tif}',
                                           masked=True)  # mask nan values for calc

        bio_file = ''.join(bio_select[(bio_select['Model'] == lu_model) &
                                           (bio_select['Scenario'] == file_scenario)][file_year])

        refugia = rioxarray.open_rasterio(path_uea / bio_file,
                                          masked=True)  # mask nan values for calc

        refugia_ref = rioxarray.open_rasterio(path_uea / biodiv_ref_warm_file,
                                              masked=True)

        # align files
        refugia = refugia.rio.reproject_match(land_use)
        refugia_ref = refugia_ref.rio.reproject_match(land_use)

        # calculate warming and land impact on reference refugia (global)
        luc_in_bio_ref = land_use * refugia_ref
        ref_bio_warm_loss = refugia_ref - refugia
        ref_bio_warm_loss.rio.to_raster(path_uea / 'ref_bio_warm_loss_temp.tif',
                                        driver='GTiff')

        # calculate refugia extents
        ref_bio_warm_loss_a = land_area_calculation(path_uea,
                                                    'ref_bio_warm_loss_temp.tif')
        refugia_ref_a = land_area_calculation(path_uea, biodiv_ref_warm_file)

        # calculate aggregated area "losses" and baseline refugia
        luc_in_bio_ref_agg = pos_val_summer(luc_in_bio_ref, squeeze=True)
        ref_bio_warm_loss_agg = pos_val_summer(ref_bio_warm_loss_a, squeeze=True)
        refugia_ref_agg = pos_val_summer(refugia_ref_a, squeeze=True)

        # calculate warming and land impact on reference refugia (regional)
        luc_in_bio_ref_regs = []
        ref_bio_warm_loss_regs = []
        refugia_ref_regs = []

        cz_values = [cz1, cz2, cz3, cz4]

        for cz in cz_values:
            cz_m = cz.rio.reproject_match(land_use)  # ensure consistent match
            luc_in_bio_ref_cz = (land_use * refugia_ref) * cz_m

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

            luc_in_bio_ref_cz = pos_val_summer(luc_in_bio_ref_cz, squeeze=True)
            ref_bio_warm_loss_cz = pos_val_summer(ref_bio_warm_loss_cz, squeeze=True)
            refugia_ref_cz = pos_val_summer(refugia_ref_cz, squeeze=True)

            # calculate regional area "losses" and baseline refugia
            luc_in_bio_ref_regs.append(luc_in_bio_ref_cz)
            ref_bio_warm_loss_regs.append(ref_bio_warm_loss_cz)
            refugia_ref_regs.append(refugia_ref_cz)

        return luc_in_bio_ref_agg, ref_bio_warm_loss_agg, refugia_ref_agg, \
            luc_in_bio_ref_regs, ref_bio_warm_loss_regs, refugia_ref_regs

    # use overlay_calculator
    def process_row(row):
        input_tif = row['file_name']
        file_year = row['year']
        file_scenario = row['scenario']
        mitigation_option = row['mitigation_option']

        try:
            # run overlay_calculator for all scenarios to retrieve areas as outputs
            luc_in_bio_ref_agg, ref_bio_warm_loss_agg, refugia_ref_agg, \
                luc_in_bio_ref_regs, ref_bio_warm_loss_regs, refugia_ref_regs = \
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
                'luc_in_refug_ref': luc_in_bio_ref_agg,
                'refug_ref_warm_loss': ref_bio_warm_loss_agg,
                'refug_ref_reg': refugia_ref_regs,
                'luc_in_refug_ref_reg': luc_in_bio_ref_regs,
                'refug_ref_warm_loss_reg': ref_bio_warm_loss_regs}

            return result_dict

        except Exception as e:
            print(f'Unsuccessful for file {input_tif}: {e}')
            return {
                'scenario': file_scenario,
                'mitigation_option': mitigation_option,
                'year': file_year,
                'refug_ref': float('nan'),
                'luc_in_refug_ref': float('nan'),
                'refug_ref_warm_loss': float('nan'),
                'refug_ref_reg': [float('nan')] * 4,
                'luc_in_refug_ref_reg': [float('nan')] * 4,
                'refug_ref_warm_loss_reg': [float('nan')] * 4}

    area_df = pd.DataFrame.from_records(lookup_sub_yrs.apply(process_row,
                                                             axis=1).values)
    area_df = area_df.reset_index(drop=True)

    cz_columns = pd.DataFrame(area_df['refug_ref_reg'].to_list(),
                              columns=['refug_ref_cz1',
                                       'refug_ref_cz2',
                                       'refug_ref_cz3',
                                       'refug_ref_cz4'])
    area_df = pd.concat([area_df.drop(columns='refug_ref_reg'),
                         cz_columns], axis=1)

    cz_columns = pd.DataFrame(area_df['luc_in_refug_ref_reg'].to_list(),
                              columns=['luc_in_refug_ref_cz1',
                                       'luc_in_refug_ref_cz2',
                                       'luc_in_refug_ref_cz3',
                                       'luc_in_refug_ref_cz4'])
    area_df = pd.concat([area_df.drop(columns='luc_in_refug_ref_reg'),
                         cz_columns], axis=1)

    cz_columns = pd.DataFrame(area_df['refug_ref_warm_loss_reg'].to_list(),
                              columns=['refug_ref_warm_loss_cz1',
                                       'refug_ref_warm_loss_cz2',
                                       'refug_ref_warm_loss_cz3',
                                       'refug_ref_warm_loss_cz4'])
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
                               'refug_ref_warm_loss',
                               'refug_ref_warm_loss_cz1',
                               'refug_ref_warm_loss_cz2',
                               'refug_ref_warm_loss_cz3',
                               'refug_ref_warm_loss_cz4'])[['luc_in_refug_ref',
                                                            'luc_in_refug_ref_cz1',
                                                            'luc_in_refug_ref_cz2',
                                                            'luc_in_refug_ref_cz3',
                                                            'luc_in_refug_ref_cz4']].sum()
    area_df.reset_index(inplace=True)

    area_df['warm_loss_perc'] = area_df['refug_ref_warm_loss'] / area_df['refug_ref'] * 100
    for i in range(1, 5):  # calculate warm loss percentages for all climate zones
        area_df[f'warm_loss_perc_cz{i}'] = area_df[f'refug_ref_warm_loss_cz{i}'] / area_df[f'refug_ref_cz{i}'] * 100

    area_df['land_loss_perc'] = area_df['luc_in_refug_ref'] / area_df['refug_ref'] * 100
    for i in range(1, 5):  # calculate land loss percentages for all climate zones
        area_df[f'land_loss_perc_cz{i}'] = area_df[f'luc_in_refug_ref_cz{i}'] / area_df[f'refug_ref_cz{i}'] * 100

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
decline_df = load_and_concat('area_df_clim_zone_temp_decline_allowed', paths)
decline_df['Decline'] = 'True'
nodecline_df = load_and_concat('area_df_clim_zone_temp_decline_not_allowed', paths)
nodecline_df['Decline'] = 'False'

area_df = nodecline_df.copy()  # choose between decline_df and nodecline_df

rcps = ['19', '26', '34', '45']  # specify RCPs that shall be plotted
area_df['RCP'] = area_df['RCP'].astype(str)
area_df = area_df.loc[area_df['RCP'].isin(rcps)]
area_df = area_df.loc[area_df['SSP'].isin(['SSP2'])]

rcp_palette = {'19': '#00adcf', '26': '#173c66', '34': '#f79320',
               '45': '#e71d24', '60': '#951b1d', 'Baseline': 'dimgrey'}

fig, axes = plt.subplots(4, 4, figsize=(11, 6), sharex=True, sharey=True)

models = ['AIM', 'GCAM', 'GLOBIOM', 'IMAGE']
cz_columns = [(f'land_loss_perc_cz{i}', f'warm_loss_perc_cz{i}') for i in range(1, 5)]

for col_idx, (x_col, y_col) in enumerate(cz_columns):
    for row_idx, model in enumerate(models):
        ax = axes[row_idx, col_idx]

        sns.lineplot(data=area_df.query(f'Model == "{model}"'), x=x_col, y=y_col,
                     hue='RCP', palette=rcp_palette, legend=False, ax=ax)

        sns.scatterplot(data=area_df.query(f'Model == "{model}"'), x=x_col,
                        y=y_col, hue='RCP', palette=rcp_palette, style='Year',
                        s=100, alpha=0.7, legend=(row_idx == 0 and col_idx == 0), ax=ax)

for i in range(4):
    for j in range(4):
        axes[i, j].plot([0, 30], [0, 30], linestyle='--', color='grey')

axes[0, 0].legend(bbox_to_anchor=(-0.05, 1.7), loc='upper left', ncols=12,
                  columnspacing=0.8, handletextpad=0.1)

axes[0, 0].set_title('Tropical')
axes[0, 1].set_title('Arid')
axes[0, 2].set_title('Temperate')
axes[0, 3].set_title('Cold & Polar')

plt.xlim(-1, 26)
plt.ylim(-5, 100)
plt.xticks([0, 13, 26])

axes[3, 0].set_xlabel('')
axes[3, 1].set_xlabel('')
axes[3, 2].set_xlabel('')
axes[3, 3].set_xlabel('')

axes[0, 0].set_ylabel('AIM')
axes[1, 0].set_ylabel('GCAM')
axes[2, 0].set_ylabel('GLOBIOM')
axes[3, 0].set_ylabel('IMAGE')

fig.supxlabel("Today's refugia lost to afforestation & bioenergy plantations (combined effect assuming all negative) [%]",
              x=0.51, y=0.03)
fig.supylabel("Today's refugia lost to global warming [%]", x=0.054)

plt.subplots_adjust(hspace=0.15)
plt.subplots_adjust(wspace=0.15)
sns.despine()
plt.show()

# %% plot global values across models and recovery assumptions
area_df = pd.concat([decline_df, nodecline_df])

rcps = ['19', '26', '34', '45']  # specify RCPs that shall be plotted
area_df['RCP'] = area_df['RCP'].astype(str)
area_df = area_df.loc[area_df['RCP'].isin(rcps)]
area_df = area_df.loc[area_df['SSP'].isin(['SSP2'])]
decline_conditions = ['False', 'True']
decline_labels = ['No recovery', 'Full recovery']

fig, axes = plt.subplots(2, 4, figsize=(11, 6), sharex=True, sharey=True)

for i, decline in enumerate(decline_conditions):
    for j, model in enumerate(models):

        data = area_df.query(f'Model == "{model}" & Decline == "{decline}"')

        sns.lineplot(data=data, x='land_loss_perc', y='warm_loss_perc', hue='RCP',
                     palette=rcp_palette, legend=False, ax=axes[i, j])
        sns.scatterplot(data=data, x='land_loss_perc', y='warm_loss_perc', hue='RCP',
                        palette=rcp_palette, style='Year', s=100, alpha=0.7,
                        legend=(i == 0 and j == 0), ax=axes[i, j])

        axes[i, j].plot([0, 30], [0, 30], linestyle='--', color='grey')

        if i == 0:
            axes[i, j].set_title(model)
        if j == 0:
            axes[i, j].set_ylabel(decline_labels[i])
        if i == 1:
            axes[i, j].set_xlabel('')

axes[0, 0].legend(bbox_to_anchor=(-0.05, 1.29), loc='upper left', ncols=12,
                  columnspacing=0.8, handletextpad=0.1)

plt.xlim(-1, 20)
plt.ylim(-5, 70)
plt.xticks([0, 5, 10, 15, 20])
plt.yticks([0, 14, 28, 42, 56, 70])

fig.supxlabel("Today's refugia 'lost' to afforestation & bioenergy plantations (combined effect assuming all negative) [%]",
              x=0.51, y=0.03)
fig.supylabel("Today's refugia lost to global warming [%]", x=0.054)

plt.subplots_adjust(hspace=0.15, wspace=0.15)
sns.despine()
plt.show()

# %% spatially-explicit estimation of warming vs luc in SSP2-26
sf_path = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/wab')
admin_sf = shapefile.Reader(sf_path / 'world-administrative-boundaries.shp')

def warm_vs_luc_plotter(filepath,  # filepath input file
                        file_year,  # year of input file (string)
                        file_scenario,  # input file SSP-RCP scenario (string)
                        biodiv_ref_warm_file,  # Bio file for ref warm (1.3C)
                        lu_model):  # AIM, GCAM, GLOBIOM or IMAGE

    # STEP1: load files for LUC, refugia, and baseline refugia
    ar_file = rioxarray.open_rasterio(filepath / f'{lu_model}_Afforestation_{file_scenario}_{file_year}.tif',
                                      masked=True)  # mask nan values for calc
    be_file = rioxarray.open_rasterio(filepath / f'{lu_model}_Bioenergy_{file_scenario}_{file_year}.tif',
                                      masked=True)  # mask nan values for calc

    bio_file = ''.join(bio_select[(bio_select['Model'] == lu_model) &
                                  (bio_select['Scenario'] == file_scenario)][file_year])

    refugia = rioxarray.open_rasterio(path_uea / bio_file,
                                      masked=True)  # mask nan values for calc

    refugia_ref = rioxarray.open_rasterio(path_uea / biodiv_ref_warm_file,
                                          masked=True)

    # align files
    be_file = be_file.rio.reproject_match(ar_file)
    luc_file = ar_file + be_file

    refugia = refugia.rio.reproject_match(luc_file)
    refugia_ref = refugia_ref.rio.reproject_match(luc_file)

    # calculate warming and land impact on reference refugia
    luc_in_bio_ref = luc_file * refugia_ref
    luc_in_bio_ref.rio.to_raster(filepath / 'luc_in_bio_ref.tif')

    ref_bio_warm_loss = refugia_ref - refugia
    ref_bio_warm_loss.rio.to_raster(path_uea / 'ref_bio_warm_loss.tif')
    land_area_calculation(path_uea, 'ref_bio_warm_loss.tif', 'ref_bio_warm_loss_a.tif')

    # calculate warming and luc loss per country
    warm_loss = rs.open(path_uea / 'ref_bio_warm_loss_a.tif', masked=True)
    luc_loss = rs.open(filepath / 'luc_in_bio_ref.tif', masked=True)
    warm_df = admin_bound_calculator('warm_loss', admin_sf, warm_loss)
    luc_df = admin_bound_calculator('luc_loss', admin_sf, luc_loss)

    loss_df = pd.merge(warm_df, luc_df, on='iso3', how='outer',
                       suffixes=['_warm', '_luc'])

    # asign values 1 = warm > luc; 2 = warm < luc; 3 = warm == luc
    loss_df['impact'] = np.where(loss_df['km2_warm'] >
                                 loss_df['km2_luc'], 1,
                                 np.where(loss_df['km2_warm'] <
                                          loss_df['km2_luc'], 2, 3))

    # plot predominance of warming vs luc impact per country
    cmap = {'1': 'crimson', '2': 'mediumblue', '3': 'grey'}
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.Robinson()})

    shape_records = list(Reader(sf_path / 'world-administrative-boundaries.shp').records())

    # plot each country with data
    for record in shape_records:
        country_iso = record.attributes['iso3']
        if country_iso in loss_df['iso3'].values:
            value = loss_df[loss_df['iso3'] == country_iso]['impact'].values[0]
            color = cmap[str(value)]
            geom = record.geometry
            ax.add_geometries([geom], ccrs.PlateCarree(), facecolor=color,
                              edgecolor='black', linewidth=0.2)

    ax.coastlines(linewidth=0.2)

    legend_patches = [
        mpatches.Patch(color='crimson', label='More warming-related loss'),
        mpatches.Patch(color='mediumblue', label='More LUC-related loss'),
        mpatches.Patch(color='grey', label='No or even loss')]

    ax.legend(bbox_to_anchor=(0.175, -0.1), handles=legend_patches, ncols=4,
              loc='lower left', fontsize=8.5, columnspacing=0.8, handletextpad=0.5,
              frameon=True)

    plt.title(f'{lu_model} {file_scenario} {file_year} \n{recovery}',
              fontsize=11, x=0.05, y=0.27, ha='left')
    plt.show()

for model in models:
    if model == 'GLOBIOM':
        path = path_globiom
    elif model == 'AIM':
        path = path_aim
    elif model == 'IMAGE':
        path = path_image
    elif model == 'GCAM':
        path = path_gcam

    warm_vs_luc_plotter(path,
                        '2100',  # adust if required
                        'SSP2-26',  # adust if required
                        'bio1.3_bin.tif',  # adust if required
                        model)

# %% comparison of refugia impact at 1.5C before and after overshoot


