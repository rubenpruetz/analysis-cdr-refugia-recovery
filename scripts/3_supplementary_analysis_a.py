
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
from shapely.geometry import shape
import cartopy.feature as cfeature
from matplotlib.patches import Patch
from required_functions import *
import shapefile
from pathlib import Path
plt.rcParams.update({'figure.dpi': 600})

path_project = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity')
path_uea = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/uea_maps/UEA_20km')
path_aim = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps')
path_gcam = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/gcam_maps')
path_globiom = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps')
path_image = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/image_maps')
path_magpie = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/magpie_maps')
path_ar6_data = Path('/Users/rpruetz/Documents/phd/datasets')
ath_ref_pot = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/reforest_potential')
path_bioeng_pot = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/Braun_et_al_2024_PB_BECCS/Results/1_source_data_figures/Fig2')

ar6_db = pd.read_csv(path_ar6_data / 'AR6_Scenarios_Database_World_v1.1.csv')
lookup_mi_luc_df = pd.read_csv(path_project / 'lookup_table_ar_bioenergy_files_all_models.csv')
lookup_mi_luc_df['year'] = lookup_mi_luc_df['year'].astype(str)

# %% get upper bound likely warming (p83.3) for SSP-RCP combinations
all_years = [str(year) for year in range(2020, 2101)]

models = ['MESSAGE-GLOBIOM 1.0', 'AIM/CGE 2.0', 'GCAM 4.2', 'IMAGE 3.0.1',
          'REMIND-MAgPIE 1.5']
scenarios = ['SSP1-19', 'SSP1-26', 'SSP1-34', 'SSP1-45', 'SSP2-19', 'SSP2-26',
             'SSP2-34', 'SSP2-45', 'SSP3-34', 'SSP3-45']
variable = ['AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|83.3th Percentile']

ar6_data = ar6_db.loc[ar6_db['Variable'].isin(variable)]
ar6_data = ar6_data.loc[ar6_data['Model'].isin(models)]
ar6_data = ar6_data.loc[ar6_data['Scenario'].isin(scenarios)]
ar6_data = ar6_data.round(2)  # round temperatures

# allow no temperature decline by calculating peak warming up until each year
for year in range(2021, 2101):
    cols_til_year = ar6_data.loc[:, '2020':str(year)]
    ar6_data[f'{year}_max'] = cols_til_year.max(axis=1)

cols = ['Model', 'Scenario', '2020'] + [f'{year}_max' for year in range(2021, 2101)]
ar6_data_stab = ar6_data[cols]
ar6_data_stab = ar6_data_stab.rename(columns={f'{year}_max': str(year) for year in all_years})

ar6_data = ar6_data[['Model', 'Scenario'] + all_years].copy()

# %% choose between biodiv recovery or no recovery after peak warming
temperature_declines = ['allowed', 'not_allowed']
models_ab = ['AIM', 'GCAM', 'GLOBIOM', 'IMAGE', 'MAgPIE']

for temperature_decline in temperature_declines:
    if temperature_decline == 'allowed':
        warm_file = ar6_data.copy()
        recovery = 'Full recovery'
    elif temperature_decline == 'not_allowed':
        warm_file = ar6_data_stab.copy()
        recovery = 'No recovery'

    bio_select = warm_file.set_index(['Model', 'Scenario'])
    bio_select = 'bio' + \
        bio_select.select_dtypes(include=np.number).astype(str) + '_bin.tif'
    bio_select.reset_index(inplace=True)

    # rename models for the subsequent step
    bio_select.replace({'Model': {'AIM/CGE 2.0': 'AIM',
                                  'MESSAGE-GLOBIOM 1.0': 'GLOBIOM',
                                  'GCAM 4.2': 'GCAM',
                                  'IMAGE 3.0.1': 'IMAGE',
                                  'REMIND-MAgPIE 1.5': 'MAgPIE'}}, inplace=True)

    # specify years for the analysis
    years = ['2030', '2040', '2050', '2060', '2070', '2080', '2090', '2100']
    lookup_sub_yrs = lookup_mi_luc_df.copy()
    lookup_sub_yrs = lookup_sub_yrs.loc[lookup_sub_yrs['year'].isin(years)]

    for model in models_ab:
        if model == 'GLOBIOM':
            path = path_globiom
        elif model == 'AIM':
            path = path_aim
        elif model == 'IMAGE':
            path = path_image
        elif model == 'GCAM':
            path = path_gcam
        elif model == 'MAgPIE':
            path = path_magpie

        start = time()  # runtime monitoring

        # use overlay_calculator
        def process_row(row):
            input_tif = row['file_name']
            file_year = row['year']
            file_scenario = row['scenario']
            mitigation_option = row['mitigation_option']

            try:
                # run overlay_calculator for all scenarios to retrieve areas as outputs
                luc_in_bio_ref_agg, ref_bio_warm_loss_agg, refugia_ref_agg, luc_in_bio_agg = \
                    overlay_calculator(input_tif,
                                       path,
                                       file_year,
                                       bio_select,
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
                    'luc_in_refug': luc_in_bio_agg}

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
                    'luc_in_refug': float('nan')}

        area_df = pd.DataFrame.from_records(lookup_sub_yrs.apply(process_row,
                                                                 axis=1).values)
        area_df = area_df.reset_index(drop=True)

        # preprocess df for plotting
        area_df = area_df.groupby(['scenario',
                                   'year',
                                   'refug_ref',
                                   'refug_ref_warm_loss'])[['luc_in_refug_ref', 'luc_in_refug']].sum()
        area_df.reset_index(inplace=True)

        # set LUC in refugia to LUC in today's refugia where warming goes below 1.3 °C
        area_df['luc_in_refug'] = area_df['luc_in_refug_ref'].where(area_df['luc_in_refug'] >
                                                                    area_df['luc_in_refug_ref'],
                                                                    area_df['luc_in_refug'])

        area_df['warm_loss_perc'] = area_df['refug_ref_warm_loss'] / area_df['refug_ref'] * 100
        area_df['luc_loss_perc'] = area_df['luc_in_refug_ref'] / area_df['refug_ref'] * 100
        area_df['total_loss'] = area_df['refug_ref_warm_loss'] + area_df['luc_in_refug']
        area_df['total_loss_perc'] = area_df['total_loss'] / area_df['refug_ref'] * 100

        area_df['SSP'] = area_df['scenario'].str.split('-').str[0]
        area_df['RCP'] = area_df['scenario'].str.split('-').str[1]
        area_df.rename(columns={'year': 'Year'}, inplace=True)
        area_df['Model'] = model

        # save for later use
        area_df.to_csv(path / f'{model}_area_df_p83.3_temp_decline_{temperature_decline}.csv', index=False)

        end = time()
        print(f'Runtime {(end - start) /60} min')

# %% plot warming (p83.3) versus land use change impact on refugia
# note: previous code needs to run first in BOTH modes: 'allowed' & 'not_allowed'
paths = {'AIM': path_aim, 'GCAM': path_gcam, 'GLOBIOM': path_globiom,
         'IMAGE': path_image, 'MAgPIE': path_magpie}
decline_df = load_and_concat('area_df_p83.3_temp_decline_allowed', paths)
decline_df['Decline'] = 'True'
nodecline_df = load_and_concat('area_df_p83.3_temp_decline_not_allowed', paths)
nodecline_df['Decline'] = 'False'
output_1 = pd.concat([decline_df, nodecline_df])
plot_df = output_1.copy()

rcps = ['19', '26', '34', '45']  # specify RCPs that shall be plotted
years = [2030, 2050, 2080, 2100]  # specify years that shall be plotted
plot_df['RCP'] = plot_df['RCP'].astype(str)
plot_df = plot_df.loc[plot_df['RCP'].isin(rcps)]
plot_df = plot_df.loc[plot_df['Year'].isin(years)]
plot_df = plot_df.loc[plot_df['SSP'].isin(['SSP2'])]
decline_conditions = ['False', 'True']
decline_labels = ['No recovery', 'Full recovery']

rcp_palette = {'19': '#00adcf', '26': '#173c66', '34': '#f79320', '45': '#e71d24'}

fig, axes = plt.subplots(2, 5, figsize=(9, 6), sharex=True, sharey=True)

for i, decline in enumerate(decline_conditions):
    for j, model in enumerate(models_ab):

        data = plot_df.query(f'Model == "{model}" & Decline == "{decline}"')

        sns.lineplot(data=data, x='luc_loss_perc', y='warm_loss_perc', hue='RCP', sort=False,
                     palette=rcp_palette, legend=False, ax=axes[i, j])
        sns.scatterplot(data=data, x='luc_loss_perc', y='warm_loss_perc', hue='RCP',
                        palette=rcp_palette, style='Year', s=100, alpha=0.7,
                        legend=(i == 0 and j == 0), ax=axes[i, j])

        axes[i, j].plot([0, 30], [0, 30], linestyle='--', color='grey')

        if i == 0:
            if model == 'MAgPIE':
                axes[i, j].set_title('REMIND-MAgPIE')
            else: axes[i, j].set_title(model)
        if j == 0:
            axes[i, j].set_ylabel(decline_labels[i], fontsize=12)
        if i == 1:
            axes[i, j].set_xlabel('')

axes[0, 0].legend(bbox_to_anchor=(-0.35, 1.32), loc='upper left', ncols=12,
                  columnspacing=0, handletextpad=0, fontsize=12)

plt.xlim(-1, 21)
plt.ylim(-5, 75)
plt.xticks([0, 7, 14, 21])
plt.yticks([0, 16, 32, 48, 64, 80])

for ax_row in axes:
    for ax in ax_row:
        ax.tick_params(axis='both', labelsize=12)

fig.supxlabel("Today's refugia 'lost' to forestation & bioenergy plantations\n(combined effect assuming all negative) [%]",
              x=0.51, y=-0.025, fontsize=14)
fig.supylabel("Today's refugia lost to 83.3rd percentile warming [%]", x=0.033, fontsize=14)

plt.subplots_adjust(hspace=0.15, wspace=0.19)
sns.despine()
plt.show()

# %% plot combined refugia loss from global warming (p83.3) and mitigation
rcps_float = [float(r) for r in rcps]
rcp_palette = {19: '#00adcf', 26: '#173c66', 34: '#f79320', 45: '#e71d24'}

plot_df2 = output_1.loc[output_1['RCP'].isin(rcps_float)]
plot_norecover = plot_df2.query('Decline == "False"').reset_index()
plot_recover = plot_df2.query('Decline == "True"').reset_index()

plt.figure(figsize=(1.2, 5.2))
sns.lineplot(data=plot_norecover, x='Year', y='total_loss_perc', hue='RCP',
             palette=rcp_palette, linestyle='-', errorbar=('pi', 90),
             estimator='median')
sns.lineplot(data=plot_recover, x='Year', y='total_loss_perc', hue='RCP',
             palette=rcp_palette, linestyle='--', errorbar=('pi', 90),
             estimator='median', legend=False)
sns.despine()

plt.xlim(2030, 2100)
plt.ylim(0, 80)
plt.xticks([2030, 2065, 2100])
plt.xlabel('')
plt.ylabel("Today's refugia lost to global warming (83.3rd percentile) and LUC\n(combined effect assuming all negative) [%]")
plt.legend(bbox_to_anchor=(1.19, 1.125), loc='upper right', ncols=4,
           columnspacing=0.8, handletextpad=0.2, handlelength=0.5, fontsize=9.5)
plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.8)
plt.show()









# %% repeat analysis by only considering likely harmful mitigation-related LUC
variable = ['AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile']

ar6_data = ar6_db.loc[ar6_db['Variable'].isin(variable)]
ar6_data = ar6_data.loc[ar6_data['Model'].isin(models)]
ar6_data = ar6_data.loc[ar6_data['Scenario'].isin(scenarios)]
ar6_data = ar6_data.round(2)  # round temperatures

# allow no temperature decline by calculating peak warming up until each year
for year in range(2021, 2101):
    cols_til_year = ar6_data.loc[:, '2020':str(year)]
    ar6_data[f'{year}_max'] = cols_til_year.max(axis=1)

cols = ['Model', 'Scenario', '2020'] + [f'{year}_max' for year in range(2021, 2101)]
ar6_data_stab = ar6_data[cols]
ar6_data_stab = ar6_data_stab.rename(columns={f'{year}_max': str(year) for year in all_years})

ar6_data = ar6_data[['Model', 'Scenario'] + all_years].copy()

# %% choose between biodiv recovery or no recovery after peak warming
for temperature_decline in temperature_declines:
    if temperature_decline == 'allowed':
        warm_file = ar6_data.copy()
        recovery = 'Full recovery'
    elif temperature_decline == 'not_allowed':
        warm_file = ar6_data_stab.copy()
        recovery = 'No recovery'

    bio_select = warm_file.set_index(['Model', 'Scenario'])
    bio_select = 'bio' + \
        bio_select.select_dtypes(include=np.number).astype(str) + '_bin.tif'
    bio_select.reset_index(inplace=True)

    # rename models for the subsequent step
    bio_select.replace({'Model': {'AIM/CGE 2.0': 'AIM',
                                  'MESSAGE-GLOBIOM 1.0': 'GLOBIOM',
                                  'GCAM 4.2': 'GCAM',
                                  'IMAGE 3.0.1': 'IMAGE',
                                  'REMIND-MAgPIE 1.5': 'MAgPIE'}}, inplace=True)

    # specify years for the analysis
    years = ['2030', '2040', '2050', '2060', '2070', '2080', '2090', '2100']
    lookup_sub_yrs = lookup_mi_luc_df.copy()
    lookup_sub_yrs = lookup_sub_yrs.loc[lookup_sub_yrs['year'].isin(years)]

    for model in models_ab:
        if model == 'GLOBIOM':
            path = path_globiom
        elif model == 'AIM':
            path = path_aim
        elif model == 'IMAGE':
            path = path_image
        elif model == 'GCAM':
            path = path_gcam
        elif model == 'MAgPIE':
            path = path_magpie

        start = time()  # runtime monitoring

        # use overlay_calculator_harm
        def process_row(row):
            input_tif = row['file_name']
            file_year = row['year']
            file_scenario = row['scenario']
            mitigation_option = row['mitigation_option']

            try:
                # run overlay_calculator_harm for all scenarios to retrieve areas as outputs
                luc_in_bio_ref_agg, ref_bio_warm_loss_agg, refugia_ref_agg, luc_in_bio_agg = \
                    overlay_calculator_harm(input_tif,
                                            path,
                                            file_year,
                                            bio_select,
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
                    'luc_in_refug': luc_in_bio_agg}

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
                    'luc_in_refug': float('nan')}

        area_df = pd.DataFrame.from_records(lookup_sub_yrs.apply(process_row,
                                                                 axis=1).values)
        area_df = area_df.reset_index(drop=True)

        # preprocess df for plotting
        area_df = area_df.groupby(['scenario',
                                   'year',
                                   'refug_ref',
                                   'refug_ref_warm_loss'])[['luc_in_refug_ref', 'luc_in_refug']].sum()
        area_df.reset_index(inplace=True)

        # set LUC in refugia to LUC in today's refugia where warming goes below 1.3 °C
        area_df['luc_in_refug'] = area_df['luc_in_refug_ref'].where(area_df['luc_in_refug'] >
                                                                    area_df['luc_in_refug_ref'],
                                                                    area_df['luc_in_refug'])

        area_df['warm_loss_perc'] = area_df['refug_ref_warm_loss'] / area_df['refug_ref'] * 100
        area_df['luc_loss_perc'] = area_df['luc_in_refug_ref'] / area_df['refug_ref'] * 100
        area_df['total_loss'] = area_df['refug_ref_warm_loss'] + area_df['luc_in_refug']
        area_df['total_loss_perc'] = area_df['total_loss'] / area_df['refug_ref'] * 100

        area_df['SSP'] = area_df['scenario'].str.split('-').str[0]
        area_df['RCP'] = area_df['scenario'].str.split('-').str[1]
        area_df.rename(columns={'year': 'Year'}, inplace=True)
        area_df['Model'] = model

        # save for later use
        area_df.to_csv(path / f'{model}_area_df_temp_decline_{temperature_decline}_harmful_luc.csv', index=False)

        end = time()
        print(f'Runtime {(end - start) /60} min')

# %% plot warming versus (likely harmful) land use change impact on refugia
# note: previous code needs to run first in BOTH modes: 'allowed' & 'not_allowed'
paths = {'AIM': path_aim, 'GCAM': path_gcam, 'GLOBIOM': path_globiom,
         'IMAGE': path_image, 'MAgPIE': path_magpie}
decline_df = load_and_concat('area_df_temp_decline_allowed_harmful_luc', paths)
decline_df['Decline'] = 'True'
nodecline_df = load_and_concat('area_df_temp_decline_not_allowed_harmful_luc', paths)
nodecline_df['Decline'] = 'False'
output_1 = pd.concat([decline_df, nodecline_df])
plot_df = output_1.copy()

rcps = ['19', '26', '34', '45']  # specify RCPs that shall be plotted
years = [2030, 2050, 2080, 2100]  # specify years that shall be plotted
plot_df['RCP'] = plot_df['RCP'].astype(str)
plot_df = plot_df.loc[plot_df['RCP'].isin(rcps)]
plot_df = plot_df.loc[plot_df['Year'].isin(years)]
plot_df = plot_df.loc[plot_df['SSP'].isin(['SSP2'])]
decline_conditions = ['False', 'True']
decline_labels = ['No recovery', 'Full recovery']

rcp_palette = {'19': '#00adcf', '26': '#173c66', '34': '#f79320', '45': '#e71d24'}

fig, axes = plt.subplots(2, 5, figsize=(9, 6), sharex=True, sharey=True)

for i, decline in enumerate(decline_conditions):
    for j, model in enumerate(models_ab):

        data = plot_df.query(f'Model == "{model}" & Decline == "{decline}"')

        sns.lineplot(data=data, x='luc_loss_perc', y='warm_loss_perc', hue='RCP', sort=False,
                     palette=rcp_palette, legend=False, ax=axes[i, j])
        sns.scatterplot(data=data, x='luc_loss_perc', y='warm_loss_perc', hue='RCP',
                        palette=rcp_palette, style='Year', s=100, alpha=0.7,
                        legend=(i == 0 and j == 0), ax=axes[i, j])

        axes[i, j].plot([0, 30], [0, 30], linestyle='--', color='grey')

        if i == 0:
            if model == 'MAgPIE':
                axes[i, j].set_title('REMIND-MAgPIE')
            else: axes[i, j].set_title(model)
        if j == 0:
            axes[i, j].set_ylabel(decline_labels[i], fontsize=12)
        if i == 1:
            axes[i, j].set_xlabel('')

axes[0, 0].legend(bbox_to_anchor=(-0.35, 1.32), loc='upper left', ncols=12,
                  columnspacing=0, handletextpad=0, fontsize=12)

plt.xlim(-1, 21)
plt.ylim(-5, 70)
plt.xticks([0, 7, 14, 21])
plt.yticks([0, 14, 28, 42, 57, 70])

for ax_row in axes:
    for ax in ax_row:
        ax.tick_params(axis='both', labelsize=12)

fig.supxlabel("Today's refugia 'lost' to forestation & bioenergy plantations\n(combined effect in likely harmed areas) [%]",
              x=0.51, y=-0.025, fontsize=14)
fig.supylabel("Today's refugia lost to gobal warming [%]", x=0.033, fontsize=14)

plt.subplots_adjust(hspace=0.15, wspace=0.19)
sns.despine()
plt.show()

# %% plot combined refugia loss from global warming and (likely harmful) mitigation
rcps_float = [float(r) for r in rcps]
rcp_palette = {19: '#00adcf', 26: '#173c66', 34: '#f79320', 45: '#e71d24'}

plot_df2 = output_1.loc[output_1['RCP'].isin(rcps_float)]
plot_norecover = plot_df2.query('Decline == "False"').reset_index()
plot_recover = plot_df2.query('Decline == "True"').reset_index()

plt.figure(figsize=(1.2, 5.2))
sns.lineplot(data=plot_norecover, x='Year', y='total_loss_perc', hue='RCP',
             palette=rcp_palette, linestyle='-', errorbar=('pi', 90),
             estimator='median')
sns.lineplot(data=plot_recover, x='Year', y='total_loss_perc', hue='RCP',
             palette=rcp_palette, linestyle='--', errorbar=('pi', 90),
             estimator='median', legend=False)
sns.despine()

plt.xlim(2030, 2100)
plt.ylim(0, 80)
plt.xticks([2030, 2065, 2100])
plt.xlabel('')
plt.ylabel("Today's refugia lost to global warming and likely harmful LUC\n(combined effect assuming all negative) [%]")
plt.legend(bbox_to_anchor=(1.19, 1.125), loc='upper right', ncols=4,
           columnspacing=0.8, handletextpad=0.2, handlelength=0.5, fontsize=9.5)
plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.8)
plt.show()

# %% country-level agreement of warming vs (likely harmful) LUC in SSP2-26 2100
sf_path = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/wab')
admin_sf = shapefile.Reader(sf_path / 'world-administrative-boundaries.shp')

# adjust if necessary
thresholds = [5]  # use lower bound thresholds to exclude very low effects
file_scenario = 'SSP2-26'
file_year = '2100'

# use warm_vs_luc_plotter to determine predominant effect at country level
for thres in thresholds:

    loss_dfs = []
    for model in models_ab:
        if model == 'GLOBIOM':
            path = path_globiom
        elif model == 'AIM':
            path = path_aim
        elif model == 'IMAGE':
            path = path_image
        elif model == 'GCAM':
            path = path_gcam
        elif model == 'MAgPIE':
            path = path_magpie

        loss_df = warm_vs_luc_plotter_harm(path,
                                           file_year,
                                           admin_sf,
                                           bio_select,
                                           file_scenario,
                                           'bio1.3_bin.tif',  # adjust if required
                                           model)
        loss_dfs.append(loss_df)

    loss_dfs = pd.concat(loss_dfs, ignore_index=True)
    loss_dfs['scenario'] = file_scenario
    loss_dfs['year'] = file_year
    loss_dfs['recovery'] = recovery

    # exclude countries for which both losses are below x% of national refugia
    land_area_calculation(path_uea, 'bio1.3_bin.tif', 'bio1.3_bin_a.tif')
    refug_ref = rs.open(path_uea / 'bio1.3_bin_a.tif', masked=True)
    is03_refug_ref = admin_bound_calculator('iso3_refug_ref', admin_sf, refug_ref)
    is03_refug_ref = is03_refug_ref.rename(columns={'km2': 'km2_ref'})
    loss_dfs = pd.merge(loss_dfs, is03_refug_ref, on='iso3')
    loss_dfs['warm_loss_perc'] = (loss_dfs['km2_warm'] / loss_dfs['km2_ref']) * 100
    loss_dfs['luc_loss_perc'] = (loss_dfs['km2_luc'] / loss_dfs['km2_ref']) * 100
    mask = (loss_dfs['warm_loss_perc'] < thres) & (loss_dfs['luc_loss_perc'] < thres)
    loss_mask = loss_dfs.copy()
    loss_mask.loc[mask, ['impact', 'warm_loss_perc', 'luc_loss_perc']] = np.nan

    # determine model agreement on predominant effect across countries
    pos_count = loss_mask[loss_mask['impact'] == 1].groupby('iso3').size()
    pos_count = pos_count.reindex(loss_mask['iso3'].unique()).fillna(0).astype(int)
    pos_count = pos_count.apply(lambda x: 'Warm' if x >= 3 else 'NS')
    pos_df = pos_count.reset_index(name='impact')

    neg_count = loss_mask[loss_mask['impact'] == -1].groupby('iso3').size()
    neg_count = neg_count.reindex(loss_mask['iso3'].unique()).fillna(0).astype(int)
    neg_count = neg_count.apply(lambda x: 'Luc' if x >= 3 else 'NS')
    neg_df = neg_count.reset_index(name='impact')

    loss_mask = pd.concat([pos_df, neg_df]).drop_duplicates()
    duplicates = loss_mask[loss_mask.duplicated(subset='iso3', keep=False)]
    loss_mask = loss_mask[~((loss_mask['impact'] == 'NS') &
                        (loss_mask['iso3'].isin(duplicates['iso3'])))]

    # plot predominance of warming vs luc impact per country
    cmap = {'Warm': 'crimson', 'Luc': 'mediumblue', 'NS': 'gainsboro'}
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.Robinson()})

    shape_records = list(Reader(sf_path / 'world-administrative-boundaries.shp').records())

    # plot each country with data
    for record in shape_records:
        country_iso = record.attributes['iso3']
        if country_iso in loss_mask['iso3'].values:
            value = loss_mask[loss_mask['iso3'] == country_iso]['impact'].values[0]
            color = cmap[str(value)]
            geom = record.geometry
            ax.add_geometries([geom], ccrs.PlateCarree(), facecolor=color,
                              edgecolor='black', linewidth=0.2)

    ax.coastlines(linewidth=0.2)
    ax.set_aspect(1.1)

    legend_patches = [
        mpatches.Patch(color='crimson', label='More warm. loss'),
        mpatches.Patch(color='mediumblue', label="More LUC 'loss'"),
        mpatches.Patch(color='gainsboro', label='No agreement')]

    ax.legend(bbox_to_anchor=(0.0, 0.315), handles=legend_patches, ncols=1,
              loc='lower left', fontsize=15, handlelength=0.65, handletextpad=0.3,
              frameon=False)

    plt.title(f"Model agreement {file_scenario} {file_year} \n({recovery} & min. 'loss' of {thres}%)",
              fontsize=15, x=0.375, y=0.125, ha='left')
    plt.show()

# %% bivariate maps of warming vs (likely harmful) LUC at country level per model (SSP2-26 2100)
output_2 = loss_dfs[['model', 'scenario', 'year', 'recovery', 'iso3',
                     'warm_loss_perc', 'luc_loss_perc']].copy()

for model in models_ab:
    bivar_map = output_2.query('model == @model').reset_index()
    bivar_map = bivar_map.dropna(subset=['warm_loss_perc', 'luc_loss_perc']).copy()

    def classify(series, bins=[0, 25, 50, 100]):
        return pd.cut(series, bins=bins, labels=False, include_lowest=True)

    bivar_map['luc_bin'] = classify(bivar_map['luc_loss_perc'])  # x-axis
    bivar_map['warm_bin'] = classify(bivar_map['warm_loss_perc'])  # y-axis
    bivar_map['bivar_class'] = bivar_map['warm_bin'] * 3 + bivar_map['luc_bin']  # row * ncol + col

    color_matrix = [['#cf0523', '#7d0323', '#150024'],
                    ['#d18792', '#7e5194', '#160e97'],
                    ['#d3d3d3', '#7f7fd7', '#1616da']]

    flat_colors = [color_matrix[row][col] for row in range(3) for col in range(3)]

    color_map = bivar_map.set_index('iso3')['bivar_class'].to_dict()

    fig = plt.figure(figsize=(11, 6))
    ax = plt.axes([0.05, 0.05, 0.7, 0.9], projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines()
    ax.set_aspect(1.1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2)

    for sr in admin_sf.shapeRecords():
        iso = sr.record['iso3']
        geom = shape(sr.shape.__geo_interface__)
        if not geom.is_valid:
            continue
        try:
            color_idx = color_map[iso]
            row = color_idx // 3
            col = color_idx % 3
            facecolor = color_matrix[2 - row][col]
        except KeyError:
            facecolor = '#dddddd'
        ax.add_geometries([geom], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor='black', linewidth=0.2)

    # plot legend
    legend_ax = fig.add_axes([0.109, 0.375, 0.155, 0.155])
    for row in range(3):
        for col in range(3):
            color = color_matrix[2 - row][col]
            legend_ax.add_patch(plt.Rectangle((col, row), 1, 1, facecolor=color))

    legend_ax.set_yticks([0.5, 1.5, 2.5])
    legend_ax.set_xticklabels(['', '', ''])
    legend_ax.set_yticklabels(['0-25', '25-50', '50-100'], fontsize=15)
    legend_ax.set_xlabel("LUC 'loss' (%)", fontsize=15, labelpad=1)
    label = legend_ax.set_ylabel('Warm. loss (%)', fontsize=15, labelpad=1)
    label.set_bbox(dict(facecolor='white', edgecolor='none', pad=2.8))
    legend_ax.tick_params(axis='both', which='both', length=0)
    legend_ax.set_xlim(0, 3)
    legend_ax.set_ylim(0, 3)
    legend_ax.set_aspect('equal')

    legend_ax.annotate('', xy=(3.5, 0), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='-|>', linewidth=0.8,
                                       fc='black'), annotation_clip=False)
    legend_ax.annotate('', xy=(0, 3.5), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='-|>', linewidth=0.8,
                                       fc='black'), annotation_clip=False)

    model_name = model
    if model == 'MAgPIE':
        model_name = 'REMIND-MAgPIE'
    fig.text(0.313, 0.27, f'{model_name} SSP2-26 2100\n({recovery})', fontsize=15)
    plt.show()
