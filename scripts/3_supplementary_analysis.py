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

ar6_db = pd.read_csv(path_ar6_data / 'AR6_Scenarios_Database_World_v1.1.csv')
lookup_mi_luc_df = pd.read_csv(path_project / 'lookup_table_ar_bioenergy_files_all_models.csv')
lookup_mi_luc_df['year'] = lookup_mi_luc_df['year'].astype(str)

# define lists
years = ['2010', '2020', '2030', '2040', '2050', '2060', '2070', '2080',
         '2090', '2100']

models = ['MESSAGE-GLOBIOM 1.0', 'AIM/CGE 2.0', 'GCAM 4.2', 'IMAGE 3.0.1',
          'REMIND-MAgPIE 1.5']
scenarios = ['SSP1-19', 'SSP1-26', 'SSP1-34', 'SSP1-45', 'SSP2-19', 'SSP2-26',
             'SSP2-34', 'SSP2-45', 'SSP3-34', 'SSP3-45']
variable = ['AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile']

variables = ['Land Cover|Built-up Area', 'Land Cover|Cropland',
             'Land Cover|Cropland|Energy Crops', 'Land Cover|Forest',
             'Land Cover|Pasture']
variables_adjust = ['Land Cover|Built-up Area', 'Other cropland',
                    'Land Cover|Forest', 'Land Cover|Cropland|Energy Crops',
                    'Land Cover|Pasture']

# %% get temperatures for SSP-RCP combinations
all_years = [str(year) for year in range(2020, 2101)]

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
temperature_decline = 'allowed'  # options: 'allowed' or 'not_allowed'

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
years = ['2100']
lookup_sub_yrs = lookup_mi_luc_df.copy()
lookup_sub_yrs = lookup_sub_yrs.loc[lookup_sub_yrs['year'].isin(years)]

# %% country-level agreement of warming vs LUC in SSP2-26 2100
sf_path = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/wab')
admin_sf = shapefile.Reader(sf_path / 'world-administrative-boundaries.shp')

# adjust if necessary
models = ['AIM', 'GCAM', 'GLOBIOM', 'IMAGE', 'MAgPIE']
thresholds = [1, 8, 10, 12, 15, 20]  # use lower bound thresholds to exclude very low effects
file_scenario = 'SSP2-26'
file_year = '2100'

# use warm_vs_luc_plotter to determine predominant effect at country level
for thres in thresholds:

    loss_dfs = []
    for model in models:
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

        loss_df = warm_vs_luc_plotter(path,
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

# %% get AR6 land cover data for SSP-RCP combinations
ar6_db = ar6_db.loc[ar6_db['Model'].isin(models)]
ar6_db = ar6_db.loc[ar6_db['Scenario'].isin(scenarios)]
lc_data = ar6_db.loc[ar6_db['Variable'].isin(variables)]
lc_data = lc_data[['Model', 'Scenario', 'Variable'] + years].copy()

lc_data = pd.melt(lc_data, id_vars=['Model', 'Scenario', 'Variable'],
                  value_vars=years, var_name='Year', value_name='Value')
lc_ar6 = lc_data.copy()

# distinguish between energy cropland and other cropland
cropland = lc_data.loc[lc_data['Variable'].isin(['Land Cover|Cropland'])]
cropland_energy = lc_data.loc[lc_data['Variable'].isin(['Land Cover|Cropland|Energy Crops'])]

cropland_other = pd.merge(cropland, cropland_energy,
                          on=['Model', 'Scenario', 'Year'],
                          suffixes=['_all', '_energy'])

cropland_other['Value'] = cropland_other['Value_all'] - cropland_other['Value_energy']
cropland_other['Variable'] = 'Other cropland'
cropland_other = cropland_other[['Model', 'Scenario', 'Variable', 'Year',
                                 'Value']].copy()
lc_data = pd.concat([lc_data, cropland_other], axis=0)

# calculate change from base year (2010)
lc_data = lc_data.loc[lc_data['Variable'].isin(variables_adjust)]
lc_2010 = lc_data.query('Year == "2010"').reset_index(drop=True)
lc_data = pd.merge(lc_data, lc_2010, on=['Model', 'Scenario', 'Variable'],
                   suffixes=['', '_2010'])

lc_data['Change'] = lc_data['Value'] - lc_data['Value_2010']
lc_data['Change'] = lc_data['Change'] / 100  # from Mha to Mkm2

# plot supplementary figure on land use changes based on AR6 Scenarios Database
lc_data['SSP'] = lc_data['Scenario'].str.split('-').str[0]
lc_data['RCP'] = lc_data['Scenario'].str.split('-').str[1]
lc_data['Year'] = lc_data['Year'].astype(int)

lc_data.replace({'Model': {'AIM/CGE 2.0': 'AIM',
                           'MESSAGE-GLOBIOM 1.0': 'GLOBIOM',
                           'IMAGE 3.0.1': 'IMAGE',
                           'GCAM 4.2': 'GCAM',
                           'REMIND-MAgPIE 1.5': 'MAgPIE'}}, inplace=True)

lc_data.replace({'Variable': {'Land Cover|Built-up Area': 'Built-up',
                              'Land Cover|Cropland|Energy Crops': 'Energy cropland',
                              'Land Cover|Forest': 'Forest',
                              'Land Cover|Pasture': 'Pasture'}}, inplace=True)

var_pal = {'Built-up': 'dimgrey',
           'Energy cropland': 'blue',
           'Forest': 'forestgreen',
           'Other cropland': 'brown',
           'Pasture': 'gold'}
all_vars = sorted(lc_data['Variable'].unique())
#%%
fig, axes = plt.subplots(5, 4, figsize=(9, 9), sharex=True, sharey=True)
sns.lineplot(data=lc_data.query('Model == "AIM" & RCP == "19"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=True, ax=axes[0, 0])
sns.lineplot(data=lc_data.query('Model == "GCAM" & RCP == "19"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 0])
sns.lineplot(data=lc_data.query('Model == "GLOBIOM" & RCP == "19"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 0])
sns.lineplot(data=lc_data.query('Model == "IMAGE" & RCP == "19"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[3, 0])
sns.lineplot(data=lc_data.query('Model == "MAgPIE" & RCP == "19"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[4, 0])

sns.lineplot(data=lc_data.query('Model == "AIM" & RCP == "26"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 1])
sns.lineplot(data=lc_data.query('Model == "GCAM" & RCP == "26"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 1])
sns.lineplot(data=lc_data.query('Model == "GLOBIOM" & RCP == "26"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 1])
sns.lineplot(data=lc_data.query('Model == "IMAGE" & RCP == "26"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[3, 1])
sns.lineplot(data=lc_data.query('Model == "MAgPIE" & RCP == "26"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[4, 1])

sns.lineplot(data=lc_data.query('Model == "AIM" & RCP == "34"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 2])
sns.lineplot(data=lc_data.query('Model == "GCAM" & RCP == "34"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 2])
sns.lineplot(data=lc_data.query('Model == "GLOBIOM" & RCP == "34"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 2])
sns.lineplot(data=lc_data.query('Model == "IMAGE" & RCP == "34"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[3, 2])
sns.lineplot(data=lc_data.query('Model == "MAgPIE" & RCP == "34"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[4, 2])

sns.lineplot(data=lc_data.query('Model == "AIM" & RCP == "45"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 3])
sns.lineplot(data=lc_data.query('Model == "GCAM" & RCP == "45"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 3])
sns.lineplot(data=lc_data.query('Model == "GLOBIOM" & RCP == "45"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 3])
sns.lineplot(data=lc_data.query('Model == "IMAGE" & RCP == "45"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[3, 3])
sns.lineplot(data=lc_data.query('Model == "MAgPIE" & RCP == "45"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[4, 3])

for ax in axes.flat:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(2010, 2100)
    ax.set_xticks([2010, 2055, 2100])
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.8)

axes[0, 0].set_ylabel('AIM', fontsize=11)
axes[1, 0].set_ylabel('GCAM', fontsize=11)
axes[2, 0].set_ylabel('GLOBIOM', fontsize=11)
axes[3, 0].set_ylabel('IMAGE', fontsize=11)
axes[4, 0].set_ylabel('REMIND-MAgPIE', fontsize=11)

axes[0, 0].set_title('RCP1.9')
axes[0, 1].set_title('RCP2.6')
axes[0, 2].set_title('RCP3.4')
axes[0, 3].set_title('RCP4.5')

axes[0, 0].legend(bbox_to_anchor=(-0.05, 1.55), loc='upper left', ncols=5,
                  columnspacing=1, handlelength=0.7, handletextpad=0.4, fontsize=11)

fig.supylabel(f'Land cover change from 2010 [Mkm$^2$] (SSP1-SSP3 range as shading)',
              x=0.03, va='center', ha='center')

plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(wspace=0.4)
sns.despine()
plt.show()