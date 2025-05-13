
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
path_globiom = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps')
path_aim = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps')
path_image = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/image_maps')
path_gcam = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/gcam_maps')
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
                              'IMAGE 3.0.1': 'IMAGE'}}, inplace=True)

# specify years for the analysis
years = ['2030', '2050', '2080', '2100']
lookup_sub_yrs = lookup_mi_luc_df.copy()
lookup_sub_yrs = lookup_sub_yrs.loc[lookup_sub_yrs['year'].isin(years)]

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

    # use overlay_calculator
    def process_row(row):
        input_tif = row['file_name']
        file_year = row['year']
        file_scenario = row['scenario']
        mitigation_option = row['mitigation_option']

        try:
            # run overlay_calculator for all scenarios to retrieve areas as outputs
            luc_in_bio_ref_agg, ref_bio_warm_loss_agg, refugia_ref_agg = \
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
                'refug_ref_warm_loss': ref_bio_warm_loss_agg}

            return result_dict

        except Exception as e:
            print(f'Unsuccessful for file {input_tif}: {e}')
            return {
                'scenario': file_scenario,
                'mitigation_option': mitigation_option,
                'year': file_year,
                'refug_ref': float('nan'),
                'luc_in_refug_ref': float('nan'),
                'refug_ref_warm_loss': float('nan')}

    area_df = pd.DataFrame.from_records(lookup_sub_yrs.apply(process_row,
                                                             axis=1).values)
    area_df = area_df.reset_index(drop=True)

    # preprocess df for plotting
    area_df = area_df.groupby(['scenario',
                               'year',
                               'refug_ref',
                               'refug_ref_warm_loss'])[['luc_in_refug_ref']].sum()
    area_df.reset_index(inplace=True)

    area_df['warm_loss_perc'] = area_df['refug_ref_warm_loss'] / area_df['refug_ref'] * 100
    area_df['land_loss_perc'] = area_df['luc_in_refug_ref'] / area_df['refug_ref'] * 100

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
area_df = pd.concat([decline_df, nodecline_df])

rcps = ['19', '26', '34', '45']  # specify RCPs that shall be plotted
area_df['RCP'] = area_df['RCP'].astype(str)
area_df = area_df.loc[area_df['RCP'].isin(rcps)]
area_df = area_df.loc[area_df['SSP'].isin(['SSP2'])]
decline_conditions = ['False', 'True']
decline_labels = ['No recovery', 'Full recovery']

rcp_palette = {'19': '#00adcf', '26': '#173c66', '34': '#f79320',
               '45': '#e71d24', '60': '#951b1d', 'Baseline': 'dimgrey'}

fig, axes = plt.subplots(2, 4, figsize=(11, 6), sharex=True, sharey=True)

for i, decline in enumerate(decline_conditions):
    for j, model in enumerate(models):

        data = area_df.query(f'Model == "{model}" & Decline == "{decline}"')

        sns.lineplot(data=data, x='land_loss_perc', y='warm_loss_perc', hue='RCP', sort=False,
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

# %% country-level agreement of warming vs LUC in SSP2-26 in 2050 and 2100
sf_path = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/wab')
admin_sf = shapefile.Reader(sf_path / 'world-administrative-boundaries.shp')

# adjust if necessary
thresholds = [5, 10]
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

        loss_df = warm_vs_luc_plotter(path,
                                      file_year,
                                      admin_sf,
                                      bio_select,
                                      file_scenario,
                                      'bio1.3_bin.tif',  # adjust if required
                                      model)
        loss_dfs.append(loss_df)

    loss_dfs = pd.concat(loss_dfs, ignore_index=True)

    # exclude countries for which both losses are below 1% of national refugia
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

    legend_patches = [
        mpatches.Patch(color='crimson', label='More warm. loss'),
        mpatches.Patch(color='mediumblue', label='More LUC loss'),
        mpatches.Patch(color='gainsboro', label='No agreement')]

    ax.legend(bbox_to_anchor=(0.01, 0.32), handles=legend_patches, ncols=1,
              loc='lower left', fontsize=12, columnspacing=0.8, handletextpad=0.5,
              frameon=False)

    plt.title(f'Model agreement {file_scenario} {file_year} \n({recovery} & min. loss of {thres}%)', fontsize=12,
              x=0.375, y=0.125, ha='left')
    plt.show()

# %% bivariate maps of warming vs LUC at country level per model (SSP2-26 2100)
loss_dfs = loss_dfs[['model', 'iso3', 'warm_loss_perc', 'luc_loss_perc']].copy()

for model in models:
    bivar_map = loss_dfs.query('model == @model').reset_index()
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
    legend_ax = fig.add_axes([0.105, 0.375, 0.16, 0.16])
    for row in range(3):
        for col in range(3):
            color = color_matrix[2 - row][col]
            legend_ax.add_patch(plt.Rectangle((col, row), 1, 1, facecolor=color))

    legend_ax.set_yticks([0.5, 1.5, 2.5])
    legend_ax.set_xticklabels(['', '', ''])
    legend_ax.set_yticklabels(['25%', '50%', '100%'], fontsize=12)
    legend_ax.set_xlabel('LUC loss', fontsize=12)
    legend_ax.set_ylabel('Warm. loss', fontsize=12)
    legend_ax.tick_params(axis='both', which='both', length=0)
    legend_ax.set_xlim(0, 3)
    legend_ax.set_ylim(0, 3)
    legend_ax.set_aspect('equal')
    legend_ax.spines[:].set_visible(False)
    fig.text(0.313, 0.27, f'{model} SSP2-26 2100\n{recovery}', fontsize=12)
    plt.show()

# %% comparison of refugia impact at 1.5C before and after overshoot
os_df = ar6_data.copy()
os_df.replace({'Model': {'AIM/CGE 2.0': 'AIM',
                         'MESSAGE-GLOBIOM 1.0': 'GLOBIOM',
                         'GCAM 4.2': 'GCAM',
                         'IMAGE 3.0.1': 'IMAGE'}}, inplace=True)

globiom_ssp119 = os_df.query('Model == "GLOBIOM" & Scenario == "SSP1-19"')
aim_ssp226 = os_df.query('Model == "AIM" & Scenario == "SSP2-26"')

# select last year before and first year after overshoot of 1.5C
globiom_os_yrs = globiom_ssp119[['Model', 'Scenario', '2035', '2069']].copy()
aim_os_yrs = aim_ssp226[['Model', 'Scenario', '2035', '2098']].copy()

os_plot_df = pd.concat([globiom_ssp119, aim_ssp226])
os_plot_df = os_plot_df[['Model', 'Scenario'] + all_years].copy()
os_plot_df = pd.melt(os_plot_df, id_vars=['Model', 'Scenario'], var_name='Year',
                     value_vars=all_years, value_name='Temp')
os_plot_df['ModScen'] = os_plot_df['Model'] + ' ' + os_plot_df['Scenario']
os_plot_df['Year'] = os_plot_df['Year'].astype(int)

# plot illustrative figure about overshoot period in selected scenarios
scen_pal = {'GLOBIOM SSP1-19': 'mediumvioletred', 'AIM SSP2-26': 'cornflowerblue'}
plt.figure(figsize=(7, 3))
plt.plot([2020, 2100], [1.5, 1.5], linewidth=1, linestyle='--', color='grey')
plt.plot([2035, 2035], [1.3, 1.5], linewidth=1, linestyle='--', color='grey')
plt.plot([2069, 2069], [1.3, 1.5], linewidth=1, linestyle='--', color='grey')
plt.plot([2098, 2098], [1.3, 1.5], linewidth=1, linestyle='--', color='grey')
sns.lineplot(os_plot_df, x='Year', y='Temp', hue='ModScen', palette=scen_pal,
             linewidth=4, alpha=0.7)
plt.xlim(2020, 2100)
plt.ylim(1.3, 1.61)
plt.xticks([2020, 2035, 2069, 2098])
plt.yticks([1.3, 1.4, 1.5, 1.6])
sns.despine()
plt.xlabel('')
plt.ylabel('Rounded median global warming [°C]\n(MAGICCv7.5.3)')
plt.legend(bbox_to_anchor=(0, 1.15), loc='upper left',
           columnspacing=1, handletextpad=0.4, ncols=2)
plt.show()

# estimate land for AR and bioenergy for the start and end year of overshoot
mitigation_options = ['Afforestation', 'Bioenergy']

for mitigation_option in mitigation_options:
    land_cover_interpolator('GLOBIOM', path_globiom, mitigation_option,
                            'SSP1-19', 2030, 2040, 2035)
    land_cover_interpolator('GLOBIOM', path_globiom, mitigation_option,
                            'SSP1-19', 2060, 2070, 2069)

    land_cover_interpolator('AIM', path_aim, mitigation_option,
                            'SSP2-26', 2030, 2040, 2035)
    land_cover_interpolator('AIM', path_aim, mitigation_option,
                            'SSP2-26', 2090, 2100, 2098)

# estimate land impact on 1.5 °C-refugia before and after overshoot
os_land_in_refugia_calculator('GLOBIOM', path_globiom, 'SSP1-19', 2035, 2069)
os_land_in_refugia_calculator('AIM', path_aim, 'SSP2-26', 2035, 2098)

# plot difference in land impact on refugia before and after overshoot
os_scenarios = ['GLOBIOM SSP1-19', 'AIM SSP2-26']

for scenario in os_scenarios:
    if scenario == 'GLOBIOM SSP1-19':
        path = path_globiom
        os_file = 'GLOBIOM_SSP1-19_pre_vs_post_os.tif'
    elif scenario == 'AIM SSP2-26':
        path = path_aim
        os_file = 'AIM_SSP2-26_pre_vs_post_os.tif'

    os_diff = rs.open(path / os_file)
    refug = rs.open(path_uea / 'bio1.5_bin.tif')

    data_os_diff = os_diff.read(1)
    data_refug = refug.read(1)

    data_os_diff[data_os_diff == 0] = np.nan  # ignore zero values

    # get the metadata
    transform = os_diff.transform
    extent_os = [transform[2], transform[2] + transform[0] * os_diff.width,
                 transform[5] + transform[4] * os_diff.height, transform[5]]

    transform = refug.transform
    extent_refug = [transform[2], transform[2] + transform[0] * refug.width,
                    transform[5] + transform[4] * refug.height, transform[5]]

    bounds_os = [-40, -20, -1, 1, 20, 40]
    norm_os = mpl.colors.BoundaryNorm(bounds_os, mpl.cm.PuOr.N, extend='both')

    fig = plt.figure(figsize=(10, 6.1))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertAzimuthalEqualArea())  # choose projection | LambertAzimuthalEqualArea())

    img_re = ax.imshow(data_refug, extent=extent_refug, transform=ccrs.PlateCarree(),
                       origin='upper', cmap='Greys', alpha=0.1)

    img_os = ax.imshow(data_os_diff, extent=extent_os, transform=ccrs.PlateCarree(),
                       origin='upper', cmap='PuOr', norm=norm_os, alpha=1)

    ax.coastlines(linewidth=0.2)

    cbar_os = plt.colorbar(img_os, ax=ax, orientation='horizontal', aspect=13, pad=0.16)
    cbar_os.ax.set_position([0.41, -0.165, 0.2, 0.501])
    cbar_os.ax.tick_params(labelsize=7)
    cbar_os.set_label('Difference in land-based mitigation within refugia at 1.5 °C\n(post- vs. pre-overshoot) [% cell area]',
                      labelpad=1, fontsize=8)
    plt.title(f'{scenario}', fontsize=8, ha='center')
    plt.show()

# calculate share of 1.5 °C climate refugia that would be lost at 1.6 °C
refugia1p5 = rioxarray.open_rasterio(path_uea / 'bio1.5_bin.tif', masked=True)
refugia1p6 = rioxarray.open_rasterio(path_uea / 'bio1.6_bin.tif', masked=True)

bio_warm_loss = refugia1p5 - refugia1p6
bio_warm_loss.rio.to_raster(path_uea / 'warm_loss1.6-1.5.tif', driver='GTiff')
lost_ref = land_area_calculation(path_uea, 'warm_loss1.6-1.5.tif')
refug1p5 = land_area_calculation(path_uea, 'bio1.5_bin.tif')

lost_ref = pos_val_summer(lost_ref, squeeze=True)
refug1p5 = pos_val_summer(refug1p5, squeeze=True)
lost_share = (lost_ref / refug1p5) * 100
print('Share of 1.5°C-refugia lost at 1.6 °C peak (%):', lost_share)
