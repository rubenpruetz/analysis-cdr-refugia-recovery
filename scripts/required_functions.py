
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

# function to convert nc to geotiff


def nc_geotiff_converter(input_nc,  # string
                         filepath,  # string + /
                         variable,  # string
                         time_band,  # options: 0-9 -> 2010-2100 (10y steps)
                         output_name):  # output tiff (string)
    nc_file = rioxarray.open_rasterio(filepath / input_nc, decode_times=False)
    data_array = nc_file[variable][time_band]
    data_array.rio.to_raster(filepath / output_name, driver='GTiff')
    # update meta data with rasterio to avoid later conflicts
    with rs.open(filepath / output_name) as src:  # specify file name
        data = src.read(1)  # assuming single band image
        profile = src.profile
        profile.update(dtype=rs.float32)

    with rs.open(filepath / output_name, 'w', **profile) as dst:
        dst.write(data, 1)


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

# function to plot land-per-removal, removal, and land


def process_data_and_plot(land_df, removal_df, cdr_option):
    land_per_removal = pd.merge(land_df, removal_df, on=['Scenario', 'Year'])
    land_per_removal['Land'] = land_per_removal['Land'] * 0.000001  # km2 to Mkm2
    land_per_removal.loc[land_per_removal['Land'] == 0, 'Removal'] = 0  # consistency rule: if land=0, removal=0
    land_per_removal['Mkm2/GtCO2'] = land_per_removal['Land'] / land_per_removal['Removal']
    land_per_removal['SSP'] = land_per_removal['Scenario'].str.split('-').str[0]
    land_per_removal['RCP'] = land_per_removal['Scenario'].str.split('-').str[1]
    rcp_pal = {'19': '#00adcf', '26': '#173c66', '34': '#f79320',
               '45': '#e71d24', '60': '#951b1d', 'Baseline': 'dimgrey'}
    rcp_order = ['19', '26', '34', '45', '60', 'Baseline']
    plot1 = sns.relplot(data=land_per_removal, x='Year', y='Mkm2/GtCO2', col='SSP', hue='RCP', linewidth=1.5, marker='o',
                        kind='line', clip_on=False, palette=rcp_pal, hue_order=rcp_order, height=4, aspect=0.42)
    plot1.set_xlabels('')
    plot1.set_ylabels(f'{cdr_option} land-per-removal (Mkm$^2$/GtCO$_2$)')
    plot1.fig.subplots_adjust(wspace=0.37)
    for ax in plot1.axes.flat:
        ax.set_xlim(2020, 2100)
        ax.set_xticks([2020, 2060, 2100])

    plot2 = sns.relplot(data=land_per_removal, x='Year', y='Removal', col='SSP', hue='RCP', linewidth=1.5, marker='o',
                        kind='line',  clip_on=False,palette=rcp_pal, hue_order=rcp_order, height=4, aspect=0.42)
    plot2.set_xlabels('')
    plot2.set_ylabels(f'{cdr_option} removal (GtCO$_2$)')
    plot2.fig.subplots_adjust(wspace=0.37)
    for ax in plot2.axes.flat:
        ax.set_xlim(2020, 2100)
        ax.set_xticks([2020, 2060, 2100])

    plot3 = sns.relplot(data=land_per_removal, x='Year', y='Land', col='SSP', hue='RCP', linewidth=1.5, marker='o',
                        kind='line',  clip_on=False, palette=rcp_pal, hue_order=rcp_order, height=4, aspect=0.42)
    plot3.set_xlabels('')
    plot3.set_ylabels(f'{cdr_option} land (Mkm$^2$)')
    plot3.fig.subplots_adjust(wspace=0.37)
    for ax in plot3.axes.flat:
        ax.set_xlim(2020, 2100)
        ax.set_xticks([2020, 2060, 2100])

    sns.move_legend(plot1, 'lower right', bbox_to_anchor=(0.85, 0.97), ncol=6, title='', columnspacing=0.8)
    sns.move_legend(plot2, 'lower right', bbox_to_anchor=(0.85, 0.97), ncol=6, title='', columnspacing=0.8)
    sns.move_legend(plot3, 'lower right', bbox_to_anchor=(0.85, 0.97), ncol=6, title='', columnspacing=0.8)
    plt.show()

    return land_per_removal[['SSP', 'RCP', 'Year', 'Land', 'Removal', 'Mkm2/GtCO2']]

# function to plot land-per-removal across models and ssps


def process_mi_data_and_plot(land_df, removal_df, cdr_option, removal_type):
    land_per_removal = pd.merge(land_df, removal_df, on=['Model', 'Scenario', 'Year'])
    land_per_removal['Land'] = land_per_removal['Land'] * 0.000001  # km2 to Mkm2
    land_per_removal.loc[land_per_removal['Land'] == 0, 'Removal'] = 0  # consistency rule: if land=0, removal=0
    land_per_removal['Mkm2/GtCO2'] = land_per_removal['Land'] / land_per_removal['Removal']
    land_per_removal['SSP'] = land_per_removal['Scenario'].str.split('-').str[0]
    land_per_removal['RCP'] = land_per_removal['Scenario'].str.split('-').str[1]

    rcp_pal = {'19': '#00adcf', '26': '#173c66', '34': '#f79320',
               '45': '#e71d24', '60': '#951b1d', 'Baseline': 'dimgrey'}
    all_rcps = sorted(land_per_removal['RCP'].unique())

    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey=True)
    sns.lineplot(data=land_per_removal.query('SSP == "SSP1" & Model == "GLOBIOM"'), x='Year',
                    y='Mkm2/GtCO2', hue='RCP', hue_order=all_rcps, palette=rcp_pal,
                    marker='o', legend=True, ax=axes[0, 0])
    sns.lineplot(data=land_per_removal.query('SSP == "SSP1" & Model == "AIM"'), x='Year',
                    y='Mkm2/GtCO2', hue='RCP', palette=rcp_pal,
                    marker='o', legend=False, ax=axes[1, 0])
    sns.lineplot(data=land_per_removal.query('SSP == "SSP1" & Model == "IMAGE"'), x='Year',
                    y='Mkm2/GtCO2', hue='RCP', palette=rcp_pal,
                    marker='o', legend=False, ax=axes[2, 0])

    sns.lineplot(data=land_per_removal.query('SSP == "SSP2" & Model == "GLOBIOM"'), x='Year',
                    y='Mkm2/GtCO2', hue='RCP', palette=rcp_pal,
                    marker='o', legend=False, ax=axes[0, 1])
    sns.lineplot(data=land_per_removal.query('SSP == "SSP2" & Model == "AIM"'), x='Year',
                    y='Mkm2/GtCO2', hue='RCP', palette=rcp_pal,
                    marker='o', legend=False, ax=axes[1, 1])
    sns.lineplot(data=land_per_removal.query('SSP == "SSP2" & Model == "IMAGE"'), x='Year',
                    y='Mkm2/GtCO2', hue='RCP', palette=rcp_pal,
                    marker='o', legend=False, ax=axes[2, 1])

    sns.lineplot(data=land_per_removal.query('SSP == "SSP3" & Model == "GLOBIOM"'), x='Year',
                    y='Mkm2/GtCO2', hue='RCP', palette=rcp_pal,
                    marker='o', legend=False, ax=axes[0, 2])
    sns.lineplot(data=land_per_removal.query('SSP == "SSP3" & Model == "AIM"'), x='Year',
                    y='Mkm2/GtCO2', hue='RCP', palette=rcp_pal,
                    marker='o', legend=False, ax=axes[1, 2])
    sns.lineplot(data=land_per_removal.query('SSP == "SSP3" & Model == "IMAGE"'), x='Year',
                    y='Mkm2/GtCO2', hue='RCP', palette=rcp_pal,
                    marker='o', legend=False, ax=axes[2, 2])

    axes[0, 0].legend(bbox_to_anchor=(-0.05, 1.35), loc='upper left', ncols=6,
                      columnspacing=2, handletextpad=0.5)

    axes[0, 0].set_title('SSP1')
    axes[0, 1].set_title('SSP2')
    axes[0, 2].set_title('SSP3')

    axes[2, 0].set_xlabel('')
    axes[2, 1].set_xlabel('')
    axes[2, 2].set_xlabel('')

    axes[0, 0].set_ylabel('GLOBIOM')
    axes[1, 0].set_ylabel('AIM')
    axes[2, 0].set_ylabel('IMAGE')

    fig.supxlabel('')
    if removal_type == 'cumulative':
        fig.supylabel(f'{cdr_option} Land-per-cumulative removal (Mkm$^2$/GtCO$_2$)', x=0.05)
    else:
        fig.supylabel(f'{cdr_option} Land-per-removal (Mkm$^2$/GtCO$_2$)', x=0.05)

    for ax in axes.flat:
        ax.set_xlim(2020, 2100)
        ax.set_xticks([2020, 2060, 2100])

    plt.subplots_adjust(hspace=0.15)
    plt.subplots_adjust(wspace=0.15)
    sns.despine()
    plt.show()

# function to interpolate between available years to estimate cumulative removal


def cum_cdr_calc(cdr_df):
    numeric_cols20 = [str(year) for year in range(2020, 2110, 10)]
    year_cols_all = [str(year) for year in range(2020, 2101)]
    cdr = cdr_df.reindex(columns=['Scenario', 'Variable'] + year_cols_all, fill_value=None)
    cdr.loc[pd.isna(cdr['2020']), '2020'] = 0
    cdr.set_index(['Scenario', 'Variable'], inplace=True)
    cdr.interpolate(method='linear', inplace=True, axis=1)
    cdr.reset_index(inplace=True)
    cum_cdr = cdr[['Scenario', 'Variable']].copy()

    for year in range(2020, 2101):
        current_year = str(year)
        year_range = [str(year) for year in range(2020, year + 1)]
        cum_cdr[current_year] = cdr[year_range].sum(axis=1)

    cum_cdr = cum_cdr[['Scenario', 'Variable'] + numeric_cols20].copy()

    cum_cdr = pd.melt(cum_cdr, id_vars=['Scenario', 'Variable'], var_name='Year',
                      value_vars=numeric_cols20, value_name='Removal')

    cum_cdr['Year'] = cum_cdr['Year'].astype(int)

    return cum_cdr

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