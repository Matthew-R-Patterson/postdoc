"""
Functions to plot data and statistical analysis on maps

add_latlon_labels = Add latitude and longitude labels to map plot

"""

import iris
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def add_latlon_labels(ax,fontsize=30):
    """ Add latitude and longitude labels to map plot
    Only works for PlateCarree projection """

    ax.set_xticks(np.arange(0,360,60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-80,81,10), crs=ccrs.PlateCarree())
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    lon_formatter = LongitudeFormatter(number_format='.0f',degree_symbol='',dateline_direction_label=True)
    lat_formatter = LatitudeFormatter(number_format='.0f',degree_symbol='')
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)


def plot_contours(data,clevs,ax,title='', colors='k', transform=ccrs.PlateCarree(),
                 clabel=None, title_fontsize=35, clabel_fontsize=25,
                 aspect=2):
    """ Add contours to a map """

    plt.title(title,fontsize=title_fontsize)
    cs = plt.contour(lons,lats,data,clevs,colors=colors,transform=transform)
    if clabel is not None: 
        plt.clabel(cs, fmt='%1.0f', colors='k', fontsize=clabel_fontsize)
    ax.coastlines()
    ax.set_aspect(aspect)

    return cs


def plot_filled_contours(data,clevs,ax,title='', cmap='RdBu_r',
                 transform=ccrs.PlateCarree(), extend='both',
                 title_fontsize=35, aspect=2):
    """ Add filled contours to a map """

    plt.title(title,fontsize=title_fontsize)
    cs = plt.contourf(lons,lats,data,clevs,cmap=cmap,extend=extend,transform=transform)
    ax.coastlines()
    ax.set_aspect(aspect)

    return cs


