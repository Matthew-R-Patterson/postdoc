""" Functions for creating plots """

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import sys

class make_map_plot:

    def __init__(self,extend='both'):
        self.extend = extend


    def add_filled_contours(self,lons,lats,field,clevs,cmap='RdBu_r'):
        cs = plt.contourf(lons,lats,field,clevs,cmap=cmap,extend=self.extend,transform=ccrs.PlateCarree())
        return cs

    def add_contours(self,lons,lats,field,clevs,colors='k',linewidths=1,contour_labels=False):
        cs = plt.contour(lons,lats,field,clevs,colors=colors,linewidths=linewidths,transform=ccrs.PlateCarree())
        if contour_labels == True:
            plt.clabel(cs, fmt='%.f',inline=True, fontsize=15)
        return cs

    def geography(self,ax,extent=[0,359.99,-55,55],borders=True):
        ax.coastlines()
        if borders == True: ax.add_feature(cfeature.BORDERS)
        #self.add_lat_lon(ax)
        ax.set_extent(extent)

def add_lat_lon(ax,fontsize=20):
    """ Add latitude longitude markers """
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    ax.set_xticks(np.arange(0,360,60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-80,81,20), crs=ccrs.PlateCarree())
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    lon_formatter = LongitudeFormatter(number_format='.0f',degree_symbol='',dateline_direction_label=True)
    lat_formatter = LatitudeFormatter(number_format='.0f',degree_symbol='')
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)



def plot_box(lon_min,lon_max,lat_min,lat_max,linewidth=3,color='k'):
    plt.plot([lon_min,lon_min],[lat_min,lat_max],color=color,linewidth=linewidth,transform=ccrs.PlateCarree())
    plt.plot([lon_max,lon_max],[lat_min,lat_max],color=color,linewidth=linewidth,transform=ccrs.PlateCarree())
    plt.plot([lon_min,lon_max],[lat_min,lat_min],color=color,linewidth=linewidth,transform=ccrs.PlateCarree())
    plt.plot([lon_min,lon_max],[lat_max,lat_max],color=color,linewidth=linewidth,transform=ccrs.PlateCarree())

def colorbar(ax,cs,labelsize=20,orientation='horizontal'):
    cb = plt.colorbar(cs,cax=ax,orientation=orientation)
    cb.ax.tick_params(labelsize=labelsize)
    return cb

def contour_plot(ax,field,clevs,lats,lons,climatology=None,clim_clevs=None,lats_ua=None,lons_ua=None,calculate_significance=True,title_fontsize=20,title='',cmap='RdBu_r',extent=[0,359.99,-10,60],plot_box=False): #[60,210,0,60]
    m = make_map_plot()
    if calculate_significance==True:
        pvals = t_test_autocorr(data_all,data_all[mask,:,:],autocorr=0)
        field = np.mean(data_all[mask],axis=0)
        cs = m.add_filled_contours(lons,lats,field,clevs,cmap=cmap)
        plt.contour(lons,lats,pvals,np.array([0.05]),colors='k',transform=ccrs.PlateCarree())
        title = title + ' (' + str(data_all[mask].shape[0]) + ')'
    else: cs = m.add_filled_contours(lons,lats,field,clevs,cmap=cmap)
    if climatology is not None: 
        if lats_ua == None: plt.contour(lons,lats,climatology,clim_clevs,colors='0.75',transform=ccrs.PlateCarree())
        else: plt.contour(lons_ua,lats_ua,climatology,clim_clevs,colors='0.75',transform=ccrs.PlateCarree())
    if plot_box == True:
        plt.plot([70,150],[20,20],color='k',transform=ccrs.PlateCarree())
        plt.plot([70,150],[50,50],color='k',transform=ccrs.PlateCarree())
        plt.plot([70,70],[20,50],color='k',transform=ccrs.PlateCarree())
        plt.plot([150,150],[20,50],color='k',transform=ccrs.PlateCarree())
    plt.title(title,fontsize=title_fontsize)
    add_lat_lon(ax)
    m.geography(ax,extent=extent)
    ax.set_aspect(1.5)
    return cs

