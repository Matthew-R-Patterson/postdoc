""" Plot the results of MCA analysis of sea level pressure and surface air temperatures for individual models """


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from eofs.standard import Eof
import os
import sys


# my modules
cwd = os.getcwd()
repo_dir = '/'
for directory in cwd.split('/')[1:]:
    repo_dir = os.path.join(repo_dir, directory)
    if directory == 'postdoc':
        break

analysis_functions_dir = os.path.join(repo_dir,'analysis_functions')
figure_dir = os.path.join(repo_dir,'figures/A2_1_SSTs_East_Asian_circulation')
data_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/MCA_data')

model_name_list = ['AWI-CM-1-1-MR','BCC-CSM2-MR','CAMS-CSM1-0', 'FGOALS-f3-L', 'FGOALS-g3', 'CanESM5', 'CanESM5-CanOE', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'CNRM-ESM2-1', 'ACCESS-ESM1-5', 'ACCESS-CM2' ,'EC-Earth3', 'EC-Earth3-Veg', 'INM-CM4-8' ,'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MIROC-ES2L', 'HadGEM3-GC31-LL', 'UKESM1-0-LL', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'GISS-E2-1-G', 'CESM2', 'CESM2-WACCM', 'NorESM2-LM', 'NorESM2-MM', 'GFDL-CM4' ,'GFDL-ESM4', 'NESM3','MCM-UA-1-0']

# Read in data
variable1_name = 'psl'
variable2_name = 'tas'
model_name = 'NESM3' #'HadGEM3-GC31-LL' #'MPI-ESM1-2-LR' #'UKESM1-0-LL'#'MPI-ESM1-2-LR'#'HadGEM3-GC31-LL'
file_name = data_dir + '/MCA_data_' + model_name + '_' + variable1_name + '_' + variable2_name + '.nc'
nc = Dataset(file_name,'r')
# dimensions
lats1 = nc.variables['lats1'][:]
lons1 = nc.variables['lons1'][:]
lats2 = nc.variables['lats2'][:]
lons2 = nc.variables['lons2'][:]
years = nc.variables['running_mean_years'][:]
# variables
reg_coeff_PC1_sat_on_slp = nc.variables['reg_coeff_pc1_var2_variable1'][:]
pvals_PC1_sat_on_slp = nc.variables['pvals_pc1_var2_variable1'][:]
reg_coeff_PC1_sat_on_sat = nc.variables['reg_coeff_pc1_var2_variable2'][:]
pvals_PC1_sat_on_sat = nc.variables['pvals_pc1_var2_variable2'][:]

reg_coeff_MCA1_sat_on_slp = nc.variables['reg_coeff_v1_variable1'][:]
pvals_MCA1_sat_on_slp = nc.variables['pvals_v1_variable1'][:]
reg_coeff_MCA1_sat_on_sat = nc.variables['reg_coeff_v1_variable2'][:]
pvals_MCA1_sat_on_sat = nc.variables['pvals_v1_variable2'][:]

u_ts1 = nc.variables['u_ts'][0,:]
v_ts1 = nc.variables['v_ts'][0,:]

# Plot 

class make_map_plot:

    def __init__(self):
        self.cmap = 'RdBu_r'
        self.extend = 'both'

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

    def add_filled_contours(self,lons,lats,field,clevs):
        cs = plt.contourf(lons,lats,field,clevs,cmap=self.cmap,extend=self.extend,transform=ccrs.PlateCarree())
        return cs

    def add_contours(self,lons,lats,field,clevs,contour_labels=False):
        cs = plt.contour(lons,lats,field,clevs,colors='k',transform=ccrs.PlateCarree())
        if contour_labels == True:
            plt.clabel(cs, fmt='%.f',inline=True, fontsize=15)
        return cs

    def geography(self,ax,borders=True):
        ax.coastlines()
        if borders == True: ax.add_feature(cfeature.BORDERS)
        #self.add_lat_lon(ax)
        ax.set_extent([0,359.99,-55,55])

def plot_box(lon_min,lon_max,lat_min,lat_max):
    plt.plot([lon_min,lon_min],[lat_min,lat_max],color='k',linewidth=3,transform=ccrs.PlateCarree())
    plt.plot([lon_max,lon_max],[lat_min,lat_max],color='k',linewidth=3,transform=ccrs.PlateCarree())
    plt.plot([lon_min,lon_max],[lat_min,lat_min],color='k',linewidth=3,transform=ccrs.PlateCarree())
    plt.plot([lon_min,lon_max],[lat_max,lat_max],color='k',linewidth=3,transform=ccrs.PlateCarree())

def colorbar(ax,cs):
    cb = plt.colorbar(cs,cax=ax,orientation='horizontal')
    cb.ax.tick_params(labelsize=20)




# set up plot
plt.figure(figsize=(20,15))
gs = gridspec.GridSpec(5,4,height_ratios=[10,0.5,10,0.5,10])
title_fontsize=30

# contour levels
a = np.arange(0.05,0.41,0.05)
clevs_tas = np.append(-a[::-1],a)
a = np.arange(5,41,5)
clevs_slp = np.append(-a[::-1],a)
clevs_pvals = np.array([0,0.05])

m = make_map_plot()

ax = plt.subplot(gs[0,0:2],projection=ccrs.PlateCarree(central_longitude=180.))
plt.title('PC1 of SAT on SAT (K)',fontsize=title_fontsize)
cs = m.add_filled_contours(lons1,lats1,reg_coeff_PC1_sat_on_sat,clevs_tas)
#m.add_contours(lons1,lats1,pvals_PC1_sat_on_sat,clevs_pvals)
m.geography(ax)
ax = plt.subplot(gs[1,0:2])
colorbar(ax,cs)

ax = plt.subplot(gs[0,2:4],projection=ccrs.PlateCarree(central_longitude=180.))
plt.title('PC1 of SAT on SLP (hPa)',fontsize=title_fontsize)
cs = m.add_filled_contours(lons1,lats1,reg_coeff_PC1_sat_on_slp,clevs_slp)
#m.add_contours(lons1,lats1,pvals_PC1_sat_on_slp,clevs_pvals)
m.geography(ax)
plot_box(60,150,10,50)
ax = plt.subplot(gs[1,2:4])
colorbar(ax,cs)

ax = plt.subplot(gs[2,0:2],projection=ccrs.PlateCarree(central_longitude=180.))
plt.title('MCA1 of SAT on SAT (K)',fontsize=title_fontsize)
cs = m.add_filled_contours(lons1,lats1,reg_coeff_MCA1_sat_on_sat,clevs_tas)
m.geography(ax)
ax = plt.subplot(gs[3,0:2])
colorbar(ax,cs)

ax = plt.subplot(gs[2,2:4],projection=ccrs.PlateCarree(central_longitude=180.))
plt.title('MCA1 of SAT on SLP (hPa)',fontsize=title_fontsize)
cs = m.add_filled_contours(lons1,lats1,reg_coeff_MCA1_sat_on_slp,clevs_slp)
m.geography(ax)
plot_box(60,150,10,50)
ax = plt.subplot(gs[3,2:4])
colorbar(ax,cs)

ax = plt.subplot(gs[4,:])
plt.title('MCA1 SAT/SLP time series',fontsize=title_fontsize)
ax.plot(years,u_ts1,color='r')
ax2 = ax.twinx()
ax2.plot(years,v_ts1,color='b')

plt.subplots_adjust(hspace=0.4)
plt.savefig(figure_dir+'/MCA1_pc1_tas_slp_'+model_name+'.png',bbox_inches='tight')
#plt.show()
