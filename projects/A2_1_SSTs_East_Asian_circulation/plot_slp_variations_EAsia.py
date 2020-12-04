""" Plot the first few EOFs of sea level pressure for the East Asian region in 
HadGEM3 and in ERA-Interim. """

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib import gridspec
import cartopy.crs as ccrs
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

import make_plots
analysis_functions_dir = os.path.join(repo_dir,'analysis_functions')
figure_dir = os.path.join(repo_dir,'figures/A2_1_SSTs_East_Asian_circulation/typical_patterns_of_variability')
sys.path.append(analysis_functions_dir)
import regress_map
from read_in_data import files_in_directory,read_in_variable, calculate_annual_mean, running_mean


def calculate_EAsia_rm_eofs(data,lats,lons,lat_min=20,lat_max=50,lon_min=110,lon_max=180):
    """ Calculates EOFs over the East Asian region.
    Regresses the principal components back onto the original data"""
    lat_mask = (lats>=lat_min)&(lats<=lat_max)
    lon_mask = (lons>=lon_min)&(lons<=lon_max)
    data_EAsia = data[:,lat_mask,:][:,:,lon_mask]
    # calculate EOFs
    coslat = np.cos(np.deg2rad(lats[lat_mask]))
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(data_EAsia, weights=wgts)
    var_frac = solver.varianceFraction()
    pcs = solver.pcs(npcs=3,pcscaling=1)
    # regress first modes onto original data
    reg_pc1,pval_pc1 = regress_map.regress_map(pcs[:,0],data,map_type='regress')
    reg_pc2,pval_pc2 = regress_map.regress_map(pcs[:,1],data,map_type='regress')
    reg_pc3,pval_pc3 = regress_map.regress_map(pcs[:,2],data,map_type='regress')
    return var_frac,reg_pc1,pval_pc1,reg_pc2,pval_pc2,reg_pc3,pval_pc3
    

def plot_mode(ax,field1,field2,lats,lons,clevs,clevs2,title='',title_fontsize=30,extent=[10,220,0,70],pvals=None):
    """ Plot the EOF patterns """
    plt.title(title,fontsize=title_fontsize)
    m = make_plots.make_map_plot()
    cs = m.add_filled_contours(lons,lats,field1,clevs)
    m.add_contours(lons,lats,field2,clevs2,contour_labels=False)
    make_plots.add_lat_lon(ax)
    m.geography(ax,extent=[10,220,0,70])
    make_plots.plot_box(lon_min=110,lon_max=180,lat_min=20,lat_max=50)
    ax.set_aspect(2)
    return cs

# select season
season = 'JJA'

# Read in HadGEM3 data and calculate annual means
full_path_psl = '/network/group/aopp/predict/AWH007_BEFORT_CMIP6/piControl/MOHC/HadGEM3-GC31-LL/piControl/Amon/psl/gn/latest/'
list_of_files_psl = files_in_directory(full_path_psl,concat_directory=True)
psl_HadGEM,lats_HadGEM,lons_HadGEM,levs_HadGEM,times_HadGEM,calendar_HadGEM,t_units_HadGEM = read_in_variable(list_of_files_psl[:],'psl')
psl_am_HadGEM, years = calculate_annual_mean(psl_HadGEM,times_HadGEM,calendar_HadGEM,t_units_HadGEM,season=season)
psl_am_HadGEM = 0.01 * psl_am_HadGEM # convert to hPa

# Read in ERA-Interim and calculate annual means
file_name = '/network/aopp/preds0/pred/data/Obs/Reanalysis/psl/mon/psl_mon_ERAInterim_1x1_197901-201512.nc'
psl_ERAI,lats_ERAI,lons_ERAI,levs_ERAI,times_ERAI,calendar_ERAI,t_units_ERAI = read_in_variable([file_name],'msl')
psl_am_ERAI, years = calculate_annual_mean(psl_ERAI,times_ERAI,calendar_ERAI,t_units_ERAI,season=season)
psl_am_ERAI = 0.01 * psl_am_ERAI # convert to hPa

# set up figure for plotting
plt.figure(figsize=(28,17))
gs = gridspec.GridSpec(5,3,height_ratios=[10,10,0.5,10,0.5])
a = np.arange(0.2,2.1,0.2)
clevs = np.append(-a[::-1],a)
clevs_mean = np.arange(1005,1031,5)

# calculate EOF patterns for model and reanalysis
var_frac,reg_pc1,pval_pc1,reg_pc2,pval_pc2,reg_pc3,pval_pc3 = calculate_EAsia_rm_eofs(psl_am_HadGEM,lats_HadGEM,lons_HadGEM)
ax = plt.subplot(gs[0,0],projection=ccrs.PlateCarree(central_longitude=180.))
cs = plot_mode(ax,reg_pc1,np.mean(psl_am_HadGEM,axis=0),lats_HadGEM,lons_HadGEM,clevs,clevs_mean,title='HadGEM3 EOF1 %.f%%' % (100*var_frac[0]))
ax = plt.subplot(gs[0,1],projection=ccrs.PlateCarree(central_longitude=180.))
cs = plot_mode(ax,-reg_pc2,np.mean(psl_am_HadGEM,axis=0),lats_HadGEM,lons_HadGEM,clevs,clevs_mean,title='HadGEM3 EOF2 %.f%%' % (100*var_frac[1]))
ax = plt.subplot(gs[0,2],projection=ccrs.PlateCarree(central_longitude=180.))
cs = plot_mode(ax,reg_pc3,np.mean(psl_am_HadGEM,axis=0),lats_HadGEM,lons_HadGEM,clevs,clevs_mean,title='HadGEM3 EOF3 %.f%%' % (100*var_frac[2]))

var_frac,reg_pc1,pval_pc1,reg_pc2,pval_pc2,reg_pc3,pval_pc3 = calculate_EAsia_rm_eofs(psl_am_ERAI,lats_ERAI,lons_ERAI)
ax = plt.subplot(gs[1,0],projection=ccrs.PlateCarree(central_longitude=180.))
cs = plot_mode(ax,-reg_pc1,np.mean(psl_am_ERAI,axis=0),lats_ERAI,lons_ERAI,clevs,clevs_mean,title='ERAI EOF1 %.f%%' % (100*var_frac[0]))
ax = plt.subplot(gs[1,1],projection=ccrs.PlateCarree(central_longitude=180.))
cs = plot_mode(ax,-reg_pc2,np.mean(psl_am_ERAI,axis=0),lats_ERAI,lons_ERAI,clevs,clevs_mean,title='ERAI EOF2 %.f%%' % (100*var_frac[1]))
ax = plt.subplot(gs[1,2],projection=ccrs.PlateCarree(central_longitude=180.))
cs = plot_mode(ax,-reg_pc3,np.mean(psl_am_ERAI,axis=0),lats_ERAI,lons_ERAI,clevs,clevs_mean,title='ERAI EOF3 %.f%%' % (100*var_frac[2]))

# do the same with a running mean applied
print('Calculating running means')
N = 10
psl_am_HadGEM_rm = running_mean(psl_am_HadGEM,N)
psl_am_ERAI_rm = running_mean(psl_am_ERAI,N)
clevs = 0.2*clevs

ax = plt.subplot(gs[2,:])
make_plots.colorbar(ax,cs,orientation='horizontal')

var_frac,reg_pc1,pval_pc1,reg_pc2,pval_pc2,reg_pc3,pval_pc3 = calculate_EAsia_rm_eofs(psl_am_HadGEM_rm,lats_HadGEM,lons_HadGEM)
ax = plt.subplot(gs[3,0],projection=ccrs.PlateCarree(central_longitude=180.))
cs = plot_mode(ax,reg_pc1,np.mean(psl_am_HadGEM,axis=0),lats_HadGEM,lons_HadGEM,clevs,clevs_mean,title='HadGEM 10yr RM EOF1 %.f%%' % (100*var_frac[0]))
ax = plt.subplot(gs[3,1],projection=ccrs.PlateCarree(central_longitude=180.))
cs = plot_mode(ax,-reg_pc2,np.mean(psl_am_HadGEM,axis=0),lats_HadGEM,lons_HadGEM,clevs,clevs_mean,title='HadGEM 10yr RM EOF2 %.f%%' % (100*var_frac[1]))
ax = plt.subplot(gs[3,2],projection=ccrs.PlateCarree(central_longitude=180.))
cs = plot_mode(ax,-reg_pc3,np.mean(psl_am_HadGEM,axis=0),lats_HadGEM,lons_HadGEM,clevs,clevs_mean,title='HadGEM 10yr RM EOF3 %.f%%' % (100*var_frac[2]))
"""
var_frac,reg_pc1,pval_pc1,reg_pc2,pval_pc2,reg_pc3,pval_pc3 = calculate_EAsia_rm_eofs(psl_am_ERAI_rm,lats_ERAI,lons_ERAI)
ax = plt.subplot(gs[3,0],projection=ccrs.PlateCarree(central_longitude=180.))
cs = plot_mode(ax,reg_pc1,np.mean(psl_am_HadGEM,axis=0),lats_ERAI,lons_ERAI,clevs,clevs_mean,title='HadGEM 10yr RM EOF1 %.f%%' % (100*var_frac[0]))
ax = plt.subplot(gs[3,1],projection=ccrs.PlateCarree(central_longitude=180.))
cs = plot_mode(ax,reg_pc2,np.mean(psl_am_HadGEM,axis=0),lats_ERAI,lons_ERAI,clevs,clevs_mean,title='HadGEM 10yr RM EOF2 %.f%%' % (100*var_frac[1]))
ax = plt.subplot(gs[3,2],projection=ccrs.PlateCarree(central_longitude=180.))
cs = plot_mode(ax,reg_pc3,np.mean(psl_am_HadGEM,axis=0),lats_ERAI,lons_ERAI,clevs,clevs_mean,title='HadGEM 10yr RM EOF3 %.f%%' % (100*var_frac[2]))
"""
ax = plt.subplot(gs[4,:])
make_plots.colorbar(ax,cs,orientation='horizontal')

plt.subplots_adjust(hspace=0.4)
plt.savefig(figure_dir+'/slp_patterns_HadGEM3_ERAI_JJA_EAsia_lons110_180_lats20_50.png',bbox_inches='tight')
plt.show()
