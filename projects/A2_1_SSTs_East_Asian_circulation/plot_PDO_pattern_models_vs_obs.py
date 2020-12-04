""" Plot figures of PDO pattern a few models and in multimodel mean """

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import sys
from matplotlib import gridspec

# my modules
cwd = os.getcwd()
repo_dir = '/'
for directory in cwd.split('/')[1:]:
    repo_dir = os.path.join(repo_dir, directory)
    if directory == 'postdoc':
        break

import make_plots
import model_list as ML
analysis_functions_dir = os.path.join(repo_dir,'analysis_functions')
figure_dir = os.path.join(repo_dir,'figures/A2_1_SSTs_East_Asian_circulation/regress_SST_indices_circulation')
loading_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/SST_index_data')
sys.path.append(analysis_functions_dir)

from read_in_data import files_in_directory, read_in_variable, calculate_annual_mean, running_mean, save_file, read_spatial_dimensions, read_time_dimension, prep_SST_index
from interpolate_grid import interpolate_grid

season = 'ANN'
model_name_list = ML.model_name_list #['HadGEM3-GC31-LL']#['BCC-CSM2-MR'] #ML.model_name_list
model_name_list.append('ERA20C')

def get_PDO_pattern(SST_index_file_name):
    nc_SST = Dataset(SST_index_file_name,'r')
    times_SST = nc_SST.variables['times'][:]
    calendar_SST = nc_SST.variables['times'].calendar
    t_units_SST = nc_SST.variables['times'].units
    PDO_pattern = nc_SST.variables['reg_coeff_NINO34'][:]
    lats = nc_SST.variables['lats'][:]
    lons = nc_SST.variables['lons'][:]
    return PDO_pattern,lats,lons


def contour_plot(field,clevs,lats,lons,title_fontsize=20,title='',cmap='RdBu_r'):
    m = make_plots.make_map_plot()
    #ax.add_feature(cfeature.LAND)
    cs = m.add_filled_contours(lons,lats,field,clevs,cmap=cmap)
    plt.title(title,fontsize=title_fontsize)
    ax.coastlines()
    #ax.add_feature(cfeature.LAND)
    return cs


plt.figure(figsize=(18,10))
gs = gridspec.GridSpec(3,2,height_ratios=[10,10,0.5])

# contour levels
a = np.arange(0.1,0.81,0.1)
clevs = np.append(-a[::-1],a)

common_lats = np.arange(-90,90.1,1)
common_lons = np.arange(0,360.1,1)
PDO_patterns_all = np.zeros([0,common_lats.shape[0],common_lons.shape[0]])

for i, model_name in enumerate(model_name_list):
    
    try:
        # read in SST index data
        if model_name == 'ERA20C': SST_index_file_name = loading_dir + '/SST_indices_'+model_name+'.nc'
        else: SST_index_file_name = loading_dir + '/SST_indices_'+model_name+'_'+ ML.ensemble_id[model_name]+'.nc'
        PDO_pattern,lats,lons = get_PDO_pattern(SST_index_file_name)

        if model_name != 'ERA20C':
            PDO_pattern_interp = interpolate_grid(PDO_pattern,lons,lats,common_lons,common_lats)
            PDO_patterns_all = np.append(PDO_patterns_all,PDO_pattern_interp.reshape(1,common_lats.shape[0],common_lons.shape[0]),axis=0)
        else: 
            ax = plt.subplot(gs[0,0],projection=ccrs.Robinson(central_longitude=180.))
            cs = contour_plot(PDO_pattern,clevs,lats,lons,title_fontsize=20,title='ERA20C')
        
        if model_name == 'CESM2':
            ax = plt.subplot(gs[1,0],projection=ccrs.Robinson(central_longitude=180.))
            cs = contour_plot(PDO_pattern,clevs,lats,lons,title_fontsize=20,title=model_name)
        elif model_name == 'CNRM-CM6-1-HR':
            ax = plt.subplot(gs[1,1],projection=ccrs.Robinson(central_longitude=180.))
            cs = contour_plot(PDO_pattern,clevs,lats,lons,title_fontsize=20,title=model_name)

    except:
        print('Error, skipping model %s' % (model_name))


# plotting
#print(PDO_patterns_all,PDO_pattern)
ax = plt.subplot(gs[0,1],projection=ccrs.Robinson(central_longitude=180.))
cs = contour_plot(np.nanmean(PDO_patterns_all,axis=0),clevs,common_lats,common_lons,title_fontsize=20,title='Ensemble-mean',cmap='RdBu_r')

ax = plt.subplot(gs[2,:])
make_plots.colorbar(ax,cs,orientation='horizontal')

figure_name = figure_dir + '/NINO34_pattern_obs_models_'+season+'.png'
print('saving to %s' % (figure_name))
plt.subplots_adjust(wspace=0.1)
plt.savefig(figure_name,bbox_inches='tight')
