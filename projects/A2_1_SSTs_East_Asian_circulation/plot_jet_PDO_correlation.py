""" Calculate EOFs of upper level zonal wind in the East Asian region on either
interannual of decadal timescales and regress onto SST in a range of models."""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
from scipy import stats
import cartopy.crs as ccrs
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
loading_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/SST_index_data')
saving_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/jet_EOF_regression_data')
figure_dir = os.path.join(repo_dir,'figures/A2_1_SSTs_East_Asian_circulation/jet_EOFs')
sys.path.append(analysis_functions_dir)

from read_in_data import files_in_directory, read_in_variable, calculate_annual_mean, running_mean, save_file, read_spatial_dimensions, read_time_dimension
from regress_map import regress_map
from significance_testing import t_test_regression,calc_autocorr
from interpolate_grid import interpolate_grid

# define East Asian region
lon_min, lon_max = 70,150
lat_min, lat_max = 20,50
season = 'JJA'
apply_running_mean = None #None
plot_multimodelmean = False
model_name_list = ML.model_name_list #['BCC-CSM2-MR','CESM2','CNRM-CM6-1-HR','HadGEM3-GC31-LL','MPI-ESM1-2-LR']  #ML.model_name_list
#model_name_list.append('ERA20C')


def prep_SST_index(index_name,nc,times_SST,calendar_SST,t_units_SST,season):
    """Read in particular SST index and calculate annual mean"""
    index_ts = nc.variables[index_name][:]
    index_annual_mean, years = calculate_annual_mean(index_ts,times_SST,calendar_SST,t_units_SST,season=season)
    return index_annual_mean, years

def change_index_sign(index,field,lats,lons,bounds,text=''):
    # remove unreasonably large values
    field[np.abs(field)>1e3] = np.nan
    lat_min,lat_max = bounds[0],bounds[1]
    lon_min,lon_max = bounds[2],bounds[3]
    lat_mask = (lats>=lat_min)&(lats<=lat_max)
    lon_mask = (lons>=lon_min)&(lons<=lon_max)
    sign_changed=False
    if np.mean(field[lat_mask,:][:,lon_mask]) < 0: 
        index = - index
        print('sign changed '+text)
        sign_changed=True
    return index,sign_changed


plt.figure(figsize=(20,15)) # 20,16
gs = gridspec.GridSpec(3,4,height_ratios=[10,10,0.5])

corr_all_pc1 = np.array([])
corr_all_pc2 = np.array([])
common_lats = np.arange(-90,90.1,1)
common_lons = np.arange(0,360.1,1)
U_pc1_map = np.zeros([0,common_lats.shape[0],common_lons.shape[0]])
U_pc2_map = np.zeros([0,common_lats.shape[0],common_lons.shape[0]])
U_clim_all = np.zeros([0,common_lats.shape[0],common_lons.shape[0]])

for i, model_name in enumerate(model_name_list):

    try:    
        # check if a regression file exists and read in data
        if apply_running_mean is not None: regression_file_name = saving_dir + '/regression_' + model_name + '_ua200_EOFs_SST_rm' + str(apply_running_mean) + '_' + season + '_corr.nc'
        else: regression_file_name = saving_dir + '/regression_' + model_name + '_ua200_EOFs_SST_'+season+'.nc'
        levs,lats_ua,lons_ua = read_spatial_dimensions(regression_file_name,lat_name='lats_ua',lon_name='lons_ua')
        levs,lats_SST,lons_SST = read_spatial_dimensions(regression_file_name,lat_name='lats_SST',lon_name='lons_SST')
        levs,lats_THF,lons_THF = read_spatial_dimensions(regression_file_name,lat_name='lats_THF',lon_name='lons_THF')
        times,calendar,units = read_time_dimension(regression_file_name)
        nc = Dataset(regression_file_name,'r')
        pcs = nc.variables['pcs'][:]
        variance_fraction = nc.variables['variance_fraction'][:]
        print('Read in %s' % (regression_file_name))
    
        # read in SST data
        SST_index_file_name = loading_dir + '/SST_indices_'+model_name+'_'+ML.ensemble_id[model_name]+'.nc'
        nc_SST = Dataset(SST_index_file_name,'r')
        times_SST = nc_SST.variables['times'][:]
        calendar_SST = nc_SST.variables['times'].calendar
        t_units_SST = nc_SST.variables['times'].units
        PDO, years = prep_SST_index('NINO34',nc_SST,times_SST,calendar_SST,t_units_SST,season)
        if apply_running_mean is not None: PDO = running_mean(PDO,apply_running_mean)

        # make sure that all indices have the same sign and add EOF maps to array
        if model_name in ['CNRM-ESM2-1','CNRM-CM6-1','INM-CM5-0']: pc_number = 1 # these models have the first 2 EOFs the other way round
        else: pc_number = 0 
        U = nc.variables['regress_U'][pc_number,:,:]
        U_pc,sign_changed = change_index_sign(pcs[:,pc_number],U,lats_ua,lons_ua,bounds=[25,35,100,160],text='U')   
        if sign_changed == True: U = -U
        SST = nc_SST.variables['reg_coeff_NINO34'][:]
        #if model_name == 'EC-Earth3': PDO = -PDO # EC-Earth sign is negative but not changed by change sign function
        #else: PDO,sign_changed = change_index_sign(PDO,SST,lats_SST,lons_SST,bounds=[-10,10,230,260],text='PDO') # [30,40,140,180]        
        corr = stats.linregress(PDO,U_pc)[2]
        if corr < 0 : print('%s is less than 0' % (model_name))
        corr_all_pc1 = np.append(corr_all_pc1,corr)
        U_pc1_map = np.append(U_pc1_map,interpolate_grid(U,lons_ua,lats_ua,common_lons,common_lats).reshape(1,common_lats.shape[0],common_lons.shape[0]),axis=0)        

        if model_name in ['CNRM-ESM2-1','CNRM-CM6-1','INM-CM5-0']: pc_number = 0 # these models have the first 2 EOFs the other way round
        else: pc_number = 1
        U = nc.variables['regress_U'][pc_number,:,:]
        U_pc, sign_changed = change_index_sign(pcs[:,pc_number],U,lats_ua,lons_ua,bounds=[30,45,100,160],text='U')
        if sign_changed == True: U = -U
        corr = stats.linregress(PDO,U_pc)[2]
        corr_all_pc2 = np.append(corr_all_pc2,corr)
        U_pc2_map = np.append(U_pc2_map,interpolate_grid(U,lons_ua,lats_ua,common_lons,common_lats).reshape(1,common_lats.shape[0],common_lons.shape[0]),axis=0)

        # add U climatology
        U_clim = nc.variables['U_climatology'][:]
        U_clim_all = np.append(U_clim_all,interpolate_grid(U_clim,lons_ua,lats_ua,common_lons,common_lats).reshape(1,common_lats.shape[0],common_lons.shape[0]),axis=0)



    except:
        print('Skipping ' + model_name)

ax = plt.subplot(gs[0,1:3])
ax.boxplot([corr_all_pc1,corr_all_pc2])
plt.xticks([1,2],['PC1','PC2'],fontsize=20)
plt.yticks(fontsize=20)
#plt.xtick_labels(['PC1','PC2'],[1,2])
plt.title('Corr(U200 PC,NINO34)',fontsize=20)
plt.ylabel('Correlation',fontsize=20)

a = np.arange(0.2,0.91,0.2)
clevs = np.append(-a[::-1],a)
a = np.arange(5,61,5)
clim_clevs = np.append(-a[::-1],a)

ax = plt.subplot(gs[1,0:2],projection=ccrs.PlateCarree(central_longitude=180.))
cs = make_plots.contour_plot(ax,np.mean(U_pc1_map,axis=0),clevs,common_lats,common_lons,climatology=np.mean(U_clim_all,axis=0),clim_clevs=clim_clevs,calculate_significance=False,title='U PC1 corr. ',extent=[70,240,0,60],plot_box=True)

ax = plt.subplot(gs[1,2:],projection=ccrs.PlateCarree(central_longitude=180.))
cs = make_plots.contour_plot(ax,np.mean(U_pc2_map,axis=0),clevs,common_lats,common_lons,climatology=np.mean(U_clim_all,axis=0),clim_clevs=clim_clevs,calculate_significance=False,title='U PC2 corr.',extent=[70,240,0,60],plot_box=True)

ax = plt.subplot(gs[2,:])
make_plots.colorbar(ax,cs,orientation='horizontal')


if apply_running_mean is not None: figure_name = figure_dir + '/correlation_NINO34_U200_EOFs_boxplots_'+season+'_rm'+str(apply_running_mean)+'.png'
else: figure_name = figure_dir + '/correlation_NINO34_U200_EOFs_boxplots_'+season+'.png'

print('saving to %s' % (figure_name))
plt.savefig(figure_name,bbox_inches='tight')
plt.show()
