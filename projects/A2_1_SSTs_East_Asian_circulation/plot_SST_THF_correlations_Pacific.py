""" Calculate correlations of SST and THF at gridpoints over the Pacific region """

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
#saving_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/jet_EOF_regression_data')
figure_dir = os.path.join(repo_dir,'figures/A2_1_SSTs_East_Asian_circulation/SST_THF_correlations')
sys.path.append(analysis_functions_dir)

from read_in_data import files_in_directory, read_in_variable, calculate_annual_mean, running_mean, save_file, read_spatial_dimensions, read_time_dimension
from regress_map import regress_map
from significance_testing import confidence_intervals

season = 'DJF'
apply_running_mean = 11 #None #11 #None
model_name_list = ['BCC-CSM2-MR','CESM2','CNRM-CM6-1-HR','HadGEM3-GC31-LL']#,'MPI-ESM1-2-LR']  ##ML.model_name_list 
model_name_list = ['BCC-CSM2-MR','FGOALS-g3','CanESM5','CNRM-CM6-1','ACCESS-CM2','INM-CM4-8','MIROC6','HadGEM3-GC31-LL','MPI-ESM1-2-LR','MRI-ESM2-0','GISS-E2-1-G','CESM2']
#model_name_list.append('ERA20C')

def regress_gridpoint(SST,THF,lats_SST,lons_SST,lats_THF,lons_THF):
    """ Calculate the correlation between THF and SST at each grid point or 
    average THF and SST over 5x5 degree boxes and calculate correlations"""
    print('Calculating correlations')
    lon_min,lon_max = 0,359
    lat_min,lat_max = -70,70
    bottom_lat_bounds = np.arange(lat_min,lat_max,5)
    left_lon_bounds = np.arange(lon_min,lon_max,5)
    corr_map = np.zeros([bottom_lat_bounds.shape[0],left_lon_bounds.shape[0]])       
    #print(corr_map.shape)
    for i, blat in enumerate(bottom_lat_bounds):
        for j,blon in enumerate(left_lon_bounds):
            lat_mask_SST = (lats_SST>=blat)&(lats_SST<=blat+5)
            lon_mask_SST = (lons_SST>=blon)&(lons_SST<=blon+5)
            lat_mask_THF = (lats_THF>=blat)&(lats_THF<=blat+5)
            lon_mask_THF = (lons_THF>=blon)&(lons_THF<=blon+5)
            SST_box = np.nanmean(np.nanmean(SST[:,lat_mask_SST,:][:,:,lon_mask_SST],axis=1),axis=1)
            THF_box = np.nanmean(np.nanmean(THF[:,lat_mask_THF,:][:,:,lon_mask_THF],axis=1),axis=1)
            corr =  stats.linregress(SST_box,THF_box)[2]
            corr_map[i,j] = corr
    return corr_map,bottom_lat_bounds,left_lon_bounds

def contour_plot(field,clevs,lats,lons,climatology=None,clim_clevs=None,lats_ua=None,lons_ua=None,calculate_significance=True,title_fontsize=20,title='',cmap='RdBu_r',extent=[0,359.99,-10,60]):
    m = make_plots.make_map_plot()
    if calculate_significance==True:
        pvals = t_test_autocorr(data_all,data_all[mask,:,:],autocorr=0)
        field = np.mean(data_all[mask],axis=0)
        cs = m.add_filled_contours(lons,lats,field,clevs,cmap=cmap)
        plt.contour(lons,lats,pvals,np.array([0.05]),colors='k',transform=ccrs.PlateCarree())
        title = title + ' (' + str(data_all[mask].shape[0]) + ')'
    else: cs = m.add_filled_contours(lons,lats,field,clevs,cmap=cmap)
    if climatology is not None: plt.contour(lons_ua,lats_ua,climatology,clim_clevs,colors='0.75',transform=ccrs.PlateCarree())
    plt.title(title,fontsize=title_fontsize)
    make_plots.add_lat_lon(ax)
    m.geography(ax,extent=extent)
    ax.set_aspect(1.5)
    return cs


#plt.figure(figsize=(20,10)) # 20,16
#gs = gridspec.GridSpec(3,2,height_ratios=[10,10,0.5])#len(model_name_list)+1,3,height_ratios=len(model_name_list)*[10]+[0.5])
plt.figure(figsize=(30,12)) # 20,16
gs = gridspec.GridSpec(4,4,height_ratios=[10,10,10,0.5])

# contour levels
a = np.arange(0.1,0.81,0.1)
clevs = np.append(-a[::-1],a)

for i, model_name in enumerate(model_name_list):

    try:
        # read in SST and THF perform correlation analysis
        if model_name != 'ERA20C':
            data_path0 = '/network/aopp/hera/mad/patterson/CMIP6/data/tos_regridded/'
            full_path = data_path0 + ML.model_institute[model_name] + '/' + model_name + '/tos/'
            list_of_files = files_in_directory(full_path,concat_directory=True,include_files_with=ML.ensemble_id[model_name])
            tos_data,lats_SST,lons_SST,levs_SST,times_SST,calendar_SST,t_units_SST = read_in_variable(list_of_files[:],'tos')
        else:
            file_name = ['/network/group/aopp/met_data/MET003_ERA20C/data/tos/mon/tos_mon_ERA20C_2.5x2.5_189002-201012.nc']
            tos_data,lats_SST,lons_SST,levs_SST,times,calendar,t_units = read_in_variable(file_name,'sst')
        SST_am, years_SST = calculate_annual_mean(tos_data,times_SST,calendar_SST,t_units_SST,season=season)

        # set all unreasonably large values to NaN
        SST_am[np.abs(SST_am) > 1e3] = np.nan

        # read in THF
        if model_name != 'ERA20C':
            data_path0 = '/network/aopp/hera/mad/patterson/CMIP6/data/piControl/'
            full_path = data_path0 + 'hfls/'+ ML.model_institute[model_name] + '/' + model_name + '/hfls/'
            list_of_files = [full_path+'hfls_Amon_'+model_name+'_piControl_'+ML.ensemble_id[model_name]+'_'+season+'.nc']
            hfls,lats_var,lons_var,levs_var,times_var,calendar_var,t_units_var = read_in_variable(list_of_files[:],'hfls')
            full_path = data_path0 + 'hfss/' + ML.model_institute[model_name] + '/' + model_name + '/hfss/'
            list_of_files = [full_path+'hfss_Amon_'+model_name+'_piControl_'+ML.ensemble_id[model_name]+'_'+season+'.nc']
            hfss,lats_THF,lons_THF,levs_THF,times_THF,calendar_THF,t_units_THF = read_in_variable(list_of_files[:],'hfss')
        THF_am = hfls + hfss
        #THF_am, years_THF = calculate_annual_mean(THF,times_THF,calendar_THF,t_units_THF,season=season)
           
        if season == 'DJF':
            THF_am = THF_am[::-1][1:][::-1]

        if apply_running_mean is not None:
            SST_am = running_mean(SST_am,apply_running_mean)
            THF_am = running_mean(THF_am,apply_running_mean)

        # calculate correlation map
        corr_map,bottom_lat_bounds,left_lon_bounds =  regress_gridpoint(SST_am,THF_am,lats_SST,lons_SST,lats_THF,lons_THF)

        # plotting
        ax = plt.subplot(gs[i],projection=ccrs.Robinson(central_longitude=180.))
        plt.title(model_name,fontsize=20)
        cs = plt.pcolormesh(left_lon_bounds,bottom_lat_bounds,corr_map,cmap='RdBu_r',vmin=-0.9,vmax=0.9,transform=ccrs.PlateCarree())
        ax.coastlines()
        #make_plots.add_lat_lon(ax)
    except:
        print('Skipping ' + model_name)

    ax = plt.subplot(gs[3,:])
    make_plots.colorbar(ax,cs,orientation='horizontal')


if apply_running_mean is not None: figure_name = figure_dir + '/SST_THF_correlation_map_individual_models_'+season+'_rm'+str(apply_running_mean)+'_2.png'
else: figure_name = figure_dir + '/SST_THF_correlation_map_individual_models_'+season+'.png'

print('saving to %s' % (figure_name))
plt.savefig(figure_name,bbox_inches='tight')
#plt.show()
