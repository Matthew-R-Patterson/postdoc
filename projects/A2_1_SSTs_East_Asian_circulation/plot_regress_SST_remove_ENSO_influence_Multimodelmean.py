""" Regress indices of Indian Ocean and ENSO variability onto circulation variables after
regressing out the influence of the other. 
e.g. regress Indian Ocean index onto zonal wind after regressing out Nino3.4 index. """

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import os
import sys
from scipy.stats import linregress
from matplotlib import gridspec
import cartopy.crs as ccrs

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
saving_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/regress_SST_indices_circulation_data')
figure_dir = os.path.join(repo_dir,'figures/A2_1_SSTs_East_Asian_circulation/regress_SST_indices_circulation')
sys.path.append(analysis_functions_dir)
import regress_map
from interpolate_grid import interpolate_grid
from read_in_data import files_in_directory, read_spatial_dimensions, read_time_dimension, read_in_variable, calculate_annual_mean, running_mean, save_file


model_name_list = ML.model_name_list #['HadGEM3-GC31-LL']#['BCC-CSM2-MR'] #ML.model_name_list
#model_name_list.append('ERA20C')

def prep_SST_index(index_name,nc,times_SST,calendar_SST,t_units_SST,season):
    """Read in particular SST index and calculate annual mean"""
    index_ts = nc.variables[index_name][:]
    index_annual_mean, years = calculate_annual_mean(index_ts,times_SST,calendar_SST,t_units_SST,season=season)
    return index_annual_mean, years

def regress_out_index(index1,index2):
    """regress index2 out of index1 such that the resultant index
    is uncorrelated with index2."""
    x = np.arange(index1.shape[0])
    slope = linregress(index2,index1)[0]
    new_index1 = index1 - slope * index2
    return new_index1 

def plot_reg(ax,field,lats,lons,clevs,field2=None,clevs2=None,title='',title_fontsize=30,extent=[10,220,0,70],pvals=None):
    """ Plot the regression patterns """
    plt.title(title,fontsize=title_fontsize)
    m = make_plots.make_map_plot()
    cs = m.add_filled_contours(lons,lats,field,clevs)
    if field2 is not None: m.add_contours(lons,lats,field2,clevs2,colors='0.75')
    #make_plots.add_lat_lon(ax,fontsize=15)
    m.geography(ax,extent=[0,359.99,-80,80]) #extent=[30,330,-15,70])
    #make_plots.plot_box(lon_min=110,lon_max=180,lat_min=20,lat_max=50)
    #ax.set_aspect(1.5)
    return cs

def do_reg(SST_index,data_all):
    """ Do regression, but only include statistically significant values"""
    regress_coeff, pvals = regress_map.regress_map(SST_index,data_all,map_type='regress')
    #pvals = t_test_regression(data_all,regress_coeff)
    #mask = np.ones_like(pvals)
    #mask[pvals>0.05] = np.nan
    #regress_coeff = regress_coeff * mask
    return regress_coeff

def get_reg_coeffs(nc,var_name,vars_all,lats_old,lons_old,lats_new,lons_new):
    """ Read in variable and interpolate onto common grid, then append to vars_all """
    var = nc.variables[var_name][:]
    var_interp = interpolate_grid(var,lons,lats,lons_new,lats_new)
    vars_all = np.append(vars_all,var_interp.reshape(1,common_lats.shape[0],common_lons.shape[0]),axis=0)
    return vars_all


chosen_level = 20000
var_name = 'ua'
apply_running_mean = None
season = 'JJA'

# define common latitudes and longitudes for plotting
common_lats = np.arange(-90,90.1,1)
common_lons = np.arange(0,360.1,1)

# set up empty regression coefficient arrays
regress_coeff_IOBM2_ua200_all = np.zeros([0,common_lats.shape[0],common_lons.shape[0]])
regress_coeff_IOBM2_no_NINO34_ua200_all = np.zeros([0,common_lats.shape[0],common_lons.shape[0]])
regress_coeff_IOD_ua200_all = np.zeros([0,common_lats.shape[0],common_lons.shape[0]])
regress_coeff_IOD_no_NINO34_ua200_all = np.zeros([0,common_lats.shape[0],common_lons.shape[0]])
regress_coeff_NINO34_ua200_all = np.zeros([0,common_lats.shape[0],common_lons.shape[0]])
U_climatology_all = np.zeros([0,common_lats.shape[0],common_lons.shape[0]])

for i, model_name in enumerate(model_name_list):

    try:
        # check if a regression file is available and if so read it in
        if apply_running_mean is not None: regression_file_name = saving_dir + '/regression_' + model_name + '_ua200_SST__NINO34_IOBM2_IOD_'+season+'_rm'+str(apply_running_mean)+'.nc'
        else: regression_file_name = saving_dir + '/regression_' + model_name + '_ua200_SST__NINO34_IOBM2_IOD_'+season+'.nc'
        nc = Dataset(regression_file_name,'r')
        lats = nc.variables['lats_ua'][:]
        lons = nc.variables['lons_ua'][:]
        regress_coeff_IOBM2_ua200_all = get_reg_coeffs(nc,'regress_coeff_IOBM2_ua200',regress_coeff_IOBM2_ua200_all,lats,lons,common_lats,common_lons)
        regress_coeff_IOBM2_no_NINO34_ua200_all = get_reg_coeffs(nc,'regress_coeff_IOBM2_no_NINO34_ua200',regress_coeff_IOBM2_no_NINO34_ua200_all,lats,lons,common_lats,common_lons)
        regress_coeff_IOD_ua200_all = get_reg_coeffs(nc,'regress_coeff_IOD_ua200',regress_coeff_IOD_ua200_all,lats,lons,common_lats,common_lons)
        regress_coeff_IOD_no_NINO34_ua200_all = get_reg_coeffs(nc,'regress_coeff_IOD_no_NINO34_ua200',regress_coeff_IOD_no_NINO34_ua200_all,lats,lons,common_lats,common_lons)
        regress_coeff_NINO34_ua200_all = get_reg_coeffs(nc,'regress_coeff_NINO34_ua200',regress_coeff_NINO34_ua200_all,lats,lons,common_lats,common_lons)
        U_climatology_all = get_reg_coeffs(nc,'U_climatology',U_climatology_all,lats,lons,common_lats,common_lons)

        print('Reading in data from %s' % (regression_file_name))


    except:
        # if not read in files and perform regression

        try:

            # read in SST data
            SST_index_file_name = loading_dir + '/SST_indices_'+model_name+'_'+ML.ensemble_id[model_name]+'.nc'
            nc_SST = Dataset(SST_index_file_name,'r')
            times_SST = nc_SST.variables['times'][:]
            calendar_SST = nc_SST.variables['times'].calendar
            t_units_SST = nc_SST.variables['times'].units
            IOBM2, years = prep_SST_index('IOBM2',nc_SST,times_SST,calendar_SST,t_units_SST,season)
            IOD, years = prep_SST_index('IOD',nc_SST,times_SST,calendar_SST,t_units_SST,season)
            NINO34, years = prep_SST_index('NINO34',nc_SST,times_SST,calendar_SST,t_units_SST,season)

            # apply running mean
            if apply_running_mean is not None:
                IOBM2 = running_mean(IOBM2,apply_running_mean)
                IOD = running_mean(IOD,apply_running_mean)
                NINO34 = running_mean(NINO34,apply_running_mean)

            # regress out other index
            IOBM2_no_NINO34 = regress_out_index(IOBM2,NINO34)
            IOD_no_NINO34 = regress_out_index(IOD,NINO34)
            NINO34_no_IOBM2 = regress_out_index(NINO34,IOBM2)

            # read in circulation indices
            data_path0 = '/network/aopp/hera/mad/patterson/CMIP6/data/piControl/ua/'
            data_path1 = '/' + var_name + '/'
            full_path = data_path0 + ML.model_institute[model_name] + '/' + model_name + data_path1
            list_of_files = [full_path+var_name+'_Amon_'+model_name+'_piControl_'+ML.ensemble_id[model_name]+'_'+season+'.nc']
            ua200,lats_ua,lons_ua,levs_ua,times_ua,calendar_ua,t_units_ua = read_in_variable(list_of_files[:],var_name,chosen_level=20000)
            U_climatology = np.mean(ua200,axis=0)

            if season == 'DJF': 
                ua200 = ua200[::-1][1:][::-1]
                
            if apply_running_mean is not None:
                ua200 = running_mean(ua200,apply_running_mean)

            # calculate regressions
            regress_coeff_IOBM2_ua200 = do_reg(IOBM2,ua200)
            regress_coeff_IOBM2_no_NINO34_ua200 = do_reg(IOBM2_no_NINO34,ua200)
            regress_coeff_IOD_ua200 = do_reg(IOD,ua200)
            regress_coeff_IOD_no_NINO34_ua200 = do_reg(IOD_no_NINO34,ua200)
            regress_coeff_NINO34_ua200 = do_reg(NINO34,ua200)

            # save regressions
            if apply_running_mean is not None:
                f = saving_dir + '/regression_' + model_name + '_ua200_SST__NINO34_IOBM2_IOD_'+season+'_rm'+str(apply_running_mean)+'.nc'
            else: 
                f = saving_dir + '/regression_' + model_name + '_ua200_SST__NINO34_IOBM2_IOD_'+season+'.nc'
            description = 'Regressions of SST indices onto 200hPa zonal wind for the model: ' + model_name
            save = save_file(f,description)
            save.add_dimension(lats_ua,'lats_ua')
            save.add_dimension(lons_ua,'lons_ua')
            save.add_times(times_ua,calendar_ua,t_units_ua,time_name='times')
            save.add_variable(U_climatology,'U_climatology',('lats_ua','lons_ua'))
            save.add_variable(regress_coeff_IOBM2_ua200,'regress_coeff_IOBM2_ua200',('lats_ua','lons_ua',))
            save.add_variable(regress_coeff_IOBM2_no_NINO34_ua200,'regress_coeff_IOBM2_no_NINO34_ua200',('lats_ua','lons_ua',))
            save.add_variable(regress_coeff_IOD_ua200,'regress_coeff_IOD_ua200',('lats_ua','lons_ua',))
            save.add_variable(regress_coeff_IOD_no_NINO34_ua200,'regress_coeff_IOD_no_NINO34_ua200',('lats_ua','lons_ua',))
            save.add_variable(regress_coeff_NINO34_ua200,'regress_coeff_NINO34_ua200',('lats_ua','lons_ua',))
            save.close_file()
            print('saved to %s' % (f))

            # interpolate onto common grid
            regress_coeff_IOBM2_ua200 = interpolate_grid(regress_coeff_IOBM2_ua200,lons_ua,lats_ua,common_lons,common_lats)

            # append
            regress_coeff_IOBM2_ua200_all = np.append(regress_coeff_IOBM2_ua200_all,regress_coeff_IOBM2_ua200.reshape(1,common_lats.shape[0],common_lons.shape[0]),axis=0)
            regress_coeff_IOBM2_no_NINO34_ua200_all = np.append(regress_coeff_IOBM2_no_NINO34_ua200_all,regress_coeff_IOBM2_no_NINO34_ua200.reshape(1,common_lats.shape[0],common_lons.shape[0]),axis=0)
            regress_coeff_IOD_ua200_all = np.append(regress_coeff_IOD_ua200_all,regress_coeff_IOD_ua200.reshape(1,common_lats.shape[0],common_lons.shape[0]),axis=0)
            regress_coeff_IOD_no_NINO34_ua200_all = np.append(regress_coeff_IOD_no_NINO34_ua200_all,regress_coeff_IOD_no_NINO34_ua200.reshape(1,common_lats.shape[0],common_lons.shape[0]),axis=0)
            regress_coeff_NINO34_ua200_all = np.append(regress_coeff_NINO34_ua200_all,regress_coeff_NINO34_ua200.reshape(1,common_lats.shape[0],common_lons.shape[0]),axis=0)
            U_climatology_all = np.append(U_climatology_all,U_climatology.reshape(1,common_lats.shape[0],common_lons.shape[0]),axis=0)

        except:
            message = 'Error, skipping model ' + model_name +' for regression analysis'
            print(message)

    


# plot
plt.figure(figsize=(25,10))
#plt.figure(figsize=(20,11))
gs = gridspec.GridSpec(3,6,height_ratios=[10,10,0.5])

a = np.arange(0.4,3.1,0.4)
clevs = np.append(-a[::-1],a)
a = np.arange(5,61,5)
clevs_clim = np.append(-a[::-1],a)

ax = plt.subplot(gs[0,0:2],projection=ccrs.Robinson(central_longitude=180.))
cs = plot_reg(ax,np.mean(regress_coeff_NINO34_ua200_all,axis=0),common_lats,common_lons,clevs,field2=np.mean(U_climatology_all,axis=0),clevs2=clevs_clim,title='NINO3.4')

ax = plt.subplot(gs[0,2:4],projection=ccrs.Robinson(central_longitude=180.))
cs = plot_reg(ax,np.mean(regress_coeff_IOBM2_ua200_all,axis=0),common_lats,common_lons,clevs,field2=np.mean(U_climatology_all,axis=0),clevs2=clevs_clim,title='IOBM')

ax = plt.subplot(gs[0,4:],projection=ccrs.Robinson(central_longitude=180.))
cs = plot_reg(ax,np.mean(regress_coeff_IOBM2_no_NINO34_ua200_all,axis=0),common_lats,common_lons,clevs,field2=np.mean(U_climatology_all,axis=0),clevs2=clevs_clim,title='IOBM, rm NINO3.4')

ax = plt.subplot(gs[1,1:3],projection=ccrs.Robinson(central_longitude=180.))
cs = plot_reg(ax,np.mean(regress_coeff_IOD_no_NINO34_ua200_all,axis=0),common_lats,common_lons,clevs,field2=np.mean(U_climatology_all,axis=0),clevs2=clevs_clim,title='IOD')

ax = plt.subplot(gs[1,3:5],projection=ccrs.Robinson(central_longitude=180.))
cs = plot_reg(ax,np.mean(regress_coeff_IOD_no_NINO34_ua200_all,axis=0),common_lats,common_lons,clevs,field2=np.mean(U_climatology_all,axis=0),clevs2=clevs_clim,title='IOD, rm NINO3.4')

ax = plt.subplot(gs[2,1:5])
make_plots.colorbar(ax,cs,orientation='horizontal')
plt.subplots_adjust(hspace=0.2)
figure_name = figure_dir + '/ua_IOBM2_regress_out_NINO34_multimodelmean_global.png'
if apply_running_mean is not None: figure_name = figure_dir + '/ua_IOBM2_regress_out_NINO34_multimodelmean_rm'+str(apply_running_mean)+'_global.png'
print('saving to %s' % (figure_name))
plt.savefig(figure_name,bbox_inches='tight')
