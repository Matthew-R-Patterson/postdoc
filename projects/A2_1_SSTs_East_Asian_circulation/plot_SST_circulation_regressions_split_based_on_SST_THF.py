""" Regress Indian Ocean / Pacific SST anomalies onto the zonal wind field
for periods when SST and THF anomalies are positively correlated and for 
time when they are negatively correlated. """

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
saving_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/jet_EOF_regression_data')
loading_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/SST_index_data')
figure_dir = os.path.join(repo_dir,'figures/A2_1_SSTs_East_Asian_circulation/jet_EOFs')
sys.path.append(analysis_functions_dir)

from read_in_data import files_in_directory, read_in_variable, calculate_annual_mean, running_mean, save_file, read_spatial_dimensions, read_time_dimension, prep_SST_index
from regress_map import regress_map
from significance_testing import confidence_intervals, t_test_regression

season = 'JJA'
apply_running_mean = 11 #11 #11 #11 #11 #11 #11 #11 #11 #11 #11 #11 #None #11 #None
plot_multimodelmean = False
model_name_list = ['EC-Earth3']#['HadGEM3-GC31-LL']#['BCC-CSM2-MR'] #ML.model_name_list

def detrend(index):
    x = np.arange(index.shape[0])
    slope = stats.linregress(x,index)[0]
    index = index - slope * x
    return index

def create_THF_index(THF,lats,lons,lat_min,lat_max,lon_min,lon_max):
    """ calculate an area average of THF """
    lat_mask = (lats>=lat_min)&(lats<=lat_max)
    lon_mask = (lons>=lon_min)&(lons<=lon_max)
    THF_index = np.mean(np.mean(THF[:,lat_mask,:][:,:,lon_mask],axis=1),axis=1)
    THF_index = (THF_index - np.mean(THF_index))/np.std(THF_index)    # normalise
    THF_index = detrend(THF_index)
    return THF_index

def do_reg(index,data_all):
    """ Do regression, but only include statistically significant values"""
    regress_coeff, pvals = regress_map(index,data_all,map_type='regress')
    pvals = t_test_regression(data_all,regress_coeff)
    mask = np.ones_like(pvals)
    mask[pvals>0.05] = np.nan
    #regress_coeff = regress_coeff * mask
    return regress_coeff



for i, model_name in enumerate(model_name_list):

    try:
        # check if a regression file exists and if so read it in
        if apply_running_mean is not None: regression_file_name = saving_dir + '/regression_' + model_name + '_SST_indices_split_corr_SST_THF_rm' + str(apply_running_mean) + '_' + season + '.nc'
        else: regression_file_name = saving_dir + '/regression_' + model_name + '_SST_indices_split_corr_SST_THF_' + season + '.nc'
        nc = Dataset(regression_file_name,'r')


    except:
        # otherwise, read in SST, THF and zonal wind files to calculate regression
        if i>=0:

            # read in SST index data
            SST_index_file_name = loading_dir + '/SST_indices_'+model_name+'_'+ML.ensemble_id[model_name]+'.nc'
            nc_SST = Dataset(SST_index_file_name,'r')
            times_SST = nc_SST.variables['times'][:]
            calendar_SST = nc_SST.variables['times'].calendar
            t_units_SST = nc_SST.variables['times'].units
            NINO34, years = prep_SST_index('NINO34',nc_SST,times_SST,calendar_SST,t_units_SST,season)
            IOBM2, years = prep_SST_index('IOBM2',nc_SST,times_SST,calendar_SST,t_units_SST,season)
            if apply_running_mean is not None: 
                NINO34 = running_mean(NINO34,apply_running_mean)
                IOBM2 = running_mean(IOBM2,apply_running_mean)

            # detrend SST
            NINO34 = detrend(NINO34)
            IOBM2 = detrend(IOBM2)

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

            # create THF index
            THF_NINO34 = create_THF_index(THF_am,lats_THF,lons_THF,-5,5,190,240)
            THF_IOBM2 = create_THF_index(THF_am,lats_THF,lons_THF,-20,20,40,110)
            if apply_running_mean is not None:
                THF_NINO34 = running_mean(THF_NINO34,apply_running_mean)
                THF_IOBM2 = running_mean(THF_IOBM2,apply_running_mean)

            # read in zonal wind
            data_path0 = '/network/aopp/hera/mad/patterson/CMIP6/data/piControl/ua/'
            full_path = data_path0 + ML.model_institute[model_name] + '/' + model_name + '/ua/'
            list_of_files = [full_path+'ua_Amon_'+model_name+'_piControl_'+ML.ensemble_id[model_name]+'_'+season+'.nc']
            ua200_am,lats_ua,lons_ua,levs_ua,times_ua,calendar_ua,t_units_ua = read_in_variable(list_of_files[:],'ua',chosen_level=20000)
            if apply_running_mean is not None:
                ua200_am = running_mean(ua200_am,apply_running_mean)

            # normalise
            IOBM2 = (IOBM2 - np.mean(IOBM2))/np.std(IOBM2)
            THF_IOBM2 = (THF_IOBM2 - np.mean(THF_IOBM2))/np.std(THF_IOBM2)
            NINO34 = (NINO34 - np.mean(NINO34))/np.std(NINO34)
            THF_NINO34 = (THF_NINO34 - np.mean(THF_NINO34))/np.std(THF_NINO34)

            # split into times when SST and THF do and don't have the same sign
            same_sign_mask_NINO34 = ((NINO34 > 0.01)&(THF_NINO34 > 0.01)) | ((NINO34 < -0.01)&(THF_NINO34 < -0.01))
            diff_sign_mask_NINO34 = ((NINO34 < -0.01)&(THF_NINO34 > 0.01)) | ((NINO34 < -0.01)&(THF_NINO34 > 0.01))
            same_sign_mask_IOBM2 = ((IOBM2 > 0.01)&(THF_IOBM2 > 0.01)) | ((IOBM2 < -0.01)&(THF_IOBM2 < -0.01))
            diff_sign_mask_IOBM2 = ((IOBM2 < -0.01)&(THF_IOBM2 > 0.01)) | ((IOBM2 < -0.01)&(THF_IOBM2 > 0.01))

            # regress out ENSO
            slope = stats.linregress(NINO34,IOBM2)[0]
            IOBM2 = IOBM2 - slope * NINO34
            IOBM2 = (IOBM2 - np.mean(IOBM2))/np.std(IOBM2)

            # do regression
            a = np.arange(0.1,1.1,0.1)
            clevs = np.append(-a[::-1],a)

            reg_coeffs = do_reg(NINO34,ua200_am)
            ax = plt.subplot(2,3,1)
            plt.title('NINO3.4')
            cs = plt.contourf(lons_ua,lats_ua,reg_coeffs,clevs,cmap='RdBu_r')
            print(NINO34.shape)
            plt.colorbar(cs)
            
            reg_coeffs = do_reg(NINO34[same_sign_mask_NINO34],ua200_am[same_sign_mask_NINO34,:,:])
            ax = plt.subplot(2,3,2)
            plt.title('NINO3.4 same sign')
            cs = plt.contourf(lons_ua,lats_ua,reg_coeffs,clevs,cmap='RdBu_r')
            print(NINO34[same_sign_mask_NINO34].shape)
            plt.colorbar(cs)

            reg_coeffs = do_reg(NINO34[diff_sign_mask_NINO34],ua200_am[diff_sign_mask_NINO34,:,:])
            ax = plt.subplot(2,3,3)
            plt.title('NINO3.4 different sign')
            print(NINO34[diff_sign_mask_NINO34].shape)
            cs = plt.contourf(lons_ua,lats_ua,reg_coeffs,clevs,cmap='RdBu_r')
            plt.colorbar(cs)

            reg_coeffs = do_reg(IOBM2,ua200_am)
            ax = plt.subplot(2,3,4)
            plt.title('IOBM')
            print(IOBM2.shape)
            cs = plt.contourf(lons_ua,lats_ua,reg_coeffs,clevs,cmap='RdBu_r')
            plt.colorbar(cs)

            reg_coeffs = do_reg(IOBM2[same_sign_mask_IOBM2],ua200_am[same_sign_mask_IOBM2,:,:])
            ax = plt.subplot(2,3,5)
            plt.title('IOBM same sign')
            print(IOBM2[same_sign_mask_IOBM2].shape)
            cs = plt.contourf(lons_ua,lats_ua,reg_coeffs,clevs,cmap='RdBu_r')
            plt.colorbar(cs)

            reg_coeffs = do_reg(IOBM2[diff_sign_mask_IOBM2],ua200_am[diff_sign_mask_IOBM2,:,:])
            ax = plt.subplot(2,3,6)
            plt.title('IOBM different sign')
            print(IOBM2[diff_sign_mask_IOBM2].shape)
            cs = plt.contourf(lons_ua,lats_ua,reg_coeffs,clevs,cmap='RdBu_r')
            plt.colorbar(cs)
            
            plt.show()
            #print(stats.linregress(THF_NINO34,NINO34)[2])
            #print(stats.linregress(THF_IOBM2,IOBM2)[2])
            #print(IOBM2.shape[0])
            #print(IOBM2[IOBM2*THF_IOBM2>0].shape,NINO34[NINO34*THF_NINO34>0].shape)

            #ax = plt.subplot()
            #plt.plot(THF_IOBM2,color='r')
            #ax2 = ax.twinx()
            #plt.plot(IOBM2,color='b')
            #plt.plot([0,600],[0,0],color='k')
            #plt.show()


        else:
            print('Skipping ' + model_name)


