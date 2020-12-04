""" Calculate and save regression coefficients of various SST indices on zonal wind
and sea level pressure and precipitation. """

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
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
loading_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/SST_index_data')
saving_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/regress_SST_circulation_data')
sys.path.append(analysis_functions_dir)
import regress_map
from read_in_data import files_in_directory, read_spatial_dimensions, read_time_dimension, read_in_variable, calculate_annual_mean, running_mean, save_file


# list of models 'AWI-CM-1-1-MR'
model_name_list = [ 'HadGEM3-GC31-LL', 'UKESM1-0-LL']#['AWI-CM-1-1-MR','BCC-CSM2-MR','CAMS-CSM1-0', 'FGOALS-f3-L', 'FGOALS-g3', 'CanESM5','CanESM5-CanOE', 'CNRM-CM6-1','CNRM-CM6-1-HR', 'CNRM-ESM2-1','ACCESS-ESM1-5', 'ACCESS-CM2' ,'EC-Earth3', 'EC-Earth3-Veg', 'INM-CM4-8' ,'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MIROC-ES2L', 'HadGEM3-GC31-LL', 'UKESM1-0-LL','MPI-ESM1-2-LR', 'MRI-ESM2-0', 'GISS-E2-1-G', 'CESM2', 'CESM2-WACCM', 'NorESM2-LM', 'NorESM2-MM', 'GFDL-CM4' ,'GFDL-ESM4', 'NESM3','MCM-UA-1-0']

model_institute = {'AWI-CM-1-1-MR': 'AWI','BCC-CSM2-MR': 'BCC','CAMS-CSM1-0':'CAMS', 'FGOALS-f3-L':'CAS', 'FGOALS-g3':'CAS', 'CanESM5':'CCCma', 'CanESM5-CanOE':'CCCma', 'CNRM-CM6-1':'CNRM-CERFACS', 'CNRM-CM6-1-HR':'CNRM-CERFACS', 'CNRM-ESM2-1':'CNRM-CERFACS', 'ACCESS-ESM1-5':'CSIRO', 'ACCESS-CM2':'CSIRO-ARCCSS' ,'EC-Earth3':'EC-Earth-Consortium', 'EC-Earth3-Veg':'EC-Earth-Consortium', 'INM-CM4-8':'INM' ,'INM-CM5-0':'INM', 'IPSL-CM6A-LR':'IPSL', 'MIROC6':'MIROC', 'MIROC-ES2L':'MIROC', 'HadGEM3-GC31-LL':'MOHC', 'UKESM1-0-LL':'MOHC', 'MPI-ESM1-2-LR':'MPI-M', 'MRI-ESM2-0':'MRI', 'GISS-E2-1-G':'NASA-GISS', 'CESM2':'NCAR', 'CESM2-WACCM':'NCAR', 'NorESM2-LM':'NCC', 'NorESM2-MM':'NCC', 'GFDL-CM4':'NOAA-GFDL' ,'GFDL-ESM4':'NOAA-GFDL', 'NESM3':'NUIST','MCM-UA-1-0':'UA'}

class prep_index_regress_save:

    def __init__(self,index_name,nc,times_SST,calendar_SST,t_units_SST,N,season='JJA'):
        """ """
        index_ts = nc.variables[index_name][:]
        index_annual_mean, years = calculate_annual_mean(index_ts,times_SST,calendar_SST,t_units_SST,season=season)
        index_running_mean = running_mean(index_annual_mean,N)
        self.index_running_mean = index_running_mean
        self.years = years
        
    def regress_variable(self,field,save_file_object,regress_coeff_name,pval_name,lat_name='lats',lon_name='lons'):
        """ Performs regression between the index running mean and a field (e.g. sea level pressure)
        The data is then saved """
        regress_coeff, pvals = regress_map.regress_map(self.index_running_mean,field,map_type='regress')
        save_file_object.add_variable(regress_coeff,regress_coeff_name,(lat_name,lon_name,))
        save_file_object.add_variable(pvals,pval_name,(lat_name,lon_name,))

# select season
season = 'JJA'

for i, model_name in enumerate(model_name_list):

    try: 
        # read in SST file and times
        SST_index_file_name = loading_dir + '/SST_indices_'+model_name+'.nc'
        nc_SST = Dataset(SST_index_file_name,'r')
        times_SST = nc_SST.variables['times'][:]
        calendar_SST = nc_SST.variables['times'].calendar
        t_units_SST = nc_SST.variables['times'].units

        # read in precipitation, zonal wind, slp
        data_path0 = '/network/group/aopp/predict/AWH007_BEFORT_CMIP6/piControl/'
        # precip
        data_path_pr = '/piControl/Amon/pr/gn/latest'
        full_path_pr = data_path0 + model_institute[model_name] + '/' + model_name + data_path_pr
        list_of_files_pr = files_in_directory(full_path_pr,concat_directory=True)
        pr_data,lats_pr,lons_pr,levs_pr,times_pr,calendar_pr,t_units_pr = read_in_variable(list_of_files_pr[:],'pr')
        # slp
        data_path_psl = '/piControl/Amon/psl/gn/latest'
        full_path_psl = data_path0 + model_institute[model_name] + '/' + model_name + data_path_psl
        list_of_files_psl = files_in_directory(full_path_psl,concat_directory=True)
        psl_data,lats_psl,lons_psl,levs_psl,times_psl,calendar_psl,t_units_psl = read_in_variable(list_of_files_psl[:],'psl')

        # truncate times so that only a common time period is used
        #earliest_common_time = np.min(times_SST)
        #if np.min(times_SST) != np.min(times_psl) & (np.min(times_psl) > np.min(times_SST)): earliest_common_time = np.min(times_psl)
        #latest_common_time = np.max(times_SST)
        #if np.max(times_SST) != np.max(times_psl) & (np.max(times_psl) < np.max(times_SST)): latest_common_time = np.max(times_psl)
        #time_mask_SST = (times_SST>=earliest_common_time)&(times_SST<=latest_common_time)
        #time_mask_psl = (times_psl>=earliest_common_time)&(times_psl<=latest_common_time)
        #times_SST = times1[time_mask1]
        #times2 = times2[time_mask2]
        #variable1_data = variable1_data[time_mask1]
        #variable2_data = variable2_data[time_mask2]

        # calculate annual means of fields
        psl_am, years = calculate_annual_mean(psl_data,times_psl,calendar_psl,t_units_psl,season=season)
        pr_am, years = calculate_annual_mean(pr_data,times_pr,calendar_pr,t_units_pr,season=season)

        # perform low pass time filtering on fields
        N = 10
        halfN = int(N/2)
        psl_rm = running_mean(psl_am,N)
        pr_rm = running_mean(pr_am,N)

        # prepare SST indices
        PDO_object = prep_index_regress_save('PDO',nc_SST,times_SST,calendar_SST,t_units_SST,N,season=season)
        IOD_object = prep_index_regress_save('IOD',nc_SST,times_SST,calendar_SST,t_units_SST,N,season=season)
        IPO_object = prep_index_regress_save('IPO',nc_SST,times_SST,calendar_SST,t_units_SST,N,season=season)
        IOBM_object = prep_index_regress_save('IOBM',nc_SST,times_SST,calendar_SST,t_units_SST,N,season=season)
        NINO34_object = prep_index_regress_save('NINO34',nc_SST,times_SST,calendar_SST,t_units_SST,N,season=season)

        # save file object
        f = saving_dir + '/SST_index_circulation_regression_' + model_name + '_'+season+'.nc'
        description = '' + model_name
        save = save_file(f,description)

        # add dimensions
        save.add_dimension(lats_psl,'lats')
        save.add_dimension(lons_psl,'lons')
        save.add_times(years[halfN:][::-1][halfN-1:][::-1],calendar_psl,t_units_psl,time_name='running_mean_years')
        
        # regression on SST indices and save
        PDO_object.regress_variable(psl_rm,save,'regress_coeff_PDO_psl','pval_PDO_psl')
        IOD_object.regress_variable(psl_rm,save,'regress_coeff_IOD_psl','pval_IOD_psl')
        IPO_object.regress_variable(psl_rm,save,'regress_coeff_IPO_psl','pval_IPO_psl')
        IOBM_object.regress_variable(psl_rm,save,'regress_coeff_IOBM_psl','pval_IOBM_psl')
        NINO34_object.regress_variable(psl_rm,save,'regress_coeff_NINO34_psl','pval_NINO34_psl')

        PDO_object.regress_variable(pr_rm,save,'regress_coeff_PDO_pr','pval_PDO_pr')
        IOD_object.regress_variable(pr_rm,save,'regress_coeff_IOD_pr','pval_IOD_pr')
        IPO_object.regress_variable(pr_rm,save,'regress_coeff_IPO_pr','pval_IPO_pr')
        IOBM_object.regress_variable(pr_rm,save,'regress_coeff_IOBM_pr','pval_IOBM_pr')
        NINO34_object.regress_variable(pr_rm,save,'regress_coeff_NINO34_pr','pval_NINO34_pr')
        save.close_file()


    except: 
        message = 'Error, skipping model ' + model_name +' for regression analysis'
        print(message)
        file = open(saving_dir + '/skipped_models.txt','a+')
        file.write(message + '\n')
        file.close()
