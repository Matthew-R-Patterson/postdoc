"""
Perform Maximum covariance analysis SSTs and sea level pressure for the
pre-industrial control runs of an ensemble of CMIP6 models. 

We are particularly interested in relationships on decadal timescales, so 
interannual data is low pass filtered prior to the MCA. 

This script performs the MCA and then another script plots the data
"""
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from eofs.standard import Eof
import os
import sys
import time

# my modules
cwd = os.getcwd()
repo_dir = '/'
for directory in cwd.split('/')[1:]:
    repo_dir = os.path.join(repo_dir, directory)
    if directory == 'postdoc':
        break

analysis_functions_dir = os.path.join(repo_dir,'analysis_functions')
figure_dir = os.path.join(repo_dir,'figures/A2_1_SSTs_East_Asian_circulation')
saving_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/MCA_data')

sys.path.append(analysis_functions_dir)
from MCA import MCA
import regress_map
from read_in_data import files_in_directory, read_spatial_dimensions, read_time_dimension, read_in_variable, calculate_annual_mean, running_mean, save_file



# read in data files
model_name_list = ['AWI-CM-1-1-MR','BCC-CSM2-MR','CAMS-CSM1-0', 'FGOALS-f3-L', 'FGOALS-g3', 'CanESM5', 'CanESM5-CanOE', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'CNRM-ESM2-1', 'ACCESS-ESM1-5', 'ACCESS-CM2' ,'EC-Earth3', 'EC-Earth3-Veg', 'INM-CM4-8' ,'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MIROC-ES2L', 'HadGEM3-GC31-LL', 'UKESM1-0-LL', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'GISS-E2-1-G', 'CESM2', 'CESM2-WACCM', 'NorESM2-LM', 'NorESM2-MM', 'GFDL-CM4' ,'GFDL-ESM4', 'NESM3','MCM-UA-1-0'] #['

model_name_list = ['FGOALS-f3-L', 'FGOALS-g3', 'CanESM5', 'CanESM5-CanOE',  'CNRM-CM6-1-HR', 'NorESM2-LM', 'NorESM2-MM', 'GFDL-CM4' ,'GFDL-ESM4','IPSL-CM6A-LR']

model_institute = {'AWI-CM-1-1-MR': 'AWI','BCC-CSM2-MR': 'BCC','CAMS-CSM1-0':'CAMS', 'FGOALS-f3-L':'CAS', 'FGOALS-g3':'CAS', 'CanESM5':'CCCma', 'CanESM5-CanOE':'CCCma', 'CNRM-CM6-1':'CNRM-CERFACS', 'CNRM-CM6-1-HR':'CNRM-CERFACS', 'CNRM-ESM2-1':'CNRM-CERFACS', 'ACCESS-ESM1-5':'CSIRO', 'ACCESS-CM2':'CSIRO-ARCCSS' ,'EC-Earth3':'EC-Earth-Consortium', 'EC-Earth3-Veg':'EC-Earth-Consortium', 'INM-CM4-8':'INM' ,'INM-CM5-0':'INM', 'IPSL-CM6A-LR':'IPSL', 'MIROC6':'MIROC', 'MIROC-ES2L':'MIROC', 'HadGEM3-GC31-LL':'MOHC', 'UKESM1-0-LL':'MOHC', 'MPI-ESM1-2-LR':'MPI-M', 'MRI-ESM2-0':'MRI', 'GISS-E2-1-G':'NASA-GISS', 'CESM2':'NCAR', 'CESM2-WACCM':'NCAR', 'NorESM2-LM':'NCC', 'NorESM2-MM':'NCC', 'GFDL-CM4':'NOAA-GFDL' ,'GFDL-ESM4':'NOAA-GFDL', 'NESM3':'NUIST','MCM-UA-1-0':'UA'}


# settings for variable 1
variable1_name = 'psl'
chosen_level1 = None

# settings for variable 2
variable2_name = 'tas'
chosen_level2 = None

# define East Asian region
lon_min, lon_max = 60,150
lat_min, lat_max = 10,50

# define tropics
lon_min_tropics, lon_max_tropics = 0,360
lat_min_tropics, lat_max_tropics = -30,30

# running mean
N = 10
halfN = int(N/2)


for i, model_name in enumerate(model_name_list):

    try: 
        data_path0 = '/network/group/aopp/predict/AWH007_BEFORT_CMIP6/piControl/'

        # variable 1
        data_path1 = '/piControl/Amon/' + variable1_name + '/gn/latest'
        full_path = data_path0 + model_institute[model_name] + '/' + model_name + data_path1
        list_of_files1 = files_in_directory(full_path,concat_directory=True)
        variable1_data,lats1,lons1,levs1,times1,calendar1,t_units1 = read_in_variable(list_of_files1[:],
                variable1_name,chosen_level=chosen_level1)

        # variable 2
        data_path2 = '/piControl/Amon/' + variable2_name + '/gn/latest'
        full_path = data_path0 + model_institute[model_name] + '/' + model_name + data_path2
        list_of_files2 = files_in_directory(full_path,concat_directory=True)
        variable2_data,lats2,lons2,levs2,times2,calendar2,t_units2 = read_in_variable(list_of_files2[:],
                variable2_name,chosen_level=chosen_level2)

        # truncate time series so that only common time periods are included
        if np.min(times1) != np.min(times2):
            if np.min(times1) > np.min(times2): earliest_common_time = np.min(times1)
            else: earliest_common_time = np.min(times2)
        else: earliest_common_time = np.min(times1)
        if np.max(times1) != np.max(times2):
            if np.max(times1) > np.max(times2): latest_common_time = np.max(times2)
            else: latest_common_time = np.max(times1)
        else: latest_common_time = np.max(times1)
        time_mask1 = (times1>=earliest_common_time)&(times1<=latest_common_time)
        time_mask2 = (times2>=earliest_common_time)&(times2<=latest_common_time)
        times1 = times1[time_mask1]
        times2 = times2[time_mask2]
        variable1_data = variable1_data[time_mask1]
        variable2_data = variable2_data[time_mask2]

        # calculate annual means
        variable1_JJA, years = calculate_annual_mean(variable1_data,times1,calendar1,t_units1,season='JJA')
        variable2_JJA, years = calculate_annual_mean(variable2_data,times2,calendar2,t_units2,season='JJA')    

        # perform low pass time filtering
        variable1_rm = running_mean(variable1_JJA,N)
        variable2_rm = running_mean(variable2_JJA,N)

        # only use variable in region over East Asia for MCA
        lon_mask_EAsia = (lons1>=lon_min)&(lons1<=lon_max)
        lat_mask_EAsia = (lats1>=lat_min)&(lats1<=lat_max)
        variable1_rm_EAsia = variable1_rm[:,lat_mask_EAsia,:][:,:,lon_mask_EAsia]
    
        # only take variable 2 in tropics
        lon_mask_tropics = (lons2>=lon_min_tropics)&(lons2<=lon_max_tropics)
        lat_mask_tropics = (lats2>=lat_min_tropics)&(lats2<=lat_max_tropics)
        variable2_rm_tropics = variable2_rm[:,lat_mask_tropics,:][:,:,lon_mask_tropics]
    
        # MCA analysis
        nLats_mask1 = lats1[lat_mask_EAsia].shape[0]
        weights_1 = np.sqrt(np.cos(np.deg2rad(lats1[lat_mask_EAsia]))).reshape(1,nLats_mask1,1)
        nLats_mask2 = lats2[lat_mask_tropics].shape[0]
        weights_2 = np.sqrt(np.cos(np.deg2rad(lats2[lat_mask_tropics]))).reshape(1,nLats_mask2,1)
        mca = MCA(variable1_rm_EAsia,variable2_rm_tropics,weightsX=weights_1,weightsY=weights_2)
        mca.do_MCA()
        frac_cov = mca.frac_cov(n=3)
        u_ts,v_ts = mca.pattern_time_series(n=3)
        u,v = mca.return_patterns(n=3)

        # calculate principal components
        print("Calculating principal components")
        solver = Eof(variable1_rm_EAsia, weights=weights_1)
        pcs_1 = solver.pcs(npcs=1, pcscaling=1)
        solver = Eof(variable2_rm_tropics, weights=weights_2)
        pcs_2 = solver.pcs(npcs=1, pcscaling=1)

        # standardise MCA time series'
        v_ts1 = (v_ts[0,:] - np.mean(v_ts[0,:]))/np.std(v_ts[0,:])
        v_ts2 = (v_ts[1,:] - np.mean(v_ts[1,:]))/np.std(v_ts[1,:])
        v_ts3 = (v_ts[2,:] - np.mean(v_ts[2,:]))/np.std(v_ts[2,:])
        u_ts1 = (u_ts[0,:] - np.mean(u_ts[0,:]))/np.std(u_ts[0,:])
        u_ts2 = (u_ts[1,:] - np.mean(u_ts[1,:]))/np.std(u_ts[1,:])
        u_ts3 = (u_ts[2,:] - np.mean(u_ts[2,:]))/np.std(u_ts[2,:])
    
        # perform regression of the first 3 MCA time series onto variables 1 and 2
        reg_coeff_v1_variable1,pvals_v1_variable1 = regress_map.regress_map(v_ts1,variable1_rm,map_type='regress')
        reg_coeff_v1_variable2,pvals_v1_variable2 = regress_map.regress_map(v_ts1,variable2_rm,map_type='regress')
        reg_coeff_v2_variable1,pvals_v2_variable1 = regress_map.regress_map(v_ts2,variable1_rm,map_type='regress')
        reg_coeff_v2_variable2,pvals_v2_variable2 = regress_map.regress_map(v_ts2,variable2_rm,map_type='regress')
        reg_coeff_v3_variable1,pvals_v3_variable1 = regress_map.regress_map(v_ts3,variable1_rm,map_type='regress')
        reg_coeff_v3_variable2,pvals_v3_variable2 = regress_map.regress_map(v_ts3,variable2_rm,map_type='regress')
    
        reg_coeff_u1_variable1,pvals_u1_variable1 = regress_map.regress_map(u_ts1,variable1_rm,map_type='regress')
        reg_coeff_u1_variable2,pvals_u1_variable2 = regress_map.regress_map(u_ts1,variable2_rm,map_type='regress')
        reg_coeff_u2_variable1,pvals_u2_variable1 = regress_map.regress_map(u_ts2,variable1_rm,map_type='regress')
        reg_coeff_u2_variable2,pvals_u2_variable2 = regress_map.regress_map(u_ts2,variable2_rm,map_type='regress')
        reg_coeff_u3_variable1,pvals_u3_variable1 = regress_map.regress_map(u_ts3,variable1_rm,map_type='regress')
        reg_coeff_u3_variable2,pvals_u3_variable2 = regress_map.regress_map(u_ts3,variable2_rm,map_type='regress')

        # and regression of first PC time series onto variables 1 and 2
        reg_coeff_pc1_var1_variable1,pvals_pc1_var1_variable1 = regress_map.regress_map(pcs_1[:,0],variable1_rm,map_type='regress')
        reg_coeff_pc1_var1_variable2,pvals_pc1_var1_variable2 = regress_map.regress_map(pcs_1[:,0],variable2_rm,map_type='regress')
        reg_coeff_pc1_var2_variable1,pvals_pc1_var2_variable1 = regress_map.regress_map(pcs_2[:,0],variable1_rm,map_type='regress')
        reg_coeff_pc1_var2_variable2,pvals_pc1_var2_variable2 = regress_map.regress_map(pcs_2[:,0],variable2_rm,map_type='regress')
    
        # save file 
        f = saving_dir + '/MCA_data_' + model_name + '_' + variable1_name + '_' + variable2_name + '.nc'
        description = 'Maximum covariance analysis of preindustrial runs using the model ' + model_name
        save = save_file(f,description)

        # add dimensions
        save.add_dimension(lats1,'lats1')
        save.add_dimension(lons1,'lons1')
        save.add_dimension(lats2,'lats2')
        save.add_dimension(lons2,'lons2')
        save.add_dimension(lats1[lat_mask_EAsia],'lats1_EAsia')
        save.add_dimension(lons1[lon_mask_EAsia],'lons1_EAsia')
        save.add_dimension(lats2[lat_mask_tropics],'lats2_tropics')
        save.add_dimension(lons2[lon_mask_tropics],'lons2_tropics')    
        save.add_dimension(np.arange(3),'MCA_mode')
        save.add_times(times1,calendar1,t_units1,time_name='times1')
        save.add_times(times2,calendar2,t_units2,time_name='times2')
        save.add_times(years[halfN:][::-1][halfN-1:][::-1],calendar2,t_units2,time_name='running_mean_years')

        # add MCA variables   
        save.add_variable(u,'u',('MCA_mode','lats1_EAsia','lons1_EAsia',))
        save.add_variable(v,'v',('MCA_mode','lats2_tropics','lons2_tropics',))
        save.add_variable(frac_cov,'frac_cov',('MCA_mode',))
        save.add_variable(u_ts,'u_ts',('MCA_mode','running_mean_years',))
        save.add_variable(v_ts,'v_ts',('MCA_mode','running_mean_years',))
    
        # add regression coefficients
        save.add_variable(reg_coeff_v1_variable1,'reg_coeff_v1_variable1',('lats1','lons1',))
        save.add_variable(reg_coeff_v1_variable2,'reg_coeff_v1_variable2',('lats2','lons2',))
        save.add_variable(reg_coeff_v2_variable1,'reg_coeff_v2_variable1',('lats1','lons1',))
        save.add_variable(reg_coeff_v2_variable2,'reg_coeff_v2_variable2',('lats2','lons2',))
        save.add_variable(reg_coeff_v3_variable1,'reg_coeff_v3_variable1',('lats1','lons1',))
        save.add_variable(reg_coeff_v3_variable2,'reg_coeff_v3_variable2',('lats2','lons2',))
    
        save.add_variable(reg_coeff_u1_variable1,'reg_coeff_u1_variable1',('lats1','lons1',))
        save.add_variable(reg_coeff_u1_variable2,'reg_coeff_u1_variable2',('lats2','lons2',))
        save.add_variable(reg_coeff_u2_variable1,'reg_coeff_u2_variable1',('lats1','lons1',))
        save.add_variable(reg_coeff_u2_variable2,'reg_coeff_u2_variable2',('lats2','lons2',))
        save.add_variable(reg_coeff_u3_variable1,'reg_coeff_u3_variable1',('lats1','lons1',))
        save.add_variable(reg_coeff_u3_variable2,'reg_coeff_u3_variable2',('lats2','lons2',))

        save.add_variable(reg_coeff_pc1_var1_variable1,'reg_coeff_pc1_var1_variable1',('lats1','lons1',))
        save.add_variable(reg_coeff_pc1_var1_variable2,'reg_coeff_pc1_var1_variable2',('lats2','lons2',))
        save.add_variable(reg_coeff_pc1_var2_variable1,'reg_coeff_pc1_var2_variable1',('lats1','lons1',))
        save.add_variable(reg_coeff_pc1_var2_variable2,'reg_coeff_pc1_var2_variable2',('lats2','lons2',))
    
        save.add_variable(pvals_v1_variable1,'pvals_v1_variable1',('lats1','lons1',))
        save.add_variable(pvals_v1_variable2,'pvals_v1_variable2',('lats2','lons2',))
        save.add_variable(pvals_v2_variable1,'pvals_v2_variable1',('lats1','lons1',))
        save.add_variable(pvals_v2_variable2,'pvals_v2_variable2',('lats2','lons2',))
        save.add_variable(pvals_v3_variable1,'pvals_v3_variable1',('lats1','lons1',))
        save.add_variable(pvals_v3_variable2,'pvals_v3_variable2',('lats2','lons2',))
    
        save.add_variable(pvals_u1_variable1,'pvals_u1_variable1',('lats1','lons1',))
        save.add_variable(pvals_u1_variable2,'pvals_u1_variable2',('lats2','lons2',))
        save.add_variable(pvals_u2_variable1,'pvals_u2_variable1',('lats1','lons1',))
        save.add_variable(pvals_u2_variable2,'pvals_u2_variable2',('lats2','lons2',))
        save.add_variable(pvals_u3_variable1,'pvals_u3_variable1',('lats1','lons1',))
        save.add_variable(pvals_u3_variable2,'pvals_u3_variable2',('lats2','lons2',))
    
        save.add_variable(pvals_pc1_var1_variable1,'pvals_pc1_var1_variable1',('lats1','lons1',))
        save.add_variable(pvals_pc1_var1_variable2,'pvals_pc1_var1_variable2',('lats2','lons2',))
        save.add_variable(pvals_pc1_var2_variable1,'pvals_pc1_var2_variable1',('lats1','lons1',))
        save.add_variable(pvals_pc1_var2_variable2,'pvals_pc1_var2_variable2',('lats2','lons2',))
    
        save.close_file()

    except: 
        message = 'Error, skipping model ' + model_name +' for variables ' + variable1_name + ' and '+ variable2_name
        print(message)
        file = open("skipped_models.txt","a+")
        file.write(message + '\n')
        file.close()


