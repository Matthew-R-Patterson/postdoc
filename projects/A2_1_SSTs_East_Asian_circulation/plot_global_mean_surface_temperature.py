""" Plot variations in the global mean surface temperature for preindustrial control runs """

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
from scipy import stats
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
figure_dir = os.path.join(repo_dir,'figures/A2_1_SSTs_East_Asian_circulation/global_mean_Ts')
sys.path.append(analysis_functions_dir)

from read_in_data import files_in_directory, read_in_variable, calculate_annual_mean, running_mean, save_file, read_spatial_dimensions, read_time_dimension, prep_SST_index
from regress_map import regress_map
from global_mean import global_mean

season = 'ANN'
model_name_list = ML.model_name_list[1:16] #['HadGEM3-GC31-LL']#['BCC-CSM2-MR'] #ML.model_name_list
model_name_list.remove('CanESM5')
model_name_list.append('HadGEM3-GC31-LL')
model_name_list.append('CESM2')


plt.figure(figsize=(30,35))
gs = gridspec.GridSpec(8,2)

for i, model_name in enumerate(model_name_list):

    try:
        #data_path0 = '/network/aopp/hera/mad/patterson/iCMIP6/data/piControl/ts/'
        data_path0 = '/network/group/aopp/predict/AWH007_BEFORT_CMIP6/piControl/' #AWI/AWI-CM-1-1-MR/piControl/Amon/tas/gn/latest
        full_path = data_path0 + ML.model_institute[model_name] + '/' + model_name + '/piControl/Amon/tas/gn/latest/'
        list_of_files = files_in_directory(full_path,concat_directory=True,exclude_files_with='ImonAnt') 
        ts,lats_ts,lons_ts,levs_ts,times_ts,calendar_ts,t_units_ts = read_in_variable(list_of_files[:],'tas')

        ts_am, years = calculate_annual_mean(ts,times_ts,calendar_ts,t_units_ts,season=season)

        ts_mean = global_mean(ts_am,lats_ts)

        ax = plt.subplot(gs[i])
        plt.title(model_name,fontsize=20)
        plt.plot(ts_mean,color='r')
        plt.xlim(0,1000)

        if i>=14: plt.xlabel('Time (years)',fontsize=20)
        if i % 2 == 0: plt.ylabel('Global mean T (K)',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

    except: 
        print('Skipping ' + model_name)

figure_name = figure_dir + '/global_mean_ts_individual_models_'+season+'_2.png'

print('saving to %s' % (figure_name))
plt.savefig(figure_name,bbox_inches='tight')

