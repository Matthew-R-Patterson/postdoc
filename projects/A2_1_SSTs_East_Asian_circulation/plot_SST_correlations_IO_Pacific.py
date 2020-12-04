""" Compare correlations of SST over the Indian Ocean basin with the Pacific 
found in reanalysis with models."""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
from matplotlib import gridspec
from scipy import stats
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
figure_dir = os.path.join(repo_dir,'figures/A2_1_SSTs_East_Asian_circulation/SST_interbasin_correlations')
#data_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/regress_SST_circulation_data')
SST_data_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/SST_index_data')
sys.path.append(analysis_functions_dir)
from read_in_data import calculate_annual_mean, running_mean

model_name_list = ['AWI-CM-1-1-MR','BCC-CSM2-MR','CAMS-CSM1-0', 'FGOALS-f3-L', 'FGOALS-g3', 'CanESM5', 'CanESM5-CanOE', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'CNRM-ESM2-1', 'ACCESS-ESM1-5', 'ACCESS-CM2' ,'EC-Earth3', 'EC-Earth3-Veg', 'INM-CM4-8' ,'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MIROC-ES2L', 'HadGEM3-GC31-LL', 'UKESM1-0-LL', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'GISS-E2-1-G', 'CESM2', 'CESM2-WACCM', 'NorESM2-LM', 'NorESM2-MM', 'GFDL-CM4' ,'GFDL-ESM4', 'NESM3','MCM-UA-1-0','ERA20C']

ensemble_id = {'AWI-CM-1-1-MR': 'r1i1p1f1','BCC-CSM2-MR': 'r1i1p1f1','CAMS-CSM1-0':'r1i1p1f1', 'FGOALS-f3-L':'r1i1p1f1', 'FGOALS-g3':'r1i1p1f1', 'CanESM5':'r1i1p1f1', 'CanESM5-CanOE':'r1i1p2f1', 'CNRM-CM6-1':'r1i1p1f2', 'CNRM-CM6-1-HR':'r1i1p1f2', 'CNRM-ESM2-1':'r1i1p1f2', 'ACCESS-ESM1-5':'r1i1p1f1', 'ACCESS-CM2':'r1i1p1f1' ,'EC-Earth3':'r1i1p1f1', 'EC-Earth3-Veg':'r1i1p1f1', 'INM-CM4-8':'r1i1p1f1' ,'INM-CM5-0':'r1i1p1f1', 'IPSL-CM6A-LR':'r1i1p1f1', 'MIROC6':'r1i1p1f1', 'MIROC-ES2L':'r1i1p1f2', 'HadGEM3-GC31-LL':'r1i1p1f1', 'UKESM1-0-LL':'r1i1p1f2', 'MPI-ESM1-2-LR':'r1i1p1f1', 'MRI-ESM2-0':'r1i1p1f1', 'GISS-E2-1-G':'r1i1p1f1', 'CESM2':'r1i1p1f1', 'CESM2-WACCM':'r1i1p1f1', 'NorESM2-LM':'r1i1p1f1', 'NorESM2-MM':'r1i1p1f1', 'GFDL-CM4':'r1i1p1f1' ,'GFDL-ESM4':'r1i1p1f1', 'NESM3':'r1i1p1f1','MCM-UA-1-0':'r1i1p1f1'}

season = 'JJA'
decadal_mean = True

def prep_index(file_name,ts_name,season):
    """ Read in a monthly time series, take the annual mean
    and detrend """
    nc = Dataset(file_name,'r')
    ts = nc.variables[ts_name][:]
    times = nc.variables['times'][:]
    calendar = nc.variables['times'].calendar
    units = nc.variables['times'].units
    ts_am, years = calculate_annual_mean(ts,times,calendar,units,season=season,decadal_mean=decadal_mean) # annual mean
    # detrend
    t = np.arange(years.shape[0])
    slope = stats.linregress(t,ts_am)[0]
    ts_am = ts_am - slope * t 
    return ts_am, years

plt.figure(figsize=(15,15))
gs = gridspec.GridSpec(2,1,height_ratios=[5,2])
title_fontsize = 30
ax = plt.subplot(gs[0])

for i, model_name in enumerate(model_name_list):
    try:
        if model_name is not 'ERA20C': file_name = SST_data_dir + '/SST_indices_'+model_name+'_'+ensemble_id[model_name]+'.nc'
        else: file_name = SST_data_dir + '/SST_indices_'+model_name+'.nc'
        IOBM, years = prep_index(file_name,'IOBM',season)
        NINO34, years = prep_index(file_name,'NINO34',season)

        if decadal_mean ==True:
            r = stats.linregress(IOBM,NINO34)[2]
            plt.scatter(r,i,color='r')
            plt.text(0.01,i-0.5,model_name,fontsize=0.5*title_fontsize)
        else:
            n_years = years.shape[0]
            n_periods = int(n_years / 110)
            for j in np.arange(n_periods):
                r = stats.linregress(IOBM[110*j:110*j+110][1:],NINO34[110*j:110*j+110][::-1][1:][::-1])[2]
                if model_name != 'ERA20C':
                    plt.scatter(r,i,color='r')
                    if j ==0: plt.text(0.01,i-0.5,model_name,fontsize=0.5*title_fontsize)
                else:
                    plt.plot([r,r],[-5,35],color='k')
                    plt.plot([0,0],[-5,35],color='0.75',linestyle='--')
    except: print('Skipping ', model_name)

plt.title('R (NINO 3.4, IOBM)',fontsize=title_fontsize) 
plt.ylim(-1,33)
plt.xticks(fontsize=0.8*title_fontsize)
plt.yticks([])
plt.plot([0,0],[-5,35],color='0.75',linestyle='--')

#plt.xlabel('Correlation',fontsize=0.8*title_fontsize)

ax = plt.subplot(gs[1])
plt.title('HadISST',fontsize=title_fontsize)
#IOBM = running_mean(IOBM,10)
#NINO34 = running_mean(NINO34,10)
ax.plot(years,IOBM,color='r')
ax2 = ax.twinx()
ax2.plot(years,NINO34,color='b')
#plt.xticks(fontsize=0.8*title_fontsize)
#plt.yticks(fontsize=0.8*title_fontsize)
plt.legend(['IOBM','NINO 3.4'],fontsize=0.5*title_fontsize)

if decadal_mean == True: figure_name = figure_dir +'/SST_interbasin_corr_IOBM_NINO34_'+season+'_decadal.png'
else: figure_name = figure_dir +'/SST_interbasin_corr_IOBM_NINO34_'+season+'.png'
#print('saving to %s' % (figure_name))
#plt.savefig(figure_name,bbox_inches='tight')
plt.show()
