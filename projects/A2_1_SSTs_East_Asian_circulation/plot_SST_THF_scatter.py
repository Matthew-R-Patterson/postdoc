""" Make scatter plots of SST and Turbulent heat flux (latent and sensible)
for different years over various ocean basins across a few different models """

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import os
import sys
from scipy.stats import linregress
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
figure_dir = os.path.join(repo_dir,'figures/A2_1_SSTs_East_Asian_circulation/SST_THF_correlations')
sys.path.append(analysis_functions_dir)

from read_in_data import files_in_directory, read_in_variable, calculate_annual_mean, running_mean
from lead_lag_analysis import lead_lag_ts
from significance_testing import confidence_intervals

season = 'ANN'
apply_running_mean = 11
plot_type = 'scatter'   # 'scatter' / 'lead-lag'
model_name_list = ['BCC-CSM2-MR','CNRM-CM6-1-HR']#,'EC-Earth3','HadGEM3-GC31-LL','MPI-ESM1-2-LR'] #'EC-Earth3' 'HadGEM3-GC31-LL' #ML.model_name_list
#model_name_list.append('ERA20C')

class region:
    """ define regions for use in analysis """
    def __init__(self,name,lat_min,lat_max,lon_min,lon_max):
        self.name = name
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lag_corrs = None

    def get_regional_index(self,data,lats,lons):
        data[np.abs(data)>1e8] = np.nan # remove unusually large values
        data[data[:]==9999] = np.nan
        lat_mask = (lats>=self.lat_min)&(lats<=self.lat_max)
        lon_mask = (lons>=self.lon_min)&(lons<=self.lon_max)
        region_index = np.nanmean(np.nanmean(data[:,lat_mask,:][:,:,lon_mask],axis=1),axis=1)
        return region_index

    def get_tendency(self,index,times):
        """ Calculate tendency as the gradient of the index in time """
        tendency = np.gradient(index,times)
        return tendency

def scatter_plot(x,y,xlabel='',ylabel='',title='',fontsize=10,skip=None):
    if skip is not None: plt.scatter(x[::skip],y[::skip])
    else: plt.scatter(x,y)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    slope,intercept,r_value,p_value,std_err = linregress(x,y)
    plt.title(title+' r = %.2f' % (r_value) ,fontsize=fontsize)

def detrend(x):
    t = np.arange(x.shape[0])
    slope,intercept,r_value,p_value,std_err = linregress(t,x)
    new_x = x - t * slope
    return new_x

IO = region('Indian Ocean',-20,20,40,110)
Pac = region('Pacific',-20,20,190,240)
TAtl = region('Trop. Atlantic',-10,10,325,345) #-20,20,330,360)
XAtl = region('Ext. Atlantic',30,60,310,340)
regions_all = [IO,Pac,TAtl,XAtl]


plt.figure(figsize=(15,10))
if plot_type == 'lead-lag':
    gs = gridspec.GridSpec(2,2)

#model_name_list.remove('AWI-CM-1-1-MR')

for i, model_name in enumerate(model_name_list):

    try:
        # read in SSTs
        if model_name != 'ERA20C':
            data_path0 = '/network/aopp/hera/mad/patterson/CMIP6/data/tos_regridded/'
            full_path = data_path0 + ML.model_institute[model_name] + '/' + model_name + '/tos/'
            list_of_files = files_in_directory(full_path,concat_directory=True,include_files_with=ML.ensemble_id[model_name])
            tos_data,lats,lons,levs,times,calendar,t_units = read_in_variable(list_of_files[:],'tos')
        else:
            file_name = ['/network/group/aopp/met_data/MET003_ERA20C/data/tos/mon/tos_mon_ERA20C_2.5x2.5_189002-201012.nc']
            tos_data,lats,lons,levs,times,calendar,t_units = read_in_variable(file_name,'sst')
        SST_am, years_SST = calculate_annual_mean(tos_data,times,calendar,t_units,season=season) 

        # read in THF
        if model_name != 'ERA20C':
            data_path0 = '/network/aopp/hera/mad/patterson/CMIP6/data/piControl/'
            full_path = data_path0 + 'hfls/'+ ML.model_institute[model_name] + '/' + model_name + '/hfls/'
            list_of_files = [full_path+'hfls_Amon_'+model_name+'_piControl_'+ML.ensemble_id[model_name]+'_'+season+'.nc']
            hfls,lats_var,lons_var,levs_var,times_var,calendar_var,t_units_var = read_in_variable(list_of_files[:],'hfls')
            full_path = data_path0 + 'hfss/' + ML.model_institute[model_name] + '/' + model_name + '/hfss/'
            list_of_files = [full_path+'hfss_Amon_'+model_name+'_piControl_'+ML.ensemble_id[model_name]+'_'+season+'.nc']
            hfss,lats_var,lons_var,levs_var,times_var,calendar_var,t_units_var = read_in_variable(list_of_files[:],'hfss')
        else: 
            file_name = ['/network/group/aopp/met_data/MET003_ERA20C/data/hfls/mon/hfls_mon_ERA20C_2.5x2.5_190001-201012.nc']
            hfls,lats_var,lons_var,levs_var,times_var,calendar_var,t_units_var = read_in_variable(file_name,'slhf')
            file_name = ['/network/group/aopp/met_data/MET003_ERA20C/data/hfss/mon/hfss_mon_ERA20C_2.5x2.5_190001-201012.nc']
            hfss,lats_var,lons_var,levs_var,times_var,calendar_var,t_units_var = read_in_variable(file_name,'sshf')
            hfls = - hfls # change the sign to be consistent with CMIP models
            hfss = - hfss

        THF = hfls + hfss
        THF_am, years_THF = calculate_annual_mean(THF,times_var,calendar_var,t_units_var,season=season) 

        # select only years 1900 onwards for ERA20C for common period with THF
        if model_name == 'ERA20C': 
            SST_am = SST_am[years_SST>=1900]
            THF_am = THF_am[years_THF>=1900]        
            years_SST = years_SST[years_SST>=1900]

        for j, region in enumerate(regions_all):

            THF_region = region.get_regional_index(THF_am,lats_var,lons_var)
            SST_region = region.get_regional_index(SST_am,lats,lons)

            SST_region = region.get_tendency(SST_region,years_SST)

            if apply_running_mean is not None:
                THF_region = running_mean(THF_region,apply_running_mean)
                SST_region = running_mean(SST_region,apply_running_mean)
    
            # detrend 
            THF_region = detrend(THF_region)
            SST_region = detrend(SST_region)
        
            if plot_type == 'scatter':
                slope,intercept,r_value,p_value,std_err = linregress(SST_region,THF_region)
                if model_name != 'ERA20C': plt.scatter(j,r_value)
                else: 
                    lo,hi = confidence_intervals(THF_region,SST_region)
                    plt.errorbar(j+0.1, r_value, yerr=hi - r_value,uplims=True, lolims=True,color='k',fmt='X')

            elif plot_type == 'lead-lag':
                lag_corr, lags = lead_lag_ts(SST_region,THF_region,max_lag=20)
                ax = plt.subplot(gs[j])
                plt.title(region.name)
                plt.plot(lags,lag_corr)
                if i==0: plt.plot([0,0],[-1,1],color='k')
                plt.ylim(-0.2,1)
                if i==0: region.lag_corrs = lag_corr.reshape(1,lags.shape[0])
                else: region.lag_corrs = np.append(region.lag_corrs,lag_corr.reshape(1,lags.shape[0]),axis=0)
                if model_name == 'MCM-UA-1-0': plt.plot(lags,np.mean(region.lag_corrs,axis=0),linewidth=3,color='k')
                plt.xlim(-20,20)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)

            #plt.figure()
            #ax = plt.subplot(1,1,1)
            #ax.plot(SST_region)
            #ax2 = ax.twinx()
            #ax2.plot(THF_region,color='r')
            #plt.show()


            # plot
            #ax = plt.subplot(gs[i,j])
            #scatter_plot(SST_region,THF_region,xlabel='SST',ylabel='THF',title=region.name,skip=10)
            #if j == 0: ax.text(-0.3,0.1,model_name,rotation=90,transform=ax.transAxes,fontsize=10)

    except: 
        print('Skipping ' + model_name)

if plot_type == 'scatter':
    plt.ylabel('Corr (SST,THF)',fontsize=20)
    plt.xticks([0,1,2,3],['Indian Ocean','Pacific','Trop. Atlantic','Ext. Atlantic'],fontsize=20)
    plt.yticks(fontsize=20)

    if apply_running_mean is not None: figure_name = figure_dir + '/corr_SST_THF_rm'+str(apply_running_mean)+'_'+season+'.png'
    else: figure_name = figure_dir + '/corr_SST_THF_'+season+'.png'

elif plot_type == 'lead-lag':
    if apply_running_mean is not None: figure_name = figure_dir + '/lagged_corr_SST_THF_rm'+str(apply_running_mean)+'_'+season+'.png'
    else: figure_name = figure_dir + '/lagged_corr_SST_THF_'+season+'.png'


print('saving to %s' % (figure_name))
#plt.savefig(figure_name,bbox_inches='tight')
plt.show()

plt.figure()
ax = plt.subplot(1,1,1)
ax.plot(SST_region)
ax2 = ax.twinx()
ax2.plot(THF_region,color='r')
plt.show()
