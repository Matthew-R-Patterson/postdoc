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
analysis_functions_dir = os.path.join(repo_dir,'analysis_functions')
loading_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/SST_index_data')
figure_dir = os.path.join(repo_dir,'figures/A2_1_SSTs_East_Asian_circulation/regress_SST_indices_circulation')
sys.path.append(analysis_functions_dir)
import regress_map
from significance_testing import t_test_regression
from read_in_data import files_in_directory, read_spatial_dimensions, read_time_dimension, read_in_variable, calculate_annual_mean, running_mean, save_file


# list of models 'AWI-CM-1-1-MR'
model_name_list = ['BCC-CSM2-MR']#['EC-Earth3']#['MPI-ESM1-2-LR']# ['CNRM-CM6-1-HR']# ['HadGEM3-GC31-LL']#['MPI-ESM1-2-LR']# [ 'HadGEM3-GC31-LL']#['AWI-CM-1-1-MR','BCC-CSM2-MR','CAMS-CSM1-0', 'FGOALS-f3-L', 'FGOALS-g3', 'CanESM5','CanESM5-CanOE', 'CNRM-CM6-1','CNRM-CM6-1-HR', 'CNRM-ESM2-1','ACCESS-ESM1-5', 'ACCESS-CM2' ,'EC-Earth3', 'EC-Earth3-Veg', 'INM-CM4-8' ,'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MIROC-ES2L', 'HadGEM3-GC31-LL', 'UKESM1-0-LL','MPI-ESM1-2-LR', 'MRI-ESM2-0', 'GISS-E2-1-G', 'CESM2', 'CESM2-WACCM', 'NorESM2-LM', 'NorESM2-MM', 'GFDL-CM4' ,'GFDL-ESM4', 'NESM3','MCM-UA-1-0']

model_institute = {'AWI-CM-1-1-MR': 'AWI','BCC-CSM2-MR': 'BCC','CAMS-CSM1-0':'CAMS', 'FGOALS-f3-L':'CAS', 'FGOALS-g3':'CAS', 'CanESM5':'CCCma', 'CanESM5-CanOE':'CCCma', 'CNRM-CM6-1':'CNRM-CERFACS', 'CNRM-CM6-1-HR':'CNRM-CERFACS', 'CNRM-ESM2-1':'CNRM-CERFACS', 'ACCESS-ESM1-5':'CSIRO', 'ACCESS-CM2':'CSIRO-ARCCSS' ,'EC-Earth3':'EC-Earth-Consortium', 'EC-Earth3-Veg':'EC-Earth-Consortium', 'INM-CM4-8':'INM' ,'INM-CM5-0':'INM', 'IPSL-CM6A-LR':'IPSL', 'MIROC6':'MIROC', 'MIROC-ES2L':'MIROC', 'HadGEM3-GC31-LL':'MOHC', 'UKESM1-0-LL':'MOHC', 'MPI-ESM1-2-LR':'MPI-M', 'MRI-ESM2-0':'MRI', 'GISS-E2-1-G':'NASA-GISS', 'CESM2':'NCAR', 'CESM2-WACCM':'NCAR', 'NorESM2-LM':'NCC', 'NorESM2-MM':'NCC', 'GFDL-CM4':'NOAA-GFDL' ,'GFDL-ESM4':'NOAA-GFDL', 'NESM3':'NUIST','MCM-UA-1-0':'UA'}

ensemble_id = {'AWI-CM-1-1-MR': 'r1i1p1f1','BCC-CSM2-MR': 'r1i1p1f1','CAMS-CSM1-0':'r1i1p1f1', 'FGOALS-f3-L':'r1i1p1f1', 'FGOALS-g3':'r1i1p1f1', 'CanESM5':'r1i1p1f1', 'CanESM5-CanOE':'r1i1p2f1', 'CNRM-CM6-1':'r1i1p1f2', 'CNRM-CM6-1-HR':'r1i1p1f2', 'CNRM-ESM2-1':'r1i1p1f2', 'ACCESS-ESM1-5':'r1i1p1f1', 'ACCESS-CM2':'r1i1p1f1' ,'EC-Earth3':'r1i1p1f1', 'EC-Earth3-Veg':'r1i1p1f1', 'INM-CM4-8':'r1i1p1f1' ,'INM-CM5-0':'r1i1p1f1', 'IPSL-CM6A-LR':'r1i1p1f1', 'MIROC6':'r1i1p1f1', 'MIROC-ES2L':'r1i1p1f2', 'HadGEM3-GC31-LL':'r1i1p1f1', 'UKESM1-0-LL':'r1i1p1f2', 'MPI-ESM1-2-LR':'r1i1p1f1', 'MRI-ESM2-0':'r1i1p1f1', 'GISS-E2-1-G':'r1i1p1f1', 'CESM2':'r1i1p1f1', 'CESM2-WACCM':'r1i1p1f1', 'NorESM2-LM':'r1i1p1f1', 'NorESM2-MM':'r1i1p1f1', 'GFDL-CM4':'r1i1p1f1' ,'GFDL-ESM4':'r1i1p1f1', 'NESM3':'r1i1p1f1','MCM-UA-1-0':'r1i1p1f1'}


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
    if field2 is not None: m.add_contours(lons,lats,field2,clevs2)
    make_plots.add_lat_lon(ax,fontsize=15)
    m.geography(ax,extent=[30,330,-15,70])
    #make_plots.plot_box(lon_min=110,lon_max=180,lat_min=20,lat_max=50)
    ax.set_aspect(2)
    return cs

def do_reg(SST_index,data_all):
    """ Do regression, but only include statistically significant values"""
    regress_coeff, pvals = regress_map.regress_map(SST_index,data_all,map_type='regress')
    pvals = t_test_regression(data_all,regress_coeff)
    mask = np.ones_like(pvals)
    mask[pvals>0.05] = np.nan
    regress_coeff = regress_coeff * mask
    return regress_coeff



chosen_level = 20000
var_name = 'ua'
apply_running_mean = 11

plt.figure(figsize=(15,15))
gs = gridspec.GridSpec(6,3,height_ratios=[10,10,10,10,10,0.5])

for i, model_name in enumerate(model_name_list):

    if i>=0:
        
        for j, season in enumerate(['DJF','MAM','JJA','SON','ANN']):

            # read in SST data
            SST_index_file_name = loading_dir + '/SST_indices_'+model_name+'_'+ensemble_id[model_name]+'.nc'
            nc_SST = Dataset(SST_index_file_name,'r')
            times_SST = nc_SST.variables['times'][:]
            calendar_SST = nc_SST.variables['times'].calendar
            t_units_SST = nc_SST.variables['times'].units
            IOBM2, years = prep_SST_index('IOBM2',nc_SST,times_SST,calendar_SST,t_units_SST,season)
            NINO34, years = prep_SST_index('NINO34',nc_SST,times_SST,calendar_SST,t_units_SST,season)

            # apply running mean
            if apply_running_mean is not None:
                IOBM2 = running_mean(IOBM2,apply_running_mean)
                NINO34 = running_mean(NINO34,apply_running_mean)

            # regress out other index
            IOBM2_no_NINO34 = regress_out_index(IOBM2,NINO34)
            NINO34_no_IOBM2 = regress_out_index(NINO34,IOBM2)

            # read in circulation indices
            data_path0 = '/network/aopp/hera/mad/patterson/CMIP6/data/piControl/ua/'
            data_path1 = '/' + var_name + '/'
            full_path = data_path0 + model_institute[model_name] + '/' + model_name + data_path1
            list_of_files = [full_path+var_name+'_Amon_'+model_name+'_piControl_'+ensemble_id[model_name]+'_'+season+'.nc']
            ua200,lats_var,lons_var,levs_var,times_var,calendar_var,t_units_var = read_in_variable(list_of_files[:],var_name,chosen_level=20000)
            ua850,lats_var,lons_var,levs_var,times_var,calendar_var,t_units_var = read_in_variable(list_of_files[:],var_name,chosen_level=85000)
            if season == 'DJF': 
                ua200 = ua200[::-1][1:][::-1]
                ua850 = ua850[::-1][1:][::-1]
            if apply_running_mean is not None:
                ua200 = running_mean(ua200,apply_running_mean)
                ua850 = running_mean(ua850,apply_running_mean)
                #nHalf = int((apply_running_mean)/2)
                #ua200 = ua200[nHalf:][::-1][nHalf:][::-1]
                #ua850 = ua850[nHalf:][::-1][nHalf:][::-1]

            # calculate regressions
            # also calculate pvalues and show regcoeffs < 0.05
            regress_coeff_IOBM2_ua200 = do_reg(IOBM2,ua200)
            regress_coeff_IOBM2_no_NINO34_ua200 = do_reg(IOBM2_no_NINO34,ua200)
            regress_coeff_NINO34_ua200 = do_reg(NINO34,ua200)
            regress_coeff_IOBM2_ua850 = do_reg(IOBM2,ua850)
            regress_coeff_IOBM2_no_NINO34_ua850 = do_reg(IOBM2_no_NINO34,ua850)
            regress_coeff_NINO34_ua850 = do_reg(NINO34,ua850)

            # plot
            a = np.arange(0.5,4.1,0.5)
            clevs = np.append(-a[::-1],a)

            ax = plt.subplot(gs[j,0],projection=ccrs.PlateCarree(central_longitude=180.))
            cs = plot_reg(ax,regress_coeff_IOBM2_ua200,lats_var,lons_var,clevs,field2=regress_coeff_IOBM2_ua850,clevs2=clevs,title='')
            if j==0: plt.title('IO SST',fontsize=25)
            ax.text(-0.3,0.1,season,transform=ax.transAxes,rotation=90,fontsize=25)
            ax = plt.subplot(gs[j,2],projection=ccrs.PlateCarree(central_longitude=180.))
            cs = plot_reg(ax,regress_coeff_IOBM2_no_NINO34_ua200,lats_var,lons_var,clevs,field2=regress_coeff_IOBM2_no_NINO34_ua850,clevs2=clevs,title='')
            if j==0: plt.title('IO SST, no NINO 3.4',fontsize=25)
            ax = plt.subplot(gs[j,1],projection=ccrs.PlateCarree(central_longitude=180.))
            cs = plot_reg(ax,regress_coeff_NINO34_ua200,lats_var,lons_var,clevs,field2=regress_coeff_NINO34_ua850,clevs2=clevs,title='')
            if j==0: plt.title('NINO 3.4',fontsize=25)


        ax = plt.subplot(gs[5,:])
        make_plots.colorbar(ax,cs,orientation='horizontal')       
        #plt.subplots_adjust(hspace=0.4)
        figure_name = figure_dir + '/ua_IOBM2_regress_out_NINO34_'+model_name+'.png'
        if apply_running_mean is not None: figure_name = figure_dir + '/ua_IOBM2_regress_out_NINO34_'+model_name+'_rm'+str(apply_running_mean)+'.png'
        print('saving to %s' % (figure_name))
        plt.savefig(figure_name,bbox_inches='tight')

    else:
        message = 'Error, skipping model ' + model_name +' for regression analysis'
        print(message)
