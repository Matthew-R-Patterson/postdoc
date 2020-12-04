""" Plot results of regression analyses between SST indices and circulation
fields including sea level pressure.
"""
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
figure_dir = os.path.join(repo_dir,'figures/A2_1_SSTs_East_Asian_circulation/regress_SST_indices_circulation')
data_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/regress_SST_circulation_data')
SST_data_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/SST_index_data')

model_name_list = ['AWI-CM-1-1-MR','BCC-CSM2-MR','CAMS-CSM1-0', 'FGOALS-f3-L', 'FGOALS-g3', 'CanESM5', 'CanESM5-CanOE', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'CNRM-ESM2-1', 'ACCESS-ESM1-5', 'ACCESS-CM2' ,'EC-Earth3', 'EC-Earth3-Veg', 'INM-CM4-8' ,'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MIROC-ES2L', 'HadGEM3-GC31-LL', 'UKESM1-0-LL', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'GISS-E2-1-G', 'CESM2', 'CESM2-WACCM', 'NorESM2-LM', 'NorESM2-MM', 'GFDL-CM4' ,'GFDL-ESM4', 'NESM3','MCM-UA-1-0']

SST_index_names = ['PDO','IPO','NINO34','IOD','IOBM']
season = 'JJA'

def read_regress_coeffs(file_name,var_name1):
    nc = Dataset(file_name,'r')
    lats = nc.variables['lats'][:]
    lons = nc.variables['lons'][:]
    regress_coeffs = nc.variables[var_name1][:]
    return regress_coeffs,lats,lons
    
def contour_plot(field,clevs,lats,lons,title_fontsize=35,title='',cmap='RdBu_r',extent=[50,290,-30,60]):
    plt.title(title,fontsize=title_fontsize)
    m = make_plots.make_map_plot()
    cs = m.add_filled_contours(lons,lats,field,clevs,cmap=cmap)
    make_plots.add_lat_lon(ax)
    m.geography(ax,extent=extent)
    ax.set_aspect(1.5)
    #ax.set_extent(extent)
    return cs

for i,model_name in enumerate(model_name_list):
    try:
        # set up figure for plotting
        plt.figure(figsize=(20,25))
        gs = gridspec.GridSpec(6,3,height_ratios=[10,10,10,10,10,0.5])
        a = np.arange(0.1,1.01,0.1)
        clevs_SST = np.append(-a[::-1],a)
        clevs_psl = 2*np.append(-a[::-1],a)
        clevs_pr = np.append(-a[::-1],a)

        for j,index_name in enumerate(SST_index_names):
            # read in regression on SST field
            file_name = SST_data_dir + '/SST_indices_'+model_name+'.nc'
            regress_coeffs_SST,lats_SST,lons_SST = read_regress_coeffs(file_name,'reg_coeff_'+index_name)
            # read in regression on sea level pressure
            file_name = data_dir + '/SST_index_circulation_regression_'+model_name+'_'+season+'.nc'
            regress_coeffs_psl,lats_psl,lons_psl = read_regress_coeffs(file_name,'regress_coeff_'+index_name+'_psl')
            # read in regression on precipitation
            regress_coeffs_pr,lats_pr,lons_pr = read_regress_coeffs(file_name,'regress_coeff_'+index_name+'_pr')

            ax = plt.subplot(gs[j,0],projection=ccrs.PlateCarree(central_longitude=180.))
            cs_SST = contour_plot(regress_coeffs_SST,clevs_SST,lats_SST,lons_SST,title=index_name+' SST') 
            ax = plt.subplot(gs[j,1],projection=ccrs.PlateCarree(central_longitude=180.))
            cs_psl = contour_plot(0.01*regress_coeffs_psl,clevs_psl,lats_psl,lons_psl,title=index_name+' psl',extent=[60,300,-30,60])
            ax = plt.subplot(gs[j,2],projection=ccrs.PlateCarree(central_longitude=180.))
            cs_pr = contour_plot(86400*regress_coeffs_pr,clevs_pr,lats_pr,lons_pr,title=index_name+' pr',cmap='RdBu',extent=[60,150,20,50])

        ax = plt.subplot(gs[j+1,0])
        make_plots.colorbar(ax,cs_SST)
        ax = plt.subplot(gs[j+1,1])
        make_plots.colorbar(ax,cs_psl)
        ax = plt.subplot(gs[j+1,2])
        make_plots.colorbar(ax,cs_pr)


        # save
        #plt.subplots_adjust(hspace=0.4)
        save_file_name = figure_dir+'/regress_SST_indices_circulation_'+model_name+'_'+season+'.png'
        print('saving to %s' % (save_file_name))
        plt.savefig(save_file_name,bbox_inches='tight')

    except: 
        message = 'Error, skipping model ' + model_name 
        print(message)

