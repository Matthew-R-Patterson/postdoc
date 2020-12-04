""" Creates composites of active/neutral Pacific SST years along with 
Indian Ocean SST composites. """

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset,num2date
from matplotlib import gridspec
import cartopy.crs as ccrs
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

import make_plots
analysis_functions_dir = os.path.join(repo_dir,'analysis_functions')
loading_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/SST_index_data')
figure_dir = os.path.join(repo_dir,'figures/A2_1_SSTs_East_Asian_circulation/IO_Pacific_SST_composites')
sys.path.append(analysis_functions_dir)
from read_in_data import files_in_directory, read_spatial_dimensions, read_time_dimension, read_in_variable, calculate_annual_mean, running_mean, save_file
from significance_testing import t_test_autocorr
from interpolate_grid import interpolate_grid

model_institute = {'AWI-CM-1-1-MR': 'AWI','BCC-CSM2-MR': 'BCC','CAMS-CSM1-0':'CAMS', 'FGOALS-f3-L':'CAS', 'FGOALS-g3':'CAS', 'CanESM5':'CCCma', 'CanESM5-CanOE':'CCCma', 'CNRM-CM6-1':'CNRM-CERFACS', 'CNRM-CM6-1-HR':'CNRM-CERFACS', 'CNRM-ESM2-1':'CNRM-CERFACS', 'ACCESS-ESM1-5':'CSIRO', 'ACCESS-CM2':'CSIRO-ARCCSS' ,'EC-Earth3':'EC-Earth-Consortium', 'EC-Earth3-Veg':'EC-Earth-Consortium', 'INM-CM4-8':'INM' ,'INM-CM5-0':'INM', 'IPSL-CM6A-LR':'IPSL', 'MIROC6':'MIROC', 'MIROC-ES2L':'MIROC', 'HadGEM3-GC31-LL':'MOHC', 'UKESM1-0-LL':'MOHC', 'MPI-ESM1-2-LR':'MPI-M', 'MRI-ESM2-0':'MRI', 'GISS-E2-1-G':'NASA-GISS', 'CESM2':'NCAR', 'CESM2-WACCM':'NCAR', 'NorESM2-LM':'NCC', 'NorESM2-MM':'NCC', 'GFDL-CM4':'NOAA-GFDL' ,'GFDL-ESM4':'NOAA-GFDL', 'NESM3':'NUIST','MCM-UA-1-0':'UA'}

ensemble_id = {'AWI-CM-1-1-MR': 'r1i1p1f1','BCC-CSM2-MR': 'r1i1p1f1','CAMS-CSM1-0':'r1i1p1f1', 'FGOALS-f3-L':'r1i1p1f1', 'FGOALS-g3':'r1i1p1f1', 'CanESM5':'r1i1p1f1', 'CanESM5-CanOE':'r1i1p2f1', 'CNRM-CM6-1':'r1i1p1f2', 'CNRM-CM6-1-HR':'r1i1p1f2', 'CNRM-ESM2-1':'r1i1p1f2', 'ACCESS-ESM1-5':'r1i1p1f1', 'ACCESS-CM2':'r1i1p1f1' ,'EC-Earth3':'r1i1p1f1', 'EC-Earth3-Veg':'r1i1p1f1', 'INM-CM4-8':'r1i1p1f1' ,'INM-CM5-0':'r1i1p1f1', 'IPSL-CM6A-LR':'r1i1p1f1', 'MIROC6':'r1i1p1f1', 'MIROC-ES2L':'r1i1p1f2', 'HadGEM3-GC31-LL':'r1i1p1f1', 'UKESM1-0-LL':'r1i1p1f2', 'MPI-ESM1-2-LR':'r1i1p1f1', 'MRI-ESM2-0':'r1i1p1f1', 'GISS-E2-1-G':'r1i1p1f1', 'CESM2':'r1i1p1f1', 'CESM2-WACCM':'r1i1p1f1', 'NorESM2-LM':'r1i1p1f1', 'NorESM2-MM':'r1i1p1f1', 'GFDL-CM4':'r1i1p1f1' ,'GFDL-ESM4':'r1i1p1f1', 'NESM3':'r1i1p1f1','MCM-UA-1-0':'r1i1p1f1'}

model_name_list = ['AWI-CM-1-1-MR','BCC-CSM2-MR','CAMS-CSM1-0', 'FGOALS-f3-L', 'FGOALS-g3', 'CanESM5', 'CanESM5-CanOE', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'CNRM-ESM2-1', 'ACCESS-ESM1-5', 'ACCESS-CM2' ,'EC-Earth3', 'EC-Earth3-Veg', 'INM-CM4-8' ,'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MIROC-ES2L', 'HadGEM3-GC31-LL', 'UKESM1-0-LL', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'GISS-E2-1-G', 'CESM2', 'CESM2-WACCM', 'NorESM2-LM', 'NorESM2-MM', 'GFDL-CM4' ,'GFDL-ESM4', 'NESM3','MCM-UA-1-0']
#model_name_list = ['HadGEM3-GC31-LL','CanESM5-CanOE', 'CNRM-CM6-1']
#model_name_list =['CanESM5']# ['CNRM-CM6-1', 'CNRM-CM6-1-HR', 'CNRM-ESM2-1', 'ACCESS-ESM1-5', 'ACCESS-CM2']# , 'EC-Earth3-Veg', 'INM-CM4-8' ,'INM-CM5-0', 'HadGEM3-GC31-LL', 'UKESM1-0-LL', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'CESM2', 'CESM2-WACCM'] #'BCC-CSM2-MR' CAMS-CSM1-0
var_name = 'psl' #psl'
chosen_level = None #None #20000
season = 'JJA'
average_over_models = True
decadal_mean = True
contour_factor = {'psl':1,'ua':0.5}

def prep_index(nc,index_name,times_SST,calendar_SST,t_units_SST,season):
    index_ts = nc.variables[index_name][:]
    index_annual_mean, years = calculate_annual_mean(index_ts,times_SST,calendar_SST,t_units_SST,season=season,decadal_mean=decadal_mean)
    index_annual_mean = (index_annual_mean - np.mean(index_annual_mean))/np.std(index_annual_mean)
    #index_running_mean = running_mean(index_annual_mean,N)
    return index_annual_mean

def contour_plot(row,column,field,clevs,lats,lons,data_all,mask,climatology=None,clim_clevs=None,calculate_significance=True,title_fontsize=35,title='',cmap='RdBu_r',extent=[50,290,-30,60]):
    ax = plt.subplot(gs[row,column],projection=ccrs.PlateCarree(central_longitude=180.))
    m = make_plots.make_map_plot()
    if calculate_significance==True:
        pvals = t_test_autocorr(data_all,data_all[mask,:,:],autocorr=0)
        field = np.mean(data_all[mask],axis=0)
        cs = m.add_filled_contours(lons,lats,field,clevs,cmap=cmap)
        plt.contour(lons,lats,pvals,np.array([0.05]),colors='k',transform=ccrs.PlateCarree())
        title = title + ' (' + str(data_all[mask].shape[0]) + ')'
    else: cs = m.add_filled_contours(lons,lats,field,clevs,cmap=cmap)
    if climatology is not None: plt.contour(lons,lats,climatology,clim_clevs,colors='0.75',transform=ccrs.PlateCarree())
    plt.title(title,fontsize=title_fontsize)
    make_plots.add_lat_lon(ax)
    m.geography(ax,extent=extent)
    ax.set_aspect(1.5)
    return cs


def take_decadal_mean(variable,times,calendar,units):
    """ Take the decadal mean of a set of data. Can be
    monthly or annual data."""
    dates = num2date(times,calendar=calendar,units=units)
    years = np.array([])
    for day in dates:
        years = np.append(years,day.timetuple()[0])
    years = years - years%10 # split years based on decade
    new_dims = variable.shape[1:]
    new_dims = (1,) + new_dims
    for i,yr in enumerate(np.unique(years)):
        mask = (years == yr)
        if i == 0:
            annual_mean = np.mean(variable[mask],axis=0).reshape(new_dims)
        else:
            new_annual_mean = np.mean(variable[mask],axis=0).reshape(new_dims)
            annual_mean = np.append(annual_mean,new_annual_mean,axis=0)
    return annual_mean, np.unique(years)

   
def concatenate_model_composites(composite_all_models_old,model_data,lons,lats,mask):
    if mask is not None: 
        composite_interp = interpolate_grid(np.mean(model_data[mask],axis=0),lons,lats,standard_lons,standard_lats)
        n = model_data[mask].shape[0]
    else: 
        composite_interp = interpolate_grid(model_data,lons,lats,standard_lons,standard_lats)
        n = 1
    composite_all_models = np.append(composite_all_models_old, composite_interp.reshape(1,n_lats,n_lons),axis=0)
    return composite_all_models, n

for i, model_name in enumerate(model_name_list):

    try:
        # read in SST index data
        SST_index_file_name = loading_dir + '/SST_indices_'+model_name+'_'+ ensemble_id[model_name]+'.nc'
        nc_SST = Dataset(SST_index_file_name,'r')
        times_SST = nc_SST.variables['times'][:]
        calendar_SST = nc_SST.variables['times'].calendar
        t_units_SST = nc_SST.variables['times'].units
        NINO34 = prep_index(nc_SST,'NINO34',times_SST,calendar_SST,t_units_SST,season)
        IOBM = prep_index(nc_SST,'IOBM',times_SST,calendar_SST,t_units_SST,season)

        # read in variable
        #data_path0 = '/network/group/aopp/predict/AWH007_BEFORT_CMIP6/piControl/'
        #data_path1 = '/piControl/Amon/'+var_name+'/gn/latest'
        data_path0 = '/network/aopp/hera/mad/patterson/CMIP6/data/piControl/ua/'
        data_path1 = '/' + var_name + '/'
        full_path = data_path0 + model_institute[model_name] + '/' + model_name + data_path1
        #list_of_files = files_in_directory(full_path,concat_directory=True,exclude_files_with='AER')
        list_of_files = [full_path+var_name+'_Amon_'+model_name+'_piControl_'+ensemble_id[model_name]+'_'+season+'.nc'] 
        toc1 = time.perf_counter()
        data_am,lats_var,lons_var,levs_var,times_var,calendar_var,t_units_var = read_in_variable(list_of_files[:],var_name,chosen_level=chosen_level)
        toc2 = time.perf_counter()
        if decadal_mean == True: data_am, years = take_decadal_mean(data_am,times_var,calendar_var,t_units_var)
        #data_am, years = calculate_annual_mean(var_data,times_var,calendar_var,t_units_var,season=season,decadal_mean=decadal_mean)
        print(toc2 - toc1)
        print(data_am.shape,IOBM.shape)
        climatology_model = np.mean(data_am,axis=0)
        data_am = data_am - np.mean(data_am,axis=0).reshape(1,lats_var.shape[0],lons_var.shape[0]) # calculate anomalies
        if var_name == 'psl': data_am = 0.01 * data_am # convert to hPa
        elif var_name == 'pr': data_am = 86400 * data_am # convert to mm/day
    
        # create composite masks
        NINO34_plus = NINO34 > 1
        NINO34_minus = NINO34 <- 1
        NINO34_neutral = (NINO34 > -0.5) & (NINO34 < 0.5)
        IOBM_plus = IOBM > 1
        IOBM_minus = IOBM <- 1
        IOBM_neutral = (IOBM > -0.5) & (IOBM < 0.5) 


        if average_over_models == False:
            # plot variable based on those composites
            a = contour_factor[var_name] * np.arange(0.2,2.1,0.2)
            clevs = np.append(-a[::-1],a)
            plt.figure(figsize=(25,15))
            gs = gridspec.GridSpec(4,3,height_ratios=[10,10,10,0.5])
            if decadal_mean == True:
                cs = contour_plot(0,0,None,clevs,lats_var,lons_var,data_am,mask=(IOBM_plus),title='IOBM+')
                cs = contour_plot(1,0,None,clevs,lats_var,lons_var,data_am,mask=(IOBM_minus),title='IOBM-')
                cs = contour_plot(2,0,np.mean(data_am[IOBM_plus,:,:],axis=0) - np.mean(data_am[IOBM_minus,:,:],axis=0),clevs,lats_var,lons_var,None,mask=None,calculate_significance=False,title='Difference')
            else:
                cs = contour_plot(0,0,None,clevs,lats_var,lons_var,data_am,mask=(NINO34_minus & IOBM_plus),title='IOBM+ NINO3.4-')
                cs = contour_plot(0,1,None,clevs,lats_var,lons_var,data_am,mask=(NINO34_neutral & IOBM_plus),title='IOBM+ NINO3.4 neutral')
                cs = contour_plot(0,2,None,clevs,lats_var,lons_var,data_am,mask=(NINO34_plus & IOBM_plus),title='IOBM+ NINO3.4+')
                cs = contour_plot(1,0,None,clevs,lats_var,lons_var,data_am,mask=(NINO34_minus & IOBM_minus),title='IOBM- NINO3.4-')
                cs = contour_plot(1,1,None,clevs,lats_var,lons_var,data_am,mask=(NINO34_neutral & IOBM_minus),title='IOBM- NINO3.4 neutral')
                cs = contour_plot(1,2,None,clevs,lats_var,lons_var,data_am,mask=(NINO34_plus & IOBM_minus),title='IOBM- NINO3.4+')
                cs = contour_plot(2,0,np.mean(data_am[NINO34_minus & IOBM_plus,:,:],axis=0) - np.mean(data_am[NINO34_minus & IOBM_minus,:,:],axis=0),clevs,lats_var,lons_var,None,mask=None,calculate_significance=False,title='Difference')
                cs = contour_plot(2,1,np.mean(data_am[NINO34_neutral & IOBM_plus,:,:],axis=0) - np.mean(data_am[NINO34_neutral & IOBM_minus,:,:],axis=0),clevs,lats_var,lons_var,None,mask=None,calculate_significance=False,title='Difference')
                cs = contour_plot(2,2,np.mean(data_am[NINO34_plus & IOBM_plus,:,:],axis=0) - np.mean(data_am[NINO34_plus & IOBM_minus,:,:],axis=0),clevs,lats_var,lons_var,None,mask=None,calculate_significance=False,title='Difference')
            ax = plt.subplot(gs[3,:])
            make_plots.colorbar(ax,cs,orientation='horizontal')
            #save figure
            if decadal_mean == True: save_file_name = figure_dir+'/IOBM_NINO34_composites_'+var_name+'_decadal_'+model_name+'_'+season+'.png'
            else: save_file_name = figure_dir+'/IOBM_NINO34_composites_'+var_name+'_'+model_name+'_'+season+'.png'
            print('saving to %s' % (save_file_name))
            plt.savefig(save_file_name,bbox_inches='tight')

        elif average_over_models == True:
            # add composites to arrays
            if i == 0:
                standard_lats = np.arange(-88,88.1,2)
                standard_lons = np.arange(0,360.1,2)
                n_lats,n_lons = standard_lats.shape[0],standard_lons.shape[0]
                if decadal_mean == False:
                    allModels_IOBM_plus_NINO_minus = np.zeros([0,n_lats,n_lons])
                    allModels_IOBM_plus_NINO_neutral = np.zeros([0,n_lats,n_lons])
                    allModels_IOBM_plus_NINO_plus = np.zeros([0,n_lats,n_lons])
                    allModels_IOBM_minus_NINO_minus = np.zeros([0,n_lats,n_lons])
                    allModels_IOBM_minus_NINO_neutral = np.zeros([0,n_lats,n_lons])
                    allModels_IOBM_minus_NINO_plus = np.zeros([0,n_lats,n_lons])
                    allModels_climatology = np.zeros([0,n_lats,n_lons])
                else: 
                    allModels_IOBM_plus = np.zeros([0,n_lats,n_lons])
                    allModels_IOBM_minus = np.zeros([0,n_lats,n_lons])
                    allModels_climatology = np.zeros([0,n_lats,n_lons])

            if decadal_mean == False:
                print(allModels_IOBM_plus_NINO_minus.shape,data_am.shape,(NINO34_minus & IOBM_plus).shape)
                allModels_IOBM_plus_NINO_minus, n_IOBM_plus_NINO_minus = concatenate_model_composites(allModels_IOBM_plus_NINO_minus,data_am,lons_var,lats_var,(NINO34_minus & IOBM_plus))
                allModels_IOBM_plus_NINO_neutral, n_IOBM_plus_NINO_neutral = concatenate_model_composites(allModels_IOBM_plus_NINO_neutral,data_am,lons_var,lats_var,(NINO34_neutral & IOBM_plus))
                allModels_IOBM_plus_NINO_plus, n_IOBM_plus_NINO_plus = concatenate_model_composites(allModels_IOBM_plus_NINO_plus,data_am,lons_var,lats_var,(NINO34_plus & IOBM_plus))
                allModels_IOBM_minus_NINO_minus, n_IOBM_minus_NINO_minus = concatenate_model_composites(allModels_IOBM_minus_NINO_minus,data_am,lons_var,lats_var,(NINO34_minus & IOBM_minus))
                allModels_IOBM_minus_NINO_neutral, n_IOBM_minus_NINO_neutral = concatenate_model_composites(allModels_IOBM_minus_NINO_neutral,data_am,lons_var,lats_var,(NINO34_neutral & IOBM_minus))
                allModels_IOBM_minus_NINO_plus, n_IOBM_minus_NINO_plus = concatenate_model_composites(allModels_IOBM_minus_NINO_plus,data_am,lons_var,lats_var,(NINO34_plus & IOBM_minus))
                allModels_climatology, n_climatology = concatenate_model_composites(allModels_climatology,climatology_model,lons_var,lats_var,mask=None)
            else: 
                allModels_IOBM_plus, n_IOBM_plus = concatenate_model_composites(allModels_IOBM_plus,data_am,lons_var,lats_var,(IOBM_plus))
                allModels_IOBM_minus, n_IOBM_minus = concatenate_model_composites(allModels_IOBM_minus,data_am,lons_var,lats_var,(IOBM_minus))
                allModels_climatology, n_climatology = concatenate_model_composites(allModels_climatology,climatology_model,lons_var,lats_var,mask=None)

    except: 
        print('Error, skipping model %s' % (model_name))


if average_over_models == True:
    # set up figure for plotting
    a = np.arange(0.2,2.1,0.2)
    clevs = contour_factor[var_name] * np.append(-a[::-1],a)
    clim_clevs = 30 * np.append(-a[::-1],a)
    climatology = np.mean(allModels_climatology,axis=0)

    plt.figure(figsize=(25,15))
    gs = gridspec.GridSpec(4,3,height_ratios=[10,10,10,0.5])
    if decadal_mean == True:
        cs = contour_plot(0,0,np.nanmean(allModels_IOBM_plus,axis=0),clevs,standard_lats,standard_lons,None,climatology=climatology,clim_clevs=clim_clevs,mask=None,calculate_significance=False,title='IOBM+')
        cs = contour_plot(1,0,np.nanmean(allModels_IOBM_minus,axis=0),clevs,standard_lats,standard_lons,None,climatology=climatology,clim_clevs=clim_clevs,mask=None,calculate_significance=False,title='IOBM-')
        cs = contour_plot(2,0,np.nanmean(allModels_IOBM_plus,axis=0) - np.nanmean(allModels_IOBM_minus,axis=0),clevs,standard_lats,standard_lons,None,climatology=climatology,clim_clevs=clim_clevs,mask=None,calculate_significance=False,title='Difference')
        ax = plt.subplot(gs[3,0])
        make_plots.colorbar(ax,cs,orientation='horizontal')

    else:    
        cs = contour_plot(0,0,np.mean(allModels_IOBM_plus_NINO_minus,axis=0),clevs,standard_lats,standard_lons,None,climatology=climatology,clim_clevs=clim_clevs,mask=None,calculate_significance=False,title='IOBM+ NINO3.4-')
        cs = contour_plot(0,1,np.mean(allModels_IOBM_plus_NINO_neutral,axis=0),clevs,standard_lats,standard_lons,None,climatology=climatology,clim_clevs=clim_clevs,mask=None,calculate_significance=False,title='IOBM+ NINO3.4 neutral')
        cs = contour_plot(0,2,np.mean(allModels_IOBM_plus_NINO_plus,axis=0),clevs,standard_lats,standard_lons,None,climatology=climatology,clim_clevs=clim_clevs,mask=None,calculate_significance=False,title='IOBM+ NINO3.4+')
        cs = contour_plot(1,0,np.mean(allModels_IOBM_minus_NINO_minus,axis=0),clevs,standard_lats,standard_lons,None,climatology=climatology,clim_clevs=clim_clevs,mask=None,calculate_significance=False,title='IOBM- NINO3.4-')
        cs = contour_plot(1,1,np.mean(allModels_IOBM_minus_NINO_neutral,axis=0),clevs,standard_lats,standard_lons,None,climatology=climatology,clim_clevs=clim_clevs,mask=None,calculate_significance=False,title='IOBM- NINO3.4 neutral')
        cs = contour_plot(1,2,np.nanmean(allModels_IOBM_minus_NINO_plus,axis=0),clevs,standard_lats,standard_lons,None,climatology=climatology,clim_clevs=clim_clevs,mask=None,calculate_significance=False,title='IOBM- NINO3.4+')
        cs = contour_plot(2,0,np.nanmean(allModels_IOBM_plus_NINO_minus,axis=0) - np.nanmean(allModels_IOBM_minus_NINO_minus,axis=0),clevs,standard_lats,standard_lons,None,climatology=climatology,clim_clevs=clim_clevs,mask=None,calculate_significance=False,title='Difference')
        cs = contour_plot(2,1,np.nanmean(allModels_IOBM_plus_NINO_neutral,axis=0) - np.nanmean(allModels_IOBM_minus_NINO_neutral,axis=0),clevs,standard_lats,standard_lons,None,climatology=climatology,clim_clevs=clim_clevs,mask=None,calculate_significance=False,title='Difference')
        cs = contour_plot(2,2,np.nanmean(allModels_IOBM_plus_NINO_plus,axis=0) - np.nanmean(allModels_IOBM_minus_NINO_plus,axis=0),clevs,standard_lats,standard_lons,None,climatology=climatology,clim_clevs=clim_clevs,mask=None,calculate_significance=False,title='Difference')
        ax = plt.subplot(gs[3,:])
        make_plots.colorbar(ax,cs,orientation='horizontal')

    # save figure
    if decadal_mean == True: save_file_name = figure_dir+'/IOBM_NINO34_composites_'+var_name+'_multimodelmean_decadal_'+season+'.png'
    else: save_file_name = figure_dir+'/IOBM_NINO34_composites_'+var_name+'_multimodelmean_'+season+'.png'
    print('saving to %s' % (save_file_name))
    plt.savefig(save_file_name,bbox_inches='tight')
