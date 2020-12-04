""" Calculate EOFs of upper level zonal wind in the East Asian region on either
interannual of decadal timescales and regress onto SST in a range of models."""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
from eofs.standard import Eof
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
figure_dir = os.path.join(repo_dir,'figures/A2_1_SSTs_East_Asian_circulation/jet_EOFs')
sys.path.append(analysis_functions_dir)

from read_in_data import files_in_directory, read_in_variable, calculate_annual_mean, running_mean, save_file, read_spatial_dimensions, read_time_dimension
from regress_map import regress_map
from significance_testing import confidence_intervals

# define East Asian region
lon_min, lon_max = 70,150
lat_min, lat_max = 20,50
season = 'DJF'
apply_running_mean = 11 #None
plot_multimodelmean = False
model_name_list = ['BCC-CSM2-MR','CESM2','CNRM-CM6-1-HR','HadGEM3-GC31-LL','MPI-ESM1-2-LR'] #['BCC-CSM2-MR','CNRM-CM6-1-HR']#,'EC-Earth3','HadGEM3-GC31-LL','MPI-ESM1-2-LR'] ##ML.model_name_list # ['HadGEM3-GC31-LL']#['BCC-CSM2-MR'] #,'CNRM-CM6-1-HR']#,'EC-Earth3','HadGEM3-GC31-LL','MPI-ESM1-2-LR']  #ML.model_name_list
#model_name_list.append('ERA20C')
#model_name_list = ['BCC-CSM2-MR','FGOALS-g3','CanESM5','CNRM-CM6-1','ACCESS-CM2','EC-EARTH','INM-CM4-8','MIROC6','HadGEM3-GC31-LL','MPI-ESM1-2-LR','MRI-ESM2-0','GISS-E2-1-G']


def calculate_U_EOF(U,SST,THF,lats_ua,lons_ua,lats_SST,lons_SST,lats_THF,lons_THF,lat_min=lat_min,lat_max=lat_max,lon_min=lon_min,lon_max=lon_max,npcs=3):
    """Function to select a given region and return the first few principal component time series
    then regress the pcs back onto the zonal wind and SST."""
    # select region
    lat_mask = (lats_ua>=lat_min) & (lats_ua<=lat_max)
    lon_mask = (lons_ua>=lon_min) & (lons_ua<=lon_max)
    #print(lats.shape,lons.shape,U.shape,lats[lat_mask].shape,lons[lon_mask].shape)
    U_region = U[:,lat_mask,:][:,:,lon_mask]
    U_climatology = np.mean(U,axis=0)

    # Calculate EOFs
    coslat = np.cos(np.deg2rad(lats_ua[lat_mask]))
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(U_region, weights=wgts)
    pcs = solver.pcs(npcs=npcs, pcscaling=1)
    variance_fraction = solver.varianceFraction()

    # perform regressions
    regress_U = np.zeros([npcs,lats_ua.shape[0],lons_ua.shape[0]])
    regress_SST = np.zeros([npcs,lats_SST.shape[0],lons_SST.shape[0]])
    regress_THF = np.zeros([npcs,lats_THF.shape[0],lons_THF.shape[0]])
    for pc_number in np.arange(npcs):
        regress_U[pc_number,:,:] = regress_map(pcs[:,pc_number],U,map_type='corr')[0]
        regress_SST[pc_number,:,:] = regress_map(pcs[:,pc_number],SST,map_type='corr')[0]
        regress_THF[pc_number,:,:] = regress_map(pcs[:,pc_number],THF,map_type='corr')[0]

    return pcs,regress_U,regress_SST,regress_THF,variance_fraction[:npcs],U_climatology


def contour_plot(field,clevs,lats,lons,climatology=None,clim_clevs=None,lats_ua=None,lons_ua=None,calculate_significance=True,title_fontsize=20,title='',cmap='RdBu_r',extent=[0,359.99,-10,60],plot_box=False): #[60,210,0,60]
    m = make_plots.make_map_plot()
    if calculate_significance==True:
        pvals = t_test_autocorr(data_all,data_all[mask,:,:],autocorr=0)
        field = np.mean(data_all[mask],axis=0)
        cs = m.add_filled_contours(lons,lats,field,clevs,cmap=cmap)
        plt.contour(lons,lats,pvals,np.array([0.05]),colors='k',transform=ccrs.PlateCarree())
        title = title + ' (' + str(data_all[mask].shape[0]) + ')'
    else: cs = m.add_filled_contours(lons,lats,field,clevs,cmap=cmap)
    if climatology is not None: plt.contour(lons_ua,lats_ua,climatology,clim_clevs,colors='0.75',transform=ccrs.PlateCarree())
    if plot_box == True:
        plt.plot([70,150],[20,20],color='k',transform=ccrs.PlateCarree())
        plt.plot([70,150],[50,50],color='k',transform=ccrs.PlateCarree())
        plt.plot([70,70],[20,50],color='k',transform=ccrs.PlateCarree())
        plt.plot([150,150],[20,50],color='k',transform=ccrs.PlateCarree())
    plt.title(title,fontsize=title_fontsize)
    make_plots.add_lat_lon(ax)
    m.geography(ax,extent=extent)
    ax.set_aspect(1.5)
    return cs


# set up for plotting
# contour levels
a = 0.2*np.arange(1,5.1,1) # 0.2
clevs_ua = np.append(-a[::-1],a)
a = 0.3*np.arange(0.2,1.1,0.2) # 0.3
clevs_SST = clevs_ua #np.append(-a[::-1],a)
clevs_THF = clevs_ua # 2
a = np.arange(10,61,10)
clim_clevs = np.append(-a[::-1],a)
plot_pc = 0

if plot_multimodelmean == False:
    plt.figure(figsize=(20,13)) # 20,16
    gs = gridspec.GridSpec(len(model_name_list)+1,3,height_ratios=len(model_name_list)*[10]+[0.5])


for i, model_name in enumerate(model_name_list):

    try:

        # check if a regression file exists and read in data
        try: 
            if apply_running_mean is not None: regression_file_name = saving_dir + '/regression_' + model_name + '_ua200_EOFs_SST_rm' + str(apply_running_mean) + '_' + season + '_corr.nc'
            else: regression_file_name = saving_dir + '/regression_' + model_name + '_ua200_EOFs_SST_'+season+'.nc'
            levs,lats_ua,lons_ua = read_spatial_dimensions(regression_file_name,lat_name='lats_ua',lon_name='lons_ua')
            levs,lats_SST,lons_SST = read_spatial_dimensions(regression_file_name,lat_name='lats_SST',lon_name='lons_SST')
            levs,lats_THF,lons_THF = read_spatial_dimensions(regression_file_name,lat_name='lats_THF',lon_name='lons_THF')
            times,calendar,units = read_time_dimension(regression_file_name)
            nc = Dataset(regression_file_name,'r')
            U_climatology = nc.variables['U_climatology'][:]
            regress_SST = nc.variables['regress_SST'][:]
            regress_U = nc.variables['regress_U'][:]
            regress_THF = nc.variables['regress_THF'][:]
            variance_fraction = nc.variables['variance_fraction'][:]
            print('Read in %s' % (regression_file_name))

        # otherwise read in SST and zonal wind files and perform regression, saving these
        except:
            print('Cannot find regression file so reading in files to create one.')
            # read in SST data
            if model_name != 'ERA20C':
                data_path0 = '/network/aopp/hera/mad/patterson/CMIP6/data/tos_regridded/'
                full_path = data_path0 + ML.model_institute[model_name] + '/' + model_name + '/tos/'
                list_of_files = files_in_directory(full_path,concat_directory=True,include_files_with=ML.ensemble_id[model_name])
                tos_data,lats_SST,lons_SST,levs_SST,times_SST,calendar_SST,t_units_SST = read_in_variable(list_of_files[:],'tos')
            else:
                file_name = ['/network/group/aopp/met_data/MET003_ERA20C/data/tos/mon/tos_mon_ERA20C_2.5x2.5_189002-201012.nc']
                tos_data,lats_SST,lons_SST,levs_SST,times,calendar,t_units = read_in_variable(file_name,'sst')
            SST_am, years_SST = calculate_annual_mean(tos_data,times_SST,calendar_SST,t_units_SST,season=season)

            # set all unreasonably large values to NaN
            SST_am[np.abs(SST_am) > 1e3] = np.nan

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
            #THF_am, years_THF = calculate_annual_mean(THF,times_THF,calendar_THF,t_units_THF,season=season)

            # read in zonal wind
            data_path0 = '/network/aopp/hera/mad/patterson/CMIP6/data/piControl/ua/'
            full_path = data_path0 + ML.model_institute[model_name] + '/' + model_name + '/ua/'
            list_of_files = [full_path+'ua_Amon_'+model_name+'_piControl_'+ML.ensemble_id[model_name]+'_'+season+'.nc']
            ua200_am,lats_ua,lons_ua,levs_ua,times_ua,calendar_ua,t_units_ua = read_in_variable(list_of_files[:],'ua',chosen_level=20000)

            if season == 'DJF':
                ua200_am = ua200_am[::-1][1:][::-1]
                THF_am = THF_am[::-1][1:][::-1]

            if apply_running_mean is not None:
                ua200_am = running_mean(ua200_am,apply_running_mean)
                SST_am = running_mean(SST_am,apply_running_mean)
                THF_am = running_mean(THF_am,apply_running_mean)

            # calculate EOFs and do regression
            pcs,regress_U,regress_SST,regress_THF,variance_fraction,U_climatology = calculate_U_EOF(ua200_am,SST_am,THF_am,lats_ua,lons_ua,lats_SST,lons_SST,lats_THF,lons_THF)

            # save 
            if apply_running_mean is not None: 
                f = saving_dir + '/regression_' + model_name + '_ua200_EOFs_SST_rm' + str(apply_running_mean) + '_' +season+'_corr.nc'
                N = int(apply_running_mean/2)
                print(times_ua.shape,pcs.shape)
                if season != 'DJF': times_ua = times_ua[N:][::-1][N:][::-1]
                else: times_ua = times_ua[N:][::-1][N+1:][::-1]
                print(times_ua.shape)
            else: 
                f = saving_dir + '/regression_' + model_name + '_ua200_EOFs_SST_'+season+'.nc'
            description = 'Regressions of EOF analysis of the East Asian jet onto U200 and SST for the model: ' + model_name \
                    +' using U over the longitudes '+str(lon_min)+' - '+str(lon_max)+' and latitudes'+str(lat_min)+' - '+str(lat_max)  
            save = save_file(f,description)
            save.add_dimension(lats_ua,'lats_ua')
            save.add_dimension(lons_ua,'lons_ua')
            save.add_dimension(lats_SST,'lats_SST')
            save.add_dimension(lons_SST,'lons_SST')
            save.add_dimension(lats_THF,'lats_THF')
            save.add_dimension(lons_THF,'lons_THF')
            save.add_times(times_ua,calendar_ua,t_units_ua,time_name='times')
            save.add_dimension(np.arange(3),'pc_number')
            save.add_variable(pcs,'pcs',('times','pc_number'))
            save.add_variable(variance_fraction,'variance_fraction',('pc_number'))
            save.add_variable(U_climatology,'U_climatology',('lats_ua','lons_ua'))
            save.add_variable(regress_SST,'regress_SST',('pc_number','lats_SST','lons_SST',))
            save.add_variable(regress_THF,'regress_THF',('pc_number','lats_THF','lons_THF',))
            save.add_variable(regress_U,'regress_U',('pc_number','lats_ua','lons_ua',))
            save.close_file()
            print('saved to %s' % (f))

        # plotting for individual models
        if plot_multimodelmean == False:

            #if model_name in ['CESM2','HadGEM3-GC31-LL']:
            #    regress_U[plot_pc,:,:] = -regress_U[plot_pc,:,:]
            #    regress_SST[plot_pc,:,:] = -regress_SST[plot_pc,:,:]
            #    regress_THF[plot_pc,:,:] = -regress_THF[plot_pc,:,:]

            ax = plt.subplot(gs[i,0],projection=ccrs.PlateCarree(central_longitude=180.))
            cs1 = contour_plot(regress_U[plot_pc,:,:],clevs_ua,lats_ua,lons_ua,climatology=U_climatology,clim_clevs=clim_clevs,lats_ua=lats_ua,lons_ua=lons_ua,calculate_significance=False,plot_box=True)
            if i == 0: plt.title('U200',fontsize=25)
            #ax.text(-0.2,0,model_name+' (%.f%%)' % (100*variance_fraction[plot_pc]),rotation=90,transform=ax.transAxes,fontsize=15)
            ax.text(-0.2,0,model_name,rotation=90,transform=ax.transAxes,fontsize=15)
            ax = plt.subplot(gs[i,1],projection=ccrs.PlateCarree(central_longitude=180.))
            cs2 = contour_plot(regress_SST[plot_pc,:,:],clevs_SST,lats_SST,lons_SST,climatology=U_climatology,clim_clevs=clim_clevs,lats_ua=lats_ua,lons_ua=lons_ua,calculate_significance=False)
            if i == 0: plt.title('SST',fontsize=25)
            ax = plt.subplot(gs[i,2],projection=ccrs.PlateCarree(central_longitude=180.))
            cs3 = contour_plot(regress_THF[plot_pc,:,:],clevs_THF,lats_THF,lons_THF,climatology=U_climatology,clim_clevs=clim_clevs,lats_ua=lats_ua,lons_ua=lons_ua,calculate_significance=False)
            if i == 0: plt.title('THF',fontsize=25)

            if i == 0:
                for j,cs in enumerate([cs1,cs2,cs3]):
                    ax = plt.subplot(gs[len(model_name_list),j])
                    make_plots.colorbar(ax,cs,orientation='horizontal')

    except:
        print('Skipping ' + model_name)


if apply_running_mean is not None: figure_name = figure_dir + '/U200_EOF'+str(plot_pc+1)+'_individual_models_SST_THF_'+season+'_rm'+str(apply_running_mean)+'_corr_global.png'
else: figure_name = figure_dir + '/U200_EOF'+str(plot_pc+1)+'_individual_models_SST_THF_'+season+'.png'

print('saving to %s' % (figure_name))
plt.savefig(figure_name,bbox_inches='tight')
