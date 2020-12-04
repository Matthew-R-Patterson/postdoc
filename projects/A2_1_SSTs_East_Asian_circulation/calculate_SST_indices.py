""" Calculates the PDO index from monthly mean data 

1) Reads in SST data over the whole globe
2) Calculates the global mean
3) Selects the region 20N-70N, 120E-270E and removes the global mean 
and then removes the seasonal cycle at each gridpoint.
4) Calculates the first EOF in the given region
5) Saves to a file """

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
from eofs.standard import Eof
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
figure_dir = os.path.join(repo_dir,'figures/A2_1_SSTs_East_Asian_circulation')
saving_dir = os.path.join(repo_dir,'projects/A2_1_SSTs_East_Asian_circulation/SST_index_data')
sys.path.append(analysis_functions_dir)
import regress_map
from read_in_data import files_in_directory, read_spatial_dimensions, read_time_dimension, read_in_variable, calculate_annual_mean, running_mean, save_file

# list of models
model_name_list =  ['AWI-CM-1-1-MR','BCC-CSM2-MR','CAMS-CSM1-0', 'FGOALS-f3-L', 'FGOALS-g3', 'CanESM5', 'CanESM5-CanOE', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'CNRM-ESM2-1', 'ACCESS-ESM1-5', 'ACCESS-CM2' ,'EC-Earth3', 'EC-Earth3-Veg', 'INM-CM4-8' ,'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MIROC-ES2L', 'HadGEM3-GC31-LL', 'UKESM1-0-LL', 'MPI-ESM1-2-LR','MRI-ESM2-0', 'GISS-E2-1-G', 'CESM2','CESM2-WACCM', 'NorESM2-LM', 'NorESM2-MM', 'GFDL-CM4' ,'GFDL-ESM4', 'NESM3','MCM-UA-1-0'] 
#model_name_list =  ['CanESM5-CanOE', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'CNRM-ESM2-1', 'ACCESS-ESM1-5', 'ACCESS-CM2' ,'EC-Earth3', 'EC-Earth3-Veg', 'INM-CM4-8' ,'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MIROC-ES2L', 'HadGEM3-GC31-LL', 'UKESM1-0-LL', 'MPI-ESM1-2-LR','MRI-ESM2-0', 'GISS-E2-1-G', 'CESM2','CESM2-WACCM', 'NorESM2-LM', 'NorESM2-MM', 'GFDL-CM4' ,'GFDL-ESM4', 'NESM3','MCM-UA-1-0']

model_institute = {'AWI-CM-1-1-MR': 'AWI','BCC-CSM2-MR': 'BCC','CAMS-CSM1-0':'CAMS', 'FGOALS-f3-L':'CAS', 'FGOALS-g3':'CAS', 'CanESM5':'CCCma', 'CanESM5-CanOE':'CCCma', 'CNRM-CM6-1':'CNRM-CERFACS', 'CNRM-CM6-1-HR':'CNRM-CERFACS', 'CNRM-ESM2-1':'CNRM-CERFACS', 'ACCESS-ESM1-5':'CSIRO', 'ACCESS-CM2':'CSIRO-ARCCSS' ,'EC-Earth3':'EC-Earth-Consortium', 'EC-Earth3-Veg':'EC-Earth-Consortium', 'INM-CM4-8':'INM' ,'INM-CM5-0':'INM', 'IPSL-CM6A-LR':'IPSL', 'MIROC6':'MIROC', 'MIROC-ES2L':'MIROC', 'HadGEM3-GC31-LL':'MOHC', 'UKESM1-0-LL':'MOHC', 'MPI-ESM1-2-LR':'MPI-M', 'MRI-ESM2-0':'MRI', 'GISS-E2-1-G':'NASA-GISS', 'CESM2':'NCAR', 'CESM2-WACCM':'NCAR', 'NorESM2-LM':'NCC', 'NorESM2-MM':'NCC', 'GFDL-CM4':'NOAA-GFDL' ,'GFDL-ESM4':'NOAA-GFDL', 'NESM3':'NUIST','MCM-UA-1-0':'UA'}

def global_mean(data,lats):
    """ Calculates the global average over a given quantity. Assumes data of the form
    (time, lats, lons) or (lats, lons)"""
    cos_lats = np.cos(np.deg2rad(lats))
    mean_cos_lats = np.mean(cos_lats)
    if data.ndim == 2: 
        global_mean = np.mean(data * cos_lats.reshape(lats.shape[0],1)) / mean_cos_lats
    elif data.ndim == 3: 
        global_mean = np.mean(np.mean(data * cos_lats.reshape(1,lats.shape[0],1),axis=2),axis=1) / mean_cos_lats
    else: 
        print('Too many or too few dimensions')
        print('Data has dimensions: ',data.shape)
        raise Error
    return global_mean


def remove_annual_cycle(data,times,t_units,calendar):
    """ Calculates the annual cycle of monthly data by averaging for each month
    assumes data in the form (times, ...) """
    dates = num2date(times,calendar=calendar,units=t_units)
    months = np.array([])
    for day in dates:
        months = np.append(months,day.timetuple()[1])
    spatial_dims = data.shape[1:]
    mean_dims = (1,) + spatial_dims
    annual_cycle_removed = np.copy(data)
    for i in np.arange(1,12+1):
        annual_cycle_removed[months==(i)] = data[months==(i)] - np.mean(data[months==(i)],axis=0).reshape(mean_dims)
    return annual_cycle_removed


def calculate_PDO(data,lats,lons,times,t_units,calendar):
    """ Calculate the Pacific Decadal Oscillation index as the first PC of SST
    between 20N and 70N
    See Newman et al (2016) doi:10.1175/JCLI-D-15-0508.1"""
    data[np.abs(data)>1e3] = np.nan # set unreasonably high values to NaN
    global_mean_removed = data - global_mean(data,lats).reshape(times.shape[0],1,1)
    annual_cycle_removed = remove_annual_cycle(global_mean_removed,times,t_units,calendar)
    lat_min, lat_max = 20, 70
    lon_min, lon_max = 120,270
    lat_mask = (lats>=lat_min)&(lats<=lat_max)
    lon_mask = (lons>=lon_min)&(lons<=lon_max)
    N_Pacific_SST = annual_cycle_removed[:,lat_mask,:][:,:,lon_mask]
    coslat = np.cos(np.deg2rad(lats[lat_mask]))
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(N_Pacific_SST, weights=wgts)
    EOF1 = solver.eofs(neofs=1)[0,:,:]
    PDO = solver.pcs(npcs=1, pcscaling=1).flatten()
    if np.nanmean(EOF1[:,lons[lon_mask]>210]) < 0: PDO = -PDO
    PDO = (PDO - np.mean(PDO))/np.std(PDO)
    return PDO


def calculate_IOBM(data,lats,lons,times,t_units,calendar):
    """ Calculate the Indian Ocean basin mode as the first EOF over the region 
    20S-20N, 40E-110E.
    See Yang et al (2007) doi:10.1029/2006GL028571"""
    data[np.abs(data)>1e3] = np.nan
    annual_cycle_removed = remove_annual_cycle(data,times,t_units,calendar)
    lat_min, lat_max = -20, 20
    lon_min, lon_max = 40,110
    lat_mask = (lats>=lat_min)&(lats<=lat_max)
    lon_mask = (lons>=lon_min)&(lons<=lon_max)
    IO_SST = annual_cycle_removed[:,lat_mask,:][:,:,lon_mask]
    coslat = np.cos(np.deg2rad(lats[lat_mask]))
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(IO_SST, weights=wgts)
    IOBM = solver.pcs(npcs=1, pcscaling=1).flatten()
    EOF1 = solver.eofs(neofs=1)[0,:,:]
    if np.nanmean(EOF1) < 0: IOBM = -IOBM
    IOBM = (IOBM - np.mean(IOBM))/np.std(IOBM)
    return IOBM


def calculate_IOBM2(data,lats,lons,times,t_units,calendar):
    """ Calculate the Indian Ocean basin mode as the average temperature 
    over the region 20S-20N, 40E-110E.
    See Yang et al (2007) doi:10.1029/2006GL028571"""
    data[np.abs(data)>1e3] = np.nan
    annual_cycle_removed = remove_annual_cycle(data,times,t_units,calendar)
    lat_min, lat_max = -20, 20
    lon_min, lon_max = 40,110
    lat_mask = (lats>=lat_min)&(lats<=lat_max)
    lon_mask = (lons>=lon_min)&(lons<=lon_max)
    IO_SST = annual_cycle_removed[:,lat_mask,:][:,:,lon_mask]
    IOBM = np.nanmean(np.nanmean(IO_SST,axis=1),axis=1)
    IOBM = (IOBM - np.mean(IOBM))/np.std(IOBM)
    return IOBM


def calculate_IOD(data,lats,lons,times,t_units,calendar):
    """ Calculate the Indian Ocean diplole 
    Calculated as the SST anomaly between the boxes
    50E-70E,10S-10N and 90E-110E,10S-ON/S
    See Saji et al (1999) doi: 10.1038/43854"""
    data[np.abs(data)>1e3] = np.nan # set unreasonably high values to NaN
    annual_cycle_removed = remove_annual_cycle(data,times,t_units,calendar)
    lat_minE, lat_maxE = -10, 0
    lon_minE, lon_maxE = 90,110
    lat_minW, lat_maxW = -10, 10
    lon_minW, lon_maxW = 50,70
    lat_maskE = (lats>=lat_minE)&(lats<=lat_maxE)
    lon_maskE = (lons>=lon_minE)&(lons<=lon_maxE)
    lat_maskW = (lats>=lat_minW)&(lats<=lat_maxW)
    lon_maskW = (lons>=lon_minW)&(lons<=lon_maxW)
    IO_SST_E = np.nanmean(np.nanmean(annual_cycle_removed[:,lat_maskE,:][:,:,lon_maskE],axis=1),axis=1)
    IO_SST_W = np.nanmean(np.nanmean(annual_cycle_removed[:,lat_maskW,:][:,:,lon_maskW],axis=1),axis=1)
    IOD = IO_SST_W - IO_SST_E
    IOD = (IOD - np.mean(IOD))/np.std(IOD)
    return IOD


def calculate_IPO(data,lats,lons,times,t_units,calendar):
    """ Calculate the Inter-decadal Pacific Oscillation index 
    Calculated as the first EOF of SST 60S to 60N over the
    Pacific  """
    data[np.abs(data)>1e3] = np.nan # set unreasonably high values to NaN
    annual_cycle_removed = remove_annual_cycle(data,times,t_units,calendar)
    lat_min, lat_max = -60, 60
    lon_min, lon_max = 120,270
    lat_mask = (lats>=lat_min)&(lats<=lat_max)
    lon_mask = (lons>=lon_min)&(lons<=lon_max)
    Pacific_SST = annual_cycle_removed[:,lat_mask,:][:,:,lon_mask]
    coslat = np.cos(np.deg2rad(lats[lat_mask]))
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(Pacific_SST, weights=wgts)
    IPO = solver.pcs(npcs=1, pcscaling=1).flatten()
    EOF1 = solver.eofs(neofs=1)[0,:,:]
    if np.nanmean(EOF1) < 0: IPO = -IPO
    IPO = (IPO - np.mean(IPO))/np.std(IPO)
    return IPO


def calculate_NINO34(data,lats,lons,times,t_units,calendar):
    """ defined as 5N-5S,190E-240E """
    data[np.abs(data)>1e3] = np.nan # set unreasonably high values to NaN
    annual_cycle_removed = remove_annual_cycle(data,times,t_units,calendar)
    lat_min, lat_max = -5, 5
    lon_min, lon_max = 190,240
    lat_mask = (lats>=lat_min)&(lats<=lat_max)
    lon_mask = (lons>=lon_min)&(lons<=lon_max)
    NINO34_SST = np.mean(np.mean(annual_cycle_removed[:,lat_mask,:][:,:,lon_mask],axis=1),axis=1)
    NINO34_SST = (NINO34_SST - np.mean(NINO34_SST))/np.std(NINO34_SST)
    return NINO34_SST


# read in data
id = 'r102i1p1f1' #'r101i1p1f1' #'r1i1p2f1' #'r2i1p1f1' #'r1i1p5f1' #'r1i1p3f1' #'r1i1p1f3' #'r1i2p1f1' # 'r1i1p1f2' #'r1i1p1f1' # ensemble member

for i, model_name in enumerate(model_name_list):
    try: 
        if model_name is not 'ERA20C': 
            data_path0 = '/network/aopp/hera/mad/patterson/CMIP6/data/tos_regridded/'
            full_path = data_path0 + model_institute[model_name] + '/' + model_name + '/tos/'
            list_of_files = files_in_directory(full_path,concat_directory=True,include_files_with=id)
            tos_data,lats,lons,levs,times,calendar,t_units = read_in_variable(list_of_files[:],'tos')
        else: 
            list_of_files = ['/network/group/aopp/met_data/MET003_ERA20C/data/tos/mon/tos_mon_ERA20C_2.5x2.5_189002-201012.nc']
            tos_data,lats,lons,levs,times,calendar,t_units = read_in_variable(list_of_files[:],'sst')
        PDO = calculate_PDO(tos_data,lats,lons,times,t_units,calendar)
        IPO = calculate_IPO(tos_data,lats,lons,times,t_units,calendar)
        IOBM = calculate_IOBM(tos_data,lats,lons,times,t_units,calendar)
        IOBM2 = calculate_IOBM2(tos_data,lats,lons,times,t_units,calendar)
        IOD = calculate_IOD(tos_data,lats,lons,times,t_units,calendar)
        NINO34 = calculate_NINO34(tos_data,lats,lons,times,t_units,calendar)

        # calculate regression onto original time series
        reg_coeff_PDO, pvals_PDO = regress_map.regress_map(PDO,tos_data,map_type='regress')
        reg_coeff_IPO, pvals_IPO = regress_map.regress_map(IPO,tos_data,map_type='regress')
        reg_coeff_IOD, pvals_IOD = regress_map.regress_map(IOD,tos_data,map_type='regress')
        reg_coeff_IOBM, pvals_IOBM = regress_map.regress_map(IOBM,tos_data,map_type='regress')
        reg_coeff_IOBM2, pvals_IOBM2 = regress_map.regress_map(IOBM2,tos_data,map_type='regress')
        reg_coeff_NINO34, pvals_NINO34 = regress_map.regress_map(NINO34,tos_data,map_type='regress')

        # save
        new_file_name = saving_dir + '/SST_indices_'+model_name+'_' + id + '.nc'
        description = 'Various indices of SSTs including the PDO index calculated as the first principal component time series of monthly 20N-70N SST with seasonal cycle removed, Interdecadal Pacific Oscillation, Indian Ocean Basin Mode, Indian Ocean Dipole and Nino 3.4 index. '
        save = save_file(new_file_name,description)
        # add dimension variables
        save.add_dimension(lats,'lats')
        save.add_dimension(lons,'lons')
        save.add_times(times,calendar,t_units,time_name='times')
        # add variables
        save.add_variable(PDO,'PDO',('times',))
        save.add_variable(reg_coeff_PDO,'reg_coeff_PDO',('lats','lons',))
        save.add_variable(pvals_PDO,'pvals_PDO',('lats','lons',))
        save.add_variable(IOD,'IOD',('times',))
        save.add_variable(reg_coeff_IOD,'reg_coeff_IOD',('lats','lons',))
        save.add_variable(pvals_IOD,'pvals_IOD',('lats','lons',))
        save.add_variable(IPO,'IPO',('times',))
        save.add_variable(reg_coeff_IPO,'reg_coeff_IPO',('lats','lons',))
        save.add_variable(pvals_IPO,'pvals_IPO',('lats','lons',))
        save.add_variable(IOBM,'IOBM',('times',))
        save.add_variable(reg_coeff_IOBM,'reg_coeff_IOBM',('lats','lons',))
        save.add_variable(pvals_IOBM,'pvals_IOBM',('lats','lons',))
        save.add_variable(IOBM2,'IOBM2',('times',))
        save.add_variable(reg_coeff_IOBM2,'reg_coeff_IOBM2',('lats','lons',))
        save.add_variable(pvals_IOBM2,'pvals_IOBM2',('lats','lons',))
        save.add_variable(NINO34,'NINO34',('times',))
        save.add_variable(reg_coeff_NINO34,'reg_coeff_NINO34',('lats','lons',))
        save.add_variable(pvals_NINO34,'pvals_NINO34',('lats','lons',))
        save.close_file()
        print('Saved to %s' % (new_file_name))
    except: 
        message = 'Error, skipping model ' + model_name +' for PDO calculations'
        print(message)
        file = open(saving_dir + '/SST_indices_skipped_models.txt','a+')
        file.write(message + '\n')
        file.close()

