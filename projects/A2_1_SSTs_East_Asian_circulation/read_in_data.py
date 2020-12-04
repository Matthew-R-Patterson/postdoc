""" Set of functions for reading in data """

import numpy as np
from netCDF4 import Dataset, num2date
import os
import sys


def files_in_directory(directory,concat_directory=False,include_files_with=None,exclude_files_with=None):
    """Creates a list of files in a given directory
    if concat_directory=True, includes the full path """
    from os import listdir
    from os.path import isfile, join
    if concat_directory == True:
        files = [join(directory,f) for f in listdir(directory) if isfile(join(directory, f))]
    else:
        files = [f for f in listdir(directory) if isfile(join(directory, f))]
    files.sort() # sort into alphabetical order
    if exclude_files_with is not None:
        files = [f for f in files if exclude_files_with not in f]
    if include_files_with is not None:
        files = [f for f in files if include_files_with in f]
    return files


def read_spatial_dimensions(file_name,lat_name=None,lon_name=None,lev_name=None):
    """ Returns the latitude, longitude and pressure (if available) dimensions """
    nc = Dataset(file_name,'r')
    lats=None
    lons=None
    levs=None
    common_latitude_names = ['lat','lats','latitude']
    common_longitude_names = ['lon','lons','longitude']
    common_pressure_names = ['level','levels','levs','p','plevs','plev']
    if lat_name is not None: lats = nc.variables[lat_name][:]
    else:
        for name in common_latitude_names:
            try:
                lats = nc.variables[name][:]
                break
            except: pass
    if lon_name is not None: lons = nc.variables[lon_name][:]
    else:
        for name in common_longitude_names:
            try:
                lons = nc.variables[name][:]
                break
            except: pass
    if lev_name is not None: levs = nc.variables[lev_name][:]
    else:
        for name in common_pressure_names:
            try:
                levs = nc.variables[name][:]
                break
            except: pass
    return levs,lats,lons


def read_time_dimension(file_name):
    """ Reads in data on the time dimension and returns it along with units
    and calendar """
    nc = Dataset(file_name)
    common_time_names = ['t','time','times']
    times = None
    calendar = None
    units = None
    for name in common_time_names:
        try:
            times = nc.variables[name][:]
            break
        except: pass
    try: calendar = nc.variables[name].calendar
    except: print('Warning: calendar not defined')
    try: units = nc.variables[name].units
    except: print('Warning: time units not defined')
    return times,calendar,units


def read_in_variable(list_of_files,variable_name,chosen_level=None,
        longitude_bounds=[0,360],latitude_bounds=[-90,90]):

    """ Reads in data from a given variable from a set of files and concatenates
    them in the time dimension

    This may be subsetted by selecting a given level (in units of hPa), or setting
    latitude or longitude bounds. For example, selecting 500hPa wind over the
    northern hemisphere: chosen_level=500,latitude_bounds=[0,90]

    returns: list_of_files, variable_data  """

    # read in file / subset
    nc = Dataset(list_of_files[0],'r')
    levs,lats,lons = read_spatial_dimensions(list_of_files[0])

    # create masks based on latitude / longitude bounds
    latMask = (lats >= latitude_bounds[0]) & (lats <= latitude_bounds[1])
    if longitude_bounds[0] < longitude_bounds[1]:
        lonMask = (lons >= longitude_bounds[0]) & (lons <= longitude_bounds[1])
    else:
        lonMask = (lons >= longitude_bounds[0]) | (lons <= longitude_bounds[1])
    if chosen_level is not None:
        lev_idx = np.argmin(np.abs(levs - chosen_level))
        print('Chosen level: %f' % (levs[lev_idx]))

    # read in data
    print('Reading in data from %s' % (list_of_files[0]))
    dims = nc.variables[variable_name].ndim
    times,calendar,t_units = read_time_dimension(list_of_files[0])
    if dims == 3:
        variable_data = nc.variables[variable_name][:,latMask,:][:,:,lonMask]
    elif dims == 4:
        if chosen_level is not None: variable_data = nc.variables[variable_name][:,lev_idx,:,:][:,latMask,:][:,:,lonMask]
        else: nc.variables[variable_name][:,:,latMask,:][:,:,:,lonMask]
    else:
        print('Too many or too few dimensions, dimensions = %d' % (dims))
        raise ValueError

    # read in other files and concatenate along the time dimension,
    # files assumed to be in time order
    for f in list_of_files[1:]:
        nc = Dataset(f,'r')
        print('Reading in data from %s' % (f))
        if dims == 3:
            new_data = nc.variables[variable_name][:,latMask,:][:,:,lonMask]
        elif dims == 4:
            if chosen_level is not None: new_data = nc.variables[variable_name][:,lev_idx,:,:][:,latMask,:][:,:,lonMask]
            else: new_data = nc.variables[variable_name][:,:,latMask,:][:,:,:,lonMask]
        else:
            print('Too many or too few dimensions, dimensions = %d' % (dims))
            raise ValueError
        new_times,calendar,t_units = read_time_dimension(f)
        variable_data = np.append(variable_data,new_data,axis=0)
        times = np.append(times,new_times)

    # return variable, dimensions
    return variable_data,lats,lons,levs,times,calendar,t_units


def calculate_annual_mean(variable,times,calendar,units,season=None,decadal_mean=False):
    """ Calculate the annual mean for a given variable. Also features an option to
    calculate the decadal mean. This takes the mean of years 2000-2009 for example.
    Can be taken over only one season if required """
    dates = num2date(times,calendar=calendar,units=units)
    months = np.array([])
    years = np.array([])
    for day in dates:
        months = np.append(months,day.timetuple()[1])
        years = np.append(years,day.timetuple()[0])
    #print(variable.shape,times.shape)
    new_dims = variable.shape[1:]
    new_dims = (1,) + new_dims
    
    if decadal_mean == True: years = years - years%10
    
    for i,yr in enumerate(np.unique(years)):
        if season == 'DJF': mask = ((years==(yr-1))&(months==12)) | (years==yr) & ((months==1)|(months==2))
        elif season == 'MAM': mask =(years==yr) & ((months==3)|(months==4)|(months==5))
        elif season == 'JJA': mask = (years==yr) & ((months==6)|(months==7)|(months==8))
        elif season == 'SON': mask = (years==yr) & ((months==9)|(months==10)|(months==11))
        elif season == None: mask = (years==yr)
        elif season == 'ANN':  mask = (years==yr)
        else:
            print('Season is not valid')
            raise NameError
        if i == 0:
            annual_mean = np.nanmean(variable[mask],axis=0).reshape(new_dims)
        else:
            new_annual_mean = np.nanmean(variable[mask],axis=0).reshape(new_dims)
            annual_mean = np.append(annual_mean,new_annual_mean,axis=0)
    return annual_mean, np.unique(years)


class save_file:
#""" Saves results of the MCA and EOF analysis for future plotting Specifically the following are saved: model name and list of files names of variable 1 and variable 2 dimensions including latitudes, longitudes and times MCA fractional covariance, u,v, u_ts, v_ts for the first 3 MCA patterns regression coefficients and pvals between time series and original fields first principal components and regression onto variables 1 and 2 """
    def __init__(self,file_name,description):
        """ Create save file object to add variables to """
        nc_write = Dataset(file_name,'w',format='NETCDF3_CLASSIC')
        self.nc = nc_write

        # create metadata
        nc_write.description = description

    def add_dimension(self,dimension_values,dimension_name,dtype=np.float64):
        """ Create dimensions """
        self.nc.createDimension(dimension_name, dimension_values.shape[0])
        var = self.nc.createVariable(dimension_name,dtype,(dimension_name,))
        self.nc.variables[dimension_name][:] = dimension_values

    def add_times(self,time_values,calendar,units,time_name='time',dtype=np.float64):
        """ Create time dimension """
        self.nc.createDimension(time_name, time_values.shape[0])
        time_variable = self.nc.createVariable(time_name,dtype,(time_name,))
        time_variable.units = units
        time_variable.calendar = calendar
        self.nc.variables[time_name][:] = time_values

    def add_variable(self,variable_data,variable_name,dimensions_tuple,dtype=np.float64):
        """ Add variable to the file """
        new_variable =  self.nc.createVariable(variable_name,dtype,dimensions_tuple)
        self.nc.variables[variable_name][:] = variable_data

    def close_file(self):
        """ close the file :) """
        self.nc.close()


def running_mean(x,N):
    """ Calculates the running mean along the first axis """
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0),axis=0)
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def prep_SST_index(index_name,nc,times_SST,calendar_SST,t_units_SST,season):
    """Read in particular SST index and calculate annual mean"""
    index_ts = nc.variables[index_name][:]
    index_annual_mean, years = calculate_annual_mean(index_ts,times_SST,calendar_SST,t_units_SST,season=season)
    return index_annual_mean, years


