"""
Test the MCA code by reproducing figure 1a) from O'Reilly et al (2018) J.Clim.
"The Impact of Tropical Precipitation on Summertime Euro-Atlantic Circulation via a
Circumglobal Wave Train"
"""

# standard python modules
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import cartopy.crs as ccrs
import os
import sys
from netCDF4 import Dataset
import time
from scipy import stats

# my modules
cwd = os.getcwd()
repo_dir = '/'
for directory in cwd.split('/')[1:]:
    repo_dir = os.path.join(repo_dir, directory)
    if directory == 'postdoc':
        break

analysis_functions_dir = os.path.join(repo_dir,'analysis_functions')
figure_dir = os.path.join(repo_dir,'testing/figures_from_testing')
sys.path.append(analysis_functions_dir)
from MCA import MCA
import regress_map

# read in 500hPa geopotential height and precip data
PrecipfileName = os.path.join(repo_dir,'testing/data_for_tests/precip_1979_2019.mon.mean.nc')
Z500fileName = os.path.join(repo_dir,'testing/data_for_tests/hgt_1979_2019.mon.mean.nc')

nc = Dataset(PrecipfileName,'r')
precip = nc.variables['precip'][:]
lats_precip = nc.variables['lat'][:] 
lons_precip = nc.variables['lon'][:]

nc = Dataset(Z500fileName,'r')
lats_Z500 = nc.variables['lat'][:]
lons_Z500 = nc.variables['lon'][:]
levs_Z500 = nc.variables['level'][:]
lev_idx = np.argmin(np.abs(levs_Z500 - 500))
Z500 = nc.variables['hgt'][:,lev_idx,:,:]

# only consider JJA for each year
Z500 = np.roll(Z500,shift=1,axis=0) # shift December round to be first element 
precip = np.roll(precip,shift=1,axis=0)
Z500 = np.mean(Z500.reshape(41,4,3,lats_Z500.shape[0],lons_Z500.shape[0])[:,2,:,:,:],axis=1)
precip = np.mean(precip.reshape(41,4,3,lats_precip.shape[0],lons_precip.shape[0])[:,2,:,:,:],axis=1)
Z500 = Z500[:38,:,:]
precip = precip[:38,:,:]


# take precip in tropics and Z500 in Euro-Atlantic
lat_min_tropics, lat_max_tropics = -15,30
lat_min_atlantic, lat_max_atlantic = 30,70
lon_min_atlantic, lon_max_atlantic = 270,30
lat_mask_tropics = (lats_precip>=lat_min_tropics)&(lats_precip<=lat_max_tropics)
lat_mask_atlantic = (lats_Z500>=lat_min_atlantic)&(lats_Z500<=lat_max_atlantic)
lon_mask_atlantic = (lons_Z500>=lon_min_atlantic)|(lons_Z500<=lon_max_atlantic)
Z500_NH = np.copy(Z500)
Z500 = Z500[:,lat_mask_atlantic,:]
Z500 = Z500[:,:,lon_mask_atlantic]
precip_NH = np.copy(precip[:,lats_precip>=-15,:])
precip = precip[:,lat_mask_tropics,:]

# take anomalies and detrend 500_NH and precip_NH
Z500_NH = (Z500_NH - np.mean(Z500_NH,axis=0))
precip_NH = (precip_NH - np.mean(precip_NH,axis=0))
t = np.arange(Z500_NH.shape[0])
for i,lat in enumerate(np.arange(lats_Z500.shape[0])):
    if i%20==0: print(i)
    for j,lon in enumerate(np.arange(lons_Z500.shape[0])):
        trend = stats.linregress(t,Z500_NH[:,i,j])[0]
        Z500_NH[:,i,j] = Z500_NH[:,i,j] - trend * t

# Do MCA analysis
nLatsMask = lats_Z500[lat_mask_atlantic].shape[0]
weights_Z500 = np.sqrt(np.cos(np.deg2rad(lats_Z500[lat_mask_atlantic]))).reshape(1,nLatsMask,1)
nLatsMask = lats_precip[lat_mask_tropics].shape[0]
weights_precip = np.sqrt(np.cos(np.deg2rad(lats_precip[lat_mask_tropics]))).reshape(1,nLatsMask,1)
mca = MCA(precip,Z500,weightsX=weights_precip,weightsY=weights_Z500)
mca.do_MCA()
frac_cov = mca.frac_cov()
u_ts,v_ts = mca.pattern_time_series(n=5)
u,v = mca.return_patterns(n=5)

# calculate EOFs as a check
from eofs.standard import Eof
precip_stand = np.copy(precip)
precip_stand = (precip_stand - np.mean(precip_stand,axis=0))/np.std(precip_stand,axis=0)
solver = Eof(precip_stand)#, weights=wgts)
eof1 = solver.eofsAsCovariance(neofs=1)
pc1 = solver.pcs(npcs=1, pcscaling=1)
"""
plt.figure()
gs = gridspec.GridSpec(3,1,height_ratios=[10,0.5,5])
ax = plt.subplot(gs[0],projection=ccrs.PlateCarree(central_longitude=180.))
plt.title('Tropical Precip EOF 1')
cs = plt.contourf(lons_precip,lats_precip[lat_mask_tropics],eof1[0,:,:],cmap='RdBu_r',transform=ccrs.PlateCarree())
#cs = plt.contourf(lons_Z500[lon_mask_atlantic],lats_Z500[lat_mask_atlantic],eof1[0,:,:],cmap='RdBu_r',transform=ccrs.PlateCarree())
ax.coastlines()
ax = plt.subplot(gs[1])
plt.colorbar(cs,cax=ax,orientation='horizontal')
ax = plt.subplot(gs[2])
plt.title('PC1 time series')
t = np.arange(1979,2019+1)
plt.plot(t,pc1)
"""

# Calculate correlation maps for a given mode
mode = 0
u = (u_ts[mode,:] - np.mean(u_ts[mode,:]))/np.std(u_ts[mode,:])
corr_coeff_Z0,pvals_Z0 = regress_map.regress_map(u,Z500_NH,map_type='regress')
#corr_coeff_Z0,pvals_Z0 = regress_map.regress_map(pc1.flatten(),Z500_NH,map_type='regress')
corr_coeff_Precip0,pvals_Precip0 = regress_map.regress_map(u,precip_NH,map_type='regress')
#corr_coeff_Precip0,pvals_Precip0 = regress_map.regress_map(pc1.flatten(),precip_NH,map_type='regress')


# Plot results
plt.figure(figsize=(20,15))
gs = gridspec.GridSpec(3,1,height_ratios=[5,0.5,5])
extent = [100,270,10,80]
aspect = 2
a = np.arange(0.2,0.81,0.2)
clevs = np.append(-a[::-1],a)
titleFontsize = 30

print('Fraction of covariance:',frac_cov)

ax = plt.subplot(gs[0,0],projection=ccrs.PlateCarree(central_longitude=270.))

plt.title('Regression MCA mode 1 of tropical precip with precip / Z500',fontsize=titleFontsize)
cs = plt.contourf(lons_precip,lats_precip[lats_precip>=-15],corr_coeff_Precip0,2*clevs,cmap='RdBu_r',extend='both',transform=ccrs.PlateCarree())
cs2 = plt.contour(lons_Z500,lats_Z500,corr_coeff_Z0,25*clevs,colors='k',transform=ccrs.PlateCarree())
plt.clabel(cs2, fmt='%.f',inline=True, fontsize=15)
plt.ylim(-15,90)
"""
plt.title('1st modes of tropical precip and Atlantic Z500')
cs = plt.contourf(lons_precip,lats_precip[lat_mask_tropics],u[mode,:,:],cmap='RdBu_r',extend='both',transform=ccrs.PlateCarree())
plt.contour(lons_Z500[lon_mask_atlantic],lats_Z500[lat_mask_atlantic],v[mode,:,:],colors='k',transform=ccrs.PlateCarree())
cs = plt.contourf(lons_Z500[lon_mask_atlantic],lats_Z500[lat_mask_atlantic],v[mode,:,:],transform=ccrs.PlateCarree())
"""
ax.coastlines()

ax = plt.subplot(gs[1])
cb = plt.colorbar(cs,cax=ax,orientation='horizontal')
cb.ax.tick_params(labelsize=0.8*titleFontsize)

ax = plt.subplot(gs[2])
t = np.arange(1979,2016+1)
slope, intercept, r_value, p_value, std_err = stats.linregress(u_ts[mode,:],v_ts[mode,:])
print('r_value= %f, p_value= %f' % (r_value,p_value))
plt.title('MCA patterns, mode 1: Precip = blue, Z500 = red, R = %.2f' % (r_value),fontsize=titleFontsize)
ax.plot(t,u_ts[mode,:],color='b')
ax2 = ax.twinx()
ax2.plot(t,v_ts[mode,:],color='r')
#ax2.plot(t,pc1.flatten(),color='r')

plt.subplots_adjust(hspace=0.4)
plt.savefig(figure_dir+'/MCA_test3_OReilly2018_Precip_Z500.png',bbox_inches='tight')

#plt.show()


