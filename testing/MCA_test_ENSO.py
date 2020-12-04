"""
Test MCA function using sea level pressure and air temperature data
"""

# standard python modules
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import cartopy.crs as ccrs
import iris
import os
import sys
from netCDF4 import Dataset
import time

# my modules
cwd = os.getcwd()
repo_dir = '/'
for directory in cwd.split('/')[1:]:
    repo_dir = os.path.join(repo_dir, directory)
    if directory == 'postdoc':
        break

analysis_functions_dir = os.path.join(repo_dir, 'analysis_functions')
figure_dir = os.path.join(repo_dir,'testing/figures_from_testing')
sys.path.append(analysis_functions_dir)
from MCA import MCA




# read in sea level pressure and surface air temperature data
SATfileName = os.path.join(repo_dir,'testing/data_for_tests/air.mon.mean.nc')
SLPfileName = os.path.join(repo_dir,'testing/data_for_tests/slp.mon.mean.nc')

nc = Dataset(SATfileName,'r')
SAT = nc.variables['air'][:] 
lats = nc.variables['lat'][:] # SLP and SAT have same lats and lons
lons = nc.variables['lon'][:]

nc = Dataset(SLPfileName,'r')
SLP = nc.variables['slp'][:]

# only consider January for each year
SLP = SLP[::12,:,:]
SAT = SAT[::12,:,:]

# take only the Pacific 
latMin, latMax = -40,40 #-30,30
lonMin, lonMax = 120,300
latMask = (lats>=latMin)&(lats<=latMax)
lonMask = (lons>=lonMin)&(lons<=lonMax)
SLP = SLP[:,latMask,:]
SLP = SLP[:,:,lonMask]
SAT = SAT[:,latMask,:]
SAT = SAT[:,:,lonMask]

# Do MCA analysis
nLatsMask = lats[latMask].shape[0]
weights = np.sqrt(np.cos(np.deg2rad(lats[latMask]))).reshape(1,nLatsMask,1)
mca = MCA(SAT,SLP,weightsX=weights,weightsY=weights)
mca.do_MCA()
frac_cov = mca.frac_cov()
u_ts,v_ts = mca.pattern_time_series(n=5)
u,v = mca.return_patterns(n=5)

# Plot results
plt.figure(figsize=(15,15))
gs = gridspec.GridSpec(4,2,height_ratios=[7,0.5,5,4])
#gs = gridspec.GridSpec(3,2,height_ratios=[10,0.5,5])
extent = [120,320,-60,60]
aspect = 2
mode = 0

# plot first patterns
ax = plt.subplot(gs[0,0],projection=ccrs.PlateCarree(central_longitude=180.))
plt.title('SAT pattern 1')
cs = plt.contourf(lons[lonMask],lats[latMask],u[mode,:,:],cmap='RdBu_r',extend='both',transform=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent(extent)
ax.set_aspect(aspect)
ax = plt.subplot(gs[1,0])
cb = plt.colorbar(cs,cax=ax,orientation='horizontal')

ax = plt.subplot(gs[0,1],projection=ccrs.PlateCarree(central_longitude=180.))
plt.title('SLP pattern 1')
cs = plt.contourf(lons[lonMask],lats[latMask],v[mode,:,:],cmap='RdBu_r',extend='both',transform=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent(extent)
ax.set_aspect(aspect)
ax = plt.subplot(gs[1,1])
cb = plt.colorbar(cs,cax=ax,orientation='horizontal')

# plot time series projections onto patterns
ax = plt.subplot(gs[2,:])
plt.title('Time series of first modes')
t = np.arange(1948,2020+1)
plt.xlabel('Year')
ax.plot(t,u_ts[mode,:],color='b')
ax2 = ax.twinx()
ax2.plot(t,v_ts[mode,:],color='r')

# plot fraction of covariance explained
ax = plt.subplot(gs[3,:])
plt.title('Covariance fraction')
plt.scatter(np.arange(1,10+1),frac_cov[:10])
plt.xlabel('Mode')

plt.subplots_adjust(hspace=0.3)
plt.savefig(figure_dir+'/MCA_test1_ENSO_SAT_SLP.png',bbox_inches='tight')
#plt.show()
