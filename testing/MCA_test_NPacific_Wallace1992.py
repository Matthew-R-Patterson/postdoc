"""
Testing the MCA code by reproducing figures 6 and 7 from Wallace et al 1992
"Singular Value Decomposition of Wintertime Sea Surface Temperature and 500-mb
Height Anomalies" - Journal of Climate
https://doi.org/10.1175/1520-0442(1992)005<0561:SVDOWS>2.0.CO;2
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

# read in 500hPa geopotential height and surface air temperature data
SATfileName = os.path.join(repo_dir,'testing/data_for_tests/air.mon.mean.nc')
Z500fileName = os.path.join(repo_dir,'testing/data_for_tests/hgt.mon.mean.nc')

nc = Dataset(SATfileName,'r')
SAT = nc.variables['air'][:]
lats = nc.variables['lat'][:] # SLP and SAT have same lats and lons
lons = nc.variables['lon'][:]

nc = Dataset(Z500fileName,'r')
levs = nc.variables['level'][:]
lev_idx = np.argmin(np.abs(levs - 500))
Z500 = nc.variables['hgt'][:,lev_idx,:,:]

# only consider January for each year
Z500 = Z500[::12,:,:]
SAT = SAT[::12,:,:]

# take only the North Pacific, but perform regression on all NH
latMin, latMax = 10,80
lonMin, lonMax = 150,240
latMask = (lats>=latMin)&(lats<=latMax)
lonMask = (lons>=lonMin)&(lons<=lonMax)
Z500_NH = np.copy(Z500[:,lats>0,:])
Z500 = Z500[:,latMask,:]
Z500 = Z500[:,:,lonMask]
SAT_NH = np.copy(SAT[:,lats>0,:])
SAT = SAT[:,latMask,:]
SAT = SAT[:,:,lonMask]

# Do MCA analysis
nLatsMask = lats[latMask].shape[0]
weights = np.sqrt(np.cos(np.deg2rad(lats[latMask]))).reshape(1,nLatsMask,1)
mca = MCA(SAT,Z500,weightsX=weights,weightsY=weights)
mca.do_MCA()
frac_cov = mca.frac_cov()
u_ts,v_ts = mca.pattern_time_series(n=5)
u,v = mca.return_patterns(n=5)


# Calculate correlation maps for modes 0 and 1
corr_coeff_Z0,pvals_Z0 = regress_map.regress_map(u_ts[0,:],Z500_NH)
corr_coeff_SAT0,pvals_SAT0 = regress_map.regress_map(v_ts[0,:],SAT_NH)
corr_coeff_Z1,pvals_Z1 = regress_map.regress_map(u_ts[1,:],Z500_NH)
corr_coeff_SAT1,pvals_SAT1 = regress_map.regress_map(v_ts[1,:],SAT_NH)

# Plot results
plt.figure(figsize=(15,12))
gs = gridspec.GridSpec(2,3,width_ratios=[10,10,0.5])
extent = [100,270,10,80]
aspect = 2
a = np.arange(0.2,0.81,0.2)
clevs = np.append(-a[::-1],a)
titleFontsize = 30

ax = plt.subplot(gs[0,0],projection=ccrs.PlateCarree(central_longitude=180.))
plt.title('SAT pattern 1',fontsize=titleFontsize)
cs = plt.contourf(lons,lats[lats>0],corr_coeff_SAT0,clevs,cmap='RdBu_r',extend='both',transform=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent(extent)
ax.set_aspect(aspect)
ax = plt.subplot(gs[0,1],projection=ccrs.PlateCarree(central_longitude=180.))
plt.title('SAT pattern 2',fontsize=titleFontsize)
cs = plt.contourf(lons,lats[lats>0],corr_coeff_SAT1,clevs,cmap='RdBu_r',extend='both',transform=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent(extent)
ax.set_aspect(aspect)
ax = plt.subplot(gs[0,2])
cb = plt.colorbar(cs,cax=ax)
cb.ax.tick_params(labelsize=0.8*titleFontsize)

ax = plt.subplot(gs[1,0],projection=ccrs.Orthographic(central_longitude=270.0, central_latitude=90.0))
plt.title('Z500 pattern 1',fontsize=titleFontsize)
cs = plt.contourf(lons,lats[lats>0],corr_coeff_Z0,clevs,cmap='RdBu_r',extend='both',transform=ccrs.PlateCarree())
ax.coastlines()
ax = plt.subplot(gs[1,1],projection=ccrs.Orthographic(central_longitude=270.0, central_latitude=90.0))
plt.title('Z500 pattern 2',fontsize=titleFontsize)
cs = plt.contourf(lons,lats[lats>0],corr_coeff_Z1,clevs,cmap='RdBu_r',extend='both',transform=ccrs.PlateCarree())
ax.coastlines()
ax = plt.subplot(gs[1,2])
cb = plt.colorbar(cs,cax=ax)
cb.ax.tick_params(labelsize=0.8*titleFontsize)

#plt.subplots_adjust(hspace=0.2)
plt.savefig(figure_dir+'/MCA_test2_Wallace1992_NPacific_SAT_Z500_corr.png',bbox_inches='tight')
plt.show()

