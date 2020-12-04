from scipy import interpolate
import numpy as np

def interpolate_grid(old_grid_data,lons_old,lats_old,lons_new,lats_new):
    X, Y = np.meshgrid(lons_old, lats_old)
    XI, YI = np.meshgrid(lons_new,lats_new)
    new_grid =  interpolate.griddata((X.flatten(),Y.flatten()),old_grid_data.flatten() , (XI,YI),method='cubic')
    return new_grid

