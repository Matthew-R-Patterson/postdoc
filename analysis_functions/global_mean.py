import numpy as np

def global_mean(field,lats):
    """ Calculate the global mean of a field. The field can be time varying or not.
    Assumes that dimensions follow time,lat,lon or lat,lon"""
    cos_weighting = np.cos(np.deg2rad(lats))
    cos_mean = np.mean(cos_weighting)
    n_dims = field.ndim
    if n_dims == 2:
        cos_weighting = cos_weighting.reshape(lats.shape[0],1)
        mean = np.mean(field * cos_weighting) / cos_mean
    elif n_dims == 3: 
        cos_weighting = cos_weighting.reshape(1,lats.shape[0],1)
        mean = np.mean(np.mean(field * cos_weighting,axis=1),axis=1) / cos_mean
    else: print('Too many or too few dimensions')
    return mean




