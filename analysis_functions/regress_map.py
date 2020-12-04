"""
Calculate the correlation between a time series and each grid-point 
of a time varying field. 
"""

import numpy as np
from scipy.stats import linregress

def _collapse_dims(array):
    """ Collapse all but the first dimension onto a single dimension such that
    the array is of the form A = A(time,space) """
    if array.ndim > 2:
        array_dims_orig = array.shape
        new_dims = array_dims_orig[:1] + (-1,)
        array = array.reshape(new_dims)
    return array

def regress_map(time_series,X,map_type='corr'):
    """ Peforms a linear regression on a time-varying field

    Parameters:
    time_series = the time series to regress (1D)
    X = N-dimensional field with the first dimension being time
    map_type = Choose whether to return a correlation map or regression map.
               'corr' / 'regress'    
    """
    # check that time series and field have the same time dimension
    if time_series.shape[0] != X.shape[0]:
        print("Field and time series size do not match")
        print("Field size:",X.shape,"Time series:",time_series.shape)
        raise ValueError

    orig_shape = X.shape
    collapsed_field = _collapse_dims(X)
    nSpace = collapsed_field.shape[1]

    print("Performing regression...")
    milestone = 20
    regress_coeff = np.zeros([nSpace])
    pvals = np.zeros([nSpace])
    for i in np.arange(nSpace):

        # print percentage complete every 10%
        if 100*float(i+1)/float(nSpace) >= milestone:
            print("%d %% complete" % (milestone))
            milestone = milestone + 20

        # do regression
        slope, intercept, r_value, p_value, std_err = linregress(time_series,collapsed_field[:,i])
        if map_type == 'corr': 
            regress_coeff[i] = r_value
        elif map_type == 'regress':
            regress_coeff[i] = slope
        else:
            print("%s is not a valid regression type" % (map_type))
            raise NameError
        pvals[i] = p_value

    # re-create original dimensions
    regress_coeff = regress_coeff.reshape(orig_shape[1:])
    pvals = pvals.reshape(orig_shape[1:])

    return regress_coeff,pvals

