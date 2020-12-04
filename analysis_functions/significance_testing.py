from scipy import stats
import numpy as np

def _collapse_dims(array):
    """ Collapse all but the first dimension onto a single dimension such that
        the array is of the form A = A(time,space) """
    if array.ndim > 2:
        array_dims_orig = array.shape
        new_dims = array_dims_orig[:1] + (-1,)
        array = array.reshape(new_dims)
    return array


def _restore_spatial_dimensions(array,orig_shape):
    """ Restore spatial dimensions which have been flattened """
    array = array.reshape(orig_shape)
    return array


def calc_autocorr(data,t=1):
    """ Models a time varying map as following a first order autocorrelation process and
    calculates r, where a_n+1 = r * a_n + E, where E is white noise. """
    if data.ndim > 1:
        orig_shape = data.shape[1:]
        data_collapsed = _collapse_dims(data)
        ac = np.array([])
        for i in np.arange(data_collapsed.shape[1]):
            ac = np.append(ac,np.corrcoef(np.array([data_collapsed[:-t,i], data_collapsed[t:,i]]))[0,1])
        autocorr = _restore_spatial_dimensions(ac,orig_shape)
    elif data.ndim == 1:
        autocorr = np.corrcoef(np.array([data[:-t], data[t:]]))[0,1]
    return autocorr


def t_test_autocorr(all_data,data_subset,autocorr=None):
    """ Calculate a t test for data which has some autocorrelation in time
    Assumes data is of the form data = data(time,spatial dimensions)
    If autocorr is None, the autocorrelation is calculated from the data.
    n = sample size, n_eff = effective sample, df = degrees of freedom"""
    n = data_subset.shape[0]
    if autocorr is None: autocorr = calc_autocorr(all_data)
    n_eff = n * (1 - autocorr)/(1 + autocorr)
    df = n_eff - 1
    test_statistic = np.sqrt(n_eff) * (np.mean(data_subset,axis=0) - np.mean(all_data,axis=0)) / np.std(data_subset,axis=0)
    pvals = stats.t.sf(np.abs(test_statistic),df) * 2
    return pvals


def t_test_regression(all_data,regress_coeffs,type='regress',autocorr=None):
    """ Calculates a t test for a regression or correlation map. """
    n = all_data.shape[0]
    if autocorr is None: autocorr = calc_autocorr(all_data)
    n_eff = n * (1 - autocorr)/(1 + autocorr)
    df = n_eff - 1
    #print(n,n_eff)
    if type=='regress': test_statistic = np.sqrt(n_eff) * (regress_coeffs) / np.std(all_data,axis=0)
    elif type=='corr': test_statistic = (np.sqrt(n_eff) - 1) * (regress_coeffs) / np.sqrt(1 - regress_coeffs**2)
    pvals = stats.t.sf(np.abs(test_statistic),df) * 2
    return pvals

def confidence_intervals(x,y,autocorr=None,alpha=0.05):
    """ Calculate confidence intervals on a correlation between two time series, x and y
    This uses a Fisher Z transformation on the correlation coefficient which is approximately
    normally distributed. An effective sample size, n_effective is included to account for
    autocorrelation of the data. Alpha is the significance level. See here for more details
    on the Fisher transform:  https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/"""

    n_sample = x.shape[0] 
    r_value = stats.linregress(x,y)[2]

    # find the autocorrelation of both time series and the largest
    if autocorr is None: 
        x_ac = calc_autocorr(x)
        y_ac = calc_autocorr(y)
        if x_ac < y_ac: autocorr = x_ac
        else: autocorr = y_ac
    n_effective = n_sample #* (1 - autocorr)/(1 + autocorr)

    # fisher transform
    r_z = np.arctanh(r_value)
    se = 1/np.sqrt(n_effective - 3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))

    return lo, hi
    

