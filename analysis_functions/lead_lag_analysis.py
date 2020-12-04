""" Functions to perform lead-lag analysis of two time series or a time series
and a time-varying map."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def lead_lag_ts(x,y,max_lag=10):
    """ orientated such that x leads y at minus lags and 
    y leads x at positive lags. """
    lags = np.arange(-max_lag,max_lag+1)
    lag_corr = np.array([])
    for lag in lags:
        corr = linregress(x,np.roll(y,-lag))[2]
        lag_corr = np.append(lag_corr,corr)
    return lag_corr,lags

"""
# testing

x = np.random.randn(1005)
y = np.array([1,1])

for i in np.arange(2,1002):
    y = np.append(y,0.5*np.random.randn(1) + x[i-2])# + x[i+1])

print(x.shape,y.shape)

max_lag =10
lag_corr,lags = lead_lag(x[:1000],y[:1000],max_lag)

plt.plot(lags,lag_corr)
plt.show()
"""
    
