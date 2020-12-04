""" Calculate the power spectrum of a given time series """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab

def power_spectrum(data,sampling_rate=1):
    ps = np.abs(np.fft.fft(data))**2
    freqs = np.fft.fftfreq(data.size, 1/sampling_rate)
    idx = np.argsort(freqs)
    return freqs[idx], ps[idx]

def plot_ps(data,sampling_rate=1,xlabel='',ylabel='',titleFontsize=20):
    freqs, ps = power_spectrum(data,sampling_rate=sampling_rate)
    plt.plot(freqs,ps,color='k')
    plt.xlim(0,np.max(freqs))
    plt.xlabel(xlabel=xlabel,fontsize=0.8*titleFontsize)
    plt.ylabel(ylabel=ylabel,fontsize=0.8*titleFontsize)

"""
t = np.arange(30*12)
f = 1/12
data = np.sin(2*np.pi*f*t) + np.random.randn(len(t))*0.2  #np.random.rand(301) - 0.5

plt.figure()
ax = plt.subplot(2,1,1)
plt.plot(t,data)
ax = plt.subplot(2,1,2)
plot_ps(data)
plt.show()
"""
