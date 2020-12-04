"""
Perform a Maximum Covariance Analysis on two time-varying data fields. 

"""

import numpy as np
from scipy.stats import linregress

class MCA:
    """ Performs MCA analysis

    Run this code as follows in which X and Y are data arrays with time on the
    first axis. 

    Sets up an MCA object
    $ mca = MCA(X,Y)

    Runs the MCA code
    $ mca.do_MCA()

    Then various methods can be called to find the fraction of covariance
    explained by each successive mode, the time series associated with each
    mode and the patterns associated with each mode. Here n is the number of
    modes to retain. 
    $ frac_cov = mca.frac_cov()
    $ u_ts,v_ts = mca.pattern_time_series(n=5)
    $ u,v = mca.return_patterns(n=5)"""
    
    def __init__(self,X,Y,weightsX=None,weightsY=None):
        """Parameters:
                X,Y = data arrays with 1 dimension of time followed by spatial
                      dimensions e.g. time x latitude x longitude. The time 
                      dimension must be the same for X and Y

                weightsX,weightsY = Arrays of weights on data e.g. with latitude
                                    Should be of the same dimensions as the 
                                    corresponding X/Y array but can be broadcast.
                                    For example, if X = X(time,latitude,longitude),
                                    weightsX could have shape 1 x nLatitudes x 1 
                                    or shape nTimes x nLatitudes x nLongitudes                                

            Calculates arrays of optimal spatial patterns from X and Y
            which maximise the spatial covariance."""

        # check that X and Y have the same number of temporal values
        if X.shape[0] != Y.shape[0]:
            print('Number of temporal values is not the same')
            print('X dimensions: ',X.shape,'Y dimensions: ', Y.shape)
            raise ValueError

        # check that the dimension of X and Y is greater than 1
        if (X.ndim <= 1) | (Y.ndim <=1):
            print('Require at least 1 dimension of space and 1 of time for X and Y')
            print('X dimensions: ',X.shape,'Y dimensions: ', Y.shape) 
            raise ValueError

        # check that weights have the same number of dimensions as X and Y
        if weightsX is not None:
            if (weightsX.ndim != X.ndim): 
                print("X Weights dimensions not equal to X dimensions")
                raise ValueError
        if weightsY is not None:
            if (weightsY.ndim != Y.ndim):
                print("Y Weights dimensions not equal to Y dimensions")
                raise ValueError

        self._X = X
        self._Y = Y
        self.nTimes = X.shape[0]
        self.weightsX = weightsX
        self.weightsY = weightsY

    def _collapse_dims(array):
        """ Collapse all but the first dimension onto a single dimension such that 
            the array is of the form A = A(time,space) """
        if array.ndim > 2:
            array_dims_orig = array.shape
            new_dims = array_dims_orig[:1] + (-1,)
            array = array.reshape(new_dims)
        return array

    def _standardise(array):
        """ Remove the time-mean and divide by the standard deviation"""
        array = (array - np.mean(array,axis=0))/np.std(array,axis=0)
        return array

    def _detrend(self,array):
        """ Remove the linear trend from each grid-point """
        nSpace = array.shape[1]
        t = np.arange(self.nTimes)
        for space_coord in np.arange(nSpace):
            trend = linregress(t,array[:,space_coord])[0]
            array[:,space_coord] = array[:,space_coord] - trend * t
        return array

    def _prepare_weights(orig_array,weights):
        """ Broadcast weights and flatten spatial dimensions such that
        the weights have the same dimension as array """
        weights_broadcast = np.ones_like(orig_array) * weights
        weights_collapsed = MCA._collapse_dims(weights_broadcast)
        return weights_collapsed

    def _weighting(self,array,weights=None):
        """ Add a weighting e.g. to account for the reduction in area
        with latitude """ 
        if weights is not None:
            array = weights * array
        else: 
            print("No weights defined, array unchanged")
        return array

    def _prepare_array(self,array,weights=None):
        """ Collapse dimensions, standardise and detrend """
        arr_2d = MCA._collapse_dims(array)
        #arr_standard = MCA._standardise(arr_2d)
        #arr_stand_detrend = MCA._detrend(self,arr_standard)
        arr_detrend = MCA._detrend(self,arr_2d)
        arr_stand_detrend = MCA._standardise(arr_detrend)
        if weights is not None:
            weights_prepped = MCA._prepare_weights(array,weights)
            arr_stand_detrend = MCA._weighting(self,arr_stand_detrend,weights=weights_prepped)
        return arr_stand_detrend

    def do_MCA(self):
        """ calculate maximum covariance patterns """
        print("Calculating MCA")
        X = MCA._prepare_array(self,self._X,weights=self.weightsX)
        Y = MCA._prepare_array(self,self._Y,weights=self.weightsY)
        covariance = (1/self.nTimes) * np.dot(X.T,Y)
        U, s, V = np.linalg.svd(covariance)

        self.U = U
        self.s = s
        self.V = V
    
    def frac_cov(self,n=20):
        """ find the fraction of total covariance associated with each set of patterns
            and return the first n """
        fraction_covariance = self.s.flatten()**2 / np.sum(self.s**2)
        return fraction_covariance[:n]

    def _restore_spatial_dimensions(array,orig_shape):
        """ Restore spatial dimensions which have been flattened """
        array = array.reshape(orig_shape)
        return array

    def pattern_time_series(self,n=1):
        """ Return the first n time series associated with the first n patterns """
        u = self.U[:,:n].reshape(self.U.shape[0],n)
        X_2d = MCA._collapse_dims(self._X)
        u_time = np.dot(u.T,X_2d.T)
        v = self.V.T[:,:n].reshape(self.V.shape[0],n)
        Y_2d = MCA._collapse_dims(self._Y)
        v_time = np.dot(v.T,Y_2d.T)
        return u_time, v_time

    def return_patterns(self,n=1):
        """ Return the first n patterns """
        u = self.U[:,:n].reshape(self.U.shape[0],n)
        v = self.V.T[:,:n].reshape(self.V.shape[0],n)
        u = u.T
        v = v.T
        new_dims_X = np.append(np.array([n]),self._X.shape[1:])
        new_dims_Y = np.append(np.array([n]),self._Y.shape[1:])
        u = MCA._restore_spatial_dimensions(u,new_dims_X)
        v = MCA._restore_spatial_dimensions(v,new_dims_Y)
        return u,v
