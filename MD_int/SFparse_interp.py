
import numpy as np

import matplotlib.pyplot as plt


from scipy.interpolate import RegularGridInterpolator # You may have some better interpolation methods



L = 120

n_freqs = 4097



# Momentum and frequency ranges (with some built in buffers)




K1 = np.arange(-4*L//3, 4*L//3)

K2 = np.arange(-4*L//3, 4*L//3)



F = np.arange(0, n_freqs)



# Load the data files

dsf_data = np.load('/Users/jfmv/Documents/Proyectos/Delafossites/Struc_dat/dsf_TLHAF_L=120_tf=4096_T=1.0.npy')





def dsf_func(k1, k2, w):

    return dsf_data[k1%L, k2%L, w]



# This constructs a rearranged array



dsf_func_data = np.array([[[dsf_func(k1, k2, w) for k1 in K1] for k2 in K2] for w in F])







# One can now construct an interpolated function over this rearranged data - this is not quite the final form



dsf_interp = RegularGridInterpolator((F, K1, K2), dsf_func_data, method='linear')



# This function converts the momentum and frequency we want to call the DSF at into the appropriate parameters with which

# to call the interpolated function (and then calls it):



def dsf(qx, qy, f):

    k1 = L*qx/(2*pi)

    k2 = L*(qx/2 + sqrt(3)*qy/2)/(2*pi)

    w = n_freqs*f/(2*pi)

    return dsf_interp((w, k2, k1)) # this has to be called in the reverse order for some reason.




# This function should call the actual values measured from the simulation if one evaluates at one of the sampled frequencies

# and momenta, that is:



# [qx, qy, f] = [2*pi*n_1/L, 2*(2*pi*n_2/L - pi*n_1/L)/sqrt(3), 2*pi*n_3/n_freqs] for integers (n_1, n_2, n_3).
