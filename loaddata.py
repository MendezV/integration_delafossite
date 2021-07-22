################################
"""
This code reads the structure factor as formatted by adam,
then samples points in the brillouin zone using a triangular grid in the conventional cell

After this initial step, the program interpolates in momentum with a grid size determined
by the first argument given to the program

The second argument given to the program is the temperature at which the structure factor
was calculated

ARGS:
L after interpolation
T for the structure factor

OUT:
.npy file with the interpolated data
"""
################################


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import time
import sys