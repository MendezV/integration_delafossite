import numpy as np
import Lattice
import StructureFactor
import Dispersion
import matplotlib.pyplot as plt
import time
from scipy import integrate
from numpy import linalg as la

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



Npoints=100
save=True
l=Lattice.TriangLattice(Npoints, save )
Vol_rec=l.Vol_BZ()

[KX,KY]=l.read_lattice()

Vertices_list, Gamma, K, Kp, M, Mp=l.FBZ_points(l.b[0,:],l.b[1,:])
T=1.0
SS=StructureFactor.StructureFac(T)
params=SS.params_fit(KX, KY)
SF_stat=SS.Static_SF(KX, KY)
SF_dyna=SS.Dynamical_SF_fit( 0.1, params,SF_stat)

##Plots of the structure factor
# plt.scatter(KX,KY, c=SF_stat)
# plt.show()

# plt.scatter(KX,KY, c=SF_dyna)
# plt.show()


#electronic parameters
J=2*5.17 #in mev
tp1=568/J #in units of Js\
tp2=0.065*tp1

U=4000/J
g=100/J
Kcou=g*g/U
fill=0.311

ed=Dispersion.Dispersion_single_band([tp1,tp2],fill,0)
[xFS_dense,yFS_dense]=ed.FS_contour(400)
print("Filling is ...",ed.filling)
print("chempot is ...",ed.mu/tp1)


ed=Dispersion.Dispersion_single_band([tp1,tp2],fill,1)
[xFS_dense2,yFS_dense2]=ed.FS_contour(400)
print("Filling is ...",ed.filling)
print("chempot is ...",ed.mu/tp1)
VV=np.array(Vertices_list)
plt.scatter(VV[:,0],VV[:,1])
plt.scatter(xFS_dense,yFS_dense)
plt.scatter(xFS_dense2,yFS_dense2,s=1)

plt.show()
