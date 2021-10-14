import numpy as np
import Lattice
import StructureFactor
import Dispersion
import matplotlib.pyplot as plt
import time
from scipy import integrate
from numpy import linalg as la

Npoints=100
save=True
l=Lattice.TriangLattice(Npoints, save )
Vol_rec=l.Vol_BZ()

[KX,KY]=l.Generate_lattice_SQ()
plt.scatter(KX,KY, s=1)
plt.show()

# Npoints=4000
# save=True
# l=Lattice.TriangLattice(Npoints, save )
# Vol_rec=l.Vol_BZ()
# [KX,KY]=l.read_lattice(sq=1)
# plt.scatter(KX[::1000],KY[::1000])
# plt.show()


Npoints=400
save=True
l=Lattice.TriangLattice(Npoints, save )
Vol_rec=l.Vol_BZ()
# [KX,KY]=l.Generate_lattice_SQ()
[KX,KY]=l.read_lattice()
plt.scatter(KX[::100],KY[::100])
plt.show()