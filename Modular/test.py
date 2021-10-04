import numpy as np
import Lattice
import StructureFactor
import Dispersion
import matplotlib.pyplot as plt

Npoints=100
save=True
l=Lattice.TriangLattice(Npoints, save )
[KX,KY]=l.read_lattice()

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
tp2=-tp1*108/568 #/tpp1
##coupling 
U=4000/J
g=100/J
Kcou=g*g/U


#Dos test
ed=Dispersion.Dispersion_single_band([tp1,tp2],25)
# [nn, earr,Dos]=ed.DOS(size_E=500, Npoi_ints=1200)
# plt.plot(earr, Dos)
# plt. show()
# plt.plot(nn, Dos)
# plt. show()
# plt.plot(earr, nn)
# plt. show()
print("Filling is ...",ed.filling)


##fermi velocity test
# x = np.linspace(-2*np.pi, 2*np.pi, 30)
# X, Y = np.meshgrid(x, x)
# [v,u] = ed.Fermi_Vel_second_NN_triang(X, Y)
# Vertices_list, Gamma, K, Kp, M, Mp=l.FBZ_points(l.b[0,:],l.b[1,:])
# VV=np.array(Vertices_list+[Vertices_list[0]])
# plt.plot(VV[:,0], VV[:,1])
# plt.quiver(X,Y,u,v)
# plt.show()


[v,u] = ed.Fermi_Vel_second_NN_triang(KX, KY)
VF2=v**2+u**2
# Vertices_list, Gamma, K, Kp, M, Mp=l.FBZ_points(l.b[0,:],l.b[1,:])
# VV=np.array(Vertices_list+[Vertices_list[0]])
# plt.plot(VV[:,0], VV[:,1])
plt.scatter(KX,KY, c=VF2)
plt.show()

##fermi energy and parameters for the paramagnon propagator
EF=ed.EF
m=EF/2
gamma=EF*1000
vmode=EF/2
gcoupl=EF/2

SF_dyna2=SS.Dynamical_SF_PM_zeroQ(KX,KY, 0.1, gamma, vmode, m)
plt.scatter(KX,KY, c=SF_dyna2)
plt.show()