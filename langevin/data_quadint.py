import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys
import time

################################
################################
################################
################################
#defining triangular lattice
a=1
a_1=a*np.array([1,0,0])
a_2=a*np.array([1/2,np.sqrt(3)/2,0])
zhat=np.array([0,0,1])

Vol_real=np.dot(np.cross(a_1,a_2),zhat)
b_1=np.cross(a_2,zhat)*(2*np.pi)/Vol_real
b_2=np.cross(zhat,a_1)*(2*np.pi)/Vol_real
Vol_rec=np.dot(np.cross(b_1,b_2),zhat)
#print(np.dot(a_2,b_2),np.dot(a_1,b_1))

Np=80
n1=np.arange(-Np,Np+1)
n2=np.arange(-Np,Np+1)

a_1=a_1[0:2]
a_2=a_2[0:2]
b_1=b_1[0:2]
b_2=b_2[0:2]


#getting the first brilloin zone from the Voronoi decomp of the recipprocal lattice
#input: reciprocal lattice vectors
#output: Points that delimit the FBZ -
#high symmetry points (for now just the triangular lattice will be implemented)
from scipy.spatial import Voronoi, voronoi_plot_2d
def FBZ_points(b_1,b_2):
    #creating reciprocal lattice
    Np=4
    n1=np.arange(-Np,Np+1)
    n2=np.arange(-Np,Np+1)
    Recip_lat=[]
    for i in n1:
        for j in n2:
            point=b_1*i+b_2*j
            Recip_lat.append(point)

    #getting the nearest neighbours to the gamma point
    Recip_lat_arr=np.array(Recip_lat)
    dist=np.round(np.sqrt(np.sum(Recip_lat_arr**2, axis=1)),decimals=10)
    sorted_dist=np.sort(list(set(dist)) )
    points=Recip_lat_arr[np.where(dist<sorted_dist[2])[0]]

    #getting the voronoi decomposition of the gamma point and the nearest neighbours
    vor = Voronoi(points)
    Vertices=(vor.vertices)

    #ordering the points counterclockwise in the -pi,pi range
    angles_list=list(np.arctan2(Vertices[:,1],Vertices[:,0]))
    Vertices_list=list(Vertices)

    #joint sorting the two lists for angles and vertices for convenience later.
    # the linear plot routine requires the points to be in order
    # atan2 takes into acount quadrant to get the sign of the angle
    angles_list, Vertices_list = (list(t) for t in zip(*sorted(zip(angles_list, Vertices_list))))

    ##getting the M points as the average of consecutive K- Kp points
    Edges_list=[]
    for i in range(len(Vertices_list)):
        Edges_list.append([(Vertices_list[i][0]+Vertices_list[i-1][0])/2,(Vertices_list[i][1]+Vertices_list[i-1][1])/2])

    Gamma=[0,0]
    K=Vertices_list[0::2]
    Kp=Vertices_list[1::2]
    M=Edges_list[0::2]
    Mp=Edges_list[1::2]

    return Vertices_list, Gamma, K, Kp, M, Mp

Vertices_list, Gamma, K, Kp, M, Mp=FBZ_points(b_1,b_2)



################################
################################
################################
################################
############
####GETTING THE DISPERSION FOR PALLADIUM
J=2*5.17
tp1=568/J #in units of J
tp2=-108/J #/tpp1
#defining tight bindind dispersion
a=1
a_1=a*np.array([1,0])
a_2=a*np.array([1/2,np.sqrt(3)/2])

#creating lattice
Np=4
n1=np.arange(-Np,Np+1)
n2=np.arange(-Np,Np+1)
lat=[]
for i in n1:
    for j in n2:
        point=a_1*i+a_2*j
        lat.append(point)

#getting the nearest neighbours to the gamma point
lat_arr=np.array(lat)
dist=np.round(np.sqrt(np.sum(lat_arr**2, axis=1)),decimals=8)
sorted_dist=np.sort(list(set(dist)) )
nns=lat_arr[np.where( (dist<sorted_dist[2]) *( dist>0 ))[0]]
nnns=lat_arr[np.where( (dist<sorted_dist[3]) *( dist>sorted_dist[1] ))[0]]
def e2d(kx,ky, mu):
    e=0
    for nn in nns:
        e=e-tp1*np.exp(1j*(nn[0]*kx+nn[1]*ky))
    for nnn in nnns:
        e=e-tp2*np.exp(1j*(nnn[0]*kx+nnn[1]*ky))
    return np.real(e)-mu

x = np.linspace(-10, 10, 300)
X, Y = np.meshgrid(x, x)
Z = e2d(X, Y, 0)
band_max=np.max(Z)
band_min=np.min(Z)
print("band maximum and minimum...",band_max,band_min)
print("bandwidth...",band_max-band_min)

print("Reading data ....." )
start = time.time()
print("Reading kpoints ....." )
KX,KY=np.loadtxt("Kpoints.dat")
print("Reading structure factor ....." )
SF=np.loadtxt(sys.argv[1])
print("Reading unique frequencies ....." )
omegas=np.loadtxt("freqs.dat")
end = time.time()
print("done with elapsed time ....." , end - start)

T=float(sys.argv[2])
i=10
################################
################################
################################
################################
########INTEGRATING TO GET THE SELF ENERGY

points = np.array([KX,KY]).T
values = SF[i,:]

def integrand(qx,qy,kx,ky,w,i,points):
    mu=0
    ed=-tp1*(2*np.cos(qx)+4*np.cos(qx/2)*np.cos(np.sqrt(3)*qy/2))
    ed=ed-tp2*(2*np.cos(np.sqrt(3)*qy)+4*np.cos(3*qx/2)*np.cos(np.sqrt(3)*qy/2))
    ed=ed-mu
    eps=1e-17
    om=w-ed
    om2=-ed

    nb_we=1/(np.exp(om/T)-1)
    nf_e=1/(np.exp(om2/T)+1)
    values = SF[i,:]
    SS=griddata(points, values, [qx,qy], method='cubic')[0]
    return np.real(2*np.pi*SS*(nb_we+nf_e)/(1+nb_we+eps*1j))


m2=(K[2][1]-Kp[2][1])/(K[2][0]-Kp[2][0])
b2=K[2][1]-m2*K[2][0]

m1=(K[0][1]-Kp[2][1])/(K[0][0]-Kp[2][0])
b1=K[0][1]-m1*K[0][0]


m4=(K[1][1]-Kp[1][1])/(K[1][0]-Kp[1][0])
b4=K[1][1]-m4*K[1][0]

m3=(K[1][1]-Kp[0][1])/(K[1][0]-Kp[0][0])
b3=K[1][1]-m3*K[1][0]

def Sigm(kx,ky,omega,points,i):
    return np.sum(np.sum( integrand(KX,KY,kx,ky,omega,i,points) ))*Vol_rec/np.prod(np.shape(KX))


yup1=lambda x: K[2][1]
yup2=lambda x: m2*x+b2
yup3=lambda x: m4*x+b4

ydwn1=lambda x : K[0][1]
ydwn2=lambda x : m1*x+b1
ydwn3=lambda x : m3*x+b3

def Sigm2(kx,ky,omega,i,points):
    c1 = integrate.dblquad(integrand,K[0][0],Kp[0][0], lambda x : K[0][1], lambda x: K[2][1], args=(kx,ky,omega,i,points))
    c2 = integrate.dblquad(integrand,Kp[2][0],K[0][0], lambda x : m1*x+b1, lambda x: m2*x+b2, args=(kx,ky,omega,i,points))
    c3 = integrate.dblquad(integrand,Kp[0][0],K[1][0], lambda x : m3*x+b3, lambda x: m4*x+b4, args=(kx,ky,omega,i,points))

    return c1[0]+c2[0]+c3[0]

def Sigm3(kx,ky,omega,i,points):
    c1 = integrate.dblquad(integrand,K[0][0],Kp[0][0], ydwn1, yup1, args=(kx,ky,omega,i,points))
    c2 = integrate.dblquad(integrand,Kp[2][0],K[0][0], ydwn2, yup2, args=(kx,ky,omega,i,points))
    c3 = integrate.dblquad(integrand,Kp[0][0],K[1][0], ydwn3, yup3, args=(kx,ky,omega,i,points))

    return c1[0]+c2[0]+c3[0]

#return np.sum(np.sum( integrand(KX,KY,kx,ky,omega,T,alph,ll) ))*Vol_rec/np.prod(np.shape(KX))

start = time.time()
print( "cosas ",Sigm(0.1,0.1,omegas[i],i,points) )
end = time.time()
print(end - start)
