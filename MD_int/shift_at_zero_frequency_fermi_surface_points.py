#!/usr/bin/python3 -u

##########################################

"""
Reads the structure factor at a given temperature, interpolates 
linearly and calculates the self energy at sparsely sampled points on
the fermi surface. This is done for frequencies up to 12*T. The value of 
-Im \Sigma(Kf,0) is substracted_


REQUIREMENTS
Path to the directory where the .npy files reside

ARGUMENTS:
1) T: temperature in units of J at which the structure factor was calculated
2) LP: size of the grid used for integration is LPxLP in the FBZ

OUTPUT:
Figures with the frequency dependent self energy at different points in the FS
A figure with the color coded points where the Self energy was calculated

"""

##########################################



############################################################
# Importing libraries
############################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator # You may have some better interpolation methods
import time
import sys



############################################################
# Reading the structure factor
############################################################
L = 120
n_freqs = 4097



# Momentum and frequency ranges (with some built in buffers)


K1 = np.arange(-4*L//3, 4*L//3)
K2 = np.arange(-4*L//3, 4*L//3)

nX,nY=np.meshgrid(K1,K2)

F = np.arange(0, n_freqs)
Ta=sys.argv[1]


print("loading data for the structure factor at T="+Ta)
s=time.time()
# Load the data files
#MAC
#dsf_data = np.load('/Users/jfmv/Documents/Proyectos/Delafossites/Struc_dat/dsf_TLHAF_L=120_tf=4096_T='+Ta+'.npy')
##DebMac
dsf_data = np.load('/Users/Felipe/Documents/Struc_dat/dsf_TLHAF_L=120_tf=4096_T='+Ta+'.npy')
#linux
# dsf_data = np.load('/home/juan/Documents/Projects/Delafossites/SF_data/dsf_TLHAF_L=120_tf=4096_T=0.5.npy')

e=time.time()
print("time for loading", e-s)



def dsf_func(k1, k2, w):

    return dsf_data[k1%L, k2%L, w]




print("reshaping the original array....")
s=time.time()
#dsf_func_data = np.array([[[dsf_func(k1, k2, w) for k1 in K1] for k2 in K2] for w in F])
dsf_func_data = np.array([ dsf_func( nX,nY, w) for w in F])

print(np.shape(dsf_func_data ))
e=time.time()
print("time for recasting", e-s)




############################################################
# Interpolating 
############################################################

s=time.time()
dsf_interp = RegularGridInterpolator((F, K1, K2), dsf_func_data, method='linear')
e=time.time()
print("time for interpolation", e-s)


def dsf(qx, qy, f):

    k1 = L*qx/(2*np.pi)
    k2 = L*(qx/2 + np.sqrt(3)*qy/2)/(2*np.pi)
    w = n_freqs*f/(2*np.pi)

    return dsf_interp((w, k2, k1)) # this has to be called in the reverse order for some reason.




############################################################
# Defining tringular lattice
############################################################

a=1
a_1=a*np.array([1,0,0])
a_2=a*np.array([1/2,np.sqrt(3)/2,0])
zhat=np.array([0,0,1])

Vol_real=np.dot(np.cross(a_1,a_2),zhat)
b_1=np.cross(a_2,zhat)*(2*np.pi)/Vol_real
b_2=np.cross(zhat,a_1)*(2*np.pi)/Vol_real
Vol_rec=np.dot(np.cross(b_1,b_2),zhat)
#print(np.dot(a_2,b_2),np.dot(a_1,b_1))



Np=200
n1=np.arange(-Np,Np+1)
n2=np.arange(-Np,Np+1)

a_1=a_1[0:2]
a_2=a_2[0:2]
b_1=b_1[0:2]
b_2=b_2[0:2]


############################################################
# Function that calculates some high symmetry proints for the FBZ of the triangular lattice
# using a voronoi decomposition of the lattice constructed above
############################################################

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

k_window_sizey = K[2][1] 
k_window_sizex = K[1][0] 

Radius_inscribed_hex=1.000000000000001*k_window_sizex

############################################################
# Function to filter recuprocal space samples restricting only to those inside the FBZ
############################################################

def hexagon(pos):
    x, y = map(abs, pos) #taking the absolute value of the rotated hexagon, only first quadrant matters
    return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge


############################################################
###denser grid for integration
############################################################
print("starting sampling in reciprocal space....")
s=time.time()
LP=int(sys.argv[2])
n1=np.arange(-LP,LP+1,1)
n2=np.arange(-LP,LP+1,1)
n_1,n_2=np.meshgrid(n1,n2)

n_1p=[]
n_2p=[]
for x in n1:
    for y in n2:
        kx=2*np.pi*x/LP
        ky=2*(2*np.pi*y/LP - np.pi*x/LP)/np.sqrt(3)
        if hexagon(( kx, ky)):
            #plt.scatter(kx_rangex[x],ky_rangey[y])
            n_1p.append(x)
            n_2p.append(y)

KXX=2*np.pi*n_1/LP
KYY= 2*(2*np.pi*n_2/LP - np.pi*n_1/LP)/np.sqrt(3)


n_1pp=np.array(n_1p)
n_2pp=np.array(n_2p)

KX=2*np.pi*n_1pp/LP
KY= 2*(2*np.pi*n_2pp/LP - np.pi*n_1pp/LP)/np.sqrt(3)
e=time.time()
print("finished sampling in reciprocal space....")
print("time for sampling was...",e-s)


# plt.scatter(KX,KY, c=dsf(KX, KY, 2*np.pi-0.005  ),s=3)
# plt.show()

############################################################
# Function that creates an array of points in reciprocal space that connects a list of specified points 
############################################################

def linpam(Kps,Npoints_q):
    Npoints=len(Kps)
    t=np.linspace(0, 1, Npoints_q)
    linparam=np.zeros([Npoints_q*(Npoints-1),2])
    for i in range(Npoints-1):
        linparam[i*Npoints_q:(i+1)*Npoints_q,0]=Kps[i][0]*(1-t)+t*Kps[i+1][0]
        linparam[i*Npoints_q:(i+1)*Npoints_q,1]=Kps[i][1]*(1-t)+t*Kps[i+1][1]

    return linparam


VV=Vertices_list+[Vertices_list[0]]
Nt=1000
kpath=linpam(VV,Nt)


############################################################
############################################################
#Defining integrand
###other Parameters
############################################################
############################################################


T=float(Ta)
J=2*5.17
tp1=568/J #in units of J
tp2=-108/J #/tpp1


############################################################
#Pd dispersion
############################################################
mu=0
def Disp(kx,ky,mu):
    ed=-tp1*(2*np.cos(kx)+4*np.cos((kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
    ed=ed-tp2*(2*np.cos(np.sqrt(3)*(ky))+4*np.cos(3*(kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
    ed=ed-mu
    return ed


x = np.linspace(-3.8, 3.8, 500)
X, Y = np.meshgrid(x, x)
Z = Disp(X, Y, mu)


############################################################
# Getting 9 points that are equally spaced angularly around the FS
############################################################

c= plt.contour(X, Y, Z, levels=[0],linewidths=3, cmap='summer');
v = c.collections[0].get_paths()[0].vertices
NFSpoints=9
xFS = v[5::int(np.size(v[:,1])/NFSpoints),0]
yFS = v[5::int(np.size(v[:,1])/NFSpoints),1]


KFx=xFS[0] #for plotting line \sim 338
KFy=yFS[0] #for plotting line \sim 338
for ell in range(np.size(xFS)):
    plt.scatter(xFS[ell],yFS[ell])
plt.savefig("T_"+str(T)+"func_fermi_surf_reference.png", dpi=200)
plt.close()


############################################################
#integrand of the self-energy. 
############################################################
#first two arguments are the arguments of the self energy, qx,qy. The second set
#of momenta kx and ky are the ones over which the integration is carried
def integrand_Disp(qx,qy,kx,ky,w):

    ed=Disp(kx+qx,ky+qy,mu)
    om=w-ed
    om2=-ed

    #for frequencies above the threshold, we set the structure factor to be evaluated at the threshold
    # it is also assumed that the structure factor is an even function of frequency
    thres=2*np.pi-0.005
    ind_valid=np.where(np.abs(om)<thres)
    om3=np.ones(np.shape(om))*thres
    om3[ind_valid]=np.abs(om[ind_valid])

    fac_p=np.exp(ed/T)*(1+np.exp(-w/T))/(1+np.exp(ed/T))
    return dsf(kx,ky, om3 )*2*np.pi*fac_p



omegas=np.linspace(0,2*np.pi,n_freqs)

# n_3=200
# w=omegas[n_3]
# print(w)
# plt.scatter(KX,KY, c=integrand_Disp(0.1,0.1,KX,KY,w),s=3)
# plt.show()

############################################################
# Integration over the FBZ 
############################################################

siz=60
Omegs=np.linspace(0 ,12*T ,siz)

#######frequency dependence in restricted range
print("starting with calculation frequency dependence at diferent points in the FS.....")
s=time.time()
sigm_FS=[]
#loop over FS points
for ell in range(np.size(xFS)):

    #phi=2*np.pi/6 #rotation angle when testing whether the integration routine gave the same result under the action of C6
    #rotating and cleaving if absolute value of rotated point's y coordinate exceeds top boundary of 1BZ
    #KFx=np.cos(phi)*xFS[ell]-np.sin(phi)*yFS[ell]
    #KFy=np.sin(phi)*xFS[ell]+np.cos(phi)*yFS[ell]
    
    #Fermi surface coordinates
    KFx=xFS[ell]
    KFy=yFS[ell]

    #volume element
    ds=Vol_rec/np.size(KX)
    sigm=[]
    S0=np.sum(integrand_Disp(KFx,KFy,KX,KY,0)*ds)

    #looping over frequencies
    for i in range(siz):
        start=time.time()

        ##uncomment below for removing the divergence at q=0 w=0
        # if (Omegs[i]!=0):
        #     SI=np.sum(integrand_Disp(KFx,KFy,KX,KY,Omegs[i])*ds)-S0
        # else:
        
        #     SS=integrand_Disp(KFx,KFy,KX,KY,Omegs[i])*ds
        #     ind=np.where(abs(KX+KY)<1e-10)[0]
        #     KX=np.delete(KX,ind)
        #     KY=np.delete(KY,ind)
        #     SS=np.delete(SS,ind)
        #     SI=np.sum(SS)-S0

        SI=np.sum(integrand_Disp(KFx,KFy,KX,KY,Omegs[i])*ds)-S0
        end=time.time()
        print("time ",end-start)
        print(i,SI)
        sigm.append(SI)
    plt.plot(Omegs,sigm, 'o', label="T="+str(T)+" ,kx="+str(np.round(KFx,3))+" ,ky="+str(np.round(KFy,3)))
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"-Im$\Sigma (k_F,\omega)$")
    plt.legend()
    plt.savefig("kx_"+str(KFx)+"_ky_"+str(KFy)+"_T_"+str(T)+"func.png", dpi=200)
    #plt.close()
e=time.time()
print("finished  calculation .....")
print("time for calc....",e-s)
