#!/usr/bin/python3 -u

##########################################

"""
Reads the structure factor at a given temperature, interpolates 
linearly and calculates the self energy at sparsely sampled points on
the fermi surface. This is done for frequencies up to 12*T. The value of 
-Im \Sigma(Kf,0) is NOT substracted


REQUIREMENTS
Path to the directory where the .npy files reside
MODIFY LINE 66 WITH THE LOCATION OF THE STRUCTURE FACTOR DATA

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
T=float(Ta)

print("loading data for the structure factor at T="+Ta)
s=time.time()
# Load the data files
#MAC
dsf_data = np.load('/Users/jfmv/Documents/Proyectos/Delafossites/Struc_dat/dsf_TLHAF_L=120_tf=4096_T='+Ta+'.npy')
##DebMac
#dsf_data = np.load('/Users/Felipe/Documents/Struc_dat/dsf_TLHAF_L=120_tf=4096_T='+Ta+'.npy')
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











def gamma2(kx,ky):
    return 2*np.cos(kx)+4*np.cos(kx/2)*np.cos(np.sqrt(3)*ky/2)


def Sf(kx,ky,lam,T):
    return 3/(lam+(1/T)*gamma2(kx,ky))
def f(lam,T,KX,KY):
    curlyN=np.size(KX)
    return np.sum(Sf(KX,KY,lam,T))/curlyN -1
##bisection method to solve the large n self consistency equation


def bisection(f,a,b,N,T,KX,KY):
    '''Approximate solution of f(x)=0 on interval [a,b] by bisection method.

    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x)=0.
    a,b : numbers
        The interval in which to search for a solution. The function returns
        None if f(a)*f(b) >= 0 since a solution is not guaranteed.
    N : (positive) integer
        The number of iterations to implement.

    Returns
    -------
    x_N : number
        The midpoint of the Nth interval computed by the bisection method. The
        initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some
        midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iteration, the bisection method fails and return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> bisection(f,1,2,25)
    1.618033990263939
    >>> f = lambda x: (2*x - 1)*(x - 3)
    >>> bisection(f,0,1,10)
    0.5
    '''
    if f(a,T,KX,KY)*f(b,T,KX,KY) >= 0:
        print("Bisection method fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1,N+1):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n,T,KX,KY)
        if f(a_n,T,KX,KY)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n,T,KX,KY)*f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            print("Found exact solution.")
            return m_n
        else:
            print("Bisection method fails.")
            return None
    return (a_n + b_n)/2


lam=bisection(f,3/T,20,50,T,KX,KY)
lam=3.018732903302169 
print("testing solution to large N equation...",lam,f(lam,T,KX,KY)+1)



#T=1.0
# alph=np.array([ 0.7097908959336873,  -0.0043594581070084135,  -0.004495974146928671, -0.024777430963518057,   0.0030982360905670333,   0.0004539363283678258])
# et=np.array([0.23331490064983912,  0.06490355420597822,    -0.03601601298488789,   -0.04655841264762831,    -0.010189892955121571, -0.006643162950435294])
# lam=4.178642027077301

#T=3.0
# alph=np.array([ 0.6071415409901372, -0.007643725101933083,    -0.004102812828401734,    -0.0064882051217971795,  0.001523532730774404, 2.9287972860276336e-05 ])
# et=np.array([0.09131678420721018,    0.005815174776661578,  -0.00670989716658747,   -0.006410702279227802,   0.0011528049552485798,  0.0003122379970753175])
# lam=3.1806350971738353 

#T=10.0
alph=np.array([0.6092518069471177,   -0.0017454331191290237,    0.0021259053889015845,   0.0004188012953199125, 0.0012489555790225417,  0.0003255774536971311])
et=np.array([0.12385676180579733,   -0.009155564378675983,   0.0008941115202702899,      -0.0005938474219710233,    0.0019469008555008608,      0.0001013876862340809])
lam=3.018732903302169 

def dsf2(qx, qy, f):

    gamma0=1
    gamma1=(1/3.0)*(np.cos(qx)+2*np.cos(qx/2)*np.cos(np.sqrt(3)*qy/2))
    gamma2=(1/3.0)*(2*np.cos(3*qx/2)*np.cos(np.sqrt(3)*qy/2)+np.cos(np.sqrt(3)*qy))
    gamma3=(1/3.0)*(np.cos(2*qx)+2*np.cos(2*qx/2)*np.cos(2*np.sqrt(3)*qy/2))
    gamma4=(1/3.0)*( np.cos(5*qx/2)*np.cos(np.sqrt(3)*qy/2) +np.cos(2*qx)*np.cos(np.sqrt(3)*qy) +np.cos(qx/2)*np.cos(3*np.sqrt(3)*qy/2) )
    gamma5=(1/3.0)*(np.cos(3*qx)+2*np.cos(3*qx/2)*np.cos(3*np.sqrt(3)*qy/2))
    gam=np.array([gamma0,gamma1,gamma2,gamma3,gamma4,gamma5])
 

    et_q=np.sum(et*gam)*(6-6*gamma1)*(6-6*gamma1)
    alpha_q=np.sum(alph*gam)
    #additional 2 pi for the correct normalization of the frequency integral
    NN=2*np.pi*np.abs( alpha_q*np.sqrt( et_q*( et_q-1 +1j*1e-17) )/np.arcsinh( np.sqrt( (et_q-1+1j*1e-17) ) ) )

    fac=NN/(np.sinh(alpha_q*f)*np.sinh(alpha_q*f)+et_q)


    return Sf(qx,qy,lam,T)*fac # this has to be called in the reverse order for some reason.



















print("starting plots")
freqn_div=np.logspace(0,3,30)
for ii,n in enumerate(freqn_div):
    w=(2*np.pi-0.005)/n
    plt.scatter(KX,KY, c=dsf(KX, KY, w  ),s=3)
    plt.title(r'$\omega =$'+str(w))
    plt.colorbar()
    plt.savefig("T_"+Ta+"_n_"+str(ii)+"omega_"+str(n)+"_.png")
    print("T_"+Ta+"omega_"+str(n)+"_.png")
    plt.close()

for ii,n in enumerate(freqn_div):
    w=(2*np.pi-0.005)/n
    plt.scatter(KX,KY, c=dsf2(KX, KY, w  ),s=3)
    plt.title(r'$\omega =$'+str(w))
    plt.colorbar()
    plt.savefig("fit_T_"+Ta+"_n_"+str(ii)+"omega_"+str(n)+"_.png")
    print("fit_T_"+Ta+"omega_"+str(n)+"_.png")
    plt.close()


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

