import numpy as np
import matplotlib.pyplot as plt
import sys

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


#######reciprocal lattice
#creating reciprocal lattice preparing for k-point integration
Recip_lat=[]
for i in n1:
    for j in n2:
        point=b_1*i+b_2*j
        Recip_lat.append(point)


################################
################################
################################
################################
#########
#####CREATING SAMPLES FOR INTEGRATION

#generating k points

#square sampling
#kxx,kyy=np.meshgrid(kx,ky)
scale_fac=0.00831
#scale_fac=0.0831
#Shrinking the reciprocal lattice
def hexsamp(b_1,b_2,scale_fac,Np_p):
    n1_p=np.arange(-Np_p,Np_p+1)
    n2_p=np.arange(-Np_p,Np_p+1)

    Recip_lat_p=[]
    for i in n1_p:
        for j in n2_p:
            point=b_1*i+b_2*j
            Recip_lat_p.append(point)

    Recip_lat_p_arr=scale_fac*np.array(Recip_lat_p)
    kxx=Recip_lat_p_arr[:,0]
    kyy=Recip_lat_p_arr[:,1]

    phi=2*np.pi/6 #rotation angle

    #rotating and cleaving if absolute value of rotated point's y coordinate exceeds top boundary of 1BZ
    kxx_rot=np.cos(phi)*kxx-np.sin(phi)*kyy
    kyy_rot=np.sin(phi)*kxx+np.cos(phi)*kyy

    kxx2=kxx[np.where(np.abs(kyy_rot)<K[2][1])]
    kyy2=kyy[np.where(np.abs(kyy_rot)<K[2][1])]

    #rotating and cleaving if absolute value of rotated point's y coordinate exceeds top boundary of 1BZ
    kxx_rot2=np.cos(-phi)*kxx2-np.sin(-phi)*kyy2
    kyy_rot2=np.sin(-phi)*kxx2+np.cos(-phi)*kyy2

    kxx3=kxx2[np.where(np.abs(kyy_rot2)<K[2][1])]
    kyy3=kyy2[np.where(np.abs(kyy_rot2)<K[2][1])]

    #cleaving if absolute value of point's y coordinate exceeds top boundary of 1BZ
    KX_p=kxx3[np.where(np.abs(kyy3)<K[2][1])]
    KY_p=kyy3[np.where(np.abs(kyy3)<K[2][1])]

    return KX_p,KY_p
KX, KY=hexsamp(b_1,b_2,scale_fac,Np)
S=np.ones(np.shape(KX))
print("volume of the FBZ",Vol_rec)
print("volume of the FBZ 2",np.sum(np.sum(S))*Vol_rec/np.prod(np.shape(KX)))


from scipy import integrate

def f(x,y): # function to integrate
    return 1


c1 = integrate.dblquad(f,K[0][0],Kp[0][0],lambda x : K[0][1], lambda x: K[2][1])

m2=(K[2][1]-Kp[2][1])/(K[2][0]-Kp[2][0])
b2=K[2][1]-m2*K[2][0]

m1=(K[0][1]-Kp[2][1])/(K[0][0]-Kp[2][0])
b1=K[0][1]-m1*K[0][0]

c2 = integrate.dblquad(f,Kp[2][0],K[0][0],lambda x : m1*x+b1, lambda x: m2*x+b2)

m4=(K[1][1]-Kp[1][1])/(K[1][0]-Kp[1][0])
b4=K[1][1]-m4*K[1][0]

m3=(K[1][1]-Kp[0][1])/(K[1][0]-Kp[0][0])
b3=K[1][1]-m3*K[1][0]

c3 = integrate.dblquad(f,Kp[0][0],K[1][0],lambda x : m3*x+b3, lambda x: m4*x+b4)


print(c1[0]+c2[0]+c3[0])


################################
################################
################################
################################
###########Static structure factor
#defining the parameter gamma in the static structure factor
def gamma(kx,ky):

    a_1=a*np.array([1,0])
    a_2=a*np.array([1/2,np.sqrt(3)/2])

    #creating lattice
    Np=4
    n1=np.arange(-Np,Np+1)
    n2=np.arange(-Np,Np+1)
    Recip_lat=[]
    for i in n1:
        for j in n2:
            point=a_1*i+a_2*j
            Recip_lat.append(point)

    #getting the nearest neighbours to the gamma point
    Recip_lat_arr=np.array(Recip_lat)
    dist=np.round(np.sqrt(np.sum(Recip_lat_arr**2, axis=1)),decimals=8)
    sorted_dist=np.sort(list(set(dist)) )
    nns=Recip_lat_arr[np.where( (dist<sorted_dist[2]) *( dist>0 ))[0]]
    e=0
    for nn in nns:
        e=e+np.exp(1j*(nn[0]*kx+nn[1]*ky))

    return np.real(e)

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

T=10

ll=bisection(f,3/T,20,50,T,KX,KY)
print("testing solution to large N equation...",ll,f(ll,T,KX,KY)+1)

################################
################################
################################
################################

###Dynamical structure factor
#temperature dependent fit parameters for the langevin dynamics
alphl=[0.0054342689, 0.00645511652936,0.0085441664872,0.008896935]
def alphfunc(T):
    return np.piecewise(T, [T <= 0.5, (T <= 1.0) & (T>0.5), (T <= 10.0) & (T>1.0), T>10.0], alphl)

T=float(sys.argv[1])
lam=bisection(f,3/T,40,170,T,KX,KY)
alph=alphfunc(T)
#dynamic structure fac
def Sfw(kx,ky,lam,T,ome, alph):
    fq=(gamma(kx,ky)**2)/T +gamma(kx,ky)*(lam-6/T)- 6*lam
    fq=alph*fq
    return -2*Sf(kx,ky,lam,T)*(fq/(ome**2+fq**2))

def Sfw2(kx,ky,lam,T,ome, alph):
    gam=2*np.cos(kx)+4*np.cos(kx/2)*np.cos(np.sqrt(3)*ky/2)
    SP=3/(lam+(1/T)*gam)
    fq=(gam**2)/T +gam*(lam-6/T)- 6*lam
    fq=alph*fq
    return -2*SP*(fq/(ome**2+fq**2))

#linear parametrization accross different points in the BZ
def linpam(Kps,Npoints_q):
    Npoints=len(Kps)
    t=np.linspace(0, 1, Npoints_q)
    linparam=np.zeros([Npoints_q*(Npoints-1),2])
    for i in range(Npoints-1):
        linparam[i*Npoints_q:(i+1)*Npoints_q,0]=Kps[i][0]*(1-t)+t*Kps[i+1][0]
        linparam[i*Npoints_q:(i+1)*Npoints_q,1]=Kps[i][1]*(1-t)+t*Kps[i+1][1]

    return linparam


VV=Vertices_list+[Vertices_list[0]]
L=[]
L=L+[K[1]]+[Gamma]+[Mp[1]]+[[np.pi,0]]
Nt=1000
kpath=linpam(L,Nt)


##geneerating arrays for imshow of momentum cut
#Nomegs=4097
Nomegs=257
omegas=np.linspace(1e-7,2*10*np.pi ,Nomegs)
#t=np.arange(0,len(kpath),1)
#t_m,omegas_m=np.meshgrid(t,omegas)
SFdat=np.array([Sfw2(KX,KY,lam,T,omega, alph) for omega in omegas])
Ks=np.array([KX,KY])
np.savetxt("Kpoints.dat",Ks)
np.savetxt("freqs.dat",omegas)
np.savetxt("SF_T="+sys.argv[1]+"_WEM.dat",SFdat)
print(np.shape(SFdat))
