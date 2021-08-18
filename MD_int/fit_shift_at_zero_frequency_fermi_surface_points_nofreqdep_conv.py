#!/usr/bin/python3 -u

##########################################

"""
Reads the structure factor at a given temperature, interpolates 
linearly and calculates the self energy at the fermi surface at zero frequecy
the sampling is carried out by generating a triangular lattice of the size of the FBZ
with the integral being calculated over the conventional unit cell



REQUIREMENTS
Path to the directory where the .npy files reside

ARGUMENTS:
1) T: temperature in units of J at which the structure factor was calculated
2) LP: size of the grid used for integration is LPxLP in the FBZ

OUTPUT:
2 figures, 
one has a color plot of -Im \Sigma across the fermi surface
one has the value of -Im \Sigma as a function of angle


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

G=np.sqrt(np.sum(b_2**2))
print(G, b_2)
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

# LP=int(sys.argv[2])
# Nsamp=LP
# n_1=np.arange(0,Nsamp,1)
# n_2=np.arange(0,Nsamp,1)

# n_1p,n_2p=np.meshgrid(n_1,n_2)
# k1=np.array([1,0])
# k2=np.array([1/2,np.sqrt(3)/2])
# KX=(1/Nsamp)*(b_1[0]*n_1p+b_2[0]*n_2p)
# KY=(1/Nsamp)*(b_1[1]*n_1p+b_2[1]*n_2p)



# Nsamp_p=100
# n_1_2=np.arange(-Nsamp_p/2,Nsamp_p/2,1)
# n_2_2=np.arange(-Nsamp_p/2,Nsamp_p/2,1)

# n_1p_2,n_2p_2=np.meshgrid(n_1_2,n_2_2)
# KX2=(2/Nsamp_p)*(b_1[0]*n_1p_2+b_2[0]*n_2p_2)
# KY2=(2/Nsamp_p)*(b_1[1]*n_1p_2+b_2[1]*n_2p_2)
# e=time.time()
# print("finished sampling in reciprocal space....")
# print("time for sampling was...",e-s)
############################################################
############################################################
#Fit for the structure factor
############################################################
############################################################



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
if(T==1.0):
    alph=np.array([ 0.7097908959336873,  -0.0043594581070084135,  -0.004495974146928671, -0.024777430963518057,   0.0030982360905670333,   0.0004539363283678258])
    et=np.array([0.23331490064983912,  0.06490355420597822,    -0.03601601298488789,   -0.04655841264762831,    -0.010189892955121571, -0.006643162950435294])
    lam=4.178642027077301

#T=2.0
if(T==2.0):
    alph=np.array([0.6222520758430777,   -0.009461521513823186,     -0.006053452180584596,   -0.010702516424885714,   0.0020003919364024714,     -1.0205060481952911e-05])
    et=np.array([ 0.09658650764378539,  0.013384215891118253,    -0.010516833605162713,      -0.01090143816114151,      0.0004144707853819521,      0.0001204480954048534])   
    lam= 3.370944783098885

#T=3.0
if(T==3.0):
    alph=np.array([ 0.6071415409901372, -0.007643725101933083,    -0.004102812828401734,    -0.0064882051217971795,  0.001523532730774404, 2.9287972860276336e-05 ])
    et=np.array([0.09131678420721018,    0.005815174776661578,  -0.00670989716658747,   -0.006410702279227802,   0.0011528049552485798,  0.0003122379970753175])
    lam=3.1806350971738353 

#T=4.0
if(T==4.0):
    alph=np.array([0.6039735698700098,  -0.006047485346815116,  -0.002472531200563213,  -0.004275139567596024,  0.0013215101055664485,  7.175902842573632e-05])
    et=np.array([0.09553529940513966,    0.0018186539475888166, -0.004739620696629819,   -0.004494332347087183,  0.0013142362882892138,  0.000291236774009811])
    lam=3.106684811722399

#T=5.0
if(T==5.0):
    alph=np.array([0.6040810534137876,  -0.004732865622040766,  -0.001127349144342822,  -0.0027706232289313806,     0.0013410007192414624,      0.0002091028903354993])
    et=np.array([0.1011836355370694,    -0.0009584775768396963, -0.003315265600935579  ,    -0.0033082929540142105,  0.0013589600008588089,  0.00027718680067235644])
    lam=3.0703759314285626 

#T=6.0
if(T==6.0):
    alph=np.array([0.6048825771130464,  0.008438891265335053,   -0.00020510810262157957,    -0.0018087562251271861 ,    0.0012503605738861706, 0.00019497167542968042]) 
    et=np.array([0.106654110761412, -0.0032025150693937156, -0.0021998569308273493, -0.0024967677601908135, 0.0012877521033920923,  0.00021783472445787648])
    lam=3.0498886949148036

#T=7.0
if(T==7.0):
    alph=np.array([0.6060957381074707,  -0.0030658995822706505, 0.0006420851435837167,  -0.0009717074866372171,  0.0013131665577559325, 0.0003018262351056688]) 
    et=np.array([0.11175745410773015,   -0.005039700509756049,  -0.0012315951887033312, -0.001841829723904746,  0.0012274174484950366,   0.00019865688157838827])
    lam=3.037203270341933

#T=8.0
if(T==8.0):
    alph=np.array([0.6070551862962202,  -0.002697101301333131,  0.00107489031858635,    -0.0005590867861957349,     0.0011475930927934638,   0.00018205256220494518])
    et=np.array([ 0.11616954883061426,  -0.003988873565023515,   -0.0004657571027108079,    -0.0013839453692115198, 0.0011044737627527907, 0.00012737532156294055]) 
    lam=3.028807008086399

 
#T=9.0
if(T==9.0):
    alph=np.array([0.6083718387945705,  0.010508584347927811,   0.0018022331604978998,  0.00012473785812128,    0.0013244365310333586,  0.00037723431092535686])
    et=np.array([0.12033706994827463,    -0.007946874704261314, 0.00030420836462192086,  -0.00090404839139119,  0.001061248299460572,   0.00014865459778067692]) 
    lam= 3.0229631820378184


#T=10.0
if(T==10.0):
    alph=np.array([0.6092518069471177,   -0.0017454331191290237,    0.0021259053889015845,   0.0004188012953199125, 0.0012489555790225417,  0.0003255774536971311])
    et=np.array([0.12385676180579733,   -0.009155564378675983,   0.0008941115202702899,      -0.0005938474219710233,    0.0019469008555008608,      0.0001013876862340809])
    lam=3.018732903302169


#T=50.0
if(T==50.0):
    alph=np.array([ 0.6201774944069754, 0.0012530224630861754,   0.005833622305428591,  0.003605079260905222,   0.0014324738571124083,   0.0006316328735677271])
    et=np.array([0.16281297212996357,   -0.021355929675790707,   0.007802291525953633,  0.002565178078007373,   -0.0004250484192336371, -3.285858634623746e-05])
    lam=3.000789439969265

#T=100.0
if(T==100.0):
    alph=np.array([0.6220438193343075,  0.0016537316919072811,  0.006387742935248672,   0.004060505526695932,    0.0014967727700990639,  0.000700872036530507])
    et=np.array([0.1697667873959355,    -0.023474171445420244,   0.009095251231202181,  0.0030821033954326386,  -0.0007082689712385551, -2.655211696552507e-05])
    lam=3.00019867333284


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


print("starting plots of SF")
freqn_div=np.logspace(0,3,30)

for ii,n in enumerate(freqn_div):
    w=(2*np.pi-0.005)/n
    plt.scatter(KX,KY, c=dsf2(KX, KY, w  ),s=3)
    plt.title(r'$\omega =$'+str(w))
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("fit_T_"+Ta+"_n_"+str(ii)+"omega_"+str(n)+"_.png")
    print("fit_T_"+Ta+"omega_"+str(n)+"_.png")
    plt.close()

print("finishing plots of SF")
############################################################
############################################################
############################################################


for ii,n in enumerate(freqn_div):
    w=(2*np.pi-0.005)/n
    plt.scatter(KX,KY, c=  dsf2(KX, KY, w  )/(2+2/(np.exp(w/T)-1)) ,s=3)
    plt.title(r'$\omega =$'+str(w))
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("CHI_fit_T_"+Ta+"_n_"+str(ii)+"omega_"+str(n)+"_.png")
    print("CHI_fit_T_"+Ta+"omega_"+str(n)+"_.png")
    plt.close()

print("finishing plots of SF")
############################################################
############################################################
############################################################


"""
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



J=2*5.17
tp1=568/J #in units of J
tp2=-108/J #/tpp1


############################################################
#Pd dispersion
############################################################
mu=0#-4*tp1
def Disp(kx,ky,mu):
    ed=-tp1*(2*np.cos(kx)+4*np.cos((kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
    ed=ed-tp2*(2*np.cos(np.sqrt(3)*(ky))+4*np.cos(3*(kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
    ed=ed-mu
    return ed


x = np.linspace(-3.4, 3.4, 2603)
print(x)
X, Y = np.meshgrid(x, x)
Z = Disp(X, Y, mu)


############################################################
# Getting 9 points that are equally spaced angularly around the FS
############################################################

c= plt.contour(X, Y, Z, levels=[0],linewidths=3, cmap='summer');
#plt.show()
v = c.collections[0].get_paths()[0].vertices
NFSpoints=9
xFS = v[5::int(np.size(v[:,1])/NFSpoints),0]
yFS = v[5::int(np.size(v[:,1])/NFSpoints),1]

############################################################
# Getting a denser sampling of the FS
############################################################
xFS_dense = v[::,0]
yFS_dense = v[::,1]
print("dense shape",np.shape(yFS_dense))
KFx=xFS[0]
KFy=yFS[0]
for ell in range(np.size(xFS)):
    plt.scatter(xFS[ell],yFS[ell])
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
    #om3=np.ones(np.shape(om))*thres
    #om3[ind_valid]=np.abs(om[ind_valid])

    fac_p=np.exp(ed/T)*(1+np.exp(-w/T))/(1+np.exp(ed/T))
    return dsf2(kx,ky, om )*2*np.pi*fac_p



omegas=np.linspace(0,2*np.pi,n_freqs)

n_3=200
w=omegas[n_3]
print(w)
# plt.scatter(KX,KY, c=integrand_Disp(0.,0.,KX,KY,w),s=1)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

############################################################
# Integration over the FBZ 
############################################################

shifts=[]
angles=[]

print("starting with calculation of Sigma theta w=0.....")
s=time.time()

for ell in range(np.size(xFS_dense)):

    KFx=xFS_dense[ell]
    KFy=yFS_dense[ell]


    ds=Vol_rec/np.size(KX)
    S0=np.sum(integrand_Disp(KFx,KFy,KX,KY,0.0)*ds)

    # uncomment below for removing divergence at q=0 w=0
    # SS=integrand_Disp(KFx,KFy,KX,KY,0)*ds
    # ind=np.where(abs(KX+KY)<1e-10)[0]
    # KX=np.delete(KX,ind)
    # KY=np.delete(KY,ind)
    # SS=np.delete(SS,ind)
    # S0=np.sum(SS)

    shifts.append(S0)
    angles.append(np.arctan2(KFy,KFx))
    print(ell, np.arctan2(KFy,KFx), S0)

e=time.time()
print("finished  calculation of Sigma theta w=0.....")
print("time for calc....",e-s)


with open("T_"+str(T)+'_fit.npy', 'wb') as f:
    np.save(f, xFS_dense)
    np.save(f, yFS_dense)
    np.save(f, angles)
    np.save(f, shifts)


with open("T_"+str(T)+'_fit.npy', 'rb') as f:

    xFS_dense = np.load(f)
    yFS_dense = np.load(f)
    angles = np.load(f)
    shifts = np.load(f)

plt.scatter(angles, shifts, s=1)
plt.xlabel(r"$\theta$")
plt.ylabel(r"-Im$\Sigma (k_F(\theta),0)$,T="+Ta)
#plt.ylim([6.9,9.8])
#plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("theta_T_"+str(T)+"func.png", dpi=200)
plt.close()


plt.scatter(xFS_dense,yFS_dense,c=shifts, s=3)
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("scatter_theta_T_"+str(T)+"func.png", dpi=200)
plt.close()


plt.scatter(xFS_dense,yFS_dense,c=np.log10(shifts-np.min(shifts)+1e-3), s=3)
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("log_scatter_theta_T_"+str(T)+"func.png", dpi=200)
plt.close()

"""