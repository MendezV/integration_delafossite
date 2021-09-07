import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import time
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

################################
################################
################################
################################
#Reading and reshaping the structure factor
L = int(sys.argv[1])
n_freqs = 4097

# Load the data files
SF = np.load('/Users/jfmv/Documents/Proyectos/Delafossites/Struc_dat/dsf_TLHAF_L=120_tf=4096_T=1.0.npy')
#SF = np.load('/Users/jfmv/Documents/Proyectos/Delafossites/integration_delafossite/test_lang_'+str(L)+'.npy')

omegas=np.linspace(0,2*np.pi,n_freqs)
if L != 120:
    Radius_inscribed_hex=0.999*4*np.pi/3
else:
    Radius_inscribed_hex=1.000000000000001*4*np.pi/3

def hexagon(pos):
    x, y = map(abs, pos) #only first quadrant matters
    return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge


# n1=np.arange(-L,L+1,1)
# n2=np.arange(-L,L+1,1)

# n_1p=[]
# n_2p=[]
# for x in n1:
#     for y in n2:
#         kx=2*np.pi*x/L
#         ky=2*(2*np.pi*y/L - np.pi*x/L)/np.sqrt(3)
#         if hexagon(( kx, ky)):
#             #plt.scatter(kx_rangex[x],ky_rangey[y])
#             n_1p.append(x)
#             n_2p.append(y)

# n_1=np.array(n_1p)
# n_2=np.array(n_2p)
n_3=0#int(n_freqs/(2*np.pi))#1000
# KX=2*np.pi*n_1/L
# KY=2*(2*np.pi*n_2/L - np.pi*n_1/L)/np.sqrt(3)

# # plt.scatter(KX,KY,c=SF[n_3,:],s =1)
# # plt.colorbar()
# # plt.gca().set_aspect('equal', adjustable='box')
# # plt.show()
# #


############################################################
###denser grid for integration
############################################################



# print("starting sampling in reciprocal space....")
# s=time.time()
# LP=int(sys.argv[1])
# n1=np.arange(-LP,LP+1,1)
# n2=np.arange(-LP,LP+1,1)
# n_1,n_2=np.meshgrid(n1,n2)

# n_1p=[]
# n_2p=[]
# for x in n1:
#     for y in n2:
#         kx=2*np.pi*x/LP
#         ky=2*(2*np.pi*y/LP - np.pi*x/LP)/np.sqrt(3)
#         if hexagon(( kx, ky)):
#             #plt.scatter(kx_rangex[x],ky_rangey[y])
#             n_1p.append(x)
#             n_2p.append(y)

# KXX=2*np.pi*n_1/LP
# KYY= 2*(2*np.pi*n_2/LP - np.pi*n_1/LP)/np.sqrt(3)


# n_1pp=np.array(n_1p)
# n_2pp=np.array(n_2p)

# KX=2*np.pi*n_1pp/LP
# KY= 2*(2*np.pi*n_2pp/LP - np.pi*n_1pp/LP)/np.sqrt(3)
# e=time.time()
# print("finished sampling in reciprocal space....")
# print("time for sampling was...",e-s)

# with open("/Users/jfmv/Documents/Proyectos/Delafossites/Struc_dat/Kpoints/KgridX"+sys.argv[1]+".npy", 'wb') as f:
#     np.save(f, KX)


# with open("/Users/jfmv/Documents/Proyectos/Delafossites/Struc_dat/Kpoints/KgridY"+sys.argv[1]+".npy", 'wb') as f:
#     np.save(f, KY)


with open("/Users/jfmv/Documents/Proyectos/Delafossites/Struc_dat/Kpoints/KgridX"+sys.argv[1]+".npy", 'rb') as f:
    KX = np.load(f)


with open("/Users/jfmv/Documents/Proyectos/Delafossites/Struc_dat/Kpoints/KgridY"+sys.argv[1]+".npy", 'rb') as f:
    KY = np.load(f)




print("shape " , np.shape(KX))
print("frequency " , 2*np.pi*n_3/n_freqs )


points = np.array([KX,KY]).T
values = SF[n_3,:]
HandwavyThres=1e-9
#SF2=np.vstack( (SF, np.zeros(np.size(KX))+HandwavyThres ) )

################################
################################
################################
################################
#Defining integrand
###other Parameters

tp1=10      #568/J #in units of Js\
tp2=-tp1*108/568 #/tpp1




def Disp(kx,ky,mu):
    ed=tp1*(kx**2+ky**2)
    ed=ed-mu
    return ed

def Disp2(kx,ky,mu):
    ed=-tp1*(2*np.cos(kx)+4*np.cos((kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
    ed=ed-tp2*(2*np.cos(np.sqrt(3)*(ky))+4*np.cos(3*(kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
    ed=ed-mu
    return ed

x = np.linspace(-3.8, 3.8, 300)
X, Y = np.meshgrid(x, x)
Z = Disp2(X, Y, 0)

Wbdw=np.max(Z)-np.min(Z)


print("The bandwidth is ....", Wbdw)
mu2=-0.5*Wbdw  

EF= mu2-np.min(Z)#fermi energy from the bottom of the band
m=Wbdw/10
gamma=Wbdw/100
vmode=EF/2
T=EF/300
gcoupl=EF/10

print("The fermi energy in units of the hopping is ....", EF)
print("The fermi energy ratio to the bandwidth is ....", EF/Wbdw)
x = np.linspace(-3.8, 3.8, 300)
X, Y = np.meshgrid(x, x)
Z = Disp2(X, Y, mu2)

c= plt.contour(X, Y, Z, levels=[0],linewidths=3, cmap='summer');
v = c.collections[0].get_paths()[0].vertices
xFS2 = v[::10,0]
yFS2 = v[::10,1]
KFx2=xFS2[0]
KFy2=yFS2[0]
plt.scatter(KFx2,KFy2)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


def integrand_Disp2(qx,qy,kx,ky,w,SF2):

    ed=Disp2(kx+qx,ky+qy,mu2)
    om=w-ed
    om2=-ed

    # values_ind=[]
    # for ee in om:
    #     ind=np.argmin( abs( omegas-np.abs(ee) ) )
    #     values_ind.append(ind)


    ##getting the closest index to the sampled omegas, if it exceeds we use the threshold column added when reading the griddata
    ##if the the value corresponds to a negative valu, it does not matter
    #since we use the absolute value of w-ek assuming symmetry of the structure factor

    values =gcoupl* gamma*om/(( (vmode**2)*qx**2 +(vmode**2)*qy**2 -om**2+m**2)**2+(om*gamma)**2)
    fac_p=(1/(1+np.exp(om2/T)) + 1/(np.exp(om/T)-1))
    return values*np.pi*fac_p

w=omegas[n_3]
print(w)


siz=100
Omegs=np.linspace(0.0,2,siz)


plt.scatter(KX,KY,c=(integrand_Disp2(KX,KY,KFx2,KFy2,0,SF)), s=1)
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

ds=Vol_rec/np.size(KX)
sigm=[]




TF=EF/260
TFi=0.000001*EF
NNT=500
TSOS=np.linspace(TFi,TF,NNT)
for T in TSOS:
    start=time.time()
    SI=np.sum(integrand_Disp2(KX,KY,KFx2,KFy2,0,SF)*ds)
    #SI=np.sum(integrand_Disp(KX,KY,KFx,KFy,Omegs[i],SF)*ds)
    end=time.time()
    #print("time ",end-start)
    #print(Omegs[i],SI)
    #print(SI)
    sigm.append(SI)



###fitting 
from scipy.optimize import curve_fit
def func(x, a, b):
    return a*(x**2) +b
popt, pcov = curve_fit(func, TSOS,np.array(sigm))

plt.plot(TSOS/EF, func(TSOS, *popt)/EF, 'r-',label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
# plt.plot(TSOS,sigm[7]*TSOS**2/(TSOS**2)[7], c='r')
plt.scatter(TSOS/EF,np.array(sigm)/EF, s=3,label='m/W=%5.3f, gamma/W=%5.3f, EF/W=%5.3f' % (m/Wbdw,gamma/Wbdw,EF/Wbdw) )
# plt.plot(TSOS,sigm[7]*TSOS**2/(TSOS**2)[7], c='r')
plt.xlabel(r"$T/E_F$")
plt.ylabel(r"-Im$\Sigma (k_F,\omega)/E_F$")
plt.legend()


plt.show()

