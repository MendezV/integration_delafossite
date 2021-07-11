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
SF = np.load('/Users/jfmv/Documents/Proyectos/Delafossites/integration_delafossite/test'+str(L)+'.npy')
#SF = np.load('/Users/jfmv/Documents/Proyectos/Delafossites/integration_delafossite/test_lang_'+str(L)+'.npy')

omegas=np.linspace(0,2*np.pi,n_freqs)
if L != 120:
    Radius_inscribed_hex=0.999*4*np.pi/3
else:
    Radius_inscribed_hex=1.000000000000001*4*np.pi/3

def hexagon(pos):
    x, y = map(abs, pos) #only first quadrant matters
    return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge


n1=np.arange(-L,L+1,1)
n2=np.arange(-L,L+1,1)

n_1p=[]
n_2p=[]
for x in n1:
    for y in n2:
        kx=2*np.pi*x/L
        ky=2*(2*np.pi*y/L - np.pi*x/L)/np.sqrt(3)
        if hexagon(( kx, ky)):
            #plt.scatter(kx_rangex[x],ky_rangey[y])
            n_1p.append(x)
            n_2p.append(y)

n_1=np.array(n_1p)
n_2=np.array(n_2p)
n_3=1000
KX=2*np.pi*n_1/L
KY=2*(2*np.pi*n_2/L - np.pi*n_1/L)/np.sqrt(3)

#
# plt.scatter(KX,KY,c=np.log10(SF[0,:]),s =1)
# plt.colorbar()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()
#
#
# q=np.sqrt(KX**2+KY**2)
# """
#
# ind=np.where(q<1e-10)[0]
# q=np.delete(q,ind)
# KX=np.delete(KX,ind)
# KY=np.delete(KY,ind)
# """
# plt.scatter(KX,KY,c=np.log10(10/q**2),s =1)
# plt.colorbar()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()
#
#
# ind=np.where(abs(KX+KY)<1e-10)[0]
# print(ind)
# KX=np.delete(KX,ind)
# KY=np.delete(KY,ind)
# SF2=np.delete(SF[0,:],ind)
# plt.scatter(KX,KY,c=np.log10(SF2),s =1)
# plt.colorbar()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

print("shape " , np.shape(KX))
print("frequency " , 2*np.pi*n_3/n_freqs )


points = np.array([KX,KY]).T
values = SF[n_3,:]
HandwavyThres=1e-4
SF2=np.vstack( (SF, np.zeros(np.size(KX))+HandwavyThres ) )

################################
################################
################################
################################
#Defining integrand
###other Parameters

T=1.0
J=2*5.17
tp1=568/J #in units of J
tp2=-108/J #/tpp1

mu=0
def Disp(kx,ky,mu):
    ed=-tp1*(2*np.cos(kx)+4*np.cos((kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
    ed=ed-tp2*(2*np.cos(np.sqrt(3)*(ky))+4*np.cos(3*(kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
    ed=ed-mu
    return ed


x = np.linspace(-3.8, 3.8, 300)
X, Y = np.meshgrid(x, x)
Z = Disp(X, Y, mu)

c= plt.contour(X, Y, Z, levels=[0],linewidths=3, cmap='summer');
v = c.collections[0].get_paths()[0].vertices
xFS = v[::10,0]
yFS = v[::10,1]
KFx=xFS[60]
KFy=yFS[60]
plt.scatter(KFx,KFy)
plt.show()


def integrand_Disp(qx,qy,kx,ky,w,SF2):

    ed=Disp(kx+qx,ky+qy,mu)
    om=w-ed
    om2=-ed

    values_ind=[]
    for ee in om:
        ind=np.argmin( abs( omegas-np.abs(ee) ) )
        values_ind.append(ind)


    ##getting the closest index to the sampled omegas, if it exceeds we use the threshold column added when reading the griddata
    ##if the the value corresponds to a negative valu, it does not matter
    #since we use the absolute value of w-ek assuming symmetry of the structure factor
    values = SF2[np.array(values_ind),np.arange(0,int(np.size(qx)))]
    fac_p=np.exp(ed/T)*(1+np.exp(-w/T))/(1+np.exp(ed/T))
    return values*2*np.pi*fac_p




w=omegas[n_3]
print(w)


siz=50
Omegs=np.linspace(0 ,6 ,siz)


SS=integrand_Disp(KX,KY,KFx,KFy,0,SF)
ind=np.where(abs(KX+KY)<1e-10)[0]
KX2=np.delete(KX,ind)
KY2=np.delete(KY,ind)
SS2=np.delete(SS,ind)
plt.scatter(KX2,KY2,c=np.log10(SS2+1e-11),s =1)
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# plt.scatter(KX,KY,c=np.log10(integrand_Disp(KX,KY,KFx,KFy,0,SF)+1e-11),s =1)
# plt.colorbar()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

ds=Vol_rec/np.size(KX)
sigm=[]
S0=0#np.sum(integrand_Disp(KX,KY,KFx,KFy,0,SF)*ds)
print("Method1,...T=",T, ds)

for i in range(siz):
    start=time.time()
    if (Omegs[i]!=0):
        SI=np.sum(integrand_Disp(KX,KY,KFx,KFy,Omegs[i],SF)*ds)-S0
    else:

        SS=integrand_Disp(KX,KY,KFx,KFy,Omegs[i],SF)*ds
        ind=np.where(abs(KX+KY)<1e-10)[0]
        KX=np.delete(KX,ind)
        KY=np.delete(KY,ind)
        SS=np.delete(SS,ind)
        SI=np.sum(SS)-S0
    #SI=np.sum(integrand_Disp(KX,KY,KFx,KFy,Omegs[i],SF)*ds)-S0
    end=time.time()
    print("time ",end-start)
    print(i,SI)
    sigm.append(SI)
plt.plot(Omegs,sigm, 'o', label="T="+str(T))
plt.xlabel(r"$\omega$")
plt.ylabel(r"-Im$\Sigma (k_F,\omega)$")
plt.legend()
plt.show()

# for i in range(n_freqs):
#     start=time.time()
#     #SI=np.sum(integrand_Disp(KX,KY,KFx,KFy,Omegs[i],SF)*ds)-S0
#     SI=np.sum(SF[i,:]*ds)
#     end=time.time()
#     print("time ",end-start)
#     print(i,SI)
#     sigm.append(SI)
#
#
# plt.plot(omegas,sigm, 'o', label="T="+str(T))
# plt.xlabel(r"$\omega$")
# plt.ylabel(r"$S(\omega)$")
# plt.legend()
# plt.show()
