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

L = 120
n_freqs = 4097

# Load the data files
dsf_data = np.load('/Users/jfmv/Documents/Proyectos/Delafossites/Struc_dat/dsf_TLHAF_L=120_tf=4096_T=1.0.npy')

##calculating the integrand in preparation for the MC integration
omegas=np.linspace(0,2*np.pi,n_freqs)

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
            n_1p.append(x)
            n_2p.append(y)

n_1=np.array(n_1p)
n_2=np.array(n_2p)
n_3=900
KX=2*np.pi*n_1/L
KY=2*(2*np.pi*n_2/L - np.pi*n_1/L)/np.sqrt(3)


values=dsf_data[n_1%L,n_2%L,n_3]
SF=np.array([dsf_data[n_1%L,n_2%L,ll] for ll in range(n_freqs)])



points = np.array([KX,KY]).T
values = SF[n_3,:]
HandwavyThres=1e-4
SF2=np.vstack( (SF, np.zeros(np.size(KX))+HandwavyThres ) )

with open('test'+str(L)+'.npy', 'wb') as f:
    np.save(f, SF2)


# plt.scatter(KX,KY,c=SF[n_3,:],s =1)
# plt.colorbar()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()



################################
################################
################################
################################
#saving the interpolated array


L = int(sys.argv[1])
n_freqs = 4097

n1=np.arange(-L,L+1,1)
n2=np.arange(-L,L+1,1)

Radius_inscribed_hex=0.999*4*np.pi/3

def hexagon(pos):
    x, y = map(abs, pos) #only first quadrant matters
    return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge


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
KXp=2*np.pi*n_1/L
KYp=2*(2*np.pi*n_2/L - np.pi*n_1/L)/np.sqrt(3)


#for interpolating with gigantic grid
SFi_l=[]
s=np.size(KX)
for i in range(n_freqs):
    start=time.time()
    SFi_l.append(griddata(points, SF[i,:], (KXp,KYp) , method='cubic'))
    end=time.time()
    print("time ",end-start)
    print(i,"/",n_freqs)
    for j in range(s):
        if np.isnan(SFi_l[i][j]):
            print(i,j,SFi_l[i][j])

SFi= np.array(SFi_l)
SFi2=np.vstack( (SFi, np.zeros(np.size(KXp))+HandwavyThres ) )


with open('test'+str(L)+'.npy', 'wb') as f:
    np.save(f, SFi2)


plt.scatter(KXp,KYp,c=SFi[n_3,:],s =1)
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
