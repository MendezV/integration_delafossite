import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import time

L = 120
n_freqs = 4097

# Load the data files
dsf_data = np.load('/Users/jfmv/Documents/Proyectos/Delafossites/Struc_dat/dsf_TLHAF_L=120_tf=4096_T=1.0.npy')

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
n_3=4000
KX=2*np.pi*n_1/L
KY=2*(2*np.pi*n_2/L - np.pi*n_1/L)/np.sqrt(3)
values=dsf_data[n_1%L,n_2%L,n_3]

#plt.scatter(KX,KY,c=values,s =1)
#plt.gca().set_aspect('equal', adjustable='box')
#plt.show()



L = 600
n_freqs = 4097


Radius_inscribed_hex=1.000000000000001*4*np.pi/3

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
KXp=2*np.pi*n_1/L
KYp=2*(2*np.pi*n_2/L - np.pi*n_1/L)/np.sqrt(3)


start=time.time()
SF=griddata(np.array([KX,KY]).T, values, (KXp,KYp) , method='cubic')
end=time.time()
print("time ",end-start)
"""
plt.scatter(KXp,KYp,c=SF,s =1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
"""

print("shape " , np.shape(KX))
print("frequency " , 2*np.pi*n_3/n_freqs )
