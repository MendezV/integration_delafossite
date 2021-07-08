import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import time

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
            #plt.scatter(kx_rangex[x],ky_rangey[y])
            n_1p.append(x)
            n_2p.append(y)

n_1=np.array(n_1p)
n_2=np.array(n_2p)
n_3=4000
KX=2*np.pi*n_1/L
KY=2*(2*np.pi*n_2/L - np.pi*n_1/L)/np.sqrt(3)


values=dsf_data[n_1%L,n_2%L,n_3]
SF=np.array([dsf_data[n_1%L,n_2%L,ll] for ll in range(n_freqs)])
print(np.shape(values), np.shape(SF), "shapeval")
#plt.scatter(KX,KY,c=values,s =1)
#plt.gca().set_aspect('equal', adjustable='box')
#plt.show()


print("shape " , np.shape(KX))
print("frequency " , 2*np.pi*n_3/n_freqs )

###other Parameters
T=10
J=2*5.17
tp1=568/J #in units of J
tp2=-108/J #/tpp1

points = np.array([KX,KY]).T
values = SF[n_3,:]
HandwavyThres=1e-4
SF2=np.vstack( (SF, np.zeros(np.size(KX))+HandwavyThres ) )



def integrand_p(qx,qy,kx,ky,w,SF2,points):

    mu=0
    ed=-tp1*(2*np.cos(kx+qx)+4*np.cos((kx+qx)/2)*np.cos(np.sqrt(3)*(ky+qy)/2))
    ed=ed-tp2*(2*np.cos(np.sqrt(3)*(ky+qy))+4*np.cos(3*(kx+qx)/2)*np.cos(np.sqrt(3)*(ky+qy)/2))
    ed=ed-mu
    eps=1e-17
    om=w-ed
    om2=-ed

    values_ind=[]
    for ee in om:
        if ee<np.max(omegas):
            ind=np.argmin( abs( omegas-np.abs(ee) )**2 )
        else:
            ind=-1

        values_ind.append(ind)

    ##getting the closest index to the sampled omegas, if it exceeds we use the threshold column added when reading the griddata
    ##if the the value corresponds to a negative valu, it does not matter
    #since we use the absolute value of w-ek assuming symmetry of the structure factor
    values = SF2[np.array(values_ind),np.arange(0,int(np.size(qx)))]
    fac_p=np.exp(ed/T)*(1+np.exp(-w/T))/(1+np.exp(ed/T))
    return np.real(2*np.pi*fac_p*values)









##########
L = 600
n_freqs = 4097

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



SFi_l=[]

for i in range(n_freqs):
    start=time.time()
    SFi_l.append(griddata(points, SF[i,:], (KXp,KYp) , method='cubic'))
    end=time.time()
    print("time ",end-start)
    print(i,"/",n_freqs)

SFi= np.array(SFi_l)
SFi2=np.vstack( (SFi, np.zeros(np.size(KXp))+HandwavyThres ) )

print(np.shape(SF),np.shape(SF2))
print(np.shape(SFi),np.shape(SFi2))

"""
plt.scatter(KXp,KYp,c=SFi,s =1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
"""
w=omegas[n_3]
print(w)

plt.scatter(KX,KY,c=(integrand_p(KX,KY,0.3,0.3,w,SF2,points)),s =1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

plt.scatter(KXp,KYp,c=(integrand_p(KXp,KYp,0.3,0.3,w,SFi2,points)),s =1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
