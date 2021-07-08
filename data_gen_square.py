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

a_1=a_1[0:2]
a_2=a_2[0:2]
b_1=b_1[0:2]
b_2=b_2[0:2]

################################
################################
################################
################################
#sampling FBZ

L = 600
n_freqs = 4097
#n_freqs = 257 #welsh
omegas=np.linspace(0,1,n_freqs)
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



################################
################################
################################
################################
###########Static structure factor
#defining the parameter gamma in the static structure factor
def gamma(kx,ky):
    return 2*np.cos(kx)+4*np.cos(kx/2)*np.cos(np.sqrt(3)*ky/2)


def Sf(kx,ky,lam,T):
    return 3/(lam+(1/T)*gamma(kx,ky))

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
lam=bisection(f,3/T,150,170,T,KX,KY)
alph=alphfunc(T)
#dynamic structure fac
def Sfw(kx,ky,lam,T,ome, alph):
    fq=(gamma(kx,ky)**2)/T +gamma(kx,ky)*(lam-6/T)- 6*lam
    fq=alph*fq
    return -2*Sf(kx,ky,lam,T)*(fq/(ome**2+fq**2))

def Sfw2(kx,ky,lam,T,ome, alph):
    eps=1e-17
    gam=2*np.cos(kx)+4*np.cos(kx/2)*np.cos(np.sqrt(3)*ky/2)
    SP=3/(lam+(1/T)*gam)
    fq=(gam**2)/T +gam*(lam-6/T)- 6*lam
    fq=alph*fq
    return np.real(-2*SP*(fq/(ome**2+fq**2+1j*eps)))




################################
################################
################################
################################

###Exporting the data to file .npy


SFdat=np.array([Sfw2(KX,KY,lam,T,omega, alph) for omega in omegas])
plt.scatter(KX,KY, c=SFdat[900,:], s=1)
plt.colorbar()
plt.show()

print(np.size(KX))


Ks=np.array([KX,KY])

with open('test_lang_'+str(L)+'.npy', 'wb') as f:
    np.save(f, SFdat)
