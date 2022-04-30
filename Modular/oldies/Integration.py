import numpy as np
import time
import matplotlib.pyplot as plt
 
def trapzn(f, a , b, n, args):
    x=np.linspace(a,b,n+1)
    delt=(b-a)/n
    return np.trapz(f(x,args))*delt

def qsimp(f, a , b, args):
    JMAX=10000
    JMIN=1000
    s=0
    st=0
    os=0
    ost=0
    EPS=1e-6
    for n in range(JMIN,JMAX):
        st=trapzn(f, a , b, n, args)
        s=(4.0*st-ost)/3.0
        if (np.abs(s-os)<EPS*np.abs(os)) or (s==0 and os==0):
            return s
        
        os=s
        ost=st
    
    return("error insuficient steps")



def f(x,args):
    [t,u]=args
    return np.cos(x)

a=-np.pi/2.0
b=np.pi/2.0
args=[0,1]
n=1000

s=time.time()
print(trapzn(f,a,b,n,args))
e=time.time()
print("time, ",e-s)

s=time.time()
print(qsimp(f,a,b,args))
e=time.time()
print("time, ",e-s)



def trapzn2x(f, x, a , b, n, args):
    y=np.linspace(a,b,n+1)
    delt=(b-a)/n
    return np.trapz(f(x,y,args))*delt

def qsimp2(f, x, a , b, args):
    JMAX=10000
    JMIN=1000
    s=0
    st=0
    os=0
    ost=0
    EPS=1e-6
    for n in range(JMIN,JMAX):
        st=trapzn2x(f, x, a , b, n, args)
        s=(4.0*st-ost)/3.0
        if (np.abs(s-os)<EPS*np.abs(os)) or (s==0 and os==0):
            return s
        
        os=s
        ost=st
    
    return("error insuficient steps")




# def f1(x,args):
#     return qsimp2(f, x,  -m*(x+1) , m*(x+1), args)

# def f2(x,args):
#     return qsimp2(f, x, -r , r, args)

# def f3(x,args):
#     return qsimp2(f, x, m*(x-1) , -m*(x-1), args)

# def int_hex(args):
#     a1=-t
#     b1=-t/2
#     a2=-t/2
#     b2=t/2
#     a3=t/2
#     b3=t
#     I1=qsimp(f1, a1 , b1, args)
#     # I2=qsimp(f2, a2 , b2, args)
#     # I3=qsimp(f3, a3 , b3, args)
#     # return I1+I2+I3
#     return I1

# print(f1(0,args))
# print(f1(np.array([0,1]),args))


r=np.sqrt(3)/2;
t=1;
u=np.sqrt(t**2-r**2);
m=(r-0)/(-(t/2)-(-t));

def f(y,x,args):
    return (np.heaviside(x,1)+np.heaviside(-x,0))*(np.heaviside(y,1)+np.heaviside(-y,0))

# f = lambda y, x: 1
from scipy import integrate
I1=integrate.dblquad(f, -t, -t/2, lambda x: -m*(x+1), lambda x: m*(x+1), args=[1],epsabs=1.49e-08, epsrel=1.49e-08)
I2=integrate.dblquad(f, -t/2, t/2, lambda x: -r, lambda x: r, args=[1],epsabs=1.49e-08, epsrel=1.49e-08)
I3=integrate.dblquad(f, t/2, t, lambda x: m*(x-1), lambda x: -m*(x-1), args=[1],epsabs=1.49e-08, epsrel=1.49e-08)

print(I1[0]+I2[0]+I3[0], np.sqrt(3)*3/2)