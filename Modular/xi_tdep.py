import numpy as np
import Lattice
import StructureFactor
import Dispersion
import matplotlib.pyplot as plt
import time
import sys
from scipy import integrate
import concurrent.futures
import functools
from traceback import print_exc
import os
from datetime import datetime
import gc
import pandas as pd



def main() -> int:

    J=2*5.17 #in mev
    tp1=568/J #in units of Js\
    tp2=-tp1*108/568 #/tpp1
    U=4000/J
    g=100/J
    Kcou=g*g/U
    fill=0.5
    
    def lor(x,al,xi):
        return (1/al)*(3/(1+(xi*x)**2))
    
    
    Npoints=1000
    Npoints_int_pre, NpointsFS_pre=1000,600
    save=True
    l=Lattice.TriangLattice(Npoints_int_pre, save,'MAC')
    Vol_rec=l.Vol_BZ()
    
    T=1
    
    SS=StructureFactor.StructureFac_fit_F(T)
    [KX,KY]=l.read_lattice()
    
    # plt.scatter(KX,KY, c=SS.Static_SF(KX,KY) )
    # plt.colorbar()
    # plt.show()
    xilist=[]
    Ts=np.arange(1,11,1)
    for T in [1,5]:
        SS=StructureFactor.StructureFac_fit_F(T)
        alpha=SS.lam-3/T
        xi=np.sqrt(3/(4*T*alpha))
        
        
        K=4*np.pi/3
        nu=1
        x=np.linspace(K*(1-nu), K*(1+nu), 100)

        plt.scatter(x/K,SS.Static_SF(x,0) )
        plt.scatter(x/K,lor(x-K,alpha,xi), label=str(xi) )
        plt.legend()
        
        
        # kx,ky=np.meshgrid(x,x)
        # plt.scatter(kx,ky, c=SS.Static_SF(kx,ky))
        # plt.colorbar()
        # plt.show()
        

        plt.savefig("lorentzian_fits"+str(T)+".png")
        plt.close()
    
    xilist=[]
    for T in Ts:
        SS=StructureFactor.StructureFac_fit_F(T)
        alpha=SS.lam-3/T
        xi=np.sqrt(3/(4*T*alpha))
        
        
        xilist.append(xi)
    print(xilist)
    plt.plot(Ts, xilist)
    plt.scatter(Ts, xilist)
    plt.ylabel(r'$\xi$')
    plt.xlabel('T/J')
    plt.savefig("corrlength"+str(T)+".png")
    plt.close()
    
    
    plt.plot(Ts, 1/np.array(xilist))
    plt.scatter(Ts, 1/np.array(xilist))
    plt.ylabel(r'$\xi$')
    plt.xlabel('T/J')
    plt.savefig("invcorrlength"+str(T)+".png")
    plt.close()
    
    
    

    
    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
