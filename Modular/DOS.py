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

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

class SelfE():

    def __init__(self, T ,ed ,SS,  Npoints_int_pre, NpointsFS_pre ,Kcou, type):
        self.T=T
        self.ed=ed #dispersion
        self.SS=SS #structure factor
        save=False
        self.Kcou=Kcou
        self.Npoints_int_pre=Npoints_int_pre

        [qxFS,qyFS]=ed.FS_contour( NpointsFS_pre)
        self.qxFS=qxFS
        self.qyFS=qyFS

        self.NpointsFS=np.size(qxFS)
        if type=="mc":
            self.latt=Lattice.TriangLattice(Npoints_int_pre, save ) #integration lattice 
            [self.kx,self.ky]=self.latt.read_lattice()
            [self.kxsq,self.kysq]=self.latt.read_lattice(sq=1)

        if type=="hex":
            self.latt=Lattice.TriangLattice(Npoints_int_pre, save ) #integration lattice 
            [self.kx,self.ky]=self.latt.read_lattice()
            [self.kxsq,self.kysq]=self.latt.read_lattice(sq=1)

        if type=="sq":
            self.latt=Lattice.SQLattice(Npoints_int_pre, save ) #integration lattice 
            [self.kx,self.ky]=self.latt.read_lattice()
            [self.kxsq,self.kysq]=self.latt.read_lattice()

    def __repr__(self):
        return "Structure factorat T={T}".format(T=self.T)


    ###################
    # INTEGRANDS FOR PARALLEL RUNS
    ###################



    def integrand_par_w(self,qp,ds,w):
        si=time.time()
        qx,qy=qp[0], qp[1]

        edd=self.ed.Disp_mu(self.kxsq+qx,self.kysq+qy)
        om=w-edd
        fac_p=(1+np.exp(-w/self.T))*(1-self.ed.nf(edd, self.T))
        del edd
        gc.collect()

        SFvar=self.SS.Dynamical_SF(self.kxsq,self.kysq,om)

        
        # fac_p=ed.nb(w-edd, T)+ed.nf(-edd, T)
        Integrand=self.Kcou*self.Kcou*SFvar*2*np.pi*fac_p
        del SFvar
        gc.collect()

        # ##removing the point at the origin
        # ind=np.where(np.abs(kx)+np.abs(ky)<np.sqrt(ds))[0]
        # Integrand=np.delete(Integrand,ind)
        S0=np.sum(Integrand*ds)
        # Vol_rec=self.latt.Vol_BZ()
        dels=10*ds*np.max(np.abs(np.diff(Integrand)))#np.sqrt(ds/Vol_rec)*Vol_rec#*np.max(np.abs(np.diff(Integrand)))*0.1
        del Integrand
        gc.collect()
        ang=np.arctan2(qy,qx)
        ei=time.time()
        print(ei-si," seconds ",qx, qy, w)

        return S0, w,dels


    ###################
    # PLOTTING ROUTINES
    ###################

    def plot_integrand(self,qx,qy,f):
        Vertices_list, Gamma, K, Kp, M, Mp=self.latt.FBZ_points(self.latt.b[0,:],self.latt.b[1,:])
        VV=np.array(Vertices_list+[Vertices_list[0]])
        Integrand=self.integrand(self.kx,self.ky,qx,qy,f)
        print("for error, maximum difference", np.max(np.diff(Integrand)))
        plt.plot(VV[:,0], VV[:,1], c='k')
        plt.scatter(self.kx,self.ky,c=Integrand, s=1)
        plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(f"integrand_{qx}_{qy}_{f}_q.png")
        
        plt.close()
        # plt.show()
        return 0

    def plot_logintegrand(self,qx,qy,f):
        Vertices_list, Gamma, K, Kp, M, Mp=self.latt.FBZ_points(self.latt.b[0,:],self.latt.b[1,:])
        VV=np.array(Vertices_list+[Vertices_list[0]])
        Integrand=self.integrand(self.kx,self.ky,qx,qy,f)
        print("for error, maximum difference", np.max(np.diff(Integrand)))
        plt.plot(VV[:,0], VV[:,1], c='k')
        xx=np.log10(Integrand)
        wh=np.where(xx>-10)
        plt.scatter(self.kx[wh],self.ky[wh],c=xx[wh], s=1)
        # plt.clim(-2,np.max(np.log10(Integrand)))
        plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(f"log_integrand_{qx}_{qy}_{f}_q.png", dpi=400)
        plt.close()
        # plt.show()
        return 0
    
    ##############
    #   random routines
    ##############
    def get_KF(self, theta):
        angles=[]
        
        for ell in range(self.NpointsFS):

            qx=self.qxFS[ell]
            qy=self.qyFS[ell]

            angles.append(np.arctan2(qy,qx))

        itheta=np.argmin((np.array(angles)-theta)**2)
        qx=self.qxFS[itheta]
        qy=self.qyFS[itheta]
        
        return [qx,qy]

    ##############
    #   PARALLEL RUNS WITH FREQUENCY DEPENDENCE
    ##############


    def parInt_w(self, qx, qy, w, sq, maxthreads):
        

        # if sq==True:
        #     kx=self.kxsq
        #     ky=self.kysq
        # else:
        #     kx=self.kx
        #     ky=self.ky
        ###determining the momenta on the FS
        shifts=[]
        delsd=[]
        
    
    
        print("energy at the point.." , qx, qy, " is " ,self.ed.Disp_mu(qx,qy))
        qp=np.array([qx,qy]).T

        Vol_rec=self.latt.Vol_BZ()
        Npoints_int=np.size(self.kxsq)
        ds=Vol_rec/Npoints_int

        Npoints_w=np.size(w)
        print(Npoints_w, "the points",int(Npoints_w/maxthreads),"chunk numtheads")
        Nthreads=int(Npoints_w/maxthreads)

        partial_integ = functools.partial(self.integrand_par_w, qp, ds)

        print("starting with calculation of Sigma")
        s=time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(partial_integ, w, chunksize=Nthreads)

            for result in results:
                shifts.append(result[0])
                delsd.append(result[2])

        e=time.time()
        print("time for calc....",e-s)

        shifts=np.array(shifts) 
        delsd=np.array(delsd)

        return [shifts, w, delsd]





    ############
    # OUTPUT
    ######


    def gen_df(self, arg, J, theta, fill, tp1,tp2, prefixd):

        [qx,qy]=self.get_KF(theta)        
        dispname=self.ed.name
        SFname=self.SS.name+"_theta_"+str(round(theta*180/np.pi, 2))
        [shifts, w, delsd]=arg
        SEarr=np.array(shifts)
        err_arr=np.array(delsd)
        
        
        if prefixd!="":
            df = pd.DataFrame({'theta': theta, "freq":w , 'SE':SEarr, 'error': err_arr, 'KFX': qx, 'KFY': qy, 'T': self.T, \
                'nu': fill,'intP':self.Npoints_int_pre, 'FS_point': self.NpointsFS, 'dispname': dispname, "t1":tp1, "t2":tp2, 'SFname': SFname, 'J':J, 'extr':prefixd})
            
        else:
            
            df = pd.DataFrame({'theta': theta, "freq":w , 'SE':SEarr, 'error': err_arr, 'KFX': qx, 'KFY': qy, 'T': self.T, \
                'nu': fill,'intP':self.Npoints_int_pre, 'FS_point': self.NpointsFS, 'dispname': dispname, "t1":tp1, "t2":tp2, 'SFname': SFname, 'J':J})
        return df
        
        



def main() -> int:
    
    
    try:
        index_sf=int(sys.argv[1])

    except (ValueError, IndexError):
        raise Exception("Input integer in the firs argument to choose structure factor")


    try:
        N_SFs=11 #number of SF's currently implemented
        a=np.arange(N_SFs)
        a[index_sf]

    except (IndexError):
        raise Exception(f"Index has to be between 0 and {N_SFs-1}")


    try:
        mod=float(sys.argv[2])

    except (ValueError, IndexError):
        raise Exception("Input float in the second argument to scale a given quantity")


    try:
        T=float(sys.argv[3])


    except (ValueError, IndexError):
        raise Exception("Input float in the third argument is the temperature")
        
    try:
        Machine=sys.argv[4]
        
        if Machine=='FMAC':
            maxthreads=8
        elif Machine=='CH1':
            maxthreads=20
        elif Machine=='UBU':
            maxthreads=12
        else:
            maxthreads=6


    except (ValueError, IndexError):
        raise Exception("Input string in the fourth argument is the machine to run,\n this affects optimization in parallel integration routines current options are: CH1 FMAC UBU")




    ##########################
    ##########################
    # parameters
    ##########################
    ##########################

    # # #electronic parameters
    J=2*5.17 #in mev
    tp1=568/J #in units of Js\
    tp2=-tp1*108/568 #/tpp1
    ##coupling 
    U=4000/J
    g=100/J
    Kcou=g*g/U
    # fill=0.67 #van hove
    fill=0.5
    

    #rotated FS parameters
    # J=2*5.17 #in mev
    # tp1=568/J #in units of Js\
    # tp2=tp1*0.258 #/tpp1
    # ##coupling 
    # U=4000/J
    # g=100/J
    # Kcou=g*g/U
    # # fill=0.67 #van hove
    # fill=0.35

    ##params quasicircular and circular FS
    # J=2*5.17 #in mev
    # tp1=568/J #in units of Js
    # tp2=0.065*tp1
    # ##coupling 
    # U=4000/J
    # g=100/J
    # Kcou=g*g/U
    # fill=0.1

    ##########################
    ##########################
    # Geometry/Lattice
    ##########################
    ##########################

    Npoints=1000
    Npoints_int_pre, NpointsFS_pre=6000,600
    save=True
    l=Lattice.TriangLattice(Npoints_int_pre, save)
    [KX,KY]=l.read_lattice(sq=1)
    # [KX,KY]=l.Generate_lattice_SQ()
    Vol_rec=l.Vol_BZ()
    l2=Lattice.SQLattice(Npoints, save)
    [KX2,KY2]=l2.Generate_lattice()
    Vol_rec2=l2.Vol_BZ()
    
    
    
    # ##########################
    # ##########################
    # # Fermi surface and structure factor
    # ##########################
    # ##########################

    ed=Dispersion.Dispersion_TB_single_band([tp1,tp2],fill)
    plt.plot(ed.nn,ed.Dos, c='k')
    # plt.plot(ed.dens2,ed.valt)
    plt.axvline(0.1, c='b')
    plt.axvline(0.2, c='orange')
    plt.axvline(0.3, c='g')
    plt.axvline(0.5, c='r')
    plt.ylabel(r"$\rho$")
    plt.xlabel(r"$\nu$")
    plt.savefig("Dos.png")
    plt.close()
    
    plt.plot(ed.earr,ed.Dos, c='k')
    
    indemin=np.argmin((ed.nn-0.25)**2)
    mu=ed.earr[indemin]
    plt.axvline(mu, c='r')
    plt.savefig("Dos_E.png")
    plt.close()
    
    
    # [dens2,bins,valt,f2 ]=ed.DOS_2(1200)
    # plt.plot(bins,valt, c='k')
    # plt.axvline(mu, c='r')
    # plt.savefig("Dos_E.png")
    # plt.close()
    
    
    
    
    
    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
