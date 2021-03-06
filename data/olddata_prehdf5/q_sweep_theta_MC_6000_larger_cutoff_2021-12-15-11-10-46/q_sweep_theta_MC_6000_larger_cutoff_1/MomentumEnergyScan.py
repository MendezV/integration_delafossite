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
    # INTEGRANDS
    ###################
    

    def integrand_de(self,kx,ky,qx,qy):
        edd=self.ed.Disp_mu(kx+qx,ky+qy)
        epsil=0.002*self.ed.bandwidth
        inn=self.ed.deltad(edd, epsil)
        # om2=-ed

        return inn

    def integrand(self,kx,ky,qx,qy,w):

        edd=self.ed.Disp_mu(kx+qx,ky+qy)
        om=w-edd

        SFvar=self.SS.Dynamical_SF(kx,ky,om)

        fac_p=(1+np.exp(-w/self.T))*(1-self.ed.nf(edd, self.T))
        # fac_p=ed.nb(w-edd, T)+ed.nf(-edd, T)
        return self.Kcou*self.Kcou*SFvar*2*np.pi*fac_p
    
    ###################
    # INTEGRANDS FOR PARALLEL RUNS
    ###################

    def integrand_preparsum(self,kx,ky,qx,qy,w):

        edd=self.ed.Disp_mu(kx+qx,ky+qy)
        om=w-edd

        SFvar=self.SS.Dynamical_SF(kx,ky,om)

        fac_p=(1+np.exp(-w/self.T))*(1-self.ed.nf(edd, self.T))
        # fac_p=ed.nb(w-edd, T)+ed.nf(-edd, T)
        return np.sum(self.Kcou*self.Kcou*SFvar*2*np.pi*fac_p)


    def integrand_parsum(self,kx,ky,qx,qy,w,Machine):
        
        if Machine=='FMAC':
            workers=10
        elif Machine=='CH1':
            workers=206
        elif Machine=='UBU':
            workers=20
        else:
            workers=10

        Npoints_int=np.size(kx)
        Vol_rec=self.latt.Vol_BZ()

        ds=Vol_rec/Npoints_int
        chunk=Npoints_int// workers

        print("starting with calculation of Sigma")
        s=time.time()
        futures = []
        integ=0

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for i in range(workers):
                cstart = chunk * i
                cstop = chunk * (i + 1) if i != workers - 1 else Npoints_int
                # futures.append(executor.submit(partial_integ, qp[cstart:cstop]))
                futures.append(executor.submit(self.integrand_preparsum, kx[cstart:cstop],ky[cstart:cstop],qx,qy, w))
                

            # 2.2. Instruct workers to process results as they come, when all are
            #      completed or .....
            concurrent.futures.as_completed(futures) # faster than cf.wait()
            # concurrent.futures.wait(fs=1000)
            # 2.3. Consolidate result as a list and return this list.
            for f in futures:
                try:
                    integ=integ+f.result()
                except:
                    print_exc()
            end = time.time() - s
            print("found {0} in {1:.4f}sec".format(end, end))
        return [integ*ds, ds]

    def integrand_par(self,kx,ky,w,ds,qp):
        si=time.time()
        qx,qy=qp[0], qp[1]

        edd=self.ed.Disp_mu(kx+qx,ky+qy)
        om=w-edd

        SFvar=self.SS.Dynamical_SF(kx,ky,om)

        fac_p=(1+np.exp(-w/self.T))*(1-self.ed.nf(edd, self.T))
        # fac_p=ed.nb(w-edd, T)+ed.nf(-edd, T)
        Integrand=self.Kcou*self.Kcou*SFvar*2*np.pi*fac_p

        # ##removing the point at the origin
        # ind=np.where(np.abs(kx)+np.abs(ky)<np.sqrt(ds))[0]
        # Integrand=np.delete(Integrand,ind)
        S0=np.sum(Integrand*ds)

        # Vol_rec=self.latt.Vol_BZ()
        dels=10*ds*np.max(np.abs(np.diff(Integrand)))#np.sqrt(ds/Vol_rec)*Vol_rec#*np.max(np.abs(np.diff(Integrand)))*0.1
        ang=np.arctan2(qy,qx)
        ei=time.time()
        print(ei-si," seconds ",qx, qy)

        return S0, ang,dels


    #####MONTE CARLO ROUTINES#####

    def MCSAMPF(self,kx,ky,omega, qx,qy):
        ss=10
        q2=qx**2 + qy**2
        return np.exp( -(self.ed.Disp_mu(kx+qx,ky+qy)-omega)**2/(2*ss*ss)   )


    def MC_points(self,omega, qx,qy ):
        print("starting with calculation samplinggg")
        s=time.time()
        x_walk = [] #this is an empty list to keep all the steps
        y_walk = [] #this is an empty list to keep all the steps
        x_0 = qx #this is the initialization
        y_0 = qy #this is the initialization
        x_walk.append(x_0)
        y_walk.append(y_0)
        # print(x_walk,y_walk)
        stepsz= 0.17


        n_iterations = 10000#000
        for i in range(n_iterations):
            
            x_prime = np.random.normal(x_walk[i],stepsz) #0.1 is the sigma in the normal distribution
            y_prime = np.random.normal(y_walk[i], stepsz) #0.1 is the sigma in the normal distribution
            alpha = self.MCSAMPF(x_prime,y_prime,omega, qx,qy)/self.MCSAMPF(x_walk[i],y_walk[i],omega, qx,qy)
            if(alpha>=1.0):
                x_walk.append(x_prime)
                y_walk.append(y_prime)
            else:
                beta = np.random.random()
                if(beta<=alpha):
                    x_walk.append(x_prime)
                    y_walk.append(y_prime)
                else:
                    x_walk.append(x_walk[i])
                    y_walk.append(y_walk[i])
                    
        x_0 = x_walk[-1]
        y_0 = y_walk[-1]
        x_walk = [] #this is an empty list to keep all the steps
        y_walk = [] #this is an empty list to keep all the steps
        x_walk.append(x_0)
        y_walk.append(y_0)        
        n_iterations = self.Npoints_int_pre * self.Npoints_int_pre * 60  #this is the number of iterations I want to make
        rat=0
        for i in range(n_iterations):
            
            x_prime = np.random.normal(x_walk[i], stepsz) #0.1 is the sigma in the normal distribution
            y_prime = np.random.normal(y_walk[i], stepsz) #0.1 is the sigma in the normal distribution
            alpha = self.MCSAMPF(x_prime,y_prime,omega, qx,qy)/self.MCSAMPF(x_walk[i],y_walk[i],omega, qx,qy)
            if(alpha>=1.0):
                x_walk.append(x_prime)
                y_walk.append(y_prime)
                rat=rat+1
            else:
                beta = np.random.random()
                if(beta<=alpha):
                    x_walk.append(x_prime)
                    y_walk.append(y_prime)
                    rat=rat+1
                else:
                    x_walk.append(x_walk[i])
                    y_walk.append(y_walk[i])

        print("the acceptance ratio for the MC walk was ..." , rat/n_iterations)

        plt.scatter(x_walk,y_walk,s=1)
        plt.savefig("samp.png")
        plt.close()
        e=time.time()
        print("time for sampling....",e-s)
        
        return np.array(x_walk),np.array(y_walk)
    
    def MC_points_par(self,omega, qx,qy, n_iterationss ):
        
        print("starting with calculation sampling")
        s=time.time()
        x_walk = [] #this is an empty list to keep all the steps
        y_walk = [] #this is an empty list to keep all the steps
        x_0 = np.random.normal(0,0.2) #this is the initialization
        y_0 = np.random.normal(0,0.2) #this is the initialization
        x_walk.append(x_0)
        y_walk.append(y_0)
        # print(x_walk,y_walk)
        stepsz= 0.17


        n_iterations = 1000000
        for i in range(n_iterations):
            
            x_prime = np.random.normal(x_walk[i],stepsz) #0.1 is the sigma in the normal distribution
            y_prime = np.random.normal(y_walk[i], stepsz) #0.1 is the sigma in the normal distribution
            alpha = self.MCSAMPF(x_prime,y_prime,omega, qx,qy)/self.MCSAMPF(x_walk[i],y_walk[i],omega, qx,qy)
            if(alpha>=1.0):
                x_walk.append(x_prime)
                y_walk.append(y_prime)
            else:
                beta = np.random.random()
                if(beta<=alpha):
                    x_walk.append(x_prime)
                    y_walk.append(y_prime)
                else:
                    x_walk.append(x_walk[i])
                    y_walk.append(y_walk[i])
                    
        x_0 = x_walk[-1]
        y_0 = y_walk[-1]
        x_walk = [] #this is an empty list to keep all the steps
        y_walk = [] #this is an empty list to keep all the steps
        x_walk.append(x_0)
        y_walk.append(y_0)        
        n_iterations=int(np.sum(n_iterationss))  #this is the number of iterations I want to make
        rat=0
        for i in range(n_iterations):
            
            x_prime = np.random.normal(x_walk[i], stepsz) #0.1 is the sigma in the normal distribution
            y_prime = np.random.normal(y_walk[i], stepsz) #0.1 is the sigma in the normal distribution
            alpha = self.MCSAMPF(x_prime,y_prime,omega, qx,qy)/self.MCSAMPF(x_walk[i],y_walk[i],omega, qx,qy)
            if(alpha>=1.0):
                x_walk.append(x_prime)
                y_walk.append(y_prime)
                rat=rat+1
            else:
                beta = np.random.random()
                if(beta<=alpha):
                    x_walk.append(x_prime)
                    y_walk.append(y_prime)
                    rat=rat+1
                else:
                    x_walk.append(x_walk[i])
                    y_walk.append(y_walk[i])

        print("the acceptance ratio for the MC walk was ..." , rat/n_iterations)
        return x_walk,y_walk
    


    def integrand_par_MC(self,kxp,kyp,w, norm,qp):
        si=time.time()
        qx,qy=qp[0], qp[1]

        kx=kxp-qx
        ky=kyp-qy

        edd=self.ed.Disp_mu(kx+qx,ky+qy)
        om=w-edd

        SFvar=self.SS.Dynamical_SF(kx,ky,om)
        fac_p=(1+np.exp(-w/self.T))*(1-self.ed.nf(edd, self.T))
        # fac_p=ed.nb(w-edd, T)+ed.nf(-edd, T)
        Integrand=self.Kcou*self.Kcou*SFvar*2*np.pi*fac_p*norm/self.MCSAMPF(kx, ky, w, qx, qy)
        S0=np.mean(Integrand)
        S2=np.mean(Integrand**2)
        dels=np.sqrt(S2-S0)/self.Npoints_int_pre

        ang=np.arctan2(qy,qx)
        ei=time.time()
        # print(ei-si," seconds ",qx, qy)

        return S0, ang,dels
    
    def integrand_par_MC_RW(self,w, norm,qp):
        si=time.time()
        qx,qy=qp[0], qp[1]

        [kx,ky]=self.MC_points(w, qx,qy )

        edd=self.ed.Disp_mu(kx+qx,ky+qy)
        om=w-edd

        SFvar=self.SS.Dynamical_SF(kx,ky,om)
        fac_p=(1+np.exp(-w/self.T))*(1-self.ed.nf(edd, self.T))
        # fac_p=ed.nb(w-edd, T)+ed.nf(-edd, T)
        Integrand=self.Kcou*self.Kcou*SFvar*2*np.pi*fac_p*norm/self.MCSAMPF(kx, ky, w, qx, qy)
        S0=np.mean(Integrand)
        S2=np.mean(Integrand**2)
        dels=np.sqrt(S2-S0)/self.Npoints_int_pre

        ang=np.arctan2(qy,qx)
        ei=time.time()
        # print(ei-si," seconds ",qx, qy)

        return S0, ang,dels

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
    
    def integrand_par_q(self,ds,w,qp):
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

    def integrand_par_submit(self,kx,ky,w,ds,qps):
        S0s=[]
        angs=[]
        delss=[]
        
        si2=time.time()
        for qp in qps:
            si=time.time()

            qx,qy=qp[0], qp[1]

            edd=self.ed.Disp_mu(kx+qx,ky+qy)
            om=w-edd

            SFvar=self.SS.Dynamical_SF(kx,ky,om)

            fac_p=(1+np.exp(-w/self.T))*(1-self.ed.nf(edd, self.T))
            # fac_p=ed.nb(w-edd, T)+ed.nf(-edd, T)
            Integrand=self.Kcou*self.Kcou*SFvar*2*np.pi*fac_p

            # ##removing the point at the origin
            # ind=np.where(np.abs(kx)+np.abs(ky)<np.sqrt(ds))[0]
            # Integrand=np.delete(Integrand,ind)
            S0=np.sum(Integrand*ds)

            # Vol_rec=self.latt.Vol_BZ()
            dels=10*ds*np.max(np.abs(np.diff(Integrand)))#np.sqrt(ds/Vol_rec)*Vol_rec#*np.max(np.abs(np.diff(Integrand)))*0.1
            ang=np.arctan2(qy,qx)
            S0s.append(S0)
            angs.append(ang)
            delss.append(dels)

            ei=time.time()
            print(ei-si," seconds ",qx, qy)
        ei2=time.time()
        print(ei2-si2," seconds ")
        return [S0s, angs,delss]

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


    ###################
    # SEQUENTIAL INTEGRALS
    ###################

    def Int_point(self,qx,qy,w):
        Vol_rec=self.latt.Vol_BZ()
        Npoints_int=np.shape(self.kx)
        

        ds=Vol_rec/Npoints_int

        print("starting with calculation of Sigma")
        s=time.time()
        

        Integrand=self.integrand(self.kx,self.ky,qx,qy,w)
        S0=np.sum(Integrand*ds)
        dels=np.sqrt(ds)
            

        e=time.time()
        print("time for calc....",e-s)


        return [S0, dels]

    def Int_point_MC(self,qx,qy,w):
        Vol_rec=self.latt.Vol_BZ()
        Npoints_int=np.shape(self.kx)
        
        

        ds=Vol_rec/Npoints_int

        print("starting with calculation of Sigma")
        s=time.time()

        [kx,ky]=self.MC_points(w, qx,qy)
        # [kx,ky]=self.MC_points(w, 0,0)
        # kx=kx-qx
        # ky=ky-qy
        norm=np.sum(self.MCSAMPF(self.kx, self.ky, w, qx, qy)*ds)
        Integrand=self.integrand(kx,ky,qx,qy,w)*norm/self.MCSAMPF(kx, ky, w, qx, qy)
        S0=np.mean(Integrand)
        S2=np.mean(Integrand**2)
        dels=np.sqrt(S2-S0)/self.Npoints_int_pre
            

        e=time.time()
        print("time for calc....",e-s)


        return [S0, dels]


    def Int_FS_nofreq(self):
        Vol_rec=self.latt.Vol_BZ()
        Npoints_int=np.shape(self.kx)
        shifts=[]
        angles=[]
        delsd=[]

        ds=Vol_rec/Npoints_int

        print("starting with calculation of Sigma")
        s=time.time()
        for ell in range(self.NpointsFS):

            qx=self.qxFS[ell]
            qy=self.qyFS[ell]

            Integrand=self.integrand(self.kx,self.ky,qx,qy,0.0)

            S0=np.sum(Integrand*ds)
            dels=np.sqrt(ds)
            shifts.append(S0)
            delsd.append(dels)
            angles.append(np.arctan2(qy,qx))
            printProgressBar(ell + 1, self.NpointsFS, prefix = 'Progress:', suffix = 'Complete', length = 50)


        e=time.time()
        print("time for calc....",e-s)

        shifts=np.array(shifts) 
        angles=np.array(angles)

        return [shifts, angles, delsd]

        
    def Int_FS_nofreq_sq(self):
        Vol_rec=self.latt.Vol_BZ()
        Npoints_int=np.shape(self.kxsq)
        shifts=[]
        angles=[]
        delsd=[]

        ds=Vol_rec/Npoints_int

        print("starting with calculation of Sigma")
        s=time.time()
        for ell in range(self.NpointsFS):

            qx=self.qxFS[ell]
            qy=self.qyFS[ell]

            Integrand=self.integrand(self.kxsq,self.kysq,qx,qy,0.0)

            S0=np.sum(Integrand*ds)
            dels=np.sqrt(ds)
            shifts.append(S0)
            delsd.append(dels)
            angles.append(np.arctan2(qy,qx))
            printProgressBar(ell + 1, self.NpointsFS, prefix = 'Progress:', suffix = 'Complete', length = 50)


        e=time.time()
        print("time for calc....",e-s)

        shifts=np.array(shifts) 
        angles=np.array(angles)

        return [shifts, angles, delsd]

    def Int_FS_quad_nofreq(self):
    
        shifts=[]
        angles=[]
        delsd=[]
        ###########FOR HEX INT
        rp=np.sqrt(3)/2
        r=2*np.pi/np.sqrt(3) #radius inscruped 
        t=r/rp
        # r=np.sqrt(3)/2;
        # t=1
        u=np.sqrt(t**2-r**2)
        m=(r-0)/(-(t/2)-(-t))

        e1=1.49e-08
        e2=1.49e-08

    

        print("starting with calculation of Sigma")
        s=time.time()
        for ell in range(self.NpointsFS):

            qx=self.qxFS[ell]
            qy=self.qyFS[ell]

            I1=integrate.dblquad(self.integrand, -t, -t/2, lambda x: -m*(x+t), lambda x: m*(x+t), args=[qx,qy,0, self.T],epsabs=e1, epsrel=e2)
            I2=integrate.dblquad(self.integrand, -t/2, t/2, lambda x: -r, lambda x: r, args=[qx,qy,0, self.T],epsabs=e1, epsrel=e2)
            I3=integrate.dblquad(self.integrand, t/2, t, lambda x: m*(x-t), lambda x: -m*(x-t), args=[qx,qy,0, self.T],epsabs=e1, epsrel=e2)


            S0=I1[0]+I2[0]+I3[0]
            dels=np.sqrt(I1[1]**2+I2[1]**2+I3[1]**2)

            shifts.append(S0)
            delsd.append(dels)
            angles.append(np.arctan2(qy,qx))
            printProgressBar(ell + 1, self.NpointsFS, prefix = 'Progress:', suffix = 'Complete', length = 50)


        e=time.time()
        print("time for calc....",e-s)

        shifts=np.array(shifts) 
        angles=np.array(angles)

        return [shifts, angles, delsd]
    
    #####################
    # PARALLEL RUN
    #####################

    def Int_FS_parsum(self, w, Machine):
        shifts=[]
        angles=[]
        delsd=[]
        


        print("starting with calculation of Sigma")
        s=time.time()
        for ell in range(self.NpointsFS):

            qx=self.qxFS[ell]
            qy=self.qyFS[ell]

            [S0,ds]=self.integrand_parsum(self.kx,self.ky,qx,qy,w,Machine)

            dels=np.sqrt(ds)
            shifts.append(S0)
            delsd.append(dels)
            angles.append(np.arctan2(qy,qx))
            # printProgressBar(ell + 1, self.NpointsFS, prefix = 'Progress:', suffix = 'Complete', length = 50)


        e=time.time()
        print("time for calc....",e-s)

        shifts=np.array(shifts) 
        angles=np.array(angles)

        return [np.array(shifts), np.array(delsd)]


    def parInt_FS(self,w, Machine, sq):
        if Machine=='FMAC':
            maxthreads=8
        elif Machine=='CH1':
            maxthreads=44
        elif Machine=='UBU':
            maxthreads=12
        else:
            maxthreads=6

        if sq==True:
            kx=self.kxsq
            ky=self.kysq
        else:
            kx=self.kx
            ky=self.ky

        Vol_rec=self.latt.Vol_BZ()
        Npoints_int=np.size(kx)
        shifts=[]
        angles=[]
        delsd=[]

        ds=Vol_rec/Npoints_int
        qp=np.array([self.qxFS, self.qyFS]).T
        Npoints_FS=np.size(self.qxFS)
        print(Npoints_FS, "the fermi surface points",int(Npoints_FS/maxthreads),"numtheads")
        Nthreads=int(Npoints_FS/maxthreads)

        partial_integ = functools.partial(self.integrand_par, kx, ky,w,ds)

        print("starting with calculation of Sigma")
        s=time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(partial_integ, qp, chunksize=Nthreads)

            for result in results:
                shifts.append(result[0])
                angles.append(result[1])
                delsd.append(result[2])

        e=time.time()
        print("time for calc....",e-s)

        shifts=np.array(shifts) 
        angles=np.array(angles)

        return [np.array(shifts), np.array(angles), np.array(delsd)]


    def par_submit_Int_FS(self,w, Machine,sq):
        
        if Machine=='FMAC':
            workers=10
        elif Machine=='CH1':
            workers=206
        elif Machine=='UBU':
            workers=20
        else:
            workers=10

        if sq==True:
            kx=self.kxsq
            ky=self.kysq
        else:
            kx=self.kx
            ky=self.ky

        Vol_rec=self.latt.Vol_BZ()
        Npoints_int=np.size(kx)
        shifts=[]
        angles=[]
        delsd=[]

        ds=Vol_rec/Npoints_int
        qp=np.array([self.qxFS, self.qyFS]).T
        Npoints_FS=np.size(self.qxFS)
        chunk=Npoints_FS// workers
        print("chunksize is ", chunk)

        # partial_integ = functools.partial(self.integrand_par_submit, self.kx, self.ky, w, ds)

        print("starting with calculation of Sigma")
        s=time.time()
        futures = []
        found=[]

        shifts=[]
        angles=[]
        delsd=[]

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for i in range(workers):
                cstart = chunk * i
                cstop = chunk * (i + 1) if i != workers - 1 else Npoints_FS
                # futures.append(executor.submit(partial_integ, qp[cstart:cstop]))
                futures.append(executor.submit(self.integrand_par_submit, kx, ky, w, ds, qp[cstart:cstop]))
                print(np.shape(qp[cstart:cstop]), cstart, cstop)

            # 2.2. Instruct workers to process results as they come, when all are
            #      completed or .....
            concurrent.futures.as_completed(futures) # faster than cf.wait()
            # concurrent.futures.wait(fs=1000)
            # 2.3. Consolidate result as a list and return this list.
            for f in futures:
                try:
                    [presh,preang,predel]=f.result()
                    shifts=shifts+presh
                    angles=angles+preang
                    delsd=delsd+predel
                except:
                    print_exc()
            foundsize = len(found)
            end = time.time() - s
            print('within statement of def _concurrent_submit():')
            print("found {0} in {1:.4f}sec".format(foundsize, end))
        return [np.array(shifts), np.array(angles), np.array(delsd)]
 
    def parInt_FS_MC(self,w, Machine):
        
        #setup parallel run
        
        if Machine=='FMAC':
            maxthreads=8
        elif Machine=='CH1':
            maxthreads=44
        elif Machine=='UBU':
            maxthreads=12
        else:
            maxthreads=6
    
            
        Npoints_FS=np.size(self.qxFS)
        print("the fermi surface points",Npoints_FS, "numtheads", int(Npoints_FS/maxthreads))
        Nthreads=int(Npoints_FS/maxthreads)
        
        # #setup points
        # print("starting with calculation sampling")
        # s=time.time()
        # futures = []
        # kxsamp=[]
        # kysamp=[]

        # workers=206
        # Ntotsamples=self.Npoints_int_pre*self.Npoints_int_pre
        # chunk=Ntotsamples// workers
        # parallel_MCS_sizes=np.ones(maxthreads)*chunk

        # with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        #     for i in range(workers):
        #         cstart = chunk * i
        #         cstop = chunk * (i + 1) if i != workers - 1 else Ntotsamples
        #         # futures.append(executor.submit(partial_integ, qp[cstart:cstop]))
        #         futures.append(executor.submit(self.MC_points_par, w, 0,0, parallel_MCS_sizes[cstart:cstop]))
        #         print(np.shape(parallel_MCS_sizes[cstart:cstop]), cstart, cstop)

        #     # 2.2. Instruct workers to process results as they come, when all are
        #     #      completed or .....
        #     concurrent.futures.as_completed(futures) # faster than cf.wait()
        #     # concurrent.futures.wait(fs=1000)
        #     # 2.3. Consolidate result as a list and return this list.
        #     for f in futures:
        #         try:
        #             [prex,prey]=f.result()
        #             kxsamp=kxsamp+prex
        #             kysamp=kysamp+prey

        #         except:
        #             print_exc()
        # kx=np.array(kxsamp)
        # ky=np.array(kysamp)
        # e=time.time()
        # print("time for sampling....",e-s, "total samples..", np.size(kx), "..intended.. ",self.Npoints_int_pre*self.Npoints_int_pre)

        
        
        # print("starting with calculation sampling")
        # s=time.time()
        # kxsamp=[]
        # kysamp=[]
        # partial_samp = functools.partial(self.MC_points_par, w, 0,0)
        # Totsamp=self.Npoints_int_pre*self.Npoints_int_pre*50
        # Nthrds_samp=440
        # chsize=Nthrds_samp//maxthreads
        # parallel_MCS_sizes=np.ones(Nthrds_samp)*int(Totsamp/Nthrds_samp)
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     results = executor.map(partial_samp, parallel_MCS_sizes, chunksize=chsize)

        #     for result in results:
        #         kxsamp=kxsamp+result[0]
        #         kysamp=kysamp+result[1]

        # kx=np.array(kxsamp)
        # ky=np.array(kysamp)
        # e=time.time()
        # print("time for sampling....",e-s, "total samples..", np.size(kx), "..intended.. ",Totsamp)
        # # plt.scatter(self.kx,self.ky, c=self.MCSAMPF(self.kx,self.ky,0,0,0) )
        # # plt.show()
        [kx,ky]=self.MC_points(w, 0,0)

        Vol_rec=self.latt.Vol_BZ()
        Npoints_int=np.size(self.kx)
        

        ds=Vol_rec/Npoints_int
        norm=np.sum(self.MCSAMPF(self.kx, self.ky, w, 0, 0 )*ds)
        qp=np.array([self.qxFS, self.qyFS]).T
        

        partial_integ = functools.partial(self.integrand_par_MC, kx, ky,w,  norm)

        shifts=[]
        angles=[]
        delsd=[]
        print("starting with calculation of Sigma")
        s=time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(partial_integ, qp, chunksize=Nthreads)

            for result in results:
                shifts.append(result[0])
                angles.append(result[1])
                delsd.append(result[2])

        e=time.time()
        print("time for calc....",e-s)

        shifts=np.array(shifts) 
        angles=np.array(angles)

        return [np.array(shifts), np.array(angles), np.array(delsd)]
    
    def parInt_FS_MC_multRW(self,w, Machine):
        
        #setup parallel run
        
        if Machine=='FMAC':
            maxthreads=8
        elif Machine=='CH1':
            maxthreads=44
        elif Machine=='UBU':
            maxthreads=12
        else:
            maxthreads=6
    
            
        Npoints_FS=np.size(self.qxFS)
        print("the fermi surface points",Npoints_FS, "numtheads", int(Npoints_FS/maxthreads))
        Nthreads=int(Npoints_FS/maxthreads)
        
        

        Vol_rec=self.latt.Vol_BZ()
        Npoints_int=np.size(self.kx)
        

        ds=Vol_rec/Npoints_int
        norm=np.sum(self.MCSAMPF(self.kx, self.ky, w, 0, 0 )*ds)
        qp=np.array([self.qxFS, self.qyFS]).T
        

        partial_integ = functools.partial(self.integrand_par_MC_RW, w,  norm)

        shifts=[]
        angles=[]
        delsd=[]
        print("starting with calculation of Sigma")
        s=time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(partial_integ, qp, chunksize=Nthreads)

            for result in results:
                shifts.append(result[0])
                angles.append(result[1])
                delsd.append(result[2])

        e=time.time()
        print("time for calc....",e-s)

        shifts=np.array(shifts) 
        angles=np.array(angles)

        return [np.array(shifts), np.array(angles), np.array(delsd)]
    


    ##############
    #   PARALLEL RUNS WITH FREQUENCY DEPENDENCE
    ##############

    def Int_FS_parsum_w(self, theta, w, Machine):
        shifts=[]
        angles=[]
        delsd=[]
        
        for ell in range(self.NpointsFS):

            qx=self.qxFS[ell]
            qy=self.qyFS[ell]

            angles.append(np.arctan2(qy,qx))

        itheta=np.argmin((np.array(angles)-theta)**2)


        qx=self.qxFS[itheta]
        qy=self.qyFS[itheta]

        print("starting with calculation of Sigma")
        s=time.time()

        for ome in w:

            [S0,ds]=self.integrand_parsum(self.kx,self.ky,qx,qy,ome, Machine)

            dels=np.sqrt(ds)
            shifts.append(S0)
            delsd.append(dels)
            # printProgressBar(ell + 1, self.NpointsFS, prefix = 'Progress:', suffix = 'Complete', length = 50)


        e=time.time()
        print("time for calc....",e-s)

        shifts=np.array(shifts) 
        angles=np.array(angles)

        return [np.array(shifts),w,  np.array(delsd)]

    def parInt_w(self, theta, w, Machine, sq):
        if Machine=='FMAC':
            maxthreads=8
        elif Machine=='CH1':
            maxthreads=10
        elif Machine=='UBU':
            maxthreads=12
        else:
            maxthreads=6

        # if sq==True:
        #     kx=self.kxsq
        #     ky=self.kysq
        # else:
        #     kx=self.kx
        #     ky=self.ky
        ###determining the momenta on the FS
        shifts=[]
        angles=[]
        delsd=[]
        
        for ell in range(self.NpointsFS):

            qx=self.qxFS[ell]
            qy=self.qyFS[ell]

            angles.append(np.arctan2(qy,qx))

        itheta=np.argmin((np.array(angles)-theta)**2)


        qx=self.qxFS[itheta]
        qy=self.qyFS[itheta]
        print("energy at the point.." , qx, qy, " is " ,self.ed.Disp_mu(qx,qy))
        print("the angle is ", angles[itheta])
        qp=np.array([qx,qy]).T

        Vol_rec=self.latt.Vol_BZ()
        Npoints_int=np.size(self.kxsq)
        ds=Vol_rec/Npoints_int

        
        
        Npoints_w=np.size(w)
        print(Npoints_w, "the fermi surface points",int(Npoints_w/maxthreads),"chunk numtheads")
        Nthreads=int(Npoints_w/maxthreads)

        partial_integ = functools.partial(self.integrand_par_w, qp, ds)

        print("starting with calculation of Sigma")
        s=time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(partial_integ, w, chunksize=Nthreads)

            for result in results:
                shifts.append(result[0])
                angles.append(result[1])
                delsd.append(result[2])

        e=time.time()
        print("time for calc....",e-s)

        shifts=np.array(shifts) 
        delsd=np.array(delsd)

        return [shifts, w, delsd]

    def parInt_q(self, theta, w, Machine, sq):
        if Machine=='FMAC':
            maxthreads=8
        elif Machine=='CH1':
            maxthreads=5
        elif Machine=='UBU':
            maxthreads=12
        else:
            maxthreads=6

        shifts=[]
        angles=[]
        delsd=[]
        
        for ell in range(self.NpointsFS):

            qx=self.qxFS[ell]
            qy=self.qyFS[ell]

            angles.append(np.arctan2(qy,qx))

        itheta=np.argmin((np.array(angles)-theta)**2)
        qx=self.qxFS[itheta]
        qy=self.qyFS[itheta]
        
        
        kloc=np.array([qx,qy])
        vf=self.ed.Fermi_Vel(qx,qy)
        [vfx,vfy]=vf
        VF=np.sqrt(vfx**2+vfy**2)
        Npoints_w=20
        KF=np.sqrt(kloc@kloc)
        amp=KF/5
        fac=amp/VF
        mesh=np.linspace(0,fac,Npoints_w)
        q2=np.array([mesh*vfx,mesh*vfy]).T
        q=np.array([qx+mesh*vfx,qy+mesh*vfy]).T
        
        
        plt.scatter( self.qxFS,self.qyFS)
        plt.scatter(q[:,0], q[:,1], c='r')
        # plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig("FS_ene_theta_"+str(round(theta*180/np.pi))+".png")
        plt.close()

        
        print("energy at the point.." , qx, qy, " is " ,self.ed.Disp_mu(qx,qy))
        print("the angle is ", angles[itheta])
        qp=np.array([qx,qy]).T

        Vol_rec=self.latt.Vol_BZ()
        Npoints_int=np.size(self.kxsq)
        ds=Vol_rec/Npoints_int

        
        print(Npoints_w, "the fermi surface points",int(Npoints_w/maxthreads),"chunk numtheads")
        Nthreads=int(Npoints_w/maxthreads)

        partial_integ = functools.partial(self.integrand_par_q, ds,w)
    

        print("starting with calculation of Sigma")
        s=time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(partial_integ, q, chunksize=Nthreads)

            for result in results:
                shifts.append(result[0])
                angles.append(result[1])
                delsd.append(result[2])

        e=time.time()
        print("time for calc....",e-s)

        shifts=np.array(shifts) 
        delsd=np.array(delsd)

        return [shifts,q2, delsd]

    ############
    # OUTPUT
    #######

    def output_res_fixed_w(self, arg, J, T , sh_job, prefixd):

        if sh_job:
            prefdata="DataRun/"
            prefim="ImgsRun/"
        else:
            path = prefixd+"dir_T_"+str(T)+"_"+self.SS.name+"_"+datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

            try:
                os.mkdir(path)
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                print ("Successfully created the directory %s " % path)
            prefdata=path+"/"
            prefim=path+"/"

        Vertices_list, Gamma, K, Kp, M, Mp=self.latt.FBZ_points(self.latt.b[0,:],self.latt.b[1,:])
        VV=np.array(Vertices_list+[Vertices_list[0]])
        dispname=self.ed.name
        SFname=self.SS.name

        ####making plots
        print(" plotting data from the run ...")

        [shifts, angles, delsd]=arg
        plt.errorbar(angles,shifts,yerr=delsd, fmt='.')
        plt.scatter(angles,shifts, s=1, c='r')
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"-Im$\Sigma (k_F(\theta),0)$ mev")
        plt.tight_layout()
        plt.savefig(prefim+f"errorbars_J={J}_T={T}_"+SFname+"_"+dispname+".png", dpi=200)
        # plt.show()
        plt.close()

        plt.scatter(angles,shifts, s=1, c='r')
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"-Im$\Sigma (k_F(\theta),0)$ mev")
        plt.tight_layout()
        plt.savefig(prefim+f"scatterplot_J={J}_T={T}_"+SFname+"_"+dispname+".png", dpi=200)
        # plt.show()
        plt.close()


        plt.plot(VV[:,0], VV[:,1], c='k')
        plt.scatter([0],[0], c='k', s=1)
        plt.scatter(self.qxFS,self.qyFS,c=shifts)
        plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(prefim+f"FSplot_J={J}_T={T}_"+SFname+"_"+dispname+".png", dpi=200)
        # plt.show()
        plt.close()

        plt.plot(VV[:,0], VV[:,1], c='k')
        plt.scatter([0],[0], c='k', s=1)
        
        mini=np.min(shifts)
        mini2=np.sort(shifts)[int(np.size(shifts)/20)]
        logmax=np.log10( np.max(shifts)-np.min(shifts) +1e-17)
        logmin=np.log10(mini2-np.min(shifts) +1e-17)
        plt.scatter(self.qxFS,self.qyFS,c=np.log10( shifts-mini +1e-17) )
        plt.clim(logmin,logmax )
        plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(prefim+f"logFSplot_J={J}_T={T}_"+SFname+"_"+dispname+".png", dpi=200)
        # plt.show()
        plt.close()

        ####saving data
        print("saving data from the run ...")

        with open(prefdata+f"kx_FS_J={J}_T={T}_"+SFname+"_"+dispname+".npy", 'wb') as f:
            np.save(f, self.qxFS)

        with open(prefdata+f"ky_FS_J={J}_T={T}_"+SFname+"_"+dispname+".npy", 'wb') as f:
            np.save(f, self.qyFS)

        with open(prefdata+f"angles_FS_J={J}_T={T}_"+SFname+"_"+dispname+".npy", 'wb') as f:
            np.save(f, angles)

        with open(prefdata+f"SelfE_J={J}_T={T}_"+SFname+"_"+dispname+".npy", 'wb') as f:
            np.save(f, shifts)

        with open(prefdata+f"errSelfE_J={J}_T={T}_"+SFname+"_"+dispname+".npy", 'wb') as f:
            np.save(f, delsd)

    def output_res_q(self, arg, J, T , theta, sh_job, prefixd):

        shifts=[]
        angles=[]
        delsd=[]
        
        for ell in range(self.NpointsFS):

            qx=self.qxFS[ell]
            qy=self.qyFS[ell]

            angles.append(np.arctan2(qy,qx))

        itheta=np.argmin((np.array(angles)-theta)**2)


        qx=self.qxFS[itheta]
        qy=self.qyFS[itheta]

        if sh_job:
            prefdata="DataRun/"
            prefim="ImgsRun/"
        else:
            path = prefixd+"dir_T_"+str(T)+"_"+self.SS.name+"_"+datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

            try:
                os.mkdir(path)
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                print ("Successfully created the directory %s " % path)

            prefdata=path+"/"
            prefim=path+"/"

        Vertices_list, Gamma, K, Kp, M, Mp=self.latt.FBZ_points(self.latt.b[0,:],self.latt.b[1,:])
        VV=np.array(Vertices_list+[Vertices_list[0]])
        dispname=self.ed.name
        SFname=self.SS.name+"_theta_"+str(round(theta*180/np.pi, 2))

        ####making plots
        print(" plotting data from the run ...")

        [shifts, qq, delsd,w]=arg
        q=np.sqrt(qq[:,0]**2+qq[:,1]**2)
        dispname=dispname+"_w_"+str(w)
        plt.errorbar(q,shifts,yerr=delsd, fmt='.')
        plt.scatter(q,shifts, s=1, c='r')
        plt.xlabel(r"$q$")
        plt.ylabel(r"-Im$\Sigma (k_F(\theta)+\vec{q},\omega=$"+str(w)+") mev")
        plt.tight_layout()
        plt.savefig(prefim+f"errorbars_J={J}_T={T}_"+SFname+"_"+dispname+".png", dpi=200)
        # plt.show()
        plt.close()

        plt.scatter(q,shifts, s=1, c='r')
        plt.xlabel(r"$\omega$ mev")
        plt.ylabel(r"-Im$\Sigma (k_F(\theta),\omega)$ mev")
        plt.tight_layout()
        plt.savefig(prefim+f"scatterplot_J={J}_T={T}_"+SFname+"_"+dispname+".png", dpi=200)
        # plt.show()
        plt.close()


        plt.plot(VV[:,0], VV[:,1], c='k')
        plt.scatter([0],[0], c='k', s=1)
        plt.scatter(self.qxFS,self.qyFS, c='b')
        plt.scatter([qx],[qy], c='r')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(prefim+f"FSplot_J={J}_T={T}_"+SFname+"_"+dispname+".png", dpi=200)
        # plt.show()
        plt.close()
        

        ####saving data
        print("saving data from the run ...")

        with open(prefdata+f"kx_FS_J={J}_T={T}_"+SFname+"_"+dispname+".npy", 'wb') as f:
            np.save(f,qx)

        with open(prefdata+f"ky_FS_J={J}_T={T}_"+SFname+"_"+dispname+".npy", 'wb') as f:
            np.save(f, qy)

        with open(prefdata+f"w_FS_J={J}_T={T}_"+SFname+"_"+dispname+".npy", 'wb') as f:
            np.save(f, w)
            
        with open(prefdata+f"qq_FS_J={J}_T={T}_"+SFname+"_"+dispname+".npy", 'wb') as f:
            np.save(f, qq)

        with open(prefdata+f"SelfE_J={J}_T={T}_"+SFname+"_"+dispname+".npy", 'wb') as f:
            np.save(f, shifts)

        with open(prefdata+f"errSelfE_J={J}_T={T}_"+SFname+"_"+dispname+".npy", 'wb') as f:
            np.save(f, delsd)

            
    def output_res_fixed_FSpoint(self, arg, J, T , theta, sh_job, prefixd):

        shifts=[]
        angles=[]
        delsd=[]
        
        for ell in range(self.NpointsFS):

            qx=self.qxFS[ell]
            qy=self.qyFS[ell]

            angles.append(np.arctan2(qy,qx))

        itheta=np.argmin((np.array(angles)-theta)**2)


        qx=self.qxFS[itheta]
        qy=self.qyFS[itheta]

        if sh_job:
            prefdata="DataRun/"
            prefim="ImgsRun/"
        else:
            path = prefixd+"dir_T_"+str(T)+"_"+self.SS.name+"_"+datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

            try:
                os.mkdir(path)
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                print ("Successfully created the directory %s " % path)

            prefdata=path+"/"
            prefim=path+"/"

        Vertices_list, Gamma, K, Kp, M, Mp=self.latt.FBZ_points(self.latt.b[0,:],self.latt.b[1,:])
        VV=np.array(Vertices_list+[Vertices_list[0]])
        dispname=self.ed.name
        SFname=self.SS.name+"_theta_"+str(round(theta*180/np.pi, 2))

        ####making plots
        print(" plotting data from the run ...")

        [shifts, w, delsd]=arg
        plt.errorbar(w,shifts,yerr=delsd, fmt='.')
        plt.scatter(w,shifts, s=1, c='r')
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"-Im$\Sigma (k_F(\theta),\omega)$ mev")
        plt.tight_layout()
        plt.savefig(prefim+f"errorbars_J={J}_T={T}_"+SFname+"_"+dispname+".png", dpi=200)
        # plt.show()
        plt.close()

        plt.scatter(w,shifts, s=1, c='r')
        plt.xlabel(r"$\omega$ mev")
        plt.ylabel(r"-Im$\Sigma (k_F(\theta),\omega)$ mev")
        plt.tight_layout()
        plt.savefig(prefim+f"scatterplot_J={J}_T={T}_"+SFname+"_"+dispname+".png", dpi=200)
        # plt.show()
        plt.close()


        plt.plot(VV[:,0], VV[:,1], c='k')
        plt.scatter([0],[0], c='k', s=1)
        plt.scatter(self.qxFS,self.qyFS, c='b')
        plt.scatter([qx],[qy], c='r')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(prefim+f"FSplot_J={J}_T={T}_"+SFname+"_"+dispname+".png", dpi=200)
        # plt.show()
        plt.close()
        

        ####saving data
        print("saving data from the run ...")

        with open(prefdata+f"kx_FS_J={J}_T={T}_"+SFname+"_"+dispname+".npy", 'wb') as f:
            np.save(f,qx)

        with open(prefdata+f"ky_FS_J={J}_T={T}_"+SFname+"_"+dispname+".npy", 'wb') as f:
            np.save(f, qy)

        with open(prefdata+f"w_FS_J={J}_T={T}_"+SFname+"_"+dispname+".npy", 'wb') as f:
            np.save(f, w)

        with open(prefdata+f"SelfE_J={J}_T={T}_"+SFname+"_"+dispname+".npy", 'wb') as f:
            np.save(f, shifts)

        with open(prefdata+f"errSelfE_J={J}_T={T}_"+SFname+"_"+dispname+".npy", 'wb') as f:
            np.save(f, delsd)



def main() -> int:
    
    
    try:
        index_sf=int(sys.argv[1])

    except (ValueError, IndexError):
        raise Exception("Input integer in the firs argument to choose structure factor")


    try:
        N_SFs=10 #number of SF's currently implemented
        a=np.arange(N_SFs)
        a[index_sf]

    except (IndexError):
        raise Exception(f"Index has to be between 0 and {N_SFs-1}")


    try:
        Rel_BW_fac=float(sys.argv[2])

    except (ValueError, IndexError):
        raise Exception("Input float in the second argument to scale the spin band width")


    try:
        T=float(sys.argv[3])


    except (ValueError, IndexError):
        raise Exception("Input float in the third argument is the temperature")
        
    try:
        Machine=sys.argv[4]


    except (ValueError, IndexError):
        raise Exception("Input string in the fourth argument is the machine to run,\n this affects optimization in parallel integration routines current options are: CH1 FMAC UBU")




    ##########################
    ##########################
    # parameters
    ##########################
    ##########################

    # # #electronic parameters
    J=2*5.17*Rel_BW_fac #in mev
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
    Npoints_int_pre, NpointsFS_pre=1000,600
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
    
    # ed=Dispersion.Dispersion_circ([tp1,tp2],fill)
    [KxFS,KyFS]=ed.FS_contour(NpointsFS_pre)
    NsizeFS=np.size(KxFS)
    # [KxFS2,KyFS2]=ed.FS_contour2(NpointsFS_pre)
    # plt.scatter(KxFS,KyFS, c=np.log10(np.abs(ed.Disp_mu(KxFS,KyFS))+1e-34) )
    # f=np.log10(np.abs(ed.Disp_mu(KxFS2,KyFS2))+1e-34)

    plt.scatter(KxFS,KyFS )
    i=500
    kloc=np.array([KxFS[i],KyFS[i]])
    vf=ed.Fermi_Vel(KxFS[i],KyFS[i])
    [vfx,vfy]=vf
    VF=np.sqrt(vfx**2+vfy**2)
    print("the fermi velocity is rough",VF)
    
    KF=np.sqrt(kloc@kloc)
    amp=KF/10
    fac=amp/KF
    fac2=amp/VF
    mesh=np.linspace(0,fac,100)
    mesh2=np.linspace(0,fac2,100)
    q=np.array([KxFS[i]*(1+mesh),KyFS[i]*(1+mesh)]).T
    q2=np.array([KxFS[i]+mesh2*vfx,KyFS[i]+mesh2*vfy]).T
    
    plt.scatter(q2[:,0], q2[:,1], c='r')
    plt.colorbar()
    plt.savefig("FS_ene.png")
    plt.close()
    # plt.show()
    # print(f"dispersion params: {tp1} \t {tp2}")
    # # ed.PlotFS(l)
    

    ##parameters for structure factors
    #matches the SF from fit at half filling
    
    '''
    EF=ed.EF
    m=EF/2
    gamma=EF*1000
    vmode=EF/2
    gcoupl=EF/2
    '''


    EF=ed.EF
    print("The fermi energy in mev is: {e}, and in units of J: {e2}, the bandwidth is:{e3}".format(e=EF*J,e2=EF, e3=ed.bandwidth))
    m=100 #in units of J
    gamma=m*2
    vmode=m*2
    gcoupl=m/20


    C=4.0
    D=1 #0.85

    #choosing the structure factor
    if index_sf==0:
        SS=StructureFactor.StructureFac_fit(T,KX, KY)
    elif index_sf==1:
        SS=StructureFactor.StructureFac_fit_F(T)
    elif index_sf==2:
        SS=StructureFactor.StructureFac_PM(T, gamma, vmode, m )
    elif index_sf==3:
        SS=StructureFactor.StructureFac_PM_Q(T, gamma, vmode, m )
    elif index_sf==4:
        SS=StructureFactor.StructureFac_PM_Q2(T, gamma, vmode, m )
    elif index_sf==5:
        SS=StructureFactor.StructureFac_fit_no_diff_peak(T)
    elif index_sf==6:
        SS=StructureFactor.MD_SF(T)
    elif index_sf==7:
        SS=StructureFactor.Langevin_SF(T, KX, KY)
    elif index_sf==8:
        SS=StructureFactor.StructureFac_diff_peak_fit(T)
    else:
        SS=StructureFactor.SF_diff_peak(T, D, C)

    # plt.scatter(KX,KY,c=SS.Dynamical_SF(KX,KY,0.1), s=0.5)
    # plt.colorbar()
    # pl.show()
    
    Momentum_cut=SS.momentum_cut_high_symmetry_path(l, 2000, 1000)

    ##########################
    ##########################
    # Calls to integration routine
    ##########################
    ##########################

    SE=SelfE(T ,ed ,SS,  Npoints_int_pre, NpointsFS_pre, Kcou, "hex")  
    


    ##################
    # integration accross frequencies for fixed FS Point
    #################
    
    
   
    thetas= np.linspace(0, np.pi/6, 6)[2:]
    for theta in thetas:
        domeg=0.1
        maxw=np.min([5*T,10]) #in unitsw of J
        warr=np.arange(0,maxw,domeg)
        for w in warr:
            sq=True
            # [shifts, w, delsd]=SE.Int_FS_parsum_w( theta, w, Machine, sq)
            [shifts, q, delsd]=SE.parInt_q( theta, w, Machine, sq)
            shifts=shifts*J
            delsd=delsd*J
            w=J*w
            job=True
            SE.output_res_q( [shifts, q, delsd, w], J, T, theta, job , "antipodal_point_check_div")
            

    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
