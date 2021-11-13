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

        [qxFS,qyFS]=ed.FS_contour( NpointsFS_pre)
        self.qxFS=qxFS
        self.qyFS=qyFS

        self.NpointsFS=np.size(qxFS)
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

    def integrand_par_w(self,qp,kx,ky,ds,w):
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
        Npoints_int=np.size(kx)
        ds=Vol_rec/Npoints_int

        
        
        Npoints_w=np.size(self.qxFS)
        print(Npoints_w, "the fermi surface points",int(Npoints_w/maxthreads),"numtheads")
        Nthreads=int(Npoints_w/maxthreads)

        partial_integ = functools.partial(self.integrand_par_w, qp, kx,ky,ds)

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
        SFname=self.SS.name

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
    # J=2*5.17*Rel_BW_fac #in mev
    # tp1=568/J #in units of Js\
    # tp2=-tp1*108/568 #/tpp1
    # ##coupling 
    # U=4000/J
    # g=100/J
    # Kcou=g*g/U
    # # fill=0.67 #van hove
    # fill=0.5
    

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
    J=2*5.17 #in mev
    tp1=568/J #in units of Js
    tp2=0.065*tp1
    ##coupling 
    U=4000/J
    g=100/J
    Kcou=g*g/U
    fill=0.1

    ##########################
    ##########################
    # Geometry/Lattice
    ##########################
    ##########################

    Npoints=1000
    Npoints_int_pre, NpointsFS_pre=1000,5000
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

    # ed=Dispersion.Dispersion_TB_single_band([tp1,tp2],fill)
    
    ed=Dispersion.Dispersion_circ([tp1,tp2],fill)
    [KxFS,KyFS]=ed.FS_contour(NpointsFS_pre)
    NsizeFS=np.size(KxFS)
    [KxFS2,KyFS2]=ed.FS_contour2(NpointsFS_pre)
    plt.scatter(KxFS,KyFS, c=np.log10(np.abs(ed.Disp_mu(KxFS,KyFS))+1e-34) )
    # f=np.log10(np.abs(ed.Disp_mu(KxFS2,KyFS2))+1e-34)

    # plt.scatter(KxFS2,KyFS2, c=f )
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
    
    # SE=SelfE(T ,ed ,SS,  Npoints_int_pre, NpointsFS_pre, gcoupl)  #paramag
    

    
    ##################
    #integration accross the FS for fixed frequency
    ##################

    w=0
    sq=False
    ind=int(0)
    SE.plot_logintegrand(KxFS[ind],KyFS[ind],w)
    ind=int(NsizeFS/2)
    SE.plot_logintegrand(KxFS[ind],KyFS[ind],w)
    ind=int(NsizeFS/3)
    SE.plot_logintegrand(KxFS[ind],KyFS[ind],w)
    ind=int(NsizeFS/5)
    SE.plot_logintegrand(KxFS[ind],KyFS[ind],w)
    [shifts, angles, delsd]=SE.parInt_FS(w, Machine,sq)
    # [shifts, angles, delsd]=SE.par_submit_Int_FS(w, Machine,sq)

    #converting to meV par_submit
    shifts=shifts*J
    delsd=delsd*J
    SE.output_res_fixed_w( [shifts, angles, delsd], J, T, False, "faithfull_reproduction_bug_diff_peak_circular_FS_0.1_filling_1000_samples" )



    ##################
    #integration accross frequencies for fixed FS Point
    ##################
    # # ind=3069
    # ind=3068
    # SE.plot_logintegrand(KxFS[ind],KyFS[ind],0)
    # # SE.plot_integrand(KxFS[ind],KyFS[ind],0)
    # # theta=1.5720884575889849
    # theta=1.5733805905417957
    # w=np.linspace(0,2,10)
    # sq=True
    # # [shifts, w, delsd]=SE.Int_FS_parsum_w( theta, w, Machine, sq)
    # [shifts, w, delsd]=SE.parInt_w( theta, w, Machine, sq)
    # shifts=shifts*J
    # delsd=delsd*J
    # w=J*w
    # SE.output_res_fixed_FSpoint( [shifts, w, delsd], J, T, theta, False , "antipodal_point_check_div")
    

    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
