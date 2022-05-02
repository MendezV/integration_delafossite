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

    def __init__(self, T ,ed ,SS,  Npoints_int_pre, NpointsFS_pre ,Kcou, type,Machine):
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
            self.latt=Lattice.TriangLattice(Npoints_int_pre, save,Machine ) #integration lattice 
            [self.kx,self.ky]=self.latt.read_lattice()
            [self.kxsq,self.kysq]=self.latt.read_lattice(option='sq')

        if type=="hex":
            self.latt=Lattice.TriangLattice(Npoints_int_pre, save,Machine ) #integration lattice 
            [self.kx,self.ky]=self.latt.read_lattice()
            [self.kxsq,self.kysq]=self.latt.read_lattice(option='sq')

        if type=="sq":
            self.latt=Lattice.SQLattice(Npoints_int_pre, save,Machine ) #integration lattice 
            [self.kx,self.ky]=self.latt.read_lattice()
            [self.kxsq,self.kysq]=self.latt.read_lattice()
            
        if type=="ed":
            self.latt=Lattice.TriangLattice(Npoints_int_pre, save,Machine ) #integration lattice 
            [self.kx,self.ky, dth,dr]=self.latt.Generate_lattice_ed(ed, 4000,40000)
            [self.kxsq,self.kysq]=[self.kx,self.ky]   #legacy
            self.kmag=np.sqrt(self.kxsq**2+self.kysq**2) #magnitude of k
            self.dr=dr #dr for the integration
            self.dth=dth #dtheta for the integration
            print("\n")
            print("differentials, dr=",self.dr ,"  dth=",self.dth)
            print("\n")

    def __repr__(self):
        return "self energy  T={T}".format(T=self.T)


    ###################
    # INTEGRANDS FOR PARALLEL RUNS
    ###################

    def integrand_par_w2(self,qp,ds,w):
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
    
    def integrand_par_w(self,qp,ds,w):
        si=time.time()
        qx,qy=qp[0], qp[1]

        edd=self.ed.Disp_mu(self.kxsq,self.kysq)
        om=w-edd
        # fac_p=(1+np.exp(-w/self.T))*(1-self.ed.nf(edd, self.T))
        fac_p=(self.ed.be_nb(om, self.T)+self.ed.nf(-edd, self.T)*om/self.T)


        SFvar=self.SS.Dynamical_SF(self.kxsq-qx,self.kysq-qy,om)

        
        # fac_p=ed.nb(w-edd, T)+ed.nf(-edd, T)
        Integrand=self.Kcou*self.Kcou*SFvar*2*np.pi*fac_p*self.kmag
        S0=np.sum(Integrand*self.dth*self.dr)
        # Vol_rec=self.latt.Vol_BZ()
        dels=10*ds*np.max(np.abs(np.diff(Integrand)))#np.sqrt(ds/Vol_rec)*Vol_rec#*np.max(np.abs(np.diff(Integrand)))*0.1

        ang=np.arctan2(qy,qx)
        ei=time.time()
        print(ei-si," seconds ",qx, qy, w)

        return S0, w,dels

    def integrand_par_w_sq(self,qp,ds,w):
        si=time.time()
        qx,qy=qp[0], qp[1]

        edd=self.ed.Disp_mu(self.kxsq,self.kysq)
        om=w-edd
        # fac_p=(1+np.exp(-w/self.T))*(1-self.ed.nf(edd, self.T))
        fac_p=(self.ed.be_nb(om, self.T)+self.ed.nf(-edd, self.T)*om/self.T)


        SFvar=self.SS.Dynamical_SF(self.kxsq-qx,self.kysq-qy,om)

        
        # fac_p=ed.nb(w-edd, T)+ed.nf(-edd, T)
        Integrand=self.Kcou*self.Kcou*SFvar*2*np.pi*fac_p
        S0=np.sum(Integrand*ds)
        # Vol_rec=self.latt.Vol_BZ()
        dels=10*ds*np.max(np.abs(np.diff(Integrand)))#np.sqrt(ds/Vol_rec)*Vol_rec#*np.max(np.abs(np.diff(Integrand)))*0.1

        ang=np.arctan2(qy,qx)
        ei=time.time()
        print(ei-si," seconds ",qx, qy, w)

        return S0, w,dels


    ###################
    # PLOTTING ROUTINES
    ###################
    
    def integrand(self,kx,ky,qx,qy,w):


        edd=self.ed.Disp_mu(kx,ky)
        om=w-edd
        fac_p=(1+np.exp(-w/self.T))*(1-self.ed.nf(edd, self.T))

        SFvar=self.SS.Dynamical_SF(kx-qx,ky-qy,om)
        Integrand=self.Kcou*self.Kcou*SFvar*2*np.pi*fac_p
    
    
        return Integrand


    def plot_integrand(self,qx,qy,f):
        Vertices_list, Gamma, K, Kp, M, Mp=self.latt.FBZ_points(self.latt.b[0,:],self.latt.b[1,:])
        VV=np.array(Vertices_list+[Vertices_list[0]])
        Integrand=self.integrand(self.kx,self.ky,qx,qy,f)
        print("for error, maximum difference", np.max(np.diff(Integrand)))
        plt.plot(VV[:,0], VV[:,1], c='k')
        wh=np.where(Integrand>1e-5)
        print('number of active points', np.shape(wh))
        plt.scatter(self.kx[wh],self.ky[wh],c=Integrand[wh], s=1, zorder=1)
        plt.colorbar()
        plt.scatter(self.kx,self.ky, s=1, zorder=0)
        # plt.scatter(self.qxFS,self.qyFS, c='r',s=0.05)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(f"integrand_{qx}_{qy}_{f}_q.png", dpi=400)
        
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
        print('number of active points', np.shape(wh))
        # plt.scatter(self.kx[wh],self.ky[wh],c=xx[wh], s=1)
        # # plt.clim(-2,np.max(np.log10(Integrand)))
        # plt.colorbar()
        # # plt.scatter(self.qxFS,self.qyFS, c='r',s=0.05)
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.savefig(f"log_integrand_{qx}_{qy}_{f}_q.png", dpi=400)
        # plt.close()
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

    def get_perp_q(self, qx,qy, Npoints_q, cutoff):
        
        kloc=np.array([qx,qy])
        vf=self.ed.Fermi_Vel(qx,qy)
        [vfx,vfy]=vf
        VF=np.sqrt(vfx**2+vfy**2)
        KF=np.sqrt(kloc@kloc)
        amp=KF/cutoff #cutoff=10 is a good value
        fac=amp/VF
        mesh=np.linspace(-fac,fac,Npoints_q)
        QX=mesh*vfx
        QY=mesh*vfy

        
        return [QX,QY]

    ##############
    #   PARALLEL RUNS WITH FREQUENCY DEPENDENCE
    ##############


    def parInt_w(self, qx, qy, w, sq,maxthreads):

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
        Nthreads=4 #int(Npoints_w/maxthreads)

        partial_integ = functools.partial(self.integrand_par_w, qp, ds)

        print("starting with calculation of Sigma")
        s=time.time()

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
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


    def gen_df(self, arg, J, theta, fill, tp1,tp2,QX,QY, prefixd):
    
        [qx,qy]=self.get_KF(theta)
        Q=np.sqrt(QX**2+QY**2) #should be multiplied by a sign, do this in post processing to avoid errors here
        dispname=self.ed.name
        SFname=self.SS.name+"_theta_"+str(round(theta*180/np.pi, 2))
        [shifts, w, delsd]=arg
        SEarr=np.array(shifts)
        err_arr=np.array(delsd)
        
        
        if prefixd!="":
            df = pd.DataFrame({'theta': theta, "freq":w , 'SE':SEarr, 'error': err_arr, 'KFX': qx, 'KFY': qy,'QFX': QX, 'QFY': QY, "Q":Q, 'T': self.T, \
                'nu': fill,'intP':self.Npoints_int_pre, 'FS_point': self.NpointsFS, 'dispname': dispname, "t1":tp1, "t2":tp2, 'SFname': SFname, 'J':J, 'extr':prefixd})
            
        else:
            
            df = pd.DataFrame({'theta': theta, "freq":w , 'SE':SEarr, 'error': err_arr, 'KFX': qx, 'KFY': qy, 'QFX': QX, 'QFY': QY, "Q":Q, 'T': self.T, \
                'nu': fill,'intP':self.Npoints_int_pre, 'FS_point': self.NpointsFS, 'dispname': dispname, "t1":tp1, "t2":tp2, 'SFname': SFname, 'J':J})
        return df
        
        



def main() -> int:
    
    
    try:
        index_sf=int(sys.argv[1])

    except (ValueError, IndexError):
        raise Exception("Input integer in the firs argument to choose structure factor")


    try:
        N_SFs=12 #number of SF's currently implemented
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
            maxthreads=10
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
    Npoints_int_pre, NpointsFS_pre=2000,600
    save=True
    l=Lattice.TriangLattice(Npoints_int_pre, save,Machine)
    Vol_rec=l.Vol_BZ()
    l2=Lattice.SQLattice(Npoints, save,Machine)
    Vol_rec2=l2.Vol_BZ()
    
    
    
    # ##########################
    # ##########################
    # # Fermi surface and structure factor
    # ##########################
    # ##########################

    ed=Dispersion.Dispersion_TB_single_band([tp1,tp2],fill,Machine)
    
    # ed=Dispersion.Dispersion_circ([tp1,tp2],fill)
    # [KxFS,KyFS]=ed.FS_contour(NpointsFS_pre)
    # NsizeFS=np.size(KxFS)
    
    
    # [KxFS2,KyFS2]=ed.FS_contour2(NpointsFS_pre)
    # plt.scatter(KxFS,KyFS, c=np.log10(np.abs(ed.Disp_mu(KxFS,KyFS))+1e-34) )
    # f=np.log10(np.abs(ed.Disp_mu(KxFS2,KyFS2))+1e-34)

    # plt.scatter(KxFS2,KyFS2, c=f )
    # plt.colorbar()
    # plt.savefig("FS_ene.png")
    # plt.close()
    # plt.show()
    # print(f"dispersion params: {tp1} \t {tp2}")
    # # ed.PlotFS(l)
    

    #parameters for structure factors
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
        [KX,KY]=l.read_lattice(option='sq')
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
        Npoints_diff=1000
        latt_dif=Lattice.TriangLattice(Npoints_diff, save,Machine)
        SS=StructureFactor.StructureFac_fit_no_diff_peak(T,latt_dif)
    elif index_sf==6:
        SS=StructureFactor.MD_SF(T)
    elif index_sf==7:
        [KX,KY]=l.read_lattice(option='sq')
        SS=StructureFactor.Langevin_SF(T, KX, KY)
    elif index_sf==8:
        Npoints_diff=1000
        latt_dif=Lattice.TriangLattice(Npoints_diff, save,Machine)
        SS=StructureFactor.StructureFac_diff_peak_fit(T,latt_dif)
    elif index_sf==9:
        SS=StructureFactor.SF_diff_peak(T, D, C)
    elif index_sf==10:
        part=mod
        Npoints_diff=1000
        latt_dif=Lattice.TriangLattice(Npoints_diff, save,Machine)
        SS=StructureFactor.StructureFac_fit_no_diff_peak_partial_subs(T,part, latt_dif)
    else:
        cut=1.5
        Npoints_diff=1000
        latt_dif=Lattice.TriangLattice(Npoints_diff, save,Machine)
        SS=StructureFactor.StructureFac_fit_no_diff_peak_cut(T,cut, latt_dif)



    # plt.scatter(KX,KY,c=SS.Dynamical_SF(KX,KY,0.1), s=0.5)
    # plt.colorbar()
    # pl.show()
    
    Momentum_cut=SS.momentum_cut_high_symmetry_path(l, 2000, 1000)

    ##########################
    ##########################
    # Calls to integration routine
    ##########################
    ##########################

    SE=SelfE(T ,ed ,SS,  Npoints_int_pre, NpointsFS_pre, Kcou, "ed", Machine)  

    ##################
    # integration accross frequencies for fixed FS Point
    #################
    
    
    
    thetas= np.linspace(-4*np.pi/6, -5*np.pi/6, 6)
    dfs=[]
    for theta in thetas:
        
        [qx,qy]=SE.get_KF(theta)
        Npoints_q=10
        cutoff=10.0
        [QX,QY]=SE.get_perp_q(qx,qy, Npoints_q, cutoff)
        
        for j in range( Npoints_q):
            domeg=1
            maxw=15 #np.min([5*T,20]) #in unitsw of J
            w=np.arange(0,maxw,domeg)
            sq=True
            # SE.plot_integrand(qx+QX[j],qy+QY[j], 0)
            # SE.plot_logintegrand(qx+QX[j],qy+QY[j], 0)
            # SE.plot_integrand(qx+QX[j],qy+QY[j], 15)
            # SE.plot_logintegrand(qx+QX[j],qy+QY[j], 15)
            [shifts, w, delsd]=SE.parInt_w( qx+QX[j],qy+QY[j], w, Machine, sq)
            shifts=shifts*J
            delsd=delsd*J
            w=J*w
            df=SE.gen_df( [shifts, w, delsd], J, theta, fill, tp1, tp2 ,QX[j],QY[j], "")
            dfs.append(df)
        
    df_fin=pd.concat(dfs)
    iden=datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    df_fin.to_hdf('data_ME_'+iden+'.h5', key='df', mode='w')

    
    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
