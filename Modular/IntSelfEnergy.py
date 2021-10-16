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

    def __init__(self, T ,ed ,SS,  Npoints_int_pre, NpointsFS_pre ,Kcou):
        self.T=T
        self.ed=ed #dispersion
        self.SS=SS #structure factor
        save=False
        self.latt=Lattice.TriangLattice(Npoints_int_pre, save ) #integration lattice 
        self.Kcou=Kcou

        [qxFS,qyFS]=ed.FS_contour( NpointsFS_pre)
        self.qxFS=qxFS
        self.qyFS=qyFS

        self.NpointsFS=np.size(qxFS)

    def __repr__(self):
        return "Structure factorat T={T}".format(T=self.T)
    

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

    def integrand_par(self,kx,ky,w,ds,qp):
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

        return S0, ang,dels

    def plot_integrand(self,qx,qy,f):
        [KX,KY]=self.latt.read_lattice()
        Integrand=self.integrand(KX,KY,qx,qy,f)
        print("for error, maximum difference", np.max(np.diff(Integrand)))
        plt.scatter(KX,KY,c=Integrand, s=1)
        plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        return 0
    def plot_logintegrand(self,qx,qy,f):
        [KX,KY]=self.latt.read_lattice()
        Integrand=self.integrand(KX,KY,qx,qy,f)
        print("for error, maximum difference", np.max(np.diff(Integrand)))
        plt.scatter(KX,KY,c=np.log10(Integrand), s=1)
        plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        return 0
    def Int_FS_nofreq(self):
        Vol_rec=self.latt.Vol_BZ()
        [KX,KY]=self.latt.read_lattice()
        Npoints_int=np.shape(KX)
        shifts=[]
        angles=[]
        delsd=[]

        ds=Vol_rec/Npoints_int

        print("starting with calculation of Sigma")
        s=time.time()
        for ell in range(self.NpointsFS):

            qx=self.qxFS[ell]
            qy=self.qyFS[ell]

            Integrand=self.integrand(KX,KY,qx,qy,0.0)

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
        [KX,KY]=self.latt.read_lattice(sq=1)
        Npoints_int=np.shape(KX)
        shifts=[]
        angles=[]
        delsd=[]

        ds=Vol_rec/Npoints_int

        print("starting with calculation of Sigma")
        s=time.time()
        for ell in range(self.NpointsFS):

            qx=self.qxFS[ell]
            qy=self.qyFS[ell]

            Integrand=self.integrand(KX,KY,qx,qy,0.0)

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

    def parInt_FS_nofreq(self):
        Mac_maxthreads=8
        Desk_maxthreads=12

        Vol_rec=self.latt.Vol_BZ()
        [kx,ky]=self.latt.read_lattice()
        Npoints_int=np.size(kx)
        shifts=[]
        angles=[]
        delsd=[]

        ds=Vol_rec/Npoints_int
        qp=np.array([self.qxFS, self.qyFS]).T
        w=0

        partial_integ = functools.partial(self.integrand_par, kx,ky,w,ds)

        print("starting with calculation of Sigma")
        s=time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(partial_integ, qp, chunksize=int(np.size(qp)/Mac_maxthreads))

            for result in results:
                shifts.append(result[0])
                angles.append(result[1])
                delsd.append(result[2])

        e=time.time()
        print("time for calc....",e-s)

        shifts=np.array(shifts) 
        angles=np.array(angles)
        delsd=np.array(delsd)

        return [shifts, angles, delsd]

    def parInt_FS_nofreq_sq(self):
        Mac_maxthreads=8
        Desk_maxthreads=12

        Vol_rec=self.latt.Vol_BZ()
        [kx,ky]=self.latt.read_lattice(sq=1)
        Npoints_int=np.size(kx)
        shifts=[]
        angles=[]
        delsd=[]

        ds=Vol_rec/Npoints_int
        qp=np.array([self.qxFS, self.qyFS]).T
        w=0

        partial_integ = functools.partial(self.integrand_par, kx,ky,w,ds)

        print("starting with calculation of Sigma")
        s=time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(partial_integ, qp, chunksize=int(np.size(qp)/Mac_maxthreads))

            for result in results:
                shifts.append(result[0])
                angles.append(result[1])
                delsd.append(result[2])

        e=time.time()
        print("time for calc....",e-s)

        shifts=np.array(shifts) 
        angles=np.array(angles)

        return [shifts, angles, delsd]

def main() -> int:
    
    
    try:
        index_sf=int(sys.argv[1])

    except (ValueError, IndexError):
        raise Exception("Input integer in the firs argument to choose structure factor")


    try:
        N_SFs=5 #number of SF's currently implemented
        a=np.arange(N_SFs)
        a[index_sf]

    except (IndexError):
        raise Exception(f"Index has to be between 0 and {N_SFs-1}")

    ##########################
    ##########################
    # parameters
    ##########################
    ##########################

    # #electronic parameters
    J=2*5.17*40 #in mev
    tp1=568/J #in units of Js\
    tp2=-tp1*108/568 #/tpp1
    ##coupling 
    U=4000/J
    g=100/J
    Kcou=g*g/U
    # fill=0.67 #van hove
    fill=0.5

    #rotated
    # J=2*5.17 #in mev
    # tp1=568/J #in units of Js\
    # tp2=tp1*0.258 #/tpp1
    # ##coupling 
    # U=4000/J
    # g=100/J
    # Kcou=g*g/U
    # # fill=0.67 #van hove
    # fill=0.35

    ###params quasicircular and circular FS
    J=2*5.17 #in mev
    tp1=568/J #in units of Js
    tp2=0.065*tp1
    ##coupling 
    U=4000/J
    g=100/J
    Kcou=g*g/U
    fill=0.1111

    ##########################
    ##########################
    # Geometry/Lattice
    ##########################
    ##########################

    Npoints=100
    Npoints_int_pre, NpointsFS_pre=1000,400
    save=True
    l=Lattice.TriangLattice(Npoints, save )
    Vol_rec=l.Vol_BZ()
    [KX,KY]=l.read_lattice(sq=1)
    # [KX,KY]=l.Generate_lattice_SQ()
    Vertices_list, Gamma, K, Kp, M, Mp=l.FBZ_points(l.b[0,:],l.b[1,:])
    VV=np.array(Vertices_list+[Vertices_list[0]])
    
    ##########################
    ##########################
    # Fermi surface and structure factor
    ##########################
    ##########################

    # ed=Dispersion.Dispersion_TB_single_band([tp1,tp2],fill)
    ed=Dispersion.Dispersion_circ([tp1,tp2],fill)

    ed.PlotFS(l)
    [KxFS,KyFS]=ed.FS_contour(NpointsFS_pre)
    

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
    T=1
    SS1=StructureFactor.StructureFac_fit(T,KX, KY)
    # SF_stat=SS.Static_SF()
    SS2=StructureFactor.StructureFac_fit_F(T)
    # SF_stat=SS.Static_SF(KX,KY)
    SS3=StructureFactor.StructureFac_PM(T, gamma, vmode, m )
    SS4=StructureFactor.StructureFac_PM_Q(T, gamma, vmode, m )
    SS5=StructureFactor.StructureFac_PM_Q2(T, gamma, vmode, m )
    SSarr=[SS1,SS2,SS3,SS4,SS5]
    
    SS=SSarr[index_sf]
    plt.scatter(KX,KY,c=SS.Dynamical_SF(KX,KY,0.1), s=0.5)
    plt.colorbar()
    plt.show()

    ##########################
    ##########################
    # Calls to integration routine
    ##########################
    ##########################

    #TODO: quadrature useful and decrease BW progressively

    #TODO: -- frequency dependence
    #TODO: -- temperaure dependence

    SE=SelfE(T ,ed ,SS,  Npoints_int_pre, NpointsFS_pre, Kcou)  #Fits
    # SE=SelfE(T ,ed ,SS,  Npoints_int_pre, NpointsFS_pre, gcoupl)  #paramag
    [shifts, angles, delsd]=SE.parInt_FS_nofreq()

    #converting to meV 
    shifts=shifts*J
    delsd=delsd*J

    SE.plot_integrand(KxFS[0],KyFS[0],0.01)
    SE.plot_logintegrand(KxFS[0],KyFS[0],0.01)
    plt.errorbar(angles,shifts,yerr=delsd, fmt='.')
    plt.scatter(angles,shifts, s=1, c='r')
    plt.show()

    plt.scatter(angles,shifts, s=1, c='r')
    plt.show()


    plt.plot(VV[:,0], VV[:,1], c='k')
    plt.scatter([0],[0], c='k', s=1)
    plt.scatter(KxFS,KyFS,c=shifts)
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
