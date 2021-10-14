import numpy as np
import Lattice
import StructureFactor
import Dispersion
import matplotlib.pyplot as plt
import time
import sys


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

    def __init__(self, T ,ed ,SS,  Npoints_int, NpointsFS_pre ):
        self.T=T
        self.ed=ed #dispersion
        self.SS=SS #structure factor
        save=False
        self.latt=Lattice.TriangLattice(Npoints_int, save ) #integration lattice 
        self.Vol_rec=self.latt.Vol_BZ()

        [KX,KY]=self.latt.read_lattice()
        self.KX=KX
        self.KY=KY

        self.params=self.SS.params_fit(KX, KY)
        self.SF_stat=self.SS.Static_SF(KX, KY)

        [qxFS,qyFS]=ed.FS_contour( NpointsFS_pre)
        self.qxFS=qxFS
        self.qyFS=qyFS

        self.NpointsFS=np.size(qxFS)

        

        
    def __repr__(self):
        return "Structure factorat T={T}".format(T=self.T)
    

    def integrand_de(self,kx,ky,qx,qy):
        # edd=ed.Disp_mu_circ(kx+qx,ky+qy)
        edd=self.ed.Disp_mu_second_NN_triang(kx+qx,ky+qy)
        epsil=0.002*self.ed.bandwidth
        inn=self.ed.deltad(edd, epsil)
        # om2=-ed

        return inn

    def integrand_Disp_fit(self,kx,ky,qx,qy,w, Kcou):

        # edd=ed.Disp_mu_circ(kx+qx,ky+qy)
        edd=self.ed.Disp_mu_second_NN_triang(kx+qx,ky+qy)
        om=w-edd
        # om2=-ed

        # SFvar=1#/(1+ed.nb(w-edd, T))
        SFvar=self.SS.Dynamical_SF_fit( om, self.params , self.SF_stat)

        fac_p=(1+np.exp(-w/self.T))*(1-self.ed.nf(edd, self.T))
        # fac_p=ed.nb(w-edd, T)+ed.nf(-edd, T)
        return Kcou*Kcou*SFvar*2*np.pi*fac_p

    def integrand_Disp_fit_2(self,kx,ky,qx,qy,w, Kcou):

        edd=self.ed.Disp_mu_second_NN_triang(kx+qx,ky+qy)
        om=w-edd
        SFvar=self.SS.Dynamical_SF_fit_2( kx, ky, om)

        fac_p=(1+np.exp(-w/self.T))*(1-self.ed.nf(edd, self.T))
        # fac_p=ed.nb(w-edd, T)+ed.nf(-edd, T)
        return Kcou*Kcou*SFvar*2*np.pi*fac_p

    def integrand_Disp_chi(self,kx, ky, qx, qy, w, gamma, vmode, m, Kcou):

        # edd=ed.Disp_mu_circ(kx+qx,ky+qy)
        edd=self.ed.Disp_mu_second_NN_triang(kx+qx,ky+qy)
        om=w-edd
        # om2=-ed

        # SFvar=1#/(1+ed.nb(w-edd, T))
        SFvar=self.SS.Dynamical_SF_PM_zeroQ(ky,kx, om, gamma, vmode, m)
        # SFvar=SS.Dynamical_SF_PM_Q(ky,kx, om, gamma, vmode, m)

        fac_p=(1+np.exp(-w/self.T))*(1-self.ed.nf(edd, self.T))
        # fac_p=ed.nb(w-edd, T)+ed.nf(-edd, T)
        return Kcou*Kcou*SFvar*2*np.pi*fac_p

    def Int_FS(self):

        shifts=[]
        shifts2=[]
        angles=[]
        delsd=[]

        print("starting with calculation of Sigma")
        s=time.time()
        for ell in range(np.size(xFS_dense)):

            qx=self.qxFS[ell]
            qy=self.qyFS[ell]

            ds=Vol_rec/np.size(KX)
            # S01=np.sum(integrand_Disp_chi(KX,KY,qx,qy,0.0, T, gamma, vmode, m)*ds)
            # S01=np.sum(integrand_Disp_fit(KX,KY,qx,qy,0.01, T, params , SF_stat)*ds)
            S01=np.sum(self.integrand(self.KX,self.KY,qx,qy,0.01, self.T, self.params , self.SF_stat)*ds)
            dels=np.sqrt(ds)
            shifts.append(S0)
            shifts2.append(S01)
            delsd.append(dels)
            # shifts2.append(S1)
            angles.append(np.arctan2(qy,qx))
            print(S0, dels,np.arctan2(qy,qx),S01)
            # print(ell, np.arctan2(KFy,KFx), S0)
            # printProgressBar(ell + 1, np.size(xFS_dense), prefix = 'Progress:', suffix = 'Complete', length = 50)


        e=time.time()
        print("time for calc....",e-s)

def main() -> int:
    Npoints=100
    save=True
    l=Lattice.TriangLattice(Npoints, save )
    Vol_rec=l.Vol_BZ()

    [KX,KY]=l.read_lattice()

    Vertices_list, Gamma, K, Kp, M, Mp=l.FBZ_points(l.b[0,:],l.b[1,:])
    T=1.0
    SS=StructureFactor.StructureFac(T)


    ###params quasicircular FS
    J=2*5.17 #in mev
    tp1=568/J #in units of Js
    tp2=0.065*tp1
    ##coupling 
    U=4000/J
    g=100/J
    Kcou=g*g/U
    fill=0.311
    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
