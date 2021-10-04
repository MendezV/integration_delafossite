import numpy as np
import time
import Lattice

#Hack for now, search for: choose dispersion wherever we want to change dispersion
class Dispersion_single_band:

    def __init__(self, hop, mu):
        self.hop=hop
        self.mu=mu

        #GRIDS AND INTEGRATION MEASURES
        print("started calculating filling for chemical potential and dispersion parameters..")

        self.Npoi_ints=1200
        self.latt_int=Lattice.TriangLattice(self.Npoi_ints, True) #temp grid for integrating and getting filling
        
        # [KX,KY]=l.Generate_lattice()
        [KX,KY]=self.latt_int.read_lattice()
        Vol_rec=self.latt_int.Vol_BZ()
        ds=Vol_rec/np.size(KX)

        #choose dispersion
        energy_k = self.Disp_second_NN_triang(KX,KY)
        energy_k_mu=self.Disp_mu_second_NN_triang(KX,KY)
        Wbdw=np.max(energy_k)-np.min(energy_k)

        #DISPERSION PARAMS
        self.bandmin=np.min(energy_k)
        self.bandmax=np.max(energy_k)
        self.bandwidth=Wbdw
        self.EF= self.mu-self.bandmin #fermi energy from the bottom of the band

        nu_fill=np.sum(np.heaviside(-energy_k_mu,1)*ds)/Vol_rec
        print("finished calculating filling for chemical potential")
        self.filling=nu_fill

    def Disp_second_NN_triang(self,kx,ky):
        [tp1,tp2]=self.hop
        ed=-tp1*(2*np.cos(kx)+4*np.cos((kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
        ed=ed-tp2*(2*np.cos(np.sqrt(3)*(ky))+4*np.cos(3*(kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
        return ed

    def Disp_mu_second_NN_triang(self,kx,ky):
        [tp1,tp2]=self.hop
        ed=-tp1*(2*np.cos(kx)+4*np.cos((kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
        ed=ed-tp2*(2*np.cos(np.sqrt(3)*(ky))+4*np.cos(3*(kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
        return ed-self.mu

    def Fermi_Vel_second_NN_triang(self,kx,ky):
        [tp1,tp2]=self.hop
        sq3x2=np.sqrt(3)*kx/2
        sq3y2=np.sqrt(3)*ky/2
        sq3y=np.sqrt(3)*ky
        vx=-tp1*(-2*np.cos(sq3y2)*np.sin(kx/2)-2*np.sin(kx)) +6*tp2*np.cos(sq3y2)*np.sin(3*kx/2)
        vy=2*np.sqrt(3)*tp1*np.cos(kx/2)*np.sin(sq3y2)-2*np.sqrt(3)*tp2*(-np.cos(3*kx/2)*np.sin(sq3y2)-np.sin(sq3y))
        return [vx,vy]

    def FS_contour(self, Np):
        c= plt.contour(X, Y, Z, levels=[0],linewidths=3, cmap='summer');
        #plt.show()
        numcont=np.shape(c.collections[0].get_paths())[0]
        if numcont==1:
            v = c.collections[0].get_paths()[0].vertices
        else:
            contourchoose=0
            v = c.collections[0].get_paths()[0].vertices
            sizecontour_prev=np.prod(np.shape(v))
            for ind in range(1,numcont):
                v = c.collections[0].get_paths()[ind].vertices
                sizecontour=np.prod(np.shape(v))
                if sizecontour>sizecontour_prev:
                    contourchoose=ind
            v = c.collections[0].get_paths()[contourchoose].vertices
        NFSpoints=4000
        xFS_dense = v[::int(np.size(v[:,1])/NFSpoints),0]
        yFS_dense = v[::int(np.size(v[:,1])/NFSpoints),1]
        return [xFS_dense,yFS_dense]
    #TODO: circular fermi surface,
    def deltad(self,x, epsil):
        return (1/(np.pi*epsil))/(1+(x/epsil)**2)

    def DOS(self,size_E, Npoi_ints):

        #DOMAIN OF THE DOS
        minE=self.bandmin-0.001*self.bandwidth
        maxE=self.bandmax+0.001*self.bandwidth
        earr=np.linspace(minE,maxE,size_E)

        #INTEGRATION LATTICE
        latt_int=Lattice.TriangLattice(Npoi_ints, False) #temp grid for integrating and getting filling
        
        # [KX,KY]=l.Generate_lattice()
        [KX,KY]=latt_int.read_lattice()
        Vol_rec=latt_int.Vol_BZ()
        ds=Vol_rec/np.size(KX)

        #DISPERSION FOR INTEGRAL: choose dispersion
        energy_k = self.Disp_second_NN_triang(KX,KY)
        #parameter for delta func approximation
        epsil=0.002*self.bandwidth

        ##DOS 
        Dos=[]
        for i in earr:
            dosi=np.sum(self.deltad(energy_k-i,epsil))*ds
            Dos.append(dosi)

        #FILLING FOR EACH CHEMICAL POTENTIAL
        ndens=[]
        for mu_ind in range(size_E):
            de=earr[1]-earr[0]
            N=np.trapz(Dos[0:mu_ind])*de
            ndens.append(N)
        nn=np.array(ndens)
        nn=nn/nn[-1]

        return [nn,earr,Dos]