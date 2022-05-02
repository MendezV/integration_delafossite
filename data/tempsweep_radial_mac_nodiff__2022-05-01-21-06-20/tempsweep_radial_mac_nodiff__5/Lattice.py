from tkinter import N, NS
import numpy as np
import scipy
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import linalg as la
import time
import matplotlib.pyplot as plt
 
class TriangLattice:

    def __init__(self, Npoints, save, Machine=None):

        self.Npoints = Npoints
        self.a =np.array([[1,0],[1/2,np.sqrt(3)/2]])  #original graphene lattice vectors: rows are basis vectors
        self.b =(2*np.pi)*np.array([[1,-1/np.sqrt(3)],[0,2/np.sqrt(3)]]) # original graphene reciprocal lattice vectors : rows are basis vectors
        self.save=save
        self.dir=dir
        #some symmetries:
        #C2z
        th1=np.pi
        self.C2z=np.array([[np.cos(th1),np.sin(th1)],[-np.sin(th1),np.cos(th1)]]) #rotation matrix 
        #C3z
        th1=2*np.pi/3
        self.C3z=np.array([[np.cos(th1),np.sin(th1)],[-np.sin(th1),np.cos(th1)]]) #rotation matrix 
        #C2x inv
        self.C2x=np.array([[1,0],[0,-1]]) #rotation matrix 
        
        if Machine is None:
            self.Machine = ""
        else:
            self.Machine = Machine 

        if self.Machine=='FMAC':
            self.lattdir="/Users/juanmendezvalderrama/Documents/Proyectos/Delafossites/Lattices/"
        elif self.Machine=='CH1':
            self.lattdir="/home/jfm343/Documents/Delafossites/Lattices/"
        elif self.Machine=='UBU':
            self.lattdir="/home/juan/Documents/Projects/Delafossites/Lattices/"
        else:
            self.lattdir="../../../Lattices/"
        
        print("Machine arg is,", self.Machine)

        self.VolBZ=self.Vol_BZ()



    def __repr__(self):
        return "lattice( LX={w}, reciprocal lattice={c})".format(h=self.Npoints, c=self.b)

    # Print iterations progress
    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
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
    #FBZ volume
    def Vol_BZ(self):
        zhat=np.array([0,0,1])
        b_1=np.array([self.b[0,0],self.b[0,1],0]) # Moire reciprocal lattice vect extended
        b_2=np.array([self.b[1,0],self.b[1,1],0]) # Moire reciprocal lattice vect extended
        Vol_rec=np.cross(b_1,b_2)@zhat
        return Vol_rec

    #hexagon where the pointy side is up
    def hexagon1(self,pos,Radius_inscribed_hex):
        y,x = map(abs, pos) #taking the absolute value of the rotated hexagon, only first quadrant matters #effective rotation
        return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge
    
    #hexagon where the flat side is up
    def hexagon2(self,pos,Radius_inscribed_hex):
        x,y = map(abs, pos) #taking the absolute value of the rotated hexagon, only first quadrant matters
        return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge

    #gets high symmetry points
    def FBZ_points(self,b_1,b_2):
        #creating reciprocal lattice
        Np=4
        n1=np.arange(-Np,Np+1)
        n2=np.arange(-Np,Np+1)
        Recip_lat=[]
        for i in n1:
            for j in n2:
                point=b_1*i+b_2*j
                Recip_lat.append(point)

        #getting the nearest neighbours to the gamma point
        Recip_lat_arr=np.array(Recip_lat)
        dist=np.round(np.sqrt(np.sum(Recip_lat_arr**2, axis=1)),decimals=10)
        sorted_dist=np.sort(list(set(dist)) )
        points=Recip_lat_arr[np.where(dist<sorted_dist[2])[0]]

        #getting the voronoi decomposition of the gamma point and the nearest neighbours
        vor = Voronoi(points)
        Vertices=(vor.vertices)

        #ordering the points counterclockwise in the -pi,pi range
        angles_list=list(np.arctan2(Vertices[:,1],Vertices[:,0]))
        Vertices_list=list(Vertices)

        # joint sorting the two lists for angles and vertices for convenience later.
        # the linear plot routine requires the points to be in order
        # atan2 takes into acount quadrant to get the sign of the angle
        angles_list, Vertices_list = (list(t) for t in zip(*sorted(zip(angles_list, Vertices_list))))

        ##getting the M points as the average of consecutive K- Kp points
        Edges_list=[]
        for i in range(len(Vertices_list)):
            Edges_list.append([(Vertices_list[i][0]+Vertices_list[i-1][0])/2,(Vertices_list[i][1]+Vertices_list[i-1][1])/2])

        Gamma=[0,0]
        K=Vertices_list[0::2]
        Kp=Vertices_list[1::2]
        M=Edges_list[0::2]
        Mp=Edges_list[1::2]

        return Vertices_list, Gamma, K, Kp, M, Mp
    
    def boundary(self):
        Vertices_list, Gamma, K, Kp, M, Mp=self.FBZ_points(self.b[0,:],self.b[1,:])
        VV=Vertices_list+[Vertices_list[0]]
        vx=np.array(VV)[:,0]
        vy=np.array(VV)[:,1]
        return [vx,vy]

    #same as Generate lattice but for the original graphene (FBZ of triangular lattice)
    def Generate_lattice(self):

        Vertices_list, Gamma, K, Kp, M, Mp=self.FBZ_points(self.b[0,:],self.b[1,:])

        k_window_sizey = K[2][1] 
        k_window_sizex = K[1][0] 

        #will filter points that are in a hexagon inscribed in a circle of radius Radius_inscribed_hex
        Radius_inscribed_hex=1.0000005*k_window_sizex


        print("starting sampling in reciprocal space....")
        s=time.time()

        #initial grid that will be filtered
        LP=self.Npoints
        nn1=np.arange(-LP,LP+1,1)
        nn2=np.arange(-LP,LP+1,1)

        nn_1,nn_2=np.meshgrid(nn1,nn2)

        nn_1p=[]
        nn_2p=[]
        sz=np.size(nn1)*np.size(nn2)
        sz1=np.size(nn1)
        sz2=np.size(nn2)
        x1,y1=[0,0]
        for x in nn1:
            x1=x1+1
            y1=0
            for y in nn2:
                y1=y1+1
                kx=(2*np.pi*x/LP)
                ky=(2*(2*np.pi*y/LP - np.pi*x/LP)/np.sqrt(3))
                if self.hexagon2( ( kx, ky), Radius_inscribed_hex ):
                    nn_1p.append(x)
                    nn_2p.append(y)
                self.printProgressBar((sz2-1)*x1+y1 , sz, prefix = 'Progress:', suffix = 'Complete', length = 50)


        e=time.time()
        print("finished sampling in reciprocal space....t=",e-s," s")

        nn_1pp=np.array(nn_1p)
        nn_2pp=np.array(nn_2p)

        KX=(2*np.pi*nn_1pp/LP)
        KY= (2*(2*np.pi*nn_2pp/LP - np.pi*nn_1pp/LP)/np.sqrt(3))

        #Making the sampling lattice commensurate with the MBZ
        fact=K[1][0]/np.max(KX)
        KX=KX*fact
        KY=KY*fact
        if self.save==True:
            with open(self.lattdir+"KgridX"+str(self.Npoints)+".npy", 'wb') as f:
                np.save(f, KX)
            with open(self.lattdir+"KgridY"+str(self.Npoints)+".npy", 'wb') as f:
                np.save(f, KY)
        
        return [KX,KY]

    
    def Generate_lattice_SQ(self):

        Vertices_list, Gamma, K, Kp, M, Mp=self.FBZ_points(self.b[0,:],self.b[1,:])

        k_window_sizey = K[2][1] 
        k_window_sizex = K[1][0] 

        #will filter points that are in a hexagon inscribed in a circle of radius Radius_inscribed_hex
        Radius_inscribed_hex=1.0000005*k_window_sizex


        print("starting sampling in reciprocal space....")
        s=time.time()

        #initial grid that will be filtered
        LP=self.Npoints
        nn1=k_window_sizex*np.arange(-LP,LP+1,1)/LP
        nn2=k_window_sizey*np.arange(-LP,LP+1,1)/LP

        nn_1,nn_2=np.meshgrid(nn1,nn2)

        nn_1p=[]
        nn_2p=[]
        sz=np.size(nn1)*np.size(nn2)
        sz1=np.size(nn1)
        sz2=np.size(nn2)
        x1,y1=[0,0]
        for x in nn1:
            x1=x1+1
            y1=0
            for y in nn2:
                y1=y1+1
                kx=x
                ky=y
                if self.hexagon2( ( kx, ky), Radius_inscribed_hex ):
                    nn_1p.append(x)
                    nn_2p.append(y)
                self.printProgressBar((sz2-1)*x1+y1 , sz, prefix = 'Progress:', suffix = 'Complete', length = 50)


        e=time.time()
        print("finished sampling in reciprocal space....t=",e-s," s")

        KX=np.array(nn_1p)
        KY=np.array(nn_2p)

        if self.save==True:
            with open(self.lattdir+"sqKgridX"+str(self.Npoints)+".npy", 'wb') as f:
                np.save(f, KX)
            with open(self.lattdir+"sqKgridY"+str(self.Npoints)+".npy", 'wb') as f:
                np.save(f, KY)
        
        return [KX,KY]
    
    def Generate_lattice_ed(self, ed, Npoints_q,NpointsFS_pre):
        print("ED starting sampling in reciprocal space....")
        s=time.time()
        
        [KxFS,KyFS]=ed.FS_contour_HT(NpointsFS_pre)
        NsizeFS=np.size(KxFS)
        ang=np.arctan2(KyFS,KxFS)
        difang=np.diff(ang) #angle is a bit not uniform, taking the mean
        d2=difang[np.where(difang<5)[0]] #eliminating the large change to 2pi at a single point
        dth=np.mean(np.abs(d2)) #dtheta for the integration
        
        
        cutoff=10 #1/cutoff of KF
        Nt=int(NpointsFS_pre/8)
        angles=np.pi*np.arange(-int(Nt),int(Nt),1)/(Nt)
        
        difang=np.diff(angles) #angle is a bit not uniform, taking the mean
        d2=difang[np.where(difang<5)[0]] #eliminating the large change to 2pi at a single point
        dth=np.mean(np.abs(d2)) #dtheta for the integration
        
        
        indarg=[]
        for i in range(np.size(angles)):
            indarg.append( np.argmin( np.abs( ang-angles[i]) ) )
        [KxFS,KyFS]=[KxFS[indarg],KyFS[indarg]]
        
        ang=np.arctan2(KyFS,KxFS)
        
        difang=np.diff(ang) 
        d2=difang[np.where(difang<5)[0]] #eliminating the large change to 2pi at a single point
        plt.plot(ang[1:],d2)
        plt.savefig("anglediff.png")
        plt.close()
        # plt.scatter(KxFS,KyFS)
        # plt.show()
        
        # ##along v
        # KXp=[]
        # KYp=[]
        # for i in range(NsizeFS):
        #     qx=KxFS[i]   
        #     qy=KyFS[i] 
        #     kloc=np.array([qx,qy])
        #     vf=ed.Fermi_Vel(qx,qy)
        #     [vfx,vfy]=vf
        #     VF=np.sqrt(vfx**2+vfy**2)
        #     KF=np.sqrt(kloc@kloc)
        #     amp=KF/cutoff #cutoff=10 is a good value
        #     fac=amp/VF
        #     mesh=np.linspace(-fac,fac,Npoints_q)
        #     QX=mesh*vfx+qx
        #     QY=mesh*vfy+qy
        #     KXp=KXp+list(QX)
        #     KYp=KYp+list(QY)
        # KX=np.array(KXp)
        # KY=np.array(KYp)
        
        #along k
        
        amp=1/cutoff #cutoff=10 is a good value
        mesh=np.linspace(-amp,amp,Npoints_q)+1
        kf=np.sqrt(KxFS**2+KyFS**2)
        KF=np.mean(np.sqrt(KxFS**2+KyFS**2)) 
        dr=(mesh[1]-mesh[0])*KF
        print(NsizeFS,2*np.pi/NsizeFS, dth)
        print(KF*2*amp/(Npoints_q+1), dr)

        KX=np.outer(mesh,KF*KxFS/kf).flatten()
        KY=np.outer(mesh,KF*KyFS/kf).flatten()
        
        e=time.time()
        print("finished sampling in reciprocal space....t=",e-s," s")
        if self.save==True:
            with open(self.lattdir+"edKgridX"+str(self.Npoints)+".npy", 'wb') as f:
                np.save(f, KX)
            with open(self.lattdir+"edKgridY"+str(self.Npoints)+".npy", 'wb') as f:
                np.save(f, KY)
        
        return [KX,KY, dth,dr]
    
    

    def read_lattice(self , option=None):

        if option==None:

            print("reading lattice from... "+"./Lattices/KgridX"+str(self.Npoints)+".npy")
            with open(self.lattdir+"KgridX"+str(self.Npoints)+".npy", 'rb') as f:
                KX = np.load(f)

            
            print("reading lattice from... "+"./Lattices/KgridY"+str(self.Npoints)+".npy")
            with open(self.lattdir+"KgridY"+str(self.Npoints)+".npy", 'rb') as f:
                KY = np.load(f)
            return [KX,KY]
        elif option=='sq':
            print("reading lattice from... "+"./Lattices/sqKgridX"+str(self.Npoints)+".npy")
            with open(self.lattdir+"sqKgridX"+str(self.Npoints)+".npy", 'rb') as f:
                KX = np.load(f)

            
            print("reading lattice from... "+"./Lattices/sqKgridY"+str(self.Npoints)+".npy")
            with open(self.lattdir+"sqKgridY"+str(self.Npoints)+".npy", 'rb') as f:
                KY = np.load(f)
            return [KX,KY]
        
        elif option=='ed':
            print("reading lattice from... "+"./Lattices/edKgridX"+str(self.Npoints)+".npy")
            with open(self.lattdir+"sqKgridX"+str(self.Npoints)+".npy", 'rb') as f:
                KX = np.load(f)

            
            print("reading lattice from... "+"./Lattices/edKgridY"+str(self.Npoints)+".npy")
            with open(self.lattdir+"edKgridY"+str(self.Npoints)+".npy", 'rb') as f:
                KY = np.load(f)
            return [KX,KY]
        else:
            print("not implemented")


    def linpam(self,Kps,Npoints_q):
        Npoints=len(Kps)
        t=np.linspace(0, 1, Npoints_q)
        linparam=np.zeros([Npoints_q*(Npoints-1),2])
        for i in range(Npoints-1):
            linparam[i*Npoints_q:(i+1)*Npoints_q,0]=Kps[i][0]*(1-t)+t*Kps[i+1][0]
            linparam[i*Npoints_q:(i+1)*Npoints_q,1]=Kps[i][1]*(1-t)+t*Kps[i+1][1]

        return linparam

    def High_symmetry_path(self, Nt_points):
    
        VV, Gamma, K, Kp, M, Mp=self.FBZ_points(self.b[0,:],self.b[1,:])
        VV=np.array(VV+[VV[0]]) #verices
        
        G=np.array([0,0])
        K=np.array([4*np.pi/3,0])
        K1=np.array([2*np.pi/3,2*np.pi/np.sqrt(3)])
        M=np.array([np.pi, np.pi/np.sqrt(3)])
        M1=np.array([0,2*np.pi/np.sqrt(3)])
        X=np.array([np.pi,0])
        Y=np.array([0, np.pi/np.sqrt(3)])
        Y1=np.array([np.pi/3, np.pi/np.sqrt(3)])

        L=[]
        # L=L+[K[0]]+[Gamma]+[M[0]]+[Kp[-1]] ##path in reciprocal space
        # L=L+[K[0]]+[Gamma]+[M[0]]+[K[0]] ##path in reciprocal space Andrei paper
        L=L+[K]+[G]+[M]+[X]

        kp_path=self.linpam(L,Nt_points)
        # plt.plot(VV[:,0], VV[:,1])
        # plt.plot(kp_path[:,0], kp_path[:,1])
        # plt.show()


        return kp_path

    def mask_KPs(self, KX,KY):
        Vertices_list, Gamma, K, Kp, M, Mp=self.FBZ_points(self.b[0,:],self.b[1,:])
        k_window_sizex = K[1][0] 
        thres=0.25
        K=np.sqrt(KX**2+KY**2)
        ind=np.where(K<k_window_sizex*thres)
        return [KX[ind],KY[ind]]



    

class SQLattice:

    def __init__(self, Npoints, save, Machine=None):

        self.Npoints = Npoints
        self.a =0.5*np.array([[1,0],[0,1]])  
        self.b =2*(2*np.pi)*np.array([[1,0],[0,1]]) 
        self.save=save
        self.dir=dir
        #some symmetries:
        #C2z
        th1=np.pi
        self.C2z=np.array([[np.cos(th1),np.sin(th1)],[-np.sin(th1),np.cos(th1)]]) #rotation matrix 
        #C3z
        th1=2*np.pi/3
        self.C3z=np.array([[np.cos(th1),np.sin(th1)],[-np.sin(th1),np.cos(th1)]]) #rotation matrix 
        #C2x inv
        self.C2x=np.array([[1,0],[0,-1]]) #rotation matrix 

        if Machine is None:
            self.Machine = ""
        else:
            self.Machine = Machine 

        if self.Machine=='FMAC':
            self.lattdir="/Users/jfmv/Documents/Proyectos/Delafossites/Lattices/"
        elif self.Machine=='CH1':
            self.lattdir="/home/jfm343/Documents/Delafossites/Lattices/"
        elif self.Machine=='UBU':
            self.lattdir="/home/juan/Documents/Projects/Delafossites/Lattices/"
        else:
            self.lattdir="../../Lattices/"
        
        print("Machine arg is,", self.Machine)

        

        self.VolBZ=self.Vol_BZ()



    def __repr__(self):
        return "lattice( LX={w}, reciprocal lattice={c})".format(h=self.Npoints, c=self.b)

    # Print iterations progress
    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
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
    #FBZ volume
    def Vol_BZ(self):
        zhat=np.array([0,0,1])
        b_1=np.array([self.b[0,0],self.b[0,1],0]) # Moire reciprocal lattice vect extended
        b_2=np.array([self.b[1,0],self.b[1,1],0]) # Moire reciprocal lattice vect extended
        Vol_rec=np.cross(b_1,b_2)@zhat
        return Vol_rec

    #hexagon where the pointy side is up
    def hexagon1(self,pos,Radius_inscribed_hex):
        y,x = map(abs, pos) #taking the absolute value of the rotated hexagon, only first quadrant matters #effective rotation
        return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge
    
    #hexagon where the flat side is up
    def hexagon2(self,pos,Radius_inscribed_hex):
        x,y = map(abs, pos) #taking the absolute value of the rotated hexagon, only first quadrant matters
        return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge

    #gets high symmetry points
    def FBZ_points(self,b_1,b_2):
        #creating reciprocal lattice
        Np=4
        n1=np.arange(-Np,Np+1)
        n2=np.arange(-Np,Np+1)
        Recip_lat=[]
        for i in n1:
            for j in n2:
                point=b_1*i+b_2*j
                Recip_lat.append(point)

        #getting the nearest neighbours to the gamma point
        Recip_lat_arr=np.array(Recip_lat)
        dist=np.round(np.sqrt(np.sum(Recip_lat_arr**2, axis=1)),decimals=10)
        sorted_dist=np.sort(list(set(dist)) )
        points=Recip_lat_arr[np.where(dist<sorted_dist[2])[0]]

        #getting the voronoi decomposition of the gamma point and the nearest neighbours
        vor = Voronoi(points)
        Vertices=(vor.vertices)

        #ordering the points counterclockwise in the -pi,pi range
        angles_list=list(np.arctan2(Vertices[:,1],Vertices[:,0]))
        Vertices_list=list(Vertices)

        # joint sorting the two lists for angles and vertices for convenience later.
        # the linear plot routine requires the points to be in order
        # atan2 takes into acount quadrant to get the sign of the angle
        angles_list, Vertices_list = (list(t) for t in zip(*sorted(zip(angles_list, Vertices_list))))

        ##getting the M points as the average of consecutive K- Kp points
        Edges_list=[]
        for i in range(len(Vertices_list)):
            Edges_list.append([(Vertices_list[i][0]+Vertices_list[i-1][0])/2,(Vertices_list[i][1]+Vertices_list[i-1][1])/2])

        Gamma=[0,0]
        K=Vertices_list[0::2]
        Kp=Vertices_list[1::2]
        M=Edges_list[0::2]
        Mp=Edges_list[1::2]

        return Vertices_list, Gamma, K, Kp, M, Mp

    #same as Generate lattice but for the original graphene (FBZ of triangular lattice)
    def Generate_lattice(self):

        #initial grid that will be filtered
        LP=self.Npoints
        nn1=np.arange(-LP,LP+1,1)
        nn2=np.arange(-LP,LP+1,1)

        print("starting sampling in reciprocal space....")
        s=time.time()
        nn_1,nn_2=np.meshgrid(nn1,nn2)

        KX=(2*np.pi*nn_1.flatten()/LP)
        KY=(2*np.pi*nn_2.flatten()/LP)

        e=time.time()
        print(" finished sampling in reciprocal space....", e-s)
        
        if self.save==True:

            with open(self.lattdir+"SQlat_KgridX"+str(self.Npoints)+".npy", 'wb') as f:
                np.save(f, KX)
            with open(self.lattdir+"SQlat_KgridY"+str(self.Npoints)+".npy", 'wb') as f:
                np.save(f, KY)
        
        return [KX,KY]

    def read_lattice(self ):

        print("reading lattice from... "+"./Lattices/SQlat_KgridX"+str(self.Npoints)+".npy")
        with open(self.lattdir+"SQlat_KgridX"+str(self.Npoints)+".npy", 'rb') as f:
            KX = np.load(f)
        
        print("reading lattice from... "+"./Lattices/SQlat_KgridY"+str(self.Npoints)+".npy")
        with open(self.lattdir+"SQlat_KgridY"+str(self.Npoints)+".npy", 'rb') as f:
            KY = np.load(f)
        return [KX,KY]


    def linpam(self,Kps,Npoints_q):
        Npoints=len(Kps)
        t=np.linspace(0, 1, Npoints_q)
        linparam=np.zeros([Npoints_q*(Npoints-1),2])
        for i in range(Npoints-1):
            linparam[i*Npoints_q:(i+1)*Npoints_q,0]=Kps[i][0]*(1-t)+t*Kps[i+1][0]
            linparam[i*Npoints_q:(i+1)*Npoints_q,1]=Kps[i][1]*(1-t)+t*Kps[i+1][1]

        return linparam

    # def High_symmetry_path(self, Nt_points):
    
    #     VV, Gamma, K, Kp, M, Mp=self.FBZ_points(self.b[0,:],self.b[1,:])
    #     VV=np.array(VV+[VV[0]]) #verices
        
    #     G=np.array([0,0])
    #     K=np.array([4*np.pi/3,0])
    #     K1=np.array([2*np.pi/3,2*np.pi/np.sqrt(3)])
    #     M=np.array([np.pi, np.pi/np.sqrt(3)])
    #     M1=np.array([0,2*np.pi/np.sqrt(3)])
    #     X=np.array([np.pi,0])
    #     Y=np.array([0, np.pi/np.sqrt(3)])
    #     Y1=np.array([np.pi/3, np.pi/np.sqrt(3)])

    #     L=[]
    #     # L=L+[K[0]]+[Gamma]+[M[0]]+[Kp[-1]] ##path in reciprocal space
    #     # L=L+[K[0]]+[Gamma]+[M[0]]+[K[0]] ##path in reciprocal space Andrei paper
    #     L=L+[K]+[G]+[M]+[X]

    #     kp_path=self.linpam(L,Nt_points)
    #     # plt.plot(VV[:,0], VV[:,1])
    #     # plt.plot(kp_path[:,0], kp_path[:,1])
    #     # plt.show()


    #     return kp_path

    # def mask_KPs(self, KX,KY):
    #     Vertices_list, Gamma, K, Kp, M, Mp=self.FBZ_points(self.b[0,:],self.b[1,:])
    #     k_window_sizex = K[1][0] 
    #     thres=0.25
    #     K=np.sqrt(KX**2+KY**2)
    #     ind=np.where(K<k_window_sizex*thres)
    #     return [KX[ind],KY[ind]]



    

