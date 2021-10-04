import numpy as np
import scipy
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import linalg as la
import time

class TriangLattice:

    def __init__(self, Npoints, save):

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


        self.VolBZ=self.Vol_BZ()



    def __repr__(self):
        return "lattice( LX={w}, reciprocal lattice={c})".format(h=self.Npoints, c=self.b)


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
        for x in nn1:
            for y in nn2:
                kx=(2*np.pi*x/LP)
                ky=(2*(2*np.pi*y/LP - np.pi*x/LP)/np.sqrt(3))
                if self.hexagon2( ( kx, ky), Radius_inscribed_hex ):
                    nn_1p.append(x)
                    nn_2p.append(y)

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
            with open("./Lattices/KgridX"+str(self.Npoints)+".npy", 'wb') as f:
                np.save(f, KX)
            with open("./Lattices/KgridY"+str(self.Npoints)+".npy", 'wb') as f:
                np.save(f, KY)
        
        return [KX,KY]

    def read_lattice(self):

        print("reading lattice from... "+"./Lattices/KgridX"+str(self.Npoints)+".npy")
        with open("./Lattices/KgridX"+str(self.Npoints)+".npy", 'rb') as f:
            KX = np.load(f)

        
        print("reading lattice from... "+"./Lattices/KgridY"+str(self.Npoints)+".npy")
        with open("./Lattices/KgridY"+str(self.Npoints)+".npy", 'rb') as f:
            KY = np.load(f)
        return [KX,KY]

