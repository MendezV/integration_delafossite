import numpy as np
import matplotlib.pyplot as plt
#defining triangular lattice
a=1
a_1=a*np.array([1,0,0])
a_2=a*np.array([1/2,np.sqrt(3)/2,0])
zhat=np.array([0,0,1])

Vol_real=np.dot(np.cross(a_1,a_2),zhat)
b_1=np.cross(a_2,zhat)*(2*np.pi)/Vol_real
b_2=np.cross(zhat,a_1)*(2*np.pi)/Vol_real
Vol_rec=np.dot(np.cross(b_1,b_2),zhat)
#print(np.dot(a_2,b_2),np.dot(a_1,b_1))

Np=80
n1=np.arange(-Np,Np+1)
n2=np.arange(-Np,Np+1)

a_1=a_1[0:2]
a_2=a_2[0:2]
b_1=b_1[0:2]
b_2=b_2[0:2]

#getting the first brilloin zone from the Voronoi decomp of the recipprocal lattice
#input: reciprocal lattice vectors
#output: Points that delimit the FBZ -
#high symmetry points (for now just the triangular lattice will be implemented)
from scipy.spatial import Voronoi, voronoi_plot_2d
def FBZ_points(b_1,b_2):
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

    #joint sorting the two lists for angles and vertices for convenience later.
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

Vertices_list, Gamma, K, Kp, M, Mp=FBZ_points(b_1,b_2)


#######reciprocal lattice
#creating reciprocal lattice preparing for k-point integration
Recip_lat=[]
for i in n1:
    for j in n2:
        point=b_1*i+b_2*j
        Recip_lat.append(point)


#########
#####CREATING SAMPLES FOR INTEGRATION

#generating k points

#square sampling
#kxx,kyy=np.meshgrid(kx,ky)
scale_fac=0.00831
#Shrinking the reciprocal lattice
def hexsamp(b_1,b_2,scale_fac,Np_p):
    n1_p=np.arange(-Np_p,Np_p+1)
    n2_p=np.arange(-Np_p,Np_p+1)

    Recip_lat_p=[]
    for i in n1_p:
        for j in n2_p:
            point=b_1*i+b_2*j
            Recip_lat_p.append(point)

    Recip_lat_p_arr=scale_fac*np.array(Recip_lat_p)
    kxx=Recip_lat_p_arr[:,0]
    kyy=Recip_lat_p_arr[:,1]

    phi=2*np.pi/6 #rotation angle

    #rotating and cleaving if absolute value of rotated point's y coordinate exceeds top boundary of 1BZ
    kxx_rot=np.cos(phi)*kxx-np.sin(phi)*kyy
    kyy_rot=np.sin(phi)*kxx+np.cos(phi)*kyy

    kxx2=kxx[np.where(np.abs(kyy_rot)<K[2][1])]
    kyy2=kyy[np.where(np.abs(kyy_rot)<K[2][1])]

    #rotating and cleaving if absolute value of rotated point's y coordinate exceeds top boundary of 1BZ
    kxx_rot2=np.cos(-phi)*kxx2-np.sin(-phi)*kyy2
    kyy_rot2=np.sin(-phi)*kxx2+np.cos(-phi)*kyy2

    kxx3=kxx2[np.where(np.abs(kyy_rot2)<K[2][1])]
    kyy3=kyy2[np.where(np.abs(kyy_rot2)<K[2][1])]

    #cleaving if absolute value of point's y coordinate exceeds top boundary of 1BZ
    KX_p=kxx3[np.where(np.abs(kyy3)<K[2][1])]
    KY_p=kyy3[np.where(np.abs(kyy3)<K[2][1])]

    return KX_p,KY_p
KX, KY=hexsamp(b_1,b_2,scale_fac,Np)
S=np.ones(np.shape(KX))
print("volume of the FBZ",Vol_rec)
print("volume of the FBZ 2",np.sum(np.sum(S))*Vol_rec/np.prod(np.shape(KX)))


from scipy import integrate

def f(x,y): # function to integrate
    return 1


c = integrate.dblquad(f,0.,2.,lambda x : 0., lambda x: 4.)
print(c)
