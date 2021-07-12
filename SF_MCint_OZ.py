import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import time
import sys
################################
################################
################################
################################
#defining triangular lattice
a=1
a_1=a*np.array([1,0,0])
a_2=a*np.array([1/2,np.sqrt(3)/2,0])
zhat=np.array([0,0,1])

Vol_real=np.dot(np.cross(a_1,a_2),zhat)
b_1=np.cross(a_2,zhat)*(2*np.pi)/Vol_real
b_2=np.cross(zhat,a_1)*(2*np.pi)/Vol_real
Vol_rec=np.dot(np.cross(b_1,b_2),zhat)

a_1=a_1[0:2]
a_2=a_2[0:2]
b_1=b_1[0:2]
b_2=b_2[0:2]

def hexagon_a(pos):
    Radius_inscribed_hex=1.000000000000001*4*np.pi/3
    x, y = map(abs, pos) #only first quadrant matters
    return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge

def nasty_function2(kx,ky,omega, T,qx,qy):
    ss=2*np.pi
    return np.exp( -Disp(kx+qx,ky+qy, omega)**2/(2*ss*ss)   )


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

#linear parametrization accross different points in the BZ
def linpam(Kps,Npoints_q):
    Npoints=len(Kps)
    t=np.linspace(0, 1, Npoints_q)
    linparam=np.zeros([Npoints_q*(Npoints-1),2])
    for i in range(Npoints-1):
        linparam[i*Npoints_q:(i+1)*Npoints_q,0]=Kps[i][0]*(1-t)+t*Kps[i+1][0]
        linparam[i*Npoints_q:(i+1)*Npoints_q,1]=Kps[i][1]*(1-t)+t*Kps[i+1][1]

    return linparam

VV=Vertices_list+[Vertices_list[0]]
Nt=1000
kpath=linpam(VV,Nt)

def hexsamp2(npoints_x,npoints_y):
    ##########################################
    #definitions of the BZ grid
    ##########################################
    k_window_sizey = K[2][1] #4*(np.pi/np.sqrt(3))*np.sin(theta/2) (half size of  MBZ from edge to edge)
    k_window_sizex = K[1][0]   #(1/np.sqrt(3))*GM2[0] ( half size of MBZ from k to k' along a diagonal)
    kn_pointsx = npoints_x
    kn_pointsy = npoints_y
    kx_rangex = np.linspace(-k_window_sizex,k_window_sizex,kn_pointsx)
    ky_rangey = np.linspace(-k_window_sizey,k_window_sizey,kn_pointsy)
    step_x = kx_rangex[1]-kx_rangex[0]
    step_y = ky_rangey[2]-ky_rangey[1]

    tot_kpoints=kn_pointsx*kn_pointsy
    step = min([ step_x, step_y])
    bz = np.zeros([kn_pointsx,kn_pointsy])

    ##########################################
    #check if a point is inside of an Hexagon inscribed in a circle of radius Radius_inscribed_hex
    ##########################################
    Radius_inscribed_hex=1.00001*k_window_sizex
    def hexagon(pos):
        x, y = map(abs, pos) #taking the absolute value of the rotated hexagon, only first quadrant matters
        return y < (3**0.5) * min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge

    ##########################################
    #Setting up kpoint arrays
    ##########################################
    #Number of relevant kpoints
    for x in range(kn_pointsx):
        for y in range(kn_pointsy):
            if hexagon((kx_rangex[x],ky_rangey[y])):
                bz[x,y]=1
    num_kpoints=int(np.sum(bz))

    #x axis of the BZ along Y axis of the plot
    #plt.imshow(bz)
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()


    #filling kpoint arrays

    KX2=[]
    KY2=[]
    for x in range(kn_pointsx):
        for y in range(kn_pointsy):

            if hexagon((kx_rangex[x],ky_rangey[y])):
                #plt.scatter(kx_rangex[x],ky_rangey[y])
                KX2.append(kx_rangex[x])
                KY2.append(ky_rangey[y])

    return np.array(KX2),np.array(KY2),step_x*step_y
ni=1601
KX_in, KY_in, dS_in=hexsamp2(ni,ni)
################################
################################
################################
################################
#Reading and reshaping the structure factor
L = int(sys.argv[1])
n_freqs = 4097

# Load the data files
#For MacOS
SF = np.load('/Users/jfmv/Documents/Proyectos/Delafossites/integration_delafossite/test'+str(L)+'.npy')
#SF = np.load('/Users/jfmv/Documents/Proyectos/Delafossites/integration_delafossite/test_lang_'+str(L)+'.npy')


#For linux
#SF = np.load('/home/juan/Documents/Projects/Delafossites/integration_delafossite/test'+str(L)+'.npy')
omegas=np.linspace(0,2*np.pi,n_freqs)
if L != 120:
    Radius_inscribed_hex=0.999*4*np.pi/3
else:
    Radius_inscribed_hex=1.000000000000001*4*np.pi/3

def hexagon(pos):
    x, y = map(abs, pos) #only first quadrant matters
    return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge


n1=np.arange(-L,L+1,1)
n2=np.arange(-L,L+1,1)

n_1p=[]
n_2p=[]
for x in n1:
    for y in n2:
        kx=2*np.pi*x/L
        ky=2*(2*np.pi*y/L - np.pi*x/L)/np.sqrt(3)
        if hexagon(( kx, ky)):
            #plt.scatter(kx_rangex[x],ky_rangey[y])
            n_1p.append(x)
            n_2p.append(y)

n_1=np.array(n_1p)
n_2=np.array(n_2p)
n_3=int(n_freqs/(2*np.pi))#1000
KX=2*np.pi*n_1/L
KY=2*(2*np.pi*n_2/L - np.pi*n_1/L)/np.sqrt(3)

# plt.scatter(KX,KY,c=SF[n_3,:],s =1)
# plt.colorbar()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()
#

print("shape " , np.shape(KX))
print("frequency " , 2*np.pi*n_3/n_freqs )


points = np.array([KX,KY]).T
values = SF[n_3,:]
HandwavyThres=1e-9
SF2=np.vstack( (SF, np.zeros(np.size(KX))+HandwavyThres ) )

################################
################################
################################
################################
#Defining integrand
###other Parameters

T=1.0
J=2*5.17
tp1=568/J #in units of J
tp2=-108/J #/tpp1

mu=0
def Disp(kx,ky,mu):
    ed=-tp1*(2*np.cos(kx)+4*np.cos((kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
    ed=ed-tp2*(2*np.cos(np.sqrt(3)*(ky))+4*np.cos(3*(kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
    ed=ed-mu
    return ed

x = np.linspace(-3.8, 3.8, 300)
X, Y = np.meshgrid(x, x)
Z = Disp(X, Y, mu)

c= plt.contour(X, Y, Z, levels=[0],linewidths=3, cmap='summer');
v = c.collections[0].get_paths()[0].vertices
xFS2 = v[::10,0]
yFS2 = v[::10,1]
KFx2=xFS2[50]
KFy2=yFS2[50]
plt.scatter(KFx2,KFy2)
plt.show()


def integrand_Disp(qx,qy,kx,ky,w,SF2):

    ed=Disp(kx+qx,ky+qy,mu)
    om=w-ed
    om2=-ed

    values_ind=[]
    for ee in om:
        ind=np.argmin( abs( omegas-np.abs(ee) ) )
        values_ind.append(ind)


    ##getting the closest index to the sampled omegas, if it exceeds we use the threshold column added when reading the griddata
    ##if the the value corresponds to a negative valu, it does not matter
    #since we use the absolute value of w-ek assuming symmetry of the structure factor
    m=4
    gamma=1
    values = gamma*om/((qx**2 +qy**2 +om**2+m**2)**2+(om*gamma)**2)
    fac_p=(1/(1+np.exp(om2/T)) + 1/(np.exp(om/T)-1))
    return values*np.pi*fac_p

w=omegas[n_3]
print(w)


siz=10
Omegs=np.linspace(0.0,2,siz)


plt.scatter(KX,KY,c=(integrand_Disp(KX,KY,KFx2,KFy2,0,SF)), s=1)
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

ds=Vol_rec/np.size(KX)
sigm=[]
sigm_MC=[]
print("Method1,...T=",T, ds)
for j in range(siz):
    start=time.time()
    SI=np.sum(integrand_Disp(KX,KY,KFx2,KFy2,Omegs[j],SF)*ds)
    #SI=np.sum(integrand_Disp(KX,KY,KFx,KFy,Omegs[i],SF)*ds)
    end=time.time()
    print("time for riemmann integration",end-start)
    #print(Omegs[i],SI)
    sigm.append(SI)

    start=time.time()
    ####################MC integration
    omega=Omegs[j]
    qx=KFx2
    qy=KFy2
    x_walk = np.empty((0)) #this is an empty list to keep all the steps
    y_walk = np.empty((0)) #this is an empty list to keep all the steps
    x_0 = qx #this is the initialization
    y_0 = qy #this is the initialization
    x_walk = np.append(x_walk,x_0)
    y_walk = np.append(y_walk,y_0)
    print(x_walk,y_walk)


    n_iterations = 100000 #this is the number of iterations I want to make
    for i in range(n_iterations):
        x_prime = np.random.normal(x_walk[i], 0.1) #0.1 is the sigma in the normal distribution
        y_prime = np.random.normal(y_walk[i], 0.1) #0.1 is the sigma in the normal distribution
        alpha = np.log(nasty_function2(x_prime,y_prime,omega, T,qx,qy))-np.log(nasty_function2(x_walk[i],y_walk[i],omega, T,qx,qy))
        if(alpha>=0.0):
            x_walk  = np.append(x_walk,x_prime)
            y_walk  = np.append(y_walk,y_prime)
        else:
            beta = np.log(np.random.random())
            if(beta<=alpha):
                x_walk  = np.append(x_walk,x_prime)
                y_walk  = np.append(y_walk,y_prime)
            else:
                x_walk = np.append(x_walk,x_walk[i])
                y_walk = np.append(y_walk,y_walk[i])

    end2=time.time()
    print("time for MCMC samples",end2-start)

    x_walk_p = np.empty((0)) #this is an empty list to keep all the steps
    y_walk_p = np.empty((0)) #this is an empty list to keep all the steps
    n=np.array([0,1,-1,2,-2])
    n1,n2=np.meshgrid(n,n)
    NG=np.size(n1)
    n1=np.reshape(n1,[NG,1])
    n2=np.reshape(n2,[NG,1])
    for i in range(n_iterations):

        if hexagon_a( (x_walk[i],y_walk[i]) ):
            x_walk_p= np.append(x_walk_p, x_walk[i])
            y_walk_p= np.append(y_walk_p, y_walk[i])
        else:
            for l in range(NG):
                if hexagon_a( ( x_walk[i]-n1[l]*b_1[0]-n2[l]*b_2[0], y_walk[i]-n1[l]*b_1[1]-n2[l]*b_2[1]) ):
                    x_walk_p = np.append(x_walk_p,x_walk[i]-n1[l]*b_1[0]-n2[l]*b_2[0])
                    y_walk_p = np.append(y_walk_p,y_walk[i]-n1[l]*b_1[1]-n2[l]*b_2[1])
                    break




    Nasty_int=np.sum(nasty_function2(KX_in, KY_in,omega,T,qx,qy )*dS_in )
    SI_MC=Nasty_int*np.mean(integrand_Disp(x_walk_p,y_walk_p,qx,qy,omega,SF) /nasty_function2(x_walk_p,y_walk_p,omega,T,qx,qy ))
    sigm_MC.append(SI_MC)
    print(j,SI, SI_MC)
    end=time.time()
    print("time for MC integration",end-start)
    print("time for rearrangement",end-end2)
    plt.scatter(KX,KY,c=(integrand_Disp(KX,KY,qx,qy,omega,SF)), s=1)
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


    plt.plot(kpath[:,0],kpath[:,1])
    plt.scatter(x_walk_p,y_walk_p,c=(integrand_Disp(x_walk_p,y_walk_p,qx,qy,omega,SF)), s=1)
    plt.show()

plt.plot(Omegs,sigm, 'o', label="T="+str(T))
plt.plot(Omegs,sigm_MC, 'o', label="T="+str(T))
plt.xlabel(r"$\omega$")
plt.ylabel(r"-Im$\Sigma (k_F,\omega)$")
plt.legend()
plt.show()





######################temperature test
ds=Vol_rec/np.size(KX)
sigm=[]
print("Method1,...T=",T, ds)
for T in np.linspace(0.01,1,10):
    start=time.time()
    SI=np.sum(integrand_Disp(KX,KY,KFx2,KFy2,0,SF)*ds)
    #SI=np.sum(integrand_Disp(KX,KY,KFx,KFy,Omegs[i],SF)*ds)
    end=time.time()
    print("time ",end-start)
    #print(Omegs[i],SI)
    print(SI)
    sigm.append(SI)


plt.plot(np.linspace(0.01,1,10),sigm, 'o', label="T="+str(T))
plt.xlabel(r"$T$")
plt.ylabel(r"-Im$\Sigma (k_F,\omega)$")
plt.legend()


plt.show()
