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

################################
################################
################################
################################
a=1
a_1=a*np.array([1,0,0])
a_2=a*np.array([1/2,np.sqrt(3)/2,0])
zhat=np.array([0,0,1])

Vol_real=np.dot(np.cross(a_1,a_2),zhat)
b_1=np.cross(a_2,zhat)*(2*np.pi)/Vol_real
b_2=np.cross(zhat,a_1)*(2*np.pi)/Vol_real
Vol_rec=np.dot(np.cross(b_1,b_2),zhat)





################################
################################
################################
################################

#Reading and reshaping the structure factor

################################
################################
################################
################################

L = int(sys.argv[1])
n_freqs = 4097

# Load the data files
#SF = np.load('/Users/jfmv/Documents/Proyectos/Delafossites/integration_delafossite/test'+str(L)+'.npy')
#SF = np.load('/Users/jfmv/Documents/Proyectos/Delafossites/integration_delafossite/test_lang_'+str(L)+'.npy')
#dsf_data = np.load('/home/juan/Documents/Projects/Delafossites/SF_data/dsf_TLHAF_L=120_tf=4096_T=0.5.npy')



Np=80
n1=np.arange(-Np,Np+1)
n2=np.arange(-Np,Np+1)

a_1=a_1[0:2]
a_2=a_2[0:2]
b_1=b_1[0:2]
b_2=b_2[0:2]

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

k_window_sizey = K[2][1] 
k_window_sizex = K[1][0] 

Radius_inscribed_hex=1.000000000000001*k_window_sizex

def hexagon(pos):
    x, y = map(abs, pos) #taking the absolute value of the rotated hexagon, only first quadrant matters
    return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge


n1=np.arange(-L,L+1,1)
n2=np.arange(-L,L+1,1)
n_1,n_2=np.meshgrid(n1,n2)

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

KXX=2*np.pi*n_1/L
KYY= 2*(2*np.pi*n_2/L - np.pi*n_1/L)/np.sqrt(3)


n_1pp=np.array(n_1p)
n_2pp=np.array(n_2p)
KX=2*np.pi*n_1pp/L
KY= 2*(2*np.pi*n_2pp/L - np.pi*n_1pp/L)/np.sqrt(3)

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

SF=dsf_data[n_1pp%L,n_2pp%L,:].T
n_3=200  #fixed frequency index for testing stuff
# plt.plot(kpath[:,0],kpath[:,1])
# plt.scatter(KXX,KYY,c=SF[n_3,:],s =3)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()





print("shapes KX and SF, numfreqs " , np.shape(KX), np.shape(SF), n_freqs)
print("frequency " , 2*np.pi*n_3/n_freqs )


points = np.array([KX,KY]).T
values = SF[n_3,:]
HandwavyThres=1e-4
SF2=np.vstack( (SF, np.zeros(np.size(KX))+HandwavyThres ) )



################################
################################
################################
################################
#Defining integrand
###other Parameters

################################
################################
################################
################################
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
NFSpoints=9
xFS = v[5::int(np.size(v[:,1])/NFSpoints),0]
yFS = v[5::int(np.size(v[:,1])/NFSpoints),1]
KFx=xFS[0]
KFy=yFS[0]
plt.scatter(xFS,yFS)
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
    values = SF2[np.array(values_ind),np.arange(0,int(np.size(qx)))]
    fac_p=np.exp(ed/T)*(1+np.exp(-w/T))/(1+np.exp(ed/T))
    return values*2*np.pi*fac_p



omegas=np.linspace(0,2*np.pi,n_freqs)
w=omegas[n_3]
print(w)


siz=30
Omegs=np.linspace(0 ,6 ,siz)


SS=integrand_Disp(KX,KY,KFx,KFy,0,SF)
ind=np.where(abs(KX+KY)<1e-10)[0]
KX2=np.delete(KX,ind)
KY2=np.delete(KY,ind)
SS2=np.delete(SS,ind)
# plt.scatter(KX2,KY2,c=np.log10(SS2+1e-11),s =1)
# plt.colorbar()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

# plt.scatter(KX,KY,c=np.log10(integrand_Disp(KX,KY,KFx,KFy,0,SF)+1e-11),s =1)
# plt.colorbar()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()
sigm_FS=[]
for ell in range(np.size(xFS)):
    phi=2*np.pi/6 #rotation angle

    #rotating and cleaving if absolute value of rotated point's y coordinate exceeds top boundary of 1BZ
    #KFx=np.cos(phi)*xFS[ell]-np.sin(phi)*yFS[ell]
    #KFy=np.sin(phi)*xFS[ell]+np.cos(phi)*yFS[ell]
    KFx=xFS[ell]
    KFy=yFS[ell]


    ds=Vol_rec/np.size(KX)
    sigm=[]
    S0=0#np.sum(integrand_Disp(KX,KY,KFx,KFy,0,SF)*ds)
    print("Method1,...T=",T, ds)

    for i in range(siz):
        start=time.time()
        if (Omegs[i]!=0):
            SI=np.sum(integrand_Disp(KX,KY,KFx,KFy,Omegs[i],SF)*ds)-S0
        else:
        
            SS=integrand_Disp(KX,KY,KFx,KFy,Omegs[i],SF)*ds
            ind=np.where(abs(KX+KY)<1e-10)[0]
            KX=np.delete(KX,ind)
            KY=np.delete(KY,ind)
            SS=np.delete(SS,ind)
            SI=np.sum(SS)-S0
        # SI=np.sum(integrand_Disp(KX,KY,KFx,KFy,Omegs[i],SF)*ds)-S0
        end=time.time()
        print("time ",end-start)
        print(i,SI)
        sigm.append(SI)
    plt.plot(Omegs,sigm, 'o', label="T="+str(T)+" ,kx="+str(np.round(KFx,3))+" ,ky="+str(np.round(KFy,3)))
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"-Im$\Sigma (k_F,\omega)$")
    plt.legend()
    plt.savefig("kx_"+str(KFx)+"_ky_"+str(KFy)+"_T_"+str(T)+"func.png", dpi=200)
    #plt.close()
# for i in range(n_freqs):
#     start=time.time()
#     #SI=np.sum(integrand_Disp(KX,KY,KFx,KFy,Omegs[i],SF)*ds)-S0
#     SI=np.sum(SF[i,:]*ds)
#     end=time.time()
#     print("time ",end-start)
#     print(i,SI)
#     sigm.append(SI)
#
#
# plt.plot(omegas,sigm, 'o', label="T="+str(T))
# plt.xlabel(r"$\omega$")
# plt.ylabel(r"$S(\omega)$")
# plt.legend()
# plt.show()
