import numpy as np
import matplotlib.pyplot as plt

"""
###################
###################RANDOM WALK IN 1D
def nasty_function(x):
    x_0 = 3.0
    a = 0.01
    return np.exp(-(x**2))/((x-x_0)**2 + a**2)
x_walk = np.empty((0)) #this is an empty list to keep all the steps
x_0 = 8.0*((np.random.random())-0.5) #this is the initialization
x_walk = np.append(x_walk,x_0)
print(x_walk)

n_iterations = 20000 #this is the number of iterations I want to make
for i in range(n_iterations):
    x_prime = np.random.normal(x_walk[i], 0.1) #0.1 is the sigma in the normal distribution
    alpha = nasty_function(x_prime)/nasty_function(x_walk[i])
    if(alpha>=1.0):
        x_walk  = np.append(x_walk,x_prime)
    else:
        beta = np.random.random()
        if(beta<=alpha):
            x_walk = np.append(x_walk,x_prime)
        else:
            x_walk = np.append(x_walk,x_walk[i])
x=np.linspace(-4,4,100)
f = nasty_function(x)
norm = sum(f*(x[1]-x[0]))
#plot(x,f/norm, linewidth=1, color='r')
count, bins = np.histogram(x_walk, bins=1000)
plt.bar(bins[:-1], count, width=bins[1]-bins[0])
#fig = figure(1, figsize=(9.5,6.5))
plt.xlabel('x')
plt.ylabel('p(x)')
plt.show()

###################
###################RANDOM WALK IN 2D

def nasty_function2(x,y):
    x_0 = 3.0
    a = 0.01
    return np.exp(-(x**2+y**2))/((x-x_0)**2 + a**2)


x_walk = np.empty((0)) #this is an empty list to keep all the steps
y_walk = np.empty((0)) #this is an empty list to keep all the steps
x_0 = 8.0*((np.random.random())-0.5) #this is the initialization
y_0 = 8.0*((np.random.random())-0.5) #this is the initialization
x_walk = np.append(x_walk,x_0)
y_walk = np.append(y_walk,y_0)
print(x_walk,y_walk)


n_iterations = 40000 #this is the number of iterations I want to make
for i in range(n_iterations):
    x_prime = np.random.normal(x_walk[i], 0.1) #0.1 is the sigma in the normal distribution
    y_prime = np.random.normal(y_walk[i], 0.1) #0.1 is the sigma in the normal distribution
    alpha = nasty_function2(x_prime,y_prime)/nasty_function2(x_walk[i],y_walk[i])
    if(alpha>=1.0):
        x_walk  = np.append(x_walk,x_prime)
        y_walk  = np.append(y_walk,y_prime)
    else:
        beta = np.random.random()
        if(beta<=alpha):
            x_walk  = np.append(x_walk,x_prime)
            y_walk  = np.append(y_walk,y_prime)
        else:
            x_walk = np.append(x_walk,x_walk[i])
            y_walk = np.append(y_walk,y_walk[i])

plt.scatter(x_walk,y_walk, s=1)
plt.show()

###################HISTOGRAMS IN 2D

# create data
x = np.random.normal(size=50000)
y = x * 3 + np.random.normal(size=50000)

# Big bins
plt.hist2d(x_walk,y_walk, bins=(50, 50), cmap=plt.cm.jet)
plt.show()

# Small bins
plt.hist2d(x_walk,y_walk, bins=(300, 300), cmap=plt.cm.jet)
plt.show()

# If you do not set the same values for X and Y, the bins won't be a square!
plt.hist2d(x_walk,y_walk, bins=(300, 30), cmap=plt.cm.jet)
plt.show()

######################################################################
######################################################################
######################################################################
######################################################################
######################################################################


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

def nasty_function2(kx,ky,omega, T,qx,qy):
    ss=10
    return np.exp( -Disp(kx+qx,ky+qy, omega)**2/(2*ss*ss)   )

omega=10
T=1.0
qx=0
qy=0
x_walk = np.empty((0)) #this is an empty list to keep all the steps
y_walk = np.empty((0)) #this is an empty list to keep all the steps
x_0 = 8.0*((np.random.random())-0.5) #this is the initialization
y_0 = 8.0*((np.random.random())-0.5) #this is the initialization
x_walk = np.append(x_walk,x_0)
y_walk = np.append(y_walk,y_0)
print(x_walk,y_walk)


n_iterations = 40000 #this is the number of iterations I want to make
for i in range(n_iterations):
    x_prime = np.random.normal(x_walk[i], 0.1) #0.1 is the sigma in the normal distribution
    y_prime = np.random.normal(y_walk[i], 0.1) #0.1 is the sigma in the normal distribution
    alpha = nasty_function2(x_prime,y_prime,omega, T,qx,qy)/nasty_function2(x_walk[i],y_walk[i],omega, T,qx,qy)
    if(alpha>=1.0):
        x_walk  = np.append(x_walk,x_prime)
        y_walk  = np.append(y_walk,y_prime)
    else:
        beta = np.random.random()
        if(beta<=alpha):
            x_walk  = np.append(x_walk,x_prime)
            y_walk  = np.append(y_walk,y_prime)
        else:
            x_walk = np.append(x_walk,x_walk[i])
            y_walk = np.append(y_walk,y_walk[i])

plt.scatter(x_walk,y_walk, s=1)
plt.show()



# Big bins
plt.hist2d(x_walk,y_walk, bins=(50, 50), cmap=plt.cm.jet)
plt.show()

# Small bins
plt.hist2d(x_walk,y_walk, bins=(300, 300), cmap=plt.cm.jet)
plt.show()

# If you do not set the same values for X and Y, the bins won't be a square!
plt.hist2d(x_walk,y_walk, bins=(300, 30), cmap=plt.cm.jet)
plt.show()
"""

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


"""
################FOR UNIFORM SAMPLES WITHIN THE HEXAGON (THROWING AWAY DATA)
x = 5*(2*np.random.random(50000)-1)
y = 5*(2*np.random.random(50000)-1)


xp=[]
yp=[]
for i in range(np.size(x)):
    if hexagon( (x[i],y[i]) ):
        xp.append(x[i])
        yp.append(y[i])

plt.scatter(xp,yp, s=1)
plt.show()


xp=[]
yp=[]

#################FOR FOLDING THE RANDOM WALK WITHIN THE HEXAGON
n=np.array([1,-1,2,-2,3,-3])
n1,n2=np.meshgrid(n,n)
NG=np.size(n1)
n1=np.reshape(n1,[NG,1])
n2=np.reshape(n2,[NG,1])
print(np.shape(n1))
for i in range(np.size(x)):

    if hexagon( (x[i],y[i]) ):
        xp.append(x[i])
        yp.append(y[i])
    else:
        for l in range(NG):
            if hexagon( ( x[i]-n1[l]*b_1[0]-n2[l]*b_2[0], y[i]-n1[l]*b_1[1]-n2[l]*b_2[1]) ):
                xp.append(x[i]-n1[l]*b_1[0]-n2[l]*b_2[0])
                yp.append(y[i]-n1[l]*b_1[1]-n2[l]*b_2[1])
                break



plt.scatter(xp,yp, s=1)
plt.show()
"""
####################SELF CONTAINED PIECE OF THE SCRIPT TO GET THE WALK IN THE RELEVANT REGION NEAR THE FERMI SURFACE


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

def nasty_function2(kx,ky,omega, T,qx,qy):
    ss=10
    return np.exp( -Disp(kx+qx,ky+qy, omega)**2/(2*ss*ss)   )

omega=10
T=1.0
qx=-0.13979933110367915
qy=-2.6056461548771535
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
    alpha = nasty_function2(x_prime,y_prime,omega, T,qx,qy)/nasty_function2(x_walk[i],y_walk[i],omega, T,qx,qy)
    if(alpha>=1.0):
        x_walk  = np.append(x_walk,x_prime)
        y_walk  = np.append(y_walk,y_prime)
    else:
        beta = np.random.random()
        if(beta<=alpha):
            x_walk  = np.append(x_walk,x_prime)
            y_walk  = np.append(y_walk,y_prime)
        else:
            x_walk = np.append(x_walk,x_walk[i])
            y_walk = np.append(y_walk,y_walk[i])
plt.scatter(x_walk,y_walk,s=1)
plt.show()

x_walk_p = np.empty((0)) #this is an empty list to keep all the steps
y_walk_p = np.empty((0)) #this is an empty list to keep all the steps
n=np.array([0,1,-1,2,-2])
n1,n2=np.meshgrid(n,n)
NG=np.size(n1)
n1=np.reshape(n1,[NG,1]).flatten()
n2=np.reshape(n2,[NG,1]).flatten()
print(n1,n2)
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





######################PLOTTING


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
plt.plot(kpath[:,0],kpath[:,1])
plt.scatter(x_walk_p,y_walk_p, s=1)
plt.show()


# Small bins
#plt.hist2d(x_walk_p,y_walk_p, bins=(300, 300), cmap=plt.cm.jet)
#plt.show()
