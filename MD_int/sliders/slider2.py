import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import time
import sys

# The parametrized function to be plotted
def f(t, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t)

t = np.linspace(0, 1, 1000)

# Define initial parameters
init_amplitude = 5
init_frequency = 3

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line, = plt.plot(t, f(t, init_amplitude, init_frequency), lw=2)
ax.set_xlabel('Time [s]')

axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
freq_slider = Slider(
    ax=axfreq,
    label='Frequency [Hz]',
    valmin=0.1,
    valmax=30,
    valinit=init_frequency,
)

# Make a vertically oriented slider to control the amplitude
axamp = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor=axcolor)
amp_slider = Slider(
    ax=axamp,
    label="Amplitude",
    valmin=0,
    valmax=10,
    valinit=init_amplitude,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(f(t, amp_slider.val, freq_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
freq_slider.on_changed(update)
amp_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    freq_slider.reset()
    amp_slider.reset()
button.on_clicked(reset)

plt.show()



###########################################
###########################################
###########################################
###########################################


J=2*5.17 #in mev
tp1=568/J #in units of Js\
tp2=-tp1*108/568 #/tpp1
U=4000/J
g=100/J
Kcou=g*g/U
Ta=sys.argv[1]
T=float(Ta)



############################################################
#Pd dispersion
############################################################

def Disp(kx,ky,mu):
    ed=-tp1*(2*np.cos(kx)+4*np.cos((kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
    ed=ed-tp2*(2*np.cos(np.sqrt(3)*(ky))+4*np.cos(3*(kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
    ed=ed-mu
    return ed

x = np.linspace(-3.4, 3.4, 2603)
X, Y = np.meshgrid(x, x)
Z = Disp(X, Y, 0)
Wbdw=np.max(Z)-np.min(Z)
print("The bandwidth is ....", Wbdw, " in units of J=",J)

mu=0.3*np.max(Z)+0.0*np.min(Z)

print("The chemical potential is ....", mu, " in units of J=",J)

x = np.linspace(-3.4, 3.4, 2603)
X, Y = np.meshgrid(x, x)
Z = Disp(X, Y, mu)

# c= plt.contour(X, Y, Z, levels=[0],linewidths=3, cmap='summer');
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()



EF= mu-np.min(Z)#fermi energy from the bottom of the band
m=EF/2
gamma=EF*1000
vmode=EF/2
gcoupl=EF/2



with open("/Users/jfmv/Documents/Proyectos/Delafossites/Struc_dat/Kpoints/KgridX"+sys.argv[2]+".npy", 'rb') as f:
    KX = np.load(f)


with open("/Users/jfmv/Documents/Proyectos/Delafossites/Struc_dat/Kpoints/KgridY"+sys.argv[2]+".npy", 'rb') as f:
    KY = np.load(f)


##########################
##########################
############################


############################################################
# Defining tringular lattice
############################################################

a=1
a_1=a*np.array([1,0,0])
a_2=a*np.array([1/2,np.sqrt(3)/2,0])
zhat=np.array([0,0,1])

Vol_real=np.dot(np.cross(a_1,a_2),zhat)
b_1=np.cross(a_2,zhat)*(2*np.pi)/Vol_real
b_2=np.cross(zhat,a_1)*(2*np.pi)/Vol_real
Vol_rec=np.dot(np.cross(b_1,b_2),zhat)
#print(np.dot(a_2,b_2),np.dot(a_1,b_1))



Np=200
n1=np.arange(-Np,Np+1)
n2=np.arange(-Np,Np+1)

a_1=a_1[0:2]
a_2=a_2[0:2]
b_1=b_1[0:2]
b_2=b_2[0:2]

G=np.sqrt(np.sum(b_2**2))
#print(G, b_2)
############################################################
# Function that calculates some high symmetry proints for the FBZ of the triangular lattice
# using a voronoi decomposition of the lattice constructed above
############################################################

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
###################
###################
###################

c= plt.contour(X, Y, Z, levels=[0],linewidths=3, cmap='summer');
#plt.show()
v = c.collections[0].get_paths()[0].vertices
NFSpoints=2000
xFS_dense = v[::int(np.size(v[:,1])/NFSpoints),0]
yFS_dense = v[::int(np.size(v[:,1])/NFSpoints),1]


plt.close()


#T=1.0
if(T==1.0):
    alph=np.array([ 0.7097908959336873,  -0.0043594581070084135,  -0.004495974146928671, -0.024777430963518057,   0.0030982360905670333,   0.0004539363283678258])
    et=np.array([0.23331490064983912,  0.06490355420597822,    -0.03601601298488789,   -0.04655841264762831,    -0.010189892955121571, -0.006643162950435294])
    lam=4.178642027077301

#T=2.0
if(T==2.0):
    alph=np.array([0.6222520758430777,   -0.009461521513823186,     -0.006053452180584596,   -0.010702516424885714,   0.0020003919364024714,     -1.0205060481952911e-05])
    et=np.array([ 0.09658650764378539,  0.013384215891118253,    -0.010516833605162713,      -0.01090143816114151,      0.0004144707853819521,      0.0001204480954048534])   
    lam= 3.370944783098885

#T=3.0
if(T==3.0):
    alph=np.array([ 0.6071415409901372, -0.007643725101933083,    -0.004102812828401734,    -0.0064882051217971795,  0.001523532730774404, 2.9287972860276336e-05 ])
    et=np.array([0.09131678420721018,    0.005815174776661578,  -0.00670989716658747,   -0.006410702279227802,   0.0011528049552485798,  0.0003122379970753175])
    lam=3.1806350971738353 

#T=4.0
if(T==4.0):
    alph=np.array([0.6039735698700098,  -0.006047485346815116,  -0.002472531200563213,  -0.004275139567596024,  0.0013215101055664485,  7.175902842573632e-05])
    et=np.array([0.09553529940513966,    0.0018186539475888166, -0.004739620696629819,   -0.004494332347087183,  0.0013142362882892138,  0.000291236774009811])
    lam=3.106684811722399

#T=5.0
if(T==5.0):
    alph=np.array([0.6040810534137876,  -0.004732865622040766,  -0.001127349144342822,  -0.0027706232289313806,     0.0013410007192414624,      0.0002091028903354993])
    et=np.array([0.1011836355370694,    -0.0009584775768396963, -0.003315265600935579  ,    -0.0033082929540142105,  0.0013589600008588089,  0.00027718680067235644])
    lam=3.0703759314285626 

#T=6.0
if(T==6.0):
    alph=np.array([0.6048825771130464,  0.008438891265335053,   -0.00020510810262157957,    -0.0018087562251271861 ,    0.0012503605738861706, 0.00019497167542968042]) 
    et=np.array([0.106654110761412, -0.0032025150693937156, -0.0021998569308273493, -0.0024967677601908135, 0.0012877521033920923,  0.00021783472445787648])
    lam=3.0498886949148036

#T=7.0
if(T==7.0):
    alph=np.array([0.6060957381074707,  -0.0030658995822706505, 0.0006420851435837167,  -0.0009717074866372171,  0.0013131665577559325, 0.0003018262351056688]) 
    et=np.array([0.11175745410773015,   -0.005039700509756049,  -0.0012315951887033312, -0.001841829723904746,  0.0012274174484950366,   0.00019865688157838827])
    lam=3.037203270341933

#T=8.0
if(T==8.0):
    alph=np.array([0.6070551862962202,  -0.002697101301333131,  0.00107489031858635,    -0.0005590867861957349,     0.0011475930927934638,   0.00018205256220494518])
    et=np.array([ 0.11616954883061426,  -0.003988873565023515,   -0.0004657571027108079,    -0.0013839453692115198, 0.0011044737627527907, 0.00012737532156294055]) 
    lam=3.028807008086399

 
#T=9.0
if(T==9.0):
    alph=np.array([0.6083718387945705,  0.010508584347927811,   0.0018022331604978998,  0.00012473785812128,    0.0013244365310333586,  0.00037723431092535686])
    et=np.array([0.12033706994827463,    -0.007946874704261314, 0.00030420836462192086,  -0.00090404839139119,  0.001061248299460572,   0.00014865459778067692]) 
    lam= 3.0229631820378184


#T=10.0
if(T==10.0):
    alph=np.array([0.6092518069471177,   -0.0017454331191290237,    0.0021259053889015845,   0.0004188012953199125, 0.0012489555790225417,  0.0003255774536971311])
    et=np.array([0.12385676180579733,   -0.009155564378675983,   0.0008941115202702899,      -0.0005938474219710233,    0.0019469008555008608,      0.0001013876862340809])
    lam=3.018732903302169


#T=50.0
if(T==50.0):
    alph=np.array([ 0.6201774944069754, 0.0012530224630861754,   0.005833622305428591,  0.003605079260905222,   0.0014324738571124083,   0.0006316328735677271])
    et=np.array([0.16281297212996357,   -0.021355929675790707,   0.007802291525953633,  0.002565178078007373,   -0.0004250484192336371, -3.285858634623746e-05])
    lam=3.000789439969265

#T=100.0
if(T==100.0):
    alph=np.array([0.6220438193343075,  0.0016537316919072811,  0.006387742935248672,   0.004060505526695932,    0.0014967727700990639,  0.000700872036530507])
    et=np.array([0.1697667873959355,    -0.023474171445420244,   0.009095251231202181,  0.0030821033954326386,  -0.0007082689712385551, -2.655211696552507e-05])
    lam=3.00019867333284

qx=xFS_dense
qy=yFS_dense


def dsf2(qx, qy, f):
    gamma0=1
    gamma1=(1/3.0)*(np.cos(qx)+2*np.cos(qx/2)*np.cos(np.sqrt(3)*qy/2))
    gamma2=(1/3.0)*(2*np.cos(3*qx/2)*np.cos(np.sqrt(3)*qy/2)+np.cos(np.sqrt(3)*qy))
    gamma3=(1/3.0)*(np.cos(2*qx)+2*np.cos(2*qx/2)*np.cos(2*np.sqrt(3)*qy/2))
    gamma4=(1/3.0)*( np.cos(5*qx/2)*np.cos(np.sqrt(3)*qy/2) +np.cos(2*qx)*np.cos(np.sqrt(3)*qy) +np.cos(qx/2)*np.cos(3*np.sqrt(3)*qy/2) )
    gamma5=(1/3.0)*(np.cos(3*qx)+2*np.cos(3*qx/2)*np.cos(3*np.sqrt(3)*qy/2))

    
    sum_et_gam=et[0]*gamma0+et[1]*gamma1+et[2]*gamma2+et[3]*gamma3+et[4]*gamma4+et[5]*gamma5
    et_q=sum_et_gam*((6-6*gamma1)**2)
    alpha_q=alph[0]*gamma0+alph[1]*gamma1+alph[2]*gamma2+alph[3]*gamma3+alph[4]*gamma4+alph[5]*gamma5
    #additional 2 pi for the correct normalization of the frequency integral
    NN=2*np.pi*np.abs( alpha_q*np.sqrt( et_q*( et_q-1 +1j*1e-17) )/np.arcsinh( np.sqrt( (et_q-1+1j*1e-17) ) ) )
    SF_stat=3/(lam+(1/T)*gamma1*6)


   
    sinhal=np.sinh(alpha_q*f)
    fac=NN/(sinhal*sinhal+et_q)
    
    

    return SF_stat*fac # this has to be called in the reverse order for some reason.

angles=np.arctan2(xFS_dense,yFS_dense)

print("shape of the angles array", np.shape(angles))

print("shape of the angles array",np.shape( dsf2(xFS_dense+xFS_dense[0],yFS_dense+yFS_dense[0],0.01) ))

# The parametrized function to be plotted
def f(t, e,  amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t)

t = np.linspace(0, 1, 1000)

# Define initial parameters
init_amplitude = 5
init_frequency = 3

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()

e=1
# line, =plt.plot(np.array(Vertices_list)[:,0],np.array(Vertices_list)[:,1],'o')
# line, =plt.plot([0],[0],'o')
# line, = plt.plot(t, f(t,e, init_amplitude, init_frequency), lw=2)
line, =plt.plot(angles,dsf2(xFS_dense+xFS_dense[int(init_amplitude)],yFS_dense+yFS_dense[int(init_amplitude)],0.01))
ax.set_xlabel('Time [s]')

axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)

# adjust the main plot to make room for the sliders
# plt.gca().set_aspect('equal', adjustable='box')
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
freq_slider = Slider(
    ax=axfreq,
    label='Frequency [Hz]',
    valmin=-10,
    valmax=20,
    valinit=init_frequency,
)

# Make a vertically oriented slider to control the amplitude
axamp = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor=axcolor)
amp_slider = Slider(
    ax=axamp,
    label="Amplitude",
    valmin=0,
    valmax=2000,
    valinit=init_amplitude,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):

    # line.set_xdata(xFS_dense+xFS_dense[int(amp_slider.val)] )
    line.set_ydata( dsf2(xFS_dense+xFS_dense[int(amp_slider.val)],yFS_dense+yFS_dense[int(amp_slider.val)],0.01) ) 
    fig.canvas.draw_idle()


# register the update function with each slider
freq_slider.on_changed(update)
amp_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    freq_slider.reset()
    amp_slider.reset()
button.on_clicked(reset)

plt.show()