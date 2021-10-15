import numpy as np
import Lattice
import StructureFactor
import Dispersion
import matplotlib.pyplot as plt
import time
 
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



Npoints=100
save=True
l=Lattice.TriangLattice(Npoints, save )
Vol_rec=l.Vol_BZ()

[KX,KY]=l.read_lattice()

T=1.0
SS=StructureFactor.StructureFac(T)
params=SS.params_fit(KX, KY)
SF_stat=SS.Static_SF(KX, KY)
SF_dyna=SS.Dynamical_SF_fit( 0.1, params,SF_stat)

##Plots of the structure factor
# plt.scatter(KX,KY, c=SF_stat)
# plt.show()

# plt.scatter(KX,KY, c=SF_dyna)
# plt.show()


#electronic parameters
J=2*5.17 #in mev
tp1=568/J #in units of Js\
tp2=-tp1*108/568 #/tpp1
##coupling 
U=4000/J
g=100/J
Kcou=g*g/U
mu=25
#mu_vanhove=89.5


#Dos test
# ed=Dispersion.Dispersion_single_band([tp1,tp2],25,0)
# [xFS_dense,yFS_dense]=ed.FS_contour( 4000 )
# [nn, earr,Dos]=ed.DOS(size_E=500, Npoi_ints=1200)
# plt.plot(earr, Dos)
# plt. show()
# plt.plot(nn, Dos)
# plt. show()
# plt.plot(earr, nn)
# plt. show()
# print("Filling is ...",ed.filling)

# ed=Dispersion.Dispersion_single_band([tp1,tp2],250,1)
# [xFS_dense,yFS_dense]=ed.FS_contour( 4000 )
# [nn, earr,Dos]=ed.DOS(size_E=500, Npoi_ints=1200)
# plt.plot(earr, Dos)
# plt. show()
# plt.plot(nn, Dos)
# plt. show()
# plt.plot(earr, nn)
# plt. show()
# print("Filling is ...",ed.filling)

####################################
####################################

##fermi velocity test    # #electronic parameters
    # J=2*5.17 #in mev
    # tp1=568/J #in units of Js\
    # tp2=-tp1*108/568 #/tpp1
    # ##coupling 
    # U=4000/J
    # g=100/J
    # Kcou=g*g/U
    # fill=0.67

    # # # ###params quasicircular FS
    # J=2*5.17 #in mev
    # tp1=568/J #in units of Js
    # tp2=0.065*tp1
    # ##coupling 
    # U=4000/J
    # g=100/J
    # Kcou=g*g/U
    # fill=0.211

    # ed=Dispersion.Dispersion_TB_single_band([tp1,tp2],fill)
    # ed=Dispersion.Dispersion_circ([tp1,tp2],fill)
    # Vertices_list, Gamma, K, Kp, M, Mp=l.FBZ_points(l.b[0,:],l.b[1,:])
    # VV=np.array(Vertices_list+[Vertices_list[0]])
    # [xFS_dense0,yFS_dense0]=ed.FS_contour(1000)
    # plt.plot(VV[:,0], VV[:,1])
    # plt.scatter(xFS_dense0,yFS_dense0, s=1)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    # plt.plot(VV[:,0], VV[:,1])
    # plt.scatter(xFS_dense0+xFS_dense0[0],yFS_dense0+yFS_dense0[0], s=1)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    # plt.scatter(xFS_dense0,yFS_dense0, s=1)
    
    # # Dos test
    # [nn, earr,Dos]=ed.DOS(size_E=500, Npoi_ints=1200)
    # print("Filling is ...",ed.filling)
    # plt.plot(earr, Dos)
    # plt. show()
    # plt.plot(nn, Dos)
    # plt. show()
    # plt.plot(earr, nn)
    # plt. show()

    
    # [xFS_dense0,yFS_dense0]=ed.FS_contour(1000)
    # x = np.linspace(-1.5*np.pi, 1.5*np.pi, 30)
    # X, Y = np.meshgrid(x, x)
    # [vx,vy] = ed.Fermi_Vel(X, Y)
    # Vertices_list, Gamma, K, Kp, M, Mp=l.FBZ_points(l.b[0,:],l.b[1,:])
    # VV=np.array(Vertices_list+[Vertices_list[0]])
    # plt.plot(VV[:,0], VV[:,1])
    # plt.scatter(xFS_dense0,yFS_dense0, s=1)
    # plt.quiver(X,Y,vx,vy)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    

    # [xFS_dense,yFS_dense]=ed.FS_contour(30)
    # x = np.linspace(-1.5*np.pi, 1.5*np.pi, 30)
    # X, Y = np.meshgrid(x, x)
    # [v,u] = ed.Fermi_Vel(xFS_dense,yFS_dense)
    # Vertices_list, Gamma, K, Kp, M, Mp=l.FBZ_points(l.b[0,:],l.b[1,:])
    # VV=np.array(Vertices_list+[Vertices_list[0]])
    # plt.plot(VV[:,0], VV[:,1])
    # # plt.scatter(xFS_dense,yFS_dense, s=1)
    # plt.scatter(xFS_dense0,yFS_dense0, s=1)
    # plt.quiver(xFS_dense,yFS_dense,v,u)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()

    # plt.plot(VV[:,0], VV[:,1])
    # [vx,vy] = ed.Fermi_Vel(xFS_dense0,yFS_dense0)
    # plt.scatter(xFS_dense0,yFS_dense0, c=np.sqrt(vx**2 +vy**2), s=1)
    # # plt.quiver(xFS_dense,yFS_dense,v,u)
    # plt.colorbar()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()

    # plt.plot(VV[:,0], VV[:,1])
    # [vx,vy] = ed.Fermi_Vel(KX,KY)
    # plt.scatter(KX,KY, c=np.sqrt(vx**2 +vy**2), s=1)
    # # plt.quiver(xFS_dense,yFS_dense,v,u)
    # plt.colorbar()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()

####################################
####################################

# Vertices_list, Gamma, K, Kp, M, Mp=l.FBZ_points(l.b[0,:],l.b[1,:])
# #rotated parameters
# tp2=-1.8*tp2
# mu=60
# ed=Dispersion.Dispersion_single_band([tp1,tp2],mu,0)
# print("filling,",ed.filling)
# [xFS_dense,yFS_dense]=ed.FS_contour( 4000)
# plt.scatter(xFS_dense,yFS_dense,s=1)

# plt.plot(np.array(Vertices_list)[:,0],np.array(Vertices_list)[:,1],'o')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()



# [v,u] = ed.Fermi_Vel(KX, KY)
# VF2=v**2+u**2
# Vertices_list, Gamma, K, Kp, M, Mp=l.FBZ_points(l.b[0,:],l.b[1,:])
# VV=np.array(Vertices_list+[Vertices_list[0]])
# plt.plot(VV[:,0], VV[:,1])
# plt.scatter(KX,KY, c=(VF2), s=2)
# plt.scatter(xFS_dense,yFS_dense, s=1, c='r')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.colorbar()
# plt.show()


# [v,u] = ed.Fermi_Vel(xFS_dense,yFS_dense)
# VF2=v**2+u**2
# # Vertices_list, Gamma, K, Kp, M, Mp=l.FBZ_points(l.b[0,:],l.b[1,:])
# # VV=np.array(Vertices_list+[Vertices_list[0]])
# # plt.plot(VV[:,0], VV[:,1])
# # plt.scatter(xFS_dense,yFS_dense, s=1, c=VF2)
# # plt.gca().set_aspect('equal', adjustable='box')
# # plt.colorbar()
# # plt.show()



# plt.scatter(KX,KY, c=SF_dyna2)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

#####################
#######INTEGRATION
#####################
# ############################################################
# # Integration over the FBZ 
# ############################################################
##fermi energy and parameters for the paramagnon propagator

# mu=20
# ed=Dispersion.Dispersion_single_band([tp1,tp2],mu,1)
# [xFS_dense,yFS_dense]=ed.FS_contour( 1000)
# print("Filling is ...",ed.filling)


# EF=ed.EF
# m=EF*2
# gamma=EF*10
# vmode=EF/2
# gcoupl=EF/2

# Npoints=500
# save=True
# l=Lattice.TriangLattice(Npoints, save )
# Vol_rec=l.Vol_BZ()

# [KX,KY]=l.read_lattice()

# print(m)
# T=1
# SS=StructureFactor.StructureFac(T)
# params=SS.params_fit(KX, KY)
# SF_stat=SS.Static_SF(KX, KY)
# s=time.time()
# def integrand_Disp(qx,qy,kx,ky,w):

#     edd=ed.Disp_mu_circ(kx+qx,ky+qy)
#     # edd=ed.Disp_mu_second_NN_triang(kx+qx,ky+qy)
#     om=w-edd
#     # om2=-ed

#     # SFvar=SS.Dynamical_SF_PM_zeroQ(ky,kx, om, gamma, vmode, m)
#     # SFvar=SS.Dynamical_SF_PM_Q(ky,kx, om, gamma, vmode, m)
#     SFvar=SS.Dynamical_SF_fit( om, params , SF_stat)
#     fac_p=np.exp(edd/T)*(1+np.exp(-w/T))/(1+np.exp(edd/T))
#     return Kcou*Kcou*SFvar*2*np.pi*fac_p

# e=time.time()
# print("finished calculation in...",e-s)

# plt.scatter(KX,KY,c=integrand_Disp(0,0,KX,KY,1))
# plt.show()
# shifts=[]
# shifts2=[]
# angles=[]

# print("starting with calculation of Sigma theta w=0.....",np.size(xFS_dense)," points")
# s=time.time()
# typedsf="2"
# for ell in range(np.size(xFS_dense)):

#     KFx=xFS_dense[ell]
#     KFy=yFS_dense[ell]
#     # relevant=np.where(np.log10(integrand_Disp(xFS_dense[inde],yFS_dense[inde],KX,KY,w))>-16)

#     ds=Vol_rec/np.size(KX)
#     S0=np.sum(integrand_Disp(KFx,KFy,KX,KY,0.01)*ds)
#     # S1=np.sum(integrand_Disp(xFS_dense[ell],yFS_dense[ell],xFS_dense,yFS_dense,w)*ds)

#     shifts.append(S0)
#     # shifts2.append(S1)
#     angles.append(np.arctan2(KFy,KFx))
#     # print(ell, np.arctan2(KFy,KFx), S0)
#     printProgressBar(ell + 1, np.size(xFS_dense), prefix = 'Progress:', suffix = 'Complete', length = 50)


# e=time.time()
# print("finished  calculation of Sigma theta w=0.....")
# print("time for calc....",e-s)





# shifts=J*np.array(shifts) #in mev
# plt.scatter(angles, shifts, s=1)
# plt.xlabel(r"$\theta$")
# plt.ylabel(r"-Im$\Sigma (k_F(\theta),0)$ mev,   T="+str(T)+r"$J$")
# plt.ylim([0,10])
# #plt.ylim([6.9,9.8])
# #plt.gca().set_aspect('equal', adjustable='box')
# plt.tight_layout()
# plt.savefig("theta_T_"+str(T)+"_mu_"+str(round(J*mu,2))+"_func_dsf"+typedsf+".png", dpi=200)
# plt.close()

# # shifts2=J*np.array(shifts2) #in mev
# # plt.scatter(angles, shifts2, s=1)
# # plt.xlabel(r"$\theta$")
# # plt.ylabel(r"-Im$\Sigma (k_F(\theta),0)$ mev,   T="+Ta+r"$J$")
# # #plt.ylim([6.9,9.8])
# # #plt.gca().set_aspect('equal', adjustable='box')
# # plt.tight_layout()
# # plt.savefig("2theta_T_"+str(T)+"_mu_"+str((J*mu).round(decimals=2))+"_func_dsf"+typedsf+".png", dpi=200)
# # plt.close()
# Vertices_list, Gamma, K, Kp, M, Mp=l.FBZ_points(l.b[0,:],l.b[1,:])

# plt.plot(np.array(Vertices_list)[:,0],np.array(Vertices_list)[:,1],'o')
# plt.plot([0],[0],'o')
# plt.scatter(xFS_dense,yFS_dense,c=shifts, s=3)
# plt.colorbar()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.tight_layout()
# plt.savefig("scatter_theta_T_"+str(T)+"_mu_"+str(round(J*mu,2))+"_func_dsf"+typedsf+".png", dpi=200)
# plt.close()

# plt.plot(np.array(Vertices_list)[:,0],np.array(Vertices_list)[:,1],'o')
# plt.plot([0],[0],'o')
# plt.scatter(xFS_dense,yFS_dense,c=np.log10(shifts-np.min(shifts)+1e-3), s=3)
# plt.colorbar()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.tight_layout()
# plt.savefig("log_scatter_theta_T_"+str(T)+"_mu_"+str(round(J*mu,2))+"_func_dsf"+typedsf+".png", dpi=200)
# plt.close()













####

#####################
#######INTEGRATION T dep
#####################
# ############################################################
# # Integration over the FBZ 
# ############################################################
mu=20
ed=Dispersion.Dispersion_single_band([tp1,tp2],mu,0)
[xFS_dense,yFS_dense]=ed.FS_contour( 1000)
print("Filling is ...",ed.filling)
# plt.scatter(xFS_dense,yFS_dense)
# plt.show()

EF=ed.EF
m=EF*6
gamma=EF*150
vmode=EF/2
gcoupl=EF/2

Npoints=1000
save=True
l=Lattice.TriangLattice(Npoints, save )
Vol_rec=l.Vol_BZ()

[KX,KY]=l.read_lattice()

print(m)
SS=StructureFactor.StructureFac(T)
params=SS.params_fit(KX, KY)
SF_stat=SS.Static_SF(KX, KY)


def integrand_Disp_fit(qx,qy,kx,ky,w, T, params , SF_stat):

    # edd=ed.Disp_mu_circ(kx+qx,ky+qy)
    edd=ed.Disp_mu_second_NN_triang(kx+qx,ky+qy)
    om=w-edd
    # om2=-ed

    # SFvar=1#/(1+ed.nb(w-edd, T))
    SFvar=SS.Dynamical_SF_fit( om, params , SF_stat)

    fac_p=(1+np.exp(-w/T))*(1-ed.nf(edd, T))
    # fac_p=ed.nb(w-edd, T)+ed.nf(-edd, T)
    return Kcou*Kcou*SFvar*2*np.pi*fac_p

def integrand_Disp_chi(qx,qy,kx,ky,w, T, gamma, vmode, m):

    # edd=ed.Disp_mu_circ(kx+qx,ky+qy)
    edd=ed.Disp_mu_second_NN_triang(kx+qx,ky+qy)
    om=w-edd
    # om2=-ed

    # SFvar=1#/(1+ed.nb(w-edd, T))
    # SFvar=SS.Dynamical_SF_PM_zeroQ(ky,kx, om, gamma, vmode, m)
    SFvar=SS.Dynamical_SF_PM_Q(ky,kx, om, gamma, vmode, m)

    fac_p=(1+np.exp(-w/T))*(1-ed.nf(edd, T))
    # fac_p=ed.nb(w-edd, T)+ed.nf(-edd, T)
    return Kcou*Kcou*SFvar*2*np.pi*fac_p

T=1
plt.scatter(KX,KY,c=integrand_Disp_fit(0,0,KX,KY,0.01, T, params , SF_stat))
plt.colorbar()
plt.show()
Ts= np.arange(1,10)
# Ts=[1,2]
# Ts=np.linspace(0.001,0.5,10)
for T in Ts:

    SS=StructureFactor.StructureFac(T)
    params=SS.params_fit(KX, KY)
    SF_stat=SS.Static_SF(KX, KY)

    shifts=[]
    shifts2=[]
    angles=[]

    print("starting with calculation of Sigma theta w=0.....",np.size(xFS_dense)," points")
    s=time.time()
    for ell in range(np.size(xFS_dense)):

        KFx=xFS_dense[ell]
        KFy=yFS_dense[ell]
        # relevant=np.where(np.log10(integrand_Disp(xFS_dense[inde],yFS_dense[inde],KX,KY,w))>-16)

        ds=Vol_rec/np.size(KX)
        S0=np.sum(integrand_Disp_fit(KFx,KFy,KX,KY,0.0, T, params , SF_stat)*ds)
        # S0=np.sum(integrand_Disp_chi(KFx,KFy,KX,KY,0.0, T, gamma, vmode, m)*ds)
        # S1=np.sum(integrand_Disp(xFS_dense[ell],yFS_dense[ell],xFS_dense,yFS_dense,w)*ds)

        shifts.append(S0)
        # shifts2.append(S1)
        angles.append(np.arctan2(KFy,KFx))
        # print(ell, np.arctan2(KFy,KFx), S0)
        printProgressBar(ell + 1, np.size(xFS_dense), prefix = 'Progress:', suffix = 'Complete', length = 50)


    e=time.time()
    print("finished  calculation of Sigma theta w=0.....")
    print("time for calc....",e-s)





    shifts=J*np.array(shifts) #in mev
    plt.scatter(angles, shifts, s=1, label="T="+str(T)+"J")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"-Im$\Sigma (k_F(\theta),0)$ mev")
    # plt.ylim([0,10])
    #plt.ylim([6.9,9.8])
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

# plt.savefig("theta_Ts_mu_"+str(round(J*mu,2))+"_func_dsf"+typedsf+".png", dpi=200)
# plt.close()
plt.legend()
plt.savefig("theta_Ts_test.png", dpi=200)
plt.close()