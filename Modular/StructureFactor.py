import numpy as np
import time
from scipy.interpolate import RegularGridInterpolator # You may have some better interpolation methods

 
class StructureFac_fit:

    #initializes temperature and parameters for the fits
    def __init__(self, T, KX, KY ):

        self.T=T
        self.KX=KX
        self.KY=KY
        ##fit parameters for different temperatures:

        #T=1.0
        if(T==1.0):
            self.alph=np.array([ 0.7097908959336873,  -0.0043594581070084135,  -0.004495974146928671, -0.024777430963518057,   0.0030982360905670333,   0.0004539363283678258])
            self.et=np.array([0.23331490064983912,  0.06490355420597822,    -0.03601601298488789,   -0.04655841264762831,    -0.010189892955121571, -0.006643162950435294])
            self.lam=4.178642027077301

        #T=2.0
        elif(T==2.0):
            self.alph=np.array([0.6222520758430777,   -0.009461521513823186,     -0.006053452180584596,   -0.010702516424885714,   0.0020003919364024714,     -1.0205060481952911e-05])
            self.et=np.array([ 0.09658650764378539,  0.013384215891118253,    -0.010516833605162713,      -0.01090143816114151,      0.0004144707853819521,      0.0001204480954048534])   
            self.lam= 3.370944783098885

        #T=3.0
        elif(T==3.0):
            self.alph=np.array([ 0.6071415409901372, -0.007643725101933083,    -0.004102812828401734,    -0.0064882051217971795,  0.001523532730774404, 2.9287972860276336e-05 ])
            self.et=np.array([0.09131678420721018,    0.005815174776661578,  -0.00670989716658747,   -0.006410702279227802,   0.0011528049552485798,  0.0003122379970753175])
            self.lam=3.1806350971738353 

        #T=4.0
        elif(T==4.0):
            self.alph=np.array([0.6039735698700098,  -0.006047485346815116,  -0.002472531200563213,  -0.004275139567596024,  0.0013215101055664485,  7.175902842573632e-05])
            self.et=np.array([0.09553529940513966,    0.0018186539475888166, -0.004739620696629819,   -0.004494332347087183,  0.0013142362882892138,  0.000291236774009811])
            self.lam=3.106684811722399

        #T=5.0
        elif(T==5.0):
            self.alph=np.array([0.6040810534137876,  -0.004732865622040766,  -0.001127349144342822,  -0.0027706232289313806,     0.0013410007192414624,      0.0002091028903354993])
            self.et=np.array([0.1011836355370694,    -0.0009584775768396963, -0.003315265600935579  ,    -0.0033082929540142105,  0.0013589600008588089,  0.00027718680067235644])
            self.lam=3.0703759314285626 

        #T=6.0
        elif(T==6.0):
            self.alph=np.array([0.6048825771130464,  0.008438891265335053,   -0.00020510810262157957,    -0.0018087562251271861 ,    0.0012503605738861706, 0.00019497167542968042]) 
            self.et=np.array([0.106654110761412, -0.0032025150693937156, -0.0021998569308273493, -0.0024967677601908135, 0.0012877521033920923,  0.00021783472445787648])
            self.lam=3.0498886949148036

        #T=7.0
        elif(T==7.0):
            self.alph=np.array([0.6060957381074707,  -0.0030658995822706505, 0.0006420851435837167,  -0.0009717074866372171,  0.0013131665577559325, 0.0003018262351056688]) 
            self.et=np.array([0.11175745410773015,   -0.005039700509756049,  -0.0012315951887033312, -0.001841829723904746,  0.0012274174484950366,   0.00019865688157838827])
            self.lam=3.037203270341933

        #T=8.0
        elif(T==8.0):
            self.alph=np.array([0.6070551862962202,  -0.002697101301333131,  0.00107489031858635,    -0.0005590867861957349,     0.0011475930927934638,   0.00018205256220494518])
            self.et=np.array([ 0.11616954883061426,  -0.003988873565023515,   -0.0004657571027108079,    -0.0013839453692115198, 0.0011044737627527907, 0.00012737532156294055]) 
            self.lam=3.028807008086399

        
        #T=9.0
        elif(T==9.0):
            self.alph=np.array([0.6083718387945705,  0.010508584347927811,   0.0018022331604978998,  0.00012473785812128,    0.0013244365310333586,  0.00037723431092535686])
            self.et=np.array([0.12033706994827463,    -0.007946874704261314, 0.00030420836462192086,  -0.00090404839139119,  0.001061248299460572,   0.00014865459778067692]) 
            self.lam= 3.0229631820378184


        #T=10.0
        elif(T==10.0):
            self.alph=np.array([0.6092518069471177,   -0.0017454331191290237,    0.0021259053889015845,   0.0004188012953199125, 0.0012489555790225417,  0.0003255774536971311])
            self.et=np.array([0.12385676180579733,   -0.009155564378675983,   0.0008941115202702899,      -0.0005938474219710233,    0.0019469008555008608,      0.0001013876862340809])
            self.lam=3.018732903302169

        #T=50.0
        elif(T==100.0):
            self.alph=np.array([ 0.6201774944069754, 0.0012530224630861754,   0.005833622305428591,  0.003605079260905222,   0.0014324738571124083,   0.0006316328735677271])
            self.et=np.array([0.16281297212996357,   -0.021355929675790707,   0.007802291525953633,  0.002565178078007373,   -0.0004250484192336371, -3.285858634623746e-05])
            self.lam=3.000789439969265
        #T=100.0
        else:
            print("T may not be fitted, setting it to T=1000")
            self.alph=np.array([0.6220438193343075,  0.0016537316919072811,  0.006387742935248672,   0.004060505526695932,    0.0014967727700990639,  0.000700872036530507])
            self.et=np.array([0.1697667873959355,    -0.023474171445420244,   0.009095251231202181,  0.0030821033954326386,  -0.0007082689712385551, -2.655211696552507e-05])
            self.lam=3.00019867333284


        self.params=self.params_fit()
        self.SF_stat=self.Static_SF()
                    
    def __repr__(self):
        return "Structure factorat T={T}".format(T=self.T)

    def Static_SF(self):
        KX=self.KX
        KY=self.KY
        gg2=2*np.cos(KX)+4*np.cos(KX/2)*np.cos(np.sqrt(3)*KY/2)
        return 3/(self.lam+(1/self.T)*gg2)
    
    def Int_Static_SF(self, KX, KY):
        curlyN=np.size(KX)
        return np.sum(self.Static_SF(KX,KY))/curlyN -1

    def params_fit(self):
        KX=self.KX
        KY=self.KY
        ##nearest neighbour expansion
        gamma0=1
        gamma1=(1/3.0)*(np.cos(KX)+2*np.cos(KX/2)*np.cos(np.sqrt(3)*KY/2))
        gamma2=(1/3.0)*(2*np.cos(3*KX/2)*np.cos(np.sqrt(3)*KY/2)+np.cos(np.sqrt(3)*KY))
        gamma3=(1/3.0)*(np.cos(2*KX)+2*np.cos(2*KX/2)*np.cos(2*np.sqrt(3)*KY/2))
        gamma4=(1/3.0)*( np.cos(5*KX/2)*np.cos(np.sqrt(3)*KY/2) +np.cos(2*KX)*np.cos(np.sqrt(3)*KY) +np.cos(KX/2)*np.cos(3*np.sqrt(3)*KY/2) )
        gamma5=(1/3.0)*(np.cos(3*KX)+2*np.cos(3*KX/2)*np.cos(3*np.sqrt(3)*KY/2))

 
        sum_et_gam=self.et[0]*gamma0+self.et[1]*gamma1+self.et[2]*gamma2+self.et[3]*gamma3+self.et[4]*gamma4+self.et[5]*gamma5
        et_q=sum_et_gam*((6-6*gamma1)**2)
        alpha_q=self.alph[0]*gamma0+self.alph[1]*gamma1+self.alph[2]*gamma2+self.alph[3]*gamma3+self.alph[4]*gamma4+self.alph[5]*gamma5
        #additional 2 pi for the correct normalization of the frequency integral
        NN=2*np.pi*np.abs( alpha_q*np.sqrt( et_q*( et_q-1 +1j*1e-17) )/np.arcsinh( np.sqrt( (et_q-1+1j*1e-17) ) ) )

        return [sum_et_gam,et_q,alpha_q,NN]


    def Dynamical_SF( self,kx,ky, f):

        [sum_et_gam,et_q,alpha_q,NN]=self.params
    
        sinhal=np.sinh(alpha_q*f)
        fac=NN/(sinhal*sinhal+et_q)
        return self.SF_stat*fac # this has to be called in the reverse order for some reason.


 
class StructureFac_fit_F:

    #initializes temperature and parameters for the fits
    def __init__(self, T ):

        self.T=T

        ##fit parameters for different temperatures:

        #T=1.0
        if(T==1.0):
            self.alph=np.array([ 0.7097908959336873,  -0.0043594581070084135,  -0.004495974146928671, -0.024777430963518057,   0.0030982360905670333,   0.0004539363283678258])
            self.et=np.array([0.23331490064983912,  0.06490355420597822,    -0.03601601298488789,   -0.04655841264762831,    -0.010189892955121571, -0.006643162950435294])
            self.lam=4.178642027077301

        #T=2.0
        elif(T==2.0):
            self.alph=np.array([0.6222520758430777,   -0.009461521513823186,     -0.006053452180584596,   -0.010702516424885714,   0.0020003919364024714,     -1.0205060481952911e-05])
            self.et=np.array([ 0.09658650764378539,  0.013384215891118253,    -0.010516833605162713,      -0.01090143816114151,      0.0004144707853819521,      0.0001204480954048534])   
            self.lam= 3.370944783098885

        #T=3.0
        elif(T==3.0):
            self.alph=np.array([ 0.6071415409901372, -0.007643725101933083,    -0.004102812828401734,    -0.0064882051217971795,  0.001523532730774404, 2.9287972860276336e-05 ])
            self.et=np.array([0.09131678420721018,    0.005815174776661578,  -0.00670989716658747,   -0.006410702279227802,   0.0011528049552485798,  0.0003122379970753175])
            self.lam=3.1806350971738353 

        #T=4.0
        elif(T==4.0):
            self.alph=np.array([0.6039735698700098,  -0.006047485346815116,  -0.002472531200563213,  -0.004275139567596024,  0.0013215101055664485,  7.175902842573632e-05])
            self.et=np.array([0.09553529940513966,    0.0018186539475888166, -0.004739620696629819,   -0.004494332347087183,  0.0013142362882892138,  0.000291236774009811])
            self.lam=3.106684811722399

        #T=5.0
        elif(T==5.0):
            self.alph=np.array([0.6040810534137876,  -0.004732865622040766,  -0.001127349144342822,  -0.0027706232289313806,     0.0013410007192414624,      0.0002091028903354993])
            self.et=np.array([0.1011836355370694,    -0.0009584775768396963, -0.003315265600935579  ,    -0.0033082929540142105,  0.0013589600008588089,  0.00027718680067235644])
            self.lam=3.0703759314285626 

        #T=6.0
        elif(T==6.0):
            self.alph=np.array([0.6048825771130464,  0.008438891265335053,   -0.00020510810262157957,    -0.0018087562251271861 ,    0.0012503605738861706, 0.00019497167542968042]) 
            self.et=np.array([0.106654110761412, -0.0032025150693937156, -0.0021998569308273493, -0.0024967677601908135, 0.0012877521033920923,  0.00021783472445787648])
            self.lam=3.0498886949148036

        #T=7.0
        elif(T==7.0):
            self.alph=np.array([0.6060957381074707,  -0.0030658995822706505, 0.0006420851435837167,  -0.0009717074866372171,  0.0013131665577559325, 0.0003018262351056688]) 
            self.et=np.array([0.11175745410773015,   -0.005039700509756049,  -0.0012315951887033312, -0.001841829723904746,  0.0012274174484950366,   0.00019865688157838827])
            self.lam=3.037203270341933

        #T=8.0
        elif(T==8.0):
            self.alph=np.array([0.6070551862962202,  -0.002697101301333131,  0.00107489031858635,    -0.0005590867861957349,     0.0011475930927934638,   0.00018205256220494518])
            self.et=np.array([ 0.11616954883061426,  -0.003988873565023515,   -0.0004657571027108079,    -0.0013839453692115198, 0.0011044737627527907, 0.00012737532156294055]) 
            self.lam=3.028807008086399

        
        #T=9.0
        elif(T==9.0):
            self.alph=np.array([0.6083718387945705,  0.010508584347927811,   0.0018022331604978998,  0.00012473785812128,    0.0013244365310333586,  0.00037723431092535686])
            self.et=np.array([0.12033706994827463,    -0.007946874704261314, 0.00030420836462192086,  -0.00090404839139119,  0.001061248299460572,   0.00014865459778067692]) 
            self.lam= 3.0229631820378184


        #T=10.0
        elif(T==10.0):
            self.alph=np.array([0.6092518069471177,   -0.0017454331191290237,    0.0021259053889015845,   0.0004188012953199125, 0.0012489555790225417,  0.0003255774536971311])
            self.et=np.array([0.12385676180579733,   -0.009155564378675983,   0.0008941115202702899,      -0.0005938474219710233,    0.0019469008555008608,      0.0001013876862340809])
            self.lam=3.018732903302169

        #T=50.0
        elif(T==100.0):
            self.alph=np.array([ 0.6201774944069754, 0.0012530224630861754,   0.005833622305428591,  0.003605079260905222,   0.0014324738571124083,   0.0006316328735677271])
            self.et=np.array([0.16281297212996357,   -0.021355929675790707,   0.007802291525953633,  0.002565178078007373,   -0.0004250484192336371, -3.285858634623746e-05])
            self.lam=3.000789439969265
        #T=100.0
        else:
            print("T may not be fitted, setting it to T=1000")
            self.alph=np.array([0.6220438193343075,  0.0016537316919072811,  0.006387742935248672,   0.004060505526695932,    0.0014967727700990639,  0.000700872036530507])
            self.et=np.array([0.1697667873959355,    -0.023474171445420244,   0.009095251231202181,  0.0030821033954326386,  -0.0007082689712385551, -2.655211696552507e-05])
            self.lam=3.00019867333284
                    
    def __repr__(self):
        return "Structure factorat T={T}".format(T=self.T)

    def Static_SF(self, KX, KY):
        gg2=2*np.cos(KX)+4*np.cos(KX/2)*np.cos(np.sqrt(3)*KY/2)
        return 3/(self.lam+(1/self.T)*gg2)
    
    def Int_Static_SF(self, KX, KY):
        curlyN=np.size(KX)
        return np.sum(self.Static_SF(KX,KY))/curlyN -1

    def Dynamical_SF( self, kx, ky, f):

        gamma0=1
        gamma1=(1/3.0)*(np.cos(kx)+2*np.cos(kx/2)*np.cos(np.sqrt(3)*ky/2))
        gamma2=(1/3.0)*(2*np.cos(3*kx/2)*np.cos(np.sqrt(3)*ky/2)+np.cos(np.sqrt(3)*ky))
        gamma3=(1/3.0)*(np.cos(2*kx)+2*np.cos(2*kx/2)*np.cos(2*np.sqrt(3)*ky/2))
        gamma4=(1/3.0)*( np.cos(5*kx/2)*np.cos(np.sqrt(3)*ky/2) +np.cos(2*kx)*np.cos(np.sqrt(3)*ky) +np.cos(kx/2)*np.cos(3*np.sqrt(3)*ky/2) )
        gamma5=(1/3.0)*(np.cos(3*kx)+2*np.cos(3*kx/2)*np.cos(3*np.sqrt(3)*ky/2))

 
        sum_et_gam=self.et[0]*gamma0+self.et[1]*gamma1+self.et[2]*gamma2+self.et[3]*gamma3+self.et[4]*gamma4+self.et[5]*gamma5
        et_q=sum_et_gam*((6-6*gamma1)**2)
        alpha_q=self.alph[0]*gamma0+self.alph[1]*gamma1+self.alph[2]*gamma2+self.alph[3]*gamma3+self.alph[4]*gamma4+self.alph[5]*gamma5
        #additional 2 pi for the correct normalization of the frequency integral
        NN=2*np.pi*np.abs( alpha_q*np.sqrt( et_q*( et_q-1 +1j*1e-17) )/np.arcsinh( np.sqrt( (et_q-1+1j*1e-17) ) ) )

    
        sinhal=np.sinh(alpha_q*f)
        fac=NN/(sinhal*sinhal+et_q)

        SF_stat=3.0/(self.lam+(1/self.T)*6.0*gamma1)
        return SF_stat*fac # this has to be called in the reverse order for some reason.



class StructureFac_PM:

    #initializes temperature and parameters for the fits
    def __init__(self, T, gamma, vmode, m ):

        self.T=T
        self.gamma=gamma
        self.vmode=vmode
        self.m=m

                    
    def __repr__(self):
        return "Paramagnon Structure factor at T={T}".format(T=self.T)


    def Dynamical_SF(self, qx, qy, f):

        # Chi_var = (gamma*om/((kx**2 +ky**2 +om**2+m**2)**2+(om*gamma)**2))
        Chi_var=0
        ##ZERO MOMENTUM PEAK
        dispi_q=np.sqrt((self.vmode**2)*qx**2 +(self.vmode**2)*qy**2+0.1*self.m**2)
        Chi_var =Chi_var+ (dispi_q*self.gamma*f/((  dispi_q**2 -f**2)**2+(f*self.gamma)**2))
        SFvar=Chi_var*(2+2/(np.exp(f/self.T)-1))

        return SFvar


class StructureFac_PM_Q:

    #initializes temperature and parameters for the fits
    def __init__(self, T, gamma, vmode, m ):

        self.T=T
        self.gamma=gamma
        self.vmode=vmode
        self.m=m

                    
    def __repr__(self):
        return "Paramagnon Structure factor at T={T} with finite momentum".format(T=self.T)


    def Dynamical_SF(self, qx, qy, f):

        #pheno-quali structure factor
        QX1=4*np.pi/3
        QY1=0
        QX2=2*np.pi/3
        QY2=2*np.pi/np.sqrt(3)
        QX3=-2*np.pi/3
        QY3=2*np.pi/np.sqrt(3)

        gamma=self.gamma
        vmode=self.vmode
        m=self.m
        
        
        # Chi_var = (gamma*om/((kx**2 +ky**2 +om**2+m**2)**2+(om*gamma)**2))
        Chi_var=0

        ##ZERO MOMENTUM PEAK with smaller mass than the rest
        dispi_q=np.sqrt((self.vmode**2)*qx**2 +(self.vmode**2)*qy**2+0.1*self.m**2)
        Chi_var =Chi_var+ (dispi_q*gamma*f/((  dispi_q**2 -f**2)**2+(f*gamma)**2))
        
        # FINITE MOMENTUM PEAKS (ASSUMING ALL VELOCITIES ARE THE SAME)
        # FINITE Q MODES ARE ALL ISOTROPIC AND THEIR MASS IS THE SAME AND LARGER THAN ZERO MOMENTUM
        dispi_q=np.sqrt((vmode**2)*(qx-QX1)**2 +(vmode**2)*(qy-QY1)**2+m**2)
        Chi_var =Chi_var+ (dispi_q*gamma*f/((  dispi_q**2 -f**2)**2+(f*gamma)**2))

        dispi_q=np.sqrt((vmode**2)*(qx-QX2)**2 +(vmode**2)*(qy-QY2)**2+m**2)
        Chi_var =Chi_var+ (dispi_q*gamma*f/((  dispi_q**2 -f**2)**2+(f*gamma)**2))

        dispi_q=np.sqrt((vmode**2)*(qx-QX3)**2 +(vmode**2)*(qy-QY3)**2+m**2)
        Chi_var =Chi_var+ (dispi_q*gamma*f/((  dispi_q**2 -f**2)**2+(f*gamma)**2))


        dispi_q=np.sqrt((vmode**2)*(qx+QX1)**2 +(vmode**2)*(qy+QY1)**2+m**2)
        Chi_var =Chi_var+ (dispi_q*gamma*f/((  dispi_q**2 -f**2)**2+(f*gamma)**2))

        dispi_q=np.sqrt((vmode**2)*(qx+QX2)**2 +(vmode**2)*(qy+QY2)**2+m**2)
        Chi_var =Chi_var+ (dispi_q*gamma*f/((  dispi_q**2 -f**2)**2+(f*gamma)**2))

        dispi_q=np.sqrt((vmode**2)*(qx+QX3)**2 +(vmode**2)*(qy+QY3)**2+m**2)
        Chi_var =Chi_var+ (dispi_q*gamma*f/((  dispi_q**2 -f**2)**2+(f*gamma)**2))

        

        SFvar=Chi_var*(2+2/(np.exp(f/self.T)-1))


        return SFvar 


class StructureFac_PM_Q2:

    #initializes temperature and parameters for the fits
    def __init__(self, T, gamma, vmode, m ):

        self.T=T
        self.gamma=gamma
        self.vmode=vmode
        self.m=m

                    
    def __repr__(self):
        return "Paramagnon Structure factor at T={T} with finite momentum".format(T=self.T)


    def Dynamical_SF(self, qx, qy, f):

        #pheno-quali structure factor
        QX1=4*np.pi/3
        QY1=0
        QX2=2*np.pi/3
        QY2=2*np.pi/np.sqrt(3)
        QX3=-2*np.pi/3
        QY3=2*np.pi/np.sqrt(3)

        gamma=self.gamma
        vmode=self.vmode
        m=self.m
        
        
        # Chi_var = (gamma*om/((kx**2 +ky**2 +om**2+m**2)**2+(om*gamma)**2))
        Chi_var=0

        # FINITE MOMENTUM PEAKS (ASSUMING ALL VELOCITIES ARE THE SAME)
        # FINITE Q MODES ARE ALL ISOTROPIC AND THEIR MASS IS THE SAME AND LARGER THAN ZERO MOMENTUM
        dispi_q=np.sqrt((vmode**2)*(qx-QX1)**2 +(vmode**2)*(qy-QY1)**2+m**2)
        Chi_var =Chi_var+ (dispi_q*gamma*f/((  dispi_q**2 -f**2)**2+(f*gamma)**2))

        dispi_q=np.sqrt((vmode**2)*(qx-QX2)**2 +(vmode**2)*(qy-QY2)**2+m**2)
        Chi_var =Chi_var+ (dispi_q*gamma*f/((  dispi_q**2 -f**2)**2+(f*gamma)**2))

        dispi_q=np.sqrt((vmode**2)*(qx-QX3)**2 +(vmode**2)*(qy-QY3)**2+m**2)
        Chi_var =Chi_var+ (dispi_q*gamma*f/((  dispi_q**2 -f**2)**2+(f*gamma)**2))


        dispi_q=np.sqrt((vmode**2)*(qx+QX1)**2 +(vmode**2)*(qy+QY1)**2+m**2)
        Chi_var =Chi_var+ (dispi_q*gamma*f/((  dispi_q**2 -f**2)**2+(f*gamma)**2))

        dispi_q=np.sqrt((vmode**2)*(qx+QX2)**2 +(vmode**2)*(qy+QY2)**2+m**2)
        Chi_var =Chi_var+ (dispi_q*gamma*f/((  dispi_q**2 -f**2)**2+(f*gamma)**2))

        dispi_q=np.sqrt((vmode**2)*(qx+QX3)**2 +(vmode**2)*(qy+QY3)**2+m**2)
        Chi_var =Chi_var+ (dispi_q*gamma*f/((  dispi_q**2 -f**2)**2+(f*gamma)**2))

        

        SFvar=Chi_var*(2+2/(np.exp(f/self.T)-1))


        return SFvar 