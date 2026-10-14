""" This module contains the fitting function for M_F. """

import numpy as np

# Fundamental constants:

c                   =  2.9979245e10       # Speed of ligth (cm/s)
kB                  =  1.3806488e-16      # Boltzmann constant (erg/K)
mH                  =  1.6735327e-24      # Mass of hydrogen atom (g)

# Hydrogen, helium, and dust abundance:

Y                   =  0.247              # Helium mass fraction
X                   =  1.0-Y              # Hydrogen mass fraction
Dsun                =  1/162              # Solar dust-to-gas ratio

# Dust absorption cross-section at Lya at Solar dust abundance,
# per H nucelon (cm^2/H), for different dust models:

sigma_dHsun_HD23    =  6.873e4*Dsun*mH/X  # Hensley & Draine (2023), Astrodust + PAH
sigma_dHsun_WD_Rv3  =  6.854e4*Dsun*mH/X  # Weingartner & Draine (2001), Milky Way, R_V = 3.1
sigma_dHsun_WD_Rv4  =  4.425e4*Dsun*mH/X  # Weingartner & Draine (2001), Milky Way, R_V = 4.0
sigma_dHsun_WD_Rv5  =  3.182e4*Dsun*mH/X  # Weingartner & Draine (2001), Milky Way, R_V = 5.5
sigma_dHsun_WD_LMC  =  7.251e4*Dsun*mH/X  # Weingartner & Draine (2001), LMC average
sigma_dHsun_WD_SMC  =  7.181e4*Dsun*mH/X  # Weingartner & Draine (2001), SMC bar

# Einstein-A coefficients:

A_Lya               =  6.265e8            # Lya Einstein-A coefficient (s^-1)
A_2ph               =  8.2206             # 2s->1s Einstein-A coefficient(s^-1)
A_2p2s              =  8.78e-7            # 2p->2s Einstein-A coefficient(s^-1)
A_2s2p              =  1.597e-9           # 2s->2p Einstein-A coefficient(s^-1)


# Unit conversion factors:

kms                 =  1e5                # 1 km/s in cm/s
pc                  =  3.086e18           # 1 pc in cm
AU                  =  1.496e13           # 1 AU in cm
Msun                =  1.989e33           # Solar mass in g

###############################################################
#######                                                 #######
#######      C O N T I N U U M  A B S O R P T I O N     #######
#######                                                 #######
###############################################################


def tau_cabs(D_Dsun, N_H, N_H2, sigma_dHsun):
    """ The cloud continuum absorption optical depth.
        Arguments:

        D_Dsun:      Dust-to-gas ratio in Solar units
        N_H:         Average total hydrogen (HI+HII+H2) column density (cm^-2)
        N_H2:        Average H2 column density (cm^-2)
        sigma_dHsun: Dust absorption cross-section at Lya at Solar
                     dust abundance, per H nucelon (cm^2/H) """

    # Dust absorption optical depth:

    tau_dabs  = 100.0*(sigma_dHsun*D_Dsun/1e-21)*(N_H/1e23)

    # Raman scattering optical depth:

    tau_H2abs = 2.1e-4*(N_H2/1e21)

    # Total continuum absorption optical depth:

    tau_cabs  = tau_dabs + tau_H2abs

    return tau_cabs

###############################################################
#######                                                 #######
#######  D E S T R U C T I O N  P R O B A B I L I T Y   #######
#######                                                 #######
###############################################################

def p_d(nHI, nHeI, nH2, T, z, M = 0.0, b_s = 0.4):
    """ The Lya destruction probability. Arguments:

        nHI:   The average H I number density (cm^-3)
        nHeI:  The average He I number density (cm^-3)
        nH2:   The average H2 number density (cm^-3)
        T:     Gas temperature
        z:     Redshift
        M:     3D turbulent Mach number (controls turbulent fluctuations)
        b_s:   Turbulent driving parameter (1/3 < b_s < 1). (default = 0.4, i.e. 'natural' mix) """

    # CMB stimulated transition rates:

    Gamma_CMB2p2s = 3.1e-6*(1.0+z)
    Gamma_CMB2s2p = 3.0*Gamma_CMB2p2s

    # 2s -> 2p rates for He I, H2, and H I:

    T_100  = T/100.0         # Temperature normalization
    Turb   = 1 + (b_s*M)**2  # Clumping factor for turbulent medium

    C_HeI  = 2.6e-9*(T_100**0.315)*nHeI*Turb
    
    C_H2   = 1.7e-9*(T_100**0.3)*nH2*Turb
    
    C_HI   = ( 4.1e-11*np.exp(-(1.6/T)**(3/4) - 0.15*T)
           + (1.6e-10 + 1.57e-12*(T**1.1))/(
            (8.9*(T**(-0.55)) + 0.078*(T**0.35))**2.0))*nHI*Turb

    # Total 2s <-> 2p rates:

    C_2s2p = C_HeI + C_H2 + C_HI
    C_2p2s = C_2s2p/3.0

    # Destruction probability from 2p -> 2s:

    R_2p2s = (2/3)*A_2p2s + C_2p2s + Gamma_CMB2p2s
    R_2s2p = A_2s2p + C_2s2p + Gamma_CMB2s2p

    p_dHI  = R_2p2s/((A_Lya/A_2ph)*(A_2ph + R_2s2p) + R_2p2s)

    # Destruction probability from H2 line absorption:

    p_dH2  = np.minimum((33.0*np.exp(-14540.0/T) + 13.0*np.exp(-15564.0/T))/(
             46.0 + 1e3*np.exp(-(1e3/T)**1.5)), 1.3e-3)*(nH2/nHI)

    # Total destruction probability:

    p_d    = p_dHI + p_dH2

    return p_d


def M_F_fit(N_HI_average, T, D_Dsun, taubar_s = 0.0, vmax = 0.0, M = 0.0, b_s = 0.4,
            N_H2_average = 0.0, N_HII_average = 0.0, nHI_average = 1e-8, nHeI_average = 0.0,
            nH2_average = 0.0, z = 0.0, sigma_dHsun = sigma_dHsun_HD23):
    
    """ The Lya force multiplier. Arguments:

        N_HI_average:  Average HI column density (cm^-2).
        
        T:             Temperature (K).
        
        taubar_s:      Relative size of emission region in optical depth.
        
        D_Dsun:        Dust-to-gas ratio in Solar units.
        
        vmax:          Cloud expansion/contraction velocity (km/s).
        
        M:             3D turbulent Mach number (controls turbulent fluctuations).

        b_s:           Turbulent driving parameter (1/3 < b_s < 1). (default = 0.4, i.e. 'natural' mix)

        N_H2_average:  Average H2 column density (cm^-2).

        N_HII_average: Average H II column density (cm^-2).

        nHI_average:   Average HI number density (cm^-3).

        nHeI_average:  Average He I number density (cm^-3).

        nH2_average:   Average H2 number density (cm^-3).
        
        z:             Redshift (default = 0.0).
        
        sigma_dHsun:   Dust absorption cross-section at Lya at Solar dust abundance,
                       per H nucelon (cm^2/H) (default = Hensley & Draine 2023). """


    # Quantities and variables of interest:

    T_100        = T/100.0                        # Convenient temperature normalization
    Turb         = 1.0 + (b_s*M)**2               # Convenient turbulence-related factor       
    
    a_v          = 4.7e-3*(T_100**(-0.5))         # Voigt parameter
    sigma0       = 5.88e-13*(T_100**(-0.5))       # Lya cross-section @ line center
    tau_cl_A     = sigma0*N_HI_average            # Lya optical depth @ line center, using average alpha_0
    tau_cl_tilde = tau_cl_A/(Turb**(2/3))         # Lya optical depth @ line center, using alpha0_tilde
    tau_cl_star  = np.sqrt(tau_cl_tilde*tau_cl_A) # Geometric mean of tau_cl_A & tau_cl_tilde
    atauA        = a_v*tau_cl_A

    # Factor related to spatial source distribution:

    taustar     = np.maximum( 1e-8, taubar_s )
    eta_s       = 3.51*np.exp(-0.674*(taustar**2.42) - 1.21*(taustar**0.414))

    # Factors related to the Lya destruction probability:

    P           = np.sqrt(6.*np.pi)*p_d(nHI_average, nHeI_average, nH2_average, T, z, M, b_s)*tau_cl_star/2.0

    Pcritpoint  = 1.8*np.exp(-(taustar**0.25))
    Pcritext    = 1.5*(taustar**(-0.6))*np.exp(-10.0*taustar) + 3.6

    # Recoil-related factors:

    xbar        = 2.54e-3*(T_100**(-0.5))      # Recoil parameter
    
    xt1         = 1.207*(atauA**(1/3))*xbar/(Turb**(1/9))
    xt1_crit    = 0.55*np.exp(-5.8*(taustar**0.5)) + 0.98
    crec        = 0.54*np.exp(-1e3*taustar) + 0.63*(1.0-np.exp(-1e3*taustar))
    eta_rec     = 0.35 + 0.65*(taustar**1.5)/(taustar**1.5 + 0.00395)
    Grec        = 1.0 + eta_rec*(0.2*((xt1/xt1_crit)**2) + 0.005*((xt1/xt1_crit)**4))


    # Factors related to continuum absorption:

    N_H_average = N_HI_average + N_HII_average + 2*N_H2_average

    epst1       = 0.4707*(atauA**(1/3))*tau_cabs(D_Dsun, N_H_average, N_H2_average, sigma_dHsun)/(Turb**(4/9))

    cabs        = ((1.0-np.exp(-20.0*taustar))/(3.0*(1.0 + (P/50)**0.25 + (P/300)**0.75))
                + 0.17*np.exp(-20.0*taustar) )
    eta_abs     = 1.1*(1.0 - np.exp(-6.54*(taustar**1.06)))
    Fabs        = ( (3.4*epst1)**cabs + (eta_abs*epst1)**(4*cabs))**(3/(4*cabs))

    # Factors related to velocity gradients:

    b           = np.sqrt(2*kB*T/mH)/kms            # Thermal velocity (km/s)
    Rdot        = (np.abs( vmax )/b)/(Turb**(1/3))  # Effective dimensionless expansion/contraction rate

    cvel        = 0.6/(1.0 + (P/1e3)**0.5)
    eta_vel     = 1.0 - np.exp(- (taustar/0.1)**0.7)
    Gvel        = 0.058*(Rdot**0.65)*(1.0-np.exp(-9.5*(taustar**0.75)))*np.exp(-0.02*(Rdot**0.6))
    Fvel        = ((1.0 + 0.75*eta_vel*Rdot)**(2*cvel/3) - 1.0)**(1/cvel)

    # Put it all together:

    A   = 1.0 + (P/(Grec*Pcritpoint)) + (xt1/xt1_crit)**(3*crec) + Fabs + 0.29*Rdot

    M_F = eta_s*(atauA**(1/3))*(Turb**(-4/9))/(
          A**(1/3) + (P/((Grec+Gvel)*Pcritext)) + Fvel)

    return M_F
     

vmax = 800.0
N_HI_average = 1e21
T = 1e4
D_Dsun = 0.1
print("Force multiplier = ",M_F_fit(N_HI_average, T, D_Dsun, taubar_s = 0.1, vmax = vmax, M = 0.0, b_s = 0.4,
            N_H2_average = 0.0, N_HII_average = 0.0, nHI_average = 1.0, nHeI_average = 0.0,
            nH2_average = 0.0, z = 0.0, sigma_dHsun = sigma_dHsun_HD23))