import numpy as np
from PhotoDetachmentCrossSections import *

def sigma_photoion_CI(E):
    # Parameters/functions for C I photoionization (from Verner+ 1996):
    Eth = 11.26
    x   = (E/2.144) - 1.133
    y   = np.sqrt(x**2  +  1.607**2)
    yw  = 0.09157
    ya  = 62.16
    P   = 5.101

    # Put it all together:
    F                 = ((x-1.0)**2 + yw**2) * (y**(0.5*P - 5.5)) * ((1 + (y/ya)**0.5)**(-P))
    sigma_photoion_CI = 5.027e-16 * F

    # Zero out the cross-section below the threshold:
    sigma_photoion_CI = np.where(E >= Eth, sigma_photoion_CI, 0.0)

    return sigma_photoion_CI

def sigma_photoion_CII(E):
    # Parameters/functions for C II photoionization (from Verner+ 1996):
    Eth = 24.38
    x   = (E/0.4058) - 4.929e1
    y   = np.sqrt(x**2  +  3.234**2)
    yw  = 2.093e0
    ya  = 1.261e2
    P   = 8.578e0

    # Put it all together:
    F                  = ((x-1.0)**2 + yw**2) * (y**(0.5*P - 5.5)) * ((1 + (y/ya)**0.5)**(-P))
    sigma_photoion_CII = 8.709e-18 * F

    # Zero out the cross-section below the threshold:
    sigma_photoion_CII = np.where(E >= Eth, sigma_photoion_CII, 0.0)

    return sigma_photoion_CII


def sigma_photoion_OI(E):
    # Parameters/functions for O I photoionization (from Verner+ 1996):
    Eth = 13.62
    x   = (E/1.240) - 8.698
    y   = np.sqrt(x**2  +  0.1271**2)
    yw  = 0.07589
    ya  = 3.784
    P   = 17.64

    # Put it all together:
    F                 = ((x-1.0)**2 + yw**2) * (y**(0.5*P - 5.5)) * ((1 + (y/ya)**0.5)**(-P))
    sigma_photoion_OI = 1.745e-15 * F

    # Zero out the cross-section below the threshold:
    sigma_photoion_OI = np.where(E >= Eth, sigma_photoion_OI, 0.0)

    return sigma_photoion_OI


def sigma_photoion_OII(E):
    # Parameters/functions for O II photoionization (from Verner+ 1996):
    Eth = 35.12
    x   = (E/1.386) - 2.131e1
    y   = np.sqrt(x**2  +  (1.503e-2)**2)
    yw  = 1.934e-2
    ya  = 3.175e1
    P   = 8.943e0

    # Put it all together:
    F                  = ((x-1.0)**2 + yw**2) * (y**(0.5*P - 5.5)) * ((1 + (y/ya)**0.5)**(-P))
    sigma_photoion_OII = 5.967e-17 * F

    # Zero out the cross-section below the threshold:
    sigma_photoion_OII = np.where(E >= Eth, sigma_photoion_OII, 0.0)

    return sigma_photoion_OII


def band_average_sigma_CI(E, mstar, Emin, Emax, PopIII = True):
    """ Band-average C I photoionization cross-section: 
        ⟨σ⟩ = ∫dE σ(E) B_ν(E,T)/E / ∫dE B_ν(E,T)/E dE ,
        over the energy band [Emin, Emax] (eV).
    """
    band = (E >= Emin) & (E <= Emax)
    if not band.any():
        raise ValueError("Energy grid does not cover the band.")
    
    if PopIII == True:
        T  = Teff_PopIII(mstar)
    else:
        T  = Teff_PopII(mstar)
    
    weight = Bnu(E[band], T) / E[band]
    num    = np.trapezoid(sigma_photoion_CI(E[band]) * weight, E[band])
    den    = np.trapezoid(weight, E[band])
    return num / den

def band_average_sigma_CII(E, mstar, Emin, Emax, PopIII = True):
    """ Band-average C II photoionization cross-section: 
        ⟨σ⟩ = ∫dE σ(E) B_ν(E,T)/E / ∫dE B_ν(E,T)/E dE ,
        over the energy band [Emin, Emax] (eV).
    """
    band = (E >= Emin) & (E <= Emax)
    if not band.any():
        raise ValueError("Energy grid does not cover the band.")
    
    if PopIII == True:
        T  = Teff_PopIII(mstar)
    else:
        T  = Teff_PopII(mstar)
    
    weight = Bnu(E[band], T) / E[band]
    num    = np.trapezoid(sigma_photoion_CII(E[band]) * weight, E[band])
    den    = np.trapezoid(weight, E[band])
    return num / den


def band_average_sigma_OI(E, mstar, Emin, Emax, PopIII = True):
    """ Band-average O I photoionization cross-section: 
        ⟨σ⟩ = ∫dE σ(E) B_ν(E,T)/E / ∫dE B_ν(E,T)/E dE ,
        over the energy band [Emin, Emax] (eV).
    """
    band = (E >= Emin) & (E <= Emax)
    if not band.any():
        raise ValueError("Energy grid does not cover the band.")
    
    if PopIII == True:
        T  = Teff_PopIII(mstar)
    else:
        T  = Teff_PopII(mstar)
    
    weight = Bnu(E[band], T) / E[band]
    num    = np.trapezoid(sigma_photoion_OI(E[band]) * weight, E[band])
    den    = np.trapezoid(weight, E[band])
    return num / den


def band_average_sigma_OII(E, mstar, Emin, Emax, PopIII = True):
    """ Band-average O II photoionization cross-section: 
        ⟨σ⟩ = ∫dE σ(E) B_ν(E,T)/E / ∫dE B_ν(E,T)/E dE ,
        over the energy band [Emin, Emax] (eV).
    """
    band = (E >= Emin) & (E <= Emax)
    if not band.any():
        raise ValueError("Energy grid does not cover the band.")
    
    if PopIII == True:
        T  = Teff_PopIII(mstar)
    else:
        T  = Teff_PopII(mstar)
    
    weight = Bnu(E[band], T) / E[band]
    num    = np.trapezoid(sigma_photoion_OII(E[band]) * weight, E[band])
    den    = np.trapezoid(weight, E[band])
    return num / den

