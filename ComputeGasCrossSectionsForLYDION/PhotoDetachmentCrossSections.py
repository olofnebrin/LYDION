import numpy as np
from pathlib import Path

# Physical constants & unit conversions
c      = 2.99792458e10    # Speed of light (cm/s)
h      = 6.626069573e-27  # Planck's constant (erg s)
kB     = 1.380648813e-16  # Boltzmann's constant (erg/K)
sigB   = 5.67051e-5       # Stefan-Boltzmann constant (cgs units)
Lsun   = 3.839e33         # Solar luminosity (erg/s)
Rsun   = 6.9634e10        # Solar radius (cm)
eV     = 1.60217725e-12   # 1 eV in erg
MB2CM2 = 1e-18            # 1 Mb  = 10⁻¹⁸ cm²

""" The black-body intensity at energy E (eV), for a
    temperature Teff (K). """
def Bnu(E, Teff):
    nu  = E * eV / h
    return ((2 * h * (nu**3))/(c**2)) * 1 / (np.exp(h*nu/(kB*Teff)) - 1)

def read_sigma(filename = "HminusCrossSec.dat"):
    """
    Parameters
    ----------
    filename : str or Path
        Text file with two whitespace-separated columns:
        1) photon energy (eV)
        2) σ (Mb)

    Returns
    -------
    E   : ndarray
        Energy grid (eV)
    sigma : ndarray
        Cross-section (cm²)
    """
    E, sigma_Mb = np.loadtxt(filename, unpack=True, comments="#")
    return E, sigma_Mb * MB2CM2     # convert Mb → cm²


def Teff_PopIII(mstar):
    """ The effective temperature Teff (K) of a star of mass mstar (Msun),
        for stellar metallicity Zstar/Zsun = 1e-8. """

    x = np.log10(mstar); x2 = x**2; x3 = x2*x
    # Zero-age main sequence luminosity: 
    L0      = - 4.9125370974699000e-3
    L1      =   4.7854346156440304e+0
    L2      = - 1.0218914733694000e+0
    L3      =   8.0491993338893300e-2
    log10_L =   L0  +  L1 * x  +  L2 * x2  +  L3 * x3
    L       =   (10**log10_L) * Lsun 

    # Zero-age main sequence radius:
    R0      =   4.6071669041284602e-1
    R1      = - 1.2482170192596500e+0
    R2      =   1.1079694078714399e+0
    R3      = - 2.1778915454158501e-1
    log10_R =   R0  +  R1 * x  +  R2 * x2  +  R3 * x3
    R       =   (10**log10_R) * Rsun

    # Get the effective temperature:
    Teff    =   ( L / (4*np.pi*(R**2)*sigB) )**(1/4)

    return Teff


def Teff_PopII(mstar):
    """ The effective temperature Teff (K) of a star of mass mstar (Msun),
        for stellar metallicity Zstar/Zsun = 1e-2. """

    x = np.log10(mstar); x2 = x**2; x3 = x2*x
    # Zero-age main sequence luminosity: 
    L0      = - 4.3387121453610497e-2
    L1      =   4.7002611105589196e+0
    L2      = - 9.3228362169805801e-1
    L3      =   5.8811945206418803e-2
    log10_L =   L0  +  L1 * x  +  L2 * x2  +  L3 * x3
    L       =   (10**log10_L) * Lsun 

    # Zero-age main sequence radius:
    R0      = - 3.6870298415699199e-1
    R1      =   9.4401642726269397e-1
    R2      = - 2.1017534522620901e-1
    R3      =   3.7277105006281598e-2
    log10_R =   R0  +  R1 * x  +  R2 * x2  +  R3 * x3
    R       =   (10**log10_R) * Rsun

    # Get the effective temperature:
    Teff    =   ( L / (4*np.pi*(R**2)*sigB) )**(1/4)

    return Teff

print("Teff for 30 Msun: ", Teff_PopII(30))

def band_average_sigma(E, sigma, mstar, Emin, Emax, PopIII = True):
    """ Band-average cross-section: 
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
    num    = np.trapezoid(sigma[band] * weight, E[band])
    den    = np.trapezoid(weight, E[band])
    return num / den
