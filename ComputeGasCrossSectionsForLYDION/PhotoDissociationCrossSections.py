import numpy as np
from PhotoDetachmentCrossSections import *

# Wavelengths (nm), 
# total cross-sections (cm^2), 
# dissociation cross-sections (cm^2),
# photoionization cross-sections (cm^2):
wavelengthOH, sigmaOHtot, sigmaOHdiss, sigmaOHion = np.loadtxt("OH_0.1nm.txt", unpack=True, comments="#")
wavelengthCH, sigmaCHtot, sigmaCHdiss, sigmaCHion = np.loadtxt("CH_0.1nm.txt", unpack=True, comments="#")
wavelengthCO, sigmaCOtot, sigmaCOdiss, sigmaCOion = np.loadtxt("CO_0.1nm.txt", unpack=True, comments="#")

# Convert from wavelength to energy (eV):
E_OH = (h * c / (wavelengthOH * 1e-7)) / eV
E_CH = (h * c / (wavelengthCH * 1e-7)) / eV
E_CO = (h * c / (wavelengthCO * 1e-7)) / eV

def band_average_sigmaOH(mstar, Emin, Emax):
    """ Band-averaged OH dissociation cross-section (cm^2), assuming a 
        black-body spectrum for a Zstar/Zsun = 0.01 Pop II star of mass mstar (Msun)."""
    return band_average_sigma(E_OH, sigmaOHdiss, mstar, Emin, Emax, False)


def band_average_sigmaCH(mstar, Emin, Emax):
    """ Band-averaged CH dissociation cross-section (cm^2), assuming a 
        black-body spectrum for a Zstar/Zsun = 0.01 Pop II star of mass mstar (Msun)."""
    return band_average_sigma(E_CH, sigmaCHdiss, mstar, Emin, Emax, False)

def band_average_sigmaCO(mstar, Emin, Emax):
    """ Band-averaged CO dissociation cross-section (cm^2), assuming a 
        black-body spectrum for a Zstar/Zsun = 0.01 Pop II star of mass mstar (Msun)."""
    return band_average_sigma(E_CO, sigmaCOdiss, mstar, Emin, Emax, False)
