from PhotoDissociationCrossSections import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings("ignore")

# Stellar mass range to plot, equal to the range the fit is valid for:
mstar = np.linspace(8.0, 160.0, num = 300)

# Also print data for this particular stellar mass:
mchar = 30.0


###############################################################
#######                                                 #######
#######           P L O T   C O S M E T I C S           #######
#######                                                 #######
###############################################################

# COMMENT THIS OUT IF YOU DON'T HAVE THE REQUIRED FONTS:

#plt.rc('font',**{'sans-serif':['RomanD'],
                 #'size': 27} )
#plt.rc('font',**{'sans-serif':['AVHershey Simplex'],
                #'size': 27} )
plt.rc('font',**{'sans-serif':['AVHershey Complex'],
                'size': 37} )
#plt.rc('font',**{'sans-serif':['Complex'],
                # 'size': 27} )

params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

# The stuff below need not be removed:

params = {'legend.fontsize': 19 }
plt.rcParams.update(params)

plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 1.7
plt.rcParams['xtick.minor.width'] = 1.7
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['ytick.minor.size'] = 8
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'

plt.rcParams['legend.frameon'] = 'False'

plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linewidth'] = 2.5
plt.rcParams['grid.linestyle'] = '--'

# Make ticks appear on all sides and point inwards
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True

###############################################################
#######                                                 #######
#######               P L O T  I T !                    #######
#######                                                 #######
###############################################################


plt.figure()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22.5, 9), sharey = True) 
lw = 2.5
fs = 29

############## OH DISSOCIATION CROSS-SECTION ##############

ax1.text(15.0, 35.0, "OH", fontsize = 1.5*fs)

# Lyman-Werner band (11.2 - 13.6 eV): 
E_LWmin  = 11.2
E_LWmax  = 13.6
sigma_OH_LW_PopII  = np.array([band_average_sigmaOH(m, E_LWmin, E_LWmax) for m in mstar])
ax1.semilogy( mstar, sigma_OH_LW_PopII / MB2CM2, color = "dodgerblue", linewidth = lw )
ax1.text(7.0, 4.9, "Lyman-Werner (11.2 - 13.6 eV)", color = "dodgerblue", fontsize = fs)

print("\nOH Lyman-Werner cross-section for mstar = 30 Msun:", 
      band_average_sigmaOH(mchar, E_LWmin, E_LWmax)/ MB2CM2)

# FUV band (5.8 - 11.2 eV): 
E_FUVmin  = 5.8
E_FUVmax  = 11.2
sigma_OH_FUV_PopII  = np.array([band_average_sigmaOH(m, E_FUVmin, E_FUVmax) for m in mstar])
ax1.semilogy( mstar, sigma_OH_FUV_PopII / MB2CM2, color = "blue", linewidth = lw )
ax1.text(50.0, 1.2, "FUV (5.8 - 11.2 eV)", color = "blue", fontsize = fs)

print("\nOH FUV cross-section for mstar = 30 Msun:", 
      band_average_sigmaOH(mchar, E_FUVmin, E_FUVmax)/ MB2CM2)


############## CH DISSOCIATION CROSS-SECTION ##############

ax2.text(15.0, 35.0, "CH", fontsize = 1.5*fs)

# Lyman-Werner band (11.2 - 13.6 eV): 
sigma_CH_LW_PopII  = np.array([band_average_sigmaCH(m, E_LWmin, E_LWmax) for m in mstar])
ax2.semilogy( mstar, sigma_CH_LW_PopII / MB2CM2, color = "dodgerblue", linewidth = lw )
ax2.text(8.0, 1.35, "Lyman-Werner (11.2 - 13.6 eV)", color = "dodgerblue", fontsize = fs)

print("\nCH Lyman-Werner cross-section for mstar = 30 Msun:", 
      band_average_sigmaCH(mchar, E_LWmin, E_LWmax)/ MB2CM2)

# FUV band (5.8 - 11.2 eV): 
sigma_CH_FUV_PopII  = np.array([band_average_sigmaCH(m, E_FUVmin, E_FUVmax) for m in mstar])
ax2.semilogy( mstar, sigma_CH_FUV_PopII / MB2CM2, color = "blue", linewidth = lw )
ax2.text(50.0, 4.1, "FUV (5.8 - 11.2 eV)", color = "blue", fontsize = fs)

print("\nCH FUV cross-section for mstar = 30 Msun:", 
      band_average_sigmaCH(mchar, E_FUVmin, E_FUVmax)/ MB2CM2)

# Optical band (1 - 5.8 eV): 
E_optmin  = 1.0
E_optmax  = 5.8
sigma_CH_opt_PopII  = np.array([band_average_sigmaCH(m, E_optmin, E_optmax) for m in mstar])
ax2.semilogy( mstar, sigma_CH_opt_PopII / MB2CM2, color = "mediumseagreen", linewidth = lw )
ax2.text(50.0, 0.45, "opt (1 - 5.8 eV)", color = "mediumseagreen", fontsize = fs)

print("\nCH optical cross-section for mstar = 30 Msun:", 
      band_average_sigmaCH(mchar, E_optmin, E_optmax)/ MB2CM2)

############## CO DISSOCIATION CROSS-SECTION ##############

ax3.text(15.0, 35.0, "CO", fontsize = 1.5*fs)

# Lyman-Werner band (11.2 - 13.6 eV): 
sigma_CO_LW_PopII  = np.array([band_average_sigmaCO(m, E_LWmin, E_LWmax) for m in mstar])
ax3.semilogy( mstar, sigma_CO_LW_PopII / MB2CM2, color = "dodgerblue", linewidth = lw )
ax3.text(8.0, 11.0, "Lyman-Werner (11.2 - 13.6 eV)", color = "dodgerblue", fontsize = fs)

print("\nCO Lyman-Werner cross-section for mstar = 30 Msun:", 
      band_average_sigmaCO(mchar, E_LWmin, E_LWmax)/ MB2CM2)
print("\n")

ax1.set_xlabel("Stellar mass (M$_{\odot}$)", labelpad = 20)
ax2.set_xlabel("Stellar mass (M$_{\odot}$)", labelpad = 20)
ax3.set_xlabel("Stellar mass (M$_{\odot}$)", labelpad = 20)
ax1.set_ylabel("Cross-section (10$^{-18}$ cm$^2$)", labelpad = 20)

ax1.set_ylim(0.1, 100.0)

ax1.set_yticks([0.1, 1, 10, 100])
ax1.set_yticklabels(["0.1", "1", "10", "100"])

ax1.minorticks_on()
ax1.tick_params(axis='both', which='major', direction='in', length=20, width=2.5)
ax1.tick_params(axis='both', which='minor', direction='in', length=10, width=2.5)
ax1.set_xticks(np.arange(0, 151, 50))
ax2.minorticks_on()
ax2.tick_params(axis='both', which='major', direction='in', length=20, width=2.5)
ax2.tick_params(axis='both', which='minor', direction='in', length=10, width=2.5)
ax2.set_xticks(np.arange(0, 151, 50))
ax3.minorticks_on()
ax3.tick_params(axis='both', which='major', direction='in', length=20, width=2.5)
ax3.tick_params(axis='both', which='minor', direction='in', length=10, width=2.5)
ax3.set_xticks(np.arange(0, 151, 50))
fig.text(0.5, 0.97, "Band-averaged OH, CH, and CO\nphotodissociation cross-sections", 
         ha='center', va='top', fontsize=1.5*fs)

plt.tight_layout(pad=0.0, rect=[0, 0, 1.0, 0.8])

plt.savefig("PhotoDissociationCrossSections.png", dpi=300)
plt.savefig("PhotoDissociationCrossSections.pdf")
plt.show()




