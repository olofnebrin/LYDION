from PhotoDetachmentCrossSections import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings("ignore")

# Read in cross-section data:
E, sigma = read_sigma("HminusCrossSec.dat")

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

fig, ax = plt.subplots()
fig.set_size_inches(11.5, 8.8)
lw = 2.5
fs = 30

plt.title("Band-averaged H$^{-}$\nphotodetachment cross-sections", y = 1.04)

# EUV3 band (54.4 - infinity eV):
E_EUV3min  = 54.4
E_EUV3max  = 100000.0
sigma_EUV3_PopIII = np.array([band_average_sigma(E, sigma, m, E_EUV3min, E_EUV3max) for m in mstar])
sigma_EUV3_PopII  = np.array([band_average_sigma(E, sigma, m, E_EUV3min, E_EUV3max, False) for m in mstar])
ax.semilogy( mstar, 3 * sigma_EUV3_PopIII / MB2CM2, color = "black", linewidth = lw )
ax.semilogy( mstar, 3 * sigma_EUV3_PopII / MB2CM2, color = "black", linewidth = lw,
             linestyle = "dashed", dash_capstyle = "round" )
ax.text(50.0, 0.35, "3 × EUV3 (54.4 - $\infty$ eV)", fontsize = fs)

print("\nEUV3 cross-section for mstar = 30 Msun:", 
      band_average_sigma(E, sigma, mchar, E_EUV3min, E_EUV3max)/ MB2CM2)

# EUV2 band (24.6 - 54.4 eV):
E_EUV2min  = 24.6
E_EUV2max  = 54.4
sigma_EUV2_PopIII = np.array([band_average_sigma(E, sigma, m, E_EUV2min, E_EUV2max) for m in mstar])
sigma_EUV2_PopII  = np.array([band_average_sigma(E, sigma, m, E_EUV2min, E_EUV2max, False) for m in mstar])
ax.semilogy( mstar, sigma_EUV2_PopIII / MB2CM2, color = "crimson", linewidth = lw )
ax.semilogy( mstar, sigma_EUV2_PopII / MB2CM2, color = "crimson", linewidth = lw,
             linestyle = "dashed", dash_capstyle = "round" )
ax.text(50.0, 0.9, "EUV2 (24.6 - 54.4 eV)", color = "crimson", fontsize = fs)

print("\nEUV2 cross-section for mstar = 30 Msun:",
      band_average_sigma(E, sigma, mchar, E_EUV2min, E_EUV2max)/ MB2CM2)

# EUV1 band (13.6 - 24.6 eV):
E_EUV1min  = 13.6
E_EUV1max  = 24.6
sigma_EUV1_PopIII = np.array([band_average_sigma(E, sigma, m, E_EUV1min, E_EUV1max) for m in mstar])
sigma_EUV1_PopII  = np.array([band_average_sigma(E, sigma, m, E_EUV1min, E_EUV1max, False) for m in mstar])
ax.semilogy( mstar, sigma_EUV1_PopIII / MB2CM2, color = "deeppink", linewidth = lw )
ax.semilogy( mstar, sigma_EUV1_PopII / MB2CM2, color = "deeppink", linewidth = lw,
             linestyle = "dashed", dash_capstyle = "round" )
ax.text(50.0, 2, "EUV1 (13.6 - 24.6 eV)", color = "deeppink", fontsize = fs)

print("\nEUV1 cross-section for mstar = 30 Msun:",
      band_average_sigma(E, sigma, mchar, E_EUV1min, E_EUV1max)/ MB2CM2)

# Lyman-Werner band (11.2 - 13.6 eV):
E_LWmin  = 11.2
E_LWmax  = 13.6
sigma_LW_PopIII = np.array([band_average_sigma(E, sigma, m, E_LWmin, E_LWmax) for m in mstar])
sigma_LW_PopII  = np.array([band_average_sigma(E, sigma, m, E_LWmin, E_LWmax, False) for m in mstar])
ax.semilogy( mstar, sigma_LW_PopIII / MB2CM2, color = "dodgerblue", linewidth = lw )
ax.semilogy( mstar, sigma_LW_PopII / MB2CM2, color = "dodgerblue", linewidth = lw,
             linestyle = "dashed", dash_capstyle = "round" )
ax.text(50.0, 5, "Lyman-Werner (11.2 - 13.6 eV)", color = "dodgerblue", fontsize = fs)

print("\nLW cross-section for mstar = 30 Msun:",
       band_average_sigma(E, sigma, mchar, E_LWmin, E_LWmax)/ MB2CM2 )

# FUV band (5.8 - 11.2 eV):
E_FUVmin  = 5.8
E_FUVmax  = 11.2
sigma_FUV_PopIII = np.array([band_average_sigma(E, sigma, m, E_FUVmin, E_FUVmax) for m in mstar])
sigma_FUV_PopII  = np.array([band_average_sigma(E, sigma, m, E_FUVmin, E_FUVmax, False) for m in mstar])
ax.semilogy( mstar, sigma_FUV_PopIII / MB2CM2, color = "blue", linewidth = lw )
ax.semilogy( mstar, sigma_FUV_PopII / MB2CM2, color = "blue", linewidth = lw, 
            linestyle = "dashed", dash_capstyle = "round" )
ax.text(50.0, 9, "FUV (5.8 - 11.2 eV)", color = "blue", fontsize = fs)

print("\nFUV cross-section for mstar = 30 Msun:",
       band_average_sigma(E, sigma, mchar, E_FUVmin, E_FUVmax)/ MB2CM2 )

# Optical band (1 - 5.8 eV):
E_optmin  = 1.0
E_optmax  = 5.8
sigma_opt_PopIII = np.array([band_average_sigma(E, sigma, m, E_optmin, E_optmax) for m in mstar])
sigma_opt_PopII  = np.array([band_average_sigma(E, sigma, m, E_optmin, E_optmax, False) for m in mstar])
ax.semilogy( mstar, sigma_opt_PopIII / MB2CM2, color = "mediumseagreen", linewidth = lw )
ax.semilogy( mstar, sigma_opt_PopII / MB2CM2, color = "mediumseagreen", linewidth = lw, 
            linestyle = "dashed", dash_capstyle = "round" )
ax.text(50.0, 24, "Optical (1 - 5.8 eV)", color = "mediumseagreen", fontsize = fs)

print("\nOptical cross-section for mstar = 30 Msun:",
       band_average_sigma(E, sigma, mchar, E_optmin, E_optmax)/ MB2CM2 )


ax.set_xlim( 0.0, 160.0 )
ax.set_ylim( 0.1, 60.0 )

ax.set_yticks([0.1, 1, 10])
ax.set_yticklabels(["0.1", "1", "10"])

ax.set_xlabel("Stellar mass (M$_{\odot}$)", labelpad = 20)
ax.set_ylabel("Cross-section (10$^{-18}$ cm$^2$)", labelpad = 20)

ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction='in', length=20, width=2.5)
ax.tick_params(axis='both', which='minor', direction='in', length=10, width=2.5)
ax.set_xticks(np.arange(0, 151, 50))
plt.tight_layout(pad=0.3)
plt.savefig("HminusBandCrossSections.png", dpi=300)
plt.savefig("HminusBandCrossSections.pdf")
plt.show()