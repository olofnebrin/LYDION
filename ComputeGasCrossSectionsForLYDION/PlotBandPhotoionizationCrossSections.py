from PhotoionizationCrossSections import *
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 9.0), sharey = True) 
lw = 2.5
fs = 30


####### O I CROSS-SECTIONS ##########

ax1.text(15.0, 35.0, "O I", fontsize = 1.5*fs)

# EUV3 band (54.4 - infinity eV):
E_EUV3min  = 54.4
E_EUV3max  = 100000.0
E_EUV3     = np.logspace( np.log10(E_EUV3min), np.log10(E_EUV3max), num = 1000 )
sigma_OI_EUV3_PopII  = np.array([band_average_sigma_OI(E_EUV3, m, E_EUV3min, E_EUV3max, False) for m in mstar])
ax1.semilogy( mstar, sigma_OI_EUV3_PopII / MB2CM2, color = "black", linewidth = lw )
ax1.text(50.0, 3.4, "EUV3 (54.4 - $\infty$ eV)", fontsize = fs)

print("\nO I EUV3 cross-section for mstar = 30 Msun:", 
      band_average_sigma_OI(E_EUV3, mchar, E_EUV3min, E_EUV3max, False)/ MB2CM2)

# EUV2 band (24.6 - 54.4 eV):
E_EUV2min  = 24.6
E_EUV2max  = 54.4
E_EUV2     = np.logspace( np.log10(E_EUV2min), np.log10(E_EUV2max), num = 1000 )
sigma_OI_EUV2_PopII  = np.array([band_average_sigma_OI(E_EUV2, m, E_EUV2min, E_EUV2max, False) for m in mstar])
ax1.semilogy( mstar, sigma_OI_EUV2_PopII / MB2CM2, color = "crimson", linewidth = lw )
ax1.text(50.0, 12.0, "EUV2 (24.6 - 54.4 eV)", color = "crimson", fontsize = fs)

print("\nO I EUV2 cross-section for mstar = 30 Msun:", 
      band_average_sigma_OI(E_EUV2, mchar, E_EUV2min, E_EUV2max, False)/ MB2CM2)

# EUV1 band (13.6 - 24.6 eV):
E_EUV1min  = 13.6
E_EUV1max  = 24.6
E_EUV1     = np.logspace( np.log10(E_EUV1min), np.log10(E_EUV1max), num = 1000 )
sigma_OI_EUV1_PopII  = np.array([band_average_sigma_OI(E_EUV1, m, E_EUV1min, E_EUV1max, False) for m in mstar])
ax1.semilogy( mstar, sigma_OI_EUV1_PopII / MB2CM2, color = "deeppink", linewidth = lw )
ax1.text(50.0, 6.3, "EUV1 (13.6 - 24.6 eV)", color = "deeppink", fontsize = fs)

print("\nO I EUV1 cross-section for mstar = 30 Msun:", 
      band_average_sigma_OI(E_EUV1, mchar, E_EUV1min, E_EUV1max, False)/ MB2CM2)


####### O II CROSS-SECTIONS #########

print("\nO II EUV3 cross-section for mstar = 30 Msun:", 
      band_average_sigma_OII(E_EUV3, mchar, E_EUV3min, E_EUV3max, False)/ MB2CM2)

print("\nO II EUV2 cross-section for mstar = 30 Msun:", 
      band_average_sigma_OII(E_EUV2, mchar, E_EUV2min, E_EUV2max, False)/ MB2CM2)

print("\nO II EUV1 cross-section for mstar = 30 Msun:", 
      band_average_sigma_OII(E_EUV1, mchar, E_EUV1min, E_EUV1max, False)/ MB2CM2)

####### C I CROSS-SECTIONS ##########

ax2.text(15.0, 35.0, "C I", fontsize = 1.5*fs)

# EUV3 band (54.4 - infinity eV):
sigma_CI_EUV3_PopII  = np.array([band_average_sigma_CI(E_EUV3, m, E_EUV3min, E_EUV3max, False) for m in mstar])
ax2.semilogy( mstar, sigma_CI_EUV3_PopII / MB2CM2, color = "black", linewidth = lw )
ax2.text(50.0, 1.8, "EUV3 (54.4 - $\infty$ eV)", fontsize = fs)

print("\nC I EUV3 cross-section for mstar = 30 Msun:", 
      band_average_sigma_CI(E_EUV3, mchar, E_EUV3min, E_EUV3max, False)/ MB2CM2)

# EUV2 band (24.6 - 54.4 eV):
sigma_CI_EUV2_PopII  = np.array([band_average_sigma_CI(E_EUV2, m, E_EUV2min, E_EUV2max, False) for m in mstar])
ax2.semilogy( mstar, sigma_CI_EUV2_PopII / MB2CM2, color = "crimson", linewidth = lw )
ax2.text(50.0, 3.85, "EUV2 (24.6 - 54.4 eV)", color = "crimson", fontsize = fs)

print("\nC I EUV2 cross-section for mstar = 30 Msun:", 
      band_average_sigma_CI(E_EUV2, mchar, E_EUV2min, E_EUV2max, False)/ MB2CM2)

# EUV1 band (13.6 - 24.6 eV):
sigma_CI_EUV1_PopII  = np.array([band_average_sigma_CI(E_EUV1, m, E_EUV1min, E_EUV1max, False) for m in mstar])
ax2.semilogy( mstar, sigma_CI_EUV1_PopII / MB2CM2, color = "deeppink", linewidth = lw )
ax2.text(50.0, 8.6, "EUV1 (13.6 - 24.6 eV)", color = "deeppink", fontsize = fs)

print("\nC I EUV1 cross-section for mstar = 30 Msun:", 
      band_average_sigma_CI(E_EUV1, mchar, E_EUV1min, E_EUV1max, False)/ MB2CM2)

# Lyman-Werner band (11.2 - 13.6 eV):
E_LWmin  = 11.2
E_LWmax  = 13.6
E_LW     = np.logspace( np.log10(E_LWmin), np.log10(E_LWmax), num = 1000 )
sigma_CI_LW_PopII  = np.array([band_average_sigma_CI(E_LW, m, E_LWmin, E_LWmax, False) for m in mstar])
ax2.semilogy( mstar, sigma_CI_LW_PopII / MB2CM2, color = "dodgerblue", linewidth = lw )
ax2.text(8.0, 20.0, "Lyman-Werner (11.2 - 13.6 eV)", color = "dodgerblue", fontsize = fs)

print("\nC I Lyman-Werner cross-section for mstar = 30 Msun:", 
      band_average_sigma_CI(E_LW, mchar, E_LWmin, E_LWmax, False)/ MB2CM2)

####### C II CROSS-SECTIONS ##########

print("\nC II EUV3 cross-section for mstar = 30 Msun:", 
      band_average_sigma_CII(E_EUV3, mchar, E_EUV3min, E_EUV3max, False)/ MB2CM2)

print("\nC II EUV2 cross-section for mstar = 30 Msun:", 
      band_average_sigma_CII(E_EUV2, mchar, E_EUV2min, E_EUV2max, False)/ MB2CM2)

print("\nC II EUV1 cross-section for mstar = 30 Msun:", 
      band_average_sigma_CII(E_EUV1, mchar, E_EUV1min, E_EUV1max, False)/ MB2CM2)


ax1.set_ylim(1.0, 60.0)


ax1.set_yticks([1, 10])
ax1.set_yticklabels(["1", "10"])

ax1.set_xlabel("Stellar mass (M$_{\odot}$)", labelpad = 20)
ax2.set_xlabel("Stellar mass (M$_{\odot}$)", labelpad = 20)
ax1.set_ylabel("Cross-section (10$^{-18}$ cm$^2$)", labelpad = 20)

ax1.minorticks_on()
ax1.tick_params(axis='both', which='major', direction='in', length=20, width=2.5)
ax1.tick_params(axis='both', which='minor', direction='in', length=10, width=2.5)
ax1.set_xticks(np.arange(0, 151, 50))
ax2.minorticks_on()
ax2.tick_params(axis='both', which='major', direction='in', length=20, width=2.5)
ax2.tick_params(axis='both', which='minor', direction='in', length=10, width=2.5)
ax2.set_xticks(np.arange(0, 151, 50))
#plt.tight_layout(pad=0.3)
#plt.title("Band-averaged O I and C I\nphotoionization cross-sections", y = 1.04)
fig.text(0.5, 0.97, "Band-averaged O I and C I\nphotoionization cross-sections", 
         ha='center', va='top', fontsize=1.5*fs)

plt.tight_layout(pad=0.0, rect=[0, 0, 1.0, 0.8])

plt.savefig("PhotoIonCIOICrossSections.png", dpi=300)
plt.savefig("PhotoIonCIOICrossSections.pdf")
plt.show()
