import h5py
import numpy as np
import matplotlib.pyplot as plt

Lsun = 3.839e33
mH   = 1.67e-24
X    = 0.76
yr   = 3.1556926e7
kyr  = 1e3 * yr
kms  = 1e5

filename_Lya =  "RHDSim_Lya_32x32_2026-01-10.h5"
#filename_Lya =  "StaticCloudTest.h5"
#filename_Lya =  "RHDSim_Lya_2025-12-23.h5"

with h5py.File(filename_Lya, "r") as file:
    M_F       = file["M_F_History"][:]
    M_F_anti  = file["M_F_History_antigrav_direction"][:]
    L_Lya     = file["L_Lya_History"][:]
    f_esc_LyC = file["f_esc_LyC_History"][:]
    f_esc_Lya = file["f_esc_Lya_History"][:]
    t         = file["Times"][:]


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

last_idx       = np.nonzero(t)[0][-1]
#last_idx_nolya = np.nonzero(t_nolya)[0][-1]

#fig, ax = plt.subplots(2, 1, sharex = True)
plt.figure()
fig, ax = plt.subplots()
fig.set_size_inches(12.5, 7.5)

########## LyC and Lyman-Werner escape fractions ######

ax.semilogy( t[0:last_idx]/kyr, f_esc_LyC[0:last_idx], linewidth = 2.5, color = "orangered",
             linestyle = 'dashed', dash_capstyle = 'round' )

ax.semilogy( t[0:last_idx]/kyr, f_esc_Lya[0:last_idx], linewidth = 2.5, color = "dodgerblue",
             linestyle = 'dashed', dash_capstyle = 'round' )

########## FORCE MULTIPLIER ######

#TotMomentum = np.sum(mom * cell_vols)

ax.semilogy( t[0:last_idx]/kyr, M_F[0:last_idx], linewidth = 2.5, color = "crimson",
             linestyle = 'dashed', dash_capstyle = 'round' )
ax.semilogy( t[0:last_idx]/kyr, M_F_anti[0:last_idx], linewidth = 2.5, color = "crimson" )
#ax.axhline( 18.4, color = "crimson", linestyle = "dashed", linewidth = 2.0)
 
ax.semilogy( t[0:last_idx]/kyr, L_Lya[0:last_idx]/(1e4*Lsun), linewidth = 2.5, color = "black" )

yl = [0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4]
ylabels = ['0.01', '0.1', '1', '10', '100', '10$^{3}$', '10$^{4}$']
ax.set_yticks(yl)
ax.set_yticklabels(ylabels)
#ax.set_title("Force multiplier M$_{F}$ and Lya luminosity", y = 1.04)


pos_MFtext  = min( 1.5 * np.max(M_F[0:last_idx]), 100 )
pos_Lyatext = min( 0.5 * np.max(L_Lya[0:last_idx]) / (1e4*Lsun), 1e5 )
pos_fescLya = 0.3
pos_fescLyC = 0.1

ax.text(10, pos_MFtext, "Lyα force multiplier: M$_{F}$ (solid), M$_{F,antigrav}$ (dashed)", color = "crimson", fontsize = 24)
ax.text(10, pos_Lyatext, "Lyα luminosity L$_{Lyα}$/10$^{4}$ L$_{\odot}$ ", color = "black", fontsize = 24)
ax.text(30, pos_fescLya, "f$_{esc,Lyα}$", color = "dodgerblue", fontsize = 28)
ax.text(130, pos_fescLyC, "f$_{esc,LyC}$", color = "orangered", fontsize = 28)

ax.set_xlabel("Time (kyrs)", labelpad = 20)
ax.set_ylim(0.01, 1e3)
# Set different lengths for major and minor ticks
ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction='in', length=20, width=2.5)
ax.tick_params(axis='both', which='minor', direction='in', length=10, width=2.5)

#plt.tight_layout(pad = 0.4)

plt.show()
