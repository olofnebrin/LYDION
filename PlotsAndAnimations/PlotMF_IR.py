import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
os.chdir(repo_root)

Lsun = 3.839e33
mH   = 1.67e-24
X    = 0.76
yr   = 3.1556926e7
kyr  = 1e3 * yr
kms  = 1e5

filename_Lya =  "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5"
#filename_Lya =  "StaticCloudTest.h5"
#filename_Lya =  "RHDSim_Lya_2025-12-23.h5"

def read_attr_if_exists(group, name, default=None):
    return group.attrs[name] if name in group.attrs else default


def nearest_curve_point(t_plot, y_plot, t_target):
    """Return the curve point closest to t_target, ignoring non-finite/non-positive y."""

    good = np.isfinite(t_plot) & np.isfinite(y_plot) & (y_plot > 0.0)

    if not np.any(good):
        raise ValueError("No valid positive points available for annotation.")

    idx_good = np.where(good)[0]
    idx      = idx_good[np.argmin(np.abs(t_plot[good] - t_target))]

    return t_plot[idx], y_plot[idx]


def add_curve_arrow(ax, text, x_text, y_text, x_arrow, y_arrow, color, fontsize = 24):
    """Add text with a 90-degree angled arrow pointing to a curve in data coordinates."""

    ax.annotate(text,
                xy = (x_arrow, y_arrow), xycoords = "data",
                xytext = (x_text, y_text), textcoords = "data",
                color = color, fontsize = fontsize,
                ha = "left", va = "center",
                arrowprops = dict(arrowstyle = "-|>",
                                  color = color,
                                  lw = 2.2,
                                  shrinkA = 4, shrinkB = 4,
                                  joinstyle = "miter",
                                  connectionstyle = "angle,angleA=0,angleB=90,rad=0"))

with h5py.File(filename_Lya, "r") as file:
    settings      = file["Settings"]
    M_F_dir       = file["M_F_dir_History"][:]
    M_F_IR        = file["M_F_IR_History"][:]
    M_F_IR_radial = file["M_F_IR_radial_net_History"][:]
    M_F_IR_anti   = file["M_F_IR_History_antigrav_direction"][:]
    L_IR          = file["L_IR_History"][:]
    L_Lya         = file["L_Lya_History"][:]
    L_bol         = read_attr_if_exists(settings, "L_bol", None)
    t             = file["Times"][:]


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

# Use only saved, non-zero times.
t_plot = t[0:last_idx] / kyr

# Plot scaling for luminosities.
L_norm = 1e6 * Lsun

# Time interval over which the bolometric luminosity reference line is drawn.
# These are in kyr, i.e. the same units as the x-axis.
plot_L_bol          = True
L_bol_t_start_kyr   = 0.0
L_bol_t_end_kyr     = 75.0

# Arrow positions.  The arrow head is placed on the curve at the specified time;
# the text position is given explicitly in data coordinates for easy tweaking.
show_arrows             = True
L_IR_arrow_time_kyr     = 64.0
L_IR_text_pos           = (6.0, 0.20)
MF_IR_rad_arrow_time_kyr = 6.0
MF_IR_rad_text_pos      = (12.0, 0.05)

#fig, ax = plt.subplots(2, 1, sharex = True)
plt.figure()
fig, ax = plt.subplots()
fig.set_size_inches(9.5, 7.5)

########## FORCE MULTIPLIER ######

#TotMomentum = np.sum(mom * cell_vols)

ax.semilogy( t_plot, M_F_IR[0:last_idx], linewidth = 3.0, color = "crimson",
             linestyle = 'dashed', dash_capstyle = 'round' )
ax.semilogy( t_plot, M_F_IR_radial[0:last_idx], linewidth = 3.0, color = "crimson",
             linestyle = '-' )
ax.semilogy( t_plot, M_F_IR_anti[0:last_idx], linewidth = 3.0, color = "crimson",
             linestyle = ':' )
#ax.semilogy( t[0:last_idx]/kyr, M_F_dir[0:last_idx], linewidth = 2.5, color = "dodgerblue" )
#ax.axhline( 18.4, color = "crimson", linestyle = "dashed", linewidth = 2.0)
 
ax.semilogy( t_plot, L_IR[0:last_idx]/L_norm, linewidth = 3.0, color = "black" )
#ax.semilogy( t[0:last_idx]/kyr, L_Lya[0:last_idx]/(1e6*Lsun), linewidth = 2.5, color = "dodgerblue" )
if plot_L_bol and L_bol is not None:
    ax.hlines( L_bol/L_norm, xmin = L_bol_t_start_kyr, xmax = L_bol_t_end_kyr,
               color = "blue", linestyle = "-", linewidth = 3.0, zorder = -1 )
    ax.text(1.0, L_bol/L_norm * 1.5, "Stellar bolometric luminosity L$_{bol}$/10$^{6}$ L$_{\\odot}$", color = "blue", fontsize = 26)


yl = [1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4]
ylabels = ['10$^{-3}$', '0.01', '0.1', '1', '10', '100', '10$^{3}$', '10$^{4}$']
ax.set_yticks(yl)
ax.set_yticklabels(ylabels)
#ax.set_title("Force multiplier M$_{F}$ and Lya luminosity", y = 1.04)


pos_MFdirtext  = min( 1.5 * np.max(M_F_dir[0:last_idx]), 0.5 )
pos_MFIRtext   = min( 1.5 * np.max(M_F_IR[0:last_idx]), 100 )
pos_IRtext     = min( 1.5 * np.max(L_IR[0:last_idx]) / L_norm, 20.0 )
#pos_fescLya    = 0.3
#pos_fescLyC = 0.1

#ax.text(0.1, pos_MFdirtext, "Direct radiation pressure: F$_{dir}$ / (L$_{\star}$/c)", color = "dodgerblue", fontsize = 24)
#ax.text(0.1, pos_MFIRtext, "IR radiation pressure: F$_{IR}$ / (L$_{IR}$/c)", color = "crimson", fontsize = 24)
#ax.text(0.1, pos_IRtext, "IR luminosity L$_{IR}$/10$^{6}$ L$_{\\odot}$ ", color = "black", fontsize = 24)

if show_arrows:
    x_arrow, y_arrow = nearest_curve_point(t_plot, L_IR[0:last_idx]/L_norm, L_IR_arrow_time_kyr)
    add_curve_arrow(ax, "IR luminosity L$_{IR}$/10$^{6}$ L$_{\\odot}$",
                    L_IR_text_pos[0], L_IR_text_pos[1],
                    x_arrow, y_arrow, "black", fontsize = 33)

    x_arrow, y_arrow = nearest_curve_point(t_plot, M_F_IR_radial[0:last_idx], MF_IR_rad_arrow_time_kyr)
    add_curve_arrow(ax, "IR trapping factor(s)",
                    MF_IR_rad_text_pos[0], MF_IR_rad_text_pos[1],
                    x_arrow, y_arrow, "crimson", fontsize = 33)
else:
    ax.text(0.1, pos_MFIRtext, "IR radiation pressure: F$_{IR}$ / (L$_{IR}$/c)", color = "crimson", fontsize = 32)
    ax.text(0.1, pos_IRtext, "IR luminosity L$_{IR}$/10$^{6}$ L$_{\\odot}$ ", color = "black", fontsize = 32)
#ax.text(30, pos_fescLya, "f$_{esc,Lyα}$", color = "dodgerblue", fontsize = 28)
#ax.text(130, pos_fescLyC, "f$_{esc,LyC}$", color = "orangered", fontsize = 28)

ax.set_xlabel("Time (kyr)", labelpad = 20)
ax.set_title("IR radiation pressure & luminosity", y = 1.04, fontsize = 35)
ax.set_ylim(1e-3, 40.0)
ax.set_xlim(-4.0, 76.0)

# X ticks.  Useful when the plot is horizontally compressed.
ax.set_xticks(np.arange(0.0, 81.0, 10.0))
ax.set_xticks(np.arange(5.0, 81.0, 5.0), minor = True)

# Set different lengths for major and minor ticks
ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction='in', length=20, width=2.5)
ax.tick_params(axis='both', which='minor', direction='in', length=10, width=2.5)

#plt.tight_layout(pad = 0.4)
plt.savefig("LYDION_Figure_IR_ForceMultiplier_Luminosity.pdf", bbox_inches = "tight")
plt.show()
