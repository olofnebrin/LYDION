#!/usr/bin/env python3
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ---------------- User inputs ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
filename   = os.path.join(script_dir, "SedovTest_E1.00e+00_rho1_gamma1.67_t0.4_128x128.h5")
outname    = os.path.join(script_dir, "SedovTaylorFourPanels_128.pdf")
save_fig   = True
# ---------------------------------------------

def R_Sedov(t, E = 1.0, rho = 1.0):
    """ The analytical Sedov-Taylor blast wave radius at time t,
        for a blast energy E. Assuming gamma = 5 / 3. """
    R_Sedov = 1.15167 * (E * (t**2.0) / rho)**(1.0/5.0)
    return R_Sedov

# --- Read data ---
with h5py.File(filename, "r") as f:
    t        = f["Times"][:]               # (N_saved,)
    rho_hist = f["GasDensity_History"][:]  # shape varies
    r        = f["r"][:]                   # (Nr+1,)
    Z        = f["Z"][:]                   # (Nz+1,)

# --- Ensure order (Nt, Nr, Nz) ---
def ensure_NtNrNz(arr, Nt):
    if arr.shape[2] == Nt:     # (Nr, Nz, Nt)
        return arr.transpose(2, 0, 1)
    elif arr.shape[0] == Nt:   # already (Nt, Nr, Nz)
        return arr
    else:
        raise ValueError(f"Unexpected shape {arr.shape} vs Nt={Nt}")

rho_hist = ensure_NtNrNz(rho_hist, len(t))
Nt, Nr, Nz = rho_hist.shape

# Meshgrid for pcolormesh
R, ZZ = np.meshgrid(r, Z, indexing='ij')

# ============================================================
# Select four snapshot indices
# ============================================================

t_min = t[0]
t_max = t[-1]
t_targets = np.array([
    t_min,
    t_min  +  (t_max - t_min) / 3,
    t_min  +  2 * (t_max - t_min) / 3,
    t_max
])

indices = [int(np.argmin(np.abs(t - target))) for target in t_targets]
print("t_min =", t_min, " R_Sedov(t_min) =", R_Sedov(t_min))


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
                'size': 26} )
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
plt.rcParams['xtick.major.size'] = 20*0.5
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size'] = 20*0.5
plt.rcParams['ytick.minor.size'] = 8
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'

plt.rcParams['legend.frameon'] = 'False'

plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linewidth'] = 3.2
plt.rcParams['grid.linestyle'] = '--'

# Make ticks appear on all sides and point inwards
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


###############################################################
#######                                                 #######
#######               P L O T  I T !                    #######
#######                                                 #######
###############################################################

# No constrained_layout here so we can pack panels manually
fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 10)
)

# Make panels close together (tweak wspace/hspace if needed)
plt.subplots_adjust(
    left   = 0.0,    # Space for y-labels
    right  = 0.88,   # Space for colorbar
    bottom = 0.10,
    top    = 0.96,
    wspace = 0.15,  # Horizontal gap between panels
    hspace = 0.05   # Vertical gap between panels
)

# Common limits and ticks
r_min, r_max = r[0], r[-1]
Z_min, Z_max = Z[0], Z[-1]
common_min = min(r_min, Z_min)
common_max = max(r_max, Z_max)
common_ticks = np.linspace(common_min, common_max, 7)

for ax in axes.flat:
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(Z_min, Z_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(common_ticks)
    ax.set_yticks(common_ticks)
    ax.set_xticklabels([f"{val:.1f}" for val in common_ticks])
    ax.set_yticklabels([f"{val:.1f}" for val in common_ticks])
    ax.tick_params(
        which = 'both',
        labelsize = 22,
        width = 1.7,
        length = 10
    )

# Density range
#vmin_rho = np.min(rho_hist)
#vmax_rho = np.max(rho_hist)
vmin_rho = 0.0
vmax_rho = 4.0

time_style = dict(
    color="white", fontsize = 25,
    path_effects=[pe.withStroke(linewidth = 3, foreground='black')]
)


images = []
theta = np.linspace(0.0, 2.0*np.pi, 512)  # for Sedov circle

for ax, idx in zip(axes.flat, indices):
    # density image
    im = ax.pcolormesh(
        R, ZZ, rho_hist[idx].T,
        vmin = vmin_rho, vmax = vmax_rho,
        shading = "auto", cmap = "viridis"
    )
    im.set_rasterized(True)
    images.append(im)

    # time label
    ax.text(
        0.05, 0.92,
        f"t = {t[idx]:.5f}",
        transform = ax.transAxes,
        ha = "left", va = "top",
        **time_style
    )

    # Plot analytical solution for comparison:
    Rs = R_Sedov(t[idx], E = 1.0, rho = 1.0)  
    x_circle = Rs * np.cos(theta)
    y_circle = Rs * np.sin(theta)

    ax.plot(
        x_circle, y_circle,
        color = 'white', linestyle = 'dashed', dash_capstyle = 'round', lw = 2.5,
        path_effects=[pe.withStroke(linewidth = 4, foreground='black')]
    )

# Add resolution label:
axes[0, 0].text( 0.1, 0.2, f"Resolution:\n{Nr} × {Nz}", 
                 fontsize = 30, color = 'white')

# Axis labels
axes[0, 0].set_ylabel("Z", fontsize = 25)
axes[1, 0].set_ylabel("Z", fontsize = 25)
axes[1, 0].set_xlabel("r", fontsize = 25)
axes[1, 1].set_xlabel("r", fontsize = 25)

axes[0, 0].tick_params(labelbottom=False)
axes[0, 1].tick_params(labelbottom=False)

# Right column: hide y labels
axes[0, 1].tick_params(labelleft=False)
axes[1, 1].tick_params(labelleft=False)

# Left-bottom panel keeps both x and y numbers
# Top-left keeps only y numbers
# Top-right keeps neither x nor y numbers
# Bottom-right keeps only x numbers

# COLOR BAR:

cbar = fig.colorbar( images[0], ax=axes.ravel().tolist(), pad = 0.02 )
cbar.set_label(r"$\rho$")

# SHOW/SAVE:

if save_fig:
    plt.savefig(outname, dpi = 600, bbox_inches = "tight")
    print(f"Saved figure to {outname}")
else:
    plt.show()
