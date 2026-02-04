import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from pathlib import Path
import sys

def add_cmastro_to_path():
    here = Path(__file__).resolve()
    # Walk upwards until we find the repo-relative path src/cmastro/src
    for parent in [here.parent, *here.parents]:
        candidate = parent / "src" / "cmastro" / "src"
        if (candidate / "cmastro" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return str(candidate)
    raise RuntimeError("Could not find src/cmastro/src containing cmastro/__init__.py")

cmastro_path = add_cmastro_to_path()
import cmastro



# ---------------- User inputs ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
#filename   = os.path.join(script_dir, "RayleighTaylor_512x1536_t15.00.h5")
filename   = os.path.join(script_dir, "RayleighTaylor_256x768_t15.00.h5")
SaveFigure = True
outname    = os.path.join(script_dir, "RayleighTaylorDensity_4snapshots.pdf")
# ---------------------------------------------

# --- Read data ---
with h5py.File(filename, "r") as f:
    t        = f["Times"][:]               # (N_saved,)
    rho_hist = f["GasDensity_History"][:]  # (Nx, Ny, Nt) or (Nt, Nx, Ny)

    if "X" in f:
        X = f["X"][:]
    else:
        X = f["r"][:]

    if "Y" in f:
        Y = f["Y"][:]
    else:
        Y = f["Z"][:]

def ensure_NtNxNy(arr):
    if arr.shape[2] == len(t):
        return arr.transpose(2, 0, 1)
    elif arr.shape[0] == len(t):
        return arr
    else:
        raise ValueError("Unexpected array shape.")

rho_hist = ensure_NtNxNy(rho_hist)
Nt, Nx_val, Ny_val = rho_hist.shape
Nx = len(X) - 1
Ny = len(Y) - 1

def to_plot_frame(frame2d):
    if frame2d.shape == (Nx, Ny):
        return frame2d.T
    elif frame2d.shape == (Ny, Nx):
        return frame2d
    else:
        raise ValueError("Unexpected RTI frame shape.")

# --- Choose 4 times ---
t_min, t_max = t[0], t[-1]
target_times = np.array([
    t_min,
    t_min + (t_max - t_min)/3,
    t_min + 2*(t_max - t_min)/3,
    t_max
])
indices = np.array([np.argmin(np.abs(t - τ)) for τ in target_times])
if len(np.unique(indices)) < 4:
    indices = np.linspace(0, Nt-1, 4, dtype=int)

print("Using indices:", indices, "Times:", t[indices])

###############################################################
#######           P L O T  C O S M E T I C S             ######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 37})
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

params = {'legend.fontsize': 19}
plt.rcParams.update(params)
plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 1.7
plt.rcParams['xtick.minor.width'] = 1.7
plt.rcParams['xtick.major.size'] = 0*20
plt.rcParams['xtick.minor.size'] = 0*8
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size'] = 0*20
plt.rcParams['ytick.minor.size'] = 0*8
plt.rcParams['xtick.major.pad'] = '15'
plt.rcParams['ytick.major.pad'] = '15'

plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.linewidth'] = 2.2
plt.rcParams['grid.linewidth'] = 2.2
plt.rcParams['grid.linestyle'] = '--'

# Make ticks appear on all sides and point inwards
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

###############################################################
#######               Figure + Panels                     #####
###############################################################

vmin_rho = np.min(rho_hist)
vmax_rho = np.max(rho_hist)

# Turn OFF constrained layout → we will manually adjust spacing
fig, axes = plt.subplots(
    1, 4,
    figsize=(18, 6),   # wide enough figure
    sharey=True,
    constrained_layout=False
)

axes = np.array(axes).ravel()
im0 = None

XX, YY = np.meshgrid(X, Y, indexing="ij")

for n, (ax, idx) in enumerate(zip(axes, indices)):
    frame = to_plot_frame(rho_hist[idx])

    im = ax.pcolormesh(
        X, Y, frame,
        vmin=vmin_rho, vmax=vmax_rho,
        shading='auto', cmap='cma:laguna',
        rasterized=True
    )
    if im0 is None:
        im0 = im

    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(Y[0], Y[-1])
    ax.set_aspect('equal', adjustable='box')

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # X label under each panel
    ax.set_xlabel("X", labelpad=10)

    if n == 0:
        ax.set_ylabel("Y", labelpad=20)

    ax.text(
        0.05, 0.93,
        f"t = {t[idx]:.3f}",
        color='white', fontsize=24,
        transform=ax.transAxes,
        path_effects=[pe.withStroke(linewidth=3, foreground='black')]
    )

###############################################################
#######               Manual Tight Layout                 #####
###############################################################

# Pull panels much closer together
fig.subplots_adjust(
    left=0.06,  # reduce whitespace at left
    right=0.88, # create room for colorbar
    bottom=0.12,
    top=0.97,
    wspace=-0.7 
)

# Colorbar
cbar = fig.colorbar(im0, ax=axes, pad=0.015, shrink=1.0)
cbar.set_label(r"$\rho$", fontsize=28)
cbar.ax.tick_params(labelsize=20)

# Save
if SaveFigure:
    plt.savefig(outname, dpi=300, bbox_inches="tight")
    print("Saved:", outname)
else:
    plt.show()
