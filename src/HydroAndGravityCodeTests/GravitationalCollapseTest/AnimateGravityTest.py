import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm

from pathlib import Path
import sys

def add_cmastro_to_path():
    here = Path(__file__).resolve()
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
filename   = os.path.join(script_dir, "GravityTest_128x256_Zfull_t0.07.h5") 
frameskip  = 1
save_mp4   = True
outname    = os.path.join(script_dir, "GravCollapseDensity.mp4")
fps        = 20
use_logrho = True        # log density is usually much nicer for collapse
rho_floor  = 1e-30       # for log plot safety (will be max(rho_floor, min positive))
# ---------------------------------------------


def ensure_NtNxNy(arr, tlen):
    """Return array shaped (Nt, Nx, Ny)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D history array, got shape {arr.shape}")
    if arr.shape[2] == tlen:          # (Nx, Ny, Nt)
        return arr.transpose(2, 0, 1) # -> (Nt, Nx, Ny)
    if arr.shape[0] == tlen:          # already (Nt, Nx, Ny)
        return arr
    raise ValueError(f"Unexpected array shape {arr.shape} vs len(t)={tlen}")


def pick_first_existing(f, keys):
    for k in keys:
        if k in f:
            return k
    return None


# --- Read data ---
with h5py.File(filename, "r") as f:
    t       = f["Times"][:]                       # (Nt,)
    r_edges = f["r"][:]                           # (Nr+1,)
    z_edges = f["Z"][:]                           # (Nz+1,)
    rho_rzt = f["GasDensity_History"][:]          # (Nr, Nz, Nt)

# Convert to (Nt, Nr, Nz) for easy frame indexing
rho_hist = np.transpose(rho_rzt, (2, 0, 1))            # (Nt, Nr, Nz)


Nt, Nx, Ny = rho_hist.shape
frames = np.arange(0, Nt, frameskip)
print(f"Loaded {Nt} snapshots; animating {len(frames)} frames.")


###############################################################
#######           P L O T   C O S M E T I C S           #######
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
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['ytick.minor.size'] = 8
plt.rcParams['xtick.major.pad'] = '15'
plt.rcParams['ytick.major.pad'] = '15'

plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linewidth'] = 2.5
plt.rcParams['grid.linestyle'] = '--'

plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


# --- Plotting setup ---
RR, ZZ = np.meshgrid(r_edges, z_edges, indexing='ij')

fig, ax = plt.subplots(1, 1, figsize=(11.5, 10.0), constrained_layout=True)
ax.set_xlim(r_edges[0], r_edges[-1])
ax.set_ylim(z_edges[0], z_edges[-1])
ax.set_aspect('equal', adjustable='box')

# Labels: most likely cylindrical
ax.set_xlabel("r", labelpad=20)
ax.set_ylabel("Z", labelpad=20)

# Color scaling
rho_all = rho_hist
rho_min = np.min(rho_all)
rho_max = np.max(rho_all)
print(f"rho range over all snapshots: [{rho_min:.3e}, {rho_max:.3e}]")

if use_logrho:
    positive = rho_all[rho_all > 0]
    if positive.size == 0:
        raise ValueError("All densities are non-positive; cannot log-scale.")
    #vmin = max(rho_floor, np.min(positive))
    #vmax = rho_max
    vmin = 1e8
    vmax = 1e12
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cb_label = r"$\rho$ (log)"
else:
    norm = None
    cb_label = r"$\rho$"

im_rho = ax.pcolormesh(
    RR, ZZ, rho_hist[0],   
    shading='auto', cmap='cma:laguna',
    norm=norm,
)

cb = fig.colorbar(im_rho, ax=ax, pad=0.02, shrink=0.7)
cb.set_label(cb_label)

# ---- Time label ----
txt = ax.text(
    0.05, 0.93, "",
    color='white', fontsize=30,
    transform=ax.transAxes,
    path_effects=[pe.withStroke(linewidth=3, foreground='black')]
)

def init():
    im_rho.set_array(rho_hist[0].ravel())  
    txt.set_text(f"t = {t[0]:.3e}")
    return im_rho, txt

def update(frame_index):
    k = frame_index
    im_rho.set_array(rho_hist[k].ravel())  
    txt.set_text(f"t = {t[k]:.3e}")
    return im_rho, txt

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

if save_mp4:
    ani.save(outname, writer="ffmpeg", fps=fps, dpi=300)
    print(f"Saved animation to {outname}")
else:
    plt.show()
