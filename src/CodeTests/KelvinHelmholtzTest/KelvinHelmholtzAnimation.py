import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe

# ---------------- User inputs ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
filename   = os.path.join(script_dir, "KelvinHelmholtz_Nx1024_Ny1024_t5.00.h5") 
frameskip  = 1
save_mp4   = True
outname    = os.path.join(script_dir, "KelvinHelmholtzDensity.mp4")
fps        = 20
# ---------------------------------------------


# --- Read data ---
with h5py.File(filename, "r") as f:
    t        = f["Times"][:]               # (N_saved,)
    rho_hist = f["GasDensity_History"][:]  # (Nx, Ny, Nt) or (Nt, Nx, Ny)

    # Try Cartesian names first, fall back to cylindrical if needed
    if "X" in f:
        X = f["X"][:]    # (Nx+1,)
    else:
        X = f["r"][:]    # (Nr+1,)

    if "Y" in f:
        Y = f["Y"][:]    # (Ny+1,)
    else:
        Y = f["Z"][:]    # (Nz+1,)

# --- Ensure correct axis order: (Nt, Nx, Ny) ---
def ensure_NtNxNy(arr):
    if arr.shape[2] == len(t):          # (Nx, Ny, Nt)
        return arr.transpose(2, 0, 1)   # -> (Nt, Nx, Ny)
    elif arr.shape[0] == len(t):        # already (Nt, Nx, Ny)
        return arr
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} vs len(t)={len(t)}")

rho_hist = ensure_NtNxNy(rho_hist)

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

# Make ticks appear on all sides and point inwards
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


# --- Plotting setup ---
#plt.rcParams.update({
#    'font.size': 16,
#    'axes.linewidth': 1.8,
#    'xtick.direction': 'in',
#    'ytick.direction': 'in',
#    'xtick.top': True,
#    'ytick.right': True
#})

XX, YY = np.meshgrid(X, Y, indexing='ij')

fig, ax = plt.subplots(
    1, 1, figsize=(11.5, 10.0),
    constrained_layout=True
)

ax.set_xlim(X[0], X[-1])
ax.set_ylim(Y[0], Y[-1])
ax.set_aspect('equal', adjustable='box')  # square cells

ax.set_xlabel("X", labelpad = 20)
ax.set_ylabel("Y", labelpad = 20)

# Optional: common ticks
common_min = min(X[0], Y[0])
common_max = max(X[-1], Y[-1])
common_ticks = np.linspace(common_min, common_max, 7)
ax.set_xticks(common_ticks)
ax.set_yticks(common_ticks)
ax.set_xticklabels([f"{val:.2f}" for val in common_ticks])
ax.set_yticklabels([f"{val:.2f}" for val in common_ticks])

# ---- Density panel ----
vmin_rho = np.min(rho_hist)
vmax_rho = np.max(rho_hist)
print(f"rho range over all snapshots: [{vmin_rho:.3e}, {vmax_rho:.3e}]")

im_rho = ax.pcolormesh(
    XX, YY, rho_hist[0].T,
    vmin=vmin_rho, vmax=vmax_rho,
    shading='auto', cmap='Spectral'
)
cb_rho = fig.colorbar(im_rho, ax=ax, pad=0.02, shrink=0.7)
cb_rho.set_label(r"$\rho$")

# ---- Time label ----
txt = ax.text(
    0.05, 0.93, "",
    color='white', fontsize=30,
    transform=ax.transAxes,
    path_effects=[pe.withStroke(linewidth=3, foreground='black')]
)

# --- Animation ---
def init():
    im_rho.set_array(rho_hist[0].T.ravel())
    txt.set_text(f"t = {t[0]:.3f}")
    return im_rho, txt

def update(k):
    im_rho.set_array(rho_hist[k].T.ravel())
    txt.set_text(f"t = {t[k]:.3f}")
    return im_rho, txt

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

# --- Output ---
if save_mp4:
    ani.save(outname, writer="ffmpeg", fps=fps, dpi=300)
    print(f"Saved animation to {outname}")
else:
    plt.show()
