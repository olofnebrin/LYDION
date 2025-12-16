import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe

# ---------------- User inputs ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
filename   = os.path.join(script_dir, "RayleighTaylor_512x1536_t15.00.h5")
frameskip  = 1
save_mp4   = True
outname    = os.path.join(script_dir, "RayleighTaylorDensity.mp4")
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

Nt, dim1, dim2 = rho_hist.shape
Nx = len(X) - 1
Ny = len(Y) - 1
print(f"rho_hist frame shape = ({dim1}, {dim2}), expected Nx={Nx}, Ny={Ny}")

# Helper: convert to correct plotting orientation
def to_plot_frame(frame2d):
    if frame2d.shape == (Nx, Ny):
        return frame2d.T
    elif frame2d.shape == (Ny, Nx):
        return frame2d
    else:
        raise ValueError(f"Unexpected frame shape {frame2d.shape}")

frames = np.arange(0, Nt, frameskip)
print(f"Loaded {Nt} snapshots; animating {len(frames)} frames.")


###############################################################
#######           P L O T   C O S M E T I C S           #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 34})
plt.rcParams.update({
    'mathtext.default': 'regular',
    'axes.linewidth': 2.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 18,
    'ytick.major.size': 18,
    'xtick.major.width': 1.8,
    'ytick.major.width': 1.8,
})


###############################################################
#######                 F I G U R E   S E T U P          #######
###############################################################

fig, ax = plt.subplots(figsize=(6, 12))  # square figure
ax.set_xlim(X[0], X[-1])
ax.set_ylim(Y[0], Y[-1])
ax.set_aspect('equal', adjustable='box')  # keeps data aspect correct

ax.set_xlabel("X", labelpad=20)
ax.set_ylabel("Y", labelpad=20)

# Use evenly spaced ticks based on data limits
ax.set_xticks(np.linspace(X[0], X[-1], 6))
ax.set_yticks(np.linspace(Y[0], Y[-1], 7))
ax.tick_params(axis='both', which='major', labelsize=22, pad=10)

###############################################################
#######                 D E N S I T Y   P L O T          #######
###############################################################

vmin_rho = np.nanmin(rho_hist)
vmax_rho = np.nanmax(rho_hist)
print(f"rho range over all snapshots: [{vmin_rho:.3e}, {vmax_rho:.3e}]")

frame0 = to_plot_frame(rho_hist[0])

im_rho = ax.pcolormesh(
    X, Y, frame0,
    vmin=vmin_rho, vmax=vmax_rho,
    shading='auto', cmap='viridis'
)

cb_rho = fig.colorbar(im_rho, ax=ax, pad=0.03, shrink=0.8)
cb_rho.set_label(r"$\rho$", fontsize=26)
cb_rho.ax.tick_params(labelsize=20)

# ---- Time label ----
txt = ax.text(
    0.05, 0.93, "",
    color='white', fontsize=20,
    transform=ax.transAxes,
    path_effects=[pe.withStroke(linewidth=3, foreground='black')]
)

# Apply tight layout to remove top/bottom white margins
plt.tight_layout(pad=0.4)

###############################################################
#######                 A N I M A T I O N                #######
###############################################################

def init():
    frame = to_plot_frame(rho_hist[0])
    im_rho.set_array(frame.ravel())
    txt.set_text(f"t = {t[0]:.3f}")
    return im_rho, txt

def update(k):
    frame = to_plot_frame(rho_hist[k])
    im_rho.set_array(frame.ravel())
    txt.set_text(f"t = {t[k]:.3f}")
    return im_rho, txt

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

# --- Output ---
if save_mp4:
    ani.save(outname, writer="ffmpeg", fps=fps, dpi=300)
    print(f"Saved animation to {outname}")
else:
    plt.show()

