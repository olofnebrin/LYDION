import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ---------------- User inputs ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
filename   = os.path.join(script_dir, "KelvinHelmholtz_Nx1024_Ny1024_t5.00.h5") 
SaveFigure = True
outname    = os.path.join(script_dir, "KelvinHelmholtzDensity_4snapshots.png")
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
t_min, t_max = t[0], t[-1]
print(f"Loaded {Nt} snapshots, t in [{t_min:.3f}, {t_max:.3f}].")

# --- Choose 4 times: 0, tmax/3, 2tmax/3, tmax ---
target_times = np.array([t_min,
                         t_min + (t_max - t_min)/3.0,
                         t_min + 2.0*(t_max - t_min)/3.0,
                         t_max])

# For each target time, pick the closest saved snapshot index
indices = np.array([np.argmin(np.abs(t - τ)) for τ in target_times])
indices = np.unique(indices)  # just in case of duplicates at low Nt

# If we ended up with <4 unique times, just spread evenly in index-space
if len(indices) < 4:
    indices = np.linspace(0, Nt-1, 4, dtype=int)

print("Using snapshot indices and times:")
for k in indices:
    print(f"  i = {k:4d}, t = {t[k]:.3f}")

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
plt.rcParams['xtick.major.size'] = 0*20
plt.rcParams['xtick.minor.size'] = 0*8
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size'] = 0*20
plt.rcParams['ytick.minor.size'] = 0*8
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

# --- Prepare meshgrid ---
XX, YY = np.meshgrid(X, Y, indexing='ij')

# Global rho range over all snapshots for consistent colorbar
vmin_rho = np.min(rho_hist)
vmax_rho = np.max(rho_hist)
print(f"rho range over all snapshots: [{vmin_rho:.3e}, {vmax_rho:.3e}]")

# --- Figure with 4 panels ---
fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 12),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

axes = axes.ravel()

# Common ticks
common_min = min(X[0], Y[0])
common_max = max(X[-1], Y[-1])
common_ticks = np.linspace(common_min, common_max, 7)
tick_labels = [f"{val:.2f}" for val in common_ticks]

# store first im for colorbar
im0 = None

for n, (ax, idx) in enumerate(zip(axes, indices)):
    rho = rho_hist[idx].T  # (Ny, Nx)

    im = ax.pcolormesh(
        XX, YY, rho,
        vmin=vmin_rho, vmax=vmax_rho,
        shading='auto', cmap='Spectral'
    )
    if im0 is None:
        im0 = im

    # Bounds and aspect
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(Y[0], Y[-1])
    ax.set_aspect('equal', adjustable='box')

    # ---- REMOVE ALL NUMERIC TICK LABELS ----
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # ---- KEEP ONLY NECESSARY AXIS LABELS ----
    # Left column gets "Y"
    if n in (0, 2):
        ax.set_ylabel("Y", labelpad=20)
    else:
        ax.set_ylabel("")

    # Bottom row gets "X"
    if n in (2, 3):
        ax.set_xlabel("X", labelpad=20)
    else:
        ax.set_xlabel("")

    # Time overlay
    ax.text(
        0.04, 0.93,
        f"t = {t[idx]:.3f}",
        color='white', fontsize=30,
        transform=ax.transAxes,
        path_effects=[pe.withStroke(linewidth=3, foreground='black')]
    )

# Shared colorbar
cbar = fig.colorbar(im0, ax=axes, pad=0.02, shrink=0.85)
cbar.set_label(r"$\rho$")

#fig.suptitle("Kelvin–Helmholtz Instability: Density Snapshots", y=0.98)


# --- Save or show ---
#if SaveFigure:
#    plt.savefig(outname, dpi=300)
#    print(f"Saved figure to {outname}")
#else:
#    plt.show()

# --- Save or show ---
if SaveFigure:
    plt.savefig(outname, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {outname}")
else:
    plt.show()

