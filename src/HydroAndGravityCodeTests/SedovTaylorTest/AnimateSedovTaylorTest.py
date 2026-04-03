import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe

# ---------------- User inputs ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
filename   = os.path.join(script_dir, "SedovTest_E1.00e+00_rho1_gamma1.67_t4.0_256x256.h5")  # HDF5 in same dir as script
frameskip  = 1
save_mp4   = True
outname    = os.path.join(script_dir, "SedovTaylorAnimation.mp4")
fps        = 20
# ---------------------------------------------

# --- Read data ---
with h5py.File(filename, "r") as f:
    t        = f["Times"][:]               # (N_saved,)
    rho_hist = f["GasDensity_History"][:]  # (N_r, N_Z, N_saved) or (N_saved, N_r, N_Z)
    P_hist   = f["P_History"][:]           # same shape as rho_hist
    r        = f["r"][:]                   # (N_r+1,)
    Z        = f["Z"][:]                   # (N_Z+1,)

# --- Ensure correct axis order: (N_saved, N_r, N_Z) ---
def ensure_NtNrNz(arr):
    if arr.shape[2] == len(t):        # (N_r, N_Z, N_saved)
        return arr.transpose(2, 0, 1) # -> (N_saved, N_r, N_Z)
    elif arr.shape[0] == len(t):      # already (N_saved, N_r, N_Z)
        return arr
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} vs len(t)={len(t)}")

rho_hist = ensure_NtNrNz(rho_hist)
P_hist   = ensure_NtNrNz(P_hist)

Nt, Nr, Nz = rho_hist.shape
frames = np.arange(0, Nt, frameskip)
print(f"Loaded {Nt} snapshots; animating {len(frames)} frames.")

# --- Plotting setup ---
plt.rcParams.update({
    'font.size': 16,
    'axes.linewidth': 1.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True
})

R, ZZ = np.meshgrid(r, Z, indexing='ij')

fig, (ax_rho, ax_P) = plt.subplots(
    1, 2, figsize=(12, 5.5), sharex=True, sharey=True,
    constrained_layout=True
)

# Common axis limits
for ax in (ax_rho, ax_P):
    ax.set_xlim(r[0], r[-1])
    ax.set_ylim(Z[0], Z[-1])
    ax.set_aspect('equal', adjustable='box') # Correct aspect ratio of panels 

ax_rho.set_xlabel("r")
ax_rho.set_ylabel("Z")
ax_P.set_xlabel("r")

# --- Force r and Z axes to have the same tick positions/labels ---
# Choose whatever range/spacing you like; here we base it on the domain
common_min = min(r[0], Z[0])
common_max = max(r[-1], Z[-1])

# e.g. 7 ticks from min to max
common_ticks = np.linspace(common_min, common_max, 7)

for ax in (ax_rho, ax_P):
    ax.set_xticks(common_ticks)
    ax.set_yticks(common_ticks)
    ax.set_xticklabels([f"{val:.2f}" for val in common_ticks])
    ax.set_yticklabels([f"{val:.2f}" for val in common_ticks])

# ---- Density panel ----
vmin_rho, vmax_rho = 0.0, 4.0
im_rho = ax_rho.pcolormesh(
    R, ZZ, rho_hist[0].T,
    vmin=vmin_rho, vmax=vmax_rho,
    shading='auto', cmap='viridis'
)
cb_rho = fig.colorbar(im_rho, ax=ax_rho, pad=0.02)
cb_rho.set_label(r"$\rho$")

# ---- Pressure panel ----
vmin_P, vmax_P = 0.0, 0.5
im_P = ax_P.pcolormesh(
    R, ZZ, P_hist[0].T,
    vmin=vmin_P, vmax=vmax_P,
    shading='auto', cmap='coolwarm'
)
cb_P = fig.colorbar(im_P, ax=ax_P, pad=0.02)
cb_P.set_label(r"$P$")

# ---- Time label (put it over the density panel) ----
txt = ax_rho.text(
    0.05, 0.93, "",
    color='white', fontsize=15,
    transform=ax_rho.transAxes,
    path_effects=[pe.withStroke(linewidth=3, foreground='black')]
)

# --- Animation ---
def init():
    im_rho.set_array(rho_hist[0].T.ravel())
    im_P.set_array(P_hist[0].T.ravel())
    txt.set_text(f"t = {t[0]:.3f}")
    return im_rho, im_P, txt

def update(k):
    im_rho.set_array(rho_hist[k].T.ravel())
    im_P.set_array(P_hist[k].T.ravel())
    txt.set_text(f"t = {t[k]:.3f}")
    return im_rho, im_P, txt

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

# --- Output ---
if save_mp4:
    ani.save(outname, writer="ffmpeg", fps=fps, dpi=200)
    print(f"Saved animation to {outname}")
else:
    plt.show()
