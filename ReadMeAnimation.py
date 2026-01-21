import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

import matplotlib as mpl
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"  


# ---------------- Yser inputs ----------------
filename  = "RHDSim_NoLya_400x400_2026-01-20.h5"
outname   = "ReadMeBanner.gif"   # ".gif" or ".mp4"
fps       = 20
frameskip = 1

# Vertical crop as fraction of full Z-range
z0_frac, z1_frac = 0.00, 0.55

# Display ranges
vmin_nH, vmax_nH = 100.0, 3e4 # For density 
vmin_T,  vmax_T  = 10.0, 3e4  # For temperature 

# Banner size
fig_w, fig_h = 14.0, 4
dpi = 150
# ---------------------------------------------

# Constants
mH  = 1.67e-24
X   = 0.76
pc  = 3.085677581e18

with h5py.File(filename, "r") as f:
    t        = f["Times"][:]
    rho_hist = f["GasDensity_History"][:]
    T_hist   = f["T_History"][:]
    r        = f["r"][:]   # edges (Nr+1,)
    Z        = f["Z"][:]   # edges (Nz+1,)

Nr = len(r) - 1
Nz = len(Z) - 1

Nt_raw, a1, a2 = rho_hist.shape

def ensure_NtNrNz(arr, a1, a2):
    if (a1, a2) == (Nr, Nz):
        return arr
    elif (a1, a2) == (Nz, Nr):
        return arr.transpose(0, 2, 1)
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} vs Nr={Nr}, Nz={Nz}")

rho_hist = ensure_NtNrNz(rho_hist, a1, a2)
T_hist   = ensure_NtNrNz(T_hist,  T_hist.shape[1], T_hist.shape[2])

nH_hist = X * rho_hist / mH

# frames where data exist
saved_idx = np.where((t > 0) | (np.arange(t.size) == 0))[0]
frames = saved_idx[::frameskip]

# crop indices
k0 = int(np.floor(z0_frac * Nz))
k1 = int(np.ceil (z1_frac * Nz))
k0 = max(0, min(k0, Nz-1))
k1 = max(k0+1, min(k1, Nz))

# crop extent in physical units (pc)
rmin, rmax = r[0]/pc, r[-1]/pc
Zmin, Zmax = Z[k0]/pc, Z[k1]/pc
extent = (rmin, rmax, Zmin, Zmax)

def frame_nH(k):
    return nH_hist[k][:, k0:k1].T  # (Nz_crop, Nr)

def frame_T(k):
    return T_hist[k][:, k0:k1].T

fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.0)

axL = fig.add_subplot(gs[0, 0])
axR = fig.add_subplot(gs[0, 1])

for ax in (axL, axR):
    ax.set_axis_off()
    ax.set_facecolor("black")  # optional; avoids white if anything bleeds through

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

# LYDION label:
import matplotlib.patheffects as pe

txt = fig.text(
    0.5, 0.5,
    r"\textsc{L y d i o n}",
    color="white",
    fontsize=80,
    ha="center",
    va="center",
    alpha=0.9,
)

txt.set_path_effects([pe.withStroke(linewidth=3, foreground="black")])



norm_nH = LogNorm(vmin=vmin_nH, vmax=vmax_nH)
norm_T  = LogNorm(vmin=vmin_T,  vmax=vmax_T)

im_nH = axL.imshow(frame_nH(frames[0]),
                   origin="lower",
                   interpolation="nearest",
                   norm=norm_nH,
                   cmap="viridis",
                   extent=extent,
                   aspect="auto")

im_T  = axR.imshow(frame_T(frames[0]),
                   origin="lower",
                   interpolation="nearest",
                   norm=norm_T,
                   cmap="coolwarm",     # your choice
                   extent=extent,
                   aspect="auto")

def init():
    k = frames[0]
    im_nH.set_data(frame_nH(k))
    im_T.set_data(frame_T(k))
    return (im_nH, im_T)

def update(k):
    im_nH.set_data(frame_nH(k))
    im_T.set_data(frame_T(k))
    return (im_nH, im_T)

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

if outname.endswith(".gif"):
    ani.save(outname, writer="pillow", fps=fps)
else:
    ani.save(outname, writer="ffmpeg", fps=fps, bitrate=2500)

print("Wrote:", outname)
