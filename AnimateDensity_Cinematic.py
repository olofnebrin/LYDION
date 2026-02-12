# AnimateDensity_Cinematic_ViewCentered.py

import h5py
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
cmastro_src = repo_root / "src" / "cmastro" / "src"
if cmastro_src.exists():
    sys.path.insert(0, str(cmastro_src))
    import cmastro

from PlotCosmetics import (
    ApplyFontStyle, CinematicAxes, AddTimeLabel, InsetColorBar, AddScaleBar, cmap_for, AddStar
)

# ---------------- user inputs ----------------
filename    = "RHDSim_NoLya_512x1024_2026-02-10.h5"
frameskip   = 1
save_mp4    = True
outname     = "density_cinematic.mp4"
fps         = 20
dpi         = 300

# view half-size (pc)
half_pc     = 4.0

# scale bar
scalebar_pc = 1.0

# dataset convention
midplane_symmetry = False   # True => center at z=0 ; False => center at z=Zmax/2

# pretty mirroring options
mirror_r = False             # show ±r
mirror_z = False             # show ±z (purely visual since your Z grid is 0..Zmax)
# ---------------------------------------------

# constants
mH  = 1.67e-24
X   = 0.76
yr  = 3.1556926e7
kyr = 1e3 * yr
pc  = 3.085677581e18

eps = 1e-30  # protect LogNorm

def ensure_NtNrNz(arr, Nr, Nz):
    a1, a2 = arr.shape[1], arr.shape[2]
    if (a1, a2) == (Nr, Nz):
        return arr
    if (a1, a2) == (Nz, Nr):
        return arr.transpose(0, 2, 1)
    raise ValueError(f"Unexpected array shape {arr.shape} vs edges Nr={Nr}, Nz={Nz}")

def mirror_edges_from_zero(edges):
    """edges assumed to start at 0. Returns [-L..L] without duplicating 0."""
    neg = -edges[:0:-1]
    return np.concatenate([neg, edges])

def mirror_field(A, mirror_r=True, mirror_z=True):
    B = A
    if mirror_r:
        B = np.concatenate([B[::-1, :], B], axis=0)
    if mirror_z:
        B = np.concatenate([B[:, ::-1], B], axis=1)
    return B

# --- read data ---
with h5py.File(filename, "r") as f:
    t        = f["Times"][:]
    rho_hist = f["GasDensity_History"][:]
    T_hist   = f["T_History"][:]
    r        = f["r"][:]
    Z        = f["Z"][:]

Nr = len(r) - 1
Nz = len(Z) - 1
rho_hist = ensure_NtNrNz(rho_hist, Nr, Nz)

saved_idx = np.where((t > 0) | (np.arange(t.size) == 0))[0]
frames = saved_idx[::frameskip]

# number density (Nt, Nr, Nz)
nH_hist = X * rho_hist / mH

# edges in pc (you confirmed r,Z stored in cm)
r_edges = r / pc
z_edges = Z / pc

# Decide camera center from your conventions
if midplane_symmetry:
    r_center = 0.0
    z_center = 0.0
else:
    r_center = 0.0
    z_center = 0.5 * z_edges[-1]

# Build plotting edges and mesh consistent with mirror flags
r_plot = mirror_edges_from_zero(r_edges) if mirror_r else r_edges
z_plot = mirror_edges_from_zero(z_edges) if mirror_z else z_edges
Rplot, Zplot = np.meshgrid(r_plot, z_plot, indexing="ij")

# Set view limits
if mirror_r:
    ax_xlim = (-half_pc, half_pc)
else:
    ax_xlim = (max(r_edges[0], r_center - half_pc), r_center + half_pc)

if mirror_z:
    ax_ylim = (-half_pc, half_pc)
else:
    ax_ylim = (max(z_edges[0], z_center - half_pc), z_center + half_pc)

# Norm: choose something robust (tweak if desired)
vmin_nH, vmax_nH = 10, 1e5
norm_nH = LogNorm(vmin=vmin_nH, vmax=vmax_nH, clip=True)

# --- style + figure ---
ApplyFontStyle(theme="cinematic", font_size=32)
fig, ax = CinematicAxes(figsize=(8, 8))

ax.set_xlim(*ax_xlim)
ax.set_ylim(*ax_ylim)

# --- star marker ---
star_r_pc = 0.0
star_z_pc = 0.0 if midplane_symmetry else z_center

AddStar(ax, star_r_pc, star_z_pc, size=220)

# initial frame
k0 = frames[0]

# IMPORTANT: nH_hist[k] is (Nr, Nz) already; do NOT transpose when using indexing="ij"
A0 = np.clip(nH_hist[k0], eps, None)
A0p = mirror_field(A0, mirror_r=mirror_r, mirror_z=mirror_z)

im = ax.pcolormesh(Rplot, Zplot, A0p, norm=norm_nH, shading="auto", cmap=cmap_for("density"))

cb = InsetColorBar(
    fig, ax, im,
    rect=(0.86, 0.1, 0.02, 0.20),
    label=r"n$_{H}$ (cm$^{-3}$)",
    labelsize=18,
    tick_labelsize=18
)
# Use ticks that match the norm range
cb.set_ticks([10.0, 1e3, 1e5])
cb.set_ticklabels([r"10", r"$10^{3}$", r"$10^{5}$"])

txt = AddTimeLabel(ax, "", pos=(0.03, 0.97), fontsize=22)
AddScaleBar(ax, length=scalebar_pc, label=f"{scalebar_pc:.1f} pc", pos=(0.08, 0.08), fontsize=18)

def init():
    k = frames[0]
    A = np.clip(nH_hist[k], eps, None)
    A = mirror_field(A, mirror_r=mirror_r, mirror_z=mirror_z)
    im.set_array(A.ravel())
    txt.set_text(f"t = {t[k]/kyr:.2f} kyr")
    return im, txt

def update(k):
    A = np.clip(nH_hist[k], eps, None)
    A = mirror_field(A, mirror_r=mirror_r, mirror_z=mirror_z)
    im.set_array(A.ravel())
    txt.set_text(f"t = {t[k]/kyr:.2f} kyr")
    return im, txt

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

if save_mp4:
    ani.save(outname, writer="ffmpeg", fps=fps, dpi=dpi)
else:
    import matplotlib.pyplot as plt
    plt.show()
