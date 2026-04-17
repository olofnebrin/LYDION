# AnimateDensityTemperature_TwoPanel_Cinematic_ViewCentered.py
#
# Two-panel cinematic animation:
#   Left  panel: density (nH) exactly like your current script
#   Right panel: temperature (T)
#
# Each panel keeps the SAME aspect/zoom as your original single panel.
# (We simply place two identical "cinematic axes" side-by-side.)
#
# Notes:
# - Uses T dataset name "T_History"
# - Uses blit=False (avoids the Matplotlib _resize_id crash you hit)
# - You can freely tweak T_vmin/T_vmax below

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
    ApplyFontStyle, AddTimeLabel, InsetColorBar, AddScaleBar, cmap_for, AddStar
)

# ---------------- user inputs ----------------
#filename    = "RHDSim_NoLya_512x1024_2026-02-10.h5"
filename    = "RHDSim_NoLya_256x512_2026-04-13.h5" 
frameskip   = 1
save_mp4    = True
outname     = "density_temperature_cinematic_2panel.mp4"
fps         = 20
dpi         = 300

# view half-size (pc)
half_pc     = 1.0

# scale bar
scalebar_pc = 0.5

# dataset convention
midplane_symmetry = False   # True => center at z=0 ; False => center at z=Zmax/2

# pretty mirroring options
mirror_r = False            # show ±r
mirror_z = False            # show ±z


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

def CinematicAxesTwoPanel(figsize=(16, 8), gap=0.02):
    """
    Create a full-canvas figure with two axis panels, left and right.
    gap is the fractional gap between panels in figure coordinates.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    # total width = 1.0; split into two panels with a small gap
    w = (1.0 - gap) / 2.0

    axL = fig.add_axes([0.0,     0.0, w, 1.0])
    axR = fig.add_axes([w + gap, 0.0, w, 1.0])

    axL.set_axis_off()
    axR.set_axis_off()

    return fig, axL, axR

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
T_hist   = ensure_NtNrNz(T_hist,   Nr, Nz)

saved_idx = np.where((t > 0) | (np.arange(t.size) == 0))[0]
frames = saved_idx[::frameskip]

# number density (Nt, Nr, Nz)
nH_hist = X * rho_hist / mH

# ---------------------------------------------

# Density:
# --- automatic density range ---
# Use the initial snapshot to define a characteristic initial cloud density,
# then center the plotted range on that value with ±3 dex.

nH0 = nH_hist[0]

nH0_pos = nH0[nH0 > 0.0]
if nH0_pos.size == 0:
    raise ValueError("Initial nH field has no positive values.")

nH0_char = np.max(nH0_pos)

# Density range: 2 dex below and above the initial cloud density
delta   = 0.4
vmin_nH = 10.0**(np.log10(nH0_char) - 2.0 - delta)
vmax_nH = 10.0**(np.log10(nH0_char) + 2.0 - delta)
norm_nH = LogNorm(vmin=vmin_nH, vmax=vmax_nH)

# Nice colorbar ticks at integer dex
logmin_nH = int(np.ceil(np.log10(vmin_nH)))
logmax_nH = int(np.floor(np.log10(vmax_nH)))
ticks_nH  = 10.0**np.arange(logmin_nH, logmax_nH + 1)

def fmt_log_tick(val):
    exp = int(np.round(np.log10(val)))
    if exp == -1:
        return '0.1'
    elif exp == 0:
        return '1'
    elif exp == 1:
        return '10'
    else:
        return f'10$^{{{exp}}}$'

ticklabels_nH = [fmt_log_tick(v) for v in ticks_nH]

# Temperature:
T_vmin, T_vmax = 1.0, 1e5   # for HII-focused, try (1e3, 2e4) or (3e3, 3e4)
norm_T = LogNorm(vmin=T_vmin, vmax=T_vmax, clip=True)

# edges in pc
r_edges = r / pc
z_edges = Z / pc

# camera center
if midplane_symmetry:
    r_center = 0.0
    z_center = 0.0
else:
    r_center = 0.0
    z_center = 0.5 * z_edges[-1]

# plotting edges and mesh
r_plot = mirror_edges_from_zero(r_edges) if mirror_r else r_edges
z_plot = mirror_edges_from_zero(z_edges) if mirror_z else z_edges
Rplot, Zplot = np.meshgrid(r_plot, z_plot, indexing="ij")


# view limits (same for both panels)
if mirror_r:
    ax_xlim = (-half_pc, half_pc)
else:
    # Keep the same physical span as ylim (which is 2*half_pc),
    # but r is >= 0 so we use [0, 2*half_pc] (shifted window).
    ax_xlim = (max(r_edges[0], r_center), r_center + 2.0 * half_pc)

if mirror_z:
    ax_ylim = (-half_pc, half_pc)
else:
    ax_ylim = (max(z_edges[0], z_center - half_pc), z_center + half_pc)




# --- style + figure ---
ApplyFontStyle(theme="cinematic", font_size=32)

# Make the *overall* figure 2x width so each panel is same size as your old 8x8
fig, axL, axR = CinematicAxesTwoPanel(figsize=(16, 8), gap=0.02)


for ax in (axL, axR):
    ax.set_xlim(*ax_xlim)
    ax.set_ylim(*ax_ylim)
    ax.set_aspect("equal", adjustable="box")


# --- star marker on both ---
star_r_pc = 0.0 if mirror_r else 0.0 + 0.002 * ax_xlim[1]  # Small offset if not mirrored to avoid sitting on the edge
star_z_pc = 0.0 if midplane_symmetry else z_center
AddStar(axL, star_r_pc, star_z_pc, size=300)
AddStar(axR, star_r_pc, star_z_pc, size=300)

# initial frame
k0 = frames[0]

A0 = mirror_field(np.clip(nH_hist[k0], eps, None), mirror_r=mirror_r, mirror_z=mirror_z)
T0 = mirror_field(np.clip(T_hist[k0],  eps, None), mirror_r=mirror_r, mirror_z=mirror_z)

# LEFT: Density
imL = axL.pcolormesh(Rplot, Zplot, A0, norm=norm_nH, shading="auto", cmap=cmap_for("density"))
cbL = InsetColorBar(
    fig, axL, imL,
    rect=(0.43, 0.10, 0.012, 0.20),  # near the middle, but inside left half
    label=r"n$_{H}$ (cm$^{-3}$)",
    labelsize=18,
    tick_labelsize=18
)

cbL.set_ticks([10.0, 1e3, 1e5])
cbL.set_ticklabels([r"10", r"$10^{3}$", r"$10^{5}$"])
#cbL.set_ticks([1e3, 1e5, 1e7])
#cbL.set_ticklabels([r"$10^{3}$", r"$10^{5}$", r"$10^{7}$"])

# RIGHT: Temperature
imR = axR.pcolormesh(Rplot, Zplot, T0, norm=norm_T, shading="auto", cmap=cmap_for("temperature"))
cbR = InsetColorBar(
    fig, axR, imR,
    rect=(0.9, 0.10, 0.012, 0.20),  # right edge of the figure
    label=r"T (K)",
    labelsize=18,
    tick_labelsize=18,
    labelcolor="black",
    tickcolors="black"
)
# choose reasonable ticks for log T
cbR.set_ticks([10.0, 1e2, 1e3, 1e4])
cbR.set_ticklabels([r"10", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$"])

# time label (put on left panel)
txt = AddTimeLabel(axL, "", pos=(0.03, 0.97), fontsize=22)

# scale bar (put on left panel; you can also add to right if you want)
if mirror_r:
    pos_ScaleBar = (0.08, 0.15)  # Lower left corner
else:
    pos_ScaleBar = (0.65, 0.10)    # Upper right corner (adjust if needed)

AddScaleBar(axL, length=scalebar_pc, label=f"{scalebar_pc:.1f} pc", 
            pos=pos_ScaleBar, fontsize=18)


def init():
    k = frames[0]

    A = mirror_field(np.clip(nH_hist[k], eps, None), mirror_r=mirror_r, mirror_z=mirror_z)
    T = mirror_field(np.clip(T_hist[k],  eps, None), mirror_r=mirror_r, mirror_z=mirror_z)

    imL.set_array(A.ravel())
    imR.set_array(T.ravel())

    txt.set_text(f"t = {t[k]/kyr:.2f} kyr")
    return imL, imR, txt

def update(k):
    A = mirror_field(np.clip(nH_hist[k], eps, None), mirror_r=mirror_r, mirror_z=mirror_z)
    T = mirror_field(np.clip(T_hist[k],  eps, None), mirror_r=mirror_r, mirror_z=mirror_z)

    imL.set_array(A.ravel())
    imR.set_array(T.ravel())

    txt.set_text(f"t = {t[k]/kyr:.2f} kyr")
    return imL, imR, txt

# IMPORTANT: blit=False avoids the Matplotlib FuncAnimation _resize_id issue you saw
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)

if save_mp4:
    ani.save(outname, writer="ffmpeg", fps=fps, dpi=dpi)
else:
    import matplotlib.pyplot as plt
    plt.show()
