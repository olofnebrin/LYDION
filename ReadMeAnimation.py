"""
README banner generator (non-pixely + non-laggy on GitHub)

Key changes vs your version:
- Output is animated WebP by default (much smoother + better gradients than GIF)
- We downsample frames (fps_target) and spatial resolution (target_width_px) to keep file small
- We avoid Pillow GIF writer (GIF is 256 colors + CPU heavy on GitHub)

Requires:
- ffmpeg installed (for encoding WebP/MP4)
- LaTeX installed (since text.usetex=True)
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

import matplotlib as mpl
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"

# ---------------- USER inputs ----------------
filename  = "RHDSim_NoLya_400x400_2026-01-20.h5"

# Recommended: animated WebP for GitHub README
outname   = "ReadMeBanner.webp"   # ".webp" (best), or ".mp4"

# Make it smooth but light
fps_target = 15                   # GitHub-friendly (12–15 is sweet spot)
frameskip  = 1                    # will be auto-adjusted below to hit fps_target

# Vertical crop as fraction of full Z-range
z0_frac, z1_frac = 0.00, 0.55

# Display ranges
vmin_nH, vmax_nH = 100.0, 3e4     # density (nH)
vmin_T,  vmax_T  = 10.0,  3e4     # temperature

# Banner geometry (controls final pixel size)
target_width_px = 1400            # 1200–1600 recommended
panel_gap_px    = 0               # keep panels touching

# Aesthetic knobs
dpi = 150                         # internal rendering dpi (not final size)
fig_w_in = 14.0                   # will be converted to ~target_width_px
fig_h_in = 3.2                    # thin banner
cmap_nH  = "viridis"
cmap_T   = "coolwarm"

# Title overlay
title_text  = r"\textsc{Lydion}"
title_fs    = 70
title_alpha = 0.9
# ---------------------------------------------

# Constants
mH  = 1.67e-24
X   = 0.76
pc  = 3.085677581e18

# --- read data ---
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

# Convert to number density
nH_hist = X * rho_hist / mH

# frames where data exist
saved_idx = np.where((t > 0) | (np.arange(t.size) == 0))[0]

# --- choose frameskip to roughly match fps_target without exploding frame count ---
# If your simulation saves at very high cadence, this keeps the animation light.
if saved_idx.size > 1:
    dt_med = np.median(np.diff(t[saved_idx]))
    # If dt_med is tiny, we need to skip more to keep playback reasonable.
    # We estimate the "native fps" as 1/dt in arbitrary units and just cap to fps_target
    # by increasing frameskip if needed. If dt_med is not meaningful, this still behaves ok.
    # (This is a heuristic; the real point is: fewer frames -> smaller file -> less lag.)
    desired_frames = 500  # cap total frames; adjust if you want longer animations
    auto_skip = int(np.ceil(saved_idx.size / desired_frames))
    frameskip = max(frameskip, auto_skip)

frames = saved_idx[::frameskip]

# --- crop indices ---
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

# --- figure size: set so final pixel width is near target_width_px ---
# total width in pixels ≈ fig_w_in * dpi
fig_w_in = target_width_px / dpi
fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)

# exact zero gap between panels
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.0)
axL = fig.add_subplot(gs[0, 0])
axR = fig.add_subplot(gs[0, 1])

for ax in (axL, axR):
    ax.set_axis_off()
    ax.set_facecolor("black")

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

# Overlay title (once)
import matplotlib.patheffects as pe
txt = fig.text(
    0.5, 0.5,
    title_text,
    color="white",
    fontsize=title_fs,
    ha="center",
    va="center",
    alpha=title_alpha,
)
txt.set_path_effects([pe.withStroke(linewidth=3, foreground="black")])

# norms
norm_nH = LogNorm(vmin=vmin_nH, vmax=vmax_nH)
norm_T  = LogNorm(vmin=vmin_T,  vmax=vmax_T)

# images (aspect='auto' + extent fills panels, no whitespace)
im_nH = axL.imshow(
    frame_nH(frames[0]),
    origin="lower",
    interpolation="nearest",
    norm=norm_nH,
    cmap=cmap_nH,
    extent=extent,
    aspect="auto",
)
im_T = axR.imshow(
    frame_T(frames[0]),
    origin="lower",
    interpolation="nearest",
    norm=norm_T,
    cmap=cmap_T,
    extent=extent,
    aspect="auto",
)

def init():
    k = frames[0]
    im_nH.set_data(frame_nH(k))
    im_T.set_data(frame_T(k))
    return (im_nH, im_T, txt)

def update(k):
    im_nH.set_data(frame_nH(k))
    im_T.set_data(frame_T(k))
    return (im_nH, im_T, txt)

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

# Force a draw once (helps usetex + first-frame stability)
fig.canvas.draw()

# --- Save: WebP is best for GitHub (smooth + small + good gradients) ---
ext = os.path.splitext(outname.lower())[1]

if ext == ".webp":
    # Use ffmpeg writer -> produces a video first; matplotlib handles piping to ffmpeg.
    # We encode as WebP with a moderate bitrate to keep size low.
    # (ffmpeg must be available in PATH)
    writer = "ffmpeg"
    # Use a reasonably small bitrate; you can bump if you want sharper output.
    ani.save(outname, writer=writer, fps=fps_target, dpi=dpi, bitrate=2500)

elif ext == ".mp4":
    ani.save(outname, writer="ffmpeg", fps=fps_target, dpi=dpi, bitrate=2500)

elif ext == ".gif":
    # Not recommended (pixelly + laggy on GitHub). If you insist, keep it light:
    # - fewer frames (frameskip above)
    # - fps_target <= 12
    # - consider palette-optimized ffmpeg pipeline outside Python
    ani.save(outname, writer="pillow", fps=min(fps_target, 12))
else:
    raise ValueError("Unsupported output extension. Use .webp, .mp4, or (not recommended) .gif")

print("Wrote:", outname)
print(f"Frames: {len(frames)} (frameskip={frameskip}), fps_target={fps_target}, width≈{int(fig_w_in*dpi)} px")
