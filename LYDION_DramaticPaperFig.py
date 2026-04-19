import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator 
from pathlib import Path
from PlotCosmetics import InsetColorBar
import sys

repo_root = Path(__file__).resolve().parent
cmastro_src = repo_root / "src" / "cmastro" / "src"

sys.path.insert(0, str(cmastro_src))

import cmastro

# ---------------- User inputs ----------------
Filename_noLya = "RHDSim_StarCluster_NoLya_64x64_2026-04-18.h5"    # Run ignoring Lyα feedback
Filename_Lya   = "RHDSim_StarCluster_Lya_64x64_2026-04-18.h5"      # Run with Lyα feedback
FigureName     = "LYDION_Figure_LyaVsNoLya.pdf"                      # Or .png

Time1_kyr      = 5.0    # Earlier snapshot time (in kyrs)
Time2_kyr      = 30.0   # Later snapshot time (in kyrs)

fontsize_cbar_label = 26   # the "nH (cm-3)" text
fontsize_cbar_ticks = 22   # the "10^3", "10^4" numbers

# Useful constants
kB  = 1.38e-16
mH  = 1.67e-24
X   = 0.76
yr  = 3.1556926e7
kyr = 1e3 * yr
pc  = 3.085677581e18

###############################################################
#######                                                 #######
#######            P L O T  C O S M E T I C S           #######
#######                                                 #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 39.5})
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

###############################################################
#######                                                 #######
#######                   P L O T   I T !               #######
#######                                                 #######
###############################################################


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def reorder_history_3d(arr, t, r_edges, Z_edges):
    """
    Reorder a 3D history array to shape (Nt, Nr, Nz).

    One axis must match len(t) (time).
    The remaining two must match (Nr, Nz) in some order.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    Nt = t.size
    Nr = len(r_edges) - 1
    Nz = len(Z_edges) - 1

    shape = arr.shape

    time_axes = [ax for ax, s in enumerate(shape) if s == Nt]
    if len(time_axes) != 1:
        raise RuntimeError(
            f"Could not uniquely identify time axis in array with shape {shape} and Nt = {Nt}"
        )
    time_axis = time_axes[0]

    arr_tfirst = np.moveaxis(arr, time_axis, 0)
    _, a1, a2 = arr_tfirst.shape

    if (a1, a2) == (Nr, Nz):
        return arr_tfirst
    elif (a1, a2) == (Nz, Nr):
        return arr_tfirst.transpose(0, 2, 1)
    else:
        raise RuntimeError(
            f"After moving time axis, spatial dims {a1, a2} do not match Nr = {Nr}, Nz = {Nz}"
        )
    
def read_run(filename, load_T=False):
    """
    Read one run and return relevant arrays.
    Set load_T=True to also load T_History (only needed for the Lyα run).
    """
    with h5py.File(filename, "r") as f:
        t        = f["Times"][:]
        r        = f["r"][:]
        Z        = f["Z"][:]
        rho_raw  = f["GasDensity_History"][:]
        T_raw    = f["T_History"][:] if load_T else None

        P_raw    = f["GasPressure_History"][:] if "GasPressure_History" in f else None
        eLya_raw = f["e_Lya_History"][:]       if "e_Lya_History"       in f else None

    rho_hist  = reorder_history_3d(rho_raw, t, r, Z)
    nH_hist   = X * rho_hist / mH
    T_hist    = reorder_history_3d(T_raw, t, r, Z) if T_raw is not None else None

    P_hist    = reorder_history_3d(P_raw,    t, r, Z) if P_raw    is not None else None
    eLya_hist = reorder_history_3d(eLya_raw, t, r, Z) if eLya_raw is not None else None

    saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]

    return dict(
        t=t,
        r=r,
        Z=Z,
        rho_hist=rho_hist,
        nH_hist=nH_hist,
        T_hist=T_hist,
        P_hist=P_hist,
        eLya_hist=eLya_hist,
        saved_idx=saved_idx
    )

def find_best_time_pair(target_time_kyr, run_noLya, run_Lya, mismatch_weight=1.0):
    """
    Find the pair of saved snapshots that best matches a requested time
    while also preferring the no-Lyα and Lyα times to be close to each other.
    """
    target_time = target_time_kyr * kyr

    saved_noLya = run_noLya["saved_idx"]
    saved_Lya   = run_Lya["saved_idx"]

    t_noLya = run_noLya["t"][saved_noLya]
    t_Lya   = run_Lya["t"][saved_Lya]

    T_noLya, T_Lya = np.meshgrid(t_noLya, t_Lya, indexing='ij')

    score = np.abs(T_noLya - target_time) + np.abs(T_Lya - target_time) \
          + mismatch_weight * np.abs(T_noLya - T_Lya)

    i_best, j_best = np.unravel_index(np.argmin(score), score.shape)

    k_noLya = saved_noLya[i_best]
    k_Lya   = saved_Lya[j_best]

    return k_noLya, k_Lya

def fmt_log_tick(val):
    exp = int(np.round(np.log10(val)))
    if exp == -1:
        return r"$0.1$"
    elif exp == 0:
        return r"$1$"
    elif exp == 1:
        return r"$10$"
    else:
        return rf"$10^{{{exp}}}$"

# -------------------------------------------------------------------
# Read both runs
# -------------------------------------------------------------------

run_noLya = read_run(Filename_noLya, load_T=True)
run_Lya   = read_run(Filename_Lya,   load_T=True)

# -------------------------------------------------------------------
# Check that the grids match
# -------------------------------------------------------------------

if (run_noLya["r"].shape != run_Lya["r"].shape) or (run_noLya["Z"].shape != run_Lya["Z"].shape):
    raise RuntimeError(
        f"Grid mismatch: r shapes {run_noLya['r'].shape} vs {run_Lya['r'].shape}, "
        f"Z shapes {run_noLya['Z'].shape} vs {run_Lya['Z'].shape}"
    )

if not (np.allclose(run_noLya["r"], run_Lya["r"]) and np.allclose(run_noLya["Z"], run_Lya["Z"])):
    raise RuntimeError("Grid mismatch: r/Z edge arrays differ between the two runs.")

# -------------------------------------------------------------------
# Find best matching snapshot pairs
# -------------------------------------------------------------------

k1_noLya, k1_Lya = find_best_time_pair(Time1_kyr, run_noLya, run_Lya)
k2_noLya, k2_Lya = find_best_time_pair(Time2_kyr, run_noLya, run_Lya)

print(f"Requested Time1 = {Time1_kyr:.3f} kyr")
print(f"  No-Lyα snapshot = {run_noLya['t'][k1_noLya]/kyr:.3f} kyr")
print(f"  Lyα snapshot    = {run_Lya['t'][k1_Lya]/kyr:.3f} kyr")

print(f"Requested Time2 = {Time2_kyr:.3f} kyr")
print(f"  No-Lyα snapshot = {run_noLya['t'][k2_noLya]/kyr:.3f} kyr")
print(f"  Lyα snapshot    = {run_Lya['t'][k2_Lya]/kyr:.3f} kyr")

# -------------------------------------------------------------------
# Prepare grids for plotting
# -------------------------------------------------------------------

r = run_noLya["r"]
Z = run_noLya["Z"]

R_edges_pc, Z_edges_pc = np.meshgrid(r / pc, Z / pc, indexing='ij')

# -------------------------------------------------------------------
# Choose a common density scale for all four panels
# -------------------------------------------------------------------

nH0_noLya = run_noLya["nH_hist"][0]
nH0_Lya   = run_Lya["nH_hist"][0]

nH0_noLya_pos = nH0_noLya[nH0_noLya > 0.0]
nH0_Lya_pos   = nH0_Lya[nH0_Lya > 0.0]

if (nH0_noLya_pos.size == 0) or (nH0_Lya_pos.size == 0):
    raise ValueError("Initial nH field has no positive values in one of the runs.")

nH0_char = max(np.max(nH0_noLya_pos), np.max(nH0_Lya_pos))

vmin_nH = 10.0**(np.log10(nH0_char) - 2.0 - 0.4)
vmax_nH = 10.0**(np.log10(nH0_char) + 2.0 - 0.4)
norm_nH = LogNorm(vmin=vmin_nH, vmax=vmax_nH)

logmin_nH = int(np.ceil(np.log10(vmin_nH)))
logmax_nH = int(np.floor(np.log10(vmax_nH)))
ticks_nH  = 10.0**np.arange(logmin_nH, logmax_nH + 1)
ticklabels_nH = [fmt_log_tick(v) for v in ticks_nH]

# -------------------------------------------------------------------
# Temperature colour scale (Lyα run only, both time snapshots)
# -------------------------------------------------------------------

T_vmin, T_vmax = 1.0, 1e5      # adjust if needed: e.g. (1e3, 3e4) for HII focus
norm_T = LogNorm(vmin=T_vmin, vmax=T_vmax, clip=True)

ticks_T      = [1.0, 1e2, 1e4]
ticklabels_T = [fmt_log_tick(v) for v in ticks_T]

cmap_T = 'coolwarm'

# -------------------------------------------------------------------
# Choose scales for Lyα-only panels
# -------------------------------------------------------------------

norm_eLya  = LogNorm(vmin=1e-16, vmax=1e-6)
norm_ratio = LogNorm(vmin=0.1,   vmax=10.0)

ticks_ratio      = [0.1, 1.0, 10.0]
ticklabels_ratio = [fmt_log_tick(v) for v in ticks_ratio]

# -------------------------------------------------------------------
# Load snapshots
# -------------------------------------------------------------------

eps = 1e-30

# Density
nH_11 = np.clip(run_noLya["nH_hist"][k1_noLya].T, vmin_nH, vmax_nH)
nH_12 = np.clip(run_Lya["nH_hist"][k1_Lya].T,     vmin_nH, vmax_nH)
nH_21 = np.clip(run_noLya["nH_hist"][k2_noLya].T, vmin_nH, vmax_nH)
nH_22 = np.clip(run_Lya["nH_hist"][k2_Lya].T,     vmin_nH, vmax_nH)

# Temperature at Time2, both runs (for bottom row)
T_b1 = np.clip(run_noLya["T_hist"][k2_noLya].T, eps, None)   # no-Lyα
T_b2 = np.clip(run_Lya["T_hist"][k2_Lya].T,     eps, None)   # Lyα

# e_Lyα at both snapshots (for col 2, top rows)

# -------------------------------------------------------------------
# Lyα-only diagnostics
# -------------------------------------------------------------------

if run_Lya["P_hist"] is None:
    raise RuntimeError("GasPressure_History not found in Lyα file.")

if run_Lya["eLya_hist"] is None:
    raise RuntimeError("e_Lya_History not found in Lyα file.")

eLya_1  = np.clip(run_Lya["eLya_hist"][k1_Lya].T, eps, None)   # col 2, row 0

Pgas_2  = run_Lya["P_hist"][k2_Lya].T
eLya_2_raw = run_Lya["eLya_hist"][k2_Lya].T

PLya_2  = eLya_2_raw / 3.0
Ratio_2 = np.clip(PLya_2 / np.maximum(Pgas_2, eps), eps, None)
eLya_2  = np.clip(eLya_2_raw, eps, None)                        # col 2, row 1

# -------------------------------------------------------------------
# Set up figure
#
# Layout:
#   Top block  (2 rows × 3 cols):
#     Row 0: no-Lyα nH t1  | Lyα nH t1  | Lyα e_Lyα t1
#     Row 1: no-Lyα nH t2  | Lyα nH t2  | Lyα e_Lyα t2
#   Bottom block (1 row × 3 cols):
#     T no-Lyα t2          | T Lyα t2   | P_Lyα/P t2
# -------------------------------------------------------------------

wgap_big   = 0.02          # horizontal gap between no-Lyα and Lyα columns
wgap_small = 0.000         # horizontal gap between Lyα density and e_Lyα columns
vgap       = 0.001         # vertical gap between top and bottom blocks
hgap       = 0.001         # vertical gap between rows within top block

wpanel = (1.0 - wgap_big - wgap_small) / 3.0
hpanel = (1.0 - vgap - hgap) / 3.0

# Column left edges (shared by top and bottom blocks)
col0_left = 0.0
col1_left = wpanel + wgap_big
col2_left = col1_left + wpanel + wgap_small

top_bottom = vgap + hpanel
top_height = 2 * hpanel + hgap

def make_ax(fig, col_left, row_bottom, width, height, sharex=None, sharey=None):
    ax = fig.add_axes([col_left, row_bottom, width, height],
                      sharex=sharex, sharey=sharey)
    return ax

# Bottom block spans all 3 cols (same left edges as top block)
bot_bottom = 0.0
bot_height = hpanel

fig = plt.figure(figsize=(24, 24))

row0_bottom = top_bottom + hpanel + hgap
row1_bottom = top_bottom

ax11 = make_ax(fig, col0_left, row0_bottom, wpanel, hpanel)                              # no-Lyα nH t1
ax12 = make_ax(fig, col1_left, row0_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)    # Lyα nH t1
ax13 = make_ax(fig, col2_left, row0_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)    # Lyα e_Lyα t1

ax21 = make_ax(fig, col0_left, row1_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)    # no-Lyα nH t2
ax22 = make_ax(fig, col1_left, row1_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)    # Lyα nH t2
ax23 = make_ax(fig, col2_left, row1_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)    # Lyα e_Lyα t2

ax31 = make_ax(fig, col0_left, bot_bottom,  wpanel, bot_height, sharex=ax11, sharey=ax11)  # T no-Lyα t2
ax32 = make_ax(fig, col1_left, bot_bottom,  wpanel, bot_height, sharex=ax11, sharey=ax11)  # T Lyα t2
ax33 = make_ax(fig, col2_left, bot_bottom,  wpanel, bot_height, sharex=ax11, sharey=ax11)  # P_Lyα/P t2

# -------------------------------------------------------------------
# Plot density panels (rows 0–1, cols 0–1)
# -------------------------------------------------------------------

cmap_nH = 'cma:laguna'

im11 = ax11.pcolormesh(R_edges_pc, Z_edges_pc, nH_11, norm=norm_nH, shading='auto',
                       cmap=cmap_nH, rasterized=True)
im12 = ax12.pcolormesh(R_edges_pc, Z_edges_pc, nH_12, norm=norm_nH, shading='auto',
                       cmap=cmap_nH, rasterized=True)
im21 = ax21.pcolormesh(R_edges_pc, Z_edges_pc, nH_21, norm=norm_nH, shading='auto',
                       cmap=cmap_nH, rasterized=True)
im22 = ax22.pcolormesh(R_edges_pc, Z_edges_pc, nH_22, norm=norm_nH, shading='auto',
                       cmap=cmap_nH, rasterized=True)

# -------------------------------------------------------------------
# Plot e_Lyα panels (rows 0–1, col 2)
# -------------------------------------------------------------------

im13 = ax13.pcolormesh(R_edges_pc, Z_edges_pc, eLya_1, norm=norm_eLya,
                       shading='auto', cmap='magma', rasterized=True)
im23 = ax23.pcolormesh(R_edges_pc, Z_edges_pc, eLya_2, norm=norm_eLya,
                       shading='auto', cmap='magma', rasterized=True)

# -------------------------------------------------------------------
# Plot bottom row: T (both runs) | P_Lyα/P
# -------------------------------------------------------------------

im31 = ax31.pcolormesh(R_edges_pc, Z_edges_pc, T_b1, norm=norm_T,
                       shading='auto', cmap=cmap_T, rasterized=True)
im32 = ax32.pcolormesh(R_edges_pc, Z_edges_pc, T_b2, norm=norm_T,
                       shading='auto', cmap=cmap_T, rasterized=True)
im33 = ax33.pcolormesh(R_edges_pc, Z_edges_pc, Ratio_2, norm=norm_ratio,
                       shading='auto', cmap='cma:unph', rasterized=True)

# Contours for P_Lyα / P on ax33
#contour_levels = [1.5, 2.0]

#cs33 = ax33.contour(
#    0.5 * (R_edges_pc[:-1, :-1] + R_edges_pc[1:, 1:]),
#    0.5 * (Z_edges_pc[:-1, :-1] + Z_edges_pc[1:, 1:]),
#    Ratio_2,
#    levels=contour_levels,
#    colors='white',
#    linewidths=1.8
#)

#ax33.clabel(
#    cs33,
#    fmt={1.5: "1.5", 2.0: "2"},
#    inline=True,
#    fontsize=14,
#    colors='white'
#)

# -------------------------------------------------------------------
# Make all panels cinematic (no ticks, spines visible)
# -------------------------------------------------------------------

for ax in (ax11, ax12, ax13, ax21, ax22, ax23, ax31, ax32, ax33):
    ax.set_xlim(r[0] / pc, r[-1] / pc)
    ax.set_ylim(Z[0] / pc, Z[-1] / pc)
    ax.set_aspect('equal', adjustable='box')

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(2.0)

# -------------------------------------------------------------------
# Column titles (top row only)
# -------------------------------------------------------------------

title_style = dict(
    ha='center',
    va='bottom',
    color='black',
    fontsize=plt.rcParams['font.size'],
    path_effects=[pe.withStroke(linewidth=3, foreground='white')]
)

# No-Lyα title: centred over col 0
ax11.text(0.5, 1.02, "Ignoring Ly\u03b1 Radiation Pressure", transform=ax11.transAxes, **title_style)

# Lyα title: centred over cols 1–2 in figure coordinates
lya_title_x = 0.5 * (col1_left + col2_left + wpanel)
lya_title_y = top_bottom + top_height
fig.text(lya_title_x, lya_title_y + 0.005, "With Ly\u03b1 Radiation Pressure", **title_style)

# -------------------------------------------------------------------
# Time labels (inside panels, white text)
# -------------------------------------------------------------------

time_style = dict(
    ha='left',
    va='top',
    color='white',
    path_effects=[pe.withStroke(linewidth=3, foreground='black')]
)

fontsize_time = 35

ax11.text(0.04, 0.96, f"t = {run_noLya['t'][k1_noLya]/kyr:.2f} kyr",
          transform=ax11.transAxes, **time_style, fontsize=fontsize_time)
ax12.text(0.04, 0.96, f"t = {run_Lya['t'][k1_Lya]/kyr:.2f} kyr",
          transform=ax12.transAxes, **time_style, fontsize=fontsize_time)
ax13.text(0.04, 0.96, f"t = {run_Lya['t'][k1_Lya]/kyr:.2f} kyr",
          transform=ax13.transAxes, **time_style, fontsize=fontsize_time)
ax21.text(0.04, 0.96, f"t = {run_noLya['t'][k2_noLya]/kyr:.2f} kyr",
          transform=ax21.transAxes, **time_style, fontsize=fontsize_time)
ax22.text(0.04, 0.96, f"t = {run_Lya['t'][k2_Lya]/kyr:.2f} kyr",
          transform=ax22.transAxes, **time_style, fontsize=fontsize_time)
ax23.text(0.04, 0.96, f"t = {run_Lya['t'][k2_Lya]/kyr:.2f} kyr",
          transform=ax23.transAxes, **time_style, fontsize=fontsize_time)
ax31.text(0.04, 0.96, f"t = {run_noLya['t'][k2_noLya]/kyr:.2f} kyr",
          transform=ax31.transAxes, **time_style, fontsize=fontsize_time)
ax32.text(0.04, 0.96, f"t = {run_Lya['t'][k2_Lya]/kyr:.2f} kyr",
          transform=ax32.transAxes, **time_style, fontsize=fontsize_time)
ax33.text(0.04, 0.96, f"t = {run_Lya['t'][k2_Lya]/kyr:.2f} kyr",
          transform=ax33.transAxes, **time_style, fontsize=fontsize_time)

# -------------------------------------------------------------------
# Colorbars (all inset, top-right of their host panel)
# -------------------------------------------------------------------

def make_inset_cbar(fig, ax, im, rect, label, ticks, ticklabels, dark=False):
    """
    Shared helper for all inset horizontal colorbars.
    dark=True  → black text/outline on white background (for bright colormaps like coolwarm)
    dark=False → white text/outline on black background (default, for dark colormaps)
    """
    fg = 'black' if dark else 'white'
    bg = 'white' if dark else 'black'

    cax  = ax.inset_axes(rect)
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.outline.set_edgecolor(fg)
    cbar.outline.set_linewidth(1.2)
    cbar.ax.tick_params(axis="x", which='both', colors=fg, labelsize=fontsize_cbar_ticks,
                        length=0, width=0, pad=5,
                        bottom=False, top=False)
    cbar.ax.minorticks_off()   # suppress log-scale minor ticks that ignore tick_params
    cbar.set_label(label, color=fg, fontsize=fontsize_cbar_label, labelpad=4)
    cbar.ax.set_facecolor(bg)
    return cbar

# Density colorbar — row 0, col 0
make_inset_cbar(fig, ax11, im11,
                rect=[0.60, 0.92, 0.34, 0.03],
                label=r"n$_{H}$ (cm$^{-3}$)",
                ticks=ticks_nH, ticklabels=ticklabels_nH)

# e_Lyα colorbar — row 0, col 2
ticks_eLya      = [1e-14, 1e-10, 1e-6]
ticklabels_eLya = [fmt_log_tick(v) for v in ticks_eLya]
make_inset_cbar(fig, ax13, im13,
                rect=[0.58, 0.92, 0.34, 0.03],
                label=r"$e_{Ly\alpha}$ (erg cm$^{-3}$)",
                ticks=ticks_eLya, ticklabels=ticklabels_eLya)

# Temperature colorbar — bottom row, col 0 (dark: coolwarm is bright)
make_inset_cbar(fig, ax31, im31,
                rect=[0.60, 0.92, 0.34, 0.03],
                label=r"T (K)",
                ticks=ticks_T, ticklabels=ticklabels_T,
                dark=True)

# P_Lyα/P colorbar — bottom row, col 2
make_inset_cbar(fig, ax33, im33,
                rect=[0.58, 0.92, 0.34, 0.03],
                label=r"P$_{Ly\alpha}$/P",
                ticks=ticks_ratio, ticklabels=ticklabels_ratio)

# -------------------------------------------------------------------
# Save figure
# -------------------------------------------------------------------

plt.savefig(FigureName, bbox_inches='tight')
plt.show()