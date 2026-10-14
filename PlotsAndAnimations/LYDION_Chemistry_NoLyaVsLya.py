import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
from pathlib import Path
import sys
import os

repo_root = Path(__file__).resolve().parents[1]
os.chdir(repo_root)
cmastro_src = repo_root / "src" / "cmastro" / "src"

sys.path.insert(0, str(cmastro_src))

import cmastro


# ---------------- User inputs ----------------

Filename_noLya = "RHDSim_StarCluster_NoLya_512x512_2026-04-19.h5"    # Run ignoring Lyα feedback
Filename_Lya   = "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5"      # Run with Lyα feedback

FigureName     = "LYDION_Figure_LyaVsNoLya_HAbundances.pdf"          # Or .png


# Fill in between 1 and 4 snapshot times (in kyr).
# Examples:
# SnapshotTimes_kyr = [20]
# SnapshotTimes_kyr = [5, 40]
# SnapshotTimes_kyr = [5, 20, 40]
# SnapshotTimes_kyr = [5, 20, 30, 40]

SnapshotTimes_kyr = [20]


# Figure size scales automatically with the number of rows.
# Four columns now, so keep roughly ~8 inches per column and row.

FigureColWidth_inches   = 8.0
FigureRowHeight_inches  = 8.0


# Layout options
wgap_big   = 0.018    # gap between no-Lyα group and Lyα group
wgap_small = 0.000    # gap within a group (xHI | xH2)
hgap       = -0.01

xmax_fraction = 0.8
zmax_fraction = 0.8
rasterized    = True


# Text sizes
fontsize_cbar_ticks  = 26
fontsize_panel_label = 60
fontsize_time        = 50
fontsize_group_title = 39.5


# -------------------------------------------------------------------
# Colormaps / abundance colorbar style
# Following the GasAbundances script.
# -------------------------------------------------------------------

cmap_abundance = 'BuPu_r'

# Per-species colormap overrides. Set to None to use cmap_abundance.
cmap_xHI = 'YlGnBu_r'
cmap_xH2 = None


# -------------------------------------------------------------------
# Abundance ranges
# -------------------------------------------------------------------

AutoScale_Abundances = False

# Shared abundance range (can be overridden per species below)
abund_vmin, abund_vmax = 1e-6, 1.0

# Per-species range overrides. Set to None to use shared values.
xHI_vmin, xHI_vmax = None, None
xH2_vmin, xH2_vmax = None, None

# Percentiles if AutoScale_Abundances = True
pct_low_log  = 1.0
pct_high_log = 99.0


# -------------------------------------------------------------------
# Colorbar options
# -------------------------------------------------------------------

# Per-quantity colorbar geometry: inset vertical bars in the top row.
cbar_x      = 0.80
cbar_y      = 0.50
cbar_width  = 0.040
cbar_height = 0.45

cbar_tick_length = 0
cbar_tick_width  = 0

# Shared abundance colorbar colors
cbar_tick_color_abund    = "black"
cbar_outline_color_abund = "black"

# Per-species overrides (as in the abundance script)
cbar_tick_color_xHI, cbar_outline_color_xHI = None, None
cbar_tick_color_xH2, cbar_outline_color_xH2 = "white", "white"

# Shared abundance ticks (used if per-species overrides are None)
ticks_abund_custom      = None
ticklabels_abund_custom = None

ticks_xHI_custom, ticklabels_xHI_custom = None, None
ticks_xH2_custom, ticklabels_xH2_custom = None, None


# -------------------------------------------------------------------
# Time label style
# -------------------------------------------------------------------

time_label_color        = "white"
time_label_stroke       = "black"
time_label_stroke_width = 4.0


# -------------------------------------------------------------------
# Panel label style
# -------------------------------------------------------------------

panel_label_default_x            = 0.94
panel_label_default_y            = 0.08
panel_label_default_ha           = 'right'
panel_label_default_va           = 'bottom'
panel_label_default_color        = 'white'
panel_label_default_stroke       = 'black'
panel_label_default_stroke_width = 3.5

# xHI tends to be light in BuPu_r -> black text with white stroke often looks better.
panel_label_overrides = {
    "xHI": dict(color='black', stroke='white'),
    "xH2": dict(color='white', stroke='black'),
}


# -------------------------------------------------------------------
# Scale bar
# -------------------------------------------------------------------

ShowScaleBar = True

# Axis names follow ax[row][col] -> ax11, ax12, ..., axN4
# Default here is top-row Lyα xHI panel.
scale_bar_ax_name = "ax13"

scale_bar_length_pc = 0.2
scale_bar_label     = "0.2 pc"

scale_bar_x_axes = 0.74*0.7
scale_bar_y_axes = 0.08

scale_bar_color = "black"
scale_bar_lw = 3.0
scale_bar_capsize_axes = 0.045
scale_bar_label_pad_axes = 0.020
scale_bar_fontsize = 34
scale_bar_stroke = "black"
scale_bar_stroke_width = 0.0


# -------------------------------------------------------------------
# Useful constants
# -------------------------------------------------------------------

kB   = 1.38e-16
mH   = 1.67e-24
X    = 0.76
yr   = 3.1556926e7
kyr  = 1e3 * yr
pc   = 3.085677581e18
c    = 2.99792458e10
Lsun = 3.839e33


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

plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def reorder_history_3d(arr, t, r_edges, Z_edges):
    """
    Reorder a 3D history array to shape (Nt, Nr, Nz).
    """
    if arr is None:
        return None

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
            f"After moving time axis, spatial dims {(a1, a2)} do not match Nr = {Nr}, Nz = {Nz}"
        )


def read_dataset_if_exists(file, name, default=None):
    return file[name][:] if name in file else default


def positive_values(arr):
    vals = np.asarray(arr)
    return vals[np.isfinite(vals) & (vals > 0.0)]


def make_lognorm_from_arrays(arrays, default_vmin, default_vmax,
                             pct_low=1.0, pct_high=99.0):
    vals_list = [positive_values(a) for a in arrays if a is not None]
    if len(vals_list) == 0:
        return LogNorm(vmin=default_vmin, vmax=default_vmax)

    vals = np.concatenate(vals_list)
    if vals.size == 0:
        return LogNorm(vmin=default_vmin, vmax=default_vmax)

    vmin = np.percentile(vals, pct_low)
    vmax = np.percentile(vals, pct_high)
    vmin = max(vmin, np.min(vals))
    vmax = min(vmax, np.max(vals))

    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin <= 0.0) or (vmax <= vmin):
        return LogNorm(vmin=default_vmin, vmax=default_vmax)

    return LogNorm(vmin=vmin, vmax=vmax)


def resolve_vmin_vmax(species_vmin, species_vmax):
    vmin = species_vmin if species_vmin is not None else abund_vmin
    vmax = species_vmax if species_vmax is not None else abund_vmax
    return vmin, vmax


def resolve_cmap(species_cmap):
    return species_cmap if species_cmap is not None else cmap_abundance


def resolve_cbar_colors(species_tick_color, species_outline_color):
    tc = species_tick_color    if species_tick_color    is not None else cbar_tick_color_abund
    oc = species_outline_color if species_outline_color is not None else cbar_outline_color_abund
    return tc, oc


def resolve_ticks(species_ticks, species_ticklabels):
    ticks  = species_ticks      if species_ticks      is not None else ticks_abund_custom
    labels = species_ticklabels if species_ticklabels is not None else ticklabels_abund_custom
    return ticks, labels


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


def get_log_ticks_from_norm(norm):
    logmin = int(np.ceil(np.log10(norm.vmin)))
    logmax = int(np.floor(np.log10(norm.vmax)))
    if logmax < logmin:
        return None, None
    ticks = 10.0 ** np.arange(logmin, logmax + 1)
    ticklabels = [fmt_log_tick(v) for v in ticks]
    return ticks, ticklabels


def use_custom_ticks(default_ticks, default_ticklabels, custom_ticks, custom_ticklabels):
    ticks = default_ticks if custom_ticks is None else custom_ticks
    if custom_ticklabels is None:
        ticklabels = default_ticklabels
        if custom_ticks is not None:
            ticklabels = None
    else:
        ticklabels = custom_ticklabels
    return ticks, ticklabels


def add_inset_colorbar(ax, mappable, ticks=None, ticklabels=None,
                       tick_color=None, outline_color=None):
    cax = ax.inset_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    cb = plt.colorbar(mappable, cax=cax, orientation='vertical')

    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)

    cb.ax.tick_params(
        labelsize=fontsize_cbar_ticks,
        width=cbar_tick_width,
        length=cbar_tick_length,
        pad=4,
        colors=tick_color,
        labelcolor=tick_color
    )

    cb.ax.minorticks_off()
    if ticks is not None:
        from matplotlib.ticker import FixedLocator, FixedFormatter
        cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
        if ticklabels is not None:
            cb.ax.yaxis.set_major_formatter(FixedFormatter(ticklabels))

    cb.outline.set_linewidth(1.5)
    cb.outline.set_edgecolor(outline_color)

    for spine in cb.ax.spines.values():
        spine.set_edgecolor(outline_color)
        spine.set_linewidth(1.5)

    return cb


def make_ax(fig, col_left, row_bottom, width, height, sharex=None, sharey=None):
    return fig.add_axes([col_left, row_bottom, width, height], sharex=sharex, sharey=sharey)


def panel_label(ax, text, species_key):
    ov = panel_label_overrides.get(species_key, {})
    x            = ov.get('x',            panel_label_default_x)
    y            = ov.get('y',            panel_label_default_y)
    ha           = ov.get('ha',           panel_label_default_ha)
    va           = ov.get('va',           panel_label_default_va)
    color        = ov.get('color',        panel_label_default_color)
    stroke       = ov.get('stroke',       panel_label_default_stroke)
    stroke_width = ov.get('stroke_width', panel_label_default_stroke_width)

    style = dict(
        ha=ha, va=va,
        color=color,
        fontsize=fontsize_panel_label,
        path_effects=[pe.withStroke(linewidth=stroke_width, foreground=stroke)]
    )
    ax.text(x, y, text, transform=ax.transAxes, **style)


def time_label(ax, time_kyr):
    style = dict(
        ha='left',
        va='top',
        color=time_label_color,
        fontsize=fontsize_time,
        path_effects=[pe.withStroke(linewidth=time_label_stroke_width, foreground=time_label_stroke)]
    )
    ax.text(0.04, 0.96, rf"t = {time_kyr:.2f} kyr", transform=ax.transAxes, **style)


def draw_scale_bar(ax):
    """Draw a simple |---| scale bar in axes coordinates, with length in pc."""

    if not ShowScaleBar:
        return

    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    xspan = xhi - xlo
    yspan = yhi - ylo

    xcen = xlo + scale_bar_x_axes * xspan
    ycen = ylo + scale_bar_y_axes * yspan

    x0 = xcen - 0.5 * scale_bar_length_pc
    x1 = xcen + 0.5 * scale_bar_length_pc

    capsize = scale_bar_capsize_axes * yspan
    label_pad = scale_bar_label_pad_axes * yspan

    path_effects = []
    if scale_bar_stroke is not None:
        path_effects = [
            pe.withStroke(linewidth=scale_bar_stroke_width, foreground=scale_bar_stroke)
        ]

    ax.plot(
        [x0, x1], [ycen, ycen],
        color=scale_bar_color,
        lw=scale_bar_lw,
        solid_capstyle="butt",
        transform=ax.transData,
        zorder=20,
        path_effects=path_effects,
    )

    for xc in (x0, x1):
        ax.plot(
            [xc, xc],
            [ycen - 0.5 * capsize, ycen + 0.5 * capsize],
            color=scale_bar_color,
            lw=scale_bar_lw,
            solid_capstyle="round",
            transform=ax.transData,
            zorder=20,
            path_effects=path_effects,
        )

    ax.text(
        xcen,
        ycen + 0.5 * capsize + label_pad,
        scale_bar_label,
        color=scale_bar_color,
        fontsize=scale_bar_fontsize,
        ha="center",
        va="bottom",
        transform=ax.transData,
        zorder=21,
        path_effects=path_effects,
    )


def find_best_time_pair(target_time_kyr, run_noLya, run_Lya, mismatch_weight=0.0):
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

    score = (
        np.abs(T_noLya - target_time)
        + np.abs(T_Lya - target_time)
        + mismatch_weight * np.abs(T_noLya - T_Lya)
    )

    i_best, j_best = np.unravel_index(np.argmin(score), score.shape)

    k_noLya = saved_noLya[i_best]
    k_Lya   = saved_Lya[j_best]

    return k_noLya, k_Lya


def read_run(filename):
    with h5py.File(filename, "r") as f:
        t = f["Times"][:]
        r = f["r"][:]
        Z = f["Z"][:]

        xHII_raw = read_dataset_if_exists(f, "xHII_History")
        xH2_raw  = read_dataset_if_exists(f, "xH2_History")

    xHII_hist = reorder_history_3d(xHII_raw, t, r, Z)
    xH2_hist  = reorder_history_3d(xH2_raw,  t, r, Z)

    saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]

    return dict(
        t=t,
        r=r,
        Z=Z,
        xHII_hist=xHII_hist,
        xH2_hist=xH2_hist,
        saved_idx=saved_idx,
    )


# -------------------------------------------------------------------
# Check requested row count
# -------------------------------------------------------------------

Nrows = len(SnapshotTimes_kyr)
if not (1 <= Nrows <= 4):
    raise ValueError(
        "SnapshotTimes_kyr must contain between 1 and 4 snapshot times. "
        f"Got {Nrows}: {SnapshotTimes_kyr}"
    )


# -------------------------------------------------------------------
# Read both runs
# -------------------------------------------------------------------

run_noLya = read_run(Filename_noLya)
run_Lya   = read_run(Filename_Lya)


# -------------------------------------------------------------------
# Check that the grids match
# -------------------------------------------------------------------

if (
    run_noLya["r"].shape != run_Lya["r"].shape
    or run_noLya["Z"].shape != run_Lya["Z"].shape
):
    raise RuntimeError(
        f"Grid mismatch: r shapes {run_noLya['r'].shape} vs {run_Lya['r'].shape}, "
        f"Z shapes {run_noLya['Z'].shape} vs {run_Lya['Z'].shape}"
    )

if not (
    np.allclose(run_noLya["r"], run_Lya["r"])
    and np.allclose(run_noLya["Z"], run_Lya["Z"])
):
    raise RuntimeError("Grid mismatch: r/Z edge arrays differ between the two runs.")


# -------------------------------------------------------------------
# Find best matching snapshot pairs
# -------------------------------------------------------------------

snapshot_pairs = []

for target_time_kyr in SnapshotTimes_kyr:
    k_noLya, k_Lya = find_best_time_pair(target_time_kyr, run_noLya, run_Lya)
    snapshot_pairs.append((k_noLya, k_Lya))

    print(f"Requested time = {target_time_kyr:.3f} kyr")
    print(f"  No-Lyα snapshot = {run_noLya['t'][k_noLya] / kyr:.3f} kyr")
    print(f"  Lyα snapshot    = {run_Lya['t'][k_Lya] / kyr:.3f} kyr")


# -------------------------------------------------------------------
# Prepare fields and plotting grids
# -------------------------------------------------------------------

r = run_noLya["r"]
Z = run_noLya["Z"]

R_edges_pc, Z_edges_pc = np.meshgrid(r / pc, Z / pc, indexing='ij')


# -------------------------------------------------------------------
# Build all requested xHI / xH2 snapshots, then choose common norms
# xHI is derived from xHII and xH2:
#   xHI = 1 - xHII - 2*xH2
# -------------------------------------------------------------------

row_data = []

all_xHI_fields = []
all_xH2_fields = []

for k_noLya, k_Lya in snapshot_pairs:
    xHII_no = run_noLya["xHII_hist"][k_noLya]
    xH2_no  = run_noLya["xH2_hist"][k_noLya]

    xHII_ly = run_Lya["xHII_hist"][k_Lya]
    xH2_ly  = run_Lya["xH2_hist"][k_Lya]

    xHI_no = np.clip(1.0 - xHII_no - 2.0 * xH2_no, 0.0, 1.0)
    xHI_ly = np.clip(1.0 - xHII_ly - 2.0 * xH2_ly, 0.0, 1.0)

    row_data.append(
        dict(
            xHI_no=xHI_no,
            xH2_no=xH2_no,
            xHI_ly=xHI_ly,
            xH2_ly=xH2_ly,
        )
    )

    all_xHI_fields.extend([xHI_no, xHI_ly])
    all_xH2_fields.extend([xH2_no, xH2_ly])


# -------------------------------------------------------------------
# Norms
# -------------------------------------------------------------------

xHI_vmin_r, xHI_vmax_r = resolve_vmin_vmax(xHI_vmin, xHI_vmax)
xH2_vmin_r, xH2_vmax_r = resolve_vmin_vmax(xH2_vmin, xH2_vmax)

if AutoScale_Abundances:
    norm_xHI = make_lognorm_from_arrays(all_xHI_fields, xHI_vmin_r, xHI_vmax_r,
                                        pct_low_log, pct_high_log)
    norm_xH2 = make_lognorm_from_arrays(all_xH2_fields, xH2_vmin_r, xH2_vmax_r,
                                        pct_low_log, pct_high_log)
else:
    norm_xHI = LogNorm(vmin=xHI_vmin_r, vmax=xHI_vmax_r)
    norm_xH2 = LogNorm(vmin=xH2_vmin_r, vmax=xH2_vmax_r)


# -------------------------------------------------------------------
# Dynamic figure layout: Nrows × 4 columns
#
# columns:
#   0: no-Lyα xHI
#   1: no-Lyα xH2
#   2: Lyα    xHI
#   3: Lyα    xH2
# -------------------------------------------------------------------

FigureWidth_inches  = FigureColWidth_inches * 4.0
FigureHeight_inches = FigureRowHeight_inches * Nrows

fig = plt.figure(figsize=(FigureWidth_inches, FigureHeight_inches))

wpanel = (1.0 - wgap_big - 2.0 * wgap_small) / 4.0
hpanel = (1.0 - (Nrows - 1) * hgap) / Nrows

col_lefts = [
    0.0,
    wpanel + wgap_small,
    2.0 * wpanel + wgap_small + wgap_big,
    3.0 * wpanel + 2.0 * wgap_small + wgap_big,
]

row_bottoms = [
    (Nrows - 1 - row) * (hpanel + hgap)
    for row in range(Nrows)
]

fig_top = row_bottoms[0] + hpanel


# axes[row][col]
axes = []

for row in range(Nrows):
    row_axes = []
    for col in range(4):
        if row == 0 and col == 0:
            ax = make_ax(fig, col_lefts[col], row_bottoms[row], wpanel, hpanel)
        else:
            ax = make_ax(fig, col_lefts[col], row_bottoms[row], wpanel, hpanel,
                         sharex=axes[0][0] if len(axes) > 0 else row_axes[0],
                         sharey=axes[0][0] if len(axes) > 0 else row_axes[0])
        row_axes.append(ax)
    axes.append(row_axes)


# -------------------------------------------------------------------
# Plot panels
# -------------------------------------------------------------------

images = []

for row, ((k_noLya, k_Lya), row_fields) in enumerate(zip(snapshot_pairs, row_data)):
    ax11, ax12, ax13, ax14 = axes[row]

    xHI_no_plot = np.clip(row_fields["xHI_no"].T, norm_xHI.vmin, norm_xHI.vmax)
    xH2_no_plot = np.clip(row_fields["xH2_no"].T, norm_xH2.vmin, norm_xH2.vmax)
    xHI_ly_plot = np.clip(row_fields["xHI_ly"].T, norm_xHI.vmin, norm_xHI.vmax)
    xH2_ly_plot = np.clip(row_fields["xH2_ly"].T, norm_xH2.vmin, norm_xH2.vmax)

    im11 = ax11.pcolormesh(R_edges_pc, Z_edges_pc, xHI_no_plot,
                           norm=norm_xHI, shading='auto',
                           cmap=resolve_cmap(cmap_xHI), rasterized=rasterized)

    im12 = ax12.pcolormesh(R_edges_pc, Z_edges_pc, xH2_no_plot,
                           norm=norm_xH2, shading='auto',
                           cmap=resolve_cmap(cmap_xH2), rasterized=rasterized)

    im13 = ax13.pcolormesh(R_edges_pc, Z_edges_pc, xHI_ly_plot,
                           norm=norm_xHI, shading='auto',
                           cmap=resolve_cmap(cmap_xHI), rasterized=rasterized)

    im14 = ax14.pcolormesh(R_edges_pc, Z_edges_pc, xH2_ly_plot,
                           norm=norm_xH2, shading='auto',
                           cmap=resolve_cmap(cmap_xH2), rasterized=rasterized)

    images.append((im11, im12, im13, im14))


# -------------------------------------------------------------------
# Panel cosmetics
# -------------------------------------------------------------------

for row_axes in axes:
    for ax in row_axes:
        ax.set_xlim(r[0] / pc, xmax_fraction * r[-1] / pc)
        ax.set_ylim(Z[0] / pc, zmax_fraction * Z[-1] / pc)
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
# Group titles (top row only)
# -------------------------------------------------------------------

title_style = dict(
    ha='center',
    va='bottom',
    color='black',
    fontsize=fontsize_group_title,
    path_effects=[pe.withStroke(linewidth=3, foreground='white')]
)

noLya_title_x = 0.5 * (col_lefts[0] + col_lefts[1] + wpanel)
Lya_title_x   = 0.5 * (col_lefts[2] + col_lefts[3] + wpanel)

fig.text(noLya_title_x, fig_top - 0.002, "Ignoring Lyα Radiation Pressure", **title_style)
fig.text(Lya_title_x,   fig_top - 0.002, "With Lyα Radiation Pressure",     **title_style)


# -------------------------------------------------------------------
# Time labels
# -------------------------------------------------------------------

for row, (k_noLya, k_Lya) in enumerate(snapshot_pairs):
    # no-Lyα columns
    time_no = run_noLya["t"][k_noLya] / kyr
    # Lyα columns
    time_ly = run_Lya["t"][k_Lya] / kyr

    time_label(axes[row][0], time_no)
    time_label(axes[row][1], time_no)
    time_label(axes[row][2], time_ly)
    time_label(axes[row][3], time_ly)


# -------------------------------------------------------------------
# Panel quantity labels (top row only)
# -------------------------------------------------------------------

panel_label(axes[0][0], r"$x_{HI}$",  "xHI")
panel_label(axes[0][1], r"$x_{H_2}$", "xH2")
panel_label(axes[0][2], r"$x_{HI}$",  "xHI")
panel_label(axes[0][3], r"$x_{H_2}$", "xH2")


# -------------------------------------------------------------------
# Scale bar
# -------------------------------------------------------------------

scale_bar_axes = {}
for row in range(Nrows):
    for col in range(4):
        scale_bar_axes[f"ax{row + 1}{col + 1}"] = axes[row][col]

if ShowScaleBar:
    if scale_bar_ax_name not in scale_bar_axes:
        raise ValueError(
            f"scale_bar_ax_name={scale_bar_ax_name!r} not recognized for a "
            f"{Nrows}-row, 4-column figure. Choose from {list(scale_bar_axes.keys())}."
        )
    draw_scale_bar(scale_bar_axes[scale_bar_ax_name])


# -------------------------------------------------------------------
# Colorbars
# Two shared quantity colorbars: xHI and xH2, in the top-left pair.
# -------------------------------------------------------------------

def make_species_cbar(ax, im, norm, species_ticks, species_ticklabels,
                      species_tick_color, species_outline_color):
    ticks_auto, labels_auto = get_log_ticks_from_norm(norm)
    ticks, labels = use_custom_ticks(ticks_auto, labels_auto,
                                     species_ticks, species_ticklabels)
    tc, oc = resolve_cbar_colors(species_tick_color, species_outline_color)
    add_inset_colorbar(ax, im, ticks=ticks, ticklabels=labels,
                       tick_color=tc, outline_color=oc)

make_species_cbar(
    axes[0][0], images[0][0], norm_xHI,
    *resolve_ticks(ticks_xHI_custom, ticklabels_xHI_custom),
    cbar_tick_color_xHI, cbar_outline_color_xHI
)

make_species_cbar(
    axes[0][1], images[0][1], norm_xH2,
    *resolve_ticks(ticks_xH2_custom, ticklabels_xH2_custom),
    cbar_tick_color_xH2, cbar_outline_color_xH2
)


# -------------------------------------------------------------------
# Save figure
# -------------------------------------------------------------------

plt.savefig(FigureName, bbox_inches='tight', dpi=600)
plt.show()
