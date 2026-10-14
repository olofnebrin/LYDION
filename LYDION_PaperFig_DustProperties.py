import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import matplotlib.patheffects as pe
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
cmastro_src = repo_root / "src" / "cmastro" / "src"

sys.path.insert(0, str(cmastro_src))

import cmastro


# ---------------- User inputs ----------------
Filename         = "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5"
FigureName       = "LYDION_Figure_DustProperties_512D0.01.pdf"
TargetTime_kyr   = 15.0

# Figure/layout options
figsize          = (24, 32)
wgap             = 0.000
hgap             = 0.000
xmax_fraction    = 0.6
zmax_fraction    = 0.6
rasterized       = True

# Text sizes
fontsize_cbar_ticks  = 26
fontsize_panel_label = 60
fontsize_time        = 50

# Colormaps
cmap_Tgas       = 'coolwarm'
cmap_dustratio  = 'BuPu_r'
cmap_qPAH       = 'cubehelix'
cmap_sigmaLya   = 'magma'
cmap_Tdust      = 'RdBu_r'
cmap_charge     = 'inferno'

# Auto-scaling options
AutoScale_Tgas       = False
AutoScale_DustRatios = False
AutoScale_qPAH       = False
AutoScale_sigmaLya   = False
AutoScale_Tdust      = False
AutoScale_Charge     = False

# Manual ranges
Tgas_vmin, Tgas_vmax             = 1, 1e5

# Dust-to-gas ratios are now plotted in units of Dsun
dustratio_vmin, dustratio_vmax   = 1e-3, 1e-1

qPAH_vmin, qPAH_vmax             = 0.0, 5.0
sigmaLya_vmin, sigmaLya_vmax     = 1e-24, 1e-23
Tdust_vmin, Tdust_vmax           = 10.0, 1.5e3

# Grain-charge ranges. These can be non-symmetric.
Zgr_Sil_vmin, Zgr_Sil_vmax       = -10.0, 1000
Zgr_C_vmin,   Zgr_C_vmax         = -10.0, 400
Zgr_PAH_vmin, Zgr_PAH_vmax       = -1.0, 5.0

# Use TwoSlopeNorm for charge? 
# False = color bar spans the actual non-symmetric range normally.
# True  = force zero to be the center color.
ChargeUseTwoSlopeNorm = False

# Percentiles used when AutoScale_* = True
pct_low_log      = 1.0
pct_high_log     = 99.0
pct_high_linear  = 99.0
pct_charge_abs   = 99.0

# Colorbar options, in axis coordinates
cbar_x      = 0.80
cbar_y      = 0.59
cbar_width  = 0.040
cbar_height = 0.34

# Turn off tick marks, but keep tick labels
cbar_tick_length = 0
cbar_tick_width  = 0

# Default colorbar tick/outline color. Can be overridden per panel below.
cbar_tick_color_default = "black"
cbar_outline_color_default = "black"

cbar_tick_color_Tgas      = "black"
cbar_tick_color_DG        = "black"
cbar_tick_color_qPAH      = "white"
cbar_tick_color_sigmaLya  = "black"
cbar_tick_color_Tdust     = "black"
cbar_tick_color_Zgr_Sil   = "white"
cbar_tick_color_Zgr_C     = "white"
cbar_tick_color_Zgr_PAH   = "white"

cbar_outline_color_Tgas      = "black"
cbar_outline_color_DG        = "black"
cbar_outline_color_qPAH      = "white"
cbar_outline_color_sigmaLya  = "black"
cbar_outline_color_Tdust     = "black"
cbar_outline_color_Zgr_Sil   = "white"
cbar_outline_color_Zgr_C     = "white"
cbar_outline_color_Zgr_PAH   = "white"

# Custom colorbar ticks/labels. Set to None for automatic/default.
ticks_Tgas_custom         = None
ticklabels_Tgas_custom    = None

ticks_DG_custom           = None
ticklabels_DG_custom      = None

ticks_qPAH_custom         = [0, 1, 2, 3, 4, 5]
ticklabels_qPAH_custom    = None

ticks_sigmaLya_custom      = [1e-24, 1e-23]
ticklabels_sigmaLya_custom = [ r"$10^{-24}$", r"$10^{-23}$" ]
# Note: intermediate decade labels (2×10^{-24} etc.) are suppressed via
# minorticks_off() in add_inset_colorbar — no extra action needed here.

ticks_Tdust_custom         = None
ticklabels_Tdust_custom    = None

ticks_Zgr_Sil_custom       = [0,250.0, 500.0, 750.0, 1000]
ticklabels_Zgr_Sil_custom  = None

ticks_Zgr_C_custom         = [0, 100, 200, 300, 400]
ticklabels_Zgr_C_custom    = None

ticks_Zgr_PAH_custom       = [-1, 0, 1, 2, 3, 4, 5]
ticklabels_Zgr_PAH_custom  = None

# Time label color/outline
time_label_color        = "black"
time_label_stroke       = "white"
time_label_stroke_width = 3.0

# -------------------------------------------------------------------
# Panel label positions and style
# Each entry: (x, y) in axes coordinates, or None to use the default.
# Default position is (0.96, 0.04) — bottom-right.
# -------------------------------------------------------------------

# Default panel label style (used unless overridden per-panel below)
panel_label_default_x            = 0.96
panel_label_default_y            = 0.04
panel_label_default_ha           = 'right'
panel_label_default_va           = 'bottom'
panel_label_default_color        = 'white'
panel_label_default_stroke       = 'black'
panel_label_default_stroke_width = 3.5

# Per-panel overrides. Any key left as None falls back to the default above.
# Keys: 'x', 'y', 'ha', 'va', 'color', 'stroke', 'stroke_width'
# Example: give σ_Lyα panel a white label with black outline at a custom position:
#   panel_label_overrides["ax21"] = dict(x=0.96, y=0.04, color='white', stroke='black')
#
# Panels (b)--(e) have their physics labels (D/Dsun, qPAH, sigma_d,Lya/H,
# D_Sil/Dsun) moved toward the middle/top. Tweak these four values if needed.
panel_label_midtop_x  = 0.50
panel_label_midtop_y  = 0.94
panel_label_midtop_ha = 'center'
panel_label_midtop_va = 'top'

panel_label_overrides = {
    "ax11": dict(color='black', stroke='white'),
    "ax12": dict(x=panel_label_midtop_x, y=panel_label_midtop_y,
                 ha=panel_label_midtop_ha, va=panel_label_midtop_va,
                 color='black', stroke='white'),
    "ax13": dict(x=panel_label_midtop_x, y=panel_label_midtop_y,
                 ha=panel_label_midtop_ha, va=panel_label_midtop_va),
    "ax21": dict(x=panel_label_midtop_x, y=panel_label_midtop_y,
                 ha=panel_label_midtop_ha, va=panel_label_midtop_va,
                 color='black', stroke='white'),
    "ax22": dict(x=panel_label_midtop_x, y=panel_label_midtop_y,
                 ha=panel_label_midtop_ha, va=panel_label_midtop_va,
                 color='black', stroke='white'),
    "ax23": dict(x=panel_label_midtop_x, y=panel_label_midtop_y,
                 ha=panel_label_midtop_ha, va=panel_label_midtop_va,
                 color='black', stroke='white'),
    "ax31": dict(color='black', stroke='white'),
    "ax32": dict(color='black', stroke='white'),
    "ax33": dict(color='black', stroke='white'),
    "ax41": dict(),
    "ax42": dict(),
    "ax43": dict(),
}

# -------------------------------------------------------------------
# Alphabetical sub-labels: (a), (b), (c), …
# Set show_abc_labels = True to enable.
# abc_label_positions: dict mapping axis name -> (x, y), or None for default.
# abc_label_default_pos: fallback position for any panel not listed.
# -------------------------------------------------------------------

show_abc_labels           = True
abc_label_default_x       = 0.04
abc_label_default_y       = 0.96
abc_label_default_ha      = 'left'
abc_label_default_va      = 'top'
abc_label_default_color   = 'white'
abc_label_default_stroke  = 'black'
abc_label_default_stroke_width = 3.5
abc_label_fontsize        = 50   # can differ from fontsize_panel_label

# Per-panel position overrides for abc labels (same key scheme as panel_label_overrides)
abc_label_overrides = {
    "ax11": dict(x=0.75, y=0.25, color='black', stroke='white'),
    "ax12": dict(color='black', stroke='white'),
    "ax13": dict(),
    "ax21": dict(color='black', stroke='white'),
    "ax22": dict(color='black', stroke='white'),
    "ax23": dict(color='black', stroke='white'),
    "ax31": dict(color='black', stroke='white'),
    "ax32": dict(color='black', stroke='white'),
    "ax33": dict(color='black', stroke='white'),
    "ax41": dict(),
    "ax42": dict(),
    "ax43": dict(),
}

# -------------------------------------------------------------------
# Scale bar
# Draws a "|---|" bar with a text label above it on one panel.
# Same structure as the gas-abundance figure; placed in the upper
# middle panel by default.
#
# scale_bar_panel   : panel key, e.g. "ax12", or None to disable
# scale_bar_length  : physical length in pc
# scale_bar_label   : text shown above the bar (LaTeX ok)
# scale_bar_x       : centre of bar in data (pc) along the R axis
#                     None centres it horizontally in the visible panel
# scale_bar_y       : vertical position of bar in data (pc) along the Z axis
#                     None places it near the bottom (10 % up)
# scale_bar_color   : line and text colour
# scale_bar_lw      : line width
# scale_bar_capsize : height of the "|" end caps in pc
# scale_bar_fontsize: label font size (falls back to fontsize_panel_label)
# scale_bar_label_pad: gap between bar and label in pc
# scale_bar_stroke        : text stroke/outline colour, or None
# scale_bar_stroke_width  : stroke width in points
# -------------------------------------------------------------------
scale_bar_panel        = "ax12"
scale_bar_length       = 0.2          # pc
scale_bar_label        = "0.2 pc"
scale_bar_x            = 0.25         # pc; set to None to center automatically
scale_bar_y            = 0.05          # pc; set to None for automatic lower placement
scale_bar_color        = 'black'
scale_bar_lw           = 3.0
scale_bar_capsize      = None         # pc; None auto-sizes to ~4 % of panel height
scale_bar_fontsize     = 40           # None -> fontsize_panel_label
scale_bar_label_pad    = None         # pc; None auto-sizes to ~2 % of panel height
scale_bar_stroke       = 'black'
scale_bar_stroke_width = 0.0

# Useful constants
Dsun = 1 / 162
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


def nearest_saved_index(t, target_time):
    saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]
    if saved_idx.size == 0:
        raise RuntimeError("No saved snapshots found in file.")
    best = saved_idx[np.argmin(np.abs(t[saved_idx] - target_time))]
    return best


def positive_values(arr):
    vals = np.asarray(arr)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    return vals


def finite_values(arr):
    vals = np.asarray(arr)
    vals = vals[np.isfinite(vals)]
    return vals


def make_lognorm_from_arrays(arrays, default_vmin, default_vmax, pct_low=1.0, pct_high=99.0):
    vals = np.concatenate([positive_values(a) for a in arrays if a is not None])
    if vals.size == 0:
        return LogNorm(vmin=default_vmin, vmax=default_vmax)

    vmin = np.percentile(vals, pct_low)
    vmax = np.percentile(vals, pct_high)

    vmin = max(vmin, np.min(vals))
    vmax = min(vmax, np.max(vals))

    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin <= 0.0) or (vmax <= vmin):
        return LogNorm(vmin=default_vmin, vmax=default_vmax)

    return LogNorm(vmin=vmin, vmax=vmax)


def make_linear_norm_from_arrays(arrays, default_vmin, default_vmax, pct_high=99.0, force_nonnegative=False):
    vals = np.concatenate([finite_values(a) for a in arrays if a is not None])
    if vals.size == 0:
        return Normalize(vmin=default_vmin, vmax=default_vmax)

    if force_nonnegative:
        vals = vals[vals >= 0.0]
        if vals.size == 0:
            return Normalize(vmin=default_vmin, vmax=default_vmax)
        vmin = default_vmin
        vmax = np.percentile(vals, pct_high)
        vmax = max(vmax, np.max(vals))
        if (not np.isfinite(vmax)) or (vmax <= vmin):
            vmax = default_vmax
        return Normalize(vmin=vmin, vmax=vmax)

    vmin = np.min(vals)
    vmax = np.max(vals)
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
        return Normalize(vmin=default_vmin, vmax=default_vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def make_charge_norm_from_array(arr, default_vmin, default_vmax, pct_abs=99.0):
    vals = finite_values(arr)

    if vals.size == 0:
        vmin = default_vmin
        vmax = default_vmax
    else:
        vmax_abs = np.percentile(np.abs(vals), pct_abs)
        vmax_abs = max(vmax_abs, np.max(np.abs(vals)))
        if (not np.isfinite(vmax_abs)) or (vmax_abs <= 0.0):
            vmin = default_vmin
            vmax = default_vmax
        else:
            vmin = -vmax_abs
            vmax =  vmax_abs

    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
        vmin = default_vmin
        vmax = default_vmax

    if ChargeUseTwoSlopeNorm and (vmin < 0.0) and (vmax > 0.0):
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        return Normalize(vmin=vmin, vmax=vmax)


def make_charge_norm_from_limits(vmin, vmax):
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
        raise ValueError(f"Invalid charge range: vmin = {vmin}, vmax = {vmax}")

    if ChargeUseTwoSlopeNorm and (vmin < 0.0) and (vmax > 0.0):
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        return Normalize(vmin=vmin, vmax=vmax)


def add_inset_colorbar(ax, mappable, ticks=None, ticklabels=None,
                       tick_color=None, outline_color=None):
    if tick_color is None:
        tick_color = cbar_tick_color_default
    if outline_color is None:
        outline_color = cbar_outline_color_default

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

    # Suppress intermediate decade labels (e.g. 2×10^{-24}) that matplotlib's
    # LogLocator injects even when explicit ticks/labels are set.
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
    ticks = 10.0**np.arange(logmin, logmax + 1)
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


def panel_label(ax, text, overrides=None):
    """
    Draw the physics label (e.g. "T (K)") on a panel.
    Position and style fall back to panel_label_default_* unless overridden
    via the `overrides` dict (keys: x, y, ha, va, color, stroke, stroke_width).
    """
    ov = overrides or {}
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


def abc_label(ax, letter, overrides=None):
    """
    Draw an alphabetical sub-label, e.g. "(a)", on a panel.
    Only called when show_abc_labels = True.
    """
    ov = overrides or {}
    x            = ov.get('x',            abc_label_default_x)
    y            = ov.get('y',            abc_label_default_y)
    ha           = ov.get('ha',           abc_label_default_ha)
    va           = ov.get('va',           abc_label_default_va)
    color        = ov.get('color',        abc_label_default_color)
    stroke       = ov.get('stroke',       abc_label_default_stroke)
    stroke_width = ov.get('stroke_width', abc_label_default_stroke_width)

    style = dict(
        ha=ha, va=va,
        color=color,
        fontsize=abc_label_fontsize,
        path_effects=[pe.withStroke(linewidth=stroke_width, foreground=stroke)]
    )
    ax.text(x, y, f"({letter})", transform=ax.transAxes, **style)


def time_label(ax, time_kyr):
    style = dict(
        ha='left',
        va='top',
        color=time_label_color,
        fontsize=fontsize_time,
        path_effects=[pe.withStroke(linewidth=time_label_stroke_width, foreground=time_label_stroke)]
    )
    ax.text(0.04, 0.96, rf"t = {time_kyr:.2f} kyr", transform=ax.transAxes, **style)


def make_ax(fig, col_left, row_bottom, width, height, sharex=None, sharey=None):
    ax = fig.add_axes([col_left, row_bottom, width, height], sharex=sharex, sharey=sharey)
    return ax


# -------------------------------------------------------------------
# Scale bar helper
# -------------------------------------------------------------------

def draw_scale_bar(ax):
    """Draw a |---| scale bar with label on the chosen panel.

    All configuration is read from the global scale_bar_* variables.
    Call this after pcolormesh and after xlim/ylim are set so the bar
    sits on top and auto-positioning uses the final panel limits.
    """
    if scale_bar_panel is None:
        return

    # Axes data limits (pc)
    xlo, xhi = ax.get_xlim()   # R axis
    ylo, yhi = ax.get_ylim()   # Z axis
    xspan = xhi - xlo
    yspan = yhi - ylo

    # --- Resolve optional parameters ---
    bar_len  = scale_bar_length
    bar_x    = scale_bar_x    if scale_bar_x    is not None else xlo + 0.5 * xspan
    bar_y    = scale_bar_y    if scale_bar_y    is not None else ylo + 0.10 * yspan
    capsize  = scale_bar_capsize   if scale_bar_capsize  is not None else 0.04 * yspan
    lpad     = scale_bar_label_pad if scale_bar_label_pad is not None else 0.02 * yspan
    fontsize = scale_bar_fontsize  if scale_bar_fontsize  is not None else fontsize_panel_label
    color    = scale_bar_color
    lw       = scale_bar_lw

    x0 = bar_x - 0.5 * bar_len   # left end
    x1 = bar_x + 0.5 * bar_len   # right end

    # Path effects for text (optional stroke)
    if scale_bar_stroke is not None:
        text_effects = [pe.withStroke(linewidth=scale_bar_stroke_width,
                                      foreground=scale_bar_stroke)]
    else:
        text_effects = []

    # --- Draw the bar ---
    # Horizontal connecting line
    ax.plot([x0, x1], [bar_y, bar_y],
            color=color, lw=lw,
            solid_capstyle='butt',
            transform=ax.transData, zorder=10)

    # Left cap "|"
    ax.plot([x0, x0], [bar_y - 0.5 * capsize, bar_y + 0.5 * capsize],
            color=color, lw=lw,
            solid_capstyle='round',
            transform=ax.transData, zorder=10)

    # Right cap "|"
    ax.plot([x1, x1], [bar_y - 0.5 * capsize, bar_y + 0.5 * capsize],
            color=color, lw=lw,
            solid_capstyle='round',
            transform=ax.transData, zorder=10)

    # Label above the bar
    ax.text(bar_x, bar_y + 0.5 * capsize + lpad,
            scale_bar_label,
            color=color,
            fontsize=fontsize,
            ha='center', va='bottom',
            transform=ax.transData, zorder=11,
            path_effects=text_effects)


# -------------------------------------------------------------------
# Read run
# -------------------------------------------------------------------

with h5py.File(Filename, "r") as f:
    t        = f["Times"][:]
    r        = f["r"][:]
    Z        = f["Z"][:]

    rho_raw       = f["GasDensity_History"][:]
    Tgas_raw      = f["T_History"][:]

    rho_d_sil_raw = f["DustDensity_Sil_History"][:]
    rho_d_c_raw   = f["DustDensity_C_History"][:]
    rho_d_pah_raw = f["DustDensity_PAH_History"][:]

    Tdust_sil_raw = f["Tdust_Sil_History"][:]
    Tdust_c_raw   = f["Tdust_C_History"][:]
    Tdust_pah_raw = f["Tdust_PAH_History"][:]

    Zgr_sil_raw   = f["Zgr_Sil_History"][:]
    Zgr_c_raw     = f["Zgr_C_History"][:]
    Zgr_pah_raw   = f["Zgr_PAH_History"][:]

    sigmaLya_raw  = f["LyaDustAbsorptionCrossSections_History"][:]

rho_hist       = reorder_history_3d(rho_raw,       t, r, Z)
Tgas_hist      = reorder_history_3d(Tgas_raw,      t, r, Z)

rho_d_sil_hist = reorder_history_3d(rho_d_sil_raw, t, r, Z)
rho_d_c_hist   = reorder_history_3d(rho_d_c_raw,   t, r, Z)
rho_d_pah_hist = reorder_history_3d(rho_d_pah_raw, t, r, Z)

Tdust_sil_hist = reorder_history_3d(Tdust_sil_raw, t, r, Z)
Tdust_c_hist   = reorder_history_3d(Tdust_c_raw,   t, r, Z)
Tdust_pah_hist = reorder_history_3d(Tdust_pah_raw, t, r, Z)

Zgr_sil_hist   = reorder_history_3d(Zgr_sil_raw,   t, r, Z)
Zgr_c_hist     = reorder_history_3d(Zgr_c_raw,     t, r, Z)
Zgr_pah_hist   = reorder_history_3d(Zgr_pah_raw,   t, r, Z)

sigmaLya_hist  = reorder_history_3d(sigmaLya_raw,  t, r, Z)

k = nearest_saved_index(t, TargetTime_kyr * kyr)
print(f"Requested target time = {TargetTime_kyr:.3f} kyr")
print(f"Using snapshot at     = {t[k] / kyr:.3f} kyr")

rho_g      = rho_hist[k]
Tgas       = Tgas_hist[k]

rho_d_sil  = rho_d_sil_hist[k]
rho_d_c    = rho_d_c_hist[k]
rho_d_pah  = rho_d_pah_hist[k]
rho_d_tot  = rho_d_sil + rho_d_c + rho_d_pah

Tdust_sil  = Tdust_sil_hist[k]
Tdust_c    = Tdust_c_hist[k]
Tdust_pah  = Tdust_pah_hist[k]

Zgr_sil    = Zgr_sil_hist[k]
Zgr_c      = Zgr_c_hist[k]
Zgr_pah    = Zgr_pah_hist[k]

sigmaLya   = sigmaLya_hist[k]

# Dust-to-gas ratios, normalized to solar dust-to-gas ratio.
DG_tot = np.where(rho_g > 0.0, (rho_d_tot / rho_g) / Dsun, np.nan)
DG_sil = np.where(rho_g > 0.0, (rho_d_sil / rho_g) / Dsun, np.nan)
DG_c   = np.where(rho_g > 0.0, (rho_d_c   / rho_g) / Dsun, np.nan)
DG_pah = np.where(rho_g > 0.0, (rho_d_pah / rho_g) / Dsun, np.nan)

qPAH = np.where(rho_d_tot > 0.0, 100.0 * rho_d_pah / rho_d_tot, 0.0)

R_edges_pc, Z_edges_pc = np.meshgrid(r / pc, Z / pc, indexing='ij')


# -------------------------------------------------------------------
# Norms
# -------------------------------------------------------------------

if AutoScale_Tgas:
    norm_Tgas = make_lognorm_from_arrays([Tgas], Tgas_vmin, Tgas_vmax, pct_low_log, pct_high_log)
else:
    norm_Tgas = LogNorm(vmin=Tgas_vmin, vmax=Tgas_vmax)

if AutoScale_DustRatios:
    norm_DG = make_lognorm_from_arrays(
        [DG_tot, DG_sil, DG_c, DG_pah],
        dustratio_vmin, dustratio_vmax, pct_low_log, pct_high_log
    )
else:
    norm_DG = LogNorm(vmin=dustratio_vmin, vmax=dustratio_vmax)

if AutoScale_qPAH:
    norm_qPAH = make_linear_norm_from_arrays([qPAH], qPAH_vmin, qPAH_vmax, pct_high_linear, force_nonnegative=True)
    norm_qPAH = Normalize(vmin=0.0, vmax=max(norm_qPAH.vmax, 1.0))
else:
    norm_qPAH = Normalize(vmin=qPAH_vmin, vmax=qPAH_vmax)

if AutoScale_sigmaLya:
    norm_sigmaLya = make_lognorm_from_arrays([sigmaLya], sigmaLya_vmin, sigmaLya_vmax, pct_low_log, pct_high_log)
else:
    norm_sigmaLya = LogNorm(vmin=sigmaLya_vmin, vmax=sigmaLya_vmax)

if AutoScale_Tdust:
    norm_Tdust = make_lognorm_from_arrays(
        [Tdust_sil, Tdust_c, Tdust_pah],
        Tdust_vmin, Tdust_vmax, pct_low_log, pct_high_log
    )
else:
    norm_Tdust = LogNorm(vmin=Tdust_vmin, vmax=Tdust_vmax)

if AutoScale_Charge:
    norm_charge_sil = make_charge_norm_from_array(Zgr_sil, Zgr_Sil_vmin, Zgr_Sil_vmax, pct_charge_abs)
    norm_charge_c   = make_charge_norm_from_array(Zgr_c,   Zgr_C_vmin,   Zgr_C_vmax,   pct_charge_abs)
    norm_charge_pah = make_charge_norm_from_array(Zgr_pah, Zgr_PAH_vmin, Zgr_PAH_vmax, pct_charge_abs)
else:
    norm_charge_sil = make_charge_norm_from_limits(Zgr_Sil_vmin, Zgr_Sil_vmax)
    norm_charge_c   = make_charge_norm_from_limits(Zgr_C_vmin,   Zgr_C_vmax)
    norm_charge_pah = make_charge_norm_from_limits(Zgr_PAH_vmin, Zgr_PAH_vmax)


# -------------------------------------------------------------------
# Clip fields for plotting
# -------------------------------------------------------------------

Tgas_plot      = np.clip(Tgas.T,      norm_Tgas.vmin,       norm_Tgas.vmax)
DG_tot_plot    = np.clip(DG_tot.T,    norm_DG.vmin,         norm_DG.vmax)
qPAH_plot      = np.clip(qPAH.T,      norm_qPAH.vmin,       norm_qPAH.vmax)
sigmaLya_plot  = np.clip(sigmaLya.T,  norm_sigmaLya.vmin,   norm_sigmaLya.vmax)

DG_sil_plot    = np.clip(DG_sil.T,    norm_DG.vmin,         norm_DG.vmax)
DG_c_plot      = np.clip(DG_c.T,      norm_DG.vmin,         norm_DG.vmax)

Tdust_sil_plot = np.clip(Tdust_sil.T, norm_Tdust.vmin,      norm_Tdust.vmax)
Tdust_c_plot   = np.clip(Tdust_c.T,   norm_Tdust.vmin,      norm_Tdust.vmax)
Tdust_pah_plot = np.clip(Tdust_pah.T, norm_Tdust.vmin,      norm_Tdust.vmax)

Zgr_sil_plot   = np.clip(Zgr_sil.T,   norm_charge_sil.vmin, norm_charge_sil.vmax)
Zgr_c_plot     = np.clip(Zgr_c.T,     norm_charge_c.vmin,   norm_charge_c.vmax)
Zgr_pah_plot   = np.clip(Zgr_pah.T,   norm_charge_pah.vmin, norm_charge_pah.vmax)


# -------------------------------------------------------------------
# Set up figure: 4 rows × 3 cols
# -------------------------------------------------------------------

wpanel = (1.0 - 2.0 * wgap) / 3.0
hpanel = (1.0 - 3.0 * hgap) / 4.0

col0_left = 0.0
col1_left = wpanel + wgap
col2_left = 2.0 * (wpanel + wgap)

row0_bottom = 3.0 * (hpanel + hgap)
row1_bottom = 2.0 * (hpanel + hgap)
row2_bottom = 1.0 * (hpanel + hgap)
row3_bottom = 0.0

fig = plt.figure(figsize=figsize)

ax11 = make_ax(fig, col0_left, row0_bottom, wpanel, hpanel)
ax12 = make_ax(fig, col1_left, row0_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)
ax13 = make_ax(fig, col2_left, row0_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)

ax21 = make_ax(fig, col0_left, row1_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)
ax22 = make_ax(fig, col1_left, row1_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)
ax23 = make_ax(fig, col2_left, row1_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)

ax31 = make_ax(fig, col0_left, row2_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)
ax32 = make_ax(fig, col1_left, row2_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)
ax33 = make_ax(fig, col2_left, row2_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)

ax41 = make_ax(fig, col0_left, row3_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)
ax42 = make_ax(fig, col1_left, row3_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)
ax43 = make_ax(fig, col2_left, row3_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)


# -------------------------------------------------------------------
# Plot panels
# -------------------------------------------------------------------

im11 = ax11.pcolormesh(R_edges_pc, Z_edges_pc, Tgas_plot,      norm=norm_Tgas,       shading='auto', cmap=cmap_Tgas,      rasterized=rasterized)
im12 = ax12.pcolormesh(R_edges_pc, Z_edges_pc, DG_tot_plot,    norm=norm_DG,         shading='auto', cmap=cmap_dustratio, rasterized=rasterized)
im13 = ax13.pcolormesh(R_edges_pc, Z_edges_pc, qPAH_plot,      norm=norm_qPAH,       shading='auto', cmap=cmap_qPAH,      rasterized=rasterized)

im21 = ax21.pcolormesh(R_edges_pc, Z_edges_pc, sigmaLya_plot,  norm=norm_sigmaLya,   shading='auto', cmap=cmap_sigmaLya,  rasterized=rasterized)
im22 = ax22.pcolormesh(R_edges_pc, Z_edges_pc, DG_sil_plot,    norm=norm_DG,         shading='auto', cmap=cmap_dustratio, rasterized=rasterized)
im23 = ax23.pcolormesh(R_edges_pc, Z_edges_pc, DG_c_plot,      norm=norm_DG,         shading='auto', cmap=cmap_dustratio, rasterized=rasterized)

im31 = ax31.pcolormesh(R_edges_pc, Z_edges_pc, Tdust_sil_plot, norm=norm_Tdust,      shading='auto', cmap=cmap_Tdust,     rasterized=rasterized)
im32 = ax32.pcolormesh(R_edges_pc, Z_edges_pc, Tdust_c_plot,   norm=norm_Tdust,      shading='auto', cmap=cmap_Tdust,     rasterized=rasterized)
im33 = ax33.pcolormesh(R_edges_pc, Z_edges_pc, Tdust_pah_plot, norm=norm_Tdust,      shading='auto', cmap=cmap_Tdust,     rasterized=rasterized)

im41 = ax41.pcolormesh(R_edges_pc, Z_edges_pc, Zgr_sil_plot,   norm=norm_charge_sil, shading='auto', cmap=cmap_charge,    rasterized=rasterized)
im42 = ax42.pcolormesh(R_edges_pc, Z_edges_pc, Zgr_c_plot,     norm=norm_charge_c,   shading='auto', cmap=cmap_charge,    rasterized=rasterized)
im43 = ax43.pcolormesh(R_edges_pc, Z_edges_pc, Zgr_pah_plot,   norm=norm_charge_pah, shading='auto', cmap=cmap_charge,    rasterized=rasterized)

ax12.set_title("Dust abundances, properties, and absorption cross-section", y = 1.04, fontsize = 65)

# -------------------------------------------------------------------
# Panel cosmetics
# -------------------------------------------------------------------

all_axes = (ax11, ax12, ax13,
            ax21, ax22, ax23,
            ax31, ax32, ax33,
            ax41, ax42, ax43)

for ax in all_axes:
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


time_kyr_actual = t[k] / kyr
time_label(ax11, time_kyr_actual)

# Scale bar (drawn after xlim/ylim are set so auto-positioning is correct)
_scale_bar_ax_map = {
    "ax11": ax11, "ax12": ax12, "ax13": ax13,
    "ax21": ax21, "ax22": ax22, "ax23": ax23,
    "ax31": ax31, "ax32": ax32, "ax33": ax33,
    "ax41": ax41, "ax42": ax42, "ax43": ax43,
}
if scale_bar_panel is not None:
    if scale_bar_panel not in _scale_bar_ax_map:
        raise ValueError(f"Unknown scale_bar_panel = {scale_bar_panel}")
    draw_scale_bar(_scale_bar_ax_map[scale_bar_panel])

panel_label(ax11, r"T (K)",                               panel_label_overrides.get("ax11"))
panel_label(ax12, r"$\mathfrak{D}/\mathfrak{D}_{\odot}$", panel_label_overrides.get("ax12"))
panel_label(ax13, r"$q_{PAH}$ (%)",                       panel_label_overrides.get("ax13"))

panel_label(ax21, r"$\sigma_{d,Ly\alpha}/H$",                   panel_label_overrides.get("ax21"))
panel_label(ax22, r"$\mathfrak{D}_{Sil}/\mathfrak{D}_{\odot}$", panel_label_overrides.get("ax22"))
panel_label(ax23, r"$\mathfrak{D}_{C}/\mathfrak{D}_{\odot}$",   panel_label_overrides.get("ax23"))

panel_label(ax31, r"$T_{d,Sil}$ (K)", panel_label_overrides.get("ax31"))
panel_label(ax32, r"$T_{d,C}$ (K)",   panel_label_overrides.get("ax32"))
panel_label(ax33, r"$T_{d,PAH}$ (K)", panel_label_overrides.get("ax33"))

panel_label(ax41, r"$Z_{gr,Sil}$", panel_label_overrides.get("ax41"))
panel_label(ax42, r"$Z_{gr,C}$",   panel_label_overrides.get("ax42"))
panel_label(ax43, r"$Z_{gr,PAH}$", panel_label_overrides.get("ax43"))

# Alphabetical sub-labels (a), (b), … — enable via show_abc_labels
if show_abc_labels:
    import string
    _abc_axes = [
        ("ax11", ax11), ("ax12", ax12), ("ax13", ax13),
        ("ax21", ax21), ("ax22", ax22), ("ax23", ax23),
        ("ax31", ax31), ("ax32", ax32), ("ax33", ax33),
        ("ax41", ax41), ("ax42", ax42), ("ax43", ax43),
    ]
    for idx, (key, ax) in enumerate(_abc_axes):
        abc_label(ax, string.ascii_lowercase[idx], abc_label_overrides.get(key))


# -------------------------------------------------------------------
# Colorbars
# -------------------------------------------------------------------

ticks_Tgas,      ticklabels_Tgas      = get_log_ticks_from_norm(norm_Tgas)
ticks_DG,        ticklabels_DG        = get_log_ticks_from_norm(norm_DG)
ticks_sigmaLya,  ticklabels_sigmaLya  = get_log_ticks_from_norm(norm_sigmaLya)
ticks_Tdust,     ticklabels_Tdust     = get_log_ticks_from_norm(norm_Tdust)

ticks_Tgas, ticklabels_Tgas = use_custom_ticks(
    ticks_Tgas, ticklabels_Tgas, ticks_Tgas_custom, ticklabels_Tgas_custom
)
ticks_DG, ticklabels_DG = use_custom_ticks(
    ticks_DG, ticklabels_DG, ticks_DG_custom, ticklabels_DG_custom
)
ticks_qPAH, ticklabels_qPAH = use_custom_ticks(
    None, None, ticks_qPAH_custom, ticklabels_qPAH_custom
)
ticks_sigmaLya, ticklabels_sigmaLya = use_custom_ticks(
    ticks_sigmaLya, ticklabels_sigmaLya, ticks_sigmaLya_custom, ticklabels_sigmaLya_custom
)
ticks_Tdust, ticklabels_Tdust = use_custom_ticks(
    ticks_Tdust, ticklabels_Tdust, ticks_Tdust_custom, ticklabels_Tdust_custom
)
ticks_Zgr_Sil, ticklabels_Zgr_Sil = use_custom_ticks(
    None, None, ticks_Zgr_Sil_custom, ticklabels_Zgr_Sil_custom
)
ticks_Zgr_C, ticklabels_Zgr_C = use_custom_ticks(
    None, None, ticks_Zgr_C_custom, ticklabels_Zgr_C_custom
)
ticks_Zgr_PAH, ticklabels_Zgr_PAH = use_custom_ticks(
    None, None, ticks_Zgr_PAH_custom, ticklabels_Zgr_PAH_custom
)

add_inset_colorbar(ax11, im11, ticks_Tgas,     ticklabels_Tgas,
                   tick_color=cbar_tick_color_Tgas,
                   outline_color=cbar_outline_color_Tgas)

add_inset_colorbar(ax12, im12, ticks_DG,       ticklabels_DG,
                   tick_color=cbar_tick_color_DG,
                   outline_color=cbar_outline_color_DG)

add_inset_colorbar(ax13, im13, ticks_qPAH,     ticklabels_qPAH,
                   tick_color=cbar_tick_color_qPAH,
                   outline_color=cbar_outline_color_qPAH)

add_inset_colorbar(ax21, im21, ticks_sigmaLya, ticklabels_sigmaLya,
                   tick_color=cbar_tick_color_sigmaLya,
                   outline_color=cbar_outline_color_sigmaLya)

add_inset_colorbar(ax22, im22, ticks_DG,       ticklabels_DG,
                   tick_color=cbar_tick_color_DG,
                   outline_color=cbar_outline_color_DG)

add_inset_colorbar(ax23, im23, ticks_DG,       ticklabels_DG,
                   tick_color=cbar_tick_color_DG,
                   outline_color=cbar_outline_color_DG)

add_inset_colorbar(ax31, im31, ticks_Tdust,    ticklabels_Tdust,
                   tick_color=cbar_tick_color_Tdust,
                   outline_color=cbar_outline_color_Tdust)

add_inset_colorbar(ax32, im32, ticks_Tdust,    ticklabels_Tdust,
                   tick_color=cbar_tick_color_Tdust,
                   outline_color=cbar_outline_color_Tdust)

add_inset_colorbar(ax33, im33, ticks_Tdust,    ticklabels_Tdust,
                   tick_color=cbar_tick_color_Tdust,
                   outline_color=cbar_outline_color_Tdust)

add_inset_colorbar(ax41, im41, ticks_Zgr_Sil,  ticklabels_Zgr_Sil,
                   tick_color=cbar_tick_color_Zgr_Sil,
                   outline_color=cbar_outline_color_Zgr_Sil)

add_inset_colorbar(ax42, im42, ticks_Zgr_C,    ticklabels_Zgr_C,
                   tick_color=cbar_tick_color_Zgr_C,
                   outline_color=cbar_outline_color_Zgr_C)

add_inset_colorbar(ax43, im43, ticks_Zgr_PAH,  ticklabels_Zgr_PAH,
                   tick_color=cbar_tick_color_Zgr_PAH,
                   outline_color=cbar_outline_color_Zgr_PAH)


# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------

plt.savefig(FigureName, bbox_inches='tight')
print(f"Saved figure to {FigureName}")