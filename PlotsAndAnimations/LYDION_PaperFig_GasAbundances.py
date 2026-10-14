import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
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
Filename         = "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5"
FigureName       = "LYDION_Figure_GasAbundances_512D0.01.pdf"
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

# -------------------------------------------------------------------
# Colormaps
# -------------------------------------------------------------------
cmap_Tgas       = 'coolwarm'       # Gas temperature
cmap_abundance  = 'BuPu_r'        # All abundance panels (shared default)
#cmap_abundance  = 'RdPu_r'

# Per-species colormap overrides. Set to None to use cmap_abundance.
cmap_xHI    = None
cmap_xH2    = None
cmap_xHeI   = None
cmap_xHeII  = None
cmap_xHeIII = None
cmap_xCI    = None
cmap_xCII   = None
cmap_xCO    = None
cmap_xOI    = None
cmap_xOII   = None
cmap_xHD    = None

# -------------------------------------------------------------------
# Auto-scaling options
# -------------------------------------------------------------------
AutoScale_Tgas       = False
AutoScale_Abundances = False   # Applies to all abundance panels

# -------------------------------------------------------------------
# Manual ranges
# -------------------------------------------------------------------
Tgas_vmin, Tgas_vmax   = 1, 1e5

# Shared abundance range (applied to all panels unless overridden below)
abund_vmin, abund_vmax  = 1e-8, 1.0

# Per-species range overrides. Set to None to use abund_vmin/abund_vmax.
xHI_vmin,    xHI_vmax    = None, None
xH2_vmin,    xH2_vmax    = None, None
xHeI_vmin,   xHeI_vmax   = None, None
xHeII_vmin,  xHeII_vmax  = None, None
xHeIII_vmin, xHeIII_vmax = None, None
xCI_vmin,    xCI_vmax    = None, None
xCII_vmin,   xCII_vmax   = None, None
xCO_vmin,    xCO_vmax    = None, None
xOI_vmin,    xOI_vmax    = None, None
xOII_vmin,   xOII_vmax   = None, None
xHD_vmin,    xHD_vmax    = None, None

# Percentiles used when AutoScale_* = True
pct_low_log   = 1.0
pct_high_log  = 99.0

# -------------------------------------------------------------------
# Colorbar options (in axis coordinates, relative to each panel).
# Increase cbar_height (e.g. 0.50–0.60) if the bar feels too short
# when the dynamic range is large (e.g. 1e-8 to 1).
# -------------------------------------------------------------------
cbar_x      = 0.80
cbar_y      = 0.50
cbar_width  = 0.040
cbar_height = 0.45

# Per-panel colorbar geometry overrides.
# Set any key to override the global value for that panel only.
# Example: cbar_geom_overrides["ax12"] = dict(y=0.40, height=0.52)
cbar_geom_overrides = {
    "ax11": {},
    "ax12": {},
    "ax13": {},
    "ax21": {},
    "ax22": {},
    "ax23": {},
    "ax31": {},
    "ax32": {},
    "ax33": {},
    "ax41": {},
    "ax42": {},
    "ax43": {},
}

# Turn off tick marks, but keep tick labels
cbar_tick_length = 0
cbar_tick_width  = 0

# -------------------------------------------------------------------
# Contour overlays
# Specify per-panel contour levels, colours, and linestyle.
# Set to None (or omit) to draw no contours on that panel.
#
# Each entry is a list of dicts, one dict per contour group.
# Keys per group:
#   levels           : list of values to contour              (required)
#   colors           : colour string or list of strings       (default: 'white')
#   linewidths       : float or list                          (default: 1.5)
#   linestyle        : 'solid', 'dashed', 'dotted', etc.     (default: 'solid')
#   rounded          : bool — round cap/join style            (default: False)
#   labels           : list of strings, one per level, or
#                      None for no labels                     (default: None)
#                      Use r"$10^{-5}$" etc. for LaTeX.
#   label_fontsize   : float, overrides contour_label_fontsize_default
#   label_color      : colour string for label text           (default: same as colors)
#   label_fmt        : format string passed to clabel if labels is None
#                      (e.g. '%.0e') — only used when labels is also None
#
# Example:
#   contour_config["ax12"] = [
#       dict(levels=[0.1, 1e-5],
#            colors='white', linestyle='dashed', rounded=True,
#            labels=[r'$0.1$', r'$10^{-5}$'],
#            label_fontsize=28, label_color='white'),
#   ]
# -------------------------------------------------------------------
contour_label_fontsize_default = 32    # fallback label fontsize for all panels
contour_config = {
    "ax11": None,
    "ax12": [ dict(levels=[1e-5, 1e-4], labels=[r'$10^{-5}$', r'$10^{-4}$'], colors='black', 
                   linestyle='dashed', rounded=True, linewidths=2.5) ],
    "ax13": [ dict(levels=[0.1, 0.4], labels=[r'$0.1$', r'$0.4$'], colors='black', 
                   linestyle='dashed', rounded=True, linewidths=2.5) ],
    "ax21": None,
    "ax22": None,
    "ax23": None,
    "ax31": None,
    "ax32": None,
    "ax33": [ dict(levels=[0.1, 0.5], labels=[r'$0.1$', r'$0.5$'], colors='black', 
                   linestyle='dashed', rounded=True, linewidths=2.5) ],
    "ax41": None,
    "ax42": None,
    "ax43": [ dict(levels=[0.8], labels=[r'$0.8$'], colors='black', 
                   linestyle='dashed', rounded=True, linewidths=2.5) ],
}

# Default colorbar tick/outline color. Can be overridden per panel below.
cbar_tick_color_default    = "black"
cbar_outline_color_default = "black"

cbar_tick_color_Tgas    = "black"
cbar_outline_color_Tgas = "black"

# Shared abundance colorbar colors (used for all abundance panels unless overridden)
cbar_tick_color_abund    = "black"
cbar_outline_color_abund = "black"

# Per-species colorbar color overrides. Set to None to use the shared abundance colors.
cbar_tick_color_xHI,    cbar_outline_color_xHI    = None, None
cbar_tick_color_xH2,    cbar_outline_color_xH2    = "white", "white"
cbar_tick_color_xHeI,   cbar_outline_color_xHeI   = None, None
cbar_tick_color_xHeII,  cbar_outline_color_xHeII  = "white", "white"
cbar_tick_color_xHeIII, cbar_outline_color_xHeIII = "white", "white"
cbar_tick_color_xCI,    cbar_outline_color_xCI    = None, None
cbar_tick_color_xCII,   cbar_outline_color_xCII   = "white", "white"
cbar_tick_color_xCO,    cbar_outline_color_xCO    = "white", "white"
cbar_tick_color_xOI,    cbar_outline_color_xOI    = None, None
cbar_tick_color_xOII,   cbar_outline_color_xOII   = None, None
cbar_tick_color_xHD,    cbar_outline_color_xHD    = "white", "white"

# -------------------------------------------------------------------
# Custom colorbar ticks/labels. Set to None for automatic.
# -------------------------------------------------------------------
ticks_Tgas_custom      = None
ticklabels_Tgas_custom = None

# Shared abundance ticks (used for all abundance panels unless overridden)
ticks_abund_custom      = None
ticklabels_abund_custom = None

# Per-species tick overrides. Set to None to use the shared abundance ticks.
ticks_xHI_custom,    ticklabels_xHI_custom    = None, None
ticks_xH2_custom,    ticklabels_xH2_custom    = None, None
ticks_xHeI_custom,   ticklabels_xHeI_custom   = None, None
ticks_xHeII_custom,  ticklabels_xHeII_custom  = None, None
ticks_xHeIII_custom, ticklabels_xHeIII_custom = None, None
ticks_xCI_custom,    ticklabels_xCI_custom    = None, None
ticks_xCII_custom,   ticklabels_xCII_custom   = None, None
ticks_xCO_custom,    ticklabels_xCO_custom    = None, None
ticks_xOI_custom,    ticklabels_xOI_custom    = None, None
ticks_xOII_custom,   ticklabels_xOII_custom   = None, None
ticks_xHD_custom,    ticklabels_xHD_custom    = None, None

# -------------------------------------------------------------------
# Time label color/outline
# -------------------------------------------------------------------
time_label_color        = "black"
time_label_stroke       = "white"
time_label_stroke_width = 3.0

# -------------------------------------------------------------------
# Panel label positions and style
# Default position: (0.96, 0.04) — bottom-right, right-aligned.
# Keys per panel: 'x', 'y', 'ha', 'va', 'color', 'stroke', 'stroke_width'
# -------------------------------------------------------------------
panel_label_default_x            = 0.96
panel_label_default_y            = 0.04
panel_label_default_ha           = 'right'
panel_label_default_va           = 'bottom'
panel_label_default_color        = 'white'
panel_label_default_stroke       = 'black'
panel_label_default_stroke_width = 3.5

panel_label_overrides = {
    "ax11": dict(color='black', stroke='white'),   # T — light colormap
    "ax12": dict(color='black', stroke='white'),
    "ax13": dict(),
    "ax21": dict(color='black', stroke='white'),
    "ax22": dict(),
    "ax23": dict(),
    "ax31": dict(color='black', stroke='white'),
    "ax32": dict(),
    "ax33": dict(),
    "ax41": dict(color='black', stroke='white'),
    "ax42": dict(),
    "ax43": dict(),
}

# -------------------------------------------------------------------
# Alphabetical sub-labels: (a), (b), (c), …
# Set show_abc_labels = True to enable.
# -------------------------------------------------------------------
show_abc_labels                = True
abc_label_default_x            = 0.04
abc_label_default_y            = 0.96
abc_label_default_ha           = 'left'
abc_label_default_va           = 'top'
abc_label_default_color        = 'white'
abc_label_default_stroke       = 'black'
abc_label_default_stroke_width = 3.5
abc_label_fontsize             = 50

abc_label_overrides = {
    "ax11": dict(x=0.75, y=0.25, color='black', stroke='white'),   # T — light colormap
    "ax12": dict(color='black', stroke='white'),
    "ax13": dict(),
    "ax21": dict(color='black', stroke='white'),
    "ax22": dict(),
    "ax23": dict(),
    "ax31": dict(color='black', stroke='white'),
    "ax32": dict(),
    "ax33": dict(),
    "ax41": dict(color='black', stroke='white'),
    "ax42": dict(),
    "ax43": dict(),
}

# -------------------------------------------------------------------
# Scale bar
# Draws a "|---|" bar with a text label above it on one panel.
#
# scale_bar_panel   : panel key, e.g. "ax11", or None to disable
# scale_bar_length  : physical length in pc
# scale_bar_label   : text shown above the bar (LaTeX ok)
#                     e.g. r"$0.1\,\mathrm{pc}$"
# scale_bar_x       : centre of bar in data (pc) along the R axis
# scale_bar_y       : vertical position of bar in data (pc) along the Z axis
# scale_bar_color   : line and text colour
# scale_bar_lw      : line width
# scale_bar_capsize : height of the "|" end caps in pc
# scale_bar_fontsize: label font size (falls back to fontsize_panel_label)
# scale_bar_label_pad: gap between bar and label in pc
# scale_bar_stroke        : text stroke/outline colour, or None
# scale_bar_stroke_width  : stroke width in points
# -------------------------------------------------------------------
scale_bar_panel        = "ax12"       # e.g. "ax11"
scale_bar_length       = 0.2          # pc
scale_bar_label        = "0.2 pc"
scale_bar_x            = 0.35         # pc — set to a value to position manually;
                                      # None centres it horizontally in the panel
scale_bar_y            = 0.6          # pc — None places it near the bottom (10 % up)
scale_bar_color        = 'black'
scale_bar_lw           = 3.0
scale_bar_capsize      = None         # pc — None auto-sizes to ~4 % of panel height
scale_bar_fontsize     = 40           # None → fontsize_panel_label
scale_bar_label_pad    = None         # pc — None auto-sizes to ~2 % of panel height
scale_bar_stroke       = 'white'
scale_bar_stroke_width = 3.0

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
    """Reorder a 3D history array to shape (Nt, Nr, Nz)."""
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
    return vals[np.isfinite(vals) & (vals > 0.0)]


def finite_values(arr):
    vals = np.asarray(arr)
    return vals[np.isfinite(vals)]


def make_lognorm_from_arrays(arrays, default_vmin, default_vmax,
                              pct_low=1.0, pct_high=99.0):
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


def resolve_vmin_vmax(species_vmin, species_vmax):
    """Return per-species range, falling back to shared abundance range."""
    vmin = species_vmin if species_vmin is not None else abund_vmin
    vmax = species_vmax if species_vmax is not None else abund_vmax
    return vmin, vmax


def resolve_cbar_colors(species_tick_color, species_outline_color):
    """Return per-species cbar colors, falling back to shared abundance colors."""
    tc = species_tick_color    if species_tick_color    is not None else cbar_tick_color_abund
    oc = species_outline_color if species_outline_color is not None else cbar_outline_color_abund
    return tc, oc


def resolve_cmap(species_cmap):
    """Return per-species colormap, falling back to shared abundance colormap."""
    return species_cmap if species_cmap is not None else cmap_abundance


def resolve_ticks(species_ticks, species_ticklabels):
    """Return per-species ticks/labels, falling back to shared abundance ticks."""
    ticks  = species_ticks      if species_ticks      is not None else ticks_abund_custom
    labels = species_ticklabels if species_ticklabels is not None else ticklabels_abund_custom
    return ticks, labels


def add_inset_colorbar(ax, mappable, ticks=None, ticklabels=None,
                       tick_color=None, outline_color=None,
                       geom_overrides=None):
    if tick_color is None:
        tick_color = cbar_tick_color_default
    if outline_color is None:
        outline_color = cbar_outline_color_default

    go = geom_overrides or {}
    _x      = go.get('x',      cbar_x)
    _y      = go.get('y',      cbar_y)
    _width  = go.get('width',  cbar_width)
    _height = go.get('height', cbar_height)

    cax = ax.inset_axes([_x, _y, _width, _height])
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

    # Suppress intermediate decade labels that matplotlib's LogLocator injects.
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
    """Draw the physics label on a panel, with per-panel style overrides."""
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
    """Draw an alphabetical sub-label, e.g. '(a)', on a panel."""
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
    return fig.add_axes([col_left, row_bottom, width, height], sharex=sharex, sharey=sharey)


def draw_contours(ax, field_2d, R_edges_pc, Z_edges_pc, config_list):
    """Overlay contour lines on a panel.

    Parameters
    ----------
    ax          : matplotlib Axes
    field_2d    : 2D array, shape (Nr, Nz), un-transposed (same as the raw
                  field before .T for pcolormesh).  We transpose internally.
    R_edges_pc  : R edge coordinates in pc (meshgrid, shape (Nr+1, Nz+1))
    Z_edges_pc  : Z edge coordinates in pc (same shape)
    config_list : list of dicts, one per contour group (see contour_config docs)
    """
    if config_list is None:
        return

    # Cell centres in pc from the edge arrays
    R_cen = 0.5 * (R_edges_pc[:-1, :-1] + R_edges_pc[1:, :-1])   # (Nr, Nz)
    Z_cen = 0.5 * (Z_edges_pc[:-1, :-1] + Z_edges_pc[:-1, 1:])   # (Nr, Nz)

    field_plot = field_2d   # shape (Nr, Nz)

    for group in config_list:
        levels          = group['levels']
        colors          = group.get('colors',          'white')
        linewidths      = group.get('linewidths',      1.5)
        linestyle       = group.get('linestyle',       'solid')
        rounded         = group.get('rounded',         False)
        labels          = group.get('labels',          None)
        label_fontsize  = group.get('label_fontsize',  contour_label_fontsize_default)
        label_fmt       = group.get('label_fmt',       None)

        # label_color defaults to the line color (first entry if a list)
        _default_lc = colors[0] if isinstance(colors, (list, tuple)) else colors
        label_color = group.get('label_color', _default_lc)

        cs = ax.contour(R_cen, Z_cen, field_plot.T,
                        levels=levels,
                        colors=colors,
                        linewidths=linewidths,
                        linestyles=linestyle)

        if rounded:
            if hasattr(cs, 'collections'):          # matplotlib < 3.8
                for lc in cs.collections:
                    lc.set_capstyle('round')
                    lc.set_joinstyle('round')
            else:                                   # matplotlib >= 3.10
                cs.set_capstyle('round')
                cs.set_joinstyle('round')

        # --- Labels ---
        if labels is not None:
            # Build a {level: label_string} mapping for clabel's fmt argument.
            fmt_dict = {lv: lb for lv, lb in zip(levels, labels)}
            clbls = ax.clabel(cs, levels=levels,
                              fmt=fmt_dict,
                              fontsize=label_fontsize,
                              inline=True)
            for txt in clbls:
                txt.set_color(label_color)
        elif label_fmt is not None:
            clbls = ax.clabel(cs, levels=levels,
                              fmt=label_fmt,
                              fontsize=label_fontsize,
                              inline=True)
            for txt in clbls:
                txt.set_color(label_color)



# -------------------------------------------------------------------
# Scale bar helper
# -------------------------------------------------------------------

def draw_scale_bar(ax):
    """Draw a |---| scale bar with label on the chosen panel.

    All configuration is read from the global scale_bar_* variables.
    Call this after pcolormesh so the bar sits on top.
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

    # Left cap  "|"
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
# Read data
# -------------------------------------------------------------------

with h5py.File(Filename, "r") as f:
    t = f["Times"][:]
    r = f["r"][:]
    Z = f["Z"][:]

    Tgas_raw    = f["T_History"][:]

    # H species — xHI and xHeI are derived (see below)
    xHII_raw    = f["xHII_History"][:]
    xH2_raw     = f["xH2_History"][:]

    # He species
    xHeII_raw   = f["xHeII_History"][:]
    xHeIII_raw  = f["xHeIII_History"][:]

    # C species
    xCI_raw     = f["xCI_History"][:]
    xCII_raw    = f["xCII_History"][:]
    xCO_raw     = f["xCO_History"][:]

    # O species
    xOI_raw     = f["xOI_History"][:]
    xOII_raw    = f["xOII_History"][:]

    # HD
    xHD_raw     = f["xHD_History"][:]

Tgas_hist   = reorder_history_3d(Tgas_raw,   t, r, Z)
xHII_hist   = reorder_history_3d(xHII_raw,   t, r, Z)
xH2_hist    = reorder_history_3d(xH2_raw,    t, r, Z)
xHeII_hist  = reorder_history_3d(xHeII_raw,  t, r, Z)
xHeIII_hist = reorder_history_3d(xHeIII_raw, t, r, Z)
xCI_hist    = reorder_history_3d(xCI_raw,    t, r, Z)
xCII_hist   = reorder_history_3d(xCII_raw,   t, r, Z)
xCO_hist    = reorder_history_3d(xCO_raw,    t, r, Z)
xOI_hist    = reorder_history_3d(xOI_raw,    t, r, Z)
xOII_hist   = reorder_history_3d(xOII_raw,   t, r, Z)
xHD_hist    = reorder_history_3d(xHD_raw,    t, r, Z)

k = nearest_saved_index(t, TargetTime_kyr * kyr)
print(f"Requested target time = {TargetTime_kyr:.3f} kyr")
print(f"Using snapshot at     = {t[k] / kyr:.3f} kyr")

Tgas   = Tgas_hist[k]
xHII   = xHII_hist[k]
xH2    = xH2_hist[k]
xHeII  = xHeII_hist[k]
xHeIII = xHeIII_hist[k]
xCI    = xCI_hist[k]
xCII   = xCII_hist[k]
xCO    = xCO_hist[k]
xOI    = xOI_hist[k]
xOII   = xOII_hist[k]
xHD    = xHD_hist[k]

# Derived neutral fractions:
#   xHI  = 1 - xHII - 2*xH2   (H nuclei not in HII or H2)
#   xHeI = 1 - xHeII - xHeIII
xHI  = np.clip(1.0 - xHII - 2.0 * xH2, 0.0, 1.0)
xHeI = np.clip(1.0 - xHeII - xHeIII,   0.0, 1.0)

R_edges_pc, Z_edges_pc = np.meshgrid(r / pc, Z / pc, indexing='ij')


# -------------------------------------------------------------------
# Norms
# -------------------------------------------------------------------

if AutoScale_Tgas:
    norm_Tgas = make_lognorm_from_arrays([Tgas], Tgas_vmin, Tgas_vmax,
                                          pct_low_log, pct_high_log)
else:
    norm_Tgas = LogNorm(vmin=Tgas_vmin, vmax=Tgas_vmax)

# Helper: build a LogNorm for one abundance field, respecting auto/manual flags.
def make_abund_norm(field, vmin, vmax):
    if AutoScale_Abundances:
        return make_lognorm_from_arrays([field], vmin, vmax, pct_low_log, pct_high_log)
    else:
        return LogNorm(vmin=vmin, vmax=vmax)

xHI_vmin_r,    xHI_vmax_r    = resolve_vmin_vmax(xHI_vmin,    xHI_vmax)
xH2_vmin_r,    xH2_vmax_r    = resolve_vmin_vmax(xH2_vmin,    xH2_vmax)
xHeI_vmin_r,   xHeI_vmax_r   = resolve_vmin_vmax(xHeI_vmin,   xHeI_vmax)
xHeII_vmin_r,  xHeII_vmax_r  = resolve_vmin_vmax(xHeII_vmin,  xHeII_vmax)
xHeIII_vmin_r, xHeIII_vmax_r = resolve_vmin_vmax(xHeIII_vmin, xHeIII_vmax)
xCI_vmin_r,    xCI_vmax_r    = resolve_vmin_vmax(xCI_vmin,    xCI_vmax)
xCII_vmin_r,   xCII_vmax_r   = resolve_vmin_vmax(xCII_vmin,   xCII_vmax)
xCO_vmin_r,    xCO_vmax_r    = resolve_vmin_vmax(xCO_vmin,    xCO_vmax)
xOI_vmin_r,    xOI_vmax_r    = resolve_vmin_vmax(xOI_vmin,    xOI_vmax)
xOII_vmin_r,   xOII_vmax_r   = resolve_vmin_vmax(xOII_vmin,   xOII_vmax)
xHD_vmin_r,    xHD_vmax_r    = resolve_vmin_vmax(xHD_vmin,    xHD_vmax)

norm_xHI    = make_abund_norm(xHI,    xHI_vmin_r,    xHI_vmax_r)
norm_xH2    = make_abund_norm(xH2,    xH2_vmin_r,    xH2_vmax_r)
norm_xHeI   = make_abund_norm(xHeI,   xHeI_vmin_r,   xHeI_vmax_r)
norm_xHeII  = make_abund_norm(xHeII,  xHeII_vmin_r,  xHeII_vmax_r)
norm_xHeIII = make_abund_norm(xHeIII, xHeIII_vmin_r, xHeIII_vmax_r)
norm_xCI    = make_abund_norm(xCI,    xCI_vmin_r,    xCI_vmax_r)
norm_xCII   = make_abund_norm(xCII,   xCII_vmin_r,   xCII_vmax_r)
norm_xCO    = make_abund_norm(xCO,    xCO_vmin_r,    xCO_vmax_r)
norm_xOI    = make_abund_norm(xOI,    xOI_vmin_r,    xOI_vmax_r)
norm_xOII   = make_abund_norm(xOII,   xOII_vmin_r,   xOII_vmax_r)
norm_xHD    = make_abund_norm(xHD,    xHD_vmin_r,    xHD_vmax_r)


# -------------------------------------------------------------------
# Clip fields for plotting
# -------------------------------------------------------------------

Tgas_plot   = np.clip(Tgas.T,   norm_Tgas.vmin,    norm_Tgas.vmax)

# Clip each abundance to its own norm range
def clip_abund(field, norm):
    return np.clip(field.T, norm.vmin, norm.vmax)

xHI_plot    = clip_abund(xHI,    norm_xHI)
xH2_plot    = clip_abund(xH2,    norm_xH2)
xHeI_plot   = clip_abund(xHeI,   norm_xHeI)
xHeII_plot  = clip_abund(xHeII,  norm_xHeII)
xHeIII_plot = clip_abund(xHeIII, norm_xHeIII)
xCI_plot    = clip_abund(xCI,    norm_xCI)
xCII_plot   = clip_abund(xCII,   norm_xCII)
xCO_plot    = clip_abund(xCO,    norm_xCO)
xOI_plot    = clip_abund(xOI,    norm_xOI)
xOII_plot   = clip_abund(xOII,   norm_xOII)
xHD_plot    = clip_abund(xHD,    norm_xHD)


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
#
# Row 1:  T (K),   xHI,    xH2
# Row 2:  xHeI,   xHeII,  xHeIII
# Row 3:  xCI,    xCII,   xCO
# Row 4:  xOI,    xOII,   xHD
# -------------------------------------------------------------------

im11 = ax11.pcolormesh(R_edges_pc, Z_edges_pc, Tgas_plot,
                        norm=norm_Tgas,    shading='auto',
                        cmap=cmap_Tgas,   rasterized=rasterized)
im12 = ax12.pcolormesh(R_edges_pc, Z_edges_pc, xHI_plot,
                        norm=norm_xHI,     shading='auto',
                        cmap=resolve_cmap(cmap_xHI),  rasterized=rasterized)
im13 = ax13.pcolormesh(R_edges_pc, Z_edges_pc, xH2_plot,
                        norm=norm_xH2,     shading='auto',
                        cmap=resolve_cmap(cmap_xH2),  rasterized=rasterized)

im21 = ax21.pcolormesh(R_edges_pc, Z_edges_pc, xHeI_plot,
                        norm=norm_xHeI,    shading='auto',
                        cmap=resolve_cmap(cmap_xHeI),   rasterized=rasterized)
im22 = ax22.pcolormesh(R_edges_pc, Z_edges_pc, xHeII_plot,
                        norm=norm_xHeII,   shading='auto',
                        cmap=resolve_cmap(cmap_xHeII),  rasterized=rasterized)
im23 = ax23.pcolormesh(R_edges_pc, Z_edges_pc, xHeIII_plot,
                        norm=norm_xHeIII,  shading='auto',
                        cmap=resolve_cmap(cmap_xHeIII), rasterized=rasterized)

im31 = ax31.pcolormesh(R_edges_pc, Z_edges_pc, xCI_plot,
                        norm=norm_xCI,     shading='auto',
                        cmap=resolve_cmap(cmap_xCI),  rasterized=rasterized)
im32 = ax32.pcolormesh(R_edges_pc, Z_edges_pc, xCII_plot,
                        norm=norm_xCII,    shading='auto',
                        cmap=resolve_cmap(cmap_xCII), rasterized=rasterized)
im33 = ax33.pcolormesh(R_edges_pc, Z_edges_pc, xCO_plot,
                        norm=norm_xCO,     shading='auto',
                        cmap=resolve_cmap(cmap_xCO),  rasterized=rasterized)

im41 = ax41.pcolormesh(R_edges_pc, Z_edges_pc, xOI_plot,
                        norm=norm_xOI,     shading='auto',
                        cmap=resolve_cmap(cmap_xOI),  rasterized=rasterized)
im42 = ax42.pcolormesh(R_edges_pc, Z_edges_pc, xOII_plot,
                        norm=norm_xOII,    shading='auto',
                        cmap=resolve_cmap(cmap_xOII), rasterized=rasterized)
im43 = ax43.pcolormesh(R_edges_pc, Z_edges_pc, xHD_plot,
                        norm=norm_xHD,     shading='auto',
                        cmap=resolve_cmap(cmap_xHD),  rasterized=rasterized)

ax12.set_title("Gas chemical abundances", y = 1.04, fontsize = 65)

# -------------------------------------------------------------------
# Contour overlays (drawn after pcolormesh so they sit on top)
# -------------------------------------------------------------------

# Map panel keys to (ax, raw_field) pairs.  Note: pass the un-transposed
# field — draw_contours handles the .T internally to match contour's (x,y,z).
_contour_panels = [
    ("ax11", ax11, Tgas),
    ("ax12", ax12, xHI),
    ("ax13", ax13, xH2),
    ("ax21", ax21, xHeI),
    ("ax22", ax22, xHeII),
    ("ax23", ax23, xHeIII),
    ("ax31", ax31, xCI),
    ("ax32", ax32, xCII),
    ("ax33", ax33, xCO),
    ("ax41", ax41, xOI),
    ("ax42", ax42, xOII),
    ("ax43", ax43, xHD),
]

for _key, _ax, _field in _contour_panels:
    draw_contours(_ax, _field, R_edges_pc, Z_edges_pc,
                  contour_config.get(_key))


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
if scale_bar_panel is not None and scale_bar_panel in _scale_bar_ax_map:
    draw_scale_bar(_scale_bar_ax_map[scale_bar_panel])

# Physics labels
panel_label(ax11, r"T (K)",                     panel_label_overrides.get("ax11"))
panel_label(ax12, r"$x_{HI}$",                  panel_label_overrides.get("ax12"))
panel_label(ax13, r"$x_{H_2}$",                 panel_label_overrides.get("ax13"))

panel_label(ax21, r"$x_{HeI}$",                 panel_label_overrides.get("ax21"))
panel_label(ax22, r"$x_{HeII}$",                panel_label_overrides.get("ax22"))
panel_label(ax23, r"$x_{HeIII}$",               panel_label_overrides.get("ax23"))

panel_label(ax31, r"$x_{CI}$",                  panel_label_overrides.get("ax31"))
panel_label(ax32, r"$x_{CII}$",                 panel_label_overrides.get("ax32"))
panel_label(ax33, r"$x_{CO}$",                  panel_label_overrides.get("ax33"))

panel_label(ax41, r"$x_{OI}$",                  panel_label_overrides.get("ax41"))
panel_label(ax42, r"$x_{OII}$",                 panel_label_overrides.get("ax42"))
panel_label(ax43, r"$x_{HD}$",                  panel_label_overrides.get("ax43"))

# Alphabetical sub-labels
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

# Temperature
ticks_Tgas, ticklabels_Tgas = get_log_ticks_from_norm(norm_Tgas)
ticks_Tgas, ticklabels_Tgas = use_custom_ticks(
    ticks_Tgas, ticklabels_Tgas, ticks_Tgas_custom, ticklabels_Tgas_custom
)
add_inset_colorbar(ax11, im11, ticks_Tgas, ticklabels_Tgas,
                   tick_color=cbar_tick_color_Tgas,
                   outline_color=cbar_outline_color_Tgas,
                   geom_overrides=cbar_geom_overrides.get("ax11", {}))

# Abundance colorbars — all log-scale; resolve ticks and colors per species
def make_abund_cbar(ax, im, norm, species_ticks, species_ticklabels,
                    species_tick_color, species_outline_color,
                    geom_overrides=None):
    ticks_auto, labels_auto = get_log_ticks_from_norm(norm)
    ticks, labels = use_custom_ticks(ticks_auto, labels_auto, species_ticks, species_ticklabels)
    tc, oc = resolve_cbar_colors(species_tick_color, species_outline_color)
    add_inset_colorbar(ax, im, ticks, labels, tick_color=tc, outline_color=oc,
                       geom_overrides=geom_overrides)

make_abund_cbar(ax12, im12, norm_xHI,
                *resolve_ticks(ticks_xHI_custom, ticklabels_xHI_custom),
                cbar_tick_color_xHI, cbar_outline_color_xHI,
                geom_overrides=cbar_geom_overrides.get("ax12", {}))

make_abund_cbar(ax13, im13, norm_xH2,
                *resolve_ticks(ticks_xH2_custom, ticklabels_xH2_custom),
                cbar_tick_color_xH2, cbar_outline_color_xH2,
                geom_overrides=cbar_geom_overrides.get("ax13", {}))

make_abund_cbar(ax21, im21, norm_xHeI,
                *resolve_ticks(ticks_xHeI_custom, ticklabels_xHeI_custom),
                cbar_tick_color_xHeI, cbar_outline_color_xHeI,
                geom_overrides=cbar_geom_overrides.get("ax21", {}))

make_abund_cbar(ax22, im22, norm_xHeII,
                *resolve_ticks(ticks_xHeII_custom, ticklabels_xHeII_custom),
                cbar_tick_color_xHeII, cbar_outline_color_xHeII,
                geom_overrides=cbar_geom_overrides.get("ax22", {}))

make_abund_cbar(ax23, im23, norm_xHeIII,
                *resolve_ticks(ticks_xHeIII_custom, ticklabels_xHeIII_custom),
                cbar_tick_color_xHeIII, cbar_outline_color_xHeIII,
                geom_overrides=cbar_geom_overrides.get("ax23", {}))

make_abund_cbar(ax31, im31, norm_xCI,
                *resolve_ticks(ticks_xCI_custom, ticklabels_xCI_custom),
                cbar_tick_color_xCI, cbar_outline_color_xCI,
                geom_overrides=cbar_geom_overrides.get("ax31", {}))

make_abund_cbar(ax32, im32, norm_xCII,
                *resolve_ticks(ticks_xCII_custom, ticklabels_xCII_custom),
                cbar_tick_color_xCII, cbar_outline_color_xCII,
                geom_overrides=cbar_geom_overrides.get("ax32", {}))

make_abund_cbar(ax33, im33, norm_xCO,
                *resolve_ticks(ticks_xCO_custom, ticklabels_xCO_custom),
                cbar_tick_color_xCO, cbar_outline_color_xCO,
                geom_overrides=cbar_geom_overrides.get("ax33", {}))

make_abund_cbar(ax41, im41, norm_xOI,
                *resolve_ticks(ticks_xOI_custom, ticklabels_xOI_custom),
                cbar_tick_color_xOI, cbar_outline_color_xOI,
                geom_overrides=cbar_geom_overrides.get("ax41", {}))

make_abund_cbar(ax42, im42, norm_xOII,
                *resolve_ticks(ticks_xOII_custom, ticklabels_xOII_custom),
                cbar_tick_color_xOII, cbar_outline_color_xOII,
                geom_overrides=cbar_geom_overrides.get("ax42", {}))

make_abund_cbar(ax43, im43, norm_xHD,
                *resolve_ticks(ticks_xHD_custom, ticklabels_xHD_custom),
                cbar_tick_color_xHD, cbar_outline_color_xHD,
                geom_overrides=cbar_geom_overrides.get("ax43", {}))


# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------

plt.savefig(FigureName, bbox_inches='tight')
print(f"Saved figure to {FigureName}")