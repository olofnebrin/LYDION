import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
cmastro_src = repo_root / "src" / "cmastro" / "src"

sys.path.insert(0, str(cmastro_src))

import cmastro


# ---------------- User inputs ----------------
Filename         = "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5"
FigureName       = "LYDION_Figure_DestructionProbability_512D0.01.pdf"
TargetTime_kyr   = 15.0

# Figure/layout options
figsize          = (12, 12)
xmax_fraction    = 0.6
zmax_fraction    = 0.6
rasterized       = True

# Text sizes
fontsize_cbar_ticks  = 35
fontsize_panel_label = 60
fontsize_time        = 50
abc_label_fontsize   = 50

print_abc_label      = False # Set to True to show the "(a)" label for this panel. Adjust abc_label_text and abc_label_fontsize as needed.

# -------------------------------------------------------------------
# Colormap
# -------------------------------------------------------------------
#cmap_pd = "cma:emph"
cmap_pd = "cubehelix_r"

# -------------------------------------------------------------------
# Auto-scaling options
# -------------------------------------------------------------------
AutoScale_pd = False

# -------------------------------------------------------------------
# Manual range for Lyα destruction probability
# -------------------------------------------------------------------
pd_vmin, pd_vmax = 1e-14, 1e-6

# Percentiles used when AutoScale_pd = True
pct_low_log   = 1.0
pct_high_log  = 99.0

# -------------------------------------------------------------------
# Colorbar options (in axis coordinates)
# -------------------------------------------------------------------
cbar_x      = 0.82
cbar_y      = 0.45
cbar_width  = 0.040
cbar_height = 0.50

# Turn off tick marks, but keep tick labels
cbar_tick_length = 0
cbar_tick_width  = 0
cbar_tick_pad    = 4

cbar_tick_color    = "black"
cbar_outline_color = "black"

# Custom colorbar ticks/labels. Set both to None for automatic decade ticks.
# Example:
# ticks_pd_custom      = [1e-11, 1e-9, 1e-7, 1e-5]
# ticklabels_pd_custom = [r"$10^{-11}$", r"$10^{-9}$", r"$10^{-7}$", r"$10^{-5}$"]
ticks_pd_custom      = None
ticklabels_pd_custom = None

# -------------------------------------------------------------------
# Time label color/outline
# -------------------------------------------------------------------
time_label_color        = "black"
time_label_stroke       = "white"
time_label_stroke_width = 3.0

# -------------------------------------------------------------------
# Panel label position and style
# -------------------------------------------------------------------
panel_label_x            = 0.96
panel_label_y            = 0.04
panel_label_ha           = "right"
panel_label_va           = "bottom"
panel_label_color        = "white"
panel_label_stroke       = "black"
panel_label_stroke_width = 4.5

# -------------------------------------------------------------------
# Alphabetical sub-label
# -------------------------------------------------------------------
show_abc_label         = True
abc_label_x            = 0.75
abc_label_y            = 0.25
abc_label_ha           = "left"
abc_label_va           = "top"
abc_label_color        = "white"
abc_label_stroke       = "black"
abc_label_stroke_width = 3.5
abc_label_text         = "a"

# -------------------------------------------------------------------
# Panel frame
# -------------------------------------------------------------------
panel_frame_linewidth = 3.0

# -------------------------------------------------------------------
# Contour overlays
# Each entry in contour_config is a list of dicts, one per contour group.
# Keys per group:
#   levels           : list of values to contour              (required)
#   colors           : colour string or list of strings       (default: 'white')
#   linewidths       : float or list                          (default: 1.5)
#   linestyle        : 'solid', 'dashed', 'dotted', etc.     (default: 'solid')
#   rounded          : bool — round cap/join style            (default: False)
#   labels           : list of strings, one per level, or None (default: None)
#   label_fontsize   : float                                  (default: contour_label_fontsize_default)
#   label_color      : colour string                          (default: same as colors)
#
# Example:
#   contour_config = [
#       dict(levels=[1e-10, 1e-8], colors='white', linestyle='dashed',
#            rounded=True, linewidths=2.5,
#            labels=[r'$10^{-10}$', r'$10^{-8}$'], label_fontsize=30),
#   ]
# -------------------------------------------------------------------
contour_label_fontsize_default = 40
contour_config = None   # set to a list of dicts to enable

#contour_config = [
#    dict(levels=[4e-9],
#         colors='white', linestyle='dashed', rounded=True, linewidths=2.5,
#         labels=[r'$3 \times 10^{-9}$'], label_fontsize=30),
#]

# -------------------------------------------------------------------
# Scale bar
# scale_bar_length  : physical length in pc
# scale_bar_label   : text shown above bar (LaTeX ok)
# scale_bar_x       : centre of bar in data (pc); None → horizontally centred
# scale_bar_y       : bottom of bar in data (pc); None → 10 % up from bottom
# scale_bar_capsize : height of "|" caps in pc; None → 4 % of Z span
# scale_bar_label_pad : gap between cap top and label bottom; None → 2 % of Z span
# scale_bar_fontsize  : None → fontsize_panel_label
# -------------------------------------------------------------------
show_scale_bar         = True
scale_bar_length       = 0.2
scale_bar_label        = "0.2 pc"
scale_bar_x            = 0.47
scale_bar_y            = 0.05
scale_bar_color        = 'black'
scale_bar_lw           = 3.5
scale_bar_capsize      = None
scale_bar_label_pad    = None
scale_bar_fontsize     = None
scale_bar_stroke       = 'white'
scale_bar_stroke_width = 3.0

# -------------------------------------------------------------------
# Useful constants
# -------------------------------------------------------------------
yr   = 3.1556926e7
kyr  = 1e3 * yr
pc   = 3.085677581e18


###############################################################
#######                                                 #######
#######            P L O T  C O S M E T I C S           #######
#######                                                 #######
###############################################################

plt.rc("font", **{"sans-serif": ["AVHershey Complex"], "size": 39.5})
params = {"mathtext.default": "regular"}
plt.rcParams.update(params)
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "GreekC"

params = {"legend.fontsize": 19}
plt.rcParams.update(params)
plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.major.width"] = 1.7
plt.rcParams["xtick.minor.width"] = 1.7
plt.rcParams["xtick.major.size"] = 20
plt.rcParams["xtick.minor.size"] = 8
plt.rcParams["ytick.major.width"] = 1.7
plt.rcParams["ytick.minor.width"] = 1.7
plt.rcParams["ytick.major.size"] = 20
plt.rcParams["ytick.minor.size"] = 8
plt.rcParams["xtick.major.pad"] = "15"
plt.rcParams["ytick.major.pad"] = "15"

plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.linewidth"] = 2.5
plt.rcParams["grid.linewidth"] = 2.5
plt.rcParams["grid.linestyle"] = "--"

plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


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


def add_inset_colorbar(ax, mappable, ticks=None, ticklabels=None,
                       tick_color=None, outline_color=None):
    if tick_color is None:
        tick_color = "black"
    if outline_color is None:
        outline_color = "black"

    cax = ax.inset_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    cb = plt.colorbar(mappable, cax=cax, orientation="vertical")

    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)

    cb.ax.tick_params(
        labelsize=fontsize_cbar_ticks,
        width=cbar_tick_width,
        length=cbar_tick_length,
        pad=cbar_tick_pad,
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


def panel_label(ax, text):
    style = dict(
        ha=panel_label_ha,
        va=panel_label_va,
        color=panel_label_color,
        fontsize=fontsize_panel_label,
        path_effects=[pe.withStroke(linewidth=panel_label_stroke_width,
                                    foreground=panel_label_stroke)]
    )
    ax.text(panel_label_x, panel_label_y, text, transform=ax.transAxes, **style)


def abc_label(ax, letter):
    style = dict(
        ha=abc_label_ha,
        va=abc_label_va,
        color=abc_label_color,
        fontsize=abc_label_fontsize,
        path_effects=[pe.withStroke(linewidth=abc_label_stroke_width,
                                    foreground=abc_label_stroke)]
    )
    if print_abc_label:
        ax.text(abc_label_x, abc_label_y, f"({letter})", transform=ax.transAxes, **style)


def time_label(ax, time_kyr):
    style = dict(
        ha="left",
        va="top",
        color=time_label_color,
        fontsize=fontsize_time,
        path_effects=[pe.withStroke(linewidth=time_label_stroke_width,
                                    foreground=time_label_stroke)]
    )
    ax.text(0.04, 0.96, rf"t = {time_kyr:.2f} kyr", transform=ax.transAxes, **style)


def draw_contours(ax, field_2d, R_edges_pc, Z_edges_pc, config_list):
    """Overlay |---| contour lines on the panel."""
    if config_list is None:
        return

    R_cen = 0.5 * (R_edges_pc[:-1, :-1] + R_edges_pc[1:, :-1])
    Z_cen = 0.5 * (Z_edges_pc[:-1, :-1] + Z_edges_pc[:-1, 1:])

    for group in config_list:
        levels         = group['levels']
        colors         = group.get('colors',         'white')
        linewidths     = group.get('linewidths',     1.5)
        linestyle      = group.get('linestyle',      'solid')
        rounded        = group.get('rounded',        False)
        labels         = group.get('labels',         None)
        label_fontsize = group.get('label_fontsize', contour_label_fontsize_default)
        label_fmt      = group.get('label_fmt',      None)
        _default_lc    = colors[0] if isinstance(colors, (list, tuple)) else colors
        label_color    = group.get('label_color',    _default_lc)

        cs = ax.contour(R_cen, Z_cen, field_2d.T,
                        levels=levels, colors=colors,
                        linewidths=linewidths, linestyles=linestyle)

        if rounded:
            if hasattr(cs, 'collections'):      # matplotlib < 3.8
                for lc in cs.collections:
                    lc.set_capstyle('round')
                    lc.set_joinstyle('round')
            else:                               # matplotlib >= 3.10
                cs.set_capstyle('round')
                cs.set_joinstyle('round')

        if labels is not None:
            fmt_dict = {lv: lb for lv, lb in zip(levels, labels)}
            clbls = ax.clabel(cs, levels=levels, fmt=fmt_dict,
                              fontsize=label_fontsize, inline=True)
            for txt in clbls:
                txt.set_color(label_color)
        elif label_fmt is not None:
            clbls = ax.clabel(cs, levels=levels, fmt=label_fmt,
                              fontsize=label_fontsize, inline=True)
            for txt in clbls:
                txt.set_color(label_color)


def draw_scale_bar(ax):
    """Draw a |---| scale bar with label above it."""
    if not show_scale_bar:
        return

    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    xspan = xhi - xlo
    yspan = yhi - ylo

    bar_len  = scale_bar_length
    bar_x    = scale_bar_x       if scale_bar_x       is not None else xlo + 0.5 * xspan
    bar_y    = scale_bar_y       if scale_bar_y       is not None else ylo + 0.10 * yspan
    capsize  = scale_bar_capsize if scale_bar_capsize  is not None else 0.04 * yspan
    lpad     = scale_bar_label_pad if scale_bar_label_pad is not None else 0.02 * yspan
    fontsize = scale_bar_fontsize  if scale_bar_fontsize  is not None else fontsize_panel_label
    color    = scale_bar_color

    x0 = bar_x - 0.5 * bar_len
    x1 = bar_x + 0.5 * bar_len

    text_effects = ([pe.withStroke(linewidth=scale_bar_stroke_width,
                                   foreground=scale_bar_stroke)]
                    if scale_bar_stroke is not None else [])

    ax.plot([x0, x1], [bar_y, bar_y],
            color=color, lw=scale_bar_lw, solid_capstyle='butt',
            transform=ax.transData, zorder=10)
    ax.plot([x0, x0], [bar_y - 0.5*capsize, bar_y + 0.5*capsize],
            color=color, lw=scale_bar_lw, solid_capstyle='round',
            transform=ax.transData, zorder=10)
    ax.plot([x1, x1], [bar_y - 0.5*capsize, bar_y + 0.5*capsize],
            color=color, lw=scale_bar_lw, solid_capstyle='round',
            transform=ax.transData, zorder=10)
    ax.text(bar_x, bar_y + 0.5*capsize + lpad, scale_bar_label,
            color=color, fontsize=fontsize, ha='center', va='bottom',
            transform=ax.transData, zorder=11, path_effects=text_effects)


# -------------------------------------------------------------------
# Read data
# -------------------------------------------------------------------

with h5py.File(Filename, "r") as f:
    t = f["Times"][:]
    r = f["r"][:]
    Z = f["Z"][:]

    pd_raw = f["pd_History"][:]

pd_hist = reorder_history_3d(pd_raw, t, r, Z)

k = nearest_saved_index(t, TargetTime_kyr * kyr)
print(f"Requested target time = {TargetTime_kyr:.3f} kyr")
print(f"Using snapshot at     = {t[k] / kyr:.3f} kyr")

pd = pd_hist[k]

R_edges_pc, Z_edges_pc = np.meshgrid(r / pc, Z / pc, indexing="ij")


# -------------------------------------------------------------------
# Norm
# -------------------------------------------------------------------

if AutoScale_pd:
    norm_pd = make_lognorm_from_arrays([pd], pd_vmin, pd_vmax,
                                        pct_low_log, pct_high_log)
else:
    norm_pd = LogNorm(vmin=pd_vmin, vmax=pd_vmax)


# -------------------------------------------------------------------
# Clip field for plotting
# -------------------------------------------------------------------

pd_plot = np.clip(pd.T, norm_pd.vmin, norm_pd.vmax)


# -------------------------------------------------------------------
# Set up figure: 1 panel
# -------------------------------------------------------------------

fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_title(r"Ly$\alpha$ destruction probability $p_d$", 
              y = 1.03, fontsize = 65)  


# -------------------------------------------------------------------
# Plot panel
# -------------------------------------------------------------------

im = ax.pcolormesh(R_edges_pc, Z_edges_pc, pd_plot,
                   norm=norm_pd, shading="auto",
                   cmap=cmap_pd, rasterized=rasterized)

ax.text(0.03, 0.1, "H II region", fontsize = 60, color = "white")

# -------------------------------------------------------------------
# Panel cosmetics
# -------------------------------------------------------------------

ax.set_xlim(r[0] / pc, xmax_fraction * r[-1] / pc)
ax.set_ylim(Z[0] / pc, zmax_fraction * Z[-1] / pc)
ax.set_aspect("equal", adjustable="box")

ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.tick_params(which="both", bottom=False, top=False, left=False, right=False,
               labelbottom=False, labelleft=False)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(panel_frame_linewidth)

draw_contours(ax, pd, R_edges_pc, Z_edges_pc, contour_config)

time_kyr_actual = t[k] / kyr
time_label(ax, time_kyr_actual)

draw_scale_bar(ax)

#panel_label(ax, r"Ly$\alpha$ destruction probability $p_d$")

if show_abc_label:
    abc_label(ax, abc_label_text)


# -------------------------------------------------------------------
# Colorbar
# -------------------------------------------------------------------

ticks_pd, ticklabels_pd = get_log_ticks_from_norm(norm_pd)
ticks_pd, ticklabels_pd = use_custom_ticks(
    ticks_pd, ticklabels_pd, ticks_pd_custom, ticklabels_pd_custom
)

add_inset_colorbar(ax, im, ticks_pd, ticklabels_pd,
                   tick_color=cbar_tick_color,
                   outline_color=cbar_outline_color)


# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------

plt.savefig(FigureName, bbox_inches="tight")
print(f"Saved figure to {FigureName}")