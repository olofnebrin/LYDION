import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.patheffects as pe
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
cmastro_src = repo_root / "src" / "cmastro" / "src"
sys.path.insert(0, str(cmastro_src))

import cmastro


# ===================================================================
# User inputs
# ===================================================================

Filename       = "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5"
FigureName     = "LYDION_Figure_EddingtonRatio_512D0.01.pdf"
TargetTime_kyr_1 = 15.0
TargetTime_kyr_2 = 76.0

# -------------------------------------------------------------------
# Figure / layout
# -------------------------------------------------------------------
show_gas_pressure_fEdd = False    # add a third row for gas-pressure Eddington ratios

figsize        = (52, 34) if show_gas_pressure_fEdd else (52, 24)
xmax_fraction  = 0.6
zmax_fraction  = 0.6
rasterized     = True

# Column widths and row heights as fractions of figure size.
# The left 2-column block shows TargetTime_kyr_1, and the right 2-column
# block shows TargetTime_kyr_2.  If show_gas_pressure_fEdd=True, a third
# row is added below Lyα and direct radiation pressure.
col_left_0    = 0.02            # left edge of column 0
col_gap       = 0.00            # gap between panels within each time block
group_gap     = 0.02            # gap between the two time blocks
col_width     = 0.20            # width of each of the 4 panel columns
row_bottom_0  = 0.045 if show_gas_pressure_fEdd else 0.06
row_gap       = -0.005 if show_gas_pressure_fEdd else -0.01
row_height    = 0.300 if show_gas_pressure_fEdd else 0.44

# -------------------------------------------------------------------
# Text sizes
# -------------------------------------------------------------------
fontsize_cbar_ticks  = 40
fontsize_panel_label = 65
fontsize_row_label   = 65       # row labels
fontsize_time        = 60
abc_label_fontsize   = 50

# -------------------------------------------------------------------
# Colormap — shared by all 6 panels.
# The Eddington ratio can be ≫1 (force > gravity) or ≪1 (gravity wins).
# We use a symmetric log norm so that f_Edd = 1 is a natural midpoint.
# -------------------------------------------------------------------
cmap_fEdd    = "RdBu_r"        # diverging: blue < 1, red > 1
#cmap_fEdd = "BuPu_r"          # diverging: blue < 1, red > 1, yellow near 1
#cmap_fEdd = "YlGnBu_r" 

# Colour scale: symmetric log around 1.
# All panels share the same norm, built from these limits.
fEdd_vmin      = 1e-2           # lower bound
fEdd_vmax      = 1e2            # upper bound  (same magnitude above 1)
fEdd_linthresh = 1e-4           # linear region half-width around 0 for SymLogNorm
                                # (only relevant if you switch to SymLogNorm)

# Use plain LogNorm (ratio is always positive for naive/anti-grav/radial in
# the dominant-outward case).  Switch to SymLogNorm if you want signed radial.
use_symlognorm = False          # True → SymLogNorm (shows negative values)

# -------------------------------------------------------------------
# Colorbar (inset, shared position — defined relative to each panel)
# -------------------------------------------------------------------
cbar_x       = 0.80
cbar_y       = 0.10
cbar_width   = 0.040
cbar_height  = 0.45

cbar_tick_length = 0
cbar_tick_width  = 0
cbar_tick_pad    = 4

cbar_tick_color    = "white"
cbar_outline_color = "white"

# Custom ticks/labels.  Set both to None for automatic decade ticks.
ticks_fEdd_custom      = None
ticklabels_fEdd_custom = None

# -------------------------------------------------------------------
# Panel labels (physics label, inside each panel)
# Block 1 = left 2x2 time block, Block 2 = right 2x2 time block.
# You can use different colours/strokes for block 2 if the background differs.
# -------------------------------------------------------------------
panel_label_x             = 0.96
panel_label_y             = 0.86
panel_label_ha            = "right"
panel_label_va            = "bottom"
panel_label_stroke_width  = 4.5

show_panel_labels_block_1 = True
show_panel_labels_block_2 = True

# Left time block (block 1)
panel_label_color_default_1 = "white"
panel_label_colors_1 = {          # per-panel overrides (None → use default)
    "ax_lya_naive" : None,
    "ax_lya_rad"   : None,
    "ax_lya_anti"  : None,
    "ax_dir_naive" : "black",
    "ax_dir_rad"   : "black",
    "ax_dir_anti"  : "black",
    "ax_gas_naive" : "black",
    "ax_gas_anti"  : "black",
}
panel_label_stroke_default_1 = "black"
panel_label_strokes_1 = {         # per-panel overrides (None → use default)
    "ax_lya_naive" : None,
    "ax_lya_rad"   : None,
    "ax_lya_anti"  : None,
    "ax_dir_naive" : "white",
    "ax_dir_rad"   : "white",
    "ax_dir_anti"  : "white",
    "ax_gas_naive" : "white",
    "ax_gas_anti"  : "white",
}

# Right time block (block 2)
panel_label_color_default_2 = "white"
panel_label_colors_2 = {          # set explicitly if the later-time backgrounds differ
    "ax_lya_naive" : "white",
    "ax_lya_rad"   : "white",
    "ax_lya_anti"  : "white",
    "ax_dir_naive" : "black",
    "ax_dir_rad"   : "black",
    "ax_dir_anti"  : "black",
    "ax_gas_naive" : "black",
    "ax_gas_anti"  : "black",
}
panel_label_stroke_default_2 = "black"
panel_label_strokes_2 = {
    "ax_lya_naive" : "black",
    "ax_lya_rad"   : "black",
    "ax_lya_anti"  : "black",
    "ax_dir_naive" : "white",
    "ax_dir_rad"   : "white",
    "ax_dir_anti"  : "white",
    "ax_gas_naive" : "white",
    "ax_gas_anti"  : "white",
}

# Row labels (left margin, centred on each row)
show_row_labels  = True
row_label_x      = 0.004         # figure x (0–1); smaller → farther left from panels
row_label_ha     = "left"        # horizontal alignment of the rotated row labels
row_label_color  = "black"
row_label_Lya    = r"Ly$\alpha$ radiation pressure"
row_label_dir    = r"Direct radiation pressure"
row_label_gas    = r"Gas pressure"

# -------------------------------------------------------------------
# Panel frame
# -------------------------------------------------------------------
panel_frame_linewidth = 4.0

# -------------------------------------------------------------------
# Time label
# -------------------------------------------------------------------
time_label_color        = "white"     # fallback/default colour
time_label_color_1      = "white"     # first time block
time_label_color_2      = "black"     # second time block; change e.g. to "black" if needed
time_label_stroke       = "black"
time_label_stroke_1     = "black"
time_label_stroke_2     = "white"     # change e.g. to "white" if time_label_color_2 = "black"
time_label_stroke_width = 3.0

# -------------------------------------------------------------------
# Contour overlays  (same format as GasAbundances figure)
# contour_config[key] = list of group dicts, or None.
# Keys: "ax_lya_naive", "ax_lya_rad", "ax_lya_anti",
#       "ax_dir_naive", "ax_dir_rad", "ax_dir_anti"
# -------------------------------------------------------------------
contour_label_fontsize_default = 30
contour_config = {
    "ax_lya_naive" : None,
    "ax_lya_rad"   : None,
    "ax_lya_anti"  : None,
    "ax_dir_naive" : None,
    "ax_dir_rad"   : None,
    "ax_dir_anti"  : None,
    "ax_gas_naive" : None,
    "ax_gas_anti"  : None,
}

# -------------------------------------------------------------------
# Scale bar
# -------------------------------------------------------------------
show_scale_bar         = True
scale_bar_length       = 0.2
scale_bar_label        = "0.2 pc"
scale_bar_panel        = "ax_lya_naive"   # which panel to place it on
scale_bar_x            = 0.55
scale_bar_y            = None
scale_bar_color        = "white"
scale_bar_lw           = 3.0
scale_bar_capsize      = None
scale_bar_label_pad    = None
scale_bar_fontsize     = None
scale_bar_stroke       = "black"
scale_bar_stroke_width = 3.0

# -------------------------------------------------------------------
# Inset zoom — single f_Edd panel
#
# inset_panel      : which main panel the zoom rectangle + connectors
#                    are drawn on (should be the f_Edd panel you zoom)
# inset_region     : [r_min, r_max, Z_min, Z_max] in pc
# inset_fig_loc    : [left, bottom, width, height] in FIGURE coordinates
#                    (0–1).  This is the position of the single inset.
# inset_connector_color / _lw : style of zoom rectangle + connectors
# inset_frame_color / _lw     : style of the inset frame
#
# The inset shows the same f_Edd field and colour scale as the parent panel.
# -------------------------------------------------------------------
show_inset            = True
inset_panel           = "ax_lya_anti"
inset_region          = [0.25, 0.35, 0.02, 0.15]  # [r_min, r_max, Z_min, Z_max] pc

# Connector corners:
# 1 = upper right, 2 = upper left, 3 = lower left, 4 = lower right
# For your current layout, this connects:
# rectangle upper-right/lower-right → inset upper-left/lower-left
inset_parent_corners  = (2, 4)
inset_box_corners     = (2, 4)

# Single inset position in figure coordinates
inset_fig_loc_base    = [0.58, 0.63, 0.20, 0.7*0.28]  # original 2x2 [left, bottom, width, height]
inset_fig_loc         = [0.5*inset_fig_loc_base[0],
                         inset_fig_loc_base[1],
                         0.5*inset_fig_loc_base[2],
                         inset_fig_loc_base[3]]      # same physical/in-panel placement after doubling width

inset_connector_color = "black"
inset_connector_lw    = 3.5
inset_frame_color     = "black"
inset_frame_lw        = 3.5

# Inset label
show_inset_label      = False
inset_label_fontsize  = 35             # None → fontsize_panel_label
inset_label_color     = "white"        # None → panel_label_color_default
inset_label_stroke    = "black"        # None → panel_label_stroke_default

# Small colorbar drawn inside the inset
inset_show_cbar       = False
inset_cbar_x          = 0.72
inset_cbar_y          = 0.08
inset_cbar_width      = 0.06
inset_cbar_height     = 0.40
inset_cbar_ticksize   = 22

# -------------------------------------------------------------------
# Masking: cells where gravity or radiation force is negligibly small
# are set to NaN and rendered as masked (transparent / background colour).
# Set each threshold to the smallest physically meaningful value for
# your simulation — anything below is treated as "no force here".
# Units: dyn cm⁻³  (same as the HDF5 force fields)
# A reasonable starting point is ~1e-3 × the typical shell peak value.
# -------------------------------------------------------------------
f_grav_mask_threshold = 0.0      # mask cells where |f_grav| <= this
f_rad_mask_threshold  = 0.0      # mask cells where |f_rad|  <= this
                                  # set both to 0.0 to mask only exact zeros
mask_color            = "grey"   # background colour for masked cells

# -------------------------------------------------------------------
# Useful constants
# -------------------------------------------------------------------
yr  = 3.1556926e7
kyr = 1e3 * yr
pc  = 3.085677581e18


###############################################################
#######                                               #######
#######          P L O T  C O S M E T I C S          #######
#######                                               #######
###############################################################

plt.rc("font", **{"sans-serif": ["AVHershey Complex"], "size": 39.5})
params = {"mathtext.default": "regular"}
plt.rcParams.update(params)
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"]      = "GreekC"

plt.rcParams["axes.unicode_minus"]  = False
plt.rcParams["xtick.major.width"]   = 1.7
plt.rcParams["xtick.minor.width"]   = 1.7
plt.rcParams["xtick.major.size"]    = 20
plt.rcParams["xtick.minor.size"]    = 8
plt.rcParams["ytick.major.width"]   = 1.7
plt.rcParams["ytick.minor.width"]   = 1.7
plt.rcParams["ytick.major.size"]    = 20
plt.rcParams["ytick.minor.size"]    = 8
plt.rcParams["xtick.major.pad"]     = "15"
plt.rcParams["ytick.major.pad"]     = "15"
plt.rcParams["legend.frameon"]      = False
plt.rcParams["axes.linewidth"]      = 2.5
plt.rcParams["grid.linewidth"]      = 2.5
plt.rcParams["grid.linestyle"]      = "--"
plt.rcParams["xtick.top"]           = True
plt.rcParams["xtick.bottom"]        = True
plt.rcParams["ytick.left"]          = True
plt.rcParams["ytick.right"]         = True
plt.rcParams["xtick.direction"]     = "in"
plt.rcParams["ytick.direction"]     = "in"


###############################################################
#######                                                 #######
#######          H E L P E R  F U N C T I O N S         #######
#######                                                 #######
###############################################################

def reorder_history_3d(arr, t, r_edges, Z_edges):
    """Reorder a 3D history array to shape (Nt, Nr, Nz)."""
    if arr is None:
        return None
    Nt = t.size
    Nr = len(r_edges) - 1
    Nz = len(Z_edges) - 1
    shape = arr.shape
    time_axes = [ax for ax, s in enumerate(shape) if s == Nt]
    if len(time_axes) != 1:
        raise RuntimeError(f"Could not uniquely identify time axis: shape={shape}, Nt={Nt}")
    time_axis = time_axes[0]
    arr_tfirst = np.moveaxis(arr, time_axis, 0)
    _, a1, a2 = arr_tfirst.shape
    if (a1, a2) == (Nr, Nz):
        return arr_tfirst
    elif (a1, a2) == (Nz, Nr):
        return arr_tfirst.transpose(0, 2, 1)
    else:
        raise RuntimeError(f"Spatial dims {(a1,a2)} don't match Nr={Nr}, Nz={Nz}")


def nearest_saved_index(t, target_time):
    saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]
    if saved_idx.size == 0:
        raise RuntimeError("No saved snapshots found.")
    return saved_idx[np.argmin(np.abs(t[saved_idx] - target_time))]


def fmt_log_tick(val):
    exp = int(np.round(np.log10(abs(val))))
    if exp == -1: return r"$0.1$"
    if exp ==  0: return r"$1$"
    if exp ==  1: return r"$10$"
    return rf"$10^{{{exp}}}$"


def get_log_ticks_from_norm(norm):
    logmin = int(np.ceil(np.log10(norm.vmin)))
    logmax = int(np.floor(np.log10(norm.vmax)))
    if logmax < logmin:
        return None, None
    ticks = 10.0**np.arange(logmin, logmax + 1)
    return ticks, [fmt_log_tick(v) for v in ticks]


def use_custom_ticks(default_ticks, default_labels, custom_ticks, custom_labels):
    ticks  = default_ticks  if custom_ticks  is None else custom_ticks
    labels = default_labels if custom_labels is None else custom_labels
    if custom_ticks is not None and custom_labels is None:
        labels = None
    return ticks, labels


def add_inset_colorbar(ax, mappable, ticks=None, ticklabels=None,
                       tick_color=None, outline_color=None):
    if tick_color   is None: tick_color   = cbar_tick_color
    if outline_color is None: outline_color = cbar_outline_color

    cax = ax.inset_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    cb  = plt.colorbar(mappable, cax=cax, orientation="vertical")

    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)

    cb.ax.tick_params(labelsize=fontsize_cbar_ticks,
                      width=cbar_tick_width, length=cbar_tick_length,
                      pad=cbar_tick_pad, colors=tick_color, labelcolor=tick_color)
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


def panel_label(ax, text, color=None, stroke=None):
    _color  = color  if color  is not None else panel_label_color_default
    _stroke = stroke if stroke is not None else panel_label_stroke_default
    ax.text(panel_label_x, panel_label_y, text,
            transform=ax.transAxes,
            ha=panel_label_ha, va=panel_label_va,
            color=_color, fontsize=fontsize_panel_label,
            path_effects=[pe.withStroke(linewidth=panel_label_stroke_width,
                                        foreground=_stroke)])


def time_label(ax, time_kyr, color=None, stroke=None):
    _color  = time_label_color  if color  is None else color
    _stroke = time_label_stroke if stroke is None else stroke
    ax.text(0.04, 0.96, rf"t = {time_kyr:.2f} kyr",
            transform=ax.transAxes, ha="left", va="top",
            color=_color, fontsize=fontsize_time,
            path_effects=[pe.withStroke(linewidth=time_label_stroke_width,
                                        foreground=_stroke)])


def draw_inset(ax, field_plot, norm, cmap_obj):
    """Draw a single f_Edd inset zoom."""
    if not show_inset:
        return

    from matplotlib.patches import Rectangle, ConnectionPatch

    r_min, r_max, Z_min, Z_max = inset_region
    fl, fb, fw, fh = inset_fig_loc   # user-specified position in fig coords

    # ── Build single inset axes ───────────────────────────────────────────
    axins = fig.add_axes([fl, fb, fw, fh])

    # ── Plot the same f_Edd field as the parent panel ─────────────────────
    kw = dict(shading="auto", rasterized=rasterized)
    im = axins.pcolormesh(R_edges_pc, Z_edges_pc, field_plot.T,
                          norm=norm, cmap=cmap_obj, **kw)

    # ── Zoom to the inset region ──────────────────────────────────────────
    axins.set_xlim(r_min, r_max)
    axins.set_ylim(Z_min, Z_max)
    axins.set_aspect("equal", adjustable="box")
    axins.set_xticks([])
    axins.set_yticks([])
    axins.tick_params(which="both", bottom=False, top=False,
                      left=False, right=False)

    for spine in axins.spines.values():
        spine.set_edgecolor(inset_frame_color)
        spine.set_linewidth(inset_frame_lw)

    # Force final axes geometry before drawing connectors. This is important
    # because set_aspect(..., adjustable="box") changes the actual visible
    # inset frame relative to the raw inset_fig_loc rectangle.
    fig.canvas.draw()

    # ── Optional inset label ──────────────────────────────────────────────
    if show_inset_label:
        _lfs     = inset_label_fontsize if inset_label_fontsize is not None else fontsize_panel_label
        _lcolor  = inset_label_color    if inset_label_color    is not None else panel_label_color_default
        _lstroke = inset_label_stroke   if inset_label_stroke   is not None else panel_label_stroke_default
        _lfx = [pe.withStroke(linewidth=2.5, foreground=_lstroke)]

        axins.text(0.96, 0.04, r"$f_{Edd}^{Ly\alpha}$",
                   transform=axins.transAxes, ha="right", va="bottom",
                   color=_lcolor, fontsize=_lfs, path_effects=_lfx, zorder=12)

    # ── Optional small colorbar inside the inset ──────────────────────────
    if inset_show_cbar:
        _cax = axins.inset_axes([inset_cbar_x, inset_cbar_y,
                                 inset_cbar_width, inset_cbar_height])
        _cb  = plt.colorbar(im, cax=_cax, orientation="vertical")
        _tks, _tklbls = get_log_ticks_from_norm(norm)
        if _tks is not None:
            _cb.set_ticks(_tks)
        if _tklbls is not None:
            _cb.set_ticklabels(_tklbls)
        _cb.ax.tick_params(labelsize=inset_cbar_ticksize,
                           width=0, length=0, pad=3,
                           colors=inset_label_color, labelcolor=inset_label_color)
        _cb.ax.minorticks_off()
        _cb.outline.set_linewidth(1.0)
        _cb.outline.set_edgecolor(inset_label_color)

    # ── Zoom rectangle on the parent panel ────────────────────────────────
    rect = Rectangle((r_min, Z_min), r_max - r_min, Z_max - Z_min,
                     linewidth=inset_connector_lw,
                     edgecolor=inset_connector_color,
                     facecolor="none", zorder=8)
    ax.add_patch(rect)

    # ── Connector lines using ConnectionPatch ─────────────────────────────
    # Corner convention:
    # 1 = upper right, 2 = upper left, 3 = lower left, 4 = lower right
    #
    # Since your inset is to the right of the zoom rectangle, connect
    # rectangle upper-right/lower-right to inset upper-left/lower-left.
    parent_corner_xy = {
        1: (r_max, Z_max),
        2: (r_min, Z_max),
        3: (r_min, Z_min),
        4: (r_max, Z_min),
    }

    inset_corner_xy = {
        1: (1.0, 1.0),
        2: (0.0, 1.0),
        3: (0.0, 0.0),
        4: (1.0, 0.0),
    }

    p1, p2 = inset_parent_corners
    i1, i2 = inset_box_corners

    con1 = ConnectionPatch(
        xyA=parent_corner_xy[p1], coordsA=ax.transData,
        xyB=inset_corner_xy[i1],  coordsB=axins.transAxes,
        color=inset_connector_color, lw=inset_connector_lw,
        zorder=15, clip_on=False
    )
    con2 = ConnectionPatch(
        xyA=parent_corner_xy[p2], coordsA=ax.transData,
        xyB=inset_corner_xy[i2],  coordsB=axins.transAxes,
        color=inset_connector_color, lw=inset_connector_lw,
        zorder=15, clip_on=False
    )

    fig.add_artist(con1)
    fig.add_artist(con2)


def draw_contours(ax, field_2d, R_edges_pc, Z_edges_pc, config_list):
    """Overlay contour lines.  field_2d shape: (Nr, Nz), un-transposed."""
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
        _def_lc        = colors[0] if isinstance(colors, (list, tuple)) else colors
        label_color    = group.get('label_color',    _def_lc)

        cs = ax.contour(R_cen, Z_cen, field_2d.T,
                        levels=levels, colors=colors,
                        linewidths=linewidths, linestyles=linestyle)

        if rounded:
            if hasattr(cs, 'collections'):
                for lc in cs.collections:
                    lc.set_capstyle('round'); lc.set_joinstyle('round')
            else:
                cs.set_capstyle('round'); cs.set_joinstyle('round')

        if labels is not None:
            fmt_dict = {lv: lb for lv, lb in zip(levels, labels)}
            clbls = ax.clabel(cs, levels=levels, fmt=fmt_dict,
                              fontsize=label_fontsize, inline=True)
            for txt in clbls: txt.set_color(label_color)
        elif label_fmt is not None:
            clbls = ax.clabel(cs, levels=levels, fmt=label_fmt,
                              fontsize=label_fontsize, inline=True)
            for txt in clbls: txt.set_color(label_color)


def draw_scale_bar(ax):
    """Draw a |---| scale bar with label above it."""
    if not show_scale_bar:
        return
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    xspan = xhi - xlo
    yspan = yhi - ylo

    bar_x    = scale_bar_x       if scale_bar_x        is not None else xlo + 0.5 * xspan
    bar_y    = scale_bar_y       if scale_bar_y        is not None else ylo + 0.10 * yspan
    capsize  = scale_bar_capsize if scale_bar_capsize  is not None else 0.04 * yspan
    lpad     = scale_bar_label_pad if scale_bar_label_pad is not None else 0.02 * yspan
    fontsize = scale_bar_fontsize  if scale_bar_fontsize  is not None else fontsize_panel_label
    color    = scale_bar_color
    x0, x1  = bar_x - 0.5*scale_bar_length, bar_x + 0.5*scale_bar_length

    fx = ([pe.withStroke(linewidth=scale_bar_stroke_width, foreground=scale_bar_stroke)]
          if scale_bar_stroke is not None else [])

    ax.plot([x0, x1], [bar_y, bar_y], color=color, lw=scale_bar_lw,
            solid_capstyle='butt', transform=ax.transData, zorder=10)
    for xc in (x0, x1):
        ax.plot([xc, xc], [bar_y - 0.5*capsize, bar_y + 0.5*capsize],
                color=color, lw=scale_bar_lw, solid_capstyle='round',
                transform=ax.transData, zorder=10)
    ax.text(bar_x, bar_y + 0.5*capsize + lpad, scale_bar_label,
            color=color, fontsize=fontsize, ha='center', va='bottom',
            transform=ax.transData, zorder=11, path_effects=fx)


def make_ax(fig, col_left, row_bottom, width, height, sharex=None, sharey=None):
    return fig.add_axes([col_left, row_bottom, width, height],
                        sharex=sharex, sharey=sharey)


###############################################################
#######                                               #######
#######         C O M P U T E  E d d i n g t o n    #######
#######                                               #######
###############################################################

def compute_eddington_ratios(f_r, f_Z, fg_r, fg_Z, rhat_r, rhat_Z):
    """
    Given radiation force/volume (f_r, f_Z) and gravitational force/volume
    (fg_r, fg_Z), return the three Eddington ratio fields, each shape (Nr, Nz).

    Cells where |f_grav| or |f_rad| is at or below their mask thresholds
    are set to NaN so pcolormesh renders them as masked.

    1. Naive    : |f_rad| / |f_grav|
    2. Radial   : (f_rad · rhat_sph) / |f_grav|
                  where rhat_sph points away from the cluster/cloud centre
                  at the origin in the meridional (r,Z) plane.
    3. Anti-grav: (-f̂_grav · f_rad) / |f_grav|
                 = -(fg_r*f_r + fg_Z*f_Z) / |f_grav|²
    """
    # Cast to float64 before squaring — float32 underflows to zero for
    # force magnitudes ~1e-23 dyn/cm³ since (1e-23)² = 1e-46 < float32 min ~1.2e-38.
    f_r, f_Z   = np.float64(f_r),  np.float64(f_Z)
    fg_r, fg_Z = np.float64(fg_r), np.float64(fg_Z)

    rhat_r = np.float64(rhat_r)
    rhat_Z = np.float64(rhat_Z)

    f_mag  = np.sqrt(f_r**2  + f_Z**2)
    fg_mag = np.sqrt(fg_r**2 + fg_Z**2)

    # Spherically radial component of the radiation force, directed away
    # from the cluster/cloud centre at (r,Z) = (0,0).
    f_radial_sph = f_r * rhat_r + f_Z * rhat_Z

    # Build mask: True where the ratio is well-defined
    valid = (fg_mag > f_grav_mask_threshold) & (f_mag > f_rad_mask_threshold)

    # Safe division — use np.where so we never actually divide by zero
    fg_safe = np.where(valid, fg_mag, 1.0)   # dummy denominator for invalid cells

    fEdd_naive    = np.where(valid, f_mag / fg_safe,                             np.nan)
    fEdd_radial   = np.where(valid, f_radial_sph / fg_safe,                      np.nan)
    fEdd_antigrav = np.where(valid, -(fg_r*f_r + fg_Z*f_Z) / (fg_safe**2),      np.nan)

    return fEdd_naive, fEdd_radial, fEdd_antigrav


def pressure_gradient(P, r_centres, Z_centres):
    """Return ∂P/∂r and ∂P/∂Z on cell centres.

    P is cell-centred. Since P is a scalar in cylindrical coordinates,
    ∇P = (∂P/∂r) rhat + (∂P/∂Z) Zhat in this axisymmetric r-Z slice;
    there is no extra 1/r factor for these components.
    """
    P64 = np.asarray(P, dtype=np.float64)
    edge_order = 2 if (P64.shape[0] >= 3 and P64.shape[1] >= 3) else 1
    dPdr, dPdZ = np.gradient(P64, r_centres, Z_centres, edge_order=edge_order)
    return dPdr, dPdZ



###############################################################
#######                                               #######
#######            R E A D  D A T A                  #######
#######                                               #######
###############################################################

with h5py.File(Filename, "r") as f:
    t   = f["Times"][:]
    r   = f["r"][:]
    Z   = f["Z"][:]

    f_Lya_r_raw      = f["f_Lya_r_History"][:]
    f_Lya_Z_raw      = f["f_Lya_Z_History"][:]
    f_stellar_r_raw  = f["f_stellar_r_History"][:]
    f_stellar_Z_raw  = f["f_stellar_Z_History"][:]
    f_gravity_r_raw  = f["f_gravity_r_History"][:]
    f_gravity_Z_raw  = f["f_gravity_Z_History"][:]

    P_raw = f["GasPressure_History"][:] if show_gas_pressure_fEdd else None


# Reorder all to (Nt, Nr, Nz)
f_Lya_r_hist     = reorder_history_3d(f_Lya_r_raw,     t, r, Z)
f_Lya_Z_hist     = reorder_history_3d(f_Lya_Z_raw,     t, r, Z)
f_stellar_r_hist = reorder_history_3d(f_stellar_r_raw, t, r, Z)
f_stellar_Z_hist = reorder_history_3d(f_stellar_Z_raw, t, r, Z)
f_gravity_r_hist = reorder_history_3d(f_gravity_r_raw, t, r, Z)
f_gravity_Z_hist = reorder_history_3d(f_gravity_Z_raw, t, r, Z)
P_hist           = reorder_history_3d(P_raw,            t, r, Z)

R_edges_pc, Z_edges_pc = np.meshgrid(r / pc, Z / pc, indexing="ij")

# Cell-centred unit vector pointing spherically radially outwards from
# the cluster/cloud centre at the origin in the meridional (r,Z) plane.
r_centres = 0.5 * (r[:-1] + r[1:])
Z_centres = 0.5 * (Z[:-1] + Z[1:])
r2D, Z2D = np.meshgrid(r_centres, Z_centres, indexing="ij")
R_sph = np.sqrt(r2D*r2D + Z2D*Z2D)

rhat_r = np.zeros_like(R_sph)
rhat_Z = np.zeros_like(R_sph)

good_rhat = R_sph > 0.0
rhat_r[good_rhat] = r2D[good_rhat] / R_sph[good_rhat]
rhat_Z[good_rhat] = Z2D[good_rhat] / R_sph[good_rhat]


###############################################################
#######                                               #######
#######     C O L O U R  N O R M                      #######
#######                                               #######
###############################################################

if use_symlognorm:
    norm_fEdd = SymLogNorm(linthresh=fEdd_linthresh,
                           vmin=-fEdd_vmax, vmax=fEdd_vmax)
else:
    norm_fEdd = LogNorm(vmin=fEdd_vmin, vmax=fEdd_vmax)


###############################################################
#######                                               #######
#######         C L I P  F I E L D S                  #######
#######                                               #######
###############################################################

def clip(field):
    """Clip to the colour norm range, preserving NaN for masked cells.
    For LogNorm we take abs() so signed quantities (radial, anti-grav)
    can be visualised on the same scale; switch use_symlognorm for signed plots.
    """
    if use_symlognorm:
        return np.where(np.isfinite(field),
                        np.clip(field, -fEdd_vmax, fEdd_vmax), np.nan)
    else:
        return np.where(np.isfinite(field),
                        np.clip(np.abs(field), norm_fEdd.vmin, norm_fEdd.vmax),
                        np.nan)


def compute_snapshot_fields(target_time_kyr):
    """Return all f_Edd fields and plot fields for the nearest saved snapshot."""
    k = nearest_saved_index(t, target_time_kyr * kyr)
    print(f"Requested target time = {target_time_kyr:.3f} kyr")
    print(f"Using snapshot at     = {t[k] / kyr:.3f} kyr")

    # Snapshot slices  (Nr, Nz)
    f_Lya_r     = f_Lya_r_hist[k]
    f_Lya_Z     = f_Lya_Z_hist[k]
    f_stellar_r = f_stellar_r_hist[k]
    f_stellar_Z = f_stellar_Z_hist[k]
    f_gravity_r = f_gravity_r_hist[k]
    f_gravity_Z = f_gravity_Z_hist[k]

    fEdd_lya_naive, fEdd_lya_rad, fEdd_lya_anti = compute_eddington_ratios(
        f_Lya_r,     f_Lya_Z,     f_gravity_r, f_gravity_Z, rhat_r, rhat_Z)

    fEdd_dir_naive, fEdd_dir_rad, fEdd_dir_anti = compute_eddington_ratios(
        f_stellar_r, f_stellar_Z, f_gravity_r, f_gravity_Z, rhat_r, rhat_Z)

    raw_fields = {
        "lya_naive": fEdd_lya_naive,
        "lya_rad":   fEdd_lya_rad,
        "lya_anti":  fEdd_lya_anti,
        "dir_naive": fEdd_dir_naive,
        "dir_rad":   fEdd_dir_rad,
        "dir_anti":  fEdd_dir_anti,
    }

    if show_gas_pressure_fEdd:
        # Gas-pressure force density is computed from scratch as -∇P,
        # using the same cell-centred pressure-gradient method as in the
        # force-integral script.
        P = P_hist[k]
        dPdr, dPdZ = pressure_gradient(P, r_centres, Z_centres)
        f_gas_r = -dPdr
        f_gas_Z = -dPdZ

        fEdd_gas_naive, fEdd_gas_rad, fEdd_gas_anti = compute_eddington_ratios(
            f_gas_r, f_gas_Z, f_gravity_r, f_gravity_Z, rhat_r, rhat_Z)

        raw_fields.update({
            "gas_naive": fEdd_gas_naive,
            "gas_rad":   fEdd_gas_rad,
            "gas_anti":  fEdd_gas_anti,
        })

    plot_fields = {key: clip(val) for key, val in raw_fields.items()}

    return {
        "k": k,
        "time_kyr": t[k] / kyr,
        "raw": raw_fields,
        "plot": plot_fields,
    }


# Compute both displayed snapshots.
snapshot_1 = compute_snapshot_fields(TargetTime_kyr_1)
snapshot_2 = compute_snapshot_fields(TargetTime_kyr_2)


###############################################################
#######                                                 #######
#######           F I G U R E  &  A X E S               #######
#######                                                 #######
###############################################################

fig = plt.figure(figsize=figsize)

# Panel positions: two time blocks. In each block columns are [naive, anti-grav]
# and rows are [Lyα, direct, gas pressure] if show_gas_pressure_fEdd=True.
block_width = 2.0 * col_width + col_gap
block_lefts = [col_left_0,
               col_left_0 + block_width + group_gap]
col_lefts_1 = [block_lefts[0] + i * (col_width + col_gap) for i in range(2)]
col_lefts_2 = [block_lefts[1] + i * (col_width + col_gap) for i in range(2)]

if show_gas_pressure_fEdd:
    row_tops = [row_bottom_0 + 2.0 * (row_height + row_gap),  # row 0 = Lyα
                row_bottom_0 + 1.0 * (row_height + row_gap),  # row 1 = direct
                row_bottom_0]                                  # row 2 = gas pressure
else:
    row_tops = [row_bottom_0 + row_height + row_gap,            # row 0 = Lyα
                row_bottom_0]                                  # row 1 = direct

ax_lya_naive_1 = make_ax(fig, col_lefts_1[0], row_tops[0], col_width, row_height)
ax_lya_anti_1  = make_ax(fig, col_lefts_1[1], row_tops[0], col_width, row_height,
                         sharex=ax_lya_naive_1, sharey=ax_lya_naive_1)
ax_dir_naive_1 = make_ax(fig, col_lefts_1[0], row_tops[1], col_width, row_height,
                         sharex=ax_lya_naive_1, sharey=ax_lya_naive_1)
ax_dir_anti_1  = make_ax(fig, col_lefts_1[1], row_tops[1], col_width, row_height,
                         sharex=ax_lya_naive_1, sharey=ax_lya_naive_1)

ax_lya_naive_2 = make_ax(fig, col_lefts_2[0], row_tops[0], col_width, row_height,
                         sharex=ax_lya_naive_1, sharey=ax_lya_naive_1)
ax_lya_anti_2  = make_ax(fig, col_lefts_2[1], row_tops[0], col_width, row_height,
                         sharex=ax_lya_naive_1, sharey=ax_lya_naive_1)
ax_dir_naive_2 = make_ax(fig, col_lefts_2[0], row_tops[1], col_width, row_height,
                         sharex=ax_lya_naive_1, sharey=ax_lya_naive_1)
ax_dir_anti_2  = make_ax(fig, col_lefts_2[1], row_tops[1], col_width, row_height,
                         sharex=ax_lya_naive_1, sharey=ax_lya_naive_1)

all_axes = [ax_lya_naive_1, ax_lya_anti_1, ax_dir_naive_1, ax_dir_anti_1,
            ax_lya_naive_2, ax_lya_anti_2, ax_dir_naive_2, ax_dir_anti_2]

if show_gas_pressure_fEdd:
    ax_gas_naive_1 = make_ax(fig, col_lefts_1[0], row_tops[2], col_width, row_height,
                             sharex=ax_lya_naive_1, sharey=ax_lya_naive_1)
    ax_gas_anti_1  = make_ax(fig, col_lefts_1[1], row_tops[2], col_width, row_height,
                             sharex=ax_lya_naive_1, sharey=ax_lya_naive_1)
    ax_gas_naive_2 = make_ax(fig, col_lefts_2[0], row_tops[2], col_width, row_height,
                             sharex=ax_lya_naive_1, sharey=ax_lya_naive_1)
    ax_gas_anti_2  = make_ax(fig, col_lefts_2[1], row_tops[2], col_width, row_height,
                             sharex=ax_lya_naive_1, sharey=ax_lya_naive_1)
    all_axes += [ax_gas_naive_1, ax_gas_anti_1, ax_gas_naive_2, ax_gas_anti_2]

fig.text(0.42, 0.955,
         "Eddington ratios for Ly$\\alpha$ and direct radiation pressure feedback",
         ha="center", va="bottom", fontsize=85)


###############################################################
#######                                               #######
#######         P L O T  P A N E L S                  #######
#######                                               #######
###############################################################

# Build a colormap that renders NaN cells in mask_color
_cmap_obj = plt.colormaps[cmap_fEdd].copy()
_cmap_obj.set_bad(color=mask_color)

_pcolormesh_kwargs = dict(norm=norm_fEdd, shading="auto",
                          cmap=_cmap_obj, rasterized=rasterized)

def plot_block(snapshot, axes):
    if show_gas_pressure_fEdd:
        ax_lya_naive, ax_lya_anti, ax_dir_naive, ax_dir_anti, ax_gas_naive, ax_gas_anti = axes
    else:
        ax_lya_naive, ax_lya_anti, ax_dir_naive, ax_dir_anti = axes
    plot_fields = snapshot["plot"]

    im_lya_naive = ax_lya_naive.pcolormesh(R_edges_pc, Z_edges_pc,
                                           plot_fields["lya_naive"].T,
                                           **_pcolormesh_kwargs)
    im_lya_anti  = ax_lya_anti.pcolormesh( R_edges_pc, Z_edges_pc,
                                           plot_fields["lya_anti"].T,
                                           **_pcolormesh_kwargs)
    im_dir_naive = ax_dir_naive.pcolormesh(R_edges_pc, Z_edges_pc,
                                           plot_fields["dir_naive"].T,
                                           **_pcolormesh_kwargs)
    im_dir_anti  = ax_dir_anti.pcolormesh( R_edges_pc, Z_edges_pc,
                                           plot_fields["dir_anti"].T,
                                           **_pcolormesh_kwargs)

    if show_gas_pressure_fEdd:
        im_gas_naive = ax_gas_naive.pcolormesh(R_edges_pc, Z_edges_pc,
                                               plot_fields["gas_naive"].T,
                                               **_pcolormesh_kwargs)
        im_gas_anti  = ax_gas_anti.pcolormesh( R_edges_pc, Z_edges_pc,
                                               plot_fields["gas_anti"].T,
                                               **_pcolormesh_kwargs)
        return im_lya_naive, im_lya_anti, im_dir_naive, im_dir_anti, im_gas_naive, im_gas_anti

    return im_lya_naive, im_lya_anti, im_dir_naive, im_dir_anti

if show_gas_pressure_fEdd:
    (im_lya_naive_1, im_lya_anti_1, im_dir_naive_1, im_dir_anti_1,
     im_gas_naive_1, im_gas_anti_1) = plot_block(
        snapshot_1, (ax_lya_naive_1, ax_lya_anti_1, ax_dir_naive_1, ax_dir_anti_1,
                     ax_gas_naive_1, ax_gas_anti_1))

    (im_lya_naive_2, im_lya_anti_2, im_dir_naive_2, im_dir_anti_2,
     im_gas_naive_2, im_gas_anti_2) = plot_block(
        snapshot_2, (ax_lya_naive_2, ax_lya_anti_2, ax_dir_naive_2, ax_dir_anti_2,
                     ax_gas_naive_2, ax_gas_anti_2))
else:
    im_lya_naive_1, im_lya_anti_1, im_dir_naive_1, im_dir_anti_1 = plot_block(
        snapshot_1, (ax_lya_naive_1, ax_lya_anti_1, ax_dir_naive_1, ax_dir_anti_1))

    im_lya_naive_2, im_lya_anti_2, im_dir_naive_2, im_dir_anti_2 = plot_block(
        snapshot_2, (ax_lya_naive_2, ax_lya_anti_2, ax_dir_naive_2, ax_dir_anti_2))


###############################################################
#######                                                 #######
#######              C O N T O U R S                    #######
#######                                                 #######
###############################################################

def draw_block_contours(snapshot, axes, suffix):
    raw_fields = snapshot["raw"]

    axes_map = {
        f"ax_lya_naive{suffix}": axes[0],
        f"ax_lya_anti{suffix}":  axes[1],
        f"ax_dir_naive{suffix}": axes[2],
        f"ax_dir_anti{suffix}":  axes[3],
    }
    field_map = {
        f"ax_lya_naive{suffix}": raw_fields["lya_naive"],
        f"ax_lya_anti{suffix}":  raw_fields["lya_anti"],
        f"ax_dir_naive{suffix}": raw_fields["dir_naive"],
        f"ax_dir_anti{suffix}":  raw_fields["dir_anti"],
    }

    if show_gas_pressure_fEdd:
        axes_map.update({
            f"ax_gas_naive{suffix}": axes[4],
            f"ax_gas_anti{suffix}":  axes[5],
        })
        field_map.update({
            f"ax_gas_naive{suffix}": raw_fields["gas_naive"],
            f"ax_gas_anti{suffix}":  raw_fields["gas_anti"],
        })

    # Use the unsuffixed contour_config names for both time blocks.
    config_key = {
        f"ax_lya_naive{suffix}": "ax_lya_naive",
        f"ax_lya_anti{suffix}":  "ax_lya_anti",
        f"ax_dir_naive{suffix}": "ax_dir_naive",
        f"ax_dir_anti{suffix}":  "ax_dir_anti",
        f"ax_gas_naive{suffix}": "ax_gas_naive",
        f"ax_gas_anti{suffix}":  "ax_gas_anti",
    }
    for key, ax in axes_map.items():
        draw_contours(ax, field_map[key], R_edges_pc, Z_edges_pc,
                      contour_config.get(config_key[key]))

if show_gas_pressure_fEdd:
    draw_block_contours(snapshot_1, (ax_lya_naive_1, ax_lya_anti_1,
                                    ax_dir_naive_1, ax_dir_anti_1,
                                    ax_gas_naive_1, ax_gas_anti_1), "_1")
    draw_block_contours(snapshot_2, (ax_lya_naive_2, ax_lya_anti_2,
                                    ax_dir_naive_2, ax_dir_anti_2,
                                    ax_gas_naive_2, ax_gas_anti_2), "_2")
else:
    draw_block_contours(snapshot_1, (ax_lya_naive_1, ax_lya_anti_1,
                                    ax_dir_naive_1, ax_dir_anti_1), "_1")
    draw_block_contours(snapshot_2, (ax_lya_naive_2, ax_lya_anti_2,
                                    ax_dir_naive_2, ax_dir_anti_2), "_2")


###############################################################
#######                                               #######
#######     P A N E L  C O S M E T I C S              #######
#######                                               #######
###############################################################

for ax in all_axes:
    ax.set_xlim(r[0] / pc, xmax_fraction * r[-1] / pc)
    ax.set_ylim(Z[0] / pc, zmax_fraction * Z[-1] / pc)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="both", bottom=False, top=False,
                   left=False, right=False,
                   labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(panel_frame_linewidth)

# The H II / H I labels are only drawn once, on the first block.
ax_lya_naive_1.text(0.03, 0.1, "H II region", color="black", fontsize=52)
ax_lya_naive_1.text(0.25, 0.3, "H I region",  color="white", fontsize=52)

# Physics labels — colour and stroke resolved per panel and per time block.
def _plabel_color(key, block=1):
    if block == 1:
        override = panel_label_colors_1.get(key)
        return override if override is not None else panel_label_color_default_1
    else:
        override = panel_label_colors_2.get(key)
        return override if override is not None else panel_label_color_default_2

def _plabel_stroke(key, block=1):
    if block == 1:
        override = panel_label_strokes_1.get(key)
        return override if override is not None else panel_label_stroke_default_1
    else:
        override = panel_label_strokes_2.get(key)
        return override if override is not None else panel_label_stroke_default_2

# Panel labels for the left time block.
if show_panel_labels_block_1:
    panel_label(ax_lya_naive_1, r"$f_{Edd}^{Ly\alpha}$ (naive)",    _plabel_color("ax_lya_naive", 1), _plabel_stroke("ax_lya_naive", 1))
    panel_label(ax_lya_anti_1,  r"$f_{Edd}^{Ly\alpha}$ (antigrav)", _plabel_color("ax_lya_anti",  1), _plabel_stroke("ax_lya_anti",  1))
    panel_label(ax_dir_naive_1, r"$f_{Edd}^{dir}$ (naive)",         _plabel_color("ax_dir_naive", 1), _plabel_stroke("ax_dir_naive", 1))
    panel_label(ax_dir_anti_1,  r"$f_{Edd}^{dir}$ (antigrav)",      _plabel_color("ax_dir_anti",  1), _plabel_stroke("ax_dir_anti",  1))
    if show_gas_pressure_fEdd:
        panel_label(ax_gas_naive_1, r"$f_{Edd}^{gas}$ (naive)",     _plabel_color("ax_gas_naive", 1), _plabel_stroke("ax_gas_naive", 1))
        panel_label(ax_gas_anti_1,  r"$f_{Edd}^{gas}$ (antigrav)",  _plabel_color("ax_gas_anti",  1), _plabel_stroke("ax_gas_anti",  1))

# Panel labels for the right time block.
if show_panel_labels_block_2:
    panel_label(ax_lya_naive_2, r"$f_{Edd}^{Ly\alpha}$ (naive)",    _plabel_color("ax_lya_naive", 2), _plabel_stroke("ax_lya_naive", 2))
    panel_label(ax_lya_anti_2,  r"$f_{Edd}^{Ly\alpha}$ (antigrav)", _plabel_color("ax_lya_anti",  2), _plabel_stroke("ax_lya_anti",  2))
    panel_label(ax_dir_naive_2, r"$f_{Edd}^{dir}$ (naive)",         _plabel_color("ax_dir_naive", 2), _plabel_stroke("ax_dir_naive", 2))
    panel_label(ax_dir_anti_2,  r"$f_{Edd}^{dir}$ (antigrav)",      _plabel_color("ax_dir_anti",  2), _plabel_stroke("ax_dir_anti",  2))
    if show_gas_pressure_fEdd:
        panel_label(ax_gas_naive_2, r"$f_{Edd}^{gas}$ (naive)",     _plabel_color("ax_gas_naive", 2), _plabel_stroke("ax_gas_naive", 2))
        panel_label(ax_gas_anti_2,  r"$f_{Edd}^{gas}$ (antigrav)",  _plabel_color("ax_gas_anti",  2), _plabel_stroke("ax_gas_anti",  2))

# Time labels on the top-left panel of each time block.
time_label(ax_lya_naive_1, snapshot_1["time_kyr"],
           color=time_label_color_1, stroke=time_label_stroke_1)
time_label(ax_lya_naive_2, snapshot_2["time_kyr"],
           color=time_label_color_2, stroke=time_label_stroke_2)

# Scale bar: draw only once, since the scale is shared.
_axes_map = {
    "ax_lya_naive": ax_lya_naive_1,
    "ax_lya_anti":  ax_lya_anti_1,
    "ax_dir_naive": ax_dir_naive_1,
    "ax_dir_anti":  ax_dir_anti_1,
}
if show_gas_pressure_fEdd:
    _axes_map.update({
        "ax_gas_naive": ax_gas_naive_1,
        "ax_gas_anti":  ax_gas_anti_1,
    })
if show_scale_bar and scale_bar_panel in _axes_map:
    draw_scale_bar(_axes_map[scale_bar_panel])

# Inset zoom panel: only the first time block gets the inset.
_plot_fields = {
    "ax_lya_naive": snapshot_1["plot"]["lya_naive"],
    "ax_lya_anti":  snapshot_1["plot"]["lya_anti"],
    "ax_dir_naive": snapshot_1["plot"]["dir_naive"],
    "ax_dir_anti":  snapshot_1["plot"]["dir_anti"],
}
if show_gas_pressure_fEdd:
    _plot_fields.update({
        "ax_gas_naive": snapshot_1["plot"]["gas_naive"],
        "ax_gas_anti":  snapshot_1["plot"]["gas_anti"],
    })
if show_inset and inset_panel in _axes_map:
    _inset_field = _plot_fields[inset_panel]
    _inset_color = inset_label_color if inset_label_color is not None else panel_label_color_default
    draw_inset(_axes_map[inset_panel], _inset_field, norm_fEdd, _cmap_obj)

# Row labels in the left margin only; both time blocks share these.
if show_row_labels:
    row_lya_y = row_tops[0] + 0.5 * row_height
    row_dir_y = row_tops[1] + 0.5 * row_height
    fig.text(row_label_x, row_lya_y, row_label_Lya,
             ha=row_label_ha, va="center", color=row_label_color,
             fontsize=fontsize_row_label, rotation=90)
    fig.text(row_label_x, row_dir_y, row_label_dir,
             ha=row_label_ha, va="center", color=row_label_color,
             fontsize=fontsize_row_label, rotation=90)
    if show_gas_pressure_fEdd:
        row_gas_y = row_tops[2] + 0.5 * row_height
        fig.text(row_label_x, row_gas_y, row_label_gas,
                 ha=row_label_ha, va="center", color=row_label_color,
                 fontsize=fontsize_row_label, rotation=90)


###############################################################
#######                                               #######
#######     S H A R E D  C O L O R B A R              #######
#######                                               #######
###############################################################

# Add the colorbar to the first Lyα anti-grav panel only; the colour scale is
# shared by all panels in both time blocks.
ticks_fEdd, ticklabels_fEdd = get_log_ticks_from_norm(norm_fEdd)
ticks_fEdd, ticklabels_fEdd = use_custom_ticks(
    ticks_fEdd, ticklabels_fEdd, ticks_fEdd_custom, ticklabels_fEdd_custom)

add_inset_colorbar(ax_lya_anti_1, im_lya_naive_1,
                   ticks_fEdd, ticklabels_fEdd,
                   tick_color=cbar_tick_color,
                   outline_color=cbar_outline_color)


###############################################################
#######                                               #######
#######         S A V E                               #######
#######                                               #######
###############################################################

plt.savefig(FigureName, bbox_inches="tight")
print(f"Saved figure to {FigureName}")
