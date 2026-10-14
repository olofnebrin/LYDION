import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
from matplotlib.patches import ConnectionPatch, Rectangle
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
from matplotlib.patches import ConnectionPatch, Rectangle
from pathlib import Path
import sys
import os

repo_root = Path(__file__).resolve().parents[1]
os.chdir(repo_root)
cmastro_src = repo_root / "src" / "cmastro" / "src"
sys.path.insert(0, str(cmastro_src))

import cmastro


# ===================================================================
# User inputs
# ===================================================================

Filenames = [
    "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5",
]
TargetTimes_kyr = [30.0]

FigureName = "LYDION_Figure_LyaSpectra_512D0.01_t30.pdf"

# Frequency-axis units:
#   "x"                   -> x = (ν - ν0)/ΔνD(T_doppler_K)
#   "velocity_wavelength" -> observer/red-positive velocity offset in km/s,
#                            Δv = c (λ - λ0)/λ0 ≃ -c (ν - ν0)/ν0
#   "velocity_frequency"  -> old frequency-positive velocity offset in km/s,
#                            Δv = c (ν - ν0)/ν0
#   "frequency"           -> raw frequency in Hz
#
# For observer-style spectra, use "velocity_wavelength": blue is negative,
# red is positive.  This is the main sign flip relative to the old code.
frequency_axis_units = "velocity_wavelength"       # "x", "velocity_wavelength", "velocity_frequency", or "frequency"

# -------------------------------------------------------------------
# Background / theme
# -------------------------------------------------------------------
# If True, only the visual background/theme is changed:
#   - figure and spectra panel interiors become black
#   - panel outlines, spectrum labels/ticks, and inset lines become white
# No data orientation, colormap, normalization, or panel placement is changed.
use_black_background = False

# -------------------------------------------------------------------
# Density panel
# -------------------------------------------------------------------

density_field = "nHI"          # "nHI" or "nH"

nHI_vmin, nHI_vmax = 0.01, 3e6
nH_vmin,  nH_vmax  = 0.01, 3e6

cmap_density  = "Blues"
#cmap_density  = "cma:laguna"
xmax_fraction = 1.0
zmax_fraction = 1.0
rasterized    = True

# Important Julia/HDF5/Python ordering note:
# The LYDION save code writes history arrays in Julia as
#     dset[i_r, i_Z, step] = field[i_r, i_Z]
# i.e. physical order (r, Z, time).  When read with h5py/NumPy,
# the raw array effectively appears with the axes reversed, so after
# moving the time axis first the spatial axes are (Z, r), not (r, Z).
#
# Therefore we transpose once to recover the physical LYDION convention:
#     field_RZ[i,j] = field at (r_i, Z_j)
#
# Later, for Matplotlib pcolormesh(r, Z, C), we still plot field_RZ.T
# because Matplotlib expects C.shape = (NZ, Nr) = (y, x).
history_spatial_order_after_time_axis = "Zr"      # "rZ" or "Zr"

# Saved Lyα edge index arrays are Julia 1-based physical cell indices.
# Keep this as "rZ": rIndex -> r-cell, ZIndex -> Z-cell.
edge_index_physical_order = "rZ"                 # "rZ" or "Zr"

# -------------------------------------------------------------------
# Spectra panels
# -------------------------------------------------------------------
# boundary is always a PHYSICAL boundary:
#   "right" = maximum r
#   "top"   = maximum Z
#   "bottom"= minimum Z
#   "left"  = minimum r / axis side, if saved
#
# pos_mode:
#   "frac" -> use pos_frac along that physical boundary
#   "peak" -> choose the brightest saved spectrum on that boundary

spectra_panels = [
    dict(
        boundary = "right",
        pos_frac = 0.0,
        pos_mode = "frac",
        x_label  = None,
        show_xticks = True,
        show_xticklabels = True,
        show_xlabel = True,
    ),
    dict(
        boundary = "right",
        pos_frac = 0.5,
        pos_mode = "frac",
        x_label  = None,
        show_xticks = True,
        show_xticklabels = False,
        show_xlabel = False,
    ),
    dict(
        boundary = "right",
        pos_frac = 1.0,
        pos_mode = "frac",
        x_label  = None,
        show_xticks = True,
        show_xticklabels = False,
        show_xlabel = False,
    ),
    dict(
        boundary = "top",
        pos_frac = 0.25,
        pos_mode = "frac",
        x_label  = None,
        show_xlabel = False,
    ),
    dict(
        boundary = "top",
        pos_frac = 0.75,
        pos_mode = "frac",
        x_label  = None,
        show_xlabel = False,
    ),
]

#freq_xlabel     = r"$\mathbf{x} \equiv (\nu - \nu_0)/\Delta\nu_D(10^4 K)$"
freq_xlabel     = "x = (ν - ν$_{Ly\\alpha}$)/Δν$_{D}$(T = 10$^4$ K)"

# Kept for backward compatibility with the old option. If frequency_axis_units
# is None, the code falls back to this old use_doppler_x behaviour.
use_doppler_x   = False

T_doppler_K     = 1e2
spec_xlim       = (-1350, 1350)      # in selected frequency_axis_units

velocity_wavelength_xlabel = (
    r"$\Delta v$ "
    r"(km s$^{-1}$)"
)
velocity_frequency_xlabel = (
    r"$\Delta v = c(\nu-\nu_{Ly\alpha})/\nu_{Ly\alpha}$ "
    r"(km s$^{-1}$)"
)
frequency_xlabel = r"$\nu$ [Hz]"

spec_line_color   = "crimson"
spec_line_lw      = 4.0
normalise_spectra = True

# Optional fill under each spectrum.
# Can be overridden per spectrum panel with e.g.
#     fill_under=True, fill_color="crimson", fill_alpha=0.20
fill_under_spectra = True
spec_fill_color    = "crimson"
spec_fill_alpha    = 0.25

# If True, all spectra are divided by the largest peak among the selected
# plotted spectra, and all spectrum panels use the same y-limits.
# If False, each spectrum keeps the old behavior: normalized by its own peak.
shared_spectrum_y_scale = True
spectrum_ymax_factor    = 1.12

# -------------------------------------------------------------------
# Figure / layout
# -------------------------------------------------------------------

figsize     = (28, 20)
n_snapshots = len(Filenames)

dens_bottom = 0.06
dens_height = 0.60
dens_width  = 0.34
col_gap     = 0.06
col_left_0  = 0.04

spec_width  = 0.22*0.75
spec_height = 0.18*0.78

spec_gap = 0.04

# -------------------------------------------------------------------
# Spectra panel placement controls
# -------------------------------------------------------------------
# These settings only move/resize the spectra panels on the page.
# They do NOT change which spectra are selected. The sampling positions
# are still controlled only by boundary, pos_frac, and pos_mode above.
#
# Matplotlib figure coordinates are used: left/bottom/width/height are
# fractions of the full figure canvas.
#
# Per-panel overrides can also be added inside any spectra_panels dict:
#     panel_dx=..., panel_dy=...             # extra shift after auto placement
#     panel_left=..., panel_bottom=...       # absolute figure placement
#     panel_width=..., panel_height=...      # override size
#     panel_gap=...                          # override boundary gap

right_panel_gap        = 0.006     # smaller -> right panels closer to nHI panel
right_panel_dx         = 0.000
right_panel_dy         = 0.000
right_panel_bottom_min = dens_bottom + 0.12 * dens_height
right_panel_bottom_max = dens_bottom + 0.78 * dens_height

# Right-side panel placement mode:
#   "physical" -> vertical position follows the sampled Z-position
#   "even"     -> right panels are evenly stacked next to the nHI panel,
#                 with the bottom/top panels aligned to the nHI panel.
right_panel_layout_mode = "even"
# If None, these are interpreted relative to the ACTUAL drawn nHI axes
# after Matplotlib has applied aspect="equal". This avoids the spectra panels
# drifting away from the visible nHI panel.
right_panel_stack_bottom = None
right_panel_stack_top    = None
right_panel_stack_bottom_frac = 0.04
right_panel_stack_top_frac    = 0.96

top_panel_gap          = 0.03     # smaller -> top panels closer to nHI panel
top_panel_dx           = 0.000
top_panel_dy           = 0.000    # negative -> move top panels downward
top_panel_left_min     = None      # None -> density-panel left edge
top_panel_left_max     = None      # None -> density-panel right edge - panel width

bottom_panel_gap       = 0.035
bottom_panel_dx        = 0.000
bottom_panel_dy        = 0.000
bottom_panel_left_min  = None      # None -> density-panel left edge
bottom_panel_left_max  = None      # None -> density-panel right edge - panel width

left_panel_gap         = 0.035
left_panel_dx          = 0.000
left_panel_dy          = 0.000
left_panel_bottom_min  = dens_bottom + 0.02 * dens_height
left_panel_bottom_max  = dens_bottom + 0.86 * dens_height

show_colorbar  = True         # toggle the density colorbar on/off

cbar_x, cbar_y = 0.8, 0.06
cbar_width     = 0.05
cbar_height    = 0.45

# -------------------------------------------------------------------
# Connector style
# -------------------------------------------------------------------

connector_color = "none"
connector_lw    = 3.0

# Optional dashed inset/connector lines. Applies to the selected-cell rectangle
# and the connector lines from the density panel to each spectrum inset.
use_dashed_inset_lines      = False
inset_line_dashes           = (6.0, 5.0)
inset_line_dash_capstyle    = "round"
inset_line_dash_joinstyle   = "round"

spec_frame_color = "black"
spec_frame_lw    = 3.0

# -------------------------------------------------------------------
# Scale bar  (drawn on the nHI density panel)
# -------------------------------------------------------------------
show_scale_bar         = True
scale_bar_length       = 0.5        # physical length in pc
scale_bar_label        = "0.5 pc"
scale_bar_x            = 0.3       # None → horizontally centred in panel
scale_bar_y            = 0.85      # None → 10 % up from bottom of panel
scale_bar_color        = "black"
scale_bar_lw           = 3.0
scale_bar_capsize      = None       # None → 4 % of panel height
scale_bar_label_pad    = None       # None → 2 % of panel height
scale_bar_fontsize     = None       # None → fontsize_panel_label
scale_bar_stroke       = "white"
scale_bar_stroke_width = 3.0

# -------------------------------------------------------------------
# Labels
# -------------------------------------------------------------------

fontsize_panel_label = 55
fontsize_time        = 48
fontsize_spec_label  = 35
fontsize_cbar_ticks  = 32
panel_frame_lw       = 3.0

# Default x-axis display for spectra panels.
# Can be overridden per spectra_panels dict with:
#     show_xticks=False
#     show_xticklabels=False
#     show_xlabel=False
#     x_label=r"$x$"
#     x_tick_values=[-100, 0, 100]
#     x_tick_labels=[r"$-100$", r"$0$", r"$100$"]
default_show_spec_xticks      = True
default_show_spec_xticklabels = True
default_show_spec_xlabel      = True
default_spec_x_label          = None
default_spec_x_tick_values    = None
default_spec_x_tick_labels    = None

time_label_color        = "black"
time_label_stroke       = "white"
time_label_stroke_width = 3.0

density_label_color       = "black"
density_label_stroke      = "white"
density_label_stroke_width = 3.5

# -------------------------------------------------------------------
# Derived background/theme colors
# -------------------------------------------------------------------

if use_black_background:
    figure_facecolor        = "black"
    density_panel_facecolor = "none"
    spectra_panel_facecolor = "black"

    panel_outline_color     = "white"
    active_connector_color  = "white"
    active_spec_frame_color = "white"

    spectra_label_color     = "white"
    spectra_tick_color      = "white"
else:
    figure_facecolor        = "white"
    density_panel_facecolor = "none"
    spectra_panel_facecolor = "white"

    panel_outline_color     = "black"
    active_connector_color  = connector_color
    active_spec_frame_color = spec_frame_color

    spectra_label_color     = "black"
    spectra_tick_color      = "black"

# -------------------------------------------------------------------
# Physical constants
# -------------------------------------------------------------------

yr   = 3.1556926e7
kyr  = 1e3 * yr
pc   = 3.085677581e18
mH   = 1.6735575e-24
X    = 0.76

c_light = 2.998e10
nu0_Lya = 2.466e15
kB      = 1.380649e-16


###############################################################
#######          P L O T  C O S M E T I C S            #######
###############################################################

plt.rc("font", **{"sans-serif": ["AVHershey Complex"], "size": 39.5})
params = {"mathtext.default": "regular"}
plt.rcParams.update(params)
plt.rcParams["mathtext.fontset"]   = "custom"
plt.rcParams["mathtext.rm"]        = "GreekC"
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["xtick.major.width"] = 1.7
plt.rcParams["xtick.minor.width"] = 1.7
plt.rcParams["xtick.major.size"]  = 16
plt.rcParams["xtick.minor.size"]  = 6
plt.rcParams["ytick.major.width"] = 1.7
plt.rcParams["ytick.minor.width"] = 1.7
plt.rcParams["ytick.major.size"]  = 16
plt.rcParams["ytick.minor.size"]  = 6
plt.rcParams["xtick.major.pad"]   = "10"
plt.rcParams["ytick.major.pad"]   = "10"

plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.linewidth"] = 2.5
plt.rcParams["grid.linewidth"] = 2.5
plt.rcParams["grid.linestyle"] = "--"

plt.rcParams["xtick.top"]       = True
plt.rcParams["xtick.bottom"]    = True
plt.rcParams["ytick.left"]      = True
plt.rcParams["ytick.right"]     = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


###############################################################
#######          H E L P E R  F U N C T I O N S        #######
###############################################################

def reorder_history_3d(arr, t, r_edges, Z_edges, spatial_order=history_spatial_order_after_time_axis):
    """
    Reorder a 3D history array to physical LYDION shape (Nt, Nr, NZ).

    This deliberately does NOT infer r/Z orientation from shape, because
    square grids make that impossible. It uses the explicit setting
    `history_spatial_order_after_time_axis`.

    After this function:
        arr_hist[k, i, j] belongs to physical cell (rCoM[i], ZCoM[j]).
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    Nt = t.size
    Nr = len(r_edges) - 1
    NZ = len(Z_edges) - 1

    shape = arr.shape
    time_axes = [ax for ax, s in enumerate(shape) if s == Nt]

    if len(time_axes) != 1:
        raise RuntimeError(
            f"Could not uniquely identify time axis: shape={shape}, Nt={Nt}"
        )

    time_axis = time_axes[0]
    arr_tfirst = np.moveaxis(arr, time_axis, 0)

    if spatial_order == "rZ":
        expected = (Nt, Nr, NZ)
        if arr_tfirst.shape != expected:
            raise RuntimeError(
                f"history_spatial_order_after_time_axis='rZ', so expected "
                f"shape {expected} after moving time axis, got {arr_tfirst.shape}."
            )
        return arr_tfirst

    if spatial_order == "Zr":
        expected = (Nt, NZ, Nr)
        if arr_tfirst.shape != expected:
            raise RuntimeError(
                f"history_spatial_order_after_time_axis='Zr', so expected "
                f"shape {expected} after moving time axis, got {arr_tfirst.shape}."
            )
        return arr_tfirst.transpose(0, 2, 1)

    raise ValueError(
        "history_spatial_order_after_time_axis must be either 'rZ' or 'Zr', "
        f"got {spatial_order!r}."
    )

def field_for_physical_rZ_plot(field_2d, r_edges, Z_edges):
    """
    Return C for ax.pcolormesh(r_edges, Z_edges, C).

    Input must already be in physical LYDION convention:
        field_2d[i,j] = field at (rCoM[i], ZCoM[j])

    Matplotlib with x=r and y=Z expects C.shape = (NZ, Nr), so the only
    allowed conversion here is field_2d.T.
    """
    Nr = len(r_edges) - 1
    NZ = len(Z_edges) - 1

    if field_2d.shape != (Nr, NZ):
        raise RuntimeError(
            f"field_2d must be physical (Nr,NZ)=({Nr},{NZ}), "
            f"got field_2d.shape={field_2d.shape}."
        )

    return field_2d.T

def nearest_saved_index(t, target_time):
    saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]
    if saved_idx.size == 0:
        raise RuntimeError("No saved snapshots found.")
    return saved_idx[np.argmin(np.abs(t[saved_idx] - target_time))]


def make_ax(fig, left, bottom, width, height, **kwargs):
    return fig.add_axes([left, bottom, width, height], **kwargs)


def inset_line_style():
    """Return the line style used for cell boxes and connector lines."""
    if use_dashed_inset_lines:
        return (0.0, inset_line_dashes)
    return "solid"


def apply_inset_line_style(artist):
    """Apply optional rounded dashed style to inset/connector artists."""
    artist.set_linestyle(inset_line_style())

    if use_dashed_inset_lines:
        # Not all Matplotlib artists expose both methods, so keep this robust.
        if hasattr(artist, "set_dash_capstyle"):
            artist.set_dash_capstyle(inset_line_dash_capstyle)
        if hasattr(artist, "set_capstyle"):
            artist.set_capstyle(inset_line_dash_capstyle)
        if hasattr(artist, "set_dash_joinstyle"):
            artist.set_dash_joinstyle(inset_line_dash_joinstyle)
        if hasattr(artist, "set_joinstyle"):
            artist.set_joinstyle(inset_line_dash_joinstyle)

    return artist


def fmt_log_tick(val):
    exp = int(np.round(np.log10(abs(val))))
    if exp == -1:
        return r"$0.1$"
    if exp == 0:
        return r"$1$"
    if exp == 1:
        return r"$10$"
    return rf"$10^{{{exp}}}$"


def add_inset_colorbar(ax, mappable, ticks, ticklabels):
    cax = ax.inset_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    cb = plt.colorbar(mappable, cax=cax, orientation="vertical")
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklabels)

    cb.ax.tick_params(
        labelsize=fontsize_cbar_ticks,
        width=0,
        length=0,
        pad=4,
        colors="black",
        labelcolor="black",
    )

    cb.ax.minorticks_off()
    cb.outline.set_linewidth(1.5)
    cb.outline.set_edgecolor("black")

    for spine in cb.ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    return cb


def draw_scale_bar(ax):
    """Draw a |---| scale bar with label above it on the density panel."""
    if not show_scale_bar:
        return
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    xspan = xhi - xlo
    yspan = yhi - ylo

    bar_x    = scale_bar_x          if scale_bar_x          is not None else xlo + 0.5 * xspan
    bar_y    = scale_bar_y          if scale_bar_y          is not None else ylo + 0.10 * yspan
    capsize  = scale_bar_capsize    if scale_bar_capsize    is not None else 0.04 * yspan
    lpad     = scale_bar_label_pad  if scale_bar_label_pad  is not None else 0.02 * yspan
    fontsize = scale_bar_fontsize   if scale_bar_fontsize   is not None else fontsize_panel_label
    color    = scale_bar_color
    x0, x1  = bar_x - 0.5 * scale_bar_length, bar_x + 0.5 * scale_bar_length

    fx = ([pe.withStroke(linewidth=scale_bar_stroke_width, foreground=scale_bar_stroke)]
          if scale_bar_stroke is not None else [])

    ax.plot([x0, x1], [bar_y, bar_y], color=color, lw=scale_bar_lw,
            solid_capstyle="butt", transform=ax.transData, zorder=10)
    for xc in (x0, x1):
        ax.plot([xc, xc], [bar_y - 0.5 * capsize, bar_y + 0.5 * capsize],
                color=color, lw=scale_bar_lw, solid_capstyle="round",
                transform=ax.transData, zorder=10)
    ax.text(bar_x, bar_y + 0.5 * capsize + lpad, scale_bar_label,
            color=color, fontsize=fontsize, ha="center", va="bottom",
            transform=ax.transData, zorder=11, path_effects=fx)


def doppler_x(nu_Hz, T_K=T_doppler_K):
    delta_nu_D = nu0_Lya * np.sqrt(2.0 * kB * T_K / (mH * c_light**2))
    return (nu_Hz - nu0_Lya) / delta_nu_D


def doppler_velocity_frequency_kms(nu_Hz):
    """
    Old frequency-positive velocity offset relative to Lyα line center:

        Δv_ν = c (ν - ν0) / ν0.

    Positive values correspond to ν > ν0, i.e. the blue side.
    """
    return c_light * (nu_Hz - nu0_Lya) / nu0_Lya / 1e5


def doppler_velocity_wavelength_kms(nu_Hz):
    """
    Observer/red-positive velocity offset relative to Lyα line center:

        Δv_λ = c (λ - λ0) / λ0.

    Since λ = c/ν, the exact expression from frequency is

        Δv_λ = c (ν0/ν - 1).

    For small offsets this is approximately

        Δv_λ ≃ -c (ν - ν0) / ν0.

    Positive values correspond to λ > λ0, i.e. the red side.
    """
    return c_light * (nu0_Lya / nu_Hz - 1.0) / 1e5


def frequency_axis_values(nu_Hz):
    """Return the spectrum x-axis values in the selected units."""
    units = frequency_axis_units

    # Backward-compatible fallback to the old boolean option.
    if units is None:
        return doppler_x(nu_Hz) if use_doppler_x else nu_Hz

    if units == "x":
        return doppler_x(nu_Hz)
    if units in ("velocity_wavelength", "velocity_observer", "velocity_lambda"):
        return doppler_velocity_wavelength_kms(nu_Hz)
    if units in ("velocity_frequency", "velocity_nu", "velocity"):
        # "velocity" is kept as a backward-compatible alias for the old sign.
        return doppler_velocity_frequency_kms(nu_Hz)
    if units == "frequency":
        return nu_Hz

    raise ValueError(
        "frequency_axis_units must be 'x', 'velocity_wavelength', "
        "'velocity_frequency', 'frequency', or None, "
        f"got {units!r}."
    )


def default_frequency_xlabel():
    """Default x-label matching the selected spectrum frequency-axis units."""
    units = frequency_axis_units

    if units is None:
        return freq_xlabel if use_doppler_x else frequency_xlabel
    if units == "x":
        return freq_xlabel
    if units in ("velocity_wavelength", "velocity_observer", "velocity_lambda"):
        return velocity_wavelength_xlabel
    if units in ("velocity_frequency", "velocity_nu", "velocity"):
        # "velocity" is kept as a backward-compatible alias for the old sign.
        return velocity_frequency_xlabel
    if units == "frequency":
        return frequency_xlabel

    raise ValueError(
        "frequency_axis_units must be 'x', 'velocity_wavelength', "
        "'velocity_frequency', 'frequency', or None, "
        f"got {units!r}."
    )


def reorder_lya_edge_spectra(J_edge_raw, t, nuC):
    """
    Reorder J_Lya_EdgeSpectra to shape (Nt, N_edge, Nν).
    """
    if J_edge_raw.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {J_edge_raw.shape}")

    Nt = t.size
    Nν = nuC.size

    shape = J_edge_raw.shape
    time_axes = [ax for ax, s in enumerate(shape) if s == Nt]
    nu_axes   = [ax for ax, s in enumerate(shape) if s == Nν]

    if len(time_axes) != 1:
        raise RuntimeError(
            f"Could not uniquely identify time axis in J_edge_raw: "
            f"shape={J_edge_raw.shape}, Nt={Nt}"
        )

    if len(nu_axes) != 1:
        raise RuntimeError(
            f"Could not uniquely identify frequency axis in J_edge_raw: "
            f"shape={J_edge_raw.shape}, Nν={Nν}"
        )

    t_ax = time_axes[0]
    nu_ax = nu_axes[0]
    edge_ax = 3 - t_ax - nu_ax

    J_reordered = np.transpose(J_edge_raw, axes=[t_ax, edge_ax, nu_ax])

    if J_reordered.shape[0] != Nt or J_reordered.shape[2] != Nν:
        raise RuntimeError(
            f"Bad reordered shape: {J_reordered.shape}, expected "
            f"(Nt={Nt}, N_edge, Nν={Nν})"
        )

    return J_reordered


def convert_saved_indices_from_save_code(ridx_raw, zidx_raw, Nr, Nz):
    """
    Convert edge-spectrum indices using SaveHDF5Data_RHD_StarCluster.jl.

    The save code writes Julia 1-based cell indices directly.

        right:  i_r = N_r - 1, i_Z = 2:(N_Z - 1)
        top:    i_Z = N_Z - 1, i_r = 2:(N_r - 2)
        bottom: i_Z = 2,       i_r = 2:(N_r - 2)

    Therefore in Python:

        i_cell = i_r - 1
        j_cell = i_Z - 1

    So saved right/top spectra are one cell inside the outermost Python cell.
    """
    ridx_raw = np.asarray(ridx_raw, dtype=int)
    zidx_raw = np.asarray(zidx_raw, dtype=int)

    if ridx_raw.shape != zidx_raw.shape:
        raise RuntimeError(
            f"rIndex and ZIndex have different shapes: {ridx_raw.shape} vs {zidx_raw.shape}"
        )

    if edge_index_physical_order == "rZ":
        i_cell = ridx_raw - 1
        j_cell = zidx_raw - 1
        ridx_phys = ridx_raw
        zidx_phys = zidx_raw
    elif edge_index_physical_order == "Zr":
        # Only use this if the save code really wrote the two index arrays swapped.
        i_cell = zidx_raw - 1
        j_cell = ridx_raw - 1
        ridx_phys = zidx_raw
        zidx_phys = ridx_raw
    else:
        raise ValueError("edge_index_physical_order must be 'rZ' or 'Zr'")

    if np.any(i_cell < 0) or np.any(i_cell >= Nr):
        raise RuntimeError(
            f"Converted r indices outside array: min={i_cell.min()}, max={i_cell.max()}, allowed=[0,{Nr-1}]."
        )

    if np.any(j_cell < 0) or np.any(j_cell >= Nz):
        raise RuntimeError(
            f"Converted Z indices outside array: min={j_cell.min()}, max={j_cell.max()}, allowed=[0,{Nz-1}]."
        )

    edge_boundary = np.full(ridx_raw.shape, "unknown", dtype=object)

    # Classify using raw Julia indices, exactly matching the save loops.
    right_mask  = (ridx_phys == Nr - 1) & (zidx_phys >= 2) & (zidx_phys <= Nz - 1)
    top_mask    = (zidx_phys == Nz - 1) & (ridx_phys >= 2) & (ridx_phys <= Nr - 2)
    bottom_mask = (zidx_phys == 2)      & (ridx_phys >= 2) & (ridx_phys <= Nr - 2)

    # Save order is right, then top, then bottom.  If there is overlap, keep right.
    edge_boundary[right_mask] = "right"
    edge_boundary[(edge_boundary == "unknown") & top_mask] = "top"
    edge_boundary[(edge_boundary == "unknown") & bottom_mask] = "bottom"

    bad = np.where(edge_boundary == "unknown")[0]
    if bad.size > 0:
        first_bad = [(int(ridx_phys[q]), int(zidx_phys[q])) for q in bad[:10]]
        raise RuntimeError(
            "Some saved edge indices do not match the save-code convention. "
            f"First bad physical Julia (i_r,i_Z) pairs: {first_bad}"
        )

    n_right  = int(np.count_nonzero(edge_boundary == "right"))
    n_top    = int(np.count_nonzero(edge_boundary == "top"))
    n_bottom = int(np.count_nonzero(edge_boundary == "bottom"))

    print("  Edge spectra are interpreted directly from SaveHDF5Data_RHD_StarCluster.jl:")
    print(f"    edge_index_physical_order = {edge_index_physical_order!r}")
    print("    Python i_cell = physical Julia r-index - 1")
    print("    Python j_cell = physical Julia Z-index - 1")
    print(f"    right/top/bottom counts = {n_right}/{n_top}/{n_bottom}")
    print(f"    right saved at Python i={Nr-2}; top at Python j={Nz-2}; bottom at Python j=1")

    return i_cell.astype(int), j_cell.astype(int), edge_boundary


def boundary_mask(edge_boundary, boundary):
    """Return a mask for a physical saved boundary."""
    if boundary == "left":
        raise RuntimeError(
            "No left/axis Lyα edge spectra are saved by SaveHDF5Data_RHD_StarCluster.jl."
        )

    if boundary not in ("right", "top", "bottom"):
        raise ValueError(f"Unknown boundary={boundary!r}")

    return edge_boundary == boundary


def choose_edge_spectrum(snap, boundary, pos_frac, pos_mode):
    """
    Choose one saved edge spectrum on a physical saved boundary.

    Returns:
        edge_idx, i_cell, j_cell, r_pc, Z_pc
    """
    i_edge = snap["i_edge"]
    j_edge = snap["j_edge"]
    edge_boundary = snap["edge_boundary"]
    rCoM = snap["rCoM"]
    ZCoM = snap["ZCoM"]

    mask = boundary_mask(edge_boundary, boundary)
    edge_indices = np.where(mask)[0]

    if edge_indices.size == 0:
        raise RuntimeError(
            f"No saved edge spectra found on boundary={boundary!r}. "
            f"Available boundaries={np.unique(edge_boundary)}"
        )

    if pos_mode == "peak":
        totals = np.sum(
            snap["J_snap"][edge_indices, :] * snap["dnu"][None, :],
            axis=1,
        )
        local_best = np.argmax(totals)

    elif pos_mode == "frac":
        pos_frac = min(max(pos_frac, 0.0), 1.0)

        if boundary == "right":
            coords = ZCoM[j_edge[edge_indices]]
        elif boundary in ("top", "bottom"):
            coords = rCoM[i_edge[edge_indices]]
        else:
            raise ValueError(f"Unknown boundary={boundary!r}")

        # Use the coordinate interval actually saved on that boundary.
        target = coords.min() + pos_frac * (coords.max() - coords.min())
        local_best = np.argmin(np.abs(coords - target))

    else:
        raise ValueError(f"Unknown pos_mode={pos_mode!r}")

    edge_idx = int(edge_indices[local_best])
    i_cell = int(i_edge[edge_idx])
    j_cell = int(j_edge[edge_idx])

    r_pc = rCoM[i_cell] / pc
    Z_pc = ZCoM[j_cell] / pc

    return edge_idx, i_cell, j_cell, r_pc, Z_pc


def clamp_panel_loc(left, bottom, width, height):
    """
    Keep floating spectrum axes inside the figure canvas.
    """
    left = min(max(left, 0.01), 0.99 - width)
    bottom = min(max(bottom, 0.01), 0.99 - height)
    return [left, bottom, width, height]


def panel_setting(sp_layout, key, default):
    """
    Read an optional spectra-panel layout setting.

    Values inside an individual spectra_panels dict override the global
    settings above. This lets you move one panel without changing the others.
    """
    if sp_layout is None:
        return default
    return sp_layout.get(key, default)


def spectrum_panel_loc(boundary, col_left, r_cell_pc, Z_cell_pc,
                       r_min_pc, r_max_pc, Z_min_pc, Z_max_pc,
                       sp_layout=None, dens_bbox=None):
    """
    Put the spectrum panel on the same physical side as the sampled boundary.

    The baseline placement is deterministic and based on the sampled physical
    coordinates. The global/per-panel layout controls then only move the
    plotted inset panel on the page; they do not change the sampled spectrum.

    Important: because ax_dens uses aspect="equal", Matplotlib can shrink the
    visible nHI panel inside the originally requested add_axes rectangle. The
    spectra panels are therefore aligned to dens_bbox=ax_dens.get_position(),
    i.e. the ACTUAL visible nHI panel.
    """
    width  = panel_setting(sp_layout, "panel_width",  spec_width)
    height = panel_setting(sp_layout, "panel_height", spec_height)

    if dens_bbox is None:
        dens_left_actual   = col_left
        dens_bottom_actual = dens_bottom
        dens_width_actual  = dens_width
        dens_height_actual = dens_height
    else:
        dens_left_actual   = dens_bbox.x0
        dens_bottom_actual = dens_bbox.y0
        dens_width_actual  = dens_bbox.width
        dens_height_actual = dens_bbox.height

    dens_right_actual = dens_left_actual + dens_width_actual
    dens_top_actual   = dens_bottom_actual + dens_height_actual

    # Absolute placement override, useful when final-tuning a paper figure.
    # Use figure coordinates: panel_left=0.45, panel_bottom=0.52, etc.
    if sp_layout is not None and ("panel_left" in sp_layout or "panel_bottom" in sp_layout):
        left_default, bottom_default, _, _ = spectrum_panel_loc(
            boundary,
            col_left,
            r_cell_pc,
            Z_cell_pc,
            r_min_pc,
            r_max_pc,
            Z_min_pc,
            Z_max_pc,
            sp_layout={},
            dens_bbox=dens_bbox,
        )
        left   = panel_setting(sp_layout, "panel_left",   left_default)
        bottom = panel_setting(sp_layout, "panel_bottom", bottom_default)
        return clamp_panel_loc(left, bottom, width, height)

    if boundary == "right":

        gap = panel_setting(sp_layout, "panel_gap", right_panel_gap)
        dx  = panel_setting(sp_layout, "panel_dx",  right_panel_dx)
        dy  = panel_setting(sp_layout, "panel_dy",  right_panel_dy)

        left = dens_right_actual + gap + dx

        layout_mode = panel_setting(
            sp_layout,
            "right_panel_layout_mode",
            right_panel_layout_mode,
        )

        if layout_mode == "even":

            slot  = panel_setting(sp_layout, "right_panel_stack_slot", 0)
            count = panel_setting(sp_layout, "right_panel_stack_count", 1)

            stack_bottom = panel_setting(
                sp_layout,
                "right_panel_stack_bottom",
                right_panel_stack_bottom,
            )
            stack_top = panel_setting(
                sp_layout,
                "right_panel_stack_top",
                right_panel_stack_top,
            )

            # Default: define the stack using fractions of the ACTUAL visible
            # density panel, not the nominal requested axes rectangle.
            if stack_bottom is None:
                bottom_frac = panel_setting(
                    sp_layout,
                    "right_panel_stack_bottom_frac",
                    right_panel_stack_bottom_frac,
                )
                stack_bottom = dens_bottom_actual + bottom_frac * dens_height_actual

            if stack_top is None:
                top_frac = panel_setting(
                    sp_layout,
                    "right_panel_stack_top_frac",
                    right_panel_stack_top_frac,
                )
                stack_top = dens_bottom_actual + top_frac * dens_height_actual

            if count <= 1:
                bottom = 0.5 * (stack_bottom + stack_top) - 0.5 * height
            else:
                bottom_low  = stack_bottom
                bottom_high = stack_top - height
                bottom = bottom_low + slot * (bottom_high - bottom_low) / (count - 1)

            bottom += dy

            return clamp_panel_loc(left, bottom, width, height)

        if layout_mode == "physical":

            frac = (Z_cell_pc - Z_min_pc) / (Z_max_pc - Z_min_pc)

            bottom_min = panel_setting(
                sp_layout,
                "panel_bottom_min",
                right_panel_bottom_min,
            )
            bottom_max = panel_setting(
                sp_layout,
                "panel_bottom_max",
                right_panel_bottom_max,
            )

            bottom = dens_bottom_actual + frac * dens_height_actual - 0.5 * height + dy
            bottom = min(max(bottom, bottom_min), bottom_max)

            return clamp_panel_loc(left, bottom, width, height)

        raise ValueError(
            "right_panel_layout_mode must be 'even' or 'physical', "
            f"got {layout_mode!r}."
        )

    if boundary == "left":
        frac = (Z_cell_pc - Z_min_pc) / (Z_max_pc - Z_min_pc)

        gap        = panel_setting(sp_layout, "panel_gap",        left_panel_gap)
        dx         = panel_setting(sp_layout, "panel_dx",         left_panel_dx)
        dy         = panel_setting(sp_layout, "panel_dy",         left_panel_dy)
        bottom_min = panel_setting(sp_layout, "panel_bottom_min", left_panel_bottom_min)
        bottom_max = panel_setting(sp_layout, "panel_bottom_max", left_panel_bottom_max)

        left = dens_left_actual - gap - width + dx
        bottom = dens_bottom_actual + frac * dens_height_actual - 0.5 * height + dy
        bottom = min(max(bottom, bottom_min), bottom_max)

        return clamp_panel_loc(left, bottom, width, height)

    if boundary == "top":
        frac = (r_cell_pc - r_min_pc) / (r_max_pc - r_min_pc)

        gap      = panel_setting(sp_layout, "panel_gap",      top_panel_gap)
        dx       = panel_setting(sp_layout, "panel_dx",       top_panel_dx)
        dy       = panel_setting(sp_layout, "panel_dy",       top_panel_dy)
        left_min = panel_setting(sp_layout, "panel_left_min", dens_left_actual if top_panel_left_min is None else top_panel_left_min)
        left_max = panel_setting(sp_layout, "panel_left_max", dens_right_actual - width if top_panel_left_max is None else top_panel_left_max)

        left = dens_left_actual + frac * dens_width_actual - 0.5 * width + dx
        left = min(max(left, left_min), left_max)
        bottom = dens_top_actual + gap + dy

        return clamp_panel_loc(left, bottom, width, height)

    if boundary == "bottom":
        frac = (r_cell_pc - r_min_pc) / (r_max_pc - r_min_pc)

        gap      = panel_setting(sp_layout, "panel_gap",      bottom_panel_gap)
        dx       = panel_setting(sp_layout, "panel_dx",       bottom_panel_dx)
        dy       = panel_setting(sp_layout, "panel_dy",       bottom_panel_dy)
        left_min = panel_setting(sp_layout, "panel_left_min", dens_left_actual if bottom_panel_left_min is None else bottom_panel_left_min)
        left_max = panel_setting(sp_layout, "panel_left_max", dens_right_actual - width if bottom_panel_left_max is None else bottom_panel_left_max)

        left = dens_left_actual + frac * dens_width_actual - 0.5 * width + dx
        left = min(max(left, left_min), left_max)
        bottom = dens_bottom_actual - gap - height + dy

        return clamp_panel_loc(left, bottom, width, height)

    raise ValueError(f"Unknown boundary={boundary!r}")

def rectangle_corners(r_min, r_max, Z_min, Z_max):
    """
    Corners in physical display coordinates x=r, y=Z.
    1=upper-right, 2=upper-left, 3=lower-left, 4=lower-right.
    """
    return {
        1: (r_max, Z_max),
        2: (r_min, Z_max),
        3: (r_min, Z_min),
        4: (r_max, Z_min),
    }


def connector_pairs_for_boundary(boundary):
    """
    Auto connector corners based on physical side.
    Parent rectangle corners use data coordinates.
    Spectrum corners use axes coordinates:
        1=upper-right, 2=upper-left, 3=lower-left, 4=lower-right.
    """
    if boundary == "right":
        return [(1, 2), (4, 3)]

    if boundary == "left":
        return [(2, 1), (3, 4)]

    if boundary == "top":
        return [(2, 3), (1, 4)]

    if boundary == "bottom":
        return [(3, 2), (4, 1)]

    raise ValueError(f"Unknown boundary={boundary!r}")


###############################################################
#######       L O A D  D A T A  (all snapshots)        #######
###############################################################

snapshots = []

for filename, target_kyr in zip(Filenames, TargetTimes_kyr):
    with h5py.File(filename, "r") as f:

        r    = f["r"][:]
        Z    = f["Z"][:]
        rCoM = f["rCoM"][:]
        ZCoM = f["ZCoM"][:]
        t    = f["Times"][:]

        k = nearest_saved_index(t, target_kyr * kyr)

        print(
            f"  {filename}: target={target_kyr} kyr, "
            f"found t={t[k]/kyr:.3f} kyr, index={k}"
        )

        Nr = len(r) - 1
        Nz = len(Z) - 1

        if len(rCoM) != Nr:
            raise RuntimeError(f"len(rCoM)={len(rCoM)} but Nr={Nr}")
        if len(ZCoM) != Nz:
            raise RuntimeError(f"len(ZCoM)={len(ZCoM)} but Nz={Nz}")

        print(f"  History spatial order after moving time axis: {history_spatial_order_after_time_axis!r}")
        print(f"  Edge index physical order: {edge_index_physical_order!r}")

        rho_hist  = reorder_history_3d(f["GasDensity_History"][:], t, r, Z)
        xHII_hist = reorder_history_3d(f["xHII_History"][:],       t, r, Z)
        xH2_hist  = reorder_history_3d(f["xH2_History"][:],        t, r, Z)

        rho  = rho_hist[k]
        xHII = xHII_hist[k]
        xH2  = xH2_hist[k]

        nH  = X * rho / mH
        xHI = np.clip(1.0 - xHII - 2.0 * xH2, 0.0, 1.0)
        nHI = xHI * nH

        nuC = f["νC"][:]
        dnu = f["Δν"][:]

        J_edge_raw = f["J_Lya_EdgeSpectra"][:]
        ridx_raw   = f["J_Lya_Edge_rIndex"][:]
        zidx_raw   = f["J_Lya_Edge_ZIndex"][:]

        J_edge_hist = reorder_lya_edge_spectra(J_edge_raw, t, nuC)
        J_snap = J_edge_hist[k, :, :]

    i_edge, j_edge, edge_boundary = convert_saved_indices_from_save_code(ridx_raw, zidx_raw, Nr, Nz)

    snapshots.append(dict(
        filename=filename,
        r=r,
        Z=Z,
        rCoM=rCoM,
        ZCoM=ZCoM,
        t_kyr=t[k] / kyr,
        nH=nH,
        nHI=nHI,
        nuC=nuC,
        dnu=dnu,
        J_snap=J_snap,
        i_edge=i_edge,
        j_edge=j_edge,
        edge_boundary=edge_boundary,
        Nr=Nr,
        Nz=Nz,
    ))


###############################################################
#######       P R E P A R E  S P E C T R A            #######
###############################################################

# Select all spectra before drawing, so optional shared normalization/y-scale
# is based on exactly the spectra that will appear in the figure.
all_selected_peaks = []

for snap in snapshots:

    nuC = snap["nuC"]
    xval_full = frequency_axis_values(nuC)

    snap["selected_spectra"] = []

    for sp in spectra_panels:

        boundary = sp["boundary"]
        pos_frac = sp.get("pos_frac", 0.5)
        pos_mode = sp.get("pos_mode", "frac")

        edge_idx, i_cell, j_cell, r_cell_pc, Z_cell_pc = choose_edge_spectrum(
            snap,
            boundary,
            pos_frac,
            pos_mode,
        )

        spec_raw = snap["J_snap"][edge_idx, :].copy()

        if spec_xlim is not None:
            xlo, xhi = spec_xlim
            mask = (xval_full >= xlo) & (xval_full <= xhi)
            xplot = xval_full[mask]
            splot_raw = spec_raw[mask]
        else:
            xplot = xval_full
            splot_raw = spec_raw

        # The wavelength/red-positive velocity conversion reverses the
        # ordering if νC is increasing.  Sort before plotting/filling so the
        # spectrum is drawn left-to-right in the displayed coordinate.
        if xplot.size > 1:
            order = np.argsort(xplot)
            xplot = xplot[order]
            splot_raw = splot_raw[order]

        if xplot.size == 0:
            raise RuntimeError(
                f"No spectral data inside spec_xlim={spec_xlim}. "
                f"Check Doppler conversion or x-limits."
            )

        peak = max(float(np.nanmax(splot_raw)), 0.0)
        all_selected_peaks.append(peak)

        snap["selected_spectra"].append(dict(
            boundary=boundary,
            pos_frac=pos_frac,
            pos_mode=pos_mode,
            edge_idx=edge_idx,
            i_cell=i_cell,
            j_cell=j_cell,
            r_cell_pc=r_cell_pc,
            Z_cell_pc=Z_cell_pc,
            xplot=xplot,
            splot_raw=splot_raw,
            peak=peak,
            x_label=sp.get("x_label", default_frequency_xlabel() if default_spec_x_label is None else default_spec_x_label),
            show_xticks=sp.get("show_xticks", default_show_spec_xticks),
            show_xticklabels=sp.get("show_xticklabels", default_show_spec_xticklabels),
            show_xlabel=sp.get("show_xlabel", default_show_spec_xlabel),
            x_tick_values=sp.get("x_tick_values", default_spec_x_tick_values),
            x_tick_labels=sp.get("x_tick_labels", default_spec_x_tick_labels),
            layout=sp,
            fill_under=sp.get("fill_under", fill_under_spectra),
            fill_color=sp.get("fill_color", spec_fill_color),
            fill_alpha=sp.get("fill_alpha", spec_fill_alpha),
        ))

    # For right-boundary spectra, assign visual stack slots.
    # This affects only where panels are drawn, not where spectra are sampled.
    # Slot 0 is the lowest-Z right panel.
    right_specs = [
        sp_data for sp_data in snap["selected_spectra"]
        if sp_data["boundary"] == "right"
    ]

    right_specs.sort(key=lambda sp_data: sp_data["Z_cell_pc"])

    for slot, sp_data in enumerate(right_specs):
        sp_data["layout"]["right_panel_stack_slot"]  = slot
        sp_data["layout"]["right_panel_stack_count"] = len(right_specs)

if len(all_selected_peaks) == 0:
    raise RuntimeError("No spectra were selected for plotting.")

shared_spectrum_peak = max(max(all_selected_peaks), 1e-30)

if normalise_spectra and shared_spectrum_y_scale:
    print(f"  Shared spectrum normalization peak = {shared_spectrum_peak:.6e}")

for snap in snapshots:
    for sp_data in snap["selected_spectra"]:

        if normalise_spectra:
            if shared_spectrum_y_scale:
                norm_peak = shared_spectrum_peak
            else:
                norm_peak = max(sp_data["peak"], 1e-30)
            sp_data["splot"] = sp_data["splot_raw"] / norm_peak
        else:
            sp_data["splot"] = sp_data["splot_raw"]

if normalise_spectra and shared_spectrum_y_scale:
    shared_spectrum_ymax = spectrum_ymax_factor
else:
    shared_spectrum_ymax = None


###############################################################
#######       F I G U R E  &  A X E S                  #######
###############################################################

fig = plt.figure(figsize=figsize, facecolor=figure_facecolor)
col_lefts = [col_left_0 + i * (dens_width + col_gap) for i in range(n_snapshots)]

if density_field == "nHI":
    norm_dens  = LogNorm(vmin=nHI_vmin, vmax=nHI_vmax)
    plot_label = "$n_{HI}$ (cm$^{-3}$)"
else:
    norm_dens  = LogNorm(vmin=nH_vmin, vmax=nH_vmax)
    plot_label = "$n_{H}$ (cm$^{-3}$)"

cmap_dens_obj = plt.colormaps[cmap_density].copy()

logmin = int(np.ceil(np.log10(norm_dens.vmin)))
logmax = int(np.floor(np.log10(norm_dens.vmax)))
cbar_ticks  = 10.0 ** np.arange(logmin, logmax + 1)
cbar_labels = [fmt_log_tick(v) for v in cbar_ticks]


###############################################################
#######       D R A W  S N A P S H O T S              #######
###############################################################

for col_idx, (snap, col_left) in enumerate(zip(snapshots, col_lefts)):

    r    = snap["r"]
    Z    = snap["Z"]
    rCoM = snap["rCoM"]
    ZCoM = snap["ZCoM"]
    Nr   = snap["Nr"]
    Nz   = snap["Nz"]

    r_pc = r / pc
    Z_pc = Z / pc

    r_min_pc = r_pc[0]
    Z_min_pc = Z_pc[0]
    r_max_pc = xmax_fraction * r_pc[-1]
    Z_max_pc = zmax_fraction * Z_pc[-1]

    field_2d = snap["nHI"] if density_field == "nHI" else snap["nH"]

    # This is the only density orientation used:
    #   x = r
    #   y = Z
    #   field[i,j] -> (rCoM[i], ZCoM[j])
    dens_C = field_for_physical_rZ_plot(field_2d, r, Z)

    ax_dens = make_ax(fig, col_left, dens_bottom, dens_width, dens_height)
    ax_dens.set_facecolor(density_panel_facecolor)

    im = ax_dens.pcolormesh(
        r_pc,
        Z_pc,
        dens_C,
        norm=norm_dens,
        cmap=cmap_dens_obj,
        shading="auto",
        rasterized=rasterized,
    )

    ax_dens.set_xlim(r_min_pc, r_max_pc)
    ax_dens.set_ylim(Z_min_pc, Z_max_pc)
    ax_dens.set_aspect("equal", adjustable="box")
    ax_dens.set_xticks([])
    ax_dens.set_yticks([])
    ax_dens.tick_params(
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    for spine in ax_dens.spines.values():
        spine.set_visible(True)
        spine.set_color(panel_outline_color)
        spine.set_linewidth(panel_frame_lw)

    ax_dens.text(
        0.96,
        0.96,
        plot_label,
        transform=ax_dens.transAxes,
        ha="right",
        va="top",
        color=density_label_color,
        fontsize=fontsize_panel_label,
        path_effects=[
            pe.withStroke(
                linewidth=density_label_stroke_width,
                foreground=density_label_stroke,
            )
        ],
    )

    ax_dens.text(
        0.04,
        0.96,
        rf"$t = {snap['t_kyr']:.1f}$ kyr",
        transform=ax_dens.transAxes,
        ha="left",
        va="top",
        color=time_label_color,
        fontsize=fontsize_time,
        path_effects=[
            pe.withStroke(
                linewidth=time_label_stroke_width,
                foreground=time_label_stroke,
            )
        ],
    )

    if col_idx == n_snapshots - 1 and show_colorbar:
        add_inset_colorbar(ax_dens, im, cbar_ticks, cbar_labels)

    if show_scale_bar:
        draw_scale_bar(ax_dens)

    fig.canvas.draw()
    dens_bbox = ax_dens.get_position()

    for sp_data in snap["selected_spectra"]:

        boundary = sp_data["boundary"]
        pos_frac = sp_data["pos_frac"]
        pos_mode = sp_data["pos_mode"]

        edge_idx   = sp_data["edge_idx"]
        i_cell     = sp_data["i_cell"]
        j_cell     = sp_data["j_cell"]
        r_cell_pc  = sp_data["r_cell_pc"]
        Z_cell_pc  = sp_data["Z_cell_pc"]
        xplot      = sp_data["xplot"]
        splot      = sp_data["splot"]
        x_label          = sp_data["x_label"]
        show_xticks      = sp_data["show_xticks"]
        show_xticklabels = sp_data["show_xticklabels"]
        show_xlabel      = sp_data["show_xlabel"]
        x_tick_values    = sp_data["x_tick_values"]
        x_tick_labels    = sp_data["x_tick_labels"]
        sp_layout        = sp_data["layout"]
        fill_under       = sp_data["fill_under"]
        fill_color       = sp_data["fill_color"]
        fill_alpha       = sp_data["fill_alpha"]

        if shared_spectrum_ymax is not None:
            ymax = shared_spectrum_ymax
        else:
            ymax = max(float(np.nanmax(splot)), 1e-30) * spectrum_ymax_factor

        print(
            f"  Spectrum boundary={boundary:>6s}, mode={pos_mode:>4s}: "
            f"edge_idx={edge_idx}, i={i_cell}, j={j_cell}, "
            f"rCoM={r_cell_pc:.4f} pc, ZCoM={Z_cell_pc:.4f} pc, "
            f"nHI_cell={snap['nHI'][i_cell, j_cell]:.3e}, "
            f"raw_peak={sp_data['peak']:.3e}"
        )

        fig_loc = spectrum_panel_loc(
            boundary,
            col_left,
            r_cell_pc,
            Z_cell_pc,
            r_min_pc,
            r_max_pc,
            Z_min_pc,
            Z_max_pc,
            sp_layout=sp_layout,
            dens_bbox=dens_bbox,
        )

        ax_spec = fig.add_axes(fig_loc, facecolor=spectra_panel_facecolor)

        if fill_under:
            ax_spec.fill_between(
                xplot,
                0.0,
                splot,
                color=fill_color,
                alpha=fill_alpha,
                linewidth=0.0,
                zorder=1,
            )

        ax_spec.plot(xplot, splot, color=spec_line_color, lw=spec_line_lw, zorder=2)
        ax_spec.set_xlim(spec_xlim if spec_xlim is not None else (xplot[0], xplot[-1]))
        ax_spec.set_ylim(0.0, ymax)
        ax_spec.set_yticks([])
        ax_spec.set_yticklabels([])
        ax_spec.tick_params(which="both", left=False, right=False, labelleft=False, colors=spectra_tick_color)

        if x_tick_values is not None:
            ax_spec.set_xticks(x_tick_values)
        if x_tick_labels is not None:
            ax_spec.set_xticklabels(x_tick_labels)

        ax_spec.tick_params(
            axis="x",
            which="both",
            top=False,
            labeltop=False,
            bottom=show_xticks,
            labelbottom=show_xticklabels,
            labelsize=fontsize_spec_label,
            direction="in",
            colors=spectra_tick_color,
        )

        if not show_xticks:
            ax_spec.tick_params(axis="x", which="both", bottom=False)
        if not show_xticklabels:
            ax_spec.tick_params(axis="x", which="both", labelbottom=False)

        if show_xlabel:
            if x_label is not None:
                ax_spec.set_xlabel(x_label, fontsize=fontsize_spec_label, color=spectra_label_color)
            else:
                ax_spec.set_xlabel(default_frequency_xlabel(), fontsize=fontsize_spec_label, color=spectra_label_color)

        for spine in ax_spec.spines.values():
            spine.set_edgecolor(active_spec_frame_color)
            spine.set_linewidth(spec_frame_lw)

        if boundary in ("right", "left"):
            label_str = rf"$Z = {Z_cell_pc:.2f}$ pc"
        else:
            label_str = rf"$r = {r_cell_pc:.2f}$ pc"

        ax_spec.text(
            0.5,
            0.88,
            label_str,
            transform=ax_spec.transAxes,
            ha="right",
            va="top",
            fontsize=fontsize_spec_label * 0.85,
            color=spectra_label_color,
        )

        # Draw the actual selected cell using interfaces.
        # Do not use rCoM +/- Δr/2; rCoM is cylindrical-volume-centered.
        rect_r_min = r_pc[i_cell]
        rect_r_max = r_pc[i_cell + 1]
        rect_Z_min = Z_pc[j_cell]
        rect_Z_max = Z_pc[j_cell + 1]

        rect = Rectangle(
            (rect_r_min, rect_Z_min),
            rect_r_max - rect_r_min,
            rect_Z_max - rect_Z_min,
            linewidth=connector_lw,
            edgecolor=active_connector_color,
            facecolor="none",
            zorder=10,
        )
        apply_inset_line_style(rect)
        ax_dens.add_patch(rect)

        fig.canvas.draw()

        rect_corners = rectangle_corners(
            rect_r_min,
            rect_r_max,
            rect_Z_min,
            rect_Z_max,
        )

        spec_corners = {
            1: (1.0, 1.0),
            2: (0.0, 1.0),
            3: (0.0, 0.0),
            4: (1.0, 0.0),
        }

        for p_corner, s_corner in connector_pairs_for_boundary(boundary):
            con = ConnectionPatch(
                xyA=rect_corners[p_corner],
                coordsA=ax_dens.transData,
                xyB=spec_corners[s_corner],
                coordsB=ax_spec.transAxes,
                color=active_connector_color,
                lw=connector_lw,
                zorder=15,
                clip_on=False,
            )
            apply_inset_line_style(con)
            fig.add_artist(con)


###############################################################
#######       S A V E                                  #######
###############################################################

plt.savefig(FigureName, bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
print(f"Saved figure to {FigureName}")