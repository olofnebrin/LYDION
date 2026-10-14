import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import matplotlib.patheffects as pe
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
cmastro_src = repo_root / "src" / "cmastro" / "src"
sys.path.insert(0, str(cmastro_src))

try:
    import cmastro
except Exception:
    cmastro = None


# ===================================================================
# User inputs
# ===================================================================

Filename   = "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5"
FigureName = "LYDION_Figure_LyaVsGasPressureForce_512D0.01.pdf"

# Three requested snapshots, in kyr.
TargetTimes_kyr = [5.0, 10.0, 15.0]

# -------------------------------------------------------------------
# Figure / layout
# -------------------------------------------------------------------
figsize       = (34, 18)       # (width, height) in inches
xmax_fraction = 0.60
zmax_fraction = 0.60
rasterized    = True

# Manual panel layout: two rows × three columns.
col_left_0   = 0.030
col_gap      = -0.06
col_width    = 0.310
row_bottom_0 = 0.055
row_gap      = 0.02
row_height   = 0.450

# -------------------------------------------------------------------
# Text sizes
# -------------------------------------------------------------------
fontsize_cbar_ticks  = 34
fontsize_panel_label = 48
fontsize_row_label   = 52
fontsize_time        = 44

# -------------------------------------------------------------------
# Colour scale
# -------------------------------------------------------------------
# Both rows are in dyn cm^-3 and share one colour scale.
cmap_force = "Spectral"       # easy to change; e.g. "magma", "cma:laguna", "inferno"
#cmap_force = "cma:unph"
#cmap_force = "magma"

# Signed anti-gravity force component, shown on a symmetric log scale.
# Positive values act opposite to gravity; negative values act along gravity.
# For SymLogNorm, force_vmin/force_vmax should usually be symmetric, e.g.
# -1e-22 and +1e-22. Set both to None for automatic symmetric limits.
#force_vmin = -3e-23              # lower signed limit, dyn cm^-3
#force_vmax =  3e-23              # upper signed limit, dyn cm^-3
force_vmin = None              # lower signed limit, dyn cm^-3
force_vmax =  None              # upper signed limit, dyn cm^-3
percentile_vmin = 1.0
percentile_vmax = 99.7

# SymLogNorm is the log-like scale that can show negative values.
# |force| > force_linthresh is logarithmic, while values close to zero
# are shown with a small linear region so the sign change is well behaved.
use_symlognorm   = True
force_linthresh  = 1e-26

# Masking thresholds. Units: dyn cm^-3.
f_grav_mask_threshold = 0.0
force_mask_threshold  = 0.0
mask_color            = "black"

# -------------------------------------------------------------------
# Optional Lyα force-fraction row
# -------------------------------------------------------------------
# This row shows the percentage of the total outward projected force
# contributed by Lyα radiation pressure,
#
#   100 |f_Lya,anti| / (|f_Lya,anti| + |f_gas,anti|).
#
# The absolute values make the plotted percentage positive even if one
# of the signed anti-gravity projections is negative in a cell.
show_Lya_force_fraction_row = False

# The original two-row layout is kept when show_Lya_force_fraction_row=False.
# When the extra row is enabled, these settings give a three-row layout.
figsize_with_fraction_row     = (34, 25)
row_height_with_fraction_row  = 0.295

cmap_fraction = "viridis"
fraction_vmin = 0.0                 # percent
fraction_vmax = 100.0               # percent
fraction_mask_denominator_below = 0.0

show_fraction_colorbar = True
fraction_colorbar_panel = (2, 0)
fraction_cbar_label = r"Ly$\alpha$ contribution [\%]"
fraction_ticks = [0, 25, 50, 75, 100]
fraction_ticklabels = [r"$0$", r"$25$", r"$50$", r"$75$", r"$100$"]

# -------------------------------------------------------------------
# Force integrals printed to terminal
# -------------------------------------------------------------------
print_force_integrals = True

# Temperature mask used only for the gas-pressure/photoionization
# force integral.  The plotted gas-pressure force field is unchanged.
use_T_mask_for_gas_force_integral = False
T_threshold = 1000.0                 # K

# Optional factor for symmetry reconstruction, if your saved domain is
# only half of the physical domain.  Keep at 1.0 if CellVolumes already
# corresponds to the domain you want to integrate.
force_integral_symmetry_factor = 1.0

# Set to True only if the HDF5 L_bol value is stored in L_sun rather than
# cgs erg s^-1.
L_bol_in_solar_units = False

# -------------------------------------------------------------------
# Colorbar: inset inside one panel, shared for all panels
# -------------------------------------------------------------------
show_colorbar      = True
colorbar_panel     = (0, 0)       # middle panel in the top row
cbar_x             = 0.67
cbar_y             = 0.33
cbar_width         = 0.045
cbar_height        = 0.6
cbar_tick_length   = 0
cbar_tick_width    = 0
cbar_tick_pad      = 4
cbar_tick_color    = "black"      # tick labels now black
cbar_outline_color = "black"      # optional, but usually looks better too
cbar_label         = r"dyn cm$^{-3}$"
cbar_label_size    = 34

# Optional stroke for the colorbar label. Set width=0 or stroke=None for no stroke.
cbar_label_stroke       = None
cbar_label_stroke_width = 0.0

# Custom ticks/labels. Set to None for automatic decade ticks.
ticks_force_custom      = None
ticklabels_force_custom = None

# -------------------------------------------------------------------
# Panel labels and row labels
# -------------------------------------------------------------------
panel_label_x            = 0.47
panel_label_y            = 0.75
panel_label_ha           = "right"
panel_label_va           = "bottom"
panel_label_stroke_width = 0.0

panel_label_color  = "black"
panel_label_stroke = "black"

# Per-panel force-label placement. Keys are (row, col), where
# row 0 = Lyα force row, row 1 = gas-pressure force row.
# These default to your current global placement.
panel_label_positions = {
    (0, 0): dict(x=panel_label_x, y=panel_label_y, ha=panel_label_ha, va=panel_label_va),
    (0, 1): dict(x=panel_label_x, y=panel_label_y, ha=panel_label_ha, va=panel_label_va),
    (0, 2): dict(x=panel_label_x, y=panel_label_y, ha=panel_label_ha, va=panel_label_va),

    (1, 0): dict(x=panel_label_x+0.073, y=panel_label_y, ha=panel_label_ha, va=panel_label_va),
    (1, 1): dict(x=panel_label_x+0.073, y=panel_label_y, ha=panel_label_ha, va=panel_label_va),
    (1, 2): dict(x=panel_label_x+0.073, y=panel_label_y, ha=panel_label_ha, va=panel_label_va),

    (2, 0): dict(x=panel_label_x+0.073, y=panel_label_y, ha=panel_label_ha, va=panel_label_va),
    (2, 1): dict(x=panel_label_x+0.073, y=panel_label_y, ha=panel_label_ha, va=panel_label_va),
    (2, 2): dict(x=panel_label_x+0.073, y=panel_label_y, ha=panel_label_ha, va=panel_label_va),
}

show_row_labels = True
row_label_x     = 0.04
row_label_ha    = "left"
row_label_color = "black"
row_label_Lya      = r"Ly$\alpha$ radiation pressure"
row_label_gas      = r"Gas pressure"
row_label_fraction = r"Ly$\alpha$ contribution"

# Labels inside panels.
label_Lya = r"$\boldsymbol{f}_{Ly\alpha}\cdot(-\hat{\boldsymbol{f}}_{grav})$"
label_gas = r"$(-\nabla P)\cdot(-\hat{\boldsymbol{f}}_{grav})$"
label_fraction = r"$|f_{Ly\alpha}|/(|f_{Ly\alpha}|+|f_P|)$"

# -------------------------------------------------------------------
# Time labels
# -------------------------------------------------------------------
time_label_color        = "black"
time_label_stroke       = "black"
time_label_stroke_width = 0.0

# -------------------------------------------------------------------
# Panel frame
# -------------------------------------------------------------------
panel_frame_linewidth = 2.5

# -------------------------------------------------------------------
# Scale bar
# -------------------------------------------------------------------
show_scale_bar         = True
scale_bar_length       = 0.2
scale_bar_label        = "0.2 pc"
scale_bar_panel        = (0, 0)      # (row, col)
scale_bar_x_axes       = 0.74
scale_bar_y_axes       = 0.08
scale_bar_color        = "black"
scale_bar_lw           = 3.0
scale_bar_capsize_axes = 0.045
scale_bar_label_pad_axes = 0.020
scale_bar_fontsize     = 45
scale_bar_stroke       = "black"
scale_bar_stroke_width = 0.0

# -------------------------------------------------------------------
# Useful constants
# -------------------------------------------------------------------
yr  = 3.1556926e7
kyr = 1e3 * yr
pc  = 3.085677581e18
c_light = 2.99792458e10
L_sun   = 3.828e33


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
plt.rcParams["axes.linewidth"]      = 3.0
plt.rcParams["grid.linewidth"]      = 3.0
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
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    Nt = t.size
    Nr = len(r_edges) - 1
    Nz = len(Z_edges) - 1

    shape = arr.shape
    time_axes = [ax for ax, s in enumerate(shape) if s == Nt]
    if len(time_axes) != 1:
        raise RuntimeError(f"Could not uniquely identify time axis: shape={shape}, Nt={Nt}")

    arr_tfirst = np.moveaxis(arr, time_axes[0], 0)
    _, a1, a2 = arr_tfirst.shape

    if (a1, a2) == (Nr, Nz):
        return arr_tfirst
    elif (a1, a2) == (Nz, Nr):
        return arr_tfirst.transpose(0, 2, 1)
    else:
        raise RuntimeError(f"Spatial dims {(a1, a2)} do not match Nr={Nr}, Nz={Nz}")


def nearest_saved_index(t, target_time):
    saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]
    if saved_idx.size == 0:
        raise RuntimeError("No saved snapshots found.")
    return saved_idx[np.argmin(np.abs(t[saved_idx] - target_time))]


def fmt_log_tick(val):
    exp = int(np.round(np.log10(abs(val))))
    if exp == -2: return r"$0.01$"
    if exp == -1: return r"$0.1$"
    if exp ==  0: return r"$1$"
    if exp ==  1: return r"$10$"
    if exp ==  2: return r"$100$"
    return rf"$10^{{{exp}}}$"


def get_log_ticks_from_norm(norm):
    logmin = int(np.ceil(np.log10(norm.vmin)))
    logmax = int(np.floor(np.log10(norm.vmax)))
    if logmax < logmin:
        return None, None
    ticks = 10.0**np.arange(logmin, logmax + 1)
    return ticks, [fmt_log_tick(v) for v in ticks]


def fmt_signed_log_tick(val):
    """Log tick labels with the user's preferred small-power formatting."""
    if val == 0:
        return r"$0$"

    sign = "-" if val < 0 else ""
    exp = int(np.round(np.log10(abs(val))))

    if exp == -2: return rf"${sign}0.01$"
    if exp == -1: return rf"${sign}0.1$"
    if exp ==  0: return rf"${sign}1$"
    if exp ==  1: return rf"${sign}10$"
    if exp ==  2: return rf"${sign}100$"
    return rf"${sign}10^{{{exp}}}$"


def get_symlog_ticks_from_norm(norm, max_ticks_per_side=4):
    """Return symmetric decade ticks for a SymLogNorm colourbar."""
    abs_vmax = max(abs(norm.vmin), abs(norm.vmax))
    logmin = int(np.ceil(np.log10(norm.linthresh)))
    logmax = int(np.floor(np.log10(abs_vmax)))
    if logmax < logmin:
        return [0.0], [r"$0$"]

    exps = np.arange(logmin, logmax + 1)

    # Avoid overcrowding the inset colourbar: keep at most max_ticks_per_side
    # logarithmic ticks on each side, always including the outermost decade.
    if exps.size > max_ticks_per_side:
        idx = np.linspace(0, exps.size - 1, max_ticks_per_side).round().astype(int)
        exps = np.unique(exps[idx])

    pos_ticks = 10.0**exps
    ticks = np.concatenate((-pos_ticks[::-1], [0.0], pos_ticks))
    labels = [fmt_signed_log_tick(v) for v in ticks]
    return ticks, labels


def use_custom_ticks(default_ticks, default_labels, custom_ticks, custom_labels):
    ticks  = default_ticks  if custom_ticks  is None else custom_ticks
    labels = default_labels if custom_labels is None else custom_labels
    if custom_ticks is not None and custom_labels is None:
        labels = None
    return ticks, labels


def add_inset_colorbar(ax, mappable, ticks=None, ticklabels=None, label=None):
    cax = ax.inset_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    cb  = plt.colorbar(mappable, cax=cax, orientation="vertical")

    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)

    cb.ax.tick_params(labelsize=fontsize_cbar_ticks,
                      width=cbar_tick_width, length=cbar_tick_length,
                      pad=cbar_tick_pad, colors=cbar_tick_color,
                      labelcolor=cbar_tick_color)
    cb.ax.minorticks_off()

    if ticks is not None:
        from matplotlib.ticker import FixedLocator, FixedFormatter
        cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
        if ticklabels is not None:
            cb.ax.yaxis.set_major_formatter(FixedFormatter(ticklabels))

    cb.outline.set_linewidth(1.5)
    cb.outline.set_edgecolor(cbar_outline_color)
    for spine in cb.ax.spines.values():
        spine.set_edgecolor(cbar_outline_color)
        spine.set_linewidth(1.5)

    label_to_use = cbar_label if label is None else label
    if label_to_use is not None:
        cb.set_label(label_to_use, color=cbar_tick_color, fontsize=cbar_label_size,
                     labelpad=8)
        if cbar_label_stroke is not None and cbar_label_stroke_width > 0.0:
            cb.ax.yaxis.label.set_path_effects([
                pe.withStroke(linewidth=cbar_label_stroke_width,
                              foreground=cbar_label_stroke)
            ])
        else:
            cb.ax.yaxis.label.set_path_effects([])

    return cb


def panel_label(ax, text, row=None, col=None, color=None, stroke=None):
    _color  = panel_label_color  if color  is None else color
    _stroke = panel_label_stroke if stroke is None else stroke

    pos = None
    if row is not None and col is not None:
        pos = panel_label_positions.get((row, col), None)

    if pos is None:
        x  = panel_label_x
        y  = panel_label_y
        ha = panel_label_ha
        va = panel_label_va
    else:
        x  = pos.get("x",  panel_label_x)
        y  = pos.get("y",  panel_label_y)
        ha = pos.get("ha", panel_label_ha)
        va = pos.get("va", panel_label_va)

    if panel_label_stroke_width > 0.0 and _stroke is not None:
        fx = [pe.withStroke(linewidth=panel_label_stroke_width,
                            foreground=_stroke)]
    else:
        fx = []

    ax.text(x, y, text,
            transform=ax.transAxes,
            ha=ha, va=va,
            color=_color, fontsize=fontsize_panel_label,
            path_effects=fx)


def time_label(ax, time_kyr):
    ax.text(0.04, 0.96, rf"t = {time_kyr:.2f} kyr",
            transform=ax.transAxes, ha="left", va="top",
            color=time_label_color, fontsize=fontsize_time,
            path_effects=[pe.withStroke(linewidth=time_label_stroke_width,
                                        foreground=time_label_stroke)])


def draw_scale_bar(ax):
    if not show_scale_bar:
        return

    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    xspan = xhi - xlo
    yspan = yhi - ylo

    bar_x = xlo + scale_bar_x_axes * xspan
    bar_y = ylo + scale_bar_y_axes * yspan
    x0 = bar_x - 0.5 * scale_bar_length
    x1 = bar_x + 0.5 * scale_bar_length

    capsize = scale_bar_capsize_axes * yspan
    label_pad = scale_bar_label_pad_axes * yspan

    fx = ([pe.withStroke(linewidth=scale_bar_stroke_width, foreground=scale_bar_stroke)]
          if scale_bar_stroke is not None else [])

    ax.plot([x0, x1], [bar_y, bar_y], color=scale_bar_color, lw=scale_bar_lw,
            solid_capstyle="butt", transform=ax.transData, zorder=10,
            path_effects=fx)
    for xc in (x0, x1):
        ax.plot([xc, xc], [bar_y - 0.5*capsize, bar_y + 0.5*capsize],
                color=scale_bar_color, lw=scale_bar_lw,
                solid_capstyle="round", transform=ax.transData, zorder=10,
                path_effects=fx)
    ax.text(bar_x, bar_y + 0.5*capsize + label_pad, scale_bar_label,
            color=scale_bar_color, fontsize=scale_bar_fontsize,
            ha="center", va="bottom", transform=ax.transData, zorder=11,
            path_effects=fx)


def make_ax(fig, col_left, row_bottom, width, height, sharex=None, sharey=None):
    return fig.add_axes([col_left, row_bottom, width, height],
                        sharex=sharex, sharey=sharey)


def pressure_gradient(P, r_centres, Z_centres):
    """
    Return ∂P/∂r and ∂P/∂Z on cell centres.

    P is cell-centred.  Since P is a scalar in cylindrical coordinates,
    ∇P = (∂P/∂r) rhat + (∂P/∂Z) Zhat in this axisymmetric r-Z slice;
    there is no extra 1/r factor for these components.
    """
    P64 = np.asarray(P, dtype=np.float64)

    # np.gradient supports non-uniform coordinate arrays.  It uses centred
    # differences in the interior and one-sided differences at the boundary.
    edge_order = 2 if (P64.shape[0] >= 3 and P64.shape[1] >= 3) else 1
    dPdr, dPdZ = np.gradient(P64, r_centres, Z_centres, edge_order=edge_order)
    return dPdr, dPdZ


def antigravity_projection(f_r, f_Z, fg_r, fg_Z):
    """
    Project a force density f onto the direction opposite to gravity.

    Returns f · (-fgrav_hat), in the same units as f.
    Positive values are outward / anti-gravity.
    """
    f_r, f_Z   = np.float64(f_r),  np.float64(f_Z)
    fg_r, fg_Z = np.float64(fg_r), np.float64(fg_Z)

    fg_mag = np.sqrt(fg_r**2 + fg_Z**2)
    valid = fg_mag > f_grav_mask_threshold
    fg_safe = np.where(valid, fg_mag, 1.0)

    proj = -(f_r * fg_r + f_Z * fg_Z) / fg_safe
    return np.where(valid, proj, np.nan)


def clip_for_plot(field, norm):
    if use_symlognorm:
        return np.where(np.isfinite(field),
                        np.clip(field, norm.vmin, norm.vmax),
                        np.nan)
    else:
        good = np.isfinite(field) & (field > force_mask_threshold)
        return np.where(good, np.clip(field, norm.vmin, norm.vmax), np.nan)


def Lya_force_fraction_percent(F_Lya_anti, F_gas_anti):
    """Return 100 |F_Lya| / (|F_Lya| + |F_gas|) using anti-gravity projections."""
    abs_Lya = np.abs(np.asarray(F_Lya_anti, dtype=np.float64))
    abs_gas = np.abs(np.asarray(F_gas_anti, dtype=np.float64))
    denom = abs_Lya + abs_gas

    good = (np.isfinite(abs_Lya) & np.isfinite(abs_gas) &
            (denom > fraction_mask_denominator_below))

    fraction = np.full_like(denom, np.nan, dtype=np.float64)
    fraction[good] = 100.0 * abs_Lya[good] / denom[good]
    return fraction


def robust_positive_limits(fields, pmin=1.0, pmax=99.7):
    vals = []
    for field in fields:
        arr = np.asarray(field)
        arr = arr[np.isfinite(arr) & (arr > force_mask_threshold)]
        if arr.size > 0:
            vals.append(arr)
    if len(vals) == 0:
        raise RuntimeError("No positive force values found for LogNorm.")

    vals = np.concatenate(vals)
    vmin = np.nanpercentile(vals, pmin)
    vmax = np.nanpercentile(vals, pmax)

    # Keep limits clean and decade-like.
    vmin = 10.0**np.floor(np.log10(max(vmin, np.nanmin(vals))))
    vmax = 10.0**np.ceil(np.log10(max(vmax, vmin * 10.0)))
    return vmin, vmax


def hdf5_value_to_python(value):
    """Convert a small HDF5 dataset/attribute value to a Python scalar if possible."""
    arr = np.asarray(value)
    if arr.shape == ():
        out = arr.item()
    elif arr.size == 1:
        out = arr.ravel()[0]
    else:
        return value

    if isinstance(out, bytes):
        out = out.decode()
    if isinstance(out, str):
        try:
            out = float(out)
        except ValueError:
            pass
    return out


def find_hdf5_dataset(f, candidate_names):
    """Return the first dataset whose final path component matches a candidate."""
    candidates = {name.lower() for name in candidate_names}

    for name in candidate_names:
        if name in f and isinstance(f[name], h5py.Dataset):
            return f[name]

    found = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.split("/")[-1].lower() in candidates:
            found.append(obj)

    f.visititems(visitor)
    return found[0] if found else None


def find_hdf5_scalar(f, candidate_names):
    """Find a scalar stored either as a dataset or as an attribute."""
    candidates = {name.lower() for name in candidate_names}

    dset = find_hdf5_dataset(f, candidate_names)
    if dset is not None and np.asarray(dset).size == 1:
        return hdf5_value_to_python(dset[()])

    found = []
    def visitor(name, obj):
        for key, val in obj.attrs.items():
            if key.lower() in candidates:
                found.append(hdf5_value_to_python(val))

    for key, val in f.attrs.items():
        if key.lower() in candidates:
            found.append(hdf5_value_to_python(val))

    f.visititems(visitor)
    return found[0] if found else None


def reorder_spatial_2d(arr, r_edges, Z_edges):
    """Reorder a 2D cell-centred array to shape (Nr, Nz)."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    Nr = len(r_edges) - 1
    Nz = len(Z_edges) - 1

    if arr.shape == (Nr, Nz):
        return arr
    elif arr.shape == (Nz, Nr):
        return arr.T
    else:
        raise RuntimeError(f"Spatial dims {arr.shape} do not match Nr={Nr}, Nz={Nz}")


def force_integral_in_Lbol_over_c(force_density, cell_volumes, L_bol, mask=None):
    """Integrate force density over volume and return the result in units L_bol/c."""
    good = np.isfinite(force_density) & np.isfinite(cell_volumes)
    if mask is not None:
        good &= mask

    F_cgs = force_integral_symmetry_factor * np.sum(
        np.where(good, force_density * cell_volumes, 0.0), dtype=np.float64
    )
    return F_cgs / (L_bol / c_light)


###############################################################
#######                                               #######
#######            R E A D  D A T A                  #######
#######                                               #######
###############################################################

with h5py.File(Filename, "r") as f:
    t = f["Times"][:]
    r = f["r"][:]
    Z = f["Z"][:]

    required = [
        "GasPressure_History",
        "f_Lya_r_History", "f_Lya_Z_History",
        "f_gravity_r_History", "f_gravity_Z_History",
    ]
    missing = [name for name in required if name not in f]
    if missing:
        raise RuntimeError("Missing required HDF5 datasets: " + ", ".join(missing))

    P_raw       = f["GasPressure_History"][:]
    f_Lya_r_raw = f["f_Lya_r_History"][:]
    f_Lya_Z_raw = f["f_Lya_Z_History"][:]
    fg_r_raw    = f["f_gravity_r_History"][:]
    fg_Z_raw    = f["f_gravity_Z_History"][:]

    T_dset = find_hdf5_dataset(f, [
        "T_History", "Temperature_History", "GasTemperature_History",
        "Temperature", "T",
    ])
    Vol_dset = find_hdf5_dataset(f, [
        "CellVolumes", "CellVolume", "Cell_Volumes", "CellVolumes_History",
    ])
    L_bol_raw = find_hdf5_scalar(f, [
        "L_bol", "Lbol", "L_bol_cgs", "Lbol_cgs",
        "Luminosity_bol", "BolometricLuminosity", "StellarLuminosityBolometric",
    ])

    T_raw = None if T_dset is None else T_dset[:]
    CellVolumes_raw = None if Vol_dset is None else Vol_dset[:]

P_hist       = reorder_history_3d(P_raw,       t, r, Z)
f_Lya_r_hist = reorder_history_3d(f_Lya_r_raw, t, r, Z)
f_Lya_Z_hist = reorder_history_3d(f_Lya_Z_raw, t, r, Z)
fg_r_hist    = reorder_history_3d(fg_r_raw,    t, r, Z)
fg_Z_hist    = reorder_history_3d(fg_Z_raw,    t, r, Z)

T_hist = None if T_raw is None else reorder_history_3d(T_raw, t, r, Z)
CellVolumes = None
if CellVolumes_raw is not None:
    if CellVolumes_raw.ndim == 2:
        CellVolumes = reorder_spatial_2d(CellVolumes_raw, r, Z)
    elif CellVolumes_raw.ndim == 3:
        # If volumes were saved with a history/time axis, use the first entry;
        # cell volumes should not change during this fixed-grid run.
        CellVolumes = reorder_history_3d(CellVolumes_raw, t, r, Z)[0]
    else:
        raise RuntimeError(f"Unexpected CellVolumes shape: {CellVolumes_raw.shape}")

L_bol = None if L_bol_raw is None else float(L_bol_raw)
if L_bol is not None and L_bol_in_solar_units:
    L_bol *= L_sun

r_centres = 0.5 * (r[:-1] + r[1:])
Z_centres = 0.5 * (Z[:-1] + Z[1:])
R_edges_pc, Z_edges_pc = np.meshgrid(r / pc, Z / pc, indexing="ij")


###############################################################
#######                                               #######
#######       C O M P U T E  F I E L D S              #######
#######                                               #######
###############################################################

def compute_snapshot_fields(target_time_kyr):
    k = nearest_saved_index(t, target_time_kyr * kyr)
    print(f"Requested target time = {target_time_kyr:.3f} kyr")
    print(f"Using snapshot at     = {t[k] / kyr:.3f} kyr")

    P    = P_hist[k]
    T    = None if T_hist is None else T_hist[k]
    fg_r = fg_r_hist[k]
    fg_Z = fg_Z_hist[k]

    # Lyα radiation-pressure force, projected anti-gravity.
    F_Lya_anti = antigravity_projection(
        f_Lya_r_hist[k], f_Lya_Z_hist[k], fg_r, fg_Z
    )

    # Gas-pressure force density is -∇P.
    dPdr, dPdZ = pressure_gradient(P, r_centres, Z_centres)
    F_gas_r = -dPdr
    F_gas_Z = -dPdZ

    # Project gas-pressure force anti-gravity.
    F_gas_anti = antigravity_projection(F_gas_r, F_gas_Z, fg_r, fg_Z)

    # Percentage of the total outward projected force contributed by Lyα.
    F_Lya_fraction_percent = Lya_force_fraction_percent(F_Lya_anti, F_gas_anti)

    result = {
        "k": k,
        "time_kyr": t[k] / kyr,
        "F_Lya_anti": F_Lya_anti,
        "F_gas_anti": F_gas_anti,
        "F_Lya_fraction_percent": F_Lya_fraction_percent,
    }

    if print_force_integrals:
        if CellVolumes is None or L_bol is None:
            print("Skipping force integrals: could not find CellVolumes and/or L_bol in the HDF5 file.")
        elif use_T_mask_for_gas_force_integral and T is None:
            print("Skipping masked gas-pressure force integral: could not find T_History/Temperature_History in the HDF5 file.")
        else:
            T_mask = None
            if use_T_mask_for_gas_force_integral:
                T_mask = np.isfinite(T) & (T > T_threshold)

            F_Lya_Lbol_c = force_integral_in_Lbol_over_c(
                F_Lya_anti, CellVolumes, L_bol
            )
            F_gas_Lbol_c = force_integral_in_Lbol_over_c(
                F_gas_anti, CellVolumes, L_bol, mask=T_mask
            )

            if T_mask is None:
                mask_info = "no T mask"
            else:
                mask_info = f"T > {T_threshold:g} K, {np.count_nonzero(T_mask)}/{T_mask.size} cells"

            print("Integrated projected forces [L_bol/c]:")
            print(f"  Lyα radiation pressure      = {F_Lya_Lbol_c:.6e}")
            print(f"  Gas/photoionization pressure = {F_gas_Lbol_c:.6e}  ({mask_info})")

    return result

snapshots = [compute_snapshot_fields(tt) for tt in TargetTimes_kyr]
all_raw_fields = []
for snap in snapshots:
    all_raw_fields.append(snap["F_Lya_anti"])
    all_raw_fields.append(snap["F_gas_anti"])


###############################################################
#######                                               #######
#######             C O L O U R  N O R M              #######
#######                                               #######
###############################################################

if use_symlognorm:
    if force_vmin is None or force_vmax is None:
        finite = np.concatenate([np.ravel(x[np.isfinite(x)]) for x in all_raw_fields])
        vmax = np.nanpercentile(np.abs(finite), percentile_vmax)
        vmax = 10.0**np.ceil(np.log10(max(vmax, force_linthresh * 10.0)))
        vmin = -vmax
    else:
        # Backwards compatible convenience: if both limits are accidentally
        # given as positive values, interpret force_vmax as the absolute range.
        if force_vmin > 0.0 and force_vmax > 0.0:
            vmax = force_vmax
            vmin = -force_vmax
        else:
            vmin, vmax = force_vmin, force_vmax

    if not (vmin < 0.0 < vmax):
        raise RuntimeError(
            "For use_symlognorm=True, use signed limits with vmin < 0 < vmax, "
            "or set force_vmin = force_vmax = None for automatic limits."
        )

    norm_force = SymLogNorm(linthresh=force_linthresh, vmin=vmin, vmax=vmax)
else:
    if force_vmin is None or force_vmax is None:
        vmin_auto, vmax_auto = robust_positive_limits(
            all_raw_fields, percentile_vmin, percentile_vmax
        )
        vmin = vmin_auto if force_vmin is None else force_vmin
        vmax = vmax_auto if force_vmax is None else force_vmax
    else:
        vmin, vmax = force_vmin, force_vmax
    norm_force = LogNorm(vmin=vmin, vmax=vmax, clip=True)

for snap in snapshots:
    snap["F_Lya_plot"] = clip_for_plot(snap["F_Lya_anti"], norm_force)
    snap["F_gas_plot"] = clip_for_plot(snap["F_gas_anti"], norm_force)
    snap["F_Lya_fraction_plot"] = np.where(
        np.isfinite(snap["F_Lya_fraction_percent"]),
        np.clip(snap["F_Lya_fraction_percent"], fraction_vmin, fraction_vmax),
        np.nan,
    )

norm_fraction = Normalize(vmin=fraction_vmin, vmax=fraction_vmax, clip=True)

if not use_symlognorm:
    ticks_default, ticklabels_default = get_log_ticks_from_norm(norm_force)
else:
    ticks_default, ticklabels_default = get_symlog_ticks_from_norm(norm_force)

ticks_force, ticklabels_force = use_custom_ticks(
    ticks_default, ticklabels_default, ticks_force_custom, ticklabels_force_custom
)


###############################################################
#######                                                 #######
#######           F I G U R E  &  A X E S               #######
#######                                                 #######
###############################################################

nrows = 3 if show_Lya_force_fraction_row else 2
figsize_plot = figsize_with_fraction_row if show_Lya_force_fraction_row else figsize
row_height_plot = row_height_with_fraction_row if show_Lya_force_fraction_row else row_height

fig = plt.figure(figsize=figsize_plot)

col_lefts = [col_left_0 + i * (col_width + col_gap) for i in range(3)]
row_bottoms = [row_bottom_0 + (nrows - 1 - row) * (row_height_plot + row_gap)
               for row in range(nrows)]

axes = [[None for _ in range(3)] for _ in range(nrows)]
axes[0][0] = make_ax(fig, col_lefts[0], row_bottoms[0], col_width, row_height_plot)
for row in range(nrows):
    for col in range(3):
        if row == 0 and col == 0:
            continue
        axes[row][col] = make_ax(fig, col_lefts[col], row_bottoms[row],
                                  col_width, row_height_plot,
                                  sharex=axes[0][0], sharey=axes[0][0])

cmap_obj = plt.get_cmap(cmap_force).copy()
cmap_obj.set_bad(mask_color)

cmap_fraction_obj = plt.get_cmap(cmap_fraction).copy()
cmap_fraction_obj.set_bad(mask_color)

ims = []
ims_fraction = []
for j, snap in enumerate(snapshots):
    im = axes[0][j].pcolormesh(R_edges_pc, Z_edges_pc, snap["F_Lya_plot"].T,
                               norm=norm_force, cmap=cmap_obj, shading="auto",
                               rasterized=rasterized)
    ims.append(im)

    axes[1][j].pcolormesh(R_edges_pc, Z_edges_pc, snap["F_gas_plot"].T,
                          norm=norm_force, cmap=cmap_obj, shading="auto",
                          rasterized=rasterized)

    if show_Lya_force_fraction_row:
        im_fraction = axes[2][j].pcolormesh(
            R_edges_pc, Z_edges_pc, snap["F_Lya_fraction_plot"].T,
            norm=norm_fraction, cmap=cmap_fraction_obj, shading="auto",
            rasterized=rasterized
        )
        ims_fraction.append(im_fraction)

# -------------------------------------------------------------------
# Axes cosmetics
# -------------------------------------------------------------------
for row in range(nrows):
    for col in range(3):
        ax = axes[row][col]
        ax.set_xlim(r[0] / pc, xmax_fraction * r[-1] / pc)
        ax.set_ylim(Z[0] / pc, zmax_fraction * Z[-1] / pc)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(which="both", bottom=False, top=False,
                       left=False, right=False, labelbottom=False,
                       labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(panel_frame_linewidth)

# -------------------------------------------------------------------
# Labels
# -------------------------------------------------------------------
for j, snap in enumerate(snapshots):
    time_label(axes[0][j], snap["time_kyr"])
    time_label(axes[1][j], snap["time_kyr"])
    panel_label(axes[0][j], label_Lya, row=0, col=j)
    panel_label(axes[1][j], label_gas, row=1, col=j)

    if show_Lya_force_fraction_row:
        time_label(axes[2][j], snap["time_kyr"])
        panel_label(axes[2][j], label_fraction, row=2, col=j)

if show_row_labels:
    y_lya = row_bottoms[0] + 0.5 * row_height_plot
    y_gas = row_bottoms[1] + 0.5 * row_height_plot
    fig.text(row_label_x, y_lya, row_label_Lya,
             ha=row_label_ha, va="center", rotation=90,
             color=row_label_color, fontsize=fontsize_row_label)
    fig.text(row_label_x, y_gas, row_label_gas,
             ha=row_label_ha, va="center", rotation=90,
             color=row_label_color, fontsize=fontsize_row_label)

    if show_Lya_force_fraction_row:
        y_fraction = row_bottoms[2] + 0.5 * row_height_plot
        fig.text(row_label_x, y_fraction, row_label_fraction,
                 ha=row_label_ha, va="center", rotation=90,
                 color=row_label_color, fontsize=fontsize_row_label)

# Shared inset colorbar.
if show_colorbar:
    row, col = colorbar_panel
    add_inset_colorbar(axes[row][col], ims[0], ticks_force, ticklabels_force)

# Separate inset colorbar for the Lyα contribution row.
if show_Lya_force_fraction_row and show_fraction_colorbar and len(ims_fraction) > 0:
    row, col = fraction_colorbar_panel
    add_inset_colorbar(axes[row][col], ims_fraction[0],
                       fraction_ticks, fraction_ticklabels,
                       label=fraction_cbar_label)

# Scale bar.
if show_scale_bar:
    row, col = scale_bar_panel
    draw_scale_bar(axes[row][col])

plt.savefig(FigureName, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure to {FigureName}")
