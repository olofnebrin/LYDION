import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.patheffects as pe
from matplotlib.ticker import FixedLocator, FixedFormatter
from pathlib import Path
import sys
import os

repo_root = Path(__file__).resolve().parents[1]
os.chdir(repo_root)
cmastro_src = repo_root / "src" / "cmastro" / "src"
sys.path.insert(0, str(cmastro_src))
try:
    import cmastro  # noqa: F401  # enables cma:* colormaps, if available
except Exception:
    pass

# ============================================================
#   U S E R   I N P U T S
# ============================================================

Filename       = "RHDSim_Lya_u4_256x256_2026-05-02.h5"
FigureName     = "LYDION_Figure_IsolatedStar_DustAbsVelocities.pdf"

# Use a target time, or set TargetTime_kyr = None to use the final saved snapshot.
TargetTime_kyr = None

# Visible domain. Set to None to show the full saved box.
xmax_pc = None
zmax_pc = None

# Figure/layout options
figsize    = (24, 8)
wgap       = 0.000
rasterized = True

# Font sizes
fontsize_cbar_ticks  = 26
fontsize_panel_label = 42
fontsize_time        = 38
fontsize_abc         = 40

# Colormaps
cmap_sigmaLya = "magma"
cmap_vrel_sil = "viridis"
cmap_vrel_c   = "plasma"

# Autoscaling. If False, manual ranges below are used.
AutoScale_sigmaLya = False
AutoScale_vrel     = False

# Manual ranges
sigmaLya_vmin, sigmaLya_vmax = 1e-25, 1e-24       # cm^2 H^-1
vrel_vmin,     vrel_vmax     = 0.1, 30.0         # km s^-1

# Percentile ranges used if autoscaling.
pct_low_log  = 1.0
pct_high_log = 99.0

# Density floors used only to avoid division by zero in velocities.
rho_floor_factor   = 1e-30
rhod_floor_factor  = 1e-30
vrel_log_floor     = 1e-6  # km s^-1, only used before taking LogNorm

# Colorbar options, in axis coordinates
cbar_x      = 0.80
cbar_y      = 0.55
cbar_width  = 0.045
cbar_height = 0.36

cbar_tick_length = 0
cbar_tick_width  = 0

# Colorbar tick/outline colors
cbar_tick_color_sigma = "black"
cbar_tick_color_vrel  = "white"
cbar_outline_sigma    = "black"
cbar_outline_vrel     = "white"

# Optional custom colorbar ticks/labels. Set to None for automatic decade ticks.
ticks_sigma_custom      = [1e-24, 1e-23]
ticklabels_sigma_custom = [r"$10^{-24}$", r"$10^{-23}$"]

ticks_vrel_custom      = None
ticklabels_vrel_custom = None

# Text style
panel_label_color        = "white"
panel_label_stroke       = "black"
panel_label_stroke_width = 5.0

time_label_color        = "white"
time_label_stroke       = "black"
time_label_stroke_width = 5.0

abc_label_color        = "white"
abc_label_stroke       = "black"
abc_label_stroke_width = 5.0

# Useful constants
yr  = 3.1556926e7
kyr = 1e3 * yr
pc  = 3.085677581e18
kms = 1e5

# ============================================================
#   P L O T   C O S M E T I C S
# ============================================================

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 39.5})
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

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
plt.rcParams['xtick.top']       = True
plt.rcParams['xtick.bottom']    = True
plt.rcParams['ytick.left']      = True
plt.rcParams['ytick.right']     = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ============================================================
#   H E L P E R   F U N C T I O N S
# ============================================================

def reorder_history_3d(arr, t, r_edges, Z_edges):
    """Return a 3D history array with shape (Nt, Nr, Nz)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    Nt = t.size
    Nr = len(r_edges) - 1
    Nz = len(Z_edges) - 1

    time_axes = [ax for ax, s in enumerate(arr.shape) if s == Nt]
    if len(time_axes) != 1:
        raise RuntimeError(
            f"Could not uniquely identify time axis in array with shape {arr.shape} and Nt = {Nt}"
        )

    arr_tfirst = np.moveaxis(arr, time_axes[0], 0)
    _, a1, a2 = arr_tfirst.shape

    if (a1, a2) == (Nr, Nz):
        return arr_tfirst
    elif (a1, a2) == (Nz, Nr):
        return arr_tfirst.transpose(0, 2, 1)
    else:
        raise RuntimeError(
            f"After moving time axis, spatial dims {(a1, a2)} do not match Nr={Nr}, Nz={Nz}"
        )


def nearest_saved_index(t, target_time):
    saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]
    if saved_idx.size == 0:
        raise RuntimeError("No saved snapshots found in file.")
    return saved_idx[np.argmin(np.abs(t[saved_idx] - target_time))]


def final_saved_index(t):
    saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]
    if saved_idx.size == 0:
        raise RuntimeError("No saved snapshots found in file.")
    return saved_idx[-1]


def require_dataset(file, name):
    if name not in file:
        available = "\n".join(sorted(file.keys()))
        raise KeyError(f"Dataset '{name}' not found in {Filename}. Available top-level keys:\n{available}")
    return file[name][:]


def positive_values(arr):
    vals = np.asarray(arr)
    return vals[np.isfinite(vals) & (vals > 0.0)]


def make_lognorm_from_arrays(arrays, default_vmin, default_vmax, pct_low=1.0, pct_high=99.0):
    chunks = [positive_values(a) for a in arrays if a is not None]
    chunks = [c for c in chunks if c.size > 0]
    if not chunks:
        return LogNorm(vmin=default_vmin, vmax=default_vmax)

    vals = np.concatenate(chunks)
    vmin = np.percentile(vals, pct_low)
    vmax = np.percentile(vals, pct_high)

    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin <= 0.0) or (vmax <= vmin):
        return LogNorm(vmin=default_vmin, vmax=default_vmax)

    return LogNorm(vmin=vmin, vmax=vmax)


def fmt_log_tick(val):
    exp = int(np.round(np.log10(val)))
    if exp == -3:
        return r"$0.001$"
    elif exp == -2:
        return r"$0.01$"
    elif exp == -1:
        return r"$0.1$"
    elif exp == 0:
        return r"$1$"
    elif exp == 1:
        return r"$10$"
    elif exp == 2:
        return r"$100$"
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


def add_inset_colorbar(ax, mappable, ticks=None, ticklabels=None,
                       tick_color="white", outline_color="white"):
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
        labelcolor=tick_color,
    )

    cb.ax.minorticks_off()
    if ticks is not None:
        cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
        if ticklabels is not None:
            cb.ax.yaxis.set_major_formatter(FixedFormatter(ticklabels))

    cb.outline.set_linewidth(1.5)
    cb.outline.set_edgecolor(outline_color)
    for spine in cb.ax.spines.values():
        spine.set_edgecolor(outline_color)
        spine.set_linewidth(1.5)
    return cb


def label_panel(ax, text, x=0.94, y=0.08, ha="right", va="bottom"):
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        color=panel_label_color,
        fontsize=fontsize_panel_label,
        path_effects=[pe.withStroke(linewidth=panel_label_stroke_width,
                                    foreground=panel_label_stroke)],
    )


def label_time(ax, time_kyr):
    ax.text(
        0.04, 0.96, rf"t = {time_kyr:.2f} kyr",
        transform=ax.transAxes,
        ha="left", va="top",
        color=time_label_color,
        fontsize=fontsize_time,
        path_effects=[pe.withStroke(linewidth=time_label_stroke_width,
                                    foreground=time_label_stroke)],
    )


def label_abc(ax, letter):
    ax.text(
        0.04, 0.08, f"({letter})",
        transform=ax.transAxes,
        ha="left", va="bottom",
        color=abc_label_color,
        fontsize=fontsize_abc,
        path_effects=[pe.withStroke(linewidth=abc_label_stroke_width,
                                    foreground=abc_label_stroke)],
    )


def velocity_from_momentum(momentum_r, momentum_Z, density, density_floor):
    """Return (u_r, u_Z), with NaN where density is too small."""
    good = np.isfinite(density) & (density > density_floor)
    u_r = np.full_like(density, np.nan, dtype=np.float64)
    u_Z = np.full_like(density, np.nan, dtype=np.float64)
    u_r[good] = momentum_r[good] / density[good]
    u_Z[good] = momentum_Z[good] / density[good]
    return u_r, u_Z


def relative_speed_kms(ur_d, uZ_d, ur_g, uZ_g):
    return np.sqrt((ur_d - ur_g)**2 + (uZ_d - uZ_g)**2) / kms

# ============================================================
#   R E A D   R U N
# ============================================================

with h5py.File(Filename, "r") as f:
    t = require_dataset(f, "Times")
    r = require_dataset(f, "r")
    Z = require_dataset(f, "Z")

    rho_raw       = require_dataset(f, "GasDensity_History")
    mom_r_raw     = require_dataset(f, "rMomentum_History")
    mom_Z_raw     = require_dataset(f, "ZMomentum_History")

    rho_d_sil_raw = require_dataset(f, "DustDensity_Sil_History")
    rho_d_c_raw   = require_dataset(f, "DustDensity_C_History")

    momd_r_sil_raw = require_dataset(f, "DustrMomentum_Sil_History")
    momd_r_c_raw   = require_dataset(f, "DustrMomentum_C_History")

    momd_Z_sil_raw = require_dataset(f, "DustZMomentum_Sil_History")
    momd_Z_c_raw   = require_dataset(f, "DustZMomentum_C_History")

    sigmaLya_raw  = require_dataset(f, "LyaDustAbsorptionCrossSections_History")

rho_hist       = reorder_history_3d(rho_raw,       t, r, Z)
mom_r_hist     = reorder_history_3d(mom_r_raw,     t, r, Z)
mom_Z_hist     = reorder_history_3d(mom_Z_raw,     t, r, Z)

rho_d_sil_hist = reorder_history_3d(rho_d_sil_raw, t, r, Z)
rho_d_c_hist   = reorder_history_3d(rho_d_c_raw,   t, r, Z)

momd_r_sil_hist = reorder_history_3d(momd_r_sil_raw, t, r, Z)
momd_r_c_hist   = reorder_history_3d(momd_r_c_raw,   t, r, Z)

momd_Z_sil_hist = reorder_history_3d(momd_Z_sil_raw, t, r, Z)
momd_Z_c_hist   = reorder_history_3d(momd_Z_c_raw,   t, r, Z)

sigmaLya_hist = reorder_history_3d(sigmaLya_raw, t, r, Z)

if TargetTime_kyr is None:
    k = final_saved_index(t)
else:
    k = nearest_saved_index(t, TargetTime_kyr * kyr)

print(f"Using file          = {Filename}")
print(f"Using snapshot      = {k}")
print(f"Snapshot time       = {t[k] / kyr:.3f} kyr")

rho_g  = rho_hist[k].astype(np.float64)
mom_r  = mom_r_hist[k].astype(np.float64)
mom_Z  = mom_Z_hist[k].astype(np.float64)

rho_d_sil = rho_d_sil_hist[k].astype(np.float64)
rho_d_c   = rho_d_c_hist[k].astype(np.float64)

momd_r_sil = momd_r_sil_hist[k].astype(np.float64)
momd_r_c   = momd_r_c_hist[k].astype(np.float64)

momd_Z_sil = momd_Z_sil_hist[k].astype(np.float64)
momd_Z_c   = momd_Z_c_hist[k].astype(np.float64)

sigmaLya = sigmaLya_hist[k].astype(np.float64)

rho_floor  = rho_floor_factor  * np.nanmax(rho_g)
rhod_floor = rhod_floor_factor * max(np.nanmax(rho_d_sil), np.nanmax(rho_d_c), 1.0)

u_r_g,   u_Z_g   = velocity_from_momentum(mom_r,       mom_Z,       rho_g,      rho_floor)
u_r_sil, u_Z_sil = velocity_from_momentum(momd_r_sil,  momd_Z_sil,  rho_d_sil,  rhod_floor)
u_r_c,   u_Z_c   = velocity_from_momentum(momd_r_c,    momd_Z_c,    rho_d_c,    rhod_floor)

vrel_sil = relative_speed_kms(u_r_sil, u_Z_sil, u_r_g, u_Z_g)
vrel_c   = relative_speed_kms(u_r_c,   u_Z_c,   u_r_g, u_Z_g)

# LogNorm cannot represent exactly zero; this affects only visualization.
vrel_sil_plot = np.where(np.isfinite(vrel_sil), np.maximum(vrel_sil, vrel_log_floor), np.nan)
vrel_c_plot   = np.where(np.isfinite(vrel_c),   np.maximum(vrel_c,   vrel_log_floor), np.nan)

# ============================================================
#   C O L O R   S C A L E S
# ============================================================

if AutoScale_sigmaLya:
    norm_sigmaLya = make_lognorm_from_arrays([sigmaLya], sigmaLya_vmin, sigmaLya_vmax,
                                             pct_low_log, pct_high_log)
else:
    norm_sigmaLya = LogNorm(vmin=sigmaLya_vmin, vmax=sigmaLya_vmax)

if AutoScale_vrel:
    norm_vrel = make_lognorm_from_arrays([vrel_sil_plot, vrel_c_plot], vrel_vmin, vrel_vmax,
                                         pct_low_log, pct_high_log)
else:
    norm_vrel = LogNorm(vmin=vrel_vmin, vmax=vrel_vmax)

sigma_ticks, sigma_ticklabels = ticks_sigma_custom, ticklabels_sigma_custom
if sigma_ticks is None:
    sigma_ticks, sigma_ticklabels = get_log_ticks_from_norm(norm_sigmaLya)

vrel_ticks, vrel_ticklabels = ticks_vrel_custom, ticklabels_vrel_custom
if vrel_ticks is None:
    vrel_ticks, vrel_ticklabels = get_log_ticks_from_norm(norm_vrel)

# Clip for plotting, preserving NaNs outside valid dust/gas density regions.
sigma_plot    = np.clip(sigmaLya.T,       norm_sigmaLya.vmin, norm_sigmaLya.vmax)
vrel_sil_plot = np.clip(vrel_sil_plot.T,  norm_vrel.vmin,     norm_vrel.vmax)
vrel_c_plot   = np.clip(vrel_c_plot.T,    norm_vrel.vmin,     norm_vrel.vmax)

R_edges_pc, Z_edges_pc = np.meshgrid(r / pc, Z / pc, indexing='ij')

# ============================================================
#   F I G U R E
# ============================================================

wpanel = (1.0 - 2.0 * wgap) / 3.0
fig = plt.figure(figsize=figsize)

ax1 = fig.add_axes([0.0,                0.0, wpanel, 1.0])
ax2 = fig.add_axes([wpanel + wgap,      0.0, wpanel, 1.0], sharex=ax1, sharey=ax1)
ax3 = fig.add_axes([2.0*(wpanel + wgap),0.0, wpanel, 1.0], sharex=ax1, sharey=ax1)
axes = [ax1, ax2, ax3]

im1 = ax1.pcolormesh(R_edges_pc, Z_edges_pc, sigma_plot,
                     norm=norm_sigmaLya, cmap=cmap_sigmaLya,
                     shading='auto', rasterized=rasterized)
im2 = ax2.pcolormesh(R_edges_pc, Z_edges_pc, vrel_sil_plot,
                     norm=norm_vrel, cmap=cmap_vrel_sil,
                     shading='auto', rasterized=rasterized)
im3 = ax3.pcolormesh(R_edges_pc, Z_edges_pc, vrel_c_plot,
                     norm=norm_vrel, cmap=cmap_vrel_c,
                     shading='auto', rasterized=rasterized)

# Axes styling: cinematic/no ticks, matching isolated-star style.
if xmax_pc is None:
    xmax_use = r[-1] / pc
else:
    xmax_use = xmax_pc

if zmax_pc is None:
    zmax_use = Z[-1] / pc
else:
    zmax_use = zmax_pc

for ax in axes:
    ax.set_xlim(r[0] / pc, xmax_use)
    ax.set_ylim(Z[0] / pc, zmax_use)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(3.0)

# Colorbars inside panels.
add_inset_colorbar(ax1, im1, sigma_ticks, sigma_ticklabels,
                   tick_color=cbar_tick_color_sigma, outline_color=cbar_outline_sigma)
add_inset_colorbar(ax2, im2, vrel_ticks, vrel_ticklabels,
                   tick_color=cbar_tick_color_vrel, outline_color=cbar_outline_vrel)
add_inset_colorbar(ax3, im3, vrel_ticks, vrel_ticklabels,
                   tick_color=cbar_tick_color_vrel, outline_color=cbar_outline_vrel)

# Text overlays.
label_time(ax1, t[k] / kyr)
label_panel(ax1, r"$\sigma_{d,Ly\alpha}/H$ (cm$^2$)")
label_panel(ax2, r"$|u_{d,Sil} - u_g|$ (km s$^{-1}$)")
label_panel(ax3, r"$|u_{d,C} - u_g|$ (km s$^{-1}$)")

for ax, letter in zip(axes, "abc"):
    label_abc(ax, letter)

plt.savefig(FigureName, bbox_inches='tight', pad_inches=0.02, dpi=300)
print(f"Saved figure        = {FigureName}")
