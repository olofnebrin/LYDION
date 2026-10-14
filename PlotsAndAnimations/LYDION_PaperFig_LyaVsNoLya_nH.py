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
import os

repo_root = Path(__file__).resolve().parents[1]
os.chdir(repo_root)
cmastro_src = repo_root / "src" / "cmastro" / "src"

sys.path.insert(0, str(cmastro_src))

import cmastro

# ---------------- User inputs ----------------
#Filename_noLya = "RHDSim_StarCluster_NoLya_512x512_2026-04-19.h5"    # Run ignoring Lyα feedback
#Filename_Lya   = "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5"      # Run with Lyα feedback
#FigureName     = "LYDION_Figure_LyaVsNoLya_512D0.01_4rows_nH.pdf"    # Or .png

Filename_noLya = "RHDSim_StarCluster_NoLya_512x512_2026-04-19.h5"    # Run ignoring Lyα feedback
Filename_Lya   = "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5"      # Run with Lyα feedback
FigureName     = "LYDION_Figure_LyaVsNoLya_512D0.01_4rows_nH.pdf"    # Or .png

Time1_kyr      = 5    # Row 0 snapshot time (in kyrs)
Time2_kyr      = 20   # Row 1 snapshot time (in kyrs)
Time3_kyr      = 30   # Row 2 snapshot time (in kyrs)
Time4_kyr      = 40   # Row 3 snapshot time (in kyrs)

fontsize_cbar_ticks  = 26  # the "10^3", "10^4" numbers
fontsize_panel_label = 40  # the "nH (cm-3)" text
fontsize_time        = 38

cmap_nH   = 'cma:laguna'
cmap_eLya = 'magma' 

# Diagnostics text options
ShowDiagnosticsText         = True
ShowLyaForceText            = True
ShowLyaMultiplierText       = True
ShowLyaEscapeFractionText   = True
ShowLyaLuminosityText       = False
ShowDirectForceText         = True
ShowIRForceText             = True

# Force variant: which Lyα force measure to show in the nH panels.
#   "radial_net"  -> uses M_F_Lya_radial (net outward radial force / (L_Lya/c)),
#                    falling back to M_F_Lya if radial is unavailable
#   "antigravity" -> uses M_F_Lya (total force multiplier, i.e. F_Lya / (L_Lya/c)),
#                    without preferring the radial-net variant
ForceVariant = "radial_net"    # "radial_net" | "antigravity"

fontsize_diagnostics_text   = 32
diagnostics_x               = 0.04
diagnostics_y               = 0.85

# Separate placement/font controls for f_esc labels in the e_Lyα panels.
# These are axes coordinates, so (0,0) = lower left and (1,1) = upper right.
fontsize_fesc_text = 38
fesc_positions = {
    0: (0.04, 0.81),   # row 0: moved away from the upper-right colorbar
    1: (0.04, 0.85),
    2: (0.04, 0.85),
    3: (0.04, 0.85),
}

# Optional scale bar. Default: lower right of the upper nH panel with Lyα.
ShowScaleBar = True
scale_bar_ax_name = "ax12"       # ax12 = upper nH panel with Lyα
scale_bar_length_pc = 0.2
scale_bar_label = "0.2 pc"

# Axes-coordinate placement of the scale bar centre.
scale_bar_x_axes = 0.74
scale_bar_y_axes = 0.08
scale_bar_color = "white"
scale_bar_lw = 3.0
scale_bar_capsize_axes = 0.045
scale_bar_label_pad_axes = 0.020
scale_bar_fontsize = 34
scale_bar_stroke = "black"
scale_bar_stroke_width = 4.0

# Useful constants
kB  = 1.38e-16
mH  = 1.67e-24
X   = 0.76
yr  = 3.1556926e7
kyr = 1e3 * yr
pc  = 3.085677581e18
c   = 2.99792458e10
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

def read_dataset_if_exists(file, name, default=None):
    return file[name][:] if name in file else default

def read_attr_if_exists(group, name, default=None):
    return group.attrs[name] if name in group.attrs else default

def read_first_existing_dataset(file, names, default=None):
    for name in names:
        if name in file:
            return file[name][:]
    return default

def safe_scalar_at_index(arr, idx):
    if arr is None:
        return None
    if idx < 0 or idx >= len(arr):
        return None
    val = arr[idx]
    try:
        val = float(np.asarray(val))
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return val

def fmt_value(x):
    if x is None or (not np.isfinite(x)):
        return None
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 100.0 or ax < 0.01:
        return f"{x:.2e}"
    elif ax >= 10.0:
        return f"{x:.1f}"
    else:
        return f"{x:.2f}"

def fmt_math_value(x):
    """
    Format number for mathtext. Uses scientific notation with superscript
    for very small / very large values.
    """
    if x is None or (not np.isfinite(x)):
        return None
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 100.0 or ax < 0.01:
        exp = int(np.floor(np.log10(ax)))
        mant = x / (10.0**exp)
        return rf"{mant:.2f}\times10^{{{exp}}}"
    elif ax >= 10.0:
        return f"{x:.1f}"
    else:
        return f"{x:.2f}"
    
def read_run(filename):
    """
    Read one run and return relevant arrays.
    """
    with h5py.File(filename, "r") as f:
        t        = f["Times"][:]
        r        = f["r"][:]
        Z        = f["Z"][:]
        rho_raw  = f["GasDensity_History"][:]

        eLya_raw = f["e_Lya_History"][:]       if "e_Lya_History"       in f else None

        settings = f["Settings"] if "Settings" in f else None
        L_bol    = read_attr_if_exists(settings, "L_bol", None) if settings is not None else None

        M_F_Lya         = read_dataset_if_exists(f, "M_F_History")
        M_F_Lya_radial  = read_dataset_if_exists(f, "M_F_radial_net_History")
        L_Lya           = read_dataset_if_exists(f, "L_Lya_History")
        f_esc_Lya       = read_first_existing_dataset(
            f,
            ["f_esc_Lya_History", "fesc_Lya_History", "f_esc_History_Lya"],
            default=None
        )

        M_F_dir         = read_dataset_if_exists(f, "M_F_dir_History")
        M_F_dir_radial  = read_dataset_if_exists(f, "M_F_dir_radial_net_History")

        M_F_IR          = read_dataset_if_exists(f, "M_F_IR_History")
        M_F_IR_radial   = read_dataset_if_exists(f, "M_F_IR_radial_net_History")
        L_IR            = read_dataset_if_exists(f, "L_IR_History")

    rho_hist  = reorder_history_3d(rho_raw, t, r, Z)
    nH_hist   = X * rho_hist / mH

    eLya_hist = reorder_history_3d(eLya_raw, t, r, Z) if eLya_raw is not None else None

    saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]

    return dict(
        t=t,
        r=r,
        Z=Z,
        rho_hist=rho_hist,
        nH_hist=nH_hist,
        eLya_hist=eLya_hist,
        saved_idx=saved_idx,
        L_bol=L_bol,
        M_F_Lya=M_F_Lya,
        M_F_Lya_radial=M_F_Lya_radial,
        L_Lya=L_Lya,
        f_esc_Lya=f_esc_Lya,
        M_F_dir=M_F_dir,
        M_F_dir_radial=M_F_dir_radial,
        M_F_IR=M_F_IR,
        M_F_IR_radial=M_F_IR_radial,
        L_IR=L_IR
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


def draw_scale_bar(ax):
    """Draw a simple |---| scale bar in axes coordinates, with length in pc."""
    if not ShowScaleBar:
        return

    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    xspan = xhi - xlo
    yspan = yhi - ylo

    # Convert centre from axes coordinates to data coordinates.
    xcen = xlo + scale_bar_x_axes * xspan
    ycen = ylo + scale_bar_y_axes * yspan

    x0 = xcen - 0.5 * scale_bar_length_pc
    x1 = xcen + 0.5 * scale_bar_length_pc

    capsize = scale_bar_capsize_axes * yspan
    label_pad = scale_bar_label_pad_axes * yspan

    path_effects = []
    if scale_bar_stroke is not None:
        path_effects = [
            pe.withStroke(
                linewidth=scale_bar_stroke_width,
                foreground=scale_bar_stroke,
            )
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


def _pick_Lya_force(run, k):
    """
    Return the Lyα force in units of L_bol/c, honouring ForceVariant.

      "radial_net"  -> prefers M_F_Lya_radial * L_Lya / L_bol,
                       falls back to M_F_Lya * L_Lya / L_bol
      "antigravity" -> always uses M_F_Lya * L_Lya / L_bol (the total
                       force multiplier, not projected onto the radial axis)

    Returns the scalar value, or None if unavailable.
    """
    L_Lya_val = safe_scalar_at_index(run["L_Lya"], k)
    L_bol_val = run["L_bol"]
    try:
        if L_bol_val is not None:
            L_bol_val = float(np.asarray(L_bol_val))
    except Exception:
        L_bol_val = None

    if (L_Lya_val is None) or (L_bol_val is None):
        return None

    M_F_val        = safe_scalar_at_index(run["M_F_Lya"], k)
    M_F_radial_val = safe_scalar_at_index(run["M_F_Lya_radial"], k)

    if ForceVariant == "radial_net":
        if M_F_radial_val is not None:
            return M_F_radial_val * L_Lya_val / L_bol_val
        elif M_F_val is not None:
            return M_F_val * L_Lya_val / L_bol_val
    else:  # "antigravity"
        if M_F_val is not None:
            return M_F_val * L_Lya_val / L_bol_val

    return None

def make_radiation_diagnostics_lines(run, k):
    """
    Build diagnostic text lines for the nH panels.
    Does NOT include f_esc_Lya — that goes in the e_Lyα panels via
    make_eLya_diagnostics_lines().
    """
    lines = []

    L_Lya_val      = safe_scalar_at_index(run["L_Lya"], k)

    M_F_dir_val        = safe_scalar_at_index(run["M_F_dir"], k)
    M_F_dir_radial_val = safe_scalar_at_index(run["M_F_dir_radial"], k)

    M_F_IR_val         = safe_scalar_at_index(run["M_F_IR"], k)
    M_F_IR_radial_val  = safe_scalar_at_index(run["M_F_IR_radial"], k)
    L_IR_val           = safe_scalar_at_index(run["L_IR"], k)

    L_bol_val = run["L_bol"]
    try:
        if L_bol_val is not None:
            L_bol_val = float(np.asarray(L_bol_val))
    except Exception:
        L_bol_val = None

    if ShowLyaForceText:
        F_Lya = _pick_Lya_force(run, k)
        if F_Lya is not None:
            F_str = fmt_math_value(F_Lya)
            if F_str is not None:
                lines.append(rf"$F_{{Ly\alpha,radial}} = {F_str}\,L_{{bol}}/c$")

    if ShowDirectForceText and (L_bol_val is not None):
        if M_F_dir_radial_val is not None:
            F_dir_over_Lbolc = M_F_dir_radial_val
            F_str = fmt_math_value(F_dir_over_Lbolc)
            if F_str is not None:
                lines.append(rf"$F_{{dir,radial}} = {F_str}\,L_{{bol}}/c$")
        elif M_F_dir_val is not None:
            F_dir_over_Lbolc = M_F_dir_val
            F_str = fmt_math_value(F_dir_over_Lbolc)
            if F_str is not None:
                lines.append(rf"$F_{{dir}} = {F_str}\,L_{{bol}}/c$")

    if ShowIRForceText and (L_bol_val is not None) and (L_IR_val is not None):
        if M_F_IR_radial_val is not None:
            F_IR_over_Lbolc = M_F_IR_radial_val * L_IR_val / L_bol_val
            F_str = fmt_math_value(F_IR_over_Lbolc)
            if F_str is not None:
                lines.append(rf"$F_{{IR,radial}} = {F_str}\,L_{{bol}}/c$")
        elif M_F_IR_val is not None:
            F_IR_over_Lbolc = M_F_IR_val * L_IR_val / L_bol_val
            F_str = fmt_math_value(F_IR_over_Lbolc)
            if F_str is not None:
                lines.append(rf"$F_{{IR}} = {F_str}\,L_{{bol}}/c$")

    if ShowLyaMultiplierText:
        if ForceVariant == "radial_net":
            M_F_val = safe_scalar_at_index(run["M_F_Lya_radial"], k)
            if M_F_val is None:
                M_F_val = safe_scalar_at_index(run["M_F_Lya"], k)
        else:
            M_F_val = safe_scalar_at_index(run["M_F_Lya"], k)
            
        if M_F_val is not None:
            M_str = fmt_math_value(M_F_val)
            if M_str is not None:
                lines.append(rf"$M_{{F,radial}} = {M_str}$")

    if ShowLyaLuminosityText and (L_bol_val is not None) and (L_Lya_val is not None):
        ratio = L_Lya_val / L_bol_val
        ratio_str = fmt_math_value(ratio)
        if ratio_str is not None:
            lines.append(rf"$L_{{Ly\alpha}} = {ratio_str}\,L_{{bol}}$")

    return lines

def make_eLya_diagnostics_lines(run, k):
    """
    Build diagnostic text lines for the e_Lyα panels.
    Currently contains only f_esc_Lya.
    """
    lines = []

    if ShowLyaEscapeFractionText:
        f_esc_val = safe_scalar_at_index(run["f_esc_Lya"], k)
        if f_esc_val is not None:
            fesc_str = fmt_math_value(f_esc_val)
            if fesc_str is not None:
                lines.append(rf"$f_{{esc,Ly\alpha}} = {fesc_str}$")

    return lines

# -------------------------------------------------------------------
# Read both runs
# -------------------------------------------------------------------

run_noLya = read_run(Filename_noLya)
run_Lya   = read_run(Filename_Lya)

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
# Find best matching snapshot pairs for all four times
# -------------------------------------------------------------------

k1_noLya, k1_Lya = find_best_time_pair(Time1_kyr, run_noLya, run_Lya)
k2_noLya, k2_Lya = find_best_time_pair(Time2_kyr, run_noLya, run_Lya)
k3_noLya, k3_Lya = find_best_time_pair(Time3_kyr, run_noLya, run_Lya)
k4_noLya, k4_Lya = find_best_time_pair(Time4_kyr, run_noLya, run_Lya)

print(f"Requested Time1 = {Time1_kyr:.3f} kyr")
print(f"  No-Lyα snapshot = {run_noLya['t'][k1_noLya]/kyr:.3f} kyr")
print(f"  Lyα snapshot    = {run_Lya['t'][k1_Lya]/kyr:.3f} kyr")

print(f"Requested Time2 = {Time2_kyr:.3f} kyr")
print(f"  No-Lyα snapshot = {run_noLya['t'][k2_noLya]/kyr:.3f} kyr")
print(f"  Lyα snapshot    = {run_Lya['t'][k2_Lya]/kyr:.3f} kyr")

print(f"Requested Time3 = {Time3_kyr:.3f} kyr")
print(f"  No-Lyα snapshot = {run_noLya['t'][k3_noLya]/kyr:.3f} kyr")
print(f"  Lyα snapshot    = {run_Lya['t'][k3_Lya]/kyr:.3f} kyr")

print(f"Requested Time4 = {Time4_kyr:.3f} kyr")
print(f"  No-Lyα snapshot = {run_noLya['t'][k4_noLya]/kyr:.3f} kyr")
print(f"  Lyα snapshot    = {run_Lya['t'][k4_Lya]/kyr:.3f} kyr")

# -------------------------------------------------------------------
# Prepare grids for plotting
# -------------------------------------------------------------------

r = run_noLya["r"]
Z = run_noLya["Z"]

R_edges_pc, Z_edges_pc = np.meshgrid(r / pc, Z / pc, indexing='ij')

# -------------------------------------------------------------------
# Choose a common density scale for all eight nH panels
# -------------------------------------------------------------------

nH0_noLya = run_noLya["nH_hist"][0]
nH0_Lya   = run_Lya["nH_hist"][0]

nH0_noLya_pos = nH0_noLya[nH0_noLya > 0.0]
nH0_Lya_pos   = nH0_Lya[nH0_Lya > 0.0]

if (nH0_noLya_pos.size == 0) or (nH0_Lya_pos.size == 0):
    raise ValueError("Initial nH field has no positive values in one of the runs.")

nH0_char = max(np.max(nH0_noLya_pos), np.max(nH0_Lya_pos))

vmin_nH = 10.0**(np.log10(nH0_char) - 2.0 - 0.5)
vmax_nH = 10.0**(np.log10(nH0_char) + 2.0 - 0.5)
norm_nH = LogNorm(vmin=vmin_nH, vmax=vmax_nH)

logmin_nH = int(np.ceil(np.log10(vmin_nH)))
logmax_nH = int(np.floor(np.log10(vmax_nH)))
ticks_nH  = 10.0**np.arange(logmin_nH, logmax_nH + 1)
ticklabels_nH = [fmt_log_tick(v) for v in ticks_nH]

# -------------------------------------------------------------------
# Choose scales for Lyα-only panels
# -------------------------------------------------------------------

norm_eLya  = LogNorm(vmin=1e-14, vmax=1e-6)

# -------------------------------------------------------------------
# Load snapshots
# -------------------------------------------------------------------

eps = 1e-30

if run_Lya["eLya_hist"] is None:
    raise RuntimeError("e_Lya_History not found in Lyα file.")

# Density — all four rows, both runs
nH_11 = np.clip(run_noLya["nH_hist"][k1_noLya].T, vmin_nH, vmax_nH)
nH_12 = np.clip(run_Lya["nH_hist"][k1_Lya].T,     vmin_nH, vmax_nH)
nH_21 = np.clip(run_noLya["nH_hist"][k2_noLya].T,  vmin_nH, vmax_nH)
nH_22 = np.clip(run_Lya["nH_hist"][k2_Lya].T,      vmin_nH, vmax_nH)
nH_31 = np.clip(run_noLya["nH_hist"][k3_noLya].T,  vmin_nH, vmax_nH)
nH_32 = np.clip(run_Lya["nH_hist"][k3_Lya].T,      vmin_nH, vmax_nH)
nH_41 = np.clip(run_noLya["nH_hist"][k4_noLya].T,  vmin_nH, vmax_nH)
nH_42 = np.clip(run_Lya["nH_hist"][k4_Lya].T,      vmin_nH, vmax_nH)

# e_Lyα — all four rows (Lyα run only, col 2)
eLya_1 = np.clip(run_Lya["eLya_hist"][k1_Lya].T, eps, None)
eLya_2 = np.clip(run_Lya["eLya_hist"][k2_Lya].T, eps, None)
eLya_3 = np.clip(run_Lya["eLya_hist"][k3_Lya].T, eps, None)
eLya_4 = np.clip(run_Lya["eLya_hist"][k4_Lya].T, eps, None)

# -------------------------------------------------------------------
# Set up figure
#
# Layout (4 rows × 3 cols, all uniform height):
#   Row 0: no-Lyα nH t1  | Lyα nH t1  | Lyα e_Lyα t1
#   Row 1: no-Lyα nH t2  | Lyα nH t2  | Lyα e_Lyα t2
#   Row 2: no-Lyα nH t3  | Lyα nH t3  | Lyα e_Lyα t3
#   Row 3: no-Lyα nH t4  | Lyα nH t4  | Lyα e_Lyα t4
# -------------------------------------------------------------------

wgap_big   = 0.02          # horizontal gap between no-Lyα and Lyα columns
wgap_small = 0.000         # horizontal gap between Lyα density and e_Lyα columns
hgap       = -0.01         # vertical gap between rows

wpanel = (1.0 - wgap_big - wgap_small) / 3.0
hpanel = (1.0 - 3.0 * hgap) / 4.0

# Column left edges
col0_left = 0.0
col1_left = wpanel + wgap_big
col2_left = col1_left + wpanel + wgap_small

# Row bottom edges (row 0 is the topmost)
row0_bottom = 3.0 * (hpanel + hgap)
row1_bottom = 2.0 * (hpanel + hgap)
row2_bottom = 1.0 * (hpanel + hgap)
row3_bottom = 0.0

# Top of the figure (for column title positioning)
fig_top = row0_bottom + hpanel

def make_ax(fig, col_left, row_bottom, width, height, sharex=None, sharey=None):
    ax = fig.add_axes([col_left, row_bottom, width, height],
                      sharex=sharex, sharey=sharey)
    return ax

fig = plt.figure(figsize=(24, 32))

ax11 = make_ax(fig, col0_left, row0_bottom, wpanel, hpanel)                              # no-Lyα nH t1
ax12 = make_ax(fig, col1_left, row0_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)    # Lyα nH t1
ax13 = make_ax(fig, col2_left, row0_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)    # Lyα e_Lyα t1

ax21 = make_ax(fig, col0_left, row1_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)    # no-Lyα nH t2
ax22 = make_ax(fig, col1_left, row1_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)    # Lyα nH t2
ax23 = make_ax(fig, col2_left, row1_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)    # Lyα e_Lyα t2

ax31 = make_ax(fig, col0_left, row2_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)    # no-Lyα nH t3
ax32 = make_ax(fig, col1_left, row2_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)    # Lyα nH t3
ax33 = make_ax(fig, col2_left, row2_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)    # Lyα e_Lyα t3

ax41 = make_ax(fig, col0_left, row3_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)    # no-Lyα nH t4
ax42 = make_ax(fig, col1_left, row3_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)    # Lyα nH t4
ax43 = make_ax(fig, col2_left, row3_bottom, wpanel, hpanel, sharex=ax11, sharey=ax11)    # Lyα e_Lyα t4

# -------------------------------------------------------------------
# Plot density panels (all rows, cols 0–1)
# -------------------------------------------------------------------

im11 = ax11.pcolormesh(R_edges_pc, Z_edges_pc, nH_11, norm=norm_nH, shading='auto',
                       cmap=cmap_nH, rasterized=True)
im12 = ax12.pcolormesh(R_edges_pc, Z_edges_pc, nH_12, norm=norm_nH, shading='auto',
                       cmap=cmap_nH, rasterized=True)
im21 = ax21.pcolormesh(R_edges_pc, Z_edges_pc, nH_21, norm=norm_nH, shading='auto',
                       cmap=cmap_nH, rasterized=True)
im22 = ax22.pcolormesh(R_edges_pc, Z_edges_pc, nH_22, norm=norm_nH, shading='auto',
                       cmap=cmap_nH, rasterized=True)
im31 = ax31.pcolormesh(R_edges_pc, Z_edges_pc, nH_31, norm=norm_nH, shading='auto',
                       cmap=cmap_nH, rasterized=True)
im32 = ax32.pcolormesh(R_edges_pc, Z_edges_pc, nH_32, norm=norm_nH, shading='auto',
                       cmap=cmap_nH, rasterized=True)
im41 = ax41.pcolormesh(R_edges_pc, Z_edges_pc, nH_41, norm=norm_nH, shading='auto',
                       cmap=cmap_nH, rasterized=True)
im42 = ax42.pcolormesh(R_edges_pc, Z_edges_pc, nH_42, norm=norm_nH, shading='auto',
                       cmap=cmap_nH, rasterized=True)

# -------------------------------------------------------------------
# Plot e_Lyα panels (all rows, col 2)
# -------------------------------------------------------------------

im13 = ax13.pcolormesh(R_edges_pc, Z_edges_pc, eLya_1, norm=norm_eLya,
                       shading='auto', cmap=cmap_eLya, rasterized=True)
im23 = ax23.pcolormesh(R_edges_pc, Z_edges_pc, eLya_2, norm=norm_eLya,
                       shading='auto', cmap=cmap_eLya, rasterized=True)
im33 = ax33.pcolormesh(R_edges_pc, Z_edges_pc, eLya_3, norm=norm_eLya,
                       shading='auto', cmap=cmap_eLya, rasterized=True)
im43 = ax43.pcolormesh(R_edges_pc, Z_edges_pc, eLya_4, norm=norm_eLya,
                       shading='auto', cmap=cmap_eLya, rasterized=True)

# -------------------------------------------------------------------
# Make all panels cinematic (no ticks, spines visible)
# -------------------------------------------------------------------

for ax in (ax11, ax12, ax13, ax21, ax22, ax23, ax31, ax32, ax33, ax41, ax42, ax43):
    ax.set_xlim(r[0] / pc, 0.8 * r[-1] / pc)
    ax.set_ylim(Z[0] / pc, 0.8 * Z[-1] / pc)
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
ax11.text(0.5, 1.04, "Ignoring Ly\u03b1 Radiation Pressure", transform=ax11.transAxes, **title_style)

# Lyα title: centred over cols 1–2 in figure coordinates
lya_title_x = 0.5 * (col1_left + col2_left + wpanel)
lya_title_y = fig_top
fig.text(lya_title_x, lya_title_y - 0.002, "With Ly\u03b1 Radiation Pressure", **title_style)

# -------------------------------------------------------------------
# Time labels (inside panels, white text)
# -------------------------------------------------------------------

time_style = dict(
    ha='left',
    va='top',
    color='white',
    path_effects=[pe.withStroke(linewidth=3, foreground='black')]
)

diagnostics_style = dict(
    ha='left',
    va='top',
    color='white',
    path_effects=[pe.withStroke(linewidth=3, foreground='black')]
)

panel_label_style_white = dict(
    ha='right',
    va='bottom',
    color='white',
    path_effects=[pe.withStroke(linewidth=3.5, foreground='black')]
)

# Row 0
ax11.text(0.04, 0.96, f"t = {run_noLya['t'][k1_noLya]/kyr:.2f} kyr",
          transform=ax11.transAxes, **time_style, fontsize=fontsize_time)
ax12.text(0.04, 0.96, f"t = {run_Lya['t'][k1_Lya]/kyr:.2f} kyr",
          transform=ax12.transAxes, **time_style, fontsize=fontsize_time)
ax13.text(0.04, 0.96, f"t = {run_Lya['t'][k1_Lya]/kyr:.2f} kyr",
          transform=ax13.transAxes, **time_style, fontsize=fontsize_time)

# Row 1
ax21.text(0.04, 0.96, f"t = {run_noLya['t'][k2_noLya]/kyr:.2f} kyr",
          transform=ax21.transAxes, **time_style, fontsize=fontsize_time)
ax22.text(0.04, 0.96, f"t = {run_Lya['t'][k2_Lya]/kyr:.2f} kyr",
          transform=ax22.transAxes, **time_style, fontsize=fontsize_time)
ax23.text(0.04, 0.96, f"t = {run_Lya['t'][k2_Lya]/kyr:.2f} kyr",
          transform=ax23.transAxes, **time_style, fontsize=fontsize_time)

# Row 2
ax31.text(0.04, 0.96, f"t = {run_noLya['t'][k3_noLya]/kyr:.2f} kyr",
          transform=ax31.transAxes, **time_style, fontsize=fontsize_time)
ax32.text(0.04, 0.96, f"t = {run_Lya['t'][k3_Lya]/kyr:.2f} kyr",
          transform=ax32.transAxes, **time_style, fontsize=fontsize_time)
ax33.text(0.04, 0.96, f"t = {run_Lya['t'][k3_Lya]/kyr:.2f} kyr",
          transform=ax33.transAxes, **time_style, fontsize=fontsize_time)

# Row 3
ax41.text(0.04, 0.96, f"t = {run_noLya['t'][k4_noLya]/kyr:.2f} kyr",
          transform=ax41.transAxes, **time_style, fontsize=fontsize_time)
ax42.text(0.04, 0.96, f"t = {run_Lya['t'][k4_Lya]/kyr:.2f} kyr",
          transform=ax42.transAxes, **time_style, fontsize=fontsize_time)
ax43.text(0.04, 0.96, f"t = {run_Lya['t'][k4_Lya]/kyr:.2f} kyr",
          transform=ax43.transAxes, **time_style, fontsize=fontsize_time)

# Diagnostics text in nH panels (forces and multiplier, but NOT f_esc)
if ShowDiagnosticsText:
    lines_no_t1 = make_radiation_diagnostics_lines(run_noLya, k1_noLya)
    lines_ly_t1 = make_radiation_diagnostics_lines(run_Lya,   k1_Lya)
    lines_no_t2 = make_radiation_diagnostics_lines(run_noLya, k2_noLya)
    lines_ly_t2 = make_radiation_diagnostics_lines(run_Lya,   k2_Lya)
    lines_no_t3 = make_radiation_diagnostics_lines(run_noLya, k3_noLya)
    lines_ly_t3 = make_radiation_diagnostics_lines(run_Lya,   k3_Lya)
    lines_no_t4 = make_radiation_diagnostics_lines(run_noLya, k4_noLya)
    lines_ly_t4 = make_radiation_diagnostics_lines(run_Lya,   k4_Lya)

    if len(lines_no_t1) > 0:
        ax11.text(diagnostics_x, diagnostics_y, "\n".join(lines_no_t1),
                  transform=ax11.transAxes, fontsize=fontsize_diagnostics_text,
                  **diagnostics_style)

    if len(lines_ly_t1) > 0:
        ax12.text(diagnostics_x, diagnostics_y, "\n".join(lines_ly_t1),
                  transform=ax12.transAxes, fontsize=fontsize_diagnostics_text,
                  **diagnostics_style)

    if len(lines_no_t2) > 0:
        ax21.text(diagnostics_x, diagnostics_y, "\n".join(lines_no_t2),
                  transform=ax21.transAxes, fontsize=fontsize_diagnostics_text,
                  **diagnostics_style)

    if len(lines_ly_t2) > 0:
        ax22.text(diagnostics_x, diagnostics_y, "\n".join(lines_ly_t2),
                  transform=ax22.transAxes, fontsize=fontsize_diagnostics_text,
                  **diagnostics_style)

    if len(lines_no_t3) > 0:
        ax31.text(diagnostics_x, diagnostics_y, "\n".join(lines_no_t3),
                  transform=ax31.transAxes, fontsize=fontsize_diagnostics_text,
                  **diagnostics_style)

    if len(lines_ly_t3) > 0:
        ax32.text(diagnostics_x, diagnostics_y, "\n".join(lines_ly_t3),
                  transform=ax32.transAxes, fontsize=fontsize_diagnostics_text,
                  **diagnostics_style)

    if len(lines_no_t4) > 0:
        ax41.text(diagnostics_x, diagnostics_y, "\n".join(lines_no_t4),
                  transform=ax41.transAxes, fontsize=fontsize_diagnostics_text,
                  **diagnostics_style)

    if len(lines_ly_t4) > 0:
        ax42.text(diagnostics_x, diagnostics_y, "\n".join(lines_ly_t4),
                  transform=ax42.transAxes, fontsize=fontsize_diagnostics_text,
                  **diagnostics_style)

# f_esc_Lya text in e_Lyα panels (col 2, all rows).
# Placement/font size are controlled separately from the force diagnostics.
if ShowDiagnosticsText:
    fesc_axes = [ax13, ax23, ax33, ax43]
    fesc_indices = [k1_Lya, k2_Lya, k3_Lya, k4_Lya]

    for row_index, (ax_fesc, k_fesc) in enumerate(zip(fesc_axes, fesc_indices)):
        esc_lines = make_eLya_diagnostics_lines(run_Lya, k_fesc)
        if len(esc_lines) > 0:
            x_fesc, y_fesc = fesc_positions.get(row_index, (diagnostics_x, diagnostics_y))
            ax_fesc.text(
                x_fesc,
                y_fesc,
                "\n".join(esc_lines),
                transform=ax_fesc.transAxes,
                fontsize=fontsize_fesc_text,
                **diagnostics_style,
            )

ax11.text(0.94, 0.08, r"$n_H$ (cm$^{-3}$)",
          transform=ax11.transAxes, fontsize=fontsize_panel_label,
          **panel_label_style_white)

ax13.text(0.94, 0.08, r"$e_{Ly\alpha}$ (erg cm$^{-3}$)",
          transform=ax13.transAxes, fontsize=fontsize_panel_label,
          **panel_label_style_white)

# -------------------------------------------------------------------
# Scale bar
# -------------------------------------------------------------------

scale_bar_axes = {
    "ax11": ax11, "ax12": ax12, "ax13": ax13,
    "ax21": ax21, "ax22": ax22, "ax23": ax23,
    "ax31": ax31, "ax32": ax32, "ax33": ax33,
    "ax41": ax41, "ax42": ax42, "ax43": ax43,
}

if ShowScaleBar:
    if scale_bar_ax_name not in scale_bar_axes:
        raise ValueError(
            f"scale_bar_ax_name={scale_bar_ax_name!r} not recognized. "
            f"Choose from {list(scale_bar_axes.keys())}."
        )
    draw_scale_bar(scale_bar_axes[scale_bar_ax_name])

# -------------------------------------------------------------------
# Colorbars (all inset, top-right of their host panel)
# -------------------------------------------------------------------

def make_inset_cbar(fig, ax, im, rect, ticks, ticklabels, dark=False):
    """
    Shared helper for all inset horizontal colorbars.
    dark=True  -> black text/outline on white background
    dark=False -> white text/outline on black background
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
                        length=0, width=0, pad=10,
                        bottom=False, top=False)
    cbar.ax.minorticks_off()
    cbar.ax.set_facecolor(bg)
    return cbar

# Density colorbar — row 0, col 0
make_inset_cbar(fig, ax11, im11,
                rect=[0.56, 0.91, 0.38, 0.045],
                ticks=ticks_nH, ticklabels=ticklabels_nH)

# e_Lyα colorbar — row 0, col 2
ticks_eLya      = [1e-14, 1e-10, 1e-6]
ticklabels_eLya = [fmt_log_tick(v) for v in ticks_eLya]
make_inset_cbar(fig, ax13, im13,
                rect=[0.54, 0.91, 0.40, 0.045],
                ticks=ticks_eLya, ticklabels=ticklabels_eLya)

# -------------------------------------------------------------------
# Save figure
# -------------------------------------------------------------------

plt.savefig(FigureName, bbox_inches='tight', dpi = 600)
plt.show()