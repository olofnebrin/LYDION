import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter, FuncFormatter
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
cmastro_src = repo_root / "src" / "cmastro" / "src"

sys.path.insert(0, str(cmastro_src))

import cmastro


###############################################################
#######                  U S E R  I N P U T             #######
###############################################################

# Files should be in the same directory as this script, unless full paths are given.
LYDION_FileName = "CloudTest_theta0.7853981633974483_rs0.5_tau1.0e8_T10000.0_umax0.0_tauabs0.0_pd0.0_256x256.h5"

# COLT/MCRT files. All are assumed to use 1e7 photons.
# The plotting code below uses the order requested in the paper figure:
#   top row:    LYDION, xcrit0/no core-skipping, xcrit1
#   bottom row: xcrit2, xcrit3, nHI
COLT_FileNames = {
    0: "Lya-cloud-xcrit0_1e7photons.hdf5",
    1: "Lya-cloud-xcrit1_1e7photons.hdf5",
    2: "Lya-cloud-xcrit2_1e7photons.hdf5",
    3: "Lya-cloud-xcrit3_1e7photons.hdf5",
}

outname = "ConeTest_eLya.pdf"
savefig = True

# Which LYDION snapshot to plot:
#   "final" -> final saved snapshot
#   integer -> explicit snapshot index
snapshot_to_plot = "final"

# Which COLT/MCRT field to compare with LYDION e_Lya.
# Aaron wrote that u_rad_r is the energy density of each cell.
COLT_eLya_dataset = "u_rad_r"

# H I number-density dataset in the LYDION static-cloud HDF5 file.
nHI_dataset = "nHI"

# Aaron's output has the full Z-domain because he did not use reflective BCs.
# LYDION likely has only the reflective half-plane.
#
# Options:
#   "upper" -> use second half of COLT Z array, assuming COLT full domain is ordered [-Zmax, ..., +Zmax]
#   "lower" -> use first half
#   "auto"  -> choose whichever half has closer total energy to LYDION, after optional scaling below
#   "full"  -> plot full COLT Z-domain; not directly comparable panel size to LYDION
colt_half_to_plot = "upper"

# Optional normalization of COLT to LYDION.
# For a strict physical comparison, keep "none".
#
# Options:
#   "none"        -> plot raw COLT field
#   "match_total" -> multiply COLT by sum(LYDION)/sum(COLT)
#   "match_max"   -> multiply COLT by max(LYDION)/max(COLT)
colt_normalization = "none"

# Extra manual scale factor applied after colt_normalization.
# Useful if Aaron's u_rad_r differs by a known unit/convention factor.
colt_scale_factor = 1.0

# Plot scale.
eLya_scale = "log"       # "log" or "linear"
nHI_scale  = "log"       # "log" or "linear"

# Shared e_Lya colorbar range for LYDION and all COLT/MCRT panels.
# Set to None for automatic.
eLya_vmin_user = None
eLya_vmax_user = 1e-7

# nHI colorbar range.
# Set to None for automatic.
nHI_vmin_user = 0.01
nHI_vmax_user = 2e3

# For log plots, automatic vmin is max_positive * log_vmin_fraction.
eLya_log_vmin_fraction = 1e-7
eLya_log_vmax_factor   = 1.0

nHI_log_vmin_fraction = 1e-8
nHI_log_vmax_factor   = 10.0

# Colormaps.
eLya_cmap = "magma"
nHI_cmap  = "cma:laguna"

# Figure size and layout.
figsize = (27, 15.0)
dpi_save = 400

# Manual panel layout in figure-fraction units.
#
# This avoids the common matplotlib issue where equal-aspect axes,
# GridSpec, and appended colorbar axes produce visually uneven spacing.
#
# Useful knobs:
#   panel_col_gap      -> horizontal gap between the six map panels
#   panel_row_gap      -> vertical gap between the two rows
#   panel_height       -> physical height of each square panel, as a fraction of figure height
#   colorbar_gap       -> gap between the last map column and the colorbars
#   colorbar_width     -> width of each colorbar
#
# The panel width is computed from panel_height and figsize so that
# the axes are square in physical units.
panel_left      = 0.075
panel_bottom    = 0.100
panel_height    = 0.390
panel_col_gap   = -0.006
panel_row_gap   = 0.02
colorbar_gap    = 0.018
colorbar_width  = 0.016


###############################################################
#######                H E L P E R S                    #######
###############################################################

def log_tick_formatter_small_decimal(x, pos=None):
    """
    Format log ticks so modest powers are written as normal numbers.

    Examples:
        1e-2 -> 0.01
        1e-1 -> 0.1
        1e0  -> 1
        1e1  -> 10
        1e2  -> 100
        1e3  -> 10^3
    """
    if x <= 0.0 or not np.isfinite(x):
        return ""

    exponent = np.log10(x)
    exponent_round = int(np.round(exponent))

    # Only label exact powers of ten.
    if not np.isclose(exponent, exponent_round, atol=1e-10, rtol=0.0):
        return ""

    if -2 <= exponent_round <= 2:
        return f"{10.0**exponent_round:g}"

    return rf"$10^{{{exponent_round}}}$"


def compact_tick_formatter(x, pos=None):
    """
    Compact linear tick labels.

    This prints 0 and 1 instead of 0.0 and 1.0, while keeping
    non-integer labels such as 0.25 or 0.5 compact.
    """
    if not np.isfinite(x):
        return ""

    if abs(x) < 5e-13:
        x = 0.0

    return f"{x:g}"


def script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def resolve_path(fname):
    if os.path.isabs(fname):
        return fname
    return os.path.join(script_dir(), fname)


def reorder_history_3d(arr, t, r_edges, Z_edges):
    """
    Reorder a 3D LYDION history array to shape (Nt, Nr, Nz).

    This follows the same logic used in the animation scripts:
    identify the time axis from Times, then identify whether the
    two spatial axes are (Nr,Nz) or (Nz,Nr).
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
            f"Could not uniquely identify time axis in array with shape {shape} and Nt={Nt}"
        )

    time_axis = time_axes[0]
    arr_tfirst = np.moveaxis(arr, time_axis, 0)

    _, a1, a2 = arr_tfirst.shape

    if (a1, a2) == (Nr, Nz):
        return arr_tfirst

    if (a1, a2) == (Nz, Nr):
        return arr_tfirst.transpose(0, 2, 1)

    raise RuntimeError(
        f"After moving time axis, spatial dims {(a1, a2)} do not match "
        f"Nr={Nr}, Nz={Nz}. Raw shape was {shape}."
    )


def read_lydion_file(fname, snapshot_to_plot):
    path = resolve_path(fname)

    with h5py.File(path, "r") as f:
        t    = f["Times"][:]
        r    = f["r"][:]
        Z    = f["Z"][:]
        rCoM = f["rCoM"][:]
        ZCoM = f["ZCoM"][:]

        e_raw = f["e_Lya_History"][:]

        if nHI_dataset not in f:
            print("\nDatasets/groups in LYDION file:")
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  Dataset: {name}, shape={obj.shape}")
                elif isinstance(obj, h5py.Group):
                    print(f"  Group:   {name}")
            f.visititems(visitor)
            raise RuntimeError(f"Dataset {nHI_dataset!r} not found in {path}")

        nHI_raw = f[nHI_dataset][:]

    e_hist = reorder_history_3d(e_raw, t, r, Z)

    # ------------------------------------------------------------------
    # Snapshot selection
    # ------------------------------------------------------------------
    # Some LYDION files may contain preallocated/default history slots.
    # In that case len(Times)-1 can point to an unused entry with t = 0.
    #
    # "final" therefore means:
    #   use the last snapshot with t > 0, if any exists;
    #   otherwise fall back to index 0.
    #
    # You can still manually set snapshot_to_plot = 5, 10, ... above.
    if snapshot_to_plot == "final":
        positive_idx = np.where(np.isfinite(t) & (t > 0.0))[0]

        if positive_idx.size > 0:
            k = int(positive_idx[-1])
        else:
            k = 0
            print("WARNING: No positive entries found in Times. Using snapshot index 0.")

    else:
        k = int(snapshot_to_plot)

    if k < 0 or k >= len(t):
        raise RuntimeError(f"snapshot_to_plot={k} outside valid range [0, {len(t)-1}]")

    # Helpful diagnostics, especially for files with unused zero-time slots.
    positive_idx = np.where(np.isfinite(t) & (t > 0.0))[0]
    print("\n========== LYDION time diagnostics ==========")
    print(f"Times shape:              {t.shape}")
    print(f"First 10 Times:           {t[:10]}")
    print(f"Last 10 Times:            {t[-10:]}")
    print(f"Number of t > 0 entries:  {positive_idx.size}")
    if positive_idx.size > 0:
        print(f"First positive index/time: {positive_idx[0]}, {t[positive_idx[0]]:.6e}")
        print(f"Last positive index/time:  {positive_idx[-1]}, {t[positive_idx[-1]]:.6e}")
    print(f"Selected snapshot:        k = {k}, t = {t[k]:.6e}")
    print("============================================\n")

    e = np.asarray(e_hist[k, :, :], dtype=float)
    nHI = np.asarray(nHI_raw, dtype=float)

    Nr = len(r) - 1
    Nz = len(Z) - 1
    if nHI.shape == (Nz, Nr):
        nHI = nHI.T
    elif nHI.shape != (Nr, Nz):
        raise RuntimeError(
            f"nHI shape {nHI.shape} does not match expected (Nr,Nz)=({Nr},{Nz}) "
            f"or transposed (Nz,Nr)=({Nz},{Nr})."
        )

    return {
        "path": path,
        "t": t,
        "snapshot_index": k,
        "r": r,
        "Z": Z,
        "rCoM": rCoM,
        "ZCoM": ZCoM,
        "e_Lya": e,
        "nHI": nHI,
    }


def read_colt_field(fname, dataset_name):
    path = resolve_path(fname)

    with h5py.File(path, "r") as f:
        if dataset_name not in f:
            print("\nDatasets/groups in COLT file:")
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  Dataset: {name}, shape={obj.shape}")
                elif isinstance(obj, h5py.Group):
                    print(f"  Group:   {name}")
            f.visititems(visitor)
            raise RuntimeError(f"Dataset {dataset_name!r} not found in {path}")

        arr = f[dataset_name][:]

    return path, np.asarray(arr, dtype=float)


def reshape_colt_to_grid(colt_flat, Nr_lydion, Nz_lydion):
    """
    Aaron suggested:
        f['a_rad_r_scat'][:].reshape((n_r, n_z))
    with n_z = 2*n_r, because he duplicated the domain instead of
    using reflective BCs.

    For comparison to LYDION half-plane, the expected shape is:
        COLT flat size = Nr_lydion * (2*Nz_lydion)

    This routine also handles the common case Nz_lydion == Nr_lydion.
    """
    arr = np.asarray(colt_flat, dtype=float)

    if arr.ndim == 2:
        return arr

    if arr.ndim != 1:
        raise RuntimeError(f"Expected 1D or 2D COLT field. Got shape {arr.shape}")

    N = arr.size

    expected = Nr_lydion * (2 * Nz_lydion)
    if N == expected:
        return arr.reshape((Nr_lydion, 2 * Nz_lydion))

    # Fallback to Aaron's n_z = 2*n_r assumption.
    n_r_float = np.sqrt(N / 2.0)
    n_r = int(round(n_r_float))
    if 2 * n_r * n_r == N:
        n_z = 2 * n_r
        print(
            f"WARNING: COLT size did not match LYDION Nr*(2*Nz) = {expected}, "
            f"but matches Aaron's n_z=2*n_r with n_r={n_r}, n_z={n_z}."
        )
        return arr.reshape((n_r, n_z))

    raise RuntimeError(
        f"Could not reshape COLT array of length {N}. "
        f"Expected {expected} = Nr_lydion*(2*Nz_lydion), "
        f"or 2*n_r^2 for integer n_r."
    )


def select_colt_half(colt_full, lydion_e, mode):
    """
    Select the half of the full-Z COLT domain corresponding to the LYDION half-plane.

    Assumes colt_full has shape (Nr, 2*Nz).
    """
    Nr, Nz2 = colt_full.shape
    NrL, NzL = lydion_e.shape

    if mode == "full":
        return colt_full, "full"

    if Nr != NrL:
        raise RuntimeError(
            f"COLT Nr={Nr} does not match LYDION Nr={NrL}. "
            "Check whether this is the corresponding run/grid."
        )

    if Nz2 == NzL:
        print("WARNING: COLT Z size already matches LYDION Nz. No half-selection applied.")
        return colt_full, "already_half"

    if Nz2 != 2 * NzL:
        raise RuntimeError(
            f"COLT Nz={Nz2} is not equal to LYDION Nz={NzL} or 2*Nz={2*NzL}."
        )

    lower = colt_full[:, :NzL]
    upper = colt_full[:, NzL:]

    if mode == "lower":
        return lower, "lower"

    if mode == "upper":
        return upper, "upper"

    if mode == "auto":
        # Compare total energy scale, without assuming orientation.
        sum_L = np.nansum(np.clip(lydion_e, 0.0, np.inf))
        sum_lower = np.nansum(np.clip(lower, 0.0, np.inf))
        sum_upper = np.nansum(np.clip(upper, 0.0, np.inf))

        err_lower = abs(np.log10((sum_lower + 1e-300) / (sum_L + 1e-300)))
        err_upper = abs(np.log10((sum_upper + 1e-300) / (sum_L + 1e-300)))

        if err_upper <= err_lower:
            return upper, "upper(auto)"
        return lower, "lower(auto)"

    raise ValueError("colt_half_to_plot must be 'upper', 'lower', 'auto', or 'full'.")


def apply_colt_normalization(colt_e, lydion_e, mode, manual_factor):
    colt_e = np.asarray(colt_e, dtype=float).copy()
    lydion_e = np.asarray(lydion_e, dtype=float)

    scale = 1.0

    if mode == "none":
        scale = 1.0

    elif mode == "match_total":
        Lsum = np.nansum(np.clip(lydion_e, 0.0, np.inf))
        Csum = np.nansum(np.clip(colt_e, 0.0, np.inf))
        if Csum <= 0.0 or not np.isfinite(Csum):
            raise RuntimeError("Cannot use match_total: COLT total is non-positive/non-finite.")
        scale = Lsum / Csum

    elif mode == "match_max":
        Lmax = np.nanmax(lydion_e)
        Cmax = np.nanmax(colt_e)
        if Cmax <= 0.0 or not np.isfinite(Cmax):
            raise RuntimeError("Cannot use match_max: COLT max is non-positive/non-finite.")
        scale = Lmax / Cmax

    else:
        raise ValueError("colt_normalization must be 'none', 'match_total', or 'match_max'.")

    scale *= float(manual_factor)
    return scale * colt_e, scale


def positive_minmax(*arrays):
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        good = np.isfinite(a) & (a > 0.0)
        if np.any(good):
            vals.append(a[good])

    if len(vals) == 0:
        return 1e-30, 1.0

    allv = np.concatenate(vals)
    return float(np.min(allv)), float(np.max(allv))


def make_norm(scale, arrays, vmin_user, vmax_user, log_vmin_fraction, log_vmax_factor):
    pos_min, pos_max = positive_minmax(*arrays)

    if scale == "log":
        if vmin_user is None:
            vmin = max(pos_max * log_vmin_fraction, pos_min)
        else:
            vmin = float(vmin_user)

        if vmax_user is None:
            vmax = log_vmax_factor * pos_max
        else:
            vmax = float(vmax_user)

        if vmin <= 0.0:
            vmin = pos_min

        return LogNorm(vmin=vmin, vmax=vmax), vmin, vmax

    if scale == "linear":
        vmin = 0.0 if vmin_user is None else float(vmin_user)
        vmax = pos_max if vmax_user is None else float(vmax_user)
        return None, vmin, vmax

    raise ValueError("scale must be 'log' or 'linear'.")


def make_colt_Z_edges_from_lydion(Z_half_edges):
    """
    Build approximate full-domain Z edges from LYDION half-plane edges.

    This is only used if colt_half_to_plot='full'. For the default half-plane
    comparison we simply use the LYDION Z edges directly.
    """
    Z = np.asarray(Z_half_edges, dtype=float)

    if np.all(Z >= -1e-14):
        # LYDION likely stores [0, Zmax]. Mirror to [-Zmax, Zmax].
        Z_pos = Z
        Z_neg = -Z_pos[::-1]
        Z_full = np.concatenate([Z_neg[:-1], Z_pos])
        return Z_full

    # If LYDION already has negative and positive Z, just return it.
    return Z


def read_colt_for_plot(fname, lydion_e, Nr, Nz):
    colt_path, colt_raw = read_colt_field(fname, COLT_eLya_dataset)
    colt_full = reshape_colt_to_grid(colt_raw, Nr, Nz)

    colt_plot, selected_half = select_colt_half(colt_full, lydion_e, colt_half_to_plot)
    colt_plot, total_colt_scale = apply_colt_normalization(
        colt_plot,
        lydion_e if colt_plot.shape == lydion_e.shape else np.nan * colt_plot,
        colt_normalization,
        colt_scale_factor,
    )

    if colt_half_to_plot != "full" and colt_plot.shape != lydion_e.shape:
        raise RuntimeError(
            f"After half-selection, COLT shape {colt_plot.shape} does not match "
            f"LYDION shape {lydion_e.shape}."
        )

    return {
        "path": colt_path,
        "raw": colt_raw,
        "full": colt_full,
        "plot": colt_plot,
        "selected_half": selected_half,
        "scale": total_colt_scale,
    }


def clipped_for_plot(arr, scale, vmin, vmax):
    if scale == "log":
        return np.clip(arr, vmin, vmax)
    return np.clip(arr, -np.inf, vmax)


###############################################################
#######                  R E A D  D A T A               #######
###############################################################

lyd = read_lydion_file(LYDION_FileName, snapshot_to_plot)

Nr = len(lyd["r"]) - 1
Nz = len(lyd["Z"]) - 1

colts = {
    xcrit: read_colt_for_plot(fname, lyd["e_Lya"], Nr, Nz)
    for xcrit, fname in COLT_FileNames.items()
}

print("\n========== Loaded comparison ==========")
print(f"LYDION file:        {lyd['path']}")
print(f"LYDION e_Lya shape: {lyd['e_Lya'].shape}")
print(f"LYDION nHI shape:   {lyd['nHI'].shape}")
for xcrit in sorted(colts):
    colt = colts[xcrit]
    print(f"COLT xcrit{xcrit} file:          {colt['path']}")
    print(f"COLT xcrit{xcrit} raw shape:     {colt['raw'].shape}")
    print(f"COLT xcrit{xcrit} full shape:    {colt['full'].shape}")
    print(f"COLT xcrit{xcrit} plotted shape: {colt['plot'].shape}")
    print(f"COLT xcrit{xcrit} half selected: {colt['selected_half']}")
    print(f"COLT xcrit{xcrit} scale:         {colt['scale']:.6e}")
print(f"COLT normalization: {colt_normalization}")
print(f"LYDION snapshot:    k={lyd['snapshot_index']}, t={lyd['t'][lyd['snapshot_index']]:.6e}")
print("=======================================\n")


###############################################################
#######                P L O T  S E T U P               #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 30})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

plt.rcParams.update({'legend.fontsize': 24})
plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 2.5
plt.rcParams['xtick.minor.width'] = 2.5
plt.rcParams['xtick.major.size'] = 18
plt.rcParams['xtick.minor.size'] = 7
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['ytick.minor.width'] = 2.5
plt.rcParams['ytick.major.size'] = 18
plt.rcParams['ytick.minor.size'] = 7
plt.rcParams['xtick.major.pad'] = '12'
plt.rcParams['ytick.major.pad'] = '12'

plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.linewidth'] = 3.0
plt.rcParams['grid.linewidth'] = 3.0
plt.rcParams['grid.linestyle'] = '--'

plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


###############################################################
#######              C O L O R  R A N G E              #######
###############################################################

if colt_half_to_plot == "full":
    Z_plot_edges = make_colt_Z_edges_from_lydion(lyd["Z"])
    r_plot_edges = lyd["r"]

    for xcrit, colt in sorted(colts.items()):
        if colt["plot"].shape != (len(r_plot_edges) - 1, len(Z_plot_edges) - 1):
            raise RuntimeError(
                f"Full COLT xcrit{xcrit} plot shape {colt['plot'].shape} does not match "
                f"constructed grid {(len(r_plot_edges)-1, len(Z_plot_edges)-1)}."
            )

    # LYDION panel remains half-plane; COLT panels get full mirrored Z.
    R_L, Z_L = np.meshgrid(lyd["r"], lyd["Z"], indexing="ij")
    R_C, Z_C = np.meshgrid(r_plot_edges, Z_plot_edges, indexing="ij")

else:
    R_L, Z_L = np.meshgrid(lyd["r"], lyd["Z"], indexing="ij")
    R_C, Z_C = R_L, Z_L


e_norm, e_vmin, e_vmax = make_norm(
    eLya_scale,
    (lyd["e_Lya"], *(colt["plot"] for colt in colts.values())),
    eLya_vmin_user,
    eLya_vmax_user,
    eLya_log_vmin_fraction,
    eLya_log_vmax_factor,
)

nHI_norm, nHI_vmin, nHI_vmax = make_norm(
    nHI_scale,
    (lyd["nHI"],),
    nHI_vmin_user,
    nHI_vmax_user,
    nHI_log_vmin_fraction,
    nHI_log_vmax_factor,
)


###############################################################
#######                    P L O T                     #######
###############################################################

fig = plt.figure(figsize=figsize, constrained_layout=False)

# ------------------------------------------------------------------
# Manual axes placement
# ------------------------------------------------------------------
# GridSpec/subplots with aspect="equal" can leave large uneven-looking
# gaps because matplotlib adjusts each axes box to preserve square pixels.
# Here we place the six map axes explicitly.  The width is computed from
# panel_height and figsize so that each map panel is square in physical units.
fig_width, fig_height = figsize
panel_width = panel_height * fig_height / fig_width

x0 = panel_left
x1 = x0 + panel_width + panel_col_gap
x2 = x1 + panel_width + panel_col_gap

y0 = panel_bottom
y1 = y0 + panel_height + panel_row_gap

cbar_x = x2 + panel_width + colorbar_gap

axL  = fig.add_axes([x0, y1, panel_width, panel_height])
axC0 = fig.add_axes([x1, y1, panel_width, panel_height], sharex=axL, sharey=axL)
axC1 = fig.add_axes([x2, y1, panel_width, panel_height], sharex=axL, sharey=axL)
axC2 = fig.add_axes([x0, y0, panel_width, panel_height], sharex=axL, sharey=axL)
axC3 = fig.add_axes([x1, y0, panel_width, panel_height], sharex=axL, sharey=axL)
axN  = fig.add_axes([x2, y0, panel_width, panel_height], sharex=axL, sharey=axL)

# Dedicated colorbar axes with the same vertical extent as the corresponding row.
cax_e = fig.add_axes([cbar_x, y1, colorbar_width, panel_height])
cax_n = fig.add_axes([cbar_x, y0, colorbar_width, panel_height])

axes = np.array([[axL, axC0, axC1], [axC2, axC3, axN]], dtype=object)

# pcolormesh expects C shape (Ny, Nx), hence .T for LYDION arrays stored as (Nr, Nz).
imL = axL.pcolormesh(
    R_L, Z_L,
    clipped_for_plot(lyd["e_Lya"].T, eLya_scale, e_vmin, e_vmax),
    shading="auto",
    cmap=eLya_cmap,
    norm=e_norm,
    rasterized=True,
)

# COLT/MCRT is already oriented oppositely relative to LYDION after the
# reshape + half-selection. Keep LYDION exactly as before, but do NOT
# transpose the COLT field here. This is the orientation correction.
colt_axes = {
    0: axC0,
    1: axC1,
    2: axC2,
    3: axC3,
}

colt_images = {}
for xcrit, ax in colt_axes.items():
    colt_images[xcrit] = ax.pcolormesh(
        R_C, Z_C,
        clipped_for_plot(colts[xcrit]["plot"], eLya_scale, e_vmin, e_vmax),
        shading="auto",
        cmap=eLya_cmap,
        norm=e_norm,
        rasterized=True,
    )

# nHI has already been reordered to the LYDION/FV convention (Nr, Nz)
# in read_lydion_file(). With R_L, Z_L built using indexing="ij",
# pcolormesh should receive this array directly. Do NOT apply the extra
# transpose used for e_Lya, since the Lyα history array and the static
# nHI array are read differently by h5py/Julia-HDF5 in this file.
imN = axN.pcolormesh(
    R_L, Z_L,
    clipped_for_plot(lyd["nHI"], nHI_scale, nHI_vmin, nHI_vmax),
    shading="auto",
    cmap=nHI_cmap,
    norm=nHI_norm,
    rasterized=True,
)

for ax in axes.flat:
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(FuncFormatter(compact_tick_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(compact_tick_formatter))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

# Shared labels: y-labels only on the left column, x-labels only on the bottom row.
axL.set_ylabel(r"$Z/R_{cl}$", labelpad=16)
axC2.set_ylabel(r"$Z/R_{cl}$", labelpad=16)
axC2.set_xlabel(r"$R/R_{cl}$", labelpad=16)
axC3.set_xlabel(r"$R/R_{cl}$", labelpad=16)
axN.set_xlabel(r"$R/R_{cl}$", labelpad=16)

# Remove duplicate y tick labels away from the left column.
for ax in (axC0, axC1, axC3, axN):
    ax.tick_params(labelleft=False)

# Remove duplicate x tick labels on the top row.
for ax in (axL, axC0, axC1):
    ax.tick_params(labelbottom=False)

# Match visible limits for half-plane comparison.
for ax in (axL, axN):
    ax.set_xlim(lyd["r"][0], lyd["r"][-1])
    ax.set_ylim(lyd["Z"][0], lyd["Z"][-1])

if colt_half_to_plot != "full":
    for ax in colt_axes.values():
        ax.set_xlim(lyd["r"][0], lyd["r"][-1])
        ax.set_ylim(lyd["Z"][0], lyd["Z"][-1])
else:
    for ax in colt_axes.values():
        ax.set_xlim(R_C.min(), R_C.max())
        ax.set_ylim(Z_C.min(), Z_C.max())

# In-panel labels only. No panel titles above the axes.
panel_style = dict(
    ha="left",
    va="top",
    color="white",
    fontsize=30,
    path_effects=[pe.withStroke(linewidth=4, foreground="black")]
)

axL.text(0.65, 0.12, r"LYDION", transform=axL.transAxes, **panel_style)
axC0.text(0.32, 0.20, "Monte Carlo RT\nNo core-skipping", transform=axC0.transAxes, **panel_style)
axC1.text(0.45, 0.20, "Monte Carlo RT" + "\n" + r"$x_{crit}=1$", transform=axC1.transAxes, **panel_style)
axC2.text(0.45, 0.20, "Monte Carlo RT" + "\n" + r"$x_{crit}=2$", transform=axC2.transAxes, **panel_style)
axC3.text(0.45, 0.20, "Monte Carlo RT" + "\n" + r"$x_{crit}=3$", transform=axC3.transAxes, **panel_style)
axN.text(0.60, 0.94, r"H I density", transform=axN.transAxes, **panel_style)
axN.text(0.06, 0.20, "Emission\nregion", **panel_style)
axN.text(0.60, 0.60, "Thick H I shell", **panel_style, rotation=-60)
axN.text(0.06, 0.90, "Lower-opacity\ncone", **panel_style, rotation=-20)

# Time label in LYDION panel.
time_label = rf"$t = {lyd['t'][lyd['snapshot_index']]:.3g}$"
#axL.text(
#    0.06, 0.84,
#    time_label,
#    transform=axL.transAxes,
#    ha="left",
#    va="top",
#    color="white",
#    fontsize=24,
#    path_effects=[pe.withStroke(linewidth=4, foreground="black")]
#)

# Small notes in COLT panels if normalization is applied.
if colt_normalization != "none" or colt_scale_factor != 1.0:
    for xcrit, ax in colt_axes.items():
        ax.text(
            0.06, 0.84,
            rf"scale $= {colts[xcrit]['scale']:.2e}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=22,
            path_effects=[pe.withStroke(linewidth=4, foreground="black")]
        )

# Separate colorbars in the dedicated GridSpec column.
# Because they are not appended to map axes, they do not distort the
# apparent spacing of the physical panels.
cbar_e = fig.colorbar(colt_images[1], cax=cax_e)
cbar_e.set_label(r"$e_{Ly\alpha}$ (erg cm$^{-3}$)", labelpad=15)

if eLya_scale == "log":
    cbar_e.ax.yaxis.set_major_locator(LogLocator(base=10))
    cbar_e.ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    cbar_e.ax.yaxis.set_minor_formatter(NullFormatter())

cbar_n = fig.colorbar(imN, cax=cax_n)
cbar_n.set_label(r"$n_{HI}$ (cm$^{-3}$)", labelpad=15)

if nHI_scale == "log":
    cbar_n.ax.yaxis.set_major_locator(LogLocator(base=10))
    cbar_n.ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    cbar_n.ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter_small_decimal))
    cbar_n.ax.yaxis.set_minor_formatter(NullFormatter())

if savefig:
    outpath = resolve_path(outname)
    fig.savefig(outpath, dpi=dpi_save, bbox_inches="tight")
    print(f"Saved figure: {outpath}")
else:
    plt.show()
