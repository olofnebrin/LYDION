import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
# The plotting code below uses the same order as the energy-density plot:
#   top row:    LYDION, xcrit0/no core-skipping, xcrit1
#   bottom row: xcrit2, xcrit3, nHI
COLT_FileNames = {
    0: "Lya-cloud-xcrit0_1e7photons.hdf5",
    1: "Lya-cloud-xcrit1_1e7photons.hdf5",
    2: "Lya-cloud-xcrit2_1e7photons.hdf5",
    3: "Lya-cloud-xcrit3_1e7photons.hdf5",
}

outname = "ConeTest_fLya.pdf"
savefig = True

# Which LYDION snapshot to plot:
#   "final" -> final saved snapshot with t > 0
#   integer -> explicit snapshot index
snapshot_to_plot = "final"

# LYDION force datasets.
LYDION_fr_dataset = "f_Lya_r_History"
LYDION_fz_dataset = "f_Lya_Z_History"

# COLT/MCRT acceleration datasets.
# The *_scat versions usually converge faster.
COLT_ar_dataset = "a_rad_r_scat"
COLT_az_dataset = "a_rad_z_scat"

#COLT_ar_dataset = "a_rad_r"
#COLT_az_dataset = "a_rad_z"

# H I number-density dataset in the LYDION static-cloud HDF5 file.
nHI_dataset = "nHI"

# Aaron's output has the full Z-domain because he did not use reflective BCs.
# LYDION likely has only the reflective half-plane.
#
# Options:
#   "upper" -> use second half of COLT Z array, assuming COLT full domain is ordered [-Zmax, ..., +Zmax]
#   "lower" -> use first half
#   "auto"  -> choose whichever half has closer total force magnitude to LYDION
colt_half_to_plot = "upper"

# For plotting MCRT force maps, fold/average the full COLT domain onto the
# upper half before computing |f|. This uses the expected midplane symmetry
# to reduce Monte Carlo noise. If the file is already half-domain, this falls
# back to the selected half above.
plot_mcrt_folded_symmetry = True

# Which MCRT force multipliers to show in the panel text.
#   "folded" -> use folded values when available, otherwise full-domain values
#   "full"   -> use full-domain/raw integral values
MCRT_MF_label_source = "folded"

# Domain used for the MCRT force-multiplier integrals.
#
# Aaron's COLT/MCRT output likely contains the full cloud, i.e. both lower
# and upper halves in Z. In that case, use "full". The half-selection above
# is then used only for plotting.
#
# Options:
#   "full"      -> integrate the full COLT domain if the file contains 2*Nz cells in Z
#   "plot_half" -> integrate only the same half that is shown in the figure
MCRT_integral_domain = "full"

# COLT/MCRT arrays are compared/integrated against the LYDION cylindrical
# cell volumes, which have shape (Nr, Nz) for one half or (Nr, 2*Nz) for
# the full domain. If Aaron's output is a flat array, this controls how the
# flat array is reshaped onto the LYDION grid.
#
# Options:
#   "r_z" -> current/default assumption: arr.reshape((Nr, Nz_or_2Nz))
#   "z_r" -> alternative flattening: arr.reshape((Nz_or_2Nz, Nr)).T
#
# Use the diagnostic prints below to check whether this matters. For 2D COLT
# arrays, the script uses the array as stored, but still checks the shape.
COLT_flattening_order = "r_z"

# For LYDION force-field integrals, h5py exposes the saved Julia arrays in
# the opposite orientation compared to CellVolumes for these square grids.
# Keeping this True makes the direct Python integral reproduce M_F_History.
# This affects only LYDION direct diagnostic integrals, not the saved M_F.
LYDION_transpose_force_fields_for_integrals = True

# LYDION simulates only the upper half-plane with midplane symmetry.
# For direct re-integration of LYDION forces from saved f_Lya and CellVolumes,
# multiply by 2 to recover the whole cloud. The saved M_F_History is left
# untouched and printed separately as the reference LYDION value.
LYDION_direct_integral_domain = "full_symmetric"  # "full_symmetric" or "saved_half"

# Physical constants.
mH_cgs   = 1.6735575e-24   # g
c_cgs    = 2.99792458e10  # cm s^-1
Lsun_cgs = 3.839e33      # erg s^-1

# Luminosity used for the COLT/MCRT force multiplier.
L_Lya_MCRT = 1e5 * Lsun_cgs

# LYDION saves physical cell volumes in cgs units.
# Use these directly for both LYDION and MCRT force-multiplier integrals.
CellVolumes_dataset = "CellVolumes"

# Plot scale.
force_scale = "log"      # "log" or "linear"
nHI_scale   = "log"      # "log" or "linear"

# Shared force colorbar range for LYDION, xcrit0, and xcrit3.
# Set to None for automatic.
force_vmin_user = 1e-29
force_vmax_user = None

# nHI colorbar range.
# Set to None for automatic.
nHI_vmin_user = 0.01
nHI_vmax_user = 2e3

# For log plots, automatic vmin is max_positive * log_vmin_fraction.
force_log_vmin_fraction = 1e-8
force_log_vmax_factor   = 1.0

nHI_log_vmin_fraction = 1e-8
nHI_log_vmax_factor   = 10.0

# Colormaps.
force_cmap = "cubehelix"
nHI_cmap   = "cma:laguna"

# Figure size and layout.
figsize = (27, 15.0)
dpi_save = 400

# Manual panel layout in figure-fraction units.
# This matches the energy-density comparison script and keeps the H I panel
# identical in placement/labels.
panel_left      = 0.075
panel_bottom    = 0.100
panel_height    = 0.390
panel_col_gap   = -0.006
panel_row_gap   = 0.02
colorbar_gap    = 0.018
colorbar_width  = 0.016

# Text label positions in axes-fraction coordinates.
panel_label_positions = {
    "LYDION": (0.55, 0.12),
    0:        (0.40, 0.18),
    1:        (0.45, 0.18),
    2:        (0.45, 0.18),
    3:        (0.45, 0.18),
    "nHI":    (0.60, 0.94),
}

# Force-multiplier label positions in axes-fraction coordinates.
MF_label_positions = {
    "LYDION": (0.9, 0.9),
    0:        (0.9, 0.9),
    1:        (0.9, 0.9),
    2:        (0.9, 0.9),
    3:        (0.9, 0.9),
}

show_MF_labels = True

# Optional quiver arrows showing force direction.
plot_arrows = False
arrow_n_r = 10                # target number of arrow columns
arrow_n_z = 10                # target number of arrow rows
arrow_min_fraction = 0.01     # suppress arrows with |f| < this * global max |f|
arrow_reference_length = 0.12 # in axis data units (R_cl units). Arrow for global max has this length.
arrow_color = "white"
arrow_width = 0.006
arrow_headwidth = 4.5
arrow_headlength = 6.0*100
arrow_headaxislength = 5.0


###############################################################
#######                H E L P E R S                    #######
###############################################################

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

        if CellVolumes_dataset not in f:
            raise RuntimeError(f"Dataset {CellVolumes_dataset!r} not found in {path}")
        dV_raw = f[CellVolumes_dataset][:]

        fr_raw = f[LYDION_fr_dataset][:]
        fz_raw = f[LYDION_fz_dataset][:]

        if nHI_dataset not in f:
            raise RuntimeError(f"Dataset {nHI_dataset!r} not found in {path}")
        nHI_raw = f[nHI_dataset][:]

        M_F_hist = f["M_F_History"][:] if "M_F_History" in f else None
        M_F_radial_net_hist = f["M_F_radial_net_History"][:] if "M_F_radial_net_History" in f else None
        M_F_radial_out_hist = f["M_F_radial_out_History"][:] if "M_F_radial_out_History" in f else None
        L_Lya_hist = f["L_Lya_History"][:] if "L_Lya_History" in f else None

    fr_hist = reorder_history_3d(fr_raw, t, r, Z)
    fz_hist = reorder_history_3d(fz_raw, t, r, Z)

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

    fr = np.asarray(fr_hist[k, :, :], dtype=float)
    fz = np.asarray(fz_hist[k, :, :], dtype=float)
    nHI = np.asarray(nHI_raw, dtype=float)
    dV = np.asarray(dV_raw, dtype=float)

    Nr = len(r) - 1
    Nz = len(Z) - 1
    if nHI.shape == (Nz, Nr):
        nHI = nHI.T
    elif nHI.shape != (Nr, Nz):
        raise RuntimeError(
            f"nHI shape {nHI.shape} does not match expected (Nr,Nz)=({Nr},{Nz}) "
            f"or transposed (Nz,Nr)=({Nz},{Nr})."
        )

    if dV.shape == (Nz, Nr):
        dV = dV.T
    elif dV.shape != (Nr, Nz):
        raise RuntimeError(
            f"CellVolumes shape {dV.shape} does not match expected (Nr,Nz)=({Nr},{Nz}) "
            f"or transposed (Nz,Nr)=({Nz},{Nr})."
        )

    M_F = None
    if M_F_hist is not None and k < len(M_F_hist):
        M_F = float(M_F_hist[k])

    M_F_radial_net = None
    if M_F_radial_net_hist is not None and k < len(M_F_radial_net_hist):
        M_F_radial_net = float(M_F_radial_net_hist[k])

    M_F_radial_out = None
    if M_F_radial_out_hist is not None and k < len(M_F_radial_out_hist):
        M_F_radial_out = float(M_F_radial_out_hist[k])

    L_Lya = None
    if L_Lya_hist is not None and k < len(L_Lya_hist):
        L_Lya = float(L_Lya_hist[k])

    return {
        "path": path,
        "t": t,
        "snapshot_index": k,
        "r": r,
        "Z": Z,
        "rCoM": rCoM,
        "ZCoM": ZCoM,
        "f_r": fr,
        "f_z": fz,
        "nHI": nHI,
        "dV": dV,
        "M_F": M_F,
        "M_F_radial_net": M_F_radial_net,
        "M_F_radial_out": M_F_radial_out,
        "L_Lya": L_Lya,
    }


def read_colt_field(fname, dataset_name):
    path = resolve_path(fname)
    with h5py.File(path, "r") as f:
        if dataset_name not in f:
            raise RuntimeError(f"Dataset {dataset_name!r} not found in {path}")
        arr = f[dataset_name][:]
    return path, np.asarray(arr, dtype=float)


def reshape_colt_to_grid(colt_flat, Nr_lydion, Nz_lydion, flattening_order=COLT_flattening_order):
    """
    Reshape a COLT/MCRT field onto the LYDION cylindrical grid convention.

    Returned shape should be either (Nr, Nz) for a half-domain file or
    (Nr, 2*Nz) for a full-domain file, so that it can be multiplied directly
    by LYDION CellVolumes with the same orientation.
    """
    arr = np.asarray(colt_flat, dtype=float)

    if arr.ndim == 2:
        if arr.shape in [(Nr_lydion, Nz_lydion), (Nr_lydion, 2 * Nz_lydion)]:
            return arr
        if arr.T.shape in [(Nr_lydion, Nz_lydion), (Nr_lydion, 2 * Nz_lydion)]:
            raise RuntimeError(
                f"COLT field has shape {arr.shape}; its transpose would match the LYDION grid. "
                f"This is ambiguous, so transpose the dataset explicitly or update the reader."
            )
        raise RuntimeError(
            f"2D COLT field has shape {arr.shape}, but expected (Nr,Nz)=({Nr_lydion},{Nz_lydion}) "
            f"or full-domain (Nr,2*Nz)=({Nr_lydion},{2*Nz_lydion})."
        )

    if arr.ndim != 1:
        raise RuntimeError(f"Expected 1D or 2D COLT field. Got shape {arr.shape}")

    N = arr.size
    expected_full = Nr_lydion * (2 * Nz_lydion)
    expected_half = Nr_lydion * Nz_lydion

    if N == expected_full:
        if flattening_order == "r_z":
            return arr.reshape((Nr_lydion, 2 * Nz_lydion))
        if flattening_order == "z_r":
            return arr.reshape((2 * Nz_lydion, Nr_lydion)).T
        raise ValueError("COLT_flattening_order must be 'r_z' or 'z_r'.")

    if N == expected_half:
        if flattening_order == "r_z":
            return arr.reshape((Nr_lydion, Nz_lydion))
        if flattening_order == "z_r":
            return arr.reshape((Nz_lydion, Nr_lydion)).T
        raise ValueError("COLT_flattening_order must be 'r_z' or 'z_r'.")

    n_r_float = np.sqrt(N / 2.0)
    n_r = int(round(n_r_float))
    if 2 * n_r * n_r == N:
        n_z = 2 * n_r
        print(
            f"WARNING: COLT size did not match LYDION Nr*(2*Nz) = {expected_full}, "
            f"but matches Aaron's n_z=2*n_r with n_r={n_r}, n_z={n_z}."
        )
        if flattening_order == "r_z":
            return arr.reshape((n_r, n_z))
        if flattening_order == "z_r":
            return arr.reshape((n_z, n_r)).T
        raise ValueError("COLT_flattening_order must be 'r_z' or 'z_r'.")

    raise RuntimeError(
        f"Could not reshape COLT array of length {N}. Expected {expected_full} = "
        f"Nr_lydion*(2*Nz_lydion), {expected_half} = Nr_lydion*Nz_lydion, "
        f"or 2*n_r^2 for integer n_r."
    )


def select_colt_half(colt_full, lydion_ref, mode):
    Nr, Nz2 = colt_full.shape
    NrL, NzL = lydion_ref.shape

    if Nr != NrL:
        raise RuntimeError(
            f"COLT Nr={Nr} does not match LYDION Nr={NrL}. Check whether this is the corresponding run/grid."
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
        # Compare total force magnitude to LYDION.
        ref_sum = np.nansum(np.abs(lydion_ref))
        lower_sum = np.nansum(np.abs(lower))
        upper_sum = np.nansum(np.abs(upper))

        err_lower = abs(np.log10((lower_sum + 1e-300) / (ref_sum + 1e-300)))
        err_upper = abs(np.log10((upper_sum + 1e-300) / (ref_sum + 1e-300)))

        if err_upper <= err_lower:
            return upper, "upper(auto)"
        return lower, "lower(auto)"

    raise ValueError("colt_half_to_plot must be 'upper', 'lower', or 'auto'.")


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


def clipped_for_plot(arr, scale, vmin, vmax):
    """
    Prepare a field for pcolormesh.

    For log plots, invalid and non-positive cells are masked. With the
    colormap bad color set to black below, this makes cells outside the
    physical cloud black rather than white/transparent. Positive values
    below vmin are left as under-range values, so set_under("black")
    also catches genuine under-range cells.
    """
    a = np.asarray(arr, dtype=float)

    if scale == "log":
        a = np.ma.masked_invalid(a)
        a = np.ma.masked_where(a <= 0.0, a)
        return np.ma.clip(a, None, vmax)

    a = np.ma.masked_invalid(a)
    return np.ma.clip(a, -np.inf, vmax)


def log_tick_formatter_small_decimal(x, pos=None):
    if x <= 0.0 or not np.isfinite(x):
        return ""

    exponent = np.log10(x)
    exponent_round = int(np.round(exponent))

    if not np.isclose(exponent, exponent_round, atol=1e-10, rtol=0.0):
        return ""

    if -2 <= exponent_round <= 2:
        return f"{10.0**exponent_round:g}"

    return rf"$10^{{{exponent_round}}}$"


def compact_tick_formatter(x, pos=None):
    """Compact linear tick labels: 0 and 1 instead of 0.0 and 1.0."""
    if not np.isfinite(x):
        return ""

    if abs(x) < 5e-13:
        x = 0.0

    return f"{x:g}"


def cell_volumes_axisymmetric(r_edges, Z_edges):
    """
    Axisymmetric cell volumes up to the physical length scale.

    If r and Z are in cm, this is the physical volume. If they are in
    units of R_cl, this is dimensionless and must be multiplied by R_cl^3.
    """
    r_edges = np.asarray(r_edges, dtype=float)
    Z_edges = np.asarray(Z_edges, dtype=float)

    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = np.diff(r_edges)
    dz = np.diff(Z_edges)

    return 2.0 * np.pi * r_mid[:, None] * dr[:, None] * dz[None, :]


def mirror_upper_half_scalar_to_full(q_half):
    """Return [lower, upper] full-domain scalar array from an upper-half array."""
    q = np.asarray(q_half, dtype=float)
    return np.concatenate((q[:, ::-1], q), axis=1)


def mirror_upper_half_vector_to_full(fr_half, fz_half):
    """
    Return [lower, upper] full-domain cylindrical force components from
    an upper-half, midplane-symmetric LYDION result.

    Under reflection Z -> -Z, the cylindrical radial component f_R is even,
    while f_Z is odd.
    """
    fr = np.asarray(fr_half, dtype=float)
    fz = np.asarray(fz_half, dtype=float)
    fr_full = np.concatenate((fr[:, ::-1], fr), axis=1)
    fz_full = np.concatenate((-fz[:, ::-1], fz), axis=1)
    return fr_full, fz_full


def fold_full_domain_vector_to_upper(fr_full, fz_full):
    """
    Fold a full [lower, upper] COLT/MCRT force field onto the upper half.

    This uses midplane symmetry to double the Monte Carlo statistics per
    upper-half cell before nonlinear diagnostics such as |f| are taken.

    Array convention: full-domain fields have shape (Nr, 2*Nz), ordered as
    [lower half, upper half] in Z. The lower half is reversed in Z and mirrored
    onto the upper half. Under Z -> -Z, f_R is even and f_Z is odd, so the
    mirrored lower-half f_Z changes sign.

    Returns arrays with shape (Nr, Nz), aligned with the LYDION upper-half
    CellVolumes.
    """
    fr = np.asarray(fr_full, dtype=float)
    fz = np.asarray(fz_full, dtype=float)

    if fr.shape != fz.shape:
        raise RuntimeError(f"fr_full shape {fr.shape} != fz_full shape {fz.shape}")

    Nr_full, Nz2 = fr.shape
    if Nz2 % 2 != 0:
        raise RuntimeError(f"Cannot fold full-domain field with odd Z size: shape={fr.shape}")

    Nz_half = Nz2 // 2
    fr_lower = fr[:, :Nz_half]
    fr_upper = fr[:, Nz_half:]
    fz_lower = fz[:, :Nz_half]
    fz_upper = fz[:, Nz_half:]

    fr_lower_mirrored = fr_lower[:, ::-1]
    fz_lower_mirrored = -fz_lower[:, ::-1]

    fr_folded = 0.5 * (fr_upper + fr_lower_mirrored)
    fz_folded = 0.5 * (fz_upper + fz_lower_mirrored)

    return fr_folded, fz_folded


def full_Z_centers_from_upper(ZCoM_upper):
    Zp = np.asarray(ZCoM_upper, dtype=float)
    return np.concatenate((-Zp[::-1], Zp))


def force_magnitude(fr, fz):
    return np.sqrt(np.asarray(fr, dtype=float)**2 + np.asarray(fz, dtype=float)**2)


def spherical_radial_force_density(fr, fz, r_centers, Z_centers):
    """
    Compute f . s_hat, where s_hat is the spherical radial unit vector
    from the cloud centre, expressed in cylindrical coordinates: 

        s_hat = (R e_R + Z e_Z) / sqrt(R^2 + Z^2).

    This R is the cylindrical radius, not the spherical radius.
    """
    R, Z = np.meshgrid(np.asarray(r_centers, dtype=float),
                       np.asarray(Z_centers, dtype=float),
                       indexing="ij")
    s = np.sqrt(R**2 + Z**2)

    rhat_R = np.zeros_like(s)
    rhat_Z = np.zeros_like(s)
    good = s > 0.0
    rhat_R[good] = R[good] / s[good]
    rhat_Z[good] = Z[good] / s[good]

    return np.asarray(fr, dtype=float) * rhat_R + np.asarray(fz, dtype=float) * rhat_Z


def force_multiplier_from_force_density(force_density, dV, L_Lya):
    integral_force = np.nansum(np.asarray(force_density, dtype=float) * dV)
    return integral_force / (L_Lya / c_cgs)


def force_multiplier_from_positive_force_density(force_density, dV, L_Lya):
    """Return integral dV max(force_density, 0) / (L_Lya/c)."""
    fd = np.asarray(force_density, dtype=float)
    integral_force = np.nansum(np.maximum(fd, 0.0) * dV)
    return integral_force / (L_Lya / c_cgs)


def force_multiplier_from_force_mag(force_mag, dV, L_Lya):
    return force_multiplier_from_force_density(force_mag, dV, L_Lya)


def format_MF_number(value):
    if value is None or not np.isfinite(value):
        return "?"
    if abs(value) >= 1e4 or (abs(value) > 0.0 and abs(value) < 1e-2):
        return f"{value:.2e}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def format_MF(value):
    return rf"$M_F = {format_MF_number(value)}$"


def format_MF_pair(M_F, M_F_radial):
    return (
        rf"$M_F = {format_MF_number(M_F)}$"
        "\n"
        rf"$M_{{F,radial}} = {format_MF_number(M_F_radial)}$"
    )


def build_quiver_indices(n, n_target):
    if n_target >= n:
        return np.arange(n, dtype=int)
    idx = np.linspace(0, n - 1, n_target)
    idx = np.unique(np.round(idx).astype(int))
    return idx


def add_quiver(ax, X, Y, U, V, global_max, min_fraction, n_r_target, n_z_target,
               color="white", width=0.006, headwidth=4.5, headlength=6.0,
               headaxislength=5.0, reference_length=0.12):
    if global_max <= 0.0 or not np.isfinite(global_max):
        return None

    nr, nz = U.shape
    i_idx = build_quiver_indices(nr, n_r_target)
    j_idx = build_quiver_indices(nz, n_z_target)

    Xs = X[np.ix_(i_idx, j_idx)]
    Ys = Y[np.ix_(i_idx, j_idx)]
    Us = U[np.ix_(i_idx, j_idx)]
    Vs = V[np.ix_(i_idx, j_idx)]
    mags = np.sqrt(Us**2 + Vs**2)

    mask = np.isfinite(Us) & np.isfinite(Vs) & np.isfinite(Xs) & np.isfinite(Ys)
    mask &= mags >= (min_fraction * global_max)

    if not np.any(mask):
        return None

    scale = global_max / reference_length

    q = ax.quiver(
        Xs[mask], Ys[mask], Us[mask], Vs[mask],
        angles="xy", scale_units="xy", scale=scale,
        pivot="mid", color=color, width=width,
        headwidth=headwidth, headlength=headlength,
        headaxislength=headaxislength,
        zorder=5,
    )
    return q


###############################################################
#######                  R E A D  D A T A               #######
###############################################################

lyd = read_lydion_file(LYDION_FileName, snapshot_to_plot)

Nr = len(lyd["r"]) - 1
Nz = len(lyd["Z"]) - 1

# Convert COLT accelerations to force per unit volume using rho = nHI * mH.
rho_ref_half = lyd["nHI"] * mH_cgs

# For full-domain COLT/MCRT integrals and symmetry folding, mirror the LYDION
# static density and cell volumes into the lower half. This assumes the static
# cloud setup is midplane symmetric.
rho_ref_full = mirror_upper_half_scalar_to_full(lyd["nHI"]) * mH_cgs
dV_half = lyd["dV"]
dV_full = mirror_upper_half_scalar_to_full(lyd["dV"])
ZCoM_full = full_Z_centers_from_upper(lyd["ZCoM"])


# ------------------------------------------------------------------
# Force multipliers
# ------------------------------------------------------------------
# Definitions:
#     M_F        = integral dV |f| / (L_Lya/c)
#     M_F_radial = integral dV (f . s_hat) / (L_Lya/c)
#
# where s_hat is the spherical radial unit vector from the cloud centre:
#     s_hat = (R e_R + Z e_Z) / sqrt(R^2 + Z^2).
# Do not confuse this spherical radial direction with the cylindrical R direction.

L_Lya_LYDION = lyd["L_Lya"] if lyd["L_Lya"] is not None and lyd["L_Lya"] > 0.0 else L_Lya_MCRT
M_F_LYDION_saved = lyd["M_F"]
M_F_radial_net_LYDION_saved = lyd["M_F_radial_net"]
M_F_radial_out_LYDION_saved = lyd["M_F_radial_out"]

# LYDION direct re-integration. Since LYDION stores only the upper half-plane,
# the correct full-cloud direct integral uses the mirror-symmetric lower half.
#
# Important: for these HDF5 files, the force-history arrays read by h5py need
# a transpose before they line up with CellVolumes[i,j]. This is diagnosed by
# requiring the direct full-symmetric integral to reproduce M_F_History.
lyd_fr_half_for_int = np.asarray(lyd["f_r"], dtype=float)
lyd_fz_half_for_int = np.asarray(lyd["f_z"], dtype=float)
if LYDION_transpose_force_fields_for_integrals:
    lyd_fr_half_for_int = lyd_fr_half_for_int.T
    lyd_fz_half_for_int = lyd_fz_half_for_int.T

if LYDION_direct_integral_domain == "full_symmetric":
    lyd_fr_int, lyd_fz_int = mirror_upper_half_vector_to_full(lyd_fr_half_for_int, lyd_fz_half_for_int)
    lyd_dV_int = dV_full
    lyd_ZCoM_int = ZCoM_full
    lyd_domain_label = "full cloud from mirrored upper half"
elif LYDION_direct_integral_domain == "saved_half":
    lyd_fr_int = lyd_fr_half_for_int
    lyd_fz_int = lyd_fz_half_for_int
    lyd_dV_int = dV_half
    lyd_ZCoM_int = lyd["ZCoM"]
    lyd_domain_label = "saved upper half only"
else:
    raise ValueError("LYDION_direct_integral_domain must be 'full_symmetric' or 'saved_half'.")

lyd_mag_int = force_magnitude(lyd_fr_int, lyd_fz_int)
lyd_frad_int = spherical_radial_force_density(
    lyd_fr_int, lyd_fz_int, lyd["rCoM"], lyd_ZCoM_int
)
M_F_LYDION_direct = force_multiplier_from_force_mag(lyd_mag_int, lyd_dV_int, L_Lya_LYDION)
M_F_radial_LYDION_direct = force_multiplier_from_force_density(lyd_frad_int, lyd_dV_int, L_Lya_LYDION)
M_F_radial_out_LYDION_direct = force_multiplier_from_positive_force_density(lyd_frad_int, lyd_dV_int, L_Lya_LYDION)

# Use the saved LYDION values for the label if available; otherwise use the direct values.
M_F_LYDION = M_F_LYDION_saved if M_F_LYDION_saved is not None else M_F_LYDION_direct
M_F_radial_LYDION = (
    M_F_radial_net_LYDION_saved
    if M_F_radial_net_LYDION_saved is not None
    else M_F_radial_LYDION_direct
)


def colt_integral_arrays(ar_full, az_full, ar_half, az_half):
    """Return MCRT force arrays for the requested integral domain."""
    if MCRT_integral_domain == "full" and ar_full.shape == dV_full.shape:
        fr_int = rho_ref_full * ar_full
        fz_int = rho_ref_full * az_full
        dV_int = dV_full
        ZCoM_int = ZCoM_full
        label = "full COLT domain"
    elif MCRT_integral_domain == "full" and ar_full.shape == dV_half.shape:
        print("WARNING: MCRT_integral_domain='full', but COLT arrays already look half-domain.")
        fr_int = rho_ref_half * ar_full
        fz_int = rho_ref_half * az_full
        dV_int = dV_half
        ZCoM_int = lyd["ZCoM"]
        label = "COLT array already half-domain"
    elif MCRT_integral_domain == "plot_half":
        fr_int = rho_ref_half * ar_half
        fz_int = rho_ref_half * az_half
        dV_int = dV_half
        ZCoM_int = lyd["ZCoM"]
        label = f"plotted {colt_half_to_plot} half only"
    else:
        raise ValueError("MCRT_integral_domain must be 'full' or 'plot_half'.")

    mag_int = force_magnitude(fr_int, fz_int)
    frad_int = spherical_radial_force_density(fr_int, fz_int, lyd["rCoM"], ZCoM_int)
    return fr_int, fz_int, mag_int, frad_int, dV_int, label


def colt_MF_for_flattening_order(ar_raw, az_raw, order):
    """Diagnostic M_F values for a possible COLT flat-array ordering."""
    ar_full_o = reshape_colt_to_grid(ar_raw, Nr, Nz, flattening_order=order)
    az_full_o = reshape_colt_to_grid(az_raw, Nr, Nz, flattening_order=order)

    if ar_full_o.shape == dV_full.shape:
        rho_o = rho_ref_full
        dV_o = dV_full
        Z_o = ZCoM_full
        label_o = "full"
    elif ar_full_o.shape == dV_half.shape:
        rho_o = rho_ref_half
        dV_o = dV_half
        Z_o = lyd["ZCoM"]
        label_o = "half"
    else:
        return None

    fr_o = rho_o * ar_full_o
    fz_o = rho_o * az_full_o
    mag_o = force_magnitude(fr_o, fz_o)
    frad_o = spherical_radial_force_density(fr_o, fz_o, lyd["rCoM"], Z_o)
    return (
        label_o,
        force_multiplier_from_force_mag(mag_o, dV_o, L_Lya_MCRT),
        force_multiplier_from_force_density(frad_o, dV_o, L_Lya_MCRT),
    )


def folded_colt_force_multipliers(fr_full, fz_full):
    """
    Return folded-MCRT estimates of M_F and M_F_radial.

    The full-domain force field is first folded/averaged across the midplane,
    then |f| and f.s_hat are computed on the upper half. The upper-half
    integral is multiplied by 2 to represent the whole cloud. This should have
    the same expectation value as the full-domain integral, but lower MC noise,
    especially for the nonlinear |f| diagnostic.
    """
    if fr_full.shape != dV_full.shape:
        return None

    fr_fold, fz_fold = fold_full_domain_vector_to_upper(fr_full, fz_full)
    mag_fold = force_magnitude(fr_fold, fz_fold)
    frad_fold = spherical_radial_force_density(
        fr_fold, fz_fold, lyd["rCoM"], lyd["ZCoM"]
    )

    M_F_fold = 2.0 * force_multiplier_from_force_mag(
        mag_fold, dV_half, L_Lya_MCRT
    )
    M_F_radial_fold = 2.0 * force_multiplier_from_force_density(
        frad_fold, dV_half, L_Lya_MCRT
    )
    M_F_radial_out_fold = 2.0 * force_multiplier_from_positive_force_density(
        frad_fold, dV_half, L_Lya_MCRT
    )

    return M_F_fold, M_F_radial_fold, M_F_radial_out_fold


def read_colt_force_for_plot(xcrit, fname):
    colt_path_r, ar_raw = read_colt_field(fname, COLT_ar_dataset)
    _,           az_raw = read_colt_field(fname, COLT_az_dataset)

    ar_full = reshape_colt_to_grid(ar_raw, Nr, Nz)
    az_full = reshape_colt_to_grid(az_raw, Nr, Nz)

    ar_half, selected_half = select_colt_half(ar_full, lyd["f_r"], colt_half_to_plot)
    az_half, _             = select_colt_half(az_full, lyd["f_z"], colt_half_to_plot)

    fr_half = rho_ref_half * ar_half
    fz_half = rho_ref_half * az_half

    # Default plotting: fold lower+upper halves before computing |f|. This uses
    # both halves of Aaron's full-domain output and reduces MC noise in the map.
    plot_mode = selected_half
    if plot_mcrt_folded_symmetry and ar_full.shape == dV_full.shape:
        fr_full_for_plot = rho_ref_full * ar_full
        fz_full_for_plot = rho_ref_full * az_full
        fr_plot, fz_plot = fold_full_domain_vector_to_upper(fr_full_for_plot, fz_full_for_plot)
        plot_mode = "folded-symmetry"
    else:
        fr_plot, fz_plot = fr_half, fz_half

    mag_plot = force_magnitude(fr_plot, fz_plot)

    fr_int, fz_int, mag_int, frad_int, dV_int, int_label = colt_integral_arrays(
        ar_full, az_full, ar_half, az_half
    )

    M_F_full = force_multiplier_from_force_mag(mag_int, dV_int, L_Lya_MCRT)
    M_F_radial_full = force_multiplier_from_force_density(frad_int, dV_int, L_Lya_MCRT)
    M_F_radial_out_full = force_multiplier_from_positive_force_density(frad_int, dV_int, L_Lya_MCRT)

    folded = folded_colt_force_multipliers(fr_int, fz_int)
    if folded is not None and MCRT_MF_label_source == "folded":
        M_F_label, M_F_radial_label, M_F_radial_out_label = folded
        label_source = "folded-symmetry"
    else:
        M_F_label = M_F_full
        M_F_radial_label = M_F_radial_full
        M_F_radial_out_label = M_F_radial_out_full
        label_source = int_label

    frad_plot = spherical_radial_force_density(fr_plot, fz_plot, lyd["rCoM"], lyd["ZCoM"])
    M_F_plot_half = force_multiplier_from_force_mag(mag_plot, dV_half, L_Lya_MCRT)
    M_F_radial_plot_half = force_multiplier_from_force_density(frad_plot, dV_half, L_Lya_MCRT)

    diag_rz = colt_MF_for_flattening_order(ar_raw, az_raw, "r_z")
    diag_zr = colt_MF_for_flattening_order(ar_raw, az_raw, "z_r")

    return {
        "xcrit": xcrit,
        "path": colt_path_r,
        "ar_raw": ar_raw,
        "az_raw": az_raw,
        "ar_full": ar_full,
        "az_full": az_full,
        "ar_half": ar_half,
        "az_half": az_half,
        "selected_half": selected_half,
        "plot_mode": plot_mode,
        "f_r": fr_plot,
        "f_z": fz_plot,
        "mag": mag_plot,
        "integral_label": int_label,
        "integral_shape": mag_int.shape,
        "M_F": M_F_full,
        "M_F_radial": M_F_radial_full,
        "M_F_radial_out": M_F_radial_out_full,
        "folded": folded,
        "label_source": label_source,
        "M_F_label": M_F_label,
        "M_F_radial_label": M_F_radial_label,
        "M_F_radial_out_label": M_F_radial_out_label,
        "M_F_plot_half": M_F_plot_half,
        "M_F_radial_plot_half": M_F_radial_plot_half,
        "diag_rz": diag_rz,
        "diag_zr": diag_zr,
    }


colts = {
    xcrit: read_colt_force_for_plot(xcrit, fname)
    for xcrit, fname in COLT_FileNames.items()
}

# Force magnitudes for plotting.
lyd_mag = force_magnitude(lyd["f_r"], lyd["f_z"])

# Global force max for quiver scaling.
force_global_max = np.nanmax([
    np.nanmax(np.abs(lyd_mag)),
    *(np.nanmax(np.abs(colt["mag"])) for colt in colts.values()),
])


def relative_difference(a, b):
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or b == 0.0:
        return None
    return (a - b) / b


print("\n========== Loaded force comparison ==========")
print(f"LYDION file:       {lyd['path']}")
print(f"LYDION f_r shape:  {lyd['f_r'].shape}")
print(f"LYDION f_z shape:  {lyd['f_z'].shape}")
print(f"LYDION nHI shape:  {lyd['nHI'].shape}")
print(f"LYDION dV shape:   {lyd['dV'].shape}")
print(f"LYDION direct integration domain: {lyd_domain_label}")
print(f"LYDION transpose force fields for integrals: {LYDION_transpose_force_fields_for_integrals}")
print(f"COLT_flattening_order used: {COLT_flattening_order}")
print(f"MCRT plot folded symmetry: {plot_mcrt_folded_symmetry}")
print(f"MCRT label source: {MCRT_MF_label_source}")
print(f"LYDION saved-half volume: {np.nansum(dV_half):.6e} cm^3")
print(f"LYDION full mirrored volume: {np.nansum(dV_full):.6e} cm^3")
print(f"LYDION snapshot:    k={lyd['snapshot_index']}, t={lyd['t'][lyd['snapshot_index']]:.6e}")
print(f"Global max |f_Lya|: {force_global_max:.6e}")
print(f"L_Lya used for LYDION M_F: {L_Lya_LYDION:.6e} erg/s")
print(f"L_Lya used for MCRT M_F:   {L_Lya_MCRT:.6e} erg/s")
print(f"LYDION M_F saved:                       {M_F_LYDION_saved}")
print(f"LYDION M_F direct:                      {M_F_LYDION_direct:.6e}")
if M_F_LYDION_saved is not None:
    diff = relative_difference(M_F_LYDION_direct, M_F_LYDION_saved)
    print(f"LYDION M_F direct - saved / saved:      {diff:.6e}")
print(f"LYDION M_F_radial_net saved:            {M_F_radial_net_LYDION_saved}")
print(f"LYDION M_F_radial_net direct:           {M_F_radial_LYDION_direct:.6e}")
if M_F_radial_net_LYDION_saved is not None:
    diff = relative_difference(M_F_radial_LYDION_direct, M_F_radial_net_LYDION_saved)
    print(f"LYDION M_F_radial_net direct-saved/saved:{diff:.6e}")
print(f"LYDION M_F_radial_out saved:            {M_F_radial_out_LYDION_saved}")
print(f"LYDION M_F_radial_out direct:           {M_F_radial_out_LYDION_direct:.6e}")
if M_F_radial_out_LYDION_saved is not None:
    diff = relative_difference(M_F_radial_out_LYDION_direct, M_F_radial_out_LYDION_saved)
    print(f"LYDION M_F_radial_out direct-saved/saved:{diff:.6e}")
print(f"LYDION M_F shown:                       {M_F_LYDION:.6e}")
print(f"LYDION M_F_radial shown:                {M_F_radial_LYDION:.6e}")
for xcrit in sorted(colts):
    colt = colts[xcrit]
    print(f"COLT xcrit{xcrit} file:                    {colt['path']}")
    print(f"COLT xcrit{xcrit} full shape:              {colt['ar_full'].shape}")
    print(f"COLT xcrit{xcrit} plotted shape:           {colt['mag'].shape}")
    print(f"COLT xcrit{xcrit} selected half:           {colt['selected_half']}")
    print(f"COLT xcrit{xcrit} plot mode:               {colt['plot_mode']}")
    print(f"COLT xcrit{xcrit} integral domain:         {colt['integral_label']}, shape={colt['integral_shape']}")
    if colt['diag_rz'] is not None and colt['diag_zr'] is not None:
        print(f"COLT xcrit{xcrit} diagnostic r_z: domain={colt['diag_rz'][0]}, M_F={colt['diag_rz'][1]:.6e}, M_F_radial={colt['diag_rz'][2]:.6e}")
        print(f"COLT xcrit{xcrit} diagnostic z_r: domain={colt['diag_zr'][0]}, M_F={colt['diag_zr'][1]:.6e}, M_F_radial={colt['diag_zr'][2]:.6e}")
    print(f"M_F COLT xcrit{xcrit} ({colt['integral_label']}):        {colt['M_F']:.6e}")
    print(f"M_F_radial COLT xcrit{xcrit} ({colt['integral_label']}):  {colt['M_F_radial']:.6e}")
    if colt['folded'] is not None:
        print(f"M_F COLT xcrit{xcrit} folded-symmetry:           {colt['folded'][0]:.6e}")
        print(f"M_F_radial COLT xcrit{xcrit} folded-symmetry:    {colt['folded'][1]:.6e}")
        print(f"M_F_radial_out COLT xcrit{xcrit} folded-symmetry:{colt['folded'][2]:.6e}")
        print(f"M_F COLT xcrit{xcrit} folded/full ratio:         {colt['folded'][0]/colt['M_F']:.6e}")
    print(f"M_F COLT xcrit{xcrit} plotted-half only:        {colt['M_F_plot_half']:.6e}")
    print(f"M_F_radial COLT xcrit{xcrit} plotted-half only: {colt['M_F_radial_plot_half']:.6e}")
    print(f"M_F label COLT xcrit{xcrit} ({colt['label_source']}):     {colt['M_F_label']:.6e}")
    print(f"M_F_rad label COLT xcrit{xcrit} ({colt['label_source']}): {colt['M_F_radial_label']:.6e}")
print("=============================================\n")


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

R_L, Z_L = np.meshgrid(lyd["r"], lyd["Z"], indexing="ij")
R_C, Z_C = R_L, Z_L

force_norm, force_vmin, force_vmax = make_norm(
    force_scale,
    (lyd_mag, *(colt["mag"] for colt in colts.values())),
    force_vmin_user,
    force_vmax_user,
    force_log_vmin_fraction,
    force_log_vmax_factor,
)

nHI_norm, nHI_vmin, nHI_vmax = make_norm(
    nHI_scale,
    (lyd["nHI"],),
    nHI_vmin_user,
    nHI_vmax_user,
    nHI_log_vmin_fraction,
    nHI_log_vmax_factor,
)

# Copy colormaps so that masking/out-of-cloud cells are black without
# changing the global matplotlib colormap objects.
def cmap_with_black_bad_under(cmap_name):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("black")
    cmap.set_under("black")
    return cmap

force_cmap_plot = cmap_with_black_bad_under(force_cmap)
nHI_cmap_plot = cmap_with_black_bad_under(nHI_cmap)

# Cell-center grids for quivers.
RQ, ZQ = np.meshgrid(lyd["rCoM"], lyd["ZCoM"], indexing="ij")


###############################################################
#######                    P L O T                     #######
###############################################################

fig = plt.figure(figsize=figsize, constrained_layout=False)

# ------------------------------------------------------------------
# Manual axes placement
# ------------------------------------------------------------------
# Same 2x3 panel order as the energy-density comparison plot:
#   top row:    LYDION, xcrit0/no core-skipping, xcrit1
#   bottom row: xcrit2, xcrit3, nHI
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
cax_f = fig.add_axes([cbar_x, y1, colorbar_width, panel_height])
cax_n = fig.add_axes([cbar_x, y0, colorbar_width, panel_height])

axes = np.array([[axL, axC0, axC1], [axC2, axC3, axN]], dtype=object)

# Make masked/transparent pixels black instead of showing the white axes background.
for ax in axes.flat:
    ax.set_facecolor("black")

# LYDION force magnitude. Same convention as before: LYDION force-history
# fields use .T for pcolormesh in this file.
imL = axL.pcolormesh(
    R_L, Z_L,
    clipped_for_plot(lyd_mag.T, force_scale, force_vmin, force_vmax),
    shading="auto",
    cmap=force_cmap_plot,
    norm=force_norm,
    rasterized=True,
)

# COLT/MCRT force magnitude. The folded/default maps are already aligned with
# the corrected energy-density plot convention, so do NOT transpose here.
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
        clipped_for_plot(colts[xcrit]["mag"], force_scale, force_vmin, force_vmax),
        shading="auto",
        cmap=force_cmap_plot,
        norm=force_norm,
        rasterized=True,
    )

# nHI has already been reordered to the LYDION/FV convention (Nr, Nz)
# in read_lydion_file(). With R_L, Z_L built using indexing="ij",
# pcolormesh should receive this array directly. This keeps the panel identical
# to the energy-density plot.
imN = axN.pcolormesh(
    R_L, Z_L,
    clipped_for_plot(lyd["nHI"], nHI_scale, nHI_vmin, nHI_vmax),
    shading="auto",
    cmap=nHI_cmap_plot,
    norm=nHI_norm,
    rasterized=True,
)

# Optional force-direction arrows.
if plot_arrows:
    add_quiver(
        axL, RQ, ZQ, lyd["f_r"], lyd["f_z"], force_global_max,
        arrow_min_fraction, arrow_n_r, arrow_n_z,
        color=arrow_color, width=arrow_width,
        headwidth=arrow_headwidth, headlength=arrow_headlength,
        headaxislength=arrow_headaxislength,
        reference_length=arrow_reference_length,
    )
    for xcrit, ax in colt_axes.items():
        add_quiver(
            ax, RQ, ZQ, colts[xcrit]["f_r"], colts[xcrit]["f_z"], force_global_max,
            arrow_min_fraction, arrow_n_r, arrow_n_z,
            color=arrow_color, width=arrow_width,
            headwidth=arrow_headwidth, headlength=arrow_headlength,
            headaxislength=arrow_headaxislength,
            reference_length=arrow_reference_length,
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
for ax in [axC0, axC1, axC3, axN]:
    ax.tick_params(labelleft=False)

# Remove duplicate x tick labels on the top row.
for ax in [axL, axC0, axC1]:
    ax.tick_params(labelbottom=False)

# Match visible limits.
for ax in axes.flat:
    ax.set_xlim(lyd["r"][0], lyd["r"][-1])
    ax.set_ylim(lyd["Z"][0], lyd["Z"][-1])

# In-panel labels only. No panel titles above the axes.
panel_style = dict(
    ha="left",
    va="top",
    color="white",
    fontsize=30,
    path_effects=[pe.withStroke(linewidth=4, foreground="black")]
)

panel_text = {
    "LYDION": r"LYDION",
    0: "Monte Carlo RT\nNo core-skipping",
    1: "Monte Carlo RT\n$x_{crit}=1$",
    2: "Monte Carlo RT\n$x_{crit}=2$",
    3: "Monte Carlo RT\n$x_{crit}=3$",
    "nHI": r"H I density",
}

x, y = panel_label_positions["LYDION"]
axL.text(x, y, panel_text["LYDION"], transform=axL.transAxes, **panel_style)
for xcrit, ax in colt_axes.items():
    x, y = panel_label_positions[xcrit]
    ax.text(x, y, panel_text[xcrit], transform=ax.transAxes, **panel_style)
x, y = panel_label_positions["nHI"]
axN.text(x, y, panel_text["nHI"], transform=axN.transAxes, **panel_style)

# Same H I density annotations as in the e_Lya comparison plot.
axN.text(0.06, 0.20, "Emission\nregion", **panel_style)
axN.text(0.60, 0.60, "Thick H I shell", **panel_style, rotation=-60)
axN.text(0.06, 0.90, "Lower-opacity\ncone", **panel_style, rotation=-20)

MF_style = dict(
    ha="right",
    va="top",
    color="white",
    fontsize=30,
    linespacing=1.15,
    path_effects=[pe.withStroke(linewidth=4, foreground="black")]
)

if show_MF_labels:
    x, y = MF_label_positions["LYDION"]
    axL.text(
        x, y,
        format_MF_pair(M_F_LYDION, M_F_radial_LYDION),
        transform=axL.transAxes,
        **MF_style,
    )

    for xcrit, ax in colt_axes.items():
        x, y = MF_label_positions[xcrit]
        ax.text(
            x, y,
            format_MF_pair(colts[xcrit]["M_F_label"], colts[xcrit]["M_F_radial_label"]),
            transform=ax.transAxes,
            **MF_style,
        )

# Time label in LYDION panel.
time_label = rf"$t = {lyd['t'][lyd['snapshot_index']]:.3g}$"
# axL.text(
#     0.06, 0.84,
#     time_label,
#     transform=axL.transAxes,
#     ha="left",
#     va="top",
#     color="white",
#     fontsize=24,
#     path_effects=[pe.withStroke(linewidth=4, foreground="black")]
# )

# Separate colorbars placed to the right of the third column.
cbar_f = fig.colorbar(colt_images[0], cax=cax_f)
cbar_f.set_label(r"$|\mathbf{f}_{Ly\alpha}|$ (dyn cm$^{-3}$)", labelpad=15)

if force_scale == "log":
    cbar_f.ax.yaxis.set_major_locator(LogLocator(base=10))
    cbar_f.ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    cbar_f.ax.yaxis.set_minor_formatter(NullFormatter())

# Separate colorbar for nHI panel.
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
