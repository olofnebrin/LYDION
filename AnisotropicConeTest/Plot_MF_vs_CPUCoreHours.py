import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter, FuncFormatter, FixedLocator
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
cmastro_src = repo_root / "src" / "cmastro" / "src"
sys.path.insert(0, str(cmastro_src))

try:
    import cmastro
except Exception:
    cmastro = None


###############################################################
#######                  U S E R  I N P U T             #######
###############################################################

# Files should be in the same directory as this script, unless full paths are given.
LYDION_FileName = "CloudTest_theta0.7853981633974483_rs0.5_tau1.0e8_T10000.0_umax0.0_tauabs0.0_pd0.0_256x256.h5"

# Aaron's MCRT/COLT files are assumed to follow this naming pattern.
# Example: Lya-cloud-xcrit0_1e7photons.hdf5
COLT_FileName_Template = "Lya-cloud-xcrit{xcrit}_{photons}photons.hdf5"

# Photon-count labels used in the filenames.
PhotonCounts = {
    "1e7": 1.0e7,
    "2e6": 2.0e6,
    "4e5": 4.0e5,
}

xcrit_values = [0, 1, 2, 3]

outname = "ConeTest_MF_vs_CPUCoreHours.pdf"
savefig = True

# Which LYDION snapshot to read:
#   "final" -> final saved snapshot with t > 0
#   integer -> explicit snapshot index
snapshot_to_plot = "final"

# LYDION CPU core hours for the full run. Fill this in by hand.
# Example: if 64 cores ran for 0.25 hours, use 16.0.
LYDION_CPU_core_hours = 109.5

# Optionally move the LYDION marker to the effective CPU time where the
# force multiplier has converged, rather than the wall-clock cost of the
# full 4-diffusion-time run. This assumes cost scales linearly with time.
use_LYDION_effective_CPU_core_hours = True
LYDION_total_evolved_diffusion_times = 4.0
LYDION_convergence_fraction = 0.99

# Which convergence time to use for the LYDION x-position:
#   "both"      -> require both M_F and M_F_radial to be converged; use the later time
#   "M_F"       -> use only M_F_History
#   "M_F_radial"-> use only M_F_radial_net_History
#   "panelwise" -> use separate x-positions in the two panels
LYDION_effective_CPU_mode = "both"

# If True, the convergence time is the earliest time after which the curve
# remains within the requested fraction of the final value. If False, it is
# just the first crossing. The former is safer if there are small transients.
LYDION_convergence_must_remain_converged = True

# LYDION force datasets.
LYDION_fr_dataset = "f_Lya_r_History"
LYDION_fz_dataset = "f_Lya_Z_History"

# COLT/MCRT acceleration datasets.
# The *_scat versions usually converge faster.
COLT_ar_dataset = "a_rad_r_scat"
COLT_az_dataset = "a_rad_z_scat"

# H I number-density dataset in the LYDION static-cloud HDF5 file.
nHI_dataset = "nHI"

# Aaron's MCRT timings.
MCRT_seconds_per_photon_no_core_skipping = 3.0
MCRT_speedup = {
    0: 1.0,
    1: 10.7,
    2: 762.0,
    3: 2710.0,
}

# Aaron's output has the full Z-domain because he did not use reflective BCs.
# LYDION likely has only the reflective half-plane.
#
# For the force multipliers below, the code uses folded symmetry whenever the
# MCRT file contains the full domain. That is the preferred/statistically best
# estimate for this symmetric cone test.
#
# Options for fallback if a file is already half-domain or plotting diagnostics:
#   "upper" -> use second half of COLT Z array, assuming full domain is ordered [-Zmax, ..., +Zmax]
#   "lower" -> use first half
#   "auto"  -> choose whichever half has closer total force magnitude to LYDION
colt_half_to_use = "upper"

# COLT/MCRT arrays are compared/integrated against the LYDION cylindrical
# cell volumes. If Aaron's output is a flat array, this controls how it is
# reshaped onto the LYDION grid.
#
# Options:
#   "r_z" -> arr.reshape((Nr, Nz_or_2Nz))
#   "z_r" -> arr.reshape((Nz_or_2Nz, Nr)).T
COLT_flattening_order = "r_z"

# For LYDION force-field integrals, h5py exposes the saved Julia arrays in
# the opposite orientation compared to CellVolumes for these square grids.
# Keeping this True makes the direct Python integral reproduce M_F_History.
LYDION_transpose_force_fields_for_integrals = True

# Physical constants.
mH_cgs   = 1.6735575e-24  # g
c_cgs    = 2.99792458e10  # cm s^-1
Lsun_cgs = 3.839e33       # erg s^-1

# Luminosity used for the COLT/MCRT force multiplier.
L_Lya_MCRT = 1e5 * Lsun_cgs

# LYDION saves physical cell volumes in cgs units.
CellVolumes_dataset = "CellVolumes"

# Plot style.
figsize = (9.0, 13.0)
dpi_save = 400

# Colors identify core-skipping level.
xcrit_colors = {
    0: "black",
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:red",
}

# Markers identify photon count.
photon_markers = {
    "1e7": "o",
    "2e6": "s",
    "4e5": "^",
}

# Marker sizes.
mcrt_marker_size = 250
lydion_marker_size = 600

# Draw faint connecting lines for fixed xcrit as photon count changes.
connect_same_xcrit = True
connection_alpha = 0.35
connection_lw = 3.0

# Axis limits. Set to None for automatic.
xlim_user = (0.05, 5e4)
ylim_MF_user = (0.0, 50.0)
ylim_MFrad_user = (0.0, 32.0)

# If True, annotate each MCRT point with its xcrit and photon-count label.
annotate_points = False

# If True, draw horizontal reference lines using the highest-photon-count,
# no-core-skipping MCRT result. This is useful as the MCRT baseline.
draw_no_core_highest_N_reference = True
no_core_reference_xcrit = 0
no_core_reference_photon_label = "1e7"
reference_line_color = "0.35"
reference_line_lw = 2.2
reference_line_alpha = 0.75
reference_line_dashes = (6.0, 5.0)

# Instead of legends, use text labels placed manually in the panels.
# Coordinates are in data units. Set add_manual_panel_labels = False to disable.
#
# The series labels are only placed in the upper panel by default. This avoids
# repeating the same information in the lower panel, leaving room there for
# photon-count labels.
add_manual_panel_labels = True
manual_series_labels = [
    {"ax": "MF", "text": "MCRT,\nno core-skipping", "x": 2.0e3, "y": 36.5, "color": xcrit_colors[0], "fontsize": 24, "ha": "center", "va": "bottom"},
    {"ax": "MF", "text": r"MCRT, $x_{crit}=1$", "x": 4e2, "y": 18.0, "color": xcrit_colors[1], "fontsize": 24, "ha": "center", "va": "bottom"},
    {"ax": "MF", "text": r"MCRT, $x_{crit}=2$", "x": 3.0, "y": 10.0, "color": xcrit_colors[2], "fontsize": 24, "ha": "center", "va": "bottom"},
    {"ax": "MF", "text": r"MCRT, $x_{crit}=3$", "x": 1.0, "y": 5.1, "color": xcrit_colors[3], "fontsize": 24, "ha": "center", "va": "top"},
    {"ax": "MF", "text": "LYDION", "x": 6.7, "y": 24.0, "color": "seagreen", "fontsize": 24, "ha": "center", "va": "bottom"},
]

# Photon-count legend.
#
# This legend only uses the no-core-skipping marker sequence by default, since
# the same marker encoding is repeated for all xcrit values. This avoids
# overcrowding the data panels with direct photon-count text labels.
add_photon_count_legend = True
photon_count_legend_axis = "radial"
photon_count_legend_reference_xcrit = 0
photon_count_legend_order = ["1e7", "2e6", "4e5"]
photon_count_legend_labels = {
    "1e7": r"$10^7$ photons",
    "2e6": r"$2\times10^6$ photons",
    "4e5": r"$4\times10^5$ photons",
}
photon_count_legend_loc = "upper left"
photon_count_legend_bbox = (0.02, 0.95)
photon_count_legend_ncol = 1
photon_count_legend_fontsize = 24
photon_count_legend_labelspacing = 0.35*0.1
photon_count_legend_handletextpad = 0.55*0.1
photon_count_legend_borderaxespad = 0.0
photon_count_legend_markerscale = 0.90

# Tick controls. These are useful when changing the figure size/aspect.
# Use FixedLocator for the major x ticks so Matplotlib does not prune
# intermediate decade labels when the figure aspect/size is changed.
x_major_ticks = [0.1, 1.0, 10.0, 100.0, 1.0e3, 1.0e4]
x_minor_subs = np.arange(2, 10) * 0.1
x_minor_tick_length = 6
x_minor_tick_width = 2.2
y_minor_intervals = 10
y_minor_tick_length = 6
y_minor_tick_width = 2.2


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

        if LYDION_fr_dataset not in f or LYDION_fz_dataset not in f:
            raise RuntimeError(f"LYDION force-history datasets not found in {path}")
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
        "M_F_History": M_F_hist,
        "M_F_radial_net_History": M_F_radial_net_hist,
        "M_F_radial_out_History": M_F_radial_out_hist,
        "L_Lya_History": L_Lya_hist,
    }


def find_convergence_time(t, y, final_value, threshold=0.99, require_stays=True):
    """
    Return the earliest time where y is within threshold of final_value.

    If require_stays=True, require all later saved snapshots to remain within
    the threshold. This avoids treating a temporary crossing/overshoot as
    convergence. The tolerance is interpreted relative to |final_value|.
    """
    if y is None or final_value is None or not np.isfinite(final_value):
        return None

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    n = min(t.size, y.size)
    t = t[:n]
    y = y[:n]

    good = np.isfinite(t) & np.isfinite(y) & (t > 0.0)
    if not np.any(good):
        return None

    t_good = t[good]
    y_good = y[good]

    tol = (1.0 - threshold) * max(abs(final_value), 1.0e-300)
    converged = np.abs(y_good - final_value) <= tol

    if require_stays:
        # stays_converged[i] is True only if every snapshot from i onward
        # is within tolerance.
        stays_converged = np.logical_and.accumulate(converged[::-1])[::-1]
        idx = np.where(stays_converged)[0]
    else:
        idx = np.where(converged)[0]

    if idx.size == 0:
        return None

    return float(t_good[int(idx[0])])


def get_lydion_effective_cpu_hours(lyd, full_cpu_hours):
    """
    Estimate the effective LYDION CPU hours from force-multiplier convergence.

    The HDF5 Times array is assumed to be in diffusion-time units for this
    static-cloud performance test. The full run evolved for
    LYDION_total_evolved_diffusion_times, so the effective cost is scaled by
    t_converged / LYDION_total_evolved_diffusion_times.
    """
    if not use_LYDION_effective_CPU_core_hours:
        return full_cpu_hours, full_cpu_hours, {}

    t = lyd["t"]
    conv = {}

    conv["M_F"] = find_convergence_time(
        t, lyd.get("M_F_History"), M_F_LYDION,
        threshold=LYDION_convergence_fraction,
        require_stays=LYDION_convergence_must_remain_converged,
    )
    conv["M_F_radial"] = find_convergence_time(
        t, lyd.get("M_F_radial_net_History"), M_F_radial_LYDION,
        threshold=LYDION_convergence_fraction,
        require_stays=LYDION_convergence_must_remain_converged,
    )

    def scale_cpu(t_conv):
        if t_conv is None:
            return full_cpu_hours
        return full_cpu_hours * t_conv / LYDION_total_evolved_diffusion_times

    if LYDION_effective_CPU_mode == "M_F":
        cpu_MF = scale_cpu(conv["M_F"])
        cpu_rad = cpu_MF
    elif LYDION_effective_CPU_mode == "M_F_radial":
        cpu_rad = scale_cpu(conv["M_F_radial"])
        cpu_MF = cpu_rad
    elif LYDION_effective_CPU_mode == "panelwise":
        cpu_MF = scale_cpu(conv["M_F"])
        cpu_rad = scale_cpu(conv["M_F_radial"])
    elif LYDION_effective_CPU_mode == "both":
        times = [tt for tt in (conv["M_F"], conv["M_F_radial"]) if tt is not None]
        if len(times) == 0:
            cpu_MF = full_cpu_hours
            cpu_rad = full_cpu_hours
            conv["both"] = None
        else:
            conv["both"] = max(times)
            cpu_both = scale_cpu(conv["both"])
            cpu_MF = cpu_both
            cpu_rad = cpu_both
    else:
        raise ValueError(
            "LYDION_effective_CPU_mode must be 'both', 'M_F', "
            "'M_F_radial', or 'panelwise'."
        )

    return cpu_MF, cpu_rad, conv


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


def force_multiplier_from_force_mag(force_mag, dV, L_Lya):
    return force_multiplier_from_force_density(force_mag, dV, L_Lya)


def mcrt_cpu_core_hours(Nphotons, xcrit):
    return (
        MCRT_seconds_per_photon_no_core_skipping
        * float(Nphotons)
        / 3600.0
        / MCRT_speedup[xcrit]
    )


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


def format_scientific_latex(value, sigfigs=1):
    """Return labels like $4\times10^5$ instead of 4e5."""
    value = float(value)
    if value <= 0.0 or not np.isfinite(value):
        return "?"

    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10.0**exponent

    if np.isclose(mantissa, 10.0, rtol=0.0, atol=10.0**(-sigfigs)):
        mantissa = 1.0
        exponent += 1

    mantissa_str = f"{mantissa:.{sigfigs}g}"
    if mantissa_str == "1":
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa_str}\times10^{{{exponent}}}$"


def format_photon_label(photon_label):
    """Format PhotonCounts keys using explicit multiplication, not 4e5-style text."""
    return format_scientific_latex(PhotonCounts[photon_label], sigfigs=1)


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


def read_mcrt_force_multipliers(xcrit, photon_label, lyd, rho_ref_half, rho_ref_full, dV_half, dV_full, ZCoM_full):
    """
    Read one MCRT/COLT file and compute folded M_F and M_F_radial.

    If the MCRT file contains the full domain, the force field is folded across
    the midplane before evaluating |f| and f.s_hat. The folded upper-half
    integral is then multiplied by 2 to represent the full cloud.
    """
    fname = COLT_FileName_Template.format(xcrit=xcrit, photons=photon_label)

    path_r, ar_raw = read_colt_field(fname, COLT_ar_dataset)
    _,      az_raw = read_colt_field(fname, COLT_az_dataset)

    Nr = len(lyd["r"]) - 1
    Nz = len(lyd["Z"]) - 1

    ar = reshape_colt_to_grid(ar_raw, Nr, Nz)
    az = reshape_colt_to_grid(az_raw, Nr, Nz)

    # Full-domain files: fold to upper half before taking |f| to reduce MC noise.
    if ar.shape == dV_full.shape:
        fr_full = rho_ref_full * ar
        fz_full = rho_ref_full * az

        fr_fold, fz_fold = fold_full_domain_vector_to_upper(fr_full, fz_full)
        mag_fold = force_magnitude(fr_fold, fz_fold)
        frad_fold = spherical_radial_force_density(
            fr_fold, fz_fold, lyd["rCoM"], lyd["ZCoM"]
        )

        M_F = 2.0 * force_multiplier_from_force_mag(
            mag_fold, dV_half, L_Lya_MCRT
        )
        M_F_radial = 2.0 * force_multiplier_from_force_density(
            frad_fold, dV_half, L_Lya_MCRT
        )
        integral_label = "folded-symmetry"

    # Half-domain files: integrate the half-domain and multiply by 2.
    elif ar.shape == dV_half.shape:
        fr_half = rho_ref_half * ar
        fz_half = rho_ref_half * az

        mag_half = force_magnitude(fr_half, fz_half)
        frad_half = spherical_radial_force_density(
            fr_half, fz_half, lyd["rCoM"], lyd["ZCoM"]
        )

        M_F = 2.0 * force_multiplier_from_force_mag(
            mag_half, dV_half, L_Lya_MCRT
        )
        M_F_radial = 2.0 * force_multiplier_from_force_density(
            frad_half, dV_half, L_Lya_MCRT
        )
        integral_label = "half-domain x2"

    else:
        raise RuntimeError(
            f"Unexpected reshaped MCRT array shape {ar.shape}. "
            f"Expected {dV_half.shape} or {dV_full.shape}."
        )

    return {
        "xcrit": xcrit,
        "photon_label": photon_label,
        "Nphotons": PhotonCounts[photon_label],
        "CPU_core_hours": mcrt_cpu_core_hours(PhotonCounts[photon_label], xcrit),
        "M_F": M_F,
        "M_F_radial": M_F_radial,
        "path": path_r,
        "integral_label": integral_label,
    }


###############################################################
#######                  R E A D  D A T A               #######
###############################################################

lyd = read_lydion_file(LYDION_FileName, snapshot_to_plot)

Nr = len(lyd["r"]) - 1
Nz = len(lyd["Z"]) - 1

rho_ref_half = lyd["nHI"] * mH_cgs
rho_ref_full = mirror_upper_half_scalar_to_full(lyd["nHI"]) * mH_cgs

dV_half = lyd["dV"]
dV_full = mirror_upper_half_scalar_to_full(lyd["dV"])
ZCoM_full = full_Z_centers_from_upper(lyd["ZCoM"])

L_Lya_LYDION = lyd["L_Lya"] if lyd["L_Lya"] is not None and lyd["L_Lya"] > 0.0 else L_Lya_MCRT

# Use saved LYDION force multipliers when available.
M_F_LYDION = lyd["M_F"]
M_F_radial_LYDION = lyd["M_F_radial_net"]

# Fallback: direct re-integration from force-density fields.
if M_F_LYDION is None or M_F_radial_LYDION is None:
    lyd_fr_half = np.asarray(lyd["f_r"], dtype=float)
    lyd_fz_half = np.asarray(lyd["f_z"], dtype=float)

    if LYDION_transpose_force_fields_for_integrals:
        lyd_fr_half = lyd_fr_half.T
        lyd_fz_half = lyd_fz_half.T

    lyd_fr_full, lyd_fz_full = mirror_upper_half_vector_to_full(
        lyd_fr_half, lyd_fz_half
    )
    lyd_mag_full = force_magnitude(lyd_fr_full, lyd_fz_full)
    lyd_frad_full = spherical_radial_force_density(
        lyd_fr_full, lyd_fz_full, lyd["rCoM"], ZCoM_full
    )

    if M_F_LYDION is None:
        M_F_LYDION = force_multiplier_from_force_mag(
            lyd_mag_full, dV_full, L_Lya_LYDION
        )

    if M_F_radial_LYDION is None:
        M_F_radial_LYDION = force_multiplier_from_force_density(
            lyd_frad_full, dV_full, L_Lya_LYDION
        )

LYDION_CPU_core_hours_MF, LYDION_CPU_core_hours_radial, LYDION_convergence_times = get_lydion_effective_cpu_hours(
    lyd, LYDION_CPU_core_hours
)

mcrt_results = []
for photon_label in PhotonCounts:
    for xcrit in xcrit_values:
        mcrt_results.append(
            read_mcrt_force_multipliers(
                xcrit, photon_label, lyd,
                rho_ref_half, rho_ref_full,
                dV_half, dV_full, ZCoM_full,
            )
        )

print("\n========== Loaded M_F performance comparison ==========")
print(f"LYDION file:       {lyd['path']}")
print(f"LYDION snapshot:   k={lyd['snapshot_index']}, t={lyd['t'][lyd['snapshot_index']]:.6e}")
print(f"LYDION CPU hours, full run:      {LYDION_CPU_core_hours:.6e}")
print(f"LYDION CPU hours, M_F marker:    {LYDION_CPU_core_hours_MF:.6e}")
print(f"LYDION CPU hours, radial marker: {LYDION_CPU_core_hours_radial:.6e}")
if use_LYDION_effective_CPU_core_hours:
    print(f"LYDION convergence fraction:     {LYDION_convergence_fraction:.4f}")
    print(f"LYDION convergence times:        {LYDION_convergence_times}")
print(f"LYDION M_F:        {M_F_LYDION:.6e}")
print(f"LYDION M_F_radial: {M_F_radial_LYDION:.6e}")
print(f"MCRT timing:       {MCRT_seconds_per_photon_no_core_skipping:g} s/photon for xcrit=0")
for xcrit in xcrit_values:
    print(f"MCRT speedup xcrit{xcrit}: {MCRT_speedup[xcrit]:.6e}")

for row in mcrt_results:
    print(
        f"MCRT xcrit{row['xcrit']}, N_gamma={format_photon_label(row['photon_label'])}: "
        f"CPU={row['CPU_core_hours']:.6e} core h, "
        f"M_F={row['M_F']:.6e}, "
        f"M_F_radial={row['M_F_radial']:.6e}, "
        f"source={row['integral_label']}, "
        f"file={row['path']}"
    )
print("=======================================================\n")


###############################################################
#######                P L O T  S E T U P               #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 28})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

plt.rcParams.update({'legend.fontsize': 18})
plt.rcParams["legend.markerscale"] = 0.85

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 2.4
plt.rcParams['xtick.minor.width'] = 2.2
plt.rcParams['xtick.major.size'] = 15
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['ytick.major.width'] = 2.4
plt.rcParams['ytick.minor.width'] = 2.2
plt.rcParams['ytick.major.size'] = 15
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['xtick.major.pad'] = '10'
plt.rcParams['ytick.major.pad'] = '10'

plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linewidth'] = 2.0
plt.rcParams['grid.linestyle'] = '--'

plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


###############################################################
#######                    P L O T                     #######
###############################################################

fig, axes = plt.subplots(
    2, 1,
    figsize=figsize,
    sharex=True,
    gridspec_kw={"hspace": 0.05},
)

ax_MF, ax_rad = axes

# Optional connecting lines for fixed core-skipping level.
if connect_same_xcrit:
    for xcrit in xcrit_values:
        rows = [row for row in mcrt_results if row["xcrit"] == xcrit]
        rows = sorted(rows, key=lambda row: row["CPU_core_hours"])

        x = np.array([row["CPU_core_hours"] for row in rows])
        y1 = np.array([row["M_F"] for row in rows])
        y2 = np.array([row["M_F_radial"] for row in rows])

        ax_MF.plot(
            x, y1,
            color=xcrit_colors[xcrit],
            lw=connection_lw,
            alpha=connection_alpha,
            zorder=1,
        )
        ax_rad.plot(
            x, y2,
            color=xcrit_colors[xcrit],
            lw=connection_lw,
            alpha=connection_alpha,
            zorder=1,
        )

# Optional horizontal reference lines for the best no-core-skipping MCRT result.
if draw_no_core_highest_N_reference:
    ref_rows = [
        row for row in mcrt_results
        if row["xcrit"] == no_core_reference_xcrit
        and row["photon_label"] == no_core_reference_photon_label
    ]
    if len(ref_rows) != 1:
        raise RuntimeError(
            "Could not uniquely identify the no-core-skipping reference row: "
            f"xcrit={no_core_reference_xcrit}, "
            f"photon_label={no_core_reference_photon_label}."
        )

    ref_row = ref_rows[0]
    for ax, yref in [(ax_MF, ref_row["M_F"]), (ax_rad, ref_row["M_F_radial"])]:
        line = ax.axhline(
            yref,
            color=reference_line_color,
            lw=reference_line_lw,
            alpha=reference_line_alpha,
            ls="--",
            zorder=0,
        )
        line.set_dashes(reference_line_dashes)
        line.set_dash_capstyle("round")

# MCRT markers.
for row in mcrt_results:
    xcrit = row["xcrit"]
    photon_label = row["photon_label"]

    color = xcrit_colors[xcrit]
    marker = photon_markers[photon_label]

    ax_MF.scatter(
        row["CPU_core_hours"],
        row["M_F"],
        s=mcrt_marker_size,
        marker=marker,
        color=color,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )

    ax_rad.scatter(
        row["CPU_core_hours"],
        row["M_F_radial"],
        s=mcrt_marker_size,
        marker=marker,
        color=color,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )

    if annotate_points:
        text = rf"$x_{{crit}}={xcrit}$, " + format_photon_label(photon_label)
        for ax, y in [(ax_MF, row["M_F"]), (ax_rad, row["M_F_radial"])]:
            ax.annotate(
                text,
                xy=(row["CPU_core_hours"], y),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=13,
                color=color,
            )

# LYDION marker.
for ax, x_cpu, y in [
    (ax_MF, LYDION_CPU_core_hours_MF, M_F_LYDION),
    (ax_rad, LYDION_CPU_core_hours_radial, M_F_radial_LYDION),
]:
    ax.scatter(
        x_cpu,
        y,
        s=lydion_marker_size,
        marker="*",
        color="seagreen",
        edgecolor="black",
        linewidth=1.4,
        zorder=5,
    )

ax_MF.set_title(r"LYDION vs. MCRT code performance", pad=16, fontsize = 33)

# Axes formatting.
for ax in axes:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(x_major_ticks))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=x_minor_subs, numticks=100))
    ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter_small_decimal))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.yaxis.set_major_formatter(FuncFormatter(compact_tick_formatter))
    ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor_intervals))

    ax.tick_params(axis="x", which="minor", top=True, bottom=True,
                   length=x_minor_tick_length, width=x_minor_tick_width)
    ax.tick_params(axis="y", which="minor", left=True, right=True,
                   length=y_minor_tick_length, width=y_minor_tick_width)

    #ax.grid(True, which="major", alpha=0.18)
    #ax.grid(True, which="minor", axis="x", alpha=0.08)

ax_MF.set_ylabel(r"$M_F$", labelpad=16, fontsize = 30)
ax_rad.set_ylabel(r"$M_{F,radial}$", labelpad=16, fontsize = 30)
ax_rad.set_xlabel(r"CPU core hours", labelpad=16, fontsize = 30)

if xlim_user is not None:
    ax_MF.set_xlim(*xlim_user)

if ylim_MF_user is not None:
    ax_MF.set_ylim(*ylim_MF_user)

if ylim_MFrad_user is not None:
    ax_rad.set_ylim(*ylim_MFrad_user)

# Direct labels, instead of legends.
if add_manual_panel_labels:
    for label in manual_series_labels:
        ax = ax_MF if label.get("ax", "MF") == "MF" else ax_rad
        ax.text(
            label["x"], label["y"], label["text"],
            color=label.get("color", "black"),
            fontsize=label.get("fontsize", 18),
            ha=label.get("ha", "center"),
            va=label.get("va", "center"),
            rotation=label.get("rotation", 0.0),
            zorder=6,
        )

if add_photon_count_legend:
    # Marker-only legend for photon counts. The handles are built from the
    # no-core-skipping sequence, but the meaning applies to all MCRT points.
    ax_phot_legend = ax_MF if photon_count_legend_axis == "MF" else ax_rad

    handles = []
    labels = []
    for photon_label in photon_count_legend_order:
        if photon_label not in photon_markers:
            continue

        handles.append(
            Line2D(
                [0], [0],
                marker=photon_markers[photon_label],
                linestyle="None",
                markerfacecolor=xcrit_colors[photon_count_legend_reference_xcrit],
                markeredgecolor="white",
                markeredgewidth=1.5,
                markersize=np.sqrt(mcrt_marker_size),
            )
        )
        labels.append(photon_count_legend_labels.get(photon_label, format_photon_label(photon_label)))

    ax_phot_legend.legend(
        handles, labels,
        loc=photon_count_legend_loc,
        bbox_to_anchor=photon_count_legend_bbox,
        ncol=photon_count_legend_ncol,
        fontsize=photon_count_legend_fontsize,
        frameon=False,
        handletextpad=photon_count_legend_handletextpad,
        labelspacing=photon_count_legend_labelspacing,
        borderaxespad=photon_count_legend_borderaxespad,
        markerscale=photon_count_legend_markerscale,
    )

if savefig:
    outpath = resolve_path(outname)
    fig.savefig(outpath, dpi=dpi_save, bbox_inches="tight")
    print(f"Saved figure: {outpath}")
else:
    plt.show()
