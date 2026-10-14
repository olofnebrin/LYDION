import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from pathlib import Path
import re


###############################################################
#######                                                 #######
#######                 U S E R   S E T U P             #######
#######                                                 #######
###############################################################

repo_root = Path(__file__).resolve().parent

data_dir = repo_root

save_name = "VelocityGradientsTest_ComovingSpectra.pdf"
save_path = data_dir / save_name

# Which umax values to plot.
# The script will look for:
#   LYDION: *umax{value}*.h5
#   MCRT:   comoving_MCRT_uniform_{value}.txt
umax_values = [10.0, 200.0]
#umax_values = [200.0]

# Simulation selection.
tau_cl_target = 5.0e8
T_target      = 9084.0
#resolution    = (64, 64)
resolution    = (128, 128)

# Source / physics selection from filename.
rs_target     = 1.0
tauabs_target = 0.0
pd_target     = 0.0

# Frequency-grid variants to compare.
#
# filename_suffix = "" searches for the normal/default file, e.g.
#   ..._128x128.h5
# filename_suffix = "_q1.03" searches for the finer-bin file, e.g.
#   ..._128x128_q1.03.h5
#
# The legend is intentionally only a style legend; velocity labels are added
# manually in the panel using velocity_text_entries below.
lydion_variants = [
    {
        "key": "default",
        "filename_suffix": "",
        "legend_label": r"LYDION default",
        "linestyle": "-",
        "linewidth": 4.5,
        "alpha": 1.0,
    },
    {
        "key": "q1.03",
        "filename_suffix": "_q1.03",
        "legend_label": r"LYDION finer bins",
        "linestyle": "--",
        "linewidth": 4.5,
        "alpha": 1.0,
    },
]

# If False, a missing optional finer-bin file is skipped with a warning.
# If True, the script stops when any listed variant is missing.
require_all_lydion_variants = False

# If several LYDION files match the same umax and variant, choose newest.
choose_newest_matching_file = True

# Spectrum edge selection:
#   "two_boundaries" : r-boundary near Z=0 and Z-boundary near r=0
#   "manual"         : use edge_indices
#   "all"            : plot all saved edge spectra individually
#   "average_all"    : average all saved edge spectra
edge_selection_mode = "average_all"

# Used only if edge_selection_mode = "manual".
edge_indices = [0]

# If edge_selection_mode is not "average_all", choose whether to
# average selected edges anyway.
plot_selected_edges_individually = False

# For midplane-symmetric runs, the lower-Z saved edge is not an escaping
# boundary. Ignore it when selecting/averaging edge spectra.
#
# This is useful for existing files where the HDF5 save code stored spectra
# at right, top, and lower-Z boundaries, while the lower-Z boundary is a
# symmetry/internal boundary rather than a true cloud escape surface.
exclude_lower_z_boundary = True

# Normalize each spectrum so integral J dx = 1.
normalize_spectra = True

# Optional: divide x by (a*tau)^(1/3), like Smith+ plots.
scale_x_by_atau13 = True

# Line styles.
mcrt_lw    = 10.0
mcrt_alpha = 0.35

# Optional x/y limits. If xlim = None, automatic limits are chosen.
xlim = None
ylim = (0.0, 0.75)

auto_xlim_threshold = 2.0e-3
auto_xlim_pad_frac  = 0.08

# Figure cosmetics.
figsize = (12.0, 9.5)
figure_title = None

legend_loc = "upper left"
legend_fontsize = 30

# User-specified colors for each velocity.
# Any umax not listed here falls back to color_cycle.
umax_colors = {
    10.0:  "dodgerblue",
    200.0: "orangered",
}

color_cycle = [
    "black",
    "dodgerblue",
    "orangered",
    "forestgreen",
    "purple",
    "goldenrod",
]

# Manual velocity labels in the panel.
# Adjust xy positions after inspecting the plotted spectra.
add_velocity_text = True
velocity_text_entries = [
    {
        "umax": 10.0,
        "xy": (0.7, 0.83),
        "text": r"10 km s$^{-1}$",
        "ha": "left",
        "va": "center",
    },
    {
        "umax": 200.0,
        "xy": (0.1, 0.55),
        "text": r"$\dot{R}_{cl}$= 200 km s$^{-1}$",
        "ha": "left",
        "va": "center",
    },
]
velocity_text_fontsize = 35

# Used only when plotting multiple LYDION edge spectra individually.
lydion_linestyle_cycle = ["-", "--", ":", "-."]

# Optional compact setup text box.
add_info_text = False
info_text_xy = (0.04, 0.94)
info_text_fontsize = 25


###############################################################
#######                                                 #######
#######                 C O N S T A N T S               #######
#######                                                 #######
###############################################################

c_cgs  = 2.99792458e10
kB_cgs = 1.380649e-16
mH_cgs = 1.6735575e-24

nu_Lya = 2.466e15  # Hz


###############################################################
#######                                                 #######
#######                 H E L P E R S                   #######
#######                                                 #######
###############################################################

rs_pattern     = re.compile(r"_rs([0-9eE+\-.]+)_")
tau_pattern    = re.compile(r"_tau([0-9eE+\-.]+)_")
T_pattern      = re.compile(r"_T([0-9eE+\-.]+)_")
umax_pattern   = re.compile(r"_umax([0-9eE+\-.]+)_")
tauabs_pattern = re.compile(r"_tauabs([0-9eE+\-.]+)_")
pd_pattern     = re.compile(r"_pd([0-9eE+\-.]+)_")
res_pattern    = re.compile(r"_(\d+)x(\d+)(?:_[^/]*)?\.h5$")


def parse_float_from_pattern(pattern, filename, label):
    match = pattern.search(filename)
    if match is None:
        raise ValueError(f"Could not parse {label} from {filename}")
    return float(match.group(1))


def doppler_width_from_T(T):
    vth = np.sqrt(2.0 * kB_cgs * T / mH_cgs)
    return (vth / c_cgs) * nu_Lya


def voigt_a_from_T(T):
    return 4.7e-4 * (T / 1.0e4)**(-0.5)


def x_scale_factor(T, tau_cl):
    if not scale_x_by_atau13:
        return 1.0

    a_v = voigt_a_from_T(T)
    return (a_v * tau_cl)**(1.0 / 3.0)


def integrate_trap(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))

    dx = np.diff(x)
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * dx))


def normalize_spectrum(y, x):
    y = np.asarray(y, dtype=float)

    if not normalize_spectra:
        return y

    area = integrate_trap(y, x)

    if not np.isfinite(area) or area <= 0.0:
        return y

    return y / area


def clean_tick_formatter(val, pos=None):
    if val == 0:
        return "0"

    if np.isclose(val, np.round(val), rtol=0.0, atol=1e-10):
        return f"{int(np.round(val))}"

    return f"{val:g}"


def velocity_label(umax):
    if np.isclose(umax, np.round(umax)):
        return rf"$u_{{max}}={int(np.round(umax))}\,\mathrm{{km\,s^{{-1}}}}$"
    return rf"$u_{{max}}={umax:g}\,\mathrm{{km\,s^{{-1}}}}$"


def mcrt_filename_for_umax(umax):
    if np.isclose(umax, np.round(umax)):
        tag = str(int(np.round(umax)))
    else:
        tag = f"{umax:g}"

    return data_dir / f"comoving_MCRT_uniform_{tag}.txt"


def color_for_umax(umax, n):
    for key, color in umax_colors.items():
        if np.isclose(umax, key, rtol=1e-10, atol=1e-10):
            return color

    return color_cycle[n % len(color_cycle)]


def file_matches_settings(filepath, umax_target, filename_suffix=""):
    filename = filepath.name

    if filename_suffix:
        if not filename.endswith(f"{filename_suffix}.h5"):
            return False
    else:
        # Default run: avoid accidentally selecting the finer-bin variant.
        if filename.endswith("_q1.03.h5"):
            return False

    try:
        rs = parse_float_from_pattern(rs_pattern, filename, "rs")
        tau_cl = parse_float_from_pattern(tau_pattern, filename, "tau_cl")
        T = parse_float_from_pattern(T_pattern, filename, "T")
        umax = parse_float_from_pattern(umax_pattern, filename, "umax")
        tauabs = parse_float_from_pattern(tauabs_pattern, filename, "tauabs")
        pd = parse_float_from_pattern(pd_pattern, filename, "pd")
    except ValueError:
        return False

    if not np.isclose(rs, rs_target, rtol=1e-8, atol=0.0):
        return False

    if not np.isclose(tau_cl, tau_cl_target, rtol=1e-8, atol=0.0):
        return False

    if not np.isclose(T, T_target, rtol=1e-8, atol=0.0):
        return False

    if not np.isclose(umax, umax_target, rtol=1e-8, atol=0.0):
        return False

    if not np.isclose(tauabs, tauabs_target, rtol=1e-8, atol=0.0):
        return False

    if not np.isclose(pd, pd_target, rtol=1e-8, atol=0.0):
        return False

    match_res = res_pattern.search(filename)
    if match_res is None:
        return False

    Nr = int(match_res.group(1))
    NZ = int(match_res.group(2))

    if (Nr, NZ) != tuple(resolution):
        return False

    return True


def find_lydion_file_for_umax(umax_target, variant):
    filename_suffix = variant.get("filename_suffix", "")

    candidates = [
        filepath
        for filepath in sorted(data_dir.glob("*.h5"))
        if file_matches_settings(filepath, umax_target, filename_suffix=filename_suffix)
    ]

    if len(candidates) == 0:
        raise RuntimeError(
            f"No matching LYDION HDF5 file found for umax={umax_target:g}, "
            f"variant={variant.get('key', filename_suffix)!r}.\n"
            f"  expected suffix = {filename_suffix!r}\n"
            f"  data_dir        = {data_dir}\n"
            f"  tau_cl     = {tau_cl_target:.4e}\n"
            f"  T          = {T_target:g}\n"
            f"  resolution = {resolution}"
        )

    if len(candidates) == 1:
        return candidates[0]

    if choose_newest_matching_file:
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = candidates[-1]
    else:
        chosen = candidates[0]

    print(f"\nMultiple LYDION files matched umax={umax_target:g}, variant={variant.get('key', filename_suffix)!r}.")
    print(f"Chosen file: {chosen.name}")
    print("Other candidates:")
    for c in candidates:
        if c != chosen:
            print(f"  {c.name}")

    return chosen


def reorder_lya_edge_spectra_safe(J_edge_raw, t, nuC, edge_ridx):
    """
    Reorder J_Lya_EdgeSpectra to shape (Nt, N_edge, N_nu).

    This explicitly uses len(J_Lya_Edge_rIndex) to identify the edge axis.
    """
    if J_edge_raw.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {J_edge_raw.shape}")

    Nt    = t.size
    Nν    = nuC.size
    Nedge = edge_ridx.size

    shape = J_edge_raw.shape

    time_axes = [ax for ax, s in enumerate(shape) if s == Nt]
    nu_axes   = [ax for ax, s in enumerate(shape) if s == Nν]
    edge_axes = [ax for ax, s in enumerate(shape) if s == Nedge]

    candidates = []

    for t_ax in time_axes:
        for e_ax in edge_axes:
            for nu_ax in nu_axes:
                if len({t_ax, e_ax, nu_ax}) == 3:
                    candidates.append((t_ax, e_ax, nu_ax))

    if len(candidates) != 1:
        raise RuntimeError(
            "Could not uniquely identify axes of J_Lya_EdgeSpectra.\n"
            f"  shape      = {shape}\n"
            f"  Nt         = {Nt}\n"
            f"  Nedge      = {Nedge}\n"
            f"  Nν         = {Nν}\n"
            f"  candidates = {candidates}\n"
            "If two dimensions have identical lengths, hard-code the axes here "
            "from the HDF5 save convention."
        )

    t_ax, e_ax, nu_ax = candidates[0]
    J_reordered = np.transpose(J_edge_raw, axes=[t_ax, e_ax, nu_ax])

    if J_reordered.shape != (Nt, Nedge, Nν):
        raise RuntimeError(
            f"Unexpected reordered shape {J_reordered.shape}, "
            f"expected {(Nt, Nedge, Nν)}."
        )

    print(
        f"    J_Lya_EdgeSpectra raw shape {shape} -> "
        f"(time, edge, freq) axes = ({t_ax}, {e_ax}, {nu_ax})"
    )

    return J_reordered


def get_true_escape_edge_mask(f):
    """
    Return a boolean mask over saved edge spectra.

    For midplane-symmetric static-cloud runs, the saved lower-Z boundary
    is a symmetry/internal boundary, not an escaping cloud boundary.
    We therefore exclude the minimum-Z saved edge group when
    exclude_lower_z_boundary = True.
    """
    edge_zidx = f["J_Lya_Edge_ZIndex"][:]

    Nedge = len(edge_zidx)
    mask = np.ones(Nedge, dtype=bool)

    if not exclude_lower_z_boundary:
        return mask

    ZCoM = f["ZCoM"][:]

    # HDF5 indices are Julia 1-based.
    edge_z = edge_zidx.astype(int) - 1

    if edge_z.min() < 0 or edge_z.max() >= ZCoM.size:
        raise RuntimeError("Edge Z-index array maps out of ZCoM bounds.")

    z_edge_pos = ZCoM[edge_z]

    # The lower/midplane saved boundary is the saved edge group with
    # the smallest Z position. This is not a true escaping boundary in
    # midplane-symmetric static-cloud runs.
    z_min = np.min(z_edge_pos)
    lower_z_mask = np.isclose(z_edge_pos, z_min, rtol=0.0, atol=1e-12)

    mask[lower_z_mask] = False

    print(
        f"    Excluding lower-Z/symmetry boundary: "
        f"{np.count_nonzero(lower_z_mask)} of {Nedge} saved edge spectra"
    )

    return mask


def pick_two_boundary_edges(edge_ridx, edge_zidx, rCoM, ZCoM, allowed_idx=None):
    """
    Auto-pick two true escape-boundary spectra:
      1) outer radial boundary near the midplane / lowest |Z|
      2) upper Z boundary near the axis / lowest r

    The optional allowed_idx is used to exclude the lower-Z symmetry boundary.
    """
    edge_r = edge_ridx.astype(int) - 1
    edge_z = edge_zidx.astype(int) - 1

    if (
        edge_r.min() < 0 or edge_r.max() >= rCoM.size
        or edge_z.min() < 0 or edge_z.max() >= ZCoM.size
    ):
        raise RuntimeError("Edge index arrays map out of rCoM/ZCoM bounds.")

    r_edge_pos = rCoM[edge_r]
    z_edge_pos = ZCoM[edge_z]

    if allowed_idx is None:
        allowed_idx = np.arange(len(edge_ridx), dtype=int)
    else:
        allowed_idx = np.asarray(allowed_idx, dtype=int)

    if allowed_idx.size == 0:
        raise RuntimeError("No allowed edge spectra were supplied.")

    # 1) Radial outer boundary: largest r among allowed edges,
    # then closest to the midplane.
    max_r = np.max(r_edge_pos[allowed_idx])
    cand_r = allowed_idx[
        np.isclose(r_edge_pos[allowed_idx], max_r, rtol=0.0, atol=1e-12)
    ]
    idx_r_boundary = int(cand_r[np.argmin(np.abs(z_edge_pos[cand_r]))])

    # 2) Upper Z boundary: largest Z among allowed edges,
    # then closest to the symmetry axis.
    max_z = np.max(z_edge_pos[allowed_idx])
    cand_z = allowed_idx[
        np.isclose(z_edge_pos[allowed_idx], max_z, rtol=0.0, atol=1e-12)
    ]
    idx_z_boundary = int(cand_z[np.argmin(r_edge_pos[cand_z])])

    print(
        "    two_boundaries selected:\n"
        f"      r-boundary edge {idx_r_boundary}: "
        f"r/Rcl={r_edge_pos[idx_r_boundary]:.6g}, "
        f"Z/Rcl={z_edge_pos[idx_r_boundary]:.6g}\n"
        f"      Z-boundary edge {idx_z_boundary}: "
        f"r/Rcl={r_edge_pos[idx_z_boundary]:.6g}, "
        f"Z/Rcl={z_edge_pos[idx_z_boundary]:.6g}"
    )

    return np.array([idx_r_boundary, idx_z_boundary], dtype=int)


def select_edge_indices(f, mode):
    edge_ridx = f["J_Lya_Edge_rIndex"][:]
    edge_zidx = f["J_Lya_Edge_ZIndex"][:]

    Nedge = len(edge_ridx)

    escape_mask = get_true_escape_edge_mask(f)
    allowed_idx = np.where(escape_mask)[0]

    if allowed_idx.size == 0:
        raise RuntimeError("No true escape edge spectra remain after filtering.")

    if mode == "two_boundaries":
        rCoM = f["rCoM"][:]
        ZCoM = f["ZCoM"][:]
        idx = pick_two_boundary_edges(
            edge_ridx,
            edge_zidx,
            rCoM,
            ZCoM,
            allowed_idx=allowed_idx,
        )

    elif mode == "manual":
        idx = np.asarray(edge_indices, dtype=int)
        if np.any(idx < 0) or np.any(idx >= Nedge):
            raise RuntimeError(
                f"Manual edge_indices={edge_indices} out of range [0,{Nedge-1}]."
            )

        if np.any(~escape_mask[idx]):
            bad = idx[~escape_mask[idx]]
            print(
                "    WARNING: manual edge selection includes lower-Z/symmetry "
                f"boundary index/indices {list(bad)}."
            )

    elif mode == "all":
        idx = allowed_idx

    elif mode == "average_all":
        idx = allowed_idx

    else:
        raise ValueError(
            "edge_selection_mode must be 'two_boundaries', 'manual', "
            "'all', or 'average_all'."
        )

    return idx

def read_lydion_final_spectrum(filepath):
    filename = filepath.name

    T = parse_float_from_pattern(T_pattern, filename, "T")
    tau_cl = parse_float_from_pattern(tau_pattern, filename, "tau_cl")

    dnu_D = doppler_width_from_T(T)
    xfac = x_scale_factor(T, tau_cl)

    with h5py.File(filepath, "r") as f:
        t = np.asarray(f["Times"][:], dtype=float)

        J_edge_raw = f["J_Lya_EdgeSpectra"][:]
        nuC = np.asarray(f["δνC"][:], dtype=float)

        edge_ridx = f["J_Lya_Edge_rIndex"][:]
        edge_zidx = f["J_Lya_Edge_ZIndex"][:]

        selected_idx = select_edge_indices(f, edge_selection_mode)

        J_edge_hist = reorder_lya_edge_spectra_safe(J_edge_raw, t, nuC, edge_ridx)

        finite_t = np.isfinite(t)
        if not np.any(finite_t):
            raise RuntimeError(f"{filename}: no finite values in Times.")

        # Use the maximum saved time. Robust against preallocated trailing zeros.
        k_final = int(np.nanargmax(t))
        t_final = float(t[k_final])

        x = (nuC / dnu_D) / xfac

        y_selected = []

        for idx in selected_idx:
            y = np.asarray(J_edge_hist[k_final, int(idx), :], dtype=float)
            y_selected.append(y)

        if edge_selection_mode == "average_all":
            y = np.mean(np.asarray(y_selected), axis=0)
            curves = [(None, y)]
        elif plot_selected_edges_individually:
            curves = [(int(idx), y) for idx, y in zip(selected_idx, y_selected)]
        else:
            y = np.mean(np.asarray(y_selected), axis=0)
            curves = [(None, y)]

    order = np.argsort(x)
    x = x[order]

    curves_out = []

    for idx, y in curves:
        y = y[order]
        y = normalize_spectrum(y, x)
        curves_out.append((idx, y))

    return {
        "filename": filename,
        "x": x,
        "curves": curves_out,
        "edge_indices": selected_idx,
        "k_final": k_final,
        "t_final": t_final,
        "T": T,
        "tau_cl": tau_cl,
    }


def read_mcrt_spectrum(filepath, T, tau_cl):
    if not filepath.exists():
        raise RuntimeError(f"MCRT file not found: {filepath}")

    data = np.loadtxt(filepath)

    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(
            f"Expected at least two columns in MCRT file {filepath}, "
            f"got shape {data.shape}."
        )

    x = np.asarray(data[:, 0], dtype=float)
    y = np.asarray(data[:, 1], dtype=float)

    xfac = x_scale_factor(T, tau_cl)
    x = x / xfac

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    y = normalize_spectrum(y, x)

    return x, y


def choose_auto_xlim(all_curves):
    ymax = 0.0

    for x, y in all_curves:
        good = np.isfinite(x) & np.isfinite(y)
        if np.any(good):
            ymax = max(ymax, float(np.nanmax(y[good])))

    if ymax <= 0.0 or not np.isfinite(ymax):
        return None

    threshold = auto_xlim_threshold * ymax
    x_keep = []

    for x, y in all_curves:
        good = np.isfinite(x) & np.isfinite(y) & (y > threshold)
        if np.any(good):
            x_keep.append(x[good])

    if len(x_keep) == 0:
        return None

    xcat = np.concatenate(x_keep)

    xmin = float(np.nanmin(xcat))
    xmax = float(np.nanmax(xcat))

    span = xmax - xmin
    if span <= 0.0 or not np.isfinite(span):
        return None

    pad = auto_xlim_pad_frac * span
    return xmin - pad, xmax + pad


def peak_positions(x, y, nmax=4):
    """
    Quick diagnostic peak finder without scipy.
    Returns up to nmax local maxima, sorted by peak height.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size < 3:
        return []

    loc = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1

    if loc.size == 0:
        return []

    order = np.argsort(y[loc])[::-1]
    loc = loc[order[:nmax]]

    return [(float(x[i]), float(y[i])) for i in loc]


###############################################################
#######                                                 #######
#######                 R E A D   D A T A               #######
#######                                                 #######
###############################################################

print("\nReading spectra:")

spectra = {}
all_curves_for_xlim = []

for n, umax in enumerate(umax_values):
    color = color_for_umax(umax, n)
    variant_data = []

    print(f"\n  umax = {umax:g} km/s")

    for variant in lydion_variants:
        try:
            lydion_file = find_lydion_file_for_umax(umax, variant)
            lydion_data = read_lydion_final_spectrum(lydion_file)
        except RuntimeError as err:
            if require_all_lydion_variants:
                raise
            print(f"    WARNING: {err}")
            continue

        variant_data.append({
            "variant": variant,
            "lydion": lydion_data,
        })

        print(f"\n    LYDION {variant['key']} file = {lydion_data['filename']}")
        print(f"      k_final     = {lydion_data['k_final']}")
        print(f"      t_final     = {lydion_data['t_final']:.6g}")
        print(f"      selected edge indices = {list(lydion_data['edge_indices'])}")

        for idx, y in lydion_data["curves"]:
            peaks = peak_positions(lydion_data["x"], y)
            print(f"      LYDION edge {idx}: strongest local peaks (x, y) = {peaks}")
            all_curves_for_xlim.append((lydion_data["x"], y))

    if len(variant_data) == 0:
        raise RuntimeError(f"No LYDION spectra could be read for umax={umax:g}.")

    # Use the first available LYDION file only to set the MCRT x-scaling.
    # T and tau_cl should be identical between variants for the same setup.
    reference_lydion = variant_data[0]["lydion"]

    mcrt_file = mcrt_filename_for_umax(umax)
    x_mcrt, y_mcrt = read_mcrt_spectrum(
        mcrt_file,
        reference_lydion["T"],
        reference_lydion["tau_cl"],
    )

    print(f"    MCRT file   = {mcrt_file.name}")
    peaks_mcrt = peak_positions(x_mcrt, y_mcrt)
    print(f"    MCRT strongest local peaks (x, y) = {peaks_mcrt}")

    spectra[umax] = {
        "variants": variant_data,
        "mcrt_file": mcrt_file.name,
        "x_mcrt": x_mcrt,
        "y_mcrt": y_mcrt,
        "color": color,
    }

    all_curves_for_xlim.append((x_mcrt, y_mcrt))


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

params = {'legend.fontsize': legend_fontsize}
plt.rcParams.update(params)
plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 2.5
plt.rcParams['xtick.minor.width'] = 2.5
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['ytick.minor.width'] = 2.5
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['ytick.minor.size'] = 8
plt.rcParams['xtick.major.pad'] = '15'
plt.rcParams['ytick.major.pad'] = '15'

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
#######                                                 #######
#######                   P L O T   I T                 #######
#######                                                 #######
###############################################################

fig, ax = plt.subplots(figsize=figsize)

legend_done = {
    "mcrt": False,
}
for variant in lydion_variants:
    legend_done[variant["key"]] = False

for umax in umax_values:
    entry = spectra[umax]
    color = entry["color"]

    # MCRT thick band. Only label the first one, so the legend remains clean.
    mcrt_label = "Monte Carlo RT" if not legend_done["mcrt"] else "_nolegend_"
    legend_done["mcrt"] = True

    ax.plot(
        entry["x_mcrt"],
        entry["y_mcrt"],
        color=color,
        linewidth=mcrt_lw,
        alpha=mcrt_alpha,
        solid_capstyle="round",
        label=mcrt_label,
        zorder=1,
    )

    # LYDION curves for each frequency-grid variant.
    for item in entry["variants"]:
        variant = item["variant"]
        lydion_data = item["lydion"]

        variant_key = variant["key"]
        variant_label = variant["legend_label"] if not legend_done[variant_key] else "_nolegend_"
        legend_done[variant_key] = True

        for n_curve, (idx, y) in enumerate(lydion_data["curves"]):
            label = variant_label if n_curve == 0 else "_nolegend_"

            linestyle = variant.get("linestyle", "-")
            if idx is not None and plot_selected_edges_individually:
                linestyle = lydion_linestyle_cycle[n_curve % len(lydion_linestyle_cycle)]

            ax.plot(
                lydion_data["x"],
                y,
                color=color,
                linewidth=variant.get("linewidth", 3.0),
                linestyle=linestyle,
                alpha=variant.get("alpha", 1.0),
                label=label,
                dash_capstyle = 'round',
                zorder=3 if variant_key == "default" else 4,
            )

if scale_x_by_atau13:
    ax.set_xlabel(r"Frequency displacement $x/(a_v\tau_{cl})^{1/3}$", labelpad=15)
else:
    ax.set_xlabel(r"$x = (\nu - \nu_{Ly\alpha})/\Delta\nu_D$", labelpad=15)

if normalize_spectra:
    if scale_x_by_atau13:
        ax.set_ylabel(r"$J(x)/\int\, dx\,J(x)$", labelpad=15)
    else:
        ax.set_ylabel(r"$J(x)/\int\, dx\,J(x)$", labelpad=15)
else:
    ax.set_ylabel(r"$J(x)$", labelpad=15)

if figure_title is not None:
    ax.set_title(figure_title, pad=20, fontsize=39.5)

if xlim is not None:
    ax.set_xlim(*xlim)
else:
    auto_xlim = choose_auto_xlim(all_curves_for_xlim)
    if auto_xlim is not None:
        ax.set_xlim(*auto_xlim)

if ylim is not None:
    ax.set_ylim(*ylim)
else:
    ax.set_ylim(bottom=0.0)

ax.xaxis.set_major_formatter(FuncFormatter(clean_tick_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(clean_tick_formatter))

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

if add_info_text:
    info_text = (
        rf"$\tau_{{cl}} = 5\times10^8$" + "\n"
        rf"$T = 9084\,\mathrm{{K}}$" + "\n"
        rf"${resolution[0]}^2$ cells"
    )

    ax.text(
        info_text_xy[0],
        info_text_xy[1],
        info_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=info_text_fontsize,
    )

if add_velocity_text:
    for item in velocity_text_entries:
        umax = item["umax"]
        text_color = color_for_umax(umax, 0)

        ax.text(
            item["xy"][0],
            item["xy"][1],
            item["text"],
            transform=ax.transAxes,
            ha=item.get("ha", "left"),
            va=item.get("va", "center"),
            fontsize=velocity_text_fontsize,
            color=text_color,
        )

ax.legend(
    loc=legend_loc,
    fontsize=legend_fontsize,
    handlelength=1.6,
    labelspacing=0.35,
    borderpad=0.25*3,
    handletextpad=0.6,
)

fig.tight_layout()
fig.savefig(save_path, bbox_inches="tight", dpi=250)
print(f"\nSaved figure: {save_path}")

plt.show()