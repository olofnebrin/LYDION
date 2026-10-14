import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator
from pathlib import Path
import re


###############################################################
#######                                                 #######
#######                 U S E R   S E T U P             #######
#######                                                 #######
###############################################################

repo_root = Path(__file__).resolve().parent

data_dir = repo_root / "LYDION_DataDustFreeStaticClouds"

save_name = "DustFree_StaticCloud_eLya_Profile.pdf"
save_path = repo_root / save_name

# Run to plot.
tau_cl     = 1.0e10
resolution = (128, 128)

# Only include files matching these choices.
require_NoRecoil = True
require_tauabs0  = True
require_pd0      = True
require_umax0    = True

# If several files match the same tau/resolution, choose newest file.
choose_newest_matching_file = True

# Which saved LYDION times to plot.
#   "fractions_final" : pick times nearest to time_fractions_of_final * t_final
#   "manual_indices"  : use time_indices
#   "manual_times"    : use saved times nearest to time_values
#   "all"             : plot all saved times, optionally thinned by max_number_of_times
#
time_selection_mode = "fractions_final"

time_fractions_of_final = [0.05, 0.10, 0.20, 0.50, 1.00]
time_indices            = [0, 5, 10, -1]
time_values             = [0.5, 1.0, 2.0, 4.0]

# Only used if time_selection_mode = "all".
max_number_of_times = 8

# The saved times in these static-cloud files appear to be in units of
# (a_v tau_cl)^(1/3) R_cl/c, as in AnimateStaticCloudTest.py.
# This lets the panel labels say t/t_diff. If you want
# t_diff = 1.5*(a_v tau_cl)^(1/3)R_cl/c, set tdiff_factor = 1.5.
tdiff_factor = 1.0

# Normalize both LYDION and analytical profiles by the final analytical peak.
# This keeps the comparison shape clean and avoids huge y-label exponents.
normalize_by_analytical_peak = False

# Plot analytical solution.
plot_analytical = True

# Line styles.
lydion_ls     = "-"
#analytical_ls = "--"
analytical_ls = "dotted"
lydion_lw     = [4.0, 4.0, 4.0, 4.0, 6.0]
analytical_lw = 6.0

# Colors. LYDION time sequence uses time_colors.
#color_LYDION = "dimgray"
color_LYDION = "dodgerblue"
color_LYDION_final = "black"
time_colors = [
    color_LYDION,
    color_LYDION,
    color_LYDION,
    color_LYDION,
    color_LYDION_final
]

analytical_color = "deeppink"

# Optional x/y limits.
xlim = (0.0, 1.1)
#ylim = (0.0, 6.5e-8)
ylim = (0.0, 6.5)

# Figure cosmetics.
figsize = (16.5*0.6, 10.5)
figure_title = r"Ly$\alpha$ energy density: LYDION vs. analytical solution"

legend_loc      = "upper right"
legend_fontsize = 37

# Text labels for LYDION times.
add_time_text_labels = True
time_text_fontsize   = 32

# Usually we only want labels for the gray pre-convergence curves,
# while the final black curve is already labelled in the legend.
label_only_nonfinal_times = True

# Manual per-curve label settings.
# Keys are curve numbers in the order given by data["k_plot"]:
#   0, 1, 2, ... correspond to increasing plotted time.
#
# coords:
#   "axes" : xy is in axes fraction, from (0,0) to (1,1).
#   "data" : xy is in data coordinates.
#
# rotation is in degrees. Negative values rotate clockwise.
#
# If a plotted curve is not listed here, the code falls back to
# automatic placement unless auto_label_unlisted_times = False.
time_label_settings = {
    0: {"xy": (0.20, 0.13), "coords": "axes", "rotation": 0.0},
    1: {"xy": (0.20, 0.27), "coords": "axes", "rotation": -2.0},
    2: {"xy": (0.20, 0.48), "coords": "axes", "rotation": -12.5},
    3: {"xy": (0.20, 0.66), "coords": "axes", "rotation": -25.0},
}

auto_label_unlisted_times = False
time_text_x_data_fraction = 0.72   # used for automatic placement

# Optional panel text.
# Main cloud-parameter label. You can add T and/or resolution directly
# under tau_cl without using separate text entries.
add_tau_text_label       = False
tau_text_xy              = (0.05, 0.93)   # axes coordinates
tau_text_fontsize        = 40
tau_text_color           = "black"
tau_text_ha              = "left"
tau_text_va              = "top"

add_temperature_to_tau_text = True
add_resolution_to_tau_text  = False
tau_text_line_spacing       = 1.05

# Optional separate resolution text. Usually keep this False if
# add_resolution_to_tau_text = True.
add_resolution_text_label = False
resolution_text_xy        = (0.05, 0.875)  # axes coordinates
resolution_text_fontsize  = 29
resolution_text_color     = "black"
resolution_text_ha        = "left"
resolution_text_va        = "top"

# Optional custom text entry. Leave custom_info_text = None to skip.
add_custom_info_text      = None 
custom_info_text          = "test"
custom_info_text_xy       = (0.05, 0.80)
custom_info_text_fontsize = 27
custom_info_text_color    = "black"

# Tick label formatting.
# For the y-axis, use labels like 2\times10^{-8} instead of 2e-08.
use_scientific_y_ticks = True
sci_tick_decimal_places = 2


###############################################################
#######                                                 #######
#######                 H E L P E R S                   #######
#######                                                 #######
###############################################################

tau_pattern    = re.compile(r"_tau([0-9eE+\-.]+)_")
res_pattern    = re.compile(r"_(\d+)x(\d+)\.h5$")
tauabs_pattern = re.compile(r"tauabs([0-9eE+\-.]+)")
pd_pattern     = re.compile(r"pd([0-9eE+\-.]+)")
umax_pattern   = re.compile(r"umax([0-9eE+\-.]+)")
T_pattern      = re.compile(r"_T([0-9eE+\-.]+)_")


def parse_float_from_pattern(pattern, filename, label):
    match = pattern.search(filename)
    if match is None:
        raise ValueError(f"Could not parse {label} from {filename}")
    return float(match.group(1))


def parse_temperature_from_filename(filename):
    match = T_pattern.search(filename)
    if match is None:
        raise ValueError(
            f"Could not parse temperature from {filename}. "
            "Expected substring like '_T10000.0_'."
        )
    return float(match.group(1))


def tau_label(tau):
    exp = np.log10(tau)
    if np.isclose(exp, np.round(exp)):
        return rf"$\tau_{{cl}}=10^{{{int(np.round(exp))}}}$"
    return rf"$\tau_{{cl}}={tau:.2g}$"


def temperature_label(T):
    """
    Clean temperature label:
      10000 -> T = 10^4 K
      100   -> T = 100 K
    """
    if T is None or not np.isfinite(T):
        return None

    exp = np.log10(T)
    if np.isclose(exp, np.round(exp), rtol=0.0, atol=1e-10):
        exp_int = int(np.round(exp))
        if exp_int >= 3:
            return rf"  $T=10^{{{exp_int}}}\,K$"

    if np.isclose(T, np.round(T), rtol=0.0, atol=1e-10):
        return rf"$T={int(np.round(T))}\,K$"

    return rf"$T={T:g}\,K$"


def make_tau_text(tau, T=None):
    lines = [tau_label(tau)]

    if add_temperature_to_tau_text:
        Tlab = temperature_label(T)
        if Tlab is not None:
            lines.append(Tlab)

    if add_resolution_to_tau_text:
        lines.append(rf"Resolution: ${resolution[0]}^2$")

    return "\n".join(lines)


def clean_tick_formatter(val, pos=None):
    if val == 0:
        return "0"

    if np.isclose(val, np.round(val), rtol=0.0, atol=1e-10):
        return f"{int(np.round(val))}"

    return f"{val:g}"


def clean_scientific_tick_formatter(val, pos=None):
    """
    Format small/large values as clean math text:
      0        -> 0
      6e-8     -> $6\\times10^{-8}$
      1e-8     -> $10^{-8}$
      2.5e-8   -> $2.5\\times10^{-8}$

    This avoids matplotlib labels like 6e-08.
    """
    if not np.isfinite(val):
        return ""

    if np.isclose(val, 0.0, rtol=0.0, atol=1e-300):
        return "0"

    sign = "-" if val < 0.0 else ""
    aval = abs(float(val))

    exp = int(np.floor(np.log10(aval)))
    mant = aval / 10.0**exp

    # Avoid labels like 10\times10^{-9}; convert to 10^{-8}.
    if np.isclose(mant, 10.0, rtol=0.0, atol=1e-10):
        mant = 1.0
        exp += 1

    # For ordinary-sized numbers, keep the simple formatter.
    if -2 <= exp <= 3:
        return clean_tick_formatter(val, pos)

    if np.isclose(mant, 1.0, rtol=0.0, atol=1e-10):
        return rf"${sign}10^{{{exp}}}$"

    mant_str = f"{mant:.{sci_tick_decimal_places}f}".rstrip("0").rstrip(".")
    return rf"${sign}{mant_str}\times10^{{{exp}}}$"


def file_matches_settings(filepath, tau_target):
    filename = filepath.name

    if require_NoRecoil and ("NoRecoil" not in filename):
        return False

    try:
        tau_file = parse_float_from_pattern(tau_pattern, filename, "tau_cl")
    except ValueError:
        return False

    if not np.isclose(tau_file, tau_target, rtol=1e-6, atol=0.0):
        return False

    match_res = res_pattern.search(filename)
    if match_res is None:
        return False

    Nr = int(match_res.group(1))
    NZ = int(match_res.group(2))

    if (Nr, NZ) != tuple(resolution):
        return False

    if require_tauabs0:
        try:
            tauabs = parse_float_from_pattern(tauabs_pattern, filename, "tauabs")
            if not np.isclose(tauabs, 0.0):
                return False
        except ValueError:
            return False

    if require_pd0:
        try:
            pd = parse_float_from_pattern(pd_pattern, filename, "pd")
            if not np.isclose(pd, 0.0):
                return False
        except ValueError:
            return False

    if require_umax0:
        try:
            umax = parse_float_from_pattern(umax_pattern, filename, "umax")
            if not np.isclose(umax, 0.0):
                return False
        except ValueError:
            return False

    return True


def find_file_for_tau(tau_target):
    candidates = [
        filepath
        for filepath in sorted(data_dir.glob("*.h5"))
        if file_matches_settings(filepath, tau_target)
    ]

    if len(candidates) == 0:
        raise RuntimeError(
            f"No matching HDF5 file found for tau_cl={tau_target:.4e}, "
            f"resolution={resolution}, data_dir={data_dir}."
        )

    if len(candidates) == 1:
        return candidates[0]

    if choose_newest_matching_file:
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = candidates[-1]
    else:
        chosen = candidates[0]

    print(f"\nMultiple files matched tau_cl={tau_target:.4e}.")
    print(f"Chosen file: {chosen.name}")
    print("Other candidates:")
    for c in candidates:
        if c != chosen:
            print(f"  {c.name}")

    return chosen


def reorder_spherical_history(arr, t):
    """Reorder e_Lya_SphericalHistory to shape (Nt, Nbins)."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D spherical history, got shape {arr.shape}")

    Nt = t.size
    s0, s1 = arr.shape

    if s0 == Nt and s1 != Nt:
        return arr
    if s1 == Nt and s0 != Nt:
        return arr.T

    raise RuntimeError(
        f"Could not identify time axis in spherical history: shape={arr.shape}, Nt={Nt}"
    )


def valid_saved_indices(t):
    """
    Return saved-time indices.

    This keeps t=0 if present, but avoids accidentally using non-finite entries.
    If your files contain preallocated trailing zeros, the final snapshot is still
    selected using nanargmax(t), not the last zero.
    """
    t = np.asarray(t, dtype=float)
    idx = np.where(np.isfinite(t) & ((t > 0.0) | (np.arange(t.size) == 0)))[0]

    if idx.size == 0:
        raise RuntimeError("No finite saved times found.")

    # If there are repeated times, keep the first occurrence except always keep the max time.
    # This is mostly cosmetic for accidental trailing/preallocated duplicates.
    keep = []
    seen = set()
    for k in idx:
        key = round(float(t[k]), 14)
        if key not in seen:
            keep.append(int(k))
            seen.add(key)

    k_final = int(np.nanargmax(t))
    if k_final not in keep:
        keep.append(k_final)

    keep = np.array(sorted(keep, key=lambda kk: float(t[kk])), dtype=int)
    return keep


def select_time_indices(t):
    saved_idx = valid_saved_indices(t)
    t_saved   = t[saved_idx]
    k_final   = int(np.nanargmax(t))
    t_final   = float(t[k_final])

    if time_selection_mode == "fractions_final":
        requested = np.asarray(time_fractions_of_final, dtype=float) * t_final
        idx = []
        for tq in requested:
            j = int(np.argmin(np.abs(t_saved - tq)))
            idx.append(int(saved_idx[j]))

    elif time_selection_mode == "manual_indices":
        idx = []
        for k in time_indices:
            kk = int(k)
            if kk < 0:
                kk = t.size + kk
            if kk < 0 or kk >= t.size:
                raise RuntimeError(f"Requested time index {k} is outside [0,{t.size-1}].")
            idx.append(kk)

    elif time_selection_mode == "manual_times":
        idx = []
        for tq in time_values:
            j = int(np.argmin(np.abs(t_saved - float(tq))))
            idx.append(int(saved_idx[j]))

    elif time_selection_mode == "all":
        idx = list(map(int, saved_idx))
        if max_number_of_times is not None and len(idx) > max_number_of_times:
            take = np.linspace(0, len(idx) - 1, int(max_number_of_times))
            idx = [idx[int(round(q))] for q in take]

    else:
        raise ValueError(
            "time_selection_mode must be 'fractions_final', 'manual_indices', "
            "'manual_times', or 'all'."
        )

    # Remove duplicates while preserving order in time.
    idx = sorted(set(idx), key=lambda kk: float(t[kk]))

    if k_final not in idx:
        idx.append(k_final)
        idx = sorted(set(idx), key=lambda kk: float(t[kk]))

    return np.array(idx, dtype=int)


def prepare_analytical_profile(f, RbinC, rCoM=None, ZCoM=None):
    if "e_Lya_Analytical" not in f:
        return None, None

    yA = np.asarray(f["e_Lya_Analytical"][:], dtype=float)
    yA = np.ravel(yA)

    if yA.size == RbinC.size:
        xA = np.asarray(RbinC, dtype=float)
    elif ZCoM is not None and yA.size == ZCoM.size:
        # This matches the convention used in AnimateStaticCloudTest.py.
        xA = np.abs(np.asarray(ZCoM, dtype=float))
    elif rCoM is not None and yA.size == rCoM.size:
        xA = np.asarray(rCoM, dtype=float)
    else:
        raise RuntimeError(
            "Could not infer analytical-profile radius array.\n"
            f"  len(e_Lya_Analytical) = {yA.size}\n"
            f"  len(R_bincenters)     = {RbinC.size}\n"
            f"  len(rCoM)             = {None if rCoM is None else rCoM.size}\n"
            f"  len(ZCoM)             = {None if ZCoM is None else ZCoM.size}\n"
        )

    good = np.isfinite(xA) & np.isfinite(yA)
    if not np.any(good):
        return None, None

    xA = xA[good]
    yA = yA[good]

    order = np.argsort(xA)
    xA = xA[order]
    yA = yA[order]

    return xA, yA


def read_profile_data(filepath):
    filename = filepath.name
    T = parse_temperature_from_filename(filename)

    with h5py.File(filepath, "r") as f:
        t = np.asarray(f["Times"][:], dtype=float)

        RbinC = np.asarray(f["R_bincenters"][:], dtype=float)
        eLya_sph_raw = np.asarray(f["e_Lya_SphericalHistory"][:], dtype=float)
        eLya_sph_hist = reorder_spherical_history(eLya_sph_raw, t)

        rCoM = np.asarray(f["rCoM"][:], dtype=float) if "rCoM" in f else None
        ZCoM = np.asarray(f["ZCoM"][:], dtype=float) if "ZCoM" in f else None

        xA, yA = prepare_analytical_profile(f, RbinC, rCoM=rCoM, ZCoM=ZCoM)

    if eLya_sph_hist.shape[1] != RbinC.size:
        raise RuntimeError(
            "Spherical history/profile radius mismatch.\n"
            f"  e_Lya_SphericalHistory reordered shape = {eLya_sph_hist.shape}\n"
            f"  len(R_bincenters) = {RbinC.size}"
        )

    k_plot = select_time_indices(t)
    k_final = int(np.nanargmax(t))
    t_final = float(t[k_final])

    return {
        "filename": filename,
        "T": T,
        "t": t,
        "RbinC": RbinC,
        "eLya_sph_hist": eLya_sph_hist,
        "xA": xA,
        "yA": yA,
        "k_plot": k_plot,
        "k_final": k_final,
        "t_final": t_final,
    }


def _time_label(t_over_tdiff):
    if t_over_tdiff < 0.095:
        return rf"$t/t_{{diff}}={t_over_tdiff:.2f}$"
    if t_over_tdiff < 9.95:
        return rf"$t/t_{{diff}}={t_over_tdiff:.1f}$"
    return rf"$t/t_{{diff}}={t_over_tdiff:.0f}$"


def setup_plot_style():
    plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 39.5})
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'GreekC'

    params = {'legend.fontsize': legend_fontsize}
    plt.rcParams.update(params)
    plt.rcParams["legend.markerscale"] = 0.75

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['xtick.major.width'] = 3.0
    plt.rcParams['xtick.minor.width'] = 3.0
    plt.rcParams['xtick.major.size'] = 20
    plt.rcParams['xtick.minor.size'] = 8
    plt.rcParams['ytick.major.width'] = 3.0
    plt.rcParams['ytick.minor.width'] = 3.0
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


def choose_normalization(data):
    if not normalize_by_analytical_peak:
        return 1.0

    yA = data["yA"]
    if yA is not None and np.any(np.isfinite(yA)):
        norm = float(np.nanmax(yA))
        if np.isfinite(norm) and norm > 0.0:
            return norm

    y_final = data["eLya_sph_hist"][data["k_final"], :]
    norm = float(np.nanmax(y_final))
    if np.isfinite(norm) and norm > 0.0:
        return norm

    return 1.0


###############################################################
#######                                                 #######
#######                 R E A D   D A T A               #######
#######                                                 #######
###############################################################

filepath = find_file_for_tau(tau_cl)
data = read_profile_data(filepath)
norm_e = choose_normalization(data)

print("\nReading LYDION Lyα energy-density profile:")
print(f"  tau_cl  = {tau_cl:.4e}")
print(f"  T       = {data['T']:.6g} K")
print(f"  file    = {data['filename']}")
print(f"  k_final = {data['k_final']}")
print(f"  t_final = {data['t_final']:.6g} in saved time units")
print("  plotted times:")
for k in data["k_plot"]:
    print(f"    k = {k:5d}, t = {data['t'][k]:.6g}, t/t_diff = {data['t'][k]/tdiff_factor:.6g}")


###############################################################
#######                                                 #######
#######            P L O T  C O S M E T I C S           #######
#######                                                 #######
###############################################################

setup_plot_style()


###############################################################
#######                                                 #######
#######                   P L O T   I T                 #######
#######                                                 #######
###############################################################

fig, ax = plt.subplots(figsize=figsize)

R_norm = data["RbinC"]

# LYDION profiles at selected times.
plotted_curves = []
for n, k in enumerate(data["k_plot"]):
    color = time_colors[n % len(time_colors)]
    #y = data["eLya_sph_hist"][k, :] / norm_e
    y = data["eLya_sph_hist"][k, :] / 1e-8
    t_over_tdiff = data["t"][k] / tdiff_factor

    line, = ax.plot(
        R_norm,
        y,
        linestyle=lydion_ls,
        linewidth=lydion_lw[n],
        color=color,
        label="LYDION (t/t$_{diff}$ = 4)" if n == len(data["k_plot"]) - 1 else "_nolegend_",
        zorder=3 + n,
    )

    plotted_curves.append((line, R_norm.copy(), y.copy(), t_over_tdiff, color))

# Add time labels by hand/automatically in the panel.
if add_time_text_labels:
    xmin_tmp, xmax_tmp = xlim if xlim is not None else (
        float(np.nanmin(R_norm)),
        float(np.nanmax(R_norm)),
    )
    x_txt_auto = xmin_tmp + time_text_x_data_fraction * (xmax_tmp - xmin_tmp)

    used_y = []
    n_final_curve = len(plotted_curves) - 1

    for n, (_, x, y, t_over_tdiff, color) in enumerate(plotted_curves):

        # Keep labels only on the gray/pre-final curves, unless requested otherwise.
        if label_only_nonfinal_times and n == n_final_curve:
            continue

        if (n not in time_label_settings) and (not auto_label_unlisted_times):
            continue

        good = np.isfinite(x) & np.isfinite(y)
        if not np.any(good):
            continue

        settings = time_label_settings.get(n, None)

        if settings is not None:
            x_txt, y_txt = settings.get(
                "xy",
                (x_txt_auto, float(np.interp(x_txt_auto, x[good], y[good]))),
            )
            coords = settings.get("coords", "axes")
            rotation = float(settings.get("rotation", 0.0))
            ha = settings.get("ha", "left")
            va = settings.get("va", "center")
            label_color = settings.get("color", color)
            alpha = settings.get("alpha", 1.0)

            transform = ax.transAxes if coords == "axes" else ax.transData

        else:
            y_txt = float(np.interp(x_txt_auto, x[good], y[good]))

            # Gentle anti-overlap in data coordinates.
            if len(used_y) > 0:
                y_span_guess = max(float(np.nanmax(y[good]) - np.nanmin(y[good])), 1e-30)
                min_sep = 0.055 * y_span_guess
                for yy in used_y:
                    if abs(y_txt - yy) < min_sep:
                        y_txt = yy + min_sep

            used_y.append(y_txt)

            x_txt = x_txt_auto
            rotation = 0.0
            ha = "left"
            va = "center"
            label_color = color
            alpha = 1.0
            transform = ax.transData

        ax.text(
            x_txt,
            y_txt,
            _time_label(t_over_tdiff),
            transform=transform,
            color=label_color,
            fontsize=time_text_fontsize,
            ha=ha,
            va=va,
            rotation=rotation,
            rotation_mode="anchor",
            alpha=alpha,
        )

# Analytical solution:
if plot_analytical and data["xA"] is not None and data["yA"] is not None:
    ax.plot(
        data["xA"],
        data["yA"] / 1e-8,
        linestyle=analytical_ls,
        linewidth=analytical_lw,
        color=analytical_color,
        dash_capstyle="round",
        label="Analytical solution",
        zorder=7,
        alpha = 1.0
    )

ax.set_xlabel(r"Spherical radius $r/R_{cl}$", labelpad=15)

if normalize_by_analytical_peak:
    ax.set_ylabel(r"$e_{Ly\alpha}/\max(e_{Ly\alpha,ana})$", labelpad=20)
else:
    ax.set_ylabel(r"$e_{Ly\alpha}$ (10$^{-8}$ erg cm$^{-3}$)", labelpad=20)

#ax.set_title(figure_title, pad=25, fontsize=41.5)

if xlim is not None:
    ax.set_xlim(*xlim)
else:
    ax.set_xlim(float(np.nanmin(R_norm)), float(np.nanmax(R_norm)))

if ylim is not None:
    ax.set_ylim(*ylim)
else:
    ax.set_ylim(bottom=0.0)

ax.xaxis.set_major_formatter(FuncFormatter(clean_tick_formatter))

if use_scientific_y_ticks:
    ax.yaxis.set_major_formatter(FuncFormatter(clean_scientific_tick_formatter))
else:
    ax.yaxis.set_major_formatter(FuncFormatter(clean_tick_formatter))

ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

if add_tau_text_label:
    ax.text(
        tau_text_xy[0],
        tau_text_xy[1],
        make_tau_text(tau_cl, data["T"]),
        transform=ax.transAxes,
        color=tau_text_color,
        fontsize=tau_text_fontsize,
        ha=tau_text_ha,
        va=tau_text_va,
        linespacing=tau_text_line_spacing,
    )

if add_resolution_text_label:
    ax.text(
        resolution_text_xy[0],
        resolution_text_xy[1],
        rf"Resolution: ${resolution[0]}^2$",
        transform=ax.transAxes,
        color=resolution_text_color,
        fontsize=resolution_text_fontsize,
        ha=resolution_text_ha,
        va=resolution_text_va,
    )

if add_custom_info_text and custom_info_text is not None:
    ax.text(
        custom_info_text_xy[0],
        custom_info_text_xy[1],
        custom_info_text,
        transform=ax.transAxes,
        color=custom_info_text_color,
        fontsize=custom_info_text_fontsize,
        ha="left",
        va="top",
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