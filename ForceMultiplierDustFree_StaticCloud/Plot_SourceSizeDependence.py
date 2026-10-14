# PlotMF_vs_rs_v2.py
# Source-size force multiplier plot.
# Defaults: tau_cl=1e8, 128^2 only, LYDION symbols only, Eq. (67) analytical fit.

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator
from pathlib import Path
import re
import math


###############################################################
#######                                                 #######
#######                 U S E R   S E T U P             #######
#######                                                 #######
###############################################################

repo_root = Path(__file__).resolve().parent

data_dir = repo_root / "LYDION_DataDustFreeStaticClouds"

save_name = "M_F_vs_rs_tau1e8.pdf"

# Fixed cloud optical depth used for this comparison.
tau_target = 1.0e8

# If None, the script infers a unique temperature from the selected filenames.
# Otherwise, set e.g. temperature = 10000.0.
temperature = None

# Choose which M_F dataset to read.
# Most likely options:
#   "M_F_radial_History"
#   "M_F_radial_net_History"
#   "M_F_History"
#
# If force_key = None, the script tries the keys below in order.
force_key = None

force_key_candidates = [
    #"M_F_radial_History",
    #"M_F_radial_net_History",
    #"M_F_Radial_History",
    "M_F_History",
]

# Only include files matching these choices.
require_NoRecoil = True
require_tauabs0  = True
require_pd0      = True
require_umax0    = True

# Optional: restrict to one or more spatial resolutions.
# Examples:
#   resolutions_to_plot = [(128, 128)]
#   resolutions_to_plot = [(256, 256)]
#   resolutions_to_plot = [(128, 128), (256, 256)]
resolutions_to_plot = [(128, 128)]

# Analytical solution from Nebrin et al. 2024, Eqs. (65)-(66),
# specialized here to a static, dust-free cloud without recoil, velocity
# gradients, or Ly-alpha destruction:
#
#     P = 0, Rdot_cl/b = 0, Dbar_n = 1, K_n,+ = K_n,-.
#
# For the present uniform-cloud tests, tau_s_bar = rs/R_cl.
plot_analytical_solution = True
analytical_npoints = 500
analytical_color = "black"
analytical_lw = 4.5

# Number of odd eigenmodes used for the analytical series.
# The source-size series is only slowly convergent near the point-source
# limit, so rs/R_cl = 0 is handled analytically below. For the smallest
# non-zero source sizes in this plot, ~1e5 terms is still cheap and gives a
# smooth curve. Lower this if you only need a quick preview.
analytical_nterms = 1000_000

# If a source size is below this value, use the exact point-source limit of
# the same series. This avoids catastrophic cancellation in
# sin(x) - x*cos(x), and avoids the slow point-source convergence.
point_source_limit = 1.0e-8

# Symbol/style choices for LYDION runs, grouped by resolution.
marker_by_resolution = {
    (32, 32): "s",
    (64, 64): "o",
    (128, 128): "*",
    (256, 256): "+",
}

color_by_resolution = {
    (32, 32): "black",
    (64, 64): "dodgerblue",
    (128, 128): "orangered",
    (256, 256): "seagreen",
}

# Optional frequency-grid tags parsed from filenames like
#
#   ..._128x128_q1.03.h5
#
# The marker is still set by the spatial resolution. Tagged frequency
# grids use a different color, while keeping the same resolution marker.
q_color_override = {
    "1.03": "mediumorchid",
}

q_label_override = {
    "1.03": r"$q=1.03$",
}

default_marker = "o"
default_color  = "black"
default_q_color = "dimgray"

marker_size = 45
marker_edge_width = 3.5

figsize = (10, 12.0)

# Linear-linear plot limits. Set to None for automatic limits.
xlim = (-0.1, 1.1)
ylim = (0.0, 4.1)

figure_title = r"Dependence on source size"


###############################################################
#######                                                 #######
#######                 R E A D   D A T A               #######
#######                                                 #######
###############################################################

rs_pattern     = re.compile(r"_rs([0-9eE+\-.]+)_")
tau_pattern    = re.compile(r"_tau([0-9eE+\-.]+)_")
res_pattern    = re.compile(r"_(\d+)x(\d+)(?:_q([^_]+))?\.h5$")
tauabs_pattern = re.compile(r"tauabs([0-9eE+\-.]+)")
pd_pattern     = re.compile(r"pd([0-9eE+\-.]+)")
umax_pattern   = re.compile(r"umax([0-9eE+\-.]+)")
T_pattern      = re.compile(r"_T([0-9eE+\-.]+)_")

# mf_data[(Nr, NZ, q_tag)] = [(rs, M_F_final, T, filename), ...]
#
# q_tag = None for the original/default frequency grid.
# q_tag = "1.03" for filenames ending in _q1.03.h5, etc.
mf_data = {}


def parse_float_from_pattern(pattern, filename, label):
    match = pattern.search(filename)
    if match is None:
        raise ValueError(f"Could not parse {label} from {filename}")
    return float(match.group(1))


def choose_force_key(h5file, filename):
    if force_key is not None:
        if force_key not in h5file:
            available = sorted(list(h5file.keys()))
            raise KeyError(
                f"{filename}: requested force_key='{force_key}' not found.\n"
                f"Available keys include:\n{available}"
            )
        return force_key

    for key in force_key_candidates:
        if key in h5file:
            return key

    available = sorted(list(h5file.keys()))
    raise KeyError(
        f"{filename}: none of the force_key_candidates were found.\n"
        f"Tried: {force_key_candidates}\n"
        f"Available keys include:\n{available}"
    )


def sort_data_key(data_key):
    Nr, NZ, q_tag = data_key

    # Plot/print default frequency grids first, then tagged grids sorted by q.
    if q_tag is None:
        q_sort = -np.inf
    else:
        try:
            q_sort = float(q_tag)
        except ValueError:
            q_sort = np.inf

    return (Nr, NZ, q_sort, "" if q_tag is None else q_tag)


for filepath in sorted(data_dir.glob("*.h5")):

    filename = filepath.name

    if require_NoRecoil and ("NoRecoil" not in filename):
        continue

    try:
        rs = parse_float_from_pattern(rs_pattern, filename, "rs")
    except ValueError as err:
        print(f"Skipping {filename}: {err}")
        continue

    try:
        tau_cl = parse_float_from_pattern(tau_pattern, filename, "tau_cl")
    except ValueError as err:
        print(f"Skipping {filename}: {err}")
        continue

    if not np.isclose(tau_cl, tau_target, rtol=1.0e-8, atol=0.0):
        continue

    match_res = res_pattern.search(filename)
    if match_res is None:
        print(f"Skipping {filename}: could not parse resolution.")
        continue

    Nr = int(match_res.group(1))
    NZ = int(match_res.group(2))
    q_tag = match_res.group(3)

    if resolutions_to_plot is not None:
        if (Nr, NZ) not in resolutions_to_plot:
            continue

    if require_tauabs0:
        try:
            tauabs = parse_float_from_pattern(tauabs_pattern, filename, "tauabs")
            if not np.isclose(tauabs, 0.0):
                continue
        except ValueError:
            print(f"Skipping {filename}: could not parse tauabs.")
            continue

    if require_pd0:
        try:
            pd = parse_float_from_pattern(pd_pattern, filename, "pd")
            if not np.isclose(pd, 0.0):
                continue
        except ValueError:
            print(f"Skipping {filename}: could not parse pd.")
            continue

    if require_umax0:
        try:
            umax = parse_float_from_pattern(umax_pattern, filename, "umax")
            if not np.isclose(umax, 0.0):
                continue
        except ValueError:
            print(f"Skipping {filename}: could not parse umax.")
            continue

    try:
        T_file = parse_float_from_pattern(T_pattern, filename, "T")
    except ValueError:
        T_file = np.nan

    with h5py.File(filepath, "r") as f:

        key = choose_force_key(f, filename)

        mf_hist = np.array(f[key][:], dtype=float)

        good = np.isfinite(mf_hist)
        if not np.any(good):
            print(f"Skipping {filename}: no finite values in {key}.")
            continue

        mf_final = mf_hist[good][-1]

    data_key = (Nr, NZ, q_tag)

    if data_key not in mf_data:
        mf_data[data_key] = []

    mf_data[data_key].append((rs, mf_final, T_file, filename))


for data_key in mf_data:
    mf_data[data_key] = sorted(mf_data[data_key], key=lambda t: t[0])

if len(mf_data) == 0:
    raise RuntimeError(
        f"No matching HDF5 files found in: {data_dir}\n"
        f"Searched for tau_cl = {tau_target:.6e}."
    )


###############################################################
#######                                                 #######
#######          N O R M A L I Z A T I O N              #######
#######                                                 #######
###############################################################

if temperature is None:
    T_values = []

    for data_key in mf_data:
        for _, _, T_file, _ in mf_data[data_key]:
            if np.isfinite(T_file):
                T_values.append(T_file)

    T_values = np.array(T_values, dtype=float)

    if len(T_values) == 0:
        raise RuntimeError(
            "Could not infer temperature from filenames. "
            "Set temperature manually in USER SETUP."
        )

    T_unique = np.unique(np.round(T_values, decimals=8))

    if len(T_unique) != 1:
        raise RuntimeError(
            "Found more than one temperature in the selected files:\n"
            f"{T_unique}\n"
            "Set temperature manually, or split the plot by T."
        )

    temperature_used = float(T_unique[0])
else:
    temperature_used = float(temperature)

T4 = temperature_used / 1.0e4
a_v = 4.7e-4 * T4**(-0.5)
MF_norm = (a_v * tau_target)**(1.0 / 3.0)

print("\nRead LYDION source-size static-cloud data:")
print(f"  tau_cl = {tau_target:.6e}")
print(f"  T      = {temperature_used:.6g} K")
print(f"  a_v    = {a_v:.6e}")
print(f"  norm   = (a_v*tau_cl)^(1/3) = {MF_norm:.6e}")

for data_key in sorted(mf_data, key=sort_data_key):
    Nr, NZ, q_tag = data_key
    q_text = "default ν-grid" if q_tag is None else f"q={q_tag}"
    print(f"  {Nr}x{NZ}, {q_text}: {len(mf_data[data_key])} runs")
    for rs, mf_final, _, filename in mf_data[data_key]:
        print(
            f"    rs/R_cl = {rs:.6g}, "
            f"M_F_final = {mf_final:.6e}, "
            f"M_F/norm = {mf_final/MF_norm:.6e}, "
            f"file = {filename}"
        )


###############################################################
#######                                                 #######
#######   A N A L Y T I C A L   S E R I E S             #######
#######                                                 #######
###############################################################

def analytical_MF_scaled_series(rs_over_Rcl, nterms=analytical_nterms):
    """
    Nebrin et al. 2024 Eqs. (65)-(66), normalized by
    (a_v*tau_cl)^(1/3), specialized to the present test:

        P = 0, Rdot_cl/b = 0, no recoil, no continuum absorption.

    With Rdot_cl/b = 0, Eq. (19) gives Dbar_n = 1 and Eq. (66) gives

        K_n,+ + K_n,- = 2 * [sqrt(2*pi/3) * n*pi/3]^(-1/3).

    Only odd modes contribute to the spatial integral in Eq. (64),
    so n_o = 2*n + 1 below. For a uniform cloud, tau_s_bar = rs/R_cl.
    """

    tau_s_bar = np.asarray(rs_over_Rcl, dtype=float)

    if np.any((tau_s_bar < 0.0) | (tau_s_bar > 1.0)):
        raise ValueError(
            "The Eqs. (65)-(66) source-size solution should only be used "
            "for 0 <= rs/R_cl <= 1."
        )

    scalar_input = (tau_s_bar.ndim == 0)
    tau_s_bar = np.atleast_1d(tau_s_bar)

    # Odd spatial eigenmodes: n_o = 1, 3, 5, ...
    n_o = 2.0*np.arange(nterms, dtype=float) + 1.0
    npi = np.pi*n_o

    prefactor = (
        4.0*np.sqrt(6.0)*math.gamma(4.0/3.0)
        * 2.0*(np.sqrt(2.0*np.pi/3.0)/3.0)**(-1.0/3.0)
    )

    MF_scaled = np.empty_like(tau_s_bar, dtype=float)

    # Exact point-source limit of the same odd-mode series.
    # Numerically this is
    #   prefactor/3 * pi^(-4/3) * sum_odd n^(-4/3)
    # = 3.510676086..., i.e. the familiar point-source coefficient.
    point_source_value = 3.5106760863748288

    point_mask = tau_s_bar <= point_source_limit
    MF_scaled[point_mask] = point_source_value

    finite_mask = ~point_mask

    if np.any(finite_mask):
        tau = tau_s_bar[finite_mask]

        # Source-size factor from Eq. (65):
        #   [sin(n*pi*tau_s_bar) - n*pi*tau_s_bar*cos(n*pi*tau_s_bar)]
        #   / tau_s_bar^3.
        #
        # The broadcasting shape is (nterms, number_of_source_sizes).
        x = npi[:, None]*tau[None, :]
        source_factor = (np.sin(x) - x*np.cos(x)) / tau[None, :]**3

        series_terms = source_factor / npi[:, None]**(13.0/3.0)
        MF_scaled[finite_mask] = prefactor*np.sum(series_terms, axis=0)

    if scalar_input:
        return float(MF_scaled[0])

    return MF_scaled


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

params = {'legend.fontsize': 35}
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
#######                   P L O T   I T !               #######
#######                                                 #######
###############################################################

fig, ax = plt.subplots(figsize=figsize)

# Analytical series from Nebrin et al. 2024 Eqs. (65)-(66).
if plot_analytical_solution:
    rs_ana = np.linspace(0.0, 1.0, analytical_npoints)
    MF_scaled_ana = analytical_MF_scaled_series(rs_ana)

    ax.plot(
        rs_ana,
        MF_scaled_ana,
        "-",
        color=analytical_color,
        lw=analytical_lw,
        label="Analytical",
        zorder=1,
    )

# LYDION predictions.
for data_key in sorted(mf_data, key=sort_data_key):

    Nr, NZ, q_tag = data_key
    resolution_key = (Nr, NZ)
    pts = mf_data[data_key]

    rsvals = np.array([p[0] for p in pts], dtype=float)
    mfvals = np.array([p[1] for p in pts], dtype=float)
    mfvals_scaled = mfvals / MF_norm

    marker = marker_by_resolution.get(resolution_key, default_marker)

    if q_tag is None:
        color = color_by_resolution.get(resolution_key, default_color)
    else:
        color = q_color_override.get(q_tag, default_q_color)

    if Nr == NZ:
        label = rf"LYDION ${Nr}^2$"
    else:
        label = rf"LYDION ${Nr}\times{NZ}$"

    if q_tag is not None:
        q_label = q_label_override.get(q_tag, rf"$q={q_tag}$")
        label = label + ", " + q_label

    ax.plot(
        rsvals,
        mfvals_scaled,
        linestyle="None",
        marker=marker,
        ms=marker_size,
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=marker_edge_width,
        alpha=0.9,
        label=label,
        zorder=3,
    )


######################

ax.set_xlabel(r"Relative source size $r_s/R_{cl}$", labelpad=15, fontsize=40.0)
ax.set_ylabel(r"$M_F/(a_v\tau_{cl})^{1/3}$", labelpad=15, fontsize=40.0)
ax.set_title(figure_title, pad=20, fontsize=45)

ax.text(
    0.2,
    0.94,
    rf"Dust-free static cloud," + "\n" + r"$\tau_{{cl}}=10^8$, T = $10^4$ K.",
    transform=ax.transAxes,
    fontsize=38,
    ha="left",
    va="top",
)

# Clean linear tick formatter.
def linear_tick_formatter(val, pos=None):
    if np.isclose(val, 0.0):
        return "0"

    absval = abs(val)

    if absval < 1.0:
        return f"{val:.2g}"
    elif absval < 10.0:
        return f"{val:.3g}"
    else:
        return f"{val:.0f}"


ax.xaxis.set_major_formatter(FuncFormatter(linear_tick_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(linear_tick_formatter))

# Keep enough minor ticks for a linear-linear plot.
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(10))

# Optional: uncomment these if you want fixed major tick spacing.
# ax.xaxis.set_major_locator(MultipleLocator(0.1))
# ax.yaxis.set_major_locator(MultipleLocator(0.1))

if xlim is not None:
    ax.set_xlim(*xlim)
else:
    all_rs = []
    for data_key in mf_data:
        all_rs.extend([p[0] for p in mf_data[data_key]])
    all_rs = np.array(all_rs, dtype=float)

    dx = np.nanmax(all_rs) - np.nanmin(all_rs)
    pad = 0.06 * dx if dx > 0.0 else 0.05 * max(np.nanmax(np.abs(all_rs)), 1.0)
    ax.set_xlim(np.nanmin(all_rs) - pad, np.nanmax(all_rs) + pad)

if ylim is not None:
    ax.set_ylim(*ylim)
else:
    all_y = []
    for data_key in mf_data:
        all_y.extend([p[1] / MF_norm for p in mf_data[data_key]])

    if plot_analytical_solution:
        all_y.extend(analytical_MF_scaled_series(np.linspace(0.0, 1.0, analytical_npoints)))

    all_y = np.array(all_y, dtype=float)

    dy = np.nanmax(all_y) - np.nanmin(all_y)
    pad = 0.10 * dy if dy > 0.0 else 0.05 * max(np.nanmax(np.abs(all_y)), 1.0)
    ax.set_ylim(np.nanmin(all_y) - pad, np.nanmax(all_y) + pad)

ax.legend(
    loc="upper right",
    bbox_to_anchor=(0.95, 0.8),
    ncol=1,
    columnspacing=0.3,
    labelspacing=0.5,
)

plt.tight_layout()
plt.savefig(repo_root / save_name, dpi=300, bbox_inches="tight")
plt.show()

