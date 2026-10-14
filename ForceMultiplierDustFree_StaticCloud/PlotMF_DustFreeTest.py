import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter, FixedLocator
from pathlib import Path
import re

# Optional scipy backend for Dawson integral only.
# If scipy is unavailable, the code uses a pure-numpy RK/asymptotic Dawson evaluator.
try:
    from scipy.special import dawsn
    SCIPY_DAWSN_AVAILABLE = True
except Exception:
    SCIPY_DAWSN_AVAILABLE = False
    print("\nWARNING: scipy not available.")
    print("         Using pure-numpy Dawson integral for the Smith+ Voigt approximation.\n")


###############################################################
#######                                                 #######
#######                 U S E R   S E T U P             #######
#######                                                 #######
###############################################################

repo_root = Path(__file__).resolve().parent

data_dir = repo_root / "LYDION_DataDustFreeStaticClouds"

save_name = "M_F_DustFree_StaticCloud.pdf"

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

# Analytical Tomaselli & Ferrara uniform-source solution:
plot_analytical_solution = True

# Optional visual reference only. This is NOT used to compute the full solution.
plot_asymptotic_solution = False

# Temperatures for the analytical solution curves.
# Set to a list of one or more values, e.g. [10000.0] or [100.0, 10000.0].
# If None, a single temperature is inferred automatically from the filenames.
analytical_temperatures = None

# One color per entry in analytical_temperatures.
# If None, a default palette is used.  Must match the length of the list
# (ignored when analytical_temperatures is None).
analytical_colors = None

asymptotic_color = "gray"

# Default color palette used when analytical_colors is None.
_default_analytical_palette = [
    "black", "crimson"
]

# Accuracy/runtime controls for the analytical integral.
n_tau_ana = 250
n_x_grid  = 6000
n_s_grid  = 3500
s_cut     = 60.0

# Polylog series truncation.
# Higher = more accurate near s=0. 800 is good for plotting.
polylog_kmax = 800

# Symbol/style choices for LYDION runs, grouped by resolution.
marker_by_resolution = {
    (32, 32): "s",
    (64, 64): "o",
    (128, 128): "*",
    (256, 256): "+"
}

color_by_resolution = {
    (32, 32): "black",
    (64, 64): "dodgerblue",
    (128, 128): "orangered",
    (256, 256): "seagreen"
}

# Optional frequency-grid tags parsed from filenames like
#
#   ..._128x128_q1.03.h5
#
# The marker is still set by the spatial resolution. Tagged frequency
# grids use a different color, so e.g. the q1.03 128^2 run uses the
# same "*" marker as the standard 128^2 run, but with a different color.
q_color_override = {
    "1.03": "mediumorchid",
}

q_label_override = {
    "1.03": r"$q=1.03$",
}

default_marker = "o"
default_color  = "black"
default_q_color = "dimgray"

marker_size = 24
marker_edge_width = 3.5

figsize = (1.11*16.5, 13.5)

xlim = (0.03, 3e10)
ylim = (0.005, 500.0)

# Deviation panel settings.
# Deviations are computed relative to the discrete MCRT values:
#
#     100 * (M_F_model - M_F_MCRT) / M_F_MCRT.
#
# For LYDION, only tau-values matching the MCRT tau grid are plotted in
# the deviation panel. For the analytical solution, the continuous analytical
# curve is interpolated to the MCRT tau grid.
plot_deviation_panel = True
deviation_ylim = (-80.0, 60.0)
deviation_major_ticks = np.arange(-80.0, 81.0, 40.0)
deviation_minor_ticks = np.arange(-80.0, 81.0, 20.0)

figure_title = r"Ly$\alpha$ force multiplier: LYDION vs. MCRT & analytical solution"


###############################################################
#######                                                 #######
#######                 R E A D   D A T A               #######
#######                                                 #######
###############################################################

tau_pattern    = re.compile(r"_tau([0-9eE+\-.]+)_")
# Accept both the original filenames
#
#   ..._128x128.h5
#
# and optional frequency-grid tags
#
#   ..._128x128_q1.03.h5
#
# The q tag is kept as a string to avoid ugly floating-point labels.
res_pattern    = re.compile(r"_(\d+)x(\d+)(?:_q([^_]+))?\.h5$")
tauabs_pattern = re.compile(r"tauabs([0-9eE+\-.]+)")
pd_pattern     = re.compile(r"pd([0-9eE+\-.]+)")
umax_pattern   = re.compile(r"umax([0-9eE+\-.]+)")
T_pattern      = re.compile(r"_T([0-9eE+\-.]+)_")

# mf_data[(Nr, NZ, q_tag)] = [(tau_cl, M_F_final, filename), ...]
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


for filepath in sorted(data_dir.glob("*.h5")):

    filename = filepath.name

    if require_NoRecoil and ("NoRecoil" not in filename):
        continue

    try:
        tau_cl = parse_float_from_pattern(tau_pattern, filename, "tau_cl")
    except ValueError as err:
        print(f"Skipping {filename}: {err}")
        continue

    match_res = res_pattern.search(filename)
    if match_res is None:
        print(f"Skipping {filename}: could not parse resolution.")
        continue

    Nr = int(match_res.group(1))
    NZ = int(match_res.group(2))
    q_tag = match_res.group(3)

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

    mf_data[data_key].append((tau_cl, mf_final, filename))


for data_key in mf_data:
    mf_data[data_key] = sorted(mf_data[data_key], key=lambda t: t[0])

if len(mf_data) == 0:
    raise RuntimeError(f"No matching HDF5 files found in: {data_dir}")

print("\nRead LYDION dust-free static-cloud data:")
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


for data_key in sorted(mf_data, key=sort_data_key):
    Nr, NZ, q_tag = data_key
    q_text = "default ν-grid" if q_tag is None else f"q={q_tag}"
    print(f"  {Nr}x{NZ}, {q_text}: {len(mf_data[data_key])} runs")
    for tau_cl, mf_final, filename in mf_data[data_key]:
        print(f"    tau_cl = {tau_cl:.4e}, M_F_final = {mf_final:.6e}, file = {filename}")


###############################################################
#######                                                 #######
#######        A N A L Y T I C A L   S O L U T I O N    #######
#######                                                 #######
###############################################################

def infer_unique_temperature_from_filenames(mf_data):
    T_values = []

    for data_key in mf_data:
        for _, _, filename in mf_data[data_key]:
            match_T = T_pattern.search(filename)
            if match_T is not None:
                T_values.append(float(match_T.group(1)))

    T_values = np.array(T_values, dtype=float)

    if len(T_values) == 0:
        raise RuntimeError(
            "Could not infer temperature from filenames. "
            "Set analytical_temperature manually."
        )

    T_unique = np.unique(np.round(T_values, decimals=8))

    if len(T_unique) != 1:
        raise RuntimeError(
            "Found more than one temperature in the selected files:\n"
            f"{T_unique}\n"
            "Set analytical_temperature manually, or split the plot by T."
        )

    return float(T_unique[0])


def trapz_integral(y, x):
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)


def cumulative_trapezoid_numpy(y, x):
    out = np.zeros_like(y)
    dx = np.diff(x)
    avg = 0.5 * (y[1:] + y[:-1])
    out[1:] = np.cumsum(avg * dx)
    return out


def dawson_asymptotic(x):
    """
    Large-x asymptotic Dawson integral:

        F(x) ~ 1/(2x) + 1/(4x^3) + 3/(8x^5) + ...

    Accurate enough for x >= ~8.
    """

    x = np.asarray(x, dtype=float)
    invx = 1.0 / x
    invx2 = invx * invx

    F = 0.5 * invx
    F += 0.25 * invx * invx2
    F += 0.375 * invx * invx2**2
    F += 0.9375 * invx * invx2**3
    F += 3.28125 * invx * invx2**4
    F += 14.765625 * invx * invx2**5

    return F


def dawson_numpy(x_grid, x_switch=8.0, hmax=0.005):
    """
    Pure-numpy Dawson integral evaluator.

    For x <= x_switch, integrates the stable ODE

        F'(x) = 1 - 2xF

    using RK4 along the sorted grid.

    For x > x_switch, uses the large-x asymptotic expansion.
    """

    x_grid = np.asarray(x_grid, dtype=float)

    if np.any(x_grid < 0):
        raise ValueError("dawson_numpy expects non-negative x_grid.")

    F_grid = np.empty_like(x_grid)

    x_current = 0.0
    F_current = 0.0

    def rhs(x, F):
        return 1.0 - 2.0 * x * F

    for i, x_target in enumerate(x_grid):

        if x_target == 0.0:
            F_grid[i] = 0.0
            continue

        if x_target > x_switch:
            F_grid[i] = dawson_asymptotic(x_target)
            continue

        while x_current < x_target:
            h = min(hmax, x_target - x_current)

            k1 = rhs(x_current, F_current)
            k2 = rhs(x_current + 0.5*h, F_current + 0.5*h*k1)
            k3 = rhs(x_current + 0.5*h, F_current + 0.5*h*k2)
            k4 = rhs(x_current + h, F_current + h*k3)

            F_current += (h/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            x_current += h

        F_grid[i] = F_current

    return F_grid


def dawson_F(x_grid):
    """
    Dawson integral F(x).

    Uses scipy.special.dawsn if available. Otherwise uses the pure-numpy
    ODE/asymptotic evaluator above.
    """

    x_grid = np.asarray(x_grid, dtype=float)

    if SCIPY_DAWSN_AVAILABLE:
        return dawsn(x_grid)

    return dawson_numpy(x_grid)


def voigt_H_Smith2015(x_grid, a_v):
    """
    Smith et al. / COLT-style second-order-in-a_v approximation to the
    Hjerting-Voigt function:

        H(a,x) ≈ exp(-x^2)
                 + 2a/sqrt(pi) * (2xF(x) - 1)
                 + a^2 exp(-x^2) * (1 - 2x^2),

    where F(x) is the Dawson integral.

    This is not the core/wing piecewise limit.
    """

    x = np.asarray(x_grid, dtype=float)
    x2 = x*x

    exp_mx2 = np.exp(-x2)
    F = dawson_F(x)

    H = (
        exp_mx2
        + (2.0*a_v/np.sqrt(np.pi)) * (2.0*x*F - 1.0)
        + a_v*a_v * exp_mx2 * (1.0 - 2.0*x2)
    )

    # Very small negative values can occur only from approximation/roundoff.
    # They are unphysical and would break sigma(x).
    H = np.maximum(H, 1e-300)

    return H


def polylog3_combination_grid(s_grid, kmax=800):
    """
    Computes

        g(s) = Li_3(e^-s) - (1/8) Li_3(e^-2s)

    with the series

        Li_3(z) = sum_{k=1}^∞ z^k/k^3.

    This avoids relying on scipy or mpmath for polylog.
    """

    s_grid = np.asarray(s_grid, dtype=float)

    k = np.arange(1, kmax + 1, dtype=float)
    inv_k3 = 1.0 / k**3

    exp1 = np.exp(-s_grid[:, None] * k[None, :])
    exp2 = np.exp(-2.0 * s_grid[:, None] * k[None, :])

    g_grid = np.sum((exp1 - 0.125 * exp2) * inv_k3[None, :], axis=1)

    # Exact s = 0 limit:
    # Li_3(1) - 1/8 Li_3(1) = 7/8 zeta(3).
    zeta3 = 1.202056903159594
    g_grid[s_grid == 0.0] = 0.875 * zeta3

    return g_grid


def compute_uniform_source_TF21_solution(
    tau_values,
    T,
    n_x_grid=6000,
    n_s_grid=3500,
    s_cut=60.0,
    polylog_kmax=800
):
    """
    Tomaselli & Ferrara 2021 uniform-source, dust-free force multiplier.

    Computes the standard-boundary-condition expression:

        M_F = (12/pi^3) sqrt(2/3) ∫_{-∞}^{∞}
              [Li_3(exp(-pi |sigma(x)|/tau))
               - (1/8) Li_3(exp(-2 pi |sigma(x)|/tau))] dx

    with sigma(x) computed from the Smith+ / COLT second-order Voigt
    approximation:

        d sigma / dx = 1 / sqrt((3/2) H(a_v, x)).

    This is not the asymptotic wing-limit solution.
    """

    tau_values = np.asarray(tau_values, dtype=float)

    T4 = T / 1.0e4
    a_v = 4.7e-4 * T4**(-0.5)

    tau_max = np.max(tau_values)

    # Use the wing scaling only to estimate a sufficiently large x_max.
    # The actual sigma(x) below is computed from H(a,x), not from the wing formula.
    x_max_estimate = (
        (s_cut * tau_max / np.pi)
        * (3.0 * a_v / np.sqrt(2.0 * np.pi / 3.0))
    )**(1.0 / 3.0)

    x_max = max(30.0, 1.25 * x_max_estimate)

    print(f"\nFull TF21 analytical solution with Smith+ Voigt approximation:")
    print(f"  T      = {T:.6g} K")
    print(f"  a_v    = {a_v:.6e}")
    print(f"  x_max  = {x_max:.6g}")
    print(f"  scipy Dawson = {SCIPY_DAWSN_AVAILABLE}")

    # Mixed x-grid: linear near core + geometric in the wing.
    x_linear_max = min(30.0, 0.25 * x_max)

    n_linear = max(1000, n_x_grid // 2)
    n_wing   = max(1000, n_x_grid // 2)

    x_grid_linear = np.linspace(0.0, x_linear_max, n_linear)

    if x_max > x_linear_max:
        x_grid_wing = np.geomspace(
            max(x_linear_max, 1e-6),
            x_max,
            n_wing
        )
        x_grid = np.unique(np.concatenate([x_grid_linear, x_grid_wing]))
    else:
        x_grid = x_grid_linear

    # Smith+ approximate Voigt and sigma(x).
    H_grid = voigt_H_Smith2015(x_grid, a_v)

    dsigma_dx = 1.0 / (np.sqrt(1.5) * H_grid)
    sigma_grid = cumulative_trapezoid_numpy(dsigma_dx, x_grid)

    # Precompute g(s) = Li_3(e^-s) - 1/8 Li_3(e^-2s).
    s_grid_core = np.linspace(0.0, 1.0, max(800, n_s_grid // 3))
    s_grid_wing = np.linspace(1.0, s_cut, n_s_grid)
    s_grid = np.unique(np.concatenate([s_grid_core, s_grid_wing]))

    print("Precomputing Li_3 series grid...")
    g_grid = polylog3_combination_grid(s_grid, kmax=polylog_kmax)

    MF_values = np.empty_like(tau_values, dtype=float)

    prefactor = (12.0 / np.pi**3) * np.sqrt(2.0 / 3.0)

    print("Integrating full analytical solution...")
    for i, tau in enumerate(tau_values):
        s_of_x = np.pi * sigma_grid / tau

        # For s > s_cut, contribution is negligible.
        integrand_pos = np.interp(
            s_of_x,
            s_grid,
            g_grid,
            left=g_grid[0],
            right=0.0
        )

        # Integrand is even in x.
        integral = 2.0 * trapz_integral(integrand_pos, x_grid)

        MF_values[i] = prefactor * integral

    return MF_values, a_v


if plot_analytical_solution or plot_asymptotic_solution:

    # Resolve the temperature list.
    if analytical_temperatures is None:
        _T_inferred = infer_unique_temperature_from_filenames(mf_data)
        _T_list = [_T_inferred]
    else:
        _T_list = list(analytical_temperatures)

    # Resolve the color list.
    if analytical_colors is not None:
        if len(analytical_colors) != len(_T_list):
            raise ValueError(
                f"analytical_colors has {len(analytical_colors)} entries "
                f"but analytical_temperatures has {len(_T_list)}."
            )
        _color_list = list(analytical_colors)
    else:
        _color_list = [
            _default_analytical_palette[i % len(_default_analytical_palette)]
            for i in range(len(_T_list))
        ]

    tau_ana = np.logspace(
        np.log10(xlim[0]),
        np.log10(xlim[1]),
        n_tau_ana
    )

    # ana_results[i] = (T, color, MF_ana_or_None, MF_asymptotic)
    ana_results = []

    for _T, _color in zip(_T_list, _color_list):

        print(f"\nAnalytical comparison temperature: T = {_T:.6g} K")

        _T4    = _T / 1.0e4
        _a_v   = 4.7e-4 * _T4**(-0.5)
        _MF_asymptotic = 0.51 * (_a_v * tau_ana)**(1.0 / 3.0)

        if plot_analytical_solution:
            _MF_ana, _ = compute_uniform_source_TF21_solution(
                tau_ana,
                _T,
                n_x_grid=n_x_grid,
                n_s_grid=n_s_grid,
                s_cut=s_cut,
                polylog_kmax=polylog_kmax
            )
        else:
            _MF_ana = None

        ana_results.append((_T, _color, _MF_ana, _MF_asymptotic))


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

params = {'legend.fontsize': 31}
plt.rcParams.update(params)
plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['xtick.minor.width'] = 2.0
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['ytick.minor.width'] = 2.0
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['ytick.minor.size'] = 8
plt.rcParams['xtick.major.pad'] = '15'
plt.rcParams['ytick.major.pad'] = '15'

plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linewidth'] = 2.5
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

if plot_deviation_panel:
    fig, (ax, ax_dev) = plt.subplots(
        2, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1.35], "hspace": 0.03},
    )
else:
    fig, ax = plt.subplots(figsize=figsize)
    ax_dev = None

#### N_HI references ################

plot_NHI = True

if plot_NHI:
    sigma0  = 5.88e-14
    tau_N20 = sigma0 * 1e20
    tau_N21 = sigma0 * 1e21 
    tau_N22 = sigma0 * 1e22 
    tau_N23 = sigma0 * 1e23

    ax.text( 0.49 * tau_N20, 15.0, r'10$^{20}$ cm$^{-2}$', rotation = 90, 
            fontsize = 27, color = 'dimgray' )
    ax.axvline( tau_N20, 0.25, 1.0, linestyle = '-', 
                lw = 2.0, color = 'dimgray', zorder = 1 )

    ax.text( 0.47 * tau_N21, 0.11, r'N$_{HI}$ = 10$^{21}$ cm$^{-2}$', rotation = 90, 
            fontsize = 27, color = 'dimgray' )
    ax.axvline( tau_N21, 0.25, 1.0, linestyle = '-', 
                lw = 2.0, color = 'dimgray', zorder = 1 )
    
    ax.text( 0.47 * tau_N22, 0.11, r'N$_{HI}$ = 10$^{22}$ cm$^{-2}$', rotation = 90, 
            fontsize = 27, color = 'dimgray' )
    ax.axvline( tau_N22, 0.25, 1.0, linestyle = '-', 
                lw = 2.0, color = 'dimgray', zorder = 1 )
    
    ax.text( 0.47 * tau_N23, 0.11, r'N$_{HI}$ = 10$^{23}$ cm$^{-2}$', rotation = 90, 
            fontsize = 27, color = 'dimgray' )
    ax.axvline( tau_N23, 0.25, 1.0, linestyle = '-', 
                lw = 2.0, color = 'dimgray', zorder = 1 )

#### MCRT DATA ######################

taucl_MCRT = np.array([0.1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
M_F_MCRT   = np.array([0.0175, 0.161, 1.0, 2.75, 4.75, 5.74, 4.34, 4.57, 8.92, 18.3])


def percent_deviation(model_values, reference_values):
    """Return 100 * (model - reference) / reference."""

    model_values = np.asarray(model_values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)

    return 100.0 * (model_values - reference_values) / reference_values


def mcrt_reference_at_matching_tau(tau_values, rtol=1e-9, atol=0.0):
    """
    Return MCRT values for tau-values that are explicitly present in the
    MCRT table. Non-matching tau-values are returned as NaN.

    This avoids silently interpolating sparse MCRT data for the LYDION points.
    """

    tau_values = np.asarray(tau_values, dtype=float)
    mcrt_values = np.full_like(tau_values, np.nan, dtype=float)

    for i, tau in enumerate(tau_values):
        matches = np.where(np.isclose(taucl_MCRT, tau, rtol=rtol, atol=atol))[0]
        if len(matches) > 0:
            mcrt_values[i] = M_F_MCRT[matches[0]]

    return mcrt_values


def interpolate_loglog(x_new, x_data, y_data):
    """
    Log-log interpolation for positive data. Used only for evaluating the
    continuous analytical curve at the MCRT tau-values.
    """

    x_new = np.asarray(x_new, dtype=float)
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    return np.exp(
        np.interp(
            np.log(x_new),
            np.log(x_data),
            np.log(y_data),
        )
    )

# Plot M_F for asymptotic wing limit:
tau_cl_asymptotic = np.logspace(3, 11, 200)
a_v = 4.72e-4
MF_asymptotic = 0.51 * (a_v * tau_cl_asymptotic) ** (1/3)
ax.plot( tau_cl_asymptotic, MF_asymptotic, linestyle = 'dashed', dash_capstyle = 'round', 
         color = 'black', lw = 4.0 )

ax.text(1.3e3, 0.125, r"$M_{F} = 0.51 (a_v \tau_{cl})^{1/3}$," + "\n" + r"asymptotic wing limit", 
        fontsize=30, rotation=21.5, ha='left', va='bottom')

# Plot M_F for low-tau limit:
tau_cl_low = np.logspace(-2, -0.2, 200)
MF_lowtau = tau_cl_low / (4 * np.sqrt(2))
ax.plot( tau_cl_low, MF_lowtau, linestyle = 'dashed', dash_capstyle = 'round', 
         color = 'black', lw = 4.0 )

ax.text(0.155, 0.0075, r"$M_{F} = 0.177 \, \tau_{cl}$", 
        fontsize=30, rotation=51, ha='left', va='bottom')

# Analytical predictions:
if plot_analytical_solution or plot_asymptotic_solution:
    for _T, _color, _MF_ana, _MF_asymptotic in ana_results:

        # Label carries the temperature only when there are multiple curves.
        _multi = len(ana_results) > 1
        _T_label = rf"$T={_T:.4g}$ K" if _multi else ""

        if plot_analytical_solution and (_MF_ana is not None):
            _ana_label = (
                rf"Analytical, {_T_label}"
                if _multi
                else r"Analytical"
            )
            ax.plot(
                tau_ana,
                _MF_ana,
                "-",
                color=_color,
                lw=4.5,
                label=_ana_label,
            )

            if plot_deviation_panel:
                _MF_ana_at_MCRT = interpolate_loglog(taucl_MCRT, tau_ana, _MF_ana)
                _dev_ana = percent_deviation(_MF_ana_at_MCRT, M_F_MCRT)

                ax_dev.plot(
                    taucl_MCRT,
                    _dev_ana,
                    "-",
                    color=_color,
                    lw=4.5,
                    alpha=0.9,
                )

        if plot_asymptotic_solution:
            _asym_label = (
                rf"$0.51(a_v\tau_{{cl}})^{{1/3}}$, {_T_label}"
                if _multi
                else r"$0.51(a_v\tau_{cl})^{1/3}$"
            )
            ax.plot(
                tau_ana,
                _MF_asymptotic,
                "--",
                color=_color,
                lw=4.0,
                dash_capstyle="round",
                label=_asym_label,
            )

# LYDION predictions:
for data_key in sorted(mf_data, key=sort_data_key):

    Nr, NZ, q_tag = data_key
    resolution_key = (Nr, NZ)
    pts = mf_data[data_key]

    tauvals = np.array([p[0] for p in pts])
    mfvals  = np.array([p[1] for p in pts])

    # Marker shows spatial resolution.
    marker = marker_by_resolution.get(resolution_key, default_marker)

    # Default runs keep the old color-by-resolution scheme. Tagged q-runs
    # override the color, while keeping the same resolution marker.
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
        tauvals,
        mfvals,
        linestyle="None",
        marker=marker,
        ms=marker_size,
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=marker_edge_width,
        alpha=0.85,
        label=label,
    )

    if plot_deviation_panel:
        mf_mcrt_match = mcrt_reference_at_matching_tau(tauvals)
        good_dev = np.isfinite(mf_mcrt_match) & np.isfinite(mfvals)

        if np.any(good_dev):
            dev_lydion = percent_deviation(mfvals[good_dev], mf_mcrt_match[good_dev])

            ax_dev.plot(
                tauvals[good_dev],
                dev_lydion,
                linestyle="None",
                marker=marker,
                ms=marker_size,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=marker_edge_width,
                alpha=0.85,
            )

# MCRT comparison points in the top panel:
ax.plot( taucl_MCRT, M_F_MCRT, linestyle = "None", marker = "x", ms = 20,
         markerfacecolor="none", markeredgecolor = 'crimson',
         markeredgewidth = marker_edge_width, alpha = 0.85,
         label = "Monte Carlo RT")

###################### 

ax.set_xscale("log")
ax.set_yscale("log")

if plot_deviation_panel:
    ax_dev.set_xscale("log")
    ax_dev.axhline(0.0, color="black", lw=2.5, alpha=0.75, linestyle = 'dashed', dash_capstyle = 'round')
    #ax_dev.axhline(20.0, color="black", lw=1.7, alpha=0.35, linestyle="--", dash_capstyle="round")
    #ax_dev.axhline(-20.0, color="black", lw=1.7, alpha=0.35, linestyle="--", dash_capstyle="round")

    ax_dev.set_xlabel(r"Cloud Ly$\alpha$ optical depth $\tau_{cl} \simeq 5.9 \times 10^{6} \, (N_{HI}/10^{20} \, cm^{-2})$", 
                      labelpad=15)
    ax_dev.set_ylabel("Deviation (%)", labelpad=15, fontsize=40)
else:
    ax.set_xlabel(r"Cloud Ly$\alpha$ optical depth $\tau_{cl}$", labelpad=15)

ax.set_ylabel(r"Ly$\alpha$ force multiplier $M_F$", labelpad=15)
ax.set_title(figure_title, pad=20, fontsize=39.5)
ax.text(17, 0.016, "Dust-free static cloud, uniform emission, T = 10$^4$ K." + "\n" + r"Ignoring atomic recoil and Ly$\alpha$ destruction.",
        fontsize = 32)

# Custom log tick formatter:
def log_tick_formatter_x(val, pos=None):
    if val <= 0:
        return ""

    exp = np.log10(val)
    if not np.isclose(exp, np.round(exp)):
        return ""

    exp = int(np.round(exp))

    if exp == 0:
        return "1"
    elif exp == 1:
        return "10"
    elif exp == 2:
        return "10$^2$"
    elif exp == -1:
        return "0.1"
    elif exp == -2:
        return "0.01"
    else:
        return rf"$10^{{{exp}}}$"


def percent_tick_formatter(val, pos=None):
    if np.isclose(val, 0.0):
        return "0"
    return f"{val:.0f}"

ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
ax.yaxis.set_major_locator(LogLocator(base=10.0))

ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter_x))
ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter_x))

ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=999))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))

ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

if plot_deviation_panel:
    ax_dev.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax_dev.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter_x))
    ax_dev.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=999))
    ax_dev.xaxis.set_minor_formatter(NullFormatter())

    ax_dev.yaxis.set_major_locator(FixedLocator(deviation_major_ticks))
    ax_dev.yaxis.set_minor_locator(FixedLocator(deviation_minor_ticks))
    ax_dev.yaxis.set_major_formatter(FuncFormatter(percent_tick_formatter))
    ax_dev.yaxis.set_minor_formatter(NullFormatter())

    ax_dev.text( 20.0, 25.0, "Deviation from Monte Carlo RT")

    if deviation_ylim is not None:
        ax_dev.set_ylim(*deviation_ylim)

    # Hide duplicated x tick labels on the top panel.
    ax.tick_params(labelbottom=False)

ax.set_xlim(*xlim)

if ylim is not None:
    ax.set_ylim(*ylim)

ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), 
          ncol = 2, columnspacing=0.3, labelspacing=0.5)

plt.tight_layout()
plt.savefig(repo_root / save_name, dpi=300, bbox_inches="tight")
plt.show()
