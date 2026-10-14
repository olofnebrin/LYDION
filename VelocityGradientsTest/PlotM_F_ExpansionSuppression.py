import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, FuncFormatter
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
import re

# Read LYDION data:
repo_root = Path(__file__).resolve().parent

# By default, search recursively from the script directory. This makes the script
# work whether the CloudTest files are placed beside the script or in a data subfolder.
data_dir = repo_root

###############################################################
#######                                                 #######
#######                 M C R T   D A T A               #######
#######                                                 #######
###############################################################

# MCRT/COLT results for the expanding-cloud test that Kevin sent.
# vel is the maximum/cloud expansion velocity in km/s, corresponding to umax
# in the LYDION file names.
MCRT_vel = np.array([
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0
], dtype=float)

MCRT_MF = np.array([
    18.45,
    18.24,
    17.53,
    13.86,
    6.772,
    2.83,
    0.9890
], dtype=float)

if MCRT_vel.size != MCRT_MF.size:
    raise ValueError("MCRT_vel and MCRT_MF must have the same length.")

if MCRT_vel.size > 0:
    good_mcrt = np.isfinite(MCRT_vel) & np.isfinite(MCRT_MF) & (MCRT_vel > 0.0) & (MCRT_MF > 0.0)
    MCRT_vel = MCRT_vel[good_mcrt]
    MCRT_MF  = MCRT_MF[good_mcrt]

    sort_mcrt = np.argsort(MCRT_vel)
    MCRT_vel = MCRT_vel[sort_mcrt]
    MCRT_MF  = MCRT_MF[sort_mcrt]
else:
    print()
    print("WARNING: MCRT_vel/MCRT_MF arrays are empty.")
    print("         The top panel will omit MCRT points, and the deviation panel will remain empty.")
    print("         Fill in the MCRT arrays near the top of this script.")
    print()


def interp_no_extrap_logx(x_new, x, y, rtol=1e-8, atol=1e-12):
    """
    Log-x interpolation, but returns NaN outside the tabulated x-range.

    Important:
    If a resolution only has one LYDION point, keep it if it matches
    one of the MCRT velocities. This avoids sparse high-resolution
    data disappearing from the deviation panel.
    """
    x_new = np.asarray(x_new, dtype=float)
    x     = np.asarray(x, dtype=float)
    y     = np.asarray(y, dtype=float)

    y_new = np.full_like(x_new, np.nan, dtype=float)

    good = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
    x = x[good]
    y = y[good]

    if x.size == 0:
        return y_new

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # First keep direct matches, even if there is only one LYDION point.
    for n, xn in enumerate(x_new):
        match = np.isclose(x, xn, rtol=rtol, atol=atol)
        if np.any(match):
            y_new[n] = np.mean(y[match])

    # Then interpolate remaining points only when we have enough points.
    if x.size >= 2:
        inside = (x_new >= x[0]) & (x_new <= x[-1]) & (x_new > 0.0) & ~np.isfinite(y_new)
        y_new[inside] = np.interp(np.log10(x_new[inside]), np.log10(x), y)

    return y_new


def percent_deviation(model_mf, reference_mf):
    """Return 100 * (model - reference) / reference."""
    model_mf     = np.asarray(model_mf, dtype=float)
    reference_mf = np.asarray(reference_mf, dtype=float)

    dev = np.full_like(model_mf, np.nan, dtype=float)
    good = np.isfinite(model_mf) & np.isfinite(reference_mf) & (reference_mf != 0.0)
    dev[good] = 100.0 * (model_mf[good] - reference_mf[good]) / reference_mf[good]

    return dev


def analytical_MF_velocity_gradient(
    vel_kms,
    tau_cl=1.0e8,
    T=10000.0,
    tau_s_bar=1.0,
    P=0.0,
    nmax=1000000,
):
    """
    Analytical force multiplier for the expanding-cloud test.

    This evaluates the explicit no-recoil/no-continuum-absorption
    series solution from Eqs. 65-66 of Nebrin et al. 2024.

    Here P = 0 for no Ly-alpha destruction, and tau_s_bar = 1 for
    uniform emission in the cloud.
    """
    vel_kms = np.asarray(vel_kms, dtype=float)

    # Voigt parameter and thermal velocity.
    # a_v = 4.7e-3 * (T/100 K)^(-1/2)
    a_v = 4.7e-3 * (T / 100.0)**(-0.5)

    k_B = 1.380649e-16        # erg K^-1
    m_H = 1.6735575e-24       # g
    b_kms = np.sqrt(2.0 * k_B * T / m_H) / 1.0e5

    Rdot_over_b = vel_kms / b_kms

    n = np.arange(nmax, dtype=float)
    n_o = 2.0*n + 1.0

    npi = n_o * np.pi
    Dbar = np.sqrt(1.0 + (3.0 / (2.0*npi**2)) * Rdot_over_b[..., None]**2)

    source_factor = (
        np.sin(npi*tau_s_bar)
        - npi*tau_s_bar*np.cos(npi*tau_s_bar)
    ) / tau_s_bar**3

    K_prefactor = np.sqrt(2.0*np.pi/3.0) * npi * Dbar / 3.0
    K_shift     = np.sqrt(np.pi) * Rdot_over_b[..., None] / 3.0

    # For very large velocities the minus branch approaches zero from above.
    # The maximum below only protects against tiny round-off errors.
    K_plus_arg  = np.maximum(K_prefactor + K_shift, 1.0e-300)
    K_minus_arg = np.maximum(K_prefactor - K_shift, 1.0e-300)

    K_plus  = K_plus_arg**(-1.0/3.0)
    K_minus = K_minus_arg**(-1.0/3.0)

    Gamma_4_3 = 0.8929795115692492

    terms = (
        source_factor
        / (npi**3 * (P + npi*Dbar))
        * (K_plus + K_minus)
    )

    MF_norm = 4.0*np.sqrt(6.0)*Gamma_4_3*np.sum(terms, axis=-1)
    MF = MF_norm * (a_v*tau_cl)**(1.0/3.0)

    return MF


# File-name parsers. Example:
# CloudTest_rs1.0_NoRecoil_tau1.0e8_T10000.0_umax300.0_tauabs0.0_pd0.0_128x128_q1.1.h5
umax_pattern   = re.compile(r"umax([0-9eE+\-.]+)_")
tau_pattern    = re.compile(r"tau([0-9eE+\-.]+)_T")
T_pattern      = re.compile(r"_T([0-9eE+\-.]+)_")
tauabs_pattern = re.compile(r"tauabs([0-9eE+\-.]+)_")
pd_pattern     = re.compile(r"pd([0-9eE+\-.]+)_")
res_pattern    = re.compile(r"_(\d+)x(\d+)_q([0-9eE+\-.]+)\.h5$")
rs_pattern     = re.compile(r"rs([0-9eE+\-.]+)_")

# Keep only the intended setup. Relax these if you want to compare variants.
target_rs     = 1.0
target_tau    = 1.0e8
target_T      = 10000.0
target_tauabs = 0.0
target_pd     = 0.0
target_q_values = [1.1, 1.03]


def infer_frequency_bin_count(h5file):
    """
    Return the number of Ly-alpha frequency bins.

    In SaveHDF5Data_StaticCloudTests.jl the frequency grid is saved as:
      "ν"   : bin edges, length N_ν + 1
      "νC"  : bin centers, length N_ν
      "δνC" : bin centers relative to line center, length N_ν
      "Δν"  : bin widths, length N_ν

    Therefore the safest definition is length("νC"), with length("Δν")
    as an equivalent fallback.
    """
    preferred_names = ["νC", "Δν", "δνC"]

    for name in preferred_names:
        if name in h5file:
            return (name, int(h5file[name].shape[0]))

    # If only edges exist, the number of bins is N_edges - 1.
    if "ν" in h5file:
        return ("ν", max(int(h5file["ν"].shape[0]) - 1, 0))

    # ASCII fallbacks, in case you later change the save names.
    preferred_ascii_names = [
        "nuC", "nu_centers", "nu_center", "nuCenters", "nu_C",
        "Delta_nu", "dnu", "dnuC", "bin_widths", "freq_centers",
        "frequency_centers", "x_centers",
    ]

    for name in preferred_ascii_names:
        if name in h5file:
            return (name, int(h5file[name].shape[0]))

    return None


def last_saved_history_index(h5file, history_name="M_F_History", time_name="Times"):
    """
    Return the index of the last actually saved history entry.

    The HDF5 history arrays are preallocated to Nsaved. If a run stops before
    filling all entries, the unused tail can remain zero. Since zero is finite,
    simply taking the last finite M_F can accidentally read an unsaved slot.

    The save routine writes Times[step] = t whenever it writes M_F_History[step],
    so the safest choice is the last positive saved time. If no positive time is
    found, fall back to the last finite, nonzero M_F, and finally the last finite
    M_F.
    """
    mf_hist = np.array(h5file[history_name][:], dtype=float)

    if time_name in h5file:
        times = np.array(h5file[time_name][:], dtype=float)
        n = min(times.size, mf_hist.size)

        # Unused preallocated entries normally stay at t = 0.
        good_time = np.isfinite(times[:n]) & (times[:n] > 0.0)
        if np.any(good_time):
            return int(np.where(good_time)[0][-1]), times, mf_hist

    good_nonzero_mf = np.isfinite(mf_hist) & (mf_hist != 0.0)
    if np.any(good_nonzero_mf):
        return int(np.where(good_nonzero_mf)[0][-1]), None, mf_hist

    good_mf = np.isfinite(mf_hist)
    if np.any(good_mf):
        return int(np.where(good_mf)[0][-1]), None, mf_hist

    return None, None, mf_hist


def canonical_q(q):
    """Map parsed q onto the requested q values, avoiding float-key surprises."""
    for target_q in target_q_values:
        if np.isclose(q, target_q, rtol=1e-10, atol=1e-12):
            return float(target_q)
    return None


# Store by resolution and frequency-grid spacing:
# mf_data[(Nr, NZ, q)] = [(umax, M_F_final), ...]
mf_data = {}
frequency_bin_counts = {}

# Recursive search allows e.g. repo_root/LYDION_DataExpansionClouds/*.h5
for filepath in sorted(data_dir.glob("**/CloudTest_rs*_NoRecoil_tau*_T*_umax*_tauabs*_pd*_*x*_q*.h5")):

    filename = filepath.name

    match_umax = umax_pattern.search(filename)
    if match_umax is None:
        print(f"Skipping {filename}: could not parse umax from file name.")
        continue
    umax = float(match_umax.group(1))

    match_tau = tau_pattern.search(filename)
    if match_tau is None:
        print(f"Skipping {filename}: could not parse tau from file name.")
        continue
    tau = float(match_tau.group(1))

    match_T = T_pattern.search(filename)
    if match_T is None:
        print(f"Skipping {filename}: could not parse T from file name.")
        continue
    T = float(match_T.group(1))

    match_tauabs = tauabs_pattern.search(filename)
    if match_tauabs is None:
        print(f"Skipping {filename}: could not parse tauabs from file name.")
        continue
    tauabs = float(match_tauabs.group(1))

    match_pd = pd_pattern.search(filename)
    if match_pd is None:
        print(f"Skipping {filename}: could not parse pd from file name.")
        continue
    pd = float(match_pd.group(1))

    match_res = res_pattern.search(filename)
    if match_res is None:
        print(f"Skipping {filename}: could not parse resolution/q from file name.")
        continue
    Nr = int(match_res.group(1))
    NZ = int(match_res.group(2))
    q  = float(match_res.group(3))

    match_rs = rs_pattern.search(filename)
    if match_rs is None:
        print(f"Skipping {filename}: could not parse rs from file name.")
        continue
    rs = float(match_rs.group(1))

    # Keep only the selected setup.
    if not np.isclose(rs, target_rs):
        continue
    if not np.isclose(tau, target_tau):
        continue
    if not np.isclose(T, target_T):
        continue
    if not np.isclose(tauabs, target_tauabs):
        continue
    if not np.isclose(pd, target_pd):
        continue
    q_key = canonical_q(q)
    if q_key is None:
        continue

    with h5py.File(filepath, "r") as f:
        if "M_F_History" not in f:
            print(f"Skipping {filename}: M_F_History not found.")
            continue

        hist_idx, times_hist, mf_hist = last_saved_history_index(f, history_name="M_F_History", time_name="Times")

        if hist_idx is None:
            print(f"Skipping {filename}: no finite values in M_F_History.")
            continue

        mf_final = float(mf_hist[hist_idx])
        if times_hist is not None and hist_idx < times_hist.size and np.isfinite(times_hist[hist_idx]):
            t_final = float(times_hist[hist_idx])
        else:
            t_final = np.nan

        freq_info = infer_frequency_bin_count(f)
        if freq_info is not None:
            freq_dataset_name, Nfreq = freq_info
            frequency_bin_counts[(Nr, NZ, q_key)] = (Nfreq, freq_dataset_name)

    data_key = (Nr, NZ, q_key)
    if data_key not in mf_data:
        mf_data[data_key] = []

    mf_data[data_key].append((umax, mf_final))

    # Useful sanity print: if q = 1.03 is missing, this will make it clear
    # whether the file was found and passed the filters.
    freq_string = "unknown"
    freq_info = frequency_bin_counts.get(data_key)
    if freq_info is not None:
        freq_string = f"{freq_info[0]} from {freq_info[1]}"
    t_string = "unknown" if not np.isfinite(t_final) else f"{t_final:.6e}"
    print(f"Found LYDION file: {filename}  |  q = {q_key:g}, umax = {umax:g}, "
          f"M_F = {mf_final:.6e}, saved index = {hist_idx}, t = {t_string}, N_freq = {freq_string}")

# Sort each set by expansion velocity:
for data_key in mf_data:
    mf_data[data_key] = sorted(mf_data[data_key], key=lambda t: t[0])

# Print all umax and M_F values for a chosen LYDION resolution:
target_resolution = (128, 128)   # change this, e.g. (32,32), (64,64), (256,256)

target_keys = [key for key in sorted(mf_data) if key[0:2] == target_resolution]

# Warn explicitly if one of the requested q values is absent at the selected resolution.
for q_wanted in target_q_values:
    if not any((key[0:2] == target_resolution and np.isclose(key[2], q_wanted)) for key in mf_data):
        print(f"WARNING: No matching LYDION files found for resolution {target_resolution[0]} x {target_resolution[1]}, q = {q_wanted:g}.")
        print(f"         Searched recursively below: {data_dir}")

if len(target_keys) == 0:
    print(f"No data found for resolution {target_resolution}.")
    print("Available resolution/q combinations:")
    for Nr, NZ, q in sorted(mf_data):
        print(f"  {Nr} x {NZ}, q = {q:g}")
else:
    print()
    print(f"LYDION results for resolution {target_resolution[0]} x {target_resolution[1]}:")

    for data_key in target_keys:
        Nr, NZ, q = data_key
        print(f"q = {q:g}")
        print("umax [km/s]       M_F")
        print("-----------------------------")

        for umax, mf_final in mf_data[data_key]:
            print(f"{umax:12.6e}   {mf_final:12.6e}")

        print()

    print("Frequency-bin counts for the selected resolution:")
    print("q              N_freq       dataset")
    print("-----------------------------------------------")
    for data_key in target_keys:
        Nr, NZ, q = data_key
        freq_info = frequency_bin_counts.get(data_key)
        if freq_info is None:
            print(f"{q:8g}       unknown      no explicit 1D frequency grid found")
        else:
            Nfreq, freq_dataset_name = freq_info
            print(f"{q:8g}       {Nfreq:6d}       {freq_dataset_name}")
    print()

if MCRT_vel.size > 0:
    print("MCRT input data:")
    print("vel [km/s]       M_F")
    print("-----------------------------")
    for vel, mf_final in zip(MCRT_vel, MCRT_MF):
        print(f"{vel:12.6e}   {mf_final:12.6e}")
    print()

# Analytical solution from Eqs. 65-66 for this setup:
# no Ly-alpha destruction (P = 0), uniform source (tau_s_bar = 1),
# no recoil, and no continuum absorption.
ana_vel_plot = np.logspace(np.log10(1.0), np.log10(2000.0), 400)
ana_MF_plot  = analytical_MF_velocity_gradient(
    ana_vel_plot,
    tau_cl=target_tau,
    T=target_T,
    tau_s_bar=1.0,
    P=0.0,
    nmax=2000,
)

ana_MF_at_MCRT = analytical_MF_velocity_gradient(
    MCRT_vel,
    tau_cl=target_tau,
    T=target_T,
    tau_s_bar=1.0,
    P=0.0,
    nmax=1000000,
)

ana_percent_dev = percent_deviation(ana_MF_at_MCRT, MCRT_MF)

if MCRT_vel.size > 0:
    print("Analytical solution:")
    print("vel [km/s]       M_F          deviation [%]")
    print("------------------------------------------")
    for vel, mf_final, dev_final in zip(MCRT_vel, ana_MF_at_MCRT, ana_percent_dev):
        print(f"{vel:12.6e}   {mf_final:12.6e}   {dev_final:12.6e}")
    print()

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

params = {'legend.fontsize': 34}
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

fig, (ax, ax_dev) = plt.subplots(
    2, 1,
    figsize=(11.5, 13.5),
    sharex=True,
    gridspec_kw={'height_ratios': [3.2, 2.4], 'hspace': 0.06}
)

xmin_plot = 0.6
xmax_plot = 2000.0

# MCRT predictions:
if MCRT_vel.size > 0:
    ax.plot(MCRT_vel, MCRT_MF,
            linestyle='None',
            marker='x',
            ms=18,
            markerfacecolor='none',
            markeredgecolor='black',
            markeredgewidth=3.5,
            alpha=0.9,
            label='Monte Carlo RT')

# Analytical solution:
ax.plot(ana_vel_plot, ana_MF_plot,
        '-',
        color='crimson',
        lw=4.0,
        alpha=0.9,
        label='Analytical solution',
        zorder = 1)

# LYDION predictions:
# Use deterministic q-dependent styles so the high-frequency run does not get
# hidden by the standard q = 1.1 symbols when both have a point at umax = 1000.
q_style = {
    1.1:  dict(marker='*', color='seagreen', ms=30, zorder=4),
    1.03: dict(marker='P', color='dodgerblue',   ms=20, zorder=5),
}
marker_list = ['*', 'o', 'P', 'D', '^', 's']
color_list  = ['seagreen', 'royalblue', 'orangered', 'purple', 'darkorange', 'teal']

for i, data_key in enumerate(sorted(mf_data)):
    Nr, NZ, q = data_key
    pts = mf_data[data_key]

    velvals = np.array([p[0] for p in pts])
    mfvals  = np.array([p[1] for p in pts])

    style = q_style.get(q, None)
    if style is None:
        marker = marker_list[i % len(marker_list)]
        color  = color_list[i % len(color_list)]
        ms     = 30
        zorder = 4
    else:
        marker = style['marker']
        color  = style['color']
        ms     = style['ms']
        zorder = style['zorder']

    # Top panel:
    ax.plot(velvals, mfvals,
            linestyle='None',
            marker=marker,
            ms=ms,
            markerfacecolor='none',
            markeredgecolor=color,
            markeredgewidth=3.5,
            alpha=0.8,
            zorder=zorder,
            label=rf'LYDION ${Nr}^{2}$, $q={q:g}$')

    # Bottom panel: percentage deviation from MCRT.
    # MCRT_vel defines the x-grid here.
    if MCRT_vel.size > 0:
        mf_lydion_at_mcrt = interp_no_extrap_logx(MCRT_vel, velvals, mfvals)
        percent_dev       = percent_deviation(mf_lydion_at_mcrt, MCRT_MF)

        good_dev = np.isfinite(percent_dev)
        ax_dev.plot(MCRT_vel[good_dev], percent_dev[good_dev],
                    linestyle='None',
                    marker=marker,
                    ms=ms,
                    markerfacecolor='none',
                    markeredgecolor=color,
                    markeredgewidth=3.0,
                    alpha=0.8,
                    zorder=zorder)

# Optional: show the MCRT reference itself at zero deviation.
if MCRT_vel.size > 0:
    ax_dev.plot(MCRT_vel, np.zeros_like(MCRT_vel),
                '-',
                color='black',
                lw=1.5,
                alpha=0.8)

# Analytical deviation from MCRT:
if MCRT_vel.size > 0:
    good_ana_dev = np.isfinite(ana_percent_dev)
    ax_dev.plot(MCRT_vel[good_ana_dev], ana_percent_dev[good_ana_dev],
                '-',
                color='crimson',
                lw=4.0,
                marker='o',
                ms=12,
                markerfacecolor='none',
                markeredgecolor='crimson',
                markeredgewidth=3.0,
                alpha=0.9)

ax.set_xscale("log")
ax.set_yscale("log")
ax_dev.set_xscale("log")

ax.set_ylabel(r'Ly$\alpha$ force multiplier $M_F$', labelpad=15)
ax.set_title('Impact of velocity gradients on M$_{F}$:\nLYDION vs. MCRT & analytical solution', pad=20, fontsize=39.5)
#ax.set_title('Ly$\\alpha$ feedback suppression by cloud expansion:\nLYDION vs. MCRT', pad=20, fontsize=39.5)

ax_dev.axhline(0.0, color='black', lw=1.5, ls='-')
ax_dev.set_xlabel(r'Cloud expansion rate $\dot{R}_{cl}$ (km s$^{-1}$)', labelpad=15)
ax_dev.set_ylabel(r'Deviation (%)', labelpad=15)

# Custom log tick formatter:
def log_tick_formatter(val, pos=None):
    if val <= 0.0:
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
        return "100"
    elif exp == 3:
        return r"$10^{3}$"
    elif exp == -1:
        return "0.1"
    elif exp == -2:
        return "0.01"
    else:
        return rf"$10^{{{exp}}}$"

#x_major_ticks = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3]
x_major_ticks = [1.0, 10.0, 100.0, 1000.0]

def velocity_tick_formatter(val, pos=None):
    if np.any(np.isclose(val, x_major_ticks, rtol=1e-8, atol=1e-12)):
        if np.isclose(val, round(val)):
            return f"{int(round(val))}"
        return f"{val:g}"
    return ""

ax.xaxis.set_major_locator(FixedLocator(x_major_ticks))
ax_dev.xaxis.set_major_locator(FixedLocator(x_major_ticks))
y_major_ticks = [0.1, 1.0, 10.0]

ax.yaxis.set_major_locator(FixedLocator(y_major_ticks))

ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
ax_dev.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))

ax.xaxis.set_major_formatter(FuncFormatter(velocity_tick_formatter))
ax_dev.xaxis.set_major_formatter(FuncFormatter(velocity_tick_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax_dev.xaxis.set_minor_formatter(NullFormatter())
ax_dev.yaxis.set_minor_locator(AutoMinorLocator(4))

ax.set_xlim(xmin_plot, xmax_plot)
ax.set_ylim(0.4, 30.0)

# Optional: adjust this after seeing the plot.
ax_dev.set_ylim(-20, 40)

ax_dev.text(1.1, 20.0, "Deviation from\nMonte Carlo RT")

ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02))

plt.tight_layout()
plt.savefig(repo_root / "M_F_ExpansionSuppression_MCRT.pdf", dpi=300, bbox_inches='tight')
plt.show()
