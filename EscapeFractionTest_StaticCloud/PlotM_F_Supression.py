import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
from matplotlib.ticker import AutoMinorLocator, SymmetricalLogLocator
from pathlib import Path
import re

# Read LYDION data:
repo_root = Path(__file__).resolve().parent
data_dir  = repo_root / "LYDION_DataDustyClouds"

###############################################################
#######                                                 #######
#######                 M C R T   D A T A               #######
#######                                                 #######
###############################################################

# Kevin's MCRT results should be entered here.
# These arrays define the tau_abs values used in the deviation panel.
# Replace the example/placeholder values below with the latest COLT/MCRT data.
MCRT_tauabs = np.array([
    0.0,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
], dtype=float)

MCRT_MF = np.array([
    18.3, # Earlier no-core-skipping result from Kevin for dust-free limit
    18.30,
    18.16,
    18.07,
    17.38,
    15.34,
    11.63,
    6.195,
    2.638,
    0.8692
], dtype=float)

if MCRT_tauabs.size != MCRT_MF.size:
    raise ValueError("MCRT_tauabs and MCRT_MF must have the same length.")

if MCRT_tauabs.size > 0:
    good_mcrt = np.isfinite(MCRT_tauabs) & np.isfinite(MCRT_MF) & (MCRT_tauabs >= 0.0) & (MCRT_MF > 0.0)
    MCRT_tauabs = MCRT_tauabs[good_mcrt]
    MCRT_MF     = MCRT_MF[good_mcrt]

    sort_mcrt = np.argsort(MCRT_tauabs)
    MCRT_tauabs = MCRT_tauabs[sort_mcrt]
    MCRT_MF     = MCRT_MF[sort_mcrt]
else:
    print()
    print("WARNING: MCRT_tauabs/MCRT_MF arrays are empty.")
    print("         The top panel will omit MCRT points, and the deviation panel will remain empty.")
    print("         Fill in Kevin's MCRT arrays near the top of this script.")
    print()


def interp_no_extrap(x_new, x, y, rtol=1e-8, atol=1e-12):
    """
    Linear interpolation, but returns NaN outside the tabulated x-range.

    Important:
    If a resolution only has one LYDION point, keep it if it matches
    one of the MCRT tau_abs values. This avoids sparse high-resolution
    data disappearing from the deviation panel.
    """
    x_new = np.asarray(x_new, dtype=float)
    x     = np.asarray(x, dtype=float)
    y     = np.asarray(y, dtype=float)

    y_new = np.full_like(x_new, np.nan, dtype=float)

    good = np.isfinite(x) & np.isfinite(y)
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
        inside = (x_new >= x[0]) & (x_new <= x[-1]) & ~np.isfinite(y_new)
        y_new[inside] = np.interp(x_new[inside], x, y)

    return y_new


def percent_deviation(model_mf, reference_mf):
    """Return 100 * (model - reference) / reference."""
    model_mf     = np.asarray(model_mf, dtype=float)
    reference_mf = np.asarray(reference_mf, dtype=float)

    dev = np.full_like(model_mf, np.nan, dtype=float)
    good = np.isfinite(model_mf) & np.isfinite(reference_mf) & (reference_mf != 0.0)
    dev[good] = 100.0 * (model_mf[good] - reference_mf[good]) / reference_mf[good]

    return dev


tauabs_pattern = re.compile(r"tauabs([0-9eE+\-.]+)")
res_pattern    = re.compile(r"_(\d+)x(\d+)\.h5$")
rs_pattern     = re.compile(r"rs([0-9eE+\-.]+)_")

# Store by resolution:
# mf_data[(Nr, NZ)] = [(tauabs, M_F_final), ...]
mf_data = {}

for filepath in sorted(data_dir.glob("*.h5")):

    filename = filepath.name

    match_tau = tauabs_pattern.search(filename)
    if match_tau is None:
        print(f"Skipping {filename}: could not parse tauabs from file name.")
        continue
    tauabs = float(match_tau.group(1))

    match_res = res_pattern.search(filename)
    if match_res is None:
        print(f"Skipping {filename}: could not parse resolution from file name.")
        continue
    Nr = int(match_res.group(1))
    NZ = int(match_res.group(2))

    match_rs = rs_pattern.search(filename)
    if match_rs is None:
        print(f"Skipping {filename}: could not parse rs from file name.")
        continue
    rs = float(match_rs.group(1))

    # Keep only uniform-source cases:
    if not np.isclose(rs, 1.0):
        continue

    with h5py.File(filepath, "r") as f:
        mf_hist = np.array(f["M_F_History"][:], dtype=float)

        good = np.isfinite(mf_hist)
        if not np.any(good):
            print(f"Skipping {filename}: no finite values in M_F_History.")
            continue

        mf_final = mf_hist[good][-1]

    data_key = (Nr, NZ)
    if data_key not in mf_data:
        mf_data[data_key] = []

    mf_data[data_key].append((tauabs, mf_final))

# Sort each set by tauabs:
for data_key in mf_data:
    mf_data[data_key] = sorted(mf_data[data_key], key=lambda t: t[0])

# Print all tauabs and M_F values for a chosen LYDION resolution:
target_resolution = (128, 128)   # change this, e.g. (32,32), (64,64), (256,256)

if target_resolution not in mf_data:
    print(f"No data found for resolution {target_resolution}.")
    print("Available resolutions:")
    for res in sorted(mf_data):
        print(f"  {res[0]} x {res[1]}")
else:
    print()
    print(f"LYDION results for resolution {target_resolution[0]} x {target_resolution[1]}:")
    print("tauabs             M_F")
    print("-----------------------------")

    for tauabs, mf_final in mf_data[target_resolution]:
        print(f"{tauabs:12.6e}   {mf_final:12.6e}")

    print()

if MCRT_tauabs.size > 0:
    print("MCRT input data:")
    print("tauabs             M_F")
    print("-----------------------------")
    for tauabs, mf_final in zip(MCRT_tauabs, MCRT_MF):
        print(f"{tauabs:12.6e}   {mf_final:12.6e}")
    print()

# Read analytical data for the uniform source case:
eps_MF_uniform, MF_ana_uniform_norm = np.loadtxt(
    repo_root / "AnalyticalMF_UniformSource_nmax300.txt",
    unpack=True
)

# Convert to M_F, since data is stored as M_F / (a*tau_cl)^(1/3):
atau              = 4.7e-4 * 1e8
MF_dust_free_ana  = 0.51 * (atau**(1.0 / 3.0))
MF_ana_uniform    = MF_ana_uniform_norm * (atau**(1.0 / 3.0))

# Convert eps_MF_uniform to tau_c using the relation
# epst_1 = 0.470651 (a*tau_cl)^(1/3) * tau_c.
tau_c_ana_uniform = eps_MF_uniform / (0.470651 * (atau**(1.0 / 3.0)))

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
    figsize=(12, 14),
    sharex=True,
    gridspec_kw={'height_ratios': [3.2, 2.4], 'hspace': 0.06}
)

# Use a symmetric-log x-axis so that the dust-free point at tau_c = 0
# can be shown together with the logarithmic dust-opacity sequence.
# Values below linthresh_x are displayed on a small linear segment.
linthresh_x = 1.0e-4
xmax_plot   = 3.0

# Analytical predictions:
ax.plot([0.0, 3.0], [MF_dust_free_ana, MF_dust_free_ana],
        '--',
        color='gray',
        lw=4.0,
        dash_capstyle='round',
        label=r'0.51$(a_{v}\tau_{cl})^{1/3}$')
tau_c_ana_plot = np.concatenate(([0.0], tau_c_ana_uniform))
MF_ana_plot    = np.concatenate(([MF_dust_free_ana], MF_ana_uniform))
ax.plot(tau_c_ana_plot, MF_ana_plot, '-', lw=4.0, color='crimson', label='Analytical solution')

# MCRT predictions:
if MCRT_tauabs.size > 0:
    ax.plot(MCRT_tauabs, MCRT_MF,
            linestyle='None',
            marker='x',
            ms=18,
            markerfacecolor='none',
            markeredgecolor='black',
            markeredgewidth=3.5,
            alpha=0.9,
            label='Monte Carlo RT')

# LYDION predictions:
marker_list = ['s', 'o', '*', 'P', '*']
color_list  = ['black', 'royalblue', 'orangered', 'seagreen', 'purple']

for i, data_key in enumerate(sorted(mf_data)):
    Nr, NZ = data_key
    pts = mf_data[data_key]

    tauvals = np.array([p[0] for p in pts])
    mfvals  = np.array([p[1] for p in pts])

    marker = marker_list[i % len(marker_list)]
    color  = color_list[i % len(color_list)]

    # Top panel:
    ax.plot(tauvals, mfvals,
            linestyle='None',
            marker=marker,
            ms=24,
            markerfacecolor='none',
            markeredgecolor=color,
            markeredgewidth=3.5,
            alpha=0.8,
            label=rf'LYDION ${Nr}^{2}$')

    # Bottom panel: percentage deviation from MCRT.
    # MCRT_tauabs defines the x-grid here.
    if MCRT_tauabs.size > 0:
        mf_lydion_at_mcrt = interp_no_extrap(MCRT_tauabs, tauvals, mfvals)
        percent_dev       = percent_deviation(mf_lydion_at_mcrt, MCRT_MF)

        good_dev = np.isfinite(percent_dev)
        ax_dev.plot(MCRT_tauabs[good_dev], percent_dev[good_dev],
                    linestyle='None',
                    marker=marker,
                    ms=20,
                    markerfacecolor='none',
                    markeredgecolor=color,
                    markeredgewidth=3.0,
                    alpha=0.8)

# Bottom panel: analytical deviation from MCRT at available MCRT tau_abs values.
if MCRT_tauabs.size > 0:
    mf_ana_at_mcrt = interp_no_extrap(MCRT_tauabs, tau_c_ana_plot, MF_ana_plot)
    ana_dev        = percent_deviation(mf_ana_at_mcrt, MCRT_MF)

    good_ana_dev = np.isfinite(ana_dev)
    ax_dev.plot(MCRT_tauabs[good_ana_dev], ana_dev[good_ana_dev],
                '-',
                color='crimson',
                lw=4.0,
                marker='o',
                ms=12,
                markerfacecolor='none',
                markeredgecolor='crimson',
                markeredgewidth=3.0,
                alpha=0.9)

    # Optional: show the MCRT reference itself at zero deviation.
    #ax_dev.plot(MCRT_tauabs, np.zeros_like(MCRT_tauabs),
    #            linestyle='None',
    #            marker='D',
    #            ms=14,
    #            markerfacecolor='none',
    #            markeredgecolor='black',
    #            markeredgewidth=3.0,
    #            alpha=0.9)

ax.set_xscale("symlog", linthresh=linthresh_x, linscale=0.7, base=10)
ax.set_yscale("log")
ax_dev.set_xscale("symlog", linthresh=linthresh_x, linscale=0.7, base=10)

ax.set_ylabel(r'Ly$\alpha$ force multiplier $M_F$', labelpad=15)
ax.set_title('Ly$\\alpha$ feedback suppression by dust:\nLYDION vs. MCRT & analytical solution', pad=20, fontsize=39.5)

ax_dev.axhline(0.0, color='black', lw=1.5, ls='-')
ax_dev.set_xlabel(r'Dust absorption optical depth $\tau_c$', labelpad=15)
ax_dev.set_ylabel(r'Deviation (%)', labelpad=15)

# Custom log/symlog tick formatter:
def log_tick_formatter(val, pos=None):
    if np.isclose(val, 0.0):
        return "0"

    if val < 0:
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
    elif exp == -1:
        return "0.1"
    elif exp == -2:
        return "0.01"
    else:
        return rf"$10^{{{exp}}}$"

x_major_ticks = [0.0, 1e-4, 1e-3, 1.0e-2, 1.0e-1, 1.0, 3.0]

ax.xaxis.set_major_locator(FixedLocator(x_major_ticks))
ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax_dev.xaxis.set_major_locator(FixedLocator(x_major_ticks))

ax.xaxis.set_minor_locator(SymmetricalLogLocator(base=10.0, linthresh=linthresh_x, subs=np.arange(2, 10)))
ax_dev.xaxis.set_minor_locator(SymmetricalLogLocator(base=10.0, linthresh=linthresh_x, subs=np.arange(2, 10)))

ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
ax_dev.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax_dev.xaxis.set_minor_formatter(NullFormatter())
ax_dev.yaxis.set_minor_locator(AutoMinorLocator(10))

ax.set_ylim(0.1, 30.0)
ax.set_xlim(-0.3e-4, xmax_plot)

# Optional: adjust this after seeing the plot.
# Since deviations are now relative to MCRT, this may need a wider range.
ax_dev.set_ylim(-20, 15)

ax_dev.text( 0.0, -15.0, "Deviation from Monte Carlo RT" )

ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02))

plt.tight_layout()
plt.savefig(repo_root / "M_F_SuppressionTest_MCRT.pdf", dpi=300, bbox_inches='tight')
plt.show()
