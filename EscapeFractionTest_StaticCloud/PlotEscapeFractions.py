import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
from pathlib import Path
import re

# Read LYDION data:
repo_root = Path(__file__).resolve().parent
data_dir  = repo_root / "LYDION_DataDustyClouds"

# Match e.g. "..._tauabs0.03_..." or "..._tauabs1.0_..."
tauabs_pattern = re.compile(r"tauabs([0-9eE+\-.]+)")
res_pattern    = re.compile(r"_(\d+)x(\d+)\.h5$")
rs_pattern     = re.compile(r"rs([0-9eE+\-.]+)_")

# Store by source type and resolution:
# lydion_data[(source_type, Nr, NZ)] = [(x_plot, fesc_percent), ...]
lydion_data = {}

for filepath in sorted(data_dir.glob("*.h5")):

    filename = filepath.name

    # Extract tauabs from file name:
    match_tau = tauabs_pattern.search(filename)
    if match_tau is None:
        print(f"Skipping {filename}: could not parse tauabs from file name.")
        continue

    tauabs = float(match_tau.group(1))

    # Extract resolution from file name:
    match_res = res_pattern.search(filename)
    if match_res is None:
        print(f"Skipping {filename}: could not parse resolution from file name.")
        continue

    Nr = int(match_res.group(1))
    NZ = int(match_res.group(2))

    # Extract source size from file name:
    match_rs = rs_pattern.search(filename)
    if match_rs is None:
        print(f"Skipping {filename}: could not parse rs from file name.")
        continue

    rs = float(match_rs.group(1))

    # Define source type:
    if np.isclose(rs, 0.0):
        source_type = "point"
    elif np.isclose(rs, 1.0):
        source_type = "uniform"
    else:
        source_type = f"rs={rs}"

    data_key = (source_type, Nr, NZ)

    with h5py.File(filepath, "r") as f:

        # Read a*tau_cl from HDF5 settings attribute:
        atau = float(f["Settings"].attrs["atau"])

        # Read escape fraction history:
        fesc_hist = np.array(f["f_esc_Lya_History"][:], dtype=float)

        # Robustly take the final finite value:
        good = np.isfinite(fesc_hist)
        if not np.any(good):
            print(f"Skipping {filename}: no finite values in f_esc_Lya_History.")
            continue

        fesc_final = fesc_hist[good][-1]

    x_plot = (atau ** (1.0 / 3.0)) * tauabs
    fesc_percent = 100.0 * fesc_final

    if data_key not in lydion_data:
        lydion_data[data_key] = []

    lydion_data[data_key].append((x_plot, fesc_percent))

# Sort each set by x:
for data_key in lydion_data:
    lydion_data[data_key] = sorted(lydion_data[data_key], key=lambda t: t[0])

# Read MCRT comparison data:
x_MCRT_uniform, fesc_MCRT_uniform = np.loadtxt(repo_root / "f_esc_tdMAX1.0_uniform_MCRT.txt", unpack=True)
x_MCRT_point,   fesc_MCRT_point   = np.loadtxt(repo_root / "f_esc_tdMAX1.0_point_MCRT.txt",   unpack=True)

# Read analytical comparison data:
Neufeld_ana_point,   fesc_ana_point_raw   = np.loadtxt(repo_root / "AnalyticalEscapeFraction_PointSource.txt",   unpack=True)
Neufeld_ana_uniform, fesc_ana_uniform_raw = np.loadtxt(repo_root / "AnalyticalEscapeFraction_UniformSource.txt", unpack=True)

# Files store the second column as -f_esc:
fesc_ana_point   = - fesc_ana_point_raw
fesc_ana_uniform = - fesc_ana_uniform_raw

# Neufeld dust parameter is \epst_1 = 0.4707 (a * tau_cl)^(1/3) * tau_abs,
# so convert before plotting:
x_ana_point   = Neufeld_ana_point / 0.4707
x_ana_uniform = Neufeld_ana_uniform / 0.4707

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

params = {'legend.fontsize': 30}
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

fig, ax = plt.subplots(figsize=(12, 10))

# MCRT comparison curves:
ax.plot(x_MCRT_uniform, fesc_MCRT_uniform, '-', lw=10, color='gray', alpha=0.7,
        label='Monte Carlo RT')
ax.plot(x_MCRT_point,   fesc_MCRT_point,   '-', lw=10, color='crimson', alpha=0.6)

# Analytical comparison curves:
ax.plot(x_ana_uniform, 100 * fesc_ana_uniform, '-', lw=3.0, color = 'black', 
        label = 'Analytical solutions')

xmax_point_ana = 20.0 # Plot analytical solution to this x, because it becomes unreliable at
                      # higher x due to numerical issues 
good_point_ana = x_ana_point <= xmax_point_ana

ax.plot(x_ana_point[good_point_ana],
        100 * fesc_ana_point[good_point_ana],
        '-',
        lw=3.0,
        color='crimson')

# Marker choices for different LYDION resolutions:
marker_list = ['s', 'o', '*', 'X', 'P', 'X', '*']

# Plot uniform-source LYDION runs in black:
uniform_keys = sorted([key for key in lydion_data if key[0] == "uniform"])
for i, data_key in enumerate(uniform_keys):
    source_type, Nr, NZ = data_key
    pts = lydion_data[data_key]

    xvals = np.array([p[0] for p in pts])
    fescs = np.array([p[1] for p in pts])

    marker = marker_list[i % len(marker_list)]

    ax.plot(xvals, fescs,
            linestyle='None',
            marker=marker,
            ms=22,
            markerfacecolor='none',
            markeredgecolor='black',
            markeredgewidth=2.5,
            alpha=0.8,
            label=rf'LYDION ${Nr}^{2}$')

# Plot point-source LYDION runs in crimson:
point_keys = sorted([key for key in lydion_data if key[0] == "point"])
for i, data_key in enumerate(point_keys):
    source_type, Nr, NZ = data_key
    pts = lydion_data[data_key]

    xvals = np.array([p[0] for p in pts])
    fescs = np.array([p[1] for p in pts])

    marker = marker_list[i % len(marker_list)]

    # Only add a legend entry if this resolution was not already used for uniform runs
    same_res_in_uniform = ("uniform", Nr, NZ) in lydion_data

    if same_res_in_uniform:
        label_here = None
    else:
        label_here = rf'LYDION point ${Nr}^{2}$'

    ax.plot(xvals, fescs,
            linestyle='None',
            marker=marker,
            ms=22,
            markerfacecolor='none',
            markeredgecolor='crimson',
            markeredgewidth=2.5,
            alpha=0.8,
            label=label_here)

# Plot any additional rs-values (if present) with blue outlines:
other_keys = sorted([key for key in lydion_data if key[0] not in ["uniform", "point"]])
for i, data_key in enumerate(other_keys):
    source_type, Nr, NZ = data_key
    pts = lydion_data[data_key]

    xvals = np.array([p[0] for p in pts])
    fescs = np.array([p[1] for p in pts])

    marker = marker_list[i % len(marker_list)]

    ax.plot(xvals, fescs,
            linestyle='None',
            marker=marker,
            ms=14,
            markerfacecolor='none',
            markeredgecolor='royalblue',
            markeredgewidth=2.0,
            alpha=0.8,
            label=rf'LYDION {source_type} ${Nr}^{2}$')


#### PLOT CUSTOMIZATION: ##################
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r'$(a_{v}\tau_{cl})^{1/3}\tau_{c}$', labelpad = 15)
ax.set_ylabel(r'Ly$\alpha$ escape fraction (%)', labelpad = 15)
ax.set_title("Ly$\\alpha$ escape in dusty clouds:\nLYDION vs. MCRT & analytical solution", pad=20, fontsize=39.5)
ax.set_ylim(0.003, 2e2)
ax.set_xlim(0.1, 50.0)

ax.text(5.0, 22.0, "Uniform source", color='black', fontsize=30, rotation = - 13)
ax.text(5.0, 0.3, "Point-like source", color='crimson', fontsize=30, rotation = - 42)

# Custom log tick formatter:
def log_tick_formatter(val, pos=None):
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
        return "100"
    elif exp == -1:
        return "0.1"
    elif exp == -2:
        return "0.01"
    else:
        return rf"$10^{{{exp}}}$"

ax.xaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_major_locator(LogLocator(base=10.0))

ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

ax.legend(loc='lower left', bbox_to_anchor=(0.05, 0.08))

plt.tight_layout()
plt.savefig(repo_root / "EscapeFractionComparison.pdf", dpi=300, bbox_inches='tight')
plt.show()