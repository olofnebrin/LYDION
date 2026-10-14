import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter


###############################################################
#######                  U S E R  I N P U T             #######
###############################################################

# Timing files should be in the same directory as this script.
# Use None to auto-discover all matching timing files in this directory.
timing_files = None

# If auto-discovering, only files matching this pattern are used.
timing_glob = "*_TimingStats.txt"

outname = "ConeTest_CPUCoreHours_vs_Resolution.pdf"
savefig = True
showfig = True

# The x-axis uses the linear grid resolution N from filenames like 128x128.
# Set to None to infer from available timing files.
xtick_resolutions = [32, 64, 128, 256]

# Expected reference scaling: CPU core hours ∝ N^reference_power.
reference_power = 3.0
reference_label = r"$\propto N^3$"

# Anchor the reference line to the data point at this resolution.
# If this resolution is unavailable, the lowest available resolution is used.
reference_anchor_resolution = 32

# Optional vertical scaling of the reference line.
reference_scale_factor = 1.0

# Plot cosmetics.
figsize = (8.0, 7.0)
dpi_save = 400
marker_size = 600
marker = "*"
marker_edge_width = 2.2

# Axis labels.
xlabel = r"Spatial resolution, $N^2$"
ylabel = r"CPU core hours"

# Manual axis limits. Set to None for automatic.
xlim_user = (28, 310)
ylim_user = (0.1, 300)


###############################################################
#######                H E L P E R S                    #######
###############################################################

def script_dir():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def resolve_path(fname):
    path = Path(fname)
    if path.is_absolute():
        return path
    return script_dir() / path


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


def resolution_tick_formatter(x, pos=None):
    """
    Format the linear resolution N as N^2, since the simulations are 2D.
    """
    if x <= 0.0 or not np.isfinite(x):
        return ""

    n = int(np.round(x))
    if not np.isclose(x, n, rtol=0.0, atol=1e-8):
        return ""

    return rf"${n}^2$"


def parse_timing_file(path):
    """
    Return (N, CPUCoreHours, metadata) from a LYDION timing statistics file.

    N is inferred from the filename pattern _NxN_TimingStats.txt, e.g. 128x128.
    CPUCoreHours is read from the file contents.
    """
    path = Path(path)

    match = re.search(r"_(\d+)x(\d+)_TimingStats\.txt$", path.name)
    if match is None:
        raise ValueError(f"Could not infer resolution from filename: {path.name}")

    Nr = int(match.group(1))
    Nz = int(match.group(2))

    if Nr != Nz:
        raise ValueError(
            f"This script assumes square grids, but found {Nr}x{Nz} in {path.name}"
        )

    metadata = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()

    if "CPUCoreHours" not in metadata:
        raise KeyError(f"CPUCoreHours not found in {path}")

    cpu_core_hours = float(metadata["CPUCoreHours"])
    return Nr, cpu_core_hours, metadata


def discover_timing_files():
    if timing_files is None:
        files = sorted(script_dir().glob(timing_glob))
    else:
        files = [resolve_path(fname) for fname in timing_files]

    if len(files) == 0:
        raise RuntimeError(
            f"No timing files found. Looked for {timing_glob!r} in {script_dir()}"
        )

    return files


def read_timing_data():
    files = discover_timing_files()

    rows = []
    for path in files:
        N, cpu_core_hours, metadata = parse_timing_file(path)
        rows.append((N, cpu_core_hours, path, metadata))

    rows.sort(key=lambda row: row[0])

    # If the same resolution appears more than once, average it and report the scatter.
    grouped = {}
    for N, cpu_core_hours, path, metadata in rows:
        grouped.setdefault(N, []).append((cpu_core_hours, path, metadata))

    N_values = []
    cpu_values = []
    cpu_err = []
    used_files = []

    for N in sorted(grouped):
        vals = np.array([item[0] for item in grouped[N]], dtype=float)
        N_values.append(N)
        cpu_values.append(np.mean(vals))
        cpu_err.append(np.std(vals, ddof=1) if vals.size > 1 else 0.0)
        used_files.extend([item[1] for item in grouped[N]])

    return (
        np.array(N_values, dtype=float),
        np.array(cpu_values, dtype=float),
        np.array(cpu_err, dtype=float),
        used_files,
        grouped,
    )


def reference_curve(N_plot, N_data, cpu_data):
    if N_data.size == 0:
        raise RuntimeError("No data points available for reference curve.")

    if reference_anchor_resolution in N_data:
        idx = int(np.where(N_data == reference_anchor_resolution)[0][0])
    else:
        idx = 0

    N0 = N_data[idx]
    cpu0 = cpu_data[idx]

    return reference_scale_factor * cpu0 * (N_plot / N0)**reference_power


###############################################################
#######                  R E A D  D A T A               #######
###############################################################

N, cpu_core_hours, cpu_err, used_files, grouped = read_timing_data()

print("\n========== Loaded timing data ==========")
for N_i, cpu_i in zip(N, cpu_core_hours):
    entries = grouped[int(N_i)]
    print(f"N = {int(N_i):4d}: CPUCoreHours = {cpu_i:.6g}  ({len(entries)} file(s))")
print("Files:")
for path in used_files:
    print(f"  {path}")
print("=======================================\n")


###############################################################
#######                P L O T  S E T U P               #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 26})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

plt.rcParams.update({'legend.fontsize': 30})
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

fig, ax = plt.subplots(figsize=figsize)

ax.set_xscale("log")
ax.set_yscale("log")

# Data points.
if np.any(cpu_err > 0.0):
    ax.errorbar(
        N,
        cpu_core_hours,
        yerr=cpu_err,
        fmt=marker,
        ms=np.sqrt(marker_size),
        mfc="white",
        mec="black",
        mew=marker_edge_width,
        ecolor="black",
        elinewidth=2.0,
        capsize=5,
        linestyle="none",
        label="LYDION",
        zorder=3,
    )
else:
    ax.scatter(
        N,
        cpu_core_hours,
        s=marker_size,
        marker=marker,
        facecolor="orangered",
        edgecolor="orangered",
        linewidth=marker_edge_width,
        label="LYDION",
        zorder=1,
    )

# Reference N^3 scaling line.
N_min = np.min(N)
N_max = np.max(N)
if xtick_resolutions is not None:
    N_min = min(N_min, min(xtick_resolutions))
    N_max = max(N_max, max(xtick_resolutions))

N_plot = np.geomspace(N_min, N_max, 256)
y_ref = reference_curve(N_plot, N, cpu_core_hours)

ref_line, = ax.plot(
    N_plot,
    y_ref,
    color="black",
    linestyle="--",
    linewidth=2.8,
    dash_capstyle="round",
    label=reference_label,
    zorder=2,
)
ref_line.set_dashes([6.0, 5.0])

ax.set_xlabel(xlabel, labelpad=14, fontsize = 30)
ax.set_ylabel(ylabel, labelpad=14, fontsize = 30)

# Log tick setup.
if xtick_resolutions is not None:
    ax.set_xticks(xtick_resolutions)
    ax.xaxis.set_major_formatter(FuncFormatter(resolution_tick_formatter))
else:
    ax.xaxis.set_major_locator(LogLocator(base=2.0))
    ax.xaxis.set_major_formatter(FuncFormatter(resolution_tick_formatter))

ax.xaxis.set_minor_locator(LogLocator(base=2.0, subs=np.arange(1.25, 2.0, 0.25)))
ax.xaxis.set_minor_formatter(NullFormatter())

ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter_small_decimal))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax.yaxis.set_minor_formatter(NullFormatter())

if xlim_user is not None:
    ax.set_xlim(xlim_user)
else:
    ax.set_xlim(0.82 * N_min, 1.22 * N_max)

if ylim_user is not None:
    ax.set_ylim(ylim_user)
else:
    ymin = min(np.min(cpu_core_hours), np.min(y_ref))
    ymax = max(np.max(cpu_core_hours), np.max(y_ref))
    ax.set_ylim(0.55 * ymin, 1.8 * ymax)

ax.legend(loc="upper left", handlelength=2.2)
ax.set_title("Performance scaling of LYDION", fontsize=34.5, pad=18)

fig.tight_layout(pad=0.45)

if savefig:
    output_path = resolve_path(outname)
    fig.savefig(output_path, dpi=dpi_save, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")

if showfig:
    plt.show()
