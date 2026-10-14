import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter, FixedLocator, FixedFormatter
from matplotlib.lines import Line2D
import os

repo_root = Path(__file__).resolve().parents[1]
os.chdir(repo_root)

# ===============================================================
#                       Constants
# ===============================================================

yr  = 3.1556926e7
kyr = 1e3 * yr
Myr = 1e6 * yr
pc  = 3.086e18
kpc = 1e3 * pc
kms = 1e5
kB  = 1.380649e-16
mH  = 1.6735575e-24   

# ===============================================================
#                       User inputs
# ===============================================================

filename = "DTypeTest_128x128_RSL0.01_sigmastar3.0_OTStrue.h5"
outname  = "DTypeTest_Profiles.pdf"
savefig  = True

# Box normalization used by Iliev / C2-Ray style plots.
L_box_pc = 15.0e3

# Times to plot. The closest available saved LYDION snapshots are used.
target_times_Myr = [10.0, 200.0, 500.0]

# Optional pressure panel.
plot_pressure = True
pressure_dataset_name = "GasPressure_History"

# Density dataset.
density_dataset_name = "GasDensity_History"

# Panel order.
# If plot_pressure=False, "P" is automatically removed.
panel_order = ["xHI", "P", "T", "nH"]

# ---------------------------------------------------------------
# Optional Arepo-RT comparison from Rahul Kannan.
# ---------------------------------------------------------------

plot_arepo_rt = True
arepo_rt_filename = "ArepoRT_Test5.txt"

# Rahul's profile file appears to store the y-values as log10(quantity),
# while the first column is r/L_box. Keep this True unless the file format
# changes.
arepo_rt_y_values_are_log10 = True

# Map quantity names in the Arepo-RT text file to the panel names used here.
arepo_rt_quantity_map = {
    "HI_Fraction": "xHI",
    "Gas Density [1/cc]": "nH",
    "Pressure [g/cm/s^2]": "P",
    "Temperature [K]": "T",
}

# ---------------------------------------------------------------
# Optional C2-Ray comparison.
# ---------------------------------------------------------------

plot_c2ray = True

# Provide one C2-Ray file per comparison time.
# If you only have the 500 Myr C2-Ray file, leave only 500.0 here.
c2ray_files_by_time_Myr = {
    500.0: "c2ray5.txt",
    10.0:  "c2ray1.txt",
    200.0: "c2ray4.txt",
}

# Column choices are 1-based, matching your comment in the C2-Ray txt file:
# 1: r/L_box
# 2: xHI
# 4: P
# 6: T
#
# C2-Ray columns. Density is not stored in the files I checked, so by
# default we derive it from P, T, and xHI assuming pure hydrogen:
#
#     P = n_particles kB T = nH (2 - xHI) kB T.
#
# If the pressure column is instead P/kB in units of K cm^-3, set
# c2ray_pressure_is_P_over_kB=True. If you later identify a reliable density
# column and prefer to use it directly, set c2ray_derive_nH_from_PT=False and
# set c2ray_col_nH to the 1-based density column.
c2ray_col_r   = 1
c2ray_col_xHI = 2
c2ray_col_P   = 4
c2ray_col_T   = 6
c2ray_col_nH  = None

c2ray_derive_nH_from_PT = True
c2ray_pressure_is_P_over_kB = False

# Line styles.
colors_by_time = {
    10.0:  "black",
    200.0: "orangered",
    500.0: "dodgerblue",
}

default_time_colors = ["tab:green", "tab:orange", "dodgerblue", "tab:red", "tab:purple"]

lw_LYDION   = 4.0
lw_AREPO_RT = 3.5
lw_C2RAY    = 3.5

ls_LYDION  = "-"
ls_AREPO_RT = ":"
ls_C2RAY   = "--"

alpha_LYDION  = 1.0
alpha_AREPO_RT = 0.95
alpha_C2RAY   = 0.95

show_legend = True
legend_loc = "best"
legend_fontsize = 25

# Legend style.
#
# If True, the legend is a compact method key only, independent of time:
#     solid = LYDION, dotted = Arepo-RT, dashed = C2-Ray.
# This avoids repeated entries like "LYDION, 10 Myr", "LYDION, 200 Myr", etc.
# The physical times are instead encoded by line color. Optional manual time
# labels can be drawn with the controls below.
legend_method_only = True
legend_method_color = "black"
legend_lw_factor = 0.9

# Optional in-panel time labels.
# These are intentionally independent of the method legend above.
# Set show_time_labels=True and adjust the positions if you want explicit
# colored time annotations inside one or more panels. Coordinates are in
# axes-fraction units.
show_time_labels = True
time_label_panels = ["xHI"]
time_label_positions_by_time = {
    10.0:  (0.07, 0.7),
    200.0: (0.26, 0.7),
    500.0: (0.53, 0.7),
}
time_label_format = r"{time:g} Myr"
time_label_fontsize = 24
time_label_stroke_width = 0.0
time_label_stroke_color = "white"

# Time-label text alignment / rotation controls.
# Use e.g. time_label_rotation = 35 or 90 if the labels are cramped.
time_label_rotation = 0.0
time_label_rotation_mode = "anchor"
time_label_ha = "left"
time_label_va = "top"

# Optional per-time overrides. Leave empty to use the global values above.
# Example:
# time_label_rotation_by_time = {10.0: 0.0, 200.0: 35.0, 500.0: 35.0}
time_label_rotation_by_time = {10.0: 81.0, 200.0: 63.0, 500.0: 50.0}
time_label_ha_by_time = {}
time_label_va_by_time = {}

# Source position. If None, tries to read Settings["Z_pos"], otherwise uses 0.
Z_source_pc = 0.0

# Radial bin width for spherical profiles.
# If None, uses roughly one spatial cell width.
radial_bin_width_pc = None

# Use only complete spherical shells, i.e. avoid radii where the shell hits
# the edge of the 2D domain.
use_complete_shell_limit = True

# Spherical profile weighting.
#
# "volume" gives the true 3D axisymmetric volume average, i.e. it naturally
#          includes the sin(theta) weighting from cylindrical volumes.
# "angle" gives a polar-angle average at fixed radius, using each cell's
#         subtended angular width as weight. This can be useful for comparing
#         2D axisymmetric calculations where you want each direction to count
#         more equally rather than weighting equatorial material more strongly.
# "cell" gives each selected 2D cell equal weight within a radial shell.
profile_weighting = "angle"

# Optional smoothing of radial profiles.
# Set to 1 for no smoothing.
profile_smoothing_window = 1

# Optional axis limits.
xlim = (0.0, 1.0)

ylim_by_panel = {
    "xHI": (5e-5, 2.0),
    "P":   (2e-16, 1e-14),
    "T":   (500, 4e4),
    "nH":  (1e-4, 5e-3),
}

# ---------------------------------------------------------------
# Manual y-ticks / y-tick labels.
# Set to None to use Matplotlib default log tick labels.
# ---------------------------------------------------------------

yticks_by_panel = {
    "xHI": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "P":   None,
    "T":   None,
    "nH":  None,
}

yticklabels_by_panel = {
    "xHI": [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$0.1$", r"$1$"],
    "P":   None,
    "T":   None,
    "nH":  None,
}

# Axis labels.
xlabel = r"$r/L_{box}$"

# Manual x-ticks / x-tick labels.
# This avoids wide labels like 0.00 and 1.00 in compact multi-panel layouts.
# Set xticks = None to use Matplotlib defaults.
xticks = [0.0, 0.25, 0.5, 0.75, 1.0]
xticklabels = [r"$0$", r"$0.25$", r"$0.5$", r"$0.75$", r"$1$"]

# Optional rotation for x tick labels.
xtick_label_rotation = 0.0
xtick_label_ha = "center"

# We remove external y-labels by default and use in-panel labels instead.
show_external_ylabels = False

ylabel_by_panel = {
    "xHI": r"$x_{HI}$",
    "P":   r"$P_{gas}$ (erg cm$^{-3}$)",
    "T":   r"$T$ (K)",
    "nH":  r"$n_{H}$ (cm$^{-3}$)",
}

panel_label_by_panel = {
    "xHI": r"$x_{HI}$",
    "P":   r"$P_{gas}$ (erg cm$^{-3}$)",
    "T":   r"$T$ (K)",
    "nH":  r"$n_{H}$ (cm$^{-3}$)",
}

show_panel_labels = True
panel_label_fontsize = 35

# User-controlled in-panel label positions.
panel_label_position_by_panel = {
    "xHI": (0.75, 0.91),
    "P":   (0.2, 0.91),
    "T":   (0.7, 0.91),
    "nH":  (0.07, 0.91),
}

# Figure layout.
# With four panels the default is now a compact 2 x 2 layout:
#   top row:    xHI, P
#   bottom row: T, nH
use_two_row_layout = True
plot_ncols = 2

figsize_by_npanel_1row = {
    2: (15.0, 6.5),
    3: (20.0, 6.5),
    4: (25.0, 6.5),
}
figsize_by_grid = {
    (2, 2): (15.0, 12.5),
}
# With right-column y tick labels moved to the right side, the horizontal
# spacing can be much smaller without label collisions.
wspace = 0.08
hspace = 0.05

# For multi-column layouts, put y tick labels of the rightmost column on the
# right. This is useful for compact 2x2 figures where each panel has its own
# y-scale.
right_column_yticks_on_right = True

# Minor tick options.
use_minor_ticks = True
x_minor_subdivisions = 5
log_minor_subs = np.arange(2, 10) * 0.1


# ===============================================================
#                    Plot cosmetics
# ===============================================================

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 30})
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

plt.rcParams.update({'legend.fontsize': legend_fontsize})
plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 1.7
plt.rcParams['xtick.minor.width'] = 1.7
plt.rcParams['xtick.major.size'] = 16
plt.rcParams['xtick.minor.size'] = 7
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size'] = 16
plt.rcParams['ytick.minor.size'] = 7
plt.rcParams['xtick.major.pad'] = '10'
plt.rcParams['ytick.major.pad'] = '10'

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


# ===============================================================
#                    Helper functions
# ===============================================================

def decode_attr(x):
    """Make HDF5 attributes robust to bytes/string/numeric storage."""
    if isinstance(x, bytes):
        return x.decode()
    return x


def read_attr_float(group, key, default=None):
    """Read a scalar attribute that may have been saved as string."""
    if key not in group.attrs:
        return default
    val = decode_attr(group.attrs[key])
    try:
        return float(val)
    except Exception:
        return default


def ensure_NtNrNz(arr, Nt, Nr, Nz, name="array"):
    """
    Return array in shape (Nt, Nr, Nz).

    LYDION save code often creates histories as (Nr, Nz, Nsaved),
    while some plotting scripts assume (Nt, Nr, Nz).
    """
    shape = arr.shape

    if shape == (Nt, Nr, Nz):
        return arr

    if shape == (Nr, Nz, Nt):
        return arr.transpose(2, 0, 1)

    if shape == (Nt, Nz, Nr):
        return arr.transpose(0, 2, 1)

    if shape == (Nz, Nr, Nt):
        return arr.transpose(2, 1, 0)

    raise ValueError(
        f"Unexpected shape for {name}: {shape}. "
        f"Expected one of (Nt,Nr,Nz)=({Nt},{Nr},{Nz}) or "
        f"(Nr,Nz,Nt)=({Nr},{Nz},{Nt})."
    )


def ensure_NrNz(arr, Nr, Nz, name="array"):
    """Return 2D array in shape (Nr, Nz)."""
    if arr.shape == (Nr, Nz):
        return arr
    if arr.shape == (Nz, Nr):
        return arr.T
    raise ValueError(f"Unexpected shape for {name}: {arr.shape}, expected ({Nr},{Nz}).")


def moving_average_nan(y, window):
    """Simple NaN-aware moving average for 1D profile smoothing."""
    if window <= 1:
        return y

    window = int(window)
    if window % 2 == 0:
        window += 1

    y_smooth = np.full_like(y, np.nan, dtype=float)
    half = window // 2

    for i in range(len(y)):
        i0 = max(0, i - half)
        i1 = min(len(y), i + half + 1)
        vals = y[i0:i1]
        good = np.isfinite(vals)
        if np.any(good):
            y_smooth[i] = np.mean(vals[good])

    return y_smooth


def spherical_average_profile(q, shell_index, shell_r_pc, weights, nbins):
    """
    Weighted spherical average of q in cylindrical 2D.

    q should have shape (Nr, Nz), and shell_index maps each cell to a
    spherical shell bin.
    """
    q_flat = q.ravel()
    w_flat = weights.ravel()
    b_flat = shell_index.ravel()

    good = (
        np.isfinite(q_flat)
        & np.isfinite(w_flat)
        & (w_flat > 0.0)
        & (b_flat >= 0)
        & (b_flat < nbins)
    )

    sum_w  = np.bincount(b_flat[good], weights=w_flat[good], minlength=nbins)
    sum_qw = np.bincount(b_flat[good], weights=q_flat[good] * w_flat[good], minlength=nbins)

    prof = np.full(nbins, np.nan)
    nonzero = sum_w > 0.0
    prof[nonzero] = sum_qw[nonzero] / sum_w[nonzero]

    return shell_r_pc, prof


def compute_cell_angle_widths(r_edges_pc, Z_edges_pc, Z_source_pc):
    """
    Approximate each 2D cell's polar-angle coverage around the source.

    The returned weights are Δθ per cell. Therefore, within each radial shell,
    averaging with these weights approximates an angle average rather than the
    true 3D volume average. This is intentionally different from volume
    weighting, which contains the physical sin(theta) factor.
    """
    Nr = len(r_edges_pc) - 1
    Nz = len(Z_edges_pc) - 1
    weights = np.zeros((Nr, Nz), dtype=float)

    for i in range(Nr):
        r0 = r_edges_pc[i]
        r1 = r_edges_pc[i + 1]

        for j in range(Nz):
            z0 = Z_edges_pc[j] - Z_source_pc
            z1 = Z_edges_pc[j + 1] - Z_source_pc

            # atan2(r, z) gives polar angle measured away from the +Z axis.
            # For r >= 0 this lies in [0, pi], so there is no periodic wrap.
            theta_corners = np.array([
                np.arctan2(r0, z0),
                np.arctan2(r1, z0),
                np.arctan2(r0, z1),
                np.arctan2(r1, z1),
            ])

            dtheta = np.nanmax(theta_corners) - np.nanmin(theta_corners)

            if np.isfinite(dtheta) and dtheta > 0.0:
                weights[i, j] = dtheta

    return weights


def positive_for_log(y):
    """Mask non-positive values for log plotting."""
    return np.where((np.isfinite(y)) & (y > 0.0), y, np.nan)


def choose_saved_index(t, target_time_Myr):
    """Choose closest saved output index."""
    target_t = target_time_Myr * Myr
    return int(np.argmin(np.abs(t - target_t)))


def make_effect_text_kwargs():
    return dict(
        ha="left",
        va="top",
        color="black",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")]
    )


def get_color_for_time(target_time_Myr, time_number):
    if target_time_Myr in colors_by_time:
        return colors_by_time[target_time_Myr]
    return default_time_colors[time_number % len(default_time_colors)]


def resolve_input_path(filename):
    """Resolve input files relative to the working directory or this script."""
    path = Path(filename)

    if path.exists():
        return path

    script_dir_path = Path(__file__).resolve().parents[1] / filename
    if script_dir_path.exists():
        return script_dir_path

    return path


def read_arepo_rt_profiles(
    arepo_filename,
    quantity_map,
    y_values_are_log10=True,
):
    """
    Read Rahul Kannan's Arepo-RT Test 5 profile file.

    Expected block format:

        Time =  10 Myrs
        HI_Fraction
        r_over_Lbox  log10_quantity
        ...

    The returned dictionary is keyed by time in Myr, then by panel name.
    Each panel entry contains x = r/L_box and y = quantity in linear units.
    """
    arepo_path = resolve_input_path(arepo_filename)

    with open(arepo_path, "r") as fp:
        raw_lines = [line.strip() for line in fp]

    profiles = {}
    i = 0

    while i < len(raw_lines):

        line = raw_lines[i]

        if (not line) or (not line.startswith("Time")):
            i += 1
            continue

        # Example: "Time =  10 Myrs"
        parts = line.replace("=", " ").split()
        try:
            time_Myr = float(parts[1])
        except Exception as exc:
            raise ValueError(f"Could not parse Arepo-RT time line: '{line}'") from exc

        i += 1

        # Skip accidental blank lines before the quantity name.
        while i < len(raw_lines) and not raw_lines[i]:
            i += 1

        if i >= len(raw_lines):
            break

        quantity_name = raw_lines[i]
        i += 1

        rows = []
        while i < len(raw_lines):

            row = raw_lines[i]

            if (not row) or row.startswith("Time"):
                break

            vals = row.split()
            if len(vals) >= 2:
                try:
                    rows.append((float(vals[0]), float(vals[1])))
                except ValueError:
                    pass

            i += 1

        if quantity_name in quantity_map and len(rows) > 0:
            panel_name = quantity_map[quantity_name]
            arr = np.asarray(rows, dtype=float)

            x = arr[:, 0]
            y = arr[:, 1]

            if y_values_are_log10:
                y = 10.0**y

            profiles.setdefault(time_Myr, {})[panel_name] = {
                "x": x,
                "y": y,
            }

        # Do not increment here if the current line is a Time line; the next
        # while-iteration should parse it. Blank lines are skipped above.

    return profiles


def derive_c2ray_nH_from_PT(P, T, xHI, pressure_is_P_over_kB=False):
    """Derive hydrogen nuclei density for pure H C2-Ray profiles.

    If pressure_is_P_over_kB=False, P is assumed to be thermal pressure in
    cgs units, erg cm^-3 = dyn cm^-2. If True, P is assumed to be P/kB in
    K cm^-3.
    """
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    xHI = np.asarray(xHI, dtype=float)

    denom = T * (2.0 - xHI)

    with np.errstate(divide="ignore", invalid="ignore"):
        if pressure_is_P_over_kB:
            nH = P / denom
        else:
            nH = P / (kB * denom)

    nH[(~np.isfinite(nH)) | (nH <= 0.0)] = np.nan

    return nH


def read_c2ray_profiles(
    c2ray_filename,
    col_r_1based,
    col_xHI_1based,
    col_P_1based,
    col_T_1based,
    col_nH_1based=None,
    derive_nH_from_PT=True,
    pressure_is_P_over_kB=False,
):
    """
    Read C2-Ray comparison data.

    Column convention is 1-based in the user inputs.
    Column 1 is assumed to already be r/L_box.

    By default nH is derived from P, T, and xHI for pure hydrogen.
    """
    data = np.loadtxt(c2ray_filename, comments="#")

    if data.ndim != 2:
        raise ValueError(f"C2-Ray file '{c2ray_filename}' was not read as a 2D table.")

    ncols = data.shape[1]

    required_cols = {
        "r":   col_r_1based,
        "xHI": col_xHI_1based,
        "P":   col_P_1based,
        "T":   col_T_1based,
    }

    if col_nH_1based is not None:
        required_cols["nH"] = col_nH_1based

    for name, col in required_cols.items():
        if col < 1 or col > ncols:
            raise ValueError(
                f"C2-Ray column for {name} is {col}, but file has only {ncols} columns."
            )

    c2ray = {
        "x":   data[:, col_r_1based   - 1],
        "xHI": data[:, col_xHI_1based - 1],
        "P":   data[:, col_P_1based   - 1],
        "T":   data[:, col_T_1based   - 1],
    }

    if derive_nH_from_PT:
        c2ray["nH"] = derive_c2ray_nH_from_PT(
            c2ray["P"],
            c2ray["T"],
            c2ray["xHI"],
            pressure_is_P_over_kB=pressure_is_P_over_kB,
        )
    elif col_nH_1based is not None:
        c2ray["nH"] = data[:, col_nH_1based - 1]

    return c2ray


def apply_manual_xticks(ax):
    """Apply compact/custom x tick labels."""
    if xticks is None:
        return

    ax.xaxis.set_major_locator(FixedLocator(xticks))

    if xticklabels is not None:
        if len(xticklabels) != len(xticks):
            raise ValueError(
                f"xticklabels has length {len(xticklabels)}, but xticks has length {len(xticks)}."
            )
        ax.xaxis.set_major_formatter(FixedFormatter(xticklabels))

    ax.tick_params(axis="x", labelrotation=xtick_label_rotation)

    for ticklabel in ax.get_xticklabels():
        ticklabel.set_ha(xtick_label_ha)


def apply_manual_yticks(ax, panel_name):
    ticks = yticks_by_panel.get(panel_name, None)
    labels = yticklabels_by_panel.get(panel_name, None)

    if ticks is None:
        return

    ax.yaxis.set_major_locator(FixedLocator(ticks))

    if labels is not None:
        ax.yaxis.set_major_formatter(FixedFormatter(labels))


def apply_y_tick_side(ax, is_right_column):
    """Place y tick labels on the outside of each subplot column."""
    if right_column_yticks_on_right and is_right_column:
        ax.yaxis.set_ticks_position("right")
        ax.yaxis.set_label_position("right")
        ax.tick_params(
            axis="y",
            which="both",
            left=True,
            right=True,
            labelleft=False,
            labelright=True,
        )
    else:
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_label_position("left")
        ax.tick_params(
            axis="y",
            which="both",
            left=True,
            right=True,
            labelleft=True,
            labelright=False,
        )


def add_time_labels(ax, panel_name):
    """Optionally draw colored time labels inside selected panels."""
    if not show_time_labels:
        return

    if panel_name not in time_label_panels:
        return

    for itime, target_time_Myr in enumerate(target_times_Myr):
        if target_time_Myr not in time_label_positions_by_time:
            continue

        x_text, y_text = time_label_positions_by_time[target_time_Myr]
        color = get_color_for_time(target_time_Myr, itime)
        label = time_label_format.format(time=target_time_Myr)

        rotation = time_label_rotation_by_time.get(target_time_Myr, time_label_rotation)
        ha = time_label_ha_by_time.get(target_time_Myr, time_label_ha)
        va = time_label_va_by_time.get(target_time_Myr, time_label_va)

        ax.text(
            x_text,
            y_text,
            label,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            rotation=rotation,
            rotation_mode=time_label_rotation_mode,
            color=color,
            fontsize=time_label_fontsize,
            path_effects=[
                pe.withStroke(
                    linewidth=time_label_stroke_width,
                    foreground=time_label_stroke_color,
                )
            ],
        )


def make_method_legend_handles():
    """Build a compact method-only legend with black line styles."""
    handles = [
        Line2D(
            [0], [0],
            color=legend_method_color,
            lw=legend_lw_factor * lw_LYDION,
            ls=ls_LYDION,
            alpha=alpha_LYDION,
            label="LYDION",
        )
    ]

    if plot_arepo_rt:
        handles.append(
            Line2D(
                [0], [0],
                color=legend_method_color,
                lw=legend_lw_factor * lw_AREPO_RT,
                ls=ls_AREPO_RT,
                alpha=alpha_AREPO_RT,
                dash_capstyle="round",
                label="Arepo-RT",
            )
        )

    if plot_c2ray:
        handles.append(
            Line2D(
                [0], [0],
                color=legend_method_color,
                lw=legend_lw_factor * lw_C2RAY,
                ls=ls_C2RAY,
                alpha=alpha_C2RAY,
                dash_capstyle="round",
                label="Capreole + C$^{2}$-Ray",
            )
        )

    return handles


# ===============================================================
#                         Read LYDION data
# ===============================================================

with h5py.File(filename, "r") as f:

    t = f["Times"][:]

    r = f["r"][:]        # cell interfaces, cm
    Z = f["Z"][:]        # cell interfaces, cm

    Nr = len(r) - 1
    Nz = len(Z) - 1
    Nt = len(t)

    if "rCoM" in f:
        rCoM = f["rCoM"][:]
    else:
        rCoM = 0.5 * (r[:-1] + r[1:])
        print("Warning: rCoM not found in file, using cell-centered r as fallback.")

    if "ZCoM" in f:
        ZCoM = f["ZCoM"][:]
    else:
        ZCoM = 0.5 * (Z[:-1] + Z[1:])
        print("Warning: ZCoM not found in file, using cell-centered Z as fallback.")

    if "CellVolumes" in f:
        CellVolumes = ensure_NrNz(f["CellVolumes"][:], Nr, Nz, name="CellVolumes")
    else:
        # Fallback for axisymmetric cylindrical cells.
        # The overall 2π factor cancels in the weighted average.
        dr2 = 0.5 * (r[1:]**2 - r[:-1]**2)
        dZ  = Z[1:] - Z[:-1]
        CellVolumes = np.outer(dr2, dZ)
        print("Warning: CellVolumes not found in file, using cylindrical fallback volumes.")

    T_hist = ensure_NtNrNz(
        f["T_History"][:], Nt, Nr, Nz, name="T_History"
    )

    xHII_hist = ensure_NtNrNz(
        f["xHII_History"][:], Nt, Nr, Nz, name="xHII_History"
    )

    xH2_hist = ensure_NtNrNz(
        f["xH2_History"][:], Nt, Nr, Nz, name="xH2_History"
    )

    if density_dataset_name not in f:
        raise KeyError(
            f"Density dataset '{density_dataset_name}' was not found in {filename}."
        )

    rho_hist = ensure_NtNrNz(
        f[density_dataset_name][:], Nt, Nr, Nz, name=density_dataset_name
    )

    xHI_hist = 1.0 - 2.0 * xH2_hist - xHII_hist

    if plot_pressure:
        if pressure_dataset_name not in f:
            raise KeyError(
                f"plot_pressure=True, but dataset '{pressure_dataset_name}' "
                f"was not found in {filename}."
            )

        P_hist = ensure_NtNrNz(
            f[pressure_dataset_name][:], Nt, Nr, Nz, name=pressure_dataset_name
        )
    else:
        P_hist = None

    if "Settings" in f and Z_source_pc is None:
        Z_pos_code = read_attr_float(f["Settings"], "Z_pos", default=None)
        if Z_pos_code is not None:
            Z_source_pc = Z_pos_code / pc

if Z_source_pc is None:
    Z_source_pc = 0.0


# Keep only actually saved outputs.
saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]

t_saved = t[saved_idx]
T_hist = T_hist[saved_idx, :, :]
xHII_hist = xHII_hist[saved_idx, :, :]
xH2_hist = xH2_hist[saved_idx, :, :]
xHI_hist = xHI_hist[saved_idx, :, :]
rho_hist = rho_hist[saved_idx, :, :]

if plot_pressure:
    P_hist = P_hist[saved_idx, :, :]


# ===============================================================
#          Construct spherical shell binning once
# ===============================================================

r_pc = r / pc
Z_pc = Z / pc
rCoM_pc = rCoM / pc
ZCoM_pc = ZCoM / pc

rr, zz = np.meshgrid(rCoM_pc, ZCoM_pc, indexing="ij")
R_sph_pc = np.sqrt(rr**2 + (zz - Z_source_pc)**2)

if radial_bin_width_pc is None:
    dr_med_pc = np.median(np.diff(r_pc))
    dZ_med_pc = np.median(np.diff(Z_pc))
    radial_bin_width_pc = min(dr_med_pc, dZ_med_pc)

if use_complete_shell_limit:
    r_extent_pc = np.max(r_pc)

    zmin_pc = np.min(Z_pc)
    zmax_pc = np.max(Z_pc)

    if (Z_source_pc > zmin_pc) and (Z_source_pc < zmax_pc):
        z_extent_pc = min(Z_source_pc - zmin_pc, zmax_pc - Z_source_pc)
    else:
        z_extent_pc = max(abs(zmax_pc - Z_source_pc), abs(Z_source_pc - zmin_pc))

    Rmax_profile_pc = min(r_extent_pc, z_extent_pc)
else:
    Rmax_profile_pc = np.nanmax(R_sph_pc)

shell_edges_pc = np.arange(0.0, Rmax_profile_pc + radial_bin_width_pc, radial_bin_width_pc)
shell_centers_pc = 0.5 * (shell_edges_pc[:-1] + shell_edges_pc[1:])
nbins = len(shell_centers_pc)

shell_index = np.digitize(R_sph_pc, shell_edges_pc) - 1
shell_index[(R_sph_pc < shell_edges_pc[0]) | (R_sph_pc >= shell_edges_pc[-1])] = -1

if profile_weighting == "volume":
    ProfileWeights = CellVolumes
elif profile_weighting == "angle":
    ProfileWeights = compute_cell_angle_widths(r_pc, Z_pc, Z_source_pc)
elif profile_weighting == "cell":
    ProfileWeights = np.ones_like(CellVolumes, dtype=float)
else:
    raise ValueError(
        "profile_weighting must be one of 'volume', 'angle', or 'cell'."
    )

x_plot = shell_centers_pc / L_box_pc


# ===============================================================
#             Compute LYDION profiles for all requested times
# ===============================================================

profiles_by_time = {}
actual_times_Myr = {}

for target_time_Myr in target_times_Myr:

    n = choose_saved_index(t_saved, target_time_Myr)
    actual_t_Myr = t_saved[n] / Myr

    actual_times_Myr[target_time_Myr] = actual_t_Myr

    xHI_2d = xHI_hist[n, :, :]
    T_2d   = T_hist[n, :, :]
    nH_2d  = rho_hist[n, :, :] / mH

    _, xHI_prof = spherical_average_profile(
        xHI_2d,
        shell_index,
        shell_centers_pc,
        ProfileWeights,
        nbins
    )

    _, T_prof = spherical_average_profile(
        T_2d,
        shell_index,
        shell_centers_pc,
        ProfileWeights,
        nbins
    )

    _, nH_prof = spherical_average_profile(
        nH_2d,
        shell_index,
        shell_centers_pc,
        ProfileWeights,
        nbins
    )

    xHI_prof = moving_average_nan(xHI_prof, profile_smoothing_window)
    T_prof   = moving_average_nan(T_prof, profile_smoothing_window)
    nH_prof  = moving_average_nan(nH_prof, profile_smoothing_window)

    profiles = {
        "xHI": xHI_prof,
        "T":   T_prof,
        "nH":  nH_prof,
    }

    if plot_pressure:
        P_2d = P_hist[n, :, :]

        _, P_prof = spherical_average_profile(
            P_2d,
            shell_index,
            shell_centers_pc,
            ProfileWeights,
            nbins
        )

        P_prof = moving_average_nan(P_prof, profile_smoothing_window)
        profiles["P"] = P_prof

    profiles_by_time[target_time_Myr] = profiles


# ===============================================================
#                    Read Arepo-RT profiles
# ===============================================================

arepo_rt_by_time = {}

if plot_arepo_rt:
    arepo_rt_by_time = read_arepo_rt_profiles(
        arepo_rt_filename,
        quantity_map=arepo_rt_quantity_map,
        y_values_are_log10=arepo_rt_y_values_are_log10,
    )


# ===============================================================
#                    Read C2-Ray profiles
# ===============================================================

c2ray_by_time = {}

if plot_c2ray:
    for c2ray_time_Myr, c2ray_filename in c2ray_files_by_time_Myr.items():
        c2ray_by_time[c2ray_time_Myr] = read_c2ray_profiles(
            c2ray_filename=resolve_input_path(c2ray_filename),
            col_r_1based=c2ray_col_r,
            col_xHI_1based=c2ray_col_xHI,
            col_P_1based=c2ray_col_P,
            col_T_1based=c2ray_col_T,
            col_nH_1based=c2ray_col_nH,
            derive_nH_from_PT=c2ray_derive_nH_from_PT,
            pressure_is_P_over_kB=c2ray_pressure_is_P_over_kB,
        )


# ===============================================================
#                         Plot
# ===============================================================

plot_panel_order = []
for panel_name in panel_order:
    if panel_name == "P" and not plot_pressure:
        continue
    plot_panel_order.append(panel_name)

n_panels = len(plot_panel_order)

if use_two_row_layout and n_panels > 2:
    ncols = int(plot_ncols)
    nrows = int(np.ceil(n_panels / ncols))
    figsize = figsize_by_grid.get((nrows, ncols), (7.5 * ncols, 6.2 * nrows))

    fig, axes_grid = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"wspace": wspace, "hspace": hspace}
    )

    axes_flat = np.asarray(axes_grid).ravel()

    for ax_extra in axes_flat[n_panels:]:
        ax_extra.set_visible(False)

    axes = list(axes_flat[:n_panels])
else:
    nrows = 1
    ncols = n_panels
    figsize = figsize_by_npanel_1row.get(n_panels, (6.3 * n_panels, 6.5))

    fig, axes_grid = plt.subplots(
        1,
        n_panels,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"wspace": wspace}
    )

    axes = list(np.atleast_1d(axes_grid))

axes_by_panel = dict(zip(plot_panel_order, axes))


for ipanel, panel_name in enumerate(plot_panel_order):

    ax = axes_by_panel[panel_name]
    panel_row = ipanel // ncols
    panel_col = ipanel % ncols
    is_bottom_row = panel_row == (nrows - 1)
    is_right_column = panel_col == (ncols - 1)

    # -----------------------
    # LYDION profiles
    # -----------------------
    for itime, target_time_Myr in enumerate(target_times_Myr):

        color = get_color_for_time(target_time_Myr, itime)
        y = positive_for_log(profiles_by_time[target_time_Myr][panel_name])

        actual_t_Myr = actual_times_Myr[target_time_Myr]

        ax.plot(
            x_plot,
            y,
            lw=lw_LYDION,
            ls=ls_LYDION,
            color=color,
            alpha=alpha_LYDION,
            label="_nolegend_" if legend_method_only else rf"LYDION, {actual_t_Myr:.1f} Myr"
        )

    # -----------------------
    # Arepo-RT profiles
    # -----------------------
    if plot_arepo_rt:
        for itime, target_time_Myr in enumerate(target_times_Myr):

            if target_time_Myr not in arepo_rt_by_time:
                continue

            if panel_name not in arepo_rt_by_time[target_time_Myr]:
                continue

            color = get_color_for_time(target_time_Myr, itime)
            arepo_rt = arepo_rt_by_time[target_time_Myr][panel_name]
            y_arepo_rt = positive_for_log(arepo_rt["y"])

            ax.plot(
                arepo_rt["x"],
                y_arepo_rt,
                lw=lw_AREPO_RT,
                ls=ls_AREPO_RT,
                dash_capstyle="round",
                color=color,
                alpha=alpha_AREPO_RT,
                label="_nolegend_" if legend_method_only else rf"Arepo-RT, {target_time_Myr:.0f} Myr"
            )

    # -----------------------
    # C2-Ray profiles
    # -----------------------
    if plot_c2ray:
        for itime, target_time_Myr in enumerate(target_times_Myr):

            if target_time_Myr not in c2ray_by_time:
                continue

            if panel_name not in c2ray_by_time[target_time_Myr]:
                continue

            color = get_color_for_time(target_time_Myr, itime)
            c2ray = c2ray_by_time[target_time_Myr]
            y_c2ray = positive_for_log(c2ray[panel_name])

            ax.plot(
                c2ray["x"],
                y_c2ray,
                lw=lw_C2RAY,
                ls=ls_C2RAY,
                dash_capstyle="round",
                color=color,
                alpha=alpha_C2RAY,
                label="_nolegend_" if legend_method_only else rf"C2-Ray, {target_time_Myr:.0f} Myr"
            )

    # -----------------------
    # Axis style
    # -----------------------
    ax.set_yscale("log")

    if is_bottom_row:
        ax.set_xlabel(xlabel, labelpad=14)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    if show_external_ylabels:
        ax.set_ylabel(ylabel_by_panel[panel_name], labelpad=14)
    else:
        ax.set_ylabel("")

    apply_y_tick_side(ax, is_right_column)

    if xlim is not None:
        ax.set_xlim(*xlim)

    apply_manual_xticks(ax)

    ylim = ylim_by_panel.get(panel_name, None)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        all_y = []

        for target_time_Myr in target_times_Myr:
            yy = positive_for_log(profiles_by_time[target_time_Myr][panel_name])
            good = np.isfinite(yy) & (yy > 0.0)
            if np.any(good):
                all_y.append(yy[good])

        if plot_arepo_rt:
            for target_time_Myr in target_times_Myr:
                if target_time_Myr not in arepo_rt_by_time:
                    continue
                if panel_name not in arepo_rt_by_time[target_time_Myr]:
                    continue

                yy = positive_for_log(arepo_rt_by_time[target_time_Myr][panel_name]["y"])
                good = np.isfinite(yy) & (yy > 0.0)
                if np.any(good):
                    all_y.append(yy[good])

        if plot_c2ray:
            for target_time_Myr in target_times_Myr:
                if target_time_Myr not in c2ray_by_time:
                    continue
                if panel_name not in c2ray_by_time[target_time_Myr]:
                    continue

                yy = positive_for_log(c2ray_by_time[target_time_Myr][panel_name])
                good = np.isfinite(yy) & (yy > 0.0)
                if np.any(good):
                    all_y.append(yy[good])

        if len(all_y) > 0:
            all_y = np.concatenate(all_y)
            ymin = np.nanmin(all_y)
            ymax = np.nanmax(all_y)

            if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                ax.set_ylim(ymin / 1.5, ymax * 1.5)

    ax.grid(False)

    if use_minor_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor_subdivisions))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=log_minor_subs))
        ax.yaxis.set_minor_formatter(NullFormatter())

    apply_manual_yticks(ax, panel_name)

    if show_panel_labels:
        label_x, label_y = panel_label_position_by_panel.get(panel_name, (0.07, 0.91))

        ax.text(
            label_x,
            label_y,
            panel_label_by_panel[panel_name],
            transform=ax.transAxes,
            fontsize=panel_label_fontsize,
            **make_effect_text_kwargs()
        )

    add_time_labels(ax, panel_name)


if show_legend:
    # Put one compact legend in the first panel.
    if legend_method_only:
        axes[0].legend(
            handles=make_method_legend_handles(),
            loc="lower right",
            bbox_to_anchor=(0.97, 0.03),
            fontsize=legend_fontsize,
        )
    else:
        axes[0].legend(
            loc="lower right",
            bbox_to_anchor=(0.97, 0.03),
            fontsize=legend_fontsize,
        )


# ===============================================================
#                         Save/show
# ===============================================================

if savefig:
    fig.savefig(outname, dpi=300, bbox_inches="tight")
else:
    plt.show()