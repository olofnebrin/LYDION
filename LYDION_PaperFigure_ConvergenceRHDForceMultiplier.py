import os

import h5py
import numpy as np
import matplotlib.pyplot as plt


###############################################################
#######                                                 #######
#######                  C O N S T A N T S              #######
#######                                                 #######
###############################################################

Lsun = 3.839e33
Msun = 1.989e33
kms  = 1.0e5

yr   = 3.1556926e7
kyr  = 1e3 * yr


###############################################################
#######                                                 #######
#######             U S E R   O P T I O N S             #######
#######                                                 #######
###############################################################

# Add/remove simulations here.
simulations = [
    {
        "filename"       : "RHDSim_StarCluster_Lya_128x128_2026-04-19.h5",
        "NoLya_filename" : "RHDSim_StarCluster_NoLya_128x128_2026-04-19.h5",
        "label"      : r"$128^2$",
        "color"      : "crimson",
        "linestyle"  : "-",
        "linewidth"  : 3.0,
        "alpha"      : 0.8,
    },
    {
        "filename"       : "RHDSim_StarCluster_Lya_181x181_2026-04-19.h5",
        "NoLya_filename" : "RHDSim_StarCluster_NoLya_181x181_2026-04-19.h5",
        "label"      : r"$181^2$",
        "color"      : "orange",
        "linestyle"  : "-",
        "linewidth"  : 3.0,
        "alpha"      : 0.8,
    },
    {
        "filename"       : "RHDSim_StarCluster_Lya_256x256_2026-04-19.h5",
        "NoLya_filename" : "RHDSim_StarCluster_NoLya_256x256_2026-04-19.h5",
        "label"      : r"$256^2$",
        "color"      : "limegreen",
        "linestyle"  : "-",
        "linewidth"  : 3.0,
        "alpha"      : 0.8,
    },
    {
        "filename"       : "RHDSim_StarCluster_Lya_362x362_2026-04-19.h5",
        "NoLya_filename" : "RHDSim_StarCluster_NoLya_362x362_2026-04-19.h5",
        "label"      : r"$362^2$",
        "color"      : "dodgerblue",
        "linestyle"  : "-",
        "linewidth"  : 3.0,
        "alpha"      : 0.8,
    },
    {
        "filename"       : "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5",
        "NoLya_filename" : "RHDSim_StarCluster_NoLya_512x512_2026-04-19.h5",
        "label"      : r"$512^2$",
        "color"      : "black",
        "linestyle"  : "-",
        "linewidth"  : 4.5,
        "alpha"      : 0.8,
    },
]

# Add the extra row with gas momentum + ionized volume fraction?
plot_extra_panels = True

# Overlay explicitly specified no-Lyα simulations in the panels where this is useful.
#
# Specify the no-Lyα filename directly in each simulation dictionary above
# using the key:
#
#     "NoLya_filename" : "your_file_name.h5"
#
# Set "NoLya_filename" : None for any resolution where you do not want
# a no-Lyα overlay.
#
# These are only plotted in:
#     f_V(x_HII > threshold)
#     P_gas,out
#
# using the same color as the Lyα simulation, but with dashed lines.
plot_NoLya_extra_panels = True
NoLya_linestyle = "--"
NoLya_linewidth_factor = 0.85
NoLya_alpha_factor = 0.85
skip_missing_NoLya_files = True

if plot_extra_panels:
    outfile = "LYDION_Figure_LyaForce_Convergence_6Panel.pdf"
else:
    outfile = "LYDION_Figure_LyaForce_Convergence_4Panel.pdf"

# Time range. Set to None to auto-use available range.
xlim = (-10.0, 160.0)

# Top row: M_F panels.
ylim_top = (0.1, 1e2)

# Middle row: percentage panels.
# This is shared by f_esc,Lya and f(xHII > threshold).
ylim_percent = (0.01, 500.0)

# Bottom row: gas momentum and L_Lya.
# Both panels share this y-range (set to None to auto-scale independently).
# Units: M_sun km/s for Pgas, L_Lya/L_Lya_norm for luminosity.
ylim_bottom = (10, 3e6)

# Normalization for the Lyα luminosity panel.
# The panel label reads "L_Lya / 10^4 L_sun", so norm = 1e4 * Lsun.
# Increase/decrease to make the L_Lya panel visually comparable
# to the gas momentum panel.
L_Lya_norm = 10 * Lsun

# Normalization for gas momentum.
# This plots momentum in M_sun km/s.
P_gas_norm = Msun * kms

# If true:
#     P_gas_out = sum dV * max(ρu · rhat, 0)
#
# If false:
#     P_gas_rad = sum dV * (ρu · rhat)
use_positive_radial_momentum_only = True

# Ionized volume threshold.
xHII_threshold = 0.5

# Legend options.
#
# Valid legend_panel names for 6-panel figure:
#   "MF_rad", "MF_anti", "fesc", "fion", "Pgas", "L_Lya"
#
# Valid legend_panel names for 4-panel figure:
#   "MF_rad", "MF_anti", "fesc", "L_Lya"
legend_panel  = "L_Lya"
legend_loc    = "lower left"
legend_bbox_to_anchor = (0.03, 0.05)
legend_fontsize = 26
legend_ncol = 2

# Panel-label text options.
show_panel_labels = True
panel_label_fontsize = 29

# Default panel-label position, used if a panel is not listed below.
panel_label_x_default = 0.07
panel_label_y_default = 0.85

# Individual panel-label positions.
# Coordinates are in axis coordinates.
panel_label_positions = {
    "MF_rad"  : (0.07, 0.85),
    "MF_anti" : (0.07, 0.85),
    "fesc"    : (0.68, 0.10),
    "fion"    : (0.55, 0.10),
    "Pgas"    : (0.07, 0.75),
    "L_Lya"   : (0.07, 0.85),
}

# Figure layout.
fig_width = 18.5
if plot_extra_panels:
    fig_height = 19.5
else:
    fig_height = 13.5

wspace = 0.03
hspace = 0.03

# Whether to use log y axes.
use_log_MF_rad    = True
use_log_MF_anti   = True
use_log_fesc_Lya  = True
use_log_fion      = True
use_log_Pgas_out  = True
use_log_L_Lya     = True

# If using log axes, non-positive values are masked automatically.
mask_nonpositive_for_log = True

# Hide y tick labels in the right column to reduce clutter.
hide_right_column_y_ticklabels = True

# Print maximum evolved times for the simulations used in this plot.
# This is useful for tables in the paper draft.
print_max_evolved_times = True
print_NoLya_max_evolved_times = True


###############################################################
#######                                                 #######
#######           P L O T   C O S M E T I C S           #######
#######                                                 #######
###############################################################

# COMMENT THIS OUT IF YOU DON'T HAVE THE REQUIRED FONTS:

#plt.rc('font',**{'sans-serif':['RomanD'],
                 #'size': 27} )
#plt.rc('font',**{'sans-serif':['AVHershey Simplex'],
                #'size': 27} )
plt.rc('font',**{'sans-serif':['AVHershey Complex'],
                'size': 35} )
#plt.rc('font',**{'sans-serif':['Complex'],
                # 'size': 27} )

params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

# The stuff below need not be removed:

params = {'legend.fontsize': 19 }
plt.rcParams.update(params)

plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 1.7
plt.rcParams['xtick.minor.width'] = 1.7
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['ytick.minor.size'] = 8
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'

plt.rcParams['legend.frameon'] = 'False'

plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linewidth'] = 2.5
plt.rcParams['grid.linestyle'] = '--'

# Make ticks appear on all sides and point inwards.
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True


###############################################################
#######                                                 #######
#######                  F U N C T I O N S              #######
#######                                                 #######
###############################################################

def get_NoLya_plot_style(sim):
    """Return plotting style for the explicitly specified no-Lyα curve.

    The no-Lyα filename is read from sim["NoLya_filename"].
    If this is None or missing, no no-Lyα curve is plotted for this simulation.
    """

    no_lya_filename = sim.get("NoLya_filename", None)

    if no_lya_filename is None:
        return None

    no_lya_sim = dict(sim)

    no_lya_sim["filename"]  = no_lya_filename
    no_lya_sim["label"]     = "_nolegend_"
    no_lya_sim["linestyle"] = NoLya_linestyle
    no_lya_sim["linewidth"] = NoLya_linewidth_factor * sim["linewidth"]
    no_lya_sim["alpha"]     = NoLya_alpha_factor * sim["alpha"]

    return no_lya_sim


def file_exists_or_warn(filename):
    """Return True if filename exists, otherwise warn or raise."""

    if os.path.exists(filename):
        return True

    msg = f"WARNING: Could not find file: {filename}"

    if skip_missing_NoLya_files:
        print(msg)
        print("         Skipping this no-Lyα overlay.")
        return False

    raise FileNotFoundError(msg)


def get_valid_time_slice(t):
    """Return slice selecting saved, non-empty time entries."""

    t = np.asarray(t)
    nonzero = np.nonzero(t)[0]

    if len(nonzero) > 0:
        last_idx = nonzero[-1]
        return slice(0, last_idx + 1)

    return slice(None)


def get_max_evolved_time(filename):
    """Return the maximum evolved time saved in filename.

    Uses the same non-empty time-entry convention as read_history(), so
    trailing zero-filled entries in preallocated output arrays are ignored.
    The returned value is in code/CGS seconds.
    """

    with h5py.File(filename, "r") as file:

        t  = np.asarray(file["Times"][:])
        sl = get_valid_time_slice(t)

        t_valid = t[sl]

    if len(t_valid) == 0:
        return np.nan

    return np.nanmax(t_valid)


def print_max_evolved_time_table():
    """Print maximum evolved times for all simulations in the plot list."""

    print()
    print("Maximum evolved times:")
    print("{:<14s}  {:<8s}  {:>14s}  {:>14s}  {}".format(
        "Resolution",
        "Run",
        "t_max [kyr]",
        "t_max [s]",
        "Filename"
    ))
    print("-" * 88)

    for sim in simulations:

        rows = [("Lyα", sim["filename"])]

        if print_NoLya_max_evolved_times:

            no_lya_filename = sim.get("NoLya_filename", None)

            if no_lya_filename is not None:
                rows.append(("No Lyα", no_lya_filename))

        for run_label, filename in rows:

            if not os.path.exists(filename):
                print("{:<14s}  {:<8s}  {:>14s}  {:>14s}  {}".format(
                    sim["label"],
                    run_label,
                    "missing",
                    "missing",
                    filename
                ))
                continue

            t_max = get_max_evolved_time(filename)

            print("{:<14s}  {:<8s}  {:14.6f}  {:14.6e}  {}".format(
                sim["label"],
                run_label,
                t_max / kyr,
                t_max,
                filename
            ))

    print()


def read_time_dataset(file, name, sl, Nt_raw, Nr, NZ):
    """Read a time-history dataset.

    Returns 3D histories in physical LYDION order:

        (Nr, NZ, Nt)

    Julia save convention in SaveHDF5Data_RHD_StarCluster.jl:

        dataset[:, :, step] = field

    i.e. physically:

        (N_r, N_Z, Nsaved)

    However, when reading Julia/HDF5 output in Python/h5py, the
    visible NumPy shape can appear reversed as:

        (Nsaved, N_Z, N_r)

    This function handles both possibilities.

    Important:
    - For Python-visible (Nsaved, N_Z, N_r), use transpose (2,1,0),
      not moveaxis(0,-1), otherwise r and Z are swapped.
    """

    arr = file[name][:]

    # 1D time histories, e.g. M_F, luminosity, escape fraction.
    if arr.ndim == 1:
        return arr[sl]

    if arr.ndim == 3:

        # Case A: Python-visible reversed order from Julia/HDF5:
        #
        #     (Nsaved, N_Z, N_r)
        #
        # Convert to:
        #
        #     (N_r, N_Z, Nt)
        if arr.shape == (Nt_raw, NZ, Nr):
            arr = arr[sl, :, :]
            return np.transpose(arr, (2, 1, 0))

        # Case B: Already visible in physical order:
        #
        #     (N_r, N_Z, Nsaved)
        #
        # Keep as:
        #
        #     (N_r, N_Z, Nt)
        if arr.shape == (Nr, NZ, Nt_raw):
            return arr[:, :, sl]

        # Case C: Some files/readers may expose:
        #
        #     (Nsaved, N_r, N_Z)
        #
        # This is not expected for your Julia HDF5 files, but is useful
        # if a test file was written directly from Python.
        if arr.shape == (Nt_raw, Nr, NZ):
            arr = arr[sl, :, :]
            return np.transpose(arr, (1, 2, 0))

        raise ValueError(
            f"Could not identify axes for dataset '{name}'. "
            f"Dataset shape is {arr.shape}. Expected one of "
            f"({Nt_raw}, {NZ}, {Nr}), ({Nr}, {NZ}, {Nt_raw}), "
            f"or ({Nt_raw}, {Nr}, {NZ})."
        )

    raise ValueError(f"Expected 1D or 3D dataset for '{name}', got shape {arr.shape}")


def read_static_2d_dataset(file, name, Nr, NZ):
    """Read a static 2D grid dataset and return physical order (Nr, NZ)."""

    arr = file[name][:]

    if arr.shape == (Nr, NZ):
        return arr

    if arr.shape == (NZ, Nr):
        return arr.T

    raise ValueError(
        f"Unexpected shape for static dataset '{name}': {arr.shape}. "
        f"Expected ({Nr}, {NZ}) or ({NZ}, {Nr})."
    )


def read_history(filename, read_extra=False, read_lya=True):
    """Read the quantities needed for the convergence plot.

    read_lya=True is used for Lyα-RHD files and reads Lyα-specific
    diagnostics such as M_F, L_Lya, and f_esc,Lya.

    read_lya=False is used for NoLyα files, which do not contain those
    datasets. In that case only the hydrodynamical/ionization quantities
    needed for the dashed comparison curves are read.
    """

    with h5py.File(filename, "r") as file:
        data = {}

        t      = file["Times"][:]
        Nt_raw = len(t)
        sl     = get_valid_time_slice(t)

        rCoM = file["rCoM"][:]
        ZCoM = file["ZCoM"][:]

        Nr = len(rCoM)
        NZ = len(ZCoM)

        data["t"] = t[sl]

        if read_lya:

            data["M_F_rad"] = read_time_dataset(
                file,
                "M_F_radial_net_History",
                sl,
                Nt_raw,
                Nr,
                NZ
            )

            data["M_F_anti"] = read_time_dataset(
                file,
                "M_F_History_antigrav_direction",
                sl,
                Nt_raw,
                Nr,
                NZ
            )

            data["L_Lya"] = read_time_dataset(
                file,
                "L_Lya_History",
                sl,
                Nt_raw,
                Nr,
                NZ
            )

            data["f_esc_Lya"] = read_time_dataset(
                file,
                "f_esc_Lya_History",
                sl,
                Nt_raw,
                Nr,
                NZ
            )

        if read_extra:

            data["rMom"] = read_time_dataset(
                file,
                "rMomentum_History",
                sl,
                Nt_raw,
                Nr,
                NZ
            )

            data["ZMom"] = read_time_dataset(
                file,
                "ZMomentum_History",
                sl,
                Nt_raw,
                Nr,
                NZ
            )

            data["xHII"] = read_time_dataset(
                file,
                "xHII_History",
                sl,
                Nt_raw,
                Nr,
                NZ
            )

            # Static grid quantities.
            data["CellVol"] = read_static_2d_dataset(file, "CellVolumes", Nr, NZ)
            data["rCoM"]    = rCoM
            data["ZCoM"]    = ZCoM

    if read_extra:
        data["P_gas_out"] = compute_radial_gas_momentum_history(data)
        data["f_ion_vol"] = compute_ionized_volume_fraction_history(data)

    return data


def compute_radial_gas_momentum_history(data):
    """Compute total gas momentum directed radially away from (r,Z)=(0,0).

    Uses the spherical radial direction in the meridional plane:

        rhat = (r, Z) / sqrt(r^2 + Z^2)

    Momentum density projected along rhat:

        p_rad = ρu_r * rhat_r + ρu_Z * rhat_Z

    If use_positive_radial_momentum_only = True, only outward-moving
    gas contributes:

        p_rad -> max(p_rad, 0).
    """

    rCoM    = np.asarray(data["rCoM"])
    ZCoM    = np.asarray(data["ZCoM"])
    CellVol = np.asarray(data["CellVol"])

    r2D, Z2D = np.meshgrid(rCoM, ZCoM, indexing="ij")

    R_sph = np.sqrt(r2D*r2D + Z2D*Z2D)

    rhat_r = np.zeros_like(R_sph)
    rhat_Z = np.zeros_like(R_sph)

    good = R_sph > 0.0
    rhat_r[good] = r2D[good] / R_sph[good]
    rhat_Z[good] = Z2D[good] / R_sph[good]

    rMom = np.asarray(data["rMom"])
    ZMom = np.asarray(data["ZMom"])

    # Expected shape after read_time_dataset:
    #     (Nr, NZ, Nt)
    if rMom.shape[:2] != CellVol.shape:
        raise ValueError(
            f"Shape mismatch in radial momentum diagnostic: "
            f"rMom[:, :, 0] has shape {rMom.shape[:2]}, "
            f"CellVol has shape {CellVol.shape}."
        )

    Nt = rMom.shape[2]
    P_rad = np.zeros(Nt)

    for n in range(Nt):

        p_rad_density = rMom[:, :, n] * rhat_r + ZMom[:, :, n] * rhat_Z

        if use_positive_radial_momentum_only:
            p_rad_density = np.maximum(p_rad_density, 0.0)

        P_rad[n] = np.sum(CellVol * p_rad_density)

    return P_rad


def compute_ionized_volume_fraction_history(data):
    """Compute volume fraction of cells with xHII > xHII_threshold.

    This is volume-weighted, not mass-weighted:

        f_ion_vol = sum_{xHII > threshold} dV / sum dV
    """

    xHII    = np.asarray(data["xHII"])
    CellVol = np.asarray(data["CellVol"])

    # Expected shape after read_time_dataset:
    #     (Nr, NZ, Nt)
    if xHII.shape[:2] != CellVol.shape:
        raise ValueError(
            f"Shape mismatch in ionized volume diagnostic: "
            f"xHII[:, :, 0] has shape {xHII.shape[:2]}, "
            f"CellVol has shape {CellVol.shape}."
        )

    Nt = xHII.shape[2]
    f_ion_vol = np.zeros(Nt)

    total_volume = np.sum(CellVol)

    for n in range(Nt):

        mask = xHII[:, :, n] > xHII_threshold

        if total_volume > 0.0:
            f_ion_vol[n] = np.sum(CellVol[mask]) / total_volume
        else:
            f_ion_vol[n] = np.nan

    return f_ion_vol


def clean_for_log(x, y):
    """Mask non-positive values before plotting on a log axis."""

    x = np.asarray(x)
    y = np.asarray(y)

    if mask_nonpositive_for_log:
        good = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    else:
        good = np.isfinite(x) & np.isfinite(y)

    return x[good], y[good]


def plot_history(ax, t, y, sim, use_log=True):
    """Plot one history curve with your preferred line settings."""

    if use_log:
        t_plot, y_plot = clean_for_log(t / kyr, y)
        ax.semilogy(t_plot, y_plot,
                    linewidth = sim["linewidth"],
                    color = sim["color"],
                    linestyle = sim["linestyle"],
                    alpha = sim["alpha"],
                    dash_capstyle = 'round',
                    label = sim["label"])
    else:
        ax.plot(t / kyr, y,
                linewidth = sim["linewidth"],
                color = sim["color"],
                linestyle = sim["linestyle"],
                alpha = sim["alpha"],
                dash_capstyle = 'round',
                label = sim["label"])


def format_axis(ax):
    """Apply common tick formatting."""

    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in',
                   length=20, width=2.5)
    ax.tick_params(axis='both', which='minor', direction='in',
                   length=10, width=2.5)

    for spine in ax.spines.values():
        spine.set_linewidth(2.5)


def get_panel_label_position(panel_name):
    """Return panel-label position for a given named panel."""

    if panel_name in panel_label_positions:
        return panel_label_positions[panel_name]

    return panel_label_x_default, panel_label_y_default


def add_panel_label(ax, panel_name, label):
    """Add panel label using panel-specific placement."""

    x, y = get_panel_label_position(panel_name)

    ax.text(x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=panel_label_fontsize,
            color="black")


def format_log_tick(val):
    """Format a power-of-10 tick value.

    Rules:
      - Non-integer powers (e.g. 0.3): use plain decimal.
      - Integer powers with |exponent| < 3: plain integer ('0.1', '1', '10', '100').
      - Integer powers with |exponent| >= 3: exponential, e.g. '10^3', '10^{-3}'.
    """

    import math

    if val <= 0:
        return ''

    log_val = math.log10(val)
    k = round(log_val)

    if abs(log_val - k) > 1e-9:
        return f'{val:g}'

    if abs(k) < 1.9:
        # Return value if 10^k < 10^1.9, so 100 is rendered as 10^2
        return f'{val:g}'
    else:
        return r'$10^{' + str(k) + r'}$'


def set_log_ticks(ax, ylim):
    """Set major log ticks and labels for every decade within ylim = (ymin, ymax).

    Always called BEFORE set_ylim so that matplotlib cannot expand the axis
    to accommodate out-of-range ticks.
    """

    import math

    ymin, ymax = ylim

    k_min = math.ceil(math.log10(ymin)  - 1e-9)
    k_max = math.floor(math.log10(ymax) + 1e-9)

    ticks  = [10.0**k for k in range(k_min, k_max + 1)]
    labels = [format_log_tick(v) for v in ticks]

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


###############################################################
#######                                                 #######
#######               P L O T  I T !                    #######
#######                                                 #######
###############################################################

if print_max_evolved_times:
    print_max_evolved_time_table()

if plot_extra_panels:
    fig, ax = plt.subplots(3, 2, sharex=True)
else:
    fig, ax = plt.subplots(2, 2, sharex=True)

fig.set_size_inches(fig_width, fig_height)

# Panel layout.
if plot_extra_panels:

    ax_MF_rad   = ax[0, 0]
    ax_MF_anti  = ax[0, 1]

    ax_fesc_Lya = ax[1, 0]
    ax_fion     = ax[1, 1]

    ax_Pgas_out = ax[2, 0]
    ax_L_Lya    = ax[2, 1]

    panel_axes = {
        "MF_rad"  : ax_MF_rad,
        "MF_anti" : ax_MF_anti,
        "fesc"    : ax_fesc_Lya,
        "fion"    : ax_fion,
        "Pgas"    : ax_Pgas_out,
        "L_Lya"   : ax_L_Lya,
    }

else:

    ax_MF_rad   = ax[0, 0]
    ax_MF_anti  = ax[0, 1]
    ax_fesc_Lya = ax[1, 0]
    ax_L_Lya    = ax[1, 1]

    panel_axes = {
        "MF_rad"  : ax_MF_rad,
        "MF_anti" : ax_MF_anti,
        "fesc"    : ax_fesc_Lya,
        "L_Lya"   : ax_L_Lya,
    }


for sim in simulations:

    data = read_history(sim["filename"], read_extra=plot_extra_panels)
    t = data["t"]

    plot_history(ax_MF_rad,
                 t,
                 data["M_F_rad"],
                 sim,
                 use_log=use_log_MF_rad)

    plot_history(ax_MF_anti,
                 t,
                 data["M_F_anti"],
                 sim,
                 use_log=use_log_MF_anti)

    plot_history(ax_fesc_Lya,
                 t,
                 100.0 * data["f_esc_Lya"],
                 sim,
                 use_log=use_log_fesc_Lya)

    plot_history(ax_L_Lya,
                 t,
                 data["L_Lya"] / L_Lya_norm,
                 sim,
                 use_log=use_log_L_Lya)

    if plot_extra_panels:

        plot_history(ax_fion,
                     t,
                     100.0 * data["f_ion_vol"],
                     sim,
                     use_log=use_log_fion)

        plot_history(ax_Pgas_out,
                     t,
                     data["P_gas_out"] / P_gas_norm,
                     sim,
                     use_log=use_log_Pgas_out)

        if plot_NoLya_extra_panels:

            no_lya_sim = get_NoLya_plot_style(sim)

            if no_lya_sim is not None and file_exists_or_warn(no_lya_sim["filename"]):

                no_lya_data = read_history(no_lya_sim["filename"],
                                           read_extra=True,
                                           read_lya=False)
                no_lya_t = no_lya_data["t"]

                plot_history(ax_fion,
                             no_lya_t,
                             100.0 * no_lya_data["f_ion_vol"],
                             no_lya_sim,
                             use_log=use_log_fion)

                plot_history(ax_Pgas_out,
                             no_lya_t,
                             no_lya_data["P_gas_out"] / P_gas_norm,
                             no_lya_sim,
                             use_log=use_log_Pgas_out)


###############################################################
#######                                                 #######
#######              A X I S   L A B E L S              #######
#######                                                 #######
###############################################################

# No y-axis labels, to save space.
if plot_extra_panels:
    ax_Pgas_out.set_xlabel("Time (kyr)", labelpad=20)
    ax_L_Lya.set_xlabel("Time (kyr)", labelpad=20)
else:
    ax_fesc_Lya.set_xlabel("Time (kyr)", labelpad=20)
    ax_L_Lya.set_xlabel("Time (kyr)", labelpad=20)

if xlim is not None:
    for axis in ax.flatten():
        axis.set_xlim(xlim)

# Set ticks FIRST, then set_ylim.  Calling set_ylim before set_yticks lets
# matplotlib expand the axis to accommodate any out-of-range ticks, which
# overrides the requested limits.

# Top row.
if ylim_top is not None:
    if use_log_MF_rad:
        set_log_ticks(ax_MF_rad, ylim_top)
    if use_log_MF_anti:
        set_log_ticks(ax_MF_anti, ylim_top)
    ax_MF_rad.set_ylim(ylim_top)
    ax_MF_anti.set_ylim(ylim_top)

# Percent row.
if ylim_percent is not None:
    if use_log_fesc_Lya:
        set_log_ticks(ax_fesc_Lya, ylim_percent)
    ax_fesc_Lya.set_ylim(ylim_percent)

    if plot_extra_panels:
        if use_log_fion:
            set_log_ticks(ax_fion, ylim_percent)
        ax_fion.set_ylim(ylim_percent)

# Bottom row.
if plot_extra_panels:
    if ylim_bottom is not None:
        if use_log_Pgas_out:
            set_log_ticks(ax_Pgas_out, ylim_bottom)
        if use_log_L_Lya:
            set_log_ticks(ax_L_Lya, ylim_bottom)
        ax_Pgas_out.set_ylim(ylim_bottom)
        ax_L_Lya.set_ylim(ylim_bottom)
else:
    # In the 4-panel version, L_Lya is still in the bottom row.
    if ylim_bottom is not None:
        if use_log_L_Lya:
            set_log_ticks(ax_L_Lya, ylim_bottom)
        ax_L_Lya.set_ylim(ylim_bottom)


###############################################################
#######                                                 #######
#######             P A N E L   L A B E L S             #######
#######                                                 #######
###############################################################

if show_panel_labels:

    add_panel_label(ax_MF_rad,
                    "MF_rad",
                    r"$M_{F,radial}$")

    add_panel_label(ax_MF_anti,
                    "MF_anti",
                    r"$M_{F,antigrav}$")

    add_panel_label(ax_fesc_Lya,
                    "fesc",
                    r"$f_{esc,Ly\alpha}$ (%)")

    add_panel_label(ax_L_Lya,
                    "L_Lya",
                    r"$L_{Ly\alpha}$ (10 L$_{\odot}$)")

    if plot_extra_panels:

        add_panel_label(ax_fion,
                        "fion",
                        r"f$_{V}$(x$_{HII}$ > 0.5) (%)")

        if use_positive_radial_momentum_only:
            Pgas_label = "Outward radial momentum" + "\n" + r"(M$_{\odot}$ km s$^{-1}$)"
        else:
            Pgas_label = r"$P_{gas,rad}$ (M$_{\odot}$ km s$^{-1}$)"

        add_panel_label(ax_Pgas_out,
                        "Pgas",
                        Pgas_label)


###############################################################
#######                                                 #######
#######          T I C K S,  L E G E N D,  S A V E      #######
#######                                                 #######
###############################################################

for axis in ax.flatten():
    format_axis(axis)

if hide_right_column_y_ticklabels:

    ax_MF_anti.tick_params(axis='y', which='both', labelleft=False)

    if plot_extra_panels:
        ax_fion.tick_params(axis='y', which='both', labelleft=False)
        ax_L_Lya.tick_params(axis='y', which='both', labelleft=False)
    else:
        ax_L_Lya.tick_params(axis='y', which='both', labelleft=False)

# Single legend, placed in the user-chosen panel.
if legend_panel not in panel_axes:
    valid_names = ", ".join(panel_axes.keys())
    raise ValueError(
        f"legend_panel = '{legend_panel}' is not valid. "
        f"Valid options are: {valid_names}."
    )

legend_axis = panel_axes[legend_panel]

if legend_bbox_to_anchor is None:
    legend_axis.legend(loc=legend_loc,
                       frameon=False,
                       fontsize=legend_fontsize,
                       ncol=legend_ncol)
else:
    legend_axis.legend(loc=legend_loc,
                       bbox_to_anchor=legend_bbox_to_anchor,
                       frameon=False,
                       fontsize=legend_fontsize,
                       ncol=legend_ncol)

plt.subplots_adjust(wspace=wspace, hspace=hspace)

plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.show()