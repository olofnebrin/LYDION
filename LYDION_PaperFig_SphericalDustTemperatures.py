import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter, FuncFormatter
from matplotlib.lines import Line2D


# ===============================================================
#                       Constants
# ===============================================================

yr  = 3.1556926e7
kyr = 1e3 * yr
Myr = 1e6 * yr
pc  = 3.085677581e18


# ===============================================================
#                       User inputs
# ===============================================================

#filename_lya    = "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5"
#filename_nolya  = "RHDSim_StarCluster_NoLya_512x512_2026-04-19.h5"

filename_lya   = "RHDSim_Lya_u0_256x256_2026-05-02.h5"
filename_nolya = "RHDSim_NoLya_u0_256x256_2026-05-02.h5"

outname = "LYDION_Figure_SphericalAverage_GasDustTemperatures.pdf"
savefig = True

# Closest saved snapshot is used in both files.
TargetTime_kyr = 30.0

# Optional figure title. Set to None for no title.
figure_title = r"Spherically averaged gas and dust temperatures"

# Source position in pc. If None, tries Settings["Z_pos"], otherwise uses 0.
Z_source_pc = None

# Radial bin width in pc. If None, use roughly one cell width.
radial_bin_width_pc = None

# Avoid spherical shells that hit the edge of the 2D domain.
use_complete_shell_limit = True

# Optional profile smoothing. Set to 1 for no smoothing.
profile_smoothing_window = 1

# Which dust temperatures to plot.
# Options: "Sil", "C", "PAH"
dust_types_to_plot = ["Sil", "C", "PAH"]

# Plot gas temperature too?
plot_gas_temperature = True

# Axes.
xlim_pc = (0.0, 0.1)          # e.g. (0.0, 5.0)
ylim_K  = (50, 5e4)

x_minor_subdivisions = 5
log_minor_subs = np.arange(2, 10) * 0.1

# Line style conventions.
lw = 4.0
alpha = 1.0

# Variable colors.
colors = {
    "Tgas":       "black",
    "Tdust_Sil":  "dodgerblue",
    "Tdust_C":    "orangered",
    "Tdust_PAH":  "limegreen",
}

labels = {
    "Tgas":       r"$T_{gas}$",
    "Tdust_Sil":  r"$T_{Sil}$",
    "Tdust_C":    r"$T_{C}$",
    "Tdust_PAH":  r"$T_{PAH}$",
}

dust_key_map = {
    "Sil": "Tdust_Sil",
    "C":   "Tdust_C",
    "PAH": "Tdust_PAH",
}

# Legend placement.
show_legend = True
legend_loc = "upper right"
legend_fontsize = 18

# In-panel labels.
show_panel_labels = True
panel_label_fontsize = 24
panel_label_pos = (0.05, 0.91)

# Figure layout.
figsize = (10.5, 15.0)
hspace = 0.05


# ===============================================================
#                    Plot cosmetics
# ===============================================================

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 28})
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
    if isinstance(x, bytes):
        return x.decode()
    return x


def read_attr_float(group, key, default=None):
    if key not in group.attrs:
        return default

    val = decode_attr(group.attrs[key])

    try:
        return float(val)
    except Exception:
        return default


def reorder_history_3d(arr, t, r_edges, Z_edges, name="array"):
    """
    Reorder a 3D history array to shape (Nt, Nr, Nz).
    This is robust to LYDION histories saved as either (Nt,Nr,Nz)
    or (Nr,Nz,Nt), etc.
    """
    if arr.ndim != 3:
        raise ValueError(f"{name}: expected 3D array, got shape {arr.shape}")

    Nt = t.size
    Nr = len(r_edges) - 1
    Nz = len(Z_edges) - 1

    shape = arr.shape

    time_axes = [ax for ax, s in enumerate(shape) if s == Nt]
    if len(time_axes) != 1:
        raise RuntimeError(
            f"{name}: could not uniquely identify time axis in shape {shape} "
            f"with Nt = {Nt}"
        )

    time_axis = time_axes[0]
    arr_tfirst = np.moveaxis(arr, time_axis, 0)
    _, a1, a2 = arr_tfirst.shape

    if (a1, a2) == (Nr, Nz):
        return arr_tfirst

    if (a1, a2) == (Nz, Nr):
        return arr_tfirst.transpose(0, 2, 1)

    raise RuntimeError(
        f"{name}: after moving time axis, spatial dims {(a1, a2)} "
        f"do not match Nr = {Nr}, Nz = {Nz}"
    )


def ensure_NrNz(arr, Nr, Nz, name="array"):
    """Return 2D array in shape (Nr, Nz)."""
    if arr.shape == (Nr, Nz):
        return arr

    if arr.shape == (Nz, Nr):
        return arr.T

    raise ValueError(
        f"{name}: unexpected shape {arr.shape}, expected ({Nr}, {Nz}) "
        f"or ({Nz}, {Nr})"
    )


def nearest_saved_index(t, target_time):
    saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]

    if saved_idx.size == 0:
        raise RuntimeError("No saved snapshots found.")

    return int(saved_idx[np.argmin(np.abs(t[saved_idx] - target_time))])


def moving_average_nan(y, window):
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
    Volume-weighted spherical average of q in an axisymmetric cylindrical grid.

    q should have shape (Nr, Nz).
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


def positive_for_log(y):
    return np.where((np.isfinite(y)) & (y > 0.0), y, np.nan)


def log_temperature_formatter(val, pos=None):
    """
    Log tick labels:
      0.1, 1, 10, 100,
      then 10^3, 10^4, ...
    """
    if val <= 0.0 or not np.isfinite(val):
        return ""

    exp = np.log10(val)
    exp_round = int(np.round(exp))

    # Only label exact powers of ten.
    if not np.isclose(exp, exp_round, atol=1e-10):
        return ""

    if exp_round == -1:
        return r"$0.1$"
    if exp_round == 0:
        return r"$1$"
    if exp_round == 1:
        return r"$10$"
    if exp_round == 2:
        return r"$100$"

    return rf"$10^{{{exp_round}}}$"


def construct_shells(r, Z, rCoM, ZCoM, CellVolumes, Z_source_pc):
    r_pc    = r / pc
    Z_pc    = Z / pc
    rCoM_pc = rCoM / pc
    ZCoM_pc = ZCoM / pc

    rr, zz = np.meshgrid(rCoM_pc, ZCoM_pc, indexing="ij")
    R_sph_pc = np.sqrt(rr**2 + (zz - Z_source_pc)**2)

    if radial_bin_width_pc is None:
        dr_med_pc = np.median(np.diff(r_pc))
        dZ_med_pc = np.median(np.diff(Z_pc))
        dr_shell_pc = min(dr_med_pc, dZ_med_pc)
    else:
        dr_shell_pc = radial_bin_width_pc

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

    shell_edges_pc = np.arange(0.0, Rmax_profile_pc + dr_shell_pc, dr_shell_pc)

    if shell_edges_pc.size < 2:
        raise RuntimeError("Too few radial shell edges. Check radial_bin_width_pc and domain size.")

    shell_centers_pc = 0.5 * (shell_edges_pc[:-1] + shell_edges_pc[1:])
    nbins = len(shell_centers_pc)

    shell_index = np.digitize(R_sph_pc, shell_edges_pc) - 1
    shell_index[(R_sph_pc < shell_edges_pc[0]) | (R_sph_pc >= shell_edges_pc[-1])] = -1

    return shell_centers_pc, shell_index, CellVolumes, nbins


def make_plot_keys():
    plot_keys = []

    if plot_gas_temperature:
        plot_keys.append("Tgas")

    for dust_type in dust_types_to_plot:
        if dust_type not in dust_key_map:
            raise ValueError(
                f"Unknown dust type '{dust_type}'. "
                f"Allowed options are {list(dust_key_map.keys())}."
            )

        plot_keys.append(dust_key_map[dust_type])

    return plot_keys


def read_profiles(filename, target_time_kyr, plot_keys):
    """
    Read one LYDION file, choose the snapshot nearest target_time_kyr,
    and return spherical averages of the requested gas/dust temperatures.
    """
    with h5py.File(filename, "r") as f:
        t = f["Times"][:]
        r = f["r"][:]
        Z = f["Z"][:]

        Nr = len(r) - 1
        Nz = len(Z) - 1

        if "rCoM" in f:
            rCoM = f["rCoM"][:]
        else:
            rCoM = 0.5 * (r[:-1] + r[1:])
            print(f"{filename}: rCoM not found, using cell-centered r fallback.")

        if "ZCoM" in f:
            ZCoM = f["ZCoM"][:]
        else:
            ZCoM = 0.5 * (Z[:-1] + Z[1:])
            print(f"{filename}: ZCoM not found, using cell-centered Z fallback.")

        if "CellVolumes" in f:
            CellVolumes = ensure_NrNz(f["CellVolumes"][:], Nr, Nz, name="CellVolumes")
        else:
            # Axisymmetric cylindrical fallback.
            # Overall 2π cancels in weighted averages.
            dr2 = 0.5 * (r[1:]**2 - r[:-1]**2)
            dZ  = Z[1:] - Z[:-1]
            CellVolumes = np.outer(dr2, dZ)
            print(f"{filename}: CellVolumes not found, using cylindrical fallback volumes.")

        if Z_source_pc is None:
            if "Settings" in f:
                Z_pos_code = read_attr_float(f["Settings"], "Z_pos", default=None)
                if Z_pos_code is None:
                    Zsrc_pc = 0.0
                else:
                    Zsrc_pc = Z_pos_code / pc
            else:
                Zsrc_pc = 0.0
        else:
            Zsrc_pc = Z_source_pc

        raw_by_key = {}

        if "Tgas" in plot_keys:
            raw_by_key["Tgas"] = reorder_history_3d(
                f["T_History"][:],
                t,
                r,
                Z,
                name="T_History"
            )

        if "Tdust_Sil" in plot_keys:
            raw_by_key["Tdust_Sil"] = reorder_history_3d(
                f["Tdust_Sil_History"][:],
                t,
                r,
                Z,
                name="Tdust_Sil_History"
            )

        if "Tdust_C" in plot_keys:
            raw_by_key["Tdust_C"] = reorder_history_3d(
                f["Tdust_C_History"][:],
                t,
                r,
                Z,
                name="Tdust_C_History"
            )

        if "Tdust_PAH" in plot_keys:
            raw_by_key["Tdust_PAH"] = reorder_history_3d(
                f["Tdust_PAH_History"][:],
                t,
                r,
                Z,
                name="Tdust_PAH_History"
            )

    k = nearest_saved_index(t, target_time_kyr * kyr)
    actual_time_kyr = t[k] / kyr

    shell_r_pc, shell_index, weights, nbins = construct_shells(
        r,
        Z,
        rCoM,
        ZCoM,
        CellVolumes,
        Zsrc_pc
    )

    profiles = {}

    for name in plot_keys:
        _, prof = spherical_average_profile(
            raw_by_key[name][k],
            shell_index,
            shell_r_pc,
            weights,
            nbins
        )

        prof = moving_average_nan(prof, profile_smoothing_window)
        profiles[name] = positive_for_log(prof)

    return {
        "r_pc": shell_r_pc,
        "profiles": profiles,
        "actual_time_kyr": actual_time_kyr,
        "Z_source_pc": Zsrc_pc,
    }


def add_panel_label(ax, text):
    if not show_panel_labels:
        return

    x, y = panel_label_pos

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=panel_label_fontsize,
        color="black",
        path_effects=[pe.withStroke(linewidth=3.0, foreground="white")]
    )


# ===============================================================
#                         Read data
# ===============================================================

plot_keys = make_plot_keys()

lya_data   = read_profiles(filename_lya,   TargetTime_kyr, plot_keys)
nolya_data = read_profiles(filename_nolya, TargetTime_kyr, plot_keys)

print(f"Lya file:    requested t = {TargetTime_kyr:.3f} kyr, using t = {lya_data['actual_time_kyr']:.3f} kyr")
print(f"No-Lya file: requested t = {TargetTime_kyr:.3f} kyr, using t = {nolya_data['actual_time_kyr']:.3f} kyr")


# ===============================================================
#                         Plot
# ===============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=figsize,
    sharex=True,
    gridspec_kw={"hspace": hspace}
)

ax_top, ax_bot = axes

runs = [
    {
        "ax": ax_top,
        "data": lya_data,
        "panel_label": rf"Ly$\alpha$, $t={lya_data['actual_time_kyr']:.2f}$ kyr",
    },
    {
        "ax": ax_bot,
        "data": nolya_data,
        "panel_label": rf"No Ly$\alpha$, $t={nolya_data['actual_time_kyr']:.2f}$ kyr",
    },
]

for run in runs:
    ax = run["ax"]
    data = run["data"]

    for name in plot_keys:
        ax.plot(
            data["r_pc"],
            data["profiles"][name],
            color=colors[name],
            lw=lw,
            alpha=alpha,
            solid_capstyle="round",
            dash_capstyle="round",
            label=labels[name],
        )

    ax.set_yscale("log")
    ax.set_ylabel(r"$T$ (K)", labelpad=14)

    if ylim_K is not None:
        ax.set_ylim(*ylim_K)

    if xlim_pc is not None:
        ax.set_xlim(*xlim_pc)

    ax.grid(False)

    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(FuncFormatter(log_temperature_formatter))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=log_minor_subs))
    ax.yaxis.set_minor_formatter(NullFormatter())

    add_panel_label(ax, run["panel_label"])

ax_bot.set_xlabel(r"$r$ (pc)", labelpad=14)
ax_bot.xaxis.set_minor_locator(AutoMinorLocator(x_minor_subdivisions))

# Avoid duplicate x tick labels on top panel.
ax_top.tick_params(labelbottom=False)

if figure_title is not None:
    fig.suptitle(figure_title, y=0.985, fontsize=25)

if show_legend:
    handles = [
        Line2D(
            [0], [0],
            color=colors[name],
            lw=lw,
            ls="-",
            label=labels[name]
        )
        for name in plot_keys
    ]

    ax_top.legend(
        handles=handles,
        loc=legend_loc,
        fontsize=legend_fontsize
    )

plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.965] if figure_title is not None else None)

if savefig:
    plt.savefig(outname, bbox_inches="tight")
    print(f"Saved {outname}")

plt.show()