import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator
from pathlib import Path

# ============================================================
#   U S E R   I N P U T S
# ============================================================

FigureName = "LYDION_Figure_IsolatedStars_RadiationForces.pdf"

# Top panel: dusty simulations.
# Each entry: (label, filename, linestyle)
DUST_RUNS = [
    (r"$u_{RMS}=0$ km s$^{-1}$", "RHDSim_Lya_u0_256x256_2026-05-02.h5", "-"),
    (r"$u_{RMS}=4$ km s$^{-1}$", "RHDSim_Lya_u4_256x256_2026-05-02.h5", "--"),
    (r"$u_{RMS}=8$ km s$^{-1}$", "RHDSim_Lya_u8_256x256_2026-05-02.h5", ":"),
]

# Bottom panel: no dust. Here we only include the run with Lyα destruction.
NODUST_RUNS = [
    (r"No dust, with Ly$\alpha$ destruction", "RHDSim_Lya_NoDust_u4_256x256_2026-05-06.h5", "-"),
]

# Force components to plot. All are radial-net forces, normalized by L_bol/c.
# Options: "Lya", "direct", "IR"
ForcesToPlot = ["Lya", "direct", "IR"]

# Skip missing files instead of crashing.
SkipMissingFiles = True

# Optional axis limits. Use None for automatic.
xlim_kyr = (-0.2, 5.2)
ylim_force = (0.4e-2, 1e3)

# Figure/cosmetic controls.
figsize = (8.5, 13.5)
fontsize_base   = 32
fontsize_title  = 36
fontsize_labels = 35
fontsize_legend = 25
lw              = 3.8
alpha_line      = 0.86

# Panel-label placement, in axes coordinates.
PanelLabelPositions = {
    "dust":   (0.06, 0.90),
    "nodust": (0.075, 0.90),
}

# X-axis tick control.
XMajorTickSpacing_kyr = 1.0
XMinorTickSpacing_kyr = 0.1

# Legend controls.
ShowLegends = True
ShowDustRunLegend   = True
ShowForceLegend     = False
ShowNoDustRunLegend = False

LegendPositions = {
    "dust_runs":   ("upper right", (0.64, 0.38)),
    "forces":      ("lower left",  (0.06, 0.2)),
    "nodust_runs": ("upper right", (0.985, 0.98)),
}

# Manual force labels with 90-degree arrows, drawn in the no-dust panel.
# Positions are in data coordinates.
ShowForceArrowLabels = True
ForceArrowLabels = [
    {
        "force": "Lya",
        "text": r"Ly$\alpha$ radiation pressure",
        "xytext": (1.0, 0.1*45.0),
        "x_arrow": 0.7,
        "fontsize": 35,
        "arrow_lw": 2.4,
    },
    {
        "force": "direct",
        "text": r"Direct radiation pressure",
        "xytext": (0.8, 0.13),
        "x_arrow": 0.4,
        "fontsize": 35,
        "arrow_lw": 2.4,
    },
]

# Fixed log ticks/labels, avoiding labels like 10^0.
yticks_force      = [0.01, 0.1, 1.0, 10.0, 100.0, 1e3]
yticklabels_force = ["0.01", "0.1", "1", "10", "100", r"$10^{3}$"]

# ============================================================
#   C O N S T A N T S
# ============================================================

c    = 2.99792458e10
yr   = 3.1556926e7
kyr  = 1e3 * yr
Lsun = 3.839e33

# ============================================================
#   P L O T   C O S M E T I C S
# ============================================================

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': fontsize_base})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 1.7
plt.rcParams['xtick.minor.width'] = 1.7
plt.rcParams['xtick.major.size']  = 18
plt.rcParams['xtick.minor.size']  = 8
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size']  = 18
plt.rcParams['ytick.minor.size']  = 8
plt.rcParams['xtick.major.pad']   = '12'
plt.rcParams['ytick.major.pad']   = '12'

plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linewidth'] = 1.7
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['xtick.top']       = True
plt.rcParams['xtick.bottom']    = True
plt.rcParams['ytick.left']      = True
plt.rcParams['ytick.right']     = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ============================================================
#   H E L P E R S
# ============================================================

FORCE_STYLE = {
    "Lya": {
        "label": r"Ly$\alpha$ radiation pressure",
        "color": "crimson",
        "lw": 4.2,
    },
    "direct": {
        "label": r"Direct radiation pressure",
        "color": "dodgerblue",
        "lw": 3.6,
    },
    "IR": {
        "label": r"IR radiation pressure",
        "color": "orangered",
        "lw": 3.6,
    },
}

DATASET_NAMES = {
    "Times": ["Times"],
    "L_Lya": ["L_Lya_History"],
    "L_IR":  ["L_IR_History"],

    # Radial-net force multipliers.
    "M_F_Lya_radial": ["M_F_radial_net_History", "M_F_radial_History"],
    "M_F_dir_radial": ["M_F_dir_radial_net_History", "M_F_dir_radial_History"],
    "M_F_IR_radial":  ["M_F_IR_radial_net_History", "M_F_IR_radial_History"],
}


def read_attr_if_exists(group, name, default=None):
    return group.attrs[name] if name in group.attrs else default


def read_first_existing_dataset(file, names):
    for name in names:
        if name in file:
            return file[name][:], name
    return None, None


def finite_saved_slice(t):
    """Return slice covering saved, nonzero/finite times.

    Includes index 0 if it is the only available snapshot; otherwise uses
    the range through the last positive finite time.
    """
    finite = np.isfinite(t)
    positive = finite & (t > 0.0)
    idx = np.where(positive)[0]
    if idx.size == 0:
        return slice(0, 1)
    return slice(0, idx[-1] + 1)


def positive_for_log_axis(y):
    """Hide non-positive/non-finite values on the log axis."""
    if y is None:
        return None
    y = np.asarray(y, dtype=float)
    y[~np.isfinite(y)] = np.nan
    y[y <= 0.0] = np.nan
    return y


def get_Lbol(file):
    settings = file["Settings"] if "Settings" in file else None
    if settings is None:
        raise RuntimeError("Could not find the Settings group in the HDF5 file.")

    L_bol = read_attr_if_exists(settings, "L_bol", None)
    if L_bol is None:
        raise RuntimeError("Could not find Settings.attrs['L_bol'] in the HDF5 file.")
    return float(L_bol)


def read_force_run(filename):
    """Read radial radiation-pressure forces normalized by L_bol/c."""
    with h5py.File(filename, "r") as f:
        t, t_name = read_first_existing_dataset(f, DATASET_NAMES["Times"])
        if t is None:
            raise RuntimeError(f"Times not found in {filename}")

        L_bol = get_Lbol(f)
        L_Lya, L_Lya_name = read_first_existing_dataset(f, DATASET_NAMES["L_Lya"])
        L_IR,  L_IR_name  = read_first_existing_dataset(f, DATASET_NAMES["L_IR"])

        M_F_Lya_radial, M_F_Lya_name = read_first_existing_dataset(f, DATASET_NAMES["M_F_Lya_radial"])
        M_F_dir_radial, M_F_dir_name = read_first_existing_dataset(f, DATASET_NAMES["M_F_dir_radial"])
        M_F_IR_radial,  M_F_IR_name  = read_first_existing_dataset(f, DATASET_NAMES["M_F_IR_radial"])

    sl = finite_saved_slice(t)
    t_kyr = np.asarray(t[sl], dtype=float) / kyr

    forces = {}
    found_names = {}

    # Normalized by ForceScale = L_bol/c.
    # Lyα:     F_rad / (L_bol/c) = M_F,Lya,rad * L_Lya/L_bol
    # Direct:  F_rad / (L_bol/c) = M_F,dir,rad
    # IR:      F_rad / (L_bol/c) = M_F,IR,rad  * L_IR/L_bol
    if M_F_Lya_radial is not None and L_Lya is not None:
        forces["Lya"] = positive_for_log_axis(np.asarray(M_F_Lya_radial[sl], dtype=float) * np.asarray(L_Lya[sl], dtype=float) / L_bol)
        found_names["Lya"] = (M_F_Lya_name, L_Lya_name)
    else:
        print(f"Warning: could not construct Lyα radial force for {filename}")

    if M_F_dir_radial is not None:
        forces["direct"] = positive_for_log_axis(np.asarray(M_F_dir_radial[sl], dtype=float))
        found_names["direct"] = (M_F_dir_name, "Settings.attrs['L_bol']")
    else:
        print(f"Warning: could not construct direct radial force for {filename}")

    if M_F_IR_radial is not None and L_IR is not None:
        forces["IR"] = positive_for_log_axis(np.asarray(M_F_IR_radial[sl], dtype=float) * np.asarray(L_IR[sl], dtype=float) / L_bol)
        found_names["IR"] = (M_F_IR_name, L_IR_name)
    else:
        print(f"Warning: could not construct IR radial force for {filename}")

    return {
        "t_kyr": t_kyr,
        "forces": forces,
        "found_names": found_names,
        "L_bol": L_bol,
    }


def load_group(run_specs, group_name):
    loaded = []
    for label, filename, linestyle in run_specs:
        path = Path(filename)
        if (not path.exists()) and SkipMissingFiles:
            print(f"Skipping missing {group_name} file: {filename}")
            continue

        run = read_force_run(filename)
        loaded.append({"label": label, "filename": filename, "linestyle": linestyle, **run})

        print(f"Loaded {group_name}: {label}  ({filename})")
        print(f"  L_bol = {run['L_bol'] / Lsun:.5e} Lsun")
        for force_name, dsets in run["found_names"].items():
            print(f"  {force_name:8s} <- {dsets}")
    return loaded


def plot_group(ax, runs, panel_title, panel_key):
    for run in runs:
        for force_name in ForcesToPlot:
            if force_name not in run["forces"]:
                continue
            style = FORCE_STYLE[force_name]
            ax.semilogy(
                run["t_kyr"], run["forces"][force_name],
                color=style["color"], linestyle=run["linestyle"],
                linewidth=style["lw"], alpha=alpha_line,
                dash_capstyle='round', solid_capstyle='round'
            )

    xlab, ylab = PanelLabelPositions[panel_key]
    ax.text(
        xlab, ylab, panel_title,
        transform=ax.transAxes, ha='left', va='top', fontsize=fontsize_title,
        path_effects=[pe.withStroke(linewidth=4, foreground='white')]
    )

    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in', length=18, width=2.5)
    ax.tick_params(axis='both', which='minor', direction='in', length=9,  width=2.0)


def make_legends(ax, runs,
                 show_run_legend=True,
                 show_force_legend=True,
                 run_legend_key="dust_runs"):
    # Color legend: force channel.
    # IR is deliberately omitted here because it is below the visible range.
    force_handles = []
    for force_name in ForcesToPlot:
        if force_name == "IR":
            continue
        style = FORCE_STYLE[force_name]
        force_handles.append(
            plt.Line2D([0], [0], color=style["color"], linewidth=style["lw"],
                       linestyle='-', label=style["label"], solid_capstyle='round')
        )

    # Style legend: simulations.
    run_handles = [
        plt.Line2D([0], [0], color='black', linewidth=lw, linestyle=run["linestyle"],
                   label=run["label"], dash_capstyle='round', solid_capstyle='round')
        for run in runs
    ]

    if show_run_legend and len(run_handles) > 0:
        loc_run, bbox_run = LegendPositions[run_legend_key]
        leg1 = ax.legend(
            handles=run_handles, loc=loc_run, bbox_to_anchor=bbox_run,
            fontsize=fontsize_legend, frameon=False, handlelength=2.8,
            borderpad=0.2, labelspacing=0.45
        )
        ax.add_artist(leg1)

    if show_force_legend and len(force_handles) > 0:
        loc_force, bbox_force = LegendPositions["forces"]
        ax.legend(
            handles=force_handles, loc=loc_force, bbox_to_anchor=bbox_force,
            fontsize=fontsize_legend, frameon=False, handlelength=2.8,
            borderpad=0.2, labelspacing=0.45
        )


def get_curve_y_at_x(run, force_name, x_value):
    """Return the positive y-value of one force curve nearest x_value."""
    if force_name not in run["forces"]:
        return None

    x = np.asarray(run["t_kyr"], dtype=float)
    y = np.asarray(run["forces"][force_name], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    if not np.any(mask):
        return None

    x_valid = x[mask]
    y_valid = y[mask]
    idx = np.argmin(np.abs(x_valid - x_value))
    return y_valid[idx]


def add_force_arrow_labels(ax, run, labels):
    """Add colored force labels with 90-degree elbow arrows."""
    for item in labels:
        force_name = item["force"]
        if force_name not in FORCE_STYLE:
            continue

        style = FORCE_STYLE[force_name]
        x_arrow = item.get("x_arrow", item["xytext"][0])
        y_arrow = get_curve_y_at_x(run, force_name, x_arrow)
        if y_arrow is None:
            continue

        ax.annotate(
            item.get("text", style["label"]),
            xy=(x_arrow, y_arrow),
            xytext=item["xytext"],
            color=style["color"],
            fontsize=item.get("fontsize", fontsize_legend),
            va=item.get("va", "center"),
            ha=item.get("ha", "left"),
            arrowprops=dict(
                arrowstyle='-|>',
                color=style["color"],
                lw=item.get("arrow_lw", 2.0),
                shrinkA=0,
                shrinkB=0,
                connectionstyle=item.get("connectionstyle", 'angle,angleA=0,angleB=90,rad=0'),
            ),
            path_effects=[
                pe.withStroke(
                    linewidth=item.get("stroke_width", 4),
                    foreground=item.get("stroke_color", "white")
                )
            ],
            annotation_clip=False,
        )


def apply_axis_limits(axes, all_runs):
    if xlim_kyr is not None:
        axes[0].set_xlim(*xlim_kyr)
    else:
        all_t = [run["t_kyr"] for run in all_runs if len(run["t_kyr"]) > 0]
        if all_t:
            axes[0].set_xlim(0.0, np.nanmax(np.concatenate(all_t)))

    if ylim_force is not None:
        for ax in axes:
            ax.set_ylim(*ylim_force)
    else:
        all_y = []
        for run in all_runs:
            for arr in run["forces"].values():
                pos = arr[np.isfinite(arr) & (arr > 0.0)]
                if pos.size > 0:
                    all_y.append(pos)
        if all_y:
            yy = np.concatenate(all_y)
            ymin = 10.0**np.floor(np.log10(np.nanmin(yy)) - 0.1)
            ymax = 10.0**np.ceil(np.log10(np.nanmax(yy)) + 0.1)
            for ax in axes:
                ax.set_ylim(ymin, ymax)

# ============================================================
#   R E A D   D A T A
# ============================================================

dust_runs   = load_group(DUST_RUNS,   "dust")
nodust_runs = load_group(NODUST_RUNS, "no-dust")

if len(dust_runs) == 0 and len(nodust_runs) == 0:
    raise RuntimeError("No runs loaded. Check filenames or set SkipMissingFiles = False.")


# ============================================================
#   P R I N T   M A X I M U M   L y α   F O R C E S
# ============================================================

def print_max_lya_forces(runs):
    """Print maximum Lyα radial force in units of L_bol/c."""
    print("")
    print("Maximum Lyα radial forces, in units of L_bol/c:")
    print("------------------------------------------------")

    global_max = -np.inf
    global_run = None
    global_time = np.nan

    for run in runs:
        if "Lya" not in run["forces"]:
            continue

        y = np.asarray(run["forces"]["Lya"], dtype=float)
        t = np.asarray(run["t_kyr"], dtype=float)

        mask = np.isfinite(y) & np.isfinite(t) & (y > 0.0)
        if not np.any(mask):
            continue

        valid_idx = np.where(mask)[0]
        local_idx = valid_idx[np.nanargmax(y[mask])]

        y_max = y[local_idx]
        t_max = t[local_idx]

        print(f"{run['label']:35s}  max = {y_max:.6e} at t = {t_max:.4f} kyr  ({run['filename']})")

        if y_max > global_max:
            global_max = y_max
            global_run = run
            global_time = t_max

    if global_run is not None:
        print("------------------------------------------------")
        print("Largest Lyα radial force:")
        print(f"  {global_max:.6e} L_bol/c")
        print(f"  simulation: {global_run['label']}  ({global_run['filename']})")
        print(f"  time:       {global_time:.4f} kyr")
    print("")

print_max_lya_forces(dust_runs + nodust_runs)

# ============================================================
#   P L O T
# ============================================================

fig, axes = plt.subplots(
    2, 1,
    figsize=figsize,
    sharex=True,
    gridspec_kw={"hspace": 0.04}
)

ax_dust, ax_nodust = axes

ax_dust.set_title(r"Ly$\alpha$ vs. direct radiation pressure", fontsize=fontsize_title, pad=20)

plot_group(ax_dust,   dust_runs,   r"$\mathfrak{D}/\mathfrak{D}_{\odot}$ = 10$^{-3}$", "dust")
plot_group(ax_nodust, nodust_runs, r"No dust, $u_{RMS}=4$ km s$^{-1}$", "nodust")

# Shared labels and limits.
ax_nodust.set_xlabel("Time (kyr)", labelpad=18, fontsize=fontsize_labels)
ax_dust.set_ylabel(r"Force $(L_{bol}/c)$", labelpad=18, fontsize=fontsize_labels)
ax_nodust.set_ylabel(r"Force $(L_{bol}/c)$", labelpad=18, fontsize=fontsize_labels)

apply_axis_limits(axes, dust_runs + nodust_runs)

# Fixed logarithmic tick labels.
for ax in axes:
    ax.set_yticks(yticks_force)
    ax.set_yticklabels(yticklabels_force)

# X-axis tick spacing.
if XMajorTickSpacing_kyr is not None:
    ax_nodust.xaxis.set_major_locator(MultipleLocator(XMajorTickSpacing_kyr))
if XMinorTickSpacing_kyr is not None:
    for ax in axes:
        ax.xaxis.set_minor_locator(MultipleLocator(XMinorTickSpacing_kyr))

# Legends.
if ShowLegends and dust_runs:
    make_legends(
        ax_dust, dust_runs,
        show_run_legend=ShowDustRunLegend,
        show_force_legend=ShowForceLegend,
        run_legend_key="dust_runs"
    )

if ShowLegends and nodust_runs:
    make_legends(
        ax_nodust, nodust_runs,
        show_run_legend=ShowNoDustRunLegend,
        show_force_legend=False,
        run_legend_key="nodust_runs"
    )

if ShowForceArrowLabels and nodust_runs:
    add_force_arrow_labels(ax_nodust, nodust_runs[0], ForceArrowLabels)

# Avoid duplicate x tick labels on top panel.
plt.setp(ax_dust.get_xticklabels(), visible=False)

fig.subplots_adjust(left=0.13, right=0.98, bottom=0.10, top=0.97)

plt.savefig(FigureName, dpi=300, bbox_inches='tight')
print(f"Saved: {FigureName}")
plt.show()
