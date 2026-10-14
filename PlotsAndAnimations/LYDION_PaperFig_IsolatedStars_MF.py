import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator
from pathlib import Path
import os

repo_root = Path(__file__).resolve().parents[1]
os.chdir(repo_root)

# ============================================================
#   U S E R   I N P U T S
# ============================================================

FigureName = "LYDION_Figure_IsolatedStars_MF.pdf"

# Top panel: dusty simulations.
# Each entry: (label, filename, color)
DUST_RUNS = [
    (r"$u_{RMS}=0$ km s$^{-1}$", "RHDSim_Lya_u0_256x256_2026-05-02.h5", "crimson"),
    (r"$u_{RMS}=4$ km s$^{-1}$", "RHDSim_Lya_u4_256x256_2026-05-02.h5", "dodgerblue"),
    (r"$u_{RMS}=8$ km s$^{-1}$", "RHDSim_Lya_u8_256x256_2026-05-02.h5", "black"),
]

# Bottom panel: no-dust / no-destruction variants.
# Edit the NoDest filename if your exact name/date differs.
NODUST_RUNS = [
    (r"With Ly$\alpha$ destruction", "RHDSim_Lya_NoDust_u4_256x256_2026-05-06.h5", "black"),
    (r"Ignoring Ly$\alpha$ destruction", "RHDSim_Lya_NoDest_u4_256x256_2026-05-06.h5", "deeppink"),
]

# Which M_F variants to plot.
# Options: "total", "radial_net", "antigravity"
MFVariantsToPlot = ["radial_net", "antigravity", "total"]

# Skip missing files instead of crashing.
SkipMissingFiles = True

# Optional axis limits. Use None for automatic.
xlim_kyr = (-0.2, 5.2)
ylim_MF  = (1.0, 1.7e3)   # default from the reference M_F plotting script

# Figure/cosmetic controls.
figsize = (10.8, 11.8)
fontsize_base   = 32
fontsize_title  = 34
fontsize_labels = 35
fontsize_legend = 28
lw              = 3.8
alpha_line      = 0.8

# Panel-label placement, in axes coordinates.
# Increase x if the label is too close to the left y-axis/tick region.
PanelLabelPositions = {
    "dust":   (0.075, 0.90),
    "nodust": (0.075, 0.20),
}

# X-axis tick control.
# Keep labelled major ticks as before, but use denser *minor* ticks.
# With XMinorTickSpacing_kyr = 0.1, the interval 0--1 kyr is split into
# 10 minor-tick intervals, without adding minor tick labels.
XMajorTickSpacing_kyr = 1.0
XMinorTickSpacing_kyr = 0.1

# Legend controls. Set ShowLegends=False if you prefer to add labels/arrows manually.
ShowLegends = True

# Top panel legends.
ShowDustRunLegend     = True
ShowDustVariantLegend = True

# Bottom panel legends.
# Default: no bottom legends, since the solid/dashed/dotted legend is already
# shown in the top panel, and the no-dust/no-destruction curves are labelled
# manually in the panel.
ShowNoDustRunLegend     = False
ShowNoDustVariantLegend = False

LegendPositions = {
    "dust_runs":       ("upper right", (0.95, 0.95)),
    "dust_variants":   ("lower left",  (0.08, 0.04)),
    "nodust_runs":     ("upper right", (0.99, 0.99)),
    "nodust_variants": ("lower left",  (0.06, 0.04)),
}

# Manual text labels for the two no-dust curves in the lower panel.
# Positions are in axes coordinates. Set ShowNoDustTextLabels=False to disable.
# Positive rotation = counter-clockwise, negative = clockwise.
ShowNoDustTextLabels = True
NoDustTextLabels = [
    {
        "text": r"With Ly$\alpha$ destruction",
        "color": "black",
        "xy": (0.4, 0.46),
        "ha": "left",
        "va": "center",
        "rotation": -4,
        "rotation_mode": "anchor",
        "stroke_width": 4,
        "stroke_color": "white",
    },
    {
        "text": r"Ignoring Ly$\alpha$ destruction",
        "color": "deeppink",
        "xy": (0.4, 0.82),
        "ha": "left",
        "va": "center",
        "rotation": -1.7,
        "rotation_mode": "anchor",
        "stroke_width": 4,
        "stroke_color": "white",
    },
]

# Fixed log ticks/labels, matching the style of the reference code.
yticks_MF      = [1.0, 10.0, 100.0, 1e3]
yticklabels_MF = ["1", "10", "100", r"$10^{3}$"]

# ============================================================
#   C O N S T A N T S
# ============================================================

yr  = 3.1556926e7
kyr = 1e3 * yr

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

MF_DATASETS = {
    "total":       ("M_F_History",                    r"$M_F$",                        "--"),
    "radial_net":  ("M_F_radial_net_History",         r"$M_{F,radial}$",              "-"),
    "antigravity": ("M_F_History_antigrav_direction", r"$M_{F,antigrav}$",            ":"),
}

# Fallback names in case older files have slightly different labels.
MF_FALLBACK_DATASETS = {
    "total":       ["M_F_History"],
    "radial_net":  ["M_F_radial_net_History", "M_F_radial_History"],
    "antigravity": ["M_F_History_antigrav_direction", "M_F_antigrav_History", "M_F_antigravity_History"],
}


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


def read_mf_run(filename):
    """Read Times and available M_F variants from a LYDION HDF5 file."""
    with h5py.File(filename, "r") as f:
        if "Times" not in f:
            raise RuntimeError(f"Times not found in {filename}")
        t = f["Times"][:]

        mf = {}
        found_names = {}
        for variant in MFVariantsToPlot:
            if variant not in MF_FALLBACK_DATASETS:
                raise ValueError(f"Unknown M_F variant: {variant}")
            arr, dataset_name = read_first_existing_dataset(f, MF_FALLBACK_DATASETS[variant])
            if arr is not None:
                mf[variant] = arr
                found_names[variant] = dataset_name
            else:
                print(f"Warning: no dataset found for variant '{variant}' in {filename}")

    sl = finite_saved_slice(t)
    t = t[sl] / kyr

    for variant in list(mf.keys()):
        arr = np.asarray(mf[variant])[sl]
        # Log-axis-safe: non-positive values are hidden rather than clipped upward.
        arr = arr.astype(float)
        arr[~np.isfinite(arr)] = np.nan
        arr[arr <= 0.0] = np.nan
        mf[variant] = arr

    return {"t_kyr": t, "mf": mf, "found_names": found_names}


def load_group(run_specs, group_name):
    loaded = []
    for label, filename, color in run_specs:
        path = Path(filename)
        if (not path.exists()) and SkipMissingFiles:
            print(f"Skipping missing {group_name} file: {filename}")
            continue
        run = read_mf_run(filename)
        loaded.append({"label": label, "filename": filename, "color": color, **run})
        print(f"Loaded {group_name}: {label}  ({filename})")
        for variant, dset in run["found_names"].items():
            print(f"  {variant:12s} <- {dset}")
    return loaded


def plot_group(ax, runs, panel_title, panel_key):
    for run in runs:
        for variant in MFVariantsToPlot:
            if variant not in run["mf"]:
                continue
            _, _, linestyle = MF_DATASETS[variant]
            ax.semilogy(
                run["t_kyr"], run["mf"][variant],
                color=run["color"], linestyle=linestyle, linewidth=lw,
                alpha=alpha_line, dash_capstyle='round', solid_capstyle='round'
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


def add_manual_text_labels(ax, labels):
    for item in labels:
        ax.text(
            item.get("xy", (0.5, 0.5))[0],
            item.get("xy", (0.5, 0.5))[1],
            item["text"],
            transform=ax.transAxes,
            color=item.get("color", "black"),
            fontsize=item.get("fontsize", fontsize_legend),
            ha=item.get("ha", "left"),
            va=item.get("va", "center"),
            rotation=item.get("rotation", 0.0),
            rotation_mode=item.get("rotation_mode", "anchor"),
            path_effects=[
                pe.withStroke(
                    linewidth=item.get("stroke_width", 4),
                    foreground=item.get("stroke_color", "white")
                )
            ]
        )


def make_legends(ax, runs, color_legend_key, style_legend_key,
                 show_color_legend=True, show_style_legend=True):
    # Color legend: simulations.
    color_handles = [
        plt.Line2D([0], [0], color=run["color"], linewidth=lw, linestyle='-', label=run["label"],
                   solid_capstyle='round')
        for run in runs
    ]

    # Style legend: M_F variants.
    style_handles = []
    for variant in MFVariantsToPlot:
        _, variant_label, linestyle = MF_DATASETS[variant]
        style_handles.append(
            plt.Line2D([0], [0], color='black', linewidth=lw, linestyle=linestyle, label=variant_label,
                       dash_capstyle='round', solid_capstyle='round')
        )

    if show_color_legend and len(color_handles) > 0:
        loc_col, bbox_col = LegendPositions[color_legend_key]
        leg1 = ax.legend(
            handles=color_handles, loc=loc_col, bbox_to_anchor=bbox_col,
            fontsize=fontsize_legend, frameon=False, handlelength=2.8,
            borderpad=0.2, labelspacing=0.45
        )
        ax.add_artist(leg1)

    if show_style_legend and len(style_handles) > 0:
        loc_sty, bbox_sty = LegendPositions[style_legend_key]
        ax.legend(
            handles=style_handles, loc=loc_sty, bbox_to_anchor=bbox_sty,
            fontsize=fontsize_legend, frameon=False, handlelength=2.8,
            borderpad=0.2, labelspacing=0.45
        )

# ============================================================
#   R E A D   D A T A
# ============================================================

dust_runs   = load_group(DUST_RUNS,   "dust")
nodust_runs = load_group(NODUST_RUNS, "no-dust")

if len(dust_runs) == 0 and len(nodust_runs) == 0:
    raise RuntimeError("No runs loaded. Check filenames or set SkipMissingFiles = False.")

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

ax_dust.set_title(r"Ly$\alpha$ force multiplier for isolated stars", fontsize=fontsize_title, pad=20)

plot_group(ax_dust,   dust_runs,   r"$\mathfrak{D}/\mathfrak{D}_{\odot}$ = 10$^{-3}$", "dust")
plot_group(ax_nodust, nodust_runs, r"No dust, u$_{RMS}$ = 4 km s$^{-1}$", "nodust")

# Shared labels and limits.
ax_nodust.set_xlabel("Time (kyr)", labelpad=18, fontsize=fontsize_labels)

ax_nodust.set_ylabel(r"Ly$\alpha$ force multiplier", labelpad=18, fontsize=fontsize_labels)
ax_dust.set_ylabel(r"Ly$\alpha$ force multiplier", labelpad=18, fontsize=fontsize_labels)

if xlim_kyr is not None:
    ax_dust.set_xlim(*xlim_kyr)
else:
    all_t = []
    for run in dust_runs + nodust_runs:
        if len(run["t_kyr"]) > 0:
            all_t.append(run["t_kyr"])
    if all_t:
        tmax = np.nanmax(np.concatenate(all_t))
        ax_dust.set_xlim(0.0, tmax)

if ylim_MF is not None:
    ax_dust.set_ylim(*ylim_MF)
    ax_nodust.set_ylim(*ylim_MF)
else:
    all_y = []
    for run in dust_runs + nodust_runs:
        for arr in run["mf"].values():
            pos = arr[np.isfinite(arr) & (arr > 0.0)]
            if pos.size > 0:
                all_y.append(pos)
    if all_y:
        yy = np.concatenate(all_y)
        ymin = 10.0**np.floor(np.log10(np.nanmin(yy)) - 0.1)
        ymax = 10.0**np.ceil(np.log10(np.nanmax(yy)) + 0.1)
        ax_dust.set_ylim(ymin, ymax)
        ax_nodust.set_ylim(ymin, ymax)

# Fixed logarithmic tick labels, avoiding labels like 10^0.
for ax in axes:
    ax.set_yticks(yticks_MF)
    ax.set_yticklabels(yticklabels_MF)

# X-axis tick spacing.
# Major ticks keep labelled tick positions like before; minor ticks are denser
# and do not get labels.
if XMajorTickSpacing_kyr is not None:
    ax_nodust.xaxis.set_major_locator(MultipleLocator(XMajorTickSpacing_kyr))
if XMinorTickSpacing_kyr is not None:
    for ax in axes:
        ax.xaxis.set_minor_locator(MultipleLocator(XMinorTickSpacing_kyr))

# Legends.
if ShowLegends and dust_runs:
    make_legends(
        ax_dust, dust_runs, "dust_runs", "dust_variants",
        show_color_legend=ShowDustRunLegend,
        show_style_legend=ShowDustVariantLegend
    )

if ShowLegends and nodust_runs:
    make_legends(
        ax_nodust, nodust_runs, "nodust_runs", "nodust_variants",
        show_color_legend=ShowNoDustRunLegend,
        show_style_legend=ShowNoDustVariantLegend
    )

if ShowNoDustTextLabels:
    add_manual_text_labels(ax_nodust, NoDustTextLabels)

# Avoid duplicate x tick labels on top panel.
plt.setp(ax_dust.get_xticklabels(), visible=False)

# A little extra room for the shared y-label.
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.97)

plt.savefig(FigureName, dpi=300, bbox_inches='tight')
print(f"Saved: {FigureName}")
plt.show()
