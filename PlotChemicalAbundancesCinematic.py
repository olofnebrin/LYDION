# PlotAbundances_TwoPanel_Cinematic_Snapshots.py
#
# Static cinematic figure showing two chosen chemical species
# at 1--3 selected times.
#
# Each row:
#   Left  panel: abundance of species_left
#   Right panel: abundance of species_right
#
# The layout / aspect ratio follows your cinematic two-panel plot.

import h5py
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
cmastro_src = repo_root / "src" / "cmastro" / "src"

if cmastro_src.exists():
    sys.path.insert(0, str(cmastro_src))

import cmastro

from PlotCosmetics import (
    ApplyFontStyle,
    AddTimeLabel,
    InsetColorBar,
    AddScaleBar,
    AddStar,
)

# ============================================================
#                       USER INPUTS
# ============================================================

filename = "RHDSim_NoLya_512x1024_2026-07-09.h5"

# Choose 1--3 target times in kyr
#target_times_kyr = [10.0, 30.0, 60.0]
target_times_kyr = [90.0, 120.0, 166.0]

# Choose the two species to plot
species_left  = "xHII"
species_right = "xHI"

save_figure = True
outname     = "abundance_cinematic_snapshots.png"
dpi         = 300

# ------------------------------------------------------------
# Figure geometry
# ------------------------------------------------------------

panel_gap = 0.00   # white space between left/right panels
row_gap   = 0.02   # white space between snapshot rows

# ------------------------------------------------------------
# View
# ------------------------------------------------------------

half_pc     = 2.0
scalebar_pc = 0.5

midplane_symmetry = False
mirror_r = True
mirror_z = False

# ------------------------------------------------------------
# Star marker
# ------------------------------------------------------------

show_star = False   # set True if you want it back

# ------------------------------------------------------------
# Abundance plotting defaults
# ------------------------------------------------------------

# Same shared abundance colormap as in your chemistry script
cmap_abundance = "BuPu_r"

# Shared abundance range
AutoScale_Abundances = False
abund_vmin, abund_vmax = 1e-8, 1.0

# Auto-scale percentiles if enabled
pct_low_log  = 1.0
pct_high_log = 99.0

# ------------------------------------------------------------
# Per-species overrides
# Set to None to use the shared defaults above
# ------------------------------------------------------------

species_style = {
    "xHI":    dict(label=r"$x_{\rm HI}$",     cmap=None, vmin=None, vmax=None, tick_color=None,   outline_color=None),
    "xHII":   dict(label=r"$x_{\rm HII}$",    cmap=None, vmin=None, vmax=None, tick_color=None,   outline_color=None),
    "xH2":    dict(label=r"$x_{\rm H_2}$",    cmap=None, vmin=None, vmax=None, tick_color="white", outline_color="white"),

    "xHeI":   dict(label=r"$x_{\rm HeI}$",    cmap=None, vmin=None, vmax=None, tick_color=None,   outline_color=None),
    "xHeII":  dict(label=r"$x_{\rm HeII}$",   cmap=None, vmin=None, vmax=None, tick_color="white", outline_color="white"),
    "xHeIII": dict(label=r"$x_{\rm HeIII}$",  cmap=None, vmin=None, vmax=None, tick_color="white", outline_color="white"),

    "xCI":    dict(label=r"$x_{\rm CI}$",     cmap=None, vmin=None, vmax=None, tick_color=None,   outline_color=None),
    "xCII":   dict(label=r"$x_{\rm CII}$",    cmap=None, vmin=None, vmax=None, tick_color="white", outline_color="white"),
    "xCO":    dict(label=r"$x_{\rm CO}$",     cmap=None, vmin=None, vmax=None, tick_color="white", outline_color="white"),

    "xOI":    dict(label=r"$x_{\rm OI}$",     cmap=None, vmin=None, vmax=None, tick_color=None,   outline_color=None),
    "xOII":   dict(label=r"$x_{\rm OII}$",    cmap=None, vmin=None, vmax=None, tick_color=None,   outline_color=None),

    "xHD":    dict(label=r"$x_{\rm HD}$",     cmap=None, vmin=None, vmax=None, tick_color="white", outline_color="white"),
}

# Shared abundance colorbar colors
cbar_tick_color_abund    = "black"
cbar_outline_color_abund = "black"

# ------------------------------------------------------------
# Colorbar geometry
# Same idea as in your current cinematic plot
# ------------------------------------------------------------

cbar_left_x   = 0.39   # move left cbar left/right
cbar_right_x  = 0.90   # move right cbar left/right
cbar_y_rel    = 0.10   # relative vertical position within each row
cbar_h_rel    = 0.20   # relative height within each row
cbar_width    = 0.012

fontsize_cbar = 18

# ------------------------------------------------------------
# Panel-label style
# ------------------------------------------------------------

panel_label_fontsize = 24
panel_label_color    = "white"
panel_label_stroke   = "black"
panel_label_stroke_width = 3.0
panel_label_pos = (0.96, 0.05)   # in axes coordinates

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

yr  = 3.1556926e7
kyr = 1e3 * yr
pc  = 3.085677581e18

eps = 1e-30


# ============================================================
#                    HELPER FUNCTIONS
# ============================================================

def ensure_NtNrNz(arr, Nr, Nz):
    a1, a2 = arr.shape[1], arr.shape[2]

    if (a1, a2) == (Nr, Nz):
        return arr

    if (a1, a2) == (Nz, Nr):
        return arr.transpose(0, 2, 1)

    raise ValueError(
        f"Unexpected array shape {arr.shape} vs edges Nr={Nr}, Nz={Nz}"
    )


def mirror_edges_from_zero(edges):
    neg = -edges[:0:-1]
    return np.concatenate([neg, edges])


def mirror_field(A, mirror_r=True, mirror_z=True):
    B = A

    if mirror_r:
        B = np.concatenate([B[::-1, :], B], axis=0)

    if mirror_z:
        B = np.concatenate([B[:, ::-1], B], axis=1)

    return B


def positive_values(arr):
    vals = np.asarray(arr)
    return vals[np.isfinite(vals) & (vals > 0.0)]


def make_lognorm_from_arrays(arrays, default_vmin, default_vmax,
                             pct_low=1.0, pct_high=99.0):
    vals_list = [positive_values(a) for a in arrays if a is not None]
    vals_list = [v for v in vals_list if v.size > 0]

    if len(vals_list) == 0:
        return LogNorm(vmin=default_vmin, vmax=default_vmax)

    vals = np.concatenate(vals_list)

    vmin = np.percentile(vals, pct_low)
    vmax = np.percentile(vals, pct_high)

    vmin = max(vmin, np.min(vals))
    vmax = min(vmax, np.max(vals))

    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin <= 0.0) or (vmax <= vmin):
        return LogNorm(vmin=default_vmin, vmax=default_vmax)

    return LogNorm(vmin=vmin, vmax=vmax)


def fmt_log_tick(val):
    exp = int(np.round(np.log10(val)))
    if exp == -1:
        return r"$0.1$"
    elif exp == 0:
        return r"$1$"
    elif exp == 1:
        return r"$10$"
    else:
        return rf"$10^{{{exp}}}$"


def get_log_ticks_and_labels(norm):
    logmin = int(np.ceil(np.log10(norm.vmin)))
    logmax = int(np.floor(np.log10(norm.vmax)))
    ticks = 10.0 ** np.arange(logmin, logmax + 1)
    labels = [fmt_log_tick(v) for v in ticks]
    return ticks, labels


def resolve_species_style(species_name):
    if species_name not in species_style:
        raise ValueError(
            f"Unsupported species '{species_name}'. "
            f"Supported: {list(species_style.keys())}"
        )

    style = species_style[species_name].copy()

    if style["cmap"] is None:
        style["cmap"] = cmap_abundance

    if style["vmin"] is None:
        style["vmin"] = abund_vmin

    if style["vmax"] is None:
        style["vmax"] = abund_vmax

    if style["tick_color"] is None:
        style["tick_color"] = cbar_tick_color_abund

    if style["outline_color"] is None:
        style["outline_color"] = cbar_outline_color_abund

    return style


def add_panel_label(ax, text):
    txt = ax.text(
        panel_label_pos[0], panel_label_pos[1], text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=panel_label_fontsize,
        color=panel_label_color,
        zorder=20,
    )
    txt.set_path_effects([
        pe.Stroke(linewidth=panel_label_stroke_width, foreground=panel_label_stroke),
        pe.Normal()
    ])
    return txt


def CinematicAxesSnapshotRows(
    nrows,
    row_figsize=(16, 8),
    panel_gap=0.02,
    row_gap=0.03,
):
    import matplotlib.pyplot as plt

    row_width, row_height = row_figsize

    total_height = row_height * (
        nrows + row_gap * (nrows - 1)
    )

    fig = plt.figure(
        figsize=(row_width, total_height),
        facecolor="white",
    )
    fig.patch.set_facecolor("white")

    row_h = 1.0 / (
        nrows + row_gap * (nrows - 1)
    )
    gap_h = row_gap * row_h
    panel_w = (1.0 - panel_gap) / 2.0

    axes = []

    for row in range(nrows):
        y0 = 1.0 - (row + 1) * row_h - row * gap_h

        axL = fig.add_axes([0.0, y0, panel_w, row_h])
        axR = fig.add_axes([panel_w + panel_gap, y0, panel_w, row_h])

        axL.set_axis_off()
        axR.set_axis_off()

        axes.append((axL, axR))

    return fig, axes


# ============================================================
#                     READ DATA
# ============================================================

with h5py.File(filename, "r") as f:
    t = f["Times"][:]
    r = f["r"][:]
    Z = f["Z"][:]

    # Directly stored species
    xHII_raw   = f["xHII_History"][:]
    xH2_raw    = f["xH2_History"][:]

    xHeII_raw  = f["xHeII_History"][:]
    xHeIII_raw = f["xHeIII_History"][:]

    xCI_raw    = f["xCI_History"][:]
    xCII_raw   = f["xCII_History"][:]
    xCO_raw    = f["xCO_History"][:]

    xOI_raw    = f["xOI_History"][:]
    xOII_raw   = f["xOII_History"][:]

    xHD_raw    = f["xHD_History"][:]

Nr = len(r) - 1
Nz = len(Z) - 1

xHII_hist   = ensure_NtNrNz(xHII_raw,   Nr, Nz)
xH2_hist    = ensure_NtNrNz(xH2_raw,    Nr, Nz)

xHeII_hist  = ensure_NtNrNz(xHeII_raw,  Nr, Nz)
xHeIII_hist = ensure_NtNrNz(xHeIII_raw, Nr, Nz)

xCI_hist    = ensure_NtNrNz(xCI_raw,    Nr, Nz)
xCII_hist   = ensure_NtNrNz(xCII_raw,   Nr, Nz)
xCO_hist    = ensure_NtNrNz(xCO_raw,    Nr, Nz)

xOI_hist    = ensure_NtNrNz(xOI_raw,    Nr, Nz)
xOII_hist   = ensure_NtNrNz(xOII_raw,   Nr, Nz)

xHD_hist    = ensure_NtNrNz(xHD_raw,    Nr, Nz)

# Derived species
xHI_hist  = np.clip(1.0 - xHII_hist - 2.0 * xH2_hist,      0.0, 1.0)
xHeI_hist = np.clip(1.0 - xHeII_hist - xHeIII_hist,        0.0, 1.0)

species_hist = {
    "xHI":    xHI_hist,
    "xHII":   xHII_hist,
    "xH2":    xH2_hist,

    "xHeI":   xHeI_hist,
    "xHeII":  xHeII_hist,
    "xHeIII": xHeIII_hist,

    "xCI":    xCI_hist,
    "xCII":   xCII_hist,
    "xCO":    xCO_hist,

    "xOI":    xOI_hist,
    "xOII":   xOII_hist,

    "xHD":    xHD_hist,
}

if species_left not in species_hist:
    raise ValueError(f"Unknown left species: {species_left}")
if species_right not in species_hist:
    raise ValueError(f"Unknown right species: {species_right}")

# ============================================================
#                  SELECT SNAPSHOT INDICES
# ============================================================

target_times_kyr = np.atleast_1d(np.asarray(target_times_kyr, dtype=float))

if not (1 <= len(target_times_kyr) <= 3):
    raise ValueError("target_times_kyr must contain between 1 and 3 times.")

saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]
saved_times_kyr = t[saved_idx] / kyr

snapshot_idx = []

for target in target_times_kyr:
    i_nearest = np.argmin(np.abs(saved_times_kyr - target))
    k = saved_idx[i_nearest]
    snapshot_idx.append(k)

    print(
        f"Requested t = {target:.3f} kyr "
        f"-> using t = {t[k] / kyr:.3f} kyr "
        f"(index {k})"
    )

# ============================================================
#                 STYLES / NORMS FOR THE TWO SPECIES
# ============================================================

style_left  = resolve_species_style(species_left)
style_right = resolve_species_style(species_right)

fields_left_for_norm  = [species_hist[species_left][k]  for k in snapshot_idx]
fields_right_for_norm = [species_hist[species_right][k] for k in snapshot_idx]

if AutoScale_Abundances:
    norm_left = make_lognorm_from_arrays(
        fields_left_for_norm,
        style_left["vmin"],
        style_left["vmax"],
        pct_low=pct_low_log,
        pct_high=pct_high_log,
    )
    norm_right = make_lognorm_from_arrays(
        fields_right_for_norm,
        style_right["vmin"],
        style_right["vmax"],
        pct_low=pct_low_log,
        pct_high=pct_high_log,
    )
else:
    norm_left  = LogNorm(vmin=style_left["vmin"],   vmax=style_left["vmax"])
    norm_right = LogNorm(vmin=style_right["vmin"],  vmax=style_right["vmax"])

ticks_left,  ticklabels_left  = get_log_ticks_and_labels(norm_left)
ticks_right, ticklabels_right = get_log_ticks_and_labels(norm_right)

# ============================================================
#                     PLOTTING GEOMETRY
# ============================================================

r_edges = r / pc
z_edges = Z / pc

if midplane_symmetry:
    r_center = 0.0
    z_center = 0.0
else:
    r_center = 0.0
    z_center = 0.5 * z_edges[-1]

r_plot = mirror_edges_from_zero(r_edges) if mirror_r else r_edges
z_plot = mirror_edges_from_zero(z_edges) if mirror_z else z_edges

Rplot, Zplot = np.meshgrid(r_plot, z_plot, indexing="ij")

if mirror_r:
    ax_xlim = (-half_pc, half_pc)
else:
    ax_xlim = (max(r_edges[0], r_center), r_center + 2.0 * half_pc)

if mirror_z:
    ax_ylim = (-half_pc, half_pc)
else:
    ax_ylim = (max(z_edges[0], z_center - half_pc), z_center + half_pc)

# ============================================================
#                       FIGURE SETUP
# ============================================================

ApplyFontStyle(theme="cinematic", font_size=32)

nrows = len(snapshot_idx)

fig, axes = CinematicAxesSnapshotRows(
    nrows=nrows,
    row_figsize=(16, 8),
    panel_gap=panel_gap,
    row_gap=row_gap,
)

# ============================================================
#                        DRAW PANELS
# ============================================================

for row, k in enumerate(snapshot_idx):

    axL, axR = axes[row]

    for ax in (axL, axR):
        ax.set_xlim(*ax_xlim)
        ax.set_ylim(*ax_ylim)
        ax.set_aspect("equal", adjustable="box")

    # Fields
    field_left = species_hist[species_left][k]
    field_right = species_hist[species_right][k]

    A = mirror_field(
        np.clip(field_left, norm_left.vmin, norm_left.vmax),
        mirror_r=mirror_r,
        mirror_z=mirror_z,
    )

    B = mirror_field(
        np.clip(field_right, norm_right.vmin, norm_right.vmax),
        mirror_r=mirror_r,
        mirror_z=mirror_z,
    )

    # Images
    imL = axL.pcolormesh(
        Rplot, Zplot, A,
        norm=norm_left,
        shading="auto",
        cmap=style_left["cmap"],
    )

    imR = axR.pcolormesh(
        Rplot, Zplot, B,
        norm=norm_right,
        shading="auto",
        cmap=style_right["cmap"],
    )

    # Optional star marker
    if show_star:
        star_r_pc = 0.0 if mirror_r else 0.002 * ax_xlim[1]
        star_z_pc = 0.0 if midplane_symmetry else z_center
        AddStar(axL, star_r_pc, star_z_pc, size=100)
        AddStar(axR, star_r_pc, star_z_pc, size=100)

    # Time label on left panel
    AddTimeLabel(
        axL,
        f"t = {t[k] / kyr:.2f} kyr",
        pos=(0.03, 0.97),
        fontsize=22,
    )

    # Species labels inside panels
    add_panel_label(axL, style_left["label"])
    add_panel_label(axR, style_right["label"])

    # Scale bar on left panel
    if mirror_r:
        pos_ScaleBar = (0.08, 0.15)
    else:
        pos_ScaleBar = (0.65, 0.10)

    AddScaleBar(
        axL,
        length=scalebar_pc,
        label=f"{scalebar_pc:.1f} pc",
        pos=pos_ScaleBar,
        fontsize=18,
    )

    # Colorbars
    ax_pos = axL.get_position()
    row_y0 = ax_pos.y0
    row_h  = ax_pos.height

    cb_y = row_y0 + cbar_y_rel * row_h
    cb_h = cbar_h_rel * row_h

    cbL = InsetColorBar(
        fig, axL, imL,
        rect=(cbar_left_x, cb_y, cbar_width, cb_h),
        label=None,
        labelsize=fontsize_cbar,
        tick_labelsize=fontsize_cbar,
        labelcolor=style_left["tick_color"],
        tickcolors=style_left["tick_color"],
    )
    cbL.set_ticks(ticks_left)
    cbL.set_ticklabels(ticklabels_left)
    cbL.ax.minorticks_off()
    cbL.outline.set_edgecolor(style_left["outline_color"])
    cbL.outline.set_linewidth(1.5)
    for spine in cbL.ax.spines.values():
        spine.set_edgecolor(style_left["outline_color"])
        spine.set_linewidth(1.5)

    cbR = InsetColorBar(
        fig, axR, imR,
        rect=(cbar_right_x, cb_y, cbar_width, cb_h),
        label=None,
        labelsize=fontsize_cbar,
        tick_labelsize=fontsize_cbar,
        labelcolor=style_right["tick_color"],
        tickcolors=style_right["tick_color"],
    )
    cbR.set_ticks(ticks_right)
    cbR.set_ticklabels(ticklabels_right)
    cbR.ax.minorticks_off()
    cbR.outline.set_edgecolor(style_right["outline_color"])
    cbR.outline.set_linewidth(1.5)
    for spine in cbR.ax.spines.values():
        spine.set_edgecolor(style_right["outline_color"])
        spine.set_linewidth(1.5)

# ============================================================
#                       SAVE / SHOW
# ============================================================

if save_figure:
    fig.savefig(
        outname,
        dpi=dpi,
        facecolor="white",
        bbox_inches=None,
        pad_inches=0.0,
    )
    print(f"Saved figure to {outname}")
else:
    import matplotlib.pyplot as plt
    plt.show()