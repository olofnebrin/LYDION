
"""LYDION clock figure for postdoc research statements.

ShowExtraRow=False: original density / Ly-alpha-energy two-clock figure.
ShowExtraRow=True : same figure style, adding temperature / neutral fraction
                    as a second row. Axes and colorbars preserve the original
                    physical dimensions and placement for each row.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, FancyArrowPatch
from pathlib import Path
import sys

# ============================================================
# USER INPUTS
# ============================================================

Filename_noLya = "RHDSim_StarCluster_NoLya_512x512_2026-04-19.h5"
Filename_Lya   = "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5"

ShowExtraRow = True    # False reproduces the original two-panel layout

FigureName = (
    "LYDION_Clock_nH_eLya_T_xHI.pdf"
    if ShowExtraRow else "LYDION_Clock_nH_eLya.pdf"
)

# Clockwise quadrants: upper right -> lower right -> lower left -> upper left.
SnapshotTimes_kyr = [5, 20, 30, 40]

DomainFraction = 0.80

# Each row retains the original 12.6 x 6.4 in layout.
FigureWidth_inches = 12.6
RowHeight_inches   = 6.4
RowGap_inches      = 0.85  # blank space between complete row layouts

# Colormaps (unchanged for original panels).
cmap_nH   = "cma:laguna"
cmap_eLya = "magma"
cmap_Tgas = "coolwarm"
cmap_xHI  = "BuPu_r"

# Logarithmic ranges for optional second row (from chemistry figure).
Tgas_vmin, Tgas_vmax = 1.0, 1e5
xHI_vmin,  xHI_vmax  = 1e-8, 1.0

# Original text sizes are preserved because each row has the same
# physical size as the original figure.
fontsize_title      = 30
fontsize_time       = 23
fontsize_cbar_ticks = 25
fontsize_cbar_label = 25

ShowTimeArrow    = False
ShowSourceMarker = True
ShowScaleBar     = False
scale_bar_length_pc = 0.2
scale_bar_label     = "0.2 pc"

# ============================================================
# CONSTANTS / COLORMAP REGISTRATION
# ============================================================

repo_root = Path(__file__).resolve().parent
cmastro_src = repo_root / "src" / "cmastro" / "src"
if cmastro_src.exists():
    sys.path.insert(0, str(cmastro_src))

try:
    import cmastro  # noqa: F401; registers cma:laguna
except ImportError:
    pass

X   = 0.76
mH  = 1.67e-24
yr  = 3.1556926e7
kyr = 1e3 * yr
pc  = 3.085677581e18

# ============================================================
# PLOT COSMETICS (original)
# ============================================================

plt.rc("font", **{"sans-serif": ["AVHershey Complex"], "size": 28})
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "GreekC"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 2.5

# ============================================================
# HELPERS
# ============================================================

def reorder_history_3d(arr, t, r_edges, Z_edges):
    """Reorder a 3D history array to shape (Nt, Nr, Nz)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    Nt = t.size
    Nr = len(r_edges) - 1
    Nz = len(Z_edges) - 1

    time_axes = [ax for ax, s in enumerate(arr.shape) if s == Nt]
    if len(time_axes) != 1:
        raise RuntimeError(
            f"Could not uniquely identify time axis in shape {arr.shape}, Nt={Nt}"
        )

    arr_tfirst = np.moveaxis(arr, time_axes[0], 0)
    if arr_tfirst.shape[1:] == (Nr, Nz):
        return arr_tfirst
    if arr_tfirst.shape[1:] == (Nz, Nr):
        return arr_tfirst.transpose(0, 2, 1)
    raise RuntimeError(
        f"Spatial dimensions {arr_tfirst.shape[1:]} do not match Nr={Nr}, Nz={Nz}"
    )


def read_run(filename, include_extra_fields=False):
    with h5py.File(filename, "r") as f:
        t = f["Times"][:]
        r = f["r"][:]
        Z = f["Z"][:]
        rho_raw = f["GasDensity_History"][:]
        e_raw = f["e_Lya_History"][:] if "e_Lya_History" in f else None
        if include_extra_fields:
            Tgas_raw = f["T_History"][:]
            xHII_raw = f["xHII_History"][:]
            xH2_raw = f["xH2_History"][:]

    rho_hist = reorder_history_3d(rho_raw, t, r, Z)
    nH_hist = X * rho_hist / mH
    e_hist = reorder_history_3d(e_raw, t, r, Z) if e_raw is not None else None
    saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]

    run = dict(
        t=t, r=r, Z=Z, nH_hist=nH_hist,
        eLya_hist=e_hist, saved_idx=saved_idx,
    )

    if include_extra_fields:
        run["Tgas_hist"] = reorder_history_3d(Tgas_raw, t, r, Z)
        xHII_hist = reorder_history_3d(xHII_raw, t, r, Z)
        xH2_hist = reorder_history_3d(xH2_raw, t, r, Z)
        run["xHI_hist"] = np.clip(1.0 - xHII_hist - 2.0*xH2_hist, 0.0, 1.0)
    return run


def nearest_saved_snapshot(run, target_time_kyr):
    idx = run["saved_idx"]
    if idx.size == 0:
        raise RuntimeError("No saved snapshots found")
    return idx[np.argmin(np.abs(run["t"][idx] - target_time_kyr*kyr))]


def fmt_log_tick(val):
    exp = int(np.round(np.log10(val)))
    if exp == -1:
        return r"$0.1$"
    if exp == 0:
        return r"$1$"
    if exp == 1:
        return r"$10$"
    return rf"$10^{{{exp}}}$"


def add_outline_text(ax, x, y, text, fontsize, **kwargs):
    defaults = dict(
        color="white", ha="center", va="center", fontsize=fontsize,
        zorder=30,
        path_effects=[pe.withStroke(linewidth=3.0, foreground="black")],
    )
    defaults.update(kwargs)
    return ax.text(x, y, text, **defaults)


def add_clock_dividers(ax, radius):
    for lw, color, z in [(5.0, "white", 20), (2.0, "black", 21)]:
        ax.plot([-radius, radius], [0, 0], color=color, lw=lw,
                solid_capstyle="butt", zorder=z)
        ax.plot([0, 0], [-radius, radius], color=color, lw=lw,
                solid_capstyle="butt", zorder=z)
        ax.add_patch(Circle((0, 0), radius, fill=False,
                            edgecolor=color, linewidth=lw, zorder=z))


def add_clockwise_arrow(ax, radius):
    if not ShowTimeArrow:
        return
    ax.add_patch(FancyArrowPatch(
        (-0.32*radius, 1.08*radius), (0.32*radius, 1.08*radius),
        connectionstyle="arc3,rad=0.42", arrowstyle="-|>",
        mutation_scale=18, linewidth=1.8,
        color="black", clip_on=False, zorder=50,
    ))
    ax.text(0, 1.23*radius, "time", ha="center", va="center",
            fontsize=0.78*fontsize_time, color="black", clip_on=False)


def add_scale_bar(ax, radius):
    if not ShowScaleBar:
        return
    x0 = -0.78*radius
    x1 = x0 + scale_bar_length_pc
    y = -0.78*radius
    effects = [pe.withStroke(linewidth=4, foreground="black")]
    ax.plot([x0, x1], [y, y], color="white", lw=3,
            solid_capstyle="butt", path_effects=effects, zorder=40)
    ax.text(0.5*(x0+x1), y+0.06*radius, scale_bar_label,
            color="white", fontsize=0.78*fontsize_time,
            ha="center", va="bottom", path_effects=effects, zorder=41)


def plot_clock_panel(ax, field_hist, norm, cmap, title,
                     snapshot_idx, radius, R_edges, Z_edges,
                     outside_circle, quadrants, label_positions,
                     times, clip_min=None, clip_max=None):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.03*radius, 1.03*radius)
    ax.set_ylim(-1.03*radius, 1.03*radius)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    first_im = None
    for i, (k, (sx, sy)) in enumerate(zip(snapshot_idx, quadrants)):
        field = field_hist[k].T
        if clip_min is not None:
            field = np.maximum(field, clip_min)
        if clip_max is not None:
            field = np.minimum(field, clip_max)
        field_masked = np.ma.masked_where(outside_circle.T, field)

        im = ax.pcolormesh(sx*R_edges, sy*Z_edges, field_masked,
                           norm=norm, shading="auto", cmap=cmap,
                           rasterized=True, zorder=1)
        if first_im is None:
            first_im = im

        tx, ty = label_positions[i]
        add_outline_text(ax, tx*radius, ty*radius,
                         f"t = {times[k]/kyr:.0f} kyr", fontsize_time)

    add_clock_dividers(ax, radius)
    if ShowSourceMarker:
        ax.plot(0, 0, marker="*", markersize=25,
                markerfacecolor="white", markeredgecolor="black",
                markeredgewidth=1.2, zorder=60)
    add_clockwise_arrow(ax, radius)
    add_scale_bar(ax, radius)
    ax.set_title(title, fontsize=fontsize_title, pad=12)
    return first_im


def add_original_colorbar(fig, mappable, rect, ticks, labels, bar_label):
    """Original compact 0.31-wide, 0.03-high bar, not a GridSpec column."""
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_ticks(ticks)
    cb.set_ticklabels(labels)
    cb.ax.tick_params(axis="x", which="both", labelsize=fontsize_cbar_ticks,
                      length=0, pad=10)
    cb.ax.minorticks_off()
    cb.set_label(bar_label, fontsize=fontsize_cbar_label, labelpad=5)
    return cb

# ============================================================
# READ DATA / NORMALIZATIONS (as in original)
# ============================================================

run_Lya = read_run(Filename_Lya, include_extra_fields=ShowExtraRow)
run_noLya = read_run(Filename_noLya)

if run_Lya["eLya_hist"] is None:
    raise RuntimeError("e_Lya_History not found in Ly-alpha run")
if not (np.allclose(run_Lya["r"], run_noLya["r"])
        and np.allclose(run_Lya["Z"], run_noLya["Z"])):
    raise RuntimeError("Ly-alpha and no-Ly-alpha grids do not match")

snapshot_idx = [nearest_saved_snapshot(run_Lya, x) for x in SnapshotTimes_kyr]
print("Clock snapshots:")
for target, k in zip(SnapshotTimes_kyr, snapshot_idx):
    print(f"  requested {target:6.2f} kyr -> saved {run_Lya['t'][k]/kyr:8.3f} kyr")

nH0_noLya = run_noLya["nH_hist"][0]
nH0_Lya = run_Lya["nH_hist"][0]
pos_no = nH0_noLya[nH0_noLya > 0]
pos_ly = nH0_Lya[nH0_Lya > 0]
if pos_no.size == 0 or pos_ly.size == 0:
    raise ValueError("Initial nH field has no positive values")

nH0_char = max(np.max(pos_no), np.max(pos_ly))
vmin_nH = 10**(np.log10(nH0_char)-2.0-0.5)
vmax_nH = 10**(np.log10(nH0_char)+2.0-0.5)
norm_nH = LogNorm(vmin=vmin_nH, vmax=vmax_nH)
ticks_nH = 10.0**np.arange(int(np.ceil(np.log10(vmin_nH))),
                           int(np.floor(np.log10(vmax_nH)))+1)

norm_eLya = LogNorm(vmin=1e-14, vmax=1e-6)
ticks_eLya = [1e-14, 1e-10, 1e-6]

# ============================================================
# COORDINATES / CLOCK QUADRANTS
# ============================================================

r = run_Lya["r"]/pc
Z = run_Lya["Z"]/pc
radius = DomainFraction * min(r[-1], Z[-1])
R_edges, Z_edges = np.meshgrid(r, Z, indexing="ij")
r_cent = 0.5*(r[:-1]+r[1:])
Z_cent = 0.5*(Z[:-1]+Z[1:])
Rc, Zc = np.meshgrid(r_cent, Z_cent, indexing="ij")
outside_circle = Rc**2 + Zc**2 > radius**2
quadrants = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
label_positions = [(0.48, 0.48), (0.48, -0.48),
                   (-0.48, -0.48), (-0.48, 0.48)]

# ============================================================
# FIGURE: REPEAT ORIGINAL FIGURE GEOMETRY FOR EACH ROW
# ============================================================

nrows = 2 if ShowExtraRow else 1
FigureHeight_inches = nrows*RowHeight_inches + (nrows-1)*RowGap_inches
fig = plt.figure(figsize=(FigureWidth_inches, FigureHeight_inches))

# A rectangle in the OLD 12.6 x 6.4-in figure's normalized coordinates
# is mapped into one row in the NEW taller figure, preserving sizes.
def row_rect(original_rect, row_index):
    x, y, w, h = original_rect
    bottom_inches = (nrows-1-row_index)*(RowHeight_inches+RowGap_inches)
    return [x,
            (bottom_inches + y*RowHeight_inches)/FigureHeight_inches,
            w,
            h*RowHeight_inches/FigureHeight_inches]

specs = [
    dict(row=0, col=0, title=r"Gas density, $n_H$",
         hist=run_Lya["nH_hist"], norm=norm_nH, cmap=cmap_nH,
         ticks=ticks_nH, cbar_label=r"$n_H$ (cm$^{-3}$)",
         clip_min=vmin_nH, clip_max=vmax_nH),
    dict(row=0, col=1, title=r"Ly$\alpha$ energy density, $e_{Ly\alpha}$",
         hist=run_Lya["eLya_hist"], norm=norm_eLya, cmap=cmap_eLya,
         ticks=ticks_eLya, cbar_label=r"$e_{Ly\alpha}$ (erg cm$^{-3}$)",
         clip_min=1e-30, clip_max=None),
]

if ShowExtraRow:
    specs.extend([
        dict(row=1, col=0, title=r"Gas temperature, $T$",
             hist=run_Lya["Tgas_hist"],
             norm=LogNorm(vmin=Tgas_vmin, vmax=Tgas_vmax),
             cmap=cmap_Tgas,
             ticks=[1, 10, 1e2, 1e3, 1e4, 1e5],
             cbar_label=r"$T$ (K)",
             clip_min=Tgas_vmin, clip_max=Tgas_vmax),
        dict(row=1, col=1, title=r"Neutral fraction, $x_{HI}$",
             hist=run_Lya["xHI_hist"],
             norm=LogNorm(vmin=xHI_vmin, vmax=xHI_vmax),
             cmap=cmap_xHI, ticks=[1e-8, 1e-4, 1],
             cbar_label=r"$x_{HI}$",
             clip_min=xHI_vmin, clip_max=xHI_vmax),
    ])

# EXACT original axis and colorbar x coordinates and widths.
clock_rects = [(0.055, 0.16, 0.44, 0.76),
               (0.505, 0.16, 0.44, 0.76)]
bar_rects = [(0.115, 0.078, 0.31, 0.030),
             (0.575, 0.078, 0.31, 0.030)]

for spec in specs:
    ax = fig.add_axes(row_rect(clock_rects[spec["col"]], spec["row"]))
    mappable = plot_clock_panel(
        ax=ax, field_hist=spec["hist"], norm=spec["norm"],
        cmap=spec["cmap"], title=spec["title"],
        snapshot_idx=snapshot_idx, radius=radius,
        R_edges=R_edges, Z_edges=Z_edges,
        outside_circle=outside_circle, quadrants=quadrants,
        label_positions=label_positions, times=run_Lya["t"],
        clip_min=spec["clip_min"], clip_max=spec["clip_max"],
    )
    add_original_colorbar(
        fig, mappable,
        row_rect(bar_rects[spec["col"]], spec["row"]),
        spec["ticks"], [fmt_log_tick(v) for v in spec["ticks"]],
        spec["cbar_label"],
    )

# ============================================================
# SAVE
# ============================================================

plt.savefig(FigureName, bbox_inches="tight", dpi=600, facecolor="white")
print(f"Saved figure to {FigureName}")
plt.show()
