# PlotDensityTemperature_TwoPanel_Cinematic_Snapshots.py
#
# Static cinematic figure showing density + temperature
# at 1--3 selected times.
#
# Each row:
#   Left  panel: density (nH)
#   Right panel: temperature (T)
#
# The aspect ratio and visual layout of EACH ROW are the same
# as in AnimateDensityTemperature_TwoPanel_Cinematic_ViewCentered.py.
#
# Example:
#
# target_times_kyr = [0.0]
#     -> one density-temperature row
#
# target_times_kyr = [0.0, 10.0]
#     -> two rows
#
# target_times_kyr = [0.0, 10.0, 20.0]
#     -> three rows

import h5py
import numpy as np
from matplotlib.colors import LogNorm
from pathlib import Path
import sys
import os

repo_root = Path(__file__).resolve().parents[1]
os.chdir(repo_root)
cmastro_src = repo_root / "src" / "cmastro" / "src"

if cmastro_src.exists():
    sys.path.insert(0, str(cmastro_src))

import cmastro

from PlotCosmetics import (
    ApplyFontStyle,
    AddTimeLabel,
    InsetColorBar,
    AddScaleBar,
    cmap_for,
    AddStar,
)

# ============================================================
#                       USER INPUTS
# ============================================================

filename = "RHDSim_NoLya_512x1024_2026-07-09.h5"

# Target times in kyr.
#
# Enter between 1 and 3 values.
# The nearest stored simulation snapshot is used.
target_times_kyr = [10.0, 30.0, 60.0]

save_figure = True
outname     = "density_temperature_cinematic_snapshots.png"
dpi         = 300


# ------------------------------------------------------------
# Figure geometry
# ------------------------------------------------------------

# Horizontal gap between density and temperature panels,
# as a fraction of the full figure width.
panel_gap = 0.00

# Vertical gap between snapshot rows,
# as a fraction of ONE row height.
#
# Examples:
#   0.00 -> rows touch
#   0.02 -> small gap
#   0.05 -> slightly stronger separation
row_gap = 0.02


# ------------------------------------------------------------
# View
# ------------------------------------------------------------

# View half-size (pc)
half_pc = 2.0

# Scale bar
scalebar_pc = 0.5

# Dataset convention
midplane_symmetry = False   # True => center at z=0
                            # False => center at z=Zmax/2

# Pretty mirroring options
mirror_r = True             # show ±r
mirror_z = False            # show ±z


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

mH  = 1.67e-24
X   = 0.76
yr  = 3.1556926e7
kyr = 1e3 * yr
pc  = 3.085677581e18

eps = 1e-30  # protect LogNorm


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
        f"Unexpected array shape {arr.shape} "
        f"vs edges Nr={Nr}, Nz={Nz}"
    )


def mirror_edges_from_zero(edges):
    """
    edges assumed to start at 0.

    Returns [-L ... L] without duplicating zero.
    """
    neg = -edges[:0:-1]
    return np.concatenate([neg, edges])


def mirror_field(A, mirror_r=True, mirror_z=True):
    B = A

    if mirror_r:
        B = np.concatenate([B[::-1, :], B], axis=0)

    if mirror_z:
        B = np.concatenate([B[:, ::-1], B], axis=1)

    return B


def CinematicAxesSnapshotRows(
    nrows,
    row_figsize=(16, 8),
    panel_gap=0.02,
    row_gap=0.03,
):
    """
    Create N rows of cinematic density-temperature panels.

    Each individual row has the same effective geometry as
    your original 16x8 two-panel animation figure.

    Parameters
    ----------
    nrows : int
        Number of snapshot rows.

    row_figsize : tuple
        Physical figure size of one row, e.g. (16, 8).

    panel_gap : float
        Horizontal gap between density and temperature panels,
        expressed as a fraction of the total figure width.

    row_gap : float
        Vertical gap between rows, expressed as a fraction of
        one row height.

    Returns
    -------
    fig
    axes
        List of [(axL, axR), ...], one pair for each time.
    """

    import matplotlib.pyplot as plt

    row_width, row_height = row_figsize

    # Increase total figure height to accommodate gaps while
    # preserving the physical height of every individual row.
    total_height = row_height * (
        nrows + row_gap * (nrows - 1)
    )

    fig = plt.figure(
        figsize=(row_width, total_height),
        facecolor="white",
    )

    fig.patch.set_facecolor("white")

    # Fractional height occupied by one actual row.
    row_h = 1.0 / (
        nrows + row_gap * (nrows - 1)
    )

    # Fractional gap between rows.
    gap_h = row_gap * row_h

    # Same horizontal geometry as original two-panel figure.
    panel_w = (1.0 - panel_gap) / 2.0

    axes = []

    for row in range(nrows):

        # Start from top row and move downward.
        y0 = 1.0 - (row + 1) * row_h - row * gap_h

        axL = fig.add_axes([
            0.0,
            y0,
            panel_w,
            row_h,
        ])

        axR = fig.add_axes([
            panel_w + panel_gap,
            y0,
            panel_w,
            row_h,
        ])

        axL.set_axis_off()
        axR.set_axis_off()

        axes.append((axL, axR))

    return fig, axes


# ============================================================
#                       READ DATA
# ============================================================

with h5py.File(filename, "r") as f:

    t        = f["Times"][:]
    rho_hist = f["GasDensity_History"][:]
    T_hist   = f["T_History"][:]

    r = f["r"][:]
    Z = f["Z"][:]


Nr = len(r) - 1
Nz = len(Z) - 1

rho_hist = ensure_NtNrNz(
    rho_hist,
    Nr,
    Nz,
)

T_hist = ensure_NtNrNz(
    T_hist,
    Nr,
    Nz,
)


# Number density
nH_hist = X * rho_hist / mH


# ============================================================
#                    SELECT SNAPSHOTS
# ============================================================

target_times_kyr = np.atleast_1d(
    np.asarray(target_times_kyr, dtype=float)
)

if not (1 <= len(target_times_kyr) <= 3):
    raise ValueError(
        "target_times_kyr must contain between 1 and 3 times."
    )


# Ignore unsaved zero-filled history slots, except index 0.
saved_idx = np.where(
    (t > 0.0) | (np.arange(t.size) == 0)
)[0]

saved_times_kyr = t[saved_idx] / kyr


snapshot_idx = []

for target in target_times_kyr:

    i_nearest = np.argmin(
        np.abs(saved_times_kyr - target)
    )

    k = saved_idx[i_nearest]
    snapshot_idx.append(k)

    print(
        f"Requested t = {target:.3f} kyr "
        f"-> using t = {t[k] / kyr:.3f} kyr "
        f"(index {k})"
    )


# ============================================================
#                       DENSITY RANGE
# ============================================================

# Use the initial snapshot to define a characteristic
# initial cloud density, then center the range around it.

nH0 = nH_hist[0]

nH0_pos = nH0[nH0 > 0.0]

if nH0_pos.size == 0:
    raise ValueError(
        "Initial nH field has no positive values."
    )

nH0_char = np.max(nH0_pos)


# Density range:
# 2 dex below and above the initial cloud density.

delta = 0.4

vmin_nH = 10.0**(
    np.log10(nH0_char) - 2.0 - delta
)

vmax_nH = 10.0**(
    np.log10(nH0_char) + 2.0 - delta
)

norm_nH = LogNorm(
    vmin=vmin_nH,
    vmax=vmax_nH,
)


# ============================================================
#                     TEMPERATURE RANGE
# ============================================================

T_vmin = 1.0
T_vmax = 1e5

norm_T = LogNorm(
    vmin=T_vmin,
    vmax=T_vmax,
    clip=True,
)


# ============================================================
#                     PLOTTING GEOMETRY
# ============================================================

# Edges in pc
r_edges = r / pc
z_edges = Z / pc


# Camera center
if midplane_symmetry:

    r_center = 0.0
    z_center = 0.0

else:

    r_center = 0.0
    z_center = 0.5 * z_edges[-1]


# Mirrored plotting edges
r_plot = (
    mirror_edges_from_zero(r_edges)
    if mirror_r
    else r_edges
)

z_plot = (
    mirror_edges_from_zero(z_edges)
    if mirror_z
    else z_edges
)


Rplot, Zplot = np.meshgrid(
    r_plot,
    z_plot,
    indexing="ij",
)


# ------------------------------------------------------------
# View limits
# ------------------------------------------------------------

if mirror_r:

    ax_xlim = (
        -half_pc,
        half_pc,
    )

else:

    # Keep the same physical span as ylim
    # (= 2 * half_pc), but with r >= 0.

    ax_xlim = (
        max(r_edges[0], r_center),
        r_center + 2.0 * half_pc,
    )


if mirror_z:

    ax_ylim = (
        -half_pc,
        half_pc,
    )

else:

    ax_ylim = (
        max(
            z_edges[0],
            z_center - half_pc,
        ),
        z_center + half_pc,
    )


# ============================================================
#                         FIGURE
# ============================================================

ApplyFontStyle(
    theme="cinematic",
    font_size=32,
)


nrows = len(snapshot_idx)

fig, axes = CinematicAxesSnapshotRows(
    nrows=nrows,
    row_figsize=(16, 8),
    panel_gap=panel_gap,
    row_gap=row_gap,
)


# ============================================================
#                      DRAW EACH ROW
# ============================================================

for row, k in enumerate(snapshot_idx):

    axL, axR = axes[row]

    # --------------------------------------------------------
    # Axis geometry
    # --------------------------------------------------------

    for ax in (axL, axR):

        ax.set_xlim(*ax_xlim)
        ax.set_ylim(*ax_ylim)

        ax.set_aspect(
            "equal",
            adjustable="box",
        )


    # --------------------------------------------------------
    # Fields
    # --------------------------------------------------------

    A = mirror_field(
        np.clip(
            nH_hist[k],
            eps,
            None,
        ),
        mirror_r=mirror_r,
        mirror_z=mirror_z,
    )

    T = mirror_field(
        np.clip(
            T_hist[k],
            eps,
            None,
        ),
        mirror_r=mirror_r,
        mirror_z=mirror_z,
    )


    # --------------------------------------------------------
    # Density
    # --------------------------------------------------------

    imL = axL.pcolormesh(
        Rplot,
        Zplot,
        A,
        norm=norm_nH,
        shading="auto",
        cmap=cmap_for("density"),
    )


    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    imR = axR.pcolormesh(
        Rplot,
        Zplot,
        T,
        norm=norm_T,
        shading="auto",
        cmap=cmap_for("temperature"),
    )


    # --------------------------------------------------------
    # Star
    # --------------------------------------------------------

    star_r_pc = (
        0.0
        if mirror_r
        else 0.002 * ax_xlim[1]
    )

    star_z_pc = (
        0.0
        if midplane_symmetry
        else z_center
    )

    #AddStar(
    #    axL,
    #    star_r_pc,
    #    star_z_pc,
    #    size=100,
    #)

    #AddStar(
    #    axR,
    #    star_r_pc,
    #    star_z_pc,
    #    size=100,
    #)


    # --------------------------------------------------------
    # Time label
    # --------------------------------------------------------

    AddTimeLabel(
        axL,
        f"t = {t[k] / kyr:.2f} kyr",
        pos=(0.03, 0.97),
        fontsize=22,
    )


    # --------------------------------------------------------
    # Scale bar
    # --------------------------------------------------------

    if mirror_r:

        pos_ScaleBar = (
            0.08,
            0.15,
        )

    else:

        pos_ScaleBar = (
            0.65,
            0.10,
        )


    AddScaleBar(
        axL,
        length=scalebar_pc,
        label=f"{scalebar_pc:.1f} pc",
        pos=pos_ScaleBar,
        fontsize=18,
    )


    # --------------------------------------------------------
    # Colorbars
    # --------------------------------------------------------
    #
    # InsetColorBar expects figure coordinates for rect.
    #
    # Therefore the vertical position must be transformed
    # separately for each row.
    #
    # Original single-row positions:
    #
    # Density:
    #     rect=(0.43, 0.10, 0.012, 0.20)
    #
    # Temperature:
    #     rect=(0.90, 0.10, 0.012, 0.20)
    #
    # We preserve those positions relative to each row.

    ax_pos = axL.get_position()

    row_y0 = ax_pos.y0
    row_h  = ax_pos.height

    cb_y = row_y0 + 0.10 * row_h
    cb_h = 0.20 * row_h


    # Density colorbar

    cbL = InsetColorBar(
        fig,
        axL,
        imL,
        rect=(
            0.40,
            cb_y,
            0.012,
            cb_h,
        ),
        label=r"n$_{H}$ (cm$^{-3}$)",
        labelsize=18,
        tick_labelsize=18,
    )

    cbL.set_ticks([
        10.0,
        1e3,
        1e5,
    ])

    cbL.set_ticklabels([
        r"10",
        r"$10^{3}$",
        r"$10^{5}$",
    ])


    # Temperature colorbar

    cbR = InsetColorBar(
        fig,
        axR,
        imR,
        rect=(
            0.90,
            cb_y,
            0.012,
            cb_h,
        ),
        label=r"T (K)",
        labelsize=18,
        tick_labelsize=18,
        labelcolor="black",
        tickcolors="black",
    )

    cbR.set_ticks([
        10.0,
        1e2,
        1e3,
        1e4,
    ])

    cbR.set_ticklabels([
        r"10",
        r"$10^{2}$",
        r"$10^{3}$",
        r"$10^{4}$",
    ])


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