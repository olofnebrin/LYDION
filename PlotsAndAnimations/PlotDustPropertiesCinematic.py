# PlotDustProperties_TwoPanel_Cinematic_Snapshots.py
#
# Static cinematic figure showing dust properties
# at 1--3 selected times.
#
# Each row:
#   Left  panel: total dust-to-gas ratio D / D_sun
#   Right panel: PAH mass fraction q_PAH (%)
#
# The aspect ratio and visual layout of EACH ROW are the same
# as in the density-temperature cinematic snapshot figure.

import h5py
import numpy as np
from matplotlib.colors import LogNorm, Normalize
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
outname     = "dust_properties_cinematic_snapshots.png"
dpi         = 300


# ------------------------------------------------------------
# Figure geometry
# ------------------------------------------------------------

# Horizontal white space between the two columns
panel_gap = 0.00

# Vertical white space between snapshot rows
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
# Dust plotting
# ------------------------------------------------------------

# Solar-neighbourhood dust-to-gas mass ratio
Dsun = 1.0 / 162.0

# Same colormaps as your dust-properties figure
cmap_DG   = "BuPu_r"
cmap_qPAH = "cubehelix"

# Dust-to-gas ratio in units of D_sun
DG_vmin = 0.1
DG_vmax = 10

# PAH percentage
qPAH_vmin = 0.0
qPAH_vmax = 10.0

norm_DG   = LogNorm(vmin=DG_vmin, vmax=DG_vmax)
norm_qPAH = Normalize(vmin=qPAH_vmin, vmax=qPAH_vmax)


# ------------------------------------------------------------
# Colorbar geometry
# ------------------------------------------------------------

# Horizontal position in full-figure coordinates
cbar_left_x  = 0.40
cbar_right_x = 0.90

# Position/size relative to each snapshot row
cbar_y_rel = 0.10
cbar_h_rel = 0.20

cbar_width = 0.012

cbar_labelsize     = 18
cbar_tick_labelsize = 18


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
    Create N rows of cinematic two-panel snapshots.

    Each individual row has the same effective geometry as
    the original 16x8 two-panel cinematic figure.
    """

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

    # Fractional height occupied by one actual row
    row_h = 1.0 / (
        nrows + row_gap * (nrows - 1)
    )

    # Fractional gap between rows
    gap_h = row_gap * row_h

    # Horizontal panel geometry
    panel_w = (1.0 - panel_gap) / 2.0

    axes = []

    for row in range(nrows):

        # Start from top row and move downward
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

    t = f["Times"][:]
    r = f["r"][:]
    Z = f["Z"][:]

    rho_raw = f["GasDensity_History"][:]

    rho_d_sil_raw = f["DustDensity_Sil_History"][:]
    rho_d_c_raw   = f["DustDensity_C_History"][:]
    rho_d_pah_raw = f["DustDensity_PAH_History"][:]


Nr = len(r) - 1
Nz = len(Z) - 1

rho_hist = ensure_NtNrNz(
    rho_raw,
    Nr,
    Nz,
)

rho_d_sil_hist = ensure_NtNrNz(
    rho_d_sil_raw,
    Nr,
    Nz,
)

rho_d_c_hist = ensure_NtNrNz(
    rho_d_c_raw,
    Nr,
    Nz,
)

rho_d_pah_hist = ensure_NtNrNz(
    rho_d_pah_raw,
    Nr,
    Nz,
)


# ============================================================
#                  DERIVED DUST QUANTITIES
# ============================================================

rho_d_tot_hist = (
    rho_d_sil_hist
    + rho_d_c_hist
    + rho_d_pah_hist
)


# Total dust-to-gas ratio, normalized to D_sun.
#
# Keep zero where gas density is zero rather than generating
# infinities. Those values will subsequently be clipped to the
# minimum plotting value.

DG_hist = np.divide(
    rho_d_tot_hist,
    rho_hist * Dsun,
    out=np.zeros_like(rho_d_tot_hist),
    where=(rho_hist > 0.0),
)


# PAH fraction of total dust mass, expressed in percent.

qPAH_hist = np.divide(
    100.0 * rho_d_pah_hist,
    rho_d_tot_hist,
    out=np.zeros_like(rho_d_pah_hist),
    where=(rho_d_tot_hist > 0.0),
)


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

    DG = mirror_field(
        np.clip(
            DG_hist[k],
            DG_vmin,
            DG_vmax,
        ),
        mirror_r=mirror_r,
        mirror_z=mirror_z,
    )


    qPAH = mirror_field(
        np.clip(
            qPAH_hist[k],
            qPAH_vmin,
            qPAH_vmax,
        ),
        mirror_r=mirror_r,
        mirror_z=mirror_z,
    )


    # --------------------------------------------------------
    # LEFT: total dust-to-gas ratio
    # --------------------------------------------------------

    imL = axL.pcolormesh(
        Rplot,
        Zplot,
        DG,
        norm=norm_DG,
        shading="auto",
        cmap=cmap_DG,
    )


    # --------------------------------------------------------
    # RIGHT: PAH fraction
    # --------------------------------------------------------

    imR = axR.pcolormesh(
        Rplot,
        Zplot,
        qPAH,
        norm=norm_qPAH,
        shading="auto",
        cmap=cmap_qPAH,
    )


    # --------------------------------------------------------
    # Star
    # --------------------------------------------------------
    #
    # Left commented out, as in your current cinematic script.

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

    # AddStar(
    #     axL,
    #     star_r_pc,
    #     star_z_pc,
    #     size=100,
    # )

    # AddStar(
    #     axR,
    #     star_r_pc,
    #     star_z_pc,
    #     size=100,
    # )


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
    # Colorbar geometry
    # --------------------------------------------------------

    ax_pos = axL.get_position()

    row_y0 = ax_pos.y0
    row_h  = ax_pos.height

    cb_y = row_y0 + cbar_y_rel * row_h
    cb_h = cbar_h_rel * row_h


    # --------------------------------------------------------
    # Dust-to-gas colorbar
    # --------------------------------------------------------

    cbL = InsetColorBar(
        fig,
        axL,
        imL,
        rect=(
            cbar_left_x,
            cb_y,
            cbar_width,
            cb_h,
        ),
        label=r"$\mathfrak{D}/\mathfrak{D}_{\odot}$",
        labelsize=cbar_labelsize,
        tick_labelsize=cbar_tick_labelsize,
        labelcolor="black",
        tickcolors="black",
    )

    cbL.set_ticks([
        0.1,
        1,
        10,
    ])

    cbL.set_ticklabels([
        r"0.1",
        r"1",
        r"10",
    ])

    cbL.ax.minorticks_off()

    cbL.outline.set_edgecolor("black")
    cbL.outline.set_linewidth(1.5)

    for spine in cbL.ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)


    # --------------------------------------------------------
    # q_PAH colorbar
    # --------------------------------------------------------

    cbR = InsetColorBar(
        fig,
        axR,
        imR,
        rect=(
            cbar_right_x,
            cb_y,
            cbar_width,
            cb_h,
        ),
        label=r"$q_{\rm PAH}$ (%)",
        labelsize=cbar_labelsize,
        tick_labelsize=cbar_tick_labelsize,
        labelcolor="white",
        tickcolors="white",
    )

    cbR.set_ticks([
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
    ])

    cbR.set_ticklabels([
        "0",
        "2",
        "4",
        "6",
        "8",
        "10",
    ])

    cbR.outline.set_edgecolor("white")
    cbR.outline.set_linewidth(1.5)

    for spine in cbR.ax.spines.values():
        spine.set_edgecolor("white")
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