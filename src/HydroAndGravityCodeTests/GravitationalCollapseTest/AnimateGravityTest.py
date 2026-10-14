import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm

from pathlib import Path
import sys


def add_cmastro_to_path():
    here = Path(__file__).resolve()

    for parent in [here.parent, *here.parents]:
        candidate = parent / "src" / "cmastro" / "src"

        if (candidate / "cmastro" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return str(candidate)

    raise RuntimeError(
        "Could not find src/cmastro/src containing cmastro/__init__.py"
    )


cmastro_path = add_cmastro_to_path()
import cmastro


# ---------------- User inputs ----------------

script_dir = os.path.dirname(os.path.abspath(__file__))

filename = os.path.join(
    script_dir,
    "GravityTest_128x256_Zfull_t0.07.h5",
)

frameskip = 1

save_mp4 = True

outname = os.path.join(
    script_dir,
    "GravCollapseDensity.mp4",
)

fps = 20

use_logrho = True

rho_floor = 1e-30

# ---------------------------------------------


def ensure_NtNrNz(arr, t, r_edges, z_edges):
    """
    Convert HDF5 history array to canonical LYDION Python ordering:

        (Nt, Nr, Nz)

    LYDION Julia arrays are physically ordered (r, Z), with histories
    written as (r, Z, time).

    HDF5.jl / h5py commonly exposes these as:

        (time, Z, r)

    so the spatial axes must then be swapped once.
    """

    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3D history array, got shape {arr.shape}"
        )

    Nt = len(t)
    Nr = len(r_edges) - 1
    Nz = len(z_edges) - 1

    # Most common h5py layout for LYDION HDF5 output:
    # (Nt, Nz, Nr)
    if arr.shape == (Nt, Nz, Nr):
        return arr.transpose(0, 2, 1)

    # Already in desired Python ordering:
    # (Nt, Nr, Nz)
    if arr.shape == (Nt, Nr, Nz):
        return arr

    # Julia-style ordering exposed directly:
    # (Nr, Nz, Nt)
    if arr.shape == (Nr, Nz, Nt):
        return arr.transpose(2, 0, 1)

    # Fully reversed alternative:
    # (Nz, Nr, Nt)
    if arr.shape == (Nz, Nr, Nt):
        return arr.transpose(2, 1, 0)

    raise ValueError(
        f"Unexpected history shape {arr.shape}. "
        f"Expected dimensions Nt={Nt}, Nr={Nr}, Nz={Nz}."
    )


# ============================================================
# READ DATA
# ============================================================

with h5py.File(filename, "r") as f:

    t = f["Times"][:]

    r_edges = f["r"][:]
    z_edges = f["Z"][:]

    rho_raw = f["GasDensity_History"][:]


print("Raw HDF5 density shape:", rho_raw.shape)
print(
    "Grid cells:",
    f"Nr={len(r_edges) - 1},",
    f"Nz={len(z_edges) - 1},",
    f"Nt={len(t)}",
)


# Convert once to:
#
#     rho_hist[time, r, Z]
#
rho_hist = ensure_NtNrNz(
    rho_raw,
    t,
    r_edges,
    z_edges,
)


Nt, Nr, Nz = rho_hist.shape


print("Canonical rho_hist shape:", rho_hist.shape)
print(
    "Expected canonical shape:",
    (
        len(t),
        len(r_edges) - 1,
        len(z_edges) - 1,
    ),
)


if Nr != len(r_edges) - 1:
    raise ValueError(
        f"Density Nr={Nr}, but r grid has "
        f"{len(r_edges) - 1} cells"
    )

if Nz != len(z_edges) - 1:
    raise ValueError(
        f"Density Nz={Nz}, but Z grid has "
        f"{len(z_edges) - 1} cells"
    )


frames = np.arange(
    0,
    Nt,
    frameskip,
)


print(
    f"Loaded {Nt} snapshots; "
    f"animating {len(frames)} frames."
)


# ============================================================
# PLOT COSMETICS
# ============================================================

plt.rc(
    "font",
    **{
        "sans-serif": ["AVHershey Complex"],
        "size": 37,
    }
)

params = {
    "mathtext.default": "regular",
}

plt.rcParams.update(params)

plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "GreekC"

params = {
    "legend.fontsize": 19,
}

plt.rcParams.update(params)

plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["xtick.major.width"] = 1.7
plt.rcParams["xtick.minor.width"] = 1.7
plt.rcParams["xtick.major.size"] = 20
plt.rcParams["xtick.minor.size"] = 8

plt.rcParams["ytick.major.width"] = 1.7
plt.rcParams["ytick.minor.width"] = 1.7
plt.rcParams["ytick.major.size"] = 20
plt.rcParams["ytick.minor.size"] = 8

plt.rcParams["xtick.major.pad"] = "15"
plt.rcParams["ytick.major.pad"] = "15"

plt.rcParams["legend.frameon"] = False

plt.rcParams["axes.linewidth"] = 2.5

plt.rcParams["grid.linewidth"] = 2.5
plt.rcParams["grid.linestyle"] = "--"

plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.bottom"] = True

plt.rcParams["ytick.left"] = True
plt.rcParams["ytick.right"] = True

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


# ============================================================
# PLOTTING SETUP
# ============================================================

fig, ax = plt.subplots(
    1,
    1,
    figsize=(11.5, 10.0),
    constrained_layout=True,
)


ax.set_xlim(
    r_edges[0],
    r_edges[-1],
)

ax.set_ylim(
    z_edges[0],
    z_edges[-1],
)

ax.set_aspect(
    "equal",
    adjustable="box",
)


ax.set_xlabel(
    "r",
    labelpad=20,
)

ax.set_ylabel(
    "Z",
    labelpad=20,
)


# ============================================================
# COLOR SCALING
# ============================================================

rho_all = rho_hist

rho_min = np.min(rho_all)
rho_max = np.max(rho_all)


print(
    f"rho range over all snapshots: "
    f"[{rho_min:.3e}, {rho_max:.3e}]"
)


if use_logrho:

    positive = rho_all[
        rho_all > 0
    ]

    if positive.size == 0:
        raise ValueError(
            "All densities are non-positive; cannot log-scale."
        )

    # vmin = max(
    #     rho_floor,
    #     np.min(positive),
    # )
    #
    # vmax = rho_max

    vmin = 1e8
    vmax = 1e12

    norm = LogNorm(
        vmin=vmin,
        vmax=vmax,
    )

    cb_label = r"$\rho$ (log)"

else:

    norm = None

    cb_label = r"$\rho$"


# ============================================================
# INITIAL FRAME
# ============================================================

# Internally:
#
#     rho_hist[k] -> (Nr, Nz)
#
# i.e. LYDION physical ordering:
#
#     rho[r, Z]
#
# Matplotlib with x=r and y=Z expects C as:
#
#     (Nz, Nr)
#
# hence ONLY the plotting transpose below.

im_rho = ax.pcolormesh(
    r_edges,
    z_edges,
    rho_hist[0].T,
    shading="auto",
    cmap="cma:laguna",
    norm=norm,
)


cb = fig.colorbar(
    im_rho,
    ax=ax,
    pad=0.02,
    shrink=0.7,
)

cb.set_label(
    cb_label
)


# ============================================================
# TIME LABEL
# ============================================================

txt = ax.text(
    0.05,
    0.93,
    "",
    color="white",
    fontsize=30,
    transform=ax.transAxes,
    path_effects=[
        pe.withStroke(
            linewidth=3,
            foreground="black",
        )
    ],
)


# ============================================================
# ANIMATION
# ============================================================

def init():

    im_rho.set_array(
        rho_hist[0].T.ravel()
    )

    txt.set_text(
        f"t = {t[0]:.3e}"
    )

    return im_rho, txt


def update(frame_index):

    k = frame_index

    im_rho.set_array(
        rho_hist[k].T.ravel()
    )

    txt.set_text(
        f"t = {t[k]:.3e}"
    )

    return im_rho, txt


ani = FuncAnimation(
    fig,
    update,
    frames=frames,
    init_func=init,
    blit=True,
)


# ============================================================
# SAVE / SHOW
# ============================================================

if save_mp4:

    ani.save(
        outname,
        writer="ffmpeg",
        fps=fps,
        dpi=300,
    )

    print(
        f"Saved animation to {outname}"
    )

else:

    plt.show()