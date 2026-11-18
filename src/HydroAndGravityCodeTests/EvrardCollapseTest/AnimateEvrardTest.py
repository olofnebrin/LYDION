import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm

from pathlib import Path
import sys

G = 6.67430e-8  # Gravitational constant in cgs units (cm^3 g^-1 s^-2)

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


#cmastro_path = add_cmastro_to_path()

# small workaround to ensure that the colormap is registered in older versions of matplotlib
import matplotlib as mpl
import matplotlib.cm as cm

if not hasattr(cm, "register_cmap"):
    def _register_cmap(name=None, cmap=None, **kwargs):
        mpl.colormaps.register(cmap, name=name, force=True)
    cm.register_cmap = _register_cmap

import cmastro
# ---------------- User inputs ----------------

script_dir = os.path.dirname(os.path.abspath(__file__))

filename = os.path.join(
    script_dir,
    "EvrardTest_128x128_Zhalf_t0.80.h5",
)

frameskip = 1

save_mp4 = True

outname = os.path.join(
    script_dir,
    "EvrardCollapseDensity.mp4",
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
    rCoM    = f["rCoM"][:]
    zCoM    = f["ZCoM"][:]
    rho_raw = f["GasDensity_History"][:]
    rho_profile = f["SphericalAverageDensity"][:]
    P_profile   = f["SphericalAveragePressure"][:]
    v_r         = f["rVelocity_History"][:]
    v_z         = f["ZVelocity_History"][:]
    P           = f["P_History"][:]
    ΔV          = f["CellVolumes"][:]
    Mcl = f["parameters"].attrs["Mcl"]
    Rcl = f["parameters"].attrs["Rcl"]

rho_u = Mcl / Rcl**3
t_u   = np.sqrt(Rcl**3 / (G * Mcl))
v_u   = Rcl / t_u
P_u   = rho_u * G * Mcl / Rcl


# Canonical (Nt, Nr, Nz) ordering, code units
rho = ensure_NtNrNz(rho_raw, t, r_edges, z_edges) / rho_u
v_r = ensure_NtNrNz(v_r,  t, r_edges, z_edges) / v_u
v_z = ensure_NtNrNz(v_z,  t, r_edges, z_edges) / v_u
P   = ensure_NtNrNz(P,   t, r_edges, z_edges) / P_u
#ΔV = ensure_NtNrNz(ΔV,  t, r_edges, z_edges) / Rcl**3

rho_profile /= rho_u
P_profile /= P_u
rCoM /= Rcl
zCoM /= Rcl

Zc = 0#0.5 * (z_edges[0] + z_edges[-1]) / Rcl 
print("v_r shape:", v_r.shape)
print("v_z shape:", v_z.shape)
# Get radial velocity profile
mesh = np.meshgrid(rCoM, zCoM-Zc, indexing='ij')
print("Mesh shape:", mesh[0].shape, mesh[1].shape)
R = np.sqrt(mesh[0]**2 + mesh[1]**2)
print("R shape:", R.shape)
rr = mesh[0] / R
zz = mesh[1] / R
v_R = v_r * rr + v_z * zz
print("v_R shape:", v_R.shape)
# Radial binning for radial velocity profile
v_r_profile = np.zeros((len(t), len(rCoM)))
R_bins = np.linspace(0, np.max(R), len(rCoM)+1)
for i in range(len(t)):
    for j in range(len(rCoM)):
        mask = (R >= R_bins[j]) & (R < R_bins[j+1])
        if np.any(mask):
            v_r_profile[i, j] = np.mean(v_R[i][mask]) 
        else:
            v_r_profile[i, j] = 0.0
gamma = 5/3
s = P_profile/rho_profile**gamma
plot_domain = (rCoM <= 0.8)
print("Cloud mass Mcl:", Mcl)
print("Cloud radius Rcl:", Rcl)
print("Rho profile shape:", rho_profile.shape)
print("Radial grid shape:", r_edges.shape)
print("Z grid shape:", z_edges.shape)
print("Radial cell centers shape:", rCoM.shape)
print("Radial velocity shape:", v_r.shape)
print("Radial velocity profile shape:", v_r_profile.shape)

fig, ax = plt.subplots(1, 3, figsize=(16, 5))

ax[0].plot(rCoM, rho_profile[-1, :].T)
ax[0].set_xlabel("r / Rcl")
ax[0].set_ylabel(r"$\rho / \rho_u$")
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[1].plot(rCoM, v_r_profile[-1, :].T)
ax[1].set_xlabel("r / Rcl")
ax[1].set_ylabel(r"$v_r / v_u$")
ax[1].set_xscale("log")
ax[2].plot(rCoM, s[-1, :])
#ax[2].plot(rCoM_cloud, P_profile[-1, :].T[cloud])
#ax[2].plot(rCoM_cloud, s[-1, :].T[cloud])
ax[2].set_xlabel("r / Rcl")
ax[2].set_ylabel(r"$P / \rho^{\gamma}$")
ax[2].set_xscale("log")
#ax[2].set_yscale("log")
plt.tight_layout()
plt.show()
"""
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
        "sans-serif": ["AVHershey Complex",  "DejaVu Sans"],
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

"""