"""Evrard collapse: volume-weighted spherical profiles and optional density animation.

Run from the EvrardCollapseTest directory, or from VS Code.
HDF5 produced by LYDION (Julia) has dataset axes reversed when read by h5py:
Julia (r, Z, time) -> Python (time, Z, r).

Comparison: LYDION-2D versus HydroCode1D (16,000 spherical radial cells).
HydroCode1D was run with G = Mcl = Rcl = 1, so its numerical output
is already in dimensionless Evrard units despite the SI column labels.
"""

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator, AutoMinorLocator
import matplotlib.patheffects as pe

G = 6.67430e-8  # cgs

###############################################################
#######           P L O T   C O S M E T I C S           #######
###############################################################
# Same Hershey fonts and general styling as the other hydro tests.
plt.rc('font', **{'family': 'sans-serif',
                  'sans-serif': ['AVHershey Complex'], 'size': 21})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

plt.rcParams.update({'legend.fontsize': 14})
plt.rcParams['legend.markerscale'] = 0.75
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 1.6
plt.rcParams['xtick.minor.width'] = 1.4
plt.rcParams['xtick.major.size'] = 12
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.major.width'] = 1.6
plt.rcParams['ytick.minor.width'] = 1.4
plt.rcParams['ytick.major.size'] = 12
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['xtick.major.pad'] = 10
plt.rcParams['ytick.major.pad'] = 10
plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.linewidth'] = 1.7
plt.rcParams['grid.linewidth'] = 1.1
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


# ---------------- User inputs ----------------
here = Path(__file__).resolve().parent
filename = here / "EvrardTest_512x512_Zhalf_t0.80.h5"

make_animation = False  # Set True to generate an MP4 after displaying the profiles

# The single, high-resolution HydroCode1D reference in Evrard units.
reference_file = here / "ReferenceSolutionEvrard" / "EvrardReferenceSolution.txt"

# Profile line appearance (easy to edit).
lydion_label = r"LYDION (512$^2$)"
lydion_color = "dodgerblue"
lydion_linewidth = 3.0
lydion_linestyle = "-"

reference_label = "Reference"   
reference_color = "black"
reference_linewidth = 3.0
reference_linestyle = "--"

profile_xlim = (0.01, 1.0)
profile_outname = here / "EvrardCollapseProfiles.pdf"
animation_outname = here / "EvrardCollapseDensity.mp4"
plot_dpi = 300
animation_dpi = 180
frameskip = 1
fps = 20

# Figure-title controls.
figure_title_fontsize = 28
figure_title_y = 0.9

# ---------------------------------------------------------------
# HELPER: DECIMAL LABELS ON COMMON LOG TICKS
# ---------------------------------------------------------------
def decimal_log_tick_formatter(x, pos):
    """Use decimal labels for 10^-2 ... 10^2, and 1eN outside that range."""
    if x <= 0 or not np.isfinite(x):
        return ""

    exponent = np.log10(x)
    if not np.isclose(exponent, np.round(exponent), atol=1e-10):
        return ""

    exponent = int(np.round(exponent))
    common = {
        -2: "0.01",
        -1: "0.1",
         0: "1",
         1: "10",
         2: "100",
    }
    if exponent in common:
        return common[exponent]
    return rf"$10^{{{exponent}}}$"


log_tick_formatter = FuncFormatter(decimal_log_tick_formatter)

# ---------------------------------------------------------------
# READ HDF5 (h5py reverses Julia's HDF5 dimension ordering)
# ---------------------------------------------------------------
with h5py.File(filename, "r") as f:
    t = np.asarray(f["Times"][:], dtype=np.float64)
    r_edges = np.asarray(f["r"][:], dtype=np.float64)
    z_edges = np.asarray(f["Z"][:], dtype=np.float64)
    rCoM = np.asarray(f["rCoM"][:], dtype=np.float64)
    zCoM = np.asarray(f["ZCoM"][:], dtype=np.float64)

    # Python array shape: (Nt, Nz, Nr), even when Nr == Nz.
    rho_hist_cgs = f["GasDensity_History"][:]
    ur_last_cgs = f["rVelocity_History"][-1, :, :].T
    uz_last_cgs = f["ZVelocity_History"][-1, :, :].T
    P_last_cgs = f["P_History"][-1, :, :].T

    Mcl = float(f["parameters"].attrs["Mcl"])
    Rcl = float(f["parameters"].attrs["Rcl"])

Nt = len(t)
Nr = len(r_edges) - 1
Nz = len(z_edges) - 1
assert rho_hist_cgs.shape == (Nt, Nz, Nr), (
    f"Unexpected density history shape: {rho_hist_cgs.shape}"
)
assert ur_last_cgs.shape == (Nr, Nz)

# Evrard dimensionless units.
rho_u = Mcl / Rcl**3
t_u = np.sqrt(Rcl**3 / (G * Mcl))
v_u = Rcl / t_u
P_u = rho_u * v_u**2

t_adm = t / t_u
r_edges_u = r_edges / Rcl
z_edges_u = z_edges / Rcl
rCoM_u = rCoM / Rcl
zCoM_u = zCoM / Rcl

rho2D = np.asarray(rho_hist_cgs[-1].T, dtype=np.float64) / rho_u
ur2D = np.asarray(ur_last_cgs, dtype=np.float64) / v_u
uz2D = np.asarray(uz_last_cgs, dtype=np.float64) / v_u
P2D = np.asarray(P_last_cgs, dtype=np.float64) / P_u

print(f"Grid: {Nr} x {Nz}; snapshots: {Nt}")
print(f"Final saved time t/t_u = {t_adm[-1]:.6f}")

# ---------------------------------------------------------------
# SPHERICAL AVERAGING IN AXISYMMETRIC CYLINDRICAL COORDINATES
# ---------------------------------------------------------------
# Reflecting lower Z boundary -> cloud is centered at Z = 0.
# If using the full Z domain, set Zc_u = 0.5*z_edges_u[-1].
Zc_u = 0.0

gamma = 5.0 / 3.0
r_mesh, z_mesh = np.meshgrid(rCoM_u, zCoM_u - Zc_u, indexing="ij")
R = np.hypot(r_mesh, z_mesh)

# Recompute cell volumes from the interface coordinates in dimensionless
# units. The stored Float32 CGS volumes can overflow.
dV = (
    np.pi
    * (r_edges_u[1:]**2 - r_edges_u[:-1]**2)[:, None]
    * np.diff(z_edges_u)[None, :]
)
assert dV.shape == (Nr, Nz)
assert np.all(np.isfinite(dV)) and np.all(dV > 0)

# One common set of spherical bins for density, velocity, entropy.
R_edges = r_edges_u.copy()
R_centers = 0.5 * (R_edges[1:] + R_edges[:-1])
Nbins = len(R_centers)

bin_index = np.searchsorted(R_edges, R.ravel(), side="right") - 1
valid = (bin_index >= 0) & (bin_index < Nbins)
idx = bin_index[valid]
weights = dV.ravel()[valid]
V_bins = np.bincount(idx, weights=weights, minlength=Nbins)


def spherical_average(field):
    """Volume-weighted shell average of a (Nr, Nz) field."""
    assert field.shape == (Nr, Nz)
    sums = np.bincount(
        idx,
        weights=field.ravel()[valid] * weights,
        minlength=Nbins,
    )
    result = np.full(Nbins, np.nan, dtype=np.float64)
    np.divide(sums, V_bins, out=result, where=V_bins > 0)
    return result


# Radial velocity and entropy evaluated CELLWISE before averaging.
vR2D = np.divide(
    r_mesh * ur2D + z_mesh * uz2D,
    R,
    out=np.zeros_like(R),
    where=R > 0,
)
entropy2D = P2D / np.maximum(rho2D, np.finfo(float).tiny)**gamma

rho_profile = spherical_average(rho2D)
vR_profile = spherical_average(vR2D)
entropy_profile = spherical_average(entropy2D)

# ---------------------------------------------------------------
# LOAD HIGH-RESOLUTION HYDROCODE1D REFERENCE
# ---------------------------------------------------------------
if not reference_file.is_file():
    raise FileNotFoundError(f"Missing HydroCode1D reference: {reference_file}")

# Columns: radius, density, radial velocity, pressure, acceleration.
reference_data = np.loadtxt(reference_file, comments="#", ndmin=2)
if reference_data.shape != (16000, 5):
    raise ValueError(
        f"Expected HydroCode1D shape (16000, 5), got {reference_data.shape}"
    )
if not np.all(np.isfinite(reference_data)):
    raise ValueError("Non-finite HydroCode1D reference values")

reference_r = reference_data[:, 0]
reference_rho = reference_data[:, 1]
reference_vR = reference_data[:, 2]
reference_P = reference_data[:, 3]
reference_entropy = reference_P / reference_rho**gamma
reference_profiles = (reference_rho, reference_vR, reference_entropy)

print(f"HydroCode1D reference: {len(reference_r)} radial cells")

# ---------------------------------------------------------------
# THREE-PANEL PROFILE: LYDION AND HYDROCODE1D AT t/t_u = 0.8
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16, 6.3))

profiles = (rho_profile, vR_profile, entropy_profile)
ylabels = (
    r"$\rho/(M_{cl}/R_{cl}^3)$",
    r"$v_r/(GM_{cl}/R_{cl})^{1/2}$",
    r"$P/\rho^\gamma$ (dimensionless)",
)
ylimits = ((1e-2, 1e3), (-1.7, 0.1), (0.0, 0.2))

for i, (a, profile, ylabel, ylim) in enumerate(zip(ax, profiles, ylabels, ylimits)):
    a.plot(
        R_centers, profile,
        color=lydion_color,
        linewidth=lydion_linewidth,
        linestyle=lydion_linestyle,
        label=lydion_label,
    )
    a.plot(
        reference_r, reference_profiles[i],
        color=reference_color,
        linewidth=reference_linewidth,
        linestyle=reference_linestyle,
        dash_capstyle="round",
        alpha=0.8,
        label=reference_label if reference_label is not None else "_nolegend_",
    )

    a.set_ylabel(ylabel, labelpad=10)
    a.set_xlabel(r"$r/R_{cl}$", labelpad=9)
    a.set_xscale("log")
    a.set_xlim(*profile_xlim)
    a.set_ylim(*ylim)
    a.tick_params(labelsize=19)

    a.xaxis.set_major_locator(LogLocator(base=10.0))
    a.xaxis.set_major_formatter(log_tick_formatter)

    if i == 0:
        a.set_yscale("log")
        a.yaxis.set_major_locator(LogLocator(base=10.0))
        a.yaxis.set_major_formatter(log_tick_formatter)

    if i == 0:
        a.legend(
        frameon=False, fontsize=18, loc="lower left",
        bbox_to_anchor=(0.05, 0.05), borderaxespad=0.0, )

    if i > 0:
        a.yaxis.set_minor_locator(AutoMinorLocator(5))

fig.suptitle(
    fr"Evrard collapse test, $t = {t_adm[-1]:.3f}\,(R_{{cl}}^3/GM_{{cl}})^{{1/2}}$",
    fontsize=figure_title_fontsize,
    y=figure_title_y,
)
fig.tight_layout(rect=(0, 0, 1, 0.935), w_pad=2.0)
fig.savefig(profile_outname, dpi=plot_dpi,
            bbox_inches="tight", pad_inches=0.05)
print(f"Saved profile figure: {profile_outname}")
plt.show()

# ---------------------------------------------------------------
# OPTIONAL DENSITY ANIMATION (PHYSICAL CGS DENSITIES)
# ---------------------------------------------------------------
if make_animation:
    frames = np.arange(0, Nt, frameskip)
    rho_positive = rho_hist_cgs[np.isfinite(rho_hist_cgs) & (rho_hist_cgs > 0)]
    if rho_positive.size == 0:
        raise ValueError("No positive finite densities for logarithmic animation")

    norm = LogNorm(vmin=float(rho_positive.min()), vmax=float(rho_positive.max()))

    fig2, ax2 = plt.subplots(figsize=(9, 8), constrained_layout=True)
    mesh = ax2.pcolormesh(
        r_edges_u, z_edges_u, rho_hist_cgs[0],
        shading="auto", norm=norm, cmap="viridis",
    )
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel(r"$r/R_{cl}$")
    ax2.set_ylabel(r"$Z/R_{cl}$")
    cb = fig2.colorbar(mesh, ax=ax2, label=r"$\rho\ [{\rm g\,cm^{-3}}]$")
    cb.ax.tick_params(labelsize=16)

    time_text = ax2.text(
        0.05, 0.93, "", transform=ax2.transAxes, color="white",
        fontsize=20,
        path_effects=[pe.withStroke(linewidth=3, foreground="black")],
    )

    def update(k):
        mesh.set_array(rho_hist_cgs[k].ravel())
        time_text.set_text(f"t/t_u = {t_adm[k]:.3f}")
        return mesh, time_text

    ani = FuncAnimation(
        fig2, update, frames=frames,
        init_func=lambda: update(frames[0]), blit=True,
    )
    ani.save(animation_outname, writer="ffmpeg", fps=fps, dpi=animation_dpi)
    print(f"Saved animation: {animation_outname}")
    plt.close(fig2)
