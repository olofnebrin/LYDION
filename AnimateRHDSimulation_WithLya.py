import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator  # if you want it later
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
cmastro_src = repo_root / "src" / "cmastro" / "src"

sys.path.insert(0, str(cmastro_src))

import cmastro

# ---------------- user inputs ----------------
filename  = "RHDSim_StarCluster_Lya_128x128_2026-04-16.h5" 
frameskip = 1                  # use >1 to skip frames
save_mp4  = True               # True to save mp4 instead of interactive show
outname   = "Animation_LyaRHD_withspectra.mp4"
fps       = 20

# which Lyα edge spectra to plot (indices along J_Lya_EdgeSpectra axis 0)
edge_indices = [50]            # e.g. [0, 5, 10]

# ν-window settings for the spectrum panel
use_nu_window       = True
nu_Lya              = 2.466e15             # Hz
nu_window_fraction  = 0.01                # total width, e.g. 0.02 = 2% of ν_Lya 
# ---------------------------------------------

# constants
kB  = 1.38e-16
mH  = 1.67e-24
X   = 0.76
yr  = 3.1556926e7
kyr = 1e3 * yr
pc  = 3.085677581e18

# -------------------------------------------------------------------
# helpers to robustly reorder datasets
# -------------------------------------------------------------------
def reorder_history_3d(arr, t, r_edges, Z_edges):
    """
    Reorder a 3D history array to shape (Nt, Nr, Nz).

    One axis must match len(t) (time).
    The remaining two must match (Nr, Nz) in some order.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    Nt = t.size
    Nr = len(r_edges) - 1
    Nz = len(Z_edges) - 1

    shape = arr.shape
    # find time axis
    time_axes = [ax for ax, s in enumerate(shape) if s == Nt]
    if len(time_axes) != 1:
        raise RuntimeError(
            f"Could not uniquely identify time axis in array with shape {shape} and Nt={Nt}"
        )
    time_axis = time_axes[0]

    # move time to front: (Nt, A, B)
    arr_tfirst = np.moveaxis(arr, time_axis, 0)
    _, a1, a2 = arr_tfirst.shape

    # now match spatial axes to (Nr, Nz)
    if (a1, a2) == (Nr, Nz):
        return arr_tfirst
    elif (a1, a2) == (Nz, Nr):
        return arr_tfirst[:, :, :].transpose(0, 2, 1)  # (Nt, Nr, Nz)
    else:
        raise RuntimeError(
            f"After moving time axis, spatial dims {a1,a2} do not match Nr={Nr}, Nz={Nz}"
        )

def reorder_lya_edge_spectra(J_edge_raw, t, nuC):
    """
    Reorder J_Lya_EdgeSpectra to shape (Nt, N_edge, N_nu).

    One axis must match len(t) (time).
    One axis must match len(nuC) (frequency).
    The remaining axis is N_edge.
    """
    if J_edge_raw.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {J_edge_raw.shape}")

    Nt  = t.size
    Nν  = nuC.size
    s0, s1, s2 = J_edge_raw.shape

    # identify time and freq axes
    axes = [s0, s1, s2]
    time_axes = [ax for ax, s in enumerate(axes) if s == Nt]
    nu_axes   = [ax for ax, s in enumerate(axes) if s == Nν]

    if len(time_axes) != 1:
        raise RuntimeError(
            f"Could not uniquely identify time axis in J_edge_raw: shape={J_edge_raw.shape}, Nt={Nt}"
        )
    if len(nu_axes) != 1:
        raise RuntimeError(
            f"Could not uniquely identify frequency axis in J_edge_raw: shape={J_edge_raw.shape}, Nν={Nν}"
        )

    t_ax  = time_axes[0]
    nu_ax = nu_axes[0]

    # edge axis is the remaining one
    edge_ax = 3 - t_ax - nu_ax  # since axes are 0,1,2 and sum is 0+1+2 = 3

    # reorder to (time, edge, freq)
    # build a permutation that puts (t_ax, edge_ax, nu_ax) -> (0,1,2)
    perm = [t_ax, edge_ax, nu_ax]
    J_reordered = np.transpose(J_edge_raw, axes=perm)

    NtL, NedgeL, NνL = J_reordered.shape
    assert NtL == Nt, f"Time length mismatch: {NtL} vs Nt={Nt}"
    assert NνL == Nν, f"Freq length mismatch: {NνL} vs Nν={Nν}"

    return J_reordered  # (Nt, N_edge, N_ν)

# -------------------------------------------------------------------
# read data
# -------------------------------------------------------------------
with h5py.File(filename, "r") as f:
    t         = f["Times"][:]            # (Nsaved,)
    r         = f["r"][:]                # edges, (Nr+1,)
    Z         = f["Z"][:]                # edges, (Nz+1,)
    rCoM      = f["rCoM"][:]             # (Nr,)
    ZCoM      = f["ZCoM"][:]             # (Nz,)

    rho_raw   = f["GasDensity_History"][:]
    T_raw     = f["T_History"][:]
    xH2_raw   = f["xH2_History"][:]
    xHII_raw  = f["xHII_History"][:]
    eLya_raw  = f["e_Lya_History"][:]

    # Lyα spectral data
    J_edge_raw = f["J_Lya_EdgeSpectra"][:]
    nuC        = f["νC"][:]
    edge_ridx  = f["J_Lya_Edge_rIndex"][:]  # 1-based (Julia)
    edge_zidx  = f["J_Lya_Edge_ZIndex"][:]

# basic sizes
Nt = t.size
Nr = len(r) - 1
Nz = len(Z) - 1

# which time indices actually have data
saved_idx = np.where((t > 0) | (np.arange(t.size) == 0))[0]

# reorder all hydro / thermo fields to (Nt, Nr, Nz)
rho_hist   = reorder_history_3d(rho_raw,  t, r, Z)
T_hist     = reorder_history_3d(T_raw,    t, r, Z)
xH2_hist   = reorder_history_3d(xH2_raw,  t, r, Z)
xHII_hist  = reorder_history_3d(xHII_raw, t, r, Z)
e_Lya_hist = reorder_history_3d(eLya_raw, t, r, Z)

# Lyα spectra -> (Nt, N_edge, N_ν)
J_edge_hist = reorder_lya_edge_spectra(J_edge_raw, t, nuC)
Nt_Lya, N_edge, N_nu = J_edge_hist.shape

# sanity check: times & Lyα time axis should match
assert Nt_Lya == Nt, f"Lyα history time dimension {Nt_Lya} != Times length {Nt}"

# hydrogen stuff
xHI_hist = 1.0 - 2.0 * xH2_hist - xHII_hist
nH_hist  = X * rho_hist / mH   # (Nt, Nr, Nz)
P_hist   = rho_hist * kB * T_hist / mH 

print("Max e_Lya:", np.max(e_Lya_hist))
print("t range in file:", t[0]/kyr, "→", t[saved_idx[-1]]/kyr, "kyr")

# grid for pcolormesh (edges, in pc)
R, ZZ = np.meshgrid(r/pc, Z/pc, indexing='ij')

# -------------------------------------------------------------------
# PLOT COSMETICS
# -------------------------------------------------------------------
plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 37})
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

params = {'legend.fontsize': 24}
plt.rcParams.update(params)
plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 1.7
plt.rcParams['xtick.minor.width'] = 1.7
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['ytick.minor.size'] = 8
plt.rcParams['xtick.major.pad'] = '15'
plt.rcParams['ytick.major.pad'] = '15'

plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linewidth'] = 2.5
plt.rcParams['grid.linestyle'] = '--'

plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# -------------------------------------------------------------------
# display ranges
# -------------------------------------------------------------------
vmin_nH, vmax_nH = 1e3, 1e7
norm_nH = LogNorm(vmin=vmin_nH, vmax=vmax_nH)

vmin_T, vmax_T = 10.0, 1e5
norm_T = LogNorm(vmin=vmin_T, vmax=vmax_T)

vmin_x, vmax_x = 1e-8, 1.0
norm_x = LogNorm(vmin=vmin_x, vmax=vmax_x)

vmin_Lya, vmax_Lya = 1e-12, 1e-6
norm_Lya = LogNorm(vmin=vmin_Lya, vmax=vmax_Lya)

# --- set ν-window (mask) for plotting spectrum ---
if use_nu_window:
    nu_center = nu_Lya
    nu_half_window = 0.5 * nu_window_fraction * nu_center
    nu_min_plot = nu_center - nu_half_window
    nu_max_plot = nu_center + nu_half_window
    nu_mask = (nuC >= nu_min_plot) & (nuC <= nu_max_plot)
else:
    nu_mask = np.ones_like(nuC, dtype=bool)

nuC_plot = nuC[nu_mask]

# --- spectrum y-limits using masked ν-range ---
Jmax = np.max(J_edge_hist[:, :, nu_mask])
if Jmax <= 0:
    Jmax = 1e-20  # small floor to avoid identical y-lims
Jmin = 0.0

# -------------------------------------------------------------------
# figure + axes
# -------------------------------------------------------------------
fig = plt.figure(figsize=(22, 12), constrained_layout=True)
gs  = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.4], height_ratios=[1, 1])

ax1     = fig.add_subplot(gs[0, 0])                         # density
ax2     = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1) # temperature
ax3     = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1) # e_Lya
ax4     = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax1) # xHII
ax_spec = fig.add_subplot(gs[:, 2])                         # Lyα spectrum

ax1.tick_params(labelbottom=False)
ax3.tick_params(labelbottom=False)

panel_label_style = dict(
    ha='right',
    va='top',
    color='black',
    path_effects=[pe.withStroke(linewidth=3, foreground='white')]
)

# --- density ---
ax1.set_ylabel("Z (pc)", labelpad=12)
im_nH = ax1.pcolormesh(R, ZZ, nH_hist[0].T, norm=norm_nH, shading='auto', cmap='cma:laguna')
cb1 = fig.colorbar(im_nH, ax=ax1, pad=0.02)
cb1.set_ticks([1e3, 1e5, 1e7])
cb1.set_ticklabels(['10$^{3}$', '10$^{5}$', '10$^{7}$'])
ax1.text(0.9, 0.2, "n$_{H}$ (cm$^{-3}$)", transform=ax1.transAxes, **panel_label_style)

ax1.set_xlim( r[0]/pc, r[-1]/pc )
ax1.set_ylim( Z[0]/pc, Z[-1]/pc )

# --- temperature ---
ax2.set_xlabel("r (pc)", labelpad=12)
ax2.set_ylabel("Z (pc)", labelpad=12)
im_T = ax2.pcolormesh(R, ZZ, T_hist[0].T, norm=norm_T, shading='auto', cmap='coolwarm')
cb2 = fig.colorbar(im_T, ax=ax2, pad=0.02)
cb2.set_ticks([10, 100, 1e3, 1e4, 1e5])
cb2.set_ticklabels(['10', '10$^{2}$', '10$^{3}$', '10$^{4}$', '10$^{5}$'])
ax2.text(0.9, 0.9, "T (K)", transform=ax2.transAxes, **panel_label_style)

# --- e_Lya ---
ax3.tick_params(labelleft=False)
im_Lya = ax3.pcolormesh(
    R, ZZ,
    np.clip(e_Lya_hist[0].T, vmin_Lya, vmax_Lya),
    norm=norm_Lya, shading='auto', cmap='cma:emph'
)
cb3 = fig.colorbar(im_Lya, ax=ax3, pad=0.02)
#cb3.set_ticks([1e-12, 1e-10, 1e-8, 1e-6])
#cb3.set_ticklabels(['10$^{-12}$', '10$^{-10}$', '10$^{-8}$', '10$^{-6}$'])
ax3.text(0.9, 0.9, "e$_{Lya}$", transform=ax3.transAxes, **panel_label_style)

# --- xHI ---
ax4.set_xlabel("r (pc)", labelpad=12)
ax4.tick_params(labelleft=False)
im_xHII = ax4.pcolormesh(
    R, ZZ,
    np.clip(xHI_hist[0].T, vmin_x, vmax_x),
    norm=norm_x, shading='auto', cmap='magma'
)
cb4 = fig.colorbar(im_xHII, ax=ax4, pad=0.02)
cb4.set_ticks([1e-8, 1e-6, 1e-4, 0.01, 1])
cb4.set_ticklabels(['10$^{-8}$', '10$^{-6}$', '10$^{-4}$', '0.01', '1'])
ax4.text(0.9, 0.9, "x$_{HI}$", transform=ax4.transAxes, **panel_label_style)

# --- spectrum panel ---
ax_spec.set_ylim(0.0, 1e-15)
ax_spec.set_xlabel(r"$\nu$ (Hz)", labelpad=12)
ax_spec.set_ylabel(r"$J_{Ly\alpha}$", labelpad=12)
ax_spec.set_xlim(nuC_plot.min(), nuC_plot.max())

spec_lines = []
k0 = saved_idx[0]
for idx in edge_indices:
    ir = edge_ridx[idx] - 1   # Julia 1-based -> Python 0-based
    iz = edge_zidx[idx] - 1
    r_pc = rCoM[ir] / pc
    Z_pc = ZCoM[iz] / pc
    label = fr"(r,Z) = ({r_pc:.2f}, {Z_pc:.2f}) pc"
    (line,) = ax_spec.plot(nuC_plot, J_edge_hist[k0, idx, nu_mask], label=label, 
                           linewidth = 3, color = 'crimson')
    spec_lines.append(line)

ax_spec.legend(loc="upper right")

# --- time label ---
time_style = dict(
    ha='left', va='top', color='white',
    path_effects=[pe.withStroke(linewidth=3, foreground='black')]
)
txt = ax1.text(0.1, 0.9, "", fontsize=28, transform=ax1.transAxes, **time_style)

# --- frames to animate: all saved indices, no clipping by Nt_Lya (they match now) ---
frames_all = saved_idx
frames = frames_all[::frameskip]

# -------------------------------------------------------------------
# animation functions
# -------------------------------------------------------------------
def init():
    k0 = frames[0]
    im_nH.set_array(nH_hist[k0].T.ravel())
    im_T.set_array(T_hist[k0].T.ravel())
    im_Lya.set_array(np.clip(e_Lya_hist[k0].T, vmin_Lya, vmax_Lya).ravel())
    im_xHII.set_array(np.clip(xHI_hist[k0].T, vmin_x, vmax_x).ravel())
    for line, idx in zip(spec_lines, edge_indices):
        line.set_ydata(J_edge_hist[k0, idx, nu_mask])
    txt.set_text(f"t = {t[k0]/kyr:.2f} kyr")
    return (im_nH, im_T, im_Lya, im_xHII, txt, *spec_lines)

def update(k):
    im_nH.set_array(nH_hist[k].T.ravel())
    im_T.set_array(T_hist[k].T.ravel())
    im_Lya.set_array(np.clip(e_Lya_hist[k].T, vmin_Lya, vmax_Lya).ravel())
    im_xHII.set_array(np.clip(xHI_hist[k].T, vmin_x, vmax_x).ravel())
    for line, idx in zip(spec_lines, edge_indices):
        line.set_ydata(J_edge_hist[k, idx, nu_mask])
    txt.set_text(f"t = {t[k]/kyr:.2f} kyr")
    return (im_nH, im_T, im_Lya, im_xHII, txt, *spec_lines)

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

if save_mp4:
    ani.save(outname, writer="ffmpeg", fps=fps, dpi=300)
else:
    plt.show()
