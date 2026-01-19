import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator  # if you want it later

# ---------------- user inputs ----------------
filename  = "StaticCloudTest.h5"        # <-- static cloud file
frameskip = 1                  # use >1 to skip frames
save_mp4  = True               # True to save mp4 instead of interactive show
outname   = "StaticCloud_eLya_withLyaSpec.mp4"
fps       = 20

# which Lyα edge spectra to plot (indices along J_Lya_EdgeSpectra axis 0)
edge_indices = [0]            # e.g. [0, 5, 10]

# Left panel mode:
left_panel_mode = "map"       # "map" or "axis"

# ν-window settings for the spectrum panel
use_nu_window       = True
nu_Lya              = 2.466e15             # Hz
nu_window_fraction  = 0.003                # total width = 2% of ν_Lya (±1% around center)
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
    edge_ax = 3 - t_ax - nu_ax

    perm = [t_ax, edge_ax, nu_ax]  # -> (time, edge, freq)
    J_reordered = np.transpose(J_edge_raw, axes=perm)

    NtL, NedgeL, NνL = J_reordered.shape
    assert NtL == Nt, f"Time length mismatch: {NtL} vs Nt={Nt}"
    assert NνL == Nν, f"Freq length mismatch: {NνL} vs Nν={Nν}"

    return J_reordered  # (Nt, N_edge, N_ν)

# -------------------------------------------------------------------
# read data (STATIC CLOUD)
# -------------------------------------------------------------------
with h5py.File(filename, "r") as f:
    t         = f["Times"][:]            # (Nsaved,)
    r         = f["r"][:]                # edges, (Nr+1,)
    Z         = f["Z"][:]                # edges, (Nz+1,)
    rCoM      = f["rCoM"][:]             # (Nr,)
    ZCoM      = f["ZCoM"][:]             # (Nz,)

    eLya_raw  = f["e_Lya_History"][:]    # 3D history
    eLya_ana  = f["e_Lya_Analytical"][:] # (Nz,) analytical profile vs ZCoM

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

# reorder e_Lya to (Nt, Nr, Nz)
e_Lya_hist = reorder_history_3d(eLya_raw, t, r, Z)

# Lyα spectra -> (Nt, N_edge, N_ν)
J_edge_hist = reorder_lya_edge_spectra(J_edge_raw, t, nuC)
Nt_Lya, N_edge, N_nu = J_edge_hist.shape
assert Nt_Lya == Nt, f"Lyα history time dimension {Nt_Lya} != Times length {Nt}"

print("Max e_Lya:", np.max(e_Lya_hist))
print("Max e_Lya_Analytical:", np.max(eLya_ana))
print("t range in file:", t[0]/kyr, "→", t[saved_idx[-1]]/kyr, "kyr")

# grid for pcolormesh (edges, in pc)
R, ZZ = np.meshgrid(r/pc, Z/pc, indexing='ij')

# -------------------------------------------------------------------
# PLOT COSMETICS (unchanged)
# -------------------------------------------------------------------
plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 37})
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

params = {'legend.fontsize': 19}
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
# e_Lya display range (tweak if needed)
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

# spectrum y-limits using masked ν-range
Jmax = np.max(J_edge_hist[:, :, nu_mask])
if Jmax <= 0:
    Jmax = 1e-20
Jmin = 0.0

# -------------------------------------------------------------------
# figure + axes (2 panels)
# -------------------------------------------------------------------
fig = plt.figure(figsize=(22, 10), constrained_layout=True)
gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.2])

ax_left = fig.add_subplot(gs[0, 0])  # left panel: map or axis slice
ax_spec = fig.add_subplot(gs[0, 1])  # spectrum

panel_label_style = dict(
    ha='right',
    va='top',
    color='black',
    path_effects=[pe.withStroke(linewidth=3, foreground='white')]
)

# --- LEFT PANEL: either 2D map or Z-axis slice at r~0 ---
if left_panel_mode == "map":
    ax_left.set_xlabel("r (pc)", labelpad=12)
    ax_left.set_ylabel("Z (pc)", labelpad=12)

    im_Lya = ax_left.pcolormesh(
        R, ZZ,
        np.clip(e_Lya_hist[0].T, vmin_Lya, vmax_Lya),
        norm=norm_Lya, shading='auto', cmap='gnuplot2'
    )
    cb = fig.colorbar(im_Lya, ax=ax_left, pad=0.02)
    ax_left.text(0.9, 0.9, "e$_{Ly\\alpha}$", transform=ax_left.transAxes, **panel_label_style)

    ax_left.set_xlim(r[0]/pc, r[-1]/pc)
    ax_left.set_ylim(Z[0]/pc, Z[-1]/pc)

elif left_panel_mode == "axis":
    # choose the cell-center closest to the axis (r ~ 0)
    ir0 = int(np.argmin(np.abs(rCoM)))
    Z_pc = ZCoM / pc

    ax_left.set_xlabel("Z (pc)", labelpad=12)
    ax_left.set_ylabel("e$_{Ly\\alpha}$", labelpad=12)
    ax_left.set_yscale("log")
    ax_left.set_xlim(Z_pc.min(), Z_pc.max())
    ax_left.set_ylim(vmin_Lya, vmax_Lya)

    # numerical (animated)
    (line_Lya,) = ax_left.plot(
        Z_pc,
        np.clip(e_Lya_hist[0, ir0, :], vmin_Lya, vmax_Lya),
        linewidth=3,
        label="Numerical"
    )
    # analytical (static)
    (line_Lya_ana,) = ax_left.plot(
        Z_pc,
        np.clip(eLya_ana, vmin_Lya, vmax_Lya),
        linewidth=3,
        linestyle="--",
        color="black",
        label="Analytical"
    )

    ax_left.legend(loc="upper right")
    ax_left.text(0.9, 0.9, f"r ≈ {rCoM[ir0]/pc:.3f} pc", transform=ax_left.transAxes,
                 **panel_label_style)

else:
    raise ValueError(f"Unknown left_panel_mode={left_panel_mode!r}. Use 'map' or 'axis'.")

# --- spectrum panel ---
ax_spec.set_ylim(0.0, 2*1.2e-13 )  # adjust if needed
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
    label = fr"edge {idx} (r={r_pc:.2f} pc, Z={Z_pc:.2f} pc)"
    (line,) = ax_spec.plot(
        nuC_plot, J_edge_hist[k0, idx, nu_mask],
        label=label, linewidth=3, color='crimson'
    )
    spec_lines.append(line)

ax_spec.legend(loc="upper right")

# --- time label on left panel ---
time_style = dict(
    ha='left', va='top', color='white',
    path_effects=[pe.withStroke(linewidth=3, foreground='black')]
)
txt = ax_left.text(0.1, 0.9, "", fontsize=28, transform=ax_left.transAxes, **time_style)

# frames to animate
frames_all = saved_idx
frames = frames_all[::frameskip]

# -------------------------------------------------------------------
# animation functions
# -------------------------------------------------------------------
def init():
    k0 = frames[0]

    if left_panel_mode == "map":
        im_Lya.set_array(np.clip(e_Lya_hist[k0].T, vmin_Lya, vmax_Lya).ravel())
        left_artists = (im_Lya,)
    else:
        line_Lya.set_ydata(np.clip(e_Lya_hist[k0, ir0, :], vmin_Lya, vmax_Lya))
        # analytical line is static (no update needed)
        left_artists = (line_Lya, line_Lya_ana)

    for line, idx in zip(spec_lines, edge_indices):
        line.set_ydata(J_edge_hist[k0, idx, nu_mask])

    txt.set_text(f"t = {t[k0]/kyr:.2f} kyr")
    return (*left_artists, txt, *spec_lines)

def update(k):
    if left_panel_mode == "map":
        im_Lya.set_array(np.clip(e_Lya_hist[k].T, vmin_Lya, vmax_Lya).ravel())
        left_artists = (im_Lya,)
    else:
        line_Lya.set_ydata(np.clip(e_Lya_hist[k, ir0, :], vmin_Lya, vmax_Lya))
        left_artists = (line_Lya, line_Lya_ana)

    for line, idx in zip(spec_lines, edge_indices):
        line.set_ydata(J_edge_hist[k, idx, nu_mask])

    txt.set_text(f"t = {t[k]/kyr:.2f} kyr")
    return (*left_artists, txt, *spec_lines)

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

if save_mp4:
    ani.save(outname, writer="ffmpeg", fps=fps, dpi=300)
else:
    plt.show()
