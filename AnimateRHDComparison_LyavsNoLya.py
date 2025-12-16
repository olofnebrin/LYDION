import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator  # kept (unused, but you had it)

# ---------------- user inputs ----------------
filename_noLya = "RHDSim_128x128_noLya.h5"     # Run ignoring Lyα feedback
filename_Lya   = "RHDSim_128x128_withLya.h5"   # Run with Lyα feedback

frameskip = 1                  # use >1 to skip frames
save_mp4  = True               # True to save mp4 instead of interactive show
outname   = "compare_noLya_vs_Lya_density_T.mp4"
fps       = 20
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

def read_run(filename):
    """
    Read one run and return dict with:
      t, r, Z, rho_hist, T_hist, nH_hist, saved_idx
    """
    with h5py.File(filename, "r") as f:
        t    = f["Times"][:]            # (Nsaved,)
        r    = f["r"][:]                # edges, (Nr+1,)
        Z    = f["Z"][:]                # edges, (Nz+1,)

        rho_raw = f["GasDensity_History"][:]
        T_raw   = f["T_History"][:]

    # reorder to (Nt, Nr, Nz)
    rho_hist = reorder_history_3d(rho_raw, t, r, Z)
    T_hist   = reorder_history_3d(T_raw,   t, r, Z)

    # which time indices actually have data
    saved_idx = np.where((t > 0) | (np.arange(t.size) == 0))[0]

    # derived fields
    nH_hist = X * rho_hist / mH   # (Nt, Nr, Nz)

    return dict(
        t=t, r=r, Z=Z,
        rho_hist=rho_hist,
        T_hist=T_hist,
        nH_hist=nH_hist,
        saved_idx=saved_idx
    )

# -------------------------------------------------------------------
# read both runs
# -------------------------------------------------------------------
runL = read_run(filename_noLya)
runR = read_run(filename_Lya)

# basic consistency checks for grids
if (runL["r"].shape != runR["r"].shape) or (runL["Z"].shape != runR["Z"].shape):
    raise RuntimeError(
        f"Grid mismatch: r shapes {runL['r'].shape} vs {runR['r'].shape}, "
        f"Z shapes {runL['Z'].shape} vs {runR['Z'].shape}"
    )
if not (np.allclose(runL["r"], runR["r"]) and np.allclose(runL["Z"], runR["Z"])):
    raise RuntimeError("Grid mismatch: r/Z edge arrays differ between the two files.")

# time arrays may differ slightly; we just animate over the minimum available saved frames
tL, tR = runL["t"], runR["t"]
savedL, savedR = runL["saved_idx"], runR["saved_idx"]

Nframes = min(savedL.size, savedR.size)
frames_pair = np.arange(Nframes)          # index into savedL/savedR
frames = frames_pair[::frameskip]

print("No-Lyα: t range:", tL[0]/kyr, "→", tL[savedL[min(savedL.size-1, Nframes-1)]]/kyr, "kyr")
print("With-Lyα: t range:", tR[0]/kyr, "→", tR[savedR[min(savedR.size-1, Nframes-1)]]/kyr, "kyr")
print("Animating frames:", len(frames), "of", Nframes)

# grid for pcolormesh (edges, in pc)
r = runL["r"]
Z = runL["Z"]
R, ZZ = np.meshgrid(r/pc, Z/pc, indexing='ij')

# -------------------------------------------------------------------
# PLOT COSMETICS  (kept identical)
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
# display ranges (kept same as your density/temperature panels)
# -------------------------------------------------------------------
vmin_nH, vmax_nH = 10, 1e5
norm_nH = LogNorm(vmin=vmin_nH, vmax=vmax_nH)

vmin_T, vmax_T = 10.0, 1e5
norm_T = LogNorm(vmin=vmin_T, vmax=vmax_T)

# -------------------------------------------------------------------
# figure + axes: 2 rows x 2 cols
# left: no Lyα, right: with Lyα
# top: density, bottom: temperature
# -------------------------------------------------------------------
fig = plt.figure(figsize=(14, 12), constrained_layout=True)
gs  = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])

ax11 = fig.add_subplot(gs[0, 0])                         # noLyα density
ax12 = fig.add_subplot(gs[1, 0], sharex=ax11, sharey=ax11) # noLyα temp
ax21 = fig.add_subplot(gs[0, 1], sharex=ax11, sharey=ax11) # Lyα density
ax22 = fig.add_subplot(gs[1, 1], sharex=ax11, sharey=ax11) # Lyα temp

ax11.tick_params(labelbottom=False)
ax21.tick_params(labelbottom=False)

panel_label_style = dict(
    ha='right',
    va='top',
    color='black',
    path_effects=[pe.withStroke(linewidth=3, foreground='white')]
)

# Titles (kept lightweight; feel free to remove if you prefer zero extras)
ax11.set_title("No Lyα feedback", pad = 20)
ax21.set_title("With Lyα feedback", pad = 20)

# --- density (no Lyα) ---
ax11.set_ylabel("Z (pc)", labelpad=12)
im_nH_L = ax11.pcolormesh(R, ZZ, runL["nH_hist"][0].T, norm=norm_nH, shading='auto', cmap='viridis')
cb1 = fig.colorbar(im_nH_L, ax=ax11, pad=0.02)
cb1.set_ticks([10, 1e3, 1e5])
cb1.set_ticklabels(['10', '10$^{3}$', '10$^{5}$'])
ax11.text(0.9, 0.2, "n$_{H}$ (cm$^{-3}$)", transform=ax11.transAxes, **panel_label_style)

# --- temperature (no Lyα) ---
ax12.set_xlabel("r (pc)", labelpad=12)
ax12.set_ylabel("Z (pc)", labelpad=12)
im_T_L = ax12.pcolormesh(R, ZZ, runL["T_hist"][0].T, norm=norm_T, shading='auto', cmap='coolwarm')
cb2 = fig.colorbar(im_T_L, ax=ax12, pad=0.02)
cb2.set_ticks([10, 100, 1e3, 1e4, 1e5])
cb2.set_ticklabels(['10', '10$^{2}$', '10$^{3}$', '10$^{4}$', '10$^{5}$'])
ax12.text(0.9, 0.9, "T (K)", transform=ax12.transAxes, **panel_label_style)

# --- density (with Lyα) ---
ax21.tick_params(labelleft=False)
im_nH_R = ax21.pcolormesh(R, ZZ, runR["nH_hist"][0].T, norm=norm_nH, shading='auto', cmap='viridis')
cb3 = fig.colorbar(im_nH_R, ax=ax21, pad=0.02)
cb3.set_ticks([10, 1e3, 1e5])
cb3.set_ticklabels(['10', '10$^{3}$', '10$^{5}$'])
ax21.text(0.9, 0.2, "n$_{H}$ (cm$^{-3}$)", transform=ax21.transAxes, **panel_label_style)

# --- temperature (with Lyα) ---
ax22.set_xlabel("r (pc)", labelpad=12)
ax22.tick_params(labelleft=False)
im_T_R = ax22.pcolormesh(R, ZZ, runR["T_hist"][0].T, norm=norm_T, shading='auto', cmap='coolwarm')
cb4 = fig.colorbar(im_T_R, ax=ax22, pad=0.02)
cb4.set_ticks([10, 100, 1e3, 1e4, 1e5])
cb4.set_ticklabels(['10', '10$^{2}$', '10$^{3}$', '10$^{4}$', '10$^{5}$'])
ax22.text(0.9, 0.9, "T (K)", transform=ax22.transAxes, **panel_label_style)

# shared limits
for ax in (ax11, ax12, ax21, ax22):
    ax.set_xlim(r[0]/pc, r[-1]/pc)
    ax.set_ylim(Z[0]/pc, Z[-1]/pc)
    ax.set_aspect('equal', adjustable='box') # Ensure correct aspect ratio of panels 

# --- time label (put in top-left like your original; same style) ---
time_style = dict(
    ha='left', va='top', color='white',
    path_effects=[pe.withStroke(linewidth=3, foreground='black')]
)

txtL = ax11.text(
    0.05, 0.95, "", fontsize=26,
    transform=ax11.transAxes, **time_style
)

txtR = ax21.text(
    0.05, 0.95, "", fontsize=26,
    transform=ax21.transAxes, **time_style
)

# -------------------------------------------------------------------
# animation functions
# -------------------------------------------------------------------
def init():
    p = frames[0]
    kL = savedL[p]
    kR = savedR[p]

    im_nH_L.set_array(runL["nH_hist"][kL].T.ravel())
    im_T_L.set_array(runL["T_hist"][kL].T.ravel())

    im_nH_R.set_array(runR["nH_hist"][kR].T.ravel())
    im_T_R.set_array(runR["T_hist"][kR].T.ravel())

    # show both times (in case they differ)
    txtL.set_text(f"t = {tL[kL]/kyr:.2f} kyr")
    txtR.set_text(f"t = {tR[kR]/kyr:.2f} kyr")

    return (im_nH_L, im_T_L, im_nH_R, im_T_R, txtL, txtR)

def update(p):
    kL = savedL[p]
    kR = savedR[p]

    im_nH_L.set_array(runL["nH_hist"][kL].T.ravel())
    im_T_L.set_array(runL["T_hist"][kL].T.ravel())

    im_nH_R.set_array(runR["nH_hist"][kR].T.ravel())
    im_T_R.set_array(runR["T_hist"][kR].T.ravel())

    txtL.set_text(f"t = {tL[kL]/kyr:.2f} kyr")
    txtR.set_text(f"t = {tR[kR]/kyr:.2f} kyr")

    return (im_nH_L, im_T_L, im_nH_R, im_T_R, txtL, txtR)

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

if save_mp4:
    ani.save(outname, writer="ffmpeg", fps=fps, dpi=300)
else:
    plt.show()
