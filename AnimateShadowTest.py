import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
cmastro_src = repo_root / "src" / "cmastro" / "src"

sys.path.insert(0, str(cmastro_src))

import cmastro


# ---------------- user inputs ----------------
filename = "ShadowTest_OTStrue_Hydrofalse.h5"

frameskip = 1 
save_mp4 = True
outname = "density_T.mp4"
fps = 20

# For non-square domains:
# True  -> physical cells have correct aspect ratio; long Z boxes look tall.
# False -> panels are visually compact, but the physical aspect ratio is not preserved.
enforce_equal_aspect = True

# Use this to make the figure taller for your 3:1 Z:r shadow box.
figsize = (16, 22)
# ---------------------------------------------

# constants
mH  = 1.67e-24
X   = 0.76
yr  = 3.1556926e7
kyr = 1e3 * yr
pc  = 3.085677581e18

# --- read data ---
with h5py.File(filename, "r") as f:
    t             = f["Times"][:]
    rho_hist      = f["GasDensity_History"][:]
    rhod_Sil_hist = f["DustDensity_Sil_History"][:]
    rhod_C_hist   = f["DustDensity_C_History"][:]
    rhod_PAH_hist = f["DustDensity_PAH_History"][:]
    J_EUV1_hist   = f["J_EUV1_History"][:]
    J_EUV2_hist   = f["J_EUV2_History"][:]
    J_EUV3_hist   = f["J_EUV3_History"][:]
    T_hist        = f["T_History"][:]
    xH2_hist      = f["xH2_History"][:]
    xHII_hist     = f["xHII_History"][:]
    xCI_hist      = f["xCI_History"][:]
    xCII_hist     = f["xCII_History"][:]
    xCO_hist      = f["xCO_History"][:]
    xOI_hist      = f["xOI_History"][:]
    xOII_hist     = f["xOII_History"][:]
    Zgr_Sil_hist  = f["Zgr_Sil_History"][:]
    Zgr_C_hist    = f["Zgr_C_History"][:]
    Zgr_PAH_hist  = f["Zgr_PAH_History"][:]
    r             = f["r"][:]      # edges, (Nr+1,)
    Z             = f["Z"][:]      # edges, (Nz+1,)

# grid sizes from saved edges
Nr = len(r) - 1
Nz = len(Z) - 1

# find indices with valid saved data
saved_idx = np.where((t > 0) | (np.arange(t.size) == 0))[0]


###############################################################
#######           P L O T   C O S M E T I C S           #######
###############################################################

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


# Ensure (Nt, Nr, Nz) ordering.
# This is important now that Nr != Nz, because the old square case could hide mistakes.
def ensure_NtNrNz(arr, name="array"):
    if arr.ndim != 3:
        raise ValueError(f"{name} should have shape (Nt, Nr, Nz) or (Nt, Nz, Nr), got {arr.shape}")

    Nt_arr, a1, a2 = arr.shape

    if (a1, a2) == (Nr, Nz):
        return arr
    elif (a1, a2) == (Nz, Nr):
        return arr.transpose(0, 2, 1)
    else:
        raise ValueError(
            f"Unexpected shape for {name}: {arr.shape}. "
            f"Expected (Nt, Nr, Nz)=({Nt_arr}, {Nr}, {Nz}) "
            f"or (Nt, Nz, Nr)=({Nt_arr}, {Nz}, {Nr})."
        )


rho_hist      = ensure_NtNrNz(rho_hist,      "rho_hist")
rhod_Sil_hist = ensure_NtNrNz(rhod_Sil_hist, "rhod_Sil_hist")
rhod_C_hist   = ensure_NtNrNz(rhod_C_hist,   "rhod_C_hist")
rhod_PAH_hist = ensure_NtNrNz(rhod_PAH_hist, "rhod_PAH_hist")
J_EUV1_hist   = ensure_NtNrNz(J_EUV1_hist,   "J_EUV1_hist")
J_EUV2_hist   = ensure_NtNrNz(J_EUV2_hist,   "J_EUV2_hist")
J_EUV3_hist   = ensure_NtNrNz(J_EUV3_hist,   "J_EUV3_hist")
T_hist        = ensure_NtNrNz(T_hist,        "T_hist")
xH2_hist      = ensure_NtNrNz(xH2_hist,      "xH2_hist")
xHII_hist     = ensure_NtNrNz(xHII_hist,     "xHII_hist")
xCI_hist      = ensure_NtNrNz(xCI_hist,      "xCI_hist")
xCII_hist     = ensure_NtNrNz(xCII_hist,     "xCII_hist")
xCO_hist      = ensure_NtNrNz(xCO_hist,      "xCO_hist")
xOI_hist      = ensure_NtNrNz(xOI_hist,      "xOI_hist")
xOII_hist     = ensure_NtNrNz(xOII_hist,     "xOII_hist")
Zgr_Sil_hist  = ensure_NtNrNz(Zgr_Sil_hist,  "Zgr_Sil_hist")
Zgr_C_hist    = ensure_NtNrNz(Zgr_C_hist,    "Zgr_C_hist")
Zgr_PAH_hist  = ensure_NtNrNz(Zgr_PAH_hist,  "Zgr_PAH_hist")

Nt, _, _ = rho_hist.shape

xHI_hist  = 1.0 - 2.0 * xH2_hist - xHII_hist
rhod_hist = rhod_Sil_hist + rhod_C_hist + rhod_PAH_hist

xCHx_hist = 1.0 - xCI_hist - xCII_hist - xCO_hist
xOHx_hist = 1.0 - xOI_hist - xOII_hist - 0.5493 * xCO_hist

DG_hist = rhod_hist / (rho_hist * 1e-2 * 1/162)   # Dust-to-gas mass ratio in Solar units

# convert to number density
nH_hist = X * rho_hist / mH   # (Nt, Nr, Nz)

# 1D grid edges for pcolormesh.
# IMPORTANT:
# Matplotlib expects C shape = (Ny, Nx) = (Nz, Nr),
# while LYDION fields are field[i,j] = field[r_index, Z_index] = (Nr, Nz).
# Therefore we plot field.T.
r_pc = r / pc
Z_pc = Z / pc

# display ranges
vmin_nH, vmax_nH = 0.1, 3e6
norm_nH = LogNorm(vmin=vmin_nH, vmax=vmax_nH)

vmin_T, vmax_T = 1, 1e5
norm_T = LogNorm(vmin=vmin_T, vmax=vmax_T)

vmin_J, vmax_J = 1e-6, 1e3
norm_J = LogNorm(vmin=vmin_J, vmax=vmax_J)

vmin_x, vmax_x = 1e-4, 1.0
norm_x = LogNorm(vmin=vmin_x, vmax=vmax_x)

# --- figure with 2x2 layout ---
fig = plt.figure(figsize=figsize, constrained_layout=True)
gs  = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])

ax1 = fig.add_subplot(gs[0, 0])                         # density
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1) # temperature
ax3 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1) # dust-to-gas
ax4 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax1) # xHII

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

im_nH = ax1.pcolormesh(
    r_pc,
    Z_pc,
    nH_hist[0].T,
    norm=norm_nH,
    shading='auto',
    cmap='cma:laguna'
)

cb1 = fig.colorbar(im_nH, ax=ax1, pad=0.02)
cb1.set_ticks([1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])
cb1.set_ticklabels([
    '1', '10', '10$^{2}$', '10$^{3}$', '10$^{4}$',
    '10$^{5}$', '10$^{6}$'
])

ax1.text(0.9, 0.2, "n$_{H}$ (cm$^{-3}$)", transform=ax1.transAxes, **panel_label_style)

ax1.set_xlim(r_pc[0], r_pc[-1])
ax1.set_ylim(Z_pc[0], Z_pc[-1])

# --- temperature ---
ax2.set_xlabel("r (pc)", labelpad=12)
ax2.set_ylabel("Z (pc)", labelpad=12)

im_T = ax2.pcolormesh(
    r_pc,
    Z_pc,
    T_hist[0].T,
    norm=norm_T,
    shading='auto',
    cmap='coolwarm'
)

cb2 = fig.colorbar(im_T, ax=ax2, pad=0.02)
cb2.set_ticks([1, 10, 100, 1e3, 1e4, 1e5])
cb2.set_ticklabels(['1', '10', '10$^{2}$', '10$^{3}$', '10$^{4}$', '10$^{5}$'])

ax2.text(0.9, 0.9, "T (K)", transform=ax2.transAxes, **panel_label_style)

# --- Ionizing intensity ---
ax3.tick_params(labelleft=False)

im_J = ax3.pcolormesh(
    r_pc,
    Z_pc,
    np.clip(J_EUV1_hist[0].T, vmin_J, vmax_J),
    norm=norm_J,
    shading='auto',
    cmap='cma:lacerta'
)

cb3 = fig.colorbar(im_J, ax=ax3, pad=0.02)
cb3.set_ticks([0.1, 1, 10])
cb3.set_ticklabels(['0.1', '1', '10'])

ax3.text(0.9, 0.9, "J$_{EUV1}$", transform=ax3.transAxes, **panel_label_style)

# --- xHII ---
ax4.set_xlabel("r (pc)", labelpad=12)
ax4.tick_params(labelleft=False)

im_xHII = ax4.pcolormesh(
    r_pc,
    Z_pc,
    np.clip(xHII_hist[0].T, vmin_x, vmax_x),
    norm=norm_x,
    shading='auto',
    cmap='magma'
)

cb4 = fig.colorbar(im_xHII, ax=ax4, pad=0.02)
cb4.set_ticks([1e-4, 0.01, 1])
cb4.set_ticklabels(['10$^{-4}$', '0.01', '1'])

ax4.text(0.9, 0.9, "x$_{HII}$", transform=ax4.transAxes, **panel_label_style)

# Preserve physical aspect ratio if desired.
# For your shadow test, this will make the Z-extended domain visibly tall.
if enforce_equal_aspect:
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_aspect('equal', adjustable='box')

# --- optional minor ticks ---
# Adjust these to taste for your shadow-box size.
for ax in (ax1, ax2, ax3, ax4):
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))

# --- time label ---
time_style = dict(
    ha='left',
    va='top',
    color='white',
    path_effects=[pe.withStroke(linewidth=3, foreground='black')]
)

txt = ax1.text(0.5, 0.9, "", fontsize=28, transform=ax1.transAxes, **time_style)

# --- frames to animate: only where data exists ---
frames = saved_idx[::frameskip]

# --- animation functions ---
def init():
    k0 = frames[0]

    im_nH.set_array(nH_hist[k0].T.ravel())
    im_T.set_array(T_hist[k0].T.ravel())
    im_J.set_array(np.clip(J_EUV1_hist[k0].T, vmin_J, vmax_J).ravel())
    im_xHII.set_array(np.clip(xHII_hist[k0].T, vmin_x, vmax_x).ravel())

    txt.set_text(f"t = {t[k0]/kyr:.2f} kyr")

    return im_nH, im_T, im_J, im_xHII, txt


def update(k):
    im_nH.set_array(nH_hist[k].T.ravel())
    im_T.set_array(T_hist[k].T.ravel())
    im_J.set_array(np.clip(J_EUV1_hist[k].T, vmin_J, vmax_J).ravel())
    im_xHII.set_array(np.clip(xHII_hist[k].T, vmin_x, vmax_x).ravel())

    txt.set_text(f"t = {t[k]/kyr:.2f} kyr")

    return im_nH, im_T, im_J, im_xHII, txt


ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

if save_mp4:
    ani.save(outname, writer="ffmpeg", fps=fps, dpi=300)
else:
    plt.show()