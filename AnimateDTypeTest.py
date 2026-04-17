import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator  # add this import
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
cmastro_src = repo_root / "src" / "cmastro" / "src"

sys.path.insert(0, str(cmastro_src))

import cmastro



# ---------------- user inputs ----------------    
filename = "DTypeTest.h5" 
frameskip = 1                 
save_mp4 = True               
outname = "DTypeTest_Animation.mp4"
fps = 20
# ---------------------------------------------

# Constants
mH  = 1.67e-24
X   = 1.0              # Hydrogen mass fraction is 1 in the D-type test 
yr  = 3.1556926e7
kyr = 1e3 * yr
Myr = 1e6 * yr
pc  = 3.085677581e18

# --- read data ---
with h5py.File(filename, "r") as f:
    t             = f["Times"][:]                          # (Nt,)
    rho_hist      = f["GasDensity_History"][:]             # (Nt, Nr, Nz) or (Nt, Nz, Nr)
    T_hist        = f["T_History"][:]                      # (Nt, Nr, Nz) or (Nt, Nz, Nr)
    xH2_hist      = f["xH2_History"][:]                    # (Nt, Nr, Nz) or (Nt, Nz, Nr)
    xHII_hist     = f["xHII_History"][:]                   # (Nt, Nr, Nz) or (Nt, Nz, Nr)
    xCI_hist      = f["xCI_History"][:]                    # (Nt, Nr, Nz) or (Nt, Nz, Nr)
    xCII_hist     = f["xCII_History"][:]                   # (Nt, Nr, Nz) or (Nt, Nz, Nr)
    xCO_hist      = f["xCO_History"][:]                    # (Nt, Nr, Nz) or (Nt, Nz, Nr)
    xOI_hist      = f["xOI_History"][:]                    # (Nt, Nr, Nz) or (Nt, Nz, Nr)
    xOII_hist     = f["xOII_History"][:]                   # (Nt, Nr, Nz) or (Nt, Nz, Nr)
    r             = f["r"][:]                              # edges, (Nr+1,)
    Z             = f["Z"][:]                              # edges, (Nz+1,)


xHI_hist  = 1.0 - 2.0 * xH2_hist - xHII_hist


# shapes (needed by ensure_NtNrNz)
Nt_raw, a1_rho, a2_rho = rho_hist.shape

# find indices with valid saved data (nonzero or t=0)
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

# Make ticks appear on all sides and point inwards
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# Ensure (Nt, Nr, Nz) ordering 
Nr = len(r) - 1
Nz = len(Z) - 1

def ensure_NtNrNz(arr, a1, a2):
    if (a1, a2) == (Nr, Nz):
        return arr
    elif (a1, a2) == (Nz, Nr):
        return arr.transpose(0, 2, 1)
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} vs edges Nr={Nr}, Nz={Nz}")

rho_hist   = ensure_NtNrNz(rho_hist,  a1_rho, a2_rho)
T_hist     = ensure_NtNrNz(T_hist,    T_hist.shape[1],   T_hist.shape[2])
xH2_hist   = ensure_NtNrNz(xH2_hist,  xH2_hist.shape[1], xH2_hist.shape[2])
xHII_hist  = ensure_NtNrNz(xHII_hist, xHII_hist.shape[1], xHII_hist.shape[2])
xCO_hist   = ensure_NtNrNz(xCO_hist, xCO_hist.shape[1], xCO_hist.shape[2])
Nt, _, _   = rho_hist.shape

xCHx_hist  = 1.0 - xCI_hist - xCII_hist - xCO_hist
xOHx_hist  = 1.0 - xOI_hist - xOII_hist - 0.5493 * xCO_hist

# convert to number density
nH_hist = X * rho_hist / mH   # (Nt, Nr, Nz)

# grid for pcolormesh (edges, in pc)
R, ZZ = np.meshgrid(r/pc, Z/pc, indexing='ij')

# display ranges
vmin_nH, vmax_nH = 1e-4, 0.01
norm_nH = LogNorm(vmin=vmin_nH, vmax=vmax_nH)

vmin_T, vmax_T = 1, 1e5
norm_T = LogNorm(vmin=vmin_T, vmax=vmax_T)

vmin_x, vmax_x = 1e-8, 1.0
norm_x = LogNorm(vmin=vmin_x, vmax=vmax_x)

# --- figure with 2x2 layout ---
fig = plt.figure(figsize=(16, 12), constrained_layout=True)
gs  = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])

ax1 = fig.add_subplot(gs[0, 0])                           # density
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)   # temperature
ax3 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)   # xH2
ax4 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax1)   # xHII

ax1.tick_params(labelbottom=False)
ax3.tick_params(labelbottom=False)

# --- density ---
ax1.set_ylabel("Z (pc)", labelpad=12)
im_nH = ax1.pcolormesh(R, ZZ, nH_hist[0].T, norm=norm_nH, shading='auto', cmap='cma:laguna')
cb1 = fig.colorbar(im_nH, ax=ax1, pad=0.02)
cb1.set_ticks([1e-4, 1e-3, 0.01])
cb1.set_ticklabels(['10$^{-4}$', '10$^{-3}$', '0.01'])
panel_label_style = dict(ha='right', va='top', color='black',
                         path_effects=[pe.withStroke(linewidth=3, foreground='white')])
ax1.text(0.9, 0.2, "n$_{H}$ (cm$^{-3}$)", transform=ax1.transAxes, **panel_label_style)

ax1.set_xlim( r[0]/pc, r[-1]/pc)
ax1.set_ylim( Z[0]/pc, Z[-1]/pc )

# --- temperature ---
ax2.set_xlabel("r (pc)", labelpad=12)
ax2.set_ylabel("Z (pc)", labelpad=12)
im_T = ax2.pcolormesh(R, ZZ, T_hist[0].T, norm=norm_T, shading='auto', cmap='coolwarm')
cb2 = fig.colorbar(im_T, ax=ax2, pad=0.02)
cb2.set_ticks([1, 10, 100, 1e3, 1e4, 1e5])
cb2.set_ticklabels(['1', '10', '10$^{2}$', '10$^{3}$', '10$^{4}$', '10$^{5}$'])
ax2.text(0.9, 0.9, "T (K)", transform=ax2.transAxes, **panel_label_style)

# --- H I fraction ---
ax3.tick_params(labelleft=False)
im_xHI = ax3.pcolormesh(R, ZZ, np.clip(xHI_hist[0].T, vmin_x, vmax_x),
                        norm=norm_x, shading='auto', cmap='cma:emph')

cb3 = fig.colorbar(im_xHI, ax=ax3, pad=0.02)
cb3.set_ticks([1e-8, 1e-6, 1e-4, 0.01, 1])
cb3.set_ticklabels(['10$^{-8}$', '10$^{-6}$', '10$^{-4}$', '0.01', '1'])
ax3.text(0.9, 0.9, "x$_{HI}$", transform=ax3.transAxes, **panel_label_style)

# --- xHII ---
ax4.set_xlabel("r (pc)", labelpad=12)
ax4.tick_params(labelleft=False)
im_xHII = ax4.pcolormesh(R, ZZ, np.clip(xHII_hist[0].T, vmin_x, vmax_x),
                         norm=norm_x, shading='auto', cmap='cma:emph')
cb4     = fig.colorbar(im_xHII, ax=ax4, pad=0.02)
cb4.set_ticks([1e-8, 1e-6, 1e-4, 0.01, 1])
cb4.set_ticklabels(['10$^{-8}$', '10$^{-6}$', '10$^{-4}$', '0.01', '1'])
ax4.text(0.9, 0.9, "x$_{HII}$", transform=ax4.transAxes, **panel_label_style)

# --- Time label ---
time_style = dict(ha='left', va='top', color='white',
                  path_effects=[pe.withStroke(linewidth=3, foreground='black')])
txt = ax1.text(0.5, 0.9, "", fontsize=28, transform=ax1.transAxes, **time_style)

# --- frames to animate: only where data exists ---
frames = saved_idx[::frameskip]

# --- animation functions ---
def init():
    k0 = frames[0]
    im_nH.set_array(nH_hist[k0].T.ravel())
    im_T.set_array(T_hist[k0].T.ravel())
    im_xHI.set_array(np.clip(xHI_hist[k0].T, vmin_x, vmax_x).ravel())
    im_xHII.set_array(np.clip(xHII_hist[k0].T, vmin_x, vmax_x).ravel())
    txt.set_text(f"t = {t[k0]/Myr:.2f} kyr")
    return im_nH, im_T, im_xHI, im_xHII, txt

def update(k):
    im_nH.set_array(nH_hist[k].T.ravel())
    im_T.set_array(T_hist[k].T.ravel())
    im_xHI.set_array(np.clip(xHI_hist[k].T, vmin_x, vmax_x).ravel())
    im_xHII.set_array(np.clip(xHII_hist[k].T, vmin_x, vmax_x).ravel())
    txt.set_text(f"t = {t[k]/Myr:.2f} kyr")
    return im_nH, im_T, im_xHI, im_xHII, txt

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

if save_mp4:
    ani.save(outname, writer="ffmpeg", fps=fps, dpi=300)
else:
    plt.show()
