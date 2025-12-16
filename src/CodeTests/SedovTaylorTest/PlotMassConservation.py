#!/usr/bin/env python3
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---------------- User inputs ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
filename   = os.path.join(script_dir, "SedovTest_E1.00e+00_rho1_gamma1.67_t0.4_512x512.h5")
outname    = os.path.join(script_dir, "TotalMass_vs_Time.png")
save_fig   = True
# ---------------------------------------------


# --- Read data ---
with h5py.File(filename, "r") as f:
    t          = f["Times"][:]               # (Nt,)
    rho_hist   = f["GasDensity_History"][:]  # (Nt, Nr, Nz) or (Nr, Nz, Nt)
    cell_vol   = f["CellVolumes"][:]         # should be (Nr, Nz) or (Nz, Nr)


# --- Ensure rho_hist has shape (Nt, Nr, Nz) ---
def ensure_NtNrNz(arr, Nt):
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array for rho_hist, got shape {arr.shape}")
    if arr.shape[0] == Nt:
        # already (Nt, Nr, Nz)
        return arr
    elif arr.shape[2] == Nt:
        # (Nr, Nz, Nt) -> transpose
        return arr.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unexpected rho_hist shape {arr.shape} vs Nt={Nt}")

rho_hist = ensure_NtNrNz(rho_hist, len(t))
Nt, Nr, Nz = rho_hist.shape


# --- Ensure cell_vol has shape (Nr, Nz) ---
def ensure_NrNz(vol, Nr, Nz):
    if vol.ndim == 2:
        if vol.shape == (Nr, Nz):
            return vol
        elif vol.shape == (Nz, Nr):
            return vol.T
        else:
            raise ValueError(f"Unexpected CellVolumes shape {vol.shape}, expected ({Nr},{Nz}) or ({Nz},{Nr})")
    elif vol.ndim == 3:
        # In case volumes were stored per time-step, take t=0 and fix orientation
        if vol.shape[0] == Nt and vol.shape[1:] == (Nr, Nz):
            return vol[0]
        elif vol.shape[0] == Nt and vol.shape[1:] == (Nz, Nr):
            return vol[0].T
        else:
            raise ValueError(f"Unexpected 3D CellVolumes shape {vol.shape}")
    else:
        raise ValueError(f"CellVolumes has unsupported ndim={vol.ndim}, shape={vol.shape}")

cell_vol = ensure_NrNz(cell_vol, Nr, Nz)


# --- Compute total mass as a function of time ---
# M(t) = Σ_{i,j} ρ(t,i,j) * V(i,j)
# cell_vol has shape (Nr, Nz), rho_hist has shape (Nt, Nr, Nz)
# broadcasting: (Nt, Nr, Nz) * (Nr, Nz) -> (Nt, Nr, Nz)
mass_t = np.sum(rho_hist * cell_vol[None, :, :], axis=(1, 2))

M0 = mass_t[0]
mass_ratio = mass_t / M0

print(f"Initial total mass M0 = {M0:.8e}")
print(f"Min(M(t)/M0) = {mass_ratio.min():.8e}")
print(f"Max(M(t)/M0) = {mass_ratio.max():.8e}")


###############################################################
#######                                                 #######
#######           P L O T   C O S M E T I C S           #######
#######                                                 #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 25})
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
plt.rcParams['xtick.major.size']  = 0.5*20
plt.rcParams['xtick.minor.size']  = 0.5*8
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size']  = 0.5*20
plt.rcParams['ytick.minor.size']  = 0.5*8
plt.rcParams['xtick.major.pad']   = '15'
plt.rcParams['ytick.major.pad']   = '15'

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


###############################################################
#######                                                 #######
#######               P L O T  I T !                    #######
#######                                                 #######
###############################################################

fig, ax = plt.subplots(figsize=(8, 6))

# If your Times are in code units, you might want to label them accordingly.
ax.plot(t, mass_ratio, lw=2.5, color="tab:blue", label=r"$M(t) / M_0$")

ax.set_xlabel(r"$t$", labelpad=12)
ax.set_ylabel(r"Total mass (normalized)", labelpad=12)
ax.set_title(r"Mass conservation", y=1.03)

# Horizontal line at 1 for perfect conservation
#ax.axhline(1.0, color="k", ls="--", lw=1.5, alpha=0.7)

ax.set_ylim(0.999, 1.001)
ax.legend(loc="best")

# Minor ticks
ax.xaxis.set_minor_locator(MultipleLocator(0.05 * (t.max() - t.min())))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))

plt.tight_layout()

if save_fig:
    plt.savefig(outname, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {outname}")
else:
    plt.show()
