import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
os.chdir(repo_root)

# ---------------- user inputs ----------------
filename    = "RHDSim.h5"
Z_slice_pc  = 0.25       # Z-plane in pc where we want the abundance profiles
save_figure = True
outname     = "abundances_Zslice.png"
# ---------------------------------------------

# constants
mH  = 1.67e-24
X   = 0.76
pc  = 3.085677581e18

# ============================================================
#                 READ HDF5 AND FIX AXIS ORDER
# ============================================================

with h5py.File(filename, "r") as f:
    t         = f["Times"][:]  
    rho_hist  = f["GasDensity_History"][:]  
    T_hist    = f["T_History"][:]  
    xH2_hist  = f["xH2_History"][:]  
    xHII_hist = f["xHII_History"][:]  

    # These are cell centers (correct ones)
    rCoM      = f["rCoM"][:] / pc     # convert to pc
    ZCoM      = f["ZCoM"][:] / pc     # convert to pc

# Dimensions
Nt_data, A1, A2 = rho_hist.shape
Nr = len(rCoM)
Nz = len(ZCoM)

print("Loaded shapes:")
print("rho_hist:", rho_hist.shape)
print("T_hist:", T_hist.shape)

# ------------------------------------------------------------
# FIX AXIS ORDER
# Your saved data is (Nt, Nz, Nr) but we want (Nt, Nr, Nz)
# ------------------------------------------------------------
if A1 == Nz and A2 == Nr:
    print("Transposing all datasets from (Nt, Nz, Nr) → (Nt, Nr, Nz)")
    rho_hist  = np.transpose(rho_hist,  (0,2,1))
    T_hist    = np.transpose(T_hist,    (0,2,1))
    xH2_hist  = np.transpose(xH2_hist,  (0,2,1))
    xHII_hist = np.transpose(xHII_hist, (0,2,1))
elif A1 == Nr and A2 == Nz:
    print("Data already in (Nt, Nr, Nz).")
else:
    raise RuntimeError("Unrecognized array shape; Nr,Nz inconsistent.")

Nt, Nr_chk, Nz_chk = rho_hist.shape
assert Nr_chk == Nr and Nz_chk == Nz

print("Final shapes (Nt, Nr, Nz):", rho_hist.shape)

# ============================================================
#             DERIVE NEUTRAL FRACTION
# ============================================================

xHI_hist = 1.0 - 2.0 * xH2_hist - xHII_hist

# ============================================================
#       SELECT LAST VALID SNAPSHOT (like animation)
# ============================================================

saved_idx = np.where((t > 0) | (np.arange(t.size) == 0))[0]
k_final = saved_idx[-1]

print("Nt =", Nt)
print("Times[0]      =", t[0]/3.1557e7, "years")
print("Times[k_final] =", t[k_final]/3.1557e7, "years")
print("k_final =", k_final)

# ============================================================
#         FIND THE Z-INDEX CLOSEST TO Z_slice_pc
# ============================================================

jZ = np.argmin(np.abs(ZCoM - Z_slice_pc))
Z_used_pc = ZCoM[jZ]

print(f"Requested Z = {Z_slice_pc:.4f} pc, using nearest ZCoM = {Z_used_pc:.4f} pc (index j={jZ})")

# ============================================================
#     EXTRACT RADIAL PROFILES AT FIXED Z = ZCoM[jZ]
# ============================================================

xH2_final  = xH2_hist[k_final, :, jZ]    # (Nr,)
xHII_final = xHII_hist[k_final, :, jZ]
xHI_final  = xHI_hist[k_final, :, jZ]

nH_final   = X * rho_hist[k_final, :, jZ] / mH
T_final    = T_hist[k_final, :, jZ]

print("Final slice checks:")
print("  xHII min/max =", np.min(xHII_final), np.max(xHII_final))
print("  xHI  min/max =", np.min(xHI_final),  np.max(xHI_final))
print("  xH2  min/max =", np.min(xH2_final),  np.max(xH2_final))

# ============================================================
#                      PLOT
# ============================================================



plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 30})
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
plt.rcParams['xtick.major.size']  = 20
plt.rcParams['xtick.minor.size']  = 8
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size']  = 20
plt.rcParams['ytick.minor.size']  = 8
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

fig, ax = plt.subplots(figsize=(10, 7))

ax.semilogy(rCoM, np.clip(xHI_final,  1e-8, 1.0), label=r'$x_{\rm HI}$', linewidth = 2.5)
ax.semilogy(rCoM, np.clip(xHII_final, 1e-8, 1.0), label=r'$x_{\rm HII}$', linewidth = 2.5)
#ax.semilogy(rCoM, np.clip(xH2_final,  1e-8, 1.0), label=r'$x_{\rm H_2}$', linewidth = 2.5)

ax.set_xlabel("r (pc)", labelpad = 12)
ax.set_ylabel("Abundance fraction", labelpad = 12)

ax.set_ylim(1e-6, 2)
ax.set_xlim(0.0, 0.2)

ax.text(0.05, 0.85, fr'$Z = {Z_used_pc:.3f}\ pc$', transform=ax.transAxes,
        ha='left', va='top')

#ax.legend(loc='best')
ax.xaxis.set_minor_locator(MultipleLocator(0.01))

plt.tight_layout()

if save_figure:
    plt.savefig(outname, dpi=300, bbox_inches='tight')
else:
    plt.show()
