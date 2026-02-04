import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import LogLocator, NullFormatter


kms = 1e5

# ---------------- User inputs ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))

filename_A = os.path.join(script_dir, "GravityTest_512x1024_Zfull_t0.07.h5")   # no symmetry
filename_B = os.path.join(script_dir, "GravityTest_512x512_Zhalf_t0.07.h5")   # mirror symmetry

SaveFigure = True
OutName    = os.path.join(script_dir, "GravCollapse_SphericalProfiles_512.pdf")
# ---------------------------------------------


def read_spherical_data(filename):
    with h5py.File(filename, "r") as f:
        T        = f["Times"][:]
        Rbins    = f["Rbins"][:] / (1e3 * kms)
        RhoHist  = f["SphericalAverageDensity"][:]
        RhoExact = f["ExactDensity"][:]
        RExact   = f["ExactRadius"][:] / (1e3 * kms)

    # Ensure correct orientation: (Nbin, Nsaved)
    if RhoHist.shape[0] != len(Rbins):
        RhoHist = RhoHist.T

    return T, Rbins, RhoHist, RhoExact, RExact


# --- Read both datasets ---
T_A, Rbins_A, RhoHist_A, RhoExact_A, RExact_A = read_spherical_data(filename_A)
T_B, Rbins_B, RhoHist_B, RhoExact_B, RExact_B = read_spherical_data(filename_B)

# Keep this if you expect same radial binning; otherwise delete it.
assert np.allclose(Rbins_A, Rbins_B), "Rbins differ between files."

Rbins = Rbins_A

# --- Choose snapshots evenly spaced in time (use overlap range) ---
t_min = max(T_A[0], T_B[0])
t_max = min(T_A[-1], T_B[-1])

t_targets = np.linspace(t_min, t_max, 4)

snap_ids_A = [np.argmin(np.abs(T_A - t)) for t in t_targets]
snap_ids_B = [np.argmin(np.abs(T_B - t)) for t in t_targets]

print("Selected snapshot times (targets and nearest actual):")
for k, t in enumerate(t_targets):
    ia = snap_ids_A[k]
    ib = snap_ids_B[k]
    print(f"  target t = {t:.4e} | A: t = {T_A[ia]:.4e} | B: t = {T_B[ib]:.4e}")

###############################################################
#######           P L O T   C O S M E T I C S           #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 25})
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

params = {'legend.fontsize': 14}
plt.rcParams.update(params)
plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 1.7
plt.rcParams['xtick.minor.width'] = 1.7
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['xtick.major.pad'] = '15'
plt.rcParams['ytick.major.pad'] = '15'

plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['grid.linewidth'] = 2.0
plt.rcParams['grid.linestyle'] = '--'

plt.rcParams['xtick.top']       = True
plt.rcParams['xtick.bottom']    = True
plt.rcParams['ytick.left']      = True
plt.rcParams['ytick.right']     = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

###############################################################
#######                  P L O T                         #######
###############################################################

# Get the resolution automatically from the filename (use A as reference)
import re
basename = os.path.basename(filename_A)
m = re.search(r'_(\d+)x(\d+)_', basename)
if m is not None:
    Nx, Ny = m.groups()
    res_label = fr"Resolution: N$_R$ = {Nx}"
else:
    res_label = "Resolution: unknown"

fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))

colors = ["dodgerblue", "orangered", "black", "deeppink"]

# Line styles to distinguish runs
ls_A_num, ls_A_exact = "-", "--"     # dataset A: solid + dashed
ls_B_num, ls_B_exact = ":", "-."     # dataset B: dotted + dash-dot

label_A = "Full Z"
label_B = "Half Z"

for c, ia, ib in zip(colors, snap_ids_A, snap_ids_B):

    # -------- Dataset A --------
    rhoA = RhoHist_A[:, ia]
    ax.plot(Rbins, rhoA, linewidth = 2.5, color = c, linestyle = ls_A_num, 
            dash_capstyle = "round",
            label=fr"{label_A}, $t = {T_A[ia]:.3f}$")

    rho_extA = rhoA[-1]
    RstepA   = np.array([0.0, RExact_A[ia], RExact_A[ia], Rbins[-1]])
    rhostepA = np.array([RhoExact_A[ia], RhoExact_A[ia], rho_extA, rho_extA])
    ax.plot(RstepA, rhostepA, linewidth = 2.0, color = c, linestyle = ls_A_exact, 
            dash_capstyle = "round")

    # -------- Dataset B --------
    rhoB = RhoHist_B[:, ib]
    ax.plot(Rbins, rhoB, linewidth = 2.5, color=c, linestyle = ls_B_num,
            label = fr"{label_B}, $t = {T_B[ib]:.3f}$")

    rho_extB = rhoB[-1]
    RstepB   = np.array([0.0, RExact_B[ib], RExact_B[ib], Rbins[-1]])
    rhostepB = np.array([RhoExact_B[ib], RhoExact_B[ib], rho_extB, rho_extB])
    ax.plot(RstepB, rhostepB, linewidth = 2.0, color = c, linestyle = ls_B_exact, 
            dash_capstyle = "round")

# --- Axes styling ---
ax.set_xlabel(r"Distance from center (10$^3$ km)", labelpad = 10)
ax.set_ylabel(r"Density $\rho$ (g cm$^{-3}$)", labelpad = 10)
ax.set_yscale("log")
ax.set_xlim(0.0, 7.0)
ax.set_ylim(1e7, 1e13)

ax.xaxis.set_minor_locator(AutoMinorLocator(10))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
ax.yaxis.set_minor_formatter(NullFormatter())


ax.legend(ncol = 2, borderaxespad = 1.2)
ax.text(2.0, 4e10, res_label, fontsize = 20)

plt.tight_layout()

if SaveFigure:
    plt.savefig(OutName, dpi = 300)
    print(f"Saved plot to {OutName}")
else:
    plt.show()
