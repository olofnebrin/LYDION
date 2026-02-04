import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

kms = 1e5

# ---------------- User inputs ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
filename   = os.path.join(script_dir, "HydroStaticTest_128x256_Zfull_t50.00.h5")

SaveFigure = True
OutName    = os.path.join(script_dir, "HydroStatic_SphericalProfiles_128.pdf")
# ---------------------------------------------


def read_first_existing(f, candidates):
    """Return f[name][:] for the first dataset name that exists, else None."""
    for name in candidates:
        if name in f:
            return f[name][:]
    return None


with h5py.File(filename, "r") as f:
    print("HDF5 keys:")
    for k in f.keys():
        print("  ", k)

    T = read_first_existing(f, ["Times", "t", "Time"])
    Rbins = read_first_existing(f, ["Rbins", "R_bins", "R", "RadiusBins"])

    # This is the one you want to plot
    RhoHist = read_first_existing(f, ["SphericalAverageDensity", "RhoHist", "RhoHistory", "DensityHistory"])

    # Optional exact profile (if present)
    RhoExact = read_first_existing(f, ["DensityExact", "RhoExact", "ExactDensity", "SphericalAverageDensityExact"])

# Basic sanity
if T is None or Rbins is None or RhoHist is None:
    raise RuntimeError("Could not find one of: Times, Rbins, SphericalAverageDensity (see printed keys above).")

# Normalize the radial bins to max radius:
Rbins = Rbins / Rbins[-1]

# Ensure orientation (Nbin, Nsaved)
if RhoHist.shape[0] != len(Rbins):
    RhoHist = RhoHist.T

Nbin, Nsaved = RhoHist.shape

# Choose snapshots evenly spaced in time
t_targets = np.linspace(T[0], T[-1], 4)
snap_ids  = [np.argmin(np.abs(T - t)) for t in t_targets]

print("Selected snapshot times:")
for it in snap_ids:
    print(f"  t = {T[it]:.6e}")

###############################################################
#######           P L O T   C O S M E T I C S           #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 25})
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

plt.rcParams.update({'legend.fontsize': 15})
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

fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))


# Get the resolution automatically from the filename (use A as reference)
import re
basename = os.path.basename(filename)
m = re.search(r'_(\d+)x(\d+)_', basename)
if m is not None:
    Nx, Ny = m.groups()
    res_label = fr"Resolution: {Nx} $\times$ {Ny}"
else:
    res_label = "Resolution: unknown"

colors = ["dodgerblue", "orangered", "black", "deeppink"]

# Optional exact profile (if present)
if RhoExact is not None:
    if RhoExact.shape[0] != len(Rbins):
        RhoExact = RhoExact.T
    ax.plot(Rbins, RhoExact / 1e-20, linewidth = 4.5, color = "black", linestyle="--",
            dash_capstyle = "round", label = "Analytical")

for c, it in zip(colors, snap_ids):
    rho = RhoHist[:, it]
    t_tdyn = T[it]
    ax.plot(Rbins, rho / 1e-20, linewidth = 2.5, color = c, linestyle = "--",
            dash_capstyle = "round", 
            label = rf"$t/t_{{dyn}} = {{{t_tdyn:.2f}}}$")

# Axes styling (linear y)
ax.set_xlabel(r"Distance from center (r/2R$_{cl}$)", labelpad=10)
ax.set_ylabel(r"Density $\rho$ (10$^{-20}$ g cm$^{-3}$)", labelpad=10)

ax.set_xlim(0.0, Rbins[-1])

# Auto y-limits from data (robust)
ax.set_ylim(-0.1, 2.0)

ax.xaxis.set_minor_locator(AutoMinorLocator(10))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

ax.legend(ncol = 2, borderaxespad = 1.2)
ax.text(0.4, 1.0, res_label, fontsize = 18)

plt.tight_layout()

if SaveFigure:
    plt.savefig(OutName, dpi=300)
    print(f"Saved plot to {OutName}")
else:
    plt.show()
