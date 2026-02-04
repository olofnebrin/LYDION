import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.patheffects as pe
from pathlib import Path
import sys

# NEW (only import you need for the clean colorbar)
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------- user inputs ----------------
here = Path(__file__).resolve().parent
filename = here / "HydroStaticTest_128x256_Zfull_t50.00.h5"
outname  = here / "HydroStaticTest_rho_final.pdf"
use_number_density = False
dpi = 300
# ---------------------------------------------

# constants (only used if use_number_density=True)
mH  = 1.67e-24
X   = 0.76
pc  = 3.085677581e18

def add_cmastro_to_path():
    here_ = Path(__file__).resolve()
    for parent in [here_.parent, *here_.parents]:
        candidate = parent / "src" / "cmastro" / "src"
        if (candidate / "cmastro" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return str(candidate)
    raise RuntimeError("Could not find src/cmastro/src containing cmastro/__init__.py")

add_cmastro_to_path()
import cmastro  

###############################################################
#######           P L O T   C O S M E T I C S           #######
###############################################################
plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 45})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

plt.rcParams.update({'legend.fontsize': 19})
plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 2.1
plt.rcParams['xtick.minor.width'] = 2.1
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.width'] = 2.1
plt.rcParams['ytick.minor.width'] = 2.1
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['ytick.minor.size'] = 8
plt.rcParams['xtick.major.pad'] = '15'
plt.rcParams['ytick.major.pad'] = '15'

plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.linewidth'] = 2.8
plt.rcParams['grid.linewidth'] = 2.8
plt.rcParams['grid.linestyle'] = '--'

plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


with h5py.File(filename, "r") as f:
    t        = f["Times"][:]                      # (Nsaved_time,)
    rho_hist = f["GasDensity_History"][:]         # 3D
    r        = f["r"][:]                          # edges (Nr+1,)
    Z        = f["Z"][:]                          # edges (Nz+1,)

Nr = len(r) - 1
Nz = len(Z) - 1

if rho_hist.ndim != 3:
    raise ValueError(f"GasDensity_History is not 3D. shape={rho_hist.shape}")

sh = rho_hist.shape

def last_valid_k(t_arr, Nt):
    Nt = int(Nt)
    if Nt <= 0:
        return 0
    if t_arr.size == 0:
        return Nt - 1
    t_use = t_arr[:min(t_arr.size, Nt)]
    saved = np.where((t_use > 0) | (np.arange(t_use.size) == 0))[0]
    if saved.size == 0:
        return 0
    return int(saved[-1])

# Determine which axis is which, extract last valid snapshot
if sh[0] == Nr and sh[1] == Nz:                  # (Nr, Nz, Nt)
    Nt = sh[2]
    k_plot = last_valid_k(t, Nt)
    rho2d = rho_hist[:, :, k_plot]
elif sh[0] == Nz and sh[1] == Nr:                # (Nz, Nr, Nt)
    Nt = sh[2]
    k_plot = last_valid_k(t, Nt)
    rho2d = rho_hist[:, :, k_plot].T
elif sh[1] == Nr and sh[2] == Nz:                # (Nt, Nr, Nz)
    Nt = sh[0]
    k_plot = last_valid_k(t, Nt)
    rho2d = rho_hist[k_plot, :, :]
elif sh[1] == Nz and sh[2] == Nr:                # (Nt, Nz, Nr)
    Nt = sh[0]
    k_plot = last_valid_k(t, Nt)
    rho2d = rho_hist[k_plot, :, :].T
else:
    raise ValueError(
        f"Unexpected GasDensity_History shape={sh}. "
        f"Expected something compatible with Nr={Nr}, Nz={Nz}."
    )

rho2d = rho2d.astype(np.float64)
if rho2d.shape != (Nr, Nz):
    raise ValueError(f"After selection rho2d.shape={rho2d.shape}, expected {(Nr, Nz)}")

t_label = t[k_plot] if (k_plot < t.size) else float(k_plot)

if use_number_density:
    C = X * rho2d / mH
    cbar_label = r"n$_H$ (cm$^{-3}$)"
else:
    C = rho2d
    cbar_label = r"$\rho$ (g cm$^{-3}$)"

R, ZZ = np.meshgrid(r/pc, Z/pc, indexing="ij")  # edges

positive = C[C > 0]
if positive.size == 0:
    raise ValueError("All density values are <= 0; cannot use LogNorm.")
vmin = np.percentile(positive, 1.0)
vmax = np.percentile(positive, 99.0)
if vmin >= vmax:
    vmin = positive.min()
    vmax = positive.max() if positive.max() > positive.min() else positive.min() * 10.0
norm = LogNorm(vmin=vmin, vmax=vmax)

############ PLOT #######################
fig, ax = plt.subplots(figsize=(8, 16))  # Matches Zmax = 2 * Rmax aspect ratio

im = ax.pcolormesh(R, ZZ, C, norm=norm, 
                   shading="auto", 
                   cmap="cma:laguna",
                   rasterized=True)

ax.set_xlabel("R (pc)", labelpad=12)
ax.set_ylabel("Z (pc)", labelpad=12)
ax.set_xlim(r[0]/pc, r[-1]/pc)
ax.set_ylim(Z[0]/pc, Z[-1]/pc)
ax.set_aspect('equal', adjustable='box')

# Colorbar: same height as panel, always
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4.5%", pad=0.1)
cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(pad=8)

panel_label_style = dict(
    ha="right", va="top", color="black",
    path_effects=[pe.withStroke(linewidth=3, foreground="white")]
)
ax.text(0.95, 0.12, cbar_label, transform=ax.transAxes, **panel_label_style,
        fontsize = 55)

time_style = dict(
    ha="left", va="top", color="white",
    path_effects=[pe.withStroke(linewidth=3, foreground="black")]
)
ax.text(0.1, 0.95, f"t/t$_{{dyn}}$ = {t_label:.6g}", transform=ax.transAxes, 
        fontsize = 55, **time_style)


ax.xaxis.set_major_locator(MultipleLocator(1.0))
ax.yaxis.set_major_locator(MultipleLocator(1.0))

ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))


fig.savefig(outname, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
print(f"Saved: {outname}")


