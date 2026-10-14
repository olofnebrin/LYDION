import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.ticker import LogLocator, NullFormatter
from pathlib import Path


# Read dust Planck & Rosseland mean opacities:

repo_root = Path(__file__).resolve().parent

# Read columns: T_d (K), kappa_P (cm^2 g^-1), kappa_R (cm^2 g^-1)
T_sil, κP_sil, κR_sil = np.loadtxt(repo_root / "kappa_thermal_silicate.txt", unpack=True)
T_C,   κP_C,   κR_C   = np.loadtxt(repo_root / "kappa_thermal_graphite.txt", unpack=True)
T_PAH, κP_PAH, κR_PAH = np.loadtxt(repo_root / "kappa_thermal_PAH.txt", unpack=True)

# Read dust mass fractions from grain_size_stats.txt:
def read_dust_fractions(filename):
    f_Sil = None
    f_C   = None
    f_PAH = None

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()

            if "=" not in line:
                continue

            lhs, rhs = line.split("=", 1)
            key = lhs.strip()
            value = rhs.strip().split()[0]

            if key == "f_PAH":
                f_PAH = float(value)

            elif key == "f_graphite":
                f_C = float(value)

            elif key == "f_silicate":
                f_Sil = float(value)

    if f_Sil is None or f_C is None or f_PAH is None:
        raise ValueError("Could not read all dust mass fractions from grain_size_stats.txt")

    if not np.isclose(f_Sil + f_C + f_PAH, 1.0):
        raise ValueError("Dust mass fractions do not sum to 1.")

    return f_Sil, f_C, f_PAH


f_Sil, f_C, f_PAH = read_dust_fractions(repo_root / "grain_size_stats.txt")

# Total dust Planck & Rosseland mean opacities:
κP_tot = f_Sil * κP_sil  +  f_C * κP_C  +  f_PAH * κP_PAH
κR_tot = f_Sil * κR_sil  +  f_C * κR_C  +  f_PAH * κR_PAH


###############################################################
#######                                                 #######
#######            P L O T  C O S M E T I C S           #######
#######                                                 #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 39.5})
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

params = {'legend.fontsize': 19}
plt.rcParams.update(params)
plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['xtick.minor.width'] = 2.0
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['ytick.minor.width'] = 2.0
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

###############################################################
#######                                                 #######
#######                   P L O T   I T !               #######
#######                                                 #######
###############################################################

fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharey=True)

ax1 = axes[0]
ax2 = axes[1]

color_Sil = 'black'
color_C   = 'orangered'
color_PAH = 'dodgerblue'
color_tot = 'darkviolet'

###############################################################
#######                                                 #######
#######              L E F T   P A N E L                #######
#######                                                 #######
###############################################################

# Planck means (solid)
ax1.loglog(T_C,   κP_C,   lw=3.5, label=r"Graphite, $\kappa_P$", color=color_C)
ax1.loglog(T_PAH, κP_PAH, lw=3.5, label=r"PAH, $\kappa_P$", color=color_PAH)
ax1.loglog(T_sil, κP_sil, lw=3.5, label=r"Silicate, $\kappa_P$", color=color_Sil)

# Rosseland means (dashed)
ax1.loglog(T_C,   κR_C,   "--", lw=3.5, label=r"Graphite, $\kappa_R$", color=color_C, dash_capstyle='round')
ax1.loglog(T_PAH, κR_PAH, "--", lw=3.5, label=r"PAH, $\kappa_R$", color=color_PAH, dash_capstyle='round')
ax1.loglog(T_sil, κR_sil, "--", lw=3.5, label=r"Silicate, $\kappa_R$", color=color_Sil, dash_capstyle='round')

ax1.set_xlabel("T (K)", labelpad = 20)
ax1.set_ylabel("$\kappa_{R}$ and $\kappa_{P}$ (cm$^2$ g$^{-1}$)", labelpad = 20)

ax1.set_xscale("log")
ax1.set_yscale("log")

# T-labels:
x_ticks  = [1.0, 10.0, 100.0, 1000.0]
x_labels = ["1", "10", "100", "1000"]
ax1.xaxis.set_major_locator(FixedLocator(x_ticks))
ax1.xaxis.set_major_formatter(FixedFormatter(x_labels))

# Opacity labels:
y_ticks  = [0.1, 1, 10, 100, 1000, 10000]
y_labels = ["0.1", "1", "10", "10$^{2}$", "10$^{3}$", "10$^{4}$"]
ax1.yaxis.set_major_locator(FixedLocator(y_ticks))
ax1.yaxis.set_major_formatter(FixedFormatter(y_labels))

# Text:
ax1.text(20.0, 3.0, "Silicate dust", fontsize=28, color=color_Sil, rotation=40)
ax1.text(13.0, 40.0, "Graphite dust", fontsize=28, color=color_C, rotation=35)
ax1.text(100.0, 3.0, "PAH dust", fontsize=28, color=color_PAH, rotation=40)
ax1.text(40, 0.4, "Solid lines: Planck mean", fontsize=24, color='dimgray')
ax1.text(40, 0.2, "Dashed lines: Rosseland mean", fontsize=24, color='dimgray')

# Minor ticks on log y-axis:
ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))

# No labels on minor ticks:
ax1.yaxis.set_minor_formatter(NullFormatter())
ax2.yaxis.set_minor_formatter(NullFormatter())

ax1.set_ylim(0.03, 1e4)

###############################################################
#######                                                 #######
#######             R I G H T   P A N E L               #######
#######                                                 #######
###############################################################

# Just check that the temperature grids match for all three dust types (they should):
if not (np.allclose(T_sil, T_C) and np.allclose(T_sil, T_PAH)):
    raise ValueError("Temperature grids in opacity tables do not match.")

# Total dust means
ax2.loglog(T_sil, κP_tot, lw=4.0, color=color_tot)
ax2.loglog(T_sil, κR_tot, "--", lw=4.0, color=color_tot, dash_capstyle='round')

ax2.set_xlabel("T (K)", labelpad = 20)

ax2.set_xscale("log")
ax2.set_yscale("log")

# T-labels:
ax2.xaxis.set_major_locator(FixedLocator(x_ticks))
ax2.xaxis.set_major_formatter(FixedFormatter(x_labels))

# Same y-scale/ticks:
ax2.yaxis.set_major_locator(FixedLocator(y_ticks))
ax2.yaxis.set_major_formatter(FixedFormatter(y_labels))

# Text:
ax2.text(10.0, 0.13, "Total dust opacity", fontsize=30, color=color_tot)
ax2.text(10.0, 0.6, "Solid: Planck mean", fontsize=24, color='dimgray')
ax2.text(10.0, 0.3, "Dashed: Rosseland mean", fontsize=24, color='dimgray')
ax2.text(10.0, 7e2,
         rf"$f_{{Sil}}={f_Sil:.4f}$" "\n"
         rf"$f_{{C}}={f_C:.4f}$" "\n"
         rf"$f_{{PAH}}={f_PAH:.4f}$",
         fontsize=24, color='dimgray')

fig.subplots_adjust(wspace=-0.3) # Space between subplots
plt.tight_layout()
plt.savefig(repo_root / "DustThermalOpacities.pdf", dpi = 300, bbox_inches='tight')
plt.show()