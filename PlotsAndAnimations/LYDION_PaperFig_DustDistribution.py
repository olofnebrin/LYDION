import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator, FixedFormatter
from pathlib import Path
from math import erf
import os


# ---------------- User inputs ----------------

repo_root = Path(__file__).resolve().parents[1]
os.chdir(repo_root)

savefig = True
outfile = repo_root / "LYDION_Figure_DustDistribution.pdf"

N_a = 2000

# Axis limits:
xmin = 1.0e-4     # μm
xmax = 20.0       # μm

# If None, chosen automatically:
ylim_sil = 0.1, 200.0
ylim_C   = 0.1, 200.0

# When choosing automatic y-limits, ignore tiny exponential tails below
# this fraction of the maximum value in each panel:
ylim_tail_fraction = 1.0e-4

# Fill options:
fill_alpha = 0.35
fill_PAH   = True
fill_C     = True
fill_Sil   = True

# Colors:
color_Sil = "orangered"
color_PAH = "dodgerblue"
color_C   = "crimson"

# Line widths:
lw_dist  = 2.8
lw_split = 2.2

# Font sizes:
fontsize_label  = 27
fontsize_tick   = 24
fontsize_panel  = 27

# Figure size:
figsize = (8.5, 9.5)

# ----- Custom tick labels -----
# Set any of these to None to use automatic log ticks instead.

x_tick_positions = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
x_tick_labels    = [r"$10^{-3}$", r"$0.01$", r"$0.1$", r"$1$", r"$10$"]

y_tick_positions_sil = [0.1, 1, 10, 100]
y_tick_labels_sil    = [r"$0.1$", r"$1$", r"$10$", r"$100$"]

y_tick_positions_C = [0.1, 1, 10, 100]
y_tick_labels_C    = [r"$0.1$", r"$1$", r"$10$", r"$100$"]


# ---------------- Constants ----------------
# cgs units, matching the Julia implementation convention.

Ang = 1.0e-8      # cm
μm  = 1.0e-4      # cm

amu = 1.66053906660e-24
mC  = 12.0 * amu

# Material densities of individual grain species (g cm⁻³).
# All carbonaceous grains (graphite & PAH) share the same bulk density:
ρgr_Sil = 3.5
ρgr_C   = 2.24

# Grain size integration limits for each species.
# Carbonaceous population is split into PAH and graphite at a_split_C:
a_min_PAH = 3.548 * Ang
a_split_C = 1.0e-3 * μm
a_min_Sil = 1.0e-3 * μm
a_max     = 10.0 * μm

# b_C = 3 × 10⁻⁵, R_V = 5.5, model B:
bC   = 3.0e-5
a_01 = 3.5 * Ang
a_02 = 30.0 * Ang
σ    = 0.4

b_C1 = 0.75 * bC
b_C2 = 0.25 * bC

# Parameters for carbonaceous grain size distribution:
a_tg = 0.0120 * μm
a_cg = 7.28   * μm
αg   = -1.90
βg   = -0.0517
Cg   = 2.86e-12

# Parameters for silicate grain size distribution:
a_ts = 0.211 * μm
a_cs = 0.1   * μm
αs   = -1.13
βs   = -0.109
Cs   = 1.04e-13


# ---------------- Grain size distribution ----------------

# Amplitudes B1, B2 of the two log-normal VSG/PAH populations
# in the carbonaceous size distribution (WD01 eqs. 2-3):
B1 = ((3.0 / (2.0*np.pi)**1.5) * np.exp(-4.5 * σ**2) / (ρgr_C * a_01**3 * σ)
      * b_C1 * mC
      / (1.0 + erf(3.0*σ/np.sqrt(2.0) + np.log(a_01 / (3.5 * Ang))/(σ*np.sqrt(2.0)))))

B2 = ((3.0 / (2.0*np.pi)**1.5) * np.exp(-4.5 * σ**2) / (ρgr_C * a_02**3 * σ)
      * b_C2 * mC
      / (1.0 + erf(3.0*σ/np.sqrt(2.0) + np.log(a_02 / (3.5 * Ang))/(σ*np.sqrt(2.0)))))


def D(a):
    """
    Eqs. (2)-(3) in Weingartner & Draine (2001).
    Sum of two log-normal populations representing VSGs/PAHs.
    """
    a = np.asarray(a)
    out = np.zeros_like(a, dtype=float)

    mask = a > 3.5 * Ang
    out[mask] = ((B1 / a[mask]) * np.exp(-0.5 * (np.log(a[mask] / a_01) / σ)**2)
               + (B2 / a[mask]) * np.exp(-0.5 * (np.log(a[mask] / a_02) / σ)**2))

    return out


def F(a, β, a_t):
    """
    Eq. (6) in Weingartner & Draine (2001).
    Curvature correction factor for the power-law part of the size distribution.
    """
    a = np.asarray(a)

    if β >= 0.0:
        return 1.0 + β * a / a_t
    else:
        return 1.0 / (1.0 - β * a / a_t)


def dngr_da_C(a):
    """
    Carbonaceous (graphite + PAH) grain size distribution dn_gr/da,
    in units of cm⁻⁴ per H nucleon.
    """
    a   = np.asarray(a)
    out = np.zeros_like(a, dtype=float)

    mask = (a > a_min_PAH) & (a < a_max)

    f = np.ones_like(a, dtype=float)
    cutoff = a >= a_tg
    f[cutoff] = np.exp(-((a[cutoff] - a_tg) / a_cg)**3)

    out[mask] = (D(a)[mask] + (Cg / a[mask]) * (a[mask] / a_tg)**αg
                 * F(a[mask], βg, a_tg) * f[mask])

    return out


def dngr_da_Sil(a):
    """
    Silicate grain size distribution dn_gr/da,
    in units of cm⁻⁴ per H nucleon.
    """
    a = np.asarray(a)
    out = np.zeros_like(a, dtype=float)

    mask = (a > a_min_Sil) & (a < a_max)

    f = np.ones_like(a, dtype=float)
    cutoff = a >= a_ts
    f[cutoff] = np.exp(-((a[cutoff] - a_ts) / a_cs)**3)

    out[mask] = ((Cs / a[mask]) * (a[mask] / a_ts)**αs
                 * F(a[mask], βs, a_ts) * f[mask])

    return out


def plot_quantity_WD01(a, dn_da):
    """
    WD01-style plotted quantity:
        10^29 a^4 (1/nH) dn/da

    Here dn_da is already per H nucleon, so no explicit nH appears.
    a must be in cm.
    """
    return 1.0e29 * a**4 * dn_da


def ylabel_WD01():
    return "$10^{29}$ a$^4$ n$_{H}^{-1}$ $\partial$n$_{gr}$/$\partial$a"


def set_log_ticks(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.tick_params(axis="both", which="major", labelsize=fontsize_tick,
                   direction="in")
    ax.tick_params(axis="both", which="minor",
                   direction="in")


def apply_custom_ticks(ax, xticks=None, xlabels=None, yticks=None, ylabels=None):
    if xticks is not None:
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        if xlabels is not None:
            ax.xaxis.set_major_formatter(FixedFormatter(xlabels))

    if yticks is not None:
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        if ylabels is not None:
            ax.yaxis.set_major_formatter(FixedFormatter(ylabels))


def auto_ylim(y, mask):
    vals = y[mask]
    vals = vals[np.isfinite(vals) & (vals > 0.0)]

    ymax_raw = vals.max()

    # Ignore negligible exponential tails, which otherwise create enormous
    # log-ranges and hide the interesting part of the distribution.
    vals = vals[vals > ylim_tail_fraction * ymax_raw]

    ymin = 10.0**np.floor(np.log10(vals.min()))
    ymax = 10.0**np.ceil(np.log10(ymax_raw))

    return ymin, ymax


def area_weighted_grain_size(a, dn_da, mask):
    """
    Area-weighted grain size:

        ā = ∫ a πa² (dn/da) da / ∫ πa² (dn/da) da
           = ∫ a³ (dn/da) da / ∫ a² (dn/da) da

    a must be in cm.
    """
    aa = a[mask]
    yy = dn_da[mask]

    numerator   = np.trapz(aa**3 * yy, aa)
    denominator = np.trapz(aa**2 * yy, aa)

    return numerator / denominator


def add_area_weighted_line(ax, a_bar_cm, y_bar, ylim, color, label,
                           x_text_factor=1.10, y_text_factor=1.0):
    """
    Add a vertical dashed line at a_bar, stopping at the grain-size
    distribution curve, plus a rotated label.
    """
    a_bar_μm = a_bar_cm / μm

    ax.plot([a_bar_μm, a_bar_μm], [ylim[0], y_bar],
            color=color, lw=lw_split, ls="--",
            dash_capstyle="round", solid_capstyle="round", alpha=0.95)

    y_text = np.sqrt(ylim[0] * y_bar) * y_text_factor

    ax.text(a_bar_μm * x_text_factor, y_text, label,
            fontsize=fontsize_panel,
            color=color,
            rotation=90,
            ha="left", va="center")


# ---------------- Compute distributions ----------------

a_μm = np.logspace(np.log10(xmin), np.log10(xmax), N_a)
a_cm = a_μm * μm

dn_da_Sil = dngr_da_Sil(a_cm)
dn_da_C   = dngr_da_C(a_cm)

y_Sil = plot_quantity_WD01(a_cm, dn_da_Sil)
y_C   = plot_quantity_WD01(a_cm, dn_da_C)

mask_Sil = (a_cm >= a_min_Sil) & (a_cm <= a_max)
mask_PAH = (a_cm >= a_min_PAH) & (a_cm <  a_split_C)
mask_Gra = (a_cm >= a_split_C) & (a_cm <= a_max)
mask_C   = mask_PAH | mask_Gra

if ylim_sil is None:
    ylim_sil = auto_ylim(y_Sil, mask_Sil)

if ylim_C is None:
    ylim_C = auto_ylim(y_C, mask_C)

# Value of the carbonaceous distribution at the PAH/graphite split,
# used for the vertical dashed line:
y_split = plot_quantity_WD01(a_split_C, dngr_da_C(np.array([a_split_C])))[0]

# Area-weighted grain sizes:
a_bar_Sil = area_weighted_grain_size(a_cm, dn_da_Sil, mask_Sil)
a_bar_PAH = area_weighted_grain_size(a_cm, dn_da_C,   mask_PAH)
a_bar_Gra = area_weighted_grain_size(a_cm, dn_da_C,   mask_Gra)

# Distribution values at the area-weighted grain sizes:
y_bar_Sil = plot_quantity_WD01(a_bar_Sil, dngr_da_Sil(np.array([a_bar_Sil])))[0]
y_bar_PAH = plot_quantity_WD01(a_bar_PAH, dngr_da_C(np.array([a_bar_PAH])))[0]
y_bar_Gra = plot_quantity_WD01(a_bar_Gra, dngr_da_C(np.array([a_bar_Gra])))[0]

print(f"Area-weighted silicate grain size: {a_bar_Sil / μm:.4e} μm")
print(f"Area-weighted PAH grain size:      {a_bar_PAH / μm:.4e} μm")
print(f"Area-weighted graphite grain size: {a_bar_Gra / μm:.4e} μm")


###############################################################
#######                                                 #######
#######           P L O T   C O S M E T I C S           #######
#######                                                 #######
###############################################################

# COMMENT THIS OUT IF YOU DON'T HAVE THE REQUIRED FONTS:

#plt.rc('font',**{'sans-serif':['RomanD'],
#                'size': 27} )
#plt.rc('font',**{'sans-serif':['AVHershey Simplex'],
#                'size': 27} )
plt.rc('font',**{'sans-serif':['AVHershey Complex'],
                'size': 37} )
#plt.rc('font',**{'sans-serif':['Complex'],
#                 'size': 27} )

params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

# The stuff below need not be removed:

params = {'legend.fontsize': 19 }
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

plt.rcParams['legend.frameon'] = 'False'

plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linewidth'] = 2.5
plt.rcParams['grid.linestyle'] = '--'

# Make ticks appear on all sides and point inwards
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True

###############################################################
#######                                                 #######
#######               P L O T  I T !                    #######
#######                                                 #######
###############################################################


# ---------------- Plot ----------------

fig, (ax_Sil, ax_C) = plt.subplots(
    2, 1,
    figsize=figsize,
    sharex=True,
    gridspec_kw={"hspace": 0.05}
)

# --- Top panel: silicate ---
ax_Sil.plot(a_μm[mask_Sil], y_Sil[mask_Sil],
            color=color_Sil, lw=lw_dist)

if fill_Sil:
    ax_Sil.fill_between(a_μm[mask_Sil], ylim_sil[0], y_Sil[mask_Sil],
                        color=color_Sil, alpha=fill_alpha, lw=0.0)

add_area_weighted_line(ax_Sil, a_bar_Sil, y_bar_Sil, ylim_sil,
                       color_Sil, r"$\bar{a}_{gr,Sil}$",
                       x_text_factor=0.50, y_text_factor=1.0)

ax_Sil.text(0.1, 0.88, "Silicate dust",
            transform=ax_Sil.transAxes,
            fontsize=fontsize_panel,
            ha="left", va="top")

# --- Bottom panel: carbonaceous ---
ax_C.plot(a_μm[mask_PAH], y_C[mask_PAH],
          color=color_PAH, lw=lw_dist)

ax_C.plot(a_μm[mask_Gra], y_C[mask_Gra],
          color=color_C, lw=lw_dist)

if fill_PAH:
    ax_C.fill_between(a_μm[mask_PAH], ylim_C[0], y_C[mask_PAH],
                      color=color_PAH, alpha=fill_alpha, lw=0.0)

if fill_C:
    ax_C.fill_between(a_μm[mask_Gra], ylim_C[0], y_C[mask_Gra],
                      color=color_C, alpha=fill_alpha, lw=0.0)

add_area_weighted_line(ax_C, a_bar_PAH, y_bar_PAH, ylim_C,
                       color_PAH, r"$\bar{a}_{gr,PAH}$",
                       x_text_factor=0.5, y_text_factor=1.0)

add_area_weighted_line(ax_C, a_bar_Gra, y_bar_Gra, ylim_C,
                       color_C, r"$\bar{a}_{gr,C}$",
                       x_text_factor=0.5, y_text_factor=1.0)

# Vertical dashed line only up to the distribution value at a_split_C:
ax_C.plot([a_split_C / μm, a_split_C / μm], [ylim_C[0], y_split],
          color="black", lw=lw_split, ls="--",
          dash_capstyle="round", solid_capstyle="round", alpha=0.9)

ax_C.text(0.1, 0.88, "Carbonaceous dust",
          transform=ax_C.transAxes,
          fontsize=fontsize_panel,
          ha="left", va="top")

ax_C.text(0.08, 0.6, "PAHs",
          transform=ax_C.transAxes,
          fontsize=fontsize_panel,
          color=color_PAH,
          ha="left", va="bottom")

ax_C.text(0.65, 0.30, "Graphite",
          transform=ax_C.transAxes,
          fontsize=fontsize_panel,
          color=color_C,
          ha="left", va="bottom")

# --- Cosmetics ---
for ax in (ax_Sil, ax_C):
    set_log_ticks(ax)
    ax.set_xlim(xmin, xmax)

    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

# Apply custom tick labels if desired:
apply_custom_ticks(ax_Sil,
                   xticks=x_tick_positions,
                   xlabels=x_tick_labels,
                   yticks=y_tick_positions_sil,
                   ylabels=y_tick_labels_sil)

apply_custom_ticks(ax_C,
                   xticks=x_tick_positions,
                   xlabels=x_tick_labels,
                   yticks=y_tick_positions_C,
                   ylabels=y_tick_labels_C)

ax_Sil.set_ylim(*ylim_sil)
ax_C.set_ylim(*ylim_C)

ax_Sil.set_ylabel(ylabel_WD01(), fontsize=fontsize_label)
ax_C.set_ylabel(ylabel_WD01(), fontsize=fontsize_label)
ax_C.set_xlabel("Grain radius a$_{gr}$ ($\\mu$m)", fontsize=fontsize_label)

plt.tight_layout(pad=0.4)

if savefig:
    fig.savefig(outfile, bbox_inches="tight", dpi=600)
    print(f"Saved figure to: {outfile}")

plt.show()