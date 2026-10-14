import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import NullFormatter, FixedLocator, NullLocator
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
cmastro_src = repo_root / "src" / "cmastro" / "src"

sys.path.insert(0, str(cmastro_src))

import cmastro


###############################################################
#######                                                 #######
#######              U S E R   O P T I O N S            #######
#######                                                 #######
###############################################################

filename = "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5"

output_name = "LYDION_FrequencyGrid_VoigtProfile.pdf"

# Temperatures to show
T_panels = [1e4, 1e2]

# Reference temperature used for the frequency-grid construction
T_ref = 30

# Max x-axis value in units of Δν_D(30 K)
xmax_plot = 1e4

# Linear-to-log transition for the symlog x-axis
linthresh = 0.3

# Number of points for the smooth Voigt curves
N_curve = 2500

# Fill bins under the Voigt curve?
fill_bins = True

# Draw interface lines?
draw_interfaces = True

# Draw all interfaces up to this x, then only every nth interface beyond
interface_dense_until = 30.0
interface_stride_wings = 1

# Colormap for bin fills
cmap_name = "cma:emph"

# y-limits
ylim = (1e-10, 5.0)

# Figure title?
show_title = True

# Draw a wing-reference scaling line in one panel?
draw_reference_scaling = True
reference_panel_index = -1
reference_power = -2.0

# Reference line starts at:
# x = reference_xrange[0], y = reference_y_start
reference_xrange = (100.0, 4000.0)
reference_y_start = 1.0e-5

reference_label = r"$\propto x^{-2}$"
reference_label_x = 900.0
reference_label_y_factor = 3.0
reference_label_rotation = -21


###############################################################
#######                                                 #######
#######                 C O N S T A N T S               #######
#######                                                 #######
###############################################################

c     = 2.99792458e10
kB    = 1.380648813e-16
mu    = 1.660539040e-24
mH    = 1.0080 * mu
nuLya = 2.466068e15
ALya  = 6.265e8

Four_pi = 4.0 * np.pi
sqrt_pi_over_2 = np.sqrt(np.pi) / 2.0
inv_sqrt_pi_over_2 = 1.0 / sqrt_pi_over_2


###############################################################
#######                                                 #######
#######                B A S I C   F U N C S            #######
#######                                                 #######
###############################################################

def Compute_b(T):
    return np.sqrt(2.0 * kB * T / mH)


def Compute_DnuD(T):
    return Compute_b(T) * nuLya / c


def Compute_a(T):
    # Same as the Julia code:
    # a(T) = ALya / (4π Δν_D(T))
    return ALya / (Four_pi * Compute_DnuD(T))


###############################################################
#######                                                 #######
#######              D A W S O N   /  H(x,a)            #######
#######                                                 #######
###############################################################

# Build a small Dawson lookup table once, then interpolate.
# This avoids scipy but stays fast enough for plotting.

_DAWSON_XMAX = 8.0
_DAWSON_N    = 8000

_dawson_x = np.linspace(0.0, _DAWSON_XMAX, _DAWSON_N)
_dawson_f = np.exp(_dawson_x * _dawson_x)

_dawson_int = np.zeros_like(_dawson_x)
_dawson_int[1:] = np.cumsum(
    0.5 * (_dawson_f[1:] + _dawson_f[:-1]) * np.diff(_dawson_x)
)

_dawson_lut = np.exp(-_dawson_x * _dawson_x) * _dawson_int


def Dawson_approx(x):
    """
    Dawson integral F(x) = exp(-x^2) ∫_0^x exp(t^2) dt.

    Uses:
    - interpolation on a precomputed LUT for |x| <= _DAWSON_XMAX
    - asymptotic expansion for large |x|
    """
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    ax = np.abs(x)

    F = np.empty_like(ax)

    mask_lut = ax <= _DAWSON_XMAX
    if np.any(mask_lut):
        F[mask_lut] = np.interp(ax[mask_lut], _dawson_x, _dawson_lut)

    mask_asym = ~mask_lut
    if np.any(mask_asym):
        xx = ax[mask_asym]
        F[mask_asym] = (
            1.0 / (2.0 * xx)
            + 1.0 / (4.0 * xx**3)
            + 3.0 / (8.0 * xx**5)
            + 15.0 / (16.0 * xx**7)
        )

    return sign * F


def H_voigt(x, a):
    """
    Same formula as in the Julia code:

    H(x,a) = exp(-x^2)
           + inv_Sqrt_π_over_2 * a * (2*x*Dawson(x) - 1)
           + a^2 * (1 - 2*x^2) * exp(-x^2)
    """
    x = np.asarray(x, dtype=float)

    EXP_xsq = np.exp(-x * x)
    Dawson  = Dawson_approx(x)
    Stuff   = (1.0 - 2.0 * x * x) * EXP_xsq

    return EXP_xsq + inv_sqrt_pi_over_2 * a * (2.0 * x * Dawson - 1.0) + (a * a) * Stuff


###############################################################
#######                                                 #######
#######              L A B E L   H E L P E R S          #######
#######                                                 #######
###############################################################

def sci_label(value, sigfig=2):
    """
    Return a compact mathtext scientific-notation label:
    4.72e-4 -> 4.72 × 10^{-4}
    """
    if value == 0:
        return r"$0$"

    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0**exponent

    mantissa_str = f"{mantissa:.{sigfig}f}".rstrip("0").rstrip(".")

    if exponent == 0:
        return rf"${mantissa_str}$"
    else:
        return rf"${mantissa_str}\times 10^{{{exponent}}}$"


def temperature_label(T):
    """
    Clean temperature labels for the in-panel annotations.
    """
    if np.isclose(T, 1e4):
        return r"$T = 10^4\,K$"
    elif np.isclose(T, 1e3):
        return r"$T = 10^3\,K$"
    elif np.isclose(T, 1e2):
        return r"$T = 100\,K$"
    elif np.isclose(T, 10):
        return r"$T = 10\,K$"
    else:
        return rf"$T = {T:g}\,K$"


def log_tick_label(val):
    if val == 0:
        return "0"
    exp = int(np.round(np.log10(abs(val))))
    if exp == -2:
        return "0.01"
    if exp == -1:
        return "0.1"
    if exp == 0:
        return "1"
    if exp == 1:
        return "10"
    if exp == 2:
        return "100"
    return rf"$10^{{{exp}}}$"


def x_tick_label(val):
    """
    Cleaner x tick labels for the symlog axis.
    """
    if np.isclose(val, 0.0):
        return "0"
    if np.isclose(val, 1.0):
        return "1"
    if np.isclose(val, 3.0):
        return "3"
    if np.isclose(val, 10.0):
        return "10"
    if np.isclose(val, 30.0):
        return "30"
    if np.isclose(val, 100.0):
        return "100"
    if np.isclose(val, 300.0):
        return "300"
    if np.isclose(val, 1000.0):
        return r"$10^3$"
    if np.isclose(val, 3000.0):
        return r"$3\times 10^3$"
    if np.isclose(val, 10000.0):
        return r"$10^4$"

    return f"{val:g}"


###############################################################
#######                                                 #######
#######           H D F 5   R E A D   H E L P E R       #######
#######                                                 #######
###############################################################

def read_dataset(file, candidate_names):
    for name in candidate_names:
        if name in file:
            return file[name][:]
    raise KeyError(f"Could not find any of these datasets: {candidate_names}")


###############################################################
#######                                                 #######
#######           P L O T   C O S M E T I C S           #######
#######                                                 #######
###############################################################

# COMMENT THIS OUT IF YOU DON'T HAVE THE REQUIRED FONTS:

plt.rc('font', **{'sans-serif': ['AVHershey Complex'],
                  'size': 34})

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


###############################################################
#######                                                 #######
#######              R E A D   F R E Q   G R I D        #######
#######                                                 #######
###############################################################

with h5py.File(filename, "r") as f:
    nu_edges   = read_dataset(f, ["ν", "nu"])
    nu_centers = read_dataset(f, ["νC", "nuC"])
    Dnu        = read_dataset(f, ["Δν", "Dnu"])

DnuD_ref = Compute_DnuD(T_ref)

x_edges   = (nu_edges   - nuLya) / DnuD_ref
x_centers = (nu_centers - nuLya) / DnuD_ref

print("Number of frequency bins = ", len(nu_centers))

# Only positive frequency displacement, since the grid is symmetric
positive_edges = x_edges[x_edges >= 0.0]
positive_edges = positive_edges[positive_edges <= xmax_plot]

if len(positive_edges) == 0:
    raise ValueError("No positive frequency interfaces found within xmax_plot.")

if positive_edges[0] > 0.0:
    positive_edges = np.insert(positive_edges, 0, 0.0)

# Dense x-sampling for the smooth curves: linear near zero, logarithmic farther out
x_linear = np.linspace(0.0, linthresh, 500)
x_log    = np.geomspace(linthresh, xmax_plot, N_curve)
x_curve  = np.unique(np.concatenate([x_linear, x_log]))


###############################################################
#######                                                 #######
#######                   P L O T                       #######
#######                                                 #######
###############################################################

fig, axes = plt.subplots(
    len(T_panels), 1,
    sharex=True,
    figsize=(19, 15.0),
    gridspec_kw={"hspace": 0.06}
)

if len(T_panels) == 1:
    axes = [axes]

cmap = plt.colormaps.get_cmap(cmap_name)
norm = Normalize(vmin=np.log10(1.0), vmax=np.log10(1.0 + xmax_plot))

# x-axis ticks — defined here so we can apply FixedLocator inside the loop,
# which prevents symlog from sneaking in its own linthresh tick on every axis.
xticks_all = np.array([0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000], dtype=float)
xticks = xticks_all[xticks_all <= xmax_plot]

for ax_index, (ax, T) in enumerate(zip(axes, T_panels)):

    a = Compute_a(T)
    DnuD_T = Compute_DnuD(T)

    # Convert x = (ν-νLya)/ΔνD(30 K) to the x entering H(x,a) at temperature T
    x_T = x_curve * DnuD_ref / DnuD_T
    H   = H_voigt(x_T, a)

    # Main Voigt-profile curve
    ax.plot(
        x_curve, H,
        color="black",
        linewidth=3.0,
        zorder=5
    )

    # Fill each frequency bin under the curve
    if fill_bins:
        for x0, x1 in zip(positive_edges[:-1], positive_edges[1:]):

            if x1 <= 0.0 or x0 >= xmax_plot:
                continue

            x0p = max(x0, 0.0)
            x1p = min(x1, xmax_plot)

            xx = np.linspace(x0p, x1p, 12)
            xx_T = xx * DnuD_ref / DnuD_T
            HH = H_voigt(xx_T, a)

            xmid = 0.5 * (x0p + x1p)
            color = cmap(norm(np.log10(1.0 + xmid)))

            ax.fill_between(
                xx, HH, ylim[0],
                color=color,
                alpha=0.5,
                linewidth=0.0,
                zorder=1
            )

    # Frequency-interface lines, clipped to the Voigt profile height
    if draw_interfaces:
        for n, xe in enumerate(positive_edges):

            if xe < 0.0 or xe > xmax_plot:
                continue

            if xe <= interface_dense_until:
                draw_this = True
            else:
                draw_this = (n % interface_stride_wings == 0)

            if draw_this:
                # Evaluate Voigt profile at this interface x-position
                xe_T = xe * DnuD_ref / DnuD_T
                H_top = float(H_voigt(np.array([xe_T]), a)[0])
                H_top = max(H_top, ylim[0])   # don't draw below the plot floor

                ax.plot(
                    [xe, xe], [ylim[0], H_top],
                    color="black",
                    linewidth=1.5,
                    alpha=0.4,
                    linestyle="--",
                    dash_capstyle='round',
                    zorder=3
                )

    ax.set_xscale("symlog", linthresh=linthresh, linscale=1.0)
    ax.set_yscale("log")
    ax.set_xlim(0.0, xmax_plot)
    ax.set_ylim(*ylim)

    # Lock x major ticks NOW, immediately after set_xscale, so symlog cannot
    # inject its automatic linthresh tick afterwards.
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.minorticks_on()

    # symlog otherwise tends to add an unlabeled minor tick inside the
    # linear core, usually near linthresh. It looks like an accidental
    # extra tick between 0 and 1, so suppress x-minor ticks explicitly.
    ax.xaxis.set_minor_locator(NullLocator())

    ax.tick_params(axis='both', which='major', direction='in', length=20, width=2.5)
    ax.tick_params(axis='both', which='minor', direction='in', length=10, width=2.5)

    # In-panel labels
    ax.text(
        0.05, 0.8,
        temperature_label(T),
        transform=ax.transAxes,
        fontsize=40,
        color="black"
    )

    ax.text(
        0.05, 0.70,
        rf"$a_v = ${sci_label(a, sigfig=2)}",
        transform=ax.transAxes,
        fontsize=30,
        color="black"
    )

    # Optional asymptotic wing reference. In the far wings,
    # H(x,a_v) ∝ x^{-2}. We draw this only in the selected panel.
    selected_index = reference_panel_index
    if selected_index < 0:
        selected_index = len(axes) + selected_index

    if draw_reference_scaling and ax_index == selected_index:

        x_ref_min, x_ref_max = reference_xrange
        x_ref_min = max(x_ref_min, linthresh)
        x_ref_max = min(x_ref_max, xmax_plot)

        x_ref = np.geomspace(x_ref_min, x_ref_max, 200)

        # Explicit normalization: the reference line starts at
        # (x_ref_min, reference_y_start), then follows x^reference_power.
        H_ref = reference_y_start * (x_ref / x_ref_min)**reference_power

        ax.plot(
            x_ref, H_ref,
            color="black",
            linewidth=2.4,
            linestyle=(0, (8, 5)),
            dash_capstyle='round',
            zorder=6
        )

        y_lab = (
            reference_label_y_factor
            * reference_y_start
            * (reference_label_x / x_ref_min)**reference_power
        )

        ax.text(
            reference_label_x, y_lab,
            reference_label,
            fontsize=35,
            color="black",
            rotation=reference_label_rotation,
            ha="center",
            va="center"
        )


###############################################################
#######                                                 #######
#######                  T I C K S                      #######
#######                                                 #######
###############################################################

# y-axis ticks
yticks_all = np.array([
    1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6,
    1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0
])
yticks = yticks_all[(yticks_all >= ylim[0]) & (yticks_all <= ylim[1])]

for ax in axes:
    ax.set_yticks(yticks)
    ax.set_yticklabels([log_tick_label(y) for y in yticks])
    ax.yaxis.set_minor_formatter(NullFormatter())

# x-axis tick labels only on the bottom panel (axes share x)
axes[-1].set_xticklabels([x_tick_label(x) for x in xticks], fontsize=30)

axes[-1].set_xlabel(
    r"Frequency displacement $|\nu-\nu_{Ly\alpha}|/\Delta\nu_D(30\,K)$",
    fontsize = 40,
    labelpad=15
)

for ax in axes:
    ax.set_ylabel(r"$H(x,a_v)$", labelpad=16, fontsize = 40)

if show_title:
    axes[0].set_title(
        r"Ly$\alpha$ frequency grid & representative Voigt profiles",
        y=1.04,
        fontsize=45
    )

plt.savefig(output_name, dpi=300, bbox_inches='tight')
plt.show()