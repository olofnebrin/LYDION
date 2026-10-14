""" Plot the Lyα emission probability per recombination from
    Storey & Hummer (1995), as used in LYDION.

    Two panels are shown:
      - Case A recombination
      - Case B recombination

    The arrays below were checked against LymanAlphaEmission.jl.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, PercentFormatter
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings("ignore")


###############################################################
#######                                                 #######
#######              U S E R   O P T I O N S            #######
#######                                                 #######
###############################################################

FigureName = "LYDION_Figure_LyaEmissionProbability.pdf"
FigureName_png = "LYDION_Figure_LyaEmissionProbability.png"

# Maximum electron density shown on the x-axis.
# The plot uses log10(ne / cm^-3). Case A data only extend to 1e10 cm^-3;
# by default we use the same flat extrapolation/clamping as the Julia code.
ne_max_plot = 1e10

# Minimum electron density shown on the x-axis.
ne_min_plot = 1e2

# If True, reproduce the Julia interpolation behaviour outside the tabulated
# density range, i.e. Flat() extrapolation by clamping to the nearest edge.
use_julia_flat_extrapolation = True

# Fine plotting grid
N_ne_fine = 450
N_T_fine  = 320

# Probability color scale shared by both panels
prob_vmin = 0.30
prob_vmax = 1.00
cmap_name = "Blues"

# Filled-contour resolution
N_contourf_levels = 300

# Overplotted contour levels
CaseA_contour_levels = [0.35, 0.40, 0.45, 0.50, 0.55]
CaseB_contour_levels = [0.70, 0.80, 0.90, 0.95, 0.99]

# Figure size/style
figsize = (10.5, 15.0)
fontsize_title = 46
fontsize_axis  = 43
fontsize_panel = 48
fontsize_cbar  = 30
fontsize_clabel = 28

# Panel label placement.  Axes coordinates: (0,0) = lower left.
panel_label_x = 0.68
panel_label_y = 0.90
panel_label_stroke_width = 3.4

# Panel-label colours.  Case A sits on a light background, Case B on a dark one.
panel_label_styles = {
    "Case A": dict(color="black", stroke="white"),
    "Case B": dict(color="white", stroke="black"),
}

# Contour styling
contour_color_default = "black"
contour_color_highlight = "white"
contour_highlight_levels = {
    "Case B": [0.99],
}
contour_lw = 2.3
contour_lw_highlight = 2.3
contour_dash_pattern = (6.0, 4.0)

# Colorbar placement.  Make it span the full vertical range of both panels.
cbar_left = 0.885
cbar_bottom = 0.11
cbar_width = 0.035
cbar_height = 0.77

# Figure title
title_y = 0.925

# Optional annotations
show_annotations = True


###############################################################
#######                                                 #######
#######              S T O R E Y  &  H U M M E R        #######
#######                                                 #######
###############################################################

logT_SH95        = np.log10([5e2, 1e3, 3e3, 5e3, 7.5e3, 1e4, 1.25e4, 1.5e4, 2e4, 3e4])
logne_SH95_CaseB = np.log10([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14])
logne_SH95_CaseA = np.log10([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10])

# Case B probabilities. Shape: (N_T, N_ne), matching the Julia interpolation
# interpolate((logT_SH95, logne_SH95_CaseB), Prob_Lyα_SH95_CaseBData, ...).
Prob_Lyα_SH95_CaseBData = np.array([
    [0.80793637, 0.90215197, 0.98157401, 0.9970534,  0.99933853, 0.99934845, 0.99967896, 0.99948547, 0.99931146, 0.99951894, 0.9997498,  0.99864818, 0.99372263],
    [0.78439303, 0.87743841, 0.97423464, 0.99680114, 0.99960243, 0.99967816, 0.9997671,  0.99939133, 0.99922946, 0.99960973, 0.99937386, 0.99906812, 0.9954834],
    [0.74159266, 0.82809599, 0.95734434, 0.99359891, 0.99892162, 0.99927343, 0.99979471, 0.99938909, 0.99899055, 0.99925979, 0.99957955, 0.99923525, 0.99703571],
    [0.71946081, 0.80133228, 0.94581237, 0.99246182, 0.99848958, 0.99940329, 0.99949246, 0.99952048, 0.99981447, 0.99956726, 0.9993515,  0.99952024, 0.99692499],
    [0.70117195, 0.7788453,  0.93441344, 0.99075058, 0.99831301, 0.99960223, 0.99940949, 0.99946884, 0.99948297, 0.99947146, 0.99965311, 0.99918254, 0.99690311],
    [0.68808416, 0.76220739, 0.92503048, 0.9891268,  0.99828523, 0.9994074,  0.99944339, 0.99952601, 0.99932358, 0.99927528, 0.99938987, 0.99904232, 0.9949615],
    [0.6775581,  0.74868826, 0.9176273,  0.98784252, 0.99835238, 0.99903921, 0.99977064, 0.99954296, 0.99930529, 0.99954743, 0.99944997, 0.99874439, 0.9916493],
    [0.66862468, 0.73785038, 0.91033402, 0.98701405, 0.99817046, 0.99892625, 0.99953078, 0.99960556, 0.9993362,  0.99941558, 0.99944365, 0.99828039, 0.98817287],
    [0.65534642, 0.72080463, 0.89823705, 0.98497224, 0.99798854, 0.9988133,  0.99929092, 0.99909416, 0.99933199, 0.9995549,  0.99924446, 0.99764512, 0.97960152],
    [0.63566376, 0.69600394, 0.87952243, 0.98121463, 0.99716593, 0.99896144, 0.99905958, 0.9996252,  0.99953649, 0.99931002, 0.99872497, 0.99561256, 0.96391965]
])

# Case A probabilities. Shape: (N_T, N_ne), matching the Julia interpolation
# interpolate((logT_SH95, logne_SH95_CaseA), Prob_Lyα_SH95_CaseAData, ...).
Prob_Lyα_SH95_CaseAData = np.array([
    [0.58042473, 0.59595911, 0.60335363, 0.58990186, 0.56623927, 0.53426043, 0.492528,   0.44534414, 0.40955195],
    [0.54712958, 0.56465163, 0.57999443, 0.57241623, 0.5541246,  0.5276139,  0.49052298, 0.44606687, 0.4101595],
    [0.48320245, 0.50151448, 0.52818381, 0.53008608, 0.52008701, 0.50298698, 0.47625175, 0.43924427, 0.40696841],
    [0.44933274, 0.46714191, 0.49852558, 0.50475148, 0.49815515, 0.48505756, 0.46319941, 0.43069186, 0.40125666],
    [0.42087234, 0.43810743, 0.47280049, 0.48268151, 0.47821493, 0.46794659, 0.4497736,  0.42097532, 0.39442497],
    [0.40023971, 0.41675887, 0.45373431, 0.46591319, 0.46300415, 0.45454038, 0.43876097, 0.41249415, 0.38810578],
    [0.3840314,  0.40007514, 0.43832241, 0.45257107, 0.45064954, 0.4434932,  0.42939783, 0.40511911, 0.38230403],
    [0.37046215, 0.38617473, 0.42546609, 0.44143608, 0.44049428, 0.43394168, 0.42127472, 0.39854551, 0.37684759],
    [0.34937867, 0.36397106, 0.40484801, 0.42349201, 0.42367822, 0.41860328, 0.40761256, 0.38699493, 0.36739625],
    [0.31960139, 0.33302781, 0.37528203, 0.39805701, 0.39984039, 0.39602958, 0.38718651, 0.36963656, 0.3518518]
])


###############################################################
#######                                                 #######
#######            S A N I T Y   C H E C K S            #######
#######                                                 #######
###############################################################

if Prob_Lyα_SH95_CaseBData.shape != (len(logT_SH95), len(logne_SH95_CaseB)):
    raise RuntimeError(
        f"Bad Case B table shape: {Prob_Lyα_SH95_CaseBData.shape}; "
        f"expected {(len(logT_SH95), len(logne_SH95_CaseB))}."
    )

if Prob_Lyα_SH95_CaseAData.shape != (len(logT_SH95), len(logne_SH95_CaseA)):
    raise RuntimeError(
        f"Bad Case A table shape: {Prob_Lyα_SH95_CaseAData.shape}; "
        f"expected {(len(logT_SH95), len(logne_SH95_CaseA))}."
    )


###############################################################
#######                                                 #######
#######           P L O T   C O S M E T I C S           #######
#######                                                 #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 38})

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

plt.rcParams['axes.linewidth'] = 3.2
plt.rcParams['grid.linewidth'] = 2.0
plt.rcParams['grid.linestyle'] = '--'

plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


###############################################################
#######                                                 #######
#######              H E L P E R S                      #######
#######                                                 #######
###############################################################

def probability_interpolated(logne_grid, logT_grid, P_grid, Xf, Yf):
    """
    Bilinear interpolation of P(logT, logne) onto the plotting grid.

    This is a small NumPy-only replacement for scipy.interpolate.

    If use_julia_flat_extrapolation=True, query points outside the table
    are clamped to the nearest table edge, matching the Julia Flat()
    extrapolation used in LymanAlphaEmission.jl.
    """

    logne_grid = np.asarray(logne_grid, dtype=float)
    logT_grid  = np.asarray(logT_grid, dtype=float)
    P_grid     = np.asarray(P_grid, dtype=float)

    if use_julia_flat_extrapolation:
        Xq = np.clip(Xf, logne_grid[0], logne_grid[-1])
        Yq = np.clip(Yf, logT_grid[0],  logT_grid[-1])
    else:
        Xq = Xf.copy()
        Yq = Yf.copy()

        outside = (
            (Xq < logne_grid[0]) | (Xq > logne_grid[-1]) |
            (Yq < logT_grid[0])  | (Yq > logT_grid[-1])
        )

        Xq = np.clip(Xq, logne_grid[0], logne_grid[-1])
        Yq = np.clip(Yq, logT_grid[0],  logT_grid[-1])

    # Lower cell indices in each dimension.
    # P_grid has shape (N_T, N_ne).
    iT = np.searchsorted(logT_grid,  Yq, side="right") - 1
    inE = np.searchsorted(logne_grid, Xq, side="right") - 1

    iT  = np.clip(iT,  0, len(logT_grid)  - 2)
    inE = np.clip(inE, 0, len(logne_grid) - 2)

    T0 = logT_grid[iT]
    T1 = logT_grid[iT + 1]
    n0 = logne_grid[inE]
    n1 = logne_grid[inE + 1]

    wt = (Yq - T0) / (T1 - T0)
    wn = (Xq - n0) / (n1 - n0)

    P00 = P_grid[iT,     inE]
    P01 = P_grid[iT,     inE + 1]
    P10 = P_grid[iT + 1, inE]
    P11 = P_grid[iT + 1, inE + 1]

    Pf = (
        (1.0 - wt) * (1.0 - wn) * P00 +
        (1.0 - wt) * wn         * P01 +
        wt         * (1.0 - wn) * P10 +
        wt         * wn         * P11
    )

    if not use_julia_flat_extrapolation:
        Pf[outside] = np.nan

    return Pf


def pct_label(v):
    if v >= 0.99:
        return f"{100*v:.0f}%"
    if v >= 0.9:
        return f"{100*v:.0f}%"
    return f"{100*v:.0f}%"


def add_panel_label(ax, label):
    style = panel_label_styles.get(label, dict(color="black", stroke="white"))

    ax.text(
        panel_label_x,
        panel_label_y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize_panel,
        color=style["color"],
        path_effects=[
            pe.withStroke(
                linewidth=panel_label_stroke_width,
                foreground=style["stroke"],
            )
        ],
        zorder=20,
    )


def stylize_contours(CS):
    """Make dashed contour lines use rounded caps/joins when supported.

    Do not call set_linestyle() here.  Some Matplotlib versions expand a
    custom dash tuple once per contour level, and then set_linestyle() sees
    a malformed list like [0, (6,4), 0, (6,4), ...].  We therefore pass
    linestyles="dashed" to ax.contour() and only adjust cap/join style here.
    """

    artists = []

    if hasattr(CS, "collections"):
        artists.extend(CS.collections)
    else:
        artists.append(CS)

    for artist in artists:
        if hasattr(artist, "set_capstyle"):
            artist.set_capstyle("round")
        if hasattr(artist, "set_dash_capstyle"):
            artist.set_dash_capstyle("round")
        if hasattr(artist, "set_joinstyle"):
            artist.set_joinstyle("round")
        if hasattr(artist, "set_dash_joinstyle"):
            artist.set_dash_joinstyle("round")


def draw_contours_with_optional_highlight(ax, Xf, Yf, Pf, clevels, panel_label):
    """Draw dashed contours, with selected levels highlighted in white."""

    highlight = contour_highlight_levels.get(panel_label, [])
    highlight = [float(v) for v in highlight]

    default_levels = [v for v in clevels if not any(np.isclose(v, h) for h in highlight)]
    contour_sets = []

    if len(default_levels) > 0:
        CS_default = ax.contour(
            Xf,
            Yf,
            Pf,
            levels=default_levels,
            colors=contour_color_default,
            linewidths=contour_lw,
            linestyles="dashed"
        )
        stylize_contours(CS_default)
        contour_sets.append((CS_default, contour_color_default, "white"))

    if len(highlight) > 0:
        CS_high = ax.contour(
            Xf,
            Yf,
            Pf,
            levels=highlight,
            colors=contour_color_highlight,
            linewidths=contour_lw_highlight,
            linestyles="dashed"
        )
        stylize_contours(CS_high)
        contour_sets.append((CS_high, contour_color_highlight, "black"))

    return contour_sets


###############################################################
#######                                                 #######
#######                  P L O T                        #######
#######                                                 #######
###############################################################

logne_min_plot = np.log10(ne_min_plot)
logne_max_plot = np.log10(ne_max_plot)

if logne_max_plot <= logne_min_plot:
    raise ValueError("ne_max_plot must be larger than ne_min_plot.")

logne_fine = np.linspace(logne_min_plot, logne_max_plot, N_ne_fine)
logT_fine  = np.linspace(logT_SH95.min(), logT_SH95.max(), N_T_fine)
Xf, Yf = np.meshgrid(logne_fine, logT_fine)

Pf_A = probability_interpolated(logne_SH95_CaseA, logT_SH95, Prob_Lyα_SH95_CaseAData, Xf, Yf)
Pf_B = probability_interpolated(logne_SH95_CaseB, logT_SH95, Prob_Lyα_SH95_CaseBData, Xf, Yf)

fig, axes = plt.subplots(
    2, 1,
    figsize=figsize,
    sharex=True,
    gridspec_kw={"hspace": 0.06, "right": 0.86}
)

cmap = plt.colormaps.get_cmap(cmap_name)
norm = Normalize(vmin=prob_vmin, vmax=prob_vmax)
levels_filled = np.linspace(prob_vmin, prob_vmax, N_contourf_levels)

panel_data = [
    (axes[0], Pf_A, CaseA_contour_levels, r"Case A"),
    (axes[1], Pf_B, CaseB_contour_levels, r"Case B"),
]

last_cf = None

for ax, Pf, clevels, label in panel_data:

    cf = ax.contourf(
        Xf,
        Yf,
        Pf,
        levels=levels_filled,
        cmap=cmap,
        norm=norm,
        extend="neither"
    )
    last_cf = cf

    contour_sets = draw_contours_with_optional_highlight(ax, Xf, Yf, Pf, clevels, label)

    for CS, text_color, stroke_color in contour_sets:
        labels = ax.clabel(
            CS,
            fmt=lambda v: pct_label(v),
            inline=True,
            inline_spacing=45,
            fontsize=fontsize_clabel,
            colors=text_color,
        )
        # No stroke on contour labels; keep the labels clean.

    add_panel_label(ax, label)

    ax.set_ylabel("log$_{10}$(T/K)", labelpad=18, fontsize=fontsize_axis)
    ax.set_ylim(logT_SH95.min(), logT_SH95.max())
    ax.set_xlim(logne_min_plot, logne_max_plot)

    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in', length=20, width=2.5)
    ax.tick_params(axis='both', which='minor', direction='in', length=10, width=2.5)

    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    # Matplotlib >= 3.10 no longer exposes cf.collections for QuadContourSet.
    # set_rasterized works directly on the contour set and is backward compatible.
    cf.set_rasterized(True)

axes[-1].set_xlabel("log$_{10}$(n$_{e}$/cm$^{-3}$)", labelpad=20, fontsize=fontsize_axis)

# Explicit integer x-ticks across the requested density range.
x_tick_min = int(np.ceil(logne_min_plot))
x_tick_max = int(np.floor(logne_max_plot))
xticks = np.arange(x_tick_min, x_tick_max + 1, 1)
axes[-1].set_xticks(xticks)
axes[-1].set_xticklabels([str(x) for x in xticks])
axes[-1].xaxis.set_minor_locator(AutoMinorLocator(5))

# One shared colorbar, so the color scale is directly comparable.
cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
cbar_ticks = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
cbar = fig.colorbar(last_cf, cax=cax, ticks=cbar_ticks)
cbar.ax.tick_params(direction='in', length=18, width=2.5, labelsize=fontsize_cbar)
cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
#cbar.set_label(r"Ly$\alpha$ emission probability", fontsize=fontsize_axis, labelpad=20)

if show_annotations:

    axes[1].text(
        0.38, 0.20,
        r"$\approx$ all Case B"  + "\n" + "recombinations" + "\n" + r"produce a Ly$\alpha$ photon",
        transform=axes[1].transAxes,
        fontsize=33,
        ha="left",
        va="bottom",
        color="white",
        path_effects=[pe.withStroke(linewidth=3.2, foreground="black")],
    )

fig.suptitle(r"Ly$\alpha$ emission probability", y=title_y, fontsize=fontsize_title)
plt.tight_layout(rect=[0, 0, 1, title_y - 0.02])
plt.savefig(FigureName, bbox_inches='tight', dpi=300)
plt.savefig(FigureName_png, bbox_inches='tight', dpi=300)
plt.show()
