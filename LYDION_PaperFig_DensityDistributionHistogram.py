import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
cmastro_src = repo_root / "src" / "cmastro" / "src"

sys.path.insert(0, str(cmastro_src))

try:
    import cmastro
except Exception:
    pass


# ---------------- User inputs ----------------

#Filename       = "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5"
Filename       = "RHDSim_Lya_u4_256x256_2026-05-02.h5"
FigureName     = "LYDION_Figure_MassWeightedDensityPDF.pdf"
TargetTime_kyr = 3.0

# Histogram variable:
#   "nH"  -> plot log10(nH / cm^-3)
#   "rho" -> plot log10(rho / g cm^-3)
DensityVariable = "nH"

# Number of bins in log10 density
Nbins = 40

# Manual x-range. Set to None for automatic percentile-based range.
log_density_min = 0
log_density_max = 9

# If automatic range is used, these percentiles are applied to selected cells.
auto_range_percentiles = (0.1, 99.9)

# Mass weighting:
#   True  -> histogram integrates to 1 in mass fraction.
#   False -> histogram integrates to 1 in cell/volume fraction if UseVolumeWeights=True,
#            otherwise number fraction.
MassWeighted = True

# If MassWeighted=False, this controls whether the histogram is volume-weighted
# or simply counts cells.
UseVolumeWeights = True

# Cuts for neutral dense gas
ApplyNeutralCut = False
xHI_min         = 0.0

ApplyTemperatureCut = True
T_max_K             = 1000.0

# Additional density cuts, applied before histogram.
# Values are in physical units of the chosen DensityVariable:
#   nH  -> cm^-3
#   rho -> g cm^-3
density_min_cut = None
density_max_cut = None

# Plot both filtered and all gas for comparison.
CompareWithAllGas = True

# Lognormal fit to the plotted PDF.
ShowLognormalFit = True

# If True, the fit is made to the filtered distribution only.
# If False and CompareWithAllGas=True, it also fits all gas.
FitFilteredOnly = True

# Histogram style
hist_lw_filtered = 4.0
hist_lw_all      = 3.0
fit_lw           = 3.0

hist_color_filtered = "black"
hist_color_all      = "0.55"
fit_color_filtered  = "crimson"
fit_color_all       = "royalblue"

hist_alpha_filtered = 1.0
hist_alpha_all      = 0.9

# Figure/style options
figsize = (11.5, 8.5)
dpi     = 600

fontsize_title  = 34
fontsize_label  = 34
fontsize_ticks  = 28
fontsize_legend = 24
fontsize_text   = 24

ShowTitle    = True
ShowInfoText = True

# Text box position in axis coordinates
info_x = 0.04
info_y = 0.96

# Useful constants
mH  = 1.67e-24
X   = 0.76
yr  = 3.1556926e7
kyr = 1e3 * yr
pc  = 3.085677581e18


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

params = {'legend.fontsize': fontsize_legend}
plt.rcParams.update(params)
plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 1.7
plt.rcParams['xtick.minor.width'] = 1.7
plt.rcParams['xtick.major.size'] = 14
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size'] = 14
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['xtick.major.pad'] = '12'
plt.rcParams['ytick.major.pad'] = '12'

plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linewidth'] = 2.0
plt.rcParams['grid.linestyle'] = '--'

plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def reorder_history_3d(arr, t, r_edges, Z_edges):
    """Reorder a 3D history array to shape (Nt, Nr, Nz).

    Supports either:
        (Nt, Nr, Nz), (Nr, Nz, Nt), etc.

    Returns:
        arr_hist with shape (Nt, Nr, Nz)
    """
    if arr is None:
        return None

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    Nt = t.size
    Nr = len(r_edges) - 1
    Nz = len(Z_edges) - 1

    shape = arr.shape
    time_axes = [ax for ax, s in enumerate(shape) if s == Nt]

    if len(time_axes) != 1:
        raise RuntimeError(
            f"Could not uniquely identify time axis in array with shape {shape} and Nt = {Nt}"
        )

    time_axis = time_axes[0]
    arr_tfirst = np.moveaxis(arr, time_axis, 0)
    _, a1, a2 = arr_tfirst.shape

    if (a1, a2) == (Nr, Nz):
        return arr_tfirst
    elif (a1, a2) == (Nz, Nr):
        return arr_tfirst.transpose(0, 2, 1)
    else:
        raise RuntimeError(
            f"After moving time axis, spatial dims {(a1, a2)} do not match Nr = {Nr}, Nz = {Nz}"
        )


def nearest_saved_index(t, target_time):
    saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]

    if saved_idx.size == 0:
        raise RuntimeError("No saved snapshots found in file.")

    best = saved_idx[np.argmin(np.abs(t[saved_idx] - target_time))]
    return best


def read_dataset_if_exists(f, name, default=None):
    return f[name][:] if name in f else default


def read_cell_volumes(f, r_edges, Z_edges):
    """Read saved LYDION cell volumes.

    CellVolumes should be saved directly from the Julia geometry.ΔV array.
    This is the correct volume weight for mass-weighted statistics:

        cell_mass = rho * CellVolumes

    Expected shape is (Nr, Nz), with fallback for accidentally transposed
    storage as (Nz, Nr).
    """
    if "CellVolumes" not in f:
        raise RuntimeError("CellVolumes not found in file. Cannot do proper mass weighting.")

    Nr = len(r_edges) - 1
    Nz = len(Z_edges) - 1

    V = f["CellVolumes"][:]

    if V.shape == (Nr, Nz):
        return V
    elif V.shape == (Nz, Nr):
        return V.T
    else:
        raise RuntimeError(
            f"CellVolumes has shape {V.shape}, expected {(Nr, Nz)} or {(Nz, Nr)}"
        )


def weighted_mean_and_std(x, w):
    good = np.isfinite(x) & np.isfinite(w) & (w > 0.0)

    if np.count_nonzero(good) < 2:
        return np.nan, np.nan

    xg = x[good]
    wg = w[good]

    W = np.sum(wg)
    mu = np.sum(wg * xg) / W
    var = np.sum(wg * (xg - mu)**2) / W

    return mu, np.sqrt(max(var, 0.0))


def make_weighted_pdf(log_density, weights, bin_edges):
    """Return PDF per dex and fraction per bin.

    The returned PDF satisfies:

        sum(pdf * dlog_density) = 1

    if the total weight is positive.
    """
    good = np.isfinite(log_density) & np.isfinite(weights) & (weights > 0.0)

    x = log_density[good]
    w = weights[good]

    if x.size == 0:
        raise RuntimeError("No valid cells left after cuts.")

    hist, edges = np.histogram(x, bins=bin_edges, weights=w)
    widths = np.diff(edges)

    total = np.sum(hist)

    if total <= 0.0:
        raise RuntimeError("Total histogram weight is zero after cuts.")

    frac = hist / total
    pdf = frac / widths
    centers = 0.5 * (edges[:-1] + edges[1:])

    return centers, pdf, frac, edges


def gaussian_pdf(x, mu, sigma):
    """Gaussian PDF in x = log10(density)."""
    if (not np.isfinite(mu)) or (not np.isfinite(sigma)) or sigma <= 0.0:
        return np.full_like(x, np.nan)

    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (np.sqrt(2.0 * np.pi) * sigma)


def make_selection_mask(density, rho, Tgas, xHI):
    """Build mask for the filtered gas sample."""
    mask = np.isfinite(density) & (density > 0.0)
    mask &= np.isfinite(rho) & (rho > 0.0)

    if ApplyNeutralCut:
        mask &= np.isfinite(xHI) & (xHI > xHI_min)

    if ApplyTemperatureCut:
        mask &= np.isfinite(Tgas) & (Tgas < T_max_K)

    if density_min_cut is not None:
        mask &= density > density_min_cut

    if density_max_cut is not None:
        mask &= density < density_max_cut

    return mask


# -------------------------------------------------------------------
# Read data
# -------------------------------------------------------------------

with h5py.File(Filename, "r") as f:
    t = f["Times"][:]
    r = f["r"][:]
    Z = f["Z"][:]

    rho_raw  = f["GasDensity_History"][:]
    T_raw    = read_dataset_if_exists(f, "T_History")
    xHII_raw = read_dataset_if_exists(f, "xHII_History")
    xH2_raw  = read_dataset_if_exists(f, "xH2_History")

    CellVolumes = read_cell_volumes(f, r, Z)

if T_raw is None:
    raise RuntimeError("T_History not found in file.")

if xHII_raw is None:
    raise RuntimeError("xHII_History not found in file.")

if xH2_raw is None:
    raise RuntimeError("xH2_History not found in file.")

rho_hist  = reorder_history_3d(rho_raw,  t, r, Z)
T_hist    = reorder_history_3d(T_raw,    t, r, Z)
xHII_hist = reorder_history_3d(xHII_raw, t, r, Z)
xH2_hist  = reorder_history_3d(xH2_raw,  t, r, Z)

k = nearest_saved_index(t, TargetTime_kyr * kyr)

print(f"Requested target time = {TargetTime_kyr:.3f} kyr")
print(f"Using snapshot at     = {t[k] / kyr:.3f} kyr")

rho  = rho_hist[k]
Tgas = T_hist[k]
xHII = xHII_hist[k]
xH2  = xH2_hist[k]

# Hydrogen neutral fraction:
# xHI = H nuclei not in HII or H2.
xHI = np.clip(1.0 - xHII - 2.0 * xH2, 0.0, 1.0)

nH = X * rho / mH

if DensityVariable == "nH":
    density = nH
    xlabel = r"$\log_{10}\left(n_H/{\rm cm}^{-3}\right)$"
    density_label = r"$n_H$"
elif DensityVariable == "rho":
    density = rho
    xlabel = r"$\log_{10}\left(\rho/{\rm g\,cm}^{-3}\right)$"
    density_label = r"$\rho$"
else:
    raise ValueError("DensityVariable must be either 'nH' or 'rho'.")

# -------------------------------------------------------------------
# Weights
# -------------------------------------------------------------------

# Correct cylindrical/nonuniform-grid cell masses using saved LYDION volumes.
cell_mass = rho * CellVolumes

if MassWeighted:
    weight_field = cell_mass
    ylabel = r"Mass fraction per dex"
    yquantity_label = "mass"
elif UseVolumeWeights:
    weight_field = CellVolumes
    ylabel = r"Volume fraction per dex"
    yquantity_label = "volume"
else:
    weight_field = np.ones_like(rho)
    ylabel = r"Cell fraction per dex"
    yquantity_label = "cell number"


# -------------------------------------------------------------------
# Build selected and all-gas samples
# -------------------------------------------------------------------

mask_selected = make_selection_mask(density, rho, Tgas, xHI)
mask_all = (
    np.isfinite(density) & (density > 0.0)
    & np.isfinite(weight_field) & (weight_field > 0.0)
)

if np.count_nonzero(mask_selected) == 0:
    raise RuntimeError("No cells satisfy the selected cuts.")

log_density_selected = np.log10(density[mask_selected])
weights_selected     = weight_field[mask_selected]

log_density_all = np.log10(density[mask_all])
weights_all     = weight_field[mask_all]

# Histogram range
if log_density_min is None or log_density_max is None:
    range_source = log_density_selected

    lo, hi = np.percentile(range_source, auto_range_percentiles)

    if log_density_min is None:
        log_density_min_use = lo
    else:
        log_density_min_use = log_density_min

    if log_density_max is None:
        log_density_max_use = hi
    else:
        log_density_max_use = log_density_max

    # Add a small buffer so edge bins do not clip the distribution.
    width = log_density_max_use - log_density_min_use
    log_density_min_use -= 0.05 * width
    log_density_max_use += 0.05 * width
else:
    log_density_min_use = log_density_min
    log_density_max_use = log_density_max

if log_density_max_use <= log_density_min_use:
    raise RuntimeError(
        f"Bad histogram range: {log_density_min_use} to {log_density_max_use}"
    )

bin_edges = np.linspace(log_density_min_use, log_density_max_use, Nbins + 1)

centers_selected, pdf_selected, frac_selected, edges = make_weighted_pdf(
    log_density_selected, weights_selected, bin_edges
)

centers_all, pdf_all, frac_all, _ = make_weighted_pdf(
    log_density_all, weights_all, bin_edges
)

# Weighted lognormal fits in log10 density
mu_selected, sigma_selected = weighted_mean_and_std(log_density_selected, weights_selected)
mu_all, sigma_all = weighted_mean_and_std(log_density_all, weights_all)

xfit = np.linspace(log_density_min_use, log_density_max_use, 800)
fit_selected = gaussian_pdf(xfit, mu_selected, sigma_selected)
fit_all      = gaussian_pdf(xfit, mu_all, sigma_all)

# Fraction retained by the cuts, using the chosen weighting.
total_weight_all = np.sum(weights_all[np.isfinite(weights_all) & (weights_all > 0.0)])
total_weight_selected = np.sum(
    weights_selected[np.isfinite(weights_selected) & (weights_selected > 0.0)]
)

selected_fraction = total_weight_selected / total_weight_all

n_cells_selected = np.count_nonzero(mask_selected)
n_cells_all      = np.count_nonzero(mask_all)

print(f"Cells used after cuts = {n_cells_selected} / {n_cells_all}")
print(f"Selected {yquantity_label} fraction = {selected_fraction:.6e}")
print(f"Filtered fit: mu = {mu_selected:.4f}, sigma = {sigma_selected:.4f} dex")

if CompareWithAllGas:
    print(f"All-gas fit:   mu = {mu_all:.4f}, sigma = {sigma_all:.4f} dex")


# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=figsize)

if CompareWithAllGas:
    ax.step(
        centers_all, pdf_all,
        where="mid",
        color=hist_color_all,
        lw=hist_lw_all,
        alpha=hist_alpha_all,
        label="All gas"
    )

    if ShowLognormalFit and (not FitFilteredOnly):
        ax.plot(
            xfit, fit_all,
            color=fit_color_all,
            lw=fit_lw,
            ls="--",
            label=rf"All gas lognormal fit: $\sigma={sigma_all:.2f}$ dex"
        )

ax.step(
    centers_selected, pdf_selected,
    where="mid",
    color=hist_color_filtered,
    lw=hist_lw_filtered,
    alpha=hist_alpha_filtered,
    label="Selected gas"
)

if ShowLognormalFit:
    ax.plot(
        xfit, fit_selected,
        color=fit_color_filtered,
        lw=fit_lw,
        ls="--",
        label=rf"Selected lognormal fit: $\sigma={sigma_selected:.2f}$ dex"
    )

# Axis labels
ax.set_xlabel(xlabel, fontsize=fontsize_label)
ax.set_ylabel(ylabel, fontsize=fontsize_label)

# Optional title
if ShowTitle:
    ax.set_title(
        rf"{ylabel}, $t = {t[k] / kyr:.2f}\,$kyr",
        fontsize=fontsize_title,
        pad=18
    )

# Limits
ax.set_xlim(log_density_min_use, log_density_max_use)

y_candidates = [np.nanmax(pdf_selected)]

if CompareWithAllGas:
    y_candidates.append(np.nanmax(pdf_all))

if ShowLognormalFit:
    y_candidates.append(np.nanmax(fit_selected))

if ShowLognormalFit and CompareWithAllGas and (not FitFilteredOnly):
    y_candidates.append(np.nanmax(fit_all))

ymax = 1.15 * np.nanmax(y_candidates)

if not np.isfinite(ymax) or ymax <= 0.0:
    ymax = 1.0

ax.set_ylim(0.0, ymax)

# Ticks
ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
ax.tick_params(axis="both", which="minor", labelsize=fontsize_ticks)

ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

# Grid
ax.grid(True, which="major", alpha=0.25)
ax.grid(True, which="minor", alpha=0.10)

# Info text
if ShowInfoText:
    cut_lines = []

    if ApplyNeutralCut:
        cut_lines.append(rf"$x_{{HI}} > {xHI_min:g}$")

    if ApplyTemperatureCut:
        cut_lines.append(rf"$T < {T_max_K:g}\,$K")

    if density_min_cut is not None:
        cut_lines.append(rf"{density_label} $> {density_min_cut:g}$")

    if density_max_cut is not None:
        cut_lines.append(rf"{density_label} $< {density_max_cut:g}$")

    if len(cut_lines) == 0:
        cut_text = "No cuts"
    else:
        cut_text = ", ".join(cut_lines)

    info_lines = [
        cut_text,
        rf"selected {yquantity_label} fraction = {selected_fraction:.3f}",
        rf"$\mu = {mu_selected:.2f}$, $\sigma = {sigma_selected:.2f}$ dex",
    ]

    text_style = dict(
        ha="left",
        va="top",
        fontsize=fontsize_text,
        color="black",
        path_effects=[pe.withStroke(linewidth=3.0, foreground="white")]
    )

    ax.text(
        info_x, info_y,
        "\n".join(info_lines),
        transform=ax.transAxes,
        **text_style
    )

# Legend
leg = ax.legend(
    loc="upper right",
    fontsize=fontsize_legend,
    handlelength=2.4,
    borderaxespad=0.8
)

for text in leg.get_texts():
    text.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

for spine in ax.spines.values():
    spine.set_linewidth(2.5)
    spine.set_color("black")

plt.savefig(FigureName, bbox_inches="tight", dpi=dpi)
plt.show()

print(f"Saved figure to {FigureName}")