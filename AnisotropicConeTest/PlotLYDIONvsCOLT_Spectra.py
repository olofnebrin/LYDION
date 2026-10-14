import os
import re
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# CGS constants used only for optional unit conversion between
# mean intensity J and radiation energy density u.
c_cgs = 2.99792458e10


###############################################################
#######                  U S E R  I N P U T             #######
###############################################################

# Files should be in the same directory as this script, unless full paths are given.
LYDION_FileName = "CloudTest_theta0.7853981633974483_rs0.5_tau1.0e8_T10000.0_umax0.0_tauabs0.0_pd0.0_256x256.h5"
COLT_FileName_xcrit0 = "Lya-cloud-xcrit0_1e7photons.hdf5"
COLT_FileName_xcrit1 = "Lya-cloud-xcrit1_1e7photons.hdf5"
COLT_FileName_xcrit2 = "Lya-cloud-xcrit2_1e7photons.hdf5"

# Set to [] to plot only LYDION.
COLT_FILES = [
    (COLT_FileName_xcrit0, "COLT no core-skipping"),
    #(COLT_FileName_xcrit1, r"COLT $x_{crit}=1$"),
    #(COLT_FileName_xcrit2, r"COLT $x_{crit}=2$"),
]

outname = "ConeTest_Spectra.pdf"
savefig = True

# Which LYDION snapshot to plot:
#   "final" -> final saved snapshot with t > 0
#   integer -> explicit snapshot index
snapshot_to_plot = "final"

# Which edge spectra to compare.
# By default we compare the spectra at the outer r and outer Z boundaries.
# LYDION save convention:
#   right: i_r = N_r - 1, j = 2:(N_Z-1)
#   top:   j_Z = N_Z - 1, i = 2:(N_r-2)
#   bottom: j_Z = 2, i = 2:(N_r-2)  [midplane side for half-domain runs]
boundaries_to_plot = ["right", "top"]

# Which edge-cell spectra to plot.
#   "pole_equator" -> default requested comparison:
#       right boundary: cell nearest the equator/midplane, i.e. minimum saved Z.
#       top boundary:   cell nearest the pole/Z-axis, i.e. minimum saved R.
#   "middle" -> one representative cell near the middle of each boundary.
#   "average" -> average over all saved edge cells on each boundary.
#
# For the anisotropic cone test, "pole_equator" is usually the cleanest:
# it compares escape along the equatorial outer-R boundary and polar outer-Z boundary.
spectrum_cell_selection = "middle"

# Spectral normalization for plotted curves.
#   "area" -> normalize integral over x to 1
#   "peak" -> normalize peak to 1
#   "none" -> no normalization
# Default is raw/non-normalized spectra.
spectrum_normalization = "none"

# What to plot for the raw/non-normalized comparison.
#   "Jx_equivalent" -> compare spectra per Doppler unit dx.
#                      LYDION: J_nu -> J_x = J_nu * Delta_nu_D.
#                      COLT:   u_x  -> J_x = c*u_x/(4*pi).
#   "energy_density_x" -> compare spectral energy density per dx.
#                         LYDION: u_x = 4*pi*J_nu*Delta_nu_D/c.
#                         COLT:   u_x as saved.
#   "raw_saved" -> plot raw saved arrays with no unit conversion.
#                  This is useful for debugging only; amplitudes are not
#                  expected to be comparable because LYDION saves J_nu,
#                  while COLT saves u_x = du/dx.
plot_spectral_quantity = "Jx_equivalent"

# LYDION spectra: for raw spectra, use J_Lya_EdgeSpectra rather than
# J_Lya_EdgeSpectra_Norm. Set True only for shape-only comparisons.
use_lydion_normalized_dataset = False

# COLT convention from Aaron: C-order, no transpose, shape (n_r, n_z), n_z=2*n_r.
COLT_flattening_order = "r_z"

# COLT has full Z-domain. LYDION edge indices refer to the simulated upper half.
# If COLT is ordered [lower half, upper half], the matching COLT index is:
# j_colt = N_Z + (j_lydion_1based - 1).
COLT_z_order = "lower_upper"  # options: "lower_upper", "upper_lower"

# COLT spectral dataset.
COLT_spectrum_dataset = "u_rad_x"

# Aaron's COLT frequency grid:
#   x in [-75, 75] with 150 bins.
# Since this is a binned quantity (u_rad_x in units of /dx), the safest default is
# to interpret [-75, 75] as bin edges and use the bin centers below.
# If Aaron instead meant that the 150 saved values are already cell centers, replace
# this by: np.linspace(-75.0, 75.0, 150)
COLT_x_edges_manual = np.linspace(-75.0, 75.0, 151)
COLT_x_centers_manual = 0.5 * (COLT_x_edges_manual[:-1] + COLT_x_edges_manual[1:])

# Static-cloud temperature used to convert LYDION δνC [Hz] to Doppler x if needed.
# If None, the code tries to parse _T..._ from LYDION_FileName.
T_for_Doppler_x = None

# Plot style.
figsize = (6.5, 7.0)
dpi_save = 300
linewidth_lydion = 3.0
linewidth_colt = 2.6
alpha_colt = 0.95
xlim_user = (-40.0, 40.0)  # e.g. (-80, 80)
ylim_user = (-0.005, 0.085)           # e.g. (-0.01, 0.1); None lets matplotlib choose

# Use colors to distinguish boundaries, and line style to distinguish code.
boundary_colors = {
    "right": "dodgerblue",
    "top": "orangered",
    "bottom": "C2",
}
LYDION_linestyle = "-"
COLT_linestyle = "--"
COLT_dash_pattern = (6.0, 3.0)
dash_capstyle = "round"


###############################################################
#######                H E L P E R S                    #######
###############################################################

def script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def resolve_path(fname):
    if os.path.isabs(fname):
        return fname
    return os.path.join(script_dir(), fname)


def choose_snapshot(t, snapshot_to_plot):
    if snapshot_to_plot == "final":
        positive_idx = np.where(np.isfinite(t) & (t > 0.0))[0]
        if positive_idx.size > 0:
            return int(positive_idx[-1])
        print("WARNING: No positive entries found in Times. Using snapshot index 0.")
        return 0
    return int(snapshot_to_plot)


def parse_temperature_from_name(fname):
    m = re.search(r"_T([0-9.eE+-]+)_", os.path.basename(fname))
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def doppler_width_Hz(T):
    # Δν_D = ν0 * sqrt(2 kB T / mH) / c
    c_cgs = 2.99792458e10
    kB_cgs = 1.380649e-16
    mH_cgs = 1.6735575e-24
    lambda_Lya_cm = 1215.67e-8
    nu0 = c_cgs / lambda_Lya_cm
    vth = np.sqrt(2.0 * kB_cgs * T / mH_cgs)
    return nu0 * vth / c_cgs


def lydion_x_from_delta_nu(delta_nu_c, T):
    delta_nu_c = np.asarray(delta_nu_c, dtype=np.float64)

    # Heuristic: if values are already modest, they may already be in Doppler units.
    # In the static-cloud files δνC is normally Hz, so this usually converts.
    if np.nanmax(np.abs(delta_nu_c)) < 1.0e5:
        print("WARNING: LYDION δνC values look small; assuming they are already Doppler x.")
        return delta_nu_c

    if T is None:
        raise RuntimeError(
            "Could not infer temperature for δνC -> x conversion. "
            "Set T_for_Doppler_x manually."
        )

    return delta_nu_c / doppler_width_Hz(T)


def normalize_spectrum(x, y, mode="area"):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y = np.where(np.isfinite(y), y, 0.0)

    if mode == "none":
        return y

    if mode == "peak":
        ymax = np.nanmax(y)
        if ymax > 0.0 and np.isfinite(ymax):
            return y / ymax
        return y

    if mode == "area":
        # Sort in x before integrating.
        idx = np.argsort(x)
        area = np.trapz(y[idx], x[idx])
        if area > 0.0 and np.isfinite(area):
            return y / area
        return y

    raise ValueError("spectrum_normalization must be 'area', 'peak', or 'none'.")


def convert_lydion_spectrum_for_plot(J_nu):
    """Convert the LYDION saved spectrum to the requested plotted quantity."""
    if plot_spectral_quantity == "raw_saved":
        return np.asarray(J_nu, dtype=np.float64)

    T = T_for_Doppler_x if T_for_Doppler_x is not None else parse_temperature_from_name(LYDION_FileName)
    if T is None:
        raise RuntimeError(
            "Need T_for_Doppler_x or a _T..._ value in LYDION_FileName to convert "
            "LYDION J_nu to a per-x spectrum."
        )

    Delta_nu_D = doppler_width_Hz(T)

    if plot_spectral_quantity == "Jx_equivalent":
        # J_x dx = J_nu dnu, with dnu = Delta_nu_D dx.
        return np.asarray(J_nu, dtype=np.float64) * Delta_nu_D

    if plot_spectral_quantity == "energy_density_x":
        # u_x = 4*pi*J_x/c, assuming LYDION J is the usual mean intensity.
        return (4.0 * np.pi / c_cgs) * np.asarray(J_nu, dtype=np.float64) * Delta_nu_D

    raise ValueError("plot_spectral_quantity must be 'Jx_equivalent', 'energy_density_x', or 'raw_saved'.")


def convert_colt_spectrum_for_plot(u_x):
    """Convert the COLT saved u_rad_x spectrum to the requested plotted quantity."""
    if plot_spectral_quantity == "raw_saved":
        return np.asarray(u_x, dtype=np.float64)

    if plot_spectral_quantity == "Jx_equivalent":
        # Aaron's u_rad_x is spectral energy density per dx.
        # For mean intensity, u_x = 4*pi*J_x/c -> J_x = c*u_x/(4*pi).
        return (c_cgs / (4.0 * np.pi)) * np.asarray(u_x, dtype=np.float64)

    if plot_spectral_quantity == "energy_density_x":
        return np.asarray(u_x, dtype=np.float64)

    raise ValueError("plot_spectral_quantity must be 'Jx_equivalent', 'energy_density_x', or 'raw_saved'.")


def plotted_spectrum_ylabel():
    if spectrum_normalization == "area":
        return r"normalized spectrum"
    if spectrum_normalization == "peak":
        return r"peak-normalized spectrum"

    if plot_spectral_quantity == "Jx_equivalent":
        return r"$J_x$"
    if plot_spectral_quantity == "energy_density_x":
        return r"$u_x$"
    return r"raw saved spectrum"


def find_dataset_recursive(h5obj, candidate_names):
    """Return the first dataset whose basename matches one of candidate_names."""
    found = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            base = name.split("/")[-1]
            if base in candidate_names:
                found.append(name)

    h5obj.visititems(visitor)
    return found[0] if found else None


def find_colt_x_centers(f, n_x, fallback_x=None):
    if COLT_x_centers_manual is not None:
        x = np.asarray(COLT_x_centers_manual, dtype=np.float64)
        if x.size != n_x:
            raise RuntimeError(
                f"COLT_x_centers_manual has length {x.size}, but u_rad_x has {n_x} bins."
            )
        return x, "manual"

    # Try common names, including inside groups such as /line/x.
    candidates_centers = [
        "x", "x_c", "xC", "x_centers", "x_center", "x_bins", "nu_x", "frequency_x",
        "doppler_x", "Doppler_x", "velocity_x"
    ]
    candidates_edges = [
        "x_edges", "x_edge", "x_grid_edges", "x_bin_edges"
    ]

    path = find_dataset_recursive(f, candidates_centers)
    if path is not None:
        arr = np.asarray(f[path][:], dtype=np.float64).squeeze()
        if arr.ndim == 1 and arr.size == n_x:
            return arr, path

    path = find_dataset_recursive(f, candidates_edges)
    if path is not None:
        edges = np.asarray(f[path][:], dtype=np.float64).squeeze()
        if edges.ndim == 1 and edges.size == n_x + 1:
            return 0.5 * (edges[:-1] + edges[1:]), path

    # If no COLT x grid was saved/found, use the LYDION x range as a plotting fallback.
    # This preserves a usable plot, but the user should verify the COLT x-grid convention.
    if fallback_x is not None:
        print(
            "WARNING: Could not find COLT x grid in file. "
            "Using a uniform grid spanning the LYDION x-range as a fallback."
        )
        return np.linspace(np.nanmin(fallback_x), np.nanmax(fallback_x), n_x), "fallback_LYDION_range"

    print("WARNING: Could not find COLT x grid in file. Using bin index as x-axis.")
    return np.arange(n_x, dtype=np.float64), "fallback_bin_index"


def reshape_colt_spectrum_field(arr, n_r_lydion, n_z_lydion):
    """
    Return shape (Nr, 2*Nz, Nx), following Aaron's C-order r_z convention by default.
    """
    arr = np.asarray(arr, dtype=np.float64)

    if arr.ndim != 2:
        raise RuntimeError(f"Expected COLT u_rad_x to be 2D (Ncell,Nx). Got shape {arr.shape}")

    n_cell, n_x = arr.shape
    expected = n_r_lydion * (2 * n_z_lydion)
    if n_cell != expected:
        raise RuntimeError(
            f"COLT u_rad_x has {n_cell} cells, but expected {expected} = "
            f"Nr*(2*Nz) with Nr={n_r_lydion}, Nz={n_z_lydion}."
        )

    if COLT_flattening_order == "r_z":
        return arr.reshape((n_r_lydion, 2 * n_z_lydion, n_x))

    if COLT_flattening_order == "z_r":
        return arr.reshape((2 * n_z_lydion, n_r_lydion, n_x)).transpose(1, 0, 2)

    raise ValueError("COLT_flattening_order must be 'r_z' or 'z_r'.")


def colt_j_from_lydion_j(j0_lydion, n_z_lydion):
    """Map LYDION upper-half 0-based Z index to COLT full-domain 0-based Z index."""
    if COLT_z_order == "lower_upper":
        return n_z_lydion + j0_lydion
    if COLT_z_order == "upper_lower":
        return j0_lydion
    raise ValueError("COLT_z_order must be 'lower_upper' or 'upper_lower'.")


def representative_middle_indices(mask, i0, j0):
    """Pick one edge cell near the middle of a selected boundary."""
    inds = np.where(mask)[0]
    if inds.size == 0:
        return inds

    ii = i0[inds]
    jj = j0[inds]

    # Choose by median along the varying coordinate.
    if np.unique(ii).size == 1:
        target = np.median(jj)
        pick = inds[np.argmin(np.abs(jj - target))]
    else:
        target = np.median(ii)
        pick = inds[np.argmin(np.abs(ii - target))]
    return np.array([pick], dtype=int)


def representative_pole_equator_indices(boundary, mask, i0, j0):
    """
    Pick the physically requested boundary cells:
      * right / outer-R boundary: nearest the equator/midplane -> smallest saved Z index.
      * top / outer-Z boundary: nearest the pole/Z-axis -> smallest saved R index.

    LYDION's saved edge spectra exclude the exact domain boundaries/ghosts, so this
    chooses the closest saved cell to those directions.
    """
    inds = np.where(mask)[0]
    if inds.size == 0:
        return inds

    if boundary == "right":
        # Outer cylindrical-R boundary, nearest the equator/midplane: minimum Z.
        jj = j0[inds]
        pick = inds[np.argmin(jj)]
        return np.array([pick], dtype=int)

    if boundary == "top":
        # Outer +Z boundary, nearest the pole/Z-axis: minimum cylindrical R.
        ii = i0[inds]
        pick = inds[np.argmin(ii)]
        return np.array([pick], dtype=int)

    # For other boundaries, fall back to the middle-cell convention.
    return representative_middle_indices(mask, i0, j0)


def select_edge_indices(boundary, mask, i0, j0):
    if spectrum_cell_selection == "average":
        return np.where(mask)[0]
    if spectrum_cell_selection == "middle":
        return representative_middle_indices(mask, i0, j0)
    if spectrum_cell_selection == "pole_equator":
        return representative_pole_equator_indices(boundary, mask, i0, j0)
    raise ValueError(
        "spectrum_cell_selection must be 'pole_equator', 'middle', or 'average'."
    )


def read_lydion_edge_spectra(fname):
    path = resolve_path(fname)
    T = T_for_Doppler_x if T_for_Doppler_x is not None else parse_temperature_from_name(fname)

    with h5py.File(path, "r") as f:
        t = np.asarray(f["Times"][:], dtype=np.float64)
        k = choose_snapshot(t, snapshot_to_plot)

        r = np.asarray(f["r"][:], dtype=np.float64)
        Z = np.asarray(f["Z"][:], dtype=np.float64)
        rCoM = np.asarray(f["rCoM"][:], dtype=np.float64)
        ZCoM = np.asarray(f["ZCoM"][:], dtype=np.float64)

        Nr = len(r) - 1
        Nz = len(Z) - 1

        delta_nu_c = np.asarray(f["δνC"][:], dtype=np.float64)
        x_lydion = lydion_x_from_delta_nu(delta_nu_c, T)

        dataset = "J_Lya_EdgeSpectra_Norm" if use_lydion_normalized_dataset and "J_Lya_EdgeSpectra_Norm" in f else "J_Lya_EdgeSpectra"
        J_edge_all = np.asarray(f[dataset][:], dtype=np.float64)
        ridx_1 = np.asarray(f["J_Lya_Edge_rIndex"][:], dtype=int)
        zidx_1 = np.asarray(f["J_Lya_Edge_ZIndex"][:], dtype=int)

    if k < 0 or k >= t.size:
        raise RuntimeError(f"snapshot index {k} outside valid range [0,{t.size-1}]")

    if J_edge_all.ndim != 3:
        raise RuntimeError(f"Expected LYDION edge spectra 3D. Got shape {J_edge_all.shape}")

    # Expected shape is (N_edge, Nν, Nsaved), but be defensive.
    N_edge = ridx_1.size
    Nnu = x_lydion.size
    if J_edge_all.shape == (N_edge, Nnu, t.size):
        J_snapshot = J_edge_all[:, :, k]
    elif J_edge_all.shape == (t.size, Nnu, N_edge):
        J_snapshot = J_edge_all[k, :, :].T
    elif J_edge_all.shape == (Nnu, N_edge, t.size):
        J_snapshot = J_edge_all[:, :, k].T
    else:
        raise RuntimeError(
            f"Could not interpret LYDION edge spectra shape {J_edge_all.shape}; "
            f"expected something compatible with N_edge={N_edge}, Nnu={Nnu}, Nt={t.size}."
        )

    i0 = ridx_1 - 1
    j0 = zidx_1 - 1

    return {
        "path": path,
        "t": t,
        "snapshot_index": k,
        "r": r,
        "Z": Z,
        "rCoM": rCoM,
        "ZCoM": ZCoM,
        "Nr": Nr,
        "Nz": Nz,
        "x": x_lydion,
        "J_edge": J_snapshot,
        "i0": i0,
        "j0": j0,
        "dataset": dataset,
        "T_for_x": T,
    }


def boundary_mask(name, i0, j0, Nr, Nz):
    if name == "right":
        return i0 == (Nr - 2)   # Julia i=N_r-1 -> Python zero-based N_r-2
    if name == "top":
        return j0 == (Nz - 2)   # Julia j=N_Z-1 -> Python zero-based N_Z-2
    if name == "bottom":
        return j0 == 1          # Julia j=2 -> Python zero-based 1
    raise ValueError("boundary name must be 'right', 'top', or 'bottom'.")


def boundary_label(name):
    if name == "right":
        return r"outer $R$ boundary"
    if name == "top":
        return r"outer $Z$ boundary"
    if name == "bottom":
        return r"lower saved boundary"
    return name


def average_lydion_boundary_spectrum(lyd, boundary):
    mask = boundary_mask(boundary, lyd["i0"], lyd["j0"], lyd["Nr"], lyd["Nz"])
    inds = select_edge_indices(boundary, mask, lyd["i0"], lyd["j0"])

    if inds.size == 0:
        raise RuntimeError(f"No LYDION edge spectra found for boundary={boundary!r}")

    # If spectrum_cell_selection == "average", this averages over the full boundary.
    # Otherwise inds has length 1, so this simply returns that selected cell spectrum.
    spec = np.mean(lyd["J_edge"][inds, :], axis=0)
    return spec, inds


def average_colt_boundary_spectrum(fname, lyd, boundary, edge_inds):
    path = resolve_path(fname)
    with h5py.File(path, "r") as f:
        if COLT_spectrum_dataset not in f:
            raise RuntimeError(f"Dataset {COLT_spectrum_dataset!r} not found in {path}")
        u_raw = np.asarray(f[COLT_spectrum_dataset][:], dtype=np.float64)
        u_grid = reshape_colt_spectrum_field(u_raw, lyd["Nr"], lyd["Nz"])
        x_colt, x_source = find_colt_x_centers(f, u_grid.shape[2], fallback_x=lyd["x"])

    specs = []
    for e in edge_inds:
        i = int(lyd["i0"][e])
        j_upper = int(lyd["j0"][e])
        j_colt = colt_j_from_lydion_j(j_upper, lyd["Nz"])
        specs.append(u_grid[i, j_colt, :])

    spec = np.mean(np.asarray(specs), axis=0)
    return path, x_colt, spec, x_source


###############################################################
#######                  R E A D  D A T A               #######
###############################################################

lyd = read_lydion_edge_spectra(LYDION_FileName)

print("\n========== Loaded spectral comparison ==========")
print(f"LYDION file:          {lyd['path']}")
print(f"LYDION spectrum dset: {lyd['dataset']}")
print(f"Spectrum normalization: {spectrum_normalization}")
print(f"LYDION snapshot:      k={lyd['snapshot_index']}, t={lyd['t'][lyd['snapshot_index']]:.6e}")
print(f"LYDION grid:          Nr={lyd['Nr']}, Nz={lyd['Nz']}")
print(f"LYDION T used for x:  {lyd['T_for_x']}")
print(f"COLT z order:         {COLT_z_order}")
print(f"COLT flattening:      {COLT_flattening_order}")
print(f"Spectral quantity:    {plot_spectral_quantity}")
print(f"Spectrum normalization: {spectrum_normalization}")
print(f"Cell selection:       {spectrum_cell_selection}")
print("===============================================\n")


###############################################################
#######                    P L O T                     #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 21})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['xtick.minor.width'] = 2.0
plt.rcParams['xtick.major.size'] = 12
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['ytick.minor.width'] = 2.0
plt.rcParams['ytick.major.size'] = 12
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['legend.frameon'] = False
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig, ax = plt.subplots(figsize=figsize)

# Store first plotted x-source diagnostics.
for boundary in boundaries_to_plot:
    spec_lyd, edge_inds = average_lydion_boundary_spectrum(lyd, boundary)
    spec_lyd_plot = convert_lydion_spectrum_for_plot(spec_lyd)
    y_lyd = normalize_spectrum(lyd["x"], spec_lyd_plot, spectrum_normalization)

    color = boundary_colors.get(boundary, "black")

    ax.plot(
        lyd["x"], y_lyd,
        linestyle=LYDION_linestyle,
        linewidth=linewidth_lydion,
        color=color,
        solid_capstyle=dash_capstyle,
        label=rf"LYDION, {boundary_label(boundary)}",
    )

    print(f"Boundary {boundary!r}: using {edge_inds.size} LYDION/COLT cell(s)")
    if edge_inds.size > 0:
        e0 = edge_inds[0]
        e1 = edge_inds[-1]
        print(
            f"  first edge index: i={lyd['i0'][e0]+1}, j={lyd['j0'][e0]+1} (Julia 1-based); "
            f"last: i={lyd['i0'][e1]+1}, j={lyd['j0'][e1]+1}"
        )

    for colt_fname, colt_label in COLT_FILES:
        colt_path, x_colt, spec_colt, x_source = average_colt_boundary_spectrum(
            colt_fname, lyd, boundary, edge_inds
        )
        spec_colt_plot = convert_colt_spectrum_for_plot(spec_colt)
        y_colt = normalize_spectrum(x_colt, spec_colt_plot, spectrum_normalization)

        line, = ax.plot(
            x_colt, y_colt,
            linestyle=COLT_linestyle,
            linewidth=linewidth_colt,
            color=color,
            alpha=alpha_colt,
            dash_capstyle=dash_capstyle,
            solid_capstyle=dash_capstyle,
            label=rf"{colt_label}, {boundary_label(boundary)}",
        )
        line.set_dashes(COLT_dash_pattern)
        print(f"  COLT file {os.path.basename(colt_path)}: x source = {x_source}")

ax.set_xlabel(r"$x = (\nu-\nu_{Ly\alpha})/\Delta\nu_D$", labelpad=10, fontsize = 25)
ax.set_ylabel(plotted_spectrum_ylabel(), labelpad=10, fontsize = 25)

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

ax.set_title("Anisotropic cone test: Spectra", pad=15)

if xlim_user is not None:
    ax.set_xlim(*xlim_user)
if ylim_user is not None:
    ax.set_ylim(*ylim_user)

#ax.legend(fontsize=12, loc="best", handlelength=3.2)
ax.text(0.05, 0.85, "Solid lines: LYDION\nDashed lines: Monte Carlo RT",
         transform=ax.transAxes, fontsize=22, color="black")
ax.text(0.05, 0.8-0.015, "Orange: Upper Z-boundary", 
        transform=ax.transAxes, fontsize = 20, color = "orangered")
ax.text(0.05, 0.75-0.015, "Blue: Outer R-boundary", 
        transform=ax.transAxes, fontsize = 20, color = "dodgerblue")
fig.tight_layout()

if savefig:
    outpath = resolve_path(outname)
    fig.savefig(outpath, dpi=dpi_save, bbox_inches="tight")
    print(f"Saved figure: {outpath}")
else:
    plt.show()
