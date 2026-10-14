import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FuncFormatter

from StarFormationCriteria import (
    velocity_from_momentum,
    self_gravitating_mask,
    converging_flow_mask,
    selected_gas_mass,
)
import os
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
os.chdir(repo_root)


###############################################################
#######                                                 #######
#######                  C O N S T A N T S              #######
#######                                                 #######
###############################################################

Msun = 1.989e33
mH   = 1.6735575e-24

yr   = 3.1556926e7
kyr  = 1e3 * yr


###############################################################
#######                                                 #######
#######             U S E R   O P T I O N S             #######
#######                                                 #######
###############################################################

# Add/remove Lyα RHD simulations here.
simulations_Lya = [
    {
        "filename"   : "RHDSim_StarCluster_Lya_128x128_2026-04-19.h5",
        "label"      : r"$128^2$",
        "color"      : "crimson",
        "linestyle"  : "-",
        "linewidth"  : 3.0,
        "alpha"      : 0.8,
    },
    {
        "filename"   : "RHDSim_StarCluster_Lya_181x181_2026-04-19.h5",
        "label"      : r"$181^2$",
        "color"      : "orange",
        "linestyle"  : "-",
        "linewidth"  : 3.0,
        "alpha"      : 0.8,
    },
    {
        "filename"   : "RHDSim_StarCluster_Lya_256x256_2026-04-19.h5",
        "label"      : r"$256^2$",
        "color"      : "limegreen",
        "linestyle"  : "-",
        "linewidth"  : 3.0,
        "alpha"      : 0.8,
    },
    {
        "filename"   : "RHDSim_StarCluster_Lya_362x362_2026-04-19.h5",
        "label"      : r"$362^2$",
        "color"      : "dodgerblue",
        "linestyle"  : "-",
        "linewidth"  : 3.0,
        "alpha"      : 0.8,
    },
    {
        "filename"   : "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5",
        "label"      : r"$512^2$",
        "color"      : "black",
        "linestyle"  : "-",
        "linewidth"  : 4.5,
        "alpha"      : 0.8,
    },
]

# Add/remove RHD simulations without Lyα here.
simulations_noLya = [
    {
        "filename"   : "RHDSim_StarCluster_noLya_128x128_2026-04-19.h5",
        "label"      : r"$128^2$",
        "color"      : "crimson",
        "linestyle"  : "--",
        "linewidth"  : 3.0,
        "alpha"      : 0.8,
    },
    {
        "filename"   : "RHDSim_StarCluster_noLya_181x181_2026-04-19.h5",
        "label"      : r"$181^2$",
        "color"      : "orange",
        "linestyle"  : "--",
        "linewidth"  : 3.0,
        "alpha"      : 0.8,
    },
    {
        "filename"   : "RHDSim_StarCluster_noLya_256x256_2026-04-19.h5",
        "label"      : r"$256^2$",
        "color"      : "limegreen",
        "linestyle"  : "--",
        "linewidth"  : 3.0,
        "alpha"      : 0.8,
    },
    {
        "filename"   : "RHDSim_StarCluster_noLya_362x362_2026-04-19.h5",
        "label"      : r"$362^2$",
        "color"      : "dodgerblue",
        "linestyle"  : "--",
        "linewidth"  : 3.0,
        "alpha"      : 0.8,
    },
    {
        "filename"   : "RHDSim_StarCluster_noLya_512x512_2026-04-19.h5",
        "label"      : r"$512^2$",
        "color"      : "black",
        "linestyle"  : "--",
        "linewidth"  : 4.5,
        "alpha"      : 0.8,
    },
]

outfile = "LYDION_Figure_StarFormingGasMass_2Panel.pdf"

# Time range. Set to None to auto-use available range.
xlim = (-10.0, 160.0)

# y-axis limits. Set to None to autoscale.
ylim_M_SF   = (-1e2, 2.6e3)
ylim_M_tot  = (-1e2, 2.6e3)

# Default: log y-axis for all mass panels.
use_log_M_SF_Lya    = False
use_log_M_tot_Lya   = False
use_log_M_SF_noLya  = False
use_log_M_tot_noLya = False

# If using log axes, non-positive values are masked automatically.
mask_nonpositive_for_log = True

# Hide y tick labels in the right column to reduce clutter.
hide_right_column_y_ticklabels = True

# Legend options.
#
# Valid names:
#   "M_SF_Lya", "M_tot_Lya", "M_SF_noLya", "M_tot_noLya"
legend_panel  = "M_tot"
legend_loc    = "lower left"

# Manual legend placement.  Because legend_loc = "lower left", these are
# axes coordinates of the legend's lower-left corner.
# Increase the first value to move the legend right, and the second to move it up.
legend_bbox_to_anchor = (0.04, 0.06)
legend_fontsize = 28
legend_ncol = 2

# Manual text labels in the top panel.
# These replace the old legend entries "With/Ignoring Lyα feedback".
# Positions are in axes coordinates. Increase x to move right, increase y to move up.
show_feedback_text_labels = True
feedback_text_fontsize = 32
feedback_text_color = "black"
feedback_text_path_effects = [pe.withStroke(linewidth=3.5, foreground="white")]

feedback_text_labels = {
    "with_Lya" : {
        "text"     : r"With Ly$\alpha$ feedback",
        "position" : (0.42, 0.35),
        "rotation" : 0.0,
        "ha"       : "left",
        "va"       : "center",
    },
    "without_Lya" : {
        "text"     : r"Ignoring Ly$\alpha$"+"\n"+"feedback",
        "position" : (0.15, 0.55),
        "rotation" : 55.0,
        "ha"       : "left",
        "va"       : "center",
    },
}

# Minor tick options.
y_minor_subdivisions = 10

# Rescale mass-axis tick labels to save horizontal space.
# Example: with mass_tick_scale = 1e3, a tick at 2000 Msun is labelled as 2.
use_scaled_mass_ticklabels = True
mass_tick_scale = 1e3
mass_axis_unit_label = r"10$^3$ M$_{\odot}$"

# Panel-label text options.
show_panel_labels = True
panel_label_fontsize = 29

# Default panel-label position, used if a panel is not listed below.
panel_label_x_default = 0.07
panel_label_y_default = 0.85

# Individual panel-label positions.
panel_label_positions = {
    "M_SF_Lya"     : (0.07, 0.85),
    "M_tot_Lya"    : (0.07, 0.85),
    "M_SF_noLya"   : (0.07, 0.85),
    "M_tot_noLya"  : (0.07, 0.85),
}

# Figure layout.
fig_width  = 9.0
fig_height = 13.5

wspace = 0.03
hspace = 0.05

# Dataset names.
density_dataset_candidates = [
    "GasDensity_History",
    "Density_History",
    "rho_History",
    "ρ_History",
]

pressure_dataset_candidates = [
    "GasPressure_History",
    "Pressure_History",
    "P_History",
]

r_momentum_dataset_candidates = [
    "rMomentum_History",
    "GasrMomentum_History",
    "GasMomentum_r_History",
    "rho_u_r_History",
    "ρu_r_History",
]

Z_momentum_dataset_candidates = [
    "ZMomentum_History",
    "GasZMomentum_History",
    "GasMomentum_Z_History",
    "rho_u_Z_History",
    "ρu_Z_History",
]

# Force-density dataset names, units dyn cm^-3.
# These are used for the anti-gravity Eddington ratio criterion.
f_Lya_r_dataset_candidates = [
    "f_Lya_r_History",
    "f_Lya_X_History",
]

f_Lya_Z_dataset_candidates = [
    "f_Lya_Z_History",
    "f_Lya_Y_History",
]

f_direct_r_dataset_candidates = [
    "f_stellar_r_History",
    "f_direct_r_History",
    "f_star_r_History",
]

f_direct_Z_dataset_candidates = [
    "f_stellar_Z_History",
    "f_direct_Z_History",
    "f_star_Z_History",
]

f_gravity_r_dataset_candidates = [
    "f_gravity_r_History",
    "f_grav_r_History",
]

f_gravity_Z_dataset_candidates = [
    "f_gravity_Z_History",
    "f_grav_Z_History",
]

temperature_dataset_candidates = [
    "T_History",
    "Temperature_History",
    "GasTemperature_History",
]

xHII_dataset_candidates = [
    "xHII_History",
    "x_HII_History",
    "IonizedFraction_History",
]

cell_volume_dataset_candidates = [
    "CellVolumes",
    "CellVolume",
    "ΔV",
]

total_gas_mass_dataset_candidates = [
    "TotalGasMass_History",
    "M_simvol_History",
    "GasMass_History",
]

# Star-forming gas criteria.
#
# Default diagnostic:
#
#   anti-gravity Eddington ratio < 1
#   div(v) < 0
#
# plus optional cold/dense/neutral cuts.
#
# The total anti-gravity Eddington ratio is computed from the sum of Lyα
# and direct stellar radiation pressure forces:
#
#   f_Edd,anti = - f_grav · (f_Lya + f_direct) / |f_grav|^2
#
# Thus f_Edd,anti > 0 means radiation opposes gravity, while
# f_Edd,anti < 0 means the radiation force points partly along gravity.
use_feedback_eddington_criterion = True
use_converging_flow_criterion    = False
use_self_gravity_criterion       = False

# Candidate star-forming cells must have f_Edd,anti < f_Edd_crit.
f_Edd_crit = 1.0

# Include these force components in the anti-gravity Eddington ratio.
use_Lya_force    = True
use_direct_force = True

# If |f_grav| is <= this value, f_Edd is undefined and the cell is not selected.
# Units are dyn cm^-3, same as the saved force density fields.
f_grav_min = 0.0

alpha_crit = 1.0
gamma_ad   = 5.0 / 3.0

use_nH_criterion   = True
use_xHII_criterion = False
use_T_criterion    = False

nH_crit   = 1e7     # cm^-3
xHII_crit = 0.5
T_crit    = 1e3     # K

# Conversion from gas mass density to hydrogen nuclei density:
#
#   nH = hydrogen_mass_fraction * rho / mH
#
# If GasDensity is hydrogen-only gas density, use 1.0.
# If GasDensity is total gas density with H/He mixture, use ~0.76.
hydrogen_mass_fraction = 0.76

# The current plotting code has used 2 * sum(rho * CellVol), because the
# saved LYDION domain is the half-plane and the plotted mass is doubled.
# Keep this equal to 2.0 for consistency with the old figure.
symmetry_factor = 2.0

# For the total gas mass panel:
#
# True  = use saved TotalGasMass_History.
# False = recompute from rho * CellVolumes.
#
# Since SaveHDF5Data_RHD_StarCluster.jl saves TotalGasMass_History,
# this should normally stay True.
use_saved_total_gas_mass = True

# Print Mgas/Mgas[0] range as a sanity check.
print_mass_conservation_diagnostic = True


###############################################################
#######                                                 #######
#######           P L O T   C O S M E T I C S           #######
#######                                                 #######
###############################################################

plt.rc('font',**{'sans-serif':['AVHershey Complex'],
                'size': 35} )

params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

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
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'

plt.rcParams['legend.frameon'] = 'False'

plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linewidth'] = 2.5
plt.rcParams['grid.linestyle'] = '--'

plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True


###############################################################
#######                                                 #######
#######                  F U N C T I O N S              #######
#######                                                 #######
###############################################################

def get_valid_time_slice(t):
    """Return slice selecting saved, non-empty time entries."""

    t = np.asarray(t)
    nonzero = np.nonzero(t)[0]

    if len(nonzero) > 0:
        last_idx = nonzero[-1]
        return slice(0, last_idx + 1)

    return slice(None)


def get_first_existing_dataset_name(file, candidates, required=True):
    """Return the first dataset name in candidates that exists in the HDF5 file."""

    for name in candidates:
        if name in file:
            return name

    if required:
        available = list(file.keys())
        raise KeyError(
            "Could not find any of the requested datasets:\n"
            f"    {candidates}\n"
            "Available top-level keys include:\n"
            f"    {available[:80]}"
        )

    return None


def read_optional_time_dataset(file, candidates, sl, Nt_raw, Nr, NZ):
    """Read an optional time-history dataset.

    Returns None if none of the candidate names exist.
    """

    name = get_first_existing_dataset_name(file, candidates, required=False)

    if name is None:
        return None

    return read_time_dataset(file, name, sl, Nt_raw, Nr, NZ)


def read_time_dataset(file, name, sl, Nt_raw, Nr, NZ):
    """Read a time-history dataset.

    Returns 3D histories in physical LYDION order:

        (Nr, NZ, Nt)

    Also handles 1D histories.
    """

    arr = file[name][:]

    if arr.ndim == 1:
        return arr[sl]

    if arr.ndim == 3:

        # Case A: Python-visible reversed order from Julia/HDF5:
        #
        #     (Nsaved, N_Z, N_r)
        #
        # Convert to:
        #
        #     (N_r, N_Z, Nt)
        if arr.shape == (Nt_raw, NZ, Nr):
            arr = arr[sl, :, :]
            return np.transpose(arr, (2, 1, 0))

        # Case B: Already visible in physical order:
        #
        #     (N_r, N_Z, Nsaved)
        if arr.shape == (Nr, NZ, Nt_raw):
            return arr[:, :, sl]

        # Case C: Python-written test files:
        #
        #     (Nsaved, N_r, N_Z)
        if arr.shape == (Nt_raw, Nr, NZ):
            arr = arr[sl, :, :]
            return np.transpose(arr, (1, 2, 0))

        raise ValueError(
            f"Could not identify axes for dataset '{name}'. "
            f"Dataset shape is {arr.shape}. Expected one of "
            f"({Nt_raw}, {NZ}, {Nr}), ({Nr}, {NZ}, {Nt_raw}), "
            f"or ({Nt_raw}, {Nr}, {NZ})."
        )

    raise ValueError(f"Expected 1D or 3D dataset for '{name}', got shape {arr.shape}")


def read_static_2d_dataset(file, name, Nr, NZ):
    """Read a static 2D grid dataset and return physical order (Nr, NZ).

    For square grids this is ambiguous, so this should only be used
    as fallback. Prefer reconstructing CellVolumes from r/Z edges.
    """

    arr = file[name][:]

    if arr.shape == (Nr, NZ):
        return arr

    if arr.shape == (NZ, Nr):
        return arr.T

    raise ValueError(
        f"Unexpected shape for static dataset '{name}': {arr.shape}. "
        f"Expected ({Nr}, {NZ}) or ({NZ}, {Nr})."
    )


def compute_axisymmetric_cell_volumes_from_edges(r_edges, Z_edges, Nr, NZ):
    """Compute axisymmetric cylindrical cell volumes in physical order (Nr, NZ).

    Assumes full 2π axisymmetry:

        dV = 2π r dr dZ
           = π (r_R^2 - r_L^2) dZ

    Returns:
        CellVol[i,j] with i = r index and j = Z index.
    """

    r_edges = np.asarray(r_edges)
    Z_edges = np.asarray(Z_edges)

    if len(r_edges) != Nr + 1:
        raise ValueError(
            f"Expected len(r_edges) = Nr + 1 = {Nr + 1}, "
            f"got {len(r_edges)}."
        )

    if len(Z_edges) != NZ + 1:
        raise ValueError(
            f"Expected len(Z_edges) = NZ + 1 = {NZ + 1}, "
            f"got {len(Z_edges)}."
        )

    ΔV = np.zeros((Nr, NZ))

    for i in range(Nr):
        annulus_area = np.pi * (r_edges[i+1]**2 - r_edges[i]**2)

        for j in range(NZ):
            ΔZ = Z_edges[j+1] - Z_edges[j]
            ΔV[i, j] = annulus_area * ΔZ

    return ΔV


def read_cell_volumes(file, Nr, NZ):
    """Read or reconstruct cell volumes in physical order (Nr, NZ)."""

    if "r" in file and "Z" in file:
        return compute_axisymmetric_cell_volumes_from_edges(
            file["r"][:],
            file["Z"][:],
            Nr,
            NZ
        )

    cellvol_name = get_first_existing_dataset_name(
        file,
        cell_volume_dataset_candidates,
        required=True
    )

    return read_static_2d_dataset(file, cellvol_name, Nr, NZ)


def read_cell_widths(file, rCoM, ZCoM):
    """Read Δr and ΔZ from grid edges, with center-difference fallback.

    Returned arrays have shapes (Nr, 1) and (1, NZ), so broadcasting gives
    dx[i,j] = min(Δr[i], ΔZ[j]) in StarFormationCriteria.py.
    """

    if "r" in file and "Z" in file:
        delta_r = np.diff(file["r"][:])
        delta_Z = np.diff(file["Z"][:])
    else:
        delta_r = np.gradient(rCoM)
        delta_Z = np.gradient(ZCoM)

    return delta_r[:, None], delta_Z[None, :]


def read_mass_history(filename):
    """Read fields and compute star-forming/total gas mass histories."""

    with h5py.File(filename, "r") as file:
        data = {}

        t      = file["Times"][:]
        Nt_raw = len(t)
        sl     = get_valid_time_slice(t)

        rCoM = file["rCoM"][:]
        ZCoM = file["ZCoM"][:]

        Nr = len(rCoM)
        NZ = len(ZCoM)

        data["t"]    = t[sl]
        data["rCoM"] = rCoM
        data["ZCoM"] = ZCoM

        data["delta_r"], data["delta_Z"] = read_cell_widths(file, rCoM, ZCoM)

        density_name = get_first_existing_dataset_name(
            file,
            density_dataset_candidates,
            required=True
        )

        data["rho"] = read_time_dataset(
            file,
            density_name,
            sl,
            Nt_raw,
            Nr,
            NZ
        )

        if use_self_gravity_criterion:
            pressure_name = get_first_existing_dataset_name(
                file,
                pressure_dataset_candidates,
                required=True
            )

            data["P"] = read_time_dataset(
                file,
                pressure_name,
                sl,
                Nt_raw,
                Nr,
                NZ
            )
        else:
            data["P"] = None

        r_momentum_name = get_first_existing_dataset_name(
            file,
            r_momentum_dataset_candidates,
            required=True
        )

        data["mom_r"] = read_time_dataset(
            file,
            r_momentum_name,
            sl,
            Nt_raw,
            Nr,
            NZ
        )

        Z_momentum_name = get_first_existing_dataset_name(
            file,
            Z_momentum_dataset_candidates,
            required=True
        )

        data["mom_Z"] = read_time_dataset(
            file,
            Z_momentum_name,
            sl,
            Nt_raw,
            Nr,
            NZ
        )

        if use_feedback_eddington_criterion:

            if use_Lya_force:
                data["f_Lya_r"] = read_optional_time_dataset(
                    file,
                    f_Lya_r_dataset_candidates,
                    sl,
                    Nt_raw,
                    Nr,
                    NZ
                )

                data["f_Lya_Z"] = read_optional_time_dataset(
                    file,
                    f_Lya_Z_dataset_candidates,
                    sl,
                    Nt_raw,
                    Nr,
                    NZ
                )
            else:
                data["f_Lya_r"] = None
                data["f_Lya_Z"] = None

            if use_direct_force:
                data["f_direct_r"] = read_optional_time_dataset(
                    file,
                    f_direct_r_dataset_candidates,
                    sl,
                    Nt_raw,
                    Nr,
                    NZ
                )

                data["f_direct_Z"] = read_optional_time_dataset(
                    file,
                    f_direct_Z_dataset_candidates,
                    sl,
                    Nt_raw,
                    Nr,
                    NZ
                )
            else:
                data["f_direct_r"] = None
                data["f_direct_Z"] = None

            f_gravity_r_name = get_first_existing_dataset_name(
                file,
                f_gravity_r_dataset_candidates,
                required=True
            )

            data["f_gravity_r"] = read_time_dataset(
                file,
                f_gravity_r_name,
                sl,
                Nt_raw,
                Nr,
                NZ
            )

            f_gravity_Z_name = get_first_existing_dataset_name(
                file,
                f_gravity_Z_dataset_candidates,
                required=True
            )

            data["f_gravity_Z"] = read_time_dataset(
                file,
                f_gravity_Z_name,
                sl,
                Nt_raw,
                Nr,
                NZ
            )

        # Needed for M_SF, even if M_tot is read directly.
        data["CellVol"] = read_cell_volumes(file, Nr, NZ)

        if use_saved_total_gas_mass:
            total_mass_name = get_first_existing_dataset_name(
                file,
                total_gas_mass_dataset_candidates,
                required=True
            )

            data["M_tot"] = symmetry_factor * read_time_dataset(
                file,
                total_mass_name,
                sl,
                Nt_raw,
                Nr,
                NZ
            ) / Msun

        if use_xHII_criterion:
            xHII_name = get_first_existing_dataset_name(
                file,
                xHII_dataset_candidates,
                required=True
            )

            data["xHII"] = read_time_dataset(
                file,
                xHII_name,
                sl,
                Nt_raw,
                Nr,
                NZ
            )
        else:
            data["xHII"] = None

        if use_T_criterion:
            T_name = get_first_existing_dataset_name(
                file,
                temperature_dataset_candidates,
                required=True
            )

            data["T"] = read_time_dataset(
                file,
                T_name,
                sl,
                Nt_raw,
                Nr,
                NZ
            )
        else:
            data["T"] = None

    if not use_saved_total_gas_mass:
        data["M_tot"] = compute_total_gas_mass_history(data)

    data["M_SF"] = compute_star_forming_gas_mass_history(data)

    if print_mass_conservation_diagnostic:
        M0 = data["M_tot"][0]
        Mmin = np.nanmin(data["M_tot"])
        Mmax = np.nanmax(data["M_tot"])

        if M0 > 0.0:
            print(
                f"{filename}: Mgas/M0 range = "
                f"{Mmin/M0:.6e} -- {Mmax/M0:.6e}"
            )
        else:
            print(f"{filename}: Mgas[0] <= 0, cannot print Mgas/M0 diagnostic.")

    return data


def compute_total_gas_mass_history(data):
    """Compute total gas mass inside the simulation volume.

        M_tot(t) = sum rho(i,j,t) * dV(i,j)

    This is only used if use_saved_total_gas_mass=False.
    """

    rho     = np.asarray(data["rho"])
    CellVol = np.asarray(data["CellVol"])

    if rho.shape[:2] != CellVol.shape:
        raise ValueError(
            f"Shape mismatch in total gas mass diagnostic: "
            f"rho[:, :, 0] has shape {rho.shape[:2]}, "
            f"CellVol has shape {CellVol.shape}."
        )

    Nt = rho.shape[2]
    M_tot = np.zeros(Nt)

    for n in range(Nt):
        M_tot[n] = selected_gas_mass(
            rho[:, :, n],
            CellVol,
            np.ones(CellVol.shape, dtype=bool),
            symmetry_factor=symmetry_factor,
        ) / Msun

    return M_tot


def anti_gravity_eddington_ratio(f_rad_r, f_rad_Z, f_gravity_r, f_gravity_Z):
    """Compute anti-gravity Eddington ratio.

        f_Edd,anti = - f_grav · f_rad / |f_grav|^2

    Positive values mean radiation opposes gravity.
    Negative values mean radiation partly points along gravity.
    """

    f_rad_r     = np.float64(f_rad_r)
    f_rad_Z     = np.float64(f_rad_Z)
    f_gravity_r = np.float64(f_gravity_r)
    f_gravity_Z = np.float64(f_gravity_Z)

    f_grav_sq = f_gravity_r**2 + f_gravity_Z**2

    return -(f_gravity_r * f_rad_r + f_gravity_Z * f_rad_Z) / f_grav_sq


def feedback_eddington_mask(data, n):
    """Return mask for cells where feedback is weaker than gravity.

    The radiation force is the selected sum of Lyα and direct stellar forces.
    Missing optional Lyα/direct fields are treated as zero, which is useful for
    no-Lyα runs.
    """

    rho_shape = data["rho"][:, :, n].shape

    f_rad_r = np.zeros(rho_shape)
    f_rad_Z = np.zeros(rho_shape)

    if use_Lya_force:
        if data["f_Lya_r"] is not None:
            f_rad_r += data["f_Lya_r"][:, :, n]
        if data["f_Lya_Z"] is not None:
            f_rad_Z += data["f_Lya_Z"][:, :, n]

    if use_direct_force:
        if data["f_direct_r"] is not None:
            f_rad_r += data["f_direct_r"][:, :, n]
        if data["f_direct_Z"] is not None:
            f_rad_Z += data["f_direct_Z"][:, :, n]

    f_gravity_r = data["f_gravity_r"][:, :, n]
    f_gravity_Z = data["f_gravity_Z"][:, :, n]

    f_grav_mag = np.sqrt(np.float64(f_gravity_r)**2 + np.float64(f_gravity_Z)**2)
    valid_gravity = f_grav_mag > f_grav_min

    f_Edd = anti_gravity_eddington_ratio(
        f_rad_r,
        f_rad_Z,
        f_gravity_r,
        f_gravity_Z,
    )

    return valid_gravity & (f_Edd < f_Edd_crit)


def compute_star_forming_gas_mass_history(data):
    """Compute mass of gas satisfying the selected star-formation criteria."""

    rho     = np.asarray(data["rho"])
    mom_r   = np.asarray(data["mom_r"])
    mom_Z   = np.asarray(data["mom_Z"])
    CellVol = np.asarray(data["CellVol"])

    if rho.shape[:2] != CellVol.shape:
        raise ValueError(
            f"Shape mismatch in star-forming gas mass diagnostic: "
            f"rho[:, :, 0] has shape {rho.shape[:2]}, "
            f"CellVol has shape {CellVol.shape}."
        )

    if use_self_gravity_criterion:
        P = np.asarray(data["P"])
    else:
        P = None

    rCoM    = data["rCoM"]
    ZCoM    = data["ZCoM"]
    delta_r = data["delta_r"]
    delta_Z = data["delta_Z"]

    xHII = data["xHII"]
    T    = data["T"]

    Nt = rho.shape[2]
    M_SF = np.zeros(Nt)

    for n in range(Nt):

        v_r, v_Z = velocity_from_momentum(
            mom_r[:, :, n],
            mom_Z[:, :, n],
            rho[:, :, n],
        )

        mask = np.ones(rho[:, :, n].shape, dtype=bool)

        if use_feedback_eddington_criterion:
            mask &= feedback_eddington_mask(data, n)

        if use_converging_flow_criterion:
            mask &= converging_flow_mask(v_r, v_Z, rCoM, ZCoM)

        if use_self_gravity_criterion:
            mask &= self_gravitating_mask(
                rho[:, :, n],
                P[:, :, n],
                v_r,
                v_Z,
                rCoM,
                ZCoM,
                delta_r,
                delta_Z,
                gamma=gamma_ad,
                alpha_crit=alpha_crit,
            )

        if use_nH_criterion:
            nH = hydrogen_mass_fraction * rho[:, :, n] / mH
            mask &= (nH > nH_crit)

        if use_xHII_criterion:
            mask &= (xHII[:, :, n] < xHII_crit)

        if use_T_criterion:
            mask &= (T[:, :, n] < T_crit)

        M_SF[n] = selected_gas_mass(
            rho[:, :, n],
            CellVol,
            mask,
            symmetry_factor=symmetry_factor,
        ) / Msun

    return M_SF

def clean_for_log(x, y):
    """Mask non-positive values before plotting on a log axis."""

    x = np.asarray(x)
    y = np.asarray(y)

    if mask_nonpositive_for_log:
        good = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    else:
        good = np.isfinite(x) & np.isfinite(y)

    return x[good], y[good]


def plot_history(ax, t, y, sim, use_log=True):
    """Plot one history curve with your preferred line settings."""

    if use_log:
        t_plot, y_plot = clean_for_log(t / kyr, y)
        ax.semilogy(t_plot, y_plot,
                    linewidth = sim["linewidth"],
                    color = sim["color"],
                    linestyle = sim["linestyle"],
                    alpha = sim["alpha"],
                    dash_capstyle = 'round',
                    label = sim["label"])
    else:
        ax.plot(t / kyr, y,
                linewidth = sim["linewidth"],
                color = sim["color"],
                linestyle = sim["linestyle"],
                alpha = sim["alpha"],
                dash_capstyle = 'round',
                label = sim["label"])


def format_axis(ax):
    """Apply common tick formatting."""

    ax.minorticks_on()

    # More minor ticks on the linear mass axis.
    # y_minor_subdivisions = 10 gives 9 minor ticks between major ticks.
    ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor_subdivisions))

    ax.tick_params(axis='both', which='major', direction='in',
                   length=20, width=2.5)
    ax.tick_params(axis='both', which='minor', direction='in',
                   length=10, width=2.5)

    for spine in ax.spines.values():
        spine.set_linewidth(2.5)


def get_panel_label_position(panel_name):
    """Return panel-label position for a given named panel."""

    if panel_name in panel_label_positions:
        return panel_label_positions[panel_name]

    return panel_label_x_default, panel_label_y_default


def add_panel_label(ax, panel_name, label):
    """Add panel label using panel-specific placement."""

    x, y = get_panel_label_position(panel_name)

    ax.text(x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=panel_label_fontsize,
            color="black")


def add_feedback_text_label(ax, key):
    """Add a manually placed text label in axes coordinates."""

    label = feedback_text_labels[key]
    x, y  = label["position"]

    ax.text(x,
            y,
            label["text"],
            transform=ax.transAxes,
            fontsize=feedback_text_fontsize,
            color=feedback_text_color,
            rotation=label["rotation"],
            ha=label["ha"],
            va=label["va"],
            path_effects=feedback_text_path_effects)


def scaled_mass_tick_formatter(y, pos):
    """Format mass-axis tick labels in units of mass_tick_scale."""

    scaled = y / mass_tick_scale

    # Avoid printing -0.
    if abs(scaled) < 1e-12:
        scaled = 0.0

    return f"{scaled:g}"


def format_log_tick(val):
    """Format a power-of-10 tick value."""

    import math

    if val <= 0:
        return ''

    log_val = math.log10(val)
    k = round(log_val)

    if abs(log_val - k) > 1e-9:
        return f'{val:g}'

    if abs(k) < 3:
        return f'{val:g}'
    else:
        return r'$10^{' + str(k) + r'}$'


def set_log_ticks(ax, ylim):
    """Set major log ticks and labels for every decade within ylim."""

    import math

    ymin, ymax = ylim

    if ymin <= 0.0:
        raise ValueError("Log-axis limits must be positive.")

    k_min = math.ceil(math.log10(ymin)  - 1e-9)
    k_max = math.floor(math.log10(ymax) + 1e-9)

    ticks  = [10.0**k for k in range(k_min, k_max + 1)]
    labels = [format_log_tick(v) for v in ticks]

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


def apply_ylim_and_ticks(ax, ylim, use_log):
    """Apply optional y-limits and custom log ticks."""

    if ylim is None:
        return

    if use_log:
        set_log_ticks(ax, ylim)

    ax.set_ylim(ylim)


def format_number_for_math_label(x):
    """Format numbers for plot labels.

    Examples:
        1e7   -> 10^{7}
        3e7   -> 3\times 10^{7}
        1000  -> 10^{3}
        0.5   -> 0.5
    """

    x = float(x)

    if not np.isfinite(x):
        return f"{x:g}"

    if x == 0.0:
        return "0"

    abs_x = abs(x)
    exponent = int(np.floor(np.log10(abs_x)))
    mantissa = x / 10.0**exponent

    # Use normal decimal/integer notation when it remains compact.
    if -2 <= exponent < 3:
        return f"{x:g}"

    if np.isclose(mantissa, 1.0):
        return rf"10^{{{exponent}}}"

    return rf"{mantissa:g}\times 10^{{{exponent}}}"


def criterion_label():
    """Create a compact label for the star-forming gas criterion."""

    parts = []

    if use_feedback_eddington_criterion:
        parts.append(
            r"$f_{Edd,antigrav}^{Ly\alpha+dir} < "
            + format_number_for_math_label(f_Edd_crit)
            + r"$"
        )

    if use_converging_flow_criterion:
        parts.append(r"$\nabla\cdot v < 0$")

    if use_self_gravity_criterion:
        parts.append(
            r"$\alpha_{vir} < "
            + format_number_for_math_label(alpha_crit)
            + r"$"
        )

    if use_nH_criterion:
        parts.append(
            r"$n_H > "
            + format_number_for_math_label(nH_crit)
            + r"$ cm$^{-3}$"
        )

    if use_xHII_criterion:
        parts.append(
            r"$x_{HII} < "
            + format_number_for_math_label(xHII_crit)
            + r"$"
        )

    if use_T_criterion:
        parts.append(
            r"$T < "
            + format_number_for_math_label(T_crit)
            + r"$ K"
        )

    if len(parts) == 0:
        return r"$M_{gas}$ selected"

    return ", ".join(parts)




###############################################################
#######                                                 #######
#######               P L O T  I T !                    #######
#######                                                 #######
###############################################################

fig, ax = plt.subplots(2, 1, sharex=True)
fig.set_size_inches(fig_width, fig_height)

plt.subplots_adjust(wspace=wspace, hspace=hspace)

# Panel layout.
ax_M_SF  = ax[0]
ax_M_tot = ax[1]

panel_axes = {
    "M_SF"  : ax_M_SF,
    "M_tot" : ax_M_tot,
}

# Plot each resolution in the same color.
#   Lyα feedback:          solid
#   Ignoring Lyα feedback: dashed, with rounded caps
for sim_Lya, sim_noLya in zip(simulations_Lya, simulations_noLya):

    data_Lya = read_mass_history(sim_Lya["filename"])
    t_Lya    = data_Lya["t"]

    data_noLya = read_mass_history(sim_noLya["filename"])
    t_noLya    = data_noLya["t"]

    plot_history(ax_M_SF,
                 t_Lya,
                 data_Lya["M_SF"],
                 sim_Lya,
                 use_log=use_log_M_SF_Lya)

    plot_history(ax_M_tot,
                 t_Lya,
                 data_Lya["M_tot"],
                 sim_Lya,
                 use_log=use_log_M_tot_Lya)

    plot_history(ax_M_SF,
                 t_noLya,
                 data_noLya["M_SF"],
                 sim_noLya,
                 use_log=use_log_M_SF_noLya)

    plot_history(ax_M_tot,
                 t_noLya,
                 data_noLya["M_tot"],
                 sim_noLya,
                 use_log=use_log_M_tot_noLya)


###############################################################
#######                                                 #######
#######              A X I S   L A B E L S              #######
#######                                                 #######
###############################################################

ax_M_tot.set_xlabel("Time (kyr)", labelpad=20)

if use_scaled_mass_ticklabels:
    ax_M_SF.set_ylabel(r"Mass (" + mass_axis_unit_label + r")", labelpad=20)
    ax_M_tot.set_ylabel(r"Mass (" + mass_axis_unit_label + r")", labelpad=20)
else:
    ax_M_SF.set_ylabel(r"Mass (M$_{\odot}$)", labelpad=20)
    ax_M_tot.set_ylabel(r"Mass (M$_{\odot}$)", labelpad=20)

if xlim is not None:
    for axis in ax:
        axis.set_xlim(xlim)

apply_ylim_and_ticks(ax_M_SF,
                     ylim_M_SF,
                     use_log_M_SF_Lya)

apply_ylim_and_ticks(ax_M_tot,
                     ylim_M_tot,
                     use_log_M_tot_Lya)

if use_scaled_mass_ticklabels:
    ax_M_SF.yaxis.set_major_formatter(FuncFormatter(scaled_mass_tick_formatter))
    ax_M_tot.yaxis.set_major_formatter(FuncFormatter(scaled_mass_tick_formatter))


###############################################################
#######                                                 #######
#######             P A N E L   L A B E L S             #######
#######                                                 #######
###############################################################

if show_panel_labels:

    add_panel_label(ax_M_SF,
                    "M_SF",
                    r"Candidate star-forming gas mass")

    add_panel_label(ax_M_tot,
                    "M_tot",
                    r"Total gas mass")


sf_text = criterion_label()

#ax_M_SF.text(0.07,
#             0.72,
#             sf_text,
#             transform=ax_M_SF.transAxes,
#             fontsize=21,
#             color="black")

if show_feedback_text_labels:

    add_feedback_text_label(ax_M_SF, "with_Lya")
    add_feedback_text_label(ax_M_SF, "without_Lya")


###############################################################
#######                                                 #######
#######          T I C K S,  L E G E N D,  S A V E      #######
#######                                                 #######
###############################################################

for axis in ax:
    format_axis(axis)

if legend_panel not in panel_axes:
    valid_names = ", ".join(panel_axes.keys())
    raise ValueError(
        f"legend_panel = '{legend_panel}' is not valid. "
        f"Valid options are: {valid_names}."
    )

legend_axis = panel_axes[legend_panel]

# One compact legend: color gives resolution.
# The line-style meaning is labelled manually in the top panel.
resolution_handles = [
    Line2D([0], [0],
           color=sim["color"],
           lw=sim["linewidth"],
           ls="-",
           label=sim["label"])
    for sim in simulations_Lya
]

legend_handles = resolution_handles

if legend_bbox_to_anchor is None:
    legend_axis.legend(handles=legend_handles,
                       loc=legend_loc,
                       fontsize=legend_fontsize,
                       ncol=legend_ncol,
                       frameon=False,
                       columnspacing=0.6)
else:
    legend_axis.legend(handles=legend_handles,
                       loc=legend_loc,
                       bbox_to_anchor=legend_bbox_to_anchor,
                       fontsize=legend_fontsize,
                       ncol=legend_ncol,
                       frameon=False,
                       columnspacing=0.6)

plt.savefig(outfile, bbox_inches="tight")
plt.show()
