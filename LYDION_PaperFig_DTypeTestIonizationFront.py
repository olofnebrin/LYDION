import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter, FuncFormatter

# ===============================================================
#                       Constants
# ===============================================================

yr  = 3.1556926e7
kyr = 1e3 * yr
Myr = 1e6 * yr
pc  = 3.086e18
kms = 1e5

# ===============================================================
#                       User inputs
# ===============================================================

#filename = "DTypeTest.h5"
filename = "DTypeTest_128x128_RSL0.01_sigmastar3.0_OTStrue.h5"
outname  = "DTypeTest_IFront_Evolution.pdf"
savefig  = True

color_LYDION = "dodgerblue"

# C2-Ray / Iliev et al. Test 5 comparison data.
# Columns: time [s], I-front radius [cm], I-front velocity [cm/s].
plot_C2Ray        = True
C2Ray_filename    = "T5_front_mellema.dat"
C2Ray_label       = r"Capreole + C$^2$-Ray"
color_C2Ray       = "crimson"


# I-front definition:
#   "xHII" : front where ionized fraction decreases through IF_threshold
#   "xHI"  : front where neutral fraction increases through IF_threshold
IF_fraction  = "xHII"     # choose "xHII" or "xHI"
IF_threshold = 0.5

# Source position. If None, tries to read Settings["Z_pos"], otherwise uses 0.
Z_source_pc = 0.0

# Radial bin width for spherical profiles.
# If None, uses roughly one spatial cell width.
radial_bin_width_pc = None

# Use only complete spherical shells, i.e. avoid radii where the shell hits
# the edge of the 2D domain. Recommended for clean radial averages.
use_complete_shell_limit = True

# Spherical profile weighting used before locating the I-front.
# Keep this consistent with LYDION_PaperFig_DTypeProfiles.py.
#
# "angle"  : polar-angle average, each direction counts more equally.
# "volume" : true 3D axisymmetric volume average, equator-weighted.
# "cell"   : each selected 2D cell has equal weight.
profile_weighting = "angle"

# Plot velocity in bottom panel.
plot_velocity = True

# Smoothing for velocity only. Set to 1 for no smoothing.
velocity_smoothing_window = 1

# Put velocity axis in log-scale, as in the AREPO-RT paper.
velocity_yscale_log = True

# Normalize I-front radius by user-specified Strömgren radius.
normalize_radius_by_rs = True
rs_pc                  = 5.4e3

# Optional horizontal reference line at R_IF / r_s = 1, or R_IF = r_s.
plot_stromgren_radius = False

# Optional axis limits.
# xlim is always in Myr.
# ylim_R is in pc if normalize_radius_by_rs = False, otherwise dimensionless R_IF/r_s.
# ylim_v is always in km/s.
xlim   = (-20,520)     # e.g. (0.0, 500.0)
ylim_R = (0.0, 1.9)    # e.g. (0.0, 1.2)
ylim_v = None          # e.g. (1e-1, 1e2) for log velocity

# Optional threshold label.
show_threshold_label = False
threshold_label_position = (0.04, 0.93)   # axes coordinates
threshold_label_fontsize = 24

# Legend options.
show_legend = True
legend_loc = "lower right"
legend_bbox_to_anchor = (0.97, 0.08)

# Minor tick options.
use_minor_ticks = True

# For linear axes, this is the number of subdivisions per major interval.
# Example: 5 gives 4 minor ticks between major ticks.
x_minor_subdivisions = 10
R_minor_subdivisions = 5
v_minor_subdivisions = 5

# For log velocity axis, minor ticks at 2,3,...,9 times each decade.
log_minor_subs = np.arange(2, 10) * 0.1


# ===============================================================
#                    Plot cosmetics
# ===============================================================

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 34})
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

params = {'legend.fontsize': 36}
plt.rcParams.update(params)
plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 1.7
plt.rcParams['xtick.minor.width'] = 1.7
plt.rcParams['xtick.major.size'] = 18
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size'] = 18
plt.rcParams['ytick.minor.size'] = 8
plt.rcParams['xtick.major.pad'] = '12'
plt.rcParams['ytick.major.pad'] = '12'

plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linewidth'] = 2.5
plt.rcParams['grid.linestyle'] = '--'

plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


# ===============================================================
#                    Helper functions
# ===============================================================

def decode_attr(x):
    """Make HDF5 attributes robust to bytes/string/numeric storage."""
    if isinstance(x, bytes):
        return x.decode()
    return x


def read_attr_float(group, key, default=None):
    """Read a scalar attribute that may have been saved as string."""
    if key not in group.attrs:
        return default
    val = decode_attr(group.attrs[key])
    try:
        return float(val)
    except Exception:
        return default


def ensure_NtNrNz(arr, Nt, Nr, Nz, name="array"):
    """
    Return array in shape (Nt, Nr, Nz).

    LYDION save code usually creates histories as (Nr, Nz, Nsaved),
    while some plotting scripts may assume (Nt, Nr, Nz). This handles both.
    """
    shape = arr.shape

    if shape == (Nt, Nr, Nz):
        return arr

    if shape == (Nr, Nz, Nt):
        return arr.transpose(2, 0, 1)

    if shape == (Nt, Nz, Nr):
        return arr.transpose(0, 2, 1)

    if shape == (Nz, Nr, Nt):
        return arr.transpose(2, 1, 0)

    raise ValueError(
        f"Unexpected shape for {name}: {shape}. "
        f"Expected one of (Nt,Nr,Nz)=({Nt},{Nr},{Nz}) or "
        f"(Nr,Nz,Nt)=({Nr},{Nz},{Nt})."
    )


def ensure_NrNz(arr, Nr, Nz, name="array"):
    """Return 2D array in shape (Nr, Nz)."""
    if arr.shape == (Nr, Nz):
        return arr
    if arr.shape == (Nz, Nr):
        return arr.T
    raise ValueError(f"Unexpected shape for {name}: {arr.shape}, expected ({Nr},{Nz}).")


def moving_average_nan(y, window):
    """Simple NaN-aware moving average for smoother velocity display."""
    if window <= 1:
        return y

    window = int(window)
    if window % 2 == 0:
        window += 1

    y_smooth = np.full_like(y, np.nan, dtype=float)
    half = window // 2

    for i in range(len(y)):
        i0 = max(0, i - half)
        i1 = min(len(y), i + half + 1)
        vals = y[i0:i1]
        good = np.isfinite(vals)
        if np.any(good):
            y_smooth[i] = np.mean(vals[good])

    return y_smooth


def compute_cell_angle_widths(r_edges_pc, Z_edges_pc, Z_source_pc):
    """
    Approximate each 2D cell's polar-angle coverage around the source.

    The returned weights are Δθ per cell. Therefore, within each radial shell,
    averaging with these weights approximates an angle average rather than the
    true 3D volume average. This is intentionally different from volume
    weighting, which contains the physical sin(theta) factor.
    """
    Nr = len(r_edges_pc) - 1
    Nz = len(Z_edges_pc) - 1
    weights = np.zeros((Nr, Nz), dtype=float)

    for i in range(Nr):
        r0 = r_edges_pc[i]
        r1 = r_edges_pc[i + 1]

        for j in range(Nz):
            z0 = Z_edges_pc[j] - Z_source_pc
            z1 = Z_edges_pc[j + 1] - Z_source_pc

            # atan2(r, z) gives polar angle measured away from the +Z axis.
            # For r >= 0 this lies in [0, pi], so there is no periodic wrap.
            theta_corners = np.array([
                np.arctan2(r0, z0),
                np.arctan2(r1, z0),
                np.arctan2(r0, z1),
                np.arctan2(r1, z1),
            ])

            dtheta = np.nanmax(theta_corners) - np.nanmin(theta_corners)

            if np.isfinite(dtheta) and dtheta > 0.0:
                weights[i, j] = dtheta

    return weights


def velocity_log_tick_formatter(y, pos):
    """
    Format velocity log-axis ticks.

    Use plain decimals/integers below 100, and powers of ten only
    for 100 and above. This avoids labels such as 10^0 and 10^1.
    """
    if not np.isfinite(y) or y <= 0.0:
        return ""

    if y < 100.0:
        return f"{y:g}"

    exponent = int(np.round(np.log10(y)))

    # Only label exact powers of ten.
    if np.isclose(y, 10.0**exponent):
        return rf"$10^{{{exponent}}}$"

    return ""


def read_C2Ray_Test5_front(filename, pc, Myr, kms, rs_pc=None, normalize_radius_by_rs=False):
    """
    Read Mellema/C2-Ray Iliev Test 5 I-front file.

    Expected columns are:
      1: time [s]
      2: I-front radius [cm]
      3: I-front velocity [cm/s]

    Lines starting with # are ignored by np.loadtxt.
    """
    data = np.loadtxt(filename, comments="#")

    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] < 2:
        raise ValueError(
            f"Expected at least two columns in {filename}: time [s], R_IF [cm]."
        )

    t_C2Ray_Myr = data[:, 0] / Myr
    R_C2Ray_pc  = data[:, 1] / pc

    if data.shape[1] >= 3:
        v_C2Ray_kms = data[:, 2] / kms
    else:
        v_C2Ray_kms = np.full_like(t_C2Ray_Myr, np.nan, dtype=float)

    if normalize_radius_by_rs:
        if rs_pc is None:
            raise ValueError("rs_pc must be provided when normalize_radius_by_rs=True.")
        R_C2Ray_plot = R_C2Ray_pc / rs_pc
    else:
        R_C2Ray_plot = R_C2Ray_pc

    return t_C2Ray_Myr, R_C2Ray_plot, v_C2Ray_kms


def find_front_radius(shell_r_pc, xprof, threshold=0.5, direction="decreasing"):
    """
    Find radius where xprof crosses threshold.

    direction = "decreasing":
        high inside -> low outside, e.g. xHII in an expanding H II region.

    direction = "increasing":
        low inside -> high outside, e.g. xHI in an expanding H II region.

    We use the outermost crossing, which is usually the desired I-front.
    """
    good = np.isfinite(shell_r_pc) & np.isfinite(xprof)
    r = shell_r_pc[good]
    x = xprof[good]

    if len(r) < 2:
        return np.nan

    order = np.argsort(r)
    r = r[order]
    x = x[order]

    if direction == "decreasing":
        crossing_indices = np.where((x[:-1] >= threshold) & (x[1:] < threshold))[0]

    elif direction == "increasing":
        crossing_indices = np.where((x[:-1] < threshold) & (x[1:] >= threshold))[0]

    else:
        raise ValueError("direction must be either 'decreasing' or 'increasing'.")

    if len(crossing_indices) == 0:
        return np.nan

    k = crossing_indices[-1]

    r1, r2 = r[k], r[k + 1]
    x1, x2 = x[k], x[k + 1]

    if x2 == x1:
        return 0.5 * (r1 + r2)

    return r1 + (threshold - x1) * (r2 - r1) / (x2 - x1)


def spherical_average_profile(q, shell_index, shell_r_pc, weights, nbins):
    """
    Weighted spherical average of q in cylindrical 2D.

    q should have shape (Nr, Nz), and shell_index maps each cell to a
    spherical shell bin.

    weights should have the same shape as q.  Pass CellVolumes for a
    volume-weighted average (equator-biased in cylindrical geometry), or
    SolidAngleWeights = dr_i * dZ_j / R_ij for an unbiased 3D spherical
    average that weights uniformly in cos(theta).
    """
    q_flat = q.ravel()
    w_flat = weights.ravel()
    b_flat = shell_index.ravel()

    good = (
        np.isfinite(q_flat)
        & np.isfinite(w_flat)
        & (w_flat > 0.0)
        & (b_flat >= 0)
        & (b_flat < nbins)
    )

    sum_w  = np.bincount(b_flat[good], weights=w_flat[good], minlength=nbins)
    sum_qw = np.bincount(b_flat[good], weights=q_flat[good] * w_flat[good], minlength=nbins)

    prof = np.full(nbins, np.nan)
    nonzero = sum_w > 0.0
    prof[nonzero] = sum_qw[nonzero] / sum_w[nonzero]

    return shell_r_pc, prof


# ===============================================================
#                  Validate I-front definition
# ===============================================================

IF_fraction = IF_fraction.strip()

if IF_fraction not in ("xHII", "xHI"):
    raise ValueError("IF_fraction must be either 'xHII' or 'xHI'.")

if IF_fraction == "xHII":
    IF_crossing_direction = "decreasing"
else:
    IF_crossing_direction = "increasing"


# ===============================================================
#                         Read data
# ===============================================================

with h5py.File(filename, "r") as f:

    t = f["Times"][:]

    r = f["r"][:]        # cell interfaces
    Z = f["Z"][:]        # cell interfaces

    Nr = len(r) - 1
    Nz = len(Z) - 1
    Nt = len(t)

    if "rCoM" in f:
        rCoM = f["rCoM"][:]
    else:
        # Fallback only. In cylindrical coordinates this is not exactly
        # the volume centroid, but better than failing.
        rCoM = 0.5 * (r[:-1] + r[1:])
        print("Warning: rCoM not found in file, using cell-centered r as fallback.")

    if "ZCoM" in f:
        ZCoM = f["ZCoM"][:]
    else:
        ZCoM = 0.5 * (Z[:-1] + Z[1:])

    if "CellVolumes" in f:
        CellVolumes = ensure_NrNz(f["CellVolumes"][:], Nr, Nz, name="CellVolumes")
    else:
        # Fallback for axisymmetric cylindrical cells.
        # The overall 2π factor cancels in the volume-weighted average.
        dr2 = 0.5 * (r[1:]**2 - r[:-1]**2)
        dZ  = Z[1:] - Z[:-1]
        CellVolumes = np.outer(dr2, dZ)

    xHII_hist = ensure_NtNrNz(
        f["xHII_History"][:], Nt, Nr, Nz, name="xHII_History"
    )

    xH2_hist = ensure_NtNrNz(
        f["xH2_History"][:], Nt, Nr, Nz, name="xH2_History"
    )

    xHI_hist = 1.0 - 2.0 * xH2_hist - xHII_hist

    if "Settings" in f and Z_source_pc is None:
        Z_pos_code = read_attr_float(f["Settings"], "Z_pos", default=None)
        if Z_pos_code is not None:
            # Z_pos was saved in code units, i.e. cm.
            Z_source_pc = Z_pos_code / pc

if Z_source_pc is None:
    Z_source_pc = 0.0


# Keep only actually saved outputs.
saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]

t_saved = t[saved_idx]
xHII_hist = xHII_hist[saved_idx, :, :]
xHI_hist  = xHI_hist[saved_idx, :, :]


# ===============================================================
#          Construct spherical shell binning once
# ===============================================================

r_pc = r / pc
Z_pc = Z / pc
rCoM_pc = rCoM / pc
ZCoM_pc = ZCoM / pc

rr, zz = np.meshgrid(rCoM_pc, ZCoM_pc, indexing="ij")
R_sph_pc = np.sqrt(rr**2 + (zz - Z_source_pc)**2)

if radial_bin_width_pc is None:
    dr_med_pc = np.median(np.diff(r_pc))
    dZ_med_pc = np.median(np.diff(Z_pc))
    radial_bin_width_pc = min(dr_med_pc, dZ_med_pc)

if use_complete_shell_limit:
    r_extent_pc = np.max(r_pc)

    zmin_pc = np.min(Z_pc)
    zmax_pc = np.max(Z_pc)

    if (Z_source_pc > zmin_pc) and (Z_source_pc < zmax_pc):
        z_extent_pc = min(Z_source_pc - zmin_pc, zmax_pc - Z_source_pc)
    else:
        z_extent_pc = max(abs(zmax_pc - Z_source_pc), abs(Z_source_pc - zmin_pc))

    Rmax_profile_pc = min(r_extent_pc, z_extent_pc)
else:
    Rmax_profile_pc = np.nanmax(R_sph_pc)

shell_edges_pc = np.arange(0.0, Rmax_profile_pc + radial_bin_width_pc, radial_bin_width_pc)
shell_centers_pc = 0.5 * (shell_edges_pc[:-1] + shell_edges_pc[1:])
nbins = len(shell_centers_pc)

shell_index = np.digitize(R_sph_pc, shell_edges_pc) - 1
shell_index[(R_sph_pc < shell_edges_pc[0]) | (R_sph_pc >= shell_edges_pc[-1])] = -1

# Choose the same spherical weighting convention as the D-type profile plot.
if profile_weighting == "volume":
    ProfileWeights = CellVolumes
elif profile_weighting == "angle":
    ProfileWeights = compute_cell_angle_widths(r_pc, Z_pc, Z_source_pc)
elif profile_weighting == "cell":
    ProfileWeights = np.ones_like(CellVolumes, dtype=float)
else:
    raise ValueError(
        "profile_weighting must be one of 'volume', 'angle', or 'cell'."
    )


# ===============================================================
#             Compute I-front radius
# ===============================================================

R_IF_pc = np.full(len(t_saved), np.nan)

for n in range(len(t_saved)):

    if IF_fraction == "xHII":

        shell_r_pc, xIF_prof = spherical_average_profile(
            xHII_hist[n, :, :],
            shell_index,
            shell_centers_pc,
            ProfileWeights,
            nbins
        )

    elif IF_fraction == "xHI":

        shell_r_pc, xIF_prof = spherical_average_profile(
            xHI_hist[n, :, :],
            shell_index,
            shell_centers_pc,
            ProfileWeights,
            nbins
        )

    R_IF_pc[n] = find_front_radius(
        shell_r_pc,
        xIF_prof,
        threshold=IF_threshold,
        direction=IF_crossing_direction
    )


# ===============================================================
#             Compute I-front velocity
# ===============================================================

t_Myr = t_saved / Myr

R_IF_cm = R_IF_pc * pc
v_IF_kms = np.full_like(R_IF_pc, np.nan, dtype=float)

good = np.isfinite(t_saved) & np.isfinite(R_IF_pc)

if np.count_nonzero(good) >= 3:
    v_tmp = np.gradient(R_IF_cm[good], t_saved[good]) / kms
    v_IF_kms[good] = v_tmp

elif np.count_nonzero(good) == 2:
    inds = np.where(good)[0]
    v_val = (
        (R_IF_cm[inds[1]] - R_IF_cm[inds[0]])
        / (t_saved[inds[1]] - t_saved[inds[0]])
        / kms
    )
    v_IF_kms[inds] = v_val

v_IF_plot = moving_average_nan(v_IF_kms, velocity_smoothing_window)


# ===============================================================
#             Apply plotting normalization
# ===============================================================

if normalize_radius_by_rs:
    R_IF_plot = R_IF_pc / rs_pc
    R_ylabel  = "$R_{IF}/R_{Ström}$"
    rs_line_y = 1.0
else:
    R_IF_plot = R_IF_pc
    R_ylabel  = "$R_{IF}$ (pc)"
    rs_line_y = rs_pc


# ===============================================================
#             Read C2-Ray / Iliev Test 5 comparison data
# ===============================================================

if plot_C2Ray:
    t_C2Ray_Myr, R_C2Ray_plot, v_C2Ray_kms = read_C2Ray_Test5_front(
        C2Ray_filename,
        pc=pc,
        Myr=Myr,
        kms=kms,
        rs_pc=rs_pc,
        normalize_radius_by_rs=normalize_radius_by_rs,
    )
else:
    t_C2Ray_Myr  = None
    R_C2Ray_plot = None
    v_C2Ray_kms  = None


# ===============================================================
#                         Plot
# ===============================================================

if plot_velocity:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 12), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4], "hspace": 0.05}
    )
else:
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = None


# --- I-front radius ---
ax1.plot(t_Myr, R_IF_plot, lw=5.0, color=color_LYDION, label="LYDION")

if plot_C2Ray:
    ax1.plot(
        t_C2Ray_Myr, R_C2Ray_plot,
        lw=5.0, ls="--", dash_capstyle="round", color=color_C2Ray,
        label=C2Ray_label
    )

if plot_stromgren_radius:
    ax1.axhline(rs_line_y, lw=2.5, ls=":", color="0.35", label="$r_s$")

ax1.set_ylabel(R_ylabel, labelpad=14)
ax1.grid(False)

if show_legend:
    ax1.legend(
    loc=legend_loc,
    bbox_to_anchor=legend_bbox_to_anchor,
    borderaxespad=0.0
)


# --- I-front velocity ---
if plot_velocity:

    if velocity_yscale_log:
        v_IF_plot_for_plot = np.where(v_IF_plot > 0.0, v_IF_plot, np.nan)
        ax2.plot(t_Myr, v_IF_plot_for_plot, lw=5.0, color=color_LYDION)
        if plot_C2Ray:
            v_C2Ray_plot = np.where(v_C2Ray_kms > 0.0, v_C2Ray_kms, np.nan)
            ax2.plot(t_C2Ray_Myr, v_C2Ray_plot, lw=5.0, ls="--", dash_capstyle="round", color=color_C2Ray)
        ax2.set_yscale("log")
        ax2.yaxis.set_major_formatter(FuncFormatter(velocity_log_tick_formatter))
    else:
        ax2.plot(t_Myr, v_IF_plot, lw=5.0, color=color_LYDION)
        if plot_C2Ray:
            ax2.plot(t_C2Ray_Myr, v_C2Ray_kms, lw=5.0, ls="--", dash_capstyle="round", color=color_C2Ray)

    ax2.set_xlabel("Time (Myr)", labelpad=14)
    ax2.set_ylabel("$v_{IF}$ (km s$^{-1}$)", labelpad=14)
    ax2.grid(False)

else:
    ax1.set_xlabel("Time (Myr)", labelpad=14)


# --- Axis limits ---
if xlim is not None:
    ax1.set_xlim(*xlim)

if ylim_R is not None:
    ax1.set_ylim(*ylim_R)
else:
    finite_R = np.isfinite(R_IF_plot)
    if np.any(finite_R):
        ax1.set_ylim(0.0, 1.08 * np.nanmax(R_IF_plot))

if plot_velocity:
    if ylim_v is not None:
        ax2.set_ylim(*ylim_v)
    else:
        if velocity_yscale_log:
            v_for_limits = [v_IF_plot]
            if plot_C2Ray:
                v_for_limits.append(v_C2Ray_kms)
            v_for_limits = np.concatenate([np.asarray(v).ravel() for v in v_for_limits])
            good_v = np.isfinite(v_for_limits) & (v_for_limits > 0.0)
            if np.any(good_v):
                vmin = np.nanmin(v_for_limits[good_v])
                vmax = np.nanmax(v_for_limits[good_v])
                if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                    ax2.set_ylim(vmin / 1.25, vmax * 1.25)
        else:
            v_for_limits = [v_IF_plot]
            if plot_C2Ray:
                v_for_limits.append(v_C2Ray_kms)
            v_for_limits = np.concatenate([np.asarray(v).ravel() for v in v_for_limits])
            if np.any(np.isfinite(v_for_limits)):
                ymin = np.nanmin(v_for_limits)
                ymax = np.nanmax(v_for_limits)
                if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                    pad = 0.08 * (ymax - ymin)
                    ax2.set_ylim(ymin - pad, ymax + pad)


# --- Minor ticks ---
if use_minor_ticks:

    # Top panel x-axis
    ax1.xaxis.set_minor_locator(AutoMinorLocator(x_minor_subdivisions))

    # Top panel y-axis
    ax1.yaxis.set_minor_locator(AutoMinorLocator(R_minor_subdivisions))

    if plot_velocity:

        # Bottom panel x-axis
        ax2.xaxis.set_minor_locator(AutoMinorLocator(x_minor_subdivisions))

        # Bottom panel y-axis
        if velocity_yscale_log:
            ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=log_minor_subs))
            ax2.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax2.yaxis.set_minor_locator(AutoMinorLocator(v_minor_subdivisions))


# --- Optional threshold label ---
if show_threshold_label:
    label_style = dict(
        ha="left", va="top", color="black",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")]
    )

    if IF_fraction == "xHII":
        threshold_text = r"$\langle x_{\rm HII}\rangle = %.1f$" % IF_threshold
    else:
        threshold_text = r"$\langle x_{\rm HI}\rangle = %.1f$" % IF_threshold

    ax1.text(
        threshold_label_position[0],
        threshold_label_position[1],
        threshold_text,
        transform=ax1.transAxes,
        fontsize=threshold_label_fontsize,
        **label_style
    )


# ===============================================================
#                         Save/show
# ===============================================================

if savefig:
    fig.savefig(outname, dpi=300, bbox_inches="tight")
else:
    plt.show()