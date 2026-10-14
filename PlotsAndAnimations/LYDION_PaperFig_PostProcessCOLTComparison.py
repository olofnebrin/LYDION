import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
from pathlib import Path
import os

repo_root = Path(__file__).resolve().parents[1]
os.chdir(repo_root)


# ============================================================
# User inputs
# ============================================================

lydion_filename = "RHDSim_Lya_NoDest_u4_256x256_2026-05-06.h5"
colt_filename   = "Lya_no_dest_xcrit3.hdf5"   # no-destruction COLT file

outname = "Compare_eLya_LYDION_COLT_withGas.png"
savefig = True

# COLT duplicated the LYDION mirror domain in Z.
# Usually:
#   "upper" = second half of COLT Z-domain, likely +Z half
#   "lower" = first half of COLT Z-domain, likely -Z half
colt_half = "upper"   # "upper" or "lower"

# COLT full-array layout:
#   "rz" = Aaron's stated suggestion: u_flat.reshape((Nr, Nz_colt))
#   "zr" = alternative:              u_flat.reshape((Nz_colt, Nr)).T
#
# If changing this did not visibly affect the plotted half-plane,
# keep "rz" and use colt_post_crop_orientation below instead.
colt_layout = "rz"    # "rz" or "zr"

# COLT orientation after selecting the half-domain.
# Use "normal" if the morphology is correct.
# Use "transpose_after_crop" if COLT appears transposed relative to LYDION/gas.
colt_post_crop_orientation = "transpose_after_crop"  # "normal" or "transpose_after_crop"

# Optional multiplicative conversion for COLT.
# Leave as 1.0 if u_rad_r is already an energy density in erg cm^-3.
# If it turns out COLT output is per steradian, try colt_scale = 4.0*np.pi.
colt_scale = 1.0

# Lyα color scale.
# If use_shared_Lya_limits=True, vmin/vmax are set from both LYDION and COLT.
# Otherwise, manual limits below are used.
use_shared_Lya_limits = True
manual_Lya_vmin = 1e-12
manual_Lya_vmax = 1e-6

# Gas panel color scales.
use_data_limits_nH = False
manual_nH_vmin = 1e3
manual_nH_vmax = 1e7

use_data_limits_T = False
manual_T_vmin = 10.0
manual_T_vmax = 1e5

# Floors for log plotting.
positive_floor = 1e-99

# Plot labels / title.
fig_title = None
left_title  = r"Lydion"
right_title = "COLT"

# Constants
pc  = 3.085677581e18
yr  = 3.1556926e7
kyr = 1e3 * yr

mH  = 1.67e-24
XH  = 0.76


# ============================================================
# Helpers
# ============================================================

def reorder_history_3d(arr, t, r_edges, Z_edges):
    """
    Reorder a 3D history array to shape (Nt, Nr, Nz).

    One axis must match len(t).
    The remaining two must match (Nr, Nz) in either order.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    Nt = t.size
    Nr = len(r_edges) - 1
    Nz = len(Z_edges) - 1

    shape = arr.shape

    time_axes = [ax for ax, s in enumerate(shape) if s == Nt]
    if len(time_axes) != 1:
        raise RuntimeError(
            f"Could not uniquely identify time axis in array with shape {shape} and Nt={Nt}"
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
            f"After moving time axis, spatial dims {(a1, a2)} "
            f"do not match Nr={Nr}, Nz={Nz}"
        )


def last_saved_index(t):
    """
    Match the convention used in the animation script:
    keep indices with t > 0, plus index 0.
    Return the final saved index.
    """
    saved_idx = np.where((t > 0) | (np.arange(t.size) == 0))[0]
    if saved_idx.size == 0:
        raise RuntimeError("No saved time indices found.")
    return saved_idx[-1], saved_idx


def safe_log_limits(arrays, pmin=0.5, pmax=99.5):
    """
    Determine positive log limits from one or more arrays.
    """
    vals = []

    for arr in arrays:
        good = np.isfinite(arr) & (arr > 0.0)
        if np.any(good):
            vals.append(arr[good])

    if len(vals) == 0:
        raise RuntimeError("No positive finite values found for LogNorm.")

    vals = np.concatenate(vals)

    vmin = np.nanpercentile(vals, pmin)
    vmax = np.nanpercentile(vals, pmax)

    if not np.isfinite(vmin) or vmin <= 0.0:
        vmin = np.nanmin(vals[vals > 0.0])

    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = np.nanmax(vals)

    return vmin, vmax


def read_lydion_final_fields(filename):
    """
    Read LYDION grid and final e_Lya, nH, and T snapshots.

    Returns
    -------
    r, Z : cell-edge arrays
    e_final : shape (Nr, Nz)
    nH_final : shape (Nr, Nz)
    T_final : shape (Nr, Nz)
    t_final : final saved time in seconds
    """
    with h5py.File(filename, "r") as f:
        t = f["Times"][:]
        r = f["r"][:]
        Z = f["Z"][:]

        e_raw   = f["e_Lya_History"][:]
        rho_raw = f["GasDensity_History"][:]
        T_raw   = f["T_History"][:]

    e_hist   = reorder_history_3d(e_raw,   t, r, Z)
    rho_hist = reorder_history_3d(rho_raw, t, r, Z)
    T_hist   = reorder_history_3d(T_raw,   t, r, Z)

    k_final, saved_idx = last_saved_index(t)

    e_final   = e_hist[k_final]
    rho_final = rho_hist[k_final]
    T_final   = T_hist[k_final]

    nH_final = XH * rho_final / mH

    print("LYDION:")
    print(f"  file          = {filename}")
    print(f"  e_raw shape   = {e_raw.shape}")
    print(f"  rho_raw shape = {rho_raw.shape}")
    print(f"  T_raw shape   = {T_raw.shape}")
    print(f"  e_hist shape  = {e_hist.shape}  (Nt, Nr, Nz)")
    print(f"  rho_hist shape= {rho_hist.shape}  (Nt, Nr, Nz)")
    print(f"  T_hist shape  = {T_hist.shape}  (Nt, Nr, Nz)")
    print(f"  final index   = {k_final}")
    print(f"  final time    = {t[k_final]/kyr:.6g} kyr")
    print(f"  r range       = {r[0]/pc:.6g} -> {r[-1]/pc:.6g} pc")
    print(f"  Z range       = {Z[0]/pc:.6g} -> {Z[-1]/pc:.6g} pc")
    print(f"  e_Lya min/max = {np.nanmin(e_final):.6e}, {np.nanmax(e_final):.6e}")
    print(f"  nH min/max    = {np.nanmin(nH_final):.6e}, {np.nanmax(nH_final):.6e}")
    print(f"  T min/max     = {np.nanmin(T_final):.6e}, {np.nanmax(T_final):.6e}")

    return r, Z, e_final, nH_final, T_final, t[k_final]


def read_colt_u_rad(
    filename,
    Nr_lydion,
    Nz_lydion,
    half="upper",
    scale=1.0,
    layout="rz",
    post_crop_orientation="normal",
):
    """
    Read COLT u_rad_r and reshape.

    Aaron's note:
        f['u_rad_r'][:].reshape((n_r, n_z))
        with n_z = 2*n_r, C-ordering, no transpose.

    However, if the cropped half-plane appears transposed relative to
    LYDION, use post_crop_orientation="transpose_after_crop".
    This is useful because after cropping, the COLT half-plane is often
    square, e.g. (256, 256), so a wrong transpose is not caught by shape.
    """
    with h5py.File(filename, "r") as f:
        u_flat = f["u_rad_r"][:]

    if u_flat.size % Nr_lydion != 0:
        raise RuntimeError(
            f"COLT u_rad_r length {u_flat.size} is not divisible by LYDION Nr={Nr_lydion}."
        )

    Nz_colt = u_flat.size // Nr_lydion

    # First interpret the full COLT duplicated domain.
    if layout == "rz":
        # Aaron's stated convention: full array is (r, Z_full)
        u_colt_full = u_flat.reshape((Nr_lydion, Nz_colt))

    elif layout == "zr":
        # Alternative convention: full array is stored as (Z_full, r)
        # and converted to (r, Z_full).
        u_colt_full = u_flat.reshape((Nz_colt, Nr_lydion)).T

    else:
        raise ValueError("layout must be 'rz' or 'zr'.")

    u_colt_full = u_colt_full * scale

    # Then crop to the LYDION half-plane.
    if Nz_colt == Nz_lydion:
        u_colt_half = u_colt_full

    elif Nz_colt == 2 * Nz_lydion:
        if half == "lower":
            u_colt_half = u_colt_full[:, :Nz_lydion]
        elif half == "upper":
            u_colt_half = u_colt_full[:, Nz_lydion:]
        else:
            raise ValueError("half must be 'upper' or 'lower'.")

    else:
        raise RuntimeError(
            f"Unexpected COLT reshaped shape {(Nr_lydion, Nz_colt)}. "
            f"Expected Nz_colt = Nz_lydion ({Nz_lydion}) or 2*Nz_lydion ({2*Nz_lydion})."
        )

    # Finally, allow an explicit orientation correction of the plotted half-plane.
    if post_crop_orientation == "normal":
        pass

    elif post_crop_orientation == "transpose_after_crop":
        if u_colt_half.shape[0] != u_colt_half.shape[1]:
            raise RuntimeError(
                "transpose_after_crop was requested, but the cropped COLT half-plane "
                f"is not square: shape={u_colt_half.shape}."
            )
        u_colt_half = u_colt_half.T

    else:
        raise ValueError(
            "post_crop_orientation must be 'normal' or 'transpose_after_crop'."
        )

    print("COLT:")
    print(f"  file                  = {filename}")
    print(f"  flat length           = {u_flat.size}")
    print(f"  full shape            = {u_colt_full.shape}")
    print(f"  plotted shape         = {u_colt_half.shape}")
    print(f"  half                  = {half}")
    print(f"  layout                = {layout}")
    print(f"  post_crop_orientation = {post_crop_orientation}")
    print(f"  scale                 = {scale:.6e}")
    print(f"  min/max               = {np.nanmin(u_colt_half):.6e}, {np.nanmax(u_colt_half):.6e}")

    return u_colt_half, u_colt_full


def set_plot_style():
    """
    Plot style close to your usual scripts, but robust if custom fonts
    are unavailable.
    """
    plt.rc("font", **{"sans-serif": ["AVHershey Complex", "DejaVu Sans"], "size": 30})

    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["axes.unicode_minus"] = False

    plt.rcParams["xtick.major.width"] = 1.7
    plt.rcParams["xtick.minor.width"] = 1.7
    plt.rcParams["xtick.major.size"] = 16
    plt.rcParams["xtick.minor.size"] = 7

    plt.rcParams["ytick.major.width"] = 1.7
    plt.rcParams["ytick.minor.width"] = 1.7
    plt.rcParams["ytick.major.size"] = 16
    plt.rcParams["ytick.minor.size"] = 7

    plt.rcParams["xtick.major.pad"] = "12"
    plt.rcParams["ytick.major.pad"] = "12"

    plt.rcParams["axes.linewidth"] = 2.5

    plt.rcParams["xtick.top"] = True
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def get_cmap(name, fallback):
    """
    Try cmastro/cmasher-style colormap names if installed;
    otherwise use matplotlib fallback.
    """
    try:
        return plt.get_cmap(name)
    except Exception:
        return plt.get_cmap(fallback)


# ============================================================
# Main
# ============================================================

script_dir = Path(__file__).resolve().parents[1]

lydion_path = script_dir / lydion_filename
colt_path   = script_dir / colt_filename
outpath     = script_dir / outname

r, Z, e_lydion, nH, T, t_final = read_lydion_final_fields(lydion_path)

Nr = len(r) - 1
Nz = len(Z) - 1

e_colt, e_colt_full = read_colt_u_rad(
    colt_path,
    Nr_lydion=Nr,
    Nz_lydion=Nz,
    half=colt_half,
    scale=colt_scale,
    layout=colt_layout,
    post_crop_orientation=colt_post_crop_orientation,
)

if e_colt.shape != e_lydion.shape:
    raise RuntimeError(
        f"Shape mismatch after COLT crop: COLT {e_colt.shape}, LYDION {e_lydion.shape}"
    )

# ----------------------------
# Normalizations
# ----------------------------

if use_shared_Lya_limits:
    Lya_vmin, Lya_vmax = safe_log_limits([e_lydion, e_colt])
else:
    Lya_vmin, Lya_vmax = manual_Lya_vmin, manual_Lya_vmax

if use_data_limits_nH:
    nH_vmin, nH_vmax = safe_log_limits([nH])
else:
    nH_vmin, nH_vmax = manual_nH_vmin, manual_nH_vmax

if use_data_limits_T:
    T_vmin, T_vmax = safe_log_limits([T])
else:
    T_vmin, T_vmax = manual_T_vmin, manual_T_vmax

print("Color scales:")
print(f"  Lyα vmin/vmax = {Lya_vmin:.6e}, {Lya_vmax:.6e}")
print(f"  nH  vmin/vmax = {nH_vmin:.6e}, {nH_vmax:.6e}")
print(f"  T   vmin/vmax = {T_vmin:.6e}, {T_vmax:.6e}")

norm_Lya = LogNorm(vmin=Lya_vmin, vmax=Lya_vmax)
norm_nH  = LogNorm(vmin=nH_vmin,  vmax=nH_vmax)
norm_T   = LogNorm(vmin=T_vmin,   vmax=T_vmax)

# Clip only for display, preserving real arrays for printed diagnostics.
e_lydion_plot = np.clip(e_lydion, Lya_vmin, Lya_vmax)
e_colt_plot   = np.clip(e_colt,   Lya_vmin, Lya_vmax)
nH_plot       = np.clip(nH,       nH_vmin,  nH_vmax)
T_plot        = np.clip(T,        T_vmin,   T_vmax)

# ----------------------------
# Diagnostics
# ----------------------------

good = np.isfinite(e_lydion) & np.isfinite(e_colt) & (e_lydion > 0) & (e_colt > 0)
if np.any(good):
    ratio = e_colt[good] / e_lydion[good]
    print("COLT / LYDION ratio over positive finite cells:")
    print(f"  median = {np.nanmedian(ratio):.6e}")
    print(f"  p16    = {np.nanpercentile(ratio, 16):.6e}")
    print(f"  p84    = {np.nanpercentile(ratio, 84):.6e}")
    print(f"  min    = {np.nanmin(ratio):.6e}")
    print(f"  max    = {np.nanmax(ratio):.6e}")


# ============================================================
# Plot
# ============================================================

set_plot_style()

cmap_Lya = get_cmap("cma:emph", "inferno")
cmap_nH  = get_cmap("cma:laguna", "viridis")
cmap_T   = get_cmap("coolwarm", "plasma")

fig, axes = plt.subplots(
    2, 2,
    figsize=(17.0, 14.0),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

ax_Lydion = axes[0, 0]
ax_COLT   = axes[0, 1]
ax_nH     = axes[1, 0]
ax_T      = axes[1, 1]

panel_label_style = dict(
    ha="right",
    va="top",
    color="black",
    path_effects=[pe.withStroke(linewidth=3.5, foreground="white")],
)

# IMPORTANT:
# Use 1D edge arrays directly.
# Since data are field[i,j] = field[r_index, Z_index],
# pcolormesh(x_edges, y_edges, C) wants C shape (Ny, Nx) = (Nz, Nr),
# hence field.T here.
im_Lydion = ax_Lydion.pcolormesh(
    r / pc,
    Z / pc,
    e_lydion_plot.T,
    shading="auto",
    norm=norm_Lya,
    cmap=cmap_Lya,
    rasterized=True,
)

im_COLT = ax_COLT.pcolormesh(
    r / pc,
    Z / pc,
    e_colt_plot.T,
    shading="auto",
    norm=norm_Lya,
    cmap=cmap_Lya,
    rasterized=True,
)

im_nH = ax_nH.pcolormesh(
    r / pc,
    Z / pc,
    nH_plot.T,
    shading="auto",
    norm=norm_nH,
    cmap=cmap_nH,
    rasterized=True,
)

im_T = ax_T.pcolormesh(
    r / pc,
    Z / pc,
    T_plot.T,
    shading="auto",
    norm=norm_T,
    cmap=cmap_T,
    rasterized=True,
)

# Titles
ax_Lydion.set_title(left_title, pad=14)
ax_COLT.set_title(right_title, pad=14)

# Axis labels
ax_nH.set_xlabel("r (pc)", labelpad=10)
ax_T.set_xlabel("r (pc)", labelpad=10)
ax_Lydion.set_ylabel("Z (pc)", labelpad=10)
ax_nH.set_ylabel("Z (pc)", labelpad=10)

for ax in axes.ravel():
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(r[0] / pc, r[-1] / pc)
    ax.set_ylim(Z[0] / pc, Z[-1] / pc)

# Panel labels
ax_Lydion.text(
    0.94, 0.93,
    r"$e_{Ly\alpha}$",
    transform=ax_Lydion.transAxes,
    **panel_label_style,
)

ax_COLT.text(
    0.94, 0.93,
    r"$u_{rad}$",
    transform=ax_COLT.transAxes,
    **panel_label_style,
)

ax_nH.text(
    0.94, 0.93,
    r"$n_H$",
    transform=ax_nH.transAxes,
    **panel_label_style,
)

ax_T.text(
    0.94, 0.93,
    r"$T$",
    transform=ax_T.transAxes,
    **panel_label_style,
)

# Time label on LYDION panel.
time_style = dict(
    ha="left",
    va="top",
    color="white",
    path_effects=[pe.withStroke(linewidth=3.5, foreground="black")],
)

ax_Lydion.text(
    0.06, 0.93,
    f"t = {t_final / kyr:.2f} kyr",
    transform=ax_Lydion.transAxes,
    fontsize=23,
    **time_style,
)

# Colorbars
cb_Lya = fig.colorbar(im_COLT, ax=[ax_Lydion, ax_COLT], pad=0.018, fraction=0.046)
cb_Lya.set_label(r"Ly$\alpha$ energy density (erg cm$^{-3}$)", labelpad=12)

cb_nH = fig.colorbar(im_nH, ax=ax_nH, pad=0.018, fraction=0.046)
cb_nH.set_label(r"$n_H$ (cm$^{-3}$)", labelpad=12)

cb_T = fig.colorbar(im_T, ax=ax_T, pad=0.018, fraction=0.046)
cb_T.set_label(r"$T$ (K)", labelpad=12)

if fig_title is not None:
    fig.suptitle(fig_title, y=1.03)

if savefig:
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved: {outpath}")
else:
    plt.show()