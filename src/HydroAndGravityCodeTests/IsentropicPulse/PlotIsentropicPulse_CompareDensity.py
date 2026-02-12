import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

# ---------------- User inputs ----------------
script_dir  = os.path.dirname(os.path.abspath(__file__))
N           = [32, 64, 128, 256, 1024]          # Grid resolution(s) to plot
ErrorNorm   = "L1"                             # "L1" or "L2"
SaveFigure  = True
OutName     = os.path.join(script_dir, "IsentropicPulseTest.pdf")
# ---------------------------------------------

# ---------- Helper: make sure (Nt, Nr, Nz) ----------
def ensure_NtNrNz(arr, Tlen):
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    if arr.shape[2] == Tlen:
        # stored as (Nr_or_Nz, Nz_or_Nr, Nt)
        return arr.transpose(2, 0, 1)  # -> (Nt, ?, ?)
    elif arr.shape[0] == Tlen:
        # already (Nt, ?, ?)
        return arr
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} vs len(T)={Tlen}")

def last_valid_final_index(T):
    if T.size == 0:
        return 0
    saved = np.where((T > 0) | (np.arange(T.size) == 0))[0]
    return int(saved[-1]) if saved.size > 0 else int(T.size - 1)

def load_final_rho(filename):
    with h5py.File(filename, "r") as f:
        T       = f["Times"][:]
        RhoHist = f["GasDensity_History"][:]
        X       = f["rCoM"][:]         # cell centers (x)
        Y       = f["ZCoM"][:]         # cell centers (y)
        dV      = f["CellVolumes"][:]  # (Nr, Nz)

    RhoHist = ensure_NtNrNz(RhoHist, len(T))
    Nt, N1, N2 = RhoHist.shape

    # Match axes to X,Y
    if (N1 == len(X)) and (N2 == len(Y)):
        pass
    elif (N1 == len(Y)) and (N2 == len(X)):
        RhoHist = RhoHist.transpose(0, 2, 1)
        Nt, N1, N2 = RhoHist.shape
    else:
        raise ValueError(
            f"Could not match history shape {RhoHist.shape} to rCoM({len(X)}) / ZCoM({len(Y)})."
        )

    i_final = last_valid_final_index(T[:Nt])
    rho2d   = RhoHist[i_final, :, :].astype(np.float64)

    # Ensure dV matches rho2d orientation
    if dV.shape == rho2d.shape:
        pass
    elif dV.T.shape == rho2d.shape:
        dV = dV.T
    else:
        raise ValueError(f"CellVolumes shape {dV.shape} incompatible with rho2d {rho2d.shape}")

    return float(T[i_final]), X.astype(np.float64), Y.astype(np.float64), dV.astype(np.float64), rho2d

def bilinear_interp_on_centers(Xr, Yr, Fr, Xc, Yc):
    """
    Fr is defined on (Xr, Yr) cell centers, shape (len(Xr), len(Yr)).
    Returns Fc on coarse centers (Xc, Yc), shape (len(Xc), len(Yc)),
    via bilinear interpolation.

    Assumes Xr and Yr are strictly increasing.
    """
    Xr = np.asarray(Xr)
    Yr = np.asarray(Yr)
    Xc = np.asarray(Xc)
    Yc = np.asarray(Yc)
    Fr = np.asarray(Fr)

    if not (np.all(np.diff(Xr) > 0) and np.all(np.diff(Yr) > 0)):
        raise ValueError("Reference grid centers must be strictly increasing for interpolation.")

    nxr = Xr.size
    nyr = Yr.size

    # Clamp query points to interior so i+1,j+1 exist
    Xq = np.clip(Xc, Xr[0], Xr[-1])
    Yq = np.clip(Yc, Yr[0], Yr[-1])

    # Indices of left cell center
    ix = np.searchsorted(Xr, Xq, side="right") - 1
    iy = np.searchsorted(Yr, Yq, side="right") - 1
    ix = np.clip(ix, 0, nxr - 2)
    iy = np.clip(iy, 0, nyr - 2)

    # Fractions
    x0 = Xr[ix]
    x1 = Xr[ix + 1]
    y0 = Yr[iy]
    y1 = Yr[iy + 1]

    # Avoid divide-by-zero (shouldn’t happen on a sane grid)
    tx = (Xq - x0) / (x1 - x0 + 1e-300)
    ty = (Yq - y0) / (y1 - y0 + 1e-300)

    # Build Fc by vectorized bilinear interpolation
    Fc = np.empty((Xc.size, Yc.size), dtype=np.float64)

    # We do it with loops over one dimension to keep it readable + robust
    for i in range(Xc.size):
        i0 = ix[i]
        i1 = i0 + 1
        wx0 = 1.0 - tx[i]
        wx1 = tx[i]

        # for each y, pick the right y-bracket
        j0 = iy
        j1 = j0 + 1

        f00 = Fr[i0, j0]
        f10 = Fr[i1, j0]
        f01 = Fr[i0, j1]
        f11 = Fr[i1, j1]

        wy0 = 1.0 - ty
        wy1 = ty

        Fc[i, :] = (
            wx0 * (wy0 * f00 + wy1 * f01) +
            wx1 * (wy0 * f10 + wy1 * f11)
        )

    return Fc

def volume_weighted_error(diff, dV, norm="L2"):
    mask = np.isfinite(diff) & np.isfinite(dV)
    if not np.any(mask):
        raise ValueError("No valid cells for error calculation.")
    Vtot = np.sum(dV[mask])
    if Vtot <= 0:
        raise ValueError("Total volume <= 0 in error calculation.")

    if norm == "L1":
        return np.sum(np.abs(diff[mask]) * dV[mask]) / Vtot
    elif norm == "L2":
        return np.sqrt(np.sum((diff[mask]**2) * dV[mask]) / Vtot)
    else:
        raise ValueError("norm must be 'L1' or 'L2'.")

def radial_profile_volume_weighted(X, Y, dV, rho2d):
    XX, YY = np.meshgrid(X, Y, indexing='ij')
    R = np.sqrt(XX**2 + YY**2)

    Nr = len(X)
    R_centers = np.linspace(0.0, np.max(R), Nr)
    dR = R_centers[1] - R_centers[0]
    R_bins = 0.5 * (R_centers[:-1] + R_centers[1:])
    R_bins = np.concatenate(([0.0], R_bins, [np.max(R) + 0.5*dR]))

    rho_avg = np.zeros(Nr, dtype=np.float64)
    for i in range(Nr):
        in_bin = (R >= R_bins[i]) & (R < R_bins[i+1])
        if np.any(in_bin):
            w = dV[in_bin]
            wsum = np.sum(w)
            rho_avg[i] = np.sum(rho2d[in_bin] * w) / wsum if wsum > 0 else 0.0
        else:
            rho_avg[i] = 0.0
    return R_centers, rho_avg


###############################################################
#######           P L O T   C O S M E T I C S           #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 26})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'
plt.rcParams.update({'legend.fontsize': 22})
plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 1.7
plt.rcParams['xtick.minor.width'] = 1.7
plt.rcParams['xtick.major.size'] = 15
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size'] = 15
plt.rcParams['ytick.minor.size'] = 8
plt.rcParams['xtick.major.pad'] = '15'
plt.rcParams['ytick.major.pad'] = '15'
plt.rcParams['legend.frameon'] = 'False'
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linewidth'] = 3.2
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


###############################################################
#######               L O A D   R E F                    #######
###############################################################

N_ref = max(N)
File_ref = os.path.join(script_dir, f"IsentropicPulseTest_{N_ref:03d}x{N_ref:03d}_t0.10.h5")
if not os.path.exists(File_ref):
    raise FileNotFoundError(f"Missing reference file for N_ref={N_ref}: {File_ref}")

t_ref, X_ref, Y_ref, dV_ref, rho_ref2d = load_final_rho(File_ref)
R_ref, rho_ref_1d = radial_profile_volume_weighted(X_ref, Y_ref, dV_ref, rho_ref2d)


###############################################################
#######               P L O T  I T !                    #######
###############################################################

fig, axs = plt.subplots(2, 1, figsize=(15, 11), sharex=False, sharey=False)
(ax_rho, ax_err) = axs

colors  = ["limegreen", "dodgerblue", "crimson", "gray"]
markers = ["o", "s", "D", "^"]
ms_num  = 8
lw_ref  = 3.0
alpha   = 0.8

Err = np.full(len(N), np.nan, dtype=np.float64)

for i, Nx in enumerate(N):
    color  = colors[i % len(colors)]
    marker = markers[i % len(markers)]

    File = os.path.join(script_dir, f"IsentropicPulseTest_{Nx:03d}x{Nx:03d}_t0.10.h5")
    if not os.path.exists(File):
        raise FileNotFoundError(f"Missing file for N={Nx}: {File}")

    t_final, X, Y, dV, rho2d = load_final_rho(File)
    R, rho_1d = radial_profile_volume_weighted(X, Y, dV, rho2d)

    # --- interpolate reference rho to coarse cell centers ---
    rho_ref_on_coarse = bilinear_interp_on_centers(X_ref, Y_ref, rho_ref2d, X, Y)

    # --- error on matching physical positions ---
    diff = rho2d - rho_ref_on_coarse
    Err[i] = volume_weighted_error(diff, dV, norm=ErrorNorm)

    ax_rho.plot(R, rho_1d, color=color, label=f"N = {Nx}", alpha=alpha,
                marker=marker, linestyle='None', markersize=ms_num)

# Reference curve
ax_rho.plot(R_ref, rho_ref_1d, color='black', label=f"ref: N = {N_ref}",
            alpha=0.8, linewidth=lw_ref)

# --- rho panel ---
ax_rho.set_ylabel(r"$\rho$", labelpad=20)
ax_rho.set_xlabel(r"Radius $r$", labelpad=20)
ax_rho.set_xlim(0.0, 0.5)
ax_rho.xaxis.set_minor_locator(AutoMinorLocator(4))
ax_rho.yaxis.set_minor_locator(AutoMinorLocator(5))
ax_rho.legend(loc="upper right", ncols=2)

# --- error panel ---
N_all = np.array(N, dtype=np.float64)
mask_ok = np.isfinite(Err) & (Err > 0.0)
if not np.all(mask_ok):
    print("Warning: some Err values are non-finite or <=0 and won't show on log axis.")
    print("N:", N_all)
    print("Err:", Err)

ax_err.plot(N_all[mask_ok], Err[mask_ok], color='black', marker='o',
            linestyle='None', markersize=ms_num)
ax_err.set_yscale('log')
ax_err.set_xscale('log')
ax_err.set_xlabel(r"Grid resolution N")
ax_err.set_ylabel(rf"$L_{{{1 if ErrorNorm=='L1' else 2}}}$ norm of $\rho - \rho_{{\rm ref}}$", labelpad=20)

ax_err.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax_err.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax_err.xaxis.set_minor_formatter(NullFormatter())
ax_err.yaxis.set_minor_formatter(NullFormatter())

# Slope guides anchored at first valid point
if np.any(mask_ok):
    i0 = np.where(mask_ok)[0][0]
    E0 = Err[i0]
    N0 = N_all[i0]
    g2 = E0 * (N_all / N0)**(-2)
    g3 = E0 * (N_all / N0)**(-3)
    ax_err.plot(N_all, g2, color='crimson', linestyle='--', label=r"$\propto N^{-2}$", linewidth=2.5)
    ax_err.plot(N_all, g3, color='black',   linestyle='--', label=r"$\propto N^{-3}$", linewidth=2.5)
    ax_err.legend(loc="upper right", ncols=2)

fig.suptitle(f"Isentropic pulse test (t = {t_ref:.3f})", fontsize=30)
plt.tight_layout()
plt.subplots_adjust(top=0.90)

if SaveFigure:
    plt.savefig(OutName)
    print(f"Saved plot to {OutName}")
else:
    plt.show()

