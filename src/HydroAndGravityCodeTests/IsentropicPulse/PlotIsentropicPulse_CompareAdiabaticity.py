import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

# ---------------- User inputs ----------------
script_dir  = os.path.dirname(os.path.abspath(__file__))
N           = [32, 64, 128, 256, 512, 1024]    # Grid resolution(s) to plot
gamma       = 5.0/3.0
Axis        = "r"                              # kept for compatibility (not used below)
ErrorNorm   = "L1"                             # "L1" or "L2"
SaveFigure  = True
OutName     = os.path.join(script_dir, "IsentropicPulseTest.pdf")
# ---------------------------------------------

# ---------- Helper: make sure (Nt, Nr, Nz) ----------
def ensure_NtNrNz(arr, Tlen):
    if arr.shape[2] == Tlen:
        # stored as (N_r_or_Z, N_Z_or_r, N_t)
        arr = arr.transpose(2, 0, 1)  # now (N_t, ?, ?)
    elif arr.shape[0] == Tlen:
        # already (N_t, ?, ?)
        pass
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} vs len(T)={Tlen}")
    return arr

# ---------- Helper: load one run and extract 1D profile + 2D L1/L2 error ----------
def load_1D_averaged_profile(filename, gamma, ErrorNorm="L1", Axis="r"):
    with h5py.File(filename, "r") as f:
        T       = f["Times"][:]
        RhoHist = f["GasDensity_History"][:]
        PHist   = f["P_History"][:]

        X  = f["rCoM"][:]          # cell centers in r (your X-direction)
        Y  = f["ZCoM"][:]          # cell centers in Z
        dV = f["CellVolumes"][:]   # (Nr, Nz)

    XX, YY = np.meshgrid(X, Y, indexing='ij')
    R = np.sqrt(XX**2 + YY**2)

    # Ensure (Nt, Nr, Nz)
    RhoHist = ensure_NtNrNz(RhoHist, len(T))
    PHist   = ensure_NtNrNz(PHist,   len(T))

    Nt, N1, N2 = RhoHist.shape

    # Decide which axis is r and which is Z by matching rCoM / ZCoM lengths
    if N1 == len(X) and N2 == len(Y):
        Nr, Nz = N1, N2
    elif N1 == len(Y) and N2 == len(X):
        # swapped: axis 1 = Z, axis 2 = r → fix by transposing axes 1 and 2
        RhoHist = RhoHist.transpose(0, 2, 1)
        PHist   = PHist.transpose(0, 2, 1)
        Nt, Nr, Nz = RhoHist.shape
    else:
        raise ValueError(
            f"Could not match history shape {RhoHist.shape} to "
            f"rCoM({len(X)}) / ZCoM({len(Y)})."
        )

    # Final index: last saved time slice in the file
    i_final = Nt - 1
    t_final = T[i_final]

    # Final snapshots (2D)
    Rho2D = RhoHist[i_final, :, :].astype(np.float64)
    P2D   = PHist[i_final, :, :].astype(np.float64)

    # ---------- FV L1/L2 norm on 2D field (volume-weighted) ----------
    mask = (Rho2D > 0.0) & np.isfinite(Rho2D) & np.isfinite(P2D) & np.isfinite(dV)
    if not np.any(mask):
        raise ValueError("No valid cells for error calculation (check rho/P).")

    e2D = np.zeros_like(Rho2D)
    e2D[mask] = P2D[mask] / (Rho2D[mask]**gamma) - 1.0

    Vtot = np.sum(dV[mask])
    if Vtot <= 0.0:
        raise ValueError("Total valid volume is <= 0; cannot compute norms.")

    if ErrorNorm == "L1":
        Err = np.sum(np.abs(e2D[mask]) * dV[mask]) / Vtot
    elif ErrorNorm == "L2":
        Err = np.sqrt(np.sum((e2D[mask]**2) * dV[mask]) / Vtot)
    else:
        raise ValueError(f"Unknown ErrorNorm={ErrorNorm}. Use 'L1' or 'L2'.")

    # ---------- 1D radial (angle-averaged) profiles for plotting ----------
    R_centers = np.linspace(0.0, np.max(R), Nr)
    dR = R_centers[1] - R_centers[0]
    R_bins = 0.5 * (R_centers[:-1] + R_centers[1:])
    R_bins = np.concatenate(([0.0], R_bins, [np.max(R) + 0.5*dR]))

    Rho_avg = np.zeros(Nr)
    P_avg   = np.zeros(Nr)

    for i in range(Nr):
        in_bin = (R >= R_bins[i]) & (R < R_bins[i+1])
        if np.any(in_bin):
            w = dV[in_bin]
            wsum = np.sum(w)
            if wsum > 0.0:
                Rho_avg[i] = np.sum(Rho2D[in_bin] * w) / wsum
                P_avg[i]   = np.sum(P2D[in_bin]   * w) / wsum
            else:
                Rho_avg[i] = 0.0
                P_avg[i]   = 0.0
        else:
            Rho_avg[i] = 0.0
            P_avg[i]   = 0.0

    return R_centers, Rho_avg, P_avg, t_final, Err


###############################################################
#######           P L O T   C O S M E T I C S           #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 26})

params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

params = {'legend.fontsize': 22}
plt.rcParams.update(params)

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

# ticks
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

###############################################################
#######               P L O T  I T !                    #######
###############################################################

fig, axs = plt.subplots(2, 1, figsize=(15, 11), sharex=False, sharey=False)
(ax_rho, ax_err) = axs

# --- Colors / labels for the two resolutions ---
colors  = ["limegreen", "dodgerblue", "crimson", "gray"]
markers = ["o", "s", "D", "^"]
ms_num  = 8
lw_Ana  = 3.0
alpha   = 0.6

Err = np.zeros(len(N), dtype=np.float64)

for i, Nx in enumerate(N):
    color  = colors[i % len(colors)]
    marker = markers[i % len(markers)]


    File = os.path.join(script_dir, f"IsentropicPulseTest_{Nx:03d}x{Nx:03d}_t0.10.h5")
    if not os.path.exists(File):
        raise FileNotFoundError(f"Missing file for N={Nx}: {File}")

    R, Rho, P, t_final, Err_i = load_1D_averaged_profile(File, gamma, ErrorNorm, Axis)
    Err[i] = Err_i

    label = f"N = {Nx}"
    ax_rho.plot(R, Rho, color=color, label=label, alpha=alpha,
                marker=marker, linestyle='None', markersize=ms_num)

# Reference = highest available resolution in N (no missing 1024 file)
N_ref = max(N)
File_ref = os.path.join(script_dir, f"IsentropicPulseTest_{N_ref:03d}x{N_ref:03d}_t0.10.h5")
R_ref, Rho_ref, P_ref, t_final_ref, Err_ref_val = load_1D_averaged_profile(File_ref, gamma, ErrorNorm, Axis)
ax_rho.plot(R_ref, Rho_ref, color='black', label=f"ref: N = {N_ref}",
            alpha=0.8, linewidth=lw_Ana)

# --- rho panel ---
ax_rho.set_ylabel(r"$\rho$", labelpad=20)
ax_rho.set_xlabel(r"Radius $r$", labelpad=20)
ax_rho.set_xlim(0.0, 1.0)
ax_rho.xaxis.set_minor_locator(AutoMinorLocator(4))
ax_rho.yaxis.set_minor_locator(AutoMinorLocator(5))
ax_rho.legend(loc="upper right")

# --- error panel ---
N_all = np.array(N, dtype=np.float64)
ax_err.plot(N_all, Err, color='black', marker='o', linestyle='None', markersize=ms_num)
ax_err.set_yscale('log')
ax_err.set_xscale('log')
ax_err.set_xlabel(r"Grid resolution N")


if ErrorNorm == "L1":
    ax_err.set_ylabel(r"$L_1$ norm of $P/\rho^{\gamma} - 1$", labelpad=20)
else:
    ax_err.set_ylabel(r"$L_2$ norm of $P/\rho^{\gamma} - 1$", labelpad=20)

# Minor ticks for log axes
ax_err.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax_err.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax_err.xaxis.set_minor_formatter(NullFormatter())
ax_err.yaxis.set_minor_formatter(NullFormatter())

# Reference slope guide
slope = 3
Err_guideline_2 = Err[0] * (N_all / N_all[0])**(-2)
Err_guideline_3 = Err[0] * (N_all / N_all[0])**(-3)
ax_err.plot(N_all, Err_guideline_2, color='crimson', linestyle='--',
            label = "$\propto N^{{-2}}$", dash_capstyle = 'round',
            linewidth = 2.5)
ax_err.plot(N_all, Err_guideline_3, color='black', linestyle='--',
            label = "$\propto N^{{-3}}$", dash_capstyle = 'round',
            linewidth = 2.5)
ax_err.legend(loc="upper right")

fig.suptitle(f"Isentropic pulse test (t = {t_final:.3f})", fontsize=30)
plt.tight_layout()
plt.subplots_adjust(top=0.90)

if SaveFigure:
    plt.savefig(OutName)
    print(f"Saved plot to {OutName}")
else:
    plt.show()

