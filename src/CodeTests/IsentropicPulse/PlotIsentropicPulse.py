import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

# ---------------- User inputs ----------------
script_dir      = os.path.dirname(os.path.abspath(__file__))
N           =  [32, 64, 128, 256] # grid resolution(s) to plot
gamma       = 5.0/3.0
Axis        = "r"   # profile direction; here "r" = X-direction in your 2D grid
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

# ---------- Helper: load one run and extract 1D profile ----------
def load_1D_averaged_profile(filename, Axis="r"):
    with h5py.File(filename, "r") as f:
        T       = f["Times"][:]
        RhoHist = f["GasDensity_History"][:]
        PHist   = f["P_History"][:]

        X = f["rCoM"][:]
        Y = f["ZCoM"][:]
        dV = f["CellVolumes"][:]
    
    #print(X.shape, Y.shape)
    XX, YY = np.meshgrid(X, Y, indexing='ij')
    R = np.sqrt(XX**2 + YY**2)
    #print(R.shape)
    # Ensure (Nt, Nr, Nz)
    RhoHist = ensure_NtNrNz(RhoHist, len(T))
    PHist   = ensure_NtNrNz(PHist,   len(T))

    Nt, N1, N2 = RhoHist.shape

    # Decide which axis is r and which is Z by matching RCoM / ZCoM lengths
    if N1 == len(X) and N2 == len(Y):
        Nr, Nz = N1, N2
    elif N1 == len(Y) and N2 == len(X):
        # swapped: axis 1 = Z, axis 2 = r → fix by transposing axes 1 and 2
        RhoHist = RhoHist.transpose(0, 2, 1)
        PHist   = PHist.transpose(0, 2, 1)
        uXHist  = uXHist.transpose(0, 2, 1)
        Nt, Nr, Nz = RhoHist.shape
    else:
        raise ValueError(
            f"Could not match history shape {RhoHist.shape} to "
            f"RCoM({len(X)}) / ZCoM({len(Y)})."
        )

    i_final = Nt - 1
    t_final = T[i_final]

    # Final snapshots
    Rho = RhoHist[i_final, :, :]
    P = PHist[i_final, :, :]

    # Angle averaged profiles based on radial bins in R
    R_centers = np.linspace(0.0, np.max(R), Nr)
    dR = R_centers[1] - R_centers[0]
    R_bins = 0.5 * (R_centers[:-1] + R_centers[1:])
    R_bins = np.concatenate(([0.0], R_bins, [np.max(R)+0.5*dR]))

    Rho_avg = np.zeros(Nr)
    P_avg   = np.zeros(Nr)
    for i in range(Nr):
        in_bin = (R >= R_bins[i]) & (R < R_bins[i+1])
        count_in_bin = np.sum(in_bin)
        if count_in_bin > 0:
            Rho_avg[i] = np.mean(Rho[in_bin])
            P_avg[i]   = np.mean(P[in_bin])
        else:
            Rho_avg[i] = 0.0
            P_avg[i]   = 0.0
    
    return R_centers, dV, Rho_avg, P_avg, t_final

def L1_norm(field, cell_volumes):
    return np.sum(np.abs(field) * cell_volumes) / np.sum(cell_volumes) 


###############################################################
#######           P L O T   C O S M E T I C S           #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 26})

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
(ax_rho, ax_L1)= axs

# --- Colors / labels for the two resolutions ---
colors = ["limegreen","dodgerblue", "crimson", "gray"]#, "deeppink"]
markers = ["o", "s", "D", "^"]#, "v"]
ms_num    = 8
lw_Ana    = 3.0
alpha     = 0.8

L1 = np.zeros(len(N)+1)

for i, (Nx, color, marker) in enumerate(zip(N, colors, markers)):

    try:
        File = os.path.join(script_dir, f"IsentropicPulseTest_{Nx:03d}x{Nx:03d}_t0.10.h5")
    except FileNotFoundError:
        print('File for N={Nx} not found. Run the simulation first.')

    R, dV, Rho, P , t_final = load_1D_averaged_profile(File, Axis)

    L1[i] = L1_norm(P/Rho**gamma - 1.0, dV)
    
    label = f"N = {Nx}"
    
    ax_rho.plot(R, Rho, color=color,  label=label, alpha = alpha, marker=marker, linestyle = 'None', markersize= ms_num)


# Plot the reference solution
N_ref = 1024
N_all = N + [N_ref]
print(N_all)
File = os.path.join(script_dir, f"IsentropicPulseTest_{N_ref:03d}x{N_ref:03d}_t0.10.h5")
R_ref, dV_ref, Rho_ref, P_ref , t_final_ref = load_1D_averaged_profile(File, Axis)
ax_rho.plot(R_ref, Rho_ref, color='black', label=f"N = {N_ref}", alpha = 0.8, linewidth=lw_Ana)
L1[-1] = L1_norm(P_ref/Rho_ref**gamma - 1.0, dV_ref)

ax_rho.set_ylabel(r"$\rho$", labelpad=20)
ax_rho.set_xlabel(r"Radius $r$", labelpad=20)
ax_rho.set_xlim(0.0, 0.5)
#ax_rho.set_ylim(-0.1, 1.1)
ax_rho.xaxis.set_minor_locator(AutoMinorLocator(4))
ax_rho.yaxis.set_minor_locator(AutoMinorLocator(5))

ax_L1.plot(N_all, L1, color='black', marker='o', linestyle='None', markersize= ms_num)
ax_L1.set_yscale('log')
ax_L1.set_xscale('log')
ax_L1.set_xlabel(r"Grid resolution N")
ax_L1.set_ylabel(r"$L_1$ norm of $P/\rho^{\gamma} - 1$", labelpad=20)

# Minor ticks for log axes
ax_L1.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax_L1.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))

# Optional: hide minor tick labels (recommended)
ax_L1.xaxis.set_minor_formatter(NullFormatter())
ax_L1.yaxis.set_minor_formatter(NullFormatter())

# Plot reference slope
slope = 2
L1_ref = L1[0] * (np.array(N_all)/N_all[0])**(-slope)
ax_L1.plot(N_all, L1_ref, color='gray', linestyle='--', label=rf"$\propto N^{{-{slope:1d}}}$")

fig.suptitle(f"Isentropic pulse test (t = {t_final:.3f})", fontsize=30)
# Put legends on each panel or one global; here: one global legend at top-right

ax_rho.legend(loc="upper right")
ax_L1.legend(loc="upper right")
plt.tight_layout()
plt.subplots_adjust(top=0.90)

# --- Save or show ---
if SaveFigure:
    #plt.savefig(OutName, dpi=600)
    plt.savefig(OutName)
    print(f"Saved plot to {OutName}")
else:
    plt.show()
