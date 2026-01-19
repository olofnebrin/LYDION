import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ---------------- User inputs ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
filename   = os.path.join(script_dir, "SedovTest_E1.00e+00_rho1_gamma1.67_t0.4_256x256.h5")
analytic_file = os.path.join(script_dir, "SedovTaylorAnalytical_E1.0_rho1.0_t0.4_gamma53.txt")

Axis        = "r"   # choose "r" or "Z" for profile direction
SaveFigure  = True
OutName     = os.path.join(script_dir, "SedovFinalProfile_Analytic.png")
# ---------------------------------------------

# --- Read numerical data ---
with h5py.File(filename, "r") as f:
    T       = f["Times"][:]
    RhoHist = f["GasDensity_History"][:]
    PHist   = f["P_History"][:]
    RCoM    = f["rCoM"][:]
    ZCoM    = f["ZCoM"][:]

# --- Ensure correct axis order: (N_saved, N_r, N_Z) ---
def ensure_NtNrNz(arr):
    if arr.shape[2] == len(T):
        return arr.transpose(2, 0, 1)
    elif arr.shape[0] == len(T):
        return arr
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} vs len(T)={len(T)}")

RhoHist = ensure_NtNrNz(RhoHist)
PHist   = ensure_NtNrNz(PHist)

Nt, Nr, Nz = RhoHist.shape
i_final = Nt - 1
t_final = T[i_final]

# --- Extract profile along r-axis at Z≈0 ---
if Axis.lower() == "r":
    j_mid = np.argmin(np.abs(ZCoM - 0.0))  # choose plane closest to Z=0
    Coord       = RCoM
    LabelCoord  = "r"
    RhoProfile  = RhoHist[i_final, :, j_mid]
    PProfile    = PHist[i_final, :, j_mid]
    MidplaneStr = f"Z ≈ {ZCoM[j_mid]:.3e}"
    print(f"Extracting final profile along r-axis at {MidplaneStr}.")
else:
    raise ValueError("Axis must be 'r' or 'Z'.")

# --- Load analytic Sedov solution ---
ana_data = np.loadtxt(analytic_file)
r_ana    = ana_data[:, 0]
rho_ana  = ana_data[:, 1]
P_ana    = ana_data[:, 3]  # adjust if your file has a different column order

###############################################################
#######                                                 #######
#######           P L O T   C O S M E T I C S           #######
#######                                                 #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 25})
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
plt.rcParams['xtick.major.size']  = 0.5*20
plt.rcParams['xtick.minor.size']  = 0.5*8
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size']  = 0.5*20
plt.rcParams['ytick.minor.size']  = 0.5*8
plt.rcParams['xtick.major.pad']   = '15'
plt.rcParams['ytick.major.pad']   = '15'

plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linewidth'] = 2.5
plt.rcParams['grid.linestyle'] = '--'

# Make ticks appear on all sides and point inwards
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

###############################################################
#######                                                 #######
#######               P L O T  I T !                    #######
#######                                                 #######
###############################################################

fig, (ax_rho, ax_P) = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle(f"Sedov-Taylor blast test (t = {t_final:.3f})", fontsize = 25)

# --- Density profile ---
ax_rho.plot(Coord, RhoProfile, lw=2.5, color="tab:blue", label="Numerical")
ax_rho.plot(r_ana, rho_ana, lw=2, color="k", ls="--", label="Analytic")
ax_rho.set_xlabel(LabelCoord)
ax_rho.set_ylabel(r"Density $\rho$")
ax_rho.set_title(f"Density along {LabelCoord}-axis", fontsize = 24, y = 1.04)
ax_rho.legend()

# --- Pressure profile ---
ax_P.plot(Coord, PProfile, lw=2.5, color="tab:red", label="Numerical")
ax_P.plot(r_ana, P_ana, lw=2, color="k", ls="--", label="Analytic")
ax_P.set_xlabel(LabelCoord)
ax_P.set_ylabel(r"Pressure $P$")
ax_P.set_title(f"Pressure along {LabelCoord}-axis", fontsize = 24, y = 1.04)
ax_P.legend()


ax_rho.xaxis.set_minor_locator(AutoMinorLocator(5)) 
ax_rho.yaxis.set_minor_locator(AutoMinorLocator(4))

ax_P.xaxis.set_minor_locator(AutoMinorLocator(5))
ax_P.yaxis.set_minor_locator(AutoMinorLocator(4))

plt.tight_layout()
plt.subplots_adjust(top=0.83)

# --- Save or show ---
if SaveFigure:
    plt.savefig(OutName, dpi=200)
    print(f"Saved plot to {OutName}")
else:
    plt.show()
