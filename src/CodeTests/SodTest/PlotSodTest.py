import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ---------------- User inputs ----------------
script_dir      = os.path.dirname(os.path.abspath(__file__))
File128         = os.path.join(script_dir, "SodTest_128x032_t0.20.h5")
File256         = os.path.join(script_dir, "SodTest_256x032_t0.20.h5")
File512         = os.path.join(script_dir, "SodTest_512x032_t0.20.h5")
File1024        = os.path.join(script_dir, "SodTest_1024x032_t0.20.h5")
File2048        = os.path.join(script_dir, "SodTest_2048x032_t0.20.h5")
AnalyticalFile  = os.path.join(script_dir, "SodTestAnalytical_EvanOConnor.dat")

Axis        = "r"   # profile direction; here "r" = X-direction in your 2D grid
SaveFigure  = True 
OutName     = os.path.join(script_dir, "SodTest.pdf")
# ---------------------------------------------

# ---------- Analytical solution ----------
AnalyticalData = np.loadtxt(AnalyticalFile, comments="#")
XAnalytical    = AnalyticalData[:, 1]
rhoAnalytical  = AnalyticalData[:, 2]
PAnalytical    = AnalyticalData[:, 3]
uAnalytical    = AnalyticalData[:, 4]

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
def load_run_1D_profile(filename, Axis="r"):
    with h5py.File(filename, "r") as f:
        T       = f["Times"][:]
        RhoHist = f["GasDensity_History"][:]
        PHist   = f["P_History"][:]
        uXHist  = f["rVelocity_History"][:]
        RCoM    = f["rCoM"][:]
        ZCoM    = f["ZCoM"][:]

    # Ensure (Nt, Nr, Nz)
    RhoHist = ensure_NtNrNz(RhoHist, len(T))
    PHist   = ensure_NtNrNz(PHist,   len(T))
    uXHist  = ensure_NtNrNz(uXHist,  len(T))

    Nt, N1, N2 = RhoHist.shape

    # Decide which axis is r and which is Z by matching RCoM / ZCoM lengths
    if N1 == len(RCoM) and N2 == len(ZCoM):
        Nr, Nz = N1, N2
    elif N1 == len(ZCoM) and N2 == len(RCoM):
        # swapped: axis 1 = Z, axis 2 = r → fix by transposing axes 1 and 2
        RhoHist = RhoHist.transpose(0, 2, 1)
        PHist   = PHist.transpose(0, 2, 1)
        uXHist  = uXHist.transpose(0, 2, 1)
        Nt, Nr, Nz = RhoHist.shape
    else:
        raise ValueError(
            f"Could not match history shape {RhoHist.shape} to "
            f"RCoM({len(RCoM)}) / ZCoM({len(ZCoM)})."
        )

    i_final = Nt - 1
    t_final = T[i_final]

    if Axis.lower() == "r":
        j_mid = np.argmin(np.abs(ZCoM - 0.5))
        Coord      = RCoM
        RhoProfile = RhoHist[i_final, :, j_mid]
        PProfile   = PHist[i_final, :, j_mid]
        UProfile   = uXHist[i_final, :, j_mid]
    else:
        raise ValueError("Axis must be 'r' (X-direction) for this setup.")

    return Coord, RhoProfile, PProfile, UProfile, t_final

# Load 128 x 32 run
Coord128, Rho128, P128, U128, t_final_128 = load_run_1D_profile(File128, Axis)

# Load 256 x 32 run
Coord256, Rho256, P256, U256, t_final_256 = load_run_1D_profile(File256, Axis)

# Load 512 x 32 run
Coord512, Rho512, P512, U512, t_final_512 = load_run_1D_profile(File512, Axis)

# Load 1024 x 32 run
Coord1024, Rho1024, P1024, U1024, t_final_1024 = load_run_1D_profile(File1024, Axis)

# Load 2048 x 32 run
Coord2048, Rho2048, P2048, U2048, t_final_2048 = load_run_1D_profile(File2048, Axis)

# Use the 256 run time for the title (they should be ~identical)
t_final = t_final_256

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

fig, axs = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=False)
(ax_rho, ax_P), (ax_u, ax_Prho) = axs

fig.suptitle(f"Sod shock tube test (t = {t_final:.3f})", fontsize=30)

# --- Colors / labels for the two resolutions ---
color128  = "limegreen"
color256  = "dodgerblue"
color512  = "crimson"
color1024 = "gray"
color2048 = "deeppink"
colorAna  = "black"
lw_num    = 3.0
lw_Ana    = 3.0
alpha     = 0.8

label128  = r"$N_x = 128$"
label256  = r"$N_x = 256$"
label512  = r"$N_x = 512$"
label1024 = r"$N_x = 1024$"
label2048 = r"$N_x = 2048$"
labelAna  = "Analytical"

# --- Density ---
ax_rho.plot(Coord128,  Rho128,  lw=lw_num, color=color128,  label=label128, alpha = alpha)
ax_rho.plot(Coord256,  Rho256,  lw=lw_num, color=color256,  label=label256, alpha = alpha)
ax_rho.plot(Coord512,  Rho512,  lw=lw_num, color=color512,  label=label512, alpha = alpha)
ax_rho.plot(Coord1024, Rho1024, lw=lw_num, color=color1024, label=label1024, alpha = alpha)
ax_rho.plot(Coord2048, Rho2048, lw=lw_num, color=color2048, label=label2048, alpha = alpha)
ax_rho.plot(XAnalytical, rhoAnalytical, lw=lw_Ana, linestyle='dashed', dash_capstyle = 'round',
            color=colorAna, label=labelAna)

ax_rho.set_ylabel(r"$\rho$", labelpad=20)
ax_rho.set_xlim(0.0, 1.0)
ax_rho.set_ylim(-0.1, 1.1)
ax_rho.xaxis.set_minor_locator(AutoMinorLocator(4))
ax_rho.yaxis.set_minor_locator(AutoMinorLocator(5))

# --- Pressure ---
ax_P.plot(Coord128,  P128,  lw=lw_num, color=color128,  label=label128, alpha = alpha)
ax_P.plot(Coord256,  P256,  lw=lw_num, color=color256,  label=label256, alpha = alpha)
ax_P.plot(Coord512,  P512,  lw=lw_num, color=color512,  label=label512, alpha = alpha)
ax_P.plot(Coord1024, P1024, lw=lw_num, color=color1024, label=label1024, alpha = alpha)
ax_P.plot(Coord2048, P2048, lw=lw_num, color=color2048, label=label2048, alpha = alpha)
ax_P.plot(XAnalytical, PAnalytical, lw=lw_Ana, linestyle='dashed', dash_capstyle = 'round',
          color=colorAna, label=labelAna)

ax_P.set_ylabel(r"$P$", labelpad=20)
ax_P.set_ylim(-0.1, 1.1)
ax_P.xaxis.set_minor_locator(AutoMinorLocator(4))
ax_P.yaxis.set_minor_locator(AutoMinorLocator(5))

# --- Velocity ---
ax_u.plot(Coord128,  U128,  lw=lw_num, color=color128,  label=label128, alpha = alpha)
ax_u.plot(Coord256,  U256,  lw=lw_num, color=color256,  label=label256, alpha = alpha)
ax_u.plot(Coord512,  U512,  lw=lw_num, color=color512,  label=label512, alpha = alpha)
ax_u.plot(Coord1024, U1024, lw=lw_num, color=color1024, label=label1024, alpha = alpha)
ax_u.plot(Coord2048, U2048, lw=lw_num, color=color2048, label=label2048, alpha = alpha)
ax_u.plot(XAnalytical, uAnalytical, lw=lw_Ana, linestyle='dashed', dash_capstyle = 'round',
          color=colorAna, label=labelAna)

ax_u.set_xlabel("X", labelpad=20)
ax_u.set_ylabel(r"$u_X$", labelpad=20)
ax_u.xaxis.set_minor_locator(AutoMinorLocator(4))
ax_u.yaxis.set_minor_locator(AutoMinorLocator(5))

# --- P/rho (≈ T up to constant factor) ---
ax_Prho.plot(Coord128,  P128 / Rho128,  lw=lw_num, color=color128,  label=label128, alpha = alpha)
ax_Prho.plot(Coord256,  P256 / Rho256,    lw=lw_num, color=color256,  label=label256, alpha = alpha)
ax_Prho.plot(Coord512,  P512 / Rho512,  lw=lw_num, color=color512,  label=label512, alpha = alpha)
ax_Prho.plot(Coord1024, P1024 / Rho1024,  lw=lw_num, color=color1024, label=label1024, alpha = alpha)
ax_Prho.plot(Coord2048, P2048 / Rho2048,  lw=lw_num, color=color2048, label=label2048, alpha = alpha)
ax_Prho.plot(XAnalytical, PAnalytical / rhoAnalytical, lw=lw_Ana, linestyle='dashed', dash_capstyle = 'round',
             color=colorAna, label=labelAna)

ax_Prho.set_xlabel("X", labelpad=20)
ax_Prho.set_ylabel(r"$P / \rho$", labelpad=20)
ax_Prho.set_ylim(0.6, 1.4)
ax_Prho.xaxis.set_minor_locator(AutoMinorLocator(4))
ax_Prho.yaxis.set_minor_locator(AutoMinorLocator(5))

# Put legends on each panel or one global; here: one global legend at top-right
handles, labels = ax_rho.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.92, 0.88))

plt.tight_layout()
plt.subplots_adjust(top=0.90)

# --- Save or show ---
if SaveFigure:
    #plt.savefig(OutName, dpi=600)
    plt.savefig(OutName)
    print(f"Saved plot to {OutName}")
else:
    plt.show()
