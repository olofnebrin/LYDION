#print(">>> Sod script starting")

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ---------------- User inputs ----------------
script_dir      = os.path.dirname(os.path.abspath(__file__))
File256         = os.path.join(script_dir, "SodTest_256x032_t0.20.h5")
File1024        = os.path.join(script_dir, "SodTest_1024x032_t0.20.h5")
AnalyticalFile  = os.path.join(script_dir, "SodTestAnalytical_EvanOConnor.dat")

Axis        = "r"   # profile direction; here "r" = X-direction in your 2D grid
SaveFigure  = True 
OutName     = os.path.join(script_dir, "SodTest.png")
# ---------------------------------------------

AnalyticalData = np.loadtxt( AnalyticalFile, comments = "#" )
XAnalytical    = AnalyticalData[:,1]
rhoAnalytical  = AnalyticalData[:,2]
PAnalytical    = AnalyticalData[:,3]
uAnalytical    = AnalyticalData[:,4]

# --- Read numerical data ---
with h5py.File(File256, "r") as f:
    T        = f["Times"][:]
    RhoHist  = f["GasDensity_History"][:]   
    PHist    = f["P_History"][:]           
    uXHist   = f["rVelocity_History"][:]        
    RCoM     = f["rCoM"][:]               
    ZCoM     = f["ZCoM"][:]               

# --- Ensure correct axis order: (N_t, N_r, N_Z) ---
def ensure_NtNrNz(arr):
    if arr.shape[2] == len(T):
        # stored as (N_r_or_Z, N_Z_or_r, N_t)
        arr = arr.transpose(2, 0, 1)  # now (N_t, ?, ?)
    elif arr.shape[0] == len(T):
        # already (N_t, ?, ?)
        pass
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} vs len(T)={len(T)}")
    return arr

RhoHist = ensure_NtNrNz(RhoHist)
PHist   = ensure_NtNrNz(PHist)
uXHist  = ensure_NtNrNz(uXHist)

Nt, N1, N2 = RhoHist.shape

# Decide which axis is r and which is Z by matching RCoM / ZCoM lengths
if N1 == len(RCoM) and N2 == len(ZCoM):
    # good: axis 1 = r, axis 2 = Z
    Nr, Nz = N1, N2
elif N1 == len(ZCoM) and N2 == len(RCoM):
    # swapped: axis 1 = Z, axis 2 = r → fix by transposing axes 1 and 2
    RhoHist = RhoHist.transpose(0, 2, 1)
    PHist   = PHist.transpose(0, 2, 1)
    uXHist  = uXHist.transpose(0, 2, 1)
    Nt, Nr, Nz = RhoHist.shape
else:
    raise ValueError(
        f"Could not match history shape {RhoHist.shape} to RCoM({len(RCoM)}) / ZCoM({len(ZCoM)})."
    )


RhoHist = ensure_NtNrNz(RhoHist)
PHist   = ensure_NtNrNz(PHist)
uXHist  = ensure_NtNrNz(uXHist)

Nt, Nr, Nz = RhoHist.shape
i_final = Nt - 1
t_final = T[i_final]

# --- Extract profile along X (r) at Z ≈ 0.5 ---
if Axis.lower() == "r":
    j_mid = np.argmin(np.abs(ZCoM - 0.5))  
    Coord       = RCoM
    LabelCoord  = "X"
    RhoProfile  = RhoHist[i_final, :, j_mid]
    PProfile    = PHist[i_final, :, j_mid]
    UProfile    = uXHist[i_final, :, j_mid]
    MidplaneStr = f"Z ≈ {ZCoM[j_mid]:.3e}"
    print(f"Extracting final profiles along X-axis at {MidplaneStr}.")
else:
    raise ValueError("Axis must be 'r' (X-direction) for this setup.")


###############################################################
#######                                                 #######
#######           P L O T   C O S M E T I C S           #######
#######                                                 #######
###############################################################

# COMMENT THIS OUT IF YOU DON'T HAVE THE REQUIRED FONTS:

#plt.rc('font',**{'sans-serif':['RomanD'],
                 #'size': 27} )
#plt.rc('font',**{'sans-serif':['AVHershey Simplex'],
                #'size': 27} )
plt.rc('font',**{'sans-serif':['AVHershey Complex'],
                'size': 22} )
#plt.rc('font',**{'sans-serif':['Complex'],
                # 'size': 27} )

params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

# The stuff below need not be removed:

params = {'legend.fontsize': 19 }
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
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'

plt.rcParams['legend.frameon'] = 'False'

plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linewidth'] = 3.2
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

fig, (ax_rho, ax_P, ax_u) = plt.subplots(
    3, 1, figsize=(8, 11), sharex=True
)
fig.suptitle(f"Sod shock tube test (t = {t_final:.3f})", fontsize = 24)

# --- Density profile ---
ax_rho.plot(Coord, RhoProfile, lw=2.5, label="Density")
ax_rho.plot(XAnalytical, rhoAnalytical, lw = 2.5, linestyle = 'dashed', color = 'black')
ax_rho.set_ylabel(r"$\rho$", labelpad = 20)
#ax_rho.set_title(f"Density along {LabelCoord}-axis", fontsize=18, y = 1.04)
ax_rho.set_ylim( -0.1, 1.1)
ax_rho.set_xlim( 0.0, 1.0 )
ax_rho.xaxis.set_minor_locator(AutoMinorLocator(4)) 
ax_rho.yaxis.set_minor_locator(AutoMinorLocator(5))

# --- Pressure profile ---
ax_P.plot(Coord, PProfile, lw=2.5, label="Pressure", color="tab:red")
ax_P.plot(XAnalytical, PAnalytical, lw = 2.5, linestyle = 'dashed', color = 'black')
ax_P.set_ylabel("P", labelpad = 20)
#ax_P.set_title(f"Pressure along {LabelCoord}-axis", fontsize=18, y = 1.04)
ax_P.set_ylim( -0.1, 1.1)
ax_P.xaxis.set_minor_locator(AutoMinorLocator(4)) 
ax_P.yaxis.set_minor_locator(AutoMinorLocator(5))

# --- Velocity profile ---
ax_u.plot(Coord, UProfile, lw=2.5, label=r"Velocity $u_x$", color="tab:green")
ax_u.plot(XAnalytical, uAnalytical, lw = 2.5, linestyle = 'dashed', color = 'black')
ax_u.set_xlabel(LabelCoord, labelpad = 20)
ax_u.set_ylabel(r"$u_X$", labelpad = 20)
#ax_u.set_title(f"Velocity along {LabelCoord}-axis", fontsize=18, y = 1.04)
ax_u.set_ylim( -0.1, 1.1)
ax_u.xaxis.set_minor_locator(AutoMinorLocator(4)) 
ax_u.yaxis.set_minor_locator(AutoMinorLocator(5))

plt.tight_layout()
plt.subplots_adjust(top=0.92)

# --- Save or show ---
if SaveFigure:
    plt.savefig(OutName, dpi=600)
    print(f"Saved plot to {OutName}")
else:
    plt.show()
