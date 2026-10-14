import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ---------------- User inputs ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
filename   = os.path.join(script_dir, "SedovTest_E1.00e+00_rho1_gamma1.67_t0.4_512x512.h5")
analytic_file = os.path.join(script_dir, "SedovTaylorAnalytical_E1.0_rho1.0_t0.4_gamma53.txt")

SphericalAverage = True
N_shells    = None   # None -> use Nr radial shells
SaveFigure  = True
OutName     = os.path.join(script_dir, "SedovFinalProfile_Analytic_SphericalAverage.pdf")

# Manual legend placement
LegendLocRho    = "upper left"
LegendBBoxRho   = (0.014, 0.98)

LegendLocP      = "upper left"
LegendBBoxP     = (0.014, 0.98)

# Manual time-label placement inside panels
ShowTimeLabels  = True
TimeLabelText   = None   # None -> use f"t = {t_final:.3f}"

TimeLabelXRho   = 0.75
TimeLabelYRho   = 0.88

TimeLabelXP     = 0.75
TimeLabelYP     = 0.88

TimeLabelFontsize = 25
# ---------------------------------------------

# --- Small helpers ---
def ensure_NtNrNz(arr, Nt):
    """Return history array with shape (Nt, Nr, Nz)."""
    if arr.shape[2] == Nt:
        return arr.transpose(2, 0, 1)
    elif arr.shape[0] == Nt:
        return arr
    else:
        raise ValueError(f"Unexpected history array shape {arr.shape} vs len(T)={Nt}")


def ensure_NrNz(arr, Nr, Nz, name="array"):
    """Return 2D array with shape (Nr, Nz)."""
    arr = np.asarray(arr)

    if arr.ndim == 2:
        if arr.shape == (Nr, Nz):
            return arr
        if arr.shape == (Nz, Nr):
            return arr.T

    if arr.ndim == 3:
        # Allow accidentally time-dependent saved volumes, using final snapshot.
        if arr.shape[0] == Nr and arr.shape[1] == Nz:
            return arr[:, :, -1]
        if arr.shape[1] == Nr and arr.shape[2] == Nz:
            return arr[-1, :, :]
        if arr.shape[0] == Nz and arr.shape[1] == Nr:
            return arr[:, :, -1].T
        if arr.shape[1] == Nz and arr.shape[2] == Nr:
            return arr[-1, :, :].T

    raise ValueError(f"Unexpected {name} shape {arr.shape}; expected something compatible with (Nr,Nz)=({Nr},{Nz}).")


def read_cell_volumes(h5file, Nr, Nz):
    """Read saved cell volumes, allowing a few common key names."""
    volume_keys = [
        "CellVolumes",
        "CellVolume",
        "Cell_Volumes",
        "CellVolumes_History",
        "Volumes",
        "Volume",
    ]

    for key in volume_keys:
        if key in h5file:
            return ensure_NrNz(h5file[key][:], Nr, Nz, name=key), key

    available = ", ".join(h5file.keys())
    raise KeyError(
        "Could not find saved cell volumes. Tried keys: "
        + ", ".join(volume_keys)
        + f"\nAvailable keys are:\n{available}"
    )


def spherical_average_profile(Q, CellVolumes, RCoM, ZCoM, N_shells=None):
    """
    Volume-weighted spherical average of a cell-centered quantity Q(r,z).

    The spherical radius is s = sqrt(r^2 + Z^2). We only average out to the
    largest radius for which shells are fully contained in the rectangular
    cylindrical domain, i.e. min(max(r), max(|Z|)).
    """
    Nr, Nz = Q.shape
    if N_shells is None:
        N_shells = Nr

    RR, ZZ = np.meshgrid(RCoM, ZCoM, indexing="ij")
    S = np.sqrt(RR**2 + ZZ**2)

    R_profile_max = min(np.max(RCoM), np.max(np.abs(ZCoM)))
    ShellEdges = np.linspace(0.0, R_profile_max, N_shells + 1)
    ShellCoM   = 0.5 * (ShellEdges[:-1] + ShellEdges[1:])

    shell_index = np.searchsorted(ShellEdges, S, side="right") - 1
    valid = (shell_index >= 0) & (shell_index < N_shells) & np.isfinite(Q) & np.isfinite(CellVolumes)

    weighted_Q = np.bincount(
        shell_index[valid].ravel(),
        weights=(Q[valid] * CellVolumes[valid]).ravel(),
        minlength=N_shells,
    )
    shell_volumes = np.bincount(
        shell_index[valid].ravel(),
        weights=CellVolumes[valid].ravel(),
        minlength=N_shells,
    )

    Q_sph = np.full(N_shells, np.nan)
    nonzero = shell_volumes > 0.0
    Q_sph[nonzero] = weighted_Q[nonzero] / shell_volumes[nonzero]

    return ShellCoM, Q_sph, shell_volumes


# --- Read numerical data ---
with h5py.File(filename, "r") as f:
    T       = f["Times"][:]
    RhoHist = f["GasDensity_History"][:]
    PHist   = f["P_History"][:]
    RCoM    = f["rCoM"][:]
    ZCoM    = f["ZCoM"][:]

    # --- Ensure correct axis order: (N_saved, N_r, N_Z) ---
    RhoHist = ensure_NtNrNz(RhoHist, len(T))
    PHist   = ensure_NtNrNz(PHist, len(T))

    Nt, Nr, Nz = RhoHist.shape
    CellVolumes, VolumeKey = read_cell_volumes(f, Nr, Nz)

print(f"Read cell volumes from HDF5 key: {VolumeKey}")

i_final = Nt - 1
t_final = T[i_final]

# --- Extract final profile ---
if SphericalAverage:
    Coord, RhoProfile, ShellVolumes = spherical_average_profile(
        RhoHist[i_final, :, :],
        CellVolumes,
        RCoM,
        ZCoM,
        N_shells=N_shells,
    )
    _, PProfile, _ = spherical_average_profile(
        PHist[i_final, :, :],
        CellVolumes,
        RCoM,
        ZCoM,
        N_shells=N_shells,
    )
    LabelCoord = "Radial distance"
    ProfileTitle = "Spherically averaged"
    print(f"Computed volume-weighted spherical averages using {len(Coord)} shells.")
else:
    j_mid = np.argmin(np.abs(ZCoM - 0.0))  # choose plane closest to Z=0
    Coord       = RCoM
    LabelCoord  = "Radial distance"
    RhoProfile  = RhoHist[i_final, :, j_mid]
    PProfile    = PHist[i_final, :, j_mid]
    ProfileTitle = "Along r-axis"
    MidplaneStr = f"Z ≈ {ZCoM[j_mid]:.3e}"
    print(f"Extracting final profile along r-axis at {MidplaneStr}.")

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
plt.rcParams['axes.linewidth'] = 2.1
plt.rcParams['grid.linewidth'] = 2.1
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

TimeLabel = TimeLabelText
if TimeLabel is None:
    TimeLabel = f"t = {t_final:.3f}"

# --- Density profile ---
ax_rho.plot(Coord, RhoProfile, lw=3.0, color="tab:blue", label=r"LYDION 512$^2$")
line_rho_ana, = ax_rho.plot(r_ana, rho_ana, lw=2, color="k", ls="--", label="Analytical")
line_rho_ana.set_dash_capstyle("round")

ax_rho.set_xlabel(LabelCoord, labelpad = 10)
ax_rho.set_ylabel("")
ax_rho.set_title(f"{ProfileTitle} density", fontsize = 24, y = 1.04)

ax_rho.legend(
    loc=LegendLocRho,
    bbox_to_anchor=LegendBBoxRho,
    bbox_transform=ax_rho.transAxes,
)

if ShowTimeLabels:
    ax_rho.text(
        TimeLabelXRho,
        TimeLabelYRho,
        TimeLabel,
        transform=ax_rho.transAxes,
        fontsize=TimeLabelFontsize,
        ha="center",
        va="center",
    )

# --- Pressure profile ---
ax_P.plot(Coord, PProfile, lw=3.0, color="tab:red", label=r"LYDION 512$^2$")
line_P_ana, = ax_P.plot(r_ana, P_ana, lw=2, color="k", ls="--", label="Analytical")
line_P_ana.set_dash_capstyle("round")

ax_P.set_xlabel(LabelCoord, labelpad = 10)
ax_P.set_ylabel("")
ax_P.set_title(f"{ProfileTitle} pressure", fontsize = 24, y = 1.04)

ax_P.legend(
    loc=LegendLocP,
    bbox_to_anchor=LegendBBoxP,
    bbox_transform=ax_P.transAxes,
)

if ShowTimeLabels:
    ax_P.text(
        TimeLabelXP,
        TimeLabelYP,
        TimeLabel,
        transform=ax_P.transAxes,
        fontsize=TimeLabelFontsize,
        ha="center",
        va="center",
    )

ax_rho.xaxis.set_minor_locator(AutoMinorLocator(5)) 
ax_rho.yaxis.set_minor_locator(AutoMinorLocator(4))

ax_P.xaxis.set_minor_locator(AutoMinorLocator(5))
ax_P.yaxis.set_minor_locator(AutoMinorLocator(4))

plt.tight_layout(pad = 0.4)
plt.subplots_adjust(top=0.88)

# --- Save or show ---
if SaveFigure:
    plt.savefig(OutName, dpi=300)
    print(f"Saved plot to {OutName}")
else:
    plt.show()