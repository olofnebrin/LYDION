import numpy as np
import matplotlib.pyplot as plt


from Nebrin2025_MF_Fit import M_F_fit


# ------------------------------------------------------------
# User choices
# ------------------------------------------------------------

N_HI_average = 2.5e23    # cm^-2
T            = 1e2       # K
D_Dsun       = 0.4*1e-3      # dust-to-gas ratio in Solar units
vmax         = 13.6       # km/s, cloud expansion/contraction velocity
b_s          = 1/3       # turbulent driving parameter

# Needed by the destruction probability p_d(...)
# Set consistently with the cloud model if desired.
nHI_average  = 1e6       # cm^-3

M_values = np.logspace(-2, 1.5, 300)   # 3D turbulent Mach number


# ------------------------------------------------------------
# Compute analytical fit
# ------------------------------------------------------------

MF_point = np.array([
    M_F_fit(
        N_HI_average=N_HI_average,
        T=T,
        D_Dsun=D_Dsun,
        taubar_s=0.5,
        vmax=vmax,
        M=M,
        b_s=b_s,
        nHI_average=nHI_average,
    )
    for M in M_values
])

MF_uniform = np.array([
    M_F_fit(
        N_HI_average=N_HI_average,
        T=T,
        D_Dsun=D_Dsun,
        taubar_s=1.0,
        vmax=vmax,
        M=M,
        b_s=b_s,
        nHI_average=nHI_average,
    )
    for M in M_values
])


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5.2, 4.0))

ax.loglog(M_values, MF_point,   lw=2.2, label=r"$\bar{\tau}_s = 0$ point source")
ax.loglog(M_values, MF_uniform, lw=2.2, ls="--", label=r"$\bar{\tau}_s = 1$ uniform source")

ax.set_xlabel(r"3D turbulent Mach number, $\mathcal{M}$", fontsize=13)
ax.set_ylabel(r"Radial force multiplier, $M_F$", fontsize=13)

title = (
    rf"$N_{{HI}}={N_HI_average:.1e}\,\mathrm{{cm^{{-2}}}}$, "
    rf"$D/D_\odot={D_Dsun:g}$, "
    rf"$v_{{max}}={vmax:g}\,\mathrm{{km\,s^{{-1}}}}$"
)
ax.set_title(title, fontsize=11)

ax.set_ylim(5.0, 100.0)

ax.tick_params(axis="both", which="major", labelsize=12, width=1.2, length=6)
ax.tick_params(axis="both", which="minor", width=1.0, length=3)

ax.legend(frameon=False, fontsize=11)

fig.tight_layout()
fig.savefig("MF_vs_Mach_AnalyticalFit.pdf", bbox_inches="tight")
plt.show()