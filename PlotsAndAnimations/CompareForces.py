import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
os.chdir(repo_root)

filename = "RHDSim_Lya_128x128_2026-06-04.h5"
#filename = "RHDSim_Lya_NoDust_u4_256x256_2026-05-06.h5" 
#filename = "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5" 
#filename =  "RHDSim_Lya_128x128_2026-04-26.h5"
#filename = "RHDSim_Lya_128x128_2026-04-10_MostlyH2.h5"

# CONSTANTS:
c    = 2.99792458e10
Lsun = 3.839e33
mH   = 1.67e-24
X    = 0.76
yr   = 3.1556926e7
kyr  = 1e3 * yr
kms  = 1e5

# Saved figure file name:
output_filename = "LYDION_Figure_FeedbackComparison_Lya_512x512.pdf"

def read_dataset_if_exists(file, name, default=None):
    return file[name][:] if name in file else default


def read_attr_if_exists(group, name, default=None):
    return group.attrs[name] if name in group.attrs else default


with h5py.File(filename, "r") as file:
    settings = file["Settings"]

    # Settings:
    raw_gravity = read_attr_if_exists(settings, "Gravity", b"false")
    Gravity = raw_gravity.decode().lower() == "true"

    # Time:
    t = file["Times"][:]

    # Bolometric luminosity:
    L_bol = read_attr_if_exists(settings, "L_bol", None)
    if L_bol is None:
        raise RuntimeError("Could not find Settings.attrs['L_bol'] in the HDF5 file.")

    # Lyα force multipliers & luminosity:
    M_F_Lya        = read_dataset_if_exists(file, "M_F_History")
    M_F_Lya_radial = read_dataset_if_exists(file, "M_F_radial_net_History")
    M_F_Lya_anti   = read_dataset_if_exists(file, "M_F_History_antigrav_direction")
    L_Lya          = read_dataset_if_exists(file, "L_Lya_History")

    # Direct radiation force multipliers:
    M_F_dir        = read_dataset_if_exists(file, "M_F_dir_History")
    M_F_dir_radial = read_dataset_if_exists(file, "M_F_dir_radial_net_History")
    M_F_dir_anti   = read_dataset_if_exists(file, "M_F_dir_antigrav_direction_History")

    # IR radiation force multipliers:
    M_F_IR         = read_dataset_if_exists(file, "M_F_IR_History")
    M_F_IR_radial  = read_dataset_if_exists(file, "M_F_IR_radial_net_History")
    M_F_IR_anti    = read_dataset_if_exists(file, "M_F_IR_History_antigrav_direction")
    L_IR           = read_dataset_if_exists(file, "L_IR_History")

    # Gas pressure forces:
    IntegratedPressureForce_radial_net = read_dataset_if_exists(file, "IntegratedPressureForce_radial_net_History")
    IntegratedPressureForce_antigrav   = read_dataset_if_exists(file, "IntegratedPressureForce_antigrav_History")

    # Gravity:
    IntegratedGravityForce_total      = read_dataset_if_exists(file, "IntegratedGravityForce_History")
    IntegratedGravityForce_radial_net = read_dataset_if_exists(file, "IntegratedGravityForce_radial_net_History")


# ------------------------------------------------------------
# Convert to forces
# ------------------------------------------------------------

ForceScale = L_bol / c

# Naive feedback forces:
F_Lya_naive = M_F_Lya * L_Lya / c if (M_F_Lya is not None and L_Lya is not None) else None
F_dir_naive = M_F_dir * L_bol / c if M_F_dir is not None else None
F_IR_naive  = M_F_IR  * L_IR  / c if (M_F_IR is not None and L_IR is not None) else None

# Radial-net feedback forces:
F_Lya_radial = M_F_Lya_radial * L_Lya / c if (M_F_Lya_radial is not None and L_Lya is not None) else None
F_dir_radial = M_F_dir_radial * L_bol / c if M_F_dir_radial is not None else None
F_IR_radial  = M_F_IR_radial  * L_IR  / c if (M_F_IR_radial is not None and L_IR is not None) else None
F_gas_radial = IntegratedPressureForce_radial_net

# Anti-gravity feedback forces:
F_Lya_anti = M_F_Lya_anti * L_Lya / c if (M_F_Lya_anti is not None and L_Lya is not None) else None
F_dir_anti = M_F_dir_anti * L_bol / c if M_F_dir_anti is not None else None
F_IR_anti  = M_F_IR_anti  * L_IR  / c if (M_F_IR_anti is not None and L_IR is not None) else None
F_gas_anti = IntegratedPressureForce_antigrav

# Gravity diagnostics:
F_grav_total  = IntegratedGravityForce_total
F_grav_inward = None
if IntegratedGravityForce_radial_net is not None:
    F_grav_inward = -IntegratedGravityForce_radial_net


# ------------------------------------------------------------
# Time range
# ------------------------------------------------------------

nonzero_idx = np.nonzero(t)[0]
if len(nonzero_idx) == 0:
    last_idx = len(t)
else:
    last_idx = nonzero_idx[-1] + 1

t_plot = t[:last_idx] / kyr


# ------------------------------------------------------------
# Plot cosmetics
# ------------------------------------------------------------

# COMMENT THIS OUT IF YOU DON'T HAVE THE REQUIRED FONTS:

#plt.rc('font',**{'sans-serif':['RomanD'],
#                 'size': 27} )
#plt.rc('font',**{'sans-serif':['AVHershey Simplex'],
#                 'size': 27} )
plt.rc('font',**{'sans-serif':['AVHershey Complex'],
                 'size': 37} )
#plt.rc('font',**{'sans-serif':['Complex'],
#                 'size': 27} )

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
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.width'] = 1.7
plt.rcParams['ytick.minor.width'] = 1.7
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['ytick.minor.size'] = 8
plt.rcParams['xtick.major.pad'] = '15'
plt.rcParams['ytick.major.pad'] = '15'

plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.linewidth'] = 2.5

# Make ticks appear on all sides and point inwards
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

ymin_plot = 4e-4
ymax_plot = 40.0


def normalized_visible(y):
    if y is None:
        return False
    yn = np.asarray(y[:last_idx], dtype=float) / ForceScale
    mask = np.isfinite(yn) & (yn > ymin_plot) & (yn < ymax_plot)
    return np.any(mask)


def plot_force(ax, y, color, lw=2.5, ls='-'):
    if y is None:
        return
    yn = np.asarray(y[:last_idx], dtype=float) / ForceScale
    mask = np.isfinite(yn) & (yn > 0.0)
    if np.any(mask):
        ax.semilogy(t_plot[mask], yn[mask],
                    linewidth=lw, color=color, linestyle=ls,
                    dash_capstyle='round')


def get_curve_y(y, at_x_idx):
    """Get normalized y value of a curve near a given time index."""
    yn = np.asarray(y[:last_idx], dtype=float) / ForceScale
    mask = np.isfinite(yn) & (yn > 0.0)
    if not np.any(mask):
        return None
    valid_idx = np.where(mask)[0]
    idx = valid_idx[np.argmin(np.abs(valid_idx - at_x_idx))]
    return yn[idx]


def get_label_y(y, factor_up=1.0, cap=100.0):
    yn = np.asarray(y[:last_idx], dtype=float) / ForceScale
    mask = np.isfinite(yn) & (yn > ymin_plot) & (yn < ymax_plot)
    if not np.any(mask):
        return None
    return min(factor_up * np.max(yn[mask]), cap)


def nudge_label_y(y_desired, used_positions, min_sep_decades=0.35):
    """Nudge y_desired in log space to avoid overlap with already-placed labels."""
    if y_desired is None:
        return None
    log_y = np.log10(y_desired)
    for sign in [+1, -1]:
        log_try = log_y
        for i in range(20):
            too_close = any(abs(log_try - np.log10(p)) < min_sep_decades
                            for p in used_positions)
            if not too_close:
                val = 10**log_try
                if ymin_plot < val < ymax_plot:
                    return val
            log_try = log_y + sign * min_sep_decades * (i + 1)
    return None


# Arrow x-positions: label at x_text, arrow tip at x_arrow on the curve
x_text  = 8.0
x_arrow = 4.0
arrow_idx = np.argmin(np.abs(t_plot - x_arrow))


def add_force_label(ax, y, label, color, factor_up, used_positions,
                    use_arrow=True,
                    x_text=x_text, x_arrow=x_arrow,
                    fontsize=32,
                    arrow_lw=2.0,
                    connectionstyle='angle,angleA=0,angleB=90,rad=0'):
    """Add a colored force label.

    If use_arrow=True, draw an elbow/angled connector from the text to the
    curve. If use_arrow=False, only draw the text label at the automatically
    chosen y-position.
    """
    if not normalized_visible(y):
        return

    raw_pos = get_label_y(y, factor_up=factor_up, cap=ymax_plot * 0.8)
    if raw_pos is None:
        return

    label_y = nudge_label_y(raw_pos, used_positions)
    if label_y is None:
        return
    used_positions.append(label_y)

    if not use_arrow:
        ax.text(x_text, label_y, label,
                color=color,
                fontsize=fontsize,
                va='center',
                ha='left',
                clip_on=False)
        return

    arrow_idx_local = np.argmin(np.abs(t_plot - x_arrow))
    curve_y = get_curve_y(y, arrow_idx_local)
    if curve_y is None:
        return

    ax.annotate(label,
                xy=(x_arrow, curve_y),
                xytext=(x_text, label_y),
                color=color,
                fontsize=fontsize,
                va='center',
                ha='left',
                arrowprops=dict(arrowstyle='-|>',
                                color=color,
                                lw=arrow_lw,
                                shrinkA=0,
                                shrinkB=0,
                                connectionstyle=connectionstyle),
                annotation_clip=False)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure()
fig, ax = plt.subplots()
fig.set_size_inches(14.2, 8.5)

ax.set_title(r"Does Ly$\alpha$ dominate the radiation pressure?", y=1.05)

lw_nonLya = 3.0

# Solid lines: radial-net feedback forces
plot_force(ax, F_dir_naive, "dodgerblue", lw=lw_nonLya, ls='--')
plot_force(ax, F_IR_naive,  "orangered",  lw=lw_nonLya, ls='--') 

plot_force(ax, F_dir_radial,  "dodgerblue", lw=lw_nonLya, ls='-')
plot_force(ax, F_IR_radial,   "orangered",  lw=lw_nonLya, ls='-')
#plot_force(ax, F_gas_radial,  "limegreen",  lw=2.5, ls='-')
plot_force(ax, F_Lya_naive,  "crimson",  lw=lw_nonLya, ls='--')
plot_force(ax, F_Lya_radial,  "crimson",    lw=4.0, ls='-')

# Dashed lines: anti-gravity feedback forces
if Gravity:
    plot_force(ax, F_dir_anti, "dodgerblue", lw=lw_nonLya, ls=':')
    plot_force(ax, F_IR_anti,  "orangered",  lw=lw_nonLya, ls=':')
    #plot_force(ax, F_gas_anti, "limegreen",  lw=lw_nonLya, ls=':')
    plot_force(ax, F_Lya_anti, "crimson",    lw=4.0, ls=':')
    #plot_force(ax, F_grav_inward, "black", lw=lw_nonLya, ls='-')
    #plot_force(ax, F_grav_total,  "black", lw=lw_nonLya, ls='--')


# ------------------------------------------------------------
# Colored text labels with arrows, avoiding overlap
# ------------------------------------------------------------

used_positions = []

add_force_label(ax, F_Lya_radial,  "Lyα radiation pressure",   "crimson",    factor_up=1.9, used_positions=used_positions, use_arrow=True)
add_force_label(ax, F_dir_radial,  "Direct radiation pressure", "dodgerblue", factor_up=0.5, used_positions=used_positions, use_arrow=False)
add_force_label(ax, F_IR_radial,   "IR radiation pressure",     "orangered",  factor_up=1.5, used_positions=used_positions, use_arrow=True)
#add_force_label(ax, F_gas_radial,  "Gas pressure",              "limegreen",  factor_up=1.2, used_positions=used_positions, use_arrow=False)
#if Gravity:
#    add_force_label(ax, F_grav_inward, "Gravity",               "black",      factor_up=1.2, used_positions=used_positions, use_arrow=True)


# ------------------------------------------------------------
# Axis labels and limits
# ------------------------------------------------------------

print("L_bol", L_bol / Lsun, "Lsun")

ax.set_xlabel("Time (kyr)", labelpad=20)
ax.set_ylabel("Force (in units of L$_{bol}$/c)", labelpad=20)
ax.text(0.29, 0.44, "Simulation: SCLyaR512", color="gray",
        ha='center', va='center', transform=ax.transAxes, fontsize=36)

ax.set_ylim(ymin_plot, ymax_plot)
ax.set_xlim(-5.0, 80.0)

yl = [1e-3, 0.01, 0.1, 1.0, 10.0]
ylabels = ['10$^{-3}$', '0.01', '0.1', '1', '10']
ax.set_yticks(yl)
ax.set_yticklabels(ylabels)

# Set different lengths for major and minor ticks
ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction='in', length=20, width=2.5)
ax.tick_params(axis='both', which='minor', direction='in', length=10, width=2.5)

plt.tight_layout(pad=0.4)
plt.savefig(output_filename, bbox_inches="tight", dpi = 300)
#plt.savefig("FeedbackComparison.png", bbox_inches="tight")
plt.show()