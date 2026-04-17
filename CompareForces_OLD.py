import h5py
import numpy as np
import matplotlib.pyplot as plt

#filename = "IdealizedRHDSim_Turbulence_false_Lya_128x128_2026-04-08.h5"
filename = "RHDSim_NoLya_128x128_2026-04-15.h5"
#filename = "RHDSim_Lya_128x128_2026-04-10_MostlyH2.h5"

# CONSTANTS:
c    = 2.99792458e10
Lsun = 3.839e33
mH   = 1.67e-24
X    = 0.76
yr   = 3.1556926e7
kyr  = 1e3 * yr
kms  = 1e5


def read_dataset_if_exists(file, name, default=None):
    return file[name][:] if name in file else default


def read_attr_if_exists(group, name, default=None):
    return group.attrs[name] if name in group.attrs else default


with h5py.File(filename, "r") as file:
    settings = file["Settings"]

    # Settings:
    raw_gravity = read_attr_if_exists(settings, "Gravity", b"false")
    Gravity     = raw_gravity.decode().lower() == "true"

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
    IntegratedGravityForce_total       = read_dataset_if_exists(file, "IntegratedGravityForce_History")
    IntegratedGravityForce_radial_net  = read_dataset_if_exists(file, "IntegratedGravityForce_radial_net_History")


# ------------------------------------------------------------
# Convert to forces
# ------------------------------------------------------------

ForceScale = L_bol / c

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

#print("Gravity flag:", Gravity)
#print("F_grav_total:", F_grav_total)
#print("F_grav_inward:", F_grav_inward)

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

ymin_plot = 1e-4
ymax_plot = 1e2


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


def get_label_y(y, factor_up=1.0, cap=100.0):
    yn = np.asarray(y[:last_idx], dtype=float) / ForceScale
    mask = np.isfinite(yn) & (yn > ymin_plot) & (yn < ymax_plot)
    if not np.any(mask):
        return None
    return min(factor_up * np.max(yn[mask]), cap)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure()
fig, ax = plt.subplots()
fig.set_size_inches(12.5, 12.5)

ax.set_title("What feedback processes dominate?", y=1.05)

# Solid lines: radial-net feedback forces
plot_force(ax, F_dir_radial,  "dodgerblue", lw=2.5, ls='-')
plot_force(ax, F_IR_radial,   "orangered",  lw=2.5, ls='-')
plot_force(ax, F_gas_radial,  "limegreen",  lw=2.5, ls='-')
plot_force(ax, F_Lya_radial,  "crimson",    lw=4.0, ls='-')

# Dashed lines: anti-gravity feedback forces, if gravity diagnostics exist
if Gravity:
    plot_force(ax, F_dir_anti, "dodgerblue", lw=2.5, ls='--')
    plot_force(ax, F_IR_anti,  "orangered",  lw=2.5, ls='--')
    plot_force(ax, F_gas_anti, "limegreen",  lw=2.5, ls='--')
    plot_force(ax, F_Lya_anti, "crimson",    lw=4.0, ls='--')

    # Gravity itself:
    plot_force(ax, F_grav_inward, "black", lw=3.0, ls='-')
    plot_force(ax, F_grav_total,  "black", lw=3.0, ls='--')


# ------------------------------------------------------------
# Colored text labels, only if visible
# ------------------------------------------------------------

x_text = 10.0

if normalized_visible(F_Lya_radial):
    pos = get_label_y(F_Lya_radial, factor_up=1.5, cap=100.0)
    if pos is not None:
        ax.text(x_text, pos, "Lyα radiation pressure", color="crimson", fontsize=28)

if normalized_visible(F_dir_radial):
    pos = get_label_y(F_dir_radial, factor_up=0.5, cap=100.0)
    if pos is not None:
        ax.text(x_text, pos, "Direct radiation pressure", color="dodgerblue", fontsize=28)

if normalized_visible(F_IR_radial):
    pos = get_label_y(F_IR_radial, factor_up=1.5, cap=100.0)
    if pos is not None:
        ax.text(x_text, pos, "IR radiation pressure", color="orangered", fontsize=28)

if normalized_visible(F_gas_radial):
    pos = get_label_y(F_gas_radial, factor_up=1.2, cap=100.0)
    if pos is not None:
        ax.text(x_text, pos, "Gas pressure", color="limegreen", fontsize=28)

if Gravity and normalized_visible(F_grav_inward):
    pos = get_label_y(F_grav_inward, factor_up=1.2, cap=100.0)
    if pos is not None:
        ax.text(x_text, pos, "Gravity", color="black", fontsize=28)


# ------------------------------------------------------------
# Axis labels and limits
# ------------------------------------------------------------

print("L_bol", L_bol / Lsun, "Lsun")

ax.set_xlabel("Time (kyrs)", labelpad=20)
ax.set_ylabel("Feedback forces (in units of L$_{bol}$/c)", labelpad=20)

ax.set_ylim(ymin_plot, ymax_plot)

yl = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0]
ylabels = ['10$^{-4}$', '10$^{-3}$', '0.01', '0.1', '1', '10', '100']
ax.set_yticks(yl)
ax.set_yticklabels(ylabels)

# Set different lengths for major and minor ticks
ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction='in', length=20, width=2.5)
ax.tick_params(axis='both', which='minor', direction='in', length=10, width=2.5)

plt.tight_layout(pad=0.4)
plt.savefig("FeedbackComparison.pdf", bbox_inches="tight")
plt.savefig("FeedbackComparison.png", bbox_inches="tight")
plt.show()