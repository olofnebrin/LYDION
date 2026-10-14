import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
cmastro_src = repo_root / "src" / "cmastro" / "src"
sys.path.insert(0, str(cmastro_src))

try:
    import cmastro   # registers cma:* colormaps if available
except Exception:
    cmastro = None

# ---------------- User inputs ----------------
Filename_noLya = "RHDSim_StarCluster_NoLya_512x512_2026-04-19.h5"    # Run ignoring Lyα feedback
Filename_Lya   = "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5"      # Run with Lyα feedback

OutName   = "LYDION_PresentationAnimation_LyaVsNoLya_nH_eLya.mp4"
fps       = 20
frameskip = 1
save_mp4  = True

# IMPORTANT for mp4: final movie size is figsize * AnimationDPI.
# These defaults give 3240 x 1080, which should encode fine with ffmpeg/libx264.
FigureWidthInch  = 18.0
FigureHeightInch = 6.0
#AnimationDPI     = 140
AnimationDPI     = 100

# Optional time cut. If None, animation continues until the longest run ends.
MaxTime_kyr = None

# Maximum time spacing between master-timeline frames [kyr].
# Prevents one run's dense snapshot cadence from making that era look slow.
# Set to None to use the raw union of saved times (original behaviour).
MaxFrameDt_kyr = 0.17

# Background color of the figure (the gap strip between panels and outer margins).
# Use any matplotlib color string, e.g. 'black', '#1a1a1a', 'white'.
FigureBgColor = '#00172e'

# Optional static check snapshot
SaveSnapshot     = False
SnapshotTime_kyr = 40.0
SnapshotOutName  = "LYDION_AnimationStyle_CheckFrame_40kyr.png"
SnapshotDPI      = 600

# Plot limits. Use 0.8 to match the paper figure cropping.
PlotExtentFraction = 1.0

# Density range: use the same logic as the paper figure.
DensityLowerDexOffset = -2.0 - 0.5
DensityUpperDexOffset = +2.0 - 0.5

# Lyα energy density range: fixed for all frames so colors do not flicker.
eLya_vmin = 1e-14
eLya_vmax = 1e-6

fontsize_cbar_ticks  = 24
fontsize_panel_label = 34
fontsize_time        = 30
fontsize_diagnostics = 24
fontsize_end_text    = 28

cmap_nH   = 'cma:laguna'
cmap_eLya = 'magma'

# Diagnostics text options.
ShowDiagnosticsText       = True
ShowLyaForceText          = True
ShowLyaMultiplierText     = False
ShowLyaEscapeFractionText = True
ShowDirectForceText       = True
ShowIRForceText           = False

# Which Lyα force measure to show if diagnostics are enabled.
ForceVariant = "radial_net"   # "radial_net" | "antigravity"

diagnostics_x = 0.04
diagnostics_y = 0.82

# Useful constants
kB   = 1.38e-16
mH   = 1.67e-24
X    = 0.76
yr   = 3.1556926e7
kyr  = 1e3 * yr
pc   = 3.085677581e18
c    = 2.99792458e10
Lsun = 3.839e33

###############################################################
#######                                                 #######
#######            P L O T  C O S M E T I C S           #######
#######                                                 #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 36})
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
plt.rcParams['grid.linewidth'] = 2.5
plt.rcParams['grid.linestyle'] = '--'

plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


###############################################################
#######                                                 #######
#######                    H E L P E R S                #######
#######                                                 #######
###############################################################

def reorder_history_3d(arr, t, r_edges, Z_edges):
    """
    Reorder a 3D history array to shape (Nt, Nr, Nz).

    One axis must match len(t) (time).
    The remaining two must match (Nr, Nz), in either order.
    """
    if arr is None:
        return None
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    Nt = t.size
    Nr = len(r_edges) - 1
    Nz = len(Z_edges) - 1

    shape = arr.shape
    time_axes = [ax for ax, s in enumerate(shape) if s == Nt]
    if len(time_axes) != 1:
        raise RuntimeError(
            f"Could not uniquely identify time axis in array with shape {shape} and Nt = {Nt}"
        )
    time_axis = time_axes[0]

    arr_tfirst = np.moveaxis(arr, time_axis, 0)
    _, a1, a2 = arr_tfirst.shape

    if (a1, a2) == (Nr, Nz):
        return arr_tfirst
    elif (a1, a2) == (Nz, Nr):
        return arr_tfirst.transpose(0, 2, 1)
    else:
        raise RuntimeError(
            f"After moving time axis, spatial dims {(a1, a2)} do not match Nr = {Nr}, Nz = {Nz}"
        )


def read_dataset_if_exists(file, name, default=None):
    return file[name][:] if name in file else default


def read_attr_if_exists(group, name, default=None):
    if group is None:
        return default
    return group.attrs[name] if name in group.attrs else default


def read_first_existing_dataset(file, names, default=None):
    for name in names:
        if name in file:
            return file[name][:]
    return default


def safe_scalar_at_index(arr, idx):
    if arr is None:
        return None
    if idx < 0 or idx >= len(arr):
        return None
    val = arr[idx]
    try:
        val = float(np.asarray(val))
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return val


def fmt_log_tick(val):
    exp = int(np.round(np.log10(val)))
    if exp == -1:
        return r"$0.1$"
    elif exp == 0:
        return r"$1$"
    elif exp == 1:
        return r"$10$"
    else:
        return rf"$10^{{{exp}}}$"


def fmt_math_value(x):
    if x is None or (not np.isfinite(x)):
        return None
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 100.0 or ax < 0.01:
        exp = int(np.floor(np.log10(ax)))
        mant = x / (10.0**exp)
        return rf"{mant:.2f}\times10^{{{exp}}}"
    elif ax >= 10.0:
        return f"{x:.1f}"
    else:
        return f"{x:.2f}"


def _scalar_attr(x):
    if x is None:
        return None
    try:
        return float(np.asarray(x))
    except Exception:
        return None


def read_run(filename):
    """
    Read one run and return relevant arrays and diagnostics.
    """
    with h5py.File(filename, "r") as f:
        t       = f["Times"][:]
        r       = f["r"][:]
        Z       = f["Z"][:]
        rho_raw = f["GasDensity_History"][:]

        eLya_raw = f["e_Lya_History"][:] if "e_Lya_History" in f else None

        settings = f["Settings"] if "Settings" in f else None
        L_bol    = read_attr_if_exists(settings, "L_bol", None)

        M_F_Lya        = read_dataset_if_exists(f, "M_F_History")
        M_F_Lya_radial = read_dataset_if_exists(f, "M_F_radial_net_History")
        L_Lya          = read_dataset_if_exists(f, "L_Lya_History")
        f_esc_Lya      = read_first_existing_dataset(
            f,
            ["f_esc_Lya_History", "fesc_Lya_History", "f_esc_History_Lya"],
            default=None
        )

        M_F_dir        = read_dataset_if_exists(f, "M_F_dir_History")
        M_F_dir_radial = read_dataset_if_exists(f, "M_F_dir_radial_net_History")
        M_F_IR         = read_dataset_if_exists(f, "M_F_IR_History")
        M_F_IR_radial  = read_dataset_if_exists(f, "M_F_IR_radial_net_History")
        L_IR           = read_dataset_if_exists(f, "L_IR_History")

    rho_hist  = reorder_history_3d(rho_raw,  t, r, Z)
    eLya_hist = reorder_history_3d(eLya_raw, t, r, Z) if eLya_raw is not None else None
    nH_hist   = X * rho_hist / mH

    saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]

    return dict(
        t=t,
        r=r,
        Z=Z,
        rho_hist=rho_hist,
        nH_hist=nH_hist,
        eLya_hist=eLya_hist,
        saved_idx=saved_idx,
        L_bol=L_bol,
        M_F_Lya=M_F_Lya,
        M_F_Lya_radial=M_F_Lya_radial,
        L_Lya=L_Lya,
        f_esc_Lya=f_esc_Lya,
        M_F_dir=M_F_dir,
        M_F_dir_radial=M_F_dir_radial,
        M_F_IR=M_F_IR,
        M_F_IR_radial=M_F_IR_radial,
        L_IR=L_IR
    )


def _pick_Lya_force(run, k):
    """
    Return Lyα force in units of L_bol/c, honouring ForceVariant.
    """
    L_Lya_val = safe_scalar_at_index(run["L_Lya"], k)
    L_bol_val = _scalar_attr(run["L_bol"])

    if (L_Lya_val is None) or (L_bol_val is None):
        return None

    M_F_val        = safe_scalar_at_index(run["M_F_Lya"], k)
    M_F_radial_val = safe_scalar_at_index(run["M_F_Lya_radial"], k)

    if ForceVariant == "radial_net":
        if M_F_radial_val is not None:
            return M_F_radial_val * L_Lya_val / L_bol_val
        elif M_F_val is not None:
            return M_F_val * L_Lya_val / L_bol_val
    else:
        if M_F_val is not None:
            return M_F_val * L_Lya_val / L_bol_val

    return None


def make_radiation_diagnostics_lines(run, k):
    lines = []

    L_bol_val = _scalar_attr(run["L_bol"])
    L_IR_val  = safe_scalar_at_index(run["L_IR"], k)

    if ShowLyaForceText:
        F_Lya = _pick_Lya_force(run, k)
        F_str = fmt_math_value(F_Lya)
        if F_str is not None:
            lines.append(rf"$F_{{Ly\alpha}} = {F_str}\,L_{{bol}}/c$")

    if ShowDirectForceText:
        M_F_dir_radial_val = safe_scalar_at_index(run["M_F_dir_radial"], k)
        M_F_dir_val        = safe_scalar_at_index(run["M_F_dir"], k)
        val = M_F_dir_radial_val if M_F_dir_radial_val is not None else M_F_dir_val
        F_str = fmt_math_value(val)
        if F_str is not None:
            lines.append(rf"$F_{{dir}} = {F_str}\,L_{{bol}}/c$")

    if ShowIRForceText and (L_bol_val is not None) and (L_IR_val is not None):
        M_F_IR_radial_val = safe_scalar_at_index(run["M_F_IR_radial"], k)
        M_F_IR_val        = safe_scalar_at_index(run["M_F_IR"], k)
        val = M_F_IR_radial_val if M_F_IR_radial_val is not None else M_F_IR_val
        if val is not None:
            F_IR = val * L_IR_val / L_bol_val
            F_str = fmt_math_value(F_IR)
            if F_str is not None:
                lines.append(rf"$F_{{IR}} = {F_str}\,L_{{bol}}/c$")

    if ShowLyaMultiplierText:
        if ForceVariant == "radial_net":
            M_F_val = safe_scalar_at_index(run["M_F_Lya_radial"], k)
            if M_F_val is None:
                M_F_val = safe_scalar_at_index(run["M_F_Lya"], k)
        else:
            M_F_val = safe_scalar_at_index(run["M_F_Lya"], k)

        M_str = fmt_math_value(M_F_val)
        if M_str is not None:
            lines.append(rf"$M_F = {M_str}$")

    return lines


def make_eLya_diagnostics_lines(run, k):
    lines = []
    if ShowLyaEscapeFractionText:
        f_esc_val = safe_scalar_at_index(run["f_esc_Lya"], k)
        fesc_str = fmt_math_value(f_esc_val)
        if fesc_str is not None:
            lines.append(rf"$f_{{esc,Ly\alpha}} = {fesc_str}$")
    return lines


def make_inset_cbar(fig, ax, im, rect, ticks, ticklabels, dark=False):
    """
    Shared helper for inset horizontal colorbars.
    """
    fg = 'black' if dark else 'white'
    bg = 'white' if dark else 'black'

    cax  = ax.inset_axes(rect)
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.outline.set_edgecolor(fg)
    cbar.outline.set_linewidth(1.2)
    cbar.ax.tick_params(axis="x", which='both', colors=fg, labelsize=fontsize_cbar_ticks,
                        length=0, width=0, pad=10, bottom=False, top=False)
    cbar.ax.minorticks_off()
    cbar.ax.set_facecolor(bg)
    return cbar


def _clip_nH(arr):
    return np.clip(arr.T, vmin_nH, vmax_nH)


def _clip_eLya(arr):
    return np.clip(arr.T, eps, None)


def build_master_timeline(runA, runB, MaxTime_kyr=None, MaxFrameDt_kyr=None):
    """
    Build a master animation timeline from the union of saved times of both runs.
    If MaxFrameDt_kyr is given, resample so no two consecutive frames are farther
    apart than that, eliminating slow-motion regions caused by dense snapshot cadence.
    """
    tA = runA["t"][runA["saved_idx"]]
    tB = runB["t"][runB["saved_idx"]]

    timeline = np.union1d(tA, tB)

    if MaxTime_kyr is not None:
        timeline = timeline[timeline <= MaxTime_kyr * kyr]

    if timeline.size == 0:
        raise RuntimeError("No frames left after applying MaxTime_kyr.")

    if MaxFrameDt_kyr is not None:
        dt_max = MaxFrameDt_kyr * kyr
        resampled = [timeline[0]]
        for t_next in timeline[1:]:
            if t_next - resampled[-1] >= dt_max:
                resampled.append(t_next)
        # Always include the final time.
        if resampled[-1] != timeline[-1]:
            resampled.append(timeline[-1])
        timeline = np.array(resampled)

    return timeline


def latest_snapshot_at_or_before(run, target_t):
    """
    Return:
      k        = snapshot index to use
      ended    = True if target_t is beyond the last saved snapshot
      exact_t  = actual snapshot time used
    """
    saved_idx = run["saved_idx"]
    ts = run["t"][saved_idx]

    pos = np.searchsorted(ts, target_t, side="right") - 1

    if pos < 0:
        pos = 0

    k = saved_idx[pos]
    ended = (target_t > ts[-1])
    exact_t = run["t"][k]
    return k, ended, exact_t


###############################################################
#######                                                 #######
#######                 R E A D  D A T A                #######
#######                                                 #######
###############################################################

run_noLya = read_run(Filename_noLya)
run_Lya   = read_run(Filename_Lya)

if run_Lya["eLya_hist"] is None:
    raise RuntimeError("e_Lya_History not found in Lyα file.")

# Check that grids match.
if (run_noLya["r"].shape != run_Lya["r"].shape) or (run_noLya["Z"].shape != run_Lya["Z"].shape):
    raise RuntimeError(
        f"Grid mismatch: r shapes {run_noLya['r'].shape} vs {run_Lya['r'].shape}, "
        f"Z shapes {run_noLya['Z'].shape} vs {run_Lya['Z'].shape}"
    )

if not (np.allclose(run_noLya["r"], run_Lya["r"]) and np.allclose(run_noLya["Z"], run_Lya["Z"])):
    raise RuntimeError("Grid mismatch: r/Z edge arrays differ between the two files.")

r = run_noLya["r"]
Z = run_noLya["Z"]
R_edges_pc, Z_edges_pc = np.meshgrid(r / pc, Z / pc, indexing='ij')

master_times = build_master_timeline(run_noLya, run_Lya, MaxTime_kyr=MaxTime_kyr,
                                      MaxFrameDt_kyr=MaxFrameDt_kyr)
master_times = master_times[::frameskip]
frames = np.arange(master_times.size)

print("No-Lyα:  t range:", run_noLya["t"][run_noLya["saved_idx"][0]] / kyr, "→",
      run_noLya["t"][run_noLya["saved_idx"][-1]] / kyr, "kyr")
print("With-Lyα: t range:", run_Lya["t"][run_Lya["saved_idx"][0]] / kyr, "→",
      run_Lya["t"][run_Lya["saved_idx"][-1]] / kyr, "kyr")
print("Animation master time range:", master_times[0] / kyr, "→", master_times[-1] / kyr, "kyr")
print("Animating frames:", len(frames))

###############################################################
#######                                                 #######
#######             D I S P L A Y  S C A L E S          #######
#######                                                 #######
###############################################################

nH0_noLya = run_noLya["nH_hist"][0]
nH0_Lya   = run_Lya["nH_hist"][0]

nH0_noLya_pos = nH0_noLya[nH0_noLya > 0.0]
nH0_Lya_pos   = nH0_Lya[nH0_Lya > 0.0]

if (nH0_noLya_pos.size == 0) or (nH0_Lya_pos.size == 0):
    raise ValueError("Initial nH field has no positive values in one of the runs.")

nH0_char = max(np.max(nH0_noLya_pos), np.max(nH0_Lya_pos))

vmin_nH = 10.0**(np.log10(nH0_char) + DensityLowerDexOffset)
vmax_nH = 10.0**(np.log10(nH0_char) + DensityUpperDexOffset)
norm_nH = LogNorm(vmin=vmin_nH, vmax=vmax_nH)

logmin_nH = int(np.ceil(np.log10(vmin_nH)))
logmax_nH = int(np.floor(np.log10(vmax_nH)))
ticks_nH  = 10.0**np.arange(logmin_nH, logmax_nH + 1)
ticklabels_nH = [fmt_log_tick(v) for v in ticks_nH]

norm_eLya       = LogNorm(vmin=eLya_vmin, vmax=eLya_vmax)
ticks_eLya      = [eLya_vmin, np.sqrt(eLya_vmin * eLya_vmax), eLya_vmax]
ticklabels_eLya = [fmt_log_tick(v) for v in ticks_eLya]

eps = 1e-30

###############################################################
#######                                                 #######
#######              S E T U P  F I G U R E             #######
#######                                                 #######
###############################################################

# One row, three columns.
wgap_big   = 0.025
wgap_small = 0.000
wpanel = (1.0 - wgap_big - wgap_small) / 3.0
hpanel = 1.0

col0_left = 0.0
col1_left = wpanel + wgap_big
col2_left = col1_left + wpanel + wgap_small

def make_ax(fig, col_left, sharex=None, sharey=None):
    return fig.add_axes([col_left, 0.0, wpanel, hpanel], sharex=sharex, sharey=sharey)

fig = plt.figure(figsize=(FigureWidthInch, FigureHeightInch))
fig.patch.set_facecolor(FigureBgColor)

ax1 = make_ax(fig, col0_left)
ax2 = make_ax(fig, col1_left, sharex=ax1, sharey=ax1)
ax3 = make_ax(fig, col2_left, sharex=ax1, sharey=ax1)

# Initial frame.
t0 = master_times[0]
k1, ended1, t1 = latest_snapshot_at_or_before(run_noLya, t0)
k2, ended2, t2 = latest_snapshot_at_or_before(run_Lya,   t0)

im1 = ax1.pcolormesh(R_edges_pc, Z_edges_pc, _clip_nH(run_noLya["nH_hist"][k1]),
                     norm=norm_nH, shading='auto', cmap=cmap_nH, rasterized=True)

im2 = ax2.pcolormesh(R_edges_pc, Z_edges_pc, _clip_nH(run_Lya["nH_hist"][k2]),
                     norm=norm_nH, shading='auto', cmap=cmap_nH, rasterized=True)

im3 = ax3.pcolormesh(R_edges_pc, Z_edges_pc, _clip_eLya(run_Lya["eLya_hist"][k2]),
                     norm=norm_eLya, shading='auto', cmap=cmap_eLya, rasterized=True)

for ax in (ax1, ax2, ax3):
    ax.set_xlim(r[0] / pc, PlotExtentFraction * r[-1] / pc)
    ax.set_ylim(Z[0] / pc, PlotExtentFraction * Z[-1] / pc)
    ax.set_aspect('equal', adjustable='box')

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(2.0)

title_style = dict(
    ha='center',
    va='bottom',
    color='black',
    fontsize=plt.rcParams['font.size'],
    path_effects=[pe.withStroke(linewidth=3, foreground='white')]
)

time_style = dict(
    ha='left',
    va='top',
    color='white',
    path_effects=[pe.withStroke(linewidth=3, foreground='black')]
)

panel_label_style_white = dict(
    ha='right',
    va='bottom',
    color='white',
    path_effects=[pe.withStroke(linewidth=3.5, foreground='black')]
)

diagnostics_style = dict(
    ha='left',
    va='top',
    color='white',
    path_effects=[pe.withStroke(linewidth=3, foreground='black')]
)

end_style = dict(
    ha='center',
    va='center',
    color='white',
    path_effects=[pe.withStroke(linewidth=4, foreground='black')]
)

# Column titles.
ax1.text(0.5, 1.02, "Ignoring Lyα Radiation Pressure", transform=ax1.transAxes, **title_style)
ax2.text(0.5, 1.02, "With Lyα Radiation Pressure",    transform=ax2.transAxes, **title_style)
ax3.text(0.5, 1.02, "With Lyα Radiation Pressure",    transform=ax3.transAxes, **title_style)

# Time labels.
txt_time1 = ax1.text(0.04, 0.96, "", transform=ax1.transAxes, fontsize=fontsize_time, **time_style)
txt_time2 = ax2.text(0.04, 0.96, "", transform=ax2.transAxes, fontsize=fontsize_time, **time_style)
txt_time3 = ax3.text(0.04, 0.96, "", transform=ax3.transAxes, fontsize=fontsize_time, **time_style)

# Panel labels.
ax1.text(0.94, 0.08, r"$n_H$ (cm$^{-3}$)", transform=ax1.transAxes,
         fontsize=fontsize_panel_label, **panel_label_style_white)
ax2.text(0.94, 0.08, r"$n_H$ (cm$^{-3}$)", transform=ax2.transAxes,
         fontsize=fontsize_panel_label, **panel_label_style_white)
ax3.text(0.94, 0.08, r"$e_{Ly\alpha}$ (erg cm$^{-3}$)", transform=ax3.transAxes,
         fontsize=fontsize_panel_label, **panel_label_style_white)

# End-of-run overlays.
txt_end1 = ax1.text(0.50, 0.50, "", transform=ax1.transAxes, fontsize=fontsize_end_text, **end_style)
txt_end2 = ax2.text(0.50, 0.50, "", transform=ax2.transAxes, fontsize=fontsize_end_text, **end_style)
txt_end3 = ax3.text(0.50, 0.50, "", transform=ax3.transAxes, fontsize=fontsize_end_text, **end_style)

# Diagnostics.
txt_diag1 = ax1.text(diagnostics_x, diagnostics_y, "", transform=ax1.transAxes,
                     fontsize=fontsize_diagnostics, **diagnostics_style)
txt_diag2 = ax2.text(diagnostics_x, diagnostics_y, "", transform=ax2.transAxes,
                     fontsize=fontsize_diagnostics, **diagnostics_style)
txt_diag3 = ax3.text(diagnostics_x, diagnostics_y, "", transform=ax3.transAxes,
                     fontsize=fontsize_diagnostics, **diagnostics_style)

if not ShowDiagnosticsText:
    txt_diag1.set_visible(False)
    txt_diag2.set_visible(False)
    txt_diag3.set_visible(False)

# Inset colorbars. One for density, one for e_Lyα.
make_inset_cbar(fig, ax1, im1,
                rect=[0.56, 0.91, 0.38, 0.045],
                ticks=ticks_nH, ticklabels=ticklabels_nH)

make_inset_cbar(fig, ax3, im3,
                rect=[0.54, 0.91, 0.40, 0.045],
                ticks=ticks_eLya, ticklabels=ticklabels_eLya)


###############################################################
#######                                                 #######
#######               A N I M A T I O N                 #######
#######                                                 #######
###############################################################

def set_frame(q):
    target_t = master_times[q]

    k_noLya, ended_noLya, actual_t_noLya = latest_snapshot_at_or_before(run_noLya, target_t)
    k_Lya,   ended_Lya,   actual_t_Lya   = latest_snapshot_at_or_before(run_Lya,   target_t)

    # Update image arrays.
    im1.set_array(_clip_nH(run_noLya["nH_hist"][k_noLya]).ravel())
    im2.set_array(_clip_nH(run_Lya["nH_hist"][k_Lya]).ravel())
    im3.set_array(_clip_eLya(run_Lya["eLya_hist"][k_Lya]).ravel())

    # Time texts: show actual frame time used; hide once a run has ended.
    if ended_noLya:
        txt_time1.set_visible(False)
        txt_end1.set_text("Simulation ended,\nshowing final snapshot")
    else:
        txt_time1.set_visible(True)
        txt_time1.set_text(f"t = {actual_t_noLya / kyr:.2f} kyr")
        txt_end1.set_text("")

    if ended_Lya:
        txt_time2.set_visible(False)
        txt_time3.set_visible(False)
        txt_end2.set_text("Simulation ended,\nshowing final snapshot")
        txt_end3.set_text("Simulation ended,\nshowing final snapshot")
    else:
        txt_time2.set_visible(True)
        txt_time3.set_visible(True)
        txt_time2.set_text(f"t = {actual_t_Lya / kyr:.2f} kyr")
        txt_time3.set_text(f"t = {actual_t_Lya / kyr:.2f} kyr")
        txt_end2.set_text("")
        txt_end3.set_text("")

    if ShowDiagnosticsText:
        txt_diag1.set_text("\n".join(make_radiation_diagnostics_lines(run_noLya, k_noLya)))
        txt_diag2.set_text("\n".join(make_radiation_diagnostics_lines(run_Lya,   k_Lya)))
        txt_diag3.set_text("\n".join(make_eLya_diagnostics_lines(run_Lya, k_Lya)))

    artists = [im1, im2, im3,
               txt_time1, txt_time2, txt_time3,
               txt_end1, txt_end2, txt_end3]

    if ShowDiagnosticsText:
        artists += [txt_diag1, txt_diag2, txt_diag3]

    return tuple(artists)


def init():
    return set_frame(0)


def update(q):
    return set_frame(q)


def find_nearest_master_frame(target_time_kyr):
    target_time = target_time_kyr * kyr
    idx = np.argmin(np.abs(master_times - target_time))
    return int(idx)


def save_snapshot_at_time(target_time_kyr, outname, dpi=600):
    q = find_nearest_master_frame(target_time_kyr)
    set_frame(q)
    fig.savefig(outname, dpi=dpi, bbox_inches='tight')
    print(f"Saved snapshot to {outname}")
    print(f"  Requested time: {target_time_kyr:.3f} kyr")
    print(f"  Master timeline time: {master_times[q]/kyr:.3f} kyr")


ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

if SaveSnapshot:
    save_snapshot_at_time(SnapshotTime_kyr, SnapshotOutName, dpi=SnapshotDPI)


if save_mp4:
    ani.save(
        OutName,
        writer="ffmpeg",
        fps=fps,
        dpi=AnimationDPI,
        extra_args=[
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            '-preset', 'medium',
            '-movflags', '+faststart'
        ]
    )
else:
    plt.show()
