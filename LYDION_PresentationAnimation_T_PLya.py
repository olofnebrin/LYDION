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
#Filename_noLya = "RHDSim_StarCluster_NoLya_512x512_2026-04-19.h5"    # Run ignoring Lyα feedback
#Filename_Lya   = "RHDSim_StarCluster_Lya_512x512_2026-04-19.h5"      # Run with Lyα feedback

Filename_noLya = "RHDSim_NoLya_NoDust_u4_256x256_2026-05-06.h5"    # Run ignoring Lyα feedback
Filename_Lya   = "RHDSim_Lya_NoDust_u4_256x256_2026-05-06.h5"      # Run with Lyα feedback

OutName   = "LYDION_PresentationAnimation_LyaVsNoLya_T_PLya.mp4"
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
# Set to None to use the raw union of saved times.
MaxFrameDt_kyr = 0.17

# Background color of the figure (the gap strip between panels and outer margins).
# Use any matplotlib color string, e.g. 'black', '#1a1a1a', 'white'.
FigureBgColor = '#00172e'

# Optional static check snapshot
SaveSnapshot     = False
SnapshotTime_kyr = 40.0
SnapshotOutName  = "LYDION_AnimationStyle_CheckFrame_T_PLya_40kyr.png"
SnapshotDPI      = 600

# Plot limits. Use 0.8 to match the paper figure cropping.
PlotExtentFraction = 1.0

# Temperature colour scale, shared by both runs and all frames.
T_vmin, T_vmax = 1.0, 1e5
cmap_T = 'coolwarm'

# P_Lyα/P colour scale, Lyα run only.
Ratio_vmin, Ratio_vmax = 0.01, 100.0
cmap_ratio = 'cma:unph'

fontsize_cbar_ticks  = 24
fontsize_panel_label = 34
fontsize_time        = 30
fontsize_end_text    = 28

# If False, panels simply freeze at their last available snapshot after a run ends.
# This avoids extra text competing with colorbars in presentation animations.
ShowEndedText = True

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


def fmt_log_tick(val):
    exp = int(np.round(np.log10(val)))
    if exp == -2:
        return r"$0.01$"
    elif exp == -1:
        return r"$0.1$"
    elif exp == 0:
        return r"$1$"
    elif exp == 1:
        return r"$10$"
    elif exp == 2:
        return r"$100$"
    else:
        return rf"$10^{{{exp}}}$"


def read_run(filename, load_T=True, load_P=False, load_eLya=False):
    """
    Read one run and return relevant arrays.
    T is needed for both runs; P and e_Lya are needed only for the Lyα run.
    """
    with h5py.File(filename, "r") as f:
        t        = f["Times"][:]
        r        = f["r"][:]
        Z        = f["Z"][:]
        T_raw    = f["T_History"][:] if (load_T and "T_History" in f) else None
        P_raw    = f["GasPressure_History"][:] if (load_P and "GasPressure_History" in f) else None
        eLya_raw = f["e_Lya_History"][:] if (load_eLya and "e_Lya_History" in f) else None

    T_hist    = reorder_history_3d(T_raw,    t, r, Z) if T_raw    is not None else None
    P_hist    = reorder_history_3d(P_raw,    t, r, Z) if P_raw    is not None else None
    eLya_hist = reorder_history_3d(eLya_raw, t, r, Z) if eLya_raw is not None else None

    saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]

    return dict(
        t=t,
        r=r,
        Z=Z,
        T_hist=T_hist,
        P_hist=P_hist,
        eLya_hist=eLya_hist,
        saved_idx=saved_idx,
    )


def make_inset_cbar(fig, ax, im, rect, ticks, ticklabels, dark=False):
    """
    Shared helper for inset horizontal colorbars.

    dark=False: white tick labels on black cbar background.
    dark=True:  black tick labels on white cbar background.
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


def _clip_T(arr):
    return np.clip(arr.T, T_vmin, T_vmax)


def _compute_ratio(run, k):
    """Compute P_Lyα/P = (e_Lyα/3) / P_gas at snapshot index k."""
    Pgas = run["P_hist"][k].T
    eLya = run["eLya_hist"][k].T
    return np.clip((eLya / 3.0) / np.maximum(Pgas, eps), Ratio_vmin, Ratio_vmax)


def build_master_timeline(runA, runB, MaxTime_kyr=None, MaxFrameDt_kyr=None):
    """
    Build a master animation timeline from the union of saved times of both runs.
    If MaxFrameDt_kyr is given, resample so no two consecutive frames are closer
    than that, eliminating slow-motion regions caused by dense snapshot cadence.
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

run_noLya = read_run(Filename_noLya, load_T=True)
run_Lya   = read_run(Filename_Lya,   load_T=True, load_P=True, load_eLya=True)

# Guards.
if run_noLya["T_hist"] is None:
    raise RuntimeError("T_History not found in no-Lyα file.")

if run_Lya["T_hist"] is None:
    raise RuntimeError("T_History not found in Lyα file.")

if run_Lya["P_hist"] is None:
    raise RuntimeError("GasPressure_History not found in Lyα file.")

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

norm_T        = LogNorm(vmin=T_vmin, vmax=T_vmax, clip=True)
ticks_T       = [1.0, 10.0, 1e2, 1e3, 1e4, 1e5]
ticklabels_T  = [fmt_log_tick(v) for v in ticks_T]

norm_ratio       = LogNorm(vmin=Ratio_vmin, vmax=Ratio_vmax, clip=True)
ticks_ratio      = [0.01, 0.1, 1.0, 10.0, 100.0]
ticklabels_ratio = [fmt_log_tick(v) for v in ticks_ratio]

eps = 1e-30

###############################################################
#######                                                 #######
#######              S E T U P  F I G U R E             #######
#######                                                 #######
###############################################################

# One row, three columns:
#   T no-Lyα | T Lyα | P_Lyα/P Lyα
wgap_big   = 0.025
wgap_small = 0.000
wpanel = (1.0 - wgap_big - wgap_small) / 3.0
hpanel = 1.0

col0_left = 0.0
col1_left = wpanel + wgap_big
col2_left = col1_left + wpanel + wgap_small
fig_top = hpanel


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

im1 = ax1.pcolormesh(R_edges_pc, Z_edges_pc, _clip_T(run_noLya["T_hist"][k1]),
                     norm=norm_T, shading='auto', cmap=cmap_T, rasterized=True)

im2 = ax2.pcolormesh(R_edges_pc, Z_edges_pc, _clip_T(run_Lya["T_hist"][k2]),
                     norm=norm_T, shading='auto', cmap=cmap_T, rasterized=True)

im3 = ax3.pcolormesh(R_edges_pc, Z_edges_pc, _compute_ratio(run_Lya, k2),
                     norm=norm_ratio, shading='auto', cmap=cmap_ratio, rasterized=True)

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

time_style_dark = dict(
    ha='left',
    va='top',
    color='black',
    path_effects=[pe.withStroke(linewidth=3, foreground='white')]
)

time_style_white = dict(
    ha='left',
    va='top',
    color='white',
    path_effects=[pe.withStroke(linewidth=3, foreground='black')]
)

panel_label_style_black = dict(
    ha='right',
    va='bottom',
    color='black',
    path_effects=[pe.withStroke(linewidth=3.5, foreground='white')]
)

panel_label_style_white = dict(
    ha='right',
    va='bottom',
    color='white',
    path_effects=[pe.withStroke(linewidth=3.5, foreground='black')]
)

end_style = dict(
    ha='center',
    va='center',
    color='white',
    path_effects=[pe.withStroke(linewidth=4, foreground='black')]
)

# Column titles.
ax1.text(0.5, 1.02, "Ignoring Lyα Radiation Pressure", transform=ax1.transAxes, **title_style)

# Lyα title: centred over cols 1–2 in figure coordinates, as in the static T/P_Lyα figure.
lya_title_x = 0.5 * (col1_left + col2_left + wpanel)
lya_title_y = fig_top
fig.text(lya_title_x, lya_title_y + 0.02, "With Lyα Radiation Pressure", **title_style)

# Time labels.
txt_time1 = ax1.text(0.04, 0.96, "", transform=ax1.transAxes, fontsize=fontsize_time, **time_style_dark)
txt_time2 = ax2.text(0.04, 0.96, "", transform=ax2.transAxes, fontsize=fontsize_time, **time_style_dark)
txt_time3 = ax3.text(0.04, 0.96, "", transform=ax3.transAxes, fontsize=fontsize_time, **time_style_white)

# Panel labels.
ax1.text(0.94, 0.08, r"$T$ (K)", transform=ax1.transAxes,
         fontsize=fontsize_panel_label, **panel_label_style_black)
ax2.text(0.94, 0.08, r"$T$ (K)", transform=ax2.transAxes,
         fontsize=fontsize_panel_label, **panel_label_style_black)
ax3.text(0.94, 0.08, r"$e_{Ly\alpha}/3P \approx P_{Ly\alpha}/P$", transform=ax3.transAxes,
         fontsize=fontsize_panel_label, **panel_label_style_white)

# End-of-run overlays.
txt_end1 = ax1.text(0.50, 0.50, "", transform=ax1.transAxes, fontsize=fontsize_end_text, **end_style)
txt_end2 = ax2.text(0.50, 0.50, "", transform=ax2.transAxes, fontsize=fontsize_end_text, **end_style)
txt_end3 = ax3.text(0.50, 0.50, "", transform=ax3.transAxes, fontsize=fontsize_end_text, **end_style)

# Inset colorbars. One for T, one for P_Lyα/P.
make_inset_cbar(fig, ax1, im1,
                rect=[0.48, 0.91, 0.46, 0.045],
                ticks=ticks_T, ticklabels=ticklabels_T,
                dark=True)

make_inset_cbar(fig, ax3, im3,
                rect=[0.48, 0.91, 0.46, 0.045],
                ticks=ticks_ratio, ticklabels=ticklabels_ratio,
                dark=False)


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
    im1.set_array(_clip_T(run_noLya["T_hist"][k_noLya]).ravel())
    im2.set_array(_clip_T(run_Lya["T_hist"][k_Lya]).ravel())
    im3.set_array(_compute_ratio(run_Lya, k_Lya).ravel())

    # Time texts: show actual snapshot time used; optionally mark panels after a run ends.
    if ended_noLya:
        txt_time1.set_visible(False)
        txt_end1.set_text("Simulation ended,\nshowing final snapshot" if ShowEndedText else "")
    else:
        txt_time1.set_visible(True)
        txt_time1.set_text(f"t = {actual_t_noLya / kyr:.2f} kyr")
        txt_end1.set_text("")

    if ended_Lya:
        txt_time2.set_visible(False)
        txt_time3.set_visible(False)
        txt_end2.set_text("Simulation ended,\nshowing final snapshot" if ShowEndedText else "")
        txt_end3.set_text("Simulation ended,\nshowing final snapshot" if ShowEndedText else "")
    else:
        txt_time2.set_visible(True)
        txt_time3.set_visible(True)
        txt_time2.set_text(f"t = {actual_t_Lya / kyr:.2f} kyr")
        txt_time3.set_text(f"t = {actual_t_Lya / kyr:.2f} kyr")
        txt_end2.set_text("")
        txt_end3.set_text("")

    artists = [im1, im2, im3,
               txt_time1, txt_time2, txt_time3,
               txt_end1, txt_end2, txt_end3]

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
