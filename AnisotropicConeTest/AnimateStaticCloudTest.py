import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import re
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

########## USER INPUT #########################
FileName  = "CloudTest_theta0.7853981633974483_rs0.5_tau1.0e8_T10000.0_umax0.0_tauabs0.0_pd0.0_256x256.h5"
frameskip = 1
save_mp4  = True
outname   = "StaticCloud_eLya_withLyaSpec.mp4"
fps       = 20

# --- Spectrum selection mode ---
spectrum_selection = "two_boundaries"   # "manual" or "two_boundaries"
# "manual": use edge_indices below
# "two_boundaries": auto-pick two edges:
#   (1) r-boundary near Z≈0
#   (2) Z-boundary near r≈0

# Manual: Which Lyα edge spectra to plot (indices along J_Lya_EdgeSpectra axis 0)
edge_indices = [0]              # e.g. [0, 5, 10]

# What to animate on the left panel: either the 2D map or the spherical average profile.
left_panel_mode = "map"        # "map" or "axis"

# Whether to use log or linear color scale for e_Lya map/profile:
eLya_scale      = "log"         # "log" or "linear"

# --- Optional: plot analytical solution in left panel (only used if left_panel_mode == "axis") ---
plot_analytical_left = True
analytical_label     = "Analytical"
analytical_linestyle = "--"
analytical_linewidth = 3

# --- Optional: plot analytical spectrum in spectrum panel (ONLY if spectrum_xaxis == "x") ---
plot_analytical_spectrum = False
analytical_spec_label    = r"Analytical"
analytical_spec_color    = "gray"
analytical_spec_ls       = "--"
analytical_spec_lw       = 5

# Spectrum x-axis:
spectrum_xaxis  = "x"           # "dnu" or "x"
#   "dnu": plot vs δν in Hz (nuC from file)
#   "x"  : plot vs x = (ν-ν_Lya)/Δν_D, with Δν_D from T parsed from filename

# ν-window settings for the spectrum panel
use_nu_window       = True
nu_Lya              = 2.466e15             # Hz
nu_window_fraction  = 3*2.5 * 0.007                # total width = frac * nu_Lya

# Auto spectrum limits
spec_ylim_factor   = 2.0       # ymax = factor * Jmax_final
spec_thresh_factor = 1e-3      # x-range where J > thresh*Jmax_final (final time)
spec_xlim_pad_frac = 0.03      # small padding on x-limits

# --- Legend cosmetics ---
left_legend_fontsize = 30
spec_legend_fontsize = 30

left_legend_loc      = "upper left"
left_legend_anchor   = (0.2, 0.95)    # axes fraction

spec_legend_loc      = "center"
spec_legend_anchor   = (0.5, 0.75)   # axes fraction

legend_handlelength  = 2.0
legend_labelspacing  = 0.35
legend_borderpad     = 0.25
legend_handletextpad = 0.6

###############################################################
#######               C O N S T A N T S                 #######
###############################################################

# Physical constants for Doppler width (cgs)
c_cgs  = 2.99792458e10
kB_cgs = 1.380649e-16
mH_cgs = 1.6735575e-24

###############################################################
#######                H E L P E R S                    #######
###############################################################

def integrate_trap(y, x):
    """Robust trapezoidal integration across numpy versions."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    dx = np.diff(x)
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * dx))

def normalize_spectrum(y, x):
    """Normalize so ∫ y dx = 1 (returns a new array)."""
    y = np.asarray(y, dtype=float)
    area = integrate_trap(y, x)
    if not np.isfinite(area) or area <= 0.0:
        return y
    return y / area

def reorder_history_3d(arr, t, r_edges, Z_edges):
    """Reorder a 3D history array to shape (Nt, Nr, Nz)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    Nt = t.size
    Nr = len(r_edges) - 1
    Nz = len(Z_edges) - 1

    shape = arr.shape
    time_axes = [ax for ax, s in enumerate(shape) if s == Nt]
    if len(time_axes) != 1:
        raise RuntimeError(
            f"Could not uniquely identify time axis in array with shape {shape} and Nt={Nt}"
        )
    time_axis = time_axes[0]

    arr_tfirst = np.moveaxis(arr, time_axis, 0)  # (Nt, A, B)
    _, a1, a2 = arr_tfirst.shape

    if (a1, a2) == (Nr, Nz):
        return arr_tfirst
    elif (a1, a2) == (Nz, Nr):
        return arr_tfirst.transpose(0, 2, 1)
    else:
        raise RuntimeError(
            f"After moving time axis, spatial dims {(a1, a2)} do not match Nr={Nr}, Nz={Nz}"
        )

def reorder_lya_edge_spectra(J_edge_raw, t, nuC):
    """Reorder J_Lya_EdgeSpectra to shape (Nt, N_edge, N_nu)."""
    if J_edge_raw.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {J_edge_raw.shape}")

    Nt  = t.size
    Nν  = nuC.size
    s0, s1, s2 = J_edge_raw.shape

    axes = [s0, s1, s2]
    time_axes = [ax for ax, s in enumerate(axes) if s == Nt]
    nu_axes   = [ax for ax, s in enumerate(axes) if s == Nν]

    if len(time_axes) != 1:
        raise RuntimeError(
            f"Could not uniquely identify time axis in J_edge_raw: shape={J_edge_raw.shape}, Nt={Nt}"
        )
    if len(nu_axes) != 1:
        raise RuntimeError(
            f"Could not uniquely identify frequency axis in J_edge_raw: shape={J_edge_raw.shape}, Nν={Nν}"
        )

    t_ax  = time_axes[0]
    nu_ax = nu_axes[0]
    edge_ax = 3 - t_ax - nu_ax

    perm = [t_ax, edge_ax, nu_ax]
    J_reordered = np.transpose(J_edge_raw, axes=perm)

    NtL, NedgeL, NνL = J_reordered.shape
    if NtL != Nt:
        raise RuntimeError(f"Time length mismatch: {NtL} vs Nt={Nt}")
    if NνL != Nν:
        raise RuntimeError(f"Freq length mismatch: {NνL} vs Nν={Nν}")

    return J_reordered

def reorder_spherical_history(arr, t):
    """Reorder e_Lya_SphericalHistory to shape (Nt, Nbins)."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D spherical history, got shape {arr.shape}")

    Nt = t.size
    s0, s1 = arr.shape

    if s0 == Nt and s1 != Nt:
        return arr
    if s1 == Nt and s0 != Nt:
        return arr.T

    raise RuntimeError(
        f"Could not identify time axis in spherical history: shape={arr.shape}, Nt={Nt}"
    )

def parse_temperature_from_filename(fname):
    """Parse temperature from substring like '_T10000.0_' in the filename."""
    m = re.search(r"_T([0-9]+(?:\.[0-9]*)?)_", fname)
    if m is None:
        return None
    return float(m.group(1))

def pick_two_boundary_edges(edge_ridx, edge_zidx, rCoM, ZCoM):
    """
    Auto-pick two edge indices:
      1) r-boundary near Z≈0: choose iz closest to 0, then among edges at that iz choose max r
      2) Z-boundary near r≈0: choose ir closest to 0, then among edges at that ir choose max |Z|
    """
    edge_r = edge_ridx.astype(int) - 1
    edge_z = edge_zidx.astype(int) - 1

    if edge_r.min() < 0 or edge_r.max() >= rCoM.size or edge_z.min() < 0 or edge_z.max() >= ZCoM.size:
        raise RuntimeError("Edge index arrays map out of rCoM/ZCoM bounds (check file consistency).")

    r_edge_pos = rCoM[edge_r]
    z_edge_pos = ZCoM[edge_z]

    iz0 = int(np.argmin(np.abs(ZCoM)))
    mask_iz0 = (edge_z == iz0)
    if np.any(mask_iz0):
        cand = np.where(mask_iz0)[0]
        i1 = cand[int(np.argmax(r_edge_pos[cand]))]
    else:
        zabs = np.abs(z_edge_pos)
        zmin = float(np.min(zabs))
        cand = np.where(zabs <= (zmin * 1.0001 + 1e-14))[0]
        i1 = cand[int(np.argmax(r_edge_pos[cand]))]
    idx_r_boundary = int(i1)

    ir0 = int(np.argmin(np.abs(rCoM)))
    mask_ir0 = (edge_r == ir0)
    if np.any(mask_ir0):
        cand = np.where(mask_ir0)[0]
        i2 = cand[int(np.argmax(np.abs(z_edge_pos[cand])))]
    else:
        rabs = np.abs(r_edge_pos)
        rmin = float(np.min(rabs))
        cand = np.where(rabs <= (rmin * 1.0001 + 1e-14))[0]
        i2 = cand[int(np.argmax(np.abs(z_edge_pos[cand])))]
    idx_z_boundary = int(i2)

    if idx_z_boundary == idx_r_boundary and np.any(mask_ir0):
        cand = np.where(mask_ir0)[0]
        order = np.argsort(-np.abs(z_edge_pos[cand]))
        for j in order:
            if int(cand[j]) != idx_r_boundary:
                idx_z_boundary = int(cand[j])
                break

    return [
        {"idx": idx_r_boundary, "label": r"LYDION: $r$-boundary near Z = 0", "color": "black"},
        {"idx": idx_z_boundary, "label": r"LYDION: $Z$-boundary near r = 0", "color": "crimson"},
    ]

###############################################################
#######                R E A D  F I L E                 #######
###############################################################

with h5py.File(FileName, "r") as f:
    t         = f["Times"][:]            # (Nt,)
    r         = f["r"][:]                # edges, (Nr+1,)   (normalized by Rcl)
    Z         = f["Z"][:]                # edges, (Nz+1,)   (normalized by Rcl)
    rCoM      = f["rCoM"][:]             # (Nr,)            (normalized by Rcl)
    ZCoM      = f["ZCoM"][:]             # (Nz,)            (normalized by Rcl)

    eLya_raw      = f["e_Lya_History"][:]    # 3D (time + 2 spatial axes, in some order)
    RbinC         = f["R_bincenters"][:]            # (Nbins,) (normalized by Rcl)
    eLya_sph_raw  = f["e_Lya_SphericalHistory"][:]  # (Nbins, Nt) or (Nt, Nbins)

    eLya_analytic = f["e_Lya_Analytical"][:] if "e_Lya_Analytical" in f else None

    J_edge_raw = f["J_Lya_EdgeSpectra"][:]
    nuC        = f["δνC"][:]                          # δν (Hz)
    edge_ridx  = f["J_Lya_Edge_rIndex"][:]            # 1-based (Julia)
    edge_zidx  = f["J_Lya_Edge_ZIndex"][:]            # 1-based (Julia)

    # Analytical spectrum for comparison (only saved as function of x)
    x_Analytical = f["x_Analytical"][:] if "x_Analytical" in f else None
    J_analytical = f["Spectrum_Analytical"][:] if "Spectrum_Analytical" in f else None

Nt = t.size
Nr = len(r) - 1
Nz = len(Z) - 1

saved_idx = np.where((t > 0) | (np.arange(t.size) == 0))[0]
if saved_idx.size == 0:
    raise RuntimeError("No saved frames found (saved_idx is empty).")

e_Lya_hist     = reorder_history_3d(eLya_raw, t, r, Z)
e_Lya_sph_hist = reorder_spherical_history(eLya_sph_raw, t)

J_edge_hist = reorder_lya_edge_spectra(J_edge_raw, t, nuC)
Nt_Lya, N_edge, N_nu = J_edge_hist.shape
if Nt_Lya != Nt:
    raise RuntimeError(f"Lyα history time dimension {Nt_Lya} != Times length {Nt}")

###############################################################
#######           P L O T  C O S M E T I C S            #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 37})
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

plt.rcParams.update({'legend.fontsize': 30})
plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 2.5
plt.rcParams['xtick.minor.width'] = 2.5
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['ytick.minor.width'] = 2.5
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['ytick.minor.size'] = 8
plt.rcParams['xtick.major.pad'] = '15'
plt.rcParams['ytick.major.pad'] = '15'

plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.linewidth'] = 3.0
plt.rcParams['grid.linewidth'] = 3.0
plt.rcParams['grid.linestyle'] = '--'

plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

###############################################################
#######                  S E T U P                      #######
###############################################################

R, ZZ = np.meshgrid(r, Z, indexing='ij')

# e_Lya range
if left_panel_mode == "map":
    eLya_max = float(np.nanmax(e_Lya_hist))
else:
    eLya_max = float(np.nanmax(e_Lya_sph_hist))

if not np.isfinite(eLya_max) or eLya_max <= 0.0:
    eLya_max = 1.0

if eLya_scale == "linear":
    vmin_Lya = 0.0
    vmax_Lya = 2.0 * eLya_max
    norm_Lya = None
elif eLya_scale == "log":
    vmin_Lya = 1e-8 * eLya_max
    vmax_Lya = 4.0 * eLya_max
    if vmin_Lya <= 0.0:
        vmin_Lya = 1e-30
    norm_Lya = LogNorm(vmin=vmin_Lya, vmax=vmax_Lya)
else:
    raise ValueError(f"Unknown eLya_scale={eLya_scale!r}. Use 'log' or 'linear'.")

# ν-window mask
if use_nu_window:
    nu_half_window = 0.5 * nu_window_fraction * nu_Lya
    nu_min_plot = -nu_half_window
    nu_max_plot = +nu_half_window
    nu_mask = (nuC >= nu_min_plot) & (nuC <= nu_max_plot)
else:
    nu_mask = np.ones_like(nuC, dtype=bool)

if not np.any(nu_mask):
    raise RuntimeError(
        f"nu_mask is empty. nuC range is [{nuC.min():.3e}, {nuC.max():.3e}] Hz, "
        f"requested window is [{nu_min_plot:.3e}, {nu_max_plot:.3e}] Hz. "
        "Increase nu_window_fraction or disable use_nu_window."
    )

nuC_plot = nuC[nu_mask]

# optional x
T_file = parse_temperature_from_filename(FileName)
if spectrum_xaxis == "x":
    if T_file is None:
        raise RuntimeError(
            "spectrum_xaxis='x' requested, but could not parse temperature from filename. "
            "Expected pattern like '_T10000.0_'."
        )
    vth = np.sqrt(2.0 * kB_cgs * T_file / mH_cgs)
    dnu_D = (vth / c_cgs) * nu_Lya
    if not np.isfinite(dnu_D) or dnu_D <= 0.0:
        raise RuntimeError(f"Computed Δν_D is invalid: {dnu_D}")
    x_plot = nuC_plot / dnu_D
else:
    x_plot = None

# spectra selection
if spectrum_selection == "two_boundaries":
    selected_specs = pick_two_boundary_edges(edge_ridx, edge_zidx, rCoM, ZCoM)
elif spectrum_selection == "manual":
    selected_specs = [{"idx": int(i), "label": "LYDION", "color": "crimson"} for i in edge_indices]
else:
    raise ValueError("spectrum_selection must be 'manual' or 'two_boundaries'.")

for s in selected_specs:
    if s["idx"] < 0 or s["idx"] >= N_edge:
        raise RuntimeError(f"Selected edge idx {s['idx']} is out of range [0, {N_edge-1}].")

# auto spectrum limits from final time
k_final = int(saved_idx[-1])
spec_x = x_plot if spectrum_xaxis == "x" else nuC_plot

y_final_all = []
for s in selected_specs:
    y = J_edge_hist[k_final, s["idx"], nu_mask]
    y = normalize_spectrum(y, spec_x)
    y_final_all.append(y)

Jmax_final = float(np.nanmax(y_final_all)) if len(y_final_all) else 1.0
if not np.isfinite(Jmax_final) or Jmax_final <= 0.0:
    Jmax_final = 1.0

spec_ymin = 0.0
spec_ymax = spec_ylim_factor * Jmax_final

thr = spec_thresh_factor * Jmax_final
mask_any = np.zeros_like(spec_x, dtype=bool)
for y in y_final_all:
    mask_any |= (y > thr)

if np.any(mask_any):
    x_lo = float(np.min(spec_x[mask_any]))
    x_hi = float(np.max(spec_x[mask_any]))
    span = x_hi - x_lo
    if not np.isfinite(span) or span <= 0.0:
        span = float(np.max(spec_x) - np.min(spec_x))
    pad = spec_xlim_pad_frac * span
    spec_xmin = x_lo - pad
    spec_xmax = x_hi + pad
else:
    spec_xmin = float(np.min(spec_x))
    spec_xmax = float(np.max(spec_x))

# Prepare analytical spectrum (ONLY if plotting in x-units)
spec_xA = None
spec_yA = None
if plot_analytical_spectrum and (spectrum_xaxis == "x") and (x_Analytical is not None) and (J_analytical is not None):
    xA_raw = np.asarray(x_Analytical, dtype=float)
    yA_raw = np.asarray(J_analytical, dtype=float)
    good = np.isfinite(xA_raw) & np.isfinite(yA_raw)
    if np.any(good):
        xg = xA_raw[good]
        yg = yA_raw[good]
        order = np.argsort(xg)
        spec_xA = xg[order]
        spec_yA = normalize_spectrum(yg[order], spec_xA)

###############################################################
#######                F I G U R E                      #######
###############################################################

# Turn OFF constrained_layout and control spacing explicitly (avoids cramped panels)
fig = plt.figure(figsize=(22, 11), constrained_layout=True)

# More space between panels so right y-ticks don't clash with left panel
gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.2], wspace=0.10)

# Also give the whole figure a bit more breathing room
fig.subplots_adjust(left=0.08, right=0.985, bottom=0.12, top=0.96)

ax_left = fig.add_subplot(gs[0, 0])
ax_spec = fig.add_subplot(gs[0, 1])

panel_label_style = dict(
    ha='right',
    va='top',
    color='black',
    path_effects=[pe.withStroke(linewidth=3, foreground='white')]
)

# --- LEFT PANEL ---
if left_panel_mode == "map":
    ax_left.set_xlabel(r"$r/R_{cl}$", labelpad=20)
    ax_left.set_ylabel(r"$Z/R_{cl}$", labelpad=20)

    e0 = np.clip(e_Lya_hist[0].T, vmin_Lya, vmax_Lya)
    im_Lya = ax_left.pcolormesh(
        R, ZZ,
        e0,
        norm=norm_Lya, shading='auto', cmap='gnuplot2'
    )
    fig.colorbar(im_Lya, ax=ax_left, pad=0.02)
    ax_left.text(0.9, 0.9, "e$_{Ly\\alpha}$", transform=ax_left.transAxes, **panel_label_style)

    ax_left.set_xlim(r[0], r[-1])
    ax_left.set_ylim(Z[0], Z[-1])

    # Minor ticks on x AND y for the colormap plot (per your request)
    ax_left.xaxis.set_minor_locator(AutoMinorLocator())
    ax_left.yaxis.set_minor_locator(AutoMinorLocator())

elif left_panel_mode == "axis":
    R_norm = RbinC
    ax_left.set_xlabel(r"$R/R_{cl}$", labelpad=20)
    ax_left.set_ylabel("Ly$\\alpha$ energy density e$_{Ly\\alpha}$", labelpad=20)
    ax_left.set_xlim(R_norm.min(), R_norm.max())
    ax_left.set_yscale("log" if eLya_scale == "log" else "linear")
    ax_left.set_ylim(vmin_Lya, vmax_Lya)

    y0 = np.clip(e_Lya_sph_hist[0, :], vmin_Lya, vmax_Lya)
    (line_Lya,) = ax_left.plot(
        R_norm, y0, linewidth=3,
        label="LYDION: Spherical average",
        color='dodgerblue'
    )

    if plot_analytical_left and (eLya_analytic is not None):
        xA = np.abs(np.asarray(ZCoM, dtype=float))
        yA = np.asarray(eLya_analytic, dtype=float)
        good = np.isfinite(xA) & np.isfinite(yA)
        if np.any(good):
            xAg = xA[good]
            yAg = yA[good]
            order = np.argsort(xAg)
            xAg = xAg[order]
            yAg = yAg[order]
            ax_left.plot(
                xAg, yAg,
                linestyle=analytical_linestyle,
                linewidth=analytical_linewidth,
                label=analytical_label,
                color="black"
            )

    ax_left.legend(
        loc=left_legend_loc,
        bbox_to_anchor=left_legend_anchor,
        fontsize=left_legend_fontsize,
        handlelength=legend_handlelength,
        labelspacing=legend_labelspacing,
        borderpad=legend_borderpad,
        handletextpad=legend_handletextpad
    )

    # Minor ticks on LEFT x-axis (per your request)
    ax_left.xaxis.set_minor_locator(AutoMinorLocator())

else:
    raise ValueError(f"Unknown left_panel_mode={left_panel_mode!r}. Use 'map' or 'axis'.")

# --- SPECTRUM PANEL ---
ax_spec.set_ylim(spec_ymin, spec_ymax)
ax_spec.set_xlim(spec_xmin, spec_xmax)

if spectrum_xaxis == "x":
    ax_spec.set_xlabel(r"$x = (\nu - \nu_{Ly\alpha})/\Delta \nu_D$", labelpad=20)
else:
    ax_spec.set_xlabel(r"$\nu - \nu_{Ly\alpha}$ (Hz)", labelpad=20)

# Minor ticks on spectrum panel (both axes)
ax_spec.minorticks_on()
ax_spec.xaxis.set_minor_locator(AutoMinorLocator())
ax_spec.yaxis.set_minor_locator(AutoMinorLocator())

spec_lines = []
k0 = int(saved_idx[0])
for s in selected_specs:
    idx   = s["idx"]
    label = s["label"]
    color = s["color"]

    y0s = J_edge_hist[k0, idx, nu_mask]
    y0s = normalize_spectrum(y0s, spec_x)

    (line,) = ax_spec.plot(
        spec_x, y0s,
        label=label, linewidth=3, color=color
    )
    spec_lines.append(line)

# Plot analytical spectrum (static) ONLY if we are plotting in x-units
line_spec_analytic = None
if plot_analytical_spectrum and (spectrum_xaxis == "x") and (spec_xA is not None) and (spec_yA is not None):
    (line_spec_analytic,) = ax_spec.plot(
        spec_xA, spec_yA,
        label=analytical_spec_label,
        color=analytical_spec_color,
        linestyle=analytical_spec_ls,
        linewidth=analytical_spec_lw
    )

ax_spec.legend(
    loc=spec_legend_loc,
    bbox_to_anchor=spec_legend_anchor,
    fontsize=spec_legend_fontsize,
    handlelength=legend_handlelength,
    labelspacing=legend_labelspacing,
    borderpad=legend_borderpad,
    handletextpad=legend_handletextpad
)

# "Emergent spectra" text inside spectrum panel, just above/near the legend region
if spectrum_xaxis  == "x":
    spec_label = "Emergent spectra: J(x)/$\int$ dx J"
else:
    spec_label = "Emergent spectra: J($\\nu$)/$\int$ $d\\nu$ J"

ax_spec.text(
    0.50, 0.90, spec_label,
    transform=ax_spec.transAxes,
    ha="center", va="top",
    fontsize=34
)

# --- Time label styles ---
time_style_map = dict(
    ha='left', va='top', color='white',
    path_effects=[pe.withStroke(linewidth=3, foreground='black')]
)
time_style_axis = dict(
    ha='left', va='top', color='black'
)
time_style = time_style_axis if left_panel_mode == "axis" else time_style_map
txt = ax_left.text(0.12, 0.12, "", fontsize=30, transform=ax_left.transAxes, **time_style)

frames_all = saved_idx
frames     = frames_all[::frameskip]

###############################################################
#######              A N I M A T I O N                  #######
###############################################################

def _time_label(val):
    return rf"$t = {val:6.2f}\,\times\,(a\tau_{{cl}})^{{1/3}}\,R_{{cl}}/c$"

def init():
    k = int(frames[0])

    if left_panel_mode == "map":
        im_Lya.set_array(np.clip(e_Lya_hist[k].T, vmin_Lya, vmax_Lya).ravel())
        left_artists = (im_Lya,)
    else:
        line_Lya.set_ydata(np.clip(e_Lya_sph_hist[k, :], vmin_Lya, vmax_Lya))
        left_artists = (line_Lya,)

    for line, s in zip(spec_lines, selected_specs):
        y = J_edge_hist[k, s["idx"], nu_mask]
        y = normalize_spectrum(y, spec_x)
        line.set_ydata(y)

    txt.set_text(_time_label(t[k]))

    if line_spec_analytic is None:
        return (*left_artists, txt, *spec_lines)
    return (*left_artists, txt, *spec_lines, line_spec_analytic)

def update(k):
    k = int(k)

    if left_panel_mode == "map":
        im_Lya.set_array(np.clip(e_Lya_hist[k].T, vmin_Lya, vmax_Lya).ravel())
        left_artists = (im_Lya,)
    else:
        line_Lya.set_ydata(np.clip(e_Lya_sph_hist[k, :], vmin_Lya, vmax_Lya))
        left_artists = (line_Lya,)

    for line, s in zip(spec_lines, selected_specs):
        y = J_edge_hist[k, s["idx"], nu_mask]
        y = normalize_spectrum(y, spec_x)
        line.set_ydata(y)

    txt.set_text(_time_label(t[k]))

    if line_spec_analytic is None:
        return (*left_artists, txt, *spec_lines)
    return (*left_artists, txt, *spec_lines, line_spec_analytic)

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

if save_mp4:
    ani.save(outname, writer="ffmpeg", fps=fps, dpi=300)
else:
    plt.show()