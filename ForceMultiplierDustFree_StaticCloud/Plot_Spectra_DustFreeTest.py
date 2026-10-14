import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from pathlib import Path
import re


###############################################################
#######                                                 #######
#######                 U S E R   S E T U P             #######
#######                                                 #######
###############################################################

repo_root = Path(__file__).resolve().parent

data_dir = repo_root / "LYDION_DataDustFreeStaticClouds"

save_name = "DustFree_StaticCloud_EmergentSpectra.pdf"
save_path = repo_root / save_name

# Runs to plot.
tau_cl_values = [1.0e8, 1.0e9, 1.0e10]
resolution    = (128, 128)

# One color per tau_cl.
color_by_tau = {
    1.0e8:  "black",
    1.0e9:  "dodgerblue",
    1.0e10: "orangered",
}

# Only include files matching these choices.
require_NoRecoil = True
require_tauabs0  = True
require_pd0      = True
require_umax0    = True

# If several files match the same tau/resolution, choose newest file.
choose_newest_matching_file = True

# Spectrum edge selection:
#   "two_boundaries" : same spirit as AnimateStaticCloudTest.py
#   "manual"         : use edge_indices
#   "all"            : plot all saved edge spectra individually
#   "average_all"    : average all saved edge spectra
#
# I strongly recommend "two_boundaries" or "manual" for debugging.
edge_selection_mode = "two_boundaries"

# Used only if edge_selection_mode = "manual".
edge_indices = [0]

# Plot individual selected LYDION spectra.
# If False, average selected spectra before plotting.
plot_selected_edges_individually = True

# Normalize each spectrum so integral J dx = 1.
normalize_spectra = True

# Plot analytical spectra saved in the HDF5 files.
plot_analytical_spectra = True

# Line styles.
lydion_ls      = "-"
analytical_ls  = "--"
lydion_lw      = 3.5
analytical_lw  = 4.0

# For edge spectra when plotting two boundaries.
edge_alpha_values = [1.0, 0.55]
edge_lw_values    = [4.0, 3.0]

# Optional x/y limits.
xlim = (-270,270)
ylim = None

# Automatic x limits are based on where spectra exceed this fraction
# of the maximum plotted normalized spectrum.
auto_xlim_threshold = 2.0e-3
auto_xlim_pad_frac  = 0.08

# Figure cosmetics.
figsize = (12.0, 10.5)
figure_title = r"Dust-free static cloud, uniform emission, $T=10^4$ K"

legend_loc = "upper left"
legend_fontsize = 30

# Optional text box.
add_info_text = False
info_text = (
    "Uniform emission, ignoring recoil," + "\n" + r"and Ly$\alpha$ destruction."
    + "\n"
    + rf"Resolution: ${resolution[0]}^2$."
)

info_text_xy = (0.04, 0.94)
info_text_fontsize = 27


###############################################################
#######                                                 #######
#######                 C O N S T A N T S               #######
#######                                                 #######
###############################################################

c_cgs  = 2.99792458e10
kB_cgs = 1.380649e-16
mH_cgs = 1.6735575e-24

nu_Lya = 2.466e15  # Hz


###############################################################
#######                                                 #######
#######                 H E L P E R S                   #######
#######                                                 #######
###############################################################

tau_pattern    = re.compile(r"_tau([0-9eE+\-.]+)_")
res_pattern    = re.compile(r"_(\d+)x(\d+)\.h5$")
tauabs_pattern = re.compile(r"tauabs([0-9eE+\-.]+)")
pd_pattern     = re.compile(r"pd([0-9eE+\-.]+)")
umax_pattern   = re.compile(r"umax([0-9eE+\-.]+)")
T_pattern      = re.compile(r"_T([0-9eE+\-.]+)_")


def parse_float_from_pattern(pattern, filename, label):
    match = pattern.search(filename)
    if match is None:
        raise ValueError(f"Could not parse {label} from {filename}")
    return float(match.group(1))


def parse_temperature_from_filename(filename):
    match = T_pattern.search(filename)
    if match is None:
        raise ValueError(
            f"Could not parse temperature from {filename}. "
            "Expected substring like '_T10000.0_'."
        )
    return float(match.group(1))


def doppler_width_from_T(T):
    vth = np.sqrt(2.0 * kB_cgs * T / mH_cgs)
    return (vth / c_cgs) * nu_Lya


def integrate_trap(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))

    dx = np.diff(x)
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * dx))


def normalize_spectrum(y, x):
    y = np.asarray(y, dtype=float)

    if not normalize_spectra:
        return y

    area = integrate_trap(y, x)

    if not np.isfinite(area) or area <= 0.0:
        return y

    return y / area


def reorder_lya_edge_spectra_safe(J_edge_raw, t, nuC, edge_ridx):
    """
    Reorder J_Lya_EdgeSpectra to shape (Nt, N_edge, N_nu).

    Safer than identifying only time/frequency axes: explicitly uses
    len(J_Lya_Edge_rIndex) to identify the edge axis.
    """
    if J_edge_raw.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {J_edge_raw.shape}")

    Nt    = t.size
    Nν    = nuC.size
    Nedge = edge_ridx.size

    shape = J_edge_raw.shape

    time_axes = [ax for ax, s in enumerate(shape) if s == Nt]
    nu_axes   = [ax for ax, s in enumerate(shape) if s == Nν]
    edge_axes = [ax for ax, s in enumerate(shape) if s == Nedge]

    candidates = []

    for t_ax in time_axes:
        for e_ax in edge_axes:
            for nu_ax in nu_axes:
                if len({t_ax, e_ax, nu_ax}) == 3:
                    candidates.append((t_ax, e_ax, nu_ax))

    if len(candidates) != 1:
        raise RuntimeError(
            "Could not uniquely identify axes of J_Lya_EdgeSpectra.\n"
            f"  shape  = {shape}\n"
            f"  Nt     = {Nt}\n"
            f"  Nedge  = {Nedge}\n"
            f"  Nν     = {Nν}\n"
            f"  candidates = {candidates}\n"
            "This usually means two dimensions have identical lengths. "
            "In that case, inspect the HDF5 save convention and hard-code the axes."
        )

    t_ax, e_ax, nu_ax = candidates[0]
    J_reordered = np.transpose(J_edge_raw, axes=[t_ax, e_ax, nu_ax])

    if J_reordered.shape != (Nt, Nedge, Nν):
        raise RuntimeError(
            f"Unexpected reordered shape {J_reordered.shape}, "
            f"expected {(Nt, Nedge, Nν)}."
        )

    print(
        f"    J_Lya_EdgeSpectra raw shape {shape} -> "
        f"(time, edge, freq) axes = ({t_ax}, {e_ax}, {nu_ax})"
    )

    return J_reordered


def pick_two_boundary_edges(edge_ridx, edge_zidx, rCoM, ZCoM):
    """
    Auto-pick two edge indices:
      1) r-boundary near Z ~ 0
      2) Z-boundary near r ~ 0

    This follows the logic used in AnimateStaticCloudTest.py.
    """
    edge_r = edge_ridx.astype(int) - 1
    edge_z = edge_zidx.astype(int) - 1

    if (
        edge_r.min() < 0 or edge_r.max() >= rCoM.size
        or edge_z.min() < 0 or edge_z.max() >= ZCoM.size
    ):
        raise RuntimeError("Edge index arrays map out of rCoM/ZCoM bounds.")

    r_edge_pos = rCoM[edge_r]
    z_edge_pos = ZCoM[edge_z]

    iz0 = int(np.argmin(np.abs(ZCoM)))
    mask_iz0 = edge_z == iz0

    if np.any(mask_iz0):
        cand = np.where(mask_iz0)[0]
        idx_r_boundary = int(cand[np.argmax(r_edge_pos[cand])])
    else:
        zabs = np.abs(z_edge_pos)
        cand = np.where(zabs <= np.min(zabs) * 1.0001 + 1e-14)[0]
        idx_r_boundary = int(cand[np.argmax(r_edge_pos[cand])])

    ir0 = int(np.argmin(np.abs(rCoM)))
    mask_ir0 = edge_r == ir0

    if np.any(mask_ir0):
        cand = np.where(mask_ir0)[0]
        idx_z_boundary = int(cand[np.argmax(np.abs(z_edge_pos[cand]))])
    else:
        rabs = np.abs(r_edge_pos)
        cand = np.where(rabs <= np.min(rabs) * 1.0001 + 1e-14)[0]
        idx_z_boundary = int(cand[np.argmax(np.abs(z_edge_pos[cand]))])

    if idx_z_boundary == idx_r_boundary and np.any(mask_ir0):
        cand = np.where(mask_ir0)[0]
        order = np.argsort(-np.abs(z_edge_pos[cand]))

        for j in order:
            if int(cand[j]) != idx_r_boundary:
                idx_z_boundary = int(cand[j])
                break

    return np.array([idx_r_boundary, idx_z_boundary], dtype=int)


def select_edge_indices(f, mode):
    edge_ridx = f["J_Lya_Edge_rIndex"][:]
    edge_zidx = f["J_Lya_Edge_ZIndex"][:]

    Nedge = len(edge_ridx)

    if mode == "two_boundaries":
        rCoM = f["rCoM"][:]
        ZCoM = f["ZCoM"][:]
        idx = pick_two_boundary_edges(edge_ridx, edge_zidx, rCoM, ZCoM)

    elif mode == "manual":
        idx = np.asarray(edge_indices, dtype=int)
        if np.any(idx < 0) or np.any(idx >= Nedge):
            raise RuntimeError(
                f"Manual edge_indices={edge_indices} out of range [0,{Nedge-1}]."
            )

    elif mode == "all":
        idx = np.arange(Nedge, dtype=int)

    elif mode == "average_all":
        idx = np.arange(Nedge, dtype=int)

    else:
        raise ValueError(
            "edge_selection_mode must be 'two_boundaries', 'manual', "
            "'all', or 'average_all'."
        )

    return idx


def file_matches_settings(filepath, tau_target):
    filename = filepath.name

    if require_NoRecoil and ("NoRecoil" not in filename):
        return False

    try:
        tau_cl = parse_float_from_pattern(tau_pattern, filename, "tau_cl")
    except ValueError:
        return False

    if not np.isclose(tau_cl, tau_target, rtol=1e-6, atol=0.0):
        return False

    match_res = res_pattern.search(filename)
    if match_res is None:
        return False

    Nr = int(match_res.group(1))
    NZ = int(match_res.group(2))

    if (Nr, NZ) != tuple(resolution):
        return False

    if require_tauabs0:
        try:
            tauabs = parse_float_from_pattern(tauabs_pattern, filename, "tauabs")
            if not np.isclose(tauabs, 0.0):
                return False
        except ValueError:
            return False

    if require_pd0:
        try:
            pd = parse_float_from_pattern(pd_pattern, filename, "pd")
            if not np.isclose(pd, 0.0):
                return False
        except ValueError:
            return False

    if require_umax0:
        try:
            umax = parse_float_from_pattern(umax_pattern, filename, "umax")
            if not np.isclose(umax, 0.0):
                return False
        except ValueError:
            return False

    return True


def find_file_for_tau(tau_target):
    candidates = [
        filepath
        for filepath in sorted(data_dir.glob("*.h5"))
        if file_matches_settings(filepath, tau_target)
    ]

    if len(candidates) == 0:
        raise RuntimeError(
            f"No matching HDF5 file found for tau_cl={tau_target:.4e}, "
            f"resolution={resolution}, data_dir={data_dir}."
        )

    if len(candidates) == 1:
        return candidates[0]

    if choose_newest_matching_file:
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = candidates[-1]
    else:
        chosen = candidates[0]

    print(f"\nMultiple files matched tau_cl={tau_target:.4e}.")
    print(f"Chosen file: {chosen.name}")
    print("Other candidates:")
    for c in candidates:
        if c != chosen:
            print(f"  {c.name}")

    return chosen


def peak_positions(x, y, nmax=4):
    """
    Quick diagnostic peak finder without scipy.
    Returns up to nmax local maxima, sorted by peak height.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size < 3:
        return []

    loc = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1

    if loc.size == 0:
        return []

    order = np.argsort(y[loc])[::-1]
    loc = loc[order[:nmax]]

    return [(float(x[i]), float(y[i])) for i in loc]


def read_final_spectra(filepath):
    """
    Read final-time LYDION spectra and saved analytical spectrum.

    Returns:
        dict with x, y_edges, edge_indices, x_ana, y_ana, t_final, k_final, T
    """
    filename = filepath.name
    T = parse_temperature_from_filename(filename)
    dnu_D = doppler_width_from_T(T)

    with h5py.File(filepath, "r") as f:
        t = np.asarray(f["Times"][:], dtype=float)

        J_edge_raw = f["J_Lya_EdgeSpectra"][:]
        nuC = np.asarray(f["δνC"][:], dtype=float)

        edge_ridx = f["J_Lya_Edge_rIndex"][:]
        edge_zidx = f["J_Lya_Edge_ZIndex"][:]

        selected_idx = select_edge_indices(f, edge_selection_mode)

        J_edge_hist = reorder_lya_edge_spectra_safe(J_edge_raw, t, nuC, edge_ridx)

        # Use the maximum saved time. This is robust against preallocated trailing zeros.
        finite_t = np.isfinite(t)
        if not np.any(finite_t):
            raise RuntimeError(f"{filename}: no finite values in Times.")

        k_final = int(np.nanargmax(t))
        t_final = float(t[k_final])

        x = nuC / dnu_D

        y_edges = []

        if edge_selection_mode == "average_all" or not plot_selected_edges_individually:
            y = np.zeros_like(x, dtype=float)

            for idx in selected_idx:
                y += np.asarray(J_edge_hist[k_final, idx, :], dtype=float)

            y /= float(len(selected_idx))
            y_edges.append((None, normalize_spectrum(y, x)))

        else:
            for idx in selected_idx:
                y = np.asarray(J_edge_hist[k_final, idx, :], dtype=float)
                y_edges.append((int(idx), normalize_spectrum(y, x)))

        # Sort in x.
        order = np.argsort(x)
        x = x[order]

        y_edges_sorted = []
        for idx, y in y_edges:
            y_edges_sorted.append((idx, y[order]))

        x_ana = None
        y_ana = None

        if plot_analytical_spectra:
            if ("x_Analytical" in f) and ("Spectrum_Analytical" in f):
                x_ana_raw = np.asarray(f["x_Analytical"][:], dtype=float)
                y_ana_raw = np.asarray(f["Spectrum_Analytical"][:], dtype=float)

                goodA = np.isfinite(x_ana_raw) & np.isfinite(y_ana_raw)

                if np.any(goodA):
                    x_ana = x_ana_raw[goodA]
                    y_ana = y_ana_raw[goodA]

                    orderA = np.argsort(x_ana)
                    x_ana = x_ana[orderA]
                    y_ana = y_ana[orderA]

                    y_ana = normalize_spectrum(y_ana, x_ana)
            else:
                print(
                    f"WARNING: {filename} has no x_Analytical/Spectrum_Analytical datasets. "
                    "Skipping analytical curve for this tau."
                )

    return {
        "filename": filename,
        "x": x,
        "y_edges": y_edges_sorted,
        "edge_indices": selected_idx,
        "x_ana": x_ana,
        "y_ana": y_ana,
        "t_final": t_final,
        "k_final": k_final,
        "T": T,
    }


def tau_label(tau):
    exp = np.log10(tau)
    if np.isclose(exp, np.round(exp)):
        return rf"$\tau_{{cl}}=10^{{{int(np.round(exp))}}}$"
    return rf"$\tau_{{cl}}={tau:.2g}$"


def clean_tick_formatter(val, pos=None):
    if val == 0:
        return "0"

    if np.isclose(val, np.round(val), rtol=0.0, atol=1e-10):
        return f"{int(np.round(val))}"

    return f"{val:g}"


def choose_auto_xlim(all_curves):
    ymax = 0.0

    for x, y in all_curves:
        good = np.isfinite(x) & np.isfinite(y)
        if np.any(good):
            ymax = max(ymax, float(np.nanmax(y[good])))

    if ymax <= 0.0 or not np.isfinite(ymax):
        return None

    threshold = auto_xlim_threshold * ymax
    x_keep = []

    for x, y in all_curves:
        good = np.isfinite(x) & np.isfinite(y) & (y > threshold)
        if np.any(good):
            x_keep.append(x[good])

    if len(x_keep) == 0:
        return None

    xcat = np.concatenate(x_keep)

    xmin = float(np.nanmin(xcat))
    xmax = float(np.nanmax(xcat))

    span = xmax - xmin
    if span <= 0.0 or not np.isfinite(span):
        return None

    pad = auto_xlim_pad_frac * span
    return xmin - pad, xmax + pad


###############################################################
#######                                                 #######
#######                 R E A D   D A T A               #######
#######                                                 #######
###############################################################

spectra = {}

print("\nReading LYDION spectra:")
for tau_cl in tau_cl_values:
    filepath = find_file_for_tau(tau_cl)
    data = read_final_spectra(filepath)
    spectra[tau_cl] = data

    print(
        f"  tau_cl = {tau_cl:.4e}, "
        f"T = {data['T']:.6g} K, "
        f"k_final = {data['k_final']}, "
        f"t_final = {data['t_final']:.6g}, "
        f"file = {data['filename']}"
    )
    print(f"    selected edge indices = {list(data['edge_indices'])}")

    for idx, y in data["y_edges"]:
        peaks = peak_positions(data["x"], y)
        print(f"    edge {idx}: strongest local peaks (x, y) = {peaks}")


###############################################################
#######                                                 #######
#######            P L O T  C O S M E T I C S           #######
#######                                                 #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 39.5})
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

params = {'legend.fontsize': legend_fontsize}
plt.rcParams.update(params)
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
#######                                                 #######
#######                   P L O T   I T                 #######
#######                                                 #######
###############################################################

fig, ax = plt.subplots(figsize=figsize)

all_curves_for_xlim = []

# Manual optical-depth text labels.
# Tweak these after you adjust figure width/xlim/ylim.
add_tau_text_labels = True

tau_text_positions = {
    1.0e8:  (45.0, 0.013),
    1.0e9:  (75.0, 0.0065),
    1.0e10: (120.0, 0.0043),
}

tau_text_fontsize = 35


for tau_cl in tau_cl_values:

    data = spectra[tau_cl]
    color = color_by_tau.get(tau_cl, "black")
    label_tau = tau_label(tau_cl)

    # Only the black/default tau_cl = 1e8 curves enter the legend.
    use_legend_label = np.isclose(tau_cl, 1.0e8)

    # Analytical spectrum first, dashed.
    if plot_analytical_spectra and data["x_ana"] is not None and data["y_ana"] is not None:

        ana_label = r"Analytical" if use_legend_label else "_nolegend_"

        ax.plot(
            data["x_ana"],
            data["y_ana"],
            linestyle=analytical_ls,
            linewidth=analytical_lw,
            color=color,
            dash_capstyle="round",
            alpha=0.85,
            label=ana_label,
        )

        all_curves_for_xlim.append((data["x_ana"], data["y_ana"]))

    # LYDION selected edge spectra.
    for n_edge_plot, (idx, y) in enumerate(data["y_edges"]):

        if use_legend_label:
            if idx is None:
                lydion_label = r"LYDION"
            elif edge_selection_mode == "two_boundaries":
                if n_edge_plot == 0:
                    lydion_label = r"LYDION ($R$)"
                elif n_edge_plot == 1:
                    lydion_label = r"LYDION ($Z$)"
                else:
                    lydion_label = rf"LYDION, edge {idx}"
            else:
                lydion_label = rf"LYDION, edge {idx}"
        else:
            lydion_label = "_nolegend_"

        alpha = edge_alpha_values[n_edge_plot % len(edge_alpha_values)]
        lw    = edge_lw_values[n_edge_plot % len(edge_lw_values)]

        ax.plot(
            data["x"],
            y,
            linestyle=lydion_ls,
            linewidth=lw,
            color=color,
            alpha=alpha,
            label=lydion_label,
        )

        all_curves_for_xlim.append((data["x"], y))


# Add optical-depth labels by hand in the panel.
if add_tau_text_labels:
    for tau_cl in tau_cl_values:
        if tau_cl not in tau_text_positions:
            continue

        x_txt, y_txt = tau_text_positions[tau_cl]
        color = color_by_tau.get(tau_cl, "black")

        ax.text(
            x_txt,
            y_txt,
            tau_label(tau_cl),
            color=color,
            fontsize=tau_text_fontsize,
            ha="left",
            va="center",
        )

ax.set_xlabel(r"$x = (\nu - \nu_{Ly\alpha})/\Delta\nu_D$", labelpad=15)

if normalize_spectra:
    ax.set_ylabel(r"$J(x)/\int dx\, J(x)$", labelpad=15)
else:
    ax.set_ylabel(r"$J(x)$", labelpad=15)

#ax.set_title(figure_title, pad=20, fontsize=39.5)

if xlim is not None:
    ax.set_xlim(*xlim)
else:
    auto_xlim = choose_auto_xlim(all_curves_for_xlim)
    if auto_xlim is not None:
        ax.set_xlim(*auto_xlim)

if ylim is not None:
    ax.set_ylim(*ylim)
else:
    ax.set_ylim(bottom=0.0)

ax.xaxis.set_major_formatter(FuncFormatter(clean_tick_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(clean_tick_formatter))

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

if add_info_text:
    ax.text(
        info_text_xy[0],
        info_text_xy[1],
        info_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=info_text_fontsize,
    )

ax.legend(
    loc=legend_loc,
    fontsize=legend_fontsize,
    handlelength=1.6,
    labelspacing=0.35,
    borderpad=0.25*3,
    handletextpad=0.6,
)

fig.tight_layout()
fig.savefig(save_path, bbox_inches="tight", dpi=250)
print(f"\nSaved figure: {save_path}")

plt.show()