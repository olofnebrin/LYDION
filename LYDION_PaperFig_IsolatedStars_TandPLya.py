import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
cmastro_src = repo_root / "src" / "cmastro" / "src"
sys.path.insert(0, str(cmastro_src))
import cmastro

# ============================================================
#   U S E R   I N P U T S
# ============================================================

# --- Row definitions ---
# Each entry: (label shown on the left, no-Lyα filename, Lyα filename)
ROWS = [
    (r"$u_{RMS} = 0$ km s$^{-1}$",
     "RHDSim_NoLya_u0_256x256_2026-05-02.h5",
     "RHDSim_Lya_u0_256x256_2026-05-02.h5"),

    (r"$u_{RMS} = 4$ km s$^{-1}$",
     "RHDSim_NoLya_u4_256x256_2026-05-02.h5",
     "RHDSim_Lya_u4_256x256_2026-05-02.h5"),

    (r"$u_{RMS} = 8$ km s$^{-1}$",
     "RHDSim_NoLya_u8_256x256_2026-05-02.h5",
     "RHDSim_Lya_u8_256x256_2026-05-02.h5"),

    (r"No dust, $u_{RMS} = 4$ km s$^{-1}$",
     "RHDSim_NoLya_NoDust_u4_256x256_2026-05-06.h5",
     "RHDSim_Lya_NoDust_u4_256x256_2026-05-06.h5"),
]

FigureName = "LYDION_Figure_IsolatedStars_T_PLya.pdf"   # or .png

# --- Font sizes ---
fontsize_title       = 36   # title text above each column
fontsize_cbar_ticks  = 30   # colorbar tick numbers
fontsize_panel_label = 40   # "T (K)" / "P_Lyα/P" text
fontsize_time        = 38   # time stamp inside panel
fontsize_row_label   = 48   # u_RMS label on the left

# --- Color maps and ranges ---
T_vmin, T_vmax = 1.0, 1e5
cmap_T         = 'coolwarm'

Ratio_vmin, Ratio_vmax = 0.1, 10.0
cmap_ratio             = 'cma:unph'

# --- Diagnostics text options ---
ShowDiagnosticsText          = False
ShowLyaForceText             = True
ShowLyaMultiplierText        = True
ShowLyaEscapeFractionText    = True
ShowLyaLuminosityText        = False
ShowDirectForceText          = False
ShowIRForceText              = False

# Force variant: "radial_net" or "antigravity"
ForceVariant = "radial_net"

fontsize_diagnostics_text = 45
diagnostics_x             = 0.04
diagnostics_y             = 0.85

# --- Spatial plotting range ---
xmax_fraction = 1.0
ymax_fraction = 1.0

# ============================================================
#   C O N S T A N T S
# ============================================================

kB   = 1.38e-16
mH   = 1.67e-24
X    = 0.76
yr   = 3.1556926e7
kyr  = 1e3 * yr
pc   = 3.085677581e18
c    = 2.99792458e10
Lsun = 3.839e33

# ============================================================
#   P L O T   C O S M E T I C S
# ============================================================

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 39.5})
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

plt.rcParams['xtick.top']       = True
plt.rcParams['xtick.bottom']    = True
plt.rcParams['ytick.left']      = True
plt.rcParams['ytick.right']     = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ============================================================
#   H E L P E R   F U N C T I O N S
# ============================================================

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
            f"After moving time axis, spatial dims {a1, a2} do not match Nr={Nr}, Nz={Nz}"
        )

def read_dataset_if_exists(file, name, default=None):
    return file[name][:] if name in file else default

def read_attr_if_exists(group, name, default=None):
    return group.attrs[name] if group is not None and name in group.attrs else default

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

def fmt_math_value(x):
    if x is None or (not np.isfinite(x)):
        return None
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 100.0 or ax < 0.01:
        exp  = int(np.floor(np.log10(ax)))
        mant = x / (10.0**exp)
        return rf"{mant:.2f}\times10^{{{exp}}}"
    elif ax >= 10.0:
        return f"{x:.1f}"
    else:
        return f"{x:.2f}"

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
        return r"$10^{2}$"
    else:
        return rf"$10^{{{exp}}}$"

def read_run(filename, load_P=False, load_eLya=False):
    """Read one run and return arrays needed for the T and P_Lyα/P figure."""
    with h5py.File(filename, "r") as f:
        t     = f["Times"][:]
        r     = f["r"][:]
        Z     = f["Z"][:]
        T_raw = f["T_History"][:]

        P_raw    = f["GasPressure_History"][:] if (load_P and "GasPressure_History" in f) else None
        eLya_raw = f["e_Lya_History"][:]       if (load_eLya and "e_Lya_History"       in f) else None

        settings = f["Settings"] if "Settings" in f else None
        L_bol    = read_attr_if_exists(settings, "L_bol", None)

        M_F_Lya        = read_dataset_if_exists(f, "M_F_History")
        M_F_Lya_radial = read_dataset_if_exists(f, "M_F_radial_net_History")
        L_Lya          = read_dataset_if_exists(f, "L_Lya_History")
        f_esc_Lya      = read_first_existing_dataset(
            f, ["f_esc_Lya_History", "fesc_Lya_History", "f_esc_History_Lya"], default=None
        )
        M_F_dir        = read_dataset_if_exists(f, "M_F_dir_History")
        M_F_dir_radial = read_dataset_if_exists(f, "M_F_dir_radial_net_History")
        M_F_IR         = read_dataset_if_exists(f, "M_F_IR_History")
        M_F_IR_radial  = read_dataset_if_exists(f, "M_F_IR_radial_net_History")
        L_IR           = read_dataset_if_exists(f, "L_IR_History")

    T_hist    = reorder_history_3d(T_raw,    t, r, Z)
    P_hist    = reorder_history_3d(P_raw,    t, r, Z) if P_raw    is not None else None
    eLya_hist = reorder_history_3d(eLya_raw, t, r, Z) if eLya_raw is not None else None

    saved_idx = np.where((t > 0.0) | (np.arange(t.size) == 0))[0]

    return dict(
        t=t, r=r, Z=Z,
        T_hist=T_hist, P_hist=P_hist, eLya_hist=eLya_hist,
        saved_idx=saved_idx, L_bol=L_bol,
        M_F_Lya=M_F_Lya, M_F_Lya_radial=M_F_Lya_radial,
        L_Lya=L_Lya, f_esc_Lya=f_esc_Lya,
        M_F_dir=M_F_dir, M_F_dir_radial=M_F_dir_radial,
        M_F_IR=M_F_IR, M_F_IR_radial=M_F_IR_radial, L_IR=L_IR
    )

def find_last_snapshot(run):
    """Return the index of the last saved snapshot."""
    return run["saved_idx"][-1]

def _pick_Lya_force(run, k):
    L_Lya_val = safe_scalar_at_index(run["L_Lya"], k)
    L_bol_val = run["L_bol"]
    try:
        if L_bol_val is not None:
            L_bol_val = float(np.asarray(L_bol_val))
    except Exception:
        L_bol_val = None

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

    L_Lya_val          = safe_scalar_at_index(run["L_Lya"], k)
    M_F_dir_val        = safe_scalar_at_index(run["M_F_dir"], k)
    M_F_dir_radial_val = safe_scalar_at_index(run["M_F_dir_radial"], k)
    M_F_IR_val         = safe_scalar_at_index(run["M_F_IR"], k)
    M_F_IR_radial_val  = safe_scalar_at_index(run["M_F_IR_radial"], k)
    L_IR_val           = safe_scalar_at_index(run["L_IR"], k)

    L_bol_val = run["L_bol"]
    try:
        if L_bol_val is not None:
            L_bol_val = float(np.asarray(L_bol_val))
    except Exception:
        L_bol_val = None

    if ShowLyaForceText:
        F_Lya = _pick_Lya_force(run, k)
        if F_Lya is not None:
            F_str = fmt_math_value(F_Lya)
            if F_str is not None:
                lines.append(rf"$F_{{Ly\alpha}} = {F_str}\,L_{{bol}}/c$")

    if ShowDirectForceText and (L_bol_val is not None):
        if M_F_dir_radial_val is not None:
            F_str = fmt_math_value(M_F_dir_radial_val)
            if F_str is not None:
                lines.append(rf"$F_{{dir}} = {F_str}\,L_{{bol}}/c$")
        elif M_F_dir_val is not None:
            F_str = fmt_math_value(M_F_dir_val)
            if F_str is not None:
                lines.append(rf"$F_{{dir}} = {F_str}\,L_{{bol}}/c$")

    if ShowIRForceText and (L_bol_val is not None) and (L_IR_val is not None):
        if M_F_IR_radial_val is not None:
            F_str = fmt_math_value(M_F_IR_radial_val * L_IR_val / L_bol_val)
            if F_str is not None:
                lines.append(rf"$F_{{IR}} = {F_str}\,L_{{bol}}/c$")
        elif M_F_IR_val is not None:
            F_str = fmt_math_value(M_F_IR_val * L_IR_val / L_bol_val)
            if F_str is not None:
                lines.append(rf"$F_{{IR}} = {F_str}\,L_{{bol}}/c$")

    if ShowLyaMultiplierText:
        if ForceVariant == "radial_net":
            M_F_val = safe_scalar_at_index(run["M_F_Lya_radial"], k)
            if M_F_val is None:
                M_F_val = safe_scalar_at_index(run["M_F_Lya"], k)
        else:
            M_F_val = safe_scalar_at_index(run["M_F_Lya"], k)

        if M_F_val is not None:
            M_str = fmt_math_value(M_F_val)
            if M_str is not None:
                lines.append(rf"$M_{{F,radial}} = {M_str}$")

    if ShowLyaLuminosityText and (L_bol_val is not None) and (L_Lya_val is not None):
        ratio = L_Lya_val / L_bol_val
        ratio_str = fmt_math_value(ratio)
        if ratio_str is not None:
            lines.append(rf"$L_{{Ly\alpha}} = {ratio_str}\,L_{{bol}}$")

    return lines

def make_ratio_diagnostics_lines(run, k):
    lines = []
    if ShowLyaEscapeFractionText:
        f_esc_val = safe_scalar_at_index(run["f_esc_Lya"], k)
        if f_esc_val is not None:
            fesc_str = fmt_math_value(f_esc_val)
            if fesc_str is not None:
                lines.append(rf"$f_{{esc,Ly\alpha}} = {fesc_str}$")
    return lines

def compute_PLya_over_P(run, k, eps=1e-30):
    """Compute P_Lyα/P = (e_Lyα/3) / P_gas at snapshot index k."""
    Pgas = run["P_hist"][k].T
    eLya = run["eLya_hist"][k].T
    PLya = eLya / 3.0
    return np.clip(PLya / np.maximum(Pgas, eps), eps, None)

# ============================================================
#   R E A D   A L L   R U N S
# ============================================================

nrows = len(ROWS)  

runs_noLya = []
runs_Lya   = []
k_noLya    = []
k_Lya      = []

for row_label, fname_no, fname_lya in ROWS:
    rno  = read_run(fname_no,  load_P=False, load_eLya=False)
    rlya = read_run(fname_lya, load_P=True,  load_eLya=True)

    # Verify matching grids
    if (rno["r"].shape != rlya["r"].shape) or (rno["Z"].shape != rlya["Z"].shape):
        raise RuntimeError(f"Grid shape mismatch for row '{row_label}'")
    if not (np.allclose(rno["r"], rlya["r"]) and np.allclose(rno["Z"], rlya["Z"])):
        raise RuntimeError(f"Grid value mismatch for row '{row_label}'")

    if rno["T_hist"] is None:
        raise RuntimeError(f"T_History not found in no-Lyα file for row '{row_label}'")
    if rlya["T_hist"] is None:
        raise RuntimeError(f"T_History not found in Lyα file for row '{row_label}'")
    if rlya["P_hist"] is None:
        raise RuntimeError(f"GasPressure_History not found in Lyα file for row '{row_label}'")
    if rlya["eLya_hist"] is None:
        raise RuntimeError(f"e_Lya_History not found in Lyα file for row '{row_label}'")

    runs_noLya.append(rno)
    runs_Lya.append(rlya)
    k_noLya.append(find_last_snapshot(rno))
    k_Lya.append(find_last_snapshot(rlya))

    print(f"Row '{row_label}':")
    print(f"  No-Lyα final snapshot: t = {rno['t'][k_noLya[-1]]/kyr:.3f} kyr")
    print(f"  Lyα    final snapshot: t = {rlya['t'][k_Lya[-1]]/kyr:.3f} kyr")

# Use grid from row 0 (all rows must share the same grid)
r = runs_noLya[0]["r"]
Z = runs_noLya[0]["Z"]
R_edges_pc, Z_edges_pc = np.meshgrid(r / pc, Z / pc, indexing='ij')

# ============================================================
#   C O M M O N   C O L O R   S C A L E S
# ============================================================

norm_T = LogNorm(vmin=T_vmin, vmax=T_vmax, clip=True)
ticks_T      = [1.0, 10.0, 1e2, 1e3, 1e4, 1e5]
ticklabels_T = [fmt_log_tick(v) for v in ticks_T]

norm_ratio = LogNorm(vmin=Ratio_vmin, vmax=Ratio_vmax, clip=True)
ticks_ratio      = [0.1, 1.0, 10.0]
ticklabels_ratio = [fmt_log_tick(v) for v in ticks_ratio]

# ============================================================
#   P R E P A R E   S N A P S H O T   D A T A
# ============================================================

eps = 1e-30

T_no  = [np.clip(runs_noLya[i]["T_hist"][k_noLya[i]].T, eps, None) for i in range(nrows)]
T_lya = [np.clip(runs_Lya[i]["T_hist"][k_Lya[i]].T,     eps, None) for i in range(nrows)]
Ratio = [compute_PLya_over_P(runs_Lya[i], k_Lya[i], eps=eps) for i in range(nrows)]

# ============================================================
#   F I G U R E   L A Y O U T
#
# nrows × 3 cols; row labels on the left margin.
#   Col 0: T, ignoring Lyα radiation pressure
#   Col 1: T, with Lyα radiation pressure
#   Col 2: P_Lyα/P, with Lyα radiation pressure
# ============================================================

left_margin  = 0.04   # fractional width reserved for row labels
wgap_big     = 0.02   # gap between no-Lyα and Lyα columns
wgap_small   = 0.000  # gap between Lyα T and P_Lyα/P columns

# Vertical layout controls:
top_margin    = 0.055  # room for column titles
bottom_margin = 0.020
hgap          = 0.012  # positive gap between rows

# Figure size
fig_width  = 24
fig_height = 8 * nrows
fig = plt.figure(figsize=(fig_width, fig_height))

# Panel width in figure coordinates
available_w = 1.0 - left_margin
wpanel = (available_w - wgap_big - wgap_small) / 3.0

# Panel height in figure coordinates.
# Use nrows, not hardwired 3, so adding/removing rows keeps the same proportions.
available_h = 1.0 - top_margin - bottom_margin
hpanel = (available_h - (nrows - 1) * hgap) / nrows

# Column left edges
col0_left = left_margin
col1_left = left_margin + wpanel + wgap_big
col2_left = left_margin + wpanel + wgap_big + wpanel + wgap_small

# Row bottom edges, row 0 is topmost
row_bottoms = [
    bottom_margin + (nrows - 1 - i) * (hpanel + hgap)
    for i in range(nrows)
]

fig_top = row_bottoms[0] + hpanel

def make_ax(fig, col_left, row_bottom, width, height, sharex=None, sharey=None):
    return fig.add_axes([col_left, row_bottom, width, height],
                        sharex=sharex, sharey=sharey)

# Build axes: axes[i] = (ax_T_no, ax_T_lya, ax_ratio) for row i
axes = []
ref_ax = None
for i in range(nrows):
    rb = row_bottoms[i]
    ax_no  = make_ax(fig, col0_left, rb, wpanel, hpanel,
                     sharex=ref_ax, sharey=ref_ax)
    ax_lya = make_ax(fig, col1_left, rb, wpanel, hpanel,
                     sharex=ref_ax, sharey=ref_ax)
    ax_rat = make_ax(fig, col2_left, rb, wpanel, hpanel,
                     sharex=ref_ax, sharey=ref_ax)
    if ref_ax is None:
        ref_ax = ax_no
    axes.append((ax_no, ax_lya, ax_rat))

# ============================================================
#   P L O T   D A T A
# ============================================================

ims_no  = []
ims_lya = []
ims_rat = []

for i in range(nrows):
    ax_no, ax_lya, ax_rat = axes[i]

    im_no = ax_no.pcolormesh(R_edges_pc, Z_edges_pc, T_no[i],
                             norm=norm_T, shading='auto', cmap=cmap_T, rasterized=True)
    im_lya = ax_lya.pcolormesh(R_edges_pc, Z_edges_pc, T_lya[i],
                               norm=norm_T, shading='auto', cmap=cmap_T, rasterized=True)
    im_rat = ax_rat.pcolormesh(R_edges_pc, Z_edges_pc, Ratio[i],
                               norm=norm_ratio, shading='auto', cmap=cmap_ratio, rasterized=True)

    ims_no.append(im_no)
    ims_lya.append(im_lya)
    ims_rat.append(im_rat)

# ============================================================
#   A X I S   S T Y L I N G   (cinematic — no ticks)
# ============================================================

all_axes = [ax for row in axes for ax in row]
for ax in all_axes:
    ax.set_xlim(r[0] / pc, xmax_fraction * r[-1] / pc)
    ax.set_ylim(Z[0] / pc, ymax_fraction * Z[-1] / pc)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(3.0)

# ============================================================
#   C O L U M N   T I T L E S   (top row only)
# ============================================================

title_style = dict(
    ha='center', va='bottom', color='black',
    fontsize=fontsize_title,
    path_effects=[pe.withStroke(linewidth=3, foreground='white')]
)

# "Ignoring Lyα" — centred over col 0
axes[0][0].text(0.5, 1.04, "Ignoring Ly\u03b1 Radiation Pressure",
                transform=axes[0][0].transAxes, **title_style)

# "With Lyα" — centred over cols 1–2
lya_title_x = 0.5 * (col1_left + col2_left + wpanel)
lya_title_y = fig_top + 0.75*0.010

fig.text(lya_title_x, lya_title_y,
         "With Ly\u03b1 Radiation Pressure", **title_style)

# ============================================================
#   R O W   L A B E L S   ( left margin, centred vertically )
# ============================================================

row_label_style = dict(
    ha='center', va='center', color='black',
    fontsize=fontsize_row_label,
    rotation=90,
)

for i, (row_label, _, _) in enumerate(ROWS):
    rb = row_bottoms[i]
    fig.text(left_margin * 0.5, rb + 0.5 * hpanel, row_label, **row_label_style)

# ============================================================
#   T E X T   O V E R L A Y S
# ============================================================

time_style_dark = dict(
    ha='left', va='top', color='black',
    path_effects=[pe.withStroke(linewidth=3, foreground='white')]
)

time_style_white = dict(
    ha='left', va='top', color='white',
    path_effects=[pe.withStroke(linewidth=6, foreground='black')]
)

diagnostics_style_white = dict(
    ha='left', va='top', color='white',
    path_effects=[pe.withStroke(linewidth=6, foreground='black')]
)

diagnostics_style_dark = dict(
    ha='left', va='top', color='black',
    path_effects=[pe.withStroke(linewidth=3, foreground='white')]
)

panel_label_style_black = dict(
    ha='right', va='bottom', color='black',
    path_effects=[pe.withStroke(linewidth=3.5, foreground='white')]
)

panel_label_style_white = dict(
    ha='right', va='bottom', color='white',
    path_effects=[pe.withStroke(linewidth=3.5, foreground='black')]
)

for i in range(nrows):
    ax_no, ax_lya, ax_rat = axes[i]
    t_no  = runs_noLya[i]["t"][k_noLya[i]] / kyr
    t_lya = runs_Lya[i]["t"][k_Lya[i]]     / kyr

    # Time stamps
    ax_no.text(0.04,  0.96, f"t = {t_no:.2f} kyr",  transform=ax_no.transAxes,
               **time_style_dark, fontsize=fontsize_time)
    ax_lya.text(0.04, 0.96, f"t = {t_lya:.2f} kyr", transform=ax_lya.transAxes,
                **time_style_dark, fontsize=fontsize_time)
    ax_rat.text(0.04, 0.96, f"t = {t_lya:.2f} kyr", transform=ax_rat.transAxes,
                **time_style_white, fontsize=fontsize_time)

    # Diagnostics text
    if ShowDiagnosticsText:
        lines_no  = make_radiation_diagnostics_lines(runs_noLya[i], k_noLya[i])
        lines_lya = make_radiation_diagnostics_lines(runs_Lya[i],   k_Lya[i])

        if lines_no:
            ax_no.text(diagnostics_x, diagnostics_y, "\n".join(lines_no),
                       transform=ax_no.transAxes, fontsize=fontsize_diagnostics_text,
                       **diagnostics_style_dark)
        if lines_lya:
            ax_lya.text(diagnostics_x, diagnostics_y, "\n".join(lines_lya),
                        transform=ax_lya.transAxes, fontsize=fontsize_diagnostics_text,
                        **diagnostics_style_dark)

        ratio_lines = make_ratio_diagnostics_lines(runs_Lya[i], k_Lya[i])
        if ratio_lines:
            ax_rat.text(diagnostics_x, diagnostics_y, "\n".join(ratio_lines),
                        transform=ax_rat.transAxes, fontsize=fontsize_diagnostics_text,
                        **diagnostics_style_white)

# Panel quantity labels — top row only (they share the same colorscale for all rows)
axes[0][0].text(0.94, 0.08, r"$T$ (K)",
                transform=axes[0][0].transAxes, fontsize=fontsize_panel_label,
                **panel_label_style_black)

axes[0][2].text(0.94, 0.08, r"$e_{Ly\alpha}/3P \approx P_{Ly\alpha}/P$",
                transform=axes[0][2].transAxes, fontsize=fontsize_panel_label,
                **panel_label_style_white)

# ============================================================
#   C O L O R B A R S   (inset, top row only)
# ============================================================

def make_inset_cbar(fig, ax, im, rect, ticks, ticklabels, dark=False):
    """
    Shared helper for inset horizontal colorbars.
    dark=True  -> black text/outline on white background
    dark=False -> white text/outline on black background
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

# Temperature colorbar — row 0, col 0
make_inset_cbar(fig, axes[0][0], ims_no[0],
                rect=[0.05, 0.8, 0.55, 0.045],
                ticks=ticks_T, ticklabels=ticklabels_T, dark=True)

# P_Lyα/P colorbar — row 0, col 2
make_inset_cbar(fig, axes[0][2], ims_rat[0],
                rect=[0.56, 0.91, 0.38, 0.045],
                ticks=ticks_ratio, ticklabels=ticklabels_ratio, dark=False)

# ============================================================
#   S A V E
# ============================================================

plt.savefig(FigureName, bbox_inches='tight', dpi=600)
print(f"Saved: {FigureName}")
plt.show()
