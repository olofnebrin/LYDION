import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
from pathlib import Path
import sys


###############################################################
#######                  U S E R  I N P U T             #######
###############################################################

# Files should be in the same directory as this script, unless full paths are given.
LYDION_FileName = "CloudTest_theta0.7853981633974483_rs0.5_tau1.0e8_T10000.0_umax0.0_tauabs0.0_pd0.0_256x256.h5"
COLT_FileName_xcrit0 = "Lya-cloud-xcrit0_1e7photons.hdf5"
COLT_FileName_xcrit3 = COLT_FileName_xcrit0.replace("xcrit0", "xcrit3")

outname = "Compare_LYDION_COLT_eLya_axis_profiles.png"
savefig = True

# Which LYDION snapshot to plot:
#   "final" -> final saved snapshot
#   integer -> explicit snapshot index
snapshot_to_plot = "final"

# Which COLT/MCRT field to compare with LYDION e_Lya.
COLT_eLya_dataset = "u_rad_r"

# Aaron's output has the full Z-domain because he did not use reflective BCs.
# Options: "upper", "lower", "auto", "full".
# For these 1D profiles, "upper" should match the half-plane LYDION output.
colt_half_to_plot = "upper"

# Optional normalization of COLT to LYDION.
# Options: "none", "match_total", "match_max".
colt_normalization = "none"
colt_scale_factor = 1.0

# Axis choices. 0 means the innermost/axis-adjacent cell.
ir_axis = 0       # near r = 0, for the Z-axis profile
jZ_midplane = 0   # near Z = 0, for the r-axis/midplane profile

# Plot limits. Set to None for automatic.
xlim_user = None
ylim_user = None

# Line colors specify the simulation; line style specifies direction.
colors = {
    "LYDION": "black",
    "MCRT, xcrit=0": "tab:blue",
    "MCRT, xcrit=3": "tab:orange",
}
linestyle_r_axis = "-"      # profile along r-axis, near Z = 0
linestyle_Z_axis = "--"     # profile along Z-axis, near r = 0
linewidth = 3.0

figsize = (9.0, 7.0)
dpi_save = 300


###############################################################
#######                O P T I O N A L  C M A S T R O  #######
###############################################################

# This is optional; the plot works without cmastro.  The upward search makes
# this script work both from the repo root and from a subdirectory.
def try_import_cmastro():
    try:
        import cmastro
        return cmastro
    except Exception:
        pass

    here = Path(__file__).resolve().parent
    for p in [here] + list(here.parents):
        cmastro_src = p / "src" / "cmastro" / "src"
        if cmastro_src.exists():
            sys.path.insert(0, str(cmastro_src))
            try:
                import cmastro
                return cmastro
            except Exception:
                return None
    return None

cmastro = try_import_cmastro()


###############################################################
#######                H E L P E R S                    #######
###############################################################

def script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def resolve_path(fname):
    if os.path.isabs(fname):
        return fname
    return os.path.join(script_dir(), fname)


def cell_centers(edges):
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def get_centers_from_file(f, edge_name, center_name):
    if center_name in f:
        return np.asarray(f[center_name][:], dtype=float)
    if edge_name in f:
        return cell_centers(f[edge_name][:])
    raise RuntimeError(f"Could not find {center_name!r} or {edge_name!r} in file.")


def reorder_history_3d(arr, t, r_edges, Z_edges):
    """
    Reorder a 3D LYDION history array to shape (Nt, Nr, Nz).
    """
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

    arr_tfirst = np.moveaxis(arr, time_axes[0], 0)
    _, a1, a2 = arr_tfirst.shape

    if (a1, a2) == (Nr, Nz):
        return arr_tfirst
    if (a1, a2) == (Nz, Nr):
        return arr_tfirst.transpose(0, 2, 1)

    raise RuntimeError(
        f"After moving time axis, spatial dims {(a1, a2)} do not match "
        f"Nr={Nr}, Nz={Nz}. Raw shape was {shape}."
    )


def read_lydion_file(fname, snapshot_to_plot):
    path = resolve_path(fname)

    with h5py.File(path, "r") as f:
        t = np.asarray(f["Times"][:], dtype=float)
        r_edges = np.asarray(f["r"][:], dtype=float)
        Z_edges = np.asarray(f["Z"][:], dtype=float)
        r_centers = get_centers_from_file(f, "r", "rCoM")
        Z_centers = get_centers_from_file(f, "Z", "ZCoM")
        e_raw = f["e_Lya_History"][:]

    e_hist = reorder_history_3d(e_raw, t, r_edges, Z_edges)

    if snapshot_to_plot == "final":
        positive_idx = np.where(np.isfinite(t) & (t > 0.0))[0]
        k = int(positive_idx[-1]) if positive_idx.size > 0 else 0
    else:
        k = int(snapshot_to_plot)

    if k < 0 or k >= len(t):
        raise RuntimeError(f"snapshot_to_plot={k} outside valid range [0, {len(t)-1}]")

    return {
        "path": path,
        "t": t,
        "snapshot_index": k,
        "r_edges": r_edges,
        "Z_edges": Z_edges,
        "r_centers": r_centers,
        "Z_centers": Z_centers,
        "e_Lya": np.asarray(e_hist[k, :, :], dtype=float),  # shape (Nr, Nz)
    }


def read_colt_field(fname, dataset_name):
    path = resolve_path(fname)
    with h5py.File(path, "r") as f:
        if dataset_name not in f:
            print("\nDatasets/groups in COLT file:")
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  Dataset: {name}, shape={obj.shape}")
                elif isinstance(obj, h5py.Group):
                    print(f"  Group:   {name}")
            f.visititems(visitor)
            raise RuntimeError(f"Dataset {dataset_name!r} not found in {path}")
        arr = f[dataset_name][:]
    return path, np.asarray(arr, dtype=float)


def reshape_colt_to_grid(colt_flat, Nr_lydion, Nz_lydion):
    """
    Aaron's output is expected to be flattened with shape (Nr, 2*Nz),
    because COLT has the full Z-domain while LYDION stores the half-plane.
    """
    arr = np.asarray(colt_flat, dtype=float)

    if arr.ndim == 2:
        return arr

    if arr.ndim != 1:
        raise RuntimeError(f"Expected 1D or 2D COLT field. Got shape {arr.shape}")

    N = arr.size
    expected = Nr_lydion * (2 * Nz_lydion)
    if N == expected:
        return arr.reshape((Nr_lydion, 2 * Nz_lydion))

    # Fallback to Aaron's n_z = 2*n_r assumption.
    n_r_float = np.sqrt(N / 2.0)
    n_r = int(round(n_r_float))
    if 2 * n_r * n_r == N:
        n_z = 2 * n_r
        print(
            f"WARNING: COLT size did not match LYDION Nr*(2*Nz) = {expected}, "
            f"but matches Aaron's n_z=2*n_r with n_r={n_r}, n_z={n_z}."
        )
        return arr.reshape((n_r, n_z))

    raise RuntimeError(
        f"Could not reshape COLT array of length {N}. Expected {expected}, "
        f"or 2*n_r^2 for integer n_r."
    )


def select_colt_half(colt_full, lydion_e, mode):
    Nr, Nz2 = colt_full.shape
    NrL, NzL = lydion_e.shape

    if mode == "full":
        return colt_full, "full"

    if Nr != NrL:
        raise RuntimeError(f"COLT Nr={Nr} does not match LYDION Nr={NrL}.")

    if Nz2 == NzL:
        print("WARNING: COLT Z size already matches LYDION Nz. No half-selection applied.")
        return colt_full, "already_half"

    if Nz2 != 2 * NzL:
        raise RuntimeError(f"COLT Nz={Nz2} is not equal to LYDION Nz={NzL} or 2*Nz={2*NzL}.")

    lower = colt_full[:, :NzL]
    upper = colt_full[:, NzL:]

    if mode == "lower":
        return lower, "lower"
    if mode == "upper":
        return upper, "upper"

    if mode == "auto":
        sum_L = np.nansum(np.clip(lydion_e, 0.0, np.inf))
        sum_lower = np.nansum(np.clip(lower, 0.0, np.inf))
        sum_upper = np.nansum(np.clip(upper, 0.0, np.inf))
        err_lower = abs(np.log10((sum_lower + 1e-300) / (sum_L + 1e-300)))
        err_upper = abs(np.log10((sum_upper + 1e-300) / (sum_L + 1e-300)))
        return (upper, "upper(auto)") if err_upper <= err_lower else (lower, "lower(auto)")

    raise ValueError("colt_half_to_plot must be 'upper', 'lower', 'auto', or 'full'.")


def apply_colt_normalization(colt_e, lydion_e, mode, manual_factor):
    colt_e = np.asarray(colt_e, dtype=float).copy()
    lydion_e = np.asarray(lydion_e, dtype=float)

    if mode == "none":
        scale = 1.0
    elif mode == "match_total":
        Lsum = np.nansum(np.clip(lydion_e, 0.0, np.inf))
        Csum = np.nansum(np.clip(colt_e, 0.0, np.inf))
        if Csum <= 0.0 or not np.isfinite(Csum):
            raise RuntimeError("Cannot use match_total: COLT total is non-positive/non-finite.")
        scale = Lsum / Csum
    elif mode == "match_max":
        Lmax = np.nanmax(lydion_e)
        Cmax = np.nanmax(colt_e)
        if Cmax <= 0.0 or not np.isfinite(Cmax):
            raise RuntimeError("Cannot use match_max: COLT max is non-positive/non-finite.")
        scale = Lmax / Cmax
    else:
        raise ValueError("colt_normalization must be 'none', 'match_total', or 'match_max'.")

    scale *= float(manual_factor)
    return scale * colt_e, scale


def read_colt_for_plot(fname, lydion_e, Nr, Nz):
    colt_path, colt_raw = read_colt_field(fname, COLT_eLya_dataset)
    colt_full = reshape_colt_to_grid(colt_raw, Nr, Nz)
    colt_plot, selected_half = select_colt_half(colt_full, lydion_e, colt_half_to_plot)
    colt_plot, total_colt_scale = apply_colt_normalization(
        colt_plot,
        lydion_e if colt_plot.shape == lydion_e.shape else np.nan * colt_plot,
        colt_normalization,
        colt_scale_factor,
    )

    if colt_half_to_plot != "full" and colt_plot.shape != lydion_e.shape:
        raise RuntimeError(
            f"After half-selection, COLT shape {colt_plot.shape} does not match LYDION shape {lydion_e.shape}."
        )

    return {
        "path": colt_path,
        "raw": colt_raw,
        "full": colt_full,
        "plot": colt_plot,  # shape (Nr, Nz), aligned with LYDION for profile extraction
        "selected_half": selected_half,
        "scale": total_colt_scale,
    }


def positive_y(y):
    y = np.asarray(y, dtype=float)
    return np.where(y > 0.0, y, np.nan)


###############################################################
#######                  R E A D  D A T A               #######
###############################################################

lyd = read_lydion_file(LYDION_FileName, snapshot_to_plot)
Nr, Nz = lyd["e_Lya"].shape

if ir_axis < 0 or ir_axis >= Nr:
    raise RuntimeError(f"ir_axis={ir_axis} outside valid range [0, {Nr-1}]")
if jZ_midplane < 0 or jZ_midplane >= Nz:
    raise RuntimeError(f"jZ_midplane={jZ_midplane} outside valid range [0, {Nz-1}]")

colt0 = read_colt_for_plot(COLT_FileName_xcrit0, lyd["e_Lya"], Nr, Nz)
colt3 = read_colt_for_plot(COLT_FileName_xcrit3, lyd["e_Lya"], Nr, Nz)

print("\n========== Loaded axis-profile comparison ==========")
print(f"LYDION file:       {lyd['path']}")
print(f"COLT xcrit0 file:  {colt0['path']}")
print(f"COLT xcrit3 file:  {colt3['path']}")
print(f"LYDION e_Lya shape: {lyd['e_Lya'].shape}")
print(f"COLT xcrit0 plotted shape: {colt0['plot'].shape}, half={colt0['selected_half']}")
print(f"COLT xcrit3 plotted shape: {colt3['plot'].shape}, half={colt3['selected_half']}")
print(f"Profile along Z-axis uses ir_axis={ir_axis}, r={lyd['r_centers'][ir_axis]:.6e}")
print(f"Profile along r-axis uses jZ_midplane={jZ_midplane}, Z={lyd['Z_centers'][jZ_midplane]:.6e}")
print(f"LYDION snapshot: k={lyd['snapshot_index']}, t={lyd['t'][lyd['snapshot_index']]:.6e}")
print("====================================================\n")


###############################################################
#######                    P L O T                     #######
###############################################################

plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 24})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['xtick.major.width'] = 2.2
plt.rcParams['xtick.minor.width'] = 2.2
plt.rcParams['ytick.major.width'] = 2.2
plt.rcParams['ytick.minor.width'] = 2.2
plt.rcParams['xtick.major.size'] = 13
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['ytick.major.size'] = 13
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['legend.frameon'] = False

fig, ax = plt.subplots(figsize=figsize)

runs = [
    ("LYDION", lyd["e_Lya"]),
    ("MCRT, xcrit=0", colt0["plot"]),
    ("MCRT, xcrit=3", colt3["plot"]),
]

for label, field in runs:
    c = colors.get(label, None)

    # Along r-axis / midplane: vary r at the first Z cell near Z = 0.
    ax.plot(
        lyd["r_centers"],
        positive_y(field[:, jZ_midplane]),
        color=c,
        linestyle=linestyle_r_axis,
        linewidth=linewidth,
        label=rf"{label}, $r$-axis",
    )

    # Along Z-axis: vary Z at the innermost r cell near r = 0.
    ax.plot(
        lyd["Z_centers"],
        positive_y(field[ir_axis, :]),
        color=c,
        linestyle=linestyle_Z_axis,
        linewidth=linewidth,
        label=rf"{label}, $Z$-axis",
    )

ax.set_yscale("log")
ax.set_xlabel(r"distance from center / $R_{cl}$", labelpad=10)
ax.set_ylabel(r"$e_{Ly\alpha}$", labelpad=12)

if xlim_user is not None:
    ax.set_xlim(*xlim_user)
else:
    xmax = max(np.nanmax(lyd["r_centers"]), np.nanmax(lyd["Z_centers"]))
    ax.set_xlim(0.0, xmax)

if ylim_user is not None:
    ax.set_ylim(*ylim_user)

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_major_locator(LogLocator(base=10))
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax.yaxis.set_minor_formatter(NullFormatter())

# Keep one legend; solid/dashed distinguish direction, colors distinguish run.
ax.legend(fontsize=15, loc="best", handlelength=3.2)

fig.tight_layout()

if savefig:
    outpath = resolve_path(outname)
    fig.savefig(outpath, dpi=dpi_save, bbox_inches="tight")
    print(f"Saved figure: {outpath}")
else:
    plt.show()
