import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe

# ---------------- user inputs ----------------
filename = "RHDSim.h5"        # <-- change to your file
frameskip = 1                 # use >1 to skip frames
save_mp4 = True               # True to save mp4 instead of interactive show
outname = "J_asym.mp4"
fps = 20
# ---------------------------------------------

# constants
yr  = 3.1556926e7
kyr = 1e3 * yr
pc  = 3.085677581e18

# --- read data ---
with h5py.File(filename, "r") as f:
    t      = f["Times"][:]                          # (Nt,)
    r      = f["r"][:]                              # edges, (Nr+1,)
    Z      = f["Z"][:]                              # edges, (Nz+1,)
    J_hist = f["J_EUV1_History"][:]                 # (Nt, a1, a2)

# --- ensure (Nt, Nr, Nz) ordering ---
Nr = len(r) - 1
Nz = len(Z) - 1
Nt_raw, a1, a2 = J_hist.shape

def ensure_NtNrNz(arr, a1, a2):
    if (a1, a2) == (Nr, Nz):
        return arr
    elif (a1, a2) == (Nz, Nr):
        return arr.transpose(0, 2, 1)
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} vs edges Nr={Nr}, Nz={Nz}")

J_hist = ensure_NtNrNz(J_hist, a1, a2)
Nt, _, _ = J_hist.shape

# indices with valid saved data (nonzero or t=0)
saved_idx = np.where((t > 0) | (np.arange(t.size) == 0))[0]
frames = saved_idx[::frameskip]

# grid for pcolormesh (edges, in pc)
R_edges, Z_edges = r/pc, Z/pc
R, ZZ = np.meshgrid(R_edges, Z_edges, indexing='ij')

# --- helper: nearest-neighbor index mapping ---
def nearest_index(src_centers, target_centers):
    idx = np.searchsorted(src_centers, target_centers)
    idx = np.clip(idx, 1, len(src_centers)-1)
    left  = src_centers[idx-1]
    right = src_centers[idx]
    use_right = (np.abs(right - target_centers) < np.abs(target_centers - left))
    idx = idx + use_right.astype(int) - 1
    return idx

Rc = 0.5*(r[:-1] + r[1:])
Zc = 0.5*(Z[:-1] + Z[1:])

use_exact_transpose = (Nr == Nz) and np.allclose(Rc, Zc, rtol=0, atol=1e-12)

if not use_exact_transpose:
    i_from_j = nearest_index(Rc, Zc)
    j_from_i = nearest_index(Zc, Rc)

def asymmetry_frame(Jk):
    if use_exact_transpose:
        Jswap = Jk.T
    else:
        Jswap = Jk[i_from_j[:, None], j_from_i[None, :]].T
    return np.abs(Jk - Jswap)

# --- display range for |J - J^T| ---
vmin_A, vmax_A = 1e-4, 1e4
norm_A = LogNorm(vmin=vmin_A, vmax=vmax_A)

# --- figure (single panel) ---
plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 37})
plt.rcParams.update({
    'axes.unicode_minus': False,
    'axes.linewidth': 2.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True
})

fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

ax.set_xlabel("r (pc)", labelpad=12)
ax.set_ylabel("Z (pc)", labelpad=12)
ax.set_xlim(R_edges[0], R_edges[-1])
ax.set_ylim(Z_edges[0], Z_edges[-1])

# initial frame
A0 = asymmetry_frame(J_hist[frames[0]])
im = ax.pcolormesh(R, ZZ, np.clip(A0.T, vmin_A, vmax_A),
                   norm=norm_A, shading='auto', cmap='inferno')

cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label(r"$|J - J^{T}|$")
cb.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4])

# time label
time_style = dict(ha='left', va='top', color='white',
                  path_effects=[pe.withStroke(linewidth=3, foreground='black')])
txt = ax.text(0.02, 0.98, "", fontsize=28, transform=ax.transAxes, **time_style)

# --- animation functions ---
def init():
    im.set_array(np.clip(A0.T, vmin_A, vmax_A).ravel())
    txt.set_text(f"t = {t[frames[0]]/kyr:.3f} kyr")
    return im, txt

def update(k):
    A = np.clip(asymmetry_frame(J_hist[k]).T, vmin_A, vmax_A)
    im.set_array(A.ravel())
    txt.set_text(f"t = {t[k]/kyr:.3f} kyr")
    return im, txt

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

if save_mp4:
    ani.save(outname, writer="ffmpeg", fps=fps, dpi=300)
else:
    plt.show()
