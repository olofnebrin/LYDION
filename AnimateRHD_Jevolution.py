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
outname = "J_EUV1.mp4"
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
    J_hist = f["J_EUV1_History"][:]

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
R, ZZ = np.meshgrid(r/pc, Z/pc, indexing='ij')

# display range (log)
vmin_J = 1e-4  # floor to keep LogNorm happy in vacuum
vmax_J = max(vmin_J, np.nanmax(J_hist))
norm_J = LogNorm(vmin=vmin_J, vmax=vmax_J)

# --- figure (single panel) ---
plt.rc('font', **{'sans-serif': ['AVHershey Complex'], 'size': 37})
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

panel_label_style = dict(
    ha='right', va='top', color='white',
    path_effects=[pe.withStroke(linewidth=3, foreground='black')]
)

ax.set_xlabel("r (pc)", labelpad=12)
ax.set_ylabel("Z (pc)", labelpad=12)
ax.set_xlim(r[0]/pc, r[-1]/pc)
ax.set_ylim(Z[0]/pc, Z[-1]/pc)

# initial image
im_J = ax.pcolormesh(R, ZZ,
                     np.clip(J_hist[frames[0]].T, vmin_J, vmax_J),
                     norm=norm_J, shading='auto', cmap='inferno')
cb = fig.colorbar(im_J, ax=ax, pad=0.02)
cb.set_label("J$_{EUV1}$")

# time label
time_style = dict(ha='left', va='top', color='white',
                  path_effects=[pe.withStroke(linewidth=3, foreground='black')])
txt = ax.text(0.02, 0.98, "", fontsize=28, transform=ax.transAxes, **time_style)

# --- animation functions ---
def init():
    k0 = frames[0]
    im_J.set_array(np.clip(J_hist[k0].T, vmin_J, vmax_J).ravel())
    txt.set_text(f"t = {t[k0]/kyr:.3f} kyr")
    return im_J, txt

def update(k):
    im_J.set_array(np.clip(J_hist[k].T, vmin_J, vmax_J).ravel())
    txt.set_text(f"t = {t[k]/kyr:.3f} kyr")
    return im_J, txt

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

if save_mp4:
    ani.save(outname, writer="ffmpeg", fps=fps, dpi=300)
else:
    plt.show()
