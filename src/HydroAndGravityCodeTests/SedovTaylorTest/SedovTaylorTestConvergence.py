import os
import h5py
import numpy as np

# ---------------- User inputs ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))

file_128 = os.path.join(script_dir, "SedovTest_E1.00e+00_rho1_gamma1.67_t0.4_128x128.h5")
file_256 = os.path.join(script_dir, "SedovTest_E1.00e+00_rho1_gamma1.67_t0.4_256x256.h5")
file_512 = os.path.join(script_dir, "SedovTest_E1.00e+00_rho1_gamma1.67_t0.4_512x512.h5")

analytic_file = os.path.join(script_dir, "SedovTaylorAnalytical_E1.0_rho1.0_t0.4_gamma53.txt")
# ---------------------------------------------

# --- Load analytic 1D profile ---
ana_data = np.loadtxt(analytic_file)
r_ana    = ana_data[:, 0]
rho_ana  = ana_data[:, 1]
P_ana    = ana_data[:, 3]   # adjust if needed

def compute_global_errors(h5name):
    """Compute L1 and L2 errors for one run using 1/N normalization."""
    with h5py.File(h5name, "r") as f:
        T        = f["Times"][:]
        RhoHist  = f["GasDensity_History"][:]
        PHist    = f["P_History"][:]
        RCoM     = f["rCoM"][:]
        ZCoM     = f["ZCoM"][:]

    # Use last snapshot (t = 0.4)
    it = -1
    t_used = T[it]
    print(f"{h5name}: using final snapshot at t = {t_used:.5f}")

    def ensure_NtNrNz(arr):
        if arr.shape[2] == len(T):
            return arr.transpose(2, 0, 1)
        elif arr.shape[0] == len(T):
            return arr
        else:
            raise ValueError(f"Unexpected array shape {arr.shape} vs len(T)={len(T)}")

    RhoHist = ensure_NtNrNz(RhoHist)
    PHist   = ensure_NtNrNz(PHist)

    rho_num = RhoHist[it, :, :]
    P_num   = PHist[it, :, :]

    R_grid, Z_grid = np.meshgrid(RCoM, ZCoM, indexing="ij")
    r_sph = np.sqrt(R_grid**2 + Z_grid**2)

    r_min, r_max = r_ana.min(), r_ana.max()
    mask = (r_sph >= r_min) & (r_sph <= r_max)

    r_flat       = r_sph[mask]
    rho_num_flat = rho_num[mask]
    P_num_flat   = P_num[mask]

    rho_ana_flat = np.interp(r_flat, r_ana, rho_ana)
    P_ana_flat   = np.interp(r_flat, r_ana, P_ana)

    N = len(r_flat)

    def L1_norm(num, ana):
        diff = np.abs(num - ana)
        return np.sum(diff) / N

    def L2_norm(num, ana):
        diff = num - ana
        return np.sqrt(np.sum(diff**2) / N)

    E_L1_rho = L1_norm(rho_num_flat, rho_ana_flat)
    E_L2_rho = L2_norm(rho_num_flat, rho_ana_flat)
    E_L1_P   = L1_norm(P_num_flat,   P_ana_flat)
    E_L2_P   = L2_norm(P_num_flat,   P_ana_flat)

    return E_L1_rho, E_L2_rho, E_L1_P, E_L2_P

# --- Compute errors for 128, 256, 512 ---
E128_L1_rho, E128_L2_rho, E128_L1_P, E128_L2_P = compute_global_errors(file_128)
E256_L1_rho, E256_L2_rho, E256_L1_P, E256_L2_P = compute_global_errors(file_256)
E512_L1_rho, E512_L2_rho, E512_L1_P, E512_L2_P = compute_global_errors(file_512)

print("\nAbsolute errors (L1, L2) with 1/N normalization:")
print(f"  128^2:  L1(rho) = {E128_L1_rho:.3e},  L2(rho) = {E128_L2_rho:.3e}")
print(f"          L1(P)   = {E128_L1_P:.3e},    L2(P)   = {E128_L2_P:.3e}")
print(f"  256^2:  L1(rho) = {E256_L1_rho:.3e},  L2(rho) = {E256_L2_rho:.3e}")
print(f"          L1(P)   = {E256_L1_P:.3e},    L2(P)   = {E256_L2_P:.3e}")
print(f"  512^2:  L1(rho) = {E512_L1_rho:.3e},  L2(rho) = {E512_L2_rho:.3e}")
print(f"          L1(P)   = {E512_L1_P:.3e},    L2(P)   = {E512_L2_P:.3e}")

# --- Helper for order given spacing ratio r = Δx_coarse / Δx_fine ---
def order(E_coarse, E_fine, ratio):
    return np.log(E_coarse / E_fine) / np.log(ratio)

# Linear resolution ratios:
# 128 -> 256: Δx ratio = 2
# 256 -> 512: Δx ratio = 2
# 128 -> 512: Δx ratio = 4
r_128_256 = 2.0
r_256_512 = 2.0
r_128_512 = 4.0

# Example: order between 128 and 512 directly
p_L1_rho_128_512 = order(E128_L1_rho, E512_L1_rho, r_128_512)
p_L2_rho_128_512 = order(E128_L2_rho, E512_L2_rho, r_128_512)
p_L1_P_128_512   = order(E128_L1_P,   E512_L1_P,   r_128_512)
p_L2_P_128_512   = order(E128_L2_P,   E512_L2_P,   r_128_512)

print("\nObserved convergence orders:")
print("  Between 128 -> 256 (Δx ratio = 2):")
print(f"    rho: L1 p ≈ {order(E128_L1_rho, E256_L1_rho, r_128_256):.3f}, "
      f"L2 p ≈ {order(E128_L2_rho, E256_L2_rho, r_128_256):.3f}")
print(f"     P : L1 p ≈ {order(E128_L1_P,   E256_L1_P,   r_128_256):.3f}, "
      f"L2 p ≈ {order(E128_L2_P,   E256_L2_P,   r_128_256):.3f}")

print("\n  Between 256 -> 512 (Δx ratio = 2):")
print(f"    rho: L1 p ≈ {order(E256_L1_rho, E512_L1_rho, r_256_512):.3f}, "
      f"L2 p ≈ {order(E256_L2_rho, E512_L2_rho, r_256_512):.3f}")
print(f"     P : L1 p ≈ {order(E256_L1_P,   E512_L1_P,   r_256_512):.3f}, "
      f"L2 p ≈ {order(E256_L2_P,   E512_L2_P,   r_256_512):.3f}")

print("\n  Between 128 -> 512 (Δx ratio = 4):")
print(f"    rho: L1 p ≈ {p_L1_rho_128_512:.3f},  L2 p ≈ {p_L2_rho_128_512:.3f}")
print(f"     P : L1 p ≈ {p_L1_P_128_512:.3f},    L2 p ≈ {p_L2_P_128_512:.3f}")
