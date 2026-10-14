"""Simple star-formation criteria helpers for LYDION analysis.

The main implemented criterion is the FIRE-like local self-gravity
criterion alpha < 1, ignoring magnetic support:

    alpha = ( |grad v|^2 + 2 (k c_s)^2 ) / (8 pi G rho),
    k     = 1 / dx,
    c_s   = sqrt(gamma * P / rho),
    dx    = min(delta_r, delta_Z).

All arrays are assumed to use LYDION physical order (Nr, NZ), i.e.
field[i, j] with i = r and j = Z.
"""

import numpy as np


G  = 6.67430e-8       # Gravitational constant (cgs units)
mH = 1.6735575e-24    # Hydrogen atom mass (grams)


def velocity_from_momentum(mom_r, mom_Z, rho):
    """Return gas velocities from momentum densities."""

    v_r = mom_r / rho
    v_Z = mom_Z / rho

    return v_r, v_Z


def sound_speed(P, rho, gamma=5.0/3.0):
    """Adiabatic sound speed: c_s = sqrt(gamma P / rho)."""

    return np.sqrt(gamma * P / rho)


def cell_width(delta_r, delta_Z):
    """LYDION star-formation cell width estimate: dx = min(delta_r, delta_Z)."""

    return np.minimum(delta_r, delta_Z)


def velocity_gradient_norm_squared(v_r, v_Z, r, Z):
    """Return |grad v|^2 on the 2D axisymmetric (r,Z) grid.

    Includes the cylindrical term (v_r/r)^2. On the axis, v_r/r is
    replaced by d(v_r)/dr.
    """

    dv_r_dr = np.gradient(v_r, r, axis=0, edge_order=2)
    dv_r_dZ = np.gradient(v_r, Z, axis=1, edge_order=2)
    dv_Z_dr = np.gradient(v_Z, r, axis=0, edge_order=2)
    dv_Z_dZ = np.gradient(v_Z, Z, axis=1, edge_order=2)

    r2D = r[:, None]
    vr_over_r = v_r / r2D

    axis_cells = (r2D == 0.0)
    vr_over_r = np.where(axis_cells, dv_r_dr, vr_over_r)

    return (
        dv_r_dr**2 + dv_r_dZ**2
      + dv_Z_dr**2 + dv_Z_dZ**2
      + vr_over_r**2
    )

def velocity_divergence_axisymmetric(vr, vZ, rCoM, ZCoM):
    """Compute div(v) in 2D axisymmetric cylindrical geometry.

        div(v) = dvr/dr + vr/r + dvZ/dZ

    At the axis, vr/r is replaced by dvr/dr.
    """

    dvr_dr = np.gradient(vr, rCoM, axis=0, edge_order=2)
    dvZ_dZ = np.gradient(vZ, ZCoM, axis=1, edge_order=2)

    r2d = rCoM[:, None]

    vr_over_r = vr / r2d
    vr_over_r[0, :] = dvr_dr[0, :]

    div_v = dvr_dr + vr_over_r + dvZ_dZ

    return div_v


def converging_flow_mask(vr, vZ, rCoM, ZCoM):
    """Return mask for locally converging flow."""

    div_v = velocity_divergence_axisymmetric(vr, vZ, rCoM, ZCoM)

    return div_v < 0.0

def virial_parameter_self_gravity(rho, P, v_r, v_Z, r, Z, delta_r, delta_Z,
                                  gamma=5.0/3.0):
    """Compute the FIRE-like local virial parameter alpha.

    alpha < 1 means the cell is locally self-gravitating.
    """

    cs = sound_speed(P, rho, gamma=gamma)
    dx = cell_width(delta_r, delta_Z)
    k  = 1.0 / dx

    gradv2 = velocity_gradient_norm_squared(v_r, v_Z, r, Z)

    return (gradv2 + 2.0 * (k * cs)**2) / (8.0 * np.pi * G * rho)


def self_gravitating_mask(rho, P, v_r, v_Z, r, Z, delta_r, delta_Z,
                          gamma=5.0/3.0, alpha_crit=1.0):
    """Return boolean mask for locally self-gravitating cells."""

    alpha = virial_parameter_self_gravity(
        rho, P, v_r, v_Z, r, Z, delta_r, delta_Z, gamma=gamma
    )

    return alpha < alpha_crit


def simple_star_forming_mask(rho, P, v_r, v_Z, r, Z, delta_r, delta_Z,
                             gamma=5.0/3.0,
                             alpha_crit=1.0,
                             nH=None,
                             nH_crit=None,
                             xHII=None,
                             xHII_crit=None,
                             T=None,
                             T_crit=None):
    """Return star-forming gas mask.

    The self-gravity criterion is always applied. Optional cuts are applied
    only when both the field and critical value are given.
    """

    mask = self_gravitating_mask(
        rho, P, v_r, v_Z, r, Z, delta_r, delta_Z,
        gamma=gamma,
        alpha_crit=alpha_crit,
    )

    if nH is not None and nH_crit is not None:
        mask &= (nH > nH_crit)

    if xHII is not None and xHII_crit is not None:
        mask &= (xHII < xHII_crit)

    if T is not None and T_crit is not None:
        mask &= (T < T_crit)

    return mask


def selected_gas_mass(rho, cell_volume, mask, symmetry_factor=2.0):
    """Return selected gas mass in grams.

    Use symmetry_factor=2 for current LYDION half-plane saved volumes, if the
    plotting script uses 2 * sum(rho * CellVol). Use 1 otherwise.
    """

    return symmetry_factor * np.sum(rho[mask] * cell_volume[mask])
