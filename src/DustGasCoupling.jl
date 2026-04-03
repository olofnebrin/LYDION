
""" FILE:       DustGasCoupling.jl

    PURPOSE:    Contains the code for computing the update in the dust and gas momenta from 
                radiation pressure & drag forces.

"""

const urel_max::Float64         = 100 * kms    # Maximum relative velocity of dust (fast-moving dust is destroyed anyway)
const urel_max_squared::Float64 = urel_max * urel_max 

""" This function caps the relative velocities, if the candidate updated 
    relative velocity exceeds the maximum allowed value. This is done by 
    scaling down the components accordingly, thus preserving directionality. """
@inline function CapRelativeVelocities( urel_r_new_candidate::Float64,
                                        urel_Z_new_candidate::Float64,
                                                          ρd::Float64 )

    urel_candidate_sq = @fastmath ( urel_r_new_candidate * urel_r_new_candidate  
                                  + urel_Z_new_candidate * urel_Z_new_candidate )

    if urel_candidate_sq > urel_max_squared * (ρd > ρd_IGNORE)
        # If relative velocities exceed cap, scale down the magnitude. Also, if the 
        # dust density is near the floor, set the relative velocity to zero:
        urel_new        = @fastmath sqrt( urel_candidate_sq )
        ReductionFactor = @fastmath (ρd > ρd_IGNORE) * urel_max / urel_new
        urel_r_new      = @fastmath urel_r_new_candidate * ReductionFactor
        urel_Z_new      = @fastmath urel_Z_new_candidate * ReductionFactor
    else 
        urel_r_new, urel_Z_new = urel_r_new_candidate, urel_Z_new_candidate
    end

    return urel_r_new, urel_Z_new 
end


function UpdateGasDustMomenta( Δt::Float64, # Time-step 
                                ρ::Float64, # Gas density 
                           ρd_Sil::Float64, # Silicate dust density 
                             ρd_C::Float64, # Graphite dust density 
                           ρd_PAH::Float64, # PAH dust density 

                        # Momentum from radiation pressure on gas:
                        Δρu_r_gas::Float64,Δρu_Z_gas::Float64,

                        # Momentum from radiation pressure on silicate grains:
                        Δρu_r_Sil::Float64, Δρu_Z_Sil::Float64,

                        # Momentum from radiation pressure on graphite grains:
                          Δρu_r_C::Float64, Δρu_Z_C::Float64,

                        # Momentum from radiation pressure on PAHs:
                        Δρu_r_PAH::Float64, Δρu_Z_PAH::Float64,

                        # Old gas momentum:
                             ρu_r::Float64, ρu_Z::Float64, 

                        # Old silicate dust momentum:
                       ρdud_r_Sil::Float64, ρdud_Z_Sil::Float64,

                       # Old graphite dust momentum:
                         ρdud_r_C::Float64, ρdud_Z_C::Float64,

                       # Old PAH dust momentum:
                       ρdud_r_PAH::Float64, ρdud_Z_PAH::Float64,

                       # Drag coefficients for each dust type:
                       K_Sil::Float64, K_C::Float64, K_PAH::Float64
                              )

    # Convenient constants, precompute:
    ρtot       = @fastmath ρ + ρd_Sil + ρd_C + ρd_PAH
    inv_ρ      = @fastmath 1 / ρ
    inv_ρd_Sil = @fastmath 1 / ρd_Sil
    inv_ρd_C   = @fastmath 1 / ρd_C
    inv_ρd_PAH = @fastmath 1 / ρd_PAH
    inv_ρtot   = @fastmath 1 / ρtot
    Δt_inv_ρ   = @fastmath Δt * inv_ρ

    # Get old velocities from momentum densities:
    u_r = @fastmath ρu_r * inv_ρ;  u_Z = @fastmath ρu_Z * inv_ρ

    ud_r_Sil = @fastmath ρdud_r_Sil * inv_ρd_Sil;  ud_Z_Sil = ρdud_Z_Sil * inv_ρd_Sil
    ud_r_C   = @fastmath ρdud_r_C   * inv_ρd_C;    ud_Z_C   = ρdud_Z_C   * inv_ρd_C
    ud_r_PAH = @fastmath ρdud_r_PAH * inv_ρd_PAH;  ud_Z_PAH = ρdud_Z_PAH * inv_ρd_PAH

    # ---- STEP 1: Update center-of-mass velocity from total radiation pressure ------

    # Old CoM velocity:
    uCoM_r     = @fastmath ( ρu_r + ρdud_r_Sil + ρdud_r_C + ρdud_r_PAH ) * inv_ρtot
    uCoM_Z     = @fastmath ( ρu_Z + ρdud_Z_Sil + ρdud_Z_C + ρdud_Z_PAH ) * inv_ρtot

    # Updated/new CoM velocities, from radiation pressure (drag terms cancel):
    uCoM_r_new = @fastmath uCoM_r + ( Δρu_r_gas + Δρu_r_Sil + Δρu_r_C + Δρu_r_PAH ) * inv_ρtot
    uCoM_Z_new = @fastmath uCoM_Z + ( Δρu_Z_gas + Δρu_Z_Sil + Δρu_Z_C + Δρu_Z_PAH ) * inv_ρtot

    # ----- STEP 2: Solve the matrix equation M (u_rel)^(••) = A for the updated relative velocities ----
    
    # Old relative velocities:
    urel_r_Sil = @fastmath ud_r_Sil - u_r;   urel_Z_Sil = @fastmath ud_Z_Sil - u_Z
    urel_r_C   = @fastmath ud_r_C   - u_r;   urel_Z_C   = @fastmath ud_Z_C   - u_Z
    urel_r_PAH = @fastmath ud_r_PAH - u_r;   urel_Z_PAH = @fastmath ud_Z_PAH - u_Z

    # Construct the right-hand side vector A_i = (u_rel)^(•)_i  +  [ (f_d,i)/ρd_i - f/ρ ] Δt,
    # further noting that e.g. f Δt = Δρu:
    A_r_Sil = @fastmath urel_r_Sil  +  ( Δρu_r_Sil * inv_ρd_Sil  -  Δρu_r_gas * inv_ρ ) 
    A_r_C   = @fastmath urel_r_C    +  ( Δρu_r_C   * inv_ρd_C    -  Δρu_r_gas * inv_ρ ) 
    A_r_PAH = @fastmath urel_r_PAH  +  ( Δρu_r_PAH * inv_ρd_PAH  -  Δρu_r_gas * inv_ρ ) 

    A_Z_Sil = @fastmath urel_Z_Sil  +  ( Δρu_Z_Sil * inv_ρd_Sil  -  Δρu_Z_gas * inv_ρ ) 
    A_Z_C   = @fastmath urel_Z_C    +  ( Δρu_Z_C   * inv_ρd_C    -  Δρu_Z_gas * inv_ρ ) 
    A_Z_PAH = @fastmath urel_Z_PAH  +  ( Δρu_Z_PAH * inv_ρd_PAH  -  Δρu_Z_gas * inv_ρ )

    A_r     = @SVector [ A_r_Sil, A_r_C, A_r_PAH ]
    A_Z     = @SVector [ A_Z_Sil, A_Z_C, A_Z_PAH ]

    # Construct the 3 × 3 matrix M, which is the same for r and Z-directions, and has the components:
    # M_ij = δ_ij [1 + Δt * K_i * (1/ρ_d,i + 1/ρ)]  +  (1 - δ_ij) Δt K_j / ρ

    M11 = @fastmath  1  +  Δt * K_Sil * ( inv_ρd_Sil + inv_ρ )
    M12 = @fastmath  K_C   * Δt_inv_ρ
    M13 = @fastmath  K_PAH * Δt_inv_ρ

    M21 = @fastmath  K_Sil * Δt_inv_ρ
    M22 = @fastmath  1  +  Δt * K_C * ( inv_ρd_C + inv_ρ )
    M23 = M13   # Identical to M13 

    M31 = M21   # Identical to M21  
    M32 = M12   # Identical to M12 
    M33 = @fastmath  1  +  Δt * K_PAH * ( inv_ρd_PAH + inv_ρ )

    M   = @SMatrix [ M11  M12  M13 ;
                     M21  M22  M23 ;
                     M31  M32  M33 ]

    # Solve for updated relative velocities:
    urel_r_new = M \ A_r 
    urel_Z_new = M \ A_Z 

    # Extract the bin-specific relative velocities:
    urel_r_Sil_new = @inbounds urel_r_new[1]
    urel_r_C_new   = @inbounds urel_r_new[2]
    urel_r_PAH_new = @inbounds urel_r_new[3]

    urel_Z_Sil_new = @inbounds urel_Z_new[1]
    urel_Z_C_new   = @inbounds urel_Z_new[2]
    urel_Z_PAH_new = @inbounds urel_Z_new[3]

    # Cap relative velocities to ≤ urel_max:
    urel_r_Sil_new, urel_Z_Sil_new = CapRelativeVelocities( urel_r_Sil_new, urel_Z_Sil_new, ρd_Sil )
    urel_r_C_new,   urel_Z_C_new   = CapRelativeVelocities( urel_r_C_new,   urel_Z_C_new,   ρd_C )
    urel_r_PAH_new, urel_Z_PAH_new = CapRelativeVelocities( urel_r_PAH_new, urel_Z_PAH_new, ρd_PAH )

    # ----- STEP 3: Recover velocities from uCoM_new and urel_new ------
    # ud_i = uCoM  +  (ρ / ρtot) urel_i
    # u    = uCoM  -  (∑_i ρd_i / ρtot) urel_i   (from momentum conservation)

    frac_gas = @fastmath ρ      * inv_ρtot
    frac_Sil = @fastmath ρd_Sil * inv_ρtot
    frac_C   = @fastmath ρd_C   * inv_ρtot
    frac_PAH = @fastmath ρd_PAH * inv_ρtot

    # Updated dust velocities & momenta:
    ud_r_Sil_new = @fastmath uCoM_r_new  +  frac_gas * urel_r_Sil_new
    ud_r_C_new   = @fastmath uCoM_r_new  +  frac_gas * urel_r_C_new
    ud_r_PAH_new = @fastmath uCoM_r_new  +  frac_gas * urel_r_PAH_new

    ud_Z_Sil_new = @fastmath uCoM_Z_new  +  frac_gas * urel_Z_Sil_new
    ud_Z_C_new   = @fastmath uCoM_Z_new  +  frac_gas * urel_Z_C_new
    ud_Z_PAH_new = @fastmath uCoM_Z_new  +  frac_gas * urel_Z_PAH_new

    ρdud_r_Sil_new = @fastmath ρd_Sil * ud_r_Sil_new
    ρdud_r_C_new   = @fastmath ρd_C   * ud_r_C_new
    ρdud_r_PAH_new = @fastmath ρd_PAH * ud_r_PAH_new

    ρdud_Z_Sil_new = @fastmath ρd_Sil * ud_Z_Sil_new
    ρdud_Z_C_new   = @fastmath ρd_C   * ud_Z_C_new
    ρdud_Z_PAH_new = @fastmath ρd_PAH * ud_Z_PAH_new

    # Updated gas velocity & momenta:
    u_r_new = @fastmath uCoM_r_new - ( frac_Sil * urel_r_Sil_new  +  frac_C * urel_r_C_new  +  frac_PAH * urel_r_PAH_new )
    u_Z_new = @fastmath uCoM_Z_new - ( frac_Sil * urel_Z_Sil_new  +  frac_C * urel_Z_C_new  +  frac_PAH * urel_Z_PAH_new )

    ρu_r_new = @fastmath ρ * u_r_new
    ρu_Z_new = @fastmath ρ * u_Z_new

    # Return updated momentum densities & velocities:
    return ρu_r_new, ρu_Z_new,  
           ρdud_r_Sil_new, ρdud_Z_Sil_new,
           ρdud_r_C_new,   ρdud_Z_C_new,
           ρdud_r_PAH_new, ρdud_Z_PAH_new,

           u_r_new, u_Z_new,
           ud_r_Sil_new, ud_Z_Sil_new,
           ud_r_C_new,   ud_Z_C_new,
           ud_r_PAH_new, ud_Z_PAH_new
end


""" A small function for checking convergence in the Picard iterations of 
    the gas-dust radiation pressure + drag update, during the subcycling.  """
@inline function CheckConvergence_GasDustMomentumUpdate( # Candidate new estimate of 'new'/updated values:
                                                         ud_r_Sil_candidate::Float64, ud_Z_Sil_candidate::Float64,
                                                           ud_r_C_candidate::Float64, ud_Z_C_candidate::Float64,
                                                         ud_r_PAH_candidate::Float64, ud_Z_PAH_candidate::Float64,
                                                              u_r_candidate::Float64, u_Z_candidate::Float64, 

                                                          # Previous estimated 'new'/updated values:
                                                               ud_r_Sil_new::Float64, ud_Z_Sil_new::Float64,
                                                                 ud_r_C_new::Float64, ud_Z_C_new::Float64,
                                                               ud_r_PAH_new::Float64, ud_Z_PAH_new::Float64,
                                                                    u_r_new::Float64, u_Z_new::Float64,  
                                                                    
                                                          # Treshold below which we ignore changes:
                                                        tol_DustGasCoupling::Float64 = 1e-4,
                                                                uth_DustGas::Float64 = 0.01 * kms
                                                          )

    Δud_r_Sil = abs( ud_r_Sil_candidate - ud_r_Sil_new ) / ( abs(ud_r_Sil_new) +  uth_DustGas )
    Δud_Z_Sil = abs( ud_Z_Sil_candidate - ud_Z_Sil_new ) / ( abs(ud_Z_Sil_new) +  uth_DustGas )
    Δud_r_C   = abs( ud_r_C_candidate   - ud_r_C_new )   / ( abs(ud_r_C_new)   +  uth_DustGas )
    Δud_Z_C   = abs( ud_Z_C_candidate   - ud_Z_C_new )   / ( abs(ud_Z_C_new)   +  uth_DustGas )
    Δud_r_PAH = abs( ud_r_PAH_candidate - ud_r_PAH_new ) / ( abs(ud_r_PAH_new) +  uth_DustGas )
    Δud_Z_PAH = abs( ud_Z_PAH_candidate - ud_Z_PAH_new ) / ( abs(ud_Z_PAH_new) +  uth_DustGas )

    Δu_r      = abs( u_r_candidate  - u_r_new )  / ( abs(u_r_new)  +  uth_DustGas )
    Δu_Z      = abs( u_Z_candidate  - u_Z_new )  / ( abs(u_Z_new)  +  uth_DustGas )

    # Get maximum change:
    Δu_max    = max( Δud_r_Sil, Δud_Z_Sil, Δud_r_C, Δud_Z_C, Δud_r_PAH, Δud_Z_PAH, Δu_r, Δu_Z )

    if Δu_max < tol_DustGasCoupling && isfinite(Δu_max)
        ConvergedDustGasCoupling = true 
    else 
        ConvergedDustGasCoupling = false 
    end

    return ConvergedDustGasCoupling
end
