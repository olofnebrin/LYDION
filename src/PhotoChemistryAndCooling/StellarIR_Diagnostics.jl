
""" FILE:       StellarIR_Diagnostics.jl

    PURPOSE:    Contains functions for computing diagnostics for stellar and IR photons, e.g. 
                radiation pressure forces, luminosities, escape fractions. 
    
"""

using .Structs: StellarRTArrays, GravityState
using .ConstantsAndFunctions
using  Base.Threads               # For CPU acceleration


export Update_fesc_LyC_and_LW_Cylindrical!, Update_M_F_IR_L_IR!


""" This function updates the LyC and Lyman-Werner escape fraction from the 
    simulation volume (assuming cylindrical coordinates) in RHD simulations. """
@inline function Update_fesc_LyC_and_LW_Cylindrical!(  RT::StellarRTArrays,
                                                      A_r::AbstractArray{Float64, 2}, # r-direction interface areas 
                                                      A_Z::AbstractArray{Float64, 2}, # Z-direction interface areas 
                                                 BC_Z_bot::Symbol = :Outflow          # Assumed BCs at the lower-Z boundary 
                                                     )

    # Lyman-Werner escape fraction: 
    RT.f_esc_LW   = Compute_f_esc_Cylindrical( RT.H_r_LW, RT.H_Z_LW, A_r, A_Z, RT.Lum_LW, BC_Z_bot )

    # Compute LyC photon number-weighted escape fraction:
    f_esc_EUV1    = Compute_f_esc_Cylindrical( RT.H_r_EUV1, RT.H_Z_EUV1, A_r, A_Z, RT.Lum_EUV1, BC_Z_bot )
    f_esc_EUV2    = Compute_f_esc_Cylindrical( RT.H_r_EUV2, RT.H_Z_EUV2, A_r, A_Z, RT.Lum_EUV2, BC_Z_bot )
    f_esc_EUV3    = Compute_f_esc_Cylindrical( RT.H_r_EUV3, RT.H_Z_EUV3, A_r, A_Z, RT.Lum_EUV3, BC_Z_bot )

    if RT.dN_dt_LyC > 0.0
        RT.f_esc_LyC  = ( RT.dN_dt_EUV1 * f_esc_EUV1  +  RT.dN_dt_EUV2 * f_esc_EUV2  +  RT.dN_dt_EUV3 * f_esc_EUV3 ) / RT.dN_dt_LyC
    else
        RT.f_esc_LyC  = 0.0
    end

    return nothing 
end


""" This function updates the direct stellar RT 'force multiplier' M_F_dir ≡ F_dir / (L_bol/c),
    where L_bol is the stellar luminosity. """
function Update_M_F_dir!( RT::StellarRTArrays,
                          ΔV::AbstractArray{Float64, 2},
                    BC_Z_bot::Symbol = :Reflect
                                )

    # Stellar luminosities in all bands:
    L_opt  = RT.Lum_opt_vol
    L_FUV  = RT.Lum_FUV_vol
    L_LW   = RT.Lum_LW_vol 
    L_EUV1 = RT.Lum_EUV1_vol
    L_EUV2 = RT.Lum_EUV2_vol
    L_EUV3 = RT.Lum_EUV3_vol

    # Unpack arrays:
    f_stellar_r = RT.f_stellar_r
    f_stellar_Z = RT.f_stellar_Z

    # Grid size & intervals:
    N_r, N_Z  = size( ΔV )       # Grid size 
    dN_r      = 1:N_r            # r-interval 
    dN_Z      = 1:N_Z            # Z-interval

    # Compute M_F_dir = ∫dV f_dir / (L_bol/c):
    M_F_dir  = 0.0
    for j in dN_Z
        @inbounds @simd for i in dN_r 
            M_F_dir += ΔV[i,j] * sqrt( f_stellar_r[i,j] * f_stellar_r[i,j]  +  f_stellar_Z[i,j] * f_stellar_Z[i,j] )
        end
    end

    # Normalize by relevant luminosity:
    L_stellar = L_opt + L_FUV + L_LW + L_EUV1 + L_EUV2 + L_EUV3

    if L_stellar > 0.0
        RT.M_F_dir = M_F_dir / (L_stellar * inv_c)
    else
        RT.M_F_dir = 0.0
    end

    return nothing 
end


""" This function updates the IR 'force multiplier' M_F_IR ≡ F_IR / (L_IR/c), and the total IR luminosity. 
    
    Useful for tracking the strength of IR radiation pressure feedback. """
function Update_M_F_IR_L_IR!( RT::StellarRTArrays,
                              ΔV::AbstractArray{Float64, 2},
                        BC_Z_bot::Symbol = :Reflect
                                )

    # Unpack arrays:
    js_IR  = RT.js_IR
    f_IR_r = RT.f_IR_r
    f_IR_Z = RT.f_IR_Z

    # Grid size & intervals:
    N_r, N_Z  = size( f_IR_r )   # Grid size 
    dN_r      = 1:N_r            # r-interval 
    dN_Z      = 1:N_Z            # Z-interval

    # Compute trapping factor M_F_IR = ∫dV f_IR / (L_IR/c) & IR luminosity L_IR = 4π ∫dV js_IR:
    M_F_IR  = 0.0
    L_IR    = 0.0
    for j in dN_Z
        @inbounds @simd for i in dN_r 
            M_F_IR += ΔV[i,j] * sqrt( f_IR_r[i,j] * f_IR_r[i,j]  +  f_IR_Z[i,j] * f_IR_Z[i,j] )
            L_IR   += Four_π * ΔV[i,j] * js_IR[i,j]
        end
    end

    # Normalize M_F_IR:
    if L_IR > 0.0
        M_F_IR = M_F_IR / (L_IR * inv_c)
    else
        M_F_IR = 0.0
    end

    # Compute the total IR luminosity in the cloud, *taking into account reflective BCs*:
    if BC_Z_bot === :Reflect 
        # If we have reflective BCs on the lower Z-plane, we double to get the total Lyα luminosity 
        # in the physical cloud:
        L_IR_tot = 2 * L_IR
    else
        L_IR_tot = L_IR
    end

    # Update M_F_IR and L_IR:
    RT.M_F_IR = M_F_IR
    RT.Lum_IR = L_IR_tot

    return nothing 
end


""" The function computes the IR force multiplier M_F_IR ≡ F_IR / (L_IR/c), the force multiplier 
    *anti-parallel* to gravity (which is a better measure of its feedback strength),
    and the IR Luminosity. """
function Update_M_F_IR_L_IR!( RT::StellarRTArrays, 
                              ΔV::AbstractArray{Float64, 2},
                       gravstate::GravityState{Float64}, 
                        BC_Z_bot::Symbol = :Reflect
                                )

    # Unpack arrays:
    js_IR    = RT.js_IR 
    f_IR_r   = RT.f_IR_r
    f_IR_Z   = RT.f_IR_Z
    f_grav_r = gravstate.f_gravity_r
    f_grav_Z = gravstate.f_gravity_Z

    # Grid size & intervals:
    N_r, N_Z  = size( f_IR_r )   # Grid size 
    dN_r      = 1:N_r            # r-interval 
    dN_Z      = 1:N_Z            # Z-interval

    # Compute IR force multiplier M_F_IR = ∫dV f_IR / (L_IR/c) & IR luminosity L = 4π ∫dV js_IR:
    M_F_IR  = 0.0
    L_IR    = 0.0
    for j in dN_Z
        @inbounds @simd for i in dN_r 
            M_F_IR += ΔV[i,j] * sqrt( f_IR_r[i,j] * f_IR_r[i,j]  +  f_IR_Z[i,j] * f_IR_Z[i,j] )
            L_IR   += Four_π * ΔV[i,j] * js_IR[i,j]
        end
    end

    # Compute M_F_IR anti-parallel to gravity, 
    # M_F_IR_antigravity = ∫dV ( - f_IR ⋅ â_grav ) / (L_IR/c)
    # NOTE: Negative values mean that radiation pressure is 'helping' gravity 

    M_F_IR_antigravity = 0.0
    for j in dN_Z 
        @inbounds @simd for i in dN_r

            # Magnitude of gravitatonal force:
            f_grav_mag = sqrt( f_grav_r[i,j] * f_grav_r[i,j]  +  f_grav_Z[i,j] * f_grav_Z[i,j] )

            if f_grav_mag > 0.0
                f_IR_antiparallel = - ( f_IR_r[i,j] * f_grav_r[i,j]  +  f_IR_Z[i,j] * f_grav_Z[i,j] ) / f_grav_mag
            else
                # In regions without gravity, just set the force component to zero: 
                f_IR_antiparallel = 0.0
            end

            # Get contribution to M_F_IR_antigravity from this cell:
            M_F_IR_antigravity += ΔV[i,j] * f_IR_antiparallel
        end
    end

    # Normalize M_F_IR & M_F_IR_antigravity:
    if L_IR > 0.0
        inv_Loverc         = 1 / (L_IR * inv_c)
        M_F_IR             = M_F_IR * inv_Loverc
        M_F_IR_antigravity = M_F_IR_antigravity * inv_Loverc
    else
        M_F_IR             = 0.0
        M_F_IR_antigravity = 0.0
    end

    # Compute the total IR luminosity in the cloud, *taking into account reflective BCs*:
    if BC_Z_bot === :Reflect 
        # If we have reflective BCs on the lower Z-plane, we double to get the total Lyα luminosity 
        # in the physical cloud:
        L_IR_tot = 2 * L_IR
    else
        L_IR_tot = L_IR
    end

    # Update M_F_IR and L_IR:
    RT.M_F_IR             = M_F_IR
    RT.M_F_IR_antigravity = M_F_IR_antigravity
    RT.Lum_IR             = L_IR_tot

    return nothing 
end