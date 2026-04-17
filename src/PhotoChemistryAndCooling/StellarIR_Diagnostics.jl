
""" FILE:       StellarIR_Diagnostics.jl

    PURPOSE:    Contains functions for computing diagnostics for stellar and IR photons, e.g. 
                radiation pressure forces, luminosities, escape fractions. 
    
"""

using .Structs: StellarRTArrays, GravityState, GeometryCylindrical
using .ConstantsAndFunctions
using  Base.Threads               # For CPU acceleration


export Update_fesc_LyC_and_LW_Cylindrical!, Update_M_F_dir!, Update_M_F_IR_L_IR!


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
    where L_bol is the stellar luminosity. The various measures are:
    
    M_F_dir            ≡ ∫dV |f_dir| / (L_bol/c)
    M_F_dir_radial_net ≡ ∫dV ( f_dir ⋅ r̂_out ) / (L_bol/c)
    M_F_dir_radial_out ≡ ∫dV max( f_dir ⋅ r̂_out, 0.0 ) / (L_bol/c) """
function Update_M_F_dir!( RT::StellarRTArrays,
                        Geom::GeometryCylindrical,
                    Z_center::Float64
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
    rCoM        = Geom.rCoM
    ZCoM        = Geom.ZCoM
    ΔV          = Geom.ΔV

    # Grid size & intervals:
    N_r, N_Z  = size( ΔV )       # Grid size 
    dN_r      = 1:N_r            # r-interval 
    dN_Z      = 1:N_Z            # Z-interval

    # Compute M_F_dir = ∫dV f_dir / (L_bol/c):
    M_F_dir            = 0.0
    M_F_dir_radial_net = 0.0
    M_F_dir_radial_out = 0.0
    for j in dN_Z
        @inbounds @simd for i in dN_r 
            # ------ Compute M_F_dir: ---------------------------------------
            M_F_dir += ΔV[i,j] * sqrt( f_stellar_r[i,j] * f_stellar_r[i,j]  +  f_stellar_Z[i,j] * f_stellar_Z[i,j] )

            # ------ Compute radially outward force multipliers: ------------

            # Distance from star/cloud center:
            Δr       = rCoM[i]
            ΔZ       = ZCoM[j] - Z_center
            distance = @fastmath sqrt( Δr * Δr  +  ΔZ * ΔZ )

            if distance > 0.0
                inv_d   = @fastmath 1.0 / distance
                e_out_r = Δr * inv_d
                e_out_Z = ΔZ * inv_d

                f_stellar_radial = f_stellar_r[i,j] * e_out_r  +  f_stellar_Z[i,j] * e_out_Z
            else
                # At the stellar position the outward direction is undefined:
                f_stellar_radial = 0.0
            end

            # Add contributions to M_F_dir_radial_net and M_F_dir_radial_out:
            M_F_dir_radial_net += ΔV[i,j] * f_stellar_radial
            M_F_dir_radial_out += ΔV[i,j] * max( f_stellar_radial, 0.0 )
        end
    end

    # Normalize by relevant luminosity:
    L_stellar = L_opt + L_FUV + L_LW + L_EUV1 + L_EUV2 + L_EUV3

    if L_stellar > 0.0
        inv_L_over_c          = 1.0 / (L_stellar * inv_c)
        RT.M_F_dir            = M_F_dir * inv_L_over_c
        RT.M_F_dir_radial_net = M_F_dir_radial_net * inv_L_over_c
        RT.M_F_dir_radial_out = M_F_dir_radial_out * inv_L_over_c
    else
        RT.M_F_dir            = 0.0
        RT.M_F_dir_radial_net = 0.0
        RT.M_F_dir_radial_out = 0.0
    end

    return nothing 
end


""" This function updates the direct stellar RT 'force multiplier' M_F_dir ≡ F_dir / (L_bol/c),
    where L_bol is the stellar luminosity. The various measures are:
    
    M_F_dir             ≡ ∫dV |f_dir| / (L_bol/c)
    M_F_dir_antigravity ≡ ∫dV ( - f_dir ⋅ â_grav ) / (L_bol/c)
    M_F_dir_radial_net  ≡ ∫dV ( f_dir ⋅ r̂_out ) / (L_bol/c)
    M_F_dir_radial_out  ≡ ∫dV max( f_dir ⋅ r̂_out, 0.0 ) / (L_bol/c) """
function Update_M_F_dir!( RT::StellarRTArrays,
                        Geom::GeometryCylindrical,
                   GravState::GravityState{Float64},
                    Z_center::Float64
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
    f_grav_r    = GravState.f_gravity_r
    f_grav_Z    = GravState.f_gravity_Z
    rCoM        = Geom.rCoM
    ZCoM        = Geom.ZCoM
    ΔV          = Geom.ΔV

    # Grid size & intervals:
    N_r, N_Z  = size( ΔV )       # Grid size 
    dN_r      = 1:N_r            # r-interval 
    dN_Z      = 1:N_Z            # Z-interval

    # Compute force multipliers:
    M_F_dir             = 0.0
    M_F_dir_antigravity = 0.0
    M_F_dir_radial_net  = 0.0
    M_F_dir_radial_out  = 0.0
    for j in dN_Z
        @inbounds @simd for i in dN_r 
            # ------ Compute M_F_dir: ---------------------------------------
            M_F_dir += ΔV[i,j] * sqrt( f_stellar_r[i,j] * f_stellar_r[i,j]  +  f_stellar_Z[i,j] * f_stellar_Z[i,j] )

            # ------ Compute force multiplier anti-parallel to gravity: ------

            # Magnitude of gravitatonal force:
            f_grav_mag = sqrt( f_grav_r[i,j] * f_grav_r[i,j]  +  f_grav_Z[i,j] * f_grav_Z[i,j] )

            if f_grav_mag > 0.0
                f_stellar_antiparallel = - ( f_stellar_r[i,j] * f_grav_r[i,j]  +  f_stellar_Z[i,j] * f_grav_Z[i,j] ) / f_grav_mag
            else
                # In regions without gravity, just set the force component to zero: 
                f_stellar_antiparallel = 0.0
            end

            # Get contribution to M_F_dir_antigravity from this cell:
            M_F_dir_antigravity += ΔV[i,j] * f_stellar_antiparallel

            # ------ Compute radially outward force multipliers: ------------

            # Distance from star/cloud center:
            Δr       = rCoM[i]
            ΔZ       = ZCoM[j] - Z_center
            distance = @fastmath sqrt( Δr * Δr  +  ΔZ * ΔZ )

            if distance > 0.0
                inv_d   = @fastmath 1.0 / distance
                e_out_r = Δr * inv_d
                e_out_Z = ΔZ * inv_d

                f_stellar_radial = f_stellar_r[i,j] * e_out_r  +  f_stellar_Z[i,j] * e_out_Z
            else
                # At the stellar position the outward direction is undefined:
                f_stellar_radial = 0.0
            end

            # Add contributions to M_F_dir_radial_net and M_F_dir_radial_out:
            M_F_dir_radial_net += ΔV[i,j] * f_stellar_radial
            M_F_dir_radial_out += ΔV[i,j] * max( f_stellar_radial, 0.0 )
        end
    end

    # Normalize by relevant luminosity:
    L_stellar = L_opt + L_FUV + L_LW + L_EUV1 + L_EUV2 + L_EUV3

    if L_stellar > 0.0
        inv_L_over_c           = 1.0 / (L_stellar * inv_c)
        RT.M_F_dir             = M_F_dir * inv_L_over_c
        RT.M_F_dir_antigravity = M_F_dir_antigravity * inv_L_over_c 
        RT.M_F_dir_radial_net  = M_F_dir_radial_net * inv_L_over_c
        RT.M_F_dir_radial_out  = M_F_dir_radial_out * inv_L_over_c
    else
        RT.M_F_dir             = 0.0
        RT.M_F_dir_antigravity = 0.0
        RT.M_F_dir_radial_net  = 0.0
        RT.M_F_dir_radial_out  = 0.0
    end

    return nothing 
end


""" This function updates the IR 'force multipliers' M_F_IR ≡ F_IR / (L_IR/c), and the total IR luminosity. 
    Useful for tracking the strength of IR radiation pressure feedback.
        
    The various measures are:
    
    M_F_IR             ≡ ∫dV |f_IR| / (L_IR/c)
    M_F_IR_antigravity ≡ ∫dV ( - f_IR ⋅ â_grav ) / (L_IR/c)
    M_F_IR_radial_net  ≡ ∫dV ( f_IR ⋅ r̂_out ) / (L_IR/c)
    M_F_IR_radial_out  ≡ ∫dV max( f_IR ⋅ r̂_out, 0.0 ) / (L_IR/c) """
function Update_M_F_IR_L_IR!( RT::StellarRTArrays,
                            Geom::GeometryCylindrical,
                        Z_center::Float64,
                        BC_Z_bot::Symbol = :Reflect
                                )

    # Unpack arrays:
    js_IR  = RT.js_IR
    f_IR_r = RT.f_IR_r
    f_IR_Z = RT.f_IR_Z
    rCoM   = Geom.rCoM
    ZCoM   = Geom.ZCoM
    ΔV     = Geom.ΔV

    # Grid size & intervals:
    N_r, N_Z  = size( f_IR_r )   # Grid size 
    dN_r      = 1:N_r            # r-interval 
    dN_Z      = 1:N_Z            # Z-interval

    # Compute force multipliers & IR luminosity L_IR = 4π ∫dV js_IR:
    M_F_IR            = 0.0
    M_F_IR_radial_net = 0.0
    M_F_IR_radial_out = 0.0
    L_IR              = 0.0
    for j in dN_Z
        @inbounds @simd for i in dN_r 
            # ------ Compute M_F_IR: ---------------------------------------
            M_F_IR += ΔV[i,j] * sqrt( f_IR_r[i,j] * f_IR_r[i,j]  +  f_IR_Z[i,j] * f_IR_Z[i,j] )

            # ------ Compute M_F_IR_radial_net: ----------------------------

            # Distance from star/cloud center:
            Δr       = rCoM[i]
            ΔZ       = ZCoM[j] - Z_center
            distance = @fastmath sqrt( Δr * Δr  +  ΔZ * ΔZ )

            if distance > 0.0
                inv_d   = @fastmath 1.0 / distance
                e_out_r = Δr * inv_d
                e_out_Z = ΔZ * inv_d

                f_IR_radial = f_IR_r[i,j] * e_out_r  +  f_IR_Z[i,j] * e_out_Z
            else
                # At the stellar position the outward direction is undefined:
                f_IR_radial = 0.0
            end

            # Add contributions to M_F_IR_radial_net and M_F_IR_radial_out:
            M_F_IR_radial_net += ΔV[i,j] * f_IR_radial
            M_F_IR_radial_out += ΔV[i,j] * max( f_IR_radial, 0.0 )

            # ------ Compute IR luminosity: -------------------------------
            L_IR   += Four_π * ΔV[i,j] * js_IR[i,j]
        end
    end

    # Normalize M_F_IR:
    if L_IR > 0.0
        inv_L_over_c      = 1.0 / (L_IR * inv_c)
        M_F_IR            = M_F_IR * inv_L_over_c
        M_F_IR_radial_net = M_F_IR_radial_net * inv_L_over_c
        M_F_IR_radial_out = M_F_IR_radial_out * inv_L_over_c
    else
        M_F_IR            = 0.0
        M_F_IR_radial_net = 0.0
        M_F_IR_radial_out = 0.0
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
    RT.M_F_IR            = M_F_IR
    RT.M_F_IR_radial_net = M_F_IR_radial_net
    RT.M_F_IR_radial_out = M_F_IR_radial_out
    RT.Lum_IR            = L_IR_tot

    return nothing 
end


""" The function computes the IR force multipliers M_F_IR ≡ F_IR / (L_IR/c),
    and the total IR Luminosity.
    
    The various measures are:
    
    M_F_IR             ≡ ∫dV |f_IR| / (L_IR/c)
    M_F_IR_antigravity ≡ ∫dV ( - f_IR ⋅ â_grav ) / (L_IR/c)
    M_F_IR_radial_net  ≡ ∫dV ( f_IR ⋅ r̂_out ) / (L_IR/c)
    M_F_IR_radial_out  ≡ ∫dV max( f_IR ⋅ r̂_out, 0.0 ) / (L_IR/c) """
function Update_M_F_IR_L_IR!( RT::StellarRTArrays, 
                            Geom::GeometryCylindrical,
                       GravState::GravityState{Float64}, 
                        Z_center::Float64,
                        BC_Z_bot::Symbol = :Reflect
                                )

    # Unpack arrays:
    js_IR    = RT.js_IR 
    f_IR_r   = RT.f_IR_r
    f_IR_Z   = RT.f_IR_Z
    f_grav_r = GravState.f_gravity_r
    f_grav_Z = GravState.f_gravity_Z
    rCoM     = Geom.rCoM
    ZCoM     = Geom.ZCoM
    ΔV       = Geom.ΔV

    # Grid size & intervals:
    N_r, N_Z  = size( f_IR_r )   # Grid size 
    dN_r      = 1:N_r            # r-interval 
    dN_Z      = 1:N_Z            # Z-interval

    # Compute IR force multipliers (except anti-gravity one) & IR luminosity L = 4π ∫dV js_IR:
    M_F_IR            = 0.0
    M_F_IR_radial_net = 0.0
    M_F_IR_radial_out = 0.0
    L_IR              = 0.0
    for j in dN_Z
        @inbounds @simd for i in dN_r 
            # ------ Compute M_F_IR: ---------------------------------------
            M_F_IR += ΔV[i,j] * sqrt( f_IR_r[i,j] * f_IR_r[i,j]  +  f_IR_Z[i,j] * f_IR_Z[i,j] )

            # ------ Compute M_F_IR_radial_net: ----------------------------

            # Distance from star/cloud center:
            Δr       = rCoM[i]
            ΔZ       = ZCoM[j] - Z_center
            distance = @fastmath sqrt( Δr * Δr  +  ΔZ * ΔZ )

            if distance > 0.0
                inv_d   = @fastmath 1.0 / distance
                e_out_r = Δr * inv_d
                e_out_Z = ΔZ * inv_d

                f_IR_radial = f_IR_r[i,j] * e_out_r  +  f_IR_Z[i,j] * e_out_Z
            else
                # At the stellar position the outward direction is undefined:
                f_IR_radial = 0.0
            end

            # Add contributions to M_F_IR_radial_net and M_F_IR_radial_out:
            M_F_IR_radial_net += ΔV[i,j] * f_IR_radial
            M_F_IR_radial_out += ΔV[i,j] * max( f_IR_radial, 0.0 )

            # ------ Compute IR luminosity: -------------------------------
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
        inv_L_over_c       = 1.0 / (L_IR * inv_c)
        M_F_IR             = M_F_IR * inv_L_over_c
        M_F_IR_antigravity = M_F_IR_antigravity * inv_L_over_c
        M_F_IR_radial_net  = M_F_IR_radial_net * inv_L_over_c
        M_F_IR_radial_out  = M_F_IR_radial_out * inv_L_over_c
    else
        M_F_IR             = 0.0
        M_F_IR_antigravity = 0.0
        M_F_IR_radial_net  = 0.0
        M_F_IR_radial_out  = 0.0
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
    RT.M_F_IR_radial_net  = M_F_IR_radial_net
    RT.M_F_IR_radial_out  = M_F_IR_radial_out
    RT.Lum_IR             = L_IR_tot

    return nothing 
end
