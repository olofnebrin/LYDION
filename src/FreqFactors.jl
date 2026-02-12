
""" FILE:    FreqFactors.jl

    PURPOSE: Contains functions for computing/updating frequency-related factors, 
             including leakage coefficients (F_{i,j,k}, F_{i,j,k}^±). 
"""

using .ConstantsAndFunctions   # Needed constants & functions
using  Base.Threads            # For CPU acceleration
using  Roots                   # Needed for root-finding in the frequency grid setup 

export UpdateVoigtAndEmissionProfile!, UpdateFreqFactors!

# Some constants needed in the evaluation of the frequency leakage coefficients,
# using the exponential scheme:
const Pmin::Float64            = 1e-3     # Taylor expand for magnitudes of P smaller than Pmin 
const Pmax::Float64            = 40.0     # Clamp for magnitudes of P greater than Pmax 
const minus_Pmax::Float64      = - Pmax 


""" This function updates the Voigt profile and emission profile for all cells.
    Mostly used in non-RHD/static tests, where these only have to be updated 
    initially, once. """
function UpdateVoigtAndEmissionProfile!( Lyα::LyαArrays,                 # The Lyα state/arrays,
                                           T::AbstractArray{Float64, 2}  # Gas temperature for every cell 
                                         )

    N_Z, N_r = size( T )
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z 
 
    # Update the Voigt parameter a, the Doppler width and its inverse,
    # for all cells:
    @threads for j in dN_Z 
        @simd for i in dN_r 
            @inbounds begin 
                Lyα.a[i,j]       = Compute_a( T[i,j] )
                Lyα.ΔνD[i,j]     = Compute_ΔνD( T[i,j] )
                Lyα.inv_ΔνD[i,j] = 1.0 / Lyα.ΔνD[i,j]
            end
        end
    end

    # Update Hvoigt[i,j,k] for every cell:
    Update_Hvoigt!( Lyα.Hvoigt, Lyα.δν, Lyα.δνC, Lyα.a, Lyα.inv_ΔνD )

    # Update the normalized emission line profile ϕ_ν for every cell:
    Update_ϕν!( Lyα.ϕ_ν, Lyα.Δν, Lyα.Hvoigt, Lyα.kcore_min, Lyα.kcore_max )

    return nothing 
end


""" This function evaluates:

    1 - (ϕ^L)(P) =  exp( - P ) 
    (χ^L)(P)     =  (ϕ^L)(P) / P
    
    These are needed for the frequency leakage coefficients. """
@inline function OneMinusϕ_L_and_χ_L( P::Float64 )
    if abs(P) < Pmin
        # Second-order Taylor expansion of exp(-P):
        OneMinusϕ_L = @fastmath( 1.0  -  P  +  0.5 * ( P * P ) )

        # Second-order Taylor expansion of 1 - exp(-P) 
        # to evaluate [1 - exp(-P)] / P: 
        χ_L         = @fastmath( 1.0  -  0.5 * P )

        return OneMinusϕ_L, χ_L
    else 
        # Compute exp(-P) once:
        OneMinusϕ_L = @fastmath exp( - P )

        # Compute [1 - exp(-P)] / P:
        χ_L         = @fastmath( ( 1.0 - OneMinusϕ_L ) / P )

        return OneMinusϕ_L, χ_L
    end
end


""" This function evaluates:

    1 - (ϕ^R)(P) =  exp( P ) 
    (χ^R)(P)     =  (ϕ^R)(P) / P
    
    These are needed for the frequency leakage coefficients. """
@inline function OneMinusϕ_R_and_χ_R( P::Float64 )
    if abs(P) < Pmin
        # Second-order Taylor expansion of exp(P):
        OneMinusϕ_R = @fastmath( 1.0  +  P  +  0.5 * ( P * P ) )

        # Second-order Taylor expansion of 1 - exp(P) to 
        # evaluate [1 - exp(P)] / P:
        χ_R         = @fastmath( - 1.0  -  0.5 * P )

        return OneMinusϕ_R, χ_R
    else
        # Compute exp(P):
        OneMinusϕ_R = @fastmath exp(P)

        # Compute [1 - exp(P)] / P:
        χ_R         = @fastmath( ( 1.0 - OneMinusϕ_R ) / P )

        return OneMinusϕ_R, χ_R
    end
end



const νLyα_over_c::Float64 = νLyα / c
const h_over_kB::Float64   = h    / kB 

""" This function updates the frequency leakage coefficients (F_{i,j,k}^±, F_{i,j,k})
    for each cell. NOTE: Assumes updated values of Hvoigt and emission profile ϕ_ν. """
@views function UpdateFreqFactors!( Lyα::LyαArrays,                 # The Lyα state/arrays 
                                      T::AbstractArray{Float64, 2}, # Gas temperature for every cell 
                                ∂u_r_∂r::AbstractArray{Float64, 2}, # ∂u_r/∂r
                                ∂u_r_∂Z::AbstractArray{Float64, 2}, # ∂u_r/∂Z
                                ∂u_Z_∂r::AbstractArray{Float64, 2}, # ∂u_Z/∂r
                                ∂u_Z_∂Z::AbstractArray{Float64, 2}, # ∂u_Z/∂Z
                             u_r_over_r::AbstractArray{Float64, 2}, # ⟨u_r / r⟩ 
                                 )
    # Get grid size:
    N_r, N_Z = size( T )

    # Needed intervals:
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z

    # Update Eddington factors D to ensure they use the latest values (from post-advection):
    @threads for k in Lyα.dN_ν
        Update_D_Edd_Serial!( Lyα.D_rr_Lyα[:,:,k], Lyα.D_ZZ_Lyα[:,:,k], Lyα.D_rZ_Lyα[:,:,k],
                              Lyα.H_r_Lyα[:,:,k],  Lyα.H_Z_Lyα[:,:,k],  Lyα.J_Lyα[:,:,k] )
    end

    # Recoil setting (true if on, false if off):
    RECOIL_ON = Lyα.Recoil

    # Get frequency leakage factors, for all interior bins:
    @threads :static for k in Lyα.dN_ν_int 
        # Get intervals:
        @inbounds δν_kp     = Lyα.νC[k+1] - Lyα.ν[k+1]   # ν̄_{k+1}   - ν_{k+1/2}
        @inbounds δν_k      = Lyα.ν[k+1]  - Lyα.νC[k]    # ν_{k+1/2} - ν̄_{k}
        @inbounds δν_km     = Lyα.ν[k]    - Lyα.νC[k-1]  # ν_{k-1/2} - ν̄_{k-1}

        # Inverses of Δν and νC are also needed:
        @inbounds Inv_Δν    = Lyα.inv_Δν[k]
        @inbounds Inv_νC_k  = Lyα.inv_νC[k]
        @inbounds Inv_νC_kp = Lyα.inv_νC[k+1]
        @inbounds Inv_νC_km = Lyα.inv_νC[k-1]

        # Also convenient, for the recoil-related terms:
        Two_Inv_νC_k        = 2.0 * Inv_νC_k
        Two_Inv_νC_kp       = 2.0 * Inv_νC_kp
        Two_Inv_νC_km       = 2.0 * Inv_νC_km 

        # Get νC[k],νC[k+1], νC[k-1] once:
        @inbounds νC_k      = Lyα.νC[k]
        @inbounds νC_kp     = Lyα.νC[k+1]
        @inbounds νC_km     = Lyα.νC[k-1]

        # Get the opacity floor once:
        α_Lyα_floor         = Lyα.α_Lyα_floor

        for j in dN_Z
            @inbounds for i in dN_r 

                # Compute (1/2) α (Δν_D)² and the recoil factor h / kB T:
                ΔνD        = Lyα.ΔνD[i,j]
                PREFACT    = 0.5 * max( Lyα.α_Lyα[i,j], α_Lyα_floor ) * ( ΔνD * ΔνD )  
                RECOIL_0   = h_over_kB / T[i,j]                                     


                # The Doppler factor (D:∇u)/c ( → (∇⋅u)/(3c) for isotropic intensity):
                Drr        = Float64( Lyα.D_rr_Lyα[i,j,k] )
                DrZ        = Float64( Lyα.D_rZ_Lyα[i,j,k] )
                DZZ        = Float64( Lyα.D_ZZ_Lyα[i,j,k] )
                D_colon_∇u = D_colon_∇u_Cylindrical( Drr,           DrZ,             DZZ,
                                                     ∂u_r_∂r[i,j],  ∂u_r_∂Z[i,j],    ∂u_Z_∂r[i,j],
                                                     ∂u_Z_∂Z[i,j],  u_r_over_r[i,j] )
                VELOCITY   = D_colon_∇u * inv_c 

                # Get K and C:
                Kkp        = PREFACT  * Lyα.Hvoigt[i, j, k+1]
                Kk         = PREFACT  * Lyα.Hvoigt[i, j, k]   
                Kkm        = PREFACT  * Lyα.Hvoigt[i, j, k-1]  
                Ckp        = VELOCITY * νC_kp 
                Ck         = VELOCITY * νC_k
                Ckm        = VELOCITY * νC_km

                if RECOIL_ON
                    # Add recoil terms:
                    Ckp   += Kkp * ( RECOIL_0  -  Two_Inv_νC_kp )
                    Ck    += Kk  * ( RECOIL_0  -  Two_Inv_νC_k ) 
                    Ckm   += Kkm * ( RECOIL_0  -  Two_Inv_νC_km )
                end

                # GET PECLET NUMBERS:
                δν_kp_over_Kkp  = δν_kp * inv( Kkp )  # δν_{k+1} / K_{k+1}
                δν_k_over_Kk    = δν_k  * inv( Kk )   # δν_{k}   / K_{k}
                δν_km_over_Kkm  = δν_km * inv( Kkm )  # δν_{k-1} / K_{k-1}

                Pkp             = clamp( Ckp * δν_kp_over_Kkp, minus_Pmax, Pmax )
                Pk              = clamp( Ck  * δν_k_over_Kk,   minus_Pmax, Pmax )
                Pkm             = clamp( Ckm * δν_km_over_Kkm, minus_Pmax, Pmax )

                # COMPUTE ϕ^L, ϕ^R, χ^L, χ^R:
                OneMinusϕ_R_Pkp, χ_R_Pkp = OneMinusϕ_R_and_χ_R( Pkp )
                OneMinusϕ_L_Pk,  χ_L_Pk  = OneMinusϕ_L_and_χ_L( Pk )
                OneMinusϕ_R_Pk,  χ_R_Pk  = OneMinusϕ_R_and_χ_R( Pk )
                OneMinusϕ_L_Pkm, χ_L_Pkm = OneMinusϕ_L_and_χ_L( Pkm )

                # COMPUTE F_{i,j,k}^+:
                denom_Fij_kp       = Inv_Δν * inv( χ_L_Pk * δν_k_over_Kk  -  χ_R_Pkp * δν_kp_over_Kkp )
                Lyα.Fij_kp[i,j,k]  = OneMinusϕ_R_Pkp * denom_Fij_kp 

                # COMPUTE F_{i,j,k}^-:
                denom_Fij_km       = Inv_Δν * inv( χ_L_Pkm * δν_km_over_Kkm  -  χ_R_Pk * δν_k_over_Kk )
                Lyα.Fij_km[i,j,k]  = OneMinusϕ_L_Pkm * denom_Fij_km 

                # COMPUTE F_{i,j,k}:
                Lyα.Fij_k[i,j,k]   = OneMinusϕ_L_Pk * denom_Fij_kp  +  OneMinusϕ_R_Pk * denom_Fij_km 

            end # End of i-loop 
        end # End of j-loop 
    end # End of k-loop (frequency)


    return nothing 
end

