
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

    N_r, N_Z = size( T )
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

    # Update Hvoigt[k,i,j] for every cell:
    Update_Hvoigt!( Lyα.Hvoigt, Lyα.δν, Lyα.δνC, Lyα.a, Lyα.inv_ΔνD )

    # Update the normalized emission line profile ϕ_ν for every cell:
    Update_ϕν!( Lyα.ϕ_ν, Lyα.Δν, Lyα.Hvoigt, Lyα.kcore_min, Lyα.kcore_max )

    return nothing 
end


const OneMinusϕ_L_Pmax::Float64       = exp( - Pmax )
const χ_L_Pmax::Float64               = ( 1.0 - OneMinusϕ_L_Pmax ) / Pmax
const OneMinusϕ_L_minus_Pmax::Float64 = exp( Pmax )
const χ_L_minus_Pmax::Float64         = ( 1.0 - OneMinusϕ_L_minus_Pmax ) / ( - Pmax )

""" This function evaluates:

    1 - (ϕ^L)(P) =  exp( - P ) 
    (χ^L)(P)     =  (ϕ^L)(P) / P
    
    These are needed for the frequency leakage coefficients. """
@inline function OneMinusϕ_L_and_χ_L( P::Float64 )

    # Exit early near the maximum (in magnitude) values of P:
    if P ≥ Pmax 
        return OneMinusϕ_L_Pmax, χ_L_Pmax
    elseif P ≤ - Pmax
        return OneMinusϕ_L_minus_Pmax, χ_L_minus_Pmax
    end

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


const OneMinusϕ_R_Pmax::Float64       = exp( Pmax )
const χ_R_Pmax::Float64               = ( 1.0 - OneMinusϕ_R_Pmax ) / Pmax
const OneMinusϕ_R_minus_Pmax::Float64 = exp( - Pmax )
const χ_R_minus_Pmax::Float64         = ( 1.0 - OneMinusϕ_R_minus_Pmax ) / ( - Pmax )

""" This function evaluates:

    1 - (ϕ^R)(P) =  exp( P ) 
    (χ^R)(P)     =  (ϕ^R)(P) / P
    
    These are needed for the frequency leakage coefficients. """
@inline function OneMinusϕ_R_and_χ_R( P::Float64 )

    # Exit early near the maximum (in magnitude) values of P:
    if P ≥ Pmax 
        return OneMinusϕ_R_Pmax, χ_R_Pmax
    elseif P ≤ - Pmax
        return OneMinusϕ_R_minus_Pmax, χ_R_minus_Pmax
    end

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

""" This function updates the frequency leakage coefficients (F_{k,i,j}^±, F_{k,i,j})
    for each cell. NOTE: Assumes updated values of Hvoigt and emission profile ϕ_ν. """
function UpdateFreqFactors!( Lyα::LyαArrays,                 # The Lyα state/arrays 
                               T::AbstractArray{Float64, 2}, # Gas temperature for every cell 
                         ∂u_r_∂r::AbstractArray{Float64, 2}, # ∂u_r/∂r
                         ∂u_r_∂Z::AbstractArray{Float64, 2}, # ∂u_r/∂Z
                         ∂u_Z_∂r::AbstractArray{Float64, 2}, # ∂u_Z/∂r
                         ∂u_Z_∂Z::AbstractArray{Float64, 2}, # ∂u_Z/∂Z
                      u_r_over_r::AbstractArray{Float64, 2}, # ⟨u_r / r⟩ 
                                 )

    # Get grid size:
    N_r, N_Z = size( T )
    N_cells  = N_r * N_Z

    # Needed frequency interval:
    dN_ν_int    = Lyα.dN_ν_int 

    # Unpack arrays:
    J           = Lyα.J_Lyα
    H_r         = Lyα.H_r_Lyα
    H_Z         = Lyα.H_Z_Lyα

    Hvoigt      = Lyα.Hvoigt
    Fij_kp      = Lyα.Fij_kp
    Fij_km      = Lyα.Fij_km
    Fij_k       = Lyα.Fij_k

    α_Lyα       = Lyα.α_Lyα
    ν           = Lyα.ν
    νC          = Lyα.νC
    inv_Δν      = Lyα.inv_Δν
    inv_νC      = Lyα.inv_νC
    ΔνD         = Lyα.ΔνD

    # Recoil setting (true if on, false if off):
    RECOIL_ON   = Lyα.Recoil

    # Get the opacity floor once:
    α_Lyα_floor = Lyα.α_Lyα_floor

    @threads for idx in 1:N_cells
        # Get cell indices from idx:
        i = ( (idx - 1) % N_r ) + 1
        j = ( (idx - 1) ÷ N_r ) + 1 

        @inbounds begin 
            # Compute (1/2) α (Δν_D)² and the recoil factor h / kB T:
            ΔνD_ij        = ΔνD[i,j]
            PREFACT       = @fastmath 0.5 * max( α_Lyα[i,j], α_Lyα_floor ) * ( ΔνD_ij * ΔνD_ij )  
            RECOIL_0      = @fastmath h_over_kB / T[i,j]  

            # Velocity gradient components needed for each bin:
            ∂u_r_∂r_ij    = ∂u_r_∂r[i,j] 
            ∂u_r_∂Z_ij    = ∂u_r_∂Z[i,j]    
            ∂u_Z_∂r_ij    = ∂u_Z_∂r[i,j]
            ∂u_Z_∂Z_ij    = ∂u_Z_∂Z[i,j]
            u_r_over_r_ij = u_r_over_r[i,j]

            @simd for k in dN_ν_int 

                # Get νC[k],νC[k+1], νC[k-1] once:
                νC_k      = νC[k]
                νC_kp     = νC[k+1]
                νC_km     = νC[k-1]
                
                # Get intervals:
                δν_kp     = νC_kp   - ν[k+1]   # ν̄_{k+1}   - ν_{k+1/2}
                δν_k      = ν[k+1]  - νC_k     # ν_{k+1/2} - ν̄_{k}
                δν_km     = ν[k]    - νC_km    # ν_{k-1/2} - ν̄_{k-1}

                # Inverses of Δν and νC are also needed:
                Inv_Δν_k  = inv_Δν[k]
                Inv_νC_k  = inv_νC[k]
                Inv_νC_kp = inv_νC[k+1]
                Inv_νC_km = inv_νC[k-1]

                # Also convenient, for the recoil-related terms:
                Two_Inv_νC_k        = @fastmath 2.0 * Inv_νC_k
                Two_Inv_νC_kp       = @fastmath 2.0 * Inv_νC_kp
                Two_Inv_νC_km       = @fastmath 2.0 * Inv_νC_km 

                # Update Eddington factors D to ensure they use the latest values (from post-advection):
                Drr, DZZ, DrZ = Compute_D_Edd( H_r[k,i,j], H_Z[k,i,j], J[k,i,j] )

                # The Doppler factor (D:∇u)/c ( → (∇⋅u)/(3c) for isotropic intensity):
                D_colon_∇u = D_colon_∇u_Cylindrical( Drr,         DrZ,           DZZ,
                                                     ∂u_r_∂r_ij,  ∂u_r_∂Z_ij,    ∂u_Z_∂r_ij,
                                                     ∂u_Z_∂Z_ij,  u_r_over_r_ij )

                VELOCITY   = @fastmath D_colon_∇u * inv_c 

                # Get K and C:
                Kkp        = @fastmath PREFACT  * Hvoigt[k+1, i, j]
                Kk         = @fastmath PREFACT  * Hvoigt[k,   i, j]   
                Kkm        = @fastmath PREFACT  * Hvoigt[k-1, i, j]  
                Ckp        = @fastmath VELOCITY * νC_kp 
                Ck         = @fastmath VELOCITY * νC_k
                Ckm        = @fastmath VELOCITY * νC_km

                if RECOIL_ON
                    # Add recoil terms:
                    Ckp   += @fastmath Kkp * ( RECOIL_0  -  Two_Inv_νC_kp )
                    Ck    += @fastmath Kk  * ( RECOIL_0  -  Two_Inv_νC_k ) 
                    Ckm   += @fastmath Kkm * ( RECOIL_0  -  Two_Inv_νC_km )
                end

                # GET PECLET NUMBERS:
                δν_kp_over_Kkp  = @fastmath δν_kp * inv( Kkp )  # δν_{k+1} / K_{k+1}
                δν_k_over_Kk    = @fastmath δν_k  * inv( Kk )   # δν_{k}   / K_{k}
                δν_km_over_Kkm  = @fastmath δν_km * inv( Kkm )  # δν_{k-1} / K_{k-1}

                Pkp             = clamp( Ckp * δν_kp_over_Kkp, minus_Pmax, Pmax )
                Pk              = clamp( Ck  * δν_k_over_Kk,   minus_Pmax, Pmax )
                Pkm             = clamp( Ckm * δν_km_over_Kkm, minus_Pmax, Pmax )

                # COMPUTE ϕ^L, ϕ^R, χ^L, χ^R:
                OneMinusϕ_R_Pkp, χ_R_Pkp = OneMinusϕ_R_and_χ_R( Pkp )
                OneMinusϕ_L_Pk,  χ_L_Pk  = OneMinusϕ_L_and_χ_L( Pk )
                OneMinusϕ_R_Pk,  χ_R_Pk  = OneMinusϕ_R_and_χ_R( Pk )
                OneMinusϕ_L_Pkm, χ_L_Pkm = OneMinusϕ_L_and_χ_L( Pkm )

                # COMPUTE F_{k,i,j}^+:
                denom_Fij_kp  = @fastmath Inv_Δν_k * inv( χ_L_Pk * δν_k_over_Kk  -  χ_R_Pkp * δν_kp_over_Kkp )
                Fij_kp[k,i,j] = @fastmath OneMinusϕ_R_Pkp * denom_Fij_kp 

                # COMPUTE F_{k,i,j}^-:
                denom_Fij_km  = @fastmath Inv_Δν_k * inv( χ_L_Pkm * δν_km_over_Kkm  -  χ_R_Pk * δν_k_over_Kk )
                Fij_km[k,i,j] = @fastmath OneMinusϕ_L_Pkm * denom_Fij_km 

                # COMPUTE F_{k,i,j}:
                Fij_k[k,i,j]  = @fastmath OneMinusϕ_L_Pk * denom_Fij_kp  +  OneMinusϕ_R_Pk * denom_Fij_km 

            end # End of k-loop 
        end # End of inbounds 
    end # End of cell loop 

    return nothing
end

