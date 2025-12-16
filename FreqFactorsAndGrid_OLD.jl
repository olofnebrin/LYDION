
""" FILE:    FreqFactorsAndGrid.jl

    PURPOSE: Contains functions for setting up a frequency grid,
             and computing frequency leakage coefficients (F_{i,j,k}, F_{i,j,k}^±). 
"""

using .ConstantsAndFunctions   # Needed constants & functions
using  Base.Threads            # For CPU acceleration
using  Roots                   # Needed for root-finding in the frequency grid setup 

export UpdateFreqFactors!

const νLyα_over_c::Float64 = νLyα / c
const h_over_kB::Float64   = h    / kB 

""" This function updates the frequency leakage coefficients (F_{i,j,k}^±, F_{i,j,k}),
    the Voigt profile for each cell (Hvoigt), and emission line profile
    for each cell (ϕ_ν). """
function UpdateFreqFactors!( ν::AbstractArray{Float64}, 
                            νC::AbstractArray{Float64},
                            Δν::AbstractArray{Float64},
                        Fij_kp::AbstractArray{Float64, 3}, 
                         Fij_k::AbstractArray{Float64, 3}, 
                        Fij_km::AbstractArray{Float64, 3}, 
                        Hvoigt::AbstractArray{Float64, 3}, 
                           ϕ_ν::AbstractArray{Float64, 3}, 
                             α::AbstractArray{Float64, 2}, 
                             T::AbstractArray{Float64, 2},
                          D_rr::AbstractArray{Float64, 3}, # Eddington factor D_rr for all cells & frequency bins
                          D_rZ::AbstractArray{Float64, 3}, # Eddington factor D_rZ for all cells & frequency bins
                          D_ZZ::AbstractArray{Float64, 3}, # Eddington factor D_ZZ for all cells & frequency bins
                       ∂u_r_∂r::AbstractArray{Float64, 2}, # ∂u_r/∂r
                       ∂u_r_∂Z::AbstractArray{Float64, 2}, # ∂u_r/∂Z
                       ∂u_Z_∂r::AbstractArray{Float64, 2}, # ∂u_Z/∂r
                       ∂u_Z_∂Z::AbstractArray{Float64, 2}, # ∂u_Z/∂Z
                    u_r_over_r::AbstractArray{Float64, 2}, # ⟨u_r / r⟩ 
                        recoil::Bool    = true,
                   α_Lyα_floor::Float64 = 0.0
                           )
    # Get grid size:
    N_r, N_Z, N_ν = size( Hvoigt )

    # Needed intervals:
    dN_r  = 1:N_r
    dN_Z  = 1:N_Z
    dN_ν  = 2:(N_ν-1)   # Only the interior frequency bins 

    # Update Hvoigt[i,j,k] for every cell:
    Update_Hvoigt!( Hvoigt, ν, νC, T )

    # Update the normalized emission line profile ϕ_ν for every cell:
    Update_ϕν!( ϕ_ν, Δν, Hvoigt )

    # Turn on/off recoil depending on user choice:
    if recoil
        REC_ONOFF = 1.0
    else
        REC_ONOFF = 0.0
    end

    @inbounds begin 
        @threads for i in dN_r
            for j in dN_Z
                Δν_D       = b(T[i,j]) * νLyα_over_c                     # Doppler width Δν_D = (b / c) νLyα
                PREFACT    = 0.5 * max( α[i,j], α_Lyα_floor ) * (Δν_D^2)  
                RECOIL_0   = h_over_kB / T[i,j]                          # Recoil factor h / (kB * T)

                for k in dN_ν
                    # The Doppler factor (D:∇u)/c ( → (∇⋅u)/(3c) for isotropic intensity):
                    D_colon_∇u = D_colon_∇u_Cylindrical( D_rr[i,j,k],    D_rZ[i,j,k],  D_ZZ[i,j,k],
                                                         ∂u_r_∂r[i,j],   ∂u_r_∂Z[i,j], ∂u_Z_∂r[i,j],
                                                         ∂u_Z_∂Z[i,j],   u_r_over_r[i,j] )
                    VELOCITY   = D_colon_∇u / c 

                    # Get K and C:
                    Kjp            = PREFACT  * Hvoigt[i, j, k+1]
                    Kj             = PREFACT  * Hvoigt[i, j, k]
                    Kjm            = PREFACT  * Hvoigt[i, j, k-1]
                    Cjp            = VELOCITY * νC[k]  +  REC_ONOFF * PREFACT * Hvoigt[i, j, k+1] * ( RECOIL_0 -  2 / νC[k+1] )
                    Cj             = VELOCITY * νC[k]  +  REC_ONOFF * PREFACT * Hvoigt[i, j, k] * ( RECOIL_0  -  2 / νC[k] ) 
                    Cjm            = VELOCITY * νC[k]  +  REC_ONOFF * PREFACT * Hvoigt[i, j, k-1] * ( RECOIL_0  -  2 / νC[k-1] ) 

                    # Get intervals:
                    Δy_m1          = νC[k+1] - ν[k+1]
                    Δy_m2          = ν[k+1]  - νC[k]
                    Δy_m3          = νC[k]   - ν[k]
                    Δy_m4          = ν[k]    - νC[k-1]

                    # FREQUENCY FACTORS F_{i,j,k}^+, F_{i,j,k}, F_{i,j,k}^- :
                    Inv_Δν         = 1 / Δν[k] 
    
                    Fij_kp[i,j,k]  = A_mplus( Inv_Δν, Δy_m1, Δy_m2, Cj, Kj, Cjp, Kjp )
       
                    Fij_k[i,j,k]   = A_m( Inv_Δν, Inv_Δν, Δy_m1, Δy_m2, Δy_m3, Δy_m4, 
                                          Cj, Kj, Cjp, Kjp, Cjm,  Kjm )
    
                    Fij_km[i,j,k]  = A_mminus( Inv_Δν, Δy_m3, Δy_m4, Cj, Kj, Cjm, Kjm )    

                end # End of k-loop (frequency)
            end # End of j-loop 
        end # End of i-loop 

    end # End of inbounds 

end

