
""" FILE:    FreqFactorsAndGrid.jl

    PURPOSE: Contains functions for setting up a frequency grid,
             and computing frequency leakage coefficients (F_{i,j,k}, F_{i,j,k}^±). 
"""

using .ConstantsAndFunctions   # Needed constants & functions
using  Base.Threads            # For CPU acceleration
using  Roots                   # Needed for root-finding in the frequency grid setup 

export UpdateVoigtAndEmissionProfile!, UpdateFreqFactors!


""" This function updates the Voigt profile and emission profile for all cells. """
function UpdateVoigtAndEmissionProfile!( Lyα::LyαArrays,                 # The Lyα state/arrays,
                                           T::AbstractArray{Float64, 2}  # Gas temperature for every cell 
                                         )
    # Update Hvoigt[i,j,k] for every cell:
    Update_Hvoigt!( Lyα.Hvoigt, Lyα.ν, Lyα.νC, T )

    # Update the normalized emission line profile ϕ_ν for every cell:
    Update_ϕν!( Lyα.ϕ_ν, Lyα.Δν, Lyα.Hvoigt, Lyα.kcore_min, Lyα.kcore_max )

    return nothing 
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
    N_r, N_Z, N_ν = size( Lyα.Hvoigt )

    # Needed intervals:
    dN_r      = 1:N_r
    dN_Z      = 1:N_Z
    dN_ν      = 1:N_ν
    dN_ν_int  = 2:(N_ν-1)   # Only the interior frequency bins 

    # Update Eddington factors D to ensure they use the latest values (from post-advection):
    @threads for k in dN_ν
        Update_D_Edd_Serial!( Lyα.D_rr_Lyα[:,:,k], Lyα.D_ZZ_Lyα[:,:,k], Lyα.D_rZ_Lyα[:,:,k],
                              Lyα.H_r_Lyα[:,:,k],  Lyα.H_Z_Lyα[:,:,k],  Lyα.J_Lyα[:,:,k] )
    end

    # Turn on/off recoil depending on user choice:
    if Lyα.Recoil
        REC_ONOFF = 1.0
    else
        REC_ONOFF = 0.0
    end

    @inbounds begin 
        @threads for i in dN_r
            for j in dN_Z
                Δν_D       = b(T[i,j]) * νLyα_over_c                                  # Doppler width Δν_D = (b / c) νLyα
                PREFACT    = 0.5 * max( Lyα.α_Lyα[i,j], Lyα.α_Lyα_floor ) * (Δν_D^2)  
                RECOIL_0   = h_over_kB / T[i,j]                                       # Recoil factor h / (kB * T)

                for k in dN_ν_int
                    # The Doppler factor (D:∇u)/c ( → (∇⋅u)/(3c) for isotropic intensity):
                    Drr        = Float64( Lyα.D_rr_Lyα[i,j,k] )
                    DrZ        = Float64( Lyα.D_rZ_Lyα[i,j,k] )
                    DZZ        = Float64( Lyα.D_ZZ_Lyα[i,j,k] )
                    D_colon_∇u = D_colon_∇u_Cylindrical( Drr,           DrZ,             DZZ,
                                                         ∂u_r_∂r[i,j],  ∂u_r_∂Z[i,j],    ∂u_Z_∂r[i,j],
                                                         ∂u_Z_∂Z[i,j],  u_r_over_r[i,j] )
                    VELOCITY   = D_colon_∇u * inv_c 

                    # Get K and C:
                    Kjp            = PREFACT  * Lyα.Hvoigt[i, j, k+1] 
                    Kj             = PREFACT  * Lyα.Hvoigt[i, j, k]    
                    Kjm            = PREFACT  * Lyα.Hvoigt[i, j, k-1]
                    Cjp            = VELOCITY * Lyα.νC[k]  +  REC_ONOFF * PREFACT * Lyα.Hvoigt[i, j, k+1] * ( RECOIL_0 -  2 / Lyα.νC[k+1] )
                    Cj             = VELOCITY * Lyα.νC[k]  +  REC_ONOFF * PREFACT * Lyα.Hvoigt[i, j, k] * ( RECOIL_0  -  2 / Lyα.νC[k] ) 
                    Cjm            = VELOCITY * Lyα.νC[k]  +  REC_ONOFF * PREFACT * Lyα.Hvoigt[i, j, k-1] * ( RECOIL_0  -  2 / Lyα.νC[k-1] ) 

                    # Get intervals:
                    Δy_m1          = Lyα.νC[k+1] - Lyα.ν[k+1]
                    Δy_m2          = Lyα.ν[k+1]  - Lyα.νC[k]
                    Δy_m3          = Lyα.νC[k]   - Lyα.ν[k]
                    Δy_m4          = Lyα.ν[k]    - Lyα.νC[k-1]

                    # FREQUENCY FACTORS F_{i,j,k}^+, F_{i,j,k}, F_{i,j,k}^- :
                    Inv_Δν             = 1 / Lyα.Δν[k] 
    
                    Lyα.Fij_kp[i,j,k]  = A_mplus( Inv_Δν, Δy_m1, Δy_m2, Cj, Kj, Cjp, Kjp )
       
                    Lyα.Fij_k[i,j,k]   = A_m( Inv_Δν, Inv_Δν, Δy_m1, Δy_m2, Δy_m3, Δy_m4, 
                                              Cj, Kj, Cjp, Kjp, Cjm,  Kjm )
    
                    Lyα.Fij_km[i,j,k]  = A_mminus( Inv_Δν, Δy_m3, Δy_m4, Cj, Kj, Cjm, Kjm )    

                end # End of k-loop (frequency)
            end # End of j-loop 
        end # End of i-loop 

    end # End of inbounds 

    return nothing 
end

