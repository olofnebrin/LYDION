
""" FILE:       PhotoThermoChemistry.jl

    PURPOSE:    Contains the full thermochemistry + dust + stellar RT update.
    
"""

using .Structs: LyαArrays, StellarRTArrays, ChemistryState, HydrodynamicsState
using .ConstantsAndFunctions
using  Base.Threads               # For CPU acceleration
using  SpecialFunctions           # For erf

include(joinpath(@__DIR__, "..", "FreqFactorsAndGrid.jl"))
include(joinpath(@__DIR__, "..", "LymanAlphaEmission.jl"))
include(joinpath(@__DIR__, "..", "LymanAlphaDestruction.jl"))
include("./GasChemistryRates.jl")
include("./UpdateChemistry.jl")
include("./GasCoolingRates.jl")
include("./StellarEmissionAbsorption.jl")  # For band opacities & stellar luminosities 


const CFL_RT::Float64                  = 0.5   # CFL factor that determines the advection time-step for stellar & Lyα RT (Default: 0.5. NOTE: Must be < 1)
const DEBUG_PhotoThermoChemistry::Bool = true  # Allow safety checks for physical values etc. Cheap & good for tracking bugs.


# Export functions: 
export UpdateRTandThermoChemistry!, StellarRTArrays, 
       GenerateStellarRTArrays, FindRSL_StellarRT, FindRSL_LyαRT



const ϵ_RSL::Float64 = 5.0         # RSL is ≥ ϵ_RSL × max(|u|, c_s) (Default: 5.0)
const c̃_min::Float64 = 100 * kms   # Lower bound on RSL, regardless of gas conditions (Default: 100 km/s)

""" This function determines a reduced speed of light c̃ for 
    stellar RT, such that it is comfortably larger than all gas 
    velocities in the simulation. """
function FindRSL_StellarRT( c̃::Float64,                   # Old reduced speed of light
                          u_r::AbstractArray{Float64, 2}, # r-direction gas velocity 
                          u_Z::AbstractArray{Float64, 2}, # Z-direction gas velocity 
                            P::AbstractArray{Float64, 2}, # Gas pressure 
                            ρ::AbstractArray{Float64, 2}, # Gas density
                          γad::AbstractArray{Float64, 2}  # Gas adiabatic index 
                           )

    N_r, N_Z = size( ρ )
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z

    c̃_new    = 0.0   # Initialize value for the new RSL 
    for j in dN_Z 
        local_max_c̃  =  c̃_new   # Local variable for tracking largest c̃

        @inbounds for i in dN_r 
            # Gas sound speed:
            c_s = sqrt( γad[i,j] * P[i,j] / ρ[i,j] )

            # Compute minimum RSL for accuracy in this cell to be ~ ϵ_RSL × gas velocity:
            c̃_min_cell  = ϵ_RSL * max( abs(u_r[i,j]), abs(u_Z[i,j]), c_s )

            # Update the largest RSL we've found:
            local_max_c̃ = max( local_max_c̃, c̃_min_cell )
        end

        c̃_new = max( c̃_new, local_max_c̃ )
    end

    # Set the new RSL:
    c̃_new = min( max( c̃_new, c̃_min ), c )

    # To avoid rapid changes in c̃ if the newly calculated RSL is small, 
    # we further limit the maximum reduction to 25 % from the last time-step:
    if c̃_new < c̃
        c̃_new = max( c̃_new, 0.75 * c̃ )
    end

    return c̃_new
end


const X_over_mH::Float64 = X / mH

function FindRSL_LyαRT( c̃_Lyα::Float64,                   # Old reduced speed of light
                          u_r::AbstractArray{Float64, 2}, # r-direction gas velocity 
                          u_Z::AbstractArray{Float64, 2}, # Z-direction gas velocity 
                            P::AbstractArray{Float64, 2}, # Gas pressure 
                            ρ::AbstractArray{Float64, 2}, # Gas density
                            T::AbstractArray{Float64, 2}, # Gas temperature 
                         xHII::AbstractArray{Float64, 2}, # H II abundance fraction 
                          xH2::AbstractArray{Float64, 2}, # H₂ abundance fraction
                          γad::AbstractArray{Float64, 2}, # Gas adiabatic index
                            r::AbstractArray{Float64, 1},
                            Z::AbstractArray{Float64, 1}
                           )
    
    N_r, N_Z  = size( ρ )
    dN_r      = 1:N_r
    dN_Z      = 1:N_Z

    c̃_Lyα_new = 0.0   # Initialize value for the new RSL 
    for j in dN_Z 
        local_max_c̃  =  c̃_Lyα_new   # Local variable for tracking largest c̃

        # Zell width along Z-axis:
        ΔZ = Z[j+1] - Z[j]

        @inbounds for i in dN_r 
            # Cell width along r-axis:
            Δr  = r[i+1] - r[i]
            # Gas sound speed:
            c_s = sqrt( γad[i,j] * P[i,j] / ρ[i,j] )
            # H I number density:
            xHI = max( 1 - xHII[i,j] - 2 * xH2[i,j], 0.0 )
            nHI = xHI * X_over_mH * ρ[i,j]
            
            # Estimate of a × max(τ_cell):
            aτ_cell     = a(T[i,j]) * nHI * σ0(T[i,j]) * max( Δr, ΔZ )

            # Compute minimum RSL for accuracy in this cell to be ~ ϵ_RSL × (t_trap/t_light) × gas velocity:
            c̃_min_cell  = ϵ_RSL * ( 1 + aτ_cell^(1/3) ) * max( abs(u_r[i,j]), abs(u_Z[i,j]), c_s )

            # Update the largest RSL we've found:
            local_max_c̃ = max( local_max_c̃, c̃_min_cell )
        end

        c̃_Lyα_new = max( c̃_Lyα_new, local_max_c̃ )
    end

    # Set the new RSL:
    c̃_Lyα_new = min( max( c̃_Lyα_new, c̃_min ), c )

    # To avoid rapid changes in c̃ if the newly calculated RSL is small, 
    # we further limit the maximum reduction to 25 % from the last time-step:
    if c̃_Lyα_new < c̃_Lyα
        c̃_Lyα_new = max( c̃_Lyα_new, 0.75 * c̃_Lyα )
    end

    return c̃_Lyα_new 
end


""" The total gas heating rate (erg cm⁻³ s⁻ꜝ) from 
    absorption of ionizing and other photons. """
function HeatingRate_total( J_opt::Float64, 
                            J_FUV::Float64, 
                             J_LW::Float64,
                           J_EUV1::Float64,
                           J_EUV2::Float64,
                           J_EUV3::Float64,
                            T_old::Float64,
                           ne_old::Float64,
                          nHI_old::Float64,
                          nH2_old::Float64,
                         nHeI_old::Float64,
                        nHeII_old::Float64,
                    SelfShielding::Float64,
                            e_Lyα::Float64 = 0.0
                             )

    (  HeatingRate_Compton(J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, ne_old, T_old, e_Lyα )
    +  HeatingRate_LW( SelfShielding * J_LW, nH2_old, nHI_old, T_old )
    +  HeatingRate_EUV1( J_EUV1, nHI_old, nH2_old ) 
    +  HeatingRate_EUV2( J_EUV2, nHI_old, nH2_old, nHeI_old ) 
    +  HeatingRate_EUV3( J_EUV3, nHI_old, nH2_old, nHeI_old, nHeII_old )  )

end


"""  This function estimates the density gradient magnitude |∇ρ|,
     for use in the Sobolev-like estimate of column density path lengths. """ 
@inline function Estimate_∇ρ( i::Int, j::Int,
                              ρ::AbstractArray{Float64, 2},
                              rCoM::AbstractArray{Float64, 1},
                              ZCoM::AbstractArray{Float64, 1} )

    N_r, N_Z = size( ρ )
    ∂ρ∂r     = 0.0
    ∂ρ∂Z     = 0.0

    if i ≥ 2 && i ≤ (N_r-1)
        # For interior points:
        Δr_L = rCoM[i]   - rCoM[i-1]
        Δr_R = rCoM[i+1] - rCoM[i]

        ∂ρ∂r = MinMod( (ρ[i,j] - ρ[i-1,j]) / Δr_L, 
                       (ρ[i+1,j] - ρ[i,j]) / Δr_R )
    elseif i == 1
        # For the innermost cell:
        Δr_R = rCoM[i+1] - rCoM[i]
        ∂ρ∂r = (ρ[i+1,j] - ρ[i,j]) / Δr_R
    elseif i == N_r 
        # For the outermost cell:
        Δr_L  = rCoM[i] - rCoM[i-1]
        ∂ρ∂r  = (ρ[i,j] - ρ[i-1,j]) / Δr_L
    end

    if j ≥ 2 && j ≤ (N_Z-1)
        # For interior points:
        ΔZ_L = ZCoM[j]   - ZCoM[j-1]
        ΔZ_R = ZCoM[j+1] - ZCoM[j]

        ∂ρ∂Z = MinMod( (ρ[i,j] - ρ[i,j-1]) / ΔZ_L, 
                       (ρ[i,j+1] - ρ[i,j]) / ΔZ_R )
    elseif j == 1
        # For the innermost cell:
        ΔZ_R = ZCoM[j+1] - ZCoM[j]
        ∂ρ∂Z = (ρ[i,j+1] - ρ[i,j]) / ΔZ_R
    elseif j == N_Z
        # For the outermost cell:
        ΔZ_L  = ZCoM[j] - ZCoM[j-1]
        ∂ρ∂Z  = (ρ[i,j] - ρ[i,j-1]) / ΔZ_L
    end

    # Compute the magnitude of the gradient:
    ∇ρ = sqrt( ∂ρ∂r * ∂ρ∂r  +  ∂ρ∂Z * ∂ρ∂Z )

    return ∇ρ
end


const S_H2_1::Float64 = 0.965;    const S_H2_2::Float64  = 0.035
const c_SH2::Float64  = 8.5e-4; const N_H2crit::Float64  = 5e14

""" The H₂ self-shielding factor, as fitted by 
    Draine & Bertoldi (1996). """
@inline function S_H2( N_H2::Float64, 
                          T::Float64
                       )
    T    = clamp( T, 1.0, 1e5 )
    x    = N_H2 / N_H2crit
    F    = sqrt( 1 + x )
    b5   = sqrt( 2 * kB * T / mH2 ) / kms 
    S_H2 = S_H2_1 / (1 + x/b5)^2  +  S_H2_2 * exp( - c_SH2 * F ) / F
    return S_H2
end


""" The HD self-shielding factor, same fit as for H₂. """
@inline function S_HD( N_HD::Float64, 
                          T::Float64
                       )
    T    = clamp( T, 1.0, 1e5 )
    x    = N_HD / N_H2crit
    F    = sqrt( 1 + x )
    b5   = sqrt( 2 * kB * T / (3 * mH) ) / kms  # Mass of HD is ≃ 3 × mH
    S_HD = S_H2_1 / (1 + x/b5)^2  +  S_H2_2 * exp( - c_SH2 * F ) / F
    return S_HD
end


const N_H2_HDcrit = 2.34e19; const c_H2_HD = 5.2e-3; const p_H2_HD = 0.238 

""" The shielding of HD by H₂. See Wolcott-Green & Haiman (2011). """
@inline function S_H2_HD( N_H2::Float64 )
    x       = N_H2 / N_H2_HDcrit
    S_H2_HD = exp( - c_H2_HD * x ) / (1 + x)^p_H2_HD
    return S_H2_HD
end


const tol_Abs::Float64 = 0.05 

function AbsorptionStep( Δt::Float64,
                          c̃::Float64,
                        nHI::Float64,          # H I number density 
                        nH2::Float64,          # H₂ number density   
                       nHeI::Float64,          # He I number density 
                      nHeII::Float64,          # He II number density   

                  J_opt_old::Float64 = 0.0,
                   dJ_optdt::Float64 = 0.0,
                  J_FUV_old::Float64 = 0.0,
                   dJ_FUVdt::Float64 = 0.0,

                   J_LW_old::Float64 = 0.0,
                    dJ_LWdt::Float64 = 0.0,
                 H_r_LW_old::Float64 = 0.0,
                 H_Z_LW_old::Float64 = 0.0,
                   dHr_LWdt::Float64 = 0.0,
                   dHZ_LWdt::Float64 = 0.0,

                 J_EUV1_old::Float64 = 0.0,
                  dJ_EUV1dt::Float64 = 0.0,
               H_r_EUV1_old::Float64 = 0.0,
               H_Z_EUV1_old::Float64 = 0.0,
                 dHr_EUV1dt::Float64 = 0.0,
                 dHZ_EUV1dt::Float64 = 0.0,

                 J_EUV2_old::Float64 = 0.0,
                  dJ_EUV2dt::Float64 = 0.0,
               H_r_EUV2_old::Float64 = 0.0,
               H_Z_EUV2_old::Float64 = 0.0,
                 dHr_EUV2dt::Float64 = 0.0,
                 dHZ_EUV2dt::Float64 = 0.0,

                 J_EUV3_old::Float64 = 0.0,
                  dJ_EUV3dt::Float64 = 0.0,
               H_r_EUV3_old::Float64 = 0.0,
               H_Z_EUV3_old::Float64 = 0.0,
                 dHr_EUV3dt::Float64 = 0.0,
                 dHZ_EUV3dt::Float64 = 0.0
                        )

    inv_c̃      = 1 / c̃   # Convenient constant 

    # Initialize gas momentum change from stellar radiation pressure:
    Δρu_r_rad  = 0.0
    Δρu_Z_rad  = 0.0

    # ------- Get opacities (cm⁻¹) in all bands ----------------------------------------------
    αc_opt     =   0.0                                        
    αc_FUV     =   0.0                                         
    αc_LW      =   nHI  * σHI_LW  +  nH2 * σH2_LW    

    αc_EUV1    = ( nHI * σHI_EUV1  +  nH2 * σH2_EUV1  )

    αc_EUV2    =   nHI * σHI_EUV2  +  nH2 * σH2_EUV2  +  nHeI * σHeI_EUV2  

    αc_EUV3    = ( nHI * σHI_EUV3  +  nH2 * σH2_EUV3  +  nHeI * σHeI_EUV3 
                 + nHeII * σHeII_EUV3  )    
                    
    #######################################################################
    #######                                                         #######
    #######                       O P T I C A L                     #######
    #######                                                         #######
    #######################################################################
        
    J_opt         = max( AbsorptionUpdateJorH( J_opt_old, Δt, dJ_optdt, αc_opt, c̃ ),
                         0.0 )

    H_r_opt       = 0.0
    H_Z_opt       = 0.0

    #######################################################################
    #######                                                         #######
    #######                       F A R - U V                       #######
    #######                                                         #######
    #######################################################################

    J_FUV         = max( AbsorptionUpdateJorH( J_FUV_old, Δt, dJ_FUVdt, αc_FUV, c̃ ),
                         0.0 )

    H_r_FUV       = 0.0
    H_Z_FUV       = 0.0

    #######################################################################
    #######                                                         #######
    #######                 L Y M A N - W E R N E R                 #######
    #######                                                         #######
    #######################################################################

    J_LW        = max( AbsorptionUpdateJorH( J_LW_old, Δt, dJ_LWdt, αc_LW, c̃ ), 0.0 )
    H_r_LW      = AbsorptionUpdateJorH( H_r_LW_old, Δt, dHr_LWdt, αc_LW, c̃ )
    H_Z_LW      = AbsorptionUpdateJorH( H_Z_LW_old, Δt, dHZ_LWdt, αc_LW, c̃ )

    # Re-normalize so that H < J:
    H_LW_mag    = sqrt( H_r_LW * H_r_LW  +  H_Z_LW * H_Z_LW )
    if H_LW_mag > J_LW
        f_LW   = H_LW_mag / J_LW
        H_r_LW = H_r_LW   / f_LW
        H_Z_LW = H_Z_LW   / f_LW 
    end

    # Compute relative changes in LW J, H_r, and H_Z:
    ΔJ_LW       = abs( J_LW   - J_LW_old )   / max( J_LW_old,        J_floor )
    ΔH_r_LW     = abs( H_r_LW - H_r_LW_old ) / max( abs(H_r_LW_old), H_floor )
    ΔH_Z_LW     = abs( H_Z_LW - H_Z_LW_old ) / max( abs(H_Z_LW_old), H_floor )

    ΔLW = max( ΔJ_LW, ΔH_r_LW, ΔH_Z_LW )
    if ΔLW > tol_Abs  
        # Change in LW was too large. Reject step: 
        REDO = true 
        return REDO, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0,
               Inf64
    end

    # Add radiation pressure momentum contribution:
    Δρu_r_rad += Four_π_over_c * ( inv_c̃ * (H_r_LW_old - H_r_LW)  +  inv_c̃ * Δt * dHr_LWdt )
    Δρu_Z_rad += Four_π_over_c * ( inv_c̃ * (H_Z_LW_old - H_Z_LW)  +  inv_c̃ * Δt * dHZ_LWdt )

    #######################################################################
    #######                                                         #######
    #######                         E U V 1                         #######
    #######                                                         #######
    #######################################################################

    J_EUV1        = max( AbsorptionUpdateJorH( J_EUV1_old, Δt, dJ_EUV1dt, αc_EUV1, c̃ ), 0.0 )
    H_r_EUV1      = AbsorptionUpdateJorH( H_r_EUV1_old, Δt, dHr_EUV1dt, αc_EUV1, c̃ )
    H_Z_EUV1      = AbsorptionUpdateJorH( H_Z_EUV1_old, Δt, dHZ_EUV1dt, αc_EUV1, c̃ )

    # Re-normalize so that H < J:
    H_EUV1_mag    = sqrt( H_r_EUV1 * H_r_EUV1  +  H_Z_EUV1 * H_Z_EUV1 )
    if H_EUV1_mag > J_EUV1
        f_EUV1   = H_EUV1_mag / J_EUV1
        H_r_EUV1 = H_r_EUV1   / f_EUV1
        H_Z_EUV1 = H_Z_EUV1   / f_EUV1 
    end

    # Compute relative changes in EUV1 J, H_r, and H_Z:
    ΔJ_EUV1       = abs( J_EUV1   - J_EUV1_old )   / max( J_EUV1_old,        J_floor )
    ΔH_r_EUV1     = abs( H_r_EUV1 - H_r_EUV1_old ) / max( abs(H_r_EUV1_old), H_floor )
    ΔH_Z_EUV1     = abs( H_Z_EUV1 - H_Z_EUV1_old ) / max( abs(H_Z_EUV1_old), H_floor )

    ΔEUV1 = max( ΔJ_EUV1, ΔH_r_EUV1, ΔH_Z_EUV1 )
    if ΔEUV1 > tol_Abs  
        # Change in EUV1 was too large. Reject step: 
        REDO = true 
        return REDO, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0,
               Inf64
    end

    # Add radiation pressure momentum contribution:
    Δρu_r_rad += Four_π_over_c * ( inv_c̃ * (H_r_EUV1_old - H_r_EUV1)  +  inv_c̃ * Δt * dHr_EUV1dt )
    Δρu_Z_rad += Four_π_over_c * ( inv_c̃ * (H_Z_EUV1_old - H_Z_EUV1)  +  inv_c̃ * Δt * dHZ_EUV1dt )

    #######################################################################
    #######                                                         #######
    #######                         E U V 2                         #######
    #######                                                         #######
    #######################################################################

    J_EUV2        = max( AbsorptionUpdateJorH( J_EUV2_old, Δt, dJ_EUV2dt, αc_EUV2, c̃ ), 0.0 )
    H_r_EUV2      = AbsorptionUpdateJorH( H_r_EUV2_old, Δt, dHr_EUV2dt, αc_EUV2, c̃ )
    H_Z_EUV2      = AbsorptionUpdateJorH( H_Z_EUV2_old, Δt, dHZ_EUV2dt, αc_EUV2, c̃ )

    # Re-normalize so that H < J:
    H_EUV2_mag    = sqrt( H_r_EUV2 * H_r_EUV2  +  H_Z_EUV2 * H_Z_EUV2 )
    if H_EUV2_mag > J_EUV2
        f_EUV2   = H_EUV2_mag / J_EUV2
        H_r_EUV2 = H_r_EUV2   / f_EUV2
        H_Z_EUV2 = H_Z_EUV2   / f_EUV2 
    end

    # Compute relative changes in EUV2 J, H_r, and H_Z:
    ΔJ_EUV2       = abs( J_EUV2   - J_EUV2_old )   / max( J_EUV2_old,        J_floor )
    ΔH_r_EUV2     = abs( H_r_EUV2 - H_r_EUV2_old ) / max( abs(H_r_EUV2_old), H_floor )
    ΔH_Z_EUV2     = abs( H_Z_EUV2 - H_Z_EUV2_old ) / max( abs(H_Z_EUV2_old), H_floor )

    ΔEUV2 = max( ΔJ_EUV2, ΔH_r_EUV2, ΔH_Z_EUV2 )
    if ΔEUV2 > tol_Abs 
        # Change in EUV2 was too large. Reject step: 
        REDO = true 
        return REDO, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0,
               Inf64
    end

    # Add radiation pressure momentum contribution:
    Δρu_r_rad += Four_π_over_c * ( inv_c̃ * (H_r_EUV2_old - H_r_EUV2)  +  inv_c̃ * Δt * dHr_EUV2dt )
    Δρu_Z_rad += Four_π_over_c * ( inv_c̃ * (H_Z_EUV2_old - H_Z_EUV2)  +  inv_c̃ * Δt * dHZ_EUV2dt )

    #######################################################################
    #######                                                         #######
    #######                         E U V 3                         #######
    #######                                                         #######
    #######################################################################

    J_EUV3        = max( AbsorptionUpdateJorH( J_EUV3_old, Δt, dJ_EUV3dt, αc_EUV3, c̃ ), 0.0 )
    H_r_EUV3      = AbsorptionUpdateJorH( H_r_EUV3_old, Δt, dHr_EUV3dt, αc_EUV3, c̃ )
    H_Z_EUV3      = AbsorptionUpdateJorH( H_Z_EUV3_old, Δt, dHZ_EUV3dt, αc_EUV3, c̃ )

    # Re-normalize so that H < J:
    H_EUV3_mag    = sqrt( H_r_EUV3 * H_r_EUV3  +  H_Z_EUV3 * H_Z_EUV3 )
    if H_EUV3_mag > J_EUV3
        f_EUV3   = H_EUV3_mag / J_EUV3
        H_r_EUV3 = H_r_EUV3 / f_EUV3
        H_Z_EUV3 = H_Z_EUV3 / f_EUV3 
    end

    # Compute relative changes in EUV2 J, H_r, and H_Z:
    ΔJ_EUV3       = abs( J_EUV3   - J_EUV3_old )   / max( J_EUV3_old,        J_floor )
    ΔH_r_EUV3     = abs( H_r_EUV3 - H_r_EUV3_old ) / max( abs(H_r_EUV3_old), H_floor )
    ΔH_Z_EUV3     = abs( H_Z_EUV3 - H_Z_EUV3_old ) / max( abs(H_Z_EUV3_old), H_floor )

    ΔEUV3 = max( ΔJ_EUV3, ΔH_r_EUV3, ΔH_Z_EUV3 )
    if ΔEUV3 > tol_Abs 
        # Change in EUV3 was too large. Reject step:
        REDO = true 
        return REDO, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
               0.0, 0.0,
               Inf64
    end

    # Add radiation pressure momentum contribution:
    Δρu_r_rad += Four_π_over_c * ( inv_c̃ * (H_r_EUV3_old - H_r_EUV3)  +  inv_c̃ * Δt * dHr_EUV3dt )
    Δρu_Z_rad += Four_π_over_c * ( inv_c̃ * (H_Z_EUV3_old - H_Z_EUV3)  +  inv_c̃ * Δt * dHZ_EUV3dt )


    # Get maximum change among all bands (for later determination of next sub-cycle time-step):
    ΔStellarRT  = max( ΔLW, ΔEUV1, ΔEUV2, ΔEUV3 )

    # In case the changes in J and H were small enough, we accept the update and return 
    # all computed values: 
    REDO = false
    return REDO, J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, 
           H_r_opt, H_r_FUV, H_r_LW, H_r_EUV1, H_r_EUV2, H_r_EUV3,
           H_Z_opt, H_Z_FUV, H_Z_LW, H_Z_EUV1, H_Z_EUV2, H_Z_EUV3,
           Δρu_r_rad, Δρu_Z_rad,
           ΔStellarRT
end


const tol_HeatCool::Float64       = 0.05 
const Quantity_threshold::Float64 = 1e-5
const ϵ_ΔT::Float64               = 1e-3       # Use step-size ϵ_ΔT * T_old to estimate ∂Λ/∂T at T = T_old
const Δt_e_max::Float64           = 1e7 * yr   # Maximum time-step to consider for heating/cooling

""" This function performs a heating/cooling update 
    over a step Δt. It solves:
    
    ∂e/∂t = Γ(e) - Λ(e) 
          ≈ [ Γ(eⁿ) - Λ(eⁿ) ]  +  ( ∂Γ/∂e + ∂Λ/∂e ) × (e - eⁿ)
          
    The integration is exact of the linearized equation, and because of this,
    the overall solution e^(n+1) is second-order accurate. """
@inline function HeatingCoolingStep( Δt::Float64, 
                                  e_old::Float64,
                                    γad::Float64,
                                      n::Float64,
                                    nHI::Float64,
                                    nH2::Float64,
                                   nHII::Float64,
                                   nHeI::Float64,
                                  nHeII::Float64,
                                 nHeIII::Float64,
                                    nHD::Float64,
                                     ne::Float64,
                                  J_opt::Float64,
                                  J_FUV::Float64,
                                   J_LW::Float64,
                                 J_EUV1::Float64,
                                 J_EUV2::Float64,
                                 J_EUV3::Float64,
                          SelfShielding::Float64,
                                      z::Float64 = 0.0,
                                  CaseB::Bool    = true
                                    )

    Conv_e_T =  ( γad - 1 ) / (n * kB)   # Conversion factor for e → T 
    T_old    =  e_old * Conv_e_T         # Get temperature T_old corresponding to e_old 

    # Get heating rate (from photoelectric heating, Compton heating off the (non-CMB) radiation field,
    # photo-ionizations/dissociations, UV pumping, and cosmic ray heating):
    HeatingRate = HeatingRate_total( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3,
                                     T_old, ne, nHI, nH2, nHeI, nHeII, 
                                     SelfShielding ) 

    # Compute the cooling rate at e_old:
    CoolingRate = Λ( T_old, nHI, nHII, nHeI, nHeII, nHeIII, ne,
                     nH2, nHD, z, CaseB )

    # Tolerance cooling/heating time-scale for accuracy purposes:
    Δt_e = min( tol_HeatCool * e_old / abs( HeatingRate - CoolingRate ), Δt_e_max )

    if Δt > Δt_e
        # Reject step if the time-step is too large:
        REDO = true 
        return REDO, 0.0, Inf, Δt_e
    end

    # Compute ∂Λ/∂e using a 4th-order central difference approximation:
    ΔT       = ϵ_ΔT * T_old
    T_p2     = T_old  +  2 * ΔT
    T_p1     = T_old  +  ΔT 
    T_m1     = T_old  -  ΔT
    T_m2     = T_old  -  2 * ΔT  
    inv_12ΔT = 1 / ( 12 * ΔT )     # Used twice below for the central difference derivatives 

    CoolingRate_p2 = Λ( T_p2, nHI, nHII, nHeI, nHeII, nHeIII, ne,
                        nH2, nHD, z, CaseB )

    CoolingRate_p1 = Λ( T_p1, nHI, nHII, nHeI, nHeII, nHeIII, ne,
                        nH2, nHD, z, CaseB )

    CoolingRate_m1 = Λ( T_m1, nHI, nHII, nHeI, nHeII, nHeIII, ne,
                        nH2, nHD, z, CaseB )

    CoolingRate_m2 = Λ( T_m2, nHI, nHII, nHeI, nHeII, nHeIII, ne,
                        nH2, nHD, z, CaseB )

    # Put it all together to get ∂Λ/∂e:
    ∂CoolingRate∂T = ( - CoolingRate_p2  +  8 * CoolingRate_p1  -  8 * CoolingRate_m1  
                       + CoolingRate_m2 ) * inv_12ΔT

    ∂CoolingRate∂e = ∂CoolingRate∂T * Conv_e_T  # Convert from ∂/∂T to ∂/∂e

    # Next, compute ∂Γ/∂e using a 4th-order central difference approximation: 
    HeatingRate_p2 = HeatingRate_total( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3,
                                        T_p2, ne, nHI, nH2, nHeI, nHeII, 
                                        SelfShielding )

    HeatingRate_p1 = HeatingRate_total( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3,
                                        T_p1, ne, nHI, nH2, nHeI, nHeII, 
                                        SelfShielding )

    HeatingRate_m1 = HeatingRate_total( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3,
                                        T_m1, ne, nHI, nH2, nHeI, nHeII, 
                                        SelfShielding )

    HeatingRate_m2 = HeatingRate_total( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3,
                                        T_m2, ne, nHI, nH2, nHeI, nHeII, 
                                        SelfShielding )

    # Put it all together to get ∂Γ/∂e:
    ∂HeatingRate∂T = ( - HeatingRate_p2  +  8 * HeatingRate_p1  -  8 * HeatingRate_m1  
                       + HeatingRate_m2 ) * inv_12ΔT

    ∂HeatingRate∂e = ∂HeatingRate∂T * Conv_e_T  # Convert from ∂/∂T to ∂/∂e

    # Finally, use all of this to update e:
    ∂HeatCool = ∂HeatingRate∂e - ∂CoolingRate∂e
    Quantity  = ∂HeatCool * Δt
    if abs( Quantity ) > Quantity_threshold
        # Exact solution of linearized equation:
        e_new = e_old  +  ( HeatingRate - CoolingRate ) * ( exp( Quantity ) - 1 ) / ∂HeatCool
    else
        # For small arguments in the exponential, we instead Taylor expand to 2:nd order:
        e_new = e_old  +  ( HeatingRate - CoolingRate ) * ( 1  +  0.5 * Quantity ) * Δt
    end

    # Check that the relative change in e was sufficiently small, and that 
    # other stability/accuracy checks are satisfied:
    Δe    = abs( e_new  -  e_old ) / e_old

    if Δe > tol_HeatCool || e_new ≤ 0.0 || !isfinite(e_new)
        # In this case, the update is deemed inaccurate or unstable, so we reject it:
        REDO = true
        return REDO, e_new, Δe, Δt_e
    else
        # If the update is stable, and the change small, we accept the updated e:
        REDO = false
        return REDO, e_new, Δe, Δt_e
    end
end


const T_floor::Float64          = 1.0             # Enforced temperature floor in the thermochemistry sub-cycling
const T_ceiling::Float64        = 1e9             # Enforced temperature ceiling in the thermochemistry sub-cycling
const ΔL_max::Float64           = 1e3 * pc        # Enforced maximum value of column density length in the estimate of N_H2, N_HD 
const tol_Chem::Float64         = 0.1             # Maximum relative change tolerance per sub-cycle time-step for the chemistry
const tol_enlargestep::Float64  = 0.5 * tol_Chem  # Recommend a 2 × larger sub-cycle time-step if relative changes were < tol_enlargestep
const J_floor::Float64          = 1e-10           # Maximum stellar J below which we don't care about the 10% rule 
const H_floor::Float64          = 1e-10           # Maximum stellar H below which we don't care about the 10% rule 
const xmin::Float64             = 1e-8            # Minimum abundance fraction below which we don't care about the 10% rule
const max_iter::Int             = 100             # Maximum number of adaptive refinements of sub-cycle time-step 

""" This functions does a single sub-cycle time-step update of
    photon (J and H) absorption, temperature, and chemical abundances.
    Strang-spliting and the 10% rule is used to reduce splitting errors 
    from treating these processes independently:
    
    1. Photon absorption update over half-step (0.5 Δtsub)
    2. Heating/cooling update over half-step (0.5 Δtsub)
    3. Chemistry update over full step (Δtsub)
    4. Heating/cooling update over half-step (0.5 Δtsub)
    5. Photon absorption update over half-step (0.5 Δtsub) """
function ThermoChemistrySubStepUpdate!( icell::Int, jcell::Int,      # Cell (i,j)
                                           Δr::Float64, ΔZ::Float64, # Cell width (for column density estimate)
                                   Δtsubguess::Float64,              # First guess of sub-cycle time-step
                                            c̃::Float64,              # Reduced speed of light for stellar RT
                                        e_old::Float64,              # Gas internal energy density from previous sub-step 
                                            ρ::Float64,              # Gas density 
                                           ∇ρ::Float64,              # Estimated magnitude of density gradient 
                                      xH2_old::Float64,              # H₂ fraction from previous sub-step 
                                     xHII_old::Float64,              # H II fraction from previous sub-step 
                                      xHD_old::Float64,              # HD fraction from previous sub-step 
                                     xDII_old::Float64,              # D II fraction from previous sub-step 
                                    xHeII_old::Float64,              # He II fraction from previous sub-step 
                                   xHeIII_old::Float64,              # He III fraction from previous sub-step 

                                    J_opt_old::Float64 = 0.0,
                                     dJ_optdt::Float64 = 0.0,
                                    J_FUV_old::Float64 = 0.0,
                                     dJ_FUVdt::Float64 = 0.0,

                                     J_LW_old::Float64 = 0.0,
                                      dJ_LWdt::Float64 = 0.0,
                                   H_r_LW_old::Float64 = 0.0,
                                   H_Z_LW_old::Float64 = 0.0,
                                     dHr_LWdt::Float64 = 0.0,
                                     dHZ_LWdt::Float64 = 0.0,

                                   J_EUV1_old::Float64 = 0.0,
                                    dJ_EUV1dt::Float64 = 0.0,
                                 H_r_EUV1_old::Float64 = 0.0,
                                 H_Z_EUV1_old::Float64 = 0.0,
                                   dHr_EUV1dt::Float64 = 0.0,
                                   dHZ_EUV1dt::Float64 = 0.0,

                                   J_EUV2_old::Float64 = 0.0,
                                    dJ_EUV2dt::Float64 = 0.0,
                                 H_r_EUV2_old::Float64 = 0.0,
                                 H_Z_EUV2_old::Float64 = 0.0,
                                   dHr_EUV2dt::Float64 = 0.0,
                                   dHZ_EUV2dt::Float64 = 0.0,

                                   J_EUV3_old::Float64 = 0.0,
                                    dJ_EUV3dt::Float64 = 0.0,
                                 H_r_EUV3_old::Float64 = 0.0,
                                 H_Z_EUV3_old::Float64 = 0.0,
                                   dHr_EUV3dt::Float64 = 0.0,
                                   dHZ_EUV3dt::Float64 = 0.0,

                                            z::Float64 = 0.0,
                                        CaseB::Bool    = true,
                               CoolingHeating::Bool    = true
                                      )

    # Safety checks:
    if DEBUG_PhotoThermoChemistry
        @assert isfinite(ρ) && ρ > 0.0 "ρ = $(ρ) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(e_old)        "e_old = $(e_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"

        @assert isfinite(J_EUV1_old) && J_EUV1_old ≥ 0.0 "J_EUV1_old = $(J_EUV1_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_r_EUV1_old)                   "H_r_EUV1_old = $(H_r_EUV1_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_Z_EUV1_old)                   "H_Z_EUV1_old = $(H_Z_EUV1_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"

        @assert isfinite(J_EUV2_old) && J_EUV2_old ≥ 0.0 "J_EUV2_old = $(J_EUV2_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_r_EUV2_old)                   "H_r_EUV2_old = $(H_r_EUV2_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_Z_EUV2_old)                   "H_Z_EUV2_old = $(H_Z_EUV2_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"

        @assert isfinite(J_EUV3_old) && J_EUV3_old ≥ 0.0 "J_EUV3_old = $(J_EUV3_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_r_EUV3_old)                   "H_r_EUV3_old = $(H_r_EUV3_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_Z_EUV3_old)                   "H_Z_EUV3_old = $(H_Z_EUV3_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"

        @assert isfinite(xH2_old)            "xH2_old = $(xH2_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xHII_old)           "xHII_old = $(xHII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xHD_old)            "xHD_old = $(xHD_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xDII_old)           "xDII_old = $(xDII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xHeII_old)          "xHeII_old = $(xHeII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xHeIII_old)         "xHeIII_old = $(xHeIII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xHII_old + 2 * xH2_old ≤ 1.0 "xHII_old + 2 * xH2_old = $(xHII_old + 2 * xH2_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xDII_old + xHD_old ≤ 1.0     "xDII_old + xHD_old = $(xDII_old + xHD_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xHeII_old + xHeIII_old ≤ 1.0 "xHeII_old + xHeIII_old = $(xHeII_old + xHeIII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
    end


    # --------- Get relevant number densities & abundance fractions  ----------------------------
    nH_old            =   X * ρ / mH
    nD_old            =   D_H_ratio * nH_old
    nHe_old           =   Y * ρ / mHe
    xHI_old           =   max( 1 - xHII_old - 2 * xH2_old, 0.0 )
    xDI_old           =   max( 1 - xDII_old  - xHD_old,    0.0 )
    xHeI_old          =   max( 1 - xHeII_old - xHeIII_old, 0.0 )
    
    nHI_old           =   xHI_old    * nH_old
    nHII_old          =   xHII_old   * nH_old 
    nH2_old           =   xH2_old    * nH_old
    nHD_old           =   xHD_old    * nD_old    # Enters the cooling rate
    nHeI_old          =   xHeI_old   * nHe_old
    nHeII_old         =   xHeII_old  * nHe_old
    nHeIII_old        =   xHeIII_old * nHe_old
    ne_old            =   nHII_old  +  nHeII_old  +  2 * nHeIII_old
    n                 =   nH_old  +  nHe_old  +  ne_old 
                               
    # --------- Get old temperature, and enforce temperature floor ----------------------------
    γad               =   Getγad( xHII_old, xH2_old, xHeII_old, xHeIII_old )
    e_old             =   max( e_old,  n * kB * T_floor / (γad - 1) )
    T_old             =   ( γad - 1 ) * e_old  / ( n * kB )

    # ------------------------------------------------------------------------------------

    # Sobolev-type column length estimate ~ cell width + density variation scale (following 
    # e.g. Hopkins+ 2023, for FIRE-3). We limit ΔL to ≤ ΔL_max to avoid infinities if ∇ρ ~ 0:
    ΔL   = min( min( Δr, ΔZ )  +  ρ / abs(∇ρ),  ΔL_max )

    # Estimated H₂, HD column densities:
    N_H2 = nH2_old * ΔL 
    N_HD = nHD_old * ΔL 

    # Get self-shielding factors:
    SelfShielding       = S_H2( N_H2, T_old ) 
    SelfShielding_HD    = S_HD( N_HD, T_old )
    SelfShielding_H2_HD = S_H2_HD( N_H2 )
    
    Δtsub               = Δtsubguess # Initialize first guess of sub-cycle time-step
    iter::Int           = 0          # Initialize iteration count
    ΔxH2_rel            = 0.0        # Initialize ΔxH2_rel, for printing in case convergence fails 
    ΔxHII_rel           = 0.0        # Initialize ΔxHII_rel, for printing in case convergence fails 
    Δne_rel             = 0.0        # Initialize Δne_rel, for printing in case convergence fails 
    Δe                  = 0.0        # Initialize Δe, for printing in case convergence fails 
    ΔStellarRT          = 0.0        # Initialize ΔStellarRT, for printing in case convergence fails 
    REDO_ABSORPTION     = false
    REDO_HEATINGCOOLING = false
    while iter < max_iter 
        iter       += 1
        Δtsub_half  = 0.5 * Δtsub

        Δρu_r_trial = 0.0  # Initialize the computed change in r-direction gas momentum 
        Δρu_Z_trial = 0.0  # Initialize the computed change in Z-direction gas momentum 

        #######################################################################
        #######                                                         #######
        #######            S T E P  1  :  A B S O R P T I O N           #######
        #######                                                         #######
        #######################################################################

        # First we update J and H for all stellar bands, over a 
        # half-substep Δtsub / 2.
        
        REDO_ABSORPTION, J_opt_trial, J_FUV_trial, J_LW_trial, J_EUV1_trial, J_EUV2_trial, J_EUV3_trial, 
        H_r_opt_trial, H_r_FUV_trial, H_r_LW_trial, H_r_EUV1_trial, H_r_EUV2_trial, H_r_EUV3_trial,
        H_Z_opt_trial, H_Z_FUV_trial, H_Z_LW_trial, H_Z_EUV1_trial, H_Z_EUV2_trial, H_Z_EUV3_trial,
        Δρu_r_trial_1, Δρu_Z_trial_1,
        ΔStellarRT_1 = AbsorptionStep( Δtsub_half, c̃, nHI_old, nH2_old, nHeI_old, nHeII_old,
                                       J_opt_old, dJ_optdt, 
                                       J_FUV_old, dJ_FUVdt, 
                                       J_LW_old, dJ_LWdt, H_r_LW_old, H_Z_LW_old, dHr_LWdt, dHZ_LWdt,
                                       J_EUV1_old, dJ_EUV1dt, H_r_EUV1_old, H_Z_EUV1_old, dHr_EUV1dt, dHZ_EUV1dt,
                                       J_EUV2_old, dJ_EUV2dt, H_r_EUV2_old, H_Z_EUV2_old, dHr_EUV2dt, dHZ_EUV2dt,
                                       J_EUV3_old, dJ_EUV3dt, H_r_EUV3_old, H_Z_EUV3_old, dHr_EUV3dt, dHZ_EUV3dt )

        if REDO_ABSORPTION
            # Change in J or H were too large, so reject update and try a smaller sub-cycle time-step: 
            Δtsub = Δtsub_half
            continue 
        else
            # If changes in J and H were small enough, we accept the trial updated J and H:
            J_opt  = J_opt_trial;  J_FUV  = J_FUV_trial;  J_LW   = J_LW_trial
            J_EUV1 = J_EUV1_trial; J_EUV2 = J_EUV2_trial; J_EUV3 = J_EUV3_trial 

            H_r_opt  = H_r_opt_trial;  H_r_FUV  = H_r_FUV_trial;  H_r_LW   = H_r_LW_trial
            H_r_EUV1 = H_r_EUV1_trial; H_r_EUV2 = H_r_EUV2_trial; H_r_EUV3 = H_r_EUV3_trial
            
            H_Z_opt  = H_Z_opt_trial;  H_Z_FUV  = H_Z_FUV_trial;  H_Z_LW   = H_Z_LW_trial
            H_Z_EUV1 = H_Z_EUV1_trial; H_Z_EUV2 = H_Z_EUV2_trial; H_Z_EUV3 = H_Z_EUV3_trial

            # Also add rad. pressure momentum change:
            Δρu_r_trial += Δρu_r_trial_1
            Δρu_Z_trial += Δρu_Z_trial_1
        end

        #######################################################################
        #######                                                         #######
        #######      S T E P  2  :  H E A T I N G / C O O L I N G       #######
        #######                                                         #######
        #######################################################################

        # Next we do an update of e from heating and cooling, over a 
        # half-substep Δtsub / 2.

        REDO_HEATINGCOOLING, e_new, 
        Δe_1, Δt_e = HeatingCoolingStep( Δtsub_half, e_old, γad, 
                                         n, nHI_old, nH2_old, nHII_old, nHeI_old, nHeII_old,
                                         nHeIII_old, nHD_old, ne_old, 
                                         J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, 
                                         SelfShielding, z, CaseB )

        if REDO_HEATINGCOOLING
            # Change in e was too large or unphysical, so retry with 
            # smaller sub-cycle time-step:
            Δtsub = min( Δtsub_half,  
                         Δt_e )
            continue 
        end 

        e_floor = n * kB * T_floor / (γad - 1)   # e corresponding to temperature floor
        if CoolingHeating 
            e_new = min( max( e_new, e_floor ),
                         n * kB * T_ceiling / (γad - 1) )
        else
            # For some tests (e.g. R-type ionization front) we ignore heating/cooling:
            e_new = e_old
        end

        ##################################################################
        #######                                                    #######
        #######         S T E P  3  :  C H E M I S T R Y           #######
        #######                                                    #######
        ##################################################################

        # Next, update the chemistry over a full sub-step Δtsub.

        # Get photo-ionization/dissociation rates (s⁻¹):
        Γ_LW        =  PhotoDissRate_H2( SelfShielding * J_LW )
        Γpd         =  PhotoDetachRate_Hminus( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3 )
        Γ_LW_HD     =  PhotoDissRate_HD( SelfShielding_HD * SelfShielding_H2_HD * J_LW )
        ΓionH2      =  IonRate_H2( J_EUV1, J_EUV2, J_EUV3 ) 
        ΓionHI      =  IonRate_HI( J_EUV1, J_EUV2, J_EUV3 ) 
        ΓionDI      =  ΓionHI
        ΓionHeI     =  IonRate_HeI( J_EUV2, J_EUV3 ) 
        ΓionHeII    =  IonRate_HeII( J_EUV3 )

        # Get the normalized temperature T / [μ(γ-1)]:
        Tnorm = e_new * mH / ( kB * ρ )

        # Update chemistry:
        ( xH2_new, xHI_new, xHII_new, xHD_new, xDI_new, xDII_new,
          xHeI_new, xHeII_new, xHeIII_new, ne_new ) =
          UpdateChemistrySubStep( Δtsub, Tnorm, ρ, xH2_old, xHII_old, xHD_old, xDII_old, xHeII_old, xHeIII_old, 
                                  ΓionHI, ΓionH2, ΓionDI, ΓionHeI, ΓionHeII, Γ_LW, Γpd, Γ_LW_HD, 
                                  z, CaseB ) 

        # Compute would-be (symmetric) relative change in H I, H II, and
        # e⁻ fractions/number densities:
        ΔxH2_rel  = abs( xH2_new - xH2_old )   / max( 0.5 * (xH2_old + xH2_new),   xmin )
        ΔxHII_rel = abs( xHII_new - xHII_old ) / max( 0.5 * (xHII_old + xHII_new), xmin )
        Δne_rel   = abs( ne_new - ne_old )     / max( 0.5 * (ne_old + ne_new),     xmin * nH_old )

        # Get the maximum relative change among them:
        Δx_rel    = maximum( (ΔxH2_rel, ΔxHII_rel, Δne_rel) )

        if Δx_rel > tol_Chem
            # Change in H I, H II, and/or e⁻ 
            # fraction was too high, reduce Δtsub & retry:
            Δtsub = Δtsub_half
            continue    
        end

        # Get updated relevant number densities & abundance fractions:
        xHI_new     =  max( 1 - xHII_new - 2 * xH2_new, 0.0 )
        xDI_new     =  max( 1 - xDII_new  - xHD_new,    0.0 )
        xHeI_new    =  max( 1 - xHeII_new - xHeIII_new, 0.0 )
    
        nHI_new     =  xHI_new    * nH_old
        nHII_new    =  xHII_new   * nH_old 
        nH2_new     =  xH2_new    * nH_old
        nHD_new     =  xHD_new    * nD_old    # Enters the cooling rate
        nHeI_new    =  xHeI_new   * nHe_old
        nHeII_new   =  xHeII_new  * nHe_old
        nHeIII_new  =  xHeIII_new * nHe_old
        ne_new      =  nHII_new  +  nHeII_new  +  2 * nHeIII_new

        # Get updated adiabatic index and gas number density:
        n           =  nH_old  +  nHe_old  +  ne_new          # Only ne will change 
        γad         =  Getγad( xHII_new, xH2_new, xHeII_new, xHeIII_new )

        #######################################################################
        #######                                                         #######
        #######      S T E P  4  :  H E A T I N G / C O O L I N G       #######
        #######                                                         #######
        #######################################################################

        # Next we do another update of e from heating and cooling, over a 
        # half-substep Δtsub / 2. Using the latest chemical abundances. 

        REDO_HEATINGCOOLING, e_new, 
        Δe_2, Δt_e = HeatingCoolingStep( Δtsub_half, e_new, γad, 
                                         n, nHI_new, nH2_new, nHII_new, nHeI_new, nHeII_new,
                                         nHeIII_new, nHD_new, ne_new, 
                                         J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, 
                                         SelfShielding, z, CaseB )

        if REDO_HEATINGCOOLING
            # Change in e was too large or unphysical, so retry with 
            # smaller sub-cycle time-step:
            Δtsub = min( Δtsub_half,  
                         Δt_e )
            continue 
        end 

        e_floor = n * kB * T_floor / (γad - 1)  # e corresponding to temperature floor
        if CoolingHeating 
            e_new = min( max( e_new, e_floor ),
                         n * kB * T_ceiling / (γad - 1) )
        else
            # For some tests (e.g. R-type ionization front) we ignore heating/cooling:
            e_new = e_old
        end

        #######################################################################
        #######                                                         #######
        #######            S T E P  5  :  A B S O R P T I O N           #######
        #######                                                         #######
        #######################################################################

        # Finally, we update J and H for all stellar bands again, over a 
        # half-substep Δtsub / 2. Using the latest chemical abundances. 

        REDO_ABSORPTION, J_opt_trial, J_FUV_trial, J_LW_trial, J_EUV1_trial, J_EUV2_trial, J_EUV3_trial, 
        H_r_opt_trial, H_r_FUV_trial, H_r_LW_trial, H_r_EUV1_trial, H_r_EUV2_trial, H_r_EUV3_trial,
        H_Z_opt_trial, H_Z_FUV_trial, H_Z_LW_trial, H_Z_EUV1_trial, H_Z_EUV2_trial, H_Z_EUV3_trial,
        Δρu_r_trial_2, Δρu_Z_trial_2,
        ΔStellarRT_2 = AbsorptionStep( Δtsub_half, c̃, nHI_new, nH2_new, nHeI_new, nHeII_new,
                                       J_opt, dJ_optdt, 
                                       J_FUV, dJ_FUVdt, 
                                       J_LW, dJ_LWdt, H_r_LW, H_Z_LW, dHr_LWdt, dHZ_LWdt,
                                       J_EUV1, dJ_EUV1dt, H_r_EUV1, H_Z_EUV1, dHr_EUV1dt, dHZ_EUV1dt,
                                       J_EUV2, dJ_EUV2dt, H_r_EUV2, H_Z_EUV2, dHr_EUV2dt, dHZ_EUV2dt,
                                       J_EUV3, dJ_EUV3dt, H_r_EUV3, H_Z_EUV3, dHr_EUV3dt, dHZ_EUV3dt )

        if REDO_ABSORPTION
            # Changes in J or H were too large, so reject update and try a smaller sub-cycle time-step: 
            Δtsub = Δtsub_half
            continue 
        else
            # If changes in J and H were small enough, we accept the trial updated J and H:
            J_opt  = J_opt_trial;  J_FUV  = J_FUV_trial;  J_LW   = J_LW_trial
            J_EUV1 = J_EUV1_trial; J_EUV2 = J_EUV2_trial; J_EUV3 = J_EUV3_trial 

            H_r_opt  = H_r_opt_trial;  H_r_FUV  = H_r_FUV_trial;  H_r_LW   = H_r_LW_trial
            H_r_EUV1 = H_r_EUV1_trial; H_r_EUV2 = H_r_EUV2_trial; H_r_EUV3 = H_r_EUV3_trial
            
            H_Z_opt  = H_Z_opt_trial;  H_Z_FUV  = H_Z_FUV_trial;  H_Z_LW   = H_Z_LW_trial
            H_Z_EUV1 = H_Z_EUV1_trial; H_Z_EUV2 = H_Z_EUV2_trial; H_Z_EUV3 = H_Z_EUV3_trial

            # Also add rad. pressure momentum change:
            Δρu_r_trial += Δρu_r_trial_2
            Δρu_Z_trial += Δρu_Z_trial_2
        end



        # Recommended next sub-cycle time-step based on the relative change 
        # in quantities:
        Δe         = Δe_1 + Δe_2
        ΔStellarRT = ΔStellarRT_1 + ΔStellarRT_2
        if ( Δx_rel < tol_enlargestep  &&  Δe < tol_enlargestep  &&  ΔStellarRT < tol_enlargestep )
            Δtsub_new = 2 * min( Δtsub, Δt_e )
        else
            Δtsub_new = Δtsub
        end

        # Get flux opacities (for use in the M1 transport, to get the asymptotic diffusion limit)
        αc_F_EUV1  = nHI_new * σHI_EUV1  +  nH2_new * σH2_EUV1 
        αc_F_EUV2  = nHI_new * σHI_EUV2  +  nH2_new * σH2_EUV2  +  nHeI_new * σHeI_EUV2  
        αc_F_EUV3  = nHI_new * σHI_EUV3  +  nH2_new * σH2_EUV3  +  nHeI_new * σHeI_EUV3  +  nHeII_new * σHeII_EUV3 
        
        # Return every updated quantity, the time-step taken, & recommended next sub-cycle time-step:
        return Δtsub, Δtsub_new, e_new, xH2_new, xHII_new, xHD_new, xDII_new, xHeII_new, xHeIII_new, 
               J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, 
               H_r_LW, H_Z_LW,  H_r_EUV1, H_Z_EUV1,  H_r_EUV2, H_Z_EUV2,  H_r_EUV3, H_Z_EUV3,
               αc_F_EUV1, αc_F_EUV2, αc_F_EUV3, 
               Δρu_r_trial, Δρu_Z_trial

    end

    # Otherwise print error:
    error( """ ThermoChemistrySubStepUpdate could not converge within $(max_iter) iterations in cell (i,j) = ($(icell),$(jcell)),
               and simulation has therefore terminated.
               
               Relative changes in quantities in this cell during the final attempted sub-cycle time-step were:
               
               - Relative change in e⁻ abundance, Δne_rel    = $(Δne_rel)
               - Relative change in H₂ abundance, ΔxH2_rel   = $(ΔxH2_rel)
               - Relative change in H II abundance, ΔxHII_rel = $(ΔxHII_rel)
               - Relative change in e, Δe_rel = $(Δe)
               - Relative change in stellar RT, ΔStellarRT = $(ΔStellarRT)
               - REDO_ABSORPTION     = $(REDO_ABSORPTION)
               - REDO_HEATINGCOOLING = $(REDO_HEATINGCOOLING) """ )

end # End of ThermoChemistrySubStepUpdate!



max_steps::Int64 = 1000000 # Maximum number of sub-cycling sub-steps per global time-step (default = 10⁶)

""" The full thermochemistry sub-cycling over a global time-step Δtglobal.
    Updates the intensities, gas internal energy, and chemical abundances.
    
    **ARGUMENTS:** 
    
    - 
    
    
    """
@views function ThermoChemistrySubCycleUpdate!(  icell::Int,  jcell::Int,
                                                    Δr::Float64, ΔZ::Float64,
                                              Δtglobal::Float64,
                                                     c̃::Float64,
                                                 e_old::Float64,
                                                     ρ::Float64,
                                                    ∇ρ::Float64,
                                               xH2_old::Float64,
                                              xHII_old::Float64, 
                                               xHD_old::Float64,
                                              xDII_old::Float64,
                                             xHeII_old::Float64,
                                            xHeIII_old::Float64,

                                         Δρu_stellar_r::AbstractArray{Float64, 2},
                                         Δρu_stellar_Z::AbstractArray{Float64, 2},
                                               αF_EUV1::AbstractArray{Float64, 2},
                                               αF_EUV2::AbstractArray{Float64, 2},
                                               αF_EUV3::AbstractArray{Float64, 2},

                                          J_opt_transp::Float64 = 0.0,
                                              dJ_optdt::Float64 = 0.0,
                                        H_r_opt_transp::Float64 = 0.0,
                                        H_Z_opt_transp::Float64 = 0.0,
                                             dHr_optdt::Float64 = 0.0,
                                             dHZ_optdt::Float64 = 0.0,

                                          J_FUV_transp::Float64 = 0.0,
                                              dJ_FUVdt::Float64 = 0.0,
                                        H_r_FUV_transp::Float64 = 0.0,
                                        H_Z_FUV_transp::Float64 = 0.0,
                                             dHr_FUVdt::Float64 = 0.0,
                                             dHZ_FUVdt::Float64 = 0.0,

                                           J_LW_transp::Float64 = 0.0,
                                               dJ_LWdt::Float64 = 0.0,
                                         H_r_LW_transp::Float64 = 0.0,
                                         H_Z_LW_transp::Float64 = 0.0,
                                              dHr_LWdt::Float64 = 0.0,
                                              dHZ_LWdt::Float64 = 0.0,

                                         J_EUV1_transp::Float64 = 0.0,
                                             dJ_EUV1dt::Float64 = 0.0,
                                       H_r_EUV1_transp::Float64 = 0.0,
                                       H_Z_EUV1_transp::Float64 = 0.0,
                                            dHr_EUV1dt::Float64 = 0.0,
                                            dHZ_EUV1dt::Float64 = 0.0,

                                         J_EUV2_transp::Float64 = 0.0,
                                             dJ_EUV2dt::Float64 = 0.0,
                                       H_r_EUV2_transp::Float64 = 0.0,
                                       H_Z_EUV2_transp::Float64 = 0.0,
                                            dHr_EUV2dt::Float64 = 0.0,
                                            dHZ_EUV2dt::Float64 = 0.0,

                                         J_EUV3_transp::Float64 = 0.0,
                                             dJ_EUV3dt::Float64 = 0.0,
                                       H_r_EUV3_transp::Float64 = 0.0,
                                       H_Z_EUV3_transp::Float64 = 0.0,
                                            dHr_EUV3dt::Float64 = 0.0,
                                            dHZ_EUV3dt::Float64 = 0.0,

                                                     z::Float64 = 0.0,
                                                 CaseB::Bool    = true,
                                        CoolingHeating::Bool    = true
                                              )
    # Initialize time, internal energy, and abundance fractions
    # using the ones from the previous global time-step:
    t_accum          = 0.0 
    e_current        = e_old 
    xH2_current      = xH2_old
    xHII_current     = xHII_old
    xHD_current      = xHD_old 
    xDII_current     = xDII_old
    xHeII_current    = xHeII_old
    xHeIII_current   = xHeIII_old
    # Similarly, initialize the intensities as those from 
    # the transport step:
    J_opt_current    = J_opt_transp
    J_FUV_current    = J_FUV_transp
    J_LW_current     = J_LW_transp
    J_EUV1_current   = J_EUV1_transp
    J_EUV2_current   = J_EUV2_transp
    J_EUV3_current   = J_EUV3_transp
    # Initialize H components:
    H_r_LW_current   =  H_r_LW_transp
    H_Z_LW_current   =  H_Z_LW_transp

    H_r_EUV1_current =  H_r_EUV1_transp
    H_Z_EUV1_current =  H_Z_EUV1_transp

    H_r_EUV2_current =  H_r_EUV2_transp
    H_Z_EUV2_current =  H_Z_EUV2_transp

    H_r_EUV3_current =  H_r_EUV3_transp
    H_Z_EUV3_current =  H_Z_EUV3_transp

    # Initialize cell flux opacities:
    αc_F_EUV1_current = 0.0
    αc_F_EUV2_current = 0.0
    αc_F_EUV3_current = 0.0


    # Initialize the momentum density change from 
    # stellar radiation pressure:
    Δρu_stellar_r[icell, jcell] = 0.0  # r-direction momentum change in gas 
    Δρu_stellar_Z[icell, jcell] = 0.0  # Z-direction momentum change in gas 
    
    # Start with the global time-step as an initial guess
    # for the sub-cycle time-step:
    Δtsub = Δtglobal

    step  = 0 # Initialize step count
    while t_accum < Δtglobal 
        # Ensure we don't overshoot the global time-step:
        Δtsub  = min( Δtsub, Δtglobal - t_accum )

        # Call sub-cycle update for this sub-step:
        result =  ThermoChemistrySubStepUpdate!( icell, jcell, Δr, ΔZ,
                  Δtsub, c̃,
                  e_current, ρ, ∇ρ,
                  xH2_current, xHII_current, xHD_current, xDII_current,
                  xHeII_current, xHeIII_current, 

                  J_opt_current, dJ_optdt, 
                  J_FUV_current, dJ_FUVdt,

                  J_LW_current, dJ_LWdt,
                  H_r_LW_current, H_Z_LW_current, dHr_LWdt, dHZ_LWdt,

                  J_EUV1_current,   dJ_EUV1dt, 
                  H_r_EUV1_current, H_Z_EUV1_current, dHr_EUV1dt, dHZ_EUV1dt,

                  J_EUV2_current, dJ_EUV2dt, 
                  H_r_EUV2_current, H_Z_EUV2_current, dHr_EUV2dt, dHZ_EUV2dt,

                  J_EUV3_current, dJ_EUV3dt,
                  H_r_EUV3_current, H_Z_EUV3_current, dHr_EUV3dt, dHZ_EUV3dt,

                  z, CaseB, CoolingHeating )

        # Unpack result:
        Δtsub, Δtsub_new, e_new, xH2_new, xHII_new, xHD_new, xDII_new, xHeII_new, xHeIII_new, 
        J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, 
        H_r_LW, H_Z_LW,  H_r_EUV1, H_Z_EUV1,  H_r_EUV2, H_Z_EUV2,  H_r_EUV3, H_Z_EUV3,
        αc_F_EUV1, αc_F_EUV2, αc_F_EUV3,
        Δρu_r_substep, Δρu_Z_substep  = result

        # Update our current internal energy & abundance fractions:
        e_current        = e_new
        xH2_current      = xH2_new
        xHII_current     = xHII_new
        xHD_current      = xHD_new 
        xDII_current     = xDII_new
        xHeII_current    = xHeII_new
        xHeIII_current   = xHeIII_new
        # Update our current intensities:
        J_opt_current    = J_opt
        J_FUV_current    = J_FUV 
        J_LW_current     = J_LW 
        J_EUV1_current   = J_EUV1
        J_EUV2_current   = J_EUV2
        J_EUV3_current   = J_EUV3 
        # Update our current H-components:
        H_r_LW_current   = H_r_LW
        H_Z_LW_current   = H_Z_LW 
        H_r_EUV1_current = H_r_EUV1
        H_Z_EUV1_current = H_Z_EUV1
        H_r_EUV2_current = H_r_EUV2 
        H_Z_EUV2_current = H_Z_EUV2 
        H_r_EUV3_current = H_r_EUV3
        H_Z_EUV3_current = H_Z_EUV3

        # Update cell flux opacities:
        αc_F_EUV1_current = αc_F_EUV1
        αc_F_EUV2_current = αc_F_EUV2
        αc_F_EUV3_current = αc_F_EUV3

        # Add stellar radiation pressure momentum from this sub-step: 
        Δρu_stellar_r[icell, jcell] += Δρu_r_substep
        Δρu_stellar_Z[icell, jcell] += Δρu_Z_substep

        # Update current time:
        t_accum += Δtsub

        # Update sub-cycle time-step using the new recommended value:
        Δtsub = Δtsub_new

        # Update the step count and warn the user if it has exceeded the specified upper limit:
        step += 1
        if step > max_steps
            error(""" The thermochemistry sub-cycling exceeded $(max_steps) steps in cell ($(icell), $(jcell)). 
                      Terminating the simulation. """)
        end
    end

    # Save latest flux opacities:
    αF_EUV1[icell, jcell] = 0.0 * αc_F_EUV1_current
    αF_EUV2[icell, jcell] = 0.0 * αc_F_EUV2_current
    αF_EUV3[icell, jcell] = 0.0 * αc_F_EUV3_current

    # Return the updated state after evolving over the global time-step:
    return e_current, xH2_current, xHII_current, xHD_current, xDII_current, xHeII_current, xHeIII_current, 
           J_opt_current, J_FUV_current, J_LW_current, J_EUV1_current, J_EUV2_current, J_EUV3_current,
           H_r_LW_current, H_Z_LW_current,  H_r_EUV1_current, H_Z_EUV1_current,  
           H_r_EUV2_current, H_Z_EUV2_current,  H_r_EUV3_current, H_Z_EUV3_current
end


function UpdateRTandThermoChemistry!( Δt_RT_max::Float64, 
                                              c̃::Float64,
                                             RT::StellarRTArrays,
                                     HydroState::HydrodynamicsState,
                                      ChemState::ChemistryState,
                                              z::Float64,
                                 CoolingHeating::Bool,

                                           # Geometry:
                                         inv_ΔV::AbstractArray{Float64, 2},
                                            A_r::AbstractArray{Float64, 2},
                                            A_Z::AbstractArray{Float64, 2},
                                              r::AbstractArray{Float64, 1},
                                           rCoM::AbstractArray{Float64, 1},
                                              Z::AbstractArray{Float64, 1},
                                           ZCoM::AbstractArray{Float64, 1},
                                     inv_r_cell::AbstractArray{Float64, 1},
                               rfirst_StellarRT::Bool,

                                          # Gas velocity field:
                                       u_r_star::AbstractArray{Float64,2},
                                       u_Z_star::AbstractArray{Float64,2},
                                        ∂u_r_∂r::AbstractArray{Float64,2},
                                        ∂u_r_∂Z::AbstractArray{Float64,2}, 
                                        ∂u_Z_∂r::AbstractArray{Float64,2}, 
                                        ∂u_Z_∂Z::AbstractArray{Float64,2},
                                     u_r_over_r::AbstractArray{Float64,2},

                                           # Lyα stuff:
                                          c̃_Lyα::Float64,
                                       LyaState::LyαArrays,
                                            Lyα::Bool = false,

                                           # Boundary conditions:
                                      BC_r_left::Symbol = :Reflect,
                                     BC_r_right::Symbol = :Outflow,
                                       BC_Z_bot::Symbol = :Reflect,
                                       BC_Z_top::Symbol = :Outflow
                                              )

    Δt_CFL_RT = GetRHDCFLTimeStep( r, Z, c̃, HydroState.u_X, HydroState.u_Y, CFL_RT ) 
    Δt_RT     = min( Δt_CFL_RT, Δt_RT_max )  
    c̃Δt_RT    = c̃ * Δt_RT    

    @assert BC_r_left  == :Reflect 
    @assert BC_r_right == :Outflow
    @assert BC_Z_bot   == :Reflect 
    @assert BC_Z_top   == :Outflow

    #######################################################################
    #######                                                         #######
    #######       S T E P  1  :  I N J E C T I O N  S T E P         #######
    #######                                                         #######
    #######################################################################

    # Stellar emission update: 
    RT.J_LW       .= RT.J_LW    .+  RT.js_LW   .* c̃ .* Δt_RT
    RT.J_EUV1     .= RT.J_EUV1  .+  RT.js_EUV1 .* c̃ .* Δt_RT
    RT.J_EUV2     .= RT.J_EUV2  .+  RT.js_EUV2 .* c̃ .* Δt_RT
    RT.J_EUV3     .= RT.J_EUV3  .+  RT.js_EUV3 .* c̃ .* Δt_RT

    #######################################################################
    #######                                                         #######
    #######       S T E P  1  :  T R A N S P O R T  S T E P         #######
    #######                                                         #######
    #######################################################################

    # We advect the J and H for all bands, using the M1 moment method. 

    UpdateStellarJold = false # Don't update Jold, since we want the pre-(injection + transport) value to compute dJ/dt
    UpdateStellarH    = false # Don't update H within the RK-2 update, since we sub-cycle H 

    # Advect LW photons:
    AdvectJandH!( c̃Δt_RT,   RT.α_F_LW,  RT.J_LW,     RT.J_LW_old,   RT.H_r_LW,   RT.H_Z_LW, 
                  RT.J_LW_stage,     RT.H_r_LW_stage,   RT.H_Z_LW_stage,
                  RT.D_rr_LW,        RT.D_ZZ_LW,  RT.D_rZ_LW, 
                  RT.K_ϕϕ_iphalf_LW, RT.K_ϕϕ_imhalf_LW,
                  RT.F_r_LW,         RT.F_Z_LW,
                  RT.F_r_Hr_LW,      RT.F_Z_Hr_LW, 
                  RT.F_r_HZ_LW,      RT.F_Z_HZ_LW, 
                  inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell,
                  u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                  UpdateStellarJold, UpdateStellarH, true, BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top,
                  rfirst_StellarRT, Val(:Threaded) )

    # Advect EUV1 photons:
    AdvectJandH!( c̃Δt_RT,   RT.α_F_EUV1,  RT.J_EUV1,     RT.J_EUV1_old,   RT.H_r_EUV1,   RT.H_Z_EUV1, 
                  RT.J_EUV1_stage,     RT.H_r_EUV1_stage,   RT.H_Z_EUV1_stage,
                  RT.D_rr_EUV1,        RT.D_ZZ_EUV1,  RT.D_rZ_EUV1, 
                  RT.K_ϕϕ_iphalf_EUV1, RT.K_ϕϕ_imhalf_EUV1,
                  RT.F_r_EUV1,         RT.F_Z_EUV1,
                  RT.F_r_Hr_EUV1,      RT.F_Z_Hr_EUV1, 
                  RT.F_r_HZ_EUV1,      RT.F_Z_HZ_EUV1, 
                  inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell,
                  u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                  UpdateStellarJold, UpdateStellarH, true, BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top,
                  rfirst_StellarRT, Val(:Threaded) )

    # Advect EUV2 photons:
    AdvectJandH!( c̃Δt_RT,   RT.α_F_EUV2,  RT.J_EUV2,     RT.J_EUV2_old,   RT.H_r_EUV2,   RT.H_Z_EUV2, 
                  RT.J_EUV2_stage,     RT.H_r_EUV2_stage,   RT.H_Z_EUV2_stage,
                  RT.D_rr_EUV2,        RT.D_ZZ_EUV2,  RT.D_rZ_EUV2, 
                  RT.K_ϕϕ_iphalf_EUV2, RT.K_ϕϕ_imhalf_EUV2,
                  RT.F_r_EUV2,         RT.F_Z_EUV2,
                  RT.F_r_Hr_EUV2,      RT.F_Z_Hr_EUV2, 
                  RT.F_r_HZ_EUV2,      RT.F_Z_HZ_EUV2, 
                  inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell,
                  u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                  UpdateStellarJold, UpdateStellarH, true, BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top,
                  rfirst_StellarRT, Val(:Threaded) )

    # Advect EUV3 photons:
    AdvectJandH!( c̃Δt_RT,   RT.α_F_EUV3,  RT.J_EUV3,     RT.J_EUV3_old,   RT.H_r_EUV3,   RT.H_Z_EUV3, 
                  RT.J_EUV3_stage,     RT.H_r_EUV3_stage,   RT.H_Z_EUV3_stage,
                  RT.D_rr_EUV3,        RT.D_ZZ_EUV3,  RT.D_rZ_EUV3, 
                  RT.K_ϕϕ_iphalf_EUV3, RT.K_ϕϕ_imhalf_EUV3,
                  RT.F_r_EUV3,         RT.F_Z_EUV3,
                  RT.F_r_Hr_EUV3,      RT.F_Z_Hr_EUV3, 
                  RT.F_r_HZ_EUV3,      RT.F_Z_HZ_EUV3, 
                  inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell,
                  u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                  UpdateStellarJold, UpdateStellarH, true, BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top,
                  rfirst_StellarRT, Val(:Threaded) )
    

    #######################################################################
    #######                                                         #######
    #######       S T E P  3  :  T H E R M O - C H E M I S T R Y    #######
    #######                                                         #######
    #######################################################################

    # After the transport step, we sub-cycle absorption, emission, chemistry,
    # and cooling/heating for all cells.

    # Unpack hydro & chemistry arrays:
    ρ      = HydroState.ρ
    e      = HydroState.e 
    T      = HydroState.T
    xH2    = ChemState.xH2
    xHII   = ChemState.xHII
    xHD    = ChemState.xHD
    xDII   = ChemState.xDII
    xHeII  = ChemState.xHeII 
    xHeIII = ChemState.xHeIII
                                              
    # Grid size:
    N_r, N_Z  = size( ρ )
    dN_r      = 1:N_r 
    dN_Z      = 1:N_Z 

    # Next, sub-cycle for each cell: 
    inv_Δt_RT = 1 / Δt_RT
    @threads for j in dN_Z
        @inbounds ΔZ = Z[j+1] - Z[j]   # Cell width in Z-direction 

        @simd for i in dN_r
            @inbounds begin

                Δr   = r[i+1] - r[i]   # Cell width in r-direction

                # Get density gradient magnitude for the cell (for the column density estimate):
                ∇ρ   = Estimate_∇ρ( i, j, ρ, rCoM, ZCoM )

                # Get estimated time-derivatives of J from emission + advection:
                dJ_optdt   = ( RT.J_opt[i,j]  - RT.J_opt_old[i,j]  ) * inv_Δt_RT
                dJ_FUVdt   = ( RT.J_FUV[i,j]  - RT.J_FUV_old[i,j]  ) * inv_Δt_RT
                dJ_LWdt    = ( RT.J_LW[i,j]   - RT.J_LW_old[i,j]   ) * inv_Δt_RT
                dJ_EUV1dt  = ( RT.J_EUV1[i,j] - RT.J_EUV1_old[i,j] ) * inv_Δt_RT
                dJ_EUV2dt  = ( RT.J_EUV2[i,j] - RT.J_EUV2_old[i,j] ) * inv_Δt_RT
                dJ_EUV3dt  = ( RT.J_EUV3[i,j] - RT.J_EUV3_old[i,j] ) * inv_Δt_RT

                # Get estimated time-derivatives of H_r and H_Z from advection:
                dHr_optdt  = ( RT.H_r_opt[i,j]  - RT.H_r_opt_old[i,j] )  * inv_Δt_RT
                dHZ_optdt  = ( RT.H_Z_opt[i,j]  - RT.H_Z_opt_old[i,j] )  * inv_Δt_RT

                dHr_FUVdt  = ( RT.H_r_FUV[i,j]  - RT.H_r_FUV_old[i,j] )  * inv_Δt_RT 
                dHZ_FUVdt  = ( RT.H_Z_FUV[i,j]  - RT.H_Z_FUV_old[i,j] )  * inv_Δt_RT

                dHr_LWdt   = ( RT.H_r_LW[i,j]   - RT.H_r_LW_old[i,j] )   * inv_Δt_RT
                dHZ_LWdt   = ( RT.H_Z_LW[i,j]   - RT.H_Z_LW_old[i,j] )   * inv_Δt_RT

                dHr_EUV1dt = ( RT.H_r_EUV1[i,j] - RT.H_r_EUV1_old[i,j] ) * inv_Δt_RT
                dHZ_EUV1dt = ( RT.H_Z_EUV1[i,j] - RT.H_Z_EUV1_old[i,j] ) * inv_Δt_RT

                dHr_EUV2dt = ( RT.H_r_EUV2[i,j] - RT.H_r_EUV2_old[i,j] ) * inv_Δt_RT 
                dHZ_EUV2dt = ( RT.H_Z_EUV2[i,j] - RT.H_Z_EUV2_old[i,j] ) * inv_Δt_RT

                dHr_EUV3dt = ( RT.H_r_EUV3[i,j] - RT.H_r_EUV3_old[i,j] ) * inv_Δt_RT
                dHZ_EUV3dt = ( RT.H_Z_EUV3[i,j] - RT.H_Z_EUV3_old[i,j] ) * inv_Δt_RT

                # Next we sub-cycle for each cell: 
                result = ThermoChemistrySubCycleUpdate!( i, j,  Δr, ΔZ,
                                                          Δt_RT,   c̃,
                                                         e[i,j],   ρ[i,j],   ∇ρ,
                                                       xH2[i,j],   xHII[i,j],
                                                       xHD[i,j],   xDII[i,j],
                                                     xHeII[i,j],   xHeIII[i,j], 

                                                 RT.Δρu_stellar_r, RT.Δρu_stellar_Z,
                                                      RT.α_F_EUV1, RT.α_F_EUV2, RT.α_F_EUV3,

                                                RT.J_opt_old[i,j], dJ_optdt,
                                              RT.H_r_opt_old[i,j], RT.H_Z_opt_old[i,j], dHr_optdt, dHZ_optdt,

                                                RT.J_FUV_old[i,j], dJ_FUVdt,
                                              RT.H_r_FUV_old[i,j], RT.H_Z_FUV_old[i,j], dHr_FUVdt, dHZ_FUVdt,

                                                 RT.J_LW_old[i,j], dJ_LWdt, 
                                               RT.H_r_LW_old[i,j], RT.H_Z_LW_old[i,j], dHr_LWdt, dHZ_LWdt,

                                               RT.J_EUV1_old[i,j], dJ_EUV1dt, 
                                             RT.H_r_EUV1_old[i,j], RT.H_Z_EUV1_old[i,j], dHr_EUV1dt, dHZ_EUV1dt,

                                               RT.J_EUV2_old[i,j], dJ_EUV2dt, 
                                             RT.H_r_EUV2_old[i,j], RT.H_Z_EUV2_old[i,j], dHr_EUV2dt, dHZ_EUV2dt,

                                               RT.J_EUV3_old[i,j], dJ_EUV3dt,
                                             RT.H_r_EUV3_old[i,j], RT.H_Z_EUV3_old[i,j], dHr_EUV3dt, dHZ_EUV3dt,

                                                          z, true, CoolingHeating )

                # Get updated quantities:
                e[i,j], xH2[i,j], xHII[i,j], xHD[i,j], xDII[i,j], xHeII[i,j], xHeIII[i,j], 
                RT.J_opt[i,j], RT.J_FUV[i,j], RT.J_LW[i,j], RT.J_EUV1[i,j], RT.J_EUV2[i,j], RT.J_EUV3[i,j],
                RT.H_r_LW[i,j],   RT.H_Z_LW[i,j],
                RT.H_r_EUV1[i,j], RT.H_Z_EUV1[i,j],  
                RT.H_r_EUV2[i,j], RT.H_Z_EUV2[i,j],
                RT.H_r_EUV3[i,j], RT.H_Z_EUV3[i,j] = result 
            end
        end # End of i-loop 
    end # End of j-loop 

    # Update the old stellar J and H:
    @threads for j in dN_Z
        @simd for i in dN_r 
            @inbounds begin
                # Update old J:
                #RT.J_opt_old[i,j]   = RT.J_opt[i,j]
                #RT.J_FUV_old[i,j]   = RT.J_FUV[i,j]
                RT.J_LW_old[i,j]     = RT.J_LW[i,j]
                RT.J_EUV1_old[i,j]   = RT.J_EUV1[i,j]
                RT.J_EUV2_old[i,j]   = RT.J_EUV2[i,j]
                RT.J_EUV3_old[i,j]   = RT.J_EUV3[i,j]

                # Update old H:
                RT.H_r_LW_old[i,j]   = RT.H_r_LW[i,j]
                RT.H_Z_LW_old[i,j]   = RT.H_Z_LW[i,j]

                RT.H_r_EUV1_old[i,j] = RT.H_r_EUV1[i,j]
                RT.H_Z_EUV1_old[i,j] = RT.H_Z_EUV1[i,j]

                RT.H_r_EUV2_old[i,j] = RT.H_r_EUV2[i,j]
                RT.H_Z_EUV2_old[i,j] = RT.H_Z_EUV2[i,j]

                RT.H_r_EUV3_old[i,j] = RT.H_r_EUV3[i,j]
                RT.H_Z_EUV3_old[i,j] = RT.H_Z_EUV3[i,j]

            end
        end # End of i-loop 
    end # End of j-loop 


    #######################################################################
    #######                                                         #######
    #######             S T E P  4  :  L Y M A N - α  R T           #######
    #######                                                         #######
    #######################################################################

    # If Lyα RT is turned on, we also call the Lyα advection + frequency diffusion 
    # solver over generally smaller sub-steps that add up to the 
    # stellar RT time-step Δt_RT:
    if Lyα 
        N_ν  = length( LyaState.νC )
        dN_ν = 1:N_ν

        # Update α (H I opacity @ line center), pd (destruction probability),
        # and js_Lyα (frequency-integrated emissivity):
        @threads for j in dN_Z 
            @inbounds for i in dN_r
                # Get all needed number densities:
                nHI, nHII, nH2       = GetHydrogenDensities( xHII[i,j], xH2[i,j], ρ[i,j] )
                nHeI, nHeII, nHeIII  = GetHeliumDensities( xHeII[i,j], xHeIII[i,j], ρ[i,j] )
                ne                   = GetElectronDensity( nHII, nHeII, nHeIII )

                # Also get the gas temperature in the cell:
                μ        = Getμ( xHII[i,j], xH2[i,j], xHeII[i,j], xHeIII[i,j] )
                γad      = Getγad( xHII[i,j], xH2[i,j], xHeII[i,j], xHeIII[i,j] )
                T[i,j]   = e[i,j] * μ * (γad - 1) * mH / ( ρ[i,j] * kB )

                # Update α (H I opacity @ line center, nHI × σ0):
                LyaState.α_Lyα[i,j]  = nHI * σ0( T[i,j] )

                # Update pd (Lyα destruction probability):
                LyaState.pd[i,j]     = pd_Lyα( nHII, ne, nHI, nHeI, nH2, T[i,j], z )

                # Update the frequency integrated emissivity: 
                LyaState.∫dν_js_Lyα[i,j] = ( ∫dν_js_recLyα( T[i,j], nHII, ne, true, true )
                                         +   ∫dν_js_collLyα( T[i,j], nHI, ne ) )

            end # End of i-loop
        end # End of j-loop 

        # Set all frequency diffusion + Doppler shift coefficients (the F's)
        # using the latest values of the gas temperature T and hydro/chemical state:
        UpdateFreqFactors!( LyaState, T, ∂u_r_∂r, ∂u_r_∂Z,  ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r  )

        # Update the flux opacity using the newly updated Hvoigt:
        @threads for k in dN_ν 
            @inbounds @simd for j in dN_Z 
                for i in dN_r
                    LyaState.α_F_Lyα[i,j,k] = LyaState.α_Lyα[i,j] * LyaState.Hvoigt[i,j,k]  
                end # End of i-loop
            end # End of j-loop 
        end # End of k-loop 


        # Determine the Lyα CFL time-step: 
        Δt_Lyα_CFL = GetRHDCFLTimeStep( r, Z, c̃_Lyα, HydroState.u_X, HydroState.u_Y, CFL_RT )

        # Start the Lyα RT solve:
        t_Lyα = 0.0
        while t_Lyα < Δt_RT
            # Set maximum Lyα RT transport time-step:
            Δt_Lyα_max = max( Δt_RT - t_Lyα, 0.0 )
            Δt_Lyα     = min( Δt_Lyα_max, Δt_Lyα_CFL )
            c̃Δt_Lyα    = c̃_Lyα * Δt_Lyα

            # Lyα emission:
            LyαEmissionUpdate!( LyaState.J_Lyα, LyaState.∫dν_js_Lyα,
                                LyaState.ϕ_ν, c̃_Lyα, Δt_Lyα )

            # Lyα Advection + frequency diffusion & Doppler shift:
            UpdateLyαJandH!( c̃Δt_Lyα, LyaState, inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell, 
                             u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                             true, BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top )

            # Lyα absorption & destruction, and change in gas momentum from Lyα radiation pressure:
            LyαAbsorptionUpdate!( LyaState.J_Lyα, 
                                  LyaState.H_r_Lyα, LyaState.H_Z_Lyα,
                                  LyaState.H_r_Lyα_old, LyaState.H_Z_Lyα_old,
                                  LyaState.Hvoigt, LyaState.α_Lyα, LyaState.αc_Lyα, LyaState.α_F_Lyα, 
                                  LyaState.pd, 
                                  LyaState.Δρu_Lyα_r, LyaState.Δρu_Lyα_Z,
                                  LyaState.Δν,
                                  c̃_Lyα, Δt_Lyα )

            # Update time:
            t_Lyα += Δt_Lyα
        end
    end

    return Δt_RT
end


""" This function computes weight factors fstar_ij for 
    each cell (i,j), which determines the fraction of 
    luminosity/emissivty & mass from the star/star cluster 
    that is in cell (i,j). Assumes a Gaussian distribution 
    with σstar, centered at Z = Zstar. """
function ComputeStellar_fij!( fstar_ij::AbstractArray{Float64, 2},
                                     r::AbstractArray{Float64, 1},
                                     Z::AbstractArray{Float64, 1},
                                 Zstar::Float64,
                                 σstar::Float64
                              )

    N_r  = length(r) - 1
    N_Z  = length(Z) - 1
    dN_r = 1:N_r
    dN_Z = 1:N_Z
    @assert size(fstar_ij) == (N_r, N_Z)
    @assert σstar > 0

    σstar_sq    = σstar^2
    ∫dV_fij     = (Two_π)^(3/2) * σstar^3   # The integral ∫dV Gaussian, over *all* space (not just simulation volume)
    inv_∫dV_fij = 1 / ∫dV_fij
    @threads for j in dN_Z
        @inbounds @simd for i in dN_r
            # The integral 2π ∫dr r * exp( - r²/ 2σ² ), from r[i] to r[i+1]:
            Integral_r = ( Two_π * σstar_sq 
                       * ( exp( - r[i]^2 / (2 * σstar_sq) )  -  exp( - r[i+1]^2 / (2 * σstar_sq)) )
                        )
            
            # The integral ∫dZ exp( - (Z - Zstar)²/ 2σ² ) from Z[j] to Z[j+1]:
            Integral_Z = ( σstar * sqrt(π/2) 
                       * ( erf((Z[j+1] - Zstar) / (sqrt(2) * σstar))  - erf((Z[j] - Zstar) / (sqrt(2) * σstar)) )
                        )

            # Fraction of total stellar mass & luminosity in cell (i,j):
            fstar_ij[i,j] = inv_∫dV_fij * Integral_r * Integral_Z
        end
    end

    return nothing 
end



function GenerateStellarRTArrays( mstarZAMS::Float64, 
                                 Zstar_Zsun::Float64,
                                         ΔV::AbstractArray{Float64, 2},
                                          r::AbstractArray{Float64, 1},
                                          Z::AbstractArray{Float64, 1},
                                  Zstar_pos::Float64,
                                 σstar_cell::Float64  = 2.0
                                 )
   
    # Grid size: 
    N_r, N_Z = size( ΔV )
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z

    # Convenient arrays: 
    zeros_c()  = zeros(Float64, N_r,   N_Z)
    zeros_rf() = zeros(Float64, N_r+1, N_Z)   # For r-interface fluxes 
    zeros_zf() = zeros(Float64, N_r,   N_Z+1) # For Z-interface fluxes 

    # Intensities (latest & old, and stage values for RK2) for all bands: 
    J_opt    = zeros_c();  J_opt_old  = copy(J_opt);  J_opt_stage  = copy(J_opt)
    J_FUV    = zeros_c();  J_FUV_old  = copy(J_FUV);  J_FUV_stage  = copy(J_opt)
    J_LW     = zeros_c();  J_LW_old   = copy(J_LW);   J_LW_stage   = copy(J_opt)
    J_EUV1   = zeros_c();  J_EUV1_old = copy(J_EUV1); J_EUV1_stage = copy(J_opt)
    J_EUV2   = zeros_c();  J_EUV2_old = copy(J_EUV2); J_EUV2_stage = copy(J_opt)
    J_EUV3   = zeros_c();  J_EUV3_old = copy(J_EUV3); J_EUV3_stage = copy(J_opt)

    # H ≡ ∫dΩ I n̂ / 4π arrays for all bands:
    H_r_opt        = zeros_c();  H_Z_opt       = zeros_c()
    H_r_opt_old    = zeros_c();  H_Z_opt_old   = zeros_c()
    H_r_opt_stage  = zeros_c();  H_Z_opt_stage = zeros_c()

    H_r_FUV        = zeros_c();  H_Z_FUV       = zeros_c()
    H_r_FUV_old    = zeros_c();  H_Z_FUV_old   = zeros_c()
    H_r_FUV_stage  = zeros_c();  H_Z_FUV_stage = zeros_c()

    H_r_LW         = zeros_c();  H_Z_LW       = zeros_c()
    H_r_LW_old     = zeros_c();  H_Z_LW_old   = zeros_c()
    H_r_LW_stage   = zeros_c();  H_Z_LW_stage = zeros_c()

    H_r_EUV1       = zeros_c();  H_Z_EUV1       = zeros_c()
    H_r_EUV1_old   = zeros_c();  H_Z_EUV1_old   = zeros_c()
    H_r_EUV1_stage = zeros_c();  H_Z_EUV1_stage = zeros_c()

    H_r_EUV2       = zeros_c();  H_Z_EUV2       = zeros_c()
    H_r_EUV2_old   = zeros_c();  H_Z_EUV2_old   = zeros_c()
    H_r_EUV2_stage = zeros_c();  H_Z_EUV2_stage = zeros_c()

    H_r_EUV3       = zeros_c();  H_Z_EUV3     = zeros_c()
    H_r_EUV3_old   = zeros_c();  H_Z_EUV3_old = zeros_c()
    H_r_EUV3_stage = zeros_c();  H_Z_EUV3_stage = zeros_c()

    # Eddington tensor components for all bands:
    D_rr_opt  = fill(1/3, N_r, N_Z);  D_ZZ_opt  = fill(1/3, N_r, N_Z);  D_rZ_opt  = zeros_c()
    D_rr_FUV  = fill(1/3, N_r, N_Z);  D_ZZ_FUV  = fill(1/3, N_r, N_Z);  D_rZ_FUV  = zeros_c()
    D_rr_LW   = fill(1/3, N_r, N_Z);  D_ZZ_LW   = fill(1/3, N_r, N_Z);  D_rZ_LW   = zeros_c()
    D_rr_EUV1 = fill(1/3, N_r, N_Z);  D_ZZ_EUV1 = fill(1/3, N_r, N_Z);  D_rZ_EUV1 = zeros_c()
    D_rr_EUV2 = fill(1/3, N_r, N_Z);  D_ZZ_EUV2 = fill(1/3, N_r, N_Z);  D_rZ_EUV2 = zeros_c()
    D_rr_EUV3 = fill(1/3, N_r, N_Z);  D_ZZ_EUV3 = fill(1/3, N_r, N_Z);  D_rZ_EUV3 = zeros_c()

    # ϕϕ-component of radiation pressure tensor K = D × J at interfaces, 
    # needed for geometric source term:
    K_ϕϕ_iphalf_LW   = zeros_c();     K_ϕϕ_imhalf_LW   = zeros_c()
    K_ϕϕ_iphalf_EUV1 = zeros_c();     K_ϕϕ_imhalf_EUV1 = zeros_c()
    K_ϕϕ_iphalf_EUV2 = zeros_c();     K_ϕϕ_imhalf_EUV2 = zeros_c()
    K_ϕϕ_iphalf_EUV3 = zeros_c();     K_ϕϕ_imhalf_EUV3 = zeros_c()

    # Interface fluxes for all bands: 
    F_r_opt     = zeros_rf(); F_Z_opt     = zeros_zf()
    F_r_Hr_opt  = zeros_rf(); F_Z_Hr_opt  = zeros_zf()
    F_r_HZ_opt  = zeros_rf(); F_Z_HZ_opt  = zeros_zf()

    F_r_FUV     = zeros_rf(); F_Z_FUV     = zeros_zf()
    F_r_Hr_FUV  = zeros_rf(); F_Z_Hr_FUV  = zeros_zf()
    F_r_HZ_FUV  = zeros_rf(); F_Z_HZ_FUV  = zeros_zf()

    F_r_LW      = zeros_rf(); F_Z_LW      = zeros_zf()
    F_r_Hr_LW   = zeros_rf(); F_Z_Hr_LW   = zeros_zf()
    F_r_HZ_LW   = zeros_rf(); F_Z_HZ_LW   = zeros_zf()

    F_r_EUV1    = zeros_rf(); F_Z_EUV1    = zeros_zf()
    F_r_Hr_EUV1 = zeros_rf(); F_Z_Hr_EUV1 = zeros_zf()
    F_r_HZ_EUV1 = zeros_rf(); F_Z_HZ_EUV1 = zeros_zf()

    F_r_EUV2    = zeros_rf(); F_Z_EUV2    = zeros_zf()
    F_r_Hr_EUV2 = zeros_rf(); F_Z_Hr_EUV2 = zeros_zf()
    F_r_HZ_EUV2 = zeros_rf(); F_Z_HZ_EUV2 = zeros_zf()

    F_r_EUV3    = zeros_rf(); F_Z_EUV3    = zeros_zf()
    F_r_Hr_EUV3 = zeros_rf(); F_Z_Hr_EUV3 = zeros_zf()
    F_r_HZ_EUV3 = zeros_rf(); F_Z_HZ_EUV3 = zeros_zf()

    # Flux opacities for all bands:
    α_F_opt       = zeros_c()
    α_F_FUV       = zeros_c()
    α_F_LW        = zeros_c()
    α_F_EUV1      = zeros_c()
    α_F_EUV2      = zeros_c()
    α_F_EUV3      = zeros_c()

    # Emissivities for all bands:
    js_opt        = zeros_c()
    js_FUV        = zeros_c()
    js_LW         = zeros_c()
    js_EUV1       = zeros_c()
    js_EUV2       = zeros_c()
    js_EUV3       = zeros_c()

    # Cell weight factors for emissivity & stellar mass density (only used here):
    fstar_ij      = zeros_c()

    # Stellar mass density for all cells:
    ρ_star        = zeros_c()

    # Arrays for tracking changes in gas momentum from radiation pressure:
    Δρu_stellar_r = zeros_c()
    Δρu_stellar_Z = zeros_c()
    f_stellar_r   = zeros_c()    # Rad. pressure force per unit volume in r-direction
    f_stellar_Z   = zeros_c()    # Rad. pressure force per unit volume in Z-direction

    # Get stellar luminosities in all bands:
    Lum_opt       = L_opt(mstarZAMS, Zstar_Zsun)  * Lsun 
    Lum_FUV       = L_FUV(mstarZAMS, Zstar_Zsun)  * Lsun
    Lum_LW        = L_LW(mstarZAMS, Zstar_Zsun)   * Lsun 
    Lum_EUV1      = L_EUV1(mstarZAMS, Zstar_Zsun) * Lsun 
    Lum_EUV2      = L_EUV2(mstarZAMS, Zstar_Zsun) * Lsun 
    Lum_EUV3      = L_EUV3(mstarZAMS, Zstar_Zsun) * Lsun 
    
    # Compute weight factors f_ij that determine the emissivity 
    # and stellar mass in a given cell (i,j):
    σstar = σstar_cell * max( r[2] - r[1], Z[2] - Z[1] )   # σ determining the Gaussian size of the smeared 'star'
    ComputeStellar_fij!( fstar_ij, r, Z, Zstar_pos, σstar )

    # Get emissivities in all bands, for all cells. Weighted 
    # so that the total luminosity over all (infinite) space is 
    # the stellar luminosity for the band in question:
    @inbounds for j in dN_Z
        for i in dN_r 
            js_opt[i,j]  = fstar_ij[i,j] * Lum_opt  / (Four_π * ΔV[i,j])
            js_FUV[i,j]  = fstar_ij[i,j] * Lum_FUV  / (Four_π * ΔV[i,j])
            js_LW[i,j]   = fstar_ij[i,j] * Lum_LW   / (Four_π * ΔV[i,j])
            js_EUV1[i,j] = fstar_ij[i,j] * Lum_EUV1 / (Four_π * ΔV[i,j])
            js_EUV2[i,j] = fstar_ij[i,j] * Lum_EUV2 / (Four_π * ΔV[i,j])
            js_EUV3[i,j] = fstar_ij[i,j] * Lum_EUV3 / (Four_π * ΔV[i,j])
        end
    end

    # Get the stellar mass density in a given cell (i,j), weighted 
    # so that the total stellar mass over all (infinite) space is 
    # mstarZAMS:
    @inbounds for j in dN_Z
        for i in dN_r 
            ρ_star[i,j] = fstar_ij[i,j] * (mstarZAMS * Msun) / ΔV[i,j]
        end
    end


    return StellarRTArrays(
        J_opt, J_opt_old,   J_opt_stage,
        J_FUV, J_FUV_old,   J_FUV_stage,
        J_LW, J_LW_old,     J_LW_stage,
        J_EUV1, J_EUV1_old, J_EUV1_stage, 
        J_EUV2, J_EUV2_old, J_EUV2_stage, 
        J_EUV3, J_EUV3_old, J_EUV3_stage, 

        H_r_opt,        H_Z_opt, 
        H_r_opt_old,    H_Z_opt_old,
        H_r_opt_stage,  H_Z_opt_stage,

        H_r_FUV,        H_Z_FUV,
        H_r_FUV_old,    H_Z_FUV_old,
        H_r_FUV_stage,  H_Z_FUV_stage,
        
        H_r_LW,         H_Z_LW,
        H_r_LW_old,     H_Z_LW_old,
        H_r_LW_stage,   H_Z_LW_stage,

        H_r_EUV1,       H_Z_EUV1, 
        H_r_EUV1_old,   H_Z_EUV1_old,
        H_r_EUV1_stage, H_Z_EUV1_stage,

        H_r_EUV2,       H_Z_EUV2, 
        H_r_EUV2_old,   H_Z_EUV2_old,
        H_r_EUV2_stage, H_Z_EUV2_stage,

        H_r_EUV3,       H_Z_EUV3,
        H_r_EUV3_old,   H_Z_EUV3_old,
        H_r_EUV3_stage, H_Z_EUV3_stage,

        D_rr_opt,  D_ZZ_opt, D_rZ_opt, 
        D_rr_FUV,  D_ZZ_FUV, D_rZ_FUV, 
        D_rr_LW,   D_ZZ_LW, D_rZ_LW,
        D_rr_EUV1, D_ZZ_EUV1, D_rZ_EUV1, 
        D_rr_EUV2, D_ZZ_EUV2, D_rZ_EUV2, 
        D_rr_EUV3, D_ZZ_EUV3, D_rZ_EUV3,

        K_ϕϕ_iphalf_LW,   K_ϕϕ_imhalf_LW,
        K_ϕϕ_iphalf_EUV1, K_ϕϕ_imhalf_EUV1,
        K_ϕϕ_iphalf_EUV2, K_ϕϕ_imhalf_EUV2,
        K_ϕϕ_iphalf_EUV3, K_ϕϕ_imhalf_EUV3,

        F_r_opt,  F_Z_opt,  F_r_Hr_opt,  F_Z_Hr_opt,  F_r_HZ_opt,  F_Z_HZ_opt,
        F_r_FUV,  F_Z_FUV,  F_r_Hr_FUV,  F_Z_Hr_FUV,  F_r_HZ_FUV,  F_Z_HZ_FUV,
        F_r_LW,   F_Z_LW,   F_r_Hr_LW,   F_Z_Hr_LW,   F_r_HZ_LW,   F_Z_HZ_LW,
        F_r_EUV1, F_Z_EUV1, F_r_Hr_EUV1, F_Z_Hr_EUV1, F_r_HZ_EUV1, F_Z_HZ_EUV1,
        F_r_EUV2, F_Z_EUV2, F_r_Hr_EUV2, F_Z_Hr_EUV2, F_r_HZ_EUV2, F_Z_HZ_EUV2,
        F_r_EUV3, F_Z_EUV3, F_r_Hr_EUV3, F_Z_Hr_EUV3, F_r_HZ_EUV3, F_Z_HZ_EUV3,

        α_F_opt, α_F_FUV, α_F_LW, α_F_EUV1, α_F_EUV2, α_F_EUV3,
        Lum_opt, Lum_FUV, Lum_LW, Lum_EUV1, Lum_EUV2, Lum_EUV3,

        js_opt, js_FUV, js_LW, js_EUV1, js_EUV2, js_EUV3, ρ_star,

        Δρu_stellar_r, Δρu_stellar_Z, 
        f_stellar_r,   f_stellar_Z
    )

end

