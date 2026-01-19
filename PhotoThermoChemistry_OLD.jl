
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
include("./GasChemicalHeating.jl")
include("./StellarEmissionAbsorption.jl")  # For band opacities & stellar luminosities 
include("./CosmicRays.jl")


const CFL_RT::Float64                  = 0.4   # CFL factor that determines the advection time-step for stellar & Lyα RT (Default: 0.5. NOTE: Must be < 1)
const DEBUG_PhotoThermoChemistry::Bool = false  # Allow safety checks for physical values etc. Cheap & good for tracking bugs.


# Export constants and functions: 
export CFL_RT, UpdateRTandThermoChemistry!, StellarRTArrays, 
       GenerateStellarRTArrays, FindRSL_StellarRT, FindRSL_LyαRT



const ϵ_RSL::Float64 = 5.0        # RSL is ≥ ϵ_RSL × max(|u|, c_s) (Default: 5.0)
const c̃_min::Float64 = 100 * kms  # Lower bound on RSL, regardless of gas conditions (Default: 100 km/s)

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


""" This function estimates the Lyα RSL needed for sufficiently accurate RHD 
    simulations. """
function FindRSL_LyαRT( c̃_Lyα::Float64,                   # Old reduced speed of light
                          u_r::AbstractArray{Float64, 2}, # r-direction gas velocity 
                          u_Z::AbstractArray{Float64, 2}, # Z-direction gas velocity 
                            P::AbstractArray{Float64, 2}, # Gas pressure 
                            ρ::AbstractArray{Float64, 2}, # Gas density
                            T::AbstractArray{Float64, 2}, # Gas temperature 
                         xHII::AbstractArray{Float64, 2}, # H II abundance fraction 
                          xH2::AbstractArray{Float64, 2}, # H₂ abundance fraction
                          γad::AbstractArray{Float64, 2}, # Gas adiabatic index
                           ΔV::AbstractArray{Float64, 2}  # Cell volumes 
                           )

    N_r, N_Z = size(ρ)
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z

    # Compute (aτ)_eff^3 = ∫dV [a(T) × α]^3:
    aτ_eff_cubed = 0.0       # Initialize aτ_eff^3 (weighted average cubed Neufeld parameter)
    α_eff_cubed  = 0.0       # Initialize α_eff^3  (weighted average cubed Lyα opacity)
    for j in dN_Z 
        @inbounds @simd for i in dN_r 
            # H I number density:
            xHI   = max( 1 - xHII[i,j] - 2 * xH2[i,j], 0.0 )
            nHI   = xHI * X_over_mH * ρ[i,j]

            # Lyα opacity at line center:
            α     = nHI * σ0(T[i,j])

            # Get contribution to aτ_eff^3:
            aα_cubed      = ( a(T[i,j]) * α )^3
            aτ_eff_cubed += ΔV[i,j] * aα_cubed

            # Get contribution to α_eff_cubed (not normalized yet):
            α_eff_cubed  += ΔV[i,j] * (α^3) * aα_cubed
        end
    end

    # Get (aτ)_eff:
    aτ_eff = aτ_eff_cubed^OneThird

    # Get α_eff (weighted average opacity) by taking 1/3-power and normalizing:
    if aτ_eff > 0.0
        α_eff  = ( α_eff_cubed^OneThird ) / aτ_eff
    else
        α_eff  = 0.0
    end

    # Find new RSL:
    c̃_new        = 0.0                 # Initialize value for the new RSL 
    LyαTrapping  = (aτ_eff^OneThird)   # (aτ_eff)^(1/3) 
    for j in dN_Z 
        local_max_c̃  =  c̃_new   # Local variable for tracking largest c̃

        @inbounds for i in dN_r 
            # Gas sound speed:
            c_s   = sqrt( γad[i,j] * P[i,j] / ρ[i,j] )

            # H I number density:
            xHI   = max( 1 - xHII[i,j] - 2 * xH2[i,j], 0.0 )
            nHI   = xHI * X_over_mH * ρ[i,j]

            # Lyα opacity at line center for the cell:
            α     = nHI * σ0(T[i,j])

            # Weight from cell:
            if α_eff > 0.0
                weight  = α / α_eff
            else
                weight  = 0.0
            end

            # Compute minimum RSL for accuracy in this cell to be ~ ϵ_RSL × (t_trap/t_light) × gas velocity:
            c̃_min_cell  = ϵ_RSL * ( 1 + weight * LyαTrapping ) * max( abs(u_r[i,j]), abs(u_Z[i,j]), c_s )

            # Update the largest RSL we've found:
            local_max_c̃ = max( local_max_c̃, c̃_min_cell )
        end

        c̃_new = max( c̃_new, local_max_c̃ )
    end

    # Clamp the new RSL:
    c̃_Lyα_new = min( max( c̃_new, c̃_min ), c )

    # To avoid rapid changes in c̃ if the newly calculated RSL is small, 
    # we further limit the maximum reduction to 25 % from the last time-step:
    if c̃_Lyα_new < c̃_Lyα
        c̃_Lyα_new = max( c̃_Lyα_new, 0.75 * c̃_Lyα )
    end

    return c̃_Lyα_new 
end


""" The total gas heating rate (erg cm⁻³ s⁻ꜝ) from 
    absorption of ionizing and other photons,
    from photoelectric heating, and from H₂ chemical heating. """
function HeatingRate_total( J_opt::Float64, 
                            J_FUV::Float64, 
                             J_LW::Float64,
                           J_EUV1::Float64,
                           J_EUV2::Float64,
                           J_EUV3::Float64,
                            T_old::Float64,
                           Td_old::Float64, # Dust temperature 
                           ne_old::Float64,
                           nH_old::Float64, # nHI_old + nHII_old + 2 × nH2_old (but pre-compute for efficiency)
                          nHI_old::Float64,
                          nH2_old::Float64,
                         nHII_old::Float64,
                         nHeI_old::Float64,
                        nHeII_old::Float64,
                    SelfShielding::Float64,
                               DG::Float64 = 0.0,
                              Γpd::Float64 = 0.0, # H⁻ photodetachment rate 
                            e_Lyα::Float64 = 0.0
                             )

    (  HeatingRate_Compton(J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, ne_old, T_old, e_Lyα )
    +  HeatingRate_LW( SelfShielding * J_LW, nH2_old, nHI_old, T_old )
    +  HeatingRate_EUV1( J_EUV1, nHI_old, nH2_old ) 
    +  HeatingRate_EUV2( J_EUV2, nHI_old, nH2_old, nHeI_old ) 
    +  HeatingRate_EUV3( J_EUV3, nHI_old, nH2_old, nHeI_old, nHeII_old )
    +  HeatingRate_PhotoElectric( J_FUV, J_LW, DG, nH_old, ne_old, T_old, e_Lyα )
    +  HeatingRate_H2chemical( T_old, Td_old, nH_old, nHI_old, nH2_old, nHII_old, ne_old, DG, Γpd )  )

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


""" This function computes the emissivity in the EUV1, EUV2, and EUV3 
    bands from recombinations to the ground state, which is useful in 
    case the OTS approximation is not used.
    
    NOTE: We must use the same band-average photon energy in the recombination 
          emissivities to get consistent ionization rates from recombinations. """
@inline function EUVRecombinationEmissivity( T::Float64, 
                                            ne::Float64, 
                                          nHII::Float64, 
                                         nHeII::Float64, 
                                        nHeIII::Float64 )

    if ne == 0.0
        # In neutral gas, we have no recombinations, so exit early:
        return 0.0, 0.0, 0.0
    end

    # EUV1 emissivity (erg s⁻¹ cm⁻³ sr⁻¹) from recombinations to H I ground state:
    js_EUV1 = krecHII_1(T)   * ne * nHII   * E_EUV1 * inv_Four_π

    # EUV2 emissivity (erg s⁻¹ cm⁻³ sr⁻¹) from recombinations to He I ground state:
    js_EUV2 = krecHeII_1(T)  * ne * nHeII  * E_EUV2 * inv_Four_π

    # EUV3 emissivity (erg s⁻¹ cm⁻³ sr⁻¹) from recombinations to He II ground state:
    js_EUV3 = krecHeIII_1(T) * ne * nHeIII * E_EUV3 * inv_Four_π

    return js_EUV1, js_EUV2, js_EUV3
end


const tol_Abs::Float64  = 0.1 # Reject an absorption update if the relative change in J or H is > tol_Abs
const Td_floor::Float64 = 0.1 # Dust temperature (Td) floor below which we don't care about changes in Td

""" This function updates the intensities J and first moment H from gas + dust absorption/scattering,
    and compute the corresponding change in the gas momentum from radiation pressure. """
function AbsorptionStep( Δt::Float64,
                          c̃::Float64,
                          ρ::Float64,          # Gas density 
                         nH::Float64,          # Total hydrogen density (pre-computed outside, even though it's just a simple sum)
                        nHI::Float64,          # H I number density 
                        nH2::Float64,          # H₂ number density   
                       nHeI::Float64,          # He I number density 
                      nHeII::Float64,          # He II number density   
                         ne::Float64,          # Electron number density (for Thomson scattering)
                         DG::Float64,          # Dust-to-gas ratio
                       Dust::DustState,        # Dust struct, for opacities 
                      T_old::Float64,          # Old gas temperature 
                     Td_old::Float64,          # Old dust temperature 
                       TCMB::Float64,          # CMB temperature 

                  J_opt_old::Float64 = 0.0,
                   dJ_optdt::Float64 = 0.0,
                H_r_opt_old::Float64 = 0.0,
                H_Z_opt_old::Float64 = 0.0,
                  dHr_optdt::Float64 = 0.0,
                  dHZ_optdt::Float64 = 0.0,

                  J_FUV_old::Float64 = 0.0,
                   dJ_FUVdt::Float64 = 0.0,
                H_r_FUV_old::Float64 = 0.0,
                H_Z_FUV_old::Float64 = 0.0,
                  dHr_FUVdt::Float64 = 0.0,
                  dHZ_FUVdt::Float64 = 0.0,

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

                   J_IR_old::Float64 = 0.0,
                 H_r_IR_old::Float64 = 0.0,
                 H_Z_IR_old::Float64 = 0.0,

                        )

    inv_c̃        = 1 / c̃       # Convenient constant 
    ρd           = DG * ρ      # Dust density (dust-to-gas ratio × gas density) 

    # Initialize gas momentum change from stellar radiation pressure:
    Δρu_r_rad    = 0.0
    Δρu_Z_rad    = 0.0

    # Initialize gas momentum change from IR radiation pressure:
    Δρu_r_rad_IR = 0.0
    Δρu_Z_rad_IR = 0.0


    #######################################################################
    #######                                                         #######
    #######                       O P T I C A L                     #######
    #######                                                         #######
    #######################################################################
        
    # Optical opacities:
    α_gas_abs_opt  =   0.0                                      # Optical gas absorption opacity 
    α_gas_F_opt    =   α_gas_abs_opt  +  ne  * σT               # Optical flux opacity (adds Thomson scattering)  
    α_abs_opt      =   α_gas_abs_opt  +  Dust.κd_abs_opt * ρd   # Absorption opacity (for J_opt)
    α_F_opt        =   α_gas_F_opt    +  Dust.κd_F_opt   * ρd   # Flux opacity (for H_opt)

    # Absorption update for optical band:
    J_opt          = max( AbsorptionUpdateJorH( J_opt_old, Δt, dJ_optdt, α_abs_opt, c̃ ), 0.0 )
    H_r_opt        = AbsorptionUpdateJorH( H_r_opt_old, Δt, dHr_optdt, α_F_opt, c̃ )
    H_Z_opt        = AbsorptionUpdateJorH( H_Z_opt_old, Δt, dHZ_optdt, α_F_opt, c̃ )

    # Compute relative changes in optical J, H_r, and H_Z:
    ΔJ_opt         = abs( J_opt   - J_opt_old )   / max( J_opt_old,        J_floor )
    ΔH_r_opt       = abs( H_r_opt - H_r_opt_old ) / max( abs(H_r_opt_old), H_floor )
    ΔH_Z_opt       = abs( H_Z_opt - H_Z_opt_old ) / max( abs(H_Z_opt_old), H_floor )

    Δopt = max( ΔJ_opt, ΔH_r_opt, ΔH_Z_opt )
    if Δopt > tol_Abs  
        # Change in optical was too large. Reject step: 
        REDO = true 
        return REDO, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0,
               Inf64, 0.0, 0.0
    end

    # Add radiation pressure momentum contribution:
    Δρu_r_rad += Four_π_over_c * ( inv_c̃ * (H_r_opt_old - H_r_opt)  +  inv_c̃ * Δt * dHr_optdt )
    Δρu_Z_rad += Four_π_over_c * ( inv_c̃ * (H_Z_opt_old - H_Z_opt)  +  inv_c̃ * Δt * dHZ_optdt )

    #######################################################################
    #######                                                         #######
    #######                       F A R - U V                       #######
    #######                                                         #######
    #######################################################################

    # FUV opacities:
    α_gas_abs_FUV  =   0.0                                      # FUV gas absorption opacity 
    α_gas_F_FUV    =   α_gas_abs_FUV  +  ne  * σT               # FUV flux opacity (adds Thomson scattering)  
    α_abs_FUV      =   α_gas_abs_FUV  +  Dust.κd_abs_FUV * ρd   # Absorption opacity (for J_FUV)
    α_F_FUV        =   α_gas_F_FUV    +  Dust.κd_F_FUV   * ρd   # Flux opacity (for H_FUV)

    # Absorption update for FUV band:
    J_FUV          = max( AbsorptionUpdateJorH( J_FUV_old, Δt, dJ_FUVdt, α_abs_FUV, c̃ ), 0.0 )
    H_r_FUV        = AbsorptionUpdateJorH( H_r_FUV_old, Δt, dHr_FUVdt, α_F_FUV, c̃ )
    H_Z_FUV        = AbsorptionUpdateJorH( H_Z_FUV_old, Δt, dHZ_FUVdt, α_F_FUV, c̃ )

    # Compute relative changes in FUV J, H_r, and H_Z:
    ΔJ_FUV         = abs( J_FUV   - J_FUV_old )   / max( J_FUV_old,        J_floor )
    ΔH_r_FUV       = abs( H_r_FUV - H_r_FUV_old ) / max( abs(H_r_FUV_old), H_floor )
    ΔH_Z_FUV       = abs( H_Z_FUV - H_Z_FUV_old ) / max( abs(H_Z_FUV_old), H_floor )

    ΔFUV = max( ΔJ_FUV, ΔH_r_FUV, ΔH_Z_FUV )
    if ΔFUV > tol_Abs  
        # Change in FUV was too large. Reject step: 
        REDO = true 
        return REDO, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0,
               Inf64, 0.0, 0.0
    end

    # Add radiation pressure momentum contribution:
    Δρu_r_rad += Four_π_over_c * ( inv_c̃ * (H_r_FUV_old - H_r_FUV)  +  inv_c̃ * Δt * dHr_FUVdt )
    Δρu_Z_rad += Four_π_over_c * ( inv_c̃ * (H_Z_FUV_old - H_Z_FUV)  +  inv_c̃ * Δt * dHZ_FUVdt )

    #######################################################################
    #######                                                         #######
    #######                 L Y M A N - W E R N E R                 #######
    #######                                                         #######
    #######################################################################

    # Lyman-Werner opacities:
    α_gas_abs_LW  =   nHI * σHI_LW  +  nH2 * σH2_LW          # LW gas absorption opacity 
    α_gas_F_LW    =   α_gas_abs_LW  +  ne  * σT              # LW flux opacity (adds Thomson scattering)  
    α_abs_LW      =   α_gas_abs_LW  +  Dust.κd_abs_LW * ρd   # Absorption opacity (for J_LW)
    α_F_LW        =   α_gas_F_LW    +  Dust.κd_F_LW   * ρd   # Flux opacity (for H_LW)

    # Absorption update for LW band:
    J_LW          = max( AbsorptionUpdateJorH( J_LW_old, Δt, dJ_LWdt, α_abs_LW, c̃ ), 0.0 )
    H_r_LW        = AbsorptionUpdateJorH( H_r_LW_old, Δt, dHr_LWdt, α_F_LW, c̃ )
    H_Z_LW        = AbsorptionUpdateJorH( H_Z_LW_old, Δt, dHZ_LWdt, α_F_LW, c̃ )

    # Compute relative changes in LW J, H_r, and H_Z:
    ΔJ_LW         = abs( J_LW   - J_LW_old )   / max( J_LW_old,        J_floor )
    ΔH_r_LW       = abs( H_r_LW - H_r_LW_old ) / max( abs(H_r_LW_old), H_floor )
    ΔH_Z_LW       = abs( H_Z_LW - H_Z_LW_old ) / max( abs(H_Z_LW_old), H_floor )

    ΔLW = max( ΔJ_LW, ΔH_r_LW, ΔH_Z_LW )
    if ΔLW > tol_Abs  
        # Change in LW was too large. Reject step: 
        REDO = true 
        return REDO, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0,
               Inf64, 0.0, 0.0
    end

    # Add radiation pressure momentum contribution:
    Δρu_r_rad += Four_π_over_c * ( inv_c̃ * (H_r_LW_old - H_r_LW)  +  inv_c̃ * Δt * dHr_LWdt )
    Δρu_Z_rad += Four_π_over_c * ( inv_c̃ * (H_Z_LW_old - H_Z_LW)  +  inv_c̃ * Δt * dHZ_LWdt )

    #######################################################################
    #######                                                         #######
    #######                         E U V 1                         #######
    #######                                                         #######
    #######################################################################

    # EUV1 opacities:
    α_gas_abs_EUV1  =   nHI * σHI_EUV1    +  nH2 * σH2_EUV1                  # EUV1 gas absorption opacity 
    α_gas_F_EUV1    =   nHI * σHI_EUV1_F  +  nH2 * σH2_EUV1_F  +  ne  * σT   # EUV1 gas flux opacity (adds Thomson scattering)
    α_abs_EUV1      =   α_gas_abs_EUV1  +  Dust.κd_abs_EUV1 * ρd             # Absorption opacity (for J_EUV1)
    α_F_EUV1        =   α_gas_F_EUV1    +  Dust.κd_F_EUV1   * ρd             # Flux opacity (for H_EUV1)

    # Absorption update for EUV1 band:
    J_EUV1        = max( AbsorptionUpdateJorH( J_EUV1_old, Δt, dJ_EUV1dt, α_abs_EUV1, c̃ ), 0.0 )
    H_r_EUV1      = AbsorptionUpdateJorH( H_r_EUV1_old, Δt, dHr_EUV1dt, α_F_EUV1, c̃ )
    H_Z_EUV1      = AbsorptionUpdateJorH( H_Z_EUV1_old, Δt, dHZ_EUV1dt, α_F_EUV1, c̃ )

    # Compute relative changes in EUV1 J, H_r, and H_Z:
    ΔJ_EUV1       = abs( J_EUV1   - J_EUV1_old )   / max( J_EUV1_old,        J_floor )
    ΔH_r_EUV1     = abs( H_r_EUV1 - H_r_EUV1_old ) / max( abs(H_r_EUV1_old), H_floor )
    ΔH_Z_EUV1     = abs( H_Z_EUV1 - H_Z_EUV1_old ) / max( abs(H_Z_EUV1_old), H_floor )

    ΔEUV1 = max( ΔJ_EUV1, ΔH_r_EUV1, ΔH_Z_EUV1 )
    if ΔEUV1 > tol_Abs  
        # Change in EUV1 was too large. Reject step: 
        REDO = true 
        return REDO, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0,
               Inf64, 0.0, 0.0
    end

    # Add radiation pressure momentum contribution:
    Δρu_r_rad += Four_π_over_c * ( inv_c̃ * (H_r_EUV1_old - H_r_EUV1)  +  inv_c̃ * Δt * dHr_EUV1dt )
    Δρu_Z_rad += Four_π_over_c * ( inv_c̃ * (H_Z_EUV1_old - H_Z_EUV1)  +  inv_c̃ * Δt * dHZ_EUV1dt )

    #######################################################################
    #######                                                         #######
    #######                         E U V 2                         #######
    #######                                                         #######
    #######################################################################

    # EUV2 opacities:
    α_gas_abs_EUV2  =   ( nHI * σHI_EUV2  +  nH2 * σH2_EUV2  
                        + nHeI * σHeI_EUV2 )                        # EUV2 gas absorption opacity 
    α_gas_F_EUV2    =   ( nHI * σHI_EUV2_F  +  nH2 * σH2_EUV2_F  
                        + nHeI * σHeI_EUV2_F )  +  ne * σT          # EUV2 gas flux opacity (adds Thomson scattering)
    α_abs_EUV2      =   α_gas_abs_EUV2  +  Dust.κd_abs_EUV2 * ρd    # Absorption opacity (for J_EUV2)
    α_F_EUV2        =   α_gas_F_EUV2    +  Dust.κd_F_EUV2   * ρd    # Flux opacity (for H_EUV2)

    # Absorption update for EUV2 band:
    J_EUV2        = max( AbsorptionUpdateJorH( J_EUV2_old, Δt, dJ_EUV2dt, α_abs_EUV2, c̃ ), 0.0 )
    H_r_EUV2      = AbsorptionUpdateJorH( H_r_EUV2_old, Δt, dHr_EUV2dt, α_F_EUV2, c̃ )
    H_Z_EUV2      = AbsorptionUpdateJorH( H_Z_EUV2_old, Δt, dHZ_EUV2dt, α_F_EUV2, c̃ )

    # Compute relative changes in EUV2 J, H_r, and H_Z:
    ΔJ_EUV2       = abs( J_EUV2   - J_EUV2_old )   / max( J_EUV2_old,        J_floor )
    ΔH_r_EUV2     = abs( H_r_EUV2 - H_r_EUV2_old ) / max( abs(H_r_EUV2_old), H_floor )
    ΔH_Z_EUV2     = abs( H_Z_EUV2 - H_Z_EUV2_old ) / max( abs(H_Z_EUV2_old), H_floor )

    ΔEUV2 = max( ΔJ_EUV2, ΔH_r_EUV2, ΔH_Z_EUV2 )
    if ΔEUV2 > tol_Abs 
        # Change in EUV2 was too large. Reject step: 
        REDO = true 
        return REDO, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0,
               Inf64, 0.0, 0.0
    end

    # Add radiation pressure momentum contribution:
    Δρu_r_rad += Four_π_over_c * ( inv_c̃ * (H_r_EUV2_old - H_r_EUV2)  +  inv_c̃ * Δt * dHr_EUV2dt )
    Δρu_Z_rad += Four_π_over_c * ( inv_c̃ * (H_Z_EUV2_old - H_Z_EUV2)  +  inv_c̃ * Δt * dHZ_EUV2dt )

    #######################################################################
    #######                                                         #######
    #######                         E U V 3                         #######
    #######                                                         #######
    #######################################################################

    # EUV3 opacities:
    α_gas_abs_EUV3   = ( nHI * σHI_EUV3  +  nH2 * σH2_EUV3  +  nHeI * σHeI_EUV3 
                       + nHeII * σHeII_EUV3  )                       # EUV3 gas absorption opacity 

    α_gas_F_EUV3     =   ( nHI * σHI_EUV3_F  +  nH2 * σH2_EUV3_F  +  nHeI * σHeI_EUV3_F
                       + nHeII * σHeII_EUV3_F  )  +  ne * σT         # EUV3 gas flux opacity (adds Thomson scattering)

    α_abs_EUV3       =   α_gas_abs_EUV3  +  Dust.κd_abs_EUV3 * ρd    # Absorption opacity (for J_EUV3)
    α_F_EUV3         =   α_gas_F_EUV3    +  Dust.κd_F_EUV3   * ρd    # Flux opacity (for H_EUV3)

    # Absorption update for EUV3 band:
    J_EUV3           = max( AbsorptionUpdateJorH( J_EUV3_old, Δt, dJ_EUV3dt, α_abs_EUV3, c̃ ), 0.0 )
    H_r_EUV3         = AbsorptionUpdateJorH( H_r_EUV3_old, Δt, dHr_EUV3dt, α_F_EUV3, c̃ )
    H_Z_EUV3         = AbsorptionUpdateJorH( H_Z_EUV3_old, Δt, dHZ_EUV3dt, α_F_EUV3, c̃ )

    # Compute relative changes in EUV2 J, H_r, and H_Z:
    ΔJ_EUV3          = abs( J_EUV3   - J_EUV3_old )   / max( J_EUV3_old,        J_floor )
    ΔH_r_EUV3        = abs( H_r_EUV3 - H_r_EUV3_old ) / max( abs(H_r_EUV3_old), H_floor )
    ΔH_Z_EUV3        = abs( H_Z_EUV3 - H_Z_EUV3_old ) / max( abs(H_Z_EUV3_old), H_floor )

    ΔEUV3 = max( ΔJ_EUV3, ΔH_r_EUV3, ΔH_Z_EUV3 )
    if ΔEUV3 > tol_Abs 
        # Change in EUV3 was too large. Reject step:
        REDO = true 
        return REDO, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0,
               Inf64, 0.0, 0.0
    end

    # Add radiation pressure momentum contribution:
    Δρu_r_rad += Four_π_over_c * ( inv_c̃ * (H_r_EUV3_old - H_r_EUV3)  +  inv_c̃ * Δt * dHr_EUV3dt )
    Δρu_Z_rad += Four_π_over_c * ( inv_c̃ * (H_Z_EUV3_old - H_Z_EUV3)  +  inv_c̃ * Δt * dHZ_EUV3dt )

    #######################################################################
    #######                                                         #######
    #######                        L Y M A N - α                    #######
    #######                                                         #######
    #######################################################################

    # TODO
    e_Lyα = 0.0

    #######################################################################
    #######                                                         #######
    #######                      I N F R A R E D                    #######
    #######                                                         #######
    #######################################################################
    
    # Compute Σ_{B} κ_{B,abs} J_{B} over all non-IR bands:
    κabsB_JB  =  ( Dust.κd_abs_opt * J_opt    +  Dust.κd_abs_FUV * J_FUV 
                 + Dust.κd_abs_LW  * J_LW     +  Dust.κd_abs_EUV1 * J_EUV1  
                 + Dust.κd_abs_EUV2 * J_EUV2  +  Dust.κd_abs_EUV3 * J_EUV3 )

    # IR emissivity & corresponding rate of change in J_IR:
    js_IR     =  ρd * κabsB_JB  +  Λdg(T_old, Td_old, DG, nH) * inv_Four_π
    dJ_IRdt   =  js_IR * c̃

    # Update J_IR:
    J_IR      =  max( AbsorptionUpdateJorH( J_IR_old, Δt, dJ_IRdt, 0.0, c̃ ), 0.0 )

    # Update dust temperature, using the new J_IR and J_{B ≠ IR} bands:
    Td_new    = ComputeTd( J_IR_old, J_opt_old, J_FUV_old, J_LW_old, J_EUV1_old, 
                           J_EUV2_old, J_EUV3_old, Dust, TCMB, T_old, nH, ρ, e_Lyα )

    # Get IR flux opacity (at the new dust temperature):
    α_F_IR    =  ρd * κR(Dust, Td_new)  +  ne * σT

    # Update H_IR:
    H_r_IR    = AbsorptionUpdateJorH( H_r_IR_old, Δt, 0.0, α_F_IR, c̃ )
    H_Z_IR    = AbsorptionUpdateJorH( H_Z_IR_old, Δt, 0.0, α_F_IR, c̃ )

    # Compute relative changes in IR J, H_r, and H_Z. Also compute relative change in Td:
    ΔJ_IR     = abs( J_IR   - J_IR_old )   / max( J_IR_old,        J_floor )
    ΔH_r_IR   = abs( H_r_IR - H_r_IR_old ) / max( abs(H_r_IR_old), H_floor )
    ΔH_Z_IR   = abs( H_Z_IR - H_Z_IR_old ) / max( abs(H_Z_IR_old), H_floor )
    ΔTd       = abs( Td_new - Td_old )     / max( abs(Td_old),     Td_floor )

    ΔIR = max( ΔJ_IR, ΔH_r_IR, ΔH_Z_IR, ΔTd )
    if ΔIR > tol_Abs 
        # Change in IR or dust temperature was too large. Reject step:
        REDO = true 
        return REDO, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0,
               Inf64, 0.0, 0.0
    end

    # Add IR radiation pressure momentum contribution:
    Δρu_r_rad_IR += Four_π_over_c * ( inv_c̃ * (H_r_IR_old - H_r_IR) )
    Δρu_Z_rad_IR += Four_π_over_c * ( inv_c̃ * (H_Z_IR_old - H_Z_IR) )

    # --------------------------------------------------------------------------

    # Get maximum change among all bands (for later determination of next sub-cycle time-step):
    ΔRT  = max( Δopt, ΔFUV, ΔLW, ΔEUV1, ΔEUV2, ΔEUV3, ΔIR )

    # In case the changes in J and H were small enough, we accept the update and return 
    # all computed values: 
    REDO = false
    return REDO, J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, J_IR,
           H_r_opt, H_r_FUV, H_r_LW, H_r_EUV1, H_r_EUV2, H_r_EUV3, H_r_IR,
           H_Z_opt, H_Z_FUV, H_Z_LW, H_Z_EUV1, H_Z_EUV2, H_Z_EUV3, H_Z_IR,
           Δρu_r_rad, Δρu_Z_rad,  Δρu_r_rad_IR, Δρu_Z_rad_IR,
           ΔRT, js_IR, Td_new
end


const tol_HeatCool::Float64       = 0.1
const Quantity_threshold::Float64 = 1e-5
const ϵ_ΔT::Float64               = 1e-3       # Use step-size ϵ_ΔT * T_old to estimate ∂Λ/∂T at T = T_old
const Δt_e_max::Float64           = 1e7 * yr   # Maximum time-step to consider for heating/cooling

""" This function performs a heating/cooling update 
    over a step Δt. It solves the linearized heating/cooling 
    equation using backward Euler (BDF1):
    
    (eⁿ⁺¹ - eⁿ)/Δt = Γ(eⁿ⁺¹) - Λ(eⁿ⁺¹) 
                   ≈ [ Γ(eⁿ) - Λ(eⁿ) ]  +  ( ∂Γ/∂e + ∂Λ/∂e ) × (eⁿ⁺¹ - eⁿ)
          
    The update is accepted if the relative change in e was small (< tol_HeatCool),
    if the updated e is physically acceptable, and if the linearized update was a 
    good approximation to a fully implicit BDF1 update. """
@inline function HeatingCoolingStep( Δt::Float64, 
                                  e_old::Float64,
                                     Td::Float64,
                                    γad::Float64,
                                      n::Float64,
                                     nH::Float64,
                                    nHI::Float64,
                                    nH2::Float64,
                                   nHII::Float64,
                                   nHeI::Float64,
                                  nHeII::Float64,
                                 nHeIII::Float64,
                                    nHD::Float64,
                                     ne::Float64,
                              Zgas_Zsun::Float64,
                                     DG::Float64,
                                    xCI::Float64, 
                                   xCII::Float64, 
                                    xCO::Float64, 
                                    xOI::Float64, 
                                   N_CO::Float64,
                                  J_opt::Float64,
                                  J_FUV::Float64,
                                   J_LW::Float64,
                                 J_EUV1::Float64,
                                 J_EUV2::Float64,
                                 J_EUV3::Float64,
                          SelfShielding::Float64,
                                      z::Float64 = 0.0,
                                    Γpd::Float64 = 0.0,
                                    OTS::Bool    = true,
                                e_floor::Float64 = 0.0
                                    )

    Conv_e_T =  ( γad - 1 ) / (n * kB)   # Conversion factor for e → T 
    T_old    =  e_old * Conv_e_T         # Get temperature T_old corresponding to e_old 

    # Get heating rate (from photoelectric heating, Compton heating off the (non-CMB) radiation field,
    # photo-ionizations/dissociations, UV pumping, and cosmic ray heating):
    HeatingRate = HeatingRate_total( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3,
                                     T_old, Td, ne, nH, nHI, nH2, nHII, nHeI, nHeII, 
                                     SelfShielding, DG, Γpd ) 

    # Compute the cooling rate at e_old (inluding dust-gas collisions):
    CoolingRate = Λ( T_old, nHI, nHII, nHeI, nHeII, nHeIII, ne,
                     nH2, nHD, Zgas_Zsun, xCI, xCII, xCO, xOI, N_CO,
                     z, OTS ) + Λdg( T_old, Td, DG, nH )

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
                        nH2, nHD, Zgas_Zsun, xCI, xCII, xCO, xOI, N_CO,
                        z, OTS ) + Λdg( T_p2, Td, DG, nH )

    CoolingRate_p1 = Λ( T_p1, nHI, nHII, nHeI, nHeII, nHeIII, ne,
                        nH2, nHD, Zgas_Zsun, xCI, xCII, xCO, xOI, N_CO,
                        z, OTS ) + Λdg( T_p1, Td, DG, nH )

    CoolingRate_m1 = Λ( T_m1, nHI, nHII, nHeI, nHeII, nHeIII, ne,
                        nH2, nHD, Zgas_Zsun, xCI, xCII, xCO, xOI, N_CO,
                        z, OTS ) + Λdg( T_m1, Td, DG, nH )

    CoolingRate_m2 = Λ( T_m2, nHI, nHII, nHeI, nHeII, nHeIII, ne,
                        nH2, nHD, Zgas_Zsun, xCI, xCII, xCO, xOI, N_CO,
                        z, OTS ) + Λdg( T_m2, Td, DG, nH )

    # Put it all together to get ∂Λ/∂e:
    ∂CoolingRate∂T = ( - CoolingRate_p2  +  8 * CoolingRate_p1  -  8 * CoolingRate_m1  
                       + CoolingRate_m2 ) * inv_12ΔT

    ∂CoolingRate∂e = ∂CoolingRate∂T * Conv_e_T  # Convert from ∂/∂T to ∂/∂e

    # Next, compute ∂Γ/∂e using a 4th-order central difference approximation: 
    HeatingRate_p2 = HeatingRate_total( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3,
                                        T_p2, Td, ne, nH, nHI, nH2, nHII, nHeI, nHeII, 
                                        SelfShielding, DG, Γpd )

    HeatingRate_p1 = HeatingRate_total( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3,
                                        T_p1, Td, ne, nH, nHI, nH2, nHII, nHeI, nHeII, 
                                        SelfShielding, DG, Γpd )

    HeatingRate_m1 = HeatingRate_total( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3,
                                        T_m1, Td, ne, nH, nHI, nH2, nHII, nHeI, nHeII, 
                                        SelfShielding, DG, Γpd )

    HeatingRate_m2 = HeatingRate_total( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3,
                                        T_m2, Td, ne, nH, nHI, nH2, nHII, nHeI, nHeII, 
                                        SelfShielding, DG, Γpd )

    # Put it all together to get ∂Γ/∂e:
    ∂HeatingRate∂T = ( - HeatingRate_p2  +  8 * HeatingRate_p1  -  8 * HeatingRate_m1  
                       + HeatingRate_m2 ) * inv_12ΔT

    ∂HeatingRate∂e = ∂HeatingRate∂T * Conv_e_T  # Convert from ∂/∂T to ∂/∂e

    # Linearized backward-Euler update of e:
    e_new = e_old  +  ( HeatingRate - CoolingRate ) * Δt / ( 1  +  ∂CoolingRate∂e * Δt )

    # ------- STABILITY & ACCURACY CHECK FOR TEMPERATURE UPDATE -------------------

    # Check that the relative change in e was sufficiently small, and that 
    # other stability/accuracy checks are satisfied:
    Δe    = abs( e_new  -  e_old ) / e_old

    # Tolerance cooling/heating time-scale for accuracy purposes:
    Δt_e  = min( tol_HeatCool * e_old / abs( HeatingRate - CoolingRate ), Δt_e_max )

    # Even if the change in e was small, we also check the residual, so that
    # our linearized solution actually approximates a fully implicit BDF1 solution.

    # The would-be new temperature:
    T_new            = ( γad - 1 ) * e_new  / ( n * kB )   
    
    # Heating rate at would-be new temperature:
    HeatingRate_new  = HeatingRate_total( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3,
                                          T_new, Td, ne, nH, nHI, nH2, nHII, nHeI, nHeII, 
                                          SelfShielding, DG, Γpd )
        
    # Cooling rate at would-be new temperature:
    CoolingRate_new  = Λ( T_new, nHI, nHII, nHeI, nHeII, nHeIII, ne,
                          nH2, nHD, Zgas_Zsun, xCI, xCII, xCO, xOI, N_CO,
                          z, OTS ) + Λdg( T_new, Td, DG, nH )

    # Relative residual checking |(e_old + (Γ - Λ(e_new)*Δtsub) - e_new)| / |e_new - e_old| :
    Res_e            = abs( e_old  +  ( HeatingRate_new  -  CoolingRate_new ) * Δt  -  e_new ) / max( abs( e_new - e_old ), e_floor ) 

    if Δe > tol_HeatCool || e_new ≤ 0.0 || !isfinite(e_new) || Res_e > tol_HeatCool
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
const xmin::Float64             = 1e-10           # Minimum abundance fraction below which we don't care about the 10% rule for primordial species 
const max_iter::Int             = 100             # Maximum number of adaptive refinements of sub-cycle time-step 
const kB_T_floor::Float64       = kB * T_floor    # Convenient constant 
const kB_T_ceil::Float64        = kB * T_ceiling  # Convenient constant 

""" This functions does a single sub-cycle time-step update of
    photon (J and H) absorption, temperature (or rather e),
    and chemical abundances, in that order.

    The 10% rule is used to reduce splitting errors 
    and retain accuracy. In other words, if J or H, e, or chemistry 
    change by more than 10%, the update is rejected, and a smaller sub-step 
    is tried. """
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
                                      xCI_old::Float64,
                                     xCII_old::Float64,
                                      xCO_old::Float64,
                                      xOI_old::Float64,
                                     xOII_old::Float64,
                                    Zgas_Zsun::Float64,
                                         Dust::DustState,

                                    J_opt_old::Float64 = 0.0,
                                     dJ_optdt::Float64 = 0.0,
                                  H_r_opt_old::Float64 = 0.0,
                                  H_Z_opt_old::Float64 = 0.0,
                                    dHr_optdt::Float64 = 0.0,
                                    dHZ_optdt::Float64 = 0.0,

                                    J_FUV_old::Float64 = 0.0,
                                     dJ_FUVdt::Float64 = 0.0,
                                  H_r_FUV_old::Float64 = 0.0,
                                  H_Z_FUV_old::Float64 = 0.0,
                                    dHr_FUVdt::Float64 = 0.0,
                                    dHZ_FUVdt::Float64 = 0.0,

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

                                     J_IR_old::Float64 = 0.0,
                                   H_r_IR_old::Float64 = 0.0,
                                  H_Z_IR_old::Float64 = 0.0,

                                            z::Float64 = 0.0,
                                          OTS::Bool    = true,  # On-the-spot approximation
                               CoolingHeating::Bool    = true
                                      )

    # Safety checks:
    if DEBUG_PhotoThermoChemistry
        @assert isfinite(ρ) && ρ > 0.0 "ρ = $(ρ) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(e_old)        "e_old = $(e_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"

        @assert isfinite(J_opt_old) && J_opt_old ≥ 0.0 "J_opt_old = $(J_opt_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_r_opt_old)                  "H_r_opt_old = $(H_r_opt_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_Z_opt_old)                  "H_Z_opt_old = $(H_Z_opt_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"

        @assert isfinite(J_FUV_old) && J_FUV_old ≥ 0.0 "J_FUV_old = $(J_FUV_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_r_FUV_old)                  "H_r_FUV_old = $(H_r_FUV_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_Z_FUV_old)                  "H_Z_FUV_old = $(H_Z_FUV_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"

        @assert isfinite(J_LW_old) && J_LW_old ≥ 0.0 "J_LW_old = $(J_LW_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_r_LW_old)                 "H_r_LW_old = $(H_r_LW_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_Z_LW_old)                 "H_Z_LW_old = $(H_Z_LW_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"

        @assert isfinite(J_EUV1_old) && J_EUV1_old ≥ 0.0 "J_EUV1_old = $(J_EUV1_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_r_EUV1_old)                   "H_r_EUV1_old = $(H_r_EUV1_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_Z_EUV1_old)                   "H_Z_EUV1_old = $(H_Z_EUV1_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"

        @assert isfinite(J_EUV2_old) && J_EUV2_old ≥ 0.0 "J_EUV2_old = $(J_EUV2_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_r_EUV2_old)                   "H_r_EUV2_old = $(H_r_EUV2_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_Z_EUV2_old)                   "H_Z_EUV2_old = $(H_Z_EUV2_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"

        @assert isfinite(J_EUV3_old) && J_EUV3_old ≥ 0.0 "J_EUV3_old = $(J_EUV3_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_r_EUV3_old)                   "H_r_EUV3_old = $(H_r_EUV3_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_Z_EUV3_old)                   "H_Z_EUV3_old = $(H_Z_EUV3_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"

        @assert isfinite(xH2_old)                      "xH2_old = $(xH2_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xHII_old)                     "xHII_old = $(xHII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xHD_old)                      "xHD_old = $(xHD_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xDII_old)                     "xDII_old = $(xDII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xHeII_old)                    "xHeII_old = $(xHeII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xHeIII_old)                   "xHeIII_old = $(xHeIII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xCI_old)                      "xCI_old = $(xCI_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xCII_old)                     "xCII_old = $(xCII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xCO_old)                      "xCO_old = $(xCO_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xOI_old)                      "xOI_old = $(xOI_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xOII_old)                     "xOII_old = $(xOII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xHII_old + 2 * xH2_old ≤ 1.0           "xHII_old + 2 * xH2_old = $(xHII_old + 2 * xH2_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xDII_old + xHD_old ≤ 1.0               "xDII_old + xHD_old = $(xDII_old + xHD_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xHeII_old + xHeIII_old ≤ 1.0           "xHeII_old + xHeIII_old = $(xHeII_old + xHeIII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xCI_old + xCII_old + xCO_old ≤ 1.0     "xCI_old + xCII_old + xCO_old = $(xCI_old + xCII_old + xCO_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xOI_old + xOII_old ≤ 1.0               "xOI_old + xOII_old = $(xOI_old + xOII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
    end

    # --------- Get gas phase metal density -----------------------------------------------------
    ρZ_old            =   Zgas_old * ρ_old 

    # --------- Get relevant number densities & abundance fractions  ----------------------------
    nH_old            =   X_over_mH * ρ_old 
    nD_old            =   D_H_ratio * nH_old
    nHe_old           =   Y_over_mHe * ρ_old 
    nC_old            =   Zgas_Zsun * C_H_ratio * nH_old
    nO_old            =   Zgas_Zsun * O_H_ratio * nH_old
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
    nCI_old           =   xCI_old    * nC_old
    nCII_old          =   xCII_old   * nC_old
    nOI_old           =   xOI_old    * nO_old
    nOII_old          =   xOII_old   * nO_old
    nCO_old           =   xCO_old    * nC_old
    nCHx_old          =   max( (1 - xCI_old  - xCII_old - xCO_old), 0.0 ) * nC_old 
    nOHx_old          =   max( (1 - xOI_old  - xOII_old - (nCO_old / nO_old)), 0.0 ) * nO_old 
    ne_old            =   GetElectronDensity( nHII_old, nHeII_old, nHeIII_old ) 
    n                 =   nH_old  +  nHe_old  +  ne_old 
                               
    # --------- Get adiabatic index for old state ---------------------------------------------
    γad               =   Getγad( xHII_old, xH2_old, xHeII_old, xHeIII_old )

    # --------- Convenient constants (avoid re-computing) -------------------------------------
    mH_inv_kBρ        =   mH / ( kB * ρ )
    inv_xH2_old       =   1  / max( xH2_old,  xmin )             # Used in 10% rule check 
    inv_xHII_old      =   1  / max( xHII_old, xmin )             # Used in 10% rule check 
    inv_xCO_old       =   1  / max( xCO_old,  xmin )             # Used in 10% rule check 
    inv_xCII_old      =   1  / max( xCII_old, xmin )             # Used in 10% rule check 
    inv_xOI_old       =   1  / max( xOI_old,  xmin )             # Used in 10% rule check 
    inv_xOII_old      =   1  / max( xOII_old, xmin )             # Used in 10% rule check 
    inv_ne_old        =   1  / max( ne_old,   xmin * nH_old )    # Used in 10% rule check
    inv_γ_minus_1     =   1  / ( γad - 1 )
    e_floor           =   n * kB_T_floor * inv_γ_minus_1         # Used in limiting updated e
    e_ceiling         =   n * kB_T_ceil  * inv_γ_minus_1         # Used in limiting updated e

    # --------- Clamp e_old and get T_old -----------------------------------------------------
    e_old             =   min( max( e_old,  e_floor ), e_ceiling )
    T_old             =   ( γad - 1 ) * e_old  / ( n * kB )

    # --------- Get dust temperature ------------------------------------------------------------
    TCMB              =   TCMB0 * (1 + z)
    Td                =   ComputeTd( J_IR_old, J_opt_old, J_FUV_old, J_LW_old, J_EUV1_old, 
                                     J_EUV2_old, J_EUV3_old, Dust, TCMB, T_old, nH_old, ρ_old  )

    # --------- Get dust-to-gas ratio & dust density ----------------
    DG                =   Dust.DG
    if Td < 1300.0
        DG_new = DG[icell, jcell]
        ρd     = DG_new * ρ
    else
        # Dust sublimated:
        DG[icell, jcell] = 0.0
        DG_new           = DG[icell, jcell]
        ρd               = 0.0
    end

    # --------- Recombination photons (if OTS is not used) ------------------------------------
    if !OTS 
        js_rec_EUV1, js_rec_EUV2, js_rec_EUV3 = EUVRecombinationEmissivity( T_old, ne_old, nHII_old, nHeII_old, nHeIII_old )

        # Add emissivities to calculation of dJ/dt in the EUV1, EUV2, EUV3 bands:
        dJ_EUV1dt += js_rec_EUV1 * c̃
        dJ_EUV2dt += js_rec_EUV2 * c̃
        dJ_EUV3dt += js_rec_EUV3 * c̃
    end

    # -----------------------------------------------------------------------------------------

    # Sobolev-type column length estimate ~ cell width + density variation scale (following 
    # e.g. Hopkins+ 2023, for FIRE-3). We limit ΔL to ≤ ΔL_max to avoid infinities if ∇ρ ~ 0:
    ΔL   = min( min( Δr, ΔZ )  +  ρ_old  / abs(∇ρ),  ΔL_max )

    # Estimated H₂, HD, and CO column densities:
    N_H2 = nH2_old * ΔL 
    N_HD = nHD_old * ΔL 
    N_CO = nCO_old * ΔL 

    # Get self-shielding factors:
    SelfShielding         = S_H2( N_H2, T_old ) 
    SelfShielding_HD      = S_HD( N_HD, T_old )
    SelfShielding_H2_HD   = S_H2_HD( N_H2 )
    
    Δtsub                 = Δtsubguess # Initialize first guess of sub-cycle time-step
    iter::Int             = 0          # Initialize iteration count
    ΔxH2_rel              = 0.0        # Initialize ΔxH2_rel, for printing in case convergence fails 
    ΔxHII_rel             = 0.0        # Initialize ΔxHII_rel, for printing in case convergence fails 
    Δne_rel               = 0.0        # Initialize Δne_rel, for printing in case convergence fails 
    Δe                    = 0.0        # Initialize Δe, for printing in case convergence fails 
    ConvergedCarbonOxygen = true       # Initialize check for carbon & oxygen chemistry 
    ΔStellarRT            = 0.0        # Initialize ΔStellarRT, for printing in case convergence fails 
    REDO_ABSORPTION       = false
    REDO_HEATINGCOOLING   = false
    while iter < max_iter 
        iter          += 1
        Δtsub_half     = 0.5 * Δtsub

        Δρu_r_trial    = 0.0  # Initialize the computed change in r-direction gas momentum from stellar radiation pressure 
        Δρu_Z_trial    = 0.0  # Initialize the computed change in Z-direction gas momentum from stellar radiation pressure 

        Δρu_r_IR_trial = 0.0  # Initialize the computed change in r-direction gas momentum from IR radiation pressure 
        Δρu_Z_IR_trial = 0.0  # Initialize the computed change in Z-direction gas momentum from IR radiation pressure 

        #######################################################################
        #######                                                         #######
        #######            S T E P  1  :  A B S O R P T I O N           #######
        #######                                                         #######
        #######################################################################

        # First we update J and H for all stellar bands, and the dust temperature,
        # over a step Δtsub:
        
        REDO_ABSORPTION, J_opt_trial, J_FUV_trial, J_LW_trial, J_EUV1_trial, J_EUV2_trial, J_EUV3_trial, J_IR_trial,
        H_r_opt_trial, H_r_FUV_trial, H_r_LW_trial, H_r_EUV1_trial, H_r_EUV2_trial, H_r_EUV3_trial, H_r_IR_trial,
        H_Z_opt_trial, H_Z_FUV_trial, H_Z_LW_trial, H_Z_EUV1_trial, H_Z_EUV2_trial, H_Z_EUV3_trial, H_Z_IR_trial,
        Δρu_r_trial_Abs, Δρu_Z_trial_Abs,  Δρu_r_IR_trial_Abs, Δρu_Z_IR_trial_Abs,
        ΔStellarRT, js_IR, Td_new = AbsorptionStep( Δtsub, c̃, ρ_old, nH_old, nHI_old, nH2_old, nHeI_old, nHeII_old, ne_old,
                                                    DG_new, Dust, T_old, Td, TCMB,
                                                    J_opt_old,  dJ_optdt,   H_r_opt_old,  H_Z_opt_old,  dHr_optdt,  dHZ_optdt,
                                                    J_FUV_old,  dJ_FUVdt,   H_r_FUV_old,  H_Z_FUV_old,  dHr_FUVdt,  dHZ_FUVdt,
                                                    J_LW_old,   dJ_LWdt,    H_r_LW_old,   H_Z_LW_old,   dHr_LWdt,   dHZ_LWdt,
                                                    J_EUV1_old, dJ_EUV1dt,  H_r_EUV1_old, H_Z_EUV1_old, dHr_EUV1dt, dHZ_EUV1dt,
                                                    J_EUV2_old, dJ_EUV2dt,  H_r_EUV2_old, H_Z_EUV2_old, dHr_EUV2dt, dHZ_EUV2dt,
                                                    J_EUV3_old, dJ_EUV3dt,  H_r_EUV3_old, H_Z_EUV3_old, dHr_EUV3dt, dHZ_EUV3dt,
                                                    J_IR_old,   H_r_IR_old, H_Z_IR_old )

        if REDO_ABSORPTION
            # Change in J or H were too large, so reject update and try a smaller sub-cycle time-step: 
            Δtsub = Δtsub_half
            continue 
        else
            # If changes in J and H were small enough, we accept the trial updated J and H:
            J_opt  = J_opt_trial;  J_FUV  = J_FUV_trial;  J_LW   = J_LW_trial
            J_EUV1 = J_EUV1_trial; J_EUV2 = J_EUV2_trial; J_EUV3 = J_EUV3_trial 
            J_IR   = J_IR_trial

            H_r_opt  = H_r_opt_trial;  H_r_FUV  = H_r_FUV_trial;  H_r_LW   = H_r_LW_trial
            H_r_EUV1 = H_r_EUV1_trial; H_r_EUV2 = H_r_EUV2_trial; H_r_EUV3 = H_r_EUV3_trial
            H_r_IR   = H_r_IR_trial
            
            H_Z_opt  = H_Z_opt_trial;  H_Z_FUV  = H_Z_FUV_trial;  H_Z_LW   = H_Z_LW_trial
            H_Z_EUV1 = H_Z_EUV1_trial; H_Z_EUV2 = H_Z_EUV2_trial; H_Z_EUV3 = H_Z_EUV3_trial
            H_Z_IR   = H_Z_IR_trial

            # Also add stellar rad. pressure momentum change:
            Δρu_r_trial    += Δρu_r_trial_Abs
            Δρu_Z_trial    += Δρu_Z_trial_Abs

            # And IR rad. pressure momentum change:
            Δρu_r_IR_trial += Δρu_r_IR_trial_Abs
            Δρu_Z_IR_trial += Δρu_Z_IR_trial_Abs
        end

        # Get photo-ionization/dissociation rates (s⁻¹), for use in chemistry 
        # and heating below:
        Γ_LW        =  PhotoDissRate_H2( SelfShielding * J_LW )
        Γpd         =  PhotoDetachRate_Hminus( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3 )
        Γ_LW_HD     =  PhotoDissRate_HD( SelfShielding_HD * SelfShielding_H2_HD * J_LW )
        ΓionH2      =  IonRate_H2( J_EUV1, J_EUV2, J_EUV3 ) 
        ΓionHI      =  IonRate_HI( J_EUV1, J_EUV2, J_EUV3 ) 
        ΓionDI      =  ΓionHI
        ΓionHeI     =  IonRate_HeI( J_EUV2, J_EUV3 ) 
        ΓionHeII    =  IonRate_HeII( J_EUV3 )
        ΓionCI      =  IonRate_CI( J_LW, J_EUV1, J_EUV2, J_EUV3 )
        ΓionOI      =  IonRate_OI( J_EUV1, J_EUV2, J_EUV3 )
        ΓdissCHx    =  PhotoDissRate_CHₓ( J_opt, J_FUV, J_LW )
        ΓdissOHx    =  PhotoDissRate_OHₓ( J_FUV, J_LW )
        ΓdissCO     =  PhotoDissRate_CO( J_LW )

        #######################################################################
        #######                                                         #######
        #######      S T E P  2  :  H E A T I N G / C O O L I N G       #######
        #######                                                         #######
        #######################################################################

        # Next we do an update of e from heating and cooling, over a 
        # substep Δtsub.

        REDO_HEATINGCOOLING, e_new, 
        Δe, Δt_e = HeatingCoolingStep( Δtsub, e_old, Td_new, γad, 
                                       n, nH_old, nHI_old, nH2_old, nHII_old, nHeI_old, nHeII_old,
                                       nHeIII_old, nHD_old, ne_old,  Zgas_Zsun, DG_new,
                                       xCI_old, xCII_old, xCO_old, xOI_old, N_CO,
                                       J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, 
                                       SelfShielding, z, Γpd, OTS, e_floor )

        if REDO_HEATINGCOOLING
            # Change in e was too large or unphysical, so retry with 
            # smaller sub-cycle time-step:
            Δtsub = min( Δtsub_half,  
                         Δt_e )
            continue 
        end 

        if CoolingHeating 
            e_new = min( max( e_new, e_floor ),
                         e_ceiling )
        else
            # For some tests (e.g. R-type ionization front) we ignore heating/cooling:
            e_new = e_old
        end

        ###################################################################################
        #######                                                                     #######
        #######     S T E P  3  :  D U S T  G R O W T H / D E S T R U C T I O N     #######
        #######                                                                     #######
        ###################################################################################

        # Next, update the dust & metal density, from considering dust growth (by accretion)
        # and destruction (by sublimation + sputtering).

        ρd_new, ρZ_new = DustGrowthDestruction_Update( ρd_old, ρZ_old, nH_old, 
                                                       T, Td_new, Zgr_C )

        DG_new        = ρd_new / ρ
        Zgas_Zsun_new = ρZ_new / ρ

        ###################################################################################
        #######                                                                     #######
        #######                    S T E P  4  :  C H E M I S T R Y                 #######
        #######                                                                     #######
        ###################################################################################

        # Finally, we update the chemistry over a sub-step Δtsub.

        # Get the normalized temperature Tnorm = T / [μ(γ-1)] = e × mH / (kB × ρ):
        Tnorm =  e_new * mH_inv_kBρ 

        # Update chemistry:
        ( xH2_new, xHI_new, xHII_new, xHD_new, xDI_new, xDII_new,
          xHeI_new, xHeII_new, xHeIII_new,
          xOI_new, xOII_new, xCI_new, xCII_new, xCO_new, ConvergedCarbonOxygen,
          ne_new, Δne ) =
          UpdateChemistrySubStep( Δtsub, Tnorm, Td_new, ρ, DG_new, xH2_old, xHII_old, xHD_old, xDII_old, xHeII_old, xHeIII_old,
                                  xOI_old, xOII_old, xCI_old, xCII_old, xCO_old, Zgas_Zsun,
                                  ΓionHI, ΓionH2, ΓionDI, ΓionHeI, ΓionHeII, Γ_LW, Γpd, Γ_LW_HD, 
                                  ΓionOI, ΓionCI, ΓdissCO, ΓdissOHx, ΓdissCHx,
                                  z, OTS ) 

        # Compute would-be relative change in H₂, H II, and
        # e⁻ fractions/number densities (NOTE: Δne is already taking abs into account, hence Δne > 0):
        ΔxH2_rel    = abs( xH2_new   - xH2_old )  * inv_xH2_old
        ΔxHII_rel   = abs( xHII_new  - xHII_old ) * inv_xHII_old
        ΔxCO_rel    = abs( xCO_new   - xCO_old )  * inv_xCO_old 
        ΔxCII_rel   = abs( xCII_new  - xCII_old ) * inv_xCII_old 
        ΔxOI_rel    = abs( xOI_new   - xOI_old )  * inv_xOI_old
        ΔxOII_rel   = abs( xOII_new  - xOII_old ) * inv_xOII_old 
        Δne_rel     = Δne * inv_ne_old

        # Get the maximum relative change among them:
        Δx_rel      = maximum( ( ΔxH2_rel, ΔxHII_rel, ΔxCO_rel, ΔxCII_rel, ΔxOI_rel, ΔxOII_rel, Δne_rel ) )

        if Δx_rel > tol_Chem || ConvergedCarbonOxygen == false 
            # Change in H₂, H II, e⁻, CO, C II, O I, and/or O II 
            # fraction was too high, reduce Δtsub & retry:
            Δtsub = Δtsub_half
            continue    
        end

        # Recommended next sub-cycle time-step based on the relative change 
        # in quantities:
        if ( Δx_rel < tol_enlargestep  &&  Δe < tol_enlargestep  &&  ΔStellarRT < tol_enlargestep )
            Δtsub_new = 2 * min( Δtsub, Δt_e )
        else
            Δtsub_new = Δtsub
        end

        # Return every updated quantity, the time-step taken, & recommended next sub-cycle time-step:
        return Δtsub, Δtsub_new, e_new, xH2_new, xHII_new, xHD_new, xDII_new, xHeII_new, xHeIII_new,
               xOI_new, xOII_new, xCI_new, xCII_new, xCO_new, 
               J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, J_IR,
               H_r_opt, H_Z_opt,    H_r_FUV, H_Z_FUV,    H_r_LW, H_Z_LW,  
               H_r_EUV1, H_Z_EUV1,  H_r_EUV2, H_Z_EUV2,  H_r_EUV3, H_Z_EUV3,
               H_r_IR, H_Z_IR,
               Δρu_r_trial, Δρu_Z_trial,  Δρu_r_IR_trial, Δρu_Z_IR_trial, Td_new, js_IR

    end

    # Otherwise print error:
    error( """ ThermoChemistrySubStepUpdate could not converge within $(max_iter) iterations in cell (i,j) = ($(icell),$(jcell)),
               and simulation has therefore terminated.
               
               Relative changes in quantities in this cell during the final attempted sub-cycle time-step were:
               
               - Relative change in e⁻ abundance, Δne_rel    = $(Δne_rel)
               - Relative change in H₂ abundance, ΔxH2_rel   = $(ΔxH2_rel)
               - Relative change in H II abundance, ΔxHII_rel = $(ΔxHII_rel)
               - Relative change in e, Δe_rel = $(Δe)
               - Converged carbon & oxygen chemistry?: $(ConvergedCarbonOxygen)
               - Relative change in stellar RT, ΔStellarRT = $(ΔStellarRT)
               - REDO_ABSORPTION     = $(REDO_ABSORPTION)
               - REDO_HEATINGCOOLING = $(REDO_HEATINGCOOLING) """ )

end # End of ThermoChemistrySubStepUpdate!



max_steps::Int64 = 1000000 # Maximum number of sub-cycling sub-steps per global time-step (default = 10⁶)

""" The full thermochemistry sub-cycling over a global time-step Δtglobal.
    Updates the intensities, gas internal energy, and chemical abundances. """
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
                                               xOI_old::Float64, 
                                              xOII_old::Float64, 
                                               xCI_old::Float64, 
                                              xCII_old::Float64, 
                                               xCO_old::Float64,
                                             Zgas_Zsun::Float64,
                                                  Dust::DustState,

                                         Δρu_stellar_r::AbstractArray{Float64, 2},
                                         Δρu_stellar_Z::AbstractArray{Float64, 2},
                                              Δρu_IR_r::AbstractArray{Float64, 2},
                                              Δρu_IR_Z::AbstractArray{Float64, 2},

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

                                           J_IR_transp::Float64 = 0.0,
                                         H_r_IR_transp::Float64 = 0.0,
                                         H_Z_IR_transp::Float64 = 0.0,

                                                     z::Float64 = 0.0,
                                                   OTS::Bool    = true,
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
    xOI_current      = xOI_old
    xOII_current     = xOII_old
    xCI_current      = xCI_old
    xCII_current     = xCII_old
    xCO_current      = xCO_old
    Td               = 30.0     # Dust temperature can be initialized arbitrarily 

    # Similarly, initialize the intensities J as those from 
    # the transport step:
    J_opt_current    = J_opt_transp
    J_FUV_current    = J_FUV_transp
    J_LW_current     = J_LW_transp
    J_EUV1_current   = J_EUV1_transp
    J_EUV2_current   = J_EUV2_transp
    J_EUV3_current   = J_EUV3_transp
    J_IR_current     = J_IR_transp

    # Initialize H components:
    H_r_opt_current  =  H_r_opt_transp
    H_Z_opt_current  =  H_Z_opt_transp

    H_r_FUV_current  =  H_r_FUV_transp
    H_Z_FUV_current  =  H_Z_FUV_transp

    H_r_LW_current   =  H_r_LW_transp
    H_Z_LW_current   =  H_Z_LW_transp

    H_r_EUV1_current =  H_r_EUV1_transp
    H_Z_EUV1_current =  H_Z_EUV1_transp

    H_r_EUV2_current =  H_r_EUV2_transp
    H_Z_EUV2_current =  H_Z_EUV2_transp

    H_r_EUV3_current =  H_r_EUV3_transp
    H_Z_EUV3_current =  H_Z_EUV3_transp

    H_r_IR_current   =  H_r_IR_transp
    H_Z_IR_current   =  H_Z_IR_transp

    # Initialize IR emissivity:
    js_IR = 0.0
    
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
                  xCI_current, xCII_current, xCO_current, 
                  xOI_current, xOII_current,
                  Zgas_Zsun,
                  Dust,

                  J_opt_current, dJ_optdt, 
                  H_r_opt_current, H_Z_opt_current, dHr_optdt, dHZ_optdt,

                  J_FUV_current, dJ_FUVdt,
                  H_r_FUV_current, H_Z_FUV_current, dHr_FUVdt, dHZ_FUVdt,

                  J_LW_current, dJ_LWdt,
                  H_r_LW_current, H_Z_LW_current, dHr_LWdt, dHZ_LWdt,

                  J_EUV1_current,   dJ_EUV1dt, 
                  H_r_EUV1_current, H_Z_EUV1_current, dHr_EUV1dt, dHZ_EUV1dt,

                  J_EUV2_current, dJ_EUV2dt, 
                  H_r_EUV2_current, H_Z_EUV2_current, dHr_EUV2dt, dHZ_EUV2dt,

                  J_EUV3_current, dJ_EUV3dt,
                  H_r_EUV3_current, H_Z_EUV3_current, dHr_EUV3dt, dHZ_EUV3dt,

                  J_IR_current, 
                  H_r_IR_current, H_Z_IR_current, 

                  z, OTS, CoolingHeating )

        # Unpack result:
        Δtsub, Δtsub_new, e_new, xH2_new, xHII_new, xHD_new, xDII_new, xHeII_new, xHeIII_new, 
        xOI_new, xOII_new, xCI_new, xCII_new, xCO_new,
        J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, J_IR,
        H_r_opt, H_Z_opt,    H_r_FUV, H_Z_FUV,    H_r_LW, H_Z_LW,  
        H_r_EUV1, H_Z_EUV1,  H_r_EUV2, H_Z_EUV2,  H_r_EUV3, H_Z_EUV3,
        H_r_IR, H_Z_IR,
        Δρu_r_substep, Δρu_Z_substep,  
        Δρu_r_IR_substep, Δρu_Z_IR_substep, Td, js_IR  = result

        # Update our current internal energy & abundance fractions:
        e_current        = e_new
        xH2_current      = xH2_new
        xHII_current     = xHII_new
        xHD_current      = xHD_new 
        xDII_current     = xDII_new
        xHeII_current    = xHeII_new
        xHeIII_current   = xHeIII_new
        xOI_current      = xOI_new
        xOII_current     = xOII_new
        xCI_current      = xCI_new
        xCII_current     = xCII_new
        xCO_current      = xCO_new
        # Update our current intensities:
        J_opt_current    = J_opt
        J_FUV_current    = J_FUV 
        J_LW_current     = J_LW 
        J_EUV1_current   = J_EUV1
        J_EUV2_current   = J_EUV2
        J_EUV3_current   = J_EUV3 
        J_IR_current     = J_IR
        # Update our current H-components:
        H_r_opt_current  = H_r_opt
        H_Z_opt_current  = H_Z_opt 
        H_r_FUV_current  = H_r_FUV
        H_Z_FUV_current  = H_Z_FUV
        H_r_LW_current   = H_r_LW
        H_Z_LW_current   = H_Z_LW 
        H_r_EUV1_current = H_r_EUV1
        H_Z_EUV1_current = H_Z_EUV1
        H_r_EUV2_current = H_r_EUV2 
        H_Z_EUV2_current = H_Z_EUV2 
        H_r_EUV3_current = H_r_EUV3
        H_Z_EUV3_current = H_Z_EUV3
        H_r_IR_current   = H_r_IR
        H_Z_IR_current   = H_Z_IR

        # Add stellar radiation pressure momentum from this sub-step: 
        Δρu_stellar_r[icell, jcell] += Δρu_r_substep
        Δρu_stellar_Z[icell, jcell] += Δρu_Z_substep

        # Add IR radiation pressure momentum from this sub-step: 
        Δρu_IR_r[icell, jcell]      += Δρu_r_IR_substep
        Δρu_IR_Z[icell, jcell]      += Δρu_Z_IR_substep

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

    # Return the updated state after evolving over the global time-step:
    return e_current, xH2_current, xHII_current, xHD_current, xDII_current, xHeII_current, xHeIII_current, 
           xOI_current, xOII_current, xCI_current, xCII_current, xCO_current,
           J_opt_current, J_FUV_current, J_LW_current, J_EUV1_current, J_EUV2_current, J_EUV3_current, 
           J_IR_current,
           H_r_opt_current,  H_Z_opt_current,   H_r_FUV_current,  H_Z_FUV_current,
           H_r_LW_current,   H_Z_LW_current,    H_r_EUV1_current, H_Z_EUV1_current,  
           H_r_EUV2_current, H_Z_EUV2_current,  H_r_EUV3_current, H_Z_EUV3_current, 
           H_r_IR_current,   H_Z_IR_current, 
           Td, js_IR
end


function UpdateRTandThermoChemistry!( Δt_RT_max::Float64, 
                                             RT::StellarRTArrays,
                                     HydroState::HydrodynamicsState,
                                      ChemState::ChemistryState,
                                           Dust::DustState,
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

                                          # Gas velocity field:
                                       u_r_star::AbstractArray{Float64,2},
                                       u_Z_star::AbstractArray{Float64,2},
                                        ∂u_r_∂r::AbstractArray{Float64,2},
                                        ∂u_r_∂Z::AbstractArray{Float64,2}, 
                                        ∂u_Z_∂r::AbstractArray{Float64,2}, 
                                        ∂u_Z_∂Z::AbstractArray{Float64,2},
                                     u_r_over_r::AbstractArray{Float64,2},

                                           # Lyα stuff:
                                       LyaState::Union{LyαArrays,Nothing},
                                            Lyα::Bool = false,

                                           # Boundary conditions:
                                      BC_r_left::Symbol = :Reflect,
                                     BC_r_right::Symbol = :Outflow,
                                       BC_Z_bot::Symbol = :Reflect,
                                       BC_Z_top::Symbol = :Outflow
                                              )

    c̃         = RT.c̃
    Δt_CFL_RT = GetRHDCFLTimeStep( r, Z, c̃, HydroState.u_X, HydroState.u_Y, CFL_RT ) 
    Δt_RT     = min( Δt_CFL_RT, Δt_RT_max )  
    c̃Δt_RT    = c̃ * Δt_RT 
    
    if Lyα
        @assert LyaState.c̃_Lyα == c̃ "Lyα and stellar RT c̃ mismatch in the photothermochemistry solver."
    end
    
    # Unpack hydro & chemistry arrays:
    ρ         = HydroState.ρ
    e         = HydroState.e 
    T         = HydroState.T
    xH2       = ChemState.xH2
    xHII      = ChemState.xHII
    xHD       = ChemState.xHD
    xDII      = ChemState.xDII
    xHeII     = ChemState.xHeII 
    xHeIII    = ChemState.xHeIII 
    xCI       = ChemState.xCI 
    xCII      = ChemState.xCII
    xCO       = ChemState.xCO
    xOI       = ChemState.xOI 
    xOII      = ChemState.xOII
    Zgas_Zsun = ChemState.Zgas_Zsun

    # On-spot-approximation (Case B) or modelling of recombination radiation:
    OTS       = RT.OTS

    #######################################################################
    #######                                                         #######
    #######       S T E P  1  :  I N J E C T I O N  S T E P         #######
    #######                                                         #######
    #######################################################################

    # Stellar emission update: 
    RT.J_opt      .= RT.J_opt   .+  RT.js_opt  .* c̃ .* Δt_RT
    RT.J_FUV      .= RT.J_FUV   .+  RT.js_FUV  .* c̃ .* Δt_RT
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

    # Select Strang-split order:
    if RT.rfirst_StellarRT == true 
        # If the last Strang-split order was r-first, change to Z-first:
        RT.rfirst_StellarRT = false 
    else
        # If the last Strang-split order was Z-first, change to r-first:
        RT.rfirst_StellarRT = true
    end

    UpdateStellarJold = false # Don't update Jold, since we want the pre-(injection + transport) value to compute dJ/dt

    # Advect IR photons:
    AdvectJandH!( c̃Δt_RT,   RT.α_F_IR,  RT.J_IR,     RT.J_IR_old,   RT.H_r_IR,   RT.H_Z_IR, 
                  RT.D_rr_IR,        RT.D_ZZ_IR,  RT.D_rZ_IR, 
                  RT.K_ϕϕ_iphalf_IR, RT.K_ϕϕ_imhalf_IR,
                  RT.F_r_IR,         RT.F_Z_IR,
                  RT.F_r_Hr_IR,      RT.F_Z_Hr_IR, 
                  RT.F_r_HZ_IR,      RT.F_Z_HZ_IR, 
                  inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell,
                  u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                  UpdateStellarJold, true, BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top,
                  RT.rfirst_StellarRT, Val(:Threaded) )

    # Advect optical photons:
    AdvectJandH!( c̃Δt_RT,   RT.α_F_opt,  RT.J_opt,     RT.J_opt_old,   RT.H_r_opt,   RT.H_Z_opt, 
                  RT.D_rr_opt,        RT.D_ZZ_opt,  RT.D_rZ_opt, 
                  RT.K_ϕϕ_iphalf_opt, RT.K_ϕϕ_imhalf_opt,
                  RT.F_r_opt,         RT.F_Z_opt,
                  RT.F_r_Hr_opt,      RT.F_Z_Hr_opt, 
                  RT.F_r_HZ_opt,      RT.F_Z_HZ_opt, 
                  inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell,
                  u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                  UpdateStellarJold, true, BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top,
                  RT.rfirst_StellarRT, Val(:Threaded) )

    # Advect FUV photons:
    AdvectJandH!( c̃Δt_RT,   RT.α_F_FUV,  RT.J_FUV,     RT.J_FUV_old,   RT.H_r_FUV,   RT.H_Z_FUV, 
                  RT.D_rr_FUV,        RT.D_ZZ_FUV,  RT.D_rZ_FUV, 
                  RT.K_ϕϕ_iphalf_FUV, RT.K_ϕϕ_imhalf_FUV,
                  RT.F_r_FUV,         RT.F_Z_FUV,
                  RT.F_r_Hr_FUV,      RT.F_Z_Hr_FUV, 
                  RT.F_r_HZ_FUV,      RT.F_Z_HZ_FUV, 
                  inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell,
                  u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                  UpdateStellarJold, true, BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top,
                  RT.rfirst_StellarRT, Val(:Threaded) )

    # Advect LW photons:
    AdvectJandH!( c̃Δt_RT,   RT.α_F_LW,  RT.J_LW,     RT.J_LW_old,   RT.H_r_LW,   RT.H_Z_LW, 
                  RT.D_rr_LW,        RT.D_ZZ_LW,  RT.D_rZ_LW, 
                  RT.K_ϕϕ_iphalf_LW, RT.K_ϕϕ_imhalf_LW,
                  RT.F_r_LW,         RT.F_Z_LW,
                  RT.F_r_Hr_LW,      RT.F_Z_Hr_LW, 
                  RT.F_r_HZ_LW,      RT.F_Z_HZ_LW, 
                  inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell,
                  u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                  UpdateStellarJold, true, BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top,
                  RT.rfirst_StellarRT, Val(:Threaded) )

    # Advect EUV1 photons:
    AdvectJandH!( c̃Δt_RT,   RT.α_F_EUV1,  RT.J_EUV1,     RT.J_EUV1_old,   RT.H_r_EUV1,   RT.H_Z_EUV1, 
                  RT.D_rr_EUV1,        RT.D_ZZ_EUV1,  RT.D_rZ_EUV1, 
                  RT.K_ϕϕ_iphalf_EUV1, RT.K_ϕϕ_imhalf_EUV1,
                  RT.F_r_EUV1,         RT.F_Z_EUV1,
                  RT.F_r_Hr_EUV1,      RT.F_Z_Hr_EUV1, 
                  RT.F_r_HZ_EUV1,      RT.F_Z_HZ_EUV1, 
                  inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell,
                  u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                  UpdateStellarJold, true, BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top,
                  RT.rfirst_StellarRT, Val(:Threaded) )

    # Advect EUV2 photons:
    AdvectJandH!( c̃Δt_RT,   RT.α_F_EUV2,  RT.J_EUV2,     RT.J_EUV2_old,   RT.H_r_EUV2,   RT.H_Z_EUV2, 
                  RT.D_rr_EUV2,        RT.D_ZZ_EUV2,  RT.D_rZ_EUV2, 
                  RT.K_ϕϕ_iphalf_EUV2, RT.K_ϕϕ_imhalf_EUV2,
                  RT.F_r_EUV2,         RT.F_Z_EUV2,
                  RT.F_r_Hr_EUV2,      RT.F_Z_Hr_EUV2, 
                  RT.F_r_HZ_EUV2,      RT.F_Z_HZ_EUV2, 
                  inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell,
                  u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                  UpdateStellarJold, true, BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top,
                  RT.rfirst_StellarRT, Val(:Threaded) )

    # Advect EUV3 photons:
    AdvectJandH!( c̃Δt_RT,   RT.α_F_EUV3,  RT.J_EUV3,     RT.J_EUV3_old,   RT.H_r_EUV3,   RT.H_Z_EUV3, 
                  RT.D_rr_EUV3,        RT.D_ZZ_EUV3,  RT.D_rZ_EUV3, 
                  RT.K_ϕϕ_iphalf_EUV3, RT.K_ϕϕ_imhalf_EUV3,
                  RT.F_r_EUV3,         RT.F_Z_EUV3,
                  RT.F_r_Hr_EUV3,      RT.F_Z_Hr_EUV3, 
                  RT.F_r_HZ_EUV3,      RT.F_Z_HZ_EUV3, 
                  inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell,
                  u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                  UpdateStellarJold, true, BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top,
                  RT.rfirst_StellarRT, Val(:Threaded) )
    

    #######################################################################
    #######                                                         #######
    #######       S T E P  3  :  T H E R M O - C H E M I S T R Y    #######
    #######                                                         #######
    #######################################################################

    # After the transport step, we sub-cycle absorption, emission, chemistry,
    # and cooling/heating for all cells.
                                              
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
                                                       xOI[i,j],   xOII[i,j],
                                                       xCI[i,j],   xCII[i,j],
                                                       xCO[i,j],   Zgas_Zsun,
                                                           Dust, 

                                                 RT.Δρu_stellar_r, RT.Δρu_stellar_Z,
                                                      RT.Δρu_IR_r, RT.Δρu_IR_Z,

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

                                                 RT.J_IR[i,j], 
                                               RT.H_r_IR[i,j], RT.H_Z_IR[i,j], 

                                                          z, OTS, CoolingHeating )

                # Get updated quantities:
                e[i,j], xH2[i,j], xHII[i,j], xHD[i,j], xDII[i,j], xHeII[i,j], xHeIII[i,j], 
                xOI[i,j], xOII[i,j], xCI[i,j], xCII[i,j], xCO[i,j],
                RT.J_opt[i,j], RT.J_FUV[i,j], RT.J_LW[i,j], RT.J_EUV1[i,j], RT.J_EUV2[i,j], RT.J_EUV3[i,j],
                RT.J_IR[i,j],
                RT.H_r_opt[i,j],  RT.H_Z_opt[i,j],
                RT.H_r_FUV[i,j],  RT.H_Z_FUV[i,j],
                RT.H_r_LW[i,j],   RT.H_Z_LW[i,j],
                RT.H_r_EUV1[i,j], RT.H_Z_EUV1[i,j],  
                RT.H_r_EUV2[i,j], RT.H_Z_EUV2[i,j],
                RT.H_r_EUV3[i,j], RT.H_Z_EUV3[i,j],
                RT.H_r_IR[i,j],   RT.H_Z_IR[i,j],
                Dust.Td[i,j],     RT.js_IR[i,j]  = result 

                # Update IR flux opacity (for accurate M1 transport in optically thick regime):
                RT.α_F_IR[i,j] = Dust.DG[i,j] * ρ[i,j] * κR(Dust, Dust.Td[i,j])

                # Update stellar RT flux opacities:
                #xHI = max( 1 - xHII[i,j] - 2 * xH2[i,j], 0.0 )
                #nHI = xHI * X_over_mH * ρ[i,j]
                #nH2 = xH2[i,j] * X_over_mH * ρ[i,j]

                #RT.α_F_EUV1[i,j] = nHI * σHI_EUV1_F  + nH2 * σH2_EUV1_F +  Dust.DG[i,j] * ρ[i,j] * Dust.κd_F_EUV1
                #RT.α_F_EUV2[i,j] = nHI * σHI_EUV2_F  +  Dust.DG[i,j] * ρ[i,j] * Dust.κd_F_EUV2
                #RT.α_F_EUV3[i,j] = nHI * σHI_EUV3_F  +  Dust.DG[i,j] * ρ[i,j] * Dust.κd_F_EUV3
                #RT.α_F_FUV[i,j]  = Dust.DG[i,j] * ρ[i,j] * Dust.κd_F_FUV


            end
        end # End of i-loop 
    end # End of j-loop 


    # Update the old stellar J and H:
    @threads for j in dN_Z
        @simd for i in dN_r 
            @inbounds begin
                # Update old J:
                RT.J_IR_old[i,j]     = RT.J_IR[i,j]
                RT.J_opt_old[i,j]    = RT.J_opt[i,j]
                RT.J_FUV_old[i,j]    = RT.J_FUV[i,j]
                RT.J_LW_old[i,j]     = RT.J_LW[i,j]
                RT.J_EUV1_old[i,j]   = RT.J_EUV1[i,j]
                RT.J_EUV2_old[i,j]   = RT.J_EUV2[i,j]
                RT.J_EUV3_old[i,j]   = RT.J_EUV3[i,j]

                # Update old H:
                RT.H_r_IR_old[i,j]   = RT.H_r_IR[i,j]
                RT.H_Z_IR_old[i,j]   = RT.H_Z_IR[i,j]

                RT.H_r_opt_old[i,j]  = RT.H_r_opt[i,j]
                RT.H_Z_opt_old[i,j]  = RT.H_Z_opt[i,j]

                RT.H_r_FUV_old[i,j]  = RT.H_r_FUV[i,j]
                RT.H_Z_FUV_old[i,j]  = RT.H_Z_FUV[i,j]

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

    # If Lyα RT is turned on, we also call the Lyα solver over the same time-step,
    # assuming same RSL (set outside to be equal in the Update RHD function):
    if Lyα 
        N_ν  = length( LyaState.νC )
        dN_ν = 1:N_ν

        # Update α (H I opacity @ line center), pd (destruction probability),
        # and ∫dν_js_Lyα (frequency-integrated emissivity):
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
                if LyaState.LyαDestruction 
                    LyaState.pd[i,j] = pd_Lyα( nHII, ne, nHI, nHeI, nH2, T[i,j], z )
                else
                    LyaState.pd[i,j] = 0.0
                end

                # Update the frequency integrated emissivity: 
                LyaState.∫dν_js_Lyα[i,j] = ( ∫dν_js_recLyα( T[i,j], nHII, ne, true, true )
                                         +   ∫dν_js_collLyα( T[i,j], nHI, ne ) )

            end # End of i-loop
        end # End of j-loop 

        # Update the Lyα Voigt profile & emission profile for all cells:
        UpdateVoigtAndEmissionProfile!( LyaState, T )

        # Update the flux opacity using the newly updated Hvoigt,
        # and the dust/continuum opacities:
        @threads for k in dN_ν 
            @inbounds @simd for j in dN_Z 
                for i in dN_r
                    # Lyα dust absorption & flux opacities:
                    ρd         = Dust.DG[i,j] * ρ[i,j]  # Dust density 
                    αd_abs_Lyα = ρd * Dust.κd_abs_Lyα   # Lyα dust absorption opacity 
                    αd_F_Lyα   = ρd * Dust.κd_F_Lyα     # Lyα dust flux opacity 

                    # Total Lyα absorption & flux opacities:
                    LyaState.αc_Lyα[i,j]    = αd_abs_Lyα
                    LyaState.α_F_Lyα[i,j,k] = LyaState.α_Lyα[i,j] * LyaState.Hvoigt[i,j,k]  +  αd_F_Lyα
                end # End of i-loop
            end # End of j-loop 
        end # End of k-loop
        
        # Lyα emission:
        LyαEmissionUpdate!( LyaState.J_Lyα, LyaState.∫dν_js_Lyα,
                            LyaState.ϕ_ν, c̃, Δt_RT )

        # Lyα Advection + Absorption (including rad. pressure) + frequency diffusion & Doppler shift:
        UpdateLyαJandH!( Δt_RT, c̃, LyaState, inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell, T,
                         u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                         true, BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top )

    end # End of Lyα update 

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
                                 σstar_cell::Float64,
                                          c̃::Float64,
                                        OTS::Bool = false  # On-the-spot approximation (if true) or not (if false)
                                 )
   
    # Grid size: 
    N_r, N_Z = size( ΔV )
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z

    # Convenient arrays: 
    zeros_c()  = zeros(Float64, N_r,   N_Z)
    zeros_rf() = zeros(Float64, N_r+1, N_Z)   # For r-interface fluxes 
    zeros_zf() = zeros(Float64, N_r,   N_Z+1) # For Z-interface fluxes 

    # Intensities (latest & old) for all bands: 
    J_IR     = zeros_c();  J_IR_old   = copy(J_IR)
    J_opt    = zeros_c();  J_opt_old  = copy(J_opt)
    J_FUV    = zeros_c();  J_FUV_old  = copy(J_FUV)
    J_LW     = zeros_c();  J_LW_old   = copy(J_LW)
    J_EUV1   = zeros_c();  J_EUV1_old = copy(J_EUV1)
    J_EUV2   = zeros_c();  J_EUV2_old = copy(J_EUV2)
    J_EUV3   = zeros_c();  J_EUV3_old = copy(J_EUV3)

    # H ≡ ∫dΩ I n̂ / 4π arrays for all bands:
    H_r_IR         = zeros_c();  H_Z_IR        = zeros_c()
    H_r_IR_old     = zeros_c();  H_Z_IR_old    = zeros_c()

    H_r_opt        = zeros_c();  H_Z_opt       = zeros_c()
    H_r_opt_old    = zeros_c();  H_Z_opt_old   = zeros_c()

    H_r_FUV        = zeros_c();  H_Z_FUV       = zeros_c()
    H_r_FUV_old    = zeros_c();  H_Z_FUV_old   = zeros_c()

    H_r_LW         = zeros_c();  H_Z_LW        = zeros_c()
    H_r_LW_old     = zeros_c();  H_Z_LW_old    = zeros_c()

    H_r_EUV1       = zeros_c();  H_Z_EUV1      = zeros_c()
    H_r_EUV1_old   = zeros_c();  H_Z_EUV1_old  = zeros_c()

    H_r_EUV2       = zeros_c();  H_Z_EUV2      = zeros_c()
    H_r_EUV2_old   = zeros_c();  H_Z_EUV2_old  = zeros_c()

    H_r_EUV3       = zeros_c();  H_Z_EUV3      = zeros_c()
    H_r_EUV3_old   = zeros_c();  H_Z_EUV3_old  = zeros_c()

    # Eddington tensor components for all bands:
    D_rr_IR   = fill(1/3, N_r, N_Z);  D_ZZ_IR   = fill(1/3, N_r, N_Z);  D_rZ_IR   = zeros_c()
    D_rr_opt  = fill(1/3, N_r, N_Z);  D_ZZ_opt  = fill(1/3, N_r, N_Z);  D_rZ_opt  = zeros_c()
    D_rr_FUV  = fill(1/3, N_r, N_Z);  D_ZZ_FUV  = fill(1/3, N_r, N_Z);  D_rZ_FUV  = zeros_c()
    D_rr_LW   = fill(1/3, N_r, N_Z);  D_ZZ_LW   = fill(1/3, N_r, N_Z);  D_rZ_LW   = zeros_c()
    D_rr_EUV1 = fill(1/3, N_r, N_Z);  D_ZZ_EUV1 = fill(1/3, N_r, N_Z);  D_rZ_EUV1 = zeros_c()
    D_rr_EUV2 = fill(1/3, N_r, N_Z);  D_ZZ_EUV2 = fill(1/3, N_r, N_Z);  D_rZ_EUV2 = zeros_c()
    D_rr_EUV3 = fill(1/3, N_r, N_Z);  D_ZZ_EUV3 = fill(1/3, N_r, N_Z);  D_rZ_EUV3 = zeros_c()

    # ϕϕ-component of radiation pressure tensor K = D × J at interfaces, 
    # needed for geometric source term:
    K_ϕϕ_iphalf_IR   = zeros_c();     K_ϕϕ_imhalf_IR   = zeros_c()
    K_ϕϕ_iphalf_opt  = zeros_c();     K_ϕϕ_imhalf_opt  = zeros_c()
    K_ϕϕ_iphalf_FUV  = zeros_c();     K_ϕϕ_imhalf_FUV  = zeros_c()
    K_ϕϕ_iphalf_LW   = zeros_c();     K_ϕϕ_imhalf_LW   = zeros_c()
    K_ϕϕ_iphalf_EUV1 = zeros_c();     K_ϕϕ_imhalf_EUV1 = zeros_c()
    K_ϕϕ_iphalf_EUV2 = zeros_c();     K_ϕϕ_imhalf_EUV2 = zeros_c()
    K_ϕϕ_iphalf_EUV3 = zeros_c();     K_ϕϕ_imhalf_EUV3 = zeros_c()

    # Interface fluxes for all bands: 
    F_r_IR      = zeros_rf(); F_Z_IR      = zeros_zf()
    F_r_Hr_IR   = zeros_rf(); F_Z_Hr_IR   = zeros_zf()
    F_r_HZ_IR   = zeros_rf(); F_Z_HZ_IR   = zeros_zf()

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
    α_F_IR        = zeros_c()
    α_F_opt       = zeros_c()
    α_F_FUV       = zeros_c()
    α_F_LW        = zeros_c()
    α_F_EUV1      = zeros_c()
    α_F_EUV2      = zeros_c()
    α_F_EUV3      = zeros_c()

    # Emissivities for all bands:
    js_IR         = zeros_c()
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
    Δρu_IR_r      = zeros_c()
    Δρu_IR_Z      = zeros_c()
    f_stellar_r   = zeros_c()    # Stellar rad. pressure force per unit volume in r-direction
    f_stellar_Z   = zeros_c()    # Stellar rad. pressure force per unit volume in Z-direction
    f_IR_r        = zeros_c()    # IR rad. pressure force per unit volume in r-direction
    f_IR_Z        = zeros_c()    # IR rad. pressure force per unit volume in Z-direction

    # Get luminosities in all bands:
    Lum_IR        = 0.0                  # Initialize IR luminosity as zero
    Lum_opt       = L_opt(mstarZAMS, Zstar_Zsun)  * Lsun 
    Lum_FUV       = L_FUV(mstarZAMS, Zstar_Zsun)  * Lsun
    Lum_LW        = L_LW(mstarZAMS, Zstar_Zsun)   * Lsun 
    Lum_EUV1      = L_EUV1(mstarZAMS, Zstar_Zsun) * Lsun 
    Lum_EUV2      = L_EUV2(mstarZAMS, Zstar_Zsun) * Lsun 
    Lum_EUV3      = L_EUV3(mstarZAMS, Zstar_Zsun) * Lsun 

    # Get photon emission rates in all bands:
    dN_dt_opt     = dNdt_opt(mstarZAMS, Zstar_Zsun)
    dN_dt_FUV     = dNdt_FUV(mstarZAMS, Zstar_Zsun)
    dN_dt_LW      = dNdt_LW(mstarZAMS, Zstar_Zsun)
    dN_dt_EUV1    = dNdt_EUV1(mstarZAMS, Zstar_Zsun)
    dN_dt_EUV2    = dNdt_EUV2(mstarZAMS, Zstar_Zsun)
    dN_dt_EUV3    = dNdt_EUV3(mstarZAMS, Zstar_Zsun)
    dN_dt_LyC     = dN_dt_EUV1 + dN_dt_EUV2 + dN_dt_EUV3
    
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

    # Get stellar luminosities in actual simulated volume (will be 0.5 * L for 
    # reflective BCs, but we compute it explicitly here to be sure):
    fstar_ij_sum = sum( fstar_ij )
    Lum_opt_vol  = Lum_opt  * fstar_ij_sum
    Lum_FUV_vol  = Lum_FUV  * fstar_ij_sum
    Lum_LW_vol   = Lum_LW   * fstar_ij_sum
    Lum_EUV1_vol = Lum_EUV1 * fstar_ij_sum
    Lum_EUV2_vol = Lum_EUV2 * fstar_ij_sum
    Lum_EUV3_vol = Lum_EUV3 * fstar_ij_sum


    # Get the stellar mass density in a given cell (i,j), weighted 
    # so that the total stellar mass over all (infinite) space is 
    # mstarZAMS:
    @inbounds for j in dN_Z
        for i in dN_r 
            ρ_star[i,j] = fstar_ij[i,j] * (mstarZAMS * Msun) / ΔV[i,j]
        end
    end

    # Initialize Strang split order:
    rfirst_StellarRT = true 

    # Initialize LyC and LW escape fractions:
    f_esc_LyC = 0.0
    f_esc_LW  = 0.0 

    # Initialize direct radiation pressure 'force multipliers':
    M_F_dir             = 0.0
    M_F_dir_antigravity = 0.0

    # Initialize IR radiation pressure force multipliers:
    M_F_IR              = 0.0
    M_F_IR_antigravity  = 0.0


    return StellarRTArrays( 
        c̃,

        J_IR,   J_IR_old,  
        J_opt,  J_opt_old,  
        J_FUV,  J_FUV_old,  
        J_LW,   J_LW_old,     
        J_EUV1, J_EUV1_old, 
        J_EUV2, J_EUV2_old,  
        J_EUV3, J_EUV3_old, 

        H_r_IR,         H_Z_IR, 
        H_r_IR_old,     H_Z_IR_old,

        H_r_opt,        H_Z_opt, 
        H_r_opt_old,    H_Z_opt_old,

        H_r_FUV,        H_Z_FUV,
        H_r_FUV_old,    H_Z_FUV_old,
        
        H_r_LW,         H_Z_LW,
        H_r_LW_old,     H_Z_LW_old,

        H_r_EUV1,       H_Z_EUV1, 
        H_r_EUV1_old,   H_Z_EUV1_old,

        H_r_EUV2,       H_Z_EUV2, 
        H_r_EUV2_old,   H_Z_EUV2_old,

        H_r_EUV3,       H_Z_EUV3,
        H_r_EUV3_old,   H_Z_EUV3_old,

        D_rr_IR,  D_ZZ_IR, D_rZ_IR, 
        D_rr_opt,  D_ZZ_opt, D_rZ_opt, 
        D_rr_FUV,  D_ZZ_FUV, D_rZ_FUV, 
        D_rr_LW,   D_ZZ_LW, D_rZ_LW,
        D_rr_EUV1, D_ZZ_EUV1, D_rZ_EUV1, 
        D_rr_EUV2, D_ZZ_EUV2, D_rZ_EUV2, 
        D_rr_EUV3, D_ZZ_EUV3, D_rZ_EUV3,

        K_ϕϕ_iphalf_IR,   K_ϕϕ_imhalf_IR,
        K_ϕϕ_iphalf_opt,  K_ϕϕ_imhalf_opt,
        K_ϕϕ_iphalf_FUV,  K_ϕϕ_imhalf_FUV,
        K_ϕϕ_iphalf_LW,   K_ϕϕ_imhalf_LW,
        K_ϕϕ_iphalf_EUV1, K_ϕϕ_imhalf_EUV1,
        K_ϕϕ_iphalf_EUV2, K_ϕϕ_imhalf_EUV2,
        K_ϕϕ_iphalf_EUV3, K_ϕϕ_imhalf_EUV3,

        F_r_IR,   F_Z_IR,   F_r_Hr_IR,   F_Z_Hr_IR,   F_r_HZ_IR,   F_Z_HZ_IR,
        F_r_opt,  F_Z_opt,  F_r_Hr_opt,  F_Z_Hr_opt,  F_r_HZ_opt,  F_Z_HZ_opt,
        F_r_FUV,  F_Z_FUV,  F_r_Hr_FUV,  F_Z_Hr_FUV,  F_r_HZ_FUV,  F_Z_HZ_FUV,
        F_r_LW,   F_Z_LW,   F_r_Hr_LW,   F_Z_Hr_LW,   F_r_HZ_LW,   F_Z_HZ_LW,
        F_r_EUV1, F_Z_EUV1, F_r_Hr_EUV1, F_Z_Hr_EUV1, F_r_HZ_EUV1, F_Z_HZ_EUV1,
        F_r_EUV2, F_Z_EUV2, F_r_Hr_EUV2, F_Z_Hr_EUV2, F_r_HZ_EUV2, F_Z_HZ_EUV2,
        F_r_EUV3, F_Z_EUV3, F_r_Hr_EUV3, F_Z_Hr_EUV3, F_r_HZ_EUV3, F_Z_HZ_EUV3,

        α_F_IR, α_F_opt, α_F_FUV, α_F_LW, α_F_EUV1, α_F_EUV2, α_F_EUV3,

        Lum_IR, Lum_opt, Lum_FUV, Lum_LW, Lum_EUV1, Lum_EUV2, Lum_EUV3,

        Lum_opt_vol, Lum_FUV_vol, Lum_LW_vol, Lum_EUV1_vol, Lum_EUV2_vol, Lum_EUV3_vol,

        dN_dt_opt, dN_dt_FUV, dN_dt_LW, dN_dt_EUV1, dN_dt_EUV2, dN_dt_EUV3, dN_dt_LyC,

        js_IR, js_opt, js_FUV, js_LW, js_EUV1, js_EUV2, js_EUV3, ρ_star,

        Δρu_stellar_r, Δρu_stellar_Z, 
        Δρu_IR_r,      Δρu_IR_Z, 
        f_stellar_r,   f_stellar_Z,
        f_IR_r,        f_IR_Z,

        rfirst_StellarRT, 

        f_esc_LyC, f_esc_LW,

        M_F_dir, M_F_dir_antigravity,

        M_F_IR, M_F_IR_antigravity,

        OTS
    )

end

