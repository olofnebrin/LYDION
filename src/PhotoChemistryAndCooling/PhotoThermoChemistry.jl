
""" FILE:       PhotoThermoChemistry.jl

    PURPOSE:    Contains the full thermochemistry + dust + stellar RT update.
    
"""

using .Structs: LyαArrays, StellarRTArrays, ChemistryState, 
       HydrodynamicsState, DustState, GeometryCylindrical, GeometryCartesian,
       StellarBandAverages, 
       iopt_RT, iFUV_RT, iLW_RT, iEUV1_RT, iEUV2_RT, iEUV3_RT, iIR_RT, 
       Nbands_RT, dNbands_RT
using .ConstantsAndFunctions
using  Base.Threads               # For CPU acceleration
using  SpecialFunctions           # For erf

using Logging
debug_logger = ConsoleLogger(stderr, Logging.Debug)

include(joinpath(@__DIR__, "..", "FreqFactors.jl"))
include(joinpath(@__DIR__, "..", "LymanAlphaEmission.jl"))
include(joinpath(@__DIR__, "..", "LymanAlphaDestruction.jl"))
include("./GasChemistryRates.jl")
include("./UpdateChemistry.jl")
include("./GasCoolingRates.jl")
include("./GasChemicalHeating.jl")
include("./StellarEmissionAbsorption.jl")  # For band opacities & stellar luminosities 
include("./CosmicRays.jl")


const CFL_RT::Float64                  = 0.4             # CFL factor that determines the advection time-step for stellar, IR & Lyα RT (Default: 0.4. NOTE: Must be < 1)
const C_RP::Float64                    = 0.1             # Sets the radiation pressure-limited maximum global time-step (Default = 0.1, should be < 1)
const C_RP_Dust::Float64               = 0.4             # Sets the radiation pressure-limited maximum global time-step for the pure dust contribution (Default = 0.4, since dust is typically coupled to gas)
const ϵ_RSL::Float64                   = 4.0             # RSL is ≥ ϵ_RSL × max(|u|, c_s) (Default: 4.0)
const max_steps::Int64                 = 1000000         # Maximum number of sub-cycling sub-steps per global time-step (default = 10⁶)
const DEBUG_PhotoThermoChemistry::Bool = false           # Allow safety checks for physical values etc. Cheap & good for tracking bugs.


# Export constants and functions: 
export CFL_RT, GetRadPressureTimeStep_GasAcceleration, GetRadPressureTimeStep_DustAcceleration,
       GetRadPressureTimeStep_GasDustAcceleration,
       UpdateRTandThermoChemistry!, StellarRTArrays, 
       GenerateStellarRTArrays, GenerateStarClusterRTArrays, GenerateStellarRTArrays_Idealized,
       FindRSL_StellarRT, FindRSL_LyαRT

    

""" Get the radiation pressure acceleration time-step limiter for gas, considering 
    only the acceleration of gas by absorbed/scatter photons *by the gas only*. 
    
    NOTE: This does not take into account coupling to dust. """
function GetRadPressureTimeStep_GasAcceleration( r::AbstractVector{Float64}, # Interface X-values
                                                 Z::AbstractVector{Float64}, # Interface Y-values
                                                 ρ::AbstractMatrix{Float64},
                                                RT::StellarRTArrays,
                                          LyαState::Union{LyαArrays, Nothing},
                                         Chemistry::ChemistryState,
                                           StellAv::StellarBandAverages = DefaultStellarBandAverages
                                                 )

    N_r  = length( r ) - 1
    N_Z  = length( Z ) - 1
    dN_r = 1:N_r 
    dN_Z = 1:N_Z

    min_Δt   = Inf64  # Initialize the minimum local time-scale
    @inbounds for j in dN_Z
        ΔZ           = Z[j+1] - Z[j] # Vertical cell width
        local_min_Δt = min_Δt        # Local variable for tracking smallest time-step 

        @simd for i in dN_r 

            # Compute the horizontal cell width (same for all j-indices):
            Δr       = r[i+1] - r[i]

            # Get all needed gas number densities:
            nHI, nHII, nH2, nHeI, nHeII, nHeIII, nDI, nDII, nHD, 
            nCI, nCII, nCIII, nCO, nCHx, nOI, nOII, nOIII, nOIV, nOHx, 
            nSI, nSII, nSIII, nSIV, ne, n, nH, nHe, nD, 
            nC, nO, nS = GetAllNumberDensities( Chemistry.xH2[i,j], Chemistry.xHII[i,j], 
                                                Chemistry.xHeII[i,j], Chemistry.xHeIII[i,j], 
                                                Chemistry.xHD[i,j], Chemistry.xDII[i,j],
                                                Chemistry.xCI[i,j], Chemistry.xCII[i,j], Chemistry.xCIII[i,j], 
                                                Chemistry.xCO[i,j],
                                                Chemistry.xOI[i,j],  Chemistry.xOII[i,j], Chemistry.xOIII[i,j],
                                                Chemistry.xOIV[i,j], Chemistry.xSII[i,j], Chemistry.xSIII[i,j],
                                                Chemistry.xSIV[i,j],
                                                ρ[i,j], 
                                                Chemistry.ZC[i,j], Chemistry.ZSil[i,j], Chemistry.Zgasrest[i,j],
                                                Chemistry.HydrogenAndHelium )

            # Get radiation pressure acceleration on gas, a_rad = (4π / c ρ) ∫dν α_F(ν) H(ν):
            a_rad_r  = 0.0  # Initialize r-direction acceleration 
            a_rad_Z  = 0.0  # Initialize Z-direction acceleration

            # Normalization constant 4π / c ρ to be multiplied by at the end:
            Normalization = Four_π_over_c / ρ[i,j]

            # Flux opacity from electron (Thomson) scattering:
            α_F_Thomson   = ne * σT

            # (1) IR: 
            α_F_IR   = α_F_Thomson

            a_rad_r += α_F_IR * RT.H_r[iIR_RT,i,j]
            a_rad_Z += α_F_IR * RT.H_Z[iIR_RT,i,j]

            # (2) optical:
            α_F_opt  = nCHx * σCH_opt  +  α_F_Thomson

            a_rad_r += α_F_opt * RT.H_r[iopt_RT,i,j]
            a_rad_Z += α_F_opt * RT.H_Z[iopt_RT,i,j]

            # (3) FUV:
            α_F_FUV  = nOHx * σOH_FUV  +  nCHx * σCH_FUV  +  nSI * σSI_FUV_E  +  α_F_Thomson

            a_rad_r += α_F_FUV * RT.H_r[iFUV_RT,i,j]
            a_rad_Z += α_F_FUV * RT.H_Z[iFUV_RT,i,j]

            # (4) LW:
            α_F_LW   = ( nHI * σHI_LW    +  nH2  * σH2_LW  +  nHD * σHD_LW 
                       + nCI * σCI_LW_E  +  nOHx * σOH_LW  +  nCHx * σCH_LW
                       + nCO * σCO_LW    +  nSI  * σSI_LW_E )  +  α_F_Thomson

            a_rad_r += α_F_LW * RT.H_r[iLW_RT,i,j]
            a_rad_Z += α_F_LW * RT.H_Z[iLW_RT,i,j]

            # (5) EUV1:
            α_F_EUV1 = ( nHI * StellAv.σHI_EUV1_E  +  nH2  * σH2_EUV1_E    +   nDI * StellAv.σHI_EUV1_E
                       + nCI * σCI_EUV1_E          +  nCII * σCII_EUV1_E   +   nOI * σOI_EUV1_E
                       + nSI * σSI_EUV1_E          +  nSII * σSII_EUV1_E )  +   α_F_Thomson

            a_rad_r += α_F_EUV1 * RT.H_r[iEUV1_RT,i,j]
            a_rad_Z += α_F_EUV1 * RT.H_Z[iEUV1_RT,i,j]

            # (6) EUV2:
            α_F_EUV2 = ( nHI * StellAv.σHI_EUV2_E  +  nH2 * σH2_EUV2_E    +  nHeI * StellAv.σHeI_EUV2_E   
                       + nDI * StellAv.σHI_EUV2_E  +  nCI * σCI_EUV2_E    +  nCII * σCII_EUV2_E 
                       + nOI * σOI_EUV2_E          +  nOII * σOII_EUV2_E
                       + nSI * σSI_EUV2_E          +  nSII * σSII_EUV2_E  +  nSIII * σSIII_EUV2_E )  +  α_F_Thomson

            a_rad_r += α_F_EUV2 * RT.H_r[iEUV2_RT,i,j]
            a_rad_Z += α_F_EUV2 * RT.H_Z[iEUV2_RT,i,j]

            # (7) EUV3:
            α_F_EUV3 = ( nHI   * StellAv.σHI_EUV3_E  +  nH2  * σH2_EUV3_E     +  nHeI  * StellAv.σHeI_EUV3_E 
                       + nHeII * StellAv.σHeII_EUV3_E
                       + nDI   * StellAv.σHI_EUV3_E  +  nCI  * σCI_EUV3_E     +  nCII  * σCII_EUV3_E
                       + nOI   * σOI_EUV3_E          +  nOII * σOII_EUV3_E    +  nOIII * σOIII_EUV3_E
                       + nSI   * σSI_EUV3_E          +  nSII * σSII_EUV3_E    +  nSIII * σSIII_EUV3_E
                       )  +  α_F_Thomson

            a_rad_r += α_F_EUV3 * RT.H_r[iEUV3_RT,i,j]
            a_rad_Z += α_F_EUV3 * RT.H_Z[iEUV3_RT,i,j]

            # (8) Lyα:
            if !isnothing(LyαState)
                # Lyα H I opacity at line center:
                αLyα      = LyαState.α_Lyα[i,j]

                # Lyα opacity from H₂ Raman scattering + electron scattering 
                # + absorption by OHₓ and CHₓ:
                αLyα_rest = ( nH2 * σRaman  +  nOHx * σOH_Lyα  +  nCHx * σCH_Lyα  
                            + α_F_Thomson )

                for k in LyαState.dN_ν 
                    # Lyα gas flux opacity:
                    αF_Lyα   = αLyα * LyαState.Hvoigt[k,i,j]  +  αLyα_rest

                    # Contribution to the acceleration from bin k:
                    a_rad_r += LyαState.Δν[k] * αF_Lyα * LyαState.H_r_Lyα[k,i,j]
                    a_rad_Z += LyαState.Δν[k] * αF_Lyα * LyαState.H_Z_Lyα[k,i,j]
                end
            end

            # Finally, multiply by 4π / (c ρ):
            a_rad_r *= Normalization
            a_rad_Z *= Normalization

            # Compute the local acceleration time-scale:
            a_rad    = hypot( a_rad_r, a_rad_Z )
            Δx       = min( Δr, ΔZ )
            Δt       = sqrt( 2 * Δx / a_rad )
            
            if Δt < local_min_Δt
                # Update the smallest time-scale found:
                local_min_Δt = Δt 
            end

        end

        min_Δt = min( min_Δt, local_min_Δt )
    end

    return C_RP * min_Δt
end


""" Get the radiation pressure acceleration time-step limiter for dust, considering 
    only the acceleration of dust by absorbed/scatter photons. 
    
    NOTE: This does not take into account coupling to gas. """
function GetRadPressureTimeStep_DustAcceleration( r::AbstractVector{Float64},   # Interface X-values
                                                  Z::AbstractVector{Float64},   # Interface Y-values
                                                 ρd::AbstractArray{Float64, 3}, # Dust densities 
                                                 RT::StellarRTArrays,
                                           LyαState::Union{LyαArrays, Nothing},
                                               Dust::DustState
                                                 )

    N_r  = length( r ) - 1
    N_Z  = length( Z ) - 1
    dN_r = 1:N_r 
    dN_Z = 1:N_Z

    min_Δt   = Inf64  # Initialize the minimum local time-scale
    @inbounds for j in dN_Z
        ΔZ           = Z[j+1] - Z[j] # Vertical cell width
        local_min_Δt = min_Δt        # Local variable for tracking smallest time-step 

        @simd for i in dN_r 

            # Compute the horizontal cell width (same for all j-indices):
            Δr       = r[i+1] - r[i]

            # Get radiation pressure acceleration on dust, a_rad = (4π / c ρd) ∫dν α_F(ν) H(ν):
            a_rad_r_Sil  = 0.0  # Initialize r-direction acceleration for silicate grains 
            a_rad_Z_Sil  = 0.0  # Initialize Z-direction acceleration for silicate grains 

            a_rad_r_C    = 0.0  # Initialize r-direction acceleration for graphite grains 
            a_rad_Z_C    = 0.0  # Initialize Z-direction acceleration for graphite grains 

            a_rad_r_PAH  = 0.0  # Initialize r-direction acceleration for PAH grains 
            a_rad_Z_PAH  = 0.0  # Initialize Z-direction acceleration for PAH grains 

            # (1) IR:
            κRoss_Sil    = κR(Dust, Dust.Trad[i,j], Sil)
            a_rad_r_Sil += Four_π_over_c * κRoss_Sil * RT.H_r[iIR_RT,i,j]
            a_rad_Z_Sil += Four_π_over_c * κRoss_Sil * RT.H_Z[iIR_RT,i,j]

            κRoss_C      = κR(Dust, Dust.Trad[i,j], Gra)
            a_rad_r_C   += Four_π_over_c * κRoss_C * RT.H_r[iIR_RT,i,j]
            a_rad_Z_C   += Four_π_over_c * κRoss_C * RT.H_Z[iIR_RT,i,j]

            κRoss_PAH    = κR(Dust, Dust.Trad[i,j], PAH)
            a_rad_r_PAH += Four_π_over_c * κRoss_PAH * RT.H_r[iIR_RT,i,j]
            a_rad_Z_PAH += Four_π_over_c * κRoss_PAH * RT.H_Z[iIR_RT,i,j]

            # (2) optical:
            a_rad_r_Sil += Four_π_over_c * Dust.κd_F_opt_Sil * RT.H_r[iopt_RT,i,j]
            a_rad_Z_Sil += Four_π_over_c * Dust.κd_F_opt_Sil * RT.H_Z[iopt_RT,i,j]

            a_rad_r_C   += Four_π_over_c * Dust.κd_F_opt_C   * RT.H_r[iopt_RT,i,j]
            a_rad_Z_C   += Four_π_over_c * Dust.κd_F_opt_C   * RT.H_Z[iopt_RT,i,j]

            a_rad_r_PAH += Four_π_over_c * Dust.κd_F_opt_PAH * RT.H_r[iopt_RT,i,j]
            a_rad_Z_PAH += Four_π_over_c * Dust.κd_F_opt_PAH * RT.H_Z[iopt_RT,i,j]

            # (3) FUV:
            a_rad_r_Sil += Four_π_over_c * Dust.κd_F_FUV_Sil * RT.H_r[iFUV_RT,i,j]
            a_rad_Z_Sil += Four_π_over_c * Dust.κd_F_FUV_Sil * RT.H_Z[iFUV_RT,i,j]

            a_rad_r_C   += Four_π_over_c * Dust.κd_F_FUV_C   * RT.H_r[iFUV_RT,i,j]
            a_rad_Z_C   += Four_π_over_c * Dust.κd_F_FUV_C   * RT.H_Z[iFUV_RT,i,j]

            a_rad_r_PAH += Four_π_over_c * Dust.κd_F_FUV_PAH * RT.H_r[iFUV_RT,i,j]
            a_rad_Z_PAH += Four_π_over_c * Dust.κd_F_FUV_PAH * RT.H_Z[iFUV_RT,i,j]

            # (4) LW:
            a_rad_r_Sil += Four_π_over_c * Dust.κd_F_LW_Sil * RT.H_r[iLW_RT,i,j]
            a_rad_Z_Sil += Four_π_over_c * Dust.κd_F_LW_Sil * RT.H_Z[iLW_RT,i,j]

            a_rad_r_C   += Four_π_over_c * Dust.κd_F_LW_C   * RT.H_r[iLW_RT,i,j]
            a_rad_Z_C   += Four_π_over_c * Dust.κd_F_LW_C   * RT.H_Z[iLW_RT,i,j]

            a_rad_r_PAH += Four_π_over_c * Dust.κd_F_LW_PAH * RT.H_r[iLW_RT,i,j]
            a_rad_Z_PAH += Four_π_over_c * Dust.κd_F_LW_PAH * RT.H_Z[iLW_RT,i,j]

            # (5) EUV1:
            a_rad_r_Sil += Four_π_over_c * Dust.κd_F_EUV1_Sil * RT.H_r[iEUV1_RT,i,j]
            a_rad_Z_Sil += Four_π_over_c * Dust.κd_F_EUV1_Sil * RT.H_Z[iEUV1_RT,i,j]

            a_rad_r_C   += Four_π_over_c * Dust.κd_F_EUV1_C   * RT.H_r[iEUV1_RT,i,j]
            a_rad_Z_C   += Four_π_over_c * Dust.κd_F_EUV1_C   * RT.H_Z[iEUV1_RT,i,j]

            a_rad_r_PAH += Four_π_over_c * Dust.κd_F_EUV1_PAH * RT.H_r[iEUV1_RT,i,j]
            a_rad_Z_PAH += Four_π_over_c * Dust.κd_F_EUV1_PAH * RT.H_Z[iEUV1_RT,i,j]

            # (6) EUV2:
            a_rad_r_Sil += Four_π_over_c * Dust.κd_F_EUV2_Sil * RT.H_r[iEUV2_RT,i,j]
            a_rad_Z_Sil += Four_π_over_c * Dust.κd_F_EUV2_Sil * RT.H_Z[iEUV2_RT,i,j]

            a_rad_r_C   += Four_π_over_c * Dust.κd_F_EUV2_C   * RT.H_r[iEUV2_RT,i,j]
            a_rad_Z_C   += Four_π_over_c * Dust.κd_F_EUV2_C   * RT.H_Z[iEUV2_RT,i,j]

            a_rad_r_PAH += Four_π_over_c * Dust.κd_F_EUV2_PAH * RT.H_r[iEUV2_RT,i,j]
            a_rad_Z_PAH += Four_π_over_c * Dust.κd_F_EUV2_PAH * RT.H_Z[iEUV2_RT,i,j]

            # (7) EUV3:
            a_rad_r_Sil += Four_π_over_c * Dust.κd_F_EUV3_Sil * RT.H_r[iEUV3_RT,i,j]
            a_rad_Z_Sil += Four_π_over_c * Dust.κd_F_EUV3_Sil * RT.H_Z[iEUV3_RT,i,j]

            a_rad_r_C   += Four_π_over_c * Dust.κd_F_EUV3_C   * RT.H_r[iEUV3_RT,i,j]
            a_rad_Z_C   += Four_π_over_c * Dust.κd_F_EUV3_C   * RT.H_Z[iEUV3_RT,i,j]

            a_rad_r_PAH += Four_π_over_c * Dust.κd_F_EUV3_PAH * RT.H_r[iEUV3_RT,i,j]
            a_rad_Z_PAH += Four_π_over_c * Dust.κd_F_EUV3_PAH * RT.H_Z[iEUV3_RT,i,j]

            # (8) Lyα:
            if !isnothing(LyαState)
                # Lyα dust flux opacities:
                κd_FLyα_Sil  = Dust.κd_F_Lyα_Sil
                κd_FLyα_C    = Dust.κd_F_Lyα_C
                κd_FLyα_PAH  = Dust.κd_F_Lyα_PAH

                for k in LyαState.dN_ν 
                    # Contribution to the acceleration from bin k:
                    a_rad_r_Sil += Four_π_over_c * LyαState.Δν[k] * κd_FLyα_Sil * LyαState.H_r_Lyα[k,i,j]
                    a_rad_Z_Sil += Four_π_over_c * LyαState.Δν[k] * κd_FLyα_Sil * LyαState.H_Z_Lyα[k,i,j]

                    a_rad_r_C   += Four_π_over_c * LyαState.Δν[k] * κd_FLyα_C   * LyαState.H_r_Lyα[k,i,j]
                    a_rad_Z_C   += Four_π_over_c * LyαState.Δν[k] * κd_FLyα_C   * LyαState.H_Z_Lyα[k,i,j]

                    a_rad_r_PAH += Four_π_over_c * LyαState.Δν[k] * κd_FLyα_PAH * LyαState.H_r_Lyα[k,i,j]
                    a_rad_Z_PAH += Four_π_over_c * LyαState.Δν[k] * κd_FLyα_PAH * LyαState.H_Z_Lyα[k,i,j]
                end
            end

            # Compute the local acceleration time-scale:
            a_rad_Sil = ( ρd[1,i,j] > ρd_IGNORE ) * hypot( a_rad_r_Sil, a_rad_Z_Sil )
            a_rad_C   = ( ρd[2,i,j] > ρd_IGNORE ) * hypot( a_rad_r_C,   a_rad_Z_C )
            a_rad_PAH = ( ρd[3,i,j] > ρd_IGNORE ) * hypot( a_rad_r_PAH, a_rad_Z_PAH )
            Δx        = min( Δr, ΔZ )
            Δt        = sqrt( 2 * Δx / max( a_rad_Sil, a_rad_C, a_rad_PAH ) )
            
            if Δt < local_min_Δt
                # Update the smallest time-scale found:
                local_min_Δt = Δt 
            end

        end

        min_Δt = min( min_Δt, local_min_Δt )
    end

    return C_RP_Dust * min_Δt
end


""" Get the radiation pressure acceleration time-step limiter for gas & dust, considering 
    the acceleration of gas and dust by absorbed/scatter photons for both species. 
    
    NOTE: This estimate basically assumes perfect dust-gas coupling. """
function GetRadPressureTimeStep_GasDustAcceleration( r::AbstractVector{Float64}, # Interface X-values
                                                     Z::AbstractVector{Float64}, # Interface Y-values
                                                     ρ::AbstractMatrix{Float64},
                                                    ρd::AbstractArray{Float64, 3},
                                                    RT::StellarRTArrays,
                                              LyαState::Union{LyαArrays, Nothing},
                                             Chemistry::ChemistryState,
                                                  Dust::DustState,
                                               StellAv::StellarBandAverages = DefaultStellarBandAverages
                                                    )

    N_r  = length( r ) - 1
    N_Z  = length( Z ) - 1
    dN_r = 1:N_r 
    dN_Z = 1:N_Z

    min_Δt   = Inf64  # Initialize the minimum local time-scale
    @inbounds for j in dN_Z
        ΔZ           = Z[j+1] - Z[j] # Vertical cell width
        local_min_Δt = min_Δt        # Local variable for tracking smallest time-step 

        @simd for i in dN_r 

            # Compute the horizontal cell width (same for all j-indices):
            Δr       = r[i+1] - r[i]

            # Get all needed gas number densities:
            nHI, nHII, nH2, nHeI, nHeII, nHeIII, nDI, nDII, nHD, 
            nCI, nCII, nCIII, nCO, nCHx, nOI, nOII, nOIII, nOIV, nOHx, 
            nSI, nSII, nSIII, nSIV, ne, n, nH, nHe, nD, 
            nC, nO, nS = GetAllNumberDensities( Chemistry.xH2[i,j], Chemistry.xHII[i,j], 
                                                Chemistry.xHeII[i,j], Chemistry.xHeIII[i,j], 
                                                Chemistry.xHD[i,j], Chemistry.xDII[i,j],
                                                Chemistry.xCI[i,j], Chemistry.xCII[i,j], Chemistry.xCIII[i,j], 
                                                Chemistry.xCO[i,j],
                                                Chemistry.xOI[i,j],  Chemistry.xOII[i,j], Chemistry.xOIII[i,j],
                                                Chemistry.xOIV[i,j], Chemistry.xSII[i,j], Chemistry.xSIII[i,j],
                                                Chemistry.xSIV[i,j],
                                                ρ[i,j], 
                                                Chemistry.ZC[i,j], Chemistry.ZSil[i,j], Chemistry.Zgasrest[i,j],
                                                Chemistry.HydrogenAndHelium )

            # Get radiation pressure acceleration on gas, a_rad = (4π / c ρ) ∫dν α_F(ν) H(ν):
            a_rad_r  = 0.0  # Initialize r-direction acceleration 
            a_rad_Z  = 0.0  # Initialize Z-direction acceleration

            # Normalization constant 4π / c ρtot to be multiplied by at the end:
            Normalization = Four_π_over_c / ( ρ[i,j] +  ρd[1,i,j] + ρd[2,i,j] + ρd[3,i,j] )

            # Flux opacity from electron (Thomson) scattering:
            α_F_Thomson   = ne * σT

            # Ignore contribution from dust if it is close to the dust floor:
            ρd_Sil_ij = ( ρd[1,i,j] > ρd_IGNORE ) * ρd[1,i,j]
            ρd_C_ij   = ( ρd[2,i,j] > ρd_IGNORE ) * ρd[2,i,j]
            ρd_PAH_ij = ( ρd[3,i,j] > ρd_IGNORE ) * ρd[3,i,j]

            # (1) IR: 
            α_F_IR = ( κR(Dust, Dust.Trad[i,j], Sil) * ρd_Sil_ij
                     + κR(Dust, Dust.Trad[i,j],   Gra) * ρd_C_ij
                     + κR(Dust, Dust.Trad[i,j], PAH) * ρd_PAH_ij
                     + α_F_Thomson )

            a_rad_r += α_F_IR * RT.H_r[iIR_RT,i,j]
            a_rad_Z += α_F_IR * RT.H_Z[iIR_RT,i,j]

            # (2) optical:
            α_F_opt  = ( nCHx * σCH_opt  +  α_F_Thomson
                       + Dust.κd_F_opt_Sil * ρd_Sil_ij
                       + Dust.κd_F_opt_C   * ρd_C_ij
                       + Dust.κd_F_opt_PAH * ρd_PAH_ij )

            a_rad_r += α_F_opt * RT.H_r[iopt_RT,i,j]
            a_rad_Z += α_F_opt * RT.H_Z[iopt_RT,i,j]

            # (3) FUV:
            α_F_FUV  = ( nOHx * σOH_FUV  +  nCHx * σCH_FUV  +  nSI * σSI_FUV_E  +  α_F_Thomson
                       + Dust.κd_F_FUV_Sil * ρd_Sil_ij
                       + Dust.κd_F_FUV_C   * ρd_C_ij
                       + Dust.κd_F_FUV_PAH * ρd_PAH_ij )

            a_rad_r += α_F_FUV * RT.H_r[iFUV_RT,i,j]
            a_rad_Z += α_F_FUV * RT.H_Z[iFUV_RT,i,j]

            # (4) LW:
            α_F_LW   = ( nHI * σHI_LW    +  nH2  * σH2_LW   +  nHD * σHD_LW 
                       + nCI * σCI_LW_E  +  nOHx * σOH_LW   +  nCHx * σCH_LW
                       + nCO * σCO_LW    +  nSI  * σSI_LW_E +  α_F_Thomson
                       + Dust.κd_F_LW_Sil * ρd_Sil_ij
                       + Dust.κd_F_LW_C   * ρd_C_ij
                       + Dust.κd_F_LW_PAH * ρd_PAH_ij )  

            a_rad_r += α_F_LW * RT.H_r[iLW_RT,i,j]
            a_rad_Z += α_F_LW * RT.H_Z[iLW_RT,i,j]

            # (5) EUV1:
            α_F_EUV1 = ( ( nHI * StellAv.σHI_EUV1_E  +  nH2  * σH2_EUV1_E    +   nDI * StellAv.σHI_EUV1_E
                         + nCI * σCI_EUV1_E          +  nCII * σCII_EUV1_E   +   nOI * σOI_EUV1_E
                         + nSI * σSI_EUV1_E          +  nSII * σSII_EUV1_E )  
                         + α_F_Thomson 
                         + Dust.κd_F_EUV1_Sil * ρd_Sil_ij
                         + Dust.κd_F_EUV1_C   * ρd_C_ij
                         + Dust.κd_F_EUV1_PAH * ρd_PAH_ij )

            a_rad_r += α_F_EUV1 * RT.H_r[iEUV1_RT,i,j]
            a_rad_Z += α_F_EUV1 * RT.H_Z[iEUV1_RT,i,j]

            # (6) EUV2:
            α_F_EUV2 = ( ( nHI * StellAv.σHI_EUV2_E  +  nH2 * σH2_EUV2_E    +  nHeI * StellAv.σHeI_EUV2_E   
                         + nDI * StellAv.σHI_EUV2_E  +  nCI * σCI_EUV2_E    +  nCII * σCII_EUV2_E 
                         + nOI * σOI_EUV2_E          +  nOII * σOII_EUV2_E
                         + nSI * σSI_EUV2_E          +  nSII * σSII_EUV2_E  +  nSIII * σSIII_EUV2_E )  
                         + α_F_Thomson 
                         + Dust.κd_F_EUV2_Sil * ρd_Sil_ij
                         + Dust.κd_F_EUV2_C   * ρd_C_ij 
                         + Dust.κd_F_EUV2_PAH * ρd_PAH_ij )

            a_rad_r += α_F_EUV2 * RT.H_r[iEUV2_RT,i,j]
            a_rad_Z += α_F_EUV2 * RT.H_Z[iEUV2_RT,i,j]

            # (7) EUV3:
            α_F_EUV3 = ( ( nHI   * StellAv.σHI_EUV3_E  +  nH2  * σH2_EUV3_E     +  nHeI  * StellAv.σHeI_EUV3_E 
                         + nHeII * StellAv.σHeII_EUV3_E
                         + nDI   * StellAv.σHI_EUV3_E  +  nCI  * σCI_EUV3_E     +  nCII  * σCII_EUV3_E
                         + nOI   * σOI_EUV3_E          +  nOII * σOII_EUV3_E    +  nOIII * σOIII_EUV3_E
                         + nSI   * σSI_EUV3_E          +  nSII * σSII_EUV3_E    +  nSIII * σSIII_EUV3_E )
                         + α_F_Thomson 
                         + Dust.κd_F_EUV3_Sil * ρd_Sil_ij
                         + Dust.κd_F_EUV3_C   * ρd_C_ij
                         + Dust.κd_F_EUV3_PAH * ρd_PAH_ij )

            a_rad_r += α_F_EUV3 * RT.H_r[iEUV3_RT,i,j]
            a_rad_Z += α_F_EUV3 * RT.H_Z[iEUV3_RT,i,j]

            # (8) Lyα:
            if !isnothing(LyαState)
                for k in LyαState.dN_ν 
                    # Lyα flux opacity (takes into account both H I and dust):
                    αFLyα    = LyαState.α_F_Lyα[k,i,j]

                    # Contribution to the acceleration from bin k:
                    a_rad_r += LyαState.Δν[k] * αFLyα * LyαState.H_r_Lyα[k,i,j]
                    a_rad_Z += LyαState.Δν[k] * αFLyα * LyαState.H_Z_Lyα[k,i,j]
                end
            end

            # Finally, multiply by 4π / (c ρtot):
            a_rad_r *= Normalization
            a_rad_Z *= Normalization

            # Compute the local acceleration time-scale:
            a_rad    = hypot( a_rad_r, a_rad_Z )
            Δx       = min( Δr, ΔZ )
            Δt       = sqrt( 2 * Δx / a_rad )
            
            if Δt < local_min_Δt
                # Update the smallest time-scale found:
                local_min_Δt = Δt 
            end

        end

        min_Δt = min( min_Δt, local_min_Δt )
    end

    return C_RP * min_Δt
end


""" This function finds the required minimum RSL for 
    RHD simulations with stellar and IR radiative transfer. """
function FindRSL_StellarRT( c̃::Float64,  # Old RSL
                           RT::StellarRTArrays,
                        Hydro::HydrodynamicsState,
                         Dust::DustState,
                         Geom::Union{GeometryCylindrical, GeometryCartesian} 
                           )

    # Gather interface arrays:
    if Geom isa GeometryCylindrical
        r = Geom.r;  Z = Geom.Z 
    elseif Geom isa GeometryCartesian
        r = Geom.X;  Z = Geom.Y 
    end

    # Gather hydro arrays:
    u_r  = Hydro.u_X 
    u_Z  = Hydro.u_Y 
    ud_r = Hydro.ud_X 
    ud_Z = Hydro.ud_Y
    P    = Hydro.P 
    ρ    = Hydro.ρ 
    ρd   = Hydro.ρd   # All dust bins 
    γad  = Hydro.γad 

    # Grid size & intervals:
    N_r, N_Z = size( ρ )
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z

    # Find new RSL:
    c̃_new    = 0.0   # Initialize value for the new RSL 
    for j in dN_Z 
        local_max_c̃  =  c̃_new   # Local variable for tracking largest c̃

        # Cell width in Z-direction:
        ΔZ = Z[j+1] - Z[j]
        @inbounds for i in dN_r 
            # Cell width in r-direction:
            Δr   = r[i+1] - r[i]

            # Gas sound speed:
            c_s  = @fastmath sqrt( γad[i,j] * P[i,j] / ρ[i,j] )

            # Get dust bin densities:
            ρd_Sil = ρd[1,i,j]
            ρd_C   = ρd[2,i,j]
            ρd_PAH = ρd[3,i,j]

            # Cell IR optical depth:
            ΔX   = max( Δr, ΔZ )
            τ_IR = @fastmath ( ρd_Sil * κR(Dust, Dust.Trad[i,j], Sil)
                             + ρd_C   * κR(Dust, Dust.Trad[i,j], Gra)
                             + ρd_PAH * κR(Dust, Dust.Trad[i,j], PAH) ) * ΔX

            # Compute minimum RSL for accuracy in this cell to be ~ ϵ_RSL × (t_trap/t_light) × gas/dust velocity:
            u           = hypot( u_r[i,j], u_Z[i,j] )
            ud_Sil      = hypot( ud_r[1,i,j], ud_Z[1,i,j] )
            ud_C        = hypot( ud_r[2,i,j], ud_Z[2,i,j] )
            ud_PAH      = hypot( ud_r[3,i,j], ud_Z[3,i,j] )
            ud_max      = max( ud_Sil, ud_C, ud_PAH )

            c̃_min_cell  = ϵ_RSL * max( 1, τ_IR ) * max( u, ud_max, c_s )

            # Update the largest RSL we've found:
            local_max_c̃ = max( local_max_c̃, c̃_min_cell )
        end

        c̃_new = max( c̃_new, local_max_c̃ )
    end

    # Set the new RSL:
    c̃_new = min( max( c̃_new, RT.c̃_min ), 1e-3 * c )

    # To avoid rapid changes in c̃ if the newly calculated RSL is small, 
    # we further limit the maximum reduction to 25 % from the last time-step:
    if c̃_new < c̃
        c̃_new = max( c̃_new, 0.75 * c̃ )
    end

    # We also limit any increase of c̃ to 4 × the old value, so as to not 
    # suddenly get stuck in tiny time-steps:
    if c̃_new > c̃
        c̃_new = min( c̃_new, 4.0 * c̃ )
    end

    return c̃_new
end


""" This function estimates the Lyα RSL needed for sufficiently accurate RHD 
    simulations. """
function FindRSL_LyαRT( c̃_Lyα::Float64, # Old RSL
                        c̃_min::Float64, # Minimum RSL
                          Lyα::Union{LyαArrays, Nothing},
                        Hydro::HydrodynamicsState,
                         Chem::ChemistryState,
                         Dust::DustState,
                         Geom::Union{GeometryCylindrical, GeometryCartesian}
                       )

    if isnothing(Lyα)
        # If we don't simulate Lyα RT, just return 0.0, since 
        # c̃ will be given by the larger of the required c̃ for stellar RT 
        # and Lyα RT.
        return 0.0 
    end

    # Gather interface arrays:
    if Geom isa GeometryCylindrical
        r = Geom.r;  Z = Geom.Z 
    elseif Geom isa GeometryCartesian
        r = Geom.X;  Z = Geom.Y 
    end

    # Gather hydro arrays:
    u_r  = Hydro.u_X 
    u_Z  = Hydro.u_Y 
    P    = Hydro.P 
    ρ    = Hydro.ρ 
    ρd   = Hydro.ρd
    γad  = Hydro.γad 
    T    = Hydro.T

    # Gather chemistry arrays:
    xHII = Chem.xHII 
    xH2  = Chem.xH2 

    # Grid size & intervals:
    N_r, N_Z = size( ρ )
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z

    # Find new RSL:
    c̃_Lyα_new = 0.0    # Initialize value for the new RSL 

    for j in dN_Z 
        local_max_c̃  =  c̃_Lyα_new    # Local variable for tracking largest c̃ 

        # Cell width in Z-direction:
        ΔZ = Z[j+1] - Z[j]
        @inbounds for i in dN_r 
            # Cell width in r-direction:
            Δr    = r[i+1] - r[i]

            # Gas sound speed:
            c_s   = @fastmath sqrt( γad[i,j] * P[i,j] / ρ[i,j] )

            # H I number density:
            xHI   = max( 1 - xHII[i,j] - 2 * xH2[i,j], 0.0 )
            nHI   = @fastmath xHI * X_over_mH * ρ[i,j]

            # Lyα opacity at line center for the cell:
            α     = @fastmath nHI * Compute_σ0( T[i,j] )

            # Lyα cell optical depth estimate:
            ΔX    = max( Δr, ΔZ )
            τcell = @fastmath α * ΔX

            # Continuum absorption cell optical depth:
            τcabs = @fastmath ( Dust.κd_abs_Lyα_Sil * ρd[1,i,j]
                              + Dust.κd_abs_Lyα_C   * ρd[2,i,j]
                              + Dust.κd_abs_Lyα_PAH * ρd[3,i,j] ) * ΔX

            # Get aτ and (aτ)^(1/3):
            aτ    = Compute_a( T[i,j] ) * τcell
            aτ_13 = @fastmath aτ^OneThird

            # Get reduction factor from Lyα absorption/destruction:
            fabs  = @fastmath 1 / ( 1  +  aτ_13 * τcabs  +  0.6 * Lyα.pd[i,j] * τcell )

            # Compute minimum RSL for accuracy in this cell to be ~ ϵ_RSL × (t_trap/t_light) × gas velocity:
            u           = hypot( u_r[i,j], u_Z[i,j] )
            c̃_min_cell  = ϵ_RSL * max( 1, aτ_13 * fabs ) * max( u, c_s )

            # Update the largest RSL we've found:
            local_max_c̃ = max( local_max_c̃, c̃_min_cell )
        end

        c̃_Lyα_new  = max( c̃_Lyα_new, local_max_c̃ )
    end

    # Clamp the new RSL:
    c̃_Lyα_new = min( max( c̃_Lyα_new, c̃_min ), c )

    # To avoid rapid changes in c̃ if the newly calculated RSL is small, 
    # we further limit the maximum reduction to 25 % from the last time-step:
    if c̃_Lyα_new < c̃_Lyα
        c̃_Lyα_new = max( c̃_Lyα_new, 0.75 * c̃_Lyα )
    end

    # We also limit any increase of c̃ to 4 × the old value, so as to not 
    # suddenly get stuck in tiny time-steps:
    if c̃_Lyα_new > c̃_Lyα
       c̃_Lyα_new = min( c̃_Lyα_new, 4.0 * c̃_Lyα )
    end

    return c̃_Lyα_new 
end


""" This function finds a suitable reduced speed of light c̃_new 
    for RHD simulations, with or without Lyα RT. """
@inline function FindRSL( c̃::Float64,  # Old RSL 
                         RT::StellarRTArrays,
                        Lyα::Union{LyαArrays, Nothing},
                      Hydro::HydrodynamicsState,
                       Chem::ChemistryState,
                       Dust::DustState,
                       Geom::Union{GeometryCylindrical, GeometryCartesian}
                         )

    c̃_new = max( FindRSL_LyαRT(c̃, RT.c̃_min, Lyα, Hydro, Chem, Dust, Geom),
                 FindRSL_StellarRT( c̃, RT, Hydro, Dust, Geom ) )

    return c̃_new
end


""" The total gas heating rate (erg cm⁻³ s⁻ꜝ) from 
    absorption of ionizing and other photons,
    from photoelectric heating (from all dust types), 
    and from H₂ chemical heating/cooling. """
function HeatingRate_total( J_opt::Float64, 
                            J_FUV::Float64, 
                             J_LW::Float64,
                           J_EUV1::Float64,
                           J_EUV2::Float64,
                           J_EUV3::Float64,
                            T_old::Float64,
                           Td_old::NTuple{3, Float64}, # Dust temperatures for each bin (Sil, C, PAH)
                          Zgr_old::NTuple{3, Float64}, # Grain charge for each bin (Sil, C, PAH)
                           ne_old::Float64,
                           nH_old::Float64, # nHI_old + nHII_old + 2 × nH2_old (but pre-compute for efficiency)
                          nHI_old::Float64,
                          nH2_old::Float64,
                         nHII_old::Float64,
                         nHeI_old::Float64,
                        nHeII_old::Float64,
                          nCI_old::Float64, 
                         nCII_old::Float64, 
                          nOI_old::Float64, 
                         nOII_old::Float64, 
                        nOIII_old::Float64, 
                          nSI_old::Float64, 
                         nSII_old::Float64, 
                        nSIII_old::Float64, 
                    SelfShielding::Float64,
                               ρd::NTuple{3, Float64}, # Dust densities for each bin (Sil, C, PAH)
                              Γpd::Float64 = 0.0,      # H⁻ photodetachment rate 
                                z::Float64 = 0.0,      # Redshift 
                            e_Lyα::Float64 = 0.0,
                  StellarAverages::StellarBandAverages = DefaultStellarBandAverages
                             )

    # Extract dust densities, temperatures, charges:
    ρd_Sil, ρd_C, ρd_PAH                = ρd
    Td_Sil_old, Td_C_old, Td_PAH_old    = Td_old
    Zgr_Sil_old, Zgr_C_old, Zgr_PAH_old = Zgr_old

    # Total gas heating rate:
    HeatingRate_tot = (  
       HeatingRate_Compton(J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, ne_old, T_old, e_Lyα, StellarAverages )

    +  HeatingRate_FUV( J_FUV, nSI_old, StellarAverages ) 

    +  HeatingRate_LW( SelfShielding, J_LW, nH2_old, nHI_old, nCI_old, nSI_old, T_old, StellarAverages )

    +  HeatingRate_EUV1( J_EUV1, nHI_old, nH2_old, nCI_old, nCII_old, nOI_old, nSI_old, nSII_old, StellarAverages ) 

    +  HeatingRate_EUV2( J_EUV2, nHI_old, nH2_old, nHeI_old, nCI_old, nCII_old, nOI_old, nOII_old, 
                         nSI_old, nSII_old, nSIII_old, StellarAverages ) 

    +  HeatingRate_EUV3( J_EUV3, nHI_old, nH2_old, nHeI_old, nHeII_old, nCI_old, nCII_old, nOI_old, nOII_old, nOIII_old,
                         nSI_old, nSII_old, nSIII_old, StellarAverages )

    +  HeatingRate_PhotoElectric( J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, ρd_Sil, Zgr_Sil_old, Sil, e_Lyα )
    +  HeatingRate_PhotoElectric( J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, ρd_C,   Zgr_C_old,   Gra, e_Lyα )
    +  HeatingRate_PhotoElectric( J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, ρd_PAH, Zgr_PAH_old, PAH, e_Lyα )

    +  HeatingRate_H2chemical( T_old, Td_Sil_old, Td_C_old, Td_PAH_old,
                               nH_old, nHI_old, nH2_old, nHII_old, ne_old, 
                               ρd_Sil, ρd_C, ρd_PAH, Γpd, z )  )

    return HeatingRate_tot
end


"""  This function estimates the density gradient magnitude |∇ρ|,
     for use in the Sobolev-like estimate of column density path lengths. """ 
@inline function Estimate_∇ρ( i::Int, j::Int,
                          Hydro::HydrodynamicsState,
                       Geometry::Union{GeometryCylindrical, GeometryCartesian} )

    ρ        = Hydro.ρ
    N_r, N_Z = size( ρ )
    ∂ρ∂r     = 0.0
    ∂ρ∂Z     = 0.0

    if Geometry isa GeometryCylindrical
        r    = Geometry.r
        Z    = Geometry.Z
        rCoM = Geometry.rCoM
        ZCoM = Geometry.ZCoM
    elseif Geometry isa GeometryCartesian
        r    = Geometry.X
        Z    = Geometry.Y
        rCoM = Geometry.XCoM
        ZCoM = Geometry.YCoM
    else 
        error("No valid coordinate system.")
    end

    if i ≥ 2 && i ≤ (N_r-1)
        # For interior points:
        Δr_L = rCoM[i]   - rCoM[i-1]
        Δr_R = rCoM[i+1] - rCoM[i]

        ∂ρ∂r = MinMod( (ρ[i,j] - ρ[i-1,j]) / Δr_L, 
                       (ρ[i+1,j] - ρ[i,j]) / Δr_R )
    elseif i == 1
        # For the innermost cell we use reflective BCs,
        # and ghost cells:
        ρ_0    = ρ[1,j]
        rCoM_0 = r[1]  -  ( rCoM[1] - r[1] )
        Δr_L   = rCoM[1] - rCoM_0
        Δr_R   = rCoM[2] - rCoM[1]

        ∂ρ∂r   = MinMod( (ρ[1,j] - ρ_0) / Δr_L, 
                         (ρ[2,j] - ρ[1,j]) / Δr_R )
    elseif i == N_r 
        # For the outermost cell we use outflow BCs, 
        # and a ghost cell:
        ρ_Np1    = ρ[N_r,j]
        rCoM_Np1 = r[N_r+1]  +  ( r[N_r+1] - rCoM[N_r] )   # Approximate extrapolation 
        Δr_L     = rCoM[N_r] - rCoM[N_r-1]
        Δr_R     = rCoM_Np1  - rCoM[N_r]
        
        ∂ρ∂r     = MinMod( (ρ[N_r,j] - ρ[N_r-1,j]) / Δr_L, 
                           (ρ_Np1    - ρ[N_r,j])   / Δr_R )
    end

    if j ≥ 2 && j ≤ (N_Z-1)
        # For interior points:
        ΔZ_L = ZCoM[j]   - ZCoM[j-1]
        ΔZ_R = ZCoM[j+1] - ZCoM[j]

        ∂ρ∂Z = MinMod( (ρ[i,j] - ρ[i,j-1]) / ΔZ_L, 
                       (ρ[i,j+1] - ρ[i,j]) / ΔZ_R )
    elseif j == 1
        # For the innermost cell we use reflective BCs,
        # and ghost cells:
        ρ_0    = ρ[i,1]
        ZCoM_0 = Z[1]  -  ( ZCoM[1] - Z[1] )
        ΔZ_L   = ZCoM[1] - ZCoM_0
        ΔZ_R   = ZCoM[2] - ZCoM[1]

        ∂ρ∂Z   = MinMod( (ρ[i,1] - ρ_0)    / ΔZ_L, 
                         (ρ[i,2] - ρ[i,1]) / ΔZ_R )
    elseif j == N_Z
        # For the outermost cell we use outflow BCs, 
        # and a ghost cell:
        ρ_Np1    = ρ[i,N_Z]
        ZCoM_Np1 = Z[N_Z+1]  +  ( Z[N_Z+1] - ZCoM[N_Z] )   # Extrapolation 
        ΔZ_L     = ZCoM[N_Z] - ZCoM[N_Z-1]
        ΔZ_R     = ZCoM_Np1  - ZCoM[N_Z]

        ∂ρ∂Z     = MinMod( (ρ[i,N_Z] - ρ[i,N_Z-1]) / ΔZ_L, 
                           (ρ_Np1 - ρ[i,N_Z])      / ΔZ_R )
    end

    # Compute the magnitude of the gradient:
    ∇ρ = @fastmath sqrt( ∂ρ∂r * ∂ρ∂r  +  ∂ρ∂Z * ∂ρ∂Z )

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
    x    = @fastmath N_H2 / N_H2crit
    F    = @fastmath sqrt( 1 + x )
    b5   = @fastmath sqrt( 2 * kB * T * inv_mH2 ) * inv_kms 

    inv_stuff = @fastmath 1 / (1 + x / b5)
    S_H2      = @fastmath S_H2_1 * inv_stuff * inv_stuff  +  S_H2_2 * exp( - c_SH2 * F ) / F
    return S_H2
end


""" The HD self-shielding factor, same fit as for H₂. """
@inline function S_HD( N_HD::Float64, 
                          T::Float64
                       )
    T    = clamp( T, 1.0, 1e5 )
    x    = @fastmath N_HD / N_H2crit
    F    = @fastmath sqrt( 1 + x )
    b5   = @fastmath sqrt( 2 * kB * T * inv_mD ) * inv_kms   

    inv_stuff = @fastmath 1 / (1 + x / b5)
    S_HD      = @fastmath S_H2_1 * inv_stuff * inv_stuff  +  S_H2_2 * exp( - c_SH2 * F ) / F
    return S_HD
end


const N_H2_HDcrit = 2.34e19; const c_H2_HD = 5.2e-3; const p_H2_HD = 0.238 

""" The shielding of HD by H₂. See Wolcott-Green & Haiman (2011). """
@inline function S_H2_HD( N_H2::Float64 )
    x       = @fastmath N_H2 / N_H2_HDcrit
    S_H2_HD = @fastmath exp( - c_H2_HD * x ) / (1 + x)^p_H2_HD
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
                                        nHeIII::Float64,
                                     StellarAv::StellarBandAverages = DefaultStellarBandAverages )

    if ne == 0.0
        # In neutral gas, we have no recombinations, so exit early:
        return 0.0, 0.0, 0.0
    end

    # EUV1 emissivity (erg s⁻¹ cm⁻³ sr⁻¹) from recombinations to H I ground state:
    js_EUV1 = @fastmath krecHII_1(T)   * ne * nHII   * StellarAv.E_EUV1 * inv_Four_π

    # EUV2 emissivity (erg s⁻¹ cm⁻³ sr⁻¹) from recombinations to He I ground state:
    js_EUV2 = @fastmath krecHeII_1(T)  * ne * nHeII  * StellarAv.E_EUV2 * inv_Four_π

    # EUV3 emissivity (erg s⁻¹ cm⁻³ sr⁻¹) from recombinations to He II ground state:
    js_EUV3 = @fastmath krecHeIII_1(T) * ne * nHeIII * StellarAv.E_EUV3 * inv_Four_π

    return js_EUV1, js_EUV2, js_EUV3
end


""" This function restores J_Lyα and H_Lyα to their old values in 
    case candidate update values were rejected, or if a substep 
    was rejected for other reasons (e.g. cooling step rejected). """
@inline function RestoreLyαRejection( i::Int, j::Int, Lyα::Union{LyαArrays,Nothing} )

    if !isnothing(Lyα)
        @inbounds for k in Lyα.dN_ν 
            Lyα.J_Lyα[k,i,j]   = Lyα.J_Lyα_old[k,i,j]
            Lyα.H_r_Lyα[k,i,j] = Lyα.H_r_Lyα_old[k,i,j]
            Lyα.H_Z_Lyα[k,i,j] = Lyα.H_Z_Lyα_old[k,i,j]
        end
    end

    return nothing 
end


""" Just a convenient small function for if the absorption step is rejected,
    where we still have to return lots of arguments. """
@inline function RejectAbsorption( ΔRT::Float64 )
    REDO = true
    return REDO, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # REDO, 7 × J
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # 7 × H_r
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # 7 × H_Z
           0.0, 0.0, 0.0, 0.0,                        # Δρu_r/Z_rad, Δρu_r/Z_rad_IR
           0.0, 0.0,                                  # Δρu_r/Z_rad_gas
           0.0, 0.0,                                  # Δρu_r/Z_rad_Sil
           0.0, 0.0,                                  # Δρu_r/Z_rad_C
           0.0, 0.0,                                  # Δρu_r/Z_rad_PAH
           ΔRT, 0.0,                                  # ΔRT, js_IR
           0.0, 0.0, 0.0,                             # Td_Sil_new, Td_C_new, Td_PAH_new
           0.0,                                       # Trad_new,
           0.0, 0.0, 0.0                              # e_Lyα_new, Δρu_r/Z_rad_Lyα
end


""" A small helper that splits a band's total momentum injection Δρu_rad into
    per-species fractions, weighted by each species flux opacity.

    Returns (Δρu_gas, Δρu_Sil, Δρu_C, Δρu_PAH). """
@inline function SplitRadPressure( Δρu_r_rad::Float64, Δρu_Z_rad::Float64,
                                     α_gas_F::Float64,
                                     α_F_Sil::Float64, α_F_C::Float64, α_F_PAH::Float64 )

    # Total flux opacity:
    α_F_tot = @fastmath α_gas_F + α_F_Sil + α_F_C + α_F_PAH

    if α_F_tot > 0.0
        inv_αF      = @fastmath 1 / α_F_tot

        # Fraction of momentum absorbed by gas and dust bins:
        Momfrac_gas = @fastmath α_gas_F * inv_αF
        Momfrac_Sil = @fastmath α_F_Sil * inv_αF
        Momfrac_C   = @fastmath α_F_C   * inv_αF
        Momfrac_PAH = @fastmath α_F_PAH * inv_αF
    else
        Momfrac_gas = 1.0; Momfrac_Sil = 0.0; Momfrac_C = 0.0; Momfrac_PAH = 0.0
    end

    # Momentum to be injected into each component:
    Δρu_r       = @fastmath Momfrac_gas * Δρu_r_rad;        Δρu_Z = @fastmath Momfrac_gas * Δρu_Z_rad
    Δρdud_r_Sil = @fastmath Momfrac_Sil * Δρu_r_rad;  Δρdud_Z_Sil = @fastmath Momfrac_Sil * Δρu_Z_rad
    Δρdud_r_C   = @fastmath Momfrac_C   * Δρu_r_rad;  Δρdud_Z_C   = @fastmath Momfrac_C   * Δρu_Z_rad
    Δρdud_r_PAH = @fastmath Momfrac_PAH * Δρu_r_rad;  Δρdud_Z_PAH = @fastmath Momfrac_PAH * Δρu_Z_rad

    return Δρu_r,     Δρu_Z,
           Δρdud_r_Sil, Δρdud_Z_Sil,
           Δρdud_r_C,   Δρdud_Z_C,
           Δρdud_r_PAH, Δρdud_Z_PAH
end


const tol_Abs::Float64     = 0.1                      # Reject an absorption update if the relative change in J or H is > tol_Abs
const Td_floor::Float64    = 1.0                      # Dust temperature (Td) floor below which we don't care about changes in Td
const J_floor::Float64     = 1e-15                    # Minimum stellar or IR J below which we don't care about the tol_Abs rule 
const H_floor::Float64     = J_floor                  # Minimum stellar or IR H below which we don't care about the tol_Abs rule 
const e_Lyα_floor::Float64 = Four_π_over_c * J_floor  # Minimum Lyα energy density below which we don't care about the tol_Abs rule 

""" This function updates the intensities J and first moment H from gas + dust absorption/scattering,
    and compute the corresponding change in the gas momentum from radiation pressure. """
@inline function AbsorptionStep( Δt::Float64,                  # Sub-cycle time-step
                          i::Int,                      # Cell i-index 
                          j::Int,                      # Cell j-index 
                          c̃::Float64,                  # Reduced speed of light for stellar/IR RT 
                         nH::Float64,                  # Total hydrogen density (pre-computed outside, even though it's just a simple sum)
                        nHI::Float64,                  # H I number density 
                        nH2::Float64,                  # H₂ number density   
                       nHeI::Float64,                  # He I number density 
                      nHeII::Float64,                  # He II number density   
                        nDI::Float64,                  # D I number density 
                        nHD::Float64,                  # HD number density 
                        nCI::Float64,                  # C I number density 
                       nCII::Float64,                  # C II number density 
                       nCHx::Float64,                  # CHₓ number density 
                        nCO::Float64,                  # CO number density 
                        nOI::Float64,                  # O I number density 
                       nOII::Float64,                  # O II number density 
                      nOIII::Float64,                  # O III number density 
                       nOHx::Float64,                  # OHₓ number density 
                        nSI::Float64,                  # S I number density 
                       nSII::Float64,                  # S II number density 
                      nSIII::Float64,                  # S III number density 
                         ne::Float64,                  # Electron number density (for Thomson scattering)
                         ρd::NTuple{3,Float64},        # Dust densities (Sil, C, PAH)
                       Dust::DustState,                # Dust struct, for opacities 
                        Lyα::Union{LyαArrays,Nothing}, # Lyα state, if used 
                      T_old::Float64,                  # Old gas temperature 
                     Td_old::NTuple{3,Float64},        # Old dust temperatures (Sil, C, PAH)
                   Trad_old::Float64,                  # Old radiation temperature 
                       TCMB::Float64,                  # CMB temperature 

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
                    StellAv::StellarBandAverages = DefaultStellarBandAverages

                        )

    # Convenient constants:
    c̃Δt    = @fastmath c̃ * Δt 
    inv_c̃  = @fastmath 1 / c̃   

    # Unpack dust variables: 
    ρd_Sil, ρd_C, ρd_PAH             = ρd 
    Td_Sil_old, Td_C_old, Td_PAH_old = Td_old 

    # Ignore dust absorption/scattering near the floor:
    ρd_Sil          = (ρd_Sil > ρd_IGNORE) * ρd_Sil
    ρd_C            = (ρd_C   > ρd_IGNORE) * ρd_C
    ρd_PAH          = (ρd_PAH > ρd_IGNORE) * ρd_PAH    

    # Initialize diagnostic gas + dust momentum change from stellar radiation pressure:
    Δρu_r_rad       = 0.0
    Δρu_Z_rad       = 0.0

    # Initialize diagnostic gas + dust momentum change from IR radiation pressure:
    Δρu_r_rad_IR    = 0.0
    Δρu_Z_rad_IR    = 0.0

    # Initialize momentum to be deposited into gas:
    Δρu_r_rad_gas   = 0.0
    Δρu_Z_rad_gas   = 0.0

    # Initialize momentum to be deposited into dust bins:
    Δρu_r_rad_Sil   = 0.0;  Δρu_Z_rad_Sil  = 0.0
    Δρu_r_rad_C     = 0.0;  Δρu_Z_rad_C    = 0.0
    Δρu_r_rad_PAH   = 0.0;  Δρu_Z_rad_PAH  = 0.0

    # Thomson scattering opacity is used many times below, so pre-compute:
    α_F_Thomson     = @fastmath ne * σT

    #######################################################################
    #######                                                         #######
    #######                       O P T I C A L                     #######
    #######                                                         #######
    #######################################################################
        
    # Optical opacities:
    @fastmath α_gas_abs_opt  =   nCHx * σCH_opt                   # Optical gas absorption opacity 
    @fastmath α_gas_F_opt    =   α_gas_abs_opt  +  α_F_Thomson    # Optical flux opacity (adds Thomson scattering)  

    @fastmath α_F_opt_Sil    =   Dust.κd_F_opt_Sil * ρd_Sil       # Silicate dust flux opacity  (for H_opt)
    @fastmath α_F_opt_C      =   Dust.κd_F_opt_C   * ρd_C         # Graphite dust flux opacity  (for H_opt)
    @fastmath α_F_opt_PAH    =   Dust.κd_F_opt_PAH * ρd_PAH       # PAH dust flux opacity  (for H_opt)

    # Total absorption opacity (for J_opt):
    @fastmath α_abs_opt      =   ( α_gas_abs_opt  
                                 + Dust.κd_abs_opt_Sil * ρd_Sil  +  Dust.κd_abs_opt_C * ρd_C 
                                 + Dust.κd_abs_opt_PAH * ρd_PAH )   

    # Total flux opacity (for H_opt):
    @fastmath α_F_opt        =   α_gas_F_opt  +  ( α_F_opt_Sil + α_F_opt_C + α_F_opt_PAH )

    # Absorption time-scale:
    @fastmath Δt_opt         =   inv_c̃ / max( α_abs_opt, α_F_opt ) 

    # Absorption update for optical band:
    J_opt          = max( AbsorptionUpdateJorH( J_opt_old, Δt, dJ_optdt, α_abs_opt, c̃ ), Jeps )
    H_r_opt        = AbsorptionUpdateJorH( H_r_opt_old, Δt, dHr_optdt, α_F_opt, c̃ )
    H_Z_opt        = AbsorptionUpdateJorH( H_Z_opt_old, Δt, dHZ_optdt, α_F_opt, c̃ )

    # Compute relative changes in optical J, H_r, and H_Z:
    ΔJ_opt         = abs( J_opt   - J_opt_old )   / ( J_opt_old  +  J_floor )
    ΔH_r_opt       = abs( H_r_opt - H_r_opt_old ) / ( abs(H_r_opt_old)  +  H_floor )
    ΔH_Z_opt       = abs( H_Z_opt - H_Z_opt_old ) / ( abs(H_Z_opt_old)  +  H_floor )

    Δopt = max( ΔJ_opt, ΔH_r_opt, ΔH_Z_opt )
    if ( Δopt > tol_Abs  &&  Δt > tol_Abs * Δt_opt ) || !isfinite(Δopt)
        # Change in optical was too large. Reject step: 
        return RejectAbsorption( Δopt )
    end

    # Change in total gas + dust momentum:
    @fastmath Δρu_r_rad_opt  = Four_π_over_c * ( inv_c̃ * (H_r_opt_old - H_r_opt)  +  inv_c̃ * Δt * dHr_optdt )
    @fastmath Δρu_Z_rad_opt  = Four_π_over_c * ( inv_c̃ * (H_Z_opt_old - H_Z_opt)  +  inv_c̃ * Δt * dHZ_optdt )

    # Add total radiation pressure momentum contribution:
    Δρu_r_rad      += Δρu_r_rad_opt
    Δρu_Z_rad      += Δρu_Z_rad_opt

    # Add the fractional contributions to gas vs. dust:
    δr_gas, δZ_gas, δr_Sil, δZ_Sil, δr_C, δZ_C, δr_PAH, δZ_PAH =
    SplitRadPressure( Δρu_r_rad_opt, Δρu_Z_rad_opt,
                      α_gas_F_opt, α_F_opt_Sil, α_F_opt_C, α_F_opt_PAH )

    Δρu_r_rad_gas += δr_gas;  Δρu_Z_rad_gas += δZ_gas
    Δρu_r_rad_Sil += δr_Sil;  Δρu_Z_rad_Sil += δZ_Sil
    Δρu_r_rad_C   += δr_C;    Δρu_Z_rad_C   += δZ_C
    Δρu_r_rad_PAH += δr_PAH;  Δρu_Z_rad_PAH += δZ_PAH

    #######################################################################
    #######                                                         #######
    #######                       F A R - U V                       #######
    #######                                                         #######
    #######################################################################

    # FUV opacities:

    # FUV gas absorption opacity:
    @fastmath α_gas_abs_FUV  = ( nOHx * σOH_FUV  +  nCHx * σCH_FUV        
                               + nSI  * σSI_FUV_E )                

    # FUV flux opacity (adds Thomson scattering):
    @fastmath α_gas_F_FUV    =   α_gas_abs_FUV  +  α_F_Thomson

    @fastmath α_F_FUV_Sil    =   Dust.κd_F_FUV_Sil * ρd_Sil   # Silicate dust flux opacity (for H_FUV)
    @fastmath α_F_FUV_C      =   Dust.κd_F_FUV_C   * ρd_C     # Graphite dust flux opacity (for H_FUV)
    @fastmath α_F_FUV_PAH    =   Dust.κd_F_FUV_PAH * ρd_PAH   # PAH dust flux opacity (for H_FUV)

    # Total FUV dust flux opacity:
    @fastmath α_F_FUV_dust   =   α_F_FUV_Sil  +  α_F_FUV_C  +  α_F_FUV_PAH

    # Total FUV absorption opacity (for J_FUV):
    @fastmath α_abs_FUV      =   ( α_gas_abs_FUV  
                                 + Dust.κd_abs_FUV_Sil * ρd_Sil  +  Dust.κd_abs_FUV_C * ρd_C 
                                 + Dust.κd_abs_FUV_PAH * ρd_PAH )

    # Total FUV flux opacity (for H_FUV):
    @fastmath α_F_FUV        =   α_gas_F_FUV  +  α_F_FUV_dust           

    # Absorption time-scale:
    @fastmath Δt_FUV         =   inv_c̃ / max( α_abs_FUV, α_F_FUV ) 

    # Absorption update for FUV band:
    J_FUV          = max( AbsorptionUpdateJorH( J_FUV_old, Δt, dJ_FUVdt, α_abs_FUV, c̃ ), Jeps )
    H_r_FUV        = AbsorptionUpdateJorH( H_r_FUV_old, Δt, dHr_FUVdt, α_F_FUV, c̃ )
    H_Z_FUV        = AbsorptionUpdateJorH( H_Z_FUV_old, Δt, dHZ_FUVdt, α_F_FUV, c̃ )

    # Compute relative changes in FUV J, H_r, and H_Z:
    ΔJ_FUV         = abs( J_FUV   - J_FUV_old )   / ( J_FUV_old  +  J_floor )
    ΔH_r_FUV       = abs( H_r_FUV - H_r_FUV_old ) / ( abs(H_r_FUV_old) +  H_floor )
    ΔH_Z_FUV       = abs( H_Z_FUV - H_Z_FUV_old ) / ( abs(H_Z_FUV_old) +  H_floor )

    ΔFUV = max( ΔJ_FUV, ΔH_r_FUV, ΔH_Z_FUV )
    if ( ΔFUV > tol_Abs  &&  Δt > tol_Abs * Δt_FUV ) || !isfinite(ΔFUV)
        # Change in FUV was too large. Reject step: 
        return RejectAbsorption( ΔFUV )
    end

    # Change in total gas + dust momentum:
    @fastmath Δρu_r_rad_FUV   = Four_π_over_c * ( inv_c̃ * (H_r_FUV_old - H_r_FUV)  +  inv_c̃ * Δt * dHr_FUVdt )
    @fastmath Δρu_Z_rad_FUV   = Four_π_over_c * ( inv_c̃ * (H_Z_FUV_old - H_Z_FUV)  +  inv_c̃ * Δt * dHZ_FUVdt )

    # Add total radiation pressure momentum contribution:
    Δρu_r_rad += Δρu_r_rad_FUV
    Δρu_Z_rad += Δρu_Z_rad_FUV

    # Add the fractional contributions to gas vs. dust:
    δr_gas, δZ_gas, δr_Sil, δZ_Sil, δr_C, δZ_C, δr_PAH, δZ_PAH =
    SplitRadPressure( Δρu_r_rad_FUV, Δρu_Z_rad_FUV,
                      α_gas_F_FUV, α_F_FUV_Sil, α_F_FUV_C, α_F_FUV_PAH )

    Δρu_r_rad_gas += δr_gas;  Δρu_Z_rad_gas += δZ_gas
    Δρu_r_rad_Sil += δr_Sil;  Δρu_Z_rad_Sil += δZ_Sil
    Δρu_r_rad_C   += δr_C;    Δρu_Z_rad_C   += δZ_C
    Δρu_r_rad_PAH += δr_PAH;  Δρu_Z_rad_PAH += δZ_PAH

    #######################################################################
    #######                                                         #######
    #######                 L Y M A N - W E R N E R                 #######
    #######                                                         #######
    #######################################################################

    # Lyman-Werner opacities:

    # LW gas absorption opacity:
    @fastmath α_gas_abs_LW  =   ( nHI * σHI_LW    +  nH2  * σH2_LW  +  nHD  * σHD_LW 
                                + nCI * σCI_LW_E  +  nOHx * σOH_LW  +  nCHx * σCH_LW
                                + nCO * σCO_LW    +  nSI  * σSI_LW_E )    

    # LW flux opacity (adds Thomson scattering) 
    @fastmath α_gas_F_LW    =   α_gas_abs_LW  +  α_F_Thomson   
    
    @fastmath α_F_LW_Sil    =   Dust.κd_F_LW_Sil * ρd_Sil   # Silicate dust flux opacity (for H_LW)
    @fastmath α_F_LW_C      =   Dust.κd_F_LW_C   * ρd_C     # Graphite dust flux opacity (for H_LW)
    @fastmath α_F_LW_PAH    =   Dust.κd_F_LW_PAH * ρd_PAH   # PAH dust flux opacity (for H_LW)

    # Total LW dust flux opacity:
    @fastmath α_F_LW_dust   =   α_F_LW_Sil  +  α_F_LW_C  +  α_F_LW_PAH

    # Total LW absorption opacity (for J_LW):
    @fastmath α_abs_LW      =   (  α_gas_abs_LW  
                                 + Dust.κd_abs_LW_Sil * ρd_Sil  +  Dust.κd_abs_LW_C * ρd_C 
                                 + Dust.κd_abs_LW_PAH * ρd_PAH )

    # Total LW flux opacity (for H_LW):
    @fastmath α_F_LW        =   α_gas_F_LW  +  α_F_LW_dust   

    # Absorption time-scale:
    @fastmath Δt_LW         =   inv_c̃ / max( α_abs_LW, α_F_LW )

    # Absorption update for LW band:
    J_LW          = max( AbsorptionUpdateJorH( J_LW_old, Δt, dJ_LWdt, α_abs_LW, c̃ ), Jeps )
    H_r_LW        = AbsorptionUpdateJorH( H_r_LW_old, Δt, dHr_LWdt, α_F_LW, c̃ )
    H_Z_LW        = AbsorptionUpdateJorH( H_Z_LW_old, Δt, dHZ_LWdt, α_F_LW, c̃ )

    # Compute relative changes in LW J, H_r, and H_Z:
    ΔJ_LW         = abs( J_LW   - J_LW_old )   / ( J_LW_old  +  J_floor )
    ΔH_r_LW       = abs( H_r_LW - H_r_LW_old ) / ( abs(H_r_LW_old)  +  H_floor )
    ΔH_Z_LW       = abs( H_Z_LW - H_Z_LW_old ) / ( abs(H_Z_LW_old)  +  H_floor )

    ΔLW = max( ΔJ_LW, ΔH_r_LW, ΔH_Z_LW )
    if ( ΔLW > tol_Abs  &&  Δt > tol_Abs * Δt_LW ) || !isfinite(ΔLW)
        # Change in LW was too large. Reject step: 
        return RejectAbsorption( ΔLW )
    end

    # Change in total gas + dust momentum:
    @fastmath Δρu_r_rad_LW = Four_π_over_c * ( inv_c̃ * (H_r_LW_old - H_r_LW)  +  inv_c̃ * Δt * dHr_LWdt )
    @fastmath Δρu_Z_rad_LW = Four_π_over_c * ( inv_c̃ * (H_Z_LW_old - H_Z_LW)  +  inv_c̃ * Δt * dHZ_LWdt )

    # Add total radiation pressure momentum contribution:
    Δρu_r_rad   += Δρu_r_rad_LW
    Δρu_Z_rad   += Δρu_Z_rad_LW

    # Add the fractional contributions to gas vs. dust:
    δr_gas, δZ_gas, δr_Sil, δZ_Sil, δr_C, δZ_C, δr_PAH, δZ_PAH =
    SplitRadPressure( Δρu_r_rad_LW, Δρu_Z_rad_LW,
                      α_gas_F_LW, α_F_LW_Sil, α_F_LW_C, α_F_LW_PAH )

    Δρu_r_rad_gas += δr_gas;  Δρu_Z_rad_gas += δZ_gas
    Δρu_r_rad_Sil += δr_Sil;  Δρu_Z_rad_Sil += δZ_Sil
    Δρu_r_rad_C   += δr_C;    Δρu_Z_rad_C   += δZ_C
    Δρu_r_rad_PAH += δr_PAH;  Δρu_Z_rad_PAH += δZ_PAH

    #######################################################################
    #######                                                         #######
    #######                         E U V 1                         #######
    #######                                                         #######
    #######################################################################

    # EUV1 opacities:

    # EUV1 gas absorption opacity (D I has same cross-section as H I):
    @fastmath α_gas_abs_EUV1  =   ( nHI * StellAv.σHI_EUV1_E  +  nH2  * σH2_EUV1_E   +  nDI * StellAv.σHI_EUV1_E  
                                  + nCI * σCI_EUV1_E          +  nCII * σCII_EUV1_E  +  nOI * σOI_EUV1_E
                                  + nSI * σSI_EUV1_E          +  nSII * σSII_EUV1_E )

    # EUV1 gas flux opacity (adds Thomson scattering)
    @fastmath α_gas_F_EUV1    =   α_gas_abs_EUV1  +  α_F_Thomson 

    @fastmath α_F_EUV1_Sil    =   Dust.κd_F_EUV1_Sil * ρd_Sil   # Silicate dust flux opacity (for H_EUV1)
    @fastmath α_F_EUV1_C      =   Dust.κd_F_EUV1_C   * ρd_C     # Graphite dust flux opacity (for H_EUV1)
    @fastmath α_F_EUV1_PAH    =   Dust.κd_F_EUV1_PAH * ρd_PAH   # PAH dust flux opacity (for H_EUV1)

    # Total EUV1 dust flux opacity:
    @fastmath α_F_EUV1_dust   =   α_F_EUV1_Sil  +  α_F_EUV1_C  +  α_F_EUV1_PAH

    # Total EUV1 absorption opacity (for J_EUV1):
    @fastmath α_abs_EUV1      =   ( α_gas_abs_EUV1  
                                  + Dust.κd_abs_EUV1_Sil * ρd_Sil  +  Dust.κd_abs_EUV1_C * ρd_C 
                                  + Dust.κd_abs_EUV1_PAH * ρd_PAH )

    # Total EUV1 flux opacity (for H_EUV1):
    @fastmath α_F_EUV1        =   α_gas_F_EUV1  +  α_F_EUV1_dust 

    # Absorption time-scale:
    @fastmath Δt_EUV1         =   inv_c̃ / max( α_abs_EUV1, α_F_EUV1 )

    # Absorption update for EUV1 band:
    J_EUV1        = max( AbsorptionUpdateJorH( J_EUV1_old, Δt, dJ_EUV1dt, α_abs_EUV1, c̃ ), Jeps )
    H_r_EUV1      = AbsorptionUpdateJorH( H_r_EUV1_old, Δt, dHr_EUV1dt, α_F_EUV1, c̃ )
    H_Z_EUV1      = AbsorptionUpdateJorH( H_Z_EUV1_old, Δt, dHZ_EUV1dt, α_F_EUV1, c̃ )

    # Compute relative changes in EUV1 J, H_r, and H_Z:
    ΔJ_EUV1       = abs( J_EUV1   - J_EUV1_old )   / ( J_EUV1_old  +  J_floor )
    ΔH_r_EUV1     = abs( H_r_EUV1 - H_r_EUV1_old ) / ( abs(H_r_EUV1_old)  +  H_floor )
    ΔH_Z_EUV1     = abs( H_Z_EUV1 - H_Z_EUV1_old ) / ( abs(H_Z_EUV1_old)  +  H_floor )

    ΔEUV1 = max( ΔJ_EUV1, ΔH_r_EUV1, ΔH_Z_EUV1 )
    if ( ΔEUV1 > tol_Abs  &&  Δt > tol_Abs * Δt_EUV1 ) || !isfinite(ΔEUV1)
        # Change in EUV1 was too large. Reject step: 
        return RejectAbsorption( ΔEUV1 )
    end

    # Change in total gas + dust momentum:
    @fastmath Δρu_r_rad_EUV1 = Four_π_over_c * ( inv_c̃ * (H_r_EUV1_old - H_r_EUV1)  +  inv_c̃ * Δt * dHr_EUV1dt )
    @fastmath Δρu_Z_rad_EUV1 = Four_π_over_c * ( inv_c̃ * (H_Z_EUV1_old - H_Z_EUV1)  +  inv_c̃ * Δt * dHZ_EUV1dt )

    # Add total radiation pressure momentum contribution:
    Δρu_r_rad     += Δρu_r_rad_EUV1
    Δρu_Z_rad     += Δρu_Z_rad_EUV1

    # Add the fractional contributions to gas vs. dust:
    δr_gas, δZ_gas, δr_Sil, δZ_Sil, δr_C, δZ_C, δr_PAH, δZ_PAH =
    SplitRadPressure( Δρu_r_rad_EUV1, Δρu_Z_rad_EUV1,
                      α_gas_F_EUV1, α_F_EUV1_Sil, α_F_EUV1_C, α_F_EUV1_PAH )

    Δρu_r_rad_gas += δr_gas;  Δρu_Z_rad_gas += δZ_gas
    Δρu_r_rad_Sil += δr_Sil;  Δρu_Z_rad_Sil += δZ_Sil
    Δρu_r_rad_C   += δr_C;    Δρu_Z_rad_C   += δZ_C
    Δρu_r_rad_PAH += δr_PAH;  Δρu_Z_rad_PAH += δZ_PAH

    #######################################################################
    #######                                                         #######
    #######                         E U V 2                         #######
    #######                                                         #######
    #######################################################################

    # EUV2 opacities:

    # EUV2 gas absorption opacity (D I has same opacity as H I):
    @fastmath α_gas_abs_EUV2  =   ( nHI  * StellAv.σHI_EUV2_E  +  nH2   * σH2_EUV2_E   +  nHeI * StellAv.σHeI_EUV2_E 
                                  + nDI  * StellAv.σHI_EUV2_E  +  nCI   * σCI_EUV2_E   +  nCII * σCII_EUV2_E
                                  + nOI  * σOI_EUV2_E          +  nOII  * σOII_EUV2_E  +  nSI  * σSI_EUV2_E
                                  + nSII * σSII_EUV2_E         +  nSIII * σSIII_EUV2_E )                     

    # EUV2 gas flux opacity (adds Thomson scattering)
    @fastmath α_gas_F_EUV2    =   α_gas_abs_EUV2  +  α_F_Thomson  

    @fastmath α_F_EUV2_Sil    =   Dust.κd_F_EUV2_Sil * ρd_Sil   # Silicate dust flux opacity (for H_EUV2)
    @fastmath α_F_EUV2_C      =   Dust.κd_F_EUV2_C   * ρd_C     # Graphite dust flux opacity (for H_EUV2)
    @fastmath α_F_EUV2_PAH    =   Dust.κd_F_EUV2_PAH * ρd_PAH   # PAH dust flux opacity (for H_EUV2)

    # Total EUV2 dust flux opacity:
    @fastmath α_F_EUV2_dust   =   α_F_EUV2_Sil  +  α_F_EUV2_C  +  α_F_EUV2_PAH

    # Total EUV2 absorption opacity (for J_EUV2):
    @fastmath α_abs_EUV2      =   ( α_gas_abs_EUV2  
                                  + Dust.κd_abs_EUV2_Sil * ρd_Sil  +  Dust.κd_abs_EUV2_C * ρd_C 
                                  + Dust.κd_abs_EUV2_PAH * ρd_PAH )

    # Total EUV2 flux opacity (for H_EUV2):
    @fastmath α_F_EUV2        =   α_gas_F_EUV2  +  α_F_EUV2_dust

    # Absorption time-scale:
    @fastmath Δt_EUV2         =   inv_c̃ / max( α_abs_EUV2, α_F_EUV2 )

    # Absorption update for EUV2 band:
    J_EUV2        = max( AbsorptionUpdateJorH( J_EUV2_old, Δt, dJ_EUV2dt, α_abs_EUV2, c̃ ), Jeps )
    H_r_EUV2      = AbsorptionUpdateJorH( H_r_EUV2_old, Δt, dHr_EUV2dt, α_F_EUV2, c̃ )
    H_Z_EUV2      = AbsorptionUpdateJorH( H_Z_EUV2_old, Δt, dHZ_EUV2dt, α_F_EUV2, c̃ )

    # Compute relative changes in EUV2 J, H_r, and H_Z:
    ΔJ_EUV2       = abs( J_EUV2   - J_EUV2_old )   / ( J_EUV2_old  +  J_floor )
    ΔH_r_EUV2     = abs( H_r_EUV2 - H_r_EUV2_old ) / ( abs(H_r_EUV2_old)  +  H_floor )
    ΔH_Z_EUV2     = abs( H_Z_EUV2 - H_Z_EUV2_old ) / ( abs(H_Z_EUV2_old)  +  H_floor )

    ΔEUV2 = max( ΔJ_EUV2, ΔH_r_EUV2, ΔH_Z_EUV2 )
    if ( ΔEUV2 > tol_Abs  &&  Δt > tol_Abs * Δt_EUV2 ) || !isfinite(ΔEUV2)   
        # Change in EUV2 was too large. Reject step: 
        return RejectAbsorption( ΔEUV2 )
    end

    # Change in total gas + dust momentum:
    @fastmath Δρu_r_rad_EUV2 = Four_π_over_c * ( inv_c̃ * (H_r_EUV2_old - H_r_EUV2)  +  inv_c̃ * Δt * dHr_EUV2dt )
    @fastmath Δρu_Z_rad_EUV2 = Four_π_over_c * ( inv_c̃ * (H_Z_EUV2_old - H_Z_EUV2)  +  inv_c̃ * Δt * dHZ_EUV2dt )

    # Add total radiation pressure momentum contribution:
    Δρu_r_rad     += Δρu_r_rad_EUV2
    Δρu_Z_rad     += Δρu_Z_rad_EUV2

    # Add the fractional contributions to gas vs. dust:
    δr_gas, δZ_gas, δr_Sil, δZ_Sil, δr_C, δZ_C, δr_PAH, δZ_PAH =
    SplitRadPressure( Δρu_r_rad_EUV2, Δρu_Z_rad_EUV2,
                      α_gas_F_EUV2, α_F_EUV2_Sil, α_F_EUV2_C, α_F_EUV2_PAH )

    Δρu_r_rad_gas += δr_gas;  Δρu_Z_rad_gas += δZ_gas
    Δρu_r_rad_Sil += δr_Sil;  Δρu_Z_rad_Sil += δZ_Sil
    Δρu_r_rad_C   += δr_C;    Δρu_Z_rad_C   += δZ_C
    Δρu_r_rad_PAH += δr_PAH;  Δρu_Z_rad_PAH += δZ_PAH

    #######################################################################
    #######                                                         #######
    #######                         E U V 3                         #######
    #######                                                         #######
    #######################################################################

    # EUV3 opacities:

    # EUV3 gas absorption opacity:
    @fastmath α_gas_abs_EUV3  = ( nHI * StellAv.σHI_EUV3_E  +  nH2  * σH2_EUV3_E   +  nHeI  * StellAv.σHeI_EUV3_E  +  nHeII * StellAv.σHeII_EUV3_E
                                + nDI * StellAv.σHI_EUV3_E  +  nCI  * σCI_EUV3_E   +  nCII  * σCII_EUV3_E
                                + nOI * σOI_EUV3_E          +  nOII * σOII_EUV3_E  +  nOIII * σOIII_EUV3_E
                                + nSI * σSI_EUV3_E          +  nSII * σSII_EUV3_E  +  nSIII * σSIII_EUV3_E )                      

    # EUV3 gas flux opacity (adds Thomson scattering):
    @fastmath α_gas_F_EUV3    =   α_gas_abs_EUV3  +  α_F_Thomson        

    @fastmath α_F_EUV3_Sil    =   Dust.κd_F_EUV3_Sil * ρd_Sil   # Silicate dust flux opacity (for H_EUV3)
    @fastmath α_F_EUV3_C      =   Dust.κd_F_EUV3_C   * ρd_C     # Graphite dust flux opacity (for H_EUV3)
    @fastmath α_F_EUV3_PAH    =   Dust.κd_F_EUV3_PAH * ρd_PAH   # PAH dust flux opacity (for H_EUV3)

    # Total EUV3 dust flux opacity:
    @fastmath α_F_EUV3_dust   =   α_F_EUV3_Sil  +  α_F_EUV3_C  +  α_F_EUV3_PAH

    # Total EUV3 absorption opacity (for J_EUV3):
    @fastmath α_abs_EUV3      =   ( α_gas_abs_EUV3  
                                  + Dust.κd_abs_EUV3_Sil * ρd_Sil  +  Dust.κd_abs_EUV3_C * ρd_C 
                                  + Dust.κd_abs_EUV3_PAH * ρd_PAH )

    # Total EUV3 flux opacity (for H_EUV3):
    @fastmath α_F_EUV3        =   α_gas_F_EUV3  +  α_F_EUV3_dust

    # Absorption time-scale:
    @fastmath Δt_EUV3         =   inv_c̃ / max( α_abs_EUV3, α_F_EUV3 )

    # Absorption update for EUV3 band:
    J_EUV3           = max( AbsorptionUpdateJorH( J_EUV3_old, Δt, dJ_EUV3dt, α_abs_EUV3, c̃ ), Jeps )
    H_r_EUV3         = AbsorptionUpdateJorH( H_r_EUV3_old, Δt, dHr_EUV3dt, α_F_EUV3, c̃ )
    H_Z_EUV3         = AbsorptionUpdateJorH( H_Z_EUV3_old, Δt, dHZ_EUV3dt, α_F_EUV3, c̃ )

    # Compute relative changes in EUV3 J, H_r, and H_Z:
    ΔJ_EUV3          = abs( J_EUV3   - J_EUV3_old )   / ( J_EUV3_old  +  J_floor )
    ΔH_r_EUV3        = abs( H_r_EUV3 - H_r_EUV3_old ) / ( abs(H_r_EUV3_old) +  H_floor )
    ΔH_Z_EUV3        = abs( H_Z_EUV3 - H_Z_EUV3_old ) / ( abs(H_Z_EUV3_old) +  H_floor )

    ΔEUV3 = max( ΔJ_EUV3, ΔH_r_EUV3, ΔH_Z_EUV3 )
    if ( ΔEUV3 > tol_Abs  &&  Δt > tol_Abs * Δt_EUV3 ) || !isfinite(ΔEUV3)  
        # Change in EUV3 was too large. Reject step:
        return RejectAbsorption( ΔEUV3 )
    end

    # Change in total gas + dust momentum:
    @fastmath Δρu_r_rad_EUV3 = Four_π_over_c * ( inv_c̃ * (H_r_EUV3_old - H_r_EUV3)  +  inv_c̃ * Δt * dHr_EUV3dt )
    @fastmath Δρu_Z_rad_EUV3 = Four_π_over_c * ( inv_c̃ * (H_Z_EUV3_old - H_Z_EUV3)  +  inv_c̃ * Δt * dHZ_EUV3dt )

    # Add total radiation pressure momentum contribution:
    Δρu_r_rad     += Δρu_r_rad_EUV3
    Δρu_Z_rad     += Δρu_Z_rad_EUV3

    # Add the fractional contributions to gas vs. dust:
    δr_gas, δZ_gas, δr_Sil, δZ_Sil, δr_C, δZ_C, δr_PAH, δZ_PAH =
    SplitRadPressure( Δρu_r_rad_EUV3, Δρu_Z_rad_EUV3,
                      α_gas_F_EUV3, α_F_EUV3_Sil, α_F_EUV3_C, α_F_EUV3_PAH )

    Δρu_r_rad_gas += δr_gas;  Δρu_Z_rad_gas += δZ_gas
    Δρu_r_rad_Sil += δr_Sil;  Δρu_Z_rad_Sil += δZ_Sil
    Δρu_r_rad_C   += δr_C;    Δρu_Z_rad_C   += δZ_C
    Δρu_r_rad_PAH += δr_PAH;  Δρu_Z_rad_PAH += δZ_PAH

    #######################################################################
    #######                                                         #######
    #######                        L Y M A N - α                    #######
    #######                                                         #######
    #######################################################################

    # If Lyα RT is simulated, the following block computes the emission 
    # and absorption of Lyα photons over the sub-step, including the 
    # resulting momentum change from radiation pressure.

    if !isnothing(Lyα)

        # Unpack Lyα RSL:
        c̃_Lyα        = Lyα.c̃_Lyα
        inv_c̃_Lyα    = @fastmath 1.0 / c̃_Lyα

        # Unpack arrays:
        J_Lyα        = Lyα.J_Lyα
        J_Lyα_old    = Lyα.J_Lyα_old
        H_r_Lyα      = Lyα.H_r_Lyα
        H_r_Lyα_old  = Lyα.H_r_Lyα_old
        H_Z_Lyα      = Lyα.H_Z_Lyα
        H_Z_Lyα_old  = Lyα.H_Z_Lyα_old
        Δν           = Lyα.Δν
        e_Lyα        = Lyα.e_Lyα
        α_F_Lyα      = Lyα.α_F_Lyα

        # Pre-compute some stuff before the bin-loop:
        @fastmath c̃∫dν_js_Lyα_ij = c̃_Lyα * Lyα.∫dν_js_Lyα[i,j]    # c̃_Lyα × frequency-integrated emissivity 
        αc_Lyα_ij                = Lyα.αc_Lyα[i,j]                # Absorption continuum opacity
        @fastmath pd_α_ij        = Lyα.pd[i,j] * Lyα.α_Lyα[i,j]   # pd × α, pre-compute once 

        # Dust flux opacities for Lyα:
        @fastmath αF_Lyα_Sil_ij = Dust.κd_F_Lyα_Sil * ρd_Sil
        @fastmath αF_Lyα_C_ij   = Dust.κd_F_Lyα_C   * ρd_C
        @fastmath αF_Lyα_PAH_ij = Dust.κd_F_Lyα_PAH * ρd_PAH

        # Lyα continuum absorption time-scale, only considering absorption/destruction processes:
        @fastmath Δt_Lyα        = inv_c̃_Lyα / ( αc_Lyα_ij  +  pd_α_ij )

        # Update Lyα J and H in all bins from absorption and emission, and compute the 
        # radiation pressure momentum to be added to the gas and dust:

        ∫dν_J_Lyα_new  = 0.0   # Initialize would-be updated ∫dν_J_Lyα
        Δρu_r_rad_Lyα  = 0.0   # Initialize change in r gas + dust momentum from Lyα (diagnostic)
        Δρu_Z_rad_Lyα  = 0.0   # Initialize change in Z gas + dust momentum from Lyα (diagnostic)
        @inbounds for k in Lyα.dN_ν
            # Bin width:
            Δν_k = Δν[k]

            # Compute dJ_Lyα/dt from Lyα emissivity:
            @fastmath dJ_Lyαdt       = c̃∫dν_js_Lyα_ij * Lyα.ϕ_ν[k,i,j]

            # Compute total absorption opacity for bin k:
            @fastmath α_abs_Lyα_ijk  = αc_Lyα_ij  +  pd_α_ij * Lyα.Hvoigt[k,i,j]

            # Update Lyα J:
            J_Lyα[k,i,j]   = max( AbsorptionUpdateJorH( J_Lyα_old[k,i,j], Δt, dJ_Lyαdt, α_abs_Lyα_ijk, c̃_Lyα ), Jeps )

            # Update Lyα H (flux opacity already computed):
            αF_Lyα_ijk     = α_F_Lyα[k,i,j]
            H_r_Lyα[k,i,j] = @fastmath AbsorptionUpdateJorH( H_r_Lyα_old[k,i,j], Δt, 0.0, αF_Lyα_ijk, c̃_Lyα  )
            H_Z_Lyα[k,i,j] = @fastmath AbsorptionUpdateJorH( H_Z_Lyα_old[k,i,j], Δt, 0.0, αF_Lyα_ijk, c̃_Lyα  )

            # Total momentum added to gas + dust from this bin:
            Δν_k_Four_π_over_c = @fastmath Δν_k * Four_π_over_c
            Δρu_r_rad_Lyα_k    = @fastmath Δν_k_Four_π_over_c * ( inv_c̃_Lyα * (H_r_Lyα_old[k,i,j] - H_r_Lyα[k,i,j]) )
            Δρu_Z_rad_Lyα_k    = @fastmath Δν_k_Four_π_over_c * ( inv_c̃_Lyα * (H_Z_Lyα_old[k,i,j] - H_Z_Lyα[k,i,j]) )

            # Add to the total momentum injected to gas & dust:
            Δρu_r_rad_Lyα  += Δρu_r_rad_Lyα_k
            Δρu_Z_rad_Lyα  += Δρu_Z_rad_Lyα_k

            # Get Lyα gas flux opacity by subtracting the dust flux opacity from the total:
            α_gas_F_Lyα_k   = max( αF_Lyα_ijk - αF_Lyα_Sil_ij - αF_Lyα_C_ij - αF_Lyα_PAH_ij, 0.0 )

            # Compute fraction of momentum from bin k to be dumped into gas vs. dust:
            δr_gas, δZ_gas, δr_Sil, δZ_Sil, δr_C, δZ_C, δr_PAH, δZ_PAH =
            SplitRadPressure( Δρu_r_rad_Lyα_k, Δρu_Z_rad_Lyα_k,
                              α_gas_F_Lyα_k, αF_Lyα_Sil_ij, αF_Lyα_C_ij, αF_Lyα_PAH_ij )

            Δρu_r_rad_gas += δr_gas;  Δρu_Z_rad_gas += δZ_gas
            Δρu_r_rad_Sil += δr_Sil;  Δρu_Z_rad_Sil += δZ_Sil
            Δρu_r_rad_C   += δr_C;    Δρu_Z_rad_C   += δZ_C
            Δρu_r_rad_PAH += δr_PAH;  Δρu_Z_rad_PAH += δZ_PAH

            # Add contribution to ∫dν_J_Lyα_new from bin k:
            @fastmath ∫dν_J_Lyα_new  += Δν_k * J_Lyα[k,i,j]
        end

        # Finally, multiply by 4π/c to get the would-be new e_Lyα:
        @fastmath e_Lyα_new = Four_π_over_c * ∫dν_J_Lyα_new 

        # Check relative change in e_Lyα:
        e_Lyα_old = e_Lyα[i,j]
        Δe_Lyα    = abs( e_Lyα_new - e_Lyα_old ) / ( e_Lyα_old  +  e_Lyα_floor )

        if ( Δe_Lyα > tol_Abs  &&  Δt > tol_Abs * Δt_Lyα ) || !isfinite(Δe_Lyα) 
            # Change in Lyα energy density was too large. Reject step:
            return RejectAbsorption( Δe_Lyα )
        end

    else 
        # If we don't simulate Lyα RT, just set e_Lyα_new to zero:
        ∫dν_J_Lyα_new = 0.0
        e_Lyα_new     = 0.0
        Δe_Lyα        = 0.0   # No change in e_Lyα
        Δρu_r_rad_Lyα = 0.0   # No Lyα radiation pressure 
        Δρu_Z_rad_Lyα = 0.0
        Δt_Lyα        = Inf   # No Lyα absorption time-scale to respect 
    end


    #######################################################################
    #######                                                         #######
    #######                      I N F R A R E D                    #######
    #######                                                         #######
    #######################################################################
    
    # Compute Σ_{B} κ_{B,abs} J_{B} over all non-IR bands, and for each dust type:
    @fastmath κabsB_Sil_JB  =  ( Dust.κd_abs_opt_Sil  * J_opt   +  Dust.κd_abs_FUV_Sil  * J_FUV 
                               + Dust.κd_abs_LW_Sil   * J_LW    +  Dust.κd_abs_EUV1_Sil * J_EUV1  
                               + Dust.κd_abs_EUV2_Sil * J_EUV2  +  Dust.κd_abs_EUV3_Sil * J_EUV3
                               + Dust.κd_abs_Lyα_Sil  * ∫dν_J_Lyα_new )

    @fastmath κabsB_C_JB    =  ( Dust.κd_abs_opt_C  * J_opt   +  Dust.κd_abs_FUV_C  * J_FUV 
                               + Dust.κd_abs_LW_C   * J_LW    +  Dust.κd_abs_EUV1_C * J_EUV1  
                               + Dust.κd_abs_EUV2_C * J_EUV2  +  Dust.κd_abs_EUV3_C * J_EUV3
                               + Dust.κd_abs_Lyα_C  * ∫dν_J_Lyα_new )

    @fastmath κabsB_PAH_JB  =  ( Dust.κd_abs_opt_PAH  * J_opt   +  Dust.κd_abs_FUV_PAH  * J_FUV 
                               + Dust.κd_abs_LW_PAH   * J_LW    +  Dust.κd_abs_EUV1_PAH * J_EUV1  
                               + Dust.κd_abs_EUV2_PAH * J_EUV2  +  Dust.κd_abs_EUV3_PAH * J_EUV3
                               + Dust.κd_abs_Lyα_PAH  * ∫dν_J_Lyα_new )

    # Get collision heating/cooling contribution from each bin to the IR emissivity: 
    @fastmath js_IR_coll_Sil  =  Λdg( T_old, Td_Sil_old, ρd_Sil, nH, Sil ) * inv_Four_π
    @fastmath js_IR_coll_C    =  Λdg( T_old, Td_C_old,   ρd_C,   nH, Gra ) * inv_Four_π
    @fastmath js_IR_coll_PAH  =  Λdg( T_old, Td_PAH_old, ρd_PAH, nH, PAH ) * inv_Four_π

    # IR emissivity & corresponding rate of change in J_IR:
    @fastmath js_IR_coll  =  js_IR_coll_Sil  +  js_IR_coll_C  +  js_IR_coll_PAH   # Total collision heating/cooling contribution 
    @fastmath js_IR       =  ( ρd_Sil * κabsB_Sil_JB  +  ρd_C * κabsB_C_JB  + ρd_PAH * κabsB_PAH_JB  
                             + js_IR_coll )
    @fastmath dJ_IRdt     =  js_IR * c̃

    # Update J_IR:
    J_IR       =  max( AbsorptionUpdateJorH( J_IR_old, Δt, dJ_IRdt, 0.0, c̃ ), Jeps )

    # Update Trad:
    Trad_new   =  ComputeTrad( Td_Sil_old, Td_C_old, Td_PAH_old, ρd_Sil, ρd_C, ρd_PAH, 
                               c̃Δt, J_IR_old, Trad_old, Dust )

    # Update dust temperatures (in case dust is present), using the new J_IR, J_{B ≠ IR},
    # e_Lyα, and Trad:
    if ρd_Sil > 0.0
        Td_Sil_new = ComputeTd( J_IR, J_opt, J_FUV, J_LW, J_EUV1,
                                J_EUV2, J_EUV3, Dust, Sil, TCMB,
                                Trad_new, T_old, nH, e_Lyα_new, 
                                Td_Sil_old )
    else
        Td_Sil_new = Td_Sil_old
    end

    if ρd_C > 0.0
        Td_C_new   = ComputeTd( J_IR, J_opt, J_FUV, J_LW, J_EUV1,
                                J_EUV2, J_EUV3, Dust, Gra, TCMB,
                                Trad_new, T_old, nH, e_Lyα_new,
                                Td_C_old )
    else
        Td_C_new   = Td_C_old
    end

    if ρd_PAH > 0.0
        Td_PAH_new = ComputeTd( J_IR, J_opt, J_FUV, J_LW, J_EUV1,
                                J_EUV2, J_EUV3, Dust, PAH, TCMB,
                                Trad_new, T_old, nH, e_Lyα_new,
                                Td_PAH_old )
    else
        Td_PAH_new = Td_PAH_old
    end

    # Get IR dust flux opacity (Rosseland-mean, evaluated at the new radiation temperature):
    @fastmath α_F_IR_Sil  =  ρd_Sil * κR(Dust, Trad_new, Sil)
    @fastmath α_F_IR_C    =  ρd_C   * κR(Dust, Trad_new, Gra)
    @fastmath α_F_IR_PAH  =  ρd_PAH * κR(Dust, Trad_new, PAH)

    # Total IR flux opacity (dust + Thomson scattering):
    @fastmath α_F_IR_gas  =  α_F_Thomson
    @fastmath α_F_IR      =  ( α_F_IR_Sil + α_F_IR_C + α_F_IR_PAH )  +  α_F_IR_gas

    # Absorption time-scale:
    @fastmath Δt_IR       =  inv_c̃ / α_F_IR

    # Update H_IR:
    H_r_IR    = AbsorptionUpdateJorH( H_r_IR_old, Δt, 0.0, α_F_IR, c̃ )
    H_Z_IR    = AbsorptionUpdateJorH( H_Z_IR_old, Δt, 0.0, α_F_IR, c̃ )

    # Compute relative changes in IR J, H_r, and H_Z.
    ΔJ_IR     = abs( J_IR   - J_IR_old )   / ( J_IR_old  +  J_floor )
    ΔH_r_IR   = abs( H_r_IR - H_r_IR_old ) / ( abs(H_r_IR_old)  +  H_floor )
    ΔH_Z_IR   = abs( H_Z_IR - H_Z_IR_old ) / ( abs(H_Z_IR_old)  +  H_floor )

    # If collisional heating is not negligible, also take into account the relative 
    # change in Td in deciding whether to accept the update:
    if abs(js_IR_coll) > tol_Abs * abs(js_IR)  # Absolute sign because the collisional contribution can be negative
        ΔTd_Sil = abs( Td_Sil_new - Td_Sil_old ) / ( abs(Td_Sil_old) +  Td_floor )
        ΔTd_C   = abs( Td_C_new   - Td_C_old )   / ( abs(Td_C_old)   +  Td_floor )
        ΔTd_PAH = abs( Td_PAH_new - Td_PAH_old ) / ( abs(Td_PAH_old) +  Td_floor )
    else
        # Otherwise ignore change in Td:
        ΔTd_Sil = 0.0;  ΔTd_C = 0.0;  ΔTd_PAH = 0.0
    end

    ΔIR = max( ΔJ_IR, ΔH_r_IR, ΔH_Z_IR, ΔTd_Sil, ΔTd_C, ΔTd_PAH )
    if ( ΔIR > tol_Abs  &&  Δt > tol_Abs * Δt_IR ) || !isfinite(ΔIR) 
        # Change in IR or dust temperatures was too large. Reject step:
        return RejectAbsorption( ΔIR )
    end

    # Add IR radiation pressure momentum contribution:
    @fastmath Δρu_r_rad_IR += Four_π_over_c * ( inv_c̃ * (H_r_IR_old - H_r_IR) )
    @fastmath Δρu_Z_rad_IR += Four_π_over_c * ( inv_c̃ * (H_Z_IR_old - H_Z_IR) )

    # Add the fractional contributions to gas vs. dust:
    δr_gas, δZ_gas, δr_Sil, δZ_Sil, δr_C, δZ_C, δr_PAH, δZ_PAH =
    SplitRadPressure( Δρu_r_rad_IR, Δρu_Z_rad_IR,
                      α_F_IR_gas, α_F_IR_Sil, α_F_IR_C, α_F_IR_PAH )

    Δρu_r_rad_gas += δr_gas;  Δρu_Z_rad_gas += δZ_gas
    Δρu_r_rad_Sil += δr_Sil;  Δρu_Z_rad_Sil += δZ_Sil
    Δρu_r_rad_C   += δr_C;    Δρu_Z_rad_C   += δZ_C
    Δρu_r_rad_PAH += δr_PAH;  Δρu_Z_rad_PAH += δZ_PAH

    # --------------------------------------------------------------------------

    # Finally, for later determination of the next sub-cycle time-step, check the ratio between 
    # the sub-step and the absorption time-scales: 
    Δopt_forRT  = min( Δopt,   Δt / Δt_opt  )
    ΔFUV_forRT  = min( ΔFUV,   Δt / Δt_FUV  )
    ΔLW_forRT   = min( ΔLW,    Δt / Δt_LW   )
    ΔEUV1_forRT = min( ΔEUV1,  Δt / Δt_EUV1 )
    ΔEUV2_forRT = min( ΔEUV2,  Δt / Δt_EUV2 )
    ΔEUV3_forRT = min( ΔEUV3,  Δt / Δt_EUV3 )
    ΔLyα_forRT  = min( Δe_Lyα, Δt / Δt_Lyα  )
    ΔIR_forRT   = min( ΔIR,    Δt / Δt_IR   )

    ΔRT = max( Δopt_forRT,  ΔFUV_forRT,  ΔLW_forRT, 
               ΔEUV1_forRT, ΔEUV2_forRT, ΔEUV3_forRT, 
               ΔLyα_forRT,  ΔIR_forRT )

    # In case the changes in J and H were small enough, we accept the update and return 
    # all computed values: 
    REDO = false
    return REDO, J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, J_IR,
           H_r_opt, H_r_FUV, H_r_LW, H_r_EUV1, H_r_EUV2, H_r_EUV3, H_r_IR,
           H_Z_opt, H_Z_FUV, H_Z_LW, H_Z_EUV1, H_Z_EUV2, H_Z_EUV3, H_Z_IR,
           Δρu_r_rad, Δρu_Z_rad,  Δρu_r_rad_IR, Δρu_Z_rad_IR,       # These are mainly diagnostics 
           Δρu_r_rad_gas, Δρu_Z_rad_gas,                            # These are the momenta to be dumped into gas
           Δρu_r_rad_Sil, Δρu_Z_rad_Sil,                            # These are the momenta to be dumped into silicate dust 
           Δρu_r_rad_C,   Δρu_Z_rad_C,                              # These are the momenta to be dumped into graphite dust 
           Δρu_r_rad_PAH, Δρu_Z_rad_PAH,                            # These are the momenta to be dumped into PAH dust 
           ΔRT, js_IR, 
           Td_Sil_new, Td_C_new, Td_PAH_new,           # New dust temperatures 
           Trad_new,                                   # New IR radiation temperature 
           e_Lyα_new, Δρu_r_rad_Lyα, Δρu_Z_rad_Lyα     # New Lyα energy density, and Lyα radiation pressure diagnostic 
end


const tol_HeatCool::Float64          = 0.1
const ϵ_TINY_HEATINGCOOLING::Float64 = 0.01       # Use a simple explicit update if Δt < ϵ_TINY_HEATINGCOOLING × t_cool 
const ϵ_ΔT::Float64                  = 1e-4       # Use step-size ϵ_ΔT * T_old to estimate ∂Λ/∂T at T = T_old
const Δt_e_max::Float64              = 1e7 * yr   # Maximum time-step to consider for heating/cooling

""" This function performs a heating/cooling update 
    over a step Δt. It solves the linearized heating/cooling 
    equation using backward Euler (BDF1):
    
    (eⁿ⁺¹ - eⁿ)/Δt = Γ(eⁿ⁺¹) - Λ(eⁿ⁺¹) 
                   ≈ [ Γ(eⁿ) - Λ(eⁿ) ]  +  ( ∂Γ/∂e - ∂Λ/∂e ) × (eⁿ⁺¹ - eⁿ)
          
    The update is accepted if the relative change in e was small (< tol_HeatCool),
    if the updated e is physically acceptable, and if the linearized update was a 
    good approximation to a fully implicit BDF1 update. """
@inline function HeatingCoolingStep( Δt::Float64, 
                                  e_old::Float64,
                                     Td::NTuple{3, Float64}, # Dust temperatures for each bin (Sil, C, PAH)
                                    Zgr::NTuple{3, Float64}, # Grain charges for each bin (Sil, C, PAH)
                                    γad::Float64,
                                      n::Float64,
                                     nH::Float64,
                                     nC::Float64, 
                                     nO::Float64,
                                     nS::Float64,
                                    nHI::Float64,
                                    nH2::Float64,
                                   nHII::Float64,
                                   nHeI::Float64,
                                  nHeII::Float64,
                                 nHeIII::Float64,
                                    nHD::Float64,
                                    nCI::Float64,
                                    nCO::Float64,
                                   nCII::Float64,
                                    nOI::Float64,
                                   nOII::Float64,
                                  nOIII::Float64,
                                    nSI::Float64,  # Yields photoionization heating
                                   nSII::Float64,  # Yields photoionization heating
                                  nSIII::Float64,  # Yields cooling + photoionization heating
                                     ne::Float64,
                                      ρ::Float64,  # Total gas density 
                                   ZSil::Float64,  # Si gas metallicity 
                               Zgasrest::Float64,  # Non-C/Si gas metallicity 
                                     ρd::NTuple{3, Float64}, # Dust densities in each bin (Sil, C, PAH)
                                   N_CO::Float64,
                                  J_opt::Float64,
                                  J_FUV::Float64,
                                   J_LW::Float64,
                                 J_EUV1::Float64,
                                 J_EUV2::Float64,
                                 J_EUV3::Float64,
                          SelfShielding::Float64,
                                  e_Lyα::Float64 = 0.0,  # Lyα energy density 
                                      z::Float64 = 0.0,
                                    Γpd::Float64 = 0.0,
                                    OTS::Bool    = true,
                                e_floor::Float64 = 0.0,
                        StellarAverages::StellarBandAverages = DefaultStellarBandAverages
                                    )

    Conv_e_T      =  ( γad - 1 ) / (n * kB)   # Conversion factor for e → T 
    T_old         =  e_old * Conv_e_T         # Get temperature T_old corresponding to e_old 

    # Get heating rate (from photoelectric heating, Compton heating off the (non-CMB) radiation field,
    # photo-ionizations/dissociations, UV pumping, and cosmic ray heating):
    HeatingRate = HeatingRate_total( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3,
                                     T_old, Td, Zgr,
                                     ne, nH, nHI, nH2, nHII, nHeI, nHeII, 
                                     nCI, nCII, nOI, nOII, nOIII, nSI, nSII, nSIII,
                                     SelfShielding, ρd, Γpd, z, e_Lyα,
                                     StellarAverages ) 

    # Compute the cooling rate at e_old (inluding dust-gas collisions):
    CoolingRate = Λ( T_old, nHI, nH2, nHII, nHeI, nHeII, nHeIII, nHD, nCI, nCO, nCII, 
                     nOI, nOII, nOIII, nSIII, ne, nH, nC, nO, nS, ρ, ZSil, Zgasrest, N_CO, z, OTS ) + Λdg_tot( T_old, Td, ρd, nH )

    if DEBUG_PhotoThermoChemistry
        @assert isfinite( HeatingRate ) "HeatingRate = $HeatingRate, e_old = $(e_old), T_old = $(T_old)"
        @assert isfinite( CoolingRate ) "CoolingRate = $CoolingRate, e_old = $(e_old), T_old = $(T_old)"
        @assert isfinite( HeatingRate - CoolingRate ) """ HeatingRate - CoolingRate = $(HeatingRate-CoolingRate) 
                                                         (HeatingRate = $HeatingRate, CoolingRate = $CoolingRate, 
                                                         e_old = $e_old, T_old = $T_old) """
    end

    # Tolerance cooling/heating time-scale for accuracy purposes:
    t_cool = e_old / abs( HeatingRate - CoolingRate )
    Δt_e   = min( tol_HeatCool * t_cool, Δt_e_max )

    if Δt > Δt_e
        # Reject step if the time-step is too large:
        REDO = true 
        return REDO, 0.0, Inf, Δt_e
    end

    # If the time-step is tiny compared to the cooling/heating time-scale, skip doing a fancy semi-implicit
    # update and use an explicit Euler update instead (this saves multiple evaluations of the cooling and heating
    # functions, and therefore boost performance):
    if Δt < ϵ_TINY_HEATINGCOOLING * t_cool
        # Explicit candidate update: 
        e_new =  e_old  +  ( HeatingRate - CoolingRate ) * Δt

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

    # ---------------------------------------------------------------------

    # Compute ∂Λ/∂e using a 1st-order finite difference approximation (higher order 
    # requires many more expensive calls to the cooling function):

    @fastmath ΔT       = ϵ_ΔT * T_old
    @fastmath inv_ΔT   = 1.0 / ΔT 
    @fastmath TplusΔT  = T_old  +  ΔT 
    
    CoolingRate_plus   = Λ( TplusΔT, nHI, nH2, nHII, nHeI, nHeII, nHeIII, nHD, nCI, nCO, nCII, 
                            nOI, nOII, nOIII, nSIII, ne, nH, nC, nO, nS, ρ, ZSil, Zgasrest, N_CO, z, OTS ) + Λdg_tot( TplusΔT, Td, ρd, nH )

    # Put it all together to get ∂Λ/∂e (NOTE: with this forward difference approximation, 
    # we can re-use the already calculated cooling rate at T_old):
    @fastmath ∂CoolingRate∂T = ( CoolingRate_plus - CoolingRate ) * inv_ΔT

    @fastmath ∂CoolingRate∂e = ∂CoolingRate∂T * Conv_e_T  # Convert from ∂/∂T to ∂/∂e   
    
    # Next, compute ∂Γ/∂e using a 1st-order finite difference approximation
    HeatingRate_plus   = HeatingRate_total( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3,
                                            TplusΔT, Td, Zgr, ne, nH, nHI, nH2, nHII, nHeI, nHeII, 
                                            nCI, nCII, nOI, nOII, nOIII, nSI, nSII, nSIII,
                                            SelfShielding, ρd, Γpd, z, e_Lyα, 
                                            StellarAverages )

    # Put it all together to get ∂Γ/∂e:
    @fastmath ∂HeatingRate∂T = ( HeatingRate_plus - HeatingRate ) * inv_ΔT

    @fastmath ∂HeatingRate∂e = ∂HeatingRate∂T * Conv_e_T  # Convert from ∂/∂T to ∂/∂e

    # Get ∂F/∂e, where F = HeatingRate - CoolingRate (derivative of ∂e∂t_adv is zero):
    @fastmath ∂F∂e  = ∂HeatingRate∂e - ∂CoolingRate∂e

    # Linearized backward-Euler update of e:
    @fastmath e_new = e_old  +  ( HeatingRate - CoolingRate ) * Δt / ( 1  -  ∂F∂e * Δt )

    # ------- STABILITY & ACCURACY CHECK FOR TEMPERATURE UPDATE -------------------

    # Check that the relative change in e was sufficiently small, and that 
    # other stability/accuracy checks are satisfied:
    Δe    = abs( e_new  -  e_old ) / e_old

    # Tolerance cooling/heating time-scale for accuracy purposes:
    Δt_e  = min( tol_HeatCool * e_old / abs( HeatingRate - CoolingRate ), Δt_e_max )

    # Even if the change in e was small, we also check the residual, so that
    # our linearized solution actually approximates a fully implicit BDF1 solution.

    # The would-be new temperature:
    T_new            = Conv_e_T * e_new  
    
    # Heating rate at would-be new temperature:
    HeatingRate_new  = HeatingRate_total( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3,
                                          T_new, Td, Zgr, ne, nH, nHI, nH2, nHII, nHeI, nHeII, 
                                          nCI, nCII, nOI, nOII, nOIII, nSI, nSII, nSIII,
                                          SelfShielding, ρd, Γpd, z, e_Lyα,
                                          StellarAverages )
        
    # Cooling rate at would-be new temperature:
    CoolingRate_new  = Λ( T_new, nHI, nH2, nHII, nHeI, nHeII, nHeIII, nHD, nCI, nCO, nCII, 
                          nOI, nOII, nOIII, nSIII, ne, nH, nC, nO, nS, ρ, ZSil, Zgasrest, N_CO, z, OTS ) + Λdg_tot( T_new, Td, ρd, nH )

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


""" Compute & update the (top-hat average) Voigt profile Hvoigt,
    emission profile ϕ_ν, and Voigt parameter a and Doppler width, 
    for a single cell (i,j). 
        
    Convenient in the RHD sub-cycling. """
@inline function Update_Hvoigt_And_ϕν_cell!( Lyα::LyαArrays,
                                               i::Int, 
                                               j::Int,
                                               T::Float64  # Temperature in cell (i,j)
                                            )

    # Get Voigt parameter a and the Doppler width & its inverse:
    a_ij             = Compute_a( T )
    Δν_D_ij          = Compute_ΔνD( T )
    inv_Δν_D_ij      = @fastmath 1 / Δν_D_ij
    Lyα.a[i,j]       = a_ij
    Lyα.ΔνD[i,j]     = Δν_D_ij
    Lyα.inv_ΔνD[i,j] = inv_Δν_D_ij

    # Update Hvoigt:
    for k in Lyα.dN_ν
        @inbounds begin 
            # Get frequency displacements from line center:
            δν_C   = Lyα.δνC[k]
            δν_max = Lyα.δν[k+1]
            δν_min = Lyα.δν[k]

            x      = @fastmath δν_C   * inv_Δν_D_ij
            xmax   = @fastmath δν_max * inv_Δν_D_ij
            xmin   = @fastmath δν_min * inv_Δν_D_ij

            # Compute Hvoigt:
            Lyα.Hvoigt[k,i,j] = Htophat( x, xmax, xmin, a_ij )
        end
    end

    # Update ϕ_ν:
    ∫dν_ϕν = 0.0
    @inbounds for k in Lyα.dN_ν 
        @fastmath ∫dν_ϕν += Lyα.Δν[k] * Lyα.Hvoigt[k,i,j]
    end
    @fastmath inv_∫dν_ϕν = 1 / ∫dν_ϕν

    @inbounds for k in Lyα.dN_ν 
        @fastmath Lyα.ϕ_ν[k,i,j] = Lyα.Hvoigt[k,i,j] * inv_∫dν_ϕν
    end

    return nothing 
end


""" This function prepares the Lyα arrays in a cell (i,j)
    for sub-cycling of emission & absorption. """
@inline function PrepareLyαSubCycling!( Lyα::LyαArrays,
                                       Dust::DustState,
                                         ρd::NTuple{3, Float64},  # (ρd_Sil, ρd_C, ρd_PAH)
                                          i::Int, 
                                          j::Int,
                                          T::Float64,  # Temperature in cell (i,j) 
                                        nHI::Float64,
                                        nH2::Float64,
                                       nHII::Float64,
                                       nHeI::Float64,
                                         ne::Float64,
                                       nOHx::Float64,
                                       nCHx::Float64,
                                          z::Float64,
                                        OTS::Bool
                                               )

    # Extract dust bin densities:
    ρd_Sil, ρd_C, ρd_PAH = ρd 

    # Ignore dust absorption near the floor:
    ρd_Sil = (ρd_Sil > ρd_IGNORE) * ρd_Sil
    ρd_C   = (ρd_C   > ρd_IGNORE) * ρd_C
    ρd_PAH = (ρd_PAH > ρd_IGNORE) * ρd_PAH

    # Update Hvoigt & ϕ_ν, as well as a and Δν_D (and its inverse):
    Update_Hvoigt_And_ϕν_cell!( Lyα, i, j, T )

    # Most recent Lyα energy density:
    e_Lyα_old = Lyα.e_Lyα[i,j]

    # Lyα frequency-integrated emissivity:
    @fastmath Lyα.∫dν_js_Lyα[i,j] = ( ∫dν_js_recLyα( T, nHII, ne, OTS )
                                  +   ∫dν_js_collLyα( T, nHI, ne ) )

    # Lyα line opacity at line center:
    α_Lyα_ij        = @fastmath nHI * Compute_σ0( T )
    Lyα.α_Lyα[i,j]  = α_Lyα_ij

    # Lyα continuum gas absorption & flux opacity:
    αc_abs_Lyα_gas  = @fastmath nH2 * σRaman   +  nOHx * σOH_Lyα  +  nCHx * σCH_Lyα
    αc_F_Lyα_gas    = @fastmath αc_abs_Lyα_gas +  ne * σT 

    # Lyα continuum absorption opacity:
    Lyα.αc_Lyα[i,j] = @fastmath (  Dust.κd_abs_Lyα_Sil * ρd_Sil
                                 + Dust.κd_abs_Lyα_C   * ρd_C
                                 + Dust.κd_abs_Lyα_PAH * ρd_PAH
                                 + αc_abs_Lyα_gas )

    # Lyα destruction probability:
    if Lyα.LyαDestruction
        Lyα.pd[i,j] = pd_Lyα( nHII, ne, nHI, nHeI, nH2, T, z )
    else
        Lyα.pd[i,j] = 0.0
    end

    # Total Lyα flux opacity (from dust and gas):
    @fastmath αc_F_Lyα  = ( Dust.κd_F_Lyα_Sil * ρd_Sil
                          + Dust.κd_F_Lyα_C   * ρd_C
                          + Dust.κd_F_Lyα_PAH * ρd_PAH
                          + αc_F_Lyα_gas )

    @inbounds for k in Lyα.dN_ν
        # Flux opacity in bin k:
        @fastmath Lyα.α_F_Lyα[k,i,j] = α_Lyα_ij * Lyα.Hvoigt[k,i,j]  +  αc_F_Lyα
    end

    return e_Lyα_old
end


const T_floor::Float64             = 0.01            # Enforced temperature floor in the thermochemistry sub-cycling
const T_ceiling::Float64           = 1e9             # Enforced temperature ceiling in the thermochemistry sub-cycling
const ΔL_max::Float64              = 1e3 * pc        # Enforced maximum value of column density length in the estimate of N_H2, N_HD 
const tol_Chem::Float64            = 0.1             # Maximum relative change tolerance per sub-cycle time-step for the chemistry
const tol_enlargestep::Float64     = 0.5 * tol_Chem  # Recommend a 2 × larger sub-cycle time-step if relative changes were < tol_enlargestep
const Z_IGNORE_CARBOXYGEN::Float64 = 1e-6 * Zsun     # Ignore changes in C/O abundances for sub-cycling tolerance, if the metallicity is below this value 
const Z_IGNORE_SULFUR::Float64     = 1e-6 * Zsun     # Ignore changes in S abundances for sub-cycling tolerance, if the metallicity is below this value
const T_IGNORE_CARBOXYGEN::Float64 = 1e5             # Ignore changes in C/O abundances for sub-cycling tolerance, if T > T_IGNORE_CARBOXYGEN
const T_IGNORE_SULFUR::Float64     = 1e5             # Ignore changes in S abundances for sub-cycling tolerance, if T > T_IGNORE_SULFUR
const xmin::Float64                = 1e-6            # Minimum abundance fraction below which we don't care about the 10% rule for primordial species 
const xmax::Float64                = 1 + xmin        # Abundances should add up to 1, to within a margin xmin 
const tol_DustDensity::Float64     = tol_Chem        # Maximum relative change tolerance per sub-cycle time-step for the dust density 
const max_iter::Int                = 100             # Maximum number of adaptive refinements of sub-cycle time-step 
const kB_T_floor::Float64          = kB * T_floor    # Convenient constant 
const kB_T_ceil::Float64           = kB * T_ceiling  # Convenient constant 
const uth_DustGas::Float64         = 0.01 * kms      # Minimum change in gas & dust velocity below which we don't care about the 10% rule 
const tol_DustGasCoupling::Float64 = 1e-4            # Dust-gas coupling update considered converged if the relative change in ud & u is below this 

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
                                            c̃::Float64,              # Reduced speed of light for stellar/IR RT
                                        e_old::Float64,              # Gas internal energy density from previous sub-step 
                                        ρ_old::Float64,              # Gas density 
                                         Lsob::Float64,              # ρ / ∇ρ
                                      xH2_old::Float64,              # H₂ fraction from previous sub-step 
                                     xHII_old::Float64,              # H II fraction from previous sub-step 
                                      xHD_old::Float64,              # HD fraction from previous sub-step 
                                     xDII_old::Float64,              # D II fraction from previous sub-step 
                                    xHeII_old::Float64,              # He II fraction from previous sub-step 
                                   xHeIII_old::Float64,              # He III fraction from previous sub-step 
                                      xCI_old::Float64,              # C I fraction from previous sub-step 
                                     xCII_old::Float64,              # C II fraction from previous sub-step 
                                    xCIII_old::Float64,              # C III fraction from previous sub-step 
                                      xCO_old::Float64,              # CO fraction from previous sub-step (nCO/nC)
                                      xOI_old::Float64,              # O I fraction from previous sub-step 
                                     xOII_old::Float64,              # O II fraction from previous sub-step 
                                    xOIII_old::Float64,              # O III fraction from previous sub-step 
                                     xOIV_old::Float64,              # O IV fraction from previous sub-step
                                     xSII_old::Float64,              # S II fraction from previous sub-step 
                                    xSIII_old::Float64,              # S III fraction from previous sub-step 
                                     xSIV_old::Float64,              # S IV fraction from previous sub-step
                                       ZC_old::Float64,  # Gas-phase C metallicity from previous sub-step
                                     ZSil_old::Float64,  # Gas-phase Si metallicity from previous sub-step
                                 Zgasrest_old::Float64,  # Gas-phase 'rest' (non-C/Si) metallicity from previous sub-step 
                                       ρd_old::NTuple{3, Float64},       # Dust density (ρd_Sil, ρd_C, ρd_PAH)
                                   ρdud_r_old::NTuple{3, Float64},       # Dust r-momentum (ρdud_r_Sil, ρdud_r_C, ρdud_r_PAH)
                                   ρdud_Z_old::NTuple{3, Float64},       # Dust Z-momentum (ρdud_Z_Sil, ρdud_Z_C, ρdud_Z_PAH)
                                     ρu_r_old::Float64,                  # Gas r-momentum 
                                     ρu_Z_old::Float64,                  # Gas Z-momentum 
                                         Dust::DustState, 
                                         Trad::Float64,
                                           Td::NTuple{3, Float64},       # Dust temperatures from last substep
                                      Zgr_old::NTuple{3, Float64},       # Dust grain charges from last substep 
                                          Lyα::Union{LyαArrays,Nothing}, # Lyα state, if used 

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
                                          OTS::Bool    = true,   # On-the-spot approximation
                                  RadPressure::Bool    = true,   # Radiation pressure feedback on gas & dust (on if true, off if false)
                                        L_box::Float64 = ΔL_max, # Box size, to limit column densities 
                               CoolingHeating::Bool    = true,   # Whether heating/cooling is on/off 
                        DustGrowthDestruction::Bool    = true,   # Dust growth & destruction (on if true, off if false),
                            HydrogenAndHelium::Bool    = true,   # Whether to use hydrogen & helium chemistry, or just hydrogen chemistry (for some tests)
                              StellarAverages::StellarBandAverages = DefaultStellarBandAverages
                                      )

    # Safety checks:
    if DEBUG_PhotoThermoChemistry
        @assert isfinite(ρ_old) && ρ_old > 0.0 "ρ_old = $(ρ_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(e_old)                "e_old = $(e_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"

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

        @assert isfinite(xH2_old)                                "xH2_old = $(xH2_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xHII_old)                               "xHII_old = $(xHII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xHD_old)                                "xHD_old = $(xHD_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xDII_old)                               "xDII_old = $(xDII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xHeII_old)                              "xHeII_old = $(xHeII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xHeIII_old)                             "xHeIII_old = $(xHeIII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xCI_old)                                "xCI_old = $(xCI_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xCII_old)                               "xCII_old = $(xCII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xCIII_old)                              "xCIII_old = $(xCIII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xCO_old)                                "xCO_old = $(xCO_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xOI_old)                                "xOI_old = $(xOI_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xOII_old)                               "xOII_old = $(xOII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xOIII_old)                              "xOIII_old = $(xOIII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xOIV_old)                               "xOIV_old = $(xOIV_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xSII_old)                               "xSII_old = $(xSII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xSIII_old)                              "xSIII_old = $(xSIII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xSIV_old)                               "xSIV_old = $(xSIV_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xHII_old + 2 * xH2_old ≤ xmax                    "xHII_old + 2 * xH2_old = $(xHII_old + 2 * xH2_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xDII_old + xHD_old ≤ xmax                        "xDII_old + xHD_old = $(xDII_old + xHD_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xHeII_old + xHeIII_old ≤ xmax                    "xHeII_old + xHeIII_old = $(xHeII_old + xHeIII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xCI_old + xCII_old + xCIII_old + xCO_old ≤ xmax  "xCI_old + xCII_old + xCO_old = $(xCI_old + xCII_old + xCIII_old + xCO_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xOI_old + xOII_old + xOIII_old + xOIV_old ≤ xmax "xOI_old + xOII_old + xOIII_old + xOIV_old = $(xOI_old + xOII_old + xOIII_old + xOIV_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xSII_old + xSIII_old + xSIV_old ≤ xmax           "xSII_old + xSIII_old + xSIV_old = $(xSII_old + xSIII_old + xSIV_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
    end

    # --------- Safety clamp --------------------------------------------------------------------
    # Clamp gas density to be ≥ floor:
    ρ_old             =   max( ρ_old, ρ_floor )

    # Clamp dust bin densities to be ≥ floor:
    ρd_Sil_old,  ρd_C_old,  ρd_PAH_old  =  ρd_old

    ρd_Sil_old        =   max( ρd_Sil_old, ρd_floor )
    ρd_C_old          =   max( ρd_C_old,   ρd_floor )
    ρd_PAH_old        =   max( ρd_PAH_old, ρd_floor )

    # Repack clamped ρd_old in a tuple for use below:
    ρd_old            =   ρd_Sil_old,  ρd_C_old,  ρd_PAH_old

    # Useful below to skip certain updates in ~ dust-free conditions:
    ρd_max_old        =   max( ρd_Sil_old, ρd_C_old, ρd_PAH_old )

    # --------- Get gas-phase metal density -----------------------------------------------------
    ρC_old            =   @fastmath ZC_old       * ρ_old
    ρSil_old          =   @fastmath ZSil_old     * ρ_old
    ρZrest_old        =   @fastmath Zgasrest_old * ρ_old
    ρd_th_subcycling  =   @fastmath DG_th * ρ_old  

    # --------- Reservoirs for gas-phase accretion onto dust ------------------------------------
    ρdplusZ_Sil       =   @fastmath ρd_Sil_old  +  ρSil_old    # For accretion of Si onto silicate grains 
    ρdplusZ_C         =   @fastmath ρd_C_old    +  ρC_old      # For accretion of C onto graphite grains 

    # NOTE: PAH accretion is done below, *after* graphite grains have accreted some carbon, 
    # and so we can't write its reservoir here.
    # --------- Get dust & gas velocities -------------------------------------------------------
    inv_ρ             =   @fastmath 1 / ρ_old
    u_r_old           =   @fastmath ρu_r_old * inv_ρ
    u_Z_old           =   @fastmath ρu_Z_old * inv_ρ

    ρdud_r_Sil_old,  ρdud_r_C_old,  ρdud_r_PAH_old  =  ρdud_r_old
    ρdud_Z_Sil_old,  ρdud_Z_C_old,  ρdud_Z_PAH_old  =  ρdud_Z_old

    # Per-bin velocity unpacking:
    if ρd_Sil_old > ρd_IGNORE  &&  ρd_Sil_old > ρd_th_subcycling
        inv_ρd_Sil     = @fastmath 1 / ρd_Sil_old
        ud_r_Sil_old   = @fastmath ρdud_r_Sil_old * inv_ρd_Sil
        ud_Z_Sil_old   = @fastmath ρdud_Z_Sil_old * inv_ρd_Sil
    else
        # Negligible dust bin density - assume it follows the gas:
        ud_r_Sil_old   = u_r_old
        ud_Z_Sil_old   = u_Z_old

        # Adjust 'old' dust momenta accordingly:
        ρdud_r_Sil_old = @fastmath ρd_Sil_old * u_r_old
        ρdud_Z_Sil_old = @fastmath ρd_Sil_old * u_Z_old
    end

    if ρd_C_old > ρd_IGNORE  &&  ρd_C_old > ρd_th_subcycling
        inv_ρd_C     = @fastmath 1 / ρd_C_old
        ud_r_C_old   = @fastmath ρdud_r_C_old * inv_ρd_C
        ud_Z_C_old   = @fastmath ρdud_Z_C_old * inv_ρd_C
    else
        # Negligible dust bin density - assume it follows the gas:
        ud_r_C_old   = u_r_old
        ud_Z_C_old   = u_Z_old

        # Adjust 'old' dust momenta accordingly:
        ρdud_r_C_old = @fastmath ρd_C_old * u_r_old
        ρdud_Z_C_old = @fastmath ρd_C_old * u_Z_old
    end

    if ρd_PAH_old > ρd_IGNORE  &&  ρd_PAH_old > ρd_th_subcycling
        inv_ρd_PAH     = @fastmath 1 / ρd_PAH_old
        ud_r_PAH_old   = @fastmath ρdud_r_PAH_old * inv_ρd_PAH
        ud_Z_PAH_old   = @fastmath ρdud_Z_PAH_old * inv_ρd_PAH
    else
        # Negligible dust bin density - assume it follows the gas:
        ud_r_PAH_old   = u_r_old
        ud_Z_PAH_old   = u_Z_old

        # Adjust 'old' dust momenta accordingly:
        ρdud_r_PAH_old = @fastmath ρd_PAH_old * u_r_old
        ρdud_Z_PAH_old = @fastmath ρd_PAH_old * u_Z_old
    end

    # --------- Get relevant number densities & abundance fractions  ----------------------------
    nHI_old, nHII_old, nH2_old, nHeI_old, nHeII_old, 
    nHeIII_old, nDI_old, nDII_old, nHD_old, 
    nCI_old, nCII_old, nCIII_old, nCO_old, nCHx_old, 
    nOI_old, nOII_old, nOIII_old, nOIV_old, nOHx_old, 
    nSI_old, nSII_old, nSIII_old, nSIV_old,
    ne_old, n_old, nH_old, nHe_old, nD_old, 
    nC_old, nO_old, nS_old = GetAllNumberDensities( xH2_old, xHII_old, 
                                            xHeII_old, xHeIII_old, 
                                            xHD_old, xDII_old,
                                            xCI_old, xCII_old, xCIII_old, xCO_old,
                                            xOI_old, xOII_old, xOIII_old, xOIV_old,
                                            xSII_old, xSIII_old, xSIV_old,
                                            ρ_old, ZC_old, ZSil_old,
                                            Zgasrest_old, HydrogenAndHelium )
                               
    # --------- Get adiabatic index for old state ---------------------------------------------
    γad               =   Getγad( xHII_old, xH2_old, xHeII_old, xHeIII_old, HydrogenAndHelium )

    # --------- Convenient constants (avoid re-computing) -------------------------------------
    inv_xH2_old       =   @fastmath 1  / ( xH2_old   +  xmin )             # Used in 10% rule check 
    inv_xHII_old      =   @fastmath 1  / ( xHII_old  +  xmin )             # Used in 10% rule check 
    inv_xCO_old       =   @fastmath 1  / ( xCO_old   +  xmin )             # Used in 10% rule check
    inv_xCII_old      =   @fastmath 1  / ( xCII_old  +  xmin )             # Used in 10% rule check
    inv_xCIII_old     =   @fastmath 1  / ( xCIII_old +  xmin )             # Used in 10% rule check
    inv_xOII_old      =   @fastmath 1  / ( xOII_old  +  xmin )             # Used in 10% rule check
    inv_xOIII_old     =   @fastmath 1  / ( xOIII_old +  xmin )             # Used in 10% rule check
    inv_xOIV_old      =   @fastmath 1  / ( xOIV_old  +  xmin )             # Used in 10% rule check
    inv_xSII_old      =   @fastmath 1  / ( xSII_old  +  xmin )             # Used in 10% rule check
    inv_xSIII_old     =   @fastmath 1  / ( xSIII_old +  xmin )             # Used in 10% rule check
    inv_xSIV_old      =   @fastmath 1  / ( xSIV_old  +  xmin )             # Used in 10% rule check
    inv_ne_old        =   @fastmath 1  / ( ne_old    +  xmin * nH_old )    # Used in 10% rule check
    inv_γ_minus_1     =   @fastmath 1  / ( γad - 1 )
    e_floor           =   @fastmath n_old * kB_T_floor * inv_γ_minus_1     # Used in limiting updated e
    e_ceiling         =   @fastmath n_old * kB_T_ceil  * inv_γ_minus_1     # Used in limiting updated e

    # --------- Clamp e_old and get T_old, and TCMB ---------------------------------------------
    e_old             =   min( max( e_old,  e_floor ), e_ceiling )
    T_old             =   @fastmath ( γad - 1 ) * e_old  / ( n_old * kB )
    TCMB              =   @fastmath TCMB0 * (1 + z)

    # --------- Lyα opacities & emissivity (if Lyα RT is simulated) ---------------------------
    if !isnothing(Lyα)
        e_Lyα_old = PrepareLyαSubCycling!( Lyα, Dust, ρd_old, icell, jcell, T_old, 
                                           nHI_old, nH2_old, nHII_old, nHeI_old, ne_old, 
                                           nOHx_old, nCHx_old, z, OTS )
    else
        # Lyα energy density is zero if not evolved:
        e_Lyα_old = 0.0 
    end

    # --------- Get initial estimate of dust temperatures --------------------------------------
    #Td_Sil = ComputeTd( J_IR_old, J_opt_old, J_FUV_old, J_LW_old, J_EUV1_old,
    #                    J_EUV2_old, J_EUV3_old, Dust, Sil, TCMB,
    #                    Trad, T_old, nH_old, e_Lyα_old )
                    
    #Td_C   = ComputeTd( J_IR_old, J_opt_old, J_FUV_old, J_LW_old, J_EUV1_old,
    #                    J_EUV2_old, J_EUV3_old, Dust, Gra, TCMB,
    #                    Trad, T_old, nH_old, e_Lyα_old )

    #Td_PAH = ComputeTd( J_IR_old, J_opt_old, J_FUV_old, J_LW_old, J_EUV1_old,
    #                    J_EUV2_old, J_EUV3_old, Dust, PAH, TCMB,
    #                    Trad, T_old, nH_old, e_Lyα_old )

    # Pack all dust temperatures in a tuple for HeatingCoolingStep etc.
    #Td     = ( Td_Sil, Td_C, Td_PAH )  

    # --------- Unpack grain charges from last substep ----------------------------------------
    Zgr_Sil_old, Zgr_C_old, Zgr_PAH_old = Zgr_old

    # --------- Recombination photons (if OTS is not used) ------------------------------------
    if !OTS 
        js_rec_EUV1, js_rec_EUV2, js_rec_EUV3 = EUVRecombinationEmissivity( T_old, ne_old, nHII_old, nHeII_old, nHeIII_old, StellarAverages )

        # Add emissivities to calculation of dJ/dt in the EUV1, EUV2, EUV3 bands:
        dJ_EUV1dt += @fastmath js_rec_EUV1 * c̃
        dJ_EUV2dt += @fastmath js_rec_EUV2 * c̃
        dJ_EUV3dt += @fastmath js_rec_EUV3 * c̃
    end

    # -----------------------------------------------------------------------------------------

    # Sobolev-type column length estimate ~ cell width + density variation scale (following 
    # e.g. Hopkins+ 2023, for FIRE-3). We limit ΔL to ≤ min( ΔL_max, L_box)
    # to avoid unphysical values if ∇ρ ~ 0:
    ΔL   = min( min( Δr, ΔZ )  +  Lsob,  ΔL_max, L_box )

    # Estimated H₂, HD, and CO column densities:
    N_H2 = @fastmath nH2_old * ΔL 
    N_HD = @fastmath nHD_old * ΔL 
    N_CO = @fastmath nCO_old * ΔL 

    # Get self-shielding factors:
    SelfShielding            = S_H2( N_H2, T_old ) 
    SelfShielding_HD         = S_HD( N_HD, T_old )
    SelfShielding_H2_HD      = S_H2_HD( N_H2 )
    
    Δtsub                    = Δtsubguess # Initialize first guess of sub-cycle time-step
    iter::Int                = 0          # Initialize iteration count
    ΔxH2_rel                 = 0.0        # Initialize ΔxH2_rel, for printing in case convergence fails 
    ΔxHII_rel                = 0.0        # Initialize ΔxHII_rel, for printing in case convergence fails 
    ΔxCO_rel                 = 0.0        # Initialize ΔxCO_rel, for printing in case convergence fails 
    ΔxCII_rel                = 0.0        # Initialize ΔxCII_rel, for printing in case convergence fails 
    ΔxCIII_rel               = 0.0        # Initialize ΔxCIII_rel, for printing in case convergence fails 
    ΔxOII_rel                = 0.0        # Initialize ΔxOII_rel, for printing in case convergence fails 
    ΔxOIII_rel               = 0.0        # Initialize ΔxOIII_rel, for printing in case convergence fails 
    ΔxOIV_rel                = 0.0        # Initialize ΔxOIV_rel, for printing in case convergence fails
    ΔxSII_rel                = 0.0        # Initialize ΔxSII_rel, for printing in case convergence fails 
    ΔxSIII_rel               = 0.0        # Initialize ΔxSIII_rel, for printing in case convergence fails 
    ΔxSIV_rel                = 0.0        # Initialize ΔxSIV_rel, for printing in case convergence fails
    Δne_rel                  = 0.0        # Initialize Δne_rel, for printing in case convergence fails 
    Δe                       = 0.0        # Initialize Δe, for printing in case convergence fails 
    ΔStellarRT               = 0.0        # Initialize ΔStellarRT, for printing in case convergence fails 
    Δρd_rel                  = 0.0        # Initialize Δρd_rel, for printing in case convergence fails 
    REDO_ABSORPTION          = false
    REDO_HEATINGCOOLING      = false
    N_REDO_ABSORPTION        = 0
    N_REDO_HEATINGCOOLING    = 0
    N_REDO_DUSTGASCOUPLING   = 0
    N_REDO_DUSTGROWTHDESTR   = 0
    N_REDO_CHEMISTRY         = 0 
    while iter < max_iter 
        iter            += 1
        Δtsub_old        = Δtsub                  # Store current substep (useful for suggesting a new one based on changes in quantities)
        Δtsub_half       = @fastmath 0.5 * Δtsub
        Δtsub_newmin     = @fastmath 0.1 * Δtsub  # If changes in quantities are very large, we retry with Δtsub ≥ Δtsub_newmin
                                                  # (i.e. a more aggressive lowering of subcycle step than simply halving)

        # Store initial old energy & temperature state in case we have to revert because of rejection:
        e_old_FirstTry   = e_old
        T_old_FirstTry   = T_old 

        # Diagnostics for radiation pressure:
        Δρu_r_trial      = 0.0  # Initialize the diagnostic: change in r-direction gas momentum from stellar radiation pressure 
        Δρu_Z_trial      = 0.0  # Initialize the diagnostic: change in Z-direction gas momentum from stellar radiation pressure 

        Δρu_r_IR_trial   = 0.0  # Initialize the diagnostic: change in r-direction gas momentum from IR radiation pressure 
        Δρu_Z_IR_trial   = 0.0  # Initialize the diagnostic: change in Z-direction gas momentum from IR radiation pressure 

        Δρu_r_Lyα_trial  = 0.0  # Initialize the diagnostic: change in r-direction gas + dust momentum from Lyα radiation pressure
        Δρu_Z_Lyα_trial  = 0.0  # Initialize the diagnostic: change in Z-direction gas + dust momentum from Lyα radiation pressure

        # Radiation pressure to be dumped into gas & dust:
        Δρu_r_gas_trial  = 0.0
        Δρu_Z_gas_trial  = 0.0

        Δρu_r_Sil_trial  = 0.0;  Δρu_Z_Sil_trial  = 0.0
        Δρu_r_C_trial    = 0.0;  Δρu_Z_C_trial    = 0.0
        Δρu_r_PAH_trial  = 0.0;  Δρu_Z_PAH_trial  = 0.0

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
        Δρu_r_rad_gas_trial_Abs,  Δρu_Z_rad_gas_trial_Abs,   # Momentum to be injected into gas
        Δρu_r_rad_Sil_trial_Abs,  Δρu_Z_rad_Sil_trial_Abs,   # Momentum to be injected into silicate dust 
        Δρu_r_rad_C_trial_Abs,    Δρu_Z_rad_C_trial_Abs,     # Momentum to be injected into graphite dust
        Δρu_r_rad_PAH_trial_Abs,  Δρu_Z_rad_PAH_trial_Abs,   # Momentum to be injected into PAH dust
        ΔStellarRT, js_IR, Td_Sil_new, Td_C_new, Td_PAH_new, 
        Trad_new,
        e_Lyα_new, Δρu_r_Lyα_trial_Abs, Δρu_Z_Lyα_trial_Abs = AbsorptionStep( Δtsub, icell, jcell, c̃, nH_old, nHI_old, nH2_old, nHeI_old, nHeII_old, 
                                                                              nDI_old, nHD_old, nCI_old, nCII_old, nCHx_old, nCO_old, 
                                                                              nOI_old, nOII_old, nOIII_old, nOHx_old, 
                                                                              nSI_old, nSII_old, nSIII_old, ne_old,
                                                                              ρd_old, Dust, Lyα, T_old, Td, Trad, TCMB,
                                                                              J_opt_old,  dJ_optdt,   H_r_opt_old,  H_Z_opt_old,  dHr_optdt,  dHZ_optdt,
                                                                              J_FUV_old,  dJ_FUVdt,   H_r_FUV_old,  H_Z_FUV_old,  dHr_FUVdt,  dHZ_FUVdt,
                                                                              J_LW_old,   dJ_LWdt,    H_r_LW_old,   H_Z_LW_old,   dHr_LWdt,   dHZ_LWdt,
                                                                              J_EUV1_old, dJ_EUV1dt,  H_r_EUV1_old, H_Z_EUV1_old, dHr_EUV1dt, dHZ_EUV1dt,
                                                                              J_EUV2_old, dJ_EUV2dt,  H_r_EUV2_old, H_Z_EUV2_old, dHr_EUV2dt, dHZ_EUV2dt,
                                                                              J_EUV3_old, dJ_EUV3dt,  H_r_EUV3_old, H_Z_EUV3_old, dHr_EUV3dt, dHZ_EUV3dt,
                                                                              J_IR_old,   H_r_IR_old, H_Z_IR_old,
                                                                              StellarAverages )

        # Pack updated dust bin temperatures in a tuple for use later:
        Td_new = ( Td_Sil_new, Td_C_new, Td_PAH_new )

        if REDO_ABSORPTION
            # Change in J or H were too large, so reject update and try a smaller sub-cycle time-step: 
            Δtsub = Δtsub_half # Preliminary suggeted new substep 

            if isfinite( ΔStellarRT )
                # If the change was large but finite, use it to infer a 
                # better retry substep to avoid wasting many retries:
                Δtsub = max( min( Δtsub,  tol_Abs * Δtsub_old / ΔStellarRT ), Δtsub_newmin  )
            end

            # Restore Lyα arrays J_Lyα and H_Lyα:
            RestoreLyαRejection( icell, jcell, Lyα )

            # Update rejection count:
            N_REDO_ABSORPTION += 1

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

            # Also add stellar rad. pressure momentum change (diagnostic):
            Δρu_r_trial      += Δρu_r_trial_Abs
            Δρu_Z_trial      += Δρu_Z_trial_Abs

            # And IR rad. pressure momentum change (diagnostic):
            Δρu_r_IR_trial   += Δρu_r_IR_trial_Abs
            Δρu_Z_IR_trial   += Δρu_Z_IR_trial_Abs

            # Add Lyα rad. pressure momentum change (diagnostic):
            Δρu_r_Lyα_trial  += Δρu_r_Lyα_trial_Abs
            Δρu_Z_Lyα_trial  += Δρu_Z_Lyα_trial_Abs

            # Radiation pressure momentum to be dumped into gas:
            @fastmath Δρu_r_gas_trial  += RadPressure * Δρu_r_rad_gas_trial_Abs
            @fastmath Δρu_Z_gas_trial  += RadPressure * Δρu_Z_rad_gas_trial_Abs

            # Radiation pressure momentum to be dumped into dust:
            @fastmath Δρu_r_Sil_trial  += RadPressure * Δρu_r_rad_Sil_trial_Abs
            @fastmath Δρu_Z_Sil_trial  += RadPressure * Δρu_Z_rad_Sil_trial_Abs
            @fastmath Δρu_r_C_trial    += RadPressure * Δρu_r_rad_C_trial_Abs
            @fastmath Δρu_Z_C_trial    += RadPressure * Δρu_Z_rad_C_trial_Abs
            @fastmath Δρu_r_PAH_trial  += RadPressure * Δρu_r_rad_PAH_trial_Abs
            @fastmath Δρu_Z_PAH_trial  += RadPressure * Δρu_Z_rad_PAH_trial_Abs
        end

        # Get photo-ionization/dissociation rates (s⁻¹), for use in chemistry 
        # and heating below:
        Γ_LW        =  PhotoDissRate_H2( SelfShielding * J_LW, StellarAverages )
        Γpd         =  PhotoDetachRate_Hminus( J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, e_Lyα_new, StellarAverages )
        Γ_LW_HD     =  PhotoDissRate_HD( SelfShielding_HD * SelfShielding_H2_HD * J_LW, StellarAverages )
        ΓionH2      =  IonRate_H2( J_EUV1, J_EUV2, J_EUV3, StellarAverages ) 
        ΓionHI      =  IonRate_HI( J_EUV1, J_EUV2, J_EUV3, StellarAverages ) 
        ΓionDI      =  ΓionHI
        ΓionHeI     =  IonRate_HeI( J_EUV2, J_EUV3, StellarAverages ) 
        ΓionHeII    =  IonRate_HeII( J_EUV3, StellarAverages )
        ΓionCI      =  IonRate_CI( J_LW, J_EUV1, J_EUV2, J_EUV3, StellarAverages )
        ΓionCII     =  IonRate_CII( J_EUV1, J_EUV2, J_EUV3, StellarAverages )
        ΓionOI      =  IonRate_OI( J_EUV1, J_EUV2, J_EUV3, StellarAverages )
        ΓionOII     =  IonRate_OII( J_EUV2, J_EUV3, StellarAverages )
        ΓionOIII    =  IonRate_OIII( J_EUV3, StellarAverages )
        ΓdissCHx    =  PhotoDissRate_CHₓ( J_opt, J_FUV, J_LW, e_Lyα_new, StellarAverages )
        ΓdissOHx    =  PhotoDissRate_OHₓ( J_FUV, J_LW, e_Lyα_new, StellarAverages )
        ΓdissCO     =  PhotoDissRate_CO( J_LW, StellarAverages )
        ΓionSI      =  IonRate_SI( J_LW, J_FUV, J_EUV1, J_EUV2, J_EUV3, StellarAverages )
        ΓionSII     =  IonRate_SII( J_EUV1, J_EUV2, J_EUV3, StellarAverages )
        ΓionSIII    =  IonRate_SIII( J_EUV2, J_EUV3, StellarAverages )

        #######################################################################
        #######                                                         #######
        #######     S T E P  2  :  D U S T - G A S  C O U P L I N G     #######
        #######                                                         #######
        #######################################################################

        # Next we update the gas & dust momenta from radiation pressure + drag, 
        # over a substep Δtsub.

        # Get grain charges, if there is dust around:
        if ρd_Sil_old ≤ ρd_IGNORE  ||  ρd_Sil_old ≤ ρd_th_subcycling
            Zgr_Sil = Zgr_Sil_old
        else 
            Zgr_Sil = FindGrainCharge( T_old, nHII_old, nHeII_old, nHeIII_old, 
                                       nDII_old, nCII_old, nCIII_old, nOII_old, nOIII_old, nOIV_old,
                                       nSII_old, nSIII_old, nSIV_old, ne_old,
                                       J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, Sil, e_Lyα_new,
                                       Zgr_Sil_old )
        end

        if ρd_C_old ≤ ρd_IGNORE  ||  ρd_C_old ≤ ρd_th_subcycling
            Zgr_C   = Zgr_C_old
        else
            Zgr_C   = FindGrainCharge( T_old, nHII_old, nHeII_old, nHeIII_old, 
                                       nDII_old, nCII_old, nCIII_old, nOII_old, nOIII_old, nOIV_old,
                                       nSII_old, nSIII_old, nSIV_old, ne_old,
                                       J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, Gra, e_Lyα_new,
                                       Zgr_C_old )
        end

        if ρd_PAH_old ≤ ρd_IGNORE  ||  ρd_PAH_old ≤ ρd_th_subcycling
            Zgr_PAH = Zgr_PAH_old
        else
            Zgr_PAH = FindGrainCharge( T_old, nHII_old, nHeII_old, nHeIII_old, 
                                       nDII_old, nCII_old, nCIII_old, nOII_old, nOIII_old, nOIV_old, 
                                       nSII_old, nSIII_old, nSIV_old, ne_old,
                                       J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, PAH, e_Lyα_new,
                                       Zgr_PAH_old )
        end

        # Initialize updated dust & gas velocities (for tracking convergence):
        u_r_new         = u_r_old
        u_Z_new         = u_Z_old

        ud_r_Sil_new    = ud_r_Sil_old;  ud_Z_Sil_new = ud_Z_Sil_old
        ud_r_C_new      = ud_r_C_old;    ud_Z_C_new   = ud_Z_C_old
        ud_r_PAH_new    = ud_r_PAH_old;  ud_Z_PAH_new = ud_Z_PAH_old

        # Initialize updated dust & gas momenta:
        ρu_r_new        = ρu_r_old
        ρu_Z_new        = ρu_Z_old

        ρdud_r_Sil_new  = ρdud_r_Sil_old;   ρdud_Z_Sil_new = ρdud_Z_Sil_old
        ρdud_r_C_new    = ρdud_r_C_old;     ρdud_Z_C_new   = ρdud_Z_C_old
        ρdud_r_PAH_new  = ρdud_r_PAH_old;   ρdud_Z_PAH_new = ρdud_Z_PAH_old

        if ρd_max_old ≤ ρd_IGNORE  ||  ρd_max_old ≤ ρd_th_subcycling
            # In case the dust has been effectively destroyed, ρd ≤ ρd_floor, we instead 
            # only apply radiation pressure on gas, and set dust velocities equal to 
            # gas velocities:
            ρu_r_new     = ρu_r_old  +  Δρu_r_gas_trial
            ρu_Z_new     = ρu_Z_old  +  Δρu_Z_gas_trial

            # Get updated velocities (needed for relative velocities later):
            u_r_new      = @fastmath ρu_r_new * inv_ρ
            u_Z_new      = @fastmath ρu_Z_new * inv_ρ

            ud_r_Sil_new = u_r_new;  ud_Z_Sil_new = u_Z_new
            ud_r_C_new   = u_r_new;  ud_Z_C_new   = u_Z_new
            ud_r_PAH_new = u_r_new;  ud_Z_PAH_new = u_Z_new

            # Updated dust momenta:
            ρdud_r_Sil_new = @fastmath ρd_Sil_old * ud_r_Sil_new
            ρdud_Z_Sil_new = @fastmath ρd_Sil_old * ud_Z_Sil_new

            ρdud_r_C_new   = @fastmath ρd_C_old   * ud_r_C_new
            ρdud_Z_C_new   = @fastmath ρd_C_old   * ud_Z_C_new

            ρdud_r_PAH_new = @fastmath ρd_PAH_old * ud_r_PAH_new
            ρdud_Z_PAH_new = @fastmath ρd_PAH_old * ud_Z_PAH_new

            # Set convergence to true, and exit the while loop:
            ConvergedDustGasCoupling = true 
        else
            # If there is dust in at least one bin, we update the momenta of 
            # gas & dust by solving the coupled radiation pressure + drag equations.

            # Iterate until updated velocities converge, and if not, redo subcycle 
            # step with smaller Δtsbub:
            ConvergedDustGasCoupling = false 
            iter_DustGasCoupling     = 0

            while ConvergedDustGasCoupling == false && iter_DustGasCoupling < 10
                # Compute first estimate of drag coefficients, using the best 
                # estimate of the dust & gas velocities at the new/updated time-step 
                # (to approximate a BDF1 update):
                Kdrag_Sil = ComputeDragCoefficientK( Zgr_Sil, ρd_Sil_old, T_old, Sil,
                                                     nHI_old, nH2_old, nHII_old, 
                                                     nHeI_old, nHeII_old, nHeIII_old,
                                                     nDII_old, nCII_old,  nCIII_old, 
                                                     nOII_old, nOIII_old, nOIV_old,
                                                     nSII_old, nSIII_old, nSIV_old,
                                                     ne_old, ud_r_Sil_new, u_r_new, ud_Z_Sil_new, u_Z_new )

                Kdrag_C   = ComputeDragCoefficientK( Zgr_C, ρd_C_old, T_old, Gra,
                                                     nHI_old, nH2_old, nHII_old, 
                                                     nHeI_old, nHeII_old, nHeIII_old, 
                                                     nDII_old, nCII_old, nCIII_old, 
                                                     nOII_old, nOIII_old, nOIV_old,
                                                     nSII_old, nSIII_old, nSIV_old,
                                                     ne_old, ud_r_C_new, u_r_new, ud_Z_C_new, u_Z_new )

                Kdrag_PAH = ComputeDragCoefficientK( Zgr_PAH, ρd_PAH_old, T_old, PAH,
                                                     nHI_old, nH2_old, nHII_old, 
                                                     nHeI_old, nHeII_old, nHeIII_old, 
                                                     nDII_old, nCII_old, nCIII_old, 
                                                     nOII_old, nOIII_old, nOIV_old,
                                                     nSII_old, nSIII_old, nSIV_old,
                                                     ne_old, ud_r_PAH_new, u_r_new, ud_Z_PAH_new, u_Z_new )

                # Get candidate update:
                ρu_r_new, ρu_Z_new,  
                ρdud_r_Sil_new, ρdud_Z_Sil_new,
                ρdud_r_C_new,   ρdud_Z_C_new,
                ρdud_r_PAH_new, ρdud_Z_PAH_new,

                u_r_candidate, u_Z_candidate,
                ud_r_Sil_candidate, ud_Z_Sil_candidate,
                ud_r_C_candidate,   ud_Z_C_candidate,
                ud_r_PAH_candidate, ud_Z_PAH_candidate  = UpdateGasDustMomenta( 
                                                          Δtsub, ρ_old, ρd_Sil_old, ρd_C_old, ρd_PAH_old,
                                                          Δρu_r_gas_trial, Δρu_Z_gas_trial, 
                                                          Δρu_r_Sil_trial, Δρu_Z_Sil_trial, 
                                                          Δρu_r_C_trial, Δρu_Z_C_trial,
                                                          Δρu_r_PAH_trial, Δρu_Z_PAH_trial,
                                                          ρu_r_old, ρu_Z_old, 
                                                          ρdud_r_Sil_old, ρdud_Z_Sil_old,
                                                          ρdud_r_C_old,   ρdud_Z_C_old,
                                                          ρdud_r_PAH_old, ρdud_Z_PAH_old,
                                                          Kdrag_Sil, Kdrag_C, Kdrag_PAH )
                                                                    
                # Check that the updated velocities have converged: 
                ConvergedDustGasCoupling = CheckConvergence_GasDustMomentumUpdate( 
                                           # Candidate new estimate of 'new'/updated values:
                                           ud_r_Sil_candidate, ud_Z_Sil_candidate,
                                           ud_r_C_candidate,   ud_Z_C_candidate,
                                           ud_r_PAH_candidate, ud_Z_PAH_candidate,
                                           u_r_candidate,      u_Z_candidate, 

                                           # Previous estimated 'new'/updated values:
                                           ud_r_Sil_new, ud_Z_Sil_new,
                                           ud_r_C_new,   ud_Z_C_new,
                                           ud_r_PAH_new, ud_Z_PAH_new,
                                           u_r_new, u_Z_new,  
                                                                    
                                           # Treshold below which we ignore changes:
                                           tol_DustGasCoupling, uth_DustGas )

                # Update the 'new' values of the velocities for the next convergence check: 
                ud_r_Sil_new = ud_r_Sil_candidate;  ud_Z_Sil_new = ud_Z_Sil_candidate
                ud_r_C_new   = ud_r_C_candidate;    ud_Z_C_new   = ud_Z_C_candidate
                ud_r_PAH_new = ud_r_PAH_candidate;  ud_Z_PAH_new = ud_Z_PAH_candidate

                u_r_new = u_r_candidate;  u_Z_new = u_Z_candidate

                # Update iteration count:
                iter_DustGasCoupling += 1                             

            end # End of while-loop / iteration 
        end

        if ConvergedDustGasCoupling == false 
            # Failed to converge, reduce Δtsub & retry:
            Δtsub = Δtsub_half

            # Restore Lyα arrays J_Lyα and H_Lyα:
            RestoreLyαRejection( icell, jcell, Lyα )

            # Update rejection count:
            N_REDO_DUSTGASCOUPLING += 1

            # Restore old state for e and T:
            e_old = e_old_FirstTry
            T_old = T_old_FirstTry

            continue    
        end

        ###################################################################################
        #######                                                                     #######
        #######     S T E P  3  :  D U S T  G R O W T H / D E S T R U C T I O N     #######
        #######                                                                     #######
        ###################################################################################

        # Next, update the dust & metal density, from considering dust growth (by accretion)
        # and destruction (by sublimation, thermal & non-thermal sputtering, 
        # chemical sputtering of graphite grains, and PAH photodissociation). 

        if DustGrowthDestruction  &&  ρd_max_old > ρd_IGNORE
            # Get dust-gas relative velocity magnitude:
            urel_Sil  = hypot( ud_r_Sil_new - u_r_new,  ud_Z_Sil_new - u_Z_new )
            urel_C    = hypot( ud_r_C_new   - u_r_new,  ud_Z_C_new   - u_Z_new )
            urel_PAH  = hypot( ud_r_PAH_new - u_r_new,  ud_Z_PAH_new - u_Z_new )

            # -------- UPDATE SILICATE DUST DENSITY --------------------------
            ρd_Sil_new   = DustGrowthDestruction_Update( ρd_Sil_old, ρdplusZ_Sil,
                                                         nH_old, nHI_old,
                                                         nHII_old, nHeII_old, nHeIII_old, 
                                                         nCII_old, nCIII_old,
                                                         T_old, Td_Sil_new,
                                                         urel_Sil, xCI_old, xCII_old, Zgr_Sil,
                                                         J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, e_Lyα_new,
                                                         Sil, Δtsub )

            # Floor silicate dust density: 
            ρd_Sil_new   = max( ρd_Sil_new, ρd_floor )

            # Recover updated Si gas density:
            ρSil_new     = max( ρdplusZ_Sil - ρd_Sil_new, 0.0 )

            # -------- UPDATE GRAPHITE DUST DENSITY --------------------------
            # Update graphite dust density:
            ρd_C_new     = DustGrowthDestruction_Update( ρd_C_old, ρdplusZ_C,
                                                         nH_old, nHI_old,
                                                         nHII_old, nHeII_old, nHeIII_old, 
                                                         nCII_old, nCIII_old,
                                                         T_old, Td_C_new,
                                                         urel_C, xCI_old, xCII_old, Zgr_C,
                                                         J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, e_Lyα_new,
                                                         Gra, Δtsub )

            # Floor graphite dust density: 
            ρd_C_new     = max( ρd_C_new, ρd_floor )

            # Recover updated C gas density (NOTE: before updating PAH, so this is 
            # a partial update):
            ρC_new       = max( ρdplusZ_C - ρd_C_new, 0.0 )

            # -------- UPDATE PAH DUST DENSITY -------------------------------
            # Reservoir for PAH accretion, evaluated after graphite accretion:
            ρdplusZ_PAH  = @fastmath ρd_PAH_old  +  ρC_new

            # Update the PAH dust density: 
            ρd_PAH_new   = DustGrowthDestruction_Update( ρd_PAH_old, ρdplusZ_PAH,
                                                         nH_old, nHI_old,
                                                         nHII_old, nHeII_old, nHeIII_old, 
                                                         nCII_old, nCIII_old,
                                                         T_old, Td_PAH_new,
                                                         urel_PAH, xCI_old, xCII_old, Zgr_PAH,
                                                         J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, e_Lyα_new,
                                                         PAH, Δtsub )

            # Floor PAH dust density: 
            ρd_PAH_new   = max( ρd_PAH_new, ρd_floor )

            # Recover updated C gas density (our final update):
            ρC_new       = max( ρdplusZ_PAH - ρd_PAH_new, 0.0 )

            # NOTE: Because we update graphite & PAH in sequential order, using the 
            # latest limited reservoirs, it follows that total carbon mass is conserved 
            # to machine precision (if both updates had used the old reservoirs, this would 
            # not be the case).

            # ------- UPDATE GAS DENSITY --------------------------------------
            # Updated gas density from loss or addition of C and Si:
            ρ_new      = max( ρ_old + (ρC_new - ρC_old) + (ρSil_new - ρSil_old), 
                              ρ_floor )

            # ------- UPDATE GAS PHASE METALLICITY ----------------------------
            inv_ρ_new  = @fastmath 1 / ρ_new
            ZC_new     = @fastmath ρC_new   * inv_ρ_new 
            ZSil_new   = @fastmath ρSil_new * inv_ρ_new 

            # For the 'rest' of the metals (non-C/Si), that density remains 
            # unaffected, so the new 'rest' metallicity is updated as follows:
            Zgasrest_new = @fastmath ρZrest_old * inv_ρ_new 

            # ------- CHECK CHANGE IN DUST DENSITY ----------------------------
            # Next we check the change in the dust bin densities, and if they are 
            # too large we redo the sub-step:

            Δρd_Sil_rel = abs( ρd_Sil_new - ρd_Sil_old ) / max( ρd_Sil_old, ρd_th_subcycling, ρd_floor )
            Δρd_C_rel   = abs( ρd_C_new   - ρd_C_old )   / max( ρd_C_old,   ρd_th_subcycling, ρd_floor )
            Δρd_PAH_rel = abs( ρd_PAH_new - ρd_PAH_old ) / max( ρd_PAH_old, ρd_th_subcycling, ρd_floor )
            Δρd_rel     = max( Δρd_Sil_rel,  Δρd_C_rel,  Δρd_PAH_rel )

            if Δρd_rel > tol_DustDensity  ||  !isfinite(Δρd_rel)
                # Change in dust density was too high, reduce Δtsub & retry:
                Δtsub = Δtsub_half  # Preliminary suggested new substep 

                # Use change in dust densities to infer a better new substep:
                if isfinite(Δρd_rel)
                    Δtsub = max( min( Δtsub, tol_DustDensity * Δtsub_old / Δρd_rel ), Δtsub_newmin )
                end

                # Restore Lyα arrays J_Lyα and H_Lyα:
                RestoreLyαRejection( icell, jcell, Lyα )

                # Update rejection count:
                N_REDO_DUSTGROWTHDESTR += 1 

                # Restore old state for e and T:
                e_old = e_old_FirstTry
                T_old = T_old_FirstTry

                continue    
            end

        else
            # If dust growth/destruction is turned off, return old values:
            ρ_new, ρd_Sil_new, ρd_C_new, ρd_PAH_new  = ρ_old, ρd_Sil_old, ρd_C_old, ρd_PAH_old 

            # No relative change:
            Δρd_rel = 0.0

            # New metallicities = old metallicities:
            ZC_new       = ZC_old
            ZSil_new     = ZSil_old
            Zgasrest_new = Zgasrest_old
        end

        # Get updated number densities: 
        nHI_new, nHII_new, nH2_new, nHeI_new, nHeII_new, 
        nHeIII_new, nDI_new, nDII_new, nHD_new, 
        nCI_new, nCII_new, nCIII_new, nCO_new,  nCHx_new, 
        nOI_new, nOII_new, nOIII_new, nOIV_new, nOHx_new, 
        nSI_new, nSII_new, nSIII_new, nSIV_new,
        ne_new, n_new, nH_new, nHe_new, nD_new, 
        nC_new, nO_new, nS_new = GetAllNumberDensities( xH2_old, xHII_old, 
                                                        xHeII_old, xHeIII_old, 
                                                        xHD_old, xDII_old,
                                                        xCI_old, xCII_old, xCIII_old, xCO_old,
                                                        xOI_old, xOII_old, xOIII_old, xOIV_old,
                                                        xSII_old, xSIII_old, xSIV_old,
                                                        ρ_new, ZC_new, ZSil_new, 
                                                        Zgasrest_new, HydrogenAndHelium )

        # Update gas internal energy density accordingly, keeping temperature constant:
        @fastmath ScaleGasDensity   = ρ_new / ρ_old
        @fastmath e_old            *= ScaleGasDensity 

        # Similarly, update gas & dust momenta in a way that keeps velocities unchanged: 
        @fastmath ρu_r_new         *= ScaleGasDensity
        @fastmath ρu_Z_new         *= ScaleGasDensity

        # Silicate dust:
        if ρd_Sil_new > ρd_IGNORE  &&  ρd_Sil_new > DG_th * ρ_new
            @fastmath ScaleDustDensity   = ρd_Sil_new / ρd_Sil_old
            @fastmath ρdud_r_Sil_new    *= ScaleDustDensity
            @fastmath ρdud_Z_Sil_new    *= ScaleDustDensity
        else
            # Otherwise set silicate dust velocities equal to gas velocities:
            @fastmath DustToGas         = ρd_Sil_new / ρ_new
            @fastmath ρdud_r_Sil_new    = ρu_r_new * DustToGas
            @fastmath ρdud_Z_Sil_new    = ρu_Z_new * DustToGas
        end

        # Graphite dust:
        if ρd_C_new > ρd_IGNORE  &&  ρd_C_new > DG_th * ρ_new
            @fastmath ScaleDustDensity   = ρd_C_new / ρd_C_old
            @fastmath ρdud_r_C_new      *= ScaleDustDensity
            @fastmath ρdud_Z_C_new      *= ScaleDustDensity
        else
            # Otherwise set silicate dust velocities equal to gas velocities:
            @fastmath DustToGas         = ρd_C_new / ρ_new
            @fastmath ρdud_r_C_new      = ρu_r_new * DustToGas
            @fastmath ρdud_Z_C_new      = ρu_Z_new * DustToGas
        end

        # PAH dust:
        if ρd_PAH_new > ρd_IGNORE  &&  ρd_PAH_new > DG_th * ρ_new
            @fastmath ScaleDustDensity   = ρd_PAH_new / ρd_PAH_old
            @fastmath ρdud_r_PAH_new    *= ScaleDustDensity
            @fastmath ρdud_Z_PAH_new    *= ScaleDustDensity
        else
            # Otherwise set silicate dust velocities equal to gas velocities:
            @fastmath DustToGas         = ρd_PAH_new / ρ_new
            @fastmath ρdud_r_PAH_new    = ρu_r_new * DustToGas
            @fastmath ρdud_Z_PAH_new    = ρu_Z_new * DustToGas
        end

        # Pack updated dust momenta into a tuple for easy return below:
        ρdud_r_new = ( ρdud_r_Sil_new, ρdud_r_C_new, ρdud_r_PAH_new )
        ρdud_Z_new = ( ρdud_Z_Sil_new, ρdud_Z_C_new, ρdud_Z_PAH_new )

        #######################################################################
        #######                                                         #######
        #######      S T E P  4  :  H E A T I N G / C O O L I N G       #######
        #######                                                         #######
        #######################################################################

        # Next we do an update of e from heating and cooling, over a 
        # substep Δtsub.

        # Pack grain charges & dust densities in tuples for use below 
        Zgr    = ( Zgr_Sil, Zgr_C, Zgr_PAH ) 
        ρd_new = ( ρd_Sil_new, ρd_C_new, ρd_PAH_new)

        # Cooling/heating update: 
        REDO_HEATINGCOOLING, e_new, 
        Δe, Δt_e = HeatingCoolingStep( Δtsub, e_old, Td_new, Zgr, γad, n_new, 
                                       nH_new, nC_new, nO_new, nS_new,
                                       nHI_new, nH2_new, nHII_new, nHeI_new, nHeII_new, nHeIII_new,
                                       nHD_new, nCI_new, nCO_new, nCII_new, 
                                       nOI_new, nOII_new, nOIII_new,
                                       nSI_new, nSII_new, nSIII_new,
                                       ne_new, ρ_new, 
                                       ZSil_new, Zgasrest_new, ρd_new, N_CO, 
                                       J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3,
                                       SelfShielding, e_Lyα_new, z, Γpd, OTS, e_floor,
                                       StellarAverages )

        if REDO_HEATINGCOOLING
            # Change in e was too large or unphysical, so retry with 
            # smaller sub-cycle time-step:
            Δtsub = min( Δtsub_half,  
                         Δt_e )

            # Restore Lyα arrays J_Lyα and H_Lyα:
            RestoreLyαRejection( icell, jcell, Lyα )

            # Update rejection count:
            N_REDO_HEATINGCOOLING += 1

            # Restore old state for e and T:
            e_old = e_old_FirstTry
            T_old = T_old_FirstTry

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
        #######                    S T E P  5  :  C H E M I S T R Y                 #######
        #######                                                                     #######
        ###################################################################################

        # Finally, we update the chemistry over a sub-step Δtsub.

        # Get the normalized temperature Tnorm = T / [μ(γ-1)] = e × mH / (kB × ρ):
        @fastmath Tnorm = e_new * mH / (kB * ρ_new)

        # Update chemistry (NOTE: number densities have changed from dust update, but not 
        # abundance fractions, hence use of x*_old yet n*_new):
        ( xH2_new,  xHI_new,   xHII_new,  xHD_new, xDI_new, xDII_new,
          xHeI_new, xHeII_new, xHeIII_new,
          xOI_new,  xOII_new,  xOIII_new, xOIV_new,
          xCI_new,  xCII_new,  xCIII_new, xCO_new, 
          xSII_new, xSIII_new, xSIV_new,
          ne_new, Δne ) =
          UpdateChemistrySubStep( Δtsub, Tnorm, Td_new, ρ_new, ρd_new, 
                                  nH_new, nC_new, nO_new, nS_new, nHI_new, nH2_new, nHII_new, ne_new,
                                  xH2_old,  xHII_old,  xHD_old,   xDII_old, xHeII_old, xHeIII_old,
                                  xOI_old,  xOII_old,  xOIII_old, xOIV_old, 
                                  xCI_old,  xCII_old,  xCIII_old, xCO_old, 
                                  xSII_old, xSIII_old, xSIV_old,
                                  ZC_new,   ZSil_new,  Zgasrest_new,
                                  ΓionHI,   ΓionH2,    ΓionDI,   ΓionHeI, ΓionHeII, Γ_LW, Γpd, Γ_LW_HD, 
                                  ΓionOI,   ΓionOII,   ΓionOIII, ΓionCI,   ΓionCII, ΓdissCO, ΓdissOHx, ΓdissCHx,
                                  ΓionSI,   ΓionSII,   ΓionSIII,
                                  z, OTS,   HydrogenAndHelium ) 

        # Next, compute would-be relative change in H₂, H II, C/O, S, and e⁻ fractions/number densities. 
        # We ignore C/O and S chemistry changes at high temperatures where they don't matter, and at very low 
        # metallicities where they also are unimportant. (NOTE: Δne is already taking abs into account, hence Δne > 0):
        CarbonOrNot = ( ZC_new       > Z_IGNORE_CARBOXYGEN  &&  T_old < T_IGNORE_CARBOXYGEN )
        OxygenOrNot = ( Zgasrest_new > Z_IGNORE_CARBOXYGEN  &&  T_old < T_IGNORE_CARBOXYGEN )
        SulfurOrNot = ( Zgasrest_new > Z_IGNORE_SULFUR      &&  T_old < T_IGNORE_SULFUR )

        ΔxH2_rel    = abs( xH2_new   - xH2_old )  * inv_xH2_old
        ΔxHII_rel   = abs( xHII_new  - xHII_old ) * inv_xHII_old
        ΔxCO_rel    = CarbonOrNot * abs( xCO_new   - xCO_old )   * inv_xCO_old
        ΔxCII_rel   = CarbonOrNot * abs( xCII_new  - xCII_old )  * inv_xCII_old
        ΔxCIII_rel  = CarbonOrNot * abs( xCIII_new - xCIII_old ) * inv_xCIII_old
        ΔxOII_rel   = OxygenOrNot * abs( xOII_new  - xOII_old )  * inv_xOII_old
        ΔxOIII_rel  = OxygenOrNot * abs( xOIII_new - xOIII_old ) * inv_xOIII_old
        ΔxOIV_rel   = OxygenOrNot * abs( xOIV_new  - xOIV_old )  * inv_xOIV_old
        ΔxSII_rel   = SulfurOrNot * abs( xSII_new  - xSII_old )  * inv_xSII_old
        ΔxSIII_rel  = SulfurOrNot * abs( xSIII_new - xSIII_old ) * inv_xSIII_old
        ΔxSIV_rel   = SulfurOrNot * abs( xSIV_new  - xSIV_old )  * inv_xSIV_old
        Δne_rel     = @fastmath Δne * inv_ne_old

        # Get the maximum relative change among them:
        Δx_rel      = maximum( ( ΔxH2_rel,  ΔxHII_rel, 
                                 ΔxCO_rel,  ΔxCII_rel,  ΔxCIII_rel, 
                                 ΔxOII_rel, ΔxOIII_rel, ΔxOIV_rel,
                                 ΔxSII_rel, ΔxSIII_rel, ΔxSIV_rel, 
                                 Δne_rel ) )

        if Δx_rel > tol_Chem || !isfinite(Δx_rel)
            # Change in H₂, H II, e⁻, C/O, or S was too high, reduce Δtsub & retry:
            Δtsub     = Δtsub_half   # Preliminary new suggested substep (halved)

            # If Δx_rel is finite & not extreme, use it to infer a better new substep, but not too agressively
            # (allow for up to a factor 10 change in substep at most):
            if isfinite(Δx_rel)  
                Δtsub = max( min( Δtsub,  tol_Chem * Δtsub_old / Δx_rel ),  Δtsub_newmin )
            end

            # Restore Lyα arrays J_Lyα and H_Lyα:
            RestoreLyαRejection( icell, jcell, Lyα )

            # Update rejection count:
            N_REDO_CHEMISTRY += 1

            # Restore old state for e and T:
            e_old = e_old_FirstTry
            T_old = T_old_FirstTry

            continue    
        end

        # Commit new Lyα energy density & reset 'old' J_Lyα and H_Lyα: 
        if !isnothing(Lyα)
            Lyα.e_Lyα[icell,jcell] = e_Lyα_new

            @inbounds for k in Lyα.dN_ν
                Lyα.J_Lyα_old[k,icell,jcell]   = Lyα.J_Lyα[k,icell,jcell]

                Lyα.H_r_Lyα_old[k,icell,jcell] = Lyα.H_r_Lyα[k,icell,jcell]
                Lyα.H_Z_Lyα_old[k,icell,jcell] = Lyα.H_Z_Lyα[k,icell,jcell]
            end
        end

        # Recommended next sub-cycle time-step based on the relative change 
        # in quantities:
        if ( Δx_rel < tol_enlargestep &&  Δe < tol_enlargestep  
             &&  ΔStellarRT < tol_enlargestep  &&  Δρd_rel < tol_enlargestep )

            # Suggested new substep:
            Δtsub_new = 2 * Δtsub_old 
            Δtsub_new = min( Δtsub_new, Δt_e )  # Limit so it doesn't exceed cooling/heating time-scale 
        else
            Δtsub_new = Δtsub
        end

        if DEBUG_PhotoThermoChemistry && iter > 20
            @info """  Small sub-steps in cell ($(icell),$(jcell)):
                       Δtsub                        = $(Δtsub / yr) yr
                       c̃ / c                        = $(c̃/c)
                       N_REDO_ABSORPTION            = $(N_REDO_ABSORPTION)
                       N_REDO_DUSTGASCOUPLING       = $(N_REDO_DUSTGASCOUPLING)
                       N_REDO_DUSTGROWTHDESTRUCTION = $(N_REDO_DUSTGROWTHDESTR)
                       N_REDO_HEATINGCOOLING        = $(N_REDO_HEATINGCOOLING)
                       N_REDO_CHEMISTRY             = $(N_REDO_CHEMISTRY)
                       
                       """
        end

        # Return every updated quantity, the time-step taken, & recommended next sub-cycle time-step:
        return Δtsub, Δtsub_new, e_new, xH2_new, xHII_new, xHD_new, xDII_new, xHeII_new, xHeIII_new,
               xOI_new, xOII_new, xOIII_new, xOIV_new,
               xCI_new, xCII_new, xCIII_new, xCO_new, 
               xSII_new, xSIII_new, xSIV_new,
               J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, J_IR,
               H_r_opt, H_Z_opt,    H_r_FUV, H_Z_FUV,    H_r_LW, H_Z_LW,  
               H_r_EUV1, H_Z_EUV1,  H_r_EUV2, H_Z_EUV2,  H_r_EUV3, H_Z_EUV3,
               H_r_IR, H_Z_IR,
               Δρu_r_trial, Δρu_Z_trial,  Δρu_r_IR_trial, Δρu_Z_IR_trial, # Diagnostics for stellar & IR radiation pressure 
               Δρu_r_Lyα_trial, Δρu_Z_Lyα_trial,                          # Diagnostics for Lyα radiation pressure
               ρu_r_new, ρu_Z_new,                   # Updated gas momentum 
               ρdud_r_new, ρdud_Z_new,               # Updated dust momenta (for Sil, C, PAH) 
               Td_new, js_IR, Zgr, 
               Trad_new,                             # Updated IR radiation temperature Trad
               ρd_new, ρ_new, 
               ZC_new, ZSil_new, Zgasrest_new        # Updated metallicities of C, Si, and rest 

    end

    # Otherwise print error:
    error( """ ThermoChemistrySubStepUpdate could not converge within $(max_iter) iterations in cell (i,j) = ($(icell),$(jcell)),
               and simulation has therefore terminated. ρ = $(ρ_old), ρd = $(ρd_old).
               
               Diagnostics for this cell during the final attempted sub-cycle time-step were:
               
               - Δtsub                                           = $(Δtsub / yr) yr
               - c̃ / c                                           = $(c̃/c)
               - Relative change in e⁻ abundance, Δne_rel        = $(Δne_rel)
               - Relative change in H₂ abundance, ΔxH2_rel       = $(ΔxH2_rel)
               - Relative change in H II abundance, ΔxHII_rel    = $(ΔxHII_rel)
               - Relative change in CO abundance, ΔxCO_rel       = $(ΔxCO_rel)
               - Relative change in C II abundance, ΔxCII_rel    = $(ΔxCII_rel)
               - Relative change in C III abundance, ΔxCIII_rel  = $(ΔxCIII_rel)
               - Relative change in O II abundance, ΔxOII_rel    = $(ΔxOII_rel)
               - Relative change in O III abundance, ΔxOIII_rel  = $(ΔxOIII_rel)
               - Relative change in O IV abundance, ΔxOIV_rel    = $(ΔxOIV_rel)
               - Relative change in S II abundance, ΔxSII_rel    = $(ΔxSII_rel)
               - Relative change in S III abundance, ΔxSIII_rel  = $(ΔxSIII_rel)
               - Relative change in S IV abundance, ΔxSIV_rel    = $(ΔxSIV_rel)
               - Relative change in e, Δe_rel                    = $(Δe)
               - Relative change in stellar RT, ΔStellarRT       = $(ΔStellarRT)
               - Relative change in dust densities, Δρd_rel      = $(Δρd_rel)
               - N_REDO_ABSORPTION                               = $(N_REDO_ABSORPTION)
               - N_REDO_DUSTGASCOUPLING                          = $(N_REDO_DUSTGASCOUPLING)
               - N_REDO_DUSTGROWTHDESTRUCTION                    = $(N_REDO_DUSTGROWTHDESTR)
               - N_REDO_HEATINGCOOLING                           = $(N_REDO_HEATINGCOOLING)
               - N_REDO_CHEMISTRY                                = $(N_REDO_CHEMISTRY) """ )

end # End of ThermoChemistrySubStepUpdate!


""" The full thermochemistry sub-cycling over a global time-step Δtglobal.
    Updates the intensities, gas internal energy, and chemical abundances. """
@views function ThermoChemistrySubCycleUpdate!(  icell::Int,  jcell::Int,
                                                    Δr::Float64, ΔZ::Float64,
                                              Δtglobal::Float64,
                                                     c̃::Float64,  # Stellar/IR reduced speed of light 
                                                 e_old::Float64,
                                                 ρ_old::Float64,
                                                  Lsob::Float64,
                                               xH2_old::Float64,
                                              xHII_old::Float64, 
                                               xHD_old::Float64,
                                              xDII_old::Float64,
                                             xHeII_old::Float64,
                                            xHeIII_old::Float64,
                                               xOI_old::Float64, 
                                              xOII_old::Float64, 
                                             xOIII_old::Float64,
                                              xOIV_old::Float64,
                                               xCI_old::Float64, 
                                              xCII_old::Float64, 
                                             xCIII_old::Float64,
                                               xCO_old::Float64,
                                              xSII_old::Float64,
                                             xSIII_old::Float64,
                                              xSIV_old::Float64,
                                                ZC_old::Float64,  # Carbon gas metallicity 
                                              ZSil_old::Float64,  # Si gas metallicity 
                                          Zgasrest_old::Float64,  # Rest of gas metallicity 
                                                ρd_old::NTuple{3,Float64}, # (Sil, C, PAH)
                                                  Dust::DustState,
                                                   Lyα::Union{LyαArrays,Nothing},

                                              ρu_r_old::Float64,
                                              ρu_Z_old::Float64,
                                            ρdud_r_old::NTuple{3,Float64}, # (Sil, C, PAH)
                                            ρdud_Z_old::NTuple{3,Float64}, # (Sil, C, PAH)
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
                                           RadPressure::Bool    = true,   # Radiation pressure feedback on gas & dust (on if true, off if false)
                                                 L_box::Float64 = ΔL_max, # Box size, to limit column densities
                                        CoolingHeating::Bool    = true,
                                 DustGrowthDestruction::Bool    = true,
                                     HydrogenAndHelium::Bool    = true,
                                       StellarAverages::StellarBandAverages = DefaultStellarBandAverages
                                              )

    # Initialize gas & dust momenta:
    ρu_r_current      = ρu_r_old 
    ρu_Z_current      = ρu_Z_old 
    ρdud_r_current    = ρdud_r_old  
    ρdud_Z_current    = ρdud_Z_old 

    # Initialize gas & dust density, and metallicity:
    ρ_current         = ρ_old
    ρd_current        = ρd_old   # (Sil, C, PAH) tuple 
    ZC_current        = ZC_old
    ZSil_current      = ZSil_old
    Zgasrest_current  = Zgasrest_old

    # Initialize time, internal energy, abundance fractions, dust temperatures,
    # and IR radiation temperature, using the ones from the previous global time-step:
    t_accum           = 0.0 
    e_current         = e_old 
    xH2_current       = xH2_old
    xHII_current      = xHII_old
    xHD_current       = xHD_old 
    xDII_current      = xDII_old
    xHeII_current     = xHeII_old
    xHeIII_current    = xHeIII_old
    xOI_current       = xOI_old
    xOII_current      = xOII_old
    xOIII_current     = xOIII_old
    xOIV_current      = xOIV_old 
    xCI_current       = xCI_old
    xCII_current      = xCII_old
    xCIII_current     = xCIII_old
    xCO_current       = xCO_old
    xSII_current      = xSII_old
    xSIII_current     = xSIII_old
    xSIV_current      = xSIV_old 
    Td                = ( Dust.Td_Sil[icell,jcell], Dust.Td_C[icell,jcell], Dust.Td_PAH[icell,jcell] )
    Trad_current      = Dust.Trad[icell, jcell] # Latest radiation temperature 

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

    # Initialize grain charges:
    Zgr   = ( Dust.Zgr_Sil[icell,jcell], 
              Dust.Zgr_C[icell,jcell], 
              Dust.Zgr_PAH[icell,jcell] )  
    
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
                  e_current, ρ_current, Lsob,
                  xH2_current, xHII_current, xHD_current, xDII_current,
                  xHeII_current, xHeIII_current, 
                  xCI_current,  xCII_current,  xCIII_current, xCO_current, 
                  xOI_current,  xOII_current,  xOIII_current, xOIV_current,
                  xSII_current, xSIII_current, xSIV_current,
                  ZC_current, ZSil_current, Zgasrest_current, 
                  ρd_current,
                  ρdud_r_current, ρdud_Z_current,     
                  ρu_r_current, ρu_Z_current,  

                  Dust, Trad_current, Td, Zgr, Lyα,

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

                  z, OTS, RadPressure, L_box, CoolingHeating, 
                  DustGrowthDestruction,
                  HydrogenAndHelium,
                  StellarAverages )

        # Unpack result:
        Δtsub, Δtsub_new, e_new, xH2_new, xHII_new, xHD_new, xDII_new, xHeII_new, xHeIII_new, 
        xOI_new, xOII_new, xOIII_new, xOIV_new,
        xCI_new, xCII_new, xCIII_new, xCO_new,
        xSII_new, xSIII_new, xSIV_new,
        J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, J_IR,
        H_r_opt, H_Z_opt,    H_r_FUV, H_Z_FUV,    H_r_LW, H_Z_LW,  
        H_r_EUV1, H_Z_EUV1,  H_r_EUV2, H_Z_EUV2,  H_r_EUV3, H_Z_EUV3,
        H_r_IR, H_Z_IR,
        Δρu_r_substep, Δρu_Z_substep,  
        Δρu_r_IR_substep, Δρu_Z_IR_substep, 
        Δρu_r_Lyα_substep, Δρu_Z_Lyα_substep, 
        ρu_r_new, ρu_Z_new,  ρdud_r_new, ρdud_Z_new,  
        Td_new, js_IR, Zgr_new, 
        Trad_new,
        ρd_new, ρ_new, 
        ZC_new, ZSil_new, Zgasrest_new  = result

        # Update our current internal energy & abundance fractions:
        e_current         = e_new
        xH2_current       = xH2_new
        xHII_current      = xHII_new
        xHD_current       = xHD_new 
        xDII_current      = xDII_new
        xHeII_current     = xHeII_new
        xHeIII_current    = xHeIII_new
        xOI_current       = xOI_new
        xOII_current      = xOII_new
        xOIII_current     = xOIII_new
        xOIV_current      = xOIV_new
        xCI_current       = xCI_new
        xCII_current      = xCII_new
        xCIII_current     = xCIII_new
        xCO_current       = xCO_new
        xSII_current      = xSII_new
        xSIII_current     = xSIII_new
        xSIV_current      = xSIV_new
        # Update our current intensities:
        J_opt_current     = J_opt
        J_FUV_current     = J_FUV 
        J_LW_current      = J_LW 
        J_EUV1_current    = J_EUV1
        J_EUV2_current    = J_EUV2
        J_EUV3_current    = J_EUV3 
        J_IR_current      = J_IR
        # Update our current H-components:
        H_r_opt_current   = H_r_opt
        H_Z_opt_current   = H_Z_opt 
        H_r_FUV_current   = H_r_FUV
        H_Z_FUV_current   = H_Z_FUV
        H_r_LW_current    = H_r_LW
        H_Z_LW_current    = H_Z_LW 
        H_r_EUV1_current  = H_r_EUV1
        H_Z_EUV1_current  = H_Z_EUV1
        H_r_EUV2_current  = H_r_EUV2 
        H_Z_EUV2_current  = H_Z_EUV2 
        H_r_EUV3_current  = H_r_EUV3
        H_Z_EUV3_current  = H_Z_EUV3
        H_r_IR_current    = H_r_IR
        H_Z_IR_current    = H_Z_IR

        # Update gas & dust momenta:
        ρu_r_current      = ρu_r_new
        ρu_Z_current      = ρu_Z_new
        ρdud_r_current    = ρdud_r_new
        ρdud_Z_current    = ρdud_Z_new

        # Update gas & dust density, and metallicity:
        ρ_current         = ρ_new
        ρd_current        = ρd_new
        ZC_current        = ZC_new
        ZSil_current      = ZSil_new
        Zgasrest_current  = Zgasrest_new

        # Update dust temperature & charge, and IR radiation temperature:
        Td                = Td_new
        Zgr               = Zgr_new
        Trad_current      = Trad_new

        # Add diagnostic stellar radiation pressure momentum from this sub-step: 
        Δρu_stellar_r[icell, jcell] += Δρu_r_substep
        Δρu_stellar_Z[icell, jcell] += Δρu_Z_substep

        # Add diagnostic IR radiation pressure momentum from this sub-step: 
        Δρu_IR_r[icell, jcell]      += Δρu_r_IR_substep
        Δρu_IR_Z[icell, jcell]      += Δρu_Z_IR_substep

        # Add diagnostic Lyα radiation pressure momentum from this sub-step:
        if !isnothing(Lyα)
            Lyα.Δρu_Lyα_r[icell, jcell] += Δρu_r_Lyα_substep
            Lyα.Δρu_Lyα_Z[icell, jcell] += Δρu_Z_Lyα_substep
        end

        # Update current time:
        t_accum += Δtsub

        # Update sub-cycle time-step using the new recommended value:
        Δtsub = Δtsub_new

        # Update the step count and warn the user if it has exceeded the specified upper limit:
        step += 1
        if step > max_steps
            # Get gas velocity magnitudes for printing:
            inv_ρ  = 1 / ρ_current
            u_r    = ρu_r_current * inv_ρ 
            u_Z    = ρu_Z_current * inv_ρ 
            u      = hypot( u_r, u_Z )

            # Get dust densiteis & velocity magnitudes for printing:
            ρd_Sil_current, ρd_C_current, ρd_PAH_current             = ρd_current
            ρdud_r_Sil_current, ρdud_r_C_current, ρdud_r_PAH_current = ρdud_r_current
            ρdud_Z_Sil_current, ρdud_Z_C_current, ρdud_Z_PAH_current = ρdud_Z_current

            ud_Sil = hypot( ρdud_r_Sil_current / ρd_Sil_current,
                            ρdud_Z_Sil_current / ρd_Sil_current )

            ud_C   = hypot( ρdud_r_C_current / ρd_C_current,
                            ρdud_Z_C_current / ρd_C_current)

            ud_PAH = hypot( ρdud_r_PAH_current / ρd_PAH_current,
                            ρdud_Z_PAH_current / ρd_PAH_current )

            # Get dust temperatures:
            Td_Sil, Td_C, Td_PAH = Td

            error(""" The thermochemistry sub-cycling exceeded $(max_steps) steps in cell ($(icell), $(jcell)). 
                      Terminating the simulation. 
                      
                      Physical quantities in the cell:
                      
                      e        = $(e_current) erg cm⁻³
                      ρ        = $(ρ_current) g cm⁻³
                      ρd_Sil   = $(ρd_Sil_current) g cm⁻³ 
                      ρd_C     = $(ρd_C_current) g cm⁻³ 
                      ρd_PAH   = $(ρd_PAH_current) g cm⁻³ 
                      Td_Sil   = $(Td_Sil) K 
                      Td_C     = $(Td_C) K 
                      Td_PAH   = $(Td_PAH) K 
                      Trad     = $(Trad_current) K
                      |u|      = $(u / kms) km s⁻¹ 
                      |ud_Sil| = $(ud_Sil / kms) km s⁻¹
                      |ud_C|   = $(ud_C / kms) km s⁻¹
                      |ud_PAH| = $(ud_PAH / kms) km s⁻¹
                      ZC       = $(ZC_current)
                      ZSil     = $(ZSil_current)
                      Zgasrest = $(Zgasrest_current)
                      xH2      = $(xH2_current) 
                      xHII     = $(xHII_current)
                      xHD      = $(xHD_current) 
                      xDII     = $(xDII_current)
                      xHeII    = $(xHeII_current)
                      xHeIII   = $(xHeIII_current) 
                      xOI      = $(xOI_current) 
                      xOII     = $(xOII_current)
                      xOIII    = $(xOIII_current)
                      xOIV     = $(xOIV_current)
                      xCI      = $(xCI_current) 
                      xCII     = $(xCII_current) 
                      xCIII    = $(xCIII_current)
                      xCO      = $(xCO_current) 
                      xSII     = $(xSII_current)
                      xSIII    = $(xSIII_current)
                      xSIV     = $(xSIV_current)
                      
                      ---------------------------- """)
        end
    end

    # Return the updated state after evolving over the global time-step:
    return e_current, xH2_current, xHII_current, xHD_current, xDII_current, xHeII_current, xHeIII_current, 
           xOI_current, xOII_current, xOIII_current, xOIV_current,
           xCI_current, xCII_current, xCIII_current, xCO_current,
           xSII_current, xSIII_current, xSIV_current,
           J_opt_current, J_FUV_current, J_LW_current, J_EUV1_current, J_EUV2_current, J_EUV3_current, 
           J_IR_current,
           H_r_opt_current,  H_Z_opt_current,   H_r_FUV_current,  H_Z_FUV_current,
           H_r_LW_current,   H_Z_LW_current,    H_r_EUV1_current, H_Z_EUV1_current,  
           H_r_EUV2_current, H_Z_EUV2_current,  H_r_EUV3_current, H_Z_EUV3_current, 
           H_r_IR_current,   H_Z_IR_current, 
           ρu_r_current, ρu_Z_current,  ρdud_r_current, ρdud_Z_current, 
           Td, js_IR, Zgr,
           Trad_current,
           ρ_current, ρd_current, 
           ZC_current, ZSil_current, Zgasrest_current
end


const BENCHMARK_PHOTOTHERMOCHEMISTRY::Bool = false     # For benchmarking (default = false)
const inv_Lmax_∇ρ::Float64                 = 1 / Mpc   # Just a large value to limit ∇ρ / ρ 

""" This function advects stellar/IR + Lyα photons with M1 over an RT step, and then 
    subcycles photo-thermochemistry and dust physics. """
function UpdateRTandThermoChemistry!( Δt_RT_max::Float64, 
                                             RT::StellarRTArrays,
                                     HydroState::HydrodynamicsState,
                                      ChemState::ChemistryState,
                                           Dust::DustState,
                                              z::Float64,
                                 CoolingHeating::Bool,

                                           # Geometry:
                                       Geometry::Union{GeometryCylindrical, GeometryCartesian},

                                          # Gas velocity field:
                                       u_r_star::AbstractArray{Float64,2},
                                       u_Z_star::AbstractArray{Float64,2},
                                        ∂u_r_∂r::AbstractArray{Float64,2},
                                        ∂u_r_∂Z::AbstractArray{Float64,2}, 
                                        ∂u_Z_∂r::AbstractArray{Float64,2}, 
                                        ∂u_Z_∂Z::AbstractArray{Float64,2},
                                     u_r_over_r::AbstractArray{Float64,2},

                                           # Lyα state (nothing if no Lyα RT):
                                       LyaState::Union{LyαArrays,Nothing},

                                           # Radiation pressure feedback on gas & dust 
                                           # (on if true, off if false):
                                    RadPressure::Bool = true,

                                           # For using different H & He cross-sections and heating rates:
                                StellarAverages::StellarBandAverages = DefaultStellarBandAverages

                                              )

    if Geometry isa GeometryCylindrical
        r = Geometry.r;  rCoM = Geometry.rCoM
        Z = Geometry.Z;  ZCoM = Geometry.ZCoM
    else
        # Cartesian otherwise:
        r = Geometry.X;  rCoM = Geometry.XCoM
        Z = Geometry.Y;  ZCoM = Geometry.YCoM
    end

    # Reduced speed of lights for stellar/IR and Lyα RT:
    c̃      = RT.c̃                                      # Stellar/IR reduced speed of light 
    c̃_Lyα  = isnothing(LyaState) ? c̃ : LyaState.c̃_Lyα  # If Lyα RT is off, just set the RSL to the stellar/IR one
    
    # CFL time-steps for stellar/IR and Lyα RT:
    Δt_CFL_StellarIR = GetRHDCFLTimeStep( r, Z, c̃, HydroState.u_X, HydroState.u_Y, CFL_RT ) 

    if isnothing(LyaState)
        Δt_CFL_Lyα = Inf 
    else
        Δt_CFL_Lyα = GetRHDCFLTimeStep( r, Z, c̃_Lyα, HydroState.u_X, HydroState.u_Y, CFL_RT ) 
    end

    # Use the most restrictive CFL time-step:
    Δt_RT = min( Δt_CFL_StellarIR, Δt_CFL_Lyα, Δt_RT_max )

    # Convenient factors:
    c̃Δt_RT_StellarIR = c̃     * Δt_RT 
    c̃Δt_RT_Lyα       = c̃_Lyα * Δt_RT 

    # Box size, to limit column densities:
    L_box     = max( abs(r[end] - r[1]), abs(Z[end] - Z[1]) )  
    
    # Unpack hydro, dust & chemistry arrays:
    ρ          = HydroState.ρ
    ρu_r       = HydroState.ρu_X
    ρu_Z       = HydroState.ρu_Y
    e          = HydroState.e_dual 
    T          = HydroState.T

    ρd         = HydroState.ρd
    ρdud_r     = HydroState.ρdud_X
    ρdud_Z     = HydroState.ρdud_Y

    xH2        = ChemState.xH2
    xHII       = ChemState.xHII
    xHD        = ChemState.xHD
    xDII       = ChemState.xDII
    xHeII      = ChemState.xHeII 
    xHeIII     = ChemState.xHeIII 
    xCI        = ChemState.xCI 
    xCII       = ChemState.xCII
    xCIII      = ChemState.xCIII
    xCO        = ChemState.xCO
    xOI        = ChemState.xOI 
    xOII       = ChemState.xOII
    xOIII      = ChemState.xOIII
    xOIV       = ChemState.xOIV
    xSII       = ChemState.xSII
    xSIII      = ChemState.xSIII
    xSIV       = ChemState.xSIV
    ZC         = ChemState.ZC
    ZSil       = ChemState.ZSil
    Zgasrest   = ChemState.Zgasrest
    Lsob       = ChemState.Lsob

    # On-spot-approximation (Case B) or modelling of recombination radiation:
    OTS        = RT.OTS

    # Whether to model hydrogen + helium + deuterium, or just hydrogen (latter is useful 
    # for idealized tests):
    HydrogenAndHelium = ChemState.HydrogenAndHelium

    # Dust growth/destruction setting:
    DustGrowthDestruction = Dust.DustGrowthDestruction

    # Boundary conditions:
    BC_r_left  = HydroState.BC_X_left
    BC_r_right = HydroState.BC_X_right
    BC_Z_bot   = HydroState.BC_Y_left
    BC_Z_top   = HydroState.BC_Y_right

    #######################################################################
    #######                                                         #######
    #######       S T E P  1  :  I N J E C T I O N  S T E P         #######
    #######                                                         #######
    #######################################################################

    # Stellar emission update: 
    @views begin
        @. RT.J[iopt_RT,  :, :] +=  RT.js_opt  * c̃Δt_RT_StellarIR
        @. RT.J[iFUV_RT,  :, :] +=  RT.js_FUV  * c̃Δt_RT_StellarIR
        @. RT.J[iLW_RT,   :, :] +=  RT.js_LW   * c̃Δt_RT_StellarIR
        @. RT.J[iEUV1_RT, :, :] +=  RT.js_EUV1 * c̃Δt_RT_StellarIR
        @. RT.J[iEUV2_RT, :, :] +=  RT.js_EUV2 * c̃Δt_RT_StellarIR
        @. RT.J[iEUV3_RT, :, :] +=  RT.js_EUV3 * c̃Δt_RT_StellarIR
    end

    #######################################################################
    #######                                                         #######
    #######       S T E P  2  :  T R A N S P O R T  S T E P         #######
    #######                                                         #######
    #######################################################################

    if BENCHMARK_PHOTOTHERMOCHEMISTRY
        # Initialize timings:
        t_StellarIR_RT = 0.0 # For stellar + IR M1 RT
        t_LyαRT        = 0.0 # For Lyα M1 RT 
        t_Subcycling   = 0.0 # For subcycling of photothermochemistry 
        t_LyαFreqSolve = 0.0 # For frequency (Doppler + redistribution) solve for Lyα

        t0 = time_ns()
    end

    # We advect the J and H for all bands, using the M1 moment method. 

    UpdateStellarJold = false # Don't update Jold, since we want the pre-(injection + transport) value to compute dJ/dt

    # Advect J and H for stellar + IR bands:
    AdvectJandH!( c̃Δt_RT_StellarIR,
                  RT.α_F,
                  RT.J,    RT.J_old,
                  RT.H_r,  RT.H_Z,
                  RT.K_ϕϕ_iphalf, RT.K_ϕϕ_imhalf,
                  RT.F_r,    RT.F_Z,
                  RT.F_r_Hr, RT.F_Z_Hr,
                  RT.F_r_HZ, RT.F_Z_HZ,
                  Geometry,
                  u_r_star, u_Z_star,
                  ∂u_r_∂r,  ∂u_r_∂Z,
                  ∂u_Z_∂r,  ∂u_Z_∂Z,
                  u_r_over_r,
                  UpdateStellarJold,
                  BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top,
                  1, Nbands_RT, RT.FluxMethod )

    if BENCHMARK_PHOTOTHERMOCHEMISTRY
        t_StellarIR_RT = time_ns() - t0

        # Reset:
        t0 = time_ns()
    end

    # Advect Lyα photons, if simulated:
    if !isnothing(LyaState)
        AdvectLyα!( c̃Δt_RT_Lyα, LyaState,   Geometry, 
                    u_r_star,   u_Z_star,   ∂u_r_∂r,  ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                    BC_r_left,  BC_r_right, BC_Z_bot, BC_Z_top ) 
    end

    if BENCHMARK_PHOTOTHERMOCHEMISTRY
        t_LyαRT = time_ns() - t0

        # Reset:
        t0 = time_ns()
    end

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

    # First compute Sobolev-like length scale ρ / ∇ρ,
    # needed for column density estimates (we do it in a separate loop 
    # to avoid threading inconsistencies where some ρ are updated):
    @threads for j in dN_Z
        @inbounds @simd for i in dN_r
            ∇ρ        = Estimate_∇ρ( i, j, HydroState, Geometry )
            Lsob[i,j] = ρ[i,j] / max( ∇ρ,  ρ[i,j] * inv_Lmax_∇ρ )
        end
    end

    # ------------- Next, sub-cycle for each cell ---------------------------

    # The work per cell can vary dramatically - some cells may need O(1) substeps, 
    # while others may need very many. In early testing, default (:dynamic) threading 
    # over cells gave only ~50-60% thread utilization, while per-cell :greedy gave 
    # ~ 90-99% thread utilization (great!) but more overhead. The method we use here 
    # is therefore over patches of cells, which keeps the threads occupied (still 
    # > 90% thread utilization), and has more work per thread, which reduces overhead. 
    
    # Patch size (default is 16 × 16, but some variations don't give much change in 
    # performance):
    Patch_size_i = 16  # Number of cells in r-direction per patch 
    Patch_size_j = 16  # Number of cells in Z-direction per patch
    
    N_patch_i    = cld( N_r, Patch_size_i )
    N_patch_j    = cld( N_Z, Patch_size_j )
    N_patches    = N_patch_i * N_patch_j

    inv_Δt_RT    = 1.0 / Δt_RT  # Convenient constant 
    @threads :greedy for patch_idx in 1:N_patches
        @inbounds begin

            # Get patch index in the r and Z-directions:
            patch_i = 1  +  (patch_idx - 1) % N_patch_i
            patch_j = 1  +  (patch_idx - 1) ÷ N_patch_i

            # Cell-index range covered by this patch:
            i_first = 1  +  (patch_i - 1) * Patch_size_i
            i_last  = min( patch_i * Patch_size_i,  N_r )

            j_first = 1  +  (patch_j - 1) * Patch_size_j
            j_last  = min( patch_j * Patch_size_j,  N_Z )

            # Loop over cells (i,j) within the patch:
            for j in j_first:j_last

                ΔZ = Z[j+1] - Z[j]       # Cell width in Z-direction

                for i in i_first:i_last

                    Δr = r[i+1] - r[i]   # Cell width in r-direction

                    # Get estimated time-derivatives of J from emission + advection
                    # for stellar RT bands:
                    dJ_optdt   = ( RT.J[iopt_RT,i,j]  - RT.J_old[iopt_RT,i,j]  ) * inv_Δt_RT
                    dJ_FUVdt   = ( RT.J[iFUV_RT,i,j]  - RT.J_old[iFUV_RT,i,j]  ) * inv_Δt_RT
                    dJ_LWdt    = ( RT.J[iLW_RT,i,j]   - RT.J_old[iLW_RT,i,j]   ) * inv_Δt_RT
                    dJ_EUV1dt  = ( RT.J[iEUV1_RT,i,j] - RT.J_old[iEUV1_RT,i,j] ) * inv_Δt_RT
                    dJ_EUV2dt  = ( RT.J[iEUV2_RT,i,j] - RT.J_old[iEUV2_RT,i,j] ) * inv_Δt_RT
                    dJ_EUV3dt  = ( RT.J[iEUV3_RT,i,j] - RT.J_old[iEUV3_RT,i,j] ) * inv_Δt_RT

                    # Get estimated time-derivatives of H_r and H_Z from advection
                    # for stellar RT bands:
                    dHr_optdt  = ( RT.H_r[iopt_RT,i,j]  - RT.H_r_old[iopt_RT,i,j] )  * inv_Δt_RT
                    dHZ_optdt  = ( RT.H_Z[iopt_RT,i,j]  - RT.H_Z_old[iopt_RT,i,j] )  * inv_Δt_RT

                    dHr_FUVdt  = ( RT.H_r[iFUV_RT,i,j]  - RT.H_r_old[iFUV_RT,i,j] )  * inv_Δt_RT 
                    dHZ_FUVdt  = ( RT.H_Z[iFUV_RT,i,j]  - RT.H_Z_old[iFUV_RT,i,j] )  * inv_Δt_RT

                    dHr_LWdt   = ( RT.H_r[iLW_RT,i,j]   - RT.H_r_old[iLW_RT,i,j] )   * inv_Δt_RT
                    dHZ_LWdt   = ( RT.H_Z[iLW_RT,i,j]   - RT.H_Z_old[iLW_RT,i,j] )   * inv_Δt_RT

                    dHr_EUV1dt = ( RT.H_r[iEUV1_RT,i,j] - RT.H_r_old[iEUV1_RT,i,j] ) * inv_Δt_RT
                    dHZ_EUV1dt = ( RT.H_Z[iEUV1_RT,i,j] - RT.H_Z_old[iEUV1_RT,i,j] ) * inv_Δt_RT

                    dHr_EUV2dt = ( RT.H_r[iEUV2_RT,i,j] - RT.H_r_old[iEUV2_RT,i,j] ) * inv_Δt_RT 
                    dHZ_EUV2dt = ( RT.H_Z[iEUV2_RT,i,j] - RT.H_Z_old[iEUV2_RT,i,j] ) * inv_Δt_RT

                    dHr_EUV3dt = ( RT.H_r[iEUV3_RT,i,j] - RT.H_r_old[iEUV3_RT,i,j] ) * inv_Δt_RT
                    dHZ_EUV3dt = ( RT.H_Z[iEUV3_RT,i,j] - RT.H_Z_old[iEUV3_RT,i,j] ) * inv_Δt_RT 

                    # Next we sub-cycle for each cell: 
                    result = ThermoChemistrySubCycleUpdate!( i, j,  Δr, ΔZ,
                                                             Δt_RT,    c̃,
                                                             e[i,j],   
                                                             ρ[i,j],   Lsob[i,j],
                                                           xH2[i,j],   xHII[i,j],
                                                           xHD[i,j],   xDII[i,j],
                                                         xHeII[i,j],   xHeIII[i,j],
                                                           xOI[i,j],   xOII[i,j],  xOIII[i,j], xOIV[i,j],
                                                           xCI[i,j],   xCII[i,j],  xCIII[i,j], xCO[i,j],  
                                                          xSII[i,j],  xSIII[i,j],  xSIV[i,j], 
                                                            ZC[i,j],   ZSil[i,j],  Zgasrest[i,j],
                                                         (ρd[1,i,j], ρd[2,i,j], ρd[3,i,j]),  # (Sil, C, PAH)
                                                               Dust, 
                                                           LyaState,   # Lyα state (or nothing)
                                                          ρu_r[i,j],   ρu_Z[i,j],
                                           (ρdud_r[1,i,j], ρdud_r[2,i,j], ρdud_r[3,i,j]), # (Sil, C, PAH)
                                           (ρdud_Z[1,i,j], ρdud_Z[2,i,j], ρdud_Z[3,i,j]), # (Sil, C, PAH)

                                             RT.Δρu_stellar_r, RT.Δρu_stellar_Z,
                                                  RT.Δρu_IR_r, RT.Δρu_IR_Z,

                                            RT.J_old[iopt_RT,i,j], dJ_optdt,
                                          RT.H_r_old[iopt_RT,i,j], RT.H_Z_old[iopt_RT,i,j], dHr_optdt, dHZ_optdt,

                                            RT.J_old[iFUV_RT,i,j], dJ_FUVdt,
                                          RT.H_r_old[iFUV_RT,i,j], RT.H_Z_old[iFUV_RT,i,j], dHr_FUVdt, dHZ_FUVdt,

                                             RT.J_old[iLW_RT,i,j], dJ_LWdt, 
                                           RT.H_r_old[iLW_RT,i,j], RT.H_Z_old[iLW_RT,i,j], dHr_LWdt, dHZ_LWdt,

                                           RT.J_old[iEUV1_RT,i,j], dJ_EUV1dt, 
                                         RT.H_r_old[iEUV1_RT,i,j], RT.H_Z_old[iEUV1_RT,i,j], dHr_EUV1dt, dHZ_EUV1dt,

                                           RT.J_old[iEUV2_RT,i,j], dJ_EUV2dt, 
                                         RT.H_r_old[iEUV2_RT,i,j], RT.H_Z_old[iEUV2_RT,i,j], dHr_EUV2dt, dHZ_EUV2dt,

                                           RT.J_old[iEUV3_RT,i,j], dJ_EUV3dt,
                                         RT.H_r_old[iEUV3_RT,i,j], RT.H_Z_old[iEUV3_RT,i,j], dHr_EUV3dt, dHZ_EUV3dt,

                                             RT.J[iIR_RT,i,j], 
                                           RT.H_r[iIR_RT,i,j], RT.H_Z[iIR_RT,i,j], 

                                            z, OTS, RadPressure, L_box, CoolingHeating,
                                            DustGrowthDestruction, HydrogenAndHelium,
                                            StellarAverages )

                    # Get updated quantities:
                    e[i,j], xH2[i,j], xHII[i,j], xHD[i,j], xDII[i,j], xHeII[i,j], xHeIII[i,j], 
                    xOI[i,j],  xOII[i,j],  xOIII[i,j], xOIV[i,j],
                    xCI[i,j],  xCII[i,j],  xCIII[i,j], xCO[i,j],
                    xSII[i,j], xSIII[i,j], xSIV[i,j],
                    RT.J[iopt_RT,i,j],  RT.J[iFUV_RT,i,j],  RT.J[iLW_RT,i,j], 
                    RT.J[iEUV1_RT,i,j], RT.J[iEUV2_RT,i,j], RT.J[iEUV3_RT,i,j],
                    RT.J[iIR_RT,i,j],
                    RT.H_r[iopt_RT,i,j],   RT.H_Z[iopt_RT,i,j],
                    RT.H_r[iFUV_RT,i,j],   RT.H_Z[iFUV_RT,i,j],
                    RT.H_r[iLW_RT,i,j],    RT.H_Z[iLW_RT,i,j],
                    RT.H_r[iEUV1_RT,i,j],  RT.H_Z[iEUV1_RT,i,j],  
                    RT.H_r[iEUV2_RT,i,j],  RT.H_Z[iEUV2_RT,i,j],
                    RT.H_r[iEUV3_RT,i,j],  RT.H_Z[iEUV3_RT,i,j],
                    RT.H_r[iIR_RT,i,j],    RT.H_Z[iIR_RT,i,j],
                    ρu_r[i,j],             ρu_Z[i,j],  
                    ρdud_r_new,            ρdud_Z_new, 
                    Td_new,                RT.js_IR[i,j],  Zgr_new,
                    Trad_new,
                    ρ[i,j],            ρd_new,  
                    ZC[i,j], ZSil[i,j], Zgasrest[i,j]  = result 

                    # Unpack tuples for updated dust quantities & save them (Sil, C, PAH):
                    ρd[1,i,j],  ρd[2,i,j],  ρd[3,i,j]             = ρd_new
                    ρdud_r[1,i,j],  ρdud_r[2,i,j],  ρdud_r[3,i,j] = ρdud_r_new
                    ρdud_Z[1,i,j],  ρdud_Z[2,i,j],  ρdud_Z[3,i,j] = ρdud_Z_new

                    Dust.Td_Sil[i,j],  Dust.Td_C[i,j],  Dust.Td_PAH[i,j]  = Td_new
                    Dust.Zgr_Sil[i,j], Dust.Zgr_C[i,j], Dust.Zgr_PAH[i,j] = Zgr_new

                    # Save IR radiation temperature:
                    Dust.Trad[i,j] = Trad_new

                    # Also get the gas temperature in the cell:
                    μ        = Getμ( xHII[i,j], xH2[i,j], xHeII[i,j], xHeIII[i,j], HydrogenAndHelium )
                    γad      = Getγad( xHII[i,j], xH2[i,j], xHeII[i,j], xHeIII[i,j], HydrogenAndHelium )
                    T[i,j]   = e[i,j] * μ * (γad - 1) * mH / ( ρ[i,j] * kB )

                    # Update IR flux opacity (for accurate M1 transport in optically thick regime):
                    RT.α_F[iIR_RT,i,j] =  ( ρd[1,i,j] * κR(Dust, Trad_new, Sil) +
                                            ρd[2,i,j] * κR(Dust, Trad_new, Gra) +
                                            ρd[3,i,j] * κR(Dust, Trad_new, PAH) )

                end # End of i-loop for patch
            end # End of j-loop for patch 
        end # End of inbounds 
    end # End of loop over patches 

    # Update the old stellar & IR J and H 
    # (NOTE: this is done automatically for Lyα in the sub-cycling):
    @threads for j in dN_Z
        @simd for i in dN_r 
            @inbounds for bin in dNbands_RT
                RT.J_old[bin,i,j]   = RT.J[bin,i,j]
                RT.H_r_old[bin,i,j] = RT.H_r[bin,i,j]
                RT.H_Z_old[bin,i,j] = RT.H_Z[bin,i,j]
            end # End of bin loop 
        end # End of i-loop 
    end # End of j-loop 

    if BENCHMARK_PHOTOTHERMOCHEMISTRY
        t_Subcycling = time_ns() - t0

        # Reset:
        t0 = time_ns()
    end


    #######################################################################
    #######                                                         #######
    #######  S T E P  4  :  L Y M A N - α  F R E Q U E N C Y        #######
    #######                 S O L V E                               #######
    #######                                                         #######
    #######################################################################

    # Frequency diffusion/advection of Lyα, if Lyα RT is simulated:
    if !isnothing(LyaState)
        FrequencyUpdateJLyα_AllCells!( c̃Δt_RT_Lyα, LyaState, T, 
                                       ∂u_r_∂r,    ∂u_r_∂Z,  ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                                       true )
    end

    if BENCHMARK_PHOTOTHERMOCHEMISTRY
        t_LyαFreqSolve = time_ns() - t0

        # Reset:
        t0 = time_ns()

        # Print results: 
        println("Stellar + IR M1 RT:  ", t_StellarIR_RT * 1e-9, " sec")
        println("Lyα M1 RT:  ", t_LyαRT * 1e-9, " sec")
        println("Subcycling of photo-chemistry, cooling etc.:  ", t_Subcycling* 1e-9, " sec")
        println("Lyα frequency solve:  ", t_LyαFreqSolve * 1e-9, " sec")
        println("--------------------------")
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


""" This function computes weight factors fstar_ij for 
    each cell (i,j), which determines the fraction of 
    luminosity/emissivty & mass from cell (i,j). 

    This particular version assumes emission from a cylinder, 
    useful for setting up plane-parallel emission in the 
    M1 shadow test.  """
function ComputeShadowTest_fij!( fstar_ij::AbstractArray{Float64, 2},
                                        r::AbstractArray{Float64, 1},
                                        Z::AbstractArray{Float64, 1},
                                       ΔV::AbstractArray{Float64, 2} )

    N_r  = length(r) - 1
    N_Z  = length(Z) - 1
    dN_r = 1:N_r
    dN_Z = 1:N_Z
    @assert size(fstar_ij) == (N_r, N_Z)

    Vcylinder_tot     = π * ( r[end]^2 ) * ( Z[4] - Z[3] )  # Total volume of cylinder
    inv_Vcylinder_tot = 1 / Vcylinder_tot 

    @inbounds @simd for i in dN_r
        fstar_ij[i,3] = ΔV[i,3] * inv_Vcylinder_tot 
    end

    return nothing 
end


""" This function sets up the stellar (+ IR) RT arrays and state,
    for use in the realistic RHD simulations, for a (fixed)
    star of a given mass mstarZAMS (in M⊙). """
function GenerateStellarRTArrays( mstarZAMS::Float64, 
                                 Zstar_Zsun::Float64,
                                         ΔV::AbstractArray{Float64, 2},
                                          r::AbstractArray{Float64, 1},
                                          Z::AbstractArray{Float64, 1},
                                  Zstar_pos::Float64,
                                 σstar_cell::Float64,
                                          c̃::Float64,
                                      c̃_min::Float64 = 100 * kms,
                                        OTS::Bool    = false,  # On-the-spot approximation (if true) or not (if false)
                                        HLL::Bool    = true
                                 )
   
    # Grid size: 
    N_r, N_Z = size( ΔV )
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z

    # Convenient arrays: 
    zeros_c()   = zeros(Float64, N_r,   N_Z)
    zeros_bc()  = zeros(Float64, Nbands_RT, N_r,   N_Z)
    zeros_brf() = zeros(Float64, Nbands_RT, N_r+1, N_Z)   # For r-interface fluxes 
    zeros_bzf() = zeros(Float64, Nbands_RT, N_r,   N_Z+1) # For Z-interface fluxes 

    # Intensities (latest & old) for all bands: 
    J = zeros_bc(); J_old = zeros_bc()

    # H ≡ ∫dΩ I n̂ / 4π arrays for all bands:
    H_r     = zeros_bc(); H_Z     = zeros_bc()
    H_r_old = zeros_bc(); H_Z_old = zeros_bc()

    # ϕϕ-component of radiation pressure tensor K = D × J at interfaces, 
    # needed for geometric source term:
    K_ϕϕ_iphalf = zeros_bc()
    K_ϕϕ_imhalf = zeros_bc()

    # Interface fluxes for all bands: 
    F_r    = zeros_brf(); F_Z    = zeros_bzf()
    F_r_Hr = zeros_brf(); F_Z_Hr = zeros_bzf()
    F_r_HZ = zeros_brf(); F_Z_HZ = zeros_bzf()

    # Flux opacities for IR (can be non-zero) & stellar bands (always set to zero,
    # since no diffusion fix is applied):
    α_F  = zeros_bc()

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

    # Get bolometric stellar luminosity:
    Lum_bol = Lum_opt + Lum_FUV + Lum_LW + Lum_EUV1 + Lum_EUV2 + Lum_EUV3

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
    σstar = σstar_cell * max( minimum(diff(r)), minimum(diff(Z)) )   # σ determining the Gaussian size of the smeared 'star' 
    ComputeStellar_fij!( fstar_ij, r, Z, Zstar_pos, σstar )          # (minimum of diff, to ensure we take cell widths from uniform grid region)

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

    # Initialize LyC and LW escape fractions:
    f_esc_LyC = 0.0
    f_esc_LW  = 0.0 

    # Initialize direct radiation pressure 'force multipliers':
    M_F_dir             = 0.0
    M_F_dir_radial_net  = 0.0
    M_F_dir_radial_out  = 0.0
    M_F_dir_antigravity = 0.0

    # Initialize IR radiation pressure force multipliers:
    M_F_IR              = 0.0
    M_F_IR_radial_net   = 0.0
    M_F_IR_radial_out   = 0.0
    M_F_IR_antigravity  = 0.0

    # Flux method for M1 transport: 
    if HLL
        FluxMethod = Val(:HLL)
    else
        FluxMethod = Val(:GLF)
    end

    return StellarRTArrays( 
        c̃, c̃_min,

        J,          J_old,  
        H_r,        H_Z, 
        H_r_old,    H_Z_old,

        K_ϕϕ_iphalf,   K_ϕϕ_imhalf,

        F_r,      F_Z,   
        F_r_Hr,   F_Z_Hr,   
        F_r_HZ,   F_Z_HZ,

        α_F,

        Lum_IR, Lum_opt, Lum_FUV, Lum_LW, Lum_EUV1, Lum_EUV2, Lum_EUV3, Lum_bol,

        Lum_opt_vol, Lum_FUV_vol, Lum_LW_vol, Lum_EUV1_vol, Lum_EUV2_vol, Lum_EUV3_vol,

        dN_dt_opt, dN_dt_FUV, dN_dt_LW, dN_dt_EUV1, dN_dt_EUV2, dN_dt_EUV3, dN_dt_LyC,

        js_IR, js_opt, js_FUV, js_LW, js_EUV1, js_EUV2, js_EUV3, ρ_star,

        Δρu_stellar_r, Δρu_stellar_Z, 
        Δρu_IR_r,      Δρu_IR_Z, 
        f_stellar_r,   f_stellar_Z,
        f_IR_r,        f_IR_Z,

        f_esc_LyC, f_esc_LW,

        M_F_dir, M_F_dir_radial_net, M_F_dir_radial_out, M_F_dir_antigravity,

        M_F_IR, M_F_IR_radial_net, M_F_IR_radial_out, M_F_IR_antigravity,

        OTS, FluxMethod
    )

end


""" This function sets up the stellar (+ IR) RT arrays and state,
    for use in the realistic RHD simulations, for a (fixed)
    star cluster of a given mass M_StarCluster (in M⊙). """
function GenerateStarClusterRTArrays( M_StarCluster::Float64, 
                                         Zstar_Zsun::Float64,
                                                 ΔV::AbstractArray{Float64, 2},
                                                  r::AbstractArray{Float64, 1},
                                                  Z::AbstractArray{Float64, 1},
                                       Zcluster_pos::Float64,
                                  Rhalf_StarCluster::Float64,      # Half-mass radius of the star cluster (in cm)
                                                  c̃::Float64,
                                              c̃_min::Float64 = 100 * kms,
                                                OTS::Bool = false,  # On-the-spot approximation (if true) or not (if false)
                                                HLL::Bool = true
                                      )
   
    # Grid size: 
    N_r, N_Z = size( ΔV )
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z

    # Convenient arrays: 
    zeros_c()   = zeros(Float64, N_r,   N_Z)
    zeros_bc()  = zeros(Float64, Nbands_RT, N_r,   N_Z)
    zeros_brf() = zeros(Float64, Nbands_RT, N_r+1, N_Z)   # For r-interface fluxes 
    zeros_bzf() = zeros(Float64, Nbands_RT, N_r,   N_Z+1) # For Z-interface fluxes 

    # Intensities (latest & old) for all bands: 
    J = zeros_bc(); J_old = zeros_bc()

    # H ≡ ∫dΩ I n̂ / 4π arrays for all bands:
    H_r     = zeros_bc(); H_Z     = zeros_bc()
    H_r_old = zeros_bc(); H_Z_old = zeros_bc()

    # ϕϕ-component of radiation pressure tensor K = D × J at interfaces, 
    # needed for geometric source term:
    K_ϕϕ_iphalf = zeros_bc()
    K_ϕϕ_imhalf = zeros_bc()

    # Interface fluxes for all bands: 
    F_r    = zeros_brf(); F_Z    = zeros_bzf()
    F_r_Hr = zeros_brf(); F_Z_Hr = zeros_bzf()
    F_r_HZ = zeros_brf(); F_Z_HZ = zeros_bzf()

    # Flux opacities for IR (can be non-zero) & stellar bands (always set to zero,
    # since no diffusion fix is applied):
    α_F  = zeros_bc()

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

    # Get photon emission rates (in s⁻¹) & luminosities (in erg s⁻¹) 
    # from the star cluster, for all bands except IR:
    dN_dt_opt, dN_dt_FUV, dN_dt_LW, 
    dN_dt_EUV1, dN_dt_EUV2, dN_dt_EUV3,
    Lum_opt, Lum_FUV, Lum_LW, 
    Lum_EUV1, Lum_EUV2, Lum_EUV3 = ClusterEmissionRates( M_StarCluster, Zstar_Zsun )

    # Initialize IR luminosity as zero:
    Lum_IR    = 0.0 

    # Get bolometric luminosity of the star cluster:
    Lum_bol   = Lum_opt + Lum_FUV + Lum_LW + Lum_EUV1 + Lum_EUV2 + Lum_EUV3

    # Get ionizing photon emission rate from the star cluster:
    dN_dt_LyC = dN_dt_EUV1 + dN_dt_EUV2 + dN_dt_EUV3

    # For a Gaussian density profile, ρ_star ∝ exp[ - (r² + Z²)/2σ² ], 
    # the half-mass radius R_half, containing half the cluster mass, is 
    # R_half ≈ 1.538 σ. This can be found using (with R being spherical radius):
    #
    # M(< R)/Mtot = erf(R/√2σ) - √(2/π)(R/σ) exp[- (R/σ)²/2 ]
    σstar = Rhalf_StarCluster / 1.538
    
    # Compute weight factors f_ij that determine the emissivity 
    # and stellar mass in a given cell (i,j):
    ComputeStellar_fij!( fstar_ij, r, Z, Zcluster_pos, σstar )

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
            ρ_star[i,j] = fstar_ij[i,j] * (M_StarCluster * Msun) / ΔV[i,j]
        end
    end

    # Initialize LyC and LW escape fractions:
    f_esc_LyC = 0.0
    f_esc_LW  = 0.0 

    # Initialize direct radiation pressure 'force multipliers':
    M_F_dir             = 0.0
    M_F_dir_radial_net  = 0.0
    M_F_dir_radial_out  = 0.0
    M_F_dir_antigravity = 0.0

    # Initialize IR radiation pressure force multipliers:
    M_F_IR              = 0.0
    M_F_IR_radial_net   = 0.0
    M_F_IR_radial_out   = 0.0
    M_F_IR_antigravity  = 0.0

    # Flux method for M1 transport: 
    if HLL
        FluxMethod = Val(:HLL)
    else
        FluxMethod = Val(:GLF)
    end

    return StellarRTArrays( 
        c̃, c̃_min,

        J,          J_old,  
        H_r,        H_Z, 
        H_r_old,    H_Z_old,

        K_ϕϕ_iphalf,   K_ϕϕ_imhalf,

        F_r,      F_Z,   
        F_r_Hr,   F_Z_Hr,   
        F_r_HZ,   F_Z_HZ,

        α_F,

        Lum_IR, Lum_opt, Lum_FUV, Lum_LW, Lum_EUV1, Lum_EUV2, Lum_EUV3, Lum_bol,

        Lum_opt_vol, Lum_FUV_vol, Lum_LW_vol, Lum_EUV1_vol, Lum_EUV2_vol, Lum_EUV3_vol,

        dN_dt_opt, dN_dt_FUV, dN_dt_LW, dN_dt_EUV1, dN_dt_EUV2, dN_dt_EUV3, dN_dt_LyC,

        js_IR, js_opt, js_FUV, js_LW, js_EUV1, js_EUV2, js_EUV3, ρ_star,

        Δρu_stellar_r, Δρu_stellar_Z, 
        Δρu_IR_r,      Δρu_IR_Z, 
        f_stellar_r,   f_stellar_Z,
        f_IR_r,        f_IR_Z,

        f_esc_LyC, f_esc_LW,

        M_F_dir, M_F_dir_radial_net, M_F_dir_radial_out, M_F_dir_antigravity,

        M_F_IR, M_F_IR_radial_net, M_F_IR_radial_out, M_F_IR_antigravity,

        OTS, FluxMethod 
    )

end


""" This function sets up the stellar (+ IR) RT arrays and state,
    for use in the idealized RHD simulation tests (e.g. R-type & D-type 
    fronts, or radiaton pressure-driven shells). """
function GenerateStellarRTArrays_Idealized( dN_dt_EUV1::Float64, 
                                            dN_dt_EUV2::Float64, 
                                            dN_dt_EUV3::Float64, 
                                                E_EUV1::Float64, 
                                                E_EUV2::Float64,
                                                E_EUV3::Float64,
                                                    ΔV::AbstractArray{Float64, 2},
                                                     r::AbstractArray{Float64, 1},
                                                     Z::AbstractArray{Float64, 1},
                                             Zstar_pos::Float64,
                                            σstar_cell::Float64,
                                                     c̃::Float64,
                                                 c̃_min::Float64 = 100 * kms,
                                                   OTS::Bool    = false,  # On-the-spot approximation (if true) or not (if false)
                                                   HLL::Bool    = true
                                           )
   
    # Grid size: 
    N_r, N_Z = size( ΔV )
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z

    # Convenient arrays: 
    zeros_c()   = zeros(Float64, N_r,   N_Z)
    zeros_bc()  = zeros(Float64, Nbands_RT, N_r,   N_Z)
    zeros_brf() = zeros(Float64, Nbands_RT, N_r+1, N_Z)   # For r-interface fluxes 
    zeros_bzf() = zeros(Float64, Nbands_RT, N_r,   N_Z+1) # For Z-interface fluxes 

    # Intensities (latest & old) for all bands: 
    J = zeros_bc(); J_old = zeros_bc()

    # H ≡ ∫dΩ I n̂ / 4π arrays for all bands:
    H_r     = zeros_bc(); H_Z     = zeros_bc()
    H_r_old = zeros_bc(); H_Z_old = zeros_bc()

    # ϕϕ-component of radiation pressure tensor K = D × J at interfaces, 
    # needed for geometric source term:
    K_ϕϕ_iphalf = zeros_bc()
    K_ϕϕ_imhalf = zeros_bc()

    # Interface fluxes for all bands: 
    F_r    = zeros_brf(); F_Z    = zeros_bzf()
    F_r_Hr = zeros_brf(); F_Z_Hr = zeros_bzf()
    F_r_HZ = zeros_brf(); F_Z_HZ = zeros_bzf()

    # Flux opacities for IR (can be non-zero) & stellar bands (always set to zero,
    # since no diffusion fix is applied):
    α_F  = zeros_bc()

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
    Lum_IR        = 0.0                  # IR dust emission is initialized to zero 
    Lum_opt       = 0.0
    Lum_FUV       = 0.0
    Lum_LW        = 0.0
    Lum_EUV1      = dN_dt_EUV1 * E_EUV1
    Lum_EUV2      = dN_dt_EUV2 * E_EUV2
    Lum_EUV3      = dN_dt_EUV3 * E_EUV3

    # Get bolometric stellar luminosity:
    Lum_bol       = Lum_opt + Lum_FUV + Lum_LW + Lum_EUV1 + Lum_EUV2 + Lum_EUV3

    # Get photon emission rates in all bands:
    dN_dt_opt     = 0.0
    dN_dt_FUV     = 0.0
    dN_dt_LW      = 0.0
    dN_dt_LyC     = dN_dt_EUV1 + dN_dt_EUV2 + dN_dt_EUV3
    
    # Compute weight factors f_ij that determine the emissivity 
    # and stellar mass in a given cell (i,j):
    σstar = σstar_cell * max( minimum(diff(r)), minimum(diff(Z)) )   # σ determining the Gaussian size of the smeared 'star' 
    ComputeStellar_fij!( fstar_ij, r, Z, Zstar_pos, σstar )          # (minimum of diff, to ensure we take cell widths from uniform grid region)

    # Get emissivities in all bands, for all cells. Weighted 
    # so that the total luminosity over all (infinite) space is 
    # the stellar luminosity for the band in question:
    @inbounds for j in dN_Z
        for i in dN_r 
            js_opt[i,j]      = fstar_ij[i,j] * Lum_opt  / (Four_π * ΔV[i,j])
            js_FUV[i,j]      = fstar_ij[i,j] * Lum_FUV  / (Four_π * ΔV[i,j])
            js_LW[i,j]       = fstar_ij[i,j] * Lum_LW   / (Four_π * ΔV[i,j])
            js_EUV1[i,j]     = fstar_ij[i,j] * Lum_EUV1 / (Four_π * ΔV[i,j])
            js_EUV2[i,j]     = fstar_ij[i,j] * Lum_EUV2 / (Four_π * ΔV[i,j])
            js_EUV3[i,j]     = fstar_ij[i,j] * Lum_EUV3 / (Four_π * ΔV[i,j])
        end
    end

    # Get stellar luminosities in actual simulated volume (will be 0.5 * L for 
    # reflective BCs, but we compute it explicitly here to be sure):
    fstar_ij_sum    = sum( fstar_ij )
    Lum_opt_vol     = Lum_opt  * fstar_ij_sum
    Lum_FUV_vol     = Lum_FUV  * fstar_ij_sum
    Lum_LW_vol      = Lum_LW   * fstar_ij_sum
    Lum_EUV1_vol    = Lum_EUV1 * fstar_ij_sum
    Lum_EUV2_vol    = Lum_EUV2 * fstar_ij_sum
    Lum_EUV3_vol    = Lum_EUV3 * fstar_ij_sum

    # In idealized tests, there is no gravity, so set the stellar 
    # density to 0:
    @inbounds for j in dN_Z
        for i in dN_r 
            ρ_star[i,j] = 0.0
        end
    end

    # Initialize LyC and LW escape fractions:
    f_esc_LyC = 0.0
    f_esc_LW  = 0.0 

    # Initialize direct radiation pressure 'force multipliers':
    M_F_dir             = 0.0
    M_F_dir_radial_net  = 0.0
    M_F_dir_radial_out  = 0.0
    M_F_dir_antigravity = 0.0

    # Initialize IR radiation pressure force multipliers:
    M_F_IR              = 0.0
    M_F_IR_radial_net   = 0.0
    M_F_IR_radial_out   = 0.0
    M_F_IR_antigravity  = 0.0

    # Flux method for M1 transport: 
    if HLL
        FluxMethod = Val(:HLL)
    else
        FluxMethod = Val(:GLF)
    end

    return StellarRTArrays( 
        c̃, c̃_min,

        J,          J_old,  
        H_r,        H_Z, 
        H_r_old,    H_Z_old,

        K_ϕϕ_iphalf,   K_ϕϕ_imhalf,

        F_r,      F_Z,   
        F_r_Hr,   F_Z_Hr,   
        F_r_HZ,   F_Z_HZ,

        α_F,

        Lum_IR, Lum_opt, Lum_FUV, Lum_LW, Lum_EUV1, Lum_EUV2, Lum_EUV3, Lum_bol,

        Lum_opt_vol, Lum_FUV_vol, Lum_LW_vol, Lum_EUV1_vol, Lum_EUV2_vol, Lum_EUV3_vol,

        dN_dt_opt, dN_dt_FUV, dN_dt_LW, dN_dt_EUV1, dN_dt_EUV2, dN_dt_EUV3, dN_dt_LyC,

        js_IR, js_opt, js_FUV, js_LW, js_EUV1, js_EUV2, js_EUV3, ρ_star,

        Δρu_stellar_r, Δρu_stellar_Z, 
        Δρu_IR_r,      Δρu_IR_Z, 
        f_stellar_r,   f_stellar_Z,
        f_IR_r,        f_IR_Z,

        f_esc_LyC, f_esc_LW,

        M_F_dir, M_F_dir_radial_net, M_F_dir_radial_out, M_F_dir_antigravity,

        M_F_IR, M_F_IR_radial_net, M_F_IR_radial_out, M_F_IR_antigravity,

        OTS, FluxMethod 
    )

end
