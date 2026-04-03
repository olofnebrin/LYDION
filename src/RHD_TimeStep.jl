
""" FILE:           RHD_TimeStep.jl

    PURPOSE:        This file contains the code that determines the global time-step
                    in RHD simulations. Takes into account the CFL time-step, gravity, 
                    and radiation pressure. 
    
"""

using .Structs: LyαArrays, StellarRTArrays, ChemistryState, 
       HydrodynamicsState, DustState, GravityState, 
       GeometryCylindrical, GeometryCartesian

export Compute_RHD_TimeStep


""" This function selects an appropriate global time-step for stability 
    and accuracy in RHD simulations. It takes into account the CFL stability 
    criterion for the gas & dust, as well as acceleration by radiation pressure 
    and gravity. 
    
    The code underlying the CFL and gravity-limited time-step can be found in 
    Hydrodynamics.jl. The code for the radiation pressure-limited time-step can 
    found in PhotoThermoChemistry.jl. """
function Compute_RHD_TimeStep( Δt_Hydro_old::Float64, 
                               Δt_Hydro_max::Float64,
                                      Hydro::HydrodynamicsState, 
                                  GravState::Union{GravityState, Nothing},
                                         RT::Union{StellarRTArrays, Nothing},
                                   LyαState::Union{LyαArrays, Nothing},
                                  ChemState::ChemistryState,
                                       Dust::DustState,
                                   Geometry::Union{GeometryCylindrical, GeometryCartesian},
                                    Gravity::Bool,
                                RadPressure::Bool
                              )

    # Unpack geometry arrays:
    if Geometry isa GeometryCylindrical
        r = Geometry.r 
        Z = Geometry.Z 
    else
        # Assumed Cartesian otherwise:
        r = Geometry.X 
        Z = Geometry.Y 
    end

    # Unpack gas & dust arrays:
    ρ    = Hydro.ρ
    ρd   = Hydro.ρd 

    # Get the minimum of the gas & dust CFL time-step:
    Δt_CFL = GetCFLTimeStep( Hydro, Geometry )  

    # Get the gravity-limited time-step, if gravity is turned on:
    if !isnothing(GravState)  &&  Gravity 
        Δt_Grav = GetGravityTimeStep( r, Z, GravState.∂Φ∂r, GravState.∂Φ∂Z, ρ, Hydro.ρdtot )      
    else
        # Ignore limit in absence of gravity:
        Δt_Grav = Inf    
    end

    # Get the radiation pressure-limited time-step, if radiation pressure feedback 
    # is turned on:
    if !isnothing(RT)  &&  RadPressure 
        # Compute acceleration time-step limits for gas-only, dust-only, and the combined 
        # forces on the dust+gas (relevant if coupled):
        Δt_gasdust = GetRadPressureTimeStep_GasDustAcceleration( r, Z, ρ, ρd, RT, LyαState, ChemState, Dust )
        Δt_gas     = GetRadPressureTimeStep_GasAcceleration( r, Z, ρ, RT, LyαState, ChemState )
        Δt_dust    = GetRadPressureTimeStep_DustAcceleration( r, Z, ρd, RT, LyαState, Dust )

        # Get minimum radiation pressure acceleration time-scale:
        Δt_RP      = min( Δt_gasdust, Δt_gas, Δt_dust )
    else
        # Ignore limit in absence of radiation pressure feedback:
        Δt_RP      = Inf 
    end

    # Set hydro time-step:
    Δt_Hydro = min( 1.25 * Δt_Hydro_old, Δt_CFL, Δt_RP, Δt_Grav, Δt_Hydro_max )


    return Δt_Hydro 
end

