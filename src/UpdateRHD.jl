module UpdateRHD

""" MODULE:         UpdateRHD

    PURPOSE:        This module contains the basic code for performing a full time-step 
                    update for hydrodynamics + thermochemistry + radiative transfer. Includes 
                    both full RHD with Lyα, and RHD with or without Lyα. 

                    The method used is an operator split one, where the hydro update is INSERT.
             
    REFERENCES:     
    
    """

__precompile__()

include( "./ConstantsAndFunctions.jl" ) 
include( "./Hydrodynamics.jl" )
include( "./SpatialGrid.jl" )
include( "./PoissonGravitySolver.jl" )
include( "PhotoChemistryAndCooling/InitializeChemistryStruct.jl" ) 
using .ConstantsAndFunctions         # Useful constants and functions 
using .Hydrodynamics                 # Hydrodynamics + gravity update & kick 
using .SpatialGrid                   # For using the spatial grid struct
using .InitializeChemistryStruct     # For using the chemistry struct/arrays
using .PoissonGravitySolver          # Only needed here to use the structs for gravity 
using  Base.Threads                  # For CPU acceleration


export UpdateRHD!


""" This function performs a full RHD update over a hydro time-step Δt_Hydro. """
function UpdateRHD!( Δt_Hydro::Float64,
                   hydrostate::HydrodynamicsState{Float64},
                    chemstate::ChemistryState{Float64},
                     geometry::GeometryCylindrical{Float64},
                    gravstate::GravityState{Float64},
                   gravcoeffs::GravityCoeffs{Float64},
                    ghostgrid::GhostPaddedGrid{Float64},
                       rfirst::Bool = true, 
                      Gravity::Bool = true,
                  RadPressure::Bool = true 
                     )

    # Geometry arrays:
    inv_ΔV      = geometry.inv_ΔV
    A_r         = geometry.A_r
    A_Z         = geometry.A_Z
    inv_r_cell  = geometry.inv_r_cell
    rCoM        = geometry.rCoM
    ZCoM        = geometry.ZCoM

    # Hydro state arrays:
    ρ_new       = hydrostate.ρ_new;    ρu_r_new    = hydrostate.ρu_X_new 
    ρu_Z_new    = hydrostate.ρu_Y_new; E_new       = hydrostate.E_new 
    ρ           = hydrostate.ρ;        ρu_r        = hydrostate.ρu_X 
    ρu_Z        = hydrostate.ρu_Y;     E           = hydrostate.E
    ρ0          = hydrostate.ρ0;       ρu_r0       = hydrostate.ρu_X0 
    ρu_Z0       = hydrostate.ρu_Y0;    E0          = hydrostate.E0
    u_r         = hydrostate.u_X;      u_Z         = hydrostate.u_Y 
    P           = hydrostate.P;        e           = hydrostate.e 
    γad         = hydrostate.γad 

    # r-direction interface hydro state & fluxes:
    ρ_L_r    = hydrostate.ρ_L_X;    ρ_R_r    = hydrostate.ρ_R_X
    u_r_L_r  = hydrostate.u_X_L_X;  u_r_R_r  = hydrostate.u_X_R_X 
    u_Z_L_r  = hydrostate.u_Y_L_X;  u_Z_R_r  = hydrostate.u_Y_R_X 
    P_L_r    = hydrostate.P_L_X;    P_R_r    = hydrostate.P_R_X 
    Fρ_r     = hydrostate.F_ρ_X;    Fρur_r   = hydrostate.F_ρu_X_X
    Fρuz_r   = hydrostate.F_ρu_Y_X; FE_r     = hydrostate.F_E_X

    # Z-direction interface hydro state & fluxes:
    ρ_L_Z    = hydrostate.ρ_L_Y;    ρ_R_Z    = hydrostate.ρ_R_Y
    u_Z_L_Z  = hydrostate.u_Y_L_Y;  u_Z_R_Z  = hydrostate.u_Y_R_Y
    u_r_L_Z  = hydrostate.u_X_L_Y;  u_r_R_Z  = hydrostate.u_X_R_Y 
    P_L_Z    = hydrostate.P_L_Y;    P_R_Z    = hydrostate.P_R_Y 
    Fρ_Z     = hydrostate.F_ρ_Y;    Fρur_Z   = hydrostate.F_ρu_X_Y 
    Fρuz_Z   = hydrostate.F_ρu_Y_Y; FE_Z     = hydrostate.F_E_Y

    # Ghost-padded arrays:
    ρg_old   = hydrostate.ρg_old;   u_rg_old = hydrostate.u_Xg_old
    u_Zg_old = hydrostate.u_Yg_old; Pg_old   = hydrostate.Pg_old 
    rg       = ghostgrid.Xg;        rCoMg    = ghostgrid.XCoMg 
    Zg       = ghostgrid.Yg;        ZCoMg    = ghostgrid.YCoMg

    # Grid size & intervals:
    N_r, N_Z    = size( ρ )
    dN_r        = 1:N_r 
    dN_Z        = 1:N_Z 

    ###############################################################
    #######                                                 #######
    #######      H Y D R O / D U S T - D Y N A M I C S      #######
    #######                                                 #######
    ###############################################################

    # Update hydro variables by advection:
    UpdateHydro!( ρ_new,  ρu_r_new, ρu_Z_new, E_new,
                  ρ,      ρu_r,     ρu_Z,     E,
                  ρ0,     ρu_r0,    ρu_Z0,    E0,
                  γad,    u_r,      u_Z,
                  inv_ΔV, A_r,      A_Z,      Δt_Hydro,

                  ρ_L_r, ρ_R_r, u_r_L_r, u_r_R_r, u_Z_L_r, u_Z_R_r, P_L_r, P_R_r,
                  Fρ_r, Fρur_r, Fρuz_r, FE_r,
                  ρ_L_Z, ρ_R_Z, u_Z_L_Z, u_Z_R_Z, u_r_L_Z, u_r_R_Z, P_L_Z, P_R_Z,
                  Fρ_Z, Fρur_Z, Fρuz_Z, FE_Z,
                  P, ρg_old, u_rg_old, u_Zg_old, 
                  Pg_old, 
                  rg, rCoMg, Zg,  ZCoMg, 

                  Val(:Cylindrical), inv_r_cell, rfirst ) 

    # Gravity solver + kick update:
    if Gravity 
        GravitySolveAndKick!( Δt_Hydro, ρu_r, ρu_r0, ρu_Z, ρu_Z0,  
                              ρ, ρ0, E, 
                              gravstate.Φ, gravstate.Φ_half, gravstate.Φ_old, 
                              gravstate.∂Φ∂r, gravstate.∂Φ∂Z, 
                              gravcoeffs.RPoiss_ijp, gravcoeffs.RPoiss_ij, gravcoeffs.RPoiss_ijm, 
                              gravcoeffs.ZPoiss_ijp, gravcoeffs.ZPoiss_ij, gravcoeffs.ZPoiss_ijm,
                              gravcoeffs.Δτ_max, 
                              gravcoeffs.Δτ_min, 
                              rCoM, ZCoM )
    end

    # Advect all gas fractions (xHII, xHeII etc):
    AdvectGasFractions!( Δt_Hydro, chemstate, hydrostate, geometry )
    
    # Initialize the internal energy density for the thermochemistry 
    # sub-cycling: 
    @threads for j in dN_Z 
        @simd for i in dN_r
            @inbounds begin 
                e[i,j] = P[i,j] / ( γad[i,j] - 1 )
            end
        end
    end 

    ###############################################################
    #######                                                 #######
    #######  S T E L L A R - R T  +  S U B - C Y C L I N G  #######
    #######                                                 #######
    ###############################################################

    # Stellar RT + thermochemistry sub-cycling (TODO):


    # Update gas temperature (+ adiabatic index):
    @threads for j in dN_Z 
        @simd for i in dN_r
            @inbounds begin 
                # Compute the (updated) mean molecular weight & adiabatic index:
                μ        = Getμ( xHII[i,j], xH2[i,j], xHeII[i,j], xHeIII[i,j] )
                γad[i,j] = Getγad( xHII[i,j], xH2[i,j], xHeII[i,j], xHeIII[i,j] )

                # Get the latest temperature:
                T[i,j]   = e[i,j] * μ * (γad[i,j] - 1) * mH / ( ρ[i,j] * kB )
            end
        end
    end 


    ###############################################################
    #######                                                 #######
    #######                 F E E D B A C K                 #######
    #######                                                 #######
    ###############################################################
    

end



end # End of module 