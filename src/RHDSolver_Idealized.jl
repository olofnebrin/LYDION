
module RHDSolver_Idealized
""" MODULE:  RHDSolver_DTypeRType

    PURPOSE: This module is similar to RHDSolver.jl, but specialized for idealized tests,
             e.g. D-type and R-type ionization front expansions, or radiation pressure-driven 
             shells. Gravity is turned off. 

"""

__precompile__()


include( "./Structs.jl" )
include( "./ConstantsAndFunctions.jl" )
using .Structs
using .ConstantsAndFunctions
using LinearAlgebra           # For the tridiagonal solver 
BLAS.set_num_threads(1)       # Ensure the tridiagonal solver is not threaded (we thread over spatial cells)

# Some basic underlying files:
include( "./WENO3.jl" )
include( "./M1Closure.jl" ) 
include( "./AdvectJandH.jl" )
include( "./SpatialGrid.jl" )
include( "./PoissonGravitySolver.jl" )
include( "./Hydrodynamics.jl" )

include( "./Dust.jl" )
include( "./DustGasCoupling.jl" )
include( "PhotoChemistryAndCooling/InitializeChemistryStruct.jl" ) 
include( "PhotoChemistryAndCooling/PhotoThermoChemistry.jl" )
include( "PhotoChemistryAndCooling/StellarIR_Diagnostics.jl" )
include( "./AdvectScalars.jl" )
include( "./Turbulence.jl" )
include( "./SaveHDF5Data_RHDTests.jl" )

using  Base.Threads              # For CPU acceleration
using  ProgressMeter             # To show time step progress
using  Dates, Printf

export SetupRHDsimulation, RestartRHDsimulation


""" This function performs a full RHD update over a hydro time-step Δt_Hydro. """
function UpdateRHD!( Δt_Hydro::Float64,
                   hydrostate::HydrodynamicsState{Float64},
                    chemstate::ChemistryState{Float64},
                    duststate::DustState,
                     geometry::GeometryCylindrical{Float64},
                    ghostgrid::GhostPaddedGrid{Float64},
                           RT::StellarRTArrays,
                  RadPressure::Bool    = true,
                   IRfeedback::Bool    = true,
                            z::Float64 = 0.0,
                        Hydro::Bool    = false,
               CoolingHeating::Bool    = false
                     )

    # Main geometry arrays:
    ΔV          = geometry.ΔV
    inv_ΔV      = geometry.inv_ΔV
    A_r         = geometry.A_r
    A_Z         = geometry.A_Z
    inv_r_cell  = geometry.inv_r_cell
    r           = geometry.r 
    rCoM        = geometry.rCoM
    Z           = geometry.Z
    ZCoM        = geometry.ZCoM

    # Weights for converting between cell center & cell-average values of 
    # conservatives & primitives:
    w_i_m1_r    = geometry.w_i_m1_r;     w_j_m1_Z     = geometry.w_j_m1_Z
    w_i_0_r     = geometry.w_i_0_r;      w_j_0_Z      = geometry.w_j_0_Z
    w_i_p1_r    = geometry.w_i_p1_r;     w_j_p1_Z     = geometry.w_j_p1_Z

    q_i_m1_r    = geometry.q_i_m1_r;     q_j_m1_Z     = geometry.q_j_m1_Z
    q_i_0_r     = geometry.q_i_0_r;      q_j_0_Z      = geometry.q_j_0_Z
    q_i_p1_r    = geometry.q_i_p1_r;     q_j_p1_Z     = geometry.q_j_p1_Z

    # WENO3 weights for interface reconstruction:
    w_plus_i_p1 = geometry.w_plus_i_p1;  w_minus_i_p1 = geometry.w_minus_i_p1   
    w_plus_j_p1 = geometry.w_plus_j_p1;  w_minus_j_p1 = geometry.w_minus_j_p1

    # Some hydro state arrays (cell-averaged):
    ρ           = hydrostate.ρ;        ρu_r        = hydrostate.ρu_X 
    ρu_Z        = hydrostate.ρu_Y;     E           = hydrostate.E
    u_r         = hydrostate.u_X;      u_Z         = hydrostate.u_Y 
    P           = hydrostate.P;        e           = hydrostate.e 
    T           = hydrostate.T;        γad         = hydrostate.γad 

    # r-direction interface velocities:
    u_r_L_r  = hydrostate.u_X_L_X;  u_r_R_r  = hydrostate.u_X_R_X 
    u_Z_L_r  = hydrostate.u_Y_L_X;  u_Z_R_r  = hydrostate.u_Y_R_X 

    # Z-direction interface velocities:
    u_Z_L_Z  = hydrostate.u_Y_L_Y;  u_Z_R_Z  = hydrostate.u_Y_R_Y
    u_r_L_Z  = hydrostate.u_X_L_Y;  u_r_R_Z  = hydrostate.u_X_R_Y 

    # Contact velocities:
    u_r_star_r = hydrostate.u_X_star_X
    u_Z_star_Z = hydrostate.u_Y_star_Y

    # Arrays related to ∇u (only needed for RHD):
    ∂u_r_∂r    = hydrostate.∂u_X_∂X
    ∂u_r_∂Z    = hydrostate.∂u_X_∂Y
    ∂u_Z_∂r    = hydrostate.∂u_Y_∂X
    ∂u_Z_∂Z    = hydrostate.∂u_Y_∂Y
    u_r_over_r = hydrostate.u_r_over_r

    # Dust arrays:
    ρd         = hydrostate.ρd
    ρdud_r     = hydrostate.ρdud_X
    ρdud_Z     = hydrostate.ρdud_Y
    ud_r       = hydrostate.ud_X
    ud_Z       = hydrostate.ud_Y

    # Boundary conditions:
    BC_Z_bot   = hydrostate.BC_Y_left

    # Grid size & intervals:
    N_r, N_Z    = size( ρ )
    dN_r        = 1:N_r 
    dN_Z        = 1:N_Z 

    ###############################################################
    #######                                                 #######
    #######      H Y D R O / D U S T - D Y N A M I C S      #######
    #######                                                 #######
    ###############################################################

    if Hydro
        # Prepare passive scalars before we advect them:
        PreparePassiveScalars!( hydrostate, chemstate )

        # Update hydro & dust variables & passive scalars by advection:
        UpdateHydro!( Δt_Hydro, hydrostate, ghostgrid, inv_ΔV, A_r, A_Z,

                      w_i_m1_r, w_i_0_r, w_i_p1_r, w_j_m1_Z, w_j_0_Z, w_j_p1_Z, 
                      q_i_m1_r, q_i_0_r, q_i_p1_r, q_j_m1_Z, q_j_0_Z, q_j_p1_Z,  

                      w_plus_i_p1, w_minus_i_p1,   
                      w_plus_j_p1, w_minus_j_p1, 

                      Val(:Cylindrical), inv_r_cell )  

        # Update total mass in simulation volume (for diagnostic purposes):
        UpdateTotalMass!( hydrostate, ΔV )

        # Recover abundance fractions from passive scalars (needed before sub-cycling):
        RecoverChemistryFromPassiveScalars!( hydrostate, chemstate )
    end

    # Update primitives to reflect latest conservatives:
    @threads for j in dN_Z 
        @inbounds @simd for i in dN_r
            # Recover primitives at geometric cell centers, from the 
            # conservative variables at the cell centers:
            u_r[i,j], u_Z[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_r[i,j], ρu_Z[i,j], E[i,j], γad[i,j] )

            # Ensure positive pressure:
            P[i,j] = max( P[i,j], P_floor )
        end
    end
    
    # Initialize the internal energy density & adiabatic index (can change after gas fraction advection)
    # for the thermochemistry sub-cycling: 
    @threads for j in dN_Z 
        @simd for i in dN_r
            @inbounds begin 
                γad[i,j] = Getγad( chemstate.xHII[i,j],  chemstate.xH2[i,j], 
                                   chemstate.xHeII[i,j], chemstate.xHeIII[i,j] )
                e[i,j]   = P[i,j] / ( γad[i,j] - 1 )
            end
        end
    end 

    # Update the ∇u components, needed for the velocity terms in the RT:
    Update_∇u_RHD!( ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                    u_r_L_r, u_r_R_r, u_r_L_Z, u_r_R_Z, 
                    u_Z_L_r, u_Z_R_r, u_Z_L_Z, u_Z_R_Z,
                    r, Z, inv_r_cell )

    ###############################################################
    #######                                                 #######
    #######          R T  +  S U B - C Y C L I N G          #######
    #######                                                 #######
    ###############################################################
    # Below we sub-cycle the RT, chemistry, and heating/cooling. 
    # We split the hydro step into (potentially) smaller RT CFL time-steps,
    # and for each RT CFL time-step we sub-cycle chemistry and heating/cooling.

    # Before sub-cycling, reset momentum change from radiation pressure to zero:
    Reset_Δρu!( RT.Δρu_stellar_r,   RT.Δρu_stellar_Z )
    Reset_Δρu!( RT.Δρu_IR_r,        RT.Δρu_IR_Z )

    # Determine an updated reduced speed of light for the radiative transfer:
    RT.c̃  = FindRSL_StellarRT( RT.c̃, u_r, u_Z, ud_r, ud_Z, P, ρ, γad )


    t_RT  = 0.0 # Initialize the RT time 
    while t_RT < Δt_Hydro
        # Set maximum RT transport time-step:
        Δt_RT_max = max( Δt_Hydro - t_RT, 0.0 )
 
        # Stellar RT + Lyα RT + thermochemistry sub-cycling:
        Δt_RT = UpdateRTandThermoChemistry!( Δt_RT_max, 
                                             RT,
                                             hydrostate,
                                             chemstate,
                                             duststate,
                                             z, CoolingHeating,
                                             geometry,
                                             u_r_star_r, u_Z_star_Z, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r, 
                                             nothing, false, 
                                             RadPressure )

        # Update time:
        t_RT += Δt_RT

    end

    ###############################################################
    #######                                                 #######
    #######    F E E D B A C K  &  D I A G N O S T I C S    #######
    #######                                                 #######
    ###############################################################

    if RadPressure && Hydro
        # If radiation pressure feedback is turned on, deposit the momentum 
        # computed during the photo-thermochemistry sub-cycling:

        inv_Δt_Hydro = 1 / Δt_Hydro  # Convenient constant
        @threads for j in dN_Z
            @simd for i in dN_r
                @inbounds begin
                    # Update stellar radiation pressure forces (Lyα case is done below),
                    # mainly for setting the radiation pressure time-step limiter:
                    RT.f_stellar_r[i,j] = RT.Δρu_stellar_r[i,j] * inv_Δt_Hydro
                    RT.f_stellar_Z[i,j] = RT.Δρu_stellar_Z[i,j] * inv_Δt_Hydro

                    # Update IR radiation pressure forces:
                    RT.f_IR_r[i,j]      = RT.Δρu_IR_r[i,j] * inv_Δt_Hydro
                    RT.f_IR_Z[i,j]      = RT.Δρu_IR_Z[i,j] * inv_Δt_Hydro
                end
            end # End of i-loop
        end # End of j-loop 

        # NOTE: Work from radiation pressure is handled below, by updaing E.
    end

    # Update LyC and Lyman-Werner escape fractions (just diagnostics):
    Update_fesc_LyC_and_LW_Cylindrical!( RT, A_r, A_Z, BC_Z_bot )

    # Update IR luminsoity & force diagnostics (just useful for comparing forces after simulation):
    Update_M_F_IR_L_IR!( RT, ΔV, BC_Z_bot )

    # Update direct radiation pressure force diagnostics (just for analyzing after simulation):
    Update_M_F_dir!( RT, ΔV, BC_Z_bot )

    # Update gas temperature (+ adiabatic index), gas energy,
    # and primitives (for computing CFL time-step):
    @threads for j in dN_Z 
        @simd for i in dN_r
            @inbounds begin 
                # Compute the (updated) mean molecular weight & adiabatic index:
                μ         = Getμ( chemstate.xHII[i,j],  chemstate.xH2[i,j], 
                                  chemstate.xHeII[i,j], chemstate.xHeIII[i,j] )

                γad[i,j]  = Getγad( chemstate.xHII[i,j],  chemstate.xH2[i,j], 
                                    chemstate.xHeII[i,j], chemstate.xHeIII[i,j] )

                # Get the latest temperature (from after thermochemistry update):
                T[i,j]    = e[i,j] * μ * (γad[i,j] - 1) * mH / ( ρ[i,j] * kB )

                # Update gas energy to reflect new temperature & momentum from radiation pressure:
                E[i,j]    = e[i,j]  +  ( ρu_r[i,j]^2 + ρu_Z[i,j]^2 ) / ( 2 * ρ[i,j] )


                # Also update gas pressure & velocities, needed for the computing 
                # the CFL time-step:
                P[i,j]    = max( e[i,j] * ( γad[i,j] - 1 ), P_floor )
                u_r[i,j]  = ρu_r[i,j] / ρ[i,j]
                u_Z[i,j]  = ρu_Z[i,j] / ρ[i,j]

                # Same for the dust:
                ud_r[i,j], ud_Z[i,j] = GetDustVelocities( ρd[i,j], ρdud_r[i,j], ρdud_Z[i,j] )

            end
        end # End of i-loop 
    end # End of j-loop 

    # Update gas velocity dispersions (useful diagnostic):
    UpdateGasVelocityDispersion!( hydrostate, ΔV )

    return nothing
end



function SetupRHDsimulation( dNdt_EUV1::Float64,  # EUV1 photon emission rate from star 
                             dNdt_EUV2::Float64,  # EUV2 photon emission rate from star 
                             dNdt_EUV3::Float64,  # EUV2 photon emission rate from star 
                                   DG₀::Float64,  # Dust-to-gas ratio (*not* normalized to Solar value)
                                   nH₀::Float64,  # H density (cm⁻³) inside cloud 
                                nH_ext::Float64,  # H density (cm⁻³) far outside cloud 
                                   Rcl::Float64,  # Radius of 'cloud', i.e. higher density region
                            σstar_cell::Float64,  # Gaussian σ (in units of cell size) that determines extent of spatially smeared star
                              Rmax_lin::Float64,
                            LZ_uniform::Float64,
                                  Rmax::Float64,
                                  Zmax::Float64,
                                    T₀::Float64,  # Temperature within cloud 
                                 T_ext::Float64,  # Temperature outside cloud 
                                  tmax::Float64,
                                 xHII₀::Float64,  # H II fraction inside the cloud 
                                  xH2₀::Float64,  # H₂ fraction inside the cloud 
                                xHeII₀::Float64,  # He II fraction inside the cloud 
                               xHeIII₀::Float64,  # He III fraction inside the cloud 
                                 xDII₀::Float64,  # D II fraction inside the cloud 
                                  xHD₀::Float64,  # HD fraction inside the cloud 
                                  xCI₀::Float64,
                                 xCII₀::Float64,
                                  xCO₀::Float64,
                                  xOI₀::Float64,
                                 xOII₀::Float64,
                              xHII_ext::Float64,  # H II fraction outside the cloud 
                               xH2_ext::Float64,  # H₂ fraction outside the cloud 
                             xHeII_ext::Float64,  # He II fraction outside the cloud 
                            xHeIII_ext::Float64,  # He III fraction outside the cloud 
                              xDII_ext::Float64,  # D II fraction outside the cloud 
                               xHD_ext::Float64,  # HD fraction outside the cloud 
                               xCI_ext::Float64,
                              xCII_ext::Float64,
                               xCO_ext::Float64,
                               xOI_ext::Float64,
                              xOII_ext::Float64,
                             Zgas_Zsun::Float64,
                                     z::Float64,
                                 Hydro::Bool    = true,     # Hydrodynamics (on/off)
                           RadPressure::Bool    = true,
                            IRfeedback::Bool    = true,
                                   OTS::Bool    = false,    # On-the-spot approximation (on if true, off if false)
                     HydrogenAndHelium::Bool    = false,
                             u_RMSturb::Float64 = 0.0,      # Initial turbulent RMS velocity 
                               N_r_lin::Int     = 64,
                               N_Z_lin::Int     = 64,
                               N_r_log::Int     = 64,
                          N_Z_log_side::Int     = 64,
                            BC_r_right::Symbol  = :Outflow,
                             BC_Z_left::Symbol  = :Reflect,
                            BC_Z_right::Symbol  = :Outflow,
                              SaveData::Bool    = true,
                             Nminsteps::Int     = 300,
                             Nmaxsteps::Int     = 100000,
                                Nsaved::Int     = 300,
                            print_info::Bool    = true,
                              FileName::String  = "RHDSim.h5",
                       FileDescription::String  = "RHD simulation test."
                             )

    PrintLYDIONLogo()

    # Initialize the RSL for M1 stellar & Lyα RT (will be updated adaptively each time-step)
    c̃          = 300 * kms  

    # Setup uniform cylindrical grid:
    LZ_uniform = min( LZ_uniform, Zmax )
    geometry   = SetupCylindricalGrid( Rmax, Rmax_lin, Zmax, LZ_uniform, 
                                       N_r_lin, N_r_log, 
                                       N_Z_lin, N_Z_log_side,
                                       BC_Z_left )

    r          = geometry.r
    rCoM       = geometry.rCoM
    Z          = geometry.Z
    ZCoM       = geometry.ZCoM
    ΔV         = geometry.ΔV

    # Grid intervals:
    N_r        = length( rCoM )
    N_Z        = length( ZCoM )
    dN_r       = 1:N_r 
    dN_Z       = 1:N_Z

    # Center star & cloud appropriately:
    if BC_Z_left == :Reflect
        # In this case there is symmetry around Z = 0, so we place the star & cloud at Z = 0:
        Zcl_pos   = 0.0
        Zstar_pos = 0.0
    else
        # Otherwise, we center the star & cloud:
        Zcl_pos   = Z[end] / 2
        Zstar_pos = Zcl_pos
    end


    if print_info
        println("Number of threads used: $(Threads.nthreads())"); flush(stdout)
        println()
        if SaveData
            println("Data will be saved to HDF5 file: $(FileName)."); flush(stdout)
            println()
        end
        print(""" Simulation setup: A central star emitting ionizing photons at 
                  rates dNdt_EUV1 = $(dNdt_EUV1), dNdt_EUV2 = $(dNdt_EUV2), dNdt_EUV3 = $(dNdt_EUV3),
                  in a cloud of initial density nH = $(nH₀) cm⁻³, metallicity Z/Z⊙ = $(Zgas_Zsun),
                  dust-to-gas ratio D/D⊙ = $(DG₀/DGsun). Initial turbulent RMS velocity 
                  is $(u_RMSturb/1e5) km s⁻¹.

                  SPATIAL RESOLUTION: $(N_r) × $(N_Z)
                  
                  PHYSICS SETTINGS:
                  
                  Hydrodynamics:              $(Hydro)
                  On-the-spot approximation:  $(OTS)

                  """)
    end

    ###############################################################
    #######                                                 #######
    #######   H Y D R O  &  T H E R M O C H E M I S T R Y   #######
    #######                                                 #######
    ###############################################################

    if print_info 
        print("\r\033[KInitialising hydro & thermochemistry arrays..."); flush(stdout)
    end

    CoolingHeating = true 

    # Set up hydro state + ghost grid arrays:
    N_scalars::Int        = 17       # Number of passive scalars (fixed from chemistry + dust)
    BC_r_left::Symbol     = :Reflect # Always reflective BCs at inner r-boundary in cylindrical coordinates
    hydrostate, ghostgrid = GenerateHydroArrays( r, rCoM, Z, ZCoM, 5 / 3, N_scalars,
                                                 BC_r_left, BC_r_right,  # Inner & outer r-BCs 
                                                 BC_Z_left, BC_Z_right   # Lower & outer Z-BCs
                                                  )

    # Gas arrays:
    ρ      = hydrostate.ρ
    ρu_r   = hydrostate.ρu_X
    ρu_Z   = hydrostate.ρu_Y
    u_r    = hydrostate.u_X 
    u_Z    = hydrostate.u_Y
    E      = hydrostate.E
    P      = hydrostate.P 
    e      = hydrostate.e 
    T      = hydrostate.T
    γad    = hydrostate.γad

    # Dust arrays:
    ρd     = hydrostate.ρd
    ρdud_r = hydrostate.ρdud_X
    ρdud_Z = hydrostate.ρdud_Y
    ud_r   = hydrostate.ud_X
    ud_Z   = hydrostate.ud_Y

    # Initialize turbulent velocity fluctuations:
    SetupTurbulence!( u_RMSturb, Rcl, Zcl_pos, u_r, u_Z, r, Z, rCoM, ZCoM, ΔV, BC_Z_left )

    # Set up chemistry state struct, containing arrays for all species tracked,
    # e.g. xHII, xH2 etc. :
    chemstate = InitializeChemistryArrays( N_r, N_Z, Zgas_Zsun, HydrogenAndHelium )

    # Cloud density properties:
    ρcl  = max( Get_ρ_From_nH( nH₀, Zgas_Zsun, HydrogenAndHelium ),     10 * ρ_floor )  # Density within cloud 
    ρext = max( Get_ρ_From_nH( nH_ext, Zgas_Zsun, HydrogenAndHelium ),  10 * ρ_floor )  # Density far outside the cloud
    δr   = max( 2 * max( minimum(diff(r)), minimum(diff(Z)) ), 0.05 * Rcl )             # Determines fall-off in density outside cloud

    if ρcl < 1e3 * ρ_floor 
        @warn "The initial cloud density ρcl is only $(ρcl/ρ_floor) times the density floor. Consider making 
               ρcl larger, or take a look at the density floor ρ_floor in the Hydrodynamics.jl file."
    end

    # Initialize cloud properties:
    @threads for j in dN_Z 
        @simd for i in dN_r 
            @inbounds begin

                # Distance from cloud center at the edge of this cell:
                R = sqrt( r[i+1]^2  +   (Z[j+1] - Zcl_pos)^2 )

                # Gas density & chemistry:
                if R < Rcl 
                    # Gas density set to cloud density:
                    ρ[i,j] = ρcl

                    # Initial chemical abundances in the cloud:
                    chemstate.xHII[i,j]   = xHII₀
                    chemstate.xH2[i,j]    = xH2₀
                    chemstate.xHeII[i,j]  = xHeII₀
                    chemstate.xHeIII[i,j] = xHeIII₀
                    chemstate.xDII[i,j]   = xDII₀
                    chemstate.xHD[i,j]    = xHD₀

                    chemstate.xCI[i,j]    = xCI₀
                    chemstate.xCII[i,j]   = xCII₀
                    chemstate.xCO[i,j]    = xCO₀
                    chemstate.xOI[i,j]    = xOI₀
                    chemstate.xOII[i,j]   = xOII₀

                    # Adiabatic index:
                    γad[i,j] = Getγad( xHII₀, xH2₀, xHeII₀, xHeIII₀ )
                else
                    # Gas density much lower outside the cloud:
                    ρ[i,j] = max( ρcl * exp( - ((R - Rcl)/δr)^2 ), ρext )

                    # Initial chemical abundances outside the cloud:
                    chemstate.xHII[i,j]   = xHII_ext
                    chemstate.xH2[i,j]    = xH2_ext
                    chemstate.xHeII[i,j]  = xHeII_ext
                    chemstate.xHeIII[i,j] = xHeIII_ext
                    chemstate.xDII[i,j]   = xDII_ext
                    chemstate.xHD[i,j]    = xHD_ext

                    chemstate.xCI[i,j]    = xCI_ext
                    chemstate.xCII[i,j]   = xCII_ext
                    chemstate.xCO[i,j]    = xCO_ext
                    chemstate.xOI[i,j]    = xOI_ext
                    chemstate.xOII[i,j]   = xOII_ext

                    # Adiabatic index:
                    γad[i,j] = Getγad( chemstate.xHII[i,j], chemstate.xH2[i,j], 
                                       chemstate.xHeII[i,j], chemstate.xHeIII[i,j] )
                end

                # Gas temperature & pressure (just needed to set the right initial CFL time-step):
                if R < Rcl
                    T[i,j] = T₀
                    P[i,j] = ρ[i,j] * kB * T[i,j] / ( Getμ(xHII₀, xH2₀, xHeII₀, xHeIII₀) * mH )
                else
                    T[i,j] = T_ext
                    P[i,j] = ρ[i,j] * kB * T[i,j] / ( Getμ(xHII₀, xH2₀, xHeII₀, xHeIII₀) * mH )
                end

                # Gas momentum density:
                ρu_r[i,j]   = ρ[i,j] * u_r[i,j]
                ρu_Z[i,j]   = ρ[i,j] * u_Z[i,j]

                # Gas internal energy density e = P / (γ - 1) = ρ kB T₀ / ((γ - 1) μ mH):
                e[i,j]      = P[i,j] / ( γad[i,j] - 1 )

                # Energy density of gas, E = e + E_kin:
                E[i,j]      = e[i,j]  +  0.5 * ρ[i,j] * (u_r[i,j]^2 + u_Z[i,j]^2)

                # Dust density:
                ρd[i,j]     = max( DG₀ * ρ[i,j], 10 * ρd_floor )

                # Dust momentum density (assume same velocity as gas initially):
                ρdud_r[i,j] = ρd[i,j] * u_r[i,j]
                ρdud_Z[i,j] = ρd[i,j] * u_Z[i,j]
                
            end
        end # End of i-loop 
    end # End of j-loop


    # Initialize total mass in simulation volume (for diagnostic purposes):
    UpdateTotalMass!( hydrostate, ΔV )

    ###############################################################
    #######                                                 #######
    #######                     D U S T                     #######
    #######                                                 #######
    ###############################################################

    # Initialize struct for dust properties/settings (no dust growth/destruction):
    Td₀       = 30.0
    duststate = SetupDustState( false, N_r, N_Z, Td₀, "DustData.txt", 
                                "DustIRData.txt", true )

    # Sanity/safety check:
    @assert size( duststate.Td ) == size( ρ ) "Dust temperature Td should have the same size 
                                               as spatial grid."


    ###############################################################
    #######                                                 #######
    #######                S T E L L A R  R T               #######
    #######                                                 #######
    ############################################################### 

    # Generate all stellar RT arrays, including J, H, inteface fluxes, 
    # emissivities, and luminosities, for all 7 RT bands:
    RT = GenerateStellarRTArrays_Idealized( dNdt_EUV1, dNdt_EUV2, dNdt_EUV3, ΔV, r, Z, 
                                            Zstar_pos, σstar_cell, c̃, OTS )

    ###############################################################
    #######                                                 #######
    #######           S E T  U P  S A V E  F I L E          #######
    #######                                                 #######
    ###############################################################

    if SaveData

        # Create save file:
        CreateHDF5File( FileName, dNdt_EUV1, dNdt_EUV2, dNdt_EUV3,
                        N_r, N_Z, Nminsteps, Nmaxsteps, Nsaved,  
                        RadPressure, IRfeedback,
                        OTS, z, 
                        tmax, σstar_cell, 
                        Zstar_pos,
                        DG₀, Zgas_Zsun,
                        BC_r_right, BC_Z_left, BC_Z_right,
                        CFL, CFL_RT, c̃,
                        FileDescription )

        # Save spatial grid:
        SaveGridHDF5File( FileName, geometry )

        # Save t = 0 state (initial conditions), which can be useful 
        # for running comparison runs with the same ICs:
        save_index = 1
        t          = 0.0
        Δt_Hydro   = 0.0

        SaveRHDTimeStepHDF5( FileName, hydrostate, chemstate, duststate, RT,
                             save_index, t, Δt_Hydro )
    end

    ###############################################################
    #######                                                 #######
    #######          T I M E  I N T E G R A T I O N         #######
    #######                                                 #######
    ###############################################################

    Δt_Resolve  = tmax / Nminsteps

    # ---- Initialize time-steps where data is saved ----
    Δt_save     = tmax / ( Nsaved - 1 )
    t_next_save = Δt_save                 # We have already saved t = 0 state, so next is at t = Δt_save  
    save_index  = 1
    save_tol    = 1e-12 * tmax # Tolerance to guard against round-off errors when deciding if to save data

    if print_info
        print("\r\033[KStarting time evolution..."); flush(stdout)
        println() # New line
    end
    
    # Initialize the progress meter:
    prog = Progress( 100; dt = 0.01, desc = "Time integration progress:",
                     barglyphs=BarGlyphs("[=> ]"), barlen = 50, color=:green ) 

    t                = 0.0        # Initialize time 
    step             = 1          # Initialize the steps
    Δt_Hydro         = Inf        # Initialize the hydro time-step 
    while t < tmax

        # Hydro & dust CFL time-step:
        Δt_CFL      = GetCFLTimeStep( r, Z, ρ, P, u_r, u_Z, ud_r, ud_Z, γad )  

        # Limit from Radiation pressure acceleration time-scale:
        if RadPressure 
            Δt_RP = GetRadPressureTimeStep( r, Z, ρ, ρd, RT.f_stellar_r,   RT.f_stellar_Z )
        else
            # Ignore limit in absence of radiation pressure feedback:
            Δt_RP   = Inf 
        end

        # Set hydro time-step:
        Δt_Hydro = min( 1.25 * Δt_Hydro, Δt_CFL, Δt_RP, Δt_Resolve, tmax - t )

        # Update hydro, gravity, thermochemistry, stellar RT:
        UpdateRHD!( Δt_Hydro, hydrostate, chemstate,
                    duststate,
                    geometry, 
                    ghostgrid, 
                    RT,
                    RadPressure, IRfeedback,
                    z, Hydro, CoolingHeating )

        # Update time & step: 
        t    += Δt_Hydro 
        step += 1

        # ----------- Decide to save data --------------
        if t ≥ t_next_save  -  save_tol # The last factor is to avoid missing a save because round-off errors 
            save_index += 1

            # Save data:
            SaveRHDTimeStepHDF5( FileName, hydrostate, chemstate, duststate, RT,
                                 save_index, t, Δt_Hydro )

            # Move the 'next save' appointment forward until it is in the future 
            while t ≥ t_next_save - save_tol
                t_next_save += Δt_save
            end
        end

        # Compute progress fraction
        progress_frac = t / tmax
        # Update the progress bar description to show simulation time and percent
        prog.desc = @sprintf("t = %.2e yr (%.1f%%)", t / yr, 100 * progress_frac)
        # Update the bar progress value
        update!( prog, Int(round(100 * progress_frac)) )

        if step ≥ Nmaxsteps 
            println("\nSimulation stopped: Reached maximum number of global time-steps ($(Nmaxsteps)).")
            break
        end

      
    end # End of time-integration 


    return nothing
end


end # End of module 