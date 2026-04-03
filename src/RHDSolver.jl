
module RHDSolver

__precompile__()


include( "./Structs.jl" )
include( "./ConstantsAndFunctions.jl" )
using .Structs
using .ConstantsAndFunctions
using LinearAlgebra           # For the tridiagonal solver 
BLAS.set_num_threads(1)       # Ensure the tridiagonal solver is not threaded (we thread over spatial cells)

# Some basic underlying files:
include( "./WENO3.jl" )
include( "./TridiagonalSolver.jl" )
include( "./M1Closure.jl" ) 
include( "./AdvectJandH.jl" )
include( "./SetupLymanAlphaArrays.jl" )
include( "./SpatialGrid.jl" )
include( "./PoissonGravitySolver.jl" )
include( "./Hydrodynamics.jl" )

include( "./UpdateLyaJandH.jl" )
include( "./DustPhysics.jl" )
include( "./DustGasCoupling.jl" )
include( "PhotoChemistryAndCooling/InitializeChemistryStruct.jl" ) 
include( "PhotoChemistryAndCooling/PhotoThermoChemistry.jl" )
include( "PhotoChemistryAndCooling/StellarIR_Diagnostics.jl" )
include( "PhotoChemistryAndCooling/StellarWinds.jl" )
include( "./RHD_TimeStep.jl" )
include( "./AdvectScalars.jl" )
include( "./Turbulence.jl" )
include( "./SaveHDF5Data_RHD.jl" )

using  Base.Threads              # For CPU acceleration
using  ProgressMeter             # To show time step progress
using  Dates, Printf


const N_dust::Int           = 3      # 3 dust bins (Sil, C, PAH) 
const N_passivescalars::Int = 19     # Number of passive scalars (for chemistry etc)
const BENCHMARK_RHD::Bool   = false  # For benchmarking performance of the RHD update. Off by default.

export SetupRHDsimulation, RestartRHDsimulation



""" This function performs a full RHD update over a hydro time-step Δt_Hydro. 
    Operator splitting is used, with the following order:
    
    (1) Gravity solver + kick over half-step Δt_Hydro / 2. (If gravity is on)

    (2) Hydro & dust advection over full step Δt_Hydro. 

    (3) Gravity solver + kick over half-step Δt_Hydro / 2. (If gravity is on)

    (4) Inject stellar wind momentum & mass over full step Δt_Hydro, 
        if stellar wind feedback is on.

    (5) Sub-cycle radiative transfer of stellar, IR, and Lyα photons (if on)
        over steps Δt_RT ≤ Δt_Hydro. And for each Δt_RT, sub-cycle photo-thermo
        chemistry and other stiff source terms (gas heating/cooling, chemistry, 
        dust-gas coupling, ...). 
"""
function UpdateRHD!( Δt_Hydro::Float64,
                   hydrostate::HydrodynamicsState{Float64},
                    chemstate::ChemistryState{Float64},
                    duststate::DustState,
                     geometry::GeometryCylindrical{Float64},
                    gravstate::GravityState{Float64},
                   gravcoeffs::GravityCoeffs{Float64},
                    ghostgrid::GhostPaddedGrid{Float64},
                     LyaState::Union{LyαArrays, Nothing},
                           RT::StellarRTArrays,
                      Gravity::Bool    = true,
                  RadPressure::Bool    = true,
                   IRfeedback::Bool    = true,
          StellarWindFeedback::Bool    = true,
                            z::Float64 = 0.0,
                    mstarZAMS::Float64 = 35.0,
                   Zstar_Zsun::Float64 = 1.0,
                        Hydro::Bool    = true,
               CoolingHeating::Bool    = true
                     )

    # Main geometry arrays:
    ΔV          = geometry.ΔV
    A_r         = geometry.A_r
    A_Z         = geometry.A_Z
    inv_r_cell  = geometry.inv_r_cell
    r           = geometry.r 
    Z           = geometry.Z

    # Some hydro state arrays (cell-averaged):
    ρ           = hydrostate.ρ;        ρu_r        = hydrostate.ρu_X 
    ρu_Z        = hydrostate.ρu_Y;     E           = hydrostate.E
    u_r         = hydrostate.u_X;      u_Z         = hydrostate.u_Y 
    P           = hydrostate.P;        e           = hydrostate.e_dual 
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

    # Boundary conditions:
    BC_Z_bot   = hydrostate.BC_Y_left

    # Grid size & intervals:
    N_r, N_Z    = size( ρ )
    dN_r        = 1:N_r 
    dN_Z        = 1:N_Z 

    if BENCHMARK_RHD
        t_HydroDust = 0.0
        t_SubCycle  = 0.0
        t0          = time_ns()
    end

    ###############################################################
    #######                                                 #######
    #######      H Y D R O / D U S T - D Y N A M I C S      #######
    #######                                                 #######
    ###############################################################

    # Gravity solver + kick update (over a half-step):
    Δt_Hydro_half = 0.5 * Δt_Hydro
    if Gravity && Hydro 
        UpdateTotalDustDensity!( hydrostate )

        GravitySolveAndKick!( Δt_Hydro_half, hydrostate, gravstate, gravcoeffs,
                              geometry )

    end

    if Hydro
        # Prepare passive scalars before we advect them:
        PreparePassiveScalars!( hydrostate, chemstate )

        # Update hydro & dust variables & passive scalars by advection (full step):
        UpdateHydro!( Δt_Hydro, hydrostate, ghostgrid, geometry )  

        # Update total mass in simulation volume (for diagnostic purposes):
        UpdateTotalMass!( hydrostate, geometry )

        # Recover abundance fractions from passive scalars (needed before sub-cycling):
        RecoverChemistryFromPassiveScalars!( hydrostate, chemstate )
    end

    # Gravity solver + kick update (over a half-step):
    if Gravity && Hydro 
        UpdateTotalDustDensity!( hydrostate )

        GravitySolveAndKick!( Δt_Hydro_half, hydrostate, gravstate, gravcoeffs,
                              geometry )

    end

    # Inject momentum and mass from stellar winds if turned on:
    if StellarWindFeedback
        # Position of star along Z-axis:
        Zmid = 0.5 * ( Z[1] + Z[end] )
        Zc   = ifelse( BC_Z_bot === :Reflect, 0.0, Zmid )

        # Inject momentum & mass:
        InjectStellarWind!( Δt_Hydro, mstarZAMS, Zstar_Zsun, hydrostate, geometry, RT.ρ_star, Zc )
    end

    @threads for j in dN_Z
        @inbounds @simd for i in dN_r
            # Adiabatic index & mean molecular weight (needed to initialize temperature):
            γad[i,j] = Getγad( chemstate.xHII[i,j],  chemstate.xH2[i,j], 
                               chemstate.xHeII[i,j], chemstate.xHeIII[i,j] )

            μ        = Getμ( chemstate.xHII[i,j],  chemstate.xH2[i,j], 
                             chemstate.xHeII[i,j], chemstate.xHeIII[i,j] )

            # Recover primitives:
            u_r[i,j], u_Z[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_r[i,j], ρu_Z[i,j], E[i,j], e[i,j], γad[i,j] )

            # Ensure positive pressure:
            P[i,j] = max( P[i,j], P_floor )

            # Get the latest temperature corresponding to e = e_dual to set the RSL for RT:
            T[i,j] = e[i,j] * μ * (γad[i,j] - 1) * mH / ( ρ[i,j] * kB )
        end
    end

    # Update the ∇u components, needed for the velocity terms in the RT:
    Update_∇u_RHD!( ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                    u_r_L_r, u_r_R_r, u_r_L_Z, u_r_R_Z, 
                    u_Z_L_r, u_Z_R_r, u_Z_L_Z, u_Z_R_Z,
                    r, Z, inv_r_cell )

    if BENCHMARK_RHD
        t_HydroDust  = (time_ns() - t0)
        t0           = 0.0
    end

    ###############################################################
    #######                                                 #######
    #######          R T  +  S U B - C Y C L I N G          #######
    #######                                                 #######
    ###############################################################
    # Below we sub-cycle the RT, chemistry, and heating/cooling. 
    # We split the hydro step into (potentially) smaller RT CFL time-steps,
    # and for each RT CFL time-step we sub-cycle chemistry and heating/cooling.

    # Before sub-cycling, reset momentum change from radiation pressure to zero:
    if !isnothing(LyaState)
        Reset_Δρu!( LyaState.Δρu_Lyα_r, LyaState.Δρu_Lyα_Z )
    end
    Reset_Δρu!( RT.Δρu_stellar_r,   RT.Δρu_stellar_Z )
    Reset_Δρu!( RT.Δρu_IR_r,        RT.Δρu_IR_Z )

    # Determine an updated reduced speed of light for the radiative transfer:
    RT.c̃  = FindRSL( RT.c̃, LyaState, hydrostate, chemstate, duststate, geometry )
    if !isnothing(LyaState)
        LyaState.c̃_Lyα = RT.c̃
    end

    # ---------------------------------------------------------------------
    # --------  Start sub-cycling of photo-thermochemistry  ---------------
    # ---------------------------------------------------------------------

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
                                             LyaState, 
                                             RadPressure )

        # Update time:
        t_RT += Δt_RT

    end

    if BENCHMARK_RHD
        t_SubCycle  = (time_ns() - t0)
        t0          = 0.0

        println("Hydro/dust advection & gravity solve:  ", t_HydroDust * 1e-9,  " s")
        println("Sub-cycling (RT + photo-thermochemistry):  ", t_SubCycle * 1e-9, " s")
        println("-------------------")
    end

    ###############################################################
    #######                                                 #######
    #######    F E E D B A C K  &  D I A G N O S T I C S    #######
    #######                                                 #######
    ###############################################################

    if RadPressure && Hydro
        # If radiation pressure feedback is turned on, compute the 
        # resulting forces from the accumulated momentum injected 
        # during the sub-cycling (just a diagnostic):

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

        # NOTE: Work from radiation pressure is handled below, by updating E.
    end

    # Update LyC and Lyman-Werner escape fractions (just diagnostics):
    Update_fesc_LyC_and_LW_Cylindrical!( RT, A_r, A_Z, BC_Z_bot )

    # Update IR luminsoity & force diagnostics (just useful for comparing forces after simulation):
    if Gravity
        Update_M_F_IR_L_IR!( RT, ΔV, gravstate, BC_Z_bot )
    else
        Update_M_F_IR_L_IR!( RT, ΔV, BC_Z_bot )
    end

    # Update direct radiation pressure force diagnostics (just for analyzing after simulation):
    Update_M_F_dir!( RT, ΔV, BC_Z_bot )

    # Update Lyα forces f_Lyα, force multiplier M_F, luminosity L_Lyα, and Lyα escape fraction
    # (all of these mainly for diagnostics):
    if !isnothing(LyaState)
        if Gravity
            # If gravity is turned on, we also save force multiplier anti-parallel to gravity:
            LyaState.M_F, LyaState.M_F_antigravity, 
            LyaState.L_Lyα = Update_f_Lyα_MF_L_Lyα!( LyaState.f_Lyα_r, LyaState.f_Lyα_Z, LyaState.Δρu_Lyα_r, LyaState.Δρu_Lyα_Z, 
                                                     LyaState.∫dν_js_Lyα, ΔV, Δt_Hydro,
                                                     gravstate.f_gravity_r, gravstate.f_gravity_Z,
                                                     BC_Z_bot )
        else
            LyaState.M_F, LyaState.L_Lyα = Update_f_Lyα_MF_L_Lyα!( LyaState.f_Lyα_r, LyaState.f_Lyα_Z, LyaState.Δρu_Lyα_r, LyaState.Δρu_Lyα_Z, 
                                                                   LyaState.∫dν_js_Lyα, ΔV, Δt_Hydro,
                                                                   BC_Z_bot )
        end

        # Save Lyα escape fraction:
        LyaState.f_esc_Lyα = Compute_f_esc_Lyα_Cylindrical( LyaState.H_r_Lyα, LyaState.H_Z_Lyα, A_r, A_Z, LyaState.Δν, 
                                                            LyaState.L_Lyα, BC_Z_bot )
    end

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
                E[i,j]    = e[i,j]  +  ( ρu_r[i,j] * ρu_r[i,j]  +  ρu_Z[i,j] * ρu_Z[i,j] ) / ( 2 * ρ[i,j] )


                # Also update gas pressure & velocities, needed for the computing 
                # the CFL time-step:
                P[i,j]    = max( e[i,j] * ( γad[i,j] - 1 ), P_floor )
                u_r[i,j]  = ρu_r[i,j] / ρ[i,j]
                u_Z[i,j]  = ρu_Z[i,j] / ρ[i,j]

            end
        end # End of i-loop 
    end # End of j-loop 

    # Update dust velocities for each bin:
    UpdateDustVelocities!( hydrostate )

    # Update gas velocity dispersions (useful diagnostic):
    UpdateGasVelocityDispersion!( hydrostate, geometry )

    return nothing
end


""" This function sets up an RHD simulation in cylindrical coordinates, with a star of mass mstarZAMS (M⊙)
    situated at the center of a cloud with a given initial properties (density, temperature, chemical abundances,
    turbulence, etc.).  """
function SetupRHDsimulation( mstarZAMS::Float64,  # Stellar mass (M⊙)
                            Zstar_Zsun::Float64,  # Stellar metallicity Z_star / Z⊙
                                   DG₀::Float64,  # Dust-to-gas ratio (*not* normalized to Solar value)
                               nH_core::Float64,  # H density (cm⁻³) inside cloud core 
                                nH_ext::Float64,  # H density (cm⁻³) far outside cloud 
                                 Rcore::Float64,  # Radius of 'cloud core', i.e. higher density region
                                   Rcl::Float64,  # Radius of 'cloud'
                            σstar_cell::Float64,  # Gaussian σ (in units of cell size) that determines extent of spatially smeared star
                              Rmax_lin::Float64,
                            LZ_uniform::Float64,
                                  Rmax::Float64,
                                  Zmax::Float64,
                                    T₀::Float64,  # Temperature within cloud 
                                 T_ext::Float64,  # Temperature outside cloud 
                    POWER_LAW_ENVELOPE::Bool,     # Whether to use a power law envelope density profile outside
                            p_ENVELOPE::Float64,  # Power law for envelope, ρ ∝ r^(-p_ENVELOPE), if used 
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
                            Zgas_Zsun₀::Float64,  # Initial gas metallicity in units of Solar metallicity 
                                     z::Float64,
                                 Hydro::Bool    = true,     # Hydrodynamics (on/off)
                                   Lyα::Bool    = true,     # Lyα RT (on/off)
                                Recoil::Bool    = true,
                        LyαDestruction::Bool    = true,
                               Gravity::Bool    = true,
                           RadPressure::Bool    = true,
                            IRfeedback::Bool    = true,
                   StellarWindFeedback::Bool    = false,
                 DustGrowthDestruction::Bool    = true,     # Dust growth & destruction (on if true, off if false)
                                   OTS::Bool    = false,    # On-the-spot approximation (on if true, off if false)
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

    if POWER_LAW_ENVELOPE
        @assert p_ENVELOPE ≥ 0.0 "Density should drop off at large radii, i.e. p_ENVELOPE ≥ 0.0. "
    end

    # Initialize the RSL for M1 stellar & Lyα RT (will be updated adaptively each time-step)
    c̃          = c̃_min 

    # Setup cylindrical grid:
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
        Zcl_pos   = 0.5 * ( Z[1] + Z[end] )
        Zstar_pos = Zcl_pos
    end


    if print_info
        println(""" 
                Number of threads used: $(Threads.nthreads())
                         Julia version: $(VERSION) """); flush(stdout)
        println("---------------------------------")
        if SaveData
            println(" Data will be saved to HDF5 file: $(FileName). "); flush(stdout)
            println()
        end
        print(""" Simulation setup: A $(mstarZAMS) M⊙ star in a cloud of 
                  initial maximum density nH = $(nH_core) cm⁻³, initial metallicity Z/Z⊙ = $(Zgas_Zsun₀),
                  dust-to-gas ratio D/D⊙ = $(DG₀/DGsun). Initial turbulent RMS velocity 
                  is $(u_RMSturb/1e5) km s⁻¹.

                  SPATIAL RESOLUTION: $(N_r) × $(N_Z)
                  
                  PHYSICS SETTINGS:
                  
                  Hydrodynamics:               $(Hydro)
                  Gravity:                     $(Gravity)
                  Lyα RT:                      $(Lyα)
                  Radiation pressure feedback: $(RadPressure)
                  Stellar wind feedback:       $(StellarWindFeedback)
                  Dust growth & destruction:   $(DustGrowthDestruction)
                  On-the-spot approximation:   $(OTS)

                  """)
    end

    ###############################################################
    #######                                                 #######
    #######   H Y D R O  &  T H E R M O C H E M I S T R Y   #######
    #######                                                 #######
    ###############################################################

    if print_info 
        print("\r\033[KInitialising hydro, dust & thermochemistry arrays..."); flush(stdout)
    end

    # Cooling & heating should be turned on:
    CoolingHeating = true  

    # Set up hydro state + ghost grid arrays:
    BC_r_left::Symbol     = :Reflect     # Always reflective BCs at inner r-boundary in cylindrical coordinates
    hydrostate, ghostgrid = GenerateHydroArrays( r, rCoM, Z, ZCoM, 5 / 3, N_dust, N_passivescalars,
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
    e      = hydrostate.e_dual
    T      = hydrostate.T
    γad    = hydrostate.γad

    # Dust arrays:
    ρd     = hydrostate.ρd
    ρdud_r = hydrostate.ρdud_X
    ρdud_Z = hydrostate.ρdud_Y
    ρdtot  = hydrostate.ρdtot

    # Initialize turbulent velocity fluctuations:
    SetupTurbulence!( u_RMSturb, Rcl, Zcl_pos, u_r, u_Z, r, Z, rCoM, ZCoM, ΔV, BC_Z_left )

    # Set up chemistry state struct, containing arrays for all species tracked,
    # e.g. xHII, xH2 etc. :
    ZC₀       = fCarbon  * Zgas_Zsun₀ * Zsun 
    ZSil₀     = fSilicon * Zgas_Zsun₀ * Zsun 
    Zgasrest₀ = (1 - fCarbon - fSilicon) * Zgas_Zsun₀ * Zsun 
    chemstate = InitializeChemistryArrays( N_r, N_Z, ZC₀, ZSil₀, Zgasrest₀, true )

    # Cloud density properties:
    ρcore  = max( Get_ρ_From_nH( nH_core, Zgas_Zsun₀, true ),  10 * ρ_floor )    # Density within cloud core
    ρext   = max( Get_ρ_From_nH( nH_ext, Zgas_Zsun₀, true ),   10 * ρ_floor )    # Density far outside the cloud
    δr     = max( 2 * max( minimum(diff(r)), minimum(diff(Z)) ), 0.05 * Rcl )   # Determines fall-off in density outside cloud, in case power law is not used

    if ρcore < 1e3 * ρ_floor 
        @warn "The initial core density ρcore is only $(ρcore/ρ_floor) times the density floor. Consider making 
               ρcore larger, or take a look at the density floor ρ_floor in the Hydrodynamics.jl file."
    end

    # Initialize cloud properties:
    @threads for j in dN_Z 
        @simd for i in dN_r 
            @inbounds begin

                # Distance from cloud center at the edge of this cell:
                R = sqrt( r[i+1]^2  +   (Z[j+1] - Zcl_pos)^2 )

                # Gas density & chemistry:
                if R < Rcore 
                    # Gas density set to core density:
                    ρ[i,j] = ρcore

                    # Initial chemical abundances in the core:
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

                    if POWER_LAW_ENVELOPE
                        # Power law envelope out to Rcl:
                        if R < Rcl
                            ρ_env  = ρcore / ( (R / Rcore)^p_ENVELOPE )
                            ρ[i,j] = max( ρ_env, ρext )
                        else
                            ρ[i,j] = ρext
                        end

                    else
                        # Fast, gaussian exponential fall-off:
                        ρ[i,j] = max( ρcore * exp( - ((R - Rcore)/δr)^2 ), ρext )
                    end

                    # Initial chemical abundances outside the core:
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
                    P[i,j] = ρ[i,j] * kB * T[i,j] / ( Getμ(xHII_ext, xH2_ext, xHeII_ext, xHeIII_ext) * mH )
                end

                # Gas momentum density:
                ρu_r[i,j]   = ρ[i,j] * u_r[i,j]
                ρu_Z[i,j]   = ρ[i,j] * u_Z[i,j]

                # Gas internal energy density e = P / (γ - 1) = ρ kB T₀ / ((γ - 1) μ mH):
                e[i,j]      = P[i,j] / ( γad[i,j] - 1 )

                # Energy density of gas, E = e + E_kin:
                E[i,j]      = e[i,j]  +  0.5 * ρ[i,j] * (u_r[i,j]^2 + u_Z[i,j]^2)

                # Dust densities:
                ρd[1,i,j]   = max( fd_Sil * DG₀ * ρ[i,j], 10 * ρd_floor )  # Silicate dust density 
                ρd[2,i,j]   = max( fd_C   * DG₀ * ρ[i,j], 10 * ρd_floor )  # Graphite dust density 
                ρd[3,i,j]   = max( fd_PAH * DG₀ * ρ[i,j], 10 * ρd_floor )  # PAH dust density 

                @assert isapprox(fd_Sil + fd_C + fd_PAH, 1.0, rtol = 0.01) "Initial dust bin density fractions don't add up to 1."

                # Dust momentum density (assume same velocity as gas initially):
                ρdud_r[1,i,j] = ρd[1,i,j] * u_r[i,j]  # Silicate dust momentum
                ρdud_Z[1,i,j] = ρd[1,i,j] * u_Z[i,j]

                ρdud_r[2,i,j] = ρd[2,i,j] * u_r[i,j]  # Graphite dust momentum
                ρdud_Z[2,i,j] = ρd[2,i,j] * u_Z[i,j]

                ρdud_r[3,i,j] = ρd[3,i,j] * u_r[i,j]  # PAH dust momentum
                ρdud_Z[3,i,j] = ρd[3,i,j] * u_Z[i,j]
                
            end
        end # End of i-loop 
    end # End of j-loop


    # Initialize total mass in simulation volume (for diagnostic purposes):
    UpdateTotalMass!( hydrostate, geometry )

    ###############################################################
    #######                                                 #######
    #######                     D U S T                     #######
    #######                                                 #######
    ###############################################################

    # Initialize struct for dust properties/settings:
    Td₀       = 30.0
    duststate = SetupDustState( DustGrowthDestruction, N_r, N_Z, Td₀ )

    # Sanity/safety check:
    @assert size( duststate.Td_Sil ) == size( ρ ) " Dust temperature Td should have the same size 
                                                    as spatial grid. "


    ###############################################################
    #######                                                 #######
    #######                  G R A V I T Y                  #######
    #######                                                 #######
    ###############################################################

    # Initialize struct + arrays for gravity state & coefficients:
    gravstate, gravcoeffs = 
    CreateGravityStateAndCoefficients( geometry.A_r, geometry.A_Z, geometry.inv_ΔV, 
                                       rCoM, ZCoM, r, Z )

    ###############################################################
    #######                                                 #######
    #######                L Y M A N - α  R T               #######
    #######                                                 #######
    ###############################################################
    
    # Generate struct with Lyα arrays + settings:
    LyαState::Union{LyαArrays,Nothing} =
    Lyα ? GenerateLyαArrays( c̃, r, Z, T, Recoil, LyαDestruction ) : nothing


    ###############################################################
    #######                                                 #######
    #######                S T E L L A R  R T               #######
    #######                                                 #######
    ############################################################### 

    # Generate all stellar RT arrays, including J, H, inteface fluxes, 
    # emissivities, and luminosities, for all 7 stellar + IR RT bands:
    RT = GenerateStellarRTArrays( mstarZAMS, Zstar_Zsun, ΔV, r, Z, 
                                  Zstar_pos, σstar_cell, c̃, OTS )

    # Add stellar mass to the non-gas mass (will also influence gravity):
    gravstate.ρ_bg .= RT.ρ_star

    ###############################################################
    #######                                                 #######
    #######           S E T  U P  S A V E  F I L E          #######
    #######                                                 #######
    ###############################################################

    if SaveData
        if Lyα
            N_ν = length( LyαState.νC ) # Number of Lyα frequency bins
        else
            N_ν = 0
        end

        # Create save file:
        CreateHDF5File( FileName, N_r, N_Z, Nminsteps, Nmaxsteps, Nsaved, Gravity, 
                        Lyα, Recoil, LyαDestruction, RadPressure, IRfeedback,
                        StellarWindFeedback,
                        DustGrowthDestruction, OTS, z, 
                        N_ν,
                        tmax, mstarZAMS, Zstar_Zsun, σstar_cell, 
                        Zstar_pos,
                        DG₀, Zgas_Zsun₀,
                        BC_r_right, BC_Z_left, BC_Z_right,
                        CFL, CFL_RT, c̃,
                        FileDescription )

        # Save spatial grid:
        SaveGridHDF5File( FileName, geometry )

        # Save Lyα frequency grid
        if Lyα
            SaveLyαFreqGridHDF5File( FileName, LyαState )  
        end

        # Save t = 0 state (initial conditions), which can be useful 
        # for running comparison runs with the same ICs:
        save_index = 1
        t          = 0.0
        Δt_Hydro   = 0.0

        if !isnothing(LyαState)
            # Update Lyα energy density before saving (will be zero - just for clarity):
            Update_e_Lyα!( LyαState.e_Lyα, LyαState.Δν, LyαState.J_Lyα )
        end

        SaveRHDTimeStepHDF5( FileName, hydrostate, chemstate, gravstate, duststate, RT,
                             LyαState, save_index, t, Δt_Hydro, Gravity )
    end

    ###############################################################
    #######                                                 #######
    #######          T I M E  I N T E G R A T I O N         #######
    #######                                                 #######
    ###############################################################

    # Initialize the gravitational potential, so that we use an appropriate
    # first time-step:
    if Gravity
        # Update total dust density ρdtot:
        UpdateTotalDustDensity!( hydrostate )

        Update_Φ!( gravstate.Φ, gravstate.Φ_half, gravstate.Φ_old, gravstate.Φ_snapshot, 
                   gravstate.ρtot, ρ, ρdtot, gravstate.ρ_bg,
                   gravcoeffs.RPoiss_ijp, gravcoeffs.RPoiss_ij, gravcoeffs.RPoiss_ijm,
                   gravcoeffs.ZPoiss_ijp, gravcoeffs.ZPoiss_ij, gravcoeffs.ZPoiss_ijm, 
                   gravcoeffs.Δτ_max, gravcoeffs.Δτ_min,
                   ΔV, rCoM, ZCoM, gravstate.Q, gravstate.P, gravstate.ℓmax, BC_Z_left )

        Update_∇Φ!( gravstate.∂Φ∂r, gravstate.∂Φ∂Z, gravstate.Φ, 
                    gravstate.f_gravity_r, gravstate.f_gravity_Z, ρ, 
                    r, Z, rCoM, ZCoM, 
                    gravstate.P, gravstate.Q,
                    BC_Z_left, gravstate.ℓmax )
    end

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
        # Set global hydro time-step:
        Δt_Hydro_max =  min( Δt_Resolve, tmax - t )
        Δt_Hydro     =  Compute_RHD_TimeStep( Δt_Hydro, Δt_Hydro_max, hydrostate, gravstate, RT,  
                                              LyαState, chemstate, duststate, geometry, 
                                              Gravity, RadPressure )

        # Update hydro, gravity, thermochemistry, stellar RT:
        UpdateRHD!( Δt_Hydro, hydrostate, chemstate,
                    duststate,
                    geometry, gravstate, gravcoeffs,
                    ghostgrid, 
                    LyαState, RT,
                    Gravity, RadPressure, IRfeedback, StellarWindFeedback, 
                    z, mstarZAMS, Zstar_Zsun, Hydro, CoolingHeating )

        # Update time & step: 
        t    += Δt_Hydro 
        step += 1

        # ----------- Decide to save data --------------
        if t ≥ t_next_save  -  save_tol # The last factor is to avoid missing a save because round-off errors 
            save_index += 1

            if SaveData
                if Lyα
                    # Update Lyα energy density for saving:
                    Update_e_Lyα!( LyαState.e_Lyα, LyαState.Δν, LyαState.J_Lyα )
                end

                # Save data:
                SaveRHDTimeStepHDF5( FileName, hydrostate, chemstate, gravstate, duststate, RT,
                                     LyαState, save_index, t, Δt_Hydro, Gravity )
            end

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


""" This function can restart an RHD simulation, using data saved 
    in an HDF5 file. """
function RestartRHDsimulation( FileName::String  = "RHDSim.h5" )

    PrintLYDIONLogo()

    println(""" 
            Number of threads used: $(Threads.nthreads())
                     Julia version: $(VERSION) """); flush(stdout)
    println("---------------------------------")
    println(" Restarting RHD simulation from checkpoint in HDF5 file $(FileName)... ")
    println( " " )

    # Initialize various variables and settings that will be needed after
    # we close the h5 open:
    Gravity = false; Lyα = false; Recoil = false; RadPressure = false; IRfeedback = false; StellarWindFeedback = false
    Hydro = true; LyαDestruction = true; DustGrowthDestruction = true
    mstarZAMS = 0.0; Zstar_Zsun = 0.0
    z = 0.0; tmax = 0.0
    t = 0.0; step = 0; Δt_Hydro = 0.0
    c̃_saved = 0.0
    BC_r_right = :Outflow; BC_Z_left = :Reflect; BC_Z_right = :Outflow

    geometry_ref   = Ref{Any}(nothing)
    hydrostate_ref = Ref{Any}(nothing)
    ghostgrid_ref  = Ref{Any}(nothing)
    chemstate_ref  = Ref{Any}(nothing)
    duststate_ref  = Ref{Any}(nothing)
    gravstate_ref  = Ref{Any}(nothing)
    gravcoeffs_ref = Ref{Any}(nothing)
    RT_ref         = Ref{Any}(nothing)
    LyαState_ref   = Ref{Any}(nothing)

    r_ref          = Ref{Vector{Float64}}(Float64[])
    Z_ref          = Ref{Vector{Float64}}(Float64[])
    ρ_ref          = Ref{Any}(nothing)
    P_ref          = Ref{Any}(nothing)
    u_r_ref        = Ref{Any}(nothing)
    u_Z_ref        = Ref{Any}(nothing)
    γad_ref        = Ref{Any}(nothing)
    ρd_ref         = Ref{Any}(nothing)
    ud_r_ref       = Ref{Any}(nothing)
    ud_Z_ref       = Ref{Any}(nothing)

    CoolingHeating = true   
    SaveData  = true   
    Nminsteps = 300; Nmaxsteps = 100000; Nsaved = 300; print_info = true

    # Useful functions for reading data from the HDF5 file:
    parse_bool(x) = x isa Bool ? x : lowercase(String(x)) == "true"
    parse_sym(x)  = x isa Symbol ? x : Symbol(String(x))
    parse_f64(x)  = x isa Float64 ? x : parse(Float64, String(x))
    parse_i(x)    = x isa Int ? x : parse(Int, String(x))

    function get_attr(settings, key::String)
        haskey(attrs(settings), key) || error("RestartRHDsimulation: missing Settings attribute \"$key\" in $FileName")
        return attrs(settings)[key]
    end

    # Open HDF5 file, get needed data for ICs, then restart simulation:
    h5open(FileName, "r") do file 

        # Get desired number of saved time-steps:
        Nsaved = length(file["Times"])

        # Next, get settings & checkpoint data:
        haskey(file, "Settings")   || error("RestartRHDsimulation: missing group /Settings in $FileName.")
        haskey(file, "Checkpoint") || error("RestartRHDsimulation: missing group /Checkpoint in $FileName.")

        Settings   = file["Settings"]
        CheckPoint = file["Checkpoint"]

        # Get settings:
        Gravity                = parse_bool(get_attr(Settings, "Gravity"))
        Lyα                    = parse_bool(get_attr(Settings, "Lya"))  
        Recoil                 = parse_bool(get_attr(Settings, "Recoil"))
        RadPressure            = parse_bool(get_attr(Settings, "RadPressure"))
        IRfeedback             = parse_bool(get_attr(Settings, "IRfeedback"))
        StellarWindFeedback    = parse_bool(get_attr(Settings, "StellarWindFeedback"))
        DustGrowthDestruction  = parse_bool(get_attr(Settings, "DustGrowth/Destruction"))
        OTS                    = parse_bool(get_attr(Settings, "OTS"))
        LyαDestruction         = haskey(attrs(Settings), "Lya_pd_on") ? parse_bool(get_attr(Settings, "Lya_pd_on")) : true
        Hydro                  = haskey(attrs(Settings), "Hydro")     ? parse_bool(get_attr(Settings, "Hydro"))     : true
        z                      = haskey(attrs(Settings), "z")         ? parse_f64(get_attr(Settings, "z"))          : 0.0

        # Boundary conditions in cylindrical coordinates:
        BC_r_right     = parse_sym(get_attr(Settings, "BC_r_right"))
        BC_Z_left      = parse_sym(get_attr(Settings, "BC_Z_left"))
        BC_Z_right     = parse_sym(get_attr(Settings, "BC_Z_right"))

        # Stellar parameters:
        mstarZAMS      = parse_f64(get_attr(Settings, "mstar_Msun"))
        Zstar_Zsun     = parse_f64(get_attr(Settings, "Z_star_Zsun"))
        σstar_cell     = parse_f64(get_attr(Settings, "σ_star_cell"))
        Zstar_pos      = parse_f64(get_attr(Settings, "Z_pos"))

        # Get desired final time of simulation:
        if haskey(attrs(Settings), "tmax")
            tmax = parse_f64(get_attr(Settings, "tmax"))
        else
            error("RestartRHDsimulation: missing Settings attribute \"tmax\" in $FileName")
        end

        # Get minimum and maximum allowed number of time-steps:
        Nminsteps = parse_i(get_attr(Settings, "Nminsteps")) 
        Nmaxsteps = parse_i(get_attr(Settings, "Nmaxsteps"))

        ###############################################################
        #######                                                 #######
        #######            C H E C K P O I N T  D A T A         #######
        #######                                                 #######
        ###############################################################

        # Checkpoint data to be used as ICs for continuing simulation:

        haskey(attrs(CheckPoint), "t_current") || error("RestartRHDsimulation: missing Checkpoint attribute \"t_current\"")
        haskey(attrs(CheckPoint), "step")      || error("RestartRHDsimulation: missing Checkpoint attribute \"step\"")
        haskey(attrs(CheckPoint), "dt")        || error("RestartRHDsimulation: missing Checkpoint attribute \"dt\"")

        t         = attrs(CheckPoint)["t_current"] isa Float64 ? attrs(CheckPoint)["t_current"] : parse(Float64, String(attrs(CheckPoint)["t_current"]))
        step      = attrs(CheckPoint)["step"]      isa Int     ? attrs(CheckPoint)["step"]      : parse(Int,     String(attrs(CheckPoint)["step"]))
        Δt_Hydro  = attrs(CheckPoint)["dt"]        isa Float64 ? attrs(CheckPoint)["dt"]        : parse(Float64, String(attrs(CheckPoint)["dt"]))

        # Get last saved RSL:
        c̃_saved   = attrs(CheckPoint)["RSL"] isa Float64 ? attrs(CheckPoint)["RSL"] : parse(Float64, String(attrs(CheckPoint)["RSL"]))

        # Get saved spatial grid data:
        haskey(file, "r")           || error("RestartRHDsimulation: missing dataset /r in $FileName.")
        haskey(file, "rCoM")        || error("RestartRHDsimulation: missing dataset /rCoM in $FileName.")
        haskey(file, "Z")           || error("RestartRHDsimulation: missing dataset /Z in $FileName.")
        haskey(file, "ZCoM")        || error("RestartRHDsimulation: missing dataset /ZCoM in $FileName.")
        haskey(file, "CellVolumes") || error("RestartRHDsimulation: missing dataset /CellVolumes in $FileName.")

        r    = Vector{Float64}(file["r"][:])
        rCoM = Vector{Float64}(file["rCoM"][:])
        Z    = Vector{Float64}(file["Z"][:])
        ZCoM = Vector{Float64}(file["ZCoM"][:])
        ΔV   = Array{Float64,2}(file["CellVolumes"][:, :])

        N_r, N_Z  = size( ΔV ) # Grid size 

        # Build geometry from saved grid + cell volumes (recomputes WENO3 coefficients etc.)
        geometry = SetupCylindricalGrid_FromSaved( r, rCoM, Z, ZCoM, ΔV )

        # Set up chemistry state struct, containing arrays for all species tracked,
        # e.g. xHII, xH2 etc. (not initialized to right values yet):
        chemstate = InitializeChemistryArrays( N_r, N_Z, 0.0, 0.0, 0.0, true )

        chemstate.xHII     .=  CheckPoint["xHII"][:, :]
        chemstate.xH2      .=  CheckPoint["xH2"][:, :]
        chemstate.xHeII    .=  CheckPoint["xHeII"][:, :]
        chemstate.xHeIII   .=  CheckPoint["xHeIII"][:, :]
        chemstate.xDII     .=  CheckPoint["xDII"][:, :]
        chemstate.xHD      .=  CheckPoint["xHD"][:, :]
        chemstate.xCI      .=  CheckPoint["xCI"][:, :]
        chemstate.xCII     .=  CheckPoint["xCII"][:, :]
        chemstate.xCO      .=  CheckPoint["xCO"][:, :]
        chemstate.xOI      .=  CheckPoint["xOI"][:, :]
        chemstate.xOII     .=  CheckPoint["xOII"][:, :]

        chemstate.ZC       .= CheckPoint["ZC"][:, :]
        chemstate.ZSil     .= CheckPoint["ZSil"][:, :]
        chemstate.Zgasrest .= CheckPoint["Zgasrest"][:, :]

        # Set up dust state struct, containing dust properties:
        Td₀       = 30.0      # Arbitrary dust temperature 
        duststate = SetupDustState( DustGrowthDestruction, N_r, N_Z, Td₀ )

        # Setup hydro state (not initialized to right values yet) + ghost grid arrays:
        BC_r_left::Symbol     = :Reflect # Always reflective BCs at inner r-boundary in cylindrical coordinates
        hydrostate, ghostgrid = GenerateHydroArrays( r, rCoM, Z, ZCoM, 5 / 3, N_dust, N_passivescalars,
                                                     BC_r_left, BC_r_right,  # Inner & outer r-BCs 
                                                     BC_Z_left, BC_Z_right   # Lower & outer Z-BCs
                                                     )

        # Gas & dust arrays:
        ρ       = hydrostate.ρ
        ρu_r    = hydrostate.ρu_X
        ρu_Z    = hydrostate.ρu_Y
        E       = hydrostate.E
        T       = hydrostate.T
        u_r     = hydrostate.u_X
        u_Z     = hydrostate.u_Y
        P       = hydrostate.P
        γad     = hydrostate.γad
        e       = hydrostate.e_dual

        ρd      = hydrostate.ρd
        ρdud_r  = hydrostate.ρdud_X
        ρdud_Z  = hydrostate.ρdud_Y
        ud_r    = hydrostate.ud_X
        ud_Z    = hydrostate.ud_Y

        # Get latest hydro state from saved file:
        ρ      .= CheckPoint["GasDensity"][:, :]
        ρu_r   .= CheckPoint["rMomentum"][:, :]
        ρu_Z   .= CheckPoint["ZMomentum"][:, :]
        E      .= CheckPoint["Energy"][:, :]
        e      .= CheckPoint["e_dual"][:, :]
        T      .= CheckPoint["T"][:, :]

        # Same for dust:
        ρd[1, :, :]     .= CheckPoint["DustDensity_Sil"][:, :]
        ρdud_r[1, :, :] .= CheckPoint["DustrMomentum_Sil"][:, :]
        ρdud_Z[1, :, :] .= CheckPoint["DustZMomentum_Sil"][:, :]
        
        ρd[2, :, :]     .= CheckPoint["DustDensity_C"][:, :]
        ρdud_r[2, :, :] .= CheckPoint["DustrMomentum_C"][:, :]
        ρdud_Z[2, :, :] .= CheckPoint["DustZMomentum_C"][:, :]
        
        ρd[3, :, :]     .= CheckPoint["DustDensity_PAH"][:, :]
        ρdud_r[3, :, :] .= CheckPoint["DustrMomentum_PAH"][:, :]
        ρdud_Z[3, :, :] .= CheckPoint["DustZMomentum_PAH"][:, :]

        dN_r = 1:N_r
        dN_Z = 1:N_Z
        @threads for j in dN_Z
            @inbounds @simd for i in dN_r
                # Get adiabatic index from chemistry state:
                γad[i,j] = Getγad( chemstate.xHII[i,j], chemstate.xH2[i,j],
                                   chemstate.xHeII[i,j], chemstate.xHeIII[i,j] )

                # Get primitive variables: 
                u_r[i,j], u_Z[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_r[i,j], ρu_Z[i,j], 
                                                           E[i,j], e[i,j], γad[i,j] )
                P[i,j]  = max( P[i,j], P_floor )

                # Mean molecular weight:
                μ       = Getμ( chemstate.xHII[i,j], chemstate.xH2[i,j],
                                chemstate.xHeII[i,j], chemstate.xHeIII[i,j] )

                # Compute T from e and make sure it is consistent with saved T:
                T_frome = e[i,j] * μ * ( γad[i,j] - 1 ) * mH / ( ρ[i,j] * kB )

                if !isapprox(T_frome, T[i,j], rtol=1e-3, atol=0.0)
                    error("Restart: inconsistent T at ($i,$j): Saved = $(T[i,j]) K, derived value from e is $(T_frome) K")
                end
                T[i,j]  = T_frome

            end
        end

        # Update dust velocities:
        UpdateDustVelocities!( hydrostate )

        # Update total dust density:
        UpdateTotalDustDensity!( hydrostate )

        # Initialize struct + arrays for gravity state & coefficients:
        gravstate, gravcoeffs = 
        CreateGravityStateAndCoefficients( geometry.A_r, geometry.A_Z, geometry.inv_ΔV, 
                                           rCoM, ZCoM, r, Z )

        # Initialize stellar RT arrays/state:
        RT = GenerateStellarRTArrays( mstarZAMS, Zstar_Zsun, ΔV, r, Z, 
                                      Zstar_pos, σstar_cell, c̃_saved, OTS )

        # Add stellar mass to gravitating mass:
        gravstate.ρ_bg .= RT.ρ_star

        # Get latest saved stellar RT state:
        RT.J_opt    .= CheckPoint["J_opt"][:, :];    RT.J_opt_old   .= RT.J_opt
        RT.H_r_opt  .= CheckPoint["H_r_opt"][:, :];  RT.H_r_opt_old .= RT.H_r_opt
        RT.H_Z_opt  .= CheckPoint["H_Z_opt"][:, :];  RT.H_Z_opt_old .= RT.H_Z_opt

        RT.J_FUV    .= CheckPoint["J_FUV"][:, :];    RT.J_FUV_old   .= RT.J_FUV
        RT.H_r_FUV  .= CheckPoint["H_r_FUV"][:, :];  RT.H_r_FUV_old .= RT.H_r_FUV
        RT.H_Z_FUV  .= CheckPoint["H_Z_FUV"][:, :];  RT.H_Z_FUV_old .= RT.H_Z_FUV

        RT.J_LW     .= CheckPoint["J_LW"][:, :];    RT.J_LW_old   .= RT.J_LW
        RT.H_r_LW   .= CheckPoint["H_r_LW"][:, :];  RT.H_r_LW_old .= RT.H_r_LW
        RT.H_Z_LW   .= CheckPoint["H_Z_LW"][:, :];  RT.H_Z_LW_old .= RT.H_Z_LW

        RT.J_EUV1   .= CheckPoint["J_EUV1"][:, :];    RT.J_EUV1_old   .= RT.J_EUV1
        RT.H_r_EUV1 .= CheckPoint["H_r_EUV1"][:, :];  RT.H_r_EUV1_old .= RT.H_r_EUV1
        RT.H_Z_EUV1 .= CheckPoint["H_Z_EUV1"][:, :];  RT.H_Z_EUV1_old .= RT.H_Z_EUV1

        RT.J_EUV2   .= CheckPoint["J_EUV2"][:, :];    RT.J_EUV2_old   .= RT.J_EUV2
        RT.H_r_EUV2 .= CheckPoint["H_r_EUV2"][:, :];  RT.H_r_EUV2_old .= RT.H_r_EUV2
        RT.H_Z_EUV2 .= CheckPoint["H_Z_EUV2"][:, :];  RT.H_Z_EUV2_old .= RT.H_Z_EUV2

        RT.J_EUV3   .= CheckPoint["J_EUV3"][:, :];    RT.J_EUV3_old   .= RT.J_EUV3
        RT.H_r_EUV3 .= CheckPoint["H_r_EUV3"][:, :];  RT.H_r_EUV3_old .= RT.H_r_EUV3
        RT.H_Z_EUV3 .= CheckPoint["H_Z_EUV3"][:, :];  RT.H_Z_EUV3_old .= RT.H_Z_EUV3

        RT.J_IR     .= CheckPoint["J_IR"][:, :];      RT.J_IR_old   .= RT.J_IR
        RT.H_r_IR   .= CheckPoint["H_r_IR"][:, :];    RT.H_r_IR_old .= RT.H_r_IR
        RT.H_Z_IR   .= CheckPoint["H_Z_IR"][:, :];    RT.H_Z_IR_old .= RT.H_Z_IR

        # Initialize Lyα state:
        if Lyα
            # Get saved frequency grid:
            @assert haskey(file, "ν") && haskey(file, "νC") && haskey(file, "Δν")

            ν    = Vector{Float64}(file["ν"][:])
            νC   = Vector{Float64}(file["νC"][:])
            Δν   = Vector{Float64}(file["Δν"][:])

            # Create Lyα state:
            LyαState = GenerateLyαArrays_FromSaved( c̃_saved, ν, νC, Δν, r, Z, T,
                                                    Recoil, LyαDestruction )

            # Set Lyα J and H to last saved values:
            LyαState.J_Lyα       .= CheckPoint["J_Lya"][:, :, :]
            LyαState.H_r_Lyα     .= CheckPoint["H_r_Lya"][:, :, :]
            LyαState.H_Z_Lyα     .= CheckPoint["H_Z_Lya"][:, :, :]

            LyαState.J_Lyα_old   .= LyαState.J_Lyα
            LyαState.H_r_Lyα_old .= LyαState.H_r_Lyα
            LyαState.H_Z_Lyα_old .= LyαState.H_Z_Lyα

            # Recompute e_Lyα:
            Update_e_Lyα!( LyαState.e_Lyα, LyαState.Δν, LyαState.J_Lyα )
        else
            LyαState = nothing 
        end

        # Finalize arrays:
        geometry_ref[]   = geometry
        hydrostate_ref[] = hydrostate
        ghostgrid_ref[]  = ghostgrid
        chemstate_ref[]  = chemstate
        duststate_ref[]  = duststate
        gravstate_ref[]  = gravstate
        gravcoeffs_ref[] = gravcoeffs
        RT_ref[]         = RT
        LyαState_ref[]   = LyαState

        r_ref[]    = r
        Z_ref[]    = Z
        ρ_ref[]    = ρ
        P_ref[]    = P
        u_r_ref[]  = u_r
        u_Z_ref[]  = u_Z
        γad_ref[]  = γad
        ρd_ref[]   = ρd
        ud_r_ref[] = ud_r
        ud_Z_ref[] = ud_Z


    end # End of h5open 

    geometry   = geometry_ref[]
    hydrostate = hydrostate_ref[]
    ghostgrid  = ghostgrid_ref[]
    chemstate  = chemstate_ref[]
    duststate  = duststate_ref[]
    gravstate  = gravstate_ref[]
    gravcoeffs = gravcoeffs_ref[]
    RT         = RT_ref[]
    LyαState   = LyαState_ref[]

    r          = r_ref[]
    Z          = Z_ref[]
    ρ          = ρ_ref[]
    P          = P_ref[]
    u_r        = u_r_ref[]
    u_Z        = u_Z_ref[]
    γad        = γad_ref[]
    ρd         = ρd_ref[]
    ud_r       = ud_r_ref[]
    ud_Z       = ud_Z_ref[]

    ###############################################################
    #######                                                 #######
    #######          T I M E  I N T E G R A T I O N         #######
    #######                                                 #######
    ###############################################################

    # Same time integration code as in the SetupRHDsimulation function.


    # Initialize the gravitational potential, so that we use an appropriate
    # first time-step:
    if Gravity
        Update_Φ!( gravstate.Φ, gravstate.Φ_half, gravstate.Φ_old, gravstate.Φ_snapshot, 
                   gravstate.ρtot, ρ, hydrostate.ρdtot, gravstate.ρ_bg,
                   gravcoeffs.RPoiss_ijp, gravcoeffs.RPoiss_ij, gravcoeffs.RPoiss_ijm,
                   gravcoeffs.ZPoiss_ijp, gravcoeffs.ZPoiss_ij, gravcoeffs.ZPoiss_ijm, 
                   gravcoeffs.Δτ_max, gravcoeffs.Δτ_min,
                   geometry.ΔV, geometry.rCoM, geometry.ZCoM, 
                   gravstate.Q, gravstate.P, gravstate.ℓmax, BC_Z_left )

        Update_∇Φ!( gravstate.∂Φ∂r, gravstate.∂Φ∂Z, gravstate.Φ, 
                    gravstate.f_gravity_r, gravstate.f_gravity_Z, ρ, 
                    geometry.r, geometry.Z, geometry.rCoM, geometry.ZCoM, 
                    gravstate.P, gravstate.Q,
                    BC_Z_left, gravstate.ℓmax )
    end

    # ---- Initialize time-steps where data is saved ----
    Δt_Resolve  = tmax / Nminsteps
    Δt_save     = tmax / ( Nsaved - 1 )
    save_tol    = 1e-12 * tmax                       # Tolerance to guard against round-off errors when deciding if to save data
    k           = floor((t + save_tol) / Δt_save)    # Number of completed save intervals
    save_index  = Int(k) + 1                         # Because save_index = 1 corresponds to t = 0
    t_next_save = (k + 1) * Δt_save

    if print_info
        print("\r\033[KStarting time evolution..."); flush(stdout)
        println() # New line
    end
    
    # Initialize the progress meter:
    prog = Progress( 100; dt = 0.01, desc = "Time integration progress:",
                     barglyphs=BarGlyphs("[=> ]"), barlen = 50, color=:green ) 

    # Start time integration from the last saved time t:
    while t < tmax
        # Set global hydro time-step:
        Δt_Hydro_max =  min( Δt_Resolve, tmax - t )
        Δt_Hydro     =  Compute_RHD_TimeStep( Δt_Hydro, Δt_Hydro_max, hydrostate, gravstate, RT,  
                                              LyαState, chemstate, duststate, geometry, 
                                              Gravity, RadPressure )

        # Update hydro, gravity, thermochemistry, stellar RT:
        UpdateRHD!( Δt_Hydro, hydrostate, chemstate,
                    duststate,
                    geometry, gravstate, gravcoeffs,
                    ghostgrid, 
                    LyαState, RT,
                    Gravity, RadPressure, IRfeedback, StellarWindFeedback,
                    z, mstarZAMS, Zstar_Zsun, Hydro, CoolingHeating )

        # Update time & step: 
        t    += Δt_Hydro 
        step += 1

        # ----------- Decide to save data --------------
        if t ≥ t_next_save  -  save_tol # The last factor is to avoid missing a save because round-off errors 
            save_index += 1

            if SaveData
                if Lyα
                    # Update Lyα energy density for saving:
                    Update_e_Lyα!( LyαState.e_Lyα, LyαState.Δν, LyαState.J_Lyα )
                end

                # Save data:
                SaveRHDTimeStepHDF5( FileName, hydrostate, chemstate, gravstate, duststate, RT,
                                     LyαState, save_index, t, Δt_Hydro, Gravity )
            end

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