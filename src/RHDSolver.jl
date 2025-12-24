
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
include( "./ExponentialScheme.jl" )
include( "./TridiagonalSolver.jl" )
include( "./M1Closure.jl" ) 
include( "./AdvectJandH.jl" )
include( "./SetupLymanAlphaArrays.jl" )
include( "./SpatialGrid.jl" )
include( "./PoissonGravitySolver.jl" )
include( "./Hydrodynamics.jl" )

include( "./UpdateLyaJandH.jl" )
include( "PhotoChemistryAndCooling/InitializeChemistryStruct.jl" )  # <- BEFORE AdvectGasFractions 
include( "PhotoChemistryAndCooling/PhotoThermoChemistry.jl" )
include( "./AdvectScalars.jl" )
include( "./AdvectGasFractions.jl" )                                 # <- AFTER its deps
include( "./Turbulence.jl" )
include( "./SaveHDF5Data_RHD.jl" )

using  Base.Threads              # For CPU acceleration
using  ProgressMeter             # To show time step progress
using  Dates, Printf

export SetupRHDsimulation, RestartRHDsimulation


""" This function performs a full RHD update over a hydro time-step Δt_Hydro. """
function UpdateRHD!( Δt_Hydro::Float64,
                   hydrostate::HydrodynamicsState{Float64},
                    chemstate::ChemistryState{Float64},
                     geometry::GeometryCylindrical{Float64},
                    gravstate::GravityState{Float64},
                   gravcoeffs::GravityCoeffs{Float64},
                    ghostgrid::GhostPaddedGrid{Float64},
                     LyaState::Union{LyαArrays, Nothing},
                           RT::StellarRTArrays,
                      Gravity::Bool    = true,
                  RadPressure::Bool    = true,
                            z::Float64 = 0.0,
                        Hydro::Bool    = false,
               CoolingHeating::Bool    = false,
                          Lyα::Bool    = true
                     )

    if Lyα
        @assert LyaState !== nothing "UpdateRHD!: Lyα = true but LyaState is nothing."
    else
        @assert LyaState === nothing "UpdateRHD!: Lyα = false but LyaState is not nothing."
    end

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

    # Update hydro variables by advection:
    if Hydro
        UpdateHydro!( Δt_Hydro, hydrostate, ghostgrid, inv_ΔV, A_r, A_Z,

                      w_i_m1_r, w_i_0_r, w_i_p1_r, w_j_m1_Z, w_j_0_Z, w_j_p1_Z, 
                      q_i_m1_r, q_i_0_r, q_i_p1_r, q_j_m1_Z, q_j_0_Z, q_j_p1_Z,  

                      w_plus_i_p1, w_minus_i_p1,   
                      w_plus_j_p1, w_minus_j_p1, 

                      Val(:Cylindrical), inv_r_cell )  

        # Update total mass in simulation volume (for diagnostic purposes):
        UpdateTotalMass!( hydrostate, ΔV )
    end

    # Gravity solver + kick update:
    if Gravity && Hydro 
        GravitySolveAndKick!( Δt_Hydro, hydrostate, gravstate, gravcoeffs,
                              ΔV, rCoM, ZCoM )

    end

    # Update primitives to reflect latest conservatives,
    # so that we can set the CFL time-step for the scalar advection:
    @threads for j in dN_Z 
        @inbounds @simd for i in dN_r
            # Recover primitives at geometric cell centers, from the 
            # conservative variables at the cell centers:
            u_r[i,j], u_Z[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_r[i,j], ρu_Z[i,j], E[i,j], γad[i,j] )

            # Ensure positive pressure:
            P[i,j] = max( P[i,j], P_floor )
        end
    end

    # Advect all gas fractions (xHII, xHeII etc):
    if Hydro
        Δt_CFL_AdvectScalars = GetCFLTimeStep( r, Z, ρ, P, u_r, u_Z, γad )  # Get CFL time-step for *updated* fluid variables 
        t_AdvectScalars      = 0.0
        while t_AdvectScalars < Δt_Hydro
            # Set sub-step time-step, limited by the CFL time-step & remaining time to make 
            # up hydro time-step:
            Δt_AdvectScalars_max = max( 0.0, Δt_Hydro - t_AdvectScalars )
            Δt_AdvectScalars     = min( Δt_CFL_AdvectScalars, Δt_AdvectScalars_max ) 

            # Advect all gas fractions:
            AdvectGasFractions!( Δt_AdvectScalars, chemstate, hydrostate, geometry )

            # Update sub-step time: 
            t_AdvectScalars += Δt_AdvectScalars

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
    if Lyα
        Reset_Δρu!( LyaState.Δρu_Lyα_r, LyaState.Δρu_Lyα_Z )
    end
    Reset_Δρu!( RT.Δρu_stellar_r,   RT.Δρu_stellar_Z )

    # Determine an updated reduced speed of light for the radiative transfer:
    if Lyα
        c̃     = FindRSL_StellarRT( RT.c̃, u_r, u_Z, P, ρ, γad )
        c̃_Lyα = FindRSL_LyαRT( LyaState.c̃_Lyα, u_r, u_Z, P, ρ, T, 
                               chemstate.xHII, chemstate.xH2, γad, r, Z )

        c̃max  = max( c̃, c̃_Lyα )
        
        # Set stellar and Lyα RT reduced speed of light to be the same, and 
        # maximum of the two:
        LyaState.c̃_Lyα = c̃max
        RT.c̃           = c̃max
    else
        RT.c̃  = FindRSL_StellarRT( RT.c̃, u_r, u_Z, P, ρ, γad )
    end


    t_RT  = 0.0 # Initialize the RT time 
    while t_RT < Δt_Hydro
        # Set maximum RT transport time-step:
        Δt_RT_max = max( Δt_Hydro - t_RT, 0.0 )
 
        # Stellar RT + Lyα RT + thermochemistry sub-cycling:
        Δt_RT = UpdateRTandThermoChemistry!( Δt_RT_max, 
                                             RT,
                                             hydrostate,
                                             chemstate,
                                             z, CoolingHeating,
                                             inv_ΔV, A_r, A_Z, 
                                             r, rCoM, Z, ZCoM, inv_r_cell,
                                             u_r_star_r, u_Z_star_Z, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r, 
                                             LyaState, Lyα )

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
                    # Add stellar radiation pressure momenta to the gas:
                    ρu_r[i,j] += RT.Δρu_stellar_r[i,j]
                    ρu_Z[i,j] += RT.Δρu_stellar_Z[i,j]

                    # Add Lyα radiation pressure momentum to the gas:
                    if Lyα
                        ρu_r[i,j] += LyaState.Δρu_Lyα_r[i,j]
                        ρu_Z[i,j] += LyaState.Δρu_Lyα_Z[i,j]
                    end

                    # Update stellar radiation pressure forces (Lyα case is done below),
                    # mainly for setting the radiation pressure time-step limiter:
                    RT.f_stellar_r[i,j] = RT.Δρu_stellar_r[i,j] * inv_Δt_Hydro
                    RT.f_stellar_Z[i,j] = RT.Δρu_stellar_Z[i,j] * inv_Δt_Hydro
                end
            end # End of i-loop
        end # End of j-loop 

        # NOTE: Work from radiation pressure is handled below, by updaing E.
    end

    # Update LyC and Lyman-Werner escape fractions (just diagnostics):
    Compute_fesc_LyC_and_LW_Cylindrical!( RT, A_r, A_Z, BC_Z_bot )


    # Update Lyα forces f_Lyα, force multiplier M_F, luminosity L_Lyα, and Lyα escape fraction
    # (all of these mainly for diagnostics and for setting the radiation pressure time-step):
    if Lyα
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
                μ        = Getμ( chemstate.xHII[i,j],  chemstate.xH2[i,j], 
                                 chemstate.xHeII[i,j], chemstate.xHeIII[i,j] )

                γad[i,j] = Getγad( chemstate.xHII[i,j],  chemstate.xH2[i,j], 
                                   chemstate.xHeII[i,j], chemstate.xHeIII[i,j] )

                # Get the latest temperature (from after thermochemistry update):
                T[i,j]   = e[i,j] * μ * (γad[i,j] - 1) * mH / ( ρ[i,j] * kB )

                # Update gas energy to reflect new temperature & momentum from radiation pressure:
                E[i,j]   = e[i,j]  +  ( ρu_r[i,j]^2 + ρu_Z[i,j]^2 ) / ( 2 * ρ[i,j] )


                # Also update gas pressure & velocities, needed for the computing 
                # the CFL time-step:
                P[i,j]   = max( e[i,j] * ( γad[i,j] - 1 ), P_floor )
                u_r[i,j] = ρu_r[i,j] / ρ[i,j]
                u_Z[i,j] = ρu_Z[i,j] / ρ[i,j]

            end
        end # End of i-loop 
    end # End of j-loop 

    return nothing
end


function SetupRHDsimulation( mstarZAMS::Float64,  # Stellar mass (M⊙)
                            Zstar_Zsun::Float64,  # Stellar metallicity Z_star / Z⊙
                                   nH₀::Float64,  # H density (cm⁻³) inside cloud 
                                nH_ext::Float64,  # H density (cm⁻³) far outside cloud 
                                   Rcl::Float64,  # Radius of 'cloud', i.e. higher density region
                               Zcl_pos::Float64,  # Initial central position of the cloud along the Z-axis 
                             Zstar_pos::Float64,  # Central position of the star along the Z-axis 
                            σstar_cell::Float64,  # Gaussian σ (in units of cell size) that determines extent of spatially smeared star
                              Rmax_lin::Float64,
                              Zmax_lin::Float64,
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
                              xHII_ext::Float64,  # H II fraction outside the cloud 
                               xH2_ext::Float64,  # H₂ fraction outside the cloud 
                             xHeII_ext::Float64,  # He II fraction outside the cloud 
                            xHeIII_ext::Float64,  # He III fraction outside the cloud 
                              xDII_ext::Float64,  # D II fraction outside the cloud 
                               xHD_ext::Float64,  # HD fraction outside the cloud 
                                     z::Float64,
                                 Hydro::Bool    = true,     # Hydrodynamics (on/off)
                                   Lyα::Bool    = true,     # Lyα RT (on/off)
                                Recoil::Bool    = true,
                        LyαDestruction::Bool    = true,
                               Gravity::Bool    = true,
                           RadPressure::Bool    = true,
                             u_RMSturb::Float64 = 0.0,      # Initial turbulent RMS velocity 
                               N_r_lin::Int     = 64,
                               N_Z_lin::Int     = 64,
                               N_r_log::Int     = 64,
                               N_Z_log::Int     = 64,
                            N_ν_coarse::Int     = 200,
                              N_ν_fine::Int     = 100,
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

    # Initialize the RSL for M1 stellar & Lyα RT (will be updated adaptively each time-step)
    c̃        = 300 * kms  

    # Setup uniform cylindrical grid:
    geometry = SetupCylindricalGrid( Rmax, Rmax_lin, Zmax, Zmax_lin, 
                                     N_r_lin, N_r_log, 
                                     N_Z_lin, N_Z_log )

    r        = geometry.r
    rCoM     = geometry.rCoM
    Z        = geometry.Z
    ZCoM     = geometry.ZCoM
    ΔV       = geometry.ΔV

    # Grid intervals:
    N_r      = length( rCoM )
    N_Z      = length( ZCoM )
    dN_r     = 1:N_r 
    dN_Z     = 1:N_Z


    if print_info
        print("\r\033[KNumber of threads used: $(Threads.nthreads())"); flush(stdout)
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
    BC_r_left::Symbol     = :Reflect # Always reflective BCs at inner r-boundary in cylindrical coordinates
    hydrostate, ghostgrid = GenerateHydroArrays( r, rCoM, Z, ZCoM, 5 / 3,
                                                 BC_r_left, BC_r_right,  # Inner & outer r-BCs 
                                                 BC_Z_left, BC_Z_right   # Lower & outer Z-BCs
                                                  )

    ρ    = hydrostate.ρ
    ρu_r = hydrostate.ρu_X
    ρu_Z = hydrostate.ρu_Y
    u_r  = hydrostate.u_X 
    u_Z  = hydrostate.u_Y
    E    = hydrostate.E
    P    = hydrostate.P 
    e    = hydrostate.e 
    T    = hydrostate.T
    γad  = hydrostate.γad

    # Initialize turbulent velocity fluctuations:
    SetupTurbulence!( u_RMSturb, Rcl, Zcl_pos, u_r, u_Z, r, Z, rCoM, ZCoM, ΔV )

    # Set up chemistry state struct, containing arrays for all species tracked,
    # e.g. xHII, xH2 etc. :
    chemstate = InitializeChemistryArrays( N_r, N_Z )

    # Cloud density properties:
    ρcl  = max( nH₀ * mH / X,  10 * ρ_floor )                     # Density within cloud 
    ρext = max( nH_ext * mH / X,  10 * ρ_floor )                  # Density far outside the cloud
    δr   = max( 2 * max( r[2] - r[1], Z[2] - Z[1] ), 0.05 * Rcl ) # Determines fall-off in density outside cloud

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
                ρu_r[i,j] = ρ[i,j] * u_r[i,j]
                ρu_Z[i,j] = ρ[i,j] * u_Z[i,j]

                # Gas internal energy density e = P / (γ - 1) = ρ kB T₀ / ((γ - 1) μ mH):
                e[i,j]    = P[i,j] / ( γad[i,j] - 1 )

                # Energy density of gas, E = e + E_kin:
                E[i,j]    = e[i,j]  +  0.5 * ρ[i,j] * (u_r[i,j]^2 + u_Z[i,j]^2)
                
            end
        end # End of i-loop 
    end # End of j-loop


    # Initialize total mass in simulation volume (for diagnostic purposes):
    UpdateTotalMass!( hydrostate, ΔV )

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
    Lyα ? GenerateLyαArrays( c̃, r, Z, T, Recoil, LyαDestruction, N_ν_coarse, N_ν_fine ) : nothing


    ###############################################################
    #######                                                 #######
    #######                S T E L L A R  R T               #######
    #######                                                 #######
    ############################################################### 

    # Generate all stellar RT arrays, including J, H, inteface fluxes, 
    # emissivities, and luminosities, for all 6 stellar RT bands:
    RT = GenerateStellarRTArrays( mstarZAMS, Zstar_Zsun, ΔV, r, Z, 
                                  Zstar_pos, σstar_cell, c̃ )

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
        CreateHDF5File( FileName, N_r, N_Z, Nsaved, Gravity, 
                        Lyα, Recoil, LyαDestruction, RadPressure, z, 
                        N_ν,
                        tmax, mstarZAMS, Zstar_Zsun, σstar_cell, 
                        Zstar_pos, BC_r_right, BC_Z_left, BC_Z_right,
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
        if Lyα
            # Update Lyα energy density before saving (will be zero - just for clarity):
            Update_e_Lyα!( LyαState.e_Lyα, LyαState.Δν, LyαState.J_Lyα )

            SaveRHDTimeStepHDF5( FileName, hydrostate, chemstate, gravstate, RT,
                                 LyαState, save_index, t, Δt_Hydro, Gravity )
        else
            SaveRHDTimeStepHDF5( FileName, hydrostate, chemstate, gravstate, RT,
                                 save_index, t, Δt_Hydro, Gravity )
        end
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

        # Hydro CFL time-step:
        Δt_CFL = GetCFLTimeStep( r, Z, ρ, P, u_r, u_Z, γad )  

        # Limit from gravitational acceleration time-scale:
        if Gravity
            Δt_Grav = GetGravityTimeStep( r, Z, gravstate.∂Φ∂r, gravstate.∂Φ∂Z )      
        else
            # Ignore limit in absence of gravity:
            Δt_Grav = Inf    
        end

        # Limit from Radiation pressure acceleration time-scale:
        if RadPressure 
            if Lyα
                Δt_RP = min( GetRadPressureTimeStep( r, Z, ρ, LyαState.f_Lyα_r, LyαState.f_Lyα_Z ),
                             GetRadPressureTimeStep( r, Z, ρ, RT.f_stellar_r,   RT.f_stellar_Z ) )
            else
                Δt_RP = GetRadPressureTimeStep( r, Z, ρ, RT.f_stellar_r,   RT.f_stellar_Z )
            end
        else
            # Ignore limit in absence of radiation pressure feedback:
            Δt_RP   = Inf 
        end

        # Set hydro time-step:
        Δt_Hydro = min( 1.25 * Δt_Hydro, Δt_CFL, Δt_RP, Δt_Grav, Δt_Resolve, tmax - t )

        # Update hydro, gravity, thermochemistry, stellar RT:
        UpdateRHD!( Δt_Hydro, hydrostate, chemstate,
                    geometry, gravstate, gravcoeffs,
                    ghostgrid, 
                    LyαState, RT,
                    Gravity, RadPressure,
                    z, Hydro, CoolingHeating, Lyα )

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

                    # Save data:
                    SaveRHDTimeStepHDF5( FileName, hydrostate, chemstate, gravstate, RT,
                                         LyαState, save_index, t, Δt_Hydro, Gravity )
                else
                    # Save data:
                    SaveRHDTimeStepHDF5( FileName, hydrostate, chemstate, gravstate, RT,
                                         save_index, t, Δt_Hydro, Gravity )
                end
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

        if step > Nmaxsteps 
            println("\nSimulation stopped: Reached maximum number of global time-steps ($(Nmaxsteps)).")
            break
        end

      
    end # End of time-integration 


    return nothing
end


""" This function can restart an RHD simulation, using data saved 
    in an HDF5 file. """
function RestartRHDsimulation( FileName::String  = "RHDSim.h5" )

    # Initialize various variables and settings that will be needed after
    # we close the h5 open:
    Gravity = false; Lyα = false; Recoil = false; RadPressure = false; Hydro = true
    LyαDestruction = true
    z = 0.0; tmax = 0.0
    t = 0.0; step = 0; Δt_Hydro = 0.0
    c̃_saved = 0.0
    BC_r_right = :Outflow; BC_Z_left = :Reflect; BC_Z_right = :Outflow

    geometry_ref   = Ref{Any}(nothing)
    hydrostate_ref = Ref{Any}(nothing)
    ghostgrid_ref  = Ref{Any}(nothing)
    chemstate_ref  = Ref{Any}(nothing)
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

        haskey(file, "Settings")   || error("RestartRHDsimulation: missing group /Settings in $FileName.")
        haskey(file, "Checkpoint") || error("RestartRHDsimulation: missing group /Checkpoint in $FileName.")

        Settings   = file["Settings"]
        CheckPoint = file["Checkpoint"]

        # Get settings:
        Gravity        = parse_bool(get_attr(Settings, "Gravity"))
        Lyα            = parse_bool(get_attr(Settings, "Lya"))  
        Recoil         = parse_bool(get_attr(Settings, "Recoil"))
        RadPressure    = parse_bool(get_attr(Settings, "RadPressure"))
        LyαDestruction = haskey(attrs(Settings), "Lya_pd_on") ? parse_bool(get_attr(Settings, "Lya_pd_on")) : true
        Hydro          = haskey(attrs(Settings), "Hydro")     ? parse_bool(get_attr(Settings, "Hydro"))     : true
        z              = haskey(attrs(Settings), "z")         ? parse_f64(get_attr(Settings, "z"))          : 0.0

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
        chemstate = InitializeChemistryArrays( N_r, N_Z )

        chemstate.xHII    .=  CheckPoint["xHII"][:, :]
        chemstate.xH2     .=  CheckPoint["xH2"][:, :]
        chemstate.xHeII   .=  CheckPoint["xHeII"][:, :]
        chemstate.xHeIII  .=  CheckPoint["xHeIII"][:, :]
        chemstate.xDII    .=  CheckPoint["xDII"][:, :]
        chemstate.xHD     .=  CheckPoint["xHD"][:, :]

        # Setup hydro state (not initialized to right values yet) + ghost grid arrays:
        BC_r_left::Symbol     = :Reflect # Always reflective BCs at inner r-boundary in cylindrical coordinates
        hydrostate, ghostgrid = GenerateHydroArrays( r, rCoM, Z, ZCoM, 5 / 3,
                                                     BC_r_left, BC_r_right,  # Inner & outer r-BCs 
                                                     BC_Z_left, BC_Z_right   # Lower & outer Z-BCs
                                                     )

        ρ     = hydrostate.ρ
        ρu_r  = hydrostate.ρu_X
        ρu_Z  = hydrostate.ρu_Y
        E     = hydrostate.E
        T     = hydrostate.T
        u_r   = hydrostate.u_X
        u_Z   = hydrostate.u_Y
        P     = hydrostate.P
        γad   = hydrostate.γad
        e     = hydrostate.e

        # Get latest hydro state from saved file:
        ρ    .= CheckPoint["GasDensity"][:, :]
        ρu_r .= CheckPoint["rMomentum"][:, :]
        ρu_Z .= CheckPoint["ZMomentum"][:, :]
        E    .= CheckPoint["Energy"][:, :]
        T    .= CheckPoint["T"][:, :]

        dN_r = 1:N_r
        dN_Z = 1:N_Z
        @threads for j in dN_Z
            @inbounds @simd for i in dN_r
                # Get adiabatic index from chemistry state:
                γad[i,j] = Getγad( chemstate.xHII[i,j], chemstate.xH2[i,j],
                                   chemstate.xHeII[i,j], chemstate.xHeIII[i,j] )

                # Get primitive variables: 
                u_r[i,j], u_Z[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_r[i,j], ρu_Z[i,j], 
                                                           E[i,j], γad[i,j] )
                P[i,j]  = max( P[i,j], P_floor )

                # Gas internal energy density: 
                e[i,j]  = P[i,j] / (γad[i,j] - 1)

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

        # Initialize struct + arrays for gravity state & coefficients:
        gravstate, gravcoeffs = 
        CreateGravityStateAndCoefficients( geometry.A_r, geometry.A_Z, geometry.inv_ΔV, 
                                           rCoM, ZCoM, r, Z )

        # Initialize stellar RT arrays/state:
        RT = GenerateStellarRTArrays( mstarZAMS, Zstar_Zsun, ΔV, r, Z, 
                                      Zstar_pos, σstar_cell, c̃_saved )

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

        # Get latest Strang-split order for hydro & stellar RT:
        hydrostate.rfirst_Hydro = parse_bool(attrs(CheckPoint)["rfirst_Hydro"])
        RT.rfirst_StellarRT     = parse_bool(attrs(CheckPoint)["rfirst_StellarRT"])

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

            # Get latest Strang-split order:
            LyαState.rfirst_Lyα   = parse_bool(attrs(CheckPoint)["rfirst_Lya"])

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
        gravstate_ref[]  = gravstate
        gravcoeffs_ref[] = gravcoeffs
        RT_ref[]         = RT
        LyαState_ref[]   = LyαState

        r_ref[]   = r
        Z_ref[]   = Z
        ρ_ref[]   = ρ
        P_ref[]   = P
        u_r_ref[] = u_r
        u_Z_ref[] = u_Z
        γad_ref[] = γad


    end # End of h5open 

    geometry   = geometry_ref[]
    hydrostate = hydrostate_ref[]
    ghostgrid  = ghostgrid_ref[]
    chemstate  = chemstate_ref[]
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

    ###############################################################
    #######                                                 #######
    #######          T I M E  I N T E G R A T I O N         #######
    #######                                                 #######
    ###############################################################

    # Same time integration code as in the SetupRHDsimulation function.

    Δt_Resolve  = tmax / Nminsteps

    # ---- Initialize time-steps where data is saved ----
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

        # Hydro CFL time-step:
        Δt_CFL = GetCFLTimeStep( r, Z, ρ, P, u_r, u_Z, γad )  

        # Limit from gravitational acceleration time-scale:
        if Gravity
            Δt_Grav = GetGravityTimeStep( r, Z, gravstate.∂Φ∂r, gravstate.∂Φ∂Z )      
        else
            # Ignore limit in absence of gravity:
            Δt_Grav = Inf    
        end

        # Limit from Radiation pressure acceleration time-scale:
        if RadPressure 
            if Lyα
                Δt_RP = min( GetRadPressureTimeStep( r, Z, ρ, LyαState.f_Lyα_r, LyαState.f_Lyα_Z ),
                             GetRadPressureTimeStep( r, Z, ρ, RT.f_stellar_r,   RT.f_stellar_Z ) )
            else
                Δt_RP = GetRadPressureTimeStep( r, Z, ρ, RT.f_stellar_r,   RT.f_stellar_Z )
            end
        else
            # Ignore limit in absence of radiation pressure feedback:
            Δt_RP   = Inf 
        end

        # Set hydro time-step:
        Δt_Hydro = min( 1.25 * Δt_Hydro, Δt_CFL, Δt_RP, Δt_Grav, Δt_Resolve, tmax - t )

        # Update hydro, gravity, thermochemistry, stellar RT:
        UpdateRHD!( Δt_Hydro, hydrostate, chemstate,
                    geometry, gravstate, gravcoeffs,
                    ghostgrid, 
                    LyαState, RT,
                    Gravity, RadPressure,
                    z, Hydro, CoolingHeating, Lyα )

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

                    # Save data:
                    SaveRHDTimeStepHDF5( FileName, hydrostate, chemstate, gravstate, RT,
                                         LyαState, save_index, t, Δt_Hydro, Gravity )
                else
                    # Save data:
                    SaveRHDTimeStepHDF5( FileName, hydrostate, chemstate, gravstate, RT,
                                         save_index, t, Δt_Hydro, Gravity )
                end
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

        if step > Nmaxsteps 
            println("\nSimulation stopped: Reached maximum number of global time-steps ($(Nmaxsteps)).")
            break
        end

      
    end # End of time-integration 


    return nothing 
end


end # End of module 