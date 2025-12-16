
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
include( "./SaveHDF5Data_RHD.jl" )

using  Base.Threads              # For CPU acceleration
using  ProgressMeter             # To show time step progress
using  Dates, Printf

export SetupRHDsimulation


""" This function performs a full RHD update over a hydro time-step Δt_Hydro. """
function UpdateRHD!( Δt_Hydro::Float64,
                            c̃::Float64,
                   hydrostate::HydrodynamicsState{Float64},
                    chemstate::ChemistryState{Float64},
                     geometry::GeometryCylindrical{Float64},
                    gravstate::GravityState{Float64},
                   gravcoeffs::GravityCoeffs{Float64},
                    ghostgrid::GhostPaddedGrid{Float64},
                     LyaState::LyαArrays,
                           RT::StellarRTArrays,
                       rfirst::Bool    = true, 
             rfirst_StellarRT::Bool    = true,
                      Gravity::Bool    = true,
                  RadPressure::Bool    = true,
                            z::Float64 = 0.0,
                        Hydro::Bool    = false,
               CoolingHeating::Bool    = false,
                          Lyα::Bool    = true,
                        c̃_Lyα::Float64 = c̃
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

                      Val(:Cylindrical), inv_r_cell, rfirst )  

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
    Reset_Δρu!( LyaState.Δρu_Lyα_r, LyaState.Δρu_Lyα_Z )
    Reset_Δρu!( RT.Δρu_stellar_r,   RT.Δρu_stellar_Z )

    # Determine an updated reduced speed of light for the radiative transfer:
    c̃     = FindRSL_StellarRT( c̃, u_r, u_Z, P, ρ, γad )
    c̃_Lyα = FindRSL_LyαRT( c̃_Lyα, u_r, u_Z, P, ρ, T, chemstate.xHII, chemstate.xH2, γad, r, Z )

    t_RT  = 0.0 # Initialize the RT time 
    while t_RT < Δt_Hydro
        # Set maximum RT transport time-step:
        Δt_RT_max = max( Δt_Hydro - t_RT, 0.0 )
 
        # Stellar RT + Lyα RT + thermochemistry sub-cycling:
        Δt_RT = UpdateRTandThermoChemistry!( Δt_RT_max,  c̃,
                                             RT,
                                             hydrostate,
                                             chemstate,
                                             z, CoolingHeating,
                                             inv_ΔV, A_r, A_Z, 
                                             r, rCoM, Z, ZCoM, inv_r_cell,
                                             rfirst_StellarRT,
                                             u_r_star_r, u_Z_star_Z, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r, 
                                             c̃_Lyα, LyaState, Lyα )

        # Update time:
        t_RT += Δt_RT

        # Update Strang split order for stellar RT:
        if rfirst_StellarRT == true 
            rfirst_StellarRT = false
        else
            rfirst_StellarRT = true 
        end
    end

    ###############################################################
    #######                                                 #######
    #######                 F E E D B A C K                 #######
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

    # Update Lyα forces f_Lyα, force multiplier, and M_F (all of these mainly for tracking 
    # and for setting the radiation pressure time-step):
    LyaState.M_F, LyaState.L_Lyα = Update_f_Lyα_MF_L_Lyα!( LyaState.f_Lyα_r, LyaState.f_Lyα_Z, LyaState.Δρu_Lyα_r, LyaState.Δρu_Lyα_Z, 
                                                           LyaState.∫dν_js_Lyα, ΔV, Δt_Hydro,
                                                           BC_Z_bot )


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

    return rfirst_StellarRT, c̃, c̃_Lyα
end



function SetupRHDsimulation( mstarZAMS::Float64,
                            Zstar_Zsun::Float64,
                                   nH₀::Float64,
                                   Rcl::Float64,  # Radius of 'cloud', i.e. higher density region
                              Rmax_lin::Float64,
                              Zmax_lin::Float64,
                                  Rmax::Float64,
                                  Zmax::Float64,
                                    T₀::Float64,
                                  tmax::Float64,
                                 xHII₀::Float64,
                                  xH2₀::Float64,
                                xHeII₀::Float64,
                               xHeIII₀::Float64,
                                 xDII₀::Float64,
                                  xHD₀::Float64,
                                     z::Float64,
                                 Hydro::Bool = true,
                                   Lyα::Bool = true,
                                Recoil::Bool = true,
                               Gravity::Bool = true,
                           RadPressure::Bool = true,
                               N_r_lin::Int  = 64,
                               N_Z_lin::Int  = 64,
                               N_r_log::Int  = 64,
                               N_Z_log::Int  = 64,
                              SaveData::Bool = true,
                             Nminsteps::Int  = 300,
                             Nmaxsteps::Int  = 100000,
                                Nsaved::Int  = 300,
                            print_info::Bool = true,
                              FileName::String  = "RHDSim.h5"
                             )

    # Setup uniform cylindrical grid:
    geometry = SetupCylindricalGrid( Rmax, Rmax_lin, Zmax, Zmax_lin, 
                                     N_r_lin, N_r_log, 
                                     N_Z_lin, N_Z_log )
    Zc        = 0.0 * Zmax
    Zstar_pos = 0.0 * Zmax

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
    hydrostate, ghostgrid = GenerateHydroArrays( r, rCoM, Z, ZCoM, 5 / 3,
                                                 :Reflect, :Outflow,  # Inner & outer r-BCs 
                                                 :Reflect, :Outflow   # Lower & outer Z-BCs
                                                  )

    ρ   = hydrostate.ρ
    u_r = hydrostate.u_X 
    u_Z = hydrostate.u_Y
    E   = hydrostate.E
    P   = hydrostate.P 
    e   = hydrostate.e 
    T   = hydrostate.T
    γad = hydrostate.γad

    # Set up chemistry state struct, containing arrays for all species tracked,
    # e.g. xHII, xH2 etc. :
    chemstate = InitializeChemistryArrays( N_r, N_Z )

    # Initialize cloud properties:
    @threads for j in dN_Z 
        @simd for i in dN_r 
            @inbounds begin

                # Distance from cloud center at the edge of this cell:
                R = sqrt( r[i+1]^2  +   (Z[j+1] - Zc)^2 )

                # Gas density:
                if R < Rcl 
                    ρ[i,j] = max( nH₀ * mH / X, 10 * ρ_floor )
                else 
                    ρ[i,j] = max( 1e-5 * nH₀ * mH / X, 10 * ρ_floor )
                end

                # Chemistry:
                chemstate.xHII[i,j]   = xHII₀
                chemstate.xH2[i,j]    = xH2₀
                chemstate.xHeII[i,j]  = xHeII₀
                chemstate.xHeIII[i,j] = xHeIII₀
                chemstate.xDII[i,j]   = xDII₀
                chemstate.xHD[i,j]    = xHD₀

                # Adiabatic index:
                γad[i,j] = Getγad( xHII₀, xH2₀, xHeII₀, xHeIII₀ )

                # Gas temperature & pressure (just needed to set the right initial CFL time-step):
                if R < Rcl
                    T[i,j] = T₀
                    P[i,j] = ρ[i,j] * kB * T[i,j] / ( Getμ(xHII₀, xH2₀, xHeII₀, xHeIII₀) * mH )
                else
                    T[i,j] = T₀
                    P[i,j] = ρ[i,j] * kB * T[i,j] / ( Getμ(xHII₀, xH2₀, xHeII₀, xHeIII₀) * mH )
                end

                # Gas internal energy density e = P / (γ - 1) = ρ kB T₀ / ((γ - 1) μ mH):
                e[i,j]   = P[i,j] / ( γad[i,j] - 1 )

                # Energy density of static gas, E = e:
                E[i,j]   = e[i,j] 
                
            end
        end # End of i-loop 
    end # End of j-loop

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
    LyαState = GenerateLyαArrays( r, Z, T, Recoil )

    ###############################################################
    #######                                                 #######
    #######                S T E L L A R  R T               #######
    #######                                                 #######
    ############################################################### 

    σstar_cell = 2.0

    # Generate all stellar RT arrays, including J, H, inteface fluxes, 
    # emissivities, and luminosities, for all 6 stellar RT bands:
    RT = GenerateStellarRTArrays( mstarZAMS, Zstar_Zsun, ΔV, r, Z, 
                                  Zstar_pos, σstar_cell )

    if SaveData
        N_ν = length( LyαState.νC ) # Number of Lyα frequency bins

        CreateHDF5File( FileName, N_r, N_Z, Nsaved, Lyα, N_ν )
        SaveGridHDF5File( FileName, geometry )
        if Lyα
            SaveLyαFreqGridHDF5File( FileName, LyαState )    # Save Lyα frequency grid
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
    t_next_save = 0.0  
    save_index  = 0
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
    c̃                = 300 * kms  # Initialize the RSL for M1 stellar RT (will be updated adaptively each time-step)
    c̃_Lyα            = c̃
    rfirst           = true       # Initialize Strang-split order (1/2 r, Z, 1/2 vs. 1/2 Z, r, 1/2 Z) for hydro 
    rfirst_StellarRT = true       # Initialize Strang-split order for stellar RT
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
            Δt_RP   = min( GetRadPressureTimeStep( r, Z, ρ, LyαState.f_Lyα_r, LyαState.f_Lyα_Z ),
                           GetRadPressureTimeStep( r, Z, ρ, RT.f_stellar_r,   RT.f_stellar_Z ) )
        else
            # Ignore limit in absence of radiation pressure feedback:
            Δt_RP   = Inf 
        end 

        # Set hydro time-step:
        Δt_Hydro = min( 1.25 * Δt_Hydro, Δt_CFL, Δt_RP, Δt_Grav, Δt_Resolve, tmax - t )

        # Update hydro, gravity, thermochemistry, stellar RT:
        rfirst_StellarRT, c̃, c̃_Lyα = UpdateRHD!( Δt_Hydro, c̃, hydrostate, chemstate,
                                                 geometry, gravstate, gravcoeffs,
                                                 ghostgrid, 
                                                 LyαState, RT,
                                                 rfirst, rfirst_StellarRT, Gravity, RadPressure,
                                                 z, Hydro, CoolingHeating, Lyα, c̃_Lyα )

        # Update time & step: 
        t    += Δt_Hydro 
        step += 1

        # Update hydro Strang-split order:
        if rfirst == true 
            rfirst = false
        else
            rfirst = true 
        end

        # Update (initial) stellar RT Strang-split order:
        if rfirst_StellarRT == true 
            rfirst_StellarRT = false
        else
            rfirst_StellarRT = true 
        end

        # ----------- Decide to save data --------------
        if t ≥ t_next_save  -  save_tol # The last factor is to avoid missing a save because round-off errors 
            save_index += 1

            if SaveData
                if Lyα
                    # Update Lyα energy density for saving:
                    Update_e_Lyα!( LyαState.e_Lyα, LyαState.Δν, LyαState.J_Lyα )

                    # Save data:
                    SaveRHDTimeStepHDF5( FileName, hydrostate, chemstate, RT,
                                         LyαState, save_index, t )
                else
                    # Save data:
                    SaveRHDTimeStepHDF5( FileName, hydrostate, chemstate, RT,
                                         save_index, t )
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