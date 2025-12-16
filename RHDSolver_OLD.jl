
module RHDSolver

__precompile__()

include("./ConstantsAndFunctions.jl")
include("./SpatialGrid.jl")
include("./Hydrodynamics.jl")
include("PhotoChemistryAndCooling/InitializeChemistryStruct.jl")  # <- BEFORE AdvectGasFractions 
include("PhotoChemistryAndCooling/PhotoThermoChemistry.jl")
include("./PoissonGravitySolver.jl")
include("./AdvectScalars.jl")
include("./AdvectGasFractions.jl")                                 # <- AFTER its deps
include( "./SaveHDF5Data.jl" )

using .ConstantsAndFunctions
using .SpatialGrid
using .Hydrodynamics
using .InitializeChemistryStruct
using .PhotoThermoChemistry
using .PoissonGravitySolver
using .AdvectScalars
using .AdvectGasFractions
using .SaveHDF5Data
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
                           RT::StellarRTArrays,
                       rfirst::Bool    = true, 
             rfirst_StellarRT::Bool    = true,
                      Gravity::Bool    = true,
                  RadPressure::Bool    = true,
                            z::Float64 = 0.0,
                        Hydro::Bool = false,
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

    # Hydro state arrays (cell-averaged):
    ρ_new       = hydrostate.ρ_new;    ρu_r_new    = hydrostate.ρu_X_new 
    ρu_Z_new    = hydrostate.ρu_Y_new; E_new       = hydrostate.E_new 
    ρ           = hydrostate.ρ;        ρu_r        = hydrostate.ρu_X 
    ρu_Z        = hydrostate.ρu_Y;     E           = hydrostate.E
    ρ0          = hydrostate.ρ0;       ρu_r0       = hydrostate.ρu_X0 
    ρu_Z0       = hydrostate.ρu_Y0;    E0          = hydrostate.E0
    u_r         = hydrostate.u_X;      u_Z         = hydrostate.u_Y 
    P           = hydrostate.P;        e           = hydrostate.e 
    T           = hydrostate.T;        γad         = hydrostate.γad 

    # Cell center variables:
    ρc          = hydrostate.ρc;       ρu_rc       = hydrostate.ρu_Xc
    ρu_Zc       = hydrostate.ρu_Yc;    Ec          = hydrostate.Ec
    u_rc        = hydrostate.u_Xc;     u_Zc        = hydrostate.u_Yc 
    Pc          = hydrostate.Pc

    # Hydro state arrays for the RK time integration:
    ρ_stage     = hydrostate.ρ_stage;        ρu_r_stage   = hydrostate.ρu_X_stage 
    ρu_Z_stage  = hydrostate.ρu_Y_stage;     E_stage      = hydrostate.E_stage

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

    # Shock indicator:
    Shock    = hydrostate.Shock

    # Boundary conditions:
    BC_r_left = hydrostate.BC_X_left;  BC_r_right = hydrostate.BC_X_right
    BC_Z_left = hydrostate.BC_Y_left;  BC_Z_right = hydrostate.BC_Y_right

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
        UpdateHydro!( ρ_new,  ρu_r_new, ρu_Z_new, E_new,
                      ρ,      ρu_r,     ρu_Z,     E,
                      ρ0,     ρu_r0,    ρu_Z0,    E0,
                      γad,    u_r,      u_Z,
                      inv_ΔV, A_r,      A_Z,      Δt_Hydro,

                      ρ_stage,  ρu_r_stage, ρu_Z_stage, E_stage,

                      ρc,     ρu_rc,    ρu_Zc,    Ec,
                      u_rc,   u_Zc,     Pc,

                      w_i_m1_r, w_i_0_r, w_i_p1_r, w_j_m1_Z, w_j_0_Z, w_j_p1_Z, 
                      q_i_m1_r, q_i_0_r, q_i_p1_r, q_j_m1_Z, q_j_0_Z, q_j_p1_Z,  

                      w_plus_i_p1, w_minus_i_p1,   
                      w_plus_j_p1, w_minus_j_p1, 

                      ρ_L_r, ρ_R_r, u_r_L_r, u_r_R_r, u_Z_L_r, u_Z_R_r, P_L_r, P_R_r,
                      Fρ_r, Fρur_r, Fρuz_r, FE_r,
                      ρ_L_Z, ρ_R_Z, u_Z_L_Z, u_Z_R_Z, u_r_L_Z, u_r_R_Z, P_L_Z, P_R_Z,
                      Fρ_Z, Fρur_Z, Fρuz_Z, FE_Z,
                      P, ρg_old, u_rg_old, u_Zg_old, 
                      Pg_old, 
                      rg, rCoMg, Zg,  ZCoMg, 

                      Val(:Cylindrical), inv_r_cell, rfirst,
                      Shock,
                      BC_r_left, BC_r_right,
                      BC_Z_left, BC_Z_right )  
    end

    # Gravity solver + kick update:
    if Gravity && Hydro 
        GravitySolveAndKick!( Δt_Hydro, ρu_r, ρu_r0, ρu_Z, ρu_Z0,  
                              ρ, ρ0, E, 
                              gravstate.Φ, gravstate.Φ_half, gravstate.Φ_old, gravstate.Φ_snapshot,
                              gravstate.∂Φ∂r, gravstate.∂Φ∂Z, 
                              gravcoeffs.RPoiss_ijp, gravcoeffs.RPoiss_ij, gravcoeffs.RPoiss_ijm, 
                              gravcoeffs.ZPoiss_ijp, gravcoeffs.ZPoiss_ij, gravcoeffs.ZPoiss_ijm,
                              gravcoeffs.Δτ_max, 
                              gravcoeffs.Δτ_min, 
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

    ###############################################################
    #######                                                 #######
    #######  S T E L L A R - R T  +  S U B - C Y C L I N G  #######
    #######                                                 #######
    ###############################################################
    # Below we sub-cycle the RT, chemistry, and heating/cooling. 
    # We split the hydro step into (potentially) smaller RT CFL time-steps,
    # and for each RT CFL time-step we sub-cycle chemistry and heating/cooling.

    # Determine an updated reduced speed of light for the radiative transfer:
    c̃ = FindRSL_StellarRT( c̃, u_r, u_Z, P, ρ, γad )

    t_RT = 0.0 # Initialize the RT time 
    while t_RT < Δt_Hydro
        # Set maximum RT transport time-step:
        Δt_RT_max = max( Δt_Hydro - t_RT, 0.0 )

        # Stellar RT + thermochemistry sub-cycling:
        Δt_RT = UpdateStellarJandThermoChemistry!( Δt_RT_max,  c̃,
                                                   RT,
                                                   hydrostate.e,    hydrostate.ρ,
                                                   chemstate.xH2,   chemstate.xHII,
                                                   chemstate.xHD,   chemstate.xDII,
                                                   chemstate.xHeII, chemstate.xHeIII,
                                                   z, CoolingHeating,
                                                   inv_ΔV, A_r, A_Z, 
                                                   r, rCoM, Z, ZCoM, inv_r_cell,
                                                   rfirst_StellarRT )

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
        @threads for j in dN_Z
            @simd for i in dN_r
                @inbounds begin
                    # Add radiation pressure momenta to the gas:
                    ρu_r[i,j] += RT.Δρu_r_rad[i,j]
                    ρu_Z[i,j] += RT.Δρu_Z_rad[i,j]
                end
            end # End of i-loop
        end # End of j-loop 

        # NOTE: Work from radiation pressure is handled below, by updaing E.
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

    return rfirst_StellarRT, c̃
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
    rc       = 0.0
    Zc       = 0.5 * Zmax

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

    if SaveData
        CreateHDF5File( FileName, N_r, N_Z, Nsaved )
        SaveGridHDF5File( FileName, geometry )
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
                                                 :Outflow, :Outflow   # Lower & outer Z-BCs
                                                  )

    ρ   = hydrostate.ρ
    u_r = hydrostate.u_X 
    u_Z = hydrostate.u_Y
    E   = hydrostate.E
    P   = hydrostate.P 
    e   = hydrostate.e 
    γad = hydrostate.γad

    # Set up chemistry state struct, containing arrays for all species tracked,
    # e.g. xHII, xH2 etc. :
    chemstate = InitializeChemistryArrays( N_r, N_Z )

    # Initialize cloud properties:
    @threads for j in dN_Z 
        @simd for i in dN_r 
            @inbounds begin

                # Distance from cloud center at the edge of this cell:
                R = sqrt( r[i+1] * r[i+1]  +   (Z[j+1] - Zc)^2 )

                # Gas density:
                if R < Rcl 
                    ρ[i,j] = max( nH₀ * mH / X, 10 * ρ_floor )
                else 
                    ρ[i,j] = max( nH₀ * mH / X, 10 * ρ_floor )
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

                # Gas pressure (just needed to set the right initial CFL time-step):
                if R < Rcl
                    P[i,j] = ρ[i,j] * kB * T₀ / ( Getμ(xHII₀, xH2₀, xHeII₀, xHeIII₀) * mH )
                else
                    P[i,j] = ρ[i,j] * kB * T₀ / ( Getμ(xHII₀, xH2₀, xHeII₀, xHeIII₀) * mH )
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
    #######                S T E L L A R  R T               #######
    #######                                                 #######
    ############################################################### 

    # Generate all stellar RT arrays, including J, H, inteface fluxes, 
    # emissivities, and luminosities, for all 6 stellar RT bands:
    RT = GenerateStellarRTArrays( mstarZAMS, Zstar_Zsun, ΔV, r, Z )


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
    c̃                = 300 * kms  # Initialize the RSL for M1 RT (will be updated adaptively each time-step)
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
            Δt_RP   = GetRadPressureTimeStep( r, Z, RT.a_r_rad, RT.a_Z_rad )
        else
            # Ignore limit in absence of radiation pressure feedback:
            Δt_RP   = Inf 
        end 

        # Set hydro time-step:
        Δt_Hydro = min( 1.25 * Δt_Hydro, Δt_CFL, Δt_RP, Δt_Grav, Δt_Resolve, tmax - t )

        #println("Maximum density (cm^-3): $(maximum(ρ)/mH)")


        # Update hydro, gravity, thermochemistry, stellar RT:
        rfirst_StellarRT, c̃ = UpdateRHD!( Δt_Hydro, c̃, hydrostate, chemstate,
                                          geometry, gravstate, gravcoeffs,
                                          ghostgrid, 
                                          RT,
                                          rfirst, rfirst_StellarRT, Gravity, RadPressure,
                                          z, Hydro, CoolingHeating )

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
                SaveRHDTimeStepHDF5( FileName, hydrostate, chemstate, RT.J_EUV1,
                                     save_index, t )
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