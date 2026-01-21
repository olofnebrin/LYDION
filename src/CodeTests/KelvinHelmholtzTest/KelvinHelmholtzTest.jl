
module KelvinHelmholtzTest

__precompile__()

include( "../../Structs.jl" )
include( "../../ConstantsAndFunctions.jl" )
using .Structs
using .ConstantsAndFunctions

include( "../../WENO3.jl" )
include( "../../SpatialGrid.jl" )
include( "../../Hydrodynamics.jl" )
include( "../SaveHDF5Data_HydroTests.jl" )

using  Base.Threads              # For CPU acceleration
using  ProgressMeter             # To show time step progress
using  Dates, Printf

export SetupKHTest



""" This function performs a hydro time-step Δt_Hydro, for the Kelvin-Helmholtz test. """
function UpdateKHTest!( Δt_Hydro::Float64,
                      hydrostate::HydrodynamicsState{Float64},
                        geometry::GeometryCartesian{Float64},
                       ghostgrid::GhostPaddedGrid{Float64}
                          )

    # Main geometry arrays:
    inv_ΔV      = geometry.inv_ΔV
    A_X         = geometry.A_X
    A_Y         = geometry.A_Y
    inv_r_cell  = geometry.inv_r_cell # Not needed/carries no significance in Cartesian coordinates, except as an argument

    # Weights for converting between cell center & cell-average values of 
    # conservatives & primitives:
    w_i_m1_X    = geometry.w_i_m1_X;     w_j_m1_Y     = geometry.w_j_m1_Y
    w_i_0_X     = geometry.w_i_0_X;      w_j_0_Y      = geometry.w_j_0_Y
    w_i_p1_X    = geometry.w_i_p1_X;     w_j_p1_Y     = geometry.w_j_p1_Y

    q_i_m1_X    = geometry.q_i_m1_X;     q_j_m1_Y     = geometry.q_j_m1_Y
    q_i_0_X     = geometry.q_i_0_X;      q_j_0_Y      = geometry.q_j_0_Y
    q_i_p1_X    = geometry.q_i_p1_X;     q_j_p1_Y     = geometry.q_j_p1_Y

    # WENO3 weights for interface reconstruction:
    w_plus_i_p1 = geometry.w_plus_i_p1;  w_minus_i_p1 = geometry.w_minus_i_p1   
    w_plus_j_p1 = geometry.w_plus_j_p1;  w_minus_j_p1 = geometry.w_minus_j_p1

    # Hydro state arrays (cell-averaged):
    ρ           = hydrostate.ρ;        ρu_X        = hydrostate.ρu_X 
    ρu_Y        = hydrostate.ρu_Y;     E           = hydrostate.E
    u_X         = hydrostate.u_X;      u_Y         = hydrostate.u_Y 
    P           = hydrostate.P;        γad         = hydrostate.γad 

    # Grid size & intervals:
    N_X, N_Y    = size( ρ )
    dN_X        = 1:N_X 
    dN_Y        = 1:N_Y 

    ###############################################################
    #######                                                 #######
    #######      H Y D R O / D U S T - D Y N A M I C S      #######
    #######                                                 #######
    ###############################################################

    # Update hydro variables by advection: 
    UpdateHydro!( Δt_Hydro, hydrostate, ghostgrid, inv_ΔV, A_X, A_Y, 

                  w_i_m1_X, w_i_0_X, w_i_p1_X, w_j_m1_Y, w_j_0_Y, w_j_p1_Y, 
                  q_i_m1_X, q_i_0_X, q_i_p1_X, q_j_m1_Y, q_j_0_Y, q_j_p1_Y,  

                  w_plus_i_p1, w_minus_i_p1,  w_plus_j_p1, w_minus_j_p1,

                  Val(:Cartesian), inv_r_cell ) 

    # Update primitives to reflect latest conservatives,
    # so that we can set the CFL time-step for the scalar advection:
    @threads for j in dN_Y 
        @inbounds @simd for i in dN_X
            # Recover primitives at geometric cell centers, from the 
            # conservative variables at the cell centers:
            u_X[i,j], u_Y[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_X[i,j], ρu_Y[i,j], E[i,j], γad[i,j] )

            # Ensure positive pressure:
            P[i,j] = max( P[i,j], P_floor )
        end
    end

    return nothing 
end



function SetupKHTest( tmax::Float64,
                   N_X_lin::Int  = 256,
                   N_Y_lin::Int  = 256,
                 Nminsteps::Int  = 300,
                 Nmaxsteps::Int  = 100000,
                    Nsaved::Int  = 300,
                print_info::Bool = true
                      )

    # Setup uniform/linear Cartesian grid:
    Xmax     = 1.0
    Ymax     = 1.0
    geometry = SetupCartesianGrid( Xmax, Xmax, Ymax, Ymax, 
                                   N_X_lin, 0, 
                                   N_Y_lin, 0 )

    X        = geometry.X
    XCoM     = geometry.XCoM
    Y        = geometry.Y
    YCoM     = geometry.YCoM

    # Grid intervals:
    N_X      = length( XCoM )
    N_Y      = length( YCoM )
    dN_X     = 1:N_X 
    dN_Y     = 1:N_Y

    # Create file:
    outdir   = @__DIR__    # .../src/CodeTests/KelvinHelmholtzTest
    FileName = joinpath( outdir, @sprintf( "KelvinHelmholtz_Nx%03d_Ny%03d_t%.2f.h5",
                         N_X_lin, N_Y_lin, tmax ) )
    CreateHDF5File( FileName, N_X, N_Y, Nsaved )
    SaveGridHDF5File( FileName, geometry )


    ###############################################################
    #######                                                 #######
    #######               H Y D R O - S E T U P             #######
    #######                                                 #######
    ###############################################################

    if print_info 
        print("\r\033[KInitialising hydro arrays..."); flush(stdout)
    end 

    # We have zero passive scalars:
    N_Scalars = 0

    # Set up hydro state + ghost grid arrays:
    hydrostate, ghostgrid = GenerateHydroArrays( X, XCoM, Y, YCoM, 5 / 3,
                                                 N_Scalars,
                                                 :Periodic, :Periodic,   # Periodic BCs in X 
                                                 :Reflect,  :Reflect     # Reflective BCs in Y 
                                                  )

    # Hydro arrays:
    ρ    = hydrostate.ρ
    ρu_X = hydrostate.ρu_X 
    ρu_Y = hydrostate.ρu_Y
    u_X  = hydrostate.u_X 
    u_Y  = hydrostate.u_Y
    E    = hydrostate.E
    P    = hydrostate.P 
    γad  = hydrostate.γad

    # Dust arrays (not coupled to gas, but need to be initialized): 
    ρd   = hydrostate.ρd
    ud_X = hydrostate.ud_X 
    ud_Y = hydrostate.ud_Y

    # Constants:
    ρ1  = 1.0
    ρ2  = 2.0
    ρm  = ( ρ1 - ρ2 ) / 2
    L   = 0.025
    U1  = 0.5
    U2  = - 0.5
    Um  = ( U1 - U2 ) / 2 
    P0  = 2.5              # Initial gas pressure 
    γ   = 5 / 3            # Adiabatic index 

    for j in dN_Y 
        for i in dN_X 

            # Set gas adiabatic index:
            γad[i,j] = γ

            # Set gas density & X-velocity:
            if YCoM[j] ≥ 0.0 && YCoM[j] ≤ 1/4
                ρ[i,j]    = ρ1  -  ρm * exp( (YCoM[j] - 1/4) / L )
                u_X[i,j]  = U1  -  Um * exp( (YCoM[j] - 1/4) / L )
            elseif YCoM[j] > 1/4 && YCoM[j] ≤ 1/2
                ρ[i,j]    = ρ2  +  ρm * exp( (- YCoM[j] + 1/4) / L )
                u_X[i,j]  = U2  +  Um * exp( (- YCoM[j] + 1/4) / L )
            elseif YCoM[j] > 1/2 && YCoM[j] ≤ 3/4
                ρ[i,j]    = ρ2  +  ρm * exp( - (3/4 - YCoM[j]) / L )
                u_X[i,j]  = U2  +  Um * exp( - (3/4 - YCoM[j]) / L )
            elseif YCoM[j] > 3/4 && YCoM[j] ≤ 1.0
                ρ[i,j]    = ρ1  -  ρm * exp( - (YCoM[j] - 3/4) / L )
                u_X[i,j]  = U1  -  Um * exp( - (YCoM[j] - 3/4) / L )
            end

            # Get the X-momentum:
            ρu_X[i,j] = ρ[i,j] * u_X[i,j] 

            # Setup Y-velocity perturbation, and corresponding Y-momentum:
            u_Y[i,j]  = 0.01 * sin(4 * π * XCoM[i])
            ρu_Y[i,j] = ρ[i,j] * u_Y[i,j]

            # Gas pressure is constant:
            P[i,j]    = P0

            # Gas energy:
            E[i,j]    = P[i,j] / (γad[i,j] - 1)  +  (ρu_X[i,j]^2 + ρu_Y[i,j]^2) / (2 * ρ[i,j])

            # Arbitrary initial dust density (not coupled to gas):
            ρd[i,j]   = max( 1e-5, ρd_floor )
        end # End of i-loop 
    end # End of j-loop


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

    t                = 0.0     # Initialize time 
    step             = 1       # Initialize the steps
    Δt_Hydro         = Inf     # Initialize the hydro time-step 
    while t < tmax

        # Hydro CFL time-step:
        Δt_CFL = GetCFLTimeStep( X, Y, ρ, P, u_X, u_Y, ud_X, ud_Y, γad )  

        # Set hydro time-step:
        Δt_Hydro = min( 1.25 * Δt_Hydro, Δt_CFL, Δt_Resolve, tmax - t )

        # Update hydro:
        UpdateKHTest!( Δt_Hydro, hydrostate, geometry,
                       ghostgrid )

        # Update time & step: 
        t    += Δt_Hydro 
        step += 1

        # ----------- Decide to save data --------------
        if t ≥ t_next_save  -  save_tol # The last factor is to avoid missing a save because round-off errors 
            save_index += 1

            SaveTimeStepHDF5( FileName, hydrostate,
                              save_index, t )

            # Move the 'next save' appointment forward until it is in the future 
            while t ≥ t_next_save - save_tol
                t_next_save += Δt_save
            end
        end

        # Compute progress fraction
        progress_frac = t / tmax
        # Update the progress bar description to show simulation time and percent
        prog.desc = @sprintf("t = %.2e (%.1f%%)", t, 100 * progress_frac)
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