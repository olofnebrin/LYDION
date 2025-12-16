
module SodTest

__precompile__()

include("../../ConstantsAndFunctions.jl")
include("../../SpatialGrid.jl")
include("../../Hydrodynamics.jl")
include("../SaveHDF5Data_HydroTests.jl")
using .ConstantsAndFunctions
using .SpatialGrid
using .Hydrodynamics
using .SaveHDF5Data_HydroTests
using  Base.Threads              # For CPU acceleration
using  ProgressMeter             # To show time step progress
using  Dates, Printf

export SetupSodTest



""" This function performs a hydro time-step Δt_Hydro, for the Sod shock tube test. """
function UpdateSodTest!( Δt_Hydro::Float64,
                      hydrostate::HydrodynamicsState{Float64},
                        geometry::GeometryCartesian{Float64},
                       ghostgrid::GhostPaddedGrid{Float64},
                          Xfirst::Bool    = true
                          )

    # Main geometry arrays:
    ΔV          = geometry.ΔV
    inv_ΔV      = geometry.inv_ΔV
    A_X         = geometry.A_X
    A_Y         = geometry.A_Y
    inv_r_cell  = geometry.inv_r_cell # Not needed/carries no significance in Cartesian coordinates, except as an argument
    X           = geometry.X 
    XCoM        = geometry.XCoM
    Y           = geometry.Y
    YCoM        = geometry.YCoM

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
    ρ_new       = hydrostate.ρ_new;    ρu_X_new    = hydrostate.ρu_X_new 
    ρu_Y_new    = hydrostate.ρu_Y_new; E_new       = hydrostate.E_new 
    ρ           = hydrostate.ρ;        ρu_X        = hydrostate.ρu_X 
    ρu_Y        = hydrostate.ρu_Y;     E           = hydrostate.E
    ρ0          = hydrostate.ρ0;       ρu_X0       = hydrostate.ρu_X0 
    ρu_Y0       = hydrostate.ρu_Y0;    E0          = hydrostate.E0
    u_X         = hydrostate.u_X;      u_Y         = hydrostate.u_Y 
    P           = hydrostate.P;        e           = hydrostate.e 
    T           = hydrostate.T;        γad         = hydrostate.γad 

    # Cell center variables:
    ρc          = hydrostate.ρc;       ρu_Xc       = hydrostate.ρu_Xc
    ρu_Yc       = hydrostate.ρu_Yc;    Ec          = hydrostate.Ec
    u_Xc        = hydrostate.u_Xc;     u_Yc        = hydrostate.u_Yc 
    Pc          = hydrostate.Pc

    # Hydro state arrays for the RK time integration:
    ρ_stage     = hydrostate.ρ_stage;        ρu_X_stage   = hydrostate.ρu_X_stage 
    ρu_Y_stage  = hydrostate.ρu_Y_stage;     E_stage      = hydrostate.E_stage

    # X-direction interface hydro state & fluxes:
    ρ_L_X    = hydrostate.ρ_L_X;    ρ_R_X    = hydrostate.ρ_R_X
    u_X_L_X  = hydrostate.u_X_L_X;  u_X_R_X  = hydrostate.u_X_R_X 
    u_Y_L_X  = hydrostate.u_Y_L_X;  u_Y_R_X  = hydrostate.u_Y_R_X 
    P_L_X    = hydrostate.P_L_X;    P_R_X    = hydrostate.P_R_X 
    Fρ_X     = hydrostate.F_ρ_X;    FρuX_X   = hydrostate.F_ρu_X_X
    FρuY_X   = hydrostate.F_ρu_Y_X; FE_X     = hydrostate.F_E_X

    # Y-direction interface hydro state & fluxes:
    ρ_L_Y    = hydrostate.ρ_L_Y;    ρ_R_Y    = hydrostate.ρ_R_Y
    u_Y_L_Y  = hydrostate.u_Y_L_Y;  u_Y_R_Y  = hydrostate.u_Y_R_Y
    u_X_L_Y  = hydrostate.u_X_L_Y;  u_X_R_Y  = hydrostate.u_X_R_Y 
    P_L_Y    = hydrostate.P_L_Y;    P_R_Y    = hydrostate.P_R_Y 
    Fρ_Y     = hydrostate.F_ρ_Y;    FρuX_Y   = hydrostate.F_ρu_X_Y 
    FρuY_Y   = hydrostate.F_ρu_Y_Y; FE_Y     = hydrostate.F_E_Y

    # Contact velocities:
    u_X_star_X = hydrostate.u_X_star_X
    u_Y_star_Y = hydrostate.u_Y_star_Y

    # Ghost-padded arrays:
    ρg_old   = hydrostate.ρg_old;   u_Xg_old = hydrostate.u_Xg_old
    u_Yg_old = hydrostate.u_Yg_old; Pg_old   = hydrostate.Pg_old 
    Xg       = ghostgrid.Xg;        XCoMg    = ghostgrid.XCoMg 
    Yg       = ghostgrid.Yg;        YCoMg    = ghostgrid.YCoMg

    # Shock indicator:
    Shock    = hydrostate.Shock

    # Boundary conditions:
    BC_X_left = hydrostate.BC_X_left;  BC_X_right = hydrostate.BC_X_right
    BC_Y_left = hydrostate.BC_Y_left;  BC_Y_right = hydrostate.BC_Y_right


    # Grid size & intervals:
    N_X, N_Y    = size( ρ )
    dN_X        = 1:N_X 
    dN_Y        = 1:N_Y 

    ###############################################################
    #######                                                 #######
    #######             H Y D R O D Y N A M I C S           #######
    #######                                                 #######
    ###############################################################

    # Update hydro variables by advection:
    UpdateHydro!( ρ_new,  ρu_X_new, ρu_Y_new, E_new,
                  ρ,      ρu_X,     ρu_Y,     E,
                  ρ0,     ρu_X0,    ρu_Y0,    E0,
                  γad,    u_X,      u_Y,
                  inv_ΔV, A_X,      A_Y,      Δt_Hydro,

                  ρ_stage,  ρu_X_stage, ρu_Y_stage, E_stage,

                  ρc,     ρu_Xc,    ρu_Yc,    Ec,
                  u_Xc,   u_Yc,     Pc,

                  w_i_m1_X, w_i_0_X, w_i_p1_X, w_j_m1_Y, w_j_0_Y, w_j_p1_Y, 
                  q_i_m1_X, q_i_0_X, q_i_p1_X, q_j_m1_Y, q_j_0_Y, q_j_p1_Y,  

                  w_plus_i_p1, w_minus_i_p1,   
                  w_plus_j_p1, w_minus_j_p1, 

                  ρ_L_X, ρ_R_X, u_X_L_X, u_X_R_X, u_Y_L_X, u_Y_R_X, P_L_X, P_R_X,
                  Fρ_X, FρuX_X, FρuY_X, FE_X,
                  ρ_L_Y, ρ_R_Y, u_Y_L_Y, u_Y_R_Y, u_X_L_Y, u_X_R_Y, P_L_Y, P_R_Y,
                  Fρ_Y, FρuX_Y, FρuY_Y, FE_Y,
                  P, u_X_star_X, u_Y_star_Y,
                  ρg_old, u_Xg_old, u_Yg_old, 
                  Pg_old, 
                  Xg, XCoMg, Yg,  YCoMg, 

                  Val(:Cartesian), inv_r_cell, Xfirst,
                  Shock,
                  BC_X_left, BC_X_right,
                  BC_Y_left, BC_Y_right )  

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



function SetupSodTest( tmax::Float64,
                    N_X_lin::Int  = 256,
                  Nminsteps::Int  = 300,
                  Nmaxsteps::Int  = 100000,
                     Nsaved::Int  = 300,
                 print_info::Bool = true
                      )

    # Setup uniform/linear Cartesian grid:
    N_Y_lin  = 32   # We only need a few cells along the Y-axis
    Xmax     = 1.0
    Ymax     = ( N_Y_lin / N_X_lin ) * Xmax
    geometry = SetupCartesianGrid( Xmax, Xmax, Ymax, Ymax, 
                                   N_X_lin, 0, 
                                   N_Y_lin, 0 )

    X        = geometry.X
    XCoM     = geometry.XCoM
    Y        = geometry.Y
    YCoM     = geometry.YCoM
    ΔV       = geometry.ΔV

    # Grid intervals:
    N_X      = length( XCoM )
    N_Y      = length( YCoM )
    dN_X     = 1:N_X 
    dN_Y     = 1:N_Y

    # Create file:
    FileName = @sprintf( "SodTest_%03dx%03d_t%.2f.h5",
                         N_X_lin, N_Y_lin, tmax )
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

    # Set up hydro state + ghost grid arrays:
    hydrostate, ghostgrid = GenerateHydroArrays( X, XCoM, Y, YCoM, 7 / 5,
                                                 :Reflect, :Reflect,   # Reflective BCs in X 
                                                 :Reflect, :Reflect    # Reflective BCs in Y 
                                                  )

    ρ    = hydrostate.ρ
    ρu_X = hydrostate.ρu_X 
    ρu_Y = hydrostate.ρu_Y
    u_X  = hydrostate.u_X 
    u_Y  = hydrostate.u_Y
    E    = hydrostate.E
    P    = hydrostate.P 
    e    = hydrostate.e 
    γad  = hydrostate.γad

    # Constants:
    γ   = 7 / 5  # Adiabatic index 

    for j in dN_Y 
        for i in dN_X 
            # Set gas adiabatic index:
            γad[i,j] = γ

            # Set the initial density and pressure
            if XCoM[i] ≤ 0.5
                ρ[i,j] = 1.0
                P[i,j] = 1.0
            else
                ρ[i,j] = 0.125
                P[i,j] = 0.1
            end

            # Gas energy (no initial kinetic energy):
            E[i,j]    = P[i,j] / (γad[i,j] - 1)  
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
    Xfirst           = true    # Initialize Strang-split order (1/2 X, Y, 1/2 X vs. 1/2 Y, X, 1/2 Y) for hydro 
    Δt_Hydro         = Inf     # Initialize the hydro time-step 
    while t < tmax

        # Hydro CFL time-step:
        Δt_CFL = GetCFLTimeStep( X, Y, ρ, P, u_X, u_Y, γad )  

        # Set hydro time-step:
        Δt_Hydro = min( 1.25 * Δt_Hydro, Δt_CFL, Δt_Resolve, tmax - t )

        # Update hydro:
        UpdateSodTest!( Δt_Hydro, hydrostate, geometry,
                        ghostgrid, Xfirst )

        # Update time & step: 
        t    += Δt_Hydro 
        step += 1

        # Update hydro Strang-split order:
        if Xfirst == true 
            Xfirst = false
        else
            Xfirst = true 
        end

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