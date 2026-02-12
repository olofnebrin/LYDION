module IsentropicPulseTest

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

export SetupIsentropicPulseTest

""" This function performs a hydro time-step Δt_Hydro, for the Isentropic Pulse test. """
function UpdateIsentropicPulseTest!( Δt_Hydro::Float64,
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
    #######             H Y D R O D Y N A M I C S           #######
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



function SetupIsentropicPulseTest(tmax::Float64,
                    N_X_lin::Int  = 256,
                  Nminsteps::Int  = 30,
                  Nmaxsteps::Int  = 100000,
                     Nsaved::Int  = 30,
                 print_info::Bool = true
                      )

    # Setup uniform/linear Cartesian grid:

    N_Y_lin  = N_X_lin  # Square domain
    Xmax     = 1.0
    Ymax     = 1.0
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
    outdir   = @__DIR__    # .../src/HydroAndGravityCodeTests/IsentropicPulse
    FileName = joinpath( outdir, @sprintf( "IsentropicPulseTest_%03dx%03d_t%.2f.h5",
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

    N_Scalars = 0  # We have no passive scalars
    
    # Set up hydro state + ghost grid arrays:
    hydrostate, ghostgrid = GenerateHydroArrays( X, XCoM, Y, YCoM, 5 / 3, N_Scalars,
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

    # Dust arrays (not coupled to gas, but need to be initialized): 
    ρd   = hydrostate.ρd
    ud_X = hydrostate.ud_X 
    ud_Y = hydrostate.ud_Y

    # Constants:
    γ    = 5 / 3  # Adiabatic index 
    Xc   = 0.0
    Yc   = 0.0 

    for j in dN_Y 
        for i in dN_X 
            # Set gas adiabatic index:
            γad[i,j] = γ

            # Set the initial density and pressure
            ρ[i,j]   = 1 + 0.4 * exp( - 80 * ((XCoM[i] - Xc)^2 + (YCoM[j] - Yc)^2) )
            P[i,j]   = ρ[i,j]^γ 
            
            # Gas energy (no initial kinetic energy):
            E[i,j]   = P[i,j] / (γad[i,j] - 1)  
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
        Δt_CFL   = GetCFLTimeStep( X, Y, ρ, P, u_X, u_Y, ud_X, ud_Y, γad )  

        # Set hydro time-step:
        Δt_Hydro = min( 1.25 * Δt_Hydro, Δt_CFL, Δt_Resolve, tmax - t )

        # Update hydro:
        UpdateIsentropicPulseTest!( Δt_Hydro, hydrostate, geometry,
                                    ghostgrid)

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