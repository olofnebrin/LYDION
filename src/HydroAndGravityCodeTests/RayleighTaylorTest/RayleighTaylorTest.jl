
module RayleighTaylorTest

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

export SetupRTTest



""" This function performs a hydro time-step Δt_Hydro, for the Rayleigh-Taylor test. """
function UpdateRTTest!( Δt_Hydro::Float64,
                      hydrostate::HydrodynamicsState{Float64},
                        geometry::GeometryCartesian{Float64},
                       ghostgrid::GhostPaddedGrid{Float64},
                            ∂Φ∂X::AbstractMatrix{Float64},
                            ∂Φ∂Y::AbstractMatrix{Float64}
                          )

    # Hydro state arrays (cell-averaged):
    ρ           = hydrostate.ρ;        ρu_X        = hydrostate.ρu_X 
    ρu_Y        = hydrostate.ρu_Y;     E           = hydrostate.E
    u_X         = hydrostate.u_X;      u_Y         = hydrostate.u_Y 
    P           = hydrostate.P;        γad         = hydrostate.γad 
    e           = hydrostate.e_dual

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
    UpdateHydro!( Δt_Hydro, hydrostate, ghostgrid, geometry )   

    ###############################################################
    #######                                                 #######
    #######            G R A V I T Y - K I C K              #######
    #######                                                 #######
    ###############################################################

    # Gravity kick to gas (not dust):
    GravityKick!( Δt_Hydro, 
                  ρu_X, ρu_Y, ρ, E,
                  ∂Φ∂X, ∂Φ∂Y )

    # Update primitives to reflect latest conservatives,
    # so that we can set the CFL time-step for the next time-step:
    @threads for j in dN_Y 
        @inbounds @simd for i in dN_X
            # Recover primitives at geometric cell centers, from the 
            # conservative variables at the cell centers:
            u_X[i,j], u_Y[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_X[i,j], ρu_Y[i,j], E[i,j], e[i,j], γad[i,j] )

            # Ensure positive pressure:
            P[i,j] = max( P[i,j], P_floor )
        end
    end

    return nothing 
end



function SetupRTTest( tmax::Float64,
                   N_X_lin::Int  = 256,
                   N_Y_lin::Int  = 256,
                 Nminsteps::Int  = 300,
                 Nmaxsteps::Int  = 100000,
                    Nsaved::Int  = 300,
                print_info::Bool = true
                      )

    # Setup uniform/linear Cartesian grid:
    Xmax     = 0.5
    Ymax     = 1.5
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

    @assert iseven( N_X ) "This test assumes even N_X to get perfectly symmetric ICs."

    # Create file:
    outdir   = @__DIR__    # .../src/CodeTests/RayleighTaylorTest
    FileName = joinpath( outdir, @sprintf( "RayleighTaylor_%03dx%03d_t%.2f.h5",
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
    hydrostate, ghostgrid = GenerateHydroArrays( X, XCoM, Y, YCoM, 7 / 5, N_Scalars,
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
    e    = hydrostate.e 
    γad  = hydrostate.γad

    # Dust arrays (not coupled to gas, but need to be initialized): 
    ρd   = hydrostate.ρd
    ud_X = hydrostate.ud_X 
    ud_Y = hydrostate.ud_Y

    # Setup gravitational potential gradient arrays:
    ∂Φ∂X = zeros( Float64, N_X, N_Y )
    ∂Φ∂Y = zeros( Float64, N_X, N_Y )

    # Constants:
    P0   = 2.5              # Constant for the initial gas pressure 
    γ    = 7 / 5            # Adiabatic index 
    Yc   = 0.5 * Ymax 
    Xc   = 0.5 * Xmax 

    # Find the index corresponding to the midpoint along the X-axis:
    imid = N_X ÷ 2

    # Fill the Y-velocity to the left of midpoint:
    for j in dN_Y 
        for i in 1:imid
            u_Y[i,j] = 0.01 * (1 + cos( 4 * π * ( XCoM[i] - Xc) )) * (1 + cos( 3 * π * (YCoM[j] - Yc) )) / 4
        end
    end

    # Fill the Y-velocity to the right of the midpoint, using exact mirroring to avoid 
    # tiny floating point errors that could cause asymmetry:
    for j in dN_Y 
        for i in (imid + 1):N_X
            imirror  = N_X + 1 - i    # Index i should mirror index imirror
            u_Y[i,j] = u_Y[imirror,j]
        end
    end

    # Proceed and fill variables:
    for j in dN_Y 
        for i in dN_X 

            # Set initial gas density:
            if YCoM[j] < Yc
                ρ[i,j] = 1.0
            else
                ρ[i,j] = 2.0
            end

            # Set gas adiabatic index:
            γad[i,j]  = γ

            # Set gravitational potential gradient:
            ∂Φ∂X[i,j] = 0.0
            ∂Φ∂Y[i,j] = 0.1 

            # Set initial gas pressure, such that we have hydrostatic equilibrium (for the assumed Φ):
            P[i,j]    = P0  -  0.1 * ( YCoM[j] - Yc ) * ρ[i,j]

            # Set Y-momentum:
            ρu_Y[i,j] = ρ[i,j] * u_Y[i,j]

            # Set X-momentum (no initial momentum along X):
            ρu_X[i,j] = 0.0

            # Set gas energy:
            E[i,j]      = P[i,j] / (γad[i,j] - 1)  +  (ρu_X[i,j]^2 + ρu_Y[i,j]^2) / (2 * ρ[i,j])

            # Set dust density to be tiny (not coupled to gas):
            ρd[i,j]     = max( 1e-5 * ρ[i,j], ρd_floor )

        end # End of i-loop 
    end # End of j-loop

    # Check that the ICs are symmetric enough:
    for j in dN_Y
        for i in dN_X
            imirror  = N_X + 1 - i

            @assert ρ[i,j]   ≈ ρ[imirror,j]   atol = 1e-15
            @assert P[i,j]   ≈ P[imirror,j]   atol = 1e-15
            @assert u_Y[i,j] ≈ u_Y[imirror,j] atol = 1e-15
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

    t                = 0.0     # Initialize time 
    step             = 1       # Initialize the steps
    Δt_Hydro         = Inf     # Initialize the hydro time-step 
    while t < tmax

        # Hydro CFL time-step:
        Δt_CFL = GetCFLTimeStep( X, Y, ρ, P, u_X, u_Y, ud_X, ud_Y, γad )  

        # Set hydro time-step:
        Δt_Hydro = min( 1.25 * Δt_Hydro, Δt_CFL, Δt_Resolve, tmax - t )

        # Update hydro:
        UpdateRTTest!( Δt_Hydro, hydrostate, geometry,
                       ghostgrid, ∂Φ∂X, ∂Φ∂Y )

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