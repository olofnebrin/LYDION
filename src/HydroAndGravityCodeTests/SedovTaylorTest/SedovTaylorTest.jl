
module SedovTaylorTest

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

export SetupSedovTaylorTest



""" This function performs a hydro time-step Δt_Hydro, for the Sedov-Taylor test. """
function UpdateSedovTest!( Δt_Hydro::Float64,
                         hydrostate::HydrodynamicsState{Float64},
                           geometry::GeometryCylindrical{Float64},
                          ghostgrid::GhostPaddedGrid{Float64}
                          )

    # Geometry arrays:
    inv_ΔV      = geometry.inv_ΔV
    A_r         = geometry.A_r
    A_Z         = geometry.A_Z
    inv_r_cell  = geometry.inv_r_cell

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
    ρ           = hydrostate.ρ;        ρu_r        = hydrostate.ρu_X 
    ρu_Z        = hydrostate.ρu_Y;     E           = hydrostate.E
    u_r         = hydrostate.u_X;      u_Z         = hydrostate.u_Y 
    P           = hydrostate.P;        γad         = hydrostate.γad 

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
    UpdateHydro!( Δt_Hydro, hydrostate, ghostgrid, inv_ΔV, A_r, A_Z, 

                 w_i_m1_r, w_i_0_r, w_i_p1_r, w_j_m1_Z, w_j_0_Z, w_j_p1_Z, 
                 q_i_m1_r, q_i_0_r, q_i_p1_r, q_j_m1_Z, q_j_0_Z, q_j_p1_Z,  

              w_plus_i_p1, w_minus_i_p1,  w_plus_j_p1, w_minus_j_p1,

                 Val(:Cylindrical), inv_r_cell )  

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

    return nothing 
end



function SetupSedovTaylorTest( ρ₀::Float64,
                                γ::Float64,
                           Eblast::Float64,
                             Rmax::Float64,
                             Zmax::Float64,
                             tmax::Float64,
                          N_r_lin::Int  = 64,
                          N_Z_lin::Int  = 64,
                        Nminsteps::Int  = 300,
                        Nmaxsteps::Int  = 100000,
                           Nsaved::Int  = 300,
                       print_info::Bool = true
                               )

    # Setup uniform/linear cylindrical grid:
    geometry = SetupCylindricalGrid( Rmax, Rmax, Zmax, Zmax, 
                                     N_r_lin, 0, 
                                     N_Z_lin, 0 )

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

    # Create file:
    outdir   = @__DIR__    # .../src/CodeTests/SedovTaylorTest
    FileName = joinpath( outdir, @sprintf( "SedovTest_E%.2e_rho%.2g_gamma%.2f_t%.1f_%dx%d.h5",
                         Eblast, ρ₀, γ, tmax, N_r_lin, N_Z_lin ) )
    CreateHDF5File( FileName, N_r, N_Z, Nsaved )
    SaveGridHDF5File( FileName, geometry )


    ###############################################################
    #######                                                 #######
    #######   H Y D R O  &  T H E R M O C H E M I S T R Y   #######
    #######                                                 #######
    ###############################################################

    if print_info 
        print("\r\033[KInitialising hydro arrays..."); flush(stdout)
    end 

    # We have no passive scalars:
    N_Scalars = 0

    # Set up hydro state + ghost grid arrays:
    hydrostate, ghostgrid = GenerateHydroArrays( r, rCoM, Z, ZCoM, γ, N_Scalars,
                                                 :Reflect, :Reflect,
                                                 :Reflect, :Reflect )

    # Hydro arrays:
    ρ    = hydrostate.ρ
    u_r  = hydrostate.u_X 
    u_Z  = hydrostate.u_Y
    E    = hydrostate.E
    P    = hydrostate.P 
    e    = hydrostate.e 
    γad  = hydrostate.γad

    # Dust arrays (not really active in the test, but needs to be initialized)
    ρd   = hydrostate.ρd
    ud_r = hydrostate.ud_X 
    ud_Z = hydrostate.ud_Y

    # Get blastwave volume:
    Vblast = 0.0                  # Initialize blast volume 
    Rblast = 5 * ( r[2] - r[1] )  # Dump blast in a volume of radius ~ 5 cells 
    for j in dN_Z
        for i in dN_r 
            # Distance from cloud center at the edge of this cell:
            R = sqrt( r[i+1] * r[i+1]  +   Z[j+1] * Z[j+1] )

            if R < Rblast 
                Vblast += ΔV[i,j]
            end
        end
    end

    # Initialize cloud properties:
    @threads for j in dN_Z 
        @simd for i in dN_r 
            @inbounds begin

                # Initial gas density & adiabatic index:
                ρ[i,j]   = ρ₀
                γad[i,j] = γ

                # Set dust density to arbitrary value (it's not coupled to gas):
                ρd[i,j]  = max( 1e-5 * ρ₀, ρd_floor )

                # Distance from cloud center at the edge of this cell:
                R = sqrt( r[i+1] * r[i+1]  +   Z[j+1] * Z[j+1] )

                # Set gas energy density:
                if R < Rblast 
                    E[i,j] = 0.5 * Eblast / Vblast    # Factor of half because we only simulate half of the Z-axis
                else
                    E[i,j] = 1e-10 * Eblast / Vblast 
                end

                # Set the gas pressure, assuming initially static gas:
                P[i,j] = max( (γad[i,j] - 1) * E[i,j], P_floor )
                
            end
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
        Δt_CFL = GetCFLTimeStep( r, Z, ρ, P, u_r, u_Z, ud_r, ud_Z, γad )  

        # Set hydro time-step:
        Δt_Hydro = min( 1.25 * Δt_Hydro, Δt_CFL, Δt_Resolve, tmax - t )

        # Update hydro:
        UpdateSedovTest!( Δt_Hydro, hydrostate, geometry,
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