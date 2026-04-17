
module HydrostaticEquilibriumTest

__precompile__()

include( "../../Structs.jl" )
include( "../../ConstantsAndFunctions.jl" )
using .Structs
using .ConstantsAndFunctions

include( "../../WENO3.jl" )
include( "../../TridiagonalSolver.jl" )
include( "../../SpatialGrid.jl" )
include( "../../PoissonGravitySolver.jl" )
include( "../../Hydrodynamics.jl" )
include( "./SaveHDF5Data_HydroStaticTest.jl" )

using  Base.Threads              # For CPU acceleration
using  ProgressMeter             # To show time step progress
using  Dates, Printf
using  Roots                     # For root-finding, needed in the analytical solution

export SetupHydrostaticTest


""" The analytical polytropic density profile for Γ = 2. """
function PolytropeDensityProfile( R::Float64, α::Float64, ρc::Float64 )
    ρ = max( ρc * sin(α * R) / (α * R), 0.0 )
    return ρ
end


""" This function saves the analytical polytrope density profile in the 
    array ρ_bins_exact. Useful for later comparison. """
function SavePolytropeDensityProfile!( R_bins::AbstractArray{Float64, 1}, 
                                 ρ_bins_exact::AbstractArray{Float64, 1}, 
                                            α::Float64, 
                                           ρc::Float64 )

    N_bins = length( R_bins )
    @assert length(ρ_bins_exact) == N_bins

    for i in 1:N_bins 
        ρ_bins_exact[i] = PolytropeDensityProfile( R_bins[i], α, ρc )
    end

    return nothing 
end


function SphericalAverageProfile!( ρ_av::AbstractArray{Float64, 1},
                             R_binedges::AbstractArray{Float64, 1},
                                 V_bins::AbstractArray{Float64, 1},
                             hydrostate::HydrodynamicsState{Float64},
                               geometry::GeometryCylindrical{Float64},
                                     Zc::Float64 = 0.0 )

    ΔV    = geometry.ΔV
    rCoM  = geometry.rCoM
    ZCoM  = geometry.ZCoM
    ρ     = hydrostate.ρ

    # Intervals:
    N_r, N_Z = size( ΔV )
    Nbins    = length( R_binedges ) - 1
    dN_Z     = 1:N_Z 
    dN_r     = 1:N_r  
    dNbin    = 1:Nbins
    
    # Initialize ρ_av and V_bins to zero before we update them:
    ρ_av   .= 0.0
    V_bins .= 0.0

    @inbounds for j in dN_Z 
        for i in dN_r 
            # Get the cell distance from center:
            R = sqrt( rCoM[i]^2 + (ZCoM[j] - Zc)^2 )

            # Find nearest bin:
            for k in dNbin
                if R_binedges[k] ≤ R && R < R_binedges[k+1] 
                    # In this case we have found the right bin, so add the contribution 
                    # from ρ[i,j] to this bin:
                    ρ_av[k]   += ρ[i,j] * ΔV[i,j]

                    # Add the contribution to the bin volume:
                    V_bins[k] += ΔV[i,j]

                    # Exit the loop (we have found the bin):
                    break 
                end
            end

        end # End of i-loop 
    end # End of j-loop 

    # Normalize to get the final profile:
    @inbounds for k in dNbin
        if V_bins[k] > 0.0
            ρ_av[k] = ρ_av[k] / V_bins[k]
        else
            ρ_av[k] = 0.0
        end
    end

    return nothing
end



""" This function performs a hydro time-step Δt_Hydro, 
    for the hydrostatic equilibrium test. """
function UpdateHydrostaticTest!( Δt_Hydro::Float64,
                               hydrostate::HydrodynamicsState{Float64},
                                 geometry::GeometryCylindrical{Float64},
                                ghostgrid::GhostPaddedGrid{Float64},
                                gravstate::GravityState{Float64},
                               gravcoeffs::GravityCoeffs{Float64}
                               )

    # Some hydro state arrays (cell-averaged):
    ρ           = hydrostate.ρ;        ρu_r        = hydrostate.ρu_X 
    ρu_Z        = hydrostate.ρu_Y;     E           = hydrostate.E
    u_r         = hydrostate.u_X;      u_Z         = hydrostate.u_Y 
    P           = hydrostate.P;        γad         = hydrostate.γad 
    e           = hydrostate.e_dual

    # Dust arrays (just to reset dust momenta):
    ρdud_r      = hydrostate.ρdud_X
    ρdud_Z      = hydrostate.ρdud_Y

    # Grid size & intervals:
    N_X, N_Y    = size( ρ )
    dN_X        = 1:N_X 
    dN_Y        = 1:N_Y 

    ###############################################################
    #######                                                 #######
    #######      H Y D R O / D U S T - D Y N A M I C S      #######
    #######                                                 #######
    ###############################################################

    Δt_Hydro_half = 0.5 * Δt_Hydro

    # Gravity solver + kick update (half step):
    UpdateTotalDustDensity!( hydrostate )
    GravitySolveAndKick!( Δt_Hydro_half, hydrostate, gravstate, gravcoeffs,
                          geometry )
                  
    # Update hydro & dust variables & passive scalars by advection (full step):
    UpdateHydro!( Δt_Hydro, hydrostate, ghostgrid, geometry )  

    # Gravity solver + kick update (half step):
    UpdateTotalDustDensity!( hydrostate )
    GravitySolveAndKick!( Δt_Hydro_half, hydrostate, gravstate, gravcoeffs,
                          geometry )

    # Reset dust momenta to zero, to avoid the dust collapsing:
    ρdud_r .= 0.0
    ρdud_Z .= 0.0

    # Update primitives to reflect latest conservatives,
    # so that we can set the CFL time-step for the next time-step:
    @threads for j in dN_Y 
        @inbounds @simd for i in dN_X
            # Recover primitives at geometric cell centers, from the 
            # conservative variables at the cell centers:
            u_r[i,j], u_Z[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_r[i,j], ρu_Z[i,j], E[i,j], e[i,j], γad[i,j] )

            # Ensure positive pressure:
            P[i,j] = max( P[i,j], P_floor )
        end
    end

    return nothing 
end



function SetupHydrostaticTest( tmax_tdyn::Float64,
                                 N_r_lin::Int    = 256,
                                BC_Z_bot::Symbol = :Reflect,
                               Nminsteps::Int    = 300,
                               Nmaxsteps::Int    = 100000,
                                  Nsaved::Int    = 300,
                              print_info::Bool   = true
                                )
    # Constants:
    ρc     = 1e4 * mH          # Initial central gas density, ~  n = 10⁴ cm⁻³
    Pc     = 1e4 * kB * 100.0  # Initial central cloud pressure ~ n × kB * T, with n = 10⁴ cm⁻³, T = 100 K
    ρext   = 1e-5 * ρc         # Initial gas density outside the cloud 
    γ      = 2.0               # Adiabatic index = 2 
    K      = Pc / (ρc^γ)       # Polytropic constant, P = K ρ^γ

    # Constant that enter the polytrope density profile:
    α      = sqrt(2 * π * G / Pc) * ρc 

    # Initial cloud radius is set by α Rcl = π:
    Rcl    = π / α

    # Mass of the cloud:
    Mcl    = (4/π) * ρc * (Rcl^3)

    # Get dynamical times-cale, t_dyn = Rcl / v_esc:
    v_esc  = sqrt(2 * G * Mcl / Rcl)
    t_dyn  = Rcl / v_esc

    # Get the maximum time for the simulation:
    tmax   = tmax_tdyn * t_dyn 

    # Setup uniform cylindrical grid:
    Rmax   = 2.0 * Rcl 
    if BC_Z_bot == :Reflect
        # If we only simulate the upper midplane, then we let Zmax = Rmax:
        Zmax    = Rmax
        N_Z_lin = N_r_lin
    else
        # If we simulate the lower midplane, we use Zmax = 2 Rmax, and twice 
        # the number of cells along the Z-axis (to keep resolution along Z the same):
        Zmax    = 2 * Rmax 
        N_Z_lin = 2 * N_r_lin
    end

    if print_info
        print(""" Hydrostatic equilibrium test for a polytropic cloud with Γ = γ = 2.

                  SPATIAL RESOLUTION: $(N_r_lin) × $(N_Z_lin)
                  
                  INITIAL CLOUD PROPERTIES:
                  
                  Rcl (pc):       $(Rcl / pc)
                  Mcl (M⊙):       $(Mcl / Msun)
                  t_dyn (yr):     $(t_dyn / yr)
                  t_max / t_dyn:  $(tmax / t_dyn)

                  """)
    end

    geometry   = SetupCylindricalGrid( Rmax, Rmax, Zmax, Zmax, 
                                       N_r_lin, 0, 
                                       N_Z_lin, 0,
                                       BC_Z_bot   
                                        )

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

    # Create arrays to store the spherically averaged density profile:
    N_bins       = N_r                              # Use N_r bins for good resolution, but not too many to under-bin
    R_binedges   = LinRange(0.0, Rmax, N_bins + 1 ) # Bin edges
    ρ_bins_num   = zeros( Float64, N_bins )         # Numerical results 
    ρ_bins_exact = zeros( Float64, N_bins )         # For storing analytical profile 
    V_bins       = zeros( Float64, N_bins )

    R_bins       = zeros( Float64, N_bins )  # Bin centers 
    @inbounds for k in 1:N_bins
        R_bins[k] = 0.5 * ( R_binedges[k] + R_binedges[k+1] )
    end

    # Store analytical profile:
    SavePolytropeDensityProfile!( R_bins, ρ_bins_exact, α, ρc )

    # Create file:
    if BC_Z_bot == :Reflect 
        BC_Z_tag = "Zhalf"
    else
        BC_Z_tag = "Zfull"
    end
    outdir   = @__DIR__    # .../src/HydroAndGravityCodeTests/HydroStaticEquilibrium
    FileName = joinpath( outdir, @sprintf( "HydroStaticTest_%03dx%03d_%s_t%.2f.h5",
                         N_r_lin, N_Z_lin, BC_Z_tag, tmax_tdyn ) )
    CreateHDF5File( FileName, N_r, N_Z, N_bins, Nsaved )
    SaveGridHDF5File( FileName, geometry, R_bins, ρ_bins_exact )


    ###############################################################
    #######                                                 #######
    #######               H Y D R O - S E T U P             #######
    #######                                                 #######
    ###############################################################

    if print_info 
        print("\r\033[KInitialising hydro arrays..."); flush(stdout)
    end 

    # We have zero passive scalars, and minimum (1) number of dust species:
    N_Scalars = 0
    N_dust    = 1

    # Set up hydro state + ghost grid arrays:
    hydrostate, ghostgrid = GenerateHydroArrays( r, rCoM, Z, ZCoM, 5 / 3, N_dust, N_Scalars,
                                                 :Reflect,  :Outflow,   # Reflective BCs at r = 0, outflow at Rmax
                                                 BC_Z_bot,  :Outflow    # User dependent BCs at Z = 0, outflow at Zmax
                                                  )

    # Hydro arrays:
    ρ      = hydrostate.ρ
    ρu_r   = hydrostate.ρu_X 
    ρu_Z   = hydrostate.ρu_Y
    u_r    = hydrostate.u_X 
    u_Z    = hydrostate.u_Y
    E      = hydrostate.E
    P      = hydrostate.P 
    e      = hydrostate.e 
    γad    = hydrostate.γad

    # Dust arrays (not coupled to gas, but need to be initialized): 
    ρd     = hydrostate.ρd
    ρdud_r = hydrostate.ρdud_X 
    ρdud_Z = hydrostate.ρdud_Y
    ud_r   = hydrostate.ud_X 
    ud_Z   = hydrostate.ud_Y

    if BC_Z_bot == :Reflect
        # If only upper midplane is simulated, center the cloud at Z = 0:
        Zc = 0.0 
    else
        # Otherwise center at midpoint of Z-range:
        Zc = 0.5 * Zmax 
    end

    # Initialize cloud properties:
    @threads for j in dN_Z 
        @simd for i in dN_r 
            @inbounds begin

                # Distance from cloud center at the edge of this cell:
                Redge = sqrt( r[i+1]^2  +  ( Z[j+1] - Zc )^2 )

                # Adiabatic index:
                γad[i,j] = γ

                # Gas density & chemistry:
                if Redge < Rcl 
                    # Gas density set to cloud density:
                    R      = sqrt( rCoM[i]^2  +  ( ZCoM[j] - Zc )^2 )
                    ρ[i,j] = max( PolytropeDensityProfile( R, α, ρc ), ρext )
                else
                    # Gas density much lower outside the cloud:
                    ρ[i,j] = ρext
                end

                # Gas momentum density:
                ρu_r[i,j]     = 0.0
                ρu_Z[i,j]     = 0.0

                # Initial gas pressure from the polytrope profile:
                P[i,j]        = max( K * (ρ[i,j]^γ), P_floor )

                # Gas internal energy density e = P / (γ - 1):
                e[i,j]        = P[i,j] / ( γad[i,j] - 1 )

                # Initial energy density of gas, E = e:
                E[i,j]        = e[i,j]  

                # Dust density:
                ρd[1,i,j]     = max( 1e-7 * ρ[i,j], 10.0 * ρd_floor )

                # Dust momentum density:
                ρdud_r[1,i,j] = 0.0
                ρdud_Z[1,i,j] = 0.0
                
            end
        end # End of i-loop 
    end # End of j-loop

    # Initialize total mass in simulation volume (for diagnostic purposes):
    UpdateTotalMass!( hydrostate, geometry )

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
    #######          T I M E  I N T E G R A T I O N         #######
    #######                                                 #######
    ###############################################################

    # Initialize the gravitational potential, so that we use an appropriate
    # first time-step:

    # Update total dust density ρdtot:
    UpdateTotalDustDensity!( hydrostate )

    Update_Φ!( gravstate.Φ, gravstate.Φ_half, gravstate.Φ_old, gravstate.Φ_snapshot, 
               gravstate.ρtot, ρ, hydrostate.ρdtot, gravstate.ρ_bg,
               gravcoeffs.RPoiss_ijp, gravcoeffs.RPoiss_ij, gravcoeffs.RPoiss_ijm,
               gravcoeffs.ZPoiss_ijp, gravcoeffs.ZPoiss_ij, gravcoeffs.ZPoiss_ijm, 
               gravcoeffs.Δτ_max, gravcoeffs.Δτ_min,
               ΔV, rCoM, ZCoM, gravstate.Q, gravstate.P, gravstate.ℓmax, BC_Z_bot )

    Update_∇Φ!( gravstate.∂Φ∂r, gravstate.∂Φ∂Z, gravstate.Φ, 
                gravstate.f_gravity_r, gravstate.f_gravity_Z, ρ, 
                r, Z, rCoM, ZCoM, gravstate.P, gravstate.Q, 
                BC_Z_bot, gravstate.ℓmax )


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
        Δt_CFL   = GetCFLTimeStep( hydrostate, geometry ) 
        
        # Gravity time-step:
        Δt_Grav  = GetGravityTimeStep( r, Z, gravstate.∂Φ∂r, gravstate.∂Φ∂Z, ρ, hydrostate.ρdtot ) 

        # Set hydro time-step:
        Δt_Hydro = min( 1.25 * Δt_Hydro, Δt_CFL, Δt_Grav, Δt_Resolve, tmax - t )

        # Update hydro:
        UpdateHydrostaticTest!( Δt_Hydro, hydrostate, geometry,
                                ghostgrid, gravstate, gravcoeffs )

        # Update time & step: 
        t    += Δt_Hydro 
        step += 1

        # ----------- Decide to save data --------------
        if t ≥ t_next_save  -  save_tol # The last factor is to avoid missing a save because round-off errors 
            save_index += 1

            # Update the spherically averaged density profile:
            SphericalAverageProfile!( ρ_bins_num, R_binedges, V_bins, hydrostate, geometry, Zc )

            # Save data: 
            SaveTimeStepHDF5( FileName, hydrostate,
                              save_index, t / t_dyn, 
                              ρ_bins_num )

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