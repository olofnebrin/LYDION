
module StaticCloudTest


__precompile__()


include( "./Structs.jl" )
include( "./FastInterpolations.jl" )
include( "./ConstantsAndFunctions.jl" )
using .Structs
using .FastInterpolations
using .ConstantsAndFunctions
using LinearAlgebra           # For the tridiagonal solver 
BLAS.set_num_threads(1)       # Ensure the tridiagonal solver is not threaded (we thread over spatial cells)

# Some basic underlying files:
include( "./WENO3.jl" )              # Needed just because it enters the geometry, but not used in practice below 
include( "./FreqFactors.jl" )
include( "./TridiagonalSolver.jl" )
include( "./M1Closure.jl" ) 
include( "./AdvectJandH.jl" )
include( "./SetupLymanAlphaArrays.jl" )
include( "./SpatialGrid.jl" )
include( "./UpdateLyaJandH.jl" )
include( "./SaveHDF5Data_StaticCloudTests.jl" )

using  Base.Threads              # For CPU acceleration
using  ProgressMeter             # To show time step progress
using  Dates, Printf

export SetupStaticCloudTest

@inline function GetHubbleFlow( r::Float64, Z::Float64, umax::Float64, Rcl::Float64, Zc::Float64 )

    # 'Hubble parameter':
    H = umax / Rcl 

    # Cylindrical velocity components, for a Hubble flow centered on Z = Zc:
    u_r      = H * r 
    u_Z      = H * ( Z - Zc )
    
    # Derivatives:
    ∂u_r_∂r  = H 
    ∂u_r_∂Z  = 0.0
    ∂u_Z_∂r  = 0.0
    ∂u_Z_∂Z  = H

    # Also get u_r / r:
    u_r_over_r = H

    return u_r, u_Z, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r
end


function SetupStaticCloudTest( τcl::Float64,              # Cloud optical depth 
                                T₀::Float64,              # Gas temperature (inside and outside)
                              τabs::Float64,              # Cloud continuum absorption optical depth 
                               pd₀::Float64,              # Destruction probability pd in the cloud 
                                rs::Float64 = 1.0,        # Relative size of emission region, Rs/Rcl
                              umax::Float64 = 0.0,        # Maximum gas velocity (km/s), at cloud edge (Hubble flow is assumed, u ∝ R) 
                            Recoil::Bool    = false,      # Whether atomic recoil is turned on (if true) or not (if false)
                             θhole::Float64 = 0.0,        # Opening angle (radians) of a hole in the cloud (0 for uniform cloud)
               IonizedSourceRegion::Bool    = false,      # Whether the emission region is ionized (if true) or not (if false)
                     tmax_unitless::Float64 = 2.0,        # Maximum time for time integration, in units (aτ_cl)^(1/3) Rcl / c̃
                          BC_Z_bot::Symbol  = :Reflect,   # Lyα boundary conditions at lower Z-boundary 
                           N_r_lin::Int     = 64,
                           N_Z_lin::Int     = 64,
                         Nminsteps::Int     = 300,
                         Nmaxsteps::Int     = 100000,
                            Nsaved::Int     = 300,
                        print_info::Bool    = true,
                          FileName::String  = "StaticCloudTest.h5"
                             )

    L_Lyα    = 1e5 * Lsun # Fix Lyα luminosity, since it has no real impact on any static cloud test  
    Rcl      = pc         # Fix the cloud radius to 1 pc, since it has no real impact on any static cloud test  
    Rmax     = 1.1 * Rcl
    Rmax_lin = Rmax 
    Zmax     = Rmax 
    Zmax_lin = Zmax   

    if BC_Z_bot === :Reflect 
        # Place cloud center at bottom (midplane symmetry assumed)
        Zc = 0.0
    else
        # Place cloud at center of box:
        Zc = 0.5 * Zmax  
    end
    
    # RSL can be set to the speed of light:
    c̃          = c

    # Other BCs:
    BC_r_left  = :Reflect
    BC_r_right = :Outflow 
    BC_Z_top   = :Outflow 

    # Setup uniform cylindrical grid:
    geometry   = SetupCylindricalGrid( Rmax, Rmax_lin, Zmax, Zmax_lin, 
                                       N_r_lin, 0, 
                                       N_Z_lin, 0,
                                       BC_Z_bot )
    r          = geometry.r
    rCoM       = geometry.rCoM
    Z          = geometry.Z
    ZCoM       = geometry.ZCoM
    ΔV         = geometry.ΔV
    A_r        = geometry.A_r 
    A_Z        = geometry.A_Z 

    # Grid intervals:
    N_r        = length( rCoM )
    N_Z        = length( ZCoM )
    dN_r       = 1:N_r 
    dN_Z       = 1:N_Z

    # Arrays for the gas velocity & velocity gradient
    u_r        = zeros( Float64, N_r, N_Z )
    u_Z        = zeros( Float64, N_r, N_Z )
    u_r_star   = zeros( Float64, N_r + 1, N_Z )
    u_Z_star   = zeros( Float64, N_r, N_Z + 1 )
    ∂u_r_∂r    = zeros( Float64, N_r, N_Z )
    ∂u_r_∂Z    = zeros( Float64, N_r, N_Z )
    ∂u_Z_∂r    = zeros( Float64, N_r, N_Z )
    ∂u_Z_∂Z    = zeros( Float64, N_r, N_Z )
    u_r_over_r = zeros( Float64, N_r, N_Z )

    # Gas temperature, uniform throughout the simulation volume:
    T          = fill( T₀, N_r, N_Z)

    # Initialize Lyα arrays:
    LyaState   = GenerateLyαArrays( c̃, r, Z, T, Recoil )
    J          = LyaState.J_Lyα
    J_old      = LyaState.J_Lyα_old
    H_r        = LyaState.H_r_Lyα
    H_Z        = LyaState.H_Z_Lyα
    H_r_old    = LyaState.H_r_Lyα_old
    H_Z_old    = LyaState.H_Z_Lyα_old
    e_Lyα      = LyaState.e_Lyα
    α_Lyα      = LyaState.α_Lyα
    α_F        = LyaState.α_F_Lyα
    αc         = LyaState.αc_Lyα
    pd         = LyaState.pd
    ∫dν_js_Lyα = LyaState.∫dν_js_Lyα
    Hvoigt     = LyaState.Hvoigt
    ϕ_ν        = LyaState.ϕ_ν
    Δν         = LyaState.Δν
    Δρu_r      = LyaState.Δρu_Lyα_r
    Δρu_Z      = LyaState.Δρu_Lyα_Z
    dN_ν       = 1:size(J, 3)        # Convenient interval 
    BC_r_left  = :Reflect            # Lyα boundary condition at inner r-boundary (always reflective in cylindrical coordinates)

    # Update the Lyα Voigt profile & emission profile for all cells:
    UpdateVoigtAndEmissionProfile!( LyaState, T )

    # Source region radius:
    Rs   = rs * Rcl 

    # Choose cloud H I density such that its optical depth is τcl:
    nHI₀ = τcl / ( Compute_σ0(T₀) * Rcl )

    # Choose continuum absorption opacity such that the cloud continuum optical depth is τabs:
    αc₀  = τabs / Rcl

    # Initialize cloud properties:

    @inbounds for j in dN_Z
        # Get gas u_r at innermost radial interfaces:
        u_r_star[1,j], _, _, _, _, _, _ = GetHubbleFlow(r[1], ZCoM[j], umax*kms, Rcl, Zc)
    end
    @inbounds for i in dN_r
        # Get gas u_Z at innermost vertical interfaces:
        _, u_Z_star[i,1], _, _, _, _, _ = GetHubbleFlow(rCoM[i], Z[1], umax*kms, Rcl, Zc)
    end

    for j in dN_Z 
        @simd for i in dN_r 
            @inbounds begin

                # Distance from cloud center to this cell:
                RCoM         = sqrt( rCoM[i]^2  +  (ZCoM[j] - Zc)^2 )

                # H I number density, opacities, destruction probability:
                θpolar       = acos( (ZCoM[j] - Zc) / RCoM )    # Polar angle
                θ            = min( θpolar, π - θpolar )        # Allow for two polar holes (in positive & negative Z directions)
                InsideCloud  = (RCoM < Rcl) && (θ > θhole)      # True if inside the H I cloud, false if not 
                InsideSource = (RCoM < Rs)

                xHII         = 0.0 # Initialize H II fraction 
                if InsideSource
                    # Set ionization fraction inside the emission region:
                    if IonizedSourceRegion
                        xHII = 1.0
                    else
                        xHII = 0.0
                    end
                end

                # Set cloud properties, if we're inside the cloud: 
                if InsideCloud
                    nHI        = nHI₀ * (1 - xHII)
                    α_Lyα[i,j] = nHI * Compute_σ0(T[i,j])
                    αc[i,j]    = αc₀
                    pd[i,j]    = pd₀
                else
                    # No H I or absorption/destruction outside the cloud: 
                    nHI        = 0.0
                    α_Lyα[i,j] = 0.0
                    αc[i,j]    = 0.0
                    pd[i,j]    = 0.0
                end

                # Lyα frequency-integrated emissivity:
                if InsideCloud && InsideSource
                    ∫dν_js_Lyα[i,j] = 3 * L_Lyα / (Four_π^2 * Rs^3)
                else
                    ∫dν_js_Lyα[i,j] = 0.0
                end

                # Get flux opacity:
                for k in dN_ν
                    α_F[i,j,k] = α_Lyα[i,j] * Hvoigt[i,j,k]
                end

                # Get gas velocity field:
                u_r[i,j], u_Z[i,j], ∂u_r_∂r[i,j], ∂u_r_∂Z[i,j], 
                ∂u_Z_∂r[i,j], ∂u_Z_∂Z[i,j], u_r_over_r[i,j] = GetHubbleFlow( rCoM[i], ZCoM[j], umax * kms, Rcl, Zc )

                # Get u at the cell interfaces:
                u_r_star[i+1,j], _, _, _, _, _, _ = GetHubbleFlow( r[i+1], ZCoM[j], umax * kms, Rcl, Zc )

                _, u_Z_star[i,j+1], _, _, _, _, _ = GetHubbleFlow( rCoM[i], Z[j+1], umax * kms, Rcl, Zc )

            end
        end # End of i-loop 
    end # End of j-loop

    # Update frequency leakage factors:
    UpdateFreqFactors!( LyaState, T, ∂u_r_∂r, ∂u_r_∂Z,  ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r )

    ###############################################################
    #######                                                 #######
    #######          T I M E  I N T E G R A T I O N         #######
    #######                                                 #######
    ###############################################################

    CFL         = 0.4
    Δt_CFL      = GetRHDCFLTimeStep(r, Z, c̃, u_r, u_Z, CFL)
    t           = 0.0
    aτcl        = Compute_a(T₀) * τcl
    tdiff       = (aτcl)^(1/3) * Rcl / c̃
    tmax        = tmax_unitless * tdiff

    # Initialize time-steps where data is saved 
    Δt_save     = tmax / ( Nsaved - 1 )
    t_next_save = 0.0  
    save_index  = 0
    save_tol    = 1e-12 * tmax # Tolerance to guard against round-off errors when deciding if to save data

    # Create arrays for storing the spherically averaged e_Lyα profile:
    Nbins_sph   = min( N_r_lin, N_Z_lin )
    V_bins_sph  = zeros( Float64, Nbins_sph )
    R_binedges  = LinRange( 0.0, Rmax, Nbins_sph + 1 )
    e_Lyα_sph   = zeros( Float64, Nbins_sph )

    # Create HDF5 file:
    CreateHDF5File( FileName, aτcl, N_r, N_Z, Nsaved, size(J,3), Nbins_sph )
    SaveGridHDF5File( FileName, geometry, Rcl )
    SaveLyαFreqGridHDF5File( FileName, LyaState )
    SaveAnalyticalSolutionHDF5File( FileName, ZCoM, τcl, T₀, Rcl, L_Lyα )
    SaveSphericalBinsHDF5File( FileName, R_binedges, Rcl )

    if print_info
        print("\r\033[KStarting time evolution..."); flush(stdout)
        println() # New line
    end
    
    # Initialize the progress meter:
    prog = Progress( 100; dt = 0.01, desc = "Time integration progress:",
                     barglyphs=BarGlyphs("[=> ]"), barlen = 50, color=:green )              

    step = 0
    while t < tmax
        Δt     = min(Δt_CFL, tmax - t)
        c̃Δt    = c̃ * Δt

        # Step 1: M1 advection of Lyα
        AdvectLyα!( c̃Δt, LyaState, geometry, 
                    u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                    BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top )

        # Step 2: Lyα emission/absorption
        LyαEmissionAbsorptionUpdate!( J, J_old, H_r, H_Z,
                                      H_r_old, H_Z_old,
                                      ∫dν_js_Lyα, ϕ_ν,
                                      Hvoigt, α_Lyα, αc, α_F, pd,
                                      Δρu_r, Δρu_Z, Δν,
                                      c̃, Δt )

        # Step 3) Frequency diffusion/advection of Lyα
        FrequencyUpdateJLyα_AllCells!( c̃Δt, LyaState, T, 
                                       ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r, 
                                       true )

        # Update Lyα forces f_Lyα, force multipliers:
        LyaState.M_F, LyaState.M_F_radial_net, 
        LyaState.M_F_radial_out,  
        LyaState.L_Lyα = Update_f_Lyα_MF_L_Lyα!( LyaState.f_Lyα_r, LyaState.f_Lyα_Z, Δρu_r, Δρu_Z, 
                                                 ∫dν_js_Lyα, ΔV, rCoM, ZCoM, Δt,
                                                 Zc, # Cloud center 
                                                 BC_Z_bot )

        # Update Lyα escape fraction:
        LyaState.f_esc_Lyα = Compute_f_esc_Lyα_Cylindrical( H_r, H_Z, A_r, A_Z, LyaState.Δν, 
                                                            LyaState.L_Lyα, BC_Z_bot )

        # Reset Δρu to zero for next time-step (to ensure correct force calculation):
        Reset_Δρu!( Δρu_r, Δρu_Z )

        # Update time & step: 
        t    += Δt
        step += 1

        if t ≥ t_next_save  -  save_tol # The last factor is to avoid missing a save because round-off errors 
            save_index += 1

            # Update Lyα energy density for saving:
            Update_e_Lyα!( e_Lyα, Δν, J )

            # Compute spherical average profile from e_Lyα(r,Z):
            SphericalAverageProfile!( e_Lyα_sph, e_Lyα, R_binedges, V_bins_sph, ΔV, rCoM, ZCoM, Zc )

            # Save data:
            SaveTimeStepHDF5( FileName, LyaState, save_index, t / tdiff, e_Lyα_sph )

            # Move the 'next save' appointment forward until it is in the future 
            while t ≥ t_next_save - save_tol
                t_next_save += Δt_save
            end
        end

        # Compute progress fraction
        progress_frac = t / tmax
        # Update the progress bar description to show simulation time and percent
        prog.desc     = @sprintf("t = %.2e yr (%.1f%%)", t / yr, 100 * progress_frac)
        # Update the bar progress value
        update!( prog, Int(round(100 * progress_frac)) )

        if step > Nmaxsteps 
            println("\nSimulation stopped: Reached maximum number of global time-steps ($(Nmaxsteps)).")
            break
        end

    end # End of time integration 


    return nothing 
end


end # End of module 