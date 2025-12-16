
module StaticCloudTest


__precompile__()


include( "./Structs.jl" )
include( "./ConstantsAndFunctions.jl" )
using .Structs
using .ConstantsAndFunctions
using LinearAlgebra           # For the tridiagonal solver 
BLAS.set_num_threads(1)       # Ensure the tridiagonal solver is not threaded (we thread over spatial cells)

# Some basic underlying files:
include( "./ExponentialScheme.jl" )
include( "./TridiagonalSolver.jl" )
include( "./M1Closure.jl" ) 
include( "./AdvectJandH.jl" )
include( "./SetupLymanAlphaArrays.jl" )
include( "./SpatialGrid.jl" )
include( "./UpdateLyaJandH.jl" )
include( "./SaveHDF5Data_RHD.jl" )

using  Base.Threads              # For CPU acceleration
using  ProgressMeter             # To show time step progress
using  Dates, Printf

export SetupStaticCloudTest



function SetupStaticCloudTest( τcl::Float64,              # Cloud optical depth 
                                T₀::Float64,              # Gas temperature (inside and outside) 
                                pd::Float64,              # Destruction probability pd in the cloud 
                                rs::Float64 = 0.1,        # Relative size of emission region, Rs/Rcl
                              umax::Float64 = 0.0,        # Maximum gas velocity (km/s), at cloud edge (Hubble flow is assumed, u ∝ R) 
                             θhole::Float64 = 0.0,        # Opening angle (radians) of a hole in the cloud (0 for uniform cloud)
                             L_Lyα::Float64 = 1e4 * Lsun, # Total Lyα luminosity within spherical radius Rs 
               IonizedSourceRegion::Bool    = false,      # Whether the emission region is ionized (if true) or not (if false)
                            Recoil::Bool    = true,       # Whether atomic recoil is turned on (if true) or not (if false)
                           N_r_lin::Int     = 64,
                           N_Z_lin::Int     = 64,
                         Nminsteps::Int     = 300,
                         Nmaxsteps::Int     = 100000,
                            Nsaved::Int     = 300,
                        print_info::Bool    = true,
                          FileName::String  = "StaticCloudTest.h5"
                             )

    Rmax     = 1.4 * Rcl
    Rmax_lin = Rmax 
    Zmax     = Rmax 
    Zmax_lin = Zmax                     

    # Setup uniform cylindrical grid:
    geometry   = SetupCylindricalGrid( Rmax, Rmax_lin, Zmax, Zmax_lin, 
                                       N_r_lin, 0, 
                                       N_Z_lin, 0 )

    r          = geometry.r
    rCoM       = geometry.rCoM
    Z          = geometry.Z
    ZCoM       = geometry.ZCoM
    ΔV         = geometry.ΔV

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

    if IonizedSourceRegion
        xHII_SourceRegion = 1.0
    else
        xHII_SourceRegion = 0.0
    end

    # Choose cloud H I density such that its optical depth is τcl:
    nHI₀ = τcl / ( σ0(T₀) * Rcl )


    # Initialize cloud properties:
    @threads for j in dN_Z 
        @simd for i in dN_r 
            @inbounds begin

                # Distance from cloud center at the edge of this cell:
                R = sqrt( r[i+1]^2  +   (Z[j+1] - Zc)^2 )

                # H I number density:
                if R < Rcl && R > Rs && θ > θhole
                    nHI = nHI₀
                elseif R < Rs
                    nHI = max( (1 - xHII_SourceRegion), 0.0 ) * nHI₀ 
                else 
                    nHI = 0.0
                end

                # Gas temperature, destruction probability:
                T[i,j]  = T₀
                pd[i,j] = pd₀

                # Gas velocity:
                
            end
        end # End of i-loop 
    end # End of j-loop

    return nothing 
end


end # End of module 