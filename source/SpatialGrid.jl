
module SpatialGrid
""" MODULE:  SpatialGrid

    PURPOSE: Contains functions for setting up a spatial grid, in 
             both cylindrical and Cartesian coordinates.

"""

__precompile__()

export SetupUniformCylindricalGrid


""" This function populates rCoM with the centers of mass along 
    the radial direction, *given* interface values r.
    
    Works for cylindrical coordinates. """
function Get_rCoM_Cylindrical!( r::AbstractArray{Float64}, 
                             rCoM::AbstractArray{Float64} )

    N_r  = length( r ) - 1  # Number of radial grid points 
    dN_r = 1:N_r

    @assert N_r == length( rCoM )

    @inbounds @simd for i in dN_r
        rCoM[i] = (2/3) * ( r[i+1]^3  -  r[i]^3 ) / ( r[i+1]^2  -  r[i]^2 )
    end
end


""" This function populates ZCoM with the centers of mass along 
    the vertical direction, *given* interface values Z.
    
    Works for both cylindrical & Cartesian 
    coordinates. """
function Get_ZCoM!( Z::AbstractArray{Float64}, 
                 ZCoM::AbstractArray{Float64} )

    N_Z  = length( Z ) - 1  # Number of vertical grid points 
    dN_Z = 1:N_Z 

    @assert N_Z == length( ZCoM )

    @inbounds @simd for j in dN_Z
        ZCoM[j] = 0.5 * ( Z[j]  +  Z[j+1] )
    end
end


""" This function populates ΔV with the cell volumes, 
    *given* interface values r and Z.
    
    Works for cylindrical coordinates. """
function Get_ΔV_Cylindrical!( r::AbstractArray{Float64}, 
                              Z::AbstractArray{Float64},
                             ΔV::AbstractArray{Float64} )

    N_r  = length( r ) - 1  # Number of radial grid points         
    N_Z  = length( Z ) - 1  # Number of vertical grid points 
    dN_r = 1:N_r
    dN_Z = 1:N_Z 

    @assert size( ΔV ) == (N_r, N_Z)

    @inbounds for j in dN_Z 
        ΔZ = Z[j+1] - Z[j]
        @simd for i in dN_r
            ΔV[i,j] = π * ( r[i+1]^2 - r[i]^2 ) * ΔZ
        end
    end
end


""" This function populates A_r with interface cell areas, 
    along the r-direction, given interface values r and Z.
    
    Works for both cylindrical coordinates. """
function Get_A_r_Cylindrical!( r::AbstractArray{Float64}, 
                               Z::AbstractArray{Float64},
                             A_r::AbstractArray{Float64} )

    N_r  = length( r ) - 1  # Number of radial grid points         
    N_Z  = length( Z ) - 1  # Number of vertical grid points 
    dN_Z = 1:N_Z 

    @assert size( A_r ) == (N_r + 1, N_Z)

    @inbounds for j in dN_Z 
        ΔZ = Z[j+1] - Z[j]
        @simd for i in 1:(N_r+1)
            A_r[i,j] = 2 * π * r[i] * ΔZ
        end
    end
end


""" This function populates A_Z with interface cell areas, 
    along the Z-direction, given interface values r and Z.
    
    Works for cylindrical coordinates. """
function Get_A_Z_Cylindrical!( r::AbstractArray{Float64}, 
                               Z::AbstractArray{Float64},
                             A_Z::AbstractArray{Float64} )

    N_r  = length( r ) - 1  # Number of radial grid points         
    N_Z  = length( Z ) - 1  # Number of vertical grid points 
    dN_r = 1:N_r 

    @assert size( A_Z ) == (N_r, N_Z + 1)

    @inbounds for j in 1:(N_Z+1) 
        @simd for i in dN_r
            A_Z[i,j] = π * ( r[i+1]^2  -  r[i]^2 ) 
        end
    end
end


function SetupUniformCylindricalGrid( rmax::Float64,
                                      Zmax::Float64,
                                       N_r::Int64,
                                       N_Z::Int64
                                     )
    # Initialize arrays:

    ΔV   = zeros( N_r, N_Z )      # Cell volumes 
    A_r  = zeros( N_r + 1, N_Z )  # r-direction cell interface areas 
    A_Z  = zeros( N_r, N_Z + 1 )  # Z-direction cell interface areas
    r    = zeros( N_r + 1 )       # r-interfaces
    rCoM = zeros( N_r )           # r-direction cell centers of mass 
    Z    = zeros( N_Z + 1 )       # Z-interfaces
    ZCoM = zeros( N_Z )           # Z-direction cell centers of mass 

    # Fill interface values:
    r   .= LinRange( 0.0, rmax, N_r + 1 )
    Z   .= LinRange( 0.0, Zmax, N_Z + 1 )

    # Get rCoM, ZCoM, ΔV, A_r, and A_Z:
    Get_rCoM_Cylindrical!( r, rCoM )
    Get_ZCoM!( Z, ZCoM ) 
    Get_ΔV_Cylindrical!( r, Z, ΔV )
    Get_A_r_Cylindrical!( r, Z, A_r )
    Get_A_Z_Cylindrical!( r, Z, A_Z )


    return ΔV, A_r, A_Z, r, rCoM, Z, ZCoM
end


end # End of module 
