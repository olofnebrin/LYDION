
""" FILE:    M1Closure.jl

    PURPOSE: Contains the M1 closure(s), and related variables of interest (e.g. wave speeds),
             for modelling radiative transfer. 

"""

using .ConstantsAndFunctions  # For constants like OneThird = 1/3
using Base.Threads            # For CPU acceleration

# Export functions:
export Compute_D_Edd, Compute_D_Edd_from_f_n,
       Update_D_Edd!, Update_D_Edd_Serial!, Update_D_Edd_Threaded!

# Export constants:
export fmin, fmax, Jeps, Heps


const DEBUG_M1Closure = false



""" The Levermore (1984) closure factor for the Eddington tensor.  """
@inline @fastmath function χ_L84( f::Float64 )
    f2 = f * f
    χ  = ( 3  +  4 * f2 ) / ( 5  +  2 * sqrt( 4  -  3 * f2 ) )

    return χ
end

 
const fmin::Float64         = 1e-10 
const fmax::Float64         = 1.0 - 1e-10  # To avoid rounding errors from calculation of χ 
const fmin_squared::Float64 = fmin * fmin 
const Jeps::Float64         = 1e-60
const Heps::Float64         = 1e-100
const Heps_squared::Float64 = Heps * Heps 

""" This function computes & returns Eddington tensor (D) components D_rr, D_ZZ, D_rZ 
    for a given cell or cell interface having H components H_r, H_Z, and mean intensity J.
    Assumes the Levermore M1 closure. """
@inline @fastmath function Compute_D_Edd( H_r::Float64,  # H in r-direction  
                                          H_Z::Float64,  # H in Z-direction  
                                            J::Float64   # J                
                                          )
    # Squared magnitude of H:
    Hmag_squared = H_r * H_r  +  H_Z * H_Z

    if J ≤ Jeps || Hmag_squared ≤ Heps_squared || Hmag_squared ≤ fmin_squared * J * J 
        # If the flux is near zero, or if we have ≈ no intensity at all,
        # we assume isotropic intensity:
        D_rr = OneThird
        D_ZZ = OneThird
        D_rZ = 0.0

        return D_rr, D_ZZ, D_rZ
    else
        # Compute f and χ:
        Hmag     = sqrt( Hmag_squared ) 
        f        = clamp( Hmag / J, fmin, fmax )
        χ        = χ_L84( f )
 
        # Convenient factors:
        inv_Hmag = 1 / Hmag
        a        = (1  -  χ) * 0.5
        b        = (3 * χ  -  1) * 0.5
        nr       = H_r * inv_Hmag  # r-component of the flux unit vector n
        nZ       = H_Z * inv_Hmag  # Z-component of the flux unit vector n

        # Get the components of D:
        D_rr     = a  +  b * nr * nr 
        D_ZZ     = a  +  b * nZ * nZ 
        D_rZ     = b * nr * nZ  

        return D_rr, D_ZZ, D_rZ
    end
    
end


""" This function computes & returns Eddington tensor (D) components D_rr, D_ZZ, D_rZ 
    for a given cell or cell interface having reduced flux f and flux direction vetors 
    nr and nZ. Assumes the Levermore M1 closure. 
    
    This form is useful in the M1 reconstruction of D at interfaces. """
@inline @fastmath function Compute_D_Edd_from_f_n( f::Float64, 
                                                  nr::Float64, 
                                                  nZ::Float64 )
       
    # Compute f:
    if f > fmin 
        f = min( f, fmax )
        χ = χ_L84( f )

        # Convenient factors:
        a = (1  -  χ) * 0.5
        b = (3 * χ  -  1) * 0.5

        # Get the components of D:
        D_rr = a  +  b * nr * nr
        D_ZZ = a  +  b * nZ * nZ
        D_rZ = b * nr * nZ

        return D_rr, D_ZZ, D_rZ
    else
        D_rr = OneThird
        D_ZZ = OneThird
        D_rZ = 0.0

        return D_rr, D_ZZ, D_rZ
    end

end



""" Updates the relevant Eddington factors D = K / J, for a single 
    frequency bin. This version does *not* use threading over spatial cells. """
@fastmath function Update_D_Edd_Serial!( D_rr::AbstractArray{TypeOfD, 2},  # rr-component of D (N_r x N_Z elements)
                                         D_ZZ::AbstractArray{TypeOfD, 2},  # ZZ-component of D (N_r x N_Z elements)
                                         D_rZ::AbstractArray{TypeOfD, 2},  # rZ-component of D (N_r x N_Z elements)
                                          H_r::AbstractArray{Float64, 2},  # H in r-direction  (N_r x N_Z elements)
                                          H_Z::AbstractArray{Float64, 2},  # H in Z-direction  (N_r x N_Z elements) 
                                            J::AbstractArray{Float64, 2}   # J                 (N_r x N_Z elements)
                                        )  where {TypeOfD<:AbstractFloat}

    N_r, N_Z    = size( J )  # Grid size
    dN_r        = 1:N_r
    dN_Z        = 1:N_Z

    for j in dN_Z 
        @inbounds @simd for i in dN_r       
            Drr, DZZ, DrZ = Compute_D_Edd( H_r[i,j], H_Z[i,j], J[i,j] )
            D_rr[i,j]     = TypeOfD( Drr )
            D_ZZ[i,j]     = TypeOfD( DZZ )
            D_rZ[i,j]     = TypeOfD( DrZ )
        end # End of i-loop
    end # End of j-loop 

    return nothing 
end


""" Updates the relevant Eddington factors D = K / J, for a single 
    frequency bin. This version does use threading over spatial cells. """
@fastmath function Update_D_Edd_Threaded!( D_rr::AbstractArray{TypeOfD, 2},  # rr-component of D (N_r x N_Z elements)
                                           D_ZZ::AbstractArray{TypeOfD, 2},  # ZZ-component of D (N_r x N_Z elements)
                                           D_rZ::AbstractArray{TypeOfD, 2},  # rZ-component of D (N_r x N_Z elements)
                                            H_r::AbstractArray{Float64, 2},  # H in r-direction  (N_r x N_Z elements)
                                            H_Z::AbstractArray{Float64, 2},  # H in Z-direction  (N_r x N_Z elements) 
                                              J::AbstractArray{Float64, 2}   # J                 (N_r x N_Z elements)
                                          )  where {TypeOfD<:AbstractFloat}

    N_r, N_Z    = size( J )  # Grid size
    dN_r        = 1:N_r
    dN_Z        = 1:N_Z

    @threads for j in dN_Z 
        @inbounds @simd for i in dN_r       
            Drr, DZZ, DrZ = Compute_D_Edd( H_r[i,j], H_Z[i,j], J[i,j] )
            D_rr[i,j]     = TypeOfD( Drr )
            D_ZZ[i,j]     = TypeOfD( DZZ )
            D_rZ[i,j]     = TypeOfD( DrZ )
        end # End of i-loop
    end # End of j-loop 

end


###############################################################
#######                                                 #######
#######                 O L D  C O D E                  #######
#######                                                 #######
###############################################################

# The code below is old, from early testing of LYDION with the 
# HLL flux, for which the eigenvalues were needed. 


""" HLL eigenvalues (in units c̃), for a given f = |H| / J
    and μ = cos θ. See Skinner & Ostriker (2013). """
@inline @fastmath function HLL_Eigenvalues( f::Float64, μ::Float64 )

    # Convenient definitions & clamps for numerical safety:
    f        = clamp( f, fmin, fmax )
    μ        = clamp( μ, - 1.0, 1.0 )
    f2       = f * f
    μ2       = μ * μ
    μf       = μ * f 
    SQRT     = sqrt( max(4 - 3 * f2, 0.0) )
    inv_SQRT = 1 / SQRT
    FAC      = sqrt( max( TwoThirds * (4  -  3 * f2 - SQRT) +  2 * μ2 * (2 - f2 - SQRT), 0.0 ) )

    # Eigenvalues (ignoring λ2 which is not needed):
    λ1   = clamp( ( μf  -  FAC ) * inv_SQRT, - 1.0, 1.0 )

    λ3   = clamp( ( μf  +  FAC ) * inv_SQRT, - 1.0, 1.0 )

    if DEBUG_M1Closure
        @assert max(λ1, λ3) ≤ 1.0 "max(λ1, λ3) = $(max(λ1, λ3))"
        @assert min(λ1, λ3) ≥ - 1.0 "min(λ1, λ3) = $(min(λ1, λ3))"
    end

    return λ1, λ3
end 


""" Computes the eigenvalues (in units c̃) in the r-direction
    for the HLL wave speeds, both left and right.

    See Skinner & Ostriker (2013). """
@inline @fastmath function M1WaveSpeeds_r( J_L::Float64, 
                                           J_R::Float64, 
                                         H_r_L::Float64, 
                                         H_r_R::Float64, 
                                         H_Z_L::Float64,
                                         H_Z_R::Float64
                                          )


    # Compute magnitudes of fluxes on the left & right:
    Hmag_L = sqrt( H_r_L * H_r_L  +  H_Z_L * H_Z_L )
    Hmag_R = sqrt( H_r_R * H_r_R  +  H_Z_R * H_Z_R )

    # Values of μ:
    μ_L = ( Hmag_L > Heps ) ? ( H_r_L / Hmag_L ) : 0.0
    μ_R = ( Hmag_R > Heps ) ? ( H_r_R / Hmag_R ) : 0.0

    # Compute f = |H| / J, but fall back to f = 0 (diffusion limit)
    # if J ≈ 0 or |H| ≈ 0 :
    f_L = ( J_L > Jeps && Hmag_L > Heps ) ? ( Hmag_L / J_L ) : 0.0
    f_R = ( J_R > Jeps && Hmag_R > Heps ) ? ( Hmag_R / J_R ) : 0.0

    # COMPUTE λ1, λ3 FOR LEFT SIDE:
    λ1_L, λ3_L = HLL_Eigenvalues( f_L, μ_L )

    # COMPUTE λ1, λ3 FOR RIGHT SIDE:
    λ1_R, λ3_R = HLL_Eigenvalues( f_R, μ_R )

    return λ1_L, λ1_R, λ3_L, λ3_R
end


""" Computes the eigenvalues (in units c̃) in the Z-direction
    for the HLL wave speeds, both left and right.

    See Skinner & Ostriker (2013). """
@inline @fastmath function M1WaveSpeeds_Z( J_L::Float64, 
                                           J_R::Float64, 
                                         H_r_L::Float64, 
                                         H_r_R::Float64, 
                                         H_Z_L::Float64,
                                         H_Z_R::Float64
                                          )


    # Compute magnitudes of fluxes on the left & right:
    Hmag_L = sqrt( H_r_L * H_r_L  +  H_Z_L * H_Z_L )
    Hmag_R = sqrt( H_r_R * H_r_R  +  H_Z_R * H_Z_R )

    # Values of μ:
    μ_L = ( Hmag_L > Heps ) ? ( H_Z_L / Hmag_L ) : 0.0
    μ_R = ( Hmag_R > Heps ) ? ( H_Z_R / Hmag_R ) : 0.0

    # Compute f = |H| / J, but fall back to f = 0 (diffusion limit)
    # if J ≈ 0 or |H| ≈ 0 :
    f_L = ( J_L > Jeps && Hmag_L > Heps ) ? ( Hmag_L / J_L ) : 0.0
    f_R = ( J_R > Jeps && Hmag_R > Heps ) ? ( Hmag_R / J_R ) : 0.0

    # COMPUTE λ1, λ3 FOR LEFT SIDE:
    λ1_L, λ3_L = HLL_Eigenvalues( f_L, μ_L )

    # COMPUTE λ1, λ3 FOR RIGHT SIDE:
    λ1_R, λ3_R = HLL_Eigenvalues( f_R, μ_R )

    return λ1_L, λ1_R, λ3_L, λ3_R
end

