
""" FILE:    M1Closure.jl

    PURPOSE: Contains the M1 closure(s), and related variables of interest (e.g. wave speeds),
             for modelling radiative transfer. 

"""

using .ConstantsAndFunctions  # For constants like OneThird = 1/3
using .ConstantsAndFunctions.FastInterpolations: CreateM1ThirdMomentLUTs, ϕ1ϕ2Fast # For the calculation of L:∇u
using Base.Threads            # For CPU acceleration

# Export functions:
export Compute_D_Edd, Compute_D_Edd_from_f_n,
       Update_D_Edd!, Update_D_Edd_Serial!, Update_D_Edd_Threaded!,
       Compute_Lijk


# Export constants:
export fmin, fmax, Jeps, Heps


const DEBUG_M1Closure = false



""" The Levermore (1984) closure factor for the Eddington tensor.  """
@inline @fastmath function χ_L84( f::Float64 )
    f2 = f * f
    χ  = ( 3  +  4 * f2 ) / ( 5  +  2 * sqrt( 4  -  3 * f2 ) )

    #if f < OneThird
    #  χ = OneThird
    #else  
    #  χ = 0.5  -  f  +  1.5 * f*f
    #end
      
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
        b_nr     = b * nr            # Used twice below 
        D_rr     = a  +  b_nr * nr 
        D_ZZ     = a  +  b * nZ * nZ 
        D_rZ     = b_nr * nZ  

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
        b_nr = b * nr            # Used twice below 
        D_rr = a  +  b_nr * nr
        D_ZZ = a  +  b * nZ * nZ
        D_rZ = b_nr * nZ

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


""" Updates the relevant Eddington factors D = K / J, for a all 
    frequency bin. This version does use threading over spatial cells. """
@fastmath function Update_D_Edd_Threaded!( D_rr::AbstractArray{TypeOfD, 3},  # rr-component of D (N_ν x N_r x N_Z elements)
                                           D_ZZ::AbstractArray{TypeOfD, 3},  # ZZ-component of D (N_ν x N_r x N_Z elements)
                                           D_rZ::AbstractArray{TypeOfD, 3},  # rZ-component of D (N_ν x N_r x N_Z elements)
                                            H_r::AbstractArray{Float64, 3},  # H in r-direction  (N_ν x N_r x N_Z elements)
                                            H_Z::AbstractArray{Float64, 3},  # H in Z-direction  (N_ν x N_r x N_Z elements) 
                                              J::AbstractArray{Float64, 3}   # J                 (N_ν x N_r x N_Z elements)
                                          )  where {TypeOfD<:AbstractFloat}

    N_ν, N_r, N_Z  = size( J )  # Grid size
    dN_ν           = 1:N_ν
    dN_r           = 1:N_r
    dN_Z           = 1:N_Z

    @threads for j in dN_Z 
        @inbounds for i in dN_r  
            @simd for k in dN_ν     
                Drr, DZZ, DrZ   = Compute_D_Edd( H_r[k,i,j], H_Z[k,i,j], J[k,i,j] )
                D_rr[k,i,j]     = TypeOfD( Drr )
                D_ZZ[k,i,j]     = TypeOfD( DZZ )
                D_rZ[k,i,j]     = TypeOfD( DrZ )
            end # End of k-loop 
        end # End of i-loop
    end # End of j-loop 

end


# Create look-up tables (LUTs) for expensive functions that 
# are needed for the calculation of the third moment, L:
const L_LUTs = CreateM1ThirdMomentLUTs()

""" This function computes the needed components of the third moment L_ijk (i,j,k are 
    spatial indices here). The components of L are evaluated following Vaytet+ (2011): 
    
    Lᵢⱼₖ = ϕ₁(Hᵢδⱼₖ + Hⱼδᵢₖ + Hₖδᵢⱼ)  +  ϕ₂ HᵢHⱼHₖ/J² 

    where ϕ₁ and ϕ₂ are functions of the reduced flux, derived by Vaytet+ for the M1 closure.

    ----------- OLD:
    This function computes the needed components of L : ∇u = L_ijk (∇_j u_k).
    For the i = r and Z components, this yields (assuming u_ϕ = 0):
    
    L_rjk (∇_j u_k) = L_rrr(∂u_r/∂r) + L_rrZ(∂u_Z/∂r) + L_rZr(∂u_r/∂Z) + L_rZZ(∂u_Z/∂Z) + L_rϕϕ(u_r/r) 

    L_Zjk (∇_j u_k) = L_Zrr(∂u_r/∂r) + L_ZrZ(∂u_Z/∂r) + L_ZZr(∂u_r/∂Z) + L_ZZZ(∂u_Z/∂Z) + L_Zϕϕ(u_r/r) 
    
    The components of L are evaluated following Vaytet+ (2011): 
    
    Lᵢⱼₖ = ϕ₁(Hᵢδⱼₖ + Hⱼδᵢₖ + Hₖδᵢⱼ)  +  ϕ₂ HᵢHⱼHₖ/J² 
    
    Function returns:
    
    (L : ∇u)_r = L_rjk (∇_j u_k) 
    (L : ∇u)_Z = L_Zjk (∇_j u_k) """
@inline @fastmath function Compute_Lijk( J::Float64,
                                       H_r::Float64,
                                       H_Z::Float64
                                        )

    # Squared magnitude of H:
    Hmag_squared = H_r * H_r  +  H_Z * H_Z 

    # If there is essentially no radiation field, set the contraction to zero:
    if J ≤ Jeps || Hmag_squared ≤ Heps_squared
        return 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0
    end
    
    # Reduced flux:
    Hmag = sqrt( Hmag_squared )
    fraw = Hmag / J 
    f    = clamp( fraw, 0.0, fmax )

    # If fraw is slightly too large, rescale H consistently with the closure:
    if f < fraw
        Hscale = f / fraw
        H_r    = Hscale * H_r
        H_Z    = Hscale * H_Z
    end

    # Get the functions ϕ1 and ϕ2 of Vaytet+ (2011), that are 
    # needed to evaluate L:
    ϕ1, ϕ2 = ϕ1ϕ2Fast( f, L_LUTs )

    # Define some common factors below:
    inv_Jsquared           = 1.0 / ( J * J )
    ϕ2_Hr_Hr_inv_Jsquared  = ϕ2 * H_r * H_r * inv_Jsquared
    ϕ2_HZ_HZ_inv_Jsquared  = ϕ2 * H_Z * H_Z * inv_Jsquared

    ϕ1_Hr                  = ϕ1 * H_r 
    ϕ1_HZ                  = ϕ1 * H_Z

    # Needed L components:
    L_rrr  = 3 * ϕ1_Hr  +  ϕ2_Hr_Hr_inv_Jsquared * H_r 

    L_rrZ  =     ϕ1_HZ  +  ϕ2_Hr_Hr_inv_Jsquared * H_Z 
    L_rZr  = L_rrZ  # By symmetry 
    L_Zrr  = L_rrZ  # By symmetry 

    L_rZZ  =     ϕ1_Hr  +  ϕ2_HZ_HZ_inv_Jsquared * H_r 
    L_ZrZ  = L_rZZ  # By symmetry 
    L_ZZr  = L_rZZ  # By symmetry 

    L_ZZZ  = 3 * ϕ1_HZ  +  ϕ2_HZ_HZ_inv_Jsquared * H_Z 

    L_rϕϕ  = ϕ1_Hr
    L_Zϕϕ  = ϕ1_HZ

    return L_rrr, L_rrZ, L_rZr, L_rZZ, L_rϕϕ,
           L_Zrr, L_ZrZ, L_ZZr, L_ZZZ, L_Zϕϕ
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
    for the HLL wave speeds, using already reconstructed f and n. """
@inline @fastmath function M1WaveSpeeds_r( f_L::Float64,
                                           f_R::Float64,
                                          nr_L::Float64,
                                          nr_R::Float64 )

    # In the r-direction, μ = n_r:
    λ1_L, λ3_L = HLL_Eigenvalues( f_L, nr_L )  # Left side 
    λ1_R, λ3_R = HLL_Eigenvalues( f_R, nr_R )  # Right side 

    return λ1_L, λ1_R, λ3_L, λ3_R
end


""" Computes the eigenvalues (in units c̃) in the Z-direction
    for the HLL wave speeds, using already reconstructed f and n. """
@inline @fastmath function M1WaveSpeeds_Z( f_L::Float64,
                                           f_R::Float64,
                                          nZ_L::Float64,
                                          nZ_R::Float64 )

    # In the Z-direction, μ = n_Z:
    λ1_L, λ3_L = HLL_Eigenvalues( f_L, nZ_L )  # Left side 
    λ1_R, λ3_R = HLL_Eigenvalues( f_R, nZ_R )  # Right side 

    return λ1_L, λ1_R, λ3_L, λ3_R
end
