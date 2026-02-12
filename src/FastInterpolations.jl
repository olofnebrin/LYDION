
module FastInterpolations
""" MODULE:  FastInterpolations

    PURPOSE: Contains code for creating fast/optimized linear look-up tables (LUTs) for
             special or expensive functions like Dawson(x), exp(-x²), that enter the 
             calculation of the Voigt profile. This can greatly speed up the calculations 
             when these functions are called for many cells, bins, over many time-steps.

"""

__precompile__()

using SpecialFunctions  # For Dawson integral, erf, etc.

export CreateVoigtLUTs, DawsonFast, ErfFast, GaussianFast, 
       xGaussianFast, OneMinusx2GaussianFast


""" A struct for a general linear look-up table. """
struct LinearLUT
    xmin::Float64               # Look-up table starts at x = xmin 
    inv_Δx::Float64             # 1 / Δx, where Δx is the uniform spacing 
    y::Vector{Float64}          # Stores y[i] 
    Nx::Int                     # Length of y[i]
end

""" A struct containing the LUTs for all the x-functions 
    that enter the calculations of the Voigt profile. """
struct VoigtLUTs
    xmax::Float64
    erf_LUT::LinearLUT                 # For erf(x)
    Dawson_LUT::LinearLUT              # For Dawson(x)
    Gaussian_LUT::LinearLUT            # For exp(-x²)
    xGaussian_LUT::LinearLUT           # For x × exp(-x²)
    OneMinusx2Gaussian_LUT::LinearLUT  # For (1 - 2x²) × exp(-x²)
end


""" This function creates a LUT for a function f(x), valid 
    over x ∈ [xmin, xmax], using Nx grid points. """
function CreateLUT( f::Function, 
                 xmin::Float64,
                 xmax::Float64,
                   Nx::Int      # Number of "interface" points
                   )

    @assert Nx ≥ 2
    @assert xmax > xmin 

    # Spacing and its inverse (number of "cells" is Nx - 1):
    Δx     = ( xmax - xmin ) / ( Nx - 1 )
    inv_Δx = 1.0 / Δx

    # Initialize y[i]:
    y      = Vector{Float64}(undef, Nx)

    # Fill y[i]:
    @inbounds for i in 1:Nx 
        x    = xmin  +  (i - 1) * Δx
        y[i] = f(x)
    end

    return LinearLUT( xmin, inv_Δx, y, Nx )
end


""" This function evaluates a LUT at x.

    NOTE: Assumes that x is inside the LUT range. """
@inline function EvaluateLUT( LUT::LinearLUT, x::Float64 )
    @inbounds begin
        # Load look-up table & its length:
        y_LUT   = LUT.y
        Nx      = LUT.Nx

        # Get x̃ = (x - xmin) / Δx:
        x̃       = ( x - LUT.xmin ) * LUT.inv_Δx 

        # Convert to closest lower index i (0-based):
        i       = Int( floor(x̃) ) 

        # Ensure that i_Julia + 1 = i + 2 < Nx:
        if i + 2 > Nx
            i = Nx - 2
        end

        # Distance from x̃ to i + 1 is:
        δx̃      = (i + 1) - x̃

        # Convert to 1-based system of Julia:
        i_Julia = i + 1   

        # Linear weighting (NOTE: δx̃ is ∈ [0, 1]) to 
        # get y(x):
        @fastmath y =  δx̃ * y_LUT[i_Julia]  +  (1.0 - δx̃) * y_LUT[i_Julia+1]

        return y
    end # End of inbounds 
end


const xmax_Voigt::Float64 = 1000.0
const Nx_Voigt::Int       = 200001  # Gives Δx = 2 xmax / (Nx - 1) = 0.01

function CreateVoigtLUTs( xmax::Float64 = xmax_Voigt, Nx::Int = Nx_Voigt )

    @assert xmax > 0.0
    @assert Nx   ≥ 2

    # We consider a range x ∈ [ - xmax, xmax ] 
    xmin = - xmax

    # Create the LUTs for erf & Dawson:
    erf_LUT     = CreateLUT( erf,    xmin, xmax, Nx )
    Dawson_LUT  = CreateLUT( dawson, xmin, xmax, Nx )

    # Create the LUTs for the rest of the functions:
    Gaussian_LUT           = CreateLUT( x -> exp( - x * x),                           xmin, xmax,       Nx )
    xGaussian_LUT          = CreateLUT( x -> x * exp( - x * x),                       xmin, xmax,       Nx )
    OneMinusx2Gaussian_LUT = CreateLUT( x -> ( 1.0 - 2.0 * (x * x) ) * exp( - x * x), xmin, xmax,       Nx )

    return VoigtLUTs( xmax, erf_LUT, Dawson_LUT, Gaussian_LUT, xGaussian_LUT, OneMinusx2Gaussian_LUT  )
end


""" This function ives the asymptotic expression for the Dawson function,

    Dawson(x) ∼ 1/(2x) + 1/(4x³) + O(x⁻⁵) """
@inline @fastmath function Dawson_Asymptotic( x::Float64 )
    inv_x  = 1.0 / x 
    inv_x3 = inv_x * inv_x * inv_x

    return 0.5 * inv_x  +  0.25 * inv_x3
end


""" This function evaluates the Dawson function in a fast manner,
    using linear interpolation of the look-up table for x of small 
    magnitude, and the asymptotic expression for large x. """
@inline function DawsonFast( x::Float64, V::VoigtLUTs )

    if abs(x) ≥ V.xmax 
        return Dawson_Asymptotic( x )
    else
        return EvaluateLUT( V.Dawson_LUT, x )
    end
end


""" This function evaluates the erf function in a fast manner,
    using linear interpolation of the look-up table for x of small 
    magnitude, and the asymptotic expression for large x. """
@inline function ErfFast( x::Float64, V::VoigtLUTs )
    if abs(x) ≥ V.xmax
        # For large magnitudes of x, erf asymptotes to +1 (for x > 0)
        # and -1 (for x < 0):
        return copysign( 1.0, x ) # Returns magnitude 1.0 with sign of x, and in Float64
    else
        return EvaluateLUT( V.erf_LUT, x )
    end
end


""" This function evaluates the Gaussian function exp(-x²) in a fast manner,
    using linear interpolation of the look-up table for x of small 
    magnitude, and the asymptotic expression for large x. """
@inline function GaussianFast( x::Float64, V::VoigtLUTs )

    if abs(x) ≥ V.xmax
        # Gaussian vanishes at large x quickly: 
        return 0.0
    else
        return EvaluateLUT( V.Gaussian_LUT, x )
    end
end


""" This function evaluates x exp(-x²) in a fast manner,
    using linear interpolation of the look-up table for x of small 
    magnitude, and the asymptotic expression for large x. """
@inline function xGaussianFast( x::Float64, V::VoigtLUTs )
    if abs(x) ≥ V.xmax
        # Gaussian vanishes at large x quickly: 
        return 0.0
    else
        return EvaluateLUT( V.xGaussian_LUT, x )
    end
end


""" This function evaluates (1 - 2x²) exp(-x²) in a fast manner,
    using linear interpolation of the look-up table for x of small 
    magnitude, and the asymptotic expression for large x. """
@inline function OneMinusx2GaussianFast( x::Float64, V::VoigtLUTs )
    if abs(x) ≥ V.xmax
        # Gaussian vanishes at large x quickly: 
        return 0.0
    else
        return EvaluateLUT( V.OneMinusx2Gaussian_LUT, x )
    end
end


end # End of module 
