 
module ExponentialScheme
""" MODULE:  ExponentialScheme 

    PURPOSE: Contains the numerically stable exponential scheme for handling 
             diffusion+advection flux terms, in both space and frequency.
"""

__precompile__()

export A_mplus, A_m, A_mminus


const Pmax = 40.0 # Maximum Péclet number to avoid overflow in exponentials


""" The function [1 - exp(-P_m)]/C_m, where P_m is the Péclet number
    in cell m. """
function ϕminus(C_m::Float64, Δy_m::Float64, K_m::Float64)
    P_m = clamp( C_m*Δy_m/(2*K_m), -Pmax, Pmax ) # Péclet number
    if abs( P_m ) == 0.0
        return Δy_m / ( 2 * K_m )
    else
        return ( 1 - exp(- P_m ) ) / C_m
    end
end


""" The function [1 - exp(P_m)]/C_m, where P_m is the Péclet number
    in cell m. """
function ϕplus(C_m::Float64, Δy_m::Float64, K_m::Float64)
    P_m = clamp( C_m*Δy_m/(2*K_m), -Pmax, Pmax ) # Péclet number
    if abs( P_m ) == 0.0
        return - Δy_m / ( 2 * K_m )
    else
        return ( 1 - exp( P_m ) ) / C_m
    end
end


""" The coefficient A_{m+1} """
function A_mplus(f_mplushalf::Float64, C_m::Float64, Δy_m::Float64, K_m::Float64, 
             C_mplus::Float64, Δy_mplus::Float64, K_mplus::Float64)

    P_mplus = clamp( C_mplus*Δy_mplus/(2*K_mplus), -Pmax, Pmax ) # Péclet number

    return f_mplushalf * ( exp( P_mplus ) ) / ( ϕminus(C_m, Δy_m, K_m) - ϕplus(C_mplus, Δy_mplus, K_mplus) )
end


""" The coefficient A_{m}. """
function A_m(f_mplushalf::Float64, f_mminushalf::Float64, C_m::Float64, Δy_m::Float64, K_m::Float64, 
                 C_mplus::Float64, Δy_mplus::Float64, K_mplus::Float64,
                C_mminus::Float64, Δy_mminus::Float64, K_mminus::Float64)

    P_m = clamp( C_m*Δy_m/(2*K_m), -Pmax, Pmax ) # Péclet number

    A_m = ( f_mplushalf * ( exp( - P_m ) ) / ( ϕminus(C_m, Δy_m, K_m) - ϕplus(C_mplus, Δy_mplus, K_mplus) )
        +   f_mminushalf * ( exp( P_m ) ) / ( ϕminus(C_mminus, Δy_mminus, K_mminus) - ϕplus(C_m, Δy_m, K_m)) )
    return A_m
end


""" The coefficient A_{m-1}. """
function A_mminus(f_mminushalf::Float64, C_m::Float64, Δy_m::Float64, K_m::Float64, 
                      C_mminus::Float64, Δy_mminus::Float64, K_mminus::Float64)

    P_mminus = clamp( C_mminus*Δy_mminus/(2*K_mminus), -Pmax, Pmax ) # Péclet number

    return f_mminushalf * ( exp( -P_mminus ) ) / ( ϕminus(C_mminus, Δy_mminus, K_mminus) - ϕplus(C_m, Δy_m, K_m) )
end


end # End of module