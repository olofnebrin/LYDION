 
""" FILE:    ExponentialScheme.jl

    PURPOSE: Contains the numerically stable exponential scheme for handling 
             diffusion + advection frequency flux terms.
"""


export A_mplus, A_m, A_mminus


const Pmax::Float64 = 40.0   # Maximum magnitude of Péclet number to avoid overflow in exponentials and/or floating point errors (default = 40).
const Pmin::Float64 = 1e-3   # Minimum magnitude of Péclet number below which we use a 2:nd order Taylor expansion (default = 10⁻³)

""" The function [1 - exp(-P_m)]/C_m, where P_m is the Péclet number
    in cell m. """
@inline function ϕminus( C_m::Float64, 
                        Δy_m::Float64, 
                         K_m::Float64
                        )

    X   = Δy_m / K_m                      # Convenient variable 
    P_m = clamp( C_m * X,  - Pmax, Pmax ) # Péclet number
    if abs( P_m ) < Pmin
        # In this case, Taylor expand the exponential to second order:
        return X  -  0.5 * C_m * (X^2)
    else 
        # In this case we use the full expression:
        return ( @fastmath 1 - exp( - P_m ) ) / C_m
    end
end


""" The function [1 - exp(P_m)]/C_m, where P_m is the Péclet number
    in cell m. """
@inline function ϕplus( C_m::Float64, 
                       Δy_m::Float64, 
                        K_m::Float64
                        )
    
    X   = Δy_m / K_m                      # Convenient variable 
    P_m = clamp( C_m * X,  - Pmax, Pmax ) # Péclet number
    if abs( P_m ) < Pmin
        # In this case, Taylor expand the exponential to second order:
        return - X  -  0.5 * C_m * (X^2)
    else
        # In this case we use the full expression:
        return ( @fastmath 1 - exp( P_m ) ) / C_m
    end
end


""" The coefficient A_m^+. 

    **ARGUMENTS:**  
    
    - f_mplushalf    :   f(y_{m+1})
    - Δy_m1          :   yCoM_{m+1} - y_{m+1}
    - Δy_m2          :   y_{m+1} - yCoM_{m}
    - C_m            :   C_{m}   (Advection coeff.)
    - K_m            :   K_{m}   (Diffusion coeff.)
    - C_mplus        :   C_{m+1} (Advection coeff.)
    - K_mplus        :   K_{m+1} (Diffusion coeff.) """
function A_mplus( f_mplushalf::Float64,
                        Δy_m1::Float64,
                        Δy_m2::Float64,
                          C_m::Float64, 
                          K_m::Float64, 
                      C_mplus::Float64, 
                      K_mplus::Float64  
                      )

    P_mplus = clamp( C_mplus * Δy_m1 / K_mplus,  - Pmax, Pmax ) # Péclet number

    return f_mplushalf * ( @fastmath exp( P_mplus ) ) / ( ϕminus(C_m, Δy_m2, K_m) - ϕplus(C_mplus, Δy_m1, K_mplus) )
end


""" The coefficient A_m.

    **ARGUMENTS:** 
    
    - f_mplushalf    :   f(y_{m+1})
    - f_mminushalf   :   f(y_{m})
    - Δy_m1          :   yCoM_{m+1} - y_{m+1}
    - Δy_m2          :   y_{m+1} - yCoM_{m}
    - Δy_m3          :   yCoM_{m} - y_{m}
    - Δy_m4          :   y_{m} - yCoM_{m-1}
    - C_m            :   C_{m}   (Advection coeff.)
    - K_m            :   K_{m}   (Diffusion coeff.) 
    - C_mplus        :   C_{m+1} (Advection coeff.)
    - K_mplus        :   K_{m+1} (Diffusion coeff.)
    - C_mminus       :   C_{m-1} (Advection coeff.)
    - K_mminus       :   K_{m-1} (Diffusion coeff.) """
function A_m( f_mplushalf::Float64, 
             f_mminushalf::Float64,
                    Δy_m1::Float64, 
                    Δy_m2::Float64, 
                    Δy_m3::Float64,
                    Δy_m4::Float64,
                      C_m::Float64, 
                      K_m::Float64, 
                  C_mplus::Float64,
                  K_mplus::Float64,
                 C_mminus::Float64,
                 K_mminus::Float64
                 )
    # Péclet numbers:
    P_m_plus  = clamp( C_m * Δy_m2 / K_m,  - Pmax, Pmax )
    P_m_minus = clamp( C_m * Δy_m3 / K_m,  - Pmax, Pmax )

    A_m = ( f_mplushalf * ( @fastmath exp( - P_m_plus ) ) / ( ϕminus(C_m, Δy_m2, K_m) - ϕplus(C_mplus, Δy_m1, K_mplus) )
        +   f_mminushalf * ( @fastmath exp( P_m_minus ) ) / ( ϕminus(C_mminus, Δy_m4, K_mminus) - ϕplus(C_m, Δy_m3, K_m)) )
    return A_m
end


""" The coefficient A_m^-.

    **ARGUMENTS:** 

    - f_mminushalf   :   f(y_{m})
    - Δy_m3          :   yCoM_{m} - y_{m}
    - Δy_m4          :   y_{m} - yCoM_{m-1}
    - C_m            :   C_{m}   (Advection coeff.)
    - K_m            :   K_{m}   (Diffusion coeff.)
    - C_mminus       :   C_{m-1} (Advection coeff.)
    - K_mminus       :   K_{m-1} (Diffusion coeff.) """
function A_mminus( f_mminushalf::Float64,
                          Δy_m3::Float64, 
                          Δy_m4::Float64,
                            C_m::Float64,
                            K_m::Float64, 
                       C_mminus::Float64, 
                       K_mminus::Float64
                       )

    P_mminus = clamp( C_mminus * Δy_m4 / K_mminus,  - Pmax, Pmax ) # Péclet number

    return f_mminushalf * ( @fastmath exp( - P_mminus ) ) / ( ϕminus(C_mminus, Δy_m4, K_mminus) - ϕplus(C_m, Δy_m3, K_m) )
end

