
module ConstantsAndFunctions
""" MODULE:  ConstantsAndFunctions 

    PURPOSE: Contains useful constants and units (e.g. c, νLyα, Lsun), 
             and functions (e.g. Voigt profile, optical depth, analytical
             spectra to compare to). 

"""

__precompile__()

using SpecialFunctions  # For Dawson integral

# Export functions:
export H, Htophat, a, σ0, b, N_HI, τ, deriv, divergence, LyαEscapeFraction, 
       LyαPressureAndForce, J_sphere_point, J_sphere_uni, PLyα_sphere_point,
       PLyα_sphere_uni

###############################################################
#######                                                 #######
#######               C O N S T A N T S                 #######
#######                                                 #######
###############################################################

const c        = 2.99792458e10  # Speed of light (cm/s)
const h        = 6.6260755e-27  # Planck's constant (erg s)
const kB       = 1.380658e-16   # Boltzmann constant (erg/K)
const mH       = 1.67353e-24    # Mass of hydrogen atom (g)
const me       = 9.1093837e-28  # Mass of electron (g)
const e        = 4.8032068e-10  # Electron charge (esu)
const νLyα     = 2.466068e15    # Lyα frequency at line center (Hz)
const ELyα     = h * νLyα       # Lyα photon energy (erg)
const ALyα     = 6.265e8        # Einstein-A coefficient for Lyα (s⁻¹)
const f_lu_Lyα = 0.41641        # Oscillator strength for Lyα

const Lsun     = 3.828e33       # Solar luminsoity (erg/s)
const Msun     = 1.989e33       # Solar mass (g)
const pc       = 3.086e18       # 1 parsec (cm)
const kms      = 1e5            # 1 km/s (cm/s)
const yr       = 3.1556926e7    # 1 year (sec)
const eV       = 1.602e-12      # 1 eV (erg)

# Export constants:
export c, h, kB, mH, me, e, νLyα, ELyα, ALyα, f_lu_Lyα, 
       Lsun, Msun, pc, kms, yr, eV 

###############################################################
#######                                                 #######
#######         U S E F U L  F U N C T I O N S          #######
#######                                                 #######
###############################################################

""" The Voigt profile. Arguments:
    x:  Frequency displacement in units of Doppler shift.
    a:  The Lyα Voigt parameter, a = 4.7e-4*(T4^(-0.5))

    This function uses the expansion Eq. (12) in:
    Smith et al. (2018), 'Discrete diffusion Lyman α radiative transfer', MNRAS, 479, 2. """
function H(x::Float64, a::Float64)
    H = exp(-x^2) + (2*a/sqrt(π))*(2*x*dawson(x) - 1) + (a^2)*exp(-x^2)*(1 - 2*(x^2))
end


""" Integral ∫dx H(x,a) from x = 0 to x. 
    Uses Eq. (13) from:
    
    Smith et al. (2018), 'Discrete diffusion Lyman α radiative transfer', MNRAS, 479, 2. """
function ∫dx_H(x::Float64, a::Float64)
    (sqrt(π)/2) * erf(x) - (2*a/sqrt(π)) * dawson(x) + (a^2)*x*exp(-x^2)
end


""" Top-hat x-average of H(x,a), over an interval Δx centered
    on x, for a Voigt parameter a. See Eq. (14) in:

    Smith et al. (2018), 'Discrete diffusion Lyman α radiative transfer', MNRAS, 479, 2.  """
function Htophat(x::Float64, a::Float64, Δx::Float64)
    if Δx > 1e-5
        Htophat = ( ∫dx_H(x + Δx/2, a) - ∫dx_H(x - Δx/2, a) ) / Δx
        return Htophat
    else
        # For very small Δx, use H(x,a) instead
        Htophat = H(x, a)
        return Htophat
    end
end


""" The thermal velocity b(T) (cm/s) as a function 
    of temperature T (K). """
function b(T::Float64)
    sqrt( 2 * kB * T / mH )
end


""" Lyα Voigt parameter (dimensionless) as a function
    of temperature T (K). """
function a(T::Float64)
    ALyα / (4 * π * νLyα * b(T)/c )
end

""" Lyα cross-section at line center (cm²) as a function
    of temperature T (K). """
function σ0(T::Float64)
    sqrt(π) * (e^2) * f_lu_Lyα / (me * νLyα * b(T))
end


""" HI column density (cm^-2). 

    ARGUMENTS:

    r   :  Array of r-values to integrate over. 
    nHI :  Corresponding array of HI number densities (cm^-3) """
function N_HI(r::AbstractVector{Float64}, nHI::AbstractVector{Float64})
    # Use the trapezoidal rule to approximate the integral
    Δr = diff(r)
    return 0.5 * sum( Δr .* (nHI[1:end-1] .+ nHI[2:end]) )
end


""" Lyα optical depth at line center. 

    ARGUMENTS:

    r   :  Array of r-values to integrate over. 
    nHI :  Corresponding array of HI number densities (cm^-3)
    T   :  Corresponding array of temperatures (K) """
function τ(r::AbstractVector{Float64}, nHI::AbstractVector{Float64}, T::AbstractVector{Float64})
    # Calculate σ0(T) at each radial point
    σ0_values = σ0.(T)
    
    # Calculate the integrand nHI(r) * σ0(T(r))
    integrand = nHI .* σ0_values

    # Use the trapezoidal rule to approximate the integral
    Δr = diff(r)
    return 0.5 * sum( Δr .* (integrand[1:end-1] .+ integrand[2:end]) )
end


""" Numerical derivative dF/dr of a function F(r). Uses central differences
    for interior points, and forward (backwards) differences at inner (outer)
    boundary. Accurate to O(Δr²) for both uniform and non-uniform grids. """
function deriv(F::AbstractVector{Float64}, r::AbstractVector{Float64})
    Nr   = length( r )
    dFdr = zeros( Nr )
    for i in 2:Nr-1
        # Central difference for interior points:
        Δrp = r[i+1] - r[i]; Δrm = r[i] - r[i-1]; K = (Δrp/Δrm)^2
        dFdr[i] = ( F[i+1] - K*F[i-1] - (1-K)*F[i] ) / ( Δrp*(1 + Δrp/Δrm) )
    end
    # Forward & backward difference for the first point & last points:
    Δr21 = r[2] - r[1]; Δr31 = r[3] - r[1]; Δr32 = r[3] - r[2]

    dFdr[1]  = ( -F[3]*(Δr21^2) + F[2]*(Δr31^2) - F[1]*(Δr31^2 - Δr21^2)) / ( Δr21*Δr31*Δr32 )
    dFdr[Nr] = ( F[Nr] - F[Nr-1] ) / ( r[Nr] - r[Nr-1] )

    return dFdr
end


""" The divergence of a function F(r): ∇⋅F = ∂ᵣ(r²F)/r². """
function divergence(F::AbstractVector{Float64}, r::AbstractVector{Float64})
    rsq   = r.^2
    return deriv( rsq .* F, r ) ./ rsq
end


""" The Lyα escape fraction from a cloud of radius Rcl (cm) 
    and total Lyα luminosity L_Lyα (erg/s), given the Lyα flux
    F_Lyα (erg/s/cm^2). """
function LyαEscapeFraction(F_Lyα::Float64, L_Lyα::Float64, Rcl::Float64)
    F_Lyα / ( L_Lyα/(4 * π * Rcl^2) )
end


""" This function computes the Lyα pressure P_Lyα, its gradient
    dP_Lyα/dr, and the Lyα force F_Lyα, for a given input 
    frequency-integrated Lyα intensity Jtot(r) ≡ ∫dν J(r,ν).  """
function LyαPressureAndForce(r::AbstractVector{Float64}, Jtot::AbstractVector{Float64},
                            ΔV::AbstractVector{Float64})
    Nr = length(r)
    ### Compute P_Lyα = 4 π Jtot / 3 c and dP_Lyα/dr ###
    PLyα     = 4 .* π .* Jtot ./ (3 .* c)
    dPLyα_dr = deriv( PLyα, r )

    ### Compute total Lyα force on cloud F_Lyα ###
    # Use F_Lyα = - ∫ dV dP_Lyα/dr ≈ - ∑_{i} ΔV_i x (dP_Lyα/dr)_i 
    FLyα  =  - sum( ΔV .* dPLyα_dr )

    return PLyα, dPLyα_dr, FLyα
end


""" This function computes the gas pressure P_gas, its gradient
    dP_gas/dr, and the total gas pressure force F_gas on the cloud.  """
function GasPressureAndForce(r::AbstractVector{Float64}, n::AbstractVector{Float64}, 
                             T::AbstractVector{Float64}, ΔV::AbstractVector{Float64})

    Nr = length(r)

    ### Compute the gas pressure P_gas ###
    Pgas = n .* kB .* T

    ### Compute dP_gas/dr ###
    dPgas_dr = zeros( Nr )
    for i in 2:Nr-1
        # Central difference for interior points:
        dPgas_dr[i] = ( Pgas[i+1] - Pgas[i-1] ) / ( r[i+1] - r[i-1] )
    end
    # Forward & backward difference for the first point & last points:
    dPgas_dr[1]   = ( Pgas[2] - Pgas[1]) / ( r[2] - r[1] )
    dPgas_dr[Nr]  = ( Pgas[Nr] - Pgas[Nr-1] ) / ( r[Nr] - r[Nr-1] )

    ### Compute total gas pressure force on cloud F_gas ###

    Fgas = - sum( ΔV .* dPgas_dr )

    return Pgas, dPgas_dr, Fgas
end

###############################################################
#######                                                 #######
#######      A N A L Y T I C A L  S O L U T I O N S     #######
#######                                                 #######
###############################################################

# Below are some analytical spectra to compare numerically
# solutions against. 

""" Dijkstra+ (2006) solution for the emergent spectrum
    from a uniform, static, dust-free sphere with a 
    central Lyα point source. Arguments:
    
    **ARGUMENTS:**
    
    x     :  Dimensionless frequency (ν - νLyα)/Δν_D
    T     :  Temperature (K)
    τcl   :  Cloud optical depth at Lyα line center 
    
    **RETURNS:**
    
    J(x)  :  Normalized emergent spectrum, J(x)/∫dx J(x) 
"""
function J_sphere_point(x::Float64, T::Float64, τcl::Float64)
    aτ = a(T)*τcl
    J  = sqrt(π^3 / 24) * (x^2 / aτ) * sech( sqrt(π^3 / 54) * abs.(x^3) / aτ )^2
    return J
end


""" Lao & Smith (2020) solution for the emergent spectrum
    from a uniform, static, dust-free sphere with a 
    uniform Lyα source. 
    
    **ARGUMENTS:**
    
    x     :  Dimensionless frequency (ν - νLyα)/Δν_D
    T     :  Temperature (K)
    τcl   :  Cloud optical depth at Lyα line center 
    
    **RETURNS:**
    
    J(x)  :  Normalized emergent spectrum, J(x)/∫dx J(x) 
"""
function J_sphere_uni(x::Float64, T::Float64, τcl::Float64)
    aτ = a(T)*τcl
    J  = - sqrt(6/π) * (x^2 / aτ) * log(1 - exp(-sqrt(2 * π^3 / 27) * abs.(x^3) / aτ))
    return J
end


# Below are the predicted Lyα pressure PLyα(r) for idealized setups, 
# which can be used to compare simulations to. We also define needed 
# series to evaluate PLyα(r).

""" The polylogarithmic function Li_s(z):
    
    Li_s(z) ≡ ∑_{n = 1 to ∞} z^n / n^s .
   
    Needed for some analytical expressions. 
"""
function Li(z::Number, s::Float64, nmax::Int = 100000)
    n      = 1:nmax
    # All terms:
    terms  = (z.^n) ./ (n.^s)
    # Sum all terms to get Li:
    Li     = sum( terms )
    return Li
end


""" Predicted Lyα pressure PLyα for a uniform, static, dust-free 
    cloud with a central Lyα point source, as found by Lao & Smith (2020).
    
    **ARGUMENTS:**
    
    r         :   Array of r-values (cm) 
    L_Lyα     :   Lyα luminosity (erg/s)
    T         :   Cloud temperature (K)
    τcl       :   Cloud optical depth at Lyα line center 
    
    **RETURNS:**
    
    PLyα(r)   :   Array of PLyα for all r-values 
 """
function PLyα_sphere_point(r::AbstractVector{Float64}, L_Lyα::Float64, T::Float64, τcl::Float64)

    Rcl  = r[end]                         
    rbar = r ./ Rcl
    aτ   = a(T)*τcl          

    # Put it all together to compute PLyα:
    stuff = (L_Lyα/(3 * c * Rcl^2)) * gamma(1/3) * (2 * aτ)^(1/3) / (2 * π^(3/2))
    X     = exp.( im .* π .* rbar )  
    PLyα  = (stuff ./ rbar) .* imag.( Li.(X, 1/3, 1000000) )

    return PLyα
end


""" Predicted Lyα pressure PLyα for a uniform, static, dust-free 
    cloud with a uniform Lyα source, as found by Lao & Smith (2020).
    
    **ARGUMENTS:**
    
    r         :   Array of r-values (cm) 
    L_Lyα     :   Lyα luminosity (erg/s)
    T         :   Cloud temperature (K)
    τcl       :   Cloud optical depth at Lyα line center 
    
    **RETURNS:**
    
    PLyα(r)   :   Array of PLyα for all r-values 
 """
function PLyα_sphere_uni(r::AbstractVector{Float64}, L_Lyα::Float64, T::Float64, τcl::Float64)

    Rcl  = r[end]                         
    rbar = r ./ Rcl
    aτ   = a(T)*τcl          

    # Put it all together to compute PLyα:
    stuff = (L_Lyα/(c * Rcl^2)) * gamma(1/3) * (2 * aτ)^(1/3) / (2 * π^(7/2))
    X     = - exp.( - im .* π .* rbar )  
    PLyα  = (stuff ./ rbar) .* imag.( Li.(X, 7/3) )

    return PLyα
end

end # End of module