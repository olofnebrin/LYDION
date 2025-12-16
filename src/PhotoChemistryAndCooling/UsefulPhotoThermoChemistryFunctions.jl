
""" FILE:        UsefulPhotoThermoChemistryFunctions.jl

    PURPOSE:     Contains some useful functions for photo-chemistry,
                 and setting up arrays both for chemistry and stellar RT. 

    REFERENCES:  (1) Black & Bodenheimer (1975) 'Evolution of rotating 
                 interstellar clouds. I. Numerical techniques.', 
                 ApJ, 199, 1, 619-632

                 (2) Omukai & Nishi (1998) 'Formation of primordial protostars',
                 ApJ, 508, 1, 141-150

                 (3) Sharda et al. (2019) 'The role of the H₂ adiabatic index in
                 the formation of the first stars', MNRAS, 490, 1, 513-526

"""


using .ConstantsAndFunctions      # For hydrogen & helium mass fractions X and Y 
using .StellarEmissionAbsorption  # For stellar luminosities 

export GetN_H2, S_H2, S_HD, S_H2_HD, Getne, 
       GenerateThermoChemistryArrays, GenerateStellarRTArrays


const S_H2_1::Float64 = 0.965;    const S_H2_2::Float64  = 0.035
const c_SH2::Float64  = 8.5e-4; const N_H2crit::Float64  = 5e14

""" The H₂ self-shielding factor, as fitted by 
    Draine & Bertoldi (1996). """
@inline function S_H2( N_H2::Float64, 
                          T::Float64
                       )
    T    = clamp( T, 1.0, 1e5 )
    x    = N_H2 / N_H2crit
    F    = sqrt( 1 + x )
    b5   = sqrt( 2 * kB * T / (2 * mH) ) / kms # Mass of H₂ is ≃ 2 × mH
    S_H2 = S_H2_1 / (1 + x/b5)^2  +  S_H2_2 * exp( - c_SH2 * F ) / F
    return S_H2
end


""" The HD self-shielding factor, same fit as for H₂. """
@inline function S_HD( N_HD::Float64, 
                          T::Float64
                       )
    T    = clamp( T, 1.0, 1e5 )
    x    = N_HD / N_H2crit
    F    = sqrt( 1 + x )
    b5   = sqrt( 2 * kB * T / (3 * mH) ) / kms  # Mass of HD is ≃ 3 × mH
    S_HD = S_H2_1 / (1 + x/b5)^2  +  S_H2_2 * exp( - c_SH2 * F ) / F
    return S_HD
end


const N_H2_HDcrit = 2.34e19; const c_H2_HD = 5.2e-3; const p_H2_HD = 0.238 

""" The shielding of HD by H₂. See Wolcott-Green & Haiman (2011). """
@inline function S_H2_HD( N_H2::Float64 )
    x       = N_H2 / N_H2_HDcrit
    S_H2_HD = exp( - c_H2_HD * x ) / (1 + x)^p_H2_HD
    return S_H2_HD
end


""" This function computes the electron nunber density
    given the fractional abundances of H II, He II, and
    He III, as well as the total number densities of hydrogen 
    and helium. """
@inline @fastmath function Getne( xHII::Float64, 
                                 xHeII::Float64,
                                xHeIII::Float64,
                                    nH::Float64,
                                   nHe::Float64
                                 )
    xHII * nH  +  ( xHeII  +  2 * xHeIII ) * nHe
end
