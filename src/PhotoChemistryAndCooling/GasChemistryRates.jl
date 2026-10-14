
""" FILE:       GasChemistryRates.jl

    PURPOSE:    Contains all gas chemistry rates involving 
                e⁻, H I, H II, H₂, D I, D II, HD, He I, He II, He III,
                C I, C II, C III, CO, CHₓ, O I, O II, O III, OHₓ, 
                S I, S II, S III, S IV.

    REFERENCES: The rates below come from several sources (most of them summarized 
                in the first few papers below):

                (1) Rosdahl et al. (2013) 'RAMSES-RT: radiation hydrodynamics in the
                cosmological context', MNRAS, 436, 3, 2188-2231

                (2) Hopkins et al. (2023) 'FIRE-3: updated stellar evolution models, 
                yields, and microphysics and fitting functions for applications 
                in galaxy simulations', MNRAS, 519, 2, 3154-3181

                (3) Glover et al. (2010) 'Modelling CO formation in the turbulent 
                interstellar medium', MNRAS, 404, 1, 2-29

                (4) Glover & Jappsen (2007) 'Star formation at very low metallicity. I.
                Chemistry and cooling at low densities', ApJ, 666, 1, 1-19

                (5) Galli & Palla (1998) 'The chemistry of the early Universe', A&A, 
                335, 403-420

                (6) Kreckel et al. (2010) 'Experimental Results for H₂ Formation from H⁻ 
                and H and Implications for First Star Formation', Science, 329, 5987 

                (7) Faure et al. (2023) 'The chemistry and excitation of H2 and HD in 
                the early Universe', MNRAS, 531, 340-354

                (8) Gong et al. (2017) 'A Simple and Accurate Network for Hydrogen and 
                Carbon Chemistry in the Interstellar Medium', ApJ, 843, 1

                (9) Schauer et al. (2019) 'The influence of streaming velocities on the
                formation of the first stars', MNRAS, 484, 3510-3521

                (10) Badnell (2006) 'Radiative recombination data for modelling dynamic 
                finite-density plasmas', ApJS, 167, 334-342 

                (11) Zatsarinny et al. (2004) 'Dielectronic recombination data for dymamic
                finite-density plasmas: IV. The carbon isoelectronic sequence', A&A, 417,
                1173-1181

                (12) Voronov (1997) 'A practical fit formula for ionization rate coefficients
                of atoms and ions by electron impact: Z = 1-28', Atomic Data and Nuclear Data 
                Tables, 65, 1-35

                (13) Kingdon & Ferland (1996) 'Rate coefficients for charge transfer between 
                hydrogen and the first 30 elements', ApJS, 106, 205-211

                (14) Colgan et al. (2003) 'Dielectronic recombination data for dynamic 
                finite-density plasmas: III. The beryllium isoelectronic sequence', A&A, 412,
                597-601

                (15) Bleda et al. (2022) 'Dielectronic recombination data for dynamic 
                finite-density plasmas: XVI. The phosphorus isoelectronic sequence', A&A, 668,
                A72 

                (16) Kaur et al. (2018) 'Dielectronic recombination data for dynamic 
                finite-density plasmas: XV. The silicon isoelectronic sequence', A&A, 610, A41

                (17) Abdel-Naby et al. (2012) 'Dielectronic recombination data for dynamic 
                finite-density plasmas: XIV. The aluminum isoelectronic sequence', A&A, 537, A40

                (19) Altun et al. (2004) 'Dielectronic recombination data for dynamic
                finite-density plasmas: VI. The boron isoelectronic sequence', A&A, 420, 775-781

"""

#include(joinpath(@__DIR__, "..", "ConstantsAndFunctions.jl"))
using .ConstantsAndFunctions  # Needed constants 
using BenchmarkTools          # For benchmarking performance of interpolations 

export krecHII_1, krecHII, krecHeII_1, krecHeII, krecHeIII_1, krecHeIII, 
       kcollion_HI, kcollion_HeI, kcollion_HeII,
       R_dustH2, kGP_H2, k3B_H2, kdiss_H2HI, kdiss_H2H2, kdiss_H2e, kdiss_H2HII,
       k50_D, k51_D, k52_D, k53_D, k54_D, k55_D, kCIe, kOIe, kCIIe, kCIIIe,
       kOIIe, kOIIIe, kOIVe, kOIIe_collion, kOIIIe_collion,
       kCIIe_collion, kOIIIHI, kCIIIHI, kCIIH2_1, kCIIH2_2, 
       kHIIOI, kOIIHI, kOIIIHI, kOIVHI, kOIIH2_1, kOIIH2_2, 
       kCHxOI, kCHxHI, kOHxCI, kOHxOI, kCIH3plus, kOIH3plus, kH3pluse,
       kSIIe, kSIIIe, kSIVe, kSIe_collion, kSIIe_collion, kSIIIe_collion,
       kSIHII, kSIIHI, kSIIIHI, kSIVHI


###############################################################
#######                                                 #######
#######        R A T E  I N T E R P O L A T I O N       #######
#######                                                 #######
###############################################################

struct ChemistryRateLUT
     log10T_min::Float64
     log10T_max::Float64
    inv_dlog10T::Float64
         log10k::Vector{Float64}
end


""" Create a linear interpolation table in log10T - log10k space. """
function InitializeChemistryRateLUT( RateFit::F,       # The analytical rate fit which we want to interpolate
                                       T_min::Float64, # Minimum considered temperature 
                                       T_max::Float64, # Maximum considered temperature 
                                           N::Int = 512
                                  ) where {F}

    log10T_min  = log10(T_min)
    log10T_max  = log10(T_max)
    dlog10T     = (log10T_max - log10T_min) / (N - 1)
    inv_dlog10T = 1.0 / dlog10T

    log10k      = Vector{Float64}(undef, N)

    @inbounds for i in 1:N
        log10T = log10T_min  +  (i - 1) * dlog10T
        T      = 10.0^log10T
        k      = RateFit(T)

        # Avoid log10(0). This should usually not matter for the rates here,
        # but protects against exact zeros in threshold-like rates.
        log10k[i] = log10( max(k, 1e-300) )
    end

    return ChemistryRateLUT(log10T_min, log10T_max, inv_dlog10T, log10k)
end


""" Evaluate a chemistry rate LUT. Linear in log10(T) and log10(k),
    where k is the rate coefficient of interest. """
@inline function EvalChemistryRateLUT( log10T::Float64, LUT::ChemistryRateLUT )

    # At end points, just clamp:
    if log10T ≤ LUT.log10T_min
        return @inbounds 10.0^LUT.log10k[1]
    elseif log10T ≥ LUT.log10T_max
        return @inbounds 10.0^LUT.log10k[end]
    end

    # Get interpolated value of log10(k), where k is the rate coefficient:
    x  = @fastmath (log10T  -  LUT.log10T_min) * LUT.inv_dlog10T
    i  = unsafe_trunc(Int, x) + 1
    f  = @fastmath x  -  (i - 1)

    log10k_i = @inbounds LUT.log10k[i]
    log10k_p = @inbounds LUT.log10k[i+1]

    # Interpolated value of log10 of rate coefficient: 
    log10k   = @fastmath log10k_i  +  f * (log10k_p - log10k_i)

    return @fastmath 10.0^log10k
end


###############################################################
#######                                                 #######
#######          H Y D R O G E N  &  H E L I U M        #######
#######                                                 #######
###############################################################


""" The H II Case A recombination rate (cm³ s⁻¹) as a function of temperature T (K).
    See ref. (1). """       
@inline function krecHII_A( T::Float64 )
    T         = clamp( T, 1.0, 1e9 )
    λHI       = @fastmath 315614.0 / T
    krecHII_A = @fastmath 1.269e-13 * (λHI^1.503) / ( 1 + (λHI/0.522)^0.47 )^1.923
    return krecHII_A
end


""" The H II Case B recombination rate (cm³ s⁻¹) as a function of temperature T (K).
    See ref. (1). """       
@inline function krecHII_B( T::Float64 )
    T         = clamp( T, 1.0, 1e9 )
    λHI       = @fastmath 315614.0 / T
    krecHII_B = @fastmath 2.753e-14 * (λHI^1.5) / ( 1 + (λHI/2.74)^0.407 )^2.242
    return krecHII_B
end


""" The H II LyC → ground state recombination rate (cm³ s⁻¹) as a function of temperature T (K).
    Useful if EUV1 recombination radiation is modelled. """       
@inline function krecHII_1( T::Float64 )
    return max( krecHII_A(T) - krecHII_B(T), 0.0 )
end


""" The H II recombination rate (cm³ s⁻¹) as a function of temperature T (K).
    Returns the Case B coefficient if on-the-spot approximation is used, 
    and the Case A rate otherwise. """
@inline function krecHII( T::Float64, 
                        OTS::Bool = true
                         )
    if OTS  
        # Case-B if on-the-spot (OTS) approximation is used:
        return krecHII_B(T)
    else
        # Otherwise Case-A:
        return krecHII_A(T)
    end
end


""" The He II Case A recombination rate (cm³ s⁻¹) as a function of temperature T (K).
    See ref. (1). """       
@inline function krecHeII_A( T::Float64 )
    T          = clamp( T, 1.0, 1e9 )
    λHeI       = @fastmath 570670.0 / T
    krecHeII_A = @fastmath 3e-14 * (λHeI^0.654)
    return krecHeII_A
end


""" The He II Case B recombination rate (cm³ s⁻¹) as a function of temperature T (K).
    See ref. (1). """       
@inline function krecHeII_B( T::Float64 )
    T          = clamp( T, 1.0, 1e9 )
    λHeI       = @fastmath 570670.0 / T
    krecHeII_B = @fastmath 1.26e-14 * (λHeI^0.75)
    return krecHeII_B
end


""" The He II continuum → ground state recombination rate (cm³ s⁻¹) as a function of temperature T (K).
    Useful if EUV2 recombination radiation is modelled. """       
@inline function krecHeII_1( T::Float64 )
    return max( krecHeII_A(T) - krecHeII_B(T), 0.0 )
end


""" The He II (Case-B or Case-A) recombination rate (cm³ s⁻¹) as a function of temperature T (K).
    See ref. (1). """
@inline function krecHeII( T::Float64, 
                         OTS::Bool = true
                          )
    if OTS
        # Case-B if on-the-spot (OTS) approximation is used:
        return krecHeII_B(T)
    else
        # Case-A:
        return krecHeII_A(T)
    end
end


""" The He III Case A recombination rate (cm³ s⁻¹) as a function of temperature T (K).
    See ref. (1). """       
@inline function krecHeIII_A( T::Float64 )
    T           = clamp( T, 1.0, 1e9 ) 
    λHeII       = @fastmath 1263030.0 / T 
    krecHeIII_A = @fastmath 2.538e-13 * (λHeII^1.503) / (1 + (λHeII/0.522)^0.47)^1.923
    return krecHeIII_A
end


""" The He III Case B recombination rate (cm³ s⁻¹) as a function of temperature T (K).
    See ref. (1). """       
@inline function krecHeIII_B( T::Float64 )
    T           = clamp( T, 1.0, 1e9 ) 
    λHeII       = @fastmath 1263030.0 / T 
    krecHeIII_B = @fastmath 5.506e-14 * (λHeII^1.5) / (1 + (λHeII/2.74)^0.407 )^2.242 
    return krecHeIII_B
end


""" The He III continuum → ground state recombination rate (cm³ s⁻¹) as a function of temperature T (K).
    Useful if EUV3 recombination radiation is modelled. """       
@inline function krecHeIII_1( T::Float64 )
    return max( krecHeIII_A(T) - krecHeIII_B(T), 0.0 )
end


""" The He III (Case-B or Case-A) recombination rate (cm³ s⁻¹) as a function of temperature T (K).
    See ref. (1). """
@inline function krecHeIII( T::Float64, 
                          OTS::Bool = true
                           )
    if OTS 
        # Case-B if on-the-spot (OTS) approximation is used:
        return krecHeIII_B(T)
    else
        # Case-A:
        return krecHeIII_A(T)
    end
end


const kcollion_HI0::Float64 = 5.85e-11; const T_collionHI::Float64 = 157809.1

""" The H I collisional ionization rate (cm³ s⁻¹) by electrons,
    as a function of temperature T (K). See ref. (1). """
@inline function kcollion_HI( T::Float64 )

    if T ≤ 300.0
        # Ignore collisional ionization in cold gas, to avoid unnecessary 
        # rate evaluations 
        return 0.0
    end

    return @fastmath kcollion_HI0 * sqrt(T) * exp( - T_collionHI / T ) / ( 1 + sqrt(T/1e5) )
end


const kcollion_HeI0::Float64 = 2.38e-11; const T_collionHeI::Float64 = 285335.4 

""" The He I collisional ionization rate (cm³ s⁻¹) by electrons,
    as a function of temperature T (K). See ref. (1). """
@inline function kcollion_HeI( T::Float64 )

    if T ≤ 300.0
        # Ignore collisional ionization in cold gas, to avoid unnecessary 
        # rate evaluations 
        return 0.0
    end

    return @fastmath kcollion_HeI0 * sqrt(T) * exp( - T_collionHeI / T ) / ( 1 + sqrt(T/1e5) )
end


const kcollion_HeII0::Float64 = 5.68e-12; const T_collionHeII::Float64 = 631515.0

""" The He II collisional ionization rate (cm³ s⁻¹) by electrons,
    as a function of temperature T (K). See ref. (1). """
@inline function kcollion_HeII( T::Float64 )

    if T ≤ 300.0
        # Ignore collisional ionization in cold gas, to avoid unnecessary 
        # rate evaluations 
        return 0.0
    end

    return @fastmath kcollion_HeII0 * sqrt(T) * exp( - T_collionHeII / T ) / ( 1 + sqrt(T/1e5) )
end


###############################################################
#######                                                 #######
#######        M O L E C U L A R  H Y D R O G E N       #######
#######               C H E M I S T R Y                 #######
#######                                                 #######
###############################################################

const Tdust_crit_H2::Float64 = 600.0

""" The combination S_{HI} × ϵ_{H₂}, i.e. the sticking coefficient times 
    the H₂ formation efficiency on dust grains, as a function of the gas 
    temperature T (K) and dust temperature Td (K). 
    From Hollenbach & McKee (1979). """
@inline function S_HI_ϵ_H2( T::Float64, Td::Float64 )

    Td = max( Td, 0.1 )

    @fastmath S_HI_ϵ = 1 / ( ( 1  +  0.04 * (T + Td)^0.5  +  0.002 * T  +  8e-6 * (T * T) ) 
                           * ( 1  +  1e4 * exp( - Tdust_crit_H2 / Td ) ) )

    return S_HI_ϵ
end


const Const_vth_HI::Float64 = sqrt( 8 * kB / (π * mH) )  # Pre-factor for vth 

""" The H₂ formation rate coefficient for dust (s⁻¹) as a function of
    temperature T (K):

    R_{dust,H₂} = (3/8) ∑_j (ρdj / ρgrj) × (1 / āgrj) × sqrt( 8 kB T / π mH ) × S_{HI} × ϵ_{H₂}
    
    See e.g. Hollenbach & McKee (1979) and Draine (2011). Uses pre-computed grain 
    properties for silicate and graphite/PAH dust. """
@inline function R_dustH2( T::Float64, 
                          ρd::NTuple{3, Float64}, 
                          Td::NTuple{3, Float64} )

    # Unpack ρd and Td:
    ρd_Sil, ρd_C, ρd_PAH = ρd 
    Td_Sil, Td_C, Td_PAH = Td 

    # Clamp temperature (just for numerical reasons):
    T   = clamp( T, 0.0, 1e9 )  

    # Get vth = sqrt( 8 kB T / π mH ):
    vth = @fastmath Const_vth_HI * sqrt( T )

    # Get S_{HI} × ϵ_{H₂}:
    Sϵ_Sil  = S_HI_ϵ_H2( T, Td_Sil )
    Sϵ_C    = S_HI_ϵ_H2( T, Td_C )
    Sϵ_PAH  = S_HI_ϵ_H2( T, Td_PAH )

    # Get R_{dust,H₂}:
    @fastmath R_DustH2 = ThreeEights * vth * ( inv_agr_Sil * ρd_Sil * inv_ρgr_Sil * Sϵ_Sil 
                                             + inv_agr_C   * ρd_C   * inv_ρgr_C   * Sϵ_C 
                                             + inv_agr_PAH * ρd_PAH * inv_ρgr_C   * Sϵ_PAH )

    return R_DustH2
end


###### H + e⁻ → H⁻ + hν #######

const k1_0::Float64 = 1.4e-18
const T_k1::Float64 = 16200.0; const inv_T_k1::Float64 = 1 / T_k1

""" H + e⁻ → H⁻ + hν rate (cm³ s⁻¹; from Galli & Palla 1998). """
@inline function k1( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    return @fastmath k1_0 * (T^0.928) * exp( - T * inv_T_k1 )
end


###### H⁻ + H → H₂ + e⁻ ########

# Fitting constants from Kreckel et al. (2010):
const c1_k2::Float64  = 1.35e-9;   const c2_k2::Float64  = 9.8493e-2;  const c3_k2::Float64  = 3.2852e-1
const c4_k2::Float64  = 5.5610e-1; const c5_k2::Float64  = 2.7710e-7;  const c6_k2::Float64  = 2.1826
const c7_k2::Float64  = 6.1910e-3; const c8_k2::Float64  = 1.0461;     const c9_k2::Float64  = 8.9712e-11
const c10_k2::Float64 = 3.0424;    const c11_k2::Float64 = 3.2576e-14; const c12_k2::Float64 = 3.7741

""" H⁻ + H → H₂ + e⁻ rate (cm³ s⁻¹; from Kreckel et al. 2010). """
@inline function k2( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    return @fastmath c1_k2 * (T^c2_k2 + c3_k2 * T^c4_k2 + c5_k2 * T^c6_k2 ) / (
                              1 + c7_k2 * T^c8_k2 + c9_k2 * T^c10_k2 + c11_k2 * T^c12_k2 )
end


###### H⁻ + H⁺ → H + H ######

const k5_0::Float64 = 2.4e-6
const T_k5::Float64 = 2e4;   const inv_T_k5::Float64 = 1 / T_k5

""" H⁻ + H⁺ → H + H rate (cm³ s⁻¹; from Glover et al. 2010, which 
    is updated relative to Glover & Jappsen 2007).  """
@inline function k5( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    return @fastmath k5_0 * (T^(-0.5)) * ( 1.0  +  T * inv_T_k5 )
end


###### H⁻ + e⁻ → H + e⁻ + e⁻ ######

const c0_k15::Float64 = - 1.801849334e1; const c1_k15::Float64 = 2.36085220e0
const c2_k15::Float64 = - 2.82744300e-1; const c3_k15::Float64 = 1.62331664e-2
const c4_k15::Float64 = - 3.36501203e-2; const c5_k15::Float64 = 1.17832978e-2
const c6_k15::Float64 = - 1.65619470e-3; const c7_k15::Float64 = 1.06827520e-4
const c8_k15::Float64 = - 2.63128581e-6

""" H⁻ + e⁻ → H + e⁻ + e⁻ rate (cm³ s⁻¹). Rate/fit is taken from 
    Glover & Jappsen (2007)/Glover et al. (2010) (they are the same 
    in both references).  """
@inline function k15( T::Float64 )
    # Convenient definitions: 
    T  = clamp( T, 1.0, 1e5 )
    x  = @fastmath log( kB * T * inv_eV )
    x2 = x*x;  x3 = x2*x; x4 = x3*x
    x5 = x4*x; x6 = x5*x; x7 = x6*x; x8 = x7*x 

    # Rate:
    @fastmath k15 = exp( c0_k15 + c1_k15*x + c2_k15*x2 + c3_k15*x3 + c4_k15*x4
                       + c5_k15*x5 + c6_k15*x6 + c7_k15*x7 + c8_k15*x8 )
    
    return k15
end


###### H⁻ + H → H + H + e⁻ ######

const k16_0::Float64  = 2.5634e-9
const c0_k16::Float64 = - 2.0372609e1;  const c1_k16::Float64 = 1.13944933e0
const c2_k16::Float64 = - 1.4210135e-1; const c3_k16::Float64 = 8.4644554e-3
const c4_k16::Float64 = - 1.4327641e-3; const c5_k16::Float64 = 2.0122503e-4
const c6_k16::Float64 = 8.6639632e-5;   const c7_k16::Float64 = - 2.5850097e-5
const c8_k16::Float64 = 2.4555012e-6;   const c9_k16::Float64 = - 8.0683825e-8

""" H⁻ + H → H + H + e⁻ rate (cm³ s⁻¹). Rate/fit is taken from 
    Glover & Jappsen (2007)/Glover et al. (2010) (they are the same 
    in both references).  """
function k16( T::Float64 )
    TeV = kB * clamp( T, 1.0, 1e5 ) * inv_eV
    if TeV ≤ 0.1
        return @fastmath k16_0 * (TeV^1.78186)
    else
        # Convenient definitions:
        x = log( TeV ); x2 = x*x; x3 = x2*x; x4 = x3*x
        x5 = x4*x; x6 = x5*x; x7 = x6*x; x8 = x7*x; x9 = x8*x 

        # Rate:
        @fastmath k16 = exp( c0_k16 + c1_k16*x + c2_k16*x2 + c3_k16*x3 + c4_k16*x4
                           + c5_k16*x5 + c6_k16*x6 + c7_k16*x7 + c8_k16*x8 + c9_k16*x9 )

        return k16
    end
end


###### H⁻ + H⁺ → H₂⁺ + e⁻ ######

const k17_0low::Float64 = 6.9e-9; const k17_0high::Float64 = 9.6e-7
const p_k17low::Float64 = - 0.35; const p_k17high::Float64 = - 0.90

""" H⁻ + H⁺ → H₂⁺ + e⁻ rate (cm³ s⁻¹). Rate/fit is taken from 
    Glover & Jappsen (2007)/Glover et al. (2010) (they are the same 
    in both references).  """
function k17( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    if T ≤ 8e3
        return @fastmath k17_0low * T^p_k17low
    else
        return @fastmath k17_0high * T^p_k17high
    end
end


###### Gas phase H₂ formation rate ######

const Γpd_CMB10::Float64 = 0.11;   const Γpd_CMB20::Float64 = 8e-8
const cpd_CMB1::Float64  = 2.13;   const cpd_CMB2::Float64  = 1.3 
const Tpd_CMB1::Float64  = 8823.0; const Tpd_CMB2::Float64  = 2300.0

""" Total H₂ formation rate (cm³ s⁻¹) via the 
    H⁻ channel, assuming the steady-state abundance of 
    the latter. See ref. (2). """ 
function kGP_H2( T::Float64, 
                ne::Float64,
               nHI::Float64,
              nHII::Float64,
               Γpd::Float64 = 0.0, # H⁻ photodetachment rate by non-CMB photons 
                 z::Float64 = 0.0  # Redshift
                 )

    # Needed rates: 
    K1  = k1(T);  K2  = k2(T); K5 = k5(T); K15 = k15(T)
    K16 = k16(T); K17 = k17(T)

    # CMB photodetachment rate of H⁻ (see Schauer et al. 2019):
    TCMB     = @fastmath TCMB0 * (1 + z)
    inv_TCMB = @fastmath 1.0 / TCMB
    Γpd_CMB  = @fastmath ( Γpd_CMB10 * (TCMB^cpd_CMB1) * exp( - Tpd_CMB1 * inv_TCMB )
                         + Γpd_CMB20 * (TCMB^cpd_CMB2) * exp( - Tpd_CMB2 * inv_TCMB ) )

    # Total photodetachment rate from non-CMB photons + CMB photons:
    Γpd_tot = @fastmath   Γpd +  Γpd_CMB

    # Full rate coefficient, assuming equilibrium abundance of H⁻:
    kGP_H2  = @fastmath ( K2 * K1 * nHI ) / ( (K2 + K16) * nHI  +  (K5 + K17) * nHII  +  K15 * ne  +  Γpd_tot )

    return kGP_H2
end


""" The 3-body H₂ formation rate (cm⁶ s⁻¹) as a function of temperature T (K).
    See ref. (2) (taken from Forrey 2013). """
@inline function k3B_H2( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    return @fastmath 6e-32 * T^(-0.25)  +  2e-31 * T^(-0.5)
end


###### Collisional destruction of H₂ ########

# Some fitting coefficients (Glover et al. 2010)
const kdisc_H2HI_low0::Float64  = 6.67e-12; const T_discH2HI_low::Float64  = 63590.0 
const kdisc_H2HI_high0::Float64 = 3.52e-9;  const T_discH2HI_high::Float64 = 43900.0

""" The H₂-H I collisional dissociation rate (cm³ s⁻¹) as a function 
    of temperature T (K) and HI number density nHI (cm⁻³).
    See Glover et al. (2010). """
function kdiss_H2HI( T::Float64, 
                   nHI::Float64
                    )
    T     = clamp( T, 100.0, 1e5 )
    inv_T = @fastmath 1.0 / T

    # Low & high-density limits of the reaction rate:
    kdisc_H2HI_low  = @fastmath kdisc_H2HI_low0 * sqrt(T) * exp( - (1.0  +  T_discH2HI_low * inv_T) )
    kdisc_H2HI_high = @fastmath kdisc_H2HI_high0 * exp( - T_discH2HI_high * inv_T )

    # Critical density for transition from low-density to high-density limits
    # (note that Glover et al. 2010 has "3.0", but in the reference cited (Lepp & Shull 1983)
    # it is actually "4.0", so I have corrected this):
    x   = log10(T * 1e-4)  # log₁₀(T₄)
    ncr = @fastmath 10^( 4.0 - 0.416 * x - 0.327 * x*x )

    return @fastmath 10^( log10(kdisc_H2HI_high) - log10(kdisc_H2HI_high/kdisc_H2HI_low) / (1 + nHI/ncr) )
end


# Some fitting coefficients (Glover et al. 2010)
const kdisc_H2H2_low0::Float64  = 5.996e-30; const T_discH2H2_low::Float64  = 54657.0
const kdisc_H2H2_high0::Float64 = 1.3e-9;    const T_discH2H2_high::Float64 = 53300.0

""" The H₂-H₂ collisional dissociation rate (cm³ s⁻¹) as a function 
    of temperature T (K) and H₂ number density nHI (cm⁻³).
    See Glover et al. (2010). """
function kdiss_H2H2( T::Float64, 
                   nH2::Float64
                    )
                    
    T     = clamp( T, 100.0, 1e5 )
    inv_T = @fastmath 1.0 / T
    
    # Low & high-density limits of the reaction rate:
    kdisc_H2H2_low  = @fastmath kdisc_H2H2_low0 * (T^4.1881) * exp( - T_discH2H2_low * inv_T ) / (
                                1.0 + 6.761e-6 * T   )^5.6881

    kdisc_H2H2_high = @fastmath kdisc_H2H2_high0 * exp( - T_discH2H2_high * inv_T )

    # Critical density for transition from low-density to high-density limits:
    x   = log10(T * 1e-4)  # log₁₀(T₄)
    ncr = @fastmath 10^( 4.845 - 1.3 * x + 1.62 * x*x )
    
    return @fastmath 10^( log10(kdisc_H2H2_high) - log10(kdisc_H2H2_high/kdisc_H2H2_low) / (1 + nH2/ncr) )
end


# Some fitting coefficients (Glover et al. 2010)
const kdisc_H2e0::Float64 = 3.73e-9; const T_discH2e::Float64 = 99430.0
const p_discH2e::Float64  = 0.1121

""" The H₂-e⁻ collisional dissociation rate (cm³ s⁻¹) as a function 
    of temperature T (K). See Glover et al. (2010). """
@inline function kdiss_H2e( T::Float64 )
    T = clamp( T, 100.0, 1e5 )
    @fastmath kdisc_H2e0 * (T^p_discH2e) * exp( - T_discH2e/T )
end


# Some fitting coefficients (Glover et al. 2010/Savin et al. 2004):
const T_discH2HII::Float64   = 21237.15
const c0_kdiscH2HII::Float64 = - 3.3232183e-7;  const c1_kdiscH2HII::Float64 = 3.3735382e-7
const c2_kdiscH2HII::Float64 = - 1.4491368e-7;  const c3_kdiscH2HII::Float64 = 3.4172805e-8
const c4_kdiscH2HII::Float64 = - 4.7813720e-9;  const c5_kdiscH2HII::Float64 = 3.9731542e-10
const c6_kdiscH2HII::Float64 = - 1.8171411e-11; const c7_kdiscH2HII::Float64 = 3.5311932e-13

""" The H₂-H II collisional dissociation rate (cm³ s⁻¹) as a function 
    of temperature T (K). See Glover et al. (2010) (and original reference
    Savin et al. 2004, and its erratum where they note that "ln" should be 
    used instead of log10).
    
    Reaction: H₂ + H⁺ → H + H₂⁺ """
@inline function kdiss_H2HII( T::Float64 )
    T  = clamp( T, 100.0, 1e5 )
    x  = @fastmath log(T); x2 = x*x; x3 = x2*x; x4 = x3*x; x5 = x4*x 
    x6 = x5*x;   x7 = x6*x

    max( 0.0, ( c0_kdiscH2HII + c1_kdiscH2HII * x + c2_kdiscH2HII * x2
    + c3_kdiscH2HII * x3 + c4_kdiscH2HII * x4 + c5_kdiscH2HII * x5
    + c6_kdiscH2HII * x6 + c7_kdiscH2HII * x7 ) * @fastmath exp( - T_discH2HII/T ) )
end



###############################################################
#######                                                 #######
#######               D E U T E R I U M                 #######
#######               C H E M I S T R Y                 #######
#######                                                 #######
###############################################################

# The reactions below are some of the key reactions governing 
# deuterium chemistry (D, D⁺, HD). They are numbered as in
# McGreer & Bryan (2008), but many of them updated. Some reaction rates, 
# like recombinations with electrons, are the same as for hydrogen, 
# so not repeated here. 

const k51_D0::Float64 = 2.18e-9; const p_k51_D::Float64 = 3.55e-1

""" Rate (cm³ s⁻¹) for D⁺ + H → D + H⁺. See Faure+ (2023). """
@inline function k51_D( T::Float64 )
    @fastmath k51_D0 * (T^p_k51_D)
end


const T_k50_D::Float64 = 4.30e1

""" Rate (cm³ s⁻¹) for D + H⁺ → D⁺ + H. See Faure+ (2023). """
@inline function k50_D( T::Float64 )
    @fastmath k51_D(T) * exp( - T_k50_D / T )
end


const k52_D0::Float64 = 1.50e-9

""" Rate (cm³ s⁻¹) for D⁺ + H₂ → H⁺ + HD. See Faure+ (2023).
    The fitted reaction rate is just a constant in this case. """
@inline function k52_D( T::Float64 )
    k52_D0
end


const k53_D0::Float64 = 8.57e-10; const T_k53_D::Float64 = 4.05e2

""" Rate (cm³ s⁻¹) for H⁺ + HD → D⁺ + H₂. See Faure+ (2023). """
@inline function k53_D( T::Float64 )
    @fastmath k53_D0 * exp( - T_k53_D / T )
end


const c0_k54_D::Float64 = - 56.4737;  const c1_k54_D::Float64 = 5.88886
const c2_k54_D::Float64 = 7.19692;    const c3_k54_D::Float64 = 2.25069
const c4_k54_D::Float64 = - 2.16903;  const c5_k54_D::Float64 = 0.317887

const k54_D0_high::Float64 = 3.17e-10; const T_k54_D::Float64 = 5207.0

""" Rate (cm³ s⁻¹) for D + H₂ → H + HD. 
    See reaction 37 in Glover & Abel (2008). """
@inline function k54_D( T::Float64 )
    if T ≤ 2e3 
        T = clamp( T, 1.0, 2e3 )
        x = @fastmath log10(T); x2 = x*x; x3 = x2*x; x4 = x3*x; x5 = x4*x 

        k54_D = @fastmath 10^( c0_k54_D + c1_k54_D*x + c2_k54_D*x2 + c3_k54_D*x3 + c4_k54_D*x4 + c5_k54_D*x5 )
        return k54_D
    else
        k54_D = @fastmath k54_D0_high * exp( - T_k54_D / T )
        return k54_D
    end
end


const k55_D0::Float64  = 5.25e-11; const T_k55_D::Float64 = 4430.0
const c_k55_D::Float64 = 173900.0

""" Rate (cm³ s⁻¹) for H + HD → D + H₂. 
    See reaction 40 in Glover & Abel (2008). """
@inline function k55_D( T::Float64 )
    T     = clamp( T, 1e2, 1e5 )
    inv_T = @fastmath 1.0 / T
    k55_D = @fastmath k55_D0 * exp( - T_k55_D * inv_T  +  c_k55_D * inv_T * inv_T )
    return k55_D
end

###############################################################
#######                                                 #######
#######           C A R B O N  &  O X Y G E N           #######
#######               C H E M I S T R Y                 #######
#######                                                 #######
###############################################################

# Below are chemical rates relevant for carbon & oxygen chemistry, 
# i.e. C I, C II, C III, CO, CHₓ, O I, O II, O III, OHₓ. 


const u_numerator_CI::Float64 = 11.26 * eV / kB 

""" C I collisional ionization rate (cm³ s⁻¹):
    C + e⁻ → C⁺ + e⁻ + e⁻ 
    See Glover & Jappsen (2007). """
@inline function kCIe( T::Float64 )
    T    = clamp( T, 1.0, 1e5 )
    u    = @fastmath u_numerator_CI / T
    return @fastmath 6.85e-8 * (u^0.25) * exp(-u) / (0.193 + u)
end


const u_numerator_OI::Float64 = 13.6 * eV / kB 

""" O I collisional ionization rate (cm³ s⁻¹):
    O + e⁻ → O⁺ + e⁻ + e⁻ 
    See Glover & Jappsen (2007). """
@inline function kOIe( T::Float64 )
    T    = clamp( T, 1.0, 1e5 )
    u    = @fastmath u_numerator_OI / T
    return @fastmath 3.59e-8 * (u^0.34) * exp(-u) / (0.073 + u)
end


const kCIIe_rr0::Float64 = 2.995e-9; const T_CIIrec::Float64 = 49550.0

""" C II recombination rate (cm³ s⁻¹) in the gas phase:
    C⁺ + e⁻ → C + hν 
    See Gong et al. (2017). """
@inline function kCIIe_Raw( T::Float64 )
    T        = clamp( T, 1.0, 1e5 )
    inv_T    = @fastmath 1.0 / T
    # Radiative recombination rate:
    α        = @fastmath sqrt(T / 6.67e-3)
    β        = @fastmath sqrt(T / 1.9436e6)
    γ        = @fastmath 0.7849  +  0.1597 * exp( - T_CIIrec * inv_T )
    kCIIe_rr = @fastmath kCIIe_rr0 / ( α * ((1 + α)^(1-γ)) * ((1 + β)^(1+γ)) )
    # Dielectronic recombination rate:
    kCIIe_dr = @fastmath (T^(-3/2)) * ( 6.346e-9 * exp( - 12.17 * inv_T )
                                      + 9.793e-9 * exp( - 73.8 * inv_T )
                                      + 1.634e-6 * exp( - 15230.0 * inv_T ) )

    return @fastmath kCIIe_rr + kCIIe_dr
end

# Make fast LUT for kCIIe:
const kCIIe_LUT = InitializeChemistryRateLUT( kCIIe_Raw, 1.0, 1e5, 1024 )

""" C II recombination rate (cm³ s⁻¹) in the gas phase:
    C⁺ + e⁻ → C + hν 
    See Gong et al. (2017). """
@inline function kCIIe( log10T::Float64 )
    log10T = clamp( log10T, 0.0, 5.0 )
    return EvalChemistryRateLUT( log10T, kCIIe_LUT )
end

# ------------------------------------------------


""" O II recombination rate (cm³ s⁻¹) in the gas phase:
    O⁺ + e⁻ → O + hν 
    See Glover & Jappsen (2007). """
@inline function kOIIe( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    if T ≤ 400.0
        KOIIe = @fastmath 1.30e-10 * (T^(-0.64))
        return KOIIe
    else
        KOIIe = @fastmath 1.41e-10 * (T^(-0.66))  +  7.4e-4 * (T^(-1.5)) * exp(-175000.0/T)
        return KOIIe
    end 
end


# Coefficients for the radiative O III recombination rate:
const A_kOIIIe::Float64   = 2.096e-9 
const B_kOIIIe::Float64   = 0.7668 
const C_kOIIIe::Float64   = 0.1070
const T0_kOIIIe::Float64  = 1.602e-1; const inv_T0_kOIIIe::Float64 = 1 / T0_kOIIIe
const T1_kOIIIe::Float64  = 4.377e6;  const inv_T1_kOIIIe::Float64 = 1 / T1_kOIIIe
const T2_kOIIIe::Float64  = 1.392e5 

# Coefficients for the dielectronic O III recombination rate (note that c4 = 0 in the fit, 
# so not shown):
const kOIIIe_c1::Float64 = 4.223e-2;  const kOIIIe_c2::Float64 = 1.879e-2
const kOIIIe_c3::Float64 = 5.227e-02; 
const kOIIIe_c5::Float64 = 8.415e0;   const kOIIIe_c6::Float64 = 3.113e2
const kOIIIe_c7::Float64 = 6.300e0;   const kOIIIe_c8::Float64 = 2.885e1

const kOIIIe_E1::Float64 = 1.513e-2 * eV / kB 
const kOIIIe_E2::Float64 = 2.719e-2 * eV / kB 
const kOIIIe_E3::Float64 = 3.300e-1 * eV / kB 
const kOIIIe_E5::Float64 = 7.312e0  * eV / kB 
const kOIIIe_E6::Float64 = 1.954e1  * eV / kB 
const kOIIIe_E7::Float64 = 3.136e1  * eV / kB 
const kOIIIe_E8::Float64 = 3.429e1  * eV / kB 

""" O III radiative + dielectronic recombination rate (cm³ s⁻¹) in the gas phase:
    O⁺⁺ + e⁻ → O⁺ + hν
    See Badnell (2006) and Zatsarinny et al. (2004) for the radiative
    and dielectronic rates, respectively. """
@inline function kOIIIe_Raw( T::Float64 )

    T     = clamp( T, 1.0, 1e9 ) 
    inv_T = @fastmath 1.0 / T 

    ################################################################
    #######                                                  #######
    #######   R A D I A T I V E  R E C O M B I N A T I O N   #######
    #######                                                  #######
    ################################################################

    # Convenient factors:
    sqrt_T_T0   = @fastmath sqrt( T * inv_T0_kOIIIe )
    sqrt_T_T1   = @fastmath sqrt( T * inv_T1_kOIIIe )

    # For low-charge ions - including O III - Badnell uses a temperature-dependent 
    # B-factor:
    Btot_kOIIIe = @fastmath B_kOIIIe  +  C_kOIIIe * exp( - T2_kOIIIe * inv_T )

    # Put it all together:
    denom     = @fastmath sqrt_T_T0 * ((1 + sqrt_T_T0)^(1 - Btot_kOIIIe)) * ((1 + sqrt_T_T1)^(1 + Btot_kOIIIe))
    kOIIIe_RR = @fastmath A_kOIIIe / denom 

    ######################################################################
    #######                                                        #######
    #######   D I E L E C T R O N I C  R E C O M B I N A T I O N   #######
    #######                                                        #######
    ######################################################################

    T_eV      = @fastmath kB * T * inv_eV  # Temperature in eV 
    kOIIIe_DR = @fastmath 1e-11 * (T_eV^(-1.5)) * ( kOIIIe_c1 * exp( - kOIIIe_E1 * inv_T ) 
                                                  + kOIIIe_c2 * exp( - kOIIIe_E2 * inv_T )
                                                  + kOIIIe_c3 * exp( - kOIIIe_E3 * inv_T )
                                                  + kOIIIe_c5 * exp( - kOIIIe_E5 * inv_T )
                                                  + kOIIIe_c6 * exp( - kOIIIe_E6 * inv_T )
                                                  + kOIIIe_c7 * exp( - kOIIIe_E7 * inv_T )
                                                  + kOIIIe_c8 * exp( - kOIIIe_E8 * inv_T ) )

    return kOIIIe_RR + kOIIIe_DR  
end

# Make fast LUT for kOIIIe:
const kOIIIe_LUT = InitializeChemistryRateLUT( kOIIIe_Raw, 1.0, 1e9, 1024 )

""" O III radiative + dielectronic recombination rate (cm³ s⁻¹) in the gas phase:
    O⁺⁺ + e⁻ → O⁺ + hν
    See Badnell (2006) and Zatsarinny et al. (2004) for the radiative
    and dielectronic rates, respectively. """
@inline function kOIIIe( log10T::Float64 )
    log10T = clamp( log10T, 0.0, 9.0 )
    return EvalChemistryRateLUT( log10T, kOIIIe_LUT )
end

# ------------------------------------------------

# Coefficients for the radiative O IV recombination rate:
const A_kOIVe::Float64   = 2.501e-9
const B_kOIVe::Float64   = 0.7844
const C_kOIVe::Float64   = 0.0447
const T0_kOIVe::Float64  = 5.235e-1; const inv_T0_kOIVe::Float64 = 1 / T0_kOIVe
const T1_kOIVe::Float64  = 4.470e6;  const inv_T1_kOIVe::Float64 = 1 / T1_kOIVe
const T2_kOIVe::Float64  = 1.642e5

# Coefficients for the dielectronic O IV recombination rate (note that c4 and above are zero, 
# so not shown):
const kOIVe_c1::Float64 = 7.373e-5;  const kOIVe_c2::Float64 = 6.993e-3
const kOIVe_c3::Float64 = 3.848e-3; 

const kOIVe_E1::Float64 = 2.421e4  # In K 
const kOIVe_E2::Float64 = 2.156e5
const kOIVe_E3::Float64 = 5.280e5

""" O IV radiative + dielectronic recombination rate (cm³ s⁻¹) in the gas phase:
    O⁺⁺⁺ + e⁻ → O⁺⁺ + hν
    See Badnell (2006) and Altun et al. (2004) for the radiative
    and dielectronic rates, respectively. """
@inline function kOIVe_Raw( T::Float64 )

    T     = clamp( T, 1.0, 1e9 ) 
    inv_T = @fastmath 1.0 / T 

    ################################################################
    #######                                                  #######
    #######   R A D I A T I V E  R E C O M B I N A T I O N   #######
    #######                                                  #######
    ################################################################

    # Convenient factors:
    sqrt_T_T0   = @fastmath sqrt( T * inv_T0_kOIVe )
    sqrt_T_T1   = @fastmath sqrt( T * inv_T1_kOIVe )

    # For low-charge ions - including O IV - Badnell uses a temperature-dependent 
    # B-factor:
    Btot_kOIVe  = @fastmath B_kOIVe  +  C_kOIVe * exp( - T2_kOIVe * inv_T )

    # Put it all together:
    denom       = @fastmath sqrt_T_T0 * ((1 + sqrt_T_T0)^(1 - Btot_kOIVe)) * ((1 + sqrt_T_T1)^(1 + Btot_kOIVe))
    kOIVe_RR    = @fastmath A_kOIVe / denom 

    ######################################################################
    #######                                                        #######
    #######   D I E L E C T R O N I C  R E C O M B I N A T I O N   #######
    #######                                                        #######
    ######################################################################

    kOIVe_DR = @fastmath (T^(-1.5)) * ( kOIVe_c1 * exp( - kOIVe_E1 * inv_T ) 
                                      + kOIVe_c2 * exp( - kOIVe_E2 * inv_T )
                                      + kOIVe_c3 * exp( - kOIVe_E3 * inv_T ) )

    return kOIVe_RR + kOIVe_DR 
end


# Make fast LUT for kOIVe:
const kOIVe_LUT = InitializeChemistryRateLUT( kOIVe_Raw, 1.0, 1e9, 1024 )

""" O IV radiative + dielectronic recombination rate (cm³ s⁻¹) in the gas phase:
    O⁺⁺⁺ + e⁻ → O⁺⁺ + hν
    See Badnell (2006) and Altun et al. (2004) for the radiative
    and dielectronic rates, respectively. """
@inline function kOIVe( log10T::Float64 )
    log10T = clamp( log10T, 0.0, 9.0 )
    return EvalChemistryRateLUT( log10T, kOIVe_LUT )
end

# ------------------------------------------------


# Coefficients for the radiative C III recombination rate:
const A_kCIIIe::Float64   = 2.067e-9 
const B_kCIIIe::Float64   = 0.8012 
const C_kCIIIe::Float64   = 0.0427
const T0_kCIIIe::Float64  = 1.643e-1; const inv_T0_kCIIIe::Float64 = 1 / T0_kCIIIe
const T1_kCIIIe::Float64  = 2.172e6;  const inv_T1_kCIIIe::Float64 = 1 / T1_kCIIIe
const T2_kCIIIe::Float64  = 6.341e4 

# Coefficients for the dielectronic C III recombination rate:
const kCIIIe_c1::Float64 = 9.146e-6;  const kCIIIe_c2::Float64 = 5.045e-4
const kCIIIe_c3::Float64 = 3.742e-3

const kCIIIe_E1::Float64 = 6.485e3 # in K 
const kCIIIe_E2::Float64 = 1.052e5 # in K 
const kCIIIe_E3::Float64 = 1.458e5 # in K 

""" C III radiative + dielectronic recombination rate (cm³ s⁻¹) in the gas phase:
    C⁺⁺ + e⁻ → C⁺ + hν
    See Badnell (2006) and Colgan et al. (2003) for the radiative
    and dielectronic rates, respectively. """
@inline function kCIIIe_Raw( T::Float64 )

    T     = clamp( T, 1.0, 1e9 ) 
    inv_T = @fastmath 1.0 / T 

    ################################################################
    #######                                                  #######
    #######   R A D I A T I V E  R E C O M B I N A T I O N   #######
    #######                                                  #######
    ################################################################

    # Convenient factors:
    sqrt_T_T0   = @fastmath sqrt( T * inv_T0_kCIIIe )
    sqrt_T_T1   = @fastmath sqrt( T * inv_T1_kCIIIe )

    # For low-charge ions - including C III - Badnell uses a temperature-dependent 
    # B-factor:
    Btot_kCIIIe = @fastmath B_kCIIIe  +  C_kCIIIe * exp( - T2_kCIIIe * inv_T )

    # Put it all together:
    denom     = @fastmath sqrt_T_T0 * ((1 + sqrt_T_T0)^(1 - Btot_kCIIIe)) * ((1 + sqrt_T_T1)^(1 + Btot_kCIIIe))
    kCIIIe_RR = @fastmath A_kCIIIe / denom 

    ######################################################################
    #######                                                        #######
    #######   D I E L E C T R O N I C  R E C O M B I N A T I O N   #######
    #######                                                        #######
    ######################################################################

    kCIIIe_DR = @fastmath (T^(-1.5)) * ( kCIIIe_c1 * exp( - kCIIIe_E1 * inv_T ) 
                                       + kCIIIe_c2 * exp( - kCIIIe_E2 * inv_T )
                                       + kCIIIe_c3 * exp( - kCIIIe_E3 * inv_T ) )

    return kCIIIe_RR + kCIIIe_DR 
end

# Make fast LUT for kCIIIe:
const kCIIIe_LUT = InitializeChemistryRateLUT( kCIIIe_Raw, 1.0, 1e9, 1024 )

""" C III radiative + dielectronic recombination rate (cm³ s⁻¹) in the gas phase:
    C⁺⁺ + e⁻ → C⁺ + hν
    See Badnell (2006) and Colgan et al. (2003) for the radiative
    and dielectronic rates, respectively. """
@inline function kCIIIe( log10T::Float64 )
    log10T = clamp( log10T, 0.0, 9.0 )
    return EvalChemistryRateLUT( log10T, kCIIIe_LUT )
end

# ---------------------------------------------------------------


# Coefficients for fit of O II collisional ionization rate 
# (note that P = 1 for O II in Voronov's fit, so not shown here):
const A_kOIIe_ion::Float64  = 0.139e-7
const K_kOIIe_ion::Float64  = 0.22
const X_kOIIe_ion::Float64  = 0.212
const dE_kOIIe_ion::Float64 = 35.1 * eV / kB 

""" O II collisional ionization rate (cm³ s⁻¹):
    O⁺ + e⁻ → O⁺⁺ + e⁻ + e⁻ 
    See Voronov (1997). """
@inline function kOIIe_collion( T::Float64 )

    T         = clamp( T, 1.0, 1e9 )
    U         = @fastmath dE_kOIIe_ion / T 
    stuff     = @fastmath ( 1  +  sqrt(U) ) / ( X_kOIIe_ion  +  U )
    kOIIe_ion = @fastmath A_kOIIe_ion * stuff * (U^K_kOIIe_ion) * exp( - U )

    return kOIIe_ion
end


# Coefficients for fit of O III collisional ionization rate 
# (note that P = 1 for O III in Voronov's fit, so not shown here):
const A_kOIIIe_ion::Float64  = 0.931e-8
const K_kOIIIe_ion::Float64  = 0.27
const X_kOIIIe_ion::Float64  = 0.270
const dE_kOIIIe_ion::Float64 = 54.9 * eV / kB 

""" O III collisional ionization rate (cm³ s⁻¹):
    O⁺⁺ + e⁻ → O⁺⁺⁺ + e⁻ + e⁻ 
    See Voronov (1997). """
@inline function kOIIIe_collion( T::Float64 )

    T          = clamp( T, 1.0, 1e9 )
    U          = @fastmath dE_kOIIIe_ion / T 
    stuff      = @fastmath ( 1  +  sqrt(U) ) / ( X_kOIIIe_ion  +  U )
    kOIIIe_ion = @fastmath A_kOIIIe_ion * stuff * (U^K_kOIIIe_ion) * exp( - U )

    return kOIIIe_ion
end


# Coefficients for fit of C II collisional ionization rate 
# (note that P = 1 for C II in Voronov's fit, so not shown here):
const A_kCIIe_ion::Float64  = 0.186e-7
const K_kCIIe_ion::Float64  = 0.24
const X_kCIIe_ion::Float64  = 0.286
const dE_kCIIe_ion::Float64 = 24.4 * eV / kB 

""" C II collisional ionization rate (cm³ s⁻¹):
    C⁺ + e⁻ → C⁺⁺ + e⁻ + e⁻ 
    See Voronov (1997). """
@inline function kCIIe_collion( T::Float64 )

    T         = clamp( T, 1.0, 1e9 )
    U         = @fastmath dE_kCIIe_ion / T 
    stuff     = @fastmath ( 1  +  sqrt(U) ) / ( X_kCIIe_ion  +  U )
    kCIIe_ion = @fastmath A_kCIIe_ion * stuff * (U^K_kCIIe_ion) * exp( - U )

    return kCIIe_ion
end


# Fitting coefficients for the charge exchange rate of O III:
const kOIIIHI_a::Float64 = 1.04e-9
const kOIIIHI_b::Float64 = 0.27
const kOIIIHI_c::Float64 = 2.02
const kOIIIHI_d::Float64 = - 5.92

""" O III charge exchange rate (cm³ s⁻¹):
    O⁺⁺ + H → O⁺ + H⁺ 
    See Kingdon & Ferland (1996). """
@inline function kOIIIHI( T::Float64 )

    if T < 100.0 || T > 1e5
        # Outside temperature range where fit is valid, plus barely any H I 
        # above the upper limit, and barely any O III below the lower limit.
        return 0.0
    end

    T4 = @fastmath T * 1e-4  # T / 10⁴ K

    return @fastmath kOIIIHI_a * (T4^kOIIIHI_b) * ( 1.0  +  kOIIIHI_c * exp( kOIIIHI_d * T4 ) )
end


# Fitting coefficients for the charge exchange rate of O IV:
const kOIVHI_a::Float64 = 3.98e-9
const kOIVHI_b::Float64 = 0.26
const kOIVHI_c::Float64 = 0.56
const kOIVHI_d::Float64 = - 2.62

""" O IV charge exchange rate (cm³ s⁻¹):
    O⁺⁺⁺ + H → O⁺⁺ + H⁺ 
    See Kingdon & Ferland (1996). """
@inline function kOIVHI( T::Float64 )

    if T < 100.0 || T > 1e5
        # Outside temperature range where fit is valid, plus barely any H I 
        # above the upper limit, and barely any O IV below the lower limit.
        return 0.0
    end

    T4 = @fastmath T * 1e-4  # T / 10⁴ K

    return @fastmath kOIVHI_a * (T4^kOIVHI_b) * ( 1.0  +  kOIVHI_c * exp( kOIVHI_d * T4 ) )
end


# Fitting coefficients for the charge exchange rate of C III:
const kCIIIHI_a::Float64 = 1.67e-13
const kCIIIHI_b::Float64 = 2.79
const kCIIIHI_c::Float64 = 304.72
const kCIIIHI_d::Float64 = - 4.07

""" C III charge exchange rate (cm³ s⁻¹):
    C⁺⁺ + H → C⁺ + H⁺ 
    See Kingdon & Ferland (1996). """
@inline function kCIIIHI( T::Float64 )

    if T < 5e3 || T > 5e4
        # Outside temperature range where fit is valid, plus barely any H I 
        # above the upper limit, and barely any C III below the lower limit.
        return 0.0
    end

    T4 = @fastmath T * 1e-4  # T / 10⁴ K

    return @fastmath kCIIIHI_a * (T4^kCIIIHI_b) * ( 1.0  +  kCIIIHI_c * exp( kCIIIHI_d * T4 ) )
end


const c1_CIH3plus::Float64 = 3.40e-8; const c2_CIH3plus::Float64 = 6.97e-9
const c3_CIH3plus::Float64 = 1.31e-7; const c4_CIH3plus::Float64 = 1.51e-4
const T1_CIH3plus::Float64 = 7.62;    const T2_CIH3plus::Float64 = 1.38
const T3_CIH3plus::Float64 = 2.66e1;  const T4_CIH3plus::Float64 = 8.11e3
const KCIH3plus0::Float64  = 1.04e-9; const p_CIH3plus::Float64  = 0.00231

""" CHₓ formation rate/C I destruction rate (cm³ s⁻¹):
    C + H₃⁺ → CHₓ + H₂ 
    See Gong et al. (2017). 
    
    NOTE: Gong+ made an erratum in 2018 where they noted 
    that the exponent should be T^(-1.5) instead of T^(+1.5) as in their 2017 paper.
    This correction is included here. """
@inline function kCIH3plus_Raw( T::Float64 )
    T    = clamp( T, 1.0, 1e5 )
    T300 = @fastmath T / 300.0

    KCIH3plus = @fastmath (  KCIH3plus0 * (T300^p_CIH3plus) 
                             + (T^(-1.5)) * ( c1_CIH3plus * exp( - T1_CIH3plus / T ) 
                             + c2_CIH3plus * exp( - T2_CIH3plus / T )
                             + c3_CIH3plus * exp( - T3_CIH3plus / T )
                             + c4_CIH3plus * exp( - T4_CIH3plus / T )) )

    return KCIH3plus 
end

# Make fast LUT for kCIH3plus:
const kCIH3plus_LUT = InitializeChemistryRateLUT( kCIH3plus_Raw, 1.0, 1e5, 1024 )

""" CHₓ formation rate/C I destruction rate (cm³ s⁻¹):
    C + H₃⁺ → CHₓ + H₂ 
    See Gong et al. (2017). 
    
    NOTE: Gong+ made an erratum in 2018 where they noted 
    that the exponent should be T^(-1.5) instead of T^(+1.5) as in their 2017 paper.
    This correction is included here. """
@inline function kCIH3plus( log10T::Float64 )
    log10T = clamp( log10T, 0.0, 5.0 )
    return EvalChemistryRateLUT( log10T, kCIH3plus_LUT )
end

# ------------------------------------------------


const kCIIH2_10::Float64 = 2.31e-13; const T_CI::Float64 = 23.0

""" CHₓ formation rate/H₂ destruction rate (cm³ s⁻¹):
    C⁺ + H₂ → CHₓ + H 
    See Gong et al. (2017). """
@inline function kCIIH2_1( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    return @fastmath kCIIH2_10 * T^(-1.3) * exp( - T_CI / T ) 
end


const kCIIH2_20::Float64 = 0.99e-13

""" C I formation rate/H₂ destruction rate (cm³ s⁻¹):
    C⁺ + H₂ → C + H + H
    See Gong et al. (2017). """
@inline function kCIIH2_2( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    return @fastmath kCIIH2_20 * T^(-1.3) * exp( - T_CI / T ) 
end


""" O⁺ charge exchange rate (cm³ s⁻¹):
    H⁺ + O → O⁺ + H
    See Gong et al. (2017). """
@inline function kHIIOI( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    return @fastmath ( 1.1e-11 * T^0.517  +  4e-10 * T^0.00669 ) * exp( - 227.0/T )
end


""" O⁺ charge exchange rate (cm³ s⁻¹):
    O⁺ + H → H⁺ + O
    See Gong et al. (2017). """
@inline function kOIIHI( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    return @fastmath 4.99e-11 * T^0.405  +  7.5e-10 * T^(-0.458) 
end


""" OHₓ formation rate (cm³ s⁻¹):
    O + H₃⁺ → OHₓ + H₂
    See Gong et al. (2017). """
@inline function kOIH3plus( T::Float64,
                          nH2::Float64,
                           ne::Float64
                           )
    T = clamp( T, 1.0, 1e5 )
    A = 6e-10 * nH2
    B = 5.3e-6 * T^(-0.5) * ne
    r = A / ( A + B )  # A branching ratio

    if ne > 0.0 || nH2 > 0.0
        return 1.99e-9 * T^(-0.190) * r
    else
        # Return 0.0 to avoid undefined values:
        return 0.0
    end
end


""" OHₓ formation rate (cm³ s⁻¹):
    O⁺ + H₂ → OHₓ + H
    See Gong et al. (2017). """
@inline function kOIIH2_1( T::Float64,
                         nH2::Float64,
                          ne::Float64
                          )
    T = clamp( T, 1.0, 1e5 )
    A = 6e-10 * nH2
    B = 5.3e-6 * T^(-0.5) * ne
    r = A / ( A + B )  # A branching ratio

    if nH2 > 0.0
        return 1.6e-9 * r
    else
        # Return 0.0 to avoid undefined values:
        return 0.0
    end
end


""" O I formation rate via H₂ (cm³ s⁻¹):
    O⁺ + H₂ → O + H + H
    See Gong et al. (2017). """
@inline function kOIIH2_2( T::Float64,
                         nH2::Float64,
                          ne::Float64
                          )
    T = clamp( T, 1.0, 1e5 )
    A = 6e-10 * nH2
    B = 5.3e-6 * T^(-0.5) * ne
    r = A / ( A + B )  # A branching ratio
    if nH2 > 0.0
        return 1.6e-9 * max( (1 - r), 0.0 )
    else
        # If there is no H₂ the overall rate is zero:
        return 0.0
    end
end


""" CO formation rate (cm³ s⁻¹):
    CHₓ + O → CO + H
    See Gong et al. (2017). The rate is just
    a constant in this case. """
@inline function kCHxOI( T::Float64 )
    7.7e-11
end


""" CHₓ destruction rate (cm³ s⁻¹):
    CHₓ + H → C + H₂
    See Gong et al. (2017). """
@inline function kCHxHI( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    @fastmath 2.81e-11 * (T^0.26)
end


const kOHxCI_0::Float64 = 7.95e-10; const p_OHxCI::Float64 = - 0.339
const T_OHxCI::Float64  = 0.108 

""" CO formation rate (cm³ s⁻¹):
    OHₓ + C → CO + H
    See Gong et al. (2017). """
@inline function kOHxCI( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    @fastmath kOHxCI_0 * (T^p_OHxCI) * exp( T_OHxCI / T )
end


""" OHₓ destruction rate (cm³ s⁻¹):
    OHₓ + O → O + O + H
    See Gong et al. (2017). The fitted rate 
    is just a constant in this case. """
@inline function kOHxOI( T::Float64 )
    3.5e-11
end


""" H₃⁺ destruction rate (cm³ s⁻¹) from two reactions:
    H₃⁺ + e⁻ → H₂ + H
    H₃⁺ + e⁻ → H + H + H
    See Gong et al. (2017). """
@inline function kH3pluse( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    # Sum of the two rates given in Gong (4.54e-7 + 8.46e-7 = 13e-7):
    return @fastmath 13e-7 * T^(-0.52)
end


###############################################################
#######                                                 #######
#######                  S U L P H U R                  #######
#######                C H E M I S T R Y                #######
#######                                                 #######
###############################################################

# Reaction rates for the chemistry of sulphur (S) are listed below. 
# Cooling by S III is particularly important in metal-enriched 
# H II regions. 

# Fitting coefficients for the radiative recombination of S II:
const A_kSIIe::Float64  = 7.438e-11 
const B_kSIIe::Float64  = 0.6530 
const C_kSIIe::Float64  = 0.1730
const T0_kSIIe::Float64 = 3.1380e0; const inv_T0_kSIIe::Float64 = 1 / T0_kSIIe
const T1_kSIIe::Float64 = 1.944e6;  const inv_T1_kSIIe::Float64 = 1 / T1_kSIIe
const T2_kSIIe::Float64 = 1.675e4

# Fitting coefficients for the dielectronic recombination of S II:
const kSIIe_c1::Float64 = 7.300e-8; const kSIIe_c2::Float64 = 2.577e-7
const kSIIe_c3::Float64 = 4.961e-8; const kSIIe_c4::Float64 = 9.520e-7
const kSIIe_c5::Float64 = 9.586e-7; const kSIIe_c6::Float64 = 6.849e-4
const kSIIe_c7::Float64 = 6.539e-4

const kSIIe_E1::Float64 = 5.077e2;  const kSIIe_E2::Float64 = 6.007e2
const kSIIe_E3::Float64 = 2.342e3;  const kSIIe_E4::Float64 = 7.269e3
const kSIIe_E5::Float64 = 2.190e4;  const kSIIe_E6::Float64 = 1.483e5
const kSIIe_E7::Float64 = 1.906e5

""" S II radiative + dielectronic recombination rate (cm³ s⁻¹) in the gas phase:
    S⁺ + e⁻ → S + hν
    See Bleda+ (2022) for both the radiative and dielectronic rates. """
@inline function kSIIe_Raw( T::Float64 )

    T = clamp( T, 1.0, 1e9 ) 

    ################################################################
    #######                                                  #######
    #######   R A D I A T I V E  R E C O M B I N A T I O N   #######
    #######                                                  #######
    ################################################################

    # Convenient factors:
    sqrt_T_T0  = @fastmath sqrt( T * inv_T0_kSIIe )
    sqrt_T_T1  = @fastmath sqrt( T * inv_T1_kSIIe )

    # For low-charge ions - including S II - Badnell uses a temperature-dependent 
    # B-factor:
    inv_T      = @fastmath 1.0 / T 
    Btot_kSIIe = @fastmath B_kSIIe  +  C_kSIIe * exp( - T2_kSIIe * inv_T )

    # Put it all together:
    denom      = @fastmath sqrt_T_T0 * ((1 + sqrt_T_T0)^(1 - Btot_kSIIe)) * ((1 + sqrt_T_T1)^(1 + Btot_kSIIe))
    kSIIe_RR   = @fastmath A_kSIIe / denom 

    ######################################################################
    #######                                                        #######
    #######   D I E L E C T R O N I C  R E C O M B I N A T I O N   #######
    #######                                                        #######
    ######################################################################

    kSIIe_DR = @fastmath (T^(-1.5)) * ( kSIIe_c1 * exp( - kSIIe_E1 * inv_T ) 
                                      + kSIIe_c2 * exp( - kSIIe_E2 * inv_T )
                                      + kSIIe_c3 * exp( - kSIIe_E3 * inv_T )
                                      + kSIIe_c4 * exp( - kSIIe_E4 * inv_T )
                                      + kSIIe_c5 * exp( - kSIIe_E5 * inv_T )
                                      + kSIIe_c6 * exp( - kSIIe_E6 * inv_T )
                                      + kSIIe_c7 * exp( - kSIIe_E7 * inv_T ) )

    return @fastmath kSIIe_RR + kSIIe_DR
end

# Make fast LUT for kSIIe:
const kSIIe_LUT = InitializeChemistryRateLUT( kSIIe_Raw, 1.0, 1e9, 1024 )

""" S II radiative + dielectronic recombination rate (cm³ s⁻¹) in the gas phase:
    S⁺ + e⁻ → S + hν
    See Bleda+ (2022) for both the radiative and dielectronic rates. """
@inline function kSIIe( log10T::Float64 )
    log10T = clamp( log10T, 0.0, 9.0 )
    return EvalChemistryRateLUT( log10T, kSIIe_LUT )
end

# -------------------------------------------------------------------------


# Fitting coefficients for the radiative recombination of S III:
const A_kSIIIe::Float64  = 2.478e-11
const B_kSIIIe::Float64  = 0.4642
const C_kSIIIe::Float64  = 0.3351
const T0_kSIIIe::Float64 = 3.294e2;  const inv_T0_kSIIIe::Float64 = 1 / T0_kSIIIe
const T1_kSIIIe::Float64 = 2.166e7;  const inv_T1_kSIIIe::Float64 = 1 / T1_kSIIIe
const T2_kSIIIe::Float64 = 7.630e5

# Fitting coefficients for the dielectronic recombination of S III:
const kSIIIe_c1::Float64 = 3.040e-7; const kSIIIe_c2::Float64 = 4.393e-7
const kSIIIe_c3::Float64 = 1.609e-6; const kSIIIe_c4::Float64 = 4.980e-6
const kSIIIe_c5::Float64 = 3.457e-5; const kSIIIe_c6::Float64 = 8.617e-3 
const kSIIIe_c7::Float64 = 9.284e-4 

const kSIIIe_E1::Float64 = 5.016e1;  const kSIIIe_E2::Float64 = 3.266e2 
const kSIIIe_E3::Float64 = 3.102e3;  const kSIIIe_E4::Float64 = 1.210e4
const kSIIIe_E5::Float64 = 4.969e4;  const kSIIIe_E6::Float64 = 2.010e5 
const kSIIIe_E7::Float64 = 2.575e5

""" S III radiative + dielectronic recombination rate (cm³ s⁻¹) in the gas phase:
    S⁺⁺ + e⁻ → S⁺ + hν
    See Kaur+ (2018) for both the radiative and dielectronic rates. """
@inline function kSIIIe_Raw( T::Float64 )

    T = clamp( T, 1.0, 1e9 ) 

    ################################################################
    #######                                                  #######
    #######   R A D I A T I V E  R E C O M B I N A T I O N   #######
    #######                                                  #######
    ################################################################

    # Convenient factors:
    sqrt_T_T0   = @fastmath sqrt( T * inv_T0_kSIIIe )
    sqrt_T_T1   = @fastmath sqrt( T * inv_T1_kSIIIe )
    inv_T       = @fastmath 1.0 / T 

    Btot_kSIIIe = @fastmath B_kSIIIe  +  C_kSIIIe * exp( - T2_kSIIIe * inv_T )

    # Put it all together:
    denom       = @fastmath sqrt_T_T0 * ((1 + sqrt_T_T0)^(1 - Btot_kSIIIe)) * ((1 + sqrt_T_T1)^(1 + Btot_kSIIIe))
    kSIIIe_RR   = @fastmath A_kSIIIe / denom 

    ######################################################################
    #######                                                        #######
    #######   D I E L E C T R O N I C  R E C O M B I N A T I O N   #######
    #######                                                        #######
    ######################################################################

    kSIIIe_DR = @fastmath (T^(-1.5)) * ( kSIIIe_c1 * exp( - kSIIIe_E1 * inv_T ) 
                                       + kSIIIe_c2 * exp( - kSIIIe_E2 * inv_T )
                                       + kSIIIe_c3 * exp( - kSIIIe_E3 * inv_T )
                                       + kSIIIe_c4 * exp( - kSIIIe_E4 * inv_T )
                                       + kSIIIe_c5 * exp( - kSIIIe_E5 * inv_T )
                                       + kSIIIe_c6 * exp( - kSIIIe_E6 * inv_T )
                                       + kSIIIe_c7 * exp( - kSIIIe_E7 * inv_T ) )

    return @fastmath kSIIIe_RR + kSIIIe_DR
end

# Make fast LUT for kSIIIe:
const kSIIIe_LUT = InitializeChemistryRateLUT( kSIIIe_Raw, 1.0, 1e9, 1024 )

""" S III radiative + dielectronic recombination rate (cm³ s⁻¹) in the gas phase:
    S⁺⁺ + e⁻ → S⁺ + hν
    See Kaur+ (2018) for both the radiative and dielectronic rates. """
@inline function kSIIIe( log10T::Float64 )
    log10T = clamp( log10T, 0.0, 9.0 )
    return EvalChemistryRateLUT( log10T, kSIIIe_LUT )
end

# -------------------------------------------------------------------------


# Fitting coefficients for the radiative recombination of S IV:
const A_kSIVe::Float64  = 2.664e-10
const B_kSIVe::Float64  = 0.6896
const C_kSIVe::Float64  = 0.0840
const T0_kSIVe::Float64 = 2.107e1;  const inv_T0_kSIVe::Float64 = 1 / T0_kSIVe
const T1_kSIVe::Float64 = 2.028e7;  const inv_T1_kSIVe::Float64 = 1 / T1_kSIVe
const T2_kSIVe::Float64 = 6.752e4

# Fitting coefficients for the dielectronic recombination of S IV:
const kSIVe_c1::Float64 = 5.817e-7; const kSIVe_c2::Float64 = 1.391e-6
const kSIVe_c3::Float64 = 1.123e-5; const kSIVe_c4::Float64 = 1.521e-4
const kSIVe_c5::Float64 = 1.875e-3; const kSIVe_c6::Float64 = 2.097e-2

const kSIVe_E1::Float64 = 3.628e2;  const kSIVe_E2::Float64 = 1.058e3
const kSIVe_E3::Float64 = 7.160e3;  const kSIVe_E4::Float64 = 3.260e4
const kSIVe_E5::Float64 = 1.235e5;  const kSIVe_E6::Float64 = 2.070e5

""" S IV radiative + dielectronic recombination rate (cm³ s⁻¹) in the gas phase:
    S⁺⁺⁺ + e⁻ → S⁺⁺ + hν
    See Abdel-Naby+ (2012) for both the radiative and dielectronic rates. """
@inline function kSIVe_Raw( T::Float64 )

    T = clamp( T, 1.0, 1e9 ) 

    ################################################################
    #######                                                  #######
    #######   R A D I A T I V E  R E C O M B I N A T I O N   #######
    #######                                                  #######
    ################################################################

    # Convenient factors:
    sqrt_T_T0   = @fastmath sqrt( T * inv_T0_kSIVe )
    sqrt_T_T1   = @fastmath sqrt( T * inv_T1_kSIVe )
    inv_T       = @fastmath 1.0 / T 

    Btot_kSIVe  = @fastmath B_kSIVe  +  C_kSIVe * exp( - T2_kSIVe * inv_T )

    # Put it all together:
    denom       = @fastmath sqrt_T_T0 * ((1 + sqrt_T_T0)^(1 - Btot_kSIVe)) * ((1 + sqrt_T_T1)^(1 + Btot_kSIVe))
    kSIVe_RR    = @fastmath A_kSIVe / denom 

    ######################################################################
    #######                                                        #######
    #######   D I E L E C T R O N I C  R E C O M B I N A T I O N   #######
    #######                                                        #######
    ######################################################################

    kSIVe_DR  = @fastmath (T^(-1.5)) * ( kSIVe_c1 * exp( - kSIVe_E1 * inv_T ) 
                                       + kSIVe_c2 * exp( - kSIVe_E2 * inv_T )
                                       + kSIVe_c3 * exp( - kSIVe_E3 * inv_T )
                                       + kSIVe_c4 * exp( - kSIVe_E4 * inv_T )
                                       + kSIVe_c5 * exp( - kSIVe_E5 * inv_T )
                                       + kSIVe_c6 * exp( - kSIVe_E6 * inv_T ))

    return @fastmath kSIVe_RR + kSIVe_DR
end

# Make fast LUT for kSIVe:
const kSIVe_LUT = InitializeChemistryRateLUT( kSIVe_Raw, 1.0, 1e9, 1024 )

""" S IV radiative + dielectronic recombination rate (cm³ s⁻¹) in the gas phase:
    S⁺⁺⁺ + e⁻ → S⁺⁺ + hν
    See Abdel-Naby+ (2012) for both the radiative and dielectronic rates. """
@inline function kSIVe( log10T::Float64 )
    log10T = clamp( log10T, 0.0, 9.0 )
    return EvalChemistryRateLUT( log10T, kSIVe_LUT )
end

# -------------------------------------------------------------------------

# Coefficients for fit of S I collisional ionization rate 
# (note that P = 1 for S I in Voronov's fit, so not shown here):
const A_kSIe_ion::Float64  = 0.549e-7
const K_kSIe_ion::Float64  = 0.25
const X_kSIe_ion::Float64  = 0.100
const dE_kSIe_ion::Float64 = 10.4 * eV / kB 

""" S I collisional ionization rate (cm³ s⁻¹):
    S + e⁻ → S⁺ + e⁻ + e⁻ 
    See Voronov (1997). """
@inline function kSIe_collion_Raw( T::Float64 )

    T         = clamp( T, 1.0, 1e9 )
    U         = @fastmath dE_kSIe_ion / T 
    stuff     = @fastmath ( 1  +  sqrt(U) ) / ( X_kSIe_ion  +  U )
    kSIe_ion  = @fastmath A_kSIe_ion * stuff * (U^K_kSIe_ion) * exp( - U )

    return kSIe_ion
end

# Make fast LUT for kSIe_collion:
const kSIe_collion_LUT = InitializeChemistryRateLUT( kSIe_collion_Raw, 1.0, 1e9, 1024 )

""" S I collisional ionization rate (cm³ s⁻¹):
    S + e⁻ → S⁺ + e⁻ + e⁻ 
    See Voronov (1997). """
@inline function kSIe_collion( log10T::Float64 )
    log10T = clamp( log10T, 0.0, 9.0 )
    return EvalChemistryRateLUT( log10T, kSIe_collion_LUT )
end

# -------------------------------------------------------------------------

# Coefficients for fit of S II collisional ionization rate 
# (note that P = 1 for S II in Voronov's fit, so not shown here):
const A_kSIIe_ion::Float64  = 0.681e-7
const K_kSIIe_ion::Float64  = 0.21
const X_kSIIe_ion::Float64  = 0.693
const dE_kSIIe_ion::Float64 = 23.3 * eV / kB 

""" S II collisional ionization rate (cm³ s⁻¹):
    S⁺ + e⁻ → S⁺⁺ + e⁻ + e⁻ 
    See Voronov (1997). """
@inline function kSIIe_collion_Raw( T::Float64 )

    T         = clamp( T, 1.0, 1e9 )
    U         = @fastmath dE_kSIIe_ion / T 
    stuff     = @fastmath ( 1  +  sqrt(U) ) / ( X_kSIIe_ion  +  U )
    kSIIe_ion = @fastmath A_kSIIe_ion * stuff * (U^K_kSIIe_ion) * exp( - U )

    return kSIIe_ion
end

# Make fast LUT for kSIIe_collion:
const kSIIe_collion_LUT = InitializeChemistryRateLUT( kSIIe_collion_Raw, 1.0, 1e9, 1024 )

""" S II collisional ionization rate (cm³ s⁻¹):
    S⁺ + e⁻ → S⁺⁺ + e⁻ + e⁻ 
    See Voronov (1997). """
@inline function kSIIe_collion( log10T::Float64 )
    log10T = clamp( log10T, 0.0, 9.0 )
    return EvalChemistryRateLUT( log10T, kSIIe_collion_LUT )
end

# -------------------------------------------------------------------------

# Coefficients for fit of S III collisional ionization rate 
# (note that P = 1 for S III in Voronov's fit, so not shown here):
const A_kSIIIe_ion::Float64  = 0.214e-7
const K_kSIIIe_ion::Float64  = 0.24
const X_kSIIIe_ion::Float64  = 0.353
const dE_kSIIIe_ion::Float64 = 34.8 * eV / kB 

""" S III collisional ionization rate (cm³ s⁻¹):
    S⁺⁺ + e⁻ → S⁺⁺⁺ + e⁻ + e⁻ 
    See Voronov (1997). """
@inline function kSIIIe_collion_Raw( T::Float64 )

    T          = clamp( T, 1.0, 1e9 )
    U          = @fastmath dE_kSIIIe_ion / T 
    stuff      = @fastmath ( 1  +  sqrt(U) ) / ( X_kSIIIe_ion  +  U )
    kSIIIe_ion = @fastmath A_kSIIIe_ion * stuff * (U^K_kSIIIe_ion) * exp( - U )

    return kSIIIe_ion
end

# Make fast LUT for kSIIIe_collion:
const kSIIIe_collion_LUT = InitializeChemistryRateLUT( kSIIIe_collion_Raw, 1.0, 1e9, 1024 )

""" S III collisional ionization rate (cm³ s⁻¹):
    S⁺⁺ + e⁻ → S⁺⁺⁺ + e⁻ + e⁻ 
    See Voronov (1997). """
@inline function kSIIIe_collion( log10T::Float64 )
    log10T = clamp( log10T, 0.0, 9.0 )
    return EvalChemistryRateLUT( log10T, kSIIIe_collion_LUT )
end

# -------------------------------------------------------------------------


const kSIHII_a::Float64 = 1e-5 * 1e-9

""" S I (ionization) charge exchange rate (cm³ s⁻¹):
    S + H⁺ → S⁺ + H 
    See Kingdon & Ferland (1996). They provide only a constant rate.
    
    Note: The value for this rate is much lower than the more recent calculations
    of Zhao+ (2005). However, Zhao+ do not provide the updated reverse (charge recombination)
    rate, which may cause errors in the relative importance of these processes. 
    Furthermore, CLOUDY still uses the constant rate from Kingdon & Ferland.
    For simplicity, we therefore use the older Kingdon & Ferland results. """
@inline function kSIHII( T::Float64 )
    return kSIHII_a 
end


# Fitting coefficients for the charge exchange rate of S II:
const kSIIHI_a::Float64 = 3.82e-7 * 1e-9
const kSIIHI_b::Float64 = 11.10
const kSIIHI_c::Float64 = 2.57e4
const kSIIHI_d::Float64 = - 8.22

""" S II charge exchange rate (cm³ s⁻¹):
    S⁺ + H → S + H⁺ 
    See Kingdon & Ferland (1996). 
    
    Note: The authors write that this rate 
    estimate is very uncertain. The reverse 
    rate was improved upon significantly by Zhao+ (2005). """
@inline function kSIIHI( T::Float64 )

    T4 = @fastmath T * 1e-4  # T / 10⁴ K

    return @fastmath kSIIHI_a * (T4^kSIIHI_b) * ( 1.0  +  kSIIHI_c * exp( kSIIHI_d * T4 ) )
end


# Fitting coefficients for the charge exchange rate of S III:
const kSIIIHI_a::Float64 = 1e-5 * 1e-9

""" S III charge exchange rate (cm³ s⁻¹):
    S⁺⁺ + H → S⁺ + H⁺ 
    See Kingdon & Ferland (1996). For this rate, 
    the authors only report a constant value."""
@inline function kSIIIHI( T::Float64 )
    return kSIIIHI_a
end


# Fitting coefficients for the charge exchange rate of S IV:
const kSIVHI_a::Float64 = 2.29e-9
const kSIVHI_b::Float64 = 4.02e-2
const kSIVHI_c::Float64 = 1.59
const kSIVHI_d::Float64 = - 6.06

""" S IV charge exchange rate (cm³ s⁻¹):
    S⁺⁺⁺ + H → S⁺⁺ + H⁺ 
    See Kingdon & Ferland (1996). """
@inline function kSIVHI( T::Float64 )

    T4 = @fastmath T * 1e-4  # T / 10⁴ K

    return @fastmath kSIVHI_a * (T4^kSIVHI_b) * ( 1.0  +  kSIVHI_c * exp( kSIVHI_d * T4 ) )
end


###############################################################
#######                                                 #######
#######               B E N C H M A R K S               #######
#######                                                 #######
###############################################################

# Below is a benchmark checking performance and accuracy of 
# LUTs for certain chemistry rates.


""" This function benchmarks accuracy and performance of the 
    log-linear interpolations of chemistry rates. """
function BenchmarkChemistryRateLUTs( N_accuracy::Int     = 10000,
                                     N_bench::Int        = 100,
                                     rel_floor::Float64  = 1e-250 )

    println()
    println("==============================================================")
    println("        CHEMISTRY RATE LUT BENCHMARKS")
    println("==============================================================")

    ###############################################################
    #######                                                 #######
    #######              C A R B O N  R A T E S             #######
    #######                                                 #######
    ###############################################################

    carbon_rates = (
        # Add/remove entries depending on which rates you have converted.
        # Format: (name, raw_function, LUT_function, T_min, T_max)

        ("kCIIe",        kCIIe_Raw,      kCIIe,      1.0, 1e5),
        ("kCIIIe",       kCIIIe_Raw,     kCIIIe,     1.0, 1e9),
        ("kCIH3plus",    kCIH3plus_Raw,  kCIH3plus,  1.0, 1e5),

        # Uncomment if/when these also have _Raw + LUT versions:
        # ("kCIIIHI",        kCIIIHI_Raw,        kCIIIHI,        1.0, 1e9),
        # ("kCIIH2_1",       kCIIH2_1_Raw,       kCIIH2_1,       1.0, 1e9),
        # ("kCIIH2_2",       kCIIH2_2_Raw,       kCIIH2_2,       1.0, 1e9),
    )

    ###############################################################
    #######                                                 #######
    #######              O X Y G E N  R A T E S             #######
    #######                                                 #######
    ###############################################################

    oxygen_rates = (
        # Add/remove entries depending on which rates you have converted.
        # Format: (name, raw_function, LUT_function, T_min, T_max)

        ("kOIIIe",       kOIIIe_Raw,     kOIIIe,     1.0, 1e9),
        ("kOIVe",        kOIVe_Raw,      kOIVe,      1.0, 1e9),
    )

    ###############################################################
    #######                                                 #######
    #######              S U L P H U R  R A T E S           #######
    #######                                                 #######
    ###############################################################

    sulphur_rates = (
        ("kSIIe",           kSIIe_Raw,           kSIIe,           1.0, 1e9),
        ("kSIIIe",          kSIIIe_Raw,          kSIIIe,          1.0, 1e9),
        ("kSIVe",           kSIVe_Raw,           kSIVe,           1.0, 1e9),
        ("kSIe_collion",    kSIe_collion_Raw,    kSIe_collion,    1.0, 1e9),
        ("kSIIe_collion",   kSIIe_collion_Raw,   kSIIe_collion,   1.0, 1e9),
        ("kSIIIe_collion",  kSIIIe_collion_Raw,  kSIIIe_collion,  1.0, 1e9),
    )

    ###############################################################
    #######                                                 #######
    #######                B E N C H M A R K                #######
    #######                                                 #######
    ###############################################################

    rate_sections = (
        ("Carbon rates",     carbon_rates),
        ("Oxygen rates",     oxygen_rates),
        ("Sulphur rates",    sulphur_rates),
    )

    for (section_name, rates) in rate_sections

        println()
        println(section_name)
        println("--------------------------------------------------------------")

        for (name, rate_raw, rate_fast, T_min, T_max) in rates

            ###############################################################
            #######            A C C U R A C Y  C H E C K           #######
            ###############################################################

            max_rel_diff = 0.0
            T_worst      = T_min

            log10T_min   = log10(T_min)
            log10T_max   = log10(T_max)
            dlog10T      = (log10T_max - log10T_min) / (N_accuracy - 1)

            @inbounds for i in 1:N_accuracy

                log10T = log10T_min + (i - 1) * dlog10T
                T      = 10.0^log10T

                k_raw  = rate_raw(T)

                # Fast LUT rates take log10T, not T.
                k_fast = rate_fast(log10T)

                # Ignore relative errors where the raw rate is essentially zero.
                if abs(k_raw) > rel_floor

                    rel_diff = abs(k_fast - k_raw) / abs(k_raw)

                    if rel_diff > max_rel_diff
                        max_rel_diff = rel_diff
                        T_worst      = T
                    end
                end
            end

            ###############################################################
            #######        P E R F O R M A N C E  C H E C K         #######
            ###############################################################

            T_bench      = Vector{Float64}(undef, N_bench)
            log10T_bench = Vector{Float64}(undef, N_bench)

            dlog10T_bench = (log10T_max - log10T_min) / (N_bench - 1)

            @inbounds for i in 1:N_bench
                log10T_bench[i] = log10T_min + (i - 1) * dlog10T_bench
                T_bench[i]      = 10.0^log10T_bench[i]
            end

            # Raw rate benchmark. Uses T.
            t_raw = @belapsed begin
                s = 0.0
                @inbounds for T in $T_bench
                    s += $rate_raw(T)
                end
                s
            end

            # LUT benchmark. Uses precomputed log10T, so log10(T) is NOT counted.
            t_fast = @belapsed begin
                s = 0.0
                @inbounds for log10T in $log10T_bench
                    s += $rate_fast(log10T)
                end
                s
            end

            raw_ns_per_call  = 1e9 * t_raw  / N_bench
            fast_ns_per_call = 1e9 * t_fast / N_bench
            speedup          = raw_ns_per_call / fast_ns_per_call

            println()
            println(name)
            println("  Max rel. diff.       = ", max_rel_diff)
            println("  T_worst              = ", T_worst, " K")
            println("  Raw                  = ", raw_ns_per_call,  " ns/call")
            println("  LUT, log10T given    = ", fast_ns_per_call, " ns/call")
            println("  Speed-up             = ", speedup, "×")
        end
    end

    println()
    println("==============================================================")

    return nothing
end

# Uncomment if you want to benchmark LUTs:
#BenchmarkChemistryRateLUTs()