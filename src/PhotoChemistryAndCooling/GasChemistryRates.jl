
""" FILE:       GasChemistryRates.jl

    PURPOSE:    Contains everything related to gas photochemistry,
                involving e⁻, H I, H II, H₂, D I, D II, HD, He I, He II, He III,
                C I, C II, CO, O I, O II, and O III.

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

"""

using .ConstantsAndFunctions  # Needed constants 

export krecHII_1, krecHII, krecHeII_1, krecHeII, krecHeIII_1, krecHeIII, 
       kcollion_HI, kcollion_HeI, kcollion_HeII,
       kdust_H2, kGP_H2, k3B_H2, kdiss_H2HI, kdiss_H2H2, kdiss_H2e, kdiss_H2HII,
       k50_D, k51_D, k52_D, k53_D, k54_D, k55_D, kCIe, kOIe, kCIIe, kOIIe, kCIIH2_1,
       kCIIH2_2, kHIIOI, kOIIHI, kOIIH2_1, kOIIH2_2, kCHxOI, kCHxHI, kOHxCI, kOHxOI,
       kCIH3plus, kOIH3plus, kH3pluse


""" The H II Case A recombination rate (cm³ s⁻¹) as a function of temperature T (K).
    See ref. (1). """       
@inline function krecHII_A( T::Float64 )
    λHI       = 315614.0 / clamp( T, 1.0, 1e9 ) 
    krecHII_A = 1.269e-13 * (λHI^1.503) / ( 1 + (λHI/0.522)^0.47 )^1.923
    return krecHII_A
end


""" The H II Case B recombination rate (cm³ s⁻¹) as a function of temperature T (K).
    See ref. (1). """       
@inline function krecHII_B( T::Float64 )
    λHI       = 315614.0 / clamp( T, 1.0, 1e9 ) 
    krecHII_B = 2.753e-14 * (λHI^1.5) / ( 1 + (λHI/2.74)^0.407 )^2.242
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
    λHeI       = 570670.0 / clamp( T, 1.0, 1e9 ) 
    krecHeII_A = 3e-14 * (λHeI^0.654)
    return krecHeII_A
end


""" The He II Case B recombination rate (cm³ s⁻¹) as a function of temperature T (K).
    See ref. (1). """       
@inline function krecHeII_B( T::Float64 )
    λHeI       = 570670.0 / clamp( T, 1.0, 1e9 ) 
    krecHeII_B = 1.26e-14 * (λHeI^0.75)
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
    λHeII       = 1263030.0 / clamp( T, 1.0, 1e9 )  
    krecHeIII_A = 2.538e-13 * (λHeII^1.503) / (1 + (λHeII/0.522)^0.47)^1.923
    return krecHeIII_A
end


""" The He III Case B recombination rate (cm³ s⁻¹) as a function of temperature T (K).
    See ref. (1). """       
@inline function krecHeIII_B( T::Float64 )
    λHeII       = 1263030.0 / clamp( T, 1.0, 1e9 )  
    krecHeIII_B = 5.506e-14 * (λHeII^1.5) / (1 + (λHeII/2.74)^0.407 )^2.242 
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


const kcollion_HI0 = 5.85e-11; const T_collionHI = 157809.1

""" The H I collisional ionization rate (cm³ s⁻¹) by electrons,
    as a function of temperature T (K). See ref. (1). """
@inline function kcollion_HI( T::Float64 )
    kcollion_HI0 * sqrt(T) * exp( - T_collionHI / T ) / ( 1 + sqrt(T/1e5) )
end


const kcollion_HeI0 = 2.38e-11; const T_collionHeI = 285335.4 

""" The He I collisional ionization rate (cm³ s⁻¹) by electrons,
    as a function of temperature T (K). See ref. (1). """
@inline function kcollion_HeI( T::Float64 )
    kcollion_HeI0 * sqrt(T) * exp( - T_collionHeI / T ) / ( 1 + sqrt(T/1e5) )
end


const kcollion_HeII0 = 5.68e-12; const T_collionHeII = 631515.0

""" The He II collisional ionization rate (cm³ s⁻¹) by electrons,
    as a function of temperature T (K). See ref. (1). """
@inline function kcollion_HeII( T::Float64 )
    kcollion_HeII0 * sqrt(T) * exp( - T_collionHeII / T ) / ( 1 + sqrt(T/1e5) )
end


###############################################################
#######                                                 #######
#######        M O L E C U L A R  H Y D R O G E N       #######
#######               C H E M I S T R Y                 #######
#######                                                 #######
###############################################################

const Tdust_crit_H2::Float64 = 600.0

""" The H₂ formation efficiency (assumed equal to sticking coefficient),
    as a function of the gas temperature T (K) and dust temperature Td (K). 
    From Hollenbach & McKee (1979). """
@inline function S_H( T::Float64, Td::Float64 )

    # Formation efficiency:
    S_H     =  1 / ( ( 1  +  0.04 * (T + Td)^0.5  +  0.002 * T  +  8e-6 * (T^2) ) 
                   * ( 1  +  1e4 * exp( - Tdust_crit_H2 / Td ) ) )

    return S_H
end


const inv_DGnorm_kdust_H2::Float64 = 100.0 # Dust-to-gas ratio is normalized by 0.01 in the rate

""" The H₂ formation rate on dust (cm³ s⁻¹) as a function of
    temperature T (K). See e.g. Hollenbach & McKee (1979) 
    and Hopkins et al. (2023). """
function kdust_H2( T::Float64, 
                  DG::Float64,
                  Td::Float64 = 30.0
                  )

    # Clamp temperature (just for numerical reasons):
    T        = clamp( T, 0.0, 1e9 )  

    # H₂ formation rate on dust :
    kdust_H2 = 3e-18 * (DG * inv_DGnorm_kdust_H2) * sqrt(T) * S_H( T, Td )

    return kdust_H2
end


###### H + e⁻ → H⁻ + hν #######

const k1_0 = 1.4e-18
const T_k1 = 16200.0

""" H + e⁻ → H⁻ + hν rate (cm³ s⁻¹; from Galli & Palla 1998). """
@inline function k1( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    return k1_0 * (T^0.928) * exp( - T/T_k1 )
end


###### H⁻ + H → H₂ + e⁻ ########

# Fitting constants from Kreckel et al. (2010):
const c1_k2  = 1.35e-9;   const c2_k2  = 9.8493e-2;  const c3_k2  = 3.2852e-1
const c4_k2  = 5.5610e-1; const c5_k2  = 2.7710e-7;  const c6_k2  = 2.1826
const c7_k2  = 6.1910e-3; const c8_k2  = 1.0461;     const c9_k2  = 8.9712e-11
const c10_k2 = 3.0424;    const c11_k2 = 3.2576e-14; const c12_k2 = 3.7741

""" H⁻ + H → H₂ + e⁻ rate (cm³ s⁻¹; from Kreckel et al. 2010). """
@inline function k2( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    return c1_k2 * (T^c2_k2 + c3_k2 * T^c4_k2 + c5_k2 * T^c6_k2 ) / (
                    1 + c7_k2 * T^c8_k2 + c9_k2 * T^c10_k2 + c11_k2 * T^c12_k2 )
end


###### H⁻ + H⁺ → H + H ######

const k5_0 = 2.4e-6
const T_k5 = 2e4 

""" H⁻ + H⁺ → H + H rate (cm³ s⁻¹; from Glover et al. 2010, which 
    is updated relative to Glover & Jappsen 2007).  """
@inline function k5( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    return k5_0 * (T^(-0.5)) * (1.0 + T/T_k5)
end


###### H⁻ + e⁻ → H + e⁻ + e⁻ ######

const c0_k15 = - 1.801849334e1; const c1_k15 = 2.36085220e0
const c2_k15 = - 2.82744300e-1; const c3_k15 = 1.62331664e-2
const c4_k15 = - 3.36501203e-2; const c5_k15 = 1.17832978e-2
const c6_k15 = - 1.65619470e-3; const c7_k15 = 1.06827520e-4
const c8_k15 = - 2.63128581e-6

""" H⁻ + e⁻ → H + e⁻ + e⁻ rate (cm³ s⁻¹). Rate/fit is taken from 
    Glover & Jappsen (2007)/Glover et al. (2010) (they are the same 
    in both references).  """
@inline function k15( T::Float64 )
    # Convenient definitions: 
    T  = clamp( T, 1.0, 1e5 )
    x  = log( T / (eV/kB) ); x2 = x^2; x3 = x2*x; x4 = x3*x
    x5 = x4*x; x6 = x5*x; x7 = x6*x; x8 = x7*x 

    # Rate:
    k15 = exp( c0_k15 + c1_k15*x + c2_k15*x2 + c3_k15*x3 + c4_k15*x4
             + c5_k15*x5 + c6_k15*x6 + c7_k15*x7 + c8_k15*x8 )
    
    return k15
end


###### H⁻ + H → H + H + e⁻ ######

const k16_0  = 2.5634e-9
const c0_k16 = - 2.0372609e1;  const c1_k16 = 1.13944933e0
const c2_k16 = - 1.4210135e-1; const c3_k16 = 8.4644554e-3
const c4_k16 = - 1.4327641e-3; const c5_k16 = 2.0122503e-4
const c6_k16 = 8.6639632e-5;   const c7_k16 = - 2.5850097e-5
const c8_k16 = 2.4555012e-6;   const c9_k16 = - 8.0683825e-8

""" H⁻ + H → H + H + e⁻ rate (cm³ s⁻¹). Rate/fit is taken from 
    Glover & Jappsen (2007)/Glover et al. (2010) (they are the same 
    in both references).  """
function k16( T::Float64 )
    TeV = clamp( T, 1.0, 1e5 ) / (eV/kB)
    if TeV ≤ 0.1
        return k16_0 * (TeV^1.78186)
    else
        # Convenient definitions:
        x = log( TeV ); x2 = x^2; x3 = x2*x; x4 = x3*x
        x5 = x4*x; x6 = x5*x; x7 = x6*x; x8 = x7*x; x9 = x8*x 

        # Rate:
        k16 = exp( c0_k16 + c1_k16*x + c2_k16*x2 + c3_k16*x3 + c4_k16*x4
                   + c5_k16*x5 + c6_k16*x6 + c7_k16*x7 + c8_k16*x8 + c9_k16*x9 )

        return k16
    end
end


###### H⁻ + H⁺ → H₂⁺ + e⁻ ######

const k17_0low = 6.9e-9; const k17_0high = 9.6e-7
const p_k17low = - 0.35; const p_k17high = - 0.90

""" H⁻ + H⁺ → H₂⁺ + e⁻ rate (cm³ s⁻¹). Rate/fit is taken from 
    Glover & Jappsen (2007)/Glover et al. (2010) (they are the same 
    in both references).  """
function k17( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    if T ≤ 8e3
        return k17_0low * T^p_k17low
    else
        return k17_0high * T^p_k17high
    end
end


###### Gas phase H₂ formation rate ######

const Γpd_CMB10 = 0.11;   const Γpd_CMB20 = 8e-8
const cpd_CMB1  = 2.13;   const cpd_CMB2  = 1.3 
const Tpd_CMB1  = 8823.0; const Tpd_CMB2  = 2300.0

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
    TCMB    =   TCMB0 * (1 + z)
    Γpd_CMB = ( Γpd_CMB10 * (TCMB^cpd_CMB1) * exp( - Tpd_CMB1 / TCMB )
            +   Γpd_CMB20 * (TCMB^cpd_CMB2) * exp( - Tpd_CMB2 / TCMB ) )
    # Total photodetachment rate from non-CMB photons + CMB photons:
    Γpd_tot =   Γpd +  Γpd_CMB

    # Full rate coefficient, assuming equilibrium abundance of H⁻:
    kGP_H2  = ( K2 * K1 * nHI ) / ( (K2 + K16) * nHI  +  (K5 + K17) * nHII  +  K15 * ne  +  Γpd_tot )

    return kGP_H2
end


""" The 3-body H₂ formation rate (cm⁶ s⁻¹) as a function of temperature T (K).
    See ref. (2) (taken from Forrey 2013). """
@inline function k3B_H2( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    return 6e-32 * T^(-0.25) + 2e-31 * T^(-0.5)
end


###### Collisional destruction of H₂ ########

# Some fitting coefficients (Glover et al. 2010)
const kdisc_H2HI_low0  = 6.67e-12; const T_discH2HI_low  = 63590.0 
const kdisc_H2HI_high0 = 3.52e-9;  const T_discH2HI_high = 43900.0

""" The H₂-H I collisional dissociation rate (cm³ s⁻¹) as a function 
    of temperature T (K) and HI number density nHI (cm⁻³).
    See Glover et al. (2010). """
function kdiss_H2HI( T::Float64, 
                   nHI::Float64
                    )
    T = clamp( T, 100.0, 1e5 )

    # Low & high-density limits of the reaction rate:
    kdisc_H2HI_low  = kdisc_H2HI_low0 * sqrt(T) * exp( - (1.0 + T_discH2HI_low/T) )
    kdisc_H2HI_high = kdisc_H2HI_high0 * exp( - T_discH2HI_high/T )

    # Critical density for transition from low-density to high-density limits
    # (note that Glover et al. 2010 has "3.0", but in the reference cited (Lepp & Shull 1983)
    # it is actually "4.0", so I have corrected this):
    x   = log10(T/1e4)
    ncr = 10^( 4.0 - 0.416 * x - 0.327 * x^2 )

    return 10^( log10(kdisc_H2HI_high) - log10(kdisc_H2HI_high/kdisc_H2HI_low) / (1 + nHI/ncr) )
end


# Some fitting coefficients (Glover et al. 2010)
const kdisc_H2H2_low0  = 5.996e-30; const T_discH2H2_low  = 54657.0
const kdisc_H2H2_high0 = 1.3e-9;    const T_discH2H2_high = 53300.0

""" The H₂-H₂ collisional dissociation rate (cm³ s⁻¹) as a function 
    of temperature T (K) and H₂ number density nHI (cm⁻³).
    See Glover et al. (2010). """
function kdiss_H2H2( T::Float64, 
                   nH2::Float64
                    )
    T = clamp( T, 100.0, 1e5 )
    
    # Low & high-density limits of the reaction rate:
    kdisc_H2H2_low  = kdisc_H2H2_low0 * (T^4.1881) * exp( - T_discH2H2_low/T ) / (
                      1.0 + 6.761e-6 * T   )^5.6881

    kdisc_H2H2_high = kdisc_H2H2_high0 * exp( - T_discH2H2_high/T )

    # Critical density for transition from low-density to high-density limits:
    x   = log10(T/1e4)
    ncr = 10^( 4.845 - 1.3 * x + 1.62 * x^2 )
    
    return 10^( log10(kdisc_H2H2_high) - log10(kdisc_H2H2_high/kdisc_H2H2_low) / (1 + nH2/ncr) )
end


# Some fitting coefficients (Glover et al. 2010)
const kdisc_H2e0 = 3.73e-9; const T_discH2e = 99430.0
const p_discH2e  = 0.1121

""" The H₂-e⁻ collisional dissociation rate (cm³ s⁻¹) as a function 
    of temperature T (K). See Glover et al. (2010). """
@inline function kdiss_H2e( T::Float64 )
    T = clamp( T, 100.0, 1e5 )
    kdisc_H2e0 * (T^p_discH2e) * exp( - T_discH2e/T )
end


# Some fitting coefficients (Glover et al. 2010/Savin et al. 2004):
const T_discH2HII   = 21237.15
const c0_kdiscH2HII = - 3.3232183e-7;  const c1_kdiscH2HII = 3.3735382e-7
const c2_kdiscH2HII = - 1.4491368e-7;  const c3_kdiscH2HII = 3.4172805e-8
const c4_kdiscH2HII = - 4.7813720e-9;  const c5_kdiscH2HII = 3.9731542e-10
const c6_kdiscH2HII = - 1.8171411e-11; const c7_kdiscH2HII = 3.5311932e-13

""" The H₂-H II collisional dissociation rate (cm³ s⁻¹) as a function 
    of temperature T (K). See Glover et al. (2010) (and original reference
    Savin et al. 2004, and its erratum where they note that "ln" should be 
    used instead of log10).
    
    Reaction: H₂ + H⁺ → H + H₂⁺ """
@inline function kdiss_H2HII( T::Float64 )
    T  = clamp( T, 100.0, 1e5 )
    x  = log(T); x2 = x^2; x3 = x2*x; x4 = x3*x; x5 = x4*x 
    x6 = x5*x;   x7 = x6*x

    max( 0.0, ( c0_kdiscH2HII + c1_kdiscH2HII * x + c2_kdiscH2HII * x2
    + c3_kdiscH2HII * x3 + c4_kdiscH2HII * x4 + c5_kdiscH2HII * x5
    + c6_kdiscH2HII * x6 + c7_kdiscH2HII * x7 ) * exp( - T_discH2HII/T ) )
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

const k51_D0 = 2.18e-9; const p_k51_D = 3.55e-1

""" Rate (cm³ s⁻¹) for D⁺ + H → D + H⁺. See Faure+ (2023). """
@inline function k51_D( T::Float64 )
    k51_D0 * (T^p_k51_D)
end


const T_k50_D = 4.30e1

""" Rate (cm³ s⁻¹) for D + H⁺ → D⁺ + H. See Faure+ (2023). """
@inline function k50_D( T::Float64 )
    k51_D(T) * exp( - T_k50_D / T )
end


const k52_D0 = 1.50e-9

""" Rate (cm³ s⁻¹) for D⁺ + H₂ → H⁺ + HD. See Faure+ (2023).
    The fitted reaction rate is just a constant in this case. """
@inline function k52_D( T::Float64 )
    k52_D0
end


const k53_D0 = 8.57e-10; const T_k53_D = 4.05e2

""" Rate (cm³ s⁻¹) for H⁺ + HD → D⁺ + H₂. See Faure+ (2023). """
@inline function k53_D( T::Float64 )
    k53_D0 * exp( - T_k53_D / T )
end


const c0_k54_D = - 56.4737; const c1_k54_D = 5.88886
const c2_k54_D = 7.19692;   const c3_k54_D = 2.25069
const c4_k54_D = - 2.16903; const c5_k54_D = 0.317887

const k54_D0_high = 3.17e-10; const T_k54_D = 5207.0

""" Rate (cm³ s⁻¹) for D + H₂ → H + HD. 
    See reaction 37 in Glover & Abel (2008). """
@inline function k54_D( T::Float64 )
    if T ≤ 2e3 
        T = clamp( T, 1.0, 2e3 )
        x = log10(T); x2 = x^2; x3 = x2*x; x4 = x3*x; x5 = x4*x 

        k54_D = 10^( c0_k54_D + c1_k54_D*x + c2_k54_D*x2 + c3_k54_D*x3 + c4_k54_D*x4 + c5_k54_D*x5 )
        return k54_D
    else
        k54_D = k54_D0_high * exp( - T_k54_D / T )
        return k54_D
    end
end


const k55_D0  = 5.25e-11; const T_k55_D = 4430.0
const c_k55_D = 173900.0

""" Rate (cm³ s⁻¹) for H + HD → D + H₂. 
    See reaction 40 in Glover & Abel (2008). """
@inline function k55_D( T::Float64 )
    T     = clamp( T, 1e2, 1e5 )
    k55_D = k55_D0 * exp( - T_k55_D / T  +  c_k55_D / (T^2) )
    return k55_D
end

###############################################################
#######                                                 #######
#######           C A R B O N  &  O X Y G E N           #######
#######               C H E M I S T R Y                 #######
#######                                                 #######
###############################################################

# Below are chemical rates relevant for carbon & oxygen chemistry, 
# i.e. C I, C II, CO, O I, O II. 


""" C I collisional ionization rate (cm³ s⁻¹):
    C + e⁻ → C⁺ + e⁻ + e⁻ 
    See Glover & Jappsen (2007). """
@inline function kCIe( T::Float64 )
    T    = clamp( T, 1.0, 1e5 )
    u    = 11.26 * eV / (kB * T)
    return 6.85e-8 * (u^0.25) * exp(-u) / (0.193 + u)
end


""" O I collisional ionization rate (cm³ s⁻¹):
    O + e⁻ → O⁺ + e⁻ + e⁻ 
    See Glover & Jappsen (2007). """
@inline function kOIe( T::Float64 )
    T    = clamp( T, 1.0, 1e5 )
    u    = 13.6 * eV / (kB * T)
    return 3.59e-8 * (u^0.34) * exp(-u) / (0.073 + u)
end


const kCIIe_rr0 = 2.995e-9; const T_CIIrec = 49550.0

""" C II recombination rate (cm³ s⁻¹) in the gas phase:
    C⁺ + e⁻ → C + hν 
    See Gong et al. (2017). """
@inline function kCIIe( T::Float64 )
    T        = clamp( T, 1.0, 1e5 )
    # Radiative recombination rate:
    α        = sqrt(T / 6.67e-3)
    β        = sqrt(T / 1.9436e6)
    γ        = 0.7849  +  0.1597 * exp( - T_CIIrec / T )
    kCIIe_rr = kCIIe_rr0 / ( α * ((1 + α)^(1-γ)) * ((1 + β)^(1+γ)) )
    # Dielectronic recombination rate:
    kCIIe_dr = (T^(-3/2)) * ( 6.346e-9 * exp( - 12.17 / T )
                            + 9.793e-9 * exp( - 73.8 / T )
                            + 1.634e-6 * exp( - 15230.0 / T ) )
    return kCIIe_rr + kCIIe_dr
end


""" O II recombination rate (cm³ s⁻¹) in the gas phase:
    O⁺ + e⁻ → O + hν 
    See Glover & Jappsen (2007). """
@inline function kOIIe( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    if T ≤ 400.0
        KOIIe = 1.30e-10 * (T^(-0.64))
        return KOIIe
    else
        KOIIe = 1.41e-10 * (T^(-0.66))  +  7.4e-4 * (T^(-1.5)) * exp(-175000.0/T)
        return KOIIe
    end 
end


const c1_CIH3plus = 3.40e-8; const c2_CIH3plus = 6.97e-9
const c3_CIH3plus = 1.31e-7; const c4_CIH3plus = 1.51e-4
const T1_CIH3plus = 7.62;    const T2_CIH3plus = 1.38
const T3_CIH3plus = 2.66e1;  const T4_CIH3plus = 8.11e3
const KCIH3plus0  = 1.04e-9; const p_CIH3plus  = 0.00231

""" CHₓ formation rate/C I destruction rate (cm³ s⁻¹):
    C + H₃⁺ → CHₓ + H₂ 
    See Gong et al. (2017). 
    
    NOTE: Gong+ made an erratum in 2018 where they noted 
    that the exponent should be T^(-1.5) instead of T^(+1.5) as in their 2017 paper.
    This correction is included here. """
@inline function kCIH3plus( T::Float64 )
    T    = clamp( T, 1.0, 1e5 )
    T300 = T / 300.0

    KCIH3plus = (  KCIH3plus0 * (T300^p_CIH3plus) 
                + (T^(-1.5)) * ( c1_CIH3plus * exp( - T1_CIH3plus / T ) 
                               + c2_CIH3plus * exp( - T2_CIH3plus / T )
                               + c3_CIH3plus * exp( - T3_CIH3plus / T )
                               + c4_CIH3plus * exp( - T4_CIH3plus / T )) )

    return KCIH3plus 
end


const kCIIH2_10 = 2.31e-13; const T_CI = 23.0

""" CHₓ formation rate/H₂ destruction rate (cm³ s⁻¹):
    C⁺ + H₂ → CHₓ + H 
    See Gong et al. (2017). """
@inline function kCIIH2_1( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    return kCIIH2_10 * T^(-1.3) * exp( - T_CI / T ) 
end


const kCIIH2_20 = 0.99e-13

""" C I formation rate/H₂ destruction rate (cm³ s⁻¹):
    C⁺ + H₂ → C + H + H
    See Gong et al. (2017). """
@inline function kCIIH2_2( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    return kCIIH2_20 * T^(-1.3) * exp( - T_CI / T ) 
end


""" O⁺ charge exchange rate (cm³ s⁻¹):
    H⁺ + O → O⁺ + H
    See Gong et al. (2017). """
@inline function kHIIOI( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    return ( 1.1e-11 * T^0.517  +  4e-10 * T^0.00669 ) * exp( - 227.0/T )
end


""" O⁺ charge exchange rate (cm³ s⁻¹):
    O⁺ + H → H⁺ + O
    See Gong et al. (2017). """
@inline function kOIIHI( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    return 4.99e-11 * T^0.405  +  7.5e-10 * T^(-0.458) 
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
    2.81e-11 * (T^0.26)
end


const kOHxCI_0 = 7.95e-10; const p_OHxCI = - 0.339
const T_OHxCI  = 0.108 

""" CO formation rate (cm³ s⁻¹):
    OHₓ + C → CO + H
    See Gong et al. (2017). """
@inline function kOHxCI( T::Float64 )
    T = clamp( T, 1.0, 1e5 )
    kOHxCI_0 * (T^p_OHxCI) * exp( T_OHxCI / T )
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
    return 13e-7 * T^(-0.52)
end

