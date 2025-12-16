
""" FILE:       GasCoolingRates.jl

    PURPOSE:    Contains everything related to gas cooling.

    REFERENCES: The cooling rates below come from several sources, conveniently
                summarized in a few papers:

                (1) Rosdahl et al. (2013) 'RAMSES-RT: radiation hydrodynamics in the
                cosmological context', MNRAS, 436, 3, 2188-2231

                (2) Hopkins et al. (2018) 'FIRE-2 simulations: physics versus numerics
                in galaxy formation', MNRAS, 480, 1, 800-863

                (3)  Hopkins et al. (2023) 'FIRE-3: updated stellar evolution models, 
                yields, and microphysics and fitting functions for applications 
                in galaxy simulations', MNRAS, 519, 2, 3154-3181

                (4) Kim et al. (2023) 'Photochemistry and Heating/Cooling of the Multiphase
                Interstellar Medium with UV Radiative Transfer for Magnetohydrodynamic
                Simulations', ApJS, 264, 1, 10
                
                (5) Deng et al. (2024) 'RIGEL: Simulating dwarf galaxies at solar mass 
                resolution with radiative transfer and feedback from individual massive
                stars', A&A, 691, A231

                (6) Gnat & Ferland (2012) 'Ion-by-ion cooling efficiencies', ApJS, 199,
                1, 20, 5 (data online at: http://wise-obs.tau.ac.il/~orlyg/ion_by_ion/)

                (7) Hollenbach & McKee (1979) 'Molecule formation and infrared emission 
                in fast interstellar shocks. I. Physical processes.', ApJS, 41, 559-592

                (8) Whitworth & Jaffa (2018) 'A simple approach to CO cooling in
                molecular clouds', A&A, 611, A20

"""

using .ConstantsAndFunctions  # Needed constants 
using Interpolations          # Fot interpolation of metal-cooling tables 
using DelimitedFiles          # For reading data in metal-cooling tables


export Λ, Λ_HIcoll, Λ_HeIIcoll, Λ_HIion, Λ_HeIion, Λ_HeIIion, Λ_HIIrec, Λ_HeIIrec,
       Λ_HeIIIrec, Λ_HeIIdirec, Λ_ff, Λ_Comp, Λ_H2, Λ_neb

######### C O L L I S I O N A L  E X C I T A T I O N ###########

const Λ_HIcoll0 = 7.5e-19
const T_HIcoll  = 118348.0

""" The cooling rate (erg s⁻¹ cm⁻³) due to collisional
    excitation of H I by electrons. """
@inline function Λ_HIcoll( T::Float64, 
                         nHI::Float64, 
                          ne::Float64
                          )
                          
    return Λ_HIcoll0 * exp( - T_HIcoll/T ) * nHI * ne / (1 + sqrt(T/1e5))
end


const Λ_HeIIcoll0 = 5.54e-17
const T_HeIIcoll  = 473638.0

""" The cooling rate (erg s⁻¹ cm⁻³) due to collisional
    excitation of He II by electrons. """
@inline function Λ_HeIIcoll( T::Float64,
                         nHeII::Float64,
                            ne::Float64
                            )
    Λ_HeIIcoll0 * (T^(-0.397)) * exp( - T_HeIIcoll/T ) * nHeII * ne / (1 + sqrt(T/1e5))
end


######### C O L L I S I O N A L  I O N I Z A T I O N ###########

const Λ_HIion0 = 1.27e-21
const T_HIion  = 157809.1

""" The cooling rate (erg s⁻¹ cm⁻³) due to collisional
    ionization of H I by electrons. """
@inline function Λ_HIion( T::Float64, 
                        nHI::Float64, 
                         ne::Float64
                         )
    Λ_HIion0 * sqrt(T) * exp( - T_HIion/T ) * nHI * ne / (1 + sqrt(T/1e5))
end


const Λ_HeIion0 = 9.38e-22
const T_HeIion  = 285335.4

""" The cooling rate (erg s⁻¹ cm⁻³) due to collisional
    ionization of He I by electrons. """
@inline function Λ_HeIion( T::Float64,
                        nHeI::Float64,
                          ne::Float64
                          )
    Λ_HeIion0 * sqrt(T) * exp( - T_HeIion/T ) * nHeI * ne / (1 + sqrt(T/1e5))
end


const Λ_HeIIion0 = 4.95e-22
const T_HeIIion  = 631515.0

""" The cooling rate (erg s⁻¹ cm⁻³) due to collisional
    ionization of He II by electrons. """
@inline function Λ_HeIIion( T::Float64,
                        nHeII::Float64,
                           ne::Float64
                            )
    Λ_HeIIion0 * sqrt(T) * exp( - T_HeIIion/T ) * nHeII * ne / (1 + sqrt(T/1e5))
end


######### R E C O M B I N A T I O N S ###########

const Λ_HIIrec0_B = 3.435e-30
const Λ_HIIrec0_A = 1.778e-29
const T_HIIrec    = 315614.0

""" The cooling rate (erg s⁻¹ cm⁻³) due to hydrogen
    recombinations: H II + e⁻ → H I + hν. See ref. (1). """
function Λ_HIIrec( T::Float64, 
                nHII::Float64,
                  ne::Float64,
               CaseB::Bool = true
                  )
    λHI = T_HIIrec / T
    if CaseB
        return Λ_HIIrec0_B * T * nHII * ne * (λHI^1.97) / (1 + (λHI/2.25)^0.376)^3.72
    else
        return Λ_HIIrec0_A * T * nHII * ne * (λHI^1.965) / (1 + (λHI/0.541)^0.502)^2.697
    end
end


""" The cooling rate (erg s⁻¹ cm⁻³) due to helium
    recombinations: He II + e⁻ → He I + hν. See ref. (1). """
@inline function Λ_HeIIrec( T::Float64, 
                        nHeII::Float64,
                           ne::Float64,
                        CaseB::Bool = true 
                           )
    return kB * T * krecHeII(T, CaseB) * nHeII * ne
end


const ωHeII0       = 1.24e-13
const T_HeIIdirec1 = 470000.0
const T_HeIIdirec2 = 94000.0

""" The cooling rate (erg s⁻¹ cm⁻³) due to helium
    di-electronic recombinations: He II + e⁻ → He I + hν. 
    See ref. (1). """
@inline function Λ_HeIIdirec( T::Float64,
                          nHeII::Float64,
                             ne::Float64
                             )
    ωHeII = ωHeII0 * (T^(-1.5)) * exp( - T_HeIIdirec1/T ) * (1 + 0.3 * exp( - T_HeIIdirec2/T ))
    return ωHeII * nHeII * ne 
end


const Λ_HeIIIrec0_B = 8 * 3.435e-30
const Λ_HeIIIrec0_A = 8 * 1.778e-29
const T_HeIIIrec    = 1263030.0

""" The cooling rate (erg s⁻¹ cm⁻³) due to helium
    recombinations: He III + e⁻ → He II + hν. See ref. (1). """
function Λ_HeIIIrec( T::Float64, 
                nHeIII::Float64,
                    ne::Float64,
                 CaseB::Bool = true
                 )
    λHeII = T_HeIIIrec / T
    if CaseB 
        return Λ_HeIIIrec0_B * T * (λHeII^1.97) * nHeIII * ne / (1 + (λHeII/2.25)^0.376)^3.72
    else
        return Λ_HeIIIrec0_A * T * (λHeII^1.965) * nHeIII * ne / (1 + (λHeII/0.541)^0.502)^2.697
    end
end


######### F R E E - F R E E / B R E M S S T R A H L U N G ###########

const βff0 = 1.43e-27

""" The cooling rate (erg s⁻¹ cm⁻³) due to free-free emission.
    See ref. (2). """
@inline function Λ_ff( T::Float64,
                    nHII::Float64,
                   nHeII::Float64,
                  nHeIII::Float64,
                      ne::Float64
                      )
    βff = βff0 * sqrt(T) * ( 1.1 + 0.34 * exp( - (5.5 - log10(T))^2 / 3 ) )
    return βff * ne * (nHII + nHeII + 4*nHeIII)
end


######### C M B  C O M P T O N  C O O L I N G / H E A T I N G ###########

const Λ_Comp0 = 5.65e-36

""" The cooling/heating rate (erg s⁻¹ cm⁻³) due to Compton cooling/heating
    from the CMB at redshift z (default = 0).
    See ref. (2). """
@inline function Λ_Comp( T::Float64, 
                        ne::Float64,
                         z::Float64 = 0.0
                          )
    Z      = 1 + z
    TCMB   = TCMB0 * Z
    Λ_Comp = Λ_Comp0 * ne * (T - TCMB) * Z^4
    return Λ_Comp
end


###############################################################
#######                                                 #######
#######        M O L E C U L A R  H Y D R O G E N       #######
#######                  C O O L I N G                  #######
#######                                                 #######
###############################################################

const Λ_H2LTE0 = 1e-22

# Fitting coefficients for cooling by H₂-H I collisions (Galli & Palla 1998):
const a0_H2HI = - 103.0; const a1_H2HI = 97.59; const a2_H2HI = - 48.05
const a3_H2HI = 10.8;    const a4_H2HI = - 0.9032

# Fitting coefficients for cooling by H₂-H₂ collision (Glover & Abel 2008, their table 8):
const a0_H2H2 = - 23.962112; const a1_H2H2  = 2.09433740; const a2_H2H2   = - 0.77151436
const a3_H2H2 =   0.43693353; const a4_H2H2 = - 0.14913216; const a5_H2H2 = - 0.033638326

# Fitting coefficients for cooling by H₂-He I collision (Glover & Abel 2008, their table 8):
const a0_H2HeI = - 23.689237; const a1_H2HeI  = 2.1892372; const a2_H2HeI    = - 0.81520438
const a3_H2HeI =   0.29036281; const a4_H2HeI = - 0.16596184; const a5_H2HeI = 0.19191375

# Fitting coefficients for cooling by H₂-H II collision (Glover & Abel 2008, their table 8):
const a0_H2HII = - 21.716699; const a1_H2HII  = 1.3865783; const a2_H2HII    = - 0.37915285
const a3_H2HII =   0.11453688; const a4_H2HII = - 0.23214154; const a5_H2HII = 0.058538864

# Fitting coefficients for cooling by H₂-e⁻ collision, at low & higher temperatures
# (Glover & Abel 2008, their table 8):
const a0_H2eLOW = - 34.286155; const a1_H2eLOW = - 48.537163; const a2_H2eLOW = - 77.121176
const a3_H2eLOW = - 51.352459; const a4_H2eLOW = - 15.169160; const a5_H2eLOW = - 0.98120322

const a0_H2eHIGH = - 22.190316; const a1_H2eHIGH  = 1.5728955; const a2_H2eHIGH    = - 0.21335100
const a3_H2eHIGH =   0.96149759; const a4_H2eHIGH = - 0.91023195; const a5_H2eHIGH =   0.13749749


""" The cooling rate (erg s⁻¹ cm⁻³) due to H₂, taking into account
    collisions with H I, H₂, He I, H II, e⁻. Following ref. (3). """
@inline function Λ_H2( T::Float64,
                     nH2::Float64,
                     nHI::Float64,
                    nHII::Float64,
                    nHeI::Float64, 
                      ne::Float64 
                      )
    T  = clamp( T, 10.0, 3e4 )                  
    # Convenient definitions:
    T3 = T / 1e3
    x  = log10(T);  x2 = x^2; x3 = x2*x; x4 = x3*x
    y  = log10(T3); y2 = y^2; y3 = y2*y; y4 = y3*y; y5 = y4*y

    # Cooling rate in LTE/high density regime (Hollenbach & McKee 1979):
    Λ_H2LTE = Λ_H2LTE0 * nH2 * (6700.0 * exp( - 5.86/T3 ) + 16000.0 * exp( - 11.7/T3 )
              + 0.03 * exp( - 0.51/T3 ) + 9.5 * (T3^3.76) * exp( - 0.0022/(T3^3)) / 
              (1 + 0.12 * (T3^2.1)) ) 

    # Low-density cooling rate due to H₂-H I collisions (Galli & Palla 1998):
    Λ_H2_HI = nH2 * nHI * 10^( a0_H2HI + a1_H2HI*x + a2_H2HI*x2 + a3_H2HI*x3 + a4_H2HI*x4 )

    # Low-density cooling rate due to H₂-H₂ collisions (Glover & Abel 2008, their table 8).
    # The fit is valid for 100 K < T ≤ 6000 K.
    if T > 100.0
        # Cooling rate contribution:
        Λ_H2_H2 = nH2 * nH2 * 10^( a0_H2H2 + a1_H2H2*y + a2_H2H2*y2 + 
                  a3_H2H2*y3 + a4_H2H2*y4 + a5_H2H2*y5 )
    else
        Λ_H2_H2 = 0.0
    end

    # Low-density cooling rate due H₂-He I collisions (Glover & Abel 2008, their table 8).
    # The fit is valid over 10 K < T < 6000 K.
    
    Λ_H2_HeI = nH2 * nHeI * 10^( a0_H2HeI + a1_H2HeI*y + a2_H2HeI*y2 + 
                   a3_H2HeI*y3 + a4_H2HeI*y4 + a5_H2HeI*y5 )


    # Low-density cooling rate due H₂-H II collisions (Glover & Abel 2008, their table 8).
    # The fit is valid for 10 K < T < 10⁴ K:
    Λ_H2_HII = nH2 * nHII * 10^( a0_H2HII + a1_H2HII*y + a2_H2HII*y2 + 
                   a3_H2HII*y3 + a4_H2HII*y4 + a5_H2HII*y5 )


    # Low-density cooling rate due H₂-e⁻ collisions (Glover & Abel 2008, their table 8).
    # The fit is valid for 10 K < T < 10⁴ K:
    if T ≤ 200.0
        # Cooling rate contribution:
        Λ_H2_e = nH2 * ne * 10^( a0_H2eLOW + a1_H2eLOW*y + a2_H2eLOW*y2 + 
                 a3_H2eLOW*y3 + a4_H2eLOW*y4 + a5_H2eLOW*y5 )
    else
        # Cooling rate contribution:
        Λ_H2_e = nH2 * ne * 10^( a0_H2eHIGH + a1_H2eHIGH*y + a2_H2eHIGH*y2 + 
                 a3_H2eHIGH*y3 + a4_H2eHIGH*y4 + a5_H2eHIGH*y5 )
    end

    # Total low-density H₂-cooling rate:
    Λ_H2_lowdens = Λ_H2_HI + Λ_H2_H2 + Λ_H2_HeI + Λ_H2_HII + Λ_H2_e

    # Total H₂-cooling rate for all densities:
    if nH2 > 1e-14
        return Λ_H2_lowdens / (1 + Λ_H2_lowdens / Λ_H2LTE)
    else
        return Λ_H2_lowdens
    end
end


###############################################################
#######                                                 #######
#######        D E U T E R I U M  C O O L I N G         #######
#######                                                 #######
###############################################################


const a0_HD = - 42.45906; const a1_HD = 21.90083
const a2_HD = - 10.1954;  const a3_HD = 2.19788
const a4_HD = - 0.17286

const D00 = - 42.56788; const D01 = 0.92433;   const D02 = 0.54962;   const D03 = - 0.07676;  const D04 = 0.00275
const D10 = 21.93385;   const D11 = 0.77952;   const D12 = - 1.06447; const D13 = 0.11864;    const D14 = - 0.00366
const D20 = - 10.19097; const D21 = - 0.54263; const D22 = 0.62343;   const D23 = - 0.07366;  const D24 = 0.002514
const D30 = 2.19906;    const D31 = 0.11711;   const D32 = - 0.13768; const D33 = 0.01759;    const D34 = - 0.000666317
const D40 = - 0.17334;  const D41 = - 0.00835; const D42 = 0.0106;    const D43 = - 0.001482; const D44 = 0.000061926

""" The cooling rate (erg s⁻¹ cm⁻³) due to HD, taking into account
    collisions with H I (collisions with He, H₂, e⁻ are sub-dominant). 
    Following ref. (3). """
@inline function Λ_HD( T::Float64,
                     nHD::Float64,
                     nHI::Float64
                      )
    T  = clamp( T, 10.0, 3e4 )
    x  = log10( T ); x2 = x^2; x3 = x2*x; x4 = x3*x
    
    # Cooling rate in LTE/high density regime, following McGreer & Bryan's (2008) version 
    # of the Lipokva+ (2005) fit:
    y  = log10( 1e6 ); y2 = y^2; y3 = y2*y; y4 = y3*y

    Λ_HD_LTE = nHD * 10^(   D00 + D01*y + D02*y2 + D03*y3 + D04*y4
                         + (D10 + D11*y + D12*y2 + D13*y3 + D14*y4) * x
                         + (D20 + D21*y + D22*y2 + D23*y3 + D24*y4) * x2 
                         + (D30 + D31*y + D32*y2 + D33*y3 + D34*y4) * x3
                         + (D40 + D41*y + D42*y2 + D43*y3 + D44*y4) * x4 )

    # Low-density cooling rate due to HD-H I collisions (Lipokva+ 2005):
    Λ_HD_lowdens = nHI * nHD * 10^( a0_HD + a1_HD*x + a2_HD*x2 + a3_HD*x3 + a4_HD*x4 )

    # Total HD-cooling rate for all densities:
    if nHD > 1e-14
        return Λ_HD_lowdens / (1 + Λ_HD_lowdens / Λ_HD_LTE)
    else
        return Λ_HD_lowdens
    end
end


###############################################################
#######                                                 #######
#######             M E T A L  C O O L I N G            #######
#######                                                 #######
###############################################################

const Λ_neb0  = 3.68e-23
const a0_neb  = 0.692;    const a1_neb = - 0.586;  const a2_neb = 0.816
const a3_neb  = - 0.505;  const a4_neb = 0.118;    const a5_neb = 7.66e-3
const a6_neb  = - 5.08e-3

""" The cooling rate (erg s⁻¹ cm⁻³) due to collisionally excited
    metal ion lines, e.g. [O II]3729 Å, important in metal-enriched
    photoionized gas. Excluding C II, which is treated separately. 
    See refs. (4)-(5). """
@inline function Λ_neb( T::Float64, 
                     nHII::Float64,
                       ne::Float64,
                   Z_Zsun::Float64 = 1.0  # Gas metallicity relative to Solar
                   )
    # Convenient definitions:
    ne2   = clamp( ne/1e2, 0.0, 1e10 )
    T4    = clamp( T, 1e3, 5e4 ) / 1e4 
    y     = log(T4); y2 = y^2; y3 = y2*y; y4 = y3*y; y5 = y4*y; y6 = y5*y
    # Fitting function:
    fneb  = 10^( a0_neb + a1_neb*y + a2_neb*y2 + a3_neb*y3 
                + a4_neb*y4 + a5_neb*y5 + a6_neb*y6 )
    # Cooling rate:
    Λ_neb = Λ_neb0 * Z_Zsun * ne * nHII * exp(-3.86/T4) * fneb / (
            sqrt(T4) * (1 + 0.12 * ne2^(0.38 - 0.12 * y)) ) 

    return Λ_neb
end


const A_CII10 = 2.4e-6
const E_CII   = kB * 91.21

""" The C II (157.74 μm) cooling rate (erg s⁻¹ cm⁻³). 
    C II is modelled as a two-level atom, using
    rates compiled in Draine (2011). """
function Λ_CII( T::Float64, 
               nH::Float64,
              nHI::Float64, 
              nH2::Float64, 
             nHeI::Float64, 
               ne::Float64,
           Z_Zsun::Float64,
             xCII::Float64
                )

    nCII    = xCII * C_H_ratio * nH * Z_Zsun

    # Convenient definitions:
    T       = clamp( T, 0.0, 1e5 )
    T2      = T / 100.0
    T4      = T / 1e4
    logT2   = log(T2)

    # C II number density:
    nCII    = C_H_ratio * nH * Z_Zsun

    # Collisional de-excitation rate coefficients (cm³ s⁻¹) for 
    # H I, H₂(para), H₂(ortho), and He I. See Draine (2011).
    k10_HI  = 7.58e-10 * T2^(0.128 + 0.009*logT2)
    k10_H2p = 4.25e-10 * T2^(0.124 - 0.018*logT2)  # para-H₂
    k10_H2o = 5.14e-10 * T2^(0.095 + 0.023*logT2)  # ortho-H₂
    k10_H2  = (3/4) * k10_H2o + (1/4) * k10_H2p    # Assuming 3:1 ortho-to-para ratio
    k10_HeI = 0.38 * k10_HI
    # Collisional de-excitation rate coefficients for e⁻. 
    # See Draine (2011), p. 196.
    k10_e   = 4.53e-8 * T4^(-0.5)

    # Total de-excitation rate:
    C10     = k10_HI * nHI  +  k10_H2 * nH2  +  k10_HeI * nHeI  +  k10_e * ne
    # Total excitation rate (using g_u/g_l = 4/2):
    C01     = 2 * C10 * exp( - E_CII / (kB * T) )
    
    # Cooling rate:
    Λ_CII   = nCII * A_CII10 * E_CII * C01 / (A_CII10 + C10 + C01)

    return Λ_CII
end


# Einstein A coefficients, transition energies, and statistical weights
# for C I (Draine 2011, p. 198):
const A_CI10 = 7.93e-8;    const A_CI21 = 2.68e-7
const E_CI10 = kB * 23.62; const E_CI21 = kB * 38.84; const E_CI20 = E_CI10 + E_CI21
const g_CI0  = 1;          const g_CI1  = 3;          const g_CI2  = 5

""" The C I (609.13 μm, 370.41 μm) cooling rate (erg s⁻¹ cm⁻³). 
    C I is modelled as a 3-level atom, using rates compiled/cited
    in Draine (2011). """
function Λ_CI( T::Float64, 
              nH::Float64,
            nHII::Float64,
             nHI::Float64, 
             nH2::Float64, 
            nHeI::Float64, 
              ne::Float64,
          Z_Zsun::Float64,
             xCI::Float64
              )

    nCI     = xCI * C_H_ratio * nH * Z_Zsun

    # Convenient definitions:
    T       = clamp( T, 0.0, 1e5 )
    T2      = T / 1e2 
    T4      = T / 1e4
    logT2   = log(T2)

    # H II rates: 
    k10_HII = 7.60e-10 * T2^(0.464 - 0.042*logT2)
    k20_HII = 3.02e-10 * T2^(1.217 - 0.126*logT2)
    k21_HII = 2.30e-9  * T2^(0.919 - 0.111*logT2)

    # H I rates:
    k10_HI  = 1.26e-11 * T2^(0.115 + 0.057*logT2)
    k20_HI  = 2.64e-11 * T2^(0.231 + 0.046*logT2)
    k21_HI  = 8.90e-11 * T2^(0.228 + 0.046*logT2)

    # H₂ rates (assume 3:1 ortho-to-para ratio at the end):
    k10_H2p = 0.67e-10 * T2^(-0.085 + 0.102*logT2)
    k10_H2o = 0.71e-10 * T2^(-0.004 + 0.049*logT2)
    k10_H2  = (3/4) * k10_H2o + (1/4) * k10_H2p

    k20_H2p = 0.86e-10 * T2^(-0.010 + 0.048*logT2)
    k20_H2o = 0.69e-10 * T2^(0.169 + 0.038*logT2)
    k20_H2  = (3/4) * k20_H2o + (1/4) * k20_H2p

    k21_H2p = 1.75e-10 * T2^(0.072 + 0.064*logT2)
    k21_H2o = 1.48e-10 * T2^(0.263 + 0.031*logT2)
    k21_H2  = (3/4) * k21_H2o + (1/4) * k21_H2p

    # He I rates:
    k10_HeI = 1.76e-11 * T2^(0.106 - 0.036*logT2)
    k20_HeI = 4.44e-11 * T2^0.029
    k21_HeI = 8.33e-11 * T2^(0.113 + 0.029*logT2)
    
    # Electron rates, using effective collision strength
    # fits from Johnson et al. (1987), as recommended by Draine (2011):
    if T ≥ 7.5 && T ≤ 1e3 
        X = log(T); X2 = X^2; X3 = X2*X; X4 = X3*X

        a0_10 = - 0.925e1;     a0_20 = - 0.769735e1; a0_21 = - 0.743870e1
        a1_10 = - 0.925141e1;  a1_20 = - 0.130743e1; a1_21 = - 0.574430
        a2_10 = 0.361184;      a2_20 = 0.697638;     a2_21 = 0.358264
        a3_10 = - 0.150892e-1; a3_20 = - 0.111338;   a3_21 = - 0.418166e-1
        a4_10 = - 0.656325e-3; a4_20 = 0.705277e-2;  a4_21 = 0.235272e-2

        Ω10 = exp( a0_10 + a1_10*X + a2_10*X2 + a3_10*X3 + a4_10*X4 )
        Ω20 = exp( a0_20 + a1_20*X + a2_20*X2 + a3_20*X3 + a4_20*X4 )
        Ω21 = exp( a0_21 + a1_21*X + a2_21*X2 + a3_21*X3 + a4_21*X4 )

        k10_e = 8.629e-8 * Ω10 / (sqrt(T4) * g_CI1)
        k20_e = 8.629e-8 * Ω20 / (sqrt(T4) * g_CI2)
        k21_e = 8.629e-8 * Ω21 / (sqrt(T4) * g_CI2)

    elseif T > 1e3 && T ≤ 1e4
        X = log(T); X2 = X^2; X3 = X2*X; X4 = X3*X

        a0_10 = 0.444600e3;    a0_20 = 0.350609e3;   a0_21 = 0.386186e3
        a1_10 = - 0.227913e3;  a1_20 = - 0.187474e3; a1_21 = - 0.202192e3
        a2_10 = 0.425952e2;    a2_20 = 0.361803e2;   a2_21 = 0.385049e2
        a3_10 = - 0.347620e1;  a3_20 = - 0.303283e1; a3_21 = - 0.319268e1
        a4_10 = 0.105085;      a4_20 = 0.938138e-1;  a4_21 = 0.978573e-1

        Ω10 = exp( a0_10 + a1_10*X + a2_10*X2 + a3_10*X3 + a4_10*X4 )
        Ω20 = exp( a0_20 + a1_20*X + a2_20*X2 + a3_20*X3 + a4_20*X4 )
        Ω21 = exp( a0_21 + a1_21*X + a2_21*X2 + a3_21*X3 + a4_21*X4 )

        k10_e = 8.629e-8 * Ω10 / (sqrt(T4) * g_CI1)
        k20_e = 8.629e-8 * Ω20 / (sqrt(T4) * g_CI2)
        k21_e = 8.629e-8 * Ω21 / (sqrt(T4) * g_CI2)

    else
        k10_e = 0.0
        k20_e = 0.0
        k21_e = 0.0
    end

    # Collect rates:
    C10   = k10_HI * nHI  +  k10_HII * nHII  +  k10_H2 * nH2  +  k10_HeI * nHeI  +  k10_e * ne 
    C20   = k20_HI * nHI  +  k20_HII * nHII  +  k20_H2 * nH2  +  k20_HeI * nHeI  +  k20_e * ne
    C21   = k21_HI * nHI  +  k21_HII * nHII  +  k21_H2 * nH2  +  k21_HeI * nHeI  +  k21_e * ne

    R10   = C10 + A_CI10
    R20   = C20
    R21   = C21 + A_CI21
    R01   = (g_CI1/g_CI0) * C10 * exp( - E_CI10 / (kB * T) )
    R02   = (g_CI2/g_CI0) * C20 * exp( - E_CI20 / (kB * T) )
    R12   = (g_CI2/g_CI1) * C21 * exp( - E_CI21 / (kB * T) ) 

    # Level ratio n1/n0:
    n1_n0 = (R01*R20 + R01*R21 + R21*R02) / (R10*R20 + R10*R21 + R12*R20)
    # Level ratio n2/n0:
    n2_n0 = (R01*R10 + R02*R12 + R12*R01) / (R10*R20 + R10*R21 + R12*R20)

    # Cooling rate from 609 μm line:
    n1      = nCI * n1_n0 / ( 1 + n1_n0 + n2_n0 )
    Λ_CI609 = n1 * A_CI10 * E_CI10

    # Cooling rate from 370 μm line:
    n2      = nCI * n2_n0 / ( 1 + n1_n0 + n2_n0 )
    Λ_CI370 = n2 * A_CI21 * E_CI21

    return Λ_CI609 + Λ_CI370
end


# Einstein A coefficients, transition energies, and statistical weights
# for O I (Draine 2011, p. 198):
const A_OI10 = 8.95e-5;    const A_OI21 = 1.70e-5
const E_OI10 = kB * 227.7; const E_OI21 = kB * 98.90; const E_OI20 = E_OI10 + E_OI21
const g_OI0  = 5;          const g_OI1  = 3;          const g_OI2  = 1

""" The O I (63.18 μm, 145.5 μm) cooling rate (erg s⁻¹ cm⁻³). 
    O I is modelled as a 3-level atom, using rates compiled in Draine (2011). """
function Λ_OI( T::Float64, 
              nH::Float64,
             nHI::Float64,
             nH2::Float64,
              ne::Float64,
          Z_Zsun::Float64,
             xOI::Float64
                )

    nOI     = xOI * O_H_ratio * nH * Z_Zsun

    # Convenient definitions:
    T       = clamp( T, 1.0, 1e5 )
    T2      = T / 1e2 
    T4      = T / 1e4
    logT2   = log(T2)

    # H I rates:
    k10_HI  = 3.57e-10 * T2^(0.419 - 0.003*logT2)
    k20_HI  = 3.19e-10 * T2^(0.369 - 0.006*logT2)
    k21_HI  = 4.34e-10 * T2^(0.755 - 0.160*logT2)

    # H₂ rates (assume 3:1 ortho-to-para ratio at the end):
    k10_H2p = 1.49e-10 * T2^(0.264 + 0.025*logT2)
    k10_H2o = 1.37e-10 * T2^(0.296 + 0.043*logT2)
    k10_H2  = (3/4) * k10_H2o + (1/4) * k10_H2p

    k20_H2p = 1.90e-10 * T2^(0.203 + 0.041*logT2)
    k20_H2o = 2.23e-10 * T2^(0.237 + 0.058*logT2)
    k20_H2  = (3/4) * k20_H2o + (1/4) * k20_H2p

    k21_H2p = 2.10e-12 * T2^(0.889 + 0.043*logT2)
    k21_H2o = 3.00e-12 * T2^(1.198 + 0.525*logT2)
    k21_H2  = (3/4) * k21_H2o + (1/4) * k21_H2p

    # Electron rates:
    Ω10     = 0.041 * (T4^0.69)  +  0.064 * (T4^1.72)    # Collisional strength 
    Ω20     = 0.0136 * (T4^0.61)  +  0.0186 * (T4^1.49)
    Ω21     = 0.00166 * (T4^0.71)  +  0.0288 * (T4^1.97)

    k10_e   = 8.629e-8 * Ω10 / (sqrt(T4) * g_OI1)
    k20_e   = 8.629e-8 * Ω20 / (sqrt(T4) * g_OI2)
    k21_e   = 8.629e-8 * Ω21 / (sqrt(T4) * g_OI2)

    # Collect rates:
    C10     = k10_HI * nHI  +  k10_H2 * nH2  +  k10_e * ne 
    C20     = k20_HI * nHI  +  k20_H2 * nH2  +  k20_e * ne 
    C21     = k21_HI * nHI  +  k21_H2 * nH2  +  k21_e * ne

    R10     = C10 + A_OI10
    R20     = C20
    R21     = C21 + A_OI21
    R01     = (g_OI1/g_OI0) * C10 * exp( - E_OI10 / (kB * T) )
    R02     = (g_OI2/g_OI0) * C20 * exp( - E_OI20 / (kB * T) )
    R12     = (g_OI2/g_OI1) * C21 * exp( - E_OI21 / (kB * T) ) 

    # Level ratio n1/n0:
    n1_n0   = (R01*R20 + R01*R21 + R21*R02) / (R10*R20 + R10*R21 + R12*R20)
    # Level ratio n2/n0:
    n2_n0   = (R01*R10 + R02*R12 + R12*R01) / (R10*R20 + R10*R21 + R12*R20)

    # Cooling rate from 63 μm line:
    n1      = nOI * n1_n0 / ( 1 + n1_n0 + n2_n0 )
    Λ_OI63  = n1 * A_OI10 * E_OI10

    # Cooling rate from 146 μm line:
    n2      = nOI * n2_n0 / ( 1 + n1_n0 + n2_n0 )
    Λ_OI146 = n2 * A_OI21 * E_OI21

    return Λ_OI63 + Λ_OI146
end


const Λ_CO_0     = 2.16e-27
const ncr_CO_0   = 1.9e4
const N_COcrit_0 = 6.66e14
const Tmin_CO    = 5.0     # Ignore CO cooling below this temperature 

""" The CO cooling rate (erg s⁻¹ cm⁻³). Uses the functional 
    form from Hollenbach & McKee (1979) (HM), which in the low-density 
    regime is the same as in  Whitworth & Jaffa (2018). The coefficient 
    in the low-density regime from HM has been correected to agree 
    with Whitworth & Jaffa."""
function Λ_CO( T::Float64, 
              nH::Float64,
             nH2::Float64,
          Z_Zsun::Float64,
             xCO::Float64,
            N_CO::Float64 = 0.0 
               )
    nCO      = xCO * C_H_ratio * nH * Z_Zsun  # The CO density

    # Convenient definitions:
    ncr_CO   = ncr_CO_0 * (T^0.5)    # Critical density (Hollenbach & McKee 1979)
    N_COcrit = N_COcrit_0 * T        # Critical column density for line trapping (Hollenbach & McKee 1979)

    # The CO cooling rate, valid in both the optically thin & low density regime, and in the 
    # dense and/or optically thick regime:
    if T > Tmin_CO
        T    = clamp( T, Tmin_CO, 1e4 )
        Λ_CO = Λ_CO_0 * nCO * nH2 * (T^1.5) / (1  +  (nH/ncr_CO) * (1  +  N_CO / N_COcrit))
    else
        # Ignore CO cooling at very low temperatures, where the fit likely breaks down:
        Λ_CO = 0.0
    end

    return Λ_CO
end

### CLOUDY - data for Carbon cooling in collisional ionization equilibrium ####

# Read in CLOUDY data:
CLOUDYdata_C                = readdlm( joinpath(@__DIR__, "CLOUDYdata/CoolingTable_C.txt"), skipstart = 19 )
T_CLOUDYdata_C              = CLOUDYdata_C[:,1]
CoolingRate_CLOUDYdata_C    = CLOUDYdata_C[:,end]
# Interpolate data:
const InterpolCoolingRate_C = LinearInterpolation( T_CLOUDYdata_C, CoolingRate_CLOUDYdata_C, 
                                           extrapolation_bc = Line() )
# Also get interpolation temperature range:
const T_CLOUDYdata_C_min    = minimum( T_CLOUDYdata_C )
const T_CLOUDYdata_C_max    = maximum( T_CLOUDYdata_C )

""" The C (carbon) cooling rate (erg s⁻¹ cm⁻³) at high 
    temperatures, assuming collisional ionization equilibrium,
    and Solar abundance ratio. See ref. (6). """
function Λ_C_CLOUDY( T::Float64, 
                  nHII::Float64, 
                    ne::Float64,
                Z_Zsun::Float64 
                    )
    if T > T_CLOUDYdata_C_min 
        nC = C_H_ratio * nHII * Z_Zsun
        return nC * ne * InterpolCoolingRate_C(T) 
    else
        return 0.0
    end
end


### CLOUDY - data for Nitrogen cooling in collisional ionization equilibrium ####

# Read in CLOUDY data:
CLOUDYdata_N                = readdlm( joinpath(@__DIR__, "CLOUDYdata/CoolingTable_N.txt"), skipstart = 20 )
T_CLOUDYdata_N              = CLOUDYdata_N[:,1]
CoolingRate_CLOUDYdata_N    = CLOUDYdata_N[:,end]
# Interpolate data:
const InterpolCoolingRate_N = LinearInterpolation( T_CLOUDYdata_N, CoolingRate_CLOUDYdata_N, 
                                           extrapolation_bc = Line() )
# Also get interpolation temperature range:
const T_CLOUDYdata_N_min    = minimum( T_CLOUDYdata_N )
const T_CLOUDYdata_N_max    = maximum( T_CLOUDYdata_N )

""" The N (nitrogen) cooling rate (erg s⁻¹ cm⁻³) at high 
    temperatures, assuming collisional ionization equilibrium,
    and Solar abundance ratio. See ref. (6). """
function Λ_N_CLOUDY( T::Float64, 
                  nHII::Float64, 
                    ne::Float64,
                Z_Zsun::Float64 
                    )
    if T > T_CLOUDYdata_N_min
        nN = N_H_ratio * nHII * Z_Zsun
        return nN * ne * InterpolCoolingRate_N(T) 
    else
        return 0.0
    end
end


### CLOUDY - data for Oxygen cooling in collisional ionization equilibrium ####

# Read in CLOUDY data:
CLOUDYdata_O                = readdlm( joinpath(@__DIR__, "CLOUDYdata/CoolingTable_O.txt"), skipstart = 21 )
T_CLOUDYdata_O              = CLOUDYdata_O[:,1]
CoolingRate_CLOUDYdata_O    = CLOUDYdata_O[:,end]
# Interpolate data:
const InterpolCoolingRate_O = LinearInterpolation( T_CLOUDYdata_O, CoolingRate_CLOUDYdata_O, 
                                           extrapolation_bc = Line() )
# Also get interpolation temperature range:
const T_CLOUDYdata_O_min    = minimum( T_CLOUDYdata_O )
const T_CLOUDYdata_O_max    = maximum( T_CLOUDYdata_O )

""" The O (oxygen) cooling rate (erg s⁻¹ cm⁻³) at high 
    temperatures, assuming collisional ionization equilibrium,
    and Solar abundance ratio. See ref. (6). """
function Λ_O_CLOUDY( T::Float64, 
                  nHII::Float64, 
                    ne::Float64,
                Z_Zsun::Float64 
                    )
    if T > T_CLOUDYdata_O_min
        nO = O_H_ratio * nHII * Z_Zsun
        return nO * ne * InterpolCoolingRate_O(T) 
    else
        return 0.0
    end
end


### CLOUDY - data for Neon cooling in collisional ionization equilibrium ####

# Read in CLOUDY data:
CLOUDYdata_Ne                = readdlm( joinpath(@__DIR__, "CLOUDYdata/CoolingTable_Ne.txt"), skipstart = 23 )
T_CLOUDYdata_Ne              = CLOUDYdata_Ne[:,1]
CoolingRate_CLOUDYdata_Ne    = CLOUDYdata_Ne[:,end]
# Interpolate data:
const InterpolCoolingRate_Ne = LinearInterpolation( T_CLOUDYdata_Ne, CoolingRate_CLOUDYdata_Ne, 
                                           extrapolation_bc = Line() )
# Also get interpolation temperature range:
const T_CLOUDYdata_Ne_min    = minimum( T_CLOUDYdata_Ne )
const T_CLOUDYdata_Ne_max    = maximum( T_CLOUDYdata_Ne )

""" The O (oxygen) cooling rate (erg s⁻¹ cm⁻³) at high 
    temperatures, assuming collisional ionization equilibrium,
    and Solar abundance ratio. See ref. (6). """
function Λ_Ne_CLOUDY( T::Float64, 
                   nHII::Float64, 
                     ne::Float64,
                 Z_Zsun::Float64 
                     )
    if T > T_CLOUDYdata_Ne_min 
        nNe = Ne_H_ratio * nHII * Z_Zsun
        return nNe * ne * InterpolCoolingRate_Ne(T) 
    else
        return 0.0
    end
end


### CLOUDY - data for Magnesium cooling in collisional ionization equilibrium ####

# Read in CLOUDY data:
CLOUDYdata_Mg                = readdlm( joinpath(@__DIR__, "CLOUDYdata/CoolingTable_Mg.txt"), skipstart = 25 )
T_CLOUDYdata_Mg              = CLOUDYdata_Mg[:,1]
CoolingRate_CLOUDYdata_Mg    = CLOUDYdata_Mg[:,end]
# Interpolate data:
const InterpolCoolingRate_Mg = LinearInterpolation( T_CLOUDYdata_Mg, CoolingRate_CLOUDYdata_Mg, 
                                           extrapolation_bc = Line() )
# Also get interpolation temperature range:
const T_CLOUDYdata_Mg_min    = minimum( T_CLOUDYdata_Mg )
const T_CLOUDYdata_Mg_max    = maximum( T_CLOUDYdata_Mg )

""" The Mg (magnesium) cooling rate (erg s⁻¹ cm⁻³) at high 
    temperatures, assuming collisional ionization equilibrium,
    and Solar abundance ratio. See ref. (6). """
function Λ_Mg_CLOUDY( T::Float64, 
                   nHII::Float64, 
                     ne::Float64,
                 Z_Zsun::Float64 
                     )
    if T > T_CLOUDYdata_Mg_min 
        nMg = Mg_H_ratio * nHII * Z_Zsun
        return nMg * ne * InterpolCoolingRate_Mg(T) 
    else
        return 0.0
    end
end


### CLOUDY - data for Silicon cooling in collisional ionization equilibrium ####

# Read in CLOUDY data:
CLOUDYdata_Si                = readdlm( joinpath(@__DIR__, "CLOUDYdata/CoolingTable_Si.txt"), skipstart = 27 )
T_CLOUDYdata_Si              = CLOUDYdata_Si[:,1]
CoolingRate_CLOUDYdata_Si    = CLOUDYdata_Si[:,end]
# Interpolate data:
const InterpolCoolingRate_Si = LinearInterpolation( T_CLOUDYdata_Si, CoolingRate_CLOUDYdata_Si, 
                                           extrapolation_bc = Line() )
# Also get interpolation temperature range:
const T_CLOUDYdata_Si_min    = minimum( T_CLOUDYdata_Si )
const T_CLOUDYdata_Si_max    = maximum( T_CLOUDYdata_Si )

""" The Si (silicon) cooling rate (erg s⁻¹ cm⁻³) at high 
    temperatures, assuming collisional ionization equilibrium,
    and Solar abundance ratio. See ref. (6). """
function Λ_Si_CLOUDY( T::Float64, 
                   nHII::Float64, 
                     ne::Float64,
                 Z_Zsun::Float64 
                     )
    if T > T_CLOUDYdata_Si_min 
        nSi = Si_H_ratio * nHII * Z_Zsun
        return nSi * ne * InterpolCoolingRate_Si(T) 
    else
        return 0.0
    end
end


### CLOUDY - data for Sulfur cooling in collisional ionization equilibrium ####

# Read in CLOUDY data:
CLOUDYdata_S                = readdlm( joinpath(@__DIR__, "CLOUDYdata/CoolingTable_S.txt"), skipstart = 29 )
T_CLOUDYdata_S              = CLOUDYdata_S[:,1]
CoolingRate_CLOUDYdata_S    = CLOUDYdata_S[:,end]
# Interpolate data:
const InterpolCoolingRate_S = LinearInterpolation( T_CLOUDYdata_S, CoolingRate_CLOUDYdata_S, 
                                           extrapolation_bc = Line() )
# Also get interpolation temperature range:
const T_CLOUDYdata_S_min    = minimum( T_CLOUDYdata_S )
const T_CLOUDYdata_S_max    = maximum( T_CLOUDYdata_S )

""" The S (sulfur) cooling rate (erg s⁻¹ cm⁻³) at high 
    temperatures, assuming collisional ionization equilibrium,
    and Solar abundance ratio. See ref. (6). """
function Λ_S_CLOUDY( T::Float64, 
                  nHII::Float64, 
                    ne::Float64,
                Z_Zsun::Float64 
                    )
    if T > T_CLOUDYdata_S_min 
        nS = S_H_ratio * nHII * Z_Zsun
        return nS * ne * InterpolCoolingRate_S(T) 
    else
        return 0.0
    end
end


### CLOUDY - data for Iron cooling in collisional ionization equilibrium ####

# Read in CLOUDY data:
CLOUDYdata_Fe                = readdlm( joinpath(@__DIR__, "CLOUDYdata/CoolingTable_Fe.txt"), skipstart = 39 )
T_CLOUDYdata_Fe              = CLOUDYdata_Fe[:,1]
CoolingRate_CLOUDYdata_Fe    = CLOUDYdata_Fe[:,end]
# Interpolate data:
const InterpolCoolingRate_Fe = LinearInterpolation( T_CLOUDYdata_Fe, CoolingRate_CLOUDYdata_Fe, 
                                           extrapolation_bc = Line() )
# Also get interpolation temperature range:
const T_CLOUDYdata_Fe_min    = minimum( T_CLOUDYdata_Fe )
const T_CLOUDYdata_Fe_max    = maximum( T_CLOUDYdata_Fe )

""" The Fe (iron) cooling rate (erg s⁻¹ cm⁻³) at high 
    temperatures, assuming collisional ionization equilibrium,
    and Solar abundance ratio. See ref. (6). """
function Λ_Fe_CLOUDY( T::Float64, 
                  nHII::Float64, 
                    ne::Float64,
                Z_Zsun::Float64 
                    )
    if T > T_CLOUDYdata_Fe_min 
        nFe = Fe_H_ratio * nHII * Z_Zsun
        return nFe * ne * InterpolCoolingRate_Fe(T) 
    else
        return 0.0
    end
end


# Interpolation coefficients (Kim et al. 2023):
const a_Z = 10.0; const T1_Z = 2e4; const T2_Z = 3.5e4

""" Total metal cooling rate (erg s⁻¹ cm⁻³). """
function Λ_Z( T::Float64, 
            nHI::Float64,
            nH2::Float64,
           nHII::Float64,
           nHeI::Float64, 
             ne::Float64, 
         Z_Zsun::Float64,
            xCI::Float64,  # nCI / nC
           xCII::Float64,  # nCII / nC
            xCO::Float64,  # nCO / nC 
            xOI::Float64,  # nOI / nO
           N_CO::Float64 = 0.0
             )

    nH   = nHI + nHII + 2 * nH2

    # Metal cooling rate at high temperatures from CLOUDY, assuming CIE:
    Λ_Z_CLOUDY = ( Λ_C_CLOUDY(T, nHII, ne, Z_Zsun)  +  Λ_N_CLOUDY(T, nHII, ne, Z_Zsun)
                 + Λ_O_CLOUDY(T, nHII, ne, Z_Zsun)  +  Λ_Ne_CLOUDY(T, nHII, ne, Z_Zsun)
                 + Λ_Mg_CLOUDY(T, nHII, ne, Z_Zsun) +  Λ_Si_CLOUDY(T, nHII, ne, Z_Zsun)
                 + Λ_S_CLOUDY(T, nHII, ne, Z_Zsun)  +  Λ_Fe_CLOUDY(T, nHII, ne, Z_Zsun))
    # Total metal cooling rate, using interpolation (Kim et al. 2023):
    S   = 1 / ( 1 + exp( - a_Z*(T - 0.5*(T1_Z + T2_Z))/(T2_Z - T1_Z) ) )

    Λ_Z = ((1 - S) * ( Λ_neb(T, nHII, ne, Z_Zsun) + Λ_CII(T, nH, nHI, nH2, nHeI, ne, Z_Zsun, xCII)
        +  Λ_CI(T, nH, nHII, nHI, nH2, nHeI, ne, Z_Zsun, xCI) + Λ_OI(T, nH, nHI, nH2, ne, Z_Zsun, xOI)
        +  Λ_CO(T, nH, nH2, Z_Zsun, xCO, N_CO) ) 
        +  S * Λ_Z_CLOUDY )

    return Λ_Z
end


###############################################################
#######                                                 #######
#######         T O T A L  C O O L I N G   R A T E      #######
#######                                                 #######
###############################################################

""" The total cooling rate (erg s⁻¹ cm⁻³) from all processes 
    considered. """
function Λ(    T::Float64, 
             nHI::Float64, 
            nHII::Float64, 
            nHeI::Float64,
           nHeII::Float64,
          nHeIII::Float64,
              ne::Float64,
             nH2::Float64,
             nHD::Float64,
               z::Float64 = 0.0,
           CaseB::Bool    = true 
        )

    # Cooling rate from all processes except Compton cooling/heating and dust cooling/heating:
    Λ1   =  ( Λ_HIcoll(T, nHI, ne) + Λ_HeIIcoll(T, nHeII, ne) 

            + Λ_HIion(T, nHI, ne) + Λ_HeIion(T, nHeI, ne) + Λ_HeIIion(T, nHeII, ne)

            + Λ_HIIrec(T, nHII, ne, CaseB) + Λ_HeIIrec(T, nHeII, ne, CaseB) + Λ_HeIIIrec(T, nHeIII, ne, CaseB) 
            + Λ_HeIIdirec(T, nHeII, ne)
    
            + Λ_ff(T, nHII, nHeII, nHeIII, ne) 
    
            + Λ_H2(T, nH2, nHI, nHII, nHeI, ne) 

            + Λ_HD(T, nHD, nHI) )

    # Correct for the CMB-induced limit following Hopkins et al. (2023):
    TCMB = TCMB0 * (1 + z)
    Λ1   = Λ1 * (T - TCMB) / (T + TCMB)

    # Add Compton and dust cooling/heating to get the total cooling rate:
    Λ    = ( Λ1  +  Λ_Comp(T, ne, z) )

    return Λ
end
