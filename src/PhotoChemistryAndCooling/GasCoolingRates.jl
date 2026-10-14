
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

# Needed constants & for interpolation of metal-cooling tables:
using .ConstantsAndFunctions  
using .ConstantsAndFunctions: LinearLUT, CreateLUTFromData, EvaluateLUTFlat
#using Interpolations           # Fot interpolation of metal-cooling tables 
using DelimitedFiles          # For reading data in metal-cooling tables
using StaticArrays            # For level population solves 


# Ignore cooling if the key species density is below nIgnoreCooling. 
# For example, O III cooling is ignored if nOIII ≤ nIgnoreCooling
const nIgnoreCooling::Float64 = 1e-30   # Default: 10⁻³⁰ cm⁻³



export Λ, Λ_HIcoll, Λ_HeIIcoll, Λ_HIion, Λ_HeIion, Λ_HeIIion, Λ_HIIrec, Λ_HeIIrec,
       Λ_HeIIIrec, Λ_HeIIdirec, Λ_ff, Λ_Comp, Λ_H2, Λ_OII, Λ_OIII, Λ_SIII 

######### C O L L I S I O N A L  E X C I T A T I O N ###########

const Λ_HIcoll0::Float64 = 7.5e-19
const T_HIcoll::Float64  = 118348.0

""" The cooling rate (erg s⁻¹ cm⁻³) due to collisional
    excitation of H I by electrons. """
@inline function Λ_HIcoll( T::Float64, 
                         nHI::Float64, 
                          ne::Float64
                          )

    if T ≤ 500.0
        # Ignore H II collisional excitation cooling in cold gas 
        # where it is negligible regardless:
        return 0.0
    end

    T5 = T * 1e-5 # T / 10⁵ K
                          
    return @fastmath Λ_HIcoll0 * exp( - T_HIcoll/T ) * nHI * ne / (1 + sqrt(T5))
end


const Λ_HeIIcoll0::Float64 = 5.54e-17
const T_HeIIcoll::Float64  = 473638.0

""" The cooling rate (erg s⁻¹ cm⁻³) due to collisional
    excitation of He II by electrons. """
@inline function Λ_HeIIcoll( T::Float64,
                         nHeII::Float64,
                            ne::Float64
                            )

    if T ≤ 500.0
        # Ignore He II collisional excitation cooling in cold gas 
        # where it is negligible regardless:
        return 0.0
    end

    T5 = T * 1e-5 # T / 10⁵ K

    return @fastmath Λ_HeIIcoll0 * (T^(-0.397)) * exp( - T_HeIIcoll/T ) * nHeII * ne / (1 + sqrt(T5))
end


######### C O L L I S I O N A L  I O N I Z A T I O N ###########

const Λ_HIion0::Float64 = 1.27e-21
const T_HIion::Float64  = 157809.1
const constT5::Float64  = sqrt(1e-5)  # For converting between sqrt(T) and sqrt(T5)

""" The cooling rate (erg s⁻¹ cm⁻³) due to collisional
    ionization of H I by electrons. """
@inline function Λ_HIion( T::Float64, 
                        nHI::Float64, 
                         ne::Float64
                         )

    if T ≤ 500.0 || nHI ≤ nIgnoreCooling
        # Ignore collisional ionization cooling in cold gas 
        # where it is negligible regardless, or if H I is 
        # negligible:
        return 0.0
    end

    sqrtT = @fastmath sqrt(T)

    return @fastmath Λ_HIion0 * sqrtT * exp( - T_HIion/T ) * nHI * ne / (1 + constT5 * sqrtT)
end


const Λ_HeIion0::Float64 = 9.38e-22
const T_HeIion::Float64  = 285335.4

""" The cooling rate (erg s⁻¹ cm⁻³) due to collisional
    ionization of He I by electrons. """
@inline function Λ_HeIion( T::Float64,
                        nHeI::Float64,
                          ne::Float64
                          )

    if T ≤ 500.0 || nHeI ≤ nIgnoreCooling
        # Ignore collisional ionization cooling in cold gas 
        # where it is negligible regardless:
        return 0.0
    end
    
    sqrtT = @fastmath sqrt(T)

    return @fastmath Λ_HeIion0 * sqrt(T) * exp( - T_HeIion/T ) * nHeI * ne / (1 + constT5 * sqrtT)
end


const Λ_HeIIion0::Float64 = 4.95e-22
const T_HeIIion::Float64  = 631515.0

""" The cooling rate (erg s⁻¹ cm⁻³) due to collisional
    ionization of He II by electrons. """
@inline function Λ_HeIIion( T::Float64,
                        nHeII::Float64,
                           ne::Float64
                            )

    if T ≤ 500.0 || nHeII ≤ nIgnoreCooling
        # Ignore collisional ionization cooling in cold gas 
        # where it is negligible regardless:
        return 0.0
    end

    sqrtT = @fastmath sqrt(T)

    return @fastmath Λ_HeIIion0 * sqrt(T) * exp( - T_HeIIion/T ) * nHeII * ne / (1 + constT5 * sqrtT)
end


######### R E C O M B I N A T I O N S ###########

const Λ_HIIrec0_B::Float64 = 3.435e-30
const Λ_HIIrec0_A::Float64 = 1.778e-29
const T_HIIrec::Float64    = 315614.0

""" The cooling rate (erg s⁻¹ cm⁻³) due to hydrogen
    recombinations: H II + e⁻ → H I + hν. See ref. (1). """
function Λ_HIIrec( T::Float64, 
                nHII::Float64,
                  ne::Float64,
               CaseB::Bool = true
                  )
    λHI = T_HIIrec / max( T, 0.1 )
    if CaseB
        return @fastmath Λ_HIIrec0_B * T * nHII * ne * (λHI^1.97) / (1 + (λHI/2.25)^0.376)^3.72
    else
        return @fastmath Λ_HIIrec0_A * T * nHII * ne * (λHI^1.965) / (1 + (λHI/0.541)^0.502)^2.697
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


const ωHeII0::Float64       = 1.24e-13
const T_HeIIdirec1::Float64 = 470000.0
const T_HeIIdirec2::Float64 = 94000.0

""" The cooling rate (erg s⁻¹ cm⁻³) due to helium
    di-electronic recombinations: He II + e⁻ → He I + hν. 
    See ref. (1). """
@inline function Λ_HeIIdirec( T::Float64,
                          nHeII::Float64,
                             ne::Float64
                             )
    inv_T = 1.0 / max( T, 0.1 )
    ωHeII = @fastmath ωHeII0 * (T^(-1.5)) * exp( - T_HeIIdirec1 * inv_T ) * (1  +  0.3 * exp( - T_HeIIdirec2 * inv_T ))

    return @fastmath ωHeII * nHeII * ne 
end


const Λ_HeIIIrec0_B::Float64 = 8 * 3.435e-30
const Λ_HeIIIrec0_A::Float64 = 8 * 1.778e-29
const T_HeIIIrec::Float64    = 1263030.0

""" The cooling rate (erg s⁻¹ cm⁻³) due to helium
    recombinations: He III + e⁻ → He II + hν. See ref. (1). """
function Λ_HeIIIrec( T::Float64, 
                nHeIII::Float64,
                    ne::Float64,
                 CaseB::Bool = true
                 )
    λHeII = T_HeIIIrec / max( T, 0.1 )
    if CaseB 
        return @fastmath Λ_HeIIIrec0_B * T * (λHeII^1.97) * nHeIII * ne / (1 + (λHeII/2.25)^0.376)^3.72
    else
        return @fastmath Λ_HeIIIrec0_A * T * (λHeII^1.965) * nHeIII * ne / (1 + (λHeII/0.541)^0.502)^2.697
    end
end


######### F R E E - F R E E / B R E M S S T R A H L U N G ###########

const βff0::Float64 = 1.42e-27

""" The cooling rate (erg s⁻¹ cm⁻³) due to free-free emission.
    See ref. (1). """
@inline function Λ_ff( T::Float64,
                    nHII::Float64,
                   nHeII::Float64,
                  nHeIII::Float64,
                      ne::Float64
                      )

    return @fastmath βff0 * sqrt(T) * ne * ( nHII  +  nHeII  +  4 * nHeIII )
end


######### C M B  C O M P T O N  C O O L I N G / H E A T I N G ###########

const Λ_Comp0::Float64 = 5.65e-36

""" The cooling/heating rate (erg s⁻¹ cm⁻³) due to Compton cooling/heating
    from the CMB at redshift z (default = 0).
    See ref. (2). """
@inline function Λ_Comp( T::Float64, 
                        ne::Float64,
                         z::Float64 = 0.0
                          )
    Z       = 1 + z
    TCMB    = @fastmath TCMB0 * Z

    Zsq     = @fastmath Z * Z      # (1 + z)²
    Zfourth = @fastmath Zsq * Zsq  # (1 + z)⁴
    Λ_Comp  = @fastmath Λ_Comp0 * ne * (T - TCMB) * Zfourth

    return Λ_Comp
end


###############################################################
#######                                                 #######
#######        M O L E C U L A R  H Y D R O G E N       #######
#######                  C O O L I N G                  #######
#######                                                 #######
###############################################################

const Λ_H2LTE0::Float64 = 1e-22

# Fitting coefficients for cooling by H₂-H I collisions (Galli & Palla 1998):
const a0_H2HI::Float64 = - 103.0; const a1_H2HI::Float64 = 97.59; const a2_H2HI::Float64 = - 48.05
const a3_H2HI::Float64 = 10.8;    const a4_H2HI::Float64 = - 0.9032

# Fitting coefficients for cooling by H₂-H₂ collision (Glover & Abel 2008, their table 8):
const a0_H2H2::Float64 = - 23.962112; const a1_H2H2::Float64  = 2.09433740; const a2_H2H2::Float64   = - 0.77151436
const a3_H2H2::Float64 =   0.43693353; const a4_H2H2::Float64 = - 0.14913216; const a5_H2H2::Float64 = - 0.033638326

# Fitting coefficients for cooling by H₂-He I collision (Glover & Abel 2008, their table 8):
const a0_H2HeI::Float64 = - 23.689237; const a1_H2HeI::Float64  = 2.1892372; const a2_H2HeI::Float64    = - 0.81520438
const a3_H2HeI::Float64 =   0.29036281; const a4_H2HeI::Float64 = - 0.16596184; const a5_H2HeI::Float64 = 0.19191375

# Fitting coefficients for cooling by H₂-H II collision (Glover & Abel 2008, their table 8):
const a0_H2HII::Float64 = - 21.716699; const a1_H2HII::Float64  = 1.3865783; const a2_H2HII::Float64    = - 0.37915285
const a3_H2HII::Float64 =   0.11453688; const a4_H2HII::Float64 = - 0.23214154; const a5_H2HII::Float64 = 0.058538864

# Fitting coefficients for cooling by H₂-e⁻ collision, at low & higher temperatures
# (Glover & Abel 2008, their table 8):
const a0_H2eLOW::Float64 = - 34.286155; const a1_H2eLOW::Float64 = - 48.537163; const a2_H2eLOW::Float64 = - 77.121176
const a3_H2eLOW::Float64 = - 51.352459; const a4_H2eLOW::Float64 = - 15.169160; const a5_H2eLOW::Float64 = - 0.98120322

const a0_H2eHIGH::Float64 = - 22.190316; const a1_H2eHIGH::Float64  = 1.5728955; const a2_H2eHIGH::Float64    = - 0.21335100
const a3_H2eHIGH::Float64 =   0.96149759; const a4_H2eHIGH::Float64 = - 0.91023195; const a5_H2eHIGH::Float64 =   0.13749749


""" The cooling rate (erg s⁻¹ cm⁻³) due to H₂, taking into account
    collisions with H I, H₂, He I, H II, e⁻. Following ref. (3). """
@inline function Λ_H2( T::Float64,
                     nH2::Float64,
                     nHI::Float64,
                    nHII::Float64,
                    nHeI::Float64, 
                      ne::Float64 
                      )

    if T ≥ 3e4
        # Ignore H₂ cooling in hot gas, which avoids having 
        # to evaluate a lot of algebra below:
        return 0.0
    end

    T  = max( T, 10.0 )      

    # Convenient definitions:
    T3 = @fastmath T * 1e-3  # T / 10³ K
    x  = @fastmath log10(T);  x2 = x*x; x3 = x2*x; x4 = x3*x
    y  = @fastmath log10(T3); y2 = y*y; y3 = y2*y; y4 = y3*y; y5 = y4*y

    # Cooling rate in LTE/high density regime (Hollenbach & McKee 1979):
    Λ_H2LTE = @fastmath Λ_H2LTE0 * nH2 * (6700.0 * exp( - 5.86/T3 ) + 16000.0 * exp( - 11.7/T3 )
              + 0.03 * exp( - 0.51/T3 ) + 9.5 * (T3^3.76) * exp( - 0.0022/(T3^3)) / 
              (1 + 0.12 * (T3^2.1)) ) 

    # Low-density cooling rate due to H₂-H I collisions (Galli & Palla 1998):
    Λ_H2_HI = @fastmath nH2 * nHI * 10^( a0_H2HI + a1_H2HI*x + a2_H2HI*x2 + a3_H2HI*x3 + a4_H2HI*x4 )

    # Low-density cooling rate due to H₂-H₂ collisions (Glover & Abel 2008, their table 8).
    # The fit is valid for 100 K < T ≤ 6000 K.
    if T > 100.0
        # Cooling rate contribution:
        Λ_H2_H2 = @fastmath nH2 * nH2 * 10^( a0_H2H2 + a1_H2H2*y + a2_H2H2*y2 + 
                  a3_H2H2*y3 + a4_H2H2*y4 + a5_H2H2*y5 )
    else
        Λ_H2_H2 = 0.0
    end

    # Low-density cooling rate due H₂-He I collisions (Glover & Abel 2008, their table 8).
    # The fit is valid over 10 K < T < 6000 K.
    Λ_H2_HeI = @fastmath nH2 * nHeI * 10^( a0_H2HeI + a1_H2HeI*y + a2_H2HeI*y2 + 
                   a3_H2HeI*y3 + a4_H2HeI*y4 + a5_H2HeI*y5 )


    # Low-density cooling rate due H₂-H II collisions (Glover & Abel 2008, their table 8).
    # The fit is valid for 10 K < T < 10⁴ K:
    Λ_H2_HII = @fastmath nH2 * nHII * 10^( a0_H2HII + a1_H2HII*y + a2_H2HII*y2 + 
                   a3_H2HII*y3 + a4_H2HII*y4 + a5_H2HII*y5 )


    # Low-density cooling rate due H₂-e⁻ collisions (Glover & Abel 2008, their table 8).
    # The fit is valid for 10 K < T < 10⁴ K:
    if T ≤ 200.0
        # Cooling rate contribution:
        Λ_H2_e = @fastmath nH2 * ne * 10^( a0_H2eLOW + a1_H2eLOW*y + a2_H2eLOW*y2 + 
                 a3_H2eLOW*y3 + a4_H2eLOW*y4 + a5_H2eLOW*y5 )
    else
        # Cooling rate contribution:
        Λ_H2_e = @fastmath nH2 * ne * 10^( a0_H2eHIGH + a1_H2eHIGH*y + a2_H2eHIGH*y2 + 
                 a3_H2eHIGH*y3 + a4_H2eHIGH*y4 + a5_H2eHIGH*y5 )
    end

    # Total low-density H₂-cooling rate:
    Λ_H2_lowdens = @fastmath Λ_H2_HI + Λ_H2_H2 + Λ_H2_HeI + Λ_H2_HII + Λ_H2_e

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


const a0_HD::Float64 = - 42.45906; const a1_HD::Float64 = 21.90083
const a2_HD::Float64 = - 10.1954;  const a3_HD::Float64 = 2.19788
const a4_HD::Float64 = - 0.17286

const D00::Float64 = - 42.57688; const D01::Float64 = 0.92433;   const D02::Float64 = 0.54962
const D03::Float64 = - 0.07676;  const D04::Float64 = 0.00275
const D10::Float64 = 21.93385;   const D11::Float64 = 0.77952;   const D12::Float64 = - 1.06447
const D13::Float64 = 0.11864;    const D14::Float64 = - 0.00366
const D20::Float64 = - 10.19097; const D21::Float64 = - 0.54263; const D22::Float64 = 0.62343   
const D23::Float64 = - 0.07366;  const D24::Float64 = 0.002514
const D30::Float64 = 2.19906;    const D31::Float64 = 0.11711;   const D32::Float64 = - 0.13768
const D33::Float64 = 0.01759;    const D34::Float64 = - 0.000666317
const D40::Float64 = - 0.17334;  const D41::Float64 = - 0.00835; const D42::Float64 = 0.0106
const D43::Float64 = - 0.001482; const D44::Float64 = 0.000061926

const y_HD_cool::Float64  = log10( 1e6 )
const y2_HD_cool::Float64 = y_HD_cool  * y_HD_cool
const y3_HD_cool::Float64 = y2_HD_cool * y_HD_cool
const y4_HD_cool::Float64 = y3_HD_cool * y_HD_cool
const b0_HD_LTE::Float64  = D00 + D01*y_HD_cool + D02*y2_HD_cool + D03*y3_HD_cool + D04*y4_HD_cool
const b1_HD_LTE::Float64  = D10 + D11*y_HD_cool + D12*y2_HD_cool + D13*y3_HD_cool + D14*y4_HD_cool
const b2_HD_LTE::Float64  = D20 + D21*y_HD_cool + D22*y2_HD_cool + D23*y3_HD_cool + D24*y4_HD_cool
const b3_HD_LTE::Float64  = D30 + D31*y_HD_cool + D32*y2_HD_cool + D33*y3_HD_cool + D34*y4_HD_cool
const b4_HD_LTE::Float64  = D40 + D41*y_HD_cool + D42*y2_HD_cool + D43*y3_HD_cool + D44*y4_HD_cool

""" The cooling rate (erg s⁻¹ cm⁻³) due to HD, taking into account
    collisions with H I (collisions with He, H₂, e⁻ are sub-dominant). 
    Following ref. (3). """
@inline function Λ_HD( T::Float64,
                     nHD::Float64,
                     nHI::Float64
                      )

    if T ≥ 3e4 || nHD ≤ nIgnoreCooling 
        # Ignore HD cooling in hot gas, which avoids having 
        # to evaluate a lot of algebra below:
        return 0.0
    end

    T  = max( T, 10.0 )
    x  = @fastmath log10( T ); x2 = x*x; x3 = x2*x; x4 = x3*x
    
    # Cooling rate in LTE/high density regime, following McGreer & Bryan's (2008) version 
    # of the Lipokva+ (2005) fit:
    Λ_HD_LTE = @fastmath nHD * 10^(  b0_HD_LTE + b1_HD_LTE*x + b2_HD_LTE*x2 + b3_HD_LTE*x3 + b4_HD_LTE*x4 )

    # Low-density cooling rate due to HD-H I collisions (Lipokva+ 2005):
    Λ_HD_lowdens = @fastmath nHI * nHD * 10^( a0_HD + a1_HD*x + a2_HD*x2 + a3_HD*x3 + a4_HD*x4 )

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

const Λ_neb0::Float64  = 3.68e-23
const a0_neb::Float64  = 0.692;    const a1_neb::Float64 = - 0.586;  const a2_neb::Float64 = 0.816
const a3_neb::Float64  = - 0.505;  const a4_neb::Float64 = 0.118;    const a5_neb::Float64 = 7.66e-3
const a6_neb::Float64  = - 5.08e-3

""" The cooling rate (erg s⁻¹ cm⁻³) due to collisionally excited
    metal ion lines, e.g. [O II]3729 Å, important in metal-enriched
    photoionized gas. Excluding C II, which is treated separately. 
    See refs. (4)-(5). """
@inline function Λ_neb( T::Float64, 
                     nHII::Float64,
                       ne::Float64,
                    Z̃rest::Float64 = 1.0  # Non-C/Si metallicity relative to Solar abundance of non-C/Si species
                   )
    # Convenient definitions:
    ne2   = clamp( ne * 1e-2, 0.0, 1e10 )  # ne / 10² cm⁻³
    T4    = clamp( T, 1e3, 5e4 ) * 1e-4    # T  / 10⁴ K  

    y     = @fastmath log(T4);  y2 = y*y;  y3 = y2*y;  y4 = y3*y;  y5 = y4*y;  y6 = y5*y

    # Fitting function:
    fneb  = @fastmath 10^( a0_neb + a1_neb*y + a2_neb*y2 + a3_neb*y3 
                         + a4_neb*y4 + a5_neb*y5 + a6_neb*y6 )
    # Cooling rate:
    Λ_neb = @fastmath Λ_neb0 * Z̃rest * ne * nHII * exp(-3.86/T4) * fneb / (
            sqrt(T4) * (1 + 0.12 * ne2^(0.38 - 0.12 * y)) ) 

    return Λ_neb
end


const A_CII10::Float64       = 2.4e-6
const E_CII::Float64         = kB * 91.21
const E_CII_over_kB::Float64 = E_CII / kB       # Convenient, pre-compute 
const A_CII10_E_CII::Float64 = A_CII10 * E_CII  # Convenient, pre-compute 

""" The C II (157.74 μm) cooling rate (erg s⁻¹ cm⁻³). 
    C II is modelled as a two-level atom, using
    rates compiled in Draine (2011). """
function Λ_CII( T::Float64, 
             nCII::Float64,
              nHI::Float64, 
              nH2::Float64, 
             nHeI::Float64, 
               ne::Float64
                )

    # Convenient definitions:
    T       = clamp( T, 1.0, 1e5 )
    T2      = @fastmath T * 1e-2    # T / 10² K
    T4      = @fastmath T * 1e-4    # T / 10⁴ K
    logT2   = @fastmath log(T2)

    # Collisional de-excitation rate coefficients (cm³ s⁻¹) for 
    # H I, H₂(para), H₂(ortho), and He I. See Draine (2011).
    k10_HI  = @fastmath 7.58e-10 * T2^(0.128 + 0.009*logT2)
    k10_H2p = @fastmath 4.25e-10 * T2^(0.124 - 0.018*logT2)  # para-H₂
    k10_H2o = @fastmath 5.14e-10 * T2^(0.095 + 0.023*logT2)  # ortho-H₂
    k10_H2  = @fastmath 0.75 * k10_H2o  +  0.25 * k10_H2p    # (3/4) and (1/3) weights, assuming 3:1 ortho-to-para ratio
    k10_HeI = @fastmath 0.38 * k10_HI
    # Collisional de-excitation rate coefficients for e⁻. 
    # See Draine (2011), p. 196.
    k10_e   = @fastmath 4.53e-8 * T4^(-0.5)

    # Total de-excitation rate:
    C10     = @fastmath k10_HI * nHI  +  k10_H2 * nH2  +  k10_HeI * nHeI  +  k10_e * ne
    # Total excitation rate (using g_u/g_l = 4/2):
    inv_T   = @fastmath 1.0 / T
    C01     = @fastmath 2 * C10 * exp( - E_CII_over_kB * inv_T )
    
    # Cooling rate:
    Λ_CII   = @fastmath nCII * A_CII10_E_CII * C01 / (A_CII10 + C10 + C01)

    return Λ_CII
end


# Einstein A coefficients, transition energies, and statistical weights
# for C I (Draine 2011, p. 198):
const A_CI10::Float64 = 7.93e-8;    const A_CI21::Float64 = 2.68e-7
const E_CI10::Float64 = kB * 23.62; const E_CI21::Float64 = kB * 38.84; const E_CI20::Float64 = E_CI10 + E_CI21

const g_CI0::Float64  = 1;  const inv_g_CI0::Float64 = 1 / g_CI0
const g_CI1::Float64  = 3;  const inv_g_CI1::Float64 = 1 / g_CI1
const g_CI2::Float64  = 5;  const inv_g_CI2::Float64 = 1 / g_CI2

# Convenient constants (pre-compute once):
const A_CI10_E_CI10::Float64  = A_CI10 * E_CI10    # For cooling rates 
const A_CI21_E_CI21::Float64  = A_CI21 * E_CI21 

const E_CI10_over_kB::Float64 = E_CI10 / kB        # In exponentials 
const E_CI20_over_kB::Float64 = E_CI20 / kB 
const E_CI21_over_kB::Float64 = E_CI21 / kB 

const Ratio_g_CI_10::Float64  = g_CI1 * inv_g_CI0
const Ratio_g_CI_20::Float64  = g_CI2 * inv_g_CI0
const Ratio_g_CI_21::Float64  = g_CI2 * inv_g_CI1

""" The C I (609.13 μm, 370.41 μm) cooling rate (erg s⁻¹ cm⁻³). 
    C I is modelled as a 3-level atom, using rates compiled/cited
    in Draine (2011). """
function Λ_CI( T::Float64, 
             nCI::Float64,
            nHII::Float64,
             nHI::Float64, 
             nH2::Float64, 
            nHeI::Float64, 
              ne::Float64
              )

    if T ≥ 5e4 || nCI ≤ nIgnoreCooling
        # Ignore C I cooling at high temperatures where C I is 
        # rare, and where we use CLOUDY cooling rates regardless
        return 0.0
    end

    # Convenient definitions:
    T       = max( T, 1.0 )
    T2      = @fastmath T * 1e-2  # T / 10² K
    T4      = @fastmath T * 1e-4  # T / 10⁴ K
    lnT2    = @fastmath log(T2)

    # H II rates: 
    k10_HII = @fastmath 7.60e-10 * T2^(0.464 - 0.042*lnT2)
    k20_HII = @fastmath 3.02e-10 * T2^(1.217 - 0.126*lnT2)
    k21_HII = @fastmath 2.30e-9  * T2^(0.919 - 0.111*lnT2)

    # H I rates (using the corrected values from Draine's erratum):
    k10_HI  = @fastmath 1.26e-10 * T2^(0.115 + 0.057*lnT2)  # 1 → 0: ³P₁ → ³P₀
    k20_HI  = @fastmath 8.90e-11 * T2^(0.228 + 0.046*lnT2)  # 2 → 0: ³P₂ → ³P₀
    k21_HI  = @fastmath 2.64e-10 * T2^(0.231 + 0.046*lnT2)  # 2 → 1: ³P₂ → ³P₁

    # H₂ rates (assume 3:1 ortho-to-para ratio at the end):
    k10_H2p = @fastmath 0.67e-10 * T2^(-0.085 + 0.102*lnT2)
    k10_H2o = @fastmath 0.71e-10 * T2^(-0.004 + 0.049*lnT2)
    k10_H2  = @fastmath 0.75 * k10_H2o  +  0.25 * k10_H2p

    k20_H2p = @fastmath 0.86e-10 * T2^(-0.010 + 0.048*lnT2)
    k20_H2o = @fastmath 0.69e-10 * T2^(0.169 + 0.038*lnT2)
    k20_H2  = @fastmath 0.75 * k20_H2o  +  0.25 * k20_H2p

    k21_H2p = @fastmath 1.75e-10 * T2^(0.072 + 0.064*lnT2)
    k21_H2o = @fastmath 1.48e-10 * T2^(0.263 + 0.031*lnT2)
    k21_H2  = @fastmath 0.75 * k21_H2o  +  0.25 * k21_H2p

    # He I rates:
    k10_HeI = @fastmath 1.76e-11 * T2^(0.106 - 0.036*lnT2)
    k20_HeI = @fastmath 4.44e-11 * T2^0.029
    k21_HeI = @fastmath 8.33e-11 * T2^(0.113 + 0.029*lnT2)
    
    # Electron rates, using effective collision strength
    # fits from Johnson et al. (1987), as recommended by Draine (2011):
    if T ≥ 7.5 && T ≤ 1e3 
        X  = @fastmath log(T)
        X2 = @fastmath X*X; X3 = @fastmath X2*X; X4 = @fastmath X3*X

        a0_10 = - 0.925e1;     a0_20 = - 0.769735e1; a0_21 = - 0.743870e1
        a1_10 = - 0.925141e1;  a1_20 = - 0.130743e1; a1_21 = - 0.574430
        a2_10 = 0.361184;      a2_20 = 0.697638;     a2_21 = 0.358264
        a3_10 = - 0.150892e-1; a3_20 = - 0.111338;   a3_21 = - 0.418166e-1
        a4_10 = - 0.656325e-3; a4_20 = 0.705277e-2;  a4_21 = 0.235272e-2

        Ω10 = @fastmath exp( a0_10 + a1_10*X + a2_10*X2 + a3_10*X3 + a4_10*X4 )
        Ω20 = @fastmath exp( a0_20 + a1_20*X + a2_20*X2 + a3_20*X3 + a4_20*X4 )
        Ω21 = @fastmath exp( a0_21 + a1_21*X + a2_21*X2 + a3_21*X3 + a4_21*X4 )

        inv_sqrtT4 = @fastmath 1 / sqrt(T4)
        k_prefact  = @fastmath 8.629e-8 * inv_sqrtT4
        k10_e      = @fastmath k_prefact * Ω10 * inv_g_CI1
        k20_e      = @fastmath k_prefact * Ω20 * inv_g_CI2
        k21_e      = @fastmath k_prefact * Ω21 * inv_g_CI2

    elseif T > 1e3 && T ≤ 1e4
        X  = @fastmath log(T)
        X2 = @fastmath X*X; X3 = @fastmath X2*X; X4 = @fastmath X3*X

        a0_10 = 0.444600e3;    a0_20 = 0.350609e3;   a0_21 = 0.386186e3
        a1_10 = - 0.227913e3;  a1_20 = - 0.187474e3; a1_21 = - 0.202192e3
        a2_10 = 0.425952e2;    a2_20 = 0.361803e2;   a2_21 = 0.385049e2
        a3_10 = - 0.347620e1;  a3_20 = - 0.303283e1; a3_21 = - 0.319268e1
        a4_10 = 0.105085;      a4_20 = 0.938138e-1;  a4_21 = 0.978573e-1

        Ω10 = @fastmath exp( a0_10 + a1_10*X + a2_10*X2 + a3_10*X3 + a4_10*X4 )
        Ω20 = @fastmath exp( a0_20 + a1_20*X + a2_20*X2 + a3_20*X3 + a4_20*X4 )
        Ω21 = @fastmath exp( a0_21 + a1_21*X + a2_21*X2 + a3_21*X3 + a4_21*X4 )

        inv_sqrtT4 = @fastmath 1 / sqrt(T4)
        k_prefact  = @fastmath 8.629e-8 * inv_sqrtT4
        k10_e      = @fastmath k_prefact * Ω10 * inv_g_CI1
        k20_e      = @fastmath k_prefact * Ω20 * inv_g_CI2
        k21_e      = @fastmath k_prefact * Ω21 * inv_g_CI2

    else
        k10_e = 0.0
        k20_e = 0.0
        k21_e = 0.0
    end

    # Collect rates:
    C10     = @fastmath k10_HI * nHI  +  k10_HII * nHII  +  k10_H2 * nH2  +  k10_HeI * nHeI  +  k10_e * ne 
    C20     = @fastmath k20_HI * nHI  +  k20_HII * nHII  +  k20_H2 * nH2  +  k20_HeI * nHeI  +  k20_e * ne
    C21     = @fastmath k21_HI * nHI  +  k21_HII * nHII  +  k21_H2 * nH2  +  k21_HeI * nHeI  +  k21_e * ne

    R10     = @fastmath C10 + A_CI10
    R20     = C20
    R21     = @fastmath C21 + A_CI21
    inv_T   = @fastmath 1.0 / T
    R01     = @fastmath Ratio_g_CI_10 * C10 * exp( - E_CI10_over_kB * inv_T )
    R02     = @fastmath Ratio_g_CI_20 * C20 * exp( - E_CI20_over_kB * inv_T )
    R12     = @fastmath Ratio_g_CI_21 * C21 * exp( - E_CI21_over_kB * inv_T ) 

    # Level ratio n1/n0 (Draine 2011, p. 198):
    inv_denom  = @fastmath 1.0 / ( R10*R20 + R10*R21 + R12*R20 )    # Used for both n1/n0 and n2/n0
    n1_n0      = @fastmath ( R01*R20 + R01*R21 + R21*R02 ) * inv_denom
    # Level ratio n2/n0:
    n2_n0      = @fastmath ( R02*R10 + R02*R12 + R12*R01 ) * inv_denom

    # Cooling rate from 609 μm line:
    n0_ntot    = @fastmath 1.0 / ( 1.0 + n1_n0 + n2_n0 )  # Used for both cooling rates 
    n1         = @fastmath nCI * n1_n0 * n0_ntot
    Λ_CI609    = @fastmath n1 * A_CI10_E_CI10

    # Cooling rate from 370 μm line:
    n2         = @fastmath nCI * n2_n0 * n0_ntot
    Λ_CI370    = @fastmath n2 * A_CI21_E_CI21

    return Λ_CI609 + Λ_CI370
end


###############################################################
#######                                                 #######
#######             O X Y G E N  C O O L I N G          #######
#######                                                 #######
###############################################################


# Einstein A coefficients, transition energies, and statistical weights
# for O I (Draine 2011, p. 198):
const A_OI10::Float64 = 8.95e-5;    const A_OI21::Float64 = 1.70e-5
const E_OI10::Float64 = kB * 227.7; const E_OI21::Float64 = kB * 98.90
const E_OI20::Float64 = E_OI10 + E_OI21

const g_OI0::Float64  = 5;  const inv_g_OI0::Float64 = 1 / g_OI0        
const g_OI1::Float64  = 3;  const inv_g_OI1::Float64 = 1 / g_OI1            
const g_OI2::Float64  = 1;  const inv_g_OI2::Float64 = 1 / g_OI2   

# Convenient constants (pre-compute):
const A_OI10_E_OI10::Float64  = A_OI10 * E_OI10    # For cooling rates 
const A_OI21_E_OI21::Float64  = A_OI21 * E_OI21

const E_OI10_over_kB::Float64 = E_OI10 / kB        # For exponentials 
const E_OI20_over_kB::Float64 = E_OI20 / kB
const E_OI21_over_kB::Float64 = E_OI21 / kB

const Ratio_g_OI_10::Float64  = g_OI1 * inv_g_OI0  # Ratios of statistical weights for upward rates 
const Ratio_g_OI_20::Float64  = g_OI2 * inv_g_OI0
const Ratio_g_OI_21::Float64  = g_OI2 * inv_g_OI1

""" The O I (63.18 μm, 145.5 μm) cooling rate (erg s⁻¹ cm⁻³). 
    O I is modelled as a 3-level atom, using rates compiled in Draine (2011). """
function Λ_OI( T::Float64, 
             nOI::Float64,
             nHI::Float64,
             nH2::Float64,
              ne::Float64
                )

    if T ≥ 5e4 || nOI ≤ nIgnoreCooling
        # Ignore O I cooling at high temperatures where O I is 
        # rare, and where we use CLOUDY cooling rates regardless
        return 0.0
    end

    # Convenient definitions:
    T       = max( T, 1.0 )
    T2      = @fastmath T * 1e-2  # T / 100 K
    T4      = @fastmath T * 1e-4  # T / 10⁴ K
    lnT2    = @fastmath log(T2)
    lnT4    = @fastmath log(T4)

    # H I rates:
    k10_HI  = @fastmath 3.57e-10 * T2^(0.419 - 0.003*lnT2)
    k20_HI  = @fastmath 3.19e-10 * T2^(0.369 - 0.006*lnT2)
    k21_HI  = @fastmath 4.34e-10 * T2^(0.755 - 0.160*lnT2)

    # H₂ rates (assume 3:1 ortho-to-para ratio at the end), 
    # corrected following Draine's erratum:
    k10_H2p = @fastmath 1.49e-10 * T2^(0.369 - 0.026*lnT2)
    k10_H2o = @fastmath 1.37e-10 * T2^(0.395 - 0.005*lnT2)
    k10_H2  = @fastmath 0.75 * k10_H2o  +  0.25 * k10_H2p

    k20_H2p = @fastmath 2.37e-10 * T2^(0.255 + 0.016*lnT2)
    k20_H2o = @fastmath 2.23e-10 * T2^(0.284 + 0.035*lnT2)
    k20_H2  = @fastmath 0.75 * k20_H2o  +  0.25 * k20_H2p

    k21_H2p = @fastmath 2.10e-12 * T2^(1.117 + 0.070*lnT2)
    k21_H2o = @fastmath 3.00e-12 * T2^(0.792 + 0.188*lnT2)
    k21_H2  = @fastmath 0.75 * k21_H2o  +  0.25 * k21_H2p

    # Electron collisional strengths, updated following Draine's erratum: 
    Ω10        = @fastmath 0.0105  * (T4^(0.4861 + 0.0054*lnT4))
    Ω20        = @fastmath 0.00459 * (T4^(0.4507 - 0.0066*lnT4))
    Ω21        = @fastmath 0.00015 * (T4^(0.4709 - 0.1396*lnT4))

    # Gather electron collisional de-excitation rates:
    inv_sqrtT4 = @fastmath 1 / sqrt(T4)
    k_prefact  = @fastmath 8.629e-8  * inv_sqrtT4
    k10_e      = @fastmath k_prefact * Ω10 * inv_g_OI1
    k20_e      = @fastmath k_prefact * Ω20 * inv_g_OI2
    k21_e      = @fastmath k_prefact * Ω21 * inv_g_OI2

    # Collect rates:
    C10     = @fastmath k10_HI * nHI  +  k10_H2 * nH2  +  k10_e * ne 
    C20     = @fastmath k20_HI * nHI  +  k20_H2 * nH2  +  k20_e * ne 
    C21     = @fastmath k21_HI * nHI  +  k21_H2 * nH2  +  k21_e * ne

    R10     = @fastmath C10 + A_OI10
    R20     = C20
    R21     = @fastmath C21 + A_OI21
    inv_T   = @fastmath 1.0 / T
    R01     = @fastmath Ratio_g_OI_10 * C10 * exp( - E_OI10_over_kB * inv_T )
    R02     = @fastmath Ratio_g_OI_20 * C20 * exp( - E_OI20_over_kB * inv_T )
    R12     = @fastmath Ratio_g_OI_21 * C21 * exp( - E_OI21_over_kB * inv_T ) 

    # Level ratio n1/n0 (Draine 2011, p. 198):
    inv_denom = @fastmath 1.0 / ( R10*R20 + R10*R21 + R12*R20)      # Used for both n1/n0 and n2/n0
    n1_n0     = @fastmath ( R01*R20 + R01*R21 + R21*R02 ) * inv_denom
    # Level ratio n2/n0:
    n2_n0     = @fastmath ( R02*R10 + R02*R12 + R12*R01 ) * inv_denom

    # Cooling rate from 63 μm line:
    n0_ntot   = @fastmath 1.0 / ( 1.0 + n1_n0 + n2_n0 )  # Used for both cooling rates 
    n1        = @fastmath nOI * n1_n0 * n0_ntot
    Λ_OI63    = @fastmath n1 * A_OI10_E_OI10

    # Cooling rate from 146 μm line:
    n2        = @fastmath nOI * n2_n0 * n0_ntot
    Λ_OI146   = @fastmath n2 * A_OI21_E_OI21

    return @fastmath Λ_OI63 + Λ_OI146
end


# O II Einstein A coefficients (s⁻¹) from the CHIANTI database (wavelengths 
# are from Draine 2011, specifically the erratum corrected one on Draine's webpage.
# Einstein A coefficients from CHIANTI differ slightly from those of Tielens 2005, but are similar,
# which is a good sanity check):
const A_OII_10::Float64 = 4.124e-5   # Einstein A coefficient for 3729.8 Å line 
const A_OII_20::Float64 = 1.635e-4   # Einstein A coefficient for 3727.1 Å line 
const A_OII_30::Float64 = 5.646e-2   # Einstein A coefficient for 2471.1 Å line 
const A_OII_40::Float64 = 2.265e-2   # Einstein A coefficient for 2471.0 Å line 
const A_OII_21::Float64 = 1.320e-7   # Einstein A coefficient for 500 μm line (very small)
const A_OII_31::Float64 = 1.106e-1   # Einstein A coefficient for 7322.2 Å line 
const A_OII_32::Float64 = 5.871e-2   # Einstein A coefficient for 7332.8 Å line 
const A_OII_41::Float64 = 5.824e-2   # Einstein A coefficient for 7320.9 Å line 
const A_OII_42::Float64 = 9.668e-2   # Einstein A coefficient for 7331.7 Å line 

# O II energy levels (Draine 2011, as corrected in his erratum, gives values of E / hc, 
# with E = 0 in the ground state level 0):
const E_OII_1::Float64  = h * c * 26811.0
const E_OII_2::Float64  = h * c * 26831.0
const E_OII_3::Float64  = h * c * 40468.0
const E_OII_4::Float64  = h * c * 40470.0

# O II line transition energies & values of A_ul × E_ul (pre-compute for cooling rates):
const E_OII_10::Float64 = E_OII_1
const E_OII_20::Float64 = E_OII_2
const E_OII_30::Float64 = E_OII_3 
const E_OII_40::Float64 = E_OII_4   
const E_OII_21::Float64 = E_OII_2 - E_OII_1 
const E_OII_31::Float64 = E_OII_3 - E_OII_1 
const E_OII_41::Float64 = E_OII_4 - E_OII_1 
const E_OII_32::Float64 = E_OII_3 - E_OII_2 
const E_OII_42::Float64 = E_OII_4 - E_OII_2
const E_OII_43::Float64 = E_OII_4 - E_OII_3

const A_OII_10_E_OII_10::Float64 = A_OII_10 * E_OII_10
const A_OII_20_E_OII_20::Float64 = A_OII_20 * E_OII_20
const A_OII_30_E_OII_30::Float64 = A_OII_30 * E_OII_30
const A_OII_40_E_OII_40::Float64 = A_OII_40 * E_OII_40
const A_OII_21_E_OII_21::Float64 = A_OII_21 * E_OII_21
const A_OII_31_E_OII_31::Float64 = A_OII_31 * E_OII_31
const A_OII_32_E_OII_32::Float64 = A_OII_32 * E_OII_32
const A_OII_41_E_OII_41::Float64 = A_OII_41 * E_OII_41
const A_OII_42_E_OII_42::Float64 = A_OII_42 * E_OII_42

# O II line transition energies in temperature units (K):
const E_OII_10_over_kB::Float64 = E_OII_1 / kB
const E_OII_20_over_kB::Float64 = E_OII_2 / kB 
const E_OII_30_over_kB::Float64 = E_OII_3 / kB
const E_OII_40_over_kB::Float64 = E_OII_4 / kB   
const E_OII_21_over_kB::Float64 = ( E_OII_2 - E_OII_1 ) / kB 
const E_OII_31_over_kB::Float64 = ( E_OII_3 - E_OII_1 ) / kB
const E_OII_41_over_kB::Float64 = ( E_OII_4 - E_OII_1 ) / kB
const E_OII_32_over_kB::Float64 = ( E_OII_3 - E_OII_2 ) / kB
const E_OII_42_over_kB::Float64 = ( E_OII_4 - E_OII_2 ) / kB
const E_OII_43_over_kB::Float64 = ( E_OII_4 - E_OII_3 ) / kB

# Statistical weights & their inverses (g = 2J + 1, also available at e.g. NIST):
const g_OII_0::Float64  = 4.0; const inv_g_OII_0::Float64 = 1 / g_OII_0  # ⁴S_{3/2}
const g_OII_1::Float64  = 6.0; const inv_g_OII_1::Float64 = 1 / g_OII_1  # ²D_{5/2}
const g_OII_2::Float64  = 4.0; const inv_g_OII_2::Float64 = 1 / g_OII_2  # ²D_{3/2}
const g_OII_3::Float64  = 4.0; const inv_g_OII_3::Float64 = 1 / g_OII_3  # ²P_{3/2}
const g_OII_4::Float64  = 2.0; const inv_g_OII_4::Float64 = 1 / g_OII_4  # ²P_{1/2}

# Ratios of statistical weights for upward rates: 
const Ratio_g_OII_10::Float64 = g_OII_1 * inv_g_OII_0
const Ratio_g_OII_20::Float64 = g_OII_2 * inv_g_OII_0
const Ratio_g_OII_30::Float64 = g_OII_3 * inv_g_OII_0
const Ratio_g_OII_40::Float64 = g_OII_4 * inv_g_OII_0
const Ratio_g_OII_21::Float64 = g_OII_2 * inv_g_OII_1
const Ratio_g_OII_31::Float64 = g_OII_3 * inv_g_OII_1
const Ratio_g_OII_41::Float64 = g_OII_4 * inv_g_OII_1
const Ratio_g_OII_32::Float64 = g_OII_3 * inv_g_OII_2
const Ratio_g_OII_42::Float64 = g_OII_4 * inv_g_OII_2
const Ratio_g_OII_43::Float64 = g_OII_4 * inv_g_OII_3

""" The O II (3729.8 Å, 3727.1 Å, 2471.1 Å, 2471.0 Å, 7322.2 Å, 7332.8 Å, 
    7320.9 Å, 7331.7 Å, 500 μm) cooling rate (erg s⁻¹ cm⁻³). O II is modelled as 
    a 5-level atom, using electron collisional rates compiled in Draine (2011), and 
    Einstein A coefficients from CHIANTI. Collisions with neutrals are irrelevant 
    in the regions where O II exist. """
function Λ_OII( T::Float64, 
             nOII::Float64, 
               ne::Float64 )

    if T ≤ 300.0 || nOII ≤ nIgnoreCooling || ne ≤ 0.0
        # Ignore O II cooling in cold gas (or where there is no O II or electrons), 
        # which avoids having to evaluate solve for the level populations etc in regions 
        # where O II cooling is negligible:
        return 0.0
    end

    T          = min( T, 1e9 )
    T4         = @fastmath T * 1e-4
    lnT4       = @fastmath log(T4)

    # Electron collisional strengths (Draine 2011, p. 499):
    Ω10        = @fastmath 0.803 * (T4^(0.023    -  0.008 * lnT4))   # 1 → 0: ²D_{5/2} → ⁴S_{3/2}
    Ω20        = @fastmath 0.550 * (T4^(0.054    -  0.004 * lnT4))   # 2 → 0: ²D_{3/2} → ⁴S_{3/2}
    Ω21        = @fastmath 1.434 * (T4^(- 0.176  +  0.004 * lnT4))   # 2 → 1: ²D_{3/2} → ²D_{5/2}
    Ω30        = @fastmath 0.140 * (T4^(0.025    -  0.006 * lnT4))   # 3 → 0: ²P_{3/2} → ⁴S_{3/2} 
    Ω40        = @fastmath 0.283 * (T4^(0.023    -  0.004 * lnT4))   # 4 → 0: ²P_{1/2} → ⁴S_{3/2} 
    Ω31        = @fastmath 0.349 * (T4^(0.060    +  0.052 * lnT4))   # 3 → 1: ²P_{3/2} → ²D_{5/2} 
    Ω41        = @fastmath 0.832 * (T4^(0.076    +  0.055 * lnT4))   # 4 → 1: ²P_{1/2} → ²D_{5/2} 
    Ω32        = @fastmath 0.326 * (T4^(0.063    +  0.052 * lnT4))   # 3 → 2: ²P_{3/2} → ²D_{3/2}
    Ω42        = @fastmath 0.485 * (T4^(0.059    +  0.052 * lnT4))   # 4 → 2: ²P_{1/2} → ²D_{3/2}
    Ω43        = @fastmath 0.322 * (T4^(0.019    +  0.037 * lnT4))   # 4 → 3: ²P_{1/2} → ²P_{3/2}

    # Electron collisional rate coefficients:
    inv_sqrtT4 = @fastmath 1.0 / sqrt(T4)
    kprefact   = @fastmath 8.629e-8 * inv_sqrtT4
    k10_e      = @fastmath kprefact * Ω10 * inv_g_OII_1
    k20_e      = @fastmath kprefact * Ω20 * inv_g_OII_2
    k21_e      = @fastmath kprefact * Ω21 * inv_g_OII_2
    k30_e      = @fastmath kprefact * Ω30 * inv_g_OII_3
    k31_e      = @fastmath kprefact * Ω31 * inv_g_OII_3
    k32_e      = @fastmath kprefact * Ω32 * inv_g_OII_3
    k40_e      = @fastmath kprefact * Ω40 * inv_g_OII_4
    k41_e      = @fastmath kprefact * Ω41 * inv_g_OII_4
    k42_e      = @fastmath kprefact * Ω42 * inv_g_OII_4
    k43_e      = @fastmath kprefact * Ω43 * inv_g_OII_4

    # Collect collisional de-excitation rates:
    C40        = @fastmath k40_e * ne
    C41        = @fastmath k41_e * ne  
    C42        = @fastmath k42_e * ne  
    C43        = @fastmath k43_e * ne  
    C30        = @fastmath k30_e * ne  
    C31        = @fastmath k31_e * ne 
    C32        = @fastmath k32_e * ne  
    C20        = @fastmath k20_e * ne  
    C21        = @fastmath k21_e * ne  
    C10        = @fastmath k10_e * ne  

    # Collect downward rates (Rij means the rate in s⁻¹ from level i → j):
    R40        = @fastmath C40  +  A_OII_40
    R41        = @fastmath C41  +  A_OII_41 
    R42        = @fastmath C42  +  A_OII_42
    R43        = C43                         # No spontaneous decay from 4 to 3
    R32        = @fastmath C32  +  A_OII_32
    R31        = @fastmath C31  +  A_OII_31
    R30        = @fastmath C30  +  A_OII_30
    R21        = @fastmath C21  +  A_OII_21
    R20        = @fastmath C20  +  A_OII_20
    R10        = @fastmath C10  +  A_OII_10

    # Collect upward rates:
    inv_T      = @fastmath 1.0 / T
    R01        = @fastmath Ratio_g_OII_10 * C10 * exp( - E_OII_10_over_kB * inv_T )
    R02        = @fastmath Ratio_g_OII_20 * C20 * exp( - E_OII_20_over_kB * inv_T )
    R03        = @fastmath Ratio_g_OII_30 * C30 * exp( - E_OII_30_over_kB * inv_T )
    R04        = @fastmath Ratio_g_OII_40 * C40 * exp( - E_OII_40_over_kB * inv_T )
    R12        = @fastmath Ratio_g_OII_21 * C21 * exp( - E_OII_21_over_kB * inv_T )
    R13        = @fastmath Ratio_g_OII_31 * C31 * exp( - E_OII_31_over_kB * inv_T )
    R14        = @fastmath Ratio_g_OII_41 * C41 * exp( - E_OII_41_over_kB * inv_T )
    R23        = @fastmath Ratio_g_OII_32 * C32 * exp( - E_OII_32_over_kB * inv_T )
    R24        = @fastmath Ratio_g_OII_42 * C42 * exp( - E_OII_42_over_kB * inv_T )
    R34        = @fastmath Ratio_g_OII_43 * C43 * exp( - E_OII_43_over_kB * inv_T )

    # Rates of de-population of levels 0 - 3:
    D0         = @fastmath R01  +  R02  +  R03  +  R04
    D1         = @fastmath R10  +  R12  +  R13  +  R14
    D2         = @fastmath R20  +  R21  +  R23  +  R24
    D3         = @fastmath R30  +  R31  +  R32  +  R34

    # Now we set up the linear system of equations to solve for the level populations. 
    # We define x = (x0, x1, x2, x3), where xi = ni / nOII. We eliminate n4 using 
    # n4 = nOII - n0 - n1 - n2 - n3, i.e. x4 = 1 - x0 - x1 - x2 - x3. Then we only 
    # need to solve for levels 0, 1, 2 and 3, stored in x. The matrix equation to solve 
    # is then Ax = b, where:
    A_00  = - (D0 + R40);  A_01 = R10 - R40;     A_02 = R20 - R40;    A_03 = R30 - R40 
    A_10  = R01 - R41;     A_11 = - (D1 + R41);  A_12 = R21 - R41;    A_13 = R31 - R41 
    A_20  = R02 - R42;     A_21 = R12 - R42;     A_22 = - (D2 + R42); A_23 = R32 - R42
    A_30  = R03 - R43;     A_31 = R13 - R43;     A_32 = R23 - R43;    A_33 = - (D3 + R43)

    A     = @SMatrix [ A_00   A_01  A_02  A_03;
                       A_10   A_11  A_12  A_13;
                       A_20   A_21  A_22  A_23;
                       A_30   A_31  A_32  A_33 ]

    b     = @SVector [ - R40, - R41, - R42, - R43 ]
    
    # Solve for x = (x0, x1, x2, x3) (NOTE: For a 4 x 4 system, I have found that 
    # inv(A) * b is ~ 3 × faster than A \ b, with no accuracy loss. For smaller systems 
    # of equations, there is not much difference, and for larger systems of equations A \ b 
    # is faster. So for this special case inv(A) is the preferred option):
    x     = inv(A) * b

    # Unpack & get level population densities:
    x0, x1, x2, x3 = x 
    n0 = @fastmath x0 * nOII
    n1 = @fastmath x1 * nOII
    n2 = @fastmath x2 * nOII
    n3 = @fastmath x3 * nOII 
    n4 = max( 1.0 - x0 - x1 - x2 - x3, 0.0 ) * nOII

    # [O II] 3729.8 Å cooling rate (1 → 0):
    Λ_OII_3729   = @fastmath n1 * A_OII_10_E_OII_10

    # [O II] 3727.1 Å cooling rate (2 → 0):
    Λ_OII_3727   = @fastmath n2 * A_OII_20_E_OII_20

    # [O II] 2471.1 Å cooling rate (3 → 0):
    Λ_OII_2471_A = @fastmath n3 * A_OII_30_E_OII_30

    # [O II] 2471.0 Å cooling rate (4 → 0):
    Λ_OII_2471_B = @fastmath n4 * A_OII_40_E_OII_40

    # [O II] 7322.2 Å cooling rate (3 → 1):
    Λ_OII_7322   = @fastmath n3 * A_OII_31_E_OII_31

    # [O II] 7332.8 Å cooling rate (3 → 2):
    Λ_OII_7333   = @fastmath n3 * A_OII_32_E_OII_32

    # [O II] 7320.9 Å cooling rate (4 → 1):
    Λ_OII_7321   = @fastmath n4 * A_OII_41_E_OII_41

    # [O II] 7331.7 Å cooling rate (4 → 2):
    Λ_OII_7332   = @fastmath n4 * A_OII_42_E_OII_42

    # [O II] 500 μm cooling rate (2 → 1):
    Λ_OII_500μm  = @fastmath n2 * A_OII_21_E_OII_21

    # Total O II cooling rate:
    Λ_OII_tot = @fastmath ( Λ_OII_3729 + Λ_OII_3727 + Λ_OII_2471_A + Λ_OII_2471_B + Λ_OII_7322 
                          + Λ_OII_7333 + Λ_OII_7321 + Λ_OII_7332   + Λ_OII_500μm )

    return Λ_OII_tot 
end


# O III Einstein A coefficients (s⁻¹) from the CHIANTI database (also see 
# Tielens 2005, p. 55 for similar values), which best reproduces the 5008/4960 
# line ratio of Storey & Zeippen (2000):
const A_OIII_32::Float64 = 2.046e-2  # Einstein A coefficient for 5008 Å line 
const A_OIII_31::Float64 = 6.791e-3  # Einstein A coefficient for 4960 Å line
const A_OIII_10::Float64 = 2.597e-5  # Einstein A coefficient for 88.36 μm line 
const A_OIII_21::Float64 = 9.632e-5  # Einstein A coefficient for 51.81 μm line 
const A_OIII_43::Float64 = 1.685     # Einstein A coefficient for 4364.4 Å line 
const A_OIII_41::Float64 = 0.2255    # Einstein A coefficient for 2321.7 Å line  

# O III level energies (Draine 2011, p. 487 gives values of E / hc, with E = 0 
# in the ground state level 0): 
const E_OIII_1::Float64  = h * c * 113.2 
const E_OIII_2::Float64  = h * c * 306.2
const E_OIII_3::Float64  = h * c * 20273.0
const E_OIII_4::Float64  = h * c * 43186.0

# O III line transition energies & values of A_ul × E_ul (pre-compute for cooling rates):
const E_OIII_10::Float64 = E_OIII_1
const E_OIII_20::Float64 = E_OIII_2
const E_OIII_30::Float64 = E_OIII_3 
const E_OIII_40::Float64 = E_OIII_4   
const E_OIII_21::Float64 = E_OIII_2 - E_OIII_1 
const E_OIII_31::Float64 = E_OIII_3 - E_OIII_1 
const E_OIII_41::Float64 = E_OIII_4 - E_OIII_1 
const E_OIII_32::Float64 = E_OIII_3 - E_OIII_2 
const E_OIII_42::Float64 = E_OIII_4 - E_OIII_2
const E_OIII_43::Float64 = E_OIII_4 - E_OIII_3

const A_OIII_32_E_OIII_32::Float64 = A_OIII_32 * E_OIII_32
const A_OIII_31_E_OIII_31::Float64 = A_OIII_31 * E_OIII_31
const A_OIII_43_E_OIII_43::Float64 = A_OIII_43 * E_OIII_43
const A_OIII_41_E_OIII_41::Float64 = A_OIII_41 * E_OIII_41
const A_OIII_10_E_OIII_10::Float64 = A_OIII_10 * E_OIII_10
const A_OIII_21_E_OIII_21::Float64 = A_OIII_21 * E_OIII_21

# O III line transition energies in temperature units (K):
const E_OIII_10_over_kB::Float64 = E_OIII_1 / kB
const E_OIII_20_over_kB::Float64 = E_OIII_2 / kB 
const E_OIII_30_over_kB::Float64 = E_OIII_3 / kB
const E_OIII_40_over_kB::Float64 = E_OIII_4 / kB   
const E_OIII_21_over_kB::Float64 = ( E_OIII_2 - E_OIII_1 ) / kB 
const E_OIII_31_over_kB::Float64 = ( E_OIII_3 - E_OIII_1 ) / kB
const E_OIII_41_over_kB::Float64 = ( E_OIII_4 - E_OIII_1 ) / kB
const E_OIII_32_over_kB::Float64 = ( E_OIII_3 - E_OIII_2 ) / kB
const E_OIII_42_over_kB::Float64 = ( E_OIII_4 - E_OIII_2 ) / kB
const E_OIII_43_over_kB::Float64 = ( E_OIII_4 - E_OIII_3 ) / kB

# Level statistical weights g = 2J + 1, & their inverses 
# (see e.g. fig. 1 in Yang & Lidz 2020):
const g_OIII_0::Float64  = 1.0; const inv_g_OIII_0::Float64 = 1 / g_OIII_0
const g_OIII_1::Float64  = 3.0; const inv_g_OIII_1::Float64 = 1 / g_OIII_1
const g_OIII_2::Float64  = 5.0; const inv_g_OIII_2::Float64 = 1 / g_OIII_2
const g_OIII_3::Float64  = 5.0; const inv_g_OIII_3::Float64 = 1 / g_OIII_3
const g_OIII_4::Float64  = 1.0; const inv_g_OIII_4::Float64 = 1 / g_OIII_4

# Ratios of statistical weights for upward rates: 
const Ratio_g_OIII_10::Float64 = g_OIII_1 * inv_g_OIII_0
const Ratio_g_OIII_20::Float64 = g_OIII_2 * inv_g_OIII_0
const Ratio_g_OIII_30::Float64 = g_OIII_3 * inv_g_OIII_0
const Ratio_g_OIII_40::Float64 = g_OIII_4 * inv_g_OIII_0
const Ratio_g_OIII_21::Float64 = g_OIII_2 * inv_g_OIII_1
const Ratio_g_OIII_31::Float64 = g_OIII_3 * inv_g_OIII_1
const Ratio_g_OIII_41::Float64 = g_OIII_4 * inv_g_OIII_1
const Ratio_g_OIII_32::Float64 = g_OIII_3 * inv_g_OIII_2
const Ratio_g_OIII_42::Float64 = g_OIII_4 * inv_g_OIII_2
const Ratio_g_OIII_43::Float64 = g_OIII_4 * inv_g_OIII_3

""" The O III (5008 Å, 4960 Å, 88.36 μm, 51.81 μm, 4364.4 Å, 2321.7 Å) cooling rate 
    (erg s⁻¹ cm⁻³). O III is modelled as a 5-level atom, using electron collisional 
    rates compiled in Draine (2011), and Einstein A coefficients from CHIANTI. 
    Collisions with neutrals are irrelevant in the regions where O III exist. 

    NOTE: Both the total O III cooling rate, and the cooling from [O III] 5008 Å + 4960 Å,
          and 4364 Å are returned. The latter is useful for modelling the line RT
          for mock data/observational predictions. """
function Λ_OIII( T::Float64, 
             nOIII::Float64,
                ne::Float64 )

    if T ≤ 1e3 || nOIII ≤ nIgnoreCooling || ne ≤ 0.0
        # Ignore O III cooling in cold gas (or where there is no O III or electrons), 
        # which avoids having to solve for the level populations etc in regions 
        # where O III cooling is negligible:
        return 0.0, 0.0, 0.0
    end

    T          = min( T, 1e9 )
    T4         = @fastmath T * 1e-4
    lnT4       = @fastmath log(T4)

    # Electron collisional strengths (Draine 2011, p. 497):
    Ω10        = @fastmath 0.522 * (T4^(0.033  -  0.009 * lnT4)) 
    Ω20        = @fastmath 0.257 * (T4^(0.081  +  0.017 * lnT4))
    Ω21        = @fastmath 1.23  * (T4^(0.053  +  0.007 * lnT4))

    Ω3stuff    = @fastmath 0.243 * (T4^(0.120  +  0.031 * lnT4))
    Ω30        = Ω3stuff                  # J = 0 → (2J + 1) = 1
    Ω31        = @fastmath 3.0 * Ω3stuff  # J = 1 → (2J + 1) = 3
    Ω32        = @fastmath 5.0 * Ω3stuff  # J = 2 → (2J + 1) = 5

    Ω4stuff    = @fastmath 0.0321 * (T4^(0.118  +  0.057 * lnT4))
    Ω40        = Ω4stuff                  # J = 0 → (2J + 1) = 1
    Ω41        = @fastmath 3.0 * Ω4stuff  # J = 1 → (2J + 1) = 3
    Ω42        = @fastmath 5.0 * Ω4stuff  # J = 2 → (2J + 1) = 5
    Ω43        = @fastmath 0.523 * (T4^(0.210  -  0.099 * lnT4))

    # Electron collisional rate coefficients:
    inv_sqrtT4 = @fastmath 1.0 / sqrt(T4)
    kprefact   = @fastmath 8.629e-8 * inv_sqrtT4
    k10_e      = @fastmath kprefact * Ω10 * inv_g_OIII_1
    k20_e      = @fastmath kprefact * Ω20 * inv_g_OIII_2
    k21_e      = @fastmath kprefact * Ω21 * inv_g_OIII_2
    k30_e      = @fastmath kprefact * Ω30 * inv_g_OIII_3
    k31_e      = @fastmath kprefact * Ω31 * inv_g_OIII_3
    k32_e      = @fastmath kprefact * Ω32 * inv_g_OIII_3
    k40_e      = @fastmath kprefact * Ω40 * inv_g_OIII_4
    k41_e      = @fastmath kprefact * Ω41 * inv_g_OIII_4
    k42_e      = @fastmath kprefact * Ω42 * inv_g_OIII_4
    k43_e      = @fastmath kprefact * Ω43 * inv_g_OIII_4

    # Collect collisional de-excitation rates:
    C40        = @fastmath k40_e * ne
    C41        = @fastmath k41_e * ne  
    C42        = @fastmath k42_e * ne  
    C43        = @fastmath k43_e * ne  
    C30        = @fastmath k30_e * ne  
    C31        = @fastmath k31_e * ne 
    C32        = @fastmath k32_e * ne  
    C20        = @fastmath k20_e * ne  
    C21        = @fastmath k21_e * ne  
    C10        = @fastmath k10_e * ne  

    # Collect downward rates (Rij means the rate in s⁻¹ from level i → j):
    R40        = C40
    R41        = @fastmath C41  +  A_OIII_41 
    R42        = C42  
    R43        = @fastmath C43  +  A_OIII_43
    R32        = @fastmath C32  +  A_OIII_32
    R31        = @fastmath C31  +  A_OIII_31
    R30        = C30  
    R21        = @fastmath C21  +  A_OIII_21
    R20        = C20
    R10        = @fastmath C10  +  A_OIII_10

    # Collect upward rates:
    inv_T      = @fastmath 1.0 / T
    R01        = @fastmath Ratio_g_OIII_10 * C10 * exp( - E_OIII_10_over_kB * inv_T )
    R02        = @fastmath Ratio_g_OIII_20 * C20 * exp( - E_OIII_20_over_kB * inv_T )
    R03        = @fastmath Ratio_g_OIII_30 * C30 * exp( - E_OIII_30_over_kB * inv_T )
    R04        = @fastmath Ratio_g_OIII_40 * C40 * exp( - E_OIII_40_over_kB * inv_T )
    R12        = @fastmath Ratio_g_OIII_21 * C21 * exp( - E_OIII_21_over_kB * inv_T )
    R13        = @fastmath Ratio_g_OIII_31 * C31 * exp( - E_OIII_31_over_kB * inv_T )
    R14        = @fastmath Ratio_g_OIII_41 * C41 * exp( - E_OIII_41_over_kB * inv_T )
    R23        = @fastmath Ratio_g_OIII_32 * C32 * exp( - E_OIII_32_over_kB * inv_T )
    R24        = @fastmath Ratio_g_OIII_42 * C42 * exp( - E_OIII_42_over_kB * inv_T )
    R34        = @fastmath Ratio_g_OIII_43 * C43 * exp( - E_OIII_43_over_kB * inv_T )

    # Rates of de-population of levels 0 - 3:
    D0         = @fastmath R01  +  R02  +  R03  +  R04
    D1         = @fastmath R10  +  R12  +  R13  +  R14
    D2         = @fastmath R20  +  R21  +  R23  +  R24
    D3         = @fastmath R30  +  R31  +  R32  +  R34

    # Now we set up the linear system of equations to solve for the level populations. 
    # We define x = (x0, x1, x2, x3), where xi = ni / nOIII. We eliminate n4 using 
    # n4 = nOIII - n0 - n1 - n2 - n3, i.e. x4 = 1 - x0 - x1 - x2 - x3. Then we only 
    # need to solve for levels 0, 1, 2 and 3, stored in x. The matrix equation to solve 
    # is then Ax = b, where:
    A_00  = - (D0 + R40);  A_01 = R10 - R40;     A_02 = R20 - R40;    A_03 = R30 - R40 
    A_10  = R01 - R41;     A_11 = - (D1 + R41);  A_12 = R21 - R41;    A_13 = R31 - R41 
    A_20  = R02 - R42;     A_21 = R12 - R42;     A_22 = - (D2 + R42); A_23 = R32 - R42
    A_30  = R03 - R43;     A_31 = R13 - R43;     A_32 = R23 - R43;    A_33 = - (D3 + R43)

    A     = @SMatrix [ A_00   A_01  A_02  A_03;
                       A_10   A_11  A_12  A_13;
                       A_20   A_21  A_22  A_23;
                       A_30   A_31  A_32  A_33 ]

    b     = @SVector [ - R40, - R41, - R42, - R43 ]

    # Solve for x = (x0, x1, x2, x3) (NOTE: For a 4 x 4 system, I have found that 
    # inv(A) * b is ~ 3 × faster than A \ b, with no accuracy loss. For smaller systems 
    # of equations, there is not much difference, and for larger systems of equations A \ b 
    # is faster. So for this special case inv(A) is the preferred option):
    x     = inv(A) * b

    # Unpack & get level population densities:
    x0, x1, x2, x3 = x 
    n0 = @fastmath x0 * nOIII
    n1 = @fastmath x1 * nOIII
    n2 = @fastmath x2 * nOIII
    n3 = @fastmath x3 * nOIII 
    n4 = max( 1.0 - x0 - x1 - x2 - x3, 0.0 ) * nOIII 

    # [O III] 5008 Å cooling rate (3 → 2):
    Λ_OIII_5008 = @fastmath n3 * A_OIII_32_E_OIII_32

    # [O III] 4960 Å cooling rate (3 → 1):
    Λ_OIII_4960 = @fastmath n3 * A_OIII_31_E_OIII_31

    # [O III] 4364 Å cooling rate (4 → 3):
    Λ_OIII_4364 = @fastmath n4 * A_OIII_43_E_OIII_43

    # [O III] 2322 Å cooling rate (4 → 1):
    Λ_OIII_2322 = @fastmath n4 * A_OIII_41_E_OIII_41

    # [O III] 88 μm cooling rate (1 → 0):
    Λ_OIII_88μm = @fastmath n1 * A_OIII_10_E_OIII_10

    # [O III] 52 μm cooling rate (2 → 1):
    Λ_OIII_52μm = @fastmath n2 * A_OIII_21_E_OIII_21

    # [O III] 5008 Å + 4960 Å cooling rate:
    Λ_OIII_5008and4960 = @fastmath Λ_OIII_5008 + Λ_OIII_4960

    # Total O III cooling rate:
    Λ_OIII_tot  = @fastmath ( Λ_OIII_5008and4960 + Λ_OIII_4364 
                + Λ_OIII_2322 + Λ_OIII_88μm + Λ_OIII_52μm )

    return Λ_OIII_tot, Λ_OIII_5008and4960, Λ_OIII_4364
end


###############################################################
#######                                                 #######
#######            S U L F U R  C O O L I N G           #######
#######                                                 #######
###############################################################

# Cooling by sulfur (S), especially S III, is important in H II regions, 
# so is added to complement the typically slightly more dominant 
# oxygen cooling there. 

# S III Einstein A coefficients (s⁻¹) from the NIST Database online, ver. 5.12 (also see 
# Tielens 2005, p. 55 for similar values):
const A_SIII_10::Float64 = 4.79e-4             # Einstein A coefficient for 33.48 μm line 
const A_SIII_21::Float64 = 2.06e-3             # Einstein A coefficient for 18.71 μm line 
const A_SIII_20::Float64 = 4.11e-8             # Einstein A coefficient for 12.00 μm line (weak!)
const A_SIII_30::Float64 = 5.25e-6             # Einstein A coefficient for 8831.0 Å line (weak!)
const A_SIII_31::Float64 = 1.85e-2 + 3.94e-5   # Einstein A coefficient for 9071.1 Å line (M1 + E2)
const A_SIII_32::Float64 = 4.78e-2 + 2.09e-4   # Einstein A coefficient for 9533.7 Å line (M1 + E2)
const A_SIII_41::Float64 = 6.61e-1             # Einstein A coefficient for 3722.7 Å line 
const A_SIII_42::Float64 = 8.82e-3             # Einstein A coefficient for 3798.0 Å line 
const A_SIII_43::Float64 = 2.08e0              # Einstein A coefficient for 6313.6 Å line (strong!)

# S III level energies (Draine 2011, p. 494 gives values of E / hc, with E = 0 
# in the ground state level 0): 
const E_SIII_1::Float64  = h * c * 298.7
const E_SIII_2::Float64  = h * c * 833.1
const E_SIII_3::Float64  = h * c * 11323.0
const E_SIII_4::Float64  = h * c * 27161.0

# S III line transition energies & values of A_ul × E_ul (pre-compute for cooling rates):
const E_SIII_10::Float64 = E_SIII_1 
const E_SIII_20::Float64 = E_SIII_2 
const E_SIII_30::Float64 = E_SIII_3 
const E_SIII_40::Float64 = E_SIII_4   
const E_SIII_21::Float64 = E_SIII_2 - E_SIII_1   
const E_SIII_31::Float64 = E_SIII_3 - E_SIII_1  
const E_SIII_41::Float64 = E_SIII_4 - E_SIII_1  
const E_SIII_32::Float64 = E_SIII_3 - E_SIII_2  
const E_SIII_42::Float64 = E_SIII_4 - E_SIII_2  
const E_SIII_43::Float64 = E_SIII_4 - E_SIII_3  

const A_SIII_43_E_SIII_43::Float64 = A_SIII_43 * E_SIII_43
const A_SIII_42_E_SIII_42::Float64 = A_SIII_42 * E_SIII_42
const A_SIII_41_E_SIII_41::Float64 = A_SIII_41 * E_SIII_41 
const A_SIII_32_E_SIII_32::Float64 = A_SIII_32 * E_SIII_32 
const A_SIII_31_E_SIII_31::Float64 = A_SIII_31 * E_SIII_31 
const A_SIII_30_E_SIII_30::Float64 = A_SIII_30 * E_SIII_30
const A_SIII_21_E_SIII_21::Float64 = A_SIII_21 * E_SIII_21 
const A_SIII_20_E_SIII_20::Float64 = A_SIII_20 * E_SIII_20
const A_SIII_10_E_SIII_10::Float64 = A_SIII_10 * E_SIII_10

# S III line transition energies in temperature units (K):
const E_SIII_10_over_kB::Float64 = E_SIII_1 / kB
const E_SIII_20_over_kB::Float64 = E_SIII_2 / kB 
const E_SIII_30_over_kB::Float64 = E_SIII_3 / kB
const E_SIII_40_over_kB::Float64 = E_SIII_4 / kB   
const E_SIII_21_over_kB::Float64 = ( E_SIII_2 - E_SIII_1 ) / kB 
const E_SIII_31_over_kB::Float64 = ( E_SIII_3 - E_SIII_1 ) / kB
const E_SIII_41_over_kB::Float64 = ( E_SIII_4 - E_SIII_1 ) / kB
const E_SIII_32_over_kB::Float64 = ( E_SIII_3 - E_SIII_2 ) / kB
const E_SIII_42_over_kB::Float64 = ( E_SIII_4 - E_SIII_2 ) / kB
const E_SIII_43_over_kB::Float64 = ( E_SIII_4 - E_SIII_3 ) / kB

# S III statistical weights, and ratios between statistical weights for updward rates:
const g_SIII_0::Float64 = 1.0                                                 # ³P₀ has g = 2J + 1 = 1 
const g_SIII_1::Float64 = 3.0; const inv_g_SIII_1::Float64 = 1.0 / g_SIII_1   # ³P₁ has g = 2J + 1 = 3
const g_SIII_2::Float64 = 5.0; const inv_g_SIII_2::Float64 = 1.0 / g_SIII_2   # ³P₂ has g = 2J + 1 = 5
const g_SIII_3::Float64 = 5.0; const inv_g_SIII_3::Float64 = 1.0 / g_SIII_3   # ¹D₂ has g = 2J + 1 = 5
const g_SIII_4::Float64 = 1.0; const inv_g_SIII_4::Float64 = 1.0 / g_SIII_4   # ¹S₀ has g = 2J + 1 = 1

const Ratio_g_SIII_10::Float64 = g_SIII_1 / g_SIII_0
const Ratio_g_SIII_20::Float64 = g_SIII_2 / g_SIII_0
const Ratio_g_SIII_30::Float64 = g_SIII_3 / g_SIII_0
const Ratio_g_SIII_40::Float64 = g_SIII_4 / g_SIII_0
const Ratio_g_SIII_21::Float64 = g_SIII_2 / g_SIII_1
const Ratio_g_SIII_31::Float64 = g_SIII_3 / g_SIII_1
const Ratio_g_SIII_41::Float64 = g_SIII_4 / g_SIII_1
const Ratio_g_SIII_32::Float64 = g_SIII_3 / g_SIII_2
const Ratio_g_SIII_42::Float64 = g_SIII_4 / g_SIII_2
const Ratio_g_SIII_43::Float64 = g_SIII_4 / g_SIII_3

""" The S III (9071.1 Å, 9533.7 Å, 8831.0 Å, 6313.6 Å, 3798.0 Å, 3722.7 Å, 
    33.48 μm, 18.71 μm, 12.00 μm) cooling rate (erg s⁻¹ cm⁻³). S III is modelled 
    as a 5-level atom, using electron collisional rates compiled in Draine (2011), 
    and Einstein A coefficients from CHIANTI. Collisions with neutrals are irrelevant 
    in the regions where S III exist.  """
function Λ_SIII( T::Float64, 
             nSIII::Float64,
                ne::Float64
                 )

    if T ≤ 1e3 || nSIII ≤ nIgnoreCooling || ne ≤ 0.0
        # Ignore S III cooling in cold gas (or if there is no S III or electrons), 
        # which avoids having to solve for the level populations etc in regions 
        # where S III cooling is negligible:
        return 0.0
    end

    T          = min( T, 1e9 )
    T4         = @fastmath T * 1e-4
    lnT4       = @fastmath log(T4)

    # Electron collisional strengths (Draine 2011, p. 497):
    Ω10        = @fastmath 3.98  * (T4^(- 0.227  -  0.100 * lnT4))  # ³P₁ → ³P₀ Note: Draine's erratum corrects level typo
    Ω20        = @fastmath 1.31  * (T4^(- 0.070  +  0.052 * lnT4))  # ³P₂ → ³P₀ 
    Ω21        = @fastmath 7.87  * (T4^(- 0.171  -  0.033 * lnT4))  # ³P₂ → ³P₁
    Ω30        = @fastmath 0.773 * (T4^(- 0.015  +  0.056 * lnT4))  # ¹D₂ → ³P₀
    Ω31        = @fastmath 2.32  * (T4^(- 0.015  +  0.057 * lnT4))  # ¹D₂ → ³P₁ 
    Ω32        = @fastmath 3.86  * (T4^(- 0.015  +  0.058 * lnT4))  # ¹D₂ → ³P₂
    Ω40        = @fastmath 0.131 * (T4^(0.043    +  0.031 * lnT4))  # ¹S₀ → ³P₀
    Ω41        = @fastmath 0.398 * (T4^(0.040    +  0.031 * lnT4))  # ¹S₀ → ³P₁
    Ω42        = @fastmath 0.655 * (T4^(0.040    +  0.029 * lnT4))  # ¹S₀ → ³P₂
    Ω43        = @fastmath 1.38  * (T4^(0.140    +  0.093 * lnT4))  # ¹S₀ → ¹D₂

    # Electron collisional rate coefficients:
    inv_sqrtT4 = @fastmath 1.0 / sqrt(T4)
    kprefact   = @fastmath 8.629e-8 * inv_sqrtT4
    k10_e      = @fastmath kprefact * Ω10 * inv_g_SIII_1
    k20_e      = @fastmath kprefact * Ω20 * inv_g_SIII_2
    k21_e      = @fastmath kprefact * Ω21 * inv_g_SIII_2
    k30_e      = @fastmath kprefact * Ω30 * inv_g_SIII_3
    k31_e      = @fastmath kprefact * Ω31 * inv_g_SIII_3
    k32_e      = @fastmath kprefact * Ω32 * inv_g_SIII_3
    k40_e      = @fastmath kprefact * Ω40 * inv_g_SIII_4
    k41_e      = @fastmath kprefact * Ω41 * inv_g_SIII_4
    k42_e      = @fastmath kprefact * Ω42 * inv_g_SIII_4
    k43_e      = @fastmath kprefact * Ω43 * inv_g_SIII_4

    # Collect collisional de-excitation rates:
    C40        = @fastmath k40_e * ne
    C41        = @fastmath k41_e * ne  
    C42        = @fastmath k42_e * ne  
    C43        = @fastmath k43_e * ne  
    C30        = @fastmath k30_e * ne  
    C31        = @fastmath k31_e * ne 
    C32        = @fastmath k32_e * ne  
    C20        = @fastmath k20_e * ne  
    C21        = @fastmath k21_e * ne  
    C10        = @fastmath k10_e * ne  

    # Collect downward rates (Rij means the rate in s⁻¹ from level i → j):
    R40        = C40
    R41        = @fastmath C41  +  A_SIII_41 
    R42        = @fastmath C42  +  A_SIII_42  
    R43        = @fastmath C43  +  A_SIII_43
    R32        = @fastmath C32  +  A_SIII_32
    R31        = @fastmath C31  +  A_SIII_31
    R30        = @fastmath C30  +  A_SIII_30 
    R21        = @fastmath C21  +  A_SIII_21
    R20        = @fastmath C20  +  A_SIII_20
    R10        = @fastmath C10  +  A_SIII_10

    # Collect upward rates:
    inv_T      = @fastmath 1.0 / T
    R01        = @fastmath Ratio_g_SIII_10 * C10 * exp( - E_SIII_10_over_kB * inv_T )
    R02        = @fastmath Ratio_g_SIII_20 * C20 * exp( - E_SIII_20_over_kB * inv_T )
    R03        = @fastmath Ratio_g_SIII_30 * C30 * exp( - E_SIII_30_over_kB * inv_T )
    R04        = @fastmath Ratio_g_SIII_40 * C40 * exp( - E_SIII_40_over_kB * inv_T )
    R12        = @fastmath Ratio_g_SIII_21 * C21 * exp( - E_SIII_21_over_kB * inv_T )
    R13        = @fastmath Ratio_g_SIII_31 * C31 * exp( - E_SIII_31_over_kB * inv_T )
    R14        = @fastmath Ratio_g_SIII_41 * C41 * exp( - E_SIII_41_over_kB * inv_T )
    R23        = @fastmath Ratio_g_SIII_32 * C32 * exp( - E_SIII_32_over_kB * inv_T )
    R24        = @fastmath Ratio_g_SIII_42 * C42 * exp( - E_SIII_42_over_kB * inv_T )
    R34        = @fastmath Ratio_g_SIII_43 * C43 * exp( - E_SIII_43_over_kB * inv_T )

    # Rates of de-population of levels 0 - 3:
    D0         = @fastmath R01  +  R02  +  R03  +  R04
    D1         = @fastmath R10  +  R12  +  R13  +  R14
    D2         = @fastmath R20  +  R21  +  R23  +  R24
    D3         = @fastmath R30  +  R31  +  R32  +  R34

    # Now we set up the linear system of equations to solve for the level populations. 
    # We define x = (x0, x1, x2, x3), where xi = ni / nSIII. We eliminate n4 using 
    # n4 = nSIII - n0 - n1 - n2 - n3, i.e. x4 = 1 - x0 - x1 - x2 - x3. Then we only 
    # need to solve for levels 0, 1, 2 and 3, stored in x. The matrix equation to solve 
    # is then Ax = b, where:
    A_00  = - (D0 + R40);  A_01 = R10 - R40;     A_02 = R20 - R40;    A_03 = R30 - R40 
    A_10  = R01 - R41;     A_11 = - (D1 + R41);  A_12 = R21 - R41;    A_13 = R31 - R41 
    A_20  = R02 - R42;     A_21 = R12 - R42;     A_22 = - (D2 + R42); A_23 = R32 - R42
    A_30  = R03 - R43;     A_31 = R13 - R43;     A_32 = R23 - R43;    A_33 = - (D3 + R43)

    A     = @SMatrix [ A_00   A_01  A_02  A_03;
                       A_10   A_11  A_12  A_13;
                       A_20   A_21  A_22  A_23;
                       A_30   A_31  A_32  A_33 ]

    b     = @SVector [ - R40, - R41, - R42, - R43 ]

    # Solve for x = (x0, x1, x2, x3) (NOTE: For a 4 x 4 system, I have found that 
    # inv(A) * b is ~ 3 × faster than A \ b, with no accuracy loss. For smaller systems 
    # of equations, there is not much difference, and for larger systems of equations A \ b 
    # is faster. So for this special case inv(A) is the preferred option):
    x     = inv(A) * b

    # Unpack & get level population densities:
    x0, x1, x2, x3 = x 
    n0 = @fastmath x0 * nSIII
    n1 = @fastmath x1 * nSIII
    n2 = @fastmath x2 * nSIII
    n3 = @fastmath x3 * nSIII 
    n4 = max( 1.0 - x0 - x1 - x2 - x3, 0.0 ) * nSIII

    # [S III] 6313.6 Å cooling rate (4 → 3):
    Λ_SIII_6313 = @fastmath n4 * A_SIII_43_E_SIII_43

    # [S III] 3798.0 Å cooling rate (4 → 2):
    Λ_SIII_3798 = @fastmath n4 * A_SIII_42_E_SIII_42

    # [S III] 3722.7 Å cooling rate (4 → 1):
    Λ_SIII_3722 = @fastmath n4 * A_SIII_41_E_SIII_41

    # [S III] 9533.7 Å cooling rate (3 → 2):
    Λ_SIII_9533 = @fastmath n3 * A_SIII_32_E_SIII_32

    # [S III] 9071.1 Å cooling rate (3 → 1):
    Λ_SIII_9071 = @fastmath n3 * A_SIII_31_E_SIII_31

    # [S III] 8831.0 Å cooling rate (3 → 0):
    Λ_SIII_8831 = @fastmath n3 * A_SIII_30_E_SIII_30

    # [S III] 18.71 μm cooling rate (2 → 1):
    Λ_SIII_18μm = @fastmath n2 * A_SIII_21_E_SIII_21

    # [S III] 12.00 μm cooling rate (2 → 0):
    Λ_SIII_12μm = @fastmath n2 * A_SIII_20_E_SIII_20

    # [S III] 33.48 μm cooling rate (1 → 0):
    Λ_SIII_33μm = @fastmath n1 * A_SIII_10_E_SIII_10

    # Total S III cooling rate:
    Λ_SIII_tot  = @fastmath ( Λ_SIII_6313 + Λ_SIII_3798 + Λ_SIII_3722 + Λ_SIII_9533
                            + Λ_SIII_9071 + Λ_SIII_8831 + Λ_SIII_18μm + Λ_SIII_12μm
                            + Λ_SIII_33μm )

    return Λ_SIII_tot 
end


const Λ_CO_0::Float64     = 2.16e-27
const ncr_CO_0::Float64   = 1.9e4
const N_COcrit_0::Float64 = 6.66e14
const Tmin_CO::Float64    = 5.0     # Clamp CO cooling below this temperature 
const Tmax_CO::Float64    = 1e4     # Clamp CO cooling above this temperature 

""" The CO cooling rate (erg s⁻¹ cm⁻³). Uses the functional 
    form from Hollenbach & McKee (1979) (HM), which in the low-density 
    regime is the same as in  Whitworth & Jaffa (2018). The coefficient 
    in the low-density regime from HM has been corrected to agree 
    with Whitworth & Jaffa."""
@inline function Λ_CO( T::Float64, 
                     nCO::Float64,
                      nH::Float64,
                     nH2::Float64,
                    N_CO::Float64 = 0.0 
                      )

    # Clamp temperature: 
    T        = clamp( T, Tmin_CO, Tmax_CO )

    # Convenient definitions:
    ncr_CO   = @fastmath ncr_CO_0 * (T^0.5)    # Critical density (Hollenbach & McKee 1979)
    N_COcrit = @fastmath N_COcrit_0 * T        # Critical column density for line trapping (Hollenbach & McKee 1979)

    # The CO cooling rate, valid in both the optically thin & low density regime, and in the 
    # dense and/or optically thick regime:
    Λ_CO = @fastmath Λ_CO_0 * nCO * nH2 * (T^1.5) / (1  +  (nH/ncr_CO) * (1  +  N_CO / N_COcrit))

    return Λ_CO
end


###############################################################
#######                                                 #######
#######      C L O U D Y  C O O L I N G  R A T E S      #######
#######                                                 #######
###############################################################


""" A convenient function for evaluating a LUT for the interpolated CLOUDY
    cooling rates. The interpolations are functions of log₁₀(T).  """
@inline function InterpolCoolingRate_CLOUDY_fast( log10T::Float64,
                                                     LUT::LinearLUT,
                                                 logTmin::Float64,
                                                 logTmax::Float64,
                                                  Λ_Tmax::Float64
                                                 )
    if log10T ≤ logTmin 
        # Below the minimum considered temperature, ignore 
        # the CLOUDY cooling rate:
        return 0.0
    elseif log10T ≥ logTmax
        # Above the maximum temperature in the data table, return 
        # the cooling rate at Tmax:
        return Λ_Tmax
    else
        # In the considered range, return the interpolated value: 
        return EvaluateLUTFlat(LUT, log10T)
    end
end

### CLOUDY - data for Carbon cooling in collisional ionization equilibrium ####

# Read in CLOUDY data:
CLOUDYdata_C                   = readdlm( joinpath(@__DIR__, "CLOUDYdata/CoolingTable_C.txt"), skipstart = 19 )
const T_CLOUDYdata_C           = Vector{Float64}( CLOUDYdata_C[:,1] )
const log10T_CLOUDYdata_C      = log10.( T_CLOUDYdata_C )
const CoolingRate_CLOUDYdata_C = Vector{Float64}( CLOUDYdata_C[:,end] )
# Interpolate data:
const InterpolCoolingRate_C = CreateLUTFromData( log10T_CLOUDYdata_C, CoolingRate_CLOUDYdata_C )

# Also get interpolation temperature range:
const T_CLOUDYdata_C_min::Float64         = minimum( T_CLOUDYdata_C ); const log10T_CLOUDYdata_C_min::Float64 = log10(T_CLOUDYdata_C_min)
const T_CLOUDYdata_C_max::Float64         = maximum( T_CLOUDYdata_C ); const log10T_CLOUDYdata_C_max::Float64 = log10(T_CLOUDYdata_C_max)
const InterpolCoolingRate_C_Tmax::Float64 = CoolingRate_CLOUDYdata_C[end]

""" The C (carbon) cooling rate (erg s⁻¹ cm⁻³) at high 
    temperatures, assuming collisional ionization equilibrium.
    See ref. (6). 
    
    NOTE: This is a function of log₁₀(T). """
@inline function Λ_C_CLOUDY( log10T::Float64, 
                                 nC::Float64,
                                 ne::Float64
                            )

    Λ_C = InterpolCoolingRate_CLOUDY_fast( log10T, 
                                           InterpolCoolingRate_C,
                                           log10T_CLOUDYdata_C_min,
                                           log10T_CLOUDYdata_C_max,
                                           InterpolCoolingRate_C_Tmax )

    return @fastmath nC * ne * Λ_C  
end


### CLOUDY - data for Nitrogen cooling in collisional ionization equilibrium ####

# Read in CLOUDY data:
CLOUDYdata_N                   = readdlm( joinpath(@__DIR__, "CLOUDYdata/CoolingTable_N.txt"), skipstart = 20 )
const T_CLOUDYdata_N           = Vector{Float64}( CLOUDYdata_N[:,1] )
const log10T_CLOUDYdata_N      = log10.( T_CLOUDYdata_N )
const CoolingRate_CLOUDYdata_N = Vector{Float64}( CLOUDYdata_N[:,end] )
# Interpolate data:
const InterpolCoolingRate_N = CreateLUTFromData( log10T_CLOUDYdata_N, CoolingRate_CLOUDYdata_N )

# Also get interpolation temperature range:
const T_CLOUDYdata_N_min::Float64         = minimum( T_CLOUDYdata_N ); const log10T_CLOUDYdata_N_min::Float64 = log10(T_CLOUDYdata_N_min)
const T_CLOUDYdata_N_max::Float64         = maximum( T_CLOUDYdata_N ); const log10T_CLOUDYdata_N_max::Float64 = log10(T_CLOUDYdata_N_max)
const InterpolCoolingRate_N_Tmax::Float64 = CoolingRate_CLOUDYdata_N[end]

""" The N (nitrogen) cooling rate (erg s⁻¹ cm⁻³) at high 
    temperatures, assuming collisional ionization equilibrium.
    See ref. (6). 
    
    NOTE: This is a function of log₁₀(T). """
@inline function Λ_N_CLOUDY( log10T::Float64, 
                                 nN::Float64,
                                 ne::Float64
                            )

    Λ_N = InterpolCoolingRate_CLOUDY_fast( log10T, 
                                           InterpolCoolingRate_N,
                                           log10T_CLOUDYdata_N_min,
                                           log10T_CLOUDYdata_N_max,
                                           InterpolCoolingRate_N_Tmax )

    return @fastmath nN * ne * Λ_N
end


### CLOUDY - data for Oxygen cooling in collisional ionization equilibrium ####

# Read in CLOUDY data:
CLOUDYdata_O                   = readdlm( joinpath(@__DIR__, "CLOUDYdata/CoolingTable_O.txt"), skipstart = 21 )
const T_CLOUDYdata_O           = Vector{Float64}( CLOUDYdata_O[:,1] )
const log10T_CLOUDYdata_O      = log10.( T_CLOUDYdata_O )
const CoolingRate_CLOUDYdata_O = Vector{Float64}( CLOUDYdata_O[:,end] )
# Interpolate data:
const InterpolCoolingRate_O = CreateLUTFromData( log10T_CLOUDYdata_O, CoolingRate_CLOUDYdata_O )

# Also get interpolation temperature range:
const T_CLOUDYdata_O_min::Float64         = minimum( T_CLOUDYdata_O ); const log10T_CLOUDYdata_O_min::Float64 = log10(T_CLOUDYdata_O_min)
const T_CLOUDYdata_O_max::Float64         = maximum( T_CLOUDYdata_O ); const log10T_CLOUDYdata_O_max::Float64 = log10(T_CLOUDYdata_O_max)
const InterpolCoolingRate_O_Tmax::Float64 = CoolingRate_CLOUDYdata_O[end]

""" The O (oxygen) cooling rate (erg s⁻¹ cm⁻³) at high 
    temperatures, assuming collisional ionization equilibrium.
    See ref. (6). 
    
    NOTE: This is a function of log₁₀(T). """
@inline function Λ_O_CLOUDY( log10T::Float64, 
                                 nO::Float64, 
                                 ne::Float64
                            )

    Λ_O = InterpolCoolingRate_CLOUDY_fast( log10T, 
                                           InterpolCoolingRate_O,
                                           log10T_CLOUDYdata_O_min,
                                           log10T_CLOUDYdata_O_max,
                                           InterpolCoolingRate_O_Tmax )

    return @fastmath nO * ne * Λ_O
end


### CLOUDY - data for Neon cooling in collisional ionization equilibrium ####

# Read in CLOUDY data:
CLOUDYdata_Ne                   = readdlm( joinpath(@__DIR__, "CLOUDYdata/CoolingTable_Ne.txt"), skipstart = 23 )
const T_CLOUDYdata_Ne           = Vector{Float64}( CLOUDYdata_Ne[:,1] )
const log10T_CLOUDYdata_Ne      = log10.( T_CLOUDYdata_Ne )
const CoolingRate_CLOUDYdata_Ne = Vector{Float64}( CLOUDYdata_Ne[:,end] )
# Interpolate data:
const InterpolCoolingRate_Ne = CreateLUTFromData( log10T_CLOUDYdata_Ne, CoolingRate_CLOUDYdata_Ne )

# Also get interpolation temperature range:
const T_CLOUDYdata_Ne_min::Float64         = minimum( T_CLOUDYdata_Ne ); const log10T_CLOUDYdata_Ne_min::Float64 = log10(T_CLOUDYdata_Ne_min)
const T_CLOUDYdata_Ne_max::Float64         = maximum( T_CLOUDYdata_Ne ); const log10T_CLOUDYdata_Ne_max::Float64 = log10(T_CLOUDYdata_Ne_max)
const InterpolCoolingRate_Ne_Tmax::Float64 = CoolingRate_CLOUDYdata_Ne[end]

""" The Ne (neon) cooling rate (erg s⁻¹ cm⁻³) at high 
    temperatures, assuming collisional ionization equilibrium.
    See ref. (6). 
    
    NOTE: This is a function of log₁₀(T). """
@inline function Λ_Ne_CLOUDY( log10T::Float64, 
                                 nNe::Float64, 
                                  ne::Float64
                             )

    Λ_Ne = InterpolCoolingRate_CLOUDY_fast( log10T, 
                                            InterpolCoolingRate_Ne,
                                            log10T_CLOUDYdata_Ne_min,
                                            log10T_CLOUDYdata_Ne_max,
                                            InterpolCoolingRate_Ne_Tmax )

    return @fastmath nNe * ne * Λ_Ne
end


### CLOUDY - data for Magnesium cooling in collisional ionization equilibrium ####

# Read in CLOUDY data:
CLOUDYdata_Mg                   = readdlm( joinpath(@__DIR__, "CLOUDYdata/CoolingTable_Mg.txt"), skipstart = 25 )
const T_CLOUDYdata_Mg           = Vector{Float64}( CLOUDYdata_Mg[:,1] )
const log10T_CLOUDYdata_Mg      = log10.( T_CLOUDYdata_Mg )
const CoolingRate_CLOUDYdata_Mg = Vector{Float64}( CLOUDYdata_Mg[:,end] )
# Interpolate data:
const InterpolCoolingRate_Mg = CreateLUTFromData( log10T_CLOUDYdata_Mg, CoolingRate_CLOUDYdata_Mg )

# Also get interpolation temperature range:
const T_CLOUDYdata_Mg_min::Float64         = minimum( T_CLOUDYdata_Mg ); const log10T_CLOUDYdata_Mg_min::Float64 = log10(T_CLOUDYdata_Mg_min)
const T_CLOUDYdata_Mg_max::Float64         = maximum( T_CLOUDYdata_Mg ); const log10T_CLOUDYdata_Mg_max::Float64 = log10(T_CLOUDYdata_Mg_max)
const InterpolCoolingRate_Mg_Tmax::Float64 = CoolingRate_CLOUDYdata_Mg[end]

""" The Mg (magnesium) cooling rate (erg s⁻¹ cm⁻³) at high 
    temperatures, assuming collisional ionization equilibrium.
    See ref. (6). 
    
    NOTE: This is a function of log₁₀(T). """
@inline function Λ_Mg_CLOUDY( log10T::Float64, 
                                 nMg::Float64, 
                                  ne::Float64
                             )

    Λ_Mg = InterpolCoolingRate_CLOUDY_fast( log10T, 
                                            InterpolCoolingRate_Mg,
                                            log10T_CLOUDYdata_Mg_min,
                                            log10T_CLOUDYdata_Mg_max,
                                            InterpolCoolingRate_Mg_Tmax )

    return @fastmath nMg * ne * Λ_Mg
end


### CLOUDY - data for Silicon cooling in collisional ionization equilibrium ####

# Read in CLOUDY data:
CLOUDYdata_Si                   = readdlm( joinpath(@__DIR__, "CLOUDYdata/CoolingTable_Si.txt"), skipstart = 27 )
const T_CLOUDYdata_Si           = Vector{Float64}( CLOUDYdata_Si[:,1] )
const log10T_CLOUDYdata_Si      = log10.( T_CLOUDYdata_Si )
const CoolingRate_CLOUDYdata_Si = Vector{Float64}( CLOUDYdata_Si[:,end] )
# Interpolate data:
const InterpolCoolingRate_Si = CreateLUTFromData( log10T_CLOUDYdata_Si, CoolingRate_CLOUDYdata_Si )

# Also get interpolation temperature range:
const T_CLOUDYdata_Si_min::Float64         = minimum( T_CLOUDYdata_Si ); const log10T_CLOUDYdata_Si_min::Float64 = log10(T_CLOUDYdata_Si_min)
const T_CLOUDYdata_Si_max::Float64         = maximum( T_CLOUDYdata_Si ); const log10T_CLOUDYdata_Si_max::Float64 = log10(T_CLOUDYdata_Si_max)
const InterpolCoolingRate_Si_Tmax::Float64 = CoolingRate_CLOUDYdata_Si[end]

""" The Si (silicon) cooling rate (erg s⁻¹ cm⁻³) at high 
    temperatures, assuming collisional ionization equilibrium.
    See ref. (6). 
    
    NOTE: This is a function of log₁₀(T). """
@inline function Λ_Si_CLOUDY( log10T::Float64, 
                                 nSi::Float64, 
                                  ne::Float64
                             )

    Λ_Si = InterpolCoolingRate_CLOUDY_fast( log10T, 
                                            InterpolCoolingRate_Si,
                                            log10T_CLOUDYdata_Si_min,
                                            log10T_CLOUDYdata_Si_max,
                                            InterpolCoolingRate_Si_Tmax )

    return @fastmath nSi * ne * Λ_Si
end


### CLOUDY - data for Sulfur cooling in collisional ionization equilibrium ####

# Read in CLOUDY data:
CLOUDYdata_S                   = readdlm( joinpath(@__DIR__, "CLOUDYdata/CoolingTable_S.txt"), skipstart = 29 )
const T_CLOUDYdata_S           = Vector{Float64}( CLOUDYdata_S[:,1] )
const log10T_CLOUDYdata_S      = log10.( T_CLOUDYdata_S )
const CoolingRate_CLOUDYdata_S = Vector{Float64}( CLOUDYdata_S[:,end] )
# Interpolate data:
const InterpolCoolingRate_S = CreateLUTFromData( log10T_CLOUDYdata_S, CoolingRate_CLOUDYdata_S )

# Also get interpolation temperature range:
const T_CLOUDYdata_S_min::Float64         = minimum( T_CLOUDYdata_S ); const log10T_CLOUDYdata_S_min::Float64 = log10(T_CLOUDYdata_S_min)
const T_CLOUDYdata_S_max::Float64         = maximum( T_CLOUDYdata_S ); const log10T_CLOUDYdata_S_max::Float64 = log10(T_CLOUDYdata_S_max)
const InterpolCoolingRate_S_Tmax::Float64 = CoolingRate_CLOUDYdata_S[end]

""" The S (sulfur) cooling rate (erg s⁻¹ cm⁻³) at high 
    temperatures, assuming collisional ionization equilibrium.
    See ref. (6). 
    
    NOTE: This is a function of log₁₀(T). """
@inline function Λ_S_CLOUDY( log10T::Float64, 
                                 nS::Float64, 
                                 ne::Float64
                            )

    Λ_S = InterpolCoolingRate_CLOUDY_fast( log10T, 
                                           InterpolCoolingRate_S,
                                           log10T_CLOUDYdata_S_min,
                                           log10T_CLOUDYdata_S_max,
                                           InterpolCoolingRate_S_Tmax )

    return @fastmath nS * ne * Λ_S
end


### CLOUDY - data for Iron cooling in collisional ionization equilibrium ####

# Read in CLOUDY data:
CLOUDYdata_Fe                   = readdlm( joinpath(@__DIR__, "CLOUDYdata/CoolingTable_Fe.txt"), skipstart = 39 )
const T_CLOUDYdata_Fe           = Vector{Float64}( CLOUDYdata_Fe[:,1] )
const log10T_CLOUDYdata_Fe      = log10.( T_CLOUDYdata_Fe )
const CoolingRate_CLOUDYdata_Fe = Vector{Float64}( CLOUDYdata_Fe[:,end] )
# Interpolate data:
const InterpolCoolingRate_Fe = CreateLUTFromData( log10T_CLOUDYdata_Fe, CoolingRate_CLOUDYdata_Fe )

# Also get interpolation temperature range:
const T_CLOUDYdata_Fe_min::Float64         = minimum( T_CLOUDYdata_Fe ); const log10T_CLOUDYdata_Fe_min::Float64 = log10(T_CLOUDYdata_Fe_min)
const T_CLOUDYdata_Fe_max::Float64         = maximum( T_CLOUDYdata_Fe ); const log10T_CLOUDYdata_Fe_max::Float64 = log10(T_CLOUDYdata_Fe_max)
const InterpolCoolingRate_Fe_Tmax::Float64 = CoolingRate_CLOUDYdata_Fe[end]

""" The Fe (iron) cooling rate (erg s⁻¹ cm⁻³) at high 
    temperatures, assuming collisional ionization equilibrium.
    See ref. (6). 
    
    NOTE: This is a function of log₁₀(T). """
@inline function Λ_Fe_CLOUDY( log10T::Float64, 
                                 nFe::Float64, 
                                  ne::Float64
                             )

    Λ_Fe = InterpolCoolingRate_CLOUDY_fast( log10T, 
                                            InterpolCoolingRate_Fe,
                                            log10T_CLOUDYdata_Fe_min,
                                            log10T_CLOUDYdata_Fe_max,
                                            InterpolCoolingRate_Fe_Tmax )

    return @fastmath nFe * ne * Λ_Fe
end


# Interpolation coefficients (Kim et al. 2023):
const a_Z::Float64 = 10.0; const T1_Z::Float64 = 2e4; const T2_Z::Float64 = 3.5e4
const k_Kim23::Float64    = a_Z / ( T2_Z - T1_Z )
const Tav_Kim23::Float64  = 0.5 * ( T1_Z + T2_Z )

""" Interpolation function between cold/warm cooling & high-temperature CLOUDY cooling 
    from Kim et al. (2023). """
@inline function S_Kim23( T::Float64 )
    @fastmath 1.0 / ( 1.0  +  exp( - k_Kim23 * (T - Tav_Kim23) ) )
end

const T_LOWTEMPCOOLING::Float64   = 1e4; @assert S_Kim23(T_LOWTEMPCOOLING) < 1e-5    "Interpolation factor should be ≪ 1 in low-temperature limit" 
const T_HIGHTEMPCOOLING::Float64  = 5e4; @assert S_Kim23(T_HIGHTEMPCOOLING) > 0.9999 "Interpolation factor should be ≃ 1 in high-temperature limit" 


const Zgasrest_sun::Float64           = (1 - fCarbon - fSilicon) * Zsun  # Solar abundance of non-C/Si metals 
const inv_Zgasrest_sun::Float64       =  1 / Zgasrest_sun

const fNitrogen_over_fZrest::Float64  = fNitrogen  / (1 - fCarbon - fSilicon) # Fraction of non-C/Si metals in N
const fNeon_over_fZrest::Float64      = fNeon      / (1 - fCarbon - fSilicon) # Fraction of non-C/Si metals in Ne
const fMagnesium_over_fZrest::Float64 = fMagnesium / (1 - fCarbon - fSilicon) # Fraction of non-C/Si metals in Mg
const fIron_over_fZrest::Float64      = fIron      / (1 - fCarbon - fSilicon) # Fraction of non-C/Si metals in Fe

""" The high-temperature limit of the metal cooling rate, using 
    the CLOUDY rates. """
@inline function Λ_Z_HighTemp( T::Float64,
                              nC::Float64,  # nCI + nCII + nCIII + nCO + nCHₓ
                              nO::Float64,  # nOI + nOII + nOIII + nOIV + nCO + nOHₓ
                              nS::Float64,  # nSI + nSII + nSIII + nSIV
                              ne::Float64,
                               ρ::Float64,  # Gas density 
                            ZSil::Float64,  # Si gas metallicity         
                        Zgasrest::Float64   # Non-C/Si gas metallicity 
                              )

    # Get trace metals number densities:
    ρZrest  = @fastmath Zgasrest * ρ   # Non-C/Si metal gas density 
    ρZSil   = @fastmath ZSil     * ρ   # Si metal gas density 

    nN      = @fastmath fNitrogen_over_fZrest  * ρZrest * inv_mN
    nNe     = @fastmath fNeon_over_fZrest      * ρZrest * inv_mNe
    nMg     = @fastmath fMagnesium_over_fZrest * ρZrest * inv_mMg
    nFe     = @fastmath fIron_over_fZrest      * ρZrest * inv_mFe
    nSi     = @fastmath ρZSil * inv_mSi

    # Get log₁₀(T) once:
    log10T  = @fastmath log10(T)

    # Metal cooling rate at high temperatures from CLOUDY, assuming CIE:
    Λ_Z_CLOUDY = ( Λ_C_CLOUDY(log10T, nC, ne)   +  Λ_N_CLOUDY(log10T, nN, ne)
                 + Λ_O_CLOUDY(log10T, nO, ne)   +  Λ_Ne_CLOUDY(log10T, nNe, ne)
                 + Λ_Mg_CLOUDY(log10T, nMg, ne) +  Λ_Si_CLOUDY(log10T, nSi, ne)
                 + Λ_S_CLOUDY(log10T, nS, ne)   +  Λ_Fe_CLOUDY(log10T, nFe, ne))

    return Λ_Z_CLOUDY
end


@inline function Λ_Z_IntermediateTemp( T::Float64,
                                      nH::Float64,  # nHI + nHII + 2 × nH₂
                                      nC::Float64,  # nCI + nCII + nCIII + nCO + nCHₓ
                                      nO::Float64,  # nOI + nOII + nOIII + nOIV + nCO + nOHₓ
                                      nS::Float64,  # nSI + nSII + nSIII + nSIV
                                     nHI::Float64,
                                     nH2::Float64,
                                    nHII::Float64,
                                    nHeI::Float64,
                                     nCI::Float64,
                                    nCII::Float64,
                                     nCO::Float64,
                                     nOI::Float64,
                                    nOII::Float64,
                                   nOIII::Float64,
                                   nSIII::Float64,
                                      ne::Float64,
                                       ρ::Float64,  # Gas density 
                                    ZSil::Float64,  # Si gas metallicity         
                                Zgasrest::Float64,  # Non-C/Si gas metallicity 
                                    N_CO::Float64 = 0.0 )

    # In the intermediate regime we use the interpolation of Kim+ 2023.

    # Total C II and C I cooling rates:
    ΛCI          = Λ_CI(T, nCI, nHII, nHI, nH2, nHeI, ne)
    ΛCII         = Λ_CII(T, nCII, nHI, nH2, nHeI, ne)

    # CO cooling rate:
    ΛCO          = Λ_CO(T, nCO, nH, nH2, N_CO)

    # Total O II and O III cooling rates:
    ΛOI          = Λ_OI(T, nOI, nHI, nH2, ne)
    ΛOII         = Λ_OII(T, nOII, ne)
    ΛOIII, _, _  = Λ_OIII(T, nOIII, ne)

    # Total S III cooling rate:
    ΛSIII        = Λ_SIII(T, nSIII, ne)

    # Total metal cooling rate in low-temperature limit (normal H II regions and 
    # colder regions):
    Λ_Z_ColdWarm = @fastmath ΛCI + ΛCII + ΛCO + ΛOI + ΛOII + ΛOIII + ΛSIII

    # Get trace metals number densities:
    ρZrest  = @fastmath Zgasrest * ρ   # Non-C/Si metal gas density 
    ρZSil   = @fastmath ZSil     * ρ   # Si metal gas density 

    nN      = @fastmath fNitrogen_over_fZrest  * ρZrest * inv_mN
    nNe     = @fastmath fNeon_over_fZrest      * ρZrest * inv_mNe
    nMg     = @fastmath fMagnesium_over_fZrest * ρZrest * inv_mMg
    nFe     = @fastmath fIron_over_fZrest      * ρZrest * inv_mFe
    nSi     = @fastmath ρZSil * inv_mSi

    # Get log₁₀(T) once:
    log10T  = @fastmath log10(T)

    # Metal cooling rate at high temperatures from CLOUDY, assuming CIE:
    Λ_Z_CLOUDY = ( Λ_C_CLOUDY(log10T, nC, ne)   +  Λ_N_CLOUDY(log10T, nN, ne)
                 + Λ_O_CLOUDY(log10T, nO, ne)   +  Λ_Ne_CLOUDY(log10T, nNe, ne)
                 + Λ_Mg_CLOUDY(log10T, nMg, ne) +  Λ_Si_CLOUDY(log10T, nSi, ne)
                 + Λ_S_CLOUDY(log10T, nS, ne)   +  Λ_Fe_CLOUDY(log10T, nFe, ne))

    # Total metal cooling rate, using interpolation (Kim et al. 2023):
    S   = S_Kim23( T )

    Λ_Z = (1.0 - S) * Λ_Z_ColdWarm  +  S * Λ_Z_CLOUDY 

    return Λ_Z
end


""" Total metal cooling rate (erg s⁻¹ cm⁻³). """
@inline function Λ_Z( T::Float64,
                     nH::Float64,  # nHI + nHII + 2 × nH₂
                     nC::Float64,  # nCI + nCII + nCIII + nCO + nCHₓ
                     nO::Float64,  # nOI + nOII + nOIII + nOIV + nCO + nOHₓ
                     nS::Float64,  # nSI + nSII + nSIII + nSIV
                    nHI::Float64,
                    nH2::Float64,
                   nHII::Float64,
                   nHeI::Float64,
                    nCI::Float64,
                   nCII::Float64,
                    nCO::Float64,
                    nOI::Float64,
                   nOII::Float64,
                  nOIII::Float64,
                  nSIII::Float64,
                     ne::Float64,
                      ρ::Float64,  # Gas density 
                   ZSil::Float64,  # Si gas metallicity         
               Zgasrest::Float64,  # Non-C/Si gas metallicity 
                   N_CO::Float64 = 0.0
                     )

    if T > T_HIGHTEMPCOOLING
        # This is the high-temperature limit, where we use 
        # CLOUDY results that assume collisional ionization equilibrium. 

        return Λ_Z_HighTemp( T, nC, nO, nS, ne, ρ, ZSil, Zgasrest )
    elseif T < T_LOWTEMPCOOLING  
        # Total C II and C I cooling rates:
        ΛCI          = Λ_CI(T, nCI, nHII, nHI, nH2, nHeI, ne)
        ΛCII         = Λ_CII(T, nCII, nHI, nH2, nHeI, ne)

        # CO cooling rate:
        ΛCO          = Λ_CO(T, nCO, nH, nH2, N_CO)

        # Total O II and O III cooling rates:
        ΛOI          = Λ_OI(T, nOI, nHI, nH2, ne)
        ΛOII         = Λ_OII(T, nOII, ne)
        ΛOIII, _, _  = Λ_OIII(T, nOIII, ne)

        # Total S III cooling rate:
        ΛSIII        = Λ_SIII(T, nSIII, ne)

        # Total metal cooling rate in low-temperature limit (metal-enriched H II regions and 
        # cold gas):
        Λ_Z_ColdWarm = @fastmath ΛCI + ΛCII + ΛCO + ΛOI + ΛOII + ΛOIII + ΛSIII

        return Λ_Z_ColdWarm
    else
        # In the intermediate regime we use the interpolation of Kim+ 2023.

        return Λ_Z_IntermediateTemp( T, nH, nC, nO, nS, nHI, nH2, nHII, nHeI, nCI, nCII, nCO, 
                                     nOI, nOII, nOIII, nSIII, ne, ρ, ZSil, Zgasrest, N_CO )
    end
end


###############################################################
#######                                                 #######
#######         T O T A L  C O O L I N G   R A T E      #######
#######                                                 #######
###############################################################

""" The total cooling rate (erg s⁻¹ cm⁻³) from all processes 
    considered. """
@inline function Λ( T::Float64, 
                  nHI::Float64,
                  nH2::Float64, 
                 nHII::Float64, 
                 nHeI::Float64,
                nHeII::Float64,
               nHeIII::Float64,
                  nHD::Float64,
                  nCI::Float64,
                  nCO::Float64,
                 nCII::Float64,
                  nOI::Float64,
                 nOII::Float64,
                nOIII::Float64,
                nSIII::Float64,
                   ne::Float64,
                   nH::Float64,
                   nC::Float64,
                   nO::Float64,
                   nS::Float64,
                    ρ::Float64,
                 ZSil::Float64,
             Zgasrest::Float64,
                 N_CO::Float64 = 0.0,
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

            + Λ_HD(T, nHD, nHI)
            
            + Λ_Z(T, nH, nC, nO, nS, nHI, nH2, nHII, nHeI, nCI, nCII, nCO, 
                  nOI, nOII, nOIII, nSIII, ne, 
                  ρ, ZSil, Zgasrest, N_CO) )

    # Correct for the CMB-induced limit following Hopkins et al. (2023):
    TCMB = @fastmath TCMB0 * (1.0 + z)
    Λ1   = @fastmath Λ1 * (T - TCMB) / (T + TCMB)

    # Add Compton and dust cooling/heating to get the total cooling rate:
    Λ    = @fastmath ( Λ1  +  Λ_Comp(T, ne, z) )

    return Λ
end
