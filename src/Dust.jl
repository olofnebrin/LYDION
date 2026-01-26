
""" FILE:        Dust.jl

    PURPOSE:     Contains the code for computing dust properties (e.g. dust temperature, charges,
                 opacities), and setting up the corresponding dust struct.

"""

using .Structs: DustState
using .ConstantsAndFunctions
using DelimitedFiles          # For reading data 
using Interpolations          # For interpolation of the Planck mean opacity
using QuadGK                  # For integrating over data 
using Printf


const ρgr_sil::Float64       = 3.5            # Density of a silicate grain (g cm⁻³)
const ρgr_C::Float64         = 2.24           # Density of a graphite grain (g cm⁻³)
const inv_ρgr_sil::Float64   = 1 / ρgr_sil    # Inverse of ρgr_sil, useful 
const inv_ρgr_C::Float64     = 1 / ρgr_C      # Inverse of ρgr_C, useful 
const agr_sil::Float64       = 0.138   * μm   # ⟨a³⟩/⟨a²⟩ for silicate grains 
const agr_C::Float64         = 0.00617 * μm   # ⟨a³⟩/⟨a²⟩ for graphite/carbonaceous grains 
const inv_agr_sil::Float64   = 1 / agr_sil    # Inverse of agr_sil, useful 
const inv_agr_C::Float64     = 1 / agr_C      # Inverse of agr_C, useful 
const fsil::Float64          = 0.726          # Fraction of total dust mass in silicate grains 
const OneMinusfsil::Float64  = 1 - fsil       # Fraction of total dust mass in graphite/carbonaceous grains 

# Absorption efficiencies evaluated at the grain sizes agr_sil and agr_C, 
# for silicate and graphite grains, respectively (see file WeingartnerDustModel.jl
# for calculation):
const Qabs_sil_FUV::Float64  = 1.25
const Qabs_sil_LW::Float64   = 1.11
const Qabs_sil_EUV1::Float64 = 0.916
const Qabs_sil_EUV2::Float64 = 0.937
const Qabs_sil_EUV3::Float64 = 0.928
const Qabs_sil_Lyα::Float64  = 1.19

const Qabs_C_FUV::Float64    = 0.135
const Qabs_C_LW::Float64     = 0.726
const Qabs_C_EUV1::Float64   = 1.64
const Qabs_C_EUV2::Float64   = 0.356
const Qabs_C_EUV3::Float64   = 0.153
const Qabs_C_Lyα::Float64    = 0.180

# Export functions:
export Λdg, ComputeTd, FindGrainCharge, SetupDustState, κP, κR


###########################################################################
#######                                                             #######
#######               D U S T  T E M P E R A T U R E                #######
#######                                                             #######
###########################################################################

# Useful constants: 
const Two_kB::Float64    = 2 * kB 
const Λdg_const::Float64 = ( fsil         * inv_agr_sil * inv_ρgr_sil
                           + OneMinusfsil * inv_agr_C   * inv_ρgr_C ) 

""" Dust collisional heating (if > 0) or cooling (< 0) rate Λdg(T,Td) (in erg s⁻¹ cm⁻³).
    Equivalently, it is the cooling rate of the gas if > 0, and heating rate if < 0.
    See Draine (2011)."""
@inline function Λdg( T::Float64, Td::Float64, ρd::Float64, nH::Float64 )

    # Factor 0.5 from assumed accomodation factor:
    Stuff = ThreeFourths * nH * vth_H(T) * 0.5 * Two_kB * ( T - Td )

    Λdg   = Stuff * ρd * Λdg_const

    return Λdg
end


""" Λdg / ρd, useful for the dust temperature solver. """
@inline function Λdg_over_ρd( T::Float64, 
                             Td::Float64,  
                             nH::Float64 ) 

    # Factor 0.5 from assumed accomodation factor:
    Stuff        = ThreeFourths * nH * vth_H(T) * 0.5 * Two_kB * ( T - Td )

    Λdg_over_ρd  = Stuff * Λdg_const

    return Λdg_over_ρd
end


const Td_max::Float64  = 2e3   # Upper limit to dust temperature (K) - assumed sublimation above

""" Planck-mean dust opacity, as computed from the stored log-log interpolation 
    in the dust struct. """
@inline function κP( Dust::DustState, T::Float64 ) 
    T = clamp( T, 1.0, Td_max )
    return 10.0^( Dust.logκP_itp( log10(T) ) )
end

""" Rosseland-mean dust opacity, as computed from the stored log-log interpolation 
    in the dust struct. """
@inline function κR( Dust::DustState, T::Float64 ) 
    T = clamp( T, 1.0, Td_max )
    return 10.0^( Dust.logκR_itp( log10(T) ) )
end


const atol_Td::Float64 = 0.1   # Absolute tolerance (K) to deem converged 
const rtol_Td::Float64 = 1e-4  # Relative tolerance to deem converged 
const dNiter_Td        = 1:50  # Consider up to 50 iterations by default 

""" This function computes the dust temperature, given the radiation background,
    and gas conditions. Uses bisection for robustness. Dust temperature is found by 
    solving:
    
    4 κP(Td) Td⁴ =  ∑_{B} κabs_B 4πJ_B  +  κabs_IR 4πJ_IR  
                 +  κabs_Lyα e_Lyα c  +  4 κP(TCMB) TCMB⁴ 
                 +  Λdg(T,Td) / ρd """
function ComputeTd( J_IR::Float64,
                   J_opt::Float64,
                   J_FUV::Float64,
                    J_LW::Float64,
                  J_EUV1::Float64,
                  J_EUV2::Float64,
                  J_EUV3::Float64,
                    Dust::DustState,
                    TCMB::Float64,
                       T::Float64,
                      nH::Float64,
                   e_Lyα::Float64 = 0.0
                     )

    # Pre-compute some stuff:
    Four_π_JIR  = Four_π * J_IR
    HeatOpt     = Dust.κd_abs_opt  * Four_π * J_opt
    HeatFUV     = Dust.κd_abs_FUV  * Four_π * J_FUV
    HeatLW      = Dust.κd_abs_LW   * Four_π * J_LW
    HeatEUV1    = Dust.κd_abs_EUV1 * Four_π * J_EUV1
    HeatEUV2    = Dust.κd_abs_EUV2 * Four_π * J_EUV2 
    HeatEUV3    = Dust.κd_abs_EUV3 * Four_π * J_EUV3
    HeatLyα     = Dust.κd_abs_Lyα  * e_Lyα * c
    HeatCMB     = 4 * κP(Dust, TCMB) * σSB * (TCMB^4)

    # The dust temperature Td is the solution to the function f(Td) = 0,
    # where:
    f( Td ) = ( HeatOpt  +  HeatFUV  +  HeatLW  +  HeatEUV1  +  HeatEUV2  +  HeatEUV3
              + HeatLyα
              + κP(Dust, Td) * Four_π_JIR
              + HeatCMB
              + Λdg_over_ρd( T, Td, nH ) 

              - 4 * κP(Dust, Td) * σSB * Td^4 )

    # Bracket expected range:
    a       = max( 0.5 * TCMB, 1.0 )    # Use small Td > 0 to avoid problems in the κP interpolation
    b       = Td_max 
    fa      = f(a)
    fb      = f(b)

    if fa * fb > 0.0
        # In this case, dust heating still prevails even at Td = Td_max, so 
        # we should just return the maximum dust temperature:
        return Td_max
    end

    # Initialize "old" temperature:
    Td_old  = 0.5 * (a + b)

    for _ in dNiter_Td 
        Td_new   = 0.5 * ( a + b )
        fmid     = f(Td_new)

        if 0.5 * (b - a) < max(atol_Td, rtol_Td * Td_new)
            # In this case it has converged, so return Td:
            return Td_new
        end

        if fa * fmid < 0.0
            b  = Td_new
            fb = fmid
        else
            a  = Td_new
            fa = fmid
        end

        Td_old = Td_new
    end

    # If bisection failed to converge, it meant that Td > Td_max, and so we take:
    return Td_max
end


###########################################################################
#######                                                             #######
#######                     D U S T  C H A R G E                    #######
#######                                                             #######
###########################################################################

# Below is the code for computing the grain charge, balancing electron collisional  
# charging against ion collisional + photoelectric charging.


""" Function needed for collisional charging, and modifications to cross-sections.
    See Draine & Sutin (1987). Here:
    
    τᵢ ≡ a_gr * kB * T / qᵢ² 
    νᵢ ≡ Z_gr * e / qᵢ 
    
    where qᵢ is the charge of an ion/electron. """
@inline function J̃( τi::Float64, 
                    νi::Float64 )

    τi = max( τi, 1e-12 ) # To avoid division by zero

    if νi > 0.0
        θ = νi / ( 1  +  νi^(-1/2) )
        J̃ = ( 1  +  (4*τi + 3*νi)^(-1/2) )^2 * exp( - θ / τi )

        return J̃
    elseif νi < 0.0
        J̃ = ( 1  +  abs(νi)/τi ) * ( 1  +  sqrt(2 / (τi + 2*abs(νi))) )
        
        return J̃
    else 
        J̃ = 1  +  sqrt( π / (2*τi) )

        return J̃
    end
end


# Create interpolations for silicates & graphites, using calculations 
# from Kimura (2016):

const KimuraData_Sil = readdlm(joinpath(@__DIR__, "KimuraSilicatePhotoElectricYield.txt"), 
                           Float64, skipstart = 9)

const KimuraData_C   = readdlm(joinpath(@__DIR__, "KimuraGraphitePhotoElectricYield.txt"), 
                           Float64, skipstart = 9)

const E_Kimura_Sil   =   KimuraData_Sil[:, 1]
const E_Kimura_C     =   KimuraData_C[:, 1]
const agr_Kimura     = [ 1.0, 10.0, 100.0, 1e4 ] * nm  # Grain sizes in the data, converted from nm to cm 
const Y_Kimura_Sil   =   KimuraData_Sil[:, 3:6]
const Y_Kimura_C     =   KimuraData_C[:, 3:6]

const itp_Y_Sil = let log10_a_Sil = log10.(agr_Kimura)
     extrapolate( interpolate( (E_Kimura_Sil, log10_a_Sil), Y_Kimura_Sil, Gridded(Linear()) ), Flat() )
end

const itp_Y_C = let log10_a_C = log10.(agr_Kimura)
     extrapolate( interpolate( (E_Kimura_C, log10_a_C), Y_Kimura_C, Gridded(Linear()) ), Flat() )
end


@inline function Ya_Sil( hν::Float64, 
                        agr::Float64 ) 
    return itp_Y_Sil( hν / eV, log10(agr) )
end


@inline function Ya_C( hν::Float64, 
                      agr::Float64 ) 
    return itp_Y_C( hν / eV, log10(agr) )
end


const W_Sil::Float64      = 4.97 * eV         # Work function for silicates (Kimura 2016)
const W_C::Float64        = 4.75 * eV         # Work function for graphites (Kimura 2016)
const Ebg_Sil::Float64    = W_Sil  -  3 * eV  # Band gap for silicates (Weingartner & Draine 2001)
const Ebg_C::Float64      = 0.0               # Band gap for graphites (Weingartner & Draine 2001)
const e²_agr_sil::Float64 = (e^2) / agr_sil
const e²_agr_C::Float64   = (e^2) / agr_C

""" The photoelectric yield for a silicate or graphite dust grain, 
    as computed by Kimura (2016). """
function Ype( hν::Float64,
             Zgr::Float64,
        Graphite::Bool = false
             )

    # Choose relevant work function & band gap energy,
    # grain size, and value of e² / a_gr:
    if !Graphite 
        W      = W_Sil
        Ebg    = Ebg_Sil
        agr    = agr_sil
        e²_agr = e²_agr_sil
    else
        W      = W_C
        Ebg    = Ebg_C
        agr    = agr_C
        e²_agr = e²_agr_C
    end

    # The ionization potential IP_V 
    if Zgr ≥ 0.0
        # WD, Eq. 2 (applies to both silicates and graphites):
        IP_V  = W  +  ( (Zgr + 0.5)  +  (Zgr + 2) * 0.3 * Ang / agr ) * e²_agr
    else
        # WD takes IP_V(Zgr < 0) = EAZ(Zgr+1):
        if !Graphite
            # WD, Eq. 5 for silicates:
            EA_Zp1 = W  -  Ebg  +  (Zgr + 0.5) * e²_agr
        else
            # WD, Eq. 4 for carbon grains:
            EA_Zp1 = W  +  (Zgr + 0.5) * e²_agr  -  e²_agr * (4 * Ang) / (agr  +  7 * Ang)
        end
        IP_V  = EA_Zp1  
    end

    # The threshold photon energy hν_pet
    # needed for photoelectric emission (WD, Eq. 6):
    if Zgr ≥ - 1
        hν_pet = IP_V
    else
        Emin   = - (Zgr + 1) * e²_agr * 1 / ( 1  +  (27 * Ang / agr)^0.75 )
        hν_pet = IP_V  +  Emin 
    end

    # If photon energy is below the threshold, return zero yield 
    # and exit early:
    if hν < hν_pet
        return 0.0
    end

    # The factor y₂ (WD, Eq. 11):
    if Zgr ≥ 0.0
        Elow  = - (Zgr + 1) * e²_agr
        Ehigh = hν  -  hν_pet  
        y₂    = ( Ehigh^2 ) * ( Ehigh  -  3 * Elow ) / ( Ehigh - Elow )^3
    else
        y₂    = 1.0
    end

    # Photoelectric yield:
    if !Graphite
        Ype = y₂ * Ya_Sil(hν, agr)
    else
        Ype = y₂ * Ya_C(hν, agr)
    end

    return Ype 
end


const le::Float64 = 10 * Ang  # Weingartner & Draine (2001) and Tielens (2005) give 10 Å

""" Electron sticking coefficient, following Weingartner & Draine (2001). """
@inline function se( agr::Float64 )
    0.5 * ( 1 - exp(- agr / le) )
end

# Pre-compute electron sticking coefficients:
const se_sil::Float64          = se( agr_sil )
const se_C::Float64            = se( agr_C )

const uth_Inv_prefact::Float64 = sqrt( π * me / (8 * kB) )

""" The photoelectric term J_pe * sqrt(π m_e / 8 kB T) / (π agr² n_e s_e), where J_pe 
    is the photoelectric emission rate (see Tielens 2005):
    
    J_pe = ∫dν (4πJ/hν) π agr² Qabs Ype(hν)  """
function PhotoElectricChargingTerm( J_FUV::Float64,
                                     J_LW::Float64,
                                   J_EUV1::Float64,
                                   J_EUV2::Float64,
                                   J_EUV3::Float64,
                                       ne::Float64,
                                        T::Float64,
                                      Zgr::Float64,
                                    e_Lyα::Float64 = 0.0,
                                 Graphite::Bool    = false
                                   )

    # Convenient thermal factor (inverse of thermal velocity):
    uth_Inv  = uth_Inv_prefact / sqrt(T)

    # Clamp ne to avoid division by zero:
    ne       = max( ne, 1e-30 )

    if Graphite
        # Absorption efficiencies:
        Qabs_FUV  = Qabs_C_FUV
        Qabs_LW   = Qabs_C_LW
        Qabs_EUV1 = Qabs_C_EUV1
        Qabs_EUV2 = Qabs_C_EUV2
        Qabs_EUV3 = Qabs_C_EUV3
        Qabs_Lyα  = Qabs_C_Lyα

        # Photo-electric yields:
        Ype_FUV   = Ype( E_FUV,  Zgr, true )
        Ype_LW    = Ype( E_LW,   Zgr, true )
        Ype_EUV1  = Ype( E_EUV1, Zgr, true )
        Ype_EUV2  = Ype( E_EUV2, Zgr, true )
        Ype_EUV3  = Ype( E_EUV3, Zgr, true )
        Ype_Lyα   = Ype( ELyα,   Zgr, true )

        # 1 / (ne * se):
        inv_ne_s  = 1 / ( ne * se_C )
    else
        # Absorption efficiencies:
        Qabs_FUV  = Qabs_sil_FUV
        Qabs_LW   = Qabs_sil_LW
        Qabs_EUV1 = Qabs_sil_EUV1
        Qabs_EUV2 = Qabs_sil_EUV2
        Qabs_EUV3 = Qabs_sil_EUV3
        Qabs_Lyα  = Qabs_sil_Lyα

        # Photo-electric yields:
        Ype_FUV   = Ype( E_FUV,  Zgr, false )
        Ype_LW    = Ype( E_LW,   Zgr, false )
        Ype_EUV1  = Ype( E_EUV1, Zgr, false )
        Ype_EUV2  = Ype( E_EUV2, Zgr, false )
        Ype_EUV3  = Ype( E_EUV3, Zgr, false )
        Ype_Lyα   = Ype( ELyα,   Zgr, false )

        # 1 / (ne * se):
        inv_ne_s  = 1 / ( ne * se_sil )
    end

    # Photoelectric term:
    PhotoElectricTerm = Four_π * ( (J_FUV * inv_E_FUV)   * (Qabs_FUV  * inv_ne_s ) * uth_Inv * Ype_FUV
                                 + (J_LW * inv_E_LW)     * (Qabs_LW   * inv_ne_s ) * uth_Inv * Ype_LW
                                 + (J_EUV1 * inv_E_EUV1) * (Qabs_EUV1 * inv_ne_s ) * uth_Inv * Ype_EUV1
                                 + (J_EUV2 * inv_E_EUV2) * (Qabs_EUV2 * inv_ne_s ) * uth_Inv * Ype_EUV2
                                 + (J_EUV3 * inv_E_EUV3) * (Qabs_EUV3 * inv_ne_s ) * uth_Inv * Ype_EUV3 )

    # Add contribution from Lyα:
    PhotoElectricTerm += (c * e_Lyα * inv_ELyα) * (Qabs_Lyα * inv_ne_s)  * uth_Inv * Ype_Lyα

    return PhotoElectricTerm
end


const vthH_over_vthe::Float64             = sqrt( me / mH )
const vthHe_over_vthe::Float64            = sqrt( me / mHe )
const Stick_ion::Float64                  = 0.5                  # Sticking coefficient of ions (0.5 by default)
const agr_sil_kB_over_e_squared::Float64  = agr_sil * kB / (e^2) # Convenient variable 
const agr_C_kB_over_e_squared::Float64    = agr_C   * kB / (e^2) 

""" The ion collisional charging term:

    ∑_i π * agr² * nᵢ * sᵢ * sqrt(8 kB T / π mᵢ) * J̃ᵢ ,
    
    normalized by π * agr² * sₑ * sqrt(8 kB T / π mₑ), as for the 
    photoelectric term. """
function IonCollisionalChargingTerm( Zgr::Float64, 
                                       T::Float64,
                                    nHII::Float64,
                                   nHeII::Float64,
                                  nHeIII::Float64,
                                      ne::Float64,
                                Graphite::Bool = false 
                                     )

    ne = max( ne, 1e-30 )

    if Graphite 
        inv_ne_se             = 1 / ( ne * se_C )
        agr_kB_over_e_squared = agr_C_kB_over_e_squared
    else
        inv_ne_se             = 1 / ( ne * se_sil )
        agr_kB_over_e_squared = agr_sil_kB_over_e_squared
    end

    # H II contribution:
    τ_HII               = agr_kB_over_e_squared * T    # agr * kB * T / ((+e)²)
    ν_HII               = Zgr                          # Zgr * e / (+e) = Zgr 
    J̃_HII               = J̃(τ_HII, ν_HII)
    HII_contribution    = nHII * Stick_ion * vthH_over_vthe * J̃_HII

    # He II contribution:
    τ_HeII              = τ_HII              # agr * kB * T / ((+e)²) = same as for H II 
    ν_HeII              = ν_HII              # Zgr * e / (+e) = Zgr = same as for H II 
    J̃_HeII              = J̃(τ_HeII, ν_HeII)
    HeII_contribution   = nHeII * Stick_ion * vthHe_over_vthe * J̃_HeII

    # He III contribution:
    τ_HeIII             = 0.25 * τ_HII       # agr * kB * T / ((+2e)²) = 0.25 * τ_HII
    ν_HeIII             = 0.5 * ν_HII        # Zgr * e / (+2e) = 0.5 * Zgr = 0.5 * ν_HII
    J̃_HeIII             = J̃(τ_HeIII, ν_HeIII)
    HeIII_contribution  = nHeIII * Stick_ion * vthHe_over_vthe * J̃_HeIII

    # Put it all together:
    CollisionalTerm = inv_ne_se * ( HII_contribution + HeII_contribution + HeIII_contribution )

    return CollisionalTerm 
end


""" The maximum grain charge that can be attained by photoelectric charging.
    See Weingartner & Draine (2001), Eq. (22). """
@inline function Zgr_pe_max( Graphite::Bool = false )

    if !Graphite
        W   = W_Sil
        agr = agr_sil
    else
        W   = W_C
        agr = agr_C
    end

    return ( ((E_EUV3 - W) / (14.4 * eV)) * (agr / Ang)  +  1/2  -  0.3 * Ang / agr ) / (
               1  +  0.3 * Ang / agr )
end


""" The minimum grain charge, set by field emission. 
    See Weingartner & Draine (2001), Eq. (24). """
@inline function Zgr_min( Graphite::Bool = false )  

    if Graphite
        U_ait   = - ( 3.9  +  0.12 * (agr_C / Ang)  +  2 * (Ang / agr_C) )

        # Most negative allowed charge:
        Zgr_min = floor( (U_ait / 14.4) * (agr_C / Ang) ) + 1
    else
        # For silicate grains:
        U_ait   = - ( 2.5  +  0.07 * (agr_sil / Ang)  +  8 * (Ang / agr_sil) )

        # Most negative allowed charge:
        Zgr_min = floor( (U_ait / 14.4) * (agr_sil / Ang) ) + 1
    end

    return Zgr_min 
end

const Zgr_min_sil::Float64         = Zgr_min( false )
const Zgr_min_C::Float64           = Zgr_min( true )
const Zgr_pe_max_sil::Float64      = Zgr_pe_max( false )
const Zgr_pe_max_C::Float64        = Zgr_pe_max( true )

const max_iter_GrainCharge::Int    = 20
const dNiter_Zgr                   = 1:max_iter_GrainCharge
const atol_Zgr::Float64            = 0.1
const rtol_Zgr::Float64            = 1e-3 

""" This function computes the grain charge using a bisection method. """
function FindGrainCharge( T::Float64, 
                       nHII::Float64,
                      nHeII::Float64,
                     nHeIII::Float64,
                         ne::Float64,
                      J_FUV::Float64,
                       J_LW::Float64,
                     J_EUV1::Float64,
                     J_EUV2::Float64,
                     J_EUV3::Float64,
                   Graphite::Bool    = false,
                      e_Lyα::Float64 = 0.0
                          )

    # Compute τₑ = agr * kB * T / ((-e)²):
    if Graphite
        τ_e  = agr_C_kB_over_e_squared * T   # τₑ = agr * kB * T / ((-e)²)
    else
        τ_e  = agr_sil_kB_over_e_squared * T  # τₑ = agr * kB * T / ((-e)²)
    end
    # Pre-factor for νₑ = Zgr * e / (-e) = - Zgr, just a minus sign:
    ν_e_prefact = - 1.0                        

    # Define the function the root of which will give Zgr 
    # (ion collisional terms + photoelectric terms - electron collisional term):
    f( Zgr ) = ( IonCollisionalChargingTerm( Zgr, T, nHII, nHeII, nHeIII, ne, Graphite )
               + PhotoElectricChargingTerm( J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, ne, T, Zgr, e_Lyα, Graphite )

               - J̃( τ_e,  ν_e_prefact * Zgr )  )

    # Bracket expected range:
    if Graphite 
        # Minimum value is expected to be ≈ - 2.5 * agr * kB * T / e²,
        # and maximum value set by photoelectric charging, so make the following bracket:
        a       = - 3 * agr_C_kB_over_e_squared * T  
        b       = Zgr_pe_max_C
        Zgr_Min = Zgr_min_C
    else
        a       = - 3 * agr_sil_kB_over_e_squared * T      
        b       = Zgr_pe_max_sil
        Zgr_Min = Zgr_min_sil
    end
    fa     = f(a)
    fb     = f(b)

    # Check that this is an OK bracket. If not, try to enlarge 
    # the region until it does bracket a root:
    if fa * fb > 0.0
        i           = 0
        fa_times_fb = fa * fb
        while fa_times_fb > 0.0 && i < 10
            # Update fa * fb:
            a          *= 2    # Make bracket Zgr_min even more negative 
            b          *= 1.2  # Make bracket Zgr_max slightly larger (this is already a rather strict upper limit, so don't change too much)
            fa          = f(a)
            fb          = f(b)
            fa_times_fb = fa * fb 

            # Update count:
            i   += 1
        end
    end

    # Handle bracket failure:
    if fa * fb > 0.0
        @warn "FindGrainCharge failed to bracket root: Returning endpoint with smaller |f|."
        if abs(fa) < abs(fb)
            return a
        else
            return b 
        end
    end

    # Bisection loop:
    for _ in dNiter_Zgr 
        Zgr_new   = 0.5 * ( a + b )
        fmid      = f(Zgr_new)

        if 0.5 * (b - a) < max( atol_Zgr, rtol_Zgr * abs(Zgr_new) )
            # In this case it has converged, so return Zgr:
            return max( Zgr_new, Zgr_Min )
        end

        if fa * fmid < 0.0
            b  = Zgr_new
            fb = fmid
        else
            a  = Zgr_new
            fa = fmid
        end
    end

    return max( 0.5 * (a + b), Zgr_Min )   
end


###########################################################################
#######                                                             #######
#######                      D U S T  D R A G                       #######
#######                                                             #######
###########################################################################

# Below is code for computing the dust drag coefficient K, such that 
# ∂(ρdud)/∂t = K * (u - ud), with K ≡ |⟨n_gr F_drag⟩| / |ud - u|.

const G0_prefact::Float64 = 8 / (3 * sqrt(π))
const G0_const::Float64   = 9 * π / 64

""" The function G₀ ≡ (8 sᵢ / 3 √π) √(1 + 9πsᵢ²/64), 
    divided by |ud - u|. sᵢ ≡ |ud - u| / √(2kT/mᵢ) can be 
    used to get the result. """
@inline function G0_over_urel( si::Float64, vth::Float64 )
    return G0_prefact * (1 / vth) * sqrt( 1 + G0_const * si * si ) 
end


const G2_const::Float64 = 3 * sqrt(π) / 4

""" The function G₂ ≡ sᵢ / ( (3√π)/4  +  sᵢ³ ),
    divided by |ud - u|. 

    sᵢ ≡ |ud - u| / √(2kT/mᵢ) can be used to get the result. """
@inline function G2_over_urel( si::Float64, vth::Float64 )
    return 1 / ( vth * (G2_const + si^3) )
end


# Convenient constants:
const vth_HI_prefact::Float64    = sqrt( 2 * kB / mH )
const vth_H2_prefact::Float64    = sqrt( 2 * kB / mH2 )
const vth_HeI_prefact::Float64   = sqrt( 2 * kB / mHe )
const vth_e_prefact::Float64     = sqrt( 2 * kB / me )
const ϕ_sil_prefact::Float64     = (e^2) / ( agr_sil * kB )
const ϕ_C_prefact::Float64       = (e^2) / ( agr_C * kB )
const Zgr_th_Kdrag::Float64      = 0.1                   # Grain charge is ignored if |Zgr| < Zgr_th_Kdrag
const T_floor_Kdrag::Float64     = 0.01
const ne_floor_K_drag::Float64   = 1e-30 

# Convenient constants for evaluating |⟨ngr Fdrag⟩| / urel:
const Fdrag_sil_prefact::Float64 = (3/2) * fsil * kB / (ρgr_sil * agr_sil)
const Fdrag_C_prefact::Float64   = (3/2) * OneMinusfsil * kB / (ρgr_C * agr_C)
const Λ_sil_prefact::Float64     = (3 / (2 * agr_sil * e)) * sqrt(kB / π)
const Λ_C_prefact::Float64       = (3 / (2 * agr_C * e))   * sqrt(kB / π)

""" This function computes the drag force coefficient K ≡ |⟨ngr Fdrag⟩| / | ud - u |,
    considering drag from H I, H₂, H II, He I, He II, He III, and silicate + graphite 
    dust in the specified mix of the assumed dust model.  """
function ComputeDragCoefficientK( Zgr_sil::Float64, # Silicate grain charge 
                                    Zgr_C::Float64, # Graphite grain charge 
                                       ρd::Float64, # Dust density 
                                        T::Float64, # Gas temperature 
                                      nHI::Float64, # H I number density 
                                      nH2::Float64, # H₂ number density 
                                     nHII::Float64, # H II number density 
                                     nHeI::Float64, # He I number density 
                                    nHeII::Float64, # He II number density 
                                   nHeIII::Float64, # He III number density 
                                       ne::Float64, # e⁻ number density 
                                     ud_r::Float64, # Dust velocity in r-direction 
                                      u_r::Float64, # Gas velocity in r-direction 
                                     ud_Z::Float64, # Dust velocity in Z-direction 
                                      u_Z::Float64  # Gas velocity in Z-direction 
                                      )


    T          = max( T, T_floor_Kdrag )
    sqrt_T     = sqrt(T)
    inv_T      = 1.0 / T
    urel       = sqrt( (ud_r - u_r)^2 + (ud_Z - u_Z)^2 )
    ne_floored = max( ne, ne_floor_K_drag )   # To avoid division by zero

    # Evaluate vth ≡ √(2kT/mᵢ) for all major species:
    vth_HI     = vth_HI_prefact  * sqrt_T
    vth_H2     = vth_H2_prefact  * sqrt_T 
    vth_HeI    = vth_HeI_prefact * sqrt_T
    vth_e      = vth_e_prefact   * sqrt_T

    # Evaluate sᵢ ≡ |ud - u| / vth for all major species
    # (some are the same, e.g. H II is same as H I, so no need to compute):
    s_HI       = urel / vth_HI 
    s_H2       = urel / vth_H2
    s_HeI      = urel / vth_HeI 
    s_e        = urel / vth_e

    # Compute G₀/urel:
    G̃0_HI      = G0_over_urel( s_HI, vth_HI )
    G̃0_HII     = G̃0_HI
    G̃0_H2      = G0_over_urel( s_H2,  vth_H2 )
    G̃0_HeI     = G0_over_urel( s_HeI, vth_HeI )
    G̃0_HeII    = G̃0_HeI
    G̃0_HeIII   = G̃0_HeI
    G̃0_e       = G0_over_urel( s_e,  vth_e )

    # Compute G₂/urel (not needed for H₂, since it is neutral):
    G̃2_HI      = G2_over_urel( s_HI, vth_HI )
    G̃2_HII     = G̃2_HI
    G̃2_HeI     = G2_over_urel( s_HeI, vth_HeI )
    G̃2_HeII    = G̃2_HeI
    G̃2_HeIII   = G̃2_HeI
    G̃2_e       = G2_over_urel( s_e,  vth_e )

    # Evaluate ϕ ≡ Z_gr * e² / a_gr * kB * T:
    ϕ_sil      = ϕ_sil_prefact * Zgr_sil * inv_T 
    ϕ_C        = ϕ_C_prefact   * Zgr_C   * inv_T 
    ϕ_sil_sq   = ϕ_sil * ϕ_sil
    ϕ_C_sq     = ϕ_C   * ϕ_C

    # Evaluate Λ ≡ [3 / (2 * a_gr * e * |ϕ|)] * sqrt(kB * T / π * ne)
    Λ_sil      = Λ_sil_prefact * ( 1 / abs(ϕ_sil) ) * sqrt( T / ne_floored )
    Λ_C        = Λ_C_prefact * ( 1 / abs(ϕ_C) ) * sqrt( T / ne_floored )

    ############# S I L I C A T E  G R A I N S #########################

    # We first add up the drag contribution from the silicate grains. 

    # Evaluate ϕ_sil² log(Λ_sil) safely:
    if abs(Zgr_sil) > Zgr_th_Kdrag
        absΛ_sil          = max( abs(Λ_sil), 1.0 )      # Ensure that ln(Λ) > 0
        ϕ_sil_sq_logΛ_sil = ϕ_sil_sq * log( absΛ_sil )
    else 
        # In case Zgr_sil = 0, Λ becomes numerically problematic, 
        # but ϕ_sil → 0, so we just get 0 regardless:
        ϕ_sil_sq_logΛ_sil = 0.0
    end

    K_drag_sil_prefact  = Fdrag_sil_prefact * ρd * T
    
    # H I contribution (Z = 0, so only need G₀ / urel):
    Sil_HI_contr        = nHI * G̃0_HI

    # H II contribution (Z = 1, Z² = 1):
    Sil_HII_contr       = nHII * ( G̃0_HII  +  ϕ_sil_sq_logΛ_sil * G̃2_HII )

    # H₂ contribution (Z = 0):
    Sil_H2_contr        = nH2 * G̃0_H2 

    # He I contribution (Z = 0):
    Sil_HeI_contr       = nHeI * G̃0_HeI 

    # He II contribution (Z = 1, Z² = 1):
    Sil_HeII_contr      = nHeII * ( G̃0_HeII  +  ϕ_sil_sq_logΛ_sil * G̃2_HeII )

    # He III contribution (Z = 2, Z² = 4):

    if abs(Zgr_sil) > Zgr_th_Kdrag
        # Factor 1 / Z = 0.5 is used:
        absΛ_sil_05             = max( abs(Λ_sil * 0.5), 1.0 )      # Ensure that ln(Λ * 0.5) > 0
        ϕ_sil_sq_logΛ_sil_HeIII = ϕ_sil_sq * log( absΛ_sil_05 )
    else
        ϕ_sil_sq_logΛ_sil_HeIII = 0.0
    end

    Sil_HeIII_contr     = nHeIII * ( G̃0_HeIII  +  4 * ϕ_sil_sq_logΛ_sil_HeIII * G̃2_HeIII )

    # e⁻ contribution (Z = - 1, Z² = 1):
    Sil_e_contr         = ne * ( G̃0_e  +  ϕ_sil_sq_logΛ_sil * G̃2_e )

    # Put it all together:
    K_drag_sil          = K_drag_sil_prefact * ( Sil_HI_contr   + Sil_HII_contr 
                                               + Sil_H2_contr   + Sil_HeI_contr 
                                               + Sil_HeII_contr + Sil_HeIII_contr 
                                               + Sil_e_contr )

    ############# G R A P H I T E  G R A I N S #########################

    # Evaluate ϕ_C² log(Λ_C) safely:
    if abs(Zgr_C) > Zgr_th_Kdrag
        absΛ_C        = max( abs(Λ_C), 1.0 )    # Ensure that ln(Λ) > 0
        ϕ_C_sq_logΛ_C = ϕ_C_sq * log( absΛ_C )
    else 
        # In case Zgr_C = 0, Λ becomes numerically problematic, 
        # but ϕ_C → 0, so we just get 0 regardless:
        ϕ_C_sq_logΛ_C = 0.0
    end

    K_drag_C_prefact  = Fdrag_C_prefact * ρd * T

    # H I contribution (Z = 0, so only need G₀ / urel):
    C_HI_contr        = nHI * G̃0_HI

    # H II contribution (Z = 1, Z² = 1):
    C_HII_contr       = nHII * ( G̃0_HII  +  ϕ_C_sq_logΛ_C * G̃2_HII )

    # H₂ contribution (Z = 0):
    C_H2_contr        = nH2 * G̃0_H2 

    # He I contribution (Z = 0):
    C_HeI_contr       = nHeI * G̃0_HeI 

    # He II contribution (Z = 1, Z² = 1):
    C_HeII_contr      = nHeII * ( G̃0_HeII  +  ϕ_C_sq_logΛ_C * G̃2_HeII )

    # He III contribution (Z = 2, Z² = 4):

    if abs(Zgr_C) > Zgr_th_Kdrag
        # Factor 1 / Z = 0.5 is used:
        absΛ_C_05           = max( abs(Λ_C * 0.5), 1.0 )    # Ensure that ln(Λ * 0.5) > 0
        ϕ_C_sq_logΛ_C_HeIII = ϕ_C_sq * log( absΛ_C_05 )
    else
        ϕ_C_sq_logΛ_C_HeIII = 0.0
    end

    C_HeIII_contr     = nHeIII * ( G̃0_HeIII  +  4 * ϕ_C_sq_logΛ_C_HeIII * G̃2_HeIII )

    # e⁻ contribution (Z = - 1, Z² = 1):
    C_e_contr         = ne * ( G̃0_e  +  ϕ_C_sq_logΛ_C * G̃2_e )

    # Put it all together:
    K_drag_C          = K_drag_C_prefact * ( C_HI_contr   + C_HII_contr 
                                           + C_H2_contr   + C_HeI_contr 
                                           + C_HeII_contr + C_HeIII_contr 
                                           + C_e_contr )


    # Total drag force coefficient:
    K = K_drag_sil + K_drag_C

    return K 
end


###########################################################################
#######                                                             #######
#######       D U S T  G R O W T H  &  D E S T R U C T I O N        #######
#######                                                             #######
###########################################################################


""" This function updates the dust density ρd from the BDF1 solution to:

    ∂ρd/∂t = ρd * [1 - ρd / (ρd + ρZ)] / tgr0  -  ρd / tdest ,
    
    for a given (zero-dust depletion) accretion growth time-scale tgr0,
    and destruction time-scale tdest. After writing a BDF1 update, one 
    obtains the quadratic formula:
    
    x² + Ax - B = 0
    
    where x = ρd_new, and:
    
    A = (ρd + ρZ) * [tgr0 * (1/Δtsub + 1/tdest) - 1]
    B = ρd_old * tgr0 * (ρd + ρZ) / Δtsub .
    
    This is used to solve for ρd_new. """
@inline function ρd_Update_BDF1( ρd_old::Float64, 
                                 ρZ_old::Float64,
                                   tgr0::Float64, 
                                  tdest::Float64,
                                  Δtsub::Float64
                                 )

    # Convenient variables:
    ρdplusZ   = ρd_old + ρZ_old   # Total dust + gas phase metal density (conserved)
    inv_tdest = 1 / tdest
    inv_Δtsub = 1 / Δtsub

    # Define A and B that enter the BDF1 update:
    A         = ρdplusZ * ( tgr0 * ( inv_Δtsub  +  inv_tdest )  -  1.0 )
    B         = ρd_old * ρdplusZ * tgr0 * inv_Δtsub    

    if A ≥ 0.0
        # In this case, simplify the quadratic solution to avoid 
        # dangerous cancellations:
        ρd_new = 2 * B / ( A + sqrt( A^2  +  4 * B ) )
    elseif A < 0.0
        # Otherwise, use original version of formula:
        A_over_2 = 0.5 * A  
        ρd_new   = - A_over_2 + sqrt( A_over_2^2  +  B ) 
    end

    # Get updated gas phase metal density:
    ρZ_new = max( ρdplusZ - ρd_new, 0.0 )

    return ρd_new, ρZ_new
end


const Td_Boisson24       = [ 50.0, 100.0, 150.0, 250.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0, 1750.0, 2000.0, 2250.0 ]
const Stick_CI_Boisson24 = [ 0.276, 0.116, 0.064, 0.092, 0.166, 0.231, 0.247, 0.295, 0.299, 0.335, 0.347, 0.380 ]

const itp_Stick_CI_Boisson24 = extrapolate( interpolate( (Td_Boisson24,), Stick_CI_Boisson24, Gridded(Linear())),
                                            Flat() )

""" The estimated chemisorption sticking coefficient of C I on amorphous carbon grains,
    as computed by Boisson et al. (2024). """
@inline function Stick_CI( Td::Float64 )
    itp_Stick_CI_Boisson24( Td )
end                                           
                                        

const agr_C_over_01::Float64   = agr_C   / (0.1 * μm)
const agr_sil_over_01::Float64 = agr_sil / (0.1 * μm)

""" This function computes the dust gas accretion time-scale (sec)
    tgr0 (in the low-depletion limit). """
@inline function Compute_tgr0( nH::Float64,
                                T::Float64,
                               Td::Float64,
                              xCI::Float64,
                        Ztot_Zsun::Float64
                              )

    # Convenient variable:
    Stuff    = (1 / Ztot_Zsun) * (1 / nH) * 1 / sqrt(T)

    # Sticking coefficient for carbon species (CO doesn't chemisorb, 
    # and C II would just recombine and enter the gas-phase again):
    Stick_C  = Stick_CI( Td ) * xCI 

    # Sticking coefficient for Si on silicate grains is not really known, 
    # but silicate grain accretion will generally be slower than the growth of
    # carbon grains, so for simplicity assume the same sticking coefficient:
    Stick_Si = Stick_C

    # Accretion time-scales for graphite/carbonaceous & silicate dust:
    tgr0_C   = 3.14e10 * yr * agr_C_over_01   * Stuff * 1 / Stick_C 
    tgr0_sil = 5.09e10 * yr * agr_sil_over_01 * Stuff * 1 / Stick_Si 
    
    # Overall dust growth time-scale:
    tgr0     = 1.0 / ( fsil / tgr0_sil  +  OneMinusfsil / tgr0_C )

    return max( tgr0, 0.1 * yr ) 
end


const ΔTsil_subl::Float64      = 68100.0 
const ΔTC_subl::Float64        = 81200.0
const inv_Td_sil_subl::Float64 = 1 / 1800.0 
const inv_Td_C_subl::Float64   = 1 / 3000.0
const tsubl_sil_0::Float64     = 6.36e3 / 3   # Pre-factor for silicates, with factor 1/3 converting from size to mass destruction time-scale 
const tsubl_C_0::Float64       = 1.36 / 3     # Pre-factor for graphite/carbonaceous grains, with factor 1/3 converting from size to mass destruction time-scale 

""" This fuction computes the sublimation time-scale (sec) for dust. """
@inline function Compute_tsubl( Td::Float64 ) 

    Td = clamp( Td, Td_floor, Td_max )

    # Sublimation time-scale for silicate dust:
    tsubl_sil = tsubl_sil_0 * agr_sil_over_01 * exp( ΔTsil_subl * (1 / Td  -  inv_Td_sil_subl) )

    # Sublimation time-scale for graphite/carbonaceous dust:
    tsubl_C   = tsubl_C_0 * agr_C_over_01 * exp( ΔTC_subl * (1 / Td  -  inv_Td_C_subl) )

    # Overall sublimation time-scale: 
    tsubl     = 1 / ( fsil / tsubl_sil  +  OneMinusfsil / tsubl_C )

    # Return time-scale, and limit it to avoid way too short time-steps in the 
    # sub-cycling:
    return max( tsubl, 0.1 * yr ) 
end


const ρ_sputter::Float64            = 1e-27 
const T_sputter::Float64            = 2e6 
const ω_sputter::Float64            = 2.5
const agr_sputter_norm_sil::Float64 = agr_sil / (0.1 * μm)
const agr_sputter_norm_C::Float64   = agr_C   / (0.1 * μm)
const tsputter_th_prefact::Float64  = OneThird * 0.17 * Gyr  # Factor 1/3 to convert to *mass* sputtering time

""" Destruction time-scale (s) from thermal sputtering. """
@inline function Compute_tsputter_thermal( ρ::Float64, 
                                           T::Float64,
                                    Graphite::Bool )

    if T ≤ 1e5
        # Ignore sputtering below 10⁵ K, where the fit breaks down anyway.
        return Inf64
    else
        if Graphite
            anorm = agr_sputter_norm_C
        else
            anorm = agr_sputter_norm_sil
        end

        # The sputtering time-scale agr / 3 |d(agr)/dt| for the grain radius:
        t_sputter = tsputter_th_prefact * anorm * (ρ_sputter/ρ) * ( (T_sputter/T)^ω_sputter + 1 )

        return max( t_sputter, 0.1 * yr )
    end
end


const asput0_sil::Float64 = - 32.4;  const asput1_sil::Float64 = 28.6
const asput2_sil::Float64 = - 11.6;  const asput3_sil::Float64 = 2.23
const asput4_sil::Float64 = - 0.208; const asput5_sil::Float64 = 0.00736

const asput0_C::Float64 = - 48.7;  const asput1_C::Float64 = 55.7
const asput2_C::Float64 = - 28.8;  const asput3_C::Float64 = 7.44
const asput4_C::Float64 = - 0.965; const asput5_C::Float64 = 0.005

const t_sputter_nonthermal_0_sil::Float64 = 0.33 * Myr * ( agr_sil / μm )
const t_sputter_nonthermal_0_C::Float64   = 0.33 * Myr * ( agr_C   / μm )

""" The non-thermal sputtering time for grains moving 
    at velocity urel (cm s⁻¹) relative to gas of density nH (cm⁻³).
    Uses fits from Hu et al. (2019). """
function Compute_tsputter_nonthermal( nH::Float64,
                                    urel::Float64,
                                Graphite::Bool
                                     )

    urel = min( abs( urel ), 1e5 * kms )

    # Non-thermal sputtering is only important at very high velocities:
    if urel ≥ 30.0 * kms
        if Graphite
            a0 = asput0_C;   a1 = asput1_C
            a2 = asput2_C;   a3 = asput3_C
            a4 = asput4_C;   a5 = asput5_C

            t_sputter0 = t_sputter_nonthermal_0_sil
        else
            a0 = asput0_sil; a1 = asput1_sil
            a2 = asput2_sil; a3 = asput3_sil
            a4 = asput4_sil; a5 = asput5_sil

            t_sputter0 = t_sputter_nonthermal_0_C
        end

        x    = log10(urel / kms); x2 = x^2; x3 = x2*x; x4 = x3*x; x5 = x4*x 
        y    = a0 + a1*x + a2*x2 + a3*x3 + a4*x4 + a5*x5 
        Ytot = max( 10^y, 0.0 )  

        # The non-thermal sputtering time:
        t_sputter = t_sputter0 * ( 1 / nH ) * ( 1 / Ytot )

        return t_sputter
    else
        # Ignore non-thermal sputtering at very low relative velocities:
        t_sputter = Inf64

        return max( t_sputter, 0.1 * yr )
    end
end


""" This function updates the dust and gas metal density, considering 
    dust growth by accretion, and destruction by sublimation, thermal
    sputtering, and non-thermal sputtering. """
@inline function DustGrowthDestruction_Update( ρd_old::Float64, 
                                               ρZ_old::Float64,  
                                                ρ_old::Float64,
                                                   nH::Float64, 
                                                    T::Float64, 
                                                   Td::Float64,
                                                 urel::Float64,
                                                  xCI::Float64,
                                            Ztot_Zsun::Float64,
                                                Δtsub::Float64
                                               )

    # Gas accretion dust growth time-scale:
    tgr0  = Compute_tgr0( nH, T, Td, xCI, Ztot_Zsun )

    # Sublimation time-sale:
    tdest_subl          = Compute_tsubl( Td )

    # Thermal sputtering time-scale:
    tdest_sputt_th_sil  = Compute_tsputter_thermal( ρ_old, T, false )
    tdest_sputt_th_C    = Compute_tsputter_thermal( ρ_old, T, true )
    tdest_sputt_th      = 1 / ( (fsil / tdest_sputt_th_sil)  +  (OneMinusfsil / tdest_sputt_th_C) )

    # Non-thermal sputtering time-scale:
    tdest_sputt_nth_sil = Compute_tsputter_nonthermal( nH, urel, false )
    tdest_sputt_nth_C   = Compute_tsputter_nonthermal( nH, urel, true )
    tdest_sputt_nth     = 1 / ( (fsil / tdest_sputt_nth_sil)  +  (OneMinusfsil / tdest_sputt_nth_C) )

    # Get dust destruction time-scale, considering all processes:
    tdest = 1 / ( (1 / tdest_subl) + (1 / tdest_sputt_th) + (1 / tdest_sputt_nth) )

    # Get updated dust & metal density:
    ρd_new, ρZ_new = ρd_Update_BDF1( ρd_old, ρZ_old, tgr0, tdest, Δtsub )

    return ρd_new, ρZ_new
end 



########################################################################################
#######                                                                          #######
#######           R E A D  I N  D U S T  D A T A  F O R  R T  B A N D S          #######
#######                                                                          #######
########################################################################################

DustData  = readdlm( joinpath(@__DIR__, "DraineDust_R_V5.5.txt"), skipstart = 48 )

# Get the λ (μm) points in the data file:
λData     = DustData[:,1]
order     = sortperm(λData)  # Get correct order, so that it goes from low to high λ instead of the reverse
λData     = λData[order]

κabsData  = DustData[:,5][order]
ωData     = DustData[:,2][order]
gData     = DustData[:,3][order]
κextData  = κabsData ./ (1 .- ωData)
κscData   = ωData .* κextData

# Get interpolations:
const κabs_itp = LinearInterpolation( λData, κabsData, extrapolation_bc = Flat() )
const κsc_itp  = LinearInterpolation( λData, κscData,  extrapolation_bc = Flat() )
const g_itp    = LinearInterpolation( λData, gData,    extrapolation_bc = Flat() )

@inline λμm_from_ν( ν::Float64 ) = (c / ν) / μm               # Compute λ in μm from ν (Hz)
@inline κabs_ν( ν::Float64 )     = κabs_itp( λμm_from_ν(ν) )
@inline κsc_ν( ν::Float64 )      = κsc_itp( λμm_from_ν(ν) )
@inline g_ν( ν::Float64 )        = g_itp( λμm_from_ν(ν) )

# Energies (eV) for integration intervals:
const Emin_opt::Float64  = 1.0;    const Emax_opt::Float64  = 5.8
const Emin_FUV::Float64  = 5.8;    const Emax_FUV::Float64  = 11.2
const Emin_LW::Float64   = 11.2;   const Emax_LW::Float64   = 13.6
const Emin_EUV1::Float64 = 13.6;   const Emax_EUV1::Float64 = 24.6
const Emin_EUV2::Float64 = 24.6;   const Emax_EUV2::Float64 = 54.4
const Emin_EUV3::Float64 = 54.4;   const Emax_EUV3::Float64 = h * c / (minimum(λData) * μm * eV)


""" Planck spectrum, needed for the SED-weighted optical properties. """
@inline function Bν( ν::Float64, Teff::Float64 )
    return (2 * h * (ν^3) / (c^2)) / ( exp(h * ν / (kB * Teff)) - 1)
end


""" Compute Planck-weighted dust properties over energy interval E (eV) ∈ [E1, E2]:

κabs = ⟨κabs⟩
κsc  = ⟨κsc⟩
κext = κabs + κsc
ω    = κsc / κext             (from band means)
ḡ    = ⟨g⟩                    (weighted by κsc * Bν)
κF   = ⟨κabs + κsc * (1 - g(ν))⟩ """
function DustBandMeans( E1::Float64, E2::Float64, Teff::Float64 )

    @assert E2 > E1 > 0.0 "E2 must be larger than E1."

    # Get frequencies (Hz) corresponding to energy interval:
    ν2 = E2 * eV / h 
    ν1 = E1 * eV / h 

    # Convenient variable change, ν = exp(x):
    x1, x2 = log(ν1), log(ν2)

    # Normalization integral, with dν = exp(x) dx:
    I0,  _ = quadgk(x -> Bν(exp(x), Teff) * exp(x), x1, x2)

    # Non-normalized integrals for all quantities of interest:
    Iabs,_ = quadgk(x -> κabs_ν(exp(x)) * Bν(exp(x), Teff) * exp(x), x1, x2)
    Isc, _ = quadgk(x -> κsc_ν(exp(x))  * Bν(exp(x), Teff) * exp(x), x1, x2)

    Ig,  _ = quadgk(x -> g_ν(exp(x)) * κsc_ν(exp(x)) * Bν(exp(x), Teff) * exp(x), x1, x2)

    IF,  _ = quadgk(x -> (κabs_ν(exp(x)) + κsc_ν(exp(x))*(1 - g_ν(exp(x)))) *
                         Bν(exp(x), Teff) * exp(x), x1, x2)

    # Normalize:
    κabs = Iabs / I0
    κsc  = Isc  / I0
    κF   = IF   / I0

    κext = κabs + κsc
    ω    = (κext > 0) ? (κsc / κext) : 0.0
    ḡ    = (Isc  > 0) ? (Ig  / Isc)  : 0.0

    return κabs, κsc, κext, ω, ḡ, κF
end


""" The Planck spectrum B_λ(T) (in cgs units), with limiting values used for numerical stability. 
    λ is in μm. """
@inline function Bλ_μm( λμm::Float64, T::Float64 )
    λ = λμm * μm
    x = h * c / (λ * kB * T)

    pref = (2 * h * c^2) / (λ^5)

    if x > 50.0
        # Wien limit: 1/(eˣ-1) ≈ e⁻ˣ
        return pref * exp(-x)
    elseif x < 1e-4
        # Rayleigh–Jeans limit: Bλ ≈ 2 c kB T / λ⁴
        return (2 * c * kB * T) / (λ^4)
    else
        return pref / expm1(x)
    end
end


""" The derivative ∂B_λ/∂T (in cgs units), with limiting values used for numerical stability.  
    λ is in μm. """
@inline function ∂Bλ∂T_μm( λμm::Float64, T::Float64 )
    λ = λμm * μm
    x = h * c / (λ * kB * T)

    if x > 50.0
        # In Wien limit, eˣ/(eˣ-1) → 1
        B = (2 * h * c^2) / (λ^5) * exp(-x)
        return B * (x / T)
    elseif x < 1e-4
        # d/dT of Rayleigh–Jeans limit: 2 c kB / λ⁴
        return (2 * c * kB) / (λ^4)
    else
        exm1 = expm1(x)              # eˣ - 1, stable
        ex   = exm1 + 1.0            # eˣ
        B    = (2 * h * c^2) / (λ^5) / exm1
        fac  = ex / exm1             # eˣ/(eˣ-1), stable for moderate x
        return B * (x / T) * fac
    end
end


""" Planck-mean absorption opacity, using κabs_itp(λ) (cm² g⁻¹ of dust). """
function Compute_κP( T::Float64,
                  λmin::Float64 = minimum(λData),
                  λmax::Float64 = maximum(λData) )

    x1, x2 = log(λmin), log(λmax)

    denom, _ = quadgk(x -> Bλ_μm(exp(x), T) * exp(x), x1, x2)
    numer, _ = quadgk(x -> κabs_itp(exp(x)) * Bλ_μm(exp(x), T) * exp(x), x1, x2)

    return numer / denom
end


""" Rosseland-mean opacity for the flux, using κ_F(λ) = κabs + κsc * (1 - g). """
function Compute_κR( T::Float64,
                  λmin::Float64 = minimum(λData),
                  λmax::Float64 = maximum(λData) )

    x1, x2 = log(λmin), log(λmax)

    κλ = λ -> (κabs_itp(λ) + κsc_itp(λ) * (1.0 - g_itp(λ)))

    denom, _ = quadgk(x -> ∂Bλ∂T_μm(exp(x), T) * exp(x), x1, x2)

    numer, _ = quadgk(x -> (∂Bλ∂T_μm(exp(x), T) * exp(x)) / κλ(exp(x)), x1, x2)

    return denom / numer   # since 1 / κR = numer / denom
end


""" Save Planck-mean absorption opacity κP(T) and Rosseland-mean flux opacity κR(T)
    to a text file.  """
function SaveκMeansTable( FileName::String,
                              Tmin::Float64 = 1.0,
                              Tmax::Float64 = 2.0e3,
                                NT::Int     = 300 )

    @assert Tmin > 0.0
    @assert Tmax > Tmin
    @assert NT ≥ 10

    Tgrid = exp.( range(log(Tmin), log(Tmax), length = NT) )

    open(FileName, "w") do io
        println(io, "# Dust mean opacities vs temperature")
        println(io, "# Dust table source: DraineDust_R_V5.5.txt (K_abs column is in cm² g⁻¹ of dust).")
        println(io, "# κP(T) = ∫ κabs(λ) B_λ dλ / ∫ B_λ dλ")
        println(io, "# κR(T) uses κ_F(λ) = κabs(λ) + κsc(λ) * ( 1 - g(λ) ) in the Rosseland integral.")
        println(io, "# λ bounds: λmin_μm = $(minimum(λData)), λmax_μm = $(maximum(λData))")
        println(io, "# Columns:")
        println(io, "# T (K)   κP (cm² g⁻¹ of dust)   κR (cm² g⁻¹ of dust)")
        for T in Tgrid
            κP = Compute_κP(T)
            κR = Compute_κR(T)
            println(io, @sprintf("%12.6e  %12.6e  %12.6e", T, κP, κR))
        end
    end

    return nothing
end


const λ_Lyα_μm::Float64 = λμm_from_ν( νLyα )   # Lyα wavelength in μm

""" Compute and save band-averaged dust optical properties (stellar RT bands + Lyα line center)
    to a text file for later reuse.
        
    All κ values are per dust mass, i.e. cm² g⁻¹ of dust. """
function SaveDustOpacities( FileName::String, Teff::Float64 = 4.8e4 )

    # Stellar RT band-averaged optical properties:
    κabs_opt,  κsc_opt,  κext_opt,  ω_opt,  g_opt,  κF_opt  = DustBandMeans( Emin_opt,  Emax_opt,  Teff )
    κabs_FUV,  κsc_FUV,  κext_FUV,  ω_FUV,  g_FUV,  κF_FUV  = DustBandMeans( Emin_FUV,  Emax_FUV,  Teff )
    κabs_LW,   κsc_LW,   κext_LW,   ω_LW,   g_LW,   κF_LW   = DustBandMeans( Emin_LW,   Emax_LW,   Teff )
    κabs_EUV1, κsc_EUV1, κext_EUV1, ω_EUV1, g_EUV1, κF_EUV1 = DustBandMeans( Emin_EUV1, Emax_EUV1, Teff )
    κabs_EUV2, κsc_EUV2, κext_EUV2, ω_EUV2, g_EUV2, κF_EUV2 = DustBandMeans( Emin_EUV2, Emax_EUV2, Teff )
    κabs_EUV3, κsc_EUV3, κext_EUV3, ω_EUV3, g_EUV3, κF_EUV3 = DustBandMeans( Emin_EUV3, Emax_EUV3, Teff )

    # Lyα (line center, not a band mean):
    κabs_Lyα = κabs_itp( λ_Lyα_μm )
    κsc_Lyα  = κsc_itp( λ_Lyα_μm )
    g_Lyα    = g_itp( λ_Lyα_μm )
    κext_Lyα = κabs_Lyα + κsc_Lyα
    ω_Lyα    = (κext_Lyα > 0.0) ? (κsc_Lyα / κext_Lyα) : 0.0
    κF_Lyα   = κabs_Lyα + κsc_Lyα * (1.0 - g_Lyα)   # same definition as in DustBandMeans :contentReference[oaicite:4]{index=4}

    # Package into rows:
    rows = [
        ("opt",  Emin_opt,  Emax_opt,  κabs_opt,  κsc_opt,  κext_opt,  ω_opt,  g_opt,  κF_opt),
        ("FUV",  Emin_FUV,  Emax_FUV,  κabs_FUV,  κsc_FUV,  κext_FUV,  ω_FUV,  g_FUV,  κF_FUV),
        ("LW",   Emin_LW,   Emax_LW,   κabs_LW,   κsc_LW,   κext_LW,   ω_LW,   g_LW,   κF_LW),
        ("EUV1", Emin_EUV1, Emax_EUV1, κabs_EUV1, κsc_EUV1, κext_EUV1, ω_EUV1, g_EUV1, κF_EUV1),
        ("EUV2", Emin_EUV2, Emax_EUV2, κabs_EUV2, κsc_EUV2, κext_EUV2, ω_EUV2, g_EUV2, κF_EUV2),
        ("EUV3", Emin_EUV3, Emax_EUV3, κabs_EUV3, κsc_EUV3, κext_EUV3, ω_EUV3, g_EUV3, κF_EUV3),
        ("Lyα",  NaN,       NaN,       κabs_Lyα,  κsc_Lyα,  κext_Lyα,  ω_Lyα,  g_Lyα,  κF_Lyα),
    ]

    open(FileName, "w") do io
        println(io, "# Dust stellar-band opacities (band means) + Lyα line center")
        println(io, "# Dust table source: DraineDust_R_V5.5.txt (K_abs column is cm^2/g_dust).")  # :contentReference[oaicite:5]{index=5}
        println(io, "# Teff_K = $(Teff)")
        println(io, "# Lyα: λ_Lyα_μm = $(λ_Lyα_μm)")
        println(io, "# Columns:")
        println(io, "# Band  Emin_eV  Emax_eV  kappa_abs  kappa_scat  kappa_ext  albedo  g  kappa_F")
        for row in rows
            band, E1, E2, κa, κs, κe, ω, g, κF = row
            # Use consistent scientific formatting
            println(io,
                @sprintf("%-4s  %12.6e  %12.6e  %12.6e  %12.6e  %12.6e  %10.6e  %10.6e  %12.6e",
                         band, E1, E2, κa, κs, κe, ω, g, κF)
            )
        end
    end

    return nothing
end


""" This function read the saved stellar-band + Lyα dust opacity table produced by SaveDustOpacities.

    Returns a Dict{String, NamedTuple} mapping band name -> (κabs, κF, ω, g, κsc, κext).
    All κ are cm² g⁻¹ of dust.
"""
function LoadDustOpacities( FileName::String )

    # read lines manually to preserve band names cleanly
    lines = readlines(FileName)

    bands = Dict{String, NamedTuple}()

    for ln in lines
        isempty(ln) && continue
        startswith(ln, "#") && continue

        parts = split(strip(ln))
        # expected columns:
        # band Emin Emax κabs κsc κext ω g κF
        @assert length(parts) ≥ 9 "Unexpected format in Dust opacity file: $FileName"

        band = parts[1]
        E1   = parse(Float64, parts[2])  # unused except for debugging
        E2   = parse(Float64, parts[3])  # unused
        κabs = parse(Float64, parts[4])
        κsc  = parse(Float64, parts[5])
        κext = parse(Float64, parts[6])
        ω    = parse(Float64, parts[7])
        g    = parse(Float64, parts[8])
        κF   = parse(Float64, parts[9])

        bands[band] = (κabs=κabs, κsc=κsc, κext=κext, ω=ω, g=g, κF=κF, Emin=E1, Emax=E2)
    end

    # sanity: require all expected keys
    for b in ("opt", "FUV", "LW", "EUV1", "EUV2", "EUV3", "Lyα")
        @assert haskey(bands, b) "Missing band '$b' in Dust opacity file: $FileName"
    end

    return bands
end


""" This function makes interpolations for the Planck & Rosseland-mean opacities, 
    using the relevant saved data file. """
function LoadκMeansInterpolations( FileName::String )
    Data  = readdlm(FileName, comments=true)
    T     = Data[:,1]
    κP    = Data[:,2]
    κR    = Data[:,3]

    logT  = log10.(T)
    logκP = log10.(κP)
    logκR = log10.(κR)

    itpP = LinearInterpolation(logT, logκP, extrapolation_bc = Flat())
    itpR = LinearInterpolation(logT, logκR, extrapolation_bc = Flat())

    return itpP, itpR
end


""" Convenient function for reading saved dust data files. """
@inline function Band( bands, key::String )
    return bands[key]
end


const Teff_SetupDustState::Float64 = 4.8e4 # Assume a fixed black-body temperature for the band-averaging

""" This function sets up a dust struct containing needed dust optical 
    properties. """
function SetupDustState( DustGrowthDestruction::Bool,
                                           N_r::Int     = 128, 
                                           N_Z::Int     = 128,
                                           Td0::Float64 = 30.0,
                               DustOpacityFile::String  = joinpath(@__DIR__, "DustData.txt"),
                               DustIRMeansFile::String  = joinpath(@__DIR__, "DustIRData.txt"),
                            RecomputeDustFiles::Bool    = true,
                                       Tmin_IR::Float64 = 1.0,
                                       Tmax_IR::Float64 = 2.0e3,
                                         NT_IR::Int     = 300 )

    # Dust temperature for every cell:
    Td            = zeros( Float64, N_r, N_Z )
    Td           .= Td0

    # Grain charges:
    Zgr_sil       = zeros( Float64, N_r, N_Z )
    Zgr_C         = zeros( Float64, N_r, N_Z )

    # Make sure the saved dust files exist (or rebuild)
    if RecomputeDustFiles || !isfile( DustOpacityFile )
        SaveDustOpacities( DustOpacityFile, Teff_SetupDustState )
    end

    if RecomputeDustFiles || !isfile( DustIRMeansFile )
        SaveκMeansTable( DustIRMeansFile, Tmin_IR, Tmax_IR, NT_IR )
    end

    # Load pre-computed opacities:
    bands = LoadDustOpacities( DustOpacityFile )

    # Get each band:
    opt   = Band( bands, "opt")
    FUV   = Band( bands, "FUV")
    LW    = Band( bands, "LW")
    EUV1  = Band( bands, "EUV1")
    EUV2  = Band( bands, "EUV2")
    EUV3  = Band( bands, "EUV3")
    Lyα   = Band( bands, "Lyα" )

    # Set up interpolations of Planck & Rosseland-mean opacities:
    logκP_itp, logκR_itp = LoadκMeansInterpolations( DustIRMeansFile )

    return DustState( # Dust growth/destruction setting:
                      DustGrowthDestruction,

                      # Dust temperature for every cell:
                      Td, 

                      # Grain charges:
                      Zgr_sil, Zgr_C,
    
                      # Band absorption opacities (cm² g⁻¹ of dust):
                      opt.κabs,
                      FUV.κabs,
                      LW.κabs,
                      EUV1.κabs,
                      EUV2.κabs,
                      EUV3.κabs,
                      Lyα.κabs,

                      # Band flux opacities (cm² g⁻¹ of dust):
                      opt.κF,
                      FUV.κF,
                      LW.κF,
                      EUV1.κF,
                      EUV2.κF,
                      EUV3.κF,
                      Lyα.κF,

                      # Band albedos:
                      opt.ω,
                      FUV.ω,
                      LW.ω,
                      EUV1.ω,
                      EUV2.ω,
                      EUV3.ω,
                      Lyα.ω,

                      # Band scattering asymmetry:
                      opt.g,
                      FUV.g,
                      LW.g,
                      EUV1.g,
                      EUV2.g,
                      EUV3.g,
                      Lyα.g,
                      
                      # Interpolations:
                      logκP_itp, logκR_itp
                      
                      )

end