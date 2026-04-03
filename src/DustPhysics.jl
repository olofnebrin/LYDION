
""" FILE:        DustPhysics.jl

    PURPOSE:     Contains the code for computing dust properties (e.g. dust temperature, charges,
                 opacities), dust growth/destruction rates, and setting up the dust struct.

"""

using .Structs: DustState
using .ConstantsAndFunctions
using DelimitedFiles          # For reading data 
using Interpolations          # For interpolation of the Planck mean opacity
using QuadGK                  # For integrating over data 
using Printf

# Dust types/bins:
const Sil = Val(:Sil)  # Silicate grains 
const Gra = Val(:Gra)  # Larger graphite grains (> 0.001 μm)
const PAH = Val(:PAH)  # Smaller PAH's (< 0.001 μm)

const ρgr_Sil::Float64       = 3.5              # Density of a silicate grain (g cm⁻³)
const ρgr_C::Float64         = 2.24             # Density of a graphite/carbonaceous grain (g cm⁻³)
const inv_ρgr_Sil::Float64   = 1 / ρgr_Sil      # Inverse of ρgr_sil, useful 
const inv_ρgr_C::Float64     = 1 / ρgr_C        # Inverse of ρgr_C, useful 
const agr_Sil::Float64       = 1.3820e-01 * μm  # ⟨a³⟩/⟨a²⟩ for silicate grains         (see file grain_size_stats.txt)
const agr_C::Float64         = 5.2953e-02 * μm  # ⟨a³⟩/⟨a²⟩ for larger graphite grains  (see file grain_size_stats.txt)
const agr_PAH::Float64       = 5.6236e-04 * μm  # ⟨a³⟩/⟨a²⟩ for smaller PAH grains      (see file grain_size_stats.txt)
const inv_agr_Sil::Float64   = 1 / agr_Sil      # Inverse of agr_sil, useful 
const inv_agr_C::Float64     = 1 / agr_C        # Inverse of agr_Graph, useful 
const inv_agr_PAH::Float64   = 1 / agr_PAH      # Inverse of agr_PAH, useful 
const fd_Sil::Float64        = 0.726444         # Initial fraction of total dust mass in silicate grains, assuming the WD01 R_V = 5.5, b_C = 3 × 10⁻⁵ model
const fd_C::Float64          = 0.251721         # Initial fraction of total dust mass in larger graphite grains, assuming the WD01 R_V = 5.5, b_C = 3 × 10⁻⁵ model
const fd_PAH::Float64        = 0.021835         # Initial fraction of total dust mass in PAH grains, assuming the WD01 R_V = 5.5, b_C = 3 × 10⁻⁵ model 

# Number of carbon atoms in a representative PAH (Li & Draine 2001):
const NC_PAH::Float64        = 468 * ((agr_PAH / (10 * Ang))^3)

# We will need Qabs evaluated for the area-weighted grain sizes to compute the grain charges 
# per bin/type. We read them from the saved data file (computed in ComputeDustOpacities.jl).
# Note, we skip the Qabs_opt, since optical photons don't contribute to grain charging:

Qabs_Data = readdlm( joinpath(@__DIR__, "DustOpacityData", "Qabs_at_area_mean.txt"), skipstart = 5 ) # Skip 5 first comment/header lines 

const Qabs_PAH_FUV::Float64  = Qabs_Data[2,4];  const Qabs_C_FUV::Float64  = Qabs_Data[2,5];  const Qabs_Sil_FUV::Float64  = Qabs_Data[2,6]
const Qabs_PAH_LW::Float64   = Qabs_Data[3,4];  const Qabs_C_LW::Float64   = Qabs_Data[3,5];  const Qabs_Sil_LW::Float64   = Qabs_Data[3,6]
const Qabs_PAH_EUV1::Float64 = Qabs_Data[4,4];  const Qabs_C_EUV1::Float64 = Qabs_Data[4,5];  const Qabs_Sil_EUV1::Float64 = Qabs_Data[4,6]
const Qabs_PAH_EUV2::Float64 = Qabs_Data[5,4];  const Qabs_C_EUV2::Float64 = Qabs_Data[5,5];  const Qabs_Sil_EUV2::Float64 = Qabs_Data[5,6]
const Qabs_PAH_EUV3::Float64 = Qabs_Data[6,4];  const Qabs_C_EUV3::Float64 = Qabs_Data[6,5];  const Qabs_Sil_EUV3::Float64 = Qabs_Data[6,6]
const Qabs_PAH_Lyα::Float64  = Qabs_Data[7,4];  const Qabs_C_Lyα::Float64  = Qabs_Data[7,5];  const Qabs_Sil_Lyα::Float64  = Qabs_Data[7,6]


# Export functions:
export GetDustAbsorptionOpacities, GetDustFluxOpacities, Λdg_tot, Λdg, ComputeTd, FindGrainCharge, 
       HeatingRate_PhotoElectric, SetupDustState, κP, κR


""" Extract the band absorption opacities for a given dust species from the struct. """
@inline function GetDustAbsorptionOpacities( Dust::DustState, ::Val{:Sil} )
    return ( Dust.κd_abs_opt_Sil,  Dust.κd_abs_FUV_Sil,  Dust.κd_abs_LW_Sil,
             Dust.κd_abs_EUV1_Sil, Dust.κd_abs_EUV2_Sil, Dust.κd_abs_EUV3_Sil,
             Dust.κd_abs_Lyα_Sil )
end

@inline function GetDustAbsorptionOpacities( Dust::DustState, ::Val{:Gra} )
    return ( Dust.κd_abs_opt_C,  Dust.κd_abs_FUV_C,  Dust.κd_abs_LW_C,
             Dust.κd_abs_EUV1_C, Dust.κd_abs_EUV2_C, Dust.κd_abs_EUV3_C,
             Dust.κd_abs_Lyα_C )
end

@inline function GetDustAbsorptionOpacities( Dust::DustState, ::Val{:PAH} )
    return ( Dust.κd_abs_opt_PAH,  Dust.κd_abs_FUV_PAH,  Dust.κd_abs_LW_PAH,
             Dust.κd_abs_EUV1_PAH, Dust.κd_abs_EUV2_PAH, Dust.κd_abs_EUV3_PAH,
             Dust.κd_abs_Lyα_PAH )
end


""" Extract the band flux opacities for a given dust species from the struct. """
@inline function GetDustFluxOpacities( Dust::DustState, ::Val{:Sil} )
    return ( Dust.κd_F_opt_Sil,  Dust.κd_F_FUV_Sil,  Dust.κd_F_LW_Sil,
             Dust.κd_F_EUV1_Sil, Dust.κd_F_EUV2_Sil, Dust.κd_F_EUV3_Sil,
             Dust.κd_F_Lyα_Sil )
end

@inline function GetDustFluxOpacities( Dust::DustState, ::Val{:Gra} )
    return ( Dust.κd_F_opt_C,  Dust.κd_F_FUV_C,  Dust.κd_F_LW_C,
             Dust.κd_F_EUV1_C, Dust.κd_F_EUV2_C, Dust.κd_F_EUV3_C,
             Dust.κd_F_Lyα_C )
end

@inline function GetDustFluxOpacities( Dust::DustState, ::Val{:PAH} )
    return ( Dust.κd_F_opt_PAH,  Dust.κd_F_FUV_PAH,  Dust.κd_F_LW_PAH,
             Dust.κd_F_EUV1_PAH, Dust.κd_F_EUV2_PAH, Dust.κd_F_EUV3_PAH,
             Dust.κd_F_Lyα_PAH )
end


###########################################################################
#######                                                             #######
#######               D U S T  T E M P E R A T U R E                #######
#######                                                             #######
###########################################################################

# Useful constants:
const Two_kB::Float64         = 2 * kB

# Per-species collisional heating prefactors 1/(agr_s * ρgr_s) (cm⁻¹ g⁻¹).
# These enter Λdg/ρd_s = (3/4) * nH * vth_H * 0.5 * 2kB * (T - Td) / (a_s * ρgr_s):
const Λdg_const_Sil::Float64  = inv_agr_Sil * inv_ρgr_Sil
const Λdg_const_Gra::Float64  = inv_agr_C   * inv_ρgr_C
const Λdg_const_PAH::Float64  = inv_agr_PAH * inv_ρgr_C   # PAH shares same bulk density with graphite grains


""" Dust collisional heating (if > 0) or cooling (< 0) rate Λdg(T,Td) (in erg s⁻¹ cm⁻³).
    Equivalently, it is the cooling rate of the gas if > 0, and heating rate if < 0.
    See Draine (2011)."""
@inline function Λdg( T::Float64, Td_Sil::Float64, ρd_Sil::Float64, nH::Float64,
                      ::Val{:Sil} )
    return ThreeFourths * nH * vth_H(T) * 0.5 * Two_kB * (T - Td_Sil) * ρd_Sil * Λdg_const_Sil
end

@inline function Λdg( T::Float64, Td_C::Float64, ρd_C::Float64, nH::Float64,
                      ::Val{:Gra} )
    return ThreeFourths * nH * vth_H(T) * 0.5 * Two_kB * (T - Td_C) * ρd_C * Λdg_const_Gra
end

@inline function Λdg( T::Float64, Td_PAH::Float64, ρd_PAH::Float64, nH::Float64,
                      ::Val{:PAH} )
    return ThreeFourths * nH * vth_H(T) * 0.5 * Two_kB * (T - Td_PAH) * ρd_PAH * Λdg_const_PAH
end


""" The total dust collisional heating (if > 0) or cooling (< 0) rate (in erg s⁻¹ cm⁻³),
    considering all dust bins. """
@inline function Λdg_tot( T::Float64, 
                         Td::NTuple{3, Float64}, 
                         ρd::NTuple{3, Float64}, 
                         nH::Float64 )

    # Unpack variables:
    ρd_Sil, ρd_C, ρd_PAH = ρd
    Td_Sil, Td_C, Td_PAH = Td

    # Only consider dust collisional heating/cooling above ~ the dust floor:
    ρd_Sil_eff = (ρd_Sil > ρd_IGNORE) * ρd_Sil
    ρd_C_eff   = (ρd_C   > ρd_IGNORE) * ρd_C
    ρd_PAH_eff = (ρd_PAH > ρd_IGNORE) * ρd_PAH

    # Get the rates:
    Λdg_Sil = Λdg( T, Td_Sil, ρd_Sil_eff, nH, Sil )
    Λdg_C   = Λdg( T, Td_C,   ρd_C_eff,   nH, Gra )
    Λdg_PAH = Λdg( T, Td_PAH, ρd_PAH_eff, nH, PAH )

    return Λdg_Sil + Λdg_C + Λdg_PAH
end


""" Λdg / ρd_s for one dust bin, useful for the dust temperature solver. """
@inline function Λdg_over_ρd( T::Float64, Td_Sil::Float64, nH::Float64, ::Val{:Sil} )
    return ThreeFourths * nH * vth_H(T) * 0.5 * Two_kB * (T - Td_Sil) * Λdg_const_Sil
end

@inline function Λdg_over_ρd( T::Float64, Td_C::Float64, nH::Float64, ::Val{:Gra} )
    return ThreeFourths * nH * vth_H(T) * 0.5 * Two_kB * (T - Td_C) * Λdg_const_Gra
end

@inline function Λdg_over_ρd( T::Float64, Td_PAH::Float64, nH::Float64, ::Val{:PAH} )
    return ThreeFourths * nH * vth_H(T) * 0.5 * Two_kB * (T - Td_PAH) * Λdg_const_PAH
end


const Td_max::Float64  = 2e3   # Upper limit to dust temperature (K) - assumed sublimation above

""" Planck-mean opacity for a dust bin, as computed from the stored log-log interpolation (cm² g⁻¹). """
@inline function κP( Dust::DustState, T::Float64, ::Val{:Sil} )
    T = clamp( T, 1.0, Td_max )
    return 10.0^( Dust.logκP_Sil_itp( log10(T) ) )
end

@inline function κP( Dust::DustState, T::Float64, ::Val{:Gra} )
    T = clamp( T, 1.0, Td_max )
    return 10.0^( Dust.logκP_C_itp( log10(T) ) )
end

@inline function κP( Dust::DustState, T::Float64, ::Val{:PAH} )
    T = clamp( T, 1.0, Td_max )
    return 10.0^( Dust.logκP_PAH_itp( log10(T) ) )
end


""" Rosseland-mean opacity for a dust bin, as computed from its log-log interpolation (cm² g⁻¹). """
@inline function κR( Dust::DustState, T::Float64, ::Val{:Sil} )
    T = clamp( T, 1.0, Td_max )
    return 10.0^( Dust.logκR_Sil_itp( log10(T) ) )
end

@inline function κR( Dust::DustState, T::Float64, ::Val{:Gra} )
    T = clamp( T, 1.0, Td_max )
    return 10.0^( Dust.logκR_C_itp( log10(T) ) )
end

@inline function κR( Dust::DustState, T::Float64, ::Val{:PAH} )
    T = clamp( T, 1.0, Td_max )
    return 10.0^( Dust.logκR_PAH_itp( log10(T) ) )
end


const atol_Td::Float64 = 0.1   # Absolute tolerance (K) to deem converged 
const rtol_Td::Float64 = 1e-4  # Relative tolerance to deem converged 
const dNiter_Td        = 1:50  # Consider up to 50 iterations by default 

""" This function computes the dust temperature, given the radiation background,
    and gas conditions. Uses bisection for robustness. Dust temperature is found by 
    solving:
    
    4 κP(Td) Td⁴ =  ∑_{B} κabs_B 4πJ_B  +  κP(Trad) 4πJ_IR  
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
                DustType::Union{Val{:Sil}, Val{:Gra}, Val{:PAH}},
                    TCMB::Float64,
                    Trad::Float64,  # Radiation temperature for J_IR 
                       T::Float64,
                      nH::Float64,
                   e_Lyα::Float64 = 0.0
                     )

    κd_abs_opt, κd_abs_FUV, κd_abs_LW, 
    κd_abs_EUV1, κd_abs_EUV2, κd_abs_EUV3, 
    κd_abs_Lyα = GetDustAbsorptionOpacities( Dust, DustType )

    # Pre-compute some stuff:
    HeatOpt     = κd_abs_opt  * Four_π * J_opt
    HeatFUV     = κd_abs_FUV  * Four_π * J_FUV
    HeatLW      = κd_abs_LW   * Four_π * J_LW
    HeatEUV1    = κd_abs_EUV1 * Four_π * J_EUV1
    HeatEUV2    = κd_abs_EUV2 * Four_π * J_EUV2 
    HeatEUV3    = κd_abs_EUV3 * Four_π * J_EUV3
    HeatIR      = κP(Dust, Trad, DustType) * Four_π * J_IR
    HeatCMB     = 4 * κP(Dust, TCMB, DustType) * σSB * (TCMB^4)
    HeatLyα     = κd_abs_Lyα  * e_Lyα  * c

    # The dust temperature Td is the solution to the function f(Td) = 0,
    # where:
    f( Td ) = ( HeatOpt  +  HeatFUV  +  HeatLW  +  HeatEUV1  +  HeatEUV2  +  HeatEUV3
              + HeatLyα  +  HeatIR   +  HeatCMB
              + Λdg_over_ρd( T, Td, nH, DustType ) 

              - 4 * κP(Dust, Td, DustType) * σSB * Td^4 )

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


const σSB_over_π::Float64 = σSB / π

""" This function estimates the updated IR radiation temperature, taken as a black-body 
    weighted by the emissivities of each dust type in such a way to preserve 
    photon and energy. 
    
    This estimate is following the method presented by Grudic et al. (2021) 
    for STARFORGE. """
@inline function ComputeTrad( Td_Sil::Float64, Td_C::Float64, Td_PAH::Float64,
                              ρd_Sil::Float64, ρd_C::Float64, ρd_PAH::Float64,
                                 c̃Δt::Float64, # c̃ Δt 
                            J_IR_old::Float64, # J_IR at the 'old' time-step n 
                            Trad_old::Float64, # Radiation temperature at the 'old' time-step n 
                                Dust::DustState )

    # Exit early if there is no dust, returning the old radiation temperature: 
    if max( ρd_Sil, ρd_C, ρd_PAH ) ≤ ρd_floor
        return Trad_old
    end

    # Compute emissivities for each dust type/bin, ignoring some constants (we only 
    # need dependence on Td and ρd for each bin):

    # Silicate grains:
    Td_Sil    = max( Td_Sil, 1e-3 )  # Avoid division by zero 
    Td_Sil_2  = @fastmath Td_Sil   * Td_Sil
    Td_Sil_4  = @fastmath Td_Sil_2 * Td_Sil_2  # Td_Sil^4 
    jSil      = ρd_Sil * κP(Dust, Td_Sil, Sil) * σSB_over_π * Td_Sil_4
    ΔJ_IR_Sil = @fastmath c̃Δt * jSil

    # Graphite grains:
    Td_C      = max( Td_C, 1e-3 )  # Avoid division by zero 
    Td_C_2    = @fastmath Td_C   * Td_C
    Td_C_4    = @fastmath Td_C_2 * Td_C_2  # Td_C^4 
    jC        = ρd_C * κP(Dust, Td_C, Gra) * σSB_over_π * Td_C_4
    ΔJ_IR_C   = @fastmath c̃Δt * jC

    # PAHs:
    Td_PAH    = max( Td_PAH, 1e-3 )  # Avoid division by zero 
    Td_PAH_2  = @fastmath Td_PAH   * Td_PAH
    Td_PAH_4  = @fastmath Td_PAH_2 * Td_PAH_2  # Td_PAH^4 
    jPAH      = ρd_PAH * κP(Dust, Td_PAH, PAH) * σSB_over_π * Td_PAH_4
    ΔJ_IR_PAH = @fastmath c̃Δt * jPAH

    # Change in J_IR from dust emission:
    ΔJ_IR    = @fastmath ΔJ_IR_Sil + ΔJ_IR_C + ΔJ_IR_PAH

    # Estimated updated radiation temperature:
    Trad_old = max( Trad_old, 1e-3 )   # Avoid division by zero 
    Trad     = @fastmath ( J_IR_old  +  ΔJ_IR ) / ( 
                           J_IR_old / Trad_old  +  (ΔJ_IR_Sil / Td_Sil) + (ΔJ_IR_C / Td_C) + (ΔJ_IR_PAH / Td_PAH) )

    return Trad 
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
        θ = νi / ( 1  +  νi^(-0.5) )
        J̃ = ( 1  +  (4*τi + 3*νi)^(-0.5) )^2 * exp( - θ / τi )

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
const e²_agr_Sil::Float64 = (e^2) / agr_Sil
const e²_agr_C::Float64   = (e^2) / agr_C
const e²_agr_PAH::Float64 = (e^2) / agr_PAH


@inline function Compute_hν_pet_and_Emin( Zgr::Float64,
                                     DustType::Union{Val{:Sil}, Val{:Gra}, Val{:PAH}}
                                         )

    if DustType === Sil
        # -------- S I L I C A T E  G R A I N S -----------
        W      = W_Sil
        Ebg    = Ebg_Sil
        agr    = agr_Sil
        e²_agr = e²_agr_Sil

        # The ionization potential IP_V 
        if Zgr ≥ 0.0
            # WD, Eq. 2 (applies to both silicates and graphites):
            IP_V  = W  +  ( (Zgr + 0.5)  +  (Zgr + 2) * 0.3 * Ang / agr ) * e²_agr
        else
            # WD takes IP_V(Zgr < 0) = EAZ(Zgr+1):
            
            # WD, Eq. 5 for silicates:
            EA_Zp1 = W  -  Ebg  +  (Zgr + 0.5) * e²_agr
            IP_V   = EA_Zp1  
        end
        
        # The threshold photon energy hν_pet
        # needed for photoelectric emission (WD, Eq. 6):
        if Zgr ≥ - 1
            hν_pet = IP_V

            # Just set Emin = 0 in this case, since we must return it:
            Emin   = 0.0
        else
            Emin   = - (Zgr + 1) * e²_agr * 1 / ( 1  +  (27 * Ang / agr)^0.75 )
            hν_pet = IP_V  +  Emin 
        end

    elseif DustType === Gra 
        # -------- G R A P H I T E  G R A I N S -----------
        W      = W_C
        Ebg    = Ebg_C
        agr    = agr_C
        e²_agr = e²_agr_C

        # The ionization potential IP_V 
        if Zgr ≥ 0.0
            # WD, Eq. 2 (applies to both silicates and graphites):
            IP_V  = W  +  ( (Zgr + 0.5)  +  (Zgr + 2) * 0.3 * Ang / agr ) * e²_agr
        else
            # WD takes IP_V(Zgr < 0) = EAZ(Zgr+1):

            # WD, Eq. 4 for carbon grains:
            EA_Zp1 = W  +  (Zgr + 0.5) * e²_agr  -  e²_agr * (4 * Ang) / (agr  +  7 * Ang)
            IP_V   = EA_Zp1  
        end

        # The threshold photon energy hν_pet
        # needed for photoelectric emission (WD, Eq. 6):
        if Zgr ≥ - 1
            hν_pet = IP_V

            # Just set Emin = 0 in this case, since we must return it:
            Emin   = 0.0
        else
            Emin   = - (Zgr + 1) * e²_agr * 1 / ( 1  +  (27 * Ang / agr)^0.75 )
            hν_pet = IP_V  +  Emin 
        end

    elseif DustType === PAH 
        # -------- P A H  G R A I N S -----------
        W      = W_C
        Ebg    = Ebg_C
        agr    = agr_PAH
        e²_agr = e²_agr_PAH

        # The ionization potential IP_V 
        if Zgr ≥ 0.0
            # WD, Eq. 2 (applies to both silicates and graphites):
            IP_V  = W  +  ( (Zgr + 0.5)  +  (Zgr + 2) * 0.3 * Ang / agr ) * e²_agr
        else
            # WD takes IP_V(Zgr < 0) = EAZ(Zgr+1):

            # WD, Eq. 4 for carbon grains:
            EA_Zp1 = W  +  (Zgr + 0.5) * e²_agr  -  e²_agr * (4 * Ang) / (agr  +  7 * Ang)
            IP_V   = EA_Zp1  
        end

        # The threshold photon energy hν_pet
        # needed for photoelectric emission (WD, Eq. 6):
        if Zgr ≥ - 1
            hν_pet = IP_V

            # Just set Emin = 0 in this case, since we must return it:
            Emin   = 0.0
        else
            Emin   = - (Zgr + 1) * e²_agr * 1 / ( 1  +  (27 * Ang / agr)^0.75 )
            hν_pet = IP_V  +  Emin 
        end
    else 
        error("Unknown dust type $(DustType) in Compute_hν_pet.")
    end

    # Return result:
    return hν_pet, Emin
end


""" The photoelectric yield for a silicate or graphite/PAH dust grain, 
    as computed by Kimura (2016). PAH grains assume graphite properties, 
    but the relevant characteristic PAH grain size. """
function Ype( hν::Float64,
             Zgr::Float64,
                ::Val{:Sil}
             )

    # Choose relevant grain size, and value of e² / a_gr:
    agr    = agr_Sil
    e²_agr = e²_agr_Sil

    # The threshold photon energy hν_pet
    # needed for photoelectric emission:
    hν_pet, _ = Compute_hν_pet_and_Emin( Zgr, Sil )

    # If photon energy is below the threshold, return zero yield 
    # and exit early:
    if hν ≤ hν_pet
        return 0.0
    end

    # The factor y₂ (WD, Eq. 11):
    if Zgr ≥ 0.0
        Elow  = - (Zgr + 1) * e²_agr
        Ehigh = hν  -  hν_pet  
        y₂    = ( Ehigh * Ehigh ) * ( Ehigh  -  3 * Elow ) / ( Ehigh - Elow )^3
    else
        y₂    = 1.0
    end

    # Photoelectric yield:
    Ype = y₂ * Ya_Sil(hν, agr)

    return Ype 
end

# Larger graphite grains:
function Ype( hν::Float64,
             Zgr::Float64,
                ::Val{:Gra}
             )

    # Choose relevant grain size, and value of e² / a_gr:
    agr    = agr_C
    e²_agr = e²_agr_C

    # The threshold photon energy hν_pet
    # needed for photoelectric emission:
    hν_pet, _ = Compute_hν_pet_and_Emin( Zgr, Gra )

    # If photon energy is below the threshold, return zero yield 
    # and exit early:
    if hν ≤ hν_pet
        return 0.0
    end

    # The factor y₂ (WD, Eq. 11):
    if Zgr ≥ 0.0
        Elow  = - (Zgr + 1) * e²_agr
        Ehigh = hν  -  hν_pet  
        y₂    = ( Ehigh * Ehigh ) * ( Ehigh  -  3 * Elow ) / ( Ehigh - Elow )^3
    else
        y₂    = 1.0
    end

    # Photoelectric yield:
    Ype = y₂ * Ya_C(hν, agr)

    return Ype 
end

# PAH grains:
function Ype( hν::Float64,
             Zgr::Float64,
                ::Val{:PAH}
             )

    # Choose relevant grain size, and value of e² / a_gr:
    agr    = agr_PAH
    e²_agr = e²_agr_PAH

    # The threshold photon energy hν_pet
    # needed for photoelectric emission:
    hν_pet, _ = Compute_hν_pet_and_Emin( Zgr, PAH )

    # If photon energy is below the threshold, return zero yield 
    # and exit early:
    if hν ≤ hν_pet
        return 0.0
    end

    # The factor y₂ (WD, Eq. 11):
    if Zgr ≥ 0.0
        Elow  = - (Zgr + 1) * e²_agr
        Ehigh = hν  -  hν_pet  
        y₂    = ( Ehigh * Ehigh ) * ( Ehigh  -  3 * Elow ) / ( Ehigh - Elow )^3
    else
        y₂    = 1.0
    end

    # Photoelectric yield:
    Ype = y₂ * Ya_C(hν, agr)

    return Ype 
end


const le::Float64 = 10 * Ang  # Weingartner & Draine (2001) and Tielens (2005) give 10 Å

""" Electron sticking coefficient, following Weingartner & Draine (2001). """
@inline function se( agr::Float64 )
    0.5 * ( 1 - exp(- agr / le) )
end

# Pre-compute electron sticking coefficients:
const se_Sil::Float64          = se( agr_Sil )
const se_C::Float64            = se( agr_C )
const se_PAH::Float64          = se( agr_PAH )

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
                                 DustType::Union{Val{:Sil}, Val{:Gra}, Val{:PAH}},
                                    e_Lyα::Float64 = 0.0
                                   )

    # Convenient thermal factor (inverse of thermal velocity):
    uth_Inv  = uth_Inv_prefact / sqrt(T)

    # Clamp ne to avoid division by zero:
    ne       = max( ne, 1e-30 )

    if DustType === Sil
        # Absorption efficiencies:
        Qabs_FUV  = Qabs_Sil_FUV;  Qabs_LW   = Qabs_Sil_LW
        Qabs_EUV1 = Qabs_Sil_EUV1; Qabs_EUV2 = Qabs_Sil_EUV2
        Qabs_EUV3 = Qabs_Sil_EUV3; Qabs_Lyα  = Qabs_Sil_Lyα

        # 1 / (ne * se):
        inv_ne_s  = 1 / ( ne * se_Sil )
    elseif DustType === Gra
        # Absorption efficiencies:
        Qabs_FUV  = Qabs_C_FUV;  Qabs_LW   = Qabs_C_LW
        Qabs_EUV1 = Qabs_C_EUV1; Qabs_EUV2 = Qabs_C_EUV2
        Qabs_EUV3 = Qabs_C_EUV3; Qabs_Lyα  = Qabs_C_Lyα

        # 1 / (ne * se):
        inv_ne_s  = 1 / ( ne * se_C )
    elseif DustType === PAH
        # Absorption efficiencies:
        Qabs_FUV  = Qabs_PAH_FUV;  Qabs_LW   = Qabs_PAH_LW
        Qabs_EUV1 = Qabs_PAH_EUV1; Qabs_EUV2 = Qabs_PAH_EUV2
        Qabs_EUV3 = Qabs_PAH_EUV3; Qabs_Lyα  = Qabs_PAH_Lyα

        # 1 / (ne * se):
        inv_ne_s  = 1 / ( ne * se_PAH )
    else
        error("Uknown dust type/bin $(DustType) in photoelectric charging.") 
    end

    # Photo-electric yields (for the desired type/bin):
    Ype_FUV  = Ype( E_FUV,  Zgr, DustType )
    Ype_LW   = Ype( E_LW,   Zgr, DustType )
    Ype_EUV1 = Ype( E_EUV1, Zgr, DustType )
    Ype_EUV2 = Ype( E_EUV2, Zgr, DustType )
    Ype_EUV3 = Ype( E_EUV3, Zgr, DustType )
    Ype_Lyα  = Ype( ELyα,   Zgr, DustType )

    # Photoelectric term:
    PhotoElectricTerm = Four_π * ( (J_FUV  * inv_E_FUV)  * (Qabs_FUV  * inv_ne_s ) * uth_Inv * Ype_FUV
                                 + (J_LW   * inv_E_LW)   * (Qabs_LW   * inv_ne_s ) * uth_Inv * Ype_LW
                                 + (J_EUV1 * inv_E_EUV1) * (Qabs_EUV1 * inv_ne_s ) * uth_Inv * Ype_EUV1
                                 + (J_EUV2 * inv_E_EUV2) * (Qabs_EUV2 * inv_ne_s ) * uth_Inv * Ype_EUV2
                                 + (J_EUV3 * inv_E_EUV3) * (Qabs_EUV3 * inv_ne_s ) * uth_Inv * Ype_EUV3 )

    # Add contribution from Lyα:
    PhotoElectricTerm += (c * e_Lyα * inv_ELyα) * (Qabs_Lyα * inv_ne_s)  * uth_Inv * Ype_Lyα

    return PhotoElectricTerm
end


const vthH_over_vthe::Float64             = sqrt( me / mH )
const vthHe_over_vthe::Float64            = sqrt( me / mHe )
const Stick_ion::Float64                  = 1.0                  # Sticking coefficient of ions (1.0 by default)
const agr_Sil_kB_over_e_squared::Float64  = agr_Sil * kB / (e^2) # Convenient variable 
const agr_C_kB_over_e_squared::Float64    = agr_C   * kB / (e^2) 
const agr_PAH_kB_over_e_squared::Float64  = agr_PAH * kB / (e^2) 

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
                                DustType::Union{Val{:Sil}, Val{:Gra}, Val{:PAH}} 
                                     )

    # Floor electron density:
    ne = max( ne, 1e-30 )

    if DustType === Sil
        inv_ne_se             = 1 / ( ne * se_Sil )
        agr_kB_over_e_squared = agr_Sil_kB_over_e_squared
    elseif DustType === Gra
        inv_ne_se             = 1 / ( ne * se_C )
        agr_kB_over_e_squared = agr_C_kB_over_e_squared
    elseif DustType === PAH
        inv_ne_se             = 1 / ( ne * se_PAH )
        agr_kB_over_e_squared = agr_PAH_kB_over_e_squared
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
@inline function Zgr_pe_max( ::Val{:Sil} )
    return ( ((E_EUV3 - W_Sil) / (14.4 * eV)) * (agr_Sil / Ang)  +  1/2  -  0.3 * Ang / agr_Sil ) / ( 
               1  +  0.3 * Ang / agr_Sil )
end

@inline function Zgr_pe_max( ::Val{:Gra} )
    return ( ((E_EUV3 - W_C) / (14.4 * eV)) * (agr_C / Ang)  +  1/2  - 0.3 * Ang / agr_C ) / ( 
               1  +  0.3 * Ang / agr_C )
end

@inline function Zgr_pe_max( ::Val{:PAH} )
    return ( ((E_EUV3 - W_C) / (14.4 * eV)) * (agr_PAH / Ang)  +  1/2  -  0.3 * Ang / agr_PAH ) / ( 
               1  +  0.3 * Ang / agr_PAH )
end


""" The minimum grain charge, set by field emission. 
    See Weingartner & Draine (2001), Eq. (24). """
@inline function Zgr_min( ::Val{:Sil} )
    # For silicate grains:
    U_ait   = - ( 2.5  +  0.07 * (agr_Sil / Ang)  +  8 * (Ang / agr_Sil) )

    # Most negative allowed charge:
    Zgr_min = floor( (U_ait / 14.4) * (agr_Sil / Ang) ) + 1

    return Zgr_min 
end  

@inline function Zgr_min( ::Val{:Gra} )
    # For graphite grains:
    U_ait   = - ( 3.9  +  0.12 * (agr_C / Ang)  +  2 * (Ang / agr_C) )

    # Most negative allowed charge:
    Zgr_min = floor( (U_ait / 14.4) * (agr_C / Ang) ) + 1

    return Zgr_min 
end  


@inline function Zgr_min( ::Val{:PAH} )
    # For PAH grains (taken as small graphite grains):
    U_ait   = - ( 3.9  +  0.12 * (agr_PAH / Ang)  +  2 * (Ang / agr_PAH) )

    # Most negative allowed charge:
    Zgr_min = floor( (U_ait / 14.4) * (agr_PAH / Ang) ) + 1

    return Zgr_min 
end  

# Minimum & maximum allowed grain charges:
const Zgr_min_Sil::Float64         = Zgr_min( Sil )
const Zgr_min_C::Float64           = Zgr_min( Gra )
const Zgr_min_PAH::Float64         = Zgr_min( PAH )
const Zgr_pe_max_Sil::Float64      = Zgr_pe_max( Sil )
const Zgr_pe_max_C::Float64        = Zgr_pe_max( Gra )
const Zgr_pe_max_PAH::Float64      = Zgr_pe_max( PAH )

const max_iter_GrainCharge::Int    = 20
const dNiter_Zgr::UnitRange        = 1:max_iter_GrainCharge
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
                   DustType::Union{Val{:Sil}, Val{:Gra}, Val{:PAH}},
                      e_Lyα::Float64 = 0.0
                          )

    # Compute τₑ = agr * kB * T / ((-e)²):
    if DustType === Sil 
        τ_e  = agr_Sil_kB_over_e_squared * T   # τₑ = agr * kB * T / ((-e)²)
    elseif DustType === Gra 
        τ_e  = agr_C_kB_over_e_squared * T     # τₑ = agr * kB * T / ((-e)²)
    elseif DustType === PAH 
        τ_e  = agr_PAH_kB_over_e_squared * T   # τₑ = agr * kB * T / ((-e)²)
    end

    # Pre-factor for νₑ = Zgr * e / (-e) = - Zgr, just a minus sign:
    ν_e_prefact = - 1.0                        

    # Define the function the root of which will give Zgr 
    # (ion collisional terms + photoelectric terms - electron collisional term):
    f( Zgr ) = ( IonCollisionalChargingTerm( Zgr, T, nHII, nHeII, nHeIII, ne, DustType )
               + PhotoElectricChargingTerm( J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, ne, T, Zgr, DustType, e_Lyα )

               - J̃( τ_e,  ν_e_prefact * Zgr )  )

    # Bracket expected range:
    if DustType === Sil 
        # Minimum value is expected to be ≈ - 2.5 * agr * kB * T / e²,
        # and maximum value set by photoelectric charging, so make the following bracket:
        a       = - 3 * agr_Sil_kB_over_e_squared * T      
        b       = Zgr_pe_max_Sil
        Zgr_Min = Zgr_min_Sil
    elseif DustType === Gra 
        a       = - 3 * agr_C_kB_over_e_squared * T  
        b       = Zgr_pe_max_C
        Zgr_Min = Zgr_min_C
    elseif DustType === PAH 
        a       = - 3 * agr_PAH_kB_over_e_squared * T  
        b       = Zgr_pe_max_PAH
        Zgr_Min = Zgr_min_PAH
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
#######          P H O T O E L E C T R I C  H E A T I N G           #######
#######                                                             #######
###########################################################################

# Below is the code needed to compute the photoelectric heating rate. This 
# is done in a self-consistent manner, taking into account the different 
# grain types/bins (silicates, graphite, PAH), and the associated photoelectric 
# yields.

@inline function Ẽ_pe( Zgr::Float64,
                        hν::Float64,
                  DustType::Union{Val{:Sil}, Val{:Gra}, Val{:PAH}} )

    if Zgr ≥ 0.0
        # Compute Δ ≡ hν - hν_pet (we don't need Emin):
        hν_pet, _ = Compute_hν_pet_and_Emin( Zgr, DustType )
        Δ         = hν - hν_pet

        if Δ ≤ 0.0
            # If Δ = 0, photoelectric charging/heating gives zero, so exit early:
            return 0.0
        end

        # Compute Elow:
        if DustType === Sil 
            Elow = @fastmath - (Zgr + 1) * e²_agr_Sil
        elseif DustType === Gra 
            Elow = @fastmath - (Zgr + 1) * e²_agr_C
        elseif DustType === PAH 
            Elow = @fastmath - (Zgr + 1) * e²_agr_PAH
        else
            error("Unknown dust type $(DustType) in Ẽ_pe.")
        end

        # Some needed quantities:
        Δcubed      = @fastmath Δ * Δ * Δ
        Stuff       = Δ - Elow
        Stuff_cubed = @fastmath Stuff * Stuff * Stuff
        
        # Put it all together:
        Ẽ  = @fastmath Δcubed * ( Δ - 2 * Elow ) / ( 2 * Stuff_cubed )

        return Ẽ
    else
        # Compute Δ ≡ hν - hν_pet and Emin:
        hν_pet, Emin = Compute_hν_pet_and_Emin( Zgr, DustType )
        Δ            = hν - hν_pet

        if Δ ≤ 0.0
            # If Δ = 0, photoelectric charging/heating gives zero, so exit early:
            return 0.0
        end
        
        # For negatively charged grains we get the simple result:
        Ẽ = @fastmath Emin  +  0.5 * Δ

        return Ẽ
    end
end

# Pre-factors for the photoelectric heating rate, for all dust types/bins:
const Γ_pe_prefact_Sil::Float64 = ThreeFourths / (ρgr_Sil * agr_Sil)
const Γ_pe_prefact_C::Float64   = ThreeFourths / (ρgr_C   * agr_C)
const Γ_pe_prefact_PAH::Float64 = ThreeFourths / (ρgr_C   * agr_PAH)

""" The contribution to the photoelectric heating rate of gas
    from a given dust type/bin. Note: Only valence electrons 
    are taken into account here, i.e. we ignore photodetachment. """
function HeatingRate_PhotoElectric( J_FUV::Float64,
                                     J_LW::Float64,
                                   J_EUV1::Float64,
                                   J_EUV2::Float64,
                                   J_EUV3::Float64,
                                       ρd::Float64,
                                      Zgr::Float64,
                                 DustType::Union{Val{:Sil}, Val{:Gra}, Val{:PAH}},
                                    e_Lyα::Float64 )

    if ρd < ρd_IGNORE
        # Ignore photoelectric heating if the dust density 
        # is near the floor:
        return 0.0
    end

    # Get the needed dust type-specific quantities:
    if DustType === Sil 
        # Pre-factor:
        Γ_pe_prefact = Γ_pe_prefact_Sil

        # Absorption efficiences:
        Qabs_FUV     = Qabs_Sil_FUV
        Qabs_LW      = Qabs_Sil_LW
        Qabs_EUV1    = Qabs_Sil_EUV1
        Qabs_EUV2    = Qabs_Sil_EUV2
        Qabs_EUV3    = Qabs_Sil_EUV3
        Qabs_Lyα     = Qabs_Sil_Lyα

        # Raw photoelectric yield Ya for each band,
        # ignoring electron escape:
        Ya_FUV       = Ya_Sil( E_FUV,  agr_Sil )
        Ya_LW        = Ya_Sil( E_LW,   agr_Sil )
        Ya_EUV1      = Ya_Sil( E_EUV1, agr_Sil )
        Ya_EUV2      = Ya_Sil( E_EUV2, agr_Sil )
        Ya_EUV3      = Ya_Sil( E_EUV3, agr_Sil )
        Ya_Lyα       = Ya_Sil( ELyα,   agr_Sil )

    elseif DustType === Gra 
        # Pre-factor:
        Γ_pe_prefact = Γ_pe_prefact_C

        # Absorption efficiences:
        Qabs_FUV     = Qabs_C_FUV
        Qabs_LW      = Qabs_C_LW
        Qabs_EUV1    = Qabs_C_EUV1
        Qabs_EUV2    = Qabs_C_EUV2
        Qabs_EUV3    = Qabs_C_EUV3
        Qabs_Lyα     = Qabs_C_Lyα

        # Raw photoelectric yield Ya for each band,
        # ignoring electron escape:
        Ya_FUV       = Ya_C( E_FUV,  agr_C )
        Ya_LW        = Ya_C( E_LW,   agr_C )
        Ya_EUV1      = Ya_C( E_EUV1, agr_C )
        Ya_EUV2      = Ya_C( E_EUV2, agr_C )
        Ya_EUV3      = Ya_C( E_EUV3, agr_C )
        Ya_Lyα       = Ya_C( ELyα,   agr_C )

    elseif DustType === PAH
        # Pre-factor:
        Γ_pe_prefact = Γ_pe_prefact_PAH

        # Absorption efficiences:
        Qabs_FUV     = Qabs_PAH_FUV
        Qabs_LW      = Qabs_PAH_LW
        Qabs_EUV1    = Qabs_PAH_EUV1
        Qabs_EUV2    = Qabs_PAH_EUV2
        Qabs_EUV3    = Qabs_PAH_EUV3
        Qabs_Lyα     = Qabs_PAH_Lyα

        # Raw photoelectric yield Ya for each band,
        # ignoring electron escape:
        Ya_FUV       = Ya_C( E_FUV,  agr_PAH )
        Ya_LW        = Ya_C( E_LW,   agr_PAH )
        Ya_EUV1      = Ya_C( E_EUV1, agr_PAH )
        Ya_EUV2      = Ya_C( E_EUV2, agr_PAH )
        Ya_EUV3      = Ya_C( E_EUV3, agr_PAH )
        Ya_Lyα       = Ya_C( ELyα,   agr_PAH )
    else
        error("Uknown dust type $(DustType) in HeatingRate_PhotoElectric.")
    end

    # Compute Ẽ for all bands:
    Ẽ_FUV  = Ẽ_pe( Zgr, E_FUV,  DustType )
    Ẽ_LW   = Ẽ_pe( Zgr, E_LW,   DustType )
    Ẽ_EUV1 = Ẽ_pe( Zgr, E_EUV1, DustType )
    Ẽ_EUV2 = Ẽ_pe( Zgr, E_EUV2, DustType )
    Ẽ_EUV3 = Ẽ_pe( Zgr, E_EUV3, DustType )
    Ẽ_Lyα  = Ẽ_pe( Zgr, ELyα,   DustType )

    # Get the photoelectric heating rate from the given dust type/bin:
    Γ_pe   = @fastmath Γ_pe_prefact * ρd * ( (Four_π * J_FUV  * inv_E_FUV)  * Qabs_FUV  * Ya_FUV  * Ẽ_FUV
                                           + (Four_π * J_LW   * inv_E_LW)   * Qabs_LW   * Ya_LW   * Ẽ_LW
                                           + (Four_π * J_EUV1 * inv_E_EUV1) * Qabs_EUV1 * Ya_EUV1 * Ẽ_EUV1
                                           + (Four_π * J_EUV2 * inv_E_EUV2) * Qabs_EUV2 * Ya_EUV2 * Ẽ_EUV2
                                           + (Four_π * J_EUV3 * inv_E_EUV3) * Qabs_EUV3 * Ya_EUV3 * Ẽ_EUV3
                                           + (c * e_Lyα * inv_ELyα)         * Qabs_Lyα  * Ya_Lyα  * Ẽ_Lyα )

    return Γ_pe 
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
const ϕ_Sil_prefact::Float64     = (e^2) / ( agr_Sil * kB )
const ϕ_C_prefact::Float64       = (e^2) / ( agr_C * kB )
const ϕ_PAH_prefact::Float64     = (e^2) / ( agr_PAH * kB )
const Zgr_th_Kdrag::Float64      = 0.1                   # Grain charge is ignored if |Zgr| < Zgr_th_Kdrag
const T_floor_Kdrag::Float64     = 0.01
const ne_floor_K_drag::Float64   = 1e-30 

# Convenient constants for evaluating |⟨ngr Fdrag⟩| / urel:
const Fdrag_Sil_prefact::Float64 = (3/2) * kB / (ρgr_Sil * agr_Sil)
const Fdrag_C_prefact::Float64   = (3/2) * kB / (ρgr_C * agr_C)
const Fdrag_PAH_prefact::Float64 = (3/2) * kB / (ρgr_C * agr_PAH)    # PAH shares same bulk density as graphite 
const Λ_Sil_prefact::Float64     = (3 / (2 * agr_Sil * e)) * sqrt(kB / π)
const Λ_C_prefact::Float64       = (3 / (2 * agr_C * e))   * sqrt(kB / π)
const Λ_PAH_prefact::Float64     = (3 / (2 * agr_PAH * e)) * sqrt(kB / π)

""" This function computes the drag force coefficient K ≡ |⟨ngr Fdrag⟩| / | ud - u |,
    considering drag from H I, H₂, H II, He I, He II, He III, and given dust type/bin.
    
    The drag coeffient enters the evolution of the dust momenta:
    
    ∂(ρd_bin ud_bin)/∂t|_drag = K * (u - ud_bin).

    From Draine & Salpeter (1979) one has:

    K_bin = 2 × (3 ρd_bin kB T) / (4 agr_bin ρgr_bin)
          × Σ_i ni [G'_0(si) + Zi² ϕs² ln|Λs/Zi| G'_2(si)]
    
    """
function ComputeDragCoefficientK( Zgr_bin::Float64, # Grain charge of this type/bin
                                   ρd_bin::Float64, # Dust density in this type/bin
                                        T::Float64, # Gas temperature 
                                 DustType::Union{Val{:Sil}, Val{:Gra}, Val{:PAH}},
                                      nHI::Float64, # H I number density 
                                      nH2::Float64, # H₂ number density 
                                     nHII::Float64, # H II number density 
                                     nHeI::Float64, # He I number density 
                                    nHeII::Float64, # He II number density 
                                   nHeIII::Float64, # He III number density 
                                       ne::Float64, # e⁻ number density 
                                     ud_r::Float64, # Dust velocity in r-direction for this bin/type 
                                      u_r::Float64, # Gas velocity in r-direction 
                                     ud_Z::Float64, # Dust velocity in Z-direction for this bin/type 
                                      u_Z::Float64  # Gas velocity in Z-direction 
                                      )

    T          = max( T, T_floor_Kdrag )
    sqrt_T     = @fastmath sqrt(T)
    inv_T      = @fastmath 1.0 / T
    urel       = @fastmath sqrt( (ud_r - u_r) * (ud_r - u_r)  +  (ud_Z - u_Z) * (ud_Z - u_Z) )
    ne_floored = max( ne, ne_floor_K_drag )   # To avoid division by zero

    # Evaluate vth ≡ √(2kT/mᵢ) for all major species:
    vth_HI     = @fastmath vth_HI_prefact  * sqrt_T
    vth_H2     = @fastmath vth_H2_prefact  * sqrt_T 
    vth_HeI    = @fastmath vth_HeI_prefact * sqrt_T
    vth_e      = @fastmath vth_e_prefact   * sqrt_T

    # Evaluate sᵢ ≡ |ud - u| / vth for all major species
    # (some are the same, e.g. H II is same as H I, so no need to compute):
    s_HI       = @fastmath urel / vth_HI 
    s_H2       = @fastmath urel / vth_H2
    s_HeI      = @fastmath urel / vth_HeI 
    s_e        = @fastmath urel / vth_e

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

    # Evaluate ϕ ≡ Z_gr * e² / a_gr * kB * T,
    # Λ ≡ [3 / (2 * a_gr * e * |ϕ|)] * sqrt(kB * T / π * ne),
    # and the drag pre-factor:
    if DustType === Sil 
        ϕ      = @fastmath ϕ_Sil_prefact * Zgr_bin * inv_T
        ϕ_sq   = @fastmath ϕ * ϕ
        Λ      = @fastmath Λ_Sil_prefact * ( 1 / abs(ϕ) ) * sqrt( T / ne_floored )

        # Drag pre-factor:
        Fdrag_prefact = Fdrag_Sil_prefact
    elseif DustType === Gra 
        ϕ      = @fastmath ϕ_C_prefact * Zgr_bin * inv_T
        ϕ_sq   = @fastmath ϕ * ϕ
        Λ      = @fastmath Λ_C_prefact * ( 1 / abs(ϕ) ) * sqrt( T / ne_floored )

        # Drag pre-factor:
        Fdrag_prefact = Fdrag_C_prefact
    elseif DustType === PAH 
        ϕ      = @fastmath ϕ_PAH_prefact * Zgr_bin * inv_T
        ϕ_sq   = @fastmath ϕ * ϕ
        Λ      = @fastmath Λ_PAH_prefact * ( 1 / abs(ϕ) ) * sqrt( T / ne_floored )

        # Drag pre-factor:
        Fdrag_prefact = Fdrag_PAH_prefact
    end

    # Evaluate ϕ² log(Λ) safely:
    if abs(Zgr_bin) > Zgr_th_Kdrag
        absΛ      = max( abs(Λ), 1.0 )      # Ensure that ln(Λ) > 0
        ϕ_sq_logΛ = @fastmath ϕ_sq * log( absΛ )
    else 
        # In case Zgr_bin = 0, Λ becomes numerically problematic, 
        # but ϕ → 0, so we just get 0 regardless:
        ϕ_sq_logΛ = 0.0
    end

    # Drag coefficient pre-factor:
    K_drag_prefact  = @fastmath Fdrag_prefact * ρd_bin * T
    
    # H I contribution (Z = 0, so only need G₀ / urel):
    HI_contr    = @fastmath nHI * G̃0_HI

    # H II contribution (Z = 1, Z² = 1):
    HII_contr   = @fastmath nHII * ( G̃0_HII  +  ϕ_sq_logΛ * G̃2_HII )

    # H₂ contribution (Z = 0):
    H2_contr    = @fastmath nH2 * G̃0_H2 

    # He I contribution (Z = 0):
    HeI_contr   = @fastmath nHeI * G̃0_HeI 

    # He II contribution (Z = 1, Z² = 1):
    HeII_contr  = @fastmath nHeII * ( G̃0_HeII  +  ϕ_sq_logΛ * G̃2_HeII )

    # He III contribution (Z = 2, Z² = 4):
    if abs(Zgr_bin) > Zgr_th_Kdrag
        # Factor 1 / Z = 0.5 is used:
        absΛ_05         = max( abs(Λ * 0.5), 1.0 )      # Ensure that ln(Λ * 0.5) > 0
        ϕ_sq_logΛ_HeIII = @fastmath ϕ_sq * log( absΛ_05 )
    else
        ϕ_sq_logΛ_HeIII = 0.0
    end

    HeIII_contr     = @fastmath nHeIII * ( G̃0_HeIII  +  4 * ϕ_sq_logΛ_HeIII * G̃2_HeIII )

    # e⁻ contribution (Z = - 1, Z² = 1):
    e_contr         = @fastmath ne * ( G̃0_e  +  ϕ_sq_logΛ * G̃2_e )

    # Put it all together to get the drag coefficient K:
    K_drag          = @fastmath K_drag_prefact * ( HI_contr   + HII_contr 
                                                 + H2_contr   + HeI_contr 
                                                 + HeII_contr + HeIII_contr 
                                                 + e_contr )

    return K_drag
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
                                ρdplusZ::Float64,  # Density in this dust bin + currently available gas-phase limiting species
                                   tgr0::Float64, 
                                  tdest::Float64,
                                  Δtsub::Float64
                                 )

    # Convenient variables:
    inv_tdest = 1 / tdest
    inv_Δtsub = 1 / Δtsub

    # If there is no dust growth (tgr0 → ∞), or negligible growht compared 
    # to destruction, only treat dust destruction 
    # to avoid dealing with infinities & complex expressions:
    if !isfinite(tgr0) || tgr0 > 1e3 * tdest
        # New dust density:
        ρd_new = ρd_old / ( 1  +  Δtsub * inv_tdest )

        return ρd_new
    end 

    # Define A and B that enter the BDF1 update:
    A         = ρdplusZ * ( tgr0 * ( inv_Δtsub  +  inv_tdest )  -  1.0 )
    B         = ρd_old * ρdplusZ * tgr0 * inv_Δtsub    

    if A ≥ 0.0
        # In this case, simplify the quadratic solution to avoid 
        # dangerous cancellations:
        ρd_new = 2 * B / ( A + sqrt( A * A  +  4 * B ) )
    elseif A < 0.0
        # Otherwise, use original version of formula:
        A_over_2 = 0.5 * A  
        ρd_new   = - A_over_2 + sqrt( A_over_2 * A_over_2  +  B ) 
    end

    # Numerical safety: Clamp ρd to be in physical range:
    ρd_new = clamp( ρd_new, 0.0, ρdplusZ )

    return ρd_new
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


# Pre-factors for vrel_X = sqrt( kB T / π m_X ):
const vrel_Si_prefact::Float64  = sqrt( kB / (Two_π * mSi))
const vrel_C_prefact::Float64   = sqrt( kB / (Two_π * mC))

# Fraction of grain mass made up by limiting species:
const f_limit_Sil::Float64      = 0.166  # Fraction of silicate grain mass made up by Si (Hirashita 2012)
const f_limit_C::Float64        = 1.0    # Fraction of graphite grain mass made up by C  (Hirashita 2012)
const f_limit_PAH::Float64      = 1.0    # Fraction of PAH grain mass made up by C (approximate)

const tgr0_Sil_prefact::Float64 = agr_Sil * ρgr_Sil * f_limit_Sil / ( 3 * vrel_Si_prefact )
const tgr0_C_prefact::Float64   = agr_C   * ρgr_C   * f_limit_C   / ( 3 * vrel_C_prefact )
const tgr0_PAH_prefact::Float64 = agr_PAH * ρgr_C   * f_limit_PAH / ( 3 * vrel_C_prefact )

@inline function Compute_tgr0( T::Float64,
                              Td::Float64,
                             xCI::Float64,
                            xCII::Float64,
                             Zgr::Float64,
                         ρdplusZ::Float64,  # Total density in dust + gas of relevant species (e.g. for graphite it would be ρd_C + ρd_PAH + ρC)
                        DustType::Union{Val{:Sil}, Val{:Gra}, Val{:PAH}} )

    if T > 1e5 
        # At very high temperatures, ignore dust growth:
        return Inf 
    end

    # Effective sticking coefficient (modelled mainly for carbon species), 
    # taking into account Coulomb modification of the cross-section
    # for ions (and ignoring CO, since it doesn't chemisorb):
    if DustType === Gra 
        # In this case, the relevant ion is C II (and so q_i = 1):
        τ_i = agr_C_kB_over_e_squared * T 
        ν_i = Zgr 
    elseif DustType === PAH 
        # In this case, the relevant ion is C II (and so q_i = 1):
        τ_i = agr_PAH_kB_over_e_squared * T 
        ν_i = Zgr
    elseif DustType === Sil 
        # In this case, the relevant ion is Sil II (and so q_i = 1):
        τ_i = agr_Sil_kB_over_e_squared * T 
        ν_i = Zgr   
        # NOTE: Since we don't follow Si chemistry, we assume that xSII = xCII.
    end
    J̃_ion   = J̃( τ_i, ν_i )
    Stick_C = Stick_CI( Td ) * ( xCI  +  xCII * J̃_ion )

    # Sticking coefficient for Si on silicate grains is not really known, 
    # but silicate grain accretion will generally be slower than the growth of
    # carbon grains, so for simplicity assume the same sticking coefficient.

    if DustType === Sil
        tgr0 = tgr0_Sil_prefact / ( sqrt(T) * Stick_C * ρdplusZ )
    elseif DustType === Gra 
        tgr0 = tgr0_C_prefact / ( sqrt(T) * Stick_C * ρdplusZ )
    elseif DustType === PAH
        tgr0 = tgr0_PAH_prefact / ( sqrt(T) * Stick_C * ρdplusZ )
    end

    return tgr0 
end


const agr_Sil_over_01::Float64 = agr_Sil / (0.1 * μm)
const agr_C_over_01::Float64   = agr_C   / (0.1 * μm)
const agr_PAH_over_01::Float64 = agr_PAH / (0.1 * μm)
const ΔTSil_subl::Float64      = 68100.0 
const ΔTC_subl::Float64        = 81200.0
const inv_Td_Sil_subl::Float64 = 1 / 1800.0 
const inv_Td_C_subl::Float64   = 1 / 3000.0
const tsubl_Sil_0::Float64     = 6.36e3 / 3   # Pre-factor for silicates, with factor 1/3 converting from size to mass destruction time-scale 
const tsubl_C_0::Float64       = 1.36 / 3     # Pre-factor for graphite/carbonaceous grains, with factor 1/3 converting from size to mass destruction time-scale 

""" This fuction computes the sublimation time-scale (sec) for dust. """
@inline function Compute_tsubl( Td::Float64, DustType::Union{Val{:Sil}, Val{:Gra}, Val{:PAH}} ) 

    Td = clamp( Td, Td_floor, Td_max )

    if DustType === Sil 
        # Sublimation time-scale for silicate dust:
        tsubl = tsubl_Sil_0 * agr_Sil_over_01 * exp( ΔTSil_subl * (1 / Td  -  inv_Td_Sil_subl) )
    elseif DustType === Gra 
        # Sublimation time-scale for graphite dust:
        tsubl = tsubl_C_0 * agr_C_over_01 * exp( ΔTC_subl * (1 / Td  -  inv_Td_C_subl) )
    elseif DustType === PAH 
        # Sublimation time-scale for PAH dust:
        tsubl = tsubl_C_0 * agr_PAH_over_01 * exp( ΔTC_subl * (1 / Td  -  inv_Td_C_subl) )
    end

    # Return time-scale, and limit it to avoid way too short time-steps in the 
    # sub-cycling:
    return max( tsubl, 0.1 * yr ) 
end


const ρ_sputter::Float64            = 1e-27 
const T_sputter::Float64            = 2e6 
const ω_sputter::Float64            = 2.5
const agr_sputter_norm_Sil::Float64 = agr_Sil / (0.1 * μm)
const agr_sputter_norm_C::Float64   = agr_C   / (0.1 * μm)
const agr_sputter_norm_PAH::Float64 = agr_PAH / (0.1 * μm)
const tsputter_th_prefact::Float64  = OneThird * 0.17 * Gyr  # Factor 1/3 to convert to *mass* sputtering time

""" Destruction time-scale (s) from thermal sputtering. """
@inline function Compute_tsputter_thermal( ρ::Float64, 
                                           T::Float64,
                                    DustType::Union{Val{:Sil}, Val{:Gra}, Val{:PAH}} )

    if T ≤ 1e5
        # Ignore sputtering below 10⁵ K, where the fit breaks down anyway.
        return Inf64
    else
        if DustType === Sil
            anorm = agr_sputter_norm_Sil
        elseif DustType === Gra 
            anorm = agr_sputter_norm_C
        elseif DustType === PAH 
            anorm = agr_sputter_norm_PAH
        end

        # The sputtering time-scale agr / 3 |d(agr)/dt| for the grain radius:
        t_sputter = tsputter_th_prefact * anorm * (ρ_sputter/ρ) * ( (T_sputter/T)^ω_sputter + 1 )

        return t_sputter
    end
end


# Fitting coefficients from Hu et al. (2019):
const asput0_sil::Float64 = - 32.4;  const asput1_sil::Float64 = 28.6
const asput2_sil::Float64 = - 11.6;  const asput3_sil::Float64 = 2.23
const asput4_sil::Float64 = - 0.208; const asput5_sil::Float64 = 0.00736

const asput0_C::Float64 = - 48.7;  const asput1_C::Float64 = 55.7
const asput2_C::Float64 = - 28.8;  const asput3_C::Float64 = 7.44
const asput4_C::Float64 = - 0.965; const asput5_C::Float64 = 0.005

const t_sputter_nonthermal_0_Sil::Float64 = 0.33 * Myr * ( agr_Sil / μm )
const t_sputter_nonthermal_0_C::Float64   = 0.33 * Myr * ( agr_C   / μm )
const t_sputter_nonthermal_0_PAH::Float64 = 0.33 * Myr * ( agr_PAH / μm )

""" The non-thermal sputtering time for grains moving 
    at velocity urel (cm s⁻¹) relative to gas of density nH (cm⁻³).
    Uses fits from Hu et al. (2019). """
function Compute_tsputter_nonthermal( nH::Float64,
                                    urel::Float64,
                                DustType::Union{Val{:Sil}, Val{:Gra}, Val{:PAH}}
                                     )

    urel = min( abs( urel ), 1e5 * kms )

    # Non-thermal sputtering is only important at very high velocities:
    if urel ≥ 30.0 * kms

        if DustType === Sil
            a0 = asput0_sil; a1 = asput1_sil
            a2 = asput2_sil; a3 = asput3_sil
            a4 = asput4_sil; a5 = asput5_sil

            t_sputter0 = t_sputter_nonthermal_0_Sil
        elseif DustType === Gra 
            a0 = asput0_C;   a1 = asput1_C
            a2 = asput2_C;   a3 = asput3_C
            a4 = asput4_C;   a5 = asput5_C

            t_sputter0 = t_sputter_nonthermal_0_C
        elseif DustType === PAH
            a0 = asput0_C;   a1 = asput1_C
            a2 = asput2_C;   a3 = asput3_C
            a4 = asput4_C;   a5 = asput5_C

            t_sputter0 = t_sputter_nonthermal_0_PAH
        end

        x    = log10(urel / kms); x2 = x*x; x3 = x2*x; x4 = x3*x; x5 = x4*x 
        y    = a0 + a1*x + a2*x2 + a3*x3 + a4*x4 + a5*x5 
        Ytot = max( 10^y, 0.0 )  

        # The non-thermal sputtering time:
        t_sputter = t_sputter0 * ( 1 / nH ) * ( 1 / Ytot )

        return t_sputter
    else
        # Ignore non-thermal sputtering at very low relative velocities:
        t_sputter = Inf64

        return t_sputter
    end
end


const Y_chemsp_C_max::Float64        = 4.9e-3
const Tmax_chemsp::Float64           = 580.0
const σ_chemsp::Float64              = 125.0
const p_chemsp::Float64              = 1.65
const inv_σ_chemsp_topower::Float64  = (1 / σ_chemsp)^p_chemsp 

""" Chemical sputtering yield per H atom collision for graphite grains, 
    as given by Lenzuni et al. (1995). See their Eq. (22). """
@inline function Y_chemsp_C( Td::Float64 )

    δT       = abs( Td - Tmax_chemsp )
    Y_chemsp = @fastmath Y_chemsp_C_max * exp( - (δT^p_chemsp) * inv_σ_chemsp_topower )

    return Y_chemsp 
end


const vrel_H_prefact::Float64  = sqrt( kB / (Two_π * mH) )  # Pre-factor for vrel_H = sqrt( kB T / π mH )
const t_chemsputter_0::Float64 = agr_C * ρgr_C / (3 * vrel_H_prefact * mC)

""" The chemical sputtering time-scale for graphite grains.
    Only chemical sputtering by collisions with H atoms is considered,
    which produces hydrocarbons that leave the grain (e.g. CH₄).

    It is assumed that H II is converted to H I on the grain, and therefore
    also contributes to the chemisputtering (Draine 1979). """
function Compute_tchemsputter( T::Float64,
                              Td::Float64, 
                             nHI::Float64,
                            nHII::Float64,
                             Zgr::Float64, # Grain charge for graphite grains 
                        DustType::Union{Val{:Sil}, Val{:Gra}, Val{:PAH}} )

    if DustType === Sil || DustType === PAH
        # No chemical sputtering of silicate grains, and also ignore for PAH grains,
        # since yield is not known:
        return Inf 
    end

    # Chemisputtering yield for graphite grains:
    Y_chemsp = Y_chemsp_C( Td )

    # Next, compute Coulomb enhancement/suppression of collision rates with H I and H II, 
    # taking grain charge into account.

    # H II collisions with grains: 
    τ_HII    = agr_C_kB_over_e_squared * T  # agr_C * kB * T / ((+e)²)
    ν_HII    = Zgr                          # Zgr * e / (+e) = Zgr 
    J̃_HII    = J̃(τ_HII, ν_HII)

    # H I collisions with grains: 
    # No Coulomb enhancement/suppression for H I collisions, i.e.  J̃_HI = 1

    # Put it all together to get the graphite grain chemical sputtering time-scale:
    t_chemsp = @fastmath t_chemsputter_0 / ( sqrt(T) * Y_chemsp * (nHI  +  nHII * J̃_HII) )

    return t_chemsp
end


const nmax_PAHdiss::Float64         = NC_PAH / 5                  # Number of IR photons emitted by the PAH following absorption of a photon
const Δϵ_IR_PAHdiss::Float64        = 0.16 * eV                   # Average energy per IR photon emitted
const k_IR_PAH::Float64             = 100.0                       # k_IR = 100 photons sec⁻¹ is the IR photon emission rate of the PAH
const Rgasconst::Float64            = 1.987204                    # The universal gas constant (in cal K⁻¹ mol⁻¹)
const ΔS_C2H2::Float64              = 10.0                        # Change in entropy due to C₂H₂ dissociation (cal K⁻¹ mol⁻¹)
const ΔS_H::Float64                 = 11.8                        # Change in entropy due to H dissociation (cal K⁻¹ mol⁻¹)
const ΔS_H2::Float64                = - 12.69                     # Change in entropy due to H₂ dissociation (cal K⁻¹ mol⁻¹)
const k0_diss_C2H2_prefact::Float64 = (kB / h) * exp( 1 + ΔS_C2H2 / Rgasconst )
const k0_diss_H_prefact::Float64    = (kB / h) * exp( 1 + ΔS_H    / Rgasconst )
const k0_diss_H2_prefact::Float64   = (kB / h) * exp( 1 + ΔS_H2   / Rgasconst )
const Ebind_PAH_C2H2_eV::Float64    = 4.6                         # Binding energy of C₂H₂ (acetylene), in eV
const Ebind_PAH_H_eV::Float64       = 4.3                         # Binding energy of H, in eV
const Ebind_PAH_H2_eV::Float64      = 3.52                        # Binding energy of H₂ in eV 
const inv_NC_PAH::Float64           = 1 / NC_PAH                  # 1 / number of carbon atoms per PAH 

""" The probability that a PAH molecule will dissociate a C₂H₂ (acetylene) molecule 
    following absorption of a photon with energy hν (in erg), *ignoring photoelectric
    emission.*
    
    Photoelectric emission will add a multiplicative factor (1 - Ya_C), i.e. one minus 
    the photoelectric emission probability. However, Ya_C depends on the grain charge 
    (via the photoelectric emission threshold), so it has been separated from this 
    function. """
@inline function Compute_Pdiss_PAH_NoPE( hν::Float64 )

    # Compute PAH internal energy after 0 IR photon emissions (E_0, in eV), and 
    # after nmax_PAHdiss IR photon emissions (E_nmax, in eV). For the former, it is 
    # simply equal to the photon energy. 
    E_0_eV    = hν / eV
    E_nmax_eV = (hν - nmax_PAHdiss * Δϵ_IR_PAHdiss ) / eV

    if E_nmax_eV ≤ 0.0 
        # In this case, cooling is so fast that Pdiss = 0:
        return 0.0
    end

    # Compute effective PAH temperature (in K) for a given excitation energy:
    Teff_0     = 2e3 * ( (E_0_eV    * inv_NC_PAH)^0.4 ) * ( 1 - 0.2 * (Ebind_PAH_C2H2_eV / E_0_eV) )
    Teff_nmax  = 2e3 * ( (E_nmax_eV * inv_NC_PAH)^0.4 ) * ( 1 - 0.2 * (Ebind_PAH_C2H2_eV / E_nmax_eV) )

    if Teff_0 ≤ 0.0 || Teff_nmax ≤ 0.0
        # In this case, the photon has too little energy to cause dissociation.
        return 0.0
    end

    # Compute kdiss_C2H2, evaluated at T = Tav = √(T0 × Tmax):
    Tav        = sqrt( Teff_0 * Teff_nmax )   # Geometric average Teff 
    k0_C2H2    = k0_diss_C2H2_prefact * Tav 
    kdiss_C2H2 = k0_C2H2 * exp( - (Ebind_PAH_C2H2_eV * eV) / (kB * Tav) )

    # Compute kdiss_H, evaluated at T = Tav:
    k0_H       = k0_diss_H_prefact * Tav 
    kdiss_H    = k0_H * exp( - (Ebind_PAH_H_eV * eV) / (kB * Tav) )

    # Compute kdiss_H2, evaluated at T = Tav:
    k0_H2      = k0_diss_H2_prefact * Tav 
    kdiss_H2   = k0_H2 * exp( - (Ebind_PAH_H2_eV * eV) / (kB * Tav) )

    # Average C₂H₂ dissociation probability for a single photon event:
    pdiss_av   = kdiss_C2H2 / ( k_IR_PAH  +  kdiss_C2H2  +  kdiss_H  +  kdiss_H2 )

    # Dissociation probability after nmax IR photons, ignoring the ionization of the PAH 
    # (i.e. photoelectric emission):
    if pdiss_av > 1e-5 
        Pdiss_PAH = 1.0 - (1.0 - pdiss_av)^(nmax_PAHdiss + 1)
    else
        # For tiny pdiss_PAH, use a Taylor expansion:
        Pdiss_PAH = (nmax_PAHdiss + 1) * pdiss_av
    end

    @assert Pdiss_PAH ≥ 0.0 "PAH photodissociation probability must be ≥ 0.0."

    return Pdiss_PAH
end

# Pre-compute Pdiss_PAH for all relevant energy bands (Pdiss is tiny for optical band, 
# so ignore that):
const Pdiss_PAH_FUV_NoPE::Float64  = Compute_Pdiss_PAH_NoPE( E_FUV )
const Pdiss_PAH_LW_NoPE::Float64   = Compute_Pdiss_PAH_NoPE( E_LW )
const Pdiss_PAH_EUV1_NoPE::Float64 = Compute_Pdiss_PAH_NoPE( E_EUV1 )
const Pdiss_PAH_EUV2_NoPE::Float64 = Compute_Pdiss_PAH_NoPE( E_EUV2 )
const Pdiss_PAH_EUV3_NoPE::Float64 = Compute_Pdiss_PAH_NoPE( E_EUV3 )
const Pdiss_PAH_Lyα_NoPE::Float64  = Compute_Pdiss_PAH_NoPE( ELyα )

""" Save the pre-computed PAH photodissociation probabilities per absorbed photon
    for the radiation bands used in LYDION. This is mainly for documentation /
    reproducibility, and is typically called once at startup. """
function WritePAHDissociationProbabilities( OutputDir::String = @__DIR__ )

    mkpath(OutputDir)
    FileName = joinpath(OutputDir, "PAH_photodissociation_probabilities.txt")

    open(FileName, "w") do io
        println(io, "# PAH C₂H₂ photodissociation probabilities per absorbed photon, ignoring photoelectric emission (1 - Ya_C).")
        println(io, "# Computed in DustPhysics.jl at startup.")
        println(io, "#")
        println(io, "# Columns:")
        println(io, "# Band   Photon energy (eV)   Dissociation probability")
        @printf(io, "%-6s  %.8e  %.8e\n", "FUV",  E_FUV  / eV, Pdiss_PAH_FUV_NoPE)
        @printf(io, "%-6s  %.8e  %.8e\n", "LW",   E_LW   / eV, Pdiss_PAH_LW_NoPE)
        @printf(io, "%-6s  %.8e  %.8e\n", "EUV1", E_EUV1 / eV, Pdiss_PAH_EUV1_NoPE)
        @printf(io, "%-6s  %.8e  %.8e\n", "EUV2", E_EUV2 / eV, Pdiss_PAH_EUV2_NoPE)
        @printf(io, "%-6s  %.8e  %.8e\n", "EUV3", E_EUV3 / eV, Pdiss_PAH_EUV3_NoPE)
        @printf(io, "%-6s  %.8e  %.8e\n", "Lyα",  ELyα   / eV, Pdiss_PAH_Lyα_NoPE)
    end

    return nothing
end

const mC2H2::Float64                          = 2 * mC  +  2 * mH  # Mass of a C₂H₂ molecule (acetylene)
const tphoto_PAH_prefact::Float64             = 4 * agr_PAH * ρgr_C / (3 * mC2H2)
const Integral_prefact_FUV_PAH_diss::Float64  = Four_π * inv_E_FUV  * Qabs_PAH_FUV  * Pdiss_PAH_FUV_NoPE
const Integral_prefact_LW_PAH_diss::Float64   = Four_π * inv_E_LW   * Qabs_PAH_LW   * Pdiss_PAH_LW_NoPE
const Integral_prefact_EUV1_PAH_diss::Float64 = Four_π * inv_E_EUV1 * Qabs_PAH_EUV1 * Pdiss_PAH_EUV1_NoPE
const Integral_prefact_EUV2_PAH_diss::Float64 = Four_π * inv_E_EUV2 * Qabs_PAH_EUV2 * Pdiss_PAH_EUV2_NoPE
const Integral_prefact_EUV3_PAH_diss::Float64 = Four_π * inv_E_EUV3 * Qabs_PAH_EUV3 * Pdiss_PAH_EUV3_NoPE
const Integral_prefact_Lyα_PAH_diss::Float64  = c      * inv_ELyα   * Qabs_PAH_Lyα  * Pdiss_PAH_Lyα_NoPE

""" This function computes the photodissociation time-scale
    for destroying a grain, only relevant for PAH's. """
@inline function Compute_tphoto_PAH( J_FUV::Float64,
                                      J_LW::Float64,
                                    J_EUV1::Float64,
                                    J_EUV2::Float64,
                                    J_EUV3::Float64,
                                     e_Lyα::Float64,
                                   Zgr_PAH::Float64,
                                  DustType::Union{Val{:Sil}, Val{:Gra}, Val{:PAH}},
                                     )

    if DustType === Sil || DustType === Gra
        # No photodissociation of large silicate & graphite grains:
        return Inf 
    end

    # Compute 1 - (probability of photoelectric emission), for each band:
    hν_pet, _ = Compute_hν_pet_and_Emin( Zgr_PAH, PAH )  # Threshold for photoelectric emission 

    FUV_NoPE  = max( 1.0 - (E_FUV  > hν_pet) * Ya_C( E_FUV,  agr_PAH ), 0.0 )
    LW_NoPE   = max( 1.0 - (E_LW   > hν_pet) * Ya_C( E_LW,   agr_PAH ), 0.0 )
    EUV1_NoPE = max( 1.0 - (E_EUV1 > hν_pet) * Ya_C( E_EUV1, agr_PAH ), 0.0 )
    EUV2_NoPE = max( 1.0 - (E_EUV2 > hν_pet) * Ya_C( E_EUV2, agr_PAH ), 0.0 )
    EUV3_NoPE = max( 1.0 - (E_EUV3 > hν_pet) * Ya_C( E_EUV3, agr_PAH ), 0.0 )
    Lyα_NoPE  = max( 1.0 - (ELyα   > hν_pet) * Ya_C( ELyα,   agr_PAH ), 0.0 )

    # Compute ∫dν [4πJ(ν)/hν] Qabs Pdiss ≈ ∑_B [4πJ_B/E_B] Qabs(E_B) Pdiss(E_B):
    Integral  = @fastmath ( Integral_prefact_FUV_PAH_diss  * FUV_NoPE  * J_FUV
                          + Integral_prefact_LW_PAH_diss   * LW_NoPE   * J_LW 
                          + Integral_prefact_EUV1_PAH_diss * EUV1_NoPE * J_EUV1
                          + Integral_prefact_EUV2_PAH_diss * EUV2_NoPE * J_EUV2
                          + Integral_prefact_EUV3_PAH_diss * EUV3_NoPE * J_EUV3
                          + Integral_prefact_Lyα_PAH_diss  * Lyα_NoPE  * e_Lyα )

    # Put it all together: 
    tphoto_PAH = tphoto_PAH_prefact / Integral 

    return tphoto_PAH
end


const t_growth_destruction_min::Float64 = 0.1 * yr 

""" This function updates the dust and gas metal density, considering 
    dust growth by accretion, and destruction by sublimation, thermal
    sputtering, non-thermal sputtering, chemical sputtering. """
@inline function DustGrowthDestruction_Update( ρd_old::Float64, 
                                              ρdplusZ::Float64,  # Density in this dust bin + currently available gas-phase limiting species
                                                ρ_old::Float64,
                                                   nH::Float64, 
                                                  nHI::Float64,
                                                 nHII::Float64,
                                                    T::Float64, 
                                                   Td::Float64,
                                                 urel::Float64,
                                                  xCI::Float64,
                                                 xCII::Float64,
                                                  Zgr::Float64,
                                                J_FUV::Float64,
                                                 J_LW::Float64,
                                               J_EUV1::Float64,
                                               J_EUV2::Float64,
                                               J_EUV3::Float64,
                                                e_Lyα::Float64,
                                             DustType::Union{Val{:Sil}, Val{:Gra}, Val{:PAH}},
                                                Δtsub::Float64
                                               )

    # Gas accretion dust growth time-scale:
    tgr0  = Compute_tgr0( T, Td, xCI, xCII, Zgr, ρdplusZ, DustType )

    # Sublimation time-sale:
    tdest_subl      = Compute_tsubl( Td, DustType )

    # Thermal sputtering time-scale:
    tdest_sputt_th  = Compute_tsputter_thermal( ρ_old, T, DustType )

    # Non-thermal sputtering time-scale:
    tdest_sputt_nth = Compute_tsputter_nonthermal( nH, urel, DustType )

    # Chemical sputtering time-scale (only relevant for graphite):
    tdest_chemsp    = Compute_tchemsputter( T, Td, nHI, nHII, Zgr, DustType )

    # Photodissociation time-scale (only relevant for PAHs):
    tdest_photo     = Compute_tphoto_PAH( J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, e_Lyα, Zgr, DustType )  

    # Get dust destruction time-scale, considering all processes:
    tdest = 1 / ( (1 / tdest_subl)   +  (1 / tdest_sputt_th)  +  (1 / tdest_sputt_nth)
                + (1 / tdest_chemsp) +  (1 / tdest_photo) )

    # If any of the time-scales are shorter than the shortest allowed time-scale, 
    # scale both of them up accordingly in a manner that preserves the relative 
    # importance of growth & destruction:
    tmin  = min( tdest, tgr0 )

    if tmin < t_growth_destruction_min
        ScaleFactor  = t_growth_destruction_min / tmin 
        tdest       *= ScaleFactor
        tgr0        *= ScaleFactor 
    end

    # Get updated dust & metal density:
    ρd_new = ρd_Update_BDF1( ρd_old, ρdplusZ, tgr0, tdest, Δtsub )

    return ρd_new
end 



########################################################################################
#######                                                                          #######
#######           R E A D  I N  D U S T  D A T A  F O R  R T  B A N D S          #######
#######                                                                          #######
########################################################################################


""" Load log-log interpolations of the Planck-mean (κP) and Rosseland-mean (κR)
    dust opacities from a saved file (kappa_thermal_*.txt produced by
    ComputeDustOpacities.jl). Returns two interpolants on log10(T). """
function LoadκMeansInterpolations( FileName::String )

    Data  = readdlm(FileName, comments = true)
    T     = Data[:, 1]
    κP    = Data[:, 2]
    κR    = Data[:, 3]

    logT  = log10.(T)
    logκP = log10.(κP)
    logκR = log10.(κR)

    itpP  = LinearInterpolation(logT, logκP, extrapolation_bc = Flat())
    itpR  = LinearInterpolation(logT, logκR, extrapolation_bc = Flat())

    return itpP, itpR
end


""" Load the band absorption (κabs) and flux (κF) opacities for one dust species
    from a saved file (kappa_graphite.txt / kappa_PAH.txt / kappa_silicate.txt
    produced by ComputeDustOpacities.jl).

    Returns a Dict{String, NamedTuple} mapping band name → (κabs, κF). """
function LoadBandOpacities( FileName::String )

    lines = readlines(FileName)
    bands = Dict{String, NamedTuple}()

    for line in lines
        isempty(line)          && continue
        startswith(line, "#")  && continue

        parts = split(strip(line))
        # Expected columns: Band  E_lo(eV)  E_hi(eV)  κabs(cm²/g)  κF(cm²/g)
        @assert length(parts) ≥ 5 "Unexpected format in band opacity file: $FileName"

        band = parts[1]
        κabs = parse(Float64, parts[4])
        κF   = parse(Float64, parts[5])

        bands[band] = (κabs = κabs, κF = κF)
    end

    # Sanity check: require all expected bands:
    for b in ("opt", "FUV", "LW", "EUV1", "EUV2", "EUV3", "Lya")
        @assert haskey(bands, b) "Missing band '$b' in band opacity file: $FileName"
    end

    return bands
end


""" This function sets up a dust struct containing needed dust optical 
    properties. """
function SetupDustState( DustGrowthDestruction::Bool,
                                           N_r::Int     = 128, 
                                           N_Z::Int     = 128,
                                           Td0::Float64 = 30.0,
                                DustOpacityDir::String  = joinpath(@__DIR__, "DustOpacityData") )

    # Save PAH acetyle dissociation probabilities (just for documentation):
    WritePAHDissociationProbabilities( DustOpacityDir )

    # Initialize dust temperature for every cell:
    Td_Sil  = fill( Td0, N_r, N_Z )
    Td_C    = fill( Td0, N_r, N_Z )
    Td_PAH  = fill( Td0, N_r, N_Z )

    # Initialize IR radiation temperature for every cell:
    Trad    = zeros( Float64, N_r, N_Z )

    # Initialize grain charges:
    Zgr_Sil = zeros( Float64, N_r, N_Z )
    Zgr_C   = zeros( Float64, N_r, N_Z )
    Zgr_PAH = zeros( Float64, N_r, N_Z )    

    # Load pre-computed opacities:
    bands_Sil = LoadBandOpacities( joinpath(DustOpacityDir, "kappa_silicate.txt") )
    bands_C   = LoadBandOpacities( joinpath(DustOpacityDir, "kappa_graphite.txt") )
    bands_PAH = LoadBandOpacities( joinpath(DustOpacityDir, "kappa_PAH.txt") )

    # Load Planck & Rosseland mean opacity interpolations:
    logκP_Sil_itp, logκR_Sil_itp = LoadκMeansInterpolations( joinpath(DustOpacityDir, "kappa_thermal_silicate.txt") )
    logκP_C_itp,   logκR_C_itp   = LoadκMeansInterpolations( joinpath(DustOpacityDir, "kappa_thermal_graphite.txt") )
    logκP_PAH_itp, logκR_PAH_itp = LoadκMeansInterpolations( joinpath(DustOpacityDir, "kappa_thermal_PAH.txt") )

    return DustState( # Dust growth/destruction setting:
                      DustGrowthDestruction,

                      # Dust temperatures for every cell:
                      Td_Sil, Td_C, Td_PAH,

                      # IR radiation temperature for every cell:
                      Trad,

                      # Grain charges:
                      Zgr_Sil, Zgr_C, Zgr_PAH,
    
                      # Band absorption opacities — silicate (cm² g⁻¹):
                      bands_Sil["opt"].κabs,  bands_Sil["FUV"].κabs,  bands_Sil["LW"].κabs,
                      bands_Sil["EUV1"].κabs, bands_Sil["EUV2"].κabs, bands_Sil["EUV3"].κabs,
                      bands_Sil["Lya"].κabs,

                      # Band absorption opacities — graphite (cm² g⁻¹):
                      bands_C["opt"].κabs,  bands_C["FUV"].κabs,  bands_C["LW"].κabs,
                      bands_C["EUV1"].κabs, bands_C["EUV2"].κabs, bands_C["EUV3"].κabs,
                      bands_C["Lya"].κabs,

                      # Band absorption opacities — PAH (cm² g⁻¹):
                      bands_PAH["opt"].κabs,  bands_PAH["FUV"].κabs,  bands_PAH["LW"].κabs,
                      bands_PAH["EUV1"].κabs, bands_PAH["EUV2"].κabs, bands_PAH["EUV3"].κabs,
                      bands_PAH["Lya"].κabs,

                      # Band flux opacities — silicate (cm² g⁻¹):
                      bands_Sil["opt"].κF,  bands_Sil["FUV"].κF,  bands_Sil["LW"].κF,
                      bands_Sil["EUV1"].κF, bands_Sil["EUV2"].κF, bands_Sil["EUV3"].κF,
                      bands_Sil["Lya"].κF,
                      
                      # Band flux opacities — graphite (cm² g⁻¹):
                      bands_C["opt"].κF,  bands_C["FUV"].κF,  bands_C["LW"].κF,
                      bands_C["EUV1"].κF, bands_C["EUV2"].κF, bands_C["EUV3"].κF,
                      bands_C["Lya"].κF,

                      # Band flux opacities — PAH (cm² g⁻¹):
                      bands_PAH["opt"].κF,  bands_PAH["FUV"].κF,  bands_PAH["LW"].κF,
                      bands_PAH["EUV1"].κF, bands_PAH["EUV2"].κF, bands_PAH["EUV3"].κF,
                      bands_PAH["Lya"].κF,
                      
                      # Interpolations for κP and κR (cm² g⁻¹):
                      logκP_Sil_itp, logκP_C_itp, logκP_PAH_itp,
                      logκR_Sil_itp, logκR_C_itp, logκR_PAH_itp
                      
                      )

end