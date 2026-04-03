
include( "./ConstantsAndFunctions.jl" )
using .ConstantsAndFunctions
using QuadGK   
using Interpolations          # For interpolations
using DelimitedFiles          # For reading data
using SpecialFunctions

const ρgr_Sil::Float64 = 3.5 
const ρgr_C::Float64   = 2.24  

# bC = 3e-5, R_V = 5.5, model B:
const bC::Float64    = 3e-5 
const a_01::Float64  = 3.5 * Ang
const a_02::Float64  = 30  * Ang
const σ::Float64     = 0.4 
const b_C1::Float64  = 0.75 * bC 
const b_C2::Float64  = 0.25 * bC 

# Parameters for carbonaceous grain size distribution:
const a_tg::Float64  = 0.0120 * μm 
const a_cg::Float64  = 7.28   * μm
const αg::Float64    = - 1.90
const βg::Float64    = - 0.0517
const Cg::Float64    = 2.86e-12 

# Parameters for silicate grain size distribution:
const a_ts::Float64  = 0.211 * μm 
const a_cs::Float64  = 0.1   * μm
const αs::Float64    = - 1.13
const βs::Float64    = - 0.109
const Cs::Float64    = 1.04e-13


const B1 = ( ( (3 / (2*π)^(3/2)) * exp( - 4.5 * σ^2) / (ρgr_C * (a_01^3) * σ) )
           * b_C1 * mC / ( 1 + erf( 3*σ/sqrt(2) + log(a_01 / (3.5 * Ang))/(σ * sqrt(2)) ) ) ) 

const B2 = ( ( (3 / (2*π)^(3/2)) * exp( - 4.5 * σ^2) / (ρgr_C * (a_02^3) * σ) )
           * b_C2 * mC / ( 1 + erf( 3*σ/sqrt(2) + log(a_02 / (3.5 * Ang))/(σ * sqrt(2)) ) ) ) 

""" Eqs. (2)-(3) in Weingartner & Draine (2001). """
function D( a::Float64 )

    if a > 3.5 * Ang
        D = ( (B1 / a) * exp( - (1/2) * (log(a / a_01)/σ)^2 ) 
            + (B2 / a) * exp( - (1/2) * (log(a / a_02)/σ)^2 ) )
    else
        D = 0.0
    end

    return D 
end


""" Eq. (6) in Weingartner & Draine (2001). """
function F( a::Float64, β::Float64, a_t::Float64 )

    if β ≥ 0.0
        return 1 + β * a / a_t 
    else
        return 1 / (1 - β * a / a_t)
    end
end


const a_min::Float64 = 3.5 * Ang 
const a_max::Float64 = 10.0 * μm 

function ∂ngr_∂a_C( a::Float64 )

    if a ≤ a_min || a ≥ a_max
        return 0.0
    end

    if a > 3.5 * Ang && a < a_tg
        f = 1.0
    else
        f = exp( - ((a - a_tg)/a_cg)^3 )
    end

    return D(a) + (Cg / a) * ((a/a_tg)^αg) * F(a, βg, a_tg) * f
end

function ∂ngr_∂a_Sil( a::Float64 )

    if a ≤ a_min || a ≥ a_max
        return 0.0
    end

    if a > 3.5 * Ang && a < a_ts 
        f = 1.0
    else
        f = exp( - ((a - a_ts)/a_cs)^3 )
    end

    return (Cs / a) * ((a/a_ts)^αs) * F(a, βs, a_ts) * f 
end


""" Computes ⟨aⁿ⟩^(1/n) (in μm) for silicate dust. """
function Compute_a_n_sil( n::Float64 )
    # Compute ∫da ∂ngr_∂a_Sil aⁿ:
    num,   _ = quadgk( a -> ∂ngr_∂a_Sil(a) * a^n, a_min, a_max; rtol = 1e-9 )

    # Compute ∫da ∂ngr_∂a_Sil:
    denom, _ = quadgk( a -> ∂ngr_∂a_Sil(a),       a_min, a_max; rtol = 1e-9 )

    return ( (num / denom)^(1/n) ) / μm
end


""" Computes ⟨aⁿ⟩^(1/n) (in μm) for graphite/carbonaceous dust. """
function Compute_a_n_C( n::Float64 )
    # Compute ∫da ∂ngr_∂a_C aⁿ:
    num,   _ = quadgk( a -> ∂ngr_∂a_C(a) * a^n, a_min, a_max; rtol = 1e-9 )

    # Compute ∫da ∂ngr_∂a_C:
    denom, _ = quadgk( a -> ∂ngr_∂a_C(a),       a_min, a_max; rtol = 1e-9 )

    return ( (num / denom)^(1/n) ) / μm
end


""" Computes the fraction of dust mass in silicate grains, ρd_sil/ρd. """
function Compute_Sil_to_tot_ratio()

    # Compute ∫da ∂ngr_∂a_Sil a^3:
    Int_Sil,   _ = quadgk( a -> ∂ngr_∂a_Sil(a) * a^3, a_min, a_max; rtol = 1e-9 )

    # Non-normalized silicate dust density:
    ρd_Sil =  ρgr_Sil * Int_Sil

    # Compute ∫da ∂ngr_∂a_C a^3:
    Int_C,   _ = quadgk( a -> ∂ngr_∂a_C(a) * a^3, a_min, a_max; rtol = 1e-9 )

    # Non-normalized graphite dust density:
    ρd_C =  ρgr_C * Int_C


    return ρd_Sil / (ρd_Sil + ρd_C)
end


# SILICATES Qabs:

SilQabsData   = readdlm( joinpath(@__DIR__, "SilicatesQabs_a1.4e-1.txt"), skipstart = 8 )

λSilQabsData  = SilQabsData[:,1]
orderSilQabs  = sortperm(λSilQabsData)  # Get correct order, so that it goes from low to high λ instead of the reverse
λSilQabsData  = λSilQabsData[orderSilQabs]
QabsSilData   = SilQabsData[:,2][orderSilQabs]

# GRAPHITES Qabs:

CQabsData   = readdlm( joinpath(@__DIR__, "GraphiteQabs_a6.3e-3.txt"), skipstart = 8 )

λCQabsData  = CQabsData[:,1]
orderCQabs  = sortperm(λCQabsData)  # Get correct order, so that it goes from low to high λ instead of the reverse
λCQabsData  = λCQabsData[orderCQabs]
QabsCData   = CQabsData[:,2][orderCQabs]

# Get interpolations:
const QabsSilData_itp = LinearInterpolation( λSilQabsData, QabsSilData, extrapolation_bc = Flat() )
const QabsCData_itp   = LinearInterpolation( λCQabsData, QabsCData, extrapolation_bc = Flat() )

@inline function QabsSil( E::Float64 )
    # Convert from E (erg) to λ (μm):
    λ = h * c / (E * μm)
    return QabsSilData_itp( λ )
end

@inline function QabsC( E::Float64 )
    # Convert from E (erg) to λ (μm):
    λ = h * c / (E * μm)
    return QabsCData_itp( λ )
end


QabsSil_FUV    = QabsSil(8.45 * eV)
QabsSil_LW     = QabsSil(12.26 * eV)
QabsSil_EUV1   = QabsSil(18.01 * eV)
QabsSil_EUV2   = QabsSil(29.89 * eV)
QabsSil_EUV3   = QabsSil(56.85 * eV)
QabsSil_Lyα    = QabsSil(ELyα)

QabsC_FUV      = QabsC(8.45 * eV)
QabsC_LW       = QabsC(12.26 * eV)
QabsC_EUV1     = QabsC(18.01 * eV)
QabsC_EUV2     = QabsC(29.89 * eV)
QabsC_EUV3     = QabsC(56.85 * eV)
QabsC_Lyα      = QabsC(ELyα)

println( "Area-weighted mean silicate grain size (μm):", (Compute_a_n_sil( 3.0 )^3) / (Compute_a_n_sil( 2.0 )^2) )
println( "Area-weighted mean graphite/carbonaceous grain size (μm):", (Compute_a_n_C( 3.0 )^3) / (Compute_a_n_C( 2.0 )^2) )
println( "Fraction of dust mass in silicate grains: ", Compute_Sil_to_tot_ratio() )

println( "Silicates Qabs_FUV: ", QabsSil_FUV )
println( "Silicates Qabs_LW: ", QabsSil_LW )
println( "Silicates Qabs_EUV1: ", QabsSil_EUV1 )
println( "Silicates Qabs_EUV2: ", QabsSil_EUV2 )
println( "Silicates Qabs_EUV3: ", QabsSil_EUV3 )
println( "Silicates Qabs_Lyα: ", QabsSil_Lyα )
println("")
println( "Graphite Qabs_FUV: ", QabsC_FUV )
println( "Graphite Qabs_LW: ", QabsC_LW )
println( "Graphite Qabs_EUV1: ", QabsC_EUV1 )
println( "Graphite Qabs_EUV2: ", QabsC_EUV2 )
println( "Graphite Qabs_EUV3: ", QabsC_EUV3 )
println( "Graphite Qabs_Lyα: ", QabsC_Lyα )