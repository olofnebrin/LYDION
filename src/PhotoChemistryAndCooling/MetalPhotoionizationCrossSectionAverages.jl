""" FILE:    MetalPhotoionizationCrossSectionAverages.jl

    PURPOSE: Compute band-averaged photoionization cross-sections and
             photoheating energies for metal species, using Verner et al. (1996) 
             fits.
"""

include(joinpath(@__DIR__, "..", "ConstantsAndFunctions.jl"))
using .ConstantsAndFunctions

struct PhotoBandAverage
    σ_N::Float64   # Photon-number weighted cross-section
    σ_E::Float64   # Energy-weighted cross-section (for absorption of J and H)
      h::Float64   # Mean excess heating energy per photoionization
end

struct PhotoSpecies
    name::Symbol
     Eth::Float64
       σ::Function
end



""" General fit to photoionization cross-sections by Verner et al. (1996). """
function CrossSectionFit_Verner96( E_eV::Float64, # Photon energy in eV 
                                     σ0::Float64, # In 10⁻¹⁸ cm² (Mb)
                                  E0_eV::Float64, 
                                     yw::Float64, 
                                     ya::Float64, 
                                      P::Float64, 
                                     y0::Float64,
                                     y1::Float64,
                                 Eth_eV::Float64 )

    if E_eV < Eth_eV
        # Exit early if photon energy is below the threshold:
        return 0.0
    end

    x = (E_eV / E0_eV ) - y0 
    y = sqrt( x * x  +  y1 * y1 )
    F = ((x - 1)^2  +  yw * yw) * (y^(0.5*P - 5.5)) * ((1 + sqrt(y/ya))^(-P))

    return σ0 * F * 1e-18 
end


""" The photon energy-dependent H I photoionization cross-section (cm²). """
function σHI( E::Float64 )
    # Get photon energy in eV:
    E_eV   = E / eV 
    # Constants for the H I cross-section:
    Eth_eV = 1.360e1 
    E0_eV  = 4.298e-1
    σ0     = 5.475e4
    ya     = 3.288e1 
    P      = 2.963e0
    yw     = 0.0
    y0     = 0.0
    y1     = 0.0

    # Get cross-section: 
    σHI    = CrossSectionFit_Verner96( E_eV, σ0, E0_eV, yw, 
                                       ya, P, y0, y1, Eth_eV )

    return σHI
end


""" The photon energy-dependent He I photoionization cross-section (cm²). """
function σHeI( E::Float64 )
    # Get photon energy in eV:
    E_eV   = E / eV 
    # Constants for the He I cross-section:
    Eth_eV = 2.459e1
    E0_eV  = 1.361e1
    σ0     = 9.492e2
    ya     = 1.469e0
    P      = 3.188e0
    yw     = 2.039e0
    y0     = 4.434e-1
    y1     = 2.136e0

    # Get cross-section: 
    σHeI   = CrossSectionFit_Verner96( E_eV, σ0, E0_eV, yw, 
                                       ya, P, y0, y1, Eth_eV )

    return σHeI
end


""" The photon energy-dependent He II photoionization cross-section (cm²). """
function σHeII( E::Float64 )
    # Get photon energy in eV:
    E_eV   = E / eV 
    # Constants for the He II cross-section:
    Eth_eV = 5.442e1
    E0_eV  = 1.720e0
    σ0     = 1.369e4
    ya     = 3.288e1
    P      = 2.963e0
    yw     = 0.0
    y0     = 0.0
    y1     = 0.0

    # Get cross-section: 
    σHeII  = CrossSectionFit_Verner96( E_eV, σ0, E0_eV, yw, 
                                       ya, P, y0, y1, Eth_eV )

    return σHeII
end


""" The photon energy-dependent C I photoionization cross-section (cm²).
    See Verner et al. (1996). """
function σCI( E::Float64 )
    # Get photon energy in eV:
    E_eV   = E / eV 
    # Constants for the C I cross-section:
    Eth_eV = 1.126e1
    E0_eV  = 2.144e0
    σ0     = 5.027e2
    ya     = 6.216e1
    P      = 5.101e0
    yw     = 9.157e-2
    y0     = 1.133e0
    y1     = 1.607e0

    # Get cross-section: 
    σCI  = CrossSectionFit_Verner96( E_eV, σ0, E0_eV, yw, 
                                     ya, P, y0, y1, Eth_eV )

    return σCI
end


""" The photon energy-dependent C II photoionization cross-section (cm²).
    See Verner et al. (1996). """
function σCII( E::Float64 )
    # Get photon energy in eV:
    E_eV   = E / eV 
    # Constants for the C II cross-section:
    Eth_eV = 2.438e1
    E0_eV  = 4.058e-1
    σ0     = 8.709e0
    ya     = 1.261e2
    P      = 8.578e0
    yw     = 2.093e0
    y0     = 4.929e1
    y1     = 3.234e0

    # Get cross-section: 
    σCII  = CrossSectionFit_Verner96( E_eV, σ0, E0_eV, yw, 
                                      ya, P, y0, y1, Eth_eV )

    return σCII
end


""" The photon energy-dependent O I photoionization cross-section (cm²).
    See Verner et al. (1996). """
function σOI( E::Float64 )
    # Get photon energy in eV:
    E_eV   = E / eV 
    # Constants for the O I cross-section:
    Eth_eV = 13.62
    E0_eV  = 1.240e0
    σ0     = 1.745e3 
    ya     = 3.784e0
    P      = 1.764e1
    yw     = 7.589e-2
    y0     = 8.698e0
    y1     = 1.271e-1

    # Get cross-section: 
    σOI  = CrossSectionFit_Verner96( E_eV, σ0, E0_eV, yw, 
                                     ya, P, y0, y1, Eth_eV )

    return σOI
end


""" The photon energy-dependent O II photoionization cross-section (cm²).
    See Verner et al. (1996). """
function σOII( E::Float64 )
    # Get photon energy in eV:
    E_eV   = E / eV 
    # Constants for the O II cross-section:
    Eth_eV = 3.512e1
    E0_eV  = 1.386e0
    σ0     = 5.967e1
    ya     = 3.175e1
    P      = 8.943e0
    yw     = 1.934e-2
    y0     = 2.131e1
    y1     = 1.503e-2

    # Get cross-section: 
    σOII   = CrossSectionFit_Verner96( E_eV, σ0, E0_eV, yw, 
                                       ya, P, y0, y1, Eth_eV )

    return σOII
end


""" The photon energy-dependent O III photoionization cross-section (cm²).
    See Verner et al. (1996). """
function σOIII( E::Float64 )
    # Get photon energy in eV:
    E_eV   = E / eV 
    # Constants for the O III cross-section:
    Eth_eV = 5.494e1
    E0_eV  = 1.723e-1
    σ0     = 6.753e2
    ya     = 3.852e2
    P      = 6.822e0
    yw     = 1.191e-1
    y0     = 3.839e-3
    y1     = 4.569e-1

    # Get cross-section: 
    σOIII  = CrossSectionFit_Verner96( E_eV, σ0, E0_eV, yw, 
                                       ya, P, y0, y1, Eth_eV )

    return σOIII
end


""" The photon energy-dependent S I photoionization cross-section (cm²).
    See Verner et al. (1996). """
function σSI( E::Float64 )
    # Get photon energy in eV:
    E_eV   = E / eV 
    # Constants for the S I cross-section:
    Eth_eV = 1.036e1
    E0_eV  = 1.808e1
    σ0     = 4.564e4
    ya     = 1.000e0  # Yes, Verner+ list 1.0 exactly... 
    P      = 1.361e1
    yw     = 6.385e-1
    y0     = 9.935e-1
    y1     = 2.486e-1

    # Get cross-section: 
    σSI  = CrossSectionFit_Verner96( E_eV, σ0, E0_eV, yw, 
                                     ya, P, y0, y1, Eth_eV )

    return σSI
end


""" The photon energy-dependent S II photoionization cross-section (cm²).
    See Verner et al. (1996). """
function σSII( E::Float64 )
    # Get photon energy in eV:
    E_eV   = E / eV 
    # Constants for the S II cross-section:
    Eth_eV = 2.333e1
    E0_eV  = 8.787e0
    σ0     = 3.136e2
    ya     = 3.442e0
    P      = 1.281e1
    yw     = 7.354e-1
    y0     = 2.782e0
    y1     = 1.788e-1

    # Get cross-section: 
    σSII  = CrossSectionFit_Verner96( E_eV, σ0, E0_eV, yw, 
                                      ya, P, y0, y1, Eth_eV )

    return σSII
end


""" The photon energy-dependent S III photoionization cross-section (cm²).
    See Verner et al. (1996). """
function σSIII( E::Float64 )
    # Get photon energy in eV:
    E_eV   = E / eV 
    # Constants for the S III cross-section:
    Eth_eV = 3.483e1
    E0_eV  = 2.027e0
    σ0     = 6.666e0
    ya     = 5.454e1
    P      = 8.611e0
    yw     = 4.109e0
    y0     = 1.568e1
    y1     = 9.421e0

    # Get cross-section: 
    σSIII  = CrossSectionFit_Verner96( E_eV, σ0, E0_eV, yw, 
                                       ya, P, y0, y1, Eth_eV )

    return σSIII
end


""" Simple mid-point integration over photon energy E. """
function IntegrateEnergyBand( f::Function, E_min::Float64, E_max::Float64, N::Int = 4096 )

    # Initialize step size (in energy, ΔE), and integrand value:
    ΔE    = (E_max - E_min) / N 
    ∫dE_f = 0.0

    @inbounds for i in 1:N
        E_mid  = E_min  +  (i - 0.5) * ΔE
        f_mid  = f(E_mid) 
        ∫dE_f += @fastmath f_mid * ΔE
    end

    return ∫dE_f
end


""" This function computes the photon number-weighted cross-section 
    between E_min and E_max:
    
    σ_N = [∫dν 4πJ σ(hν) / hν] / [∫dν 4πJ / hν] """
function Compute_σ_N( J::Function,     σ::Function,
                  E_min::Float64,  E_max::Float64, N::Int = 4096 )

    numerator = IntegrateEnergyBand(
        E -> J(E) * σ(E) / E,
        E_min, E_max, N
    )

    denominator = IntegrateEnergyBand(
        E -> J(E) / E,
        E_min, E_max, N
    )

    if denominator ≤ 0.0
        return 0.0
    end

    # Put it all together: 
    σ_N = numerator / denominator

    return σ_N
end


""" This function computes the photon energy-weighted cross-section between 
    energy E_min and E_max:
    
    σ_E = [∫dν 4πJ σ(hν)] / [∫dν 4πJ] """
function Compute_σ_E( J::Function,    σ::Function, 
                  E_min::Float64, E_max::Float64, N::Int = 4096 )

    numerator = IntegrateEnergyBand(
        E -> J(E) * σ(E),
        E_min, E_max, N
    )

    denominator = IntegrateEnergyBand(
        E -> J(E),
        E_min, E_max, N
    )

    if denominator ≤ 0.0
        return 0.0
    end

    # Put it all together:
    σ_E = numerator / denominator

    return σ_E
end


""" This function computes the photoheating from the excess energy 
    between energy E_min and E_max. """
function Compute_photoheating( J::Function,    σ::Function, 
                           E_min::Float64, E_max::Float64,  E_th::Float64,
                               N::Int = 4096 )

    numerator = IntegrateEnergyBand(
        E -> J(E) * σ(E) * max(E - E_th, 0.0) / E,
        E_min, E_max, N
    )

    denominator = IntegrateEnergyBand(
        E -> J(E) * σ(E) / E,
        E_min, E_max, N
    )

    if denominator ≤ 0.0
        return 0.0
    end

    # Put it all together:
    heating = numerator / denominator

    return heating 
end


###############################################################
#######                                                 #######
#######              B A N D   A V E R A G E S          #######
#######                                                 #######
###############################################################


# Stellar radiation bands, in photon energy units:
const PhotoBands = (
    opt  = (1.0  * eV, 5.8  * eV),
    FUV  = (5.8  * eV, 11.2 * eV),
    LW   = (11.2 * eV, 13.6 * eV),
    EUV1 = (13.6 * eV, 24.6 * eV),
    EUV2 = (24.6 * eV, 54.4 * eV),
    EUV3 = (54.4 * eV, 100.0 * eV),
)


# Species for which we compute photoionization cross-sections:
const PhotoSpeciesList = (
    #HI    = PhotoSpecies(:HI,    13.60  * eV, σHI),
    #HeI   = PhotoSpecies(:HeI,   24.59  * eV, σHeI),
    #HeII  = PhotoSpecies(:HeII,  54.42  * eV, σHeII),

    CI    = PhotoSpecies(:CI,    11.26  * eV, σCI),
    CII   = PhotoSpecies(:CII,   24.38  * eV, σCII),

    OI    = PhotoSpecies(:OI,    13.62  * eV, σOI),
    OII   = PhotoSpecies(:OII,   35.12  * eV, σOII),
    OIII  = PhotoSpecies(:OIII,  54.94  * eV, σOIII),

    SI    = PhotoSpecies(:SI,    10.36  * eV, σSI),
    SII   = PhotoSpecies(:SII,   23.33  * eV, σSII),
    SIII  = PhotoSpecies(:SIII,  34.83  * eV, σSIII),
)


""" Black-body spectral shape as a function of photon energy E (erg).

    This is proportional to B_ν(T), written as a function of E = hν.
    Constants cancel in all weighted averages below. """
@inline function BlackBodyShape_E( E::Float64, T::Float64 )

    x = E / (kB * T)

    if x > 700.0
        return 0.0
    end

    return (E^3) / expm1(x)
end


""" Compute all band-averaged photoionization quantities for one species 
    in one given photon energy band. """
function ComputePhotoBandAverage( J::Function,
                            Species::PhotoSpecies,
                              E_min::Float64,
                              E_max::Float64,
                                  N::Int = 4096 )

    # No ionization possible if the entire band is below threshold:
    if E_max ≤ Species.Eth
        return PhotoBandAverage(0.0, 0.0, 0.0)
    end

    # Compute number-weighted and energy-weighted cross-sections, and heating.
    # NOTE: Use the full band interval. The cross-section function itself
    # returns zero below the ionization threshold, so the numerator is automatically restricted
    # to ionizing photons, while the denominator remains the full band average.
    σ_N = Compute_σ_N(J, Species.σ, E_min, E_max, N)
    σ_E = Compute_σ_E(J, Species.σ, E_min, E_max, N)
    h   = Compute_photoheating(J, Species.σ, E_min, E_max, Species.Eth, N)

    return PhotoBandAverage(σ_N, σ_E, h)
end


###############################################################
#######                                                 #######
#######     D E F A U L T   B A N D   A V E R A G E S   #######
#######                                                 #######
###############################################################

# The code below computes the default band-averaged photoionization 
# cross-sections (number & energy-weighted), and corresponding
# photoheating.  

# Default black-body effective temperature used to compute band averages
# is set to 4.8 × 10⁴ K, which represents roughly the spectrum of a 
# 30 M⊙ star:
const T_PHOTOAVERAGES::Float64 = 4.8e4

# Default spectral shape used for the averages:
@inline J_PhotoAverages( E::Float64 ) = BlackBodyShape_E(E, T_PHOTOAVERAGES)

# C I:
const CI_LW_Avg     = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.CI, PhotoBands.LW...)
const CI_EUV1_Avg   = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.CI, PhotoBands.EUV1...)
const CI_EUV2_Avg   = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.CI, PhotoBands.EUV2...)
const CI_EUV3_Avg   = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.CI, PhotoBands.EUV3...)

# C II:
const CII_EUV1_Avg  = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.CII, PhotoBands.EUV1...)
const CII_EUV2_Avg  = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.CII, PhotoBands.EUV2...)
const CII_EUV3_Avg  = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.CII, PhotoBands.EUV3...)

# O I:
const OI_EUV1_Avg   = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.OI, PhotoBands.EUV1...)
const OI_EUV2_Avg   = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.OI, PhotoBands.EUV2...)
const OI_EUV3_Avg   = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.OI, PhotoBands.EUV3...)

# O II:
const OII_EUV2_Avg  = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.OII, PhotoBands.EUV2...)
const OII_EUV3_Avg  = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.OII, PhotoBands.EUV3...)

# O III:
const OIII_EUV3_Avg = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.OIII, PhotoBands.EUV3...)

# S I:
const SI_FUV_Avg    = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.SI, PhotoBands.FUV...)
const SI_LW_Avg     = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.SI, PhotoBands.LW...)
const SI_EUV1_Avg   = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.SI, PhotoBands.EUV1...)
const SI_EUV2_Avg   = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.SI, PhotoBands.EUV2...)
const SI_EUV3_Avg   = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.SI, PhotoBands.EUV3...)

# S II:
const SII_EUV1_Avg  = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.SII, PhotoBands.EUV1...)
const SII_EUV2_Avg  = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.SII, PhotoBands.EUV2...)
const SII_EUV3_Avg  = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.SII, PhotoBands.EUV3...)

# S III:
const SIII_EUV2_Avg = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.SIII, PhotoBands.EUV2...)
const SIII_EUV3_Avg = ComputePhotoBandAverage(J_PhotoAverages, PhotoSpeciesList.SIII, PhotoBands.EUV3...)


###############################################################
#######                                                 #######
#######           G A T H E R  R E S U L T S            #######
#######                                                 #######
###############################################################

# Below the results for the cross-sections & heating are 
# gathered into handy constants. 

# C I:
const σCI_LW_N      = CI_LW_Avg.σ_N
const σCI_LW_E      = CI_LW_Avg.σ_E
const hCI_LW        = CI_LW_Avg.h

const σCI_EUV1_N    = CI_EUV1_Avg.σ_N
const σCI_EUV1_E    = CI_EUV1_Avg.σ_E
const hCI_EUV1      = CI_EUV1_Avg.h

const σCI_EUV2_N    = CI_EUV2_Avg.σ_N
const σCI_EUV2_E    = CI_EUV2_Avg.σ_E
const hCI_EUV2      = CI_EUV2_Avg.h

const σCI_EUV3_N    = CI_EUV3_Avg.σ_N
const σCI_EUV3_E    = CI_EUV3_Avg.σ_E
const hCI_EUV3      = CI_EUV3_Avg.h

println("σCI_LW_N   = $(σCI_LW_N/1e-18)   × 10⁻¹⁸ cm²")
println("σCI_EUV1_N = $(σCI_EUV1_N/1e-18) × 10⁻¹⁸ cm²")
println("σCI_EUV2_N = $(σCI_EUV2_N/1e-18) × 10⁻¹⁸ cm²")
println("σCI_EUV3_N = $(σCI_EUV3_N/1e-18) × 10⁻¹⁸ cm²")
println("")

# C II:
const σCII_EUV1_N   = CII_EUV1_Avg.σ_N
const σCII_EUV1_E   = CII_EUV1_Avg.σ_E
const hCII_EUV1     = CII_EUV1_Avg.h

const σCII_EUV2_N   = CII_EUV2_Avg.σ_N
const σCII_EUV2_E   = CII_EUV2_Avg.σ_E
const hCII_EUV2     = CII_EUV2_Avg.h

const σCII_EUV3_N   = CII_EUV3_Avg.σ_N
const σCII_EUV3_E   = CII_EUV3_Avg.σ_E
const hCII_EUV3     = CII_EUV3_Avg.h

# O I:
const σOI_EUV1_N    = OI_EUV1_Avg.σ_N
const σOI_EUV1_E    = OI_EUV1_Avg.σ_E
const hOI_EUV1      = OI_EUV1_Avg.h

const σOI_EUV2_N    = OI_EUV2_Avg.σ_N
const σOI_EUV2_E    = OI_EUV2_Avg.σ_E
const hOI_EUV2      = OI_EUV2_Avg.h

const σOI_EUV3_N    = OI_EUV3_Avg.σ_N
const σOI_EUV3_E    = OI_EUV3_Avg.σ_E
const hOI_EUV3      = OI_EUV3_Avg.h

println("σOI_EUV1_N = $(σOI_EUV1_N/1e-18) × 10⁻¹⁸ cm²")
println("σOI_EUV2_N = $(σOI_EUV2_N/1e-18) × 10⁻¹⁸ cm²")
println("σOI_EUV3_N = $(σOI_EUV3_N/1e-18) × 10⁻¹⁸ cm²")

# O II:
const σOII_EUV2_N   = OII_EUV2_Avg.σ_N
const σOII_EUV2_E   = OII_EUV2_Avg.σ_E
const hOII_EUV2     = OII_EUV2_Avg.h

const σOII_EUV3_N   = OII_EUV3_Avg.σ_N
const σOII_EUV3_E   = OII_EUV3_Avg.σ_E
const hOII_EUV3     = OII_EUV3_Avg.h

# O III:
const σOIII_EUV3_N  = OIII_EUV3_Avg.σ_N
const σOIII_EUV3_E  = OIII_EUV3_Avg.σ_E
const hOIII_EUV3    = OIII_EUV3_Avg.h

# S I:
const σSI_FUV_N     = SI_FUV_Avg.σ_N
const σSI_FUV_E     = SI_FUV_Avg.σ_E
const hSI_FUV       = SI_FUV_Avg.h

const σSI_LW_N      = SI_LW_Avg.σ_N
const σSI_LW_E      = SI_LW_Avg.σ_E
const hSI_LW        = SI_LW_Avg.h

const σSI_EUV1_N    = SI_EUV1_Avg.σ_N
const σSI_EUV1_E    = SI_EUV1_Avg.σ_E
const hSI_EUV1      = SI_EUV1_Avg.h

const σSI_EUV2_N    = SI_EUV2_Avg.σ_N
const σSI_EUV2_E    = SI_EUV2_Avg.σ_E
const hSI_EUV2      = SI_EUV2_Avg.h

const σSI_EUV3_N    = SI_EUV3_Avg.σ_N
const σSI_EUV3_E    = SI_EUV3_Avg.σ_E
const hSI_EUV3      = SI_EUV3_Avg.h

# S II:
const σSII_EUV1_N   = SII_EUV1_Avg.σ_N
const σSII_EUV1_E   = SII_EUV1_Avg.σ_E
const hSII_EUV1     = SII_EUV1_Avg.h

const σSII_EUV2_N   = SII_EUV2_Avg.σ_N
const σSII_EUV2_E   = SII_EUV2_Avg.σ_E
const hSII_EUV2     = SII_EUV2_Avg.h

const σSII_EUV3_N   = SII_EUV3_Avg.σ_N
const σSII_EUV3_E   = SII_EUV3_Avg.σ_E
const hSII_EUV3     = SII_EUV3_Avg.h

# S III:
const σSIII_EUV2_N  = SIII_EUV2_Avg.σ_N
const σSIII_EUV2_E  = SIII_EUV2_Avg.σ_E
const hSIII_EUV2    = SIII_EUV2_Avg.h

const σSIII_EUV3_N  = SIII_EUV3_Avg.σ_N
const σSIII_EUV3_E  = SIII_EUV3_Avg.σ_E
const hSIII_EUV3    = SIII_EUV3_Avg.h