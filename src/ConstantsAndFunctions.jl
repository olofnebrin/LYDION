
module ConstantsAndFunctions
""" MODULE:  ConstantsAndFunctions 

    PURPOSE: Contains useful constants and units (e.g. c, νLyα, Lsun), 
             and functions (e.g. Voigt profile, optical depth, analytical
             spectra to compare to). 

"""

__precompile__()

include( "./FastInterpolations.jl" )
using SpecialFunctions     # For Dawson integral
using Base.Threads         # For CPU acceleration
using Roots                # Needed for root-finding in the frequency grid setup 
using QuadGK               # For numerical integration 
using .FastInterpolations  # For fast interpolations of special/expensive functions 

export LyαArrays, GenerateLyαArrays, H, Htophat, 
       Compute_b, Compute_ΔνD, Compute_a, Compute_σ0, 
       Update_Hvoigt!, Update_ϕν!, 
       Compute_e_Lyα, Update_e_Lyα!,
       AbsorptionUpdateJorH, LyαEmissionUpdate!, LyαEmissionAbsorptionUpdate!, 
       Update_f_Lyα_MF_L_Lyα!, Reset_Δρu!, AbsorptionUpdateH, 
       Update_∇u_RHD!, D_colon_∇u_Cylindrical, 
       J_sphere_uni, J_sphere_point,
       eLyα_sphere_point, eLyα_sphere_uni,
       D_colon_∇u_Cartesian, Compute_f_esc_Cylindrical, Compute_f_esc_Lyα_Cylindrical,
       vth_H, GetHydrogenDensities, GetHeliumDensities, GetDeuteriumDensities, 
       GetCarbonDensities, GetOxygenDensities, GetElectronDensity,
       GetElectronDensityFromFracs, GetTotalNumberDensity,
       GetAllNumberDensities, Get_ρ_From_nH, Compute_nC_over_nO,
       MinMod, FreeFallTime, SphericalAverageProfile!,
       GetBlackBodyIonizingEmission, PrintLYDIONLogo


###############################################################
#######                                                 #######
#######               C O N S T A N T S                 #######
#######                                                 #######
###############################################################


####### CONVENIENT UNIT CONVERSIONS #######
const Lsun::Float64       = 3.839e33              # Solar luminsoity (erg/s)
const Msun::Float64       = 1.988435e33           # Solar mass (g)
const AU::Float64         = 1.495978707e13        # 1 AU (cm)
const pc::Float64         = 3.085677581467192e18  # 1 parsec (cm)
const kpc::Float64        = 1e3 * pc              # 1 kiloparsec (cm)
const Mpc::Float64        = 1e6 * pc              # 1 Megaparsec (cm)
const Gpc::Float64        = 1e9 * pc              # 1 Gigaparsec (cm)
const kms::Float64        = 1e5                   # 1 km/s (cm/s)
const inv_kms::Float64    = 1 / kms               # Inverse of 1 km/s (convenient)
const day::Float64        = 86400.0               # 1 day (seconds)
const yr::Float64         = 365.24 * day          # 1 year (sec)
const Myr::Float64        = 1e6 * yr              # 1 million years (sec)
const Gyr::Float64        = 1e9 * yr              # 1 billion years (sec)
const eV::Float64         = 1.60217725e-12        # 1 eV (erg)
const Ang::Float64        = 1e-8                  # 1 Ångström (cm)
const nm::Float64         = 1e-7                  # 1 nanometer (cm)
const μm::Float64         = 1e-4                  # 1 micron (cm)
const F_Habing::Float64   = 1.6e-3                # Habing flux (erg cm⁻² s⁻¹) - see Tielens (2005)
                                                  # (equivaletly, u_Habing = F_Habing/c = 5.3 × 10⁻¹³ erg cm⁻³)

####### PHYSICAL CONSTANTS ################
const c::Float64           = 2.99792458e10         # Speed of light (cm/s)
const h::Float64           = 6.626069573e-27       # Planck's constant (erg s)
const G::Float64           = 6.6743e-8             # The gravitational constant (dyn cm² g⁻²)
const kB::Float64          = 1.380648813e-16       # Boltzmann constant (erg/K)

const mu::Float64          = 1.660539069e-24       # Atomic mass unit (g)
const mH::Float64          = 1.0080 * mu           # Mass of hydrogen atom (g)
const mH2::Float64         = 2 * mH                # Mass of H₂ molecule (g)
const mD::Float64          = 2.014  * mu           # Mass of deuterium atom (g) 
const me::Float64          = 9.109382917e-28       # Mass of electron (g)
const mHe::Float64         = 4.0026 * mu           # Mass of helium atom (g)
const mC::Float64          = 12.011 * mu           # Mass of carbon atom (g)
const mO::Float64          = 15.999 * mu           # Mass of oxygen atom (g)
const mCO::Float64         = mC  +  mO             # Mass of CO molecule (g)
const mN::Float64          = 14.007 * mu           # Mass of nitrogen atom (g)
const mNe::Float64         = 20.180 * mu           # Mass of neon atom (g)
const mMg::Float64         = 24.305 * mu           # Mass of magnesium atom (g)
const mSi::Float64         = 28.086 * mu           # Mass of silicon atom (g)
const mS::Float64          = 32.065 * mu           # Mass of sulfur atom (g)
const mFe::Float64         = 55.845 * mu           # Mass of iron atom (g)

const inv_mH::Float64      = 1 / mH 
const inv_mH2::Float64     = 1 / mH2 
const inv_mHe::Float64     = 1 / mHe 
const inv_mD::Float64      = 1 / mD 
const inv_mC::Float64      = 1 / mC 
const inv_mO::Float64      = 1 / mO 
const inv_mN::Float64      = 1 / mN 
const inv_mNe::Float64     = 1 / mNe
const inv_mMg::Float64     = 1 / mMg
const inv_mSi::Float64     = 1 / mSi
const inv_mS::Float64      = 1 / mS
const inv_mFe::Float64     = 1 / mFe

const e::Float64           = 4.80320451e-10        # Electron charge (esu)
const q_e::Float64         = - e                   # Charge of electron (esu)
const q_HII::Float64       = - e                   # Charge of proton (esu)
const q_HeII::Float64      = - e                   # Charge of He II (esu)
const q_HeIII::Float64     = - 2 * e               # Charge of He III (esu)
const σSB::Float64         = 5.670374419e-5        # Stefan-Boltzmann constant (cgs units)
const νLyα::Float64        = 2.466068e15           # Lyα frequency at line center (Hz)
const νLyC::Float64        = 3.289842e15           # LyC limit frequency (Hz)
const ELyα::Float64        = h * νLyα              # Lyα photon energy (erg)
const ELyC::Float64        = h * νLyC              # LyC limit photon energy (i.e. Rydberg) (erg)
const inv_ELyα::Float64    = 1 / ELyα              # Inverse of ELyα, convenient 
const ALyα::Float64        = 6.265e8               # Einstein-A coefficient for Lyα (s⁻¹)
const A_2γ::Float64        = 8.2206                # Einstein-A coefficient for two-photon decay (s⁻¹) 
const A_2p2s::Float64      = 8.78e-7               # Einstein-A coefficient for 2p → 2s transition (s⁻¹)
const A_2s2p::Float64      = 1.597e-9              # Einstein-A coefficient for 2s → 2p transition (s⁻¹)
const f_lu_Lyα::Float64    = 0.41641               # Oscillator strength for Lyα
const σT::Float64          = 6.6524587321e-25      # Thomson scattering cross-section (cm²)
const σHminus_Lyα::Float64 = 5.477e-18             # Lyα H⁻ photodetachment cross-section (cm²) (McLaughling et al. 2017)
const σOH_Lyα::Float64     = 4.07e-18              # Lyα OH photodissociation cross-section (cm²) (Heays et al. 2017)
const σCH_Lyα::Float64     = 0.05e-18              # Lyα CH photodissociation cross-section (cm²) (Heays et al. 2017)
const σRaman::Float64      = 2.1e-25               # Lyα H₂ Raman scattering cross-section for Lyα (see Nebrin et al. 2025 & references therein)  

####### PARAMETERS ########################
const Y::Float64          = 0.247                          # Helium mass fraction (of pristine gas)
const D_H_ratio::Float64  = 2.53e-5                        # Deuterium-to-hydrogen abundance nD/nH        (Cooke et al. 2018)
const X_D::Float64        = D_H_ratio * (1 - Y) * mD / mH  # Deuterium mass fraction (of pristine gas)
const X::Float64          = 1 - Y - X_D                    # Hydrogen mass fraction (of pristine gas), corrected slightly for deuterium 

const Zsun::Float64       = 0.0142                         # Solar metallicity M_Z / (M_H + M_He + M_Z)   (Asplund et al. 2009)
const MZ_over_MH::Float64 = 0.0199                         # M_Z / M_H                                    (Asplund et al. 2009)
const fCarbon::Float64    = 3.54e-3 / MZ_over_MH           # Fraction of metal mass in carbon @ Solar abundance ratios     (Asplund et al. 2009)
const fNitrogen::Float64  = 1.04e-3 / MZ_over_MH           # Fraction of metal mass in nitrogen @ Solar abundance ratios   (Asplund et al. 2009)
const fOxygen::Float64    = 8.59e-3 / MZ_over_MH           # Fraction of metal mass in oxygen @ Solar abundance ratios     (Asplund et al. 2009)
const fNeon::Float64      = 1.88e-3 / MZ_over_MH           # Fraction of metal mass in neon @ Solar abundance ratios       (Asplund et al. 2009)
const fMagnesium::Float64 = 1.06e-3 / MZ_over_MH           # Fraction of metal mass in magnesium @ Solar abundance ratios  (Asplund et al. 2009)
const fSilicon::Float64   = 9.07e-4 / MZ_over_MH           # Fraction of metal mass in silicon @ Solar abundance ratios    (Asplund et al. 2009) 
const fSulfur::Float64    = 4.63e-4 / MZ_over_MH           # Fraction of metal mass in sulfur @ Solar abundance ratios     (Asplund et al. 2009)
const fIron::Float64      = 1.94e-3 / MZ_over_MH           # Fraction of metal mass in iron @ Solar abundance ratios       (Asplund et al. 2009)

# Fraction of non-C/Si metals in the form of oxygen, needed 
# to recover the oxygen density from Zgasrest_Zsun (the ratio between 
# metals in Zgasrest_Zsun is assumed fixed): 
const fO_over_frest::Float64 = fOxygen / (1 - fCarbon - fSilicon)
 

const DGsun::Float64      = 1 / 162                        # Solar dust-to-gas ratio
const TCMB0::Float64      = 2.726                          # CMB temperature at redshift z = 0

####### CONVENIENT CONSTANTS ##############
const OneThird::Float64          = 1 / 3
const TwoThirds::Float64         = 2 / 3
const ThreeEights::Float64       = 3 / 8
const ThreeFourths::Float64      = 3 / 4 
const Four_π::Float64            = 4 * π 
const inv_Four_π::Float64        = 1 / Four_π
const Two_π::Float64             = 2 * π
const Sqrt_π::Float64            = sqrt(π) 
const Sqrt_2::Float64            = sqrt(2)
const Sqrt_π_over_2::Float64     = Sqrt_π / 2
const inv_Sqrt_π_over_2::Float64 = 1 / Sqrt_π_over_2
const Four_π_over_c::Float64     = Four_π / c    
const inv_c::Float64             = 1 / c   
const X_over_mH::Float64         = X / mH    
const Y_over_mHe::Float64        = Y / mHe   
const X_D_over_mD::Float64       = X_D / mD
const inv_Zsun::Float64          = 1 / Zsun


####### BAND ENERGIES ####################

# Photon band energies:
const E_opt::Float64      = 3.67  * eV       # 1-5.8 eV band (optical)
const E_FUV::Float64      = 8.45  * eV       # 5.8-11.2 eV band (FUV)
const E_LW::Float64       = 12.26 * eV       # 11.2-13.6 eV band (Lyman-Werner)
const E_EUV1::Float64     = 18.01 * eV       # 13.6-24.6 eV band (H I ionizing)
const E_EUV2::Float64     = 29.89 * eV       # 24.6-54.4 eV band (He I ionizing)
const E_EUV3::Float64     = 56.85 * eV       # 54.4-∞ eV band (He II ionizing) 

# Inverse energies enter a lot of expressions, so pre-compute:
const inv_E_opt::Float64  = 1 / E_opt
const inv_E_FUV::Float64  = 1 / E_FUV
const inv_E_LW::Float64   = 1 / E_LW
const inv_E_EUV1::Float64 = 1 / E_EUV1
const inv_E_EUV2::Float64 = 1 / E_EUV2
const inv_E_EUV3::Float64 = 1 / E_EUV3

# Export constants:
export c, h, G, kB, e, q_e, q_HII, q_HeII, q_HeIII,
       σSB, νLyα, νLyC, ELyα, ELyC, inv_ELyα, 
       ALyα, A_2γ, A_2p2s, A_2s2p, f_lu_Lyα, 
       σT, σHminus_Lyα, σOH_Lyα, σCH_Lyα, σRaman,
       X, Y, X_D, Zsun, DGsun, TCMB0, inv_Zsun,

       fCarbon, fNitrogen, fOxygen, fNeon, fMagnesium, fSilicon, fSulfur, fIron, 
       mH, mH2, mD, me, mHe, mC, mO, mCO, mSi,
       inv_mH2, inv_mD, inv_mN, inv_mNe, inv_mMg, inv_mSi, inv_mS, inv_mFe,

       Lsun, Msun, AU, pc, kpc, Mpc, Gpc, kms, inv_kms, day, yr, Myr, Gyr, eV, Ang, nm, μm, F_Habing,
       OneThird, TwoThirds, ThreeEights, ThreeFourths, Four_π, inv_Four_π, Two_π, Sqrt_π, Sqrt_2,
       Sqrt_π_over_2, inv_Sqrt_π_over_2, Four_π_over_c, inv_c, X_over_mH, Y_over_mHe, 

       E_opt, E_FUV, E_LW, E_EUV1, E_EUV2, E_EUV3,
       inv_E_opt, inv_E_FUV, inv_E_LW, inv_E_EUV1, inv_E_EUV2, inv_E_EUV3



###############################################################
#######                                                 #######
#######         U S E F U L  F U N C T I O N S          #######
#######                                                 #######
###############################################################

# Create look-up tables (LUTs) for expensive functions that 
# are needed for the Voigt profile:
const VoigtLUT = CreateVoigtLUTs()

""" The Voigt profile. Arguments:
    x:  Frequency displacement in units of Doppler shift.
    a:  The Lyα Voigt parameter, a ≃ 4.7 × 10⁻⁴ T₄^(-0.5)

    This function uses the expansion Eq. (12) in:
    Smith et al. (2018), 'Discrete diffusion Lyman α radiative transfer', MNRAS, 479, 2. """
@inline function H( x::Float64,
                    a::Float64
                   )

    # Get expensive functions from look-up tables or asymptotic limits when needed:
    EXP_xsq  = GaussianFast( x, VoigtLUT )            # exp(-x²) 
    Dawson   = DawsonFast( x, VoigtLUT )              # dawson(x)
    Stuff    = OneMinusx2GaussianFast( x, VoigtLUT )  # (1 - 2x²) exp(-x²)

    # Put it all together:
    H        = EXP_xsq  +  inv_Sqrt_π_over_2 * a * (2.0 * x * Dawson - 1)  +  (a * a) * Stuff
    return H
end


""" Integral ∫dx H(x,a) from x = 0 to x. 
    Uses Eq. (13) from:
    
    Smith et al. (2018), 'Discrete diffusion Lyman α radiative transfer', MNRAS, 479, 2. """
@inline function ∫dx_H( x::Float64, 
                        a::Float64
                       )

    # Get expensive functions from look-up tables or asymptotic limits when needed:
    Erf      = ErfFast( x, VoigtLUT )        # erf(x)
    Dawson   = DawsonFast( x, VoigtLUT )     # dawson(x)
    Stuff    = xGaussianFast( x, VoigtLUT )  # x exp(-x²)

    return Sqrt_π_over_2 * Erf  -  inv_Sqrt_π_over_2 * a * Dawson  +  (a * a) * Stuff
end


const Δx_crit::Float64 = 1e-5

""" Top-hat x-average of H(x,a), between xmax and xmin around frequency bin x, 
    for a Voigt parameter a. See Eq. (14) in:

    Smith et al. (2018), 'Discrete diffusion Lyman α radiative transfer', MNRAS, 479, 2.  """
@inline function Htophat( x::Float64,
                       xmax::Float64,
                       xmin::Float64,
                          a::Float64
                         )
    Δx = xmax - xmin # x_{j+1/2} - x_{j-1/2}
    if Δx > Δx_crit
        Htophat = ( ∫dx_H(xmax, a) - ∫dx_H(xmin, a) ) / Δx
        return Htophat
    else
        # For very small Δx, use H(x,a) instead
        Htophat = H(x, a)
        return Htophat
    end
end


const b_PREFACT::Float64 = sqrt( 2 * kB / mH )

""" The thermal velocity b(T) (cm/s) as a function 
    of temperature T (K). """
@inline @fastmath function Compute_b( T::Float64 )
    b_PREFACT * sqrt( T )
end


const νLyα_over_c::Float64 = νLyα / c

""" The Doppler width as a function of temperature T (K). """
@inline @fastmath function Compute_ΔνD( T::Float64 )
    Compute_b(T) * νLyα_over_c
end


const a_PREFACT::Float64 = ALyα / (Four_π * νLyα * 1/c)

""" Lyα Voigt parameter (dimensionless) as a function
    of temperature T (K). """
@inline @fastmath function Compute_a( T::Float64 )
    a_PREFACT / Compute_b(T)
end


const σ0_PREFACT::Float64 = Sqrt_π * (e^2) * f_lu_Lyα / ( me * νLyα )

""" Lyα cross-section at line center (cm²) as a function
    of temperature T (K). """
@inline @fastmath function Compute_σ0( T::Float64 )
    σ0_PREFACT / Compute_b(T)
end


""" Compute & update the (top-hat average) Voigt profile Hvoigt[i,j,k]
    for each cell. """
function Update_Hvoigt!( Hvoigt::AbstractArray{Float64, 3}, # Top-hat average Voigt profile for each cell (UPDATED)
                             δν::AbstractArray{Float64},    # Frequency interfaces displacement from line center 
                            δνC::AbstractArray{Float64},    # Frequency bin centers displacement from line center
                              a::AbstractArray{Float64, 2}, # Voigt parameters for each cell 
                       inv_Δν_D::AbstractArray{Float64, 2}  # Inverse Doppler width for each cell 
                         )

    N_r, N_Z, N_ν = size( Hvoigt ) # Grid size 
    dN_r = 1:N_r
    dN_Z = 1:N_Z
    dN_ν = 1:N_ν 

    @threads for k in dN_ν
        @inbounds begin 
            δν_C   = δνC[k]
            δν_max = δν[k+1]
            δν_min = δν[k]
            
            for j in dN_Z
                @simd for i in dN_r
                    # Inverse Doppler width:
                    inv_ΔνD = inv_Δν_D[i,j]

                    # Bin-averaged Voigt profile:
                    Hvoigt[i,j,k] = Htophat( δν_C   * inv_ΔνD, # x 
                                             δν_max * inv_ΔνD, # xmax 
                                             δν_min * inv_ΔνD, # xmin
                                             a[i,j] )
                                            
                end # End of i-loop
            end # End of j-loop

        end # End of inbounds 
    end # End of k-loop

    return nothing 
end


""" Compute & update the normalized emission profile ϕ_ν for each cell, 
    assuming ϕ_ν = Hvoigt / ∫dν Hvoigt ≈ Hvoigt[i,j,k] / ∑_k Δν[k] Hvoigt[i,j,k]. """
function Update_ϕν!( ϕ_ν::AbstractArray{Float64, 3},  # Normalized emission line profile (UPDATED)
                      Δν::AbstractArray{Float64, 1},  # Frequency bin widths
                  Hvoigt::AbstractArray{Float64, 3},  # Voigt profile for each cell
                    kmin::Int, 
                    kmax::Int 
                   )
    N_r, N_Z, _ = size( Hvoigt )
    dN_r = 1:N_r
    dN_Z = 1:N_Z
    dk   = kmin:kmax 

    @threads for j in dN_Z
        @inbounds begin 
            for i in dN_r

                SUM = 0.0
                @simd for k in dk 
                    SUM += Δν[k] * Hvoigt[i,j,k]
                end
                inv_SUM = 1 / SUM
                
                @simd for k in dk 
                    ϕ_ν[i,j,k] = Hvoigt[i,j,k] * inv_SUM
                end

            end # End of i-loop
        end # End of inbounds
    end # End of j-loop

    return nothing 
end


""" This function computes the Lyα energy density, for a given cell/position. """
@inline function Compute_e_Lyα( Δν::AbstractArray{Float64},
                                 J::AbstractArray{Float64, 1} )

    
    N_ν = length( Δν )

    ∫dν_J = 0.0 # Initialize the estimated integral
    @inbounds for k in 1:N_ν
        ∫dν_J += Δν[k] * J[k]
    end

    # Get Lyα energy density:
    e_Lyα = Four_π_over_c * ∫dν_J

    return e_Lyα
end


""" Update the Lyα energy density e_Lyα(r,Z) = 4π ∫dν J(r,Z,ν) / c. """
@inline function Update_e_Lyα!( e_Lyα::AbstractArray{Float64, 2}, 
                                   Δν::AbstractArray{Float64},
                                    J::AbstractArray{Float64, 3}
                                )

    # Grid size & intervals:
    N_r, N_Z, N_ν = size( J )
    dN_r          = 1:N_r 
    dN_Z          = 1:N_Z 
    dN_ν          = 1:N_ν

    @threads for j in dN_Z 
        @inbounds for i in dN_r 

            ∫dν_J = 0.0 # Initialize the estimated integral
            @simd for k in dN_ν
                ∫dν_J += Δν[k] * J[i,j,k]
            end

            # Get Lyα energy density:
            e_Lyα[i,j] = Four_π_over_c * ∫dν_J

        end # End of i-loop 
    end # End of j-loop 

    return nothing 
end


""" A single emission + absorption update for J, over a time-step Δt,
    using backward Euler. """
@inline function AbsorptionUpdateJorH( Jold::Float64,
                                      Δtsub::Float64,
                                       dJdt::Float64,
                                         αc::Float64,
                                          c̃::Float64 
                                      )

    # Backward Euler update:
    Jnew =  ( Jold  +  dJdt * Δtsub ) / ( 1  +  αc * c̃ * Δtsub )

    return Jnew 
end


""" This function updates J and H as a result of Lyα emission,
    absorption of J (including by pd > 0), and absorption of H
    (i.e. flux absorption):

    (1/c̃)(∂J/∂t) = (∫dν js) ϕν  -  [αc_Lyα + α_Lyα * Hvoigt * pd] J
    
    (1/c̃)(∂H/∂t) = - α_F_Lyα * H
    
    It also computes the corresponding 
    change in gas momentum density from Lyα radiation pressure:

    Δρu = (4π/c) ∫dν ∫dt α_F * H 
        
        = (4π/c) (1/c̃) ∫dν (H_old - H)

    where we have used (1/c̃)(∂H/∂t) = - α_F H (for the absorption step).
    
    NOTE: This function also updates the old values of H. """
function LyαEmissionAbsorptionUpdate!( J_Lyα::AbstractArray{Float64, 3},
                                   J_Lyα_old::AbstractArray{Float64, 3},
                                     H_r_Lyα::AbstractArray{Float64, 3}, 
                                     H_Z_Lyα::AbstractArray{Float64, 3},
                                 H_r_old_Lyα::AbstractArray{Float64, 3},
                                 H_Z_old_Lyα::AbstractArray{Float64, 3},
                                  ∫dν_js_Lyα::AbstractArray{Float64, 2},
                                         ϕ_ν::AbstractArray{Float64, 3},
                                      Hvoigt::AbstractArray{Float64, 3},
                                       α_Lyα::AbstractArray{Float64, 2},
                                      αc_Lyα::AbstractArray{Float64, 2},
                                     α_F_Lyα::AbstractArray{Float64, 3},
                                          pd::AbstractArray{Float64, 2},
                                   Δρu_Lyα_r::AbstractArray{Float64, 2},
                                   Δρu_Lyα_Z::AbstractArray{Float64, 2},
                                          Δν::AbstractArray{Float64},
                                           c̃::Float64,
                                          Δt::Float64 )

    # Grid size & intervals:
    N_r, N_Z, N_ν  = size( J_Lyα )   # Grid size 
    dN_r           = 1:N_r           # r-interval 
    dN_Z           = 1:N_Z           # Z-interval
    dN_ν           = 1:N_ν           # ν-interval

    @threads for k in dN_ν
        @inbounds @simd for j in dN_Z 
            for i in dN_r 
                # Absorption opacity for J:
                αabs_J = αc_Lyα[i,j]  +  α_Lyα[i,j] * Hvoigt[i,j,k] * pd[i,j]

                # Change in J from emission:
                dJdt   = ∫dν_js_Lyα[i,j] * ϕ_ν[i,j,k] * c̃

                # Update J:
                J_Lyα[i,j,k] = AbsorptionUpdateJorH( J_Lyα_old[i,j,k], Δt, dJdt, αabs_J, c̃ )

                # Store & set old (un-updated) H components:
                H_r_old_Lyα[i,j,k] = H_r_Lyα[i,j,k] 
                H_Z_old_Lyα[i,j,k] = H_Z_Lyα[i,j,k]

                # Flux opacity for H_r and H_Z:
                αabs_H = α_F_Lyα[i,j,k]

                # Candidate updates for H_r and H_Z:
                H_r_Lyα[i,j,k]  = AbsorptionUpdateJorH( H_r_old_Lyα[i,j,k], Δt, 0.0, αabs_H, c̃ )
                H_Z_Lyα[i,j,k]  = AbsorptionUpdateJorH( H_Z_old_Lyα[i,j,k], Δt, 0.0, αabs_H, c̃ )

            end # End of i-loop 
        end # End of j-loop 
    end # End of k-loop 


    # Compute change in gas momentum from Lyα radiation pressure: 
    inv_c̃  = 1 / c̃      # Convenient constant 
    @threads for j in dN_Z 
        @inbounds @simd for i in dN_r 

            # Compute ∫dν (H_old - H):
            ∫dν_ΔH_r = 0.0
            ∫dν_ΔH_Z = 0.0
            for k in dN_ν
                ∫dν_ΔH_r += Δν[k] * ( H_r_old_Lyα[i,j, k] - H_r_Lyα[i,j, k] )
                ∫dν_ΔH_Z += Δν[k] * ( H_Z_old_Lyα[i,j, k] - H_Z_Lyα[i,j, k] )
            end

            # Change in gas momentum from Lyα radiation pressure:
            Δρu_Lyα_r[i,j] += Four_π_over_c * inv_c̃ * ∫dν_ΔH_r
            Δρu_Lyα_Z[i,j] += Four_π_over_c * inv_c̃ * ∫dν_ΔH_Z
        end
    end

    # Update old J and H components a final time:
    J_Lyα_old   .= J_Lyα 
    H_r_old_Lyα .= H_r_Lyα 
    H_Z_old_Lyα .= H_Z_Lyα

    return nothing 
end


""" This function updates the Lyα radiation pressure force per unit volume components 
    f_Lyα = (f_Lyα_r, f_Lyα_Z) from the accumulated change in momentum over time-step Δt. 
    
    The function also computes the Lyα force multiplier M_F of various kinds:
        
    M_F            = ∫dV |f_Lyα| / (L_Lyα/c)
    M_F_radial_net = ∫dV ( f_Lyα ⋅ r̂_out ) / (L_Lyα/c) 
    M_F_radial_out = ∫dV max( f_Lyα ⋅ r̂_out, 0.0 ) / (L_Lyα/c) 
    
    The two M_F_radial_net/out measure the force directed radially from 
    the star or cloud center, with the "out" version only including the 
    outward-directed components. """
@inline function Update_f_Lyα_MF_L_Lyα!( f_Lyα_r::AbstractArray{Float64, 2}, 
                                         f_Lyα_Z::AbstractArray{Float64, 2},
                                       Δρu_Lyα_r::AbstractArray{Float64, 2},
                                       Δρu_Lyα_Z::AbstractArray{Float64, 2},
                                          ∫dν_js::AbstractArray{Float64, 2},
                                              ΔV::AbstractArray{Float64, 2},
                                            rCoM::AbstractArray{Float64, 1},
                                            ZCoM::AbstractArray{Float64, 1},
                                              Δt::Float64,
                                        Z_center::Float64,
                                        BC_Z_bot::Symbol = :Reflect
                                        )

    # Grid size & intervals:
    N_r, N_Z  = size( f_Lyα_r )  # Grid size 
    dN_r      = 1:N_r            # r-interval 
    dN_Z      = 1:N_Z            # Z-interval

    # Save radiation pressure force per unit volume: 
    inv_Δt = 1 / Δt     # Convenient constant 
    @threads for j in dN_Z 
        @inbounds @simd for i in dN_r 
            f_Lyα_r[i,j]  = Δρu_Lyα_r[i,j] * inv_Δt
            f_Lyα_Z[i,j]  = Δρu_Lyα_Z[i,j] * inv_Δt
        end
    end

    # Compute force multiplier M_F = ∫dV |f_Lyα| / (L_Lyα/c) & Lyα luminosity L_Lyα = 4π ∫dV (∫dν js):
    M_F   = 0.0
    L_Lyα = 0.0
    for j in dN_Z
        @inbounds @simd for i in dN_r 
            M_F   += ΔV[i,j] * sqrt( f_Lyα_r[i,j] * f_Lyα_r[i,j]  +  f_Lyα_Z[i,j] * f_Lyα_Z[i,j] )
            L_Lyα += Four_π * ΔV[i,j] * ∫dν_js[i,j]
        end
    end

    # Compute radial force multipliers relative to star/cloud center:
    M_F_radial_net = 0.0 
    M_F_radial_out = 0.0 
    for j in dN_Z 
        @inbounds @simd for i in dN_r
            # Distance from star/cloud center:
            Δr       = rCoM[i]
            ΔZ       = ZCoM[j] - Z_center
            distance = @fastmath sqrt( Δr * Δr  +  ΔZ * ΔZ )

            if distance > 0.0
                inv_d   = @fastmath 1.0 / distance
                e_out_r = Δr * inv_d
                e_out_Z = ΔZ * inv_d

                f_Lyα_radial = f_Lyα_r[i,j] * e_out_r  +  f_Lyα_Z[i,j] * e_out_Z
            else
                # At the stellar position the outward direction is undefined:
                f_Lyα_radial = 0.0
            end

            # Add contributions to M_F_radial_net and M_F_radial_out:
            M_F_radial_net += ΔV[i,j] * f_Lyα_radial
            M_F_radial_out += ΔV[i,j] * max( f_Lyα_radial, 0.0 )
        end
    end

    # Normalize force multipliers:
    if L_Lyα > 0.0
        inv_L_over_c   = 1.0 / (L_Lyα / c)
        M_F            = M_F * inv_L_over_c
        M_F_radial_net = M_F_radial_net * inv_L_over_c
        M_F_radial_out = M_F_radial_out * inv_L_over_c
    else
        M_F            = 0.0
        M_F_radial_net = 0.0
        M_F_radial_out = 0.0
    end

    # Compute the total Lyα luminosity in the cloud, *taking into account reflective BCs*:
    if BC_Z_bot === :Reflect 
        # If we have reflective BCs on the lower Z-plane, we double to get the total Lyα luminosity 
        # in the physical cloud:
        L_Lyα_tot = 2 * L_Lyα
    else
        L_Lyα_tot = L_Lyα
    end

    return M_F, M_F_radial_net, M_F_radial_out, L_Lyα_tot
end


""" This function updates the Lyα radiation pressure force per unit volume components 
    f_Lyα = (f_Lyα_r, f_Lyα_Z) from the accumulated change in momentum over time-step Δt. 
    
    The function also computes the Lyα force multiplier M_F of various kinds:
        
    M_F             = ∫dV |f_Lyα| / (L_Lyα/c)
    M_F_antigravity = ∫dV ( - f_Lyα ⋅ â_grav ) / (L_Lyα/c)
    M_F_radial_net  = ∫dV ( f_Lyα ⋅ r̂_out ) / (L_Lyα/c) 
    M_F_radial_out  = ∫dV max( f_Lyα ⋅ r̂_out, 0.0 ) / (L_Lyα/c) 
    
    M_F_antigravity is the force multiplier *anti-parallel* to gravity, which is a 
    better measure of its feedback strength. Related, the two M_F_radial_net/out measure 
    the force directed radially from the star or cloud center, with the "out" version 
    only including the outward-directed components. """
@inline function Update_f_Lyα_MF_L_Lyα!( f_Lyα_r::AbstractArray{Float64, 2}, 
                                         f_Lyα_Z::AbstractArray{Float64, 2},
                                       Δρu_Lyα_r::AbstractArray{Float64, 2},
                                       Δρu_Lyα_Z::AbstractArray{Float64, 2},
                                          ∫dν_js::AbstractArray{Float64, 2},
                                              ΔV::AbstractArray{Float64, 2},
                                            rCoM::AbstractArray{Float64, 1},
                                            ZCoM::AbstractArray{Float64, 1},
                                              Δt::Float64,
                                        f_grav_r::AbstractArray{Float64, 2}, 
                                        f_grav_Z::AbstractArray{Float64, 2}, 
                                        Z_center::Float64,
                                        BC_Z_bot::Symbol = :Reflect
                                        )

    # Grid size & intervals:
    N_r, N_Z  = size( f_Lyα_r )  # Grid size 
    dN_r      = 1:N_r            # r-interval 
    dN_Z      = 1:N_Z            # Z-interval

    # Save radiation pressure force per unit volume: 
    inv_Δt = 1 / Δt     # Convenient constant 
    @threads for j in dN_Z 
        @inbounds @simd for i in dN_r 
            f_Lyα_r[i,j]  = Δρu_Lyα_r[i,j] * inv_Δt
            f_Lyα_Z[i,j]  = Δρu_Lyα_Z[i,j] * inv_Δt
        end
    end

    # Compute force multiplier M_F = ∫dV |f_Lyα| / (L_Lyα/c) & Lyα luminosity L_Lyα = 4π ∫dV (∫dν js):
    M_F   = 0.0
    L_Lyα = 0.0
    for j in dN_Z
        @inbounds @simd for i in dN_r 
            M_F   += ΔV[i,j] * sqrt( f_Lyα_r[i,j] * f_Lyα_r[i,j]  +  f_Lyα_Z[i,j] * f_Lyα_Z[i,j] )
            L_Lyα += Four_π * ΔV[i,j] * ∫dν_js[i,j]
        end
    end

    # Compute force multiplier anti-parallel to gravity, 
    # M_F_antigravity = ∫dV ( - f_Lyα ⋅ â_grav ) / (L_Lyα/c)
    # as well as radially outward force multipliers.
    # NOTE: Negative values for M_F_antigravity mean that radiation pressure is 'helping' gravity 

    M_F_antigravity = 0.0
    M_F_radial_net  = 0.0 
    M_F_radial_out  = 0.0 
    for j in dN_Z 
        @inbounds @simd for i in dN_r

            # ------ Compute force multiplier anti-parallel to gravity: --------

            # Magnitude of gravitatonal force:
            f_grav_mag = sqrt( f_grav_r[i,j] * f_grav_r[i,j]  +  f_grav_Z[i,j] * f_grav_Z[i,j] )

            if f_grav_mag > 0.0
                f_Lyα_antiparallel = - ( f_Lyα_r[i,j] * f_grav_r[i,j]  +  f_Lyα_Z[i,j] * f_grav_Z[i,j] ) / f_grav_mag
            else
                # In regions without gravity, just set the force component to zero: 
                f_Lyα_antiparallel = 0.0
            end

            # Get contribution to M_F_antigravity from this cell:
            M_F_antigravity += ΔV[i,j] * f_Lyα_antiparallel

            # ------ Compute radially outward force multipliers: ------------

            # Distance from star/cloud center:
            Δr       = rCoM[i]
            ΔZ       = ZCoM[j] - Z_center
            distance = @fastmath sqrt( Δr * Δr  +  ΔZ * ΔZ )

            if distance > 0.0
                inv_d   = @fastmath 1.0 / distance
                e_out_r = Δr * inv_d
                e_out_Z = ΔZ * inv_d

                f_Lyα_radial = f_Lyα_r[i,j] * e_out_r  +  f_Lyα_Z[i,j] * e_out_Z
            else
                # At the stellar position the outward direction is undefined:
                f_Lyα_radial = 0.0
            end

            # Add contributions to M_F_radial_net and M_F_radial_out:
            M_F_radial_net += ΔV[i,j] * f_Lyα_radial
            M_F_radial_out += ΔV[i,j] * max( f_Lyα_radial, 0.0 )
        end
    end

    # Normalize force multipliers:
    if L_Lyα > 0.0
        inv_L_over_c    = 1.0 / (L_Lyα / c)
        M_F             = M_F * inv_L_over_c
        M_F_antigravity = M_F_antigravity * inv_L_over_c
        M_F_radial_net  = M_F_radial_net * inv_L_over_c
        M_F_radial_out  = M_F_radial_out * inv_L_over_c
    else
        M_F             = 0.0
        M_F_antigravity = 0.0
        M_F_radial_net  = 0.0
        M_F_radial_out  = 0.0
    end

    # Compute the total Lyα luminosity in the cloud, *taking into account reflective BCs*:
    if BC_Z_bot === :Reflect 
        # If we have reflective BCs on the lower Z-plane, we double to get the total Lyα luminosity 
        # in the physical cloud:
        L_Lyα_tot = 2 * L_Lyα
    else
        L_Lyα_tot = L_Lyα
    end

    return M_F, M_F_antigravity, M_F_radial_net, M_F_radial_out, L_Lyα_tot
end


""" This function just resets Δρu (for Lyα or stellar/IR RT bands),
    i.e. sets it to zero. Just a convenient function. """
@inline function Reset_Δρu!( Δρu_r::AbstractArray{Float64, 2},
                             Δρu_Z::AbstractArray{Float64, 2}
                             )
    Δρu_r .= 0.0
    Δρu_Z .= 0.0

    return nothing 
end


""" A single absorption update for H ∈ (H_r, H_Z),
    over a time-step Δt, using backward Euler. """
@inline @fastmath function AbsorptionUpdateH( Hold::Float64, # Old value of H ∈ (H_r, H_Z)
                                                 α::Float64, # Opacity 
                                               c̃Δt::Float64  # Reduced speed of light × time-step 
                                            )

    # Backward Euler update:
    Hnew =  Hold / ( 1  +  α * c̃Δt )

    return Hnew 
end


""" This function updates the components of ∇u, using interface values of 
    u obtained from the hydro solver.  """
function Update_∇u_RHD!( ∂u_r_∂r::AbstractArray{Float64, 2},
                         ∂u_r_∂Z::AbstractArray{Float64, 2}, 
                         ∂u_Z_∂r::AbstractArray{Float64, 2},
                         ∂u_Z_∂Z::AbstractArray{Float64, 2},
                      u_r_over_r::AbstractArray{Float64, 2},
                         u_r_L_r::AbstractArray{Float64, 2},
                         u_r_R_r::AbstractArray{Float64, 2},
                         u_r_L_Z::AbstractArray{Float64, 2},
                         u_r_R_Z::AbstractArray{Float64, 2},
                         u_Z_L_r::AbstractArray{Float64, 2},
                         u_Z_R_r::AbstractArray{Float64, 2},
                         u_Z_L_Z::AbstractArray{Float64, 2},
                         u_Z_R_Z::AbstractArray{Float64, 2},
                               r::AbstractArray{Float64, 1},
                               Z::AbstractArray{Float64, 1},
                      inv_r_cell::AbstractArray{Float64, 1}
                          )

    # NOTE: Below, e.g. u_r_L_r[i,j] means the value of u_r on the left 
    # side of the interface i-1/2. And u_r_R_r[i,j] refers to the value of u_r
    # on the right side of the interface i-1/2. 
    
    N_r  = length( r ) - 1
    N_Z  = length( Z ) - 1
    dN_r = 1:N_r 
    dN_Z = 1:N_Z

    @threads for j in dN_Z 
        inv_ΔZ  = 1 / ( Z[j+1] - Z[j] )
        @inbounds @simd for i in dN_r
            inv_Δr          = 1 / ( r[i+1] - r[i] )

            ∂u_r_∂r[i,j]    = ( u_r_L_r[i+1,j] - u_r_R_r[i,j] ) * inv_Δr
            ∂u_r_∂Z[i,j]    = ( u_r_L_Z[i,j+1] - u_r_R_Z[i,j] ) * inv_ΔZ
            ∂u_Z_∂r[i,j]    = ( u_Z_L_r[i+1,j] - u_Z_R_r[i,j] ) * inv_Δr
            ∂u_Z_∂Z[i,j]    = ( u_Z_L_Z[i,j+1] - u_Z_R_Z[i,j] ) * inv_ΔZ

            # Also get ⟨u_r / r⟩:
            u_r_over_r[i,j] = 0.5 * ( u_r_L_r[i+1,j] + u_r_R_r[i,j] ) * inv_r_cell[i] 
        end
    end

    return nothing 
end


""" This function returns D:∇u in cylindrical coordinates (r,Z), assuming 
    axisymmetry (∂/∂ϕ = 0 & u_ϕ = 0). """
@inline @fastmath function D_colon_∇u_Cylindrical( D_rr::Float64, D_rZ::Float64,    D_ZZ::Float64,
                                                ∂u_r_∂r::Float64, ∂u_r_∂Z::Float64, ∂u_Z_∂r::Float64,
                                                ∂u_Z_∂Z::Float64, u_r_over_r::Float64 )

    # Handy relation from tr(D) = 1:
    D_ϕϕ = max( 1  -  D_rr  - D_ZZ, 0.0 )

    # Compute D:∇u (using D_Zr = D_rZ):
    D_colon_∇u = ( D_rr * ∂u_r_∂r  +  D_rZ * ( ∂u_r_∂Z  + ∂u_Z_∂r ) 
                 + D_ZZ * ∂u_Z_∂Z  +  D_ϕϕ * u_r_over_r )

    return D_colon_∇u
end


""" This function returns D:∇u in Cartesian coordinates (r,Z) → (X,Y). """
@inline @fastmath function D_colon_∇u_Cartesian( D_rr::Float64, D_rZ::Float64,    D_ZZ::Float64,
                                              ∂u_r_∂r::Float64, ∂u_r_∂Z::Float64, ∂u_Z_∂r::Float64,
                                              ∂u_Z_∂Z::Float64 )

    # Compute D:∇u (using D_Zr = D_rZ):
    D_colon_∇u = ( D_rr * ∂u_r_∂r  +  D_rZ * ( ∂u_r_∂Z  + ∂u_Z_∂r ) 
                 + D_ZZ * ∂u_Z_∂Z  )

    return D_colon_∇u
end


""" This function computes the escape fraction in a given frequency bin/band 
    for a simulation in cylindrical coordinates. Assumes that the total luminosity 
    in the simulation box is L_simvol.  """
@inline function Compute_f_esc_Cylindrical( H_r::AbstractArray{Float64, 2}, 
                                            H_Z::AbstractArray{Float64, 2},
                                            A_r::AbstractArray{Float64, 2},
                                            A_Z::AbstractArray{Float64, 2},
                                     L_allspace::Float64,
                                       BC_Z_bot::Symbol = :Reflect
                                          )

    N_r, N_Z = size( H_r )
    dN_r     = 1:N_r 
    dN_Z     = 1:N_Z
    L_esc    = 0.0 # Initialize luminosity leaked 

    if BC_Z_bot === :Reflect 
        # If we only simulate Z > 0 with reflective BCs, then the luminosity 
        # in simulated volume is half of that in all space 
        # (the true full luminosity of the source):
        L_simvol = 0.5 * L_allspace
    else
        L_simvol = L_allspace
    end

    # Contribution from leakage through outer-r boundary 
    @inbounds @simd for j in dN_Z 
        L_esc += A_r[N_r+1,j] * Four_π * max( H_r[N_r,j], 0.0 )
    end

    # Contribution from leakage through upper-Z boundary 
    @inbounds @simd for i in dN_r 
        L_esc += A_Z[i,N_Z+1] * Four_π * max( H_Z[i,N_Z], 0.0 )
    end

    if BC_Z_bot === :Outflow 
        # Leakage through lower-Z boundary if there are outflow BCs there:
        @inbounds @simd for i in dN_r 
            L_esc += A_Z[i,1] * Four_π * max( - H_Z[i,1], 0.0 )
        end
    end

    # Escape fraction:
    if L_simvol > 0.0
        f_esc = L_esc / L_simvol
    else 
        f_esc = 0.0
    end

    return f_esc 
end


""" The Lyα escape fraction in cylindrical coordinates. Computed as:

    f_esc_Lyα = ∫dν ∫dA ⋅ 4π H_Lyα / L_simvol 

    where L_Lyα_simvol is the Lyα luminosity (erg s⁻¹) in the simulated volume.  """
@views @inline function Compute_f_esc_Lyα_Cylindrical( H_r_Lya::AbstractArray{Float64, 3}, 
                                                       H_Z_Lya::AbstractArray{Float64, 3},
                                                           A_r::AbstractArray{Float64, 2},
                                                           A_Z::AbstractArray{Float64, 2},
                                                            Δν::AbstractArray{Float64, 1},
                                                L_Lyα_allspace::Float64,
                                                      BC_Z_bot::Symbol = :Reflect
                                                       )

    N_ν  = length( Δν ) # Number of frequency bins 
    dN_ν = 1:N_ν

    # If Lyα luminosity is zero, exit early by setting f_esc_Lyα = 0:
    if L_Lyα_allspace ≤ 0.0
        return 0.0
    end

    f_esc_Lyα = 0.0
    @inbounds @simd for k in dN_ν
        f_esc_Lyα += Δν[k] * Compute_f_esc_Cylindrical( H_r_Lya[:,:,k], H_Z_Lya[:,:,k], A_r, A_Z, 
                                                        L_Lyα_allspace, BC_Z_bot )
    end

    return f_esc_Lyα 
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
function J_sphere_point( x::Float64, 
                         T::Float64, 
                       τcl::Float64
                       )
    aτ = Compute_a(T)*τcl
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
function J_sphere_uni( x::Float64, 
                       T::Float64, 
                     τcl::Float64
                     )
    aτ = Compute_a(T)*τcl
    J  = - sqrt(6/π) * (x^2 / aτ) * log(1 - exp(-sqrt(2 * π^3 / 27) * abs.(x^3) / aτ))
    return J
end


# Below are the predicted Lyα pressure PLyα(r) for idealized setups, 
# which can be used to compare simulations to. We also define needed 
# series to evaluate PLyα(r).


""" The polylogarithmic function evaluated for a unit circle z = exp(iθ),
    Li_s( z = exp(iθ) ):
    
    Li_s(z) ≡ ∑_{n = 1 to ∞} z^n / n^s .
   
    Needed for some analytical expressions. 
"""
function Li_UnitCircle( θ::Float64, 
                        s::Float64,
                        K::Int64 = 20 )

    if θ == 0.0
        return zeta(s) + 0 * im
    end

    Sum = gamma(1-s) * exp( (s-1)*(log(θ) - im*π/2) )
    for k in 0:K 
        Sum += zeta(s-k) * (im*θ)^k / factorial(k)
    end

    return Sum
end



""" Predicted Lyα energy density e_Lyα for a uniform, static, dust-free 
    cloud with a central Lyα point source, as found by Lao & Smith (2020).
    
    **ARGUMENTS:**
    
    r         :   Array of r-values (cm) 
    L_Lyα     :   Lyα luminosity (erg/s)
    T         :   Cloud temperature (K)
    τcl       :   Cloud optical depth at Lyα line center 
    
    **RETURNS:**
    
    e_Lyα(r)  :   Array of e_Lyα for all r-values 
 """
function eLyα_sphere_point( r::AbstractVector{Float64}, 
                        L_Lyα::Float64, 
                            T::Float64, 
                          τcl::Float64
                          )
    Rcl  = r[end]                         
    rbar = r ./ Rcl
    aτ   = Compute_a(T)*τcl          

    # Put it all together to compute e_Lyα:
    stuff  = (L_Lyα/(c * Rcl^2)) * gamma(1/3) * (2 * aτ)^(1/3) / (2 * π^(3/2))

    e_Lyα  = (stuff ./ rbar) .* imag.( Li_UnitCircle.( π .* rbar, 1/3) )

    return e_Lyα
end


""" Predicted Lyα energy density e_Lyα for a uniform, static, dust-free 
    cloud with a uniform Lyα source, as found by Lao & Smith (2020).
    
    **ARGUMENTS:**
    
    r         :   Array of r-values (cm) 
    L_Lyα     :   Lyα luminosity (erg/s)
    T         :   Cloud temperature (K)
    τcl       :   Cloud optical depth at Lyα line center 
    
    **RETURNS:**
    
    e_Lyα(r)  :   Array of e_Lyα for all r-values 
 """
function eLyα_sphere_uni( r::AbstractVector{Float64}, 
                      L_Lyα::Float64, 
                          T::Float64,
                        τcl::Float64,
                        Rcl::Float64
                        )

    rbar = min.( r ./ Rcl, 1.0 )
    aτ   = Compute_a(T)*τcl          

    # Put it all together to compute e_Lyα:
    stuff = (3 * L_Lyα/(c * Rcl^2)) * gamma(1/3) * (2 * aτ)^(1/3) / (2 * π^(7/2))

    # The argument for the polylogarithmic function is z = exp( - i π rbar ),
    # or z = exp( i π (1 - rbar) ), and so θ =  π (1 - rbar). With this the 
    # Lyα energy density is:
    e_Lyα  = (stuff ./ rbar) .* imag.( Li_UnitCircle.( π .* (1 .- rbar), 7/3) )

    return e_Lyα
end


###############################################################
#######                                                 #######
#######            M I S C E L L A N E O U S            #######
#######                                                 #######
###############################################################


const vth_H_prefact::Float64 = sqrt( 8 * kB / (π * mH) )

""" The thermal velocity for hydrogen, sqrt( 8 kB T / π mH) """
function vth_H( T::Float64 )
    return vth_H_prefact * sqrt( T )
end

""" This function computes the H I, H II, and H₂ number densities,
    given the fractional abundances of H II and H₂, and the (primordial)
    gas density ρ. """
@inline @fastmath function GetHydrogenDensities( xHII::Float64, 
                                                  xH2::Float64,
                                                    ρ::Float64,
                                    HydrogenAndHelium::Bool = true )

    if HydrogenAndHelium
        # By default we have hydrogen, helium (and deuterium):
        nH = ρ * X_over_mH
    else
        # In some idealized tests we only have hydrogen:
        nH = ρ * inv_mH
    end

    xHI   =   max( 1 - xHII - 2 * xH2, 0.0 )
    nHI   =   xHI  * nH
    nHII  =   xHII * nH 
    nH2   =   xH2  * nH

    return nHI, nHII, nH2, nH
end


""" This function computes the He I, He II, and He III number densities,
    given the fractional abundances of He II and He III, and the (primordial)
    gas density ρ. """
@inline @fastmath function GetHeliumDensities( xHeII::Float64, 
                                              xHeIII::Float64,
                                                   ρ::Float64,
                                   HydrogenAndHelium::Bool = true )

    if HydrogenAndHelium
        # By default we have hydrogen, helium (and deuterium):
        nHe = ρ * Y_over_mHe
    else
        # In some idealized tests we only have hydrogen:
        nHe = 0.0
    end

    xHeI    =   max( 1 - xHeII - xHeIII, 0.0 )
    nHeI    =   xHeI   * nHe
    nHeII   =   xHeII  * nHe
    nHeIII  =   xHeIII * nHe

    return nHeI, nHeII, nHeIII, nHe
end


""" This function computes the D I, D II, and HD number densities,
    given the fractional abundances of HD and D II, and the (primordial)
    gas density ρ. """
@inline @fastmath function GetDeuteriumDensities( xHD::Float64, 
                                                 xDII::Float64,
                                                    ρ::Float64,
                                    HydrogenAndHelium::Bool = true )

    # Total deuterium number density:
    if HydrogenAndHelium
        # By default we have hydrogen, helium, and deuterium:
        nD = ρ * X_D_over_mD 
    else
        # In some idealized tests we only have hydrogen:
        nD = 0.0
    end

    # Get number densities 
    nDI   =   max( 1 - xDII  - xHD, 0.0 ) * nD
    nDII  =   xDII * nD
    nHD   =   xHD  * nD    

    return nDI, nDII, nHD, nD
end


""" This function computes the C I, C II, CO, and CHₓ number densities,
    given the fractional abundances of C I, C II, and CO, and the carbon
    gas density ρCarbon. """
@inline @fastmath function GetCarbonDensities( xCI::Float64, 
                                              xCII::Float64, 
                                               xCO::Float64, 
                                           ρCarbon::Float64 )

    # Get carbon number density:
    nC   = ρCarbon * inv_mC 

    # Get number densities of all species:
    nCI  = xCI  * nC 
    nCII = xCII * nC 
    nCO  = xCO  * nC
    nCHx = max( 1 - xCI - xCII - xCO, 0.0 ) * nC 

    return nCI, nCII, nCO, nCHx, nC
end


""" This function computes the O I, O II, and OHₓ number densities,
    given the fractional abundances of O I, O II, and the number density 
    of CO, and the oxygen density ρOxygen. """
@inline @fastmath function GetOxygenDensities( xOI::Float64, 
                                              xOII::Float64, 
                                               nCO::Float64, 
                                           ρOxygen::Float64 )

    # Get oxygen number density:
    nO   = ρOxygen * inv_mO 

    # Get number densities of all species:
    nOI  = xOI  * nO 
    nOII = xOII * nO 
    nOHx = clamp( nO - nOI - nOII - nCO, 0.0, nO )

    return nOI, nOII, nOHx, nO
end


""" This function computes the electron number density 
    from the number densities of H II, He II, and He III. """
@inline @fastmath function GetElectronDensity( nHII::Float64, 
                                              nHeII::Float64, 
                                             nHeIII::Float64,
                                               nDII::Float64,
                                               nCII::Float64,
                                               nOII::Float64
                                              )

    return nHII  +  nHeII  +  2 * nHeIII  +  nDII  +  nCII  +  nOII 
end


const fCarbon_over_mC::Float64 = fCarbon * inv_mC 
const fOxygen_over_mO::Float64 = fOxygen * inv_mO

""" This function computes the electron number density 
    from the abundance fractions of H II, He II, He III,
    D II, C II, and O II. """
@inline @fastmath function GetElectronDensityFromFracs( xHII::Float64, 
                                                       xHeII::Float64, 
                                                      xHeIII::Float64,
                                                        xDII::Float64,
                                                        xCII::Float64,
                                                        xOII::Float64,
                                                           ρ::Float64,
                                                          ZC::Float64,  # Carbon gas metallicity 
                                                        ZSil::Float64,  # Si gas metallicity 
                                                    Zgasrest::Float64,  # Non-C/Si gas metallicity 
                                           HydrogenAndHelium::Bool = true 
                                                        )

    # Total gas-phase metallicity:
    Zgas       = ZC  +  ZSil  +  Zgasrest
    Z          = clamp( Zgas, 0.0, 1.0 )

    # Get pristine (H + He + D) gas density:
    ρ_pristine = max( (1 - Z), 0.0 ) * ρ

    if HydrogenAndHelium
        # By default we have hydrogen, helium (and deuterium):
        nH  = ρ_pristine * X_over_mH
        nHe = ρ_pristine * Y_over_mHe
        nD  = ρ_pristine * X_D_over_mD
    else
        # In some idealized tests we only have hydrogen:
        nH  = ρ_pristine * inv_mH
        nHe = 0.0
        nD  = 0.0
    end

    # Get nHII:
    nHII       = xHII * nH

    # Get nHeII and nHeIII:
    nHeII      = xHeII  * nHe 
    nHeIII     = xHeIII * nHe 

    # Get nDII:
    nDII       = xDII * nD 

    # Get nCII:
    nC         = ZC * ρ * inv_mC 
    nCII       = xCII * nC 

    # Get nOII:
    nO         = fO_over_frest * Zgasrest * ρ * inv_mO 
    nOII       = xOII * nO

    # Get electron density:
    ne         = GetElectronDensity( nHII, nHeII, nHeIII, nDII, nCII, nOII )

    return ne
end


""" Computes the total gas number density. """
@inline function GetTotalNumberDensity( nHI::Float64,   nHII::Float64,    nH2::Float64, 
                                        nHeI::Float64, nHeII::Float64, nHeIII::Float64,
                                        nDI::Float64,   nDII::Float64,    nHD::Float64, 
                                        nCI::Float64,   nCII::Float64,    nCO::Float64,
                                        nCHx::Float64,   nOI::Float64,   nOII::Float64,
                                        nOHx::Float64,    ne::Float64 )

    n  = ( nHI + nHII + nH2 + nHeI + nHeII + nHeIII + nDI + nDII + nHD
         + nCI + nCII + nCO + nCHx + nOI + nOII + nOHx + ne )

    return n
end


""" This function recovers all gas number densities. """
function GetAllNumberDensities( xH2::Float64,
                               xHII::Float64, 
                              xHeII::Float64, 
                             xHeIII::Float64,
                                xHD::Float64,
                               xDII::Float64,
                                xCI::Float64,
                               xCII::Float64,
                                xCO::Float64,
                                xOI::Float64,
                               xOII::Float64,
                                  ρ::Float64,
                                 ZC::Float64,  # Carbon gas metallicity 
                               ZSil::Float64,  # Silicon gas metallicity  
                           Zgasrest::Float64,  # Non-C/Si gas metallicity 
                  HydrogenAndHelium::Bool = true )

    # Total gas-phase metallicity: 
    Zgas  = ZC  +  ZSil  +  Zgasrest

    # Get pristine (H + He + D) gas density:
    Z          = clamp( Zgas, 0.0, 1.0 )
    ρ_pristine = max( (1 - Z), 0.0 ) * ρ

    # Get hydrogen densities:
    nHI, nHII, nH2, nH        = GetHydrogenDensities( xHII, xH2, ρ_pristine, HydrogenAndHelium )

    # Get helium densities:
    nHeI, nHeII, nHeIII, nHe  = GetHeliumDensities( xHeII, xHeIII, ρ_pristine, HydrogenAndHelium )

    # Get deuterium densities:
    nDI, nDII, nHD, nD        = GetDeuteriumDensities( xHD, xDII, ρ_pristine, HydrogenAndHelium )

    # Get carbon species densities:
    ρCarbon                   = @fastmath ZC * ρ
    nCI, nCII, nCO, nCHx, nC  = GetCarbonDensities( xCI, xCII, xCO, ρCarbon )

    # Get oxygen species densities:
    ρOxygen                   = @fastmath fO_over_frest * Zgasrest * ρ
    nOI, nOII, nOHx, nO       = GetOxygenDensities( xOI, xOII, nCO, ρOxygen )

    # Get electron density:
    ne                        = GetElectronDensity( nHII, nHeII, nHeIII, nDII, nCII, nOII )

    # Get total number density (that would e.g. enter the pressure P = n kB T):
    n  = ( nHI + nHII + nH2 + nHeI + nHeII + nHeIII + nDI + nDII + nHD
         + nCI + nCII + nCO + nCHx + nOI + nOII + nOHx + ne )

    return nHI, nHII, nH2, nHeI, nHeII, nHeIII, nDI, nDII, nHD, 
           nCI, nCII, nCO, nCHx, nOI, nOII, nOHx, ne, n, nH, nHe, nD, nC, nO
end


""" This is a small useful function for relating ρ to 
    the hydrogen number density nH. Only used for initialization 
    in simulations. """
@inline function Get_ρ_From_nH( nH::Float64, 
                         Zgas_Zsun::Float64, 
                 HydrogenAndHelium::Bool )

    if HydrogenAndHelium
        # In case we model hydrogen, helium, deuterium & metals, 
        # we recover the gas density as follows.

        # Pristine (H, He, D) mass density:
        ρ_pristine = nH * mH / X 

        # Use (1 - Z)ρ = ρ_pristine:
        Z = clamp( Zgas_Zsun * Zsun, 0.0, 1.0 )
        ρ = ρ_pristine / ( 1 - Z )

        return ρ
    else
        # In case we only model hydrogen (and metals), we recover 
        # the gas density as follows.

        # Hydrogen density (for X = 1):
        ρ_H = nH * mH 

        # Use (1 - Z)ρ = ρ_H:
        Z   = clamp( Zgas_Zsun * Zsun, 0.0, 1.0 )
        ρ   = ρ_H / ( 1 - Z )

        return ρ
    end
end


const mO_over_mC::Float64 = mO / mC

""" This function computes nC / nO as:

    nC / nO = (ZC × ρ / mC) / (fO_over_frest × Zgasrest × ρ / mO)
            = [ZC / (fO_over_frest × Zgasrest)] × (mO/mC) 
            
    This is used in both the C/O chemistry, and when advecting the gas abundances 
    as passive scalars. """
@inline function Compute_nC_over_nO( ZC::Float64,
                               Zgasrest::Float64 )

    if Zgasrest == 0.0
        # If there is no other metals than carbon - therefore no oxygen -
        # just set nC_over_nO to zero to avoid division by zero when used 
        # elsewhere:
        return 0.0
    end

    nC_over_nO = ( ZC / ( fO_over_frest * Zgasrest ) ) * mO_over_mC

    return nC_over_nO 
end


""" MinMod slope limiter. Used in the hydrodynamics, M1 RT, 
    and in the estimate for ∇ρ for the Sobolev-like 
    column density estimates. """
@inline function MinMod( A::Float64,
                         B::Float64 
                        )

    if A * B ≤ 0.0
        return 0.0
    else
        # If abs(A) < abs(B) return A, otherwise 
        # return B:
        return ifelse( abs(A) < abs(B), A, B )
    end
end


""" The free-fall time, useful for setting up some simulation tests. """
@inline function FreeFallTime( nH::Float64, Zgas_Zsun )

    # Get the gas density from nH:
    ρ = Get_ρ_From_nH( nH, Zgas_Zsun, true )

    # Free-fall time: 
    tff = sqrt( 3 * π / (32 * G * ρ) )

    return tff 
end


""" This function  computes the spherically averaged profile ρ(R), for 
    some array (e.g. density) ρ(r,Z) in cylindrical coordinates (r,Z). 
    It does the spherical averaging centered on Z = Zc. This function 
    can be used to compute spherically averaged profiles of various 
    quantities, e.g. densities, abundance fractions,
    or the Lyα energy density. """
function SphericalAverageProfile!( ρ_av::AbstractArray{Float64, 1},    # Spherically averaged ρ
                                      ρ::AbstractArray{Float64, 2},    # Array ρ(r,Z)
                             R_binedges::AbstractArray{Float64, 1},    # Spherical R-bin edges 
                                 V_bins::AbstractArray{Float64, 1},    # Bin volumes 
                                     ΔV::AbstractArray{Float64, 2},    # Cell volumes 
                                   rCoM::AbstractArray{Float64, 1},    # Cell CoM in r-direction 
                                   ZCoM::AbstractArray{Float64, 1},    # Cell CoM in Z-direction 
                                     Zc::Float64 = 0.0 )


    # Intervals:
    N_r, N_Z = size( ΔV )
    Nbins    = length( R_binedges ) - 1
    dN_Z     = 1:N_Z 
    dN_r     = 1:N_r  
    dNbin    = 1:Nbins
    
    # Initialize ρ_av and V_bins to zero before we update them:
    ρ_av   .= 0.0
    V_bins .= 0.0

    @inbounds for j in dN_Z 
        for i in dN_r 
            # Get the cell distance from center:
            R = sqrt( rCoM[i]^2 + (ZCoM[j] - Zc)^2 )

            # Find nearest bin:
            for k in dNbin
                if R_binedges[k] ≤ R && R < R_binedges[k+1] 
                    # In this case we have found the right bin, so add the contribution 
                    # from ρ[i,j] to this bin:
                    ρ_av[k]   += ρ[i,j] * ΔV[i,j]

                    # Add the contribution to the bin volume:
                    V_bins[k] += ΔV[i,j]

                    # Exit the loop (we have found the bin):
                    break 
                end
            end

        end # End of i-loop 
    end # End of j-loop 

    # Normalize to get the final profile:
    @inbounds for k in dNbin
        if V_bins[k] > 0.0
            ρ_av[k] = ρ_av[k] / V_bins[k]
        else
            ρ_av[k] = 0.0
        end
    end

    return nothing
end


""" This function computes the number of ionizing photons in each ionizing 
    band (EUV1, EUV2, EUV3), assuming a black-body spectrum of temperature 
    Teff (K), and a total ionizing output dNdt_LyC. Useful for idealized tests. """
function GetBlackBodyIonizingEmission( dNdt_LyC::Float64, Teff::Float64 )

    # Convenient variable:
    inv_kBT = 1 / ( kB * Teff )

    # Photon weight, ∝ Bν / ν ∝ x²/[exp(x) - 1], where x ≡ hν / kB T:
    ϕ  = ( x::Float64 ) -> (x^2) / expm1( x )

    # Minimum x of each band:
    x0 = ( 13.6 * eV ) * inv_kBT
    x1 = ( 24.6 * eV ) * inv_kBT
    x2 = ( 54.4 * eV ) * inv_kBT

    # Total ionizing photons (LyC): [13.6 eV, ∞):
    I_tot, _ = quadgk( ϕ, x0, Inf, rtol = 1e-10 )

    # The ionizing photons in each band (EUV1, EUV2, EUV3):
    I1, _ = quadgk( ϕ, x0, x1,  rtol = 1e-10 )
    I2, _ = quadgk( ϕ, x1, x2,  rtol = 1e-10 )
    I3, _ = quadgk( ϕ, x2, Inf, rtol = 1e-10 )

    @assert isfinite(I_tot) && I_tot > 0
    @assert abs( ( I1 + I2 + I3 ) / I_tot - 1 ) < 1e-10

    # Normalize and multiply by the total ionziing output to 
    # get emission rates for each band:
    dNdt_EUV1 = dNdt_LyC * ( I1 / I_tot )
    dNdt_EUV2 = dNdt_LyC * ( I2 / I_tot )
    dNdt_EUV3 = dNdt_LyC * ( I3 / I_tot )

    return dNdt_EUV1, dNdt_EUV2, dNdt_EUV3
end


""" Prints LYDION logo. """
function PrintLYDIONLogo( )
    # Define some ANSI codes:
    b   = "\x1b[1;36m"         # Cyan blue 
    r   = "\x1b[1;31m"         # Red 
    g   = "\x1b[38;2;0;102;0m" # Green 
    db  = "\x1b[38;2;0;0;255m" # Blue
    p   = "\x1b[1;35m"         # Purple 

    LYDION_logo = """

    $(b)ooooo            ooooo  oooo      $(r)ooooooooo        ooooo        $(p)ooooooo        oooo   oooo 
    $(b)888               888  88         $(r)888    88o       888       $(p)o888   888o       8888o  88  
    $(b)888                 888           $(r)888    888       888       $(p)888     888       88 888o88  
    $(b)888      o          888           $(r)888    888       888       $(p)888o   o888       88   8888  
   $(b)o888ooooo88         o888o         $(r)o888ooo88        o888o        $(p)88ooo88        o88o    88

    """

    println(LYDION_logo)
    println("")
end


end # End of module