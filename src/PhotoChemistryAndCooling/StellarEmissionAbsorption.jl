

""" FILE:       StellarEmissionAbsorption.jl

    PURPOSE:    Contains everything related to stellar emission, 
                i.e. photon emission rates and luminosities in different bands
                for stars of different mass. The band energies can be found in 
                ConstantsAndFunctions.jl.

    REFERENCES: The emission rates and mean stellar photon energies are taken
                from the following references, assuming Solar metallicity:

                (1) Deng et al. (2024) 'RIGEL: Simulating dwarf galaxies at solar mass resolution with
                radiative transfer and feedback from individual massive stars', A&A, 691, A231
                (see related data link: https://bitbucket.org/ftldengyw/rigel-tables/src/main/)

                (2) Kannan et al. (2020) 'Simulating the interstellar medium of galaxies with 
                radiative transfer, non-equilibrium thermochemistry, and dust', MNRAS, 499, 5732-5748

                (3) Baczynski et al. (2015) 'Fervent: chemistry-coupled, ionizing and non-ionizing radiative 
                feedback in hydrodynamical simulations', MNRAS, 454, 380-411

                (4) Ploeckinger et al. (2025) 'Hybrid-chimes: A model for radiative cooling and the abundances of ions
                and molecules in simulations of galaxy formation', submitted to MNRAS (arXiv:2506.15773)

"""


using .ConstantsAndFunctions  # Needed constants & functions
using DelimitedFiles          # For reading data from files

# Include file containing metal photoionization cross-sections:
include(joinpath(@__DIR__, "MetalPhotoionizationCrossSectionAverages.jl"))

# Export functions & default band struct:
export dNdt_opt, L_opt, dNdt_FUV, L_FUV, dNdt_LW, L_LW,
       dNdt_EUV1, L_EUV1, dNdt_EUV2, L_EUV2, dNdt_EUV3, L_EUV3, L_bol,
       ClusterEmissionRates,
       HeatingRate_FUV, HeatingRate_LW, HeatingRate_EUV1, HeatingRate_EUV2, HeatingRate_EUV3, 
       HeatingRate_Compton,
       IonRate_HI, IonRate_H2, IonRate_HeI, IonRate_HeII, 
       IonRate_CI, IonRate_CII, IonRate_OI, IonRate_OII, IonRate_OIII, 
       IonRate_SI, IonRate_SII, IonRate_SIII,
       PhotoDissRate_H2, PhotoDetachRate_Hminus, PhotoDissRate_HD, PhotoDissRate_CO,
       PhotoDissRate_OHₓ, PhotoDissRate_CHₓ, StromGrenRadius,
       DefaultStellarBandAverages, GetBBStellarAverages

# Average stellar radiation pressure-weighted photon energy (erg): 
# p = ( ∫dν 4πJσ ) / ( ∫dν 4πJσ/hν ) ,
# needed to compute radiation pressure from gas absorption (not on dust) - see ref. 2:

const pH2_LW     = 12.26 * eV       # From Lyman-Werner absorption (11.2-13.6 eV)

const pH2_EUV1   = 18.98 * eV       # From absorption of H I ionizing photons (13.6-24.6 eV)
const pHI_EUV1   = 16.75 * eV

const pH2_EUV2   = 28.35 * eV       # From absorption of He I ionizing photons (24.6-54.4 eV)
const pHI_EUV2   = 28.30 * eV
const pHeI_EUV2  = 28.41 * eV

const pH2_EUV3   = 56.38 * eV       # From absorption of He II ionizing photons (54.4-∞ eV)
const pHI_EUV3   = 56.38 * eV
const pHeI_EUV3  = 56.40 * eV
const pHeII_EUV3 = 56.43 * eV

export pH2_LW, pH2_EUV1, pHI_EUV1, pH2_EUV2, pHI_EUV2, pHeI_EUV2,
       pH2_EUV3, pHI_EUV3, pHeI_EUV3, pHeII_EUV3 

# Cross-sections (cm²) in each band from different species
# (see ref. 2 for many of them):

const σHminus_opt  = 20e-18
const σCH_opt      = 0.368e-18

const σHminus_FUV  = 7.7e-18
const σOH_FUV      = 2.03e-18
const σCH_FUV      = 3.81e-18

const σHI_LW       = 4.68e-24                   # Absorption of LW photons by Lyman lines in H I (not included in ref. 2)
const σH2_LW_diss  = 2.47e-18                   # Figure taken from Baczynski et al. (2015), which is a factor ~ 10 larger than in ref. 2
const σH2_LW_pump  = 6.94 * σH2_LW_diss         # UV pumping of H₂, as given in Baczynski et al. (2015)
const σH2_LW       = σH2_LW_diss + σH2_LW_pump  # Total LW absorption cross-section for H₂, from photodissociation + UV pumping 
const σHD_LW       = (1.55/1.39) * σH2_LW_diss 
const σHminus_LW   = 7.3e-18
const σOH_LW       = 4.02e-18
const σCH_LW       = 1.18e-18
const σCO_LW       = 17.3e-18 

const σHI_EUV1     = 3.37e-18
const σH2_EUV1     = 5.09e-18 
const σHminus_EUV1 = 3.2e-18

const σHI_EUV2     = 0.79e-18
const σH2_EUV2     = 2.42e-18
const σHeI_EUV2    = 5.10e-18
const σHminus_EUV2 = 0.71e-18

const σHI_EUV3     = 0.11e-18
const σH2_EUV3     = 0.32e-18
const σHeI_EUV3    = 0.77e-18
const σHeII_EUV3   = 1.42e-18
const σHminus_EUV3 = 0.092e-18

# Radiation pressure/flux/energy-weighted cross-sections, which differ slightly from photon 
# number-weighted cross-sections:
const σHI_EUV1_E   = σHI_EUV1 * pHI_EUV1 / E_EUV1
const σH2_EUV1_E   = σH2_EUV1 * pH2_EUV1 / E_EUV1

const σHI_EUV2_E   = σHI_EUV2  * pHI_EUV2 / E_EUV2 
const σH2_EUV2_E   = σH2_EUV2  * pH2_EUV2 / E_EUV2 
const σHeI_EUV2_E  = σHeI_EUV2 * pHeI_EUV2 / E_EUV2 

const σHI_EUV3_E   = σHI_EUV3   * pHI_EUV3   / E_EUV3 
const σH2_EUV3_E   = σH2_EUV3   * pH2_EUV3   / E_EUV3 
const σHeI_EUV3_E  = σHeI_EUV3  * pHeI_EUV3  / E_EUV3 
const σHeII_EUV3_E = σHeII_EUV3 * pHeII_EUV3 / E_EUV3

export σHminus_opt, σCH_opt,
       σHminus_FUV, σOH_FUV, σCH_FUV,
       σHI_LW, σH2_LW, σHD_LW, σHminus_LW, σOH_LW, σCH_LW, σCO_LW,
       σHI_EUV1, σH2_EUV1, σHminus_EUV1, 
       σHI_EUV2, σH2_EUV2, σHeI_EUV2, σHminus_EUV2,
       σHI_EUV3, σH2_EUV3, σHeI_EUV3, σHeII_EUV3, σHminus_EUV3, 
       σHI_EUV1_E, σH2_EUV1_E,  σHI_EUV2_E, σH2_EUV2_E, σHeI_EUV2_E,  
       σHI_EUV3_E, σH2_EUV3_E, σHeI_EUV3_E, σHeII_EUV3_E

# Mean photo-heating energy (erg) in each band:

const hH2_LW     = 0.40  * eV   # From LW photodissociation of H₂. See ref. 3 

const hHI_EUV1   = 3.15  * eV   # See ref. 2
const hH2_EUV1   = 3.78  * eV   # See ref. 2

const hHI_EUV2   = 13.10 * eV   # See ref. 2
const hH2_EUV2   = 14.75 * eV   # See ref. 2
const hHeI_EUV2  = 3.81  * eV   # See ref. 2

const hHI_EUV3   = 42.78 * eV   # See ref. 2
const hH2_EUV3   = 41.18 * eV   # See ref. 2
const hHeI_EUV3  = 31.80 * eV   # See ref. 2
const hHeII_EUV3 = 2.03  * eV   # See ref. 2

export hHI_EUV1, hH2_EUV1, hHI_EUV2, hH2_EUV2, hHeI_EUV2,
       hHI_EUV3, hH2_EUV3, hHeI_EUV3, hHeII_EUV3

# Make a struct for the default H & He cross-sections & heating energies:
const DefaultStellarBandAverages = StellarBandAverages(
    E_opt, E_FUV, E_LW, E_EUV1, E_EUV2, E_EUV3,
    inv_E_opt, inv_E_FUV, inv_E_LW, inv_E_EUV1, inv_E_EUV2, inv_E_EUV3,

    σHI_EUV1,  σHI_EUV2,  σHI_EUV3,
    σHeI_EUV2, σHeI_EUV3, σHeII_EUV3,

    σHI_EUV1_E,  σHI_EUV2_E,  σHI_EUV3_E,
    σHeI_EUV2_E, σHeI_EUV3_E, σHeII_EUV3_E,

    hHI_EUV1,  hHI_EUV2,  hHI_EUV3,
    hHeI_EUV2, hHeI_EUV3, hHeII_EUV3
)


# Below are photon emission rates and luminosities in the
# different bands, as given in ref. (1). The bolometric luminosity
# is also given at the end. First we read in the fitting coefficients from 
# ref. (1):

OptData   = readdlm( joinpath(@__DIR__, "RIGELfits/Opt.txt") ) 
FUVData   = readdlm( joinpath(@__DIR__, "RIGELfits/FUV.txt") ) 
LWData    = readdlm( joinpath(@__DIR__, "RIGELfits/LW.txt") ) 
EUV1Data  = readdlm( joinpath(@__DIR__, "RIGELfits/EUV1.txt") ) 
EUV2Data  = readdlm( joinpath(@__DIR__, "RIGELfits/EUV2.txt") ) 
EUV3Data  = readdlm( joinpath(@__DIR__, "RIGELfits/EUV3.txt") ) 


""" The optical photon output dN_opt/dt (s⁻¹) from a star of ZAMS 
    mass mstarZAMS (M⊙) and metallicity Zstar_Zsun (in Solar units). """
function dNdt_opt( mstarZAMS::Float64,
                  Zstar_Zsun::Float64 = 1.0 
                   )
    x = log10( mstarZAMS )
    # Limits on metallicity used in the fit:
    Zstar_Zsun = clamp( Zstar_Zsun, 1e-8, 1.0 )
    # Metallicity bin centers:
    Zbins   = Float64.( OptData[2, 1:6] )
    # Find where the input Zstar_Zsun lies:
    indxlow = findlast(Z -> Z ≤ Zstar_Zsun, Zbins)
    indxup  = findfirst(Z -> Z ≥ Zstar_Zsun, Zbins)
    Zlow    = Zbins[indxlow]
    Zup     = Zbins[indxup]
    # Get log10(x0) for the lower & upper limits:
    x0low   = log10( OptData[3,indxlow] )
    x0up    = log10( OptData[3,indxup] )

    # Compute log10(A) for lower & upper limits:
    b0low   = OptData[3+indxlow,1]
    b1low   = OptData[3+indxlow,2] 
    b2low   = OptData[3+indxlow,3]
    b3low   = OptData[3+indxlow,4]
    b4low   = OptData[3+indxlow,5]

    b0up    = OptData[3+indxup,1]
    b1up    = OptData[3+indxup,2] 
    b2up    = OptData[3+indxup,3]
    b3up    = OptData[3+indxup,4]
    b4up    = OptData[3+indxup,5]

    # Copute log10(A) for lower & upper limits
    if x ≤ x0low
        log10_Alow = b0low + b1low*x + b2low*(x^2) + b3low*(x^3) + b4low*(x^4)
    else
        log10_Alow = ( b0low + b1low*x0low + b2low*(x0low^2) + b3low*(x0low^3) + b4low*(x0low^4)  
                   +  (b1low + 2*b2low*x0low + 3*b3low*(x0low^2) + 4*b4low*(x0low^3))*(x - x0low) )
    end

    if Zlow == Zup 
        # In this case we don't need interpolation, and are finished:
        return 10^log10_Alow
    end

    if x ≤ x0up
        log10_Aup = b0up + b1up*x + b2up*(x^2) + b3up*(x^3) + b4up*(x^4)
    else
        log10_Aup = ( b0up + b1up*x0up + b2up*(x0up^2) + b3up*(x0up^3) + b4up*(x0up^4) 
                  +  (b1up + 2*b2up*x0up + 3*b3up*(x0up^2) + 4*b4up*(x0up^3))*(x - x0up) )
    end

    # Use linear interpolation between them:
    log10_A = log10_Alow + ((log10_Aup - log10_Alow) / log10(Zup/Zlow)) * log10(Zstar_Zsun/Zlow)
    
    return 10^log10_A 
end

""" The optical luminosity L_opt (L⊙) of a star of ZAMS mass mstarZAMS (M⊙)
    and metallicity Zstar_Zsun (in Solar units). """
function L_opt( mstarZAMS::Float64, 
               Zstar_Zsun::Float64 = 1.0,
                        A::StellarBandAverages = DefaultStellarBandAverages 
               )
    dNdt_opt(mstarZAMS, Zstar_Zsun) * A.E_opt / Lsun
end


""" The FUV photon output dN_FUV/dt (s⁻¹) from a star of ZAMS 
    mass mstarZAMS (M⊙) and metallicity Zstar_Zsun (in Solar units). """
function dNdt_FUV( mstarZAMS::Float64,
                  Zstar_Zsun::Float64 = 1.0 
                   )
    x = log10( mstarZAMS )
    # Limits on metallicity used in the fit:
    Zstar_Zsun = clamp( Zstar_Zsun, 1e-8, 1.0 )
    # Metallicity bin centers:
    Zbins   = Float64.( FUVData[2, 1:6] )
    # Find where the input Zstar_Zsun lies:
    indxlow = findlast(Z -> Z ≤ Zstar_Zsun, Zbins)
    indxup  = findfirst(Z -> Z ≥ Zstar_Zsun, Zbins)
    Zlow    = Zbins[indxlow]
    Zup     = Zbins[indxup]
    # Get log10(x0) for the lower & upper limits:
    x0low   = log10( FUVData[3,indxlow] )
    x0up    = log10( FUVData[3,indxup] )

    # Compute log10(A) for lower & upper limits:
    b0low   = FUVData[3+indxlow,1]
    b1low   = FUVData[3+indxlow,2] 
    b2low   = FUVData[3+indxlow,3]
    b3low   = FUVData[3+indxlow,4]
    b4low   = FUVData[3+indxlow,5]

    b0up    = FUVData[3+indxup,1]
    b1up    = FUVData[3+indxup,2] 
    b2up    = FUVData[3+indxup,3]
    b3up    = FUVData[3+indxup,4]
    b4up    = FUVData[3+indxup,5]

    # Copute log10(A) for lower & upper limits
    if x ≤ x0low
        log10_Alow = b0low + b1low*x + b2low*(x^2) + b3low*(x^3) + b4low*(x^4)
    else
        log10_Alow = ( b0low + b1low*x0low + b2low*(x0low^2) + b3low*(x0low^3) + b4low*(x0low^4)  
                   +  (b1low + 2*b2low*x0low + 3*b3low*(x0low^2) + 4*b4low*(x0low^3))*(x - x0low) )
    end

    if Zlow == Zup 
        # In this case we don't need interpolation, and are finished:
        return 10^log10_Alow
    end

    if x ≤ x0up
        log10_Aup = b0up + b1up*x + b2up*(x^2) + b3up*(x^3) + b4up*(x^4)
    else
        log10_Aup = ( b0up + b1up*x0up + b2up*(x0up^2) + b3up*(x0up^3) + b4up*(x0up^4) 
                  +  (b1up + 2*b2up*x0up + 3*b3up*(x0up^2) + 4*b4up*(x0up^3))*(x - x0up) )
    end

    # Use linear interpolation between them:
    log10_A = log10_Alow + ((log10_Aup - log10_Alow) / log10(Zup/Zlow)) * log10(Zstar_Zsun/Zlow)
    
    return 10^log10_A 
end

""" The far-UV luminosity L_FUV (L⊙) of a star of ZAMS mass mstarZAMS (M⊙)
    and metallicity Zstar_Zsun (in Solar units). """
function L_FUV( mstarZAMS::Float64,
               Zstar_Zsun::Float64 = 1.0,
                        A::StellarBandAverages = DefaultStellarBandAverages 
                )
    dNdt_FUV(mstarZAMS, Zstar_Zsun) * A.E_FUV / Lsun
end


""" The Lyman-Werner photon output dN_LW/dt (s⁻¹) from a star of ZAMS 
    mass mstarZAMS (M⊙) and metallicity Zstar_Zsun (in Solar units). """
function dNdt_LW( mstarZAMS::Float64,
                 Zstar_Zsun::Float64 = 1.0 
                   )
    x = log10( mstarZAMS )
    # Limits on metallicity used in the fit:
    Zstar_Zsun = clamp( Zstar_Zsun, 1e-8, 0.007042253521126761 )
    # Metallicity bin centers:
    Zbins   = Float64.( LWData[2, 1:3] )
    # Find where the input Zstar_Zsun lies:
    indxlow = findlast(Z -> Z ≤ Zstar_Zsun, Zbins)
    indxup  = findfirst(Z -> Z ≥ Zstar_Zsun, Zbins)
    Zlow    = Zbins[indxlow]
    Zup     = Zbins[indxup]
    # Get log10(x0) for the lower & upper limits:
    x0low   = log10( LWData[3,indxlow] )
    x0up    = log10( LWData[3,indxup] )

    # Compute log10(A) for lower & upper limits:
    b0low   = LWData[3+indxlow,1]
    b1low   = LWData[3+indxlow,2] 
    b2low   = LWData[3+indxlow,3]
    b3low   = LWData[3+indxlow,4]
    b4low   = LWData[3+indxlow,5]

    b0up    = LWData[3+indxup,1]
    b1up    = LWData[3+indxup,2] 
    b2up    = LWData[3+indxup,3]
    b3up    = LWData[3+indxup,4]
    b4up    = LWData[3+indxup,5]

    # Copute log10(A) for lower & upper limits
    if x ≤ x0low
        log10_Alow = b0low + b1low*x + b2low*(x^2) + b3low*(x^3) + b4low*(x^4)
    else
        log10_Alow = ( b0low + b1low*x0low + b2low*(x0low^2) + b3low*(x0low^3) + b4low*(x0low^4)  
                   +  (b1low + 2*b2low*x0low + 3*b3low*(x0low^2) + 4*b4low*(x0low^3))*(x - x0low) )
    end

    if Zlow == Zup 
        # In this case we don't need interpolation, and are finished:
        return 10^log10_Alow
    end

    if x ≤ x0up
        log10_Aup = b0up + b1up*x + b2up*(x^2) + b3up*(x^3) + b4up*(x^4) 
    else
        log10_Aup = ( b0up + b1up*x0up + b2up*(x0up^2) + b3up*(x0up^3) + b4up*(x0up^4)
                  +  (b1up + 2*b2up*x0up + 3*b3up*(x0up^2) + 4*b4up*(x0up^3))*(x - x0up) )
    end

    # Use linear interpolation between them:
    log10_A = log10_Alow + ((log10_Aup - log10_Alow) / log10(Zup/Zlow)) * log10(Zstar_Zsun/Zlow)
    
    return 10^log10_A 
end

""" The Lyman-Werner luminosity L_LW (L⊙) of a star of ZAMS mass mstarZAMS (M⊙)
    and metallicity Zstar_Zsun (in Solar units). """
function L_LW( mstarZAMS::Float64, 
              Zstar_Zsun::Float64 = 1.0,
                       A::StellarBandAverages = DefaultStellarBandAverages 
              )
    dNdt_LW(mstarZAMS, Zstar_Zsun) * A.E_LW / Lsun
end


""" The H I ionizing photon output dN_opt/dt (s⁻¹) from a star of ZAMS 
    mass mstarZAMS (M⊙) and metallicity Zstar_Zsun (in Solar units). """
function dNdt_EUV1( mstarZAMS::Float64,
                   Zstar_Zsun::Float64 = 1.0 
                   )
    x = log10( mstarZAMS )
    # Limits on metallicity used in the fit:
    Zstar_Zsun = clamp( Zstar_Zsun, 1e-8, 1.0 )
    # Metallicity bin centers:
    Zbins   = Float64.( EUV1Data[2, 1:4] )
    # Find where the input Zstar_Zsun lies:
    indxlow = findlast(Z -> Z ≤ Zstar_Zsun, Zbins)
    indxup  = findfirst(Z -> Z ≥ Zstar_Zsun, Zbins)
    Zlow    = Zbins[indxlow]
    Zup     = Zbins[indxup]
    # Get log10(x0) for the lower & upper limits:
    x0low   = log10( EUV1Data[3,indxlow] )
    x0up    = log10( EUV1Data[3,indxup] )

    # Compute log10(A) for lower & upper limits:
    b0low   = EUV1Data[3+indxlow,1]
    b1low   = EUV1Data[3+indxlow,2] 
    b2low   = EUV1Data[3+indxlow,3]
    b3low   = EUV1Data[3+indxlow,4]
    b4low   = EUV1Data[3+indxlow,5]

    b0up    = EUV1Data[3+indxup,1]
    b1up    = EUV1Data[3+indxup,2] 
    b2up    = EUV1Data[3+indxup,3]
    b3up    = EUV1Data[3+indxup,4]
    b4up    = EUV1Data[3+indxup,5]

    # Copute log10(A) for lower & upper limits
    if x ≤ x0low
        log10_Alow = b0low + b1low*x + b2low*(x^2) + b3low*(x^3) + b4low*(x^4)
    else
        log10_Alow = ( b0low + b1low*x0low + b2low*(x0low^2) + b3low*(x0low^3) + b4low*(x0low^4)  
                   +  (b1low + 2*b2low*x0low + 3*b3low*(x0low^2) + 4*b4low*(x0low^3))*(x - x0low) )
    end

    if Zlow == Zup 
        # In this case we don't need interpolation, and are finished:
        return 10^log10_Alow
    end

    if x ≤ x0up
        log10_Aup = b0up + b1up*x + b2up*(x^2) + b3up*(x^3) + b4up*(x^4)
    else
        log10_Aup = ( b0up + b1up*x0up + b2up*(x0up^2) + b3up*(x0up^3) + b4up*(x0up^4)  
                  +  (b1up + 2*b2up*x0up + 3*b3up*(x0up^2) + 4*b4up*(x0up^3))*(x - x0up) )
    end

    # Use linear interpolation between them:
    log10_A = log10_Alow + ((log10_Aup - log10_Alow) / log10(Zup/Zlow)) * log10(Zstar_Zsun/Zlow)
    
    return 10^log10_A 
end

""" The H I ionizing luminosity L_EUV1 (L⊙) of a star of ZAMS mass mstarZAMS (M⊙)
    and metallicity Zstar_Zsun (in Solar units). """
function L_EUV1( mstarZAMS::Float64, 
                Zstar_Zsun::Float64 = 1.0,
                         A::StellarBandAverages = DefaultStellarBandAverages
                 )
    dNdt_EUV1(mstarZAMS, Zstar_Zsun) * A.E_EUV1 / Lsun
end


""" The He I ionizing photon output dN_EUV2/dt (s⁻¹) from a star of ZAMS 
    mass mstarZAMS (M⊙) and metallicity Zstar_Zsun (in Solar units). """
function dNdt_EUV2( mstarZAMS::Float64,
                   Zstar_Zsun::Float64 = 1.0
                   )
    x = log10( mstarZAMS )
    # Limits on metallicity used in the fit:
    Zstar_Zsun = clamp( Zstar_Zsun, 1e-8, 1.0 )
    # Metallicity bin centers:
    Zbins   = Float64.( EUV2Data[2, 1:4] )
    # Find where the input Zstar_Zsun lies:
    indxlow = findlast(Z -> Z ≤ Zstar_Zsun, Zbins)
    indxup  = findfirst(Z -> Z ≥ Zstar_Zsun, Zbins)
    Zlow    = Zbins[indxlow]
    Zup     = Zbins[indxup]
    # Get log10(x0) for the lower & upper limits:
    x0low   = log10( EUV2Data[3,indxlow] )
    x0up    = log10( EUV2Data[3,indxup] )

    # Compute log10(A) for lower & upper limits:
    b0low   = EUV2Data[3+indxlow,1]
    b1low   = EUV2Data[3+indxlow,2] 
    b2low   = EUV2Data[3+indxlow,3]
    b3low   = EUV2Data[3+indxlow,4]
    b4low   = EUV2Data[3+indxlow,5]

    b0up    = EUV2Data[3+indxup,1]
    b1up    = EUV2Data[3+indxup,2] 
    b2up    = EUV2Data[3+indxup,3]
    b3up    = EUV2Data[3+indxup,4]
    b4up    = EUV2Data[3+indxup,5]

    # Copute log10(A) for lower & upper limits
    if x ≤ x0low
        log10_Alow = b0low + b1low*x + b2low*(x^2) + b3low*(x^3) + b4low*(x^4)
    else
        log10_Alow = ( b0low + b1low*x0low + b2low*(x0low^2) + b3low*(x0low^3) + b4low*(x0low^4)  
                   +  (b1low + 2*b2low*x0low + 3*b3low*(x0low^2) + 4*b4low*(x0low^3))*(x - x0low) )
    end

    if Zlow == Zup 
        # In this case we don't need interpolation, and are finished:
        return 10^log10_Alow
    end

    if x ≤ x0up
        log10_Aup = b0up + b1up*x + b2up*(x^2) + b3up*(x^3) + b4up*(x^4)
    else
        log10_Aup = ( b0up + b1up*x0up + b2up*(x0up^2) + b3up*(x0up^3) + b4up*(x0up^4)  
                  +  (b1up + 2*b2up*x0up + 3*b3up*(x0up^2) + 4*b4up*(x0up^3))*(x - x0up) )
    end

    # Use linear interpolation between them:
    log10_A = log10_Alow + ((log10_Aup - log10_Alow) / log10(Zup/Zlow)) * log10(Zstar_Zsun/Zlow)
    
    return 10^log10_A 
end

""" The He I ionizing luminosity L_EUV2 (L⊙) of a star of ZAMS mass mstarZAMS (M⊙)
    and metallicity Zstar_Zsun (in Solar units). """
function L_EUV2( mstarZAMS::Float64,
                Zstar_Zsun::Float64 = 1.0,
                         A::StellarBandAverages = DefaultStellarBandAverages
                 )
    dNdt_EUV2(mstarZAMS, Zstar_Zsun) * A.E_EUV2 / Lsun
end


""" The optical photon output dN_opt/dt (s⁻¹) from a star of ZAMS 
    mass mstarZAMS (M⊙) and metallicity Zstar_Zsun (in Solar units). """
function dNdt_EUV3( mstarZAMS::Float64,
                   Zstar_Zsun::Float64 = 1.0 
                   )
    x = log10( mstarZAMS )
    # Limits on metallicity used in the fit:
    Zstar_Zsun = clamp( Zstar_Zsun, 1e-8, 1.0 )
    # Metallicity bin centers:
    Zbins   = Float64.( EUV3Data[2, 1:6] )
    # Find where the input Zstar_Zsun lies:
    indxlow = findlast(Z -> Z ≤ Zstar_Zsun, Zbins)
    indxup  = findfirst(Z -> Z ≥ Zstar_Zsun, Zbins)
    Zlow    = Zbins[indxlow]
    Zup     = Zbins[indxup]
    # Get log10(x0) for the lower & upper limits:
    x0low   = log10( EUV3Data[3,indxlow] )
    x0up    = log10( EUV3Data[3,indxup] )

    # Compute log10(A) for lower & upper limits:
    b0low   = EUV3Data[3+indxlow,1]
    b1low   = EUV3Data[3+indxlow,2] 
    b2low   = EUV3Data[3+indxlow,3]
    b3low   = EUV3Data[3+indxlow,4]
    b4low   = EUV3Data[3+indxlow,5]

    b0up    = EUV3Data[3+indxup,1]
    b1up    = EUV3Data[3+indxup,2] 
    b2up    = EUV3Data[3+indxup,3]
    b3up    = EUV3Data[3+indxup,4]
    b4up    = EUV3Data[3+indxup,5]

    # Copute log10(A) for lower & upper limits
    if x ≤ x0low
        log10_Alow = b0low + b1low*x + b2low*(x^2) + b3low*(x^3) + b4low*(x^4)
    else
        log10_Alow = ( b0low + b1low*x0low + b2low*(x0low^2) + b3low*(x0low^3) + b4low*(x0low^4)  
                   +  (b1low + 2*b2low*x0low + 3*b3low*(x0low^2) + 4*b4low*(x0low^3))*(x - x0low) )
    end

    if Zlow == Zup 
        # In this case we don't need interpolation, and are finished:
        return 10^log10_Alow
    end

    if x ≤ x0up
        log10_Aup = b0up + b1up*x + b2up*(x^2) + b3up*(x^3) + b4up*(x^4) 
    else
        log10_Aup = ( b0up + b1up*x0up + b2up*(x0up^2) + b3up*(x0up^3) + b4up*(x0up^4)  
                  +  (b1up + 2*b2up*x0up + 3*b3up*(x0up^2) + 4*b4up*(x0up^3))*(x - x0up) )
    end

    # Use linear interpolation between them:
    log10_A = log10_Alow + ((log10_Aup - log10_Alow) / log10(Zup/Zlow)) * log10(Zstar_Zsun/Zlow)
    
    return 10^log10_A 
end

""" The He II ionizing luminosity L_EUV3 (L⊙) of a star of ZAMS mass mstarZAMS (M⊙)
    and metallicity Zstar_Zsun (in Solar units). """
function L_EUV3( mstarZAMS::Float64,
                Zstar_Zsun::Float64 = 1.0,
                         A::StellarBandAverages = DefaultStellarBandAverages 
                 )
    dNdt_EUV3(mstarZAMS, Zstar_Zsun) * A.E_EUV3 / Lsun
end


""" The bolometric luminosity L_bol (L⊙) of a star of mass mstar (M⊙)
    and metallicity Zstar_Zsun (in Solar units). """
function L_bol( mstarZAMS::Float64, 
               Zstar_Zsun::Float64 = 1.0,
                        A::StellarBandAverages = DefaultStellarBandAverages 
               )
    L_opt(mstarZAMS, Zstar_Zsun, A) + L_FUV(mstarZAMS, Zstar_Zsun, A) + L_LW(mstarZAMS, Zstar_Zsun, A) +
    L_EUV1(mstarZAMS, Zstar_Zsun, A) + L_EUV2(mstarZAMS, Zstar_Zsun, A) + L_EUV3(mstarZAMS, Zstar_Zsun, A)
end

################################################################################
#######                                                                  #######
#######              S T A R  C L U S T E R  E M I S S I O N             #######
#######                                                                  #######
################################################################################

# The functions below are useful for computing/initializing the photon emission 
# rates from a star cluster of a given total mass and IMF (Kroupa).

const p_high_Kroupa::Float64      = 2.3 
const p_low_Kroupa::Float64       = 1.3 
const mstar_Kroupa_break::Float64 = 0.5  # Kroupa mass break (high-mass tail above this mass)
const C_Kroupa_break::Float64     = mstar_Kroupa_break^(p_high_Kroupa - p_low_Kroupa)
const mstar_Kroupa_min::Float64   = 0.08 
const mstar_Kroupa_max::Float64   = 100.0

""" This function returns the unnormalized Kroupa IMF. """
function ∂N∂m_Kroupa_unnormalized( mstar::Float64 )

    if mstar < mstar_Kroupa_min || mstar > mstar_Kroupa_max
        # Outside the mass range there are no stars, so return 0:
        return 0.0
    elseif mstar < mstar_Kroupa_break 
        return mstar^(- p_low_Kroupa)
    else
        # High-mass tail, with a constant pre-factor set so 
        # IMF is continuous at the mass break:
        return C_Kroupa_break * mstar^(- p_high_Kroupa)
    end
end


""" This function computes the normalization factor for the Kroupa IMF, 
    such that it produces a cluster of a given mass M_StarCluster (M⊙).
    
    Thus:
    
    A_Kroupa = M_StarCluster / [ ∫dm m ∂N/∂m ] ,
    
    where ∫dm m ∂N/∂m can be computed analytically. """
function NormalizationFactor_Kroupa( M_StarCluster::Float64 )

    I_low   = ( mstar_Kroupa_break^(2 - p_low_Kroupa) - mstar_Kroupa_min^(2 - p_low_Kroupa) ) / (
              2 - p_low_Kroupa)

    I_upper = C_Kroupa_break * ( mstar_Kroupa_max^(2 - p_high_Kroupa) - mstar_Kroupa_break^(2 - p_high_Kroupa) ) / (
              2 - p_high_Kroupa)

    # Total integral:
    I       = I_low  +  I_upper

    # Get normalization factor:
    A_Kroupa = M_StarCluster / I 

    return A_Kroupa
end


""" This function computes the photon emission rates in 
    all stellar bands (opt, FUV, LW, EUV1, EUV2, EUV3) 
    from a star cluster of a given cluster mass M_StarCluster (M⊙)
    and metallicity Zstar_Zsun (in Solar units), and assuming the 
    Kroupa IMF. 
    
    Thus, for e.g. the FUV band, with a change of variables to 
    get integration in log-space: 
    
    Ṅ_FUV_SC = ∫dm (∂N/∂m) Ṅ_FUV(m)
             = ∫(dm/m) m (∂N/∂m)(m) Ṅ_FUV(m)
             = ∫dx exp(x) (∂N/∂m)[exp(x)] Ṅ_FUV[exp(x)] """
function ClusterEmissionRates( M_StarCluster::Float64, Zstar_Zsun::Float64 )

    x_min        = log( mstar_Kroupa_min )
    x_max        = log( mstar_Kroupa_max )
    Δx           = ( x_max - x_min ) / 3000  # Step-size in log-space, with 3000 integration points 

    # Get normalization constant for the IMF, such that it is consistent with 
    # a star cluster of the given mass:
    A_Kroupa     = NormalizationFactor_Kroupa( M_StarCluster )

    # Initialize cluster emission rates:
    dNdt_opt_SC  = 0.0
    dNdt_FUV_SC  = 0.0
    dNdt_LW_SC   = 0.0
    dNdt_EUV1_SC = 0.0
    dNdt_EUV2_SC = 0.0
    dNdt_EUV3_SC = 0.0

    # Integrate numerically to find emission rates:
    x = x_min # Start at x = x_min 
    while x < x_max
        # Get stellar mass corresponding to x_mid = x + 0.5 Δx (midpoint):
        x_mid     = x + 0.5 * Δx 
        mstar_mid = exp( x_mid )

        # Get normalized Kroupa IMF ∂N⋆/∂m⋆ at this mass:
        ∂N∂m = A_Kroupa * ∂N∂m_Kroupa_unnormalized( mstar_mid )

        # Integrands:
        if mstar_mid ≥ 8.0
            # Only include emission above 8 M⊙, above which the fits 
            # are applicable (see Denge et al. 2024):
            Integrand_opt  = mstar_mid * ∂N∂m * dNdt_opt( mstar_mid, Zstar_Zsun )
            Integrand_FUV  = mstar_mid * ∂N∂m * dNdt_FUV( mstar_mid, Zstar_Zsun )
            Integrand_LW   = mstar_mid * ∂N∂m * dNdt_LW( mstar_mid, Zstar_Zsun )
            Integrand_EUV1 = mstar_mid * ∂N∂m * dNdt_EUV1( mstar_mid, Zstar_Zsun )
            Integrand_EUV2 = mstar_mid * ∂N∂m * dNdt_EUV2( mstar_mid, Zstar_Zsun )
            Integrand_EUV3 = mstar_mid * ∂N∂m * dNdt_EUV3( mstar_mid, Zstar_Zsun )
        else
            Integrand_opt  = 0.0
            Integrand_FUV  = 0.0
            Integrand_LW   = 0.0
            Integrand_EUV1 = 0.0
            Integrand_EUV2 = 0.0
            Integrand_EUV3 = 0.0
        end

        # Add contributions to emission rates:
        dNdt_opt_SC   += Δx * Integrand_opt
        dNdt_FUV_SC   += Δx * Integrand_FUV
        dNdt_LW_SC    += Δx * Integrand_LW
        dNdt_EUV1_SC  += Δx * Integrand_EUV1
        dNdt_EUV2_SC  += Δx * Integrand_EUV2
        dNdt_EUV3_SC  += Δx * Integrand_EUV3

        # Update x:
        x += Δx 
    end

    # Also get luminosities:
    Lum_opt   = dNdt_opt_SC  * E_opt 
    Lum_FUV   = dNdt_FUV_SC  * E_FUV
    Lum_LW    = dNdt_LW_SC   * E_LW
    Lum_EUV1  = dNdt_EUV1_SC * E_EUV1
    Lum_EUV2  = dNdt_EUV2_SC * E_EUV2
    Lum_EUV3  = dNdt_EUV3_SC * E_EUV3

    return dNdt_opt_SC, dNdt_FUV_SC, dNdt_LW_SC, 
           dNdt_EUV1_SC, dNdt_EUV2_SC, dNdt_EUV3_SC,
           Lum_opt, Lum_FUV, Lum_LW, Lum_EUV1, Lum_EUV2, Lum_EUV3
end


################################################################################
#######                                                                  #######
#######                   P H O T O - H E A T I N G                      #######
#######                                                                  #######
################################################################################

# Pre-compute σ_N × h for metals in the FUV band:
const σSI_FUV_N_hSI_FUV::Float64 = σSI_FUV_N * hSI_FUV

""" Heating rate (erg cm⁻³ s⁻¹) from absorptions of stellar photons 
    in the FUV-band. This is only provided by metal ions that are 
    easily photoionized. """
@inline function HeatingRate_FUV( J_FUV::Float64, 
                                    nSI::Float64,
                                      A::StellarBandAverages = DefaultStellarBandAverages 
                                  )

    @fastmath ( Four_π * J_FUV * A.inv_E_FUV) * ( nSI * σSI_FUV_N_hSI_FUV )
end


# Pre-compute σ_N × h for metals & H₂ in the LW band:
const σH2_LW_diss_hH2_LW::Float64 = σH2_LW_diss * hH2_LW 
const σCI_LW_N_hCI_LW::Float64    = σCI_LW_N    * hCI_LW
const σSI_LW_N_hSI_LW::Float64    = σSI_LW_N    * hSI_LW

const Crad::Float64 = 2e-7  # Typical radiative de-excitation rate for H₂ (see ref. 3)

""" Heating rate (erg cm⁻³ s⁻¹) from absorptions of stellar photons 
    in the LW-band. """
@inline function HeatingRate_LW( SelfShielding::Float64, # Self-shielding for H₂
                                          J_LW::Float64,
                                           nH2::Float64,
                                           nHI::Float64,
                                           nCI::Float64,
                                           nSI::Float64,
                                             T::Float64,
                                             A::StellarBandAverages = DefaultStellarBandAverages 
                               )

    # 4πJ_LW / E_LW (used multiple times):
    Four_π_J_LW_over_E_LW    = @fastmath Four_π * J_LW * A.inv_E_LW

    # H₂ line absorption is self-shielded:
    Four_π_J_LW_over_E_LW_H2 = @fastmath SelfShielding * Four_π_J_LW_over_E_LW

    # Heating from photodissociation of H₂:
    HeatingRate_PhotoDissH2  = @fastmath Four_π_J_LW_over_E_LW_H2 * nH2 * σH2_LW_diss_hH2_LW 

    # Average energy injected per UV pumping event (i.e. no photodissociation):
    T          = clamp( T, 10.0, 1e5 )
    Cdex       = @fastmath 1e-12 * sqrt(T) * ( 1.4 * exp( - 18100.0 / (T + 1200.0)) * nH2  
                                             + exp(- 1000.0/T) * nHI )
    E_H2_pump  = @fastmath 2 * eV * Cdex / (Cdex + Crad)

    # Heating from vibrational excitation (but not photodissociation) of H₂:
    HeatingRate_PumpH2  = @fastmath Four_π_J_LW_over_E_LW_H2 * nH2 * σH2_LW_pump * E_H2_pump

    # Heating rate from photoionization of C I:
    HeatingRate_CI = @fastmath Four_π_J_LW_over_E_LW * ( nCI * σCI_LW_N_hCI_LW )

    # Heating rate from photoionization of S I:
    HeatingRate_SI = @fastmath Four_π_J_LW_over_E_LW * ( nSI * σSI_LW_N_hSI_LW )

    return HeatingRate_PhotoDissH2 + HeatingRate_PumpH2 + HeatingRate_CI + HeatingRate_SI
end


# Pre-compute σ_N × h for metals & H₂ in the EUV1 band:
const σH2_EUV1_hH2_EUV1::Float64     = σH2_EUV1    * hH2_EUV1
const σCI_EUV1_N_hCI_EUV1::Float64   = σCI_EUV1_N  * hCI_EUV1
const σCII_EUV1_N_hCII_EUV1::Float64 = σCII_EUV1_N * hCII_EUV1
const σOI_EUV1_N_hOI_EUV1::Float64   = σOI_EUV1_N  * hOI_EUV1
const σSI_EUV1_N_hSI_EUV1::Float64   = σSI_EUV1_N  * hSI_EUV1
const σSII_EUV1_N_hSII_EUV1::Float64 = σSII_EUV1_N * hSII_EUV1

""" Heating rate (erg cm⁻³ s⁻¹) from absorptions of stellar photons 
    in the EUV1-band. """
@inline function HeatingRate_EUV1( J_EUV1::Float64, 
                                      nHI::Float64, 
                                      nH2::Float64,
                                      nCI::Float64,
                                     nCII::Float64,
                                      nOI::Float64,
                                      nSI::Float64,
                                     nSII::Float64,
                                        A::StellarBandAverages = DefaultStellarBandAverages 
                                  )

    @fastmath ( Four_π * J_EUV1 * A.inv_E_EUV1) * ( nHI  * A.σHI_EUV1_N * A.hHI_EUV1  +  nH2  * σH2_EUV1_hH2_EUV1
                                                  + nCI  * σCI_EUV1_N_hCI_EUV1        +  nCII * σCII_EUV1_N_hCII_EUV1
                                                  + nOI  * σOI_EUV1_N_hOI_EUV1        +  nSI  * σSI_EUV1_N_hSI_EUV1
                                                  + nSII * σSII_EUV1_N_hSII_EUV1 )
end


# Pre-compute σ_N × h for metals and H₂ in the EUV2 band: 
const σH2_EUV2_hH2_EUV2::Float64       = σH2_EUV2     * hH2_EUV2
const σCI_EUV2_N_hCI_EUV2::Float64     = σCI_EUV2_N   * hCI_EUV2
const σCII_EUV2_N_hCII_EUV2::Float64   = σCII_EUV2_N  * hCII_EUV2
const σOI_EUV2_N_hOI_EUV2::Float64     = σOI_EUV2_N   * hOI_EUV2
const σOII_EUV2_N_hOII_EUV2::Float64   = σOII_EUV2_N  * hOII_EUV2
const σSI_EUV2_N_hSI_EUV2::Float64     = σSI_EUV2_N   * hSI_EUV2
const σSII_EUV2_N_hSII_EUV2::Float64   = σSII_EUV2_N  * hSII_EUV2
const σSIII_EUV2_N_hSIII_EUV2::Float64 = σSIII_EUV2_N * hSIII_EUV2

""" Heating rate (erg cm⁻³ s⁻¹) from absorptions of stellar photons 
    in the EUV2-band. """
@inline function HeatingRate_EUV2( J_EUV2::Float64, 
                                      nHI::Float64, 
                                      nH2::Float64,
                                     nHeI::Float64,
                                      nCI::Float64,
                                     nCII::Float64,
                                      nOI::Float64,
                                     nOII::Float64,
                                      nSI::Float64,
                                     nSII::Float64,
                                    nSIII::Float64,
                                        A::StellarBandAverages = DefaultStellarBandAverages 
                                  )

    @fastmath ( Four_π * J_EUV2 * A.inv_E_EUV2) * ( 
      nHI   * A.σHI_EUV2_N * A.hHI_EUV2  +  nH2  * σH2_EUV2_hH2_EUV2      +  nHeI * A.σHeI_EUV2_N * A.hHeI_EUV2

      # Metal contributions: 
    + nCI   * σCI_EUV2_N_hCI_EUV2    +  nCII * σCII_EUV2_N_hCII_EUV2  +  nOI  * σOI_EUV2_N_hOI_EUV2
    + nOII  * σOII_EUV2_N_hOII_EUV2  +  nSI  * σSI_EUV2_N_hSI_EUV2    +  nSII * σSII_EUV2_N_hSII_EUV2
    + nSIII * σSIII_EUV2_N_hSIII_EUV2 )
end


# Pre-compute σ_N × h for metals and H₂ in the EUV3 band: 
const σH2_EUV3_hH2_EUV3::Float64       = σH2_EUV3     * hH2_EUV3
const σCI_EUV3_N_hCI_EUV3::Float64     = σCI_EUV3_N   * hCI_EUV3
const σCII_EUV3_N_hCII_EUV3::Float64   = σCII_EUV3_N  * hCII_EUV3
const σOI_EUV3_N_hOI_EUV3::Float64     = σOI_EUV3_N   * hOI_EUV3
const σOII_EUV3_N_hOII_EUV3::Float64   = σOII_EUV3_N  * hOII_EUV3
const σOIII_EUV3_N_hOIII_EUV3::Float64 = σOIII_EUV3_N * hOIII_EUV3
const σSI_EUV3_N_hSI_EUV3::Float64     = σSI_EUV3_N   * hSI_EUV3
const σSII_EUV3_N_hSII_EUV3::Float64   = σSII_EUV3_N  * hSII_EUV3
const σSIII_EUV3_N_hSIII_EUV3::Float64 = σSIII_EUV3_N * hSIII_EUV3

""" Heating rate (erg cm⁻³ s⁻¹) from absorptions of stellar photons 
    in the EUV3-band. """
@inline function HeatingRate_EUV3( J_EUV3::Float64, 
                                      nHI::Float64, 
                                      nH2::Float64,
                                     nHeI::Float64,
                                    nHeII::Float64,
                                      nCI::Float64,
                                     nCII::Float64,
                                      nOI::Float64,
                                     nOII::Float64,
                                    nOIII::Float64,
                                      nSI::Float64,
                                     nSII::Float64,
                                    nSIII::Float64,
                                        A::StellarBandAverages = DefaultStellarBandAverages
                                  )

    @fastmath ( Four_π * J_EUV3 * A.inv_E_EUV3) * ( 
        nHI * A.σHI_EUV3_N  * A.hHI_EUV3   +  nH2   * σH2_EUV3_hH2_EUV3  
    +  nHeI * A.σHeI_EUV3_N * A.hHeI_EUV3  +  nHeII * A.σHeII_EUV3_N * A.hHeII_EUV3
    
    # Metal contribution:
    +  nCI * σCI_EUV3_N_hCI_EUV3  +  nCII * σCII_EUV3_N_hCII_EUV3
    +  nOI * σOI_EUV3_N_hOI_EUV3  +  nOII * σOII_EUV3_N_hOII_EUV3  +  nOIII * σOIII_EUV3_N_hOIII_EUV3
    +  nSI * σSI_EUV3_N_hSI_EUV3  +  nSII * σSII_EUV3_N_hSII_EUV3  +  nSIII * σSIII_EUV3_N_hSIII_EUV3 )
end


const ConstantComptonHeating::Float64 = σT / (me * c)  # Convenient constant 

""" Compton heating/cooling rate (erg cm⁻³ s⁻¹) from  
    radiation backgrounds (see e.g. Ploeckinger et al. 2025):

    (4π ne σT / (me c²)) ∫dν J (hν - 4 kB T) """
@inline function HeatingRate_Compton( J_opt::Float64,
                                      J_FUV::Float64,
                                       J_LW::Float64,
                                     J_EUV1::Float64, 
                                     J_EUV2::Float64,
                                     J_EUV3::Float64,
                                         ne::Float64,
                                          T::Float64,
                                      e_Lyα::Float64 = 0.0,
                                          A::StellarBandAverages = DefaultStellarBandAverages
                                       )
    # Convenient variable:
    Four_kBT             = @fastmath 4 * kB * T

    # Heating/cooling rate, considering the stellar intensity in all bands: 
    HeatingRate_Compton  = @fastmath Four_π_over_c * ne * ConstantComptonHeating * (  J_opt  * (A.E_opt  - Four_kBT)
                                                                                    + J_FUV  * (A.E_FUV  - Four_kBT)
                                                                                    + J_LW   * (A.E_LW   - Four_kBT)
                                                                                    + J_EUV1 * (A.E_EUV1 - Four_kBT)
                                                                                    + J_EUV2 * (A.E_EUV2 - Four_kBT)
                                                                                    + J_EUV3 * (A.E_EUV3 - Four_kBT) )

    # Add contribution from Lyα photons:
    if e_Lyα > 0.0
        HeatingRate_Compton += @fastmath ne * ConstantComptonHeating * e_Lyα * (ELyα - Four_kBT)
    end

    return HeatingRate_Compton
end

################################################################################
#######                                                                  #######
#######    P H O T O - I O N I Z A T I O N / D I S S O C I A T I O N     #######
#######                                                                  #######
################################################################################

# Below are rates of photoionization of H I, He I, He II, and photo-dissociation 
# of H₂.

""" H₂ photodissociation rate (s⁻¹) 
    from absorptions of stellar Lyman-Werner photons. """
@inline function PhotoDissRate_H2( J_LW::Float64, A::StellarBandAverages = DefaultStellarBandAverages )
    @fastmath Four_π * J_LW * σH2_LW_diss * A.inv_E_LW
end


""" CO photodissociation rate (s⁻¹) 
    from absorptions of stellar Lyman-Werner photons. """
@inline function PhotoDissRate_CO( J_LW::Float64, A::StellarBandAverages = DefaultStellarBandAverages )
    @fastmath Four_π * J_LW * σCO_LW * A.inv_E_LW
end


""" OHₓ (represented by OH) photodissociation rate (s⁻¹) 
    from absorptions of stellar photons & Lyα photons. """
@inline function PhotoDissRate_OHₓ( J_FUV::Float64, 
                                     J_LW::Float64,
                                    e_Lyα::Float64 = 0.0,
                                        A::StellarBandAverages = DefaultStellarBandAverages
                                      )
    @fastmath (  Four_π * ( J_FUV * σOH_FUV * A.inv_E_FUV  +  J_LW * σOH_LW * A.inv_E_LW )

    +  c * e_Lyα * σOH_Lyα * inv_ELyα  )
end


""" CHₓ (represented by CH) photodissociation rate (s⁻¹) 
    from absorptions of stellar photons & Lyα photons. """
@inline function PhotoDissRate_CHₓ( J_opt::Float64,
                                    J_FUV::Float64, 
                                     J_LW::Float64,
                                    e_Lyα::Float64 = 0.0,
                                        A::StellarBandAverages = DefaultStellarBandAverages
                                    )
    @fastmath (  Four_π * ( J_opt * σCH_opt * A.inv_E_opt +  J_FUV * σCH_FUV * A.inv_E_FUV  +  J_LW * σCH_LW * A.inv_E_LW )
    
    +  c * e_Lyα * σCH_Lyα * inv_ELyα  )
end


""" H⁻ photodetachment rate (s⁻¹) 
    from absorptions of stellar photons & Lyα photons. """
@inline function PhotoDetachRate_Hminus( J_opt::Float64,
                                         J_FUV::Float64,
                                          J_LW::Float64,
                                        J_EUV1::Float64,
                                        J_EUV2::Float64,
                                        J_EUV3::Float64,
                                         e_Lyα::Float64 = 0.0,
                                             A::StellarBandAverages = DefaultStellarBandAverages
                                       )

    @fastmath (  Four_π * ( J_opt  * σHminus_opt  * A.inv_E_opt  
                          + J_FUV  * σHminus_FUV  * A.inv_E_FUV  
                          + J_LW   * σHminus_LW   * A.inv_E_LW
                          + J_EUV1 * σHminus_EUV1 * A.inv_E_EUV1 
                          + J_EUV2 * σHminus_EUV2 * A.inv_E_EUV2  
                          + J_EUV3 * σHminus_EUV3 * A.inv_E_EUV3 )

                +  c * e_Lyα * σHminus_Lyα * inv_ELyα  )
end


""" HD photodissociation rate (s⁻¹) 
    from absorptions of stellar Lyman-Werner photons. """
@inline function PhotoDissRate_HD( J_LW::Float64,
                                      A::StellarBandAverages = DefaultStellarBandAverages )
    @fastmath Four_π * J_LW * σHD_LW * A.inv_E_LW
end


""" H₂ photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_H2( J_EUV1::Float64,
                             J_EUV2::Float64,
                             J_EUV3::Float64,
                                  A::StellarBandAverages = DefaultStellarBandAverages
                              )
    @fastmath Four_π * ( J_EUV1 * σH2_EUV1 * A.inv_E_EUV1 
                       + J_EUV2 * σH2_EUV2 * A.inv_E_EUV2 
                       + J_EUV3 * σH2_EUV3 * A.inv_E_EUV3  )
end


""" H I photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_HI( J_EUV1::Float64, 
                             J_EUV2::Float64,
                             J_EUV3::Float64,
                                  A::StellarBandAverages = DefaultStellarBandAverages
                             )
    @fastmath Four_π * ( J_EUV1 * A.σHI_EUV1_N * A.inv_E_EUV1 
                       + J_EUV2 * A.σHI_EUV2_N * A.inv_E_EUV2 
                       + J_EUV3 * A.σHI_EUV3_N * A.inv_E_EUV3  )
end


""" He I photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_HeI( J_EUV2::Float64,
                              J_EUV3::Float64,
                                   A::StellarBandAverages = DefaultStellarBandAverages
                             )
    @fastmath Four_π * ( J_EUV2 * A.σHeI_EUV2_N * A.inv_E_EUV2 
                       + J_EUV3 * A.σHeI_EUV3_N * A.inv_E_EUV3  )
end


""" He II photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_HeII( J_EUV3::Float64,
                                    A::StellarBandAverages = DefaultStellarBandAverages )
    Four_π * J_EUV3 * A.σHeII_EUV3_N * A.inv_E_EUV3 
end


""" C I photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_CI( J_LW::Float64,
                           J_EUV1::Float64, 
                           J_EUV2::Float64,
                           J_EUV3::Float64, 
                                A::StellarBandAverages = DefaultStellarBandAverages
                             )
    @fastmath Four_π * ( J_LW   * σCI_LW_N   * A.inv_E_LW
                       + J_EUV1 * σCI_EUV1_N * A.inv_E_EUV1 
                       + J_EUV2 * σCI_EUV2_N * A.inv_E_EUV2 
                       + J_EUV3 * σCI_EUV3_N * A.inv_E_EUV3  )
end


""" C II photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_CII( J_EUV1::Float64, 
                              J_EUV2::Float64,
                              J_EUV3::Float64, 
                                   A::StellarBandAverages = DefaultStellarBandAverages
                             )
    @fastmath Four_π * ( J_EUV1 * σCII_EUV1_N * A.inv_E_EUV1 
                       + J_EUV2 * σCII_EUV2_N * A.inv_E_EUV2 
                       + J_EUV3 * σCII_EUV3_N * A.inv_E_EUV3  )
end


""" O I photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_OI( J_EUV1::Float64, 
                             J_EUV2::Float64,
                             J_EUV3::Float64, 
                                  A::StellarBandAverages = DefaultStellarBandAverages
                             )
    @fastmath Four_π * ( J_EUV1 * σOI_EUV1_N * A.inv_E_EUV1 
                       + J_EUV2 * σOI_EUV2_N * A.inv_E_EUV2 
                       + J_EUV3 * σOI_EUV3_N * A.inv_E_EUV3  )
end


""" O II photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_OII( J_EUV2::Float64,
                              J_EUV3::Float64, 
                                   A::StellarBandAverages = DefaultStellarBandAverages
                             )
    @fastmath Four_π * ( J_EUV2 * σOII_EUV2_N * A.inv_E_EUV2 
                       + J_EUV3 * σOII_EUV3_N * A.inv_E_EUV3  )
end


""" O III photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_OIII( J_EUV3::Float64, 
                                    A::StellarBandAverages = DefaultStellarBandAverages
                              )
    @fastmath Four_π * J_EUV3 * σOIII_EUV3_N * A.inv_E_EUV3 
end


""" S I photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_SI( J_LW::Float64,
                            J_FUV::Float64,
                           J_EUV1::Float64, 
                           J_EUV2::Float64,
                           J_EUV3::Float64, 
                                A::StellarBandAverages = DefaultStellarBandAverages
                             )
    @fastmath Four_π * ( J_LW   * σSI_LW_N   * A.inv_E_LW 
                       + J_FUV  * σSI_FUV_N  * A.inv_E_FUV 
                       + J_EUV1 * σSI_EUV1_N * A.inv_E_EUV1 
                       + J_EUV2 * σSI_EUV2_N * A.inv_E_EUV2 
                       + J_EUV3 * σSI_EUV3_N * A.inv_E_EUV3  )
end


""" S II photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_SII( J_EUV1::Float64, 
                              J_EUV2::Float64,
                              J_EUV3::Float64, 
                                   A::StellarBandAverages = DefaultStellarBandAverages
                             )
    @fastmath Four_π * ( J_EUV1 * σSII_EUV1_N * A.inv_E_EUV1 
                       + J_EUV2 * σSII_EUV2_N * A.inv_E_EUV2 
                       + J_EUV3 * σSII_EUV3_N * A.inv_E_EUV3  )
end


""" S III photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_SIII( J_EUV2::Float64,
                               J_EUV3::Float64, 
                                    A::StellarBandAverages = DefaultStellarBandAverages
                              )
    @fastmath Four_π * ( J_EUV2 * σSIII_EUV2_N * A.inv_E_EUV2 
                       + J_EUV3 * σSIII_EUV3_N * A.inv_E_EUV3  )
end


""" The Strömgren radius (cm) for a star of mass mstar (M⊙)
    in a cloud of hydrogen number density nH (cm⁻³). """
@inline function StromGrenRadius( mstar::Float64, 
                                     nH::Float64,
                                      T::Float64 = 1e4,
                                  CaseB::Bool    = true,
                             Zstar_Zsun::Float64 = 1.0
                                 )
    dNdt_LyC = dNdt_EUV1(mstar, Zstar_Zsun) + dNdt_EUV2(mstar, Zstar_Zsun) + dNdt_EUV3(mstar, Zstar_Zsun)
    
    return ( 3 * dNdt_LyC / (Four_π * krecHII(T, CaseB) * (nH^2)) )^(1/3)
end


################################################################################
#######                                                                  #######
#######   P H O T O - I O N I Z A T I O N  C R O S S - S E C T I O N S   #######
#######                                                                  #######
################################################################################

# The code below is for computing SED photon number & energy-weighted photoionization 
# cross-sections & photoheating. For use in idealized tests where specific SEDs 
# are used. 


""" This function computes the mean photon energy, in an interval between E_min 
    and E_max:
    
    E_av = [∫dν 4πJ] / [∫dν 4πJ / hν] """
function Compute_meanphotonenergy( J::Function, 
                               E_min::Float64, E_max::Float64, 
                                   N::Int = 4096 )

    numerator = IntegrateEnergyBand(
        E -> J(E),
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
    E_av = numerator / denominator            

    return E_av 
end



""" This function computes the H I, He I, and He II photoionization 
    cross-sections (number & energy-weighted), the photoheating, and 
    mean photon energies, assuming a black-body spectrum 
    of temperature Teff (K). """
function ComputeHydrogenHelium_σ_and_heating_BB( Teff::Float64, 
                                                E_min::Float64, 
                                                E_max::Float64, N::Int = 4096 )

    # Ionization thresholds:
    Eth_HI   = 13.6 * eV
    Eth_HeI  = 24.6 * eV
    Eth_HeII = 54.4 * eV

    # Black-body spectral shape:
    J(E)     = BlackBodyShape_E( E, Teff )

    # Photon number-weighted cross-sections:
    σHI_N    = Compute_σ_N( J, σHI,   E_min, E_max, N )
    σHeI_N   = Compute_σ_N( J, σHeI,  E_min, E_max, N )
    σHeII_N  = Compute_σ_N( J, σHeII, E_min, E_max, N )

    # Energy-weighted cross-sections:
    σHI_E    = Compute_σ_E( J, σHI,   E_min, E_max, N  )
    σHeI_E   = Compute_σ_E( J, σHeI,  E_min, E_max, N  )
    σHeII_E  = Compute_σ_E( J, σHeII, E_min, E_max, N  )

    # Mean excess heating energy per photoionization:
    hHI      = Compute_photoheating( J, σHI,   E_min, E_max, Eth_HI,   N )
    hHeI     = Compute_photoheating( J, σHeI,  E_min, E_max, Eth_HeI,  N )
    hHeII    = Compute_photoheating( J, σHeII, E_min, E_max, Eth_HeII, N )

    # Average photon energies: 
    E_av     = Compute_meanphotonenergy( J, E_min, E_max, N )

    return σHI_N, σHeI_N, σHeII_N, 
           σHI_E, σHeI_E, σHeII_E, 
           hHI,   hHeI,   hHeII,
           E_av
end


function GetBBStellarAverages( Teff::Float64 )

    # Get averages for the H/He ionizing bands:
    σHI_EUV1_BB,   σHeI_EUV1_BB,   σHeII_EUV1_BB, 
    σHI_EUV1_BB_E, σHeI_EUV1_BB_E, σHeII_EUV1_BB_E, 
    hHI_EUV1_BB,   hHeI_EUV1_BB,   hHeII_EUV1_BB,
    E_EUV1_BB = ComputeHydrogenHelium_σ_and_heating_BB( Teff, 13.6 * eV, 24.6 * eV )

    σHI_EUV2_BB,   σHeI_EUV2_BB,   σHeII_EUV2_BB, 
    σHI_EUV2_BB_E, σHeI_EUV2_BB_E, σHeII_EUV2_BB_E, 
    hHI_EUV2_BB,   hHeI_EUV2_BB,   hHeII_EUV2_BB,
    E_EUV2_BB = ComputeHydrogenHelium_σ_and_heating_BB( Teff, 24.6 * eV, 54.4 * eV )

    σHI_EUV3_BB,   σHeI_EUV3_BB,   σHeII_EUV3_BB, 
    σHI_EUV3_BB_E, σHeI_EUV3_BB_E, σHeII_EUV3_BB_E, 
    hHI_EUV3_BB,   hHeI_EUV3_BB,   hHeII_EUV3_BB,
    E_EUV3_BB = ComputeHydrogenHelium_σ_and_heating_BB( Teff, 54.4 * eV, 100.0 * eV )

    # Also get photon energies in other bands (opt, FUV, LW):
    _,   _,   _, 
    _,   _,   _, 
    _,   _,   _,
    E_opt_BB = ComputeHydrogenHelium_σ_and_heating_BB( Teff, 1.0 * eV, 5.8 * eV )

    _,   _,   _, 
    _,   _,   _, 
    _,   _,   _,
    E_FUV_BB = ComputeHydrogenHelium_σ_and_heating_BB( Teff, 5.8 * eV, 11.2 * eV )

    _,   _,   _, 
    _,   _,   _, 
    _,   _,   _,
    E_LW_BB = ComputeHydrogenHelium_σ_and_heating_BB( Teff, 11.2 * eV, 13.6 * eV )

    # Save in struct:
    return StellarBandAverages( E_opt_BB, E_FUV_BB, E_LW_BB, E_EUV1_BB, E_EUV2_BB, E_EUV3_BB,
                                1 / E_opt_BB, 1 / E_FUV_BB, 1  / E_LW_BB, 1 / E_EUV1_BB, 1 / E_EUV2_BB, 1 / E_EUV3_BB,
                                σHI_EUV1_BB, σHI_EUV2_BB, σHI_EUV3_BB, σHeI_EUV2_BB, σHeI_EUV3_BB, σHeII_EUV3_BB, 
                                σHI_EUV1_BB_E, σHI_EUV2_BB_E, σHI_EUV3_BB_E, σHeI_EUV2_BB_E, σHeI_EUV3_BB_E, σHeII_EUV3_BB_E,
                                hHI_EUV1_BB, hHI_EUV2_BB, hHI_EUV3_BB, hHeI_EUV2_BB, hHeI_EUV3_BB, hHeII_EUV3_BB 
                                 )
end
