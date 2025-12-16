

""" FILE:       StellarEmissionAbsorption.jl

    PURPOSE:    Contains everything related to stellar emission, 
                i.e. photon emission rates and luminosities in different bands
                for stars of different mass. 

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

# Export functions:
export dNdt_opt, L_opt, dNdt_FUV, L_FUV, dNdt_LW, L_LW,
       dNdt_EUV1, L_EUV1, dNdt_EUV2, L_EUV2, dNdt_EUV3, L_EUV3, L_bol,
       HeatingRate_LW, HeatingRate_EUV1, HeatingRate_EUV2, HeatingRate_EUV3, 
       HeatingRate_PhotoElectric, HeatingRate_Compton,
       IonRate_HI, IonRate_H2, IonRate_HeI, IonRate_HeII, IonRate_OI, IonRate_CI,
       PhotoDissRate_H2, PhotoDetachRate_Hminus, PhotoDissRate_HD, PhotoDissRate_CO,
       PhotoDissRate_OHₓ, PhotoDissRate_CHₓ, StromGrenRadius

# Average stellar photon energy in each band (see ref. 2):

const E_opt      = 3.67  * eV       # 1-5.8 eV band (optical)
const E_FUV      = 8.45  * eV       # 5.8-11.2 eV band (FUV)
const E_LW       = 12.26 * eV       # 11.2-13.6 eV band (Lyman-Werner)
const E_EUV1     = 18.01 * eV       # 13.6-24.6 eV band (H I ionizing)
const E_EUV2     = 29.89 * eV       # 24.6-54.4 eV band (He I ionizing)
const E_EUV3     = 56.85 * eV       # 54.4-∞ eV band (He II ionizing) 

export E_opt, E_FUV, E_LW, E_EUV1, E_EUV2, E_EUV3

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
const σCI_LW       = 17.2e-18
const σOH_LW       = 4.02e-18
const σCH_LW       = 1.18e-18
const σCO_LW       = 17.3e-18 

const σHI_EUV1     = 3.37e-18
const σH2_EUV1     = 5.09e-18 
const σHminus_EUV1 = 3.2e-18
const σCI_EUV1     = 12.1e-18
const σOI_EUV1     = 7.86e-18

const σHI_EUV2     = 0.79e-18
const σH2_EUV2     = 2.42e-18
const σHeI_EUV2    = 5.10e-18
const σHminus_EUV2 = 0.71e-18
const σCI_EUV2     = 5.68e-18
const σOI_EUV2     = 10.8e-18

const σHI_EUV3     = 0.11e-18
const σH2_EUV3     = 0.32e-18
const σHeI_EUV3    = 0.77e-18
const σHeII_EUV3   = 1.42e-18
const σHminus_EUV3 = 0.092e-18
const σCI_EUV3     = 1.61e-18
const σOI_EUV3     = 4.73e-18

export σHminus_opt, σCH_opt,
       σHminus_FUV, σOH_FUV, σCH_FUV,
       σHI_LW, σH2_LW, σHD_LW, σHminus_LW, σCI_LW, σOH_LW, σCH_LW, σCO_LW,
       σHI_EUV1, σH2_EUV1, σHminus_EUV1, σCI_EUV1, σOI_EUV1,
       σHI_EUV2, σH2_EUV2, σHeI_EUV2, σHminus_EUV2, σCI_EUV2, σOI_EUV2,
       σHI_EUV3, σH2_EUV3, σHeI_EUV3, σHeII_EUV3, σHminus_EUV3, σCI_EUV3, σOI_EUV3

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
               Zstar_Zsun::Float64 = 1.0
               )
    dNdt_opt(mstarZAMS, Zstar_Zsun) * E_opt / Lsun
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
               Zstar_Zsun::Float64 = 1.0
                )
    dNdt_FUV(mstarZAMS, Zstar_Zsun) * E_FUV / Lsun
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
              Zstar_Zsun::Float64 = 1.0 
              )
    dNdt_LW(mstarZAMS, Zstar_Zsun) * E_LW / Lsun
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
                Zstar_Zsun::Float64 = 1.0
                 )
    dNdt_EUV1(mstarZAMS, Zstar_Zsun) * E_EUV1 / Lsun
end


""" The He I ionizing photon output dN_opt/dt (s⁻¹) from a star of ZAMS 
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
                Zstar_Zsun::Float64 = 1.0
                 )
    dNdt_EUV2(mstarZAMS, Zstar_Zsun) * E_EUV2 / Lsun
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
                Zstar_Zsun::Float64 = 1.0 
                 )
    dNdt_EUV3(mstarZAMS, Zstar_Zsun) * E_EUV3 / Lsun
end


""" The bolometric luminosity L_bol (L⊙) of a star of mass mstar (M⊙)
    and metallicity Zstar_Zsun (in Solar units). """
function L_bol( mstarZAMS::Float64, 
               Zstar_Zsun::Float64 = 1.0 
               )
    L_opt(mstarZAMS, Zstar_Zsun) + L_FUV(mstarZAMS, Zstar_Zsun) + L_LW(mstarZAMS, Zstar_Zsun) +
    L_EUV1(mstarZAMS, Zstar_Zsun) + L_EUV2(mstarZAMS, Zstar_Zsun) + L_EUV3(mstarZAMS, Zstar_Zsun)
end


################################################################################
#######                                                                  #######
#######                   P H O T O - H E A T I N G                      #######
#######                                                                  #######
################################################################################

const Crad::Float64 = 2e-7  # Typical radiative de-excitation rate for H₂ (see ref. 3)

""" Heating rate (erg cm⁻³ s⁻¹) from absorptions of stellar photons 
    in the LW-band. """
@inline function HeatingRate_LW( J_LW::Float64,
                                  nH2::Float64,
                                  nHI::Float64,
                                    T::Float64 
                                  )
    # Heating from photodissociation of H₂:
    HeatingRate_PhotoDissH2 = ( Four_π * J_LW / E_LW) * nH2 * σH2_LW_diss * hH2_LW 

    # Average energy injected per UV pumping event (i.e. no photodissociation):
    T          = clamp( T, 10.0, 1e5 )
    Cdex       = 1e-12 * sqrt(T) * ( 1.4 * exp( - 18100.0 / (T + 1200.0)) * nH2  
                                 +   exp(- 1000.0/T) * nHI )
    E_H2_pump  = 2 * eV * Cdex / (Cdex + Crad)

    # Heating from vibrational excitation (but not photodissociation) of H₂:
    HeatingRate_PumpH2  = ( Four_π * J_LW / E_LW) * nH2 * σH2_LW_pump * E_H2_pump

    return HeatingRate_PhotoDissH2 + HeatingRate_PumpH2
end


""" Heating rate (erg cm⁻³ s⁻¹) from absorptions of stellar photons 
    in the EUV1-band. """
@inline function HeatingRate_EUV1( J_EUV1::Float64, 
                                      nHI::Float64, 
                                      nH2::Float64 
                                  )

    ( Four_π * J_EUV1 / E_EUV1) * ( nHI * σHI_EUV1 * hHI_EUV1  +  nH2 * σH2_EUV1 * hH2_EUV1 )
end


""" Heating rate (erg cm⁻³ s⁻¹) from absorptions of stellar photons 
    in the EUV2-band. """
@inline function HeatingRate_EUV2( J_EUV2::Float64, 
                                      nHI::Float64, 
                                      nH2::Float64,
                                     nHeI::Float64
                                  )

    ( Four_π * J_EUV2 / E_EUV2) * ( 
    nHI * σHI_EUV2 * hHI_EUV2  +  nH2 * σH2_EUV2 * hH2_EUV2  +  nHeI * σHeI_EUV2 * hHeI_EUV2 )
end


""" Heating rate (erg cm⁻³ s⁻¹) from absorptions of stellar photons 
    in the EUV3-band. """
@inline function HeatingRate_EUV3( J_EUV3::Float64, 
                                      nHI::Float64, 
                                      nH2::Float64,
                                     nHeI::Float64,
                                    nHeII::Float64
                                  )

    ( Four_π * J_EUV3 / E_EUV3) * ( 
        nHI * σHI_EUV3 * hHI_EUV3  +  nH2 * σH2_EUV3 * hH2_EUV3  
    +  nHeI * σHeI_EUV3 * hHeI_EUV3  + nHeII * σHeII_EUV3 * hHeII_EUV3 )
end


const Γpe_0::Float64 = 1.3e-24

""" Photoelectric heating rate (erg cm⁻³ s⁻¹) from absorptions of stellar photons 
    in the FUV & LW bands, as well as Lyα photons, by dust. """
@inline function HeatingRate_PhotoElectric( J_FUV::Float64,
                                             J_LW::Float64,
                                                D::Float64,  # Dust-to-gas ratio
                                               nH::Float64,
                                               ne::Float64,
                                                T::Float64,
                                            e_Lyα::Float64 = 0.0
                                              )
    # Compute the charging parameter & G0:
    F_eff =  Four_π * ( J_FUV + J_LW )  +  c * e_Lyα
    G0    =  F_eff / F_Habing    # Flux normalized by Habing value (see ConstantsAndFunctions.jl)
    ψ_FUV =  G0 * sqrt(T) / ne 

    # Efficiency parameter:
    ϵ     = ( 0.0487 / (1 + 0.004 * (ψ_FUV / 0.5)^0.73) +
              0.0365 * (T/1e4)^0.7 / (1 + 2e-4 * (ψ_FUV / 0.5)) )

    # The heating rate:
    Γpe   =  Γpe_0 * ( D / 0.01 ) * ϵ * G0 * nH 

    return Γpe
end


""" Compton heating/cooling rate (erg cm⁻³ s⁻¹) from stellar 
    radiation backgrounds (see e.g. Ploeckinger et al. 2025):

    (4π ne σT / (me c²)) ∫dν J (hν - 4 kB T) 
    """
@inline function HeatingRate_Compton( J_opt::Float64,
                                      J_FUV::Float64,
                                       J_LW::Float64,
                                     J_EUV1::Float64, 
                                     J_EUV2::Float64,
                                     J_EUV3::Float64,
                                         ne::Float64,
                                          T::Float64,
                                      e_Lyα::Float64 = 0.0
                                       )
    # Convenient variable:
    Four_kBT             = 4 * kB * T

    # Heating/cooling rate, considering the stellar intensity in all bands: 
    HeatingRate_Compton  = Four_π * ne * (σT / (me * c^2)) * (  J_opt  * (E_opt - Four_kBT)
                                                              + J_FUV  * (E_FUV - Four_kBT)
                                                              + J_LW   * (E_LW - Four_kBT)
                                                              + J_EUV1 * (E_EUV1 - Four_kBT)
                                                              + J_EUV2 * (E_EUV2 - Four_kBT)
                                                              + J_EUV3 * (E_EUV3 - Four_kBT) )

    # Add contribution from Lyα photons:
    if e_Lyα > 0.0
        HeatingRate_Compton += ne * (σT / (me * c)) * e_Lyα * (ELyα - Four_kBT)
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
@inline function PhotoDissRate_H2( J_LW::Float64 )
    Four_π * J_LW * σH2_LW_diss / E_LW
end


""" CO photodissociation rate (s⁻¹) 
    from absorptions of stellar Lyman-Werner photons. """
@inline function PhotoDissRate_CO( J_LW::Float64 )
    Four_π * J_LW * σCO_LW / E_LW
end


""" OHₓ (represented by OH) photodissociation rate (s⁻¹) 
    from absorptions of stellar photons. """
@inline function PhotoDissRate_OHₓ( J_FUV::Float64, 
                                     J_LW::Float64
                                      )
    Four_π * ( J_FUV * σOH_FUV / E_FUV  +  J_LW * σOH_LW / E_LW )
end


""" CHₓ (represented by CH) photodissociation rate (s⁻¹) 
    from absorptions of stellar photons. """
@inline function PhotoDissRate_CHₓ( J_opt::Float64,
                                    J_FUV::Float64, 
                                     J_LW::Float64
                                    )
    Four_π * ( J_opt * σCH_opt / E_opt +  J_FUV * σCH_FUV / E_FUV  +  J_LW * σCH_LW / E_LW )
end


""" H⁻ photodetachment rate (s⁻¹) 
    from absorptions of stellar photons. """
@inline function PhotoDetachRate_Hminus( J_opt::Float64,
                                         J_FUV::Float64,
                                          J_LW::Float64,
                                        J_EUV1::Float64,
                                        J_EUV2::Float64,
                                        J_EUV3::Float64
                                       )

    Four_π * ( J_opt  * σHminus_opt  / E_opt  
             + J_FUV  * σHminus_FUV  / E_FUV  
             + J_LW   * σHminus_LW   / E_LW
             + J_EUV1 * σHminus_EUV1 / E_EUV1 
             + J_EUV2 * σHminus_EUV2 / E_EUV2  
             + J_EUV3 * σHminus_EUV3 / E_EUV3 )
end


""" HD photodissociation rate (s⁻¹) 
    from absorptions of stellar Lyman-Werner photons. """
@inline function PhotoDissRate_HD( J_LW::Float64 )
    Four_π * J_LW * σHD_LW / E_LW
end


""" H₂ photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_H2( J_EUV1::Float64,
                             J_EUV2::Float64,
                             J_EUV3::Float64
                              )
    Four_π * ( J_EUV1 * σH2_EUV1 / E_EUV1 
             + J_EUV2 * σH2_EUV2 / E_EUV2 
             + J_EUV3 * σH2_EUV3 / E_EUV3  )
end


""" H I photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_HI( J_EUV1::Float64, 
                             J_EUV2::Float64,
                             J_EUV3::Float64
                             )
    Four_π * ( J_EUV1 * σHI_EUV1 / E_EUV1 
             + J_EUV2 * σHI_EUV2 / E_EUV2 
             + J_EUV3 * σHI_EUV3 / E_EUV3  )
end


""" He I photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_HeI( J_EUV2::Float64,
                              J_EUV3::Float64
                             )
    Four_π * ( J_EUV2 * σHeI_EUV2 / E_EUV2 
             + J_EUV3 * σHeI_EUV3 / E_EUV3  )
end


""" He II photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_HeII( J_EUV3::Float64 )
    Four_π * J_EUV3 * σHeII_EUV3 / E_EUV3 
end


""" O I photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_OI( J_EUV1::Float64, 
                             J_EUV2::Float64,
                             J_EUV3::Float64
                             )
    Four_π * ( J_EUV1 * σOI_EUV1 / E_EUV1 
             + J_EUV2 * σOI_EUV2 / E_EUV2 
             + J_EUV3 * σOI_EUV3 / E_EUV3  )
end


""" C I photoionization rate (s⁻¹) from absorptions of
    ionizing stellar photons. """
@inline function IonRate_CI( J_LW::Float64,
                           J_EUV1::Float64, 
                           J_EUV2::Float64,
                           J_EUV3::Float64
                             )
    Four_π * ( J_LW   * σCI_LW   / E_LW
             + J_EUV1 * σCI_EUV1 / E_EUV1 
             + J_EUV2 * σCI_EUV2 / E_EUV2 
             + J_EUV3 * σCI_EUV3 / E_EUV3  )
end


""" The Strömgren radius (cm) for a star of mass mstar (M⊙)
    in a cloud of hydrogen number density nH (cm⁻³). """
@inline function StromGrenRadius( mstar::Float64, 
                                     nH::Float64,
                                      T::Float64 = 1e4,
                                  CaseB::Bool    = true,
                             Zstar_Zsun::Float64 = 1.0
                                 )
    dNdt_LyC = dNdt_EUV1(mstar, Zstar_Zsun) + 0.0*dNdt_EUV2(mstar, Zstar_Zsun) + 0.0*dNdt_EUV3(mstar, Zstar_Zsun)
    
    return ( 3 * dNdt_LyC / (Four_π * krecHII(T, CaseB) * (nH^2)) )^(1/3)
end

#println(StromGrenRadius(35.0, 1e4, 2.4466e4, true, 1.0)/pc)
