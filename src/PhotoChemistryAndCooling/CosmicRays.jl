module CosmicRays

""" MODULE:         CosmicRays

    PURPOSE:        This module contains everything related to cosmic ray ionization and heating.
             
    REFERENCES:     (1) Gong et al. (2017) 'A Simple and Accurate Network for Hydrogen and 
                    Carbon Chemistry in the Interstellar Medium', ApJ, 843, 1
    
                    (2) Deng et al. (2024) 'RIGEL: Simulating dwarf galaxies at solar mass resolution 
                    with radiative transfer and feedback from individual massive stars', Astronomy & Astrophysics, 
                    691, id.A231
    
    """

__precompile__()

const ζHI_MW::Float64 = 2e-16   # Cosmic ray ionizaton rate of H I in the Milky Way (see Gong et al. 2017)

export ζHI_MW, ζH2, ζHeI, ζCI, CosmicRayHeating


""" Cosmic ray ionization rate (s⁻¹) of H₂, 
    as a function of the H I ionizaton rate. """
@inline function ζH2(ζHI::Float64)
    2 * ζHI
end


""" Cosmic ray ionization rate (s⁻¹) of He I, 
    as a function of the H I ionizaton rate. """
@inline function ζHeI(ζHI::Float64)
    1.1 * ζHI
end


""" Cosmic ray ionization rate (s⁻¹) of C I, 
    as a function of the H I ionizaton rate. """
@inline function ζCI(ζHI::Float64)
    3.85 * ζHI
end


""" Cosmic ray heating rate (erg cm⁻³ s⁻¹).
    See Deng et al. (2024). """
@inline function CosmicRayHeating( ζHI::Float64,
                                   nHI::Float64,
                                   nH2::Float64,
                                  nHeI::Float64,
                                   nCI::Float64
                                  )
    1.602e-11 * (ζHI * nHI  +  ζH2(ζHI) * nH2  +  ζHeI(ζHI) * nHeI  +  ζCI(ζHI) * nCI)
end


end # End of module 