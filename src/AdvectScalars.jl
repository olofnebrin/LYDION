
""" MODULE:         AdvectScalars

    PURPOSE:        This module contains code for preparing passive scalars (e.g. ρx_HII),
                    and recovering scalars like xHII from the passive scalar ρx_HII. Note that
                    the advection of the passive scalars themselves takes place in the hydro update.
    
"""


using .Structs: HydrodynamicsState, ChemistryState, DustState
using .ConstantsAndFunctions
using  Base.Threads              # For CPU acceleration


export PreparePassiveScalars!, RecoverChemistryFromPassiveScalars!


const Zgas_floor::Float64 = 1e-10 * Zsun 

""" This function prepares all passive scalars for advection. """
@views function PreparePassiveScalars!( HydroState::HydrodynamicsState, 
                                         Chemistry::ChemistryState )

    # Unpack:
    ρ  = HydroState.ρ
    S  = HydroState.Scalars 

    # Initial values of H I, He I, D I, CHₓ, OHₓ are determined from conservation:
    @. Chemistry.xHI_adv  = max( 1 - Chemistry.xHII  - 2 * Chemistry.xH2, 0.0 )
    @. Chemistry.xHeI_adv = max( 1 - Chemistry.xHeII - Chemistry.xHeIII,  0.0 )
    @. Chemistry.xDI_adv  = max( 1 - Chemistry.xDII  - Chemistry.xHD,     0.0 )
    @. Chemistry.xCHx_adv = max( 1 - Chemistry.xCI   - Chemistry.xCII  -  Chemistry.xCO, 0.0 )
    @. Chemistry.xOHx_adv = max( 1 - Chemistry.xOI   - Chemistry.xOII  
                                 - Compute_nC_over_nO( Chemistry.ZC, Chemistry.Zgasrest ) * Chemistry.xCO, 
                                 0.0 )
    
    # Prepare passive scalars for advection in the hydro code:
    @. S[:,:,1]  = ρ * Chemistry.xHI_adv 
    @. S[:,:,2]  = ρ * Chemistry.xHII
    @. S[:,:,3]  = ρ * Chemistry.xH2 
    @. S[:,:,4]  = ρ * Chemistry.xHeI_adv
    @. S[:,:,5]  = ρ * Chemistry.xHeII
    @. S[:,:,6]  = ρ * Chemistry.xHeIII
    @. S[:,:,7]  = ρ * Chemistry.xDI_adv 
    @. S[:,:,8]  = ρ * Chemistry.xDII
    @. S[:,:,9]  = ρ * Chemistry.xHD 

    # Carbon species:
    @. S[:,:,10] = ρ * Chemistry.xCHx_adv * max( Chemistry.ZC,  Zgas_floor )
    @. S[:,:,11] = ρ * Chemistry.xCI      * max( Chemistry.ZC,  Zgas_floor )
    @. S[:,:,12] = ρ * Chemistry.xCII     * max( Chemistry.ZC,  Zgas_floor )
    @. S[:,:,13] = ρ * Chemistry.xCO      * max( Chemistry.ZC,  Zgas_floor )
    # Oxygen species:
    @. S[:,:,14] = ρ * Chemistry.xOHx_adv * max( Chemistry.Zgasrest,  Zgas_floor )
    @. S[:,:,15] = ρ * Chemistry.xOI      * max( Chemistry.Zgasrest,  Zgas_floor )
    @. S[:,:,16] = ρ * Chemistry.xOII     * max( Chemistry.Zgasrest,  Zgas_floor )

    # We advect carbon metallicity (which controls carbonaceous dust growth), silicate 
    # metallicity (controls silicate dust growth), and the rest of the metallicity:
    @. S[:,:,17] = ρ * max( Chemistry.ZC,       Zgas_floor )
    @. S[:,:,18] = ρ * max( Chemistry.ZSil,     Zgas_floor )
    @. S[:,:,19] = ρ * max( Chemistry.Zgasrest, Zgas_floor )

    return nothing 
end


""" This function recovers all passive scalars after advection. """
function RecoverChemistryFromPassiveScalars!( HydroState::HydrodynamicsState, 
                                               Chemistry::ChemistryState)

    # Unpack:
    ρ  = HydroState.ρ
    S  = HydroState.Scalars 

    # Grid size:
    N_r, N_Z = size( ρ )
    dN_r     = 1:N_r 
    dN_Z     = 1:N_Z 

    @threads for j in dN_Z 
        @inbounds @simd for i in dN_r 
            # Convenient variable:
            inv_ρ  = 1 / max( ρ[i,j], ρ_floor )

            # Recover gas metallicity (for carbon, silicate, and rest):
            Chemistry.ZC[i,j]       = max( S[i,j,17] * inv_ρ,  Zgas_floor )
            Chemistry.ZSil[i,j]     = max( S[i,j,18] * inv_ρ,  Zgas_floor )
            Chemistry.Zgasrest[i,j] = max( S[i,j,19] * inv_ρ,  Zgas_floor )

            # Convenient variables:
            inv_ZC              = 1 / Chemistry.ZC[i,j]
            inv_Zgasrest        = 1 / Chemistry.Zgasrest[i,j]

            # Recover preliminary abundance fractions:
            Chemistry.xHI_adv[i,j]   = S[i,j,1]  * inv_ρ
            Chemistry.xHII[i,j]      = S[i,j,2]  * inv_ρ
            Chemistry.xH2[i,j]       = S[i,j,3]  * inv_ρ
            Chemistry.xHeI_adv[i,j]  = S[i,j,4]  * inv_ρ
            Chemistry.xHeII[i,j]     = S[i,j,5]  * inv_ρ
            Chemistry.xHeIII[i,j]    = S[i,j,6]  * inv_ρ
            Chemistry.xDI_adv[i,j]   = S[i,j,7]  * inv_ρ
            Chemistry.xDII[i,j]      = S[i,j,8]  * inv_ρ
            Chemistry.xHD[i,j]       = S[i,j,9]  * inv_ρ
            Chemistry.xCHx_adv[i,j]  = S[i,j,10] * inv_ρ * inv_ZC
            Chemistry.xCI[i,j]       = S[i,j,11] * inv_ρ * inv_ZC
            Chemistry.xCII[i,j]      = S[i,j,12] * inv_ρ * inv_ZC
            Chemistry.xCO[i,j]       = S[i,j,13] * inv_ρ * inv_ZC
            Chemistry.xOHx_adv[i,j]  = S[i,j,14] * inv_ρ * inv_Zgasrest
            Chemistry.xOI[i,j]       = S[i,j,15] * inv_ρ * inv_Zgasrest
            Chemistry.xOII[i,j]      = S[i,j,16] * inv_ρ * inv_Zgasrest

            ######## Ensure machine-precision normalization ######################
            inv_xHtot    = 1 / ( Chemistry.xHI_adv[i,j]  +  Chemistry.xHII[i,j]   +  2 * Chemistry.xH2[i,j] )
            inv_xHetot   = 1 / ( Chemistry.xHeI_adv[i,j] +  Chemistry.xHeII[i,j]  +  Chemistry.xHeIII[i,j]  )
            inv_xDtot    = 1 / ( Chemistry.xDI_adv[i,j]  +  Chemistry.xDII[i,j]   +  Chemistry.xHD[i,j]     )
            inv_xCtot    = 1 / ( Chemistry.xCHx_adv[i,j] +  Chemistry.xCI[i,j]    +  Chemistry.xCII[i,j]  +  Chemistry.xCO[i,j] )

            # Safety checks:
            @assert isfinite(inv_xHtot) """ Non-finite inv_xHtot in cell (i,j)=($i,$j)
                
            xHI_adv  = $(Chemistry.xHI_adv[i,j])
            xHII     = $(Chemistry.xHII[i,j])
            xH2      = $(Chemistry.xH2[i,j])
                
            Occurred after gas fraction advection, but before renormalization. """

            @assert isfinite(inv_xHetot) """ Non-finite inv_xHetot in cell (i,j)=($i,$j)
                
            xHeI_adv  = $(Chemistry.xHeI_adv[i,j])
            xHeII     = $(Chemistry.xHeII[i,j])
            xHeIII    = $(Chemistry.xHeIII[i,j])
                
            Occurred after gas fraction advection, but before renormalization. """

            @assert isfinite(inv_xDtot) """ Non-finite inv_xDtot in cell (i,j)=($i,$j)
                
            xDI_adv  = $(Chemistry.xDI_adv[i,j])
            xDII     = $(Chemistry.xDII[i,j])
            xHD      = $(Chemistry.xHD[i,j])
                
            Occurred after gas fraction advection, but before renormalization. """

            @assert isfinite(inv_xCtot) """ Non-finite inv_xCtot in cell (i,j)=($i,$j)
                
            xCHx_adv  = $(Chemistry.xCHx_adv[i,j])
            xCI       = $(Chemistry.xCI[i,j])
            xCII      = $(Chemistry.xCII[i,j])
            xCO       = $(Chemistry.xCO[i,j])
            ZC        = $(Chemistry.ZC[i,j])
                
            Occurred after gas fraction advection, but before renormalization. """


            # Normalize hydrogen species (H I is not needed after advection):
            Chemistry.xHII[i,j]   = Chemistry.xHII[i,j]   * inv_xHtot
            Chemistry.xH2[i,j]    = Chemistry.xH2[i,j]    * inv_xHtot
            # Normalize helium species (He I is not needed after advection):
            Chemistry.xHeII[i,j]  = Chemistry.xHeII[i,j]  * inv_xHetot
            Chemistry.xHeIII[i,j] = Chemistry.xHeIII[i,j] * inv_xHetot
            # Normalize deuterium species (D I is not needed after advection):
            Chemistry.xDII[i,j]   = Chemistry.xDII[i,j]   * inv_xDtot
            Chemistry.xHD[i,j]    = Chemistry.xHD[i,j]    * inv_xDtot

            # Normalize carbon species (CHₓ is not needed after advection):
            Chemistry.xCI[i,j]    = Chemistry.xCI[i,j]    * inv_xCtot
            Chemistry.xCII[i,j]   = Chemistry.xCII[i,j]   * inv_xCtot
            Chemistry.xCO[i,j]    = Chemistry.xCO[i,j]    * inv_xCtot

            # Normalize oxygen species (OHₓ is not needed after advection, since it's 
            # obtained from conservation), in such a way that the CO abundance 
            # remain consistent/unchanged:
            nC_over_nO =  Compute_nC_over_nO( Chemistry.ZC[i,j], Chemistry.Zgasrest[i,j] )
            f_renorm   =  ( max( ( 1 - nC_over_nO * Chemistry.xCO[i,j] ), 0.0 ) 
                          / ( Chemistry.xOHx_adv[i,j]  +  Chemistry.xOI[i,j]  +  Chemistry.xOII[i,j] ) )
            
            Chemistry.xOI[i,j]    = Chemistry.xOI[i,j]  * f_renorm
            Chemistry.xOII[i,j]   = Chemistry.xOII[i,j] * f_renorm

        end # End of i-loop 
    end # End of j-loop 

    return nothing
end


