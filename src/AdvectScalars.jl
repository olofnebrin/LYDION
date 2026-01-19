
""" MODULE:         AdvectScalars

    PURPOSE:        This module contains code for preparing passive scalars (e.g. ρx_HII),
                    and recovering scalars like xHII from the passive scalar ρx_HII. Note that
                    the advection of the passive scalars themselves takes place in the hydro update.
    
"""


using .Structs: HydrodynamicsState, ChemistryState, DustState
using .ConstantsAndFunctions
using  Base.Threads              # For CPU acceleration


export PreparePassiveScalars!, RecoverChemistryFromPassiveScalars!


const C_over_O::Float64 = C_H_ratio / O_H_ratio  # nC / nO

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
    @. Chemistry.xOHx_adv = max( 1 - Chemistry.xOI   - Chemistry.xOII  -  C_over_O * Chemistry.xCO, 0.0 )
    
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
    @. S[:,:,10] = ρ * Chemistry.xCHx_adv * Chemistry.Zgas_Zsun
    @. S[:,:,11] = ρ * Chemistry.xCI      * Chemistry.Zgas_Zsun
    @. S[:,:,12] = ρ * Chemistry.xCII     * Chemistry.Zgas_Zsun
    @. S[:,:,13] = ρ * Chemistry.xCO      * Chemistry.Zgas_Zsun
    @. S[:,:,14] = ρ * Chemistry.xOHx_adv * Chemistry.Zgas_Zsun
    @. S[:,:,15] = ρ * Chemistry.xOI      * Chemistry.Zgas_Zsun
    @. S[:,:,16] = ρ * Chemistry.xOII     * Chemistry.Zgas_Zsun
    @. S[:,:,17] = ρ * Chemistry.Zgas_Zsun

    return nothing 
end



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
            inv_ρ        = 1 / max( ρ[i,j], ρ_floor )

            # Recover gas metallicity:
            Chemistry.Zgas_Zsun[i,j] = S[i,j,17] * inv_ρ

            # Another convenient variable:
            inv_ZgasZsun = 1 / max( Chemistry.Zgas_Zsun[i,j], 1e-10 )

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
            Chemistry.xCHx_adv[i,j]  = S[i,j,10] * inv_ρ * inv_ZgasZsun
            Chemistry.xCI[i,j]       = S[i,j,11] * inv_ρ * inv_ZgasZsun
            Chemistry.xCII[i,j]      = S[i,j,12] * inv_ρ * inv_ZgasZsun
            Chemistry.xCO[i,j]       = S[i,j,13] * inv_ρ * inv_ZgasZsun
            Chemistry.xOHx_adv[i,j]  = S[i,j,14] * inv_ρ * inv_ZgasZsun
            Chemistry.xOI[i,j]       = S[i,j,15] * inv_ρ * inv_ZgasZsun
            Chemistry.xOII[i,j]      = S[i,j,16] * inv_ρ * inv_ZgasZsun

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
            f_renorm =  ( max( ( 1 - C_over_O * Chemistry.xCO[i,j] ), 0.0 ) 
                        / ( Chemistry.xOHx_adv[i,j] +  Chemistry.xOI[i,j]  +  Chemistry.xOII[i,j] ) )
            
            Chemistry.xOI[i,j]    = Chemistry.xOI[i,j]  * f_renorm
            Chemistry.xOII[i,j]   = Chemistry.xOII[i,j] * f_renorm

        end
    end

    return nothing
end


