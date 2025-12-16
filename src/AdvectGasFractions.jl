
""" FILE:           AdvectGasFractions.jl

    PURPOSE:        This module contains the basic code for 
                    advecting all gas fractions, e.g. x_HI.
    
"""


using  Base.Threads                  # For CPU acceleration

export AdvectGasFractions!


function AdvectGasFractions!( Δt::Float64,
                       chemstate::ChemistryState{Float64},
                      hydrostate::HydrodynamicsState{Float64},
                        geometry::GeometryCylindrical{Float64}
                              )

    # Unpack gas fractions:
    xHI_adv   = chemstate.xHI_adv;   xHI_adv_old   = chemstate.xHI_adv_old
    xHII      = chemstate.xHII;      xHII_old      = chemstate.xHII_old 
    xH2       = chemstate.xH2;       xH2_old       = chemstate.xH2_old
    xHeI_adv  = chemstate.xHeI_adv;  xHeI_adv_old  = chemstate.xHeI_adv_old 
    xHeII     = chemstate.xHeII;     xHeII_old     = chemstate.xHeII_old
    xHeIII    = chemstate.xHeIII;    xHeIII_old    = chemstate.xHeIII_old
    xDI_adv   = chemstate.xDI_adv;   xDI_adv_old   = chemstate.xDI_adv_old
    xDII      = chemstate.xDII;      xDII_old      = chemstate.xDII_old
    xHD       = chemstate.xHD;       xHD_old       = chemstate.xHD_old

    # Grid size & intervals: 
    N_r, N_Z  = size( xHII )
    dN_r      = 1:N_r 
    dN_Z      = 1:N_Z

    # Initialize the H I, He I, D I fractions:
    xHI_adv  .= max.( 1.0  .-  xHII   .-  2 .* xH2, 0.0 )
    xHeI_adv .= max.( 1.0  .-  xHeII  .-  xHeIII,   0.0 )
    xDI_adv  .= max.( 1.0  .-  xDII   .-  xHD,      0.0 )

    # Advect all gas fractions:
    species      = ( xHI_adv,  xHII,  xH2,
                     xHeI_adv, xHeII, xHeIII,
                     xDI_adv,  xDII,  xHD )

    species_old  = ( xHI_adv_old,  xHII_old,  xH2_old,
                     xHeI_adv_old, xHeII_old, xHeIII_old,
                     xDI_adv_old,  xDII_old,  xHD_old )

    Nspecies     = length( species )

    for k in 1:Nspecies
        AdvectScalar!( Δt, species[k], species_old[k], hydrostate.ρ, hydrostate.F_ρ_X, hydrostate.F_ρ_Y,
                       geometry.inv_ΔV, geometry.A_r, geometry.A_Z, true )
    end

    # Re-normalize fractions to ensure they add up to 1 exactly:
    @threads for j in dN_Z 
        @simd for i in dN_r
            @inbounds begin
                # Normalization factors for all species: 
                inv_xHtot   = 1 / ( xHI_adv[i,j]  +  xHII[i,j]   +  2 * xH2[i,j] )
                inv_xHetot  = 1 / ( xHeI_adv[i,j] +  xHeII[i,j]  +  xHeIII[i,j]  )
                inv_xDtot   = 1 / ( xDI_adv[i,j]  +  xDII[i,j]   +  xHD[i,j]     )

                # Safety checks:
                @assert isfinite(inv_xHtot) """ Non-finite inv_xHtot in cell (i,j)=($i,$j)
                
                xHI_adv  = $(xHI_adv[i,j])
                xHII     = $(xHII[i,j])
                xH2      = $(xH2[i,j])
                
                Occurred after gas fraction advection, but before renormalization. """

                @assert isfinite(inv_xHetot) """ Non-finite inv_xHetot in cell (i,j)=($i,$j)
                
                xHeI_adv  = $(xHeI_adv[i,j])
                xHeII     = $(xHeII[i,j])
                xHeIII    = $(xHeIII[i,j])
                
                Occurred after gas fraction advection, but before renormalization. """

                @assert isfinite(inv_xDtot) """ Non-finite inv_xDtot in cell (i,j)=($i,$j)
                
                xDI_adv  = $(xDI_adv[i,j])
                xDII     = $(xDII[i,j])
                xHD      = $(xHD[i,j])
                
                Occurred after gas fraction advection, but before renormalization. """


                # Normalize hydrogen species:
                xHII[i,j]   = xHII[i,j]   * inv_xHtot
                xH2[i,j]    = xH2[i,j]    * inv_xHtot
                # Normalize helium species:
                xHeII[i,j]  = xHeII[i,j]  * inv_xHetot
                xHeIII[i,j] = xHeIII[i,j] * inv_xHetot
                # Normalize deuterium species:
                xDII[i,j]   = xDII[i,j]   * inv_xDtot
                xHD[i,j]    = xHD[i,j]    * inv_xDtot

            end # End of inbounds 
        end # End of i-loop 
    end # End of j-loop 

    return nothing 
end

