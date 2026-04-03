
""" FILE:       StellarWinds.jl

    PURPOSE:    Contains everything needed for modelling stellar wind feedback.  

    REFERENCES: The fit to the wind mass loss rate, and wind velocity, is taken from:

                (1) Deng et al. (2024) 'RIGEL: Simulating dwarf galaxies at solar mass resolution with
                radiative transfer and feedback from individual massive stars', A&A, 691, A231
                (see related data link: https://bitbucket.org/ftldengyw/rigel-tables/src/main/)

"""

using  DelimitedFiles          # For reading data from files
using .Structs: HydrodynamicsState, GeometryCylindrical

export dmdt_wind, u_wind, dpdt_wind, InjectStellarWind!


dmdtwindData  = readdlm( joinpath(@__DIR__, "RIGELfits/Mw_W.txt") )  # Data for fitting the wind mass loss rate (in M⊙/yr)
uwindData     = readdlm( joinpath(@__DIR__, "RIGELfits/Vw_V.txt") )  # Data for fitting wind velocity (in km/s)


""" The stellar wind mass loss rate (g s⁻¹) for a star of ZAMS mass mstarZAMS (M⊙)
    and metallicity Zstar_Zsun (in Solar units). """
function dmdt_wind( mstarZAMS::Float64, 
                   Zstar_Zsun::Float64 )

    x = log10( mstarZAMS )
    if Zstar_Zsun < 1e-6
        # We ignore stellar winds at these low metallicities
        return 0.0
    end

    # Limits on metallicity used in the fit:
    Zstar_Zsun = clamp( Zstar_Zsun, 1e-6, 1.0 )

    # Metallicity bin centers:
    Zbins   = Float64.( dmdtwindData[2, 1:5] )
    # Find where the input Zstar_Zsun lies:
    indxlow = findlast(Z -> Z ≤ Zstar_Zsun, Zbins)
    indxup  = findfirst(Z -> Z ≥ Zstar_Zsun, Zbins)
    Zlow    = Zbins[indxlow]
    Zup     = Zbins[indxup]
    # Get log10(x0) for the lower & upper limits:
    x0low   = log10( dmdtwindData[3,indxlow] )
    x0up    = log10( dmdtwindData[3,indxup] )

    # Compute log10(A) for lower & upper limits:
    b0low   = dmdtwindData[3+indxlow,1]
    b1low   = dmdtwindData[3+indxlow,2] 
    b2low   = dmdtwindData[3+indxlow,3]
    b3low   = dmdtwindData[3+indxlow,4]
    b4low   = dmdtwindData[3+indxlow,5]

    b0up    = dmdtwindData[3+indxup,1]
    b1up    = dmdtwindData[3+indxup,2] 
    b2up    = dmdtwindData[3+indxup,3]
    b3up    = dmdtwindData[3+indxup,4]
    b4up    = dmdtwindData[3+indxup,5]

    # Copute log10(A) for lower & upper limits
    if x ≤ x0low
        log10_Alow = b0low + b1low*x + b2low*(x^2) + b3low*(x^3) + b4low*(x^4)
    else
        log10_Alow = ( b0low + b1low*x0low + b2low*(x0low^2) + b3low*(x0low^3) + b4low*(x0low^4)  
                   +  (b1low + 2*b2low*x0low + 3*b3low*(x0low^2) + 4*b4low*(x0low^3))*(x - x0low) )
    end

    if Zlow == Zup 
        # In this case we don't need interpolation, and are finished:
        return ( 10^log10_Alow ) * Msun/yr
    end

    if x ≤ x0up
        log10_Aup = b0up + b1up*x + b2up*(x^2) + b3up*(x^3) 
    else
        log10_Aup = ( b0up + b1up*x0up + b2up*(x0up^2) + b3up*(x0up^3)  
                  +  (b1up + 2*b2up*x0up + 3*b3up*(x0up^2) + 4*b4up*(x0up^3))*(x - x0up) )
    end

    # Use linear interpolation between them:
    log10_A = log10_Alow + ((log10_Aup - log10_Alow) / log10(Zup/Zlow)) * log10(Zstar_Zsun/Zlow)

    dmdtwind = ( 10^log10_A ) * Msun/yr
    return dmdtwind 
end


""" The stellar wind velocity (cm s⁻¹) for a star of ZAMS mass mstarZAMS (M⊙)
    and metallicity Zstar_Zsun (in Solar units). """
function u_wind( mstarZAMS::Float64, 
                Zstar_Zsun::Float64 )

    x = log10( mstarZAMS )
    if Zstar_Zsun < 1e-6
        # We ignore stellar winds at these low metallicities
        return 0.0
    end

    # Limits on metallicity used in the fit:
    Zstar_Zsun = clamp( Zstar_Zsun, 1e-6, 1.0 )

    # Metallicity bin centers:
    Zbins   = Float64.( uwindData[2, 1:5] )
    # Find where the input Zstar_Zsun lies:
    indxlow = findlast(Z -> Z ≤ Zstar_Zsun, Zbins)
    indxup  = findfirst(Z -> Z ≥ Zstar_Zsun, Zbins)
    Zlow    = Zbins[indxlow]
    Zup     = Zbins[indxup]
    # Get log10(x0) for the lower & upper limits:
    x0low   = log10( uwindData[3,indxlow] )
    x0up    = log10( uwindData[3,indxup] )

    # Compute log10(A) for lower & upper limits:
    b0low   = uwindData[3+indxlow,1]
    b1low   = uwindData[3+indxlow,2] 
    b2low   = uwindData[3+indxlow,3]
    b3low   = uwindData[3+indxlow,4]
    b4low   = uwindData[3+indxlow,5]

    b0up    = uwindData[3+indxup,1]
    b1up    = uwindData[3+indxup,2] 
    b2up    = uwindData[3+indxup,3]
    b3up    = uwindData[3+indxup,4]
    b4up    = uwindData[3+indxup,5]

    # Copute log10(A) for lower & upper limits
    if x ≤ x0low
        log10_Alow = b0low + b1low*x + b2low*(x^2) + b3low*(x^3) + b4low*(x^4)
    else
        log10_Alow = ( b0low + b1low*x0low + b2low*(x0low^2) + b3low*(x0low^3) + b4low*(x0low^4)  
                   +  (b1low + 2*b2low*x0low + 3*b3low*(x0low^2) + 4*b4low*(x0low^3))*(x - x0low) )
    end

    if Zlow == Zup 
        # In this case we don't need interpolation, and are finished:
        return ( 10^log10_Alow ) * kms
    end

    if x ≤ x0up
        log10_Aup = b0up + b1up*x + b2up*(x^2) + b3up*(x^3) 
    else
        log10_Aup = ( b0up + b1up*x0up + b2up*(x0up^2) + b3up*(x0up^3)  
                  +  (b1up + 2*b2up*x0up + 3*b3up*(x0up^2) + 4*b4up*(x0up^3))*(x - x0up) )
    end

    # Use linear interpolation between them:
    log10_A = log10_Alow + ((log10_Aup - log10_Alow) / log10(Zup/Zlow)) * log10(Zstar_Zsun/Zlow)

    uwind = ( 10^log10_A ) * kms
    return uwind  
end


""" The momentum injection rate (cgs units) from stellar winds, from a star of
    ZAMS mass mstarZAMS (M⊙) and metallicity Zstar_Zsun (in Solar units). """
@inline function dpdt_wind( mstarZAMS::Float64, 
                           Zstar_Zsun::Float64 )

    return dmdt_wind(mstarZAMS, Zstar_Zsun) * u_wind(mstarZAMS, Zstar_Zsun)
end


""" This function injects the stellar wind momentum & mass over a time-step Δt. """
function InjectStellarWind!( Δt::Float64, 
                      mstarZAMS::Float64, 
                     Zstar_Zsun::Float64,
                          Hydro::HydrodynamicsState,
                       Geometry::GeometryCylindrical,
                         ρ_star::AbstractMatrix{Float64},
                             Zc::Float64 = 0.0 )

    # Unpack arrays:
    ρ      = Hydro.ρ
    ρu_r   = Hydro.ρu_X
    ρu_Z   = Hydro.ρu_Y 
    E      = Hydro.E 
    rCoM   = Geometry.rCoM
    ZCoM   = Geometry.ZCoM
    ΔV     = Geometry.ΔV 
    inv_ΔV = Geometry.inv_ΔV 

    # Grid:
    N_r, N_Z = size( ΔV )
    dN_r     = 1:N_r 
    dN_Z     = 1:N_Z

    # Convenient constant:
    inv_mstarZAMS_cgs = 1 / ( mstarZAMS * Msun )

    # Mass and momentum from the stellar wind to be dumped into the gas
    # over the time-step Δt:
    Δm_wind = dmdt_wind( mstarZAMS, Zstar_Zsun ) * Δt 
    Δp_wind = Δm_wind * u_wind( mstarZAMS, Zstar_Zsun )
    
    @threads for j in dN_Z 
        @inbounds @simd for i in dN_r
            # Distance from position of the star, located ar (r,Z) = (0, Zc):
            δZ         = ZCoM[j] - Zc
            d_star     = hypot( rCoM[i], δZ )

            # Fraction of star's mass in this cell:
            f_ij       = ρ_star[i,j] * ΔV[i,j] * inv_mstarZAMS_cgs

            # Stellar wind mass & momentum to be added to the cell:
            Δm_ij      = f_ij * Δm_wind 
            Δp_ij      = f_ij * Δp_wind 

            # Corresponding mass & momentum densities:
            Δρ_ij      = Δm_ij * inv_ΔV[i,j]
            Δρu_ij     = Δp_ij * inv_ΔV[i,j]

            # Get r & Z unit vectors pointing from the star's position:
            inv_d_star = 1 / d_star
            nr         = rCoM[i] * inv_d_star
            nZ         = δZ      * inv_d_star

            # Change in momentum in r & Z-directions:
            Δρu_r      = Δρu_ij * nr
            Δρu_Z      = Δρu_ij * nZ

            # Compute internal energy density, consistent with E:
            e_int      = max( E[i,j] - ( ρu_r[i,j] * ρu_r[i,j]  +  ρu_Z[i,j] * ρu_Z[i,j] ) / ( 2 * ρ[i,j] ), 
                              e_dual_floor )

            # Add momentum density:
            ρu_r[i,j] += Δρu_r
            ρu_Z[i,j] += Δρu_Z

            # Update gas density:
            ρ[i,j]    += Δρ_ij

            # Update gas energy, keeping internal energy density the same:
            E[i,j]     = e_int  +  ( ρu_r[i,j] * ρu_r[i,j]  +  ρu_Z[i,j] * ρu_Z[i,j] ) / ( 2 * ρ[i,j] )
        end
    end

    return nothing 
end


