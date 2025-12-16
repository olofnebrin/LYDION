
module ConstantsAndFunctions
""" MODULE:  ConstantsAndFunctions 

    PURPOSE: Contains useful constants and units (e.g. c, νLyα, Lsun), 
             and functions (e.g. Voigt profile, optical depth, analytical
             spectra to compare to). 

"""

__precompile__()

using SpecialFunctions  # For Dawson integral
using Base.Threads      # For CPU acceleration
using Roots             # Needed for root-finding in the frequency grid setup 

export LyαArrays, GenerateLyαArrays, H, Htophat, b, a, σ0, 
       Update_Hvoigt!, Update_ϕν!, Update_e_Lyα!,
       AbsorptionUpdateJorH, LyαEmissionUpdate!, LyαAbsorptionUpdate!, Update_f_Lyα_MF_L_Lyα!,
       Reset_Δρu!, AbsorptionUpdateH, Update_∇u_RHD!, D_colon_∇u_Cylindrical, 
       D_colon_∇u_Cartesian, GetHydrogenDensities, GetHeliumDensities, GetElectronDensity,
       GetElectronDensityFromFracs, MinMod


###############################################################
#######                                                 #######
#######               C O N S T A N T S                 #######
#######                                                 #######
###############################################################

####### PHYSICAL CONSTANTS ################
const c::Float64          = 2.99792458e10         # Speed of light (cm/s)
const h::Float64          = 6.626069573e-27       # Planck's constant (erg s)
const G::Float64          = 6.6743e-8             # The gravitational constant (dyn cm² g⁻²)
const kB::Float64         = 1.380648813e-16       # Boltzmann constant (erg/K)
const mH::Float64         = 1.6735327e-24         # Mass of hydrogen atom (g)
const mH2::Float64        = 2 * mH                # Mass of H₂ molecule (g)
const me::Float64         = 9.109382917e-28       # Mass of electron (g)
const mHe::Float64        = 4 * mH                # Mass of helium atom (g)
const mC::Float64         = 12 * mH               # Mass of carbon atom (g)
const mCO::Float64        = (12 + 16) * mH        # Mass of CO molecule (g)
const e::Float64          = 4.80320451e-10        # Electron charge (esu)
const σSB::Float64        = 5.670374419e-5        # Stefan-Boltzmann constant (cgs units)
const νLyα::Float64       = 2.466068e15           # Lyα frequency at line center (Hz)
const νLyC::Float64       = 3.289842e15           # LyC limit frequency (Hz)
const ELyα::Float64       = h * νLyα              # Lyα photon energy (erg)
const ELyC::Float64       = h * νLyC              # LyC limit photon energy (i.e. Rydberg) (erg)
const ALyα::Float64       = 6.265e8               # Einstein-A coefficient for Lyα (s⁻¹)
const A_2γ::Float64       = 8.2206                # Einstein-A coefficient for two-photon decay (s⁻¹) 
const A_2p2s::Float64     = 8.78e-7               # Einstein-A coefficient for 2p → 2s transition (s⁻¹)
const A_2s2p::Float64     = 1.597e-9              # Einstein-A coefficient for 2s → 2p transition (s⁻¹)
const f_lu_Lyα::Float64   = 0.41641               # Oscillator strength for Lyα
const σT::Float64         = 6.6524587321e-25      # Thomson scattering cross-section (cm²)

####### PARAMETERS ########################
const Y::Float64          = 0.247                 # Helium mass fraction 
const X::Float64          = 1 - Y                 # Hydrogen mass fraction
const Dsun::Float64       = 1 / 162               # Solar dust-to-gas ratio
const TCMB0::Float64      = 2.726                 # CMB temperature at redshift z = 0
const D_H_ratio::Float64  = 2.53e-5               # Deuterium-to-hydrogen abundance nD/nH  (Cooke et al. 2018)
const C_H_ratio::Float64  = 2.95e-4               # Carbon-to-hydrogen abundance nC/nH     (Asplund et al. 2009)
const N_H_ratio::Float64  = 7.41e-5               # Nitrogen-to-hydrogen abundance nN/nH   (Asplund et al. 2009)
const O_H_ratio::Float64  = 5.37e-4               # Oxygen-to-hydrogen abundance nC/nH     (Asplund et al. 2009)
const Ne_H_ratio::Float64 = 9.33e-5               # Neon-to-hydrogen abundance nNe/nH      (Asplund et al. 2009)
const Mg_H_ratio::Float64 = 4.37e-5               # Magnesium-to-hydrogen abundance nMg/nH (Asplund et al. 2009)
const Si_H_ratio::Float64 = 3.55e-5               # Silicon-to-hydrogen abundance nSi/nH   (Asplund et al. 2009)
const S_H_ratio::Float64  = 1.45e-5               # Sulfur-to-hydrogen abundance nSi/nH    (Asplund et al. 2009)
const Fe_H_ratio::Float64 = 3.47e-5               # Iron-to-hydrogen abundance nSi/nH      (Asplund et al. 2009)

####### CONVENIENT UNIT CONVERSIONS #######
const Lsun::Float64       = 3.839e33              # Solar luminsoity (erg/s)
const Msun::Float64       = 1.988435e33           # Solar mass (g)
const AU::Float64         = 1.495978707e13        # 1 AU (cm)
const pc::Float64         = 3.085677581467192e18  # 1 parsec (cm)
const Mpc::Float64        = 1e6 * pc              # 1 Megaparsec (cm)
const Gpc::Float64        = 1e9 * pc              # 1 Gigaparsec (cm)
const kms::Float64        = 1e5                   # 1 km/s (cm/s)
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

####### CONVENIENT CONSTANTS ##############
const Four_π::Float64        = 4 * π 
const Two_π::Float64         = 2 * π
const Sqrt_π::Float64        = sqrt(π) 
const Sqrt_2::Float64        = sqrt(2)
const Four_π_over_c::Float64 = Four_π / c              

# Export constants:
export c, h, G, kB, mH, mH2, me, mHe, mC, mCO, e, σSB, νLyα, νLyC, ELyα, ELyC, ALyα, A_2γ, A_2p2s,
       A_2s2p, f_lu_Lyα, σT, X, Y, Dsun, TCMB0, D_H_ratio, C_H_ratio, N_H_ratio, O_H_ratio,
       Ne_H_ratio, Mg_H_ratio, Si_H_ratio, S_H_ratio, Fe_H_ratio,
       Lsun, Msun, AU, pc, Mpc, Gpc, kms, day, yr, Myr, Gyr, eV, Ang, nm, μm, F_Habing,
       Four_π, Two_π, Sqrt_π, Sqrt_2, Four_π_over_c


###############################################################
#######                                                 #######
#######         U S E F U L  F U N C T I O N S          #######
#######                                                 #######
###############################################################

""" The Voigt profile. Arguments:
    x:  Frequency displacement in units of Doppler shift.
    a:  The Lyα Voigt parameter, a ≃ 4.7 × 10⁻⁴ T₄^(-0.5)

    This function uses the expansion Eq. (12) in:
    Smith et al. (2018), 'Discrete diffusion Lyman α radiative transfer', MNRAS, 479, 2. """
@inline function H( x::Float64,
                    a::Float64
                   )
    xsq     = x^2
    EXP_xsq = exp(-xsq)  # exp(-x²)

    # Put it all together:
    H       = EXP_xsq + (2*a/Sqrt_π)*(2*x*dawson(x) - 1) + (a^2)*EXP_xsq*(1 - 2*xsq)
    return H
end


""" Integral ∫dx H(x,a) from x = 0 to x. 
    Uses Eq. (13) from:
    
    Smith et al. (2018), 'Discrete diffusion Lyman α radiative transfer', MNRAS, 479, 2. """
@inline function ∫dx_H( x::Float64, 
                        a::Float64
                       )
    (Sqrt_π/2) * erf(x) - (2*a/Sqrt_π) * dawson(x) + (a^2)*x*exp(-x^2)
end


const Δx_crit::Float64 = 1e-5

""" Top-hat x-average of H(x,a), between xmax and xmin around frequency bin x, 
    for a Voigt parameter a. See Eq. (14) in:

    Smith et al. (2018), 'Discrete diffusion Lyman α radiative transfer', MNRAS, 479, 2.  """
function Htophat( x::Float64,
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
@inline @fastmath function b( T::Float64 )
    b_PREFACT * sqrt( T )
end


const a_PREFACT::Float64 = ALyα / (Four_π * νLyα * 1/c)

""" Lyα Voigt parameter (dimensionless) as a function
    of temperature T (K). """
@inline @fastmath function a( T::Float64 )
    a_PREFACT / b(T)
end


const σ0_PREFACT::Float64 = Sqrt_π * (e^2) * f_lu_Lyα / ( me * νLyα )

""" Lyα cross-section at line center (cm²) as a function
    of temperature T (K). """
@inline @fastmath function σ0( T::Float64 )
    σ0_PREFACT / b(T)
end


const νLyα_over_c::Float64 = νLyα / c

""" Compute & update the (top-hat average) Voigt profile Hvoigt[i,j,k]
    for each cell. """
function Update_Hvoigt!( Hvoigt::AbstractArray{Float64, 3}, # Top-hat average Voigt profile for each cell (UPDATED)
                              ν::AbstractArray{Float64},   # Frequency interfaces 
                             νC::AbstractArray{Float64},   # Frequency bin centers
                              T::AbstractArray{Float64}    # Temperature in each cell
                         )

    N_r, N_Z, N_ν = size( Hvoigt ) # Grid size 
    dN_r = 1:N_r
    dN_Z = 1:N_Z
    dN_ν = 1:N_ν

    for k in dN_ν
        νmax  = ν[k+1]   # Right interface frequency 
        νmin  = ν[k]     # Left interface frequency
        
        @threads for j in dN_Z
            @inbounds begin 
                @simd for i in dN_r
                    # Doppler width:
                    Δν_D          = b(T[i,j]) * νLyα_over_c 

                    # Bin-averaged Voigt profile:
                    Hvoigt[i,j,k] = Htophat( (νC[k] - νLyα) / Δν_D, 
                                             (νmax  - νLyα) / Δν_D, 
                                             (νmin  - νLyα) / Δν_D,
                                             a( T[i,j] ) )

                end # End of i-loop
            end 
        end # End of j-loop
    end # End of k-loop

end


""" Compute & update the normalized emission profile ϕ_ν for each cell, 
    assuming ϕ_ν = Hvoigt / ∫dν Hvoigt ≈ Hvoigt[i,j,k] / ∑_k Δν[k] Hvoigt[i,j,k]. """
@views function Update_ϕν!( ϕ_ν::AbstractArray{Float64, 3},  # Normalized emission line profile (UPDATED)
                             Δν::AbstractArray{Float64, 1},  # Frequency bin widths
                         Hvoigt::AbstractArray{Float64, 3},  # Voigt profile for each cell
                           kmin::Int, 
                           kmax::Int 
                           )
    N_r, N_Z, N_ν = size( Hvoigt )
    dN_r = 1:N_r
    dN_Z = 1:N_Z
    dN_ν = 1:N_ν

    @threads for j in dN_Z
        @inbounds begin 
            @simd for i in dN_r

                SUM                 = sum( Δν[k] * Hvoigt[i,j, k] for k in kmin:kmax )
                ϕ_ν[i,j,kmin:kmax] .= Hvoigt[i,j,kmin:kmax] ./ SUM

            end # End of i-loop
        end # End of inbounds
    end # End of j-loop

end


const Four_π_c::Float64 = Four_π / c 

""" Update the Lyα energy density e_Lyα(r,Z) = 4π ∫dν J(r,Z,ν) / c. """
function Update_e_Lyα!( e_Lyα::AbstractArray{Float64, 2}, 
                           Δν::AbstractArray{Float64},
                            J::AbstractArray{Float64, 3}
                         )

    # Grid size & intervals:
    N_r, N_Z, N_ν = size( J )
    dN_r          = 1:N_r 
    dN_Z          = 1:N_Z 
    dN_ν          = 1:N_ν 

    for i in dN_r
        for j in dN_Z
            ∫dν_J = 0.0 # Initialize the estimated integral

            # Estimate the integral:
            @inbounds for k in dN_ν
                ∫dν_J += Δν[k] * J[i,j,k]
            end

            # Get Lyα energy density:
            @inbounds e_Lyα[i,j] = Four_π_c * ∫dν_J
        end
    end # End of for-loop

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
    (i.e. flux absorption). """
function LyαEmissionUpdate!( J::AbstractArray{Float64, 3},
                        js_Lyα::AbstractArray{Float64, 2},
                           ϕ_ν::AbstractArray{Float64, 3},
                             c̃::Float64,
                            Δt::Float64 )

    # Grid size & intervals:
    N_r, N_Z, N_ν  = size( J )   # Grid size 
    dN_r           = 1:N_r       # r-interval 
    dN_Z           = 1:N_Z       # Z-interval
    dN_ν           = 1:N_ν       # ν-interval

    @threads for k in dN_ν
        @inbounds @simd for j in dN_Z 
            for i in dN_r 
                # Update J:
                J[i,j,k] += js_Lyα[i,j] * ϕ_ν[i,j,k] * c̃ * Δt
            end # End of i-loop 
        end # End of j-loop 
    end # End of k-loop 

    return nothing 
end



""" This function updates J and H as a result of Lyα
    absorption of J (including by pd > 0), and absorption of H
    (i.e. flux absorption). It also computes the corresponding 
    change in gas momentum density from Lyα radiation pressure:

    Δρu = (4π/c) ∫dν ∫dt α_F * H 
        
        = (4π/c) (1/c̃) ∫dν (H_old - H)

    where we have used (1/c̃)(∂H/∂t) = - α_F H (for the absorption step). """
function LyαAbsorptionUpdate!( J_Lyα::AbstractArray{Float64, 3},
                             H_r_Lyα::AbstractArray{Float64, 3}, 
                             H_Z_Lyα::AbstractArray{Float64, 3},
                         H_r_old_Lyα::AbstractArray{Float64, 3},
                         H_Z_old_Lyα::AbstractArray{Float64, 3},
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

                # Flux opacity for H_r and H_Z:
                αabs_H = α_F_Lyα[i,j,k]

                # Update J:
                J_Lyα[i,j,k] = AbsorptionUpdateJorH( J_Lyα[i,j,k], Δt, 0.0, αabs_J, c̃ )

                # Store old (un-updated) H components:
                H_r_old_Lyα[i,j,k] = H_r_Lyα[i,j,k] 
                H_Z_old_Lyα[i,j,k] = H_Z_Lyα[i,j,k]

                # Candidate updates for H_r and H_Z:
                H_r_new  = AbsorptionUpdateJorH( H_r_old_Lyα[i,j,k], Δt, 0.0, αabs_H, c̃ )
                H_Z_new  = AbsorptionUpdateJorH( H_Z_old_Lyα[i,j,k], Δt, 0.0, αabs_H, c̃ )

                # Ensure that H ≤ J:
                H_mag = sqrt( H_r_new * H_r_new  +  H_Z_new * H_Z_new )
                if H_mag > J_Lyα[i,j,k]
                    f              = H_mag / J_Lyα[i,j,k]
                    H_r_Lyα[i,j,k] = H_r_new / f
                    H_Z_Lyα[i,j,k] = H_Z_new / f
                else
                    H_r_Lyα[i,j,k] = H_r_new
                    H_Z_Lyα[i,j,k] = H_Z_new
                end

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
            Δρu_Lyα_r[i,j] += Four_π_c * inv_c̃ * ∫dν_ΔH_r
            Δρu_Lyα_Z[i,j] += Four_π_c * inv_c̃ * ∫dν_ΔH_Z
        end
    end

    return nothing 
end


@inline function Update_f_Lyα_MF_L_Lyα!( f_Lyα_r::AbstractArray{Float64, 2}, 
                                         f_Lyα_Z::AbstractArray{Float64, 2},
                                       Δρu_Lyα_r::AbstractArray{Float64, 2},
                                       Δρu_Lyα_Z::AbstractArray{Float64, 2},
                                          ∫dν_js::AbstractArray{Float64, 2},
                                              ΔV::AbstractArray{Float64, 2},
                                              Δt::Float64,
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

    # Compute force multiplier M_F = ∫dV f_Lyα / (L_Lyα/c) & Lyα luminosity L_Lyα = 4π ∫dV (∫dν js):
    M_F   = 0.0
    L_Lyα = 0.0
    for j in dN_Z
        @inbounds @simd for i in dN_r 
            M_F   += ΔV[i,j] * sqrt( f_Lyα_r[i,j] * f_Lyα_r[i,j]  +  f_Lyα_Z[i,j] * f_Lyα_Z[i,j] )
            L_Lyα += Four_π * ΔV[i,j] * ∫dν_js[i,j]
        end
    end

    # Normalize M_F:
    if L_Lyα > 0.0
        M_F = M_F / (L_Lyα / c)
    else
        M_F = 0.0
    end

    # Compute the total Lyα luminosity in the cloud, *taking into account reflective BCs*:
    if BC_Z_bot === :Reflect 
        # If we have reflective BCs on the lower Z-plane, we double to get the total Lyα luminosity 
        # in the physical cloud:
        L_Lyα_tot = 2 * L_Lyα
    else
        L_Lyα_tot = L_Lyα
    end

    return M_F, L_Lyα_tot
end


""" This function just resets Δρu (for Lyα or stellar RT bands),
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


const X_over_mH::Float64 = X / mH

""" This function computes the H I, H II, and H₂ number densities,
    given the fractional abundances of H II and H₂, and the gas density ρ. """
@inline @fastmath function GetHydrogenDensities( xHII::Float64, 
                                                  xH2::Float64,
                                                    ρ::Float64 )

    nH    =   ρ * X_over_mH
    xHI   =   max( 1 - xHII - 2 * xH2, 0.0 )
    nHI   =   xHI    * nH
    nHII  =   xHII   * nH 
    nH2   =   xH2    * nH

    return nHI, nHII, nH2
end


const Y_over_mHe::Float64 = Y / mHe

""" This function computes the He I, He II, and He III number densities,
    given the fractional abundances of He II and He III, and the gas density ρ. """
@inline @fastmath function GetHeliumDensities( xHeII::Float64, 
                                              xHeIII::Float64,
                                                   ρ::Float64 )

    nHe     =   ρ * Y_over_mHe
    xHeI    =   max( 1 - xHeII - xHeIII, 0.0 )
    nHeI    =   xHeI   * nHe
    nHeII   =   xHeII  * nHe
    nHeIII  =   xHeIII * nHe

    return nHeI, nHeII, nHeIII
end


""" This function computes the electron number density 
    from the number densities of H II, He II, and He III. """
@inline @fastmath function GetElectronDensity( nHII::Float64, 
                                              nHeII::Float64, 
                                             nHeIII::Float64
                                              )

    return nHII  +  nHeII  +  2 * nHeIII
end


""" This function computes the electron number density 
    from the abundance fractions of H II, He II, and He III. """
@inline @fastmath function GetElectronDensityFromFracs( xHII::Float64, 
                                                       xHeII::Float64, 
                                                      xHeIII::Float64,
                                                           ρ::Float64
                                                        )

    nHII   = xHII * ρ * X_over_mH
    nHe    = ρ * Y_over_mHe
    nHeII  = xHeII  * nHe 
    nHeIII = xHeIII * nHe 

    return nHII  +  nHeII  +  2 * nHeIII
end


""" MinMod slope limiter. Used in the hydrodynamics, M1 RT, 
    and in the estimate for ∇ρ for the Sobolev-like 
    column density estimates. """
@inline function MinMod( A::Float64,
                         B::Float64 
                        )
    0.5 * (sign(A) + sign(B)) * min( abs(A), abs(B) )
end


end # End of module