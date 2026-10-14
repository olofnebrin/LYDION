module Structs 

# Stellar/IR RT band indices:
const iopt_RT::Int          = 1
const iFUV_RT::Int          = 2
const iLW_RT::Int           = 3
const iEUV1_RT::Int         = 4
const iEUV2_RT::Int         = 5
const iEUV3_RT::Int         = 6
const iIR_RT::Int           = 7
const Nbands_RT::Int        = 7
const dNbands_RT::UnitRange = 1:Nbands_RT


export LyαArrays, StellarRTArrays, HydrodynamicsState, ChemistryState,
       GhostPaddedGrid, StellarBandAverages, GeometryCylindrical, 
       GeometryCartesian, GravityState, GravityCoeffs, DustState,
       iopt_RT, iFUV_RT, iLW_RT, iEUV1_RT, iEUV2_RT, iEUV3_RT, iIR_RT, 
       Nbands_RT, dNbands_RT


""" A useful struct for storing all Lyα related arrays/quantities/settings. """
mutable struct LyαArrays{FluxMethod}
    # Lyα reduced speed of light:
    c̃_Lyα::Float64

    # Intensities (latest & old) for all bands: 
    J_Lyα::Array{Float64, 3};  J_Lyα_old::Array{Float64, 3}

    # Lyα energy density:
    e_Lyα::Array{Float64, 2}

    # H ≡ ∫dΩ I n̂ / 4π arrays
    H_r_Lyα::Array{Float64, 3};        H_Z_Lyα::Array{Float64, 3}
    H_r_Lyα_old::Array{Float64, 3};    H_Z_Lyα_old::Array{Float64, 3}

    # ϕϕ-component of radiation pressure tensor K = D × J at interfaces, 
    # needed for geometric source term. To reduce memory storage, we make them scratch 
    # arrays of size N_r × N_Z × nthread:
    K_ϕϕ_iphalf_Lyα::Array{Float64, 3};     K_ϕϕ_imhalf_Lyα::Array{Float64, 3}

    # Interface flux scratch arrays, of size of order N_r × N_Z × nthread: 
    F_r_Lyα::Array{Float64, 3};     F_Z_Lyα::Array{Float64, 3}
    F_r_Hr_Lyα::Array{Float64, 3};  F_Z_Hr_Lyα::Array{Float64, 3}
    F_r_HZ_Lyα::Array{Float64, 3};  F_Z_HZ_Lyα::Array{Float64, 3}

    # Lyα opacity at line center α = nHI × σ0:
    α_Lyα::Array{Float64, 2}

    # Flux opacity:
    α_F_Lyα::Array{Float64, 3}

    # Lyα continuum absorption opacity & destruction probability:
    αc_Lyα::Array{Float64, 2}
    pd::Array{Float64, 2}

    # (Frequency-integrated) emissivitiy, Voigt profile & normalized line profile:
    ∫dν_js_Lyα::Array{Float64, 2}
    Hvoigt::Array{Float64, 3}
    ϕ_ν::Array{Float64, 3}

    # Frequency diffusion + Doppler shift arrays:
    Fij_kp::Array{Float64, 3}
    Fij_k::Array{Float64, 3}
    Fij_km::Array{Float64, 3}

    # Frequency grid:
    ν::Array{Float64, 1}
    νC::Array{Float64, 1}
    Δν::Array{Float64, 1}

    # Store δν ≡ ν - νLyα and δνC ≡ νC - νLyα, which are needed in the Voigt parameter 
    # calculations:
    δν::Array{Float64, 1}
    δνC::Array{Float64, 1}

    # Also store inverses of νC and Δν, which are used in the frequency factors update:
    inv_Δν::Array{Float64, 1}
    inv_νC::Array{Float64, 1}

    # Voigt parameter and Doppler widths, and its inverse, for each cell:
    a::Array{Float64, 2}
    ΔνD::Array{Float64, 2}
    inv_ΔνD::Array{Float64, 2}

    # Bounds on the min and max bin indices to consider in M1 advection: 
    kcore_min::Int 
    kcore_max::Int 

    # Recoil setting (true if on, false if off):
    Recoil::Bool 

    # Lyα destruction (pd > 0) setting (true if on, false if off):
    LyαDestruction::Bool

    # Opacity floor (very small), set to avoid 0 in the frequency leakage coefficients 
    α_Lyα_floor::Float64 

    # Gas momentum change from Lyα radiation pressure, in r and Z-directions:
    Δρu_Lyα_r::Array{Float64, 2}
    Δρu_Lyα_Z::Array{Float64, 2}

    # Arrays for storing the Lyα radiation pressure force, in r and Z-directions:
    f_Lyα_r::Array{Float64, 2}
    f_Lyα_Z::Array{Float64, 2}

    # Force multiplier, Lyα luminosity, Lyα escape fraction:
    M_F::Float64
    M_F_radial_net::Float64        # Force multiplier radially outward (can be negative if force is radially inward)
    M_F_radial_out::Float64        # Force multiplier radially outward (only outward components counted) 
    M_F_antigravity::Float64       # Force multiplier anti-parallel to gravity 
    L_Lyα::Float64
    f_esc_Lyα::Float64

    # Save number of frequency bins N_ν, dN_ν = 1:N_ν, and dN_ν_int = 2:(N_ν-1),
    # which are used repeatedly:
    N_ν::Int 
    dN_ν::UnitRange{Int}
    dN_ν_int::UnitRange{Int}

    # Flux method for M1 transport, HLL or GLF:
    FluxMethod::Val{FluxMethod}

end


""" A struct for storing spectrum-averaged stellar band quantities, just for 
    H I (and D I), He I, He II. Other species (eq. H₂, O I, CHₓ) use 
    default values always for simplicity.

    The "_N" cross-sections are photon-number weighted and should be used
    for photoionization and photoheating rates.

    The "_E" cross-sections are energy-weighted and should be used for
    absorption/damping of J and H. """
struct StellarBandAverages
    # Mean photon energies in each stellar band:
    E_opt::Float64
    E_FUV::Float64
    E_LW::Float64
    E_EUV1::Float64
    E_EUV2::Float64
    E_EUV3::Float64

    # Inverse mean photon energies:
    inv_E_opt::Float64
    inv_E_FUV::Float64
    inv_E_LW::Float64
    inv_E_EUV1::Float64
    inv_E_EUV2::Float64
    inv_E_EUV3::Float64

    # Photon-number weighted cross-sections:
    σHI_EUV1_N::Float64
    σHI_EUV2_N::Float64
    σHI_EUV3_N::Float64

    σHeI_EUV2_N::Float64
    σHeI_EUV3_N::Float64
    σHeII_EUV3_N::Float64

    # Energy-weighted cross-sections:
    σHI_EUV1_E::Float64
    σHI_EUV2_E::Float64
    σHI_EUV3_E::Float64

    σHeI_EUV2_E::Float64
    σHeI_EUV3_E::Float64
    σHeII_EUV3_E::Float64

    # Photoheating excess energies:
    hHI_EUV1::Float64
    hHI_EUV2::Float64
    hHI_EUV3::Float64

    hHeI_EUV2::Float64
    hHeI_EUV3::Float64
    hHeII_EUV3::Float64
end


""" A struct for organizing stellar RT arrays, and the stellar mass density. """
mutable struct StellarRTArrays{FluxMethod}
    # Stellar RT reduced speed of light:
    c̃::Float64

    # Minimum RSL:
    c̃_min::Float64

    # Intensities (latest & old) for all bands, 
    # layout J[bin,i,j] where bin ∈ 1:7 is opt, FUV, 
    # LW, EUV1, EUV2, EUV3, IR: 
    J::Array{Float64, 3}
    J_old::Array{Float64, 3}

    # H ≡ ∫dΩ I n̂ / 4π arrays for all bands:
    H_r::Array{Float64, 3}
    H_Z::Array{Float64, 3}
    H_r_old::Array{Float64, 3}
    H_Z_old::Array{Float64, 3}

    # ϕϕ-component of radiation pressure tensor K = D × J at interfaces, 
    # needed for geometric source term:
    K_ϕϕ_iphalf::Array{Float64, 3}
    K_ϕϕ_imhalf::Array{Float64, 3}

    # Flux components for all bands: 
    F_r::Array{Float64, 3};     F_Z::Array{Float64, 3}
    F_r_Hr::Array{Float64, 3};  F_Z_Hr::Array{Float64, 3}
    F_r_HZ::Array{Float64, 3};  F_Z_HZ::Array{Float64, 3}

    # Flux opacities for all bands, for the asymptotic diffusion fix (only 
    # relevant for IR):
    α_F::Array{Float64, 3}

    # Luminosities in all bands, over infinite space (i.e. the full luminosity of the star):
    Lum_IR::Float64
    Lum_opt::Float64
    Lum_FUV::Float64
    Lum_LW::Float64
    Lum_EUV1::Float64
    Lum_EUV2::Float64
    Lum_EUV3::Float64
    Lum_bol::Float64    # Stellar bolometric luminosity, i.e. sum of the non-IR band luminosities 

    # Stellar luminosities in all bands, in the simulated volume:
    Lum_opt_vol::Float64
    Lum_FUV_vol::Float64
    Lum_LW_vol::Float64
    Lum_EUV1_vol::Float64
    Lum_EUV2_vol::Float64
    Lum_EUV3_vol::Float64

    # Stellar photon emission rates in all non-IR bands:
    dN_dt_opt::Float64
    dN_dt_FUV::Float64
    dN_dt_LW::Float64
    dN_dt_EUV1::Float64
    dN_dt_EUV2::Float64
    dN_dt_EUV3::Float64
    dN_dt_LyC::Float64    # Total LyC (EUV1 + EUV2 + EUV3) emission rate 

    # Emissivities for all bands:
    js_IR::Array{Float64, 2}
    js_opt::Array{Float64, 2}
    js_FUV::Array{Float64, 2}
    js_LW::Array{Float64, 2}
    js_EUV1::Array{Float64, 2}
    js_EUV2::Array{Float64, 2}
    js_EUV3::Array{Float64, 2}

    # Stellar mass density for all cells:
    ρ_star::Array{Float64, 2}

    # Arrays for tracking gas momentum changes from direct & indirect (IR) radiation pressure:
    Δρu_stellar_r::Array{Float64, 2}
    Δρu_stellar_Z::Array{Float64, 2}
    Δρu_IR_r::Array{Float64, 2}
    Δρu_IR_Z::Array{Float64, 2}
    f_stellar_r::Array{Float64, 2}
    f_stellar_Z::Array{Float64, 2}
    f_IR_r::Array{Float64, 2}
    f_IR_Z::Array{Float64, 2}

    # LyC & LW escape fraction:
    f_esc_LyC::Float64 
    f_esc_LW::Float64

    # Direct stellar radiation pressure 'force multipliers' (expected ~ 1 for absorption from point source):
    M_F_dir::Float64
    M_F_dir_radial_net::Float64    # Force multiplier radially outward (can be negative if force is radially inward)
    M_F_dir_radial_out::Float64    # Force multiplier radially outward (only outward components counted) 
    M_F_dir_antigravity::Float64   # Force multiplier anti-parallel to gravity 

    # IR radiation pressure force multipliers (useful diagnostics):
    M_F_IR::Float64
    M_F_IR_radial_net::Float64     # Force multiplier radially outward (can be negative if force is radially inward)
    M_F_IR_radial_out::Float64     # Force multiplier radially outward (only outward components counted) 
    M_F_IR_antigravity::Float64    # Force multiplier anti-parallel to gravity 

    # On-the-spot approximation (Case B) or modelling of recombination photons:
    OTS::Bool

    # Flux method for M1, either HLL or GLF:
    FluxMethod::Val{FluxMethod}

end


""" A struct for the chemistry state of the gas. 
    Stores fractional populations of all major gas 
    species tracked in the code. """
mutable struct ChemistryState{T<:AbstractFloat}

    # Primordial species (H I, H II, H₂, He I, He II, He III, 
    # D I, D II, HD): 
    xHI_adv::Array{T, 2}
    xHI_adv_old::Array{T, 2}

    xHII::Array{T, 2} 
    xHII_old::Array{T, 2}

    xH2::Array{T, 2} 
    xH2_old::Array{T, 2} 

    xHeI_adv::Array{T, 2}
    xHeI_adv_old::Array{T, 2}   

    xHeII::Array{T, 2}   
    xHeII_old::Array{T, 2}

    xHeIII::Array{T, 2}
    xHeIII_old::Array{T, 2}

    xDI_adv::Array{T, 2}
    xDI_adv_old::Array{T, 2}

    xDII::Array{T, 2}
    xDII_old::Array{T, 2}

    xHD::Array{T, 2}
    xHD_old::Array{T, 2}

    # Metals (C I, C II, C III, CHₓ, CO, O I, O II, O III, O IV, OHₓ, 
    # S I, S II, S III, S IV): 
    xCI::Array{T, 2}
    xCI_old::Array{T, 2}

    xCII::Array{T, 2} 
    xCII_old::Array{T, 2}

    xCIII::Array{T, 2} 
    xCIII_old::Array{T, 2}

    xCHx_adv::Array{T, 2}       # CHₓ is only advected for accuracy, but otherwise obtained from conservation 
    xCHx_adv_old::Array{T, 2}

    xCO::Array{T, 2}            # Defined as nCO / nC
    xCO_old::Array{T, 2}

    xOI::Array{T, 2} 
    xOI_old::Array{T, 2}

    xOII::Array{T, 2} 
    xOII_old::Array{T, 2}

    xOIII::Array{T, 2} 
    xOIII_old::Array{T, 2}

    xOIV::Array{T, 2} 
    xOIV_old::Array{T, 2}

    xOHx_adv::Array{T, 2}       # OHₓ is only advected for accuracy, but otherwise obtained from conservation 
    xOHx_adv_old::Array{T, 2}

    xSI_adv::Array{T, 2}        # S I is only advected for accuracy, but otherwise obtained from conservation 
    xSI_adv_old::Array{T, 2}

    xSII::Array{T, 2} 
    xSII_old::Array{T, 2}

    xSIII::Array{T, 2} 
    xSIII_old::Array{T, 2}

    xSIV::Array{T, 2} 
    xSIV_old::Array{T, 2}

    # Gas-phase metallicities:
    ZC::Array{T, 2}        # Carbon gas metallicity 
    ZSil::Array{T, 2}      # Si gas metallicity 
    Zgasrest::Array{T, 2}  # Rest of gas metallicity 

    # Sobolev-like density length scale ρ / ∇ρ:
    Lsob::Array{T, 2}

    # Setting for whether to model hydrogen + helium + deuterium, or pure 
    # hydrogen gas (useful for idealized tests):
    HydrogenAndHelium::Bool

end


""" A struct for storing the dust properties. The dust is advected and experiences 
    radiation pressure, drag, and gravity, but can also change as a result of 
    dust destruction and dust growth. This struct also stores dust 
    optical properties (opacities etc.) for all bands & dust types. """
mutable struct DustState{ITP_P, ITP_R}

    # Dust growth/destruction setting (on if true, off if false):
    DustGrowthDestruction::Bool

    # Dust temperature for every cell (silicates, graphite, PAH):
    Td_Sil::Array{Float64, 2}
    Td_C::Array{Float64, 2}
    Td_PAH::Array{Float64, 2}

    # Radiation temperature for the IR field, which determines e.g. the 
    # Rosseland-mean opacity:
    Trad::Array{Float64, 2}

    # Grain charges (silicates, graphite, PAH):
    Zgr_Sil::Array{Float64, 2}
    Zgr_C::Array{Float64, 2}
    Zgr_PAH::Array{Float64, 2}

    # Band absorption opacities for silicate grains (cm² g⁻¹ of dust in bin):
    κd_abs_opt_Sil::Float64
    κd_abs_FUV_Sil::Float64
    κd_abs_LW_Sil::Float64
    κd_abs_EUV1_Sil::Float64
    κd_abs_EUV2_Sil::Float64
    κd_abs_EUV3_Sil::Float64
    κd_abs_Lyα_Sil::Float64

    # Band absorption opacities for graphite grains (cm² g⁻¹ of dust in bin):
    κd_abs_opt_C::Float64
    κd_abs_FUV_C::Float64
    κd_abs_LW_C::Float64
    κd_abs_EUV1_C::Float64
    κd_abs_EUV2_C::Float64
    κd_abs_EUV3_C::Float64
    κd_abs_Lyα_C::Float64

    # Band absorption opacities for PAH grains (cm² g⁻¹ of dust in bin):
    κd_abs_opt_PAH::Float64
    κd_abs_FUV_PAH::Float64
    κd_abs_LW_PAH::Float64
    κd_abs_EUV1_PAH::Float64
    κd_abs_EUV2_PAH::Float64
    κd_abs_EUV3_PAH::Float64
    κd_abs_Lyα_PAH::Float64

    # Band flux opacities for silicate grains (cm² g⁻¹ of dust in bin):
    κd_F_opt_Sil::Float64
    κd_F_FUV_Sil::Float64
    κd_F_LW_Sil::Float64
    κd_F_EUV1_Sil::Float64
    κd_F_EUV2_Sil::Float64
    κd_F_EUV3_Sil::Float64
    κd_F_Lyα_Sil::Float64

    # Band flux opacities for graphite grains (cm² g⁻¹ of dust in bin):
    κd_F_opt_C::Float64
    κd_F_FUV_C::Float64
    κd_F_LW_C::Float64
    κd_F_EUV1_C::Float64
    κd_F_EUV2_C::Float64
    κd_F_EUV3_C::Float64
    κd_F_Lyα_C::Float64

    # Band flux opacities for PAH grains (cm² g⁻¹ of dust in bin):
    κd_F_opt_PAH::Float64
    κd_F_FUV_PAH::Float64
    κd_F_LW_PAH::Float64
    κd_F_EUV1_PAH::Float64
    κd_F_EUV2_PAH::Float64
    κd_F_EUV3_PAH::Float64
    κd_F_Lyα_PAH::Float64

    # --- IR mean opacity interpolations ---
    # Store log₁₀(κ) vs. log₁₀(T) interpolations:
    logκP_Sil_itp::ITP_P; logκP_C_itp::ITP_P; logκP_PAH_itp::ITP_P
    logκR_Sil_itp::ITP_R; logκR_C_itp::ITP_R; logκR_PAH_itp::ITP_R

    # Store dust cross-sections per H atom in the Lyα band, for diagnostic purposes:
    σ_Lyα_per_H::Array{Float64, 2}
end


""" A struct for holding main hydro arrays, e.g. gas densities 
    and velocities. """
mutable struct HydrodynamicsState{T<:AbstractFloat}
    # Total number of dust species/bins (default will be 3, 
    # for silicates, graphite, and PAH grains):
    N_dust::Int

    # Updated hydro state variables:
    ρ_new::Array{T, 2} 
    ρu_X_new::Array{T, 2} 
    ρu_Y_new::Array{T, 2}
    E_new::Array{T, 2}
    e_dual_new::Array{T, 2}
    # "Old" hydro state variables:
    ρ::Array{T, 2}
    ρu_X::Array{T, 2}
    ρu_Y::Array{T, 2}     
    E::Array{T, 2}
    e_dual::Array{T, 2}
    # Arrays for storing pre-update state in the RK time integration:
    ρ_stage::Array{T, 2}
    ρu_X_stage::Array{T, 2}    
    ρu_Y_stage::Array{T, 2}
    E_stage::Array{T, 2}
    e_dual_stage::Array{T, 2}
    # Cell-averaged gas velocities & adiabatic indices:
    γad::Array{T, 2}
    u_X::Array{T, 2}
    u_Y::Array{T, 2} 
    # Reconstructed X-interface primitive variables:
    ρ_L_X::Array{T, 2}
    ρ_R_X::Array{T, 2}
    u_X_L_X::Array{T, 2} 
    u_X_R_X::Array{T, 2} 
    u_Y_L_X::Array{T, 2} 
    u_Y_R_X::Array{T, 2} 
    P_L_X::Array{T, 2}
    P_R_X::Array{T, 2}
    e_dual_L_X::Array{T, 2} 
    e_dual_R_X::Array{T, 2}
    # Interface X-fluxes:
    F_ρ_X::Array{T, 2}
    F_ρu_X_X::Array{T, 2}
    F_ρu_Y_X::Array{T, 2}
    F_E_X::Array{T, 2}
    F_e_dual_X::Array{T, 2}
    # Reconstructed Y-interface primitive variables:
    ρ_L_Y::Array{T, 2} 
    ρ_R_Y::Array{T, 2} 
    u_Y_L_Y::Array{T, 2} 
    u_Y_R_Y::Array{T, 2}
    u_X_L_Y::Array{T, 2}
    u_X_R_Y::Array{T, 2} 
    P_L_Y::Array{T, 2} 
    P_R_Y::Array{T, 2}
    e_dual_L_Y::Array{T, 2} 
    e_dual_R_Y::Array{T, 2}
    # Interface Y-fluxes:
    F_ρ_Y::Array{T, 2} 
    F_ρu_X_Y::Array{T, 2} 
    F_ρu_Y_Y::Array{T, 2} 
    F_E_Y::Array{T, 2}
    F_e_dual_Y::Array{T, 2}
    # Contact velocities:
    u_X_star_X::Array{T, 2}
    u_Y_star_Y::Array{T, 2}
    # Gas velocity gradient-related arrays (only needed for RHD):
    ∂u_X_∂X::Array{T, 2}
    ∂u_X_∂Y::Array{T, 2}
    ∂u_Y_∂X::Array{T, 2}
    ∂u_Y_∂Y::Array{T, 2}
    u_r_over_r::Array{T, 2}
    # Gas pressure, internal energy density e, and temperature T:
    P::Array{T, 2}
    e::Array{T, 2}
    T::Array{T, 2}
    # Ghost-padded variables:
    ρg_old::Array{T, 2} 
    u_Xg_old::Array{T, 2} 
    u_Yg_old::Array{T, 2} 
    Pg_old::Array{T, 2}
    e_dualg_old::Array{T, 2}
    # Cell-centered primitives:
    ρc::Array{T, 2} 
    ρu_Xc::Array{T, 2} 
    ρu_Yc::Array{T, 2} 
    Ec::Array{T, 2} 
    u_Xc::Array{T, 2} 
    u_Yc::Array{T, 2} 
    Pc::Array{T, 2} 
    # Shock indicator:
    Shock::Array{Int, 2}
    # Boundary conditions:
    BC_X_left::Symbol
    BC_X_right::Symbol
    BC_Y_left::Symbol
    BC_Y_right::Symbol

    # All dust arrays, of the form ~ (N_dust, N_X, N_Y):
    ρd_new::Array{T, 3} 
    ρdud_X_new::Array{T, 3} 
    ρdud_Y_new::Array{T, 3} 
    ρd::Array{T, 3} 
    ρdud_X::Array{T, 3} 
    ρdud_Y::Array{T, 3} 
    ρd_stage::Array{T, 3} 
    ρdud_X_stage::Array{T, 3} 
    ρdud_Y_stage::Array{T, 3}
    ud_X::Array{T, 3} 
    ud_Y::Array{T, 3} 
    ρd_L_X::Array{T, 3} 
    ρd_R_X::Array{T, 3} 
    ud_X_L_X::Array{T, 3} 
    ud_X_R_X::Array{T, 3} 
    ud_Y_L_X::Array{T, 3} 
    ud_Y_R_X::Array{T, 3}
    F_ρd_X::Array{T, 3} 
    F_ρdud_X_X::Array{T, 3} 
    F_ρdud_Y_X::Array{T, 3}
    ρd_L_Y::Array{T, 3} 
    ρd_R_Y::Array{T, 3} 
    ud_Y_L_Y::Array{T, 3} 
    ud_Y_R_Y::Array{T, 3} 
    ud_X_L_Y::Array{T, 3} 
    ud_X_R_Y::Array{T, 3} 
    F_ρd_Y::Array{T, 3} 
    F_ρdud_X_Y::Array{T, 3} 
    F_ρdud_Y_Y::Array{T, 3} 
    ρdg_old::Array{T, 3} 
    ud_Xg_old::Array{T, 3} 
    ud_Yg_old::Array{T, 3}
    ρdc::Array{T, 3}
    ρdud_Xc::Array{T, 3} 
    ρdud_Yc::Array{T, 3} 
    ud_Xc::Array{T, 3} 
    ud_Yc::Array{T, 3}

    # Arrays for advecting passive scalars (e.g. ρx_HII, ρZC, etc):
    Scalars::Array{T, 3}         # Size (N_X, N_Y, N_Scalars)
    Scalars_old::Array{T, 3}     # Size (N_X, N_Y, N_Scalars)
    Scalars_stage::Array{T, 3}   # Size (N_X, N_Y, N_Scalars)
    ρ_save::Array{T, 2}          # Size (N_X, N_Y), for storing the gas density in RK2 updates of scalar advection 

    # Total dust density:
    ρdtot::Array{T, 2} 

    # Total gas & dust mass (g) in simulation volume:
    M_simvol::T
    Md_simvol::T  

    # Velocity dispersion diagnostics (cm s⁻¹):
    σ_X::T 
    σ_Y::T 
    σ::T 

    # Measures of the total gas pressure force (relevant for measuring photoionization feedback 
    # in RHD simulations in cylindrical coordinates, otherwise unused):
    IntegratedPressureForce::T                # ∫dV |- ∇P|
    IntegratedPressureForce_radial_net::T     # ∫dV ( - ∇P ⋅ r̂_out )
    IntegratedPressureForce_radial_out::T     # ∫dV max( - ∇P ⋅ r̂_out, 0.0 )
    IntegratedPressureForce_antigrav::T       # ∫dV ( - ∇P ⋅ â_grav )

    # Pressure at interfaces obtained from Riemann solver:
    P_interface_X::Array{T, 2}
    P_interface_Y::Array{T, 2}
    P_interface_X_snapshot::Array{T, 2}  # Store snapshot of P_interface_X from step n (since P_interface_X is updated each RK stage)
    P_interface_Y_snapshot::Array{T, 2}  # Store snapshot of P_interface_Y from step n (since P_interface_Y is updated each RK stage)
end


""" A small struct for the ghost-padded spatial grid. """
struct GhostPaddedGrid{T<:AbstractFloat}
    Xg::Array{T, 1}
    XCoMg::Array{T, 1}
    Yg::Array{T, 1}
    YCoMg::Array{T, 1}
end


""" Struct for a cylindrical grid, including WENO3 coefficients. """
struct GeometryCylindrical{T<:AbstractFloat} 
    coord::Val{:Cylindrical}
    # Interface values of r and Z:
    r::Array{T, 1}                # length N_r+1
    Z::Array{T, 1}                # length N_Z+1
    # Cell centers of mass:
    rCoM::Array{T, 1}             # length N_r
    ZCoM::Array{T, 1}             # length N_Z
    # Cell volumes & inverse volumes:
    ΔV::Array{T, 2}               # (N_r, N_Z)
    inv_ΔV::Array{T, 2}           # (N_r, N_Z)
    # Interface areas:
    A_r::Array{T, 2}              # (N_r+1, N_Z)
    A_Z::Array{T, 2}              # (N_r, N_Z+1)
    # Values of ⟨1/r⟩ for all cells:
    inv_r_cell::Array{T, 1}       # length N_r
    # Simulation box volume = sum(ΔV):
    Vsim::Float64

    # WENO3 weights for converting cell-averaged conservatives 
    # to cell-center conservatives:
    w_i_m1_r::Array{T, 1}         # length N_r
    w_i_0_r::Array{T, 1}          # length N_r
    w_i_p1_r::Array{T, 1}         # length N_r

    w_j_m1_Z::Array{T, 1}         # length N_Z
    w_j_0_Z::Array{T, 1}          # length N_Z
    w_j_p1_Z::Array{T, 1}         # length N_Z

    # WENO3 weights for converting cell-centered primitives 
    # to cell-averaged primitives:
    q_i_m1_r::Array{T, 1}         # length N_r
    q_i_0_r::Array{T, 1}          # length N_r
    q_i_p1_r::Array{T, 1}         # length N_r

    q_j_m1_Z::Array{T, 1}         # length N_Z
    q_j_0_Z::Array{T, 1}          # length N_Z
    q_j_p1_Z::Array{T, 1}         # length N_Z

    # WENO3 weights for reconstructing interfaces:
    w_plus_i_p1::Array{T, 1}      # length N_r
    w_minus_i_p1::Array{T, 1}     # length N_r

    w_plus_j_p1::Array{T, 1}      # length N_Z
    w_minus_j_p1::Array{T, 1}     # length N_Z

    # Below are various intervals & inverse intervals 
    # that are needed in the M1 advection. It is efficient to 
    # compute them once and store them, especially for the Lyα RT
    # to avoid lots of repeated calculations:

    Δr_L::Array{T, 1}           # r[i+1]    - rCoM[i] for interior cells 
    Δr_R::Array{T, 1}           # rCoM[i+1] - r[i+1]  for interior cells 
    inv_dr_i_im1::Array{T, 1}   # 1 / ( rCoM[i]   - rCoM[i-1] ) for interior cells 
    inv_dr_ip1_i::Array{T, 1}   # 1 / ( rCoM[i+1] - rCoM[i]   ) for interior cells 
    inv_dr_ip2_ip1::Array{T, 1} # 1 / ( rCoM[i+2] - rCoM[i+1] ) for interior cells 

    # Corresponding values at inner & outer r-boundaries:
    inv_dr_i_im1_InnerBoundary::T
    inv_dr_ip1_i_InnerBoundary::T
    inv_dr_ip2_ip1_InnerBoundary::T
    Δr_L_InnerBoundary::T
    Δr_R_InnerBoundary::T

    inv_dr_i_im1_OuterBoundary::T
    inv_dr_ip1_i_OuterBoundary::T
    inv_dr_ip2_ip1_OuterBoundary::T
    Δr_L_OuterBoundary::T 
    Δr_R_OuterBoundary::T

    # Similarly for Z-intervals:
    ΔZ_L::Array{T, 1}           # Z[j+1]    - ZCoM[j] for interior cells 
    ΔZ_R::Array{T, 1}           # ZCoM[j+1] - Z[j+1]  for interior cells 
    inv_dZ_j_jm1::Array{T, 1}   # 1 / ( ZCoM[j]   - ZCoM[j-1] ) for interior cells 
    inv_dZ_jp1_j::Array{T, 1}   # 1 / ( ZCoM[j+1] - ZCoM[j]   ) for interior cells 
    inv_dZ_jp2_jp1::Array{T, 1} # 1 / ( ZCoM[j+2] - ZCoM[j+1] ) for interior cells 

    # Corresponding values at inner & outer Z-boundaries:
    inv_dZ_j_jm1_InnerBoundary::T
    inv_dZ_jp1_j_InnerBoundary::T
    inv_dZ_jp2_jp1_InnerBoundary::T
    ΔZ_L_InnerBoundary::T
    ΔZ_R_InnerBoundary::T

    inv_dZ_j_jm1_OuterBoundary::T
    inv_dZ_jp1_j_OuterBoundary::T
    inv_dZ_jp2_jp1_OuterBoundary::T
    ΔZ_L_OuterBoundary::T
    ΔZ_R_OuterBoundary::T

end


""" Struct for a Cartesian grid, including WENO3 coefficients. """
struct GeometryCartesian{T<:AbstractFloat} 
    coord::Val{:Cartesian}
    # Interface values of X and Z:
    X::Array{T, 1}                # length N_X+1
    Y::Array{T, 1}                # length N_Y+1
    # Cell centers of mass:
    XCoM::Array{T, 1}             # length N_X
    YCoM::Array{T, 1}             # length N_Y
    # Cell volumes & inverse volumes:
    ΔV::Array{T, 2}               # (N_X, N_Y)
    inv_ΔV::Array{T, 2}           # (N_X, N_Y)
    # Interface areas:
    A_X::Array{T, 2}              # (N_X+1, N_Y)
    A_Y::Array{T, 2}              # (N_X, N_Y+1)
    # NOT NEEDED IN CARTESIAN COORDINATES:
    inv_r_cell::Array{T, 1}       # length N_X
    # Simulation box volume = sum(ΔV):
    Vsim::Float64

    # WENO3 weights for converting cell-averaged conservatives 
    # to cell-center conservatives:
    w_i_m1_X::Array{T, 1}         # length N_X
    w_i_0_X::Array{T, 1}          # length N_X
    w_i_p1_X::Array{T, 1}         # length N_X

    w_j_m1_Y::Array{T, 1}         # length N_Y
    w_j_0_Y::Array{T, 1}          # length N_Y
    w_j_p1_Y::Array{T, 1}         # length N_Y

    # WENO3 weights for converting cell-centered primitives 
    # to cell-averaged primitives:
    q_i_m1_X::Array{T, 1}         # length N_X
    q_i_0_X::Array{T, 1}          # length N_X
    q_i_p1_X::Array{T, 1}         # length N_X

    q_j_m1_Y::Array{T, 1}         # length N_Y
    q_j_0_Y::Array{T, 1}          # length N_Y
    q_j_p1_Y::Array{T, 1}         # length N_Y

    # WENO3 weights for reconstructing interfaces:
    w_plus_i_p1::Array{T, 1}      # length N_X
    w_minus_i_p1::Array{T, 1}     # length N_X

    w_plus_j_p1::Array{T, 1}      # length N_Y
    w_minus_j_p1::Array{T, 1}     # length N_Y

    # Below are various intervals & inverse intervals 
    # that are needed in the M1 advection. It is efficient to 
    # compute them once and store them, especially for the Lyα RT
    # to avoid lots of repeated calculations:

    ΔX_L::Array{T, 1}           # X[i+1]    - XCoM[i] for interior cells 
    ΔX_R::Array{T, 1}           # XCoM[i+1] - X[i+1]  for interior cells 
    inv_dX_i_im1::Array{T, 1}   # 1 / ( XCoM[i]   - XCoM[i-1] ) for interior cells 
    inv_dX_ip1_i::Array{T, 1}   # 1 / ( XCoM[i+1] - XCoM[i]   ) for interior cells 
    inv_dX_ip2_ip1::Array{T, 1} # 1 / ( XCoM[i+2] - XCoM[i+1] ) for interior cells 

    # Corresponding values at inner & outer X-boundaries:
    inv_dX_i_im1_InnerBoundary::T
    inv_dX_ip1_i_InnerBoundary::T
    inv_dX_ip2_ip1_InnerBoundary::T
    ΔX_L_InnerBoundary::T
    ΔX_R_InnerBoundary::T

    inv_dX_i_im1_OuterBoundary::T
    inv_dX_ip1_i_OuterBoundary::T
    inv_dX_ip2_ip1_OuterBoundary::T
    ΔX_L_OuterBoundary::T 
    ΔX_R_OuterBoundary::T

    # Similarly for Y-intervals:
    ΔY_L::Array{T, 1}           # Y[j+1]    - YCoM[j] for interior cells 
    ΔY_R::Array{T, 1}           # YCoM[j+1] - Y[j+1]  for interior cells 
    inv_dY_j_jm1::Array{T, 1}   # 1 / ( YCoM[j]   - YCoM[j-1] ) for interior cells 
    inv_dY_jp1_j::Array{T, 1}   # 1 / ( YCoM[j+1] - YCoM[j]   ) for interior cells 
    inv_dY_jp2_jp1::Array{T, 1} # 1 / ( YCoM[j+2] - YCoM[j+1] ) for interior cells 

    # Corresponding values at inner & outer Y-boundaries:
    inv_dY_j_jm1_InnerBoundary::T
    inv_dY_jp1_j_InnerBoundary::T
    inv_dY_jp2_jp1_InnerBoundary::T
    ΔY_L_InnerBoundary::T
    ΔY_R_InnerBoundary::T

    inv_dY_j_jm1_OuterBoundary::T
    inv_dY_jp1_j_OuterBoundary::T
    inv_dY_jp2_jp1_OuterBoundary::T
    ΔY_L_OuterBoundary::T
    ΔY_R_OuterBoundary::T
end


""" A struct for the gravitational potential, its gradient components,
    and the gravitational force per unit volume (f_grav). """
mutable struct GravityState{T<:AbstractFloat}
    Φ::Array{T, 2} 
    Φ_half::Array{T, 2} 
    Φ_old::Array{T, 2}  
    Φ_snapshot::Array{T, 2}
    ∂Φ∂r::Array{T, 2}   
    ∂Φ∂Z::Array{T, 2}
    ρtot::Array{T, 2}        # Total gravitating mass 
    ρ_bg::Array{T, 2}        # Static non-gas gravitating mass density, e.g. from star & dark matter 
    f_gravity_r::Array{T, 2}
    f_gravity_Z::Array{T, 2}
    P::Array{T, 2}           # Legendre polynomials (for BCs), one array for each thread (to avoid race conditions)
    Q::Array{T, 1}           # Multipole moments (for BCs)
    ℓmax::Int                # Maximum ℓ to use in Legendre polynomials

    # Force diagnostics:
    IntegratedGravityForce::T                # ∫dV |f_gravity|, same as ∫dV ( f_gravity ⋅ â_grav )
    IntegratedGravityForce_radial_net::T     # ∫dV ( f_gravity ⋅ r̂_out ), NOTE: negative typically
end

""" A struct for the Poisson equation coefficients. These never 
    change for a given fixed grid. """
struct GravityCoeffs{T<:AbstractFloat}
    RPoiss_ijp::Array{T, 2}
    RPoiss_ij::Array{T, 2}
    RPoiss_ijm::Array{T, 2}
    ZPoiss_ijp::Array{T, 2}
    ZPoiss_ij::Array{T, 2} 
    ZPoiss_ijm::Array{T, 2}
    Δτ_max::T
    Δτ_min::T 
end


end # End of module 