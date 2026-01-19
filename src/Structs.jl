module Structs 


export LyαArrays, StellarRTArrays, HydrodynamicsState, ChemistryState,
       GhostPaddedGrid, StellarRTArrays, GeometryCylindrical, GeometryCartesian,
       GravityState, GravityCoeffs


""" A useful struct for storing all Lyα related arrays/quantities/settings. """
mutable struct LyαArrays 
    # Lyα reduced speed of light:
    c̃_Lyα::Float64

    # Intensities (latest & old) for all bands: 
    J_Lyα::Array{Float64, 3};  J_Lyα_old::Array{Float64, 3}

    # Lyα energy density:
    e_Lyα::Array{Float64, 2}

    # H ≡ ∫dΩ I n̂ / 4π arrays
    H_r_Lyα::Array{Float64, 3};        H_Z_Lyα::Array{Float64, 3}
    H_r_Lyα_old::Array{Float64, 3};    H_Z_Lyα_old::Array{Float64, 3}

    # Eddington tensor components:
    D_rr_Lyα::Array{Float32, 3};  D_ZZ_Lyα::Array{Float32, 3};  D_rZ_Lyα::Array{Float32, 3}

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

    # Strang-split order for M1 advection (changes automatically every advection step):
    rfirst_Lyα::Bool 

    # Force multiplier, Lyα luminosity, Lyα escape fraction:
    M_F::Float64
    M_F_antigravity::Float64       # Force multiplier anti-parallel to gravity 
    L_Lyα::Float64
    f_esc_Lyα::Float64

end


""" A struct for organizing stellar RT arrays, and the stellar mass density. """
mutable struct StellarRTArrays
    # Stellar RT reduced speed of light:
    c̃::Float64

    # Intensities (latest & old) for all bands: 
    J_IR::Array{Float64, 2};      J_IR_old::Array{Float64, 2}
    J_opt::Array{Float64, 2};     J_opt_old::Array{Float64, 2}
    J_FUV::Array{Float64, 2};     J_FUV_old::Array{Float64, 2}
    J_LW::Array{Float64, 2};      J_LW_old::Array{Float64, 2}
    J_EUV1::Array{Float64, 2};    J_EUV1_old::Array{Float64, 2}
    J_EUV2::Array{Float64, 2};    J_EUV2_old::Array{Float64, 2}
    J_EUV3::Array{Float64, 2};    J_EUV3_old::Array{Float64, 2}

    # H ≡ ∫dΩ I n̂ / 4π arrays for all bands:
    H_r_IR::Array{Float64, 2};         H_Z_IR::Array{Float64, 2}
    H_r_IR_old::Array{Float64, 2};     H_Z_IR_old::Array{Float64, 2}

    H_r_opt::Array{Float64, 2};        H_Z_opt::Array{Float64, 2}
    H_r_opt_old::Array{Float64, 2};    H_Z_opt_old::Array{Float64, 2}

    H_r_FUV::Array{Float64, 2};        H_Z_FUV::Array{Float64, 2}
    H_r_FUV_old::Array{Float64, 2};    H_Z_FUV_old::Array{Float64, 2}

    H_r_LW::Array{Float64, 2};         H_Z_LW::Array{Float64, 2}
    H_r_LW_old::Array{Float64, 2};     H_Z_LW_old::Array{Float64, 2}

    H_r_EUV1::Array{Float64, 2};       H_Z_EUV1::Array{Float64, 2}
    H_r_EUV1_old::Array{Float64, 2};   H_Z_EUV1_old::Array{Float64, 2}

    H_r_EUV2::Array{Float64, 2};       H_Z_EUV2::Array{Float64, 2}
    H_r_EUV2_old::Array{Float64, 2};   H_Z_EUV2_old::Array{Float64, 2}

    H_r_EUV3::Array{Float64, 2};       H_Z_EUV3::Array{Float64, 2}
    H_r_EUV3_old::Array{Float64, 2};   H_Z_EUV3_old::Array{Float64, 2}

    # Eddington tensor componens for all bands:
    D_rr_IR::Array{Float32, 2};   D_ZZ_IR::Array{Float32, 2};   D_rZ_IR::Array{Float32, 2}
    D_rr_opt::Array{Float32, 2};  D_ZZ_opt::Array{Float32, 2};  D_rZ_opt::Array{Float32, 2}
    D_rr_FUV::Array{Float32, 2};  D_ZZ_FUV::Array{Float32, 2};  D_rZ_FUV::Array{Float32, 2}
    D_rr_LW::Array{Float32, 2};   D_ZZ_LW::Array{Float32, 2};   D_rZ_LW::Array{Float32, 2}
    D_rr_EUV1::Array{Float32, 2}; D_ZZ_EUV1::Array{Float32, 2}; D_rZ_EUV1::Array{Float32, 2}
    D_rr_EUV2::Array{Float32, 2}; D_ZZ_EUV2::Array{Float32, 2}; D_rZ_EUV2::Array{Float32, 2}
    D_rr_EUV3::Array{Float32, 2}; D_ZZ_EUV3::Array{Float32, 2}; D_rZ_EUV3::Array{Float32, 2}

    # ϕϕ-component of radiation pressure tensor K = D × J at interfaces, 
    # needed for geometric source term:
    K_ϕϕ_iphalf_IR::Array{Float64, 2};     K_ϕϕ_imhalf_IR::Array{Float64, 2}
    K_ϕϕ_iphalf_opt::Array{Float64, 2};    K_ϕϕ_imhalf_opt::Array{Float64, 2}
    K_ϕϕ_iphalf_FUV::Array{Float64, 2};    K_ϕϕ_imhalf_FUV::Array{Float64, 2}
    K_ϕϕ_iphalf_LW::Array{Float64, 2};     K_ϕϕ_imhalf_LW::Array{Float64, 2}
    K_ϕϕ_iphalf_EUV1::Array{Float64, 2};   K_ϕϕ_imhalf_EUV1::Array{Float64, 2}
    K_ϕϕ_iphalf_EUV2::Array{Float64, 2};   K_ϕϕ_imhalf_EUV2::Array{Float64, 2}
    K_ϕϕ_iphalf_EUV3::Array{Float64, 2};   K_ϕϕ_imhalf_EUV3::Array{Float64, 2}

    # Flux components for all bands: 
    F_r_IR::Array{Float64, 2};      F_Z_IR::Array{Float64, 2}
    F_r_Hr_IR::Array{Float64, 2};   F_Z_Hr_IR::Array{Float64, 2}
    F_r_HZ_IR::Array{Float64, 2};   F_Z_HZ_IR::Array{Float64, 2}

    F_r_opt::Array{Float64, 2};     F_Z_opt::Array{Float64, 2}
    F_r_Hr_opt::Array{Float64, 2};  F_Z_Hr_opt::Array{Float64, 2}
    F_r_HZ_opt::Array{Float64, 2};  F_Z_HZ_opt::Array{Float64, 2}

    F_r_FUV::Array{Float64, 2};     F_Z_FUV::Array{Float64, 2}
    F_r_Hr_FUV::Array{Float64, 2};  F_Z_Hr_FUV::Array{Float64, 2}
    F_r_HZ_FUV::Array{Float64, 2};  F_Z_HZ_FUV::Array{Float64, 2}

    F_r_LW::Array{Float64, 2};      F_Z_LW::Array{Float64, 2}
    F_r_Hr_LW::Array{Float64, 2};   F_Z_Hr_LW::Array{Float64, 2}
    F_r_HZ_LW::Array{Float64, 2};   F_Z_HZ_LW::Array{Float64, 2}

    F_r_EUV1::Array{Float64, 2};    F_Z_EUV1::Array{Float64, 2}
    F_r_Hr_EUV1::Array{Float64, 2}; F_Z_Hr_EUV1::Array{Float64, 2}
    F_r_HZ_EUV1::Array{Float64, 2}; F_Z_HZ_EUV1::Array{Float64, 2}

    F_r_EUV2::Array{Float64, 2};    F_Z_EUV2::Array{Float64, 2}
    F_r_Hr_EUV2::Array{Float64, 2}; F_Z_Hr_EUV2::Array{Float64, 2}
    F_r_HZ_EUV2::Array{Float64, 2}; F_Z_HZ_EUV2::Array{Float64, 2}

    F_r_EUV3::Array{Float64, 2};    F_Z_EUV3::Array{Float64, 2}
    F_r_Hr_EUV3::Array{Float64, 2}; F_Z_Hr_EUV3::Array{Float64, 2}
    F_r_HZ_EUV3::Array{Float64, 2}; F_Z_HZ_EUV3::Array{Float64, 2}

    # Flux opacities for all bands:
    α_F_IR::Array{Float64, 2}
    α_F_opt::Array{Float64, 2}
    α_F_FUV::Array{Float64, 2}
    α_F_LW::Array{Float64, 2}
    α_F_EUV1::Array{Float64, 2}
    α_F_EUV2::Array{Float64, 2}
    α_F_EUV3::Array{Float64, 2}

    # Luminosities in all bands, over infinite space (i.e. the full luminosity of the star):
    Lum_IR::Float64
    Lum_opt::Float64
    Lum_FUV::Float64
    Lum_LW::Float64
    Lum_EUV1::Float64
    Lum_EUV2::Float64
    Lum_EUV3::Float64

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

    # Strang-split order for M1 advection (changes automatically every advection step):
    rfirst_StellarRT::Bool 

    # LyC & LW escape fraction:
    f_esc_LyC::Float64 
    f_esc_LW::Float64

    # Direct stellar radiation pressure 'force multipliers' (expected ~ 1 for absorption from point source):
    M_F_dir::Float64
    M_F_dir_antigravity::Float64   # Force multiplier anti-parallel to gravity 

    # IR radiation pressure force multipliers (useful diagnostics):
    M_F_IR::Float64
    M_F_IR_antigravity::Float64   # Force multiplier anti-parallel to gravity 

    # On-the-spot approximation (Case B) or modelling of recombination photons:
    OTS::Bool

end


""" A struct for the chemistry state of the gas. 
    Stores fractional populations of all major gas 
    species tracked in the code. """
mutable struct ChemistryState{T<:AbstractFloat}

    # Primordial species (H I, H II, He II, He III, D I, D II, HD): 
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

    # Metals (C I, C II, CHₓ, CO, O I, O II, OHₓ): 
    xCI::Array{T, 2}
    xCI_old::Array{T, 2}

    xCII::Array{T, 2} 
    xCII_old::Array{T, 2}

    xCHx_adv::Array{T, 2}       # CHₓ is only advected for accuracy, but otherwise obtained from conservation 
    xCHx_adv_old::Array{T, 2}

    xCO::Array{T, 2}            # Defined as nCO / nC
    xCO_old::Array{T, 2}

    xOI::Array{T, 2} 
    xOI_old::Array{T, 2}

    xOII::Array{T, 2} 
    xOII_old::Array{T, 2}

    xOHx_adv::Array{T, 2}       # OHₓ is only advected for accuracy, but otherwise obtained from conservation 
    xOHx_adv_old::Array{T, 2}

    # Overall gas metallicity:
    Zgas_Zsun::Array{T, 2}

end


""" A struct for storing the dust properties. The dust is advected and experiences 
    radiation pressure, drag, and gravity, but can also change as a result of 
    dust destruction and dust growth. This struct also stores dust 
    optical properties (opacities etc.) for all bands. """
mutable struct DustState{ITP_P, ITP_R}

    # Dust temperature for every cell:
    Td::Array{Float64, 2}

    # Grain charges (silicates & graphite):
    Zgr_sil::Array{Float64, 2}
    Zgr_C::Array{Float64, 2}

    # Band absorption opacities (cm² g⁻¹ of dust):
    κd_abs_opt::Float64
    κd_abs_FUV::Float64
    κd_abs_LW::Float64
    κd_abs_EUV1::Float64
    κd_abs_EUV2::Float64
    κd_abs_EUV3::Float64
    κd_abs_Lyα::Float64

    # Band flux opacities (cm² g⁻¹ of dust):
    κd_F_opt::Float64
    κd_F_FUV::Float64
    κd_F_LW::Float64
    κd_F_EUV1::Float64
    κd_F_EUV2::Float64
    κd_F_EUV3::Float64
    κd_F_Lyα::Float64

    # Band albedos:
    ωd_opt::Float64
    ωd_FUV::Float64
    ωd_LW::Float64
    ωd_EUV1::Float64
    ωd_EUV2::Float64
    ωd_EUV3::Float64
    ωd_Lyα::Float64

    # Band scattering asymmetry:
    gd_opt::Float64
    gd_FUV::Float64
    gd_LW::Float64
    gd_EUV1::Float64
    gd_EUV2::Float64
    gd_EUV3::Float64
    gd_Lyα::Float64

    # --- IR mean opacity interpolations ---
    # Store log₁₀(κ) vs. log₁₀(T) interpolations:
    logκP_itp::ITP_P
    logκR_itp::ITP_R
end


""" A struct for holding main hydro arrays, e.g. gas densities 
    and velocities. """
mutable struct HydrodynamicsState{T<:AbstractFloat}
    # Updated hydro state variables:
    ρ_new::Array{T, 2} 
    ρu_X_new::Array{T, 2} 
    ρu_Y_new::Array{T, 2}
    E_new::Array{T, 2}
    # "Old" hydro state variables:
    ρ::Array{T, 2}
    ρu_X::Array{T, 2}
    ρu_Y::Array{T, 2}     
    E::Array{T, 2}
    # Arrays for storing old hydro state, for use in the gravity solver:
    ρ0::Array{T, 2}
    ρu_X0::Array{T, 2}    
    ρu_Y0::Array{T, 2}
    E0::Array{T, 2}
    # Arrays for storing pre-update state in the RK time integration:
    ρ_stage::Array{T, 2}
    ρu_X_stage::Array{T, 2}    
    ρu_Y_stage::Array{T, 2}
    E_stage::Array{T, 2}
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
    # Interface X-fluxes:
    F_ρ_X::Array{T, 2}
    F_ρu_X_X::Array{T, 2}
    F_ρu_Y_X::Array{T, 2}
    F_E_X::Array{T, 2}
    # Reconstructed Y-interface primitive variables:
    ρ_L_Y::Array{T, 2} 
    ρ_R_Y::Array{T, 2} 
    u_Y_L_Y::Array{T, 2} 
    u_Y_R_Y::Array{T, 2}
    u_X_L_Y::Array{T, 2}
    u_X_R_Y::Array{T, 2} 
    P_L_Y::Array{T, 2} 
    P_R_Y::Array{T, 2}
    # Interface Y-fluxes:
    F_ρ_Y::Array{T, 2} 
    F_ρu_X_Y::Array{T, 2} 
    F_ρu_Y_Y::Array{T, 2} 
    F_E_Y::Array{T, 2}
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

    # All dust arrays:
    ρd_new::Array{T, 2} 
    ρdud_X_new::Array{T, 2} 
    ρdud_Y_new::Array{T, 2} 
    ρd::Array{T, 2} 
    ρdud_X::Array{T, 2} 
    ρdud_Y::Array{T, 2} 
    ρd0::Array{T, 2} 
    ρdud_X0::Array{T, 2} 
    ρdud_Y0::Array{T, 2} 
    ρd_stage::Array{T, 2} 
    ρdud_X_stage::Array{T, 2} 
    ρdud_Y_stage::Array{T, 2}
    ud_X::Array{T, 2} 
    ud_Y::Array{T, 2} 
    ρd_L_X::Array{T, 2} 
    ρd_R_X::Array{T, 2} 
    ud_X_L_X::Array{T, 2} 
    ud_X_R_X::Array{T, 2} 
    ud_Y_L_X::Array{T, 2} 
    ud_Y_R_X::Array{T, 2}
    F_ρd_X::Array{T, 2} 
    F_ρdud_X_X::Array{T, 2} 
    F_ρdud_Y_X::Array{T, 2}
    ρd_L_Y::Array{T, 2} 
    ρd_R_Y::Array{T, 2} 
    ud_Y_L_Y::Array{T, 2} 
    ud_Y_R_Y::Array{T, 2} 
    ud_X_L_Y::Array{T, 2} 
    ud_X_R_Y::Array{T, 2} 
    F_ρd_Y::Array{T, 2} 
    F_ρdud_X_Y::Array{T, 2} 
    F_ρdud_Y_Y::Array{T, 2} 
    ρdg_old::Array{T, 2} 
    ud_Xg_old::Array{T, 2} 
    ud_Yg_old::Array{T, 2}
    ρdc::Array{T, 2}
    ρdud_Xc::Array{T, 2} 
    ρdud_Yc::Array{T, 2} 
    ud_Xc::Array{T, 2} 
    ud_Yc::Array{T, 2}

    # Arrays for advecting passive scalars (e.g. ρx_HII, ρZ, etc):
    Scalars::Array{Float64, 3}         # Size (N_X, N_Y, N_Scalars)
    Scalars_old::Array{Float64, 3}     # Size (N_X, N_Y, N_Scalars)
    Scalars_stage::Array{Float64, 3}   # Size (N_X, N_Y, N_Scalars)
    ρ_save::Array{Float64, 2}          # Size (N_X, N_Y), for storing the gas density in RK2 updates of scalar advection 
    
    # Strang split order:
    rfirst_Hydro::Bool

    # Total mass (g) in simulation volume:
    M_simvol::T
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
    P::Array{T, 1}           # Legendre polynomials (for BCs)
    Q::Array{T, 1}           # Multipole moments (for BCs)
    ℓmax::Int                # Maximum ℓ to use in Legendre polynomials
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