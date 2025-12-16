""" FILE:    SetupLymanAlphaArrays.jl

    PURPOSE: Contains the function for setting up the Lyα arrays/struct. 

"""

using .Structs: LyαArrays
using .ConstantsAndFunctions
using Roots                     # Needed for root-finding in the frequency grid setup


const τ_αfloor::Float64 = 1e-5  # Floor optical depth for use in the frequency leakage coefficients

export GenerateLyαArrays


function GenerateLyαArrays( r::AbstractArray{Float64, 1},
                            Z::AbstractArray{Float64, 1},
                            T::AbstractArray{Float64, 2}, # Array of gas temperatures (only sets initial Hvoigt values)
                       Recoil::Bool    = true,            # Atomic recoil (on if true, off if false)
                      Ncoarse::Int     = 40,             # Number of coarse bins (far from line center)
                        Nfine::Int     = 10,             # Number of fine bins (near line center)
                Δν_coarseband::Float64 = 1 * νLyα,        # Width of coarse grid in blue/red directions 
                         Tmin::Float64 = 30.0             # Temperature which sets Doppler width resolution near line center (default = 30 K)
                           )

    ###############################################################
    #######                                                 #######
    #######            F R E Q U E N C Y  G R I D           #######
    #######                                                 #######
    ###############################################################
    # The code below sets up the frequency bins/grid for Lyα

    # Convenient variables:
    Ncoarse_div2 = div(Ncoarse, 2)
    Nfine_div2   = div(Nfine, 2)
    # Get fine grid resolution (= Doppler width @ temperature Tmin):
    Δν_min       = (b(Tmin) / c) * νLyα

    # Get other needed factors:
    ν_fine_L     = νLyα - Nfine_div2 * Δν_min 
    ν_fine_R     = νLyα + Nfine_div2 * Δν_min
    ν_max        = ν_fine_R + Δν_coarseband              # Maximum frequency
    ν_min        = max( ν_fine_L - Δν_coarseband, 0.0 )  # Minimum frequency 

    # Find geometric factor ϵ_L for the left
    f_L(ϵ) = Δν_min * (ϵ^Ncoarse_div2 - 1)/(ϵ - 1) - (ν_fine_L - ν_min)
    ϵ_L    = find_zero(f_L, 1.01 )

    # Find geometric factor ϵ_R for the right:
    f_R(ϵ) = Δν_min * (ϵ^Ncoarse_div2 - 1)/(ϵ - 1) - (ν_max - ν_fine_R)
    ϵ_R    = find_zero(f_R, 1.01 )
    # Use minimum of the two, to ensure all ν > 0:
    ϵ      = min( ϵ_L, ϵ_R )

    # Generate fine grid near line center:
    ν_fine     = collect( ν_fine_L:Δν_min:ν_fine_R )
    # Generate left coarse grid:
    ν_coarse_L = reverse( [ ν_fine_L - Δν_min * (ϵ^n - 1) / (ϵ - 1) for n in 0:(Ncoarse_div2 - 1) ] )
    # Generate right coarse grid:
    ν_coarse_R = [ ν_fine_R + Δν_min * (ϵ^n - 1) / (ϵ - 1) for n in 0:(Ncoarse_div2 - 1) ]
    # Combine coarse & fine grids to get full bin interface grid:
    ν = vcat( ν_coarse_L[1:end-1], ν_fine, ν_coarse_R[2:end] )

    # Get Δν
    N_ν = length(ν) - 1
    Δν  = diff(ν)

    # Get bin centers νC:
    νC  = ν[1:end-1] .+ (Δν ./ 2)

    # Get Voigt profile for each cell:
    N_r    = size( T, 1 )
    N_Z    = size( T, 2 )
    Hvoigt = zeros( N_r, N_Z, N_ν )     # Initialize Hvoigt
    Update_Hvoigt!( Hvoigt, ν, νC, T )  # Compute & fill the array with values

    # Save ~ core region for 3 × 10⁴ K, for fall-back use in advection,
    # where only "active" bins are advected:
    Δν_D3e4   = (b(3e4) / c) * νLyα 
    kcore_min = findfirst( ≥(νLyα - 4 * Δν_D3e4), νC )
    kcore_max = findlast( ≤(νLyα + 4 * Δν_D3e4), νC )


    ###############################################################
    #######                                                 #######
    #######           A D V E C T I O N  A R R A Y S        #######
    #######                                                 #######
    ###############################################################

    # Convenient arrays: 
    zeros_c()  = zeros( Float64, N_r,   N_Z,   N_ν )
    zeros_rf() = zeros( Float64, N_r+1, N_Z,   N_ν )  # For r-interface fluxes 
    zeros_zf() = zeros( Float64, N_r,   N_Z+1, N_ν )  # For Z-interface fluxes 

    # Intensities (latest & old, and stage values for RK2) for all bands: 
    J_Lyα  = zeros_c();  J_Lyα_old  = zeros_c();  J_Lyα_stage  = zeros_c()

    # Lyα energy density:
    e_Lyα  = zeros( Float64, N_r, N_Z )

    # H ≡ ∫dΩ I n̂ / 4π arrays for all bands:
    H_r_Lyα      = zeros_c();  H_Z_Lyα     = zeros_c()
    H_r_Lyα_old  = zeros_c();  H_Z_Lyα_old = zeros_c()

    # Eddington tensor components:
    D_rr_Lyα  = fill( 1/3, N_r, N_Z, N_ν )
    D_ZZ_Lyα  = fill( 1/3, N_r, N_Z, N_ν )
    D_rZ_Lyα  = zeros_c()

    # ϕϕ-component of radiation pressure tensor K = D × J at interfaces, 
    # needed for geometric source term:
    K_ϕϕ_iphalf_Lyα = zeros_c();   K_ϕϕ_imhalf_Lyα = zeros_c()

    # Interface fluxes for all bands: 
    F_r_Lyα     = zeros_rf(); F_Z_Lyα     = zeros_zf()
    F_r_Hr_Lyα  = zeros_rf(); F_Z_Hr_Lyα  = zeros_zf()
    F_r_HZ_Lyα  = zeros_rf(); F_Z_HZ_Lyα  = zeros_zf()

    # Lyα H I opacity at line center: 
    α_Lyα       = zeros( Float64, N_r, N_Z )

    # Flux opacity:
    α_F_Lyα     = zeros_c()

    # Lyα continuum absorption opacity & destruction probability:
    αc_Lyα      = zeros( Float64, N_r, N_Z )
    pd          = zeros( Float64, N_r, N_Z )

    # (Frequency-integrated) emissivitiy, Voigt profile & normalized line profile:
    ∫dν_js_Lyα  = zeros( Float64, N_r, N_Z )
    ϕ_ν         = zeros_c()
    Update_ϕν!( ϕ_ν, Δν, Hvoigt, kcore_min, kcore_max )

    # Frequency diffusion + Doppler shift arrays:
    Fij_kp      = zeros_c()
    Fij_k       = zeros_c()
    Fij_km      = zeros_c()

    # Floor value of α_Lyα, only for use in computing the frequency coefficients,
    # set so that floor optical depth across the simulation volume is ~ τ_αfloor ≪ 1:
    α_Lyα_floor = τ_αfloor * 1 / max( maximum(r), maximum(Z) )

    @assert α_Lyα_floor * max( maximum(r), maximum(Z) ) < 0.1 "α_Lyα_floor = $(α_Lyα_floor). It should be smaller - reduce τ_αfloor."

    # Gas momentum change from Lyα radiation pressure, in r and Z-directions:
    Δρu_Lyα_r      = zeros( Float64, N_r, N_Z )
    Δρu_Lyα_Z      = zeros( Float64, N_r, N_Z )

    # Arrays for storing the Lyα radiation pressure force, in r and Z-directions:
    f_Lyα_r        = zeros( Float64, N_r, N_Z )
    f_Lyα_Z        = zeros( Float64, N_r, N_Z )

    # Strang-split order for M1 advection:
    rfirst_Lyα     = true 

    # Force multiplier & Lyα luminosity:
    M_F::Float64   = 0.0
    L_Lyα::Float64 = 0.0

    return LyαArrays( J_Lyα, J_Lyα_old, J_Lyα_stage, e_Lyα,
                      H_r_Lyα, H_Z_Lyα, H_r_Lyα_old, H_Z_Lyα_old, 
                      D_rr_Lyα, D_ZZ_Lyα, D_rZ_Lyα, K_ϕϕ_iphalf_Lyα, K_ϕϕ_imhalf_Lyα, 
                      F_r_Lyα, F_Z_Lyα, F_r_Hr_Lyα, F_Z_Hr_Lyα, F_r_HZ_Lyα, F_Z_HZ_Lyα, 
                      α_Lyα, α_F_Lyα, αc_Lyα, pd, ∫dν_js_Lyα, Hvoigt, ϕ_ν, Fij_kp, Fij_k, Fij_km, 
                      ν, νC, Δν, kcore_min, kcore_max,
                      Recoil, α_Lyα_floor, Δρu_Lyα_r, Δρu_Lyα_Z, f_Lyα_r, f_Lyα_Z,
                      rfirst_Lyα,
                      M_F, L_Lyα )
end
