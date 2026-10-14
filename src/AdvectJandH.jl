
""" FILE:        AdvectJandH.jl

    PURPOSE:     Contains the code for advecting the mean intensity J, and the
                 first moment H (the Eddington tensor D = K / J is estimated 
                 using an M1 closure). Uses an explicit finite volume method, with 
                 2:nd order (linear) spatial reconstruction, and 1:st order 
                 time integration. The method is dimensionally unsplit. 

    REFERENCES:  (1) Nebrin et al. (2026) 'Lyman-α Pressure Strongly Enhances Pre-Supernova 
                     Feedback at Cosmic Dawn: The First Multi-Dimensional Lyman-α Radiation 
                     Hydrodynamics Simulations', arXiv.

"""

using .ConstantsAndFunctions  # For slope limiters 
using Base.Threads            # For CPU acceleration

# Export functions:
export GetRHDCFLTimeStep, AdvectJandH!


const DEBUG_M1Advection::Bool = false  # For debugging (e.g. checking correct array sizes etc)


""" The default RHD CFL time-step. Computes min( Δr / c̃ ),
    over all cells. To avoid unnecessary allocations, this is 
    done in a for loop. """
@inline @fastmath function GetRHDCFLTimeStep( r::AbstractVector{Float64},   # Interface r-values
                                              Z::AbstractVector{Float64},   # Interface Z-values
                                              c̃::Float64,                   # Reduced speed of light,
                                            u_r::AbstractArray{Float64, 2}, # Gas velocity in r-direction 
                                            u_Z::AbstractArray{Float64, 2}, # Gas velocity in Z-direction
                                            CFL::Float64
                                             )

    N_r         = length(r) - 1  # Grid size
    N_Z         = length(Z) - 1
    dN_r        = 1:N_r
    dN_Z        = 1:N_Z

    min_ΔtRHD   = Inf64  # Initialize the minimum local time-scale
    @inbounds for j in dN_Z
        ΔZ                = Z[j+1] - Z[j] # Vertical cell width
        local_min_ΔtRHD   = min_ΔtRHD     # Local variable for tracking smallest CFL time

        @simd for i in dN_r
            # Compute the horizontal cell width (same for all j-indices):
            Δr = r[i+1] - r[i]

            # Compute the local CFL time-scale, taking into account gas velocities
            # because of the advection & Doppler terms in the RT moment equations:
            u    = hypot( u_r[i,j], u_Z[i,j] )
            Δt   = 0.5 * min( Δr, ΔZ ) / ( c̃ + u )   # Factor 0.5 is 1 / Dimensions = 1 /2.
            
            if Δt < local_min_ΔtRHD 
                # Update the smallest time-scale found:
                local_min_ΔtRHD = Δt 
            end
        end

        min_ΔtRHD = min( min_ΔtRHD, local_min_ΔtRHD )
    end
    
    # The final default CFL time-step is simply the CFL factor times 
    # min_ΔtRHD:
    return CFL * min_ΔtRHD
end


###############################################################
#######                                                 #######
#######                   F L U X E S                   #######
#######                                                 #######
###############################################################


""" Computes the Global Lax-Friedrich (GLF) fluxes for J, H_r, and H_Z. 
    For cells with τ_cell ≳ 1, corrections are used to get the 
    asymptotic diffusion limit. """
@inline function GetFluxGLF( FJ_L::Float64,
                             FJ_R::Float64,
                            FHr_L::Float64, FHr_R::Float64,
                            FHZ_L::Float64, FHZ_R::Float64,
                              J_L::Float64, 
                              J_R::Float64, 
                             Hr_L::Float64, Hr_R::Float64,
                             HZ_L::Float64, HZ_R::Float64,
                              τ_L::Float64,
                              τ_R::Float64,
                              J_i::Float64, J_ip1::Float64 )

    # Effective cell optical depth & correction factor for diffusion limit:
    τ_cell       = @fastmath τ_L + τ_R

    # Compute ϵ, the factor needed to interpolate between optically thick & thin 
    # regimes smoothly:
    numerator_ϵ  = @fastmath 0.2 + τ_cell
    ϵ            = @fastmath numerator_ϵ / ( numerator_ϵ  +  τ_cell * τ_cell )  # → 1 for thin, → 1 / τ_cell for thick 
    oneminusϵ    = @fastmath 1.0 - ϵ

    # Diffusion limit - ∇J / 3α ≈ - (J_{i+1/2} - J_{i}) / (3 τ):
    τ_eff        = @fastmath max( τ_cell, 1e-300 ) # Avoid division by zero 
    F_J_diff     = @fastmath - OneThird * ( J_ip1 - J_i ) / τ_eff

    # GLF flux:
    F_J_GLF      = @fastmath 0.5 * ( FJ_L  +  FJ_R )  -  0.5 * ( J_R - J_L )  

    # Flux for J, interpolating between GLF and diffusion limit:
    F_J          = @fastmath ϵ * F_J_GLF  +  oneminusϵ * F_J_diff

    # Flux for H_r and H_Z, corrected for diffusion limit:
    HalfTimesϵ   = @fastmath 0.5 * ϵ 
    F_Hr         = @fastmath 0.5 * ( FHr_L  +  FHr_R )  -  HalfTimesϵ * ( Hr_R - Hr_L ) 
    F_HZ         = @fastmath 0.5 * ( FHZ_L  +  FHZ_R )  -  HalfTimesϵ * ( HZ_R - HZ_L )

    return F_J, F_Hr, F_HZ
end


# GLF flux function is the same for both directions since 
# wave speeds are not needed:
@inline function GetFluxM1( ::Val{:GLF},
                            ::Val{direction},
                        FJ_L::Float64,
                        FJ_R::Float64,
                       FHr_L::Float64, FHr_R::Float64,
                       FHZ_L::Float64, FHZ_R::Float64,
                         J_L::Float64,
                         J_R::Float64,
                        Hr_L::Float64, Hr_R::Float64,
                        HZ_L::Float64, HZ_R::Float64,
                         τ_L::Float64,
                         τ_R::Float64,
                         J_i::Float64, J_ip1::Float64,
                         f_L::Float64,
                         f_R::Float64,
                        nr_L::Float64, nr_R::Float64,
                        nZ_L::Float64, nZ_R::Float64 ) where {direction}

    return GetFluxGLF( FJ_L, FJ_R,
                      FHr_L, FHr_R,
                      FHZ_L, FHZ_R,
                        J_L, J_R,
                       Hr_L, Hr_R,
                       HZ_L, HZ_R,
                        τ_L, τ_R,
                        J_i, J_ip1 )
end


""" Computes the HLL fluxes for J, H_r, and H_Z. 
    For cells with τ_cell ≳ 1, corrections are used to get the 
    asymptotic diffusion limit (same as for GLF). """
@inline @fastmath function GetFluxHLL( FJ_L::Float64,
                                       FJ_R::Float64,
                                      FHr_L::Float64, FHr_R::Float64,
                                      FHZ_L::Float64, FHZ_R::Float64,
                                        J_L::Float64,
                                        J_R::Float64,
                                       Hr_L::Float64, Hr_R::Float64,
                                       HZ_L::Float64, HZ_R::Float64,
                                       λ1_L::Float64,
                                       λ1_R::Float64,
                                       λ3_L::Float64,
                                       λ3_R::Float64,
                                        τ_L::Float64,
                                        τ_R::Float64,
                                        J_i::Float64, J_ip1::Float64 )

    # Effective cell optical depth:
    τ_cell      = τ_L + τ_R

    # Compute ϵ, the factor needed to interpolate between optically thick & thin 
    # regimes smoothly:
    numerator_ϵ    = 0.2 + τ_cell
    ϵ              = numerator_ϵ / ( numerator_ϵ  +  τ_cell * τ_cell ) # → 1 for thin, → 1 / τ_cell for thick 
    oneminusϵ      = 1.0 - ϵ
    HalfOneminusϵ  = 0.5 * oneminusϵ   # Convenient factor, used twice below for final fluxes for H_r and H_Z 

    # Diffusion limit - ∇J / 3α ≈ - (J_{i+1/2} - J_{i}) / (3 τ):
    τ_eff       = max( τ_cell, 1e-300 )
    F_J_diff    = - OneThird * ( J_ip1 - J_i ) / τ_eff

    # Wavespeeds in units of c̃:
    S_R         = max( max( λ3_L, λ3_R ), 0.0 )
    S_L         = min( min( λ1_L, λ1_R ), 0.0 )
    inv_ΔS      = inv( max( S_R - S_L, 1e-300 ) )
    S_R_S_L     = S_R * S_L

    # Standard HLL fluxes:
    F_J_HLL     = ( S_R * FJ_L   -  S_L * FJ_R   +  S_R_S_L * ( J_R - J_L ) )   * inv_ΔS

    F_Hr_HLL    = ( S_R * FHr_L  -  S_L * FHr_R  +  S_R_S_L * ( Hr_R - Hr_L ) ) * inv_ΔS

    F_HZ_HLL    = ( S_R * FHZ_L  -  S_L * FHZ_R  +  S_R_S_L * ( HZ_R - HZ_L ) ) * inv_ΔS

    # Full flux, which is asymptotically correct in the diffusion limit:
    F_J         = ϵ * F_J_HLL  +  oneminusϵ * F_J_diff
    F_Hr        = ϵ * F_Hr_HLL +  HalfOneminusϵ * ( FHr_L + FHr_R )
    F_HZ        = ϵ * F_HZ_HLL +  HalfOneminusϵ * ( FHZ_L + FHZ_R )

    return F_J, F_Hr, F_HZ
end

# HLL flux for r/X-direction:
@inline function GetFluxM1( ::Val{:HLL},
                            ::Val{:r},
                        FJ_L::Float64,
                        FJ_R::Float64,
                       FHr_L::Float64, FHr_R::Float64,
                       FHZ_L::Float64, FHZ_R::Float64,
                         J_L::Float64,
                         J_R::Float64,
                        Hr_L::Float64, Hr_R::Float64,
                        HZ_L::Float64, HZ_R::Float64,
                         τ_L::Float64,
                         τ_R::Float64,
                         J_i::Float64, J_ip1::Float64,
                         f_L::Float64,
                         f_R::Float64,
                        nr_L::Float64, nr_R::Float64,
                        nZ_L::Float64, nZ_R::Float64 )

    λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_r( f_L, f_R, nr_L, nr_R )

    return GetFluxHLL( FJ_L, FJ_R,
                      FHr_L, FHr_R,
                      FHZ_L, FHZ_R,
                        J_L, J_R,
                       Hr_L, Hr_R,
                       HZ_L, HZ_R,
                       λ1_L, λ1_R,
                       λ3_L, λ3_R,
                        τ_L, τ_R,
                        J_i, J_ip1 )
end

# HLL flux for Z/Y-direction:
@inline function GetFluxM1( ::Val{:HLL},
                            ::Val{:Z},
                        FJ_L::Float64,
                        FJ_R::Float64,
                       FHr_L::Float64, FHr_R::Float64,
                       FHZ_L::Float64, FHZ_R::Float64,
                         J_L::Float64,
                         J_R::Float64,
                        Hr_L::Float64, Hr_R::Float64,
                        HZ_L::Float64, HZ_R::Float64,
                         τ_L::Float64,
                         τ_R::Float64,
                         J_i::Float64, J_ip1::Float64,
                         f_L::Float64,
                         f_R::Float64,
                        nr_L::Float64, nr_R::Float64,
                        nZ_L::Float64, nZ_R::Float64 )

    λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_Z( f_L, f_R, nZ_L, nZ_R )

    return GetFluxHLL( FJ_L, FJ_R,
                      FHr_L, FHr_R,
                      FHZ_L, FHZ_R,
                        J_L, J_R,
                       Hr_L, Hr_R,
                       HZ_L, HZ_R,
                       λ1_L, λ1_R,
                       λ3_L, λ3_R,
                        τ_L, τ_R,
                        J_i, J_ip1 )
end


""" Get advection flux (u/c) * J or (u/c) * H evaluated at interface 
    with velocity u_contact, obtained from the hydro solver. 
    Uses upwind, following Skinner et al. (2019). """
@inline @fastmath function GetAdvectionFlux( u_contact_over_c::Float64, # Gas velocity u at the interface (contact velocity), divided by c
                                                          J_L::Float64, # J_L or H_L 
                                                          J_R::Float64  # J_R or H_R 
                                             ) 
    
    return u_contact_over_c * ifelse( u_contact_over_c < 0.0, J_R, J_L )
end


""" Add the O(u/c) aberration/stretching term in the H equation. 
    In vector notation it reads:

        (1/c̃) ∂H/∂t   = - (1/c) H⋅∇u

    Or in index notation:

        (1/c̃) ∂H_i/∂t = - (1/c) H_j ∂_j u_i

    In 2D this gives a change in H:
        ΔH_r = - (c̃Δt/c) * ( H_r ∂u_r/∂r + H_Z ∂u_r/∂Z )
        ΔH_Z = - (c̃Δt/c) * ( H_r ∂u_Z/∂r + H_Z ∂u_Z/∂Z )

    The old H values (from last advection update) must be used on the right, so that the
    update is truly explicit, and not order-dependent. """
@inline @fastmath function ΔH_Aberration( H_r_old::Float64,
                                          H_Z_old::Float64,
                                       c̃Δt_over_c::Float64,
                                          ∂u_r_∂r::Float64,
                                          ∂u_r_∂Z::Float64,
                                          ∂u_Z_∂r::Float64,
                                          ∂u_Z_∂Z::Float64 )

    ΔH_r = - c̃Δt_over_c * ( H_r_old * ∂u_r_∂r  +  H_Z_old * ∂u_r_∂Z )
    ΔH_Z = - c̃Δt_over_c * ( H_r_old * ∂u_Z_∂r  +  H_Z_old * ∂u_Z_∂Z )

    return ΔH_r, ΔH_Z
end


""" Solves the PDE (1/c̃) (∂Q/∂t) + ∇⋅F = 0 over a time-step Δt,
    using explicit time-integration, and the finite-volume 
    method. 
    
    **ARGUMENTS:**
    
    Q        :   Value of Q at "old" time-step n
    Δw       :   c̃ × Δt 
    inv_ΔV   :   Inverse volume 1 / ΔV of cell i 
    Ar_L     :   Area of left interface i - 1/2
    Ar_R     :   Area of right interface i + 1/2
    AZ_L     :   Area of left interface j - 1/2
    AZ_R     :   Area of right interface j + 1/2
    Fr_L     :   Flux F at left interface i - 1/2
    Fr_R     :   Flux F at right interface i + 1/2
    FZ_L     :   Flux F at left interface j - 1/2
    FZ_R     :   Flux F at right interface j + 1/2

    **RETURNS:**

    Qnew    :   Value of Q at time-step n + 1  """
@inline @fastmath  function UpdateConservativeQuantity( Q::Float64, 
                                                      c̃Δt::Float64,
                                                   inv_ΔV::Float64,
                                                     Ar_L::Float64,                                     
                                                     Ar_R::Float64,
                                                     AZ_L::Float64,
                                                     AZ_R::Float64,
                                                     Fr_L::Float64,
                                                     Fr_R::Float64,
                                                     FZ_L::Float64,
                                                     FZ_R::Float64
                                                       )

    FluxTerms_r = Ar_R * Fr_R  -  Ar_L * Fr_L
    FluxTerms_Z = AZ_R * FZ_R  -  AZ_L * FZ_L
    Qnew        = Q  -  ( c̃Δt * inv_ΔV ) * ( FluxTerms_r + FluxTerms_Z )
    
    return Qnew
end


""" This function performs a second-order reconstruction of J and H 
    on an interface using PLM. """
@inline @fastmath function ReconstructInterface( J_i::Float64,   J_ip1::Float64,  J_ip2::Float64,  J_im1::Float64,
                                                Hr_i::Float64,  Hr_ip1::Float64, Hr_ip2::Float64, Hr_im1::Float64,
                                                HZ_i::Float64,  HZ_ip1::Float64, HZ_ip2::Float64, HZ_im1::Float64,
 
                                                Δr_L::Float64, Δr_R::Float64, 
                                                inv_dr_i_im1::Float64, inv_dr_ip1_i::Float64, inv_dr_ip2_ip1::Float64,
                              
                                                α_F_i::Float64, α_F_ip1::Float64 )

    # (Flux) optical depth between cell centers & inteface:
    τ_L        = α_F_i   * Δr_L
    τ_R        = α_F_ip1 * Δr_R

    # Compute reduced flux components:
    inv_J_i    = 1 / max( J_i,   Jeps )
    inv_J_ip1  = 1 / max( J_ip1, Jeps )
    inv_J_ip2  = 1 / max( J_ip2, Jeps )
    inv_J_im1  = 1 / max( J_im1, Jeps )

    fr_i       = Hr_i   * inv_J_i
    fr_ip1     = Hr_ip1 * inv_J_ip1
    fr_ip2     = Hr_ip2 * inv_J_ip2
    fr_im1     = Hr_im1 * inv_J_im1
    
    fZ_i       = HZ_i   * inv_J_i 
    fZ_ip1     = HZ_ip1 * inv_J_ip1
    fZ_ip2     = HZ_ip2 * inv_J_ip2
    fZ_im1     = HZ_im1 * inv_J_im1

    # Compute slope-limited derivatives for J:
    ∂J∂r_ip1_i  = (J_ip1 - J_i) * inv_dr_ip1_i  # Used twice below 

    ∂J∂r_L      = MinMod( (J_i   - J_im1) * inv_dr_i_im1,
                          ∂J∂r_ip1_i )
     
    ∂J∂r_R      = MinMod( (J_ip2 - J_ip1) * inv_dr_ip2_ip1,
                          ∂J∂r_ip1_i )

    # Compute sloe-limited derivatives of fr (r-component of reduced flux):
    ∂fr∂r_ip1_i = (fr_ip1 - fr_i) * inv_dr_ip1_i  # Used twice below 

    ∂fr∂r_L     = MinMod( (fr_i   - fr_im1)   * inv_dr_i_im1,
                          ∂fr∂r_ip1_i )
     
    ∂fr∂r_R     = MinMod( (fr_ip2 - fr_ip1)   * inv_dr_ip2_ip1,
                          ∂fr∂r_ip1_i )

    # Compute slope-limited derivatives of fZ (Z-component of reduced flux):  
    ∂fZ∂r_ip1_i = (fZ_ip1 - fZ_i) * inv_dr_ip1_i  # Used twice below 

    ∂fZ∂r_L     = MinMod( (fZ_i   - fZ_im1)   * inv_dr_i_im1,
                          ∂fZ∂r_ip1_i )
     
    ∂fZ∂r_R     = MinMod( (fZ_ip2 - fZ_ip1)   * inv_dr_ip2_ip1,
                          ∂fZ∂r_ip1_i )                

    # Get all J and f (reduced flux) interface variables:
    J_L      = max( J_i    +  ∂J∂r_L * Δr_L, Jeps )
    J_R      = max( J_ip1  -  ∂J∂r_R * Δr_R, Jeps )

    fr_L     = fr_i     +  ∂fr∂r_L * Δr_L
    fr_R     = fr_ip1   -  ∂fr∂r_R * Δr_R

    fZ_L     = fZ_i     +  ∂fZ∂r_L * Δr_L
    fZ_R     = fZ_ip1   -  ∂fZ∂r_R * Δr_R

    # Ensure causality, i.e. f ≤ 1:
    f_L = sqrt( fr_L * fr_L  +  fZ_L * fZ_L )
    if f_L > 1.0
        # Renormalize:
        inv_f = inv( f_L )
        fr_L *= inv_f 
        fZ_L *= inv_f
        f_L   = 1.0
    end

    f_R = sqrt( fr_R * fr_R  +  fZ_R * fZ_R )
    if f_R > 1.0
        # Renormalize:
        inv_f = inv( f_R )
        fr_R *= inv_f 
        fZ_R *= inv_f
        f_R   = 1.0
    end

    # Compute H_r and H_Z from the interface J and f values,
    # and preserving the flux directions (via nr and nZ):
    Hr_L     = fr_L  * J_L
    Hr_R     = fr_R  * J_R

    HZ_L     = fZ_L  * J_L
    HZ_R     = fZ_R  * J_R

    # Compute flux normal vectors:
    if f_L > 0.0
        inv_f_L = inv( f_L )
        nr_L    = fr_L * inv_f_L
        nZ_L    = fZ_L * inv_f_L
    else
        nr_L    = 0.0
        nZ_L    = 0.0
    end

    if f_R > 0.0
        inv_f_R = inv( f_R )
        nr_R    = fr_R * inv_f_R
        nZ_R    = fZ_R * inv_f_R
    else
        nr_R    = 0.0
        nZ_R    = 0.0
    end

    return J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, τ_L, τ_R,
           f_L, f_R, nr_L, nr_R, nZ_L, nZ_R
end



""" Enforce cell-centered M1 realizability: |H| / J ≤ 1. """
@inline @fastmath function EnforceCausalReducedFlux( J::Float64,
                                                   H_r::Float64,
                                                   H_Z::Float64 )

    J            = max( J, Jeps )
    Hmag_squared = H_r * H_r  +  H_Z * H_Z

    if Hmag_squared > J * J
        Hmag  = sqrt( Hmag_squared )
        fac   = J / Hmag
        H_r  *= fac
        H_Z  *= fac
    end

    return H_r, H_Z
end


""" This function computes K_ϕϕ at the innermost cylindrical axis r = 0 (i.e. i = 1/2),
    needed for the geometric source term. This is done separately here to ensure that 
    small errors do not creep in from reconstruction. """
@inline @fastmath function Compute_K_ϕϕ_InnerAxis( J_axis::Float64, HZ_axis::Float64 )

    J_axis = max( J_axis, Jeps )
    fZ     = HZ_axis / J_axis 
    fZ     = clamp( fZ, - 1.0, 1.0 )
    f      = abs( fZ )    # We enforce that there is only axis along Z-axis at r = 0
    
    # Set radial unit flux vector at axis to zero exactly, 
    # and then set the vertical unit flux vector:
    nr     = 0.0
    if f > 0.0
        nZ = sign(fZ) 
    else
        nZ = 0.0 
    end

    # Compute Eddington factor Drr: 
    D_rr, D_ZZ, _ = Compute_D_Edd_from_f_n( f, nr, nZ )

    # Get K_ϕϕ at the inner axis r = 0:
    K_ϕϕ_InnerAxis = clamp( 1.0 - D_rr - D_ZZ, 0.0, 1.0 ) * J_axis 

    return K_ϕϕ_InnerAxis
end 



@inline function AdvectJandH_Cylindrical!( c̃Δt::Float64,                   # c̃ × Δt 
                                           α_F::AbstractArray{Float64, 3}, # Flux opacities              (N_ν × N_r x N_Z)
   
                                             J::AbstractArray{Float64, 3}, # J (latest/updated values)   (N_ν × N_r x N_Z)
                                         J_old::AbstractArray{Float64, 3}, # J (old values)              (N_ν × N_r x N_Z)

                                           H_r::AbstractArray{Float64, 3}, # r-direction component of H  (N_ν × N_r x N_Z)
                                           H_Z::AbstractArray{Float64, 3}, # Z-direction component of H  (N_ν × N_r x N_Z)

                                   K_ϕϕ_iphalf::AbstractArray{Float64, 3}, # ϕϕ-components of K at i+1/2           (N_ν × N_r x N_Z)
                                   K_ϕϕ_imhalf::AbstractArray{Float64, 3}, # ϕϕ-components of K at i-1/2           (N_ν × N_r x N_Z)

                                           F_r::AbstractArray{Float64, 3}, # r-direction flux of J               (N_ν × N_r+1 x N_Z)
                                           F_Z::AbstractArray{Float64, 3}, # Z-direction flux of J               (N_ν × N_r   x N_Z+1)

                                        F_r_Hr::AbstractArray{Float64, 3}, # r-direction flux of H_r             (N_ν × N_r+1 x N_Z)
                                        F_Z_Hr::AbstractArray{Float64, 3}, # Z-direction flux of H_r             (N_ν × N_r   x N_Z+1)
                                        F_r_HZ::AbstractArray{Float64, 3}, # r-direction flux of H_Z             (N_ν × N_r+1 x N_Z)
                                        F_Z_HZ::AbstractArray{Float64, 3}, # Z-direction flux of H_Z             (N_ν × N_r   x N_Z+1)

                                        inv_ΔV::AbstractArray{Float64, 2}, # Inverse cell volumes 1 / ΔV         (N_r   x N_Z)
                                           A_r::AbstractArray{Float64, 2}, # Cell interface areas in r-direction (N_r+1 x N_Z)
                                           A_Z::AbstractArray{Float64, 2}, # Cell interface areas in Z-direction (N_r   x N_Z+1)
                                    inv_r_cell::AbstractArray{Float64},    # ⟨1/r⟩ for every cell                (N_r)

                                   # Gas velocity field:
                                      u_r_star::AbstractArray{Float64,2},
                                      u_Z_star::AbstractArray{Float64,2},
                                       ∂u_r_∂r::AbstractArray{Float64,2},
                                       ∂u_r_∂Z::AbstractArray{Float64,2}, 
                                       ∂u_Z_∂r::AbstractArray{Float64,2}, 
                                       ∂u_Z_∂Z::AbstractArray{Float64,2},
                                    u_r_over_r::AbstractArray{Float64,2},

                                # Needed intervals:
                            Δr_L::AbstractVector{Float64}, Δr_R::AbstractVector{Float64},
                                  inv_dr_i_im1::AbstractVector{Float64},
                                  inv_dr_ip1_i::AbstractVector{Float64},
                                inv_dr_ip2_ip1::AbstractVector{Float64},

                            Δr_L_InnerBoundary::Float64, Δr_R_InnerBoundary::Float64,
                    inv_dr_i_im1_InnerBoundary::Float64,
                    inv_dr_ip1_i_InnerBoundary::Float64,
                  inv_dr_ip2_ip1_InnerBoundary::Float64,
 
                            Δr_L_OuterBoundary::Float64, Δr_R_OuterBoundary::Float64,
                    inv_dr_i_im1_OuterBoundary::Float64,
                    inv_dr_ip1_i_OuterBoundary::Float64,
                  inv_dr_ip2_ip1_OuterBoundary::Float64,

                             ΔZ_L::AbstractVector{Float64}, ΔZ_R::AbstractVector{Float64},
                                  inv_dZ_j_jm1::AbstractVector{Float64},
                                  inv_dZ_jp1_j::AbstractVector{Float64},
                                inv_dZ_jp2_jp1::AbstractVector{Float64},

                            ΔZ_L_InnerBoundary::Float64, ΔZ_R_InnerBoundary::Float64,
                    inv_dZ_j_jm1_InnerBoundary::Float64,
                    inv_dZ_jp1_j_InnerBoundary::Float64,
                  inv_dZ_jp2_jp1_InnerBoundary::Float64,
 
                            ΔZ_L_OuterBoundary::Float64, ΔZ_R_OuterBoundary::Float64,
                    inv_dZ_j_jm1_OuterBoundary::Float64,
                    inv_dZ_jp1_j_OuterBoundary::Float64,
                  inv_dZ_jp2_jp1_OuterBoundary::Float64,
            
                                    # Other stuff:
                                    UpdateJold::Bool,         # Whether to update J_old = J at the end (should be false for stellar RT)

                                    BC_r_right::Symbol,       # Boundary condition at i = N_r + 1/2
                                      BC_Z_bot::Symbol,       # Boundary condition at j = 1/2 
                                      BC_Z_top::Symbol,       # Boundary condition at j = N_Z + 1/2 
                                          kmin::Int    = 1,
                                          kmax::Int    = size( J, 1 ),
                                    FluxSolver::Val    = Val(:HLL)
                                    ) 

    # Grid & intervals:
    N_ν, N_r, N_Z   = size( J )
    dN_r            = 1:N_r
    dN_Z            = 1:N_Z
    dN_ν            = max( kmin, 1):min( kmax, N_ν )  # Range of frequency bins to advect 
    dN_rint         = 2:(N_r-2)
    dN_Zint         = 2:(N_Z-2)

    # Convenient constant:
    c̃Δt_over_c      = @fastmath c̃Δt * inv_c

    if DEBUG_M1Advection
        @assert size( u_r_star, 1 ) == N_r + 1 "Contact velocity in r-direction/X-direction must have size $(N_r+1)."
        @assert size( u_Z_star, 2 ) == N_Z + 1 "Contact velocity in Z-direction/Y-direction must have size $(N_Z+1)."
    end

    ###############################################################
    #######                                                 #######
    #######           C O M P U T E  F L U X E S            #######
    #######                                                 #######
    ###############################################################

    # We start by reconstructing & computing the interface fluxes, 
    # starting with the r-direction (order does not matter here).

    # Intervals used many times below (for each j and bin):
    Δr_L_1              = @inbounds Δr_L[1] 
    Δr_R_1              = @inbounds Δr_R[1]
    inv_dr_i_im1__1     = @inbounds inv_dr_i_im1[1]
    inv_dr_ip1_i__1     = @inbounds inv_dr_ip1_i[1]
    inv_dr_ip2_ip1__1   = @inbounds inv_dr_ip2_ip1[1]

    Δr_L_Nm1            = @inbounds Δr_L[N_r-1]
    Δr_R_Nm1            = @inbounds Δr_R[N_r-1]
    inv_dr_i_im1__Nm1   = @inbounds inv_dr_i_im1[N_r-1]
    inv_dr_ip1_i__Nm1   = @inbounds inv_dr_ip1_i[N_r-1]
    inv_dr_ip2_ip1__Nm1 = @inbounds inv_dr_ip2_ip1[N_r-1]
    
    # r-direction fluxes:
    @threads for j in dN_Z
        @inbounds begin

            u_contact_over_c_1j = @fastmath u_r_star[1,j] * inv_c
            u_contact_over_c_2j = @fastmath u_r_star[2,j] * inv_c

            @simd for k in dN_ν

                J_1j   = J[k,1,j]
                J_2j   = J[k,2,j]
                H_r_1j = H_r[k,1,j]
                H_r_2j = H_r[k,2,j]
                H_Z_1j = H_Z[k,1,j]
                H_Z_2j = H_Z[k,2,j]

                α_F_1j = α_F[k,1,j]

                ##############################################################################
                ##############          INTERFACE i = 1/2          ###########################
                ##############################################################################

                # NOTE: In cylindrical coordinates we always have reflective BCs at the innermost 
                # boundary, so use this instead of adding a more expensive if statement in an inner 
                # loop:

                # Ghost cell i = 0:
                J_0    = J_1j 
                Hr_0   = - H_r_1j   # Reflect radial flux
                HZ_0   = H_Z_1j
                # Ghost cell i = -1:
                J_m1   = J_2j 
                Hr_m1  = - H_r_2j   # Reflect radial flux
                HZ_m1  = H_Z_2j

                # Reconstruct variables at interface r_{1/2} (i.e. r[1]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R  = ReconstructInterface( J_0,     J_1j,   J_2j,   J_m1,
                                                    Hr_0,    H_r_1j, H_r_2j, Hr_m1,
                                                    HZ_0,    H_Z_1j, H_Z_2j, HZ_m1,

                                                    Δr_L_InnerBoundary, Δr_R_InnerBoundary,
                                                    inv_dr_i_im1_InnerBoundary,
                                                    inv_dr_ip1_i_InnerBoundary,
                                                    inv_dr_ip2_ip1_InnerBoundary,
                    
                                                    α_F_1j, α_F_1j  # Assume ghost cell 0 has same opacity as cell i = 1
                                                   )

                # Compute interface D-values:
                D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # Compute K = D × J at the interface:
                Krr_L = @fastmath D_rr_L * J_L
                Krr_R = @fastmath D_rr_R * J_R
                KrZ_L = @fastmath D_rZ_L * J_L
                KrZ_R = @fastmath D_rZ_R * J_R

                # Fluxes at i = 1/2:
                F_r_1j, F_r_Hr_1j, F_r_HZ_1j = GetFluxM1( FluxSolver, Val(:r), 
                                                          Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                          J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R,   
                                                          τ_L, τ_R, J_0, J_1j, f_L, f_R,
                                                          nr_L, nr_R, nZ_L, nZ_R )

                # Add advection fluxes:
                F_r[k,1,j]           = @fastmath F_r_1j    + GetAdvectionFlux( u_contact_over_c_1j, J_L, J_R )
                F_r_Hr[k,1,j]        = @fastmath F_r_Hr_1j + GetAdvectionFlux( u_contact_over_c_1j, Hr_L, Hr_R )
                F_r_HZ[k,1,j]        = @fastmath F_r_HZ_1j + GetAdvectionFlux( u_contact_over_c_1j, HZ_L, HZ_R )

                # Fill K_ϕϕ at i = 1/2 for cell i = 1, needed for the geometric source term, 
                # in a way that enforces H_r = 0 at r = 0 to machine precision:
                K_ϕϕ_R               = Compute_K_ϕϕ_InnerAxis( J_R, HZ_R )
                K_ϕϕ_imhalf[k,1,j]   = K_ϕϕ_R 

                ##############################################################################
                ##############          INTERFACE i = 3/2          ###########################
                ##############################################################################

                # Reconstruct variables at interface r_{3/2} (i.e. r[2]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R = ReconstructInterface( J_1j,   J_2j,   J[k,3,j],   J_0,
                                                   H_r_1j, H_r_2j, H_r[k,3,j], Hr_0,
                                                   H_Z_1j, H_Z_2j, H_Z[k,3,j], HZ_0,

                                                   Δr_L_1, Δr_R_1,
                                                   inv_dr_i_im1__1,
                                                   inv_dr_ip1_i__1,
                                                   inv_dr_ip2_ip1__1,

                                                   α_F_1j, α_F[k,2,j] )

                # Compute interface D-values:
                D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # Compute K = D × J at the interface:
                Krr_L = @fastmath D_rr_L * J_L 
                Krr_R = @fastmath D_rr_R * J_R 
                KrZ_L = @fastmath D_rZ_L * J_L
                KrZ_R = @fastmath D_rZ_R * J_R

                # Fluxes at i = 3/2:
                F_r_2j, F_r_Hr_2j, F_r_HZ_2j = GetFluxM1( FluxSolver, Val(:r), 
                                                          Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                          J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                          τ_L, τ_R, J_1j, J_2j, f_L, f_R,
                                                          nr_L, nr_R, nZ_L, nZ_R )

                # Add advection fluxes:
                F_r[k,2,j]          = @fastmath F_r_2j     +  GetAdvectionFlux( u_contact_over_c_2j, J_L, J_R )
                F_r_Hr[k,2,j]       = @fastmath F_r_Hr_2j  +  GetAdvectionFlux( u_contact_over_c_2j, Hr_L, Hr_R )
                F_r_HZ[k,2,j]       = @fastmath F_r_HZ_2j  +  GetAdvectionFlux( u_contact_over_c_2j, HZ_L, HZ_R )

                # Get K_ϕϕ at interfaces:
                K_ϕϕ_L              = clamp( 1.0 - D_rr_L - D_ZZ_L, 0.0, 1.0 ) * J_L 
                K_ϕϕ_R              = clamp( 1.0 - D_rr_R - D_ZZ_R, 0.0, 1.0 ) * J_R 

                K_ϕϕ_iphalf[k,1,j]  = K_ϕϕ_L    # K_ϕϕ at i+1/2 for cell i = 1
                K_ϕϕ_imhalf[k,2,j]  = K_ϕϕ_R    # K_ϕϕ at i-1/2 for cell i = 2

            end # End of k-loop 

            ##############################################################################
            ##############        INTERIOR INTERFACES 2 ≤ i ≤ N_r-2          #############
            ##############################################################################

            for i in dN_rint
                ip1 = i + 1
                ip2 = i + 2
                im1 = i - 1

                Δr_L_i            = Δr_L[i]
                Δr_R_i            = Δr_R[i]
                inv_dr_i_im1__i   = inv_dr_i_im1[i]
                inv_dr_ip1_i__i   = inv_dr_ip1_i[i]
                inv_dr_ip2_ip1__i = inv_dr_ip2_ip1[i]

                # u_r_star[i+1] / c is used for every bin:
                u_contact_over_c_ip1 = @fastmath u_r_star[ip1,j] * inv_c

                @simd for k in dN_ν

                    # Used twice below:
                    J_i   = J[k,i,j]
                    J_ip1 = J[k,ip1,j]

                    # Reconstruct variables at interface r_{i+1/2} (i.e. r[i+1]):
                    J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                    τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                    nZ_L, nZ_R = ReconstructInterface( J_i,        J_ip1,        J[k,ip2,j],   J[k,im1,j],
                                                       H_r[k,i,j], H_r[k,ip1,j], H_r[k,ip2,j], H_r[k,im1,j],
                                                       H_Z[k,i,j], H_Z[k,ip1,j], H_Z[k,ip2,j], H_Z[k,im1,j],

                                                       Δr_L_i, Δr_R_i,
                                                       inv_dr_i_im1__i,
                                                       inv_dr_ip1_i__i,
                                                       inv_dr_ip2_ip1__i,

                                                       α_F[k,i,j], α_F[k,ip1,j] )

                    # Compute interface D-values:
                    D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                    D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                    # Compute K = D × J at the interface:
                    Krr_L = @fastmath D_rr_L * J_L 
                    Krr_R = @fastmath D_rr_R * J_R 
                    KrZ_L = @fastmath D_rZ_L * J_L
                    KrZ_R = @fastmath D_rZ_R * J_R

                    # Radial flux of J, H_r, and H_Z:
                    F_r_ip1, F_r_Hr_ip1, F_r_HZ_ip1 = GetFluxM1( FluxSolver, Val(:r), 
                                                                 Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                                 J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                                 τ_L, τ_R, J_i, J_ip1, f_L, f_R,
                                                                 nr_L, nr_R, nZ_L, nZ_R )

                    # Add advection fluxes:
                    F_r[k,ip1,j]         = @fastmath F_r_ip1     +  GetAdvectionFlux( u_contact_over_c_ip1, J_L, J_R )
                    F_r_Hr[k,ip1,j]      = @fastmath F_r_Hr_ip1  +  GetAdvectionFlux( u_contact_over_c_ip1, Hr_L, Hr_R )
                    F_r_HZ[k,ip1,j]      = @fastmath F_r_HZ_ip1  +  GetAdvectionFlux( u_contact_over_c_ip1, HZ_L, HZ_R )

                    # Get K_ϕϕ at interfaces:
                    K_ϕϕ_L               = clamp( 1.0 - D_rr_L - D_ZZ_L, 0.0, 1.0 ) * J_L 
                    K_ϕϕ_R               = clamp( 1.0 - D_rr_R - D_ZZ_R, 0.0, 1.0 ) * J_R 

                    K_ϕϕ_iphalf[k,i,j]   = K_ϕϕ_L    # K_ϕϕ at i+1/2 for cell i
                    K_ϕϕ_imhalf[k,ip1,j] = K_ϕϕ_R    # K_ϕϕ at (i+1)-1/2 for cell i + 1

                end # End of k-loop 
            end # End of interior i ∈ 2:(N_r-2) loop


            # u_r_star[N_r] / c and u_r_star[N_r+1] / c is used for every bin:
            u_contact_over_c_Nj   = @fastmath u_r_star[N_r,j]   * inv_c
            u_contact_over_c_Np1j = @fastmath u_r_star[N_r+1,j] * inv_c

            @simd for k in dN_ν

                J_N     = J[k, N_r, j]
                J_Nm1   = J[k, N_r - 1, j]
                H_r_N   = H_r[k, N_r, j]
                H_r_Nm1 = H_r[k, N_r - 1, j]
                H_Z_N   = H_Z[k, N_r, j]
                H_Z_Nm1 = H_Z[k, N_r - 1, j]

                α_F_N   = α_F[k,N_r,j]

                # Boundary conditions at i = N_r + 1:
                if BC_r_right === :Reflect
                    # Ghost cell i = N_r + 1:
                    J_Np1   =   J_N
                    Hr_Np1  = - H_r_N    # Reflect 
                    HZ_Np1  =   H_Z_N 
                    # Ghost cell i = N_R + 2:
                    J_Np2   =   J_Nm1
                    Hr_Np2  = - H_r_Nm1  # Reflect 
                    HZ_Np2  =   H_Z_Nm1  
                elseif BC_r_right === :Outflow
                    # Ghost cell i = N_r + 1:
                    J_Np1   =  J_N
                    Hr_Np1  =  max( H_r_N, 0.0 )
                    HZ_Np1  =  H_Z_N
                    # Ghost cell i = N_R + 2:
                    J_Np2   =  J_Np1
                    Hr_Np2  =  Hr_Np1
                    HZ_Np2  =  HZ_Np1
                end

                ##############################################################################
                ##############         INTERFACE i = N_r - 1/2         #######################
                ##############################################################################

                # Reconstruct variables at interface r_{N_r-1/2} (i.e. r[N_r]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R = ReconstructInterface( J_Nm1,   J_N,   J_Np1,    J[k,N_r-2,j],
                                                   H_r_Nm1, H_r_N, Hr_Np1,   H_r[k,N_r-2,j],
                                                   H_Z_Nm1, H_Z_N, HZ_Np1,   H_Z[k,N_r-2,j],

                                                   Δr_L_Nm1, Δr_R_Nm1,
                                                   inv_dr_i_im1__Nm1,
                                                   inv_dr_ip1_i__Nm1,
                                                   inv_dr_ip2_ip1__Nm1,

                                                   α_F[k,N_r-1,j], α_F_N )

                # Compute interface D-values:
                D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # K = D × J at the interface i = N_r - 1/2:
                Krr_L = @fastmath D_rr_L * J_L
                Krr_R = @fastmath D_rr_R * J_R
                KrZ_L = @fastmath D_rZ_L * J_L
                KrZ_R = @fastmath D_rZ_R * J_R

                # Fluxes at interface i = N_r - 1/2:
                F_r_Nj, F_r_Hr_Nj, F_r_HZ_Nj = GetFluxM1( FluxSolver, Val(:r), 
                                                          Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                          J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                          τ_L, τ_R, J_Nm1, J_N, f_L, f_R,
                                                          nr_L, nr_R, nZ_L, nZ_R )

                # Add advection fluxes:
                F_r[k,N_r,j]      = @fastmath F_r_Nj     +  GetAdvectionFlux( u_contact_over_c_Nj, J_L, J_R )
                F_r_Hr[k,N_r,j]   = @fastmath F_r_Hr_Nj  +  GetAdvectionFlux( u_contact_over_c_Nj, Hr_L, Hr_R )
                F_r_HZ[k,N_r,j]   = @fastmath F_r_HZ_Nj  +  GetAdvectionFlux( u_contact_over_c_Nj, HZ_L, HZ_R )

                # Get K_ϕϕ at interfaces:
                K_ϕϕ_L                 = clamp( 1.0 - D_rr_L - D_ZZ_L, 0.0, 1.0 ) * J_L 
                K_ϕϕ_R                 = clamp( 1.0 - D_rr_R - D_ZZ_R, 0.0, 1.0 ) * J_R 

                K_ϕϕ_iphalf[k,N_r-1,j] = K_ϕϕ_L    # K_ϕϕ at i+1/2 for cell i = N_r - 1
                K_ϕϕ_imhalf[k,N_r,j]   = K_ϕϕ_R    # K_ϕϕ at i-1/2 for cell i = N_r 

                ##############################################################################
                ##############         INTERFACE i = N_r + 1/2         #######################
                ##############################################################################

                # Reconstruct variables at interface r_{N_r+1/2} (i.e. r[N_r+1]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R = ReconstructInterface( J_N,   J_Np1,    J_Np2,    J_Nm1,
                                                   H_r_N, Hr_Np1,   Hr_Np2,   H_r_Nm1,
                                                   H_Z_N, HZ_Np1,   HZ_Np2,   H_Z_Nm1,

                                                   Δr_L_OuterBoundary, Δr_R_OuterBoundary,
                                                   inv_dr_i_im1_OuterBoundary,
                                                   inv_dr_ip1_i_OuterBoundary,
                                                   inv_dr_ip2_ip1_OuterBoundary,

                                                   α_F_N, α_F_N  # Assume opacity in ghost cell N_r+1 is same as for cell N_r
                                                  )

                # Compute interface D-values:
                D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # K = D × J at the interface i = N_r + 1/2:
                Krr_L = @fastmath D_rr_L * J_L
                Krr_R = @fastmath D_rr_R * J_R
                KrZ_L = @fastmath D_rZ_L * J_L
                KrZ_R = @fastmath D_rZ_R * J_R

                # Fluxes at interface i = N_r + 1/2:
                F_r_Np1j, F_r_Hr_Np1j, F_r_HZ_Np1j = GetFluxM1( FluxSolver, Val(:r), 
                                                                Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                                τ_L, τ_R, J_N, J_Np1, f_L, f_R,
                                                                nr_L, nr_R, nZ_L, nZ_R )

                # Add advection fluxes:
                F_r[k,N_r+1,j]       = @fastmath F_r_Np1j     +  GetAdvectionFlux( u_contact_over_c_Np1j, J_L, J_R )
                F_r_Hr[k,N_r+1,j]    = @fastmath F_r_Hr_Np1j  +  GetAdvectionFlux( u_contact_over_c_Np1j, Hr_L, Hr_R )
                F_r_HZ[k,N_r+1,j]    = @fastmath F_r_HZ_Np1j  +  GetAdvectionFlux( u_contact_over_c_Np1j, HZ_L, HZ_R )

                # Get K_ϕϕ at interface i = N_r + 1/2:
                K_ϕϕ_L               = clamp( 1.0 - D_rr_L - D_ZZ_L, 0.0, 1.0 ) * J_L 
                K_ϕϕ_iphalf[k,N_r,j] = K_ϕϕ_L    # K_ϕϕ at i+1/2 for cell i = N_r

            end # End of k-loop 

        end # End of inbounds     
    end # End of j-loop 

    # Intervals used many times below (for each i and bin):
    ΔZ_L_1            = @inbounds ΔZ_L[1]
    ΔZ_R_1            = @inbounds ΔZ_R[1]
    inv_dZ_j_jm1__1   = @inbounds inv_dZ_j_jm1[1]
    inv_dZ_jp1_j__1   = @inbounds inv_dZ_jp1_j[1]
    inv_dZ_jp2_jp1__1 = @inbounds inv_dZ_jp2_jp1[1]

    # Next, Z-direction fluxes:
    @threads for i in dN_r
        @inbounds begin

            u_contact_over_c_i1 = @fastmath u_Z_star[i,1] * inv_c
            u_contact_over_c_i2 = @fastmath u_Z_star[i,2] * inv_c

            @simd for k in dN_ν

                # Load commonly used values of J and H, and α_F:
                J_i1   = J[k,i,1]
                J_i2   = J[k,i,2]
                H_r_i1 = H_r[k,i,1]
                H_r_i2 = H_r[k,i,2]
                H_Z_i1 = H_Z[k,i,1]
                H_Z_i2 = H_Z[k,i,2]

                α_F_i1 = α_F[k,i,1]

                ##############################################################################
                ####################         INTERFACE j = 1/2         #######################
                ##############################################################################

                if BC_Z_bot === :Reflect
                    # Ghost cell j = 0:
                    J_0   =   J_i1
                    HZ_0  = - H_Z_i1   # Reflect
                    Hr_0  =   H_r_i1
                    # Ghost cell j = -1 (mirror of cell j = 2):
                    J_m1  =   J_i2
                    HZ_m1 = - H_Z_i2   # Reflect
                    Hr_m1 =   H_r_i2
                elseif BC_Z_bot === :Outflow 
                    # Ghost cell j = 0:
                    J_0   =   J_i1
                    HZ_0  =   min( H_Z_i1, 0.0 )  
                    Hr_0  =   H_r_i1 
                    # Ghost cell j = -1:
                    J_m1  =   J_0
                    HZ_m1 =   HZ_0 
                    Hr_m1 =   Hr_0 
                end

                # Reconstruct variables at interface Z_{1/2} (i.e. Z[1]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R = ReconstructInterface( J_0,    J_i1,   J_i2,   J_m1,
                                                   Hr_0,   H_r_i1, H_r_i2, Hr_m1,
                                                   HZ_0,   H_Z_i1, H_Z_i2, HZ_m1,

                                                   ΔZ_L_InnerBoundary, ΔZ_R_InnerBoundary,
                                                   inv_dZ_j_jm1_InnerBoundary,
                                                   inv_dZ_jp1_j_InnerBoundary,
                                                   inv_dZ_jp2_jp1_InnerBoundary,

                                                   α_F_i1, α_F_i1 # Assume same opacity in ghost cell j = 0 as in j = 1
                                                 )

                # Compute interface D-values:
                _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # Compute K = D × J at the interface, with the right BC for KrZ_L:
                KZZ_L = @fastmath D_ZZ_L * J_L
                KZZ_R = @fastmath D_ZZ_R * J_R
                KrZ_L = @fastmath D_rZ_L * J_L
                KrZ_R = @fastmath D_rZ_R * J_R

                # Fluxes at interface j = 1/2:
                F_Z_i1, F_Z_Hr_i1, F_Z_HZ_i1 = GetFluxM1( FluxSolver, Val(:Z), 
                                                          HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                          J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                          τ_L, τ_R, J_0, J_i1, f_L, f_R,
                                                          nr_L, nr_R, nZ_L, nZ_R  )

                # Add advection fluxes:
                F_Z[k,i,1]    = @fastmath F_Z_i1     +  GetAdvectionFlux( u_contact_over_c_i1, J_L, J_R )
                F_Z_Hr[k,i,1] = @fastmath F_Z_Hr_i1  +  GetAdvectionFlux( u_contact_over_c_i1, Hr_L, Hr_R )
                F_Z_HZ[k,i,1] = @fastmath F_Z_HZ_i1  +  GetAdvectionFlux( u_contact_over_c_i1, HZ_L, HZ_R )

                ##############################################################################
                ####################         INTERFACE j = 3/2         #######################
                ##############################################################################

                # Reconstruct variables at interface Z_{3/2} (i.e. Z[2]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R = ReconstructInterface( J_i1,   J_i2,   J[k,i,3],   J_0,
                                                   H_r_i1, H_r_i2, H_r[k,i,3], Hr_0,
                                                   H_Z_i1, H_Z_i2, H_Z[k,i,3], HZ_0,

                                                   ΔZ_L_1, ΔZ_R_1,
                                                   inv_dZ_j_jm1__1,
                                                   inv_dZ_jp1_j__1,
                                                   inv_dZ_jp2_jp1__1,

                                                   α_F_i1, α_F[k,i,2] )

                # Compute interface D-values:
                _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # Compute K = D × J at the interface j = 3/2:
                KZZ_L = @fastmath D_ZZ_L * J_L
                KZZ_R = @fastmath D_ZZ_R * J_R
                KrZ_L = @fastmath D_rZ_L * J_L
                KrZ_R = @fastmath D_rZ_R * J_R

                # Fluxes at interface j = 3/2:
                F_Z_i2, F_Z_Hr_i2, F_Z_HZ_i2 = GetFluxM1( FluxSolver, Val(:Z), 
                                                          HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                          J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                          τ_L, τ_R, J_i1, J_i2, f_L, f_R,
                                                          nr_L, nr_R, nZ_L, nZ_R )
            
                # Add advection fluxes:
                F_Z[k,i,2]    = @fastmath F_Z_i2     +  GetAdvectionFlux( u_contact_over_c_i2, J_L, J_R )
                F_Z_Hr[k,i,2] = @fastmath F_Z_Hr_i2  +  GetAdvectionFlux( u_contact_over_c_i2, Hr_L, Hr_R )
                F_Z_HZ[k,i,2] = @fastmath F_Z_HZ_i2  +  GetAdvectionFlux( u_contact_over_c_i2, HZ_L, HZ_R )

            end # End of k-loop 
        end # End of inbounds 
    end # End of i-loop 

    @threads for j in dN_Zint
        @inbounds for i in dN_r
            jp1 = j + 1
            jp2 = j + 2
            jm1 = j - 1

            # Commonly used intervals below:
            ΔZ_L_j            = ΔZ_L[j]
            ΔZ_R_j            = ΔZ_R[j]
            inv_dZ_j_jm1__j   = inv_dZ_j_jm1[j]
            inv_dZ_jp1_j__j   = inv_dZ_jp1_j[j]
            inv_dZ_jp2_jp1__j = inv_dZ_jp2_jp1[j]

            u_contact_over_c_jp1 = @fastmath u_Z_star[i,jp1] * inv_c

            @simd for k in dN_ν

                # Values of J used twice below:
                J_ij    = J[k,i,j]
                J_i_jp1 = J[k,i,jp1]

                # Reconstruct at Z_{j+1/2}
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R =
                ReconstructInterface( J_ij,       J_i_jp1,      J[k,i,jp2],   J[k,i,jm1],
                                      H_r[k,i,j], H_r[k,i,jp1], H_r[k,i,jp2], H_r[k,i,jm1],
                                      H_Z[k,i,j], H_Z[k,i,jp1], H_Z[k,i,jp2], H_Z[k,i,jm1],

                                      ΔZ_L_j, ΔZ_R_j,
                                      inv_dZ_j_jm1__j,
                                      inv_dZ_jp1_j__j,
                                      inv_dZ_jp2_jp1__j,

                                      α_F[k,i,j], α_F[k,i,jp1] )

                # Interface D and K
                _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                KZZ_L = @fastmath D_ZZ_L * J_L;   KZZ_R = @fastmath D_ZZ_R * J_R
                KrZ_L = @fastmath D_rZ_L * J_L;   KrZ_R = @fastmath D_rZ_R * J_R

                # Write interior Z-fluxes:
                F_Z_jp1, F_Z_Hr_jp1, F_Z_HZ_jp1 = GetFluxM1( FluxSolver, Val(:Z), 
                                                             HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                             J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                             τ_L, τ_R, J_ij, J_i_jp1, f_L, f_R,
                                                             nr_L, nr_R, nZ_L, nZ_R )

                # Add advection fluxes:
                F_Z[k,i,jp1]    = @fastmath F_Z_jp1     +  GetAdvectionFlux( u_contact_over_c_jp1, J_L, J_R )
                F_Z_Hr[k,i,jp1] = @fastmath F_Z_Hr_jp1  +  GetAdvectionFlux( u_contact_over_c_jp1, Hr_L, Hr_R )
                F_Z_HZ[k,i,jp1] = @fastmath F_Z_HZ_jp1  +  GetAdvectionFlux( u_contact_over_c_jp1, HZ_L, HZ_R )

            end # End of k-loop 
        end # End of i-loop 
    end # End of j-loop 


    # Intervals used many times below (for each i and bin):
    ΔZ_L_Nm1            = @inbounds ΔZ_L[N_Z-1] 
    ΔZ_R_Nm1            = @inbounds ΔZ_R[N_Z-1]
    inv_dZ_j_jm1__Nm1   = @inbounds inv_dZ_j_jm1[N_Z-1]
    inv_dZ_jp1_j__Nm1   = @inbounds inv_dZ_jp1_j[N_Z-1]
    inv_dZ_jp2_jp1__Nm1 = @inbounds inv_dZ_jp2_jp1[N_Z-1]

    @threads for i in dN_r 
        @inbounds begin 

            u_contact_over_c_N   = @fastmath u_Z_star[i,N_Z] * inv_c
            u_contact_over_c_Np1 = @fastmath u_Z_star[i,N_Z+1] * inv_c

            @simd for k in dN_ν

                # Load commonly used values of J and H, and α_F:
                J_N     = J[k,i,N_Z]
                J_Nm1   = J[k,i,N_Z-1]
                H_r_N   = H_r[k,i,N_Z]
                H_r_Nm1 = H_r[k,i,N_Z-1]
                H_Z_N   = H_Z[k,i,N_Z]
                H_Z_Nm1 = H_Z[k,i,N_Z-1]
                α_F_N   = α_F[k,i,N_Z]

                # Boundary conditions at interface j = N_Z + 1/2:
                if BC_Z_top === :Reflect
                    # Ghost cell j = N_Z + 1:
                    J_Np1   =   J_N
                    HZ_Np1  = - H_Z_N    # Reflect
                    Hr_Np1  =   H_r_N 
                    # Ghost cell j = N_Z + 2 (mirror of cell j = N_Z - 1):
                    J_Np2   =   J_Nm1
                    HZ_Np2  = - H_Z_Nm1  # Reflect
                    Hr_Np2  =   H_r_Nm1
                elseif BC_Z_top === :Outflow
                    # Ghost cell j = N_Z + 1:
                    J_Np1   =   J_N
                    HZ_Np1  =   max( H_Z_N, 0.0 )    
                    Hr_Np1  =   H_r_N 
                    # Ghost cell j = N_Z + 2 (copy of cell j = N_Z + 1):
                    J_Np2   =   J_Np1
                    HZ_Np2  =   HZ_Np1
                    Hr_Np2  =   Hr_Np1 
                end

                ##############################################################################
                ##############         INTERFACE j = N_Z - 1/2         #######################
                ##############################################################################

                # Reconstruct variables at interface Z_{N_Z-1/2} (i.e. Z[N_Z]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R = ReconstructInterface( J_Nm1,   J_N,   J_Np1,  J[k,i,N_Z-2],
                                                   H_r_Nm1, H_r_N, Hr_Np1, H_r[k,i,N_Z-2],
                                                   H_Z_Nm1, H_Z_N, HZ_Np1, H_Z[k,i,N_Z-2],

                                                   ΔZ_L_Nm1, ΔZ_R_Nm1,
                                                   inv_dZ_j_jm1__Nm1,
                                                   inv_dZ_jp1_j__Nm1,
                                                   inv_dZ_jp2_jp1__Nm1,

                                                   α_F[k,i,N_Z-1], α_F_N )

                # Compute interface D-values:
                _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # Compute K = D × J at the interface j = N_Z - 1/2:
                KZZ_L = @fastmath D_ZZ_L * J_L
                KZZ_R = @fastmath D_ZZ_R * J_R
                KrZ_L = @fastmath D_rZ_L * J_L
                KrZ_R = @fastmath D_rZ_R * J_R

                # Fluxes at interface j = N_Z - 1/2:
                F_Z_N, F_Z_Hr_N, F_Z_HZ_N = GetFluxM1( FluxSolver, Val(:Z), 
                                                       HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                       J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                       τ_L, τ_R, J_Nm1, J_N, f_L, f_R,
                                                       nr_L, nr_R, nZ_L, nZ_R )

                # Add advection fluxes:
                F_Z[k,i,N_Z]    = @fastmath F_Z_N     +  GetAdvectionFlux( u_contact_over_c_N, J_L, J_R )
                F_Z_Hr[k,i,N_Z] = @fastmath F_Z_Hr_N  +  GetAdvectionFlux( u_contact_over_c_N, Hr_L, Hr_R )
                F_Z_HZ[k,i,N_Z] = @fastmath F_Z_HZ_N  +  GetAdvectionFlux( u_contact_over_c_N, HZ_L, HZ_R )

                ##############################################################################
                ###############        INTERFACE j = N_Z + 1/2         #######################
                ##############################################################################

                # Reconstruct variables at interface Z_{N_Z+1/2} (i.e. Z[N_Z+1]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R = ReconstructInterface( J_N,   J_Np1,   J_Np2,   J_Nm1,
                                                   H_r_N, Hr_Np1,  Hr_Np2,  H_r_Nm1,
                                                   H_Z_N, HZ_Np1,  HZ_Np2,  H_Z_Nm1,

                                                   ΔZ_L_OuterBoundary, ΔZ_R_OuterBoundary,
                                                   inv_dZ_j_jm1_OuterBoundary,
                                                   inv_dZ_jp1_j_OuterBoundary,
                                                   inv_dZ_jp2_jp1_OuterBoundary,

                                                   α_F_N, α_F_N  # Assume opacity in ghost cell j = N_Z+1 is same as in j = N_Z
                                                 )

                # Compute interface D-values:
                _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # Compute K = D × J at the interface j = N_Z + 1/2:
                KZZ_L = @fastmath D_ZZ_L * J_L
                KZZ_R = @fastmath D_ZZ_R * J_R
                KrZ_L = @fastmath D_rZ_L * J_L
                KrZ_R = @fastmath D_rZ_R * J_R

                # Fluxes at interface j = N_Z + 1/2:
                F_Z_Np1, F_Z_Hr_Np1, F_Z_HZ_Np1 = GetFluxM1( FluxSolver, Val(:Z), 
                                                             HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                             J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                             τ_L, τ_R, J_N, J_Np1, f_L, f_R,
                                                             nr_L, nr_R, nZ_L, nZ_R )
            
                # Add advection fluxes:
                F_Z[k,i,N_Z+1]    = @fastmath F_Z_Np1     +  GetAdvectionFlux( u_contact_over_c_Np1, J_L, J_R )
                F_Z_Hr[k,i,N_Z+1] = @fastmath F_Z_Hr_Np1  +  GetAdvectionFlux( u_contact_over_c_Np1, Hr_L, Hr_R )
                F_Z_HZ[k,i,N_Z+1] = @fastmath F_Z_HZ_Np1  +  GetAdvectionFlux( u_contact_over_c_Np1, HZ_L, HZ_R )


            end # End of k-loop 
        end # End of inbounds 
    end # End of i-loop 

    ###############################################################
    #######                                                 #######
    #######             U P D A T E  J  A N D  H            #######
    #######                                                 #######
    ###############################################################

    # With all fluxes computes, we can now update J and the components 
    # of H over the time-step Δt:

    ####  UPDATE J AND H  ##############################################################

    @threads for j in dN_Z 
        @inbounds for i in dN_r 

            # Area & volume arrays (accessed many times):
            A_r_ij     = A_r[i,j] 
            A_r_ip1_j  = A_r[i+1,j]
            A_Z_ij     = A_Z[i,j] 
            A_Z_i_jp1  = A_Z[i,j+1]
            inv_ΔV_ij  = inv_ΔV[i,j]

            # Velocity gradient terms (used multiple times):
            ∂u_r_∂r_ij    = ∂u_r_∂r[i,j]
            ∂u_r_∂Z_ij    = ∂u_r_∂Z[i,j]
            ∂u_Z_∂r_ij    = ∂u_Z_∂r[i,j]
            ∂u_Z_∂Z_ij    = ∂u_Z_∂Z[i,j]
            u_r_over_r_ij = u_r_over_r[i,j]

            @simd for k in dN_ν
                # Store old J and H values, needed for the radiation pressure work term, 
                # and aberration term for H:
                J_stored   = J[k,i,j]
                H_r_stored = H_r[k,i,j]
                H_Z_stored = H_Z[k,i,j]

                # Compute values of Eddington tensor (at the stored/old values for consistency 
                # with an explicit update) for use below:
                Drr, DZZ, DrZ = Compute_D_Edd( H_r_stored, H_Z_stored, J_stored )

                # Update J and H from the flux terms:
                J_new      = max( UpdateConservativeQuantity( J_stored, c̃Δt, inv_ΔV_ij, 
                                                              A_r_ij, A_r_ip1_j,
                                                              A_Z_ij, A_Z_i_jp1, 
                                                              F_r[k,i,j], F_r[k,i+1,j],
                                                              F_Z[k,i,j], F_Z[k,i,j+1] ),
                                  Jeps )

                H_Z_new    = UpdateConservativeQuantity( H_Z_stored, c̃Δt, inv_ΔV_ij,
                                                         A_r_ij, A_r_ip1_j,
                                                         A_Z_ij, A_Z_i_jp1,
                                                         F_r_HZ[k,i,j], F_r_HZ[k,i+1,j],
                                                         F_Z_HZ[k,i,j], F_Z_HZ[k,i,j+1] )

                H_r_new    = UpdateConservativeQuantity( H_r_stored, c̃Δt, inv_ΔV_ij,
                                                         A_r_ij, A_r_ip1_j,
                                                         A_Z_ij, A_Z_i_jp1,
                                                         F_r_Hr[k,i,j], F_r_Hr[k,i+1,j],
                                                         F_Z_Hr[k,i,j], F_Z_Hr[k,i,j+1] )

                # Get D:∇u, needed below, and also add contribution from geometric source 
                # term for cylindrical coordinates:
                D_colon_∇u = D_colon_∇u_Cylindrical( Drr, DrZ, DZZ, 
                                                     ∂u_r_∂r_ij, ∂u_r_∂Z_ij, ∂u_Z_∂r_ij, ∂u_Z_∂Z_ij,
                                                     u_r_over_r_ij )

                # In cylindrical coordinates, add contribution from geometric source term,
                # (1/c̃) ∂H_r/∂t = K_ϕϕ/r ≈ 0.5 [(K_ϕϕ)_{i+1/2} + (K_ϕϕ)_{i-1/2}] ⟨1/r⟩ using
                # ⟨1/r⟩ ≡ ∫dV (1/r) / ΔV = 2π * Δr * ΔZ / ΔV = 1 / [( r[i+1] + r[i] ) / 2]:
                @fastmath K_ϕϕ_av    = 0.5 * ( K_ϕϕ_imhalf[k,i,j]  +  K_ϕϕ_iphalf[k,i,j] )
                @fastmath H_r_new   += c̃Δt * K_ϕϕ_av * inv_r_cell[i]

                # Add aberration update of H_r and H_Z:
                ΔH_r, ΔH_Z = ΔH_Aberration( H_r_stored, H_Z_stored, c̃Δt_over_c,
                                            ∂u_r_∂r_ij, ∂u_r_∂Z_ij, ∂u_Z_∂r_ij, ∂u_Z_∂Z_ij )

                H_r_new    = @fastmath H_r_new  +  ΔH_r
                H_Z_new    = @fastmath H_Z_new  +  ΔH_Z

                # Finally, also add the contribution from (1/c̃) ∂J/∂t = - (D:∇u/c) J: 
                J_new      = max( J_new  -  c̃Δt_over_c * D_colon_∇u * J_stored,  Jeps ) 

                # Ensure causality: 
                H_r_new, H_Z_new  = EnforceCausalReducedFlux( J_new, H_r_new, H_Z_new )

                # Save updated values:
                H_r[k,i,j] = H_r_new
                H_Z[k,i,j] = H_Z_new
                J[k,i,j]   = J_new 

            end # End of k-loop 
        end # End of i-loop 
    end # End of j-loop 

    # Update J_old if the option is turned on:
    if UpdateJold
        copyto!( J_old, J )
    end

    return nothing    
end





@inline function AdvectJandH_Cartesian!( c̃Δt::Float64,                   # c̃ × Δt 
                                         α_F::AbstractArray{Float64, 3}, # Flux opacities              (N_ν × N_r x N_Z)
   
                                           J::AbstractArray{Float64, 3}, # J (latest/updated values)   (N_ν × N_r x N_Z)
                                       J_old::AbstractArray{Float64, 3}, # J (old values)              (N_ν × N_r x N_Z)

                                         H_r::AbstractArray{Float64, 3}, # r-direction component of H  (N_ν × N_r x N_Z)
                                         H_Z::AbstractArray{Float64, 3}, # Z-direction component of H  (N_ν × N_r x N_Z)

                                         F_r::AbstractArray{Float64, 3}, # r-direction flux of J               (N_ν × N_r+1 x N_Z)
                                         F_Z::AbstractArray{Float64, 3}, # Z-direction flux of J               (N_ν × N_r   x N_Z+1)

                                      F_r_Hr::AbstractArray{Float64, 3}, # r-direction flux of H_r             (N_ν × N_r+1 x N_Z)
                                      F_Z_Hr::AbstractArray{Float64, 3}, # Z-direction flux of H_r             (N_ν × N_r   x N_Z+1)
                                      F_r_HZ::AbstractArray{Float64, 3}, # r-direction flux of H_Z             (N_ν × N_r+1 x N_Z)
                                      F_Z_HZ::AbstractArray{Float64, 3}, # Z-direction flux of H_Z             (N_ν × N_r   x N_Z+1)

                                      inv_ΔV::AbstractArray{Float64, 2}, # Inverse cell volumes 1 / ΔV         (N_r   x N_Z)
                                         A_r::AbstractArray{Float64, 2}, # Cell interface areas in r-direction (N_r+1 x N_Z)
                                         A_Z::AbstractArray{Float64, 2}, # Cell interface areas in Z-direction (N_r   x N_Z+1)

                                # Gas velocity field:
                                    u_r_star::AbstractArray{Float64,2},
                                    u_Z_star::AbstractArray{Float64,2},
                                     ∂u_r_∂r::AbstractArray{Float64,2},
                                     ∂u_r_∂Z::AbstractArray{Float64,2}, 
                                     ∂u_Z_∂r::AbstractArray{Float64,2}, 
                                     ∂u_Z_∂Z::AbstractArray{Float64,2},

                                # Needed intervals:
                            Δr_L::AbstractVector{Float64}, Δr_R::AbstractVector{Float64},
                                inv_dr_i_im1::AbstractVector{Float64},
                                inv_dr_ip1_i::AbstractVector{Float64},
                              inv_dr_ip2_ip1::AbstractVector{Float64},

                            Δr_L_InnerBoundary::Float64, Δr_R_InnerBoundary::Float64,
                  inv_dr_i_im1_InnerBoundary::Float64,
                  inv_dr_ip1_i_InnerBoundary::Float64,
                inv_dr_ip2_ip1_InnerBoundary::Float64,
 
                          Δr_L_OuterBoundary::Float64, Δr_R_OuterBoundary::Float64,
                  inv_dr_i_im1_OuterBoundary::Float64,
                  inv_dr_ip1_i_OuterBoundary::Float64,
                inv_dr_ip2_ip1_OuterBoundary::Float64,

                            ΔZ_L::AbstractVector{Float64}, ΔZ_R::AbstractVector{Float64},
                                inv_dZ_j_jm1::AbstractVector{Float64},
                                inv_dZ_jp1_j::AbstractVector{Float64},
                              inv_dZ_jp2_jp1::AbstractVector{Float64},

                          ΔZ_L_InnerBoundary::Float64, ΔZ_R_InnerBoundary::Float64,
                  inv_dZ_j_jm1_InnerBoundary::Float64,
                  inv_dZ_jp1_j_InnerBoundary::Float64,
                inv_dZ_jp2_jp1_InnerBoundary::Float64,
 
                            ΔZ_L_OuterBoundary::Float64, ΔZ_R_OuterBoundary::Float64,
                  inv_dZ_j_jm1_OuterBoundary::Float64,
                  inv_dZ_jp1_j_OuterBoundary::Float64,
                inv_dZ_jp2_jp1_OuterBoundary::Float64,
            
                                  # Other stuff:
                                  UpdateJold::Bool,         # Whether to update J_old = J at the end (should be false for stellar RT)

                                   BC_r_left::Symbol,       # Boundary condition at i = 1/2 
                                  BC_r_right::Symbol,       # Boundary condition at i = N_r + 1/2
                                    BC_Z_bot::Symbol,       # Boundary condition at j = 1/2 
                                    BC_Z_top::Symbol,       # Boundary condition at j = N_Z + 1/2 
                                        kmin::Int    = 1,
                                        kmax::Int    = size( J, 1 ),
                                  FluxSolver::Val    = Val(:HLL)
                                   ) 

    # Grid & intervals:
    N_ν, N_r, N_Z   = size( J )
    dN_r            = 1:N_r
    dN_Z            = 1:N_Z
    dN_ν            = max( kmin, 1):min( kmax, N_ν )  # Range of frequency bins to advect 
    dN_rint         = 2:(N_r-2)
    dN_Zint         = 2:(N_Z-2)

    # Convenient constant:
    c̃Δt_over_c      = @fastmath c̃Δt * inv_c

    if DEBUG_M1Advection
        @assert size( u_r_star, 1 ) == N_r + 1 "Contact velocity in r-direction/X-direction must have size $(N_r+1)."
        @assert size( u_Z_star, 2 ) == N_Z + 1 "Contact velocity in Z-direction/Y-direction must have size $(N_Z+1)."
    end

    ###############################################################
    #######                                                 #######
    #######           C O M P U T E  F L U X E S            #######
    #######                                                 #######
    ###############################################################

    # We start by reconstructing & computing the interface fluxes, 
    # starting with the r-direction (order does not matter here).

    # Intervals used many times below (for each j and bin):
    Δr_L_1              = @inbounds Δr_L[1] 
    Δr_R_1              = @inbounds Δr_R[1]
    inv_dr_i_im1__1     = @inbounds inv_dr_i_im1[1]
    inv_dr_ip1_i__1     = @inbounds inv_dr_ip1_i[1]
    inv_dr_ip2_ip1__1   = @inbounds inv_dr_ip2_ip1[1]

    Δr_L_Nm1            = @inbounds Δr_L[N_r-1]
    Δr_R_Nm1            = @inbounds Δr_R[N_r-1]
    inv_dr_i_im1__Nm1   = @inbounds inv_dr_i_im1[N_r-1]
    inv_dr_ip1_i__Nm1   = @inbounds inv_dr_ip1_i[N_r-1]
    inv_dr_ip2_ip1__Nm1 = @inbounds inv_dr_ip2_ip1[N_r-1]
    
    # r-direction fluxes:
    @threads for j in dN_Z
        @inbounds begin

            u_contact_over_c_1j = @fastmath u_r_star[1,j] * inv_c
            u_contact_over_c_2j = @fastmath u_r_star[2,j] * inv_c

            @simd for k in dN_ν

                J_1j   = J[k,1,j]
                J_2j   = J[k,2,j]
                H_r_1j = H_r[k,1,j]
                H_r_2j = H_r[k,2,j]
                H_Z_1j = H_Z[k,1,j]
                H_Z_2j = H_Z[k,2,j]

                α_F_1j = α_F[k,1,j]

                ##############################################################################
                ##############          INTERFACE i = 1/2          ###########################
                ##############################################################################

                if BC_r_left === :Reflect 
                    # Ghost cell i = 0:
                    J_0    = J_1j 
                    Hr_0   = - H_r_1j   # Reflect radial flux
                    HZ_0   = H_Z_1j
                    # Ghost cell i = -1:
                    J_m1   = J_2j 
                    Hr_m1  = - H_r_2j   # Reflect radial flux
                    HZ_m1  = H_Z_2j
                elseif BC_r_left === :Outflow  
                    # Ghost cell i = 0:
                    J_0    = J_1j 
                    Hr_0   = min( H_r_1j, 0.0 )          
                    HZ_0   = H_Z_1j
                    # Ghost cell i = -1 (no gradients, so equal to cells i = 0 and 1):
                    J_m1   = J_0 
                    Hr_m1  = Hr_0   
                    HZ_m1  = HZ_0
                end

                # Reconstruct variables at interface r_{1/2} (i.e. r[1]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R  = ReconstructInterface( J_0,     J_1j,   J_2j,   J_m1,
                                                    Hr_0,    H_r_1j, H_r_2j, Hr_m1,
                                                    HZ_0,    H_Z_1j, H_Z_2j, HZ_m1,

                                                    Δr_L_InnerBoundary, Δr_R_InnerBoundary,
                                                    inv_dr_i_im1_InnerBoundary,
                                                    inv_dr_ip1_i_InnerBoundary,
                                                    inv_dr_ip2_ip1_InnerBoundary,
                    
                                                    α_F_1j, α_F_1j  # Assume ghost cell 0 has same opacity as cell i = 1
                                                   )

                # Compute interface D-values:
                D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # Compute K = D × J at the interface:
                Krr_L = @fastmath D_rr_L * J_L
                Krr_R = @fastmath D_rr_R * J_R
                KrZ_L = @fastmath D_rZ_L * J_L
                KrZ_R = @fastmath D_rZ_R * J_R

                # Fluxes at i = 1/2:
                F_r_1j, F_r_Hr_1j, F_r_HZ_1j = GetFluxM1( FluxSolver, Val(:r), 
                                                          Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                          J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R,   
                                                          τ_L, τ_R, J_0, J_1j, f_L, f_R,
                                                          nr_L, nr_R, nZ_L, nZ_R )

                # Add advection fluxes:
                F_r[k,1,j]           = @fastmath F_r_1j    + GetAdvectionFlux( u_contact_over_c_1j, J_L, J_R )
                F_r_Hr[k,1,j]        = @fastmath F_r_Hr_1j + GetAdvectionFlux( u_contact_over_c_1j, Hr_L, Hr_R )
                F_r_HZ[k,1,j]        = @fastmath F_r_HZ_1j + GetAdvectionFlux( u_contact_over_c_1j, HZ_L, HZ_R )

                ##############################################################################
                ##############          INTERFACE i = 3/2          ###########################
                ##############################################################################

                # Reconstruct variables at interface r_{3/2} (i.e. r[2]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R = ReconstructInterface( J_1j,   J_2j,   J[k,3,j],   J_0,
                                                   H_r_1j, H_r_2j, H_r[k,3,j], Hr_0,
                                                   H_Z_1j, H_Z_2j, H_Z[k,3,j], HZ_0,

                                                   Δr_L_1, Δr_R_1,
                                                   inv_dr_i_im1__1,
                                                   inv_dr_ip1_i__1,
                                                   inv_dr_ip2_ip1__1,

                                                   α_F_1j, α_F[k,2,j] )

                # Compute interface D-values:
                D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # Compute K = D × J at the interface:
                Krr_L = @fastmath D_rr_L * J_L 
                Krr_R = @fastmath D_rr_R * J_R 
                KrZ_L = @fastmath D_rZ_L * J_L
                KrZ_R = @fastmath D_rZ_R * J_R

                # Fluxes at i = 3/2:
                F_r_2j, F_r_Hr_2j, F_r_HZ_2j = GetFluxM1( FluxSolver, Val(:r), Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                          J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                          τ_L, τ_R, J_1j, J_2j, f_L, f_R,
                                                          nr_L, nr_R, nZ_L, nZ_R )

                # Add advection fluxes:
                F_r[k,2,j]          = @fastmath F_r_2j     +  GetAdvectionFlux( u_contact_over_c_2j, J_L, J_R )
                F_r_Hr[k,2,j]       = @fastmath F_r_Hr_2j  +  GetAdvectionFlux( u_contact_over_c_2j, Hr_L, Hr_R )
                F_r_HZ[k,2,j]       = @fastmath F_r_HZ_2j  +  GetAdvectionFlux( u_contact_over_c_2j, HZ_L, HZ_R )

            end # End of k-loop 

            ##############################################################################
            ##############        INTERIOR INTERFACES 2 ≤ i ≤ N_r-2          #############
            ##############################################################################

            for i in dN_rint
                ip1 = i + 1
                ip2 = i + 2
                im1 = i - 1

                Δr_L_i            = Δr_L[i]
                Δr_R_i            = Δr_R[i]
                inv_dr_i_im1__i   = inv_dr_i_im1[i]
                inv_dr_ip1_i__i   = inv_dr_ip1_i[i]
                inv_dr_ip2_ip1__i = inv_dr_ip2_ip1[i]

                # u_r_star[i+1] / c is used for every bin:
                u_contact_over_c_ip1 = @fastmath u_r_star[ip1,j] * inv_c

                @simd for k in dN_ν

                    # Used twice below:
                    J_i   = J[k,i,j]
                    J_ip1 = J[k,ip1,j]

                    # Reconstruct variables at interface r_{i+1/2} (i.e. r[i+1]):
                    J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                    τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                    nZ_L, nZ_R = ReconstructInterface( J_i,        J_ip1,        J[k,ip2,j],   J[k,im1,j],
                                                       H_r[k,i,j], H_r[k,ip1,j], H_r[k,ip2,j], H_r[k,im1,j],
                                                       H_Z[k,i,j], H_Z[k,ip1,j], H_Z[k,ip2,j], H_Z[k,im1,j],

                                                       Δr_L_i, Δr_R_i,
                                                       inv_dr_i_im1__i,
                                                       inv_dr_ip1_i__i,
                                                       inv_dr_ip2_ip1__i,

                                                       α_F[k,i,j], α_F[k,ip1,j] )

                    # Compute interface D-values:
                    D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                    D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                    # Compute K = D × J at the interface:
                    Krr_L = @fastmath D_rr_L * J_L 
                    Krr_R = @fastmath D_rr_R * J_R 
                    KrZ_L = @fastmath D_rZ_L * J_L
                    KrZ_R = @fastmath D_rZ_R * J_R

                    # Radial flux of J, H_r, and H_Z:
                    F_r_ip1, F_r_Hr_ip1, F_r_HZ_ip1 = GetFluxM1( FluxSolver, Val(:r), Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                                 J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                                 τ_L, τ_R, J_i, J_ip1, f_L, f_R,
                                                                 nr_L, nr_R, nZ_L, nZ_R )

                    # Add advection fluxes:
                    F_r[k,ip1,j]         = @fastmath F_r_ip1     +  GetAdvectionFlux( u_contact_over_c_ip1, J_L, J_R )
                    F_r_Hr[k,ip1,j]      = @fastmath F_r_Hr_ip1  +  GetAdvectionFlux( u_contact_over_c_ip1, Hr_L, Hr_R )
                    F_r_HZ[k,ip1,j]      = @fastmath F_r_HZ_ip1  +  GetAdvectionFlux( u_contact_over_c_ip1, HZ_L, HZ_R )

                end # End of k-loop 
            end # End of interior i ∈ 2:(N_r-2) loop


            # u_r_star[N_r] / c and u_r_star[N_r+1] / c is used for every bin:
            u_contact_over_c_Nj   = @fastmath u_r_star[N_r,j]   * inv_c
            u_contact_over_c_Np1j = @fastmath u_r_star[N_r+1,j] * inv_c

            @simd for k in dN_ν

                J_N     = J[k, N_r, j]
                J_Nm1   = J[k, N_r - 1, j]
                H_r_N   = H_r[k, N_r, j]
                H_r_Nm1 = H_r[k, N_r - 1, j]
                H_Z_N   = H_Z[k, N_r, j]
                H_Z_Nm1 = H_Z[k, N_r - 1, j]

                α_F_N   = α_F[k,N_r,j]

                # Boundary conditions at i = N_r + 1:
                if BC_r_right === :Reflect
                    # Ghost cell i = N_r + 1:
                    J_Np1   =   J_N
                    Hr_Np1  = - H_r_N    # Reflect 
                    HZ_Np1  =   H_Z_N 
                    # Ghost cell i = N_R + 2:
                    J_Np2   =   J_Nm1
                    Hr_Np2  = - H_r_Nm1  # Reflect 
                    HZ_Np2  =   H_Z_Nm1  
                elseif BC_r_right === :Outflow
                    # Ghost cell i = N_r + 1:
                    J_Np1   =  J_N
                    Hr_Np1  =  max( H_r_N, 0.0 )
                    HZ_Np1  =  H_Z_N
                    # Ghost cell i = N_R + 2:
                    J_Np2   =  J_Np1
                    Hr_Np2  =  Hr_Np1
                    HZ_Np2  =  HZ_Np1
                end

                ##############################################################################
                ##############         INTERFACE i = N_r - 1/2         #######################
                ##############################################################################

                # Reconstruct variables at interface r_{N_r-1/2} (i.e. r[N_r]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R = ReconstructInterface( J_Nm1,   J_N,   J_Np1,    J[k,N_r-2,j],
                                                   H_r_Nm1, H_r_N, Hr_Np1,   H_r[k,N_r-2,j],
                                                   H_Z_Nm1, H_Z_N, HZ_Np1,   H_Z[k,N_r-2,j],

                                                   Δr_L_Nm1, Δr_R_Nm1,
                                                   inv_dr_i_im1__Nm1,
                                                   inv_dr_ip1_i__Nm1,
                                                   inv_dr_ip2_ip1__Nm1,

                                                   α_F[k,N_r-1,j], α_F_N )

                # Compute interface D-values:
                D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # K = D × J at the interface i = N_r - 1/2:
                Krr_L = @fastmath D_rr_L * J_L
                Krr_R = @fastmath D_rr_R * J_R
                KrZ_L = @fastmath D_rZ_L * J_L
                KrZ_R = @fastmath D_rZ_R * J_R

                # Fluxes at interface i = N_r - 1/2:
                F_r_Nj, F_r_Hr_Nj, F_r_HZ_Nj = GetFluxM1( FluxSolver, Val(:r), 
                                                          Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                          J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                          τ_L, τ_R, J_Nm1, J_N, f_L, f_R,
                                                          nr_L, nr_R, nZ_L, nZ_R )

                # Add advection fluxes:
                F_r[k,N_r,j]      = @fastmath F_r_Nj     +  GetAdvectionFlux( u_contact_over_c_Nj, J_L, J_R )
                F_r_Hr[k,N_r,j]   = @fastmath F_r_Hr_Nj  +  GetAdvectionFlux( u_contact_over_c_Nj, Hr_L, Hr_R )
                F_r_HZ[k,N_r,j]   = @fastmath F_r_HZ_Nj  +  GetAdvectionFlux( u_contact_over_c_Nj, HZ_L, HZ_R )

                ##############################################################################
                ##############         INTERFACE i = N_r + 1/2         #######################
                ##############################################################################

                # Reconstruct variables at interface r_{N_r+1/2} (i.e. r[N_r+1]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R = ReconstructInterface( J_N,   J_Np1,    J_Np2,    J_Nm1,
                                                   H_r_N, Hr_Np1,   Hr_Np2,   H_r_Nm1,
                                                   H_Z_N, HZ_Np1,   HZ_Np2,   H_Z_Nm1,

                                                   Δr_L_OuterBoundary, Δr_R_OuterBoundary,
                                                   inv_dr_i_im1_OuterBoundary,
                                                   inv_dr_ip1_i_OuterBoundary,
                                                   inv_dr_ip2_ip1_OuterBoundary,

                                                   α_F_N, α_F_N  # Assume opacity in ghost cell N_r+1 is same as for cell N_r
                                                  )

                # Compute interface D-values:
                D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # K = D × J at the interface i = N_r + 1/2:
                Krr_L = @fastmath D_rr_L * J_L
                Krr_R = @fastmath D_rr_R * J_R
                KrZ_L = @fastmath D_rZ_L * J_L
                KrZ_R = @fastmath D_rZ_R * J_R

                # Fluxes at interface i = N_r + 1/2:
                F_r_Np1j, F_r_Hr_Np1j, F_r_HZ_Np1j = GetFluxM1( FluxSolver, Val(:r), 
                                                                Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                                τ_L, τ_R, J_N, J_Np1, f_L, f_R,
                                                                nr_L, nr_R, nZ_L, nZ_R )

                # Add advection fluxes:
                F_r[k,N_r+1,j]       = @fastmath F_r_Np1j     +  GetAdvectionFlux( u_contact_over_c_Np1j, J_L, J_R )
                F_r_Hr[k,N_r+1,j]    = @fastmath F_r_Hr_Np1j  +  GetAdvectionFlux( u_contact_over_c_Np1j, Hr_L, Hr_R )
                F_r_HZ[k,N_r+1,j]    = @fastmath F_r_HZ_Np1j  +  GetAdvectionFlux( u_contact_over_c_Np1j, HZ_L, HZ_R )

            end # End of k-loop 

        end # End of inbounds     
    end # End of j-loop 

    # Intervals used many times below (for each i and bin):
    ΔZ_L_1            = @inbounds ΔZ_L[1]
    ΔZ_R_1            = @inbounds ΔZ_R[1]
    inv_dZ_j_jm1__1   = @inbounds inv_dZ_j_jm1[1]
    inv_dZ_jp1_j__1   = @inbounds inv_dZ_jp1_j[1]
    inv_dZ_jp2_jp1__1 = @inbounds inv_dZ_jp2_jp1[1]

    # Next, Z-direction fluxes:
    @threads for i in dN_r
        @inbounds begin

            u_contact_over_c_i1 = @fastmath u_Z_star[i,1] * inv_c
            u_contact_over_c_i2 = @fastmath u_Z_star[i,2] * inv_c

            @simd for k in dN_ν

                # Load commonly used values of J and H, and α_F:
                J_i1   = J[k,i,1]
                J_i2   = J[k,i,2]
                H_r_i1 = H_r[k,i,1]
                H_r_i2 = H_r[k,i,2]
                H_Z_i1 = H_Z[k,i,1]
                H_Z_i2 = H_Z[k,i,2]

                α_F_i1 = α_F[k,i,1]

                ##############################################################################
                ####################         INTERFACE j = 1/2         #######################
                ##############################################################################

                if BC_Z_bot === :Reflect
                    # Ghost cell j = 0:
                    J_0   =   J_i1
                    HZ_0  = - H_Z_i1   # Reflect
                    Hr_0  =   H_r_i1
                    # Ghost cell j = -1 (mirror of cell j = 2):
                    J_m1  =   J_i2
                    HZ_m1 = - H_Z_i2   # Reflect
                    Hr_m1 =   H_r_i2
                elseif BC_Z_bot === :Outflow 
                    # Ghost cell j = 0:
                    J_0   =   J_i1
                    HZ_0  =   min( H_Z_i1, 0.0 )  
                    Hr_0  =   H_r_i1 
                    # Ghost cell j = -1:
                    J_m1  =   J_0
                    HZ_m1 =   HZ_0 
                    Hr_m1 =   Hr_0 
                end

                # Reconstruct variables at interface Z_{1/2} (i.e. Z[1]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R = ReconstructInterface( J_0,    J_i1,   J_i2,   J_m1,
                                                   Hr_0,   H_r_i1, H_r_i2, Hr_m1,
                                                   HZ_0,   H_Z_i1, H_Z_i2, HZ_m1,

                                                   ΔZ_L_InnerBoundary, ΔZ_R_InnerBoundary,
                                                   inv_dZ_j_jm1_InnerBoundary,
                                                   inv_dZ_jp1_j_InnerBoundary,
                                                   inv_dZ_jp2_jp1_InnerBoundary,

                                                   α_F_i1, α_F_i1 # Assume same opacity in ghost cell j = 0 as in j = 1
                                                 )

                # Compute interface D-values:
                _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # Compute K = D × J at the interface, with the right BC for KrZ_L:
                KZZ_L = @fastmath D_ZZ_L * J_L
                KZZ_R = @fastmath D_ZZ_R * J_R
                KrZ_L = @fastmath D_rZ_L * J_L
                KrZ_R = @fastmath D_rZ_R * J_R

                # Fluxes at interface j = 1/2:
                F_Z_i1, F_Z_Hr_i1, F_Z_HZ_i1 = GetFluxM1( FluxSolver, Val(:Z), 
                                                          HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                          J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                          τ_L, τ_R, J_0, J_i1, f_L, f_R,
                                                          nr_L, nr_R, nZ_L, nZ_R )

                # Add advection fluxes:
                F_Z[k,i,1]    = @fastmath F_Z_i1     +  GetAdvectionFlux( u_contact_over_c_i1, J_L, J_R )
                F_Z_Hr[k,i,1] = @fastmath F_Z_Hr_i1  +  GetAdvectionFlux( u_contact_over_c_i1, Hr_L, Hr_R )
                F_Z_HZ[k,i,1] = @fastmath F_Z_HZ_i1  +  GetAdvectionFlux( u_contact_over_c_i1, HZ_L, HZ_R )

                ##############################################################################
                ####################         INTERFACE j = 3/2         #######################
                ##############################################################################

                # Reconstruct variables at interface Z_{3/2} (i.e. Z[2]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R = ReconstructInterface( J_i1,   J_i2,   J[k,i,3],   J_0,
                                                   H_r_i1, H_r_i2, H_r[k,i,3], Hr_0,
                                                   H_Z_i1, H_Z_i2, H_Z[k,i,3], HZ_0,

                                                   ΔZ_L_1, ΔZ_R_1,
                                                   inv_dZ_j_jm1__1,
                                                   inv_dZ_jp1_j__1,
                                                   inv_dZ_jp2_jp1__1,

                                                   α_F_i1, α_F[k,i,2] )

                # Compute interface D-values:
                _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # Compute K = D × J at the interface j = 3/2:
                KZZ_L = @fastmath D_ZZ_L * J_L
                KZZ_R = @fastmath D_ZZ_R * J_R
                KrZ_L = @fastmath D_rZ_L * J_L
                KrZ_R = @fastmath D_rZ_R * J_R

                # Fluxes at interface j = 3/2:
                F_Z_i2, F_Z_Hr_i2, F_Z_HZ_i2 = GetFluxM1( FluxSolver, Val(:Z), 
                                                          HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                          J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                          τ_L, τ_R, J_i1, J_i2, f_L, f_R,
                                                          nr_L, nr_R, nZ_L, nZ_R )
            
                # Add advection fluxes:
                F_Z[k,i,2]    = @fastmath F_Z_i2     +  GetAdvectionFlux( u_contact_over_c_i2, J_L, J_R )
                F_Z_Hr[k,i,2] = @fastmath F_Z_Hr_i2  +  GetAdvectionFlux( u_contact_over_c_i2, Hr_L, Hr_R )
                F_Z_HZ[k,i,2] = @fastmath F_Z_HZ_i2  +  GetAdvectionFlux( u_contact_over_c_i2, HZ_L, HZ_R )

            end # End of k-loop 
        end # End of inbounds 
    end # End of i-loop 

    @threads for j in dN_Zint
        @inbounds for i in dN_r
            jp1 = j + 1
            jp2 = j + 2
            jm1 = j - 1

            # Commonly used intervals below:
            ΔZ_L_j            = ΔZ_L[j]
            ΔZ_R_j            = ΔZ_R[j]
            inv_dZ_j_jm1__j   = inv_dZ_j_jm1[j]
            inv_dZ_jp1_j__j   = inv_dZ_jp1_j[j]
            inv_dZ_jp2_jp1__j = inv_dZ_jp2_jp1[j]

            u_contact_over_c_jp1 = @fastmath u_Z_star[i,jp1] * inv_c

            @simd for k in dN_ν

                # Values of J used twice below:
                J_ij    = J[k,i,j]
                J_i_jp1 = J[k,i,jp1]

                # Reconstruct at Z_{j+1/2}
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R =
                ReconstructInterface( J_ij,       J_i_jp1,      J[k,i,jp2],   J[k,i,jm1],
                                      H_r[k,i,j], H_r[k,i,jp1], H_r[k,i,jp2], H_r[k,i,jm1],
                                      H_Z[k,i,j], H_Z[k,i,jp1], H_Z[k,i,jp2], H_Z[k,i,jm1],

                                      ΔZ_L_j, ΔZ_R_j,
                                      inv_dZ_j_jm1__j,
                                      inv_dZ_jp1_j__j,
                                      inv_dZ_jp2_jp1__j,

                                      α_F[k,i,j], α_F[k,i,jp1] )

                # Interface D and K
                _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                KZZ_L = @fastmath D_ZZ_L * J_L;   KZZ_R = @fastmath D_ZZ_R * J_R
                KrZ_L = @fastmath D_rZ_L * J_L;   KrZ_R = @fastmath D_rZ_R * J_R

                # Write interior Z-fluxes:
                F_Z_jp1, F_Z_Hr_jp1, F_Z_HZ_jp1 = GetFluxM1( FluxSolver, Val(:Z), 
                                                             HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                             J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                             τ_L, τ_R, J_ij, J_i_jp1, f_L, f_R,
                                                             nr_L, nr_R, nZ_L, nZ_R )

                # Add advection fluxes:
                F_Z[k,i,jp1]    = @fastmath F_Z_jp1     +  GetAdvectionFlux( u_contact_over_c_jp1, J_L, J_R )
                F_Z_Hr[k,i,jp1] = @fastmath F_Z_Hr_jp1  +  GetAdvectionFlux( u_contact_over_c_jp1, Hr_L, Hr_R )
                F_Z_HZ[k,i,jp1] = @fastmath F_Z_HZ_jp1  +  GetAdvectionFlux( u_contact_over_c_jp1, HZ_L, HZ_R )

            end # End of k-loop 
        end # End of i-loop 
    end # End of j-loop 


    # Intervals used many times below (for each i and bin):
    ΔZ_L_Nm1            = @inbounds ΔZ_L[N_Z-1] 
    ΔZ_R_Nm1            = @inbounds ΔZ_R[N_Z-1]
    inv_dZ_j_jm1__Nm1   = @inbounds inv_dZ_j_jm1[N_Z-1]
    inv_dZ_jp1_j__Nm1   = @inbounds inv_dZ_jp1_j[N_Z-1]
    inv_dZ_jp2_jp1__Nm1 = @inbounds inv_dZ_jp2_jp1[N_Z-1]

    @threads for i in dN_r 
        @inbounds begin 

            u_contact_over_c_N   = @fastmath u_Z_star[i,N_Z] * inv_c
            u_contact_over_c_Np1 = @fastmath u_Z_star[i,N_Z+1] * inv_c

            @simd for k in dN_ν

                # Load commonly used values of J and H, and α_F:
                J_N     = J[k,i,N_Z]
                J_Nm1   = J[k,i,N_Z-1]
                H_r_N   = H_r[k,i,N_Z]
                H_r_Nm1 = H_r[k,i,N_Z-1]
                H_Z_N   = H_Z[k,i,N_Z]
                H_Z_Nm1 = H_Z[k,i,N_Z-1]
                α_F_N   = α_F[k,i,N_Z]

                # Boundary conditions at interface j = N_Z + 1/2:
                if BC_Z_top === :Reflect
                    # Ghost cell j = N_Z + 1:
                    J_Np1   =   J_N
                    HZ_Np1  = - H_Z_N    # Reflect
                    Hr_Np1  =   H_r_N 
                    # Ghost cell j = N_Z + 2 (mirror of cell j = N_Z - 1):
                    J_Np2   =   J_Nm1
                    HZ_Np2  = - H_Z_Nm1  # Reflect
                    Hr_Np2  =   H_r_Nm1
                elseif BC_Z_top === :Outflow
                    # Ghost cell j = N_Z + 1:
                    J_Np1   =   J_N
                    HZ_Np1  =   max( H_Z_N, 0.0 )    
                    Hr_Np1  =   H_r_N 
                    # Ghost cell j = N_Z + 2 (copy of cell j = N_Z + 1):
                    J_Np2   =   J_Np1
                    HZ_Np2  =   HZ_Np1
                    Hr_Np2  =   Hr_Np1 
                end

                ##############################################################################
                ##############         INTERFACE j = N_Z - 1/2         #######################
                ##############################################################################

                # Reconstruct variables at interface Z_{N_Z-1/2} (i.e. Z[N_Z]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R = ReconstructInterface( J_Nm1,   J_N,   J_Np1,  J[k,i,N_Z-2],
                                                   H_r_Nm1, H_r_N, Hr_Np1, H_r[k,i,N_Z-2],
                                                   H_Z_Nm1, H_Z_N, HZ_Np1, H_Z[k,i,N_Z-2],

                                                   ΔZ_L_Nm1, ΔZ_R_Nm1,
                                                   inv_dZ_j_jm1__Nm1,
                                                   inv_dZ_jp1_j__Nm1,
                                                   inv_dZ_jp2_jp1__Nm1,

                                                   α_F[k,i,N_Z-1], α_F_N )

                # Compute interface D-values:
                _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # Compute K = D × J at the interface j = N_Z - 1/2:
                KZZ_L = D_ZZ_L * J_L
                KZZ_R = D_ZZ_R * J_R
                KrZ_L = D_rZ_L * J_L
                KrZ_R = D_rZ_R * J_R

                # Fluxes at interface j = N_Z - 1/2:
                F_Z_N, F_Z_Hr_N, F_Z_HZ_N = GetFluxM1( FluxSolver, Val(:Z), 
                                                       HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                       J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                       τ_L, τ_R, J_Nm1, J_N, f_L, f_R,
                                                       nr_L, nr_R, nZ_L, nZ_R  )

                # Add advection fluxes:
                F_Z[k,i,N_Z]    = @fastmath F_Z_N     +  GetAdvectionFlux( u_contact_over_c_N, J_L, J_R )
                F_Z_Hr[k,i,N_Z] = @fastmath F_Z_Hr_N  +  GetAdvectionFlux( u_contact_over_c_N, Hr_L, Hr_R )
                F_Z_HZ[k,i,N_Z] = @fastmath F_Z_HZ_N  +  GetAdvectionFlux( u_contact_over_c_N, HZ_L, HZ_R )

                ##############################################################################
                ###############        INTERFACE j = N_Z + 1/2         #######################
                ##############################################################################

                # Reconstruct variables at interface Z_{N_Z+1/2} (i.e. Z[N_Z+1]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R = ReconstructInterface( J_N,   J_Np1,   J_Np2,   J_Nm1,
                                                   H_r_N, Hr_Np1,  Hr_Np2,  H_r_Nm1,
                                                   H_Z_N, HZ_Np1,  HZ_Np2,  H_Z_Nm1,

                                                   ΔZ_L_OuterBoundary, ΔZ_R_OuterBoundary,
                                                   inv_dZ_j_jm1_OuterBoundary,
                                                   inv_dZ_jp1_j_OuterBoundary,
                                                   inv_dZ_jp2_jp1_OuterBoundary,

                                                   α_F_N, α_F_N  # Assume opacity in ghost cell j = N_Z+1 is same as in j = N_Z
                                                 )

                # Compute interface D-values:
                _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # Compute K = D × J at the interface j = N_Z + 1/2:
                KZZ_L = @fastmath D_ZZ_L * J_L
                KZZ_R = @fastmath D_ZZ_R * J_R
                KrZ_L = @fastmath D_rZ_L * J_L
                KrZ_R = @fastmath D_rZ_R * J_R

                # Fluxes at interface j = N_Z + 1/2:
                F_Z_Np1, F_Z_Hr_Np1, F_Z_HZ_Np1 = GetFluxM1( FluxSolver, Val(:Z), 
                                                             HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                             J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                             τ_L, τ_R, J_N, J_Np1, f_L, f_R,
                                                             nr_L, nr_R, nZ_L, nZ_R )
            
                # Add advection fluxes:
                F_Z[k,i,N_Z+1]    = @fastmath F_Z_Np1     +  GetAdvectionFlux( u_contact_over_c_Np1, J_L, J_R )
                F_Z_Hr[k,i,N_Z+1] = @fastmath F_Z_Hr_Np1  +  GetAdvectionFlux( u_contact_over_c_Np1, Hr_L, Hr_R )
                F_Z_HZ[k,i,N_Z+1] = @fastmath F_Z_HZ_Np1  +  GetAdvectionFlux( u_contact_over_c_Np1, HZ_L, HZ_R )


            end # End of k-loop 
        end # End of inbounds 
    end # End of i-loop 

    ###############################################################
    #######                                                 #######
    #######             U P D A T E  J  A N D  H            #######
    #######                                                 #######
    ###############################################################

    # With all fluxes computes, we can now update J and the components 
    # of H over the time-step Δt:

    ####  UPDATE J AND H  ##############################################################

    @threads for j in dN_Z 
        @inbounds for i in dN_r 

            # Area & volume arrays (accessed many times):
            A_r_ij     = A_r[i,j] 
            A_r_ip1_j  = A_r[i+1,j]
            A_Z_ij     = A_Z[i,j] 
            A_Z_i_jp1  = A_Z[i,j+1]
            inv_ΔV_ij  = inv_ΔV[i,j]

            # Velocity gradient terms (used multiple times):
            ∂u_r_∂r_ij    = ∂u_r_∂r[i,j]
            ∂u_r_∂Z_ij    = ∂u_r_∂Z[i,j]
            ∂u_Z_∂r_ij    = ∂u_Z_∂r[i,j]
            ∂u_Z_∂Z_ij    = ∂u_Z_∂Z[i,j]

            @simd for k in dN_ν
                # Store old J and H values, needed for the radiation pressure work term, 
                # and aberration term for H:
                J_stored   = J[k,i,j]
                H_r_stored = H_r[k,i,j]
                H_Z_stored = H_Z[k,i,j]

                # Compute values of Eddington tensor (at the stored/old values for consistency 
                # with an explicit update) for use below:
                Drr, DZZ, DrZ = Compute_D_Edd( H_r_stored, H_Z_stored, J_stored )

                # Update J and H from the flux terms:
                J_new      = max( UpdateConservativeQuantity( J_stored, c̃Δt, inv_ΔV_ij, 
                                                              A_r_ij, A_r_ip1_j,
                                                              A_Z_ij, A_Z_i_jp1, 
                                                              F_r[k,i,j], F_r[k,i+1,j],
                                                              F_Z[k,i,j], F_Z[k,i,j+1] ),
                                  Jeps )

                H_Z_new    = UpdateConservativeQuantity( H_Z_stored, c̃Δt, inv_ΔV_ij,
                                                         A_r_ij, A_r_ip1_j,
                                                         A_Z_ij, A_Z_i_jp1,
                                                         F_r_HZ[k,i,j], F_r_HZ[k,i+1,j],
                                                         F_Z_HZ[k,i,j], F_Z_HZ[k,i,j+1] )

                H_r_new    = UpdateConservativeQuantity( H_r_stored, c̃Δt, inv_ΔV_ij,
                                                         A_r_ij, A_r_ip1_j,
                                                         A_Z_ij, A_Z_i_jp1,
                                                         F_r_Hr[k,i,j], F_r_Hr[k,i+1,j],
                                                         F_Z_Hr[k,i,j], F_Z_Hr[k,i,j+1] )

                # Get D:∇u, needed below:
                D_colon_∇u = D_colon_∇u_Cartesian( Drr, DrZ, DZZ, 
                                                   ∂u_r_∂r_ij, ∂u_r_∂Z_ij, ∂u_Z_∂r_ij, ∂u_Z_∂Z_ij )

                # Add aberration update of H_r and H_Z:
                ΔH_r, ΔH_Z = ΔH_Aberration( H_r_stored, H_Z_stored, c̃Δt_over_c,
                                            ∂u_r_∂r_ij, ∂u_r_∂Z_ij, ∂u_Z_∂r_ij, ∂u_Z_∂Z_ij )

                H_r_new = @fastmath H_r_new  +  ΔH_r
                H_Z_new = @fastmath H_Z_new  +  ΔH_Z

                # Finally, also add the contribution from (1/c̃) ∂J/∂t = - (D:∇u/c) J: 
                J_new   = max( J_new  -  c̃Δt_over_c * D_colon_∇u * J_stored,  Jeps ) 

                # Ensure causality: 
                H_r_new, H_Z_new  = EnforceCausalReducedFlux( J_new, H_r_new, H_Z_new )

                # Save updated values:
                H_r[k,i,j] = H_r_new
                H_Z[k,i,j] = H_Z_new
                J[k,i,j]   = J_new 

            end # End of k-loop 
        end # End of i-loop 
    end # End of j-loop 

    # Update J_old if the option is turned on:
    if UpdateJold
        copyto!( J_old, J )
    end

    return nothing    
end


""" This function performs an RT advection step of a given number of frequency bins, using 
    the M1 two-moment RT solver. """
function AdvectJandH!( c̃Δt::Float64,                   # c̃ × Δt 
                       α_F::AbstractArray{Float64, 3}, # Flux opacities              (N_ν × N_r x N_Z)

                         J::AbstractArray{Float64, 3}, # J (latest/updated values)   (N_ν × N_r x N_Z)
                     J_old::AbstractArray{Float64, 3}, # J (old values)              (N_ν × N_r x N_Z)

                       H_r::AbstractArray{Float64, 3}, # r-direction component of H  (N_ν × N_r x N_Z)
                       H_Z::AbstractArray{Float64, 3}, # Z-direction component of H  (N_ν × N_r x N_Z)

               K_ϕϕ_iphalf::AbstractArray{Float64, 3}, # ϕϕ-components of K at i+1/2           (N_ν × N_r x N_Z)
               K_ϕϕ_imhalf::AbstractArray{Float64, 3}, # ϕϕ-components of K at i-1/2           (N_ν × N_r x N_Z)

                       F_r::AbstractArray{Float64, 3}, # r-direction flux of J               (N_ν × N_r+1 x N_Z)
                       F_Z::AbstractArray{Float64, 3}, # Z-direction flux of J               (N_ν × N_r   x N_Z+1)

                    F_r_Hr::AbstractArray{Float64, 3}, # r-direction flux of H_r             (N_ν × N_r+1 x N_Z)
                    F_Z_Hr::AbstractArray{Float64, 3}, # Z-direction flux of H_r             (N_ν × N_r   x N_Z+1)
                    F_r_HZ::AbstractArray{Float64, 3}, # r-direction flux of H_Z             (N_ν × N_r+1 x N_Z)
                    F_Z_HZ::AbstractArray{Float64, 3}, # Z-direction flux of H_Z             (N_ν × N_r   x N_Z+1)

                    # Geometry struct:
                  geometry::Union{GeometryCylindrical, GeometryCartesian},

                # Gas velocity field:
                  u_r_star::AbstractArray{Float64,2},
                  u_Z_star::AbstractArray{Float64,2},
                   ∂u_r_∂r::AbstractArray{Float64,2},
                   ∂u_r_∂Z::AbstractArray{Float64,2}, 
                   ∂u_Z_∂r::AbstractArray{Float64,2}, 
                   ∂u_Z_∂Z::AbstractArray{Float64,2},
                u_r_over_r::AbstractArray{Float64,2},
            
                # Other stuff:
                UpdateJold::Bool         = true,           # Whether to update J_old = J at the end (should be false for stellar RT)

                 BC_r_left::Symbol       = :Reflect,       # Boundary condition at i = 1/2 
                BC_r_right::Symbol       = :Outflow,       # Boundary condition at i = N_r + 1/2
                  BC_Z_bot::Symbol       = :Reflect,       # Boundary condition at j = 1/2 
                  BC_Z_top::Symbol       = :Outflow,       # Boundary condition at j = N_Z + 1/2 
                      kmin::Int          = 1,              # Advect bins between kmin and...
                      kmax::Int          = size( J, 1 ),   # kmax 
                FluxSolver::Val          = Val(:HLL)       # M1 flux solver choice, HLL or GLF  
                                   )

    # Needed geometry (intervals, areas etc):
    inv_ΔV     = geometry.inv_ΔV

    if geometry isa GeometryCylindrical

        # Interface areas:
        A_r        = geometry.A_r 
        A_Z        = geometry.A_Z

        # For the geometric source term:
        inv_r_cell = geometry.inv_r_cell 

        # r-intervals:
        Δr_L            = geometry.Δr_L
        Δr_R            = geometry.Δr_R
        inv_dr_i_im1    = geometry.inv_dr_i_im1
        inv_dr_ip1_i    = geometry.inv_dr_ip1_i
        inv_dr_ip2_ip1  = geometry.inv_dr_ip2_ip1
                             
        Δr_L_InnerBoundary           = geometry.Δr_L_InnerBoundary
        Δr_R_InnerBoundary           = geometry.Δr_R_InnerBoundary
        inv_dr_i_im1_InnerBoundary   = geometry.inv_dr_i_im1_InnerBoundary
        inv_dr_ip1_i_InnerBoundary   = geometry.inv_dr_ip1_i_InnerBoundary
        inv_dr_ip2_ip1_InnerBoundary = geometry.inv_dr_ip2_ip1_InnerBoundary
                             
        Δr_L_OuterBoundary           = geometry.Δr_L_OuterBoundary
        Δr_R_OuterBoundary           = geometry.Δr_R_OuterBoundary
        inv_dr_i_im1_OuterBoundary   = geometry.inv_dr_i_im1_OuterBoundary
        inv_dr_ip1_i_OuterBoundary   = geometry.inv_dr_ip1_i_OuterBoundary
        inv_dr_ip2_ip1_OuterBoundary = geometry.inv_dr_ip2_ip1_OuterBoundary

        # Z-intervals:
        ΔZ_L            = geometry.ΔZ_L
        ΔZ_R            = geometry.ΔZ_R
        inv_dZ_j_jm1    = geometry.inv_dZ_j_jm1
        inv_dZ_jp1_j    = geometry.inv_dZ_jp1_j
        inv_dZ_jp2_jp1  = geometry.inv_dZ_jp2_jp1
                             
        ΔZ_L_InnerBoundary           = geometry.ΔZ_L_InnerBoundary
        ΔZ_R_InnerBoundary           = geometry.ΔZ_R_InnerBoundary
        inv_dZ_j_jm1_InnerBoundary   = geometry.inv_dZ_j_jm1_InnerBoundary
        inv_dZ_jp1_j_InnerBoundary   = geometry.inv_dZ_jp1_j_InnerBoundary
        inv_dZ_jp2_jp1_InnerBoundary = geometry.inv_dZ_jp2_jp1_InnerBoundary
                             
        ΔZ_L_OuterBoundary           = geometry.ΔZ_L_OuterBoundary
        ΔZ_R_OuterBoundary           = geometry.ΔZ_R_OuterBoundary
        inv_dZ_j_jm1_OuterBoundary   = geometry.inv_dZ_j_jm1_OuterBoundary
        inv_dZ_jp1_j_OuterBoundary   = geometry.inv_dZ_jp1_j_OuterBoundary
        inv_dZ_jp2_jp1_OuterBoundary = geometry.inv_dZ_jp2_jp1_OuterBoundary

        # Call M1 advection:
        AdvectJandH_Cylindrical!(  c̃Δt,           α_F,        
                                   J,             J_old, 
                                   H_r,           H_Z, 
                                   K_ϕϕ_iphalf,   K_ϕϕ_imhalf,
                                   F_r,           F_Z,                     # Fluxes of J 
                                   F_r_Hr,        F_Z_Hr,                  # Fluxes of H_r 
                                   F_r_HZ,        F_Z_HZ,                  # Fluxes of H_Z 
                                   inv_ΔV,  A_r,  A_Z, inv_r_cell, 
                                   u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,

                                   # Intervals & inverse intervals, needed for interface reconstruction 
                                   # & cell optical depths:
                                   Δr_L, Δr_R,
                                   inv_dr_i_im1,
                                   inv_dr_ip1_i,
                                   inv_dr_ip2_ip1,
                             
                                   Δr_L_InnerBoundary, Δr_R_InnerBoundary,
                                   inv_dr_i_im1_InnerBoundary,
                                   inv_dr_ip1_i_InnerBoundary,
                                   inv_dr_ip2_ip1_InnerBoundary,
                             
                                   Δr_L_OuterBoundary, Δr_R_OuterBoundary,
                                   inv_dr_i_im1_OuterBoundary,
                                   inv_dr_ip1_i_OuterBoundary,
                                   inv_dr_ip2_ip1_OuterBoundary,

                                   ΔZ_L, ΔZ_R,
                                   inv_dZ_j_jm1,
                                   inv_dZ_jp1_j,
                                   inv_dZ_jp2_jp1,
   
                                   ΔZ_L_InnerBoundary, ΔZ_R_InnerBoundary,
                                   inv_dZ_j_jm1_InnerBoundary,
                                   inv_dZ_jp1_j_InnerBoundary,
                                   inv_dZ_jp2_jp1_InnerBoundary,
                              
                                   ΔZ_L_OuterBoundary, ΔZ_R_OuterBoundary,
                                   inv_dZ_j_jm1_OuterBoundary,
                                   inv_dZ_jp1_j_OuterBoundary,
                                   inv_dZ_jp2_jp1_OuterBoundary,

                                   # Other stuff:
                                   UpdateJold, 
                                   BC_r_right,  BC_Z_bot,  BC_Z_top,
                                   kmin, kmax,
                                   FluxSolver )

    elseif geometry isa GeometryCartesian

        # Interface areas:
        A_r        = geometry.A_X 
        A_Z        = geometry.A_Y

        # X-intervals:
        Δr_L            = geometry.ΔX_L
        Δr_R            = geometry.ΔX_R
        inv_dr_i_im1    = geometry.inv_dX_i_im1
        inv_dr_ip1_i    = geometry.inv_dX_ip1_i
        inv_dr_ip2_ip1  = geometry.inv_dX_ip2_ip1
                             
        Δr_L_InnerBoundary           = geometry.ΔX_L_InnerBoundary
        Δr_R_InnerBoundary           = geometry.ΔX_R_InnerBoundary
        inv_dr_i_im1_InnerBoundary   = geometry.inv_dX_i_im1_InnerBoundary
        inv_dr_ip1_i_InnerBoundary   = geometry.inv_dX_ip1_i_InnerBoundary
        inv_dr_ip2_ip1_InnerBoundary = geometry.inv_dX_ip2_ip1_InnerBoundary
                             
        Δr_L_OuterBoundary           = geometry.ΔX_L_OuterBoundary
        Δr_R_OuterBoundary           = geometry.ΔX_R_OuterBoundary
        inv_dr_i_im1_OuterBoundary   = geometry.inv_dX_i_im1_OuterBoundary
        inv_dr_ip1_i_OuterBoundary   = geometry.inv_dX_ip1_i_OuterBoundary
        inv_dr_ip2_ip1_OuterBoundary = geometry.inv_dX_ip2_ip1_OuterBoundary

        # Y-intervals:
        ΔZ_L            = geometry.ΔY_L
        ΔZ_R            = geometry.ΔY_R
        inv_dZ_j_jm1    = geometry.inv_dY_j_jm1
        inv_dZ_jp1_j    = geometry.inv_dY_jp1_j
        inv_dZ_jp2_jp1  = geometry.inv_dY_jp2_jp1
                             
        ΔZ_L_InnerBoundary           = geometry.ΔY_L_InnerBoundary
        ΔZ_R_InnerBoundary           = geometry.ΔY_R_InnerBoundary
        inv_dZ_j_jm1_InnerBoundary   = geometry.inv_dY_j_jm1_InnerBoundary
        inv_dZ_jp1_j_InnerBoundary   = geometry.inv_dY_jp1_j_InnerBoundary
        inv_dZ_jp2_jp1_InnerBoundary = geometry.inv_dY_jp2_jp1_InnerBoundary
                             
        ΔZ_L_OuterBoundary           = geometry.ΔY_L_OuterBoundary
        ΔZ_R_OuterBoundary           = geometry.ΔY_R_OuterBoundary
        inv_dZ_j_jm1_OuterBoundary   = geometry.inv_dY_j_jm1_OuterBoundary
        inv_dZ_jp1_j_OuterBoundary   = geometry.inv_dY_jp1_j_OuterBoundary
        inv_dZ_jp2_jp1_OuterBoundary = geometry.inv_dY_jp2_jp1_OuterBoundary

        # Call M1 advection:
        AdvectJandH_Cartesian!( c̃Δt,           α_F,        
                                J,             J_old, 
                                H_r,           H_Z, 
                                F_r,           F_Z,                     # Fluxes of J 
                                F_r_Hr,        F_Z_Hr,                  # Fluxes of H_r 
                                F_r_HZ,        F_Z_HZ,                  # Fluxes of H_Z 
                                inv_ΔV,  A_r,  A_Z, 
                                u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, 

                                # Intervals & inverse intervals, needed for interface reconstruction 
                                # & cell optical depths:
                                Δr_L, Δr_R,
                                inv_dr_i_im1,
                                inv_dr_ip1_i,
                                inv_dr_ip2_ip1,
                             
                                Δr_L_InnerBoundary, Δr_R_InnerBoundary,
                                inv_dr_i_im1_InnerBoundary,
                                inv_dr_ip1_i_InnerBoundary,
                                inv_dr_ip2_ip1_InnerBoundary,
                             
                                Δr_L_OuterBoundary, Δr_R_OuterBoundary,
                                inv_dr_i_im1_OuterBoundary,
                                inv_dr_ip1_i_OuterBoundary,
                                inv_dr_ip2_ip1_OuterBoundary,

                                ΔZ_L, ΔZ_R,
                                inv_dZ_j_jm1,
                                inv_dZ_jp1_j,
                                inv_dZ_jp2_jp1,
  
                                ΔZ_L_InnerBoundary, ΔZ_R_InnerBoundary,
                                inv_dZ_j_jm1_InnerBoundary,
                                inv_dZ_jp1_j_InnerBoundary,
                                inv_dZ_jp2_jp1_InnerBoundary,
                              
                                ΔZ_L_OuterBoundary, ΔZ_R_OuterBoundary,
                                inv_dZ_j_jm1_OuterBoundary,
                                inv_dZ_jp1_j_OuterBoundary,
                                inv_dZ_jp2_jp1_OuterBoundary,

                                # Other stuff:
                                UpdateJold, 
                                BC_r_left,  BC_r_right,  BC_Z_bot,  BC_Z_top,
                                kmin, kmax,
                                FluxSolver )
    end

    return nothing 
end

