
""" FILE:        AdvectJandH.jl

    PURPOSE:     Contains the code for advecting the mean intensity J, and the
                 first moment H (the Eddington tensor D = K / J is estimated 
                 using an M1 closure). Uses an explicit finite volume method, with 
                 2:nd order (linear) spatial reconstruction, and 1:st order 
                 time integration. The method is dimensionally unsplit. 

    REFERENCES:  (1) Skinner & Ostriker (2013) 'A Two-moment Radiation Hydrodynamics 
                     Module in Athena Using a Time-explicit Godunov Method', ApJS, 206, 2

"""

using .ConstantsAndFunctions  # For slope limiters 
using Base.Threads            # For CPU acceleration

# Export functions:
export GetRHDCFLTimeStep, AdvectJandH!


const DEBUG_M1Advection::Bool = false  # For debugging (e.g. checking correct array sizes etc)
const inv_c::Float64          = 1 / c  # Convenient constant 


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
    ϵ            = @fastmath numerator_ϵ / ( numerator_ϵ  +  τ_cell * τ_cell)  # → 1 for thin, → 1 / τ_cell for thick 
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


""" Get advection flux (u/c) * J or (u/c) * H evaluated at interface 
    with velocity u_contact, obtained from the hydro solver. 
    Uses upwind, following Skinner et al. (2019). """
@inline @fastmath function GetAdvectionFlux( u_contact_over_c::Float64, # Gas velocity u at the interface (contact velocity), divided by c
                                                          J_L::Float64, # J_L or H_L 
                                                          J_R::Float64  # J_R or H_R 
                                             ) 
    
    return u_contact_over_c * ifelse( u_contact_over_c < 0.0, J_R, J_L )
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

    Qnew    :   Value of Q at time-step n + 1 
"""
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




@inline function AdvectJandH_Threaded!( c̃Δt::Float64,                   # c̃ × Δt 
                               α_F::AbstractArray{Float64, 2}, # Flux opacities              (N_r x N_Z)

                         J::AbstractArray{Float64, 2}, # J (latest/updated values)   (N_r x N_Z)
                     J_old::AbstractArray{Float64, 2}, # J (old values)              (N_r x N_Z)

                       H_r::AbstractArray{Float64, 2}, # r-direction component of H  (N_r x N_Z)
                       H_Z::AbstractArray{Float64, 2}, # Z-direction component of H  (N_r x N_Z)

                      D_rr::AbstractArray{TypeOfD, 2}, # rr-components of the Eddington tensor (N_r x N_Z)
                      D_ZZ::AbstractArray{TypeOfD, 2}, # ZZ-components of the Eddington tensor (N_r x N_Z)
                      D_rZ::AbstractArray{TypeOfD, 2}, # rZ-components of the Eddington tensor (N_r x N_Z)
               K_ϕϕ_iphalf::AbstractArray{Float64, 2}, # ϕϕ-components of K at i+1/2           (N_r x N_Z)
               K_ϕϕ_imhalf::AbstractArray{Float64, 2}, # ϕϕ-components of K at i-1/2           (N_r x N_Z)

                       F_r::AbstractArray{Float64, 2}, # r-direction flux of J               (N_r+1 x N_Z)
                       F_Z::AbstractArray{Float64, 2}, # Z-direction flux of J               (N_r   x N_Z+1)

                    F_r_Hr::AbstractArray{Float64, 2}, # r-direction flux of H_r             (N_r+1 x N_Z)
                    F_Z_Hr::AbstractArray{Float64, 2}, # Z-direction flux of H_r             (N_r   x N_Z+1)
                    F_r_HZ::AbstractArray{Float64, 2}, # r-direction flux of H_Z             (N_r+1 x N_Z)
                    F_Z_HZ::AbstractArray{Float64, 2}, # Z-direction flux of H_Z             (N_r   x N_Z+1)

                    inv_ΔV::AbstractArray{Float64, 2}, # Inverse cell volumes 1 / ΔV         (N_r   x N_Z)
                       A_r::AbstractArray{Float64, 2}, # Cell interface areas in r-direction (N_r+1 x N_Z)
                       A_Z::AbstractArray{Float64, 2}, # Cell interface areas in Z-direction (N_r   x N_Z+1)
                inv_r_cell::AbstractArray{Float64},    # ⟨1/r⟩ for every cell                (N_r)
                         r::AbstractArray{Float64},

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

         CylindricalCoords::Bool,         # Cylindrical coordinates assumed if true, otherwise Cartesian 
                 BC_r_left::Symbol,       # Boundary condition at i = 1/2 
                BC_r_right::Symbol,       # Boundary condition at i = N_r + 1/2
                  BC_Z_bot::Symbol,       # Boundary condition at j = 1/2 
                  BC_Z_top::Symbol        # Boundary condition at j = N_Z + 1/2 
                                   )  where {TypeOfD<:AbstractFloat}

    # Grid & intervals:
    N_r, N_Z        = size( J )
    dN_r            = 1:N_r
    dN_Z            = 1:N_Z
    dN_rint         = 2:(N_r-2)
    dN_Zint         = 2:(N_Z-2)

    # Convenient constant:
    c̃Δt_over_c      = c̃Δt * inv_c

    if DEBUG_M1Advection
        @assert size( u_r_star, 1 ) == N_r + 1 "Contact velocity in r-direction/X-direction must have size $(N_r+1)."
        @assert size( u_Z_star, 2 ) == N_Z + 1 "Contact velocity in Z-direction/Y-direction must have size $(N_Z+1)."
    end

    # First update the Eddington factors:
    Update_D_Edd_Threaded!( D_rr, D_ZZ, D_rZ, H_r, H_Z, J )

    ###############################################################
    #######                                                 #######
    #######           C O M P U T E  F L U X E S            #######
    #######                                                 #######
    ###############################################################

    # We start by reconstructing & computing the interface fluxes, 
    # starting with the r-direction (order does not matter here).
    
    # r-direction fluxes:
    @threads for j in dN_Z
        @inbounds begin
            ##############################################################################
            ##############          INTERFACE i = 1/2          ###########################
            ##############################################################################

            if BC_r_left === :Reflect 
                # Ghost cell i = 0:
                J_0    = J[1,j] 
                Hr_0   = - H_r[1,j]   # Reflect radial flux
                HZ_0   = H_Z[1,j]
                # Ghost cell i = -1:
                J_m1   = J[2,j] 
                Hr_m1  = - H_r[2,j]   # Reflect radial flux
                HZ_m1  = H_Z[2,j]
            elseif BC_r_left === :Outflow  
                # Ghost cell i = 0:
                J_0    = J[1,j] 
                Hr_0   = min( H_r[1,j], 0.0 )          
                HZ_0   = H_Z[1,j]
                # Ghost cell i = -1 (no gradients, so equal to cells i = 0 and 1):
                J_m1   = J_0 
                Hr_m1  = Hr_0   
                HZ_m1  = HZ_0
            end

            # Reconstruct variables at interface r_{1/2} (i.e. r[1]):
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
            τ_L, τ_R, f_L, f_R, nr_L, nr_R,
            nZ_L, nZ_R  = ReconstructInterface( J_0,     J[1,j],   J[2,j],   J_m1,
                                                Hr_0,    H_r[1,j], H_r[2,j], Hr_m1,
                                                HZ_0,    H_Z[1,j], H_Z[2,j], HZ_m1,

                                                Δr_L_InnerBoundary, Δr_R_InnerBoundary,
                                                inv_dr_i_im1_InnerBoundary,
                                                inv_dr_ip1_i_InnerBoundary,
                                                inv_dr_ip2_ip1_InnerBoundary,
                    
                                                α_F[1,j], α_F[1,j]  # Assume ghost cell 0 has same opacity as cell i = 1
                                               )

            # Compute interface D-values:
            D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            # Compute K = D × J at the interface:
            Krr_L = D_rr_L * J_L
            Krr_R = D_rr_R * J_R
            KZZ_L = D_ZZ_L * J_L
            KZZ_R = D_ZZ_R * J_R
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at i = 1/2:
            F_r[1,j], F_r_Hr[1,j], F_r_HZ[1,j] = GetFluxGLF( Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                             J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R,   
                                                             τ_L, τ_R, J_0, J[1,j] )

            # Add advection fluxes:
            u_contact_over_c    = u_r_star[1,j] * inv_c
            F_r[1,j]           += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_r_Hr[1,j]        += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_r_HZ[1,j]        += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

            # Fill K_ϕϕ at i = 1/2 for cell i = 1, needed for the geometric source term:
            K_ϕϕ_R           = clamp( 1.0 - D_rr_R - D_ZZ_R, 0.0, 1.0 ) * J_R
            K_ϕϕ_imhalf[1,j] = K_ϕϕ_R 

            ##############################################################################
            ##############          INTERFACE i = 3/2          ###########################
            ##############################################################################

            # Reconstruct variables at interface r_{3/2} (i.e. r[2]):
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
            τ_L, τ_R, f_L, f_R, nr_L, nr_R,
            nZ_L, nZ_R = ReconstructInterface( J[1,j],   J[2,j],   J[3,j],   J_0,
                                               H_r[1,j], H_r[2,j], H_r[3,j], Hr_0,
                                               H_Z[1,j], H_Z[2,j], H_Z[3,j], HZ_0,

                                               Δr_L[1], Δr_R[1],
                                               inv_dr_i_im1[1],
                                               inv_dr_ip1_i[1],
                                               inv_dr_ip2_ip1[1],

                                               α_F[1,j], α_F[2,j] )

            # Compute interface D-values:
            D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            # Compute K = D × J at the interface:
            Krr_L = D_rr_L * J_L 
            Krr_R = D_rr_R * J_R 
            KZZ_L = D_ZZ_L * J_L 
            KZZ_R = D_ZZ_R * J_R 
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at i = 3/2:
            F_r[2,j], F_r_Hr[2,j], F_r_HZ[2,j] = GetFluxGLF( Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                             J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                             τ_L, τ_R, J[1,j], J[2,j] )

            # Add advection fluxes:
            u_contact_over_c    = u_r_star[2,j] * inv_c
            F_r[2,j]           += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_r_Hr[2,j]        += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_r_HZ[2,j]        += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

            # Get K_ϕϕ at interfaces:
            K_ϕϕ_L           = clamp( 1.0 - D_rr_L - D_ZZ_L, 0.0, 1.0 ) * J_L 
            K_ϕϕ_R           = clamp( 1.0 - D_rr_R - D_ZZ_R, 0.0, 1.0 ) * J_R 

            K_ϕϕ_iphalf[1,j] = K_ϕϕ_L    # K_ϕϕ at i+1/2 for cell i = 1
            K_ϕϕ_imhalf[2,j] = K_ϕϕ_R    # K_ϕϕ at i-1/2 for cell i = 2

            ##############################################################################
            ##############        INTERIOR INTERFACES 2 ≤ i ≤ N_r-2          #############
            ##############################################################################
            @simd for i in dN_rint
                ip1 = i + 1
                ip2 = i + 2
                im1 = i - 1

                # Reconstruct variables at interface r_{i+1/2} (i.e. r[i+1]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R = ReconstructInterface( J[i,j],   J[ip1,j],   J[ip2,j],   J[im1,j],
                                                   H_r[i,j], H_r[ip1,j], H_r[ip2,j], H_r[im1,j],
                                                   H_Z[i,j], H_Z[ip1,j], H_Z[ip2,j], H_Z[im1,j],

                                                   Δr_L[i], Δr_R[i],
                                                   inv_dr_i_im1[i],
                                                   inv_dr_ip1_i[i],
                                                   inv_dr_ip2_ip1[i],

                                                   α_F[i,j], α_F[ip1,j]  )

                # Compute interface D-values:
                D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # Compute K = D × J at the interface:
                Krr_L = D_rr_L * J_L 
                Krr_R = D_rr_R * J_R 
                KZZ_L = D_ZZ_L * J_L
                KZZ_R = D_ZZ_R * J_R
                KrZ_L = D_rZ_L * J_L
                KrZ_R = D_rZ_R * J_R

                # Radial flux of J, H_r, and H_Z:
                F_r[i+1,j], F_r_Hr[i+1,j], F_r_HZ[i+1,j] = GetFluxGLF( Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                                       J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                                       τ_L, τ_R, J[i,j], J[ip1,j] )

                # Add advection fluxes:
                u_contact_over_c     = u_r_star[i+1,j] * inv_c
                F_r[i+1,j]          += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
                F_r_Hr[i+1,j]       += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
                F_r_HZ[i+1,j]       += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

                # Get K_ϕϕ at interfaces:
                K_ϕϕ_L             = clamp( 1.0 - D_rr_L - D_ZZ_L, 0.0, 1.0 ) * J_L 
                K_ϕϕ_R             = clamp( 1.0 - D_rr_R - D_ZZ_R, 0.0, 1.0 ) * J_R 

                K_ϕϕ_iphalf[i,j]   = K_ϕϕ_L    # K_ϕϕ at i+1/2 for cell i
                K_ϕϕ_imhalf[i+1,j] = K_ϕϕ_R    # K_ϕϕ at (i+1)-1/2 for cell i + 1

            end # End of interior i ∈ 2:(N_r-2) loop 

            # Boundary conditions at i = N_r + 1:
            if BC_r_right === :Reflect
                # Ghost cell i = N_r + 1:
                J_Np1   =   J[N_r, j]
                Hr_Np1  = - H_r[N_r, j]    # Reflect 
                HZ_Np1  =   H_Z[N_r, j]  
                # Ghost cell i = N_R + 2:
                J_Np2   =   J[N_r-1, j]
                Hr_Np2  = - H_r[N_r-1, j]  # Reflect 
                HZ_Np2  =   H_Z[N_r-1, j]  
            elseif BC_r_right === :Outflow
                # Ghost cell i = N_r + 1:
                J_Np1   =  J[N_r, j]
                Hr_Np1  =  max( H_r[N_r, j], 0.0 )
                HZ_Np1  =  H_Z[N_r, j]
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
            nZ_L, nZ_R = ReconstructInterface( J[N_r-1,j],   J[N_r,j],   J_Np1,    J[N_r-2,j],
                                               H_r[N_r-1,j], H_r[N_r,j], Hr_Np1,   H_r[N_r-2,j],
                                               H_Z[N_r-1,j], H_Z[N_r,j], HZ_Np1,   H_Z[N_r-2,j],

                                               Δr_L[N_r-1], Δr_R[N_r-1],
                                               inv_dr_i_im1[N_r-1],
                                               inv_dr_ip1_i[N_r-1],
                                               inv_dr_ip2_ip1[N_r-1],

                                               α_F[N_r-1,j], α_F[N_r,j] )

            # Compute interface D-values:
            D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            # K = D × J at the interface i = N_r - 1/2:
            Krr_L = D_rr_L * J_L
            Krr_R = D_rr_R * J_R
            KZZ_L = D_ZZ_L * J_L
            KZZ_R = D_ZZ_R * J_R 
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at interface i = N_r - 1/2:
            F_r[N_r,j], F_r_Hr[N_r,j], F_r_HZ[N_r,j] = GetFluxGLF( Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                                   J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                                   τ_L, τ_R, J[N_r-1,j], J[N_r,j] )

            # Add advection fluxes:
            u_contact_over_c    = u_r_star[N_r,j] * inv_c
            F_r[N_r,j]         += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_r_Hr[N_r,j]      += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_r_HZ[N_r,j]      += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

            # Get K_ϕϕ at interfaces:
            K_ϕϕ_L                = clamp( 1.0 - D_rr_L - D_ZZ_L, 0.0, 1.0 ) * J_L 
            K_ϕϕ_R                = clamp( 1.0 - D_rr_R - D_ZZ_R, 0.0, 1.0 ) * J_R 

            K_ϕϕ_iphalf[N_r-1,j]  = K_ϕϕ_L    # K_ϕϕ at i+1/2 for cell i = N_r - 1
            K_ϕϕ_imhalf[N_r,j]    = K_ϕϕ_R    # K_ϕϕ at i-1/2 for cell i = N_r 

            ##############################################################################
            ##############         INTERFACE i = N_r + 1/2         #######################
            ##############################################################################

            # Reconstruct variables at interface r_{N_r+1/2} (i.e. r[N_r+1]):
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
            τ_L, τ_R, f_L, f_R, nr_L, nr_R,
            nZ_L, nZ_R = ReconstructInterface( J[N_r,j],   J_Np1,    J_Np2,    J[N_r-1,j],
                                               H_r[N_r,j], Hr_Np1,   Hr_Np2,   H_r[N_r-1,j],
                                               H_Z[N_r,j], HZ_Np1,   HZ_Np2,   H_Z[N_r-1,j],

                                               Δr_L_OuterBoundary, Δr_R_OuterBoundary,
                                               inv_dr_i_im1_OuterBoundary,
                                               inv_dr_ip1_i_OuterBoundary,
                                               inv_dr_ip2_ip1_OuterBoundary,

                                               α_F[N_r,j], α_F[N_r,j]  # Assume opacity in ghost cell N_r+1 is same as for cell N_r
                                              )

            # Compute interface D-values:
            D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            # K = D × J at the interface i = N_r + 1/2:
            Krr_L = D_rr_L * J_L
            Krr_R = D_rr_R * J_R
            KZZ_L = D_ZZ_L * J_L 
            KZZ_R = D_ZZ_R * J_R 
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at interface i = N_r + 1/2:
            F_r[N_r+1,j], F_r_Hr[N_r+1,j], F_r_HZ[N_r+1,j] = GetFluxGLF( Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                                         J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                                         τ_L, τ_R, J[N_r,j], J_Np1 )

            # Add advection fluxes:
            u_contact_over_c   = u_r_star[N_r+1,j] * inv_c
            F_r[N_r+1,j]      += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_r_Hr[N_r+1,j]   += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_r_HZ[N_r+1,j]   += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

            # Get K_ϕϕ at interface i = N_r + 1/2:
            K_ϕϕ_L              = clamp( 1.0 - D_rr_L - D_ZZ_L, 0.0, 1.0 ) * J_L 
            K_ϕϕ_iphalf[N_r,j]  = K_ϕϕ_L    # K_ϕϕ at i+1/2 for cell i = N_r

        end # End of inbounds     
    end # End of j-loop 


    # Next, Z-direction fluxes:
    @threads for i in dN_r
        @inbounds begin
            ##############################################################################
            ####################         INTERFACE j = 1/2         #######################
            ##############################################################################

            if BC_Z_bot === :Reflect
                # Ghost cell j = 0:
                J_0   =   J[i,1]
                HZ_0  = - H_Z[i,1]   # Reflect
                Hr_0  =   H_r[i,1] 
                # Ghost cell j = -1 (mirror of cell j = 2):
                J_m1  =   J[i,2]
                HZ_m1 = - H_Z[i,2]   # Reflect
                Hr_m1 =   H_r[i,2] 
            elseif BC_Z_bot === :Outflow 
                # Ghost cell j = 0:
                J_0   =   J[i,1]
                HZ_0  =   min( H_Z[i,1], 0.0 )  
                Hr_0  =   H_r[i,1] 
                # Ghost cell j = -1:
                J_m1  =   J_0
                HZ_m1 =   HZ_0 
                Hr_m1 =   Hr_0 
            end

            # Reconstruct variables at interface Z_{1/2} (i.e. Z[1]):
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
            τ_L, τ_R, f_L, f_R, nr_L, nr_R,
            nZ_L, nZ_R = ReconstructInterface( J_0,    J[i,1],   J[i,2],   J_m1,
                                               Hr_0,   H_r[i,1], H_r[i,2], Hr_m1,
                                               HZ_0,   H_Z[i,1], H_Z[i,2], HZ_m1,

                                               ΔZ_L_InnerBoundary, ΔZ_R_InnerBoundary,
                                               inv_dZ_j_jm1_InnerBoundary,
                                               inv_dZ_jp1_j_InnerBoundary,
                                               inv_dZ_jp2_jp1_InnerBoundary,

                                               α_F[i,1], α_F[i,1] # Assume same opacity in ghost cell j = 0 as in j = 1
                                             )

            # Compute interface D-values:
            _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            # Compute K = D × J at the interface, with the right BC for KrZ_L:
            KZZ_L = D_ZZ_L * J_L
            KZZ_R = D_ZZ_R * J_R
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at interface j = 1/2:
            F_Z[i,1], F_Z_Hr[i,1], F_Z_HZ[i,1] = GetFluxGLF( HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                             J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                             τ_L, τ_R, J_0, J[i,1] )

            # Add advection fluxes:
            u_contact_over_c  = u_Z_star[i,1] * inv_c
            F_Z[i,1]         += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_Z_Hr[i,1]      += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_Z_HZ[i,1]      += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

            ##############################################################################
            ####################         INTERFACE j = 3/2         #######################
            ##############################################################################

            # Reconstruct variables at interface Z_{3/2} (i.e. Z[2]):
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
            τ_L, τ_R, f_L, f_R, nr_L, nr_R,
            nZ_L, nZ_R = ReconstructInterface( J[i,1],   J[i,2],   J[i,3],   J_0,
                                               H_r[i,1], H_r[i,2], H_r[i,3], Hr_0,
                                               H_Z[i,1], H_Z[i,2], H_Z[i,3], HZ_0,

                                               ΔZ_L[1], ΔZ_R[1],
                                               inv_dZ_j_jm1[1],
                                               inv_dZ_jp1_j[1],
                                               inv_dZ_jp2_jp1[1],

                                               α_F[i,1], α_F[i,2] )

            # Compute interface D-values:
            _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            # Compute K = D × J at the interface j = 3/2:
            KZZ_L = D_ZZ_L * J_L
            KZZ_R = D_ZZ_R * J_R
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at interface j = 3/2:
            F_Z[i,2], F_Z_Hr[i,2], F_Z_HZ[i,2] = GetFluxGLF( HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                             J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                             τ_L, τ_R, J[i,1], J[i,2] )
            
            # Add advection fluxes:
            u_contact_over_c  = u_Z_star[i,2] * inv_c
            F_Z[i,2]         += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_Z_Hr[i,2]      += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_Z_HZ[i,2]      += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

        end # End of inbounds 
    end # End of i-loop 


    @threads for j in dN_Zint
        @inbounds @simd for i in dN_r
            jp1 = j + 1
            jp2 = j + 2
            jm1 = j - 1

            # Reconstruct at Z_{j+1/2}
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, τ_L, τ_R, f_L, f_R, nr_L, nr_R,
            nZ_L, nZ_R =
            ReconstructInterface( J[i,j],   J[i,jp1],   J[i,jp2],   J[i,jm1],
                                  H_r[i,j], H_r[i,jp1], H_r[i,jp2], H_r[i,jm1],
                                  H_Z[i,j], H_Z[i,jp1], H_Z[i,jp2], H_Z[i,jm1],

                                  ΔZ_L[j], ΔZ_R[j],
                                  inv_dZ_j_jm1[j],
                                  inv_dZ_jp1_j[j],
                                  inv_dZ_jp2_jp1[j],

                                  α_F[i,j], α_F[i,jp1] )

            # Interface D and K
            _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            KZZ_L = D_ZZ_L * J_L;   KZZ_R = D_ZZ_R * J_R
            KrZ_L = D_rZ_L * J_L;   KrZ_R = D_rZ_R * J_R

            # Write interior Z-fluxes:
            F_Z[i,jp1], F_Z_Hr[i,jp1], F_Z_HZ[i,jp1] = GetFluxGLF( HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                                   J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                                   τ_L, τ_R, J[i,j], J[i,jp1] )

            # Add advection fluxes:
            u_contact_over_c   = u_Z_star[i,jp1] * inv_c
            F_Z[i,jp1]        += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_Z_Hr[i,jp1]     += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_Z_HZ[i,jp1]     += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

        end # End of inbounds 
    end # End of j-loop 

    @threads for i in dN_r
        @inbounds begin
            # Boundary conditions at interface j = N_Z + 1/2:
            if BC_Z_top === :Reflect
                # Ghost cell j = N_Z + 1:
                J_Np1   =   J[i,N_Z]
                HZ_Np1  = - H_Z[i,N_Z]    # Reflect
                Hr_Np1  =   H_r[i,N_Z] 
                # Ghost cell j = N_Z + 2 (mirror of cell j = N_Z - 1):
                J_Np2   =   J[i,N_Z-1]
                HZ_Np2  = - H_Z[i,N_Z-1]  # Reflect
                Hr_Np2  =   H_r[i,N_Z-1] 
            elseif BC_Z_top === :Outflow
                # Ghost cell j = N_Z + 1:
                J_Np1   =   J[i,N_Z]
                HZ_Np1  =   max( H_Z[i,N_Z], 0.0 )    
                Hr_Np1  =   H_r[i,N_Z]  
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
            nZ_L, nZ_R = ReconstructInterface( J[i,N_Z-1],   J[i,N_Z],   J_Np1,     J[i,N_Z-2],
                                               H_r[i,N_Z-1], H_r[i,N_Z], Hr_Np1,    H_r[i,N_Z-2],
                                               H_Z[i,N_Z-1], H_Z[i,N_Z], HZ_Np1,    H_Z[i,N_Z-2],

                                               ΔZ_L[N_Z-1], ΔZ_R[N_Z-1],
                                               inv_dZ_j_jm1[N_Z-1],
                                               inv_dZ_jp1_j[N_Z-1],
                                               inv_dZ_jp2_jp1[N_Z-1],

                                               α_F[i,N_Z-1], α_F[i,N_Z] )

            # Compute interface D-values:
            _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            # Compute K = D × J at the interface j = N_Z - 1/2:
            KZZ_L = D_ZZ_L * J_L
            KZZ_R = D_ZZ_R * J_R
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at interface j = N_Z - 1/2:
            F_Z[i,N_Z], F_Z_Hr[i,N_Z], F_Z_HZ[i,N_Z] = GetFluxGLF( HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                                   J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                                   τ_L, τ_R, J[i,N_Z-1], J[i,N_Z] )

            # Add advection fluxes:
            u_contact_over_c  = u_Z_star[i,N_Z] * inv_c
            F_Z[i,N_Z]       += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_Z_Hr[i,N_Z]    += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_Z_HZ[i,N_Z]    += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

            ##############################################################################
            ###############        INTERFACE j = N_Z + 1/2         #######################
            ##############################################################################

            # Reconstruct variables at interface Z_{N_Z+1/2} (i.e. Z[N_Z+1]):
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
            τ_L, τ_R, f_L, f_R, nr_L, nr_R,
            nZ_L, nZ_R = ReconstructInterface( J[i,N_Z],   J_Np1,     J_Np2,     J[i,N_Z-1],
                                               H_r[i,N_Z], Hr_Np1,    Hr_Np2,    H_r[i,N_Z-1],
                                               H_Z[i,N_Z], HZ_Np1,    HZ_Np2,    H_Z[i,N_Z-1],

                                               ΔZ_L_OuterBoundary, ΔZ_R_OuterBoundary,
                                               inv_dZ_j_jm1_OuterBoundary,
                                               inv_dZ_jp1_j_OuterBoundary,
                                               inv_dZ_jp2_jp1_OuterBoundary,

                                               α_F[i,N_Z], α_F[i,N_Z]  # Assume opacity in ghost cell j = N_Z+1 is same as in j = N_Z
                                              )

            # Compute interface D-values:
            _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            # Compute K = D × J at the interface j = N_Z + 1/2:
            KZZ_L = D_ZZ_L * J_L
            KZZ_R = D_ZZ_R * J_R
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at interface j = N_Z + 1/2:
            F_Z[i,N_Z+1], F_Z_Hr[i,N_Z+1], F_Z_HZ[i,N_Z+1] = GetFluxGLF( HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                                         J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                                         τ_L, τ_R, J[i,N_Z], J_Np1 )
            
            # Add advection fluxes:
            u_contact_over_c   = u_Z_star[i,N_Z+1] * inv_c
            F_Z[i,N_Z+1]      += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_Z_Hr[i,N_Z+1]   += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_Z_HZ[i,N_Z+1]   += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

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
        @inbounds @simd for i in dN_r
            # Store old J value, needed for the radiation pressure work term:
            J_stored = J[i,j]

            # Update J and H from the flux terms:
            J[i,j]   = max( UpdateConservativeQuantity( J[i,j], c̃Δt, inv_ΔV[i,j], 
                                                        A_r[i,j], A_r[i+1,j],
                                                        A_Z[i,j], A_Z[i,j+1], 
                                                        F_r[i,j], F_r[i+1,j],
                                                        F_Z[i,j], F_Z[i,j+1] ),
                            Jeps )

            H_Z[i,j] = UpdateConservativeQuantity( H_Z[i,j], c̃Δt, inv_ΔV[i,j],
                                                   A_r[i,j], A_r[i+1,j],
                                                   A_Z[i,j], A_Z[i,j+1],
                                                   F_r_HZ[i,j], F_r_HZ[i+1,j],
                                                   F_Z_HZ[i,j], F_Z_HZ[i,j+1] )

            H_r[i,j] = UpdateConservativeQuantity( H_r[i,j], c̃Δt, inv_ΔV[i,j],
                                                   A_r[i,j], A_r[i+1,j],
                                                   A_Z[i,j], A_Z[i,j+1],
                                                   F_r_Hr[i,j], F_r_Hr[i+1,j],
                                                   F_Z_Hr[i,j], F_Z_Hr[i,j+1] )

            # Get D:∇u, needed below, and also add contribution from geometric source 
            # term if using cylindrical coordinates:

            Drr = Float64(D_rr[i,j]);  DrZ = Float64(D_rZ[i,j]);  DZZ = Float64(D_ZZ[i,j])
            if CylindricalCoords
                D_colon_∇u = D_colon_∇u_Cylindrical( Drr, DrZ, DZZ, 
                                                     ∂u_r_∂r[i,j], ∂u_r_∂Z[i,j], ∂u_Z_∂r[i,j], ∂u_Z_∂Z[i,j],
                                                     u_r_over_r[i,j] )

                # If using cylindrical coordinates, add contribution from geometric source term,
                # (1/c̃) ∂H_r/∂t = K_ϕϕ/r ≈ 0.5 [(K_ϕϕ)_{i+1/2} + (K_ϕϕ)_{i-1/2}] ⟨1/r⟩ using
                # ⟨1/r⟩ ≡ ∫dV (1/r) / ΔV = 2π * Δr * ΔZ / ΔV = 1 / [( r[i+1] + r[i] ) / 2]:
                @fastmath K_ϕϕ_av    = 0.5 * ( K_ϕϕ_imhalf[i,j]  +  K_ϕϕ_iphalf[i,j] )
                @fastmath H_r[i,j]  += c̃Δt * K_ϕϕ_av * inv_r_cell[i]
            else
                D_colon_∇u = D_colon_∇u_Cartesian( Drr, DrZ, DZZ, 
                                                   ∂u_r_∂r[i,j], ∂u_r_∂Z[i,j], ∂u_Z_∂r[i,j], ∂u_Z_∂Z[i,j] )
            end

            # Finally, also add the contribution from (1/c̃) ∂J/∂t = - (D:∇u/c) J: 
            J[i,j] = max( J[i,j]  -  c̃Δt_over_c * D_colon_∇u * J_stored,  Jeps ) 

        end # End of i-loop 
    end # End of j-loop 


    # Update J_old if the option is turned on:
    if UpdateJold
        copyto!( J_old, J )
    end

    return nothing    
end


@inline function AdvectJandH_Serial!( c̃Δt::Float64,                   # c̃ × Δt 
                               α_F::AbstractArray{Float64, 2}, # Flux opacities              (N_r x N_Z)

                         J::AbstractArray{Float64, 2}, # J (latest/updated values)   (N_r x N_Z)
                     J_old::AbstractArray{Float64, 2}, # J (old values)              (N_r x N_Z)

                       H_r::AbstractArray{Float64, 2}, # r-direction component of H  (N_r x N_Z)
                       H_Z::AbstractArray{Float64, 2}, # Z-direction component of H  (N_r x N_Z)

                      D_rr::AbstractArray{TypeOfD, 2}, # rr-components of the Eddington tensor (N_r x N_Z)
                      D_ZZ::AbstractArray{TypeOfD, 2}, # ZZ-components of the Eddington tensor (N_r x N_Z)
                      D_rZ::AbstractArray{TypeOfD, 2}, # rZ-components of the Eddington tensor (N_r x N_Z)
               K_ϕϕ_iphalf::AbstractArray{Float64, 2}, # ϕϕ-components of K at i+1/2           (N_r x N_Z)
               K_ϕϕ_imhalf::AbstractArray{Float64, 2}, # ϕϕ-components of K at i-1/2           (N_r x N_Z)

                       F_r::AbstractArray{Float64, 2}, # r-direction flux of J               (N_r+1 x N_Z)
                       F_Z::AbstractArray{Float64, 2}, # Z-direction flux of J               (N_r   x N_Z+1)

                    F_r_Hr::AbstractArray{Float64, 2}, # r-direction flux of H_r             (N_r+1 x N_Z)
                    F_Z_Hr::AbstractArray{Float64, 2}, # Z-direction flux of H_r             (N_r   x N_Z+1)
                    F_r_HZ::AbstractArray{Float64, 2}, # r-direction flux of H_Z             (N_r+1 x N_Z)
                    F_Z_HZ::AbstractArray{Float64, 2}, # Z-direction flux of H_Z             (N_r   x N_Z+1)

                    inv_ΔV::AbstractArray{Float64, 2}, # Inverse cell volumes 1 / ΔV         (N_r   x N_Z)
                       A_r::AbstractArray{Float64, 2}, # Cell interface areas in r-direction (N_r+1 x N_Z)
                       A_Z::AbstractArray{Float64, 2}, # Cell interface areas in Z-direction (N_r   x N_Z+1)
                inv_r_cell::AbstractArray{Float64},    # ⟨1/r⟩ for every cell                (N_r)
                         r::AbstractArray{Float64},

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

         CylindricalCoords::Bool,         # Cylindrical coordinates assumed if true, otherwise Cartesian 
                 BC_r_left::Symbol,       # Boundary condition at i = 1/2 
                BC_r_right::Symbol,       # Boundary condition at i = N_r + 1/2
                  BC_Z_bot::Symbol,       # Boundary condition at j = 1/2 
                  BC_Z_top::Symbol        # Boundary condition at j = N_Z + 1/2 
                                   )  where {TypeOfD<:AbstractFloat}

    # Grid & intervals:
    N_r, N_Z        = size( J )
    dN_r            = 1:N_r
    dN_Z            = 1:N_Z
    dN_rint         = 2:(N_r-2)
    dN_Zint         = 2:(N_Z-2)

    # Convenient constant:
    c̃Δt_over_c      = c̃Δt * inv_c

    if DEBUG_M1Advection
        @assert size( u_r_star, 1 ) == N_r + 1 "Contact velocity in r-direction/X-direction must have size $(N_r+1)."
        @assert size( u_Z_star, 2 ) == N_Z + 1 "Contact velocity in Z-direction/Y-direction must have size $(N_Z+1)."
    end

    # First update the Eddington factors:
    Update_D_Edd_Serial!( D_rr, D_ZZ, D_rZ, H_r, H_Z, J )

    ###############################################################
    #######                                                 #######
    #######           C O M P U T E  F L U X E S            #######
    #######                                                 #######
    ###############################################################

    # We start by reconstructing & computing the interface fluxes, 
    # starting with the r-direction (order does not matter here).
    
    # r-direction fluxes:
    for j in dN_Z
        @inbounds begin
            ##############################################################################
            ##############          INTERFACE i = 1/2          ###########################
            ##############################################################################

            if BC_r_left === :Reflect 
                # Ghost cell i = 0:
                J_0    = J[1,j] 
                Hr_0   = - H_r[1,j]   # Reflect radial flux
                HZ_0   = H_Z[1,j]
                # Ghost cell i = -1:
                J_m1   = J[2,j] 
                Hr_m1  = - H_r[2,j]   # Reflect radial flux
                HZ_m1  = H_Z[2,j]
            elseif BC_r_left === :Outflow  
                # Ghost cell i = 0:
                J_0    = J[1,j] 
                Hr_0   = min( H_r[1,j], 0.0 )          
                HZ_0   = H_Z[1,j]
                # Ghost cell i = -1 (no gradients, so equal to cells i = 0 and 1):
                J_m1   = J_0 
                Hr_m1  = Hr_0   
                HZ_m1  = HZ_0
            end

            # Reconstruct variables at interface r_{1/2} (i.e. r[1]):
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
            τ_L, τ_R, f_L, f_R, nr_L, nr_R,
            nZ_L, nZ_R  = ReconstructInterface( J_0,     J[1,j],   J[2,j],   J_m1,
                                                Hr_0,    H_r[1,j], H_r[2,j], Hr_m1,
                                                HZ_0,    H_Z[1,j], H_Z[2,j], HZ_m1,

                                                Δr_L_InnerBoundary, Δr_R_InnerBoundary,
                                                inv_dr_i_im1_InnerBoundary,
                                                inv_dr_ip1_i_InnerBoundary,
                                                inv_dr_ip2_ip1_InnerBoundary,
                    
                                                α_F[1,j], α_F[1,j]  # Assume ghost cell 0 has same opacity as cell i = 1
                                               )

            # Compute interface D-values:
            D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            # Compute K = D × J at the interface:
            Krr_L = D_rr_L * J_L
            Krr_R = D_rr_R * J_R
            KZZ_L = D_ZZ_L * J_L
            KZZ_R = D_ZZ_R * J_R
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at i = 1/2:
            F_r[1,j], F_r_Hr[1,j], F_r_HZ[1,j] = GetFluxGLF( Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                             J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R,   
                                                             τ_L, τ_R, J_0, J[1,j] )

            # Add advection fluxes:
            u_contact_over_c    = u_r_star[1,j] * inv_c
            F_r[1,j]           += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_r_Hr[1,j]        += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_r_HZ[1,j]        += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

            # Fill K_ϕϕ at i = 1/2 for cell i = 1, needed for the geometric source term:
            K_ϕϕ_R           = clamp( 1.0 - D_rr_R - D_ZZ_R, 0.0, 1.0 ) * J_R
            K_ϕϕ_imhalf[1,j] = K_ϕϕ_R 

            ##############################################################################
            ##############          INTERFACE i = 3/2          ###########################
            ##############################################################################

            # Reconstruct variables at interface r_{3/2} (i.e. r[2]):
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
            τ_L, τ_R, f_L, f_R, nr_L, nr_R,
            nZ_L, nZ_R = ReconstructInterface( J[1,j],   J[2,j],   J[3,j],   J_0,
                                               H_r[1,j], H_r[2,j], H_r[3,j], Hr_0,
                                               H_Z[1,j], H_Z[2,j], H_Z[3,j], HZ_0,

                                               Δr_L[1], Δr_R[1],
                                               inv_dr_i_im1[1],
                                               inv_dr_ip1_i[1],
                                               inv_dr_ip2_ip1[1],

                                               α_F[1,j], α_F[2,j] )

            # Compute interface D-values:
            D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            # Compute K = D × J at the interface:
            Krr_L = D_rr_L * J_L 
            Krr_R = D_rr_R * J_R 
            KZZ_L = D_ZZ_L * J_L 
            KZZ_R = D_ZZ_R * J_R 
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at i = 3/2:
            F_r[2,j], F_r_Hr[2,j], F_r_HZ[2,j] = GetFluxGLF( Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                             J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                             τ_L, τ_R, J[1,j], J[2,j] )

            # Add advection fluxes:
            u_contact_over_c    = u_r_star[2,j] * inv_c
            F_r[2,j]           += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_r_Hr[2,j]        += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_r_HZ[2,j]        += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

            # Get K_ϕϕ at interfaces:
            K_ϕϕ_L           = clamp( 1.0 - D_rr_L - D_ZZ_L, 0.0, 1.0 ) * J_L 
            K_ϕϕ_R           = clamp( 1.0 - D_rr_R - D_ZZ_R, 0.0, 1.0 ) * J_R 

            K_ϕϕ_iphalf[1,j] = K_ϕϕ_L    # K_ϕϕ at i+1/2 for cell i = 1
            K_ϕϕ_imhalf[2,j] = K_ϕϕ_R    # K_ϕϕ at i-1/2 for cell i = 2

            ##############################################################################
            ##############        INTERIOR INTERFACES 2 ≤ i ≤ N_r-2          #############
            ##############################################################################
            @simd for i in dN_rint
                ip1 = i + 1
                ip2 = i + 2
                im1 = i - 1

                # Reconstruct variables at interface r_{i+1/2} (i.e. r[i+1]):
                J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                τ_L, τ_R, f_L, f_R, nr_L, nr_R,
                nZ_L, nZ_R = ReconstructInterface( J[i,j],   J[ip1,j],   J[ip2,j],   J[im1,j],
                                                   H_r[i,j], H_r[ip1,j], H_r[ip2,j], H_r[im1,j],
                                                   H_Z[i,j], H_Z[ip1,j], H_Z[ip2,j], H_Z[im1,j],

                                                   Δr_L[i], Δr_R[i],
                                                   inv_dr_i_im1[i],
                                                   inv_dr_ip1_i[i],
                                                   inv_dr_ip2_ip1[i],

                                                   α_F[i,j], α_F[ip1,j]  )

                # Compute interface D-values:
                D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
                D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

                # Compute K = D × J at the interface:
                Krr_L = D_rr_L * J_L 
                Krr_R = D_rr_R * J_R 
                KZZ_L = D_ZZ_L * J_L
                KZZ_R = D_ZZ_R * J_R
                KrZ_L = D_rZ_L * J_L
                KrZ_R = D_rZ_R * J_R

                # Radial flux of J, H_r, and H_Z:
                F_r[i+1,j], F_r_Hr[i+1,j], F_r_HZ[i+1,j] = GetFluxGLF( Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                                       J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                                       τ_L, τ_R, J[i,j], J[ip1,j] )

                # Add advection fluxes:
                u_contact_over_c     = u_r_star[i+1,j] * inv_c
                F_r[i+1,j]          += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
                F_r_Hr[i+1,j]       += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
                F_r_HZ[i+1,j]       += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

                # Get K_ϕϕ at interfaces:
                K_ϕϕ_L             = clamp( 1.0 - D_rr_L - D_ZZ_L, 0.0, 1.0 ) * J_L 
                K_ϕϕ_R             = clamp( 1.0 - D_rr_R - D_ZZ_R, 0.0, 1.0 ) * J_R 

                K_ϕϕ_iphalf[i,j]   = K_ϕϕ_L    # K_ϕϕ at i+1/2 for cell i
                K_ϕϕ_imhalf[i+1,j] = K_ϕϕ_R    # K_ϕϕ at (i+1)-1/2 for cell i + 1

            end # End of interior i ∈ 2:(N_r-2) loop 

            # Boundary conditions at i = N_r + 1:
            if BC_r_right === :Reflect
                # Ghost cell i = N_r + 1:
                J_Np1   =   J[N_r, j]
                Hr_Np1  = - H_r[N_r, j]    # Reflect 
                HZ_Np1  =   H_Z[N_r, j]  
                # Ghost cell i = N_R + 2:
                J_Np2   =   J[N_r-1, j]
                Hr_Np2  = - H_r[N_r-1, j]  # Reflect 
                HZ_Np2  =   H_Z[N_r-1, j]  
            elseif BC_r_right === :Outflow
                # Ghost cell i = N_r + 1:
                J_Np1   =  J[N_r, j]
                Hr_Np1  =  max( H_r[N_r, j], 0.0 )
                HZ_Np1  =  H_Z[N_r, j]
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
            nZ_L, nZ_R = ReconstructInterface( J[N_r-1,j],   J[N_r,j],   J_Np1,    J[N_r-2,j],
                                               H_r[N_r-1,j], H_r[N_r,j], Hr_Np1,   H_r[N_r-2,j],
                                               H_Z[N_r-1,j], H_Z[N_r,j], HZ_Np1,   H_Z[N_r-2,j],

                                               Δr_L[N_r-1], Δr_R[N_r-1],
                                               inv_dr_i_im1[N_r-1],
                                               inv_dr_ip1_i[N_r-1],
                                               inv_dr_ip2_ip1[N_r-1],

                                               α_F[N_r-1,j], α_F[N_r,j] )

            # Compute interface D-values:
            D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            # K = D × J at the interface i = N_r - 1/2:
            Krr_L = D_rr_L * J_L
            Krr_R = D_rr_R * J_R
            KZZ_L = D_ZZ_L * J_L
            KZZ_R = D_ZZ_R * J_R 
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at interface i = N_r - 1/2:
            F_r[N_r,j], F_r_Hr[N_r,j], F_r_HZ[N_r,j] = GetFluxGLF( Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                                   J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                                   τ_L, τ_R, J[N_r-1,j], J[N_r,j] )

            # Add advection fluxes:
            u_contact_over_c    = u_r_star[N_r,j] * inv_c
            F_r[N_r,j]         += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_r_Hr[N_r,j]      += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_r_HZ[N_r,j]      += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

            # Get K_ϕϕ at interfaces:
            K_ϕϕ_L                = clamp( 1.0 - D_rr_L - D_ZZ_L, 0.0, 1.0 ) * J_L 
            K_ϕϕ_R                = clamp( 1.0 - D_rr_R - D_ZZ_R, 0.0, 1.0 ) * J_R 

            K_ϕϕ_iphalf[N_r-1,j]  = K_ϕϕ_L    # K_ϕϕ at i+1/2 for cell i = N_r - 1
            K_ϕϕ_imhalf[N_r,j]    = K_ϕϕ_R    # K_ϕϕ at i-1/2 for cell i = N_r 

            ##############################################################################
            ##############         INTERFACE i = N_r + 1/2         #######################
            ##############################################################################

            # Reconstruct variables at interface r_{N_r+1/2} (i.e. r[N_r+1]):
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
            τ_L, τ_R, f_L, f_R, nr_L, nr_R,
            nZ_L, nZ_R = ReconstructInterface( J[N_r,j],   J_Np1,    J_Np2,    J[N_r-1,j],
                                               H_r[N_r,j], Hr_Np1,   Hr_Np2,   H_r[N_r-1,j],
                                               H_Z[N_r,j], HZ_Np1,   HZ_Np2,   H_Z[N_r-1,j],

                                               Δr_L_OuterBoundary, Δr_R_OuterBoundary,
                                               inv_dr_i_im1_OuterBoundary,
                                               inv_dr_ip1_i_OuterBoundary,
                                               inv_dr_ip2_ip1_OuterBoundary,

                                               α_F[N_r,j], α_F[N_r,j]  # Assume opacity in ghost cell N_r+1 is same as for cell N_r
                                              )

            # Compute interface D-values:
            D_rr_L, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            D_rr_R, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            # K = D × J at the interface i = N_r + 1/2:
            Krr_L = D_rr_L * J_L
            Krr_R = D_rr_R * J_R
            KZZ_L = D_ZZ_L * J_L 
            KZZ_R = D_ZZ_R * J_R 
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at interface i = N_r + 1/2:
            F_r[N_r+1,j], F_r_Hr[N_r+1,j], F_r_HZ[N_r+1,j] = GetFluxGLF( Hr_L, Hr_R,  Krr_L, Krr_R,  KrZ_L, KrZ_R,
                                                                         J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                                         τ_L, τ_R, J[N_r,j], J_Np1 )

            # Add advection fluxes:
            u_contact_over_c   = u_r_star[N_r+1,j] * inv_c
            F_r[N_r+1,j]      += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_r_Hr[N_r+1,j]   += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_r_HZ[N_r+1,j]   += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

            # Get K_ϕϕ at interface i = N_r + 1/2:
            K_ϕϕ_L              = clamp( 1.0 - D_rr_L - D_ZZ_L, 0.0, 1.0 ) * J_L 
            K_ϕϕ_iphalf[N_r,j]  = K_ϕϕ_L    # K_ϕϕ at i+1/2 for cell i = N_r

        end # End of inbounds     
    end # End of j-loop 


    # Next, Z-direction fluxes:
    for i in dN_r
        @inbounds begin
            ##############################################################################
            ####################         INTERFACE j = 1/2         #######################
            ##############################################################################

            if BC_Z_bot === :Reflect
                # Ghost cell j = 0:
                J_0   =   J[i,1]
                HZ_0  = - H_Z[i,1]   # Reflect
                Hr_0  =   H_r[i,1] 
                # Ghost cell j = -1 (mirror of cell j = 2):
                J_m1  =   J[i,2]
                HZ_m1 = - H_Z[i,2]   # Reflect
                Hr_m1 =   H_r[i,2] 
            elseif BC_Z_bot === :Outflow 
                # Ghost cell j = 0:
                J_0   =   J[i,1]
                HZ_0  =   min( H_Z[i,1], 0.0 )  
                Hr_0  =   H_r[i,1] 
                # Ghost cell j = -1:
                J_m1  =   J_0
                HZ_m1 =   HZ_0 
                Hr_m1 =   Hr_0 
            end

            # Reconstruct variables at interface Z_{1/2} (i.e. Z[1]):
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
            τ_L, τ_R, f_L, f_R, nr_L, nr_R,
            nZ_L, nZ_R = ReconstructInterface( J_0,    J[i,1],   J[i,2],   J_m1,
                                               Hr_0,   H_r[i,1], H_r[i,2], Hr_m1,
                                               HZ_0,   H_Z[i,1], H_Z[i,2], HZ_m1,

                                               ΔZ_L_InnerBoundary, ΔZ_R_InnerBoundary,
                                               inv_dZ_j_jm1_InnerBoundary,
                                               inv_dZ_jp1_j_InnerBoundary,
                                               inv_dZ_jp2_jp1_InnerBoundary,

                                               α_F[i,1], α_F[i,1] # Assume same opacity in ghost cell j = 0 as in j = 1
                                             )

            # Compute interface D-values:
            _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            # Compute K = D × J at the interface, with the right BC for KrZ_L:
            KZZ_L = D_ZZ_L * J_L
            KZZ_R = D_ZZ_R * J_R
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at interface j = 1/2:
            F_Z[i,1], F_Z_Hr[i,1], F_Z_HZ[i,1] = GetFluxGLF( HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                             J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                             τ_L, τ_R, J_0, J[i,1] )

            # Add advection fluxes:
            u_contact_over_c  = u_Z_star[i,1] * inv_c
            F_Z[i,1]         += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_Z_Hr[i,1]      += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_Z_HZ[i,1]      += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

            ##############################################################################
            ####################         INTERFACE j = 3/2         #######################
            ##############################################################################

            # Reconstruct variables at interface Z_{3/2} (i.e. Z[2]):
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
            τ_L, τ_R, f_L, f_R, nr_L, nr_R,
            nZ_L, nZ_R = ReconstructInterface( J[i,1],   J[i,2],   J[i,3],   J_0,
                                               H_r[i,1], H_r[i,2], H_r[i,3], Hr_0,
                                               H_Z[i,1], H_Z[i,2], H_Z[i,3], HZ_0,

                                               ΔZ_L[1], ΔZ_R[1],
                                               inv_dZ_j_jm1[1],
                                               inv_dZ_jp1_j[1],
                                               inv_dZ_jp2_jp1[1],

                                               α_F[i,1], α_F[i,2] )

            # Compute interface D-values:
            _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            # Compute K = D × J at the interface j = 3/2:
            KZZ_L = D_ZZ_L * J_L
            KZZ_R = D_ZZ_R * J_R
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at interface j = 3/2:
            F_Z[i,2], F_Z_Hr[i,2], F_Z_HZ[i,2] = GetFluxGLF( HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                             J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                             τ_L, τ_R, J[i,1], J[i,2] )
            
            # Add advection fluxes:
            u_contact_over_c  = u_Z_star[i,2] * inv_c
            F_Z[i,2]         += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_Z_Hr[i,2]      += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_Z_HZ[i,2]      += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

        end # End of inbounds 
    end # End of i-loop 


    for j in dN_Zint
        @inbounds @simd for i in dN_r
            jp1 = j + 1
            jp2 = j + 2
            jm1 = j - 1

            # Reconstruct at Z_{j+1/2}
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, τ_L, τ_R, f_L, f_R, nr_L, nr_R,
            nZ_L, nZ_R =
            ReconstructInterface( J[i,j],   J[i,jp1],   J[i,jp2],   J[i,jm1],
                                  H_r[i,j], H_r[i,jp1], H_r[i,jp2], H_r[i,jm1],
                                  H_Z[i,j], H_Z[i,jp1], H_Z[i,jp2], H_Z[i,jm1],

                                  ΔZ_L[j], ΔZ_R[j],
                                  inv_dZ_j_jm1[j],
                                  inv_dZ_jp1_j[j],
                                  inv_dZ_jp2_jp1[j],

                                  α_F[i,j], α_F[i,jp1] )

            # Interface D and K
            _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            KZZ_L = D_ZZ_L * J_L;   KZZ_R = D_ZZ_R * J_R
            KrZ_L = D_rZ_L * J_L;   KrZ_R = D_rZ_R * J_R

            # Write interior Z-fluxes:
            F_Z[i,jp1], F_Z_Hr[i,jp1], F_Z_HZ[i,jp1] = GetFluxGLF( HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                                   J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                                   τ_L, τ_R, J[i,j], J[i,jp1] )

            # Add advection fluxes:
            u_contact_over_c   = u_Z_star[i,jp1] * inv_c
            F_Z[i,jp1]        += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_Z_Hr[i,jp1]     += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_Z_HZ[i,jp1]     += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

        end # End of inbounds 
    end # End of j-loop 

    for i in dN_r
        @inbounds begin
            # Boundary conditions at interface j = N_Z + 1/2:
            if BC_Z_top === :Reflect
                # Ghost cell j = N_Z + 1:
                J_Np1   =   J[i,N_Z]
                HZ_Np1  = - H_Z[i,N_Z]    # Reflect
                Hr_Np1  =   H_r[i,N_Z] 
                # Ghost cell j = N_Z + 2 (mirror of cell j = N_Z - 1):
                J_Np2   =   J[i,N_Z-1]
                HZ_Np2  = - H_Z[i,N_Z-1]  # Reflect
                Hr_Np2  =   H_r[i,N_Z-1] 
            elseif BC_Z_top === :Outflow
                # Ghost cell j = N_Z + 1:
                J_Np1   =   J[i,N_Z]
                HZ_Np1  =   max( H_Z[i,N_Z], 0.0 )    
                Hr_Np1  =   H_r[i,N_Z]  
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
            nZ_L, nZ_R = ReconstructInterface( J[i,N_Z-1],   J[i,N_Z],   J_Np1,     J[i,N_Z-2],
                                               H_r[i,N_Z-1], H_r[i,N_Z], Hr_Np1,    H_r[i,N_Z-2],
                                               H_Z[i,N_Z-1], H_Z[i,N_Z], HZ_Np1,    H_Z[i,N_Z-2],

                                               ΔZ_L[N_Z-1], ΔZ_R[N_Z-1],
                                               inv_dZ_j_jm1[N_Z-1],
                                               inv_dZ_jp1_j[N_Z-1],
                                               inv_dZ_jp2_jp1[N_Z-1],

                                               α_F[i,N_Z-1], α_F[i,N_Z] )

            # Compute interface D-values:
            _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            # Compute K = D × J at the interface j = N_Z - 1/2:
            KZZ_L = D_ZZ_L * J_L
            KZZ_R = D_ZZ_R * J_R
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at interface j = N_Z - 1/2:
            F_Z[i,N_Z], F_Z_Hr[i,N_Z], F_Z_HZ[i,N_Z] = GetFluxGLF( HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                                   J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                                   τ_L, τ_R, J[i,N_Z-1], J[i,N_Z] )

            # Add advection fluxes:
            u_contact_over_c  = u_Z_star[i,N_Z] * inv_c
            F_Z[i,N_Z]       += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_Z_Hr[i,N_Z]    += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_Z_HZ[i,N_Z]    += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

            ##############################################################################
            ###############        INTERFACE j = N_Z + 1/2         #######################
            ##############################################################################

            # Reconstruct variables at interface Z_{N_Z+1/2} (i.e. Z[N_Z+1]):
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
            τ_L, τ_R, f_L, f_R, nr_L, nr_R,
            nZ_L, nZ_R = ReconstructInterface( J[i,N_Z],   J_Np1,     J_Np2,     J[i,N_Z-1],
                                               H_r[i,N_Z], Hr_Np1,    Hr_Np2,    H_r[i,N_Z-1],
                                               H_Z[i,N_Z], HZ_Np1,    HZ_Np2,    H_Z[i,N_Z-1],

                                               ΔZ_L_OuterBoundary, ΔZ_R_OuterBoundary,
                                               inv_dZ_j_jm1_OuterBoundary,
                                               inv_dZ_jp1_j_OuterBoundary,
                                               inv_dZ_jp2_jp1_OuterBoundary,

                                               α_F[i,N_Z], α_F[i,N_Z]  # Assume opacity in ghost cell j = N_Z+1 is same as in j = N_Z
                                              )

            # Compute interface D-values:
            _, D_ZZ_L, D_rZ_L = Compute_D_Edd_from_f_n( f_L, nr_L, nZ_L )
            _, D_ZZ_R, D_rZ_R = Compute_D_Edd_from_f_n( f_R, nr_R, nZ_R )

            # Compute K = D × J at the interface j = N_Z + 1/2:
            KZZ_L = D_ZZ_L * J_L
            KZZ_R = D_ZZ_R * J_R
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at interface j = N_Z + 1/2:
            F_Z[i,N_Z+1], F_Z_Hr[i,N_Z+1], F_Z_HZ[i,N_Z+1] = GetFluxGLF( HZ_L, HZ_R,  KrZ_L, KrZ_R,  KZZ_L, KZZ_R,
                                                                         J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
                                                                         τ_L, τ_R, J[i,N_Z], J_Np1 )
            
            # Add advection fluxes:
            u_contact_over_c   = u_Z_star[i,N_Z+1] * inv_c
            F_Z[i,N_Z+1]      += GetAdvectionFlux( u_contact_over_c, J_L, J_R )
            F_Z_Hr[i,N_Z+1]   += GetAdvectionFlux( u_contact_over_c, Hr_L, Hr_R )
            F_Z_HZ[i,N_Z+1]   += GetAdvectionFlux( u_contact_over_c, HZ_L, HZ_R )

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
    for j in dN_Z
        @inbounds @simd for i in dN_r
            # Store old J value, needed for the radiation pressure work term:
            J_stored = J[i,j]

            # Update J and H from the flux terms:
            J[i,j]   = max( UpdateConservativeQuantity( J[i,j], c̃Δt, inv_ΔV[i,j], 
                                                        A_r[i,j], A_r[i+1,j],
                                                        A_Z[i,j], A_Z[i,j+1], 
                                                        F_r[i,j], F_r[i+1,j],
                                                        F_Z[i,j], F_Z[i,j+1] ),
                            Jeps )

            H_Z[i,j] = UpdateConservativeQuantity( H_Z[i,j], c̃Δt, inv_ΔV[i,j],
                                                   A_r[i,j], A_r[i+1,j],
                                                   A_Z[i,j], A_Z[i,j+1],
                                                   F_r_HZ[i,j], F_r_HZ[i+1,j],
                                                   F_Z_HZ[i,j], F_Z_HZ[i,j+1] )

            H_r[i,j] = UpdateConservativeQuantity( H_r[i,j], c̃Δt, inv_ΔV[i,j],
                                                   A_r[i,j], A_r[i+1,j],
                                                   A_Z[i,j], A_Z[i,j+1],
                                                   F_r_Hr[i,j], F_r_Hr[i+1,j],
                                                   F_Z_Hr[i,j], F_Z_Hr[i,j+1] )

            # Get D:∇u, needed below, and also add contribution from geometric source 
            # term if using cylindrical coordinates:

            Drr = Float64(D_rr[i,j]);  DrZ = Float64(D_rZ[i,j]);  DZZ = Float64(D_ZZ[i,j])
            if CylindricalCoords
                D_colon_∇u = D_colon_∇u_Cylindrical( Drr, DrZ, DZZ, 
                                                     ∂u_r_∂r[i,j], ∂u_r_∂Z[i,j], ∂u_Z_∂r[i,j], ∂u_Z_∂Z[i,j],
                                                     u_r_over_r[i,j] )

                # If using cylindrical coordinates, add contribution from geometric source term,
                # (1/c̃) ∂H_r/∂t = K_ϕϕ/r ≈ 0.5 [(K_ϕϕ)_{i+1/2} + (K_ϕϕ)_{i-1/2}] ⟨1/r⟩ using
                # ⟨1/r⟩ ≡ ∫dV (1/r) / ΔV = 2π * Δr * ΔZ / ΔV = 1 / [( r[i+1] + r[i] ) / 2]:
                @fastmath K_ϕϕ_av    = 0.5 * ( K_ϕϕ_imhalf[i,j]  +  K_ϕϕ_iphalf[i,j] )
                @fastmath H_r[i,j]  += c̃Δt * K_ϕϕ_av * inv_r_cell[i]
            else
                D_colon_∇u = D_colon_∇u_Cartesian( Drr, DrZ, DZZ, 
                                                   ∂u_r_∂r[i,j], ∂u_r_∂Z[i,j], ∂u_Z_∂r[i,j], ∂u_Z_∂Z[i,j] )
            end

            # Finally, also add the contribution from (1/c̃) ∂J/∂t = - (D:∇u/c) J: 
            J[i,j] = max( J[i,j]  -  c̃Δt_over_c * D_colon_∇u * J_stored,  Jeps ) 

        end # End of i-loop 
    end # End of j-loop 


    # Update J_old if the option is turned on:
    if UpdateJold
        copyto!( J_old, J )
    end

    return nothing    
end


const ThreadsOrNot = Union{Val{:Threaded}, Val{:Serial}}

function AdvectJandH!( c̃Δt::Float64,                   # c̃ × Δt 
                       α_F::AbstractArray{Float64, 2}, # Flux opacities              (N_r x N_Z)

                         J::AbstractArray{Float64, 2}, # J (latest/updated values)   (N_r x N_Z)
                     J_old::AbstractArray{Float64, 2}, # J (old values)              (N_r x N_Z)

                       H_r::AbstractArray{Float64, 2}, # r-direction component of H  (N_r x N_Z)
                       H_Z::AbstractArray{Float64, 2}, # Z-direction component of H  (N_r x N_Z)

                      D_rr::AbstractArray{TypeOfD, 2}, # rr-components of the Eddington tensor (N_r x N_Z)
                      D_ZZ::AbstractArray{TypeOfD, 2}, # ZZ-components of the Eddington tensor (N_r x N_Z)
                      D_rZ::AbstractArray{TypeOfD, 2}, # rZ-components of the Eddington tensor (N_r x N_Z)
               K_ϕϕ_iphalf::AbstractArray{Float64, 2}, # ϕϕ-components of K at i+1/2           (N_r x N_Z)
               K_ϕϕ_imhalf::AbstractArray{Float64, 2}, # ϕϕ-components of K at i-1/2           (N_r x N_Z)

                       F_r::AbstractArray{Float64, 2}, # r-direction flux of J               (N_r+1 x N_Z)
                       F_Z::AbstractArray{Float64, 2}, # Z-direction flux of J               (N_r   x N_Z+1)

                    F_r_Hr::AbstractArray{Float64, 2}, # r-direction flux of H_r             (N_r+1 x N_Z)
                    F_Z_Hr::AbstractArray{Float64, 2}, # Z-direction flux of H_r             (N_r   x N_Z+1)
                    F_r_HZ::AbstractArray{Float64, 2}, # r-direction flux of H_Z             (N_r+1 x N_Z)
                    F_Z_HZ::AbstractArray{Float64, 2}, # Z-direction flux of H_Z             (N_r   x N_Z+1)

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
                 Threading::ThreadsOrNot = Val(:Threaded)
                                   )  where {TypeOfD<:AbstractFloat}

    # Needed geoetry (intervals, areas etc):
    inv_ΔV     = geometry.inv_ΔV
    inv_r_cell = geometry.inv_r_cell  # Same signature for both cylindrical & Cartesian, but only used in former

    if geometry isa GeometryCylindrical
        # We have cylindrical coordinates:
        CylindricalCoords = true 

        r          = geometry.r

        # Interface areas:
        A_r        = geometry.A_r 
        A_Z        = geometry.A_Z

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
    elseif geometry isa GeometryCartesian
        # We don't have cylindrical coordinates (we have Cartesian):
        CylindricalCoords = false 

        r          = geometry.X

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
    end

    # Call M1 advection:
    if Threading isa Val{:Threaded}
        AdvectJandH_Threaded!(         c̃Δt,           α_F,        
                          J,             J_old, 
                          H_r,           H_Z, 
                          D_rr,          D_ZZ,        D_rZ,  # Eddington factors D = K / J
                          K_ϕϕ_iphalf,   K_ϕϕ_imhalf,
                          F_r,           F_Z,                     # Fluxes of J 
                          F_r_Hr,        F_Z_Hr,                  # Fluxes of H_r 
                          F_r_HZ,        F_Z_HZ,                  # Fluxes of H_Z 
                          inv_ΔV,  A_r,  A_Z, inv_r_cell, r,
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
                          CylindricalCoords,
                          BC_r_left,  BC_r_right,  BC_Z_bot,  BC_Z_top )

    else
        AdvectJandH_Serial!(         c̃Δt,           α_F,        
                          J,             J_old, 
                          H_r,           H_Z, 
                          D_rr,          D_ZZ,        D_rZ,  # Eddington factors D = K / J
                          K_ϕϕ_iphalf,   K_ϕϕ_imhalf,
                          F_r,           F_Z,                     # Fluxes of J 
                          F_r_Hr,        F_Z_Hr,                  # Fluxes of H_r 
                          F_r_HZ,        F_Z_HZ,                  # Fluxes of H_Z 
                          inv_ΔV,  A_r,  A_Z, inv_r_cell, r,
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
                          CylindricalCoords,
                          BC_r_left,  BC_r_right,  BC_Z_bot,  BC_Z_top )
    end

    return nothing 
end

