
""" MODULE:      AdvectJandH

    PURPOSE:     Contains the code for advecting the mean intensity J, and the
                 first moment H (the Eddington tensor D = K / J is estimated 
                 using an M1 closure). Uses an explicit Godunov method which is 
                 first-order in space and time.

    REFERENCES:  (1) Skinner & Ostriker (2013) 'A Two-moment Radiation Hydrodynamics 
                     Module in Athena Using a Time-explicit Godunov Method', ApJS, 206, 2

"""


#include( "./M1Closure.jl" )
#using .M1Closure        # For the M1 closure 
using Base.Threads      # For CPU acceleration

# Export functions:
export GetRHDCFLTimeStep, AdvectJandH!, AdvectJandH_Serial!


""" The default RHD CFL time-step. Computes min( Δr / c̃ ),
    over all cells. To avoid unnecessary allocations, this is 
    done in a for loop. """
@inline function GetRHDCFLTimeStep( r::AbstractVector{Float64}, # Interface r-values
                                    Z::AbstractVector{Float64}, # Interface Z-values
                                    c̃::Float64,                 # Reduced speed of light,
                                  CFL::Float64
                                   )

    N_r         = length(r) - 1  # Grid size
    N_Z         = length(Z) - 1
    dN_r        = 1:N_r
    dN_Z        = 1:N_Z
    inv_c̃       = 1 / c̃

    min_ΔtRHD   = Inf64  # Initialize the minimum local time-scale

    @inbounds for j in dN_Z
        ΔZ                = Z[j+1] - Z[j] # Vertical cell width
        local_min_ΔtRHD   = min_ΔtRHD     # Local variable for tracking smallest CFL time

        @simd for i in dN_r
            # Compute the horizontal cell width (same for all j-indices):
            Δr = r[i+1] - r[i]

            # Compute the local CFL time-scale:
            Δt = @fastmath min( Δr, ΔZ ) * inv_c̃
            
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


""" MinMod slope limiter. """
@inline @fastmath function MinMod( A::Float64,
                                   B::Float64 )

    0.5 * ( sign(A) + sign(B) ) * min( abs(A), abs(B) )
end


@inline @fastmath function GetFluxHLL( F_L::Float64,
                                       F_R::Float64,
                                       H_L::Float64,
                                       H_R::Float64,
                                      λ1_L::Float64, 
                                      λ1_R::Float64,
                                      λ3_L::Float64, 
                                      λ3_R::Float64,
                                       τ_L::Float64,
                                       τ_R::Float64
                                       )

    # Wavespeeds (in units of c̃) from eigenvalues:
    S_R     = max( max( λ3_L, λ3_R ), 0.0 )
    S_L     = min( min( λ1_L, λ1_R ), 0.0 )

    # Effective cell optical depth:
    τ_cell  = τ_L + τ_R

    # Optical depth correction:
    ϵ       = min( 1, 1 / τ_cell )
    #ϵ       = tanh( 1 / τ_cell )
    #ϵ       = 1 / ( 1 + τ_cell )
    #ϵ       = 1
    ϵ_sq    = ϵ * ϵ

    # Total flux valid in all regimes:
    #F      = ( ( ϵ_sq * ( S_R * F_L  -  S_L * F_R )  +  ϵ * S_R * S_L * ( H_R - H_L ) ) / ( S_R - S_L )
    #
    #       +   max( (1 - ϵ_sq), 0.0 ) * 0.5 * ( F_L  +  F_R ) )

    F      = ( ( S_R * F_L  -  S_L * F_R )  +  S_R * S_L * ( H_R - H_L ) ) / ( S_R - S_L )
    
    return F
end


""" The HLL flux for J, corrected so that the asymptotic 
    diffusion limit is obtained when the cell 
    optical depth is ≳ 1. """
@inline @fastmath function GetFluxHLL_J( H_L::Float64,
                                         H_R::Float64,
                                         J_L::Float64,
                                         J_R::Float64,
                                        λ1_L::Float64, 
                                        λ1_R::Float64,
                                        λ3_L::Float64, 
                                        λ3_R::Float64,
                                         τ_L::Float64,
                                         τ_R::Float64,
                                       J_ip1::Float64,
                                         J_i::Float64
                                         )

    # Effective cell optical depth:
    τ_cell  = τ_L + τ_R

    # Wavespeeds (in units of c̃) from eigenvalues:
    S_R     = max( max( λ3_L, λ3_R ), 0.0 )
    S_L     = min( min( λ1_L, λ1_R ), 0.0 )

    # Optical depth correction:
    ϵ      = min( 1, 1 / τ_cell )
    #ϵ       = tanh( 1 / τ_cell )
    #ϵ      = 1 / ( 1  +  τ_cell )
    #ϵ       = 1

    # Total flux valid in all regimes:
    F_J    = ( S_R * H_L  -  S_L * H_R  +  ϵ * S_R * S_L * ( J_R - J_L ) ) / ( S_R - S_L )
    
    return F_J
end


""" Solves the PDE (1/c̃) (∂Q/∂t) + ∇⋅F = 0 over a time-step Δt,
    using explicit time-integration, and the finite-volume 
    method. 
    
    **ARGUMENTS:**
    
    Q       :   Value of Q at "old" time-step n
    Δw      :   c̃ × Δt 
    inv_ΔV  :   Inverse volume 1 / ΔV of cell i 
    A_L     :   Area of left interface i - 1/2
    A_R     :   Area of right interface i + 1/2
    F_L     :   Flux F at left interface i - 1/2
    F_R     :   Flux F at right interface i + 1/2

    **RETURNS:**

    Qnew    :   Value of Q at time-step n + 1 
"""
@inline function UpdateConservativeQuantity( Q::Float64, 
                                           c̃Δt::Float64,
                                        inv_ΔV::Float64,
                                           A_L::Float64,
                                           A_R::Float64,
                                           F_L::Float64,
                                           F_R::Float64
                                             )

    Qnew = Q  -  ( c̃Δt * inv_ΔV ) * ( A_R * F_R  -  A_L * F_L )
    return Qnew
end


@inline function ReconstructInterface( J_i::Float64,   J_ip1::Float64,  J_ip2::Float64,  J_im1::Float64,
                                       Hr_i::Float64,  Hr_ip1::Float64, Hr_ip2::Float64, Hr_im1::Float64,
                                       HZ_i::Float64,  HZ_ip1::Float64, HZ_ip2::Float64, HZ_im1::Float64,
 
                                       rCoM_i::Float64, rCoM_ip1::Float64, rCoM_ip2::Float64, rCoM_im1::Float64, 
                                       r_ip1::Float64,
                              
                                       α_F_i::Float64, α_F_ip1::Float64 )

    # Intervals for the derivatives: 
    dr_i_im1   = rCoM_i   - rCoM_im1
    dr_ip1_i   = rCoM_ip1 - rCoM_i
    dr_ip2_ip1 = rCoM_ip2 - rCoM_ip1 

    # Intervals between cell centers & inteface:
    Δr_L       = r_ip1    - rCoM_i 
    Δr_R       = rCoM_ip1 - r_ip1 

    # (Flux) optical depth between cell centers & inteface:
    τ_L        = α_F_i   * Δr_L
    τ_R        = α_F_ip1 * Δr_R

    # Compute reduced fluxes:
    Hmag_i   = sqrt( Hr_i * Hr_i  +  HZ_i * HZ_i )          # Magnitude of H in cell i (will be re-used)
    Hmag_ip1 = sqrt( Hr_ip1 * Hr_ip1  +  HZ_ip1 * HZ_ip1 )  # Magnitude of H in cell i+1 (will be re-used)

    f_i      = Hmag_i   / max( J_i, Jeps ) 
    f_ip1    = Hmag_ip1 / max( J_ip1, Jeps )
    f_ip2    = sqrt( Hr_ip2 * Hr_ip2  +  HZ_ip2 * HZ_ip2 ) / max( J_ip2, Jeps )
    f_im1    = sqrt( Hr_im1 * Hr_im1  +  HZ_im1 * HZ_im1 ) / max( J_im1, Jeps )

    # Compute flux unit vectors in cells i and i+1:
    if Hmag_i > Heps
        inv_Hmag_i   =  1 / Hmag_i
        nr_i         =  Hr_i * inv_Hmag_i
        nZ_i         =  HZ_i * inv_Hmag_i
    else
        nr_i         =  0.0
        nZ_i         =  0.0
    end

    if Hmag_ip1 > Heps 
        inv_Hmag_ip1 =  1 / Hmag_ip1
        nr_ip1       =  Hr_ip1 * inv_Hmag_ip1
        nZ_ip1       =  HZ_ip1 * inv_Hmag_ip1
    else
        nr_ip1       =  0.0
        nZ_ip1       =  0.0
    end

    # Compute slope-limited derivatives for J:
    ∂J∂r_L   = MinMod( (J_i   - J_im1)   / dr_i_im1,
                       (J_ip1 - J_i)     / dr_ip1_i )
     
    ∂J∂r_R   = MinMod( (J_ip2 - J_ip1)   / dr_ip2_ip1,
                       (J_ip1 - J_i)     / dr_ip1_i )

    # Compute slope-limited derivatives for f:
    ∂f∂r_L   = MinMod( (f_i   - f_im1) / dr_i_im1,
                       (f_ip1 - f_i)   / dr_ip1_i )
     
    ∂f∂r_R   = MinMod( (f_ip2 - f_ip1) / dr_ip2_ip1,
                       (f_ip1 - f_i)   / dr_ip1_i )

    # Get all J and f (reduced flux) interface variables:
    J_L      = max( J_i    +  ∂J∂r_L * Δr_L, Jeps )
    J_R      = max( J_ip1  -  ∂J∂r_R * Δr_R, Jeps )

    f_L      = clamp( f_i     +  ∂f∂r_L * Δr_L,  0.0, 1.0 )
    f_R      = clamp( f_ip1   -  ∂f∂r_R * Δr_R,  0.0, 1.0 )

    # Compute H_r and H_Z from the interface J and f values,
    # and preserving the flux directions (via nr and nZ):
    Hr_L     = f_L * nr_i   * J_L
    Hr_R     = f_R * nr_ip1 * J_R

    HZ_L     = f_L * nZ_i   * J_L
    HZ_R     = f_R * nZ_ip1 * J_R
    
    return J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, τ_L, τ_R
end



@fastmath function AdvectJandH_Serial!( c̃Δt::Float64,                   # c̃ × Δt 
                                        α_F::AbstractArray{Float64, 2}, # Flux opacities              (N_r x N_Z)

                                          J::AbstractArray{Float64, 2}, # J (latest/updated values)   (N_r x N_Z)
                                      J_old::AbstractArray{Float64, 2}, # J (old values)              (N_r x N_Z)

                                        H_r::AbstractArray{Float64, 2}, # r-direction component of H  (N_r x N_Z)
                                        H_Z::AbstractArray{Float64, 2}, # Z-direction component of H  (N_r x N_Z)

                                       D_rr::AbstractArray{Float64, 2}, # rr-components of the Eddington tensor (N_r x N_Z)
                                       D_ZZ::AbstractArray{Float64, 2}, # ZZ-components of the Eddington tensor (N_r x N_Z)
                                       D_rZ::AbstractArray{Float64, 2}, # rZ-components of the Eddington tensor (N_r x N_Z)

                                        F_r::AbstractArray{Float64, 2}, # r-direction flux of J               (N_r+1 x N_Z)
                                        F_Z::AbstractArray{Float64, 2}, # Z-direction flux of J               (N_r   x N_Z+1)

                                     F_r_Hr::AbstractArray{Float64, 2}, # r-direction flux of H_r             (N_r+1 x N_Z)
                                     F_Z_Hr::AbstractArray{Float64, 2}, # Z-direction flux of H_r             (N_r   x N_Z+1)
                                     F_r_HZ::AbstractArray{Float64, 2}, # r-direction flux of H_Z             (N_r+1 x N_Z)
                                     F_Z_HZ::AbstractArray{Float64, 2}, # Z-direction flux of H_Z             (N_r   x N_Z+1)

                                     inv_ΔV::AbstractArray{Float64, 2}, # Inverse cell volumes 1 / ΔV         (N_r   x N_Z)
                                        A_r::AbstractArray{Float64, 2}, # Cell interface areas in r-direction (N_r+1 x N_Z)
                                        A_Z::AbstractArray{Float64, 2}, # Cell interface areas in Z-direction (N_r   x N_Z+1)
                                          r::AbstractArray{Float64},    # r-interface values                  (N_r+1)
                                       rCoM::AbstractArray{Float64},    # r-center values                     (N_r)
                                          Z::AbstractArray{Float64},    # Z-interface values                  (N_Z+1)
                                       ZCoM::AbstractArray{Float64},    # Z-center values                     (N_Z) 
                                 inv_r_cell::AbstractArray{Float64},    # ⟨1/r⟩ for every cell                (N_r)

                          CylindricalCoords::Bool     = true,           # Cylindrical coordinates assumed if true, otherwise Cartesian 
                                  BC_r_left::Symbol   = :Reflect,       # Boundary condition at i = 1/2 
                                 BC_r_right::Symbol   = :Outflow,       # Boundary condition at i = N_r + 1/2
                                   BC_Z_bot::Symbol   = :Reflect,       # Boundary condition at j = 1/2 
                                   BC_Z_top::Symbol   = :Outflow        # Boundary condition at j = N_Z + 1/2
                                       )

    N_r, N_Z = size( J )
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z
    dN_rint  = 2:(N_r-2)
    dN_Zint  = 2:(N_Z-2)
    c̃Δt_half = c̃Δt / 2

    # Needed ghost cell CoM radii, obtained by mirroring:
    @inbounds begin 
        # Ghost CoM r-positions mirrored around r[1] = r_{1/2}:
        rCoM_0   = r[1] - (rCoM[1] - r[1]) 
        rCoM_m1  = r[1] - (rCoM[2] - r[1])

        # Ghost CoM r-positions mirrored around r[N_r+1]:
        rCoM_Np1 = r[N_r+1] + (r[N_r+1] - rCoM[N_r])
        rCoM_Np2 = r[N_r+1] + (r[N_r+1] - rCoM[N_r-1])

        # Ghost CoM Z-positions mirrored around Z[1] = Z_{1/2}:
        ZCoM_0   = Z[1] - (ZCoM[1] - Z[1])
        ZCoM_m1  = Z[1] - (ZCoM[2] - Z[1])

        # Ghost CoM Z-positions mirrored around Z[N_Z+1] = Z_{N_Z+1/2}:
        ZCoM_Np1  = Z[N_Z+1] - (ZCoM[N_Z] - Z[N_Z+1])
        ZCoM_Np2  = Z[N_Z+1] - (ZCoM[N_Z-1] - Z[N_Z+1])
    end

    ################################################################################
    ############                                     ###############################
    ############  1. FIRST HALF-STEP IN r-DIRECTION  ###############################
    ############                                     ###############################
    ################################################################################

    # First update the Eddington factors:
    Update_D_Edd_Serial!( D_rr, D_ZZ, D_rZ, H_r, H_Z, J )

    @inbounds for j in dN_Z
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
        else
            # Ghost cell i = 0:
            J_0    = J[1,j] 
            Hr_0   = H_r[1,j]           
            HZ_0   = H_Z[1,j]
            # Ghost cell i = -1 (no gradients, so equal to cells i = 0 and 1):
            J_m1   = J_0 
            Hr_m1  = Hr_0   
            HZ_m1  = HZ_0
        end

        # Reconstruct variables at interface r_{1/2} (i.e. r[1]):
        J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
        τ_L, τ_R = ReconstructInterface( J_0,     J[1,j],   J[2,j],   J_m1,
                                         Hr_0,    H_r[1,j], H_r[2,j], Hr_m1,
                                         HZ_0,    H_Z[1,j], H_Z[2,j], HZ_m1,
                                         rCoM_0,  rCoM[1],  rCoM[2],  rCoM_m1, 
                                         r[1],               # Interface r_{1/2}
                                         α_F[1,j], α_F[1,j]  # Assume ghost cell 0 has same opacity as cell i = 1
                                          )

        # Compute interface D-values:
        D_rr_L, _, D_rZ_L = Compute_D_Edd( Hr_L, HZ_L, J_L )
        D_rr_R, _, D_rZ_R = Compute_D_Edd( Hr_R, HZ_R, J_R )

        # Get eigenvalues for wave speeds:
        λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_r( J_L, J_R,  Hr_L, Hr_R,  HZ_L, HZ_R )

        # Compute K = D × J at the interface:
        Krr_L = D_rr_L * J_L
        Krr_R = D_rr_R * J_R
        KrZ_L = D_rZ_L * J_L
        KrZ_R = D_rZ_R * J_R

        # Fluxes at i = 1/2:
        F_r[1,j]    = GetFluxHLL_J( Hr_L, Hr_R, J_L, J_R,   λ1_L, λ1_R, λ3_L, λ3_R,   
                                    τ_L, τ_R,  J[1,j], J_0 )                             # Radial flux of J at i = 1/2
        F_r_Hr[1,j] = GetFluxHLL( Krr_L, Krr_R, Hr_L, Hr_R, λ1_L, λ1_R, λ3_L, λ3_R,
                                  τ_L, τ_R, )                                            # Radial flux of H_r at i = 1/2
        F_r_HZ[1,j] = GetFluxHLL( KrZ_L, KrZ_R, HZ_L, HZ_R, λ1_L, λ1_R, λ3_L, λ3_R,
                                  τ_L, τ_R, )                                            # Radial flux of H_Z at i = 1/2

        ##############################################################################
        ##############          INTERFACE i = 3/2          ###########################
        ##############################################################################

        # Reconstruct variables at interface r_{3/2} (i.e. r[2]):
        J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
        τ_L, τ_R = ReconstructInterface( J[1,j],   J[2,j],   J[3,j],   J_0,
                                         H_r[1,j], H_r[2,j], H_r[3,j], Hr_0,
                                         H_Z[1,j], H_Z[2,j], H_Z[3,j], HZ_0,
                                         rCoM[1],  rCoM[2],  rCoM[3],  rCoM_0, 
                                         r[2],               # Interface r_{3/2}
                                         α_F[1,j], α_F[2,j] )

        # Compute interface D-values:
        D_rr_L, _, D_rZ_L = Compute_D_Edd( Hr_L, HZ_L, J_L )
        D_rr_R, _, D_rZ_R = Compute_D_Edd( Hr_R, HZ_R, J_R )

        # Get eigenvalues for wave speeds:
        λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_r( J_L, J_R,  Hr_L, Hr_R,  HZ_L, HZ_R )

        # Compute K = D × J at the interface:
        Krr_L = D_rr_L * J_L 
        Krr_R = D_rr_R * J_R 
        KrZ_L = D_rZ_L * J_L
        KrZ_R = D_rZ_R * J_R

        # Fluxes at i = 3/2:
        F_r[2,j]    = GetFluxHLL_J( Hr_L, Hr_R, J_L, J_R,   λ1_L, λ1_R, λ3_L, λ3_R,   
                                    τ_L, τ_R,  J[2,j], J[1,j] )                         # Radial flux of J at i = 3/2
        F_r_Hr[2,j] = GetFluxHLL( Krr_L, Krr_R, Hr_L, Hr_R, λ1_L, λ1_R, λ3_L, λ3_R,
                                  τ_L, τ_R, )                                           # Radial flux of H_r at i = 3/2
        F_r_HZ[2,j] = GetFluxHLL( KrZ_L, KrZ_R, HZ_L, HZ_R, λ1_L, λ1_R, λ3_L, λ3_R,
                                  τ_L, τ_R, )                                           # Radial flux of H_Z at i = 3/2


        ##############################################################################
        ##############        INTERIOR INTERFACES 2 ≤ i ≤ N_r-2          #############
        ##############################################################################
        @simd for i in dN_rint
            ip1 = i + 1
            ip2 = i + 2
            im1 = i - 1

            # Reconstruct variables at interface r_{i+1/2} (i.e. r[i+1]):
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
            τ_L, τ_R = ReconstructInterface( J[i,j],   J[ip1,j],   J[ip2,j],   J[im1,j],
                                             H_r[i,j], H_r[ip1,j], H_r[ip2,j], H_r[im1,j],
                                             H_Z[i,j], H_Z[ip1,j], H_Z[ip2,j], H_Z[im1,j],
                                             rCoM[i],  rCoM[ip1],  rCoM[ip2],  rCoM[im1], 
                                             r[ip1], # Interface r_{i+1/2}
                                             α_F[i,j], α_F[ip1,j]   )

            # Compute interface D-values:
            D_rr_L, _, D_rZ_L = Compute_D_Edd( Hr_L, HZ_L, J_L )
            D_rr_R, _, D_rZ_R = Compute_D_Edd( Hr_R, HZ_R, J_R )

            # Get eigenvalues for wave speeds:
            λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_r( J_L,   J_R,
                                                     Hr_L,  Hr_R,
                                                     HZ_L,  HZ_R )

            # Compute K = D × J at the interface:
            Krr_L = D_rr_L * J_L 
            Krr_R = D_rr_R * J_R 
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Radial flux of J:
            F_r[i+1,j]    = GetFluxHLL_J( Hr_L, Hr_R, J_L, J_R, 
                                          λ1_L,  λ1_R,  λ3_L,  λ3_R,  
                                          τ_L,  τ_R,  J[ip1,j], J[i,j] )

            # Radial flux of H_r:
            F_r_Hr[i+1,j] = GetFluxHLL( Krr_L, Krr_R,
                                        Hr_L,  Hr_R,
                                        λ1_L,  λ1_R,  λ3_L,  λ3_R,  τ_L, τ_R, )

            # Radial flux of H_Z:
            F_r_HZ[i+1,j] = GetFluxHLL( KrZ_L, KrZ_R,
                                        HZ_L,  HZ_R,
                                        λ1_L,  λ1_R,  λ3_L,  λ3_R,  τ_L, τ_R, )

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
        else
            # Ghost cell i = N_r + 1:
            J_Np1   =  J[N_r, j]
            Hr_Np1  =  H_r[N_r, j]
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
        τ_L, τ_R = ReconstructInterface( J[N_r-1,j],   J[N_r,j],   J_Np1,    J[N_r-2,j],
                                         H_r[N_r-1,j], H_r[N_r,j], Hr_Np1,   H_r[N_r-2,j],
                                         H_Z[N_r-1,j], H_Z[N_r,j], HZ_Np1,   H_Z[N_r-2,j],
                                         rCoM[N_r-1],  rCoM[N_r],  rCoM_Np1, rCoM[N_r-2], 
                                         r[N_r],               # Interface r_{N_r-1/2}
                                         α_F[N_r-1,j], α_F[N_r,j] )

        # Compute interface D-values:
        D_rr_L, _, D_rZ_L = Compute_D_Edd( Hr_L, HZ_L, J_L )
        D_rr_R, _, D_rZ_R = Compute_D_Edd( Hr_R, HZ_R, J_R )

        # Get eigenvalues for wave speeds:
        λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_r( J_L, J_R,  Hr_L, Hr_R,  HZ_L, HZ_R )

        # K = D × J at the interface i = N_r - 1/2:
        Krr_L = D_rr_L * J_L
        Krr_R = D_rr_R * J_R
        KrZ_L = D_rZ_L * J_L
        KrZ_R = D_rZ_R * J_R

        # Fluxes at interface i = N_r - 1/2:
        F_r[N_r, j]   = GetFluxHLL_J( Hr_L, Hr_R, J_L, J_R, λ1_L, λ1_R, λ3_L, λ3_R, 
                                      τ_L, τ_R,  J[N_r,j], J[N_r-1,j] )                   # Radial flux of J at i = N_r-1/2
        F_r_Hr[N_r,j] = GetFluxHLL( Krr_L, Krr_R, Hr_L, Hr_R, λ1_L, λ1_R, λ3_L, λ3_R,
                                    τ_L, τ_R, )                                           # Radial flux of H_r at i = N_r-1/2
        F_r_HZ[N_r,j] = GetFluxHLL( KrZ_L, KrZ_R, HZ_L, HZ_R, λ1_L, λ1_R, λ3_L, λ3_R,
                                    τ_L, τ_R, )                                           # Radial flux of H_Z at i = N_r-1/2

        ##############################################################################
        ##############         INTERFACE i = N_r + 1/2         #######################
        ##############################################################################

        # Reconstruct variables at interface r_{N_r+1/2} (i.e. r[N_r+1]):
        J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
        τ_L, τ_R = ReconstructInterface( J[N_r,j],   J_Np1,    J_Np2,    J[N_r-1,j],
                                         H_r[N_r,j], Hr_Np1,   Hr_Np2,   H_r[N_r-1,j],
                                         H_Z[N_r,j], HZ_Np1,   HZ_Np2,   H_Z[N_r-1,j],
                                         rCoM[N_r],  rCoM_Np1, rCoM_Np2, rCoM[N_r-1], 
                                         r[N_r+1],               # Interface r_{N_r+1/2}
                                         α_F[N_r,j], α_F[N_r,j]  # Assume opacity in ghost cell N_r+1 is same as for cell N_r
                                          )

        # Compute interface D-values:
        D_rr_L, _, D_rZ_L = Compute_D_Edd( Hr_L, HZ_L, J_L )
        D_rr_R, _, D_rZ_R = Compute_D_Edd( Hr_R, HZ_R, J_R )

        # Get eigenvalues for wave speeds:
        λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_r( J_L, J_R,  Hr_L, Hr_R,  HZ_L, HZ_R )

        # K = D × J at the interface i = N_r + 1/2:
        Krr_L = D_rr_L * J_L
        Krr_R = D_rr_R * J_R
        KrZ_L = D_rZ_L * J_L
        KrZ_R = D_rZ_R * J_R

        # Fluxes at interface i = N_r + 1/2:
        F_r[N_r+1, j]   = GetFluxHLL_J( Hr_L, Hr_R, J_L, J_R, λ1_L, λ1_R, λ3_L, λ3_R, 
                                        τ_L, τ_R,  J_Np1, J[N_r,j] )                       # Radial flux of J at i = N_r+1/2
        F_r_Hr[N_r+1,j] = GetFluxHLL( Krr_L, Krr_R, Hr_L, Hr_R, λ1_L, λ1_R, λ3_L, λ3_R, 
                                      τ_L, τ_R, )                                          # Radial flux of H_r at i = N_r+1/2
        F_r_HZ[N_r+1,j] = GetFluxHLL( KrZ_L, KrZ_R, HZ_L, HZ_R, λ1_L, λ1_R, λ3_L, λ3_R,
                                      τ_L, τ_R, )                                          # Radial flux of H_Z at i = N_r+1/2


        ####  UPDATE J AND H  ##############################################################
        @simd for i in dN_r
            J[i,j]   = max( UpdateConservativeQuantity( J[i,j], c̃Δt_half, inv_ΔV[i,j], 
                                                        A_r[i,j], A_r[i+1,j], 
                                                        F_r[i,j], F_r[i+1,j] ), 
                            Jeps )

            H_r[i,j] = UpdateConservativeQuantity( H_r[i,j], c̃Δt_half, inv_ΔV[i,j], 
                                                   A_r[i,j], A_r[i+1,j], 
                                                   F_r_Hr[i,j], F_r_Hr[i+1,j] )

            H_Z[i,j] = UpdateConservativeQuantity( H_Z[i,j], c̃Δt_half, inv_ΔV[i,j], 
                                                   A_r[i,j], A_r[i+1,j], 
                                                   F_r_HZ[i,j], F_r_HZ[i+1,j] )  
                                                   
            # If using cylindrical coordinates, add contribution from geometric source term,
            # (1/c̃) ∂H_r/∂t = K_ϕϕ/r = (1 - D_rr - D_ZZ) * J / r, using,
            # ⟨1/r⟩ ≡ ∫dV (1/r) / ΔV = 2π * Δr * ΔZ / ΔV = 1 / [( r[i+1] + r[i] ) / 2] (pre-computed)
            if CylindricalCoords 
                K_ϕϕ      = clamp( 1 - D_rr[i,j] - D_ZZ[i,j], 0.0, 1.0 ) * J[i,j]
                H_r[i,j] += c̃Δt_half * K_ϕϕ * inv_r_cell[i]
            end
        end
    end

    ################################################################################
    ############                                     ###############################
    ############  2. FULL STEP IN Z-DIRECTION        ###############################
    ############                                     ###############################
    ################################################################################

    # First update the Eddington factors:
    Update_D_Edd_Serial!( D_rr, D_ZZ, D_rZ, H_r, H_Z, J )

    @inbounds @simd for i in dN_r
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
        else
            # Ghost cell j = 0:
            J_0   =   J[i,1]
            HZ_0  =   H_Z[i,1]   
            Hr_0  =   H_r[i,1] 
            # Ghost cell j = -1:
            J_m1  =   J_0
            HZ_m1 =   HZ_0 
            Hr_m1 =   Hr_0 
        end

        # Reconstruct variables at interface Z_{1/2} (i.e. Z[1]):
        J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
        τ_L, τ_R = ReconstructInterface( J_0,    J[i,1],   J[i,2],   J_m1,
                                         Hr_0,   H_r[i,1], H_r[i,2], Hr_m1,
                                         HZ_0,   H_Z[i,1], H_Z[i,2], HZ_m1,
                                         ZCoM_0, ZCoM[1],  ZCoM[2],  ZCoM_m1, 
                                         Z[1],              # Interface Z_{1/2}
                                         α_F[i,1], α_F[i,1] # Assume same opacity in ghost cell j = 0 as in j = 1
                                            )

        # Compute interface D-values:
        _, D_ZZ_L, D_rZ_L = Compute_D_Edd( Hr_L, HZ_L, J_L )
        _, D_ZZ_R, D_rZ_R = Compute_D_Edd( Hr_R, HZ_R, J_R )

        # Get eigenvalues for wave speeds:
        λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_Z( J_L,   J_R,
                                                 Hr_L,  Hr_R,
                                                 HZ_L,  HZ_R )

        # Compute K = D × J at the interface, with the right BC for KrZ_L:
        KZZ_L = D_ZZ_L * J_L
        KZZ_R = D_ZZ_R * J_R
        KrZ_L = D_rZ_L * J_L
        KrZ_R = D_rZ_R * J_R

        # Fluxes at interface j = 1/2:
        F_Z[i,1]     = GetFluxHLL_J( HZ_L, HZ_R, J_L, J_R, λ1_L, λ1_R, λ3_L, λ3_R, 
                                     τ_L, τ_R,  J[i,1], J_0 )
        F_Z_HZ[i,1]  = GetFluxHLL( KZZ_L, KZZ_R, HZ_L, HZ_R, λ1_L, λ1_R, λ3_L, λ3_R, τ_L, τ_R )
        F_Z_Hr[i,1]  = GetFluxHLL( KrZ_L, KrZ_R, Hr_L, Hr_R, λ1_L, λ1_R, λ3_L, λ3_R, τ_L, τ_R )

        ##############################################################################
        ####################         INTERFACE j = 3/2         #######################
        ##############################################################################

        # Reconstruct variables at interface Z_{3/2} (i.e. Z[2]):
        J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
        τ_L, τ_R = ReconstructInterface( J[i,1],   J[i,2],   J[i,3],   J_0,
                                         H_r[i,1], H_r[i,2], H_r[i,3], Hr_0,
                                         H_Z[i,1], H_Z[i,2], H_Z[i,3], HZ_0,
                                         ZCoM[1],  ZCoM[2],  ZCoM[3],  ZCoM_0, 
                                         Z[2],              # Interface Z_{3/2}
                                         α_F[i,1], α_F[i,2] )

        # Compute interface D-values:
        _, D_ZZ_L, D_rZ_L = Compute_D_Edd( Hr_L, HZ_L, J_L )
        _, D_ZZ_R, D_rZ_R = Compute_D_Edd( Hr_R, HZ_R, J_R )

        # Get eigenvalues for wave speeds:
        λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_Z( J_L,   J_R,
                                                 Hr_L,  Hr_R,
                                                 HZ_L,  HZ_R )

        # Compute K = D × J at the interface j = 3/2:
        KZZ_L = D_ZZ_L * J_L
        KZZ_R = D_ZZ_R * J_R
        KrZ_L = D_rZ_L * J_L
        KrZ_R = D_rZ_R * J_R

        # Fluxes at interface j = 3/2:
        F_Z[i,2]     = GetFluxHLL_J( HZ_L, HZ_R, J_L, J_R, λ1_L, λ1_R, λ3_L, λ3_R, 
                                     τ_L, τ_R,  J[i,2], J[i,1] )
        F_Z_HZ[i,2]  = GetFluxHLL( KZZ_L, KZZ_R, HZ_L, HZ_R, λ1_L, λ1_R, λ3_L, λ3_R, τ_L, τ_R )
        F_Z_Hr[i,2]  = GetFluxHLL( KrZ_L, KrZ_R, Hr_L, Hr_R, λ1_L, λ1_R, λ3_L, λ3_R, τ_L, τ_R )

    end


    for j in dN_Zint
        # Convenient indices: 
        jp1 = j + 1
        jp2 = j + 2
        jm1 = j - 1

        @inbounds @simd for i in dN_r

            # Reconstruct variables at interface Z_{j+1/2} (i.e. Z[j+1]):
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
            τ_L, τ_R = ReconstructInterface( J[i,j],   J[i,jp1],   J[i,jp2],   J[i,jm1],
                                             H_r[i,j], H_r[i,jp1], H_r[i,jp2], H_r[i,jm1],
                                             H_Z[i,j], H_Z[i,jp1], H_Z[i,jp2], H_Z[i,jm1],
                                             ZCoM[j],  ZCoM[jp1],  ZCoM[jp2],  ZCoM[jm1], 
                                             Z[jp1],              # Interface Z_{j+1/2}
                                             α_F[i,j], α_F[i,jp1] )

            # Compute interface D-values:
            _, D_ZZ_L, D_rZ_L = Compute_D_Edd( Hr_L, HZ_L, J_L )
            _, D_ZZ_R, D_rZ_R = Compute_D_Edd( Hr_R, HZ_R, J_R )

            # Get eigenvalues for wave speeds:
            λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_Z( J_L,   J_R,
                                                     Hr_L,  Hr_R,
                                                     HZ_L,  HZ_R )

            # Compute K = D × J at the interface j + 1/2:
            KZZ_L = D_ZZ_L * J_L
            KZZ_R = D_ZZ_R * J_R
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Fluxes at interface j + 1/2:
            F_Z[i,jp1]     = GetFluxHLL_J( HZ_L, HZ_R, J_L, J_R, λ1_L, λ1_R, λ3_L, λ3_R, 
                                           τ_L, τ_R,  J[i,jp1], J[i,j] )
            F_Z_HZ[i,jp1]  = GetFluxHLL( KZZ_L, KZZ_R, HZ_L, HZ_R, λ1_L, λ1_R, λ3_L, λ3_R, τ_L, τ_R )
            F_Z_Hr[i,jp1]  = GetFluxHLL( KrZ_L, KrZ_R, Hr_L, Hr_R, λ1_L, λ1_R, λ3_L, λ3_R, τ_L, τ_R )
        end
    end

    @inbounds @simd for i in dN_r

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
        else
            # Ghost cell j = N_Z + 1:
            J_Np1   =   J[i,N_Z]
            HZ_Np1  =   H_Z[i,N_Z]    
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
        τ_L, τ_R = ReconstructInterface( J[i,N_Z-1],   J[i,N_Z],   J_Np1,     J[i,N_Z-2],
                                         H_r[i,N_Z-1], H_r[i,N_Z], Hr_Np1,    H_r[i,N_Z-2],
                                         H_Z[i,N_Z-1], H_Z[i,N_Z], HZ_Np1,    H_Z[i,N_Z-2],
                                         ZCoM[N_Z-1],  ZCoM[N_Z],  ZCoM_Np1,  ZCoM[N_Z-2], 
                                         Z[N_Z],            # Interface Z_{N_Z-1/2}
                                         α_F[i,N_Z-1], α_F[i,N_Z] )

        # Compute interface D-values:
        _, D_ZZ_L, D_rZ_L = Compute_D_Edd( Hr_L, HZ_L, J_L )
        _, D_ZZ_R, D_rZ_R = Compute_D_Edd( Hr_R, HZ_R, J_R )

        # Get eigenvalues for wave speeds:
        λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_Z( J_L,   J_R,
                                                 Hr_L,  Hr_R,
                                                 HZ_L,  HZ_R )

        # Compute K = D × J at the interface j = N_Z - 1/2:
        KZZ_L = D_ZZ_L * J_L
        KZZ_R = D_ZZ_R * J_R
        KrZ_L = D_rZ_L * J_L
        KrZ_R = D_rZ_R * J_R

        # Fluxes at interface j = N_Z - 1/2:
        F_Z[i,N_Z]     = GetFluxHLL_J( HZ_L, HZ_R, J_L, J_R, λ1_L, λ1_R, λ3_L, λ3_R, 
                                       τ_L, τ_R,  J[i,N_Z], J[i,N_Z-1] )
        F_Z_HZ[i,N_Z]  = GetFluxHLL( KZZ_L, KZZ_R, HZ_L, HZ_R, λ1_L, λ1_R, λ3_L, λ3_R, τ_L, τ_R )
        F_Z_Hr[i,N_Z]  = GetFluxHLL( KrZ_L, KrZ_R, Hr_L, Hr_R, λ1_L, λ1_R, λ3_L, λ3_R, τ_L, τ_R )

        ##############################################################################
        ###############        INTERFACE j = N_Z + 1/2         #######################
        ##############################################################################

        # Reconstruct variables at interface Z_{N_Z+1/2} (i.e. Z[N_Z+1]):
        J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
        τ_L, τ_R = ReconstructInterface( J[i,N_Z],   J_Np1,     J_Np2,     J[i,N_Z-1],
                                         H_r[i,N_Z], Hr_Np1,    Hr_Np2,    H_r[i,N_Z-1],
                                         H_Z[i,N_Z], HZ_Np1,    HZ_Np2,    H_Z[i,N_Z-1],
                                         ZCoM[N_Z],  ZCoM_Np1,  ZCoM_Np2,  ZCoM[N_Z-1], 
                                         Z[N_Z+1],               # Interface Z_{N_Z+1/2}
                                         α_F[i,N_Z], α_F[i,N_Z]  # Assume opacity in ghost cell j = N_Z+1 is same as in j = N_Z
                                          )

        # Compute interface D-values:
        _, D_ZZ_L, D_rZ_L = Compute_D_Edd( Hr_L, HZ_L, J_L )
        _, D_ZZ_R, D_rZ_R = Compute_D_Edd( Hr_R, HZ_R, J_R )

        # Get eigenvalues for wave speeds:
        λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_Z( J_L,   J_R,
                                                 Hr_L,  Hr_R,
                                                 HZ_L,  HZ_R )

        # Compute K = D × J at the interface j = N_Z + 1/2:
        KZZ_L = D_ZZ_L * J_L
        KZZ_R = D_ZZ_R * J_R
        KrZ_L = D_rZ_L * J_L
        KrZ_R = D_rZ_R * J_R

        # Fluxes at interface j = N_Z + 1/2:
        F_Z[i,N_Z+1]     = GetFluxHLL_J( HZ_L, HZ_R, J_L, J_R, λ1_L, λ1_R, λ3_L, λ3_R, 
                                         τ_L, τ_R,  J_Np1, J[i,N_Z] )
        F_Z_HZ[i,N_Z+1]  = GetFluxHLL( KZZ_L, KZZ_R, HZ_L, HZ_R, λ1_L, λ1_R, λ3_L, λ3_R, τ_L, τ_R )
        F_Z_Hr[i,N_Z+1]  = GetFluxHLL( KrZ_L, KrZ_R, Hr_L, Hr_R, λ1_L, λ1_R, λ3_L, λ3_R, τ_L, τ_R )

    end

    ####  UPDATE J AND H  ##############################################################
    for j in dN_Z
        @inbounds @simd for i in dN_r
            J[i,j]   = max( UpdateConservativeQuantity( J[i,j], c̃Δt, inv_ΔV[i,j], 
                                                        A_Z[i,j], A_Z[i,j+1], 
                                                        F_Z[i,j], F_Z[i,j+1] ),
                            Jeps )

            H_Z[i,j] = UpdateConservativeQuantity( H_Z[i,j], c̃Δt, inv_ΔV[i,j],
                                                   A_Z[i,j], A_Z[i,j+1],
                                                   F_Z_HZ[i,j], F_Z_HZ[i,j+1] )

            H_r[i,j] = UpdateConservativeQuantity( H_r[i,j], c̃Δt, inv_ΔV[i,j],
                                                   A_Z[i,j], A_Z[i,j+1],
                                                   F_Z_Hr[i,j], F_Z_Hr[i,j+1] )
        end
    end

    ################################################################################
    ############                                     ###############################
    ############  3. FINAL HALF-STEP IN r-DIRECTION  ###############################
    ############                                     ###############################
    ################################################################################

    # First update the Eddington factors:
    Update_D_Edd_Serial!( D_rr, D_ZZ, D_rZ, H_r, H_Z, J )

    @inbounds for j in dN_Z
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
        else
            # Ghost cell i = 0:
            J_0    = J[1,j] 
            Hr_0   = H_r[1,j]           
            HZ_0   = H_Z[1,j]
            # Ghost cell i = -1 (no gradients, so equal to cells i = 0 and 1):
            J_m1   = J_0 
            Hr_m1  = Hr_0   
            HZ_m1  = HZ_0
        end

        # Reconstruct variables at interface r_{1/2} (i.e. r[1]):
        J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
        τ_L, τ_R = ReconstructInterface( J_0,     J[1,j],   J[2,j],   J_m1,
                                         Hr_0,    H_r[1,j], H_r[2,j], Hr_m1,
                                         HZ_0,    H_Z[1,j], H_Z[2,j], HZ_m1,
                                         rCoM_0,  rCoM[1],  rCoM[2],  rCoM_m1, 
                                         r[1],               # Interface r_{1/2}
                                         α_F[1,j], α_F[1,j]  # Assume ghost cell 0 has same opacity as cell i = 1
                                          )

        # Compute interface D-values:
        D_rr_L, _, D_rZ_L = Compute_D_Edd( Hr_L, HZ_L, J_L )
        D_rr_R, _, D_rZ_R = Compute_D_Edd( Hr_R, HZ_R, J_R )

        # Get eigenvalues for wave speeds:
        λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_r( J_L, J_R,  Hr_L, Hr_R,  HZ_L, HZ_R )

        # Compute K = D × J at the interface:
        Krr_L = D_rr_L * J_L
        Krr_R = D_rr_R * J_R
        KrZ_L = D_rZ_L * J_L
        KrZ_R = D_rZ_R * J_R

        # Fluxes at i = 1/2:
        F_r[1,j]    = GetFluxHLL_J( Hr_L, Hr_R, J_L, J_R,   λ1_L, λ1_R, λ3_L, λ3_R,   
                                    τ_L, τ_R,  J[1,j], J_0 )                             # Radial flux of J at i = 1/2
        F_r_Hr[1,j] = GetFluxHLL( Krr_L, Krr_R, Hr_L, Hr_R, λ1_L, λ1_R, λ3_L, λ3_R,
                                  τ_L, τ_R, )                                            # Radial flux of H_r at i = 1/2
        F_r_HZ[1,j] = GetFluxHLL( KrZ_L, KrZ_R, HZ_L, HZ_R, λ1_L, λ1_R, λ3_L, λ3_R,
                                  τ_L, τ_R, )                                            # Radial flux of H_Z at i = 1/2

        ##############################################################################
        ##############          INTERFACE i = 3/2          ###########################
        ##############################################################################

        # Reconstruct variables at interface r_{3/2} (i.e. r[2]):
        J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
        τ_L, τ_R = ReconstructInterface( J[1,j],   J[2,j],   J[3,j],   J_0,
                                         H_r[1,j], H_r[2,j], H_r[3,j], Hr_0,
                                         H_Z[1,j], H_Z[2,j], H_Z[3,j], HZ_0,
                                         rCoM[1],  rCoM[2],  rCoM[3],  rCoM_0, 
                                         r[2],               # Interface r_{3/2}
                                         α_F[1,j], α_F[2,j] )

        # Compute interface D-values:
        D_rr_L, _, D_rZ_L = Compute_D_Edd( Hr_L, HZ_L, J_L )
        D_rr_R, _, D_rZ_R = Compute_D_Edd( Hr_R, HZ_R, J_R )

        # Get eigenvalues for wave speeds:
        λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_r( J_L, J_R,  Hr_L, Hr_R,  HZ_L, HZ_R )

        # Compute K = D × J at the interface:
        Krr_L = D_rr_L * J_L 
        Krr_R = D_rr_R * J_R 
        KrZ_L = D_rZ_L * J_L
        KrZ_R = D_rZ_R * J_R

        # Fluxes at i = 3/2:
        F_r[2,j]    = GetFluxHLL_J( Hr_L, Hr_R, J_L, J_R,   λ1_L, λ1_R, λ3_L, λ3_R,   
                                    τ_L, τ_R,  J[2,j], J[1,j] )                         # Radial flux of J at i = 3/2
        F_r_Hr[2,j] = GetFluxHLL( Krr_L, Krr_R, Hr_L, Hr_R, λ1_L, λ1_R, λ3_L, λ3_R,
                                  τ_L, τ_R, )                                           # Radial flux of H_r at i = 3/2
        F_r_HZ[2,j] = GetFluxHLL( KrZ_L, KrZ_R, HZ_L, HZ_R, λ1_L, λ1_R, λ3_L, λ3_R,
                                  τ_L, τ_R, )                                           # Radial flux of H_Z at i = 3/2


        ##############################################################################
        ##############        INTERIOR INTERFACES 2 ≤ i ≤ N_r-2          #############
        ##############################################################################
        @simd for i in dN_rint
            ip1 = i + 1
            ip2 = i + 2
            im1 = i - 1

            # Reconstruct variables at interface r_{i+1/2} (i.e. r[i+1]):
            J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
            τ_L, τ_R = ReconstructInterface( J[i,j],   J[ip1,j],   J[ip2,j],   J[im1,j],
                                             H_r[i,j], H_r[ip1,j], H_r[ip2,j], H_r[im1,j],
                                             H_Z[i,j], H_Z[ip1,j], H_Z[ip2,j], H_Z[im1,j],
                                             rCoM[i],  rCoM[ip1],  rCoM[ip2],  rCoM[im1], 
                                             r[ip1], # Interface r_{i+1/2}
                                             α_F[i,j], α_F[ip1,j]   )

            # Compute interface D-values:
            D_rr_L, _, D_rZ_L = Compute_D_Edd( Hr_L, HZ_L, J_L )
            D_rr_R, _, D_rZ_R = Compute_D_Edd( Hr_R, HZ_R, J_R )

            # Get eigenvalues for wave speeds:
            λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_r( J_L,   J_R,
                                                     Hr_L,  Hr_R,
                                                     HZ_L,  HZ_R )

            # Compute K = D × J at the interface:
            Krr_L = D_rr_L * J_L 
            Krr_R = D_rr_R * J_R 
            KrZ_L = D_rZ_L * J_L
            KrZ_R = D_rZ_R * J_R

            # Radial flux of J:
            F_r[i+1,j]    = GetFluxHLL_J( Hr_L, Hr_R, J_L, J_R, 
                                          λ1_L,  λ1_R,  λ3_L,  λ3_R,  
                                          τ_L,  τ_R,  J[ip1,j], J[i,j] )

            # Radial flux of H_r:
            F_r_Hr[i+1,j] = GetFluxHLL( Krr_L, Krr_R,
                                        Hr_L,  Hr_R,
                                        λ1_L,  λ1_R,  λ3_L,  λ3_R,  τ_L, τ_R, )

            # Radial flux of H_Z:
            F_r_HZ[i+1,j] = GetFluxHLL( KrZ_L, KrZ_R,
                                        HZ_L,  HZ_R,
                                        λ1_L,  λ1_R,  λ3_L,  λ3_R,  τ_L, τ_R, )

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
        else
            # Ghost cell i = N_r + 1:
            J_Np1   =  J[N_r, j]
            Hr_Np1  =  H_r[N_r, j]
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
        τ_L, τ_R = ReconstructInterface( J[N_r-1,j],   J[N_r,j],   J_Np1,    J[N_r-2,j],
                                         H_r[N_r-1,j], H_r[N_r,j], Hr_Np1,   H_r[N_r-2,j],
                                         H_Z[N_r-1,j], H_Z[N_r,j], HZ_Np1,   H_Z[N_r-2,j],
                                         rCoM[N_r-1],  rCoM[N_r],  rCoM_Np1, rCoM[N_r-2], 
                                         r[N_r],               # Interface r_{N_r-1/2}
                                         α_F[N_r-1,j], α_F[N_r,j] )

        # Compute interface D-values:
        D_rr_L, _, D_rZ_L = Compute_D_Edd( Hr_L, HZ_L, J_L )
        D_rr_R, _, D_rZ_R = Compute_D_Edd( Hr_R, HZ_R, J_R )

        # Get eigenvalues for wave speeds:
        λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_r( J_L, J_R,  Hr_L, Hr_R,  HZ_L, HZ_R )

        # K = D × J at the interface i = N_r - 1/2:
        Krr_L = D_rr_L * J_L
        Krr_R = D_rr_R * J_R
        KrZ_L = D_rZ_L * J_L
        KrZ_R = D_rZ_R * J_R

        # Fluxes at interface i = N_r - 1/2:
        F_r[N_r, j]   = GetFluxHLL_J( Hr_L, Hr_R, J_L, J_R, λ1_L, λ1_R, λ3_L, λ3_R, 
                                      τ_L, τ_R,  J[N_r,j], J[N_r-1,j] )                   # Radial flux of J at i = N_r-1/2
        F_r_Hr[N_r,j] = GetFluxHLL( Krr_L, Krr_R, Hr_L, Hr_R, λ1_L, λ1_R, λ3_L, λ3_R,
                                    τ_L, τ_R, )                                           # Radial flux of H_r at i = N_r-1/2
        F_r_HZ[N_r,j] = GetFluxHLL( KrZ_L, KrZ_R, HZ_L, HZ_R, λ1_L, λ1_R, λ3_L, λ3_R,
                                    τ_L, τ_R, )                                           # Radial flux of H_Z at i = N_r-1/2

        ##############################################################################
        ##############         INTERFACE i = N_r + 1/2         #######################
        ##############################################################################

        # Reconstruct variables at interface r_{N_r+1/2} (i.e. r[N_r+1]):
        J_L, J_R, Hr_L, Hr_R, HZ_L, HZ_R, 
        τ_L, τ_R = ReconstructInterface( J[N_r,j],   J_Np1,    J_Np2,    J[N_r-1,j],
                                         H_r[N_r,j], Hr_Np1,   Hr_Np2,   H_r[N_r-1,j],
                                         H_Z[N_r,j], HZ_Np1,   HZ_Np2,   H_Z[N_r-1,j],
                                         rCoM[N_r],  rCoM_Np1, rCoM_Np2, rCoM[N_r-1], 
                                         r[N_r+1],               # Interface r_{N_r+1/2}
                                         α_F[N_r,j], α_F[N_r,j]  # Assume opacity in ghost cell N_r+1 is same as for cell N_r
                                          )

        # Compute interface D-values:
        D_rr_L, _, D_rZ_L = Compute_D_Edd( Hr_L, HZ_L, J_L )
        D_rr_R, _, D_rZ_R = Compute_D_Edd( Hr_R, HZ_R, J_R )

        # Get eigenvalues for wave speeds:
        λ1_L, λ1_R, λ3_L, λ3_R = M1WaveSpeeds_r( J_L, J_R,  Hr_L, Hr_R,  HZ_L, HZ_R )

        # K = D × J at the interface i = N_r + 1/2:
        Krr_L = D_rr_L * J_L
        Krr_R = D_rr_R * J_R
        KrZ_L = D_rZ_L * J_L
        KrZ_R = D_rZ_R * J_R

        # Fluxes at interface i = N_r + 1/2:
        F_r[N_r+1, j]   = GetFluxHLL_J( Hr_L, Hr_R, J_L, J_R, λ1_L, λ1_R, λ3_L, λ3_R, 
                                        τ_L, τ_R,  J_Np1, J[N_r,j] )                       # Radial flux of J at i = N_r+1/2
        F_r_Hr[N_r+1,j] = GetFluxHLL( Krr_L, Krr_R, Hr_L, Hr_R, λ1_L, λ1_R, λ3_L, λ3_R, 
                                      τ_L, τ_R, )                                          # Radial flux of H_r at i = N_r+1/2
        F_r_HZ[N_r+1,j] = GetFluxHLL( KrZ_L, KrZ_R, HZ_L, HZ_R, λ1_L, λ1_R, λ3_L, λ3_R,
                                      τ_L, τ_R, )                                          # Radial flux of H_Z at i = N_r+1/2


        ####  UPDATE J AND H  ##############################################################
        @simd for i in dN_r
            J[i,j]   = max( UpdateConservativeQuantity( J[i,j], c̃Δt_half, inv_ΔV[i,j], 
                                                        A_r[i,j], A_r[i+1,j], 
                                                        F_r[i,j], F_r[i+1,j] ), 
                            Jeps )

            H_r[i,j] = UpdateConservativeQuantity( H_r[i,j], c̃Δt_half, inv_ΔV[i,j], 
                                                   A_r[i,j], A_r[i+1,j], 
                                                   F_r_Hr[i,j], F_r_Hr[i+1,j] )

            H_Z[i,j] = UpdateConservativeQuantity( H_Z[i,j], c̃Δt_half, inv_ΔV[i,j], 
                                                   A_r[i,j], A_r[i+1,j], 
                                                   F_r_HZ[i,j], F_r_HZ[i+1,j] )  
                                                   
            # If using cylindrical coordinates, add contribution from geometric source term,
            # (1/c̃) ∂H_r/∂t = K_ϕϕ/r = (1 - D_rr - D_ZZ) * J / r, using,
            # ⟨1/r⟩ ≡ ∫dV (1/r) / ΔV = 2π * Δr * ΔZ / ΔV = 1 / [( r[i+1] + r[i] ) / 2] (pre-computed)
            if CylindricalCoords 
                K_ϕϕ      = clamp( 1 - D_rr[i,j] - D_ZZ[i,j], 0.0, 1.0 ) * J[i,j]
                H_r[i,j] += c̃Δt_half * K_ϕϕ * inv_r_cell[i]
            end
        end
    end

    # Update Jold:
    J_old .= J 

    return nothing
end
