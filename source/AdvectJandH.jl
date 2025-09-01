
module AdvectJandH
""" MODULE:  AdvectJandH

    PURPOSE: Contains the code for advecting the mean intensity J, and the
             first moment H (the Eddington tensor D = K / J is estimated 
             using an M1 closure). Uses an explicit Godunov method which is 
             first-order in space and time.

"""

__precompile__()

include( "./M1Closure.jl" )
using .M1Closure        # For the M1 closure 
using Base.Threads      # For CPU acceleration

# Export functions:
export AdvectJandH!


""" The Global Lax Friedrich (GLF) flux function,
    for a conserved variable Q in (J, H_r, H_Z).  """
@inline function GetFluxGLF( F_L::Float64, 
                             F_R::Float64, 
                             Q_L::Float64, 
                             Q_R::Float64 )

    return 0.5 * ( ( F_L  +  F_R )  -  ( Q_R  -  Q_L ) )
end 


""" Solves the PDE (1/cred) (∂Q/∂t) + ∇⋅F = 0 over a time-step Δt,
    using explicit time-integration, and the finite-volume 
    method. 
    
    **ARGUMENTS:**
    
    Q       :   Value of Q at "old" time-step n
    Δt      :   Time-step 
    ΔV      :   Volume of cell i 
    A_L     :   Area of left interface i - 1/2
    A_R     :   Area of right interface i + 1/2
    F_L     :   Flux F at left interface i - 1/2
    F_R     :   Flux F at right interface i + 1/2
    cred    :   Reduced speed of light 

    **RETURNS:**

    Qnew    :   Value of Q at time-step n + 1 
"""
@inline function UpdateConservativeQuantity( Q::Float64, 
                                            Δt::Float64,
                                            ΔV::Float64,
                                           A_L::Float64,
                                           A_R::Float64,
                                           F_L::Float64,
                                           F_R::Float64,
                                          cred::Float64
                                             )

    Qnew = Q  -  ( cred * Δt / ΔV ) * ( A_R * F_R  -  A_L * F_L )
    return Qnew
end


function AdvectJandH!( Δt::Float64, # Time-step 
                     cred::Float64, # Reduced speed of light 

                        J::AbstractArray{Float64, 2}, # J (latest/updated values)  (N_r x N_Z)
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

                       ΔV::AbstractArray{Float64, 2}, # Cell volumes                        (N_r   x N_Z)
                      A_r::AbstractArray{Float64, 2}, # Cell interface areas in r-direction (N_r+1 x N_Z)
                      A_Z::AbstractArray{Float64, 2}, # Cell interface areas in Z-direction (N_r   x N_Z+1)

                BC_r_left::Symbol   = :Reflect,       # Boundary condition at i = 1/2 
               BC_r_right::Symbol   = :Outflow,       # Boundary condition at i = N_r + 1/2
                 BC_Z_bot::Symbol   = :Outflow,       # Boundary condition at j = 1/2 
                 BC_Z_top::Symbol   = :Outflow        # Boundary condition at j = N_Z + 1/2
                  )

    N_r, N_Z = size( J )
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z
    dN_rm1   = 1:(N_r-1)
    dN_Zm1   = 1:(N_Z-1)

    Δt_half  = Δt / 2

    # Initialize J_old:
    J_old   .= J

    ################################################################################
    ############                                     ###############################
    ############  1. FIRST HALF-STEP IN r-DIRECTION  ###############################
    ############                                     ###############################
    ################################################################################

    # First update the Eddington factors:
    Update_D_Edd!( D_rr, D_ZZ, D_rZ, H_r, H_Z, J )

    @threads for j in dN_Z
        ####  Inner radial boundary condition at i = 1/2  ############################
        J_L, Hr_L, HZ_L = J[1,j], H_r[1,j], H_Z[1,j]

        if BC_r_left === :Reflect 
            Jg    = J_L 
            Hrg   = - Hr_L            # Reflect radial flux
            HZg   = HZ_L
            Krr_g = D_rr[1,j]   * J_L 
            KrZ_g = - D_rZ[1,j] * J_L # Reflect K_rZ since it is odd 
        else
            Jg    = J_L 
            Hrg   = Hr_L            
            HZg   = HZ_L
            Krr_g = D_rr[1,j] * J_L 
            KrZ_g = D_rZ[1,j] * J_L 
        end

        F_r[1,j]    = GetFluxGLF( Hrg, Hr_L, Jg, J_L )                # Radial flux of J at i = 1/2
        F_r_Hr[1,j] = GetFluxGLF( Krr_g, D_rr[1,j] * J_L, Hrg, Hr_L ) # Radial flux of H_r at i = 1/2
        F_r_HZ[1,j] = GetFluxGLF( KrZ_g, D_rZ[1,j] * J_L, HZg, HZ_L ) # Radial flux of H_Z at i = 1/2

        ####  Get fluxes at interior i-values  #######################################
        @inbounds @simd for i in dN_rm1
            # Radial flux of J:
            F_r[i+1,j]    = GetFluxGLF( H_r[i,j], H_r[i+1,j], J[i,j], J[i+1,j] )

            # Radial flux of H_r:
            F_r_Hr[i+1,j] = GetFluxGLF( D_rr[i,j] * J[i,j], D_rr[i+1,j] * J[i+1,j],
                                        H_r[i,j],           H_r[i+1,j] )

            # Radial flux of H_Z:
            F_r_HZ[i+1,j] = GetFluxGLF( D_rZ[i,j] * J[i,j], D_rZ[i+1,j] * J[i+1,j],
                                        H_Z[i,j],           H_Z[i+1,j] )
        end

        ####  Outer radial boundary condition at i = N_r + 1/2  ######################

        J_R  = J[N_r, j]
        Hr_R = H_r[N_r, j]
        HZ_R = H_Z[N_r, j]

        if BC_r_right === :Reflect 
            Jg    = J_R 
            Hrg   = - Hr_R 
            HZg   = HZ_R 
            Krr_g = D_rr[N_r,j]    * J_R 
            KrZ_g = - D_rZ[N_r, j] * J_R
        else
            Jg    = J_R 
            Hrg   = Hr_R 
            HZg   = HZ_R 
            Krr_g = D_rr[N_r,j]  * J_R 
            KrZ_g = D_rZ[N_r, j] * J_R
        end

        F_r[N_r+1,j]    = GetFluxGLF( Hr_R, Hrg, J_R, Jg )                 # Radial flux of J at i = N_r + 1/2
        F_r_Hr[N_r+1,j] = GetFluxGLF( D_rr[N_r,j] * J_R, Krr_g, Hr_R, Hrg) # Radial flux of H_r at i = N_r + 1/2
        F_r_HZ[N_r+1,j] = GetFluxGLF( D_rZ[N_r,j] * J_R, KrZ_g, HZ_R, HZg) # Radial flux of H_Z at i = N_r + 1/2

        ####  UPDATE J AND H  ##############################################################
        @inbounds @simd for i in dN_r
            J[i,j]   = UpdateConservativeQuantity( J[i,j], Δt_half, ΔV[i,j], 
                                                   A_r[i,j], A_r[i+1,j], 
                                                   F_r[i,j], F_r[i+1,j], cred )

            H_r[i,j] = UpdateConservativeQuantity( H_r[i,j], Δt_half, ΔV[i,j], 
                                                   A_r[i,j], A_r[i+1,j], 
                                                   F_r_Hr[i,j], F_r_Hr[i+1,j], cred )

            H_Z[i,j] = UpdateConservativeQuantity( H_Z[i,j], Δt_half, ΔV[i,j], 
                                                   A_r[i,j], A_r[i+1,j], 
                                                   F_r_HZ[i,j], F_r_HZ[i+1,j], cred )              
        end
    end

    ################################################################################
    ############                                     ###############################
    ############  2. FULL STEP IN Z-DIRECTION        ###############################
    ############                                     ###############################
    ################################################################################

    # First update the Eddington factors:
    Update_D_Edd!( D_rr, D_ZZ, D_rZ, H_r, H_Z, J )

    ####  Bottom boundary condition at j = 1/2  #################################
    @inbounds @simd for i in dN_r
        J_B, Hr_B, HZ_B = J[i,1], H_r[i,1], H_Z[i,1]

        if BC_Z_bot === :Reflect
            Jg    = J_B 
            HZg   = - HZ_B
            Hrg   = Hr_B 
            KZZ_g = D_ZZ[i,1]   * J_B 
            KrZ_g = - D_rZ[i,1] * J_B # K_rZ is odd 
        else
            Jg    = J_B 
            HZg   = HZ_B
            Hrg   = Hr_B 
            KZZ_g = D_ZZ[i,1] * J_B 
            KrZ_g = D_rZ[i,1] * J_B 
        end

        # Z-direction flux at j = 1/2:
        F_Z[i,1]    = GetFluxGLF( HZg,   H_Z[i,1],           Jg,   J[i,1] )   # Z-direction flux of J
        F_Z_HZ[i,1] = GetFluxGLF( KZZ_g, D_ZZ[i,1] * J[i,1], HZg,  H_Z[i,1] ) # Z-direction flux of H_Z 
        F_Z_Hr[i,1] = GetFluxGLF( KrZ_g, D_rZ[i,1] * J[i,1], Hrg,  H_r[i,1] ) # Z-direction flux of H_r
    end

    ####  Get fluxes at interior j-values  #######################################
    @threads for j in dN_Zm1
        @inbounds @simd for i in dN_r
            F_Z[i,j+1]    = GetFluxGLF( H_Z[i,j], H_Z[i,j+1], J[i,j], J[i,j+1] )

            F_Z_HZ[i,j+1] = GetFluxGLF( D_ZZ[i,j] * J[i,j], D_ZZ[i,j+1] * J[i,j+1], H_Z[i,j], H_Z[i,j+1] )

            F_Z_Hr[i,j+1] = GetFluxGLF( D_rZ[i,j] * J[i,j], D_rZ[i,j+1] * J[i,j+1], H_r[i,j], H_r[i,j+1] )
        end
    end

    ####  Top boundary condition at j = N_Z + 1/2  ###############################
    @inbounds @simd for i in dN_r
        J_T, Hr_T, HZ_T = J[i, N_Z], H_r[i, N_Z], H_Z[i, N_Z]

        if BC_Z_top === :Reflect
            Jg    = J_T 
            HZg   = - HZ_T 
            Hrg   = Hr_T 
            KZZ_g = D_ZZ[i, N_Z]   * J_T 
            KrZ_g = - D_rZ[i, N_Z] * J_T
        else
            Jg    = J_T 
            HZg   = HZ_T 
            Hrg   = Hr_T 
            KZZ_g = D_ZZ[i, N_Z] * J_T 
            KrZ_g = D_rZ[i, N_Z] * J_T
        end

        # Z-direction flux at j = N_Z + 1/2:
        F_Z[i, N_Z+1]    = GetFluxGLF( HZ_T, HZg, J_T, Jg )
        F_Z_HZ[i, N_Z+1] = GetFluxGLF( D_ZZ[i, N_Z] * J_T, KZZ_g, HZ_T, HZg )
        F_Z_Hr[i, N_Z+1] = GetFluxGLF( D_rZ[i, N_Z] * J_T, KrZ_g, Hr_T, Hrg )
    end

    ####  UPDATE J AND H  ##############################################################
    @threads for j in dN_Z
        @inbounds @simd for i in dN_r
            J[i,j]   = UpdateConservativeQuantity( J[i,j], Δt, ΔV[i,j], 
                                                   A_Z[i,j], A_Z[i,j+1], 
                                                   F_Z[i,j], F_Z[i,j+1], cred )

            H_Z[i,j] = UpdateConservativeQuantity( H_Z[i,j], Δt, ΔV[i,j],
                                                   A_Z[i,j], A_Z[i,j+1],
                                                   F_Z_HZ[i,j], F_Z_HZ[i,j+1], cred )

            H_r[i,j] = UpdateConservativeQuantity( H_r[i,j], Δt, ΔV[i,j],
                                                   A_Z[i,j], A_Z[i,j+1],
                                                   F_Z_Hr[i,j], F_Z_Hr[i,j+1], cred )
        end
    end

    ################################################################################
    ############                                     ###############################
    ############  3. FINAL HALF-STEP IN r-DIRECTION  ###############################
    ############                                     ###############################
    ################################################################################

    # First update the Eddington factors:
    Update_D_Edd!( D_rr, D_ZZ, D_rZ, H_r, H_Z, J )

    @threads for j in dN_Z
        ####  Inner radial boundary condition at i = 1/2  ############################
        J_L, Hr_L, HZ_L = J[1,j], H_r[1,j], H_Z[1,j]

        if BC_r_left === :Reflect 
            Jg    = J_L 
            Hrg   = - Hr_L            # Reflect radial flux
            HZg   = HZ_L
            Krr_g = D_rr[1,j]   * J_L 
            KrZ_g = - D_rZ[1,j] * J_L # Reflect K_rZ since it is odd 
        else
            Jg    = J_L 
            Hrg   = Hr_L            
            HZg   = HZ_L
            Krr_g = D_rr[1,j] * J_L 
            KrZ_g = D_rZ[1,j] * J_L 
        end

        F_r[1,j]    = GetFluxGLF( Hrg, Hr_L, Jg, J_L )                # Radial flux of J at i = 1/2
        F_r_Hr[1,j] = GetFluxGLF( Krr_g, D_rr[1,j] * J_L, Hrg, Hr_L ) # Radial flux of H_r at i = 1/2
        F_r_HZ[1,j] = GetFluxGLF( KrZ_g, D_rZ[1,j] * J_L, HZg, HZ_L ) # Radial flux of H_Z at i = 1/2

        ####  Get fluxes at interior i-values  #######################################
        @inbounds @simd for i in dN_rm1
            # Radial flux of J:
            F_r[i+1,j]    = GetFluxGLF( H_r[i,j], H_r[i+1,j], J[i,j], J[i+1,j] )

            # Radial flux of H_r:
            F_r_Hr[i+1,j] = GetFluxGLF( D_rr[i,j] * J[i,j], D_rr[i+1,j] * J[i+1,j],
                                        H_r[i,j],           H_r[i+1,j] )

            # Radial flux of H_Z:
            F_r_HZ[i+1,j] = GetFluxGLF( D_rZ[i,j] * J[i,j], D_rZ[i+1,j] * J[i+1,j],
                                        H_Z[i,j],           H_Z[i+1,j] )
        end

        ####  Outer radial boundary condition at i = N_r + 1/2  ######################

        J_R  = J[N_r, j]
        Hr_R = H_r[N_r, j]
        HZ_R = H_Z[N_r, j]

        if BC_r_right === :Reflect 
            Jg    = J_R 
            Hrg   = - Hr_R 
            HZg   = HZ_R 
            Krr_g = D_rr[N_r,j]    * J_R 
            KrZ_g = - D_rZ[N_r, j] * J_R
        else
            Jg    = J_R 
            Hrg   = Hr_R 
            HZg   = HZ_R 
            Krr_g = D_rr[N_r,j]  * J_R 
            KrZ_g = D_rZ[N_r, j] * J_R
        end

        F_r[N_r+1,j]    = GetFluxGLF( Hr_R, Hrg, J_R, Jg )                 # Radial flux of J at i = N_r + 1/2
        F_r_Hr[N_r+1,j] = GetFluxGLF( D_rr[N_r,j] * J_R, Krr_g, Hr_R, Hrg) # Radial flux of H_r at i = N_r + 1/2
        F_r_HZ[N_r+1,j] = GetFluxGLF( D_rZ[N_r,j] * J_R, KrZ_g, HZ_R, HZg) # Radial flux of H_Z at i = N_r + 1/2

        ####  UPDATE J AND H  ##############################################################
        @inbounds @simd for i in dN_r
            J[i,j]   = UpdateConservativeQuantity( J[i,j], Δt_half, ΔV[i,j], 
                                                   A_r[i,j], A_r[i+1,j], 
                                                   F_r[i,j], F_r[i+1,j], cred )

            H_r[i,j] = UpdateConservativeQuantity( H_r[i,j], Δt_half, ΔV[i,j], 
                                                   A_r[i,j], A_r[i+1,j], 
                                                   F_r_Hr[i,j], F_r_Hr[i+1,j], cred )

            H_Z[i,j] = UpdateConservativeQuantity( H_Z[i,j], Δt_half, ΔV[i,j], 
                                                   A_r[i,j], A_r[i+1,j], 
                                                   F_r_HZ[i,j], F_r_HZ[i+1,j], cred )              
        end
    end

    # Ensure positive J:
    J .= max.( J, 0.0 )

    return nothing
end


end # End of module 
