
module PoissonGravitySolver
""" MODULE:  PoissonGravitySolver

    PURPOSE: Contains the solver for the Poisson equation for 
             gravity, ∇²Φ = 4πGρ. Uses a Successive Over-Relaxation 
             method, with red-black ordering to allow for parallelization.

"""

__precompile__()

include("./ConstantsAndFunctions.jl")
using .ConstantsAndFunctions
using  Base.Threads           # For CPU acceleration



export CreatePoissonCoefficients, PoissonGravitySolver!, Update_∇Φ!


""" This function creates coefficients that enter the discretized 
    Poisson equation. With these coefficients created, they do not 
    need to be re-computed every time-step and iteration. """
function CreatePoissonCoefficients( A_r::AbstractArray{T, 2},
                                    A_Z::AbstractArray{T, 2},
                                 inv_ΔV::AbstractArray{T, 2},
                                   rCoM::AbstractArray{T},
                                   ZCoM::AbstractArray{T},
                                      r::AbstractArray{T},
                                      Z::AbstractArray{T},
                                      )  where {T<:AbstractFloat}

    N_r  = length( rCoM )
    N_Z  = length( ZCoM )
    dN_r = 1:N_r
    dN_Z = 1:N_Z

    # Initialize gravitational potential and its gradient components:
    Φ           = zeros( T, N_r, N_Z )
    ∂Φ∂r        = zeros( T, N_r, N_Z )
    ∂Φ∂Z        = zeros( T, N_r, N_Z )

    # Initialize coefficients for the Poisson equation:
    RPoiss_ijp  = zeros( T, N_r, N_Z ) 
    RPoiss_ij   = zeros( T, N_r, N_Z ) 
    RPoiss_ijm  = zeros( T, N_r, N_Z )
    ZPoiss_ijp  = zeros( T, N_r, N_Z ) 
    ZPoiss_ij   = zeros( T, N_r, N_Z ) 
    ZPoiss_ijm  = zeros( T, N_r, N_Z )    
    
    # Estimate the Jacobi radius ρ_Jacobi, needed for optimal convergence in 
    # the Chebychev-accelerated SOR solver:
    Δr_av        = sum( diff(r) ) / N_r   # Average r-spacing 
    ΔZ_av        = sum( diff(Z) ) / N_Z   # Average Z-spacing 

    ρJacobi      = ( cos(π/(N_r-1))  +  ((Δr_av/ΔZ_av)^2) * cos(π/(N_Z-1)) ) / ( 1 + (Δr_av/ΔZ_av)^2 )
    ρJacobi_sq   = (ρJacobi^2)   

    # Fill the coefficients in the interior cells for the Poisson equation: 
    @threads for j in dN_Z 
        @simd for i in dN_r
            @inbounds begin 
                # r-direction terms:
                if i < N_r 
                    RPoiss_ijp[i,j] = A_r[i+1,j] * inv_ΔV[i,j] / (rCoM[i+1] - rCoM[i])
                else 
                    RPoiss_ijp[i,j] = 0.0
                end
                if i > 1
                    RPoiss_ijm[i,j] = A_r[i,j]   * inv_ΔV[i,j] / (rCoM[i]   - rCoM[i-1])
                else
                    RPoiss_ijm[i,j] = 0.0
                end
    
                RPoiss_ij[i,j]  = RPoiss_ijp[i,j]  +  RPoiss_ijm[i,j]

                # Z-direction terms:
                if j < N_Z 
                    ZPoiss_ijp[i,j] = A_Z[i,j+1] * inv_ΔV[i,j] / (ZCoM[j+1] - ZCoM[j])
                else
                    ZPoiss_ijp[i,j] = 0.0
                end
                if j > 1
                    ZPoiss_ijm[i,j] = A_Z[i,j]   * inv_ΔV[i,j] / (ZCoM[j] - ZCoM[j-1])
                else
                    ZPoiss_ijm[i,j] = 0.0
                end

                ZPoiss_ij[i,j]  = ZPoiss_ijp[i,j]  +  ZPoiss_ijm[i,j]
            end
        end
    end

    return Φ, ∂Φ∂r, ∂Φ∂Z, RPoiss_ijp, RPoiss_ij, RPoiss_ijm, ZPoiss_ijp, ZPoiss_ij, ZPoiss_ijm, ρJacobi_sq
end


const Four_π_G::Float64 = Four_π * G   # 4πG
const max_iter::Int     = 10000        # Maximum number of iterations 
const tol::Float64      = 1e-3         # Maixmum relative residual tolerance 
const check_every::Int  = 20           # Check for convergence every 20 step (by default)

@views function PoissonGravitySolver!( Φ::AbstractArray{T, 2},
                                       ρ::AbstractArray{T, 2},
                              RPoiss_ijp::AbstractArray{T, 2},
                               RPoiss_ij::AbstractArray{T, 2},
                              RPoiss_ijm::AbstractArray{T, 2},
                              ZPoiss_ijp::AbstractArray{T, 2},
                               ZPoiss_ij::AbstractArray{T, 2},
                              ZPoiss_ijm::AbstractArray{T, 2},
                              ρJacobi_sq::T                        # ρJacobi^2 / 4, precomputed 
                                       ) where {T<:AbstractFloat}

    N_r, N_Z        = size( Φ )
    dN_r            = 1:(N_r-1)     # Interior r-interval (i = N_r is fixed by BC) 
    dN_Z            = 1:(N_Z-1)     # Interior Z-interval (j = N_Z is fixed by BC)
    Four_π_G_corr   = T( Four_π_G ) # Ensure correct type for 4πG

    # Scale for convergence check:
    Four_π_G_max    = Four_π_G_corr * maximum( ρ )

    # Enforce Dirichlet values once before iteration (and keep them fixed)
    Φ[N_r, :]      .= 0
    Φ[:, N_Z]      .= 0
    
    # Iterate until convergence or convergence failure:
    iter            = 0       # Initialize iteration counter 
    Converged::Bool = false   # Initialize convergence check 
    #ω  =  T( 2 / ( 1  +  sin(π / max(N_r, N_Z)) ) )
    ω               = T( 2 / ( 1  +  sqrt(1 - ρJacobi_sq) ) )

    while iter < max_iter && !Converged

        # Update iteration count:
        iter += 1

        #### BLACK SWEEP: Even (i + j)
        @threads for j in dN_Z 
            @simd for i in dN_r 
                @inbounds begin 

                    if iseven(i+j)
                        # Ensure that ghost points are mirrored:
                        ip1  = i+1         # Note: ip1 ≤ N_r for the values of i considered  
                        jp1  = j+1         # Note: jp1 ≤ N_Z for the values of j considered 
                        im1  = max(i-1, 1)
                        jm1  = max(j-1, 1)

                        # Gauss-Seidel update:
                        Φ_GS   = ( - Four_π_G_corr * ρ[i,j]  
                                   + RPoiss_ijp[i,j] * Φ[ip1,j]  +  RPoiss_ijm[i,j] * Φ[im1,j]
                                   + ZPoiss_ijp[i,j] * Φ[i,jp1]  +  ZPoiss_ijm[i,j] * Φ[i,jm1] ) / ( RPoiss_ij[i,j] + ZPoiss_ij[i,j] )

                        # SOR over-relaxation:
                        Φ[i,j] = (1 - ω) * Φ[i,j]  +  ω * Φ_GS
                    end

                end # End of inbounds 
            end # End of i-loop 
        end # End of j-loop 

        #### RED SWEEP: Odd (i + j)

         @threads  for j in dN_Z 
            @simd for i in dN_r 
                @inbounds begin 

                    if isodd(i+j)
                        # Ensure that ghost points are mirrored:
                        ip1  = min(i+1, N_r)
                        jp1  = min(j+1, N_Z)
                        im1  = max(i-1, 1)
                        jm1  = max(j-1, 1)

                        # Gauss-Seidel update:
                        Φ_GS   = ( - Four_π_G_corr * ρ[i,j]  
                                   + RPoiss_ijp[i,j] * Φ[ip1,j]  +  RPoiss_ijm[i,j] * Φ[im1,j]
                                   + ZPoiss_ijp[i,j] * Φ[i,jp1]  +  ZPoiss_ijm[i,j] * Φ[i,jm1] ) / ( RPoiss_ij[i,j] + ZPoiss_ij[i,j] )

                        # SOR over-relaxation:
                        Φ[i,j] = (1 - ω) * Φ[i,j]  +  ω * Φ_GS
                    end

                end # End of inbounds 
            end # End of i-loop 
        end # End of j-loop 


        # Check convergence:
        if iter % check_every == 0
            Residual_max = zero(T) 

            for j in 2:(N_Z-1)
                #for i in dN_r
                for i in 2:(N_r-1)
                    @inbounds begin
                        # Ensure that ghost points are mirrored:
                        ip1  = min(i+1, N_r)
                        jp1  = min(j+1, N_Z)
                        im1  = max(i-1, 1)
                        jm1  = max(j-1, 1)

                        # Compute relative residual ≡ (∇²Φ - 4πGρ)/4πGρ at (i,j):
                        Residual = abs(  RPoiss_ijp[i,j] * Φ[ip1,j]  -  RPoiss_ij[i,j] * Φ[i,j]  +  RPoiss_ijm[i,j] * Φ[im1,j]
                                       + ZPoiss_ijp[i,j] * Φ[i,jp1]  -  ZPoiss_ij[i,j] * Φ[i,j]  +  ZPoiss_ijm[i,j] * Φ[i,jm1]
                                       - Four_π_G_corr * ρ[i,j]  ) / ( Four_π_G_corr * ρ[i,j] )

                        # Update maximum residual found:
                        Residual_max = max( Residual_max, Residual )
                    end
                end
            end

            # Flag as converged if residual is within tolerance:
            if Residual_max < T(tol)
                Converged = true 
                println(iter)
            end
            #println(Residual_max)
        end # End of convergence check 

    end # End of iterations 

    if !Converged
        @warn "Poisson solver did not converge after $(max_iter) iterations! "
    end

end


@fastmath function Update_∇Φ!( ∂Φ∂r::AbstractArray{T, 2}, 
                               ∂Φ∂Z::AbstractArray{T, 2},
                                  Φ::AbstractArray{T, 2},
                               rCoM::AbstractArray{T},
                               ZCoM::AbstractArray{T} ) where {T<:AbstractFloat}
    N_r, N_Z = size( Φ )
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z 
    dN_rm    = 2:(N_r-1)

    ##################################################################
    #######                                                    #######
    #######    S T E P  1  :  C O M P U T E  ∂Φ/∂r             #######
    #######                                                    #######
    ##################################################################

    @threads for j in dN_Z

        @inbounds begin 
            # Boundary i = 1: Second-order accurate forward difference:
            h1 = rCoM[2] - rCoM[1]
            h2 = rCoM[3] - rCoM[2] 
            A0 = - (2 * h1  +  h2) / (h1 * (h1 + h2))
            A1 =   (h1 + h2)       / (h1 * h2)
            A2 = -  h1             / (h2 * (h1 + h2))
        
            ∂Φ∂r[1,j] = A0 * Φ[1,j]  +  A1 * Φ[2,j]  +  A2 * Φ[3,j]

            # Interior points (2 ≤ i ≤ N_r): Second-order generalized 
            # central finite difference:
            @simd for i in dN_rm 
                # Coefficients for the second-order differences:
                Δm  = rCoM[i]   - rCoM[i-1]
                Δp  = rCoM[i+1] - rCoM[i]
                am  = - Δp / (Δm * (Δm + Δp))
                a0  = ( Δp - Δm ) / (Δm * Δp)
                ap  = Δm / (Δp * (Δm + Δp))
                
                # Estimate of derivative: 
                ∂Φ∂r[i,j] = ap * Φ[i+1,j]  +  a0 * Φ[i,j]  +  am * Φ[i-1,j]
            end

            # Boundary i = N_r: Second-order accurate backward difference:

            h1 = rCoM[N_r] - rCoM[N_r-1]
            h2 = rCoM[N_r-1] - rCoM[N_r-2] 
            B0 = (2 * h1  +  h2) / (h1 * (h1 + h2))
            B1 = - (h1 + h2)     / (h1 * h2)
            B2 =  h1             / (h2 * (h1 + h2)) 
        
            ∂Φ∂r[N_r,j] = B0 * Φ[N_r,j]  +  B1 * Φ[N_r-1,j]  +  B2 * Φ[N_r-2,j]

        end # End of inbounds 
    end # End fo j-loop


    ##################################################################
    #######                                                    #######
    #######    S T E P  2  :  C O M P U T E  ∂Φ/∂Z             #######
    #######                                                    #######
    ##################################################################

    @threads for j in dN_Z
        @inbounds begin
            if j == 1
                # Lower boundary: Use forward difference:
                h1 = ZCoM[2] - ZCoM[1]
                h2 = ZCoM[3] - ZCoM[2]
                c0 = - (2 * h1  +  h2) / (h1 * (h1 + h2))
                c1 =   (h1 + h2)       / (h1 * h2)
                c2 = -  h1             / (h2 * (h1 + h2))

                @simd for i in dN_r 
                    ∂Φ∂Z[i,1] = c0 * Φ[i,1]  +  c1 * Φ[i,2]  +  c2 * Φ[i,3] 
                end
            elseif j == N_Z 
                # Upper boundary: Use backward difference:
                h1 = ZCoM[N_Z]   - ZCoM[N_Z-1]
                h2 = ZCoM[N_Z-1] - ZCoM[N_Z-2]
                d0 =   (2 * h1 + h2) / (h1 * (h1 + h2))
                d1 = - (h1 + h2)     / (h1 * h2)
                d2 =    h1           / (h2 * (h1 + h2))

                @simd for i in dN_r
                    ∂Φ∂Z[i,N_Z] = d0 * Φ[i,N_Z]  +  d1 * Φ[i,N_Z-1]  +  d2 * Φ[i,N_Z-2]
                end
            else
                # Interior points: Use generalized central difference:
                Δm = ZCoM[j]   - ZCoM[j-1]
                Δp = ZCoM[j+1] - ZCoM[j]
                bm = -  Δp        / (Δm * (Δm + Δp))
                b0 =  ( Δp - Δm ) / (Δm * Δp)
                bp =    Δm        / (Δp * (Δm + Δp))

                @simd for i in dN_r
                    ∂Φ∂Z[i,j] = bm * Φ[i,j-1]  +  b0 * Φ[i,j]  +  bp * Φ[i,j+1]
                end
            end    

        end # End of inbounds 
    end # End of j-loop 

    return nothing
end


end # End of module 