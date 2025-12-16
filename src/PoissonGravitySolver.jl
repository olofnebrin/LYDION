
""" FILE:    PoissonGravitySolver.jl

    PURPOSE: Contains the ADI solver for the Poisson equation for 
             gravity, ∇²Φ = 4πGρ. Uses an ADI method with pseudo-time 
             stepping, solving:

             ∂Φ/∂τ = ∇²Φ - 4πGρ

             until it converges to an equilibrium ∇²Φ = 4πGρ. 

"""


using .Structs: GravityState, GravityCoeffs

using .ConstantsAndFunctions  # Useful constants & functions 
using  Base.Threads           # For CPU acceleration


export CreateGravityStateAndCoefficients, 
       Update_Φ!, Update_∇Φ!, ShouldWeUpdateΦ


""" This function creates coefficients that enter the discretized 
    Poisson equation. With these coefficients created, they do not 
    need to be re-computed every time-step and iteration. """
function CreateGravityStateAndCoefficients( A_r::AbstractArray{Float64, 2},
                                            A_Z::AbstractArray{Float64, 2},
                                         inv_ΔV::AbstractArray{Float64, 2},
                                           rCoM::AbstractArray{Float64},
                                           ZCoM::AbstractArray{Float64},
                                              r::AbstractArray{Float64},
                                              Z::AbstractArray{Float64},
                                            ) 

    N_r  = length( rCoM )
    N_Z  = length( ZCoM )
    dN_r = 1:N_r
    dN_Z = 1:N_Z

    # Initialize gravitational potential and its gradient components:
    Φ           = zeros( Float64, N_r, N_Z )
    Φ_half      = zeros( Float64, N_r, N_Z )
    Φ_old       = zeros( Float64, N_r, N_Z )
    Φ_snapshot  = zeros( Float64, N_r, N_Z )
    ∂Φ∂r        = zeros( Float64, N_r, N_Z )
    ∂Φ∂Z        = zeros( Float64, N_r, N_Z )

    # Initialize coefficients for the Poisson equation:
    RPoiss_ijp  = zeros( Float64, N_r, N_Z ) 
    RPoiss_ij   = zeros( Float64, N_r, N_Z ) 
    RPoiss_ijm  = zeros( Float64, N_r, N_Z )
    ZPoiss_ijp  = zeros( Float64, N_r, N_Z ) 
    ZPoiss_ij   = zeros( Float64, N_r, N_Z ) 
    ZPoiss_ijm  = zeros( Float64, N_r, N_Z )    

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

    # Maximum and smallest pseudo-time step
    Δτ_max   = max( r[end]^2, Z[end]^2 ) / 4
    Δτ_min   = min( minimum(diff(r))^2, minimum(diff(Z))^2 ) / 4

    return GravityState(Φ, Φ_half, Φ_old, Φ_snapshot, ∂Φ∂r, ∂Φ∂Z), 
           GravityCoeffs(RPoiss_ijp, RPoiss_ij, RPoiss_ijm, ZPoiss_ijp, ZPoiss_ij, ZPoiss_ijm, Δτ_max, Δτ_min)
end


# Pre-allocate scratch arrays (Ref is used so that we can fill their sizes later):
const scratch_r = Ref{Vector{NTuple{7, Vector{Float64}}}}(Vector{NTuple{7, Vector{Float64}}}())
const scratch_Z = Ref{Vector{NTuple{7, Vector{Float64}}}}(Vector{NTuple{7, Vector{Float64}}}())

""" Initializes the scratch arrays for the ADI updates. """
function InitializeScratchArrays!( N_r::Int,
                                   N_Z::Int
                                  )
    # Number of threads:
    nthreads = Threads.nthreads()

    # For the r-sweep:
    scratch_r[] = [
        (Vector{Float64}(undef, N_r-1),   # A
         Vector{Float64}(undef, N_r-1),   # B
         Vector{Float64}(undef, N_r-1),   # C
         Vector{Float64}(undef, N_r-1),   # D
         Vector{Float64}(undef, N_r-1),   # C_prime 
         Vector{Float64}(undef, N_r-1),   # D_prime 
         Vector{Float64}(undef, N_r-1))   # x (solution)
        for _ in 1:nthreads
    ]

    # For the Z-sweep:
    scratch_Z[] = [
        (Vector{Float64}(undef, N_Z-1), # A
         Vector{Float64}(undef, N_Z-1), # B 
         Vector{Float64}(undef, N_Z-1), # C 
         Vector{Float64}(undef, N_Z-1), # D 
         Vector{Float64}(undef, N_Z-1), # C_prime 
         Vector{Float64}(undef, N_Z-1), # D_prime
         Vector{Float64}(undef, N_Z-1)) # x (solution)
        for _ in 1:nthreads
    ]
end


const P_0::Float64 = 1.0

@inline @fastmath function Get_Pℓ( x::Float64 )

    x2  = x  * x 
    x4  = x2 * x2  
    x6  = x4 * x2
    x8  = x6 * x2
    x10 = x8 * x2

    P_2  = 0.5     * ( 3 * x2      -  1 )
    P_4  = (1/8)   * ( 35 * x4     -  30 * x2     +  3 )
    P_6  = (1/16)  * ( 231 * x6    -  315 * x4    +  105 * x2   -  5)
    P_8  = (1/128) * ( 6435 * x8   -  12012 * x6  +  6930 * x4  -  1260 * x2  +  35 )
    P_10 = (1/256) * ( 46189 * x10 -  109395 * x8 +  90090 * x6 -  30030 * x4 + 3465 * x2  -  63 )

    return P_0, P_2, P_4, P_6, P_8, P_10
end


function Set_Φ_BC!( Φ::AbstractArray{Float64, 2},
                    ρ::AbstractArray{Float64, 2},
                   ΔV::AbstractArray{Float64, 2},
                 rCoM::AbstractArray{Float64, 1},
                 ZCoM::AbstractArray{Float64, 1}
                    )

    N_r, N_Z = size( ρ )
    dN_r     = 1:N_r 
    dN_Z     = 1:N_Z 

    Q_0::Float64 = 0.0; Q_2::Float64  = 0.0
    Q_4::Float64 = 0.0; Q_6::Float64  = 0.0
    Q_8::Float64 = 0.0; Q_10::Float64 = 0.0
    for j in dN_Z 
        for i in dN_r 
            @inbounds begin

                R   = sqrt( rCoM[i] * rCoM[i]  +  ZCoM[j] * ZCoM[j] )
                R2  = R  * R 
                R4  = R2 * R2 
                R6  = R4 * R2 
                R8  = R6 * R2
                R10 = R8 * R2

                # Legendre polynomials Pℓ(x) with x = ZCoM / R:
                x    = ZCoM[j] / R 
                P_0, P_2, P_4, P_6, P_8, P_10 = Get_Pℓ(x)


                Q_0  += ρ[i,j] * P_0 * ΔV[i,j]
                Q_2  += ρ[i,j] * R2 * P_2 * ΔV[i,j]
                Q_4  += ρ[i,j] * R4 * P_4 * ΔV[i,j]
                Q_6  += ρ[i,j] * R6 * P_6 * ΔV[i,j]
                Q_8  += ρ[i,j] * R8 * P_8 * ΔV[i,j]
                Q_10 += ρ[i,j] * R10 * P_10 * ΔV[i,j]
            end
        end
    end

    # Correct by a factor 2 if only the upper Z-plane is simulated,
    # in order to take into account mass below the Z = 0 plane:
    Q_0 *= 2; Q_2  *= 2
    Q_4 *= 2; Q_6  *= 2
    Q_8 *= 2; Q_10 *= 2

    # Outer r-boundary, i = N_r 
    @simd for j in dN_Z 
        @inbounds begin
            R   = sqrt( rCoM[N_r] * rCoM[N_r]  +  ZCoM[j] * ZCoM[j] )
            R3  = R  * R * R
            R5  = R3 * R * R
            R7  = R5 * R * R
            R9  = R7 * R * R
            R11 = R9 * R * R

            # Legendre polynomials Pℓ(x) with x = ZCoM / R:
            x   = ZCoM[j] / R
            P_0, P_2, P_4, P_6, P_8, P_10 = Get_Pℓ(x)

            # Multipole expnaded potential at the i = N_r boundary:
            Φ[N_r,j] = - G * ( P_0  * Q_0  / R
                             + P_2  * Q_2  / R3 
                             + P_4  * Q_4  / R5 
                             + P_6  * Q_6  / R7
                             + P_8  * Q_8  / R9
                             + P_10 * Q_10 / R11 )
        end
    end

    # Outer Z-boundary, j = N_Z 
    @simd for i in dN_r 
        @inbounds begin
            R   = sqrt( rCoM[i] * rCoM[i]  +  ZCoM[N_Z] * ZCoM[N_Z] )
            R3  = R  * R * R
            R5  = R3 * R * R
            R7  = R5 * R * R
            R9  = R7 * R * R
            R11 = R9 * R * R

            # Legendre polynomials Pℓ(x) with x = ZCoM / R:
            x   = ZCoM[N_Z] / R
            P_0, P_2, P_4, P_6, P_8, P_10 = Get_Pℓ(x)

            # Multipole expnaded potential at the i = N_r boundary:
            Φ[i,N_Z] = - G * ( P_0  * Q_0  / R
                             + P_2  * Q_2  / R3 
                             + P_4  * Q_4  / R5 
                             + P_6  * Q_6  / R7
                             + P_8  * Q_8  / R9
                             + P_10 * Q_10 / R11 )
        end
    end

    return nothing 
end


const Four_π_G::Float64 = Four_π * G   # 4πG
const tol_Grav::Float64 = 1e-5         # Maximum relative residual tolerance 
const tol_ΔΦ::Float64   = 10 * eps()   # If the relative change in Φ is less than ∼ 10⁻¹⁵ (near round-off errors) we also deem it converged
const check_every::Int  = 10           # Check for convergence every 10 step (by default)

@views function Update_Φ_ADI!( Φ::AbstractArray{Float64, 2},
                          Φ_half::AbstractArray{Float64, 2},
                           Φ_old::AbstractArray{Float64, 2},
                               ρ::AbstractArray{Float64, 2},
                      RPoiss_ijp::AbstractArray{Float64, 2},
                       RPoiss_ij::AbstractArray{Float64, 2},
                      RPoiss_ijm::AbstractArray{Float64, 2},
                      ZPoiss_ijp::AbstractArray{Float64, 2},
                       ZPoiss_ij::AbstractArray{Float64, 2},
                      ZPoiss_ijm::AbstractArray{Float64, 2},
                          Δτ_max::Float64,
                          Δτ_min::Float64,
                          k_iter::Int,
                              ΔV::AbstractArray{Float64, 2}, 
                            rCoM::AbstractArray{Float64, 1},
                            ZCoM::AbstractArray{Float64, 1}  
                               )

    # Grid size:
    N_r, N_Z   = size( Φ )
    # Intervals:
    dN_r       = 1:(N_r-1) # All cells except outer boundary 
    dN_Z       = 1:(N_Z-1) # All cells except outer boundary 
    dN_r_int   = 2:(N_r-2) # Exclude inner & outer boundary
    dN_Z_int   = 2:(N_Z-2) # Exclude inner & outer boundary

    # Enforce Dirichlet BC values once before iteration (and keep them fixed)
    Set_Φ_BC!( Φ, ρ, ΔV, rCoM, ZCoM )
    Φ_half .= Φ 

    # Initialize the scratch arrays on the very first call 
    # (or if the thread count changed):
    if isempty(scratch_r[]) || length(scratch_r[]) != Threads.nthreads()
        InitializeScratchArrays!( N_r, N_Z )
    end
    scratch_r_local = scratch_r[]

    # Set up factor for pseudo-time stepping: 
    α_ADI     = (Δτ_min / Δτ_max)^(1/(k_iter-1))

    # Set up iterations:
    k         = 0       # Initialize iteration count 
    Δτ        = Δτ_max  # Initial pseudo-time step
    Δw        = 1 / Δτ  # Convenient parameter 
    Converged = false   # Initialize convergence check  
    while k < k_iter && !Converged

        ##############################################
        #########                            #########
        #########    r-SWEEP: n → n + ⚈      #########
        #########                            #########
        ##############################################

        @threads :static for j in dN_Z 

            tid = Threads.threadid()
            Ar_tot, Br_tot, Cr_tot, Dr_tot, Cprimer_tot, Dprimer_tot, xr = scratch_r_local[tid]

            ### Setup tridiagonal system in r-direction for each Z-index j ###
            @inbounds begin 
                # Clamp j-values:
                jm1 = max(j-1, 1)

                # COEFFICIENTS FOR THE INNER BOUNDARY (i = 1):
                Ar_tot[1]      = 0.0
                Br_tot[1]      = - ( RPoiss_ij[1,j]  +  Δw  -  RPoiss_ijm[1,j] )
                Cr_tot[1]      = RPoiss_ijp[1,j]
                Dr_tot[1]      = ( Four_π_G * ρ[1,j]  -  Φ_old[1,j] * Δw
                               -  (ZPoiss_ijp[1,j]*Φ_old[1,j+1]  -  ZPoiss_ij[1,j]* Φ_old[1,j]  +  ZPoiss_ijm[1,j]*Φ_old[1,jm1]))

                # INTERIOR COEFFICIENTS:
                # Lower diagonal:
                Ar_tot[dN_r_int]  .= RPoiss_ijm[dN_r_int,j]
                # Diagonal:
                Br_tot[dN_r_int]  .= .- ( RPoiss_ij[dN_r_int,j] .+ Δw )
                # Upper diagonal:
                Cr_tot[dN_r_int]  .= RPoiss_ijp[dN_r_int,j]
                # Right-hand side:
                Dr_tot[dN_r_int]  .= ( Four_π_G * ρ[dN_r_int,j]  .-  Φ_old[dN_r_int,j] .* Δw
                                  .-  (ZPoiss_ijp[dN_r_int,j].*Φ_old[dN_r_int,j+1]  .-  ZPoiss_ij[dN_r_int,j].* Φ_old[dN_r_int,j]  .+  ZPoiss_ijm[dN_r_int,j].*Φ_old[dN_r_int,jm1]))

                # NEAR OUTER BOUNDARY (i = N_r - 1), WITH BOUNDARY CONDITION AT i = N_r:
                Ar_tot[N_r-1]      = RPoiss_ijm[N_r-1,j]
                # Diagonal:
                Br_tot[N_r-1]      = - ( RPoiss_ij[N_r-1,j] .+ Δw )
                # Upper diagonal:
                Cr_tot[N_r-1]      = 0.0
                # Right-hand side:
                Dr_tot[N_r-1]      = ( Four_π_G * ρ[N_r-1,j]  -  Φ_old[N_r-1,j] * Δw  -  RPoiss_ijp[N_r-1,j] * Φ[N_r,j]
                                   -  (ZPoiss_ijp[N_r-1,j]*Φ_old[N_r-1,j+1]  -  ZPoiss_ij[N_r-1,j]* Φ_old[N_r-1,j]  +  ZPoiss_ijm[N_r-1,j]*Φ_old[N_r-1,jm1]))


                # Solve tridiagonal system in the r-direction:
                Φ_half[dN_r,j] .= TridiagonalSolve( Ar_tot, Br_tot, Cr_tot, Dr_tot, Cprimer_tot, Dprimer_tot, xr ) 

            end # End of inbounds 
        end # End of j-loop


        ##############################################
        #########                            #########
        #########    Z-SWEEP: n + ⚈ → n + 1  #########
        #########                            #########
        ##############################################

        scratch_Z_local = scratch_Z[]

        @threads :static for i in dN_r
            @inbounds begin

                tid = Threads.threadid()
                AZ_tot, BZ_tot, CZ_tot, DZ_tot, CprimeZ_tot, DprimeZ_tot, xZ = scratch_Z_local[tid]

                # Clamp i-values:
                im1 = max(i-1, 1)

                # COEFFICIENTS FOR THE LOWER BOUNDARY (j = 1):
                AZ_tot[1]      = 0.0
                BZ_tot[1]      = - ( ZPoiss_ij[i,1]  +  Δw  -  ZPoiss_ijm[i,1] )
                CZ_tot[1]      = ZPoiss_ijp[i,1]
                DZ_tot[1]      = ( Four_π_G * ρ[i,1]  -  Φ_half[i,1] * Δw
                               -  (RPoiss_ijp[i,1]*Φ_half[i+1,1]  -  RPoiss_ij[i,1]* Φ_half[i,1]  +  RPoiss_ijm[i,1]*Φ_half[im1,1]) )

                # INTERIOR COEFFICIENTS:
                # Lower diagonal:
                AZ_tot[dN_Z_int]  .= ZPoiss_ijm[i,dN_Z_int]
                # Diagonal:
                BZ_tot[dN_Z_int]  .= .- ( ZPoiss_ij[i,dN_Z_int] .+ Δw )
                # Upper diagonal:
                CZ_tot[dN_Z_int]  .= ZPoiss_ijp[i,dN_Z_int]
                # Right-hand side:
                DZ_tot[dN_Z_int]  .= ( Four_π_G * ρ[i,dN_Z_int]  .-  Φ_half[i,dN_Z_int] .* Δw
                                  .-  (RPoiss_ijp[i,dN_Z_int].*Φ_half[i+1,dN_Z_int]  .-  RPoiss_ij[i,dN_Z_int].* Φ_half[i,dN_Z_int]  .+  RPoiss_ijm[i,dN_Z_int].*Φ_half[im1,dN_Z_int]))

                # NEAR OUTER BOUNDARY (j = N_Z - 1), WITH BOUNDARY CONDITION AT j = N_Z:
                AZ_tot[N_Z-1]      = ZPoiss_ijm[i,N_Z-1]
                # Diagonal:
                BZ_tot[N_Z-1]      = - ( ZPoiss_ij[i,N_Z-1] + Δw )
                # Upper diagonal:
                CZ_tot[N_Z-1]      = 0.0
                # Right-hand side:
                DZ_tot[N_Z-1]      = ( Four_π_G * ρ[i,N_Z-1]  -  Φ_half[i,N_Z-1] * Δw  -  ZPoiss_ijp[i,N_Z-1] * Φ[i,N_Z]
                                   -  (RPoiss_ijp[i,N_Z-1]*Φ_half[i+1,N_Z-1]  -  RPoiss_ij[i,N_Z-1] * Φ_half[i,N_Z-1]  +  RPoiss_ijm[i,N_Z-1]*Φ_half[im1,N_Z-1]))

                # Solve tridiagonal system in the Z-direction:
                Φ[i, dN_Z] .= TridiagonalSolve( AZ_tot, BZ_tot, CZ_tot, DZ_tot, CprimeZ_tot, DprimeZ_tot, xZ ) 
                
            end # End of inbounds 
        end # End of i-loop 


        # Check convergence:
        if k % check_every == 0
            Residual_max = 0.0 
            ΔΦ_max       = 0.0

            for j in dN_Z_int
                for i in dN_r_int
                    @inbounds begin
                        # Ensure that ghost points are mirrored:
                        ip1  = min(i+1, N_r)
                        jp1  = min(j+1, N_Z)
                        im1  = max(i-1, 1)
                        jm1  = max(j-1, 1)

                        # Scale that determines relative residual if ρ happens to be very small:
                        #SafetyScale = abs( RPoiss_ij[i,j] + ZPoiss_ij[i,j] ) * abs( Φ[i,j] )

                        # Compute relative residual ≡ (∇²Φ - 4πGρ)/4πGρ at (i,j):
                        Residual = abs(  RPoiss_ijp[i,j] * Φ[ip1,j]  -  RPoiss_ij[i,j] * Φ[i,j]  +  RPoiss_ijm[i,j] * Φ[im1,j]
                                       + ZPoiss_ijp[i,j] * Φ[i,jp1]  -  ZPoiss_ij[i,j] * Φ[i,j]  +  ZPoiss_ijm[i,j] * Φ[i,jm1]
                                       - Four_π_G * ρ[i,j]  ) / ( Four_π_G * ρ[i,j] )

                        # Compute relative change (Φ - Φ_old) / Φ_old:
                        ΔΦ       = abs( Φ[i,j] - Φ_old[i,j] ) / abs( Φ_old[i,j] )

                        # Update maximum residual & relative change found:
                        Residual_max = max( Residual_max, Residual )
                        ΔΦ_max       = max( ΔΦ_max, ΔΦ )
                    end
                end
            end

            # Flag as converged if residual is within tolerance:
            if Residual_max < tol_Grav || ΔΦ_max < tol_ΔΦ
                Converged = true 
            end
        end # End of convergence check

        # Update iteration:
        k     += 1

        # Update Φ_old:
        Φ_old .= Φ

        # Update the pseudo-time step:
        Δτ     = (α_ADI^k) * Δτ_max
        Δw     = 1 / Δτ

    end # End of pseudo-time iterations

    return Converged
end


const MaxAttempt::Int = 8

function Update_Φ!( Φ::AbstractArray{Float64, 2},
               Φ_half::AbstractArray{Float64, 2},
                Φ_old::AbstractArray{Float64, 2},
           Φ_snapshot::AbstractArray{Float64, 2},
                    ρ::AbstractArray{Float64, 2},
           RPoiss_ijp::AbstractArray{Float64, 2},
            RPoiss_ij::AbstractArray{Float64, 2},
           RPoiss_ijm::AbstractArray{Float64, 2},
           ZPoiss_ijp::AbstractArray{Float64, 2},
            ZPoiss_ij::AbstractArray{Float64, 2},
           ZPoiss_ijm::AbstractArray{Float64, 2},
               Δτ_max::Float64,
               Δτ_min::Float64,
                   ΔV::AbstractArray{Float64, 2}, 
                 rCoM::AbstractArray{Float64, 1},
                 ZCoM::AbstractArray{Float64, 1}  
                    )

    N_r, N_Z    = size( ρ )
    k_iter      = Int( round( max( N_r, N_Z ) / 2 ) )
    k_iter      = max( k_iter, 2 )  # Ensure that k_iter - 1 ≥ 1 
    
    # Store snapshot of Φ from previous converged time-step:
    Φ_snapshot .= Φ

    Converged::Bool = false 
    Attempt::Int    = 0
    while !Converged && Attempt < MaxAttempt
        # Reset Φ to last converged value:
        Φ        .= Φ_snapshot
        Φ_old    .= Φ_snapshot

        # Do a set of ADI iterations & check for convergence:
        Converged = Update_Φ_ADI!( Φ, Φ_half, Φ_old, ρ, 
                                   RPoiss_ijp, RPoiss_ij, RPoiss_ijm,
                                   ZPoiss_ijp, ZPoiss_ij, ZPoiss_ijm,
                                   Δτ_max, Δτ_min, k_iter,
                                   ΔV, rCoM, ZCoM )

        if !Converged
            # Update ADI attempt count:
            Attempt  += 1

            # Update number of iterations in the ADI solver for next attempt
            # by a factor 2:
            k_iter   *= 2
        end
    end

    if !Converged
        error( "Gravity solver could not converge over $(MaxAttempt) sets of ADI iteration cycles." )
    end

    return nothing 
end


@fastmath function Update_∇Φ!( ∂Φ∂r::AbstractArray{Float64, 2}, 
                               ∂Φ∂Z::AbstractArray{Float64, 2},
                                  Φ::AbstractArray{Float64, 2},
                               rCoM::AbstractArray{Float64},
                               ZCoM::AbstractArray{Float64} ) 
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


""" This function checks whether the residual for the updated 
    density field is large enough that we *should* solve the 
    Poisson equation again. Returns true if we should, false otherwise. """
function ShouldWeUpdateΦ( Φ::AbstractArray{Float64, 2},
                          ρ::AbstractArray{Float64, 2},
                 RPoiss_ijp::AbstractArray{Float64, 2},
                  RPoiss_ij::AbstractArray{Float64, 2},
                 RPoiss_ijm::AbstractArray{Float64, 2},
                 ZPoiss_ijp::AbstractArray{Float64, 2},
                  ZPoiss_ij::AbstractArray{Float64, 2},
                 ZPoiss_ijm::AbstractArray{Float64, 2}
                         )

    N_r, N_Z = size( Φ )
    dN_r     = 1:(N_r-1)
    dN_Z     = 1:(N_Z-1) 

    ShouldWeUpdate::Bool = false  
    for j in dN_Z 
        for i in dN_r 
            @inbounds begin 
                # Ensure that ghost points are mirrored:
                ip1  = min(i+1, N_r)
                jp1  = min(j+1, N_Z)
                im1  = max(i-1, 1)
                jm1  = max(j-1, 1)

                # Compute Φ':
                Φ_prime  = ( - Four_π_G * ρ[i,j]  
                             + RPoiss_ijp[i,j] * Φ[ip1,j]  +  RPoiss_ijm[i,j] * Φ[im1,j]
                             + ZPoiss_ijp[i,j] * Φ[i,jp1]  +  ZPoiss_ijm[i,j] * Φ[i,jm1] ) / ( RPoiss_ij[i,j] + ZPoiss_ij[i,j] )

                # Compute measure of error and decide to update if it is large enough:
                if abs( Φ_prime - Φ[i,j] ) > tol_Grav * abs(Φ[i,j])
                    ShouldWeUpdate = true 
                    return ShouldWeUpdate 
                end

            end # End of inbounds
        end # End of i-loop 
    end # End of j-loop 

    return ShouldWeUpdate
end

