
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
                                           ℓmax::Int    = 10,
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

    # Array for storing non-gas mass density (e.g. star and/or dark matter):
    ρ_bg        = zeros( Float64, N_r, N_Z )

    # Array for storing total mass density:
    ρtot        = zeros( Float64, N_r, N_Z )

    # Arrays for tracking the gravitational force per unit volume, - ρ ∇Φ:
    f_gravity_r = zeros( Float64, N_r, N_Z )
    f_gravity_Z = zeros( Float64, N_r, N_Z )

    # Vectors for storing Legendre polynomials & multipole moments (for use in BCs)
    P           = zeros( Float64, ℓmax + 1 )
    Q           = zeros( Float64, ℓmax + 1 )

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

    return GravityState(Φ, Φ_half, Φ_old, Φ_snapshot, ∂Φ∂r, ∂Φ∂Z, ρtot, ρ_bg, f_gravity_r, f_gravity_Z, P, Q, ℓmax), 
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


""" This function computes and stores the Legendre polynomials up 
    to degree ℓmax, evaluated at x. Useful in the multipole expansion
    of the BCs. """
@inline @fastmath function Get_Pℓ!( P::AbstractArray{Float64, 1},
                                    x::Float64,
                                 ℓmax::Int )
    @inbounds begin
        P[1] = 1.0
        if ℓmax == 0
            return nothing 
        end

        P[2] = x 
        if ℓmax == 1
            return nothing 
        end

        # Recurrence to get the other Legendre polynomials:
        for ℓ in 1:(ℓmax-1)
            P[ℓ+2] = ( (2 * ℓ + 1) * x * P[ℓ+1]  -  ℓ * P[ℓ] ) / ( ℓ + 1 )
        end
    end

    return nothing 
end


""" This function computes the multipole moments:
    
    Qℓ ≡ ∫dV R^ℓ Pℓ(x) 
       ≈ ∑_{i,j} ΔV_{i,j} (R_{i,j})^ℓ Pℓ(x_{i,j}),
       
    where x = Z / R. """
function ComputeMultiPoleMoments!( Q::AbstractArray{Float64, 1},
                                   P::AbstractArray{Float64, 1},
                                ρtot::AbstractArray{Float64, 2},
                                rCoM::AbstractArray{Float64, 1},
                                ZCoM::AbstractArray{Float64, 1},
                                  ΔV::AbstractArray{Float64, 2},
                                ℓmax::Int,
                            BC_Z_bot::Symbol = :Reflect
                                 )


    @inbounds begin
        # Reset Qℓ:
        Q .= 0.0

        # Grid size & intervals:
        N_r, N_Z = size( ρtot )
        dN_r     = 1:N_r 
        dN_Z     = 1:N_Z

        if BC_Z_bot == :Reflect
            # If we have mirror symmetry around Z = 0, we use multipole 
            # expansion around Z = 0:
            zcenter = 0.0
        else
            # If no symmetry, we expand around the center along the Z-axis:
            zcenter = 0.5 * ( ZCoM[end] + ZCoM[1] )
        end

        for j in dN_Z

            zcom = ZCoM[j]

            for i in dN_r 

                rcom = rCoM[i]
                R    = sqrt( rcom * rcom  +  (zcom - zcenter) * (zcom - zcenter) )
                x    = ( zcom - zcenter ) / R

                # Compute all Legendre polynomials up to ℓmax:
                Get_Pℓ!( P, x, ℓmax )

                ρΔV    = ρtot[i,j] * ΔV[i,j]
                Rpower = 1.0   # We start with R^(ℓ = 0) = 1
                for ℓ in 0:ℓmax 
                    Q[ℓ+1] += ρΔV * Rpower * P[ℓ+1]

                    # Update power of R:
                    Rpower *= R
                end
            end
        end

        if BC_Z_bot == :Reflect
            # Mirror symmetry around Z = 0: In this case, the moment has to 
            # be doubled because of the mass below Z = 0
            for ℓ in 0:ℓmax 
                Q[ℓ+1] *= 2
            end
        end
        
    end
    
    return nothing 
end


function Set_Φ_BC!( Φ::AbstractArray{Float64, 2},
                    P::AbstractArray{Float64, 1},
                    Q::AbstractArray{Float64, 1},
                 rCoM::AbstractArray{Float64, 1},
                 ZCoM::AbstractArray{Float64, 1},
                 ℓmax::Int    = 10,
             BC_Z_bot::Symbol = :Reflect
                    )

    N_r, N_Z = size( Φ )
    dN_r     = 1:N_r 
    dN_Z     = 1:N_Z

    if BC_Z_bot == :Reflect
        # If we have mirror symmetry around Z = 0, we use multipole 
        # expansion around Z = 0:
        zcenter = 0.0
    else
        # If no symmetry, we expand around the center along the Z-axis:
        zcenter = 0.5 * ( ZCoM[end] + ZCoM[1] )
    end

    # OUTER r-BOUNDARY i = N_r:
    for j in dN_Z 
        @inbounds begin
            R   = sqrt( rCoM[N_r] * rCoM[N_r]  +  (ZCoM[j] - zcenter) * (ZCoM[j] - zcenter) )
            x   = ( ZCoM[j] - zcenter ) / R 

            Get_Pℓ!( P, x, ℓmax ) 
            inv_R      = 1 / R 
            inv_Rpower = inv_R     # 1 / R^(ℓ+1), starting at ℓ = 0
            ∑PQ_over_R = 0.0
            for ℓ in 0:ℓmax 
                ∑PQ_over_R += P[ℓ+1] * Q[ℓ+1] * inv_Rpower

                # Update power of R:
                inv_Rpower *= inv_R
            end

            # Put it all together to get the boundary potential:
            Φ[N_r,j] = - G * ∑PQ_over_R
        end
    end

    # OUTER Z-BOUNDARY j = N_Z:
    for i in dN_r 
        @inbounds begin
            R   = sqrt( rCoM[i] * rCoM[i]  +  (ZCoM[N_Z] - zcenter) * (ZCoM[N_Z] - zcenter) )
            x   = ( ZCoM[N_Z] - zcenter ) / R 

            Get_Pℓ!( P, x, ℓmax ) 
            inv_R      = 1 / R 
            inv_Rpower = inv_R     # 1 / R^(ℓ+1), starting at ℓ = 0
            ∑PQ_over_R = 0.0
            for ℓ in 0:ℓmax 
                ∑PQ_over_R += P[ℓ+1] * Q[ℓ+1] * inv_Rpower

                # Update power of R:
                inv_Rpower *= inv_R
            end

            # Put it all together to get the boundary potential:
            Φ[i,N_Z] = - G * ∑PQ_over_R
        end
    end

    # LOWER Z-BOUNDARY j = 1 (only relevant if we don't assume mirror symmetry):
    if BC_Z_bot != :Reflect
        for i in dN_r 
            @inbounds begin
                R   = sqrt( rCoM[i] * rCoM[i]  +  (ZCoM[1] - zcenter) * (ZCoM[1] - zcenter) )
                x   = ( ZCoM[1] - zcenter ) / R 

                Get_Pℓ!( P, x, ℓmax ) 
                inv_R      = 1 / R 
                inv_Rpower = inv_R     # 1 / R^(ℓ+1), starting at ℓ = 0
                ∑PQ_over_R = 0.0
                for ℓ in 0:ℓmax 
                    ∑PQ_over_R += P[ℓ+1] * Q[ℓ+1] * inv_Rpower

                    # Update power of R:
                    inv_Rpower *= inv_R
                end

                # Put it all together to get the boundary potential:
                Φ[i,1] = - G * ∑PQ_over_R
            end
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
                            ρtot::AbstractArray{Float64, 2},
                               ρ::AbstractArray{Float64, 2}, # Gass density 
                              ρd::AbstractArray{Float64, 2}, # Dust density 
                            ρ_bg::AbstractArray{Float64, 2}, # Background density (e.g. stars, or DM)
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
                            ZCoM::AbstractArray{Float64, 1},
                               Q::AbstractArray{Float64, 1},
                               P::AbstractArray{Float64, 1},
                            ℓmax::Int    = 10,
                        BC_Z_bot::Symbol = :Reflect
                               )

    # Grid size:
    N_r, N_Z   = size( Φ )
    # Intervals:
    dN_r       = 1:(N_r-1) # All cells except outer boundary 
    dN_Z       = 1:(N_Z-1) # All cells except outer boundary 
    dN_r_int   = 2:(N_r-2) # Exclude inner & outer boundary
    dN_Z_int   = 2:(N_Z-2) # Exclude inner & outer boundary

    # Fill total gravitating mass:
    ρtot      .= ρ  .+  ρd  .+  ρ_bg

    # Enforce Dirichlet BC values once before iteration (and keep them fixed)
    ComputeMultiPoleMoments!( Q, P, ρtot, rCoM, ZCoM, ΔV, ℓmax, BC_Z_bot )
    Set_Φ_BC!( Φ, P, Q, rCoM, ZCoM, ℓmax, BC_Z_bot )
    Φ_half .= Φ 
    Φ_old  .= Φ

    # Initialize the scratch arrays on the very first call 
    # (or if the thread count changed):
    if isempty(scratch_r[]) || length(scratch_r[]) != Threads.nthreads()
        InitializeScratchArrays!( N_r, N_Z )
    end
    scratch_r_local = scratch_r[]

    # Set up factor for pseudo-time stepping: 
    α_ADI     = (Δτ_min / Δτ_max)^(1/(k_iter-1))

    # For the r-sweep, we consider j-values that do (do not)
    # incorporate j = 1 if we use (do not use) reflective BCs: 
    j_range   = (BC_Z_bot == :Reflect) ? dN_Z : 2:(N_Z-1)

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

        if BC_Z_bot != :Reflect
            Φ_half[:,1] .= Φ[:,1]   # Ensure BC is fixed 
        end

        @threads :static for j in j_range

            tid = Threads.threadid()
            Ar_tot, Br_tot, Cr_tot, Dr_tot, Cprimer_tot, Dprimer_tot, xr = scratch_r_local[tid]

            ### Setup tridiagonal system in r-direction for each Z-index j ###
            @inbounds begin 
                # Clamp j-values:
                jm1 = max(j-1, 1)

                # COEFFICIENTS FOR THE INNER BOUNDARY (i = 1) - REFLECTIVE BCs:
                Ar_tot[1]      = 0.0
                Br_tot[1]      = - ( RPoiss_ij[1,j]  +  Δw  -  RPoiss_ijm[1,j] )
                Cr_tot[1]      = RPoiss_ijp[1,j]
                Dr_tot[1]      = ( Four_π_G * ρtot[1,j]  -  Φ_old[1,j] * Δw
                               -  (ZPoiss_ijp[1,j]*Φ_old[1,j+1]  -  ZPoiss_ij[1,j]* Φ_old[1,j]  +  ZPoiss_ijm[1,j]*Φ_old[1,jm1]))

                # INTERIOR COEFFICIENTS:
                # Lower diagonal:
                Ar_tot[dN_r_int]  .= RPoiss_ijm[dN_r_int,j]
                # Diagonal:
                Br_tot[dN_r_int]  .= .- ( RPoiss_ij[dN_r_int,j] .+ Δw )
                # Upper diagonal:
                Cr_tot[dN_r_int]  .= RPoiss_ijp[dN_r_int,j]
                # Right-hand side:
                Dr_tot[dN_r_int]  .= ( Four_π_G .* ρtot[dN_r_int,j]  .-  Φ_old[dN_r_int,j] .* Δw
                                  .- (ZPoiss_ijp[dN_r_int,j].*Φ_old[dN_r_int,j+1]  .-  ZPoiss_ij[dN_r_int,j].* Φ_old[dN_r_int,j]  .+  ZPoiss_ijm[dN_r_int,j].*Φ_old[dN_r_int,jm1]))


                # NEAR OUTER BOUNDARY (i = N_r - 1), WITH BOUNDARY CONDITION AT i = N_r:
                Ar_tot[N_r-1]      = RPoiss_ijm[N_r-1,j]
                # Diagonal:
                Br_tot[N_r-1]      = - ( RPoiss_ij[N_r-1,j] .+ Δw )
                # Upper diagonal:
                Cr_tot[N_r-1]      = 0.0
                # Right-hand side:
                Dr_tot[N_r-1]      = ( Four_π_G * ρtot[N_r-1,j]  -  Φ_old[N_r-1,j] * Δw  -  RPoiss_ijp[N_r-1,j] * Φ[N_r,j]
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
                if BC_Z_bot == :Reflect 
                    # In this case we assume reflective BCs, so Φ[0] = Φ[1]:
                    AZ_tot[1]      = 0.0
                    BZ_tot[1]      = - ( ZPoiss_ij[i,1]  +  Δw  -  ZPoiss_ijm[i,1] )
                    CZ_tot[1]      = ZPoiss_ijp[i,1]
                    DZ_tot[1]      = ( Four_π_G * ρtot[i,1]  -  Φ_half[i,1] * Δw
                                   -  (RPoiss_ijp[i,1]*Φ_half[i+1,1]  -  RPoiss_ij[i,1]* Φ_half[i,1]  +  RPoiss_ijm[i,1]*Φ_half[im1,1]) )
                else
                    # In this case, Φ should just equal the BC at j = 1, which is easiest 
                    # implemented as follows:
                    AZ_tot[1]      = 0.0
                    BZ_tot[1]      = 1.0
                    CZ_tot[1]      = 0.0
                    DZ_tot[1]      = Φ[i,1]
                end

                # INTERIOR COEFFICIENTS:
                # Lower diagonal:
                AZ_tot[dN_Z_int]  .= ZPoiss_ijm[i,dN_Z_int]
                # Diagonal:
                BZ_tot[dN_Z_int]  .= .- ( ZPoiss_ij[i,dN_Z_int] .+ Δw )
                # Upper diagonal:
                CZ_tot[dN_Z_int]  .= ZPoiss_ijp[i,dN_Z_int]
                # Right-hand side:
                DZ_tot[dN_Z_int]  .= ( Four_π_G .* ρtot[i,dN_Z_int]  .-  Φ_half[i,dN_Z_int] .* Δw
                                  .-  (RPoiss_ijp[i,dN_Z_int].*Φ_half[i+1,dN_Z_int]  .-  RPoiss_ij[i,dN_Z_int].* Φ_half[i,dN_Z_int]  .+  RPoiss_ijm[i,dN_Z_int].*Φ_half[im1,dN_Z_int]))

                # NEAR OUTER BOUNDARY (j = N_Z - 1), WITH BOUNDARY CONDITION AT j = N_Z:
                AZ_tot[N_Z-1]      = ZPoiss_ijm[i,N_Z-1]
                # Diagonal:
                BZ_tot[N_Z-1]      = - ( ZPoiss_ij[i,N_Z-1] + Δw )
                # Upper diagonal:
                CZ_tot[N_Z-1]      = 0.0
                # Right-hand side:
                DZ_tot[N_Z-1]      = ( Four_π_G * ρtot[i,N_Z-1]  -  Φ_half[i,N_Z-1] * Δw  -  ZPoiss_ijp[i,N_Z-1] * Φ[i,N_Z]
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
                                       - Four_π_G * ρtot[i,j]  ) / ( Four_π_G * ρtot[i,j] )

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
                 ρtot::AbstractArray{Float64, 2},
                    ρ::AbstractArray{Float64, 2},
                   ρd::AbstractArray{Float64, 2},
                 ρ_bg::AbstractArray{Float64, 2},
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
                 ZCoM::AbstractArray{Float64, 1},
                    Q::AbstractArray{Float64, 1},
                    P::AbstractArray{Float64, 1},
                 ℓmax::Int    = 20,
             BC_Z_bot::Symbol = :Reflect

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
        Converged = Update_Φ_ADI!( Φ, Φ_half, Φ_old, ρtot, ρ, ρd, ρ_bg,
                                   RPoiss_ijp, RPoiss_ij, RPoiss_ijm,
                                   ZPoiss_ijp, ZPoiss_ij, ZPoiss_ijm,
                                   Δτ_max, Δτ_min, k_iter,
                                   ΔV, rCoM, ZCoM, Q, P, ℓmax, BC_Z_bot )

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
                        f_gravity_r::AbstractArray{Float64, 2},
                        f_gravity_Z::AbstractArray{Float64, 2},
                                  ρ::AbstractArray{Float64, 2},
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


    # Update gravitational force per unit volume (mainly for tracking):
    @threads for j in dN_Z
        @inbounds @simd for i in dN_r
            f_gravity_r[i,j] = - ρ[i,j] * ∂Φ∂r[i,j]
            f_gravity_Z[i,j] = - ρ[i,j] * ∂Φ∂Z[i,j]
        end
    end

    return nothing
end


""" This function checks whether the residual for the updated 
    density field is large enough that we *should* solve the 
    Poisson equation again. Returns true if we should, false otherwise. """
function ShouldWeUpdateΦ( Φ::AbstractArray{Float64, 2},
                          ρ::AbstractArray{Float64, 2},
                         ρd::AbstractArray{Float64, 2},
                       ρ_bg::AbstractArray{Float64, 2},
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
                Φ_prime  = ( - Four_π_G * ( ρ[i,j] + ρd[i,j] + ρ_bg[i,j] )
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

