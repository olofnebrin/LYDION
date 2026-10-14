
""" FILE:        UpdateLyaJandH.jl

    PURPOSE:     Contains the code for updating the Lyα mean intensity J and first moment 
                 H, as a result of advection, and frequency diffusion + Doppler shifting.
                 Note that Lyα emission + destruction/absorption is operator split,
                 and treated separately.

    REFERENCES:  (1) 

"""


using .Structs: LyαArrays, GeometryCylindrical, GeometryCartesian
using  Base.Threads               # For CPU acceleration


export Initialize_ν_Scratch2D!, AdvectLyα!, FrequencyUpdateJLyα_AllCells!, UpdateLyαJandH! 

const DEBUG_Lyα::Bool     = false    # For debugging/ensuring correct array sizes (default: false)
const LOG_LYA_TIMES::Bool = false      # Set true to print per-step timings, only useful for benchmarking performance (default: false)


const scratch_ν = Ref{Vector{NTuple{7, Vector{Float64}}}}(Vector{NTuple{7, Vector{Float64}}}())

""" This function initializes scratch arrays, needed to be able to 
    do the frequency update in an efficient and thread-safe manner. """
function Initialize_ν_Scratch2D!( N_ν::Int64 )
    nthreads = Threads.maxthreadid()
    scratch_ν[] = [
        ( Vector{Float64}(undef, N_ν-2),  # A
          Vector{Float64}(undef, N_ν-2),  # B
          Vector{Float64}(undef, N_ν-2),  # C
          Vector{Float64}(undef, N_ν-2),  # D
          Vector{Float64}(undef, N_ν-2),  # C_prime
          Vector{Float64}(undef, N_ν-2),  # D_prime
          Vector{Float64}(undef, N_ν-2) ) # x (solution)
        for _ in 1:nthreads
    ]
end


const ϵ_Active::Float64 = 1e-10 

""" This function finds which frequency bins are 'active', i.e. 
    contains a non-negligible Lyα J. """
@views function FindActiveBins( Lyα::LyαArrays, 
                                 ΔV::AbstractArray{Float64, 2}
                               )

    # Get Lyα intensity & bin widths:
    J    = Lyα.J_Lyα
    Δν   = Lyα.Δν

    # Grid size & intervals:
    N_ν, N_r, N_Z  = size( J )   # Number of bins 
    dN_r           = 1:N_r 
    dN_Z           = 1:N_Z 
    dN_ν           = Lyα.dN_ν    # Useful ν-interval, 1:N_ν

    # Start by computing the total Lyα energy over the whole simulation box:

    Etot = 0.0
    for j in dN_Z 
        @inbounds for i in dN_r 
            ΔV_ij = ΔV[i,j]
            @simd for k in dN_ν
                Etot += Δν[k] * max( J[k,i,j] - Jeps, 0.0 ) * ΔV_ij
            end
        end # End of i-loop 
    end # End of j-loop 

    kmin = N_ν   # Initialize smallest value of k containing active J 
    kmax = 1     # Initialize largest value of k containing active J 
    for k in dN_ν

        Ebin = 0.0 # Initialize energy in bin 
        for j in dN_Z
            @inbounds @simd for i in dN_r
                Ebin += Δν[k] * max( J[k,i,j] - 10 * Jeps, 0.0 ) * ΔV[i,j]
            end
        end

        # If the energy in bin k, over all cells, is > ϵ_Active * Etot, 
        # use an interval such that (kmin, kmax) contains k:
        if Ebin > ϵ_Active * Etot  
            kmin = min( kmin, k )
            kmax = max( kmax, k )
        end
    end

    # Ensure that a few bins around the core are always flagged as active:
    kmax = max( kmax, Lyα.kcore_max )
    kmin = min( kmin, Lyα.kcore_min )

    return kmin, kmax
end

""" This function performs the M1 advection of Lyα J and H over a given time-step (times c̃), 
    for all (active) frequency bins. 
        
    NOTE: (1) Lyα emission and absorption/destruction is operator split and *not* 
              treated here. By default it is treated in the thermochemistry sub-cycling.
              
          (2) The solution here works for both cylindrical coordinates and Cartesian 
              coordinates, as specified by geometry.  """
@views function AdvectLyα!( c̃Δt::Float64,                   # Reduced speed of light (c̃ ≤ c) × time-step 
                            Lyα::LyαArrays,                 # The Lyα state/arrays 
                       geometry::Union{GeometryCylindrical, GeometryCartesian},

                          # Gas velocity field:
                       u_r_star::AbstractArray{Float64,2},
                       u_Z_star::AbstractArray{Float64,2},
                        ∂u_r_∂r::AbstractArray{Float64,2},
                        ∂u_r_∂Z::AbstractArray{Float64,2}, 
                        ∂u_Z_∂r::AbstractArray{Float64,2}, 
                        ∂u_Z_∂Z::AbstractArray{Float64,2},
                     u_r_over_r::AbstractArray{Float64,2},

                      BC_r_left::Symbol       = :Reflect,   # Boundary condition at i = 1/2 
                     BC_r_right::Symbol       = :Outflow,   # Boundary condition at i = N_r + 1/2
                       BC_Z_bot::Symbol       = :Reflect,   # Boundary condition at j = 1/2 
                       BC_Z_top::Symbol       = :Outflow    # Boundary condition at j = N_Z + 1/2
                                 )

    # Unpack arrays:

    # Lyα J and H:
    J   = Lyα.J_Lyα;    Jold    = Lyα.J_Lyα_old   
    H_r = Lyα.H_r_Lyα;  H_r_old = Lyα.H_r_Lyα_old
    H_Z = Lyα.H_Z_Lyα;  H_Z_old = Lyα.H_Z_Lyα_old

    # Numerical fluxes for spatial advection: 
    F_r    = Lyα.F_r_Lyα;     F_Z    = Lyα.F_Z_Lyα
    F_r_Hr = Lyα.F_r_Hr_Lyα;  F_Z_Hr = Lyα.F_Z_Hr_Lyα
    F_r_HZ = Lyα.F_r_HZ_Lyα;  F_Z_HZ = Lyα.F_Z_HZ_Lyα

    # ϕϕ-component of radiation pressure tensor K = D × J at interfaces, 
    # needed for geometric source term:
    K_ϕϕ_iphalf = Lyα.K_ϕϕ_iphalf_Lyα
    K_ϕϕ_imhalf = Lyα.K_ϕϕ_imhalf_Lyα

    # Flux opacity:
    α_F  = Lyα.α_F_Lyα

    # -------------------------------------------------------------------------------------
    # -------  M 1  A D V E C T I O N  O F  L Y M A N - α  F O R  A L L  B I N S  ---------
    # -------------------------------------------------------------------------------------

    UpdateLyaJold = true  # We want to update Jold to J when the advection step is finished 

    # Find frequency bins that are active (non-negligible J):
    kmin, kmax = FindActiveBins( Lyα, geometry.ΔV )

    AdvectJandH!( c̃Δt, α_F, J, Jold, H_r, H_Z, K_ϕϕ_iphalf, K_ϕϕ_imhalf, 
                  F_r, F_Z, F_r_Hr, F_Z_Hr, F_r_HZ, F_Z_HZ, 
                  geometry, u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,

                  # Other stuff:
                  UpdateLyaJold, 
                  BC_r_left,  BC_r_right,  BC_Z_bot,  BC_Z_top,
                  kmin, kmax, Lyα.FluxMethod )


    # Update old H components (important for sub-cycling of emission/absorption,
    # so old values are up to date) (J_old is already set to J within AdvectJandH!):
    copyto!( H_r_old, H_r )
    copyto!( H_Z_old, H_Z )

    return nothing 
end


""" The frequency update of J for Lyα, from frequency diffusion, 
    Doppler shifts, and atomic recoil. H is also updated for consistency 
    with assumed fixed Eddington tensor. A BDF1 method is used over a 
    time-step Δt, with Δw ≡ 1 / c̃ Δt. """
@inline @fastmath function FrequencyUpdateJLyα!( Δw::Float64, 
                                                  J::Array{Float64,3},
                                               Jold::Array{Float64,3},
                                             Fij_kp::Array{Float64,3}, 
                                              Fij_k::Array{Float64,3}, 
                                             Fij_km::Array{Float64,3},
                                                A_ν::Vector{Float64},
                                                B_ν::Vector{Float64},
                                                C_ν::Vector{Float64},
                                                D_ν::Vector{Float64},

                                           Cprime_ν::Vector{Float64},
                                           Dprime_ν::Vector{Float64},
                                                x_ν::Vector{Float64},  # Holds the tridiagonal solution (length N_ν - 2)
                                                  i::Int,
                                                  j::Int,
                                                N_ν::Int
                                                )

    # Number of interior frequency bins included in the tridiagonal solve.
    Nint = N_ν - 2

    @inbounds @simd for row in 1:Nint
        # row = 1 corresponds to k = 2, row = Nint corresponds to k = N_ν - 1.
        k = row + 1

        # Lower diagonal:
        A_ν[row] = Fij_km[k,i,j]

        # Diagonal:
        B_ν[row] = - ( Fij_k[k,i,j] + Δw )

        # Upper diagonal:
        C_ν[row] = Fij_kp[k,i,j]

        # Right-hand side:
        D_ν[row] = - Δw * Jold[k,i,j]
    end

    # Solve tridiagonal system:
    x_sol = TridiagonalSolve( A_ν, B_ν, C_ν, D_ν, Cprime_ν, Dprime_ν, x_ν )

    # Copy solution back to J.
    @inbounds @simd for row in 1:Nint
        k = row + 1
        J[k,i,j] = x_sol[row]
    end
    
    return nothing 
end


""" This function does the frequency update for all spatial bins (i,j). """
@views function FrequencyUpdateJLyα_AllCells!( c̃Δt::Float64,                   # Reduced speed of light (c̃ ≤ c) × time-step 
                                               Lyα::LyαArrays,                 # The Lyα state/arrays 

                                                 T::AbstractArray{Float64, 2}, # Gas temperature for every cell      (N_r   x N_Z)

                                              # Gas velocity field:
                                           ∂u_r_∂r::AbstractArray{Float64,2},
                                           ∂u_r_∂Z::AbstractArray{Float64,2}, 
                                           ∂u_Z_∂r::AbstractArray{Float64,2}, 
                                           ∂u_Z_∂Z::AbstractArray{Float64,2},
                                        u_r_over_r::AbstractArray{Float64,2},

                                 UpdateFreqFactors::Bool = true
                                     )

    # Lyα J and H:
    J   = Lyα.J_Lyα;    Jold    = Lyα.J_Lyα_old   
    H_r = Lyα.H_r_Lyα;  H_r_old = Lyα.H_r_Lyα_old
    H_Z = Lyα.H_Z_Lyα;  H_Z_old = Lyα.H_Z_Lyα_old
    
    # Arrays for frequency diffusion:
    Fij_kp = Lyα.Fij_kp;   Fij_k = Lyα.Fij_k;   Fij_km = Lyα.Fij_km

    # Grid size & intervals:
    N_ν, N_r, N_Z  = size( J )               # Grid size 
    dN_ν           = Lyα.dN_ν
    Δw             = @fastmath 1 / c̃Δt       # Useful constant for frequency diffusion/advection

    if LOG_LYA_TIMES
        t_freq_factors = 0.0
        t_freq_solve   = 0.0
        t_freq_Hcorr   = 0.0
    end

    # Update frequency leakage factors so that they use the latest, post-advection, 
    # Eddington factors D:
    if UpdateFreqFactors
        if LOG_LYA_TIMES
            t0 = time_ns()
        end

        UpdateFreqFactors!( Lyα, # The Lyα state/arrays 
                              T, # Gas temperature for every cell 
                        ∂u_r_∂r, # ∂u_r/∂r
                        ∂u_r_∂Z, # ∂u_r/∂Z
                        ∂u_Z_∂r, # ∂u_Z/∂r
                        ∂u_Z_∂Z, # ∂u_Z/∂Z
                     u_r_over_r, # ⟨u_r / r⟩ 
                       )

        if LOG_LYA_TIMES
            t_freq_factors = time_ns() - t0
        end
    end

    # Initialize the scratch arrays on the very first call 
    # (or if the thread count changed):
    if isempty(scratch_ν[]) || length(scratch_ν[]) != Threads.maxthreadid() || length(scratch_ν[][1][1]) != N_ν-2
        Initialize_ν_Scratch2D!( N_ν )
    end
    local_scratch = scratch_ν[]

    # -------------------------------------------------------------------------------------
    # -------  F R E Q U E N C Y  S O L V E  F O R  A L L  C E L L S  (i,j)  --------------
    # -------------------------------------------------------------------------------------

    N_cells = N_r * N_Z

    if LOG_LYA_TIMES
        t0 = time_ns()
    end

    @threads :static for idx in 1:N_cells
        tid  = Threads.threadid()
        A_ν, B_ν, C_ν, D_ν, Cprime_ν, Dprime_ν, x_ν = local_scratch[tid] # Scratch arrays

        # Map linear idx → (i,j), consistent with loop over j as outer:
        i = (idx - 1) % N_r + 1
        j = (idx - 1) ÷ N_r + 1

        @inbounds begin
            # Update J by frequency diffusion & Doppler shifting:
            FrequencyUpdateJLyα!( Δw,
                                  J, Jold,
                                  Fij_kp, Fij_k, Fij_km,
                                  A_ν, B_ν, C_ν, D_ν, Cprime_ν, Dprime_ν, x_ν,
                                  i, j, N_ν )
        end
    end

    if LOG_LYA_TIMES
        t_freq_solve = time_ns() - t0
    end

    # -------------------------------------------------------------------------------------
    # ------------------------   F L U X  C O R R E C T I O N   ---------------------------
    # -------------------------------------------------------------------------------------

    # Update H such that the reduced flux f = |H|/J stay constant (consistent with the fixed-D 
    # assumption in the frequency update), and also update Jold: 
    if LOG_LYA_TIMES
        t0 = time_ns()
    end

    @threads :static for idx in 1:N_cells

        # Map linear idx → (i,j), consistent with loop over j as outer:
        i = (idx - 1) % N_r + 1
        j = (idx - 1) ÷ N_r + 1 

        @inbounds @simd for k in dN_ν
            @fastmath begin
                # Enforce floor J ≥ Jeps::
                J_kij    = max( J[k,i,j], Jeps )

                # Load values of J (again, but floored), H_r, and H_Z: 
                J[k,i,j] = J_kij
                Hr_kij   = H_r[k,i,j]
                HZ_kij   = H_Z[k,i,j]

                # Preliminary re-scaling of H_r and H_Z:
                invJold  = 1 / max( Jold[k,i,j], Jeps )
                Ratio    = J_kij  * invJold
                Hr_kij   = Hr_kij * Ratio
                HZ_kij   = HZ_kij * Ratio 

                # Also enforce causality f ≤ 1 (can change D slightly if triggered):
                Hmag_squared = Hr_kij * Hr_kij  +  HZ_kij * HZ_kij
                if Hmag_squared > J_kij * J_kij
                    ScaleFact  = J_kij / sqrt( Hmag_squared ) 
                    Hr_kij = Hr_kij * ScaleFact 
                    HZ_kij = HZ_kij * ScaleFact 
                end 

                # Save updated H_r and H_Z:
                H_r[k,i,j]     = Hr_kij
                H_Z[k,i,j]     = HZ_kij

                # Finally, update Jold:
                Jold[k,i,j]    = J_kij

                # Also update H_r_old and H_Z_old a final time:
                H_r_old[k,i,j] = Hr_kij
                H_Z_old[k,i,j] = HZ_kij

            end # End of fastmath 

        end # End of k-loop 
    end # End of cell loop 

    if LOG_LYA_TIMES
        t_freq_Hcorr = time_ns() - t0

        println("  UpdateFreqFactors!: ", t_freq_factors * 1e-9, " s")
        println("  tridiagonal solve:  ", t_freq_solve   * 1e-9, " s")
        println("  H correction:       ", t_freq_Hcorr   * 1e-9, " s")
        println("  frequency subtotal: ", (t_freq_factors + t_freq_solve + t_freq_Hcorr) * 1e-9, " s")
    end

    return nothing 
end


""" This function updates the Lyα mean intensity J(r,Z,ν) and first moment components,
    H_r(r,Z,ν) and H_Z(r,Z,ν), over a time-step Δt. Advection and frequency 
    diffusion + Doppler shifting are operator split. 
    
    Advection is treated explicitly using an M1 moment method, while the 
    frequency diffusion + atomic recoil + Doppler shifting is treated implicitly for
    numerical stability. 
    
    NOTE: (1) Lyα emission and absorption/destruction is operator split and *not* 
              treated here. By default it is treated in the thermochemistry sub-cycling.
              
          (2) The solution here works for both cylindrical coordinates and Cartesian 
              coordinates (with r → X, Z → Y), as long as appropriate cell volumes ΔV 
              and interface areas A_r and A_Z are used. 
"""
@views function UpdateLyαJandH!( Δt::Float64,                   # Reduced speed of light (c̃ ≤ c) × time-step 
                                  c̃::Float64,                   # Reduced speed of light 
                                Lyα::LyαArrays,                 # The Lyα state/arrays 
                           geometry::Union{GeometryCylindrical, GeometryCartesian},

                                  T::AbstractArray{Float64, 2}, # Gas temperature for every cell      (N_r   x N_Z)

                  UpdateFreqFactors::Bool,

                          # Gas velocity field:
                           u_r_star::AbstractArray{Float64,2},
                           u_Z_star::AbstractArray{Float64,2},
                            ∂u_r_∂r::AbstractArray{Float64,2},
                            ∂u_r_∂Z::AbstractArray{Float64,2}, 
                            ∂u_Z_∂r::AbstractArray{Float64,2}, 
                            ∂u_Z_∂Z::AbstractArray{Float64,2},
                         u_r_over_r::AbstractArray{Float64,2},

                          BC_r_left::Symbol       = :Reflect,   # Boundary condition at i = 1/2 
                         BC_r_right::Symbol       = :Outflow,   # Boundary condition at i = N_r + 1/2
                           BC_Z_bot::Symbol       = :Reflect,   # Boundary condition at j = 1/2 
                           BC_Z_top::Symbol       = :Outflow    # Boundary condition at j = N_Z + 1/2
                                 )
    
    # Convenient variable:
    c̃Δt = @fastmath c̃ * Δt       

    if LOG_LYA_TIMES
        t_advect      = 0.0
        t_freqsolve   = 0.0
    end

    ################################################################################
    ############                                     ###############################
    ############          1. ADVECTION STEP          ###############################
    ############                                     ###############################
    ################################################################################

    # We first advect the Lyα J and H:

    if LOG_LYA_TIMES
        t0 = time_ns()
    end

    AdvectLyα!( c̃Δt, Lyα, geometry, 
                u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r, 
                BC_r_left,  BC_r_right,  BC_Z_bot,  BC_Z_top )

    if LOG_LYA_TIMES
        t_advect = time_ns() - t0
    end

    ################################################################################
    ############                                     ###############################
    ############          2. FREQUENCY STEP          ###############################
    ############                                     ###############################
    ################################################################################

    # Next, do the frequency solve:

    if LOG_LYA_TIMES
        t0 = time_ns()
    end

    FrequencyUpdateJLyα_AllCells!( c̃Δt, Lyα, T, 
                                   ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,  
                                   UpdateFreqFactors )

    if LOG_LYA_TIMES
        t_freqsolve = time_ns() - t0

        println("Advection:  ", t_advect * 1e-9, " s")
        println("Freq solve:  ", t_freqsolve * 1e-9, " s")
        println("-------------------")
    end

    return nothing 
end

