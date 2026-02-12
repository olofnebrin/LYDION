
""" FILE:        UpdateLyaJandH.jl

    PURPOSE:     Contains the code for updating the Lyα mean intensity J and first moment 
                 H, as a result of advection, and frequency diffusion + Doppler shifting.
                 Note that Lyα emission + destruction/absorption is operator split,
                 and treated separately.

    REFERENCES:  (1) 

"""


using .Structs: LyαArrays, GeometryCylindrical, GeometryCartesian
using  Base.Threads               # For CPU acceleration


export Initialize_ν_Scratch2D!, UpdateLyαJandH! 

const DEBUG_Lyα::Bool     = true    # For debugging/ensuring correct array sizes 
const LOG_LYA_TIMES::Bool = false   # Set true to print per-step timings, only useful for benchmarking performance


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


""" The frequency update of J for Lyα, from frequency diffusion, 
    Doppler shifts, and atomic recoil. H is also updated for consistency 
    with assumed fixed Eddington tensor. A BDF1 method is used over a 
    time-step Δt, with Δw ≡ 1 / c̃ Δt. """
@views @fastmath function FrequencyUpdateJLyα!( Δw::Float64, 
                                                 J::AbstractArray{Float64},
                                              Jold::AbstractArray{Float64},
                                            Fij_kp::AbstractArray{Float64}, 
                                             Fij_k::AbstractArray{Float64}, 
                                            Fij_km::AbstractArray{Float64},
                                               A_ν::AbstractArray{Float64},
                                               B_ν::AbstractArray{Float64},
                                               C_ν::AbstractArray{Float64},
                                               D_ν::AbstractArray{Float64},

                                          Cprime_ν::AbstractArray{Float64},
                                          Dprime_ν::AbstractArray{Float64},
                                               x_ν::AbstractArray{Float64},  # Holds the tridiagonal solution (length N_ν - 2)
                                          dN_ν_int::UnitRange,               # Must be 2:(N_ν-1), where N_ν is the number of frequency bins
                                               )

    @inbounds begin 
        # Lower diagonal (copyto! is slightly better than .=):
        A_ν         .= Fij_km[dN_ν_int]
        # Diagonal:
        B_ν         .= .- ( Fij_k[dN_ν_int] .+ Δw )
        # Upper diagonal:
        C_ν         .= Fij_kp[dN_ν_int]
        # Right-hand side:
        D_ν         .= .- Δw .* Jold[dN_ν_int] 

        # Solve tridiagonal system:
        J[dN_ν_int] .= TridiagonalSolve( A_ν, B_ν, C_ν, D_ν, Cprime_ν, Dprime_ν, x_ν )

    end
    
    return nothing 
end


const ϵ_Active::Float64 = 1e-10 

@views function FindActiveBins( Lyα::LyαArrays, 
                             inv_ΔV::AbstractArray{Float64, 2}
                               )

    # Get Lyα intensity & bin widths:
    J    = Lyα.J_Lyα
    Δν   = Lyα.Δν

    # Grid size & intervals:
    N_r, N_Z, N_ν  = size( J )   # Number of bins 
    dN_r           = 1:N_r 
    dN_Z           = 1:N_Z 
    dN_ν           = 1:N_ν       # Useful ν-interval

    # Start by computing the total Lyα energy over the whole simulation box:
    Etot = 0.0
    for k in dN_ν
        for j in dN_Z
            @inbounds @simd for i in dN_r
                Etot += Δν[k] * max( J[i,j, k] - Jeps, 0.0 ) / inv_ΔV[i,j]
            end
        end
    end

    kmin = N_ν   # Initialize smallest value of k containing active J 
    kmax = 1     # Initialize largest value of k containing active J 
    for k in dN_ν

        Ebin = 0.0 # Initialize energy in bin 
        for j in dN_Z
            @inbounds @simd for i in dN_r
                Ebin += Δν[k] * max( J[i,j, k] - 10 * Jeps, 0.0 ) / inv_ΔV[i,j]
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
    
    # Lyα J and H:
    J   = Lyα.J_Lyα;    Jold    = Lyα.J_Lyα_old   
    H_r = Lyα.H_r_Lyα;  H_r_old = Lyα.H_r_Lyα_old
    H_Z = Lyα.H_Z_Lyα;  H_Z_old = Lyα.H_Z_Lyα_old

    # Numerical fluxes for spatial advection: 
    F_r    = Lyα.F_r_Lyα;     F_Z    = Lyα.F_Z_Lyα
    F_r_Hr = Lyα.F_r_Hr_Lyα;  F_Z_Hr = Lyα.F_Z_Hr_Lyα
    F_r_HZ = Lyα.F_r_HZ_Lyα;  F_Z_HZ = Lyα.F_Z_HZ_Lyα

    # Eddington tensor components:
    D_rr   = Lyα.D_rr_Lyα;    D_ZZ = Lyα.D_ZZ_Lyα;   D_rZ = Lyα.D_rZ_Lyα

    # ϕϕ-component of radiation pressure tensor K = D × J at interfaces, 
    # needed for geometric source term:
    K_ϕϕ_iphalf = Lyα.K_ϕϕ_iphalf_Lyα;   K_ϕϕ_imhalf = Lyα.K_ϕϕ_imhalf_Lyα

    # Flux opacity:
    α_F    = Lyα.α_F_Lyα
    
    # Arrays for frequency diffusion:
    Fij_kp = Lyα.Fij_kp;   Fij_k = Lyα.Fij_k;   Fij_km = Lyα.Fij_km

    # Needed geoetry (intervals, areas etc):
    inv_ΔV     = geometry.inv_ΔV
    inv_r_cell = geometry.inv_r_cell  # Same signature for both cylindrical & Cartesian, but only used in former

    if geometry isa GeometryCylindrical
        # We have cylindrical coordinates:
        CylindricalCoords = true 

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


    # Grid size & intervals:
    N_r, N_Z, N_ν  = size( J )               # Grid size 
    dN_r           = 1:N_r                   # Useful r-interval 
    dN_Z           = 1:N_Z                   # Useful Z-interval
    dN_ν           = Lyα.dN_ν                # 1:N_ν, useful ν-interval
    dN_ν_int       = Lyα.dN_ν_int            # 2:(N_ν-1), useful ν-interval for the frequency update 
    c̃Δt            = @fastmath c̃ * Δt        # Convenient variable 
    Δw             = @fastmath 1 / c̃Δt       # Useful constant for frequency diffusion/advection

    # Make sure the scratch arrays for interface fluxes & Kϕϕ have the correct size:
    if DEBUG_Lyα
        nthreads = Threads.maxthreadid()
        @assert size( K_ϕϕ_iphalf, 3 ) == nthreads
        @assert size( K_ϕϕ_imhalf, 3 ) == nthreads
        @assert size( F_r, 3 )         == nthreads
        @assert size( F_Z, 3 )         == nthreads
        @assert size( F_r_Hr, 3 )      == nthreads
        @assert size( F_Z_Hr, 3 )      == nthreads
        @assert size( F_r_HZ, 3 )      == nthreads
        @assert size( F_Z_HZ, 3 )      == nthreads
    end

    t_activebin   = 0.0
    t_advect      = 0.0
    t_freqfactors = 0.0
    t_freqsolve   = 0.0

    ################################################################################
    ############                                     ###############################
    ############          1. ADVECTION STEP          ###############################
    ############                                     ###############################
    ################################################################################

    # We first advect the Lyα J and H

    UpdateLyaJold = true  # We also want to update Jold to J when the advection step is finished 

    if LOG_LYA_TIMES
        t0 = time_ns()
    end

    # Find frequency bins that are active (non-negligible J):
    kmin, kmax    = FindActiveBins( Lyα, inv_ΔV )

    if LOG_LYA_TIMES
        t_activebin += (time_ns() - t0)
    end

    # Select Strang-split order:
    if Lyα.rfirst_Lyα == true 
        # If the last Strang-split order was r-first, change to Z-first:
        Lyα.rfirst_Lyα = false 
    else
        # If the last Strang-split order was Z-first, change to r-first:
        Lyα.rfirst_Lyα = true
    end

    if LOG_LYA_TIMES
        t0 = time_ns()
    end

    # Update old H components:
    copyto!( H_r_old, H_r )
    copyto!( H_Z_old, H_Z )

    @threads :static for k in kmin:kmax
        # Get thread ID for the interface flux & Kϕϕ scratch arrays:
        tid = Threads.threadid() 
        @inbounds begin

            # Spatial advection of J & H, and absorption of H (flux-absorption)
            # for every frequency bin:
            AdvectJandH!( c̃Δt,                  α_F[:,:,k],        
                          J[:,:,k],             Jold[:,:,k], 
                          H_r[:,:,k],           H_Z[:,:,k], 
                          D_rr[:,:,k],          D_ZZ[:,:,k],        D_rZ[:,:,k],  # Eddington factors D = K / J
                          K_ϕϕ_iphalf[:,:,tid], K_ϕϕ_imhalf[:,:,tid],
                          F_r[:,:,tid],         F_Z[:,:,tid],                     # Fluxes of J 
                          F_r_Hr[:,:,tid],      F_Z_Hr[:,:,tid],                  # Fluxes of H_r 
                          F_r_HZ[:,:,tid],      F_Z_HZ[:,:,tid],                  # Fluxes of H_Z 
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
                          UpdateLyaJold, 
                          CylindricalCoords,
                          BC_r_left,  BC_r_right,  BC_Z_bot,  BC_Z_top,
                          Lyα.rfirst_Lyα, Val(:Serial) )
            
        end
    end

    if LOG_LYA_TIMES
        t_advect += (time_ns() - t0)
    end

    ################################################################################
    ############                                     ###############################
    ############          2. FREQUENCY STEP          ###############################
    ############                                     ###############################
    ################################################################################

    if LOG_LYA_TIMES
        t0 = time_ns()
    end

    # Update frequency leakage factors so that they use the latest, post-advection, 
    # Eddington factors D:
    UpdateFreqFactors!( Lyα, # The Lyα state/arrays 
                          T, # Gas temperature for every cell 
                    ∂u_r_∂r, # ∂u_r/∂r
                    ∂u_r_∂Z, # ∂u_r/∂Z
                    ∂u_Z_∂r, # ∂u_Z/∂r
                    ∂u_Z_∂Z, # ∂u_Z/∂Z
                 u_r_over_r, # ⟨u_r / r⟩ 
                      )

    if LOG_LYA_TIMES
        t_freqfactors += (time_ns() - t0)
    end

    # Initialize the scratch arrays on the very first call 
    # (or if the thread count changed):
    if isempty(scratch_ν[]) || length(scratch_ν[]) != Threads.maxthreadid() || length(scratch_ν[][1][1]) != N_ν-2
        Initialize_ν_Scratch2D!( N_ν )
    end
    local_scratch = scratch_ν[]

    if LOG_LYA_TIMES
        t0 = time_ns()
    end

    # Solve the frequency diffusion for every cell (i,j):
    @threads for j in dN_Z

        tid  = Threads.threadid()
        A_ν, B_ν, C_ν, D_ν, Cprime_ν, Dprime_ν, x_ν = local_scratch[tid] # Scratch arrays

        @simd for i in dN_r
            @inbounds begin
                # Update J by frequency diffusion & Doppler shifting:
                FrequencyUpdateJLyα!( Δw, 
                                      J[i,j,:],       Jold[i,j,:], 
                                      Fij_kp[i,j,:],  Fij_k[i,j,:], Fij_km[i,j,:],
                                      A_ν, B_ν, C_ν, D_ν, Cprime_ν, Dprime_ν, x_ν, 
                                      dN_ν_int )

            end
        end # End of i-loop 
    end # End of j-loop 

    # Update H such that the reduced flux f = |H|/J stay constant (consistent with the fixed-D 
    # assumption in the frequency update), and also update Jold: 
    @threads for k in dN_ν
        for j in dN_Z
            @inbounds @simd for i in dN_r

                @fastmath begin
                    # Enforce floor J ≥ Jeps::
                    J[i,j,k] = max( J[i,j,k], Jeps )

                    # Load values of J (again, but floored), H_r, and H_Z: 
                    J_ijk    = J[i,j,k]
                    Hr_ijk   = H_r[i,j,k]
                    HZ_ijk   = H_Z[i,j,k]

                    # Preliminary re-scaling of H_r and H_Z:
                    invJold  = 1 / max( Jold[i,j,k], Jeps )
                    Ratio    = J_ijk * invJold
                    Hr_ijk   = Hr_ijk * Ratio
                    HZ_ijk   = HZ_ijk * Ratio 

                    # Also enforce causality f ≤ 1 (can change D slightly if triggered):
                    Hmag_squared = Hr_ijk * Hr_ijk  +  HZ_ijk * HZ_ijk
                    if Hmag_squared > J_ijk * J_ijk
                        ScaleFact  = J_ijk / sqrt( Hmag_squared ) 
                        Hr_ijk = Hr_ijk * ScaleFact 
                        HZ_ijk = HZ_ijk * ScaleFact 
                    end 

                    # Save updated H_r and H_Z:
                    H_r[i,j,k]     = Hr_ijk
                    H_Z[i,j,k]     = HZ_ijk

                    # Finally, update Jold:
                    Jold[i,j,k]    = J_ijk

                    # Also update H_r_old and H_Z_old a final time:
                    H_r_old[i,j,k] = Hr_ijk
                    H_Z_old[i,j,k] = HZ_ijk

                end # End of fastmath 

            end # End of i-loop 
        end # End of j-loop 
    end # End of k-loop 

    if LOG_LYA_TIMES
        t_freqsolve += (time_ns() - t0)

        println("Active bins:  ", t_activebin*1e-9,  " s")
        println("Advection:  ", t_advect*1e-9, " s")
        println("Freq factors:  ", t_freqfactors*1e-9, " s")
        println("Freq solve:  ", t_freqsolve*1e-9, " s")
        println("-------------------")
    end

    return nothing 
end


