
""" FILE:        AdvectDust.jl

    PURPOSE:     Contains the basic code for advecting dust, 
                 including gravity, but ignoring radiation pressure.
                 Second-order cell reconstruction is used, and the HLL 
                 flux function. 

"""

using .Structs: DustState, GravityState
using .ConstantsAndFunctions
using  Base.Threads                            # For CPU acceleration

const CFL_Dust::Float64 = 0.4                  # CFL factor for dust time-stepping (default: 0.4)
const ρd_floor::Float64 = 1e-20 * mH           # Dust density floor (default: 10⁻²⁰ mH)


""" This function computes the CFL time-step for dust advection. """
@inline function GetDustCFLTimeStep( ud_r::AbstractArray{Float64, 2}, 
                                     ud_Z::AbstractArray{Float64, 2},
                                        r::AbstractVector{Float64},
                                        Z::AbstractVector{Float64}
                                     )

    N_r         = length(r) - 1  # Grid size
    N_Z         = length(Z) - 1
    dN_r        = 1:N_r
    dN_Z        = 1:N_Z

    min_ΔtDust  = Inf64  # Initialize the minimum local time-scale
    @inbounds for j in dN_Z
        ΔZ                = Z[j+1] - Z[j] # Vertical cell width
        local_min_ΔtDust  = min_ΔtDust     # Local variable for tracking smallest CFL time

        @simd for i in dN_r
            # Compute the horizontal cell width (same for all j-indices):
            Δr = r[i+1] - r[i]

            # Compute the local CFL time-scale:
            Δt = min( Δr / abs(ud_r[i,j]),
                      ΔZ / abs(ud_Z[i,j])  )
            
            if Δt < local_min_ΔtDust 
                # Update the smallest time-scale found:
                local_min_ΔtDust = Δt 
            end
        end

        min_ΔtDust = min( min_ΔtDust, local_min_ΔtDust )
    end
    
    # The final default CFL time-step is simply the CFL factor times 
    # min_ΔtDust:
    return CFL_Dust * min_ΔtDust
end


""" Just a small useful function for computing dust velocities in a single cell,
    e.g. the ghost cells. """
@inline function GetDustVelocities( ρd::Float64, ρdud_r::Float64, ρdud_Z::Float64 )

    if ρd > ρd_floor
        inv_ρd  = 1 / ρd
        ud_r    = ρdud_r * inv_ρd 
        ud_Z    = ρdud_Z * inv_ρd
    else
        ud_r    = 0.0
        ud_Z    = 0.0
    end

    return ud_r, ud_Z 
end


""" This function updates dust velocities in all interior (i.e. non-ghost)
    cells. """
@inline function UpdateDustVelocities!( ρd::AbstractArray{Float64, 2},
                                    ρdud_r::AbstractArray{Float64, 2},
                                    ρdud_Z::AbstractArray{Float64, 2},
                                      ud_r::AbstractArray{Float64, 2},
                                      ud_Z::AbstractArray{Float64, 2}
                                      )

    N_r, N_Z = size( ρd )
    dN_r     = 1:N_r 
    dN_Z     = 1:N_Z 

    @threads for j in dN_Z 
        @inbounds @simd for i in dN_r

            ud_r[i,j], ud_Z[i,j] = GetDustVelocities( ρd[i,j], ρdud_r[i,j], ρdud_Z[i,j] )

        end # End of i-loop 
    end # End of j-loop 

    return nothing 
end


""" This function computes HLL fluxes for
    the conserved dust variables: 

    - F_ρd   (dust density), 
    - F_ρdud (dust momentum), 
    - F_ρdvd (dust transverse momentum) """
function GetHLLFlux_Dust( ρd_L::Float64, 
                          ρd_R::Float64, 
                          ud_L::Float64,
                          ud_R::Float64,
                          vd_L::Float64,
                          vd_R::Float64
                           )

    # Compute left & right momenta:
    ρdud_L   = ρd_L * ud_L   
    ρdud_R   = ρd_R * ud_R     
    ρdvd_L   = ρd_L * vd_L   
    ρdvd_R   = ρd_R * vd_R               

    # Left & right fluxes of dust density & momenta:
    F_ρd_L   = ρd_L * ud_L 
    F_ρd_R   = ρd_R * ud_R
    F_ρdud_L = ρdud_L * ud_L 
    F_ρdud_R = ρdud_R * ud_R  
    F_ρdvd_L = ρdvd_L * ud_L 
    F_ρdvd_R = ρdvd_R * ud_R 

    # Wave speeds:
    S_L      = min( ud_L, ud_R ) 
    S_R      = max( ud_L, ud_R )

    # Get HLL fluxes:
    if S_L ≥ 0.0
        F_ρd      = F_ρd_L
        F_ρdud    = F_ρdud_L
        F_ρdvd    = F_ρdvd_L
    elseif S_R ≤ 0.0
        F_ρd      = F_ρd_R
        F_ρdud    = F_ρdud_R
        F_ρdvd    = F_ρdvd_R
    else
        inv_denom = 1 / ( S_R - S_L )
        F_ρd      = ( S_R * F_ρd_L  - S_L * F_ρd_R     +  S_L * S_R * (ρd_R  - ρd_L) )    * inv_denom
        F_ρdud    = ( S_R * F_ρdud_L - S_L * F_ρdud_R  +  S_L * S_R * (ρdud_R - ρdud_L) ) * inv_denom
        F_ρdvd    = ( S_R * F_ρdvd_L - S_L * F_ρdvd_R  +  S_L * S_R * (ρdvd_R - ρdvd_L) ) * inv_denom
    end

    return F_ρd, F_ρdud, F_ρdvd 
end


""" This function computes the pressureless-dust Riemann fluxes used in the paper
    (cf. their Eq. 32 for density flux; analogous for momentum components).

    Conserved dust variables (in the sweep/normal direction):
      - ρd      : dust density
      - ρdud    : dust momentum in the normal direction (u ≡ v_{d,x} in the paper)
      - ρdvd    : dust transverse momentum (v ≡ v_{d,y}/v_{d,z})

    Inputs are left/right primitive states (ρd, u, v).

    Flux logic (based on normal velocities u_L, u_R):
      1) u_L>0 and u_R>0  -> upwind from left
      2) u_L<0 and u_R<0  -> upwind from right
      3) u_L<0 and u_R>0  -> diverging -> vacuum -> flux = 0
      4) u_L>0 and u_R<0  -> converging -> sum of both one-sided fluxes
"""
function GetHLLFlux_Dust_OLD( ρd_L::Float64,
                          ρd_R::Float64,
                          ud_L::Float64,
                          ud_R::Float64,
                          vd_L::Float64,
                          vd_R::Float64 )

    # Left & right momenta:
    ρdud_L = ρd_L * ud_L
    ρdud_R = ρd_R * ud_R
    ρdvd_L = ρd_L * vd_L
    ρdvd_R = ρd_R * vd_R

    # Left & right fluxes (normal direction is "u"):
    F_ρd_L   = ρd_L   * ud_L
    F_ρd_R   = ρd_R   * ud_R
    F_ρdud_L = ρdud_L * ud_L          # = ρd_L * ud_L^2
    F_ρdud_R = ρdud_R * ud_R          # = ρd_R * ud_R^2
    F_ρdvd_L = ρdvd_L * ud_L          # = ρd_L * vd_L * ud_L
    F_ρdvd_R = ρdvd_R * ud_R          # = ρd_R * vd_R * ud_R

    # Piecewise "exact" pressureless Riemann flux:
    if (ud_L ≥ 0.0) && (ud_R ≥ 0.0)
        # Both states move right -> upwind (left)
        return F_ρd_L, F_ρdud_L, F_ρdvd_L

    elseif (ud_L ≤ 0.0) && (ud_R ≤ 0.0)
        # Both states move left -> upwind (right)
        return F_ρd_R, F_ρdud_R, F_ρdvd_R

    elseif (ud_L < 0.0) && (ud_R > 0.0)
        # Diverging -> vacuum forms at interface -> zero flux
        return 0.0, 0.0, 0.0

    else
        # Converging (ud_L > 0, ud_R < 0) -> sum of both one-sided fluxes
        return (F_ρd_L   + F_ρd_R),
               (F_ρdud_L + F_ρdud_R),
               (F_ρdvd_L + F_ρdvd_R)
    end
end




""" Reconstruct dust primitive variables at interface r_{i+1/2} (or Z_{j+1/2}). 
    Uses second-order linear reconstruction. If this produces any negative interface 
    dust density, it falls back to first-order reconstruction. """
function ReconstructInterface_Dust( ρd_i::Float64,    ρd_ip1::Float64,   ρd_ip2::Float64,   ρd_im1::Float64,
                                  ud_r_i::Float64,  ud_r_ip1::Float64, ud_r_ip2::Float64, ud_r_im1::Float64,
                                  ud_Z_i::Float64,  ud_Z_ip1::Float64, ud_Z_ip2::Float64, ud_Z_im1::Float64,
 
                                  rCoM_i::Float64, rCoM_ip1::Float64, rCoM_ip2::Float64, rCoM_im1::Float64, 
                                   r_ip1::Float64
                                   )

    # Intervals for the derivatives: 
    dr_i_im1   = rCoM_i   - rCoM_im1
    dr_ip1_i   = rCoM_ip1 - rCoM_i
    dr_ip2_ip1 = rCoM_ip2 - rCoM_ip1 

    # Intervals between cell centers & inteface:
    Δr_L       = r_ip1    - rCoM_i 
    Δr_R       = rCoM_ip1 - r_ip1 

    # Compute slope-limited derivatives for ρd:
    ∂ρd∂r_L    = MinMod( (ρd_i   - ρd_im1)   / dr_i_im1,
                         (ρd_ip1 - ρd_i)     / dr_ip1_i )
     
    ∂ρd∂r_R    = MinMod( (ρd_ip2 - ρd_ip1)   / dr_ip2_ip1,
                         (ρd_ip1 - ρd_i)     / dr_ip1_i )

    # Compute slope-limited derivatives for ud_r:
    ∂ud_r∂r_L  = MinMod( (ud_r_i   - ud_r_im1)   / dr_i_im1,
                         (ud_r_ip1 - ud_r_i)     / dr_ip1_i )
     
    ∂ud_r∂r_R  = MinMod( (ud_r_ip2 - ud_r_ip1)   / dr_ip2_ip1,
                         (ud_r_ip1 - ud_r_i)     / dr_ip1_i )

    # Compute slope-limited derivatives for ud_Z:
    ∂ud_Z∂r_L  = MinMod( (ud_Z_i   - ud_Z_im1)   / dr_i_im1,
                         (ud_Z_ip1 - ud_Z_i)     / dr_ip1_i )
     
    ∂ud_Z∂r_R  = MinMod( (ud_Z_ip2 - ud_Z_ip1)   / dr_ip2_ip1,
                         (ud_Z_ip1 - ud_Z_i)     / dr_ip1_i )

    # Get all ρd and ud interface variables:
    ρd_L       = ρd_i    +  ∂ρd∂r_L * Δr_L
    ρd_R       = ρd_ip1  -  ∂ρd∂r_R * Δr_R

    ud_r_L     = ud_r_i     +  ∂ud_r∂r_L * Δr_L
    ud_r_R     = ud_r_ip1   -  ∂ud_r∂r_R * Δr_R

    ud_Z_L     = ud_Z_i     +  ∂ud_Z∂r_L * Δr_L
    ud_Z_R     = ud_Z_ip1   -  ∂ud_Z∂r_R * Δr_R

    # In case reconstructed densities are negative, fall back to first-order:
    if ρd_L ≤ ρd_floor || ρd_R ≤ ρd_floor
        ρd_L   = ρd_i
        ρd_R   = ρd_ip1
        ud_r_L = ud_r_i
        ud_r_R = ud_r_ip1
        ud_Z_L = ud_Z_i
        ud_Z_R = ud_Z_ip1
    end

    return ρd_L, ρd_R, ud_r_L, ud_r_R, ud_Z_L, ud_Z_R
end


""" This function advects the dust in the r-direction, 
    using a forward Euler method. """
function ForwardEulerDust_r!( Δt::Float64,
                              ρd::AbstractArray{Float64, 2},
                          ρdud_r::AbstractArray{Float64, 2},
                          ρdud_Z::AbstractArray{Float64, 2},
                            ud_r::AbstractArray{Float64, 2},
                            ud_Z::AbstractArray{Float64, 2},
                          F_ρd_r::AbstractArray{Float64, 2},
                      F_ρdud_r_r::AbstractArray{Float64, 2},
                      F_ρdud_Z_r::AbstractArray{Float64, 2},
                          inv_ΔV::AbstractArray{Float64, 2},
                             A_r::AbstractArray{Float64, 2},
                               r::AbstractVector{Float64},
                            rCoM::AbstractVector{Float64},
                         
                       BC_r_left::Symbol,
                      BC_r_right::Symbol
                             )

    N_r, N_Z = size( ρd )
    dN_r     = 1:N_r 
    dN_rint  = 2:(N_r-2)
    dN_Z     = 1:N_Z 

    # Needed ghost cell CoM radii, obtained by mirroring:
    @inbounds begin 
        # Ghost CoM r-positions mirrored around r[1] = r_{1/2}:
        rCoM_0   = r[1] - (rCoM[1] - r[1]) 
        rCoM_m1  = r[1] - (rCoM[2] - r[1])

        # Ghost CoM r-positions mirrored around r[N_r+1]:
        rCoM_Np1 = r[N_r+1] + (r[N_r+1] - rCoM[N_r])
        rCoM_Np2 = r[N_r+1] + (r[N_r+1] - rCoM[N_r-1])
    end

    # Update dust velocities for all cells (excluding ghost cells):
    UpdateDustVelocities!( ρd, ρdud_r, ρdud_Z, ud_r, ud_Z )

    @threads for j in dN_Z 
        @inbounds begin 
            ##############################################################################
            ##############          INTERFACE i = 1/2          ###########################
            ##############################################################################

            if BC_r_left === :Reflect 
                # Ghost cell i = 0:
                ρd_0       = ρd[1,j] 
                ρdud_r_0   = - ρdud_r[1,j]   # Reflect radial momenta
                ρdud_Z_0   = ρdud_Z[1,j] 
                # Ghost cell i = -1:
                ρd_m1      = ρd[2,j] 
                ρdud_r_m1  = - ρdud_r[2,j]   # Reflect radial flux
                ρdud_Z_m1  = ρdud_Z[2,j]
            elseif BC_r_left === :Outflow  
                # Ghost cell i = 0:
                ρd_0       = ρd[1,j] 
                ρdud_r_0   = min( ρdud_r[1,j], 0.0 )          
                ρdud_Z_0   = ρdud_Z[1,j]
                # Ghost cell i = -1 (no gradients, so equal to cells i = 0 and 1):
                ρd_m1      = ρd_0 
                ρdud_r_m1  = ρdud_r_0   
                ρdud_Z_m1  = ρdud_Z_0
            end

            # Get dust velocities in ghost cells:
            ud_r_m1, ud_Z_m1 =  GetDustVelocities( ρd_m1, ρdud_r_m1, ρdud_Z_m1 )
            ud_r_0,  ud_Z_0  =  GetDustVelocities( ρd_0,  ρdud_r_0,  ρdud_Z_0 )

            # Reconstruct variables at interface r_{1/2} (i.e. r[1]):
            ρd_L, ρd_R, ud_r_L, ud_r_R, ud_Z_L, ud_Z_R = ReconstructInterface_Dust( ρd_0,    ρd[1,j],   ρd[2,j],   ρd_m1, 
                                                                                    ud_r_0,  ud_r[1,j], ud_r[2,j], ud_r_m1,
                                                                                    ud_Z_0,  ud_Z[1,j], ud_Z[2,j], ud_Z_m1,
                                                                                    rCoM_0,  rCoM[1],  rCoM[2],    rCoM_m1, 
                                                                                    r[1]               # Interface r_{1/2}
                                                                                    )

            # Fluxes at i = 1/2:
            F_ρd_r[1,j], F_ρdud_r_r[1,j], F_ρdud_Z_r[1,j] = GetHLLFlux_Dust( ρd_L, ρd_R, ud_r_L, ud_r_R, ud_Z_L, ud_Z_R )

            ##############################################################################
            ##############          INTERFACE i = 3/2          ###########################
            ##############################################################################

            # Reconstruct variables at interface r_{3/2} (i.e. r[2]):
            ρd_L, ρd_R, ud_r_L, ud_r_R, ud_Z_L, ud_Z_R = ReconstructInterface_Dust( ρd[1,j],   ρd[2,j],   ρd[3,j],   ρd_0, 
                                                                                    ud_r[1,j], ud_r[2,j], ud_r[3,j], ud_r_0,
                                                                                    ud_Z[1,j], ud_Z[2,j], ud_Z[3,j], ud_Z_0,
                                                                                    rCoM[1],   rCoM[2],   rCoM[3],   rCoM_0, 
                                                                                    r[2]               # Interface r_{3/2}
                                                                                    )

            # Fluxes at i = 3/2:
            F_ρd_r[2,j], F_ρdud_r_r[2,j], F_ρdud_Z_r[2,j] = GetHLLFlux_Dust( ρd_L, ρd_R, ud_r_L, ud_r_R, ud_Z_L, ud_Z_R )

            ##############################################################################
            ##############        INTERIOR INTERFACES 2 ≤ i ≤ N_r-2          #############
            ##############################################################################
            @simd for i in dN_rint
                ip1 = i + 1
                ip2 = i + 2
                im1 = i - 1

                # Reconstruct variables at interface r_{i+1/2} (i.e. r[i+1]):
                ρd_L, ρd_R, ud_r_L, ud_r_R, ud_Z_L, ud_Z_R = ReconstructInterface_Dust( ρd[i,j],   ρd[ip1,j],   ρd[ip2,j],   ρd[im1,j], 
                                                                                        ud_r[i,j], ud_r[ip1,j], ud_r[ip2,j], ud_r[im1,j],
                                                                                        ud_Z[i,j], ud_Z[ip1,j], ud_Z[ip2,j], ud_Z[im1,j],
                                                                                        rCoM[i],   rCoM[ip1],   rCoM[ip2],   rCoM[im1], 
                                                                                        r[ip1]      # Interface r_{i+1/2}
                                                                                    )

                # Fluxes at i = 3/2:
                F_ρd_r[ip1,j], F_ρdud_r_r[ip1,j], F_ρdud_Z_r[ip1,j] = GetHLLFlux_Dust( ρd_L, ρd_R, ud_r_L, ud_r_R, ud_Z_L, ud_Z_R )

            end # End of i-loop 

            # Boundary conditions at i = N_r + 1:
            if BC_r_right === :Reflect
                # Ghost cell i = N_r + 1:
                ρd_Np1      =   ρd[N_r, j]
                ρdud_r_Np1  = - ρdud_r[N_r, j]    # Reflect 
                ρdud_Z_Np1  =   ρdud_Z[N_r, j]  
                # Ghost cell i = N_R + 2:
                ρd_Np2      =   ρd[N_r-1, j]
                ρdud_r_Np2  = - ρdud_r[N_r-1, j]  # Reflect 
                ρdud_Z_Np2  =   ρdud_Z[N_r-1, j]  
            elseif BC_r_right === :Outflow
                # Ghost cell i = N_r + 1:
                ρd_Np1      =  ρd[N_r, j]
                ρdud_r_Np1  =  max( ρdud_r[N_r, j], 0.0 )
                ρdud_Z_Np1  =  ρdud_Z[N_r, j]
                # Ghost cell i = N_R + 2:
                ρd_Np2      =  ρd_Np1
                ρdud_r_Np2  =  ρdud_r_Np1
                ρdud_Z_Np2  =  ρdud_Z_Np1
            end

            # Get dust velocities in ghost cells:
            ud_r_Np1, ud_Z_Np1  =  GetDustVelocities( ρd_Np1, ρdud_r_Np1, ρdud_Z_Np1 )
            ud_r_Np2, ud_Z_Np2  =  GetDustVelocities( ρd_Np2, ρdud_r_Np2, ρdud_Z_Np2 )

            ##############################################################################
            ##############         INTERFACE i = N_r - 1/2         #######################
            ##############################################################################

            # Reconstruct variables at interface r_{N_r-1/2} (i.e. r[N_r]):
            ρd_L, ρd_R, ud_r_L, ud_r_R, ud_Z_L, ud_Z_R = ReconstructInterface_Dust( ρd[N_r-1,j],   ρd[N_r,j],   ρd_Np1,    ρd[N_r-2,j], 
                                                                                    ud_r[N_r-1,j], ud_r[N_r,j], ud_r_Np1,  ud_r[N_r-2,j],
                                                                                    ud_Z[N_r-1,j], ud_Z[N_r,j], ud_Z_Np1,  ud_Z[N_r-2,j],
                                                                                    rCoM[N_r-1],   rCoM[N_r],   rCoM_Np1,  rCoM[N_r-2], 
                                                                                    r[N_r]      # Interface r_{N_r-1/2}
                                                                                    )
            
            # Fluxes at i = N_r - 1/2:
            F_ρd_r[N_r,j], F_ρdud_r_r[N_r,j], F_ρdud_Z_r[N_r,j] = GetHLLFlux_Dust( ρd_L, ρd_R, ud_r_L, ud_r_R, ud_Z_L, ud_Z_R )


            ##############################################################################
            ##############         INTERFACE i = N_r + 1/2         #######################
            ##############################################################################

            # Reconstruct variables at interface r_{N_r+1/2} (i.e. r[N_r+1]):
            ρd_L, ρd_R, ud_r_L, ud_r_R, ud_Z_L, ud_Z_R = ReconstructInterface_Dust( ρd[N_r,j],   ρd_Np1,    ρd_Np2,    ρd[N_r-1,j], 
                                                                                    ud_r[N_r,j], ud_r_Np1,  ud_r_Np2,  ud_r[N_r-1,j],
                                                                                    ud_Z[N_r,j], ud_Z_Np1,  ud_Z_Np2,  ud_Z[N_r-1,j],
                                                                                    rCoM[N_r],   rCoM_Np1,  rCoM_Np2,  rCoM[N_r-1], 
                                                                                    r[N_r+1]      # Interface r_{N_r+1/2}
                                                                                    )

            # Fluxes at i = N_r + 1/2:
            F_ρd_r[N_r+1,j], F_ρdud_r_r[N_r+1,j], F_ρdud_Z_r[N_r+1,j] = GetHLLFlux_Dust( ρd_L, ρd_R, ud_r_L, ud_r_R, ud_Z_L, ud_Z_R )


            ####  UPDATE ρd AND ρdud  ##############################################################
            @simd for i in dN_r 
                ρd_new  = UpdateConservativeQuantity( ρd[i,j],  Δt, inv_ΔV[i,j], 
                                                      A_r[i,j], A_r[i+1,j], 
                                                      F_ρd_r[i,j], F_ρd_r[i+1,j] )

                if ρd_new > ρd_floor
                    ρd[i,j]     = ρd_new

                    ρdud_r[i,j] = UpdateConservativeQuantity( ρdud_r[i,j],  Δt, inv_ΔV[i,j], 
                                                              A_r[i,j], A_r[i+1,j], 
                                                              F_ρdud_r_r[i,j], F_ρdud_r_r[i+1,j] )


                    ρdud_Z[i,j] = UpdateConservativeQuantity( ρdud_Z[i,j],  Δt, inv_ΔV[i,j], 
                                                              A_r[i,j], A_r[i+1,j], 
                                                              F_ρdud_Z_r[i,j], F_ρdud_Z_r[i+1,j] )
                else
                    ρd[i,j]     = ρd_floor
                    ρdud_r[i,j] = 0.0
                    ρdud_Z[i,j] = 0.0
                end

            end # End of i-loop 
        end # End of inbounds 
    end # End of j-loop

    return nothing 
end


""" This function advects the dust in the Z-direction, 
    using a forward Euler method. """
function ForwardEulerDust_Z!( Δt::Float64,
                              ρd::AbstractArray{Float64, 2},
                          ρdud_r::AbstractArray{Float64, 2},
                          ρdud_Z::AbstractArray{Float64, 2},
                            ud_r::AbstractArray{Float64, 2},
                            ud_Z::AbstractArray{Float64, 2},
                          F_ρd_Z::AbstractArray{Float64, 2},
                      F_ρdud_r_Z::AbstractArray{Float64, 2},
                      F_ρdud_Z_Z::AbstractArray{Float64, 2},
                          inv_ΔV::AbstractArray{Float64, 2},
                             A_Z::AbstractArray{Float64, 2},
                               Z::AbstractVector{Float64},
                            ZCoM::AbstractVector{Float64},

                        BC_Z_bot::Symbol,
                        BC_Z_top::Symbol
                            )

    N_r, N_Z = size( ρd )
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z
    dN_Zint  = 2:(N_Z-2)

    # Needed ghost cell CoM radii, obtained by mirroring:
    @inbounds begin 
        # Ghost CoM Z-positions mirrored around Z[1] = Z_{1/2}:
        ZCoM_0   = Z[1] - (ZCoM[1] - Z[1])
        ZCoM_m1  = Z[1] - (ZCoM[2] - Z[1])

        # Ghost CoM Z-positions mirrored around Z[N_Z+1] = Z_{N_Z+1/2}:
        ZCoM_Np1  = Z[N_Z+1] - (ZCoM[N_Z] - Z[N_Z+1])
        ZCoM_Np2  = Z[N_Z+1] - (ZCoM[N_Z-1] - Z[N_Z+1])
    end

    # Update dust velocities for all cells (excluding ghost cells):
    UpdateDustVelocities!( ρd, ρdud_r, ρdud_Z, ud_r, ud_Z )


    # Compute fluxes and update J and H:
    @threads for i in dN_r
        @inbounds begin
            ##############################################################################
            ####################         INTERFACE j = 1/2         #######################
            ##############################################################################

            if BC_Z_bot === :Reflect
                # Ghost cell j = 0:
                ρd_0      =   ρd[i,1]
                ρdud_Z_0  = - ρdud_Z[i,1]   # Reflect
                ρdud_r_0  =   ρdud_r[i,1] 
                # Ghost cell j = -1 (mirror of cell j = 2):
                ρd_m1     =   ρd[i,2]
                ρdud_Z_m1 = - ρdud_Z[i,2]   # Reflect
                ρdud_r_m1 =   ρdud_r[i,2] 
            elseif BC_Z_bot === :Outflow 
                # Ghost cell j = 0:
                ρd_0      =   ρd[i,1]
                ρdud_Z_0  =   min( ρdud_Z[i,1], 0.0 )  
                ρdud_r_0  =   ρdud_r[i,1] 
                # Ghost cell j = -1:
                ρd_m1     =   ρd_0
                ρdud_Z_m1 =   ρdud_Z_0 
                ρdud_r_m1 =   ρdud_r_0 
            end

            # Get dust velocities in ghost cells:
            ud_r_m1, ud_Z_m1 =  GetDustVelocities( ρd_m1, ρdud_r_m1, ρdud_Z_m1 )
            ud_r_0,  ud_Z_0  =  GetDustVelocities( ρd_0,  ρdud_r_0,  ρdud_Z_0 )

            # Reconstruct variables at interface Z_{1/2} (i.e. Z[1]):
            ρd_L, ρd_R, ud_r_L, ud_r_R, ud_Z_L, ud_Z_R = ReconstructInterface_Dust( ρd_0,    ρd[i,1],   ρd[i,2],   ρd_m1, 
                                                                                    ud_r_0,  ud_r[i,1], ud_r[i,2], ud_r_m1,
                                                                                    ud_Z_0,  ud_Z[i,1], ud_Z[i,2], ud_Z_m1,
                                                                                    ZCoM_0,  ZCoM[1],   ZCoM[2],   ZCoM_m1, 
                                                                                    Z[1]               # Interface Z_{1/2}
                                                                                    )

            # Fluxes at j = 1/2:
            F_ρd_Z[i,1], F_ρdud_r_Z[i,1], F_ρdud_Z_Z[i,1] = GetHLLFlux_Dust( ρd_L, ρd_R, ud_Z_L, ud_Z_R, ud_r_L, ud_r_R )

            ##############################################################################
            ##############          INTERFACE j = 3/2          ###########################
            ##############################################################################

            # Reconstruct variables at interface Z_{3/2} (i.e. Z[2]):
            ρd_L, ρd_R, ud_r_L, ud_r_R, ud_Z_L, ud_Z_R = ReconstructInterface_Dust( ρd[i,1],    ρd[i,2],   ρd[i,3],   ρd_0, 
                                                                                    ud_r[i,1],  ud_r[i,2], ud_r[i,3], ud_r_0,
                                                                                    ud_Z[i,1],  ud_Z[i,2], ud_Z[i,3], ud_Z_0,
                                                                                    ZCoM[1],    ZCoM[2],   ZCoM[3],   ZCoM_0, 
                                                                                    Z[2]               # Interface Z_{3/2}
                                                                                    )

            # Fluxes at j = 3/2:
            F_ρd_Z[i,2], F_ρdud_r_Z[i,2], F_ρdud_Z_Z[i,2] = GetHLLFlux_Dust( ρd_L, ρd_R, ud_Z_L, ud_Z_R, ud_r_L, ud_r_R )

        end # End of inbounds 
    end # End of i-loop 

    @threads for j in dN_Zint
        @inbounds @simd for i in dN_r
            jp1 = j + 1
            jp2 = j + 2
            jm1 = j - 1

            # Reconstruct at Z_{j+1/2}:
            ρd_L, ρd_R, ud_r_L, ud_r_R, ud_Z_L, ud_Z_R = ReconstructInterface_Dust( ρd[i,j],    ρd[i,jp1],   ρd[i,jp2],   ρd[i,jm1], 
                                                                                    ud_r[i,j],  ud_r[i,jp1], ud_r[i,jp2], ud_r[i,jm1],
                                                                                    ud_Z[i,j],  ud_Z[i,jp1], ud_Z[i,jp2], ud_Z[i,jm1],
                                                                                    ZCoM[j],    ZCoM[jp1],   ZCoM[jp2],   ZCoM[jm1], 
                                                                                    Z[jp1]               # Interface Z_{j+1/2}
                                                                                    )

            # Fluxes at j + 1/2:
            F_ρd_Z[i,jp1], F_ρdud_r_Z[i,jp1], F_ρdud_Z_Z[i,jp1] = GetHLLFlux_Dust( ρd_L, ρd_R, ud_Z_L, ud_Z_R, ud_r_L, ud_r_R )

        end # End of i-loop 
    end # End of j-loop 

    @threads for i in dN_r
        @inbounds begin
            # Boundary conditions at interface j = N_Z + 1/2:
            if BC_Z_top === :Reflect
                # Ghost cell j = N_Z + 1:
                ρd_Np1      =   ρd[i,N_Z]
                ρdud_Z_Np1  = - ρdud_Z[i,N_Z]    # Reflect
                ρdud_r_Np1  =   ρdud_r[i,N_Z] 
                # Ghost cell j = N_Z + 2 (mirror of cell j = N_Z - 1):
                ρd_Np2      =   ρd[i,N_Z-1]
                ρdud_Z_Np2  = - ρdud_Z[i,N_Z-1]  # Reflect
                ρdud_r_Np2  =   ρdud_r[i,N_Z-1] 
            elseif BC_Z_top === :Outflow
                # Ghost cell j = N_Z + 1:
                ρd_Np1      =   ρd[i,N_Z]
                ρdud_Z_Np1  =   max( ρdud_Z[i,N_Z], 0.0 )    
                ρdud_r_Np1  =   ρdud_r[i,N_Z]  
                # Ghost cell j = N_Z + 2 (copy of cell j = N_Z + 1):
                ρd_Np2      =   ρd_Np1
                ρdud_Z_Np2  =   ρdud_Z_Np1
                ρdud_r_Np2  =   ρdud_r_Np1 
            end

            ##############################################################################
            ##############         INTERFACE j = N_Z - 1/2         #######################
            ##############################################################################

            # Get dust velocities in ghost cells:
            ud_r_Np1, ud_Z_Np1 = GetDustVelocities( ρd_Np1, ρdud_r_Np1, ρdud_Z_Np1 )
            ud_r_Np2, ud_Z_Np2 = GetDustVelocities( ρd_Np2, ρdud_r_Np2, ρdud_Z_Np2 )

            # Reconstruct at Z_{N_Z-1/2} (i.e. Z[N_Z])::
            ρd_L, ρd_R, ud_r_L, ud_r_R, ud_Z_L, ud_Z_R = ReconstructInterface_Dust( ρd[i,N_Z-1],    ρd[i,N_Z],   ρd_Np1,     ρd[i,N_Z-2], 
                                                                                    ud_r[i,N_Z-1],  ud_r[i,N_Z], ud_r_Np1,   ud_r[i,N_Z-2],
                                                                                    ud_Z[i,N_Z-1],  ud_Z[i,N_Z], ud_Z_Np1,   ud_Z[i,N_Z-2],
                                                                                    ZCoM[N_Z-1],    ZCoM[N_Z],   ZCoM_Np1,   ZCoM[N_Z-2], 
                                                                                    Z[N_Z]               # Interface Z_{N_Z-1/2}
                                                                                    )

            # Fluxes at j = N_Z - 1/2:
            F_ρd_Z[i,N_Z], F_ρdud_r_Z[i,N_Z], F_ρdud_Z_Z[i,N_Z] = GetHLLFlux_Dust( ρd_L, ρd_R, ud_Z_L, ud_Z_R, ud_r_L, ud_r_R )

            ##############################################################################
            ###############        INTERFACE j = N_Z + 1/2         #######################
            ##############################################################################

            # Reconstruct variables at interface Z_{N_Z+1/2} (i.e. Z[N_Z+1]):
            ρd_L, ρd_R, ud_r_L, ud_r_R, ud_Z_L, ud_Z_R = ReconstructInterface_Dust( ρd[i,N_Z],    ρd_Np1,     ρd_Np2,     ρd[i,N_Z-1], 
                                                                                    ud_r[i,N_Z],  ud_r_Np1,   ud_r_Np2,   ud_r[i,N_Z-1],
                                                                                    ud_Z[i,N_Z],  ud_Z_Np1,   ud_Z_Np2,   ud_Z[i,N_Z-1],
                                                                                    ZCoM[N_Z],    ZCoM_Np1,   ZCoM_Np2,   ZCoM[N_Z-1], 
                                                                                    Z[N_Z+1]               # Interface Z_{N_Z-1/2}
                                                                                    )

            # Fluxes at j = N_Z - 1/2:
            F_ρd_Z[i,N_Z+1], F_ρdud_r_Z[i,N_Z+1], F_ρdud_Z_Z[i,N_Z+1] = GetHLLFlux_Dust( ρd_L, ρd_R, ud_Z_L, ud_Z_R, ud_r_L, ud_r_R )

        end # End of inbounds 
    end # End of i-loop 

    ####  UPDATE ρd AND ρdud  ##############################################################
    @threads for j in dN_Z
        @inbounds @simd for i in dN_r
            ρd_new      = UpdateConservativeQuantity( ρd[i,j],  Δt, inv_ΔV[i,j], 
                                                      A_Z[i,j], A_Z[i,j+1], 
                                                      F_ρd_Z[i,j], F_ρd_Z[i,j+1] )

            if ρd_new > ρd_floor
                ρd[i,j]     = ρd_new

                ρdud_r[i,j] = UpdateConservativeQuantity( ρdud_r[i,j],  Δt, inv_ΔV[i,j], 
                                                          A_Z[i,j], A_Z[i,j+1], 
                                                          F_ρdud_r_Z[i,j], F_ρdud_r_Z[i,j+1] )


                ρdud_Z[i,j] = UpdateConservativeQuantity( ρdud_Z[i,j],  Δt, inv_ΔV[i,j], 
                                                          A_Z[i,j], A_Z[i,j+1], 
                                                          F_ρdud_Z_Z[i,j], F_ρdud_Z_Z[i,j+1] )

            else
                ρd[i,j]     = ρd_floor
                ρdud_r[i,j] = 0.0
                ρdud_Z[i,j] = 0.0
            end

        end
    end

    return nothing 
end


""" This function performs a second-order RK2 update in the r-direction
    for dust. """
function RK2_Dust_r!( Δt::Float64,
                      ρd::AbstractArray{Float64, 2},
                  ρdud_r::AbstractArray{Float64, 2},
                  ρdud_Z::AbstractArray{Float64, 2},
                ρd_stage::AbstractArray{Float64, 2},
            ρdud_r_stage::AbstractArray{Float64, 2},
            ρdud_Z_stage::AbstractArray{Float64, 2},
                    ud_r::AbstractArray{Float64, 2},
                    ud_Z::AbstractArray{Float64, 2},
                  F_ρd_r::AbstractArray{Float64, 2},
              F_ρdud_r_r::AbstractArray{Float64, 2},
              F_ρdud_Z_r::AbstractArray{Float64, 2},
                  inv_ΔV::AbstractArray{Float64, 2},
                     A_r::AbstractArray{Float64, 2},
                       r::AbstractVector{Float64},
                    rCoM::AbstractVector{Float64},
                         
               BC_r_left::Symbol,
              BC_r_right::Symbol )


    # Save the state at the start, before the update (needed later in RK2):
    @. ρd_stage        = ρd
    @. ρdud_r_stage    = ρdud_r
    @. ρdud_Z_stage    = ρdud_Z 

    # Step 1: Forward Euler step in r:  U → U^(1)
    ForwardEulerDust_r!( Δt, ρd, ρdud_r, ρdud_Z,
                         ud_r, ud_Z,
                         F_ρd_r, F_ρdud_r_r, F_ρdud_Z_r, 
                         inv_ΔV, A_r, r, rCoM,
                         BC_r_left, BC_r_right )

    # Step 2: Forward Euler step in r: U^(1) -> U^(2)
    ForwardEulerDust_r!( Δt, ρd, ρdud_r, ρdud_Z,
                         ud_r, ud_Z,
                         F_ρd_r, F_ρdud_r_r, F_ρdud_Z_r, 
                         inv_ΔV, A_r, r, rCoM,
                         BC_r_left, BC_r_right )

    # Step 3: Combine to get RK2 update:
    #
    # U^(n+1) = 0.5 × ( U_start + U^(2) )
    #
    # Note: ρd, ρdud_r, ρdud_Z contain U^(2).
    @. ρd       = 0.5 * ( ρd_stage      + ρd )
    @. ρdud_r   = 0.5 * ( ρdud_r_stage  + ρdud_r )
    @. ρdud_Z   = 0.5 * ( ρdud_Z_stage  + ρdud_Z )

    return nothing 
end


""" This function performs a second-order RK2 update in the Z-direction
    for dust. """
function RK2_Dust_Z!( Δt::Float64,
                      ρd::AbstractArray{Float64, 2},
                  ρdud_r::AbstractArray{Float64, 2},
                  ρdud_Z::AbstractArray{Float64, 2},
                ρd_stage::AbstractArray{Float64, 2},
            ρdud_r_stage::AbstractArray{Float64, 2},
            ρdud_Z_stage::AbstractArray{Float64, 2},
                    ud_r::AbstractArray{Float64, 2},
                    ud_Z::AbstractArray{Float64, 2},
                  F_ρd_Z::AbstractArray{Float64, 2},
              F_ρdud_r_Z::AbstractArray{Float64, 2},
              F_ρdud_Z_Z::AbstractArray{Float64, 2},
                  inv_ΔV::AbstractArray{Float64, 2},
                     A_Z::AbstractArray{Float64, 2},
                       Z::AbstractVector{Float64},
                    ZCoM::AbstractVector{Float64},

                BC_Z_bot::Symbol,
                BC_Z_top::Symbol )

    # Save the state at the start, before the update (needed later in RK2):
    @. ρd_stage        = ρd
    @. ρdud_r_stage    = ρdud_r
    @. ρdud_Z_stage    = ρdud_Z 

    # Step 1: Forward Euler step in Z:  U → U^(1)
    ForwardEulerDust_Z!( Δt, ρd, ρdud_r, ρdud_Z,
                         ud_r, ud_Z,
                         F_ρd_Z, F_ρdud_r_Z, F_ρdud_Z_Z, 
                         inv_ΔV, A_Z, Z, ZCoM,
                         BC_Z_bot, BC_Z_top )

    # Step 2: Forward Euler in Z again: U^(1) → U^(2)
    ForwardEulerDust_Z!( Δt, ρd, ρdud_r, ρdud_Z,
                         ud_r, ud_Z,
                         F_ρd_Z, F_ρdud_r_Z, F_ρdud_Z_Z, 
                         inv_ΔV, A_Z, Z, ZCoM,
                         BC_Z_bot, BC_Z_top )

    # Step 3: Combine to get RK2 update:
    #
    # U^(n+1) = 0.5 × ( U_start + U^(2) )
    #
    # Note: ρd, ρdud_r, ρdud_Z contain U^(2).
    @. ρd       = 0.5 * ( ρd_stage     + ρd )
    @. ρdud_r   = 0.5 * ( ρdud_r_stage + ρdud_r )
    @. ρdud_Z   = 0.5 * ( ρdud_Z_stage + ρdud_Z )

    return nothing 
end


@inline function AdvectDust_rfirst( Δt::Float64,
                                    ρd::AbstractArray{Float64, 2},
                                ρd_old::AbstractArray{Float64, 2},
                                ρdud_r::AbstractArray{Float64, 2},
                                ρdud_Z::AbstractArray{Float64, 2},
                              ρd_stage::AbstractArray{Float64, 2},
                          ρdud_r_stage::AbstractArray{Float64, 2},
                          ρdud_Z_stage::AbstractArray{Float64, 2},
                                  ud_r::AbstractArray{Float64, 2},
                                  ud_Z::AbstractArray{Float64, 2},
                                F_ρd_r::AbstractArray{Float64, 2},
                            F_ρdud_r_r::AbstractArray{Float64, 2},
                            F_ρdud_Z_r::AbstractArray{Float64, 2},
                                F_ρd_Z::AbstractArray{Float64, 2},
                            F_ρdud_r_Z::AbstractArray{Float64, 2},
                            F_ρdud_Z_Z::AbstractArray{Float64, 2},
                                inv_ΔV::AbstractArray{Float64, 2},
                                   A_r::AbstractArray{Float64, 2},
                                   A_Z::AbstractArray{Float64, 2},
                                     r::AbstractVector{Float64},
                                  rCoM::AbstractVector{Float64},
                                     Z::AbstractVector{Float64},
                                  ZCoM::AbstractVector{Float64},
 
                             BC_r_left::Symbol,
                            BC_r_right::Symbol,
                              BC_Z_bot::Symbol,
                              BC_Z_top::Symbol
                                )

    Δt_half = 0.5 * Δt

    ################################################################################
    ############                                     ###############################
    ############  1. FIRST HALF-STEP IN r-DIRECTION  ###############################
    ############                                     ###############################
    ################################################################################

    RK2_Dust_r!( Δt_half, ρd, ρdud_r, ρdud_Z,
                 ρd_stage, ρdud_r_stage, ρdud_Z_stage,
                 ud_r, ud_Z,
                 F_ρd_r, F_ρdud_r_r, F_ρdud_Z_r, 
                 inv_ΔV, A_r, r, rCoM,
                 BC_r_left, BC_r_right )

    ################################################################################
    ############                                     ###############################
    ############  2. FULL STEP IN Z-DIRECTION        ###############################
    ############                                     ###############################
    ################################################################################

    RK2_Dust_Z!( Δt, ρd, ρdud_r, ρdud_Z,
                 ρd_stage, ρdud_r_stage, ρdud_Z_stage,
                 ud_r, ud_Z,
                 F_ρd_Z, F_ρdud_r_Z, F_ρdud_Z_Z, 
                 inv_ΔV, A_Z, Z, ZCoM,
                 BC_Z_bot, BC_Z_top )

    ################################################################################
    ############                                     ###############################
    ############  3. FINAL HALF-STEP IN r-DIRECTION  ###############################
    ############                                     ###############################
    ################################################################################

    RK2_Dust_r!( Δt_half, ρd, ρdud_r, ρdud_Z,
                 ρd_stage, ρdud_r_stage, ρdud_Z_stage,
                 ud_r, ud_Z,
                 F_ρd_r, F_ρdud_r_r, F_ρdud_Z_r, 
                 inv_ΔV, A_r, r, rCoM,
                 BC_r_left, BC_r_right )

    # Update old dust densities:
    ρd_old .= ρd 

    return nothing 
end



@inline function AdvectDust_Zfirst( Δt::Float64,
                                    ρd::AbstractArray{Float64, 2},
                                ρd_old::AbstractArray{Float64, 2},
                                ρdud_r::AbstractArray{Float64, 2},
                                ρdud_Z::AbstractArray{Float64, 2},
                              ρd_stage::AbstractArray{Float64, 2},
                          ρdud_r_stage::AbstractArray{Float64, 2},
                          ρdud_Z_stage::AbstractArray{Float64, 2},
                                  ud_r::AbstractArray{Float64, 2},
                                  ud_Z::AbstractArray{Float64, 2},
                                F_ρd_r::AbstractArray{Float64, 2},
                            F_ρdud_r_r::AbstractArray{Float64, 2},
                            F_ρdud_Z_r::AbstractArray{Float64, 2},
                                F_ρd_Z::AbstractArray{Float64, 2},
                            F_ρdud_r_Z::AbstractArray{Float64, 2},
                            F_ρdud_Z_Z::AbstractArray{Float64, 2},
                                inv_ΔV::AbstractArray{Float64, 2},
                                   A_r::AbstractArray{Float64, 2},
                                   A_Z::AbstractArray{Float64, 2},
                                     r::AbstractVector{Float64},
                                  rCoM::AbstractVector{Float64},
                                     Z::AbstractVector{Float64},
                                  ZCoM::AbstractVector{Float64},
 
                             BC_r_left::Symbol,
                            BC_r_right::Symbol,
                              BC_Z_bot::Symbol,
                              BC_Z_top::Symbol
                                )

    Δt_half = 0.5 * Δt

    ################################################################################
    ############                                     ###############################
    ############  1. FIRST HALF-STEP IN Z-DIRECTION  ###############################
    ############                                     ###############################
    ################################################################################

    RK2_Dust_Z!( Δt_half, ρd, ρdud_r, ρdud_Z,
                 ρd_stage, ρdud_r_stage, ρdud_Z_stage,
                 ud_r, ud_Z,
                 F_ρd_Z, F_ρdud_r_Z, F_ρdud_Z_Z, 
                 inv_ΔV, A_Z, Z, ZCoM,
                 BC_Z_bot, BC_Z_top )

    ################################################################################
    ############                                     ###############################
    ############  2. FULL STEP IN r-DIRECTION        ###############################
    ############                                     ###############################
    ################################################################################

    RK2_Dust_r!( Δt, ρd, ρdud_r, ρdud_Z,
                 ρd_stage, ρdud_r_stage, ρdud_Z_stage,
                 ud_r, ud_Z,
                 F_ρd_r, F_ρdud_r_r, F_ρdud_Z_r, 
                 inv_ΔV, A_r, r, rCoM,
                 BC_r_left, BC_r_right )

    ################################################################################
    ############                                     ###############################
    ############  3. FINAL HALF-STEP IN Z-DIRECTION  ###############################
    ############                                     ###############################
    ################################################################################

    RK2_Dust_Z!( Δt_half, ρd, ρdud_r, ρdud_Z,
                 ρd_stage, ρdud_r_stage, ρdud_Z_stage,
                 ud_r, ud_Z,
                 F_ρd_Z, F_ρdud_r_Z, F_ρdud_Z_Z, 
                 inv_ΔV, A_Z, Z, ZCoM,
                 BC_Z_bot, BC_Z_top )

    # Update old dust densities:
    ρd_old .= ρd 

    return nothing 
end


@inline function AdvectDust_RK2!( Δt::Float64,
                                  ρd::AbstractArray{Float64, 2},
                              ρd_old::AbstractArray{Float64, 2},
                              ρdud_r::AbstractArray{Float64, 2},
                              ρdud_Z::AbstractArray{Float64, 2},
                            ρd_stage::AbstractArray{Float64, 2},
                        ρdud_r_stage::AbstractArray{Float64, 2},
                        ρdud_Z_stage::AbstractArray{Float64, 2},
                                ud_r::AbstractArray{Float64, 2},
                                ud_Z::AbstractArray{Float64, 2},
                              F_ρd_r::AbstractArray{Float64, 2},
                          F_ρdud_r_r::AbstractArray{Float64, 2},
                          F_ρdud_Z_r::AbstractArray{Float64, 2},
                              F_ρd_Z::AbstractArray{Float64, 2},
                          F_ρdud_r_Z::AbstractArray{Float64, 2},
                          F_ρdud_Z_Z::AbstractArray{Float64, 2},
                              inv_ΔV::AbstractArray{Float64, 2},
                                 A_r::AbstractArray{Float64, 2},
                                 A_Z::AbstractArray{Float64, 2},
                                   r::AbstractVector{Float64},
                                rCoM::AbstractVector{Float64},
                                   Z::AbstractVector{Float64},
                                ZCoM::AbstractVector{Float64},
 
                           BC_r_left::Symbol,
                          BC_r_right::Symbol,
                            BC_Z_bot::Symbol,
                            BC_Z_top::Symbol,
                              rfirst::Bool
                              )

    if rfirst 
        AdvectDust_rfirst( Δt, ρd, ρd_old, ρdud_r, ρdud_Z,
                           ρd_stage, ρdud_r_stage, ρdud_Z_stage,
                           ud_r, ud_Z, 
                           F_ρd_r, F_ρdud_r_r, F_ρdud_Z_r,
                           F_ρd_Z, F_ρdud_r_Z, F_ρdud_Z_Z,
                           inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM,
 
                           BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top )
    else
        AdvectDust_Zfirst( Δt, ρd, ρd_old, ρdud_r, ρdud_Z,
                           ρd_stage, ρdud_r_stage, ρdud_Z_stage,
                           ud_r, ud_Z, 
                           F_ρd_r, F_ρdud_r_r, F_ρdud_Z_r,
                           F_ρd_Z, F_ρdud_r_Z, F_ρdud_Z_Z,
                           inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM,
 
                           BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top )
    end

    return nothing 
end


""" This function updates dust momenta from gravity, over a time-step Δt. """
function GravityKickDust!( Δt::Float64,
                           ρd::AbstractArray{Float64, 2},
                       ρdud_r::AbstractArray{Float64, 2},
                       ρdud_Z::AbstractArray{Float64, 2}, 
                         ∂Φ∂r::AbstractArray{Float64, 2},
                         ∂Φ∂Z::AbstractArray{Float64, 2}
                            )

    N_r, N_Z = size( ρd )
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z

    @threads for j in dN_Z 
        @inbounds @simd for i in dN_r 
            # Update momenta, but only if the density is above the floor: 
            ρd_ij  =  ρd[i,j]

            if ρd_ij > ρd_floor
                ρdud_r[i,j] -=  ρd_ij * ∂Φ∂r[i,j] * Δt 
                ρdud_Z[i,j] -=  ρd_ij * ∂Φ∂Z[i,j] * Δt
            end

        end # End of i-loop 
    end # End of j-loop

    return nothing 
end


""" This function advects the dust over a global (hydro) time-step Δt_Hydro. 
    In case the dust CFL time-step 
    is shorter than Δt_Hydro, sub-cycling is used. """
@views function AdvectDust!( Δt_Hydro::Float64, 
                                 Dust::DustState,                             
                               inv_ΔV::AbstractArray{Float64, 2},
                                  A_r::AbstractArray{Float64, 2},
                                  A_Z::AbstractArray{Float64, 2},
                                    r::AbstractVector{Float64},
                                 rCoM::AbstractVector{Float64},
                                    Z::AbstractVector{Float64},
                                 ZCoM::AbstractVector{Float64},
 
                            BC_r_left::Symbol,
                           BC_r_right::Symbol,
                             BC_Z_bot::Symbol,
                             BC_Z_top::Symbol
                            )

    # Unpack dust arrays & settings:
    ρd            = Dust.ρd
    ρd_old        = Dust.ρd_old
    ρdud_r        = Dust.ρdud_r
    ρdud_Z        = Dust.ρdud_Z
    ρd_stage      = Dust.ρd_stage
    ρdud_r_stage  = Dust.ρdud_r_stage
    ρdud_Z_stage  = Dust.ρdud_Z_stage
    ud_r          = Dust.ud_r
    ud_Z          = Dust.ud_Z

    # Arrays for interface fluxes:
    F_ρd_r        = Dust.F_ρd_r
    F_ρdud_r_r    = Dust.F_ρdud_r_r
    F_ρdud_Z_r    = Dust.F_ρdud_Z_r
    F_ρd_Z        = Dust.F_ρd_Z
    F_ρdud_r_Z    = Dust.F_ρdud_r_Z
    F_ρdud_Z_Z    = Dust.F_ρdud_Z_Z

    # Advect dust:
    t = 0.0
    while t < Δt_Hydro
        # Update dust velocities (for CFL time-step calculation):
        UpdateDustVelocities!( ρd, ρdud_r, ρdud_Z, 
                               ud_r, ud_Z )
        # Get maximum time-step:
        Δt_CFL      = GetDustCFLTimeStep( ud_r, ud_Z, r, Z )
        Δt          = min( Δt_CFL, Δt_Hydro - t )

        # Select Strang-split order:
        Dust.rfirst = !Dust.rfirst

        # Advect dust:
        AdvectDust_RK2!( Δt, ρd, ρd_old, 
                         ρdud_r, ρdud_Z,
                         ρd_stage, ρdud_r_stage, ρdud_Z_stage,
                         ud_r, ud_Z, 
                         F_ρd_r, F_ρdud_r_r, F_ρdud_Z_r, 
                         F_ρd_Z, F_ρdud_r_Z, F_ρdud_Z_Z,
                         inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, 
                         BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top, 
                         Dust.rfirst )

        # Update dust velocities again (to keep it synced for later use):
        UpdateDustVelocities!( ρd, ρdud_r, ρdud_Z, 
                               ud_r, ud_Z )

        # Update time: 
        t += Δt
    end

    return nothing 
end


""" This function advects the dust over a global (hydro) time-step Δt_Hydro. 
    In case the dust CFL time-step is shorter than Δt_Hydro, sub-cycling is used. 
    Gravitational acceleration is also taken into account in this version, 
    with Strang-splitting. """
@views function AdvectDust!( Δt_Hydro::Float64, 
                                 Dust::DustState,  
                            GravState::GravityState,                              
                               inv_ΔV::AbstractArray{Float64, 2},
                                  A_r::AbstractArray{Float64, 2},
                                  A_Z::AbstractArray{Float64, 2},
                                    r::AbstractVector{Float64},
                                 rCoM::AbstractVector{Float64},
                                    Z::AbstractVector{Float64},
                                 ZCoM::AbstractVector{Float64},
 
                            BC_r_left::Symbol,
                           BC_r_right::Symbol,
                             BC_Z_bot::Symbol,
                             BC_Z_top::Symbol
                            )

    # Unpack dust arrays & settings:
    ρd            = Dust.ρd
    ρd_old        = Dust.ρd_old
    ρdud_r        = Dust.ρdud_r
    ρdud_Z        = Dust.ρdud_Z
    ρd_stage      = Dust.ρd_stage
    ρdud_r_stage  = Dust.ρdud_r_stage
    ρdud_Z_stage  = Dust.ρdud_Z_stage
    ud_r          = Dust.ud_r
    ud_Z          = Dust.ud_Z

    # Arrays for interface fluxes:
    F_ρd_r        = Dust.F_ρd_r
    F_ρdud_r_r    = Dust.F_ρdud_r_r
    F_ρdud_Z_r    = Dust.F_ρdud_Z_r
    F_ρd_Z        = Dust.F_ρd_Z
    F_ρdud_r_Z    = Dust.F_ρdud_r_Z
    F_ρdud_Z_Z    = Dust.F_ρdud_Z_Z

    # Gradients of gravitational potential:
    ∂Φ∂r          = GravState.∂Φ∂r
    ∂Φ∂Z          = GravState.∂Φ∂Z

    # Advect dust:
    t = 0.0
    while t < Δt_Hydro
        # Update dust velocities (for CFL time-step calculation):
        UpdateDustVelocities!( ρd, ρdud_r, ρdud_Z, 
                               ud_r, ud_Z )
        # Get maximum time-step:
        Δt_CFL      = GetDustCFLTimeStep( ud_r, ud_Z, r, Z )
        Δt          = min( Δt_CFL, Δt_Hydro - t )
        Δt_half     = 0.5 * Δt

        # Select Strang-split order:
        Dust.rfirst = !Dust.rfirst

        # Do a half-step gravity kick:
        GravityKickDust!( Δt_half, ρd, ρdud_r, ρdud_Z,
                          ∂Φ∂r, ∂Φ∂Z )

        # Advect dust over a full step:
        AdvectDust_RK2!( Δt, ρd, ρd_old, 
                         ρdud_r, ρdud_Z,
                         ρd_stage, ρdud_r_stage, ρdud_Z_stage,
                         ud_r, ud_Z, 
                         F_ρd_r, F_ρdud_r_r, F_ρdud_Z_r, 
                         F_ρd_Z, F_ρdud_r_Z, F_ρdud_Z_Z,
                         inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, 
                         BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top, 
                         Dust.rfirst )

        # Do a final half-step gravity kick:
        GravityKickDust!( Δt_half, ρd, ρdud_r, ρdud_Z,
                          ∂Φ∂r, ∂Φ∂Z )

        # Update dust velocities again (to keep it synced for later use):
        UpdateDustVelocities!( ρd, ρdud_r, ρdud_Z, 
                               ud_r, ud_Z )

        # Update time: 
        t += Δt
    end

    return nothing 
end

