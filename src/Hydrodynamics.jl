
""" FILE:           Hydrodynamics.jl

    PURPOSE:        This file contains the basic 2D hydrodynamics code, relying on a 
                    finite volume method, with third-order WENO3 cell reconstruction.
                    Strang splitting is used to dimensionally split the problem, which 
                    formally reduces the convergence to second order in space. 

                    The code can be applied to axi-symmetric problems in cylindrical coordinates, 
                    as well as 2D Cartesian problems. 
             
    REFERENCES:     (1) Mignone et al. (2007) 'PLUTO: a Numerical Code for Computational Astrophysics',
                        ApJS, 170, 1

                    (2) Mignone (2014) 'High-order conservative reconstruction schemes for finite 
                        volume methods in cylindrical and spherical coordinates', Journal of Computational 
                        Physics, 270, pp. 784-814

                    (3) Verrier et al. (2025) 'A new multifluid method for dusty astrophysical flows: 
                        Application to turbulent protostellar collapses', A&A, 701, A174
    
    """


using .Structs: HydrodynamicsState, GhostPaddedGrid, GravityState, GravityCoeffs

using .ConstantsAndFunctions # Needed constants 
using Base.Threads           # For CPU acceleration
using Printf


export GetPrimitive, GetCFLTimeStep, 
       GetGravityTimeStep, GetRadPressureTimeStep, Getγad, Getμ, UpdateHydro!, 
       GravitySolveAndKick!, GravityKick!, UpdateTotalMass!, GenerateHydroArrays

###############################################################
#######                                                 #######
#######           H Y D R O  S E T T I N G S            #######
#######                                                 #######
###############################################################

# Below are the main settings controlling the hydro solver,
# which you can change.

const CFL::Float64         =  0.4        # CFL factor for gas tim                            (< 1, default: 0.4)
const C_Grav::Float64      =  0.1        # Sets the gravity-limited time-step                (default: 0.1)
const C_RP::Float64        =  0.1        # Sets the rad. presure-limited time-step           (default: 0.1)
const ρ_floor::Float64     =  1e-10 * mH # Gas density floor                                 (default: 10⁻¹⁰ mH per cm⁻³)
const ρd_floor::Float64    =  1e-13 * mH # Dust density floor                                (default: 10⁻¹³ mH per cm⁻³)
const P_floor::Float64     =  1e-30      # Gas pressure floor                                (default: 10⁻³⁰ in cgs units)
const γmon::Float64        =  5 / 3      # Adiabatic index of monoatomic gas
const γdia::Float64        =  7 / 5      # Adiabatic index of diatomic gas
const DEBUG_Hydro::Bool    =  true       # Checks for non-physical values & errors, on if true (default: true)

export CFL, C_Grav, C_RP, ρ_floor, ρd_floor, P_floor, γmon, γdia  


###############################################################
#######                                                 #######
#######          U S E F U L  F U N C T I O N S         #######
#######                                                 #######
###############################################################


""" Converts from conservative variables (ρ, ρu_X, ρu_Y, E) to 
    primitive ones (u_X, u_Y, P).  """
@inline function GetPrimitive(  ρ::Float64, # Gas density
                             ρu_X::Float64, # Gas momentum per unit volume in r-direction
                             ρu_Y::Float64, # Gas momentum per unit volume in Z-direction
                                E::Float64, # Total gas energy 
                              γad::Float64  # Adiabatic index 
                               )
    inv_ρ = 1 / ρ
    u_X   = ρu_X * inv_ρ
    u_Y   = ρu_Y * inv_ρ
    u_sq  = u_X * u_X  +  u_Y * u_Y
    P     = (γad - 1) * ( E  -  ρ * u_sq / 2 )
    return u_X, u_Y, P
end


""" Converts from primitive variables (ρ, u_X, u_Y, P) to 
    conservative ones (ρu_X, ρu_Y, E). """
@inline @fastmath function GetConservative( ρ::Float64, # Gas density 
                                          u_X::Float64, # Gas velocity in R-direction
                                          u_Y::Float64, # Gas velocity in R-direction
                                            P::Float64, # Gas pressure 
                                          γad::Float64  # Adiabatic index
                                           )
    ρu_X = ρ * u_X 
    ρu_Y = ρ * u_Y
    E    = P / (γad - 1)  +  ρ * (u_X * u_X  +  u_Y * u_Y) / 2 
    return ρu_X, ρu_Y, E
end


""" The default CFL time-step. Computes min( Δr / (cs + u), Δr / ud ),
    over all cells. To avoid unnecessary allocations, this is 
    done in a for loop. """
@inline function GetCFLTimeStep(  X::AbstractVector{Float64}, # Interface r-values
                                  Y::AbstractVector{Float64}, # Interface Z-values
                                  ρ::AbstractMatrix{Float64}, # Gas density 
                                  P::AbstractMatrix{Float64}, # Gas pressure 
                                u_X::AbstractMatrix{Float64}, # Gas velocity in r-direction
                                u_Y::AbstractMatrix{Float64}, # Gas velocity in Z-direction 
                               ud_r::AbstractMatrix{Float64}, # Dust velocity in r-direction
                               ud_Z::AbstractMatrix{Float64}, # Dust velocity in Z-direction 
                                γad::AbstractMatrix{Float64}  # Adiabatic index 
                                 )

    N_X, N_Y    = size( ρ )  # Grid size
    dN_X        = 1:N_X
    dN_Y        = 1:N_Y

    if DEBUG_Hydro
        @assert length(X) == N_X + 1 "X must have N_X + 1 entries"
        @assert length(Y) == N_Y + 1 "Y must have N_Y + 1 entries"
    end

    min_ΔtHydro = Inf64  # Initialize the minimum local time-scale

    @inbounds for j in dN_Y
        ΔY                = Y[j+1] - Y[j] # Vertical cell width
        local_min_ΔtHydro = min_ΔtHydro   # Local variable for tracking smallest CFL time

        @simd for i in dN_X
            # Compute the horizontal cell width (same for all j-indices):
            ΔX = X[i+1] - X[i]

            # Compute the sound speed in the cell:
            if DEBUG_Hydro
                @assert P[i,j] > 0.0 && ρ[i,j] > 0.0 "P[$(i),$(j)] = $(P[i,j]) and ρ[$(i),$(j)] = $(ρ[i,j]) in GetCFLTimeStep."
            end

            cs = sqrt( γad[i,j] * P[i,j] / ρ[i,j] )

            # Compute the local CFL time-scale:
            Δt = min( ΔX / (cs + abs(u_X[i,j])), ΔX / abs(ud_r[i,j]), 
                      ΔY / (cs + abs(u_Y[i,j])), ΔY / abs(ud_Z[i,j])  )
            
            if Δt < local_min_ΔtHydro 
                # Update the smallest time-scale found:
                local_min_ΔtHydro = Δt 
            end
        end

        min_ΔtHydro = min( min_ΔtHydro, local_min_ΔtHydro )
    end
    
    # The final default CFL time-step is simply the CFL factor times 
    # min_ΔtHydro:
    return CFL * min_ΔtHydro
end


@inline function GetGravityTimeStep( X::AbstractVector{Float64}, # Interface X-values
                                     Y::AbstractVector{Float64}, # Interface Y-values
                                  ∂Φ∂X::AbstractMatrix{Float64}, # X-component of ∇Φ  
                                  ∂Φ∂Y::AbstractMatrix{Float64}  # Y-component of ∇Φ 
                                    )

    N_X, N_Y = size( ∂Φ∂X )
    dN_X     = 1:N_X 
    dN_Y     = 1:N_Y 

    min_Δt = Inf64  # Initialize the minimum local time-scale

    @inbounds for j in dN_Y
        ΔY           = Y[j+1] - Y[j] # Vertical cell width
        local_min_Δt = min_Δt        # Local variable for tracking smallest time-step 

        @simd for i in dN_X 

            # Compute the horizontal cell width (same for all j-indices):
            ΔX = X[i+1] - X[i]

            # Compute the local acceleration time-scale:
            Δt_g  = min( sqrt( 2 * ΔX / abs(∂Φ∂X[i,j]) ), 
                         sqrt( 2 * ΔY / abs(∂Φ∂Y[i,j]) ) )
            
            if Δt_g < local_min_Δt
                # Update the smallest time-scale found:
                local_min_Δt = Δt_g
            end

        end

        min_Δt = min( min_Δt, local_min_Δt )
    end

    return C_Grav * min_Δt
end


""" This function computes the maximum time-step that can be taken 
    to retain accuracy, for a given radiation pressure force 
    per unit volume f_rad = (f_rad_X, f_rad_Y). Computed as:
    
    Δt ≤ C_RP × sqrt( 2 Δr / abs(a_rad) ) 
       = C_RP × sqrt( 2 Δr ρ / abs(f_rad) ) """
@inline function GetRadPressureTimeStep( X::AbstractVector{Float64}, # Interface X-values
                                         Y::AbstractVector{Float64}, # Interface Y-values
                                         ρ::AbstractMatrix{Float64}, # Gas density 
                                   f_rad_X::AbstractMatrix{Float64}, # X-component of radiation pressure force per unit volume 
                                   f_rad_Y::AbstractMatrix{Float64}, # Y-component of radiation pressure force per unit volume 
                                         )

    N_X, N_Y = size( f_rad_X )
    dN_X     = 1:N_X 
    dN_Y     = 1:N_Y 

    min_Δt   = Inf64  # Initialize the minimum local time-scale
    @inbounds for j in dN_Y
        ΔY           = Y[j+1] - Y[j] # Vertical cell width
        local_min_Δt = min_Δt   # Local variable for tracking smallest time-step 

        @simd for i in dN_X 

            # Compute the horizontal cell width (same for all j-indices):
            ΔX = X[i+1] - X[i]

            # Compute the local acceleration time-scale:
            Δt = min( sqrt( 2 * ΔX * ρ[i,j] / abs(f_rad_X[i,j]) ), 
                      sqrt( 2 * ΔY * ρ[i,j] / abs(f_rad_Y[i,j]) ) )
            
            if Δt < local_min_Δt
                # Update the smallest time-scale found:
                local_min_Δt = Δt 
            end

        end

        min_Δt = min( min_Δt, local_min_Δt )
    end

    return C_RP * min_Δt
end


const inv_γmon_m1::Float64 = 1 / ( γmon  -  1 )
const inv_γdia_m1::Float64 = 1 / ( γdia  -  1 )

""" Adiabatic index of the gas, taking into account 
    H₂, approximating γH2 ≈ γdia. For further discussion
    and more accurate expressions, see:

    Omukai & Nishi (1998) 'Formation of primordial protostars',
    ApJ, 508, 1, 141-150

    Sharda et al. (2019) 'The role of the H₂ adiabatic index in
    the formation of the first stars', MNRAS, 490, 1, 513-526  """
@inline function Getγad( xHII::Float64,
                          xH2::Float64, 
                        xHeII::Float64,
                       xHeIII::Float64 
                        )
   
    # Get y(i) = n(i) / nH for i ∈ (H₂, H I, H II)
    yH2  = xH2;  yHII = xHII;  yHI = 1  -  xHII  -  2 * xH2

    # Next get y(i) = n(i) / nH for i ∈ (He I, He II, He III).
    # Use nHe / nH = Y/(4 × X)
    nHe_nH = Y / (4 * X)
    yHeII  = xHeII * nHe_nH
    yHeIII = xHeIII * nHe_nH
    yHeI   = ( 1  -  xHeII  -  xHeIII ) * nHe_nH

    # Finally get y(e⁻) using n(e⁻) = n(H II) + n(He II) + 2 × n(He III):
    ye     = xHII  +  ( xHeII  +  2 * xHeIII ) * nHe_nH

    # Compute ∑_i y(i) for all i, and for all i except H₂:
    ytot   = yH2  +  yHI  +  yHII  +  yHeI  +  yHeII  +  yHeIII  +  ye 
    yrest  = ytot  -  yH2

    # Final expression for adiabatic index:
    γad    = 1  +  ytot / ( yH2 * inv_γdia_m1  +  yrest * inv_γmon_m1 )

    if DEBUG_Hydro
        @assert isfinite(γad) && γad ≥ 1 "γad = $(γad), which is unphysical."
    end

    return γad
end


const Δx_Getμ::Float64 = 1e-10  # Constant threshold for considering limiting values

""" This function gives the mean molecular weight μ given
    known fractional abundances of H II, H₂, He II, and He III.

    See: Black & Bodenheimer (1975) 'Evolution of rotating 
         interstellar clouds. I. Numerical techniques.',
         ApJ, 199, 1, 619-632 """
@inline function Getμ( xHII::Float64, 
                        xH2::Float64, 
                      xHeII::Float64, 
                     xHeIII::Float64 )

    # For added numerical stability:
    xHII   = clamp( xHII,   Δx_Getμ,      1.0 - Δx_Getμ )
    xH2    = clamp( xH2,    Δx_Getμ / 2, (1.0 - Δx_Getμ) / 2)
    xHeII  = clamp( xHeII,  Δx_Getμ,      1.0 - Δx_Getμ )
    xHeIII = clamp( xHeIII, Δx_Getμ,      1.0 - Δx_Getμ )

    ##### H Y D R O G E N #####

    # Below the variables x and y are defined as 
    #
    #  x ≡ ρ(HII) / [ ρ(HII) + ρ(HI) ]
    #  y = ρ(HI) / [ ρ(HI) + ρ(H₂) ]
    #
    # These can be expressed in terms of the abundance 
    # fractions, but care must be taken for limiting values 
    # to avoid overflow or wrong results.

    if abs( xHII - 1.0 ) ≤ 2 * Δx_Getμ 
        # In this case, consider hydrogen fully 
        # ionized:
        x = 1.0; y = 1.0
    elseif abs( 2 * xH2 - 1.0 ) ≤ 2 * Δx_Getμ 
        # In this case, consider hydrogen fully molecular:
        x = 0.0; y = 0.0
    else
        # Otherwise use full formulas: 
        x = xHII / ( 1 - 2 * xH2 )
        y = (1 - xHII - 2 * xH2) / (1 - xHII)
    end

    ##### H E L I U M #####

    # Below the variables z1 and z2 are defined as 
    #
    #  z1 ≡ ρ(HeII) / [ ρ(HeII) + ρ(HeI) ]
    #  z2 = ρ(HeIII) / [ ρ(HeIII) + ρ(HeII) ]
    #
    # These can be expressed in terms of the abundance 
    # fractions, but care must be taken for limiting values 
    # to avoid overflow or wrong results.

    if abs( xHeII - 1.0 ) ≤ 2 * Δx_Getμ 
        # Consider helium singly ionized:
        z1 = 1.0; z2 = 0.0
    elseif abs( xHeIII - 1.0 ) ≤ 2 * Δx_Getμ
        # Consider heloum doubly ionized:
        z1 = 1.0; z2 = 1.0
    elseif abs( xHeIII + xHeII ) ≤ 2 * Δx_Getμ
        # Consider helium neutral:
        z1 = 0.0; z2 = 0.0
    else
        # Otherwise use full formulas:
        z1 = xHeII / ( 1 - xHeIII )
        z2 = xHeIII / ( xHeIII + xHeII )
    end

    # Get mean molecular weight:
    μ   = 4 / ( 2 * X * (1  +  y  +  2 * x * y)  +  Y * (1  +  z1  +  z1 * z2) )
    return μ
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


###############################################################
#######                                                 #######
#######      C E L L  R E C O N S T R U C T I O N       #######
#######                                                 #######
###############################################################


const ng::Int64 = 2    # Number of ghost layers per side 
const dng       = 1:ng # Convenient interval, often needed below 


const ϵ_Shock_HLL::Float64         = 1.0 # Critical value of shock-indicator for switching to HLL (default = 1.0)
const Shock_HLL::Int               = 1   # Just an arbitrary constant to track cells where we want to revert to HLL 
const ϵ_Shock_HLL_and_PLM::Float64 = 4.0 # Critical value of shock-indicator to flag as a very strong shock, 
                                         # where we revert to HLL + PLM reconstruction (default = 4.0)

@inline function UpdateShockFinder!( Shock::AbstractMatrix{Int}, 
                                         P::AbstractMatrix{Float64}
                                     )

    N_X, N_Y = size( P )
    dN_X     = 1:N_X 
    dN_Y     = 1:N_Y

    @threads for j in dN_Y
        @inbounds @simd for i in dN_X
            # Convenient indices, clamped around boundaries:
            ip1   = min( i + 1, N_X )
            im1   = max( i - 1, 1 )
            jp1   = min( j + 1, N_Y )
            jm1   = max( j - 1, 1 )

            # Compute shock indicator:
            inv_P   = 1 / max( min( P[i,j], P[im1,j], P[ip1,j], P[i,jm1], P[i,jp1] ), P_floor )
            ΔP      = ( abs( P[ip1,j] - P[im1,j] ) + abs( P[i,jp1] - P[i,jm1] ) ) * inv_P

            if ΔP ≥ ϵ_Shock_HLL_and_PLM 
                # In this case, we have a strong shock, and flag it so that we use 
                # both HLL + linear reconstruction (instead of WENO3) near it:
                Shock[i,j] = Shock_HLL_and_PLM  # (This arbitrary constant is defined in the WENO3 module)
            elseif ΔP ≥ ϵ_Shock_HLL && ΔP < ϵ_Shock_HLL_and_PLM 
                # In this case, we have a notable shock, and flag it so that we use 
                # HLL near the shock:
                Shock[i,j] = Shock_HLL 
            else
                # If no strong pressure gradients exist, we don't flag the cell:
                Shock[i,j] = 0
            end

        end # End of i-loop
    end # End of j-loop 

    return nothing 
end


@inline function SteepOrNot( Qc::Float64, Qm1::Float64, Q::Float64, Qp1::Float64 )
    if Qc > max( Q, Qp1, Qm1 ) || Qc < min( Q, Qp1, Qm1 )
        # Flag as steep:
        return 1
    else
        return 0
    end
end


""" This function recovers the geometric cell center values of 
    conservative variables (e.g. ρc) along the X-direction,
    from the cell-averaged values (e.g. ρ), using the appropriate 
    weights for third-order accuracy. """
@inline function RecoverPointwiseFromAverages_X!( ρc::AbstractMatrix{Float64},
                                               ρu_Xc::AbstractMatrix{Float64},
                                               ρu_Yc::AbstractMatrix{Float64},
                                                  Ec::AbstractMatrix{Float64},
                                                 ρdc::AbstractMatrix{Float64},
                                             ρdud_Xc::AbstractMatrix{Float64},
                                             ρdud_Yc::AbstractMatrix{Float64},
                                                  
                                                   ρ::AbstractMatrix{Float64},
                                                ρu_X::AbstractMatrix{Float64},
                                                ρu_Y::AbstractMatrix{Float64},
                                                   E::AbstractMatrix{Float64},
                                                 γad::AbstractMatrix{Float64},
                                                  ρd::AbstractMatrix{Float64},
                                              ρdud_X::AbstractMatrix{Float64},
                                              ρdud_Y::AbstractMatrix{Float64},
                                                   
                                            w_i_m1_X::AbstractVector{Float64},
                                             w_i_0_X::AbstractVector{Float64},
                                            w_i_p1_X::AbstractVector{Float64},

                                               Shock::AbstractMatrix{Int},
                                           
                                           BC_X_left::Symbol  = :Reflect,
                                          BC_X_right::Symbol  = :Outflow
                                                )

    N_X, N_Y = size( ρ )
    dN_Xint  = 2:(N_X - 1)
    dN_Y     = 1:N_Y 

    # Innermost boundary i = 1:
    @threads for j in dN_Y
        @inbounds begin
            if BC_X_left == :Reflect 
                ρ_0      =   ρ[1,j]
                ρu_X_0   = - ρu_X[1,j]
                ρu_Y_0   =   ρu_Y[1,j]
                E_0      =   E[1,j]

                ρd_0     =   ρd[1,j]
                ρdud_X_0 = - ρdud_X[1,j]
                ρdud_Y_0 =   ρdud_Y[1,j]
            elseif BC_X_left == :Outflow
                # If not reflective, we have outflow conditions: 
                ρ_0      =   ρ[1,j]
                ρu_X_0   =   ρu_X[1,j]
                ρu_Y_0   =   ρu_Y[1,j]
                E_0      =   E[1,j] 

                ρd_0     =   ρd[1,j]
                ρdud_X_0 =   ρdud_X[1,j]
                ρdud_Y_0 =   ρdud_Y[1,j]
            elseif BC_X_left == :Periodic 
                ρ_0      =   ρ[N_X,j]
                ρu_X_0   =   ρu_X[N_X,j]
                ρu_Y_0   =   ρu_Y[N_X,j]
                E_0      =   E[N_X,j] 

                ρd_0     =   ρd[N_X,j]
                ρdud_X_0 =   ρdud_X[N_X,j]
                ρdud_Y_0 =   ρdud_Y[N_X,j]
            else 
                error("No boundary conditions assigned in the X (or r) direction at i = 1!")
            end

            # RECONSTRUCT ⟨ρ⟩ᵢ → ρ(rcᵢ) AT i = 1:
            ρc[1,j]      = w_i_m1_X[1] * ρ_0  +  w_i_0_X[1] * ρ[1,j]  +  w_i_p1_X[1] * ρ[2,j]
            Steep_ρ      = SteepOrNot( ρc[1,j], ρ_0, ρ[1,j], ρ[2,j] )

            ρdc[1,j]     = w_i_m1_X[1] * ρd_0  +  w_i_0_X[1] * ρd[1,j]  +  w_i_p1_X[1] * ρd[2,j]   # Dust 
            Steep_ρd     = SteepOrNot( ρdc[1,j], ρd_0, ρd[1,j], ρd[2,j] )

            # RECONSTRUCT ⟨ρu_X⟩ᵢ → ρu_X(rcᵢ) AT i = 1:
            ρu_Xc[1,j]   = w_i_m1_X[1] * ρu_X_0  +  w_i_0_X[1] * ρu_X[1,j]  +  w_i_p1_X[1] * ρu_X[2,j]
            Steep_ρu_X   = SteepOrNot( ρu_Xc[1,j], ρu_X_0, ρu_X[1,j], ρu_X[2,j] )

            ρdud_Xc[1,j] = w_i_m1_X[1] * ρdud_X_0  +  w_i_0_X[1] * ρdud_X[1,j]  +  w_i_p1_X[1] * ρdud_X[2,j]  # Dust
            Steep_ρdud_X = SteepOrNot( ρdud_Xc[1,j], ρdud_X_0, ρdud_X[1,j], ρdud_X[2,j] )

            # RECONSTRUCT ⟨ρu_Y⟩ᵢ → ρu_Y(rcᵢ) AT i = 1:
            ρu_Yc[1,j]   = w_i_m1_X[1] * ρu_Y_0  +  w_i_0_X[1] * ρu_Y[1,j]  +  w_i_p1_X[1] * ρu_Y[2,j]
            Steep_ρu_Y   = SteepOrNot( ρu_Yc[1,j], ρu_Y_0, ρu_Y[1,j], ρu_Y[2,j] )

            ρdud_Yc[1,j] = w_i_m1_X[1] * ρdud_Y_0  +  w_i_0_X[1] * ρdud_Y[1,j]  +  w_i_p1_X[1] * ρdud_Y[2,j]
            Steep_ρdud_Y = SteepOrNot( ρdud_Yc[1,j], ρdud_Y_0, ρdud_Y[1,j], ρdud_Y[2,j] )

            # RECONSTRUCT ⟨E⟩ᵢ → E(rcᵢ) AT i = 1:
            Ec[1,j]    = w_i_m1_X[1] * E_0  +  w_i_0_X[1] * E[1,j]  +  w_i_p1_X[1] * E[2,j]
            Steep_E    = SteepOrNot( Ec[1,j], E_0, E[1,j], E[2,j] )

            # GET STEEPNESS INDICATOR:
            Steep = max( Steep_ρ, Steep_ρd, Steep_ρu_X, Steep_ρdud_X, Steep_ρu_Y, Steep_ρdud_Y, Steep_E )

            # SECOND-ORDER SAFETY FALLBACK:
            _, _, Pc_candidate = GetPrimitive( ρc[1,j], ρu_Xc[1,j], ρu_Yc[1,j], Ec[1,j], γad[1,j] )
            if Shock[1,j] == Shock_HLL_and_PLM || Steep == 1 || ρc[1,j] ≤ ρ_floor || ρdc[1,j] ≤ ρd_floor || Pc_candidate ≤ P_floor || !isfinite(Pc_candidate)
                ρc[1,j]      = max( ρ[1,j], ρ_floor )
                ρu_Xc[1,j]   = ρu_X[1,j]
                ρu_Yc[1,j]   = ρu_Y[1,j]
                Ec[1,j]      = E[1,j]

                ρdc[1,j]     = max( ρd[1,j], ρd_floor )
                ρdud_Xc[1,j] = ρdud_X[1,j]
                ρdud_Yc[1,j] = ρdud_Y[1,j]
            end
        end
    end
    
    # Interior cells 2 ≤ i ≤ N_X - 1: 
    @threads for j in dN_Y
        @inbounds @simd for i in dN_Xint
            # RECONSTRUCT ⟨ρ⟩ᵢ → ρ(rcᵢ):
            ρc[i,j]      = w_i_m1_X[i] * ρ[i-1,j]  +  w_i_0_X[i] * ρ[i,j]  +  w_i_p1_X[i] * ρ[i+1,j]
            Steep_ρ      = SteepOrNot( ρc[i,j], ρ[i-1,j], ρ[i,j], ρ[i+1,j] )

            ρdc[i,j]     = w_i_m1_X[i] * ρd[i-1,j]  +  w_i_0_X[i] * ρd[i,j]  +  w_i_p1_X[i] * ρd[i+1,j]  # Dust 
            Steep_ρd     = SteepOrNot( ρdc[i,j], ρd[i-1,j], ρd[i,j], ρd[i+1,j] )

            # RECONSTRUCT ⟨ρu_X⟩ᵢ → ρu_X(rcᵢ):
            ρu_Xc[i,j]   = w_i_m1_X[i] * ρu_X[i-1,j]  +  w_i_0_X[i] * ρu_X[i,j]  +  w_i_p1_X[i] * ρu_X[i+1,j]
            Steep_ρu_X   = SteepOrNot( ρu_Xc[i,j], ρu_X[i-1,j], ρu_X[i,j], ρu_X[i+1,j] )

            ρdud_Xc[i,j] = w_i_m1_X[i] * ρdud_X[i-1,j]  +  w_i_0_X[i] * ρdud_X[i,j]  +  w_i_p1_X[i] * ρdud_X[i+1,j]  # Dust 
            Steep_ρdud_X = SteepOrNot( ρdud_Xc[i,j], ρdud_X[i-1,j], ρdud_X[i,j], ρdud_X[i+1,j] )

            # RECONSTRUCT ⟨ρu_Y⟩ᵢ → ρu_Y(rcᵢ):
            ρu_Yc[i,j]   = w_i_m1_X[i] * ρu_Y[i-1,j]  +  w_i_0_X[i] * ρu_Y[i,j]  +  w_i_p1_X[i] * ρu_Y[i+1,j]
            Steep_ρu_Y   = SteepOrNot( ρu_Yc[i,j], ρu_Y[i-1,j], ρu_Y[i,j], ρu_Y[i+1,j] )

            ρdud_Yc[i,j] = w_i_m1_X[i] * ρdud_Y[i-1,j]  +  w_i_0_X[i] * ρdud_Y[i,j]  +  w_i_p1_X[i] * ρdud_Y[i+1,j]
            Steep_ρdud_Y = SteepOrNot( ρdud_Yc[i,j], ρdud_Y[i-1,j], ρdud_Y[i,j], ρdud_Y[i+1,j] )

            # RECONSTRUCT ⟨E⟩ᵢ → E(rcᵢ):
            Ec[i,j]      = w_i_m1_X[i] * E[i-1,j]  +  w_i_0_X[i] * E[i,j]  +  w_i_p1_X[i] * E[i+1,j]
            Steep_E      = SteepOrNot( Ec[i,j], E[i-1,j], E[i,j], E[i+1,j] )

            # GET STEEPNESS INDICATOR:
            Steep = max( Steep_ρ, Steep_ρd, Steep_ρu_X, Steep_ρdud_X, Steep_ρu_Y, Steep_ρdud_Y, Steep_E )

            # SECOND-ORDER SAFETY FALLBACK:
            _, _, Pc_candidate = GetPrimitive( ρc[i,j], ρu_Xc[i,j], ρu_Yc[i,j], Ec[i,j], γad[i,j] )
            if Shock[i,j] == Shock_HLL_and_PLM || Steep == 1 || ρc[i,j] ≤ ρ_floor || ρdc[i,j] ≤ ρd_floor || Pc_candidate ≤ P_floor || !isfinite(Pc_candidate)
                ρc[i,j]      = max( ρ[i,j], ρ_floor )
                ρu_Xc[i,j]   = ρu_X[i,j]
                ρu_Yc[i,j]   = ρu_Y[i,j]
                Ec[i,j]      = E[i,j]

                ρdc[i,j]     = max( ρd[i,j], ρd_floor )
                ρdud_Xc[i,j] = ρdud_X[i,j]
                ρdud_Yc[i,j] = ρdud_Y[i,j]
            end

        end
    end

    # Outermost boundary i = N_X:
    @threads for j in dN_Y
        @inbounds begin
            if BC_X_right == :Outflow
                ρ_Np1      =   ρ[N_X,j]
                ρu_X_Np1   =   max( ρu_X[N_X,j], 0.0 )
                ρu_Y_Np1   =   ρu_Y[N_X,j]
                E_Np1      =   E[N_X,j]

                ρd_Np1     =   ρd[N_X,j]
                ρdud_X_Np1 =   max( ρdud_X[N_X,j], 0.0 )
                ρdud_Y_Np1 =   ρdud_Y[N_X,j]
            elseif BC_X_right == :Reflect
                # If not outflow, we have reflective boundary conditions: 
                ρ_Np1      =   ρ[N_X,j]
                ρu_X_Np1   = - ρu_X[N_X,j]
                ρu_Y_Np1   =   ρu_Y[N_X,j]
                E_Np1      =   E[N_X,j]

                ρd_Np1     =   ρd[N_X,j]
                ρdud_X_Np1 = - ρdud_X[N_X,j]
                ρdud_Y_Np1 =   ρdud_Y[N_X,j]
            elseif BC_X_right == :Periodic 
                ρ_Np1      =   ρ[1,j]
                ρu_X_Np1   =   ρu_X[1,j]
                ρu_Y_Np1   =   ρu_Y[1,j]
                E_Np1      =   E[1,j]

                ρd_Np1     =   ρd[1,j]
                ρdud_X_Np1 =   ρdud_X[1,j]
                ρdud_Y_Np1 =   ρdud_Y[1,j]
            else
                error("No boundary conditions assigned in the X (or r) direction at i = N_X!") 
            end

            # RECONSTRUCT ⟨ρ⟩ᵢ → ρ(rcᵢ) AT i = N_X:
            ρc[N_X,j]      =  w_i_m1_X[N_X] * ρ[N_X-1,j]  +  w_i_0_X[N_X] * ρ[N_X,j]  +  w_i_p1_X[N_X] * ρ_Np1
            Steep_ρ        = SteepOrNot( ρc[N_X,j], ρ[N_X-1,j], ρ[N_X,j], ρ_Np1 )

            ρdc[N_X,j]     =  w_i_m1_X[N_X] * ρd[N_X-1,j]  +  w_i_0_X[N_X] * ρd[N_X,j]  +  w_i_p1_X[N_X] * ρd_Np1  # Dust 
            Steep_ρd       = SteepOrNot( ρdc[N_X,j], ρd[N_X-1,j], ρd[N_X,j], ρd_Np1 )

            # RECONSTRUCT ⟨ρu_X⟩ᵢ → ρu_X(rcᵢ) AT i = N_X:
            ρu_Xc[N_X,j]   = w_i_m1_X[N_X] * ρu_X[N_X-1,j]  +  w_i_0_X[N_X] * ρu_X[N_X,j]  +  w_i_p1_X[N_X] * ρu_X_Np1
            Steep_ρu_X     = SteepOrNot( ρu_Xc[N_X,j], ρu_X[N_X-1,j], ρu_X[N_X,j], ρu_X_Np1 )

            ρdud_Xc[N_X,j] = w_i_m1_X[N_X] * ρdud_X[N_X-1,j]  +  w_i_0_X[N_X] * ρdud_X[N_X,j]  +  w_i_p1_X[N_X] * ρdud_X_Np1  # Dust 
            Steep_ρdud_X   = SteepOrNot( ρdud_Xc[N_X,j], ρdud_X[N_X-1,j], ρdud_X[N_X,j], ρdud_X_Np1 )

            # RECONSTRUCT ⟨ρu_Y⟩ᵢ → ρu_Y(rcᵢ) AT i = N_X:
            ρu_Yc[N_X,j]   = w_i_m1_X[N_X] * ρu_Y[N_X-1,j]  +  w_i_0_X[N_X] * ρu_Y[N_X,j]  +  w_i_p1_X[N_X] * ρu_Y_Np1
            Steep_ρu_Y     = SteepOrNot( ρu_Yc[N_X,j], ρu_Y[N_X-1,j], ρu_Y[N_X,j], ρu_Y_Np1 )

            ρdud_Yc[N_X,j] = w_i_m1_X[N_X] * ρdud_Y[N_X-1,j]  +  w_i_0_X[N_X] * ρdud_Y[N_X,j]  +  w_i_p1_X[N_X] * ρdud_Y_Np1
            Steep_ρdud_Y   = SteepOrNot( ρdud_Yc[N_X,j], ρdud_Y[N_X-1,j], ρdud_Y[N_X,j], ρdud_Y_Np1 )

            # RECONSTRUCT ⟨E⟩ᵢ → E(rcᵢ) AT i = N_X:
            Ec[N_X,j]      = w_i_m1_X[N_X] * E[N_X-1,j]  +  w_i_0_X[N_X] * E[N_X,j]  +  w_i_p1_X[N_X] * E_Np1
            Steep_E        = SteepOrNot( Ec[N_X,j], E[N_X-1,j], E[N_X,j], E_Np1 )

            # GET STEEPNESS INDICATOR:
            Steep = max( Steep_ρ, Steep_ρd, Steep_ρu_X, Steep_ρdud_X, Steep_ρu_Y, Steep_ρdud_Y, Steep_E )

            # SECOND-ORDER SAFETY FALLBACK:
            _, _, Pc_candidate = GetPrimitive( ρc[N_X,j], ρu_Xc[N_X,j], ρu_Yc[N_X,j], Ec[N_X,j], γad[N_X,j] )
            if Shock[N_X,j] == Shock_HLL_and_PLM|| Steep == 1 || ρc[N_X,j] ≤ ρ_floor || ρdc[N_X,j] ≤ ρd_floor || Pc_candidate ≤ P_floor || !isfinite(Pc_candidate)
                ρc[N_X,j]      = max( ρ[N_X,j], ρ_floor )
                ρu_Xc[N_X,j]   = ρu_X[N_X,j]
                ρu_Yc[N_X,j]   = ρu_Y[N_X,j]
                Ec[N_X,j]      = E[N_X,j]

                ρdc[N_X,j]     = max( ρd[N_X,j], ρd_floor )
                ρdud_Xc[N_X,j] = ρdud_X[N_X,j]
                ρdud_Yc[N_X,j] = ρdud_Y[N_X,j]
            end
        end
    end

    return nothing 
end


""" This function recovers the geometric cell center values of 
    conservative variables (e.g. ρc) along the Y-direction,
    from the cell-averaged values (e.g. ρ), using the appropriate 
    weights for third-order accuracy. """
@inline function RecoverPointwiseFromAverages_Y!( ρc::AbstractMatrix{Float64},
                                               ρu_Xc::AbstractMatrix{Float64},
                                               ρu_Yc::AbstractMatrix{Float64},
                                                  Ec::AbstractMatrix{Float64},
                                                 ρdc::AbstractMatrix{Float64},
                                             ρdud_Xc::AbstractMatrix{Float64},
                                             ρdud_Yc::AbstractMatrix{Float64},
                                                  
                                                   ρ::AbstractMatrix{Float64},
                                                ρu_X::AbstractMatrix{Float64},
                                                ρu_Y::AbstractMatrix{Float64},
                                                   E::AbstractMatrix{Float64},
                                                 γad::AbstractMatrix{Float64},
                                                  ρd::AbstractMatrix{Float64},
                                              ρdud_X::AbstractMatrix{Float64},
                                              ρdud_Y::AbstractMatrix{Float64},
                                                   
                                            w_j_m1_Y::AbstractVector{Float64},
                                             w_j_0_Y::AbstractVector{Float64},
                                            w_j_p1_Y::AbstractVector{Float64},

                                               Shock::AbstractMatrix{Int},
                                           
                                           BC_Y_left::Symbol  = :Reflect,
                                          BC_Y_right::Symbol  = :Outflow
                                                )

    N_X, N_Y = size( ρ )
    dN_Yint  = 2:(N_Y - 1)
    dN_X     = 1:N_X

    # Innermost boundary j = 1:
    @threads for i in dN_X
        @inbounds begin
            if BC_Y_left == :Reflect 
                ρ_0      =   ρ[i,1]
                ρu_X_0   =   ρu_X[i,1]
                ρu_Y_0   = - ρu_Y[i,1]
                E_0      =   E[i,1]

                ρd_0     =   ρd[i,1]
                ρdud_X_0 =   ρdud_X[i,1]
                ρdud_Y_0 = - ρdud_Y[i,1]
            elseif BC_Y_left == :Outflow
                # If not reflective, we have outflow conditions: 
                ρ_0      =   ρ[i,1]
                ρu_X_0   =   ρu_X[i,1]
                ρu_Y_0   =   ρu_Y[i,1]
                E_0      =   E[i,1] 

                ρd_0     =   ρd[i,1]
                ρdud_X_0 =   ρdud_X[i,1]
                ρdud_Y_0 =   ρdud_Y[i,1]
            elseif BC_Y_left == :Periodic
                ρ_0      =   ρ[i,N_Y]
                ρu_X_0   =   ρu_X[i,N_Y]
                ρu_Y_0   =   ρu_Y[i,N_Y]
                E_0      =   E[i,N_Y] 

                ρd_0     =   ρd[i,N_Y]
                ρdud_X_0 =   ρdud_X[i,N_Y]
                ρdud_Y_0 =   ρdud_Y[i,N_Y]
            else 
                error("No boundary conditions assigned in the Y (or Z) direction at j = 1!")
            end

            # RECONSTRUCT ⟨ρ⟩ⱼ → ρ(Ycⱼ) AT j = 1:
            ρc[i,1]      = w_j_m1_Y[1] * ρ_0  +  w_j_0_Y[1] * ρ[i,1]  +  w_j_p1_Y[1] * ρ[i,2]
            Steep_ρ      = SteepOrNot( ρc[i,1], ρ_0, ρ[i,1], ρ[i,2] )

            ρdc[i,1]     = w_j_m1_Y[1] * ρd_0  +  w_j_0_Y[1] * ρd[i,1]  +  w_j_p1_Y[1] * ρd[i,2]  # Dust 
            Steep_ρd     = SteepOrNot( ρdc[i,1], ρd_0, ρd[i,1], ρd[i,2] )

            # RECONSTRUCT ⟨ρu_X⟩ⱼ → ρu_X(Ycⱼ) AT j = 1:
            ρu_Xc[i,1]   = w_j_m1_Y[1] * ρu_X_0  +  w_j_0_Y[1] * ρu_X[i,1]  +  w_j_p1_Y[1] * ρu_X[i,2]
            Steep_ρu_X   = SteepOrNot( ρu_Xc[i,1], ρu_X_0, ρu_X[i,1], ρu_X[i,2] )

            ρdud_Xc[i,1] = w_j_m1_Y[1] * ρdud_X_0  +  w_j_0_Y[1] * ρdud_X[i,1]  +  w_j_p1_Y[1] * ρdud_X[i,2]  # Dust 
            Steep_ρdud_X = SteepOrNot( ρdud_Xc[i,1], ρdud_X_0, ρdud_X[i,1], ρdud_X[i,2] )

            # RECONSTRUCT ⟨ρu_Y⟩ⱼ → ρu_Y(Ycⱼ) AT j = 1:
            ρu_Yc[i,1]   = w_j_m1_Y[1] * ρu_Y_0  +  w_j_0_Y[1] * ρu_Y[i,1]  +  w_j_p1_Y[1] * ρu_Y[i,2]
            Steep_ρu_Y   = SteepOrNot( ρu_Yc[i,1], ρu_Y_0, ρu_Y[i,1], ρu_Y[i,2] )

            ρdud_Yc[i,1] = w_j_m1_Y[1] * ρdud_Y_0  +  w_j_0_Y[1] * ρdud_Y[i,1]  +  w_j_p1_Y[1] * ρdud_Y[i,2]  # Dust 
            Steep_ρdud_Y = SteepOrNot( ρdud_Yc[i,1], ρdud_Y_0, ρdud_Y[i,1], ρdud_Y[i,2] )

            # RECONSTRUCT ⟨E⟩ⱼ → E(Ycⱼ) AT j = 1:
            Ec[i,1]    = w_j_m1_Y[1] * E_0  +  w_j_0_Y[1] * E[i,1]  +  w_j_p1_Y[1] * E[i,2]
            Steep_E    = SteepOrNot( Ec[i,1], E_0, E[i,1], E[i,2] )

            # GET STEEPNESS INDICATOR:
            Steep = max( Steep_ρ, Steep_ρd, Steep_ρu_X, Steep_ρdud_X, Steep_ρu_Y, Steep_ρdud_Y, Steep_E )

            # SECOND-ORDER SAFETY FALLBACK:
            _, _, Pc_candidate = GetPrimitive( ρc[i,1], ρu_Xc[i,1], ρu_Yc[i,1], Ec[i,1], γad[i,1] )
            if Shock[i,1] == Shock_HLL_and_PLM || Steep == 1 || ρc[i,1] ≤ ρ_floor || ρdc[i,1] ≤ ρd_floor || Pc_candidate ≤ P_floor || !isfinite(Pc_candidate)
                ρc[i,1]      = max( ρ[i,1], ρ_floor )
                ρu_Xc[i,1]   = ρu_X[i,1]
                ρu_Yc[i,1]   = ρu_Y[i,1]
                Ec[i,1]      = E[i,1]

                ρdc[i,1]     = max( ρd[i,1], ρd_floor )
                ρdud_Xc[i,1] = ρdud_X[i,1]
                ρdud_Yc[i,1] = ρdud_Y[i,1]
            end
        end
    end
    
    # Interior cells 2 ≤ j ≤ N_Y - 1: 
    @threads for j in dN_Yint
        @inbounds @simd for i in dN_X
            # RECONSTRUCT ⟨ρ⟩ⱼ → ρ(Ycⱼ):
            ρc[i,j]      = w_j_m1_Y[j] * ρ[i,j-1]  +  w_j_0_Y[j] * ρ[i,j]  +  w_j_p1_Y[j] * ρ[i,j+1]
            Steep_ρ      = SteepOrNot( ρc[i,j], ρ[i,j-1], ρ[i,j], ρ[i,j+1] )

            ρdc[i,j]     = w_j_m1_Y[j] * ρd[i,j-1]  +  w_j_0_Y[j] * ρd[i,j]  +  w_j_p1_Y[j] * ρd[i,j+1]  # Dust 
            Steep_ρd     = SteepOrNot( ρdc[i,j], ρd[i,j-1], ρd[i,j], ρd[i,j+1] )

            # RECONSTRUCT ⟨ρu_X⟩ⱼ → ρu_X(Ycⱼ):
            ρu_Xc[i,j]   = w_j_m1_Y[j] * ρu_X[i,j-1]  +  w_j_0_Y[j] * ρu_X[i,j]  +  w_j_p1_Y[j] * ρu_X[i,j+1]
            Steep_ρu_X   = SteepOrNot( ρu_Xc[i,j], ρu_X[i,j-1], ρu_X[i,j], ρu_X[i,j+1] )

            ρdud_Xc[i,j] = w_j_m1_Y[j] * ρdud_X[i,j-1]  +  w_j_0_Y[j] * ρdud_X[i,j]  +  w_j_p1_Y[j] * ρdud_X[i,j+1]  # Dust 
            Steep_ρdud_X = SteepOrNot( ρdud_Xc[i,j], ρdud_X[i,j-1], ρdud_X[i,j], ρdud_X[i,j+1] )

            # RECONSTRUCT ⟨ρu_Y⟩ⱼ → ρu_Y(Ycⱼ):
            ρu_Yc[i,j]   = w_j_m1_Y[j] * ρu_Y[i,j-1]  +  w_j_0_Y[j] * ρu_Y[i,j]  +  w_j_p1_Y[j] * ρu_Y[i,j+1]
            Steep_ρu_Y   = SteepOrNot( ρu_Yc[i,j], ρu_Y[i,j-1], ρu_Y[i,j], ρu_Y[i,j+1] )

            ρdud_Yc[i,j] = w_j_m1_Y[j] * ρdud_Y[i,j-1]  +  w_j_0_Y[j] * ρdud_Y[i,j]  +  w_j_p1_Y[j] * ρdud_Y[i,j+1]  # Dust 
            Steep_ρdud_Y = SteepOrNot( ρdud_Yc[i,j], ρdud_Y[i,j-1], ρdud_Y[i,j], ρdud_Y[i,j+1] )

            # RECONSTRUCT ⟨E⟩ⱼ → E(Ycⱼ):
            Ec[i,j]    = w_j_m1_Y[j] * E[i,j-1]  +  w_j_0_Y[j] * E[i,j]  +  w_j_p1_Y[j] * E[i,j+1]
            Steep_E    = SteepOrNot( Ec[i,j], E[i,j-1], E[i,j], E[i,j+1] )

            # GET STEEPNESS INDICATOR:
            Steep = max( Steep_ρ, Steep_ρu_X, Steep_ρu_Y, Steep_E )

            # SECOND-ORDER SAFETY FALLBACK:
            _, _, Pc_candidate = GetPrimitive( ρc[i,j], ρu_Xc[i,j], ρu_Yc[i,j], Ec[i,j], γad[i,j] )
            if Shock[i,j] == Shock_HLL_and_PLM || Steep == 1 || ρc[i,j] ≤ ρ_floor || ρdc[i,j] ≤ ρd_floor || Pc_candidate ≤ P_floor || !isfinite(Pc_candidate)
                ρc[i,j]      = max( ρ[i,j], ρ_floor )
                ρu_Xc[i,j]   = ρu_X[i,j]
                ρu_Yc[i,j]   = ρu_Y[i,j]
                Ec[i,j]      = E[i,j]

                ρdc[i,j]     = max( ρd[i,j], ρd_floor )
                ρdud_Xc[i,j] = ρdud_X[i,j]
                ρdud_Yc[i,j] = ρdud_Y[i,j]
            end
        end
    end

    # Outermost boundary j = N_Y:
    @threads for i in dN_X
        @inbounds begin
            if BC_Y_right == :Outflow
                ρ_Np1      =   ρ[i,N_Y]
                ρu_X_Np1   =   ρu_X[i,N_Y] 
                ρu_Y_Np1   =   max( ρu_Y[i,N_Y], 0.0 )
                E_Np1      =   E[i,N_Y]

                ρd_Np1     =   ρd[i,N_Y]
                ρdud_X_Np1 =   ρdud_X[i,N_Y] 
                ρdud_Y_Np1 =   max( ρdud_Y[i,N_Y], 0.0 )
            elseif BC_Y_right == :Reflect 
                # If not outflow, we have reflective boundary conditions: 
                ρ_Np1      =   ρ[i,N_Y]
                ρu_X_Np1   =   ρu_X[i,N_Y]
                ρu_Y_Np1   = - ρu_Y[i,N_Y]
                E_Np1      =   E[i,N_Y]

                ρd_Np1     =   ρd[i,N_Y]
                ρdud_X_Np1 =   ρdud_X[i,N_Y]
                ρdud_Y_Np1 = - ρdud_Y[i,N_Y]
            elseif BC_Y_right == :Periodic 
                ρ_Np1      =   ρ[i,1]
                ρu_X_Np1   =   ρu_X[i,1]
                ρu_Y_Np1   =   ρu_Y[i,1]
                E_Np1      =   E[i,1]

                ρd_Np1     =   ρd[i,1]
                ρdud_X_Np1 =   ρdud_X[i,1]
                ρdud_Y_Np1 =   ρdud_Y[i,1]
            else 
                error("No boundary conditions assigned in the Y (or Z) direction at j = N_Y!")
            end

            # RECONSTRUCT ⟨ρ⟩ⱼ → ρ(Ycⱼ) AT j = N_Y:
            ρc[i,N_Y]      = w_j_m1_Y[N_Y] * ρ[i,N_Y-1]  +  w_j_0_Y[N_Y] * ρ[i,N_Y]  +  w_j_p1_Y[N_Y] * ρ_Np1
            Steep_ρ        = SteepOrNot( ρc[i,N_Y], ρ[i,N_Y-1], ρ[i,N_Y], ρ_Np1 )

            ρdc[i,N_Y]     = w_j_m1_Y[N_Y] * ρd[i,N_Y-1]  +  w_j_0_Y[N_Y] * ρd[i,N_Y]  +  w_j_p1_Y[N_Y] * ρd_Np1  # Dust 
            Steep_ρd       = SteepOrNot( ρdc[i,N_Y], ρd[i,N_Y-1], ρd[i,N_Y], ρd_Np1 )

            # RECONSTRUCT ⟨ρu_X⟩ⱼ → ρu_X(Ycⱼ) AT j = N_Y:
            ρu_Xc[i,N_Y]   = w_j_m1_Y[N_Y] * ρu_X[i,N_Y-1]  +  w_j_0_Y[N_Y] * ρu_X[i,N_Y]  +  w_j_p1_Y[N_Y] * ρu_X_Np1
            Steep_ρu_X     = SteepOrNot( ρu_Xc[i,N_Y], ρu_X[i,N_Y-1], ρu_X[i,N_Y], ρu_X_Np1 )

            ρdud_Xc[i,N_Y] = w_j_m1_Y[N_Y] * ρdud_X[i,N_Y-1]  +  w_j_0_Y[N_Y] * ρdud_X[i,N_Y]  +  w_j_p1_Y[N_Y] * ρdud_X_Np1  # Dust 
            Steep_ρdud_X   = SteepOrNot( ρdud_Xc[i,N_Y], ρdud_X[i,N_Y-1], ρdud_X[i,N_Y], ρdud_X_Np1 )

            # RECONSTRUCT ⟨ρu_Y⟩ⱼ → ρu_Y(Ycⱼ) AT j = N_Y:
            ρu_Yc[i,N_Y]   = w_j_m1_Y[N_Y] * ρu_Y[i,N_Y-1]  +  w_j_0_Y[N_Y] * ρu_Y[i,N_Y]  +  w_j_p1_Y[N_Y] * ρu_Y_Np1
            Steep_ρu_Y     = SteepOrNot( ρu_Yc[i,N_Y], ρu_Y[i,N_Y-1], ρu_Y[i,N_Y], ρu_Y_Np1 )

            ρdud_Yc[i,N_Y] = w_j_m1_Y[N_Y] * ρdud_Y[i,N_Y-1]  +  w_j_0_Y[N_Y] * ρdud_Y[i,N_Y]  +  w_j_p1_Y[N_Y] * ρdud_Y_Np1  # Dust 
            Steep_ρdud_Y   = SteepOrNot( ρdud_Yc[i,N_Y], ρdud_Y[i,N_Y-1], ρdud_Y[i,N_Y], ρdud_Y_Np1 )

            # RECONSTRUCT ⟨E⟩ⱼ → E(Ycⱼ) AT j = N_Y:
            Ec[i,N_Y]      = w_j_m1_Y[N_Y] * E[i,N_Y-1]  +  w_j_0_Y[N_Y] * E[i,N_Y]  +  w_j_p1_Y[N_Y] * E_Np1
            Steep_E        = SteepOrNot( Ec[i,N_Y], E[i,N_Y-1], E[i,N_Y], E_Np1 )

            # GET STEEPNESS INDICATOR:
            Steep = max( Steep_ρ, Steep_ρd, Steep_ρu_X, Steep_ρdud_X, Steep_ρu_Y, Steep_ρdud_Y, Steep_E )

            # SECOND-ORDER SAFETY FALLBACK:
            _, _, Pc_candidate = GetPrimitive( ρc[i,N_Y], ρu_Xc[i,N_Y], ρu_Yc[i,N_Y], Ec[i,N_Y], γad[i,N_Y] )
            if Shock[i,N_Y] == Shock_HLL_and_PLM || Steep == 1 || ρc[i,N_Y] ≤ ρ_floor || ρdc[i,N_Y] ≤ ρd_floor || Pc_candidate ≤ P_floor || !isfinite(Pc_candidate)
                ρc[i,N_Y]      = max( ρ[i,N_Y], ρ_floor )
                ρu_Xc[i,N_Y]   = ρu_X[i,N_Y]
                ρu_Yc[i,N_Y]   = ρu_Y[i,N_Y]
                Ec[i,N_Y]      = E[i,N_Y]

                ρdc[i,N_Y]     = max( ρd[i,N_Y], ρd_floor )
                ρdud_Xc[i,N_Y] = ρdud_X[i,N_Y]
                ρdud_Yc[i,N_Y] = ρdud_Y[i,N_Y]
            end
        end
    end

    return nothing 
end


""" This function computes/updates the primitive variables at the 
    geometric cell centers, using the conservative variables at the 
    geometric cell centers. """
@inline function RecoverPointwisePrimitivesFromConservatives!( ρc::AbstractMatrix{Float64},
                                                            ρu_Xc::AbstractMatrix{Float64},
                                                            ρu_Yc::AbstractMatrix{Float64},
                                                               Ec::AbstractMatrix{Float64},
                                                              ρdc::AbstractMatrix{Float64},
                                                          ρdud_Xc::AbstractMatrix{Float64},
                                                          ρdud_Yc::AbstractMatrix{Float64},
                                                               
                                                             u_Xc::AbstractMatrix{Float64},
                                                             u_Yc::AbstractMatrix{Float64},
                                                               Pc::AbstractMatrix{Float64},
                                                              γad::AbstractMatrix{Float64},
                                                            ud_Xc::AbstractMatrix{Float64},
                                                            ud_Yc::AbstractMatrix{Float64}, )

    N_X, N_Y = size( ρc )
    dN_X     = 1:N_X 
    dN_Y     = 1:N_Y 

    @threads for j in dN_Y 
        @inbounds @simd for i in dN_X 
            # Recover primitives at geometric cell centers, from the 
            # conservative variables at the cell centers:
            u_Xc[i,j], u_Yc[i,j], Pc[i,j] = GetPrimitive( ρc[i,j], ρu_Xc[i,j], ρu_Yc[i,j], Ec[i,j], γad[i,j] )

            ud_Xc[i,j], ud_Yc[i,j] = GetDustVelocities( ρdc[i,j], ρdud_Xc[i,j], ρdud_Yc[i,j] )

            # Ensure positive pressure:
            Pc[i,j] = max( Pc[i,j], P_floor )
        end
    end

    return nothing 
end


""" This function recovers cell-averaraged primitives (u_X, u_Y, P), from 
    the cell-centered ones (u_Xc, u_Yc, Pc), along the X-direction. """
@inline function RecoverCellAveragedPrimitives_X!( u_Xc::AbstractMatrix{Float64},
                                                   u_Yc::AbstractMatrix{Float64},
                                                     Pc::AbstractMatrix{Float64},
                                                  ud_Xc::AbstractMatrix{Float64},
                                                  ud_Yc::AbstractMatrix{Float64},

                                                    u_X::AbstractMatrix{Float64},
                                                    u_Y::AbstractMatrix{Float64},
                                                      P::AbstractMatrix{Float64},
                                                   ud_X::AbstractMatrix{Float64},
                                                   ud_Y::AbstractMatrix{Float64},
                                                   
                                               q_i_m1_X::AbstractVector{Float64},
                                                q_i_0_X::AbstractVector{Float64},
                                               q_i_p1_X::AbstractVector{Float64},

                                                  Shock::AbstractMatrix{Int},
                                           
                                              BC_X_left::Symbol  = :Reflect,
                                             BC_X_right::Symbol  = :Outflow
                                                  )

    N_X, N_Y = size( P )
    dN_Xint  = 2:(N_X - 1)
    dN_Y     = 1:N_Y

    # Innermost cell i = 1:
    @threads for j in dN_Y
        @inbounds begin

            if BC_X_left == :Reflect
                u_Xc_0  = - u_Xc[1,j]
                u_Yc_0  =   u_Yc[1,j] 
                Pc_0    =   Pc[1,j]

                ud_Xc_0 = - ud_Xc[1,j]
                ud_Yc_0 =   ud_Yc[1,j] 
            elseif BC_X_left == :Outflow
                u_Xc_0  =   u_Xc[1,j]
                u_Yc_0  =   u_Yc[1,j] 
                Pc_0    =   Pc[1,j]

                ud_Xc_0 =   ud_Xc[1,j]
                ud_Yc_0 =   ud_Yc[1,j] 
            elseif BC_X_left == :Periodic 
                u_Xc_0  =   u_Xc[N_X,j]
                u_Yc_0  =   u_Yc[N_X,j]
                Pc_0    =   Pc[N_X,j]

                ud_Xc_0 =   ud_Xc[N_X,j]
                ud_Yc_0 =   ud_Yc[N_X,j]
            else 
                error("No boundary conditions assigned in the X (or r) direction at i = 1!")
            end
            
            u_X[1,j]    = q_i_m1_X[1] * u_Xc_0  +  q_i_0_X[1] * u_Xc[1,j]  +  q_i_p1_X[1] * u_Xc[2,j]
            Steep_u_X   = SteepOrNot( u_X[1,j], u_Xc_0, u_Xc[1,j], u_Xc[2,j] )

            u_Y[1,j]    = q_i_m1_X[1] * u_Yc_0  +  q_i_0_X[1] * u_Yc[1,j]  +  q_i_p1_X[1] * u_Yc[2,j]
            Steep_u_Y   = SteepOrNot( u_Y[1,j], u_Yc_0, u_Yc[1,j], u_Yc[2,j] )

            P[1,j]      = q_i_m1_X[1] * Pc_0  +  q_i_0_X[1] * Pc[1,j]  +  q_i_p1_X[1] * Pc[2,j]
            Steep_P     = SteepOrNot( P[1,j], Pc_0, Pc[1,j], Pc[2,j] )

            ud_X[1,j]   = q_i_m1_X[1] * ud_Xc_0  +  q_i_0_X[1] * ud_Xc[1,j]  +  q_i_p1_X[1] * ud_Xc[2,j]  # Dust 
            Steep_ud_X  = SteepOrNot( ud_X[1,j], ud_Xc_0, ud_Xc[1,j], ud_Xc[2,j] )

            ud_Y[1,j]   = q_i_m1_X[1] * ud_Yc_0  +  q_i_0_X[1] * ud_Yc[1,j]  +  q_i_p1_X[1] * ud_Yc[2,j]
            Steep_ud_Y  = SteepOrNot( ud_Y[1,j], ud_Yc_0, ud_Yc[1,j], ud_Yc[2,j] )

            # GET STEEPNESS INDICATOR:
            Steep = max( Steep_u_X, Steep_ud_X, Steep_u_Y, Steep_ud_Y, Steep_P )

            # SECOND-ORDER SAFETY FALLBACK:
            if Shock[1,j] == Shock_HLL_and_PLM || Steep == 1 || P[1,j] ≤ P_floor || !isfinite(P[1,j])
                u_X[1,j]  = u_Xc[1,j] 
                u_Y[1,j]  = u_Yc[1,j]
                P[1,j]    = Pc[1,j] 

                ud_X[1,j] = ud_Xc[1,j] 
                ud_Y[1,j] = ud_Yc[1,j]
            end    
        end
    end

    # Interior cells 2 ≤ i ≤ N_X - 1:
    @threads for j in dN_Y
        @inbounds @simd for i in dN_Xint

            u_X[i,j]    = q_i_m1_X[i] * u_Xc[i-1,j]  +  q_i_0_X[i] * u_Xc[i,j]  +  q_i_p1_X[i] * u_Xc[i+1,j]
            Steep_u_X   = SteepOrNot( u_X[i,j], u_Xc[i-1,j], u_Xc[i,j], u_Xc[i+1,j] )

            u_Y[i,j]    = q_i_m1_X[i] * u_Yc[i-1,j]  +  q_i_0_X[i] * u_Yc[i,j]  +  q_i_p1_X[i] * u_Yc[i+1,j]
            Steep_u_Y   = SteepOrNot( u_Y[i,j], u_Yc[i-1,j], u_Yc[i,j], u_Yc[i+1,j] )

            P[i,j]      = q_i_m1_X[i] * Pc[i-1,j]  +  q_i_0_X[i] * Pc[i,j]  +  q_i_p1_X[i] * Pc[i+1,j]
            Steep_P     = SteepOrNot( P[i,j], Pc[i-1,j], Pc[i,j], Pc[i+1,j] )

            ud_X[i,j]   = q_i_m1_X[i] * ud_Xc[i-1,j]  +  q_i_0_X[i] * ud_Xc[i,j]  +  q_i_p1_X[i] * ud_Xc[i+1,j]  # Dust 
            Steep_ud_X  = SteepOrNot( ud_X[i,j], ud_Xc[i-1,j], ud_Xc[i,j], ud_Xc[i+1,j] )

            ud_Y[i,j]   = q_i_m1_X[i] * ud_Yc[i-1,j]  +  q_i_0_X[i] * ud_Yc[i,j]  +  q_i_p1_X[i] * ud_Yc[i+1,j]
            Steep_ud_Y  = SteepOrNot( ud_Y[i,j], ud_Yc[i-1,j], ud_Yc[i,j], ud_Yc[i+1,j] )

            # GET STEEPNESS INDICATOR:
            Steep = max( Steep_u_X, Steep_ud_X, Steep_u_Y, Steep_ud_Y, Steep_P )

            # SECOND-ORDER SAFETY FALLBACK:
            if Shock[i,j] == Shock_HLL_and_PLM || Steep == 1 || P[i,j] ≤ P_floor || !isfinite(P[i,j])
                u_X[i,j]  = u_Xc[i,j] 
                u_Y[i,j]  = u_Yc[i,j]
                P[i,j]    = Pc[i,j] 

                ud_X[i,j] = ud_Xc[i,j] 
                ud_Y[i,j] = ud_Yc[i,j]
            end    
        end
    end

    # Outermost cell i = N_X:
    @threads for j in dN_Y
        @inbounds begin

            if BC_X_right == :Reflect
                u_Xc_Np1  = - u_Xc[N_X,j] 
                u_Yc_Np1  =   u_Yc[N_X,j]
                Pc_Np1    =   Pc[N_X,j]

                ud_Xc_Np1 = - ud_Xc[N_X,j] 
                ud_Yc_Np1 =   ud_Yc[N_X,j]
            elseif BC_X_right == :Outflow
                u_Xc_Np1  =   max( u_Xc[N_X,j], 0.0 )
                u_Yc_Np1  =   u_Yc[N_X,j]
                Pc_Np1    =   Pc[N_X,j]

                ud_Xc_Np1 =   max( ud_Xc[N_X,j], 0.0 )
                ud_Yc_Np1 =   ud_Yc[N_X,j]
            elseif BC_X_right == :Periodic 
                u_Xc_Np1  =   u_Xc[1,j]
                u_Yc_Np1  =   u_Yc[1,j]
                Pc_Np1    =   Pc[1,j]

                ud_Xc_Np1 =   ud_Xc[1,j]
                ud_Yc_Np1 =   ud_Yc[1,j]
            else 
                error("No boundary conditions assigned in the X (or r) direction at i = N_X!")
            end
            
            u_X[N_X,j]   = q_i_m1_X[N_X] * u_Xc[N_X-1,j]  +  q_i_0_X[N_X] * u_Xc[N_X,j]  +  q_i_p1_X[N_X] * u_Xc_Np1
            Steep_u_X    = SteepOrNot( u_X[N_X,j], u_Xc[N_X-1,j], u_Xc[N_X,j], u_Xc_Np1 )

            u_Y[N_X,j]   = q_i_m1_X[N_X] * u_Yc[N_X-1,j]  +  q_i_0_X[N_X] * u_Yc[N_X,j]  +  q_i_p1_X[N_X] * u_Yc_Np1
            Steep_u_Y    = SteepOrNot( u_Y[N_X,j], u_Yc[N_X-1,j], u_Yc[N_X,j], u_Yc_Np1 )

            P[N_X,j]     = q_i_m1_X[N_X] * Pc[N_X-1,j]  +  q_i_0_X[N_X] * Pc[N_X,j]  +  q_i_p1_X[N_X] * Pc_Np1
            Steep_P      = SteepOrNot( P[N_X,j], Pc[N_X-1,j], Pc[N_X,j], Pc_Np1 )

            ud_X[N_X,j]  = q_i_m1_X[N_X] * ud_Xc[N_X-1,j]  +  q_i_0_X[N_X] * ud_Xc[N_X,j]  +  q_i_p1_X[N_X] * ud_Xc_Np1  # Dust 
            Steep_ud_X   = SteepOrNot( ud_X[N_X,j], ud_Xc[N_X-1,j], ud_Xc[N_X,j], ud_Xc_Np1 )

            ud_Y[N_X,j]  = q_i_m1_X[N_X] * ud_Yc[N_X-1,j]  +  q_i_0_X[N_X] * ud_Yc[N_X,j]  +  q_i_p1_X[N_X] * ud_Yc_Np1
            Steep_ud_Y   = SteepOrNot( ud_Y[N_X,j], ud_Yc[N_X-1,j], ud_Yc[N_X,j], ud_Yc_Np1 )

            # GET STEEPNESS INDICATOR:
            Steep = max( Steep_u_X, Steep_ud_X, Steep_u_Y, Steep_ud_Y, Steep_P )

            # SECOND-ORDER SAFETY FALLBACK:
            if Shock[N_X,j] == Shock_HLL_and_PLM || Steep == 1 || P[N_X,j] ≤ P_floor || !isfinite(P[N_X,j])
                u_X[N_X,j]  = u_Xc[N_X,j] 
                u_Y[N_X,j]  = u_Yc[N_X,j]
                P[N_X,j]    = Pc[N_X,j] 

                ud_X[N_X,j] = ud_Xc[N_X,j] 
                ud_Y[N_X,j] = ud_Yc[N_X,j]
            end    
        end
    end

    return nothing 
end


""" This function recovers cell-averaraged primitives (u_X, u_Y, P), from 
    the cell-centered ones (u_Xc, u_Yc, Pc), along the Y-direction. """
@inline function RecoverCellAveragedPrimitives_Y!( u_Xc::AbstractMatrix{Float64},
                                                   u_Yc::AbstractMatrix{Float64},
                                                     Pc::AbstractMatrix{Float64},
                                                  ud_Xc::AbstractMatrix{Float64},
                                                  ud_Yc::AbstractMatrix{Float64},

                                                    u_X::AbstractMatrix{Float64},
                                                    u_Y::AbstractMatrix{Float64},
                                                      P::AbstractMatrix{Float64},
                                                   ud_X::AbstractMatrix{Float64},
                                                   ud_Y::AbstractMatrix{Float64},
                                                   
                                               q_j_m1_Y::AbstractVector{Float64},
                                                q_j_0_Y::AbstractVector{Float64},
                                               q_j_p1_Y::AbstractVector{Float64},

                                                  Shock::AbstractMatrix{Int},
                                           
                                              BC_Y_left::Symbol  = :Reflect,
                                             BC_Y_right::Symbol  = :Outflow
                                                  )

    N_X, N_Y = size( P )
    dN_X     = 1:N_X 
    dN_Yint  = 2:(N_Y - 1)

    # Innermost cell j = 1:
    @threads for i in dN_X
        @inbounds begin

            if BC_Y_left == :Reflect
                u_Xc_0  =   u_Xc[i,1]
                u_Yc_0  = - u_Yc[i,1] 
                Pc_0    =   Pc[i,1]

                ud_Xc_0 =   ud_Xc[i,1]
                ud_Yc_0 = - ud_Yc[i,1] 
            elseif BC_Y_left == :Outflow
                u_Xc_0  =   u_Xc[i,1]
                u_Yc_0  =   min( u_Yc[i,1], 0.0 ) 
                Pc_0    =   Pc[i,1]

                ud_Xc_0 =   ud_Xc[i,1]
                ud_Yc_0 =   min( ud_Yc[i,1], 0.0 ) 
            elseif BC_Y_left == :Periodic 
                u_Xc_0  =   u_Xc[i,N_Y]
                u_Yc_0  =   u_Yc[i,N_Y] 
                Pc_0    =   Pc[i,N_Y]

                ud_Xc_0 =   ud_Xc[i,N_Y]
                ud_Yc_0 =   ud_Yc[i,N_Y] 
            else 
                error("No boundary conditions assigned in the Y (or Z) direction at j = 1!")
            end
            
            u_X[i,1]    = q_j_m1_Y[1] * u_Xc_0  +  q_j_0_Y[1] * u_Xc[i,1]  +  q_j_p1_Y[1] * u_Xc[i,2]
            Steep_u_X   = SteepOrNot( u_X[i,1], u_Xc_0, u_Xc[i,1], u_Xc[i,2] )

            u_Y[i,1]    = q_j_m1_Y[1] * u_Yc_0  +  q_j_0_Y[1] * u_Yc[i,1]  +  q_j_p1_Y[1] * u_Yc[i,2]
            Steep_u_Y   = SteepOrNot( u_Y[i,1], u_Yc_0, u_Yc[i,1], u_Yc[i,2] )

            P[i,1]      = q_j_m1_Y[1] * Pc_0  +  q_j_0_Y[1] * Pc[i,1]  +  q_j_p1_Y[1] * Pc[i,2]
            Steep_P     = SteepOrNot( P[i,1], Pc_0, Pc[i,1], Pc[i,2] )

            ud_X[i,1]   = q_j_m1_Y[1] * ud_Xc_0  +  q_j_0_Y[1] * ud_Xc[i,1]  +  q_j_p1_Y[1] * ud_Xc[i,2]  # Dust 
            Steep_ud_X  = SteepOrNot( ud_X[i,1], ud_Xc_0, ud_Xc[i,1], ud_Xc[i,2] )

            ud_Y[i,1]   = q_j_m1_Y[1] * ud_Yc_0  +  q_j_0_Y[1] * ud_Yc[i,1]  +  q_j_p1_Y[1] * ud_Yc[i,2]
            Steep_ud_Y  = SteepOrNot( ud_Y[i,1], ud_Yc_0, ud_Yc[i,1], ud_Yc[i,2] )

            # GET STEEPNESS INDICATOR:
            Steep = max( Steep_u_X, Steep_ud_X, Steep_u_Y, Steep_ud_Y, Steep_P )

            # SECOND-ORDER SAFETY FALLBACK:
            if Shock[i,1] == Shock_HLL_and_PLM || Steep == 1 || P[i,1] ≤ P_floor || !isfinite(P[i,1])
                u_X[i,1]  = u_Xc[i,1] 
                u_Y[i,1]  = u_Yc[i,1]
                P[i,1]    = Pc[i,1] 

                ud_X[i,1] = ud_Xc[i,1] 
                ud_Y[i,1] = ud_Yc[i,1]
            end    
        end
    end

    # Interior cells 2 ≤ j ≤ N_Y - 1:
    @threads for j in dN_Yint
        @inbounds @simd for i in dN_X

            u_X[i,j]    = q_j_m1_Y[j] * u_Xc[i,j-1]  +  q_j_0_Y[j] * u_Xc[i,j]  +  q_j_p1_Y[j] * u_Xc[i,j+1]
            Steep_u_X   = SteepOrNot( u_X[i,j], u_Xc[i,j-1], u_Xc[i,j], u_Xc[i,j+1] )

            u_Y[i,j]    = q_j_m1_Y[j] * u_Yc[i,j-1]  +  q_j_0_Y[j] * u_Yc[i,j]  +  q_j_p1_Y[j] * u_Yc[i,j+1]
            Steep_u_Y   = SteepOrNot( u_Y[i,j], u_Yc[i,j-1], u_Yc[i,j], u_Yc[i,j+1] )

            P[i,j]      = q_j_m1_Y[j] * Pc[i,j-1]  +  q_j_0_Y[j] * Pc[i,j]  +  q_j_p1_Y[j] * Pc[i,j+1]
            Steep_P     = SteepOrNot( P[i,j], Pc[i,j-1], Pc[i,j], Pc[i,j+1] )

            ud_X[i,j]   = q_j_m1_Y[j] * ud_Xc[i,j-1]  +  q_j_0_Y[j] * ud_Xc[i,j]  +  q_j_p1_Y[j] * ud_Xc[i,j+1]  # Dust 
            Steep_ud_X  = SteepOrNot( ud_X[i,j], ud_Xc[i,j-1], ud_Xc[i,j], ud_Xc[i,j+1] )

            ud_Y[i,j]   = q_j_m1_Y[j] * ud_Yc[i,j-1]  +  q_j_0_Y[j] * ud_Yc[i,j]  +  q_j_p1_Y[j] * ud_Yc[i,j+1]
            Steep_ud_Y  = SteepOrNot( ud_Y[i,j], ud_Yc[i,j-1], ud_Yc[i,j], ud_Yc[i,j+1] )

            # GET STEEPNESS INDICATOR:
            Steep = max( Steep_u_X, Steep_ud_X, Steep_u_Y, Steep_ud_Y, Steep_P )

            # SECOND-ORDER SAFETY FALLBACK:
            if Shock[i,j] == Shock_HLL_and_PLM || Steep == 1 || P[i,j] ≤ P_floor || !isfinite(P[i,j])
                u_X[i,j]  = u_Xc[i,j] 
                u_Y[i,j]  = u_Yc[i,j]
                P[i,j]    = Pc[i,j] 

                ud_X[i,j] = ud_Xc[i,j] 
                ud_Y[i,j] = ud_Yc[i,j]
            end    
        end
    end

    # Outermost cell j = N_Y:
    @threads for i in dN_X
        @inbounds begin

            if BC_Y_right == :Reflect
                u_Xc_Np1  =   u_Xc[i,N_Y] 
                u_Yc_Np1  = - u_Yc[i,N_Y]
                Pc_Np1    =   Pc[i,N_Y]

                ud_Xc_Np1 =   ud_Xc[i,N_Y] 
                ud_Yc_Np1 = - ud_Yc[i,N_Y]
            elseif BC_Y_right == :Outflow
                u_Xc_Np1  =   u_Xc[i,N_Y]
                u_Yc_Np1  =   max( u_Yc[i,N_Y], 0.0 )
                Pc_Np1    =   Pc[i,N_Y]

                ud_Xc_Np1 =   ud_Xc[i,N_Y]
                ud_Yc_Np1 =   max( ud_Yc[i,N_Y], 0.0 )
            elseif BC_Y_right == :Periodic 
                u_Xc_Np1  =   u_Xc[i,1] 
                u_Yc_Np1  =   u_Yc[i,1]
                Pc_Np1    =   Pc[i,1]

                ud_Xc_Np1 =   ud_Xc[i,1] 
                ud_Yc_Np1 =   ud_Yc[i,1]
            else 
                error("No boundary conditions assigned in the Y (or Z) direction at j = N_Y!")
            end
            
            u_X[i,N_Y]  = q_j_m1_Y[N_Y] * u_Xc[i,N_Y-1]  +  q_j_0_Y[N_Y] * u_Xc[i,N_Y]  +  q_j_p1_Y[N_Y] * u_Xc_Np1
            Steep_u_X   = SteepOrNot( u_X[i,N_Y], u_Xc[i,N_Y-1], u_Xc[i,N_Y], u_Xc_Np1 )

            u_Y[i,N_Y]  = q_j_m1_Y[N_Y] * u_Yc[i,N_Y-1]  +  q_j_0_Y[N_Y] * u_Yc[i,N_Y]  +  q_j_p1_Y[N_Y] * u_Yc_Np1
            Steep_u_Y   = SteepOrNot( u_Y[i,N_Y], u_Yc[i,N_Y-1], u_Yc[i,N_Y], u_Yc_Np1 )

            P[i,N_Y]    = q_j_m1_Y[N_Y] * Pc[i,N_Y-1]  +  q_j_0_Y[N_Y] * Pc[i,N_Y]  +  q_j_p1_Y[N_Y] * Pc_Np1
            Steep_P     = SteepOrNot( P[i,N_Y], Pc[i,N_Y-1], Pc[i,N_Y], Pc_Np1 )

            ud_X[i,N_Y] = q_j_m1_Y[N_Y] * ud_Xc[i,N_Y-1]  +  q_j_0_Y[N_Y] * ud_Xc[i,N_Y]  +  q_j_p1_Y[N_Y] * ud_Xc_Np1  # Dust
            Steep_ud_X  = SteepOrNot( ud_X[i,N_Y], ud_Xc[i,N_Y-1], ud_Xc[i,N_Y], ud_Xc_Np1 )

            ud_Y[i,N_Y] = q_j_m1_Y[N_Y] * ud_Yc[i,N_Y-1]  +  q_j_0_Y[N_Y] * ud_Yc[i,N_Y]  +  q_j_p1_Y[N_Y] * ud_Yc_Np1
            Steep_ud_Y  = SteepOrNot( ud_Y[i,N_Y], ud_Yc[i,N_Y-1], ud_Yc[i,N_Y], ud_Yc_Np1 )

            # GET STEEPNESS INDICATOR:
            Steep = max( Steep_u_X, Steep_ud_X, Steep_u_Y, Steep_ud_Y, Steep_P )

            # SECOND-ORDER SAFETY FALLBACK:
            if Shock[i,N_Y] == Shock_HLL_and_PLM || Steep == 1 || P[i,N_Y] ≤ P_floor || !isfinite(P[i,N_Y])
                u_X[i,N_Y]  = u_Xc[i,N_Y] 
                u_Y[i,N_Y]  = u_Yc[i,N_Y]
                P[i,N_Y]    = Pc[i,N_Y] 

                ud_X[i,N_Y] = ud_Xc[i,N_Y] 
                ud_Y[i,N_Y] = ud_Yc[i,N_Y]
            end    
        end
    end

    return nothing 
end


""" This function updates primitive variables in the 
    ghost cells, needed for proper boundary conditions & 
    flux calculations at those boundaries. """
@views function FillGhosts!( ρg::AbstractMatrix{Float64},
                           u_Xg::AbstractMatrix{Float64},
                           u_Yg::AbstractMatrix{Float64},
                             Pg::AbstractMatrix{Float64},
                            ρdg::AbstractMatrix{Float64},
                          ud_Xg::AbstractMatrix{Float64},
                          ud_Yg::AbstractMatrix{Float64},
                      BC_X_left::Symbol = :Reflect,
                     BC_X_right::Symbol = :Outflow,
                      BC_Y_left::Symbol = :Reflect, 
                     BC_Y_right::Symbol = :Outflow )

    N_Xg, N_Yg = size( ρg )     # Size of ghost-padded arrays  
    N_X        = N_Xg - 2 * ng  # Number of physical cells in X-direction 
    N_Y        = N_Yg - 2 * ng  # Number of physical cells in Y-direction
    i_int_min  = ng + 1         # First interior index in X-direction 
    i_int_max  = ng + N_X       # Last interior index in X-direction
    j_int_min  = ng + 1         # First interior index in Y-direction 
    j_int_max  = ng + N_Y       # Last interior index in Y-direction

    @inbounds begin
        ##########################################                       
        ######      X = 0 LOWER BOUNDARY    ######
        ##########################################

        if BC_X_left == :Reflect 
            for k in dng
                # Extrapolate density, pressure, Y-velocity:
                ρg[ng+1-k, :]    .=    ρg[ng+k, :]
                Pg[ng+1-k, :]    .=    Pg[ng+k, :]
                u_Yg[ng+1-k, :]  .=    u_Yg[ng+k, :]

                ρdg[ng+1-k, :]   .=    ρdg[ng+k, :]
                ud_Yg[ng+1-k, :] .=    ud_Yg[ng+k, :]
                # Reflect X-velocity:
                u_Xg[ng+1-k, :]  .= .- u_Xg[ng+k, :] 
                ud_Xg[ng+1-k, :] .= .- ud_Xg[ng+k, :] 
            end
        elseif BC_X_left == :Outflow 
            for k in dng
                # Extrapolate density, pressure, Y-velocity:
                ρg[ng+1-k, :]    .=  ρg[ng+k, :]
                Pg[ng+1-k, :]    .=  Pg[ng+k, :]
                u_Yg[ng+1-k, :]  .=  u_Yg[ng+k, :]

                ρdg[ng+1-k, :]   .=  ρdg[ng+k, :]
                ud_Yg[ng+1-k, :] .=  ud_Yg[ng+k, :]
                # Only allow outflow (i.e. u_X ≤ 0.0):
                u_Xg[ng+1-k, :]  .=  min.( u_Xg[ng+k, :], 0.0 )
                ud_Xg[ng+1-k, :] .=  min.( ud_Xg[ng+k, :], 0.0 )
            end
        elseif BC_X_left == :Periodic
            # Left ghost cells:
            left_ghost  = 1:ng 
            # Left ghost cells should look like the last interior cells:
            left_source = (i_int_max - ng + 1):i_int_max
            # Periodic BCs:
            ρg[left_ghost, :]    .= ρg[left_source, :]
            Pg[left_ghost, :]    .= Pg[left_source, :]
            u_Xg[left_ghost, :]  .= u_Xg[left_source, :]
            u_Yg[left_ghost, :]  .= u_Yg[left_source, :]

            ρdg[left_ghost, :]   .= ρdg[left_source, :]
            ud_Xg[left_ghost, :] .= ud_Xg[left_source, :]
            ud_Yg[left_ghost, :] .= ud_Yg[left_source, :]
        end

        ##############################################                      
        ######      X = Xmax OUTER BOUNDARY     ######
        ##############################################

        if BC_X_right == :Reflect 
            for k in dng
                # Extrapolate density, pressure, Y-velocity:
                ρg[i_int_max + k, :]    .=    ρg[i_int_max + 1 - k, :]
                Pg[i_int_max + k, :]    .=    Pg[i_int_max + 1 - k, :]
                u_Yg[i_int_max + k, :]  .=    u_Yg[i_int_max + 1 - k, :]

                ρdg[i_int_max + k, :]   .=    ρdg[i_int_max + 1 - k, :]
                ud_Yg[i_int_max + k, :] .=    ud_Yg[i_int_max + 1 - k, :]
                # Reflect X-velocity: 
                u_Xg[i_int_max + k, :]  .= .- u_Xg[i_int_max + 1 - k, :]
                ud_Xg[i_int_max + k, :] .= .- ud_Xg[i_int_max + 1 - k, :]
            end
        elseif BC_X_right == :Outflow 
            # Extrapolate density, pressure, Y-velocity:
            ρg[i_int_max+1:i_int_max+ng, :]    .= ρg[i_int_max:i_int_max, :]
            Pg[i_int_max+1:i_int_max+ng, :]    .= Pg[i_int_max:i_int_max, :]
            u_Yg[i_int_max+1:i_int_max+ng, :]  .= u_Yg[i_int_max:i_int_max, :]

            ρdg[i_int_max+1:i_int_max+ng, :]   .= ρdg[i_int_max:i_int_max, :]
            ud_Yg[i_int_max+1:i_int_max+ng, :] .= ud_Yg[i_int_max:i_int_max, :]
            # Only allow outflow (i.e u_X ≥ 0.0):
            u_Xg[i_int_max+1:i_int_max+ng, :]  .= max.( u_Xg[i_int_max:i_int_max, :], 0.0 )
            ud_Xg[i_int_max+1:i_int_max+ng, :] .= max.( ud_Xg[i_int_max:i_int_max, :], 0.0 )
        elseif BC_X_right == :Periodic 
            # Right ghost cells:
            right_ghost  = (i_int_max + 1):(i_int_max + ng)
            # Right ghost cells should look like the first interior cells:
            right_source = i_int_min:(i_int_min + ng - 1)
            # Periodic BCs:
            ρg[right_ghost, :]    .= ρg[right_source, :]
            Pg[right_ghost, :]    .= Pg[right_source, :]
            u_Xg[right_ghost, :]  .= u_Xg[right_source, :]
            u_Yg[right_ghost, :]  .= u_Yg[right_source, :]

            ρdg[right_ghost, :]   .= ρdg[right_source, :]
            ud_Xg[right_ghost, :] .= ud_Xg[right_source, :]
            ud_Yg[right_ghost, :] .= ud_Yg[right_source, :]
        end  

        ##########################################
        ######      Y = 0 LOWER BOUNDARY    ######
        ##########################################

        if BC_Y_left == :Reflect
            for k in dng
                # Extrapolate density, pressure, X-velocity:
                ρg[:, ng+1-k]    .=    ρg[:, ng+k]
                Pg[:, ng+1-k]    .=    Pg[:, ng+k]
                u_Xg[:, ng+1-k]  .=    u_Xg[:, ng+k]

                ρdg[:, ng+1-k]   .=    ρdg[:, ng+k]
                ud_Xg[:, ng+1-k] .=    ud_Xg[:, ng+k]
                # Reflect Y-velocity:
                u_Yg[:, ng+1-k]  .= .- u_Yg[:, ng+k]
                ud_Yg[:, ng+1-k] .= .- ud_Yg[:, ng+k]
            end
        elseif BC_Y_left == :Outflow
            for k in dng
                ρg[:, ng+1-k]    .=  ρg[:, ng+k]
                Pg[:, ng+1-k]    .=  Pg[:, ng+k]
                u_Xg[:, ng+1-k]  .=  u_Xg[:, ng+k]

                ρdg[:, ng+1-k]   .=  ρdg[:, ng+k]
                ud_Xg[:, ng+1-k] .=  ud_Xg[:, ng+k]
                # Forbid inflow (i.e. u_Y ≤ 0):
                u_Yg[:, ng+1-k]  .=  min.( u_Yg[:, ng+k], 0.0 )
                ud_Yg[:, ng+1-k] .=  min.( ud_Yg[:, ng+k], 0.0 )
            end
        elseif BC_Y_left == :Periodic
            # Left ghost cells:
            left_ghost  = 1:ng
            # Left ghost cells should look like the last interior cells:
            left_source = (j_int_max - ng + 1):j_int_max  
            # Periodic BCs:
            ρg[:, left_ghost]    .= ρg[:, left_source]
            Pg[:, left_ghost]    .= Pg[:, left_source]
            u_Xg[:, left_ghost]  .= u_Xg[:, left_source]
            u_Yg[:, left_ghost]  .= u_Yg[:, left_source]

            ρdg[:, left_ghost]   .= ρdg[:, left_source]
            ud_Xg[:, left_ghost] .= ud_Xg[:, left_source]
            ud_Yg[:, left_ghost] .= ud_Yg[:, left_source]
        end

        ##############################################
        ######      Y = Ymax OUTER BOUNDARY     ######
        ##############################################

        if BC_Y_right == :Reflect
            for k in dng
                # # Extrapolate density, pressure, X-velocity:
                ρg[:, j_int_max + k]    .=    ρg[:, j_int_max + 1 - k]
                Pg[:, j_int_max + k]    .=    Pg[:, j_int_max + 1 - k]
                u_Xg[:, j_int_max + k]  .=    u_Xg[:, j_int_max + 1 - k]

                ρdg[:, j_int_max + k]   .=    ρdg[:, j_int_max + 1 - k]
                ud_Xg[:, j_int_max + k] .=    ud_Xg[:, j_int_max + 1 - k]
                # Reflect Y-velocity:
                u_Yg[:, j_int_max + k]  .= .- u_Yg[:, j_int_max + 1 - k]
                ud_Yg[:, j_int_max + k] .= .- ud_Yg[:, j_int_max + 1 - k]
            end
        elseif BC_Y_right == :Outflow
            # Extrapolate density, pressure, X-velocity:
            ρg[:, j_int_max+1:j_int_max+ng]    .= ρg[:, j_int_max:j_int_max]
            Pg[:, j_int_max+1:j_int_max+ng]    .= Pg[:, j_int_max:j_int_max]
            u_Xg[:, j_int_max+1:j_int_max+ng]  .= u_Xg[:, j_int_max:j_int_max]

            ρdg[:, j_int_max+1:j_int_max+ng]   .= ρdg[:, j_int_max:j_int_max]
            ud_Xg[:, j_int_max+1:j_int_max+ng] .= ud_Xg[:, j_int_max:j_int_max]
            # Only allow outflow (u_Y ≥ 0.0):
            u_Yg[:, j_int_max+1:j_int_max+ng]  .= max.( u_Yg[:, j_int_max:j_int_max], 0.0 )
            ud_Yg[:, j_int_max+1:j_int_max+ng] .= max.( ud_Yg[:, j_int_max:j_int_max], 0.0 )
        elseif BC_Y_right == :Periodic
            # Right ghost cells:
            right_ghost  = (j_int_max + 1):(j_int_max + ng)
            # Right ghost cells should look like the first interior cells:
            right_source = j_int_min:(j_int_min + ng - 1)
            # Periodic BCs:
            ρg[:, right_ghost]    .= ρg[:, right_source]
            Pg[:, right_ghost]    .= Pg[:, right_source]
            u_Xg[:, right_ghost]  .= u_Xg[:, right_source]
            u_Yg[:, right_ghost]  .= u_Yg[:, right_source]

            ρdg[:, right_ghost]   .= ρdg[:, right_source]
            ud_Xg[:, right_ghost] .= ud_Xg[:, right_source]
            ud_Yg[:, right_ghost] .= ud_Yg[:, right_source]
        end
        
    end # End of inbounds 

    return nothing
end


""" This function updates the X-interface values 
    of primitive variables W ∈ (ρ, u, P), using
    a third-order accurate WENO3 approximation,
    or lower-order fall back methods. """
function UpdateInterfaceValues_X!( ρ_L_X::AbstractMatrix{Float64},  # Left interface value of ρ
                                   ρ_R_X::AbstractMatrix{Float64},  # Right interface value of ρ 
                                   P_L_X::AbstractMatrix{Float64},  # Left interface value of P
                                   P_R_X::AbstractMatrix{Float64},  # Right interface value of P 
                                 u_X_L_X::AbstractMatrix{Float64},  # Left interface value of u_X
                                 u_X_R_X::AbstractMatrix{Float64},  # Right interface value of u_X 
                                 u_Y_L_X::AbstractMatrix{Float64},  # Left interface value of u_Y
                                 u_Y_R_X::AbstractMatrix{Float64},  # Right interface value of u_Y 
                                  ρd_L_X::AbstractMatrix{Float64},  # Left interface value of ρd
                                  ρd_R_X::AbstractMatrix{Float64},  # Right interface value of ρd
                                ud_X_L_X::AbstractMatrix{Float64},  # Left interface value of ud_X
                                ud_X_R_X::AbstractMatrix{Float64},  # Right interface value of ud_X 
                                ud_Y_L_X::AbstractMatrix{Float64},  # Left interface value of ud_Y
                                ud_Y_R_X::AbstractMatrix{Float64},  # Right interface value of ud_Y 
                                      ρg::AbstractMatrix{Float64},  # Ghost-padded ρ    (size N_X + 4 × N_Y + 4)
                                      Pg::AbstractMatrix{Float64},  # Ghost-padded P    (size N_X + 4 × N_Y + 4)
                                    u_Xg::AbstractMatrix{Float64},  # Ghost-padded u_X  (size N_X + 4 × N_Y + 4)
                                    u_Yg::AbstractMatrix{Float64},  # Ghost-padded u_Y  (size N_X + 4 × N_Y + 4)
                                     ρdg::AbstractMatrix{Float64},  # Ghost-padded ρd   (size N_X + 4 × N_Y + 4)
                                   ud_Xg::AbstractMatrix{Float64},  # Ghost-padded ud_X (size N_X + 4 × N_Y + 4)
                                   ud_Yg::AbstractMatrix{Float64},  # Ghost-padded ud_Y (size N_X + 4 × N_Y + 4)
                                      Xg::AbstractVector{Float64},  # Ghost-padded X-interfaces                     (length N_X + 5)
                                   XCoMg::AbstractVector{Float64},  # Ghost-padded X center-of-mass                 (length N_X + 4)
                                   Shock::AbstractMatrix{Int},      # Shock indicator
                             w_plus_i_p1::AbstractVector{Float64},  # WENO3 weights w⁺_{i,1}
                            w_minus_i_p1::AbstractVector{Float64},  # WENO3 weights w⁻_{i,1}
                                      )

    # Grid size:
    N_X = size( ρg, 1 )  -  2 * ng
    N_Y = size( ρg, 2 )  -  2 * ng

    @threads for j in 1:N_Y
        j0 = j + ng  # Ghost-padded j-index 

        @inbounds @simd for i in 1:(N_X+1)
            i0  = i  + ng          # Ghost-padded face index 
            k   = i0 - 1           # Ghost-padded left cell index 
            icl = min(i, N_X)
            im1 = max(icl - 1, 1)  # Clamped i - 1, appropriately used in places where mirroring is justified

            # X-grid related quantities:
            X_im1     = Xg[i0-1]
            X_i       = Xg[i0]
            X_ip1     = Xg[i0+1]

            XCoM_im2  = XCoMg[k-1]
            XCoM_im1  = XCoMg[k]
            XCoM_i    = XCoMg[k+1]
            XCoM_ip1  = XCoMg[k+2]

            # Get shock indicators for cells i and i - 1:
            Shock_i   = Shock[icl,j]
            Shock_im1 = Shock[im1,j]

            ###### DENSITY ######################
            ρ_im2   = ρg[k-1, j0]
            ρ_im1   = ρg[k,   j0]
            ρ_i     = ρg[k+1, j0]
            ρ_ip1   = ρg[k+2, j0]

            ρd_im2   = ρdg[k-1, j0]
            ρd_im1   = ρdg[k,   j0]
            ρd_i     = ρdg[k+1, j0]
            ρd_ip1   = ρdg[k+2, j0]

            # Get candidate left & right ρ at interface i-1/2,
            # along with maximum reconstruction order:
            ρL_X, ρR_X, ReconstructionOrder_ρ = Get_QL_and_QR( ρ_im2, ρ_im1, ρ_i, ρ_ip1,
                                                               X_im1, X_i,   X_ip1,
                                                               XCoM_im2, XCoM_im1, XCoM_i, XCoM_ip1,
                                                               Shock_i, Shock_im1,
                                                               N_X, w_plus_i_p1[im1], w_minus_i_p1[icl], 
                                                               true, Int8(3) ) 

            ρdL_X, ρdR_X, ReconstructionOrder_ρd = Get_QL_and_QR( ρd_im2, ρd_im1, ρd_i, ρd_ip1,
                                                                  X_im1, X_i,   X_ip1,
                                                                  XCoM_im2, XCoM_im1, XCoM_i, XCoM_ip1,
                                                                  Shock_i, Shock_im1,
                                                                  N_X, w_plus_i_p1[im1], w_minus_i_p1[icl], 
                                                                  true, Int8(3) ) 

            # Set maximum allowed reconstruction order for the interface:
            MaxReconstructionOrder = min( ReconstructionOrder_ρ, ReconstructionOrder_ρd )

            ###### PRESSURE #####################
            P_im2   = Pg[k-1, j0]
            P_im1   = Pg[k,   j0]
            P_i     = Pg[k+1, j0]
            P_ip1   = Pg[k+2, j0]

            # Get candidate left & right P at interface i-1/2,
            # along with maximum reconstruction order:
            PL_X, PR_X, ReconstructionOrder_P = Get_QL_and_QR( P_im2, P_im1, P_i, P_ip1,
                                                               X_im1, X_i,   X_ip1,
                                                               XCoM_im2, XCoM_im1, XCoM_i, XCoM_ip1,
                                                               Shock_i, Shock_im1,
                                                               N_X, w_plus_i_p1[im1], w_minus_i_p1[icl],
                                                               true, Int8(3) ) 

            # Update maximum allowed reconstruction order for the interface:
            MaxReconstructionOrder = min( MaxReconstructionOrder, ReconstructionOrder_P )

            ###### X VELOCITY ###################
            u_X_im2  = u_Xg[k-1, j0]
            u_X_im1  = u_Xg[k,   j0]
            u_X_i    = u_Xg[k+1, j0]
            u_X_ip1  = u_Xg[k+2, j0]

            ud_X_im2 = ud_Xg[k-1, j0]
            ud_X_im1 = ud_Xg[k,   j0]
            ud_X_i   = ud_Xg[k+1, j0]
            ud_X_ip1 = ud_Xg[k+2, j0]

            # Get candidate left & right u_X at interface i-1/2,
            # along with maximum reconstruction order:
            uX_L_X, uX_R_X, ReconstructionOrder_uX = Get_QL_and_QR( u_X_im2, u_X_im1, u_X_i, u_X_ip1,
                                                                    X_im1, X_i,   X_ip1,
                                                                    XCoM_im2, XCoM_im1, XCoM_i, XCoM_ip1,
                                                                    Shock_i, Shock_im1,
                                                                    N_X, w_plus_i_p1[im1], w_minus_i_p1[icl],
                                                                    false, Int8(3) ) 

            udX_L_X, udX_R_X, ReconstructionOrder_udX = Get_QL_and_QR( ud_X_im2, ud_X_im1, ud_X_i, ud_X_ip1,
                                                                       X_im1, X_i,   X_ip1,
                                                                       XCoM_im2, XCoM_im1, XCoM_i, XCoM_ip1,
                                                                       Shock_i, Shock_im1,
                                                                       N_X, w_plus_i_p1[im1], w_minus_i_p1[icl],
                                                                       false, Int8(3) ) 

            # Update maximum allowed reconstruction order for the interface:
            MaxReconstructionOrder = min( MaxReconstructionOrder, ReconstructionOrder_uX, ReconstructionOrder_udX )

            ###### Y VELOCITY ###################
            u_Y_im2  = u_Yg[k-1, j0]
            u_Y_im1  = u_Yg[k,   j0]
            u_Y_i    = u_Yg[k+1, j0]
            u_Y_ip1  = u_Yg[k+2, j0]

            ud_Y_im2 = ud_Yg[k-1, j0]
            ud_Y_im1 = ud_Yg[k,   j0]
            ud_Y_i   = ud_Yg[k+1, j0]
            ud_Y_ip1 = ud_Yg[k+2, j0]

            # Get candidate left & right u_Y at interface i-1/2,
            # along with maximum reconstruction order:
            uY_L_X, uY_R_X, ReconstructionOrder_uY = Get_QL_and_QR( u_Y_im2, u_Y_im1, u_Y_i, u_Y_ip1,
                                                                    X_im1, X_i,   X_ip1,
                                                                    XCoM_im2, XCoM_im1, XCoM_i, XCoM_ip1,
                                                                    Shock_i, Shock_im1,
                                                                    N_X, w_plus_i_p1[im1], w_minus_i_p1[icl],
                                                                    false, Int8(3) ) 

            udY_L_X, udY_R_X, ReconstructionOrder_udY = Get_QL_and_QR( ud_Y_im2, ud_Y_im1, ud_Y_i, ud_Y_ip1,
                                                                       X_im1, X_i,   X_ip1,
                                                                       XCoM_im2, XCoM_im1, XCoM_i, XCoM_ip1,
                                                                       Shock_i, Shock_im1,
                                                                       N_X, w_plus_i_p1[im1], w_minus_i_p1[icl],
                                                                       false, Int8(3) ) 

            # Update maximum allowed reconstruction order for the interface:
            MaxReconstructionOrder = min( MaxReconstructionOrder, ReconstructionOrder_uY, ReconstructionOrder_udY )

            # Next check if the reconstructon order for all primitives are consistent. If not,
            # re-do the ones that have too high reconstruction order:
            
            if ReconstructionOrder_ρ > MaxReconstructionOrder
                ρL_X, ρR_X, _ = Get_QL_and_QR( ρ_im2, ρ_im1, ρ_i, ρ_ip1,
                                               X_im1, X_i,   X_ip1,
                                               XCoM_im2, XCoM_im1, XCoM_i, XCoM_ip1,
                                               Shock_i, Shock_im1,
                                               N_X, w_plus_i_p1[im1], w_minus_i_p1[icl], 
                                               true, MaxReconstructionOrder ) 
            end
            ρ_L_X[i,j], ρ_R_X[i,j] = max(ρL_X, ρ_floor), max(ρR_X, ρ_floor) # Commit  

            if ReconstructionOrder_ρd > MaxReconstructionOrder
                ρdL_X, ρdR_X, _ = Get_QL_and_QR( ρd_im2, ρd_im1, ρd_i, ρd_ip1,
                                                 X_im1, X_i,   X_ip1,
                                                 XCoM_im2, XCoM_im1, XCoM_i, XCoM_ip1,
                                                 Shock_i, Shock_im1,
                                                 N_X, w_plus_i_p1[im1], w_minus_i_p1[icl], 
                                                 true, MaxReconstructionOrder ) 
            end
            ρd_L_X[i,j], ρd_R_X[i,j] = max(ρdL_X, ρd_floor), max(ρdR_X, ρd_floor) # Commit  

            if ReconstructionOrder_P > MaxReconstructionOrder
                PL_X, PR_X, _ = Get_QL_and_QR( P_im2, P_im1, P_i, P_ip1,
                                               X_im1, X_i,   X_ip1,
                                               XCoM_im2, XCoM_im1, XCoM_i, XCoM_ip1,
                                               Shock_i, Shock_im1,
                                               N_X, w_plus_i_p1[im1], w_minus_i_p1[icl], 
                                               true, MaxReconstructionOrder ) 
            end
            P_L_X[i,j], P_R_X[i,j] = max(PL_X, P_floor), max(PR_X, P_floor) # Commit 

            if ReconstructionOrder_uX > MaxReconstructionOrder
                uX_L_X, uX_R_X, _ = Get_QL_and_QR( u_X_im2, u_X_im1, u_X_i, u_X_ip1,
                                                   X_im1, X_i,   X_ip1,
                                                   XCoM_im2, XCoM_im1, XCoM_i, XCoM_ip1,
                                                   Shock_i, Shock_im1,
                                                   N_X, w_plus_i_p1[im1], w_minus_i_p1[icl], 
                                                   false, MaxReconstructionOrder ) 
            end
            u_X_L_X[i,j], u_X_R_X[i,j] = uX_L_X, uX_R_X # Commit
            
            if ReconstructionOrder_udX > MaxReconstructionOrder
                udX_L_X, udX_R_X, _ = Get_QL_and_QR( ud_X_im2, ud_X_im1, ud_X_i, ud_X_ip1,
                                                     X_im1, X_i,   X_ip1,
                                                     XCoM_im2, XCoM_im1, XCoM_i, XCoM_ip1,
                                                     Shock_i, Shock_im1,
                                                     N_X, w_plus_i_p1[im1], w_minus_i_p1[icl], 
                                                     false, MaxReconstructionOrder ) 
            end
            ud_X_L_X[i,j], ud_X_R_X[i,j] = udX_L_X, udX_R_X # Commit

            if ReconstructionOrder_uY > MaxReconstructionOrder
                uY_L_X, uY_R_X, _ = Get_QL_and_QR( u_Y_im2, u_Y_im1, u_Y_i, u_Y_ip1,
                                                   X_im1, X_i,   X_ip1,
                                                   XCoM_im2, XCoM_im1, XCoM_i, XCoM_ip1,
                                                   Shock_i, Shock_im1,
                                                   N_X, w_plus_i_p1[im1], w_minus_i_p1[icl], 
                                                   false, MaxReconstructionOrder )
            end
            u_Y_L_X[i,j], u_Y_R_X[i,j] = uY_L_X, uY_R_X # Commit 

            if ReconstructionOrder_udY > MaxReconstructionOrder
                udY_L_X, udY_R_X, _ = Get_QL_and_QR( ud_Y_im2, ud_Y_im1, ud_Y_i, ud_Y_ip1,
                                                     X_im1, X_i,   X_ip1,
                                                     XCoM_im2, XCoM_im1, XCoM_i, XCoM_ip1,
                                                     Shock_i, Shock_im1,
                                                     N_X, w_plus_i_p1[im1], w_minus_i_p1[icl], 
                                                     false, MaxReconstructionOrder )
            end
            ud_Y_L_X[i,j], ud_Y_R_X[i,j] = udY_L_X, udY_R_X # Commit 

        end

    end # End of j-loop 
    
    return nothing
end


""" This function updates the Y-interface values 
    of primitive variables W ∈ (ρ, u, P), using
    a third-order accurate WENO3 approximation,
    or lower-order fall back methods. """
function UpdateInterfaceValues_Y!( ρ_L_Y::AbstractMatrix{Float64},  # Left interface value of ρ  
                                   ρ_R_Y::AbstractMatrix{Float64},  # Right interface value of ρ 
                                   P_L_Y::AbstractMatrix{Float64},  # Left interface value of P  
                                   P_R_Y::AbstractMatrix{Float64},  # Right interface value of P
                                 u_X_L_Y::AbstractMatrix{Float64},  # Left interface value of u_X 
                                 u_X_R_Y::AbstractMatrix{Float64},  # Right interface value of u_X 
                                 u_Y_L_Y::AbstractMatrix{Float64},  # Left interface value of u_Y 
                                 u_Y_R_Y::AbstractMatrix{Float64},  # Right interface value of u_Y  
                                  ρd_L_Y::AbstractMatrix{Float64},  # Left interface value of ρd  
                                  ρd_R_Y::AbstractMatrix{Float64},  # Right interface value of ρd
                                ud_X_L_Y::AbstractMatrix{Float64},  # Left interface value of ud_X 
                                ud_X_R_Y::AbstractMatrix{Float64},  # Right interface value of ud_X 
                                ud_Y_L_Y::AbstractMatrix{Float64},  # Left interface value of ud_Y 
                                ud_Y_R_Y::AbstractMatrix{Float64},  # Right interface value of ud_Y
                                      ρg::AbstractMatrix{Float64},  # Ghost-padded ρ    (size N_X + 4 × N_Y + 4)
                                      Pg::AbstractMatrix{Float64},  # Ghost-padded P    (size N_X + 4 × N_Y + 4)
                                    u_Xg::AbstractMatrix{Float64},  # Ghost-padded u_X  (size N_X + 4 × N_Y + 4)
                                    u_Yg::AbstractMatrix{Float64},  # Ghost-padded u_Y  (size N_X + 4 × N_Y + 4)
                                     ρdg::AbstractMatrix{Float64},  # Ghost-padded ρd   (size N_X + 4 × N_Y + 4)
                                   ud_Xg::AbstractMatrix{Float64},  # Ghost-padded ud_X (size N_X + 4 × N_Y + 4)
                                   ud_Yg::AbstractMatrix{Float64},  # Ghost-padded ud_Y (size N_X + 4 × N_Y + 4)
                                      Yg::AbstractVector{Float64},  # Ghost-padded Y-interfaces                     (length N_Y + 5)
                                   YCoMg::AbstractVector{Float64},  # Ghost-padded Y center-of-mass                 (length N_Y + 4)
                                   Shock::AbstractMatrix{Int},      # Shock indicator 
                             w_plus_j_p1::AbstractVector{Float64},  # WENO3 weights w⁺_{j,1}
                            w_minus_j_p1::AbstractVector{Float64},  # WENO3 weights w⁻_{j,1}
                                  )

    # Grid size:
    N_X = size( ρg, 1 )  -  2 * ng
    N_Y = size( ρg, 2 )  -  2 * ng

    @threads for j in 1:(N_Y+1)                  
        j0  = j + ng                        # Ghost-padded interface index
        jcl = min(j, N_Y)
        jm1 = max(jcl - 1, 1)  # Clamped j - 1, appropriately used in places where mirroring is justified

        # Indices needed for interface j-1/2:
        c_m2     = j0 - 2                  # j-2
        c_m1     = j0 - 1                  # j-1 
        c_0      = j0                      # j   
        c_p1     = j0 + 1                  # j+1

        # Interfaces & CoM positions:
        Y_im1    = Yg[j0-1]               
        Y_i      = Yg[j0]                  # The interface y_{j-1/2}
        Y_ip1    = Yg[j0+1]               

        YCoM_im2 = YCoMg[c_m2]
        YCoM_im1 = YCoMg[c_m1]
        YCoM_i   = YCoMg[c_0]
        YCoM_ip1 = YCoMg[c_p1]

        @inbounds @simd for i in 1:N_X     
            i0 = i + ng     # Ghost-padded i-index for cell centers

            # Get shock indicators for cells j and j - 1:
            Shock_i   = Shock[i,jcl]
            Shock_im1 = Shock[i,jm1]

            ###### DENSITY ######################
            ρ_im2    = ρg[i0, c_m2]
            ρ_im1    = ρg[i0, c_m1]
            ρ_i      = ρg[i0, c_0]
            ρ_ip1    = ρg[i0, c_p1]

            ρd_im2   = ρdg[i0, c_m2]
            ρd_im1   = ρdg[i0, c_m1]
            ρd_i     = ρdg[i0, c_0]
            ρd_ip1   = ρdg[i0, c_p1]

            # Get candidate left & right ρ at interface j-1/2,
            # along with maximum reconstruction order:
            ρL_Y, ρR_Y, ReconstructionOrder_ρ = Get_QL_and_QR( ρ_im2, ρ_im1, ρ_i, ρ_ip1,
                                                               Y_im1, Y_i, Y_ip1,
                                                               YCoM_im2, YCoM_im1, YCoM_i, YCoM_ip1,
                                                               Shock_i, Shock_im1,
                                                               N_Y, w_plus_j_p1[jm1], w_minus_j_p1[jcl], 
                                                               true, Int8(3) ) 

            ρdL_Y, ρdR_Y, ReconstructionOrder_ρd = Get_QL_and_QR( ρd_im2, ρd_im1, ρd_i, ρd_ip1,
                                                                  Y_im1, Y_i, Y_ip1,
                                                                  YCoM_im2, YCoM_im1, YCoM_i, YCoM_ip1,
                                                                  Shock_i, Shock_im1,
                                                                  N_Y, w_plus_j_p1[jm1], w_minus_j_p1[jcl], 
                                                                  true, Int8(3) ) 

            # Set maximum allowed reconstruction order for the interface:
            MaxReconstructionOrder = min( ReconstructionOrder_ρ, ReconstructionOrder_ρd )

            ###### PRESSURE #####################
            P_im2   = Pg[i0, c_m2]
            P_im1   = Pg[i0, c_m1]
            P_i     = Pg[i0, c_0]
            P_ip1   = Pg[i0, c_p1]

            # Get candidate left & right P at interface j-1/2,
            # along with maximum reconstruction order:
            PL_Y, PR_Y, ReconstructionOrder_P = Get_QL_and_QR( P_im2, P_im1, P_i, P_ip1,
                                                               Y_im1, Y_i, Y_ip1,
                                                               YCoM_im2, YCoM_im1, YCoM_i, YCoM_ip1,
                                                               Shock_i, Shock_im1,
                                                               N_Y, w_plus_j_p1[jm1], w_minus_j_p1[jcl],
                                                               true, Int8(3) ) 

            # Update maximum allowed reconstruction order for the interface:
            MaxReconstructionOrder = min( MaxReconstructionOrder, ReconstructionOrder_P )

            ###### X VELOCITY ###################
            u_X_im2  = u_Xg[i0, c_m2]
            u_X_im1  = u_Xg[i0, c_m1]
            u_X_i    = u_Xg[i0, c_0]
            u_X_ip1  = u_Xg[i0, c_p1]

            ud_X_im2 = ud_Xg[i0, c_m2]
            ud_X_im1 = ud_Xg[i0, c_m1]
            ud_X_i   = ud_Xg[i0, c_0]
            ud_X_ip1 = ud_Xg[i0, c_p1]

            # Get candidate left & right u_X at interface j-1/2,
            # along with maximum reconstruction order:
            uX_L_Y, uX_R_Y, ReconstructionOrder_uX = Get_QL_and_QR( u_X_im2, u_X_im1, u_X_i, u_X_ip1,
                                                                    Y_im1, Y_i, Y_ip1,
                                                                    YCoM_im2, YCoM_im1, YCoM_i, YCoM_ip1,
                                                                    Shock_i, Shock_im1,
                                                                    N_Y, w_plus_j_p1[jm1], w_minus_j_p1[jcl],
                                                                    false, Int8(3) )
                                                                    
            udX_L_Y, udX_R_Y, ReconstructionOrder_udX = Get_QL_and_QR( ud_X_im2, ud_X_im1, ud_X_i, ud_X_ip1,
                                                                       Y_im1, Y_i, Y_ip1,
                                                                       YCoM_im2, YCoM_im1, YCoM_i, YCoM_ip1,
                                                                       Shock_i, Shock_im1,
                                                                       N_Y, w_plus_j_p1[jm1], w_minus_j_p1[jcl],
                                                                       false, Int8(3) )

            # Update maximum allowed reconstruction order for the interface:
            MaxReconstructionOrder = min( MaxReconstructionOrder, ReconstructionOrder_uX, ReconstructionOrder_udX )

            ###### Y VELOCITY ###################
            u_Y_im2  = u_Yg[i0, c_m2]
            u_Y_im1  = u_Yg[i0, c_m1]
            u_Y_i    = u_Yg[i0, c_0]
            u_Y_ip1  = u_Yg[i0, c_p1]

            ud_Y_im2 = ud_Yg[i0, c_m2]
            ud_Y_im1 = ud_Yg[i0, c_m1]
            ud_Y_i   = ud_Yg[i0, c_0]
            ud_Y_ip1 = ud_Yg[i0, c_p1]

            # Get candidate left & right u_Y at interface j-1/2,
            # along with maximum reconstruction order:
            uY_L_Y, uY_R_Y, ReconstructionOrder_uY = Get_QL_and_QR( u_Y_im2, u_Y_im1, u_Y_i, u_Y_ip1,
                                                                    Y_im1, Y_i, Y_ip1,
                                                                    YCoM_im2, YCoM_im1, YCoM_i, YCoM_ip1,
                                                                    Shock_i, Shock_im1,
                                                                    N_Y, w_plus_j_p1[jm1], w_minus_j_p1[jcl], 
                                                                    false, Int8(3) )
                                                                    
            udY_L_Y, udY_R_Y, ReconstructionOrder_udY = Get_QL_and_QR( ud_Y_im2, ud_Y_im1, ud_Y_i, ud_Y_ip1,
                                                                       Y_im1, Y_i, Y_ip1,
                                                                       YCoM_im2, YCoM_im1, YCoM_i, YCoM_ip1,
                                                                       Shock_i, Shock_im1,
                                                                       N_Y, w_plus_j_p1[jm1], w_minus_j_p1[jcl], 
                                                                       false, Int8(3) )

            # Update maximum allowed reconstruction order for the interface:
            MaxReconstructionOrder = min( MaxReconstructionOrder, ReconstructionOrder_uY, ReconstructionOrder_udY )

            # Next check if the reconstructon order for all primitives are consistent. If not,
            # re-do the ones that have too high reconstruction order:
            
            if ReconstructionOrder_ρ > MaxReconstructionOrder
                ρL_Y, ρR_Y, _ = Get_QL_and_QR( ρ_im2, ρ_im1, ρ_i, ρ_ip1,
                                               Y_im1, Y_i, Y_ip1,
                                               YCoM_im2, YCoM_im1, YCoM_i, YCoM_ip1,
                                               Shock_i, Shock_im1,
                                               N_Y, w_plus_j_p1[jm1], w_minus_j_p1[jcl],
                                               true, MaxReconstructionOrder ) 
            end
            ρ_L_Y[i,j], ρ_R_Y[i,j] = max(ρL_Y, ρ_floor), max(ρR_Y, ρ_floor) # Commit  

            if ReconstructionOrder_ρd > MaxReconstructionOrder
                ρdL_Y, ρdR_Y, _ = Get_QL_and_QR( ρd_im2, ρd_im1, ρd_i, ρd_ip1,
                                                 Y_im1, Y_i, Y_ip1,
                                                 YCoM_im2, YCoM_im1, YCoM_i, YCoM_ip1,
                                                 Shock_i, Shock_im1,
                                                 N_Y, w_plus_j_p1[jm1], w_minus_j_p1[jcl],
                                                 true, MaxReconstructionOrder ) 
            end
            ρd_L_Y[i,j], ρd_R_Y[i,j] = max(ρdL_Y, ρd_floor), max(ρdR_Y, ρd_floor) # Commit  

            if ReconstructionOrder_P > MaxReconstructionOrder
                PL_Y, PR_Y, _ = Get_QL_and_QR( P_im2, P_im1, P_i, P_ip1,
                                               Y_im1, Y_i, Y_ip1,
                                               YCoM_im2, YCoM_im1, YCoM_i, YCoM_ip1,
                                               Shock_i, Shock_im1,
                                               N_Y, w_plus_j_p1[jm1], w_minus_j_p1[jcl],
                                               true, MaxReconstructionOrder ) 
            end
            P_L_Y[i,j], P_R_Y[i,j] = max(PL_Y, P_floor), max(PR_Y, P_floor) # Commit  

            if ReconstructionOrder_uX > MaxReconstructionOrder
                uX_L_Y, uX_R_Y, _ = Get_QL_and_QR( u_X_im2, u_X_im1, u_X_i, u_X_ip1,
                                                   Y_im1, Y_i, Y_ip1,
                                                   YCoM_im2, YCoM_im1, YCoM_i, YCoM_ip1,
                                                   Shock_i, Shock_im1,
                                                   N_Y, w_plus_j_p1[jm1], w_minus_j_p1[jcl],
                                                   false, MaxReconstructionOrder ) 
            end
            u_X_L_Y[i,j], u_X_R_Y[i,j] = uX_L_Y, uX_R_Y # Commit  

            if ReconstructionOrder_udX > MaxReconstructionOrder
                udX_L_Y, udX_R_Y, _ = Get_QL_and_QR( ud_X_im2, ud_X_im1, ud_X_i, ud_X_ip1,
                                                     Y_im1, Y_i, Y_ip1,
                                                     YCoM_im2, YCoM_im1, YCoM_i, YCoM_ip1,
                                                     Shock_i, Shock_im1,
                                                     N_Y, w_plus_j_p1[jm1], w_minus_j_p1[jcl],
                                                     false, MaxReconstructionOrder ) 
            end
            ud_X_L_Y[i,j], ud_X_R_Y[i,j] = udX_L_Y, udX_R_Y # Commit  

            if ReconstructionOrder_uY > MaxReconstructionOrder
                uY_L_Y, uY_R_Y, _ = Get_QL_and_QR( u_Y_im2, u_Y_im1, u_Y_i, u_Y_ip1,
                                                   Y_im1, Y_i, Y_ip1,
                                                   YCoM_im2, YCoM_im1, YCoM_i, YCoM_ip1,
                                                   Shock_i, Shock_im1,
                                                   N_Y, w_plus_j_p1[jm1], w_minus_j_p1[jcl],
                                                   false, MaxReconstructionOrder ) 
            end
            u_Y_L_Y[i,j], u_Y_R_Y[i,j] = uY_L_Y, uY_R_Y # Commit  

            if ReconstructionOrder_udY > MaxReconstructionOrder
                udY_L_Y, udY_R_Y, _ = Get_QL_and_QR( ud_Y_im2, ud_Y_im1, ud_Y_i, ud_Y_ip1,
                                                     Y_im1, Y_i, Y_ip1,
                                                     YCoM_im2, YCoM_im1, YCoM_i, YCoM_ip1,
                                                     Shock_i, Shock_im1,
                                                     N_Y, w_plus_j_p1[jm1], w_minus_j_p1[jcl],
                                                     false, MaxReconstructionOrder ) 
            end
            ud_Y_L_Y[i,j], ud_Y_R_Y[i,j] = udY_L_Y, udY_R_Y # Commit  

        end # End of i-loop 
    end # End of j-loop

    return nothing 
end


###############################################################
#######                                                 #######
#######           F L U X  F U N C T I O N S            #######
#######                                                 #######
###############################################################

# Below are two approximations for the fluxes at interfaces,
# namely the HLLC and HLL approximations. The former is generally
# more accurate, and used by default.

""" A function to compute the gas wave speeds. """
@inline function GetGasWaveSpeeds( ρ_L::Float64, ρ_R::Float64, u_L::Float64, u_R::Float64, 
                                   P_L::Float64, P_R::Float64, γad_L::Float64, γad_R::Float64 )

    # Sound speeds on left & right interface:
    c_L  = sqrt( γad_L * P_L / ρ_L )
    c_R  = sqrt( γad_R * P_R / ρ_R )

    # Wave speeds:
    S_L  = min( u_L - c_L, u_R - c_R )
    S_R  = max( u_L + c_L, u_R + c_R )

    return S_L, S_R
end


""" This function computes HLLC fluxes for
    the conserved variables: 

    - F_ρ  (density), 
    - F_ρu (momentum), 
    - F_ρv (transverse momentum)
    - F_E  (energy) 

    Also returns the contact speed, and wave speeds. """
function GetFlux_HLLC( ρ_L::Float64, 
                       ρ_R::Float64, 
                       u_L::Float64, 
                       u_R::Float64,
                       v_L::Float64,     
                       v_R::Float64, 
                       P_L::Float64,  
                       P_R::Float64, 
                     γad_L::Float64,
                     γad_R::Float64
                     )

    if DEBUG_Hydro
        @assert P_L ≥ 0.0 && isfinite(P_L) "P_L must be ≥ 0 in the HLLC flux. Here P_L = $(P_L), γad_R = $(γad_R), γad_L = $(γad_L). "
        @assert P_R ≥ 0.0 && isfinite(P_R) "P_R must be ≥ 0 in the HLLC flux. Here P_R = $(P_R), γad_R = $(γad_R), γad_L = $(γad_L). "
        @assert ρ_L ≥ 0.0 && isfinite(ρ_L) "ρ_L must be ≥ 0 in the HLLC flux. Here ρ_L = $(ρ_L). "
        @assert ρ_R ≥ 0.0 && isfinite(ρ_R) "ρ_R must be ≥ 0 in the HLLC flux. Here ρ_R = $(ρ_R)"
        @assert isfinite(u_L) "u_L must be finite in the HLLC flux. Here u_L = $(u_L). "
        @assert isfinite(u_R) "u_R must be finite in the HLLC flux. Here u_R = $(u_R). "
        @assert isfinite(v_L) "v_L must be finite in the HLLC flux. Here v_L = $(v_L). "
        @assert isfinite(v_R) "v_R must be finite in the HLLC flux. Here v_R = $(v_R). "
    end

    # Left & right energies:
    E_L    = P_L / (γad_L - 1)  +  0.5 * ρ_L * (u_L^2 + v_L^2)
    E_R    = P_R / (γad_R - 1)  +  0.5 * ρ_R * (u_R^2 + v_R^2)

    # Estimate wave speeds S_L and S_R:
    S_L, S_R = GetGasWaveSpeeds( ρ_L, ρ_R,
                                 u_L, u_R,
                                 P_L, P_R,
                                 γad_L, γad_R )


    # Conserved variables:
    ρu_L   = ρ_L * u_L 
    ρu_R   = ρ_R * u_R 
    ρv_L   = ρ_L * v_L   # Transverse momentum in left state
    ρv_R   = ρ_R * v_R   # Transverse momentum in right state 

    # Estimate wave speed in star region (contact speed):
    S_star = ( P_R - P_L + ρu_L * (S_L - u_L) - ρu_R * (S_R - u_R)  ) / 
             ( ρ_L * (S_L - u_L) - ρ_R * (S_R - u_R) )

    if !isfinite(S_star)
        # In case we get e.g. division by zero, use expected physical 
        # value:
        S_star = ( ρ_L * u_L  +  ρ_R * u_R ) / ( ρ_L + ρ_R )
    end

    S_star = clamp( S_star, S_L, S_R ) 

    # Compute fluxes for left state:
    F_ρ_L  = ρu_L 
    F_ρu_L = ρ_L * u_L^2  +  P_L  
    F_ρv_L = ρv_L * u_L 
    F_E_L  = u_L * (E_L + P_L)

    # Compute fluxes for right state:
    F_ρ_R  = ρu_R 
    F_ρu_R = ρ_R * u_R^2  +  P_R 
    F_ρv_R = ρv_R * u_R
    F_E_R  = u_R * (E_R + P_R )

    # Now, based on wave speeds, compute the HLLC flux:
    if S_L ≥ 0.0
        # In this case, all waves move right, so use left state fluxes:
        F_ρ  = F_ρ_L 
        F_ρu = F_ρu_L 
        F_ρv = F_ρv_L
        F_E  = F_E_L

    elseif S_R ≤ 0.0
        # In this case, all waves move left, so use right state fluxes:
        F_ρ  = F_ρ_R
        F_ρu = F_ρu_R 
        F_ρv = F_ρv_R
        F_E  = F_E_R 

    elseif S_L ≤ 0.0 && S_star ≥ 0.0
        # Get relevant quantities in the left star state:
        α_star_L  = ρ_L * ( S_L - u_L ) / ( S_L - S_star )
        ρ_L_star  = α_star_L 
        ρu_L_star = α_star_L * S_star
        ρv_L_star = ρ_L_star * v_L     # Tangential velocity is unchanged 
        E_L_star  = α_star_L * ( E_L / ρ_L + (S_star - u_L) * (S_star + P_L / (ρ_L * (S_L - u_L))) )

        # Get fluxes:
        F_ρ       = S_L * ( ρ_L_star  - ρ_L )   +  F_ρ_L
        F_ρu      = S_L * ( ρu_L_star - ρu_L )  +  F_ρu_L
        F_ρv      = S_L * ( ρv_L_star - ρv_L)   +  F_ρv_L
        F_E       = S_L * ( E_L_star  - E_L )   +  F_E_L

    else # (In this case we have the F_star_R fluxes)
        # Get relevant quantities in the right star state:
        α_star_R  = ρ_R * ( S_R - u_R ) / ( S_R - S_star )
        ρ_R_star  = α_star_R 
        ρu_R_star = α_star_R * S_star
        ρv_R_star = ρ_R_star * v_R 
        E_R_star  = α_star_R * ( E_R / ρ_R + (S_star - u_R) * (S_star + P_R / (ρ_R * (S_R - u_R))) )

        # Get fluxes:
        F_ρ       = S_R * ( ρ_R_star - ρ_R )    +  F_ρ_R 
        F_ρu      = S_R * ( ρu_R_star - ρu_R )  +  F_ρu_R
        F_ρv      = S_R * ( ρv_R_star - ρv_R )  +  F_ρv_R
        F_E       = S_R * ( E_R_star - E_R )    +  F_E_R 
    end

    return F_ρ, F_ρu, F_ρv, F_E, S_star, S_L, S_R
end


""" This function computes HLL fluxes for
    the conserved variables: 

    - F_ρ  (density), 
    - F_ρu (momentum), 
    - F_ρv (transverse momentum)
    - F_E  (energy) 

    Also returns the contact speed, and wave speeds. """
function GetFlux_HLL( ρ_L::Float64, 
                      ρ_R::Float64, 
                      u_L::Float64, 
                      u_R::Float64,
                      v_L::Float64,     
                      v_R::Float64, 
                      P_L::Float64,  
                      P_R::Float64, 
                    γad_L::Float64,
                    γad_R::Float64
                  )

    if DEBUG_Hydro
        @assert P_L ≥ 0.0 && isfinite(P_L) "P_L must be ≥ 0 in the HLL flux. Here P_L = $(P_L), γad_R = $(γad_R), γad_L = $(γad_L). "
        @assert P_R ≥ 0.0 && isfinite(P_R) "P_R must be ≥ 0 in the HLL flux. Here P_R = $(P_R), γad_R = $(γad_R), γad_L = $(γad_L). "
        @assert ρ_L ≥ 0.0 && isfinite(ρ_L) "ρ_L must be ≥ 0 in the HLL flux. Here ρ_L = $(ρ_L). "
        @assert ρ_R ≥ 0.0 && isfinite(ρ_R) "ρ_R must be ≥ 0 in the HLL flux. Here ρ_R = $(ρ_R)"
        @assert isfinite(u_L) "u_L must be finite in the HLL flux. Here u_L = $(u_L). "
        @assert isfinite(u_R) "u_R must be finite in the HLL flux. Here u_R = $(u_R). "
        @assert isfinite(v_L) "v_L must be finite in the HLL flux. Here v_L = $(v_L). "
        @assert isfinite(v_R) "v_R must be finite in the HLL flux. Here v_R = $(v_R). "
    end
    
    # Left & right energies:
    E_L    = P_L / (γad_L - 1)  +  0.5 * ρ_L * (u_L^2 + v_L^2)
    E_R    = P_R / (γad_R - 1)  +  0.5 * ρ_R * (u_R^2 + v_R^2)

    # Estimate wave speeds S_L and S_R:
    S_L, S_R = GetGasWaveSpeeds( ρ_L, ρ_R,
                                 u_L, u_R,
                                 P_L, P_R,
                                 γad_L, γad_R )

    # Conserved variables:
    ρu_L   = ρ_L * u_L 
    ρu_R   = ρ_R * u_R 
    ρv_L   = ρ_L * v_L   # Transverse momentum in left state
    ρv_R   = ρ_R * v_R   # Transverse momentum in right state 

    # Compute fluxes for left state:
    F_ρ_L  = ρu_L 
    F_ρu_L = ρ_L * u_L^2  +  P_L  
    F_ρv_L = ρv_L * u_L 
    F_E_L  = u_L * (E_L + P_L)

    # Compute fluxes for right state:
    F_ρ_R  = ρu_R 
    F_ρu_R = ρ_R * u_R^2  +  P_R 
    F_ρv_R = ρv_R * u_R
    F_E_R  = u_R * (E_R + P_R)

    if S_L ≥ 0.0
        # In this case, all waves move right, so use left state fluxes:
        F_ρ  = F_ρ_L 
        F_ρu = F_ρu_L 
        F_ρv = F_ρv_L
        F_E  = F_E_L
    elseif S_R ≤ 0.0
        # In this case, all waves move left, so use right state fluxes:
        F_ρ  = F_ρ_R
        F_ρu = F_ρu_R 
        F_ρv = F_ρv_R
        F_E  = F_E_R 
    else
        # In this case we have a mixed case, and use the HLL formula:
        inv_denom = 1 / ( S_R  -  S_L )
        F_ρ       = ( S_R * F_ρ_L  - S_L * F_ρ_R  + S_L * S_R * (ρ_R  - ρ_L)  ) * inv_denom
        F_ρu      = ( S_R * F_ρu_L - S_L * F_ρu_R + S_L * S_R * (ρu_R - ρu_L) ) * inv_denom
        F_ρv      = ( S_R * F_ρv_L - S_L * F_ρv_R + S_L * S_R * (ρv_R - ρv_L) ) * inv_denom
        F_E       = ( S_R * F_E_L  - S_L * F_E_R  + S_L * S_R * (E_R - E_L)   ) * inv_denom

    end

    # Estimate wave speed in star region (contact speed). This is not needed in 
    # the HLL flux itself, but will be needed in the M1 RT advection:
    S_star = ( P_R - P_L + ρu_L * (S_L - u_L) - ρu_R * (S_R - u_R)  ) / 
             ( ρ_L * (S_L - u_L) - ρ_R * (S_R - u_R) )

    if !isfinite(S_star)
        # In case we get e.g. division by zero, use expected physical 
        # value:
        S_star = ( ρ_L * u_L  +  ρ_R * u_R ) / ( ρ_L + ρ_R )
    end

    S_star = clamp( S_star, S_L, S_R ) 

    return F_ρ, F_ρu, F_ρv, F_E, S_star, S_L, S_R
end


""" A Local Lax-Friedrich (LLF) flux function for dust. See Verrier et al. (2025). """
@inline function GetLLFFlux_Dust( ρd_L::Float64, ρd_R::Float64,
                                  ud_L::Float64, ud_R::Float64,   # Normal components 
                                  vd_L::Float64, vd_R::Float64 )  # Transverse components 

    # Conserved variables at left & right sides:
    ρdud_L  = ρd_L * ud_L
    ρdud_R  = ρd_R * ud_R
    ρdvd_L  = ρd_L * vd_L
    ρdvd_R  = ρd_R * vd_R

    # Corresponding fluxes:
    F_ρ_L   = ρdud_L
    F_ρ_R   = ρdud_R
    F_ρud_L = ρdud_L * ud_L
    F_ρud_R = ρdud_R * ud_R
    F_ρvd_L = ρdvd_L * ud_L
    F_ρvd_R = ρdvd_R * ud_R

    # Maximum dust wave speed; 
    S       = max( abs( ud_L ), abs( ud_R ) )

    # Get interface fluxes: 
    F_ρd    = 0.5 * ( F_ρ_L   + F_ρ_R )   - 0.5 * S * ( ρd_R   - ρd_L )
    F_ρdud  = 0.5 * ( F_ρud_L + F_ρud_R ) - 0.5 * S * ( ρdud_R - ρdud_L )
    F_ρdvd  = 0.5 * ( F_ρvd_L + F_ρvd_R ) - 0.5 * S * ( ρdvd_R - ρdvd_L )

    return F_ρd, F_ρdud, F_ρdvd
end


""" The HLL flux function for dust, for a given set of wave speeds. """
@inline function GetHLLFlux_Dust( ρd_L::Float64, ρd_R::Float64,
                                  ud_L::Float64, ud_R::Float64,  # Normal components 
                                  vd_L::Float64, vd_R::Float64,  # Transverse components 
                                  S_L::Float64,   S_R::Float64 )

    # Conserved variables at left & right sides:
    ρdud_L  = ρd_L * ud_L
    ρdud_R  = ρd_R * ud_R
    ρdvd_L  = ρd_L * vd_L
    ρdvd_R  = ρd_R * vd_R

    # Corresponding fluxes:
    F_ρ_L   = ρdud_L
    F_ρ_R   = ρdud_R
    F_ρud_L = ρdud_L * ud_L
    F_ρud_R = ρdud_R * ud_R
    F_ρvd_L = ρdvd_L * ud_L
    F_ρvd_R = ρdvd_R * ud_R

    if S_L ≥ 0.0
        return F_ρ_L, F_ρud_L, F_ρvd_L
    elseif S_R ≤ 0.0
        return F_ρ_R, F_ρud_R, F_ρvd_R
    else
        inv_ΔS = 1.0 / (S_R - S_L)
        F_ρd   = ( S_R * F_ρ_L   - S_L * F_ρ_R   + S_L * S_R * (ρd_R   - ρd_L) )   * inv_ΔS
        F_ρdud = ( S_R * F_ρud_L - S_L * F_ρud_R + S_L * S_R * (ρdud_R - ρdud_L) ) * inv_ΔS
        F_ρdvd = ( S_R * F_ρvd_L - S_L * F_ρvd_R + S_L * S_R * (ρdvd_R - ρdvd_L) ) * inv_ΔS
        return F_ρd, F_ρdud, F_ρdvd
    end
end


""" The HLLgd flux function for dust, presented in Verrier et al. (2025).
    Uses gas wave speeds Sg_L and Sg_R to inform the choice. """
@inline function GetHLLgdFlux_Dust( ρd_L::Float64, ρd_R::Float64,
                                    ud_L::Float64, ud_R::Float64,  # Normal components 
                                    vd_L::Float64, vd_R::Float64,  # Transverse components 
                                    Sg_L::Float64, Sg_R::Float64 )

    # Conserved variables at left & right sides:
    ρdud_L = ρd_L * ud_L
    ρdud_R = ρd_R * ud_R
    ρdvd_L = ρd_L * vd_L
    ρdvd_R = ρd_R * vd_R

    # Corresponding fluxes:
    F_ρ_L   = ρdud_L
    F_ρ_R   = ρdud_R
    F_ρud_L = ρdud_L * ud_L
    F_ρud_R = ρdud_R * ud_R
    F_ρvd_L = ρdvd_L * ud_L
    F_ρvd_R = ρdvd_R * ud_R

    # (12): gas fan fully to the right
    if Sg_L > 0.0
        if ud_L > 0.0 && ud_R > 0.0
            return F_ρ_L, F_ρud_L, F_ρvd_L
        else
            return GetLLFFlux_Dust( ρd_L, ρd_R, ud_L, ud_R, vd_L, vd_R )
        end
    end

    # (13): gas fan fully to the left
    if Sg_R < 0.0
        if ud_L < 0.0 && ud_R < 0.0
            return F_ρ_R, F_ρud_R, F_ρvd_R
        else
            return GetLLFFlux_Dust( ρd_L, ρd_R, ud_L, ud_R, vd_L, vd_R )
        end
    end

    # (14): mixed gas fan; use HLL(d) only if BOTH dust normal velocities lie inside the gas fan
    if (Sg_L < ud_L < Sg_R) && (Sg_L < ud_R < Sg_R)
        return GetHLLFlux_Dust( ρd_L, ρd_R, ud_L, ud_R, vd_L, vd_R, Sg_L, Sg_R )
    else
        return GetLLFFlux_Dust( ρd_L, ρd_R, ud_L, ud_R, vd_L, vd_R )
    end
end


""" This function computes fluxes for
    the conserved variables (for gas and dust): 

    - F_ρ  (density), 
    - F_ρu (momentum), 
    - F_ρv (transverse momentum)
    - F_E  (energy) 

    The function also returns the contact speed. 

    It uses HLLC flux function in regions away from shocks, 
    and the more robust HLL flux function near shocks. """
@inline function GetFlux( ρ_L::Float64, 
                          ρ_R::Float64, 
                          u_L::Float64, 
                          u_R::Float64,
                          v_L::Float64,     
                          v_R::Float64, 
                          P_L::Float64,  
                          P_R::Float64, 
                        γad_L::Float64,
                        γad_R::Float64,
                         ρd_L::Float64, 
                         ρd_R::Float64, 
                         ud_L::Float64, 
                         ud_R::Float64,
                         vd_L::Float64,     
                         vd_R::Float64, 
                      Shock_L::Int     = 0,
                      Shock_R::Int     = 0
                          )

    if max( Shock_L, Shock_R ) ≥ Shock_HLL 
        # In this case we have a significant shock, and revert to the HLL 
        # flux function to avoid e.g. the carbuncle instability:
        F_ρ, F_ρu, F_ρv, F_E, S_star, S_L, S_R = GetFlux_HLL( ρ_L, ρ_R, u_L, u_R, v_L, v_R, P_L, P_R, γad_L, γad_R )
    else
        # If we don't have a significant shock, we stick to the HLLC flux function: 
        F_ρ, F_ρu, F_ρv, F_E, S_star, S_L, S_R = GetFlux_HLLC( ρ_L, ρ_R, u_L, u_R, v_L, v_R, P_L, P_R, γad_L, γad_R )
    end

    # Get dust flux:
    F_ρd, F_ρdud, F_ρdvd = GetHLLgdFlux_Dust( ρd_L, ρd_R, ud_L, ud_R, vd_L, vd_R, S_L, S_R )

    return F_ρ, F_ρu, F_ρv, F_E, S_star, F_ρd, F_ρdud, F_ρdvd
end


###############################################################
#######                                                 #######
#######            C O M P U T E  F L U X               #######
#######                                                 #######
###############################################################


""" Compute & update the flux in the X-direction. Also updates
    the contact speed. """
function ComputeFlux_X!( F_ρ_X::AbstractMatrix{Float64}, 
                      F_ρu_X_X::AbstractMatrix{Float64}, 
                      F_ρu_Y_X::AbstractMatrix{Float64}, 
                         F_E_X::AbstractMatrix{Float64}, 
                         ρ_L_X::AbstractMatrix{Float64}, 
                         ρ_R_X::AbstractMatrix{Float64},
                       u_X_L_X::AbstractMatrix{Float64}, 
                       u_X_R_X::AbstractMatrix{Float64},
                       u_Y_L_X::AbstractMatrix{Float64}, 
                       u_Y_R_X::AbstractMatrix{Float64},
                         P_L_X::AbstractMatrix{Float64}, 
                         P_R_X::AbstractMatrix{Float64},

                        F_ρd_X::AbstractMatrix{Float64}, 
                    F_ρdud_X_X::AbstractMatrix{Float64}, 
                    F_ρdud_Y_X::AbstractMatrix{Float64}, 
                        ρd_L_X::AbstractMatrix{Float64}, 
                        ρd_R_X::AbstractMatrix{Float64},
                      ud_X_L_X::AbstractMatrix{Float64}, 
                      ud_X_R_X::AbstractMatrix{Float64},
                      ud_Y_L_X::AbstractMatrix{Float64}, 
                      ud_Y_R_X::AbstractMatrix{Float64},
                             ρ::AbstractMatrix{Float64},
                    u_X_star_X::AbstractMatrix{Float64},
                           γad::AbstractMatrix{Float64},
                         Shock::AbstractMatrix{Int}
                         )

    N_X, N_Y = size( ρ ) # Grid size
    dN_Xp1   = 1:N_X+1
    dN_Y     = 1:N_Y

    @threads for j in dN_Y
        @simd for i in dN_Xp1
            @inbounds begin 
                # Unpack primitive variables for gas & dust:
                ρL   = ρ_L_X[i,j];   ρR  = ρ_R_X[i,j]
                uXL  = u_X_L_X[i,j]; uXR = u_X_R_X[i,j]
                vL   = u_Y_L_X[i,j]; vR  = u_Y_R_X[i,j]   # Transverse velocities
                PL   = P_L_X[i,j];   PR  = P_R_X[i,j]

                ρdL  = ρd_L_X[i,j];   ρdR  = ρd_R_X[i,j]
                udXL = ud_X_L_X[i,j]; udXR = ud_X_R_X[i,j]
                vdL  = ud_Y_L_X[i,j]; vdR  = ud_Y_R_X[i,j]   # Transverse velocities

                # Adiabatic indices:
                γL = γad[clamp(i-1, 1, N_X), j]; γR = γad[clamp(i, 1, N_X), j]

                # Shock indicators for cells i-1 and i:
                Shock_L = Shock[clamp(i-1, 1, N_X), j]
                Shock_R = Shock[clamp(i, 1, N_X), j]

                # Get the 1D flux:
                F_ρ_X[i,j], F_ρu_X_X[i,j], F_ρu_Y_X[i,j], F_E_X[i,j], 
                u_X_star_X[i,j], 
                F_ρd_X[i,j], F_ρdud_X_X[i,j], F_ρdud_Y_X[i,j] = GetFlux( ρL, ρR, uXL, uXR, vL, vR, PL, PR, γL, γR,
                                                                         ρdL, ρdR, udXL, udXR, vdL, vdR,
                                                                         Shock_L, Shock_R )

            end
        end # End of i-loop
    end # End of j-loop

    return nothing
end


""" Compute & update the flux in the Y-direction. Also updates
    the contact speed. """
function ComputeFlux_Y!( F_ρ_Y::AbstractMatrix{Float64}, 
                      F_ρu_X_Y::AbstractMatrix{Float64}, 
                      F_ρu_Y_Y::AbstractMatrix{Float64}, 
                         F_E_Y::AbstractMatrix{Float64}, 
                         ρ_L_Y::AbstractMatrix{Float64}, 
                         ρ_R_Y::AbstractMatrix{Float64},
                       u_Y_L_Y::AbstractMatrix{Float64}, 
                       u_Y_R_Y::AbstractMatrix{Float64},
                       u_X_L_Y::AbstractMatrix{Float64}, 
                       u_X_R_Y::AbstractMatrix{Float64},
                         P_L_Y::AbstractMatrix{Float64}, 
                         P_R_Y::AbstractMatrix{Float64},

                       F_ρd_Y::AbstractMatrix{Float64}, 
                    F_ρdud_X_Y::AbstractMatrix{Float64}, 
                    F_ρdud_Y_Y::AbstractMatrix{Float64}, 
                        ρd_L_Y::AbstractMatrix{Float64}, 
                        ρd_R_Y::AbstractMatrix{Float64},
                      ud_Y_L_Y::AbstractMatrix{Float64}, 
                      ud_Y_R_Y::AbstractMatrix{Float64},
                      ud_X_L_Y::AbstractMatrix{Float64}, 
                      ud_X_R_Y::AbstractMatrix{Float64},
                             ρ::AbstractMatrix{Float64},
                    u_Y_star_Y::AbstractMatrix{Float64},
                           γad::AbstractMatrix{Float64},
                         Shock::AbstractMatrix{Int}
                         )

    N_X, N_Y = size( ρ ) # Grid size
    dN_X     = 1:N_X 
    dN_Yp1   = 1:N_Y+1

    @threads for j in dN_Yp1
        @simd for i in dN_X
            @inbounds begin 
                # Unpack primitive variables for gas & dust:
                ρL   = ρ_L_Y[i,j];   ρR  = ρ_R_Y[i,j]
                uL   = u_Y_L_Y[i,j]; uR  = u_Y_R_Y[i,j]
                vL   = u_X_L_Y[i,j]; vR  = u_X_R_Y[i,j] # Transverse velocities 
                PL   = P_L_Y[i,j];   PR  = P_R_Y[i,j]

                ρdL  = ρd_L_Y[i,j];   ρdR  = ρd_R_Y[i,j]
                udL  = ud_Y_L_Y[i,j]; udR  = ud_Y_R_Y[i,j]
                vdL  = ud_X_L_Y[i,j]; vdR  = ud_X_R_Y[i,j] # Transverse velocities 

                # Adiabatic indices:
                γL = γad[i, clamp(j-1, 1, N_Y)]; γR = γad[i, clamp(j, 1, N_Y)]

                # Shock indicators for cells j-1 and j:
                Shock_L = Shock[i, clamp(j-1, 1, N_Y)]
                Shock_R = Shock[i, clamp(j, 1, N_Y)]

                # Get the 1D flux:
                F_ρ_Y[i,j], F_ρu_Y_Y[i,j], F_ρu_X_Y[i,j], F_E_Y[i,j], 
                u_Y_star_Y[i,j],
                F_ρd_Y[i,j], F_ρdud_Y_Y[i,j], F_ρdud_X_Y[i,j] = GetFlux( ρL, ρR, uL, uR, vL, vR, PL, PR, γL, γR,
                                                                         ρdL, ρdR, udL, udR, vdL, vdR, 
                                                                         Shock_L, Shock_R )

            end
        end # End of i-loop
    end # End of j-loop


    return nothing
end


###############################################################
#######                                                 #######
#######         T I M E  I N T E G R A T I O N          #######
#######                                                 #######
###############################################################

# Inform the reader if the chosen Courant factor is problematic:
if CFL ≥ 1.0
    @warn "\nThe target Courant factor has been set to CFL = $(CFL), which is too high for stability.\n" *
     "You can choose a lower value in the settings of Hydrodynamics.jl.\n"
elseif CFL ≤ 0.0 || !isfinite(CFL) 
    error("\nThe target Courant factor has been set to CFL = $(CFL), which is unphysical.\nChange it to a positive, finite value in" * 
          " the settings of Hydrodynamics.jl.\n")
end



""" Solves the PDE ∂Q/∂t + ∇⋅F = S in 1D over a time-step Δt,
    using explicit time-integration, and the finite-volume 
    method. 
    
    **ARGUMENTS:**
    
    Q       :   Value of Q at "old" time-step n
    Δt      :   Time-step 
    S       :   Source term in cell i 
    inv_ΔV  :   Inverse volume 1 / ΔV of cell i 
    A_L     :   Area of left interface i - 1/2
    A_R     :   Area of right interface i + 1/2
    F_L     :   Flux F at left interface i - 1/2
    F_R     :   Flux F at right interface i + 1/2

    **RETURNS:**

    Qnew    :   Value of Q at time-step n + 1 
"""
@inline function UpdateConservativeQuantity( Q::Float64, 
                                            Δt::Float64,
                                             S::Float64,
                                        inv_ΔV::Float64, 
                                           A_L::Float64,
                                           A_R::Float64,
                                           F_L::Float64,
                                           F_R::Float64
                                             )

    Qnew = Q  +  S * Δt  -  (Δt * inv_ΔV) * ( A_R * F_R  -  A_L * F_L )
    return Qnew
end


""" Update all the conserved quantities (ρ, ρu_X, ρu_Y, E) 
    in the X-direction sweep. Cartesian or cylindrical 
    (X,Z → r,Z) coordinates are assumed depending on the last 
    argument. """
function UpdateConserved_X!( ρ::AbstractMatrix{Float64},
                         ρ_old::AbstractMatrix{Float64},
                          ρu_X::AbstractMatrix{Float64},
                      ρu_X_old::AbstractMatrix{Float64},
                          ρu_Y::AbstractMatrix{Float64},
                      ρu_Y_old::AbstractMatrix{Float64},
                             E::AbstractMatrix{Float64},
                         E_old::AbstractMatrix{Float64},
                         F_ρ_X::AbstractMatrix{Float64}, 
                      F_ρu_X_X::AbstractMatrix{Float64},
                      F_ρu_Y_X::AbstractMatrix{Float64},
                         F_E_X::AbstractMatrix{Float64},

                            ρd::AbstractMatrix{Float64},
                        ρd_old::AbstractMatrix{Float64},
                        ρdud_X::AbstractMatrix{Float64},
                    ρdud_X_old::AbstractMatrix{Float64},
                        ρdud_Y::AbstractMatrix{Float64},
                    ρdud_Y_old::AbstractMatrix{Float64},
                        F_ρd_X::AbstractMatrix{Float64}, 
                    F_ρdud_X_X::AbstractMatrix{Float64},
                    F_ρdud_Y_X::AbstractMatrix{Float64},

                        inv_ΔV::AbstractMatrix{Float64},  # Inverse volume 1 / ΔV
                           A_X::AbstractMatrix{Float64},  # Interface areas in X-direction 
                            Δt::Float64,
                         P_L_X::AbstractMatrix{Float64},  # Needed for geometric source term 
                            Pc::AbstractMatrix{Float64},  # Needed for geometric source term 
                         P_R_X::AbstractMatrix{Float64},  # Needed for geometric source term 
                    inv_r_cell::AbstractVector{Float64},  # ⟨1/r⟩, also only needed for geometric source term
                              ::Val{:Cartesian}
                             )

    N_X, N_Y = size( ρ ) # Grid size
    dN_X     = 1:N_X
    dN_Y     = 1:N_Y
    
    @threads for j in dN_Y

        @simd for i in dN_X
            @inbounds begin 

                # -------- Interface areas -----------------------------
                A_L       = A_X[i,j]    # Left interface area   
                A_R       = A_X[i+1,j]  # Right interface area 

                # -------- Source terms --------------------------------
                # No density source term (for both gas & dust):
                S_ρ       = 0.0
                # No momentum source term in the Y-direction (for both gas & dust):
                S_ρu_Y    = 0.0 
                # No momentum source term in the X-direction (for both gas & dust):  
                S_ρu_X    = 0.0
                # No energy source term:
                S_E       = 0.0

                # ------- Updates --------------------------------------
                # Update density:
                ρ[i,j]      = max( UpdateConservativeQuantity(ρ_old[i,j], Δt, S_ρ, inv_ΔV[i,j], A_L, A_R, F_ρ_X[i,j], F_ρ_X[i+1,j]),
                                   ρ_floor )
                # Update momentum in X-direction:
                ρu_X[i,j]   = UpdateConservativeQuantity(ρu_X_old[i,j], Δt, S_ρu_X, inv_ΔV[i,j], A_L, A_R, F_ρu_X_X[i,j], F_ρu_X_X[i+1,j])
                # Update momentum in Y-direction:
                ρu_Y[i,j]   = UpdateConservativeQuantity(ρu_Y_old[i,j], Δt, S_ρu_Y, inv_ΔV[i,j], A_L, A_R, F_ρu_Y_X[i,j], F_ρu_Y_X[i+1,j])
                # Update energy:
                E[i,j]      = UpdateConservativeQuantity(E_old[i,j], Δt, S_E, inv_ΔV[i,j], A_L, A_R, F_E_X[i,j], F_E_X[i+1,j])

                # Update dust density:
                ρd[i,j]     = max( UpdateConservativeQuantity(ρd_old[i,j], Δt, S_ρ, inv_ΔV[i,j], A_L, A_R, F_ρd_X[i,j], F_ρd_X[i+1,j]),
                                   ρd_floor )
                # Update dust momentum in X-direction:
                ρdud_X[i,j] = UpdateConservativeQuantity(ρdud_X_old[i,j], Δt, S_ρu_X, inv_ΔV[i,j], A_L, A_R, F_ρdud_X_X[i,j], F_ρdud_X_X[i+1,j])
                # Update dust momentum in Y-direction:
                ρdud_Y[i,j] = UpdateConservativeQuantity(ρdud_Y_old[i,j], Δt, S_ρu_Y, inv_ΔV[i,j], A_L, A_R, F_ρdud_Y_X[i,j], F_ρdud_Y_X[i+1,j])

                if DEBUG_Hydro
                    @assert isfinite(ρ[i,j])    "ρ[$(i),$(j)] = $(ρ[i,j]) following the update in UpdateConserved_X!."
                    @assert isfinite(ρu_X[i,j]) "ρu_X[$(i),$(j)] = $(ρu_X[i,j]) following the update in UpdateConserved_X!."
                    @assert isfinite(ρu_Y[i,j]) "ρu_Y[$(i),$(j)] = $(ρu_Y[i,j]) following the update in UpdateConserved_X!."
                    @assert isfinite(E[i,j])    "E[$(i),$(j)] = $(E[i,j]) following the update in UpdateConserved_X!."

                    @assert isfinite(ρd[i,j])     "ρd[$(i),$(j)] = $(ρd[i,j]) following the update in UpdateConserved_X!."
                    @assert isfinite(ρdud_X[i,j]) "ρdud_X[$(i),$(j)] = $(ρdud_X[i,j]) following the update in UpdateConserved_X!."
                    @assert isfinite(ρdud_Y[i,j]) "ρdud_Y[$(i),$(j)] = $(ρdud_Y[i,j]) following the update in UpdateConserved_X!."
                end
            end
        end # End of i-loop 
    end # End of j-loop 

end


# Weight factors for the 3rd-order accurate geometric source term ⟨ P / r ⟩,
# see Mignone (2014), Eq. 65 & 67. 
γ̂_minus::Float64 = 1 / 6  # Weight for P_{i-1/2}
γ̂::Float64       = 2 / 3  # Weight for Pc_{i}    (cell-center value, *not* cell-averaged)
γ̂_plus::Float64  = 1 / 6  # Weight for P_{i+1/2}

""" Update all the conserved quantities (ρ, ρu_X, ρu_Y, E) 
    in the X-direction sweep. Cartesian or cylindrical 
    (X,Z → r,Z) coordinates are assumed depending on the last 
    argument. """
function UpdateConserved_X!( ρ::AbstractMatrix{Float64},
                         ρ_old::AbstractMatrix{Float64},
                          ρu_X::AbstractMatrix{Float64},
                      ρu_X_old::AbstractMatrix{Float64},
                          ρu_Y::AbstractMatrix{Float64},
                      ρu_Y_old::AbstractMatrix{Float64},
                             E::AbstractMatrix{Float64},
                         E_old::AbstractMatrix{Float64},
                         F_ρ_X::AbstractMatrix{Float64}, 
                      F_ρu_X_X::AbstractMatrix{Float64},
                      F_ρu_Y_X::AbstractMatrix{Float64},
                         F_E_X::AbstractMatrix{Float64},

                            ρd::AbstractMatrix{Float64},
                        ρd_old::AbstractMatrix{Float64},
                        ρdud_X::AbstractMatrix{Float64},
                    ρdud_X_old::AbstractMatrix{Float64},
                        ρdud_Y::AbstractMatrix{Float64},
                    ρdud_Y_old::AbstractMatrix{Float64},
                        F_ρd_X::AbstractMatrix{Float64}, 
                    F_ρdud_X_X::AbstractMatrix{Float64},
                    F_ρdud_Y_X::AbstractMatrix{Float64},

                        inv_ΔV::AbstractMatrix{Float64},  # Inverse volume 1 / ΔV
                           A_X::AbstractMatrix{Float64},  # Interface areas in X-direction 
                            Δt::Float64,
                         P_L_X::AbstractMatrix{Float64},  # Needed for geometric source term 
                            Pc::AbstractMatrix{Float64},  # Needed for geometric source term 
                         P_R_X::AbstractMatrix{Float64},  # Needed for geometric source term 
                    inv_r_cell::AbstractVector{Float64},  # ⟨1/r⟩, also needed for geometric source term
                              ::Val{:Cylindrical}
                             )

    N_X, N_Y = size( ρ ) # Grid size
    dN_X     = 1:N_X
    dN_Y     = 1:N_Y
    
    @threads for j in dN_Y

        @simd for i in dN_X
            @inbounds begin 

                # -------- Interface areas -----------------------------
                A_L       = A_X[i,j]    # Left interface area   
                A_R       = A_X[i+1,j]  # Right interface area 

                # -------- Source terms --------------------------------
                # No density source term:
                S_ρ       = 0.0
                # No momentum source term in the Y-direction:
                S_ρu_Y    = 0.0 
                # Momentum source term in the X-direction (i.e. r-direction) in cylindrical coordinates: 
                S_ρu_X    = ( γ̂_minus * P_L_X[i+1,j]  +  γ̂ * Pc[i,j]  +  γ̂_plus * P_R_X[i,j] ) * inv_r_cell[i]
                S_ρdud_X  = 0.0  # No dust pressure, hence no geometric source term
                # No energy source term:
                S_E       = 0.0

                # ------- Updates --------------------------------------
                # Update density:
                ρ[i,j]      = max( UpdateConservativeQuantity(ρ_old[i,j], Δt, S_ρ, inv_ΔV[i,j], A_L, A_R, F_ρ_X[i,j], F_ρ_X[i+1,j]),
                                   ρ_floor )
                # Update momentum in X-direction:
                ρu_X[i,j]   = UpdateConservativeQuantity(ρu_X_old[i,j], Δt, S_ρu_X, inv_ΔV[i,j], A_L, A_R, F_ρu_X_X[i,j], F_ρu_X_X[i+1,j])
                # Update momentum in Y-direction:
                ρu_Y[i,j]   = UpdateConservativeQuantity(ρu_Y_old[i,j], Δt, S_ρu_Y, inv_ΔV[i,j], A_L, A_R, F_ρu_Y_X[i,j], F_ρu_Y_X[i+1,j])
                # Update energy:
                E[i,j]      = UpdateConservativeQuantity(E_old[i,j], Δt, S_E, inv_ΔV[i,j], A_L, A_R, F_E_X[i,j], F_E_X[i+1,j])

                # Update dust density:
                ρd[i,j]     = max( UpdateConservativeQuantity(ρd_old[i,j], Δt, S_ρ, inv_ΔV[i,j], A_L, A_R, F_ρd_X[i,j], F_ρd_X[i+1,j]),
                                   ρd_floor )
                # Update dust momentum in X-direction:
                ρdud_X[i,j] = UpdateConservativeQuantity(ρdud_X_old[i,j], Δt, S_ρdud_X, inv_ΔV[i,j], A_L, A_R, F_ρdud_X_X[i,j], F_ρdud_X_X[i+1,j])
                # Update dust momentum in Y-direction:
                ρdud_Y[i,j] = UpdateConservativeQuantity(ρdud_Y_old[i,j], Δt, S_ρu_Y, inv_ΔV[i,j], A_L, A_R, F_ρdud_Y_X[i,j], F_ρdud_Y_X[i+1,j])


                if DEBUG_Hydro
                    @assert isfinite(ρ[i,j])    "ρ[$(i),$(j)] = $(ρ[i,j]) following the update in UpdateConserved_X!."
                    @assert isfinite(ρu_X[i,j]) "ρu_X[$(i),$(j)] = $(ρu_X[i,j]) following the update in UpdateConserved_X!."
                    @assert isfinite(ρu_Y[i,j]) "ρu_Y[$(i),$(j)] = $(ρu_Y[i,j]) following the update in UpdateConserved_X!."
                    @assert isfinite(E[i,j])    "E[$(i),$(j)] = $(E[i,j]) following the update in UpdateConserved_X!."

                    @assert isfinite(ρd[i,j])     "ρd[$(i),$(j)] = $(ρd[i,j]) following the update in UpdateConserved_X!."
                    @assert isfinite(ρdud_X[i,j]) "ρdud_X[$(i),$(j)] = $(ρdud_X[i,j]) following the update in UpdateConserved_X!."
                    @assert isfinite(ρdud_Y[i,j]) "ρdud_Y[$(i),$(j)] = $(ρdud_Y[i,j]) following the update in UpdateConserved_X!."
                end
            end
        end # End of i-loop 
    end # End of j-loop 

end


""" Update all the conserved quantities (ρ, ρu_X, ρu_Y, E) 
    in the Z-direction sweep. """
function UpdateConserved_Y!( ρ::AbstractMatrix{Float64},
                         ρ_old::AbstractMatrix{Float64},
                          ρu_X::AbstractMatrix{Float64},
                      ρu_X_old::AbstractMatrix{Float64},
                          ρu_Y::AbstractMatrix{Float64},
                      ρu_Y_old::AbstractMatrix{Float64},
                             E::AbstractMatrix{Float64},
                         E_old::AbstractMatrix{Float64},
                         F_ρ_Y::AbstractMatrix{Float64}, 
                      F_ρu_X_Y::AbstractMatrix{Float64},
                      F_ρu_Y_Y::AbstractMatrix{Float64},
                         F_E_Y::AbstractMatrix{Float64},

                            ρd::AbstractMatrix{Float64},
                        ρd_old::AbstractMatrix{Float64},
                        ρdud_X::AbstractMatrix{Float64},
                    ρdud_X_old::AbstractMatrix{Float64},
                        ρdud_Y::AbstractMatrix{Float64},
                    ρdud_Y_old::AbstractMatrix{Float64},
                        F_ρd_Y::AbstractMatrix{Float64}, 
                    F_ρdud_X_Y::AbstractMatrix{Float64},
                    F_ρdud_Y_Y::AbstractMatrix{Float64},

                        inv_ΔV::AbstractMatrix{Float64},  # Inverse volume 1 / ΔV
                           A_Y::AbstractMatrix{Float64},  # Interface areas in Y-direction 
                            Δt::Float64
                             )

    N_X, N_Y = size( ρ ) # Grid size
    dN_X     = 1:N_X
    dN_Y     = 1:N_Y
    
    @threads for j in dN_Y
        @simd for i in dN_X
            @inbounds begin 
                # -------- Interface areas -----------------------------
                A_L       = A_Y[i,j]    # Left interface area   
                A_R       = A_Y[i,j+1]  # Right interface area 

                # -------- Source terms --------------------------------
                # No density source term (for gas and dust):
                S_ρ       = 0.0
                # No momentum source term in the Y-direction (for gas and dust):
                S_ρu_Y    = 0.0 
                # No momentum source term in the X-direction (for gas and dust):  
                S_ρu_X    = 0.0
                # No energy source term:
                S_E       = 0.0
                
                # ------- Updates --------------------------------------
                # Update density:
                ρ[i,j]     = max( UpdateConservativeQuantity(ρ_old[i,j], Δt, S_ρ, inv_ΔV[i,j], A_L, A_R, F_ρ_Y[i,j], F_ρ_Y[i,j+1]),
                                  ρ_floor )
                # Update momentum in X-direction:
                ρu_X[i,j]  = UpdateConservativeQuantity(ρu_X_old[i,j], Δt, S_ρu_X, inv_ΔV[i,j], A_L, A_R, F_ρu_X_Y[i,j], F_ρu_X_Y[i,j+1])
                # Update momentum in Y-direction:
                ρu_Y[i,j]  = UpdateConservativeQuantity(ρu_Y_old[i,j], Δt, S_ρu_Y, inv_ΔV[i,j], A_L, A_R, F_ρu_Y_Y[i,j], F_ρu_Y_Y[i,j+1])
                # Update energy:
                E[i,j]     = UpdateConservativeQuantity(E_old[i,j], Δt, S_E, inv_ΔV[i,j], A_L, A_R, F_E_Y[i,j], F_E_Y[i,j+1])

                # Update dust density:
                ρd[i,j]     = max( UpdateConservativeQuantity(ρd_old[i,j], Δt, S_ρ, inv_ΔV[i,j], A_L, A_R, F_ρd_Y[i,j], F_ρd_Y[i,j+1]),
                                   ρd_floor )
                # Update dust momentum in X-direction:
                ρdud_X[i,j] = UpdateConservativeQuantity(ρdud_X_old[i,j], Δt, S_ρu_X, inv_ΔV[i,j], A_L, A_R, F_ρdud_X_Y[i,j], F_ρdud_X_Y[i,j+1])
                # Update dust momentum in Y-direction:
                ρdud_Y[i,j] = UpdateConservativeQuantity(ρdud_Y_old[i,j], Δt, S_ρu_Y, inv_ΔV[i,j], A_L, A_R, F_ρdud_Y_Y[i,j], F_ρdud_Y_Y[i,j+1])


                if DEBUG_Hydro
                    @assert isfinite(ρ[i,j])    "ρ[$(i),$(j)] = $(ρ[i,j]) following the update in UpdateConserved_X!."
                    @assert isfinite(ρu_X[i,j]) "ρu_X[$(i),$(j)] = $(ρu_X[i,j]) following the update in UpdateConserved_X!."
                    @assert isfinite(ρu_Y[i,j]) "ρu_Y[$(i),$(j)] = $(ρu_Y[i,j]) following the update in UpdateConserved_X!."
                    @assert isfinite(E[i,j])    "E[$(i),$(j)] = $(E[i,j]) following the update in UpdateConserved_X!."

                    @assert isfinite(ρd[i,j])     "ρd[$(i),$(j)] = $(ρd[i,j]) following the update in UpdateConserved_X!."
                    @assert isfinite(ρdud_X[i,j]) "ρdud_X[$(i),$(j)] = $(ρdud_X[i,j]) following the update in UpdateConserved_X!."
                    @assert isfinite(ρdud_Y[i,j]) "ρdud_Y[$(i),$(j)] = $(ρdud_Y[i,j]) following the update in UpdateConserved_X!."
                end
            end
        end # End of i-loop 
    end # End of j-loop 

end


const CoordSystem = Union{Val{:Cartesian}, Val{:Cylindrical}}

function ForwardEuler_X!( ρ_new::AbstractMatrix{Float64},
                       ρu_X_new::AbstractMatrix{Float64},
                       ρu_Y_new::AbstractMatrix{Float64},
                          E_new::AbstractMatrix{Float64},
                              ρ::AbstractMatrix{Float64},
                           ρu_X::AbstractMatrix{Float64},
                           ρu_Y::AbstractMatrix{Float64},
                              E::AbstractMatrix{Float64},
                            γad::AbstractMatrix{Float64},
                            u_X::AbstractMatrix{Float64},
                            u_Y::AbstractMatrix{Float64},

                         ρd_new::AbstractMatrix{Float64},
                     ρdud_X_new::AbstractMatrix{Float64},
                     ρdud_Y_new::AbstractMatrix{Float64},
                             ρd::AbstractMatrix{Float64},
                         ρdud_X::AbstractMatrix{Float64},
                         ρdud_Y::AbstractMatrix{Float64},
                           ud_X::AbstractMatrix{Float64},
                           ud_Y::AbstractMatrix{Float64},

                         inv_ΔV::AbstractMatrix{Float64},
                            A_X::AbstractMatrix{Float64},
                             Δt::Float64,

                        # Conservative & primitive variables @ cell centers:
                             ρc::AbstractMatrix{Float64}, 
                          ρu_Xc::AbstractMatrix{Float64}, 
                          ρu_Yc::AbstractMatrix{Float64},
                             Ec::AbstractMatrix{Float64},         
                           u_Xc::AbstractMatrix{Float64}, 
                           u_Yc::AbstractMatrix{Float64}, 
                             Pc::AbstractMatrix{Float64},  
                             
                            ρdc::AbstractMatrix{Float64}, 
                        ρdud_Xc::AbstractMatrix{Float64}, 
                        ρdud_Yc::AbstractMatrix{Float64},  
                          ud_Xc::AbstractMatrix{Float64}, 
                          ud_Yc::AbstractMatrix{Float64}, 

                        # Weights to convert between cell-averages and cell-centers: 
                       w_i_m1_X::AbstractVector{Float64}, 
                        w_i_0_X::AbstractVector{Float64},
                       w_i_p1_X::AbstractVector{Float64},  
                       q_i_m1_X::AbstractVector{Float64}, 
                        q_i_0_X::AbstractVector{Float64}, 
                       q_i_p1_X::AbstractVector{Float64},  

                        # Weights for WENO3 interface reconstruction:
                    w_plus_i_p1::AbstractVector{Float64},  # WENO3 weights w⁺_{i,1}
                   w_minus_i_p1::AbstractVector{Float64},  # WENO3 weights w⁻_{i,1} 

                          ρ_L_X::AbstractMatrix{Float64},
                          ρ_R_X::AbstractMatrix{Float64},
                        u_X_L_X::AbstractMatrix{Float64},
                        u_X_R_X::AbstractMatrix{Float64},
                        u_Y_L_X::AbstractMatrix{Float64},
                        u_Y_R_X::AbstractMatrix{Float64},
                          P_L_X::AbstractMatrix{Float64},
                          P_R_X::AbstractMatrix{Float64},
                          F_ρ_X::AbstractMatrix{Float64},
                       F_ρu_X_X::AbstractMatrix{Float64},
                       F_ρu_Y_X::AbstractMatrix{Float64},
                          F_E_X::AbstractMatrix{Float64},
                              P::AbstractMatrix{Float64},
                     u_X_star_X::AbstractMatrix{Float64},

                         ρd_L_X::AbstractMatrix{Float64},
                         ρd_R_X::AbstractMatrix{Float64},
                       ud_X_L_X::AbstractMatrix{Float64},
                       ud_X_R_X::AbstractMatrix{Float64},
                       ud_Y_L_X::AbstractMatrix{Float64},
                       ud_Y_R_X::AbstractMatrix{Float64},
                         F_ρd_X::AbstractMatrix{Float64},
                     F_ρdud_X_X::AbstractMatrix{Float64},
                     F_ρdud_Y_X::AbstractMatrix{Float64},
                                       
                        # Ghost‐padded primitives & grid:
                         ρg_old::AbstractMatrix{Float64}, 
                       u_Xg_old::AbstractMatrix{Float64}, 
                       u_Yg_old::AbstractMatrix{Float64}, 
                         Pg_old::AbstractMatrix{Float64},

                        ρdg_old::AbstractMatrix{Float64}, 
                      ud_Xg_old::AbstractMatrix{Float64}, 
                      ud_Yg_old::AbstractMatrix{Float64}, 

                             Xg::AbstractVector{Float64},
                          XCoMg::AbstractVector{Float64},

                       CoordSys::CoordSystem,
                     inv_r_cell::AbstractVector{Float64},
  
                        # Shock indicator:
                          Shock::AbstractMatrix{Int}, 
 
                        # BCs:
                      BC_X_left::Symbol, BC_X_right::Symbol,
                      BC_Y_left::Symbol, BC_Y_right::Symbol
                          )

    # (a) Get cell-center conservatives (in X-direction) from 
    # cell-averaged conservatives:
    RecoverPointwiseFromAverages_X!( ρc, ρu_Xc, ρu_Yc, Ec,      # Cell-center values (updated)
                                     ρdc, ρdud_Xc, ρdud_Yc,
                                     ρ,  ρu_X,  ρu_Y,  E, γad,  # Cell-averaged values 
                                     ρd, ρdud_X,  ρdud_Y,
                                     w_i_m1_X, w_i_0_X, w_i_p1_X,
                                     Shock,
                                     BC_X_left, BC_X_right )

    # (b) Get cell-center primitives (in X-direction) from 
    # cell-center conservatives:
    RecoverPointwisePrimitivesFromConservatives!( ρc, ρu_Xc, ρu_Yc, Ec,
                                                  ρdc, ρdud_Xc, ρdud_Yc,
                                                  u_Xc, u_Yc, Pc, γad,
                                                  ud_Xc, ud_Yc )

    # (c) Get cell-averaged primitives (in X-direction) from cell-center primitives:
    RecoverCellAveragedPrimitives_X!( u_Xc, u_Yc, Pc,    # Cell-centered primitives 
                                      ud_Xc, ud_Yc,
                                      u_X,  u_Y,  P,     # Cell-averaged primitives (updated)
                                      ud_X, ud_Y,
                                      q_i_m1_X, q_i_0_X, q_i_p1_X,
                                      Shock,
                                      BC_X_left, BC_X_right )

    # Update shock indicators for the Riemann solver:
    UpdateShockFinder!( Shock, P )

    # (d) Fill ghost-arrays with values from the updated 
    # cell-averaged primitives: 
    @views @inbounds begin 
        ρg_old[ng+1:end-ng, ng+1:end-ng]    .= ρ
        u_Xg_old[ng+1:end-ng, ng+1:end-ng]  .= u_X
        u_Yg_old[ng+1:end-ng, ng+1:end-ng]  .= u_Y
        Pg_old[ng+1:end-ng, ng+1:end-ng]    .= P

        ρdg_old[ng+1:end-ng, ng+1:end-ng]   .= ρd
        ud_Xg_old[ng+1:end-ng, ng+1:end-ng] .= ud_X
        ud_Yg_old[ng+1:end-ng, ng+1:end-ng] .= ud_Y
    end
    FillGhosts!( ρg_old, u_Xg_old, u_Yg_old, Pg_old,
                 ρdg_old, ud_Xg_old, ud_Yg_old, 
                 BC_X_left, BC_X_right, 
                 BC_Y_left, BC_Y_right )

    # (e) Reconstruct X-interfaces:
    UpdateInterfaceValues_X!( ρ_L_X, ρ_R_X,  P_L_X, P_R_X,  u_X_L_X, u_X_R_X,  u_Y_L_X, u_Y_R_X,
                              ρd_L_X, ρd_R_X,  ud_X_L_X, ud_X_R_X,  ud_Y_L_X, ud_Y_R_X,
                              ρg_old, Pg_old, u_Xg_old, u_Yg_old, 
                              ρdg_old, ud_Xg_old, ud_Yg_old, 
                              Xg, XCoMg, Shock, w_plus_i_p1, w_minus_i_p1 )

    # (f) Get X-direction fluxes:
    ComputeFlux_X!( F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,
                    ρ_L_X, ρ_R_X, u_X_L_X, u_X_R_X,
                    u_Y_L_X, u_Y_R_X, P_L_X, P_R_X,

                    F_ρd_X, F_ρdud_X_X, F_ρdud_Y_X, 
                    ρd_L_X, ρd_R_X, ud_X_L_X, ud_X_R_X,
                    ud_Y_L_X, ud_Y_R_X,

                    ρ, u_X_star_X, γad, Shock )

    # (g) Update over time-step Δt (write into *_new):
    UpdateConserved_X!( ρ_new, ρ, ρu_X_new, ρu_X, ρu_Y_new, ρu_Y,
                        E_new, E,
                        F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,

                        ρd_new, ρd, ρdud_X_new, ρdud_X, ρdud_Y_new, ρdud_Y,
                        F_ρd_X, F_ρdud_X_X, F_ρdud_Y_X,

                        inv_ΔV, A_X, Δt,
                        P_L_X, Pc, P_R_X, inv_r_cell, CoordSys )

    # (h) Update old variables:
    ρ      .= ρ_new 
    ρu_X   .= ρu_X_new
    ρu_Y   .= ρu_Y_new
    E      .= E_new

    ρd     .= ρd_new 
    ρdud_X .= ρdud_X_new
    ρdud_Y .= ρdud_Y_new

    return nothing 
end


function ForwardEuler_Y!( ρ_new::AbstractMatrix{Float64},
                       ρu_X_new::AbstractMatrix{Float64},
                       ρu_Y_new::AbstractMatrix{Float64},
                          E_new::AbstractMatrix{Float64},
                              ρ::AbstractMatrix{Float64},
                           ρu_X::AbstractMatrix{Float64},
                           ρu_Y::AbstractMatrix{Float64},
                              E::AbstractMatrix{Float64},
                            γad::AbstractMatrix{Float64},
                            u_X::AbstractMatrix{Float64},
                            u_Y::AbstractMatrix{Float64},

                         ρd_new::AbstractMatrix{Float64},
                     ρdud_X_new::AbstractMatrix{Float64},
                     ρdud_Y_new::AbstractMatrix{Float64},
                             ρd::AbstractMatrix{Float64},
                         ρdud_X::AbstractMatrix{Float64},
                         ρdud_Y::AbstractMatrix{Float64},
                           ud_X::AbstractMatrix{Float64},
                           ud_Y::AbstractMatrix{Float64},

                         inv_ΔV::AbstractMatrix{Float64},
                            A_Y::AbstractMatrix{Float64},
                             Δt::Float64,

                            # Conservative & primitive variables @ cell centers:
                             ρc::AbstractMatrix{Float64}, 
                          ρu_Xc::AbstractMatrix{Float64}, 
                          ρu_Yc::AbstractMatrix{Float64},
                             Ec::AbstractMatrix{Float64},         
                           u_Xc::AbstractMatrix{Float64}, 
                           u_Yc::AbstractMatrix{Float64}, 
                             Pc::AbstractMatrix{Float64},    
                             
                            ρdc::AbstractMatrix{Float64}, 
                        ρdud_Xc::AbstractMatrix{Float64}, 
                        ρdud_Yc::AbstractMatrix{Float64},   
                          ud_Xc::AbstractMatrix{Float64}, 
                          ud_Yc::AbstractMatrix{Float64}, 

                            # Weights to convert between cell-averages and cell-centers: 
                       w_j_m1_Y::AbstractVector{Float64}, 
                        w_j_0_Y::AbstractVector{Float64}, 
                       w_j_p1_Y::AbstractVector{Float64},  
                       q_j_m1_Y::AbstractVector{Float64}, 
                        q_j_0_Y::AbstractVector{Float64}, 
                       q_j_p1_Y::AbstractVector{Float64},  

                            # Weights for WENO3 interface reconstruction:
                    w_plus_j_p1::AbstractVector{Float64},  # WENO3 weights w⁺_{j,1}
                   w_minus_j_p1::AbstractVector{Float64},  # WENO3 weights w⁻_{j,1}

                          ρ_L_Y::AbstractMatrix{Float64},
                          ρ_R_Y::AbstractMatrix{Float64},
                        u_Y_L_Y::AbstractMatrix{Float64},
                        u_Y_R_Y::AbstractMatrix{Float64},
                        u_X_L_Y::AbstractMatrix{Float64},
                        u_X_R_Y::AbstractMatrix{Float64},
                          P_L_Y::AbstractMatrix{Float64},
                          P_R_Y::AbstractMatrix{Float64},
                          F_ρ_Y::AbstractMatrix{Float64},
                       F_ρu_X_Y::AbstractMatrix{Float64},
                       F_ρu_Y_Y::AbstractMatrix{Float64},
                          F_E_Y::AbstractMatrix{Float64},
                              P::AbstractMatrix{Float64},
                     u_Y_star_Y::AbstractMatrix{Float64},

                         ρd_L_Y::AbstractMatrix{Float64},
                         ρd_R_Y::AbstractMatrix{Float64},
                       ud_Y_L_Y::AbstractMatrix{Float64},
                       ud_Y_R_Y::AbstractMatrix{Float64},
                       ud_X_L_Y::AbstractMatrix{Float64},
                       ud_X_R_Y::AbstractMatrix{Float64},
                         F_ρd_Y::AbstractMatrix{Float64},
                     F_ρdud_X_Y::AbstractMatrix{Float64},
                     F_ρdud_Y_Y::AbstractMatrix{Float64},
                                       
                            # Ghost‐padded primitives & grid:
                         ρg_old::AbstractMatrix{Float64}, 
                       u_Xg_old::AbstractMatrix{Float64}, 
                       u_Yg_old::AbstractMatrix{Float64}, 
                         Pg_old::AbstractMatrix{Float64},

                        ρdg_old::AbstractMatrix{Float64}, 
                      ud_Xg_old::AbstractMatrix{Float64}, 
                      ud_Yg_old::AbstractMatrix{Float64}, 

                             Yg::AbstractVector{Float64},
                          YCoMg::AbstractVector{Float64},

  
                            # Shock indicator:
                          Shock::AbstractMatrix{Int}, 
 
                            # BCs:
                      BC_X_left::Symbol, BC_X_right::Symbol,
                      BC_Y_left::Symbol, BC_Y_right::Symbol
                       )

    # (a) Get cell-center conservatives (in Y-direction) from 
    # cell-averaged conservatives:
    RecoverPointwiseFromAverages_Y!( ρc, ρu_Xc, ρu_Yc, Ec,      # Cell-center values (updated)
                                     ρdc, ρdud_Xc, ρdud_Yc,
                                     ρ,  ρu_X,  ρu_Y,  E, γad,  # Cell-averaged values 
                                     ρd,  ρdud_X,  ρdud_Y,
                                     w_j_m1_Y, w_j_0_Y, w_j_p1_Y,
                                     Shock,
                                     BC_Y_left, BC_Y_right )

    # (b) Get cell-center primitives (in Y-direction) from 
    # cell-center conservatives:
    RecoverPointwisePrimitivesFromConservatives!( ρc, ρu_Xc, ρu_Yc, Ec,
                                                  ρdc, ρdud_Xc, ρdud_Yc,
                                                  u_Xc, u_Yc, Pc, γad,
                                                  ud_Xc, ud_Yc )

    # (c) Get cell-averaged primitives (in Y-direction) from cell-center primitives:
    RecoverCellAveragedPrimitives_Y!( u_Xc, u_Yc, Pc,    # Cell-centered primitives 
                                      ud_Xc, ud_Yc,
                                      u_X,  u_Y,  P,     # Cell-averaged primitives (updated)
                                      ud_X,  ud_Y,
                                      q_j_m1_Y, q_j_0_Y, q_j_p1_Y,
                                      Shock,
                                      BC_Y_left, BC_Y_right )

    # Update shock indicators for the Riemann solver:
    UpdateShockFinder!( Shock, P )

    # (d) Fill ghost-arrays with values from the updated 
    # cell-averaged primitives: 
    @views @inbounds begin 
        ρg_old[ng+1:end-ng, ng+1:end-ng]    .= ρ
        u_Xg_old[ng+1:end-ng, ng+1:end-ng]  .= u_X
        u_Yg_old[ng+1:end-ng, ng+1:end-ng]  .= u_Y
        Pg_old[ng+1:end-ng, ng+1:end-ng]    .= P

        ρdg_old[ng+1:end-ng, ng+1:end-ng]   .= ρd
        ud_Xg_old[ng+1:end-ng, ng+1:end-ng] .= ud_X
        ud_Yg_old[ng+1:end-ng, ng+1:end-ng] .= ud_Y
    end
    FillGhosts!( ρg_old, u_Xg_old, u_Yg_old, Pg_old,
                 ρdg_old, ud_Xg_old, ud_Yg_old,
                 BC_X_left, BC_X_right, 
                 BC_Y_left, BC_Y_right )

    # (e) Reconstruct Y-interfaces:
    UpdateInterfaceValues_Y!( ρ_L_Y, ρ_R_Y,  P_L_Y, P_R_Y,  u_X_L_Y, u_X_R_Y,  u_Y_L_Y, u_Y_R_Y,
                              ρd_L_Y, ρd_R_Y,  ud_X_L_Y, ud_X_R_Y,  ud_Y_L_Y, ud_Y_R_Y, 
                              ρg_old, Pg_old, u_Xg_old, u_Yg_old,
                              ρdg_old, ud_Xg_old, ud_Yg_old,
                              Yg, YCoMg, Shock, w_plus_j_p1, w_minus_j_p1 )

    # (f) Get Y-direction fluxes:
    ComputeFlux_Y!( F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,
                    ρ_L_Y, ρ_R_Y, u_Y_L_Y, u_Y_R_Y,
                    u_X_L_Y, u_X_R_Y, P_L_Y, P_R_Y,

                    F_ρd_Y, F_ρdud_X_Y, F_ρdud_Y_Y, 
                    ρd_L_Y, ρd_R_Y, ud_Y_L_Y, ud_Y_R_Y,
                    ud_X_L_Y, ud_X_R_Y, 

                    ρ, u_Y_star_Y, γad, Shock )

    # (g) Update over time-step Δt (write into *_new):
    UpdateConserved_Y!( ρ_new, ρ, ρu_X_new, ρu_X,
                        ρu_Y_new, ρu_Y, E_new, E,
                        F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,

                        ρd_new, ρd, ρdud_X_new, ρdud_X,
                        ρdud_Y_new, ρdud_Y, 
                        F_ρd_Y, F_ρdud_X_Y, F_ρdud_Y_Y,

                        inv_ΔV, A_Y, Δt )

    # (h) Update old variables:
    ρ      .= ρ_new
    ρu_X   .= ρu_X_new
    ρu_Y   .= ρu_Y_new
    E      .= E_new

    ρd     .= ρd_new
    ρdud_X .= ρdud_X_new
    ρdud_Y .= ρdud_Y_new

    return nothing 
end


""" This function performs a forward Euler update in the X-direction for all 
    passive scalars. """
function ForwardEuler_X_Scalars!( Scalars::AbstractArray{Float64, 3},  # Array of passive scalars, for all cells 
                              Scalars_old::AbstractArray{Float64, 3},  # Array of old passive scalars, for all cells 
                                    ρ_old::AbstractMatrix{Float64},    # Gas density *before* forward Euler update 
                                    F_ρ_X::AbstractMatrix{Float64},
                                   inv_ΔV::AbstractMatrix{Float64},
                                      A_X::AbstractMatrix{Float64},
                                       Δt::Float64
                                 )

    N_X, N_Y, N_Scalars = size( Scalars )
    dN_X                = 1:N_X 
    dN_Y                = 1:N_Y 
    dN_Scalars          = 1:N_Scalars

    for k in dN_Scalars 
        @threads for j in dN_Y
            @inbounds @simd for i in dN_X 
                # Clamp indices that enter x, which implicitly assumes 
                # zero gradient extrapolation BC's:
                ip1 = min( i + 1, N_X )
                im1 = max( i - 1, 1 )        
                
                # Reconstruct x on interfaces, using a simple first-order method: 
                if F_ρ_X[i,j] ≥ 0.0
                    xL = Scalars_old[im1,j,k] / ρ_old[im1,j]  # Flow left → right
                else
                    xL = Scalars_old[i,j,k] / ρ_old[i,j]      # Flow right → left
                end

                if F_ρ_X[i+1,j] ≥ 0.0
                    xR = Scalars_old[i,j,k] / ρ_old[i,j]      # Flow left → right
                else
                    xR = Scalars_old[ip1,j,k] / ρ_old[ip1,j]  # Flow right → left
                end

                # Get left (i-1/2) & right (i+1/2) interface fluxes:
                F_L_X   = F_ρ_X[i,j]   * xL
                F_R_X   = F_ρ_X[i+1,j] * xR

                # Get interface areas:
                A_R     = A_X[i+1,j]
                A_L     = A_X[i,j]

                # Advect scalar ρx over a time-step Δt:
                Scalars[i,j,k] = Scalars_old[i,j,k] - ( Δt * inv_ΔV[i,j] ) * ( A_R * F_R_X  -  A_L * F_L_X )

                # Floor scalars, since they are all supposed to be positive:
                Scalars[i,j,k] = max( Scalars[i,j,k], 0.0 )

            end # End of i-loop 
        end # End of j-loop 
    end # End of loop over passive scalars 

    return nothing 
end


""" This function performs a forward Euler update in the Y-direction for all 
    passive scalars. """
function ForwardEuler_Y_Scalars!( Scalars::AbstractArray{Float64, 3},  # Array of passive scalars, for all cells 
                              Scalars_old::AbstractArray{Float64, 3},  # Array of old passive scalars, for all cells 
                                    ρ_old::AbstractMatrix{Float64},    # Gas density *before* forward Euler update 
                                    F_ρ_Y::AbstractMatrix{Float64},
                                   inv_ΔV::AbstractMatrix{Float64},
                                      A_Y::AbstractMatrix{Float64},
                                       Δt::Float64
                                 )

    N_X, N_Y, N_Scalars = size( Scalars )
    dN_X                = 1:N_X 
    dN_Y                = 1:N_Y 
    dN_Scalars          = 1:N_Scalars

    for k in dN_Scalars 
        @threads for j in dN_Y
            @inbounds @simd for i in dN_X 
                # Clamp indices that enter x, which implicitly assumes 
                # zero gradient extrapolation BC's:
                jp1 = min( j + 1, N_Y )
                jm1 = max( j - 1, 1 )         
                
                # Reconstruct x on interfaces, using a simple first-order method: 
                if F_ρ_Y[i,j] ≥ 0.0
                    xL = Scalars_old[i,jm1,k] / ρ_old[i,jm1]  # Flow left → right
                else
                    xL = Scalars_old[i,j,k] / ρ_old[i,j]      # Flow right → left
                end

                if F_ρ_Y[i,j+1] ≥ 0.0
                    xR = Scalars_old[i,j,k] / ρ_old[i,j]      # Flow left → right
                else
                    xR = Scalars_old[i,jp1,k] / ρ_old[i,jp1]  # Flow right → left
                end

                # Get left (j-1/2) & right (j+1/2) interface fluxes:
                F_L_Y   = F_ρ_Y[i,j]   * xL
                F_R_Y   = F_ρ_Y[i,j+1] * xR

                # Get interface areas:
                A_R     = A_Y[i,j+1]
                A_L     = A_Y[i,j]

                # Advect scalar ρx over a time-step Δt:
                Scalars[i,j,k] = Scalars_old[i,j,k] - ( Δt * inv_ΔV[i,j] ) * ( A_R * F_R_Y  -  A_L * F_L_Y )

                # Floor scalars, since they are all supposed to be positive:
                Scalars[i,j,k] = max( Scalars[i,j,k], 0.0 )

            end # End of i-loop 
        end # End of j-loop 
    end # End of loop over passive scalars 

    return nothing 
end


function RK2_X!( ρ_new::AbstractMatrix{Float64},
              ρu_X_new::AbstractMatrix{Float64},
              ρu_Y_new::AbstractMatrix{Float64},
                 E_new::AbstractMatrix{Float64},
                     ρ::AbstractMatrix{Float64},
                  ρu_X::AbstractMatrix{Float64},
                  ρu_Y::AbstractMatrix{Float64},
                     E::AbstractMatrix{Float64},
                   γad::AbstractMatrix{Float64},
                   u_X::AbstractMatrix{Float64},
                   u_Y::AbstractMatrix{Float64},

                ρd_new::AbstractMatrix{Float64},
            ρdud_X_new::AbstractMatrix{Float64},
            ρdud_Y_new::AbstractMatrix{Float64},
                    ρd::AbstractMatrix{Float64},
                ρdud_X::AbstractMatrix{Float64},
                ρdud_Y::AbstractMatrix{Float64},
                  ud_X::AbstractMatrix{Float64},
                  ud_Y::AbstractMatrix{Float64},

                inv_ΔV::AbstractMatrix{Float64},
                   A_X::AbstractMatrix{Float64},
                    Δt::Float64,

                # Scratch arrays to store state at the start:
               ρ_stage::AbstractMatrix{Float64},
            ρu_X_stage::AbstractMatrix{Float64},
            ρu_Y_stage::AbstractMatrix{Float64},
               E_stage::AbstractMatrix{Float64},

              ρd_stage::AbstractMatrix{Float64},
          ρdud_X_stage::AbstractMatrix{Float64},
          ρdud_Y_stage::AbstractMatrix{Float64},

                # Conservative & primitive variables @ cell centers:
                    ρc::AbstractMatrix{Float64}, 
                 ρu_Xc::AbstractMatrix{Float64}, 
                 ρu_Yc::AbstractMatrix{Float64},
                    Ec::AbstractMatrix{Float64},         
                  u_Xc::AbstractMatrix{Float64}, 
                  u_Yc::AbstractMatrix{Float64}, 
                    Pc::AbstractMatrix{Float64},  
                    
                   ρdc::AbstractMatrix{Float64}, 
               ρdud_Xc::AbstractMatrix{Float64}, 
               ρdud_Yc::AbstractMatrix{Float64},      
                 ud_Xc::AbstractMatrix{Float64}, 
                 ud_Yc::AbstractMatrix{Float64},

                # Weights to convert between cell-averages and cell-centers: 
              w_i_m1_X::AbstractVector{Float64}, 
               w_i_0_X::AbstractVector{Float64},
              w_i_p1_X::AbstractVector{Float64},  
              q_i_m1_X::AbstractVector{Float64}, 
               q_i_0_X::AbstractVector{Float64}, 
              q_i_p1_X::AbstractVector{Float64},  

                # Weights for WENO3 interface reconstruction:
           w_plus_i_p1::AbstractVector{Float64},  # WENO3 weights w⁺_{i,1}
          w_minus_i_p1::AbstractVector{Float64},  # WENO3 weights w⁻_{i,1} 

                 ρ_L_X::AbstractMatrix{Float64},
                 ρ_R_X::AbstractMatrix{Float64},
               u_X_L_X::AbstractMatrix{Float64},
               u_X_R_X::AbstractMatrix{Float64},
               u_Y_L_X::AbstractMatrix{Float64},
               u_Y_R_X::AbstractMatrix{Float64},
                 P_L_X::AbstractMatrix{Float64},
                 P_R_X::AbstractMatrix{Float64},
                 F_ρ_X::AbstractMatrix{Float64},
              F_ρu_X_X::AbstractMatrix{Float64},
              F_ρu_Y_X::AbstractMatrix{Float64},
                 F_E_X::AbstractMatrix{Float64},
                     P::AbstractMatrix{Float64},
            u_X_star_X::AbstractMatrix{Float64},

                ρd_L_X::AbstractMatrix{Float64},
                ρd_R_X::AbstractMatrix{Float64},
              ud_X_L_X::AbstractMatrix{Float64},
              ud_X_R_X::AbstractMatrix{Float64},
              ud_Y_L_X::AbstractMatrix{Float64},
              ud_Y_R_X::AbstractMatrix{Float64},
                F_ρd_X::AbstractMatrix{Float64},
            F_ρdud_X_X::AbstractMatrix{Float64},
            F_ρdud_Y_X::AbstractMatrix{Float64},
                                       
                # Ghost‐padded primitives & grid:
                ρg_old::AbstractMatrix{Float64}, 
              u_Xg_old::AbstractMatrix{Float64}, 
              u_Yg_old::AbstractMatrix{Float64}, 
                Pg_old::AbstractMatrix{Float64},

               ρdg_old::AbstractMatrix{Float64}, 
             ud_Xg_old::AbstractMatrix{Float64}, 
             ud_Yg_old::AbstractMatrix{Float64}, 

                    Xg::AbstractVector{Float64},
                 XCoMg::AbstractVector{Float64},

              CoordSys::CoordSystem,
            inv_r_cell::AbstractVector{Float64},
  
                # Shock indicator:
                 Shock::AbstractMatrix{Int}, 

                # Arrays for advection of passive scalars:
               Scalars::AbstractArray{Float64, 3},
           Scalars_old::AbstractArray{Float64, 3},
         Scalars_stage::AbstractArray{Float64, 3},
                ρ_save::AbstractMatrix{Float64},
 
                # BCs:
                BC_X_left::Symbol, BC_X_right::Symbol,
                BC_Y_left::Symbol, BC_Y_right::Symbol
                     )

    # Save the state at the start, before the update (needed later in RK2):
    @. ρ_stage       = ρ
    @. ρu_X_stage    = ρu_X 
    @. ρu_Y_stage    = ρu_Y
    @. E_stage       = E 
    @. ρd_stage      = ρd
    @. ρdud_X_stage  = ρdud_X
    @. ρdud_Y_stage  = ρdud_Y 
    @. Scalars_stage = Scalars
    @. Scalars_old   = Scalars

    # Step 1: Forward Euler in X:  U → U^(1)
    ForwardEuler_X!( ρ_new, ρu_X_new, ρu_Y_new, E_new,
                     ρ, ρu_X, ρu_Y, E,
                     γad, u_X, u_Y, 

                     ρd_new, ρdud_X_new, ρdud_Y_new,
                     ρd, ρdud_X, ρdud_Y, ud_X, ud_Y, 
                     
                     inv_ΔV, A_X, Δt,

                     ρc, ρu_Xc, ρu_Yc, Ec,
                     u_Xc, u_Yc, Pc,

                     ρdc, ρdud_Xc, ρdud_Yc, 
                     ud_Xc, ud_Yc, 

                     w_i_m1_X, w_i_0_X, w_i_p1_X,
                     q_i_m1_X, q_i_0_X, q_i_p1_X,
                     w_plus_i_p1, w_minus_i_p1,

                     ρ_L_X, ρ_R_X,
                     u_X_L_X, u_X_R_X,
                     u_Y_L_X, u_Y_R_X,
                     P_L_X, P_R_X,
                     F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,
                     P, u_X_star_X,

                     ρd_L_X, ρd_R_X,
                     ud_X_L_X, ud_X_R_X,
                     ud_Y_L_X, ud_Y_R_X,
                     F_ρd_X, F_ρdud_X_X, F_ρdud_Y_X, 

                     ρg_old, u_Xg_old, u_Yg_old, Pg_old,

                     ρdg_old, ud_Xg_old, ud_Yg_old, 

                     Xg, XCoMg,
                     CoordSys, inv_r_cell,
                     Shock,
                     BC_X_left, BC_X_right, 
                     BC_Y_left, BC_Y_right )

    ForwardEuler_X_Scalars!( Scalars, Scalars_old, ρ_stage, F_ρ_X, inv_ΔV, A_X, Δt )  # Advect passive scalars too

    # Step 2: Forward Euler step in X: U^(1) -> U^(2)
    @. ρ_save = ρ # Save density from after step 1 (before we update ρ), needed for scalar advection

    ForwardEuler_X!( ρ_new, ρu_X_new, ρu_Y_new, E_new,
                     ρ, ρu_X, ρu_Y, E,
                     γad, u_X, u_Y, 

                     ρd_new, ρdud_X_new, ρdud_Y_new,
                     ρd, ρdud_X, ρdud_Y, ud_X, ud_Y, 
                     
                     inv_ΔV, A_X, Δt,

                     ρc, ρu_Xc, ρu_Yc, Ec,
                     u_Xc, u_Yc, Pc,

                     ρdc, ρdud_Xc, ρdud_Yc, 
                     ud_Xc, ud_Yc, 

                     w_i_m1_X, w_i_0_X, w_i_p1_X,
                     q_i_m1_X, q_i_0_X, q_i_p1_X,
                     w_plus_i_p1, w_minus_i_p1,

                     ρ_L_X, ρ_R_X,
                     u_X_L_X, u_X_R_X,
                     u_Y_L_X, u_Y_R_X,
                     P_L_X, P_R_X,
                     F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,
                     P, u_X_star_X,

                     ρd_L_X, ρd_R_X,
                     ud_X_L_X, ud_X_R_X,
                     ud_Y_L_X, ud_Y_R_X,
                     F_ρd_X, F_ρdud_X_X, F_ρdud_Y_X, 

                     ρg_old, u_Xg_old, u_Yg_old, Pg_old,

                     ρdg_old, ud_Xg_old, ud_Yg_old, 

                     Xg, XCoMg,
                     CoordSys, inv_r_cell,
                     Shock,
                     BC_X_left, BC_X_right, 
                     BC_Y_left, BC_Y_right )

    @. Scalars_old = Scalars  # Update old values of passive scalars 
    ForwardEuler_X_Scalars!( Scalars, Scalars_old, ρ_save, F_ρ_X, inv_ΔV, A_X, Δt )  # Advect passive scalars too

    # Step 3: Combine to get RK2 update:
    #
    # U^(n+1) = 0.5 × ( U_start + U^(2) )
    #
    # Note: ρ, ρu_X, ρu_Y, E contain U^(2).
    @. ρ       = 0.5 * ( ρ_stage       + ρ )
    @. ρu_X    = 0.5 * ( ρu_X_stage    + ρu_X )
    @. ρu_Y    = 0.5 * ( ρu_Y_stage    + ρu_Y )
    @. E       = 0.5 * ( E_stage       + E )
    @. ρd      = 0.5 * ( ρd_stage      + ρd )
    @. ρdud_X  = 0.5 * ( ρdud_X_stage  + ρdud_X )
    @. ρdud_Y  = 0.5 * ( ρdud_Y_stage  + ρdud_Y )
    @. Scalars = 0.5 * ( Scalars_stage + Scalars )

    return nothing 
end


function RK2_Y!( ρ_new::AbstractMatrix{Float64},
              ρu_X_new::AbstractMatrix{Float64},
              ρu_Y_new::AbstractMatrix{Float64},
                 E_new::AbstractMatrix{Float64},
                     ρ::AbstractMatrix{Float64},
                  ρu_X::AbstractMatrix{Float64},
                  ρu_Y::AbstractMatrix{Float64},
                     E::AbstractMatrix{Float64},
                   γad::AbstractMatrix{Float64},
                   u_X::AbstractMatrix{Float64},
                   u_Y::AbstractMatrix{Float64},

                ρd_new::AbstractMatrix{Float64},
            ρdud_X_new::AbstractMatrix{Float64},
            ρdud_Y_new::AbstractMatrix{Float64},
                    ρd::AbstractMatrix{Float64},
                ρdud_X::AbstractMatrix{Float64},
                ρdud_Y::AbstractMatrix{Float64},
                  ud_X::AbstractMatrix{Float64},
                  ud_Y::AbstractMatrix{Float64},

                inv_ΔV::AbstractMatrix{Float64},
                   A_Y::AbstractMatrix{Float64},
                    Δt::Float64,

                # Scratch arrays to store state at the start:
               ρ_stage::AbstractMatrix{Float64},
            ρu_X_stage::AbstractMatrix{Float64},
            ρu_Y_stage::AbstractMatrix{Float64},
               E_stage::AbstractMatrix{Float64},

              ρd_stage::AbstractMatrix{Float64},
          ρdud_X_stage::AbstractMatrix{Float64},
          ρdud_Y_stage::AbstractMatrix{Float64},

                # Conservative & primitive variables @ cell centers:
                    ρc::AbstractMatrix{Float64}, 
                 ρu_Xc::AbstractMatrix{Float64}, 
                 ρu_Yc::AbstractMatrix{Float64},
                    Ec::AbstractMatrix{Float64},         
                  u_Xc::AbstractMatrix{Float64}, 
                  u_Yc::AbstractMatrix{Float64}, 
                    Pc::AbstractMatrix{Float64}, 
                    
                   ρdc::AbstractMatrix{Float64}, 
               ρdud_Xc::AbstractMatrix{Float64}, 
               ρdud_Yc::AbstractMatrix{Float64},       
                 ud_Xc::AbstractMatrix{Float64}, 
                 ud_Yc::AbstractMatrix{Float64}, 

                # Weights to convert between cell-averages and cell-centers: 
              w_j_m1_Y::AbstractVector{Float64}, 
               w_j_0_Y::AbstractVector{Float64}, 
              w_j_p1_Y::AbstractVector{Float64},  
              q_j_m1_Y::AbstractVector{Float64}, 
               q_j_0_Y::AbstractVector{Float64}, 
              q_j_p1_Y::AbstractVector{Float64},  

                # Weights for WENO3 interface reconstruction:
           w_plus_j_p1::AbstractVector{Float64},  # WENO3 weights w⁺_{j,1}
          w_minus_j_p1::AbstractVector{Float64},  # WENO3 weights w⁻_{j,1}

                 ρ_L_Y::AbstractMatrix{Float64},
                 ρ_R_Y::AbstractMatrix{Float64},
               u_Y_L_Y::AbstractMatrix{Float64},
               u_Y_R_Y::AbstractMatrix{Float64},
               u_X_L_Y::AbstractMatrix{Float64},
               u_X_R_Y::AbstractMatrix{Float64},
                 P_L_Y::AbstractMatrix{Float64},
                 P_R_Y::AbstractMatrix{Float64},
                 F_ρ_Y::AbstractMatrix{Float64},
              F_ρu_X_Y::AbstractMatrix{Float64},
              F_ρu_Y_Y::AbstractMatrix{Float64},
                 F_E_Y::AbstractMatrix{Float64},
                     P::AbstractMatrix{Float64},
            u_Y_star_Y::AbstractMatrix{Float64},

                ρd_L_Y::AbstractMatrix{Float64},
                ρd_R_Y::AbstractMatrix{Float64},
              ud_Y_L_Y::AbstractMatrix{Float64},
              ud_Y_R_Y::AbstractMatrix{Float64},
              ud_X_L_Y::AbstractMatrix{Float64},
              ud_X_R_Y::AbstractMatrix{Float64},
                F_ρd_Y::AbstractMatrix{Float64},
            F_ρdud_X_Y::AbstractMatrix{Float64},
            F_ρdud_Y_Y::AbstractMatrix{Float64},
                                       
                # Ghost‐padded primitives & grid:
                ρg_old::AbstractMatrix{Float64}, 
              u_Xg_old::AbstractMatrix{Float64}, 
              u_Yg_old::AbstractMatrix{Float64}, 
                Pg_old::AbstractMatrix{Float64},

               ρdg_old::AbstractMatrix{Float64}, 
             ud_Xg_old::AbstractMatrix{Float64}, 
             ud_Yg_old::AbstractMatrix{Float64}, 

                    Yg::AbstractVector{Float64},
                 YCoMg::AbstractVector{Float64},

                # Shock indicator:
                 Shock::AbstractMatrix{Int},

                # Arrays for advection of passive scalars:
               Scalars::AbstractArray{Float64, 3},
           Scalars_old::AbstractArray{Float64, 3},
         Scalars_stage::AbstractArray{Float64, 3},
                ρ_save::AbstractMatrix{Float64},
 
                # BCs:
             BC_X_left::Symbol, BC_X_right::Symbol,
             BC_Y_left::Symbol, BC_Y_right::Symbol
                       )

    # Save the state at the start, before the update (needed later in RK2):
    @. ρ_stage       = ρ
    @. ρu_X_stage    = ρu_X 
    @. ρu_Y_stage    = ρu_Y
    @. E_stage       = E 
    @. ρd_stage      = ρd
    @. ρdud_X_stage  = ρdud_X
    @. ρdud_Y_stage  = ρdud_Y 
    @. Scalars_stage = Scalars
    @. Scalars_old   = Scalars

    # Step 1: Forward Euler in Y: U → U^(1)
    ForwardEuler_Y!( ρ_new, ρu_X_new, ρu_Y_new, E_new,
                     ρ, ρu_X, ρu_Y, E,
                     γad, u_X, u_Y, 

                     ρd_new, ρdud_X_new, ρdud_Y_new,
                     ρd, ρdud_X, ρdud_Y, 
                     ud_X, ud_Y, 
                     
                     inv_ΔV, A_Y, Δt,


                     ρc, ρu_Xc, ρu_Yc, Ec,
                     u_Xc, u_Yc, Pc,

                     ρdc, ρdud_Xc, ρdud_Yc, 
                     ud_Xc, ud_Yc, 

                     w_j_m1_Y, w_j_0_Y, w_j_p1_Y,
                     q_j_m1_Y, q_j_0_Y, q_j_p1_Y,
                     w_plus_j_p1, w_minus_j_p1,

                     ρ_L_Y, ρ_R_Y,
                     u_Y_L_Y, u_Y_R_Y,
                     u_X_L_Y, u_X_R_Y,
                     P_L_Y, P_R_Y,
                     F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,
                     P, u_Y_star_Y,

                     ρd_L_Y, ρd_R_Y,
                     ud_Y_L_Y, ud_Y_R_Y,
                     ud_X_L_Y, ud_X_R_Y,
                     F_ρd_Y, F_ρdud_X_Y, F_ρdud_Y_Y, 

                     ρg_old, u_Xg_old, u_Yg_old, Pg_old,

                     ρdg_old, ud_Xg_old, ud_Yg_old, 

                     Yg, YCoMg,
                     Shock,
                     BC_X_left, BC_X_right,
                     BC_Y_left, BC_Y_right )


    ForwardEuler_Y_Scalars!( Scalars, Scalars_old, ρ_stage, F_ρ_Y, inv_ΔV, A_Y, Δt )  # Advect passive scalars too

    # Step 2: Forward Euler in Y again: U^(1) → U^(2)
    @. ρ_save = ρ # Save density from after step 1 (before we update ρ), needed for scalar advection

    ForwardEuler_Y!( ρ_new, ρu_X_new, ρu_Y_new, E_new,
                     ρ, ρu_X, ρu_Y, E,
                     γad, u_X, u_Y, 

                     ρd_new, ρdud_X_new, ρdud_Y_new,
                     ρd, ρdud_X, ρdud_Y, 
                     ud_X, ud_Y, 
                     
                     inv_ΔV, A_Y, Δt,


                     ρc, ρu_Xc, ρu_Yc, Ec,
                     u_Xc, u_Yc, Pc,

                     ρdc, ρdud_Xc, ρdud_Yc, 
                     ud_Xc, ud_Yc, 

                     w_j_m1_Y, w_j_0_Y, w_j_p1_Y,
                     q_j_m1_Y, q_j_0_Y, q_j_p1_Y,
                     w_plus_j_p1, w_minus_j_p1,

                     ρ_L_Y, ρ_R_Y,
                     u_Y_L_Y, u_Y_R_Y,
                     u_X_L_Y, u_X_R_Y,
                     P_L_Y, P_R_Y,
                     F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,
                     P, u_Y_star_Y,

                     ρd_L_Y, ρd_R_Y,
                     ud_Y_L_Y, ud_Y_R_Y,
                     ud_X_L_Y, ud_X_R_Y,
                     F_ρd_Y, F_ρdud_X_Y, F_ρdud_Y_Y, 

                     ρg_old, u_Xg_old, u_Yg_old, Pg_old,

                     ρdg_old, ud_Xg_old, ud_Yg_old, 

                     Yg, YCoMg,
                     Shock,
                     BC_X_left, BC_X_right,
                     BC_Y_left, BC_Y_right )


    @. Scalars_old = Scalars  # Update old values of passive scalars 
    ForwardEuler_Y_Scalars!( Scalars, Scalars_old, ρ_save, F_ρ_Y, inv_ΔV, A_Y, Δt )  # Advect passive scalars too

    # Step 3: Combine to get RK2 update:
    #
    # U^(n+1) = 0.5 × ( U_start + U^(2) )
    #
    # Note: ρ, ρu_X, ρu_Y, E contain U^(2).
    @. ρ       = 0.5 * ( ρ_stage       + ρ )
    @. ρu_X    = 0.5 * ( ρu_X_stage    + ρu_X )
    @. ρu_Y    = 0.5 * ( ρu_Y_stage    + ρu_Y )
    @. E       = 0.5 * ( E_stage       + E )
    @. ρd      = 0.5 * ( ρd_stage      + ρd )
    @. ρdud_X  = 0.5 * ( ρdud_X_stage  + ρdud_X )
    @. ρdud_Y  = 0.5 * ( ρdud_Y_stage  + ρdud_Y )
    @. Scalars = 0.5 * ( Scalars_stage + Scalars )

    return nothing
end


function UpdateHydro_StrangSplit_Xfirst!( ρ_new::AbstractMatrix{Float64},
                                       ρu_X_new::AbstractMatrix{Float64},
                                       ρu_Y_new::AbstractMatrix{Float64},
                                          E_new::AbstractMatrix{Float64},
                                              ρ::AbstractMatrix{Float64},
                                           ρu_X::AbstractMatrix{Float64},
                                           ρu_Y::AbstractMatrix{Float64},
                                              E::AbstractMatrix{Float64},
                                            γad::AbstractMatrix{Float64},
                                            u_X::AbstractMatrix{Float64},
                                            u_Y::AbstractMatrix{Float64},

                                         ρd_new::AbstractMatrix{Float64},
                                     ρdud_X_new::AbstractMatrix{Float64},
                                     ρdud_Y_new::AbstractMatrix{Float64},
                                             ρd::AbstractMatrix{Float64},
                                         ρdud_X::AbstractMatrix{Float64},
                                         ρdud_Y::AbstractMatrix{Float64},
                                           ud_X::AbstractMatrix{Float64},
                                           ud_Y::AbstractMatrix{Float64},

                                         inv_ΔV::AbstractMatrix{Float64},
                                            A_X::AbstractMatrix{Float64},
                                            A_Y::AbstractMatrix{Float64},
                                             Δt::Float64,

                                             # Scratch arrays for RK integration:
                                        ρ_stage::AbstractMatrix{Float64},
                                     ρu_X_stage::AbstractMatrix{Float64},
                                     ρu_Y_stage::AbstractMatrix{Float64},
                                        E_stage::AbstractMatrix{Float64},

                                       ρd_stage::AbstractMatrix{Float64},
                                   ρdud_X_stage::AbstractMatrix{Float64},
                                   ρdud_Y_stage::AbstractMatrix{Float64},

                                             # Conservative & primitive variables @ cell centers:
                                             ρc::AbstractMatrix{Float64}, 
                                          ρu_Xc::AbstractMatrix{Float64}, 
                                          ρu_Yc::AbstractMatrix{Float64},
                                             Ec::AbstractMatrix{Float64},         
                                           u_Xc::AbstractMatrix{Float64}, 
                                           u_Yc::AbstractMatrix{Float64}, 
                                             Pc::AbstractMatrix{Float64},  
                                             
                                            ρdc::AbstractMatrix{Float64}, 
                                        ρdud_Xc::AbstractMatrix{Float64}, 
                                        ρdud_Yc::AbstractMatrix{Float64},     
                                          ud_Xc::AbstractMatrix{Float64}, 
                                          ud_Yc::AbstractMatrix{Float64},  

                                            # Weights to convert between cell-averages and cell-centers: 
                                       w_i_m1_X::AbstractVector{Float64}, 
                                        w_i_0_X::AbstractVector{Float64},
                                       w_i_p1_X::AbstractVector{Float64},  
                                       w_j_m1_Y::AbstractVector{Float64}, 
                                        w_j_0_Y::AbstractVector{Float64}, 
                                       w_j_p1_Y::AbstractVector{Float64}, 
                                       q_i_m1_X::AbstractVector{Float64}, 
                                        q_i_0_X::AbstractVector{Float64}, 
                                       q_i_p1_X::AbstractVector{Float64},  
                                       q_j_m1_Y::AbstractVector{Float64}, 
                                        q_j_0_Y::AbstractVector{Float64}, 
                                       q_j_p1_Y::AbstractVector{Float64},  

                                           # Weights for WENO3 interface reconstruction:
                                    w_plus_i_p1::AbstractVector{Float64},  # WENO3 weights w⁺_{i,1}
                                   w_minus_i_p1::AbstractVector{Float64},  # WENO3 weights w⁻_{i,1} 
                                    w_plus_j_p1::AbstractVector{Float64},  # WENO3 weights w⁺_{j,1}
                                   w_minus_j_p1::AbstractVector{Float64},  # WENO3 weights w⁻_{j,1}

                                          ρ_L_X::AbstractMatrix{Float64},
                                          ρ_R_X::AbstractMatrix{Float64},
                                        u_X_L_X::AbstractMatrix{Float64},
                                        u_X_R_X::AbstractMatrix{Float64},
                                        u_Y_L_X::AbstractMatrix{Float64},
                                        u_Y_R_X::AbstractMatrix{Float64},
                                          P_L_X::AbstractMatrix{Float64},
                                          P_R_X::AbstractMatrix{Float64},
                                          F_ρ_X::AbstractMatrix{Float64},
                                       F_ρu_X_X::AbstractMatrix{Float64},
                                       F_ρu_Y_X::AbstractMatrix{Float64},
                                          F_E_X::AbstractMatrix{Float64},

                                          ρ_L_Y::AbstractMatrix{Float64},
                                          ρ_R_Y::AbstractMatrix{Float64},
                                        u_Y_L_Y::AbstractMatrix{Float64},
                                        u_Y_R_Y::AbstractMatrix{Float64},
                                        u_X_L_Y::AbstractMatrix{Float64},
                                        u_X_R_Y::AbstractMatrix{Float64},
                                          P_L_Y::AbstractMatrix{Float64},
                                          P_R_Y::AbstractMatrix{Float64},
                                          F_ρ_Y::AbstractMatrix{Float64},
                                       F_ρu_X_Y::AbstractMatrix{Float64},
                                       F_ρu_Y_Y::AbstractMatrix{Float64},
                                          F_E_Y::AbstractMatrix{Float64},
                                              P::AbstractMatrix{Float64},
                                     u_X_star_X::AbstractMatrix{Float64},
                                     u_Y_star_Y::AbstractMatrix{Float64},

                                         ρd_L_X::AbstractMatrix{Float64},
                                         ρd_R_X::AbstractMatrix{Float64},
                                       ud_X_L_X::AbstractMatrix{Float64},
                                       ud_X_R_X::AbstractMatrix{Float64},
                                       ud_Y_L_X::AbstractMatrix{Float64},
                                       ud_Y_R_X::AbstractMatrix{Float64},
                                         F_ρd_X::AbstractMatrix{Float64},
                                     F_ρdud_X_X::AbstractMatrix{Float64},
                                     F_ρdud_Y_X::AbstractMatrix{Float64},

                                         ρd_L_Y::AbstractMatrix{Float64},
                                         ρd_R_Y::AbstractMatrix{Float64},
                                       ud_Y_L_Y::AbstractMatrix{Float64},
                                       ud_Y_R_Y::AbstractMatrix{Float64},
                                       ud_X_L_Y::AbstractMatrix{Float64},
                                       ud_X_R_Y::AbstractMatrix{Float64},
                                         F_ρd_Y::AbstractMatrix{Float64},
                                     F_ρdud_X_Y::AbstractMatrix{Float64},
                                     F_ρdud_Y_Y::AbstractMatrix{Float64},
                                       
                                         # Ghost‐padded primitives & grid:
                                         ρg_old::AbstractMatrix{Float64}, 
                                       u_Xg_old::AbstractMatrix{Float64}, 
                                       u_Yg_old::AbstractMatrix{Float64}, 
                                         Pg_old::AbstractMatrix{Float64},

                                        ρdg_old::AbstractMatrix{Float64}, 
                                      ud_Xg_old::AbstractMatrix{Float64}, 
                                      ud_Yg_old::AbstractMatrix{Float64}, 

                                             Xg::AbstractVector{Float64},
                                          XCoMg::AbstractVector{Float64},
                                             Yg::AbstractVector{Float64},
                                          YCoMg::AbstractVector{Float64},

                                       CoordSys::CoordSystem,
                                     inv_r_cell::AbstractVector{Float64},
  
                                        # Shock indicator:
                                          Shock::AbstractMatrix{Int}, 

                                        # Arrays for advection of passive scalars:
                                        Scalars::AbstractArray{Float64, 3},
                                    Scalars_old::AbstractArray{Float64, 3},
                                  Scalars_stage::AbstractArray{Float64, 3},
                                         ρ_save::AbstractMatrix{Float64},
 
                                          # BCs:
                                      BC_X_left::Symbol, BC_X_right::Symbol,
                                      BC_Y_left::Symbol, BC_Y_right::Symbol

                                          )

    Δt_half  = 0.5 * Δt

    #####################################################################
    #######                                                       #######
    #######   1)  H A L F - S T E P  I N  X - D I R E C T I O N   #######
    #######                                                       #######
    #####################################################################

    RK2_X!( ρ_new, ρu_X_new, ρu_Y_new, E_new,
            ρ, ρu_X, ρu_Y, E, γad, u_X, u_Y,

            ρd_new, ρdud_X_new, ρdud_Y_new, 
            ρd, ρdud_X, ρdud_Y, ud_X, ud_Y,

            inv_ΔV, A_X, Δt_half,

            ρ_stage, ρu_X_stage, ρu_Y_stage, E_stage,

            ρd_stage, ρdud_X_stage, ρdud_Y_stage,

            ρc, ρu_Xc, ρu_Yc, Ec,         
            u_Xc, u_Yc, Pc,
            
            ρdc, ρdud_Xc, ρdud_Yc,         
            ud_Xc, ud_Yc, 

            w_i_m1_X, w_i_0_X, w_i_p1_X,  q_i_m1_X, q_i_0_X, q_i_p1_X,  

            w_plus_i_p1, w_minus_i_p1,

            ρ_L_X, ρ_R_X, u_X_L_X, u_X_R_X, u_Y_L_X, u_Y_R_X, P_L_X, P_R_X,
            F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X, P, u_X_star_X,

            ρd_L_X, ρd_R_X, ud_X_L_X, ud_X_R_X, ud_Y_L_X, ud_Y_R_X, 
            F_ρd_X, F_ρdud_X_X, F_ρdud_Y_X, 
                                       
            ρg_old, u_Xg_old, u_Yg_old, Pg_old, 

            ρdg_old, ud_Xg_old, ud_Yg_old,
            
            Xg, XCoMg, 

            CoordSys, inv_r_cell, Shock, 

            Scalars, Scalars_old, Scalars_stage, ρ_save,
 
            BC_X_left, BC_X_right,
            BC_Y_left, BC_Y_right )

    #####################################################################
    #######                                                       #######
    #######   2)  F U L L - S T E P  I N  Y - D I R E C T I O N   #######
    #######                                                       #######
    #####################################################################

    RK2_Y!( ρ_new, ρu_X_new, ρu_Y_new, E_new,
            ρ, ρu_X, ρu_Y, E, γad, u_X, u_Y,

            ρd_new, ρdud_X_new, ρdud_Y_new, 
            ρd, ρdud_X, ρdud_Y, ud_X, ud_Y,

            inv_ΔV, A_Y, Δt,

            ρ_stage, ρu_X_stage, ρu_Y_stage, E_stage,

            ρd_stage, ρdud_X_stage, ρdud_Y_stage,

            ρc, ρu_Xc, ρu_Yc, Ec, u_Xc, u_Yc, Pc,

            ρdc, ρdud_Xc, ρdud_Yc, ud_Xc, ud_Yc,          

            w_j_m1_Y, w_j_0_Y, w_j_p1_Y,  q_j_m1_Y, q_j_0_Y, q_j_p1_Y,  

            w_plus_j_p1, w_minus_j_p1, 

            ρ_L_Y, ρ_R_Y, u_Y_L_Y, u_Y_R_Y, u_X_L_Y, u_X_R_Y, P_L_Y, P_R_Y,
            F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y, P, u_Y_star_Y,

            ρd_L_Y, ρd_R_Y, ud_Y_L_Y, ud_Y_R_Y, ud_X_L_Y, ud_X_R_Y, 
            F_ρd_Y, F_ρdud_X_Y, F_ρdud_Y_Y,
                                       
            ρg_old, u_Xg_old, u_Yg_old, Pg_old, 

            ρdg_old, ud_Xg_old, ud_Yg_old,
            
            Yg, YCoMg, 
            
            Shock,

            Scalars, Scalars_old, Scalars_stage, ρ_save,
            
            BC_X_left, BC_X_right,
            BC_Y_left, BC_Y_right )

    ###############################################################################
    #######                                                                 #######
    #######   3) F I N A L  H A L F - S T E P  I N  X - D I R E C T I O N   #######
    #######                                                                 #######
    ###############################################################################

    RK2_X!( ρ_new, ρu_X_new, ρu_Y_new, E_new,
            ρ, ρu_X, ρu_Y, E, γad, u_X, u_Y,

            ρd_new, ρdud_X_new, ρdud_Y_new, 
            ρd, ρdud_X, ρdud_Y, ud_X, ud_Y,

            inv_ΔV, A_X, Δt_half,

            ρ_stage, ρu_X_stage, ρu_Y_stage, E_stage,

            ρd_stage, ρdud_X_stage, ρdud_Y_stage,

            ρc, ρu_Xc, ρu_Yc, Ec,         
            u_Xc, u_Yc, Pc,
            
            ρdc, ρdud_Xc, ρdud_Yc,         
            ud_Xc, ud_Yc, 

            w_i_m1_X, w_i_0_X, w_i_p1_X,  q_i_m1_X, q_i_0_X, q_i_p1_X,  

            w_plus_i_p1, w_minus_i_p1,

            ρ_L_X, ρ_R_X, u_X_L_X, u_X_R_X, u_Y_L_X, u_Y_R_X, P_L_X, P_R_X,
            F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X, P, u_X_star_X,

            ρd_L_X, ρd_R_X, ud_X_L_X, ud_X_R_X, ud_Y_L_X, ud_Y_R_X, 
            F_ρd_X, F_ρdud_X_X, F_ρdud_Y_X, 
                                       
            ρg_old, u_Xg_old, u_Yg_old, Pg_old, 

            ρdg_old, ud_Xg_old, ud_Yg_old,
            
            Xg, XCoMg, 

            CoordSys, inv_r_cell, Shock, 

            Scalars, Scalars_old, Scalars_stage, ρ_save,
 
            BC_X_left, BC_X_right,
            BC_Y_left, BC_Y_right )

    
    return nothing
end



function UpdateHydro_StrangSplit_Yfirst!( ρ_new::AbstractMatrix{Float64},
                                       ρu_X_new::AbstractMatrix{Float64},
                                       ρu_Y_new::AbstractMatrix{Float64},
                                          E_new::AbstractMatrix{Float64},
                                              ρ::AbstractMatrix{Float64},
                                           ρu_X::AbstractMatrix{Float64},
                                           ρu_Y::AbstractMatrix{Float64},
                                              E::AbstractMatrix{Float64},
                                            γad::AbstractMatrix{Float64},
                                            u_X::AbstractMatrix{Float64},
                                            u_Y::AbstractMatrix{Float64},

                                         ρd_new::AbstractMatrix{Float64},
                                     ρdud_X_new::AbstractMatrix{Float64},
                                     ρdud_Y_new::AbstractMatrix{Float64},
                                             ρd::AbstractMatrix{Float64},
                                         ρdud_X::AbstractMatrix{Float64},
                                         ρdud_Y::AbstractMatrix{Float64},
                                           ud_X::AbstractMatrix{Float64},
                                           ud_Y::AbstractMatrix{Float64},

                                         inv_ΔV::AbstractMatrix{Float64},
                                            A_X::AbstractMatrix{Float64},
                                            A_Y::AbstractMatrix{Float64},
                                             Δt::Float64,

                                            # Scratch arrays for RK integration:
                                        ρ_stage::AbstractMatrix{Float64},
                                     ρu_X_stage::AbstractMatrix{Float64},
                                     ρu_Y_stage::AbstractMatrix{Float64},
                                        E_stage::AbstractMatrix{Float64},

                                       ρd_stage::AbstractMatrix{Float64},
                                   ρdud_X_stage::AbstractMatrix{Float64},
                                   ρdud_Y_stage::AbstractMatrix{Float64},

                                            # Conservative & primitive variables @ cell centers:
                                             ρc::AbstractMatrix{Float64}, 
                                          ρu_Xc::AbstractMatrix{Float64}, 
                                          ρu_Yc::AbstractMatrix{Float64},
                                             Ec::AbstractMatrix{Float64},         
                                           u_Xc::AbstractMatrix{Float64}, 
                                           u_Yc::AbstractMatrix{Float64}, 
                                             Pc::AbstractMatrix{Float64}, 
                                             
                                            ρdc::AbstractMatrix{Float64}, 
                                        ρdud_Xc::AbstractMatrix{Float64}, 
                                        ρdud_Yc::AbstractMatrix{Float64},     
                                          ud_Xc::AbstractMatrix{Float64}, 
                                          ud_Yc::AbstractMatrix{Float64}, 

                                            # Weights to convert between cell-averages and cell-centers: 
                                       w_i_m1_X::AbstractVector{Float64}, 
                                        w_i_0_X::AbstractVector{Float64},
                                       w_i_p1_X::AbstractVector{Float64},  
                                       w_j_m1_Y::AbstractVector{Float64}, 
                                        w_j_0_Y::AbstractVector{Float64}, 
                                       w_j_p1_Y::AbstractVector{Float64}, 
                                       q_i_m1_X::AbstractVector{Float64}, 
                                        q_i_0_X::AbstractVector{Float64}, 
                                       q_i_p1_X::AbstractVector{Float64},  
                                       q_j_m1_Y::AbstractVector{Float64}, 
                                        q_j_0_Y::AbstractVector{Float64}, 
                                       q_j_p1_Y::AbstractVector{Float64},  

                                            # Weights for WENO3 interface reconstruction:
                                    w_plus_i_p1::AbstractVector{Float64},  # WENO3 weights w⁺_{i,1}
                                   w_minus_i_p1::AbstractVector{Float64},  # WENO3 weights w⁻_{i,1} 
                                    w_plus_j_p1::AbstractVector{Float64},  # WENO3 weights w⁺_{j,1}
                                   w_minus_j_p1::AbstractVector{Float64},  # WENO3 weights w⁻_{j,1}


                                          ρ_L_X::AbstractMatrix{Float64},
                                          ρ_R_X::AbstractMatrix{Float64},
                                        u_X_L_X::AbstractMatrix{Float64},
                                        u_X_R_X::AbstractMatrix{Float64},
                                        u_Y_L_X::AbstractMatrix{Float64},
                                        u_Y_R_X::AbstractMatrix{Float64},
                                          P_L_X::AbstractMatrix{Float64},
                                          P_R_X::AbstractMatrix{Float64},
                                          F_ρ_X::AbstractMatrix{Float64},
                                       F_ρu_X_X::AbstractMatrix{Float64},
                                       F_ρu_Y_X::AbstractMatrix{Float64},
                                          F_E_X::AbstractMatrix{Float64},

                                          ρ_L_Y::AbstractMatrix{Float64},
                                          ρ_R_Y::AbstractMatrix{Float64},
                                        u_Y_L_Y::AbstractMatrix{Float64},
                                        u_Y_R_Y::AbstractMatrix{Float64},
                                        u_X_L_Y::AbstractMatrix{Float64},
                                        u_X_R_Y::AbstractMatrix{Float64},
                                          P_L_Y::AbstractMatrix{Float64},
                                          P_R_Y::AbstractMatrix{Float64},
                                          F_ρ_Y::AbstractMatrix{Float64},
                                       F_ρu_X_Y::AbstractMatrix{Float64},
                                       F_ρu_Y_Y::AbstractMatrix{Float64},
                                          F_E_Y::AbstractMatrix{Float64},
                                              P::AbstractMatrix{Float64},
                                     u_X_star_X::AbstractMatrix{Float64},
                                     u_Y_star_Y::AbstractMatrix{Float64},

                                         ρd_L_X::AbstractMatrix{Float64},
                                         ρd_R_X::AbstractMatrix{Float64},
                                       ud_X_L_X::AbstractMatrix{Float64},
                                       ud_X_R_X::AbstractMatrix{Float64},
                                       ud_Y_L_X::AbstractMatrix{Float64},
                                       ud_Y_R_X::AbstractMatrix{Float64},
                                         F_ρd_X::AbstractMatrix{Float64},
                                     F_ρdud_X_X::AbstractMatrix{Float64},
                                     F_ρdud_Y_X::AbstractMatrix{Float64},

                                         ρd_L_Y::AbstractMatrix{Float64},
                                         ρd_R_Y::AbstractMatrix{Float64},
                                       ud_Y_L_Y::AbstractMatrix{Float64},
                                       ud_Y_R_Y::AbstractMatrix{Float64},
                                       ud_X_L_Y::AbstractMatrix{Float64},
                                       ud_X_R_Y::AbstractMatrix{Float64},
                                         F_ρd_Y::AbstractMatrix{Float64},
                                     F_ρdud_X_Y::AbstractMatrix{Float64},
                                     F_ρdud_Y_Y::AbstractMatrix{Float64},
                                        
                                    # Ghost‐padded primitives & grid:
                                         ρg_old::AbstractMatrix{Float64}, 
                                       u_Xg_old::AbstractMatrix{Float64}, 
                                       u_Yg_old::AbstractMatrix{Float64}, 
                                         Pg_old::AbstractMatrix{Float64},

                                        ρdg_old::AbstractMatrix{Float64}, 
                                      ud_Xg_old::AbstractMatrix{Float64}, 
                                      ud_Yg_old::AbstractMatrix{Float64}, 

                                             Xg::AbstractVector{Float64},
                                          XCoMg::AbstractVector{Float64},
                                             Yg::AbstractVector{Float64},
                                          YCoMg::AbstractVector{Float64},

                                       CoordSys::CoordSystem,
                                     inv_r_cell::AbstractVector{Float64}, 

                                    # Shock indicator:
                                          Shock::AbstractMatrix{Int}, 

                                    # Arrays for advection of passive scalars:
                                        Scalars::AbstractArray{Float64, 3},
                                    Scalars_old::AbstractArray{Float64, 3},
                                  Scalars_stage::AbstractArray{Float64, 3},
                                         ρ_save::AbstractMatrix{Float64},

                                      # BCs:
                                      BC_X_left::Symbol, BC_X_right::Symbol,
                                      BC_Y_left::Symbol, BC_Y_right::Symbol
                             )

    Δt_half  = 0.5 * Δt

    #####################################################################
    #######                                                       #######
    #######   1)  H A L F - S T E P  I N  Y - D I R E C T I O N   #######
    #######                                                       #######
    #####################################################################
    
    RK2_Y!( ρ_new, ρu_X_new, ρu_Y_new, E_new,
            ρ, ρu_X, ρu_Y, E, γad, u_X, u_Y,

            ρd_new, ρdud_X_new, ρdud_Y_new, 
            ρd, ρdud_X, ρdud_Y, ud_X, ud_Y,

            inv_ΔV, A_Y, Δt_half,

            ρ_stage, ρu_X_stage, ρu_Y_stage, E_stage,

            ρd_stage, ρdud_X_stage, ρdud_Y_stage,

            ρc, ρu_Xc, ρu_Yc, Ec, u_Xc, u_Yc, Pc,

            ρdc, ρdud_Xc, ρdud_Yc, ud_Xc, ud_Yc,          

            w_j_m1_Y, w_j_0_Y, w_j_p1_Y,  q_j_m1_Y, q_j_0_Y, q_j_p1_Y,  

            w_plus_j_p1, w_minus_j_p1, 

            ρ_L_Y, ρ_R_Y, u_Y_L_Y, u_Y_R_Y, u_X_L_Y, u_X_R_Y, P_L_Y, P_R_Y,
            F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y, P, u_Y_star_Y,

            ρd_L_Y, ρd_R_Y, ud_Y_L_Y, ud_Y_R_Y, ud_X_L_Y, ud_X_R_Y, 
            F_ρd_Y, F_ρdud_X_Y, F_ρdud_Y_Y,
                                       
            ρg_old, u_Xg_old, u_Yg_old, Pg_old, 

            ρdg_old, ud_Xg_old, ud_Yg_old,
            
            Yg, YCoMg, 
            
            Shock,

            Scalars, Scalars_old, Scalars_stage, ρ_save,
            
            BC_X_left, BC_X_right,
            BC_Y_left, BC_Y_right )

    #####################################################################
    #######                                                       #######
    #######   2)  F U L L - S T E P  I N  X - D I R E C T I O N   #######
    #######                                                       #######
    #####################################################################
    
    RK2_X!( ρ_new, ρu_X_new, ρu_Y_new, E_new,
            ρ, ρu_X, ρu_Y, E, γad, u_X, u_Y,

            ρd_new, ρdud_X_new, ρdud_Y_new, 
            ρd, ρdud_X, ρdud_Y, ud_X, ud_Y,

            inv_ΔV, A_X, Δt,

            ρ_stage, ρu_X_stage, ρu_Y_stage, E_stage,

            ρd_stage, ρdud_X_stage, ρdud_Y_stage,

            ρc, ρu_Xc, ρu_Yc, Ec,         
            u_Xc, u_Yc, Pc,
            
            ρdc, ρdud_Xc, ρdud_Yc,         
            ud_Xc, ud_Yc, 

            w_i_m1_X, w_i_0_X, w_i_p1_X,  q_i_m1_X, q_i_0_X, q_i_p1_X,  

            w_plus_i_p1, w_minus_i_p1,

            ρ_L_X, ρ_R_X, u_X_L_X, u_X_R_X, u_Y_L_X, u_Y_R_X, P_L_X, P_R_X,
            F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X, P, u_X_star_X,

            ρd_L_X, ρd_R_X, ud_X_L_X, ud_X_R_X, ud_Y_L_X, ud_Y_R_X, 
            F_ρd_X, F_ρdud_X_X, F_ρdud_Y_X, 
                                       
            ρg_old, u_Xg_old, u_Yg_old, Pg_old, 

            ρdg_old, ud_Xg_old, ud_Yg_old,
            
            Xg, XCoMg, 

            CoordSys, inv_r_cell, Shock, 

            Scalars, Scalars_old, Scalars_stage, ρ_save,
 
            BC_X_left, BC_X_right,
            BC_Y_left, BC_Y_right )

    ###############################################################################
    #######                                                                 #######
    #######   3) F I N A L  H A L F - S T E P  I N  Y - D I R E C T I O N   #######
    #######                                                                 #######
    ###############################################################################
    
    RK2_Y!( ρ_new, ρu_X_new, ρu_Y_new, E_new,
            ρ, ρu_X, ρu_Y, E, γad, u_X, u_Y,

            ρd_new, ρdud_X_new, ρdud_Y_new, 
            ρd, ρdud_X, ρdud_Y, ud_X, ud_Y,

            inv_ΔV, A_Y, Δt_half,

            ρ_stage, ρu_X_stage, ρu_Y_stage, E_stage,

            ρd_stage, ρdud_X_stage, ρdud_Y_stage,

            ρc, ρu_Xc, ρu_Yc, Ec, u_Xc, u_Yc, Pc,

            ρdc, ρdud_Xc, ρdud_Yc, ud_Xc, ud_Yc,          

            w_j_m1_Y, w_j_0_Y, w_j_p1_Y,  q_j_m1_Y, q_j_0_Y, q_j_p1_Y,  

            w_plus_j_p1, w_minus_j_p1, 

            ρ_L_Y, ρ_R_Y, u_Y_L_Y, u_Y_R_Y, u_X_L_Y, u_X_R_Y, P_L_Y, P_R_Y,
            F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y, P, u_Y_star_Y,

            ρd_L_Y, ρd_R_Y, ud_Y_L_Y, ud_Y_R_Y, ud_X_L_Y, ud_X_R_Y, 
            F_ρd_Y, F_ρdud_X_Y, F_ρdud_Y_Y,
                                       
            ρg_old, u_Xg_old, u_Yg_old, Pg_old, 

            ρdg_old, ud_Xg_old, ud_Yg_old,
            
            Yg, YCoMg, 
            
            Shock,

            Scalars, Scalars_old, Scalars_stage, ρ_save,
            
            BC_X_left, BC_X_right,
            BC_Y_left, BC_Y_right )

    return nothing
end


function UpdateHydro!( ρ_new::AbstractMatrix{Float64},
                    ρu_X_new::AbstractMatrix{Float64},
                    ρu_Y_new::AbstractMatrix{Float64},
                       E_new::AbstractMatrix{Float64},
                           ρ::AbstractMatrix{Float64},
                        ρu_X::AbstractMatrix{Float64},
                        ρu_Y::AbstractMatrix{Float64},
                           E::AbstractMatrix{Float64},
                          ρ0::AbstractMatrix{Float64},
                       ρu_X0::AbstractMatrix{Float64},
                       ρu_Y0::AbstractMatrix{Float64},
                          E0::AbstractMatrix{Float64},

                         γad::AbstractMatrix{Float64},
                         u_X::AbstractMatrix{Float64},
                         u_Y::AbstractMatrix{Float64},

                      ρd_new::AbstractMatrix{Float64},
                  ρdud_X_new::AbstractMatrix{Float64},
                  ρdud_Y_new::AbstractMatrix{Float64},
                          ρd::AbstractMatrix{Float64},
                      ρdud_X::AbstractMatrix{Float64},
                      ρdud_Y::AbstractMatrix{Float64},
                         ρd0::AbstractMatrix{Float64},
                     ρdud_X0::AbstractMatrix{Float64},
                     ρdud_Y0::AbstractMatrix{Float64},

                        ud_X::AbstractMatrix{Float64},
                        ud_Y::AbstractMatrix{Float64},

                      inv_ΔV::AbstractMatrix{Float64},
                         A_X::AbstractMatrix{Float64},
                         A_Y::AbstractMatrix{Float64},
                          Δt::Float64,

                     # Scratch arrays for RK integration: 
                     ρ_stage::AbstractMatrix{Float64},
                  ρu_X_stage::AbstractMatrix{Float64},
                  ρu_Y_stage::AbstractMatrix{Float64},
                     E_stage::AbstractMatrix{Float64},

                    ρd_stage::AbstractMatrix{Float64},
                ρdud_X_stage::AbstractMatrix{Float64},
                ρdud_Y_stage::AbstractMatrix{Float64},

                        # Conservative & primitive variables @ cell centers:
                          ρc::AbstractMatrix{Float64}, 
                       ρu_Xc::AbstractMatrix{Float64}, 
                       ρu_Yc::AbstractMatrix{Float64},
                          Ec::AbstractMatrix{Float64},         
                        u_Xc::AbstractMatrix{Float64}, 
                        u_Yc::AbstractMatrix{Float64}, 
                          Pc::AbstractMatrix{Float64},
                          
                         ρdc::AbstractMatrix{Float64}, 
                     ρdud_Xc::AbstractMatrix{Float64}, 
                     ρdud_Yc::AbstractMatrix{Float64},     
                       ud_Xc::AbstractMatrix{Float64}, 
                       ud_Yc::AbstractMatrix{Float64},

                        # Weights to convert between cell-averages and cell-centers: 
                    w_i_m1_X::AbstractVector{Float64}, 
                     w_i_0_X::AbstractVector{Float64},
                    w_i_p1_X::AbstractVector{Float64},  
                    w_j_m1_Y::AbstractVector{Float64}, 
                     w_j_0_Y::AbstractVector{Float64}, 
                    w_j_p1_Y::AbstractVector{Float64}, 
                    q_i_m1_X::AbstractVector{Float64}, 
                     q_i_0_X::AbstractVector{Float64}, 
                    q_i_p1_X::AbstractVector{Float64},  
                    q_j_m1_Y::AbstractVector{Float64}, 
                     q_j_0_Y::AbstractVector{Float64}, 
                    q_j_p1_Y::AbstractVector{Float64},  

                   # Weights for WENO3 interface reconstruction:
                 w_plus_i_p1::AbstractVector{Float64},  # WENO3 weights w⁺_{i,1}
                w_minus_i_p1::AbstractVector{Float64},  # WENO3 weights w⁻_{i,1} 
                 w_plus_j_p1::AbstractVector{Float64},  # WENO3 weights w⁺_{j,1}
                w_minus_j_p1::AbstractVector{Float64},  # WENO3 weights w⁻_{j,1}

                       ρ_L_X::AbstractMatrix{Float64},
                       ρ_R_X::AbstractMatrix{Float64},
                     u_X_L_X::AbstractMatrix{Float64},
                     u_X_R_X::AbstractMatrix{Float64},
                     u_Y_L_X::AbstractMatrix{Float64},
                     u_Y_R_X::AbstractMatrix{Float64},
                       P_L_X::AbstractMatrix{Float64},
                       P_R_X::AbstractMatrix{Float64},
                       F_ρ_X::AbstractMatrix{Float64},
                    F_ρu_X_X::AbstractMatrix{Float64},
                    F_ρu_Y_X::AbstractMatrix{Float64},
                       F_E_X::AbstractMatrix{Float64},

                       ρ_L_Y::AbstractMatrix{Float64},
                       ρ_R_Y::AbstractMatrix{Float64},
                     u_Y_L_Y::AbstractMatrix{Float64},
                     u_Y_R_Y::AbstractMatrix{Float64},
                     u_X_L_Y::AbstractMatrix{Float64},
                     u_X_R_Y::AbstractMatrix{Float64},
                       P_L_Y::AbstractMatrix{Float64},
                       P_R_Y::AbstractMatrix{Float64},
                       F_ρ_Y::AbstractMatrix{Float64},
                    F_ρu_X_Y::AbstractMatrix{Float64},
                    F_ρu_Y_Y::AbstractMatrix{Float64},
                       F_E_Y::AbstractMatrix{Float64},
                           P::AbstractMatrix{Float64},
                  u_X_star_X::AbstractMatrix{Float64},
                  u_Y_star_Y::AbstractMatrix{Float64},
            
                      ρd_L_X::AbstractMatrix{Float64},
                      ρd_R_X::AbstractMatrix{Float64},
                    ud_X_L_X::AbstractMatrix{Float64},
                    ud_X_R_X::AbstractMatrix{Float64},
                    ud_Y_L_X::AbstractMatrix{Float64},
                    ud_Y_R_X::AbstractMatrix{Float64},
                      F_ρd_X::AbstractMatrix{Float64},
                  F_ρdud_X_X::AbstractMatrix{Float64},
                  F_ρdud_Y_X::AbstractMatrix{Float64},

                      ρd_L_Y::AbstractMatrix{Float64},
                      ρd_R_Y::AbstractMatrix{Float64},
                    ud_Y_L_Y::AbstractMatrix{Float64},
                    ud_Y_R_Y::AbstractMatrix{Float64},
                    ud_X_L_Y::AbstractMatrix{Float64},
                    ud_X_R_Y::AbstractMatrix{Float64},
                      F_ρd_Y::AbstractMatrix{Float64},
                  F_ρdud_X_Y::AbstractMatrix{Float64},
                  F_ρdud_Y_Y::AbstractMatrix{Float64},      
                                       
                    # Ghost‐padded primitives & grid:
                      ρg_old::AbstractMatrix{Float64}, 
                    u_Xg_old::AbstractMatrix{Float64}, 
                    u_Yg_old::AbstractMatrix{Float64}, 
                      Pg_old::AbstractMatrix{Float64},

                     ρdg_old::AbstractMatrix{Float64}, 
                   ud_Xg_old::AbstractMatrix{Float64}, 
                   ud_Yg_old::AbstractMatrix{Float64}, 

                          Xg::AbstractVector{Float64},
                       XCoMg::AbstractVector{Float64},
                          Yg::AbstractVector{Float64},
                       YCoMg::AbstractVector{Float64},

                    CoordSys::CoordSystem,
                  inv_r_cell::AbstractVector{Float64},
                      Xfirst::Bool,

                    # Shock indicator:
                       Shock::AbstractMatrix{Int},

                    # Arrays for advection of passive scalars:
                     Scalars::AbstractArray{Float64, 3},
                 Scalars_old::AbstractArray{Float64, 3},
               Scalars_stage::AbstractArray{Float64, 3},
                      ρ_save::AbstractMatrix{Float64}, 

                      # BCs:
                   BC_X_left::Symbol  = :Reflect, BC_X_right::Symbol  = :Outflow,
                   BC_Y_left::Symbol  = :Reflect, BC_Y_right::Symbol  = :Outflow 
                   )

    # Save old variables, for use in the 2nd-order gravity kick function:
    @. ρ0       =  ρ  
    @. ρu_X0    =  ρu_X  
    @. ρu_Y0    =  ρu_Y 
    @. E0       =  E
    @. ρd0      =  ρd  

    # Update shock indicators (so it uses latest values from e.g. gas cooling/heating):
    UpdateShockFinder!( Shock, P )

    if Xfirst
        UpdateHydro_StrangSplit_Xfirst!(
            ρ_new, ρu_X_new, ρu_Y_new, E_new,
            ρ, ρu_X, ρu_Y, E,         
            γad, u_X, u_Y,
            
            ρd_new, ρdud_X_new, ρdud_Y_new,
            ρd, ρdud_X, ρdud_Y,       
            ud_X, ud_Y,
            
            inv_ΔV, A_X, A_Y, Δt,

            ρ_stage, ρu_X_stage, ρu_Y_stage, E_stage,

            ρd_stage, ρdud_X_stage, ρdud_Y_stage,

            ρc, ρu_Xc, ρu_Yc, Ec, u_Xc, u_Yc, Pc,     
            
            ρdc, ρdud_Xc, ρdud_Yc, ud_Xc, ud_Yc, 

            w_i_m1_X, w_i_0_X, w_i_p1_X, w_j_m1_Y, w_j_0_Y, w_j_p1_Y, 
            q_i_m1_X, q_i_0_X, q_i_p1_X, q_j_m1_Y, q_j_0_Y, q_j_p1_Y,  
            w_plus_i_p1,  
            w_minus_i_p1,   
            w_plus_j_p1,  
            w_minus_j_p1, 

            ρ_L_X, ρ_R_X, u_X_L_X, u_X_R_X, u_Y_L_X, u_Y_R_X, P_L_X, P_R_X, F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,
            ρ_L_Y, ρ_R_Y, u_Y_L_Y, u_Y_R_Y, u_X_L_Y, u_X_R_Y, P_L_Y, P_R_Y, F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,
            P, u_X_star_X, u_Y_star_Y,

            ρd_L_X, ρd_R_X, ud_X_L_X, ud_X_R_X, ud_Y_L_X, ud_Y_R_X, F_ρd_X, F_ρdud_X_X, F_ρdud_Y_X,
            ρd_L_Y, ρd_R_Y, ud_Y_L_Y, ud_Y_R_Y, ud_X_L_Y, ud_X_R_Y, F_ρd_Y, F_ρdud_X_Y, F_ρdud_Y_Y, 

            ρg_old, u_Xg_old, u_Yg_old, Pg_old,

            ρdg_old, ud_Xg_old, ud_Yg_old,

            Xg, XCoMg, Yg, YCoMg, 
            CoordSys, inv_r_cell,
            Shock,
            Scalars, Scalars_old, Scalars_stage, ρ_save,
            BC_X_left, BC_X_right, 
            BC_Y_left, BC_Y_right )

    else
        UpdateHydro_StrangSplit_Yfirst!(
            ρ_new, ρu_X_new, ρu_Y_new, E_new,
            ρ, ρu_X, ρu_Y, E,         
            γad, u_X, u_Y,
            
            ρd_new, ρdud_X_new, ρdud_Y_new,
            ρd, ρdud_X, ρdud_Y,       
            ud_X, ud_Y,
            
            inv_ΔV, A_X, A_Y, Δt,

            ρ_stage, ρu_X_stage, ρu_Y_stage, E_stage,

            ρd_stage, ρdud_X_stage, ρdud_Y_stage,

            ρc, ρu_Xc, ρu_Yc, Ec, u_Xc, u_Yc, Pc,

            ρdc, ρdud_Xc, ρdud_Yc, ud_Xc, ud_Yc, 

            w_i_m1_X, w_i_0_X, w_i_p1_X, w_j_m1_Y, w_j_0_Y, w_j_p1_Y, 
            q_i_m1_X, q_i_0_X, q_i_p1_X, q_j_m1_Y, q_j_0_Y, q_j_p1_Y, 
            w_plus_i_p1,  
            w_minus_i_p1,   
            w_plus_j_p1,  
            w_minus_j_p1,  

            ρ_L_X, ρ_R_X, u_X_L_X, u_X_R_X, u_Y_L_X, u_Y_R_X, P_L_X, P_R_X, F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,
            ρ_L_Y, ρ_R_Y, u_Y_L_Y, u_Y_R_Y, u_X_L_Y, u_X_R_Y, P_L_Y, P_R_Y, F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,
            P, u_X_star_X, u_Y_star_Y,

            ρd_L_X, ρd_R_X, ud_X_L_X, ud_X_R_X, ud_Y_L_X, ud_Y_R_X, F_ρd_X, F_ρdud_X_X, F_ρdud_Y_X,
            ρd_L_Y, ρd_R_Y, ud_Y_L_Y, ud_Y_R_Y, ud_X_L_Y, ud_X_R_Y, F_ρd_Y, F_ρdud_X_Y, F_ρdud_Y_Y, 

            ρg_old, u_Xg_old, u_Yg_old, Pg_old,

            ρdg_old, ud_Xg_old, ud_Yg_old,

            Xg, XCoMg, Yg, YCoMg, 
            CoordSys, inv_r_cell,
            Shock,
            Scalars, Scalars_old, Scalars_stage, ρ_save,
            BC_X_left, BC_X_right, 
            BC_Y_left, BC_Y_right )

    end

    # Update of primitives, for the later purpose of computing the CFL time-step:
    if Xfirst 
        # Get cell-center conservatives (in X-direction) from 
        # cell-averaged conservatives:
        RecoverPointwiseFromAverages_X!( ρc, ρu_Xc, ρu_Yc, Ec,      # Cell-center values (updated)
                                         ρdc, ρdud_Xc, ρdud_Yc,
                                         ρ,  ρu_X,  ρu_Y,  E, γad,  # Cell-averaged values 
                                         ρd,  ρdud_X,  ρdud_Y,
                                         w_i_m1_X, w_i_0_X, w_i_p1_X,
                                         Shock,
                                         BC_X_left, BC_X_right )

        # Get cell-center primitives (in X-direction) from 
        # cell-center conservatives:
        RecoverPointwisePrimitivesFromConservatives!( ρc, ρu_Xc, ρu_Yc, Ec,
                                                      ρdc, ρdud_Xc, ρdud_Yc,
                                                      u_Xc, u_Yc, Pc, γad,
                                                      ud_Xc, ud_Yc )

        # Get cell-averaged primitives (in X-direction) from cell-center primitives:
        RecoverCellAveragedPrimitives_X!( u_Xc, u_Yc, Pc,    # Cell-centered primitives 
                                          ud_Xc, ud_Yc,
                                          u_X,  u_Y,  P,     # Cell-averaged primitives (updated)
                                          ud_X,  ud_Y,
                                          q_i_m1_X, q_i_0_X, q_i_p1_X,
                                          Shock,
                                          BC_X_left, BC_X_right )

    else
        # Get cell-center conservatives (in Y-direction) from 
        # cell-averaged conservatives:
        RecoverPointwiseFromAverages_Y!( ρc, ρu_Xc, ρu_Yc, Ec,      # Cell-center values (updated)
                                         ρdc, ρdud_Xc, ρdud_Yc,
                                         ρ,  ρu_X,  ρu_Y,  E, γad,  # Cell-averaged values 
                                         ρd,  ρdud_X,  ρdud_Y,
                                         w_j_m1_Y, w_j_0_Y, w_j_p1_Y,
                                         Shock,
                                         BC_Y_left, BC_Y_right )

        # Get cell-center primitives (in Y-direction) from 
        # cell-center conservatives:
        RecoverPointwisePrimitivesFromConservatives!( ρc, ρu_Xc, ρu_Yc, Ec,
                                                      ρdc, ρdud_Xc, ρdud_Yc,
                                                      u_Xc, u_Yc, Pc, γad,
                                                      ud_Xc, ud_Yc )

        # Get cell-averaged primitives (in Y-direction) from cell-center primitives:
        RecoverCellAveragedPrimitives_Y!( u_Xc, u_Yc, Pc,    # Cell-centered primitives 
                                          ud_Xc, ud_Yc, 
                                          u_X,  u_Y,  P,     # Cell-averaged primitives (updated)
                                          ud_X,  ud_Y,
                                          q_j_m1_Y, q_j_0_Y, q_j_p1_Y,
                                          Shock,
                                          BC_Y_left, BC_Y_right )
    end

    # Update shock indicators:
    UpdateShockFinder!( Shock, P )

    return nothing 
end


function UpdateHydro!( Δt::Float64, 
               HydroState::HydrodynamicsState,
                GhostGrid::GhostPaddedGrid,
                   inv_ΔV::AbstractMatrix{Float64},
                      A_X::AbstractMatrix{Float64},
                      A_Y::AbstractMatrix{Float64},
                  # Weights to convert between cell-averages and cell-centers: 
                 w_i_m1_X::AbstractVector{Float64}, 
                  w_i_0_X::AbstractVector{Float64},
                 w_i_p1_X::AbstractVector{Float64},  
                 w_j_m1_Y::AbstractVector{Float64}, 
                  w_j_0_Y::AbstractVector{Float64}, 
                 w_j_p1_Y::AbstractVector{Float64}, 
                 q_i_m1_X::AbstractVector{Float64}, 
                  q_i_0_X::AbstractVector{Float64}, 
                 q_i_p1_X::AbstractVector{Float64},  
                 q_j_m1_Y::AbstractVector{Float64}, 
                  q_j_0_Y::AbstractVector{Float64}, 
                 q_j_p1_Y::AbstractVector{Float64},  

                   # Weights for WENO3 interface reconstruction:
              w_plus_i_p1::AbstractVector{Float64},  # WENO3 weights w⁺_{i,1}
             w_minus_i_p1::AbstractVector{Float64},  # WENO3 weights w⁻_{i,1} 
              w_plus_j_p1::AbstractVector{Float64},  # WENO3 weights w⁺_{j,1}
             w_minus_j_p1::AbstractVector{Float64},  # WENO3 weights w⁻_{j,1}

                 CoordSys::CoordSystem,
               inv_r_cell::AbstractVector{Float64}
                      )

    # Select Strang-split order:    
    HydroState.rfirst_Hydro = !HydroState.rfirst_Hydro   # Change order from its previous value 
    Xfirst                  =  HydroState.rfirst_Hydro

    # Update hydrodynamics: 
    UpdateHydro!( HydroState.ρ_new,
                  HydroState.ρu_X_new,
                  HydroState.ρu_Y_new,
                  HydroState.E_new,
                  HydroState.ρ,
                  HydroState.ρu_X,
                  HydroState.ρu_Y,
                  HydroState.E,
                  HydroState.ρ0,
                  HydroState.ρu_X0,
                  HydroState.ρu_Y0,
                  HydroState.E0,

                  HydroState.γad,
                  HydroState.u_X,
                  HydroState.u_Y,

                  HydroState.ρd_new,
                  HydroState.ρdud_X_new,
                  HydroState.ρdud_Y_new,
                  HydroState.ρd,
                  HydroState.ρdud_X,
                  HydroState.ρdud_Y,
                  HydroState.ρd0,
                  HydroState.ρdud_X0,
                  HydroState.ρdud_Y0,

                  HydroState.ud_X,
                  HydroState.ud_Y,

                  inv_ΔV,
                  A_X,
                  A_Y,
                  Δt,

                  # Scratch arrays for RK integration: 
                  HydroState.ρ_stage,
                  HydroState.ρu_X_stage,
                  HydroState.ρu_Y_stage,
                  HydroState.E_stage,

                  HydroState.ρd_stage,
                  HydroState.ρdud_X_stage,
                  HydroState.ρdud_Y_stage,

                  # Conservative & primitive variables @ cell centers:
                  HydroState.ρc, 
                  HydroState.ρu_Xc, 
                  HydroState.ρu_Yc,
                  HydroState.Ec,         
                  HydroState.u_Xc, 
                  HydroState.u_Yc, 
                  HydroState.Pc,     
                  
                  HydroState.ρdc, 
                  HydroState.ρdud_Xc, 
                  HydroState.ρdud_Yc,  
                  HydroState.ud_Xc, 
                  HydroState.ud_Yc, 

                  # Weights to convert between cell-averages and cell-centers: 
                  w_i_m1_X,  w_i_0_X,  w_i_p1_X,  w_j_m1_Y,  w_j_0_Y,  w_j_p1_Y, 
                  q_i_m1_X,   q_i_0_X,  q_i_p1_X,  q_j_m1_Y, q_j_0_Y,  q_j_p1_Y,  

                   # Weights for WENO3 interface reconstruction:
                 w_plus_i_p1,  # WENO3 weights w⁺_{i,1}
                w_minus_i_p1,  # WENO3 weights w⁻_{i,1} 
                 w_plus_j_p1,  # WENO3 weights w⁺_{j,1}
                w_minus_j_p1,  # WENO3 weights w⁻_{j,1}

                  HydroState.ρ_L_X,
                  HydroState.ρ_R_X,
                  HydroState.u_X_L_X,
                  HydroState.u_X_R_X,
                  HydroState.u_Y_L_X,
                  HydroState.u_Y_R_X,
                  HydroState.P_L_X,
                  HydroState.P_R_X,
                  HydroState.F_ρ_X,
                  HydroState.F_ρu_X_X,
                  HydroState.F_ρu_Y_X,
                  HydroState.F_E_X,

                  HydroState.ρ_L_Y,
                  HydroState.ρ_R_Y,
                  HydroState.u_Y_L_Y,
                  HydroState.u_Y_R_Y,
                  HydroState.u_X_L_Y,
                  HydroState.u_X_R_Y,
                  HydroState.P_L_Y,
                  HydroState.P_R_Y,
                  HydroState.F_ρ_Y,
                  HydroState.F_ρu_X_Y,
                  HydroState.F_ρu_Y_Y,
                  HydroState.F_E_Y,
                  HydroState.P,
                  HydroState.u_X_star_X,
                  HydroState.u_Y_star_Y,
                  
                  HydroState.ρd_L_X,
                  HydroState.ρd_R_X,
                  HydroState.ud_X_L_X,
                  HydroState.ud_X_R_X,
                  HydroState.ud_Y_L_X,
                  HydroState.ud_Y_R_X,
                  HydroState.F_ρd_X,
                  HydroState.F_ρdud_X_X,
                  HydroState.F_ρdud_Y_X,

                  HydroState.ρd_L_Y,
                  HydroState.ρd_R_Y,
                  HydroState.ud_Y_L_Y,
                  HydroState.ud_Y_R_Y,
                  HydroState.ud_X_L_Y,
                  HydroState.ud_X_R_Y,
                  HydroState.F_ρd_Y,
                  HydroState.F_ρdud_X_Y,
                  HydroState.F_ρdud_Y_Y,   
                                       
                  # Ghost‐padded primitives & grid:
                  HydroState.ρg_old, 
                  HydroState.u_Xg_old, 
                  HydroState.u_Yg_old, 
                  HydroState.Pg_old,

                  HydroState.ρdg_old, 
                  HydroState.ud_Xg_old, 
                  HydroState.ud_Yg_old, 

                  GhostGrid.Xg,
                  GhostGrid.XCoMg,
                  GhostGrid.Yg,
                  GhostGrid.YCoMg,

                 CoordSys,
                 inv_r_cell,
                 Xfirst,

                 # Shock indicator:
                 HydroState.Shock,

                 # Arrays for advection of passive scalars:
                 HydroState.Scalars,
                 HydroState.Scalars_old,
                 HydroState.Scalars_stage,
                 HydroState.ρ_save, 

                 # BCs:
                 HydroState.BC_X_left, HydroState.BC_X_right,
                 HydroState.BC_Y_left, HydroState.BC_Y_right 
                   )

    return nothing 
end


function GravitySolveAndKick!( Δt::Float64,
                       HydroState::HydrodynamicsState,
                        GravState::GravityState,
                       GravCoeffs::GravityCoeffs,
                               ΔV::AbstractArray{Float64},
                             XCoM::AbstractArray{Float64}, 
                             YCoM::AbstractArray{Float64} )

    # Unpack some hydro & dust variables for convenience:
    ρu_X     = HydroState.ρu_X
    ρu_Y     = HydroState.ρu_Y 
    ρ        = HydroState.ρ       # Latest densities from after hydro advecton step 
    ρ0       = HydroState.ρ0      # Old densities from *before* hydro advecton step 
    E        = HydroState.E
    ρdud_X   = HydroState.ρdud_X
    ρdud_Y   = HydroState.ρdud_Y
    ρd       = HydroState.ρd
    ρd0      = HydroState.ρd0
    BC_Z_bot = HydroState.BC_Y_left

    # Unpack non-gas gravitating mass density:
    ρ_bg       = GravState.ρ_bg
    ρtot       = GravState.ρtot

    # Grid size & intervals: 
    N_X, N_Y   = size( ρ )
    dN_X       = 1:N_X
    dN_Y       = 1:N_Y 

    # Fill ρ0 with the estimated ρ^(n+1/2), similarly for dust:
    @. ρ0      = 0.5 * ( ρ0  + ρ )
    @. ρd0     = 0.5 * ( ρd0 + ρd )

    # Check whether we should update Φ or not:
    ShouldWeUpdate = ShouldWeUpdateΦ( GravState.Φ, 
                                      ρ0, ρd, ρ_bg,
                                      GravCoeffs.RPoiss_ijp, GravCoeffs.RPoiss_ij, GravCoeffs.RPoiss_ijm,
                                      GravCoeffs.ZPoiss_ijp, GravCoeffs.ZPoiss_ij, GravCoeffs.ZPoiss_ijm )

    # If ShouldWeUpdate = true, we solve the Poisson equation for the updated 
    # potential Φ, and the updated gravitational acceleration:
    if ShouldWeUpdate 
        Update_Φ!( GravState.Φ, GravState.Φ_half, GravState.Φ_old, GravState.Φ_snapshot, 
                   ρtot, ρ0, ρd0, ρ_bg,
                   GravCoeffs.RPoiss_ijp, GravCoeffs.RPoiss_ij, GravCoeffs.RPoiss_ijm,
                   GravCoeffs.ZPoiss_ijp, GravCoeffs.ZPoiss_ij, GravCoeffs.ZPoiss_ijm, 
                   GravCoeffs.Δτ_max, GravCoeffs.Δτ_min,
                   ΔV, XCoM, YCoM, GravState.Q, GravState.P, GravState.ℓmax, BC_Z_bot )
                           
        Update_∇Φ!( GravState.∂Φ∂r, GravState.∂Φ∂Z, GravState.Φ, 
                    GravState.f_gravity_r, GravState.f_gravity_Z, ρ, XCoM, YCoM )
    end

    # Finally, we perform a second-order in time gravity kick to the
    # hydro variables:
    @threads for j in dN_Y 
        @inbounds for i in dN_X
            Inv_2ρ       = 1 / (2 * ρ[i,j]) # Repeated quantity below

            # Get internal energy from after the advection step:
            e_adv        =  E[i,j]  -  (ρu_X[i,j]^2  +  ρu_Y[i,j]^2) * Inv_2ρ 

            # Update momenta to second-order accuracy, using ρ0 = ρ^(n+1/2): 
            ρu_X[i,j]   -=  ρ0[i,j] * GravState.∂Φ∂r[i,j] * Δt 
            ρu_Y[i,j]   -=  ρ0[i,j] * GravState.∂Φ∂Z[i,j] * Δt 

            # Update gas energy, ensuring the internal energy remains unchanged:
            E[i,j]       =  e_adv  +  (ρu_X[i,j]^2  +  ρu_Y[i,j]^2) * Inv_2ρ  
            
            # Update dust momenta:
            ρdud_X[i,j] -=  ρd0[i,j] * GravState.∂Φ∂r[i,j] * Δt 
            ρdud_Y[i,j] -=  ρd0[i,j] * GravState.∂Φ∂Z[i,j] * Δt 

        end
    end

    return nothing 
end


""" This function *only* does a gravity kick to the hydro,
    i.e. no Poisson equation solve. It is mainly used for 
    hydro tests with a fixed potential Φ.
    
    NOTE: It ignores dust, since we don't want to kick dust 
          during a hydro test. """
function GravityKick!( Δt::Float64,
                     ρu_X::AbstractMatrix{Float64}, 
                     ρu_Y::AbstractMatrix{Float64},
                        ρ::AbstractMatrix{Float64}, # Latest densities from after hydro advecton step 
                       ρ0::AbstractMatrix{Float64}, # Old densities from *before* hydro advecton step 
                        E::AbstractMatrix{Float64},

                     ∂Φ∂X::AbstractMatrix{Float64},
                     ∂Φ∂Y::AbstractMatrix{Float64}
                      )

    # Grid size & intervals: 
    N_X, N_Y = size( ρ )
    dN_X     = 1:N_X
    dN_Y     = 1:N_Y 

    # Fill ρ0 with the estimated ρ^(n+1/2):
    @. ρ0    = 0.5 * ( ρ0 + ρ )

    # Finally, we perform a second-order in time gravity kick to the
    # hydro variables:
    @threads for j in dN_Y 
        @inbounds for i in dN_X
            Inv_2ρ       = 1 / (2 * ρ[i,j]) # Repeated quantity below

            # Get internal energy from after the advection step:
            e_adv        =  E[i,j]  -  (ρu_X[i,j]^2  +  ρu_Y[i,j]^2) * Inv_2ρ

            # Update momenta to second-order accuracy, using ρ0 = ρ^(n+1/2): 
            ρu_X[i,j]   -=  ρ0[i,j] * ∂Φ∂X[i,j] * Δt 
            ρu_Y[i,j]   -=  ρ0[i,j] * ∂Φ∂Y[i,j] * Δt 

            # Update gas energy, ensuring the internal energy remains unchanged:
            E[i,j]       =  e_adv  +  (ρu_X[i,j]^2  +  ρu_Y[i,j]^2) * Inv_2ρ  

        end
    end

    return nothing 
end


###############################################################
#######                                                 #######
#######            M I S C E L L A N E O U S            #######
#######                                                 #######
###############################################################

# Some other useful functions for setting up arrays, 
# and to setup tests of the hydro code. 


@inline function UpdateTotalMass!( HydroState::HydrodynamicsState,
                                           ΔV::AbstractMatrix{Float64}
                                  )
    
    # Initialize total mass to zero before we compute it:
    M = 0.0

    # Grid size: 
    N_r, N_Z = size( ΔV )
    dN_r     = 1:N_r 
    dN_Z     = 1:N_Z 

    # Add mass from all cells:
    for j in dN_Z 
        @inbounds @simd for i in dN_r 
            # Add mass contribution from cell:
            M += ΔV[i,j] * HydroState.ρ[i,j]
        end
    end

    # Store computed mass:
    HydroState.M_simvol = M 

    return nothing 
end


""" Initialize all the hydro arrays (cell-centered, interface,
    slope and flux) for an (X, Y) grid.  """
function GenerateHydroArrays( X::AbstractVector{Float64},   
                           XCoM::AbstractVector{Float64},
                              Y::AbstractVector{Float64},   
                           YCoM::AbstractVector{Float64},
                              γ::Float64 = γmon,
                      N_Scalars::Int     = 6,
                      BC_X_left::Symbol  = :Reflect,
                     BC_X_right::Symbol  = :Outflow,
                      BC_Y_left::Symbol  = :Reflect, 
                     BC_Y_right::Symbol  = :Outflow 
                             )
    
    # Grid size:
    N_X = length(X)-1 
    N_Y = length(Y)-1

    @assert N_X == length(XCoM) "We must have length(X) = length(XCoM) + 1"
    @assert N_Y == length(YCoM) "We must have length(Y) = length(YCoM) + 1"

    # Cell-averaged primitive & conserved variables:
    ρ          = zeros(Float64, N_X,   N_Y)
    ρu_X       = zeros(Float64, N_X,   N_Y)
    ρu_Y       = zeros(Float64, N_X,   N_Y)
    E          = zeros(Float64, N_X,   N_Y)
    ρ_new      = zeros(Float64, N_X,   N_Y)
    ρu_X_new   = zeros(Float64, N_X,   N_Y)
    ρu_Y_new   = zeros(Float64, N_X,   N_Y)
    E_new      = zeros(Float64, N_X,   N_Y)
    ρ0         = zeros(Float64, N_X,   N_Y)
    ρu_X0      = zeros(Float64, N_X,   N_Y)
    ρu_Y0      = zeros(Float64, N_X,   N_Y)
    E0         = zeros(Float64, N_X,   N_Y)
    P          = zeros(Float64, N_X,   N_Y)
    u_X        = zeros(Float64, N_X,   N_Y)
    u_Y        = zeros(Float64, N_X,   N_Y)

    ρd         = zeros(Float64, N_X,   N_Y)
    ρdud_X     = zeros(Float64, N_X,   N_Y)
    ρdud_Y     = zeros(Float64, N_X,   N_Y)
    ρd_new     = zeros(Float64, N_X,   N_Y)
    ρdud_X_new = zeros(Float64, N_X,   N_Y)
    ρdud_Y_new = zeros(Float64, N_X,   N_Y)
    ρd0        = zeros(Float64, N_X,   N_Y)
    ρdud_X0    = zeros(Float64, N_X,   N_Y)
    ρdud_Y0    = zeros(Float64, N_X,   N_Y)
    ud_X       = zeros(Float64, N_X,   N_Y)
    ud_Y       = zeros(Float64, N_X,   N_Y)

    # Cell-centered primitives:
    ρc         = zeros(Float64, N_X,   N_Y)
    ρu_Xc      = zeros(Float64, N_X,   N_Y)
    ρu_Yc      = zeros(Float64, N_X,   N_Y)
    Ec         = zeros(Float64, N_X,   N_Y)   
    u_Xc       = zeros(Float64, N_X,   N_Y)
    u_Yc       = zeros(Float64, N_X,   N_Y) 
    Pc         = zeros(Float64, N_X,   N_Y) 

    ρdc        = zeros(Float64, N_X,   N_Y)
    ρdud_Xc    = zeros(Float64, N_X,   N_Y)
    ρdud_Yc    = zeros(Float64, N_X,   N_Y) 
    ud_Xc      = zeros(Float64, N_X,   N_Y)
    ud_Yc      = zeros(Float64, N_X,   N_Y) 

    # Arrays for storing pre-RK update state:
    ρ_stage      = zeros(Float64, N_X,   N_Y)
    ρu_X_stage   = zeros(Float64, N_X,   N_Y)
    ρu_Y_stage   = zeros(Float64, N_X,   N_Y)
    E_stage      = zeros(Float64, N_X,   N_Y)

    ρd_stage     = zeros(Float64, N_X,   N_Y)
    ρdud_X_stage = zeros(Float64, N_X,   N_Y)
    ρdud_Y_stage = zeros(Float64, N_X,   N_Y)

    # Adiabatic index:
    γad        = fill(γ, size(ρ))

    # Shock indicator:
    Shock      = zeros(Int,     N_X,   N_Y)

    # Ghost-padded coordinates/cells:
    Xg         = zeros(Float64, (N_X + 1)  + 2*ng)        # Ghost-padded X-interfaces  
    XCoMg      = zeros(Float64,  N_X       + 2*ng)        # Ghost-padded X cell centers
    Yg         = zeros(Float64, (N_Y + 1)  + 2*ng)        # Ghost-padded Y-interfaces   
    YCoMg      = zeros(Float64,  N_Y       + 2*ng)        # Ghost-padded Y cell centers

    # Fill ghost-padded coordinates/cells: 
    Xg[ng+1 : ng+N_X+1]    .= X
    Yg[ng+1 : ng+N_Y+1]    .= Y 

    XCoMg[ng+1 : ng+N_X]   .= XCoM
    YCoMg[ng+1 : ng+N_Y]   .= YCoM

    # Linear extrapolation:
    ΔX_L       = X[2]   - X[1]
    ΔX_R       = X[end] - X[end-1]   
    ΔY_L       = Y[2]   - Y[1]
    ΔY_R       = Y[end] - Y[end-1]  
  
    @inbounds for k in dng 
      Xg[ng+1 - k]      = X[1]   - k * ΔX_L
      Xg[ng+N_X+1 + k]  = X[end] + k * ΔX_R
      Yg[ng+1 - k]      = Y[1]   - k * ΔY_L
      Yg[ng+N_Y+1 + k]  = Y[end] + k * ΔY_R
    end

    # Linear extrapolation for ghost cell centers: 
    ΔXC_L = XCoM[2]      - XCoM[1]
    ΔXC_R = XCoM[end]    - XCoM[end-1]
    ΔYC_L = YCoM[2]      - YCoM[1]
    ΔYC_R = YCoM[end]    - YCoM[end-1]

    @inbounds for k in dng
        XCoMg[ng+1 - k]     = XCoM[1]   - k * ΔXC_L
        XCoMg[ng+N_X + k]   = XCoM[end] + k * ΔXC_R
        YCoMg[ng+1 - k]     = YCoM[1]   - k * ΔYC_L
        YCoMg[ng+N_Y + k]   = YCoM[end] + k * ΔYC_R
    end

    # Ghost-padded arrays:
    ρg_old     = zeros(Float64, N_X + 2*ng,   N_Y + 2*ng)
    Pg_old     = zeros(Float64, N_X + 2*ng,   N_Y + 2*ng)
    u_Xg_old   = zeros(Float64, N_X + 2*ng,   N_Y + 2*ng)
    u_Yg_old   = zeros(Float64, N_X + 2*ng,   N_Y + 2*ng)

    ρdg_old    = zeros(Float64, N_X + 2*ng,   N_Y + 2*ng)
    ud_Xg_old  = zeros(Float64, N_X + 2*ng,   N_Y + 2*ng)
    ud_Yg_old  = zeros(Float64, N_X + 2*ng,   N_Y + 2*ng)

    # X-direction interface values (N_X + 1 × N_Y)
    ρ_L_X      = zeros(Float64, N_X+1, N_Y)
    ρ_R_X      = zeros(Float64, N_X+1, N_Y)
    u_X_L_X    = zeros(Float64, N_X+1, N_Y)
    u_X_R_X    = zeros(Float64, N_X+1, N_Y)
    u_Y_L_X    = zeros(Float64, N_X+1, N_Y)
    u_Y_R_X    = zeros(Float64, N_X+1, N_Y)
    P_L_X      = zeros(Float64, N_X+1, N_Y)
    P_R_X      = zeros(Float64, N_X+1, N_Y)

    ρd_L_X     = zeros(Float64, N_X+1, N_Y)
    ρd_R_X     = zeros(Float64, N_X+1, N_Y)
    ud_X_L_X   = zeros(Float64, N_X+1, N_Y)
    ud_X_R_X   = zeros(Float64, N_X+1, N_Y)
    ud_Y_L_X   = zeros(Float64, N_X+1, N_Y)
    ud_Y_R_X   = zeros(Float64, N_X+1, N_Y)

    # X-direction fluxes (N_X+1 × N_Y)
    F_ρ_X      = zeros(Float64, N_X+1, N_Y)
    F_ρu_X_X   = zeros(Float64, N_X+1, N_Y)
    F_ρu_Y_X   = zeros(Float64, N_X+1, N_Y)
    F_E_X      = zeros(Float64, N_X+1, N_Y)

    F_ρd_X     = zeros(Float64, N_X+1, N_Y)
    F_ρdud_X_X = zeros(Float64, N_X+1, N_Y)
    F_ρdud_Y_X = zeros(Float64, N_X+1, N_Y)

    # Y-direction interface values (N_X × N_Y + 1)
    ρ_L_Y      = zeros(Float64, N_X, N_Y+1)
    ρ_R_Y      = zeros(Float64, N_X, N_Y+1)
    u_Y_L_Y    = zeros(Float64, N_X, N_Y+1)
    u_Y_R_Y    = zeros(Float64, N_X, N_Y+1)
    u_X_L_Y    = zeros(Float64, N_X, N_Y+1)
    u_X_R_Y    = zeros(Float64, N_X, N_Y+1)
    P_L_Y      = zeros(Float64, N_X, N_Y+1)
    P_R_Y      = zeros(Float64, N_X, N_Y+1)

    ρd_L_Y     = zeros(Float64, N_X, N_Y+1)
    ρd_R_Y     = zeros(Float64, N_X, N_Y+1)
    ud_Y_L_Y   = zeros(Float64, N_X, N_Y+1)
    ud_Y_R_Y   = zeros(Float64, N_X, N_Y+1)
    ud_X_L_Y   = zeros(Float64, N_X, N_Y+1)
    ud_X_R_Y   = zeros(Float64, N_X, N_Y+1)

    # Y-direction fluxes (N_X × N_Y + 1)
    F_ρ_Y      = zeros(Float64, N_X, N_Y+1)
    F_ρu_X_Y   = zeros(Float64, N_X, N_Y+1)
    F_ρu_Y_Y   = zeros(Float64, N_X, N_Y+1)
    F_E_Y      = zeros(Float64, N_X, N_Y+1)

    F_ρd_Y     = zeros(Float64, N_X, N_Y+1)
    F_ρdud_X_Y = zeros(Float64, N_X, N_Y+1)
    F_ρdud_Y_Y = zeros(Float64, N_X, N_Y+1)

    # Contact velocities (N_X + 1 × N_Y and N_X × N_Y + 1)
    u_X_star_X = zeros(Float64, N_X+1, N_Y)
    u_Y_star_Y = zeros(Float64, N_X, N_Y+1)

    # Arrays related to ∇u (only needed for RHD):
    ∂u_X_∂X    = zeros(Float64, N_X, N_Y)
    ∂u_X_∂Y    = zeros(Float64, N_X, N_Y)
    ∂u_Y_∂X    = zeros(Float64, N_X, N_Y)
    ∂u_Y_∂Y    = zeros(Float64, N_X, N_Y)
    u_r_over_r = zeros(Float64, N_X, N_Y)

    # Gas internal energy density e & temperature T 
    # (needed for thermochemistry):
    e          = zeros(Float64, N_X, N_Y)
    T          = zeros(Float64, N_X, N_Y)

    # Strang split order:
    rfirst_Hydro = true 

    # Total mass in simulation volume:
    M_simvol = 0.0

    # Arrays needed for advection of passive scalars:
    Scalars       = zeros(Float64, N_X, N_Y, N_Scalars)
    Scalars_old   = zeros(Float64, N_X, N_Y, N_Scalars)
    Scalars_stage = zeros(Float64, N_X, N_Y, N_Scalars)
    ρ_save        = zeros(Float64, N_X,   N_Y)

    # Create structs:
    hydrostate = HydrodynamicsState( ρ_new, ρu_X_new, ρu_Y_new, E_new, ρ, ρu_X, ρu_Y, E, 
                                     ρ0, ρu_X0, ρu_Y0, E0, ρ_stage, ρu_X_stage, ρu_Y_stage, E_stage,
                                     γad, u_X, u_Y, ρ_L_X, ρ_R_X,
                                     u_X_L_X, u_X_R_X, u_Y_L_X, u_Y_R_X, P_L_X, P_R_X,
                                     F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X, ρ_L_Y, ρ_R_Y, u_Y_L_Y,
                                     u_Y_R_Y, u_X_L_Y, u_X_R_Y, P_L_Y, P_R_Y, F_ρ_Y, F_ρu_X_Y, 
                                     F_ρu_Y_Y, F_E_Y, 
                                     u_X_star_X, u_Y_star_Y,
                                     ∂u_X_∂X, ∂u_X_∂Y, ∂u_Y_∂X, ∂u_Y_∂Y, u_r_over_r,
                                     P, e, T, ρg_old, u_Xg_old, u_Yg_old, Pg_old,
                                     ρc, ρu_Xc, ρu_Yc, Ec, u_Xc, u_Yc, Pc,
                                     Shock,
                                     BC_X_left, BC_X_right, BC_Y_left, BC_Y_right,

                                     ρd_new, ρdud_X_new, ρdud_Y_new, ρd, ρdud_X, ρdud_Y, 
                                     ρd0, ρdud_X0, ρdud_Y0, ρd_stage, ρdud_X_stage, ρdud_Y_stage, 
                                     ud_X, ud_Y, 
                                     ρd_L_X, ρd_R_X, ud_X_L_X, ud_X_R_X, ud_Y_L_X, ud_Y_R_X, 
                                     F_ρd_X, F_ρdud_X_X, F_ρdud_Y_X, 
                                     ρd_L_Y, ρd_R_Y, ud_Y_L_Y, ud_Y_R_Y, ud_X_L_Y, ud_X_R_Y, 
                                     F_ρd_Y, F_ρdud_X_Y, F_ρdud_Y_Y, 
                                     ρdg_old, ud_Xg_old, ud_Yg_old, 
                                     ρdc, ρdud_Xc, ρdud_Yc, ud_Xc, ud_Yc, 

                                     Scalars, Scalars_old, Scalars_stage, ρ_save,
                                     rfirst_Hydro, M_simvol )

    ghostgrid  = GhostPaddedGrid( Xg, XCoMg, Yg, YCoMg )
  

    return hydrostate, ghostgrid
end

