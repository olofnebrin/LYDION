
module Hydrodynamics

""" MODULE:         Hydrodynamics

    PURPOSE:        This module contains the basic 2D hydrodynamics code, relying on a 
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
    
    """

__precompile__()

include( "./ConstantsAndFunctions.jl" )
include( "./WENO3.jl" )
include( "./PoissonGravitySolver.jl" )
using .ConstantsAndFunctions # Needed constants 
using .WENO3                 # For WENO3 cell reconstruction 
using .PoissonGravitySolver  # For the gravity solver & kick 
using Base.Threads           # For CPU acceleration
using Printf


export GetCFLTimeStep, GetGravityTimeStep, UpdateHydro!, 
       GravitySolveAndKick!, GenerateHydroArrays

###############################################################
#######                                                 #######
#######           H Y D R O  S E T T I N G S            #######
#######                                                 #######
###############################################################

# Below are the main settings controlling the hydro solver,
# which you can change.

const CFL::Float64         =  0.5       # Default Courant factor                           (< 1, default: 0.5)
const Cgrav::Float64       =  0.1       # Sets the gravity-limited time-step               (default: 0.1)
const Crad::Float64        =  0.1       # Sets the stellar rad. presure-limited time-step  (default: 0.1)
const ρ_floor::Float64     =  1e-7 * mH # Gas density floor                                (default: 10⁻⁷ mH per cm⁻³)
const P_floor::Float64     =  1e-30     # Gas pressure floor                               (default: 1e-30 in cgs units)
const γrad::Float64        =  4 / 3     # Adiabatic index for radiation 

export CFL, FluxChoice, ρ_floor  

###############################################################
#######                                                 #######
#######             H Y D R O  S T R U C T S            #######
#######                                                 #######
###############################################################

mutable struct HydrodynamicsState{T<:AbstractFloat}
    # Updated hydro state variables:
    ρ_new::Array{T, 2} 
    ρu_X_new::Array{T, 2} 
    ρu_Y_new::Array{T, 2}
    E_new::Array{T, 2}
    # "Old" hydro state variables:
    ρ::Array{T, 2}
    ρu_X::Array{T, 2}
    ρu_Y::Array{T, 2}     
    E::Array{T, 2}
    # Arrays for storing hydro state for use in the gravity solver:
    ρ0::Array{T, 2}
    ρu_X0::Array{T, 2}    
    ρu_Y0::Array{T, 2}
    E0::Array{T, 2}
    # Gas velocities & adiabatic indices:
    γad::Array{T, 2}
    u_r::Array{T, 2}
    u_Z::Array{T, 2} 
    # Reconstructed X-interface primitive variables:
    ρ_L_X::Array{T, 2}
    ρ_R_X::Array{T, 2}
    u_X_L_X::Array{T, 2} 
    u_X_R_X::Array{T, 2} 
    u_Y_L_X::Array{T, 2} 
    u_Y_R_X::Array{T, 2} 
    P_L_X::Array{T, 2}
    P_R_X::Array{T, 2}
    # Interface X-fluxes:
    F_ρ_X::Array{T, 2},
    F_ρu_X_X::Array{T, 2}
    F_ρu_Y_X::Array{T, 2}
    F_E_X::Array{T, 2}
    # Reconstructed Y-interface primitive variables:
    ρ_L_Y::Array{T, 2} 
    ρ_R_Y::Array{T, 2} 
    u_Y_L_Y::Array{T, 2} 
    u_Y_R_Y::Array{T, 2}
    u_X_L_Y::Array{T, 2}
    u_X_R_Y::Array{T, 2} 
    P_L_Y::Array{T, 2} 
    P_R_Y::Array{T, 2}
    # Interface Y-fluxes:
    F_ρ_Y::Array{T, 2} 
    F_ρu_X_Y::Array{T, 2} 
    F_ρu_Y_Y::Array{T, 2} 
    F_E_Y::Array{T, 2}
    # Gas pressure & internal energy density e:
    P::Array{T, 2}
    e::Array{T, 2}
    # Ghost-padded variables:
    ρg_old::Array{T, 2} 
    u_Xg_old::Array{T, 2} 
    u_Yg_old::Array{T, 2} 
    Pg_old::Array{T, 2}
end


struct GhostPaddedGrid{T<:AbstractFloat}
    Xg::Array{T, 2}
    XCoMg::Array{T, 2}
    Yg::Array{T, 2}
    YCoMg::Array{T, 2}
end

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
    u_X  = ρu_X / ρ
    u_Y  = ρu_Y / ρ
    u_sq = u_X * u_X  +  u_Y * u_Y
    P    = (γad - 1) * ( E  -  ρ * u_sq / 2 )
    return u_X, u_Y, P
end


""" Converts from primitive variables (ρ, u_X, u_Y, P) to 
    conservative ones (ρu_X, ρu_Y, E). """
@inline function GetConservative( ρ::Float64, # Gas density 
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


""" The default CFL time-step. Computes min( Δr / (cs + u) ),
    over all cells. To avoid unnecessary allocations, this is 
    done in a for loop. """
@inline function GetCFLTimeStep(  X::AbstractVector{Float64}, # Interface r-values
                                  Y::AbstractVector{Float64}, # Interface Z-values
                                  ρ::AbstractMatrix{Float64}, # Gas density 
                                  P::AbstractMatrix{Float64}, # Gas pressure 
                                u_X::AbstractMatrix{Float64}, # Gas velocity in r-direction
                                u_Y::AbstractMatrix{Float64}, # Gas velocity in Z-direction 
                                γad::AbstractMatrix{Float64}  # Adiabatic index 
                                 )

    N_X, N_Y    = size( ρ )  # Grid size
    dN_X        = 1:N_X
    dN_Y        = 1:N_Y
    @assert length(X) == N_X + 1 "X must have N_X + 1 entries"
    @assert length(Y) == N_Y + 1 "Y must have N_Y + 1 entries"

    min_ΔtHydro = Inf64  # Initialize the minimum local time-scale

    @inbounds for j in dN_Y
        ΔY                = Y[j+1] - Y[j] # Vertical cell width
        local_min_ΔtHydro = min_ΔtHydro   # Local variable for tracking smallest CFL time

        @simd for i in dN_X
            # Compute the horizontal cell width (same for all j-indices):
            ΔX = X[i+1] - X[i]

            # Compute the sound speed in the cell:
            if P[i,j] < 0.0
                error("Negative P at i = $(i), j = $(j).")
            elseif ρ[i,j] < 0.0
                error("Negative ρ at i = $(i), j = $(j).")
            end
            cs = sqrt( γad[i,j] * P[i,j] / ρ[i,j] )

            # Compute the local CFL time-scale:
            Δt = min( ΔX / (cs + abs(u_X[i,j])), 
                      ΔY / (cs + abs(u_Y[i,j])) )
            
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


function GetGravityTimeStep( X::AbstractVector{Float64}, # Interface X-values
                             Y::AbstractVector{Float64}, # Interface Y-values
                          ∂Φ∂X::AbstractMatrix{Float64}, # X-component of ∇Φ  
                          ∂Φ∂Y::AbstractMatrix{Float64}, # Y-component of ∇Φ 
                             )

    N_X, N_Y = size( ∂Φ∂X )
    dN_X     = 1:N_X 
    dN_Y     = 1:N_Y 

    min_Δt = Inf64  # Initialize the minimum local time-scale

    @inbounds for j in dN_Y
        ΔY           = Y[j+1] - Y[j] # Vertical cell width
        local_min_Δt = min_Δt   # Local variable for tracking smallest time-step 

        @simd for i in dN_X 

            # Compute the horizontal cell width (same for all j-indices):
            ΔX = X[i+1] - X[i]

            # Compute the local acceleration time-scale:
            Δt = min( sqrt( 2 * ΔX / abs(∂Φ∂X[i,j]) ), 
                      sqrt( 2 * ΔY / abs(∂Φ∂Y[i,j]) ) )
            
            if Δt < local_min_Δt
                # Update the smallest time-scale found:
                local_min_Δt = Δt 
            end

        end

        min_Δt = min( min_Δt, local_min_Δt )
    end

    return Cgrav * min_Δt
end


###############################################################
#######                                                 #######
#######      C E L L  R E C O N S T R U C T I O N       #######
#######                                                 #######
###############################################################


const ng::Int64 = 2    # Number of ghost layers per side 
const dng       = 1:ng # Convenient interval, often needed below 

""" This function updates primitive variables in the 
    ghost cells, needed for proper boundary conditions & 
    flux calculations at those boundaries. """
@views function FillGhosts!( ρg::AbstractMatrix{Float64},
                           u_Xg::AbstractMatrix{Float64},
                           u_Yg::AbstractMatrix{Float64},
                             Pg::AbstractMatrix{Float64} )


    @inbounds begin
        ######## r = 0  (reflective) ########
        for k in dng
            # even parity (scalars and tangential component)
            ρg[ng+1-k, :]   .=    ρg[ng+k, :]
            Pg[ng+1-k, :]   .=    Pg[ng+k, :]
            u_Yg[ng+1-k, :] .=    u_Yg[ng+k, :]
            # odd parity (normal component)
            u_Xg[ng+1-k, :] .= .- u_Xg[ng+k, :]
        end
        # (optional) enforce machine-zero radial velocity on the wall interface
        #u_Xg[ng, :] .= 0.0

        ######## r = r_max (outflow, zero-gradient, forbid inflow) ########
        ρg[end-ng+1:end, :]   .= ρg[end-ng:end-ng, :]
        Pg[end-ng+1:end, :]   .= Pg[end-ng:end-ng, :]
        u_Yg[end-ng+1:end, :] .= u_Yg[end-ng:end-ng, :]
        # outward normal is +r, so inflow is u_X < 0 ⇒ clamp with max
        u_Xg[end-ng+1:end, :] .= max.(u_Xg[end-ng:end-ng, :], 0.0)
        #u_Xg[end-ng+1:end, :] .= - u_Xg[end-ng:end-ng, :]

        ######## Z = 0 (reflective) ########
        for k in dng
            ρg[:, ng+1-k]   .=    ρg[:, ng+k]
            Pg[:, ng+1-k]   .=    Pg[:, ng+k]
            u_Xg[:, ng+1-k] .=    u_Xg[:, ng+k]   # tangential
            u_Yg[:, ng+1-k] .= .- u_Yg[:, ng+k]   # normal
        end
        # (optional) enforce machine-zero normal velocity on the wall interface
        #u_Yg[:, ng] .= 0.0

        ######## Z = Z_max (outflow, zero-gradient, forbid inflow) ########
        ρg[:, end-ng+1:end]   .= ρg[:, end-ng:end-ng]
        Pg[:, end-ng+1:end]   .= Pg[:, end-ng:end-ng]
        u_Xg[:, end-ng+1:end] .= u_Xg[:, end-ng:end-ng]
        # outward normal is +z, so inflow is u_Y < 0 ⇒ clamp with max (not min)
        u_Yg[:, end-ng+1:end] .= max.(u_Yg[:, end-ng:end-ng], 0.0)
        #u_Yg[:, end-ng+1:end] .= - u_Yg[:, end-ng:end-ng]
    end
    return nothing
end


""" This function updates the X-interface values 
    of primitive variables W ∈ (ρ, u, P), using
    a third-order accurate WENO3 approximation. 
    """
function UpdateInterfaceValues_X!( WL_X::AbstractMatrix{Float64},  # Left interface value of the primitive variable  
                                   WR_X::AbstractMatrix{Float64},  # Right interface value of the primitive variable  
                                     Wg::AbstractMatrix{Float64},  # Ghost-padded primitive variable, e.g. ρg(X,Y) (size N_X + 4 × N_Y + 4)
                                     Xg::AbstractVector{Float64},  # Ghost-padded X-interfaces                     (length N_X + 5)
                                  XCoMg::AbstractVector{Float64}   # Ghost-padded X center-of-mass                 (length N_X + 4)
                                      )

    # Grid size:
    N_X = size( Wg, 1 ) - 2*ng
    N_Y = size( Wg, 2 ) - 2*ng


    @threads for j in 1:N_Y
        j0 = j + ng  # Ghost-padded j-index 

        @inbounds @simd for i in 2:N_X
            i0  = i  + ng     # Ghost-padded face index 
            k   = i0 - 1      # Ghost-padded left cell index 

            W_im2 = Wg[k-1, j0]
            W_im1 = Wg[k,   j0]
            W_i   = Wg[k+1, j0]
            W_ip1 = Wg[k+2, j0]

            X_im1 = Xg[i0-1]
            X_i   = Xg[i0]
            X_ip1 = Xg[i0+1]

            XCoM_im2 = XCoMg[k-1]
            XCoM_im1 = XCoMg[k]
            XCoM_i   = XCoMg[k+1]
            XCoM_ip1 = XCoMg[k+2]

            # Left & right states at interface i-1/2:
            WL_X[i,j], WR_X[i,j] = Get_QL_and_QR(  W_im2, W_im1, W_i, W_ip1,
                                                   X_im1, X_i,   X_ip1,
                                                   XCoM_im2, XCoM_im1, XCoM_i, XCoM_ip1,
                                                   N_X ) 

        end

        # Boundary interfaces, using ghost cells (which in the hydro update are already filled):
        @inbounds begin
            # Left boundary i = 1:
            WL_X[1, j]     = Wg[ng, j0]
            WR_X[1, j]     = Wg[ng+1, j0]
            # Right boundary i = N_X+1:
            WL_X[N_X+1, j] = Wg[N_X+ng, j0]    
            WR_X[N_X+1, j] = Wg[N_X+ng+1, j0]
        end
    end
    
    return nothing
end



""" This function updates the Y-interface values 
    of primitive variables W ∈ (ρ, u, P), using
    a third-order accurate WENO3 approximation. """
function UpdateInterfaceValues_Y!( WL_Y::AbstractMatrix{Float64},  # Right interface value of the primitive variable  
                                   WR_Y::AbstractMatrix{Float64},  # Left interface value of the primitive variable  
                                     Wg::AbstractMatrix{Float64},  # Ghost-padded primitive variable, e.g. ρg(X,Y) (size N_X + 4 × N_Y + 4)
                                     Yg::AbstractVector{Float64},  # Ghost-padded Y-interfaces                     (length N_Y + 5)
                                  YCoMg::AbstractVector{Float64}   # Ghost-padded Y center-of-mass                 (length N_Y + 4)
                                  )

    # Grid size:
    N_X = size( Wg, 1 ) - 2*ng
    N_Y = size( Wg, 2 ) - 2*ng


    @threads for j in 2:N_Y                  
        j0 = j + ng                        # Ghost-padded interface index

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

            W_im2 = Wg[i0, c_m2]
            W_im1 = Wg[i0, c_m1]
            W_i   = Wg[i0, c_0]
            W_ip1 = Wg[i0, c_p1]

            # Left & right states at Y_{j-1/2}:
            WL_Y[i,j], WR_Y[i,j] = Get_QL_and_QR( W_im2, W_im1, W_i, W_ip1,
                                                  Y_im1, Y_i,   Y_ip1,
                                                  YCoM_im2, YCoM_im1, YCoM_i, YCoM_ip1,
                                                  N_Y )

        end
    end

    # Boundary interfaces:
    @threads for i in 1:N_X
        i0 = i + ng
        @inbounds begin
            # Bottom boundary j = 1:
            WL_Y[i, 1]       = Wg[i0, ng]       # Ghost cell below
            WR_Y[i, 1]       = Wg[i0, ng+1]     # First interior

            #Ttop boundary j = N_Y+1: 
            WL_Y[i, N_Y+1]   = Wg[i0, N_Y+ng]   # Last interior
            WR_Y[i, N_Y+1]   = Wg[i0, N_Y+ng+1] # Top ghost
        end
    end

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


""" This function computes HLLC fluxes for
    the conserved variables: 

    - F_ρ  (density), 
    - F_ρu (momentum), 
    - F_ρv (transverse momentum)
    - F_E  (energy) 

 """
function GetFlux( ρ_L::Float64, 
                  ρ_R::Float64, 
                  u_L::Float64, 
                  u_R::Float64,
                  v_L::Float64,     
                  v_R::Float64, 
                  P_L::Float64,  
                  P_R::Float64, 
                γad_L::Float64,
                γad_R::Float64,
               PLyα_L::Float64 = 0.0,
               PLyα_R::Float64 = 0.0
                  )

    @assert P_L ≥ 0.0 && isfinite(P_L) "P_L must be ≥ 0 in the HLLC flux. Here P_L = $(P_L). "
    @assert P_R ≥ 0.0 && isfinite(P_R) "P_R must be ≥ 0 in the HLLC flux. Here P_R = $(P_R)"
    @assert ρ_L ≥ 0.0 && isfinite(ρ_L) "ρ_L must be ≥ 0 in the HLLC flux. Here ρ_L = $(ρ_L). "
    @assert ρ_R ≥ 0.0 && isfinite(ρ_R) "ρ_R must be ≥ 0 in the HLLC flux. Here ρ_R = $(ρ_R)"
    @assert isfinite(u_L) "u_L must be finite in the HLLC flux. Here u_L = $(u_L). "
    @assert isfinite(u_R) "u_R must be finite in the HLLC flux. Here u_R = $(u_R). "
    @assert isfinite(v_L) "v_L must be finite in the HLLC flux. Here v_L = $(v_L). "
    @assert isfinite(v_R) "v_R must be finite in the HLLC flux. Here v_R = $(v_R). "


    # Left & right energies:
    E_L    = P_L / (γad_L - 1)  +  0.5 * ρ_L * (u_L^2 + v_L^2)
    E_R    = P_R / (γad_R - 1)  +  0.5 * ρ_R * (u_R^2 + v_R^2)

    # Effective sound speeds:
    c_L    = sqrt( γad_L * P_L / ρ_L  +  γrad * PLyα_L / ρ_L )
    c_R    = sqrt( γad_R * P_R / ρ_R  +  γrad * PLyα_R / ρ_R )

    # Estimate wave speeds S_L and S_R:
    S_L    = min( u_L - c_L, u_R - c_R )
    S_R    = max( u_L + c_L, u_R + c_R )

    # Conserved variables:
    ρu_L   = ρ_L * u_L 
    ρu_R   = ρ_R * u_R 
    ρv_L   = ρ_L * v_L   # Transverse momentum in left state
    ρv_R   = ρ_R * v_R   # Transverse momentum in right state 

    # Estimate wave speed in star region (contact speed):
    S_star = ( P_R - P_L + ρu_L * (S_L - u_L) - ρu_R * (S_R - u_R)  ) / 
             ( ρ_L * (S_L - u_L) - ρ_R * (S_R - u_R) )

    #@assert isfinite(S_star) "$(ρ_L * (S_L - u_L) - ρ_R * (S_R - u_R))"
    #if !isfinite(S_star)
    #    S_star = 0.0
    #end

    # Compute fluxes for left state:
    F_ρ_L  = ρu_L 
    F_ρu_L = ρ_L * u_L^2  +  P_L  + PLyα_L
    F_ρv_L = ρv_L * u_L 
    F_E_L  = u_L * (E_L + P_L + PLyα_L)

    # Compute fluxes for right state:
    F_ρ_R  = ρu_R 
    F_ρu_R = ρ_R * u_R^2  +  P_R  +  PLyα_R
    F_ρv_R = ρv_R * u_R
    F_E_R  = u_R * (E_R + P_R + PLyα_R)

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

    @assert isfinite(F_ρ) 
    @assert isfinite(F_ρu)
    @assert isfinite(F_ρv)
    @assert isfinite(F_E) 

    return F_ρ, F_ρu, F_ρv, F_E
end


###############################################################
#######                                                 #######
#######            C O M P U T E  F L U X               #######
#######                                                 #######
###############################################################


""" Compute & update the flux in the X-direction. """
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
                             ρ::AbstractMatrix{Float64},
                           u_X::AbstractMatrix{Float64},
                           u_Y::AbstractMatrix{Float64},
                             P::AbstractMatrix{Float64},
                           γad::AbstractMatrix{Float64}
                         )

    N_X, N_Y = size( ρ ) # Grid size
    dN_Xp1   = 1:N_X+1
    dN_Y     = 1:N_Y

    @threads for j in dN_Y
        @simd for i in dN_Xp1
            @inbounds begin 
                # Unpack primitive variables:
                ρL  = ρ_L_X[i,j];   ρR  = ρ_R_X[i,j]
                uXL = u_X_L_X[i,j]; uXR = u_X_R_X[i,j]
                vL  = u_Y_L_X[i,j]; vR  = u_Y_R_X[i,j]   # Transverse velocities
                PL  = P_L_X[i,j];   PR  = P_R_X[i,j]

                # Adiabatic indices:
                γL = γad[clamp(i-1, 1, N_X), j]; γR = γad[clamp(i, 1, N_X), j]

                # Get the 1D flux:
                F_ρ_X[i,j], F_ρu_X_X[i,j], F_ρu_Y_X[i,j], F_E_X[i,j] = GetFlux(ρL, ρR, uXL, uXR, vL, vR, PL, PR, γL, γR)

            end
        end # End of i-loop
    end # End of j-loop

    return nothing
end


""" Compute & update the flux in the Y-direction. """
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
                             ρ::AbstractMatrix{Float64},
                           u_X::AbstractMatrix{Float64},
                           u_Y::AbstractMatrix{Float64},
                             P::AbstractMatrix{Float64},
                           γad::AbstractMatrix{Float64}
                         )

    N_X, N_Y = size( ρ ) # Grid size
    dN_X     = 1:N_X 
    dN_Yp1   = 1:N_Y+1

    @threads for j in dN_Yp1
        @simd for i in dN_X
            @inbounds begin 
                # Unpack primitive variables:
                ρL  = ρ_L_Y[i,j];   ρR  = ρ_R_Y[i,j]
                uL  = u_Y_L_Y[i,j]; uR  = u_Y_R_Y[i,j]
                vL  = u_X_L_Y[i,j]; vR  = u_X_R_Y[i,j] # Transverse velocities 
                PL  = P_L_Y[i,j];   PR  = P_R_Y[i,j]

                # Adiabatic indices:
                γL = γad[i, clamp(j-1, 1, N_Y)]; γR = γad[i, clamp(j, 1, N_Y)]

                # Get the 1D flux:
                F_ρ_Y[i,j], F_ρu_Y_Y[i,j], F_ρu_X_Y[i,j], F_E_Y[i,j] = GetFlux(ρL, ρR, uL, uR, vL, vR, PL, PR, γL, γR)

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
                        inv_ΔV::AbstractMatrix{Float64},  # Inverse volume 1 / ΔV
                           A_X::AbstractMatrix{Float64},  # Interface areas in X-direction 
                            Δt::Float64,
                         P_L_X::AbstractMatrix{Float64},  # Needed for geometric source term 
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
                # No momentum source term in the Y-direction:
                S_ρu_Y    = 0.0 
                # No momentum source term in the X-direction:  
                S_ρu_X    = 0.0
                # No energy source term:
                S_E       = 0.0

                # ------- Updates --------------------------------------
                # Update density:
                ρ[i,j]    = max( UpdateConservativeQuantity(ρ_old[i,j], Δt, 0.0, inv_ΔV[i,j], A_L, A_R, F_ρ_X[i,j], F_ρ_X[i+1,j]),
                                 ρ_floor )
                # Update momentum in X-direction:
                ρu_X[i,j] = UpdateConservativeQuantity(ρu_X_old[i,j], Δt, S_ρu_X, inv_ΔV[i,j], A_L, A_R, F_ρu_X_X[i,j], F_ρu_X_X[i+1,j])
                # Update momentum in Y-direction:
                ρu_Y[i,j] = UpdateConservativeQuantity(ρu_Y_old[i,j], Δt, S_ρu_Y, inv_ΔV[i,j], A_L, A_R, F_ρu_Y_X[i,j], F_ρu_Y_X[i+1,j])
                # Update energy:
                E[i,j]    = UpdateConservativeQuantity(E_old[i,j], Δt, S_E, inv_ΔV[i,j], A_L, A_R, F_E_X[i,j], F_E_X[i+1,j])
            end
        end # End of i-loop 
    end # End of j-loop 

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
                        inv_ΔV::AbstractMatrix{Float64},  # Inverse volume 1 / ΔV
                           A_X::AbstractMatrix{Float64},  # Interface areas in X-direction 
                            Δt::Float64,
                         P_L_X::AbstractMatrix{Float64},  # Needed for geometric source term 
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
                # No momentum source term in the Y-direction:
                S_ρu_Y    = 0.0 
                # Momentum source term in the X-direction (i.e. r-direction) in cylindrical coordinates:  
                S_ρu_X    = 0.5 * ( P_L_X[i,j]  +  P_R_X[i,j] ) * inv_r_cell[i]
                # No energy source term:
                S_E       = 0.0

                # ------- Updates --------------------------------------
                # Update density:
                ρ[i,j]    = max( UpdateConservativeQuantity(ρ_old[i,j], Δt, 0.0, inv_ΔV[i,j], A_L, A_R, F_ρ_X[i,j], F_ρ_X[i+1,j]),
                                 ρ_floor )
                # Update momentum in X-direction:
                ρu_X[i,j] = UpdateConservativeQuantity(ρu_X_old[i,j], Δt, S_ρu_X, inv_ΔV[i,j], A_L, A_R, F_ρu_X_X[i,j], F_ρu_X_X[i+1,j])
                # Update momentum in Y-direction:
                ρu_Y[i,j] = UpdateConservativeQuantity(ρu_Y_old[i,j], Δt, S_ρu_Y, inv_ΔV[i,j], A_L, A_R, F_ρu_Y_X[i,j], F_ρu_Y_X[i+1,j])
                # Update energy:
                E[i,j]    = UpdateConservativeQuantity(E_old[i,j], Δt, S_E, inv_ΔV[i,j], A_L, A_R, F_E_X[i,j], F_E_X[i+1,j])

                @assert isfinite(ρ[i,j])
                @assert isfinite(ρu_X[i,j])
                @assert isfinite(E[i,j]) "F_E_X[i+1,j] = $(F_E_X[i+1,j]) at (i,j) = $(i),$(j)"
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
                # No momentum source term in the Y-direction:
                S_ρu_Y    = 0.0 
                # No momentum source term in the X-direction:  
                S_ρu_X    = 0.0
                # No energy source term:
                S_E       = 0.0
                
                # ------- Updates --------------------------------------
                # Update density:
                ρ[i,j]    = max( UpdateConservativeQuantity(ρ_old[i,j], Δt, 0.0, inv_ΔV[i,j], A_L, A_R, F_ρ_Y[i,j], F_ρ_Y[i,j+1]),
                                 ρ_floor )
                # Update momentum in X-direction:
                ρu_X[i,j] = UpdateConservativeQuantity(ρu_X_old[i,j], Δt, S_ρu_X, inv_ΔV[i,j], A_L, A_R, F_ρu_X_Y[i,j], F_ρu_X_Y[i,j+1])
                # Update momentum in Y-direction:
                ρu_Y[i,j] = UpdateConservativeQuantity(ρu_Y_old[i,j], Δt, S_ρu_Y, inv_ΔV[i,j], A_L, A_R, F_ρu_Y_Y[i,j], F_ρu_Y_Y[i,j+1])
                # Update energy:
                E[i,j]    = UpdateConservativeQuantity(E_old[i,j], Δt, S_E, inv_ΔV[i,j], A_L, A_R, F_E_Y[i,j], F_E_Y[i,j+1])

                @assert isfinite(ρ[i,j])
                @assert isfinite(ρu_X[i,j])
                @assert isfinite(E[i,j])
            end
        end # End of i-loop 
    end # End of j-loop 

end


const CoordSystem = Union{Val{:Cartesian}, Val{:Cylindrical}}

function UpdateHydro_StrangSplit!( ρ_new::AbstractMatrix{Float64},
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
                                  inv_ΔV::AbstractMatrix{Float64},
                                     A_X::AbstractMatrix{Float64},
                                     A_Y::AbstractMatrix{Float64},
                                      Δt::Float64,

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
                                       
                             # Ghost‐padded primitives & grid:
                                  ρg_old::AbstractMatrix{Float64}, 
                                u_Xg_old::AbstractMatrix{Float64}, 
                                u_Yg_old::AbstractMatrix{Float64}, 
                                  Pg_old::AbstractMatrix{Float64},
                                  ρg_new::AbstractMatrix{Float64}, 
                                u_Xg_new::AbstractMatrix{Float64},
                                u_Yg_new::AbstractMatrix{Float64},
                                  Pg_new::AbstractMatrix{Float64},
                                      Xg::AbstractVector{Float64},
                                   XCoMg::AbstractVector{Float64},
                                      Yg::AbstractVector{Float64},
                                   YCoMg::AbstractVector{Float64},

                                CoordSys::CoordSystem,
                              inv_r_cell::AbstractVector{Float64}
                             )

    N_X, N_Y = size( ρ )
    dN_X     = 1:N_X
    dN_Y     = 1:N_Y
    Δt_half  = Δt / 2

    ######### 1) HALF-STEP IN X-DIRECTION #########

    # Update primitives:
    @threads for j in dN_Y 
        @inbounds @simd for i in dN_X
            u_X[i,j], u_Y[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_X[i,j], ρu_Y[i,j], E[i,j], γad[i,j] )
            P[i,j] = max( P[i,j], P_floor )
        end
    end

    # ghosts from current state
    @views @inbounds begin 
        ρg_old[ng+1:end-ng, ng+1:end-ng]   .= ρ
        u_Xg_old[ng+1:end-ng, ng+1:end-ng] .= u_X
        u_Yg_old[ng+1:end-ng, ng+1:end-ng] .= u_Y
        Pg_old[ng+1:end-ng, ng+1:end-ng]   .= P
    end
    FillGhosts!( ρg_old, u_Xg_old, u_Yg_old, Pg_old )

    # X interfaces
    UpdateInterfaceValues_X!( ρ_L_X, ρ_R_X, ρg_old, Xg, XCoMg )
    UpdateInterfaceValues_X!( u_X_L_X, u_X_R_X, u_Xg_old, Xg, XCoMg )
    UpdateInterfaceValues_X!( u_Y_L_X, u_Y_R_X, u_Yg_old, Xg, XCoMg )
    UpdateInterfaceValues_X!( P_L_X,  P_R_X,  Pg_old,  Xg, XCoMg )
    ρ_L_X .= max.(ρ_L_X, ρ_floor);  ρ_R_X .= max.(ρ_R_X, ρ_floor)
    P_L_X .= max.(P_L_X,  P_floor); P_R_X .= max.(P_R_X,  P_floor)

    # X fluxes
    ComputeFlux_X!( F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,
                    ρ_L_X, ρ_R_X, u_X_L_X, u_X_R_X,
                    u_Y_L_X, u_Y_R_X, P_L_X, P_R_X,
                    ρ, u_X, u_Y, P, γad )

    # Update over Δt/2 (write into *_new)
    UpdateConserved_X!( ρ_new, ρ, ρu_X_new, ρu_X, ρu_Y_new, ρu_Y,
                        E_new, E,
                        F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,
                        inv_ΔV, A_X, Δt_half,
                        P_L_X, P_R_X, inv_r_cell, CoordSys )

    # Update old variables:
    ρ    .= ρ_new 
    ρu_X .= ρu_X_new
    ρu_Y .= ρu_Y_new
    E    .= E_new

    ######### 2) FULL-STEP IN Y-DIRECTION #########
    # Use current (half-step) state as the "old" state; do NOT alias.

    # Update primitives:
    @threads for j in dN_Y 
        @inbounds @simd for i in dN_X
            u_X[i,j], u_Y[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_X[i,j], ρu_Y[i,j], E[i,j], γad[i,j] )
            P[i,j] = max( P[i,j], P_floor )
        end
    end

    # ghosts from current state
    @views @inbounds begin 
        ρg_old[ng+1:end-ng, ng+1:end-ng]   .= ρ
        u_Xg_old[ng+1:end-ng, ng+1:end-ng] .= u_X
        u_Yg_old[ng+1:end-ng, ng+1:end-ng] .= u_Y
        Pg_old[ng+1:end-ng, ng+1:end-ng]   .= P
    end
    FillGhosts!( ρg_old, u_Xg_old, u_Yg_old, Pg_old )

    # Y interfaces
    UpdateInterfaceValues_Y!( ρ_L_Y, ρ_R_Y, ρg_old, Yg, YCoMg )
    UpdateInterfaceValues_Y!( u_Y_L_Y, u_Y_R_Y, u_Yg_old, Yg, YCoMg )
    UpdateInterfaceValues_Y!( u_X_L_Y, u_X_R_Y, u_Xg_old, Yg, YCoMg )
    UpdateInterfaceValues_Y!( P_L_Y,  P_R_Y,  Pg_old,  Yg, YCoMg )
    ρ_L_Y .= max.(ρ_L_Y, ρ_floor);  ρ_R_Y .= max.(ρ_R_Y, ρ_floor)
    P_L_Y .= max.(P_L_Y,  P_floor); P_R_Y .= max.(P_R_Y,  P_floor)

    # Y fluxes
    ComputeFlux_Y!( F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,
                    ρ_L_Y, ρ_R_Y, u_Y_L_Y, u_Y_R_Y,
                    u_X_L_Y, u_X_R_Y, P_L_Y, P_R_Y,
                    ρ, u_X, u_Y, P, γad )

    # Update over Δt (read from current state, write into *_new)
    UpdateConserved_Y!( ρ_new, ρ, ρu_X_new, ρu_X,
                        ρu_Y_new, ρu_Y, E_new, E,
                        F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,
                        inv_ΔV, A_Y, Δt )

    # Update old variables:
    ρ    .= ρ_new
    ρu_X .= ρu_X_new
    ρu_Y .= ρu_Y_new
    E    .= E_new

    ######### 3) FINAL HALF-STEP IN X-DIRECTION #########

    @threads for j in dN_Y 
        @inbounds @simd for i in dN_X
            u_X[i,j], u_Y[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_X[i,j], ρu_Y[i,j], E[i,j], γad[i,j] )
            P[i,j] = max( P[i,j], P_floor )
        end
    end

    # ghosts from current state
    @views @inbounds begin 
        ρg_old[ng+1:end-ng, ng+1:end-ng]   .= ρ
        u_Xg_old[ng+1:end-ng, ng+1:end-ng] .= u_X
        u_Yg_old[ng+1:end-ng, ng+1:end-ng] .= u_Y
        Pg_old[ng+1:end-ng, ng+1:end-ng]   .= P
    end
    FillGhosts!( ρg_old, u_Xg_old, u_Yg_old, Pg_old )

    # X interfaces
    UpdateInterfaceValues_X!( ρ_L_X, ρ_R_X, ρg_old, Xg, XCoMg )
    UpdateInterfaceValues_X!( u_X_L_X, u_X_R_X, u_Xg_old, Xg, XCoMg )
    UpdateInterfaceValues_X!( u_Y_L_X, u_Y_R_X, u_Yg_old, Xg, XCoMg )
    UpdateInterfaceValues_X!( P_L_X,  P_R_X,  Pg_old,  Xg, XCoMg )
    ρ_L_X .= max.(ρ_L_X, ρ_floor);  ρ_R_X .= max.(ρ_R_X, ρ_floor)
    P_L_X .= max.(P_L_X,  P_floor); P_R_X .= max.(P_R_X,  P_floor)

    # X fluxes
    ComputeFlux_X!( F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,
                    ρ_L_X, ρ_R_X, u_X_L_X, u_X_R_X,
                    u_Y_L_X, u_Y_R_X, P_L_X, P_R_X,
                    ρ, u_X, u_Y, P, γad )

    # Update over Δt/2
    UpdateConserved_X!( ρ_new, ρ, ρu_X_new, ρu_X,
                        ρu_Y_new, ρu_Y, E_new, E,
                        F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,
                        inv_ΔV, A_X, Δt_half,
                        P_L_X, P_R_X, inv_r_cell, CoordSys )

    # Update old variables:
    ρ    .= ρ_new
    ρu_X .= ρu_X_new
    ρu_Y .= ρu_Y_new
    E    .= E_new

    # Update primitives:
    @threads for j in dN_Y 
        @inbounds @simd for i in dN_X
            u_X[i,j], u_Y[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_X[i,j], ρu_Y[i,j], E[i,j], γad[i,j] )
            P[i,j] = max( P[i,j], P_floor )
        end
    end

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
                                         inv_ΔV::AbstractMatrix{Float64},
                                            A_X::AbstractMatrix{Float64},
                                            A_Y::AbstractMatrix{Float64},
                                             Δt::Float64,

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
                                        
                                   # Ghost‐padded primitives & grid:
                                         ρg_old::AbstractMatrix{Float64}, 
                                       u_Xg_old::AbstractMatrix{Float64}, 
                                       u_Yg_old::AbstractMatrix{Float64}, 
                                         Pg_old::AbstractMatrix{Float64},
                                         ρg_new::AbstractMatrix{Float64}, 
                                       u_Xg_new::AbstractMatrix{Float64},
                                       u_Yg_new::AbstractMatrix{Float64},
                                         Pg_new::AbstractMatrix{Float64},
                                             Xg::AbstractVector{Float64},
                                          XCoMg::AbstractVector{Float64},
                                             Yg::AbstractVector{Float64},
                                          YCoMg::AbstractVector{Float64},

                                       CoordSys::CoordSystem,
                                     inv_r_cell::AbstractVector{Float64}
                             )

    N_X, N_Y = size( ρ )
    dN_X     = 1:N_X
    dN_Y     = 1:N_Y
    Δt_half  = Δt / 2

    ######### 1) HALF-STEP IN Y-DIRECTION #########

    # Update primitives:
    @threads for j in dN_Y 
        @inbounds @simd for i in dN_X
            u_X[i,j], u_Y[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_X[i,j], ρu_Y[i,j], E[i,j], γad[i,j] )
            P[i,j] = max( P[i,j], P_floor )
        end
    end

    # ghosts from current state
    @views @inbounds begin 
        ρg_old[ng+1:end-ng, ng+1:end-ng]   .= ρ
        u_Xg_old[ng+1:end-ng, ng+1:end-ng] .= u_X
        u_Yg_old[ng+1:end-ng, ng+1:end-ng] .= u_Y
        Pg_old[ng+1:end-ng, ng+1:end-ng]   .= P
    end
    FillGhosts!( ρg_old, u_Xg_old, u_Yg_old, Pg_old )

    # Y interfaces
    UpdateInterfaceValues_Y!( ρ_L_Y, ρ_R_Y, ρg_old, Yg, YCoMg )
    UpdateInterfaceValues_Y!( u_Y_L_Y, u_Y_R_Y, u_Yg_old, Yg, YCoMg )
    UpdateInterfaceValues_Y!( u_X_L_Y, u_X_R_Y, u_Xg_old, Yg, YCoMg )
    UpdateInterfaceValues_Y!( P_L_Y,  P_R_Y,  Pg_old,  Yg, YCoMg )
    ρ_L_Y .= max.(ρ_L_Y, ρ_floor);  ρ_R_Y .= max.(ρ_R_Y, ρ_floor)
    P_L_Y .= max.(P_L_Y,  P_floor); P_R_Y .= max.(P_R_Y,  P_floor)

    # Y fluxes
    ComputeFlux_Y!( F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,
                    ρ_L_Y, ρ_R_Y, u_Y_L_Y, u_Y_R_Y,
                    u_X_L_Y, u_X_R_Y, P_L_Y, P_R_Y,
                    ρ, u_X, u_Y, P, γad )

    # Update over Δt/2 
    UpdateConserved_Y!( ρ_new, ρ, ρu_X_new, ρu_X,
                        ρu_Y_new, ρu_Y, E_new, E,
                        F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,
                        inv_ΔV, A_Y, Δt_half )

    # Update old variables:
    ρ    .= ρ_new
    ρu_X .= ρu_X_new
    ρu_Y .= ρu_Y_new
    E    .= E_new

    ######### 2) FULL-STEP IN X-DIRECTION #########

    @threads for j in dN_Y 
        @inbounds @simd for i in dN_X
            u_X[i,j], u_Y[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_X[i,j], ρu_Y[i,j], E[i,j], γad[i,j] )
            P[i,j] = max( P[i,j], P_floor )
        end
    end

    # ghosts from current state
    @views @inbounds begin 
        ρg_old[ng+1:end-ng, ng+1:end-ng]   .= ρ
        u_Xg_old[ng+1:end-ng, ng+1:end-ng] .= u_X
        u_Yg_old[ng+1:end-ng, ng+1:end-ng] .= u_Y
        Pg_old[ng+1:end-ng, ng+1:end-ng]   .= P
    end
    FillGhosts!( ρg_old, u_Xg_old, u_Yg_old, Pg_old )

    # X interfaces
    UpdateInterfaceValues_X!( ρ_L_X, ρ_R_X, ρg_old, Xg, XCoMg )
    UpdateInterfaceValues_X!( u_X_L_X, u_X_R_X, u_Xg_old, Xg, XCoMg )
    UpdateInterfaceValues_X!( u_Y_L_X, u_Y_R_X, u_Yg_old, Xg, XCoMg )
    UpdateInterfaceValues_X!( P_L_X,  P_R_X,  Pg_old,  Xg, XCoMg )
    ρ_L_X .= max.(ρ_L_X, ρ_floor);  ρ_R_X .= max.(ρ_R_X, ρ_floor)
    P_L_X .= max.(P_L_X,  P_floor); P_R_X .= max.(P_R_X,  P_floor)

    # X fluxes
    ComputeFlux_X!( F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,
                    ρ_L_X, ρ_R_X, u_X_L_X, u_X_R_X,
                    u_Y_L_X, u_Y_R_X, P_L_X, P_R_X,
                    ρ, u_X, u_Y, P, γad )

    # Update over Δt
    UpdateConserved_X!( ρ_new, ρ, ρu_X_new, ρu_X,
                        ρu_Y_new, ρu_Y, E_new, E,
                        F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,
                        inv_ΔV, A_X, Δt,
                        P_L_X, P_R_X, inv_r_cell, CoordSys )

    # Update old variables:
    ρ    .= ρ_new
    ρu_X .= ρu_X_new
    ρu_Y .= ρu_Y_new
    E    .= E_new

    ######### 3) FINAL HALF-STEP IN Y-DIRECTION #########

    # Update primitives:
    @threads for j in dN_Y 
        @inbounds @simd for i in dN_X
            u_X[i,j], u_Y[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_X[i,j], ρu_Y[i,j], E[i,j], γad[i,j] )
            P[i,j] = max( P[i,j], P_floor )
        end
    end

    # ghosts from current state
    @views @inbounds begin 
        ρg_old[ng+1:end-ng, ng+1:end-ng]   .= ρ
        u_Xg_old[ng+1:end-ng, ng+1:end-ng] .= u_X
        u_Yg_old[ng+1:end-ng, ng+1:end-ng] .= u_Y
        Pg_old[ng+1:end-ng, ng+1:end-ng]   .= P
    end
    FillGhosts!( ρg_old, u_Xg_old, u_Yg_old, Pg_old )

    # Y interfaces
    UpdateInterfaceValues_Y!( ρ_L_Y, ρ_R_Y, ρg_old, Yg, YCoMg )
    UpdateInterfaceValues_Y!( u_Y_L_Y, u_Y_R_Y, u_Yg_old, Yg, YCoMg )
    UpdateInterfaceValues_Y!( u_X_L_Y, u_X_R_Y, u_Xg_old, Yg, YCoMg )
    UpdateInterfaceValues_Y!( P_L_Y,  P_R_Y,  Pg_old,  Yg, YCoMg )
    ρ_L_Y .= max.(ρ_L_Y, ρ_floor);  ρ_R_Y .= max.(ρ_R_Y, ρ_floor)
    P_L_Y .= max.(P_L_Y,  P_floor); P_R_Y .= max.(P_R_Y,  P_floor)

    # Y fluxes
    ComputeFlux_Y!( F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,
                    ρ_L_Y, ρ_R_Y, u_Y_L_Y, u_Y_R_Y,
                    u_X_L_Y, u_X_R_Y, P_L_Y, P_R_Y,
                    ρ, u_X, u_Y, P, γad )

    # Update over Δt/2 
    UpdateConserved_Y!( ρ_new, ρ, ρu_X_new, ρu_X,
                        ρu_Y_new, ρu_Y, E_new, E,
                        F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,
                        inv_ΔV, A_Y, Δt_half )

    # Update old variables:
    ρ    .= ρ_new
    ρu_X .= ρu_X_new
    ρu_Y .= ρu_Y_new
    E    .= E_new

    # Update primitives:
    @threads for j in dN_Y 
        @inbounds @simd for i in dN_X
            u_X[i,j], u_Y[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_X[i,j], ρu_Y[i,j], E[i,j], γad[i,j] )
            P[i,j] = max( P[i,j], P_floor )
        end
    end

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
                      inv_ΔV::AbstractMatrix{Float64},
                         A_X::AbstractMatrix{Float64},
                         A_Y::AbstractMatrix{Float64},
                          Δt::Float64,

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
                                       
                    # Ghost‐padded primitives & grid:
                      ρg_old::AbstractMatrix{Float64}, 
                    u_Xg_old::AbstractMatrix{Float64}, 
                    u_Yg_old::AbstractMatrix{Float64}, 
                      Pg_old::AbstractMatrix{Float64},
                      ρg_new::AbstractMatrix{Float64}, 
                    u_Xg_new::AbstractMatrix{Float64},
                    u_Yg_new::AbstractMatrix{Float64},
                      Pg_new::AbstractMatrix{Float64},
                          Xg::AbstractVector{Float64},
                       XCoMg::AbstractVector{Float64},
                          Yg::AbstractVector{Float64},
                       YCoMg::AbstractVector{Float64},

                    CoordSys::CoordSystem,
                  inv_r_cell::AbstractVector{Float64},
                      Xfirst::Bool = true 
                   )

    # Save old variables:
    @. ρ0     =  ρ  
    @. ρu_X0  =  ρu_X  
    @. ρu_Y0  =  ρu_Y 
    @. E0     =  E

    if Xfirst
        # STEP 1: Get U^(1) = U^(n) + F[U^(n)] Δt:
        UpdateHydro_StrangSplit!(
            ρ_new, ρu_X_new, ρu_Y_new, E_new,
            ρ, ρu_X, ρu_Y, E,         
            γad, u_X, u_Y, inv_ΔV, A_X, A_Y, Δt,
            ρ_L_X, ρ_R_X, u_X_L_X, u_X_R_X, u_Y_L_X, u_Y_R_X, P_L_X, P_R_X, F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,
            ρ_L_Y, ρ_R_Y, u_Y_L_Y, u_Y_R_Y, u_X_L_Y, u_X_R_Y, P_L_Y, P_R_Y, F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,
            P,
            ρg_old, u_Xg_old, u_Yg_old, Pg_old,
            ρg_new, u_Xg_new, u_Yg_new, Pg_new,
            Xg, XCoMg, Yg, YCoMg,
            CoordSys, inv_r_cell )

        # STEP 2: Get U^(2) = = U^(1) + F[U^(1)] Δt:
        UpdateHydro_StrangSplit!(
            ρ_new, ρu_X_new, ρu_Y_new, E_new,
            ρ, ρu_X, ρu_Y, E,
            γad, u_X, u_Y, inv_ΔV, A_X, A_Y, Δt,
            ρ_L_X, ρ_R_X, u_X_L_X, u_X_R_X, u_Y_L_X, u_Y_R_X, P_L_X, P_R_X, F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,
            ρ_L_Y, ρ_R_Y, u_Y_L_Y, u_Y_R_Y, u_X_L_Y, u_X_R_Y, P_L_Y, P_R_Y, F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,
            P,
            ρg_old, u_Xg_old, u_Yg_old, Pg_old,
            ρg_new, u_Xg_new, u_Yg_new, Pg_new,
            Xg, XCoMg, Yg, YCoMg,
            CoordSys, inv_r_cell )
    else
        # STEP 1: Get U^(1) = U^(n) + F[U^(n)] Δt:
        UpdateHydro_StrangSplit_Yfirst!(
            ρ_new, ρu_X_new, ρu_Y_new, E_new,
            ρ, ρu_X, ρu_Y, E,         
            γad, u_X, u_Y, inv_ΔV, A_X, A_Y, Δt,
            ρ_L_X, ρ_R_X, u_X_L_X, u_X_R_X, u_Y_L_X, u_Y_R_X, P_L_X, P_R_X, F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,
            ρ_L_Y, ρ_R_Y, u_Y_L_Y, u_Y_R_Y, u_X_L_Y, u_X_R_Y, P_L_Y, P_R_Y, F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,
            P,
            ρg_old, u_Xg_old, u_Yg_old, Pg_old,
            ρg_new, u_Xg_new, u_Yg_new, Pg_new,
            Xg, XCoMg, Yg, YCoMg,
            CoordSys, inv_r_cell )

        # STEP 2: Get U^(2) = U^(1) + F[U^(1)] Δt:
        UpdateHydro_StrangSplit_Yfirst!(
            ρ_new, ρu_X_new, ρu_Y_new, E_new,
            ρ, ρu_X, ρu_Y, E,
            γad, u_X, u_Y, inv_ΔV, A_X, A_Y, Δt,
            ρ_L_X, ρ_R_X, u_X_L_X, u_X_R_X, u_Y_L_X, u_Y_R_X, P_L_X, P_R_X, F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,
            ρ_L_Y, ρ_R_Y, u_Y_L_Y, u_Y_R_Y, u_X_L_Y, u_X_R_Y, P_L_Y, P_R_Y, F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,
            P,
            ρg_old, u_Xg_old, u_Yg_old, Pg_old,
            ρg_new, u_Xg_new, u_Yg_new, Pg_new,
            Xg, XCoMg, Yg, YCoMg,
            CoordSys, inv_r_cell )
    end


    # STEP 3: Get combined second-order RK2 update, U^(n+1) = 0.5 × [ U^n + U^(2) ]
    @. ρ     =  0.5 * ( ρ0     +  ρ ) 
    @. ρu_X  =  0.5 * ( ρu_X0  +  ρu_X ) 
    @. ρu_Y  =  0.5 * ( ρu_Y0  +  ρu_Y ) 
    @. E     =  0.5 * ( E0     +  E ) 

    # STEP 4: Update primitives:
    N_X, N_Y = size( ρ )
    dN_X     = 1:N_X
    dN_Y     = 1:N_Y
    @threads for j in dN_Y
        @inbounds @simd for i in dN_X
            u_X[i,j], u_Y[i,j], P[i,j] = GetPrimitive( ρ[i,j], ρu_X[i,j], ρu_Y[i,j], E[i,j], γad[i,j] )
            P[i,j] = max( P[i,j], P_floor )
        end
    end

    return nothing 
end


function GravitySolveAndKick!( Δt::Float64,
                             ρu_X::AbstractMatrix{Float64}, 
                            ρu_X0::AbstractMatrix{Float64}, 
                             ρu_Y::AbstractMatrix{Float64},
                            ρu_Y0::AbstractMatrix{Float64}, 
                                ρ::AbstractMatrix{Float64}, # Latest densities from after hydro advecton step 
                               ρ0::AbstractMatrix{Float64}, # Old densities from before hydro advecton step 
                                E::AbstractMatrix{Float64},

                                Φ::AbstractMatrix{Float64},
                           Φ_half::AbstractMatrix{Float64},
                            Φ_old::AbstractMatrix{Float64},
                             ∂Φ∂X::AbstractMatrix{Float64},
                             ∂Φ∂Y::AbstractMatrix{Float64},
                       RPoiss_ijp::AbstractMatrix{Float64}, 
                        RPoiss_ij::AbstractMatrix{Float64},
                       RPoiss_ijm::AbstractMatrix{Float64},
                       ZPoiss_ijp::AbstractMatrix{Float64}, 
                        ZPoiss_ij::AbstractMatrix{Float64},
                       ZPoiss_ijm::AbstractMatrix{Float64},
                           Δτ_max::Float64,
                           Δτ_min::Float64,

                             XCoM::AbstractArray{Float64}, 
                             YCoM::AbstractArray{Float64} )

    # Grid size & intervals: 
    N_X, N_Y = size( ρ )
    dN_X     = 1:N_X
    dN_Y     = 1:N_Y 

    # Fill ρ0 with the estimated ρ^(n+1/2):
    @. ρ0    = 0.5 * ( ρ0 + ρ )
    # Similarly, fill ρu_X0 and ρu_Y0 with their estimated values at t^(n+1/2):
    @. ρu_X0 = 0.5 * ( ρu_X0 + ρu_X )
    @. ρu_Y0 = 0.5 * ( ρu_Y0 + ρu_Y )

    # Check whether we should update Φ or not:
    ShouldWeUpdate = ShouldWeUpdateΦ( Φ, ρ0, RPoiss_ijp, RPoiss_ij, RPoiss_ijm,
                                      ZPoiss_ijp, ZPoiss_ij, ZPoiss_ijm )

    # If ShouldWeUpdate = true, we solve the Poisson equation for the updated 
    # potential Φ, and the updated gravitational acceleration:
    if ShouldWeUpdate 
        Update_Φ_ADI!( Φ, Φ_half, Φ_old, ρ0, RPoiss_ijp, RPoiss_ij, RPoiss_ijm,
                       ZPoiss_ijp, ZPoiss_ij, ZPoiss_ijm, Δτ_max, Δτ_min )
                           
        Update_∇Φ!( ∂Φ∂X, ∂Φ∂Y, Φ, XCoM, YCoM )
    end

    # Finally, we perform a second-order in time gravity kick to the
    # hydro variables:
    @threads for j in dN_Y 
        @inbounds for i in dN_X
            # Update momenta to second-order accuracy, using ρ0 = ρ^(n+1/2): 
            ρu_X[i,j] -=  ρ0[i,j] * ∂Φ∂X[i,j] * Δt 
            ρu_Y[i,j] -=  ρ0[i,j] * ∂Φ∂Y[i,j] * Δt 

            # Update energy to second-order accuracy, using ρu_X0 = ρu_X^(n+1/2)
            # and ρu_Y0 = ρu_Y^(n+1/2):
            E[i,j]    -= ( ∂Φ∂X[i,j] * ρu_X0[i,j]  +  ∂Φ∂Y[i,j] * ρu_Y0[i,j] ) * Δt
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


""" Initialize all the hydro arrays (cell-centered, interface,
    slope and flux) for an (X, Y) grid. 

    **ARGUMENTS:**

    - X, Y       : interface vectors of length N_X+1, N_Y+1
    - XCoM, YCoM : cell-center vectors of length N_X,   N_Y """
function GenerateHydroArrays( X::AbstractVector{Float64},   
                           XCoM::AbstractVector{Float64},
                              Y::AbstractVector{Float64},   
                           YCoM::AbstractVector{Float64},
                              γ::Float64 = 5 / 3
                             )
    
  # Grid size:
  N_X = length(X)-1 
  N_Y = length(Y)-1

  @assert N_X == length(XCoM) "We must have length(X) = length(XCoM) + 1"
  @assert N_Y == length(YCoM) "We must have length(Y) = length(YCoM) + 1"

  # Cell-centered primitive & conserved variables:
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

  # Adiabatic index:
  γad        = fill(γ, size(ρ))

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

  ρg_new     = zeros(Float64, N_X + 2*ng,   N_Y + 2*ng)
  Pg_new     = zeros(Float64, N_X + 2*ng,   N_Y + 2*ng)
  u_Xg_new   = zeros(Float64, N_X + 2*ng,   N_Y + 2*ng)
  u_Yg_new   = zeros(Float64, N_X + 2*ng,   N_Y + 2*ng)

  # X-direction interface values (N_X + 1 × N_Y)
  ρ_L_X      = zeros(Float64, N_X+1, N_Y)
  ρ_R_X      = zeros(Float64, N_X+1, N_Y)
  u_X_L_X    = zeros(Float64, N_X+1, N_Y)
  u_X_R_X    = zeros(Float64, N_X+1, N_Y)
  u_Y_L_X    = zeros(Float64, N_X+1, N_Y)
  u_Y_R_X    = zeros(Float64, N_X+1, N_Y)
  P_L_X      = zeros(Float64, N_X+1, N_Y)
  P_R_X      = zeros(Float64, N_X+1, N_Y)

  # Y-direction fluxes (N_X+1 × N_Y)
  F_ρ_X      = zeros(Float64, N_X+1, N_Y)
  F_ρu_X_X   = zeros(Float64, N_X+1, N_Y)
  F_ρu_Y_X   = zeros(Float64, N_X+1, N_Y)
  F_E_X      = zeros(Float64, N_X+1, N_Y)

  # Y-direction interface values (N_X × N_Y + 1)
  ρ_L_Y      = zeros(Float64, N_X, N_Y+1)
  ρ_R_Y      = zeros(Float64, N_X, N_Y+1)
  u_Y_L_Y    = zeros(Float64, N_X, N_Y+1)
  u_Y_R_Y    = zeros(Float64, N_X, N_Y+1)
  u_X_L_Y    = zeros(Float64, N_X, N_Y+1)
  u_X_R_Y    = zeros(Float64, N_X, N_Y+1)
  P_L_Y      = zeros(Float64, N_X, N_Y+1)
  P_R_Y      = zeros(Float64, N_X, N_Y+1)

  # Y-direction fluxes (N_X × N_Y + 1)
  F_ρ_Y      = zeros(Float64, N_X, N_Y+1)
  F_ρu_X_Y   = zeros(Float64, N_X, N_Y+1)
  F_ρu_Y_Y   = zeros(Float64, N_X, N_Y+1)
  F_E_Y      = zeros(Float64, N_X, N_Y+1)

  return ( # Cell-center values:
           ρ,     ρu_X,     ρu_Y,     E, 
           ρ_new, ρu_X_new, ρu_Y_new, E_new, 
           ρ0,    ρu_X0,    ρu_Y0,    E0, 
           P, u_X, u_Y, γad,
           # X-interface values:
           ρ_L_X, ρ_R_X, 
           u_X_L_X, u_X_R_X, 
           u_Y_L_X, u_Y_R_X,
           P_L_X, P_R_X,
           F_ρ_X, F_ρu_X_X, F_ρu_Y_X, F_E_X,
           # Y-interface values:
           ρ_L_Y, ρ_R_Y, 
           u_Y_L_Y, u_Y_R_Y, 
           u_X_L_Y, u_X_R_Y, 
           P_L_Y, P_R_Y,
           F_ρ_Y, F_ρu_X_Y, F_ρu_Y_Y, F_E_Y,
           # Ghost-padded variables:
           ρg_old, u_Xg_old, u_Yg_old, Pg_old,
           ρg_new, u_Xg_new, u_Yg_new, Pg_new,

           # Ghost-padded grid:
           Xg, XCoMg, Yg, YCoMg
         )
end


end   # End of module 