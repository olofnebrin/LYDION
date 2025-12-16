
""" FILE:    SpatialGrid

    PURPOSE: Contains functions for setting up a spatial grid, in 
             both cylindrical and Cartesian coordinates.

"""


using .Structs: GeometryCylindrical, GeometryCartesian
using  StaticArrays, LinearAlgebra  # For some matrix algebra to determine some WENO3 coefficients 

export SetupCylindricalGrid, SetupCartesianGrid


###############################################################
#######                                                 #######
#######         C Y L I N D R I C A L  G R I D          #######
#######                                                 #######
###############################################################


""" Compute the cell centroid in a cell extending from 
    r_{i-1/2} to r_{i+1/2}. """
function Get_Single_rCoM_Cylindrical( r_phalf::T, # r_{i+1/2}
                                      r_mhalf::T  # r_{i-1/2}
                                      ) where {T<:AbstractFloat}

    rCoM = (T(2)/T(3)) * ( r_phalf^3  -  r_mhalf^3 ) / ( r_phalf^2  -  r_mhalf^2 )

    return rCoM
end

""" This function populates rCoM with the centers of mass along 
    the radial direction, *given* interface values r.
    
    Works only for the cylindrical r-coordinate. """
function Get_rCoM_Cylindrical!( r::AbstractArray{T}, 
                             rCoM::AbstractArray{T} ) where {T<:AbstractFloat}

    N_r  = length( r ) - 1  # Number of radial grid points 
    dN_r = 1:N_r

    @assert N_r == length( rCoM )

    @inbounds @simd for i in dN_r
        rCoM[i] = Get_Single_rCoM_Cylindrical( r[i+1], r[i] )
    end

    return nothing
end


""" This function populates ZCoM with the centers of mass along 
    the vertical direction, *given* interface values Z.
    
    Works for both cylindrical & Cartesian 
    coordinates. """
function Get_ZCoM!( Z::AbstractArray{T}, 
                 ZCoM::AbstractArray{T} ) where {T<:AbstractFloat}

    N_Z  = length( Z ) - 1  # Number of vertical grid points 
    dN_Z = 1:N_Z 

    @assert N_Z == length( ZCoM )

    @inbounds @simd for j in dN_Z
        ZCoM[j] = T(0.5) * ( Z[j]  +  Z[j+1] )
    end

    return nothing
end


""" This function populates ΔV with the cell volumes, 
    *given* interface values r and Z.
    
    Works for cylindrical coordinates. """
function Get_ΔV_Cylindrical!( r::AbstractArray{T}, 
                              Z::AbstractArray{T},
                             ΔV::AbstractArray{T, 2} ) where {T<:AbstractFloat}

    N_r  = length( r ) - 1  # Number of radial grid points         
    N_Z  = length( Z ) - 1  # Number of vertical grid points 
    dN_r = 1:N_r
    dN_Z = 1:N_Z 

    @assert size( ΔV ) == (N_r, N_Z)

    @inbounds for j in dN_Z 
        ΔZ = Z[j+1] - Z[j]
        @simd for i in dN_r
            ΔV[i,j] = T(π) * ( r[i+1]^2 - r[i]^2 ) * ΔZ
        end
    end

    return nothing
end


""" This function populates A_r with interface cell areas, 
    along the r-direction, given interface values r and Z.
    
    Works for cylindrical coordinates. """
function Get_A_r_Cylindrical!( r::AbstractArray{T}, 
                               Z::AbstractArray{T},
                             A_r::AbstractArray{T, 2} ) where {T<:AbstractFloat}

    N_r  = length( r ) - 1  # Number of radial grid points         
    N_Z  = length( Z ) - 1  # Number of vertical grid points 
    dN_Z = 1:N_Z 

    @assert size( A_r ) == (N_r + 1, N_Z)

    @inbounds for j in dN_Z 
        ΔZ = Z[j+1] - Z[j]
        @simd for i in 1:(N_r+1)
            A_r[i,j] = T(2) * T(π) * r[i] * ΔZ
        end
    end

    return nothing
end


""" This function populates A_Z with interface cell areas, 
    along the Z-direction, given interface values r and Z.
    
    Works for cylindrical coordinates. """
function Get_A_Z_Cylindrical!( r::AbstractArray{T}, 
                               Z::AbstractArray{T},
                             A_Z::AbstractArray{T, 2} ) where {T<:AbstractFloat}

    N_r  = length( r ) - 1  # Number of radial grid points         
    N_Z  = length( Z ) - 1  # Number of vertical grid points 
    dN_r = 1:N_r 

    @assert size( A_Z ) == (N_r, N_Z + 1)

    @inbounds for j in 1:(N_Z+1) 
        @simd for i in dN_r
            A_Z[i,j] = T(π) * ( r[i+1]^2  -  r[i]^2 ) 
        end
    end

    return nothing
end


""" This function populates ⟨1/r⟩, needed for the geometric 
    source term in problems in cylindrical coordinates. """
function Get_inv_r_cell_Cylindrical!( inv_r_cell::AbstractArray{T},
                                               r::AbstractArray{T}
                                     ) where {T<:AbstractFloat}

    N_r  = length( r ) - 1  # Number of radial grid points     
    dN_r = 1:N_r 

    @assert length( inv_r_cell ) == N_r

    @simd for i in dN_r 
        @inbounds inv_r_cell[i] = T(2) / ( r[i+1]  +  r[i] )
    end

    return nothing
end


""" This function computes w⁺_{i,s} and w⁻_{i,s} for the cylindrical r-coordinate,
    for all radial cells i, and with s ∈ (-1, 0, 1). These coefficients are needed 
    for the WENO3 interface reconstruction, and are constants for a given grid. """
function Get_WENO3_w_plus_w_minus_r!( r::AbstractArray{T, 1},
                            w_plus_i_p1::AbstractArray{T, 1},  # w⁺_{i,1}
                           w_minus_i_p1::AbstractArray{T, 1}   # w⁻_{i,1}
                                     ) where {T<:AbstractFloat}

  N_r  = length(r) - 1
  dN_r = 1:N_r

  for i in dN_r 

    r_iphalf = r[i+1]  # r_{i+1/2}
    r_imhalf = r[i]    # r_{i-1/2}

    # r_{i-3/2}:
    if i ≥ 2
      r_im3halfs = r[i-1] 
    else
      # In case i = 1 we use mirroring to find r_{-1/2}:
      r_im3halfs = r[1] - ( r[2] - r[1] )    # r_{-1/2} = r_{1/2} - ( r_{3/2} - r_{1/2} )
    end

    # r_{i+3/2}:
    if i ≤ N_r - 1
      r_ip3halfs = r[i+2]
    else
      # In case i = N_r we use extrapolation to find r_{N_r + 1 + 1/2}:
      r_ip3halfs = r[N_r+1] + ( r[N_r+1] - r[N_r] )
    end

    # Compute all β's (for the cylindrical r-coordinate, m = 1):
    β_im1_0, β_i_0, β_ip1_0 = Compute_βs( 0, 1, r_im3halfs, r_imhalf, r_iphalf, r_ip3halfs )
    β_im1_1, β_i_1, β_ip1_1 = Compute_βs( 1, 1, r_im3halfs, r_imhalf, r_iphalf, r_ip3halfs )
    β_im1_2, β_i_2, β_ip1_2 = Compute_βs( 2, 1, r_im3halfs, r_imhalf, r_iphalf, r_ip3halfs )

    # Set up 3 × 3 system, Aᵀ * w = RHS, to solve for w's:
    RHS_plus  = @SVector [ 1.0, r_iphalf, r_iphalf^2 ]      # RHS we need to find w⁺
    RHS_minus = @SVector [ 1.0, r_imhalf, r_imhalf^2 ]      # RHS we need to find w⁻

    A   = @SMatrix [ β_im1_0  β_im1_1  β_im1_2;
                     β_i_0    β_i_1    β_i_2;
                     β_ip1_0  β_ip1_1  β_ip1_2 ]

    # Solve for w⁺ = ( w⁺_{i,-1}, w⁺_{i,0}, w⁺_{i,1} ):
    w_plus   = A' \ RHS_plus  # NOTE: A' means Aᵀ, transpose of A. See Eq. (21) in Mignone (2014).

    # Solve for w⁻ = ( w⁻_{i,-1}, w⁻_{i,0}, w⁻_{i,1} ):
    w_minus  = A' \ RHS_minus

    # Unpack results, only saving the coefficients we need:
    w_plus_i_m1,  w_plus_i_0,  w_plus_i_p1[i]  = w_plus
    w_minus_i_m1, w_minus_i_0, w_minus_i_p1[i] = w_minus  

  end 

  return nothing 
end


""" This function computes w⁺_{j,s} and w⁻_{j,s} for the cylindrical Z-coordinate (a Cartesian one),
    for all verical cells j, and with s ∈ (-1, 0, 1). These coefficients are needed 
    for the WENO3 interface reconstruction, and are constants for a given grid. """
function Get_WENO3_w_plus_w_minus_Z!( Z::AbstractArray{T, 1},
                            w_plus_j_p1::AbstractArray{T, 1},  # w⁺_{j,1}
                           w_minus_j_p1::AbstractArray{T, 1}   # w⁻_{j,1}
                                     ) where {T<:AbstractFloat}

  N_Z  = length(Z) - 1
  dN_Z = 1:N_Z

  for j in dN_Z 

    Z_jphalf = Z[j+1]  # Z_{j+1/2}
    Z_jmhalf = Z[j]    # Z_{j-1/2}

    # Z_{j-3/2}:
    if j ≥ 2
      Z_jm3halfs = Z[j-1] 
    else
      # In case j = 1 we use mirroring to find Z_{-1/2}:
      Z_jm3halfs = Z[1] - ( Z[2] - Z[1] )    # Z_{-1/2} = Z_{1/2} - ( Z_{3/2} - Z_{1/2} )
    end

    # Z_{j+3/2}:
    if j ≤ N_Z - 1
      Z_jp3halfs = Z[j+2]
    else
      # In case i = N_r we use extrapolation to find Z_{N_Z + 1 + 1/2}:
      Z_jp3halfs = Z[N_Z+1] + ( Z[N_Z+1] - Z[N_Z] )
    end

    # Compute all β's (for the cylindrical Z-coordinate, m = 0):
    β_im1_0, β_i_0, β_ip1_0 = Compute_βs( 0, 0, Z_jm3halfs, Z_jmhalf, Z_jphalf, Z_jp3halfs )
    β_im1_1, β_i_1, β_ip1_1 = Compute_βs( 1, 0, Z_jm3halfs, Z_jmhalf, Z_jphalf, Z_jp3halfs )
    β_im1_2, β_i_2, β_ip1_2 = Compute_βs( 2, 0, Z_jm3halfs, Z_jmhalf, Z_jphalf, Z_jp3halfs )

    # Set up 3 × 3 system, Aᵀ * w = RHS, to solve for w's:
    RHS_plus  = @SVector [ 1.0, Z_jphalf, Z_jphalf^2 ]      # RHS we need to find w⁺
    RHS_minus = @SVector [ 1.0, Z_jmhalf, Z_jmhalf^2 ]      # RHS we need to find w⁻

    A   = @SMatrix [ β_im1_0  β_im1_1  β_im1_2;
                     β_i_0    β_i_1    β_i_2;
                     β_ip1_0  β_ip1_1  β_ip1_2 ]

    # Solve for w⁺ = ( w⁺_{j,-1}, w⁺_{j,0}, w⁺_{j,1} ):
    w_plus   = A' \ RHS_plus  # NOTE: A' means Aᵀ, transpose of A. See Eq. (21) in Mignone (2014).

    # Solve for w⁻ = ( w⁻_{j,-1}, w⁻_{j,0}, w⁻_{j,1} ):
    w_minus  = A' \ RHS_minus

    # Unpack results, only saving the coefficients we need:
    w_plus_j_m1,  w_plus_j_0,  w_plus_j_p1[j]  = w_plus
    w_minus_j_m1, w_minus_j_0, w_minus_j_p1[j] = w_minus  

  end 

  return nothing 
end


""" This function computes all the WENO3 weights needed to convert 
    between cell-averaged and cell-center values of conservative and 
    primitive variables. Intended for cylindrical coordinates. """
function Get_WENO3_CellAvCenter_weights_Cylindrical!( w_i_m1_r::AbstractArray{T}, q_i_m1_r::AbstractArray{T},
                                                       w_i_0_r::AbstractArray{T}, q_i_0_r::AbstractArray{T},
                                                      w_i_p1_r::AbstractArray{T}, q_i_p1_r::AbstractArray{T}, 

                                                      w_j_m1_Z::AbstractArray{T}, q_j_m1_Z::AbstractArray{T},
                                                       w_j_0_Z::AbstractArray{T}, q_j_0_Z::AbstractArray{T},
                                                      w_j_p1_Z::AbstractArray{T}, q_j_p1_Z::AbstractArray{T},

                                                             r::AbstractArray{T}, rCoM::AbstractArray{T},
                                                             Z::AbstractArray{T}, ZCoM::AbstractArray{T}
                                
                                                  ) where {T<:AbstractFloat}
 
    N_r  = length( rCoM )
    N_Z  = length( ZCoM ) 
    dN_r = 1:N_r  
    dN_Z = 1:N_Z

    @assert N_r ≥ 3 "Need N_r ≥ 3 for WENO3."
    @assert N_Z ≥ 3 "Need N_Z ≥ 3 for WENO3."

    ###############################################################
    #######                                                 #######
    #######           r - C O E F F I C I E N T S           #######
    #######                                                 #######
    ###############################################################

    # Use mirroring to get rCoM at i = 0:
    r_0      = r[1] - ( r[2] - r[1] )    # r_{-1/2} = r_{1/2} - ( r_{3/2} - r_{1/2} )
    rCoM_0   = Get_Single_rCoM_Cylindrical( r[1], r_0 )

    # Use extrapolation to get rCoM at i = N_r + 2:
    r_Np2    = r[N_r+1]  +  ( r[N_r+1] - r[N_r] )
    rCoM_Np1 = Get_Single_rCoM_Cylindrical( r_Np2, r[N_r+1] )

    # Loop over cells: 
    for i in dN_r
        # Cell geometric center:
        rc_i = ( r[i] + r[i+1] ) / 2

        # ⟨Δr⟩_{i,i-1}, ⟨Δr⟩_{i,i}, and ⟨Δr⟩_{i,i+1}:
        if i ≥ 2 && i ≤ (N_r - 1)
            Δr_i_im1   = rCoM[i-1] - rc_i
            Δr_i_i     = rCoM[i]   - rc_i
            Δr_i_ip1   = rCoM[i+1] - rc_i
        elseif i == 1
            Δr_i_im1   = rCoM_0   -  rc_i
            Δr_i_i     = rCoM[1]  -  rc_i
            Δr_i_ip1   = rCoM[2]  -  rc_i
        elseif i == N_r
            Δr_i_im1   = rCoM[N_r-1] - rc_i
            Δr_i_i     = rCoM[N_r]   - rc_i
            Δr_i_ip1   = rCoM_Np1    - rc_i
        end 

        # ⟨(Δr)²⟩_{i,i}:
        Δr2        = r[i+1]^2 - r[i]^2
        Δr3        = r[i+1]^3 - r[i]^3
        Δr4        = r[i+1]^4 - r[i]^4
        Δrsq_i_i   = 0.5 * (Δr4 / Δr2)  -  (4/3) * (Δr3 / Δr2) * rc_i  +  rc_i^2 

        # ⟨(Δr)²⟩_{i,i+1}:
        if i ≤ (N_r - 1)
            Δr2_p1     = r[i+2]^2 - r[i+1]^2
            Δr3_p1     = r[i+2]^3 - r[i+1]^3
            Δr4_p1     = r[i+2]^4 - r[i+1]^4
        else
            Δr2_p1     = r_Np2^2 - r[N_r+1]^2
            Δr3_p1     = r_Np2^3 - r[N_r+1]^3
            Δr4_p1     = r_Np2^4 - r[N_r+1]^4
        end
        Δrsq_i_ip1 = 0.5 * (Δr4_p1 / Δr2_p1)  -  (4/3) * (Δr3_p1 / Δr2_p1) * rc_i  +  rc_i^2 

        # ⟨(Δr)²⟩_{i,i-1}:
        if i ≥ 2
            Δr2_m1     = r[i]^2 - r[i-1]^2
            Δr3_m1     = r[i]^3 - r[i-1]^3
            Δr4_m1     = r[i]^4 - r[i-1]^4
        else
            Δr2_m1     = r[1]^2 - r_0^2
            Δr3_m1     = r[1]^3 - r_0^3
            Δr4_m1     = r[1]^4 - r_0^4
        end
        Δrsq_i_im1 = 0.5 * (Δr4_m1 / Δr2_m1)  -  (4/3) * (Δr3_m1 / Δr2_m1) * rc_i  +  rc_i^2 

        # Compute the weights w_{i,-1}, w_{i,0}, and w_{i,1},
        # needed to go go from ⟨U⟩ᵢ → U(rCᵢ) (cell averaged → cell center value):
        w_i_m1_r[i], w_i_0_r[i], w_i_p1_r[i] = Compute_w_k( Δr_i_im1, Δr_i_i, Δr_i_ip1,  
                                                            Δrsq_i_im1, Δrsq_i_i, Δrsq_i_ip1 )

        # Check that the weights add up to 1:
        sum_w_i = w_i_m1_r[i] +  w_i_0_r[i] + w_i_p1_r[i]
        @assert isapprox(sum_w_i, 1; atol = 1e-12) "Error: w_{i,-1} + w_{i,0} + w_{i,1} = $(sum_w_i) at i = $(i). They must add to 1."

        # Compute weights q_{i,-1}, q_{i,0}, and q_{i,1},
        # needed to go go from W(rCᵢ) → ⟨W⟩ᵢ (cell center → cell average value),
        # for a primitive variable W:
        if i ≥ 2 && i ≤ (N_r - 1)
            rc_im1 = ( r[i-1] + r[i] )   / 2
            rc_ip1 = ( r[i+1] + r[i+2] ) / 2
            q_i_m1_r[i], q_i_0_r[i], q_i_p1_r[i] =  Compute_q_k( rc_im1, rc_i, rc_ip1,  Δrsq_i_i )
        elseif i == 1
            rc_im1 = ( r_0  + r[1] ) / 2
            rc_ip1 = ( r[2] + r[3] ) / 2
            q_i_m1_r[i], q_i_0_r[i], q_i_p1_r[i] =  Compute_q_k( rc_im1, rc_i, rc_ip1,  Δrsq_i_i )
        elseif i == N_r 
            rc_im1 = ( r[N_r-1] + r[N_r] ) / 2
            rc_ip1 = ( r[N_r+1] + r_Np2 )  / 2
            q_i_m1_r[i], q_i_0_r[i], q_i_p1_r[i] =  Compute_q_k( rc_im1, rc_i, rc_ip1,  Δrsq_i_i )
        end

        # Check that the weights add up to 1:
        sum_q_i = q_i_m1_r[i] + q_i_0_r[i] + q_i_p1_r[i]
        @assert isapprox(sum_q_i, 1; atol = 1e-12) "Error: q_{i,-1} + q_{i,0} + q_{i,1} = $(sum_q_i) at i = $(i). They must add to 1."

    end # End of loop 

    ###############################################################
    #######                                                 #######
    #######           Z - C O E F F I C I E N T S           #######
    #######                                                 #######
    ###############################################################

    # Use mirroring/extrapolation to get ZCoM at j = 0:
    Z_0      = Z[1] - ( Z[2] - Z[1] )    # Z_{-1/2} = Z_{1/2} - ( Z_{3/2} - Z_{1/2} )
    ZCoM_0   = ( Z_0  +  Z[1] ) / 2

    # Use extrapolation to get ZCoM at j = N_Z + 2:
    Z_Np2    = Z[N_Z+1]  +  ( Z[N_Z+1] - Z[N_Z] )
    ZCoM_Np1 = ( Z_Np2  +  Z[N_Z+1] ) / 2

    for j in dN_Z
        # Cell geometric center (= centroid in this case):
        Zc_j = ( Z[j] + Z[j+1] ) / 2

        # ⟨ΔZ⟩_{j,j-1} and ⟨ΔZ⟩_{j,j+1}:
        if j ≥ 2 && j ≤ (N_Z - 1)
            ΔZ_j_jm1   = ZCoM[j-1] - Zc_j
            ΔZ_j_jp1   = ZCoM[j+1] - Zc_j
        elseif j == 1
            ΔZ_j_jm1   = ZCoM_0  - Zc_j
            ΔZ_j_jp1   = ZCoM[2] - Zc_j
        elseif j == N_Z
            ΔZ_j_jm1   = ZCoM[N_Z-1] - Zc_j
            ΔZ_j_jp1   = ZCoM_Np1    - Zc_j
        end

        # ⟨(ΔZ)²⟩_{j,j}:
        ΔZsq_j_j   = (1/12) * ( Z[j+1] - Z[j] )^2

        # ⟨(ΔZ)²⟩_{j,j+1}:
        if j ≤ (N_Z - 1)
            ΔZ_p1   = Z[j+2]   - Z[j+1]
            ΔZ2_p1  = Z[j+2]^2 - Z[j+1]^2
            ΔZ3_p1  = Z[j+2]^3 - Z[j+1]^3
        elseif j == N_Z
            ΔZ_p1   = Z_Np2   - Z[N_Z+1]
            ΔZ2_p1  = Z_Np2^2 - Z[N_Z+1]^2
            ΔZ3_p1  = Z_Np2^3 - Z[N_Z+1]^3
        end
        ΔZsq_j_jp1 = (1/3) * (ΔZ3_p1 / ΔZ_p1)  -  (ΔZ2_p1 / ΔZ_p1) * Zc_j  +  Zc_j^2 

        # ⟨(ΔZ)²⟩_{j,j-1}:
        if j ≥ 2
            ΔZ_m1   = Z[j]   - Z[j-1]
            ΔZ2_m1  = Z[j]^2 - Z[j-1]^2
            ΔZ3_m1  = Z[j]^3 - Z[j-1]^3
        else
            ΔZ_m1   = Z[1]   - Z_0
            ΔZ2_m1  = Z[1]^2 - Z_0^2
            ΔZ3_m1  = Z[1]^3 - Z_0^3
        end
        ΔZsq_j_jm1 = (1/3) * (ΔZ3_m1 / ΔZ_m1)  -  (ΔZ2_m1 / ΔZ_m1) * Zc_j  +  Zc_j^2 

        # ⟨ΔZ⟩_{j,j} = ZCⱼ - ZCⱼ = 0:
        ΔZ_j_j = 0.0

        # Compute the weights w_{j,-1}, w_{j,0}, and w_{j,1},
        # needed to go go from ⟨U⟩ⱼ → U(ZCⱼ) (cell averaged → cell centroid value):
        w_j_m1_Z[j], w_j_0_Z[j], w_j_p1_Z[j] = Compute_w_k( ΔZ_j_jm1, ΔZ_j_j, ΔZ_j_jp1,  
                                                            ΔZsq_j_jm1, ΔZsq_j_j, ΔZsq_j_jp1 )

        # Check that the weights add up to 1:
        sum_w_j = w_j_m1_Z[j] +  w_j_0_Z[j] + w_j_p1_Z[j]
        @assert isapprox(sum_w_j, 1; atol = 1e-12) "Error: w_{j,-1} + w_{j,0} + w_{j,1} = $(sum_w_j) at j = $(j). They must add to 1."

        # Compute weights q_{j,-1}, q_{j,0}, and q_{j,1},
        # needed to go go from W(ZCⱼ) → ⟨W⟩ⱼ (cell center → cell average value),
        # for a primitive variable W:
        if j ≥ 2 && j ≤ (N_Z - 1)
            Zc_jm1 = ( Z[j-1] + Z[j] )   / 2
            Zc_jp1 = ( Z[j+1] + Z[j+2] ) / 2
            q_j_m1_Z[j], q_j_0_Z[j], q_j_p1_Z[j] =  Compute_q_k( Zc_jm1, Zc_j, Zc_jp1, ΔZsq_j_j )
        elseif j == 1
            Zc_jm1 = ( Z_0  + Z[1] ) / 2
            Zc_jp1 = ( Z[2] + Z[3] ) / 2
            q_j_m1_Z[j], q_j_0_Z[j], q_j_p1_Z[j] =  Compute_q_k( Zc_jm1, Zc_j, Zc_jp1, ΔZsq_j_j )
        elseif j == N_Z
            Zc_jm1 = ( Z[N_Z-1] + Z[N_Z] ) / 2
            Zc_jp1 = ( Z[N_Z+1] + Z_Np2  ) / 2
            q_j_m1_Z[j], q_j_0_Z[j], q_j_p1_Z[j] =  Compute_q_k( Zc_jm1, Zc_j, Zc_jp1, ΔZsq_j_j )
        end

        # Check that the weights add up to 1:
        sum_q_j = q_j_m1_Z[j] + q_j_0_Z[j] + q_j_p1_Z[j]
        @assert isapprox(sum_q_j, 1; atol = 1e-12) "Error: q_{j,-1} + q_{j,0} + q_{j,1} = $(sum_q_j) at j = $(j). They must add to 1."

    end # End of loop 

    return nothing 
end


""" This function creates a cylindrical grid,
    for r ∈ [0, rmax], Z ∈ [0, Zmax]. 

    A linear N_r_lin × N_Z_lin grid is used for 
    r < rmax_lin, and Z < Zmax_lin. A logarithmic 
    grid is used outside this region, stretching to 
    rmax and Zmax, and with N_r_log × N_Z_log cells. """
function SetupCylindricalGrid( rmax::T,     rmax_lin::T,
                               Zmax::T,     Zmax_lin::T,
                            N_r_lin::Int64,  N_r_log::Int64,
                            N_Z_lin::Int64,  N_Z_log::Int64
                              ) where {T<:AbstractFloat}

    N_r = N_r_lin + N_r_log
    N_Z = N_Z_lin + N_Z_log

    @assert rmax > 0.0 "We must have rmax > 0.0."
    @assert Zmax > 0.0 "We must have Zmax > 0.0."
    @assert N_r  ≥ 3   "We need N_r ≥ 3"
    @assert N_Z  ≥ 3   "We need N_Z ≥ 3"

    # Initialize arrays:
    ΔV           = zeros( T, N_r, N_Z )      # Cell volumes 
    inv_ΔV       = zeros( T, N_r, N_Z )      # Inverse cell volumes, 1 / ΔV, often needed in advection updates
    A_r          = zeros( T, N_r + 1, N_Z )  # r-direction cell interface areas 
    A_Z          = zeros( T, N_r, N_Z + 1 )  # Z-direction cell interface areas
    r            = zeros( T, N_r + 1 )       # r-interfaces
    rCoM         = zeros( T, N_r )           # r-direction cell centers of mass 
    Z            = zeros( T, N_Z + 1 )       # Z-interfaces
    ZCoM         = zeros( T, N_Z )           # Z-direction cell centers of mass 
    inv_r_cell   = zeros( T, N_r ) 

    # Initialize WENO3 weights for conversion between cell-averages & cell-centers:
    w_i_m1_r     = zeros( T, N_r )
    w_i_0_r      = zeros( T, N_r )
    w_i_p1_r     = zeros( T, N_r )
    w_j_m1_Z     = zeros( T, N_Z )
    w_j_0_Z      = zeros( T, N_Z )
    w_j_p1_Z     = zeros( T, N_Z )

    q_i_m1_r     = zeros( T, N_r )
    q_i_0_r      = zeros( T, N_r )
    q_i_p1_r     = zeros( T, N_r )
    q_j_m1_Z     = zeros( T, N_Z )
    q_j_0_Z      = zeros( T, N_Z )
    q_j_p1_Z     = zeros( T, N_Z )

    # Initialize WENO3 weights for interface reconstruction:  
    w_plus_i_p1  = zeros( T, N_r )
    w_minus_i_p1 = zeros( T, N_r )

    w_plus_j_p1  = zeros( T, N_Z )
    w_minus_j_p1 = zeros( T, N_Z )

    # Create linear and logarithmic grid: 
    r_lin        = LinRange( 0.0, rmax_lin, N_r_lin + 1 )
    Z_lin        = LinRange( 0.0, Zmax_lin, N_Z_lin + 1 )

    if N_r_log ≥ 2
        q_r          = (rmax / rmax_lin)^( 1 / N_r_log )          # Geometric factor 
        r_log        = [ rmax_lin * q_r^(i) for i in 0:N_r_log ]  # N_r_log + 1 interfaces, starting at r = rmax_lin
    end
    if N_Z_log ≥ 2
        q_Z          = (Zmax / Zmax_lin)^( 1 / N_Z_log )          # Geometric factor 
        Z_log        = [ Zmax_lin * q_Z^(i) for i in 0:N_Z_log ]  # N_Z_log + 1 interfaces, starting at Z = Zmax_lin
    end

    # Fill interface values:
    if N_r_log ≥ 2
        r  .= vcat( r_lin, r_log[2:end] )
    else
        r  .= r_lin
    end
    if N_Z_log ≥ 2
        Z  .= vcat( Z_lin, Z_log[2:end] )
    else
        Z  .= Z_lin
    end

    # Get rCoM, ZCoM, ΔV, A_r, A_Z, inv_r_cell ≡ ⟨1/r⟩, and WENO3 weights:
    Get_rCoM_Cylindrical!( r, rCoM )
    Get_ZCoM!( Z, ZCoM ) 
    Get_ΔV_Cylindrical!( r, Z, ΔV )
    Get_A_r_Cylindrical!( r, Z, A_r )
    Get_A_Z_Cylindrical!( r, Z, A_Z )
    Get_inv_r_cell_Cylindrical!( inv_r_cell, r )
    Get_WENO3_CellAvCenter_weights_Cylindrical!( w_i_m1_r, q_i_m1_r, w_i_0_r, q_i_0_r, w_i_p1_r, q_i_p1_r, 
                                                 w_j_m1_Z, q_j_m1_Z, w_j_0_Z, q_j_0_Z, w_j_p1_Z, q_j_p1_Z, 
                                                 r, rCoM, Z, ZCoM )
    Get_WENO3_w_plus_w_minus_r!( r, w_plus_i_p1, w_minus_i_p1 )
    Get_WENO3_w_plus_w_minus_Z!( Z, w_plus_j_p1, w_minus_j_p1 )

    #println( w_plus_j_p1[1:3], w_minus_j_p1[1:3])
    #println(r[2])

    # Get 1 / ΔV:
    inv_ΔV    .= T(1.0) ./ ΔV

    return GeometryCylindrical{T}( Val(:Cylindrical), r, Z, rCoM, ZCoM, ΔV, inv_ΔV, A_r, A_Z, inv_r_cell, 
                                   w_i_m1_r, w_i_0_r, w_i_p1_r, w_j_m1_Z, w_j_0_Z, w_j_p1_Z,
                                   q_i_m1_r, q_i_0_r, q_i_p1_r, q_j_m1_Z, q_j_0_Z, q_j_p1_Z,
                                   w_plus_i_p1, w_minus_i_p1, 
                                   w_plus_j_p1, w_minus_j_p1  )
end


###############################################################
#######                                                 #######
#######           C A R T E S I A N  G R I D            #######
#######                                                 #######
###############################################################

# Below are the equivalent functions for setting up a Cartesian grid.
# Some of it can be borrowed from the cylindrical Z-coordinate.

""" This function populates ΔV with the cell volumes, 
    *given* interface values X and Y.
    
    Works for Cartesian coordinates. """
function Get_ΔV_Cartesian!( X::AbstractArray{T}, 
                            Y::AbstractArray{T},
                           ΔV::AbstractArray{T, 2} ) where {T<:AbstractFloat}

    N_X  = length( X ) - 1  # Number of X grid points         
    N_Y  = length( Y ) - 1  # Number of Y grid points 
    dN_X = 1:N_X
    dN_Y = 1:N_Y 

    @assert size( ΔV ) == (N_X, N_Y)

    @inbounds for j in dN_Y
        ΔY = Y[j+1] - Y[j]
        @simd for i in dN_X
            ΔV[i,j] = ( X[i+1] - X[i] ) * ΔY
        end
    end

    return nothing
end


""" This function populates A_X with interface cell areas, 
    along the X-direction, given interface values X and Y.
    
    Works for Cartesian coordinates. (NOTE: No dependence on Y,
    but for compatabilitiy with the cylindrical-focused hydro code, 
    we still use a matrix for A_X). """
function Get_A_X_Cartesian!( X::AbstractArray{T}, 
                             Y::AbstractArray{T},
                           A_X::AbstractArray{T, 2} ) where {T<:AbstractFloat}

    N_X  = length( X ) - 1  # Number of X grid points         
    N_Y  = length( Y ) - 1  # Number of Y grid points 
    dN_Y = 1:N_Y 

    @assert size( A_X ) == (N_X + 1, N_Y)

    @inbounds for j in dN_Y 
        ΔY = Y[j+1] - Y[j]
        @simd for i in 1:(N_X+1)
            A_X[i,j] = ΔY
        end
    end

    return nothing
end


""" This function populates A_X with interface cell areas, 
    along the Y-direction, given interface values X and Y.
    
    Works for Cartesian coordinates. (NOTE: No dependence on X,
    but for compatabilitiy with the cylindrical-focused hydro code, 
    we still use a matrix for A_Y). """
function Get_A_Y_Cartesian!( X::AbstractArray{T}, 
                             Y::AbstractArray{T},
                           A_Y::AbstractArray{T, 2} ) where {T<:AbstractFloat}

    N_X  = length( X ) - 1  # Number of X grid points         
    N_Y  = length( Y ) - 1  # Number of Y grid points 
    dN_X = 1:N_X 

    @assert size( A_Y ) == (N_X, N_Y + 1)

    @inbounds for j in 1:(N_Y+1) 
        @simd for i in dN_X
            ΔX       = X[i+1] - X[i]
            A_Y[i,j] = ΔX
        end
    end

    return nothing
end


""" This function computes all the WENO3 weights needed to convert 
    between cell-averaged and cell-center values of conservative and 
    primitive variables. """
function Get_WENO3_CellAvCenter_weights_Cartesian!( w_i_m1_X::AbstractArray{T}, q_i_m1_X::AbstractArray{T},
                                                     w_i_0_X::AbstractArray{T}, q_i_0_X::AbstractArray{T},
                                                    w_i_p1_X::AbstractArray{T}, q_i_p1_X::AbstractArray{T}, 

                                                    w_j_m1_Y::AbstractArray{T}, q_j_m1_Y::AbstractArray{T},
                                                     w_j_0_Y::AbstractArray{T}, q_j_0_Y::AbstractArray{T},
                                                    w_j_p1_Y::AbstractArray{T}, q_j_p1_Y::AbstractArray{T},

                                                           X::AbstractArray{T}, XCoM::AbstractArray{T},
                                                           Y::AbstractArray{T}, YCoM::AbstractArray{T}
                                
                                                   ) where {T<:AbstractFloat}
 
    N_X  = length( XCoM )
    N_Y  = length( YCoM ) 
    dN_X = 1:N_X  
    dN_Y = 1:N_Y

    @assert N_X ≥ 3 "Need N_X ≥ 3 for WENO3."
    @assert N_Y ≥ 3 "Need N_Y ≥ 3 for WENO3."

    ###############################################################
    #######                                                 #######
    #######           X - C O E F F I C I E N T S           #######
    #######                                                 #######
    ###############################################################

    # Use mirroring/extrapolation to get XCoM at i = 0:
    X_0      = X[1] - ( X[2] - X[1] )    # X_{-1/2} = X_{1/2} - ( X_{3/2} - X_{1/2} )
    XCoM_0   = ( X_0  +  X[1] ) / 2

    # Use extrapolation to get XCoM at i = N_X + 2:
    X_Np2    = X[N_X+1]  +  ( X[N_X+1] - X[N_X] )
    XCoM_Np1 = ( X_Np2  +  X[N_X+1] ) / 2

    for i in dN_X
        # Cell geometric center (= centroid in this case):
        Xc_i = ( X[i] + X[i+1] ) / 2

        # ⟨ΔX⟩_{i,i-1} and ⟨ΔX⟩_{i,i+1}:
        if i ≥ 2 && i ≤ (N_X - 1)
            ΔX_i_im1   = XCoM[i-1] - Xc_i
            ΔX_i_ip1   = XCoM[i+1] - Xc_i
        elseif i == 1
            ΔX_i_im1   = XCoM_0  - Xc_i
            ΔX_i_ip1   = XCoM[2] - Xc_i
        elseif i == N_X
            ΔX_i_im1   = XCoM[N_X-1] - Xc_i
            ΔX_i_ip1   = XCoM_Np1    - Xc_i
        end

        # ⟨(ΔX)²⟩_{i,i}:
        ΔXsq_i_i   = (1/12) * ( X[i+1] - X[i] )^2

        # ⟨(ΔX)²⟩_{i,i+1}:
        if i ≤ (N_X - 1)
            ΔX_p1   = X[i+2]   - X[i+1]
            ΔX2_p1  = X[i+2]^2 - X[i+1]^2
            ΔX3_p1  = X[i+2]^3 - X[i+1]^3
        elseif i == N_X
            ΔX_p1   = X_Np2   - X[N_X+1]
            ΔX2_p1  = X_Np2^2 - X[N_X+1]^2
            ΔX3_p1  = X_Np2^3 - X[N_X+1]^3
        end
        ΔXsq_i_ip1 = (1/3) * (ΔX3_p1 / ΔX_p1)  -  (ΔX2_p1 / ΔX_p1) * Xc_i  +  Xc_i^2 

        # ⟨(ΔX)²⟩_{i,i-1}:
        if i ≥ 2
            ΔX_m1   = X[i]   - X[i-1]
            ΔX2_m1  = X[i]^2 - X[i-1]^2
            ΔX3_m1  = X[i]^3 - X[i-1]^3
        else
            ΔX_m1   = X[1]   - X_0
            ΔX2_m1  = X[1]^2 - X_0^2
            ΔX3_m1  = X[1]^3 - X_0^3
        end
        ΔXsq_i_im1 = (1/3) * (ΔX3_m1 / ΔX_m1)  -  (ΔX2_m1 / ΔX_m1) * Xc_i  +  Xc_i^2 

        # ⟨ΔX⟩_{i,i} = XCⱼ - XCⱼ = 0:
        ΔX_i_i = 0.0

        # Compute the weights w_{i,-1}, w_{i,0}, and w_{i,1},
        # needed to go go from ⟨U⟩ⱼ → U(XCⱼ) (cell averaged → cell centroid value):
        w_i_m1_X[i], w_i_0_X[i], w_i_p1_X[i] = Compute_w_k( ΔX_i_im1, ΔX_i_i, ΔX_i_ip1,  
                                                            ΔXsq_i_im1, ΔXsq_i_i, ΔXsq_i_ip1 )

        # Check that the weights add up to 1:
        sum_w_i = w_i_m1_X[i] +  w_i_0_X[i] + w_i_p1_X[i]
        @assert isapprox(sum_w_i, 1; atol = 1e-12) "Error: w_{i,-1} + w_{i,0} + w_{i,1} = $(sum_w_i) at i = $(i). They must add to 1."

        # Compute weights q_{i,-1}, q_{i,0}, and q_{i,1},
        # needed to go go from W(XCⱼ) → ⟨W⟩ⱼ (cell center → cell average value),
        # for a primitive variable W:
        if i ≥ 2 && i ≤ (N_X - 1)
            Xc_im1 = ( X[i-1] + X[i] )   / 2
            Xc_ip1 = ( X[i+1] + X[i+2] ) / 2
            q_i_m1_X[i], q_i_0_X[i], q_i_p1_X[i] =  Compute_q_k( Xc_im1, Xc_i, Xc_ip1, ΔXsq_i_i )
        elseif i == 1
            Xc_im1 = ( X_0  + X[1] ) / 2
            Xc_ip1 = ( X[2] + X[3] ) / 2
            q_i_m1_X[i], q_i_0_X[i], q_i_p1_X[i] =  Compute_q_k( Xc_im1, Xc_i, Xc_ip1, ΔXsq_i_i )
        elseif i == N_X
            Xc_im1 = ( X[N_X-1] + X[N_X] ) / 2
            Xc_ip1 = ( X[N_X+1] + X_Np2  ) / 2
            q_i_m1_X[i], q_i_0_X[i], q_i_p1_X[i] =  Compute_q_k( Xc_im1, Xc_i, Xc_ip1, ΔXsq_i_i )
        end

        # Check that the weights add up to 1:
        sum_q_i = q_i_m1_X[i] + q_i_0_X[i] + q_i_p1_X[i]
        @assert isapprox(sum_q_i, 1; atol = 1e-12) "Error: q_{i,-1} + q_{i,0} + q_{i,1} = $(sum_q_i) at i = $(i). They must add to 1."

    end # End of loop 

    ###############################################################
    #######                                                 #######
    #######           Y - C O E F F I C I E N T S           #######
    #######                                                 #######
    ###############################################################

    # Use mirroring/extrapolation to get YCoM at j = 0:
    Y_0      = Y[1] - ( Y[2] - Y[1] )    # Y_{-1/2} = Y_{1/2} - ( Y_{3/2} - Y_{1/2} )
    YCoM_0   = ( Y_0  +  Y[1] ) / 2

    # Use extrapolation to get YCoM at j = N_Y + 2:
    Y_Np2    = Y[N_Y+1]  +  ( Y[N_Y+1] - Y[N_Y] )
    YCoM_Np1 = ( Y_Np2  +  Y[N_Y+1] ) / 2

    for j in dN_Y
        # Cell geometric center (= centroid in this case):
        Yc_j = ( Y[j] + Y[j+1] ) / 2

        # ⟨ΔY⟩_{j,j-1} and ⟨ΔY⟩_{j,j+1}:
        if j ≥ 2 && j ≤ (N_Y - 1)
            ΔY_j_jm1   = YCoM[j-1] - Yc_j
            ΔY_j_jp1   = YCoM[j+1] - Yc_j
        elseif j == 1
            ΔY_j_jm1   = YCoM_0  - Yc_j
            ΔY_j_jp1   = YCoM[2] - Yc_j
        elseif j == N_Y
            ΔY_j_jm1   = YCoM[N_Y-1] - Yc_j
            ΔY_j_jp1   = YCoM_Np1    - Yc_j
        end

        # ⟨(ΔY)²⟩_{j,j}:
        ΔYsq_j_j   = (1/12) * ( Y[j+1] - Y[j] )^2

        # ⟨(ΔY)²⟩_{j,j+1}:
        if j ≤ (N_Y - 1)
            ΔY_p1   = Y[j+2]   - Y[j+1]
            ΔY2_p1  = Y[j+2]^2 - Y[j+1]^2
            ΔY3_p1  = Y[j+2]^3 - Y[j+1]^3
        elseif j == N_Y
            ΔY_p1   = Y_Np2   - Y[N_Y+1]
            ΔY2_p1  = Y_Np2^2 - Y[N_Y+1]^2
            ΔY3_p1  = Y_Np2^3 - Y[N_Y+1]^3
        end
        ΔYsq_j_jp1 = (1/3) * (ΔY3_p1 / ΔY_p1)  -  (ΔY2_p1 / ΔY_p1) * Yc_j  +  Yc_j^2 

        # ⟨(ΔY)²⟩_{j,j-1}:
        if j ≥ 2
            ΔY_m1   = Y[j]   - Y[j-1]
            ΔY2_m1  = Y[j]^2 - Y[j-1]^2
            ΔY3_m1  = Y[j]^3 - Y[j-1]^3
        else
            ΔY_m1   = Y[1]   - Y_0
            ΔY2_m1  = Y[1]^2 - Y_0^2
            ΔY3_m1  = Y[1]^3 - Y_0^3
        end
        ΔYsq_j_jm1 = (1/3) * (ΔY3_m1 / ΔY_m1)  -  (ΔY2_m1 / ΔY_m1) * Yc_j  +  Yc_j^2 

        # ⟨ΔY⟩_{j,j} = YCⱼ - YCⱼ = 0:
        ΔY_j_j = 0.0

        # Compute the weights w_{j,-1}, w_{j,0}, and w_{j,1},
        # needed to go go from ⟨U⟩ⱼ → U(YCⱼ) (cell averaged → cell centroid value):
        w_j_m1_Y[j], w_j_0_Y[j], w_j_p1_Y[j] = Compute_w_k( ΔY_j_jm1, ΔY_j_j, ΔY_j_jp1,  
                                                            ΔYsq_j_jm1, ΔYsq_j_j, ΔYsq_j_jp1 )

        # Check that the weights add up to 1:
        sum_w_j = w_j_m1_Y[j] +  w_j_0_Y[j] + w_j_p1_Y[j]
        @assert isapprox(sum_w_j, 1; atol = 1e-12) "Error: w_{j,-1} + w_{j,0} + w_{j,1} = $(sum_w_j) at j = $(j). They must add to 1."

        # Compute weights q_{j,-1}, q_{j,0}, and q_{j,1},
        # needed to go go from W(YCⱼ) → ⟨W⟩ⱼ (cell center → cell average value),
        # for a primitive variable W:
        if j ≥ 2 && j ≤ (N_Y - 1)
            Yc_jm1 = ( Y[j-1] + Y[j] )   / 2
            Yc_jp1 = ( Y[j+1] + Y[j+2] ) / 2
            q_j_m1_Y[j], q_j_0_Y[j], q_j_p1_Y[j] =  Compute_q_k( Yc_jm1, Yc_j, Yc_jp1, ΔYsq_j_j )
        elseif j == 1
            Yc_jm1 = ( Y_0  + Y[1] ) / 2
            Yc_jp1 = ( Y[2] + Y[3] ) / 2
            q_j_m1_Y[j], q_j_0_Y[j], q_j_p1_Y[j] =  Compute_q_k( Yc_jm1, Yc_j, Yc_jp1, ΔYsq_j_j )
        elseif j == N_Y
            Yc_jm1 = ( Y[N_Y-1] + Y[N_Y] ) / 2
            Yc_jp1 = ( Y[N_Y+1] + Y_Np2  ) / 2
            q_j_m1_Y[j], q_j_0_Y[j], q_j_p1_Y[j] =  Compute_q_k( Yc_jm1, Yc_j, Yc_jp1, ΔYsq_j_j )
        end

        # Check that the weights add up to 1:
        sum_q_j = q_j_m1_Y[j] + q_j_0_Y[j] + q_j_p1_Y[j]
        @assert isapprox(sum_q_j, 1; atol = 1e-12) "Error: q_{j,-1} + q_{j,0} + q_{j,1} = $(sum_q_j) at j = $(j). They must add to 1."

    end # End of loop 

    return nothing 
end


""" This function creates a Cartesian grid,
    for X ∈ [0, Xmax], Y ∈ [0, Ymax]. 

    A linear N_X_lin × N_Y_lin grid is used for 
    X < Xmax_lin, and Y < Ymax_lin. A logarithmic 
    grid is used outside this region, stretching to 
    Xmax and Ymax, and with N_X_log × N_Y_log cells. """
function SetupCartesianGrid( Xmax::T,     Xmax_lin::T,
                             Ymax::T,     Ymax_lin::T,
                          N_X_lin::Int64,  N_X_log::Int64,
                          N_Y_lin::Int64,  N_Y_log::Int64
                              ) where {T<:AbstractFloat}

    N_X = N_X_lin + N_X_log
    N_Y = N_Y_lin + N_Y_log

    @assert Xmax > 0.0 "We must have Xmax > 0.0."
    @assert Ymax > 0.0 "We must have Ymax > 0.0."
    @assert N_X  ≥ 3   "We need N_X ≥ 3"
    @assert N_Y  ≥ 3   "We need N_Y ≥ 3"

    # Initialize arrays:
    ΔV           = zeros( T, N_X, N_Y )      # Cell volumes 
    inv_ΔV       = zeros( T, N_X, N_Y )      # Inverse cell volumes, 1 / ΔV, often needed in advection updates
    A_X          = zeros( T, N_X + 1, N_Y )  # X-direction cell interface areas 
    A_Y          = zeros( T, N_X, N_Y + 1 )  # Y-direction cell interface areas
    X            = zeros( T, N_X + 1 )       # X-interfaces
    XCoM         = zeros( T, N_X )           # X-direction cell centers of mass 
    Y            = zeros( T, N_Y + 1 )       # Y-interfaces
    YCoM         = zeros( T, N_Y )           # Y-direction cell centers of mass 
    inv_r_cell   = zeros( T, N_X )           # NOT USED IN CARTESIAN (but we still need as an argument in various hydro functions)

    # Initialize WENO3 weights for conversion between cell-averages & cell-centers:
    w_i_m1_X     = zeros( T, N_X )
    w_i_0_X      = zeros( T, N_X )
    w_i_p1_X     = zeros( T, N_X )
    w_j_m1_Y     = zeros( T, N_Y )
    w_j_0_Y      = zeros( T, N_Y )
    w_j_p1_Y     = zeros( T, N_Y )

    q_i_m1_X     = zeros( T, N_X )
    q_i_0_X      = zeros( T, N_X )
    q_i_p1_X     = zeros( T, N_X )
    q_j_m1_Y     = zeros( T, N_Y )
    q_j_0_Y      = zeros( T, N_Y )
    q_j_p1_Y     = zeros( T, N_Y )

    # Initialize WENO3 weights for interface reconstruction:  
    w_plus_i_p1  = zeros( T, N_X )
    w_minus_i_p1 = zeros( T, N_X )

    w_plus_j_p1  = zeros( T, N_Y )
    w_minus_j_p1 = zeros( T, N_Y )

    # Create linear and logarithmic grid: 
    X_lin        = LinRange( 0.0, Xmax_lin, N_X_lin + 1 )
    Y_lin        = LinRange( 0.0, Ymax_lin, N_Y_lin + 1 )

    if N_X_log ≥ 2
        q_X          = (Xmax / Xmax_lin)^( 1 / N_X_log )          # Geometric factor 
        X_log        = [ Xmax_lin * q_X^(i) for i in 0:N_X_log ]  # N_X_log + 1 interfaces, starting at X = Xmax_lin
    end
    if N_Y_log ≥ 2
        q_Y          = (Ymax / Ymax_lin)^( 1 / N_Y_log )          # Geometric factor 
        Y_log        = [ Ymax_lin * q_Y^(i) for i in 0:N_Y_log ]  # N_Y_log + 1 interfaces, starting at Y = Ymax_lin
    end

    # Fill interface values:
    if N_X_log ≥ 2
        X  .= vcat( X_lin, X_log[2:end] )
    else
        X  .= X_lin
    end
    if N_Y_log ≥ 2
        Y  .= vcat( Y_lin, Y_log[2:end] )
    else
        Y  .= Y_lin
    end

    # Get XCoM, YCoM, ΔV, A_X, A_Y, inv_r_cell ≡ ⟨1/r⟩, and WENO3 weights:
    Get_ZCoM!( X, XCoM )
    Get_ZCoM!( Y, YCoM ) 
    Get_ΔV_Cartesian!( X, Y, ΔV )
    Get_A_X_Cartesian!( X, Y, A_X )
    Get_A_Y_Cartesian!( X, Y, A_Y )
    Get_WENO3_CellAvCenter_weights_Cartesian!( w_i_m1_X, q_i_m1_X, w_i_0_X, q_i_0_X, w_i_p1_X, q_i_p1_X, 
                                               w_j_m1_Y, q_j_m1_Y, w_j_0_Y, q_j_0_Y, w_j_p1_Y, q_j_p1_Y, 
                                               X, XCoM, Y, YCoM )
    Get_WENO3_w_plus_w_minus_Z!( X, w_plus_i_p1, w_minus_i_p1 )
    Get_WENO3_w_plus_w_minus_Z!( Y, w_plus_j_p1, w_minus_j_p1 )

    # Get 1 / ΔV:
    inv_ΔV    .= T(1.0) ./ ΔV

    return GeometryCartesian{T}( Val(:Cartesian), X, Y, XCoM, YCoM, ΔV, inv_ΔV, A_X, A_Y, inv_r_cell, 
                                 w_i_m1_X, w_i_0_X, w_i_p1_X, w_j_m1_Y, w_j_0_Y, w_j_p1_Y,
                                 q_i_m1_X, q_i_0_X, q_i_p1_X, q_j_m1_Y, q_j_0_Y, q_j_p1_Y,
                                 w_plus_i_p1, w_minus_i_p1, 
                                 w_plus_j_p1, w_minus_j_p1 )
end
