
""" FILE:           WENO3.jl

    PURPOSE:        This file contains the code underlying the WENO3 reconstruction. 

                    The code can be applied to axi-symmetric problems in cylindrical coordinates, 
                    as well as 2D Cartesian problems. For a given cell interface, WENO3 is used 
                    if the interface is not deemed to be near a shock (which can lead to instabilities/large errors).
                    Near a shock, the more diffusive 2nd-order accurate PLM is used instead. 
             
    REFERENCES:     (1) Mignone (2014) 'High-order conservative reconstruction schemes for finite 
                        volume methods in cylindrical and spherical coordinates', Journal of Computational 
                        Physics, 270, pp. 784-814

                    (2) Verma et al. (2019) 'Fourth-order accurate finite-volume CWENO scheme for astrophysical 
                        MHD problems', MNRAS, 482, pp. 416-437

                    (3) Titarev & Toro (2004) 'Finite-volume WENO schemes for three-dimensional conservation laws',
                        Journal of Computational Physics, 201, pp. 238-260
    
    """

using .ConstantsAndFunctions  # For MinMod 
using Base.Threads            # For CPU acceleration


const Shock_HLL_and_PLM::Int = 2  # Just a number to track very strong shocks where we revert to HLL + PLM reconstruction 


export Shock_HLL_and_PLM, Compute_βs, Compute_w_k, Compute_q_k, Get_QL_and_QR



""" β coefficients needed for computing the weights w that determine the interface 
    cell reconstruction. See Eq. (23) in Mignone (2014). """
function Compute_βs( n::Int, m::Int, 
            ξ_im3halfs::Float64, # ξ_{i-3/2}
              ξ_imhalf::Float64, # ξ_{i-1/2}
              ξ_iphalf::Float64, # ξ_{i+1/2}
            ξ_ip3halfs::Float64  # ξ_{i+3/2}
                     )

  # β_{i-1,n}, β_{i,n}, and β_{i+1,n}:
  β_im1_n = ((m + 1) / (n + m + 1)) * ( ξ_imhalf^(n+m+1)   - ξ_im3halfs^(n+m+1) ) / ( ξ_imhalf^(m+1)   - ξ_im3halfs^(m+1) )
  β_i_n   = ((m + 1) / (n + m + 1)) * ( ξ_iphalf^(n+m+1)   - ξ_imhalf^(n+m+1) )   / ( ξ_iphalf^(m+1)   - ξ_imhalf^(m+1) )
  β_ip1_n = ((m + 1) / (n + m + 1)) * ( ξ_ip3halfs^(n+m+1) - ξ_iphalf^(n+m+1) )   / ( ξ_ip3halfs^(m+1) - ξ_iphalf^(m+1) )

  return β_im1_n, β_i_n, β_ip1_n
end


""" This function computes the weights w_{k,-1}, w_{k,0}, and w_{k,1},
    needed to go go from ⟨U⟩ₖ → U(ξcₖ) (cell averaged → cell center value),
    for a conservative variable U. The reconstruction is obtained using:
    
    U(ξcₖ) ≃ wₖ₋₁ ⟨U⟩ₖ₋₁  +  wₖ ⟨U⟩ₖ  +  wₖ₊₁ ⟨U⟩ₖ₊₁ """
function Compute_w_k( Δξ_k_km1::Float64,  # ⟨Δξ⟩_{k,k-1}
                        Δξ_k_k::Float64,  # ⟨Δξ⟩_{k,k}
                      Δξ_k_kp1::Float64,  # ⟨Δξ⟩_{k,k+1}
                    Δξsq_k_km1::Float64,  # ⟨(Δξ)²⟩_{k,k-1}
                      Δξsq_k_k::Float64,  # ⟨(Δξ)²⟩_{k,k}
                    Δξsq_k_kp1::Float64   # ⟨(Δξ)²⟩_{k,k+1}
                     )

  # The factor D_k:
  D_k    = ( ( Δξsq_k_kp1  -  Δξsq_k_km1 ) * ( Δξ_k_k    -  Δξ_k_km1 )
           - ( Δξsq_k_k    -  Δξsq_k_km1 ) * ( Δξ_k_kp1  -  Δξ_k_km1 ) )

  # The factor w_{k,0}:
  w_k_0  = (1 / D_k) * ( ( Δξ_k_kp1 - Δξ_k_km1 ) * Δξsq_k_km1  -  ( Δξsq_k_kp1 - Δξsq_k_km1 ) * Δξ_k_km1 )

  # The factor w_{k,1}:
  w_k_p1 = (1 / D_k) * ( ( Δξsq_k_k - Δξsq_k_km1 ) * Δξ_k_km1  -  ( Δξ_k_k - Δξ_k_km1 ) * Δξsq_k_km1 )

  # The factor w_{k,-1}:
  w_k_m1 = 1 - w_k_0 - w_k_p1

  return w_k_m1, w_k_0, w_k_p1
end


""" This function computes the weights q_{k,-1}, q_{k,0}, and q_{k,1},
    needed to go go from W(ξcₖ) → ⟨W⟩ₖ (cell center → cell averaged value),
    for a primitive variable W. The reconstruction is obtained using:
    
    ⟨W⟩ₖ ≃ qₖ₋₁ W(ξcₖ₋₁)  +  qₖ W(ξcₖ)  +  qₖ₊₁ W(ξcₖ₊₁) """
function Compute_q_k( ξc_km1::Float64, # ξcₖ₋₁
                        ξc_k::Float64, # ξcₖ
                      ξc_kp1::Float64, # ξcₖ₊₁
                    Δξsq_k_k::Float64  # ⟨(Δξ)²⟩_{k,k}
                      )

  # Convenient intervals:
  h_m    = ξc_k   -  ξc_km1  # ξcₖ   - ξcₖ₋₁
  h_p    = ξc_kp1 -  ξc_k    # ξcₖ₊₁ - ξcₖ

  # The factor q_{k,-1}:
  q_k_m1 = h_p * Δξsq_k_k / ( h_p * (h_m^2) + h_m * (h_p^2) )

  # The factor q_{k,0}:
  q_k_0  = 1  -  ( h_m + h_p ) * Δξsq_k_k / ( h_p * (h_m^2) + h_m * (h_p^2) )

  # The factor q_{k,1}:
  q_k_p1 = h_m * Δξsq_k_k / ( h_p * (h_m^2) + h_m * (h_p^2) )

  return q_k_m1, q_k_0, q_k_p1
end


""" MinMod slope limiter. """
@inline function MinMod( A::Float64,
                         B::Float64 
                        )
    0.5 * (sign(A) + sign(B)) * min( abs(A), abs(B) )
end


const C_ref::Float64      = 20.0
const Wref_floor::Float64 = 1e-40

""" Compute left interface value W_L at i-1/2, and 
    right interface value W_R, also at i-1/2. """
function Get_QL_and_QR( W_im2::Float64,      # We need W[i-2] and W[i-1] for the left-side estimate of W_{i-1/2}
                        W_im1::Float64, 
                          W_i::Float64,
                        W_ip1::Float64,      # We need W[i+1] for the right-side estimate of W_{i-1/2}
                        X_im1::Float64,
                          X_i::Float64,
                        X_ip1::Float64,
                     XCoM_im2::Float64, 
                     XCoM_im1::Float64, 
                       XCoM_i::Float64, 
                     XCoM_ip1::Float64,
                      Shock_i::Int,          # Shock indicator for cell i     (equal to 0, 1 or 2)
                    Shock_im1::Int,          # Shock indicator for cell i - 1 (equal to 0, 1 or 2)
                        Ngrid::Int, 
                 w_plus_im1_1::Float64,      # w⁺_{i-1, 1}
                  w_minus_i_1::Float64,      # w⁻_{i, 1}
             EnsurePositivity::Bool = false,
       MaxReconstructionOrder::Int8 = Int8(3)
                         )

    # If at least one of the cells around the interface i - 1/2 
    # has been flagged as containing a shock, then we revert from 
    # WENO3 to a piecewise linear approximation (see e.g. ref. 2):
    if max( Shock_i, Shock_im1 ) == Shock_HLL_and_PLM || MaxReconstructionOrder < Int8(3)
        # Slopes in cell i-1 (for W_L):
        sL_im1 = (W_im1 - W_im2) / (XCoM_im1 - XCoM_im2)
        sR_im1 = (W_i   - W_im1) / (XCoM_i   - XCoM_im1)
        s_im1  = MinMod( sL_im1, sR_im1 )

        # Slopes in cell i (for W_R):
        sL_i   = (W_i   - W_im1) / (XCoM_i   - XCoM_im1)
        sR_i   = (W_ip1 - W_i  ) / (XCoM_ip1 - XCoM_i)
        s_i    = MinMod( sL_i, sR_i )

        # Get left & right states: 
        W_L    = W_im1  +  s_im1 * ( X_i - XCoM_im1 )
        W_R    = W_i    +  s_i   * ( X_i - XCoM_i )

        # Measure change between interface and cell centers:
        ΔW_L   = abs( W_L - W_im1 )
        ΔW_R   = abs( W_R - W_i )

        # Save chosen reconstruction order:
        ReconstructionOrder = Int8(2)

        if (EnsurePositivity && ( W_L ≤ 0.0 || W_R ≤ 0.0 || ΔW_L > 0.9 * W_im1 || ΔW_R > 0.9 * W_i )) || MaxReconstructionOrder == Int8(1)
          # If we want positive primitives, yet the PLM reconstruction 
          # *still* produces negative values or too large changes from cell-avearges,
          # fall back to a first-order method (i.e. no reconstruction):
          W_L = W_im1 
          W_R = W_i 

          # Save chosen reconstruction order:
          ReconstructionOrder = Int8(1)
        end

        return W_L, W_R, ReconstructionOrder
    end

    # If the cell is *not* flagged as being near a shock, we proceed with 
    # WENO3 reconstruction: 

    d_minus_i_0  = w_minus_i_1  * ( XCoM_ip1 - XCoM_i ) / ( X_i - XCoM_i )
    d_minus_i_1  = 1 - d_minus_i_0

    d_plus_im1_0 = w_plus_im1_1 * ( XCoM_i - XCoM_im1 ) / ( X_i - XCoM_im1 )
    d_plus_im1_1 = 1 - d_plus_im1_0 

    Wref_im1     = max( (C_ref / Ngrid) * max( abs(W_im2), abs(W_im1), abs(W_i) ), (C_ref / Ngrid) * Wref_floor )
    Wref_i       = max( (C_ref / Ngrid) * max( abs(W_im1), abs(W_i), abs(W_ip1) ), (C_ref / Ngrid) * Wref_floor )

    # Linear forward & backward estimates of W_L at i-1/2 
    # (obtained from Mignone's expressions, shifting by i → i-1):
    ΔX_L         = X_i - X_im1
    ΔW_LF_im1    = ΔX_L * ( W_i   - W_im1 ) / ( XCoM_i - XCoM_im1 )
    ΔW_LB_im1    = ΔX_L * ( W_im1 - W_im2 ) / ( XCoM_im1 - XCoM_im2 )

    W_LF         = W_im1  +  ΔW_LF_im1 * ( X_i  -  XCoM_im1 ) / ΔX_L
    W_LB         = W_im1  +  ΔW_LB_im1 * ( X_i  -  XCoM_im1 ) / ΔX_L

    # Linear forward & backward estimates of W_R at i-1/2:
    ΔX_R         = X_ip1 - X_i 
    ΔW_RF_i      = ΔX_R * ( W_ip1 - W_i )   / ( XCoM_ip1 - XCoM_i )
    ΔW_RB_i      = ΔX_R * ( W_i   - W_im1 ) / ( XCoM_i - XCoM_im1 )

    W_RF         = W_i  +  ΔW_RF_i * ( X_i - XCoM_i ) / ΔX_R
    W_RB         = W_i  +  ΔW_RB_i * ( X_i - XCoM_i ) / ΔX_R

    # Compute α factors for left side: 
    αplus_im1_0  = d_plus_im1_0 * ( 1  +  ((ΔW_LF_im1 - ΔW_LB_im1)^2) / (ΔW_LF_im1^2  +  Wref_im1^2) )
    αplus_im1_1  = d_plus_im1_1 * ( 1  +  ((ΔW_LF_im1 - ΔW_LB_im1)^2) / (ΔW_LB_im1^2  +  Wref_im1^2) )

    # Compute α factors for right side:
    αminus_i_0   = d_minus_i_0 * ( 1  +  ((ΔW_RF_i - ΔW_RB_i)^2) / (ΔW_RF_i^2  +  Wref_i^2))
    αminus_i_1   = d_minus_i_1 * ( 1  +  ((ΔW_RF_i - ΔW_RB_i)^2) / (ΔW_RB_i^2  +  Wref_i^2))

    # Weights for W_L:
    ωplus_im1_0  = αplus_im1_0  / ( αplus_im1_0 + αplus_im1_1 ) 
    ωplus_im1_1  = αplus_im1_1  / ( αplus_im1_0 + αplus_im1_1 )

    # Weights for W_R:
    ωminus_i_0   = αminus_i_0 / ( αminus_i_0 + αminus_i_1 )
    ωminus_i_1   = αminus_i_1 / ( αminus_i_0 + αminus_i_1 )

    # LEFT SIDE ESTIMATE OF W_{i-1/2} (i.e. W_L_X[i,j] in the case of X-direction):
    
    W_L = ωplus_im1_0 * W_LF   +  ωplus_im1_1 * W_LB 

    # RIGHT SIDE ESTIMATE OF W_{i-1/2} (i.e. W_R_X[i,j] in the case of X-direction):

    W_R = ωminus_i_0 * W_RF  +  ωminus_i_1 * W_RB 

    # Save chosen reconstruction order:
    ReconstructionOrder = Int8(3)

    # FALL BACK TO 2nd-ORDER (OR 1st-ORDER) METHOD FOR NEGATIVE INTERFACE STATES 
    # (relevant for ρ, P), OR TOO LARGE DEVIATIONS FROM CELL AVERAGES (see refs. 2 & 3):

    ΔW_L = abs( W_L - W_im1 )
    ΔW_R = abs( W_R - W_i )

    if EnsurePositivity && ( W_L ≤ 0.0 || W_R ≤ 0.0 || ΔW_L > 0.9 * W_im1 || ΔW_R > 0.9 * W_i )
        # Slopes in cell i-1 (for W_L):
        sL_im1 = (W_im1 - W_im2) / (XCoM_im1 - XCoM_im2)
        sR_im1 = (W_i   - W_im1) / (XCoM_i   - XCoM_im1)
        s_im1  = MinMod( sL_im1, sR_im1 )

        # Slopes in cell i (for W_R):
        sL_i   = (W_i   - W_im1) / (XCoM_i   - XCoM_im1)
        sR_i   = (W_ip1 - W_i  ) / (XCoM_ip1 - XCoM_i)
        s_i    = MinMod( sL_i, sR_i )

        # Get left & right states: 
        W_L    = W_im1  +  s_im1 * ( X_i - XCoM_im1 )
        W_R    = W_i    +  s_i   * ( X_i - XCoM_i )

        # Measure change between interface and cell centers:
        ΔW_L = abs( W_L - W_im1 )
        ΔW_R = abs( W_R - W_i )

        # Save chosen reconstruction order:
        ReconstructionOrder = Int8(2)

        # Check again if we have positive interfaces & values not too far off from 
        # cell-averaged ones. If not, revert to a 1st-order method 
        # (no interface reconstruction):
        if W_L ≤ 0.0 || W_R ≤ 0.0 || ΔW_L > 0.9 * W_im1 || ΔW_R > 0.9 * W_i
          W_L = W_im1 
          W_R = W_i
          
          # Save chosen reconstruction order:
          ReconstructionOrder = Int8(1)
        end 
    end

    return W_L, W_R, ReconstructionOrder
end
