
""" MODULE:         AdvectScalars

    PURPOSE:        This module contains the basic time integration 
                    code that handles the advection of scalars, e.g. 
                    abundance fractions like xHI = nHI / nH 
                    (treated in AdvectGasFractions.jl using this module).
    
"""


using .ConstantsAndFunctions
using  Base.Threads              # For CPU acceleration


export AdvectScalar!


function AdvectScalar!( Δt::Float64,
                         x::AbstractArray{Float64, 2},
                     x_old::AbstractArray{Float64, 2},
                         ρ::AbstractArray{Float64, 2}, 
                     F_ρ_X::AbstractArray{Float64, 2}, 
                     F_ρ_Y::AbstractArray{Float64, 2}, 
                    inv_ΔV::AbstractArray{Float64, 2}, 
                       A_X::AbstractArray{Float64, 2}, 
                       A_Y::AbstractArray{Float64, 2}, 
            ClampAboveZero::Bool  = true
                      )

    N_X, N_Y = size( ρ ) # Grid size 
    dN_X     = 1:N_X
    dN_Y     = 1:N_Y 
    Δt_half  = Δt / 2

    # Update x_old:
    x_old .= x 

    ######### 1) HALF-STEP IN X-DIRECTION #########
    
    @threads for j in dN_Y
        @simd for i in dN_X
            @inbounds begin
                # Clamp indices that enter x, which implicitly assumes 
                # zero gradient extrapolation BC's:
                ip1 = min( i + 1, N_X )
                im1 = max( i - 1, 1 )           

                # Reconstruct x on interfaces, using a simple first-order method: 
                if F_ρ_X[i,j] ≥ 0.0
                    xL = x_old[im1,j]  # Flow left → right
                else
                    xL = x_old[i,j]    # Flow right → left
                end

                if F_ρ_X[i+1,j] ≥ 0.0
                    xR = x_old[i,j]    # Flow left → right
                else
                    xR = x_old[ip1,j]  # Flow right → left
                end

                # Get left (i-1/2) & right (i+1/2) interface fluxes:
                F_L_X   = F_ρ_X[i,j]   * xL
                F_R_X   = F_ρ_X[i+1,j] * xR

                # Get interface areas:
                A_R     = A_X[i+1,j]
                A_L     = A_X[i,j]

                # Update x over a half-step Δt / 2:
                x[i,j]  = x_old[i,j] - (Δt_half * inv_ΔV[i,j] / ρ[i,j]) * ( A_R * F_R_X  -  A_L * F_L_X )

                if ClampAboveZero
                    x[i,j] = max( x[i,j], 0.0 )
                end

            end # End of inbounds 
        end # End of i-loop
    end # End of j-loop

    # Update x_old:
    x_old .= x 
    
    ######### 2) FULL STEP IN Y-DIRECTION ###############

    @threads for j in dN_Y
        @simd for i in dN_X
            @inbounds begin
                # Clamp indices that enter x, which implicitly assumes 
                # zero gradient extrapolation BC's:
                jp1 = min( j + 1, N_Y )
                jm1 = max( j - 1, 1 ) 

                # Reconstruct x on interfaces, using a simple first-order method: 
                if F_ρ_Y[i,j] ≥ 0.0
                    xL = x_old[i,jm1]  # Flow left → right
                else
                    xL = x_old[i,j]    # Flow right → left
                end

                if F_ρ_Y[i,j+1] ≥ 0.0
                    xR = x_old[i,j]    # Flow left → right
                else
                    xR = x_old[i,jp1]  # Flow right → left
                end

                # Get left (j-1/2) & right (j+1/2) interface fluxes:
                F_L_Y   = F_ρ_Y[i,j]   * xL
                F_R_Y   = F_ρ_Y[i,j+1] * xR

                # Get interface areas:
                A_R     = A_Y[i,j+1]
                A_L     = A_Y[i,j]

                # Update x over a full step Δt:
                x[i,j]  = x_old[i,j] - (Δt * inv_ΔV[i,j] / ρ[i,j]) * ( A_R * F_R_Y  -  A_L * F_L_Y )

                if ClampAboveZero
                    x[i,j] = max( x[i,j], 0.0 )
                end

            end # End of inbounds 
        end # End of i-loop 
    end # End of j-loop 

    # Update x_old:
    x_old .= x

    ######### 3) FINAL HALF-STEP IN X-DIRECTION #########
    
    @threads for j in dN_Y
        @simd for i in dN_X
            @inbounds begin
                # Clamp indices that enter x, which implicitly assumes 
                # zero gradient extrapolation BC's:
                ip1 = min( i + 1, N_X )
                im1 = max( i - 1, 1 )           

                # Reconstruct x on interfaces, using a simple first-order method: 
                if F_ρ_X[i,j] ≥ 0.0
                    xL = x_old[im1,j]  # Flow left → right
                else
                    xL = x_old[i,j]    # Flow right → left
                end

                if F_ρ_X[i+1,j] ≥ 0.0
                    xR = x_old[i,j]    # Flow left → right
                else
                    xR = x_old[ip1,j]  # Flow right → left
                end

                # Get left (i-1/2) & right (i+1/2) interface fluxes:
                F_L_X   = F_ρ_X[i,j]   * xL
                F_R_X   = F_ρ_X[i+1,j] * xR

                # Get interface areas:
                A_R     = A_X[i+1,j]
                A_L     = A_X[i,j]

                # Update x over a half-step Δt / 2:
                x[i,j]  = x_old[i,j] - (Δt_half * inv_ΔV[i,j] / ρ[i,j]) * ( A_R * F_R_X  -  A_L * F_L_X )

                if ClampAboveZero
                    x[i,j] = max( x[i,j], 0.0 )
                end

            end # End of inbounds 
        end # End of i-loop
    end # End of j-loop 

    # Update x_old:
    x_old .= x

    return nothing
end

