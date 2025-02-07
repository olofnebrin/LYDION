
module SpatialFactorsAndGrid

""" MODULE:  SpatialFactorsAndGrid

    PURPOSE: Contains functions for setting up a spatial grid/cells,
             and computing spatial leakage coefficients (R_{i,j}, R_{i±1/2,j}). 
"""

__precompile__()

include( "./ConstantsAndFunctions.jl" )
include( "./ExponentialScheme.jl" )
using .ConstantsAndFunctions           # Needed constants & functions
using .ExponentialScheme               # To handle diffusion+advection flux
using Base.Threads                     # For CPU acceleration

export UpdateSpatialFactors!, GetSpatialFactors


# Set a very small minimum optical depth (over scale of whole simulated volume)
# to prevent division by zero in opacity-free regions:
const τmin = 1e-8 


""" This function **updates** the spatial leakage factors (R_{i,j,k}, R_{i±1,j,k},
    Z_{i,j,k}, Z_{i,j±1,k}), as well as D_{i,j,k}, using needed information from 
    the spatial grid/cells. """
function UpdateSpatialFactors!( R::AbstractVector{Float64},   # R-coordinate vector  
                               ΔR::AbstractVector{Float64},   # R-cell width ≡ R_{i+1/2} - R_{i-1/2}
                                Z::AbstractVector{Float64},   # Z-coordinate vector 
                               ΔZ::AbstractVector{Float64},   # Z-cell width ≡ Z_{j+1/2} - Z_{j-1/2}
                               ΔV::AbstractMatrix{Float64},   # Cell volumes
                                α::AbstractMatrix{Float64},   # Lyα opacity at line center ≡ n_HI * σ_0
                               αc::AbstractMatrix{Float64},   # Continuum opacity 
                               pd::AbstractMatrix{Float64},   # Lyα destruction probability for each cell 
                                u::Tuple{AbstractArray{Float64, 2}, AbstractArray{Float64, 2}},  # Gas velocity u = (u_R, u_Z)
                             Hijk::AbstractArray{Float64, 3}, # Voigt profile for each cell
                             Dijk::AbstractArray{Float64, 3}, # Effective destruction term for each cell    (IS UDPATED)
                           Rip_jk::AbstractArray{Float64, 3}, # R-direction leakage coefficient R_{i+1,j,k} (IS UPDATED)
                             Rijk::AbstractArray{Float64, 3}, # R-direction leakage coefficient R_{i,j,k}   (IS UPDATED)
                           Rim_jk::AbstractArray{Float64, 3}, # R-direction leakage coefficient R_{i-1,j,k} (IS UPDATED)
                           Zijp_k::AbstractArray{Float64, 3}, # Z-direction leakage coefficient Z_{i,j+1,k} (IS UPDATED)
                             Zijk::AbstractArray{Float64, 3}, # Z-direction leakage coefficient Z_{i,j,k}   (IS UPDATED)
                           Zijm_k::AbstractArray{Float64, 3}  # Z-direction leakage coefficient Z_{i,j-1,k} (IS UPDATED) 
                           )

    # Get grid size:
    NR = length(R); NZ = length(Z); Nν = size( Hijk, 3)
    # Get R and Z velocity components:
    u_R, u_Z = u
    # Set maximum value of spatial diffusion coefficients 
    # (to prevent division by zero in vacua):
    Kmax = max( R[NR], Z[NZ] ) / (3 * τmin)

    @inbounds begin

        ΔR[1]  = Rmid[1]
        ΔR[NR] = Rmid[NR] - Rmid[NR-1]
        for i in 2:NR-1
            ΔR[i] = ( R[i+1] - R[i-1] ) / 2
        end

        # COMPUTE ΔZ 
        for j in 2:NZ-1
            ΔZ[j] = ( Z[i+1] - Z[i-1] ) / 2
        end

        @threads for i = 2:NR-1
            for j = 2:NZ-1
                # CELL VOLUME:
                ΔV[i,j] = π * (Rmid[i]^2 - Rmid[i-1]^2) * (Zmid[j] - Zmid[j-1])

                @simd for k in 2:Nν-1
                    #############################################
                    #########                           #########
                    #########     R-LEAKAGE FACTORS     #########
                    #########                           #########
                    ############################################# 

                    # Needed f(R_{i±1/2}), C (advection) 
                    # and K (diffusion):
                    f_iphalf = 2 * Rmid[i] / (Rmid[i]^2 - Rmid[i-1]^2)
                    f_imhalf = 2 * Rmid[i-1] / (Rmid[i]^2 - Rmid[i-1]^2)
                    C_i      = u_R[i,j] / c
                    C_iplus  = u_R[i+1,j] / c 
                    C_iminus = u_R[i-1,j] / c
                    K_i      = min( 1 / ( 3 * ( α[i,j]*Hijk[i,j,k] + αc[i,j] ) ), Kmax )
                    K_iplus  = min( 1 / ( 3 * ( α[i+1,j]*Hijk[i+1,j,k] + αc[i+1,j] ) ), Kmax )
                    K_iminus = min( 1 / ( 3 * ( α[i-1,j]*Hijk[i-1,j,k] + αc[i-1,j] ) ), Kmax )

                    # R-leakage coefficients, using the exponential scheme:
                    Rip_jk[i,j,k] = A_mplus(f_iphalf, C_i, ΔR[i], K_i, C_iplus, ΔR[i+1], K_iplus)
                    Rijk[i,j,k]   = A_m(f_iphalf, f_imhalf, C_i, ΔR[i], K_i, 
                                        C_iplus, ΔR[i+1], K_iplus, 
                                        C_iminus, ΔR[i-1], K_iminus)
                    Rim_jk[i,j,k] = A_mminus(f_imhalf, C_i, ΔR[i], K_i, C_iminus, ΔR[i-1], K_iminus)

                    #############################################
                    #########                           #########
                    #########     Z-LEAKAGE FACTORS     #########
                    #########                           #########
                    ############################################# 

                    # Needed f(Z_{i±1/2}), C (advection) 
                    # and K (diffusion):
                    f_jphalf = 1 / (Zmid[j] - Zmid[j-1])
                    f_jmhalf = f_jphalf
                    C_j      = u_Z[i,j] / c
                    C_jplus  = u_Z[i,j+1] / c 
                    C_jminus = u_Z[i,j-1] / c
                    K_j      = min( 1 / ( 3 * ( α[i,j]*Hijk[i,j,k] + αc[i,j] ) ), Kmax )
                    K_jplus  = min( 1 / ( 3 * ( α[i,j+1]*Hijk[i,j+1,k] + αc[i,j+1] ) ), Kmax )
                    K_jminus = min( 1 / ( 3 * ( α[i,j-1]*Hijk[i,j-1,k] + αc[i,j-1] ) ), Kmax )

                    # Z-leakage coefficients, using the exponential scheme:
                    Zijp_k[i,j,k] = A_mplus(f_jphalf, C_j, ΔZ[j], K_j, C_jplus, ΔZ[j+1], K_jplus)
                    Zijk[i,j,k]   = A_m(f_jphalf, f_jmhalf, C_j, ΔZ[j], K_j, 
                                        C_jplus, ΔZ[j+1], K_jplus, 
                                        C_jminus, ΔZ[j-1], K_jminus)
                    Zijm_k[i,j,k] = A_mminus(f_jmhalf, C_j, ΔZ[j], K_j, C_jminus, ΔZ[j-1], K_jminus)

                end # End of k-loop

            end # End of j-loop
        end # End of i-loop

        ############ CLOUD CENTER (i = 1) ###############

        @threads for j = 1:NZ
            @simd for k = 2:Nν-1

                # R-leakage coefficients, using the exponential scheme:
                Rip_jk[1,j,k] = A_mplus(f_iphalf, C_i, ΔR[1], K_i, C_iplus, ΔR[2], K_iplus)
                Rijk[1,j,k]   = A_m(f_iphalf, f_imhalf, C_i, ΔR[1], K_i, 
                                    C_iplus, ΔR[2], K_iplus, 
                                    C_iminus, ΔR[i-1], K_iminus)
                Rim_jk[i,j,k] = A_mminus(f_imhalf, C_i, ΔR[i], K_i, C_iminus, ΔR[i-1], K_iminus)

                # Z-leakage coefficients, using the exponential scheme:
                Zijp_k[1,j,k] = A_mplus(f_jphalf, C_j, ΔZ[j], K_j, C_jplus, ΔZ[j+1], K_jplus)
                Zijk[1,j,k]   = A_m(f_jphalf, f_jmhalf, C_j, ΔZ[j], K_j, 
                                    C_jplus, ΔZ[j+1], K_jplus, 
                                    C_jminus, ΔZ[j-1], K_jminus)
                Zijm_k[1,j,k] = A_mminus(f_jmhalf, C_j, ΔZ[j], K_j, C_jminus, ΔZ[j-1], K_jminus)
            end
        end




        ##################################################
        #########                                #########
        #########        DESTRUCTION TERM        #########
        #########                                #########
        ################################################## 
        # Compute D_ijk ≡ α_c*(1-ω) + p_d*α*H_ijk  
        # for every cell:
        @threads for i = 1:NR
            for j = 1:NZ
                @simd for k = 1:Nν
                    Dijk[i,j,k] = αc[i,j] + pd[i,j]*α[i,j]*Hijk[i,j,k]
                end
            end
        end 

    end # End of inbounds
end





######### OLD BELOW

function UpdateSpatialFactors!( Δr::AbstractVector{Float64}, ΔV::AbstractVector{Float64},
                               βNr::AbstractVector{Float64}, Ri_pj::AbstractMatrix{Float64}, 
                             Ri_mj::AbstractMatrix{Float64}, ∇dotu::AbstractVector{Float64},
                               Dij::AbstractMatrix{Float64}, r::AbstractVector{Float64}, 
                               Hij::AbstractMatrix{Float64}, α::AbstractVector{Float64}, 
                                αc::AbstractVector{Float64}, pd::AbstractVector{Float64} )

    Nr = length(r); Nν = size( Hij, 2 )
    @inbounds begin
        # Calculate midpoints (interface values) of r, i.e. r_{i±1/2}:
        rmid = [ (r[i] + r[i+1]) / 2 for i in 1:Nr-1 ]
        push!(rmid, r[Nr] + (r[Nr] - r[Nr-1]) / 2)      # Add r_{Nr+1/2}

        # Get Δr_i = r_{i+1/2} - r_{i-1/2}:
        Δr[1]  = rmid[1]
        Δr[Nr] = rmid[Nr] - rmid[Nr-1]
        for i in 2:Nr-1
             Δr[i] = (r[i+1] - r[i-1])/2
        end

        @threads for j in 1:Nν
            βNr[j]   = 1/( 1 + sqrt(3) * ( α[Nr] * Hij[Nr,j] + αc[Nr])* Δr[Nr] / 2 ) - 1
        end

        # CELL VOLUMES ΔV AND RADIAL FLUX FACTORS R_{i±1/2,j}:
        ΔV[1]  = (4*π/3) * rmid[1]^3                     # Volume of innermost shell
        ΔV[Nr] = (4*π/3) * ( rmid[Nr]^3 - rmid[Nr-1]^3 ) # Volume of outermost shell

        @threads for i = 2:Nr-1
            # Volume element ΔV_i = (4π/3)*(r_{i+1/2}^3 - r_{i-1/2}^3) :
            ΔV[i] = (4*π/3)*( rmid[i]^3 - rmid[i-1]^3 )
            @simd for j = 2:Nν-1
                # RADIAL FLUX FACTORS:
                # R_{i+1/2,j}
                Ri_pj[i,j] =  8 * π * rmid[i]^2 / (
                3 * ΔV[i] * ( (α[i]*Hij[i,j] + αc[i])*Δr[i] + (α[i+1]*Hij[i+1,j] + αc[i+1])*Δr[i+1]) ) 

                # R_{i-1/2,j}
                Ri_mj[i,j] =  8 * π * rmid[i-1]^2 / (
                3 * ΔV[i] * ( (α[i]*Hij[i,j] + αc[i])*Δr[i] + (α[i-1]*Hij[i-1,j] + αc[i-1])*Δr[i-1]) )  
            end
        end

        # Cloud center values:
        Ri_pj[1,2:Nν-1]  = 8 * π * rmid[1]^2 ./ (
        3 .* ΔV[1] .* ( (α[1].*Hij[1,2:Nν-1] .+ αc[1]).*Δr[1] .+ (α[2].*Hij[2,2:Nν-1] .+ αc[2]).*Δr[2]) )
        Ri_mj[1,2:Nν-1]  = Ri_pj[1,2:Nν-1]

        # Cloud edge values:
        Ri_pj[Nr,2:Nν-1] = 8 * π * rmid[Nr]^2 ./ ( 3 .* ΔV[Nr] .* (α[Nr] .* Hij[Nr,2:Nν-1] .+ αc[Nr]) .* Δr[Nr] ) 
        Ri_mj[Nr,2:Nν-1] = 8 * π * rmid[Nr-1]^2 ./ (
        3 .* ΔV[Nr] .* ( (α[Nr].*Hij[Nr,2:Nν-1] .+ αc[Nr]).*Δr[Nr] .+ (α[Nr-1].*Hij[Nr-1,2:Nν-1] .+ αc[Nr-1]).*Δr[Nr-1]))

        # Calculate the effective destruction factor
        # D_ij ≡ [ αc*(1-ω) + p_d*α*H_ij + (∇⋅u)/3c ] for every cell:
        @threads for i in 1:Nr
            @simd for j = 1:Nν
                Dij[i,j] = αc[i] + pd[i]*α[i]*Hij[i,j] + ∇dotu[i]/(3*c)
            end
        end
        
        
    end # End of inbounds
end


function GetSpatialFactors( r::AbstractVector{Float64}, ν::AbstractVector{Float64},
                          nHI::AbstractVector{Float64}, T::AbstractVector{Float64}, 
                           αc::AbstractVector{Float64}, pd::AbstractVector{Float64}, 
                          Hij::AbstractMatrix{Float64}, ∇dotu::AbstractVector{Float64} )

    Nr = length( r ); Nν = length( ν )

    # Get α = nHI*σ0:
    α     = nHI .* σ0.(T)

    # Get other factors:
    ΔV    = similar( r )
    Δr    = similar( r )
    βNr   = zeros( Nν )
    Ri_pj = zeros( Nr, Nν )
    Ri_mj = similar( Ri_pj )
    Dij   = similar( Ri_pj )

    # Get their values:
    UpdateSpatialFactors!( Δr, ΔV, βNr, Ri_pj, Ri_mj, ∇dotu, Dij, r, Hij, α, αc, pd )

    return ΔV, βNr, Ri_pj, Ri_mj, Dij, α
end


end # End of module