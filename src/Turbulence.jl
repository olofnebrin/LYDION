""" FILE:        Turbulence.jl

    PURPOSE:     Contains the code for generating (2D) turbulent initial conditions, as well 
                 e.g. computing the cloud turublent Mach number. 


"""

using .ConstantsAndFunctions  # For constants  

export SetupTurbulence!


const Nmodes::Int = 512

""" This function perturbs the gas velocity u = (u_r, u_Z),
    such that a given cloud-scale turbulent RMS velocity u_RMSturb and 
    velocity power spectrum is generated within a radius Rcl. """
function SetupTurbulence!( u_RMSturb::Float64,
                                 Rcl::Float64,
                                  Zc::Float64,
                                 u_r::AbstractArray{Float64, 2}, 
                                 u_Z::AbstractArray{Float64, 2},
                                   r::AbstractArray{Float64, 1}, 
                                   Z::AbstractArray{Float64, 1}, 
                                rCoM::AbstractArray{Float64, 1}, 
                                ZCoM::AbstractArray{Float64, 1}, 
                                  ΔV::AbstractArray{Float64, 2},
                            BC_Z_bot::Symbol  = :Reflect,
                                   p::Float64 = 2.0
                           )

    kmin        = Two_π / Rcl             # Largest scale (cloud-scale driving)
    Δr_min      = minimum( diff(r) )
    ΔZ_min      = minimum( diff(Z) )
    Δmin        = min( Δr_min, ΔZ_min )
    kmax        = π / (4 * Δmin)          # Smallest scale (set to ~ 8 cells)


    N_r  = length( rCoM ) 
    N_Z  = length( ZCoM )
    dN_r = 1:N_r 
    dN_Z = 1:N_Z 

    # Initialize χ:
    χ    = zeros( size(u_r) )

    # If we assume midplane symmetry (symmetry around Z = Zc), we construct a 
    # parity conistent χ by evaluating it at the mirrored positions:
    χ_m  = zeros( size(u_r) )


    for m in 1:Nmodes
        # Generate random k-values (i.e. random scales for the velocity fluctuations, within the 
        # stated range), with a log-uniform distribution:
        k    = exp( rand() * ( log(kmax) - log(kmin) ) + log(kmin) )

        # The corresponding sampling pdf is p_MC ∝ k⁻¹:
        p_MC = 1 / k 

        # Random orientation in (r,Z): 
        θ    = Two_π * rand()
        kr   = k * cos(θ)
        kZ   = k * sin(θ)

        # Random phase:
        ϕ  = Two_π * rand()

        # Target power spectrum P(k) ∝ k⁻ᵖ:
        P  = k^(-p)

        # Amplitude of mode (not normalized yet) that would ensure convergence to 
        # target power spectrum P:
        δu_k_amplitude  = sqrt( P / p_MC )

        # We can relate to amplitude A of χ as follows. Using δu_r = - r ∂χ/∂Z and 
        # δu_Z = 2χ  +  r ∂χ/∂r, we get, δu_r ~ r × k_Z × χ_k and δu_Z = 2χ  +  r × k_r × χ_k,
        # so that for k_r larger than r (typical) we have δu_k ~ r × k × χ_k. Thus, up to a constant:
        A = δu_k_amplitude / k


        # Add mode to χ:
        @inbounds for j in dN_Z
            for i in dN_r 
                # Get mode contribution to χ:
                χ[i,j] += A * sin( kr * rCoM[i]  +  kZ * (ZCoM[j] - Zc)  +  ϕ )

                # Also build the mirrored field χ_m corresponding to Z' = 2 Zc - Z:
                if BC_Z_bot == :Reflect
                    χ_m[i,j] += A * sin( kr * rCoM[i]  -  kZ * (ZCoM[j] - Zc)  +  ϕ )
                end

            end # End of i-loop 
        end # End of j-loop 

    end

    # Dampen the velocity fluctuations outside the cloud:
    Δ = 0.2 * Rcl  # Sets how fast the fluctuations are dampened
    @inbounds for j in dN_Z 
        for i in dN_r 
            R      = sqrt( rCoM[i]^2 + ( ZCoM[j] - Zc )^2 )
            Weight = 0.5 * ( 1 - tanh( (R - Rcl) / Δ ) )

            χ[i,j] *= Weight

            if BC_Z_bot == :Reflect
                χ_m[i,j] *= Weight
            end
        end # End of i-loop 
    end # End of j-loop 

    # STEP 2: Get the velocity fluctuations:
    δu_r   = zeros( size(u_r) )
    δu_Z   = zeros( size(u_Z) )

    # If reflecting, compute velocities both for χ and for the mirrored χ_m, then
    # project to the correct midplane parity: u_r even, u_Z odd around Z = Zc.
    δu_r_m = zeros( size(u_r) )
    δu_Z_m = zeros( size(u_Z) )

    @inbounds for j in dN_Z

        # Get ZCoM[j-1] and ZCoM[j+1], valid both at interior and edge cells:
        if j == 1
            ZCoM_m1 = Z[1] - (ZCoM[1] - Z[1])    # Mirror to get ZCoM[0]
            ZCoM_p1 = ZCoM[j+1] 
        elseif j == N_Z 
            ZCoM_m1 = ZCoM[j-1]
            ZCoM_p1 = ZCoM[N_Z] + (ZCoM[N_Z] - ZCoM[N_Z-1]) # Extrapolate to get ZCoM[N_Z+1]
        else
            ZCoM_m1 = ZCoM[j-1]
            ZCoM_p1 = ZCoM[j+1] 
        end

        for i in dN_r 

            # Get rCoM[i-1] and rCoM[i+1], and χ at those positions,
            # valid both at interior and edge cells:
            if i == 1
                rCoM_m1 = r[1] - (rCoM[1] - r[1])    # Mirror to get rCoM[0]
                rCoM_p1 = rCoM[i+1] 

                χ_im1   = χ[1,j]      # Even parity at at r = 0 (gives δu_r = 0), since it is a scalar
                χ_ip1   = χ[i+1,j]

                χm_im1  = χ_m[1,j]    # Even partiy at r = 0 (gives δu_r = 0), since it is a scalar
                χm_ip1  = χ_m[i+1,j]
            elseif i == N_r 
                rCoM_m1 = rCoM[i-1]
                rCoM_p1 = rCoM[N_r] + (rCoM[N_r] - rCoM[N_r-1]) # Extrapolate to get rCoM[N_r+1]

                χ_im1   = χ[i-1,j]    
                χ_ip1   = χ[i,j]      # Outflow/extrapolation BC at r = Rmax 

                χm_im1  = χ_m[i-1,j]    
                χm_ip1  = χ_m[i,j]    # Outflow/extrapolation BC at r = Rmax 
            else
                rCoM_m1 = rCoM[i-1]
                rCoM_p1 = rCoM[i+1] 

                χ_im1   = χ[i-1,j]
                χ_ip1   = χ[i+1,j]

                χm_im1  = χ_m[i-1,j]
                χm_ip1  = χ_m[i+1,j]
            end

            # Also get the χ at the j+1 and j-1 points:
            if j == 1
                if BC_Z_bot == :Reflect
                    χ_jm1  = - χ[i,1]      # Mirror BC at Z = 0
                    χ_jp1  = χ[i,j+1]

                    χm_jm1 = - χ_m[i,1]    # Mirror BC at Z = 0 
                    χm_jp1 = χ_m[i,j+1]
                else
                    χ_jm1  = χ[i,1]      
                    χ_jp1  = χ[i,j+1]

                    χm_jm1 = χ_m[i,1]    
                    χm_jp1 = χ_m[i,j+1]
                end
            elseif j == N_Z 
                χ_jm1  = χ[i,j-1]   
                χ_jp1  = χ[i,N_Z]    # Outflow/extrapolation BC at Z = Zmax

                χm_jm1 = χ_m[i,j-1]   
                χm_jp1 = χ_m[i,N_Z]  # Outflow/extrapolation BC at Z = Zmax
            else
                χ_jm1  = χ[i,j-1]
                χ_jp1  = χ[i,j+1]

                χm_jm1 = χ_m[i,j-1]
                χm_jp1 = χ_m[i,j+1]
            end

            # Coefficients for the second-order central difference 
            # approximation to derivatives of χ, valid on non-uniform grids:
            Δm_r  = rCoM[i] - rCoM_m1
            Δp_r  = rCoM_p1 - rCoM[i]
            am_r  = - Δp_r / (Δm_r * (Δm_r + Δp_r))
            a0_r  = ( Δp_r - Δm_r ) / (Δm_r * Δp_r)
            ap_r  = Δm_r / (Δp_r * (Δm_r + Δp_r))

            Δm_Z  = ZCoM[j] - ZCoM_m1
            Δp_Z  = ZCoM_p1 - ZCoM[j]
            am_Z  = - Δp_Z / (Δm_Z * (Δm_Z + Δp_Z))
            a0_Z  = ( Δp_Z - Δm_Z ) / (Δm_Z * Δp_Z)
            ap_Z  = Δm_Z / (Δp_Z * (Δm_Z + Δp_Z))

            # Central difference derivatives of χ:
            ∂χ∂r = ap_r * χ_ip1  +  a0_r * χ[i,j]  +  am_r * χ_im1
            ∂χ∂Z = ap_Z * χ_jp1  +  a0_Z * χ[i,j]  +  am_Z * χ_jm1

            # Velocity fluctuations (not normalized yet):
            δu_r[i,j] = - rCoM[i] * ∂χ∂Z
            δu_Z[i,j] =   2 * χ[i,j]  +  rCoM[i] * ∂χ∂r

            if BC_Z_bot == :Reflect
                # Derivatives of χ_m, stored at the mirrored positions Z' = 2 Zc - Z.
                # NOTE: χ_m[i,j] is a function of ZCoM[j], but corresponds to the physical
                # location Z' where d/dZ introduces a sign flip.

                ∂χm∂r  = ap_r * χm_ip1  +  a0_r * χ_m[i,j]  +  am_r * χm_im1
                dχm_dZ = ap_Z * χm_jp1  +  a0_Z * χ_m[i,j]  +  am_Z * χm_jm1
                ∂χm∂Z  = - dχm_dZ

                δu_r_m[i,j] = - rCoM[i] * ∂χm∂Z
                δu_Z_m[i,j] =   2 * χ_m[i,j]  +  rCoM[i] * ∂χm∂r
            end

        end # End of i-loop 
    end # End of j-loop 

    # For reflecting BCs at Z = Zc, we enforce the same for the velocity 
    # fluctuations using the mirrored field:
    if BC_Z_bot == :Reflect
        @inbounds for j in dN_Z
            for i in dN_r

                δu_r[i,j] = 0.5 * ( δu_r[i,j]  +  δu_r_m[i,j] )
                δu_Z[i,j] = 0.5 * ( δu_Z[i,j]  -  δu_Z_m[i,j] )

            end # End of i-loop 
        end # End of j-loop 
    end

    # Normalize to the Mach number:
    u_squared = 0.0
    Vcl       = 0.0
    @inbounds for j in dN_Z 
        for i in dN_r
            R   = sqrt( rCoM[i]^2 + ( ZCoM[j] - Zc )^2 ) 

            if R ≤ Rcl 
                u_squared += ΔV[i,j] * ( δu_r[i,j]^2  +  δu_Z[i,j]^2 )
                Vcl       += ΔV[i,j]
            end
        end
    end

    # RMS velocity fluctuation for the cloud, before normalization:
    u_RMS     = sqrt( u_squared / Vcl )

    # Get the scale factor to generate desired RMS velocity:
    ScaleFact = u_RMSturb / max( u_RMS, 1e-30 )

    # Get the normalized velocity fluctuations for all cells:
    δu_r    .*= ScaleFact
    δu_Z    .*= ScaleFact

    # Add velocity fluctuations to u_r and u_Z:
    u_r     .+= δu_r
    u_Z     .+= δu_Z

    return nothing 
end


