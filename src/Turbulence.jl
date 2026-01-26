
""" FILE:        Turbulence.jl

    PURPOSE:     Contains the code for generating (2D) turbulent initial conditions, as well 
                 e.g. computing the cloud turublent Mach number. 


"""

using .ConstantsAndFunctions  # For constants  

export SetupTurbulence!


const Nmodes::Int = 256

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
                                   p::Float64 = 2.0,
                                  R0::Float64 = 0.5 * Rcl
                           )

    kmin        = Two_π / Rcl             # Largest scale (cloud-scale driving)
    Δr_min      = minimum( diff(r) )
    ΔZ_min      = minimum( diff(Z) )
    Δmin        = min( Δr_min, ΔZ_min )
    kmax        = π / (4 * Δmin)          # Smallest scale (grid scale)


    N_r  = length( rCoM ) 
    N_Z  = length( ZCoM )
    dN_r = 1:N_r 
    dN_Z = 1:N_Z 

    # Initialize ψ:
    ψ    = zeros( size(u_r) )


    for m in 1:Nmodes
        # Generate random k-values (i.e. random scales for the velocity fluctuations, within the 
        # stated range), with a log-uniform distribution:
        k  = exp( rand() * ( log(kmax) - log(kmin) ) + log(kmin) )

        # Random orientation in (r,Z): 
        θ  = Two_π * rand()
        kr = k * cos(θ)
        kZ = k * sin(θ)

        # Random phases:
        ϕr = Two_π * rand()
        ϕZ = Two_π * rand()

        # Amplitude of mode (not normalized yet):
        A  = k^(-p)

        # Add mode to ψ:
        @inbounds for j in dN_Z
            for i in dN_r 

                # Radial dependence of amplitudes, goes to zero near r = 0
                # in a way consistent with the solenoidal constraints that S(r) ∝ r² near 
                # the cylindrical axis:
                S = (rCoM[i]^2) / ( 1 + (rCoM[i] / R0)^2 )

                # Get mode contribution to ψ:
                ψ[i,j] += A * S * sin(kr * rCoM[i]  +  ϕr) * sin(kZ * (ZCoM[j] - Zc)  +  ϕZ)
            end
        end

    end

    # STEP 2: Get the velocity fluctuations:
    δu_r = zeros( size(u_r) )
    δu_Z = zeros( size(u_Z) )
    @inbounds for j in 2:(N_Z-1)
        for i in 2:(N_r-1)
            # Central difference derivatives of ψ:
            #∂ψ∂r = ( ψ[i+1,j] - ψ[i-1,j] ) / ( rCoM[i+1] - rCoM[i-1] )
            #∂ψ∂Z = ( ψ[i,j+1] - ψ[i,j-1] ) / ( ZCoM[j+1] - ZCoM[j-1] )

            # Coefficients for the second-order central difference 
            # approximation to derivatives of ψ, valid on non-uniform grids:
            Δm_r  = rCoM[i]   - rCoM[i-1]
            Δp_r  = rCoM[i+1] - rCoM[i]
            am_r  = - Δp_r / (Δm_r * (Δm_r + Δp_r))
            a0_r  = ( Δp_r - Δm_r ) / (Δm_r * Δp_r)
            ap_r  = Δm_r / (Δp_r * (Δm_r + Δp_r))

            Δm_Z  = ZCoM[j]   - ZCoM[j-1]
            Δp_Z  = ZCoM[j+1] - ZCoM[j]
            am_Z  = - Δp_Z / (Δm_Z * (Δm_Z + Δp_Z))
            a0_Z  = ( Δp_Z - Δm_Z ) / (Δm_Z * Δp_Z)
            ap_Z  = Δm_Z / (Δp_Z * (Δm_Z + Δp_Z))
                
            # Central difference derivatives of ψ:
            ∂ψ∂r = ap_r * ψ[i+1,j]  +  a0_r * ψ[i,j]  +  am_r * ψ[i-1,j]
            ∂ψ∂Z = ap_Z * ψ[i,j+1]  +  a0_Z * ψ[i,j]  +  am_Z * ψ[i,j-1]

            # Velocity fluctuations (not normalized yet):
            inv_rCoM  =   1 / rCoM[i]
            δu_r[i,j] = - inv_rCoM * ∂ψ∂Z
            δu_Z[i,j] =   inv_rCoM * ∂ψ∂r
        end
    end

    @inbounds for j in dN_Z
        # Velocity at i = 1:
        δu_r[1,j] = 0.0
        δu_Z[1,j] = δu_Z[2,j]   # Mirror 
    end

    @inbounds for i in dN_r
        # Velocity at j = N_Z (extrapolate):
        δu_r[i, N_Z] = δu_r[i, N_Z-1]
        δu_Z[i, N_Z] = δu_Z[i, N_Z-1] 
    end

    @inbounds for i in dN_r
        # Velocity at j = 1:
        δu_r[i, 1] = δu_r[i, 2] # Extrapolate 

        if BC_Z_bot == :Reflect
            δu_Z[i, 1] = 0.0        # Mirror 
        else
            δu_Z[i, 1] = δu_Z[i, 2] # Extrapolate
        end
    end
    
    @inbounds for j in dN_Z
        # Velocity at i = N_r (extrapolate):
        δu_r[N_r, j] = δu_r[N_r-1, j]
        δu_Z[N_r, j] = δu_Z[N_r-1, j]
    end


    # Dampen the velocity fluctuations outside the cloud:
    Δ = 2 * Δmin 
    @inbounds for j in dN_Z 
        for i in dN_r 
            R          = sqrt( rCoM[i]^2 + ( ZCoM[j] - Zc )^2 )
            Weight     = 0.5 * ( 1 - tanh( (R - Rcl) / Δ ) )
            δu_r[i,j] *= Weight
            δu_Z[i,j] *= Weight
        end
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

