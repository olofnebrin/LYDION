
module FreqFactorsAndGrid

""" MODULE:  FreqFactorsAndGrid

    PURPOSE: Contains functions for setting up a frequency grid,
             and computing frequency leakage coefficients (F_{i,j}, F_{i,j±1/2}). 
"""

__precompile__()

include( "./ConstantsAndFunctions.jl" )
include( "./ExponentialScheme.jl" )
using .ConstantsAndFunctions  # Needed constants & functions
using .ExponentialScheme      # The exponential scheme to handle diffusion+advection flux
using Base.Threads            # For CPU acceleration

export GetFrequencies, GetFreqFactors, UpdateFreqFactors!

const Pmax = 40.0  # Maximum Peclet number to consider in exponentials


""" Generates suitable frequency cells for a given set of
    cloud properties. The cells are created so that the Lyα line 
    center is always well-resolved.
    
    **ARGUMENTS:**
    
    - r        :   Array of radial cell points (cm)
    - nHI      :   Array of HI number densities at radial cell points (cm^-3)
    - T        :   Array of gas temperatures at radial cell points (K)
    - Ncoarse  :   Number of coarser cells in the wing reagions. 
    - ηfreq    :   Maximum/minimum frequency to consider, |ν_max/min| ≈ ηfreq x (aτ_cl)^(1/3)

    **RETURNS:** 

    - ν        :   Array of frequency cells (Hz)
    - Δν       :   Array of difference in frequency between bins (Hz)
    
    """
function GetFrequencies(r::AbstractVector{Float64}, nHI::AbstractVector{Float64}, 
                        T::AbstractVector{Float64}, Ncoarse = 500, ηfreq = 20.0)
    # Compute τ_cl:
    τcl = τ(r, nHI, T)

    Δν_D = (b.(T) ./ c) .* νLyα

    # Get maximum and minimum temperature, and corresponding a(T)
    # for the latter:
    Tmax = maximum( T )
    Tmin = minimum( T ); aTmin = a(Tmin)

    # Get the minimum and maximum Doppler width:
    Δν_Dmin = (b(Tmin)/c)*νLyα
    Δν_Dmax = (b(Tmax)/c)*νLyα

    # Generate frequency grid that resolves down to at least Δν_Dmin around
    # line center:
    ν_finegrid_min   = νLyα - 10.0*Δν_Dmax                     # Maximum frequency with fine gridding
    ν_finegrid_max   = νLyα + 10.0*Δν_Dmax                     # Minimum frequency with fine gridding
    Δν_fine          = 0.2*Δν_Dmin                             # Spacing of fine gridding
    ν_finegrid       = ν_finegrid_min:Δν_fine:ν_finegrid_max   # Fine grid near line center

    # Generate coarser gridding outside:
    Δν_max           = ηfreq * Δν_Dmin * ( aTmin * τcl )^(1/3) # Maximum considered frequency width
    ν_coarsegrid_min = ν_finegrid_min - Δν_max
    ν_coarsegrid_max = ν_finegrid_max + Δν_max

    ν_coarsegrid_low = LinRange( ν_coarsegrid_min, ν_finegrid_min - Δν_fine, div( Ncoarse, 2 ) )
    ν_coarsegrid_up  = LinRange( ν_finegrid_max + Δν_fine, ν_coarsegrid_max, div( Ncoarse, 2 ) )

    # Put together to get the full frequency grid:
    ν = collect( vcat( ν_coarsegrid_low, ν_finegrid, ν_coarsegrid_up ) )

    # Get Δν
    Nν = length( ν )
    Δν = zeros( Nν )
    @inbounds @simd for j in 2:Nν-1
        Δν[j] = (ν[j+1] - ν[j-1])/2
    end
    Δν[1] = Δν[2]; Δν[Nν] = Δν[Nν-1]

    # Get Voigt profile for each cell:
    Nr  = length(r)
    Hij = zeros( Nr, Nν )
    @inbounds for i in 1:Nr
        @simd for j in 1:Nν
            Δx        = Δν[j]/Δν_D[i]
            Hij[i,j]  = Htophat( (ν[j] - νLyα)/Δν_D[i],  a(T[i]),  Δx )
        end
    end

    return ν, Δν, Hij
end


function UpdateFreqFactors!( ν::AbstractVector{Float64}, Fi_jp::AbstractMatrix{Float64}, 
                          Fi_j::AbstractMatrix{Float64}, Fi_jm::AbstractMatrix{Float64}, 
                           Hij::AbstractMatrix{Float64}, ϕ_ν::AbstractMatrix{Float64}, 
                            Δν::AbstractVector{Float64}, α::AbstractVector{Float64}, 
                          Δν_D::AbstractVector{Float64}, ∇dotu::AbstractVector{Float64}, 
                             T::AbstractVector{Float64} )

    Nr = length(α); Nν = length(ν)

    @inbounds begin
    
        @threads for i in 1:Nr
            @simd for j = 1:Nν
                Δx            = Δν[j]/Δν_D[i]
                Hij[i,j]      = Htophat( (ν[j] - νLyα)/Δν_D[i],  a(T[i]),  Δx )
            end
            # NORMALIZED EMISSION PROFILE:
            @view( ϕ_ν[i,:] ) .= Hij[i,:] ./ sum(Δν[k] * Hij[i, k] for k in 1:Nν) 
        end

    
        @threads for i in 1:Nr
            
            # Convenient factors:
            PREFACT  = (1/2) * α[i] * (Δν_D[i]^2)
            VELOCITY = ∇dotu[i] / (3 * c)
            RECOIL_0 = h / (kB * T[i])

            @simd for j in 2:Nν-1
                # Get K and C:
                Kjp         = PREFACT * Hij[i, j+1]
                Kj          = PREFACT * Hij[i, j]
                Kjm         = PREFACT * Hij[i, j-1]
                Cjp         = VELOCITY * ν[j]  +  PREFACT * Hij[i, j+1] * ( RECOIL_0 - 2/ν[j+1] )
                Cj          = VELOCITY * ν[j]  +  PREFACT * Hij[i, j] * ( RECOIL_0 - 2/ν[j] )
                Cjm         = VELOCITY * ν[j]  +  PREFACT * Hij[i, j-1] * ( RECOIL_0 - 2/ν[j-1] )  

                # FREQUENCY FACTORS F_{i,j+1/2}, F_{i,j}, F_{i,j-1/2}:
                Fi_jp[i,j]  = A_mplus(1/Δν[j], Cj, Δν[j], Kj, 
                                              Cjp, Δν[j+1], Kjp)

                Fi_j[i,j]   = A_m(1/Δν[j], 1/Δν[j], 
                                   Cj, Δν[j], Kj, 
                                  Cjp, Δν[j+1], Kjp, 
                                  Cjm, Δν[j-1], Kjm)

                Fi_jm[i,j]  = A_mminus(1/Δν[j], Cj, Δν[j], Kj, 
                                               Cjm, Δν[j-1], Kjm)
            
            end

        end

    end # End of inbounds

    return 
end


""" Compute **new** frequency related factors H_{i,j}, (ϕ_ν)_{i,j},
    F_{i,j+1/2}, F_{i,j}, and F_{i,j-1/2}.

    **ARGUMENTS:**

    - Δν      :   Array of frequency bin widths (Hz)
    - α       :   Array of HI opacities (cm^-1) in each radial cell, ≡ n_{HI}*σ_0
    - Δν_D    :   Array of Doppler widths (Hz) for each radial cell 
    - ∇dotu   :   Array of velocity divergences for each radial cell, ≡ ∇⋅u 

    **RETURNS:**

    - ϕ_ν     :   Matrix of emission profile values, for each cell, normalized so ∫dν ϕ_ν = 1
    - Fi_jp   :   Matrix of frequency flux/leakage factors F_{i,j+1/2}
    - Fi_j    :   Matrix of frequency flux/leakage factors F_{i,j}
    - Fi_jm   :   Matrix of frequency flux/leakage factors F_{i,j-1/2}
    
    """
function GetFreqFactors( ν::AbstractVector{Float64}, Hij::AbstractMatrix{Float64}, Δν::AbstractVector{Float64}, 
                         α::AbstractVector{Float64}, Δν_D::AbstractVector{Float64},
                     ∇dotu::AbstractVector{Float64}, T::AbstractVector{Float64} )

    # Get grid size:
    Nr = length( α ); Nν = length( ν ) 

    # Create arrays for needed factors:
    ϕ_ν      = similar( Hij )
    Fi_jp    = similar( Hij )
    Fi_j     = similar( Hij )
    Fi_jm    = similar( Hij )

    # Get their values:
    UpdateFreqFactors!( ν, Fi_jp, Fi_j, Fi_jm, Hij, 
                        ϕ_ν, Δν, α, Δν_D, ∇dotu, T )

    return ϕ_ν, Fi_jp, Fi_j, Fi_jm
end


end # End of module