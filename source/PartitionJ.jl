
module PartitionJ
""" MODULE:  PartitionJ

    PURPOSE: . 

"""

__precompile__()

using Base.Threads      # For CPU acceleration


const τ_small::Float64 = 1e-3
const τ_large::Float64 = 1e3

""" Function f such that the total Lyα intensity is 
    partitioned into trapped & streaming photons according to:
    
    J_t = f * J 
    J_s = (1 - f) * J 

    Following Rosdahl & Teyssier (2015), we use:

    f   = exp( - 2 / 3 * τ_cell ) ,

    where τ_cell is the (harmonic mean) cell optical depth.
"""
@inline function f_partition( τ_cell::Float64 )

    if τ_cell < τ_small
        # No trapped photons if the optical depth is near zero:
        return 0.0
    elseif τ_cell > τ_large
        # For large optical depths we use a Taylor expansion:
        f = 1  -  2 / (3 * τ_cell)
        return f
    else
        # Otherwise we use the general formula:
        f = @fastmath exp( - 2 / (3 * τ_cell) )
        return f 
    end

end


function PartitionJ!( J_Lyα_t::AbstractArray{Float64, 3}, # Trapped Lyα intensity
                      J_Lyα_s::AbstractArray{Float64, 3}, # Streaming Lyα intensity
                       Hvoigt::AbstractArray{Float64, 3}, # Voigt profile for all cells,
                            α::AbstractArray{Float64, 2}, # Lyα opacity α = nHI * σ0 for all cells 
                     )

    N_r, N_Z, N_ν = size( J_Lyα )
    dN_r = 1:N_r
    dN_Z = 1:N_Z
    dN_ν = 1:N_ν

    for k in dN_ν
        @threads for j in dN_Z
            @inbounds @simd for i in dN_r
                # Total intensity:
                J_Lyα_tot      = J_Lyα_t[i,j,k] + J_Lyα_s[i,j,k]

                # Compute cell optical depth & partition factor:
                τ_cell         = Compute_τ_cell( α[i,j,k], Hvoigt[i,j,k] )
                f              = f_partition( τ_cell )

                # Partition trapped & streaming photons:
                J_Lyα_t[i,j,k] = f * J_Lyα_tot
                J_Lyα_s[i,j,k] = max( (1 - f), 0.0 ) * J_Lyα_tot

            end # End of i-loop
        end # End of j-loop 
    end # End of ν-loop

end



end # End of module 
