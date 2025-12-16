
module M1Closure
""" MODULE:  M1Closure

    PURPOSE: Contains the M1 closure(s) for modelling spatial 
             radiative transfer. 

"""

__precompile__()

using Base.Threads      # For CPU acceleration

# Export functions:
export Update_D_Edd!, Update_D_Edd_Lyα!



""" The Levermore (1984) closure factor for the Eddington tensor.  """
@inline function χ_L84( f::Float64 )
    f2 = f^2
    χ  = ( 3  +  4 * f2 ) / ( 5  +  2 * sqrt( 4  -  3 * f2 ) )

    return χ
end


const OneThird::Float64 = 1 / 3       # For isotropic limit, often used  
const fmax::Float64     = 1.0 - 1e-12 # To avoid rounding errors from calculation of χ 
const Jeps::Float64     = 1e-30
const Heps::Float64     = 1e-30

""" Updates the relevant Eddington factors D = K / J, for a single 
    frequency bin. """
function Update_D_Edd!( D_rr::AbstractArray{Float64, 2},  # rr-component of D (N_r x N_Z elements)
                        D_ZZ::AbstractArray{Float64, 2},  # ZZ-component of D (N_r x N_Z elements)
                        D_rZ::AbstractArray{Float64, 2},  # rZ-component of D (N_r x N_Z elements)
                         H_r::AbstractArray{Float64, 2},  # H in r-direction  (N_r x N_Z elements)
                         H_Z::AbstractArray{Float64, 2},  # H in Z-direction  (N_r x N_Z elements) 
                           J::AbstractArray{Float64, 2}   # J                 (N_r x N_Z elements)
                       )

    N_r, N_Z    = size( J )  # Grid size
    dN_r        = 1:N_r
    dN_Z        = 1:N_Z

    @threads for j in dN_Z 
        @inbounds @simd for i in dN_r 

            J_ij    = J[i,j]
            H_r_ij  = H_r[i,j]
            H_Z_ij  = H_Z[i,j]
            Hmag_ij = hypot( H_r_ij, H_Z_ij ) 

            if J_ij < Jeps || Hmag_ij < Heps
                # If the flux is zero, or if we have no intensity at all,
                # we assume isotropic intensity:
                D_rr[i,j] = OneThird
                D_ZZ[i,j] = OneThird
                D_rZ[i,j] = 0.0

            else
                # Compute f and χ:
                f = clamp( Hmag_ij / J_ij, 0.0, fmax )
                χ = χ_L84( f )
 
                # Convenient factors:
                inv_Hmag_ij = 1 / Hmag_ij
                a           = (1  -  χ) / 2
                b           = (3 * χ  -  1) / 2
                nr          = H_r_ij * inv_Hmag_ij  # r-component of the flux unit vector n
                nZ          = H_Z_ij * inv_Hmag_ij  # Z-component of the flux unit vector n

                # Get the components of D:
                D_rr[i,j]   = a  +  b * nr * nr 
                D_ZZ[i,j]   = a  +  b * nZ * nZ 
                D_rZ[i,j]   = b * nr * nZ  
            end

        end # End of i-loop
    end # End of j-loop 

end


""" This function updates the Eddington factors for all Lyα frequency bins. """
@views function Update_D_Edd_Lyα!( D_rr_Lyα::AbstractArray{Float64, 3}, 
                                   D_ZZ_Lyα::AbstractArray{Float64, 3},
                                   D_rZ_Lyα::AbstractArray{Float64, 3},
                                   H_r_Lyα::AbstractArray{Float64, 3},
                                   H_Z_Lyα::AbstractArray{Float64, 3},
                                   J_Lyα::AbstractArray{Float64, 3}
                                  )
    N_ν = size( J_Lyα, 3 )

    for k in 1:N_ν
        @inbounds begin 
            Update_D_Edd!( D_rr_Lyα[:, :, k], D_ZZ_Lyα[:, :, k], D_rZ_Lyα[:, :, k],
                           H_r_Lyα[:, :, k],  H_Z_Lyα[:, :, k],  J_Lyα[:, :, k] )
        end
    end

end


end # End of module 
