
""" FILE:     LymanAlphaEmission.jl

    PURPOSE:    Contains everything related to the Lyα emissivity. This includes both realistic emissivities 
                from recombinations and collisional excitation, and also a toy model setup corresponding to a 
                fixed Lyα luminosity within a specified radius. 

    REFERENCES: The probabilities of Lyα emission (PLyα) from Case-B & Case-A recombinations
                are taken from:

                (1) Cantalupo et al. (2008) 'Mapping neutral hydrogen during reionization
                with the Lyα emisson from quasar ionization fronts', ApJ, 672, 1, 48-58

                (2) Dijkstra (2014) 'Lyα Emitting Galaxies as a Probe of Reionisation',
                PASA, 31, e040
                
                (3) Storey & Hummer (1995) 'Recombination line intensities for hydrogenic ions-IV. 
                Total recombination coefficients and machine-readable tables for Z=1 to 8', 
                MNRAS, 272, 1, 41-48

                Collisional excitation is modelled using the fits in:

                (4) Smith et al. (2022) 'The physics of Lyman-α escape from disc-like galaxies',
                MNRAS, 517, 1, 1-27
"""

using .ConstantsAndFunctions  # Needed constants & functions
using Interpolations          # For fitting the probability of Lyα emission following a recombination 

export ∫dν_js_recLyα, ∫dν_js_collLyα

####################################################################
#########                                                  #########
#########   R E C O M B I N A T I O N  E M I S S I O N     #########
#########                                                  #########
####################################################################

# Data from Storey & Hummer (1995) for the probability of Lyα emission 
# following Case B recombination:

const logT_SH95  = log10.([5e2, 1e3, 3e3, 5e3, 7.5e3, 1e4, 1.25e4, 1.5e4, 2e4, 3e4])  # Temperature grid, log(T / K)
const logne_SH95 = log10.([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])                        # Density grid, log(ne / cm⁻³)

const Prob_Lyα_SH95_Data = [
  0.80793637  0.90215197  0.98157401  0.99705340  0.99933853  0.99934845  0.99967896; # T = 5e2    K 
  0.78439303  0.87743841  0.97423464  0.99680114  0.99960243  0.99967816  0.99976710; # T = 1e3    K 
  0.74159266  0.82809599  0.95734434  0.99359891  0.99892162  0.99927343  0.99979471; # T = 3e3    K 
  0.71946081  0.80133228  0.94581237  0.99246182  0.99848958  0.99940329  0.99949246; # T = 5e3    K 
  0.70117195  0.77884530  0.93441344  0.99075058  0.99831301  0.99960223  0.99940949; # T = 7.5e3  K 
  0.68808416  0.76220739  0.92503048  0.98912680  0.99828523  0.99940740  0.99944339; # T = 1e4    K 
  0.67755810  0.74868826  0.91762730  0.98784252  0.99835238  0.99903921  0.99977064; # T = 1.25e4 K 
  0.66862468  0.73785038  0.91033402  0.98701405  0.99817046  0.99892625  0.99953078; # T = 1.5e4  K 
  0.65534642  0.72080463  0.89823705  0.98497224  0.99798854  0.99881330  0.99929092; # T = 2e4    K 
  0.63566376  0.69600394  0.87952243  0.98121463  0.99716593  0.99896144  0.99905958  # T = 3e4    K
]

# Create the interpolation over the data:

const itp_Prob_Lyα_SH95 = extrapolate( interpolate((logT_SH95, logne_SH95), Prob_Lyα_SH95_Data, Gridded(Linear())),
                                       Flat() )

""" The probability of Lyα emission following recombination, as a function 
    of the gas temperature T (K) and electron number density ne (cm⁻³). Uses 
    an interpolation of data from Storey & Hummer (1995). """
@inline function Prob_Lyα_SH95( T::Float64, 
                               ne::Float64 )

    log10T  = clamp( log10(T), logT_SH95[1], logT_SH95[end] )
    log10ne = clamp( log10(ne), logne_SH95[1], logne_SH95[end] )

    return itp_Prob_Lyα_SH95( log10T, log10ne )
end


""" The (frequency-integrated) Lyα emissivity (erg s⁻¹ cm⁻³ sr⁻¹) from 
    recombinations (Case-B or Case-A), as a function of temperature T (K), HII number
    density nHII (cm⁻³), and electron number density ne (cm⁻³). """
function ∫dν_js_recLyα( T::Float64, 
                     nHII::Float64, 
                       ne::Float64, 
                    CaseB::Bool = true,
            DensityEffect::Bool = true   # Whether to include the effect of gas density on the Lyα emission probability (for Case B)
                        )

    if CaseB
        # Case-B rate 
        if DensityEffect
            # Interpolation of data from Storey & Hummer (1995):
            Prob_Lyα  = Prob_Lyα_SH95( T, ne )
        else
            # Fit from ref. 1, ignoring the increased emission probability at high densities:
            T4        = T / 1e4
            Prob_Lyα  = max( 0.0, min( 1.0, 0.686 - 0.106 * log10(T4) - 0.009*(T4^(-0.44)) ) )
        end

        # Emissivity:
        ∫dν_js_rec    = Prob_Lyα * krecHII(T) * nHII * ne * ELyα / Four_π
        return ∫dν_js_rec
    else
        # Case-A rate (see ref. 2):
        T4          = T / 1e4
        Prob_Lyα    = max( 0.0, min( 1.0, 0.41 - 0.165 * log10(T4) - 0.015*(T4^(-0.44)) ) )

        # Emissivity:
        ∫dν_js_rec  = Prob_Lyα * krecHII(T, false) * nHII * ne * ELyα / Four_π
        return ∫dν_js_rec
    end
end


##################################################################################### 
#########                                                                   #########
#########   C O L L I S I O N A L  E X C I T A T I O N  E M I S S I O N     #########
#########                                                                   #########
##################################################################################### 

# Transition energies (1 - n⁻²) * 13.6 eV in units of temperature:
const T_Ryd = ELyC / kB
const T_12  = (1 - 1/2^2) * T_Ryd
const T_13  = (1 - 1/3^2) * T_Ryd
const T_14  = (1 - 1/4^2) * T_Ryd
const T_15  = (1 - 1/5^2) * T_Ryd

""" The (frequency-integrated) Lyα emissivity (erg s⁻¹ cm⁻³ sr⁻¹) from 
    collisional excitation, as a function of temperature T (K), HI number
    density nHI (cm⁻³), and electron number density ne (cm⁻³). See ref. 4. """
function ∫dν_js_collLyα( T::Float64, 
                       nHI::Float64, 
                        ne::Float64 
                         )
    # Convenient variables:
    T6         = min( T, 3e5 ) / 1e6
    T6_squared = T6 * T6 
    T6_cubed   = T6_squared * T6 

    # Effective collision strengths:
    Y12        = 0.267486 + 1.57257*T6 - 6.44026*T6_squared + 11.5401*T6_cubed
    Y13        = 0.0940821 + 2.90748*T6 - 6.69528*T6_squared + 7.86419*T6_cubed 
    Y14        = 0.0415028 + 1.13673*T6 - 3.97454*T6_squared + 6.05941*T6_cubed
    Y15        = 0.032277 + 0.40856*T6 - 1.34604*T6_squared + 2.07558*T6_cubed

    # Effective collisional excitation rate coefficient:
    qcoll = ( 8.63e-6 / (2*sqrt(T)) ) * (
            @fastmath Y12*exp(-T_12/T) + Y13*exp(-T_13/T) + Y14*exp(-T_14/T) + Y15*exp(-T_15/T) )

    return qcoll * nHI * ne * ELyα / Four_π
end

