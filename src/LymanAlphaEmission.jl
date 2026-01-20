
""" FILE:       LymanAlphaEmission.jl

    PURPOSE:    Contains everything related to the Lyα emissivity from 
                recombinations and collisional excitation.

    REFERENCES: The probabilities of Lyα emission (PLyα) from Case B & Case A recombinations
                are taken from:
                
                (1) Storey & Hummer (1995) 'Recombination line intensities for hydrogenic ions-IV. 
                Total recombination coefficients and machine-readable tables for Z=1 to 8', 
                MNRAS, 272, 1, 41-48

                Collisional excitation is modelled using the fits in:

                (2) Smith et al. (2022) 'The physics of Lyman-α escape from disc-like galaxies',
                MNRAS, 517, 1, 1-27
"""

using .ConstantsAndFunctions  # Needed constants & functions
using Interpolations          # For fitting the probability of Lyα emission following a recombination 

# Uncomment *only* if you are testing this file separately (i.e. not in an RHD simulation):
# include("PhotoChemistryAndCooling/GasChemistryRates.jl")

export ∫dν_js_recLyα, ∫dν_js_collLyα

####################################################################
#########                                                  #########
#########   R E C O M B I N A T I O N  E M I S S I O N     #########
#########                                                  #########
####################################################################

# Data from Storey & Hummer (1995) for the probability of Lyα emission 
# following Case B and Case A recombination:

const logT_SH95        = log10.([5e2, 1e3, 3e3, 5e3, 7.5e3, 1e4, 1.25e4, 1.5e4, 2e4, 3e4])               # Temperature grid, log(T / K)
const logne_SH95_CaseB = log10.([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14])  # Density grid for Case B, log(ne / cm⁻³)
const logne_SH95_CaseA = log10.([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10])                          # Density grid for Case A, log(ne / cm⁻³)

const Prob_Lyα_SH95_CaseBData = [ 
  0.80793637 0.90215197 0.98157401 0.9970534  0.99933853 0.99934845 0.99967896 0.99948547 0.99931146 0.99951894 0.9997498  0.99864818 0.99372263; # T = 5e2    K 
  0.78439303 0.87743841 0.97423464 0.99680114 0.99960243 0.99967816 0.9997671  0.99939133 0.99922946 0.99960973 0.99937386 0.99906812 0.9954834;  # T = 1e3    K
  0.74159266 0.82809599 0.95734434 0.99359891 0.99892162 0.99927343 0.99979471 0.99938909 0.99899055 0.99925979 0.99957955 0.99923525 0.99703571; # T = 3e3    K 
  0.71946081 0.80133228 0.94581237 0.99246182 0.99848958 0.99940329 0.99949246 0.99952048 0.99981447 0.99956726 0.9993515  0.99952024 0.99692499; # T = 5e3    K 
  0.70117195 0.7788453  0.93441344 0.99075058 0.99831301 0.99960223 0.99940949 0.99946884 0.99948297 0.99947146 0.99965311 0.99918254 0.99690311; # T = 7.5e3  K
  0.68808416 0.76220739 0.92503048 0.9891268  0.99828523 0.9994074  0.99944339 0.99952601 0.99932358 0.99927528 0.99938987 0.99904232 0.9949615;  # T = 1e4    K 
  0.6775581  0.74868826 0.9176273  0.98784252 0.99835238 0.99903921 0.99977064 0.99954296 0.99930529 0.99954743 0.99944997 0.99874439 0.9916493;  # T = 1.25e4 K 
  0.66862468 0.73785038 0.91033402 0.98701405 0.99817046 0.99892625 0.99953078 0.99960556 0.9993362  0.99941558 0.99944365 0.99828039 0.98817287; # T = 1.5e4  K
  0.65534642 0.72080463 0.89823705 0.98497224 0.99798854 0.9988133  0.99929092 0.99909416 0.99933199 0.9995549  0.99924446 0.99764512 0.97960152; # T = 2e4    K 
  0.63566376 0.69600394 0.87952243 0.98121463 0.99716593 0.99896144 0.99905958 0.9996252  0.99953649 0.99931002 0.99872497 0.99561256 0.96391965  # T = 3e4    K
]

const Prob_Lyα_SH95_CaseAData = [ 
  0.58042473 0.59595911 0.60335363 0.58990186 0.56623927 0.53426043 0.492528   0.44534414 0.40955195; # T = 5e2    K 
  0.54712958 0.56465163 0.57999443 0.57241623 0.5541246  0.5276139  0.49052298 0.44606687 0.4101595;  # T = 1e3    K 
  0.48320245 0.50151448 0.52818381 0.53008608 0.52008701 0.50298698 0.47625175 0.43924427 0.40696841; # T = 3e3    K 
  0.44933274 0.46714191 0.49852558 0.50475148 0.49815515 0.48505756 0.46319941 0.43069186 0.40125666; # T = 5e3    K 
  0.42087234 0.43810743 0.47280049 0.48268151 0.47821493 0.46794659 0.4497736  0.42097532 0.39442497; # T = 7.5e3  K
  0.40023971 0.41675887 0.45373431 0.46591319 0.46300415 0.45454038 0.43876097 0.41249415 0.38810578; # T = 1e4    K 
  0.3840314  0.40007514 0.43832241 0.45257107 0.45064954 0.4434932  0.42939783 0.40511911 0.38230403; # T = 1.25e4 K 
  0.37046215 0.38617473 0.42546609 0.44143608 0.44049428 0.43394168 0.42127472 0.39854551 0.37684759; # T = 1.5e4  K 
  0.34937867 0.36397106 0.40484801 0.42349201 0.42367822 0.41860328 0.40761256 0.38699493 0.36739625; # T = 2e4    K 
  0.31960139 0.33302781 0.37528203 0.39805701 0.39984039 0.39602958 0.38718651 0.36963656 0.3518518   # T = 3e4    K
 ]

# Create the interpolations over the data:

const itp_ProbCaseB_Lyα_SH95 = extrapolate( interpolate((logT_SH95, logne_SH95_CaseB), Prob_Lyα_SH95_CaseBData, Gridded(Linear())),
                                            Flat() )

const itp_ProbCaseA_Lyα_SH95 = extrapolate( interpolate((logT_SH95, logne_SH95_CaseA), Prob_Lyα_SH95_CaseAData, Gridded(Linear())),
                                            Flat() )

""" The probability of Lyα emission following recombination, as a function 
    of the gas temperature T (K) and electron number density ne (cm⁻³). Uses 
    an interpolation of data from Storey & Hummer (1995). """
@inline function Prob_Lyα_SH95( T::Float64, 
                               ne::Float64,
                            CaseB::Bool )

    log10T  = clamp( log10(T), logT_SH95[1], logT_SH95[end] )

    if CaseB
        log10ne = clamp( log10(ne), logne_SH95_CaseB[1], logne_SH95_CaseB[end] )

        return itp_ProbCaseB_Lyα_SH95( log10T, log10ne )
    else
        log10ne = clamp( log10(ne), logne_SH95_CaseA[1], logne_SH95_CaseA[end] )

        return itp_ProbCaseA_Lyα_SH95( log10T, log10ne )
    end
end


""" The (frequency-integrated) Lyα emissivity (erg s⁻¹ cm⁻³ sr⁻¹) from 
    recombinations, as a function of temperature T (K), HII number
    density nHII (cm⁻³), and electron number density ne (cm⁻³).
    
    If the on-the-spot (OTS) approximation is used, the Case B 
    recombination coefficient & Lyα emission probability is adopted.
    Otherwise, the Case A recombination coefficient & Lyα emission probability 
    is used. """
function ∫dν_js_recLyα( T::Float64, 
                     nHII::Float64, 
                       ne::Float64, 
                      OTS::Bool = false
                        )

    if OTS
        # On-the-spot approximation → Case B rate:
        Prob_Lyα    = Prob_Lyα_SH95( T, ne, true )

        # Emissivity:
        ∫dν_js_rec  = Prob_Lyα * krecHII(T, true) * nHII * ne * ELyα * inv_Four_π
        return ∫dν_js_rec
    else
        # Case A rate:
        Prob_Lyα    = Prob_Lyα_SH95( T, ne, false )

        # Emissivity:
        ∫dν_js_rec  = Prob_Lyα * krecHII(T, false) * nHII * ne * ELyα * inv_Four_π
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
    density nHI (cm⁻³), and electron number density ne (cm⁻³). See ref. 2. """
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

