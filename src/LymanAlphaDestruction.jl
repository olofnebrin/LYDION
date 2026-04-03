
""" FILE:       LymanAlphaDestruction.jl

    PURPOSE:    Contains constants/parameters and functions related to
                non-dust destruction of Lyα photons. This includes 
                inelastic Raman scattering, 2p ⇌ 2s transitions, 
                and absorption by H₂ lines. 

    REFERENCES: The parameters and fits are taken from:

                (1) Nebrin et al. (2025) 'Lyman-α feedback prevails at
                Cosmic Dawn: implications for the first galaxies, stars,
                and star clusters', MNRAS, 537, 2, 1646-1687

"""

using .ConstantsAndFunctions  # Needed constants & functions


######################################################################
#######                                                        #######
#######      D E S T R U C T I O N  P R O B A B I L I T Y      #######
#######                                                        #######
######################################################################

export pd_Lyα

""" The Lyα destruction probability, as estimated in ref. (1).

    **ARGUMENTS:**

    - nHII  :  H II number density (cm^-3)
    - ne    :  e⁻ number density (cm^-3)
    - nHI   :  H I number density (cm^-3)
    - nHeI  :  He I number density (cm^-3)
    - nH2   :  H2 number density (cm^-3)
    - T     :  Gas temperature (K)
    - z     :  Redshift (default = 0)

    **RETURNS:**

    - pd    :  Lyα destruction probability per scattering event """
function pd_Lyα( nHII::Float64, 
                   ne::Float64, 
                  nHI::Float64, 
                 nHeI::Float64,
                  nH2::Float64,
                    T::Float64, 
                    z::Float64 = 0.0 
                    )
    T         = clamp( T, 1.0, 1e5 )
    T_100     = T / 100.0        
    logT      = log10( T )
    logT_sq   = logT * logT       # logT^2
    logT_cu   = logT_sq * logT    # logT^3
    logT_4    = logT_cu * logT    # logT^4 

    # CMB stimulated transition rates:
    Γ_CMB2p2s = 3.1e-6 * (1+z)    # 2p → 2s
    Γ_CMB2s2p = 3 * Γ_CMB2p2s     # 2s → 2p

    # 2s → 2p rates for charged particles (H II, e⁻):
    C_e       = 5e-5 * exp( - (2.5/T)^2 ) * ne / (1 + T_100^0.45)
    C_HII     = ( 6.98e-3 * (1 + 0.933*logT) * exp( - (601/T)^0.406 ) / (T^0.5)
              +   10^(-7.69 + 1.21*logT + 0.455*logT_sq - 0.199*logT_cu + 0.0182*logT_4)  ) * nHII

    # 2s → 2p rates for neutral atoms/molecules (He I, H2, and H I):
    C_HeI     = 2.6e-9 * (T_100^0.315) * nHeI
    C_H2      = 1.7e-9 * (T_100^0.3) * nH2
    C_HI      = ( 4.1e-11 * exp(-(1.6/T)^(3/4) - 0.15*T)
              +  (1.6e-10 + 1.57e-12 * (T^1.1)) / ( (8.9 * (T^(-0.55)) + 0.078 * (T^0.35))^2 ) ) * nHI

    # Total 2s → 2p and 2p → 2s rates:
    C_2s2p    = C_e  +  C_HII  +  C_HeI  +  C_H2  +  C_HI 
    C_2p2s    = OneThird * C_2s2p 

    # Destruction probability from 2p → 2s:

    R_2p2s    = TwoThirds * A_2p2s  +  C_2p2s  +  Γ_CMB2p2s
    R_2s2p    = A_2s2p + C_2s2p + Γ_CMB2s2p

    pd_HI     = R_2p2s / ( (ALyα/A_2γ) * (A_2γ + R_2s2p) + R_2p2s )

    # Destruction probability from H2 line absorption:
    if nHI > 1e-20
        pd_H2 = min( ( 33.0 * exp(-14540.0/T) + 13.0 * exp(-15564.0/T) ) / (
                      46.0 + 1e3 * exp(-(1e3/T)^1.5) ), 1.3e-3) * (nH2/nHI)
    else
        # Ignore the H₂ destruction probability 
        pd_H2 = 0.0
    end

    # Total destruction probability:
    pd        = min( pd_HI + pd_H2, 1.0 )

    return pd
end
