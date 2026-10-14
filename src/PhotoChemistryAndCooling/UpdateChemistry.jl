
""" FILE:        UpdateChemistry.jl

    PURPOSE:     Contains the basic time-update of the chemical
                 abundances of H I, H II, H₂, D I, D II, HD, He I, He II, 
                 He III, C I, C II, CHₓ, CO, O I, O II, O III, O IV, OHₓ,
                 S I, S II, S III, and S IV. 

"""

using .ConstantsAndFunctions               # Needed constants
using  StaticArrays, LinearAlgebra         # For taking inverse etc

export UpdateChemistrySubStep


""" A simple semi-implicit BDF1 solution to ẋ = C - Dx, 
    keeping the creation term C and destruction term D constant.
    Used only for sequential semi-implicit updates of species (e.g. sulfur). """
@inline function BDF1_ChemistryUpdate( xold::Float64,
                                          C::Float64, 
                                          D::Float64, 
                                      Δtsub::Float64 )

    return @fastmath ( xold  +  C * Δtsub ) / ( 1.0  +  D * Δtsub )
end


const xTINY::Float64 = 1e-20          # To avoid division by zero 

""" Update of the carbon and oxygen species over the sub-step Δtsub. """
function CarbonOxygenChemistry( xOI_old::Float64, 
                               xOII_old::Float64,
                              xOIII_old::Float64,
                               xOIV_old::Float64,
                                xCI_old::Float64,
                               xCII_old::Float64, 
                              xCIII_old::Float64, 
                                xCO_old::Float64,
                                     nC::Float64,
                                     nO::Float64,
                                nHI_new::Float64,
                               nHII_new::Float64,
                                nH2_new::Float64,
                                 ne_new::Float64,
                                 ΓionH2::Float64,
                                 ΓionOI::Float64,
                                ΓionOII::Float64,
                               ΓionOIII::Float64,
                                 ΓionCI::Float64,
                                ΓionCII::Float64,
                                ΓdissCO::Float64,
                               ΓdissOHx::Float64,
                               ΓdissCHx::Float64,
                                      T::Float64,
                                  Δtsub::Float64,
                                 log10T::Float64 = log10(T)  # log₁₀(T), can be computed once outside for efficiency 
                                )

    nH          = @fastmath nHI_new  +  nHII_new  +  2 * nH2_new
    C_over_O    = nC / max(nO,  xTINY * nH)
    xCO_max     = min( 1 / C_over_O, 1.0 )    # Maximum value of xCO = nCO / nC 

    # Get old CHₓ fraction (for nCHx_old):
    xCHx_old    = max( 1 - xCI_old - xCII_old - xCIII_old - xCO_old, 0.0 )

    # Densities:
    nCI_old     = @fastmath xCI_old  * nC         
    nCHx_old    = @fastmath xCHx_old * nC         
    nOI_old     = @fastmath xOI_old  * nO    
    
    # Limit nH₃⁺ in extreme cases:
    nH3plus_max = @fastmath OneThird * nH

    # Needed rates:
    KOIe       = kOIe(T)
    KHIIOI     = kHIIOI(T)
    KOIIe      = kOIIe(T)
    KOIIHI     = kOIIHI(T)
    KOIIH2_1   = kOIIH2_1(T, nH2_new, ne_new)
    KOIIH2_2   = kOIIH2_2(T, nH2_new, ne_new)
    KOIIH2     = @fastmath KOIIH2_1 + KOIIH2_2
    KOIIe_ion  = kOIIe_collion(T)  # Collisional ionziation, O II → O III 
    KOIIIe_ion = kOIIIe_collion(T) # Collisional ionziation, O III → O IV
    KOIIIHI    = kOIIIHI(T)        # Charge exchange, O III → O II 
    KOIVHI     = kOIVHI(T)         # Charge exchange, O IV → O III
    KOIIIe     = kOIIIe(log10T)    # Recombination of O III to O II (uses log10(T) for LUT) 
    KOIVe      = kOIVe(log10T)     # Recombination of O IV to O III (uses log10(T) for LUT) 
    KOHxCI     = kOHxCI(T)
    KOHxOI     = kOHxOI(T)
    KCHxOI     = kCHxOI(T)
    KCHxHI     = kCHxHI(T)
    KOIH3plus  = kOIH3plus(T, nH2_new, ne_new)
    KCIH3plus  = kCIH3plus(log10T) # Uses log10(T) for LUT
    KH3pluse   = kH3pluse(T)

    KCIe       = kCIe(T)
    KCIIe      = kCIIe(log10T)     # Recombination of C II to C I (uses log10(T) for LUT)
    KCIIH2_1   = kCIIH2_1(T)
    KCIIH2_2   = kCIIH2_2(T)
    KCIIH2     = @fastmath KCIIH2_1 + KCIIH2_2
    KCIIe_ion  = kCIIe_collion(T)  # Collisional ionziation, C II → C III 
    KCIIIHI    = kCIIIHI(T)        # Charge exchange, C III → C II
    KCIIIe     = kCIIIe(log10T)    # Recombination of C III to C II (uses log10(T) for LUT)

    # Determine  H₃⁺
    if ΓionH2 > 0.0 && nH2_new > 0.0
        nH3plus = max( min( ΓionH2 * nH2_new / (KCIH3plus * nCI_old  +  KOIH3plus * nOI_old  +  KH3pluse * ne_new), 
                            nH3plus_max ), 0.0 )
    else
        nH3plus = 0.0
    end

    ###############################################################
    #######                                                 #######
    #######         O X Y G E N  C H E M I S T R Y          #######
    #######                                                 #######
    ###############################################################

    # We first do a preliminary/predictor update of the oxygen chemistry, 
    # including CO. We update O I, O II, O III, O IV, CO, and eliminate OHₓ by
    # conservation: xOHₓ = 1 - xOI - xOII - xOIII - xOIV - (nC/nO) * xCO.

    # Creation rates:
    C_OII_to_OI   =  @fastmath KOIIe * ne_new  +  KOIIHI * nHI_new  +  KOIIH2_2 * nH2_new 
    C_OHx_to_OI   =  @fastmath ΓdissOHx  +  KOHxOI * nO * xOI_old

    C_OI_to_OII   =  @fastmath ΓionOI  +  KOIe * ne_new  +  KHIIOI * nHII_new 
    C_OIII_to_OII =  @fastmath KOIIIHI * nHI_new  +  KOIIIe * ne_new    # Charge exchange + recombination of O III

    C_OII_to_OIII =  @fastmath ΓionOII  +  KOIIe_ion * ne_new
    C_OIV_to_OIII =  @fastmath KOIVHI * nHI_new  +  KOIVe * ne_new      # Charge exchange + recombination of O IV

    C_OIII_to_OIV =  @fastmath ΓionOIII  +  KOIIIe_ion * ne_new 

    C_OI_to_CO    =  @fastmath KCHxOI * nO * xCHx_old
    C_OHx_to_CO   =  @fastmath KOHxCI * nO * xCI_old 

    # Total destruction rates for O I, O II, O III, and O IV (for some of the terms, we can re-use 
    # the creation rates):
    D_OI          =  @fastmath ( C_OI_to_OII  +  KCHxOI * nCHx_old
                               + KOIH3plus * nH3plus )

    D_OII         =  @fastmath ( KOIIe * ne_new  +  KOIIHI * nHI_new  +  KOIIH2 * nH2_new  
                               + C_OII_to_OIII )

    D_OIII        =  @fastmath C_OIII_to_OII + C_OIII_to_OIV     
    
    D_OIV         =  C_OIV_to_OIII

    # Set up linear equations & solve:
    A_O         =  @SVector [ C_OHx_to_OI, 0.0, 0.0, 0.0, C_OHx_to_CO ]

    B_O_11      =  - ( D_OI  +  C_OHx_to_OI )
    B_O_12      =  C_OII_to_OI - C_OHx_to_OI
    B_O_13      =  - C_OHx_to_OI
    B_O_14      =  - C_OHx_to_OI
    B_O_15      =  C_over_O * ( ΓdissCO  -  C_OHx_to_OI )

    B_O_21      =  C_OI_to_OII
    B_O_22      =  - D_OII
    B_O_23      =  C_OIII_to_OII
    B_O_24      =  0.0
    B_O_25      =  0.0

    B_O_31      =  0.0
    B_O_32      =  C_OII_to_OIII
    B_O_33      =  - D_OIII
    B_O_34      =  C_OIV_to_OIII 
    B_O_35      =  0.0

    B_O_41      =  0.0
    B_O_42      =  0.0
    B_O_43      =  C_OIII_to_OIV
    B_O_44      =  - D_OIV
    B_O_45      =  0.0 

    B_O_51      =  C_OI_to_CO - C_OHx_to_CO
    B_O_52      =  - C_OHx_to_CO
    B_O_53      =  - C_OHx_to_CO
    B_O_54      =  - C_OHx_to_CO
    B_O_55      =  - ( ΓdissCO  +  C_over_O * C_OHx_to_CO )

    B_O         =  @SMatrix [ B_O_11  B_O_12  B_O_13  B_O_14  B_O_15;
                              B_O_21  B_O_22  B_O_23  B_O_24  B_O_25;
                              B_O_31  B_O_32  B_O_33  B_O_34  B_O_35;
                              B_O_41  B_O_42  B_O_43  B_O_44  B_O_45;
                              B_O_51  B_O_52  B_O_53  B_O_54  B_O_55 ]

    xO_old  =  @SVector [ xOI_old, xOII_old, xOIII_old, xOIV_old, xCO_old ]
    xO_new  = (I - Δtsub * B_O) \ ( xO_old  +  A_O * Δtsub ) 

    # Get predictor O I, O II, O III, O IV, CO fractions:
    xOI_star, xOII_star, xOIII_star, xOIV_star, xCO_star = xO_new

    # Clamp for safety:
    xOI_star   = max( xOI_star,   0.0 )
    xOII_star  = max( xOII_star,  0.0 )
    xOIII_star = max( xOIII_star, 0.0 )
    xOIV_star  = max( xOIV_star,  0.0 )
    xCO_star   = clamp( xCO_star, 0.0, xCO_max )

    # Get OHₓ from conservation:
    xOHx_star  = clamp( 1 - xOI_star - xOII_star - xOIII_star - xOIV_star - C_over_O * xCO_star, 
                        0.0 , 1.0 )
    nOHx_star  = @fastmath xOHx_star * nO 

    # Get nOI_new:
    nOI_star   = @fastmath xOI_star * nO

    # Update H₃⁺:
    if ΓionH2 > 0.0  &&  nH2_new > 0.0
        nH3plus_star = max( min( ΓionH2 * nH2_new / (KCIH3plus * nCI_old  +  KOIH3plus * nOI_star  +  KH3pluse * ne_new), 
                                 nH3plus_max ), 0.0 )
    else
        nH3plus_star = 0.0
    end

    ###############################################################
    #######                                                 #######
    #######         C A R B O N  C H E M I S T R Y          #######
    #######                                                 #######
    ###############################################################

    # Next we do a corrector update of the carbon species C I, C II, C III,
    # and CO (again), using the predictor values for O-bearing species 
    # (except CO).

    C_CII_to_CI   = @fastmath KCIIe * ne_new  +  KCIIH2_2 * nH2_new 
    C_CHx_to_CI   = @fastmath KCHxHI * nHI_new  +  ΓdissCHx
    D_CI          = @fastmath ΓionCI  +  KCIe * ne_new  +  KOHxCI * nOHx_star  +  KCIH3plus * nH3plus_star
    
    C_CI_to_CII   = @fastmath ΓionCI  +  KCIe * ne_new 
    C_CIII_to_CII = @fastmath KCIIIe * ne_new  +  KCIIIHI * nHI_new   # Recombination + charge exchange 
    D_CII         = @fastmath KCIIe * ne_new  +  KCIIH2 * nH2_new  +  ΓionCII  +  KCIIe_ion * ne_new

    C_CII_to_CIII = @fastmath ΓionCII  +  KCIIe_ion * ne_new
    D_CIII        = @fastmath C_CIII_to_CII

    C_CI_to_CO    = @fastmath KOHxCI * nOHx_star
    C_CHx_to_CO   = @fastmath KCHxOI * nOI_star

    # Set up linear equations & solve:
    A_C           =  @SVector [ C_CHx_to_CI, 0.0, 0.0, C_CHx_to_CO ]
    B_C_11        =  - ( D_CI + C_CHx_to_CI )
    B_C_12        =  C_CII_to_CI - C_CHx_to_CI
    B_C_13        =  - C_CHx_to_CI
    B_C_14        =  ΓdissCO - C_CHx_to_CI

    B_C_21        =  C_CI_to_CII 
    B_C_22        =  - D_CII
    B_C_23        =  C_CIII_to_CII
    B_C_24        =  0.0

    B_C_31        =  0.0
    B_C_32        =  C_CII_to_CIII
    B_C_33        =  - D_CIII 
    B_C_34        =  0.0

    B_C_41        =  C_CI_to_CO - C_CHx_to_CO
    B_C_42        =  - C_CHx_to_CO
    B_C_43        =  - C_CHx_to_CO
    B_C_44        =  - ( ΓdissCO  +  C_CHx_to_CO )

    B_C           =  @SMatrix [ B_C_11   B_C_12  B_C_13  B_C_14;
                                B_C_21   B_C_22  B_C_23  B_C_24;
                                B_C_31   B_C_32  B_C_33  B_C_34;
                                B_C_41   B_C_42  B_C_43  B_C_44 ]

    xC_old      =  @SVector [ xCI_old, xCII_old, xCIII_old, xCO_old ]
    xC_new      = (I - Δtsub * B_C) \ ( xC_old  +  A_C * Δtsub ) 

    # Get corrector/final C I, C II, C III, CO fractions:
    xCI_new, xCII_new, xCIII_new, xCO_new = xC_new 

    # Clamp xCO to its physical range:
    xCO_new     = clamp( xCO_new, 0.0, xCO_max )

    ###############################################################
    #######                                                 #######
    #######       R E N O R M A L I Z E  O X Y G E N        #######
    #######                                                 #######
    ###############################################################

    # Renormalize oxygen species to make it consistent with corrector update for CO.
    # We do this by multiplying xOI_star, xOII_star, xOIII_star, xOIV_star, 
    # and xOHx_star by the factor:
    # 
    # f_renorm = (1 - C_over_O * xCO_new) / (xOI_star + xOII_star + xOIII_star + xOIV_star + xOHx_star)

    f_renorm  = max( 1.0 - C_over_O * xCO_new,  0.0 ) / max( 
                xOI_star + xOII_star + xOIII_star + xOIV_star + xOHx_star,  xTINY )

    xOI_new   = max( f_renorm * xOI_star,   0.0 )
    xOII_new  = max( f_renorm * xOII_star,  0.0 )
    xOIII_new = max( f_renorm * xOIII_star, 0.0 )
    xOIV_new  = max( f_renorm * xOIV_star,  0.0 )
    # We also have xOHx_new = f_renorm * xOHx_star, but there is no need to return it, since 
    # we always compute xOHx by remainder when needed. 

    # Enforce positivity:
    xCI_new   = max( xCI_new,   0.0 )
    xCII_new  = max( xCII_new,  0.0 )
    xCIII_new = max( xCIII_new, 0.0 )
    xCO_new   = clamp( xCO_new, 0.0, xCO_max )

    # Renormalize so they add up to 1 exactly:
    ∑xC = xCI_new + xCII_new + xCIII_new + xCO_new   # Sum of all C species except CHₓ

    if ∑xC > 1.0
        inv_∑xC    = 1.0 / ∑xC
        xCI_new   *= inv_∑xC
        xCII_new  *= inv_∑xC
        xCIII_new *= inv_∑xC
        xCO_new   *= inv_∑xC
        xCHx_new   = 0.0      # If ∑xC > 1 initially, there should be no CHₓ
    else 
        xCHx_new  = 1.0 - ∑xC
    end

    return xOI_new, xOII_new, xOIII_new, xOIV_new, xCI_new, xCII_new, xCIII_new, xCO_new
end


""" Update of the sulphur species S II, S III, S IV, over the sub-step Δtsub. """
function SulphurChemistryUpdate( xSII_old::Float64, 
                                xSIII_old::Float64,
                                 xSIV_old::Float64,
                                  nHI_new::Float64,
                                 nHII_new::Float64,
                                   ne_new::Float64,
                                   ΓionSI::Float64,
                                  ΓionSII::Float64,
                                 ΓionSIII::Float64,
                                        T::Float64,
                                    Δtsub::Float64,
                                   log10T::Float64 = log10(T)  # log₁₀(T), can be computed once outside for efficiency
                                  )

    # Get xSI from conservation:
    xSI_old = max( 1.0 - xSII_old - xSIII_old - xSIV_old, 0.0 ) 

    # Recombination rates:
    KSIIe   = kSIIe(log10T)   # Uses log10(T) because of LUT
    KSIIIe  = kSIIIe(log10T)  # Uses log10(T) because of LUT
    KSIVe   = kSIVe(log10T)   # Uses log10(T) because of LUT

    # Charge exchange rates:
    KSIHII  = kSIHII(T)   # Ionization of S I via charge exchange 
    KSIIHI  = kSIIHI(T)
    KSIIIHI = kSIIIHI(T)
    KSIVHI  = kSIVHI(T)

    # Collisional ionization rates: 
    KSIe_collion   = kSIe_collion(log10T)     # Uses log10(T) because of LUT
    KSIIe_collion  = kSIIe_collion(log10T)    # Uses log10(T) because of LUT
    KSIIIe_collion = kSIIIe_collion(log10T)   # Uses log10(T) because of LUT

    # Convenient products, evaluate once:
    KSIIe_ne_new          = @fastmath KSIIe  * ne_new    # Recombinations 
    KSIIIe_ne_new         = @fastmath KSIIIe * ne_new
    KSIVe_ne_new          = @fastmath KSIVe  * ne_new

    KSIHII_nHII_new       = @fastmath KSIHII  * nHII_new # Charge exchange ionization 
    KSIIHI_nHI_new        = @fastmath KSIIHI  * nHI_new  # Charge exchange recombination
    KSIIIHI_nHI_new       = @fastmath KSIIIHI * nHI_new
    KSIVHI_nHI_new        = @fastmath KSIVHI  * nHI_new

    KSIe_collion_ne_new   = @fastmath KSIe_collion   * ne_new  # Collisional ionization 
    KSIIe_collion_ne_new  = @fastmath KSIIe_collion  * ne_new
    KSIIIe_collion_ne_new = @fastmath KSIIIe_collion * ne_new

    ###################################################                      
    ######          STEP 1: UPDATE S I           ######
    ###################################################
    # First we do a preliminary update of S I 
    # for accuracy purposes. 

    # Creation of S I via recombinations of S II and charge 
    # exchange S II → S I via H I:
    C_SI = @fastmath ( KSIIe_ne_new  +  KSIIHI_nHI_new ) * xSII_old
    
    # Destruction of S I via photoionization, collisional 
    # ionization, and ionization charge exchange S I → S II via H II:
    D_SI = @fastmath ΓionSI  +  KSIe_collion_ne_new  +  KSIHII_nHII_new

    # Preliminary update of S I:
    xSI_new = BDF1_ChemistryUpdate( xSI_old, C_SI, D_SI, Δtsub )

    ###################################################                      
    ######          STEP 2: UPDATE S II          ######
    ###################################################

    # Creation of S II via photoionization and collisional ioniation of S I, 
    # recombinations of S III, and charge exchange S III → S II via H I, 
    # and S I → S II via H II:
    C_SII = @fastmath ( ( ΓionSI  +  KSIe_collion_ne_new ) * xSI_new  
                      +   KSIIIe_ne_new   * xSIII_old
                      +   KSIIIHI_nHI_new * xSIII_old
                      +   KSIHII_nHII_new * xSI_new )

    # Destruction of S II via photoionization, collisional 
    # ionization, recombinations, and charge exchange S II → S I via H I:
    D_SII = @fastmath ( ΓionSII  +  KSIIe_collion_ne_new
                      + KSIIe_ne_new  +  KSIIHI_nHI_new )

    # Preliminary update of S II:
    xSII_new = BDF1_ChemistryUpdate( xSII_old, C_SII, D_SII, Δtsub )

    ###################################################                      
    ######          STEP 3: UPDATE S III         ######
    ###################################################

    # Creation of S III via photoionization and collisional ionization of S II, 
    # recombinations of S IV, and charge exchange S IV → S III via H I:
    C_SIII = @fastmath ( ( ΓionSII  +  KSIIe_collion_ne_new ) * xSII_new  
                       +   KSIVe_ne_new   * xSIV_old
                       +   KSIVHI_nHI_new * xSIV_old )

    # Destruction of S III via photoionization, recombinations,
    # collisional ionization, and charge exchange S III → S II via H I:
    D_SIII = @fastmath ( ΓionSIII  +  KSIIIe_ne_new
                       + KSIIIe_collion_ne_new
                       + KSIIIHI_nHI_new )

    # Preliminary update of S III:
    xSIII_new = BDF1_ChemistryUpdate( xSIII_old, C_SIII, D_SIII, Δtsub )

    ###################################################                      
    ######          STEP 4: UPDATE S IV          ######
    ###################################################

    # Creation of S IV via photoionization and collisional 
    # ionization of S III:
    C_SIV = @fastmath ( ΓionSIII  +  KSIIIe_collion_ne_new ) * xSIII_new

    # Destruction of S IV via recombinations, and charge exchange 
    # S IV → S III via H I: 
    D_SIV = @fastmath ( KSIVe_ne_new  +  KSIVHI_nHI_new )

    # Preliminary update of S IV:
    xSIV_new = BDF1_ChemistryUpdate( xSIV_old, C_SIV, D_SIV, Δtsub )

    ###################################################                      
    ######          STEP 5: RENORMALIZE          ######
    ###################################################

    # Renormalize so all fractions add up to 1 exactly. 
    # Do so by correcting the largest fraction (following 
    # the RAMSES RT approach):

    Δf_S  = ( xSI_new + xSII_new + xSIII_new + xSIV_new ) - 1.0
    idx_S = argmax( ( xSI_new, xSII_new, xSIII_new, xSIV_new ) )

    if idx_S == 1
        # In this case, S I is the biggest so no need to correct
        # because we will get it from conservation regardless:
    elseif idx_S == 2
        # In this case, S II is the biggest, so correct it:
        xSII_new  = max( xSII_new   -  Δf_S, 0.0 )
    elseif idx_S == 3
        # In this case, S III is the biggest, so correct it:
        xSIII_new = max( xSIII_new  -  Δf_S, 0.0 )
    else 
        # In this case S IV is the biggest, so correct it:
        xSIV_new  = max( xSIV_new   -  Δf_S, 0.0 )
    end

    # To further ensure that xSI is always ≥ 0, the ions must 
    # add up to ≤ 1, so ensure this: 
    ∑xSions = @fastmath xSII_new + xSIII_new + xSIV_new
    if ∑xSions > 1.0
        inv_∑xSions  = @fastmath 1.0 / ∑xSions
        xSII_new    *= inv_∑xSions
        xSIII_new   *= inv_∑xSions
        xSIV_new    *= inv_∑xSions
    end

    # Return updated S II, S III, S IV:
    return xSII_new, xSIII_new, xSIV_new
end


""" This function computes updated abundances of 
    the species H₂, H I, H II, HD, D I, D II, He I, He II, He III, e⁻,
    O I, O II, O III, O IV, C I, C II, C III, CO, S II, S III, and S IV, 
    over a sub-cycle time-step Δtsub. 

    (Note that not all species enters the arguments below, because 
    we can e.g. get the H I, D I, He I, and e⁻ abundances from particle and 
    charge conservation.) """
@inline function UpdateChemistrySubStep( Δtsub::Float64,
                                         Tnorm::Float64,
                                            Td::NTuple{3, Float64}, # Dust temperatures (Td_Sil, Td_C, Td_PAH) (K)
                                             ρ::Float64,
                                            ρd::NTuple{3, Float64}, # Dust densities (ρd_Sil, ρd_C, ρd_PAH)
                                            nH::Float64, # Total hydrogen number density 
                                            nC::Float64, # Total carbon number density 
                                            nO::Float64, # Total oxygen number density 
                                            nS::Float64, # Total sulfur number density 
                                       nHI_old::Float64,
                                       nH2_old::Float64,
                                      nHII_old::Float64,
                                        ne_old::Float64,
                                       xH2_old::Float64,
                                      xHII_old::Float64, 
                                       xHD_old::Float64,
                                      xDII_old::Float64,
                                     xHeII_old::Float64,
                                    xHeIII_old::Float64,
                                       xOI_old::Float64, 
                                      xOII_old::Float64,
                                     xOIII_old::Float64,
                                      xOIV_old::Float64,
                                       xCI_old::Float64,
                                      xCII_old::Float64, 
                                     xCIII_old::Float64, 
                                       xCO_old::Float64,
                                      xSII_old::Float64,
                                     xSIII_old::Float64,
                                      xSIV_old::Float64,
                                            ZC::Float64,  # Carbon gas metallicity
                                          ZSil::Float64,  # Si gas metallicity
                                      Zgasrest::Float64,  # Non-C/Si gas metallicity
                                        ΓionHI::Float64 = 0.0,
                                        ΓionH2::Float64 = 0.0,
                                        ΓionDI::Float64 = 0.0,
                                       ΓionHeI::Float64 = 0.0,
                                      ΓionHeII::Float64 = 0.0,
                                          Γ_LW::Float64 = 0.0,
                                           Γpd::Float64 = 0.0,
                                       Γ_LW_HD::Float64 = 0.0,
                                        ΓionOI::Float64 = 0.0,
                                       ΓionOII::Float64 = 0.0,
                                      ΓionOIII::Float64 = 0.0,
                                        ΓionCI::Float64 = 0.0,
                                       ΓionCII::Float64 = 0.0,
                                       ΓdissCO::Float64 = 0.0,
                                      ΓdissOHx::Float64 = 0.0,
                                      ΓdissCHx::Float64 = 0.0,
                                        ΓionSI::Float64 = 0.0,
                                       ΓionSII::Float64 = 0.0,
                                      ΓionSIII::Float64 = 0.0,
                                             z::Float64 = 0.0,
                                         CaseB::Bool    = true,
                             HydrogenAndHelium::Bool    = true
                                        )

    # Get xDI_old, needed for the deuterium update:
    xDI_old = max( 1 - xDII_old - xHD_old, 0.0 )

    # Get temperature by computing μ and then using 
    # T = μ × (γad - 1) × Tnorm (where Tnorm ≡ T/[μ(γad - 1)]) :
    μ   = Getμ( xHII_old, xH2_old, xHeII_old, xHeIII_old, HydrogenAndHelium )
    γad = Getγad( xHII_old, xH2_old, xHeII_old, xHeIII_old, HydrogenAndHelium )
    T   = μ * (γad - 1) * Tnorm

    ###################################################                      
    ######      STEP 1: UPDATE H I, H II, H₂     ######
    ###################################################

    # The following rates appear several times, so 
    # compute them once for convenience/optimization:
    RdustH2     = R_dustH2(T, ρd, Td)
    KGP_H2      = kGP_H2(T, ne_old, nHI_old, nHII_old, Γpd, z)
    K3B_H2      = k3B_H2(T)
    Kdiss_H2HI  = kdiss_H2HI(T, nHI_old)
    Kdiss_H2H2  = kdiss_H2H2(T, nH2_old)
    Kdiss_H2e   = kdiss_H2e(T)
    Kdiss_H2HII = kdiss_H2HII(T)
    KrecHII     = krecHII(T, CaseB)
    Kcollion_HI = kcollion_HI(T)

    # Creation rate for H₂ is C_H2_0 × xHI, where: 
    C_H2_0  = ( RdustH2 
              + KGP_H2   * ne_old 
              + K3B_H2   * nHI_old * (nHI_old + nH2_old / 8) )

    # H₂ destruction term (using un-updated values):
    D_H2    = ( Kdiss_H2HI  * nHI_old 
              + Kdiss_H2H2  * nH2_old
              + Kdiss_H2e   * ne_old
              + Kdiss_H2HII * nHII_old
              + Γ_LW + ΓionH2 )

    # Creation rate for H II is C_HII_0 × xHI, where:
    C_HII_0 = ( Kcollion_HI * ne_old + ΓionHI )

    # H II destruction term (using un-updated values):
    D_HII   =   KrecHII * ne_old  +  Kdiss_H2HII * xH2_old * nH

    B_H_11  = - ( D_H2   +  2 * C_H2_0 )
    B_H_12  = - C_H2_0 
    B_H_21  = - 2 * C_HII_0
    B_H_22  = - ( D_HII  +  C_HII_0 )

    # Solve for the updated abundances:
    xH_old  = @SVector [ xH2_old, xHII_old ]
    A_H     = @SVector [ C_H2_0, C_HII_0 ]
    B_H     = @SMatrix [ B_H_11  B_H_12;
                         B_H_21  B_H_22 ]

    xH_new  = (I - Δtsub * B_H) \ ( xH_old  +  A_H * Δtsub ) 

    # Get updated H₂ and H II fractions:
    xH2_new, xHII_new = xH_new

    # Get xHI from conservation:
    xHI_new = max( 1 -  xHII_new - 2 * xH2_new, 0.0 )

    # Next we enforce particle conservation, i.e.
    # 2 * xH2 + xHI + xHII = 1. This is done by correcting
    # the largest fraction so that this holds. 

    Δf_H  = ( 2 * xH2_new + xHI_new + xHII_new ) - 1.0
    idx_H = argmax( ( 2 * xH2_new, xHI_new, xHII_new ) )

    if idx_H == 1
        # In this case, H₂ is the biggest so correct it now 
        # (to avoid errors in μ and temperature calculation):
        xH2_new  = max( xH2_new  -  0.5 * Δf_H, 0.0 )
    elseif idx_H == 2
        # In this case, H I is the biggest, so correct it:
        xHI_new  = max( xHI_new  -  Δf_H, 0.0 )
    else
        # In this case, H II is the biggest, so correct it:
        xHII_new = max( xHII_new -  Δf_H, 0.0 )
    end

    # We also update the temperature (for use in helium
    # chemistry step) using the known value of Tnorm 
    # and the new μ and γ (computed below):
    μ       = Getμ( xHII_new, xH2_new, xHeII_old, xHeIII_old, HydrogenAndHelium )
    γad     = Getγad( xHII_new, xH2_new, xHeII_old, xHeIII_old, HydrogenAndHelium ) 
    T       = μ * (γad - 1) * Tnorm 

    # Update the electron number density:
    ne_new1 = GetElectronDensityFromFracs( xHII_new, xHeII_old, xHeIII_old, 
                                           xDII_old, xCII_old,  xCIII_old, 
                                           xOII_old, xOIII_old, xOIV_old, 
                                           xSII_old, xSIII_old, xSIV_old,
                                           ρ, ZC, ZSil, Zgasrest, HydrogenAndHelium )

    # Get change in ne during the H update (for accuracy checking):
    Δne_H   = abs( ne_new1 - ne_old )

    #########################################################                      
    ######      STEP 2: UPDATE He I, He II, He III     ######
    #########################################################

    # Pre-compute needed rates for He:
    KrecHeII      = krecHeII(T, CaseB)
    Kcollion_HeI  = kcollion_HeI(T)
    Kcollion_HeII = kcollion_HeII(T)
    KrecHeIII     = krecHeIII(T, CaseB) 

    # Creation rate for He II is 
    #
    # (ẋ_HeII)_(+) = C_HeII_0 × xHeI  +  C_HeII_1 × xHeIII ,
    #
    # where: 
    C_HeII_0 =  Kcollion_HeI * ne_new1  +  ΓionHeI 
    C_HeII_1 =  KrecHeIII * ne_new1 

    # Destruction rate for He II (from coll. ionization, recombinations, photoionization) 
    # is:
    #
    # (ẋ_HeII)_(-) = - D_HeII × xHeII ,
    #
    # where:
    D_HeII   =  ( KrecHeII + Kcollion_HeII ) * ne_new1  +  ΓionHeII  

    # Creation rate for He III (coll. ionization & photoionization of He II) is:
    #
    # (ẋ_HeIII)_(+) = C_HeIII × xHeII ,
    #
    # where:
    C_HeIII  =  Kcollion_HeII * ne_new1  +  ΓionHeII  

    # Destruction rate for He III (from recombinations) is:
    #
    # (ẋ_HeIII)_(-) = - D_HeIII × xHeIII ,
    #
    # where:
    D_HeIII  =  KrecHeIII * ne_new1

    # Next we set up 2 × 2 system for (He II, He III) and solve, 
    # and get He I from conservation.
    A_He = @SVector [ C_HeII_0,
                      0.0  ] 

    B_He_11 = - ( D_HeII + C_HeII_0 )
    B_He_12 = C_HeII_1 - C_HeII_0
    B_He_21 = C_HeIII
    B_He_22 = - D_HeIII
    B_He    = @SMatrix [ B_He_11   B_He_12;
                         B_He_21   B_He_22 ]

    xHe_old = @SVector [ xHeII_old, xHeIII_old ]
    xHe_new = (I - Δtsub * B_He) \ ( xHe_old + A_He * Δtsub )
    
    # Unpack updated He fractions:
    xHeII_new, xHeIII_new = xHe_new
    
    # Guard against tiny floating errors & also get He I fraction:
    xHeII_new   = max( xHeII_new, 0.0)
    xHeIII_new  = max( xHeIII_new, 0.0)
    xHeI_new    = max( 1 - xHeII_new - xHeIII_new, 0.0 )

    # Get updated electron number density:
    ne_new2     = GetElectronDensityFromFracs( xHII_new, xHeII_new, xHeIII_new, 
                                               xDII_old, xCII_old,  xCIII_old,    
                                               xOII_old, xOIII_old, xOIV_old,
                                               xSII_old, xSIII_old, xSIV_old,
                                               ρ, ZC, ZSil, Zgasrest, HydrogenAndHelium )

    # Get change in ne during the He update (for accuracy checking):
    Δne_He      = abs( ne_new2 - ne_new1 )

    # We also update the temperature (for use in deuterium
    # chemistry step) using the known value of Tnorm 
    # and the new μ and γ (computed below):
    μ      = Getμ( xHII_new, xH2_new, xHeII_new, xHeIII_new, HydrogenAndHelium )
    γad    = Getγad( xHII_new, xH2_new, xHeII_new, xHeIII_new, HydrogenAndHelium ) 
    T      = μ * (γad - 1) * Tnorm 

    ##########################################                       
    ###### STEP 3: UPDATE D I, D II, HD ######
    ##########################################
    # Next we update the deuterium chemistry. 
    # This is a set of three coupled ODE's which 
    # are linear, so they can be inverted in one go.

    # Updated number densities:
    nH2_new     = xH2_new  * nH 
    nHI_new     = xHI_new  * nH 
    nHII_new    = xHII_new * nH 

    # Needed reaction rates:
    Kcollion_HI = kcollion_HI(T)    # D + e⁻  → D⁺ + e⁻ (same as for H I, but using updated T)
    KrecHII     = krecHII(T, CaseB) # D⁺ + e⁻ → D + hν  (same as for H I, but using updated T)
    K50_D       = k50_D(T)          # D + H⁺  → D⁺ + H
    K51_D       = k51_D(T)          # D⁺ + H  → D + H⁺
    K52_D       = k52_D(T)          # D⁺ + H₂ → H⁺ + HD
    K53_D       = k53_D(T)          # H⁺ + HD → D⁺ + H₂
    K54_D       = k54_D(T)          # D + H₂  → H + HD
    K55_D       = k55_D(T)          # H + HD  → D + H₂

    # Matrix elements:
    A11 = - (ΓionDI  +  Kcollion_HI * ne_new2  +  K50_D * nHII_new  +  K54_D * nH2_new)
    A12 = KrecHII * ne_new2  +  K51_D * nHI_new 
    A13 = K55_D * nHI_new  +  Γ_LW_HD

    A21 = ΓionDI  +  Kcollion_HI * ne_new2  +  K50_D * nHII_new 
    A22 = - (KrecHII * ne_new2  +  K51_D * nHI_new  +  K52_D * nH2_new )
    A23 = K53_D * nHII_new 

    A31 = K54_D * nH2_new 
    A32 = K52_D * nH2_new 
    A33 = - (Γ_LW_HD  +  K53_D * nHII_new  +  K55_D * nHI_new)

    # Solve for the updated abundances:
    xD_old = @SVector [ xDI_old, xDII_old, xHD_old ]
    A_D    = @SMatrix [ A11  A12  A13;
                        A21  A22  A23;
                        A31  A32  A33 ]
    xD_new = (I - Δtsub * A_D) \ xD_old

    xDI_new, xDII_new, xHD_new = xD_new

    # Ensure that the fractions add up to 1 
    # (may not do so because of small numerical errors):
    Δf_D  =  xDI_new  +  xDII_new  +  xHD_new  -  1.0
    idx_D =  argmax( (xDI_new, xDII_new, xHD_new) )

    if idx_D == 1
        # In this case, D I is the biggest, so correct it:
        xDI_new   = max( xDI_new  - Δf_D, 0.0 )
    elseif idx_D == 2
        # In this case, D II is the biggest, so correct it:
        xDII_new  = max( xDII_new - Δf_D, 0.0 ) 
    else
        # In this case, HD is the biggest, so correct it:
        xHD_new   = max( xHD_new  - Δf_D, 0.0 ) 
    end 

    # Get updated electron number density:
    ne_new3 = GetElectronDensityFromFracs( xHII_new, xHeII_new, xHeIII_new, 
                                           xDII_new, xCII_old,  xCIII_old,  
                                           xOII_old, xOIII_old, xOIV_old,
                                           xSII_old, xSIII_old, xSIV_old,
                                           ρ, ZC, ZSil, Zgasrest, HydrogenAndHelium )

    # Get change in ne during the D update (for accuracy checking):
    Δne_D   = abs( ne_new3 - ne_new2 )

    ############################################################################################                   
    #########  STEP 4: UPDATE O I, O II, O III, O IV, OHₓ, CO, C I, C II, C III, CHₓ  ##########
    ############################################################################################
    # Next we update the carbon & oxygen chemistry. 

    # Compute log₁₀(T) once, for use in C/O/S chemistry LUTs:
    log10T = @fastmath log10(T)

    nTINY = @fastmath xTINY * nH
    if nC > nTINY  &&  nO > nTINY
        xOI_new, xOII_new, xOIII_new, xOIV_new, xCI_new, xCII_new, xCIII_new, xCO_new = 
        CarbonOxygenChemistry( xOI_old, xOII_old, xOIII_old, xOIV_old,
                               xCI_old, xCII_old, xCIII_old,
                               xCO_old, nC, nO, nHI_new, nHII_new,
                               nH2_new, ne_new3, ΓionH2, ΓionOI, ΓionOII, ΓionOIII,
                               ΓionCI, ΓionCII, ΓdissCO, ΓdissOHx, ΓdissCHx,
                               T, Δtsub, log10T )
    else 
        # In case there is no carbon or oxygen, just return placeholder values 
        # that allow the sub-cycling to continue 
        xOI_new   = xOI_old
        xOII_new  = xOII_old
        xOIII_new = xOIII_old
        xOIV_new  = xOIV_old
        xCI_new   = xCI_old
        xCII_new  = xCII_old
        xCIII_new = xCIII_old
        xCO_new   = xCO_old
    end

    # Get an updated electron number density:
    ne_new4    = GetElectronDensityFromFracs( xHII_new, xHeII_new, xHeIII_new, 
                                              xDII_new, xCII_new, xCIII_new, 
                                              xOII_new, xOIII_new, xOIV_new,
                                              xSII_old, xSIII_old, xSIV_old,
                                              ρ, ZC, ZSil, Zgasrest, HydrogenAndHelium )

    ######################################################################################                    
    ######################  STEP 5: UPDATE S I, S II, S III, S IV  #######################
    ######################################################################################
    # Next we update the sulfur chemistry. 

    if nS > nTINY
        xSII_new, xSIII_new, xSIV_new = SulphurChemistryUpdate( xSII_old, xSIII_old, xSIV_old, nHI_new,
                                                                nHII_new, ne_new4, ΓionSI, ΓionSII, ΓionSIII,
                                                                T, Δtsub, log10T )
    else
        # In case there is no sulfur, return placeholder values to allow subcycling 
        # to continue: 
        xSII_new, xSIII_new, xSIV_new =  xSII_old, xSIII_old, xSIV_old
    end

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------

    # Get final electron number density:
    ne_final    = GetElectronDensityFromFracs( xHII_new, xHeII_new, xHeIII_new, 
                                               xDII_new, xCII_new,  xCIII_new, 
                                               xOII_new, xOIII_new, xOIV_new,
                                               xSII_new, xSIII_new, xSIV_new,
                                               ρ, ZC, ZSil, Zgasrest, HydrogenAndHelium )

    # Get change in ne during the C/O update (for accuracy checking):
    Δne_CarbOxy = abs( ne_new4 - ne_new3 )

    # Get change in ne during the sulfur update (for accuracy checking):
    Δne_Sulfur  = abs( ne_final - ne_new4 )

    # Get change in ne since the very start:
    Δne_tot     = abs( ne_final - ne_old )

    # Now we have updated every species. ✓

    # Get maximum change in electron number density during each sub-update:
    Δne         = max( Δne_H, Δne_He, Δne_D, Δne_CarbOxy, Δne_Sulfur, Δne_tot )

    return xH2_new,  xHI_new,   xHII_new,  xHD_new, xDI_new, xDII_new,
           xHeI_new, xHeII_new, xHeIII_new, 
           xOI_new,  xOII_new,  xOIII_new, xOIV_new,
           xCI_new,  xCII_new,  xCIII_new, xCO_new, 
           xSII_new, xSIII_new, xSIV_new,
           ne_final, Δne
end

