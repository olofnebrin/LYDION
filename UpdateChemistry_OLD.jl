
""" FILE:        UpdateChemistry.jl

    PURPOSE:     Contains the basic time-update of the chemical
                 abundances of H I, H II, H₂, D I, D II, HD, He I, He II, 
                 He III. 

    REFERENCES:  The semi-implicit time-update scheme employed here is 
                 mostly based on the following references:

                 (1) Nickerson et al. (2018) 'A simple model for molecular hydrogen
                 chemistry coupled to radiation hydrodynamics', MNRAS, 479, 3, 3206-3226

                 (2) Rosdahl et al. (2013) 'RAMSES-RT: radiation hydrodynamics in the
                 cosmological context', MNRAS, 436, 3, 2188-2231

"""

using .ConstantsAndFunctions               # Needed constants
using  StaticArrays, LinearAlgebra         # For taking inverse etc

export UpdateChemistrySubStep



""" The basic BDF1 numerical solution to 

    ∂x/∂t = C - D*x ,

    where C is the effective creation term, D the 
    effective destruction term, and Δt the time-step.
    Useful in the carbon & oxygen chemistry update. 
 """ 
@inline @fastmath function x_new( x_old::Float64, 
                                      C::Float64,
                                      D::Float64, 
                                     Δt::Float64 
                                 )
    return ( x_old  +  C * Δt) / ( 1  +  D * Δt )
end


""" This function computes the residual for the BDF1 update,
    which is useful for checking convergence in case C and D 
    depend on un-updated/earlier iterative values of x. """
@inline function BDF1_residual( x_new::Float64, 
                                x_old::Float64, 
                                    C::Float64,
                                    D::Float64,
                                   Δt::Float64 )

    return abs( x_new - ( x_old + C * Δt) / ( 1 + D * Δt ) )
end


const tol_CarbOxy::Float64  = 1e-3  # Consider solution converged if relative residuals are < tol_CarbOxy
const xmin_CarbOxy::Float64 = 1e-8  # Ignore changes in abundances < xmin_CarbOxy when monitoring convergence 
const δ_CarbOxy::Float64    = 1e-8  # Abundances should add up to 1, to within a tolerance δ_CarbOxy
const max_iter_CarbOxy::Int = 10    # Maximum number of iterations (per sub-cycle step) for the carbon-oxygen update

function CarbonOxygenChemistry_OLD( xOI_old::Float64, 
                               xOII_old::Float64,
                                xCI_old::Float64,
                               xCII_old::Float64, 
                                xCO_old::Float64,
                                     nC::Float64,
                                     nO::Float64,
                                nHI_new::Float64,
                               nHII_new::Float64,
                                nH2_new::Float64,
                                 ne_new::Float64,
                                 ΓionH2::Float64,
                                 ΓionOI::Float64,
                                 ΓionCI::Float64,
                                ΓdissCO::Float64,
                               ΓdissOHx::Float64,
                               ΓdissCHx::Float64,
                                      T::Float64,
                                  Δtsub::Float64
                                )

    # Convenient conversion factor:
    C_over_O    = nC / nO

    # Get old CHₓ and OHₓ fractions:
    xCHx_old    = max( 1 - xCI_old - xCII_old - xCO_old, 0.0 )
    xOHx_old    = max( 1 - xOI_old - xOII_old - xCO_old * C_over_O, 0.0 )

    # Limit nH₃⁺ in extreme cases:
    nH          = nHI_new + nHII_new + 2 * nH2_new
    nH3plus_max = nH / 3

    # Initialize our estimates for x^(n+1) that are to be inserted
    # into the right-hand side of [x^(n+1) - x^(n)]/Δtsub = A[x^(n+1))]
    xOI_new   = xOI_old 
    xOII_new  = xOII_old 
    xOHx_new  = xOHx_old
    xCI_new   = xCI_old 
    xCII_new  = xCII_old 
    xCO_new   = xCO_old 
    xO_CO_new = xCO_new * C_over_O
    xCHx_new  = xCHx_old

    # Similarly for densities:
    nOI_new   = xOI_new  * nO 
    nOII_new  = xOII_new * nO 
    nOHx_new  = xOHx_new * nO
    nCI_new   = xCI_new  * nC 
    nCHx_new  = xCHx_new * nC

    # Also initialize variables to check convergence:
    xOI_prev  = xOI_new
    xOII_prev = xOII_new
    xOHx_prev = xOHx_new
    xCI_prev  = xCI_new
    xCII_prev = xCII_new
    xCO_prev  = xCO_new
    xCHx_prev = xCHx_new

    # Needed rates:
    KOIe      = kOIe(T)
    KHIIOI    = kHIIOI(T)
    KOIIe     = kOIIe(T)
    KOIIHI    = kOIIHI(T)
    KOIIH2_1  = kOIIH2_1(T, nH2_new, ne_new)
    KOIIH2_2  = kOIIH2_2(T, nH2_new, ne_new)
    KOIIH2    = KOIIH2_1 + KOIIH2_2
    KOHxCI    = kOHxCI(T)
    KOHxOI    = kOHxOI(T)
    KCHxOI    = kCHxOI(T)
    KCHxHI    = kCHxHI(T)
    KOIH3plus = kOIH3plus(T, nH2_new, ne_new)
    KCIH3plus = kCIH3plus(T)
    KH3pluse  = kH3pluse(T)

    KCIe      = kCIe(T)
    KCIIe     = kCIIe(T)
    KCIIH2_1  = kCIIH2_1(T)
    KCIIH2_2  = kCIIH2_2(T)
    KCIIH2    = KCIIH2_1 + KCIIH2_2

    # Determine  H₃⁺
    if ΓionH2 > 0.0 && nH2_new > 0.0
        nH3plus = min( ΓionH2 * nH2_new / (KCIH3plus * nCI_new  +  KOIH3plus * nOI_new  +  KH3pluse * ne_new), 
                       nH3plus_max )
    else
        nH3plus = 0.0
    end

    iter        = 0     # Initialize iterations
    Converged   = false # Initialize convergence check 
    while iter < max_iter_CarbOxy
        # Update O I:
        xO_CO_new =    xCO_new * C_over_O
        C_OI      =  ( (KOIIe * ne_new  +  KOIIHI * nHI_new  +  KOIIH2_2 * nH2_new) * xOII_new
                     + KOHxOI * nOHx_new * xOI_new  # From OHₓ + O → O + O + H
                     + ΓdissCO * xO_CO_new  )
        D_OI      =    ΓionOI  +  KOIe * ne_new  +  KHIIOI * nHII_new  +  KCHxOI * nCHx_new  +  KOIH3plus * nH3plus
        xOI_new   =    x_new( xOI_old, C_OI, D_OI, Δtsub )
        nOI_new   =    xOI_new * nO

        # Update O II:
        C_OII     =  ( ΓionOI  +  KOIe * ne_new  +  KHIIOI * nHII_new ) * xOI_new
        D_OII     =    KOIIe * ne_new  +  KOIIHI * nHI_new  +  KOIIH2 * nH2_new 
        xOII_new  =    x_new( xOII_old, C_OII, D_OII, Δtsub )
        nOII_new  =    xOII_new * nO

        # Determine new H₃⁺ (using updated O I):
        if ΓionH2 > 0.0 && nH2_new > 0.0
            nH3plus = min( ΓionH2 * nH2_new / (KCIH3plus * nCI_new  +  KOIH3plus * nOI_new  +  KH3pluse * ne_new), 
                           nH3plus_max )
        else
            nH3plus = 0.0
        end

        # Update OHₓ:
        C_OHx     =    KOIH3plus * nH3plus * xOI_new  +  KOIIH2_1 * nH2_new * xOII_new
        D_OHx     =    KOHxCI * nCI_new  +  KOHxOI * nOI_new  +  ΓdissOHx
        xOHx_new  =    x_new( xOHx_old, C_OHx, D_OHx, Δtsub )
        nOHx_new  =    xOHx_new * nO

        # Update C I:
        C_CI      =  ( ( KCIIe * ne_new  +  KCIIH2_2 * nH2_new ) * xCII_new
                       + KCHxHI * nHI_new * xCHx_new   # From CHₓ + H → C + H₂
                       + ΓdissCO  * xCO_new
                       + ΓdissCHx * xCHx_new )
        D_CI      =      ΓionCI  +  KCIe * ne_new  +  KOHxCI * nOHx_new  +  KCIH3plus * nH3plus
        xCI_new   =      x_new( xCI_old, C_CI, D_CI, Δtsub )
        nCI_new   =      xCI_new * nC

        # Determine new H₃⁺ (using updated C I):
        if ΓionH2 > 0.0 && nH2_new > 0.0
            nH3plus = min( ΓionH2 * nH2_new / (KCIH3plus * nCI_new  +  KOIH3plus * nOI_new  +  KH3pluse * ne_new), 
                           nH3plus_max )
        else
            nH3plus = 0.0
        end

        # Update C II:
        C_CII     =  ( ΓionCI  +  KCIe * ne_new ) * xCI_new
        D_CII     =    KCIIe * ne_new  +  KCIIH2 * nH2_new  
        xCII_new  =    x_new( xCII_old, C_CII, D_CII, Δtsub )

        # Update CHₓ:
        C_CHx     =    KCIH3plus * nH3plus * xCI_new  +  KCIIH2_1 * nH2_new * xCII_new
        D_CHx     =    KCHxOI * nOI_new  +  KCHxHI * nHI_new  +  ΓdissCHx
        xCHx_new  =    x_new( xCHx_old, C_CHx, D_CHx, Δtsub )
        nCHx_new  =    xCHx_new * nC

        # Update CO:
        C_CO      =    KCHxOI * nOI_new * xCHx_new  +  KOHxCI * nOHx_new * xCI_new
        D_CO      =    ΓdissCO
        xCO_new   =    x_new( xCO_old, C_CO, D_CO, Δtsub )

        # Enforce conservation of carbon species:
        Δf_C      =    (xCI_new  +  xCII_new  +  xCHx_new  +  xCO_new) - 1.0
        idx_C     =    argmax( ( xCI_new, xCII_new, xCHx_new, xCO_new ) )
     
        if idx_C == 1
            # In this case, C I is the biggest, so correct it:
            xCI_new   = max( xCI_new - Δf_C, 0.0 )
        elseif idx_C == 2 
            # In this case, C II is the biggest, so correct it:
            xCII_new  = max( xCII_new - Δf_C, 0.0 ) 
        elseif idx_C == 3
            # In this case, CHₓ is the biggest, so correct it:
            xCHx_new  = max( xCHx_new - Δf_C, 0.0 ) 
        else
            # In this case CO is the biggest, so correct it:
            xCO_new  = max( xCO_new - Δf_C, 0.0 ) 
        end 

        # Enforce conservation of oxygen species:
        xO_CO_new =    xCO_new * C_over_O
        Δf_O      =   (xOI_new  +  xOII_new  +  xOHx_new  +  xO_CO_new) - 1.0
        idx_O     =    argmax( ( xOI_new, xOII_new, xOHx_new ) )
        
        if idx_O == 1
            # In this case, O I is the biggest, so correct it:
            xOI_new   = max( xOI_new - Δf_O, 0.0 )
        elseif idx_O == 2 
            # In this case, O II is the biggest, so correct both:
            xOII_new  = max( xOII_new - Δf_O, 0.0 ) 
        elseif idx_O == 3
            # In this case, OHₓ is the biggest, so correct it:
            xOHx_new  = max( xOHx_new - Δf_O, 0.0)
        end 

        
        xO_CO_new = xCO_new * C_over_O
        nOI_new   = xOI_new  * nO
        nOII_new  = xOII_new * nO
        nOHx_new  = xOHx_new * nO
        nCI_new   = xCI_new  * nC
        nCHx_new  = xCHx_new * nC

        # Total oxygen abundance after correction (should add up to 1):
        xOtot =  xOI_new  +  xOII_new  +  xOHx_new  +  xO_CO_new

        # Total carbon abundance (should add up to 1):
        xCtot =  xCI_new  +  xCII_new  +  xCHx_new  +  xCO_new 

        if ΓionH2 > 0.0 && nH2_new > 0.0
            nH3plus = min( ΓionH2 * nH2_new / (KCIH3plus * nCI_new  +  KOIH3plus * nOI_new  +  KH3pluse * ne_new), 
                           nH3plus_max )
        else
            nH3plus = 0.0
        end

        # Residual for O I:
        C_OI      =  ( (KOIIe * ne_new  +  KOIIHI * nHI_new  +  KOIIH2_2 * nH2_new) * xOII_new
                     + KOHxOI * nOHx_new * xOI_new  # From OHₓ + O → O + O + H
                     + ΓdissCO * xO_CO_new  )
        D_OI      =    ΓionOI  +  KOIe * ne_new  +  KHIIOI * nHII_new  +  KCHxOI * nCHx_new  +  KOIH3plus * nH3plus
        Res_OI    =    BDF1_residual( xOI_new, xOI_old, C_OI, D_OI, Δtsub ) / max( xOI_old, xOI_new, xmin_CarbOxy )

        # Residual for O II:
        C_OII     =  ( ΓionOI  +  KOIe * ne_new  +  KHIIOI * nHII_new ) * xOI_new
        D_OII     =    KOIIe * ne_new  +  KOIIHI * nHI_new  +  KOIIH2 * nH2_new 
        Res_OII   =    BDF1_residual( xOII_new, xOII_old, C_OII, D_OII, Δtsub ) / max( xOII_old, xOII_new, xmin_CarbOxy )

        # Residual for OHₓ:
        C_OHx     =    KOIH3plus * nH3plus * xOI_new  +  KOIIH2_1 * nH2_new * xOII_new
        D_OHx     =    KOHxCI * nCI_new  +  KOHxOI * nOI_new  +  ΓdissOHx
        Res_OHₓ   =    BDF1_residual( xOHx_new, xOHx_old, C_OHx, D_OHx, Δtsub ) / max( xOHx_old, xOHx_new, xmin_CarbOxy )

        # Residual for C I:
        C_CI      =  ( ( KCIIe * ne_new  +  KCIIH2_2 * nH2_new ) * xCII_new
                       + KCHxHI * nHI_new * xCHx_new   # From CHₓ + H → C + H₂
                       + ΓdissCO  * xCO_new
                       + ΓdissCHx * xCHx_new )
        D_CI      =      ΓionCI  +  KCIe * ne_new  +  KOHxCI * nOHx_new  +  KCIH3plus * nH3plus 
        Res_CI    =    BDF1_residual( xCI_new, xCI_old, C_CI, D_CI, Δtsub ) / max( xCI_old, xCI_new, xmin_CarbOxy )

        # Residual for C II:
        C_CII     =  ( ΓionCI  +  KCIe * ne_new ) * xCI_new
        D_CII     =    KCIIe * ne_new  +  KCIIH2 * nH2_new  
        Res_CII   =    BDF1_residual( xCII_new, xCII_old, C_CII, D_CII, Δtsub ) / max( xCII_old, xCII_new, xmin_CarbOxy )

        # Residual for CHₓ:
        C_CHx     =    KCIH3plus * nH3plus * xCI_new  +  KCIIH2_1 * nH2_new * xCII_new
        D_CHx     =    KCHxOI * nOI_new  +  KCHxHI * nHI_new  +  ΓdissCHx
        Res_CHₓ   =    BDF1_residual( xCHx_new, xCHx_old, C_CHx, D_CHx, Δtsub ) / max( xCHx_old, xCHx_new, xmin_CarbOxy )

        # Residual for CO:
        C_CO      =    KCHxOI * nOI_new * xCHx_new  +  KOHxCI * nOHx_new * xCI_new
        D_CO      =    ΓdissCO
        Res_CO    =    BDF1_residual( xCO_new, xCO_old, C_CO, D_CO, Δtsub ) / max( xCO_old, xCO_new, xmin_CarbOxy )

        # Get final maximum residual for BDF1:
        ΔRes      =    maximum((Res_OI, Res_OII, Res_OHₓ, Res_CI, Res_CII, Res_CHₓ, Res_CO))

        # Also track changes between iterations:
        Δx_iter = maximum(( abs( xOI_new  - xOI_prev )  / max( xOI_new,  xOI_prev,  xmin_CarbOxy ),
                            abs( xOII_new - xOII_prev ) / max( xOII_new, xOII_prev, xmin_CarbOxy ),
                            abs( xOHx_new - xOHx_prev ) / max( xOHx_new, xOHx_prev, xmin_CarbOxy ),
                            abs( xCI_new  - xCI_prev )  / max( xCI_new,  xCI_prev,  xmin_CarbOxy ),
                            abs( xCII_new - xCII_prev ) / max( xCII_new, xCII_prev, xmin_CarbOxy ),
                            abs( xCHx_new - xCHx_prev ) / max( xCHx_new, xCHx_prev, xmin_CarbOxy ),
                            abs( xCO_new  - xCO_prev )  / max( xCO_new,  xCO_prev,  xmin_CarbOxy )
                           ))

        if (ΔRes < tol_CarbOxy || Δx_iter < tol_CarbOxy) && abs(xOtot - 1.0) < δ_CarbOxy && abs(xCtot - 1.0) < δ_CarbOxy
            Converged = true
            break 
        end

        # Update iteration number & convergence check variables:
        iter     += 1
        xOI_prev  = xOI_new
        xOII_prev = xOII_new
        xOHx_prev = xOHx_new
        xCI_prev  = xCI_new
        xCII_prev = xCII_new
        xCO_prev  = xCO_new
        xCHx_prev = xCHx_new

    end # End of while loop

    return xOI_new, xOII_new, xCI_new, xCII_new, xCO_new, Converged
end


const ϵneg::Float64 = 1e-12   # If any abundance has x < - ϵneg (default: ϵneg = 10⁻¹²), reject the update. If not, clamp to positive.

function CarbonOxygenChemistry( xOI_old::Float64, 
                               xOII_old::Float64,
                                xCI_old::Float64,
                               xCII_old::Float64, 
                                xCO_old::Float64,
                                     nC::Float64,
                                     nO::Float64,
                                nHI_new::Float64,
                               nHII_new::Float64,
                                nH2_new::Float64,
                                 ne_new::Float64,
                                 ΓionH2::Float64,
                                 ΓionOI::Float64,
                                 ΓionCI::Float64,
                                ΓdissCO::Float64,
                               ΓdissOHx::Float64,
                               ΓdissCHx::Float64,
                                      T::Float64,
                                  Δtsub::Float64 
                                )

    # Convenient conversion factor:
    C_over_O    = nC / nO

    # Get old CHₓ fraction (for nCHx_old):
    xCHx_old    = max( 1 - xCI_old - xCII_old - xCO_old, 0.0 )
    xOHx_old    = max( 1 - xOI_old - xOII_old - C_over_O * xCO_old, 0.0 )

    # Densities:
    nCI_old     = xCI_old  * nC         
    nCHx_old    = xCHx_old * nC         
    nOI_old     = xOI_old  * nO       

    # Limit nH₃⁺ in extreme cases:
    nH          = nHI_new + nHII_new + 2 * nH2_new
    nH3plus_max = nH / 3

    # Needed rates:
    KOIe      = kOIe(T)
    KHIIOI    = kHIIOI(T)
    KOIIe     = kOIIe(T)
    KOIIHI    = kOIIHI(T)
    KOIIH2_1  = kOIIH2_1(T, nH2_new, ne_new)
    KOIIH2_2  = kOIIH2_2(T, nH2_new, ne_new)
    KOIIH2    = KOIIH2_1 + KOIIH2_2
    KOHxCI    = kOHxCI(T)
    KOHxOI    = kOHxOI(T)
    KCHxOI    = kCHxOI(T)
    KCHxHI    = kCHxHI(T)
    KOIH3plus = kOIH3plus(T, nH2_new, ne_new)
    KCIH3plus = kCIH3plus(T)
    KH3pluse  = kH3pluse(T)

    KCIe      = kCIe(T)
    KCIIe     = kCIIe(T)
    KCIIH2_1  = kCIIH2_1(T)
    KCIIH2_2  = kCIIH2_2(T)
    KCIIH2    = KCIIH2_1 + KCIIH2_2

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
    # including CO. We update O I, O II, CO, and eliminate OHₓ by
    # conservation: xOHₓ = 1 - xOI - xOII - (nC/nO) * xCO.

    C_OII_to_OI =  KOIIe * ne_new  +  KOIIHI * nHI_new  +  KOIIH2_2 * nH2_new 
    C_OHx_to_OI =  ΓdissOHx  +  KOHxOI * nO * xOI_old
    D_OI        =  ( ΓionOI  +  KOIe * ne_new  +  KHIIOI * nHII_new  +  KCHxOI * nCHx_old
                +    KOIH3plus * nH3plus )

    C_OI_to_OII =  ΓionOI  +  KOIe * ne_new  +  KHIIOI * nHII_new 
    D_OII       =  KOIIe * ne_new  +  KOIIHI * nHI_new  +  KOIIH2 * nH2_new  

    C_OI_to_CO  =  KCHxOI * nO * xCHx_old
    C_OHx_to_CO =  KOHxCI * nO * xCI_old 

    # Set up linear equations & solve:
    A_O         =  @SVector [ C_OHx_to_OI, 0.0, C_OHx_to_CO ]
    B_O_11      =  - ( D_OI  +  C_OHx_to_OI )
    B_O_12      =  C_OII_to_OI - C_OHx_to_OI
    B_O_13      =  C_over_O * ( ΓdissCO  -  C_OHx_to_OI )
    B_O_21      =  C_OI_to_OII
    B_O_22      =  - D_OII
    B_O_23      =  0.0
    B_O_31      =  C_OI_to_CO - C_OHx_to_CO
    B_O_32      =  - C_OHx_to_CO
    B_O_33      =  - ( ΓdissCO  +  C_over_O * C_OHx_to_CO )
    B_O         =  @SMatrix [ B_O_11   B_O_12  B_O_13;
                              B_O_21   B_O_22  B_O_23;
                              B_O_31   B_O_32  B_O_33 ]

    xO_old  =  @SVector [ xOI_old, xOII_old, xCO_old ]
    xO_new  = (I - Δtsub * B_O) \ ( xO_old  +  A_O * Δtsub ) 

    # Get predictor O I, O II, CO fractions:
    xOI_star, xOII_star, xCO_star = xO_new
    
    # Get OHₓ from conservation:
    xOHx_star = min( max( 1 - xOI_star - xOII_star - C_over_O * xCO_star, 0.0 ), 1.0 )
    nOHx_star = xOHx_star * nO 

    # Get nOI_new:
    nOI_star  = xOI_star * nO

    # Update H₃⁺:
    if ΓionH2 > 0.0 && nH2_new > 0.0
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

    # Next we do a corrector update of the carbon species C I, C II, 
    # and CO (again), using the predictor values for O-bearing species 
    # (except CO).

    C_CII_to_CI = KCIIe * ne_new  +  KCIIH2_2 * nH2_new 
    C_CHx_to_CI = KCHxHI * nHI_new  +  ΓdissCHx
    D_CI        = ΓionCI  +  KCIe * ne_new  +  KOHxCI * nOHx_star  +  KCIH3plus * nH3plus_star
    
    C_CI_to_CII = ΓionCI  +  KCIe * ne_new 
    D_CII       = KCIIe * ne_new  +  KCIIH2 * nH2_new

    C_CI_to_CO  = KOHxCI * nOHx_star
    C_CHx_to_CO = KCHxOI * nOI_star

    # Set up linear equations & solve:
    A_C         =  @SVector [ C_CHx_to_CI, 0.0, C_CHx_to_CO ]
    B_C_11      =  - ( D_CI + C_CHx_to_CI )
    B_C_12      =  C_CII_to_CI - C_CHx_to_CI
    B_C_13      =  ΓdissCO - C_CHx_to_CI
    B_C_21      =  C_CI_to_CII 
    B_C_22      =  - D_CII
    B_C_23      =  0.0
    B_C_31      =  C_CI_to_CO - C_CHx_to_CO
    B_C_32      =  - C_CHx_to_CO
    B_C_33      =  - ( ΓdissCO  +  C_CHx_to_CO )
    B_C         =  @SMatrix [ B_C_11   B_C_12  B_C_13;
                              B_C_21   B_C_22  B_C_23;
                              B_C_31   B_C_32  B_C_33 ]

    xC_old      =  @SVector [ xCI_old, xCII_old, xCO_old ]
    xC_new      = (I - Δtsub * B_C) \ ( xC_old  +  A_C * Δtsub ) 

    # Get corrector/final C I, C II, CO fractions:
    xCI_new, xCII_new, xCO_new = xC_new 

    # Renormalize oxygen species to make it consistent with corrector update for CO:
    f_renorm = (1 - C_over_O * xCO_new) / (1 - C_over_O * xCO_star)
    xOI_new  = f_renorm * xOI_star
    xOII_new = f_renorm * xOII_star

    # Get new values of OHₓ and CHₓ:
    xOHx_new = max( 1 - xOI_new - xOII_new - C_over_O * xCO_new, 0.0 )
    xCHx_new = max( 1 - xCI_new - xCII_new - xCO_new, 0.0 )

    # Check accuracy/convergence:
    ΔOI        = abs( xOI_new   - xOI_old )  / max( xOI_old,  xmin )
    ΔOII       = abs( xOII_new  - xOII_old ) / max( xOII_old, xmin )
    ΔOHx       = abs( xOHx_new  - xOHx_old ) / max( xOHx_old,  xmin )
    ΔCI        = abs( xCI_new   - xCI_old )  / max( xCI_old,  xmin )
    ΔCII       = abs( xCII_new  - xCII_old ) / max( xCII_old, xmin )
    ΔCHx       = abs( xCHx_new  - xCHx_old ) / max( xCHx_old,  xmin )
    ΔCO        = abs( xCO_new   - xCO_old )  / max( xCO_old,  xmin )
    ΔCarbOxy   = maximum( (ΔOI, ΔOII, ΔOHx, ΔCI, ΔCII, ΔCHx, ΔCO) )

    Positivity = ( xOI_new ≥ - ϵneg && xOII_new ≥ - ϵneg && xCI_new ≥ - ϵneg &&
                   xCII_new ≥ - ϵneg && xCO_new ≥ - ϵneg )

    if ΔCarbOxy < tol_Chem && Positivity
        # If all updated abundances changed by < 10% and have positive or at most 
        # tiny negative values (x ≥ - ϵneg) we accept the update, and clamp away the 
        # tiny negative values:
        xOI_new   = max( xOI_new,  0.0 )
        xOII_new  = max( xOII_new, 0.0 )
        xCI_new   = max( xCI_new,  0.0 )
        xCII_new  = max( xCII_new, 0.0 )
        xCO_new   = max( xCO_new,  0.0 )

        Converged = true 
    else 
        Converged = false 
    end

    return xOI_new, xOII_new, xCI_new, xCII_new, xCO_new, Converged
end


""" This function computes updated abundances of 
    the species H₂, H I, H II, HD, D I, D II, He I, He II, He III, e⁻,
    over a sub-cycle time-step Δtsub. Uses method of refs. (1)-(2).

    (Note that not all species enters the arguments below, because 
    we can get the H I, D I, He I, and e⁻ abundances from particle and 
    charge conservation.) """
function UpdateChemistrySubStep( Δtsub::Float64,
                                 Tnorm::Float64,
                                     ρ::Float64,
                                    DG::Float64,
                               xH2_old::Float64,
                              xHII_old::Float64, 
                               xHD_old::Float64,
                              xDII_old::Float64,
                             xHeII_old::Float64,
                            xHeIII_old::Float64,
                               xOI_old::Float64, 
                              xOII_old::Float64,
                               xCI_old::Float64,
                              xCII_old::Float64, 
                               xCO_old::Float64,
                             Zgas_Zsun::Float64,
                                ΓionHI::Float64 = 0.0,
                                ΓionH2::Float64 = 0.0,
                                ΓionDI::Float64 = 0.0,
                               ΓionHeI::Float64 = 0.0,
                              ΓionHeII::Float64 = 0.0,
                                  Γ_LW::Float64 = 0.0,
                                   Γpd::Float64 = 0.0,
                               Γ_LW_HD::Float64 = 0.0,
                                ΓionOI::Float64 = 0.0,
                                ΓionCI::Float64 = 0.0,
                               ΓdissCO::Float64 = 0.0,
                              ΓdissOHx::Float64 = 0.0,
                              ΓdissCHx::Float64 = 0.0,
                                     z::Float64 = 0.0,
                                 CaseB::Bool = true
                                 )
    # Get initial abundance fractions and needed densities:
    xHI_old    = max( 1 - xHII_old - 2 * xH2_old, 0.0 )
    xDI_old    = max( 1 - xDII_old  - xHD_old,    0.0 )
    xHeI_old   = max( 1 - xHeII_old - xHeIII_old, 0.0 )

    nH         = X_over_mH  * ρ  
    nHe        = Y_over_mHe * ρ 
    nC         = Zgas_Zsun  * C_H_ratio * nH  
    nO         = Zgas_Zsun  * O_H_ratio * nH  
    nHI_old    = xHI_old    * nH
    nHII_old   = xHII_old   * nH 
    nH2_old    = xH2_old    * nH 
    nHeII_old  = xHeII_old  * nHe 
    nHeIII_old = xHeIII_old * nHe 
    ne_old     = GetElectronDensity( nHII_old, nHeII_old, nHeIII_old )

    # Get temperature by computing μ and then using 
    # T = μ × (γad - 1) × Tnorm (where Tnorm ≡ T/[μ(γad - 1)]) :
    μ   = Getμ( xHII_old, xH2_old, xHeII_old, xHeIII_old )
    γad = Getγad( xHII_old, xH2_old, xHeII_old, xHeIII_old )
    T   = μ * (γad - 1) * Tnorm

    ###################################################                      
    ######      STEP 1: UPDATE H I, H II, H₂     ######
    ###################################################

    # The following rates appear several times, so 
    # compute them once for convenience/optimization:
    Kdust_H2    = kdust_H2(T, DG)
    KGP_H2      = kGP_H2(T, ne_old, nHI_old, nHII_old, Γpd, z)
    K3B_H2      = k3B_H2(T)
    Kdiss_H2HI  = kdiss_H2HI(T, nHI_old)
    Kdiss_H2H2  = kdiss_H2H2(T, nH2_old)
    Kdiss_H2e   = kdiss_H2e(T)
    Kdiss_H2HII = kdiss_H2HII(T)
    KrecHII     = krecHII(T, CaseB)
    Kcollion_HI = kcollion_HI(T)

    # Creation rate for H₂ is C_H2_0 × xHI, where: 
    C_H2_0  = ( Kdust_H2 * nH 
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
        xH2_new  = max( xH2_new - Δf_H / 2, 0.0 )
    elseif idx_H == 2
        # In this case, H I is the biggest, so correct it:
        xHI_new  = max( xHI_new - Δf_H, 0.0 )
    else
        # In this case, H II is the biggest, so correct it:
        xHII_new = max( xHII_new - Δf_H, 0.0 )
    end

    # We also update the temperature (for use in helium
    # chemistry step) using the known value of Tnorm 
    # and the new μ and γ (computed below):
    μ      = Getμ( xHII_new, xH2_new, xHeII_old, xHeIII_old )
    γad    = Getγad( xHII_new, xH2_new, xHeII_old, xHeIII_old ) 
    T      = μ * (γad - 1) * Tnorm 

    # Update the electron number density:
    ne_new = GetElectronDensityFromFracs( xHII_new, xHeII_old, xHeIII_old, ρ )

    # Get change in ne during the H update (for accuracy checking):
    Δne_H  = abs( ne_new - ne_old )

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
    C_HeII_0 =  Kcollion_HeI * ne_new  +  ΓionHeI 
    C_HeII_1 =  KrecHeIII * ne_new 

    # Destruction rate for He II (from coll. ionization, recombinations, photoionization) 
    # is:
    #
    # (ẋ_HeII)_(-) = - D_HeII × xHeII ,
    #
    # where:
    D_HeII   =  ( KrecHeII + Kcollion_HeII ) * ne_new  +  ΓionHeII  

    # Creation rate for He III (coll. ionization & photoionization of He II) is:
    #
    # (ẋ_HeIII)_(+) = C_HeIII × xHeII ,
    #
    # where:
    C_HeIII  =  Kcollion_HeII * ne_new  +  ΓionHeII  

    # Destruction rate for He III (from recombinations) is:
    #
    # (ẋ_HeIII)_(-) = - D_HeIII × xHeIII ,
    #
    # where:
    D_HeIII  =  KrecHeIII * ne_new

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

    # Get a final update of the electron number density:
    ne_final    = GetElectronDensityFromFracs( xHII_new, xHeII_new, xHeIII_new, ρ )

    # Get change in ne during the He update (for accuracy checking):
    Δne_He      = abs( ne_final - ne_new )

    # Get change in ne since the very start:
    Δne_tot     = abs( ne_final - ne_old )

    # We also update the temperature (for use in deuterium
    # chemistry step) using the known value of Tnorm 
    # and the new μ and γ (computed below):
    μ      = Getμ( xHII_new, xH2_new, xHeII_new, xHeIII_new )
    γad    = Getγad( xHII_new, xH2_new, xHeII_new, xHeIII_new ) 
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
    Kcollion_HI = kcollion_HI(T)    # D + e⁻  → D⁺ + e⁻ (same as for H I)
    KrecHII     = krecHII(T, CaseB) # D⁺ + e⁻ → D + hν  (same as for H I)
    K50_D       = k50_D(T)          # D + H⁺  → D⁺ + H
    K51_D       = k51_D(T)          # D⁺ + H  → D + H⁺
    K52_D       = k52_D(T)          # D⁺ + H₂ → H⁺ + HD
    K53_D       = k53_D(T)          # H⁺ + HD → D⁺ + H₂
    K54_D       = k54_D(T)          # D + H₂  → H + HD
    K55_D       = k55_D(T)          # H + HD  → D + H₂

    # Matrix elements:
    A11 = - (ΓionDI  +  Kcollion_HI * ne_final  +  K50_D * nHII_new  +  K54_D * nH2_new)
    A12 = KrecHII * ne_final  +  K51_D * nHI_new 
    A13 = K55_D * nHI_new  +  Γ_LW_HD

    A21 = ΓionDI  +  Kcollion_HI * ne_final  +  K50_D * nHII_new 
    A22 = - (KrecHII * ne_final  +  K51_D * nHI_new  +  K52_D * nH2_new )
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
        xDI_new   = max( xDI_new - Δf_D, 0.0 )
    elseif idx_D == 2
        # In this case, D II is the biggest, so correct it:
        xDII_new  = max( xDII_new - Δf_D, 0.0 ) 
    else
        # In this case, HD is the biggest, so correct it:
        xHD_new   = max( xHD_new - Δf_D, 0.0 ) 
    end 

    ########################################################################                      
    ######  STEP 4: UPDATE O I, O II, O III, OHₓ, CO, C I, C II, CHₓ  ######
    ########################################################################
    # Next we update the carbon & oxygen chemistry. 

    if nC > 0.0 && nO > 0.0
        xOI_new, xOII_new, xCI_new, xCII_new, xCO_new, ConvergedCarbonOxygen = 
        CarbonOxygenChemistry( xOI_old, xOII_old, xCI_old, xCII_old,
                               xCO_old, nC, nO, nHI_new, nHII_new,
                               nH2_new, ne_new, ΓionH2, ΓionOI, ΓionCI,
                               ΓdissCO, ΓdissOHx, ΓdissCHx,
                               T, Δtsub )
    else 
        # In case there is no carbon or oxygen, just return placeholder values 
        # that allow the sub-cycling to continue 
        xOI_new  = xOI_old
        xOII_new = xOII_old
        xCI_new  = xCI_old
        xCII_new = xCII_old
        xCO_new  = xCO_old
        ConvergedCarbonOxygen = true
    end

    # Now we have updated every species. ✓

    # Get maximum change in electron number density during each sub-update:
    Δne = max( Δne_H, Δne_He, Δne_tot )

    return xH2_new, xHI_new, xHII_new, xHD_new, xDI_new, xDII_new,
           xHeI_new, xHeII_new, xHeIII_new, 
           xOI_new, xOII_new, xCI_new, xCII_new, xCO_new, ConvergedCarbonOxygen,
           ne_final, Δne
end

