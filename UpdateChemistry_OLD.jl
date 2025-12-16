
module UpdateChemistry
""" MODULE:      UpdateChemistry

    PURPOSE:     Contains the basic time-update of the chemical
                 abundances of H I, H II, H₂, D I, D II, HD, He I, He II, 
                 He III, C I, C II, CO, O I, O II, and e⁻. 

    REFERENCES:  The semi-implicit time-update scheme employed here is 
                 mostly based on the following references:

                 (1) Nickerson et al. (2018) 'A simple model for molecular hydrogen
                 chemistry coupled to radiation hydrodynamics', MNRAS, 479, 3, 3206-3226

                 (2) Rosdahl et al. (2013) 'RAMSES-RT: radiation hydrodynamics in the
                 cosmological context', MNRAS, 436, 3, 2188-2231

"""

__precompile__()


include(joinpath(@__DIR__, "..", "ConstantsAndFunctions.jl"))
# include( joinpath(@__DIR__, "..", "Dust", "DustProperties.jl") )
include(joinpath(@__DIR__, "..", "Hydrodynamics.jl"))
include( "./GasChemistryRates.jl" )
include( "./UsefulPhotoThermoChemistryFunctions.jl" )
using .ConstantsAndFunctions               # Needed constants
# using .DustProperties
using .GasChemistryRates                   # Needed rates
using .Hydrodynamics                       # Useful functions to compute μ and γad 
using .UsefulPhotoThermoChemistryFunctions # Useful functions 
using  StaticArrays, LinearAlgebra         # For taking inverse etc (needed for deuterium chemsitry)

export UpdateChemistrySubStep



""" The basic BDF1 numerical solution to 

    ∂x/∂t = C - D*x ,

    where C is the effective creation term, D the 
    effective destruction term, and Δt the time-step.
    Useful for the chemistry time-stepping. 
 """ 
@inline @fastmath function x_new( x_old::Float64, 
                                      C::Float64,
                                      D::Float64, 
                                     Δt::Float64 
                                 )
    return ( x_old  +  C * Δt) / (1  +  D * Δt )
end



""" This function updates the abundances of 
    the species H₂, H I, H II, HD, D I, D II, He I, He II, He III, e⁻,
    over a sub-cycle time-step Δtsub. Uses method of refs. (1)-(2).

    (Note that not all species enters the arguments below, because 
    we can get the H I, D I, He I, and e⁻ abundances from particle and 
    charge conservation.)
    
    **ARGUMENTS:** 
    
    - Δtsub       :   Sub-cycle time-step (sec)
    - Tnorm       :   T/[μ x (γad - 1)], normalized gas temperature (K)
    - ρ           :   Gas density (g cm⁻³) 
    - xH2_old     :   H₂ fraction at previous (sub-)time-step 
    - xHII_old    :   H II fraction at previous (sub-)time-step
    - xHD_old     :   HD fraction at previous (sub-)time-step 
    - xDII_old    :   D II fraction at previous (sub-)time-step
    - xHeII_old   :   He II fraction at previous (sub-)time-step 
    - xHeIII_old  :   He III fraction at previous (sub-)time-step  
    - ΓionHI      :   H I photoionization rate (s⁻¹)              (default = 0)
    - ΓionH2      :   H₂ photoionization rate (s⁻¹)               (default = 0)
    - ΓionDI      :   D I photoionization rate (s⁻¹)              (default = 0)
    - ΓionHeI     :   He I photoionization rate (s⁻¹)             (default = 0)
    - ΓionHeII    :   He II photoionization rate (s⁻¹)            (default = 0)
    - Γ_LW        :   H₂ Lyman-Werner dissociation rate (s⁻¹)     (default = 0)
    - Γpd         :   H⁻ photodetachment rate (s⁻¹)               (default = 0)
    - Γ_LW_HD     :   HD Lyman-Werner dissociation rate (s⁻¹)     (default = 0)
    - D_Dsun      :   Dust-to-gas ratio in Solar units            (default = 1)
    - Tdust       :   Dust temperature (K)                        (default = 10 K)
    - z           :   Redshift                                    (default = 0)
    - CaseB       :   Case B recombination if true, Case A if not (default = true)
    
    **RETURNS:**
    
    Updated fractions & e⁻ density: xH2_new, xHI_new, xHII_new, xHD_new, xDI_new, xDII_new,
                                    xHeI_new, xHeII_new, xHeIII_new, ne_new """
function UpdateChemistrySubStep( Δtsub::Float64,
                                 Tnorm::Float64,
                                     ρ::Float64,
                               xH2_old::Float64,
                              xHII_old::Float64, 
                               xHD_old::Float64,
                              xDII_old::Float64,
                             xHeII_old::Float64,
                            xHeIII_old::Float64,
                                ΓionHI::Float64 = 0.0,
                                ΓionH2::Float64 = 0.0,
                                ΓionDI::Float64 = 0.0,
                               ΓionHeI::Float64 = 0.0,
                              ΓionHeII::Float64 = 0.0,
                                  Γ_LW::Float64 = 0.0,
                                   Γpd::Float64 = 0.0,
                               Γ_LW_HD::Float64 = 0.0,
                                     z::Float64 = 0.0,
                                 CaseB::Bool = true
                                 )
    # Get initial abundance fractions and needed densities:
    xHI_old    = max( 1 - xHII_old - 2 * xH2_old, 0.0 )
    xDI_old    = max( 1 - xDII_old  - xHD_old,    0.0 )
    xHeI_old   = max( 1 - xHeII_old - xHeIII_old, 0.0 )

    nH         = X * ρ / mH 
    nHe        = Y * ρ / mHe
    nC         = C_H_ratio  * nH 
    nO         = O_H_ratio  * nH
    nHI_old    = xHI_old    * nH
    nHII_old   = xHII_old   * nH 
    nH2_old    = xH2_old    * nH 
    nHeII_old  = xHeII_old  * nHe 
    nHeIII_old = xHeIII_old * nHe 
    ne_old     = nHII_old  +  nHeII_old  +  2 * nHeIII_old 

    # Get temperature by computing μ and then using 
    # T = μ × (γad - 1) × Tnorm (where Tnorm ≡ T/[μ(γad - 1)]) :
    μ   = Getμ(xHII_old, xH2_old, xHeII_old, xHeIII_old)
    γad = Getγad(xHII_old, xH2_old, xHeII_old, xHeIII_old)
    T   = μ * (γad - 1) * Tnorm

    ##########################################                       
    ######       STEP 1: UPDATE H₂      ######
    ##########################################
    # This is a putative update of H₂, meant 
    # for getting a better H₂-estimate in the 
    # updates of H I and H II.

    # The following rates appear several times, so 
    # compute them once for convenience/optimization:
    # TODO: Kdust_H2    = kdust_H2(T, ρd_Sil, ρd_C, T_dust_bins_Sil, T_dust_bins_C, agr_bins, ρgr_Sil, ρgr_C)
    Kdust_H2    = 0.0
    KGP_H2      = kGP_H2(T, ne_old, nHI_old, nHII_old, Γpd, z)
    K3B_H2      = k3B_H2(T)
    Kdiss_H2HI  = kdiss_H2HI(T, nHI_old)
    Kdiss_H2H2  = kdiss_H2H2(T, nH2_old)
    Kdiss_H2e   = kdiss_H2e(T)
    Kdiss_H2HII = kdiss_H2HII(T)

    # H₂ creation term (using un-updated values):
    C_H2     = ( Kdust_H2 * nH 
               + KGP_H2 * ne_old 
               + K3B_H2 * nHI_old * (nHI_old + nH2_old / 8)  ) * xHI_old

    # H₂ destruction term (using un-updated values):
    D_H2     = ( Kdiss_H2HI  * nHI_old 
               + Kdiss_H2H2  * nH2_old
               + Kdiss_H2e   * ne_old
               + Kdiss_H2HII * nHII_old
               + Γ_LW + ΓionH2 )

    # Get putative updated H₂ fraction: 
    xH2_new  = x_new( xH2_old, C_H2, D_H2, Δtsub )

    ##########################################                       
    ######      STEP 2: UPDATE H I      ######
    ##########################################
    # Next we update H I, using updated 
    # value of the H₂-fraction, but otherwise 
    # un-updated values. 

    # The following rates appear several times, so 
    # compute them once for convenience/optimization:
    KrecHII     = krecHII(T, CaseB)
    Kcollion_HI = kcollion_HI(T)

    # H I creation term (from H₂ destruction & recombinations):
    C_HI    = ( 2 * xH2_new * ( Kdiss_H2HI * nHI_old  
                              + Kdiss_H2H2 * nH2_old  # These processes produce *two* H I atoms
                              + Kdiss_H2e  * ne_old
                              + Γ_LW + ΓionH2 ) 

              + xH2_new * Kdiss_H2HII * nHII_old      # This process only produce one H I atom 

              + KrecHII * ne_old * xHII_old ) 

    # H I destruction term (from photoionization & H₂ formation):
    D_HI    = ( 2 * ( Kdust_H2 * nH 
                    + KGP_H2 * ne_old 
                    + K3B_H2 * nHI_old * (nHI_old + nH2_old / 8) )
                    
              + Kcollion_HI * ne_old + ΓionHI )

    # Get updated H I fraction:
    xHI_new = x_new( xHI_old, C_HI, D_HI, Δtsub )

    ##########################################                       
    ######      STEP 3: UPDATE H II     ######
    ##########################################
    # Next we update H II, using the newly
    # estimated H I fraction.  

    # H II creation term (collisional & photo-ionizations)
    C_HII    = ( Kcollion_HI * ne_old + ΓionHI ) * xHI_new 

    # H II destruction term (recombinations & reaction H₂ + H⁺ → H + H₂⁺,
    # using updated H₂ number density):
    D_HII    = KrecHII * ne_old + Kdiss_H2HII * xH2_new * nH

    # Get updated H II fraction:
    xHII_new = x_new( xHII_old, C_HII, D_HII, Δtsub )

    ##############################################                       
    ######   STEP 4: ENFORCE CONSERVATION   ######
    ##############################################
    # Next we enforce particle conservation, i.e.
    # 2*xH2 + xHI + xHII = 1. This is done by correcting
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
    # chemistry steps) using the known value of Tnorm 
    # and the new μ and γ (computed below):
    μ      = Getμ(xHII_new, xH2_new, xHeII_old, xHeIII_old)
    γad    = Getγad(xHII_new, xH2_new, xHeII_old, xHeIII_old) 
    T      = μ * (γad - 1) * Tnorm 

    # Update the electron number density:
    ne_new = Getne( xHII_new, xHeII_old, xHeIII_old, nH, nHe )

    ##########################################                       
    ###### STEP 5: UPDATE D I, D II, HD ######
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
    A11 = - (ΓionDI  +  Kcollion_HI * ne_new  +  K50_D * nHII_new  +  K54_D * nH2_new)
    A12 = KrecHII * ne_new  +  K51_D * nHI_new 
    A13 = K55_D * nHI_new  +  Γ_LW_HD

    A21 = ΓionDI  +  Kcollion_HI * ne_new  +  K50_D * nHII_new 
    A22 = - (KrecHII * ne_new  +  K51_D * nHI_new  +  K52_D * nH2_new )
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

    ##########################################                       
    ######     STEP 6: UPDATE He I      ######
    ##########################################
    # Next we update He I, using updated 
    # value of the H₂-fraction, but otherwise 
    # un-updated values. 

    # For convenience/optimization:
    KrecHeII     = krecHeII(T, CaseB)
    Kcollion_HeI = kcollion_HeI(T)

    # He I creation term (recombinations):
    C_HeI        = KrecHeII * ne_new * xHeII_old

    # He I destruction term (collisional & photo-ionization):
    D_HeI        = Kcollion_HeI * ne_new + ΓionHeI

    # Updated He I fraction: 
    xHeI_new     = x_new( xHeI_old, C_HeI, D_HeI, Δtsub )  

    ##########################################                       
    ######     STEP 7: UPDATE He II     ######
    ##########################################

    # For convenience/optimization:
    Kcollion_HeI  = kcollion_HeI(T)
    Kcollion_HeII = kcollion_HeII(T)
    KrecHeII      = krecHeII(T, CaseB)
    KrecHeIII     = krecHeIII(T, CaseB) 

    # He II creation term (from He I collional and 
    # photo-ionization, and He III recombinations): 
    C_HeII        = ( (Kcollion_HeI * ne_new + ΓionHeI) * xHeI_new
                    +  KrecHeIII * ne_new * xHeIII_old ) 

    # He II destruction term (recombinations, collisional
    # and photo-ionizations):
    D_HeII        = ( (KrecHeII + Kcollion_HeII) * ne_new
                    + ΓionHeII )

    # Updated He II fraction:
    xHeII_new     = x_new( xHeII_old, C_HeII, D_HeII, Δtsub )

    ##########################################                       
    ######     STEP 8: UPDATE He III    ######
    ##########################################

    # He III creation term (collisional and photo-ionization
    # of He II)
    C_HeIII       = ( Kcollion_HeII * ne_new + ΓionHeII ) * xHeII_new

    # He III destruction term (recombinations):
    D_HeIII       = KrecHeIII * ne_new 

    xHeIII_new    = x_new( xHeIII_old, C_HeIII, D_HeIII, Δtsub )

    ##############################################                       
    ######   STEP 9: ENFORCE CONSERVATION   ######
    ##############################################

    Δf_He  = ( xHeI_new + xHeII_new + xHeIII_new ) - 1.0
    idx_He = argmax( ( xHeI_new, xHeII_new, xHeIII_new ) )

    if idx_He == 1
        # In this case, He I is the biggest, so correct it:
        xHeI_new   = max( xHeI_new - Δf_He, 0.0 )
    elseif idx_He == 2
        # In this case, He II is the biggest, so correct it:
        xHeII_new  = max( xHeII_new - Δf_He, 0.0 ) 
    else
        # In this case, He III is the biggest, so correct it:
        xHeIII_new = max( xHeIII_new - Δf_He, 0.0 ) 
    end 

    # Get a final estimate of the electron density:
    ne_new = Getne( xHII_new, xHeII_new, xHeIII_new, nH, nHe )

    ##############################################                       
    ######  STEP 10: UPDATE O II, OHₓ, CO   ######
    ##############################################
    # TODO: Next we update the oxygen chemistry. 

    #xOI_new, xOII_new, xCI_new, xCII_new, xCO_new, ConvergedCarbonOxygen = 
    #CarbonOxygenChemistry( xOI_old, xOII_old, xCI_old, xCII_old,
    #                       xCO_old, nC, nO, nHI_new, nHII_new,
    #                       nH2_new, ne_new, ΓionH2, ΓionOI, ΓionCI,
    #                       ΓdissCO, ΓdissOHx, ΓdissCHx,
    #                       T, Δtsub )

    # Now we have updated every species. ✓

    return xH2_new, xHI_new, xHII_new, xHD_new, xDI_new, xDII_new,
           xHeI_new, xHeII_new, xHeIII_new, ne_new
end


end # End of module 