
module PhotoThermoChemistry
""" MODULE:     PhotoThermoChemistry

    PURPOSE:    Contains the full thermochemistry + dust + stellar RT update.
    
"""

__precompile__()

include(joinpath(@__DIR__, "..", "ConstantsAndFunctions.jl"))
include( "./GasChemistryRates.jl" )
include( "./GasCoolingRates.jl" )
include( "./StellarEmissionAbsorption.jl" )
include( "./CosmicRays.jl" )
include( "./UpdateStellarJ.jl" )
include( "./UpdateChemistry.jl" )
include( "./UsefulPhotoThermoChemistryFunctions.jl" )
include( joinpath(@__DIR__, "..", "Dust", "SilicateDustOpacities.jl") )
include( joinpath(@__DIR__, "..", "Dust", "GraphiteDustOpacities.jl") )
include( joinpath(@__DIR__, "..", "Dust", "DustProperties.jl") )
include( joinpath(@__DIR__, "..", "Dust", "DustCoagulation.jl") )
include( joinpath(@__DIR__, "..", "Dust", "DustDynamics.jl") )
include( joinpath(@__DIR__, "..", "Dust", "DustDestruction.jl") )
include( joinpath(@__DIR__, "..", "Hydrodynamics", "Hydrodynamics.jl") )
using .ConstantsAndFunctions
using .GasChemistryRates
using .GasCoolingRates
using .StellarEmissionAbsorption            # Stellar absorption, ionization, heating
using .CosmicRays                           # Cosmic ray heating and ionization
using .UsefulPhotoThermoChemistryFunctions  # Useful functions 
using .SilicateDustOpacities
using .GraphiteDustOpacities
using .DustProperties         # For computing the grain temperatures and charges
using .DustCoagulation        # For dust coagulation 
using .DustDynamics           # Needed to match the assumed dust density floor 
using .DustDestruction
using .UpdateChemistry
using .UpdateStellarJ
using .Hydrodynamics          # For the gas density floor, around which we don't inject radiation pressure
using Base.Threads            # For CPU acceleration
using Printf: @sprintf


export UpdateStellarJandThermoChemistry!


function HeatingRate_total( J_opt::Float64, 
                            J_FUV::Float64, 
                             J_LW::Float64,
                           J_EUV1::Float64,
                           J_EUV2::Float64,
                           J_EUV3::Float64,
                            T_old::Float64,
                                ρ::Float64,
                           ρd_old::Float64,
                           nH_old::Float64, 
                           ne_old::Float64,
                          nHI_old::Float64,
                          nH2_old::Float64,
                         nHeI_old::Float64,
                        nHeII_old::Float64,
                          nCI_old::Float64,
                    SelfShielding::Float64,
                             PLyα::Float64,
                              ζHI::Float64
                             )
    
    (  HeatingRate_PhotoElectric( J_FUV, J_LW, ρd_old / ρ, nH_old, ne_old, T_old, PLyα )
    +  HeatingRate_Compton(J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, ne_old, T_old, PLyα )
    +  HeatingRate_LW( SelfShielding * J_LW, nH2_old, nHI_old, T_old )
    +  HeatingRate_EUV1( J_EUV1, nHI_old, nH2_old ) 
    +  HeatingRate_EUV2( J_EUV2, nHI_old, nH2_old, nHeI_old ) 
    +  HeatingRate_EUV3( J_EUV3, nHI_old, nH2_old, nHeI_old, nHeII_old ) 
    +  CosmicRayHeating( ζHI, nHI_old, nH2_old, nHeI_old, nCI_old ) )

end



const T_floor::Float64          = 1e-3     # Enforced temperature floor in the thermochemistry sub-cycling
const T_ceiling::Float64        = 1e9      # Enforced temperature ceiling in the thermochemistry sub-cycling
const dust_mask_factor::Float64 = 1.1      # We ignore dust absorption for dust densities ρd < dust_mask_factor * ρd_floor
const tol::Float64              = 0.1      # Maximum relative change tolerance per sub-cycle time-step 
const xmin::Float64             = 1e-8     # Minimum abundance fraction below which we don't care about the 10% rule
const max_iter::Int             = 100      # Maximum number of adaptive refinements of sub-cycle time-step 

# Pre-compute weights needed for dust coagulation:
const W_ijk                     = PreCompute_W_ijk( agr_bins )

""" This functions does a single sub-cycle time-step update of
    intensities, gas internal energy, and chemical abundances. 
    
    **ARGUMENTS:**  

    - Δtsubguess   :  Input guess of sub-cycle time-step (which can then adaptively change)
    - e_old        :  Internal energy density (erg cm⁻³) at previous sub-cycle time-step
    - nH_old       :  Total (H I, H II, H₂) hydrogen number density (cm⁻³)
    - nH2_old      :  H₂ number density at previous sub-cycle time-step (cm⁻³)
    - nHII_old     :  H II number density at previous sub-cycle time-step (cm⁻³)
    - nHeII_old    :  He II number density at previous sub-cycle time-step (cm⁻³)
    - nHeIII_old   :  He III number density at previous sub-cycle time-step (cm⁻³)
    - J_LW_old     :  Lyman-Werner intensity at previous sub-cycle time-step (erg cm⁻² s⁻¹ st⁻¹)
    - J_EUV1_old   :  EUV1-band intensity at previous sub-cycle time-step    (erg cm⁻² s⁻¹ st⁻¹)
    - J_EUV2_old   :  EUV2-band intensity at previous sub-cycle time-step    (erg cm⁻² s⁻¹ st⁻¹)
    - J_EUV3_old   :  EUV3-band intensity at previous sub-cycle time-step    (erg cm⁻² s⁻¹ st⁻¹)
    - CaseB        :  True (false) if Case B recombinations are (not) used   (default = true)
    
    **RETURNS:**

    - Δtsub        :  The adaptively chosen sub-cycle time-step      (sec)
    - Δtsubguess   :  Guess of sub-cycle time-step for next sub-step (sec)
    - e_new        :  Updated internal energy density (erg cm⁻³)
    - nH2_new      :  Updated H₂ number density     (cm⁻³)
    - nHIInew      :  Updated H II number density   (cm⁻³) 
    - nHeIInew     :  Updated He II number density  (cm⁻³) 
    - nHeIIInew    :  Updated He III number density (cm⁻³) 
    - J_LW         :  Updated Lyman-Werner intensity (erg cm⁻² s⁻¹ st⁻¹)
    - J_EUV1       :  Updated EUV1-band intensity    (erg cm⁻² s⁻¹ st⁻¹)
    - J_EUV2       :  Updated EUV2-band intensity    (erg cm⁻² s⁻¹ st⁻¹)
    - J_EUV3       :  Updated EUV3-band intensity    (erg cm⁻² s⁻¹ st⁻¹)

    """
@views function ThermoChemistrySubStepUpdate!( icell::Int,
                                          Δtsubguess::Float64,
                                               e_old::Float64,
                                                   ρ::Float64,
                                             xH2_old::Float64,
                                            xHII_old::Float64,
                                             xHD_old::Float64,
                                            xDII_old::Float64, 
                                           xHeII_old::Float64,
                                          xHeIII_old::Float64,
                                             xCI_old::Float64,
                                            xCII_old::Float64,
                                             xCO_old::Float64,
                                             xOI_old::Float64,
                                            xOII_old::Float64,
                                                   u::Float64,
                                           J_opt_old::Float64 = 0.0,
                                            dJ_optdt::Float64 = 0.0,
                                           J_FUV_old::Float64 = 0.0,
                                            dJ_FUVdt::Float64 = 0.0,
                                            J_LW_old::Float64 = 0.0,
                                             dJ_LWdt::Float64 = 0.0,
                                          J_EUV1_old::Float64 = 0.0,
                                           dJ_EUV1dt::Float64 = 0.0,
                                          J_EUV2_old::Float64 = 0.0,
                                           dJ_EUV2dt::Float64 = 0.0,
                                          J_EUV3_old::Float64 = 0.0,
                                           dJ_EUV3dt::Float64 = 0.0,
                                                N_H2::Float64 = 0.0,
                                                N_HD::Float64 = 0.0,
                                                N_CO::Float64 = 0.0,
                                               ∇dotu::Float64 = 0.0,  # Needed for Sobolev optical depth & adiabatic cooling/heating
                                                fred::Float64 = 1.0,
                                              Z_Zsun::Float64 = 1.0,
                                                   z::Float64 = 0.0,
                                                PLyα::Float64 = 0.0,
                                                 ζHI::Float64 = 0.0,
                                               CaseB::Bool    = true,
                                      CoolingHeating::Bool    = true
                                                )


    # Get relevant number densities & abundance fractions:
    nH_old            =   X * ρ / mH
    nD_old            =   D_H_ratio * nH_old
    nHe_old           =   Y * ρ / mHe
    nC_old            =   C_H_ratio * nH_old
    nO_old            =   O_H_ratio * nH_old
    xHI_old           =   max( 1 - xHII_old - 2 * xH2_old, 0.0 )
    xDI_old           =   max( 1 - xDII_old  - xHD_old,    0.0 )
    xHeI_old          =   max( 1 - xHeII_old - xHeIII_old, 0.0 )
    
    nHI_old           =   xHI_old    * nH_old
    nHII_old          =   xHII_old   * nH_old 
    nH2_old           =   xH2_old    * nH_old
    nHD_old           =   xHD_old    * nD_old    # Enters the cooling rate
    nHeI_old          =   xHeI_old   * nHe_old
    nHeII_old         =   xHeII_old  * nHe_old
    nHeIII_old        =   xHeIII_old * nHe_old
    nCI_old           =   xCI_old    * nC_old
    nCII_old          =   xCII_old   * nC_old
    nOI_old           =   xOI_old    * nO_old
    nOII_old          =   xOII_old   * nO_old
    nCO_old           =   xCO_old    * nC_old
    nCHx_old          =   max((1 - xCI_old  - xCII_old - xCO_old), 0.0 ) * nC_old 
    nOHx_old          =   max((1 - xOI_old  - xOII_old - (nCO_old / nO_old)), 0.0 ) * nO_old 
    ne_old            =   nHII_old  +  nHeII_old  +  2 * nHeIII_old
    n                 =   nH_old  +  nHe_old  +  ne_old 

    # We should only include dust absorption from cells above the dust density floor:
    ρd_Sil_old_masked =  ρd_Sil_tot_old * ( ρd_Sil_tot_old ≥ dust_mask_factor * ηfloor_dust * ρ ) 
    ρd_C_old_masked   =  ρd_C_tot_old * ( ρd_C_tot_old ≥ dust_mask_factor * ηfloor_dust * ρ ) 

    # ------- Get opacities (cm⁻¹) in all bands ----------------------------------------------
    αc_opt        = ( nCHx_old * σCH_opt   +  κabs_opt_Sil( ρd_Sil_row, ρd_Sil_tot_old ) * ρd_Sil_old_masked  
                    + κabs_opt_C( ρd_C_row, ρd_C_tot_old ) * ρd_C_old_masked )
    αc_FUV        = ( nOHx_old * σOH_FUV   +  nCHx_old * σCH_FUV  +  κabs_FUV_Sil( ρd_Sil_row, ρd_Sil_tot_old ) * ρd_Sil_old_masked
                    + κabs_FUV_C( ρd_C_row, ρd_C_tot_old ) * ρd_C_old_masked )
    αc_LW         = ( nHI_old  * σHI_LW    +  nH2_old * σH2_LW    +  nCI_old * σCI_LW  
                    + nCHx_old * σCH_LW    +  nOHx_old * σOH_LW   +  nCO_old * σCO_LW    
                    + κabs_LW_Sil( ρd_Sil_row, ρd_Sil_tot_old ) * ρd_Sil_old_masked
                    + κabs_LW_C( ρd_C_row, ρd_C_tot_old )       * ρd_C_old_masked )


    αc_EUV1       = ( nHI_old  * σHI_EUV1  +  nH2_old * σH2_EUV1  +  nCI_old * σCI_EUV1  +  nOI_old * σOI_EUV1
                    + κabs_EUV1_Sil( ρd_Sil_row, ρd_Sil_tot_old ) * ρd_Sil_old_masked
                    + κabs_EUV1_C( ρd_C_row, ρd_C_tot_old )       * ρd_C_old_masked )

    αc_EUV2       = ( nHI_old  * σHI_EUV2  +  nH2_old * σH2_EUV2  +  nHeI_old * σHeI_EUV2  +  nCI_old * σCI_EUV2  
                    + nOI_old * σOI_EUV2  +  κabs_EUV2_Sil( ρd_Sil_row, ρd_Sil_tot_old ) * ρd_Sil_old_masked
                    + κabs_EUV2_C( ρd_C_row, ρd_C_tot_old ) * ρd_C_old_masked )

    αc_EUV3       = ( nHI_old * σHI_EUV3      +  nH2_old * σH2_EUV3  +  nHeI_old * σHeI_EUV3 
                    + nHeII_old * σHeII_EUV3  +  nCI_old * σCI_EUV3  +  nOI_old * σOI_EUV3  
                    + κabs_EUV3_Sil( ρd_Sil_row, ρd_Sil_tot_old ) * ρd_Sil_old_masked
                    + κabs_EUV3_C( ρd_C_row, ρd_C_tot_old )       * ρd_C_old_masked )


    # --------- Get old temperature, and enforce temperature floor ----------------------------
    γad           =   Getγad(xHII_old, xH2_old, xHeII_old, xHeIII_old)
    e_old         =   max( e_old, n * kB * T_floor / (γad - 1) )
    T_old         =   (γad - 1) * e_old  / (n * kB)

    # Get dominant creation rates for H I, H₂, He I, He II, which goes into setting the 
    # physical lower limits on updated J following absorption:
    C_HI          =   krecHII(T_old, CaseB) * nHII_old * ne_old

    Γpd           =   PhotoDetachRate_Hminus( J_opt_old, J_FUV_old, J_LW_old, J_EUV1_old, J_EUV2_old, J_EUV3_old )
    C_H2          =   ( kGP_H2(T_old, ne_old, nHI_old, nHII_old, Γpd, z) * ne_old * nHI_old 
                      + kdust_H2(T_old, ρd_Sil_row, ρd_C_row, T_dust_Sil_row, T_dust_C_row, agr_bins, ρgr_Sil, ρgr_C) * nH_old * nHI_old 
                      + k3B_H2(T_old) * (nHI_old^2) * (nHI_old + nH2_old / 8) )

    C_HeI         =   krecHeII(T_old, CaseB) * nHeII_old * ne_old  

    C_HeII        =   krecHeIII(T_old, CaseB ) * nHeIII_old * ne_old 

    C_CI          =   kCIIe(T_old) * nCII_old * ne_old  +  kCIIH2_2(T_old) * nCII_old * nH2_old

    C_CO          =   kCHxOI(T_old) * nCHx_old * nOI_old  +  kOHxCI(T_old) * nOHx_old * nCI_old

    #C_OI          =   kOIe

    # -------- Get the OHₓ creation rate - for this we need the H₃⁺ density also ----------
    ΓionH2        =    IonRate_H2( J_EUV1_old, J_EUV2_old, J_EUV3_old )  +  2 * ζHI   
    nH3plus       =    ΓionH2 * nH2_old / (
                       kCIH3plus(T_old) * nCI_old  +  kOIH3plus(T_old, nH2_old, ne_old) * nOI_old  +  kH3pluse(T_old) * ne_old)
    C_OHx         =   ( kOIH3plus(T_old, nH2_old, ne_old) * nH3plus * nOI_old  +
                        kOIIH2_1(T_old, nH2_old, ne_old) * nH2_old * nOII_old )

    # -------- Get the CHₓ creation rate --------------------------------------------------

    C_CHx         =   kCIH3plus(T_old) * nH3plus * nCI_old  +  kCIIH2_1(T_old) * nH2_old * nCII_old

    # ------- H₂ and HD self-shielding factors --------------------------------------------
    SelfShielding       =  S_H2( N_H2, T_old )  # Self-shielding of H₂
    SelfShielding_HD    =  S_HD( N_HD, T_old )  # Self-shielding of HD
    SelfShielding_H2_HD =  S_H2_HD( N_H2 )      # Shielding of HD by H₂

    # Compute the CO Sobolev column density (needed for CO cooling), and limit 
    # the adopted CO column density by this:
    if abs(∇dotu) > 0.0 && nCO_old > 0.0
        # If there are velocity gradients, we limit the CO column density accordingly:
        N_CO_Sobolev =  nCO_old * sqrt(2 * kB * T_old / mCO) / (abs(∇dotu) / 3)
        N_CO         =  min( N_CO,  N_CO_Sobolev )
    end

    # Reduced speed of light:
    cred = fred * c

    # ------------------------------------------------------------------------------------
    
    Δtsub     = Δtsubguess # Initialize first guess of sub-cycle time-step
    iter::Int = 0          # Initialize iteration count
    while iter < max_iter 
        iter += 1

        #######################################################################
        #######                                                         #######
        #######    S T E P  1  :  U P D A T E  I N T E N S I T Y,       #######
        #######                   H E A T I N G,                        #######
        #######                   P H O T O - D E S T R U C T I O N     ####### 
        #######                                                         #######
        #######################################################################
        # First we update the intensity by absorption, using old number 
        # densities

        # Update intensities in all bands, limited by the maximum absorption possible:

        # ----- Optical intensities ----------------------------------------------------------------------------------

        αc_opt_dust   = κabs_opt_Sil( ρd_Sil_row, ρd_Sil_tot_old ) * ρd_Sil_old_masked  
                      + κabs_opt_C( ρd_C_row, ρd_C_tot_old ) * ρd_C_old_masked
        
        J_opt_min     = max( ( J_opt_old  +  dJ_optdt * Δtsub 
                           - (cred * E_opt / Four_π) * (nCHx_old + C_CHx * Δtsub) ) / 
                             (1 + cred * αc_opt_dust * Δtsub), 0.0 )

        J_opt         = max( AbsorptionUpdateStellarJ( J_opt_old, Δtsub, dJ_optdt, αc_opt, fred ),
                             J_opt_min, 
                             0.0 )

        J_opt_av      = max( AbsorptionUpdateStellarJ_average( J_opt_old, J_opt, Δtsub, dJ_optdt, αc_opt, fred, J_opt_min ), 0.0 )

        #ΔJ_opt        = abs( J_opt - J_opt_old ) / ( 1e-10 + (J_opt + J_opt_old) / 2 )
        #if ΔJ_opt > tol 
        #    Δtsub = Δtsub / 2
        #    continue    
        #end

        # ----- Far-UV intensities ----------------------------------------------------------------------------------

        αc_FUV_dust   = κabs_FUV_Sil( ρd_Sil_row, ρd_Sil_tot_old ) * ρd_Sil_old_masked  
                      + κabs_FUV_C( ρd_C_row, ρd_C_tot_old ) * ρd_C_old_masked

        J_FUV_min     = max( ( J_FUV_old  +  dJ_FUVdt * Δtsub 
                           - (cred * E_FUV / Four_π) * (nOHx_old + C_OHx * Δtsub) ) / 
                             (1 + cred * αc_FUV_dust * Δtsub), 0.0 )

        J_FUV         = max( AbsorptionUpdateStellarJ( J_FUV_old, Δtsub, dJ_FUVdt, αc_FUV, fred ),
                             J_FUV_min, 
                             0.0 )

        J_FUV_av      = max( AbsorptionUpdateStellarJ_average( J_FUV_old, J_FUV, Δtsub, dJ_FUVdt, αc_FUV, fred, J_FUV_min ), 0.0 )

        #ΔJ_FUV        = abs( J_FUV - J_FUV_old ) / ( 1e-10 + (J_FUV + J_FUV_old) / 2 )
        #if ΔJ_FUV > tol 
        #    Δtsub = Δtsub / 2
        #    continue    
        #end

        # ----- Lyman-Werner intensities ----------------------------------------------------------------------------------

        αc_LW_dust    = κabs_LW_Sil( ρd_Sil_row, ρd_Sil_tot_old ) * ρd_Sil_old_masked  
                      + κabs_LW_C( ρd_C_row, ρd_C_tot_old ) * ρd_C_old_masked

        J_LW_min      = max( ( J_LW_old  +  dJ_LWdt * Δtsub 
                           - (cred * E_LW / Four_π) * (nHI_old + nH2_old + nCI_old + nCO_old + nOHx_old + (C_HI + C_H2 + C_CI + C_CO + C_OHx) * Δtsub) ) / 
                             (1 + cred * αc_LW_dust * Δtsub), 0.0 )

        J_LW          = max( AbsorptionUpdateStellarJ( J_LW_old, Δtsub, dJ_LWdt, αc_LW, fred ),
                             J_LW_min, 
                             0.0  )

        J_LW_av       = max( AbsorptionUpdateStellarJ_average( J_LW_old, J_LW, Δtsub, dJ_LWdt, αc_LW, fred, J_LW_min ) , 0.0 )

        #ΔJ_LW         = abs( J_LW - J_LW_old ) / ( 1e-10 + (J_LW + J_LW_old) / 2 )
        #if ΔJ_LW > tol 
        #    Δtsub = Δtsub / 2
        #    continue    
        #end

        # ----- EUV1 intensities ----------------------------------------------------------------------------------

        αc_EUV1_dust  = κabs_EUV1_Sil( ρd_Sil_row, ρd_Sil_tot_old ) * ρd_Sil_old_masked  
                      + κabs_EUV1_C( ρd_C_row, ρd_C_tot_old ) * ρd_C_old_masked

        J_EUV1_min    = max( ( J_EUV1_old  +  dJ_EUV1dt * Δtsub 
                           - (cred * E_EUV1 / Four_π) * (nHI_old + nH2_old + (C_HI + C_H2) * Δtsub) ) / 
                             (1 + cred * αc_EUV1_dust * Δtsub), 0.0 )

        J_EUV1        = max( AbsorptionUpdateStellarJ( J_EUV1_old, Δtsub, dJ_EUV1dt, αc_EUV1, fred ),
                             J_EUV1_min, 
                             0.0 )

        J_EUV1_av     = max( AbsorptionUpdateStellarJ_average( J_EUV1_old, J_EUV1, Δtsub, dJ_EUV1dt, αc_EUV1, fred, J_EUV1_min ), 0.0 )

        #ΔJ_EUV1       = abs( J_EUV1 - J_EUV1_old ) / ( 1e-10 + (J_EUV1 + J_EUV1_old) / 2 )
        #if ΔJ_EUV1 > tol 
        #    Δtsub = Δtsub / 2
        #    continue    
        #end
        
        # ----- EUV2 intensities ----------------------------------------------------------------------------------

        αc_EUV2_dust  = κabs_EUV2_Sil( ρd_Sil_row, ρd_Sil_tot_old ) * ρd_Sil_old_masked  
                      + κabs_EUV2_C( ρd_C_row, ρd_C_tot_old ) * ρd_C_old_masked

        J_EUV2_min    = max( ( J_EUV2_old  +  dJ_EUV2dt * Δtsub 
                           - (cred * E_EUV2 / Four_π) * (nHI_old + nH2_old + nHeI_old + (C_HI + C_H2 + C_HeI) * Δtsub) ) / 
                             (1 + cred * αc_EUV2_dust * Δtsub), 0.0 )

        J_EUV2        = max( AbsorptionUpdateStellarJ( J_EUV2_old, Δtsub, dJ_EUV2dt, αc_EUV2, fred ),
                             J_EUV2_min, 
                             0.0 )

        J_EUV2_av     = max( AbsorptionUpdateStellarJ_average( J_EUV2_old, J_EUV2, Δtsub, dJ_EUV2dt, αc_EUV2, fred, J_EUV2_min ), 0.0 )

        #ΔJ_EUV2       = abs( J_EUV2 - J_EUV2_old ) / ( 1e-10 + (J_EUV2 + J_EUV2_old) / 2 )
        #if ΔJ_EUV2 > tol 
        #    Δtsub = Δtsub / 2
        #    continue    
        #end

        # ----- EUV3 intensities ----------------------------------------------------------------------------------

        αc_EUV3_dust  = κabs_EUV3_Sil( ρd_Sil_row, ρd_Sil_tot_old ) * ρd_Sil_old_masked  
                      + κabs_EUV3_C( ρd_C_row, ρd_C_tot_old ) * ρd_C_old_masked

        J_EUV3_min    = max( ( J_EUV3_old  +  dJ_EUV3dt * Δtsub 
                           - (cred * E_EUV3 / Four_π) * (nHI_old + nH2_old + nHeI_old + nHeII_old + (C_HI + C_H2 + C_HeI + C_HeII) * Δtsub) ) / 
                             (1 + cred * αc_EUV3_dust * Δtsub), 0.0 )

        J_EUV3        = max( AbsorptionUpdateStellarJ( J_EUV3_old, Δtsub, dJ_EUV3dt, αc_EUV3, fred ),
                             J_EUV3_min, 
                             0.0 )

        J_EUV3_av     = max( AbsorptionUpdateStellarJ_average( J_EUV3_old, J_EUV3, Δtsub, dJ_EUV3dt, αc_EUV3, fred, J_EUV3_min ), 0.0 )
        
        #ΔJ_EUV3       = abs( J_EUV3 - J_EUV3_old ) / ( 1e-10 + (J_EUV3 + J_EUV3_old) / 2 )
        #if ΔJ_EUV3 > tol 
        #    Δtsub = Δtsub / 2
        #    continue    
        #end

        # Collect largest change in intensity:
        #ΔJ = max( ΔJ_opt, ΔJ_FUV, ΔJ_LW, ΔJ_EUV1, ΔJ_EUV2, ΔJ_EUV3 )

        # Get heating rate (from photoelectric heating, Compton heating off the (non-CMB) radiation field,
        # photo-ionizations/dissociations, UV pumping, and cosmic ray heating):
        HeatingRate = HeatingRate_total( J_opt_av, J_FUV_av, J_LW_av, J_EUV1_av, J_EUV2_av, J_EUV3_av,
                                         T_old, ρ, ρd_Sil_old_masked + ρd_C_old_masked, 
                                         nH_old, ne_old, nHI_old, nH2_old, 
                                         nHeI_old, nHeII_old, nCI_old, SelfShielding, 
                                         PLyα, ζHI ) 

        # Get photo-ionization/dissociation rates (s⁻¹):
        Γ_LW        =  PhotoDissRate_H2( SelfShielding * J_LW_av )
        Γpd         =  PhotoDetachRate_Hminus( J_opt_av, J_FUV_av, J_LW_av, J_EUV1_av, J_EUV2_av, J_EUV3_av )
        Γ_LW_HD     =  PhotoDissRate_HD( SelfShielding_HD * SelfShielding_H2_HD * J_LW_av )
        ΓionH2      =  IonRate_H2( J_EUV1_av, J_EUV2_av, J_EUV3_av )  +  ζH2(ζHI)
        ΓionHI      =  IonRate_HI( J_EUV1_av, J_EUV2_av, J_EUV3_av )  +  ζHI
        ΓionDI      =  ΓionHI
        ΓionHeI     =  IonRate_HeI( J_EUV2_av, J_EUV3_av )  +  ζHeI(ζHI)
        ΓionHeII    =  IonRate_HeII( J_EUV3_av )
        ΓionCI      =  IonRate_CI( J_LW_av, J_EUV1_av, J_EUV2_av, J_EUV3_av )  +  ζCI(ζHI)
        ΓionOI      =  IonRate_OI( J_EUV1_av, J_EUV2_av, J_EUV3_av )
        ΓdissCHx    =  PhotoDissRate_CHₓ( J_opt_av, J_FUV_av, J_LW_av )
        ΓdissOHx    =  PhotoDissRate_OHₓ( J_FUV_av, J_LW_av )
        ΓdissCO     =  PhotoDissRate_CO( J_LW_av )

        # Update the dust temperatures:
        UpdateTdust_bins!( T_dust_Sil_row, J_opt_av, J_FUV_av, J_LW_av, J_EUV1_av, J_EUV2_av, J_EUV3_av,
                           nHI_old, nH2_old, nHII_old, nHeI_old, nHeII_old, nHeIII_old, ne_old, T_old,
                           PLyα, false, z )
        UpdateTdust_bins!( T_dust_C_row, J_opt_av, J_FUV_av, J_LW_av, J_EUV1_av, J_EUV2_av, J_EUV3_av,
                           nHI_old, nH2_old, nHII_old, nHeI_old, nHeII_old, nHeIII_old, ne_old, T_old,
                           PLyα, true, z )

        # Update the dust charges:
        UpdateZgr_bins!( Zgr_Sil_row, J_FUV_av, J_LW_av, J_EUV1_av, J_EUV2_av, J_EUV3_av, 
                         ne_old, T_old, false, PLyα, icell )
        UpdateZgr_bins!( Zgr_C_row, J_FUV_av, J_LW_av, J_EUV1_av, J_EUV2_av, J_EUV3_av, 
                         ne_old, T_old, true, PLyα, icell )

        ################################################################
        #######                                                  #######
        #######    S T E P  2  :  U P D A T E  I N T E R N A L   #######
        #######                          E N E R G Y             #######
        #######                                                  #######
        ################################################################
        # In this step we try to update the internal energy e, using 
        # a second-order Patankar method.

        # Compute the cooling rate:
        CoolingRate = Λ( T_old, nHI_old, nHII_old, nHeI_old, nHeII_old, nHeIII_old, ne_old,
                         nH2_old, nHD_old, Z_Zsun, 
                         z, xCI_old, xCII_old, xCO_old, xOI_old, N_CO,
                         CaseB )

        ΔT = 1e-4 * T_old

        CoolingRate_2 = Λ( T_old + ΔT, nHI_old, nHII_old, nHeI_old, nHeII_old, nHeIII_old, ne_old,
                           nH2_old, nHD_old, Z_Zsun, 
                           T_dust_Sil_row, T_dust_C_row,
                           ρd_Sil_row, ρd_C_row,
                           Zgr_Sil_row, Zgr_C_row,
                           z, xCI_old, xCII_old, xCO_old, xOI_old, N_CO,
                           CaseB )

        ∂Λ = (CoolingRate_2 - CoolingRate) / ΔT

        e_new  = ( e_old  +  ( HeatingRate - CoolingRate ) * Δtsub  +  ∂Λ * e_old * Δtsub ) / ( 
                       1  +  ∂Λ * Δtsub )

        # Check that the relative change in e was sufficiently small, and that 
        # other stability/accuracy checks are satisfied:
        Δe = abs( e_new  -  e_old ) / e_old


        Δt_e  = tol * e_old / abs(HeatingRate - CoolingRate)
        #Δt_e = tol * min( e_old / abs(HeatingRate), 
        #                  e_old / abs(CoolingRate) )

        if Δe > tol || e_new ≤ 0.0 || !isfinite(e_new) || Δtsub > Δt_e
            Δtsub = min( Δtsub / 2,  
                         Δt_e )
            continue 
        end

        # Correct so that the temperature is between the ceiling and floor temperatures:
        if CoolingHeating 
            e_new = min( max( e_new, n * kB * T_floor / (γad - 1) ),
                         n * kB * T_ceiling / (γad - 1) )
        else
            # For some tests (e.g. R-type ionization front) we ignore heating/cooling:
            e_new = e_old
        end

        ##################################################################
        #######                                                    #######
        #######        S T E P  3  :  U P D A T E  D U S T         #######
        #######                                                    #######
        ##################################################################
        # Here we update the dust bin densities, taking into account dust growth
        # by coagulation, and destruction by sublimation, thermal sputtering, 
        # shattering, non-thermal sputtering, chemisputtering,
        # Coulomb explosions, and (if turned on) radiative torques.

        # Get temperature from the updated internal energy (but the
        # adiabatic index has not been updated yet, since chemistry is
        # unchanged):
        T  =  (γad - 1) * e_new  / (n * kB)

        # ------- COAGULATION UPDATE --------------------------------------------

        #### Silicate grains: ####

        ρd_Sil_1_coag, ρd_Sil_2_coag, ρd_Sil_3_coag, ρd_Sil_4_coag, ρd_Sil_5_coag,
        ρd_Sil_6_coag, ρd_Sil_7_coag, 
        Δρd_Sil_rel_coag  = UpdateDustCoagulation( ρd_Sil_row, agr_bins, mgr_bins_Sil,
                                                   ud_Sil_row, Zgr_Sil_row, T, nH_old, 
                                                   Δtsub, W_ijk, ρgr_Sil )

        #if Δρd_Sil_rel_coag > tol 
            # Change in total silicate dust density was not conserved well enough 
            # (before renormalization to machine precision), so we reject the update 
            # and try again with a smaller sub-cycle time-step:
        #    Δtsub = Δtsub / 2
        #    continue    
        #end

        # If accepted, we collect the candidate silicate dust bin densities:
        ρd_Sil_coag = ( ρd_Sil_1_coag, ρd_Sil_2_coag, ρd_Sil_3_coag, ρd_Sil_4_coag, ρd_Sil_5_coag,
                        ρd_Sil_6_coag, ρd_Sil_7_coag )

        #### Similar update for graphite grains: ####

        ρd_C_1_coag, ρd_C_2_coag, ρd_C_3_coag, ρd_C_4_coag, ρd_C_5_coag,
        ρd_C_6_coag, ρd_C_7_coag, 
        Δρd_C_rel_coag  = UpdateDustCoagulation( ρd_C_row, agr_bins, mgr_bins_C,
                                                 ud_C_row, Zgr_C_row, T, nH_old, 
                                                 Δtsub, W_ijk, ρgr_C )
        
        #if Δρd_C_rel_coag > tol 
        #    Δtsub = Δtsub / 2
        #    continue    
        #end

        ρd_C_coag = ( ρd_C_1_coag, ρd_C_2_coag, ρd_C_3_coag, ρd_C_4_coag, ρd_C_5_coag,
                      ρd_C_6_coag, ρd_C_7_coag )

        # Save maximum change in dust density (before renormalization), to determine the 
        # next sub-cycle time-step:
        Δρd_coag = max( Δρd_Sil_rel_coag, Δρd_C_rel_coag )

        # ------- DUST DESTRUCTION UPDATE ---------------------------------------

        # Next we do the dust destruction update, using as initial conditions the 
        # dust growth-updated dust densities.

        # Get candidate updates for the silicate dust density bins:
        ρd_Sil_candidate, Δρd_dest_Sil = UpdateDustDestruction( ρd_Sil_coag, T_dust_Sil_row, Zgr_Sil_row,
                                                  ρ,           T,              nH_old, 
                                                  ne_old,      xHI_old,        xHII_old,
                                                  J_opt_av,    J_FUV_av,       J_LW_av, 
                                                  J_EUV1_av,   J_EUV2_av,      J_EUV3_av,
                                                  ud_Sil_row,  u,              Δtsub,     
                                                  false )

        # Get candidate updates for the graphite dust density bins:
        ρd_C_candidate, Δρd_dest_C   = UpdateDustDestruction( ρd_C_coag, T_dust_C_row, Zgr_C_row,
                                                  ρ,         T,            nH_old, 
                                                  ne_old,    xHI_old,      xHII_old,
                                                  J_opt_av,  J_FUV_av,     J_LW_av, 
                                                  J_EUV1_av, J_EUV2_av,    J_EUV3_av,
                                                  ud_C_row,  u,            Δtsub,     
                                                  true )

        #Δρd_dest = max( Δρd_dest_Sil, Δρd_dest_C )
        Δρd_dest = 0.0
        #if Δρd_dest > tol 
        #    Δtsub = Δtsub / 2
        #    continue
        #end

        ##################################################################
        #######                                                    #######
        #######    S T E P  4  :  U P D A T E  C H E M I S T R Y   #######
        #######                                                    #######
        ##################################################################
        # Try update chemistry & check if relative changes is < 10%:

        Tnorm = e_new * mH / ( kB * ρ )

        ( xH2_new, xHI_new, xHII_new, xHD_new, xDI_new, xDII_new,
          xHeI_new, xHeII_new, xHeIII_new, xCI_new, xCII_new, xCO_new,
          xOI_new, xOII_new, ne_new, ConvergedCarbonOxygen ) =
          UpdateChemistrySubStep( Δtsub, Tnorm, ρ, xH2_old, xHII_old, xHD_old, xDII_old, xHeII_old, xHeIII_old, 
                                  xCI_old, xCII_old, xCO_old, xOII_old, xOI_old, 
                                  ρd_Sil_candidate, ρd_C_candidate, T_dust_Sil_row, T_dust_C_row,
                                  ΓionHI, ΓionH2, ΓionDI, ΓionHeI, ΓionHeII, ΓionCI, ΓionOI, Γ_LW, Γpd, Γ_LW_HD, 
                                  ΓdissCHx, ΓdissOHx, ΓdissCO,
                                  z, CaseB ) 

        # Compute would-be (symmetric) relative change in H I, H II, and
        # e⁻ fractions/number densities:
        ΔxH2_rel  = abs( xH2_new - xH2_old )   / max( (xH2_old + xH2_new) / 2,   xmin )
        ΔxHII_rel = abs( xHII_new - xHII_old ) / max( (xHII_old + xHII_new) / 2, xmin )
        Δne_rel   = abs( ne_new - ne_old )     / max( (ne_old + ne_new) / 2,     xmin * nH_old )

        # Get the maximum relative change among them:
        Δx_rel    = maximum( (ΔxH2_rel, ΔxHII_rel, Δne_rel) )

        if Δx_rel > tol || !ConvergedCarbonOxygen
            # Change in H I, H II, C II, O II, CO, and/or e⁻ 
            # fraction was too high, reduce Δtsub & retry:
            Δtsub = Δtsub / 2

            continue    
        end

        # Recommended next sub-cycle time-step based on the relative change 
        # in quantities:
        #if ( Δx_rel < 0.5 * tol  &&  Δe < 0.5 * tol &&
        #     Δρd_coag < 0.5 * tol  &&  Δρd_dest < 0.5 * tol )
        if ( Δx_rel < 0.5 * tol  &&  Δe < 0.5 * tol )
            Δtsub_new = 2 * min( Δtsub, Δt_e )
        else
            Δtsub_new = Δtsub
        end

        # At this point we accept the candidate dust densities:
        ρd_Sil_row .= ρd_Sil_candidate
        ρd_C_row   .= ρd_C_candidate
        
        # Return every updated quantity, the time-step taken, & recommended next sub-cycle time-step:
        return Δtsub, Δtsub_new, e_new, xH2_new, xHII_new, xHD_new, xDII_new, xHeII_new, xHeIII_new, 
               xCI_new, xCII_new, xCO_new, xOI_new, xOII_new,
               J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, 
               J_opt_av, J_FUV_av, J_LW_av, J_EUV1_av, J_EUV2_av, J_EUV3_av

    end

    # Otherwise print error:
    error(" ThermoChemistrySubStepUpdate could not converge within $(max_iter)\n" * 
          " iterations in cell $(icell)." )

end



max_steps::Int64 = 100000 # Maximum number of sub-cycling sub-steps per global time-step (default = 10⁵)

""" The full thermochemistry sub-cycling over a global time-step Δtglobal.
    Updates the intensities, gas internal energy, and chemical abundances.
    
    **ARGUMENTS:** 
    
    - 
    
    
    """
@views function ThermoChemistrySubCycleUpdate!(    icell::Int,
                                        Δtglobal::Float64,
                                           e_old::Float64,
                                               ρ::Float64,
                                         xH2_old::Float64,
                                        xHII_old::Float64, 
                                         xHD_old::Float64,
                                        xDII_old::Float64,
                                       xHeII_old::Float64,
                                      xHeIII_old::Float64,
                                         xCI_old::Float64,
                                        xCII_old::Float64,
                                         xCO_old::Float64,
                                         xOI_old::Float64,
                                        xOII_old::Float64,                                  
                                          ρd_Sil::AbstractMatrix{Float64},
                                 T_dust_bins_Sil::AbstractMatrix{Float64},
                                    Zgr_bins_Sil::AbstractMatrix{Float64},
                                            ρd_C::AbstractMatrix{Float64},
                                   T_dust_bins_C::AbstractMatrix{Float64},
                                      Zgr_bins_C::AbstractMatrix{Float64},
                                          ud_Sil::AbstractMatrix{Float64},
                                            ud_C::AbstractMatrix{Float64},
                                               u::AbstractVector{Float64},
                                Δρu_rad_dust_Sil::AbstractMatrix{Float64},
                                  Δρu_rad_dust_C::AbstractMatrix{Float64},
                                         Δρu_rad::AbstractVector{Float64},
                                    J_opt_transp::Float64 = 0.0,
                                        dJ_optdt::Float64 = 0.0,
                                    J_FUV_transp::Float64 = 0.0,
                                        dJ_FUVdt::Float64 = 0.0,
                                     J_LW_transp::Float64 = 0.0,
                                         dJ_LWdt::Float64 = 0.0,
                                   J_EUV1_transp::Float64 = 0.0,
                                       dJ_EUV1dt::Float64 = 0.0,
                                   J_EUV2_transp::Float64 = 0.0,
                                       dJ_EUV2dt::Float64 = 0.0,
                                   J_EUV3_transp::Float64 = 0.0,
                                       dJ_EUV3dt::Float64 = 0.0,
                                            N_H2::Float64 = 0.0,
                                            N_HD::Float64 = 0.0,
                                            N_CO::Float64 = 0.0,
                                           ∇dotu::Float64 = 0.0,
                                            fred::Float64 = 1.0,
                                          Z_Zsun::Float64 = 1.0,
                                               z::Float64 = 0.0,
                                            PLyα::Float64 = 0.0,
                                             ζHI::Float64 = 0.0,
                                           CaseB::Bool    = true,
                                  CoolingHeating::Bool    = true
                                        )
    # Initialize time, internal energy, and abundance fractions
    # using the ones from the previous global time-step:
    t_accum        = 0.0 
    e_current      = e_old 
    xH2_current    = xH2_old
    xHII_current   = xHII_old
    xHD_current    = xHD_old 
    xDII_current   = xDII_old
    xHeII_current  = xHeII_old
    xHeIII_current = xHeIII_old
    xCI_current    = xCI_old 
    xCII_current   = xCII_old 
    xCO_current    = xCO_old
    xOI_current    = xOI_old
    xOII_current   = xOII_old
    # Similarly, initialize the intensities as those from 
    # the transport step:
    J_opt_current  = J_opt_transp
    J_FUV_current  = J_FUV_transp
    J_LW_current   = J_LW_transp
    J_EUV1_current = J_EUV1_transp
    J_EUV2_current = J_EUV2_transp
    J_EUV3_current = J_EUV3_transp

    # Initialize the momentum density change from 
    # stellar radiation pressure, as well as the radiation
    # pressure acceleration:
    Δρu_rad[icell, jcell] = 0.0  # Momentum change in gas 
    
    # Start with the global time-step as an initial guess
    # for the sub-cycle time-step:
    Δtsub = Δtglobal

    # Initialize step count:
    step  = 0

    while t_accum < Δtglobal 
        # Ensure we don't overshoot the global time-step:
        Δtsub  = min( Δtsub, Δtglobal - t_accum )

        # Call sub-cycle update for this sub-step:
        result =  ThermoChemistrySubStepUpdate!( icell, Δtsub, e_current, ρ, 
                  xH2_current, xHII_current, xHD_current, xDII_current,
                  xHeII_current, xHeIII_current, 
                  xCI_current, xCII_current, xCO_current, 
                  xOI_current, xOII_current, 
                  J_opt_current, dJ_optdt, J_FUV_current, dJ_FUVdt,
                  J_LW_current, dJ_LWdt, J_EUV1_current, dJ_EUV1dt,
                  J_EUV2_current, dJ_EUV2dt, J_EUV3_current, dJ_EUV3dt,
                  N_H2, N_HD, N_CO, ∇dotu, fred, Z_Zsun, z,
                  PLyα, ζHI, CaseB, CoolingHeating )

        # Unpack result:
        Δtsub, Δtsub_new, e_new, xH2_new, xHII_new, xHD_new, xDII_new, xHeII_new, xHeIII_new, 
        xCI_new, xCII_new, xCO_new, xOI_new, xOII_new,
        J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, 
        J_opt_av, J_FUV_av, J_LW_av, J_EUV1_av, J_EUV2_av, J_EUV3_av = result

        # Update our current internal energy & abundance fractions:
        e_current      = e_new
        xH2_current    = xH2_new
        xHII_current   = xHII_new
        xHD_current    = xHD_new 
        xDII_current   = xDII_new
        xHeII_current  = xHeII_new
        xHeIII_current = xHeIII_new
        xCI_current    = xCI_new 
        xCII_current   = xCII_new 
        xCO_current    = xCO_new 
        xOI_current    = xOI_new
        xOII_current   = xOII_new
        # Update our current intensities:
        J_opt_current  = J_opt
        J_FUV_current  = J_FUV 
        J_LW_current   = J_LW 
        J_EUV1_current = J_EUV1
        J_EUV2_current = J_EUV2
        J_EUV3_current = J_EUV3 

        # Add the radiation pressure momentum density for this sub-step, 
        # first from gas absorption:
        nH     = X * ρ / mH
        nHe    = Y * ρ / mHe
        nH2    = xH2_current * nH 
        nHI    = max( (1 - 2 * xH2_current - xHII_current) * nH, 0.0 ) 
        nHII   = xHII_current * nH 
        nHeI   = max( (1 - xHeII_current - xHeIII_current) * nHe, 0.0 )
        nHeII  = xHeII_current * nHe
        nHeIII = xHeIII_current * nHe
        ne     = nHII  +  nHeII  +  2 * nHeIII

        # Radiation pressure acceleration from gas absorption:
        a_rad  = (1/ρ) * ( ( Four_π * J_LW_av / (c * E_LW) ) * nH2 * σH2_LW * pH2_LW    

                       +   ( Four_π * J_EUV1_av / (c * E_EUV1) ) * ( nH2 * σH2_EUV1 * pH2_EUV1  +  nHI * σHI_EUV1 * pHI_EUV1 )   

                       +   ( Four_π * J_EUV2_av / (c * E_EUV2) ) * ( nH2 * σH2_EUV2 * pH2_EUV2  +  nHI * σHI_EUV2 * pHI_EUV2
                                                                   + nHeI * σHeI_EUV2 * pHeI_EUV2 )
                                                  
                       +   ( Four_π * J_EUV3_av / (c * E_EUV3) ) * ( nH2 * σH2_EUV3 * pH2_EUV3     +  nHI * σHI_EUV3 * pHI_EUV3
                                                                  + nHeI * σHeI_EUV3 * pHeI_EUV3  +  nHeII * σHeII_EUV3 * pHeII_EUV3 ) 
                                 ) * (ρ ≥ 1.1 * ρ_floor) 

        # Add contribution from Thomson scattering in all stellar RT bands:
        Jtot_av  =  J_opt_av  +  J_FUV_av  +  J_LW_av  +  J_EUV1_av  +  J_EUV2_av  +  J_EUV3_av
        a_rad   +=  (1/ρ) * (Four_π * Jtot_av / c) * ne * σT * (ρ ≥ 1.1 * ρ_floor)

        # Now that we have the gas acceleration from stellar rad. pressure, 
        # get the momentum change for the gas:
        Δρu_rad[icell] += Δtsub * ρ * a_rad

        # Update current time:
        t_accum += Δtsub

        # Update sub-cycle time-step using the new recommended value:
        Δtsub = Δtsub_new

        # Update the step count and warn the user if it has exceeded the specified upper limit:
        step += 1
        if step > max_steps
            error("The thermochemistry sub-cycling exceeded $(max_steps) in cell $(icell). Terminating the simulation.")
        end
    end


    # Return the updated state after evolving over the global time-step:
    return e_current, xH2_current, xHII_current, xHD_current, xDII_current, xHeII_current, xHeIII_current, 
           xCI_current, xCII_current, xCO_current, xOI_current, xOII_current,
           J_opt_current, J_FUV_current, J_LW_current, J_EUV1_current, J_EUV2_current, J_EUV3_current
end


const dNband::UnitRange{Int64} = 1:6  # Interval for stellar RT bands (i.e. range from 1 to # of stellar RT bands, 6)

function UpdateStellarJandThermoChemistry!( Δtglobal::Float64, 
                                             Lum_opt::Float64,
                                             Lum_FUV::Float64,
                                              Lum_LW::Float64,
                                            Lum_EUV1::Float64,
                                            Lum_EUV2::Float64, 
                                            Lum_EUV3::Float64,
                                              Z_Zsun::Float64,
                                              ρd_Sil::AbstractMatrix{Float64},
                                     T_dust_bins_Sil::AbstractMatrix{Float64},
                                        Zgr_bins_Sil::AbstractMatrix{Float64},
                                                ρd_C::AbstractMatrix{Float64},
                                       T_dust_bins_C::AbstractMatrix{Float64},
                                          Zgr_bins_C::AbstractMatrix{Float64},
                                              ud_Sil::AbstractMatrix{Float64},
                                                ud_C::AbstractMatrix{Float64},
                                                   u::AbstractVector{Float64},
                                                fred::Float64,
                                    Δρu_rad_dust_Sil::AbstractMatrix{Float64},
                                      Δρu_rad_dust_C::AbstractMatrix{Float64},
                                             Δρu_rad::AbstractVector{Float64},
                                               J_opt::AbstractVector{Float64},
                                           J_opt_old::AbstractVector{Float64},
                                               J_FUV::AbstractVector{Float64},
                                           J_FUV_old::AbstractVector{Float64},
                                                J_LW::AbstractVector{Float64},
                                            J_LW_old::AbstractVector{Float64},
                                              J_EUV1::AbstractVector{Float64},
                                          J_EUV1_old::AbstractVector{Float64},
                                              J_EUV2::AbstractVector{Float64},
                                          J_EUV2_old::AbstractVector{Float64},
                                              J_EUV3::AbstractVector{Float64},
                                          J_EUV3_old::AbstractVector{Float64},
                                                  ΔV::AbstractVector{Float64},
                                                   r::AbstractVector{Float64},
                                                rCoM::AbstractVector{Float64},
                                               A_opt::AbstractVector{Float64},
                                               A_FUV::AbstractVector{Float64},
                                                A_LW::AbstractVector{Float64},
                                              A_EUV1::AbstractVector{Float64},
                                              A_EUV2::AbstractVector{Float64},
                                              A_EUV3::AbstractVector{Float64},
                                               B_opt::AbstractVector{Float64},
                                               B_FUV::AbstractVector{Float64},
                                                B_LW::AbstractVector{Float64},
                                              B_EUV1::AbstractVector{Float64},
                                              B_EUV2::AbstractVector{Float64},
                                              B_EUV3::AbstractVector{Float64},
                                               C_opt::AbstractVector{Float64},
                                               C_FUV::AbstractVector{Float64},
                                                C_LW::AbstractVector{Float64},
                                              C_EUV1::AbstractVector{Float64},
                                              C_EUV2::AbstractVector{Float64},
                                              C_EUV3::AbstractVector{Float64},
                                               D_opt::AbstractVector{Float64},
                                               D_FUV::AbstractVector{Float64},
                                                D_LW::AbstractVector{Float64},
                                              D_EUV1::AbstractVector{Float64},
                                              D_EUV2::AbstractVector{Float64},
                                              D_EUV3::AbstractVector{Float64},
                                                   e::AbstractVector{Float64},
                                                   ρ::AbstractVector{Float64},
                                                 xH2::AbstractVector{Float64},
                                                xHII::AbstractVector{Float64},
                                                 xHD::AbstractVector{Float64},
                                                xDII::AbstractVector{Float64},
                                               xHeII::AbstractVector{Float64},
                                              xHeIII::AbstractVector{Float64},
                                                 xCI::AbstractVector{Float64},
                                                xCII::AbstractVector{Float64},
                                                 xCO::AbstractVector{Float64},
                                                 xOI::AbstractVector{Float64},
                                                xOII::AbstractVector{Float64},
                                                N_H2::AbstractVector{Float64},
                                                N_HD::AbstractVector{Float64},
                                                N_CO::AbstractVector{Float64},
                                               ∇dotu::AbstractVector{Float64},
                                                PLyα::AbstractVector{Float64},
                                                 ζHI::Float64 = 0.0,
                                                   z::Float64 = 0.0,
                                      CoolingHeating::Bool    = true
                                              )

    # Next do a transport step for all intensities. We do this in a thread-safe 
    # manner using scratch arrays:
    @threads for band in dNband 
        # Get the thread ID for the thread-local allocations:
        tid = threadid()

        local_J, local_J_old, local_Lum    = nothing, nothing, 0.0
        local_A, local_B, local_C, local_D = nothing, nothing, nothing, nothing 

        if band == 1 
            local_J     =  J_opt 
            local_J_old =  J_opt_old 
            local_Lum   =  Lum_opt 
            local_A     =  A_opt 
            local_B     =  B_opt
            local_C     =  C_opt 
            local_D     =  D_opt 
        elseif band == 2
            local_J     =  J_FUV
            local_J_old =  J_FUV_old 
            local_Lum   =  Lum_FUV 
            local_A     =  A_FUV
            local_B     =  B_FUV 
            local_C     =  C_FUV 
            local_D     =  D_FUV 
        elseif band == 3
            local_J     =  J_LW 
            local_J_old =  J_LW_old 
            local_Lum   =  Lum_LW
            local_A     =  A_LW 
            local_B     =  B_LW 
            local_C     =  C_LW
            local_D     =  D_LW
        elseif band == 4
            local_J     =  J_EUV1 
            local_J_old =  J_EUV1_old 
            local_Lum   =  Lum_EUV1 
            local_A     =  A_EUV1 
            local_B     =  B_EUV1 
            local_C     =  C_EUV1 
            local_D     =  D_EUV1 
        elseif band == 5 
            local_J     =  J_EUV2 
            local_J_old =  J_EUV2_old 
            local_Lum   =  Lum_EUV2 
            local_A     =  A_EUV2 
            local_B     =  B_EUV2 
            local_C     =  C_EUV2 
            local_D     =  D_EUV2 
        elseif band == 6
            local_J     =  J_EUV3 
            local_J_old =  J_EUV3_old 
            local_Lum   =  Lum_EUV3 
            local_A     =  A_EUV3 
            local_B     =  B_EUV3 
            local_C     =  C_EUV3 
            local_D     =  D_EUV3 
        end 

        TransportUpdateStellarJ!( local_J, local_J_old, Δtglobal, local_Lum,
                                  ΔV, rCoM, local_A, local_B, local_C, local_D, fred )

    end # End of transport step for-loop


    # Compute H₂ and HD column densities up to cell i:
    N_H2[1]   = 0.0; N_HD[1]   = 0.0  # No self-shielding in the central cell
    N_H2_encl = 0.0; N_HD_encl = 0.0
    Nr        = length( rCoM )
    for i in 2:Nr 
        @inbounds begin 
            nH         = X * ρ[i] / mH
            Δr         = r[i] - r[i-1]
            N_H2_encl += xH2[i-1] * nH * Δr
            N_HD_encl += xHD[i-1] * D_H_ratio * nH * Δr
            N_H2[i]    = N_H2_encl
            N_HD[i]    = N_HD_encl
        end
    end

    # If the cloud is metal-enriched, compute a first estimate of
    # the CO column density from cell i to the cloud edge i = Nr.
    # This is later corrected for Doppler shift (ala Sobolev):

    dN = 1:Nr # We need this interval potentially twice
    if Z_Zsun > 0.0 
        for i in dN 
            ∫dr_nCO = 0.0
            for k in i:Nr
                @inbounds begin 
                    nCO      = xCO[k] * C_H_ratio * X * ρ[k] / mH
                    Δr       = r[k+1] - r[k]
                    ∫dr_nCO += nCO * Δr
                end
            end
            N_CO[i] = ∫dr_nCO
        end
    end

    # Next, sub-cycle for each cell:     
    @threads for i in dN
        @inbounds begin
            # Get estimated time-derivatives:
            dJ_optdt  = ( J_opt[i]  - J_opt_old[i]  ) / Δtglobal 
            dJ_FUVdt  = ( J_FUV[i]  - J_FUV_old[i]  ) / Δtglobal 
            dJ_LWdt   = ( J_LW[i]   - J_LW_old[i]   ) / Δtglobal 
            dJ_EUV1dt = ( J_EUV1[i] - J_EUV1_old[i] ) / Δtglobal
            dJ_EUV2dt = ( J_EUV2[i] - J_EUV2_old[i] ) / Δtglobal
            dJ_EUV3dt = ( J_EUV3[i] - J_EUV3_old[i] ) / Δtglobal


            # Next do thermo-chemistry sub-cycling for each cell: 
            result = ThermoChemistrySubCycleUpdate!( i, Δtglobal, e[i], ρ[i], 
                                                         xH2[i], xHII[i],
                                                         xHD[i], xDII[i],
                                                       xHeII[i], xHeIII[i], 
                                                         xCI[i], xCII[i],
                                                         xCO[i], xOI[i], xOII[i],
                                                         ρd_Sil, T_dust_bins_Sil, Zgr_bins_Sil,
                                                           ρd_C, T_dust_bins_C, Zgr_bins_C,
                                                         ud_Sil, ud_C,    u,
                                               Δρu_rad_dust_Sil, Δρu_rad_dust_C, Δρu_rad,
                                                   J_opt_old[i], dJ_optdt,
                                                   J_FUV_old[i], dJ_FUVdt,
                                                    J_LW_old[i], dJ_LWdt, 
                                                  J_EUV1_old[i], dJ_EUV1dt,
                                                  J_EUV2_old[i], dJ_EUV2dt, 
                                                  J_EUV3_old[i], dJ_EUV3dt,
                                                        N_H2[i], N_HD[i], N_CO[i],
                                                       ∇dotu[i], fred, Z_Zsun, z,
                                                        PLyα[i], ζHI, true, CoolingHeating )

            # Get updated quantities:
            e[i], xH2[i], xHII[i], xHD[i], xDII[i], xHeII[i], xHeIII[i], xCI[i], xCII[i], xCO[i], xOI[i], xOII[i],
            J_opt[i], J_FUV[i], J_LW[i], J_EUV1[i], J_EUV2[i], J_EUV3[i] = result 

        end # End of inbounds 
    end # End of i-loop

    # Update the old stellar intensities:
    @inbounds @simd for i in dN
        J_opt_old[i]  = J_opt[i]
        J_FUV_old[i]  = J_FUV[i]
        J_LW_old[i]   = J_LW[i] 
        J_EUV1_old[i] = J_EUV1[i]
        J_EUV2_old[i] = J_EUV2[i] 
        J_EUV3_old[i] = J_EUV3[i]
    end

end


end # End of module