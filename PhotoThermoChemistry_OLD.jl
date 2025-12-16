
module PhotoThermoChemistry
""" MODULE:     PhotoThermoChemistry

    PURPOSE:    Contains the full thermochemistry + dust + stellar RT update.
    
"""

__precompile__()

include( joinpath(@__DIR__, "..", "ConstantsAndFunctions.jl") )
include( joinpath(@__DIR__, "..", "Hydrodynamics.jl") )
include( joinpath(@__DIR__, "..", "AdvectJandH.jl") )
include( "./GasChemistryRates.jl" )
include( "./UpdateChemistry.jl" )
include( "./GasCoolingRates.jl" )
include( "./StellarEmissionAbsorption.jl" )
include( "./UpdateStellarJ.jl" )

using .ConstantsAndFunctions
using .Hydrodynamics
using .AdvectJandH
using .GasChemistryRates
using .UpdateChemistry
using .GasCoolingRates
using .StellarEmissionAbsorption  # For band opacities & stellar luminosities 
using .UpdateStellarJ
using  Base.Threads               # For CPU acceleration


const CFL_RT::Float64                  = 0.3   # CFL factor that determines the advection time-step for stellar & Lyα RT (Default: 0.3. NOTE: Must be < 1)
const DEBUG_PhotoThermoChemistry::Bool = true  # Allow safety checks for physical values etc. Cheap & good for tracking bugs.


# Export functions: 
export UpdateStellarJandThermoChemistry!, StellarRTArrays, 
       GenerateStellarRTArrays, FindRSL_StellarRT



const ϵ_RSL::Float64 = 10.0        # RSL is ≥ ϵ_RSL × max(|u| + c_s)
const c̃_min::Float64 = 100 * kms   # Lower bound on RSL, regardless of gas conditions 

""" This function determines a reduced speed of light c̃ for 
    stellar RT, such that it is comfortably larger than all gas 
    velocities in the simulation. """
function FindRSL_StellarRT( c̃::Float64,                   # Old reduced speed of light
                          u_r::AbstractArray{Float64, 2}, # r-direction gas velocity 
                          u_Z::AbstractArray{Float64, 2}, # Z-direction gas velocity 
                            P::AbstractArray{Float64, 2}, # Gas pressure 
                            ρ::AbstractArray{Float64, 2}, # Gas density
                          γad::AbstractArray{Float64, 2}  # Gas adiabatic index 
                           )

    N_r, N_Z = size( ρ )
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z

    c̃_new    = 0.0   # Initialize value for the new RSL 
    for j in dN_Z 
        local_max_c̃  =  c̃_new   # Local variable for tracking largest c̃

        @inbounds for i in dN_r 
            # Gas sound speed:
            c_s = sqrt( γad[i,j] * P[i,j] / ρ[i,j] )

            # Compute minimum RSL for accuracy in this cell to be ~ ϵ_RSL × gas velocity:
            c̃_min_cell = ϵ_RSL * max( abs(u_r[i,j]) + c_s,  abs(u_Z[i,j]) + c_s )

            # Update the largest RSL we've found:
            local_max_c̃ = max( local_max_c̃, c̃_min_cell )
        end

        c̃_new = max( c̃_new, local_max_c̃ )
    end

    # Set the new RSL:
    c̃_new = min( max( c̃_new, c̃_min ), c )

    # To avoid rapid changes in c̃ if the newly calculated RSL is small, 
    # we further limit the maximum reduction to 25 % from the last time-step:
    if c̃_new < c̃
        c̃_new = max( c̃_new, 0.75 * c̃ )
    end

    return c̃_new
end


""" The total gas heating rate (erg cm⁻³ s⁻ꜝ) from 
    absorption of ionizing and other photons. """
function HeatingRate_total( J_opt::Float64, 
                            J_FUV::Float64, 
                             J_LW::Float64,
                           J_EUV1::Float64,
                           J_EUV2::Float64,
                           J_EUV3::Float64,
                            T_old::Float64,
                           ne_old::Float64,
                          nHI_old::Float64,
                          nH2_old::Float64,
                         nHeI_old::Float64,
                        nHeII_old::Float64,
                    SelfShielding::Float64,
                             PLyα::Float64 = 0.0
                             )

    # TODO: Add photoelectric heating, cosmic ray heating, following LYDION-1D
    
    (  0.0 * HeatingRate_Compton(J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, ne_old, T_old, PLyα )
    +  0.0 * HeatingRate_LW( SelfShielding * J_LW, nH2_old, nHI_old, T_old )
    +  HeatingRate_EUV1( J_EUV1, nHI_old, nH2_old ) 
    +  HeatingRate_EUV2( J_EUV2, nHI_old, nH2_old, nHeI_old ) 
    +  HeatingRate_EUV3( J_EUV3, nHI_old, nH2_old, nHeI_old, nHeII_old )  )

end


const S_H2_1::Float64 = 0.965;    const S_H2_2::Float64  = 0.035
const c_SH2::Float64  = 8.5e-4; const N_H2crit::Float64  = 5e14

""" The H₂ self-shielding factor, as fitted by 
    Draine & Bertoldi (1996). """
@inline function S_H2( N_H2::Float64, 
                          T::Float64
                       )
    T    = clamp( T, 1.0, 1e5 )
    x    = N_H2 / N_H2crit
    F    = sqrt( 1 + x )
    b5   = sqrt( 2 * kB * T / mH2 ) / kms 
    S_H2 = S_H2_1 / (1 + x/b5)^2  +  S_H2_2 * exp( - c_SH2 * F ) / F
    return S_H2
end


""" The HD self-shielding factor, same fit as for H₂. """
@inline function S_HD( N_HD::Float64, 
                          T::Float64
                       )
    T    = clamp( T, 1.0, 1e5 )
    x    = N_HD / N_H2crit
    F    = sqrt( 1 + x )
    b5   = sqrt( 2 * kB * T / (3 * mH) ) / kms  # Mass of HD is ≃ 3 × mH
    S_HD = S_H2_1 / (1 + x/b5)^2  +  S_H2_2 * exp( - c_SH2 * F ) / F
    return S_HD
end


const N_H2_HDcrit = 2.34e19; const c_H2_HD = 5.2e-3; const p_H2_HD = 0.238 

""" The shielding of HD by H₂. See Wolcott-Green & Haiman (2011). """
@inline function S_H2_HD( N_H2::Float64 )
    x       = N_H2 / N_H2_HDcrit
    S_H2_HD = exp( - c_H2_HD * x ) / (1 + x)^p_H2_HD
    return S_H2_HD
end


const T_floor::Float64          = 1.0        # Enforced temperature floor in the thermochemistry sub-cycling
const T_ceiling::Float64        = 1e9        # Enforced temperature ceiling in the thermochemistry sub-cycling
const tol::Float64              = 0.1        # Maximum relative change tolerance per sub-cycle time-step 
const tol_enlargestep::Float64  = 0.5 * tol  # Recommend a 2 × larger sub-cycle time-step if relative changes were < tol_enlargestep
const ϵ_ΔT::Float64             = 1e-3       # Use step-size ϵ_ΔT * T_old to estimate ∂Λ/∂T at T = T_old
const J_floor::Float64          = 1e-10      # Maximum stellar J below which we don't care about the 10% rule 
const H_floor::Float64          = 1e-10      # Maximum stellar H below which we don't care about the 10% rule 
const xmin::Float64             = 1e-8       # Minimum abundance fraction below which we don't care about the 10% rule
const max_iter::Int             = 100        # Maximum number of adaptive refinements of sub-cycle time-step 
const Δt_e_max::Float64         = 1e6 * yr   # Maximum time-step to consider for heating/cooling
const Four_π_over_c::Float64    = Four_π / c # Convenient constant for the radiation pressure momentum computation 

""" This functions does a single sub-cycle time-step update of
    intensities, gas internal energy, and chemical abundances. """
function ThermoChemistrySubStepUpdate!( icell::Int, jcell::Int,  # Cell (i,j)
                                   Δtsubguess::Float64,          # First guess of sub-cycle time-step
                                            c̃::Float64,          # Reduced speed of light for stellar RT
                                        e_old::Float64,          # Gas internal energy density from previous sub-step 
                                            ρ::Float64,          # Gas density 
                                      xH2_old::Float64,          # H₂ fraction from previous sub-step 
                                     xHII_old::Float64,          # H II fraction from previous sub-step 
                                      xHD_old::Float64,          # HD fraction from previous sub-step 
                                     xDII_old::Float64,          # D II fraction from previous sub-step 
                                    xHeII_old::Float64,          # He II fraction from previous sub-step 
                                   xHeIII_old::Float64,          # He III fraction from previous sub-step 

                                    J_opt_old::Float64 = 0.0,
                                     dJ_optdt::Float64 = 0.0,
                                    J_FUV_old::Float64 = 0.0,
                                     dJ_FUVdt::Float64 = 0.0,

                                     J_LW_old::Float64 = 0.0,
                                      dJ_LWdt::Float64 = 0.0,
                                   H_r_LW_old::Float64 = 0.0,
                                   H_Z_LW_old::Float64 = 0.0,
                                     dHr_LWdt::Float64 = 0.0,
                                     dHZ_LWdt::Float64 = 0.0,

                                   J_EUV1_old::Float64 = 0.0,
                                    dJ_EUV1dt::Float64 = 0.0,
                                 H_r_EUV1_old::Float64 = 0.0,
                                 H_Z_EUV1_old::Float64 = 0.0,
                                   dHr_EUV1dt::Float64 = 0.0,
                                   dHZ_EUV1dt::Float64 = 0.0,

                                   J_EUV2_old::Float64 = 0.0,
                                    dJ_EUV2dt::Float64 = 0.0,
                                 H_r_EUV2_old::Float64 = 0.0,
                                 H_Z_EUV2_old::Float64 = 0.0,
                                   dHr_EUV2dt::Float64 = 0.0,
                                   dHZ_EUV2dt::Float64 = 0.0,

                                   J_EUV3_old::Float64 = 0.0,
                                    dJ_EUV3dt::Float64 = 0.0,
                                 H_r_EUV3_old::Float64 = 0.0,
                                 H_Z_EUV3_old::Float64 = 0.0,
                                   dHr_EUV3dt::Float64 = 0.0,
                                   dHZ_EUV3dt::Float64 = 0.0,

                                            z::Float64 = 0.0,
                                        CaseB::Bool    = true,
                               CoolingHeating::Bool    = true
                                      )

    # Safety checks:
    if DEBUG_PhotoThermoChemistry
        @assert isfinite(ρ) && ρ > 0.0 "ρ = $(ρ) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(e_old)        "e_old = $(e_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"

        @assert isfinite(J_EUV1_old) && J_EUV1_old ≥ 0.0 "J_EUV1_old = $(J_EUV1_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_r_EUV1_old)                   "H_r_EUV1_old = $(H_r_EUV1_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(H_Z_EUV1_old)                   "H_Z_EUV1_old = $(H_Z_EUV1_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"

        @assert isfinite(xH2_old)            "xH2_old = $(xH2_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xHII_old)           "xHII_old = $(xHII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xHD_old)            "xHD_old = $(xHD_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xDII_old)           "xDII_old = $(xDII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xHeII_old)          "xHeII_old = $(xHeII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert isfinite(xHeIII_old)         "xHeIII_old = $(xHeIII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xHII_old + 2 * xH2_old ≤ 1.0 "xHII_old + 2 * xH2_old = $(xHII_old + 2 * xH2_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xDII_old + xHD_old ≤ 1.0     "xDII_old + xHD_old = $(xDII_old + xHD_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
        @assert xHeII_old + xHeIII_old ≤ 1.0 "xHeII_old + xHeIII_old = $(xHeII_old + xHeIII_old) in cell (i,j) = ($(icell), $(jcell)). Cannot proceed!"
    end


    # Get relevant number densities & abundance fractions:
    nH_old            =   X * ρ / mH
    nD_old            =   D_H_ratio * nH_old
    nHe_old           =   Y * ρ / mHe
    # TODO: nC_old            =   C_H_ratio * nH_old
    # TODO: nO_old            =   O_H_ratio * nH_old
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
    # TODO: nCI_old           =   xCI_old    * nC_old
    # TODO: nCII_old          =   xCII_old   * nC_old
    # TODO: nOI_old           =   xOI_old    * nO_old
    # TODO: nOII_old          =   xOII_old   * nO_old
    # TODO: nCO_old           =   xCO_old    * nC_old
    # TODO: nCHx_old          =   max((1 - xCI_old  - xCII_old - xCO_old), 0.0 ) * nC_old 
    # TODO: nOHx_old          =   max((1 - xOI_old  - xOII_old - (nCO_old / nO_old)), 0.0 ) * nO_old 
    ne_old            =   nHII_old  +  nHeII_old  +  2 * nHeIII_old
    n                 =   nH_old  +  nHe_old  +  ne_old 

    # ------- Get opacities (cm⁻¹) in all bands ----------------------------------------------
    # TODO: Dust, and other gas species:

    αc_opt        = 0.0                                        
    αc_FUV        = 0.0                                         
    αc_LW         = nHI_old  * σHI_LW    +  nH2_old * σH2_LW    

    αc_EUV1       = ( nHI_old  * σHI_EUV1  +  nH2_old * σH2_EUV1  )

    αc_EUV2       = nHI_old  * σHI_EUV2  +  nH2_old * σH2_EUV2  +  nHeI_old * σHeI_EUV2  

    αc_EUV3       = ( nHI_old * σHI_EUV3      +  nH2_old * σH2_EUV3  +  nHeI_old * σHeI_EUV3 
                    + nHeII_old * σHeII_EUV3  )                                 

    # --------- Get old temperature, and enforce temperature floor ----------------------------
    γad           =   Getγad( xHII_old, xH2_old, xHeII_old, xHeIII_old )
    e_old         =   max( e_old,  n * kB * T_floor / (γad - 1) )
    T_old         =   ( γad - 1 ) * e_old  / ( n * kB )

    # ------------------------------------------------------------------------------------

    N_H2 = nH2_old * (6 * pc / 128) 

    SelfShielding       = S_H2( N_H2, T_old ) 
    #SelfShielding       = 1.0
    SelfShielding_HD    = 1.0 # TODO
    SelfShielding_H2_HD = 1.0 # TODO
    
    Δtsub        = Δtsubguess # Initialize first guess of sub-cycle time-step
    iter::Int    = 0          # Initialize iteration count
    ΔxH2_rel     = 0.0        # Initialize ΔxH2_rel, for printing in case convergence fails 
    ΔxHII_rel    = 0.0        # Initialize ΔxHII_rel, for printing in case convergence fails 
    Δne_rel      = 0.0        # Initialize Δne_rel, for printing in case convergence fails 
    Δe           = 0.0        # Initialize Δe, for printing in case convergence fails 
    while iter < max_iter 
        iter += 1

        #######################################################################
        #######                                                         #######
        #######    S T E P  1  :  U P D A T E  I N T E N S I T Y,       #######
        #######                   H E A T I N G,                        #######
        #######                   P H O T O - D E S T R U C T I O N,    ####### 
        #######                   R A D I A T I O N  P R E S S U R E    #######
        #######                                                         #######
        #######################################################################
        # First we update J and H for all stellar bands. We also compute a trial 
        # radiation pressure update (only accepted if the whole sub-step is accepted):

        Δρu_r_trial   = 0.0  # Initialize the computed change in r-direction gas momentum 
        Δρu_Z_trial   = 0.0  # Initialize the computed change in Z-direction gas momentum 

        #######################################################################
        #######                                                         #######
        #######                       O P T I C A L                     #######
        #######                                                         #######
        #######################################################################

        αc_opt_dust   = 0.0 # TODO
        
        J_opt_min     = max( ( J_opt_old  +  dJ_optdt * Δtsub ) / 
                             (1 + c̃ * αc_opt_dust * Δtsub), 0.0 )

        J_opt         = max( AbsorptionUpdateStellarJorH( J_opt_old, Δtsub, dJ_optdt, αc_opt, c̃ ),
                             J_opt_min, 
                             0.0 )

        J_opt_av      = max( AbsorptionUpdateStellarJ_average( J_opt_old, J_opt, Δtsub, dJ_optdt, αc_opt, c̃, J_opt_min ), 0.0 )


        #######################################################################
        #######                                                         #######
        #######                       F A R - U V                       #######
        #######                                                         #######
        #######################################################################

        αc_FUV_dust   = 0.0 # TODO

        J_FUV_min     = max( ( J_FUV_old  +  dJ_FUVdt * Δtsub ) / 
                             (1 + c̃ * αc_FUV_dust * Δtsub), 0.0 )  # TODO: Add term - (c̃ * E_FUV / Four_π) * (nOHx_old + C_OHx * Δtsub) )

        J_FUV         = max( AbsorptionUpdateStellarJorH( J_FUV_old, Δtsub, dJ_FUVdt, αc_FUV, c̃ ),
                             J_FUV_min, 
                             0.0 )

        J_FUV_av      = max( AbsorptionUpdateStellarJ_average( J_FUV_old, J_FUV, Δtsub, dJ_FUVdt, αc_FUV, c̃, J_FUV_min ), 0.0 )

        #######################################################################
        #######                                                         #######
        #######                 L Y M A N - W E R N E R                 #######
        #######                                                         #######
        #######################################################################

        αc_LW_dust  = 0.0 # TODO

        J_LW        = max( AbsorptionUpdateStellarJorH( J_LW_old, Δtsub, dJ_LWdt, αc_LW, c̃ ), 0.0 )
        H_r_LW      = AbsorptionUpdateStellarJorH( H_r_LW_old, Δtsub, dHr_LWdt, αc_LW, c̃ )
        H_Z_LW      = AbsorptionUpdateStellarJorH( H_Z_LW_old, Δtsub, dHZ_LWdt, αc_LW, c̃ )
        J_LW_av     = 0.5 * ( J_LW + J_LW_old )

        # Re-normalize so that H < J:
        H_LW_mag    = sqrt( H_r_LW * H_r_LW  +  H_Z_LW * H_Z_LW )
        if H_LW_mag > J_LW
            f_LW   = H_LW_mag / J_LW
            H_r_LW = H_r_LW   / f_LW
            H_Z_LW = H_Z_LW   / f_LW 
        end

        # Compute relative changes in LW J, H_r, and H_Z:
        ΔJ_LW       = abs( J_LW   - J_LW_old )   / max( J_LW_old,        J_floor )
        ΔH_r_LW     = abs( H_r_LW - H_r_LW_old ) / max( abs(H_r_LW_old), H_floor )
        ΔH_Z_LW     = abs( H_Z_LW - H_Z_LW_old ) / max( abs(H_Z_LW_old), H_floor )

        ΔLW = max( ΔJ_LW, ΔH_r_LW, ΔH_Z_LW )
        if ΔLW > tol 
            # Change in LW was too large. Reduce sub-cycle time-step & try again: 
            Δtsub = Δtsub / 2
            continue
        end

        # Compute momentum added to the gas in this sub-step:
        Δρu_r_trial  += Four_π_over_c * αc_LW * H_r_LW * Δtsub 
        Δρu_Z_trial  += Four_π_over_c * αc_LW * H_Z_LW * Δtsub 

        #######################################################################
        #######                                                         #######
        #######                         E U V 1                         #######
        #######                                                         #######
        #######################################################################

        αc_EUV1_dust  = 0.0 # TODO

        J_EUV1        = max( AbsorptionUpdateStellarJorH( J_EUV1_old, Δtsub, dJ_EUV1dt, αc_EUV1, c̃ ), 0.0 )
        H_r_EUV1      = AbsorptionUpdateStellarJorH( H_r_EUV1_old, Δtsub, dHr_EUV1dt, αc_EUV1, c̃ )
        H_Z_EUV1      = AbsorptionUpdateStellarJorH( H_Z_EUV1_old, Δtsub, dHZ_EUV1dt, αc_EUV1, c̃ )
        J_EUV1_av     = 0.5 * ( J_EUV1 + J_EUV1_old )

        # Re-normalize so that H < J:
        H_EUV1_mag    = sqrt( H_r_EUV1 * H_r_EUV1  +  H_Z_EUV1 * H_Z_EUV1 )
        if H_EUV1_mag > J_EUV1
            f_EUV1   = H_EUV1_mag / J_EUV1
            H_r_EUV1 = H_r_EUV1   / f_EUV1
            H_Z_EUV1 = H_Z_EUV1   / f_EUV1 
        end

        # Compute relative changes in EUV1 J, H_r, and H_Z:
        ΔJ_EUV1       = abs( J_EUV1   - J_EUV1_old )   / max( J_EUV1_old,        J_floor )
        ΔH_r_EUV1     = abs( H_r_EUV1 - H_r_EUV1_old ) / max( abs(H_r_EUV1_old), H_floor )
        ΔH_Z_EUV1     = abs( H_Z_EUV1 - H_Z_EUV1_old ) / max( abs(H_Z_EUV1_old), H_floor )

        ΔEUV1 = max( ΔJ_EUV1, ΔH_r_EUV1, ΔH_Z_EUV1 )
        if ΔEUV1 > tol 
            # Change in EUV1 was too large. Reduce sub-cycle time-step & try again: 
            Δtsub = Δtsub / 2
            continue
        end

        # Compute momentum added to the gas in this sub-step:
        Δρu_r_trial  += Four_π_over_c * αc_EUV1 * H_r_EUV1 * Δtsub 
        Δρu_Z_trial  += Four_π_over_c * αc_EUV1 * H_Z_EUV1 * Δtsub 

        #######################################################################
        #######                                                         #######
        #######                         E U V 2                         #######
        #######                                                         #######
        #######################################################################

        αc_EUV2_dust  = 0.0 # TODO

        J_EUV2        = max( AbsorptionUpdateStellarJorH( J_EUV2_old, Δtsub, dJ_EUV2dt, αc_EUV2, c̃ ), 0.0 )
        H_r_EUV2      = AbsorptionUpdateStellarJorH( H_r_EUV2_old, Δtsub, dHr_EUV2dt, αc_EUV2, c̃ )
        H_Z_EUV2      = AbsorptionUpdateStellarJorH( H_Z_EUV2_old, Δtsub, dHZ_EUV2dt, αc_EUV2, c̃ )
        J_EUV2_av     = 0.5 * ( J_EUV2 + J_EUV2_old )

        # Re-normalize so that H < J:
        H_EUV2_mag    = sqrt( H_r_EUV2 * H_r_EUV2  +  H_Z_EUV2 * H_Z_EUV2 )
        if H_EUV2_mag > J_EUV2
            f_EUV2   = H_EUV2_mag / J_EUV2
            H_r_EUV2 = H_r_EUV2   / f_EUV2
            H_Z_EUV2 = H_Z_EUV2   / f_EUV2 
        end

        # Compute relative changes in EUV2 J, H_r, and H_Z:
        ΔJ_EUV2       = abs( J_EUV2   - J_EUV2_old )   / max( J_EUV2_old,        J_floor )
        ΔH_r_EUV2     = abs( H_r_EUV2 - H_r_EUV2_old ) / max( abs(H_r_EUV2_old), H_floor )
        ΔH_Z_EUV2     = abs( H_Z_EUV2 - H_Z_EUV2_old ) / max( abs(H_Z_EUV2_old), H_floor )

        ΔEUV2 = max( ΔJ_EUV2, ΔH_r_EUV2, ΔH_Z_EUV2 )
        if ΔEUV2 > tol 
            # Change in EUV2 was too large. Reduce sub-cycle time-step & try again: 
            Δtsub = Δtsub / 2
            continue
        end

        # Compute momentum added to the gas in this sub-step:
        Δρu_r_trial  += Four_π_over_c * αc_EUV2 * H_r_EUV2 * Δtsub 
        Δρu_Z_trial  += Four_π_over_c * αc_EUV2 * H_Z_EUV2 * Δtsub 

        #######################################################################
        #######                                                         #######
        #######                         E U V 3                         #######
        #######                                                         #######
        #######################################################################

        αc_EUV3_dust  = 0.0 # TODO

        J_EUV3        = max( AbsorptionUpdateStellarJorH( J_EUV3_old, Δtsub, dJ_EUV3dt, αc_EUV3, c̃ ), 0.0 )
        H_r_EUV3      = AbsorptionUpdateStellarJorH( H_r_EUV3_old, Δtsub, dHr_EUV3dt, αc_EUV3, c̃ )
        H_Z_EUV3      = AbsorptionUpdateStellarJorH( H_Z_EUV3_old, Δtsub, dHZ_EUV3dt, αc_EUV3, c̃ )
        J_EUV3_av     = 0.5 * ( J_EUV3 + J_EUV3_old )

        # Re-normalize so that H < J:
        H_EUV3_mag    = sqrt( H_r_EUV3 * H_r_EUV3  +  H_Z_EUV3 * H_Z_EUV3 )
        if H_EUV3_mag > J_EUV3
            f_EUV3   = H_EUV3_mag / J_EUV3
            H_r_EUV3 = H_r_EUV3 / f_EUV3
            H_Z_EUV3 = H_Z_EUV3 / f_EUV3 
        end

        # Compute relative changes in EUV2 J, H_r, and H_Z:
        ΔJ_EUV3       = abs( J_EUV3   - J_EUV3_old )   / max( J_EUV3_old,        J_floor )
        ΔH_r_EUV3     = abs( H_r_EUV3 - H_r_EUV3_old ) / max( abs(H_r_EUV3_old), H_floor )
        ΔH_Z_EUV3     = abs( H_Z_EUV3 - H_Z_EUV3_old ) / max( abs(H_Z_EUV3_old), H_floor )

        ΔEUV3 = max( ΔJ_EUV3, ΔH_r_EUV3, ΔH_Z_EUV3 )
        if ΔEUV3 > tol 
            # Change in EUV3 was too large. Reduce sub-cycle time-step & try again: 
            Δtsub = Δtsub / 2
            continue
        end

        #######################################################################
        #######                                                         #######
        #######                    L Y M A N - α                        #######
        #######                                                         #######
        #######################################################################

        #for k in dNbin
        #    J_Lyα[k] .= max( AbsorptionUpdateStellarJorH( J_Lyα_old[k], Δtsub, dJ_EUV3dt, αc_EUV3, c̃ ), 0.0 )
        #end



        # Compute momentum added to the gas in this sub-step:
        Δρu_r_trial  += Four_π_over_c * αc_EUV3 * H_r_EUV3 * Δtsub 
        Δρu_Z_trial  += Four_π_over_c * αc_EUV3 * H_Z_EUV3 * Δtsub 

        # Get maximum change among all bands (for later determination of next sub-cycle time-step):
        ΔStellarRT  = max( ΔLW, ΔEUV1, ΔEUV2, ΔEUV3 )

        # Get heating rate (from photoelectric heating, Compton heating off the (non-CMB) radiation field,
        # photo-ionizations/dissociations, UV pumping, and cosmic ray heating):
        HeatingRate = HeatingRate_total( J_opt_av, J_FUV_av, J_LW_av, J_EUV1_av, J_EUV2_av, J_EUV3_av,
                                         T_old, ne_old, nHI_old, nH2_old, nHeI_old, nHeII_old, 
                                         SelfShielding ) 

        # Get photo-ionization/dissociation rates (s⁻¹):
        Γ_LW        =  PhotoDissRate_H2( SelfShielding * J_LW_av )
        Γpd         =  PhotoDetachRate_Hminus( J_opt_av, J_FUV_av, J_LW_av, J_EUV1_av, J_EUV2_av, J_EUV3_av )
        Γ_LW_HD     =  PhotoDissRate_HD( SelfShielding_HD * SelfShielding_H2_HD * J_LW_av )
        ΓionH2      =  IonRate_H2( J_EUV1_av, J_EUV2_av, J_EUV3_av )  # TODO: +  ζHeI(ζHI)+  ζH2(ζHI)
        ΓionHI      =  IonRate_HI( J_EUV1_av, J_EUV2_av, J_EUV3_av )  # TODO: +  ζHeI(ζHI)+  ζHI
        ΓionDI      =  ΓionHI
        ΓionHeI     =  IonRate_HeI( J_EUV2_av, J_EUV3_av )  # TODO: +  ζHeI(ζHI)
        ΓionHeII    =  IonRate_HeII( J_EUV3_av )

        # TODO: ΓionCI      =  IonRate_CI( J_LW_av, J_EUV1_av, J_EUV2_av, J_EUV3_av )  +  ζCI(ζHI)
        # TODO: ΓionOI      =  IonRate_OI( J_EUV1_av, J_EUV2_av, J_EUV3_av )
        # TODO: ΓdissCHx    =  PhotoDissRate_CHₓ( J_opt_av, J_FUV_av, J_LW_av )
        # TODO: ΓdissOHx    =  PhotoDissRate_OHₓ( J_FUV_av, J_LW_av )
        # TODO: ΓdissCO     =  PhotoDissRate_CO( J_LW_av )

        # TODO: Update the dust temperatures:

        # TODO: Update the dust charges:

        ################################################################
        #######                                                  #######
        #######    S T E P  2  :  U P D A T E  I N T E R N A L   #######
        #######                          E N E R G Y             #######
        #######                                                  #######
        ################################################################
        # In this step we try to update the internal energy e, using 
        # a linearized backward Euler solver. 

        # Compute the cooling rate at e_old:
        CoolingRate = Λ( T_old, nHI_old, nHII_old, nHeI_old, nHeII_old, nHeIII_old, ne_old,
                         nH2_old, nHD_old, z, CaseB )

        # Compute ∂Λ/∂e using a 4th-order central difference approximation:
        ΔT   = ϵ_ΔT * T_old
        T_p2 = T_old  +  2 * ΔT
        T_p1 = T_old  +  ΔT 
        T_m1 = T_old  -  ΔT
        T_m2 = T_old  -  2 * ΔT  

        CoolingRate_p2 = Λ( T_p2, nHI_old, nHII_old, nHeI_old, nHeII_old, nHeIII_old, ne_old,
                            nH2_old, nHD_old, z, CaseB )

        CoolingRate_p1 = Λ( T_p1, nHI_old, nHII_old, nHeI_old, nHeII_old, nHeIII_old, ne_old,
                            nH2_old, nHD_old, z, CaseB )

        CoolingRate_m1 = Λ( T_m1, nHI_old, nHII_old, nHeI_old, nHeII_old, nHeIII_old, ne_old,
                            nH2_old, nHD_old, z, CaseB )

        CoolingRate_m2 = Λ( T_m2, nHI_old, nHII_old, nHeI_old, nHeII_old, nHeIII_old, ne_old,
                            nH2_old, nHD_old, z, CaseB )

        # Put it all together:
        ∂CoolingRate∂T = ( - CoolingRate_p2  +  8 * CoolingRate_p1  -  8 * CoolingRate_m1  
                           + CoolingRate_m2 ) / ( 12 * ΔT )

        ∂CoolingRate∂e = ∂CoolingRate∂T * ( γad - 1 ) / ( n * kB )  # Convert from ∂/∂T to ∂/∂e

        # Linearized backward-Euler update of e:
        e_new = e_old  +  ( HeatingRate - CoolingRate ) * Δtsub / ( 1  +  ∂CoolingRate∂e * Δtsub )

        # ------- STABILITY & ACCURACY CHECK FOR TEMPERATURE UPDATE -------------------

        # Check that the relative change in e was sufficiently small, and that 
        # other stability/accuracy checks are satisfied:
        Δe    = abs( e_new  -  e_old ) / e_old

        # Tolerance cooling/heating time-scale for accuracy purposes:
        Δt_e  = min( tol * e_old / abs( HeatingRate - CoolingRate ), Δt_e_max )

        if Δe > tol || e_new ≤ 0.0 || !isfinite(e_new) || Δtsub > Δt_e
            Δtsub = min( Δtsub / 2,  
                         Δt_e )
            continue 
        end

        # Even if the change in e was < 10%, we also check the residual, so that
        # our linearized solution actually approximates a fully implicit BDF1 solution.

        # The would-be new temperature:
        T_new            = ( γad - 1 ) * e_new  / ( n * kB )     
        
        # Cooling rate at would-be new temperature:
        CoolingRate_new  = Λ( T_new, nHI_old, nHII_old, nHeI_old, nHeII_old, nHeIII_old, ne_old,
                              nH2_old, nHD_old, z, CaseB )

        # Relative residual checking |(e_old + (Γ - Λ(e_new)*Δtsub) - e_new)| / |e_new - e_old| :
        e_floor          = n * kB * T_floor / (γad - 1)   # e corresponding to temperature floor
        Res_e            = abs( e_old  +  ( HeatingRate  -  CoolingRate_new ) * Δtsub  -  e_new ) / max( abs( e_new - e_old ), e_floor ) 

        if Res_e > tol 
            Δtsub = min( Δtsub / 2,  
                         Δt_e )
            continue 
        end

        # Correct so that the temperature is between the ceiling and floor temperatures:
        if CoolingHeating 
            e_new = min( max( e_new, e_floor ),
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
        
        # TODO 

        ##################################################################
        #######                                                    #######
        #######    S T E P  4  :  U P D A T E  C H E M I S T R Y   #######
        #######                                                    #######
        ##################################################################
        # Try update chemistry & check if relative changes is < 10%:

        Tnorm = e_new * mH / ( kB * ρ )

        ( xH2_new, xHI_new, xHII_new, xHD_new, xDI_new, xDII_new,
          xHeI_new, xHeII_new, xHeIII_new, ne_new ) =
          UpdateChemistrySubStep( Δtsub, Tnorm, ρ, xH2_old, xHII_old, xHD_old, xDII_old, xHeII_old, xHeIII_old, 
                                  ΓionHI, ΓionH2, ΓionDI, ΓionHeI, ΓionHeII, Γ_LW, Γpd, Γ_LW_HD, 
                                  z, CaseB ) 

        # Compute would-be (symmetric) relative change in H I, H II, and
        # e⁻ fractions/number densities:
        ΔxH2_rel  = abs( xH2_new - xH2_old )   / max( 0.5 * (xH2_old + xH2_new),   xmin )
        ΔxHII_rel = abs( xHII_new - xHII_old ) / max( 0.5 * (xHII_old + xHII_new), xmin )
        Δne_rel   = abs( ne_new - ne_old )     / max( 0.5 * (ne_old + ne_new),     xmin * nH_old )

        # Get the maximum relative change among them:
        Δx_rel    = maximum( (ΔxH2_rel, ΔxHII_rel, Δne_rel) )

        if Δx_rel > tol 
            # Change in H I, H II, and/or e⁻ 
            # fraction was too high, reduce Δtsub & retry:
            Δtsub = Δtsub / 2

            continue    
        end

        # Recommended next sub-cycle time-step based on the relative change 
        # in quantities:
        if ( Δx_rel < tol_enlargestep  &&  Δe < tol_enlargestep  &&  Res_e < tol_enlargestep  &&  ΔStellarRT < tol_enlargestep )
            Δtsub_new = 2 * min( Δtsub, Δt_e )
        else
            Δtsub_new = Δtsub
        end

        # Get flux opacities (for use in the M1 transport, to get the asymptotic diffusion limit)
        nHI   = xHI_new   * nH_old 
        nH2   = xH2_new   * nH_old 
        nHeI  = xHeI_new  * nHe_old
        nHeII = xHeII_new * nHe_old

        αc_F_EUV1  = nHI * σHI_EUV1  +  nH2 * σH2_EUV1 
        αc_F_EUV2  = nHI * σHI_EUV2  +  nH2 * σH2_EUV2  +  nHeI * σHeI_EUV2  
        αc_F_EUV3  = nHI * σHI_EUV3  +  nH2 * σH2_EUV3  +  nHeI * σHeI_EUV3  +  nHeII * σHeII_EUV3 
        
        # Return every updated quantity, the time-step taken, & recommended next sub-cycle time-step:
        return Δtsub, Δtsub_new, e_new, xH2_new, xHII_new, xHD_new, xDII_new, xHeII_new, xHeIII_new, 
               J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, 
               H_r_LW, H_Z_LW,  H_r_EUV1, H_Z_EUV1,  H_r_EUV2, H_Z_EUV2,  H_r_EUV3, H_Z_EUV3,
               αc_F_EUV1, αc_F_EUV2, αc_F_EUV3, 
               Δρu_r_trial, Δρu_Z_trial

    end

    # Otherwise print error:
    error( """ ThermoChemistrySubStepUpdate could not converge within $(max_iter) iterations in cell (i,j) = ($(icell),$(jcell)),
               and simulation has therefore terminated.
               
               Relative changes in quantities in this cell during the final attempted sub-cycle time-step were:
               
               - Relative change in H₂ abundance, ΔxH2_rel   = $(ΔxH2_rel)
               - Relative change in H II abundance, ΔxHII_rel = $(ΔxHII_rel)
               - Relative change in e, Δe_rel = $(Δe) """ )

end # End of ThermoChemistrySubStepUpdate!



max_steps::Int64 = 1000000 # Maximum number of sub-cycling sub-steps per global time-step (default = 10⁶)

""" The full thermochemistry sub-cycling over a global time-step Δtglobal.
    Updates the intensities, gas internal energy, and chemical abundances.
    
    **ARGUMENTS:** 
    
    - 
    
    
    """
@views function ThermoChemistrySubCycleUpdate!(  icell::Int, jcell::Int,
                                              Δtglobal::Float64,
                                                     c̃::Float64,
                                                 e_old::Float64,
                                                     ρ::Float64,
                                               xH2_old::Float64,
                                              xHII_old::Float64, 
                                               xHD_old::Float64,
                                              xDII_old::Float64,
                                             xHeII_old::Float64,
                                            xHeIII_old::Float64,

                                             Δρu_r_rad::AbstractArray{Float64, 2},
                                             Δρu_Z_rad::AbstractArray{Float64, 2},
                                               a_r_rad::AbstractArray{Float64, 2},
                                               a_Z_rad::AbstractArray{Float64, 2},
                                               αF_EUV1::AbstractArray{Float64, 2},
                                               αF_EUV2::AbstractArray{Float64, 2},
                                               αF_EUV3::AbstractArray{Float64, 2},

                                          J_opt_transp::Float64 = 0.0,
                                              dJ_optdt::Float64 = 0.0,
                                        H_r_opt_transp::Float64 = 0.0,
                                        H_Z_opt_transp::Float64 = 0.0,
                                             dHr_optdt::Float64 = 0.0,
                                             dHZ_optdt::Float64 = 0.0,

                                          J_FUV_transp::Float64 = 0.0,
                                              dJ_FUVdt::Float64 = 0.0,
                                        H_r_FUV_transp::Float64 = 0.0,
                                        H_Z_FUV_transp::Float64 = 0.0,
                                             dHr_FUVdt::Float64 = 0.0,
                                             dHZ_FUVdt::Float64 = 0.0,

                                           J_LW_transp::Float64 = 0.0,
                                               dJ_LWdt::Float64 = 0.0,
                                         H_r_LW_transp::Float64 = 0.0,
                                         H_Z_LW_transp::Float64 = 0.0,
                                              dHr_LWdt::Float64 = 0.0,
                                              dHZ_LWdt::Float64 = 0.0,

                                         J_EUV1_transp::Float64 = 0.0,
                                             dJ_EUV1dt::Float64 = 0.0,
                                       H_r_EUV1_transp::Float64 = 0.0,
                                       H_Z_EUV1_transp::Float64 = 0.0,
                                            dHr_EUV1dt::Float64 = 0.0,
                                            dHZ_EUV1dt::Float64 = 0.0,

                                         J_EUV2_transp::Float64 = 0.0,
                                             dJ_EUV2dt::Float64 = 0.0,
                                       H_r_EUV2_transp::Float64 = 0.0,
                                       H_Z_EUV2_transp::Float64 = 0.0,
                                            dHr_EUV2dt::Float64 = 0.0,
                                            dHZ_EUV2dt::Float64 = 0.0,

                                         J_EUV3_transp::Float64 = 0.0,
                                             dJ_EUV3dt::Float64 = 0.0,
                                       H_r_EUV3_transp::Float64 = 0.0,
                                       H_Z_EUV3_transp::Float64 = 0.0,
                                            dHr_EUV3dt::Float64 = 0.0,
                                            dHZ_EUV3dt::Float64 = 0.0,

                                                     z::Float64 = 0.0,
                                                 CaseB::Bool    = true,
                                        CoolingHeating::Bool    = true
                                              )
    # Initialize time, internal energy, and abundance fractions
    # using the ones from the previous global time-step:
    t_accum          = 0.0 
    e_current        = e_old 
    xH2_current      = xH2_old
    xHII_current     = xHII_old
    xHD_current      = xHD_old 
    xDII_current     = xDII_old
    xHeII_current    = xHeII_old
    xHeIII_current   = xHeIII_old
    # Similarly, initialize the intensities as those from 
    # the transport step:
    J_opt_current    = J_opt_transp
    J_FUV_current    = J_FUV_transp
    J_LW_current     = J_LW_transp
    J_EUV1_current   = J_EUV1_transp
    J_EUV2_current   = J_EUV2_transp
    J_EUV3_current   = J_EUV3_transp
    # Initialize H components:
    H_r_LW_current   =  H_r_LW_transp
    H_Z_LW_current   =  H_Z_LW_transp

    H_r_EUV1_current =  H_r_EUV1_transp
    H_Z_EUV1_current =  H_Z_EUV1_transp

    H_r_EUV2_current =  H_r_EUV2_transp
    H_Z_EUV2_current =  H_Z_EUV2_transp

    H_r_EUV3_current =  H_r_EUV3_transp
    H_Z_EUV3_current =  H_Z_EUV3_transp

    # Initialize cell flux opacities:
    αc_F_EUV1_current = 0.0
    αc_F_EUV2_current = 0.0
    αc_F_EUV3_current = 0.0


    # Initialize the momentum density change from 
    # stellar radiation pressure:
    Δρu_r_rad[icell, jcell] = 0.0  # r-direction momentum change in gas 
    Δρu_Z_rad[icell, jcell] = 0.0  # Z-direction momentum change in gas 
    
    # Start with the global time-step as an initial guess
    # for the sub-cycle time-step:
    Δtsub = Δtglobal

    step  = 0 # Initialize step count
    while t_accum < Δtglobal 
        # Ensure we don't overshoot the global time-step:
        Δtsub  = min( Δtsub, Δtglobal - t_accum )

        # Call sub-cycle update for this sub-step:
        result =  ThermoChemistrySubStepUpdate!( icell, jcell, Δtsub, c̃,
                  e_current, ρ, 
                  xH2_current, xHII_current, xHD_current, xDII_current,
                  xHeII_current, xHeIII_current, 

                  J_opt_current, dJ_optdt, 
                  J_FUV_current, dJ_FUVdt,

                  J_LW_current, dJ_LWdt,
                  H_r_LW_current, H_Z_LW_current, dHr_LWdt, dHZ_LWdt,

                  J_EUV1_current,   dJ_EUV1dt, 
                  H_r_EUV1_current, H_Z_EUV1_current, dHr_EUV1dt, dHZ_EUV1dt,

                  J_EUV2_current, dJ_EUV2dt, 
                  H_r_EUV2_current, H_Z_EUV2_current, dHr_EUV2dt, dHZ_EUV2dt,

                  J_EUV3_current, dJ_EUV3dt,
                  H_r_EUV3_current, H_Z_EUV3_current, dHr_EUV3dt, dHZ_EUV3dt,

                  z, CaseB, CoolingHeating )

        # Unpack result:
        Δtsub, Δtsub_new, e_new, xH2_new, xHII_new, xHD_new, xDII_new, xHeII_new, xHeIII_new, 
        J_opt, J_FUV, J_LW, J_EUV1, J_EUV2, J_EUV3, 
        H_r_LW, H_Z_LW,  H_r_EUV1, H_Z_EUV1,  H_r_EUV2, H_Z_EUV2,  H_r_EUV3, H_Z_EUV3,
        αc_F_EUV1, αc_F_EUV2, αc_F_EUV3,
        Δρu_r_substep, Δρu_Z_substep  = result

        # Update our current internal energy & abundance fractions:
        e_current        = e_new
        xH2_current      = xH2_new
        xHII_current     = xHII_new
        xHD_current      = xHD_new 
        xDII_current     = xDII_new
        xHeII_current    = xHeII_new
        xHeIII_current   = xHeIII_new
        # Update our current intensities:
        J_opt_current    = J_opt
        J_FUV_current    = J_FUV 
        J_LW_current     = J_LW 
        J_EUV1_current   = J_EUV1
        J_EUV2_current   = J_EUV2
        J_EUV3_current   = J_EUV3 
        # Update our current H-components:
        H_r_LW_current   = H_r_LW
        H_Z_LW_current   = H_Z_LW 
        H_r_EUV1_current = H_r_EUV1
        H_Z_EUV1_current = H_Z_EUV1
        H_r_EUV2_current = H_r_EUV2 
        H_Z_EUV2_current = H_Z_EUV2 
        H_r_EUV3_current = H_r_EUV3
        H_Z_EUV3_current = H_Z_EUV3

        # Update cell flux opacities:
        αc_F_EUV1_current = αc_F_EUV1
        αc_F_EUV2_current = αc_F_EUV2
        αc_F_EUV3_current = αc_F_EUV3

        # Add radiation pressure momentum from this sub-step: 
        Δρu_r_rad[icell, jcell] += Δρu_r_substep
        Δρu_Z_rad[icell, jcell] += Δρu_Z_substep

        # Update current time:
        t_accum += Δtsub

        # Update sub-cycle time-step using the new recommended value:
        Δtsub = Δtsub_new

        # Update the step count and warn the user if it has exceeded the specified upper limit:
        step += 1
        if step > max_steps
            error(""" The thermochemistry sub-cycling exceeded $(max_steps) steps in cell ($(icell), $(jcell)). 
                      Terminating the simulation. """)
        end
    end

    # Save latest flux opacities:
    αF_EUV1[icell, jcell] = αc_F_EUV1_current
    αF_EUV2[icell, jcell] = αc_F_EUV2_current
    αF_EUV3[icell, jcell] = αc_F_EUV3_current

    # Get the average radiation pressure acceleration:
    a_r_rad[icell, jcell] = Δρu_r_rad[icell, jcell] / ( ρ * Δtglobal )
    a_Z_rad[icell, jcell] = Δρu_Z_rad[icell, jcell] / ( ρ * Δtglobal )


    # Return the updated state after evolving over the global time-step:
    return e_current, xH2_current, xHII_current, xHD_current, xDII_current, xHeII_current, xHeIII_current, 
           J_opt_current, J_FUV_current, J_LW_current, J_EUV1_current, J_EUV2_current, J_EUV3_current,
           H_r_LW_current, H_Z_LW_current,  H_r_EUV1_current, H_Z_EUV1_current,  
           H_r_EUV2_current, H_Z_EUV2_current,  H_r_EUV3_current, H_Z_EUV3_current
end


""" A struct for organizing stellar RT arrays. """
struct StellarRTArrays
    # Intensities (latest & old) for all bands: 
    J_opt::Array{Float64, 2};     J_opt_old::Array{Float64, 2};  J_opt_stage::Array{Float64, 2}
    J_FUV::Array{Float64, 2};     J_FUV_old::Array{Float64, 2};  J_FUV_stage::Array{Float64, 2}
    J_LW::Array{Float64, 2};      J_LW_old::Array{Float64, 2};   J_LW_stage::Array{Float64, 2}
    J_EUV1::Array{Float64, 2};    J_EUV1_old::Array{Float64, 2}; J_EUV1_stage::Array{Float64, 2}
    J_EUV2::Array{Float64, 2};    J_EUV2_old::Array{Float64, 2}; J_EUV2_stage::Array{Float64, 2}
    J_EUV3::Array{Float64, 2};    J_EUV3_old::Array{Float64, 2}; J_EUV3_stage::Array{Float64, 2}

    # H ≡ ∫dΩ I n̂ / 4π arrays for all bands:
    H_r_opt::Array{Float64, 2};        H_Z_opt::Array{Float64, 2}
    H_r_opt_old::Array{Float64, 2};    H_Z_opt_old::Array{Float64, 2}
    H_r_opt_stage::Array{Float64, 2};  H_Z_opt_stage::Array{Float64, 2}

    H_r_FUV::Array{Float64, 2};        H_Z_FUV::Array{Float64, 2}
    H_r_FUV_old::Array{Float64, 2};    H_Z_FUV_old::Array{Float64, 2}
    H_r_FUV_stage::Array{Float64, 2};  H_Z_FUV_stage::Array{Float64, 2}

    H_r_LW::Array{Float64, 2};         H_Z_LW::Array{Float64, 2}
    H_r_LW_old::Array{Float64, 2};     H_Z_LW_old::Array{Float64, 2}
    H_r_LW_stage::Array{Float64, 2};   H_Z_LW_stage::Array{Float64, 2}

    H_r_EUV1::Array{Float64, 2};       H_Z_EUV1::Array{Float64, 2}
    H_r_EUV1_old::Array{Float64, 2};   H_Z_EUV1_old::Array{Float64, 2}
    H_r_EUV1_stage::Array{Float64, 2}; H_Z_EUV1_stage::Array{Float64, 2}

    H_r_EUV2::Array{Float64, 2};       H_Z_EUV2::Array{Float64, 2}
    H_r_EUV2_old::Array{Float64, 2};   H_Z_EUV2_old::Array{Float64, 2}
    H_r_EUV2_stage::Array{Float64, 2}; H_Z_EUV2_stage::Array{Float64, 2}

    H_r_EUV3::Array{Float64, 2};       H_Z_EUV3::Array{Float64, 2}
    H_r_EUV3_old::Array{Float64, 2};   H_Z_EUV3_old::Array{Float64, 2}
    H_r_EUV3_stage::Array{Float64, 2}; H_Z_EUV3_stage::Array{Float64, 2}

    # Eddington tensor componens for all bands:
    D_rr_opt::Array{Float64, 2};  D_ZZ_opt::Array{Float64, 2};  D_rZ_opt::Array{Float64, 2}
    D_rr_FUV::Array{Float64, 2};  D_ZZ_FUV::Array{Float64, 2};  D_rZ_FUV::Array{Float64, 2}
    D_rr_LW::Array{Float64, 2};   D_ZZ_LW::Array{Float64, 2};   D_rZ_LW::Array{Float64, 2}
    D_rr_EUV1::Array{Float64, 2}; D_ZZ_EUV1::Array{Float64, 2}; D_rZ_EUV1::Array{Float64, 2}
    D_rr_EUV2::Array{Float64, 2}; D_ZZ_EUV2::Array{Float64, 2}; D_rZ_EUV2::Array{Float64, 2}
    D_rr_EUV3::Array{Float64, 2}; D_ZZ_EUV3::Array{Float64, 2}; D_rZ_EUV3::Array{Float64, 2}

    # ϕϕ-component of radiation pressure tensor K = D × J at interfaces, 
    # needed for geometric source term:
    K_ϕϕ_iphalf_LW::Array{Float64, 2};     K_ϕϕ_imhalf_LW::Array{Float64, 2}
    K_ϕϕ_iphalf_EUV1::Array{Float64, 2};   K_ϕϕ_imhalf_EUV1::Array{Float64, 2}
    K_ϕϕ_iphalf_EUV2::Array{Float64, 2};   K_ϕϕ_imhalf_EUV2::Array{Float64, 2}
    K_ϕϕ_iphalf_EUV3::Array{Float64, 2};   K_ϕϕ_imhalf_EUV3::Array{Float64, 2}

    # Flux components for all bands: 
    F_r_opt::Array{Float64, 2};     F_Z_opt::Array{Float64, 2}
    F_r_Hr_opt::Array{Float64, 2};  F_Z_Hr_opt::Array{Float64, 2}
    F_r_HZ_opt::Array{Float64, 2};  F_Z_HZ_opt::Array{Float64, 2}

    F_r_FUV::Array{Float64, 2};     F_Z_FUV::Array{Float64, 2}
    F_r_Hr_FUV::Array{Float64, 2};  F_Z_Hr_FUV::Array{Float64, 2}
    F_r_HZ_FUV::Array{Float64, 2};  F_Z_HZ_FUV::Array{Float64, 2}

    F_r_LW::Array{Float64, 2};      F_Z_LW::Array{Float64, 2}
    F_r_Hr_LW::Array{Float64, 2};   F_Z_Hr_LW::Array{Float64, 2}
    F_r_HZ_LW::Array{Float64, 2};   F_Z_HZ_LW::Array{Float64, 2}

    F_r_EUV1::Array{Float64, 2};    F_Z_EUV1::Array{Float64, 2}
    F_r_Hr_EUV1::Array{Float64, 2}; F_Z_Hr_EUV1::Array{Float64, 2}
    F_r_HZ_EUV1::Array{Float64, 2}; F_Z_HZ_EUV1::Array{Float64, 2}

    F_r_EUV2::Array{Float64, 2};    F_Z_EUV2::Array{Float64, 2}
    F_r_Hr_EUV2::Array{Float64, 2}; F_Z_Hr_EUV2::Array{Float64, 2}
    F_r_HZ_EUV2::Array{Float64, 2}; F_Z_HZ_EUV2::Array{Float64, 2}

    F_r_EUV3::Array{Float64, 2};    F_Z_EUV3::Array{Float64, 2}
    F_r_Hr_EUV3::Array{Float64, 2}; F_Z_Hr_EUV3::Array{Float64, 2}
    F_r_HZ_EUV3::Array{Float64, 2}; F_Z_HZ_EUV3::Array{Float64, 2}

    # Flux opacities for all bands:
    α_F_opt::Array{Float64, 2}
    α_F_FUV::Array{Float64, 2}
    α_F_LW::Array{Float64, 2}
    α_F_EUV1::Array{Float64, 2}
    α_F_EUV2::Array{Float64, 2}
    α_F_EUV3::Array{Float64, 2}

    # Stellar luminosities in all bands:
    Lum_opt::Float64
    Lum_FUV::Float64
    Lum_LW::Float64
    Lum_EUV1::Float64
    Lum_EUV2::Float64
    Lum_EUV3::Float64

    # Emissivities for all bands:
    js_opt::Array{Float64, 2}
    js_FUV::Array{Float64, 2}
    js_LW::Array{Float64, 2}
    js_EUV1::Array{Float64, 2}
    js_EUV2::Array{Float64, 2}
    js_EUV3::Array{Float64, 2}

    # Arrays for tracking gas momentum changes from radiation pressure:
    Δρu_r_rad::Array{Float64, 2}
    Δρu_Z_rad::Array{Float64, 2}
    a_r_rad::Array{Float64, 2}
    a_Z_rad::Array{Float64, 2}

end



function UpdateStellarJandThermoChemistry!( Δt_RT_max::Float64, 
                                                    c̃::Float64,
                                                   RT::StellarRTArrays,
                                                    e::AbstractArray{Float64, 2},
                                                    ρ::AbstractArray{Float64, 2},
                                                  xH2::AbstractArray{Float64, 2},
                                                 xHII::AbstractArray{Float64, 2},
                                                  xHD::AbstractArray{Float64, 2},
                                                 xDII::AbstractArray{Float64, 2},
                                                xHeII::AbstractArray{Float64, 2},
                                               xHeIII::AbstractArray{Float64, 2},
                                                    z::Float64,
                                       CoolingHeating::Bool,

                                              # Geometry:
                                               inv_ΔV::AbstractArray{Float64, 2},
                                                  A_r::AbstractArray{Float64, 2},
                                                  A_Z::AbstractArray{Float64, 2},
                                                    r::AbstractArray{Float64, 1},
                                                 rCoM::AbstractArray{Float64, 1},
                                                    Z::AbstractArray{Float64, 1},
                                                 ZCoM::AbstractArray{Float64, 1},
                                           inv_r_cell::AbstractArray{Float64, 1},
                                     rfirst_StellarRT::Bool

                                              )

    Δt_CFL_RT = GetRHDCFLTimeStep( r, Z, c̃, CFL_RT ) 
    Δt_RT     = min( Δt_CFL_RT, Δt_RT_max )  
    c̃Δt       = c̃ * Δt_RT    
    
    #######################################################################
    #######                                                         #######
    #######       S T E P  1  :  I N J E C T I O N  S T E P         #######
    #######                                                         #######
    #######################################################################

    # Stellar emission update: 
    #RT.J_LW       .= RT.J_LW    .+  0.5 .* RT.js_LW   .* c̃ .* Δt_RT
    #RT.J_EUV1     .= RT.J_EUV1  .+  0.5 .* RT.js_EUV1 .* c̃ .* Δt_RT
    #RT.J_EUV2     .= RT.J_EUV2  .+  0.5 .* RT.js_EUV2 .* c̃ .* Δt_RT
    #RT.J_EUV3     .= RT.J_EUV3  .+  0.5 .* RT.js_EUV3 .* c̃ .* Δt_RT


    #######################################################################
    #######                                                         #######
    #######       S T E P  2  :  T R A N S P O R T  S T E P         #######
    #######                                                         #######
    #######################################################################

    # Next we advect the intensities and fluxes for all bands,
    # using the M1 moment method.

    UpdateJold = false # Don't update Jold, since we want the pre-(injection + transport) value to compute dJ/dt

    # Advect LW photons:
    AdvectJandH!( c̃Δt,   RT.js_LW,   RT.α_F_LW,  RT.J_LW,     RT.J_LW_old,   RT.H_r_LW,   RT.H_Z_LW, 
                  RT.J_LW_stage,     RT.H_r_LW_stage,   RT.H_Z_LW_stage,
                  RT.D_rr_LW,        RT.D_ZZ_LW,  RT.D_rZ_LW, 
                  RT.K_ϕϕ_iphalf_LW, RT.K_ϕϕ_imhalf_LW,
                  RT.F_r_LW,         RT.F_Z_LW,
                  RT.F_r_Hr_LW,      RT.F_Z_Hr_LW, 
                  RT.F_r_HZ_LW,      RT.F_Z_HZ_LW, 
                  inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell,
                  UpdateJold, true, :Reflect, :Outflow, :Outflow, :Outflow,
                  rfirst_StellarRT, Val(:Threaded) )

    # Advect EUV1 photons:
    AdvectJandH!( c̃Δt,   RT.js_EUV1,   RT.α_F_EUV1,  RT.J_EUV1,     RT.J_EUV1_old,   RT.H_r_EUV1,   RT.H_Z_EUV1, 
                  RT.J_EUV1_stage,     RT.H_r_EUV1_stage,   RT.H_Z_EUV1_stage,
                  RT.D_rr_EUV1,        RT.D_ZZ_EUV1,  RT.D_rZ_EUV1, 
                  RT.K_ϕϕ_iphalf_EUV1, RT.K_ϕϕ_imhalf_EUV1,
                  RT.F_r_EUV1,         RT.F_Z_EUV1,
                  RT.F_r_Hr_EUV1,      RT.F_Z_Hr_EUV1, 
                  RT.F_r_HZ_EUV1,      RT.F_Z_HZ_EUV1, 
                  inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell,
                  UpdateJold, true, :Reflect, :Outflow, :Outflow, :Outflow,
                  rfirst_StellarRT, Val(:Threaded) )

    # Advect EUV2 photons:
    AdvectJandH!( c̃Δt,   RT.js_EUV2,   RT.α_F_EUV2,  RT.J_EUV2,     RT.J_EUV2_old,   RT.H_r_EUV2,   RT.H_Z_EUV2, 
                  RT.J_EUV2_stage,     RT.H_r_EUV2_stage,   RT.H_Z_EUV2_stage,
                  RT.D_rr_EUV2,        RT.D_ZZ_EUV2,  RT.D_rZ_EUV2, 
                  RT.K_ϕϕ_iphalf_EUV2, RT.K_ϕϕ_imhalf_EUV2,
                  RT.F_r_EUV2,         RT.F_Z_EUV2,
                  RT.F_r_Hr_EUV2,      RT.F_Z_Hr_EUV2, 
                  RT.F_r_HZ_EUV2,      RT.F_Z_HZ_EUV2, 
                  inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell,
                  UpdateJold, true, :Reflect, :Outflow, :Outflow, :Outflow,
                  rfirst_StellarRT, Val(:Threaded) )

    # Advect EUV3 photons:
    AdvectJandH!( c̃Δt,   RT.js_EUV3,   RT.α_F_EUV3,  RT.J_EUV3,     RT.J_EUV3_old,   RT.H_r_EUV3,   RT.H_Z_EUV3, 
                  RT.J_EUV3_stage,     RT.H_r_EUV3_stage,   RT.H_Z_EUV3_stage,
                  RT.D_rr_EUV3,        RT.D_ZZ_EUV3,  RT.D_rZ_EUV3, 
                  RT.K_ϕϕ_iphalf_EUV3, RT.K_ϕϕ_imhalf_EUV3,
                  RT.F_r_EUV3,         RT.F_Z_EUV3,
                  RT.F_r_Hr_EUV3,      RT.F_Z_Hr_EUV3, 
                  RT.F_r_HZ_EUV3,      RT.F_Z_HZ_EUV3, 
                  inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell,
                  UpdateJold, true, :Reflect, :Outflow, :Outflow, :Outflow,
                  rfirst_StellarRT, Val(:Threaded) )
    

    #######################################################################
    #######                                                         #######
    #######       S T E P  3  :  T H E R M O - C H E M I S T R Y    #######
    #######                                                         #######
    #######################################################################

    # After the transport step, we sub-cycle absorption, emission, chemistry,
    # and cooling/heating for all cells.

    N_r, N_Z  = size( ρ )
    dN_r      = 1:N_r 
    dN_Z      = 1:N_Z 

    # Next, sub-cycle for each cell: 
    inv_Δt_RT = 1 / Δt_RT
    @threads for j in dN_Z
        @simd for i in dN_r
            @inbounds begin
                # Get estimated time-derivatives of J from emission + advection:
                dJ_optdt   = ( RT.J_opt[i,j]  - RT.J_opt_old[i,j]  ) * inv_Δt_RT
                dJ_FUVdt   = ( RT.J_FUV[i,j]  - RT.J_FUV_old[i,j]  ) * inv_Δt_RT
                dJ_LWdt    = ( RT.J_LW[i,j]   - RT.J_LW_old[i,j]   ) * inv_Δt_RT
                dJ_EUV1dt  = ( RT.J_EUV1[i,j] - RT.J_EUV1_old[i,j] ) * inv_Δt_RT
                dJ_EUV2dt  = ( RT.J_EUV2[i,j] - RT.J_EUV2_old[i,j] ) * inv_Δt_RT
                dJ_EUV3dt  = ( RT.J_EUV3[i,j] - RT.J_EUV3_old[i,j] ) * inv_Δt_RT

                # Get estimated time-derivatives of H_r and H_Z from advection:
                dHr_optdt  = ( RT.H_r_opt[i,j]  - RT.H_r_opt_old[i,j] )  * inv_Δt_RT
                dHZ_optdt  = ( RT.H_Z_opt[i,j]  - RT.H_Z_opt_old[i,j] )  * inv_Δt_RT

                dHr_FUVdt  = ( RT.H_r_FUV[i,j]  - RT.H_r_FUV_old[i,j] )  * inv_Δt_RT 
                dHZ_FUVdt  = ( RT.H_Z_FUV[i,j]  - RT.H_Z_FUV_old[i,j] )  * inv_Δt_RT

                dHr_LWdt   = ( RT.H_r_LW[i,j]   - RT.H_r_LW_old[i,j] )   * inv_Δt_RT
                dHZ_LWdt   = ( RT.H_Z_LW[i,j]   - RT.H_Z_LW_old[i,j] )   * inv_Δt_RT

                dHr_EUV1dt = ( RT.H_r_EUV1[i,j] - RT.H_r_EUV1_old[i,j] ) * inv_Δt_RT
                dHZ_EUV1dt = ( RT.H_Z_EUV1[i,j] - RT.H_Z_EUV1_old[i,j] ) * inv_Δt_RT

                dHr_EUV2dt = ( RT.H_r_EUV2[i,j] - RT.H_r_EUV2_old[i,j] ) * inv_Δt_RT 
                dHZ_EUV2dt = ( RT.H_Z_EUV2[i,j] - RT.H_Z_EUV2_old[i,j] ) * inv_Δt_RT

                dHr_EUV3dt = ( RT.H_r_EUV3[i,j] - RT.H_r_EUV3_old[i,j] ) * inv_Δt_RT
                dHZ_EUV3dt = ( RT.H_Z_EUV3[i,j] - RT.H_Z_EUV3_old[i,j] ) * inv_Δt_RT

                # Next we sub-cycle for each cell: 
                result = ThermoChemistrySubCycleUpdate!( i, j, Δt_RT, c̃,
                                                           e[i,j], ρ[i,j], 
                                                         xH2[i,j], xHII[i,j],
                                                         xHD[i,j], xDII[i,j],
                                                       xHeII[i,j], xHeIII[i,j], 

                                                     RT.Δρu_r_rad, RT.Δρu_Z_rad,
                                                       RT.a_r_rad, RT.a_Z_rad,
                                                      RT.α_F_EUV1, RT.α_F_EUV2, RT.α_F_EUV3,

                                                RT.J_opt_old[i,j], dJ_optdt,
                                              RT.H_r_opt_old[i,j], RT.H_Z_opt_old[i,j], dHr_optdt, dHZ_optdt,

                                                RT.J_FUV_old[i,j], dJ_FUVdt,
                                              RT.H_r_FUV_old[i,j], RT.H_Z_FUV_old[i,j], dHr_FUVdt, dHZ_FUVdt,

                                                 RT.J_LW_old[i,j], dJ_LWdt, 
                                               RT.H_r_LW_old[i,j], RT.H_Z_LW_old[i,j], dHr_LWdt, dHZ_LWdt,

                                               RT.J_EUV1_old[i,j], dJ_EUV1dt, 
                                             RT.H_r_EUV1_old[i,j], RT.H_Z_EUV1_old[i,j], dHr_EUV1dt, dHZ_EUV1dt,

                                               RT.J_EUV2_old[i,j], dJ_EUV2dt, 
                                             RT.H_r_EUV2_old[i,j], RT.H_Z_EUV2_old[i,j], dHr_EUV2dt, dHZ_EUV2dt,

                                               RT.J_EUV3_old[i,j], dJ_EUV3dt,
                                             RT.H_r_EUV3_old[i,j], RT.H_Z_EUV3_old[i,j], dHr_EUV3dt, dHZ_EUV3dt,

                                                          z, true, CoolingHeating )

                # Get updated quantities:
                e[i,j], xH2[i,j], xHII[i,j], xHD[i,j], xDII[i,j], xHeII[i,j], xHeIII[i,j], 
                RT.J_opt[i,j], RT.J_FUV[i,j], RT.J_LW[i,j], RT.J_EUV1[i,j], RT.J_EUV2[i,j], RT.J_EUV3[i,j],
                RT.H_r_LW[i,j],   RT.H_Z_LW[i,j],
                RT.H_r_EUV1[i,j], RT.H_Z_EUV1[i,j],  
                RT.H_r_EUV2[i,j], RT.H_Z_EUV2[i,j],
                RT.H_r_EUV3[i,j], RT.H_Z_EUV3[i,j] = result 

            end
        end # End of i-loop 
    end # End of j-loop 

    # Update the old stellar J and H:
    @threads for j in dN_Z
        @simd for i in dN_r 
            @inbounds begin
                # Update old J:
                #RT.J_opt_old[i,j]   = RT.J_opt[i,j]
                #RT.J_FUV_old[i,j]   = RT.J_FUV[i,j]
                RT.J_LW_old[i,j]     = RT.J_LW[i,j]
                RT.J_EUV1_old[i,j]   = RT.J_EUV1[i,j]
                RT.J_EUV2_old[i,j]   = RT.J_EUV2[i,j]
                RT.J_EUV3_old[i,j]   = RT.J_EUV3[i,j]

                # Update old H:
                RT.H_r_LW_old[i,j]   = RT.H_r_LW[i,j]
                RT.H_Z_LW_old[i,j]   = RT.H_Z_LW[i,j]

                RT.H_r_EUV1_old[i,j] = RT.H_r_EUV1[i,j]
                RT.H_Z_EUV1_old[i,j] = RT.H_Z_EUV1[i,j]

                RT.H_r_EUV2_old[i,j] = RT.H_r_EUV2[i,j]
                RT.H_Z_EUV2_old[i,j] = RT.H_Z_EUV2[i,j]

                RT.H_r_EUV3_old[i,j] = RT.H_r_EUV3[i,j]
                RT.H_Z_EUV3_old[i,j] = RT.H_Z_EUV3[i,j]

            end
        end # End of i-loop 
    end # End of j-loop 

    return Δt_RT
end


using SpecialFunctions  # For erf

function GenerateStellarRTArrays( mstarZAMS::Float64, 
                                 Zstar_Zsun::Float64,
                                         ΔV::AbstractArray{Float64, 2},
                                          r::AbstractArray{Float64, 1},
                                          Z::AbstractArray{Float64, 1}
                                 )
   
    # Grid size: 
    N_r, N_Z = size( ΔV )
    dN_r     = 1:N_r
    dN_Z     = 1:N_Z

    # Convenient arrays: 
    zeros_c()  = zeros(Float64, N_r,   N_Z)
    zeros_rf() = zeros(Float64, N_r+1, N_Z)   # For r-interface fluxes 
    zeros_zf() = zeros(Float64, N_r,   N_Z+1) # For Z-interface fluxes 

    # Intensities (latest & old, and stage values for RK2) for all bands: 
    J_opt    = zeros_c();  J_opt_old  = copy(J_opt);  J_opt_stage  = copy(J_opt)
    J_FUV    = zeros_c();  J_FUV_old  = copy(J_FUV);  J_FUV_stage  = copy(J_opt)
    J_LW     = zeros_c();  J_LW_old   = copy(J_LW);   J_LW_stage   = copy(J_opt)
    J_EUV1   = zeros_c();  J_EUV1_old = copy(J_EUV1); J_EUV1_stage = copy(J_opt)
    J_EUV2   = zeros_c();  J_EUV2_old = copy(J_EUV2); J_EUV2_stage = copy(J_opt)
    J_EUV3   = zeros_c();  J_EUV3_old = copy(J_EUV3); J_EUV3_stage = copy(J_opt)

    # H ≡ ∫dΩ I n̂ / 4π arrays for all bands:
    H_r_opt        = zeros_c();  H_Z_opt       = zeros_c()
    H_r_opt_old    = zeros_c();  H_Z_opt_old   = zeros_c()
    H_r_opt_stage  = zeros_c();  H_Z_opt_stage = zeros_c()

    H_r_FUV        = zeros_c();  H_Z_FUV       = zeros_c()
    H_r_FUV_old    = zeros_c();  H_Z_FUV_old   = zeros_c()
    H_r_FUV_stage  = zeros_c();  H_Z_FUV_stage = zeros_c()

    H_r_LW         = zeros_c();  H_Z_LW       = zeros_c()
    H_r_LW_old     = zeros_c();  H_Z_LW_old   = zeros_c()
    H_r_LW_stage   = zeros_c();  H_Z_LW_stage = zeros_c()

    H_r_EUV1       = zeros_c();  H_Z_EUV1       = zeros_c()
    H_r_EUV1_old   = zeros_c();  H_Z_EUV1_old   = zeros_c()
    H_r_EUV1_stage = zeros_c();  H_Z_EUV1_stage = zeros_c()

    H_r_EUV2       = zeros_c();  H_Z_EUV2       = zeros_c()
    H_r_EUV2_old   = zeros_c();  H_Z_EUV2_old   = zeros_c()
    H_r_EUV2_stage = zeros_c();  H_Z_EUV2_stage = zeros_c()

    H_r_EUV3       = zeros_c();  H_Z_EUV3     = zeros_c()
    H_r_EUV3_old   = zeros_c();  H_Z_EUV3_old = zeros_c()
    H_r_EUV3_stage = zeros_c();  H_Z_EUV3_stage = zeros_c()

    # Eddington tensor components for all bands:
    D_rr_opt  = fill(1/3, N_r, N_Z);  D_ZZ_opt  = fill(1/3, N_r, N_Z);  D_rZ_opt  = zeros_c()
    D_rr_FUV  = fill(1/3, N_r, N_Z);  D_ZZ_FUV  = fill(1/3, N_r, N_Z);  D_rZ_FUV  = zeros_c()
    D_rr_LW   = fill(1/3, N_r, N_Z);  D_ZZ_LW   = fill(1/3, N_r, N_Z);  D_rZ_LW   = zeros_c()
    D_rr_EUV1 = fill(1/3, N_r, N_Z);  D_ZZ_EUV1 = fill(1/3, N_r, N_Z);  D_rZ_EUV1 = zeros_c()
    D_rr_EUV2 = fill(1/3, N_r, N_Z);  D_ZZ_EUV2 = fill(1/3, N_r, N_Z);  D_rZ_EUV2 = zeros_c()
    D_rr_EUV3 = fill(1/3, N_r, N_Z);  D_ZZ_EUV3 = fill(1/3, N_r, N_Z);  D_rZ_EUV3 = zeros_c()

    # ϕϕ-component of radiation pressure tensor K = D × J at interfaces, 
    # needed for geometric source term:
    K_ϕϕ_iphalf_LW   = zeros_c();     K_ϕϕ_imhalf_LW   = zeros_c()
    K_ϕϕ_iphalf_EUV1 = zeros_c();     K_ϕϕ_imhalf_EUV1 = zeros_c()
    K_ϕϕ_iphalf_EUV2 = zeros_c();     K_ϕϕ_imhalf_EUV2 = zeros_c()
    K_ϕϕ_iphalf_EUV3 = zeros_c();     K_ϕϕ_imhalf_EUV3 = zeros_c()

    # Interface fluxes for all bands: 
    F_r_opt     = zeros_rf(); F_Z_opt     = zeros_zf()
    F_r_Hr_opt  = zeros_rf(); F_Z_Hr_opt  = zeros_zf()
    F_r_HZ_opt  = zeros_rf(); F_Z_HZ_opt  = zeros_zf()

    F_r_FUV     = zeros_rf(); F_Z_FUV     = zeros_zf()
    F_r_Hr_FUV  = zeros_rf(); F_Z_Hr_FUV  = zeros_zf()
    F_r_HZ_FUV  = zeros_rf(); F_Z_HZ_FUV  = zeros_zf()

    F_r_LW      = zeros_rf(); F_Z_LW      = zeros_zf()
    F_r_Hr_LW   = zeros_rf(); F_Z_Hr_LW   = zeros_zf()
    F_r_HZ_LW   = zeros_rf(); F_Z_HZ_LW   = zeros_zf()

    F_r_EUV1    = zeros_rf(); F_Z_EUV1    = zeros_zf()
    F_r_Hr_EUV1 = zeros_rf(); F_Z_Hr_EUV1 = zeros_zf()
    F_r_HZ_EUV1 = zeros_rf(); F_Z_HZ_EUV1 = zeros_zf()

    F_r_EUV2    = zeros_rf(); F_Z_EUV2    = zeros_zf()
    F_r_Hr_EUV2 = zeros_rf(); F_Z_Hr_EUV2 = zeros_zf()
    F_r_HZ_EUV2 = zeros_rf(); F_Z_HZ_EUV2 = zeros_zf()

    F_r_EUV3    = zeros_rf(); F_Z_EUV3    = zeros_zf()
    F_r_Hr_EUV3 = zeros_rf(); F_Z_Hr_EUV3 = zeros_zf()
    F_r_HZ_EUV3 = zeros_rf(); F_Z_HZ_EUV3 = zeros_zf()

    # Flux opacities for all bands:
    α_F_opt     = zeros_c()
    α_F_FUV     = zeros_c()
    α_F_LW      = zeros_c()
    α_F_EUV1    = zeros_c()
    α_F_EUV2    = zeros_c()
    α_F_EUV3    = zeros_c()

    # Emissivities for all bands:
    js_opt      = zeros_c()
    js_FUV      = zeros_c()
    js_LW       = zeros_c()
    js_EUV1     = zeros_c()
    js_EUV2     = zeros_c()
    js_EUV3     = zeros_c()

    # Arrays for tracking changes in gas momentum from radiation pressure:
    Δρu_r_rad     = zeros_c()
    Δρu_Z_rad     = zeros_c()
    a_r_rad       = zeros_c()
    a_Z_rad       = zeros_c()

    # Get stellar luminosities in all bands:
    Lum_opt       = L_opt(mstarZAMS, Zstar_Zsun)  * Lsun 
    Lum_FUV       = L_FUV(mstarZAMS, Zstar_Zsun)  * Lsun
    Lum_LW        = L_LW(mstarZAMS, Zstar_Zsun)   * Lsun 
    Lum_EUV1      = L_EUV1(mstarZAMS, Zstar_Zsun) * Lsun 
    Lum_EUV2      = L_EUV2(mstarZAMS, Zstar_Zsun) * Lsun 
    Lum_EUV3      = L_EUV3(mstarZAMS, Zstar_Zsun) * Lsun  

    #Lum_opt  = 0.0
    #Lum_FUV  = 0.0
    #Lum_LW   = 0.0
    #Lum_EUV1 = 1e6 * Lsun
    #Lum_EUV2 = 0.0
    #Lum_EUV3 = 0.0

    # Fill emissivities consistent with a star 
    # at (i,j) = (1,1). Since the Z = 0 midplane cuts through the 
    # star, we only have half the luminosity in the upper simulated 
    # place Z > 0, so we need to reduce the cell emissivity in 
    # half to get correct luminosity:

    HalfSpaceMultiplier = 1.0
    σs       = 7 * (r[2] - r[1])

    js0_LW   = 0.0 * HalfSpaceMultiplier * (Lum_LW / Four_π)   / ( (2*π)^(3/2) * σs^3 )
    js0_EUV1 = HalfSpaceMultiplier * (Lum_EUV1 / Four_π) / ( (2*π)^(3/2) * σs^3 )
    js0_EUV2 = 0.0 * HalfSpaceMultiplier * (Lum_EUV2 / Four_π) / ( (2*π)^(3/2) * σs^3 )
    js0_EUV3 = 0.0 * HalfSpaceMultiplier * (Lum_EUV3/ Four_π) / ( (2*π)^(3/2) * σs^3 )
    Zc = 0.25 * pc
    @inbounds for j in dN_Z 
        for i in dN_r 
            EXP_factors = exp( - r[i]^2 / (2 * σs^2)) - exp( - r[i+1]^2 / (2 * σs^2))
            Z_factors   = erf((Z[j+1] - Zc) / (sqrt(2) * σs)) - erf((Z[j] - Zc) / (sqrt(2) * σs))

            f_ij = 2 * π * σs^3 * sqrt(π/2) * EXP_factors * Z_factors / ΔV[i,j]

            js_LW[i,j]   = js0_LW   * f_ij
            js_EUV1[i,j] = js0_EUV1 * f_ij
            js_EUV2[i,j] = js0_EUV2 * f_ij
            js_EUV3[i,j] = js0_EUV3 * f_ij
        end
    end

    #js_EUV1[1,1] = (Lum_EUV1 / Four_π) / ΔV[1,1]

    return StellarRTArrays(
        J_opt, J_opt_old,   J_opt_stage,
        J_FUV, J_FUV_old,   J_FUV_stage,
        J_LW, J_LW_old,     J_LW_stage,
        J_EUV1, J_EUV1_old, J_EUV1_stage, 
        J_EUV2, J_EUV2_old, J_EUV2_stage, 
        J_EUV3, J_EUV3_old, J_EUV3_stage, 

        H_r_opt,        H_Z_opt, 
        H_r_opt_old,    H_Z_opt_old,
        H_r_opt_stage,  H_Z_opt_stage,

        H_r_FUV,        H_Z_FUV,
        H_r_FUV_old,    H_Z_FUV_old,
        H_r_FUV_stage,  H_Z_FUV_stage,
        
        H_r_LW,         H_Z_LW,
        H_r_LW_old,     H_Z_LW_old,
        H_r_LW_stage,   H_Z_LW_stage,

        H_r_EUV1,       H_Z_EUV1, 
        H_r_EUV1_old,   H_Z_EUV1_old,
        H_r_EUV1_stage, H_Z_EUV1_stage,

        H_r_EUV2,       H_Z_EUV2, 
        H_r_EUV2_old,   H_Z_EUV2_old,
        H_r_EUV2_stage, H_Z_EUV2_stage,

        H_r_EUV3,       H_Z_EUV3,
        H_r_EUV3_old,   H_Z_EUV3_old,
        H_r_EUV3_stage, H_Z_EUV3_stage,

        D_rr_opt,  D_ZZ_opt, D_rZ_opt, 
        D_rr_FUV,  D_ZZ_FUV, D_rZ_FUV, 
        D_rr_LW,   D_ZZ_LW, D_rZ_LW,
        D_rr_EUV1, D_ZZ_EUV1, D_rZ_EUV1, 
        D_rr_EUV2, D_ZZ_EUV2, D_rZ_EUV2, 
        D_rr_EUV3, D_ZZ_EUV3, D_rZ_EUV3,

        K_ϕϕ_iphalf_LW,   K_ϕϕ_imhalf_LW,
        K_ϕϕ_iphalf_EUV1, K_ϕϕ_imhalf_EUV1,
        K_ϕϕ_iphalf_EUV2, K_ϕϕ_imhalf_EUV2,
        K_ϕϕ_iphalf_EUV3, K_ϕϕ_imhalf_EUV3,

        F_r_opt,  F_Z_opt,  F_r_Hr_opt,  F_Z_Hr_opt,  F_r_HZ_opt,  F_Z_HZ_opt,
        F_r_FUV,  F_Z_FUV,  F_r_Hr_FUV,  F_Z_Hr_FUV,  F_r_HZ_FUV,  F_Z_HZ_FUV,
        F_r_LW,   F_Z_LW,   F_r_Hr_LW,   F_Z_Hr_LW,   F_r_HZ_LW,   F_Z_HZ_LW,
        F_r_EUV1, F_Z_EUV1, F_r_Hr_EUV1, F_Z_Hr_EUV1, F_r_HZ_EUV1, F_Z_HZ_EUV1,
        F_r_EUV2, F_Z_EUV2, F_r_Hr_EUV2, F_Z_Hr_EUV2, F_r_HZ_EUV2, F_Z_HZ_EUV2,
        F_r_EUV3, F_Z_EUV3, F_r_Hr_EUV3, F_Z_Hr_EUV3, F_r_HZ_EUV3, F_Z_HZ_EUV3,

        α_F_opt, α_F_FUV, α_F_LW, α_F_EUV1, α_F_EUV2, α_F_EUV3,
        Lum_opt, Lum_FUV, Lum_LW, Lum_EUV1, Lum_EUV2, Lum_EUV3,

        js_opt, js_FUV, js_LW, js_EUV1, js_EUV2, js_EUV3,

        Δρu_r_rad, Δρu_Z_rad, 
        a_r_rad,   a_Z_rad
    )

end



end # End of module 