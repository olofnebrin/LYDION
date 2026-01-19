
""" FILE:        DustGasCoupling.jl

    PURPOSE:     Contains the code for computing the update in the dust and gas momenta from 
                 radiation pressure & drag forces.

"""


const urel_max::Float64         = 100 * kms    # Maximum relative velocity of dust (fast-moving dust is destroyed anyway)
const urel_max_squared::Float64 = urel_max^2  

function UpdateGasDustMomenta( Δt::Float64, # Time-step 
                                ρ::Float64, # Gas density 
                               ρd::Float64, # Dust density 
                        Δρu_r_gas::Float64,
                       Δρu_r_dust::Float64,
                        Δρu_Z_gas::Float64,
                       Δρu_Z_dust::Float64,
                             ρu_r::Float64, 
                             ρu_Z::Float64, 
                           ρdud_r::Float64, 
                           ρdud_Z::Float64,
                                K::Float64  # Drag coefficient 
                              )

    # Other convenient constants:
    inv_ρtot   = 1 / ( ρd + ρ ) 
    inv_ρd     = 1 / ρd 
    inv_ρ      = 1 / ρ
    ud_r       = ρdud_r * inv_ρd
    ud_Z       = ρdud_Z * inv_ρd
    u_r        = ρu_r   * inv_ρ
    u_Z        = ρu_Z   * inv_ρ

    # STEP 1) Update the dust+gas center-of-mass velocity:
    uCoM_r     = ( ρdud_r + ρu_r ) * inv_ρtot  # Old r-CoM velocity 
    uCoM_Z     = ( ρdud_Z + ρu_Z ) * inv_ρtot  # Old Z-CoM velocity

    uCoM_r_new = uCoM_r  +  ( Δρu_r_dust + Δρu_r_gas ) * inv_ρtot 
    uCoM_Z_new = uCoM_Z  +  ( Δρu_Z_dust + Δρu_Z_gas ) * inv_ρtot 

    # STEP 2) Update relative dust-gas velocity, urel = ud - u, 
    # and cap it to avoid super large values (which leads to dust destruction 
    # anyway):
    urel_r     = ud_r - u_r               # Old r-relative velocity 
    urel_Z     = ud_Z - u_Z               # Old Z-relative velocity 
    B          = K * ( inv_ρd + inv_ρ )   # Convenient variable 
    inv_denom  = 1 / ( 1  +  B * Δt )

    urel_r_new = ( urel_r  +  (Δρu_r_dust * inv_ρd  -  Δρu_r_gas * inv_ρ) ) * inv_denom
    urel_Z_new = ( urel_Z  +  (Δρu_Z_dust * inv_ρd  -  Δρu_Z_gas * inv_ρ) ) * inv_denom

    # Compute would-be updated relative velocity, and if it exceeds urel_max, 
    # reduce the components magnitude such that the relative velocity magnitude is urel_max:
    urel_new_squared   = urel_r_new * urel_r_new  +  urel_Z_new * urel_Z_new 
    if urel_new_squared  > urel_max_squared
       urel_new        = sqrt( urel_new_squared )
       ReductionFactor = urel_max / urel_new
       urel_r_new      = urel_r_new * ReductionFactor
       urel_Z_new      = urel_Z_new * ReductionFactor
    end

    # STEP 3) Obtain updated gas and dust momenta/velocities
    # from uCoM and urel:
    frac_gas   = ρ  * inv_ρtot
    frac_dust  = ρd * inv_ρtot

    ud_r_new   = uCoM_r_new  +  frac_gas * urel_r_new 
    ud_Z_new   = uCoM_Z_new  +  frac_gas * urel_Z_new 

    u_r_new    = uCoM_r_new  -  frac_dust * urel_r_new 
    u_Z_new    = uCoM_Z_new  -  frac_dust * urel_Z_new 

    ρdud_r_new = ρd * ud_r_new
    ρdud_Z_new = ρd * ud_Z_new
    ρu_r_new   = ρ  * u_r_new
    ρu_Z_new   = ρ  * u_Z_new


    return ρu_r_new, ρu_Z_new, ρdud_r_new, ρdud_Z_new,
           u_r_new, u_Z_new, ud_r_new, ud_Z_new
end



function UpdateGasDustMomenta_OLD( Δt::Float64, # Time-step 
                                ρ::Float64, # Gas density 
                               ρd::Float64, # Dust density 
                        Δρu_r_gas::Float64,
                       Δρu_r_dust::Float64,
                        Δρu_Z_gas::Float64,
                       Δρu_Z_dust::Float64,
                             ρu_r::Float64, 
                             ρu_Z::Float64, 
                           ρdud_r::Float64, 
                           ρdud_Z::Float64,
                                K::Float64  # Drag coefficient 
                              )

    if ρd ≤ ρd_floor
       # Gas momentum, with only gas radiation pressure:
       ρu_r_new   = ρu_r  +  Δρu_r_gas
       ρu_Z_new   = ρu_Z  +  Δρu_Z_gas

       # Gas velocity:
       inv_ρ      = 0.0 
       u_r_new    = ρu_r_new * inv_ρ
       u_Z_new    = ρu_Z_new * inv_ρ

       # Zero momentum (and velocity) for dust:
       ρdud_r_new = 0.0
       ρdud_Z_new = 0.0

       return ρu_r_new, ρu_Z_new, ρdud_r_new, ρdud_Z_new,
              u_r_new, u_Z_new, 0.0, 0.0
    end

    inv_Δt   = 1.0 / Δt

    # Get radiation pressure forces on gas & dust:
    f_gas_r  = Δρu_r_gas  * inv_Δt 
    f_gas_Z  = Δρu_Z_gas  * inv_Δt 
    f_dust_Z = Δρu_Z_dust * inv_Δt 
    f_dust_r = Δρu_r_dust * inv_Δt 

    # Convenient variable, where 
    # Δ ≡ (ρd/Δt + K) * (ρ/Δt + K) - K² 
    #   = (ρd ρ / Δt²) + K (ρ + ρd) / Δt
    # so:
    inv_Δ    = 1 / ( ρd * ρ * inv_Δt * inv_Δt  +  K * ( ρ + ρd ) * inv_Δt )

    # Updated dust velocities: 
    ud_r_new = inv_Δ * ( ( ρ * inv_Δt  +  K ) * ( ρdud_r * inv_Δt  +  f_dust_r )  
                         + K * ( ρu_r * inv_Δt  +  f_gas_r ) )

    ud_Z_new = inv_Δ * ( ( ρ * inv_Δt  +  K ) * ( ρdud_Z * inv_Δt  +  f_dust_Z )  
                         + K * ( ρu_Z * inv_Δt  +  f_gas_Z ) )
    
    # Updated gas velocities:
    u_r_new  = inv_Δ * ( ( ρd * inv_Δt  +  K ) * ( ρu_r * inv_Δt  +  f_gas_r )
                         + K * ( ρdud_r * inv_Δt  +  f_dust_r )   )
    u_Z_new  = inv_Δ * ( ( ρd * inv_Δt  +  K ) * ( ρu_Z * inv_Δt  +  f_gas_Z )
                         + K * ( ρdud_Z * inv_Δt  +  f_dust_Z )  )


    # Clamp dust velocity to have a maximum relative velocity of udmax: 
    ud_r_new = clamp( ud_r_new, u_r_new - udmax, u_r_new + udmax )
    ud_Z_new = clamp( ud_Z_new, u_Z_new - udmax, u_Z_new + udmax )

    # Get updated momenta:
    ρu_r_new   = ρ  * u_r_new
    ρu_Z_new   = ρ  * u_Z_new
    ρdud_r_new = ρd * ud_r_new
    ρdud_Z_new = ρd * ud_Z_new

    return ρu_r_new, ρu_Z_new, ρdud_r_new, ρdud_Z_new,
           u_r_new, u_Z_new, ud_r_new, ud_Z_new
end