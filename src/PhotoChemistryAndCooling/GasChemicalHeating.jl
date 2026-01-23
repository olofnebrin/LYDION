
""" FILE:       GasChemicalHeating.jl

    PURPOSE:    Contains the estimated gas heating rate from chemical reactions. Specifically, 
                H₂ formation on dust grains and in the gas phase. 
    
"""

using .ConstantsAndFunctions               # Needed constants

export HeatingRate_H2chemical


# Energy release from various channels:
const E_H2_gr_th::Float64  = 0.2  * eV
const E_H2_gr_vib::Float64 = 4.2  * eV
const E_H2_Hminus::Float64 = 3.53 * eV 
const E_H2_mol::Float64    = 4.48 * eV 

function HeatingRate_H2chemical( T::Float64, 
                                Td::Float64,
                                nH::Float64,
                               nHI::Float64,
                               nH2::Float64,
                              nHII::Float64,
                                ne::Float64, 
                                ρd::Float64, 
                               Γpd::Float64 = 0.0,
                                 z::Float64 = 0.0 )

    inv_nH = 1 / nH 
    xHI    = nHI * inv_nH
    xH2    = nH2 * inv_nH

    # Critical density:
    if xHI > 0.0 || xH2 > 0.0
        EXP1       = exp( - (400.0 / T)^2 )
        EXP2       = exp( - 12000.0 / ( T + 1200.0 ) )
        ncrit_H2   = 1e6 * (T^(-0.5)) / ( 1.6 * xHI * EXP1  +  1.4 * xH2 * EXP2 )
    else
        # In fully ionized gas, set ncrit_H2 = Inf:
        ncrit_H2   = Inf 
    end

    # Fraction of excitation energy actually going to heat:
    f_exc_heat  = 1 / ( 1 + ncrit_H2 / nH )

    # Chemical heating from H₂ formation on dust grains (0.2 eV into thermal energy of H₂
    # leaving grain, 4.2 eV from vibrational & rotational excitation of the H₂ molecule):
    Γgr      = R_dustH2(T, ρd, Td) * nHI * ( E_H2_gr_th  +  E_H2_gr_vib * f_exc_heat )

    # Chemical heating from H₂ formation via H⁻ (3.53 eV is excitation energy initially, of which a 
    # fraction f_exc_heat goes into heat):
    Γ_Hminus = kGP_H2( T, ne, nHI, nHII, Γpd, z ) * nHI * ne * ( E_H2_Hminus * f_exc_heat )

    # Chemical heating from 3-body H₂ formation (4.48 eV released):
    Γ_3body  = k3B_H2(T) * nHI * (nHI + nH2 / 8) * nHI * E_H2_mol

    # Gas *cooling* from collisional dissociation (same 4.48 eV):
    Λ_diss   =  ( kdiss_H2HI(T, nHI)  * nHI  * nH2
                + kdiss_H2H2(T, nH2)  * nH2  * nH2
                + kdiss_H2e(T)        * ne   * nH2 
                + kdiss_H2HII(T)      * nHII * nH2 ) * E_H2_mol

    # Put it all together to get net heating rate (cooling if < 0):
    Γ_H2chemical = Γgr + Γ_Hminus + Γ_3body - Λ_diss

    return Γ_H2chemical
end