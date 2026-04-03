
""" FILE:           InitializeChemistryStruct.jl

    PURPOSE:        This file contains the basic code for 
                    initializing the gas chemistry state struct/arrays.
    
"""

export InitializeChemistryArrays


""" This function initializes arrays & struct for the 
    chemistry state. 
    
    NOTE: This function does *not* fill all
    the arrays with values, i.e. the abundances are all zero.
    They need to be filled by hand later. """
function InitializeChemistryArrays( N_r::Int, 
                                    N_Z::Int, 
                                 ZC_val::Float64,
                               ZSil_val::Float64,
                           Zgasrest_val::Float64,
                      HydrogenAndHelium::Bool )

    xHI_adv    = zeros( Float64, N_r, N_Z );  xHI_adv_old  = zeros( Float64, N_r, N_Z )
    xHII       = zeros( Float64, N_r, N_Z );  xHII_old     = zeros( Float64, N_r, N_Z )
    xH2        = zeros( Float64, N_r, N_Z );  xH2_old      = zeros( Float64, N_r, N_Z )
    xHeI_adv   = zeros( Float64, N_r, N_Z );  xHeI_adv_old = zeros( Float64, N_r, N_Z )
    xHeII      = zeros( Float64, N_r, N_Z );  xHeII_old    = zeros( Float64, N_r, N_Z )
    xHeIII     = zeros( Float64, N_r, N_Z );  xHeIII_old   = zeros( Float64, N_r, N_Z )
    xDI_adv    = zeros( Float64, N_r, N_Z );  xDI_adv_old  = zeros( Float64, N_r, N_Z )
    xDII       = zeros( Float64, N_r, N_Z );  xDII_old     = zeros( Float64, N_r, N_Z )
    xHD        = zeros( Float64, N_r, N_Z );  xHD_old      = zeros( Float64, N_r, N_Z )

    xCI        = zeros( Float64, N_r, N_Z );  xCI_old      = zeros( Float64, N_r, N_Z )
    xCII       = zeros( Float64, N_r, N_Z );  xCII_old     = zeros( Float64, N_r, N_Z )
    xCHx_adv   = zeros( Float64, N_r, N_Z );  xCHx_adv_old = zeros( Float64, N_r, N_Z )
    xCO        = zeros( Float64, N_r, N_Z );  xCO_old      = zeros( Float64, N_r, N_Z )
    xOI        = zeros( Float64, N_r, N_Z );  xOI_old      = zeros( Float64, N_r, N_Z )
    xOII       = zeros( Float64, N_r, N_Z );  xOII_old     = zeros( Float64, N_r, N_Z )
    xOHx_adv   = zeros( Float64, N_r, N_Z );  xOHx_adv_old = zeros( Float64, N_r, N_Z )

    ZC         = zeros( Float64, N_r, N_Z )
    ZSil       = zeros( Float64, N_r, N_Z )
    Zgasrest   = zeros( Float64, N_r, N_Z )
    Lsob       = zeros( Float64, N_r, N_Z )

    ZC        .= ZC_val
    ZSil      .= ZSil_val
    Zgasrest  .= Zgasrest_val

    return ChemistryState( xHI_adv, xHI_adv_old,  xHII, xHII_old,  xH2, xH2_old,  
                           xHeI_adv, xHeI_adv_old,  xHeII, xHeII_old,  xHeIII, xHeIII_old,
                           xDI_adv, xDI_adv_old,  xDII, xDII_old,  xHD, xHD_old,
                           xCI, xCI_old,  xCII, xCII_old,  xCHx_adv, xCHx_adv_old,
                           xCO, xCO_old,  xOI, xOI_old,  xOII, xOII_old,  xOHx_adv, xOHx_adv_old,
                           ZC, ZSil, Zgasrest, Lsob, HydrogenAndHelium )
end
