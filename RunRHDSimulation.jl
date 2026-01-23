

include( "src/RHDSolver.jl" )
using .RHDSolver                         # The full RHD solver module
using .RHDSolver.ConstantsAndFunctions   # For constants (e.g. unit conversions)
using  Dates                             # To have date in the file name for convenience


tmax::Float64                      =  3e5 * yr     # Stop simulation at t = tmax (unless it stops automatically earlier)
Nsaved::Int                        =  300          # Target number of saved time-steps (if saving is turned on -- see below)
Nminsteps::Int                     =  300          # Minimum number of time-steps taken in simulation 
Nmaxsteps::Int                     =  100000       # Maximum number of time-steps taken in simulation 

#########################################################################
#################  S T E L L A R  P R O P E R T I E S  ##################
#########################################################################

mstarZAMS::Float64                 =  35            # Initial mass of the star (M⊙)
Zstar_Zsun::Float64                =  1.0           # Metallicity of the star in Solar units 
σstar_cell::Float64                =  4             # Gaussian size of stellar emission region in cell size units (≳ 4 for approximately spherical H II regions)
Zstar_pos::Float64                 =  0 * pc        # Position of star along the cylindrical Z-axis (0 is the lower corner)

#########################################################################
######  C L O U D  P R O P E R T I E S  &  S P A T I A L  G R I D  ######
#########################################################################

z::Float64                         =  0.0           # Redshift 
DG₀::Float64                       =  1.0 * DGsun   # Dust-to-gas ratio (NOTE: *not* normalized to Solar value. To get e.g. Solar, set DG₀ = DGsun)
Zgas_Zsun::Float64                 =  1.0           # Gas metallicity in Solar units 
nH₀::Float64                       =  3e3           # Initial cloud density (cm⁻³)
nH_ext::Float64                    =  1e-7 * nH₀    # Initial density far outside the cloud (cm⁻³)
T₀::Float64                        =  500           # Initial cloud temperature (K)
T_ext::Float64                     =  500           # Temperature outside the cloud 
Rcl::Float64                       =  1 * pc        # Radius of cloud (i.e. high-density region)
u_RMSturb::Float64                 =  3 * kms       # Initial RMS turbulence velocity 
Zcl_pos::Float64                   =  Zstar_pos     # Initial position of cloud along Z-axis 

Rmax_lin::Float64                  =  3 * pc        # Uniform/linear grid out to cylindrical r = Rmax_lin 
Zmax_lin::Float64                  =  3 * pc        # Uniform/linear grid out to cylindrical Z = Zmax_lin 
Rmax::Float64                      =  3 * pc        # Logarithmic grid from Rmax_lin to Rmax (typically lower-resolution region)
Zmax::Float64                      =  3 * pc        # Logarithmic grid from Zmax_lin to Zmax (typically lower-resolution region)
N_r_lin::Int                       =  64
N_Z_lin::Int                       =  64
N_r_log::Int                       =  0
N_Z_log::Int                       =  0
 
BC_r_right::Symbol                 =  :Outflow      # BCs at outer r-boundary (:Outflow or :Reflect)
BC_Z_bot::Symbol                   =  :Reflect      # BCs at lower Z-boundary (:Outflow or :Reflect)
BC_Z_top::Symbol                   =  :Outflow      # BCs at upper Z-boundary (:Outflow or :Reflect) 

######################################################################
######  C H E M I S T R Y  I N I T I A L  C O N D I T I O N S  #######
######################################################################

# Below are the initial conditions for the chemical abundance fractions,
# e.g. xHII ≡ nHII / nH, xHeII ≡ nHeII / nHe, etc. NOTE: For a given species, 
# they must all add up to 1, e.g. xHI + xHII + 2 * xH2 = 1 (xHI is determined by conservation below).

xHII₀::Float64                     =  1e-4         # Initial H II fraction inside cloud               (≤ 1)
xH2₀::Float64                      =  0.0          # Initial H₂ fraction inside cloud                 (≤ 0.5)
xHeII₀::Float64                    =  0.0          # Initial He II fraction (nHeII/nHe) inside cloud  (≤ 1)
xHeIII₀::Float64                   =  0.0          # Initial He III fraction inside cloud             (≤ 1)
xDII₀::Float64                     =  0.0          # Initial D II fraction (nDII/nD) inside cloud     (≤ 1)
xHD₀::Float64                      =  0.0          # Initial HD fraction inside cloud                 (≤ 1)
xCI₀::Float64                      =  1.0          # Initial C I fraction (nCI/nC) inside cloud       (≤ 1)
xCII₀::Float64                     =  0.0          # Initial C II fraction inside cloud               (≤ 1)
xCO₀::Float64                      =  0.0          # Initial CO fraction inside cloud                 (≤ 1)
xOI₀::Float64                      =  1.0          # Initial O I fraction (nOI/nO) inside cloud       (≤ 1)
xOII₀::Float64                     =  0.0          # Initial O II fraction inside cloud               (≤ 1)

xHII_ext::Float64                  =  1.0          # Initial H II fraction outside cloud 
xH2_ext::Float64                   =  0.0          # Initial H₂ fraction outside cloud 
xHeII_ext::Float64                 =  0.0          # Initial He II fraction outside cloud 
xHeIII_ext::Float64                =  0.0          # Initial He III fraction outside cloud 
xDII_ext::Float64                  =  0.0          # Initial D II fraction outside cloud 
xHD_ext::Float64                   =  0.0          # Initial HD fraction outside cloud 
xCI_ext::Float64                   =  1.0         
xCII_ext::Float64                  =  0.0  
xCO_ext::Float64                   =  0.0  
xOI_ext::Float64                   =  1.0         
xOII_ext::Float64                  =  0.0 

######################################################################
###############  P H Y I S I C S  S E T T I N G S  ###################
######################################################################

Hydro::Bool                        =  true       # Hydrodynamics                    (on if true, off if false)
Gravity::Bool                      =  true       # Self-gravity                     (on if true, off if false)
Lyα::Bool                          =  false      # Lyα RT/RHD                       (on if true, off if false)
Recoil::Bool                       =  true       # Atomic recoil for Lyα RT         (on if true, off if false)
LyαDestruction::Bool               =  true       # Lyα destruction (pd > 0)         (on if true, off if false)
RadPressure::Bool                  =  true       # Radiation pressure feedback      (on if true, off if false - NOTE: Lyα feedback is only on if Lyα RT is on)
IRfeedback::Bool                   =  true       # IR radiation pressure feedback   (on if true, off if false - NOTE: Is only on if also RadPressure = true)
OTS::Bool                          =  false      # On-the-spot/Case B approximation (on if true, off if false - NOTE: If off, recombination radiation is modelled self-consistently)

# Number of coarse & fine Lyα frequency grid bins (only relevant if Lyα RT is on)
N_ν_coarse::Int                    =  200        # Default: 200 (bins out in the wings)
N_ν_fine::Int                      =  100        # Default: 100 (bins near line center, uniformally spread)

######################################################################
###################  O U T P U T  F I L E  ###########################
######################################################################

SaveData::Bool   =  true          # Whether to save data in HDF5 file or not 

# Get current date: 
date = Dates.format(now(), "yyyy-mm-dd")

# Get resolution:
N_r = N_r_lin + N_r_log
N_Z = N_Z_lin + N_Z_log

if Lyα
    if RadPressure
        FileDescription  = "RHD simulation of a $(mstarZAMS) Msun star, with Lya RT & feedback. Spatial resolution: $(N_r)x$(N_Z)."
    else
        FileDescription  = "RHD simulation of a $(mstarZAMS) Msun star, with Lya RT, but not radiation pressure feedback. Spatial resolution: $(N_r)x$(N_Z)."
    end
    FileName         = "RHDSim_Lya_$(N_r)x$(N_Z)_" * date * ".h5"
else
    FileDescription  = "RHD simulation of a $(mstarZAMS) Msun star, ignoring Lya feedback. Spatial resolution: $(N_r)x$(N_Z)."
    FileName         = "RHDSim_NoLya_$(N_r)x$(N_Z)_" * date * ".h5"
end


###############################################################
#######                                                 #######
#######         S T A R T  S I M U L A T I O N          #######
#######                                                 #######
###############################################################

# The call below will automatically setup the simulation, start 
# running it, and save data throughout the run into the HDF5 file. 

print_info = true 

SetupRHDsimulation( mstarZAMS, Zstar_Zsun, DG₀, nH₀, nH_ext, Rcl, Zcl_pos, Zstar_pos, 
                    σstar_cell, Rmax_lin, Zmax_lin, Rmax, Zmax,
                    T₀, T_ext, tmax, 
                    xHII₀, xH2₀, xHeII₀, xHeIII₀, xDII₀, xHD₀, xCI₀, xCII₀, xCO₀, xOI₀, xOII₀,
                    xHII_ext, xH2_ext, xHeII_ext, xHeIII_ext, xDII_ext, xHD_ext, 
                    xCI_ext, xCII_ext, xCO_ext, xOI_ext, xOII_ext, Zgas_Zsun,
                    z, Hydro, Lyα, Recoil, LyαDestruction, Gravity,
                    RadPressure, IRfeedback, OTS, u_RMSturb, N_r_lin, N_Z_lin, N_r_log, N_Z_log,
                    N_ν_coarse, N_ν_fine, BC_r_right, BC_Z_bot, BC_Z_top,
                    SaveData, Nminsteps, Nmaxsteps, Nsaved,
                    print_info, FileName, FileDescription )
