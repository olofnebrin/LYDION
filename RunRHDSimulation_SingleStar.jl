

include( "src/RHDSolver_SingleStar.jl" )
using .RHDSolver_SingleStar                         # The full RHD solver module
using .RHDSolver_SingleStar.ConstantsAndFunctions   # For constants (e.g. unit conversions)
using  Dates                                        # To have date in the file name for convenience


tmax::Float64                      =  2e5 * yr      # Stop simulation at t = tmax (unless it stops automatically earlier)
Nsaved::Int                        =  600           # Target number of saved time-steps (if saving is turned on -- see below)
Nminsteps::Int                     =  300           # Minimum number of time-steps taken in simulation 
Nmaxsteps::Int                     =  100000        # Maximum number of time-steps taken in simulation 
FluxMethodStellarRT::String        =  "HLL"         # Flux method for stellar/IR M1 RT (HLL or GLF) 
FluxMethodLyαRT::String            =  "GLF"         # Flux method for Lyα M1 RT, if used (HLL or GLF) 


#########################################################################
#################  S T E L L A R  P R O P E R T I E S  ##################
#########################################################################

mstarZAMS::Float64                 =  35            # Initial mass of the star (M⊙)
Zstar_Zsun::Float64                =  1.0           # Metallicity of the star in Solar units 
δZstar_pos::Float64                =  0.7 * pc      # Position of star along Z-axis, relative to the cloud center (0 means placing the star at the cloud center)
σstar_cell::Float64                =  3             # Gaussian size of stellar emission region in cell size units (≳ 5 for approximately spherical H II regions)

#########################################################################
######  C L O U D  P R O P E R T I E S  &  S P A T I A L  G R I D  ######
#########################################################################

# If POWER_LAW_ENVELOPE = true, the density outside Rcore falls off as ρ ∝ r^(-p_ENVELOPE),
# up until nH = nH_ext. If POWER_LAW_ENVELOPE = false, there is a much sharper cut-off
# in the density outside Rcl:
POWER_LAW_ENVELOPE::Bool           =  false
p_ENVELOPE::Float64                =  2.0

z::Float64                         =  0.0           # Redshift 
DG₀::Float64                       =  1.0 * DGsun   # Dust-to-gas ratio (NOTE: *not* normalized to Solar value. To get e.g. Solar, set DG₀ = DGsun)
Zgas_Zsun₀::Float64                =  Zstar_Zsun    # Initial gas metallicity in Solar units (metals can be depleted onto dust, and dust can be returned to gas)
nH_core::Float64                   =  3e3           # Initial core density (cm⁻³)
nH_ext::Float64                    =  1e1           # Initial density far outside the cloud (cm⁻³)
T₀::Float64                        =  500           # Initial cloud temperature (K)
T_ext::Float64                     =  500           # Temperature outside the cloud 
Rcore::Float64                     =  1.0 * pc 
Rcl::Float64                       =  1.0 * pc      # Radius of cloud (i.e. high-density region)
u_RMSturb::Float64                 =  4 * kms       # Initial RMS turbulence velocity 

# Grid settings below. If BC_Z_bot = :Reflect (Case 1), there is assumed symmetry at Z = 0. If 
# BC_Z_bot = :Outflow, there is no assumed symmetry, and the star + cloud is instead centered at Z = Zmax / 2.
# For Case 1, you can use either only a uniform grid (set N_r_log = N_Z_log_side = 0), or include a logarithmic 
# grid stretching from r = Rmax_lin to r = Rmax, and from Z = LZ_uniform to Zmax. For Case 2, the same holds 
# for the r-logarithmic grid (if used), but the Z-logarithmic grid (if used) stretches in both directions, with 
# N_Z_log_side cells per side (upper logarithmic region, and lower logarithmic region).

Rmax_lin::Float64                  =  2.0 * pc      # Uniform/linear grid out to cylindrical r = Rmax_lin 
LZ_uniform::Float64                =  4.0 * pc      # Total extent of uniform grid along the Z-axis.  
Rmax::Float64                      =  2.0 * pc      # Logarithmic grid from Rmax_lin to Rmax (typically lower-resolution region)
Zmax::Float64                      =  4.0 * pc      # Maximum extent along Z-axis. Should be ≥ LZ_uniform.
N_r_lin::Int                       =  512           # Number of cells along r-direction in the uniform grid section 
N_Z_lin::Int                       =  1024          # Number of cells along Z-direction in the uniform grid section 
N_r_log::Int                       =  0             # Number of cells along r-direction in the logarithmic grid section 
N_Z_log_side::Int                  =  0             # Number of cells along Z-direction (per side if BC_Z_bot = :Outflow) in the logarithmic grid section  
 
BC_r_right::Symbol                 =  :Outflow      # BCs at outer r-boundary (:Outflow or :Reflect)
BC_Z_bot::Symbol                   =  :Outflow      # BCs at lower Z-boundary (:Outflow or :Reflect)
BC_Z_top::Symbol                   =  :Outflow      # BCs at upper Z-boundary (:Outflow or :Reflect) 

######################################################################
######  C H E M I S T R Y  I N I T I A L  C O N D I T I O N S  #######
######################################################################

# Below are the initial conditions for the chemical abundance fractions,
# e.g. xHII ≡ nHII / nH, xHeII ≡ nHeII / nHe, etc. NOTE: For a given species, 
# they must all add up to 1, e.g. xHI + xHII + 2 * xH2 = 1 (xHI is determined by conservation below).

xHII₀::Float64                     =  1e-4         # Initial H II fraction inside cloud               (≤ 1)
xH2₀::Float64                      =  0.1          # Initial H₂ fraction inside cloud                 (≤ 0.5)
xHeII₀::Float64                    =  0.0          # Initial He II fraction (nHeII/nHe) inside cloud  (≤ 1)
xHeIII₀::Float64                   =  0.0          # Initial He III fraction inside cloud             (≤ 1)
xDII₀::Float64                     =  0.0          # Initial D II fraction (nDII/nD) inside cloud     (≤ 1)
xHD₀::Float64                      =  0.0          # Initial HD fraction inside cloud                 (≤ 1)
xCI₀::Float64                      =  1.0          # Initial C I fraction (nCI/nC) inside cloud       (≤ 1)
xCII₀::Float64                     =  0.0          # Initial C II fraction inside cloud               (≤ 1)
xCIII₀::Float64                    =  0.0          # Initial C III fraction inside cloud              (≤ 1)
xCO₀::Float64                      =  0.0          # Initial CO fraction inside cloud                 (≤ 1)
xOI₀::Float64                      =  1.0          # Initial O I fraction (nOI/nO) inside cloud       (≤ 1)
xOII₀::Float64                     =  0.0          # Initial O II fraction inside cloud               (≤ 1)
xOIII₀::Float64                    =  0.0          # Initial O III fraction inside cloud              (≤ 1)
xOIV₀::Float64                     =  0.0          # Initial O IV fraction inside cloud               (≤ 1)
xSII₀::Float64                     =  0.0          # Initial S II fraction inside cloud               (≤ 1)
xSIII₀::Float64                    =  0.0          # Initial S III fraction inside cloud              (≤ 1)
xSIV₀::Float64                     =  0.0          # Initial S IV fraction inside cloud               (≤ 1)

xHII_ext::Float64                  =  1e-4         # Initial H II fraction outside cloud 
xH2_ext::Float64                   =  0.1          # Initial H₂ fraction outside cloud 
xHeII_ext::Float64                 =  0.0          # Initial He II fraction outside cloud 
xHeIII_ext::Float64                =  0.0          # Initial He III fraction outside cloud 
xDII_ext::Float64                  =  0.0          # Initial D II fraction outside cloud 
xHD_ext::Float64                   =  0.0          # Initial HD fraction outside cloud 
xCI_ext::Float64                   =  1.0         
xCII_ext::Float64                  =  0.0  
xCIII_ext::Float64                 =  0.0 
xCO_ext::Float64                   =  0.0  
xOI_ext::Float64                   =  1.0         
xOII_ext::Float64                  =  0.0 
xOIII_ext::Float64                 =  0.0 
xOIV_ext::Float64                  =  0.0
xSII_ext::Float64                  =  0.0 
xSIII_ext::Float64                 =  0.0 
xSIV_ext::Float64                  =  0.0  

######################################################################
#################  P H Y S I C S  S E T T I N G S  ###################
######################################################################

c̃_min::Float64                     =  100 * kms  # Minimum reduced speed of light   (cm s⁻¹)
Hydro::Bool                        =  true       # Hydrodynamics                    (on if true, off if false)
Gravity::Bool                      =  true       # Self-gravity                     (on if true, off if false)
Lyα::Bool                          =  false      # Lyα RT/RHD                       (on if true, off if false)
Recoil::Bool                       =  true       # Atomic recoil for Lyα RT         (on if true, off if false)
LyαDestruction::Bool               =  true       # Lyα destruction (pd > 0)         (on if true, off if false)
RadPressure::Bool                  =  true       # Radiation pressure feedback      (on if true, off if false - NOTE: Lyα feedback is only on if Lyα RT is on)
IRfeedback::Bool                   =  true       # IR radiation pressure feedback   (on if true, off if false - NOTE: Is only on if also RadPressure = true)
StellarWindFeedback::Bool          =  false      # Stellar wind feedback            (on if true, off if false - WARNING: Wind feeedback can slow down the simulation significantly because of small CFL time-steps)
DustGrowthDestruction::Bool        =  true       # Dust growth & destruction        (on if true, off if false)
OTS::Bool                          =  false      # On-the-spot/Case B approximation (on if true, off if false - NOTE: If off, recombination radiation is modelled self-consistently)

######################################################################
###################  O U T P U T  F I L E  ###########################
######################################################################

SaveData::Bool   =  true          # Whether to save data in HDF5 file or not 

# Get current date: 
date = Dates.format(now(), "yyyy-mm-dd")

# Get resolution:
N_r = N_r_lin + N_r_log

if BC_Z_bot == :Reflect
    N_Z = N_Z_lin + N_Z_log_side
else
    N_Z = N_Z_lin + N_Z_log_side + N_Z_log_side
end

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

SetupRHDsimulation_SingleStar( mstarZAMS, Zstar_Zsun, DG₀, nH_core, nH_ext, 
                               Rcore, Rcl, 
                               σstar_cell, Rmax_lin, LZ_uniform, Rmax, Zmax,
                               T₀, T_ext, POWER_LAW_ENVELOPE, p_ENVELOPE,
                               tmax, 
                               xHII₀, xH2₀,   xHeII₀, xHeIII₀, xDII₀, xHD₀, 
                               xCI₀,  xCII₀,  xCIII₀, xCO₀, 
                               xOI₀,  xOII₀,  xOIII₀, xOIV₀,
                               xSII₀, xSIII₀, xSIV₀,
                               xHII_ext, xH2_ext,   xHeII_ext, xHeIII_ext, xDII_ext, xHD_ext, 
                               xCI_ext,  xCII_ext,  xCIII_ext, xCO_ext, 
                               xOI_ext,  xOII_ext,  xOIII_ext, xOIV_ext, 
                               xSII_ext, xSIII_ext, xSIV_ext, 
                               Zgas_Zsun₀,
                               z, c̃_min, Hydro, Lyα, Recoil, LyαDestruction, Gravity,
                               RadPressure, IRfeedback, StellarWindFeedback, DustGrowthDestruction, OTS, 
                               u_RMSturb, N_r_lin, N_Z_lin, N_r_log, N_Z_log_side,
                               BC_r_right, BC_Z_bot, BC_Z_top,
                               SaveData, Nminsteps, Nmaxsteps, Nsaved,
                               print_info, FileName, FileDescription,
                               δZstar_pos, FluxMethodStellarRT, FluxMethodLyαRT )
