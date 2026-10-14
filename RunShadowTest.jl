
""" This file sets up a shadow test, with an ionizing source placed below a dense neutral cloud. """

include( "src/RHDSolver_Idealized.jl" )
using .RHDSolver_Idealized                         # The full RHD solver module
using .RHDSolver_Idealized.ConstantsAndFunctions   # For constants (e.g. unit conversions)
using  Dates                                       # To have date in the file name for convenience



tmax::Float64                      =  2e5 * yr      # Stop simulation at t = tmax (unless it stops automatically earlier)
Nsaved::Int                        =  300           # Target number of saved time-steps (if saving is turned on -- see below)
Nminsteps::Int                     =  300           # Minimum number of time-steps taken in simulation 
Nmaxsteps::Int                     =  100000        # Maximum number of time-steps taken in simulation 
FluxMethodStellarRT::String        =  "HLL"         # Flux method for stellar/IR M1 RT (HLL or GLF) 

#########################################################################
#################  S T E L L A R  P R O P E R T I E S  ##################
#########################################################################

Teff::Float64                      =  1e5           # Effective temperature of black-body 
dNdt_LyC::Float64                  =  5e48          # Total ionizing photon emission rate 
σstar_cell::Float64                =  3             # Gaussian size of stellar emission region (if not plane source) in cell size units (≳ 4 for approximately spherical H II regions)

# Compute the emission rates in the three ionizing bands:
dNdt_EUV1, dNdt_EUV2, dNdt_EUV3    =  GetBlackBodyIonizingEmission( dNdt_LyC, Teff )

#########################################################################
######  C L O U D  P R O P E R T I E S  &  S P A T I A L  G R I D  ######
#########################################################################

z::Float64                         =  0.0           # Redshift 
DG₀::Float64                       =  0.0 * DGsun   # Dust-to-gas ratio (NOTE: *not* normalized to Solar value. To get e.g. Solar, set DG₀ = DGsun)
Zgas_Zsun::Float64                 =  0.0           # Gas metallicity in Solar units 
nH₀::Float64                       =  1e5           # Initial cloud density (cm⁻³)
nH_ext::Float64                    =  0.01          # Initial density far outside the cloud (cm⁻³)
T₀::Float64                        =  100           # Initial cloud temperature (K)
T_ext::Float64                     =  100           # Temperature outside the cloud 
Rcl::Float64                       =  2 * pc        # Radius of cloud (i.e. 'high'-density region)
u_RMSturb::Float64                 =  0 * kms       # Initial RMS turbulence velocity 

# Grid settings below. If BC_Z_bot = :Reflect (Case 1), there is assumed symmetry at Z = 0. If 
# BC_Z_bot = :Outflow, there is no assumed symmetry, and the star + cloud is instead centered at Z = Zmax / 2.
# For Case 1, you can use either only a uniform grid (set N_r_log = N_Z_log_side = 0), or include a logarithmic 
# grid stretching from r = Rmax_lin to r = Rmax, and from Z = LZ_uniform to Zmax. For Case 2, the same holds 
# for the r-logarithmic grid (if used), but the Z-logarithmic grid (if used) stretches in both directions, with 
# N_Z_log_side cells per side (upper logarithmic region, and lower logarithmic region).

Rmax_lin::Float64                  =  2 * Rcl       # Uniform/linear grid out to cylindrical r = Rmax_lin 
LZ_uniform::Float64                =  3 * Rmax_lin  # Total extent of uniform grid along the Z-axis.  
Rmax::Float64                      =  Rmax_lin      # Logarithmic grid from Rmax_lin to Rmax (typically lower-resolution region)
Zmax::Float64                      =  LZ_uniform    # Maximum extent along Z-axis. Should be ≥ LZ_uniform.
N_r_lin::Int                       =  128           # Number of cells along r-direction in the uniform grid section 
N_Z_lin::Int                       =  3 * N_r_lin   # Number of cells along Z-direction in the uniform grid section 
N_r_log::Int                       =  0             # Number of cells along r-direction in the logarithmic grid section 
N_Z_log_side::Int                  =  0             # Number of cells along Z-direction (per side if BC_Z_bot = :Outflow) in the logarithmic grid section  
 
BC_r_right::Symbol                 =  :Outflow      # BCs at outer r-boundary (:Outflow or :Reflect)
BC_Z_bot::Symbol                   =  :Reflect      # BCs at lower Z-boundary (:Outflow or :Reflect)
BC_Z_top::Symbol                   =  :Outflow      # BCs at upper Z-boundary (:Outflow or :Reflect) 

# Place the cloud center at the midpoint of the Z-axis:
Zcl_pos_norm = 0.5

######################################################################
######  C H E M I S T R Y  I N I T I A L  C O N D I T I O N S  #######
######################################################################

# Below are the initial conditions for the chemical abundance fractions,
# e.g. xHII ≡ nHII / nH, xHeII ≡ nHeII / nHe, etc. NOTE: For a given species, 
# they must all add up to 1, e.g. xHI + xHII + 2 * xH2 = 1 (xHI is determined by conservation below).

HydrogenAndHelium::Bool            =  false        # If false, the (non-metal) gas is purely hydrogen, i.e. ignoring helium & deuterium.
                                                   # In that case, the non-hydrogen and non-metal abundances below are arbitrary.

xHII₀::Float64                     =  0.0          # Initial H II fraction inside cloud               (≤ 1)
xH2₀::Float64                      =  0.0          # Initial H₂ fraction inside cloud                 (≤ 0.5)
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

xHII_ext::Float64                  =  0.0          # Initial H II fraction outside cloud 
xH2_ext::Float64                   =  0.0          # Initial H₂ fraction outside cloud 
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
###############  P H Y I S I C S  S E T T I N G S  ###################
######################################################################

c̃_min::Float64                     =  100 * kms  # Minimum reduced speed of light   (cm s⁻¹)
Hydro::Bool                        =  false      # Hydrodynamics                    (on if true, off if false)
RadPressure::Bool                  =  false      # Radiation pressure feedback      (on if true, off if false)
IRfeedback::Bool                   =  false      # IR radiation pressure feedback   (on if true, off if false - NOTE: Is only on if also RadPressure = true)
OTS::Bool                          =  true       # On-the-spot/Case B approximation (on if true, off if false - NOTE: If off, recombination radiation is modelled self-consistently)

######################################################################
###################  O U T P U T  F I L E  ###########################
######################################################################

SaveData::Bool   =  true          # Whether to save data in HDF5 file or not 

# HDF5 file name where the data will be saved:
FileName  = "ShadowTest_OTS$(OTS)_Hydro$(Hydro).h5"

# Get current date: 
date = Dates.format(now(), "yyyy-mm-dd")

# Get resolution:
N_r = N_r_lin + N_r_log

if BC_Z_bot == :Reflect
    N_Z = N_Z_lin + N_Z_log_side
else
    N_Z = N_Z_lin + N_Z_log_side + N_Z_log_side
end

FileDescription  = "Idealized RHD test, with dNdt_EUV1 = $(dNdt_EUV1), dNdt_EUV2 = $(dNdt_EUV2),  dNdt_EUV3 = $(dNdt_EUV3). Spatial resolution: $(N_r)x$(N_Z)."


###############################################################
#######                                                 #######
#######         S T A R T  S I M U L A T I O N          #######
#######                                                 #######
###############################################################

# The call below will automatically setup the simulation, start 
# running it, and save data throughout the run into the HDF5 file. 

print_info = true 

SetupRHDsimulation( dNdt_EUV1, dNdt_EUV2, dNdt_EUV3, Teff, DG₀, nH₀, nH_ext, Rcl, 
                    σstar_cell, Rmax_lin, LZ_uniform, Rmax, Zmax, true,
                    T₀, T_ext, tmax, 
                    xHII₀, xH2₀,   xHeII₀, xHeIII₀, xDII₀, xHD₀, 
                    xCI₀,  xCII₀,  xCIII₀, xCO₀, 
                    xOI₀,  xOII₀,  xOIII₀, xOIV₀,
                    xSII₀, xSIII₀, xSIV₀,
                    xHII_ext, xH2_ext,   xHeII_ext, xHeIII_ext, xDII_ext, xHD_ext, 
                    xCI_ext,  xCII_ext,  xCIII_ext, xCO_ext, 
                    xOI_ext,  xOII_ext,  xOIII_ext, xOIV_ext, 
                    xSII_ext, xSIII_ext, xSIV_ext, 
                    Zgas_Zsun,
                    z, c̃_min, Zcl_pos_norm, Hydro, 
                    RadPressure, IRfeedback, OTS, HydrogenAndHelium,
                    u_RMSturb, N_r_lin, N_Z_lin, N_r_log, N_Z_log_side,
                    BC_r_right, BC_Z_bot, BC_Z_top,
                    SaveData, Nminsteps, Nmaxsteps, Nsaved,
                    print_info, FileName, FileDescription,
                    FluxMethodStellarRT )
