

include( "src/ConstantsAndFunctions.jl" )
include( "src/RHDSolver.jl" )
using .ConstantsAndFunctions   # For constants (e.g. unit conversions)
using .RHDSolver               # The full RHD solver module

tmax::Float64                      =  3e5 * yr     # Stop simulation at t = tmax (unless it stops automatically earlier)
mstarZAMS::Float64                 =  35           # Initial mass of the star 
Zstar_Zsun::Float64                =  1            # Metallicity of the star 
z::Float64                         =  20.0         # Redshift 
nH₀::Float64                       =  1e4          # Initial cloud density (cm⁻³)
T₀::Float64                        =  600          # Initial cloud temperature (K)
Rcl::Float64                       =  1 * pc       # Radius of cloud (i.e. high-density region)
Rmax_lin::Float64                  =  5 * pc       # Uniform/linear grid out to cylindrical r = Rmax_lin 
Zmax_lin::Float64                  =  5 * pc       # Uniform/linear grid out to cylindrical Z = Zmax_lin 
Rmax::Float64                      =  10 * pc      # Logarithmic grid from Rmax_lin to Rmax (typically lower-resolution region)
Zmax::Float64                      =  10 * pc      # Logarithmic grid from Zmax_lin to Zmax (typically lower-resolution region)
N_r_lin::Int                       =  100
N_Z_lin::Int                       =  100
N_r_log::Int                       =  0
N_Z_log::Int                       =  0

xHII₀::Float64                     =  0.0 * 2e-4
xH2₀::Float64                      =  0.0
xHeII₀::Float64                    =  0.0
xHeIII₀::Float64                   =  0.0
xDII₀::Float64                     =  0.0
xHD₀::Float64                      =  0.0

Hydro::Bool                        =  true          # Hydrodynamics               (on if true, off if false)
Gravity::Bool                      =  true          # Self-gravity                (on if true, off if false)
Lyα::Bool                          =  true          # Lyα RT/RHD                  (on if true, off if false)
Recoil::Bool                       =  true 
RadPressure::Bool                  =  true          # Radiation pressure feedback (on if true, off if false)
SaveData::Bool                     =  true
Nminsteps::Int                     =  300
Nmaxsteps::Int                     =  100000

###############################################################
#######                                                 #######
#######         S T A R T  S I M U L A T I O N          #######
#######                                                 #######
###############################################################


SetupRHDsimulation( mstarZAMS, Zstar_Zsun, nH₀, 
                    Rcl, Rmax_lin, Zmax_lin, Rmax, Zmax, T₀,
                    tmax, xHII₀, xH2₀, xHeII₀, xHeIII₀,
                    xDII₀, xHD₀, z, Hydro, Lyα, Recoil, Gravity, RadPressure, 
                    N_r_lin, N_Z_lin, N_r_log, N_Z_log,
                    SaveData, Nminsteps, Nmaxsteps )

