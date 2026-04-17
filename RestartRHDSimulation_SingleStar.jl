""" This code can be used to restart a single-star RHD simulation. """

include( "src/RHDSolver_SingleStar.jl" )
using .RHDSolver_SingleStar.ConstantsAndFunctions   # For constants (e.g. unit conversions)
using .RHDSolver_SingleStar                         # The full RHD solver module


# Name of HDF5 file containing simulation you want to restart:
FileName = "RHDSim_Lya_128x128_2026-04-10_MostlyH2.h5"
#FileName = "IdealizedRHDSim_Turbulence_false_Lya_64x64_2026-04-08.h5"

# Restart the simulation:
RestartRHDsimulation_SingleStar( FileName )