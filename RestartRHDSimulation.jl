

include( "src/RHDSolver.jl" )
using .RHDSolver.ConstantsAndFunctions   # For constants (e.g. unit conversions)
using .RHDSolver                         # The full RHD solver module


# Name of HDF5 file containing simulation you want to restart:
FileName = "RHDSim_NoLya_256x256_2026-01-30.h5"

# Restart the simulation:
RestartRHDsimulation( FileName )