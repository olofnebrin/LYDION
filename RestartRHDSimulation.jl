

include( "src/ConstantsAndFunctions.jl" )
include( "src/RHDSolver.jl" )
using .ConstantsAndFunctions   # For constants (e.g. unit conversions)
using .RHDSolver               # The full RHD solver module


# Name of HDF5 file containing simulation you want to restart:
FileName = "RHDSim_NoLya_512x512_2025-12-24.h5"

# Restart the simulation:
RestartRHDsimulation( FileName )