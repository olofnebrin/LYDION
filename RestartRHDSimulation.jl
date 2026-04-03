

include( "src/RHDSolver.jl" )
using .RHDSolver.ConstantsAndFunctions   # For constants (e.g. unit conversions)
using .RHDSolver                         # The full RHD solver module


# Name of HDF5 file containing simulation you want to restart:
FileName = "RHDSim_NoLya_128x128_2026-03-21.h5"

# Restart the simulation:
RestartRHDsimulation( FileName )