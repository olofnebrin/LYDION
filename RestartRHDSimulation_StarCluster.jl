""" This code can be used to restart a single-star RHD simulation. """

include( "src/RHDSolver_StarCluster.jl" )
using .RHDSolver_StarCluster.ConstantsAndFunctions   # For constants (e.g. unit conversions)
using .RHDSolver_StarCluster                         # The full RHD solver module


# Name of HDF5 file containing simulation you want to restart:
#FileName = "RHDSim_NoLya_128x128_2026-03-21.h5"
FileName = "RHDSim_StarCluster_Lya_128x128_2026-04-16.h5" 

# Restart the simulation:
RestartRHDsimulation_StarCluster( FileName )