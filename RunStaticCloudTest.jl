
include( "src/ConstantsAndFunctions.jl" )
include( "src/StaticCloudTest.jl" )
using .ConstantsAndFunctions   # For constants (e.g. unit conversions)
using .StaticCloudTest         # The module setting up the no-hydro static cloud test

τcl::Float64   = 1e8
T₀::Float64    = 1e2
τabs::Float64  = 0.0
pd₀::Float64   = 0.0
rs::Float64    = 0.5
umax::Float64  = 0.0
θhole::Float64 = π



###############################################################
#######                                                 #######
#######         S T A R T  S I M U L A T I O N          #######
#######                                                 #######
###############################################################

SetupStaticCloudTest( τcl, T₀, τabs, pd₀, rs, umax )


