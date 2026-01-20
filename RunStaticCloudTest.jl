

include( "src/StaticCloudTest.jl" )
using .StaticCloudTest                         # The module setting up the no-hydro static cloud test
using .StaticCloudTest.ConstantsAndFunctions   # For constants (e.g. unit conversions)

τcl::Float64   = 1e8   # Cloud Lyα optical depth 
T₀::Float64    = 1e2   # Cloud temperature (K)
τabs::Float64  = 0.0   # Cloud continuum absorption optical depth 
pd₀::Float64   = 0.0   # Lyα destruction probability
rs::Float64    = 0.5   # Relative size of source region, Rs/Rcl
umax::Float64  = 0.0   # Maximum gas velocity (km s⁻¹) at the cloud edge, assuming Hubble flow
θhole::Float64 = π     # Angle of any hole in the cloud 



###############################################################
#######                                                 #######
#######         S T A R T  S I M U L A T I O N          #######
#######                                                 #######
###############################################################

SetupStaticCloudTest( τcl, T₀, τabs, pd₀, rs, umax )


