

include( "src/StaticCloudTest.jl" )
using .StaticCloudTest                         # The module setting up the no-hydro static cloud test
using .StaticCloudTest.ConstantsAndFunctions   # For constants (e.g. unit conversions)

τcl::Float64                = 1e8          # Cloud Lyα optical depth 
T::Float64                  = 1e4          # Cloud temperature (K)
τabs::Float64               = 0.0          # Cloud continuum absorption optical depth 
pd::Float64                 = 0.0          # Lyα destruction probability
rs::Float64                 = 1.0          # Relative size of source region, Rs/Rcl
umax::Float64               = 0.0          # Maximum gas velocity (km s⁻¹) at the cloud edge, assuming Hubble flow
Recoil::Bool                = true         # Whether atomic recoil is turned on (true) or not (false)
θcone::Float64              = 0.0          # Angle of any cone-like hole in the cloud (radians)
τcone::Float64              = τcl          # Optical depth in the cone direction
SmoothCone::Bool            = true         # Smooth transition to the cone or not 
Δμ_cone::Float64            = 0.025        # Width of transition in cos(θ), if SmoothCone = true 
IonizedSourceRegion::Bool   = false        # Whether gas in emission region is ionized (if true) or not (if false)
tmax_unitless::Float64      = 4.0          # Maximum simulation time, in units (aτcl)^(1/3) × (Rcl / c̃) (default: 4.0)
BC_Z_bot::Symbol            = :Reflect     # Reflective or outflow BCs at Z = 0 (reflective if midplane symmetry, outflow if not)
N_r_lin::Int                = 32           # Number of cells along r-direction (uniform grid)
N_Z_lin::Int                = 32           # Number of cells along Z-direction (uniform grid)

SavePerformanceTiming::Bool = true 

if Recoil == true
    if θcone > 0.0
        FileName::String = "CloudTest_theta$(θcone)_rs$(rs)_tau$(τcl)_T$(T)_umax$(umax)_tauabs$(τabs)_pd$(pd)_$(N_r_lin)x$(N_Z_lin).h5"
    else 
        FileName::String = "CloudTest_rs$(rs)_tau$(τcl)_T$(T)_umax$(umax)_tauabs$(τabs)_pd$(pd)_$(N_r_lin)x$(N_Z_lin).h5"
    end
else
    if θcone > 0.0
        FileName::String = "CloudTest_theta$(θcone)_rs$(rs)_NoRecoil_tau$(τcl)_T$(T)_umax$(umax)_tauabs$(τabs)_pd$(pd)_$(N_r_lin)x$(N_Z_lin).h5"
    else
        FileName::String = "CloudTest_rs$(rs)_NoRecoil_tau$(τcl)_T$(T)_umax$(umax)_tauabs$(τabs)_pd$(pd)_$(N_r_lin)x$(N_Z_lin).h5"
    end
end

###############################################################
#######                                                 #######
#######         S T A R T  S I M U L A T I O N          #######
#######                                                 #######
###############################################################

q = 1.1 # Use default frequency binning 

SetupStaticCloudTest( τcl, T, τabs, pd, rs, umax, Recoil, θcone, τcone, SmoothCone, Δμ_cone,
                      IonizedSourceRegion, tmax_unitless, BC_Z_bot, N_r_lin, N_Z_lin,
                      30, 1000000, 300, true, FileName, q, SavePerformanceTiming )


