""" This code sets up a test of static, dust-free clouds 
    of varying optical depth, to test the prediction 
    M_F ~ (aᵥτcl)^(1/3). """

include("src/StaticCloudTest.jl")
using .StaticCloudTest
using .StaticCloudTest.ConstantsAndFunctions

# -----------------------------------------------------------
# Default settings shared by all runs
# -----------------------------------------------------------

T::Float64                = 1e4
τcl::Float64              = 1e8
τabs::Float64             = 0.0   # No dust absorption 
pd::Float64               = 0.0   # No Lyα destruction 
umax::Float64             = 0.0
Recoil::Bool              = false # No recoil 
θhole::Float64            = 0.0
IonizedSourceRegion::Bool = false
tmax_unitless::Float64    = 4.0
BC_Z_bot::Symbol          = :Reflect
N_r_lin::Int              = 128
N_Z_lin::Int              = 128

# -----------------------------------------------------------
# List of cloud optical depths
# -----------------------------------------------------------

# The following continuum optical depths correspond roughly to 
# 0 < (aτ)^(1/3) τabs < 1, comparable to Fig. 3 in Nebrin+ (2025).
#τcl_list::Vector{Float64} = [
#    0.1,
#    1.0,
#    1e1,
#    1e2,
#    1e3,
#    1e4,
#    1e5,
#    1e6,
#    1e7,
#    1e8,
#    1e9,
#    1e10
# ]

rs_list::Vector{Float64} = [
    0.05,
    0.25,
    0.5,
    0.75
 ]


# -----------------------------------------------------------
# Function for determining the file name
# -----------------------------------------------------------

function MakeFileName(τcl::Float64, T::Float64, τabs::Float64, pd::Float64,
                      rs::Float64, umax::Float64, Recoil::Bool, θhole::Float64,
                      N_r_lin::Int, N_Z_lin::Int)

    if Recoil == true
        if θhole > 0.0
            return "CloudTest_theta$(θhole)_rs$(rs)_tau$(τcl)_T$(T)_umax$(umax)_tauabs$(τabs)_pd$(pd)_$(N_r_lin)x$(N_Z_lin).h5"
        else
            return "CloudTest_rs$(rs)_tau$(τcl)_T$(T)_umax$(umax)_tauabs$(τabs)_pd$(pd)_$(N_r_lin)x$(N_Z_lin).h5"
        end
    else
        if θhole > 0.0
            return "CloudTest_theta$(θhole)_rs$(rs)_NoRecoil_tau$(τcl)_T$(T)_umax$(umax)_tauabs$(τabs)_pd$(pd)_$(N_r_lin)x$(N_Z_lin).h5"
        else
            return "CloudTest_rs$(rs)_NoRecoil_tau$(τcl)_T$(T)_umax$(umax)_tauabs$(τabs)_pd$(pd)_$(N_r_lin)x$(N_Z_lin).h5"
        end
    end
end

# -----------------------------------------------------------
# Run all cases
# -----------------------------------------------------------

for rs in rs_list

    FileName::String = MakeFileName(τcl, T, τabs, pd, rs, umax, Recoil, θhole,
                                    N_r_lin, N_Z_lin)

    println("--------------------------------------------------")
    println("Running static cloud test with:")
    println("τcl   = ", τcl)
    println("T     = ", T)
    println("τabs  = ", τabs)
    println("pd    = ", pd)
    println("rs    = ", rs)
    println("umax  = ", umax)
    println("Recoil = ", Recoil)
    println("File   = ", FileName)
    println("--------------------------------------------------")

    # Cone-related parameters (irrelevant here, but must be set):
    τcone      = τcl 
    SmoothCone = true  
    Δμcone     = 0.0

    SetupStaticCloudTest( τcl, T, τabs, pd, rs, umax, Recoil, θhole, τcone, SmoothCone, Δμcone,
                          IonizedSourceRegion, tmax_unitless, BC_Z_bot,
                          N_r_lin, N_Z_lin,
                          300, 10000000, 300, true, FileName )
end
