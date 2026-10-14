""" This code sets up a test of Lyα escape fractions from 
    dusty uniform clouds, to compare against Fig. 3 
    in Nebrin et al. (2025). The setup there is 
    a static cloud with τcl = 10⁸, T = 10⁴ K, no 
    Lyα destruction (pd = 0), no recoil, but varying 
    continuum absorption optical depth. """

include("src/StaticCloudTest.jl")
using .StaticCloudTest
using .StaticCloudTest.ConstantsAndFunctions

# -----------------------------------------------------------
# Default settings shared by all runs
# -----------------------------------------------------------

τcl::Float64              = 1e8
T::Float64                = 1e4
pd::Float64               = 0.0
rs::Float64               = 0.0
umax::Float64             = 0.0
Recoil::Bool              = false   
θhole::Float64            = 0.0
IonizedSourceRegion::Bool = false
tmax_unitless::Float64    = 4.0
BC_Z_bot::Symbol          = :Reflect
N_r_lin::Int              = 128
N_Z_lin::Int              = 128

# -----------------------------------------------------------
# Dust test suite: vary continuum absorption optical depth
# -----------------------------------------------------------

# The following continuum optical depths correspond roughly to 
# 0 < (aτ)^(1/3) τabs < 1, comparable to Fig. 3 in Nebrin+ (2025).
#τabs_list::Vector{Float64} = [
#    0.0,
#    1e-4,
#    3e-4,
#    1e-3,
#    3e-3,
#    1e-2,
#    3e-2,
#    1e-1,
#    3e-1,
#    1.0
# ]

τabs_list::Vector{Float64} = [
    3e-3,
    1e-2,
    3e-2,
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

for τabs in τabs_list

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

    SetupStaticCloudTest( τcl, T, τabs, pd, rs, umax, Recoil, θhole,
                          IonizedSourceRegion, tmax_unitless, BC_Z_bot,
                          N_r_lin, N_Z_lin,
                          300, 1000000, 300, true, FileName )
end