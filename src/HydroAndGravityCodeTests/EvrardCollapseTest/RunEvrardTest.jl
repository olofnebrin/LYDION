include("EvrardCollapseTest.jl")
include( "../../ConstantsAndFunctions.jl" )

using .EvrardCollapseTest
using .ConstantsAndFunctions

""" This is a test of the gravity solver + its coupling to hydrodynamics.
    It sets up the Evrard test, an adiabatic collapse of a spherical cloud of radius Rcl
    and mass Mcl following Springel 2005. 
    Following their setup, the simulation is evolved until tmax = 0.8 sqrt(Rcl^3 / (G Mcl))
    by default. """


Mcl::Float64                       =  10*Msun        # Mass of the cloud
Rcl::Float64                       =  1*pc           # Radius of the cloud
tmax_adm::Float64                  =  0.8        # Adimensional final time , 0.8 by default 
N_r_lin::Int                       =  4          # Number of cells along cylindrical r-axis. 
N_r_log::Int                       =  128         # Number of logarithmic cells along cylindrical r-axis. 
BC_Z_bot::Symbol                   =  :Reflect    # Use :Reflect to only simulate upper midplane, and :Outflow to simulate both sides
Nminsteps::Int                     =  300
Nmaxsteps::Int                     =  100000

###############################################################
#######                                                 #######
#######         S T A R T  S I M U L A T I O N          #######
#######                                                 #######
###############################################################

SetupEvrardTest( Mcl, Rcl, tmax_adm, N_r_lin, N_r_log, BC_Z_bot, 
                  Nminsteps, Nmaxsteps )

