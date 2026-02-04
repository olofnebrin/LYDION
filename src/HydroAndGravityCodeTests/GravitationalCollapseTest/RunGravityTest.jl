""" This is a test of the gravity solver + its coupling to hydrodynamics.
    It sets up a nearly pressure-free collapse of a spherical cloud of density 
    ρ = 10⁹ g cm⁻³ and radius Rcl = 6500 km, following Skinner et al. (2019).
    Following their setup, the simulation is evolved until tmax = 0.065 sec 
    by default. """


include( "GravitationalCollapseTest.jl" )
using .GravitationalCollapseTest         # The gravitational collapse test setup 


tmax::Float64                      =  0.065       # Final time (sec), 0.065 by default 
N_r::Int                           =  512         # Number of cells along cylindrical r-axis. 
BC_Z_bot::Symbol                   =  :Outflow    # Use :Reflect to only simulate upper midplane, and :Outflow to simulate both sides
Nminsteps::Int                     =  300
Nmaxsteps::Int                     =  100000

###############################################################
#######                                                 #######
#######         S T A R T  S I M U L A T I O N          #######
#######                                                 #######
###############################################################

SetupGravityTest( tmax, N_r, BC_Z_bot, 
                  Nminsteps, Nmaxsteps )

