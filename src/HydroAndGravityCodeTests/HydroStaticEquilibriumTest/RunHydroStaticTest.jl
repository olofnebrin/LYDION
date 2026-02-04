""" This is a hydrostatic equilibrium test, which stresses both the gravity 
    solver and its coupling to hydrodynamics. A polytrope P ∝ ρ^Γ, with Γ = γ = 2
    is set up, with the hydrostatic equilibrium profile ρ ∝ sin(αR)/αR.
    The simulation is evolved for tmax_tdyn number of dynamical times 
    (t_dyn = Rcl / v_esc). Goal is for the code to reasonably maintain 
    equilibrium over this time. """


include( "HydrostaticEquilibriumTest.jl" )
using .HydrostaticEquilibriumTest         # The hydrostatic equilibrium test setup 


tmax_tdyn::Float64                 =  50.0        # Final time in units of t_dyn 
N_r::Int                           =  128         # Number of cells along cylindrical r-axis. 
BC_Z_bot::Symbol                   =  :Outflow    # Use :Reflect to only simulate upper midplane, and :Outflow to simulate both sides
Nminsteps::Int                     =  300
Nmaxsteps::Int                     =  100000

###############################################################
#######                                                 #######
#######         S T A R T  S I M U L A T I O N          #######
#######                                                 #######
###############################################################

SetupHydrostaticTest( tmax_tdyn, N_r, BC_Z_bot, 
                      Nminsteps, Nmaxsteps )

