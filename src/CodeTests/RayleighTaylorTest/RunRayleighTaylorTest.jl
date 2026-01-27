
include( "RayleighTaylorTest.jl" )
using .RayleighTaylorTest         # The Kelvin-Helmholtz test setup 


tmax::Float64                      =  15.0
N_X::Int                           =  256
N_Y::Int                           =  3 * N_X
Nminsteps::Int                     =  300
Nmaxsteps::Int                     =  100000

###############################################################
#######                                                 #######
#######         S T A R T  S I M U L A T I O N          #######
#######                                                 #######
###############################################################

SetupRTTest( tmax, N_X, N_Y, 
             Nminsteps, Nmaxsteps )