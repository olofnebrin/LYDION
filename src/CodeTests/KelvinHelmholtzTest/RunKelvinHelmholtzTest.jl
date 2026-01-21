
include( "KelvinHelmholtzTest.jl" )
using .KelvinHelmholtzTest         # The Kelvin-Helmholtz test setup 


tmax::Float64                      =  5.0
N_X::Int                           =  128
N_Y::Int                           =  N_X
Nminsteps::Int                     =  300
Nmaxsteps::Int                     =  100000

###############################################################
#######                                                 #######
#######         S T A R T  S I M U L A T I O N          #######
#######                                                 #######
###############################################################

SetupKHTest( tmax, N_X, N_Y, 
             Nminsteps, Nmaxsteps )