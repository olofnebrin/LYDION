
include( "SodTest.jl" )
using .SodTest         # The Sod shock tube test setup 


tmax::Float64                      =  0.2
N_X::Int                           =  100
Nminsteps::Int                     =  300
Nmaxsteps::Int                     =  100000

###############################################################
#######                                                 #######
#######         S T A R T  S I M U L A T I O N          #######
#######                                                 #######
###############################################################

SetupSodTest( tmax, N_X,
              Nminsteps, Nmaxsteps )