
include( "IsentropicPulseTest.jl" )
using .IsentropicPulseTest         # The Isentropic Pulse test setup 


tmax::Float64                      =  0.1
N_X::Int                           =  2048
Nminsteps::Int                     =  300
Nmaxsteps::Int                     =  100000

###############################################################
#######                                                 #######
#######         S T A R T  S I M U L A T I O N          #######
#######                                                 #######
###############################################################

SetupIsentropicPulseTest( tmax, N_X,
              Nminsteps, Nmaxsteps )