
include( "IsentropicPulseTest.jl" )
using .IsentropicPulseTest         # The Isentropic Pulse test setup 


tmax::Float64                      =  0.1
Nminsteps::Int                     =  300
Nmaxsteps::Int                     =  100000
N::Vector{Int}                     =  [32, 64, 128, 256, 1024]

###############################################################
#######                                                 #######
#######         S T A R T  S I M U L A T I O N S        #######
#######                                                 #######
###############################################################

for N_X in N
    SetupIsentropicPulseTest( tmax, N_X,
              Nminsteps, Nmaxsteps)
end