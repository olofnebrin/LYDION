

include( "SedovTaylorTest.jl" )
using .SedovTaylorTest         # The Sedov-Taylor test setup 

ρ₀::Float64                        =  1.0
γ::Float64                         =  5 / 3
Eblast::Float64                    =  1.0
tmax::Float64                      =  0.4*4          # Stop simulation at t = tmax (default: 0.4)
Rmax::Float64                      =  1.5          # Simulation volume size in r-direction    
Zmax::Float64                      =  Rmax      
N_r::Int                           =  256
N_Z::Int                           =  N_r

SaveData::Bool                     =  true
Nminsteps::Int                     =  300
Nmaxsteps::Int                     =  100000

###############################################################
#######                                                 #######
#######         S T A R T  S I M U L A T I O N          #######
#######                                                 #######
###############################################################

SetupSedovTaylorTest( ρ₀, γ, Eblast, Rmax, Zmax,
                      tmax, N_r, N_Z, 
                      Nminsteps, Nmaxsteps )

