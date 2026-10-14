

include( "src/RHDSolver_SingleStar.jl" )
using .RHDSolver_SingleStar                         # The full RHD solver module
using .RHDSolver_SingleStar.ConstantsAndFunctions   # For constants (e.g. unit conversions)
using  Dates                                        # To have date in the file name for convenience
using  Profile                                      # Julia built-in sampling profiler
using  Base.Threads                                # For Threads.nthreads() in the header

# Optional graphical profiler. Keep this false unless ProfileView.jl is installed.
# The built-in text profiler below works without any extra packages.
const USE_PROFILEVIEW::Bool = false

if USE_PROFILEVIEW
    using ProfileView
end

tmax::Float64                      =  2e3 * yr      # Short profiling run: stop simulation at t = tmax
Nsaved::Int                        =  20            # Irrelevant when SaveData=false, but kept small
Nminsteps::Int                     =  20            # Minimum number of time-steps taken in simulation 
Nmaxsteps::Int                     =  1000          # Safety cap for the profiling run 

#########################################################################
#################  S T E L L A R  P R O P E R T I E S  ##################
#########################################################################

mstarZAMS::Float64                 =  35            # Initial mass of the star (M⊙)
Zstar_Zsun::Float64                =  1.0           # Metallicity of the star in Solar units 
σstar_cell::Float64                =  3             # Gaussian size of stellar emission region in cell size units (≳ 5 for approximately spherical H II regions)

#########################################################################
######  C L O U D  P R O P E R T I E S  &  S P A T I A L  G R I D  ######
#########################################################################

# If POWER_LAW_ENVELOPE = true, the density outside Rcore falls off as ρ ∝ r^(-p_ENVELOPE),
# up until nH = nH_ext. If POWER_LAW_ENVELOPE = false, there is a much sharper cut-off
# in the density outside Rcl:
POWER_LAW_ENVELOPE::Bool           =  false
p_ENVELOPE::Float64                =  2.0

z::Float64                         =  0.0           # Redshift 
DG₀::Float64                       =  1.0 * DGsun   # Dust-to-gas ratio (NOTE: *not* normalized to Solar value. To get e.g. Solar, set DG₀ = DGsun)
Zgas_Zsun₀::Float64                =  Zstar_Zsun    # Initial gas metallicity in Solar units (metals can be depleted onto dust, and dust can be returned to gas)
nH_core::Float64                   =  3e3           # Initial core density (cm⁻³)
nH_ext::Float64                    =  1e-4          # Initial density far outside the cloud (cm⁻³)
T₀::Float64                        =  500           # Initial cloud temperature (K)
T_ext::Float64                     =  500           # Temperature outside the cloud 
Rcore::Float64                     =  1.0 * pc 
Rcl::Float64                       =  1.0 * pc      # Radius of cloud (i.e. high-density region)
u_RMSturb::Float64                 =  4 * kms       # Initial RMS turbulence velocity 

# Grid settings below. If BC_Z_bot = :Reflect (Case 1), there is assumed symmetry at Z = 0. If 
# BC_Z_bot = :Outflow, there is no assumed symmetry, and the star + cloud is instead centered at Z = Zmax / 2.
# For Case 1, you can use either only a uniform grid (set N_r_log = N_Z_log_side = 0), or include a logarithmic 
# grid stretching from r = Rmax_lin to r = Rmax, and from Z = LZ_uniform to Zmax. For Case 2, the same holds 
# for the r-logarithmic grid (if used), but the Z-logarithmic grid (if used) stretches in both directions, with 
# N_Z_log_side cells per side (upper logarithmic region, and lower logarithmic region).

Rmax_lin::Float64                  =  2.0 * pc      # Uniform/linear grid out to cylindrical r = Rmax_lin 
LZ_uniform::Float64                =  2.0 * pc      # Total extent of uniform grid along the Z-axis.  
Rmax::Float64                      =  2.0 * pc      # Logarithmic grid from Rmax_lin to Rmax (typically lower-resolution region)
Zmax::Float64                      =  2.0 * pc      # Maximum extent along Z-axis. Should be ≥ LZ_uniform.
N_r_lin::Int                       =  128           # Number of cells along r-direction in the uniform grid section 
N_Z_lin::Int                       =  128           # Number of cells along Z-direction in the uniform grid section 
N_r_log::Int                       =  0             # Number of cells along r-direction in the logarithmic grid section 
N_Z_log_side::Int                  =  0             # Number of cells along Z-direction (per side if BC_Z_bot = :Outflow) in the logarithmic grid section  
 
BC_r_right::Symbol                 =  :Outflow      # BCs at outer r-boundary (:Outflow or :Reflect)
BC_Z_bot::Symbol                   =  :Reflect      # BCs at lower Z-boundary (:Outflow or :Reflect)
BC_Z_top::Symbol                   =  :Outflow      # BCs at upper Z-boundary (:Outflow or :Reflect) 

######################################################################
######  C H E M I S T R Y  I N I T I A L  C O N D I T I O N S  #######
######################################################################

# Below are the initial conditions for the chemical abundance fractions,
# e.g. xHII ≡ nHII / nH, xHeII ≡ nHeII / nHe, etc. NOTE: For a given species, 
# they must all add up to 1, e.g. xHI + xHII + 2 * xH2 = 1 (xHI is determined by conservation below).

xHII₀::Float64                     =  1e-4         # Initial H II fraction inside cloud               (≤ 1)
xH2₀::Float64                      =  0.1          # Initial H₂ fraction inside cloud                 (≤ 0.5)
xHeII₀::Float64                    =  0.0          # Initial He II fraction (nHeII/nHe) inside cloud  (≤ 1)
xHeIII₀::Float64                   =  0.0          # Initial He III fraction inside cloud             (≤ 1)
xDII₀::Float64                     =  0.0          # Initial D II fraction (nDII/nD) inside cloud     (≤ 1)
xHD₀::Float64                      =  0.0          # Initial HD fraction inside cloud                 (≤ 1)
xCI₀::Float64                      =  1.0          # Initial C I fraction (nCI/nC) inside cloud       (≤ 1)
xCII₀::Float64                     =  0.0          # Initial C II fraction inside cloud               (≤ 1)
xCIII₀::Float64                    =  0.0          # Initial C III fraction inside cloud              (≤ 1)
xCO₀::Float64                      =  0.0          # Initial CO fraction inside cloud                 (≤ 1)
xOI₀::Float64                      =  1.0          # Initial O I fraction (nOI/nO) inside cloud       (≤ 1)
xOII₀::Float64                     =  0.0          # Initial O II fraction inside cloud               (≤ 1)
xOIII₀::Float64                    =  0.0          # Initial O III fraction inside cloud              (≤ 1)
xOIV₀::Float64                     =  0.0          # Initial O IV fraction inside cloud               (≤ 1)
xSII₀::Float64                     =  0.0          # Initial S II fraction inside cloud               (≤ 1)
xSIII₀::Float64                    =  0.0          # Initial S III fraction inside cloud              (≤ 1)
xSIV₀::Float64                     =  0.0          # Initial S IV fraction inside cloud               (≤ 1)

xHII_ext::Float64                  =  1e-4         # Initial H II fraction outside cloud 
xH2_ext::Float64                   =  0.1          # Initial H₂ fraction outside cloud 
xHeII_ext::Float64                 =  0.0          # Initial He II fraction outside cloud 
xHeIII_ext::Float64                =  0.0          # Initial He III fraction outside cloud 
xDII_ext::Float64                  =  0.0          # Initial D II fraction outside cloud 
xHD_ext::Float64                   =  0.0          # Initial HD fraction outside cloud 
xCI_ext::Float64                   =  1.0         
xCII_ext::Float64                  =  0.0  
xCIII_ext::Float64                 =  0.0 
xCO_ext::Float64                   =  0.0  
xOI_ext::Float64                   =  1.0         
xOII_ext::Float64                  =  0.0 
xOIII_ext::Float64                 =  0.0 
xOIV_ext::Float64                  =  0.0
xSII_ext::Float64                  =  0.0 
xSIII_ext::Float64                 =  0.0 
xSIV_ext::Float64                  =  0.0  

######################################################################
###############  P H Y I S I C S  S E T T I N G S  ###################
######################################################################

c̃_min::Float64                     =  100 * kms  # Minimum reduced speed of light   (cm s⁻¹)
Hydro::Bool                        =  true       # Hydrodynamics                    (on if true, off if false)
Gravity::Bool                      =  true       # Self-gravity                     (on if true, off if false)
Lyα::Bool                          =  true       # Lyα RT/RHD                       (on if true, off if false)
Recoil::Bool                       =  true       # Atomic recoil for Lyα RT         (on if true, off if false)
LyαDestruction::Bool               =  true       # Lyα destruction (pd > 0)         (on if true, off if false)
RadPressure::Bool                  =  true       # Radiation pressure feedback      (on if true, off if false - NOTE: Lyα feedback is only on if Lyα RT is on)
IRfeedback::Bool                   =  true       # IR radiation pressure feedback   (on if true, off if false - NOTE: Is only on if also RadPressure = true)
StellarWindFeedback::Bool          =  false      # Stellar wind feedback            (on if true, off if false - WARNING: Wind feeedback can slow down the simulation significantly because of small CFL time-steps)
DustGrowthDestruction::Bool        =  true       # Dust growth & destruction        (on if true, off if false)
OTS::Bool                          =  true       # On-the-spot/Case B approximation (on if true, off if false - NOTE: If off, recombination radiation is modelled self-consistently)

######################################################################
###################  O U T P U T  F I L E  ###########################
######################################################################

SaveData::Bool   =  false         # Turn off HDF5 output so the profile is not dominated by I/O 

# Get current date: 
date = Dates.format(now(), "yyyy-mm-dd")

# Get resolution:
N_r = N_r_lin + N_r_log

if BC_Z_bot == :Reflect
    N_Z = N_Z_lin + N_Z_log_side
else
    N_Z = N_Z_lin + N_Z_log_side + N_Z_log_side
end

if Lyα
    if RadPressure
        FileDescription  = "RHD simulation of a $(mstarZAMS) Msun star, with Lya RT & feedback. Spatial resolution: $(N_r)x$(N_Z)."
    else
        FileDescription  = "RHD simulation of a $(mstarZAMS) Msun star, with Lya RT, but not radiation pressure feedback. Spatial resolution: $(N_r)x$(N_Z)."
    end
    FileName         = "Profile_RHDSim_Lya_$(N_r)x$(N_Z)_" * date * ".h5"
else
    FileDescription  = "RHD simulation of a $(mstarZAMS) Msun star, ignoring Lya feedback. Spatial resolution: $(N_r)x$(N_Z)."
    FileName         = "Profile_RHDSim_NoLya_$(N_r)x$(N_Z)_" * date * ".h5"
end


# Keep terminal output quiet during profiling, otherwise printing itself can
# contaminate the profile and make it harder to read.
print_info::Bool = false

###############################################################
#######                                                 #######
#######         S T A R T  S I M U L A T I O N          #######
#######                                                 #######
###############################################################

# The function below performs one complete profiling simulation.  Wrapping
# the run in a function avoids global-scope overhead in the driver itself and
# makes the profiler output cleaner.
function RunProfileSimulation(; warmup::Bool = false)

    # For the warm-up pass, use only one time-step. This compiles most of the
    # relevant code without forcing two full 256² profiling simulations.
    Nminsteps_run = warmup ? 1 : Nminsteps
    Nmaxsteps_run = warmup ? 1 : Nmaxsteps

    SetupRHDsimulation_SingleStar( mstarZAMS, Zstar_Zsun, DG₀, nH_core, nH_ext, 
                                   Rcore, Rcl, 
                                   σstar_cell, Rmax_lin, LZ_uniform, Rmax, Zmax,
                                   T₀, T_ext, POWER_LAW_ENVELOPE, p_ENVELOPE,
                                   tmax, 
                                   xHII₀, xH2₀,   xHeII₀, xHeIII₀, xDII₀, xHD₀, 
                                   xCI₀,  xCII₀,  xCIII₀, xCO₀, 
                                   xOI₀,  xOII₀,  xOIII₀, xOIV₀,
                                   xSII₀, xSIII₀, xSIV₀,
                                   xHII_ext, xH2_ext,   xHeII_ext, xHeIII_ext, xDII_ext, xHD_ext, 
                                   xCI_ext,  xCII_ext,  xCIII_ext, xCO_ext, 
                                   xOI_ext,  xOII_ext,  xOIII_ext, xOIV_ext, 
                                   xSII_ext, xSIII_ext, xSIV_ext, 
                                   Zgas_Zsun₀,
                                   z, c̃_min, Hydro, Lyα, Recoil, LyαDestruction, Gravity,
                                   RadPressure, IRfeedback, StellarWindFeedback, DustGrowthDestruction, OTS, 
                                   u_RMSturb, N_r_lin, N_Z_lin, N_r_log, N_Z_log_side,
                                   BC_r_right, BC_Z_bot, BC_Z_top,
                                   SaveData, Nminsteps_run, Nmaxsteps_run, Nsaved,
                                   print_info, FileName, FileDescription )

    return nothing
end

println("============================================================")
println(" LYDION-2D RHD PROFILING RUN")
println("============================================================")
println("Resolution:       $(N_r)x$(N_Z)")
println("tmax:             $(tmax / yr) yr")
println("SaveData:         $(SaveData)")
println("Lyα:              $(Lyα)")
println("Hydro:            $(Hydro)")
println("Gravity:          $(Gravity)")
println("RadPressure:      $(RadPressure)")
println("IRfeedback:       $(IRfeedback)")
println("Dust physics:     $(DustGrowthDestruction)")
println("Threads:          $(Threads.nthreads())")
println("============================================================")

# Warm up first, otherwise the profile is polluted by JIT compilation.
println("Warm-up run...")
RunProfileSimulation(warmup = true)

# Actual CPU profile.
println("Profiling run...")
Profile.clear()
GC.gc()

@profile RunProfileSimulation(warmup = false)

# Write full profile output to a file. The tree profile can be enormous and can
# otherwise push the useful flat ranking out of the VS Code terminal scrollback.
ProfileFile = "RHD_profile_$(N_r)x$(N_Z)_$(date).txt"

flat_io = IOBuffer()
Profile.print(flat_io, format = :flat, sortedby = :count, maxdepth = 80)
flat_text = String(take!(flat_io))

open(ProfileFile, "w") do io
    println(io, "============================================================")
    println(io, " LYDION-2D RHD PROFILE")
    println(io, "============================================================")
    println(io, "Resolution:       $(N_r)x$(N_Z)")
    println(io, "tmax:             $(tmax / yr) yr")
    println(io, "SaveData:         $(SaveData)")
    println(io, "Lyα:              $(Lyα)")
    println(io, "Hydro:            $(Hydro)")
    println(io, "Gravity:          $(Gravity)")
    println(io, "RadPressure:      $(RadPressure)")
    println(io, "IRfeedback:       $(IRfeedback)")
    println(io, "Dust physics:     $(DustGrowthDestruction)")
    println(io, "Threads:          $(Threads.nthreads())")
    println(io)

    println(io, "============================================================")
    println(io, " FLAT PROFILE, SORTED BY SAMPLE COUNT")
    println(io, "============================================================")
    print(io, flat_text)

    println(io, "\n============================================================")
    println(io, " TREE PROFILE")
    println(io, "============================================================")
    Profile.print(io, format = :tree, combine = true, maxdepth = 80)
end

# Print only the high-count end of the flat ranking to the terminal. In current
# Julia output this tends to be ordered from low count to high count, so the
# useful entries are near the bottom, not the top.
println("\n============================================================")
println(" HIGHEST-COUNT ENTRIES FROM FLAT PROFILE")
println(" Full profile written to: $(ProfileFile)")
println("============================================================")
flat_lines = split(flat_text, '\n')
Nlines_terminal = min(160, length(flat_lines))
first_line = max(1, length(flat_lines) - Nlines_terminal + 1)
for n in first_line:length(flat_lines)
    println(flat_lines[n])
end
println("\n... terminal output truncated intentionally. See $(ProfileFile) for the full flat and tree profiles.")

# Open a graphical flame graph if ProfileView.jl is available/requested.
if USE_PROFILEVIEW
    ProfileView.view()
else
    println("\nProfileView was not requested. To use it, install ProfileView.jl and set USE_PROFILEVIEW = true near the top of this file.")
end
