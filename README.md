# LYDION: Lyman-α Radiation Hydrodynamics & Stellar Feedback Simulations

**LYDION** (**LY**man-α **DI**ffus**ION**) is a fully coupled 2D Lyman-α (Lyα) radiation hydrodynamics code, capable of simulating **Lyα radiation pressure feedback** on-the-fly. This is done using a novel M1 moment method for Lyα radiative transfer. Also implemented in LYDION is stellar and IR radiative transfer, detailed non-equilibrium ISM photo-chemistry and heating/cooling, and dust dynamics and dust physics.

Taken together, LYDION allows realistic simulations of stellar feedback around young massive stars, from Lyα and non-Lyα radiation pressure, and photoionization. This can be used to assess the relative importance of these processes, and inform feedback prescriptions in 3D galaxy/star formation simulations.

This has become increasingly urgent recently, following the launch of the James Webb Space Telescope (JWST). With
the JWST, observations of dust-poor galaxies at Cosmic Dawn has become possible, which is precisely 
the regime where Lyα feedback can strongly regulate star formation.

A full description and test of LYDION can be found at: TODO

LYDION is primarily written in the Julia programming language ([Bezanson et al. 2017](https://doi.org/10.1137/141000671)), allowing for both fast development and good performance.

# Dependencies

Besides the core Julia language, the following external packages are used:

- HDF5.jl
- Interpolations.jl
- ProgressMeter.jl
- Roots.jl
- SpecialFunctions.jl

These dependencies are listed in 'Project.toml', and can be installed automatically, as shown below.

# Installation

Clone the repository and activate the project environment:

```bash
git clone https://github.com/olofnebrin/LYDION.git
cd LYDION
julia --project=.
```

In the Julia REPL, all required dependencies can be installed using:

```using Pkg
Pkg.instantiate()
```

# How to run

The file RunRHDSimulation.jl contains the basic setup for running an RHD 
simulation, with or without Lyα feedback. Cloud initial conditions, the 
mass of the star, physics settings, resolution, etc., can all be changed
in that file. Running the file will set up the simulation, run it, and 
continuously save output data to an HDF5 file.

To launch a simulation from the repository root:

```bash
julia --project=. RunRHDSimulation.jl
```
