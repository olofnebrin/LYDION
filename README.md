
![LYDION simulation overview](README_Figure.png)
---
# Lyman-α Radiation Hydrodynamics & Stellar Feedback Simulations
**LYDION** (**LY**man-α **DI**ffus**ION**) is the first multidimensional (2D axisymmetric) Lyman-α (Lyα) radiation hydrodynamics (RHD) code, capable of simulating **Lyα radiation pressure feedback** on-the-fly. This is done using a novel M1 moment method for Lyα radiative transfer. Also implemented in LYDION is stellar and IR radiative transfer, detailed non-equilibrium ISM photochemistry and heating/cooling, and dust physics and dynamics.

Taken together, LYDION allows realistic simulations of stellar feedback around young massive stars and star clusters, from Lyα and non-Lyα radiation pressure, and photoionization. This can be used to assess the relative importance of these processes, and inform feedback prescriptions in 3D galaxy/star formation simulations.

This has become increasingly urgent recently, following the launch of the *James Webb Space Telescope* (*JWST*). With the *JWST*, observations of dust-poor galaxies at Cosmic Dawn have become possible, which is precisely the regime where Lyα feedback can strongly regulate star formation.

A full description and tests of LYDION can be found in the methodology paper: [Nebrin et al. (2026), arXiv:2606.02711](https://arxiv.org/abs/2606.02711)

LYDION is primarily written in the Julia programming language ([Bezanson et al. 2017](https://doi.org/10.1137/141000671)), allowing for both fast development and good performance.

# Dependencies

Besides the core Julia language, the following external packages are used:

- HDF5.jl
- Interpolations.jl
- Roots.jl
- SpecialFunctions.jl
- StaticArrays.jl
- QuadGK.jl
- ProgressMeter.jl

These dependencies are listed in 'Project.toml', and can be installed automatically, as shown below.

# Installation

Clone the repository and activate the project environment:

```bash
git clone https://github.com/olofnebrin/LYDION.git
cd LYDION
julia --project=.
```

In the Julia REPL, all required dependencies can be installed using:

```julia
using Pkg
Pkg.instantiate()
```

# How to run an RHD simulation

The files **RunRHDSimulation_SingleStar.jl** and **RunRHDSimulation_StarCluster.jl**
contain the basic setup for running an RHD simulation of a single star or star cluster
embedded in a cloud, respectively, with or without Lyα feedback. In the star cluster setup, 
a Kroupa IMF-averaged (over 0.08 - 100 M⊙), spatially extended stellar source is used.
Cloud initial conditions, the mass of the star or star cluster, physics settings, 
resolution, and so on, can all be changed in those files. 
Running the relevant file will set up the simulation, start the simulation, 
and continuously save output data to an HDF5 file.

To launch, for example, a simulation of a single star from the repository root, 
using 8 CPU threads, do:

```bash
julia --project=. --threads=8 RunRHDSimulation_SingleStar.jl
```

Since checkpoint data and settings are saved, simulations can be restarted 
from the output in the HDF5 file. This can be done by specifying the 
right HDF5 file name in **RestartRHDSimulation_SingleStar.jl** (for the single star
setup), and then do (again assuming 8 CPU threads):

```bash
julia --project=. --threads=8 RestartRHDSimulation_SingleStar.jl
```

# Simulation output 

Simulation results in the HDF5 file can be used for post-simulation analysis. 
A variety of data is saved, e.g. the gas density and temperature, dust bin densities,
chemical abundances, escape fractions, force diagnostics 
(see **SaveHDF5Data_RHD_SingleStar.jl** and **SaveHDF5Data_RHD_StarCluster.jl** 
for a full list). 

You can analyze this data however you want. Some simple Python scripts for making simulation animations and plots are included, some of which were used to create figures for 
the methodology paper (note however that they may use, e.g., non-standard fonts). 

# Lyα radiative transfer tests

LYDION can also be run in Lyα radiative transfer-only mode, without evolving the
full radiation hydrodynamics. Several example and validation setups are included
in the repository, many of which are described further in the methodology paper.

Example run scripts include:

- `RunStaticCloudTest.jl`   (primary script)
- `RunMFStaticCloudTest.jl`
- `RunLyaEscapeFractionStaticCloudTest.jl`
- `RunSourceSizeStaticCloudTest.jl`
- `RunVelocityGradientSpectraTest.jl`

These setups can be useful both for reproducing validation tests in the
methodology paper and for experimenting with Lyα transport in idealized
configurations. Simulation data (e.g. the Lyα energy density, emergent spectra, 
and force multiplier) are saved in an HDF5 file. The full list of saved quantities 
can be found in **SaveHDF5Data_StaticCloudTests.jl**.

# Citation

If you use LYDION in scientific work, or use some of its new modelling
techniques, please cite:

[Nebrin et al. (2026), arXiv:2606.02711](https://arxiv.org/abs/2606.02711)

The LYDION v1.0.0 software release associated with this paper is permanently archived on Zenodo:

[Nebrin (2026), LYDION v1.0.0](https://doi.org/10.5281/zenodo.23374329)

# License

LYDION is distributed under the MIT License. See the `LICENSE` file for details.