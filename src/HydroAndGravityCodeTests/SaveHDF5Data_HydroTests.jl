

using HDF5
using HDF5: create_dataset, h5open, resize!, attrs, datatype


export CreateHDF5File, SaveGridHDF5File, SaveTimeStepHDF5


function CreateHDF5File( FileName::String, 
                              N_r::Int,
                              N_Z::Int, 
                           Nsaved::Int
                         )

    h5open(FileName, "w") do file

        # Create a 1D dataset for time history:
        create_dataset(file, "Times", Float64,
                       (Nsaved,),
                       chunk = (1,))

        # Cell centers rCoM: one value per cell.
        create_dataset(file, "rCoM", Float32,
                       (N_r,) )

        # Cell centers ZCoM: one value per cell.
        create_dataset(file, "ZCoM", Float32,
                       (N_Z,) )
        
        # Cell interface radii r: should have N_r+1 entries if storing cell boundaries.
        create_dataset(file, "r", Float32,
                       (N_r+1,) )

        # Cell interface radii Z: should have N_Z+1 entries if storing cell boundaries.
        create_dataset(file, "Z", Float32,
                       (N_Z+1,) )
        
        # Cell volumes ΔV: one value per cell.
        create_dataset(file, "CellVolumes", Float32,
                       (N_r, N_Z) )

        ############################# QUANTITIES #################################

        # Create a 3D dataset (r, Z, time) for the gas density:
        create_dataset(file, "GasDensity_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the r-direction velocity:
        create_dataset(file, "rVelocity_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the Z-direction velocity:
        create_dataset(file, "ZVelocity_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the gas pressure:
        create_dataset(file, "P_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

    end # End of h5open
    
    return nothing
end


""" Save the grid setup in the created HDF5 file (cylindrical coordinates). """
function SaveGridHDF5File( FileName::String,
                           geometry::GeometryCylindrical{Float64}
                           )

    h5open(FileName, "r+") do file

       dset_r               = file["r"]
       dset_r[:]            = geometry.r
       dset_rCoM            = file["rCoM"]
       dset_rCoM[:]         = geometry.rCoM
       dset_Z               = file["Z"]
       dset_Z[:]            = geometry.Z
       dset_ZCoM            = file["ZCoM"]
       dset_ZCoM[:]         = geometry.ZCoM
       dset_ΔV              = file["CellVolumes"]
       dset_ΔV[:,:]         = geometry.ΔV

    end # End of h5open

    return nothing
end


""" Save the grid setup in the created HDF5 file (Cartesian coordinates). """
function SaveGridHDF5File( FileName::String,
                           geometry::GeometryCartesian{Float64}
                           )

    h5open(FileName, "r+") do file
        # Below, "r" means X, and "Z" means Y, for Cartesian coordinates. 

        dset_r       = file["r"]
        dset_r[:]    = geometry.X
        dset_rCoM    = file["rCoM"]
        dset_rCoM[:] = geometry.XCoM

        dset_Z       = file["Z"]
        dset_Z[:]    = geometry.Y
        dset_ZCoM    = file["ZCoM"]
        dset_ZCoM[:] = geometry.YCoM

        dset_ΔV      = file["CellVolumes"]
        dset_ΔV[:,:] = geometry.ΔV

    end

    return nothing
end



""" Save simulation data from one time-step into a
    given prepared HDF5 file. """
function SaveTimeStepHDF5( FileName::String, 
                         hydrostate::HydrodynamicsState{Float64},
                               step::Int,
                                  t::Float64
                               )

    h5open(FileName, "r+") do file 

        # Save gas densities at current time-step:
        dset_ρ                      = file["GasDensity_History"]
        dset_ρ[:, :, step]          = hydrostate.ρ

        # Save r-direction velocities at current time-step:
        dset_u_r                    = file["rVelocity_History"]
        dset_u_r[:, :, step]        = hydrostate.u_X

        # Save Z-direction velocities at current time-step:
        dset_u_Z                    = file["ZVelocity_History"]
        dset_u_Z[:, :, step]        = hydrostate.u_Y

        # Save gas pressures at current time-step:
        dset_P                      = file["P_History"]
        dset_P[:, :, step]          = hydrostate.P

        # Save the time of the current time-step:
        dset_t                      = file["Times"]
        dset_t[step]                = t

    end # End of h5open

    return nothing
end

