
using .Structs
using .ConstantsAndFunctions     # Needed constants  
using HDF5
using HDF5: create_dataset, h5open, resize!, attrs, datatype


export CreateHDF5File, SaveGridHDF5File, SaveLyαFreqGridHDF5File, SaveRHDTimeStepHDF5


function CreateHDF5File( FileName::String, 
                              N_r::Int,
                              N_Z::Int, 
                           Nsaved::Int,
                              Lyα::Bool,    # Whether to add storage for Lyα arrays 
                              N_ν::Int = 0  # Number of Lyα frequency bins 
                        )

    h5open(FileName, "w") do file

        # Create a 1D dataset for time history:
        create_dataset(file, "Times", Float64,
                       (Nsaved,),
                       chunk = (1,))

        # Cell centers rCoM: one value per cell.
        create_dataset(file, "rCoM", Float64,
                       (N_r,) )

        # Cell centers ZCoM: one value per cell.
        create_dataset(file, "ZCoM", Float64,
                       (N_Z,) )
        
        # Cell interface radii r: should have N_r+1 entries if storing cell boundaries.
        create_dataset(file, "r", Float64,
                       (N_r+1,) )

        # Cell interface radii Z: should have N_Z+1 entries if storing cell boundaries.
        create_dataset(file, "Z", Float64,
                       (N_Z+1,) )
        
        # Cell volumes ΔV: one value per cell.
        create_dataset(file, "CellVolumes", Float64,
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

        # Create a 3D dataset (r, Z, time) for the gas temperature:
        create_dataset(file, "T_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the H II fraction:
        create_dataset(file, "xHII_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the H₂ fraction:
        create_dataset(file, "xH2_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the He II fraction:
        create_dataset(file, "xHeII_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))
        
        # Create a 3D dataset (r, Z, time) for the He III fraction:
        create_dataset(file, "xHeIII_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the D II fraction:
        create_dataset(file, "xDII_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the HD fraction:
        create_dataset(file, "xHD_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the LW intensity:
        create_dataset(file, "J_LW_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the EUV1 intensity:
        create_dataset(file, "J_EUV1_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the EUV2 intensity:
        create_dataset(file, "J_EUV2_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the EUV3 intensity:
        create_dataset(file, "J_EUV3_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the stellar radiation pressure forces:
        create_dataset(file, "f_stellar_r_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        create_dataset(file, "f_stellar_Z_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        if Lyα
            @assert N_ν > 0 "When Lyα = true, N_ν must be > 0 in CreateHDF5File."



            # Create a 3D dataset (r, Z, time) for the Lyα energy density e_Lyα:
            create_dataset(file, "e_Lya_History", Float32,
                          (N_r, N_Z, Nsaved);
                          chunk = (N_r, N_Z, 1),
                          filters = Filters.Deflate(3))

            # Create a 3D dataset (r, Z, time) for the Lyα emissivity ∫dν_js_Lyα:
            create_dataset(file, "js_Lya_History", Float32,
                          (N_r, N_Z, Nsaved);
                          chunk = (N_r, N_Z, 1),
                          filters = Filters.Deflate(3))

            # Create a 3D dataset (r, Z, time) for the Lyα destruction probability pd:
            create_dataset(file, "pd_History", Float32,
                          (N_r, N_Z, Nsaved);
                          chunk = (N_r, N_Z, 1),
                          filters = Filters.Deflate(3))

            # Create a 3D dataset (r, Z, time) for the Lyα forces:
            create_dataset(file, "f_Lya_r_History", Float32,
                          (N_r, N_Z, Nsaved);
                          chunk = (N_r, N_Z, 1),
                          filters = Filters.Deflate(3))

            create_dataset(file, "f_Lya_Z_History", Float32,
                          (N_r, N_Z, Nsaved);
                          chunk = (N_r, N_Z, 1),
                          filters = Filters.Deflate(3))

            # Global Lyα force multiplier & luminosity per saved time:
            create_dataset(file, "M_F_History", Float64,
                          (Nsaved,),
                          chunk = (1,))

            create_dataset(file, "L_Lya_History", Float64,
                          (Nsaved,),
                          chunk = (1,))

            # Frequency grid for Lyα:
            create_dataset(file, "ν",  Float64, (N_ν+1,))  # Bin edges
            create_dataset(file, "νC", Float64, (N_ν,))    # Bin centers
            create_dataset(file, "Δν", Float64, (N_ν,))    # Bin widths

            # Also create storage for saving spectra at outer edge cells:

            # Number of edge cells:
            # Right side:  i = N_r - 1, j = 2:(N_Z-1) → N_Z - 2 cells 
            # Top side:    j = N_Z - 1, i = 2:(N_r-2) → N_r - 3 cells 
            # Bottom side: j = 2,       i = 2:(N_r-2) → N_r - 3 cells 
            N_edge = (N_Z - 2)  +  2 * (N_r - 3)

            # Spectral history: [edge_index, ν_index, time]
            create_dataset(file, "J_Lya_EdgeSpectra", Float32,
                          (N_edge, N_ν, Nsaved);
                           chunk = (N_edge, N_ν, 1),
                           filters = Filters.Deflate(3))

            # Integer index mapping from edge index → (i_r, i_Z)
            create_dataset(file, "J_Lya_Edge_rIndex", Int32, (N_edge,))
            create_dataset(file, "J_Lya_Edge_ZIndex", Int32, (N_edge,))

            # Fill index mapping (purely in index-space, no geometry needed):
            dset_ridx = file["J_Lya_Edge_rIndex"]
            dset_zidx = file["J_Lya_Edge_ZIndex"]

            edge = 1
            # Right side, i.e. i = N_r - 1, j ∈ 2:(N_Z-1)
            @inbounds for j in 2:(N_Z-1)
                dset_ridx[edge] = N_r - 1
                dset_zidx[edge] = j
                edge += 1
            end

            # Top side, i.e. j = N_Z - 1, i = 2:(N_r-2)
            @inbounds for i in 2:(N_r-2)
                dset_ridx[edge] = i
                dset_zidx[edge] = N_Z - 1
                edge += 1
            end

            # Bottom side, i.e. j = 2, i = 2:(N_r-2) 
            @inbounds for i in 2:(N_r-2)
                dset_ridx[edge] = i
                dset_zidx[edge] = 2
                edge += 1
            end

            @assert edge == N_edge + 1
        end

    end # End of h5open
    
    return nothing
end


""" Save the spatial grid setup in the created HDF5 file. """
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


""" Save the Lyα frequency grid setup in the created HDF5 file. """
function SaveLyαFreqGridHDF5File( FileName::String,
                                       Lya::LyαArrays
                                )

    h5open(FileName, "r+") do file

       dset_ν         = file["ν"]
       dset_ν[:]      = Lya.ν
       dset_νC        = file["νC"]
       dset_νC[:]     = Lya.νC
       dset_Δν        = file["Δν"]
       dset_Δν[:]     = Lya.Δν

    end # End of h5open

    return nothing
end


""" Save simulation data from one time-step into a
    given prepared HDF5 file. """
function SaveRHDTimeStepHDF5( FileName::String, 
                            hydrostate::HydrodynamicsState{Float64},
                             chemstate::ChemistryState{Float64},
                                    RT::StellarRTArrays,
                                  step::Int,
                                     t::Float64
                               )

    h5open(FileName, "r+") do file 

         # Save the time of the current time-step:
        dset_t                        = file["Times"]
        dset_t[step]                  = t

        # Save gas densities at current time-step:
        dset_ρ                        = file["GasDensity_History"]
        dset_ρ[:, :, step]            = hydrostate.ρ

        # Save r-direction velocities at current time-step:
        dset_u_r                      = file["rVelocity_History"]
        dset_u_r[:, :, step]          = hydrostate.u_X

        # Save Z-direction velocities at current time-step:
        dset_u_Z                      = file["ZVelocity_History"]
        dset_u_Z[:, :, step]          = hydrostate.u_Y

        # Save temperatures at current time-step:
        dset_T                        = file["T_History"]
        dset_T[:, :, step]            = hydrostate.T

        # Save xHII at current time-step:
        dset_xHII                     = file["xHII_History"]
        dset_xHII[:, :, step]         = chemstate.xHII

        # Save xH2 at current time-step:
        dset_xH2                      = file["xH2_History"]
        dset_xH2[:, :, step]          = chemstate.xH2

        # Save xHeII at current time-step:
        dset_xHeII                    = file["xHeII_History"]
        dset_xHeII[:, :, step]        = chemstate.xHeII

        # Save xHeIII at current time-step:
        dset_xHeIII                   = file["xHeIII_History"]
        dset_xHeIII[:, :, step]       = chemstate.xHeIII

        # Save xDII at current time-step:
        dset_xDII                     = file["xDII_History"]
        dset_xDII[:, :, step]         = chemstate.xDII

        # Save xHD at current time-step:
        dset_xHD                      = file["xHD_History"]
        dset_xHD[:, :, step]          = chemstate.xHD

        # Save LW intensity at current time-step:
        dset_J_LW                     = file["J_LW_History"]
        dset_J_LW[:, :, step]         = RT.J_LW

        # Save EUV1 intensity at current time-step:
        dset_J_EUV1                   = file["J_EUV1_History"]
        dset_J_EUV1[:, :, step]       = RT.J_EUV1

        # Save EUV2 intensity at current time-step:
        dset_J_EUV2                   = file["J_EUV2_History"]
        dset_J_EUV2[:, :, step]       = RT.J_EUV2

        # Save EUV3 intensity at current time-step:
        dset_J_EUV3                   = file["J_EUV3_History"]
        dset_J_EUV3[:, :, step]       = RT.J_EUV3

        # Save stellar radiation pressure forces:
        dset_f_stellar_r              = file["f_stellar_r_History"]
        dset_f_stellar_r[:, :, step]  = RT.f_stellar_r

        dset_f_stellar_Z              = file["f_stellar_Z_History"]
        dset_f_stellar_Z[:, :, step]  = RT.f_stellar_Z

    end # End of h5open

    return nothing
end


""" Save simulation data from one time-step into a
    given prepared HDF5 file. """
function SaveRHDTimeStepHDF5( FileName::String, 
                            hydrostate::HydrodynamicsState{Float64},
                             chemstate::ChemistryState{Float64},
                                    RT::StellarRTArrays,
                                   Lya::LyαArrays,
                                  step::Int,
                                     t::Float64
                               )

    h5open(FileName, "r+") do file 

        # Save the time of the current time-step:
        dset_t                        = file["Times"]
        dset_t[step]                  = t

        # Save gas densities at current time-step:
        dset_ρ                        = file["GasDensity_History"]
        dset_ρ[:, :, step]            = hydrostate.ρ

        # Save r-direction velocities at current time-step:
        dset_u_r                      = file["rVelocity_History"]
        dset_u_r[:, :, step]          = hydrostate.u_X

        # Save Z-direction velocities at current time-step:
        dset_u_Z                      = file["ZVelocity_History"]
        dset_u_Z[:, :, step]          = hydrostate.u_Y

        # Save temperatures at current time-step:
        dset_T                        = file["T_History"]
        dset_T[:, :, step]            = hydrostate.T

        # Save xHII at current time-step:
        dset_xHII                     = file["xHII_History"]
        dset_xHII[:, :, step]         = chemstate.xHII

        # Save xH2 at current time-step:
        dset_xH2                      = file["xH2_History"]
        dset_xH2[:, :, step]          = chemstate.xH2

        # Save xHeII at current time-step:
        dset_xHeII                    = file["xHeII_History"]
        dset_xHeII[:, :, step]        = chemstate.xHeII

        # Save xHeIII at current time-step:
        dset_xHeIII                   = file["xHeIII_History"]
        dset_xHeIII[:, :, step]       = chemstate.xHeIII

        # Save xDII at current time-step:
        dset_xDII                     = file["xDII_History"]
        dset_xDII[:, :, step]         = chemstate.xDII

        # Save xHD at current time-step:
        dset_xHD                      = file["xHD_History"]
        dset_xHD[:, :, step]          = chemstate.xHD

        # Save LW intensity at current time-step:
        dset_J_LW                     = file["J_LW_History"]
        dset_J_LW[:, :, step]         = RT.J_LW

        # Save EUV1 intensity at current time-step:
        dset_J_EUV1                   = file["J_EUV1_History"]
        dset_J_EUV1[:, :, step]       = RT.J_EUV1

        # Save EUV2 intensity at current time-step:
        dset_J_EUV2                   = file["J_EUV2_History"]
        dset_J_EUV2[:, :, step]       = RT.J_EUV2

        # Save EUV3 intensity at current time-step:
        dset_J_EUV3                   = file["J_EUV3_History"]
        dset_J_EUV3[:, :, step]       = RT.J_EUV3

        # Save stellar radiation pressure forces:
        dset_f_stellar_r              = file["f_stellar_r_History"]
        dset_f_stellar_r[:, :, step]  = RT.f_stellar_r

        dset_f_stellar_Z              = file["f_stellar_Z_History"]
        dset_f_stellar_Z[:, :, step]  = RT.f_stellar_Z

        ###############################################################
        #######                                                 #######
        #######          S A V E  L Y M A N - α  D A T A        #######
        #######                                                 #######
        ###############################################################

        # Save Lyα energy density for all cells:
        dset_e_Lyα                  = file["e_Lya_History"]
        dset_e_Lyα[:, :, step]      = Lya.e_Lyα

        # Save Lyα frequency-integrated emissivity:
        dset_∫dν_js_Lyα             = file["js_Lya_History"]
        dset_∫dν_js_Lyα[:, :, step] = Lya.∫dν_js_Lyα

        # Save Lyα forces:
        dset_f_Lya_r                = file["f_Lya_r_History"]
        dset_f_Lya_r[:, :, step]    = Lya.f_Lyα_r

        dset_f_Lya_Z                = file["f_Lya_Z_History"]
        dset_f_Lya_Z[:, :, step]    = Lya.f_Lyα_Z

        # Save Lyα destruction probability:
        dset_pd                     = file["pd_History"]
        dset_pd[:, :, step]         = Lya.pd

        # Save force multiplier:
        dset_M_F                    = file["M_F_History"]
        dset_M_F[step]              = Lya.M_F 

        # Save Lyα luminosity:
        dset_L_Lyα                  = file["L_Lya_History"]
        dset_L_Lyα[step]            = Lya.L_Lyα

        # Save Lyα spectrum at the outer boundary cells:
        dset_J_edge  = file["J_Lya_EdgeSpectra"]
        dset_ridx    = file["J_Lya_Edge_rIndex"]
        dset_zidx    = file["J_Lya_Edge_ZIndex"]

        N_edge       = length(dset_ridx)
        @inbounds for edge in 1:N_edge
            # Get i and j indices:
            i_r = dset_ridx[edge]
            i_Z = dset_zidx[edge]

            # Access & save the Lyα spectrum J(νC) at the edge indices: 
            dset_J_edge[edge, :, step] = @view Lya.J_Lyα[i_r, i_Z, :]
        end

    end # End of h5open

    return nothing
end
