
""" This file contains functions for saving data from 
    static cloud tests in an HDF5 file.  """

using .Structs
using .ConstantsAndFunctions     # Needed constants  
using HDF5
using HDF5: create_dataset, h5open, resize!, attrs, datatype


export CreateHDF5File, SaveGridHDF5File, SaveLyαFreqGridHDF5File, SaveSphericalBinsHDF5File,
       SaveAnalyticalSolutionHDF5File, SaveTimeStepHDF5


const N_x_analytical::Int = 300 

function CreateHDF5File( FileName::String, 
                             aτcl::Float64,
                              N_r::Int,
                              N_Z::Int, 
                           Nsaved::Int,
                              N_ν::Int,  # Number of Lyα frequency bins 
                        Nbins_sph::Int   # Number of radial bins for the spherical averaged profile(s)
                        )

    h5open(FileName, "w") do file

        settings = create_group(file, "Settings")
        
        # Save aτ_cl:
        attrs(settings)["atau"]  = string(aτcl)

        # Create a 1D dataset for time history, in units of (aτ)^(1/3) Rcl / c̃:
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

        # Create arrays for spectra:
        create_dataset(file, "x_Analytical", Float64,
                       (N_x_analytical,) )

        create_dataset(file, "Spectrum_Analytical", Float64,
                       (N_x_analytical,) )

        # Create array for *analytical* e_Lyα profile (for any comparisons with numerical results):
        create_dataset(file, "e_Lya_Analytical", Float64,
                       (N_Z,) )

        # Also set up arrays for storing the spherically averaged numerical e_Lyα profile:
        @assert Nbins_sph > 0 "Nbins_sph must be > 0 in CreateHDF5File."

        create_dataset(file, "R_binedges",   Float64, (Nbins_sph+1,))
        create_dataset(file, "R_bincenters", Float64, (Nbins_sph,))

        create_dataset(file, "e_Lya_SphericalHistory", Float32,
                       (Nbins_sph, Nsaved);
                       chunk = (Nbins_sph, 1),
                       filters = Filters.Deflate(3))

        ############################# QUANTITIES #################################

        @assert N_ν > 0 "N_ν must be > 0 in CreateHDF5File."
        
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

        # Global Lyα force multiplier per saved time:
        create_dataset(file, "M_F_History", Float64,
                       (Nsaved,),
                        chunk = (1,))

        # Global net radial Lyα force multiplier per saved time:
        create_dataset(file, "M_F_radial_net_History", Float64,
                       (Nsaved,),
                        chunk = (1,))

        # Global outward radial Lyα force multiplier per saved time:
        create_dataset(file, "M_F_radial_out_History", Float64,
                       (Nsaved,),
                        chunk = (1,))                

        # Lyα luminosity per saved time: 
        create_dataset(file, "L_Lya_History", Float64,
                       (Nsaved,),
                        chunk = (1,))

        # Lyα escape fraction from simulation box:
        create_dataset(file, "f_esc_Lya_History", Float64,
                       (Nsaved,),
                        chunk = (1,))

        # Frequency grid for Lyα:
        create_dataset(file, "ν",   Float64, (N_ν+1,))  # Bin edges
        create_dataset(file, "νC",  Float64, (N_ν,))    # Bin centers
        create_dataset(file, "δνC", Float64, (N_ν,))    # Bin centers relative to line center
        create_dataset(file, "Δν",  Float64, (N_ν,))    # Bin widths

        # Also create storage for saving spectra at outer edge cells:

        # Number of edge cells:
        # Right side:  i = N_r - 1, j = 2:(N_Z-1) → N_Z - 2 cells 
        # Top side:    j = N_Z - 1, i = 2:(N_r-2) → N_r - 3 cells 
        # Bottom side: j = 2,       i = 2:(N_r-2) → N_r - 3 cells 
        N_edge = (N_Z - 2)  +  2 * (N_r - 3)

        # Raw spectral history: [edge_index, ν_index, time]
        create_dataset(file, "J_Lya_EdgeSpectra", Float32,
                       (N_edge, N_ν, Nsaved);
                        chunk = (N_edge, N_ν, 1),
                        filters = Filters.Deflate(3))

        # Normalized spectral history:
        create_dataset(file, "J_Lya_EdgeSpectra_Norm", Float32,
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

    end # End of h5open
    
    return nothing
end


""" Save the spatial grid setup in the created HDF5 file,
    in units of Rcl. """
function SaveGridHDF5File( FileName::String,
                           geometry::GeometryCylindrical{Float64},
                                Rcl::Float64
                           )

    h5open(FileName, "r+") do file

       dset_r               = file["r"]
       dset_r[:]            = geometry.r ./ Rcl
       dset_rCoM            = file["rCoM"]
       dset_rCoM[:]         = geometry.rCoM ./ Rcl 
       dset_Z               = file["Z"]
       dset_Z[:]            = geometry.Z ./ Rcl 
       dset_ZCoM            = file["ZCoM"]
       dset_ZCoM[:]         = geometry.ZCoM ./ Rcl
       dset_ΔV              = file["CellVolumes"]
       dset_ΔV[:,:]         = geometry.ΔV 

    end # End of h5open

    return nothing
end


""" Save the radial bins for the spherically averaged profile(s). """
function SaveSphericalBinsHDF5File( FileName::String,
                                  R_binedges::AbstractArray{Float64,1},
                                         Rcl::Float64
                                   )

    Nbins     = length( R_binedges ) - 1

    # Get bin centers:
    R_centers = 0.5 .* ( R_binedges[1:Nbins] .+ R_binedges[2:Nbins+1] ) 

    # Save, in units of Rcl:
    h5open(FileName, "r+") do file
        file["R_binedges"][:]   = R_binedges ./ Rcl 
        file["R_bincenters"][:] = R_centers  ./ Rcl 
    end

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
       dset_δνC       = file["δνC"]
       dset_δνC[:]    = Lya.δνC

    end # End of h5open

    return nothing
end


""" Save the Lyα analytical solution for the same setup (if there is one)
    in the created HDF5 file. """
function SaveAnalyticalSolutionHDF5File( FileName::String,
                                             ZCoM::AbstractArray{Float64, 1},
                                              τcl::Float64,
                                                T::Float64,
                                              Rcl::Float64,
                                            L_Lyα::Float64,
                                    UniformSource::Bool = true
                                        )

    xAnalytical = LinRange( - 2.5, 2.5, N_x_analytical ) .* ((Compute_a(T) * τcl)^OneThird)

    h5open(FileName, "r+") do file
       
       if UniformSource
         dset_x_Spec             = file["x_Analytical"]
         dset_x_Spec[:]          = xAnalytical

         dset_Spec_Analytic      = file["Spectrum_Analytical"]
         dset_Spec_Analytic[:]   = J_sphere_uni.( xAnalytical, T, τcl )

         dset_e_Lyα_Analytic     = file["e_Lya_Analytical"]
         dset_e_Lyα_Analytic[:]  = eLyα_sphere_uni( ZCoM, L_Lyα, T, τcl, Rcl )
       end

    end # End of h5open

    return nothing
end


""" Save simulation data from one time-step into a
    given prepared HDF5 file. """
function SaveTimeStepHDF5( FileName::String, 
                                Lya::LyαArrays,
                               step::Int,
                                  t::Float64,
                           eLya_sph::AbstractArray{Float64,1} # Spherically averaged e_Lyα profile 
                               )

    Δν  = Lya.Δν # Read bin widths from the Lyα state directly instead of opening file to extract it 
    N_ν = length( Δν )

    h5open(FileName, "r+") do file 

        # Save the time of the current time-step:
        dset_t                        = file["Times"]
        dset_t[step]                  = t

        ###############################################################
        #######                                                 #######
        #######          S A V E  L Y M A N - α  D A T A        #######
        #######                                                 #######
        ###############################################################

        # Save Lyα energy density for all cells:
        dset_e_Lyα                  = file["e_Lya_History"]
        dset_e_Lyα[:, :, step]      = Lya.e_Lyα

        # Save spherically averaged Lyα energy density:
        dset_e_sph                  = file["e_Lya_SphericalHistory"]
        dset_e_sph[:, step]         = eLya_sph

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

        # Save force multipliers:
        dset_M_F                    = file["M_F_History"]
        dset_M_F[step]              = Lya.M_F 

        dset_M_F_radial_net         = file["M_F_radial_net_History"]
        dset_M_F_radial_net[step]   = Lya.M_F_radial_net

        dset_M_F_radial_out         = file["M_F_radial_out_History"]
        dset_M_F_radial_out[step]   = Lya.M_F_radial_out

        # Save Lyα luminosity:
        dset_L_Lyα                  = file["L_Lya_History"]
        dset_L_Lyα[step]            = Lya.L_Lyα

        # Save Lyα escape fraction:
        dset_f_esc_Lyα              = file["f_esc_Lya_History"]
        dset_f_esc_Lyα[step]        = Lya.f_esc_Lyα

        # Save Lyα spectrum at the outer boundary cells:
        dset_J_edge       = file["J_Lya_EdgeSpectra"]        # Raw spectra, not normalized 
        dset_J_edge_Norm  = file["J_Lya_EdgeSpectra_Norm"]   # Normalized spectra, such that ∫dν J = 1
        dset_ridx         = file["J_Lya_Edge_rIndex"]
        dset_zidx         = file["J_Lya_Edge_ZIndex"]

        N_edge       = length(dset_ridx)
        @inbounds for edge in 1:N_edge
            # Get i and j indices:
            i_r = dset_ridx[edge]
            i_Z = dset_zidx[edge]

            # Access & save the Lyα spectrum J(νC) at the edge indices: 
            J                          = @view Lya.J_Lyα[i_r, i_Z, :]
            dset_J_edge[edge, :, step] = J

            # Do the same for the normalized spectra (TODO):
            ∫dν_J = 0.0
            for k in 1:N_ν 
                ∫dν_J += J[k] * Δν[k]
            end

            if ∫dν_J > 0.0
                dset_J_edge_Norm[edge, :, step] = J ./ ∫dν_J
            else 
                dset_J_edge_Norm[edge, :, step] = J
            end

        end

    end # End of h5open

    return nothing
end
