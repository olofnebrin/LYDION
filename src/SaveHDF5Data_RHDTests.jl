
using .Structs
using .ConstantsAndFunctions     # Needed constants  
using HDF5
using HDF5: create_dataset, create_group, h5open, resize!, attrs, datatype


export CreateHDF5File, SaveGridHDF5File, SaveRHDTimeStepHDF5


""" This function sets up an HDF5 file of a given name, containing storage 
    for all arrays and settings for an idealized RHD simulations of a star 
    emitting ionizing photons, in 2D cylindrical coordinates. """
function CreateHDF5File( FileName::String,                # File name 
                        dNdt_EUV1::Float64,               # EUV1 photon emission rate 
                        dNdt_EUV2::Float64,               # EUV2 photon emission rate 
                        dNdt_EUV3::Float64,               # EUV3 photon emission rate 
                            L_bol::Float64,               # Bolometric luminosity (erg s⁻¹)
                              N_r::Int,                   # Number of cells along r-axis  
                              N_Z::Int,                   # Number of cells along Z-axis 
                        Nminsteps::Int,                   # Minimum number of time-steps in the simulation (sets largest Δt)
                        Nmaxsteps::Int,                   # Maximum number of time-steps in the simulation 
                           Nsaved::Int,                   # Number of target saved time-steps 
                      RadPressure::Bool,                  # Whether radiation pressure feedback is on/off
                       IRfeedback::Bool,                  # Whether in addition to stellar radiation pressure feedback, IR feedback is on/off
                            c̃_min::Float64,               # Minimum RSL (cm s⁻¹)
                              OTS::Bool,                  # Whether the on-the-spot approximation is on/off
                                z::Float64,               # Redshift 
                             tmax::Float64 = 1e5 * yr,    # Desired time tmax to end the simulation  
                      σ_star_cell::Float64 = 3.0,         # Gaussian extent of star(s) in units of cell width 
                            Z_pos::Float64 = 0.0,         # Central Z-position of stellar mass
                              DG₀::Float64 = Dsun,        # Initial dust-to-gas ratio 
                        Zgas_Zsun::Float64 = 1.0,         # Gas metallicity
                       BC_r_right::Symbol  = :Outflow,    # BCs at outer r-boundary 
                        BC_Z_left::Symbol  = :Reflect,    # BCs at lower Z-boundary
                       BC_Z_right::Symbol  = :Outflow,    # BCs at upper Z-boundary
                              CFL::Float64 = 0.4,         # CFL factor for hydrodynamics 
                           CFL_RT::Float64 = 0.4,         # CFL factor for M1 RT (both stellar & Lyα)
                                c̃::Float64 = 300 * kms,   # RSL
              FluxMethodStellarRT::String  = "HLL",       # Flux method choice for stellar/IR M1 RT. Can be HLL or GLF
                  FileDescription::String  = "RHD test in $(N_r) × $(N_Z) resolution"
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

        ############################# SETTINGS ###################################

        settings = create_group(file, "Settings")

        # Save simulation/file description:
        attrs(settings)["FileDescription"] = FileDescription

        # Save desired tmax & stellar emission rates and properties:
        attrs(settings)["dNdt_EUV1"]   = dNdt_EUV1
        attrs(settings)["dNdt_EUV2"]   = dNdt_EUV2
        attrs(settings)["dNdt_EUV3"]   = dNdt_EUV3
        attrs(settings)["L_bol"]       = L_bol       # Bolometric luminosity 
        attrs(settings)["tmax"]        = string(tmax)
        attrs(settings)["σ_star_cell"] = string(σ_star_cell)
        attrs(settings)["Z_pos"]       = string(Z_pos)

        # Save initial dust-to-gas ratio:
        attrs(settings)["Initial_DTG"] = string(DG₀)

        # Save initial gas metallicity: 
        attrs(settings)["Initial_Zgas_Zsun"] = string(Zgas_Zsun)

        # Save redshift:
        attrs(settings)["z"] = string(z)

        # Save BCs:
        attrs(settings)["BC_r_right"]  = string(BC_r_right)
        attrs(settings)["BC_Z_left"]   = string(BC_Z_left)
        attrs(settings)["BC_Z_right"]  = string(BC_Z_right)

        # Save CFL factors:
        attrs(settings)["CFL"]         = string(CFL)
        attrs(settings)["CFL_RT"]      = string(CFL_RT)

        # Set flux choice for stellar/IR M1 RT:
        if FluxMethodStellarRT == "HLL"
            attrs(settings)["FluxMethodStellarRT"] = "HLL"
        elseif FluxMethodStellarRT == "GLF"
            attrs(settings)["FluxMethodStellarRT"] = "GLF"
        else 
            error("Unknown flux method $(FluxMethodStellarRT) for stellar/IR M1 RT.")
        end

        # Save general radiation pressure feedback setting (on/off):
        attrs(settings)["RadPressure"] = string(RadPressure)

        # Save IR radiation pressure feedback setting (on/off) (NOTE: Only matters if RadPressure = true):
        attrs(settings)["IRfeedback"]  = string(IRfeedback)

        # Save minimum RSL, c̃_min (in cm s⁻¹):
        attrs(settings)["RSL_min"]     = string(c̃_min)

        # Save on-the-spot approximation setting (on/off):
        attrs(settings)["OTS"]         = string(OTS)

        # Minimum & maximum number of time-steps:
        attrs(settings)["Nminsteps"] = Nminsteps
        attrs(settings)["Nmaxsteps"] = Nmaxsteps

        ############################# QUANTITIES #################################

        # Create a 3D dataset (r, Z, time) for the gas density:
        create_dataset(file, "GasDensity_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the r-direction momentum density:
        create_dataset(file, "rMomentum_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the Z-direction momentum density:
        create_dataset(file, "ZMomentum_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the gas temperature:
        create_dataset(file, "T_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the gas pressure:
        create_dataset(file, "GasPressure_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the dust temperatures:
        create_dataset(file, "Tdust_Sil_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        create_dataset(file, "Tdust_C_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        create_dataset(file, "Tdust_PAH_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the silicate grain charge:
        create_dataset(file, "Zgr_Sil_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the graphite grain charge:
        create_dataset(file, "Zgr_C_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the PAH grain charge:
        create_dataset(file, "Zgr_PAH_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the silicate dust density:
        create_dataset(file, "DustDensity_Sil_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the graphite dust density:
        create_dataset(file, "DustDensity_C_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the PAH dust density:
        create_dataset(file, "DustDensity_PAH_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the r-direction dust momentum densities:
        create_dataset(file, "DustrMomentum_Sil_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        create_dataset(file, "DustrMomentum_C_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        create_dataset(file, "DustrMomentum_PAH_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the Z-direction dust momentum densities:
        create_dataset(file, "DustZMomentum_Sil_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        create_dataset(file, "DustZMomentum_C_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        create_dataset(file, "DustZMomentum_PAH_History", Float32,
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

        # Create a 3D dataset (r, Z, time) for the C I fraction:
        create_dataset(file, "xCI_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the C II fraction:
        create_dataset(file, "xCII_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the C III fraction:
        create_dataset(file, "xCIII_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the CO fraction:
        create_dataset(file, "xCO_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the O I fraction:
        create_dataset(file, "xOI_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the O II fraction:
        create_dataset(file, "xOII_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the O III fraction:
        create_dataset(file, "xOIII_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the O IV fraction:
        create_dataset(file, "xOIV_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the S II fraction:
        create_dataset(file, "xSII_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the S III fraction:
        create_dataset(file, "xSIII_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the S IV fraction:
        create_dataset(file, "xSIV_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the optical intensity:
        create_dataset(file, "J_opt_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the FUV intensity:
        create_dataset(file, "J_FUV_History", Float32,
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

        # Create a 3D dataset (r, Z, time) for the IR intensity:
        create_dataset(file, "J_IR_History", Float32,
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

        # Global stellar radiation 'force multiplier' F_rad / (L/c) per saved time:
        create_dataset(file, "M_F_dir_History", Float64,
                       (Nsaved,),
                       chunk = (1,))

        # Global net radial stellar radiation 'force multiplier' F_rad / (L/c) per saved time:
        create_dataset(file, "M_F_dir_radial_net_History", Float64,
                       (Nsaved,),
                       chunk = (1,))

        # Global outward radial stellar radiation 'force multiplier' F_rad / (L/c) per saved time:
        create_dataset(file, "M_F_dir_radial_out_History", Float64,
                       (Nsaved,),
                       chunk = (1,))

        # Global stellar radiation 'force multiplier' F_rad / (L/c) *anti-parallel to gravity*
        # per saved time:
        create_dataset(file, "M_F_dir_antigrav_direction_History", Float64,
                       (Nsaved,),
                       chunk = (1,))

        # LyC (photon-weighted) escape fracton from simulation volume:
        create_dataset(file, "f_esc_LyC_History", Float64,
                       (Nsaved,),
                       chunk = (1,))

        # Lyman-Werner photon escape fracton from simulation volume:
        create_dataset(file, "f_esc_LW_History", Float64,
                       (Nsaved,),
                       chunk = (1,))

        # Global IR force multiplier per saved time:
        create_dataset(file, "M_F_IR_History", Float64,
                       (Nsaved,),
                       chunk = (1,))

        # Global net radial IR force multiplier per saved time:
        create_dataset(file, "M_F_IR_radial_net_History", Float64,
                       (Nsaved,),
                       chunk = (1,))

        # Global outward radial IR force multiplier per saved time:
        create_dataset(file, "M_F_IR_radial_out_History", Float64,
                       (Nsaved,),
                       chunk = (1,))

        # IR force multiplier, in direction anti-parallel 
        # to the gravitational field:
        create_dataset(file, "M_F_IR_History_antigrav_direction", Float64,
                       (Nsaved,),
                       chunk = (1,))

        # IR luminosity (2 × simulation luminosity if reflective Z BCs are used):
        create_dataset(file, "L_IR_History", Float64,
                       (Nsaved,),
                       chunk = (1,))

        # r-direction mass-weighted velocity dispersion, per saved time:
        create_dataset(file, "rVelocityDispersion_History", Float64,
                      (Nsaved,),
                      chunk = (1,))

        # Z-direction mass-weighted velocity dispersion, per saved time:
        create_dataset(file, "ZVelocityDispersion_History", Float64,
                      (Nsaved,),
                      chunk = (1,))

        # Total mass-weighted velocity dispersion, per saved time:
        create_dataset(file, "VelocityDispersion_History", Float64,
                      (Nsaved,),
                      chunk = (1,))

        # Total gas & dust mass in simulation volume:
        create_dataset(file, "TotalGasMass_History", Float64,
                      (Nsaved,),
                      chunk = (1,))

        create_dataset(file, "TotalDustMass_History", Float64,
                      (Nsaved,),
                      chunk = (1,))

        # Total gas pressure force magnitude per saved time:
        create_dataset(file, "IntegratedPressureForce_History", Float64,
                       (Nsaved,),
                       chunk = (1,))

        # Net radial total gas pressure force per saved time:
        create_dataset(file, "IntegratedPressureForce_radial_net_History", Float64,
                       (Nsaved,),
                       chunk = (1,))

        # Outward radial total gas pressure force per saved time:
        create_dataset(file, "IntegratedPressureForce_radial_out_History", Float64,
                       (Nsaved,),
                       chunk = (1,))


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



""" Save simulation data from one time-step into a
    given prepared HDF5 file. """
function SaveRHDTimeStepHDF5( FileName::String, 
                            hydrostate::HydrodynamicsState{Float64},
                             chemstate::ChemistryState{Float64},
                                  dust::DustState,
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

        # Save r-direction momentum densities at current time-step:
        dset_ρu_r                     = file["rMomentum_History"]
        dset_ρu_r[:, :, step]         = hydrostate.ρu_X

        # Save Z-direction momentum densities at current time-step:
        dset_ρu_Z                     = file["ZMomentum_History"]
        dset_ρu_Z[:, :, step]         = hydrostate.ρu_Y

        # Save temperatures at current time-step:
        dset_T                        = file["T_History"]
        dset_T[:, :, step]            = hydrostate.T

        # Save gas pressure at current time-step:
        dset_P                        = file["GasPressure_History"]
        dset_P[:, :, step]            = hydrostate.P

        # Save dust temperatures at current time-step:
        dset_Td_Sil                   = file["Tdust_Sil_History"]
        dset_Td_Sil[:, :, step]       = dust.Td_Sil

        dset_Td_C                     = file["Tdust_C_History"]
        dset_Td_C[:, :, step]         = dust.Td_C

        dset_Td_PAH                   = file["Tdust_PAH_History"]
        dset_Td_PAH[:, :, step]       = dust.Td_PAH

        # Save grain charges at current time-step 
        dset_Zgr_Sil                  = file["Zgr_Sil_History"]
        dset_Zgr_Sil[:, :, step]      = dust.Zgr_Sil

        dset_Zgr_C                    = file["Zgr_C_History"]
        dset_Zgr_C[:, :, step]        = dust.Zgr_C

        dset_Zgr_PAH                  = file["Zgr_PAH_History"]
        dset_Zgr_PAH[:, :, step]      = dust.Zgr_PAH

        # Save dust densities at current time-step:
        dset_ρd_Sil                   = file["DustDensity_Sil_History"]
        dset_ρd_Sil[:, :, step]       = @view hydrostate.ρd[1,:,:]

        dset_ρd_C                     = file["DustDensity_C_History"]
        dset_ρd_C[:, :, step]         = @view hydrostate.ρd[2,:,:]

        dset_ρd_PAH                   = file["DustDensity_PAH_History"]
        dset_ρd_PAH[:, :, step]       = @view hydrostate.ρd[3,:,:]

        # Save r-direction momentum densities at current time-step:
        dset_ρdud_r_Sil               = file["DustrMomentum_Sil_History"]
        dset_ρdud_r_Sil[:, :, step]   = @view hydrostate.ρdud_X[1,:,:]

        dset_ρdud_r_C                 = file["DustrMomentum_C_History"]
        dset_ρdud_r_C[:, :, step]     = @view hydrostate.ρdud_X[2,:,:]

        dset_ρdud_r_PAH               = file["DustrMomentum_PAH_History"]
        dset_ρdud_r_PAH[:, :, step]   = @view hydrostate.ρdud_X[3,:,:]

        # Save Z-direction momentum densities at current time-step:
        dset_ρdud_Z_Sil               = file["DustZMomentum_Sil_History"]
        dset_ρdud_Z_Sil[:, :, step]   = @view hydrostate.ρdud_Y[1,:,:]

        dset_ρdud_Z_C                 = file["DustZMomentum_C_History"]
        dset_ρdud_Z_C[:, :, step]     = @view hydrostate.ρdud_Y[2,:,:]

        dset_ρdud_Z_PAH               = file["DustZMomentum_PAH_History"]
        dset_ρdud_Z_PAH[:, :, step]   = @view hydrostate.ρdud_Y[3,:,:]

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

        # Save xCI at current time-step:
        dset_xCI                      = file["xCI_History"]
        dset_xCI[:, :, step]          = chemstate.xCI

        # Save xCII at current time-step:
        dset_xCII                     = file["xCII_History"]
        dset_xCII[:, :, step]         = chemstate.xCII

        # Save xCIII at current time-step:
        dset_xCIII                    = file["xCIII_History"]
        dset_xCIII[:, :, step]        = chemstate.xCIII

        # Save xCO at current time-step:
        dset_xCO                      = file["xCO_History"]
        dset_xCO[:, :, step]          = chemstate.xCO

        # Save xOI at current time-step:
        dset_xOI                      = file["xOI_History"]
        dset_xOI[:, :, step]          = chemstate.xOI

        # Save xOII at current time-step:
        dset_xOII                     = file["xOII_History"]
        dset_xOII[:, :, step]         = chemstate.xOII

        # Save xOIII at current time-step:
        dset_xOIII                    = file["xOIII_History"]
        dset_xOIII[:, :, step]        = chemstate.xOIII

        # Save xOIV at current time-step:
        dset_xOIV                     = file["xOIV_History"]
        dset_xOIV[:, :, step]         = chemstate.xOIV

        # Save xSII at current time-step:
        dset_xSII                     = file["xSII_History"]
        dset_xSII[:, :, step]         = chemstate.xSII

        # Save xSIII at current time-step:
        dset_xSIII                    = file["xSIII_History"]
        dset_xSIII[:, :, step]        = chemstate.xSIII

        # Save xSIV at current time-step:
        dset_xSIV                     = file["xSIV_History"]
        dset_xSIV[:, :, step]         = chemstate.xSIV

        # Save optical intensity at current time-step:
        dset_J_opt                    = file["J_opt_History"]
        dset_J_opt[:, :, step]        = @view RT.J[iopt_RT, :, :]

        # Save FUV intensity at current time-step:
        dset_J_FUV                    = file["J_FUV_History"]
        dset_J_FUV[:, :, step]        = @view RT.J[iFUV_RT, :, :]

        # Save LW intensity at current time-step:
        dset_J_LW                     = file["J_LW_History"]
        dset_J_LW[:, :, step]         = @view RT.J[iLW_RT, :, :]

        # Save EUV1 intensity at current time-step:
        dset_J_EUV1                   = file["J_EUV1_History"]
        dset_J_EUV1[:, :, step]       = @view RT.J[iEUV1_RT, :, :]

        # Save EUV2 intensity at current time-step:
        dset_J_EUV2                   = file["J_EUV2_History"]
        dset_J_EUV2[:, :, step]       = @view RT.J[iEUV2_RT, :, :]

        # Save EUV3 intensity at current time-step:
        dset_J_EUV3                   = file["J_EUV3_History"]
        dset_J_EUV3[:, :, step]       = @view RT.J[iEUV3_RT, :, :]

        # Save IR intensity at current time-step:
        dset_J_IR                     = file["J_IR_History"]
        dset_J_IR[:, :, step]         = @view RT.J[iIR_RT, :, :]

        # Save stellar radiation pressure forces:
        dset_f_stellar_r              = file["f_stellar_r_History"]
        dset_f_stellar_r[:, :, step]  = RT.f_stellar_r

        dset_f_stellar_Z              = file["f_stellar_Z_History"]
        dset_f_stellar_Z[:, :, step]  = RT.f_stellar_Z

        # Save LyC and LW escape fractions:
        dset_f_esc_LyC                = file["f_esc_LyC_History"]
        dset_f_esc_LyC[step]          = RT.f_esc_LyC

        dset_f_esc_LW                 = file["f_esc_LW_History"]
        dset_f_esc_LW[step]           = RT.f_esc_LW

        # Save direct stellar radiation pressure force multipliers:
        dset_M_F_dir                  = file["M_F_dir_History"]
        dset_M_F_dir[step]            = RT.M_F_dir

        dset_M_F_dir_radial_net       = file["M_F_dir_radial_net_History"]
        dset_M_F_dir_radial_net[step] = RT.M_F_dir_radial_net

        dset_M_F_dir_radial_out       = file["M_F_dir_radial_out_History"]
        dset_M_F_dir_radial_out[step] = RT.M_F_dir_radial_out

        dset_M_F_dir_antigrav         = file["M_F_dir_antigrav_direction_History"]
        dset_M_F_dir_antigrav[step]   = RT.M_F_dir_antigravity

        # Save IR force multipliers:
        dset_M_F_IR                   = file["M_F_IR_History"]
        dset_M_F_IR[step]             = RT.M_F_IR

        dset_M_F_IR_radial_net        = file["M_F_IR_radial_net_History"]
        dset_M_F_IR_radial_net[step]  = RT.M_F_IR_radial_net

        dset_M_F_IR_radial_out        = file["M_F_IR_radial_out_History"]
        dset_M_F_IR_radial_out[step]  = RT.M_F_IR_radial_out
        
        dset_M_F_IR_antigrav          = file["M_F_IR_History_antigrav_direction"]
        dset_M_F_IR_antigrav[step]    = RT.M_F_IR_antigravity

        # Save IR luminosity:
        dset_L_IR                     = file["L_IR_History"]
        dset_L_IR[step]               = RT.Lum_IR

        # Save mass-weighted velocity dispersion components & 
        # total velocity dispersion:
        dset_σ_r                      = file["rVelocityDispersion_History"]
        dset_σ_r[step]                = hydrostate.σ_X

        dset_σ_Z                      = file["ZVelocityDispersion_History"]
        dset_σ_Z[step]                = hydrostate.σ_Y 

        dset_σ                        = file["VelocityDispersion_History"]
        dset_σ[step]                  = hydrostate.σ

        # Save total gas & dust mass in simulation volume:
        dset_M_simvol                 = file["TotalGasMass_History"]
        dset_M_simvol[step]           = hydrostate.M_simvol

        dset_Md_simvol                = file["TotalDustMass_History"]
        dset_Md_simvol[step]          = hydrostate.Md_simvol

        # Save total gas pressure force measures: 
        dset_IntegratedPressureForce                  = file["IntegratedPressureForce_History"]
        dset_IntegratedPressureForce[step]            = hydrostate.IntegratedPressureForce

        dset_IntegratedPressureForce_radial_net       = file["IntegratedPressureForce_radial_net_History"]
        dset_IntegratedPressureForce_radial_net[step] = hydrostate.IntegratedPressureForce_radial_net

        dset_IntegratedPressureForce_radial_out       = file["IntegratedPressureForce_radial_out_History"]
        dset_IntegratedPressureForce_radial_out[step] = hydrostate.IntegratedPressureForce_radial_out

    end # End of h5open

    return nothing
end
