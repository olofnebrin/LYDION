
using .Structs
using .ConstantsAndFunctions     # Needed constants  
using HDF5
using HDF5: create_dataset, create_group, h5open, resize!, attrs, datatype


export CreateHDF5File, SaveGridHDF5File, SaveLyαFreqGridHDF5File, SaveRHDTimeStepHDF5


""" This function sets up an HDF5 file of a given name, containing storage 
    for all arrays and settings for RHD simulations of a single star of mass mstar (M⊙)
    in 2D cylindrical coordinates. """
function CreateHDF5File( FileName::String,                # File name 
                              N_r::Int,                   # Number of cells along r-axis  
                              N_Z::Int,                   # Number of cells along Z-axis 
                        Nminsteps::Int,                   # Minimum number of time-steps in the simulation (sets largest Δt)
                        Nmaxsteps::Int,                   # Maximum number of time-steps in the simulation 
                           Nsaved::Int,                   # Number of target saved time-steps 
                          Gravity::Bool,                  # Whether to add storage for gravity & gravity is on/off
                              Lyα::Bool,                  # Whether to add storage for Lyα arrays & Lyα RT is on/off  
                           Recoil::Bool,                  # Whether atomic recoil is on/off (relevant only if Lyα is on)
                   LyαDestruction::Bool,                  # Whether Lyα destruction (pd > 0) is on/off (relevant only if Lyα is on)
                      RadPressure::Bool,                  # Whether radiation pressure feedback is on/off
                       IRfeedback::Bool,                  # Whether in addition to stellar radiation pressure feedback, IR feedback is on/off
              StellarWindFeedback::Bool,                  # Whether stellar wind feedback is on/off
            DustGrowthDestruction::Bool,                  # Whether dust/growth destruction is on/off
                              OTS::Bool,                  # Whether the on-the-spot approximation is on/off
                                z::Float64,               # Redshift 
                              N_ν::Int = 0,               # Number of Lyα frequency bins 
                             tmax::Float64 = 1e5 * yr,    # Desired time tmax to end the simulation  
                            mstar::Float64 = 30.0,        # Stellar mass (Msun)
                            L_bol::Float64 = 1e5 * Lsun,  # Bolometric luminosity of the star (erg s⁻¹) 
                         dNdt_LyC::Float64 = 1e49,        # Ionizing (EUV1 - EUV3) photon emission rate (s⁻¹)
                      Z_star_Zsun::Float64 = 0.0,         # Metallicity of star(s) in units of Solar metallicity  
                      σ_star_cell::Float64 = 3.0,         # Gaussian extent of star(s) in units of cell width 
                            Z_pos::Float64 = 0.0,         # Central Z-position of stellar mass
                              DG₀::Float64 = Dsun,        # Initial dust-to-gas ratio 
                       Zgas_Zsun₀::Float64 = 1.0,         # Initial gas metallicity
                       BC_r_right::Symbol  = :Outflow,    # BCs at outer r-boundary 
                        BC_Z_left::Symbol  = :Reflect,    # BCs at lower Z-boundary
                       BC_Z_right::Symbol  = :Outflow,    # BCs at upper Z-boundary
                              CFL::Float64 = 0.4,         # CFL factor for hydrodynamics 
                           CFL_RT::Float64 = 0.4,         # CFL factor for M1 RT (both stellar & Lyα)
                                c̃::Float64 = 300 * kms,   # RSL for both stellar & Lyα RT
                  FileDescription::String  = "RHD simulation in $(N_r) × $(N_Z) resolution"
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

        # Save desired tmax & stellar properties:
        attrs(settings)["tmax"]        = string(tmax)
        attrs(settings)["mstar_Msun"]  = string(mstar)
        attrs(settings)["L_bol"]       = L_bol          # Save bolometric luminosity as a Float
        attrs(settings)["dNdt_LyC"]    = dNdt_LyC       # Save ionizing (EUV1-EUV3) photon emission rate as a Float
        attrs(settings)["Z_star_Zsun"] = string(Z_star_Zsun)
        attrs(settings)["σ_star_cell"] = string(σ_star_cell)
        attrs(settings)["Z_pos"]       = string(Z_pos)

        # Save initial dust-to-gas ratio:
        attrs(settings)["Initial_DTG"]       = string(DG₀)

        # Save initial gas metallicity: 
        attrs(settings)["Initial_Zgas_Zsun"] = string(Zgas_Zsun₀)

        # Save redshift:
        attrs(settings)["z"] = string(z)

        # Save BCs:
        attrs(settings)["BC_r_right"]  = string(BC_r_right)
        attrs(settings)["BC_Z_left"]   = string(BC_Z_left)
        attrs(settings)["BC_Z_right"]  = string(BC_Z_right)

        # Save CFL factors:
        attrs(settings)["CFL"]         = string(CFL)
        attrs(settings)["CFL_RT"]      = string(CFL_RT)

        # Save gravity setting (on/off):
        attrs(settings)["Gravity"]     = string(Gravity)

        # Save Lyα setting (on/off):
        attrs(settings)["Lya"]         = string(Lyα)

        # Save atomic recoil for Lyα setting (on/off) - only relevant if Lyα RT is on:
        attrs(settings)["Recoil"]      = string(Recoil)

        # Save Lyα destruction (pd > 0) setting (on/off) - only relevant if Lyα RT is on:
        attrs(settings)["Lya_pd_on"]   = string(LyαDestruction)

        # Save general radiation pressure feedback setting (on/off):
        attrs(settings)["RadPressure"] = string(RadPressure)

        # Save IR radiation pressure feedback setting (on/off) (NOTE: Only matters if RadPressure = true):
        attrs(settings)["IRfeedback"]  = string(IRfeedback)

        # Save stellar wind feedback setting (on/off):
        attrs(settings)["StellarWindFeedback"] = string(StellarWindFeedback)

        # Save on-the-spot approximation setting (on/off):
        attrs(settings)["OTS"]         = string(OTS)

        # Save dust growth/destruction setting (on/off):
        attrs(settings)["DustGrowth/Destruction"]  = string(DustGrowthDestruction)

        # Minimum & maximum number of time-steps:
        attrs(settings)["Nminsteps"] = Nminsteps
        attrs(settings)["Nmaxsteps"] = Nmaxsteps


        ############################# RESTART AT CHECKPOINT ######################

        # We also store needed arrays for re-starting a simulation:

        CheckPoint = create_group(file, "Checkpoint")

        # For saving time and time-step at latest CheckPoint:
        attrs(CheckPoint)["t_current"]  = 0.0
        attrs(CheckPoint)["dt"]         = 0.0
        attrs(CheckPoint)["step"]       = 0

        create_dataset(CheckPoint, "GasDensity", Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "rMomentum",  Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "ZMomentum",  Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "Energy",     Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "e_dual",     Float64, (N_r, N_Z))

        # Thermo-chemistry CheckPoint
        create_dataset(CheckPoint, "T",        Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "xHII",     Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "xH2",      Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "xHeII",    Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "xHeIII",   Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "xDII",     Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "xHD",      Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "xCI",      Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "xCII",     Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "xCO",      Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "xOI",      Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "xOII",     Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "ZC",       Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "ZSil",     Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "Zgasrest", Float64, (N_r, N_Z))

        # Dust CheckPoint:
        create_dataset(CheckPoint, "DustDensity_Sil",   Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "DustrMomentum_Sil", Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "DustZMomentum_Sil", Float64, (N_r, N_Z))

        create_dataset(CheckPoint, "DustDensity_C",   Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "DustrMomentum_C", Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "DustZMomentum_C", Float64, (N_r, N_Z))

        create_dataset(CheckPoint, "DustDensity_PAH",   Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "DustrMomentum_PAH", Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "DustZMomentum_PAH", Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "Trad",              Float64, (N_r, N_Z)) # IR radiation temperature, related to dust temperatures

        # Stellar RT CheckPoint:
        attrs(CheckPoint)["RSL"] = c̃

        create_dataset(CheckPoint, "J_opt",    Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "H_r_opt",  Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "H_Z_opt",  Float64, (N_r, N_Z))

        create_dataset(CheckPoint, "J_FUV",    Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "H_r_FUV",  Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "H_Z_FUV",  Float64, (N_r, N_Z))

        create_dataset(CheckPoint, "J_LW",     Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "H_r_LW",   Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "H_Z_LW",   Float64, (N_r, N_Z))

        create_dataset(CheckPoint, "J_EUV1",   Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "H_r_EUV1", Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "H_Z_EUV1", Float64, (N_r, N_Z))

        create_dataset(CheckPoint, "J_EUV2",   Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "H_r_EUV2", Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "H_Z_EUV2", Float64, (N_r, N_Z))

        create_dataset(CheckPoint, "J_EUV3",   Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "H_r_EUV3", Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "H_Z_EUV3", Float64, (N_r, N_Z))

        create_dataset(CheckPoint, "J_IR",     Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "H_r_IR",   Float64, (N_r, N_Z))
        create_dataset(CheckPoint, "H_Z_IR",   Float64, (N_r, N_Z))

        # Lyα RT CheckPoint:
        if Lyα
            create_dataset(CheckPoint, "J_Lya", Float64,
                          (N_r, N_Z, N_ν);
                          chunk = (N_r, N_Z, 1))

            create_dataset(CheckPoint, "H_r_Lya", Float64,
                          (N_r, N_Z, N_ν);
                          chunk = (N_r, N_Z, 1))

            create_dataset(CheckPoint, "H_Z_Lya", Float64,
                          (N_r, N_Z, N_ν);
                          chunk = (N_r, N_Z, 1))
        end

        ############################# QUANTITIES #################################

        # Create a 3D dataset (r, Z, time) for the gas density:
        create_dataset(file, "GasDensity_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the gas pressure:
        create_dataset(file, "GasPressure_History", Float32,
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

        # Create a 3D dataset (r, Z, time) for the IR radiation temperature:
        create_dataset(file, "Trad_History", Float32,
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

        # Create a 3D dataset (r, Z, time) for the Lyα dust absorption cross-section 
        # per H nucleus:
        create_dataset(file, "LyaDustAbsorptionCrossSections_History", Float32,
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

        # Create a 3D dataset (r, Z, time) for the carbon gas metallicity:
        create_dataset(file, "ZC_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the Si gas metallicity:
        create_dataset(file, "ZSil_History", Float32,
                       (N_r, N_Z, Nsaved);
                       chunk = (N_r, N_Z, 1),
                       filters = Filters.Deflate(3))

        # Create a 3D dataset (r, Z, time) for the rest of the gas metallicity:
        create_dataset(file, "Zgasrest_History", Float32,
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

        # Total gas pressure force anti-parallel to gravity per saved time:
        create_dataset(file, "IntegratedPressureForce_antigrav_History", Float64,
                       (Nsaved,),
                       chunk = (1,))

        if Gravity
            # Create a 3D dataset (r, Z, time) for the gravitational force per unit volume:
            create_dataset(file, "f_gravity_r_History", Float32,
                           (N_r, N_Z, Nsaved);
                            chunk = (N_r, N_Z, 1),
                            filters = Filters.Deflate(3))

            create_dataset(file, "f_gravity_Z_History", Float32,
                           (N_r, N_Z, Nsaved);
                            chunk = (N_r, N_Z, 1),
                            filters = Filters.Deflate(3))

            # Gravitational force diagnostics:
            create_dataset(file, "IntegratedGravityForce_History", Float64,
                           (Nsaved,),
                            chunk = (1,))

            create_dataset(file, "IntegratedGravityForce_radial_net_History", Float64,
                           (Nsaved,),
                            chunk = (1,))
        end

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

            # Lyα force multiplier, in direction anti-parallel 
            # to the gravitational field:
            create_dataset(file, "M_F_History_antigrav_direction", Float64,
                          (Nsaved,),
                          chunk = (1,))

            # Lyα luminosity (2 × simulation luminosity if reflective Z BCs are used):
            create_dataset(file, "L_Lya_History", Float64,
                          (Nsaved,),
                          chunk = (1,))

            # Lyα escape fraction from simulation box:
            create_dataset(file, "f_esc_Lya_History", Float64,
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
                             gravstate::GravityState{Float64},
                                  dust::DustState,
                                    RT::StellarRTArrays,
                                   Lya::Union{LyαArrays, Nothing},
                                  step::Int,
                                     t::Float64,
                                    Δt::Float64,
                               Gravity::Bool
                               )

    h5open(FileName, "r+") do file 

        # Save the time of the current time-step:
        dset_t                        = file["Times"]
        dset_t[step]                  = t

        # Save gas densities at current time-step:
        dset_ρ                        = file["GasDensity_History"]
        dset_ρ[:, :, step]            = hydrostate.ρ

        # Save gas pressure at current time-step:
        dset_P                        = file["GasPressure_History"]
        dset_P[:, :, step]            = hydrostate.P

        # Save r-direction momentum densities at current time-step:
        dset_ρu_r                     = file["rMomentum_History"]
        dset_ρu_r[:, :, step]         = hydrostate.ρu_X

        # Save Z-direction momentum densities at current time-step:
        dset_ρu_Z                     = file["ZMomentum_History"]
        dset_ρu_Z[:, :, step]         = hydrostate.ρu_Y

        # Save temperatures at current time-step:
        dset_T                        = file["T_History"]
        dset_T[:, :, step]            = hydrostate.T

        # Save dust temperatures at current time-step:
        dset_Td_Sil                   = file["Tdust_Sil_History"]
        dset_Td_Sil[:, :, step]       = dust.Td_Sil

        dset_Td_C                     = file["Tdust_C_History"]
        dset_Td_C[:, :, step]         = dust.Td_C

        dset_Td_PAH                   = file["Tdust_PAH_History"]
        dset_Td_PAH[:, :, step]       = dust.Td_PAH

        # Save IR radiation temperature at current time-step:
        dset_Trad                     = file["Trad_History"]
        dset_Trad[:, :, step]         = dust.Trad

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

        # Save Lyα dust absorption cross-section per H nucelus for all cells, 
        # at current time-step:
        dset_σLyα_dust                = file["LyaDustAbsorptionCrossSections_History"]
        dset_σLyα_dust[:, :, step]    = dust.σ_Lyα_per_H

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

        # Save xCO at current time-step:
        dset_xCO                      = file["xCO_History"]
        dset_xCO[:, :, step]          = chemstate.xCO

        # Save xOI at current time-step:
        dset_xOI                      = file["xOI_History"]
        dset_xOI[:, :, step]          = chemstate.xOI

        # Save xOII at current time-step:
        dset_xOII                     = file["xOII_History"]
        dset_xOII[:, :, step]         = chemstate.xOII

        # Save metallicities at current time-step:
        dset_ZC                       = file["ZC_History"]
        dset_ZC[:, :, step]           = chemstate.ZC

        dset_ZSil                     = file["ZSil_History"]
        dset_ZSil[:, :, step]         = chemstate.ZSil

        dset_Zgasrest                 = file["Zgasrest_History"]
        dset_Zgasrest[:, :, step]     = chemstate.Zgasrest

        # Save optical intensity at current time-step:
        dset_J_opt                    = file["J_opt_History"]
        dset_J_opt[:, :, step]        = RT.J_opt

        # Save FUV intensity at current time-step:
        dset_J_FUV                    = file["J_FUV_History"]
        dset_J_FUV[:, :, step]        = RT.J_FUV

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

        # Save IR intensity at current time-step:
        dset_J_IR                     = file["J_IR_History"]
        dset_J_IR[:, :, step]         = RT.J_IR

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

        dset_IntegratedPressureForce_antigrav         = file["IntegratedPressureForce_antigrav_History"]
        dset_IntegratedPressureForce_antigrav[step]   = hydrostate.IntegratedPressureForce_antigrav

        if Gravity
            # Save gravitational forces per unit volume:
            dset_f_gravity_r              = file["f_gravity_r_History"]
            dset_f_gravity_r[:, :, step]  = gravstate.f_gravity_r

            dset_f_gravity_Z              = file["f_gravity_Z_History"]
            dset_f_gravity_Z[:, :, step]  = gravstate.f_gravity_Z

            # Save gravitational force diagnostics:
            dset_IntegratedGravityForce              = file["IntegratedGravityForce_History"]
            dset_IntegratedGravityForce[step]        = gravstate.IntegratedGravityForce

            dset_IntegratedGravityForce_radial       = file["IntegratedGravityForce_radial_net_History"]
            dset_IntegratedGravityForce_radial[step] = gravstate.IntegratedGravityForce_radial_net
        end

        ###############################################################
        #######                                                 #######
        #######          S A V E  L Y M A N - α  D A T A        #######
        #######                                                 #######
        ###############################################################

        if !isnothing(Lya)
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

            # Save force multipliers:
            dset_M_F                    = file["M_F_History"]
            dset_M_F[step]              = Lya.M_F

            dset_M_F_radial_net         = file["M_F_radial_net_History"]
            dset_M_F_radial_net[step]   = Lya.M_F_radial_net

            dset_M_F_radial_out         = file["M_F_radial_out_History"]
            dset_M_F_radial_out[step]   = Lya.M_F_radial_out
        
            dset_M_F_antigrav           = file["M_F_History_antigrav_direction"]
            dset_M_F_antigrav[step]     = Lya.M_F_antigravity

            # Save Lyα luminosity:
            dset_L_Lyα                  = file["L_Lya_History"]
            dset_L_Lyα[step]            = Lya.L_Lyα

            # Save Lyα escape fraction:
            dset_f_esc_Lyα              = file["f_esc_Lya_History"]
            dset_f_esc_Lyα[step]        = Lya.f_esc_Lyα

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
        end

        ###############################################################
        #######                                                 #######
        #######           C H E C K P O I N T  D A T A          #######
        #######                                                 #######
        ###############################################################

        CheckPoint = file["Checkpoint"]
        
        # Current time & time-step:
        attrs(CheckPoint)["t_current"] = t
        attrs(CheckPoint)["step"]      = step
        attrs(CheckPoint)["dt"]        = Δt  

        # Current RSL:
        attrs(CheckPoint)["RSL"]       = RT.c̃

        # Hydro CheckPoint:
        CheckPoint["GasDensity"][:, :] = hydrostate.ρ
        CheckPoint["rMomentum"][:, :]  = hydrostate.ρu_X
        CheckPoint["ZMomentum"][:, :]  = hydrostate.ρu_Y
        CheckPoint["Energy"][:, :]     = hydrostate.E
        CheckPoint["e_dual"][:, :]     = hydrostate.e_dual 
        CheckPoint["T"][:, :]          = hydrostate.T

        # Chemistry CheckPoint:
        CheckPoint["xHII"][:, :]       = chemstate.xHII
        CheckPoint["xH2"][:, :]        = chemstate.xH2
        CheckPoint["xHeII"][:, :]      = chemstate.xHeII
        CheckPoint["xHeIII"][:, :]     = chemstate.xHeIII
        CheckPoint["xDII"][:, :]       = chemstate.xDII
        CheckPoint["xHD"][:, :]        = chemstate.xHD
        CheckPoint["xCI"][:, :]        = chemstate.xCI 
        CheckPoint["xCII"][:, :]       = chemstate.xCII 
        CheckPoint["xCO"][:, :]        = chemstate.xCO
        CheckPoint["xOI"][:, :]        = chemstate.xOI 
        CheckPoint["xOII"][:, :]       = chemstate.xOII 
        CheckPoint["ZC"][:, :]         = chemstate.ZC
        CheckPoint["ZSil"][:, :]       = chemstate.ZSil
        CheckPoint["Zgasrest"][:, :]   = chemstate.Zgasrest

        # Dust CheckPoint:
        CheckPoint["DustDensity_Sil"][:, :]   = @view hydrostate.ρd[1, :, :]
        CheckPoint["DustDensity_C"][:, :]     = @view hydrostate.ρd[2, :, :]
        CheckPoint["DustDensity_PAH"][:, :]   = @view hydrostate.ρd[3, :, :]

        CheckPoint["DustrMomentum_Sil"][:, :] = @view hydrostate.ρdud_X[1, :, :]
        CheckPoint["DustrMomentum_C"][:, :]   = @view hydrostate.ρdud_X[2, :, :]
        CheckPoint["DustrMomentum_PAH"][:, :] = @view hydrostate.ρdud_X[3, :, :]

        CheckPoint["DustZMomentum_Sil"][:, :] = @view hydrostate.ρdud_Y[1, :, :]
        CheckPoint["DustZMomentum_C"][:, :]   = @view hydrostate.ρdud_Y[2, :, :]
        CheckPoint["DustZMomentum_PAH"][:, :] = @view hydrostate.ρdud_Y[3, :, :]
        CheckPoint["Trad"][:, :]              = dust.Trad

        # Stellar RT CheckPoint:
        CheckPoint["J_opt"][:, :]      = RT.J_opt
        CheckPoint["J_FUV"][:, :]      = RT.J_FUV
        CheckPoint["J_LW"][:, :]       = RT.J_LW
        CheckPoint["J_EUV1"][:, :]     = RT.J_EUV1
        CheckPoint["J_EUV2"][:, :]     = RT.J_EUV2
        CheckPoint["J_EUV3"][:, :]     = RT.J_EUV3 
        CheckPoint["J_IR"][:, :]       = RT.J_IR

        CheckPoint["H_r_opt"][:, :]    = RT.H_r_opt
        CheckPoint["H_r_FUV"][:, :]    = RT.H_r_FUV
        CheckPoint["H_r_LW"][:, :]     = RT.H_r_LW
        CheckPoint["H_r_EUV1"][:, :]   = RT.H_r_EUV1
        CheckPoint["H_r_EUV2"][:, :]   = RT.H_r_EUV2
        CheckPoint["H_r_EUV3"][:, :]   = RT.H_r_EUV3 
        CheckPoint["H_r_IR"][:, :]     = RT.H_r_IR

        CheckPoint["H_Z_opt"][:, :]    = RT.H_Z_opt
        CheckPoint["H_Z_FUV"][:, :]    = RT.H_Z_FUV
        CheckPoint["H_Z_LW"][:, :]     = RT.H_Z_LW
        CheckPoint["H_Z_EUV1"][:, :]   = RT.H_Z_EUV1
        CheckPoint["H_Z_EUV2"][:, :]   = RT.H_Z_EUV2
        CheckPoint["H_Z_EUV3"][:, :]   = RT.H_Z_EUV3 
        CheckPoint["H_Z_IR"][:, :]     = RT.H_Z_IR

        # Lyα CheckPoint:
        if !isnothing(Lya)
            CheckPoint["J_Lya"][:, :, :]   = Lya.J_Lyα
            CheckPoint["H_r_Lya"][:, :, :] = Lya.H_r_Lyα
            CheckPoint["H_Z_Lya"][:, :, :] = Lya.H_Z_Lyα
        end

    end # End of h5open

    return nothing
end
