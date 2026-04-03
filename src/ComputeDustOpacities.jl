module ComputeDustOpacities 

include( "./ConstantsAndFunctions.jl" )
using .ConstantsAndFunctions  # For constants 
using QuadGK   
using Interpolations          # For interpolations
using DelimitedFiles          # For reading data
using SpecialFunctions
using Printf

###############################################################
#######                                                 #######
#######               C O N S T A N T S                 #######
#######                                                 #######
###############################################################

# Material densities of individual grain species (g cm⁻³).
# All carbonaceous grains (graphite & PAH) share the same bulk density:
const ρgr_Sil::Float64 = 3.5 
const ρgr_C::Float64   = 2.24  

# Grain size integration limits for each species.
# The carbonaceous population is split at 0.01 μm:
#   PAH     : a_min_PAH  ≤  a  <  a_split_C    (PAH optical data)
#   Graphite: a_split_C  ≤  a  ≤  a_max        (Gra_81 optical data)
#   Silicate: a_min_Sil  ≤  a  ≤  a_max        (suvSil_81 optical data)
const a_min_PAH::Float64           = 3.548 * Ang # Minimum grain size in data file, and very close to WD01 lower cutoff (3.5 Å)
const a_split_C::Float64           = 1e-3 * μm   # PAH / graphite boundary
const a_PAH_to_Gra_switch::Float64 = 0.01 * μm   # For carbon grains, use the PAH file data below this grain size, since the PAH file has the correct PAH-graphite interpolation
const a_min_Sil::Float64           = 1e-3 * μm   # Draine silicate grid lower limit
const a_max::Float64               = 10.0 * μm   # Upper limit for all species

# b_C = 3 × 10⁻⁵, R_V = 5.5, model B:
const bC::Float64    = 3e-5 
const a_01::Float64  = 3.5 * Ang
const a_02::Float64  = 30  * Ang
const σ::Float64     = 0.4 
const b_C1::Float64  = 0.75 * bC 
const b_C2::Float64  = 0.25 * bC 

# Parameters for carbonaceous grain size distribution:
const a_tg::Float64  = 0.0120 * μm 
const a_cg::Float64  = 7.28   * μm
const αg::Float64    = - 1.90
const βg::Float64    = - 0.0517
const Cg::Float64    = 2.86e-12 

# Parameters for silicate grain size distribution:
const a_ts::Float64  = 0.211 * μm 
const a_cs::Float64  = 0.1   * μm
const αs::Float64    = - 1.13
const βs::Float64    = - 0.109
const Cs::Float64    = 1.04e-13

###############################################################
#######                                                 #######
#######   G R A I N  S I Z E  D I S T R I B U T I O N   #######
#######                                                 #######
###############################################################

# Amplitudes B1, B2 of the two log-normal VSG/PAH populations
# in the carbonaceous size distribution (WD01 eqs. 2-3):
const B1 = ( ( (3 / (2*π)^(3/2)) * exp( - 4.5 * σ^2) / (ρgr_C * (a_01^3) * σ) )
           * b_C1 * mC / ( 1 + erf( 3*σ/sqrt(2) + log(a_01 / (3.5 * Ang))/(σ * sqrt(2)) ) ) ) 

const B2 = ( ( (3 / (2*π)^(3/2)) * exp( - 4.5 * σ^2) / (ρgr_C * (a_02^3) * σ) )
           * b_C2 * mC / ( 1 + erf( 3*σ/sqrt(2) + log(a_02 / (3.5 * Ang))/(σ * sqrt(2)) ) ) ) 

""" Eqs. (2)-(3) in Weingartner & Draine (2001). 
    Sum of two log-normal populations representing VSGs/PAHs. """
function D( a::Float64 )

    if a > 3.5 * Ang
        D = ( (B1 / a) * exp( - (1/2) * (log(a / a_01)/σ)^2 ) 
            + (B2 / a) * exp( - (1/2) * (log(a / a_02)/σ)^2 ) )
    else
        D = 0.0
    end

    return D 
end


""" Eq. (6) in Weingartner & Draine (2001). 
    Curvature correction factor for the power-law part of the size distribution. """
function F( a::Float64, β::Float64, a_t::Float64 )

    if β ≥ 0.0
        return 1 + β * a / a_t 
    else
        return 1 / (1 - β * a / a_t)
    end
end


""" Carbonaceous (graphite + PAH) grain size distribution ∂ngr/∂a,
    in units of cm⁻⁴ per H nucleon (WD01, R_V = 5.5, b_C = 3e-5, model B).
    Valid for a_min_PAH ≤ a ≤ a_max; returns 0 outside this range.
    The PAH and graphite sub-populations are obtained by integrating
    this distribution over [a_min_PAH, a_split_C) and [a_split_C, a_max]
    respectively. """
function ∂ngr_∂a_C( a::Float64 )

    # Return zero outside the full carbonaceous size range:
    if a ≤ a_min_PAH || a ≥ a_max
        return 0.0
    end

    # Exponential cutoff factor (eq. 4 in WD01):
    if a < a_tg
        f = 1.0
    else
        f = exp( - ((a - a_tg)/a_cg)^3 )
    end

    return D(a) + (Cg / a) * ((a/a_tg)^αg) * F(a, βg, a_tg) * f
end


""" Silicate grain size distribution ∂ngr/∂a,
    in units of cm⁻⁴ per H nucleon (WD01, R_V = 5.5, model B).
    Valid for a_min_Sil ≤ a ≤ a_max; returns 0 outside this range. """
function ∂ngr_∂a_Sil( a::Float64 )

    # Return zero outside the silicate size range:
    if a ≤ a_min_Sil || a ≥ a_max
        return 0.0
    end

    # Exponential cutoff factor (eq. 4 in WD01):
    if a < a_ts 
        f = 1.0
    else
        f = exp( - ((a - a_ts)/a_cs)^3 )
    end

    return (Cs / a) * ((a/a_ts)^αs) * F(a, βs, a_ts) * f 
end

###############################################################
#######                                                 #######
#######     I N T E R P O L A T E  D U S T  D A T A     #######
#######                                                 #######
###############################################################

# Band edge energies in erg. Bands defined in ConstantsAndFunctions.jl:
#   opt  : 1.0  - 5.8   eV
#   FUV  : 5.8  - 11.2  eV
#   LW   : 11.2 - 13.6  eV  (Lyman-Werner)
#   EUV1 : 13.6 - 24.6  eV  (H I ionizing)
#   EUV2 : 24.6 - 54.4  eV  (He I ionizing)
#   EUV3 : 54.4 - 1000  eV  (He II ionizing; upper limit well inside Draine grid)
const E_bands_lo = [ 1.0 * eV,   5.8 * eV,  11.2 * eV, 13.6 * eV, 24.6 * eV,  54.4 * eV ]
const E_bands_hi = [ 5.8 * eV,  11.2 * eV,  13.6 * eV, 24.6 * eV, 54.4 * eV, 1000.0 * eV ]
const N_bands    = length(E_bands_lo)

# Corresponding wavelength edges in cm (λ = hc/E).
# Note the inversion: a high-energy (short-λ) edge maps to λ_lo, and vice versa:
const λ_bands_lo_cm = [ h * c / E_hi for E_hi in E_bands_hi ]  # short-λ edge of each band
const λ_bands_hi_cm = [ h * c / E_lo for E_lo in E_bands_lo ]  # long-λ edge of each band


"""
    ReadDraineDustFile(filepath::String; isPAH::Bool=false)

Parse a Draine optical properties file (e.g. Gra_81, suvSil_81, or PAH_30.txt).

The PAH file differs from the graphite/silicate files in two ways:
  - 5 data columns: w, Q_ext, Q_abs, Q_sca, g  (vs. 4: w, Q_abs, Q_sca, g)
  - NWAV = 1201  (vs. 241)
  - NRAD = 30    (vs. 81)

Returns:
  - a_grid_cm  :: Vector{Float64}  -- grain radii in cm  (ascending)
  - λ_grid_cm  :: Vector{Float64}  -- wavelengths in cm  (ascending)
  - Q_abs_grid :: Matrix{Float64}  -- Q_abs[i_a, i_λ]
  - Q_F_grid   :: Matrix{Float64}  -- Q_F[i_a, i_λ] = Q_abs + (1-g)*Q_sca
"""
function ReadDraineDustFile( filepath::String; isPAH::Bool = false )

    lines = readlines(filepath)

    # Set grid dimensions based on file type:
    NRAD = isPAH ? 30   : 81
    NWAV = isPAH ? 1201 : 241

    # Allocate output arrays (filled in the loop below):
    a_grid_μm  = Vector{Float64}(undef, NRAD)
    Q_abs_grid = Matrix{Float64}(undef, NRAD, NWAV)
    Q_F_grid   = Matrix{Float64}(undef, NRAD, NWAV)
    λ_grid_μm  = Vector{Float64}(undef, NWAV)

    # λ_read: the wavelength grid is identical for every grain block,
    # so we only need to read it from the first block:
    λ_read    = false
    grain_idx = 0   # Counts which grain block we are currently reading
    i         = 1   # Current line index

    while i ≤ length(lines)

        line = strip(lines[i])

        # The sentinel string "= radius(micron)" appears exactly once per grain
        # block, marking its start. All other lines (global headers, blank lines,
        # column headers) do not contain this string and fall through to else:
        if occursin("= radius(micron)", line)

            grain_idx += 1

            # The grain radius sits before the "=" on this line, e.g.:
            # "1.000E-03 = radius(micron) Graphite..."
            a_grid_μm[grain_idx] = parse(Float64, split(line, "=")[1])

            i += 2  # Skip column header line

            # Read the NWAV data lines for this grain:
            for j in 1:NWAV

                # Fix a formatting quirk in the PAH file where two consecutive numbers
                # can appear fused when a negative-exponent value has no leading space,
                # e.g. "2.048E-12-1.07E-08". Insert a space before the stray minus sign:
                raw   = replace(strip(lines[i]), r"(\d)(-)" => s"\1 \2")
                parts = split(raw)

                λ_μm  = parse(Float64, parts[1])

                # PAH file has extra Q_ext in column 2; Q_abs is then column 3.
                # Graphite/silicate files have Q_abs directly in column 2:
                if isPAH
                    q_abs = parse(Float64, parts[3])
                    q_sca = parse(Float64, parts[4])
                    g_cos = parse(Float64, parts[5])
                else
                    q_abs = parse(Float64, parts[2])
                    q_sca = parse(Float64, parts[3])
                    g_cos = parse(Float64, parts[4])
                end

                # Only store λ on the first grain block; it is the same for all:
                !λ_read && ( λ_grid_μm[j] = λ_μm )

                # Store absorption efficiency directly:
                Q_abs_grid[grain_idx, j] = q_abs

                # Flux absorption efficiency: Q_F = Q_abs + (1 - g)*Q_sca.
                # The (1-g) factor reduces the contribution of forward-scattered
                # photons (g → 1), which transfer little net momentum:
                Q_F_grid[grain_idx, j] = q_abs + (1.0 - g_cos) * q_sca

                i += 1
            end

            λ_read = true  # λ grid is now stored; skip it for subsequent grains

        else
            i += 1  # Non-data line (header, blank): just advance
        end

    end

    # Convert micron → cm using μm from ConstantsAndFunctions:
    a_grid_cm = a_grid_μm .* μm   # Grain radii in cm (already ascending in file)

    # Draine's λ runs large→small (1000 μm down to 0.001 μm); reverse to
    # ascending order as required by Interpolations.jl's Gridded interpolator:
    λ_grid_cm  = reverse(λ_grid_μm) .* μm
    Q_abs_grid = Q_abs_grid[:, end:-1:1]  # Reverse λ axis, keep a axis unchanged
    Q_F_grid   = Q_F_grid[:,   end:-1:1]  # Same

    return a_grid_cm, λ_grid_cm, Q_abs_grid, Q_F_grid
end


""" BuildDustInterpolations(filepath::String; isPAH::Bool=false)

  Returns two 2D interpolation functions on (a_cm, λ_cm):
    Q_abs_itp(a_cm, λ_cm)  ->  Q_abs(a, λ)
    Q_F_itp(a_cm, λ_cm)    ->  Q_F(a, λ) = Q_abs + (1-g)*Q_sca

  Interpolation is bilinear in (log a, log λ), which is natural since both
  Q values vary smoothly and by many orders of magnitude across the grid.
  Extrapolation outside the grid uses flat (constant) boundary conditions.
"""
function BuildDustInterpolations( filepath::String; isPAH::Bool = false )

    # Read data from Draine file:
    a_cm, λ_cm, Q_abs_grid, Q_F_grid = ReadDraineDustFile(filepath; isPAH = isPAH)

    log_a = log.(a_cm)
    log_λ = log.(λ_cm)

    # Build interpolants in log(Q) space for accuracy across many orders of magnitude:
    itp_logQabs = extrapolate(
        interpolate((log_a, log_λ), log.(Q_abs_grid), Gridded(Linear())), Flat())
    itp_logQF   = extrapolate(
        interpolate((log_a, log_λ), log.(Q_F_grid),   Gridded(Linear())), Flat())

    # Wrap as closures taking physical (a_cm, λ_cm) and returning Q values:
    Q_abs_itp(a_cm, λ_cm) = exp(itp_logQabs(log(a_cm), log(λ_cm)))
    Q_F_itp(a_cm, λ_cm)   = exp(itp_logQF(  log(a_cm), log(λ_cm)))

    return Q_abs_itp, Q_F_itp
end


""" ComputeBandOpacities(Q_abs_itp, Q_F_itp, ∂ngr_∂a, ρgr, Teff, a_low, a_high)

Compute the BB-spectrum- and grain-size-averaged dust opacities for each
radiation band, for a single dust species.

The opacity per gram of dust is:

  κ_X(b) = ∫ Q_X_BB(a,b) π a² (∂ngr/∂a) da
            ─────────────────────────────────────
            ∫ (4π/3) a³ ρgr (∂ngr/∂a) da

where Q_X_BB(a,b) is the BB-weighted average of Q_X(a,λ) over band b.
The denominator is the total dust mass per H nucleon, independent of band.

Arguments:
  - Q_abs_itp  : interpolant Q_abs(a_cm, λ_cm), from BuildDustInterpolations
  - Q_F_itp    : interpolant Q_F(a_cm, λ_cm),   from BuildDustInterpolations
  - ∂ngr_∂a    : grain size distribution ∂ngr/∂a (cm⁻⁴ per H nucleon)
  - ρgr        : material density of grain species (g cm⁻³)
  - Teff       : effective blackbody temperature (K)
  - a_low      : lower grain size integration limit (cm)
  - a_high     : upper grain size integration limit (cm)

Returns:
  - κ_abs  :: Vector{Float64} (length N_bands) -- absorption opacity (cm² g⁻¹)
  - κ_F    :: Vector{Float64} (length N_bands) -- flux-mean opacity   (cm² g⁻¹)
"""
function ComputeBandOpacities( Q_abs_itp,
                               Q_F_itp,
                               ∂ngr_∂a::Function,
                                   ρgr::Float64,
                                  Teff::Float64,
                                 a_low::Float64,
                                a_high::Float64 )

    # Planck function B_λ(λ, T) in cgs (erg s⁻¹ cm⁻³ sr⁻¹).
    # Only the spectral shape matters here since the normalisation cancels
    # in the BB-weighted average:
    inv_kBT   = 1.0 / (kB * Teff)
    B_λ(λ_cm) = (2.0 * h * c^2 / λ_cm^5) / expm1(h * c * inv_kBT / λ_cm)

    # BB-weighted average of Q_X(a, λ) over band b, at fixed grain size a.
    # This is the inner integral, evaluated once per (a, b) pair during the
    # outer quadrature over grain size:
    function Q_BB(Q_itp, a_cm, b::Int)
        λ_lo = λ_bands_lo_cm[b]
        λ_hi = λ_bands_hi_cm[b]
        num, _ = quadgk( λ -> Q_itp(a_cm, λ) * B_λ(λ), λ_lo, λ_hi, rtol=1e-4 )
        den, _ = quadgk( λ -> B_λ(λ),                  λ_lo, λ_hi, rtol=1e-4 )
        return num / den
    end

    # Denominator: total dust mass per H nucleon (same for all bands):
    denom, _ = quadgk( a -> (Four_π / 3) * (a^3) * ρgr * ∂ngr_∂a(a),
                       a_low, a_high, rtol=1e-4 )

    # Initialize output opacity vectors:
    κ_abs = Vector{Float64}(undef, N_bands)
    κ_F   = Vector{Float64}(undef, N_bands)

    for b in 1:N_bands

        # Numerators: BB-averaged cross-section weighted by the size distribution:
        num_abs, _ = quadgk( a -> Q_BB(Q_abs_itp, a, b) * π * (a^2) * ∂ngr_∂a(a),
                             a_low, a_high, rtol=1e-4 )
        num_F,   _ = quadgk( a -> Q_BB(Q_F_itp,   a, b) * π * (a^2) * ∂ngr_∂a(a),
                             a_low, a_high, rtol=1e-4 )

        κ_abs[b] = num_abs / denom
        κ_F[b]   = num_F   / denom

    end

    return κ_abs, κ_F
end


###############################################################
#######                                                 #######
#######         C O M P U T E  D U S T  D A T A         #######
#######                                                 #######
###############################################################

# Band labels for output (6 standard radiation bands + Lyα pseudo-band):
const band_labels = ["opt", "FUV", "LW", "EUV1", "EUV2", "EUV3", "Lya"]

# Lyα as a narrow pseudo-band: 10.19 – 10.21 eV.
# At 10.2 eV this sits just below the LW band, inside FUV.
# The band is only 0.02 eV wide, so the BB weight is essentially
# constant across it and we evaluate Q at the line centre directly:
const E_Lya_lo    = 10.19 * eV
const E_Lya_hi    = 10.21 * eV
const λ_Lya_cm    = c / νLyα   # Central Lyα wavelength in cm


""" SaveOpacities(κ_abs, κ_F, filepath, species, Teff)

  Write opacity table to a plain-text file. Columns are:
    Band  E_low (eV)  E_high (eV)  κ_abs (cm² g⁻¹)  κ_F (cm² g⁻¹)

  The last row is always the Lyα pseudo-band.
"""
function SaveOpacities( κ_abs::Vector{Float64},
                          κ_F::Vector{Float64},
                     filepath::String,
                      species::String,
                         Teff::Float64 )

    open(filepath, "w") do f

        # Header:
        println(f, "# Dust opacities for species: $species")
        println(f, "# Computed by ComputeDustOpacities.jl (LYDION)")
        println(f, "# Grain size distribution: Weingartner & Draine (2001), R_V = 5.5, b_C = 3e-5")
        println(f, "# BB temperature: Teff = $(Teff) K")
        println(f, "# Columns: Band  E_low (eV)  E_high (eV)  κ_abs (cm² g⁻¹)  κ_F (cm² g⁻¹)")
        println(f, "#")

        # Standard bands (indices 1:N_bands in κ_abs, κ_F):
        E_lo_eV = E_bands_lo ./ eV
        E_hi_eV = E_bands_hi ./ eV
        for b in 1:N_bands
            @printf(f, "%-6s  %10.4f  %10.4f  %14.6e  %14.6e\n",
                    band_labels[b], E_lo_eV[b], E_hi_eV[b], κ_abs[b], κ_F[b])
        end

        # Lyα pseudo-band (index N_bands+1 in κ_abs, κ_F):
        @printf(f, "%-6s  %10.4f  %10.4f  %14.6e  %14.6e\n",
                band_labels[end], E_Lya_lo/eV, E_Lya_hi/eV, κ_abs[end], κ_F[end])

    end

    println("Saved opacities to: $filepath")
end


""" ComputeLyαOpacity(Q_abs_itp, Q_F_itp, ∂ngr_∂a, ρgr, Teff, a_low, a_high)

Compute dust opacities at the Lyα wavelength (10.19 - 10.21 eV).
The band is only 0.02 eV wide, so the BB-weighted average of Q reduces
to a direct evaluation at the Lyα line centre λ_Lya = c / νLyα.

Arguments and return values are the same as ComputeBandOpacities, but
returns a scalar pair (κ_abs_Lya, κ_F_Lya) instead of a vector.
"""
function ComputeLyαOpacity( Q_abs_itp,
                             Q_F_itp,
                             ∂ngr_∂a::Function,
                                 ρgr::Float64,
                                Teff::Float64,
                               a_low::Float64,
                              a_high::Float64 )

    # Numerators: Q evaluated at λ_Lya, weighted by the size distribution:
    num_abs, _ = quadgk( a -> Q_abs_itp(a, λ_Lya_cm) * π * a^2 * ∂ngr_∂a(a),
                         a_low, a_high, rtol=1e-4 )
    num_F,   _ = quadgk( a -> Q_F_itp(  a, λ_Lya_cm) * π * a^2 * ∂ngr_∂a(a),
                         a_low, a_high, rtol=1e-4 )

    # Denominator: total dust mass per H nucleon:
    denom,   _ = quadgk( a -> (Four_π/3) * a^3 * ρgr * ∂ngr_∂a(a),
                         a_low, a_high, rtol=1e-4 )

    return num_abs / denom, num_F / denom
end


""" ComputeGrainSizeStats(Q_abs_PAH_itp, Q_abs_C_itp, Q_abs_Sil_itp, outdir)

Compute and save grain size statistics for each dust species:
  - Area-weighted mean grain size <a>_area = ∫a³(∂n/∂a)da / ∫a²(∂n/∂a)da
  - Mass fraction of total dust in each species: f_s = M_s / M_tot
  - BB-band-averaged Q_abs evaluated at the area-weighted mean grain size
    for each species, for use in grain charging (Dust.jl)
"""
function ComputeGrainSizeStats( Q_abs_PAH_itp,
                                Q_abs_C_itp,
                                Q_abs_Sil_itp,
                                outdir::String,
                                Teff::Float64  )

    # --- Area-weighted mean grain sizes ---
    num2_PAH, _ = quadgk( a -> ∂ngr_∂a_C(a)   * a^2, a_min_PAH, a_split_C; rtol=1e-9 )
    num3_PAH, _ = quadgk( a -> ∂ngr_∂a_C(a)   * a^3, a_min_PAH, a_split_C; rtol=1e-9 )
    a_area_PAH  = num3_PAH / num2_PAH

    num2_C,   _ = quadgk( a -> ∂ngr_∂a_C(a)   * a^2, a_split_C, a_max;     rtol=1e-9 )
    num3_C,   _ = quadgk( a -> ∂ngr_∂a_C(a)   * a^3, a_split_C, a_max;     rtol=1e-9 )
    a_area_C    = num3_C / num2_C

    num2_Sil, _ = quadgk( a -> ∂ngr_∂a_Sil(a) * a^2, a_min_Sil, a_max;     rtol=1e-9 )
    num3_Sil, _ = quadgk( a -> ∂ngr_∂a_Sil(a) * a^3, a_min_Sil, a_max;     rtol=1e-9 )
    a_area_Sil  = num3_Sil / num2_Sil

    # --- Dust mass fractions ---
    M_PAH, _ = quadgk( a -> ∂ngr_∂a_C(a)   * a^3, a_min_PAH, a_split_C; rtol=1e-9 )
    M_C,   _ = quadgk( a -> ∂ngr_∂a_C(a)   * a^3, a_split_C, a_max;     rtol=1e-9 )
    M_Sil, _ = quadgk( a -> ∂ngr_∂a_Sil(a) * a^3, a_min_Sil, a_max;     rtol=1e-9 )
    M_PAH *= ρgr_C;  M_C *= ρgr_C;  M_Sil *= ρgr_Sil
    M_tot  = M_PAH + M_C + M_Sil
    f_PAH  = M_PAH / M_tot;  f_C = M_C / M_tot;  f_Sil = M_Sil / M_tot

    # --- Q_abs at area-weighted mean grain sizes, per band ---
    # For graphite, use PAH optical data below 0.01 μm (smooth transition),
    # graphite data above:
    Q_abs_C_at_area(λ_cm) = a_area_C < a_PAH_to_Gra_switch ?
                             Q_abs_PAH_itp(a_area_C, λ_cm) :
                             Q_abs_C_itp(  a_area_C, λ_cm)

    # BB-weighted Q_abs for each bin at area-weighted mean grain size:
    inv_kBT = 1.0 / (kB * Teff)   
    B_λ(λ)  = (2 * h * (c^2) / (λ^5)) / expm1( h * c * inv_kBT / λ )

    function Q_abs_band(Q_at_a::Function, b::Int)
        λ_lo = λ_bands_lo_cm[b]
        λ_hi = λ_bands_hi_cm[b]
        num, _ = quadgk( λ -> Q_at_a(λ) * B_λ(λ), λ_lo, λ_hi; rtol=1e-4 )
        den, _ = quadgk( λ -> B_λ(λ),              λ_lo, λ_hi; rtol=1e-4 )
        return num / den
    end

    # Evaluate Q_abs at a_area for each species and each band + Lyα:
    Qabs_PAH_bands = [ Q_abs_band(λ -> Q_abs_PAH_itp(a_area_PAH, λ), b) for b in 1:N_bands ]
    Qabs_C_bands   = [ Q_abs_band(Q_abs_C_at_area, b)                    for b in 1:N_bands ]
    Qabs_Sil_bands = [ Q_abs_band(λ -> Q_abs_Sil_itp(a_area_Sil, λ), b) for b in 1:N_bands ]

    # Lyα: evaluate directly at line centre
    Qabs_PAH_Lya = Q_abs_PAH_itp(a_area_PAH, λ_Lya_cm)
    Qabs_C_Lya   = Q_abs_C_at_area(λ_Lya_cm)
    Qabs_Sil_Lya = Q_abs_Sil_itp(a_area_Sil, λ_Lya_cm)

    # --- Save grain size stats ---
    filepath = joinpath(outdir, "grain_size_stats.txt")
    open(filepath, "w") do f
        println(f, "# Grain size statistics for WD01 dust model (R_V = 5.5, b_C = 3e-5)")
        println(f, "# PAH/graphite split at a_split_C = $(a_split_C/μm) μm")
        println(f, "# Graphite optical data (PAH file) used for a < $(a_PAH_to_Gra_switch/μm) μm")
        println(f, "#")
        println(f, "# Area-weighted mean grain sizes <a>_area = ∫a³(∂n/∂a)da / ∫a²(∂n/∂a)da:")
        @printf(f, "  a_area_PAH      = %.4e cm  =  %.4e μm\n", a_area_PAH, a_area_PAH/μm)
        @printf(f, "  a_area_graphite = %.4e cm  =  %.4e μm\n", a_area_C,   a_area_C/μm)
        @printf(f, "  a_area_silicate = %.4e cm  =  %.4e μm\n", a_area_Sil, a_area_Sil/μm)
        println(f, "#")
        println(f, "# Dust mass fractions f_s = M_s / M_tot:")
        @printf(f, "  f_PAH           = %.6f\n", f_PAH)
        @printf(f, "  f_graphite      = %.6f\n", f_C)
        @printf(f, "  f_silicate      = %.6f\n", f_Sil)
        @printf(f, "  f_PAH + f_C + f_Sil = %.6f  (sanity check, should be 1)\n",
                f_PAH + f_C + f_Sil)
    end
    println("Saved grain size stats to: $filepath")

    # --- Save Q_abs at representative grain sizes ---
    filepath_Q = joinpath(outdir, "Qabs_at_area_mean.txt")
    open(filepath_Q, "w") do f
        println(f, "# Q_abs evaluated at area-weighted mean grain sizes, BB-averaged per band")
        println(f, "# Grain sizes: a_PAH=$(a_area_PAH/μm) μm, a_C=$(a_area_C/μm) μm, a_Sil=$(a_area_Sil/μm) μm")
        println(f, "# BB temperature for band averaging: Teff = $(Teff) K")
        println(f, "# Columns: Band  E_lo(eV)  E_hi(eV)  Q_abs_PAH  Q_abs_graphite  Q_abs_silicate")
        println(f, "#")
        E_lo_eV = E_bands_lo ./ eV
        E_hi_eV = E_bands_hi ./ eV
        for b in 1:N_bands
            @printf(f, "%-6s  %10.4f  %10.4f  %12.6e  %12.6e  %12.6e\n",
                    band_labels[b], E_lo_eV[b], E_hi_eV[b],
                    Qabs_PAH_bands[b], Qabs_C_bands[b], Qabs_Sil_bands[b])
        end
        @printf(f, "%-6s  %10.4f  %10.4f  %12.6e  %12.6e  %12.6e\n",
                "Lya", E_Lya_lo/eV, E_Lya_hi/eV,
                Qabs_PAH_Lya, Qabs_C_Lya, Qabs_Sil_Lya)
    end
    println("Saved Q_abs at area-weighted mean sizes to: $filepath_Q")

    return a_area_PAH, a_area_C, a_area_Sil, f_PAH, f_C, f_Sil
end


""" ComputeThermalOpacities(Q_abs_itp, Q_F_itp, ∂ngr_∂a, ρgr, a_low, a_high, outdir, species)

Compute Planck-mean and Rosseland-mean dust opacities as a function of
dust temperature T_d, and save to file.

The approach is as follows: First build λ-dependent opacities κ_abs(λ) and κ_F(λ) 
by integrating over the grain size distribution,
then take the BB-weighted means over λ:

  κ_P(T)   = ∫ κ_abs(λ) B_λ(T) dλ  /  ∫ B_λ(T) dλ

  1/κ_R(T) = ∫ (1/κ_F(λ)) (∂B_λ/∂T) dλ  /  ∫ (∂B_λ/∂T) dλ

where κ_abs(λ) and κ_F(λ) are in cm² per gram of dust of this species.
"""
function ComputeThermalOpacities( Q_abs_itp,
                                  Q_F_itp,
                                  ∂ngr_∂a::Function,
                                      ρgr::Float64,
                                    a_low::Float64,
                                   a_high::Float64,
                                   outdir::String,
                                  species::String )

    # Wavelength grid limits: full Draine grid in cm:
    λ_lo = 1e-3 * μm
    λ_hi = 1e3  * μm

    # Denominator of opacity: total dust mass per H nucleon (independent of λ):
    denom, _ = quadgk( a -> (Four_π / 3) * a^3 * ρgr * ∂ngr_∂a(a),
                       a_low, a_high; rtol = 1e-6 )

    # Build κ_abs(λ) and κ_F(λ) by integrating over grain size distribution.
    # These are opacity per gram of dust at each wavelength λ (cm² g⁻¹),
    # exactly analogous to the old κabs_itp(λ) and κ_F_itp(λ):
    function κ_abs_λ(λ_cm)
        num, _ = quadgk( a -> Q_abs_itp(a, λ_cm) * π * a^2 * ∂ngr_∂a(a),
                         a_low, a_high; rtol = 1e-5 )
        return num / denom
    end

    function κ_F_λ(λ_cm)
        num, _ = quadgk( a -> Q_F_itp(a, λ_cm) * π * a^2 * ∂ngr_∂a(a),
                         a_low, a_high; rtol = 1e-5 )
        return num / denom
    end

    # Planck function B_λ(λ, T) in cgs, with three regimes for stability:
    function B_λ(λ_cm, T)
        x    = h * c / (λ_cm * kB * T)
        pref = (2.0 * h * c^2) / λ_cm^5
        if x > 50.0
            return pref * exp(-x)               # Wien limit
        elseif x < 1e-4
            return (2.0 * c * kB * T) / λ_cm^4 # Rayleigh-Jeans limit
        else
            return pref / expm1(x)
        end
    end

    # ∂B_λ/∂T in cgs, with three regimes for stability:
    function dBdT_λ(λ_cm, T)
        x = h * c / (λ_cm * kB * T)
        if x > 50.0
            B = (2.0 * h * c^2 / λ_cm^5) * exp(-x)
            return B * (x / T)                          # Wien limit
        elseif x < 1e-4
            return (2.0 * c * kB) / λ_cm^4             # Rayleigh-Jeans limit
        else
            exm1 = expm1(x)
            ex   = exm1 + 1.0
            B    = (2.0 * h * c^2 / λ_cm^5) / exm1
            return B * (x / T) * (ex / exm1)           # general case
        end
    end

    # Integrate over log(λ) for accuracy across the wide wavelength range
    # (same technique as the old code):
    x_lo = log(λ_lo)
    x_hi = log(λ_hi)

    # Temperature grid: logarithmic from 5 K to 2000 K:
    T_min  = 5.0
    T_max  = 2000.0
    N_T    = 100
    T_grid = exp.( range( log(T_min), log(T_max), length = N_T ) )

    κ_P = Vector{Float64}(undef, N_T)
    κ_R = Vector{Float64}(undef, N_T)

    for (k, T) in enumerate(T_grid)

        # Integrate over log(λ): substituting λ = exp(x), dλ = exp(x) dx = λ dx:
        norm_P, _ = quadgk( x -> B_λ(exp(x), T)    * exp(x), x_lo, x_hi; rtol = 1e-6 )
        norm_R, _ = quadgk( x -> dBdT_λ(exp(x), T) * exp(x), x_lo, x_hi; rtol = 1e-6 )

        num_P, _  = quadgk( x -> κ_abs_λ(exp(x)) * B_λ(exp(x), T)            * exp(x),
                            x_lo, x_hi; rtol = 1e-4 )
        num_R, _  = quadgk( x -> (1.0 / κ_F_λ(exp(x))) * dBdT_λ(exp(x), T)  * exp(x),
                            x_lo, x_hi; rtol = 1e-4 )

        κ_P[k] = num_P / norm_P
        κ_R[k] = norm_R / num_R   # Harmonic mean: invert at the end

    end

    # Save to file:
    filepath = joinpath(outdir, "kappa_thermal_$(species).txt")
    open(filepath, "w") do f
        println(f, "# Thermal dust opacities for species: $species")
        println(f, "# Computed by ComputeDustOpacities.jl (LYDION)")
        println(f, "# Grain size distribution: Weingartner & Draine (2001), R_V = 5.5, b_C = 3e-5")
        println(f, "# Columns: T_d (K)  κ_P (cm² g⁻¹)  κ_R (cm² g⁻¹)")
        println(f, "#")
        for k in 1:N_T
            @printf(f, "%14.6e  %14.6e  %14.6e\n", T_grid[k], κ_P[k], κ_R[k])
        end
    end

    println("Saved thermal opacities to: $filepath")

    return T_grid, κ_P, κ_R
end


""" RunComputeDustOpacities(path_Gra, path_Sil, path_PAH, outdir; Teff)

Main driver: compute and save opacity tables for all three dust species
(graphite, PAH, silicate) across all radiation bands including Lyα.

Arguments:
  - path_Gra : path to Draine graphite file  (Gra_81)
  - path_Sil : path to Draine silicate file  (suvSil_81)
  - path_PAH : path to Draine PAH file       (PAH_30.txt)
  - outdir   : output directory for .txt files (created if absent)
  - Teff     : blackbody temperature (K), default 3e4 K

Output files:
  - kappa_graphite.txt  : graphite grains, a ∈ [a_split_C, a_max]
  - kappa_PAH.txt       : PAH grains,      a ∈ [a_min_PAH, a_split_C)
  - kappa_silicate.txt  : silicate grains, a ∈ [a_min_Sil, a_max]
"""
function RunComputeDustOpacities( path_Gra::String,
                                  path_Sil::String,
                                  path_PAH::String,
                                    outdir::String,
                                      Teff::Float64 = 4.8e4 )

    println("Building dust interpolations...")
    Q_abs_C,   Q_F_C   = BuildDustInterpolations(path_Gra)
    Q_abs_Sil, Q_F_Sil = BuildDustInterpolations(path_Sil)
    Q_abs_PAH, Q_F_PAH = BuildDustInterpolations(path_PAH; isPAH = true)

    # Composite interpolants for the graphite bin:
    # Use PAH file optical data for a < 0.01 μm (smooth PAH→graphite transition),
    # graphite file data for a ≥ 0.01 μm:
    Q_abs_Gra(a, λ) = a < a_PAH_to_Gra_switch ? Q_abs_PAH(a, λ) : Q_abs_C(a, λ)
    Q_F_Gra(a, λ)   = a < a_PAH_to_Gra_switch ? Q_F_PAH(a, λ)   : Q_F_C(a, λ)

    println("Computing band opacities (this may take a minute)...")

    # Graphite: a ∈ [a_split_C, a_max], composite optical data:
    κ_abs_C,   κ_F_C   = ComputeBandOpacities( Q_abs_Gra, Q_F_Gra,
                                               ∂ngr_∂a_C, ρgr_C, Teff,
                                               a_split_C, a_max )

    # PAH: a ∈ [a_min_PAH, a_split_C), PAH optical data:
    κ_abs_PAH, κ_F_PAH = ComputeBandOpacities( Q_abs_PAH, Q_F_PAH,
                                               ∂ngr_∂a_C, ρgr_C, Teff,
                                               a_min_PAH, a_split_C )

    # Silicate: unchanged:
    κ_abs_Sil, κ_F_Sil = ComputeBandOpacities( Q_abs_Sil, Q_F_Sil,
                                               ∂ngr_∂a_Sil, ρgr_Sil, Teff,
                                               a_min_Sil, a_max )

    # Lyα:
    κ_abs_Lya_C,   κ_F_Lya_C   = ComputeLyαOpacity(  Q_abs_Gra, Q_F_Gra,
                                                     ∂ngr_∂a_C, ρgr_C, Teff,
                                                     a_split_C, a_max )
    κ_abs_Lya_PAH, κ_F_Lya_PAH = ComputeLyαOpacity(  Q_abs_PAH, Q_F_PAH,
                                                     ∂ngr_∂a_C, ρgr_C, Teff,
                                                     a_min_PAH, a_split_C )
    κ_abs_Lya_Sil, κ_F_Lya_Sil = ComputeLyαOpacity(  Q_abs_Sil, Q_F_Sil,
                                                     ∂ngr_∂a_Sil, ρgr_Sil, Teff,
                                                     a_min_Sil, a_max )

    mkpath(outdir)
    SaveOpacities( vcat(κ_abs_C,   κ_abs_Lya_C),
                   vcat(κ_F_C,     κ_F_Lya_C),
                   joinpath(outdir, "kappa_graphite.txt"),
                   "Graphite (a ≥ $(a_split_C/μm) μm, PAH optical data for a < $(a_PAH_to_Gra_switch/μm) μm)", Teff )

    SaveOpacities( vcat(κ_abs_PAH, κ_abs_Lya_PAH),
                   vcat(κ_F_PAH,   κ_F_Lya_PAH),
                   joinpath(outdir, "kappa_PAH.txt"),
                   "PAH (a < $(a_split_C/μm) μm)", Teff )

    SaveOpacities( vcat(κ_abs_Sil, κ_abs_Lya_Sil),
                   vcat(κ_F_Sil,   κ_F_Lya_Sil),
                   joinpath(outdir, "kappa_silicate.txt"),
                   "Silicate", Teff )

    println("Computing thermal opacities (this can take a few minutes)...")
    ComputeThermalOpacities( Q_abs_Gra, Q_F_Gra, ∂ngr_∂a_C,   ρgr_C,   a_split_C, a_max,     outdir, "graphite" )
    ComputeThermalOpacities( Q_abs_PAH, Q_F_PAH, ∂ngr_∂a_C,   ρgr_C,   a_min_PAH, a_split_C, outdir, "PAH" )
    ComputeThermalOpacities( Q_abs_Sil, Q_F_Sil, ∂ngr_∂a_Sil, ρgr_Sil, a_min_Sil, a_max,     outdir, "silicate" )

    # Grain size stats + Q_abs at representative grain sizes:
    ComputeGrainSizeStats( Q_abs_PAH, Q_abs_C, Q_abs_Sil, outdir, Teff )

    println("Done!")
end


# Compute dust opacities & characteristic grain sizes:
Teff = 4.8e4 

RunComputeDustOpacities(
    joinpath(@__DIR__, "Gra_81"),
    joinpath(@__DIR__, "suvSil_81"),
    joinpath(@__DIR__, "PAH_30.txt"),
    joinpath(@__DIR__, "DustOpacityData"),
    Teff
)


end # End of module