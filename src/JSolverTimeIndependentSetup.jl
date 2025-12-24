module JSolverTimeIndependentSetup


include( "./Structs.jl" )
include( "./ConstantsAndFunctions.jl" )
using .Structs
using .ConstantsAndFunctions

# Some basic underlying files:
include( "./WENO3.jl" )
include( "./ExponentialScheme.jl" )
include( "./TridiagonalSolver.jl" )
include( "./M1Closure.jl" ) 
include( "./AdvectJandH.jl" )
include( "./SetupLymanAlphaArrays.jl" )
include( "./SpatialGrid.jl" )
include( "./FreqFactorsAndGrid.jl" )
include( "./UpdateLyaJandH.jl" )
include( "LymanAlphaEmission.jl" )
include( "./PhotoChemistryAndCooling/GasChemistryRates.jl" )


if !haskey(ENV, "MPLBACKEND")
    ENV["MPLBACKEND"] = "TkAgg"
end
using PyCall
const plt  = pyimport("matplotlib.pyplot")
const anim = pyimport("matplotlib.animation")

function JSolver_UniformCloud_New(nHcl::Float64 = 7380.0, T0::Float64 = 5e4,
                                  recoil::Bool = false, c̃::Float64 = 1e2 * kms,
                                  BC_r_left::Symbol  = :Reflect,    # i = 1/2
                                  BC_r_right::Symbol = :Outflow,    # i = N_r+1/2
                                  BC_Z_bot::Symbol   = :Reflect,    # j = 1/2
                                  BC_Z_top::Symbol   = :Outflow;    # j = N_Z+1/2
                                  save_anim::Bool    = true,
                                  out_video::String  = "lya_e_and_spectrum_new.mp4",
                                  save_stride::Int   = 10,
                                  fps::Int           = 20)

    #######################################################################
    #######                  P R O B L E M  S E T U P                #######
    #######################################################################

    N_r   = 2*16; dN_r = 1:N_r
    N_Z   = 2*16; dN_Z = 1:N_Z
    rmax  = pc
    Zmax  = pc
    Rcl   = 0.75 * rmax
    Rs    = 0.5 * Rcl
    L_Lyα = 1e5 * Lsun

    ϵ = 0.0

    println("Threads: ", Threads.nthreads())

    geom = SetupCylindricalGrid(rmax, rmax, Zmax, Zmax, N_r, 0, N_Z, 0)

    ΔV         = geom.ΔV
    inv_ΔV     = geom.inv_ΔV
    A_r        = geom.A_r 
    A_Z        = geom.A_Z 
    r          = geom.r 
    rCoM       = geom.rCoM 
    Z          = geom.Z 
    ZCoM       = geom.ZCoM
    inv_r_cell = geom.inv_r_cell 

    # Gas state (hydro is OFF – just zeros)
    u_r        = zeros(Float64, N_r, N_Z)
    u_Z        = zeros(Float64, N_r, N_Z)
    u_r_star   = zeros(Float64, N_r + 1, N_Z)
    u_Z_star   = zeros(Float64, N_r, N_Z + 1)
    ∂u_r_∂r    = zeros(Float64, N_r, N_Z)
    ∂u_r_∂Z    = zeros(Float64, N_r, N_Z)
    ∂u_Z_∂r    = zeros(Float64, N_r, N_Z)
    ∂u_Z_∂Z    = zeros(Float64, N_r, N_Z)
    u_r_over_r = zeros(Float64, N_r, N_Z)

    # Uniform T
    #T = fill(T0, N_r, N_Z)
    T = zeros( Float64, N_r, N_Z )
    @inbounds for j in dN_Z
        @simd for i in dN_r
            R = hypot(r[i+1], Z[j+1])

            if R > Rs
                T[i,j] = T0
            else 
                T[i,j] = T0
            end 
        end
    end

    #######################################################################
    #######           L y α   A R R A Y S   &   F R E Q S            #######
    #######################################################################

    # Build Lyα state with your new struct-based setup
    LyaState = GenerateLyαArrays(r, Z, T, recoil)

    J          = LyaState.J_Lyα
    H_r        = LyaState.H_r_Lyα
    H_Z        = LyaState.H_Z_Lyα
    H_r_old    = LyaState.H_r_Lyα_old
    H_Z_old    = LyaState.H_Z_Lyα_old
    e_Lyα      = LyaState.e_Lyα
    α_Lyα      = LyaState.α_Lyα
    α_F        = LyaState.α_F_Lyα
    αc         = LyaState.αc_Lyα
    pd         = LyaState.pd
    ∫dν_js_Lyα = LyaState.∫dν_js_Lyα
    Hvoigt     = LyaState.Hvoigt
    ϕ_ν        = LyaState.ϕ_ν
    νC         = LyaState.νC
    Δν         = LyaState.Δν
    Δρu_r      = LyaState.Δρu_Lyα_r
    Δρu_Z      = LyaState.Δρu_Lyα_Z

    N_ν  = length(νC)
    dN_ν = 1:N_ν

    # Set α and js_Lyα to match the OLD test
    @inbounds for j in dN_Z
        @simd for i in dN_r
            R = hypot(r[i+1], Z[j+1])

            # H I opacity at line center:
            #α[i,j] = (R <= Rcl && R >  Rs) ? (nHI * σ0(T0)) : 1e-30

            if R > Rs #&& j > 2
                # Neutral shell 
                xHII = 0.0
                xHI  = 1 - xHII
            else 
                # Ionized gas 
                xHII = 1 - ϵ 
                xHI  = 1 - xHII 
            end 

            #xHII = max( 1.0 / (1 + (R/Rs)^10), 1e-8 )
            #xHI  = 1 - xHII

            if R < Rcl 
                nH = nHcl 
            else
                nH = 1e-13 * nHcl
            end

            #if R > Rcl && j ≤ 0
            #    nH = nHcl
            #end

            nHI        = xHI * nH 
            nHII       = xHII * nH 
            ne         = nHII
            α_Lyα[i,j] = nHI * σ0(T[i,j])

            # Frequency-integrated emissivity (inside Rs only):
            ∫dν_js_Lyα[i,j] = ( ∫dν_js_recLyα( T[i,j], nHII, ne, true, true ) 
                            +   ∫dν_js_collLyα( T[i,j], nHI, ne ) )
            #if R ≤ Rs
            #    js_Lyα[i,j] = 3 * L_Lyα / (Four_π^2 * Rs^3)
            #else
            #    js_Lyα[i,j] = 0.0
            #end
        end
    end

    # No continuum absorption / destruction for this test
    αc .= 0.0
    pd .= 0.0

    # Update frequency factors:
    UpdateFreqFactors!( LyaState, T, ∂u_r_∂r, ∂u_r_∂Z,  ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r )

    # Flux opacity α_F = α * Hvoigt
    @threads for k in dN_ν
        for j in dN_Z 
            for i in dN_r
                α_F[i,j,k] = α_Lyα[i,j] * Hvoigt[i,j,k] 
            end
        end
    end

    #######################################################################
    #######                   S P E C T R U M   S I T E              #######
    #######################################################################

    # Same choice as in the old JSolverTimeIndependentSetup_OLD
    j0 = argmin(abs.(ZCoM .- Rcl))
    i0 = 2

    ν_axis     = collect(νC)
    ν_center   = ν_axis[cld(length(ν_axis), 2)]      # mid-bin
    Δν_spec    = 3 * 6e-3 * ν_center                
    ν_lo       = ν_center - 0.5 * Δν_spec
    ν_hi       = ν_center + 0.5 * Δν_spec

    idx_lo     = searchsortedfirst(ν_axis, ν_lo)
    idx_hi     = searchsortedlast(ν_axis,  ν_hi)
    ν_axis_sub = ν_axis[idx_lo:idx_hi]

    #######################################################################
    #######                  F R A M E   B U F F E R S               #######
    #######################################################################

    e_frames = Matrix{Float64}[]      # e_Lyα snapshots
    J_frames = Vector{Float64}[]      # spectrum snapshots (sub-range)
    t_frames = Float64[]

    #######################################################################
    #######                  T I M E   I N T E G R A T I O N          #######
    #######################################################################

    CFL    = 0.3
    Δt_CFL = GetRHDCFLTimeStep(r, Z, c̃, u_r, u_Z, CFL)
    t      = 0.0
    tmax   = 10 * Rcl / c̃
    step   = 0

    while t < tmax
        step += 1
        Δt     = min(Δt_CFL, tmax - t)
        c̃Δt    = c̃ * Δt

        # 1) Lyα emission
        LyαEmissionUpdate!( J, ∫dν_js_Lyα, ϕ_ν, c̃, Δt )

        # 2) Advection + frequency diffusion + Doppler shift (new solver)
        UpdateLyαJandH!( c̃Δt, c̃, LyaState,
                         inv_ΔV, A_r, A_Z, r, rCoM, Z, ZCoM, inv_r_cell,
                         T,
                         u_r_star, u_Z_star, ∂u_r_∂r, ∂u_r_∂Z, ∂u_Z_∂r, ∂u_Z_∂Z, u_r_over_r,
                         true, BC_r_left, BC_r_right, BC_Z_bot, BC_Z_top )

        # 3) Lyα absorption & destruction
        #LyαAbsorptionUpdate!( J, H_r, H_Z,
        #                      H_r_old, H_Z_old,
        #                      Hvoigt, α_Lyα, αc, α_F, pd,
        #                      Δρu_r, Δρu_Z, Δν,
        #                      c̃, Δt )

        # 4) Update energy density
        Update_e_Lyα!( e_Lyα, Δν, J )

        t += Δt
        println("t/tmax = $(t/tmax)")

        # Record frames
        if step % save_stride == 0
            push!(e_frames, copy(e_Lyα))
            HZ_slice = collect(@view J[i0, j0, :])
            push!(J_frames, HZ_slice[idx_lo:idx_hi])
            push!(t_frames, t)
        end
    end

    #######################################################################
    #######                   B U I L D   A N I M A T I O N           #######
    #######################################################################

    if save_anim && !isempty(e_frames)
        # --- log10(e_Lyα): clamp to [e_floor, 2 * e_max] across all frames ---
        e_max   = maximum(map(maximum, e_frames))
        if e_max <= 0
            e_max = 1.0
        end
        e_floor = 1e-7 * e_max
        vminL   = log10(e_floor)
        vmaxL   = log10(2 * e_max)

        # precompute transformed frames (log10 of clamped e)
        e_frames_log = [ log10.(clamp.(E, e_floor, e_max)) for E in e_frames ]

        # Spectrum limits in window
        jmax = maximum(map(maximum, J_frames))
        jylo, jyhi = 0.0, (jmax > 0 ? 1.05 * jmax : 1.0)

        # Figure with two panels
        pyret = plt[:subplots](
            1, 2;
            figsize     = (12, 5.4),
            gridspec_kw = Dict("width_ratios" => [2.0, 2.0],
                               "wspace"       => 0.24)
        )
        fig   = pyret[1]
        axes  = pyret[2]
        ax_img  = axes[1]
        ax_spec = axes[2]

        # Initial image (log10 e_Lyα)
        im = ax_img.imshow(permutedims(e_frames_log[1]);
                           origin="lower", aspect="equal",
                           cmap="CMRmap", vmin=vminL, vmax=vmaxL,
                           interpolation="nearest")
        cb = fig.colorbar(im, ax=ax_img); cb.set_label("log10 e_Lyα")
        ax_img.set_xlabel("r-index"); ax_img.set_ylabel("Z-index")
        ttl = ax_img.set_title("Lyα energy density  —  t/tmax = 0.0")

        # Initial spectrum
        line_list = ax_spec.plot(ν_axis_sub, J_frames[1], "C0-"; linewidth=1.5)
        line = line_list[1]
        ax_spec.set_xlabel("Frequency ν")
        ax_spec.set_ylabel("J(ν)")
        ax_spec.set_xlim(ν_axis_sub[1], ν_axis_sub[end])
        ax_spec.set_ylim(jylo, jyhi)
        ax_spec.set_title("Spectrum at (i=$(i0), j=$(j0))")

        tmax_anim = tmax
        function _update(k)
            f = k + 1  # Python 0-based → Julia 1-based
            im.set_data(permutedims(e_frames_log[f]))
            line.set_ydata(J_frames[f])
            ttl.set_text("Lyα energy density  —  t/tmax = $(round(t_frames[f]/tmax_anim, digits=3))")
            return (im, line)
        end

        ani = anim.FuncAnimation(fig, _update, length(e_frames);
                                 interval = round(Int, 1000 / fps))

        if endswith(lowercase(out_video), ".gif")
            ani.save(out_video, writer="pillow", dpi=150)
            println("Saved → ", out_video)
        else
            try
                ani.save(out_video, writer="ffmpeg", dpi=180)
                println("Saved → ", out_video)
            catch err
                @warn "ffmpeg not found; falling back to GIF via PillowWriter" err
                gif_name = replace(out_video, r"\.mp4$" => ".gif")
                ani.save(gif_name, writer="pillow", dpi=150)
                println("Saved → ", gif_name)
            end
        end

        plt.close(fig)
    else
        # Fallback: show static final map
        plt.figure()
        plt.imshow(permutedims(log10.(clamp.(e_Lyα,
                                             maximum(e_Lyα) * 1e-3,
                                             maximum(e_Lyα))));
                   origin="lower", aspect="equal", cmap="inferno")
        plt.colorbar()
        plt.title("Final log10 e_Lyα (new solver)")
        plt.show()
    end

    return nothing
end


# Optional: run directly if this file is the main script
JSolver_UniformCloud_New()

end # module
