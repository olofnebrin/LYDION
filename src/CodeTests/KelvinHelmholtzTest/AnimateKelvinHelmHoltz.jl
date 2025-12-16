#!/usr/bin/env julia

using HDF5
ENV["GKSwstype"] = "100"  # disable GUI, render to memory
using Plots
using Printf
using LaTeXStrings

gr()  # GR backend for Plots

# ---------------- User inputs ----------------
script_dir = @__DIR__
filename   = joinpath(script_dir, "KelvinHelmholtz_Nx1024_Ny1024_t5.00.h5")
frameskip  = 1
save_mp4   = true
outname    = joinpath(script_dir, "KelvinHelmholtzDensity.mp4")
fps        = 20
# ---------------------------------------------

#################### READ DATA ####################

t        = nothing
rho_hist = nothing
X        = nothing
Y        = nothing

h5open(filename, "r") do f
    global t, rho_hist, X, Y

    t        = read(f["Times"])               # (Nt,)
    rho_hist = read(f["GasDensity_History"])  # (Nx,Ny,Nt) or (Nt,Nx,Ny)

    # Try Cartesian names first, fall back to cylindrical
    X = haskey(f, "X") ? read(f["X"]) : read(f["r"])
    Y = haskey(f, "Y") ? read(f["Y"]) : read(f["Z"])
end

# --- Ensure correct axis order: (Nt, Nx, Ny) ---
function ensure_NtNxNy(arr, t)
    sz = size(arr)
    length(sz) == 3 || error("Expected 3D array, got size $sz")

    if sz[3] == length(t)        # (Nx, Ny, Nt)
        return permutedims(arr, (3, 1, 2))   # -> (Nt, Nx, Ny)
    elseif sz[1] == length(t)    # already (Nt, Nx, Ny)
        return arr
    else
        error("Unexpected array shape $sz vs length(t)=$(length(t))")
    end
end

rho_hist = ensure_NtNxNy(rho_hist, t)
Nt, Nx, Ny = size(rho_hist)
frames = 1:frameskip:Nt
println("Loaded $Nt snapshots; animating $(length(frames)) frames.")

################ GRID & COLOR LIMITS ##############

# X, Y are edges (Nx+1, Ny+1), make cell centers:
Xc = (X[1:end-1] .+ X[2:end]) ./ 2
Yc = (Y[1:end-1] .+ Y[2:end]) ./ 2

vmin_rho = minimum(rho_hist)
vmax_rho = maximum(rho_hist)
@printf("rho range over all snapshots: [%.3e, %.3e]\n", vmin_rho, vmax_rho)

# Common ticks on both axes, like in Python
common_min   = min(first(X), first(Y))
common_max   = max(last(X),  last(Y))
common_ticks = collect(range(common_min, common_max; length=7))
tick_labels  = [@sprintf("%.2f", val) for val in common_ticks]

################# PLOT STYLE ######################

default(
    framestyle    = :box,
    aspect_ratio  = :equal,
    linewidth     = 3.5,
    tickdirection = :in,
    xticks        = (common_ticks, tick_labels),
    yticks        = (common_ticks, tick_labels),
    guidefont     = font(22),   # bigger for crispness
    tickfont      = font(18),
    legendfont    = font(18),
    size          = (1600, 1400),  # more pixels for text & details
)

################# INITIAL FIGURE ##################

# First frame
k0 = frames[1]
ρ0 = @view rho_hist[k0, :, :]

p = heatmap(
    Xc, Yc, ρ0',
    c      = :haline,
    clims  = (vmin_rho, vmax_rho),
    xlabel = "X",
    ylabel = "Y",
    xlims  = (first(X), last(X)),
    ylims  = (first(Y), last(Y)),
    colorbar_title = L"\rho",
)

# Precompute annotation position
xlab = Xc[1] + 0.05 * (Xc[end] - Xc[1])
ylab = Yc[end] - 0.05 * (Yc[end] - Yc[1])

################# ANIMATION #######################

anim = @animate for k in frames
    ρ = @view rho_hist[k, :, :]

    # Overplot new heatmap (no new colorbar each time)
    heatmap!(
        p, Xc, Yc, ρ';
        clims     = (vmin_rho, vmax_rho),
        colorbar  = false,  # keep existing bar from initial plot
    )

    # Time label – draw over the top
    tstr = @sprintf("t = %.3f", t[k])
    annotate!(p, xlab, ylab, text(tstr, :white, 26, :left, :top))
end

################## OUTPUT #########################

if save_mp4
    # dpi↑ for crisper labels in the video
    mp4(anim, outname; fps = fps, dpi = 300)
    println("Saved animation to $outname")
else
    gif(anim, joinpath(script_dir, "KelvinHelmholtzDensity.gif"); fps = fps, dpi = 300)
end
