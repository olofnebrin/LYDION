
module GravityTest
""" MODULE:  GravityTest

    PURPOSE: Simple gravitational collapse. 

"""

__precompile__()


include("./Hydrodynamics.jl")
include("./SpatialGrid.jl")
include("./PoissonGravitySolver.jl")
include("./ConstantsAndFunctions.jl")
using .Hydrodynamics
using .SpatialGrid
using .PoissonGravitySolver
using .ConstantsAndFunctions

using Printf, PyCall    # PyCall pulls in matplotlib


export SetupUniformCylindricalGrid


function RunTest_Cylindrical( rmax::Float64 = 1 * pc,
                              Zmax::Float64 = 1 * rmax, 
                              T_cl::Float64 = 100.0,
                             T_ext::Float64 = 100.0,
                              tmax::Float64 = 2e6 * yr,
                               N_r::Int64   = 128, 
                               N_Z::Int64   = N_r, 
                          SaveData::Bool    = true )

    Rcl  = 0.75 * pc 
    ρ_cl = 1e5 * mH 

    #Rcl  = 1.0
    #ρ_cl = 1.0
    # Set up cylindrical spatial grid:
    ΔV, inv_ΔV, A_r, A_Z, r, rCoM, 
    Z, ZCoM, inv_r_cell = SetupUniformCylindricalGrid( rmax, Zmax, N_r, N_Z )

    # Initialize potential, its gradient, and all Poisson coefficients:
    Φ, ∂Φ∂r, ∂Φ∂Z, RPoiss_ijp, RPoiss_ij, RPoiss_ijm, 
    ZPoiss_ijp, ZPoiss_ij, ZPoiss_ijm,
    ρJacobi_sq_4 = CreatePoissonCoefficients( A_r, A_Z, inv_ΔV, rCoM, ZCoM, r, Z )

    # Initialize hydro arrays: 
    ρ, ρu_r, ρu_Z, E, ρ_new, ρu_r_new, ρu_Z_new, E_new,
    P, u_r, u_Z, γad,
    ρ_L_r, ρ_R_r, u_r_L_r, u_r_R_r, u_Z_L_r, u_Z_R_r, P_L_r, P_R_r,
    Fρ_r, Fρur_r, Fρuz_r, FE_r,
    ρ_L_Z, ρ_R_Z, u_Z_L_Z, u_Z_R_Z, u_r_L_Z, u_r_R_Z, P_L_Z, P_R_Z,
    Fρ_Z, Fρur_Z, Fρuz_Z, FE_Z,
    ρg_old, u_rg_old, u_Zg_old, Pg_old,
    ρg_new, u_rg_new, u_Zg_new, Pg_new,
    rg, rCoMg, Zg, ZCoMg = GenerateHydroArrays( r, rCoM, Z, ZCoM, 5 / 3 )

    #println( length(rCoMg) - length(rCoM))


    #Zc = Zmax / 2
    Zc = 0.0

    # Set up gas cloud initial conditions: 
    for j in 1:N_Z
        @simd for i in 1:N_r
            @inbounds begin
                
                R = sqrt( r[i+1]^2  +  (Z[j+1] - Zc)^2 )

                if R < Rcl && R > 0.5 * Rcl
                    cosθ   = r[i+1] / sqrt(r[i+1] + Z[j+1]^2) 
                    ρ[i,j] = ρ_cl 
                    P[i,j] = ρ[i,j] * kB * T_cl / mH
                else
                    ρ[i,j] = 0.001 * ρ_cl
                    P[i,j] = ρ[i,j] * kB * T_ext / mH
                end

                E[i,j] = P[i,j] / (γad[i,j] - 1)

            end
        end # End of i-loop 
    end # End of j-loop 


    Δt_fid       = tmax / 300
    t, next_dump = 0.0, Δt_fid
    frames, ts   = Matrix{Float64}[], Float64[]

    while t < tmax

        # Solve for gravitational potential & its gradient:
        PoissonGravitySolver!( Φ, ρ, RPoiss_ijp, RPoiss_ij, RPoiss_ijm,
                               ZPoiss_ijp, ZPoiss_ij, ZPoiss_ijm, ρJacobi_sq_4 )

        Update_∇Φ!( ∂Φ∂r, ∂Φ∂Z, Φ, rCoM, ZCoM )
        
        # Choose time-step: 
        Δt = min( GetCFLTimeStep( r, Z, ρ, P, u_r, u_Z, γad ),
                  GetGravityTimeStep( r, Z, ∂Φ∂r, ∂Φ∂Z ),
                  tmax - t,   Δt_fid )

        #Δt_CFL = GetCFLTimeStep( r, Z, ρ, P, u_r, u_Z, γad )
        #Δt     = min( Δt_CFL,
        #              tmax - t, 
        #              Δt_fid )

        #println(Δt_CFL)

        # Update hydro variables:
        UpdateHydro_StrangSplit!(
                  ρ_new,  ρu_r_new, ρu_Z_new, E_new,
                  ρ,      ρu_r,     ρu_Z,     E,
                  γad,    u_r,      u_Z,
                  inv_ΔV, A_r,      A_Z,      Δt,

                  ρ_L_r, ρ_R_r, u_r_L_r, u_r_R_r, u_Z_L_r, u_Z_R_r, P_L_r, P_R_r,
                  Fρ_r, Fρur_r, Fρuz_r, FE_r,
                  ρ_L_Z, ρ_R_Z, u_Z_L_Z, u_Z_R_Z, u_r_L_Z, u_r_R_Z, P_L_Z, P_R_Z,
                  Fρ_Z, Fρur_Z, Fρuz_Z, FE_Z,
                  P, ρg_old, u_rg_old, u_Zg_old, Pg_old,
                  ρg_new, u_rg_new, u_Zg_new, Pg_new, rg, rCoMg, Zg,  ZCoMg, 
                  Val(:Cylindrical), inv_r_cell )      
                  
        # Gravity kick:
        GravityKick!( ρu_r, ρu_Z, ρ, E, ∂Φ∂r, ∂Φ∂Z, Δt )


        # Update time:
        t += Δt
        println(t/tmax)

        # Decide whether to save data from the time-step (TODO):
        if t ≥ next_dump - 1e-12
            push!( frames, copy(ρ./mH) ) ;  push!(ts, t)
            # @printf("  dumped frame @ t = %.4f   Δt = %.3e\n", t, Δt)
            next_dump += Δt_fid
        end
    end

    # ───────────── 5. build mp4 with imshow (safe) ─────────────
    plt, mpla = pyimport("matplotlib.pyplot"), pyimport("matplotlib.animation")

    tys = ts ./ yr  # times in years

    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(frames[1]', origin="lower",
                   extent=(0,rmax, 0,Zmax),
                   cmap="viridis", vmin=0.0, vmax= 5e5, interpolation="nearest")

    fig.colorbar(im, ax=ax, label="ρ")
    ax.set(aspect="equal", xlabel = "X", ylabel = "Y",
           title=@sprintf("t = %.3f yrs", tys[1]))

    function _update(k)
        f = k + 1 # Julia index
        im.set_data(frames[f]')
        ax.set_title(@sprintf("Time: t = %.3f yrs", tys[f]))
        return (im,)
    end

    # After you've filled `frames` and `ts`:
    duration_s = 15.0
    nframes    = length(frames)
    fps        = nframes / duration_s

    anim = mpla.FuncAnimation(fig, _update, nframes; interval=1000.0/fps)
    anim.save("GravityTest_density.mp4", writer="ffmpeg", dpi=400, fps=fps)


    #anim = mpla.FuncAnimation(fig, _update, length(frames); interval=50)
    #anim.save("Cartesian_sedov_density.mp4", writer="ffmpeg", dpi=400)
    plt.close(fig)
    println("Saved animation → GravityTest_density.mp4")

end


RunTest_Cylindrical()

end # End of module 