# FILE: LYDION_Lya_GPU_ToyBenchmark.jl
#
# PURPOSE:
#   Toy LYDION-like Lyα RT benchmark comparing
#       CPU: Float64 + Threads
#       GPU: Float32 + Metal.jl + KernelAbstractions.jl
#
#   The toy problem is meant to mimic the expensive structure of the real code:
#     1. Explicit M1-like spatial advection of J, H_r, H_Z on arrays J[i,j,k]
#        with i=r, j=Z, k=frequency. CPU loops thread over k, as in AdvectLyα!.
#     2. Implicit backward-Euler frequency diffusion solve: one tridiagonal
#        Thomas solve over k for every spatial cell (i,j), as in FrequencyUpdateJLyα_AllCells!.
#
# HOW TO RUN ON YOUR MAC:
#   1. In Julia REPL/pkg mode:
#        ] add BenchmarkTools KernelAbstractions Metal
#   2. Start Julia with threads, e.g. from terminal:
#        JULIA_NUM_THREADS=8 julia LYDION_Lya_GPU_ToyBenchmark.jl
#
# NOTES:
#   - GPU benchmark uses Float32, CPU benchmark uses Float64.
#   - The "GPU no transfers" timing keeps arrays resident on the GPU.
#   - The "GPU with transfers" timing includes CPU->GPU upload and GPU->CPU download
#     every step, which is a pessimistic estimate if only part of LYDION is on GPU.
#   - This is not a correctness benchmark for your full RT scheme. It is a speed/profiling
#     proxy with similar memory layout and loop/kernels.

using Random
using Statistics
using Printf
using Base.Threads
using BenchmarkTools
using KernelAbstractions
using Metal

Metal.allowscalar(false)

const Jeps64 = 1.0e-30
const Jeps32 = Float32(1.0e-20)

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

@inline clamp_index(i::Int, N::Int) = ifelse(i < 1, 1, ifelse(i > N, N, i))

@inline function m1_chi(fmag::T) where {T}
    # Levermore/M1-like closure factor. Clamped for numerical safety.
    f = min(max(fmag, zero(T)), one(T))
    return (T(3) + T(4)*f*f) / (T(5) + T(2)*sqrt(max(T(4) - T(3)*f*f, T(1.0e-12))))
end

@inline function m1_D_components(J::T, Hr::T, HZ::T, eps::T) where {T}
    invJ = inv(max(J, eps))
    fr   = Hr * invJ
    fz   = HZ * invJ
    f2   = min(fr*fr + fz*fz, one(T))
    f    = sqrt(f2)
    χ    = m1_chi(f)

    # D = (1-χ)/2 I + (3χ-1)/2 n n in 2D r-Z plane.
    a    = (one(T) - χ) / T(2)
    b    = (T(3)*χ - one(T)) / T(2)
    invf = inv(max(f, T(1.0e-12)))
    nr   = fr * invf
    nz   = fz * invf

    # In the isotropic limit f≈0, b≈0, so nr/nz choice is harmless.
    Drr  = a + b*nr*nr
    Dzz  = a + b*nz*nz
    Drz  = b*nr*nz
    return Drr, Dzz, Drz
end

@inline function glf_flux_J(JL::T, JR::T, HL::T, HR::T, τcell::T) where {T}
    # Similar spirit to your GetFluxGLF: thin -> GLF; thick -> diffusion-like.
    ϵ      = (T(0.2) + τcell) / (T(0.2) + τcell + τcell*τcell)
    Fglf   = T(0.5)*(HL + HR) - T(0.5)*(JR - JL)
    Fdiff  = - (JR - JL) / (T(3)*max(τcell, T(1.0e-12)))
    return ϵ*Fglf + (one(T)-ϵ)*Fdiff
end

@inline function glf_flux_H(FL::T, FR::T, HL::T, HR::T, τcell::T) where {T}
    ϵ = (T(0.2) + τcell) / (T(0.2) + τcell + τcell*τcell)
    return T(0.5)*(FL + FR) - T(0.5)*ϵ*(HR - HL)
end

# -----------------------------------------------------------------------------
# Initial data
# -----------------------------------------------------------------------------

function make_initial_state(::Type{T}, Nr::Int, Nz::Int, Nν::Int; seed::Int=1234) where {T}
    rng = MersenneTwister(seed)

    J  = Array{T}(undef, Nr, Nz, Nν)
    Hr = Array{T}(undef, Nr, Nz, Nν)
    HZ = Array{T}(undef, Nr, Nz, Nν)

    r0  = T(0.48*Nr)
    z0  = T(0.53*Nz)
    σr  = T(0.18*Nr)
    σz  = T(0.18*Nz)
    k0  = T(0.50*(Nν+1))
    σk  = T(0.14*Nν)

    @inbounds for k in 1:Nν, j in 1:Nz, i in 1:Nr
        spatial = exp(-((T(i)-r0)^2/(T(2)*σr^2) + (T(j)-z0)^2/(T(2)*σz^2)))
        freq    = exp(-((T(k)-k0)^2/(T(2)*σk^2)))
        noise   = T(1) + T(0.02)*(T(rand(rng)) - T(0.5))
        Jij     = T(1.0e-6) + spatial*freq*noise
        J[i,j,k]  = Jij
        Hr[i,j,k] = T(0.15) * Jij * tanh((T(i)-r0)/max(σr,T(1)))
        HZ[i,j,k] = T(0.10) * Jij * tanh((T(j)-z0)/max(σz,T(1)))
    end

    return J, Hr, HZ
end

function make_frequency_coeffs(::Type{T}, Nr::Int, Nz::Int, Nν::Int) where {T}
    Fkp = Array{T}(undef, Nr, Nz, Nν)
    Fk  = Array{T}(undef, Nr, Nz, Nν)
    Fkm = Array{T}(undef, Nr, Nz, Nν)

    k0 = T(0.5*(Nν+1))
    @inbounds for k in 1:Nν, j in 1:Nz, i in 1:Nr
        # Smooth, line-center-weighted coefficients. Signs match the convention
        # in your FrequencyUpdateJLyα!: A=Fkm, B=-(Fk+Δw), C=Fkp.
        wing  = abs(T(k)-k0) / max(k0, T(1))
        diff  = T(0.05) / (T(1) + T(30)*wing^2)
        drift = T(0.005) * tanh((T(k)-k0)/max(k0,T(1)))

        Fkp[i,j,k] = diff + max(drift, zero(T))
        Fkm[i,j,k] = diff + max(-drift, zero(T))
        Fk[i,j,k]  = Fkp[i,j,k] + Fkm[i,j,k]
    end
    return Fkp, Fk, Fkm
end

# -----------------------------------------------------------------------------
# CPU kernels: Float64 + Threads, with real LYDION-like layout J[i,j,k]
# -----------------------------------------------------------------------------

function advect_cpu!(Jnew, Hrnew, HZnew, J, Hr, HZ, αF, cdt, invdr, invdz)
    Nr, Nz, Nν = size(J)

    @threads :static for k in 1:Nν
        @inbounds for j in 1:Nz, i in 1:Nr
            im = clamp_index(i-1, Nr); ip = clamp_index(i+1, Nr)
            jm = clamp_index(j-1, Nz); jp = clamp_index(j+1, Nz)

            Jij  = J[i,j,k]
            Hrij = Hr[i,j,k]
            Hzij = HZ[i,j,k]

            # Neighbor values.
            JL = J[im,j,k]; JR = J[ip,j,k]
            JB = J[i,jm,k]; JT = J[i,jp,k]
            HrL = Hr[im,j,k]; HrR = Hr[ip,j,k]
            HrB = Hr[i,jm,k]; HrT = Hr[i,jp,k]
            HZL = HZ[im,j,k]; HZR = HZ[ip,j,k]
            HZB = HZ[i,jm,k]; HZT = HZ[i,jp,k]

            # M1 tensor at center and neighbors, used as H-equation fluxes.
            Drr, Dzz, Drz = m1_D_components(Jij, Hrij, Hzij, Jeps64)
            DrrL, _, DrzL = m1_D_components(JL, HrL, HZL, Jeps64)
            DrrR, _, DrzR = m1_D_components(JR, HrR, HZR, Jeps64)
            _, DzzB, DrzB = m1_D_components(JB, HrB, HZB, Jeps64)
            _, DzzT, DrzT = m1_D_components(JT, HrT, HZT, Jeps64)

            τrL = αF[im,j,k] + αF[i,j,k]
            τrR = αF[i,j,k]  + αF[ip,j,k]
            τzB = αF[i,jm,k] + αF[i,j,k]
            τzT = αF[i,j,k]  + αF[i,jp,k]

            # J fluxes at left/right/bottom/top interfaces.
            FJL = glf_flux_J(JL, Jij, HrL, Hrij, τrL)
            FJR = glf_flux_J(Jij, JR, Hrij, HrR, τrR)
            FJB = glf_flux_J(JB, Jij, HZB, Hzij, τzB)
            FJT = glf_flux_J(Jij, JT, Hzij, HZT, τzT)

            # H_r fluxes: r-flux ~ D_rr J, z-flux ~ D_rZ J.
            FHrL = glf_flux_H(DrrL*JL, Drr*Jij, HrL, Hrij, τrL)
            FHrR = glf_flux_H(Drr*Jij, DrrR*JR, Hrij, HrR, τrR)
            FHrB = glf_flux_H(DrzB*JB, Drz*Jij, HrB, Hrij, τzB)
            FHrT = glf_flux_H(Drz*Jij, DrzT*JT, Hrij, HrT, τzT)

            # H_Z fluxes: r-flux ~ D_rZ J, z-flux ~ D_ZZ J.
            FHZL = glf_flux_H(DrzL*JL, Drz*Jij, HZL, Hzij, τrL)
            FHZR = glf_flux_H(Drz*Jij, DrzR*JR, Hzij, HZR, τrR)
            FHZB = glf_flux_H(DzzB*JB, Dzz*Jij, HZB, Hzij, τzB)
            FHZT = glf_flux_H(Dzz*Jij, DzzT*JT, Hzij, HZT, τzT)

            Jn  = Jij  - cdt*((FJR - FJL)*invdr + (FJT - FJB)*invdz)
            Hrn = Hrij - cdt*((FHrR - FHrL)*invdr + (FHrT - FHrB)*invdz)
            Hzn = Hzij - cdt*((FHZR - FHZL)*invdr + (FHZT - FHZB)*invdz)

            # Simple causality/floor, same spirit as your post-frequency correction.
            Jn = max(Jn, Jeps64)
            H2 = Hrn*Hrn + Hzn*Hzn
            if H2 > Jn*Jn
                s = Jn / sqrt(H2)
                Hrn *= s
                Hzn *= s
            end

            Jnew[i,j,k]  = Jn
            Hrnew[i,j,k] = Hrn
            HZnew[i,j,k] = Hzn
        end
    end
    return nothing
end

function frequency_cpu!(Jnew, Jold, Fkp, Fk, Fkm, Δw)
    Nr, Nz, Nν = size(Jold)
    nint = Nν - 2

    @threads :static for idx in 1:(Nr*Nz)
        i = (idx - 1) % Nr + 1
        j = (idx - 1) ÷ Nr + 1

        # Scratch arrays per task. This intentionally allocates, but only inside
        # the benchmarked function. For a closer optimized CPU comparison, see
        # frequency_cpu_scratch! below.
        A  = Vector{Float64}(undef, nint)
        B  = Vector{Float64}(undef, nint)
        C  = Vector{Float64}(undef, nint)
        D  = Vector{Float64}(undef, nint)
        Cp = Vector{Float64}(undef, nint)
        Dp = Vector{Float64}(undef, nint)
        x  = Vector{Float64}(undef, nint)

        @inbounds begin
            for kk in 1:nint
                k = kk + 1
                A[kk] = Fkm[i,j,k]
                B[kk] = -(Fk[i,j,k] + Δw)
                C[kk] = Fkp[i,j,k]
                D[kk] = -Δw * Jold[i,j,k]
            end

            Cp[1] = C[1] / B[1]
            Dp[1] = D[1] / B[1]
            for kk in 2:nint
                denom  = B[kk] - A[kk]*Cp[kk-1]
                Cp[kk] = C[kk] / denom
                Dp[kk] = (D[kk] - A[kk]*Dp[kk-1]) / denom
            end
            x[nint] = Dp[nint]
            for kk in (nint-1):-1:1
                x[kk] = Dp[kk] - Cp[kk]*x[kk+1]
            end

            Jnew[i,j,1]  = Jold[i,j,1]
            Jnew[i,j,Nν] = Jold[i,j,Nν]
            for kk in 1:nint
                Jnew[i,j,kk+1] = max(x[kk], Jeps64)
            end
        end
    end
    return nothing
end

function make_cpu_scratch(Nν::Int)
    nint = Nν - 2
    return [(Vector{Float64}(undef,nint), Vector{Float64}(undef,nint), Vector{Float64}(undef,nint),
             Vector{Float64}(undef,nint), Vector{Float64}(undef,nint), Vector{Float64}(undef,nint),
             Vector{Float64}(undef,nint)) for _ in 1:Threads.maxthreadid()]
end

function frequency_cpu_scratch!(Jnew, Jold, Fkp, Fk, Fkm, Δw, scratch)
    Nr, Nz, Nν = size(Jold)
    nint = Nν - 2

    @threads :static for idx in 1:(Nr*Nz)
        tid = Threads.threadid()
        A, B, C, D, Cp, Dp, x = scratch[tid]
        i = (idx - 1) % Nr + 1
        j = (idx - 1) ÷ Nr + 1

        @inbounds begin
            for kk in 1:nint
                k = kk + 1
                A[kk] = Fkm[i,j,k]
                B[kk] = -(Fk[i,j,k] + Δw)
                C[kk] = Fkp[i,j,k]
                D[kk] = -Δw * Jold[i,j,k]
            end

            Cp[1] = C[1] / B[1]
            Dp[1] = D[1] / B[1]
            for kk in 2:nint
                denom  = B[kk] - A[kk]*Cp[kk-1]
                Cp[kk] = C[kk] / denom
                Dp[kk] = (D[kk] - A[kk]*Dp[kk-1]) / denom
            end
            x[nint] = Dp[nint]
            for kk in (nint-1):-1:1
                x[kk] = Dp[kk] - Cp[kk]*x[kk+1]
            end

            Jnew[i,j,1]  = Jold[i,j,1]
            Jnew[i,j,Nν] = Jold[i,j,Nν]
            for kk in 1:nint
                Jnew[i,j,kk+1] = max(x[kk], Jeps64)
            end
        end
    end
    return nothing
end

# -----------------------------------------------------------------------------
# GPU kernels: Float32 + Metal + KernelAbstractions
# -----------------------------------------------------------------------------

@kernel function advect_gpu_kernel!(Jnew, Hrnew, HZnew, J, Hr, HZ, αF, cdt, invdr, invdz, Nr::Int, Nz::Int, Nν::Int)
    I = @index(Global, Cartesian)
    i = I[1]
    j = I[2]
    k = I[3]

    if i <= Nr && j <= Nz && k <= Nν
        im = ifelse(i == 1, 1, i-1)
        ip = ifelse(i == Nr, Nr, i+1)
        jm = ifelse(j == 1, 1, j-1)
        jp = ifelse(j == Nz, Nz, j+1)

        Jij  = J[i,j,k]
        Hrij = Hr[i,j,k]
        Hzij = HZ[i,j,k]

        JL = J[im,j,k]; JR = J[ip,j,k]
        JB = J[i,jm,k]; JT = J[i,jp,k]
        HrL = Hr[im,j,k]; HrR = Hr[ip,j,k]
        HrB = Hr[i,jm,k]; HrT = Hr[i,jp,k]
        HZL = HZ[im,j,k]; HZR = HZ[ip,j,k]
        HZB = HZ[i,jm,k]; HZT = HZ[i,jp,k]

        Drr, Dzz, Drz = m1_D_components(Jij, Hrij, Hzij, Jeps32)
        DrrL, _, DrzL = m1_D_components(JL, HrL, HZL, Jeps32)
        DrrR, _, DrzR = m1_D_components(JR, HrR, HZR, Jeps32)
        _, DzzB, DrzB = m1_D_components(JB, HrB, HZB, Jeps32)
        _, DzzT, DrzT = m1_D_components(JT, HrT, HZT, Jeps32)

        τrL = αF[im,j,k] + αF[i,j,k]
        τrR = αF[i,j,k]  + αF[ip,j,k]
        τzB = αF[i,jm,k] + αF[i,j,k]
        τzT = αF[i,j,k]  + αF[i,jp,k]

        FJL = glf_flux_J(JL, Jij, HrL, Hrij, τrL)
        FJR = glf_flux_J(Jij, JR, Hrij, HrR, τrR)
        FJB = glf_flux_J(JB, Jij, HZB, Hzij, τzB)
        FJT = glf_flux_J(Jij, JT, Hzij, HZT, τzT)

        FHrL = glf_flux_H(DrrL*JL, Drr*Jij, HrL, Hrij, τrL)
        FHrR = glf_flux_H(Drr*Jij, DrrR*JR, Hrij, HrR, τrR)
        FHrB = glf_flux_H(DrzB*JB, Drz*Jij, HrB, Hrij, τzB)
        FHrT = glf_flux_H(Drz*Jij, DrzT*JT, Hrij, HrT, τzT)

        FHZL = glf_flux_H(DrzL*JL, Drz*Jij, HZL, Hzij, τrL)
        FHZR = glf_flux_H(Drz*Jij, DrzR*JR, Hzij, HZR, τrR)
        FHZB = glf_flux_H(DzzB*JB, Dzz*Jij, HZB, Hzij, τzB)
        FHZT = glf_flux_H(Dzz*Jij, DzzT*JT, Hzij, HZT, τzT)

        Jn  = Jij  - cdt*((FJR - FJL)*invdr + (FJT - FJB)*invdz)
        Hrn = Hrij - cdt*((FHrR - FHrL)*invdr + (FHrT - FHrB)*invdz)
        Hzn = Hzij - cdt*((FHZR - FHZL)*invdr + (FHZT - FHZB)*invdz)

        Jn = max(Jn, Jeps32)
        H2 = Hrn*Hrn + Hzn*Hzn
        if H2 > Jn*Jn
            s = Jn / sqrt(H2)
            Hrn *= s
            Hzn *= s
        end

        Jnew[i,j,k]  = Jn
        Hrnew[i,j,k] = Hrn
        HZnew[i,j,k] = Hzn
    end
end

const MAX_Nν_GPU = 512

@kernel function frequency_gpu_kernel_scratch!(Jnew, Jold, Cp, Dp, Fkp, Fk, Fkm, Δw, Nr::Int, Nz::Int, Nν::Int)
    I = @index(Global, Cartesian)
    i = I[1]
    j = I[2]

    if i <= Nr && j <= Nz
        # Thomas solve done serially in k inside one GPU work item.
        # Cp and Dp are global scratch arrays with the same layout J[i,j,k].
        # This is deliberately close to your real per-cell tridiagonal solve.
        nint = Nν - 2

        k = 2
        A = Fkm[i,j,k]
        B = -(Fk[i,j,k] + Δw)
        C = Fkp[i,j,k]
        D = -Δw * Jold[i,j,k]
        Cp[i,j,1] = C / B
        Dp[i,j,1] = D / B

        for kk in 2:nint
            k = kk + 1
            A = Fkm[i,j,k]
            B = -(Fk[i,j,k] + Δw)
            C = Fkp[i,j,k]
            D = -Δw * Jold[i,j,k]
            denom     = B - A*Cp[i,j,kk-1]
            Cp[i,j,kk] = C / denom
            Dp[i,j,kk] = (D - A*Dp[i,j,kk-1]) / denom
        end

        Jnew[i,j,1]  = Jold[i,j,1]
        Jnew[i,j,Nν] = Jold[i,j,Nν]
        Jnew[i,j,Nν-1] = max(Dp[i,j,nint], Jeps32)

        # Back-substitution. x is stored directly in Jnew.
        for kk in (MAX_Nν_GPU-1):-1:1
            if kk <= nint - 1
                Jnew[i,j,kk+1] = max(Dp[i,j,kk] - Cp[i,j,kk]*Jnew[i,j,kk+2], Jeps32)
            end
        end
    end
end

function launch_advect_gpu!(backend, Jnew, Hrnew, HZnew, J, Hr, HZ, αF, cdt, invdr, invdz)
    Nr, Nz, Nν = size(J)
    kernel! = advect_gpu_kernel!(backend, 256)
    kernel!(Jnew, Hrnew, HZnew, J, Hr, HZ, αF, Float32(cdt), Float32(invdr), Float32(invdz), Nr, Nz, Nν;
            ndrange=(Nr,Nz,Nν))
    KernelAbstractions.synchronize(backend)
    return nothing
end

function launch_frequency_gpu!(backend, Jnew, Jold, Cp, Dp, Fkp, Fk, Fkm, Δw)
    Nr, Nz, Nν = size(Jold)
    if Nν > MAX_Nν_GPU
        error("Nν=$Nν exceeds MAX_Nν_GPU=$MAX_Nν_GPU. Increase MAX_Nν_GPU in the script.")
    end
    kernel! = frequency_gpu_kernel_scratch!(backend, 128)
    kernel!(Jnew, Jold, Cp, Dp, Fkp, Fk, Fkm, Float32(Δw), Nr, Nz, Nν; ndrange=(Nr,Nz))
    KernelAbstractions.synchronize(backend)
    return nothing
end

# -----------------------------------------------------------------------------
# Benchmark driver
# -----------------------------------------------------------------------------

function median_seconds(t)
    return BenchmarkTools.median(t).time * 1e-9
end

function run_benchmark(; Nr::Int=256, Nz::Int=256, Nν::Int=128, nsteps::Int=10)
    @info "Benchmark setup" Nr Nz Nν nsteps threads=Threads.nthreads()

    if Nν > MAX_Nν_GPU
        error("Nν=$Nν exceeds MAX_Nν_GPU=$MAX_Nν_GPU. Increase MAX_Nν_GPU in the script.")
    end

    cdt64  = 0.15
    invdr64 = 1.0
    invdz64 = 1.0
    Δw64  = 5.0

    J64, Hr64, HZ64 = make_initial_state(Float64, Nr, Nz, Nν)
    αF64 = fill(2.0, Nr, Nz, Nν)
    Fkp64, Fk64, Fkm64 = make_frequency_coeffs(Float64, Nr, Nz, Nν)

    Jnew64  = similar(J64)
    Hrnew64 = similar(Hr64)
    HZnew64 = similar(HZ64)
    scratch = make_cpu_scratch(Nν)

    # CPU warmup.
    advect_cpu!(Jnew64, Hrnew64, HZnew64, J64, Hr64, HZ64, αF64, cdt64, invdr64, invdz64)
    frequency_cpu_scratch!(Jnew64, J64, Fkp64, Fk64, Fkm64, Δw64, scratch)

    cpu_adv = @benchmark advect_cpu!($Jnew64, $Hrnew64, $HZnew64, $J64, $Hr64, $HZ64, $αF64, $cdt64, $invdr64, $invdz64) samples=10 evals=1
    cpu_fre = @benchmark frequency_cpu_scratch!($Jnew64, $J64, $Fkp64, $Fk64, $Fkm64, $Δw64, $scratch) samples=10 evals=1

    function one_cpu_step!()
        advect_cpu!(Jnew64, Hrnew64, HZnew64, J64, Hr64, HZ64, αF64, cdt64, invdr64, invdz64)
        frequency_cpu_scratch!(J64, Jnew64, Fkp64, Fk64, Fkm64, Δw64, scratch)
        return nothing
    end
    cpu_step = @benchmark $one_cpu_step!() samples=10 evals=1

    # GPU setup.
    backend = MetalBackend()

    J32  = Float32.(J64)
    Hr32 = Float32.(Hr64)
    HZ32 = Float32.(HZ64)
    αF32 = Float32.(αF64)
    Fkp32 = Float32.(Fkp64)
    Fk32  = Float32.(Fk64)
    Fkm32 = Float32.(Fkm64)

    Jg  = MtlArray(J32);   Jnewg  = similar(Jg)
    Hrg = MtlArray(Hr32);  Hrnewg = similar(Hrg)
    HZg = MtlArray(HZ32);  HZnewg = similar(HZg)
    αFg = MtlArray(αF32)
    Fkpg = MtlArray(Fkp32); Fkg = MtlArray(Fk32); Fkmg = MtlArray(Fkm32)
    Cpg = similar(Jg); Dpg = similar(Jg)

    # GPU warmup/compile.
    launch_advect_gpu!(backend, Jnewg, Hrnewg, HZnewg, Jg, Hrg, HZg, αFg, Float32(cdt64), Float32(invdr64), Float32(invdz64))
    launch_frequency_gpu!(backend, Jnewg, Jg, Cpg, Dpg, Fkpg, Fkg, Fkmg, Float32(Δw64))

    gpu_adv = @benchmark launch_advect_gpu!($backend, $Jnewg, $Hrnewg, $HZnewg, $Jg, $Hrg, $HZg, $αFg, $(Float32(cdt64)), $(Float32(invdr64)), $(Float32(invdz64))) samples=20 evals=1
    gpu_fre = @benchmark launch_frequency_gpu!($backend, $Jnewg, $Jg, $Cpg, $Dpg, $Fkpg, $Fkg, $Fkmg, $(Float32(Δw64))) samples=20 evals=1

    function one_gpu_step_no_transfer!()
        launch_advect_gpu!(backend, Jnewg, Hrnewg, HZnewg, Jg, Hrg, HZg, αFg, Float32(cdt64), Float32(invdr64), Float32(invdz64))
        launch_frequency_gpu!(backend, Jg, Jnewg, Cpg, Dpg, Fkpg, Fkg, Fkmg, Float32(Δw64))
        return nothing
    end
    gpu_step_no_transfer = @benchmark $one_gpu_step_no_transfer!() samples=20 evals=1

    function one_gpu_step_with_transfer!()
        Jtmp  = MtlArray(J32);   Jnewtmp  = similar(Jtmp)
        Hrtmp = MtlArray(Hr32);  Hrnewtmp = similar(Hrtmp)
        HZtmp = MtlArray(HZ32);  HZnewtmp = similar(HZtmp)
        launch_advect_gpu!(backend, Jnewtmp, Hrnewtmp, HZnewtmp, Jtmp, Hrtmp, HZtmp, αFg, Float32(cdt64), Float32(invdr64), Float32(invdz64))
        launch_frequency_gpu!(backend, Jtmp, Jnewtmp, Cpg, Dpg, Fkpg, Fkg, Fkmg, Float32(Δw64))
        Array(Jtmp) # force GPU->CPU transfer of main output
        return nothing
    end
    gpu_step_with_transfer = @benchmark $one_gpu_step_with_transfer!() samples=10 evals=1

    # Report.
    t_cpu_adv  = median_seconds(cpu_adv)
    t_cpu_fre  = median_seconds(cpu_fre)
    t_cpu_step = median_seconds(cpu_step)
    t_gpu_adv  = median_seconds(gpu_adv)
    t_gpu_fre  = median_seconds(gpu_fre)
    t_gpu_step = median_seconds(gpu_step_no_transfer)
    t_gpu_tr   = median_seconds(gpu_step_with_transfer)

    println("\n--- LYDION-like Lyα RT toy benchmark ---")
    @printf("Grid: Nr = %d, Nz = %d, Nν = %d\n", Nr, Nz, Nν)
    @printf("CPU threads: %d\n", Threads.nthreads())
    println("CPU: Float64 + Threads")
    println("GPU: Float32 + Metal.jl / KernelAbstractions.jl")

    println("\n--- Median wall times ---")
    @printf("CPU M1 advection:                         %.6e s\n", t_cpu_adv)
    @printf("GPU M1 advection, no transfers:           %.6e s\n", t_gpu_adv)
    @printf("Speed-up, M1 advection:                   %.3f x\n", t_cpu_adv/t_gpu_adv)

    @printf("\nCPU frequency Thomas solve:                %.6e s\n", t_cpu_fre)
    @printf("GPU frequency Thomas solve, no transfers:  %.6e s\n", t_gpu_fre)
    @printf("Speed-up, frequency solve:                %.3f x\n", t_cpu_fre/t_gpu_fre)

    @printf("\nCPU one toy RT step:                       %.6e s\n", t_cpu_step)
    @printf("GPU one toy RT step, no transfers:         %.6e s\n", t_gpu_step)
    @printf("GPU one toy RT step, with transfers:       %.6e s\n", t_gpu_tr)
    @printf("Speed-up, full step no transfers:          %.3f x\n", t_cpu_step/t_gpu_step)
    @printf("Speed-up, full step with transfers:        %.3f x\n", t_cpu_step/t_gpu_tr)

    println("\n--- Interpretation ---")
    println("Use the no-transfer number if J/H and the RT coefficients stay resident on the GPU across many RT substeps.")
    println("Use the with-transfer number if every RT step has to upload/download the large 3D arrays.")
    println("If the with-transfer speed-up is poor but no-transfer is good, the next step is porting enough of the surrounding RT/thermochemistry pipeline to avoid host-device traffic.")

    return (; t_cpu_adv, t_gpu_adv, t_cpu_fre, t_gpu_fre, t_cpu_step, t_gpu_step, t_gpu_tr)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Change these to match your typical production Lyα grid.
    run_benchmark(Nr=256, Nz=256, Nν=128, nsteps=10)
end

run_benchmark(Nr=512, Nz=512, Nν=128, nsteps=10)
