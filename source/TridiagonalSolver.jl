
module TridiagonalSolver
""" - **MODULE:**  TridiagonalSolver

    - **PURPOSE:** Contains triagonal solvers, e.g. the Thomas algorithm,
                   capable of solving equations of the form:
                   A * x_{i-1}  +  B * x_i  +  C * x_{i+1}  =  D

    """

__precompile__()

using Base.Threads            # For CPU acceleration

export ThomasAlgorithm, CyclicReduction


""" Thomas algorithm to solve tridiagonal equations of the form:

    A * x_{i-1}  +  B * x_i  +  C * x_{i+1}  =  D
    
    - **ARGUMENTS** :   A, B, C, D 
    - **RETURNS**   :   x 
    """
function ThomasAlgorithm( A::AbstractVector{Float64}, B::AbstractVector{Float64}, 
                          C::AbstractVector{Float64}, D::AbstractVector{Float64} )
    n       = length(A)
    C_prime = similar(A)    # Modified upper diagonal
    D_prime = similar(A)    # Modified right-hand side
    
    # Forward elimination
    C_prime[1] = C[1] / B[1]
    D_prime[1] = D[1] / B[1]
    @inbounds @simd for i = 2:n-1
        denom      = B[i] - A[i] * C_prime[i-1]
        C_prime[i] = C[i] / denom
        D_prime[i] = (D[i] - A[i] * D_prime[i-1]) / denom
    end
    D_prime[n] = (D[n] - A[n-1] * D_prime[n-1]) / (B[n] - A[n-1] * C_prime[n-1])
    
    # Back substitution
    x = similar(A)
    x[n] = D_prime[n]
    @inbounds @simd for i = n-1:-1:1
        x[i] = D_prime[i] - C_prime[i] * x[i+1]
    end
    return x
end


using LinearAlgebra

function CyclicReduction(a::Vector{Float64}, b::Vector{Float64}, c::Vector{Float64}, d::Vector{Float64})
    n = length(b)
    @assert ispow2(n) "Cyclic reduction here assumes n is a power of two."

    # Store all levels
    levels = Int(log2(n))

    # We will store the systems at each level
    as = Vector{Vector{Float64}}(undef, levels+1)
    bs = Vector{Vector{Float64}}(undef, levels+1)
    cs = Vector{Vector{Float64}}(undef, levels+1)
    ds = Vector{Vector{Float64}}(undef, levels+1)

    as[1] = copy(a)
    bs[1] = copy(b)
    cs[1] = copy(c)
    ds[1] = copy(d)

    # Forward reduction
    # At level 1, we eliminate even indices from the original system and keep odd indices.
    # At each subsequent level, we again eliminate half of the equations from the current system.
    # Index pattern:
    # - At the first level, from [1,2,3,4,...,n], we keep [1,3,5,...]
    # - At the second level, from those kept, we keep every second one, etc.

    for lvl in 1:levels
        a_cur = as[lvl]
        b_cur = bs[lvl]
        c_cur = cs[lvl]
        d_cur = ds[lvl]

        m = length(b_cur)
        half = m ÷ 2

        a_new = zeros(Float64, half)
        b_new = zeros(Float64, half)
        c_new = zeros(Float64, half)
        d_new = zeros(Float64, half)

        # At this level, we are keeping the odd indices of the current system:
        # i_old = 1,3,5,... (2*j - 1 for j=1:half)
        @simd for j in 1:half
            @inbounds begin 
                i = 2*j - 1  # odd index in current level system
                iL = i - 1   # left neighbor
                iR = i + 1   # right neighbor

                inL = (iL >= 1 && iL <= m)
                inR = (iR >= 1 && iR <= m)

                # Compute AL, CR, DL, DR only if neighbors exist
                AL = inL ? (a_cur[i]*a_cur[iL]/b_cur[iL]) : 0.0
                CR = inR ? (c_cur[i]*c_cur[iR]/b_cur[iR]) : 0.0
                DL = inL ? (a_cur[i]*d_cur[iL]/b_cur[iL]) : 0.0
                DR = inR ? (c_cur[i]*d_cur[iR]/b_cur[iR]) : 0.0

                b_new[j] = b_cur[i]
                if inL
                    b_new[j] -= (a_cur[i]*c_cur[iL]/b_cur[iL])
                end
                if inR
                    b_new[j] -= (c_cur[i]*a_cur[iR]/b_cur[iR])
                end

                d_new[j] = d_cur[i] - DL - DR
                a_new[j] = inL ? (-AL) : 0.0
                c_new[j] = inR ? (-CR) : 0.0
            end
        end

        as[lvl+1] = a_new
        bs[lvl+1] = b_new
        cs[lvl+1] = c_new
        ds[lvl+1] = d_new
    end

    # Now we have a single-equation system at the top:
    x_top = ds[end][1] / bs[end][1]

    # Back substitution
    # We'll reconstruct the solution level by level.
    # At the top level, we have just one equation: x[1].
    x = [x_top]

    # Each backward step doubles the number of equations.
    # At each level going downward, we know the solution at the "odd" indices of that level
    # and must solve for the "even" indices.
    for lvl in reverse(1:levels)
        a_cur = as[lvl]
        b_cur = bs[lvl]
        c_cur = cs[lvl]
        d_cur = ds[lvl]

        m = length(b_cur)
        half = m ÷ 2

        # The x array at the next level up (x) has length half,
        # corresponding to indices i=1,3,5,... of the current level.
        # We create a new array of length m for the current level.
        x_fine = zeros(Float64, m)

        # Place known values from the coarser level into odd indices
        @simd for j in 1:half
            i = 2*j - 1
            x_fine[i] = x[j]
        end

        # Solve for the even indices:
        # For each even index i=2,4,6,..., we use:
        # x[i] = (d[i] - a[i]*x[i-1] - c[i]*x[i+1]) / b[i]
        # If i-1 or i+1 is out of range, their contributions vanish.
        @simd for i in 1:m
            if x_fine[i] == 0.0  # not assigned yet, must be an even index
                iL = i - 1
                iR = i + 1
                xL = (iL>=1 && iL<=m) ? x_fine[iL] : 0.0
                xR = (iR>=1 && iR<=m) ? x_fine[iR] : 0.0
                x_fine[i] = (d_cur[i] - a_cur[i]*xL - c_cur[i]*xR)/b_cur[i]
            end
        end

        x = x_fine
    end

    return x
end




end # End of module