
""" FILE:    TridiagonalSolver.jl

    PURPOSE: Contains the Julia LinearAlgebra.jl solver (imported elsewhere),
             capable of solving tridiagonal systems of equations 
             of the form:

             A_i * x_{i-1}  +  B_i * x_i  +  C_i * x_{i+1}  =  D_i

    """

export TridiagonalSolve



""" Tridiagonal solver for tridiagonal equations of the form:

    A_i * xnew_{i-1}  +  B_i * xnew_i  +  C_i * xnew_{i+1}  =  D

    Uses Julia's LinearAlgebra.jl solver, which is more robust than the Thomas 
    algorithm.
    
    - **ARGUMENTS** :   A, 
                        B, 
                        C, 
                        D, 
                        C_prime (same length as A), 
                        D_prime (same length as A), 
                        xnew    (same length as A) 

    - **RETURNS**   :   xnew   
    """
@inline function TridiagonalSolve( A::AbstractVector{Float64},
                                   B::AbstractVector{Float64},
                                   C::AbstractVector{Float64},
                                   D::AbstractVector{Float64},
                             C_prime::AbstractVector{Float64},   # Scratch array for sub-diagonal → dl  (length n-1)
                             D_prime::AbstractVector{Float64},   # Scratch array for super-diagonal → du  (length n-1)
                                xnew::AbstractVector{Float64}    # Solution vector (length n)
                                  )

    n = length(A)  # All vectors have the same length

    # ---- Build LAPACK inputs in place ---------------------------------
    @inbounds copyto!(C_prime, 1, A, 2, n-1)   # Sub-diagonal   dl ← A[2:end]
    @inbounds copyto!(D_prime, 1, C, 1, n-1)   # Super-diagonal du ← C[1:end-1]

    # gtsv! overwrites its RHS argument with the solution,
    # so we protect the original RHS, D, by copying it to xnew first:
    copyto!(xnew, D)         

    # Since B has the right length, we don't need a scratch array for it, 
    # and use it directly:
    LinearAlgebra.LAPACK.gtsv!(C_prime, B, D_prime, xnew)  # Argument order:  sub-diag,  main-diag, super-diag,  RHS-copy
                                                           #                  (A)       (B)         (C)          (D → xnew)

    return xnew # Now holds the solution vector xnew. D is unchanged.
end
