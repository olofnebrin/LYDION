
module DustyStaticClouds

using Interpolations        # For interpolation of needed integrals (to speed up code)
using QuadGK                # For numerical integration
using DelimitedFiles        # For saving data in text files



""" This is the series acceleration method of Cohen, Villegas, 
    and Zagier (2000). It is suitable for accelerating 
    alternating series of the form ∑ (-1)^n a_n, and converge
    faster than Euler's method. 
    
    Reference: Cohen et al. (2000) 'Convergence Acceleration of
    Alternating Series', Experimental Mathematics, 9 """
function Cohen_accel(a_n)
    T = eltype(a_n)  # Get type of a_n
    n = length(a_n)
    d = (3 + sqrt(T(8)))^n
    d = (d + 1/d)/2
    b = -T(1)
    c = -d
    s = T(0)
    for k = 0:n-1
        c  = b - c
        s += c*a_n[k+1]
        b  = (k + n) * (k - n) * b / ((k + 1/T(2)) * (k + 1))
    end
    return s / d
end


# Specify the file path
current_dir = @__DIR__
file_path   = string(current_dir, "/", "DustyStaticClouds_NoRecoil.txt")

# Read the data from the text file, with header row
data, header    = readdlm(file_path, '\t', Float64, '\n', header=true)

# Extract numeric data (skipping the header)
ϵt_n_data       = data[:, 1]  # ϵtilde_n values
Int_Fn_data     = data[:, 2]  # ∫dz F_n values
Int_Fn_zsq_data = data[:, 3]  # ∫dz z^2 F_n values
D_n_data        = data[:, 4]  # D_n values

# Make interpolations:
Int_Fn          = interpolate((ϵt_n_data,), Int_Fn_data, Gridded(Linear()))
Int_Fn_zsq      = interpolate((ϵt_n_data,), Int_Fn_zsq_data, Gridded(Linear()))
D_n             = interpolate((ϵt_n_data,), D_n_data, Gridded(Linear()))



""" The cloud-averaged Lyα radiation pressure, in units of L_Lyα (a_v τ_cl)^(1/3) /(4πc R_cl^2).
    Arguments:
    ϵt_1_array:   Array of values for ϵ-tilde_1
    τbar_s:       Relative optical depth of emission region (0 for point source, 1 for uniform source)
    P:            √(6π)p_d*τ_cl/2, Lyα destruction parameter
    nmax:         Maximum n to sum up to (recommendation: ~ 1e6)"""
function P_Lyα(ϵt_1_array, τbar_s::Float64, P = 0.0, nmax = 1000000, 
               filename = "P_Lya_DustyStatic.txt", save = true)

    n = 1:nmax
    X = n.*π.*τbar_s

    # Intialize P_Lyα storage array:
    P_Lyα_matrix = []

    for ϵt_1 in ϵt_1_array

        # First get ∫dz F_n and D_n:
        ϵt_1vals       = ϵt_1./(n.^(4.0/3.0))
        integrals_vals = Int_Fn.(ϵt_1vals)
        D_nvals        = D_n.(ϵt_1vals)

        # Initialize array to store P_Lyα for current ϵt_1

        P_Lyα_row = [ϵt_1] # First element in the row is the current ϵt_1

        if τbar_s == 0.0
            # Point source limit:
            stuff = (1.0/3.0)./((P .+ n.*π.*D_nvals) .* n.^(1.0/3.0))
        else
            stuff = (sin.(X) .- X*cos.(X))./((τbar_s^3).*((n.*π).^3).*(P .+ n.*π.*D_nvals).* n.^(1.0/3.0))
        end   
        terms     = -3.0 .* (-1).^n .* stuff .* integrals_vals
        prefactor = sqrt(6)*( (3/(2*(π^3)))^(1/6) )  # Numerical prefactor

        # Get P_Lyα/[ (a_v τ_cl)^(1/3) L_Lyα/(4πc R_cl^2) ]:

        P_Lyα_val = prefactor*sum(terms)
        push!(P_Lyα_row, P_Lyα_val)

        # Append the current row to the full matrix:
        push!(P_Lyα_matrix, P_Lyα_row)
    end

    if save == true
        # Get current directory
        CurrentDir = @__DIR__
        
        #Save the matrix in a text file, 
        FilePath = string(CurrentDir, "/", filename)
        open(FilePath, "w") do io
            # Write description:
            println(io, "# Dusty, static clouds, & no recoil.")
            println(io, "# First column: ϵt_1 values")
            println(io, "# Second column: Values of P_Lyα/[ (a*tau_cl)^(1/3) L_Lyα/(4πc R_cl^2) ]")
            
            # Write the matrix data
            writedlm(io, P_Lyα_matrix)
        end
        println("Done!")
    end  

    return P_Lyα_matrix
end


""" Lyα force multiplier M_F in units of (a_v τ_cl)^(1/3). Arguments:
    ϵt_1_array:   Array of values for ϵ-tilde_1
    τbar_s:       Relative optical depth of emission region (0 for point source, 1 for uniform source)
    P:            √(6π)p_d*τ_cl/2, Lyα destruction parameter
    nmax_direct:  Maximum n to sum up to, after which Euler-Maclaurin formual is used """
function M_F(ϵt_1_array, τbar_s::Float64, P = 0.0, nmax_direct = 100000,
            filename = "M_F_DustyStatic.txt", save = true)

    no_direct = 2 .* (0:nmax_direct) .+ 1
    X_direct  = no_direct .* π .* τbar_s

    # Intialize M_F storage array:
    M_F_matrix = []

    for ϵt_1 in ϵt_1_array
        # First get ∫dz F_n and D_n:
        ϵt_1vals_direct = ϵt_1 ./ (no_direct .^(4.0 / 3.0))
        integrals_vals_direct = Int_Fn.(ϵt_1vals_direct)
        D_nvals_direct = D_n.(ϵt_1vals_direct)

        # Initialize array to store M_F for current ϵt_1
        M_F_row = [ϵt_1]  # First element in the row is the current ϵt_1

        # Handle direct summation (for small n)
        if τbar_s == 0.0
            # Point source limit:
            stuff_direct = (1.0 / 3.0) ./ ((P .+ no_direct .* π .* D_nvals_direct) .* no_direct .^(1.0 / 3.0))
        else
            stuff_direct = (sin.(X_direct) .- X_direct .* cos.(X_direct)) ./ ((τbar_s^3) .* ((no_direct .* π) .^ 3) .* (P .+ no_direct .* π .* D_nvals_direct) .* no_direct .^(1.0 / 3.0))
        end
        terms_direct = stuff_direct .* integrals_vals_direct

        # Prefactor
        prefactor = 4.0 * sqrt(6) * ((3 / (2 * (π^3)))^(1/6))

        # Direct sum for small n:
        M_F_val = prefactor * sum(terms_direct)

        # Add Euler-Maclaurin correction for large n (above nmax_direct)
        function integrand(n)
            no           = 2 * n + 1
            X            = no * π * τbar_s

            ϵt_1val      = ϵt_1 / (no ^ (4.0 / 3.0))
            integral_val = Int_Fn(ϵt_1val)
            D_nval       = D_n(ϵt_1val)

            if τbar_s == 0.0
                stuff = (1.0 / 3.0) / ((P + no * π * D_nval) * no ^ (1.0 / 3.0))
            else
                stuff = (sin(X) - X * cos(X)) / ((τbar_s^3) * (no * π) ^ 3 * (P + no * π * D_nval) * no ^ (1.0 / 3.0))
            end
            return stuff * integral_val
        end

        # Boundary term (the 0.5 term)
        boundary_correction = 0.5 * integrand(nmax_direct + 1)
        
        # Numerical integration for the remaining terms (Euler-Maclaurin tail)
        EulerMac_correction = quadgk(integrand, nmax_direct + 1, Inf, rtol = 1e-4)[1]

        # Add both corrections to the final result
        M_F_val += prefactor * (boundary_correction + EulerMac_correction)

        push!(M_F_row, M_F_val)

        # Append the current row to the full matrix:
        push!(M_F_matrix, M_F_row)
    end

    if save == true
        # Get current directory
        CurrentDir = @__DIR__

        # Save the matrix in a text file
        FilePath = string(CurrentDir, "/", filename)
        open(FilePath, "w") do io
            # Write description:
            println(io, "# Dusty, static clouds, & no recoil.")
            println(io, "# First column: ϵt_1 values")
            println(io, "# Second column: Values of M_F/(a*tau_cl)^(1/3)")

            # Write the matrix data
            writedlm(io, M_F_matrix)
        end
        println("Done!")
    end

    return M_F_matrix
end

""" Lyα escape fraction from dusty static clouds, ignoring recoil. Arguments:
    ϵt_1_array:   Array of values for ϵ-tilde_1
    τbar_s:       Relative optical depth of emission region (0 for point source, 1 for uniform source)
    P:            √(6π)p_d*τ_cl/2, Lyα destruction parameter
    nmax:         Maximum n to sum up to """
function f_escLyα(ϵt_1_array, τbar_s::Float64, P = 0.0, nmax = 300,
    filename = "fesc_DustyStatic.txt", save = true)

    nvals   = 1:nmax
    n       = collect(Float64, nvals)  # Ensure n is an array of Float64
    X       = n .* π .* τbar_s

    # Intialize f_escLyα storage array:
    f_esc_matrix = []

    for ϵt_1 in ϵt_1_array
        # First get ∫dz z^2 F_n and D_n:
        ϵt_1vals       = ϵt_1 ./ (n .^ (4.0 / 3.0))
        integrals_vals = Int_Fn_zsq.(ϵt_1vals)
        D_nvals        = D_n.(ϵt_1vals)

        # Initialize array to store f_escLyα for current ϵt_1
        f_esc_row = [ϵt_1]  # First element in the row is the current ϵt_1

        if τbar_s == 0.0
            # Point source limit:
            stuff = (1.0 / 3.0) .* (n .* π) ./ (P .+ n .* π .* D_nvals)
        else
            stuff = (sin.(X) .- X .* cos.(X)) ./ ((τbar_s^3) .* ((n .* π) .^ 2) .* (P .+ n .* π .* D_nvals))
        end   

        # Element-wise multiplication for terms
        #terms     = -3.0 .* (-1.0) .^ n .* stuff .* integrals_vals

        terms_nonalt = stuff .* integrals_vals

        f_esc_val    = - 3.0 .* Cohen_accel(terms_nonalt)

        # Get f_escLyα:
        #f_esc_val = sum(terms)
        push!(f_esc_row, f_esc_val)

        # Append the current row to the full matrix:
        push!(f_esc_matrix, f_esc_row)
    end

    if save == true
        # Get current directory
        CurrentDir = @__DIR__
        
        #Save the matrix in a text file, 
        FilePath = string(CurrentDir, "/", filename)
        open(FilePath, "w") do io
            # Write description:
            println(io, "# Dusty, static clouds, & no recoil.")
            println(io, "# First column: ϵt_1 values")
            println(io, "# Second column: Values of - f_escLyα")
            
            # Write the matrix data
            writedlm(io, f_esc_matrix)
        end
        println("Done!")
    end  

    return f_esc_matrix
end


# Example usage:
#ϵt_1_array = [0.0, 1.0, 10.0, 100.0]  # Log-uniform array for ϵt_1
FileNameEscape  = "AnalyticalEscapeFraction_UniformSource.txt"
FileNameMF      = "AnalyticalMF_UniformSource_nmax300.txt"
ϵt_1_array      = exp10.( range(-3, stop = 2, length = 300) )
τbar_s          = 1.0  # Relative source size 
P               = 0.0
nmax            = 300 
f_esc_matrix    = f_escLyα(ϵt_1_array, τbar_s, P, nmax, FileNameEscape )
M_F_matrix      = M_F( ϵt_1_array, τbar_s, P, 300, FileNameMF )

###############################################################
#######                                                 #######
#######     S E R I E S  A C C E L E R A T I O N        #######
#######                                                 #######
###############################################################



# Define the series terms a_n for the alternating harmonic series
#n_terms = 100  # Set the number of terms
#a_n = [1.0 / k for k in 1:n_terms]  # Terms for the series without (-1)^n
# Use the Cohen-Villegas-Zagier acceleration method on the series
#approximation = Cohen_accel(a_n)
# Compare the result with the known value of log(2)
#exact_value = log(2)
#println("Cohen-Villegas-Zagier accelerated approximation: ", approximation)
#println("Exact value of log(2): ", exact_value)
#println("Difference: ", abs(approximation - exact_value))

end

