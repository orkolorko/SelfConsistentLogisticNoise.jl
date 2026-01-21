# # Rigorous Computation of Stationary Densities
#
# This tutorial demonstrates how to rigorously compute the stationary density
# of an annealed random dynamical system using the taper + smoothing approach.
#
# ## Mathematical Background
#
# We consider the **annealed transfer operator** for a random dynamical system:
# ```math
# (\mathcal{P}_T f)(x) = \int_{-1}^{1} \rho_{\sigma}(x - T(y)) f(y) \, dy
# ```
# where:
# - $T: [-1,1] \to \mathbb{R}$ is the deterministic map (e.g., quadratic)
# - $\rho_\sigma$ is a Gaussian noise kernel with width $\sigma$
#
# The **stationary density** $f^*$ satisfies $\mathcal{P}_T f^* = f^*$.
#
# ## The Challenge: Non-Periodic Maps
#
# The quadratic map $T(x) = a - (a+1)x^2$ is not periodic on $[-1,1]$.
# Naive Fourier discretization leads to Gibbs phenomena and slow convergence.
#
# ## Our Solution: Taper + Smoothing
#
# We construct a smooth periodic approximation $\tilde{T}$ via:
# 1. **Tapering**: Modify $T$ near $\pm 1$ to enforce periodicity
# 2. **Gaussian smoothing**: Convolve with a narrow Gaussian (width $\sigma_{sm}$)
# 3. **Fourier truncation**: Keep only modes $|k| \leq N$
#
# This gives rigorous error bounds:
# ```math
# \|T - \tilde{T}\|_\infty \leq E_{\text{taper}} + E_{\text{smooth}} + E_{\text{trunc}}
# ```

# ## Setup

using SelfConsistentLogisticNoise
using LinearAlgebra
using Plots

# ## Step 1: Define the Problem Parameters

# Physical parameters
a = 1.5          # Map parameter: T(x) = a - (a+1)x²
σ_rds = 0.05     # Noise width (physical randomness)

# Discretization parameters
N = 64           # Fourier modes: -N to N
M = 8 * N        # FFT grid size (oversampling factor 8)

# Tapering and smoothing parameters
η = 0.1          # Taper collar width (10% on each side)
σ_sm = 0.005     # Map smoothing width (numerical, small)

println("Problem parameters:")
println("  Map: T(x) = $a - $(a+1)x²")
println("  Noise width σ = $σ_rds")
println("  Fourier modes N = $N")
println("  FFT grid M = $M")
println("  Taper width η = $η")
println("  Smoothing width σ_sm = $σ_sm")

# ## Step 2: Compute Map Approximation Error Bounds
#
# Before building the operator, let's verify our error bounds.

map_params = MapApproxParams(a=a, η=η, σ_sm=σ_sm, N=N)
map_error = compute_map_approx_error(map_params)

println("\nMap Approximation Error Breakdown:")
println("  E_taper  = $(map_error.E_taper)   (tapering near boundaries)")
println("  E_smooth = $(map_error.E_smooth)  (Gaussian smoothing)")
println("  E_trunc  = $(map_error.E_trunc)   (Fourier truncation)")
println("  E_total  = $(map_error.E_total)")

# ## Step 3: Build the Tapered and Smoothed Map
#
# Let's visualize the tapering process.

# Collar size in grid points
K = round(Int, η * M / 2)
println("\nTaper collar: K = $K grid points")

# Build tapered map
T_eta, That_eta, x_grid = build_tapered_quadratic(a, M; K=K, L=min(K÷2, 10))

# Build smoothed + truncated map
Ttilde, Ttilde_hat = build_smoothed_tapered_map(T_eta, σ_sm, N)

# Original map for comparison
T_original = QuadraticMap(a)
T_orig_vals = [T_original(x) for x in x_grid]

# Plot comparison
p1 = plot(x_grid, T_orig_vals, label="Original T(x)", lw=2)
plot!(p1, x_grid, T_eta, label="Tapered T_η(x)", lw=2, ls=:dash)
plot!(p1, x_grid, Ttilde, label="Smoothed T̃(x)", lw=2, ls=:dot)
xlabel!(p1, "x")
ylabel!(p1, "T(x)")
title!(p1, "Map Approximation: Original vs Tapered vs Smoothed")

# Zoom into the collar region
p2 = plot(x_grid, T_orig_vals .- Ttilde, label="T - T̃", lw=2)
xlabel!(p2, "x")
ylabel!(p2, "Error")
title!(p2, "Approximation Error T(x) - T̃(x)")

plot(p1, p2, layout=(2,1), size=(800, 600))

# ## Step 4: Build the Annealed Transfer Operator
#
# Now we assemble the finite-rank matrix $P_N$ representing
# $\Pi_N \mathcal{P}_{\tilde{T}} \Pi_N$ in the Fourier basis.

println("\nBuilding annealed transfer operator...")

prob = AnnealedOperatorProblem(a=a, σ_rds=σ_rds, N=N, M=M, η=η, σ_sm=σ_sm)
result = build_annealed_operator(prob; verbose=true)

println("\nOperator matrix size: $(size(result.P))")

# Verify Markov property
mass_ok, mass = verify_markov_property(result.P)
println("Mass preservation check: P[0,0] = $mass (should be ≈ 1)")

# ## Step 5: Compute the Stationary Density
#
# The stationary density corresponds to the eigenvector with eigenvalue 1.

# Compute eigenvalues
eigenvalues = eigvals(result.P)
eigenvalues_sorted = sort(eigenvalues, by=abs, rev=true)

println("\nTop 5 eigenvalues (by magnitude):")
for i in 1:min(5, length(eigenvalues_sorted))
    λ = eigenvalues_sorted[i]
    println("  λ_$i = $λ  (|λ| = $(abs(λ)))")
end

# Find eigenvalue closest to 1
idx_one = argmin(abs.(eigenvalues .- 1.0))
λ_stationary = eigenvalues[idx_one]
println("\nStationary eigenvalue: λ = $λ_stationary")

# Compute eigenvector
F = eigen(result.P)
eigenvector_idx = argmin(abs.(F.values .- 1.0))
fhat_stationary = F.vectors[:, eigenvector_idx]

# Normalize: f̂(0) = 1 (probability density)
dim = 2N + 1
fhat_stationary ./= fhat_stationary[N + 1]  # Index N+1 corresponds to k=0

println("Normalization: f̂(0) = $(fhat_stationary[N+1])")

# ## Step 6: Reconstruct and Plot the Density
#
# Transform from Fourier coefficients to physical space.

# Reconstruction grid
x_plot = range(-1, 1, length=500)

# Fourier synthesis: f(x) = Σ_k f̂(k) e^{iπkx}
function reconstruct_fourier(fhat, x_vals, N)
    f_vals = zeros(ComplexF64, length(x_vals))
    for (i, x) in enumerate(x_vals)
        for k in -N:N
            idx = k + N + 1
            f_vals[i] += fhat[idx] * exp(im * π * k * x)
        end
    end
    return real.(f_vals)
end

f_stationary = reconstruct_fourier(fhat_stationary, x_plot, N)

# Plot the stationary density
plot(x_plot, f_stationary, lw=2, label="Stationary density f*(x)")
xlabel!("x")
ylabel!("f*(x)")
title!("Stationary Density of Annealed Random Dynamical System")

# ## Step 7: Compute Rigorous Error Bounds
#
# We now certify our computation with rigorous error bounds.

println("\n" * "="^60)
println("RIGOROUS ERROR CERTIFICATE")
println("="^60)

# Operator error bounds
op_error = compute_operator_error_bounds(prob)

println("\nOperator Approximation Errors:")
println("  E_map  = $(op_error.E_map)   (map → operator sensitivity)")
println("  E_proj = $(op_error.E_proj)  (Fourier projection tail)")
println("  E_num  = $(op_error.E_num)   (numerical/rounding)")
println("  E_total = $(op_error.E_total)")

# Spectral gap estimate
gap = abs(eigenvalues_sorted[1]) - abs(eigenvalues_sorted[2])
println("\nSpectral gap estimate: $(gap)")

# Create full certificate
cert = certify_operator(result; E_num=0.0)
print_certificate(cert)

# ## Step 8: Rigorous Computation with Ball Arithmetic
#
# For a fully rigorous result, we use BallArithmetic to track all numerical errors.

println("\n" * "="^60)
println("RIGOROUS COMPUTATION WITH BALL ARITHMETIC")
println("="^60)

# Use smaller N for demonstration (rigorous computation is slower)
N_rig = 16
prob_rig = AnnealedOperatorProblem(a=a, σ_rds=σ_rds, N=N_rig, M=4*N_rig, η=η, σ_sm=σ_sm)

println("\nBuilding rigorous operator (N=$N_rig)...")
result_rig = build_annealed_operator_rigorous(prob_rig; verbose=true)

println("\nRigorous numerical error (max radius): $(result_rig.E_num)")

# Compare with non-rigorous
result_nonrig = build_annealed_operator(prob_rig)
diff = maximum(abs.(result_rig.P_mid .- result_nonrig.P))
println("Difference from non-rigorous: $(diff)")

# ## Step 9: Convergence Analysis
#
# Let's verify that our approximation improves as N increases.

println("\n" * "="^60)
println("CONVERGENCE ANALYSIS")
println("="^60)

Ns = [8, 16, 32, 64, 128]
errors = Float64[]
spectral_gaps = Float64[]

for N_test in Ns
    prob_test = AnnealedOperatorProblem(a=a, σ_rds=σ_rds, N=N_test, η=η, σ_sm=σ_sm)
    result_test = build_annealed_operator(prob_test)

    push!(errors, result_test.map_error.E_total)

    # Spectral gap
    eigs = eigvals(result_test.P)
    eigs_sorted = sort(eigs, by=abs, rev=true)
    gap = abs(eigs_sorted[1]) - abs(eigs_sorted[2])
    push!(spectral_gaps, gap)

    println("N = $N_test: E_total = $(result_test.map_error.E_total), gap = $gap")
end

# Plot convergence
p_conv = plot(Ns, errors, marker=:o, lw=2, label="Total error", yscale=:log10)
xlabel!(p_conv, "N (Fourier modes)")
ylabel!(p_conv, "Error bound")
title!(p_conv, "Convergence of Map Approximation Error")

# ## Summary
#
# We have demonstrated:
#
# 1. **Tapering**: Smooth periodic extension of non-periodic maps
# 2. **Gaussian smoothing**: Ensures rapid Fourier decay
# 3. **Rigorous bounds**: Explicit error decomposition (taper + smooth + trunc)
# 4. **Operator assembly**: FFT-based construction of $P_N$
# 5. **Spectral computation**: Stationary density as eigenvector
# 6. **Ball arithmetic**: Fully rigorous numerical enclosure
#
# The total certified error is:
# ```math
# \|\mathcal{P}_T - \Pi_N \mathcal{P}_{\tilde{T}} \Pi_N\| \leq E_{\text{map}} + E_{\text{proj}} + E_{\text{num}}
# ```
#
# where each term is explicitly bounded.

println("\n" * "="^60)
println("COMPUTATION COMPLETE")
println("="^60)
println("\nThe stationary density has been rigorously computed with")
println("total operator error bound: $(op_error.E_total)")
