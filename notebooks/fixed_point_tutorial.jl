# # Computing a Self-Consistent Fixed Point
#
# This tutorial demonstrates how to compute a fixed point of the self-consistent
# noisy transfer operator for the logistic map.
#
# ## Mathematical Setup
#
# We consider the logistic map $T_a(x) = a \cdot x(1-x)$ on the torus $\mathbb{T} = [0,1)$
# with periodized Gaussian noise of variance $\sigma^2$.
#
# The self-consistent operator is:
# ```math
# \mathcal{T}(f) = P_{c(f)} f
# ```
# where $c(f) = \delta \cdot G(m(f))$ and $m(f) = \langle \cos(2\pi x), f \rangle$.
#
# Our goal is to find a fixed point $f^*$ such that $\mathcal{T}(f^*) = f^*$.

# ## Step 1: Load the Package

using SelfConsistentLogisticNoise
using LinearAlgebra

# ## Step 2: Set Parameters
#
# We fix:
# - $a = 3.83$ (logistic parameter, in the chaotic regime)
# - $\sigma = 0.02$ (noise standard deviation)
# - $\delta = 0.1$ (coupling strength)
# - $N = 64$ (Fourier truncation level)

a = 3.83      # Logistic parameter
σ = 0.02      # Noise standard deviation
δ = 0.1       # Coupling strength
N = 64        # Fourier truncation (2N+1 = 129 modes)

# ## Step 3: Build the Problem
#
# The `build_problem` function constructs:
# 1. The logistic map $T_a$
# 2. The Gaussian noise kernel $\bar{\rho}_\sigma$
# 3. The Fourier discretization parameters
# 4. The coupling (here, linear: $G(m) = m$)
# 5. The transfer matrix $B_a$ (computed via FFT)

prob = build_problem(
    a = a,
    σ = σ,
    N = N,
    δ = δ,
    coupling_type = :linear,  # G(m) = m
    cache = true              # Cache B matrix to disk
)

# Let's examine the problem structure:
println("Map parameter a = ", prob.map.a)
println("Noise σ = ", prob.noise.σ)
println("Fourier modes: N = ", prob.disc.N, " (", 2*prob.disc.N+1, " coefficients)")
println("FFT grid size M = ", prob.disc.M)
println("Coupling δ = ", get_delta(prob.coupling))

# ## Step 4: Solve for the Fixed Point
#
# We use damped Picard iteration:
# ```math
# \hat{f}^{(n+1)} = (1-\alpha) \hat{f}^{(n)} + \alpha \, \widehat{\mathcal{T}(f^{(n)})}
# ```
# with normalization $\hat{f}_0 = 1$ enforced at each step.

result = solve_fixed_point(
    prob;
    α = 0.3,           # Damping parameter (0 < α ≤ 1)
    tol = 1e-12,       # Convergence tolerance
    maxit = 5000,      # Maximum iterations
    init = :uniform,   # Start from uniform density
    verbose = true     # Print progress
)

# Check the result:
println("\n=== Results ===")
println("Converged: ", result.converged)
println("Iterations: ", result.iterations)
println("Final residual: ", result.residual)
println("Observable m(f*) = ", result.m)
println("Shift c(f*) = ", result.c)

# ## Step 5: Verify Normalization and Reality
#
# A density must satisfy:
# 1. $\hat{f}_0 = 1$ (integral equals 1)
# 2. $\hat{f}_{-k} = \overline{\hat{f}_k}$ (real-valued density)

fhat = result.fhat

# Check normalization
println("\nNormalization check: f̂₀ = ", fhat[idx(0, N)])

# Check conjugate symmetry (reality)
max_asymmetry = maximum(abs(fhat[idx(k, N)] - conj(fhat[idx(-k, N)])) for k in 1:N)
println("Max conjugate asymmetry: ", max_asymmetry)

# ## Step 6: Reconstruct the Density
#
# Convert from Fourier coefficients to real-space density:

x_grid, f_density = reconstruct_density(fhat; npts = 500)

# Verify the density is real and non-negative (approximately):
println("\nDensity properties:")
println("  Max imaginary part: ", maximum(abs.(imag.(f_density))))
println("  Min value: ", minimum(real.(f_density)))
println("  Max value: ", maximum(real.(f_density)))
println("  Integral (trapezoidal): ", sum(f_density) * (x_grid[2] - x_grid[1]))

# ## Step 7: Examine Fourier Coefficients
#
# The Fourier coefficients should decay for smooth densities:

println("\nFourier coefficient magnitudes:")
for k in [0, 1, 2, 5, 10, 20, N]
    println("  |f̂_$k| = ", abs(fhat[idx(k, N)]))
end

# ## Step 8: Effect of the Coupling
#
# Compare with δ = 0 (no self-consistency):

prob_uncoupled = build_problem(a=a, σ=σ, N=N, δ=0.0)
result_uncoupled = solve_fixed_point(prob_uncoupled; α=0.3, tol=1e-12)

println("\n=== Comparison: δ=0 vs δ=$δ ===")
println("δ = 0:   m = ", result_uncoupled.m, ", c = ", result_uncoupled.c)
println("δ = $δ: m = ", result.m, ", c = ", result.c)

# ## Step 9: Rigorous Verification (Optional)
#
# If BallArithmetic.jl is available, we can rigorously verify the fixed point:

#=
# Uncomment to run rigorous verification:
rig = verify_fixed_point(prob, result.fhat; τ=0.01, verbose=true)

if rig.verified
    println("\n✓ Rigorously verified!")
    println("  Error bound: ‖f* - f̃‖₂ ≤ ", rig.r)
    println("  Kantorovich h = ", rig.h, " ≤ 1/2")
else
    println("\n✗ Verification failed")
    println("  h = ", rig.h)
end
=#

# ## Summary
#
# We computed a self-consistent fixed point for the noisy logistic map with:
# - Parameter $a = 3.83$
# - Noise $\sigma = 0.02$
# - Coupling $\delta = 0.1$
#
# The solver converged in $(result.iterations) iterations to a density with
# observable value $m(f^*) = $(round(result.m, digits=6))$.

println("\n=== Tutorial Complete ===")
