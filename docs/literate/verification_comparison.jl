# # Comparison of Newton-Kantorovich and Krawczyk Verification Methods
#
# This tutorial compares two rigorous verification approaches for certifying
# the existence of fixed points in self-consistent noisy dynamical systems:
#
# 1. **Newton-Kantorovich** (NK): Based on Corollary 1 from Version6.tex
# 2. **Krawczyk Contraction** (KC): Based on the Krawczyk CAP approach
#
# Both methods provide Computer-Assisted Proofs (CAPs) that rigorously certify
# the existence and uniqueness of a true fixed point near a numerical approximation.

# ## Mathematical Background
#
# ### The Self-Consistent Problem
#
# We seek fixed points of the operator:
# ```math
# T(f) = P_{c(f)} f
# ```
# where:
# - $P_c$ is the transfer operator for the noisy logistic map shifted by $c$
# - $c(f) = \delta G(m(f))$ is the self-consistent shift
# - $m(f) = \langle \phi, f \rangle$ is the observable
#
# ### Newton-Kantorovich Approach
#
# The NK method verifies the **Kantorovich condition**:
# ```math
# h = M^2 \gamma \Delta \leq \frac{1}{2}
# ```
# where:
# - $M = \|J^{-1}\|$ is the inverse Jacobian norm
# - $\gamma$ is the Lipschitz constant of $DT$
# - $\Delta$ is the residual bound
#
# ### Krawczyk Contraction Approach
#
# The KC method checks if the preconditioned map $G(u) = u - A F(u)$ is a contraction:
# ```math
# Y + Z \cdot r \leq r
# ```
# where:
# - $Y = \|A F(0)\|$ (preconditioned residual)
# - $Z = \|I - A J\|$ (contraction factor)
# - $r$ is the ball radius
#
# Both give error bounds on $\|f^* - \tilde{f}\|_2$.

# ## Setup

using SelfConsistentLogisticNoise
using LinearAlgebra
using Printf

# ## Step 1: Define the Problem

# Physical parameters
a = 3.83          # Logistic map parameter
σ = 0.02          # Noise width
δ = 0.1           # Coupling strength
N = 128            # Fourier truncation

println("="^70)
println("SELF-CONSISTENT FIXED POINT VERIFICATION")
println("="^70)
println("\nProblem Parameters:")
println("  a = $a (logistic map)")
println("  σ = $σ (noise width)")
println("  δ = $δ (coupling strength)")
println("  N = $N (Fourier modes)")

# ## Step 2: Compute Numerical Fixed Point

println("\n" * "-"^70)
println("Step 1: Computing Numerical Fixed Point")
println("-"^70)

# Build problem
prob = build_problem(a=a, σ=σ, N=N, δ=δ, coupling_type=:linear, cache=false)

# Solve using hybrid Picard→Newton method for best accuracy
result = solve_hybrid(prob; α=0.3, picard_tol=1e-6, newton_tol=1e-14, verbose=false)

println("  Solver converged: $(result.converged)")
println("  Final residual: $(result.residual)")
println("  Observable m = $(result.m)")
println("  Iterations: $(result.iterations)")

# Extract candidate
fhat = result.fhat

# ## Step 3: Newton-Kantorovich Verification

println("\n" * "-"^70)
println("Step 2: Newton-Kantorovich Verification")
println("-"^70)

# Run NK verification
τ_NK = 0.01  # Analyticity strip parameter
nk_result = verify_fixed_point(prob, fhat; τ=τ_NK, verbose=false)

println("\nNewton-Kantorovich Constants:")
println("  M_N = $(nk_result.M_N)  (numerical inverse bound)")
println("  ε_J = $(nk_result.ε_J)  (Jacobian discretization error)")
println("  M   = $(nk_result.M)   (certified inverse bound)")
println("  Δ   = $(nk_result.Δ)   (total residual)")
println("  γ   = $(nk_result.γ)   (Lipschitz constant)")

println("\nResidual Breakdown:")
println("  δ_N   = $(nk_result.residual_bounds.δ_N)  (finite-dim residual)")
println("  e_T   = $(nk_result.residual_bounds.e_T)  (truncation error)")
println("  e_mis = $(nk_result.residual_bounds.e_mis)  (mismatch error)")

println("\nKantorovich Condition:")
println("  h = M² γ Δ = $(nk_result.h)")
println("  Condition h ≤ 1/2: $(nk_result.h <= 0.5 ? "SATISFIED ✓" : "FAILED ✗")")

if nk_result.verified
    println("\nNK VERIFICATION: SUCCESS ✓")
    println("  Certified error radius: r = $(nk_result.r)")
else
    println("\nNK VERIFICATION: FAILED ✗")
    println("  Reason: Kantorovich condition not satisfied")
end

# ## Step 4: Krawczyk Verification

println("\n" * "-"^70)
println("Step 3: Krawczyk Contraction Verification")
println("-"^70)

# Run Krawczyk verification
kraw_result = certify_krawczyk(prob, fhat; tol=1e-4, maxit=30, verbose=false)

println("\nKrawczyk Constants:")
println("  Y = $(kraw_result.Y)  (preconditioned residual ||A F(0)||)")
println("  Z = $(kraw_result.Z)  (contraction factor ||I - A J||)")
println("  r = $(kraw_result.r)  (ball radius)")
println("  Iterations: $(kraw_result.iterations)")

println("\nContraction Condition:")
println("  Y + Z·r = $(kraw_result.Y + kraw_result.Z * kraw_result.r)")
println("  r       = $(kraw_result.r)")
println("  Y + Z·r ≤ r: $(kraw_result.Y + kraw_result.Z * kraw_result.r <= kraw_result.r ? "SATISFIED ✓" : "NOT CHECKED")")

if kraw_result.verified
    println("\nKRAWCZYK VERIFICATION: SUCCESS ✓")
    println("  Certified error radius: r = $(kraw_result.r)")
else
    println("\nKRAWCZYK VERIFICATION: FAILED ✗")
    println("  Reason: $(kraw_result.reason)")
end

# ## Step 5: Full CAP Verification (Krawczyk + Truncation)

println("\n" * "-"^70)
println("Step 4: Full CAP Verification (Krawczyk + Truncation Error)")
println("-"^70)

# Run full CAP verification
τ_CAP = 0.1  # Analyticity strip for truncation bounds
cap_result = verify_fixed_point_CAP(prob, fhat; τ=τ_CAP, verbose=false)

println("\nCAP Error Decomposition:")
println("  Finite-dim error (Krawczyk r): $(cap_result.krawczyk.r)")
println("  Truncation error e_T:          $(cap_result.truncation_error)")
println("  Total error:                   $(cap_result.total_error)")

if cap_result.verified
    println("\nFULL CAP VERIFICATION: SUCCESS ✓")
else
    println("\nFULL CAP VERIFICATION: FAILED ✗")
end

# ## Step 6: Comparison

println("\n" * "="^70)
println("COMPARISON SUMMARY")
println("="^70)

println("\n┌────────────────────────┬─────────────────────┬─────────────────────┐")
println("│        Method          │     Error Radius    │      Verified       │")
println("├────────────────────────┼─────────────────────┼─────────────────────┤")
@printf("│ Newton-Kantorovich     │  %17.2e  │  %17s  │\n",
        nk_result.verified ? nk_result.r : Inf,
        nk_result.verified ? "✓ YES" : "✗ NO")
@printf("│ Krawczyk (finite-dim)  │  %17.2e  │  %17s  │\n",
        kraw_result.verified ? kraw_result.r : Inf,
        kraw_result.verified ? "✓ YES" : "✗ NO")
@printf("│ Full CAP (+ truncation)│  %17.2e  │  %17s  │\n",
        cap_result.verified ? cap_result.total_error : Inf,
        cap_result.verified ? "✓ YES" : "✗ NO")
println("└────────────────────────┴─────────────────────┴─────────────────────┘")

# ## Step 7: Convergence Analysis

println("\n" * "-"^70)
println("Step 5: Convergence Analysis (varying N)")
println("-"^70)

Ns = [16, 32, 64]
nk_radii = Float64[]
kraw_radii = Float64[]
nk_verified = Bool[]
kraw_verified = Bool[]

println("\n  N   │  NK radius   │ NK ok │ Kraw radius │ Kraw ok")
println("──────┼──────────────┼───────┼─────────────┼────────")

for N_test in Ns
    # Build and solve
    prob_test = build_problem(a=a, σ=σ, N=N_test, δ=δ, coupling_type=:linear, cache=false)
    result_test = solve_hybrid(prob_test; α=0.3, picard_tol=1e-6, newton_tol=1e-14, verbose=false)

    if !result_test.converged
        push!(nk_radii, Inf)
        push!(kraw_radii, Inf)
        push!(nk_verified, false)
        push!(kraw_verified, false)
        @printf(" %3d  │  %10s  │   ✗   │  %10s │   ✗\n", N_test, "NC", "NC")
        continue
    end

    # NK verification
    nk_test = verify_fixed_point(prob_test, result_test.fhat; τ=0.01, verbose=false)
    push!(nk_radii, nk_test.verified ? nk_test.r : Inf)
    push!(nk_verified, nk_test.verified)

    # Krawczyk verification
    kraw_test = certify_krawczyk(prob_test, result_test.fhat; verbose=false)
    push!(kraw_radii, kraw_test.verified ? kraw_test.r : Inf)
    push!(kraw_verified, kraw_test.verified)

    @printf(" %3d  │  %10.2e  │   %s   │  %10.2e │   %s\n",
            N_test,
            nk_test.verified ? nk_test.r : Inf,
            nk_test.verified ? "✓" : "✗",
            kraw_test.verified ? kraw_test.r : Inf,
            kraw_test.verified ? "✓" : "✗")
end

# ## Step 8: Parameter Sensitivity

println("\n" * "-"^70)
println("Step 6: Coupling Strength Sensitivity (varying δ)")
println("-"^70)

δs = [0.0, 0.05, 0.1, 0.2, 0.3]

println("\n  δ    │  NK radius   │ NK ok │ Kraw radius │ Kraw ok │   m(f)")
println("───────┼──────────────┼───────┼─────────────┼─────────┼──────────")

for δ_test in δs
    # Build and solve
    prob_test = build_problem(a=a, σ=σ, N=N, δ=δ_test, coupling_type=:linear, cache=false)
    result_test = solve_hybrid(prob_test; α=0.3, picard_tol=1e-6, newton_tol=1e-14, verbose=false)

    if !result_test.converged
        @printf(" %4.2f  │  %10s  │   ✗   │  %10s │   ✗     │    NC\n", δ_test, "NC", "NC")
        continue
    end

    # NK verification
    nk_test = verify_fixed_point(prob_test, result_test.fhat; τ=0.01, verbose=false)

    # Krawczyk verification
    kraw_test = certify_krawczyk(prob_test, result_test.fhat; verbose=false)

    @printf(" %4.2f  │  %10.2e  │   %s   │  %10.2e │   %s     │  %7.4f\n",
            δ_test,
            nk_test.verified ? nk_test.r : Inf,
            nk_test.verified ? "✓" : "✗",
            kraw_test.verified ? kraw_test.r : Inf,
            kraw_test.verified ? "✓" : "✗",
            result_test.m)
end

# ## Discussion
#
# ### Key Differences
#
# 1. **Structure of the proof**:
#    - NK uses the Kantorovich theorem on the Newton iteration
#    - Krawczyk uses a preconditioned contraction mapping
#
# 2. **Error bounds**:
#    - NK bounds depend on $M^2 \gamma \Delta$
#    - Krawczyk bounds depend on $Y/(1-Z)$
#
# 3. **Computational cost**:
#    - NK requires computing Gaussian smoothing constants $S_{\tau,\sigma}$
#    - Krawczyk requires computing $\|I - AJ\|$ using interval/ball arithmetic
#
# ### When to Use Each Method
#
# - **Newton-Kantorovich**: Better theoretical understanding, explicit dependence on
#   analyticity strip τ, useful when Lipschitz constants are known analytically
#
# - **Krawczyk**: Often gives tighter bounds in practice, more robust for
#   problems with small coupling, simpler implementation using BallArithmetic

println("\n" * "="^70)
println("VERIFICATION COMPARISON COMPLETE")
println("="^70)
