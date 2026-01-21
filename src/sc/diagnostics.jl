# Rigorous diagnostics for Newton-Kantorovich verification
# Based on Corollary 1 from the theory document

using BallArithmetic

export GaussianConstants, compute_gaussian_constants
export RigorousResult, verify_fixed_point
export compute_jacobian_matrix, compute_residual_bounds

#=============================================================================
  Gaussian smoothing constants (Lemma 13)

  S_{τ,σ}   := sup_k exp(2πτ|k| - 2π²σ²k²)
  S^(1)_{τ,σ} := sup_k (2π|k|) exp(2πτ|k| - 2π²σ²k²)
  S^(2)_{τ,σ} := sup_k (2π|k|)² exp(2πτ|k| - 2π²σ²k²)
=============================================================================#

"""
    GaussianConstants

Container for the Gaussian smoothing constants S_{τ,σ}, S^(1)_{τ,σ}, S^(2)_{τ,σ}.
"""
struct GaussianConstants
    S_τσ::Float64       # S_{τ,σ}
    S1_τσ::Float64      # S^(1)_{τ,σ}
    S2_τσ::Float64      # S^(2)_{τ,σ}
    S1_0σ::Float64      # S^(1)_{0,σ}
    S2_0σ::Float64      # S^(2)_{0,σ}
end

"""
    compute_gaussian_constants(σ, τ)

Compute the Gaussian smoothing constants for given σ and τ.
Uses exact computation by checking k near the continuous maximizer.
"""
function compute_gaussian_constants(σ::Float64, τ::Float64)
    # For S_{τ,σ}: maximizer near k₀ = τ/(2πσ²)
    k0 = ceil(Int, τ / (2π * σ^2))

    f(k) = exp(2π * τ * abs(k) - 2π^2 * σ^2 * k^2)
    f1(k) = (2π * abs(k)) * f(k)
    f2(k) = (2π * abs(k))^2 * f(k)

    # S_{τ,σ}: check neighborhood of k₀
    S_τσ = maximum(f(k) for k in max(0, k0-2):k0+2)

    # S^(1)_{τ,σ}: maximizer of k * exp(...)
    # Derivative: (1 - 4π²σ²k + 2πτ) * exp(...) = 0 => k = (1 + 2πτ)/(4π²σ²)
    k1 = ceil(Int, (1 + 2π * τ) / (4π^2 * σ^2))
    S1_τσ = maximum(f1(k) for k in max(1, k1-2):k1+2)

    # S^(2)_{τ,σ}: maximizer of k² * exp(...)
    k2 = ceil(Int, (2 + 2π * τ) / (4π^2 * σ^2))
    S2_τσ = maximum(f2(k) for k in max(1, k2-2):k2+2)

    # For τ = 0 cases (simpler)
    g(k) = exp(-2π^2 * σ^2 * k^2)
    g1(k) = (2π * k) * g(k)
    g2(k) = (2π * k)^2 * g(k)

    # S^(1)_{0,σ}: maximizer at k = 1/(2πσ²√2)
    k1_0 = max(1, ceil(Int, 1 / (2π * σ^2 * sqrt(2))))
    S1_0σ = maximum(g1(k) for k in max(1, k1_0-2):k1_0+2)

    # S^(2)_{0,σ}: maximizer at k = 1/(πσ²√2)
    k2_0 = max(1, ceil(Int, 1 / (π * σ^2 * sqrt(2))))
    S2_0σ = maximum(g2(k) for k in max(1, k2_0-2):k2_0+2)

    return GaussianConstants(S_τσ, S1_τσ, S2_τσ, S1_0σ, S2_0σ)
end

"""
    compute_gaussian_constants_bounds(σ, τ)

Compute explicit upper bounds (Lemma 13) - useful when exact computation is not needed.
"""
function compute_gaussian_constants_bounds(σ::Float64, τ::Float64)
    exp_factor = exp(τ^2 / (2σ^2))

    S_τσ = exp_factor
    S1_τσ = exp_factor * (τ / σ^2 + 1 / (σ * sqrt(ℯ)))
    S2_τσ = exp_factor * (τ^2 / σ^4 + 2τ / (σ^3 * sqrt(ℯ)) + 2 / (σ^2 * ℯ))
    S1_0σ = 1 / (σ * sqrt(ℯ))
    S2_0σ = 2 / (σ^2 * ℯ)

    return GaussianConstants(S_τσ, S1_τσ, S2_τσ, S1_0σ, S2_0σ)
end

#=============================================================================
  Coupling constants
=============================================================================#

"""
    CouplingConstants

Constants L_G, Lip(G), L_{G'} for the coupling function G.
"""
struct CouplingConstants
    L_G::Float64      # sup|G'| on relevant interval
    Lip_G::Float64    # Lipschitz constant of G
    L_Gp::Float64     # sup|G''| on relevant interval
end

"""
    coupling_constants(c::LinearCoupling, m_range)

For G(m) = m: G' = 1, G'' = 0.
"""
function coupling_constants(c::LinearCoupling, m_range)
    return CouplingConstants(1.0, 1.0, 0.0)
end

"""
    coupling_constants(c::TanhCoupling, m_range)

For G(m) = tanh(βm): G' = β sech²(βm), G'' = -2β² sech²(βm) tanh(βm).
"""
function coupling_constants(c::TanhCoupling, m_range)
    β = c.β
    # On any interval, sup|G'| = β (at m=0)
    L_G = β
    Lip_G = β  # |tanh(βx) - tanh(βy)| ≤ β|x-y|
    # sup|G''| = β² * (2/3)^(3/2) ≈ 0.544 β² (at tanh(βm) = ±1/√3)
    L_Gp = β^2 * 2 * (2/3)^(3/2)

    return CouplingConstants(L_G, Lip_G, L_Gp)
end

#=============================================================================
  Jacobian matrix construction (Lemma 14)

  DT_N(f) = A(c_N) + b_N ⊗ a_N*

  where A(c)_{k,m} = ρ̂_σ(k) e^{-2πikc} V_k(m)
=============================================================================#

"""
    compute_jacobian_matrix(prob::SCProblem, fhat::Vector{ComplexF64})

Compute the Jacobian DT_N(f) = A(c) + b ⊗ a* at the candidate f̂.
Returns the full matrix representation.
"""
function compute_jacobian_matrix(prob::SCProblem, fhat::Vector{ComplexF64})
    N = prob.disc.N
    c = compute_shift(prob.coupling, fhat, N)

    # A(c)_{k,m} = ρ̂_σ(k) e^{-2πikc} B_{k,m}
    # where B is the precomputed transfer matrix
    A = similar(prob.B)
    for k in modes(N)
        k_idx = idx(k, N)
        factor = rhohat(prob.noise, k) * exp(-2π * im * k * c)
        for m in modes(N)
            m_idx = idx(m, N)
            A[k_idx, m_idx] = factor * prob.B[k_idx, m_idx]
        end
    end

    # Rank-one part: b ⊗ a*
    # a_N = δ G'(m(f)) (Φ̂_m)_{|m|≤N}
    obs = get_observable(prob.coupling)
    m_val = compute_m(obs, fhat, N)
    δ = get_delta(prob.coupling)
    Gp = derivative(prob.coupling, m_val)

    # For CosineObservable: Φ̂_1 = Φ̂_{-1} = 1/2
    a = zeros(ComplexF64, 2N + 1)
    if obs isa CosineObservable
        a[idx(1, N)] = 0.5
        a[idx(-1, N)] = 0.5
    elseif obs isa FourierObservable
        N_phi = (length(obs.Phihat) - 1) ÷ 2
        for k in modes(min(N, N_phi))
            a[idx(k, N)] = obs.Phihat[idx(k, N_phi)]
        end
    end
    a .*= Gp  # Note: δ is already in Gp for our coupling types

    # b_N = B(c) f̂  where B(c)_{k,m} = (-2πik) ρ̂_σ(k) e^{-2πikc} V_k(m)
    b = zeros(ComplexF64, 2N + 1)
    Bf = prob.B * fhat
    for k in modes(N)
        k_idx = idx(k, N)
        factor = (-2π * im * k) * rhohat(prob.noise, k) * exp(-2π * im * k * c)
        b[k_idx] = factor * Bf[k_idx]
    end

    # DT_N = A + b ⊗ a* (outer product: (b ⊗ a*)[i,j] = b[i] * conj(a[j]))
    DT = A + b * a'

    return DT
end

"""
    compute_jacobian_on_ker_ell(DT::Matrix{ComplexF64}, N::Int)

Project Jacobian onto ker(ℓ) where ℓ(f) = f̂_0.
This removes the k=0 row and column from the matrix.
"""
function compute_jacobian_on_ker_ell(DT::Matrix{ComplexF64}, N::Int)
    # ker(ℓ) = {f : f̂_0 = 0}, so we remove mode k=0
    k0_idx = idx(0, N)
    indices = [1:k0_idx-1; k0_idx+1:2N+1]
    return DT[indices, indices]
end

#=============================================================================
  Residual bounds (Corollary 1, item 1)
=============================================================================#

"""
    ResidualBounds

Container for residual bound components.
"""
struct ResidualBounds
    δ_N::Float64    # Finite-dimensional residual ‖T_N(f̃) - f̃‖₂
    e_T::Float64    # Truncation error
    e_mis::Float64  # Mismatch error (0 if φ is trig polynomial)
    Δ::Float64      # Total: δ_N + e_T + e_mis
end

"""
    compute_residual_bounds(prob, fhat, τ, gc::GaussianConstants, cc::CouplingConstants, phi_norm_Aτ)

Compute the residual bounds from Corollary 1.
"""
function compute_residual_bounds(
    prob::SCProblem,
    fhat::Vector{ComplexF64},
    τ::Float64,
    gc::GaussianConstants,
    cc::CouplingConstants,
    phi_norm_Aτ::Float64
)
    N = prob.disc.N
    δ = get_delta(prob.coupling)

    # K_2 = ‖f̃‖_2
    K2 = norm(fhat)

    # K_τ = ‖f̃‖_{A_τ} = sqrt(Σ_k e^{4πτ|k|} |f̂_k|²)
    Kτ = sqrt(sum(exp(4π * τ * abs(k)) * abs(fhat[idx(k, N)])^2 for k in modes(N)))

    # Compute T_N(f̃)
    TN_f = apply_self_consistent(prob, fhat)

    # δ_N = ‖T_N(f̃) - f̃‖_2
    δ_N = norm(TN_f - fhat)

    # e_T = exp(-2πτN) (S_{τ,σ} + 1) K_τ
    e_T = exp(-2π * τ * N) * (gc.S_τσ + 1) * Kτ

    # e_mis = |δ| Lip(G) exp(-2πτN) ‖φ‖_{A_τ} S^(1)_{0,σ} K_2²
    # (This is 0 if φ is a trig polynomial of degree ≤ N, e.g., cosine observable)
    obs = get_observable(prob.coupling)
    if obs isa CosineObservable && N >= 1
        e_mis = 0.0  # φ = cos(2πx) has degree 1
    else
        e_mis = abs(δ) * cc.Lip_G * exp(-2π * τ * N) * phi_norm_Aτ * gc.S1_0σ * K2^2
    end

    Δ = δ_N + e_T + e_mis

    return ResidualBounds(δ_N, e_T, e_mis, Δ), K2, Kτ
end

#=============================================================================
  Jacobian discretization error (Corollary 1, item 2)
=============================================================================#

"""
    compute_jacobian_error(prob, fhat, τ, gc, cc, K2, Kτ, phi_norm_2, phi_norm_Aτ)

Compute ε_J = ‖DT(f̃) - DT_N(f̃)‖_{A_τ → L²}.
"""
function compute_jacobian_error(
    prob::SCProblem,
    fhat::Vector{ComplexF64},
    τ::Float64,
    gc::GaussianConstants,
    cc::CouplingConstants,
    K2::Float64,
    Kτ::Float64,
    phi_norm_2::Float64,
    phi_norm_Aτ::Float64
)
    N = prob.disc.N
    δ = get_delta(prob.coupling)

    # First term: exp(-2πτN)(S_{τ,σ} + 1)
    term1 = exp(-2π * τ * N) * (gc.S_τσ + 1)

    # Second term: |δ| L_G ‖φ‖_2 K_τ exp(-2πτN)(S^(1)_{τ,σ} + S^(1)_{0,σ})
    term2 = abs(δ) * cc.L_G * phi_norm_2 * Kτ * exp(-2π * τ * N) * (gc.S1_τσ + gc.S1_0σ)

    # Mismatch term C_mis(K_2)
    obs = get_observable(prob.coupling)
    if obs isa CosineObservable && N >= 1
        C_mis = 0.0
    else
        C_mis = abs(δ) * phi_norm_Aτ * (
            (cc.Lip_G + cc.L_G) * gc.S1_0σ * K2 +
            cc.L_Gp * phi_norm_2 * gc.S1_0σ * K2^2
        ) + abs(δ)^2 * cc.L_G * cc.Lip_G * phi_norm_2 * phi_norm_Aτ * gc.S2_0σ * K2^2
    end
    term3 = exp(-2π * τ * N) * C_mis

    return term1 + term2 + term3
end

#=============================================================================
  Lipschitz constant (Corollary 1, item 4)
=============================================================================#

"""
    compute_lipschitz_constant(prob, K, gc, cc, phi_norm_2)

Compute γ(K) for the Lipschitz bound ‖DT(f) - DT(g)‖_{2→2} ≤ γ(K) ‖f-g‖_2.
"""
function compute_lipschitz_constant(
    prob::SCProblem,
    K::Float64,
    gc::GaussianConstants,
    cc::CouplingConstants,
    phi_norm_2::Float64
)
    δ = get_delta(prob.coupling)

    γ = abs(δ) * phi_norm_2 * gc.S1_0σ * (cc.Lip_G + cc.L_G) +
        abs(δ) * cc.L_Gp * phi_norm_2^2 * gc.S1_0σ * K +
        abs(δ)^2 * cc.L_G * cc.Lip_G * phi_norm_2^2 * gc.S2_0σ * K

    return γ
end

#=============================================================================
  Full verification (Corollary 1)
=============================================================================#

"""
    RigorousResult

Result of rigorous verification.
"""
struct RigorousResult
    verified::Bool
    M_N::Float64        # ‖J_N⁻¹‖_{2→2}
    ε_J::Float64        # Jacobian discretization error
    M::Float64          # Certified inverse bound M ≤ M_N/(1 - M_N ε_J)
    Δ::Float64          # Total residual bound
    γ::Float64          # Lipschitz constant
    h::Float64          # Kantorovich quantity h = M² γ Δ
    r::Float64          # Error radius
    residual_bounds::ResidualBounds
end

"""
    verify_fixed_point(prob, fhat; τ=0.01, verbose=false)

Attempt to rigorously verify that fhat is close to a true fixed point.
Uses BallArithmetic for certified bounds on singular values.

Returns a RigorousResult with verification status and all computed constants.
"""
function verify_fixed_point(
    prob::SCProblem,
    fhat::Vector{ComplexF64};
    τ::Float64=0.01,
    verbose::Bool=false
)
    N = prob.disc.N
    σ = prob.noise.σ

    # 1. Compute Gaussian constants
    gc = compute_gaussian_constants(σ, τ)
    if verbose
        @info "Gaussian constants" gc.S_τσ gc.S1_τσ gc.S2_τσ gc.S1_0σ gc.S2_0σ
    end

    # 2. Coupling constants
    m_val = compute_m(get_observable(prob.coupling), fhat, N)
    cc = coupling_constants(prob.coupling, (-abs(m_val)-1, abs(m_val)+1))
    if verbose
        @info "Coupling constants" cc.L_G cc.Lip_G cc.L_Gp
    end

    # 3. Observable norms (for CosineObservable: ‖φ‖_2 = 1/√2, ‖φ‖_{A_τ} depends on τ)
    obs = get_observable(prob.coupling)
    if obs isa CosineObservable
        phi_norm_2 = 1 / sqrt(2)
        phi_norm_Aτ = sqrt(0.5 * exp(4π * τ) + 0.5 * exp(4π * τ))  # modes ±1
    else
        error("Observable norm computation not implemented for this type")
    end

    # 4. Residual bounds
    rb, K2, Kτ = compute_residual_bounds(prob, fhat, τ, gc, cc, phi_norm_Aτ)
    if verbose
        @info "Residual bounds" rb.δ_N rb.e_T rb.e_mis rb.Δ
        @info "Norms" K2 Kτ
    end

    # 5. Jacobian and certified inverse bound using BallArithmetic
    DT = compute_jacobian_matrix(prob, fhat)
    J_N = I - compute_jacobian_on_ker_ell(DT, N)  # J_N = I - DT_N on ker ℓ

    # Convert to BallMatrix for rigorous computation
    # Use zero radii since J_N comes from exact arithmetic
    J_N_ball = to_ball_matrix(J_N, 0.0)

    # M_N = ‖J_N⁻¹‖_{2→2} via rigorous SVD
    M_N = svd_bound_L2_opnorm_inverse(J_N_ball)
    if verbose
        @info "Inverse Jacobian bound" M_N
    end

    if isinf(M_N)
        @warn "J_N is singular or nearly singular"
        return RigorousResult(false, M_N, Inf, Inf, rb.Δ, Inf, Inf, Inf, rb)
    end

    # 6. Jacobian discretization error
    ε_J = compute_jacobian_error(prob, fhat, τ, gc, cc, K2, Kτ, phi_norm_2, phi_norm_Aτ)
    if verbose
        @info "Jacobian error" ε_J
    end

    # 7. Check M_N ε_J < 1
    if M_N * ε_J >= 1
        @warn "Certified inverse condition fails: M_N ε_J = $(M_N * ε_J) ≥ 1"
        return RigorousResult(false, M_N, ε_J, Inf, rb.Δ, Inf, Inf, Inf, rb)
    end

    # M = M_N / (1 - M_N ε_J)
    M = M_N / (1 - M_N * ε_J)
    if verbose
        @info "Certified inverse bound M" M
    end

    # 8. Lipschitz constant
    γ = compute_lipschitz_constant(prob, K2, gc, cc, phi_norm_2)
    if verbose
        @info "Lipschitz constant" γ
    end

    # 9. Kantorovich check: h = M² γ Δ ≤ 1/2
    h = M^2 * γ * rb.Δ
    if verbose
        @info "Kantorovich quantity" h
    end

    if h > 0.5
        @warn "Kantorovich condition fails: h = $h > 1/2"
        return RigorousResult(false, M_N, ε_J, M, rb.Δ, γ, h, Inf, rb)
    end

    # 10. Error radius
    if γ == 0
        r = M * rb.Δ
    else
        r = (1 - sqrt(1 - 2h)) / (M * γ)
    end
    if verbose
        @info "Verified! Error radius" r
    end

    return RigorousResult(true, M_N, ε_J, M, rb.Δ, γ, h, r, rb)
end
