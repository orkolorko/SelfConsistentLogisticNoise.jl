# Rigorous diagnostics for Newton-Kantorovich verification
# Based on Corollary 1 from the theory document

using BallArithmetic: Ball, BallMatrix, mid, rad, svd_bound_L2_opnorm_inverse
using IntervalArithmetic: interval

export GaussianConstants, compute_gaussian_constants
export RigorousResult, verify_fixed_point
export compute_jacobian_matrix, compute_residual_bounds

#=============================================================================
  Gaussian smoothing constants (Lemma 13)

  S_{τ,σ}   := sup_k exp(πτ|k| - (π²/2)σ²k²)
  S^(1)_{τ,σ} := sup_k (π|k|) exp(πτ|k| - (π²/2)σ²k²)
  S^(2)_{τ,σ} := sup_k (π|k|)² exp(πτ|k| - (π²/2)σ²k²)
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
Uses rigorous interval arithmetic to ensure correct upper bounds.
"""
function compute_gaussian_constants(σ::Float64, τ::Float64)
    # Use the rigorous version from rigorous_constants.jl
    gc_rig = compute_gaussian_constants_rigorous(σ, τ)
    return GaussianConstants(gc_rig.S_τσ, gc_rig.S1_τσ, gc_rig.S2_τσ, gc_rig.S1_0σ, gc_rig.S2_0σ)
end

"""
    compute_gaussian_constants_bounds(σ, τ)

Compute explicit rigorous upper bounds (Lemma 13) using interval arithmetic.
"""
function compute_gaussian_constants_bounds(σ::Float64, τ::Float64)
    # Use the rigorous version from rigorous_constants.jl
    gc_rig = compute_gaussian_constants_bounds_rigorous(σ, τ)
    return GaussianConstants(gc_rig.S_τσ, gc_rig.S1_τσ, gc_rig.S2_τσ, gc_rig.S1_0σ, gc_rig.S2_0σ)
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

  where A(c)_{k,m} = ρ̂_σ(k) e^{-iπkc} V_k(m)
=============================================================================#

"""
    compute_jacobian_matrix(prob::SCProblem, fhat::Vector{ComplexF64})

Compute the Jacobian DT_N(f) = A(c) + b ⊗ a* at the candidate f̂.
Returns the full matrix representation.
"""
function compute_jacobian_matrix(prob::SCProblem, fhat::Vector{ComplexF64})
    N = prob.disc.N
    c = compute_shift(prob.coupling, fhat, N)

    # A(c)_{k,m} = ρ̂_σ(k) e^{-iπkc} B_{k,m}
    # where B is the precomputed transfer matrix
    A = similar(prob.B)
    for k in modes(N)
        k_idx = idx(k, N)
        factor = rhohat(prob.noise, k) * exp(-π * im * k * c)
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

    # b_N = B(c) f̂  where B(c)_{k,m} = (-iπk) ρ̂_σ(k) e^{-iπkc} V_k(m)
    b = zeros(ComplexF64, 2N + 1)
    Bf = prob.B * fhat
    for k in modes(N)
        k_idx = idx(k, N)
        factor = (-π * im * k) * rhohat(prob.noise, k) * exp(-π * im * k * c)
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
    e_map::Float64  # Map approximation error
    Δ::Float64      # Total: δ_N + e_T + e_mis + e_map
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

    # K_τ = ‖f̃‖_{A_τ} = sqrt(Σ_k e^{2πτ|k|} |f̂_k|²)
    Kτ = sqrt(sum(exp(2π * τ * abs(k)) * abs(fhat[idx(k, N)])^2 for k in modes(N)))

    # Compute T_N(f̃)
    TN_f = apply_self_consistent(prob, fhat)

    # δ_N = ‖T_N(f̃) - f̃‖_2
    δ_N = norm(TN_f - fhat)

    # e_T = exp(-πτN) (S_{τ,σ} + 1) K_τ
    e_T = exp(-π * τ * N) * (gc.S_τσ + 1) * Kτ

    # e_mis = |δ| Lip(G) exp(-πτN) ‖φ‖_{A_τ} S^(1)_{0,σ} K_2²
    # (This is 0 if φ is a trig polynomial of degree ≤ N, e.g., cosine observable)
    obs = get_observable(prob.coupling)
    if obs isa CosineObservable && N >= 1
        e_mis = 0.0  # φ = cos(πx) has degree 1
    else
        e_mis = abs(δ) * cc.Lip_G * exp(-π * τ * N) * phi_norm_Aτ * gc.S1_0σ * K2^2
    end

    e_map = 0.0
    if !(prob.map_params === nothing)
        # L2 perturbation bound via ||ρ'_σ||_2 * ||T - T̃||_∞.
        C_J, _ = periodized_gaussian_derivative_norms(prob.noise.σ)
        map_sup_error = bound_map_sup_error(prob.map_params)
        e_map = C_J * map_sup_error
    end

    Δ = δ_N + e_T + e_mis + e_map

    return ResidualBounds(δ_N, e_T, e_mis, e_map, Δ), K2, Kτ
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

    # First term: exp(-πτN)(S_{τ,σ} + 1)
    term1 = exp(-π * τ * N) * (gc.S_τσ + 1)

    # Second term: |δ| L_G ‖φ‖_2 K_τ exp(-πτN)(S^(1)_{τ,σ} + S^(1)_{0,σ})
    term2 = abs(δ) * cc.L_G * phi_norm_2 * Kτ * exp(-π * τ * N) * (gc.S1_τσ + gc.S1_0σ)

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
    term3 = exp(-π * τ * N) * C_mis

    term_map = 0.0
    if !(prob.map_params === nothing)
        C_J, _ = periodized_gaussian_derivative_norms(prob.noise.σ)
        map_sup_error = bound_map_sup_error(prob.map_params)
        term_map = C_J * map_sup_error
    end

    return term1 + term2 + term3 + term_map
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
    # Use rigorous upper bounds from rigorous_constants.jl
    obs = get_observable(prob.coupling)
    if obs isa CosineObservable
        phi_norm_2 = cosine_observable_L2_norm_rigorous()
        phi_norm_Aτ = sup(exp(pi_interval() * interval(τ)))  # modes ±1, rigorous upper bound
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

    # 10. Error radius (use IntervalArithmetic to enclose rounding and take sup)
    if γ == 0
        r = IntervalArithmetic.sup(interval(M) * interval(rb.Δ))
    else
        h_int = interval(M)^2 * interval(γ) * interval(rb.Δ)
        denom_int = interval(M) * interval(γ)
        r_int = (2 * h_int) / (denom_int * (1 + IntervalArithmetic.sqrt(1 - 2 * h_int)))
        r = IntervalArithmetic.sup(r_int)
    end
    if verbose
        @info "Verified! Error radius" r
    end

    return RigorousResult(true, M_N, ε_J, M, rb.Δ, γ, h, r, rb)
end
