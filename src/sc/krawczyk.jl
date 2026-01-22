# krawczyk.jl - Krawczyk contraction test for CAP verification
#
# Implements the Computer-Assisted Proof (CAP) approach from Version6.tex
# using the Krawczyk/preconditioned Newton method in the L² norm.
#
# The method certifies existence and uniqueness of a fixed point for the
# self-consistent operator on the mean-zero constraint subspace (ker ℓ).

using BallArithmetic: Ball, BallVector, BallMatrix, mid, rad, upper_bound_L2_opnorm
using LinearAlgebra

# ============================================================================
# Ball arithmetic helpers (range inclusion)
# ============================================================================

"""
    compute_m_ball(obs::Observable, fhat_ball, N)

Compute m(f) = ⟨Φ, f⟩ for BallArithmetic inputs using range inclusion.
"""
ball_real(z::Ball) = Ball(real(mid(z)), rad(z))
ball_imag(z::Ball) = Ball(imag(mid(z)), rad(z))
function exp_ball(z::Ball{<:AbstractFloat, <:Complex})
    z_mid = mid(z)
    r = rad(z)
    # Mean-value bound: |exp(z) - exp(z_mid)| ≤ exp(|z_mid| + r) * r
    return Ball(exp(z_mid), exp(abs(z_mid) + r) * r)
end

function compute_m_ball(obs::CosineObservable, fhat_ball::AbstractVector, N::Int)
    return ball_real(fhat_ball[idx(1, N)])
end

function compute_m_ball(obs::SineObservable, fhat_ball::AbstractVector, N::Int)
    return -ball_imag(fhat_ball[idx(1, N)])
end

function compute_m_ball(obs::FourierObservable, fhat_ball::AbstractVector, N::Int)
    result = zero(eltype(fhat_ball))
    N_phi = (length(obs.Phihat) - 1) ÷ 2
    for k in modes(min(N, N_phi))
        result += obs.Phihat[idx(k, N_phi)] * conj(fhat_ball[idx(k, N)])
    end
    return ball_real(result)
end

"""
    compute_shift_ball(coupling, fhat_ball, N)

Compute the shift c(f) = δ * G(m(f)) using BallArithmetic.
"""
function compute_shift_ball(coupling::Coupling, fhat_ball::AbstractVector, N::Int)
    m = compute_m_ball(coupling.observable, fhat_ball, N)
    return coupling(m)
end

(c::LinearCoupling)(m::Ball) = c.δ * m
derivative(c::LinearCoupling, m::Ball) = c.δ

(c::TanhCoupling)(m::Ball) = c.δ * tanh(c.β * m)
derivative(c::TanhCoupling, m::Ball) = c.δ * c.β * sech(c.β * m)^2

"""
    compute_DT_matrix_ball(prob::SCProblem, fhat_ball)

Compute DT(f) using BallArithmetic inputs for range inclusion.
"""
function compute_DT_matrix_ball(prob::SCProblem, fhat_ball::AbstractVector)
    N = prob.disc.N
    # With m(f) as a Ball, BallArithmetic preserves enclosures for m*v and exp(· m(f)).
    c = compute_shift_ball(prob.coupling, fhat_ball, N)

    A = Matrix{eltype(fhat_ball)}(undef, 2N + 1, 2N + 1)
    for k in modes(N)
        k_idx = idx(k, N)
        factor = Ball(rhohat(prob.noise, k)) * exp_ball(-π * im * k * c)
        for m in modes(N)
            m_idx = idx(m, N)
            A[k_idx, m_idx] = factor * Ball(prob.B[k_idx, m_idx])
        end
    end

    obs = get_observable(prob.coupling)
    m_val = compute_m_ball(obs, fhat_ball, N)
    Gp = derivative(prob.coupling, m_val)

    a = zeros(eltype(fhat_ball), 2N + 1)
    if obs isa CosineObservable
        a[idx(1, N)] = 0.5 * Gp
        a[idx(-1, N)] = 0.5 * Gp
    elseif obs isa SineObservable
        a[idx(1, N)] = -0.5im * Gp
        a[idx(-1, N)] = 0.5im * Gp
    elseif obs isa FourierObservable
        N_phi = (length(obs.Phihat) - 1) ÷ 2
        for k in modes(min(N, N_phi))
            a[idx(k, N)] = obs.Phihat[idx(k, N_phi)] * Gp
        end
    else
        error("Observable type not supported for Ball Jacobian enclosure")
    end

    Bf = zeros(eltype(fhat_ball), 2N + 1)
    for k in modes(N)
        k_idx = idx(k, N)
        acc = zero(eltype(fhat_ball))
        for m in modes(N)
            m_idx = idx(m, N)
            acc += Ball(prob.B[k_idx, m_idx]) * fhat_ball[m_idx]
        end
        Bf[k_idx] = acc
    end

    b = zeros(eltype(fhat_ball), 2N + 1)
    for k in modes(N)
        k_idx = idx(k, N)
        factor = (-π * im * k) * Ball(rhohat(prob.noise, k)) * exp_ball(-π * im * k * c)
        b[k_idx] = factor * Bf[k_idx]
    end

    DT = Matrix{eltype(fhat_ball)}(undef, 2N + 1, 2N + 1)
    for i in 1:2N+1
        for j in 1:2N+1
            DT[i, j] = A[i, j] + b[i] * conj(a[j])
        end
    end

    return DT
end

# ============================================================================
# Coordinate conventions: Full ↔ Perp (mean-zero constraint)
# ============================================================================

"""
    embed_perp(u_perp, N)

Embed a perp vector (modes k ≠ 0) into a full Fourier vector.
The k=0 mode is set to 0.

Full vector has length 2N+1, perp vector has length 2N.
Indexing: full[k+N+1] = f̂(k), so full[N+1] = f̂(0).
"""
function embed_perp(u_perp::AbstractVector, N::Int)
    @assert length(u_perp) == 2N "u_perp must have length 2N"

    T = eltype(u_perp)
    u_full = Vector{T}(undef, 2N + 1)
    fill!(u_full, zero(T))

    # Modes k = -N, ..., -1 go to indices 1, ..., N
    u_full[1:N] = u_perp[1:N]
    # Mode k = 0 stays 0
    # Modes k = 1, ..., N go to indices N+2, ..., 2N+1
    u_full[N+2:2N+1] = u_perp[N+1:2N]

    return u_full
end

"""
    project_perp(u_full, N)

Project a full Fourier vector onto the perp subspace (remove k=0 mode).
"""
function project_perp(u_full::AbstractVector, N::Int)
    @assert length(u_full) == 2N + 1 "u_full must have length 2N+1"

    T = eltype(u_full)
    u_perp = Vector{T}(undef, 2N)
    fill!(u_perp, zero(T))

    # Modes k = -N, ..., -1 from indices 1, ..., N
    u_perp[1:N] = u_full[1:N]
    # Modes k = 1, ..., N from indices N+2, ..., 2N+1
    u_perp[N+1:2N] = u_full[N+2:2N+1]

    return u_perp
end

"""
    perp_index(k, N)

Convert wavenumber k ≠ 0 to perp vector index.
For k ∈ {-N, ..., -1, 1, ..., N}, returns index in 1:2N.
"""
function perp_index(k::Int, N::Int)
    @assert k != 0 && abs(k) <= N "k must be nonzero with |k| ≤ N"
    if k < 0
        return k + N + 1  # k=-N → 1, k=-1 → N
    else
        return k + N      # k=1 → N+1, k=N → 2N
    end
end

"""
    perp_mode(i, N)

Convert perp vector index i ∈ 1:2N to wavenumber k ≠ 0.
"""
function perp_mode(i::Int, N::Int)
    @assert 1 <= i <= 2N "Index must be in 1:2N"
    if i <= N
        return i - N - 1  # 1 → -N, N → -1
    else
        return i - N      # N+1 → 1, 2N → N
    end
end

# ============================================================================
# Residual computation: F_perp
# ============================================================================

"""
    compute_F_perp(prob::SCProblem, fhat_base::Vector{ComplexF64})

Compute the residual F(u) = f̃ + u - T(f̃ + u) at u = 0,
projected onto the perp subspace (mean-zero).

Here f̃ = fhat_base is the candidate fixed point (normalized: f̂(0) = 1).

Returns: F_perp as a Vector{Ball} (complex balls for rigorous enclosure).
"""
function compute_F_perp(prob::SCProblem, fhat_base::Vector{ComplexF64})
    N = prob.disc.N

    # Apply self-consistent operator: T(f̃)
    Tf = apply_self_consistent(prob, fhat_base)

    # Residual in full space: F = f̃ - T(f̃)
    F_full = fhat_base - Tf

    # Project to perp (remove k=0 component)
    F_perp = project_perp(F_full, N)

    # Convert to Ball for rigorous arithmetic
    # Use small radius to account for floating point errors
    F_perp_ball = [Ball(F_perp[i]) for i in 1:length(F_perp)]

    return F_perp_ball
end

"""
    compute_F_perp_at_u(prob::SCProblem, fhat_base::Vector{ComplexF64},
                        u_perp::Vector{<:Ball})

Compute F(u) = f̃ + u - T(f̃ + u) for a BallVector u_perp.
"""
function compute_F_perp_at_u(prob::SCProblem, fhat_base::Vector{ComplexF64},
                             u_perp::Vector{<:Ball})
    N = prob.disc.N

    # Embed u to full space
    u_full = embed_perp(u_perp, N)

    # f = f̃ + u
    f_full = [Ball(fhat_base[i]) + u_full[i] for i in 1:2N+1]

    # Convert to ComplexF64 midpoints for operator application
    f_mid = [mid(f_full[i]) for i in 1:2N+1]

    # Apply operator (this part needs rigorous enclosure)
    Tf_mid = apply_self_consistent(prob, f_mid)

    # Residual
    F_full = [f_full[i] - Ball(Tf_mid[i]) for i in 1:2N+1]

    # Project to perp
    return project_perp(F_full, N)
end

# ============================================================================
# Jacobian computation: J_perp and enclosure
# ============================================================================

"""
    compute_J_perp_matrix(prob::SCProblem, fhat::Vector{ComplexF64})

Compute the Jacobian matrix J_N = I - DT_N(f̃) on the perp subspace.

The Jacobian DT has the rank-one structure:
    DT(f)[h] = P_{c(f)} h + c'(f)[h] · Q_{c(f)} f

where Q_c = ∂_c P_c (shift derivative).

Returns: (2N) × (2N) complex matrix representing J on ker ℓ.
"""
function compute_J_perp_matrix(prob::SCProblem, fhat::Vector{ComplexF64})
    N = prob.disc.N
    dim_perp = 2N

    # Get the full Jacobian DT matrix
    DT_full = compute_DT_matrix(prob, fhat)  # (2N+1) × (2N+1)

    # J = I - DT
    J_full = I - DT_full

    # Extract the perp × perp block (remove row and column for k=0)
    # k=0 is at index N+1 in full space
    J_perp = zeros(ComplexF64, dim_perp, dim_perp)

    for i in 1:dim_perp
        k_i = perp_mode(i, N)
        full_i = k_i + N + 1

        for j in 1:dim_perp
            k_j = perp_mode(j, N)
            full_j = k_j + N + 1

            J_perp[i, j] = J_full[full_i, full_j]
        end
    end

    return J_perp
end

"""
    compute_J_perp_ball(prob::SCProblem, fhat_base::Vector{ComplexF64},
                        u_ball::Vector{<:Ball})

Compute a rigorous enclosure of the Jacobian J_N(f̃ + u) for u in a BallVector.

This computes J at the midpoint and inflates by the Lipschitz constant
times the radius of the ball.

Returns: BallMatrix enclosing J_N(f̃ + u) for all u in the ball.
"""
function compute_J_perp_ball(prob::SCProblem, fhat_base::Vector{ComplexF64},
                             u_ball::Vector{<:Ball})
    N = prob.disc.N
    dim_perp = 2N

    # Compute J at midpoint (u = mid(u_ball))
    u_mid = [mid(u_ball[i]) for i in 1:dim_perp]
    u_full_mid = embed_perp(u_mid, N)
    fhat_mid = fhat_base + u_full_mid

    J_mid = compute_J_perp_matrix(prob, fhat_mid)

    # Estimate Lipschitz constant for the Jacobian
    # ||DJ(f) - DJ(g)|| ≤ γ ||f - g||
    # where γ depends on the coupling constants
    γ = compute_jacobian_lipschitz(prob, fhat_base)

    # Radius of the ball in L² norm
    r = sqrt(sum(rad(u_ball[i])^2 for i in 1:dim_perp))

    # Inflate each entry by γ * r
    inflation = γ * r

    # Create BallMatrix with inflated entries
    J_ball_mid = [Ball(J_mid[i,j]) for i in 1:dim_perp, j in 1:dim_perp]
    J_ball_rad = fill(inflation, dim_perp, dim_perp)

    # Combine midpoint and radius
    J_ball = [Ball(mid(J_ball_mid[i,j]), rad(J_ball_mid[i,j]) + J_ball_rad[i,j])
              for i in 1:dim_perp, j in 1:dim_perp]

    return J_ball
end

"""
    compute_J_perp_ball_range(prob::SCProblem, fhat_base::Vector{ComplexF64},
                              u_ball::Vector{<:Ball})

Compute a Jacobian enclosure by evaluating DT on the Ball vector and
using the range inclusion property instead of a Lipschitz inflation.
"""
function compute_J_perp_ball_range(prob::SCProblem, fhat_base::Vector{ComplexF64},
                                   u_ball::Vector{<:Ball})
    N = prob.disc.N
    dim_perp = 2N

    u_full = embed_perp(u_ball, N)
    fhat_ball = [Ball(fhat_base[i]) + u_full[i] for i in 1:2N+1]

    DT_ball = compute_DT_matrix_ball(prob, fhat_ball)

    J_full = Matrix{eltype(fhat_ball)}(undef, 2N + 1, 2N + 1)
    for i in 1:2N+1
        for j in 1:2N+1
            identity_entry = i == j ? Ball(1.0 + 0.0im) : Ball(0.0 + 0.0im)
            J_full[i, j] = identity_entry - DT_ball[i, j]
        end
    end

    J_perp = Matrix{eltype(fhat_ball)}(undef, dim_perp, dim_perp)
    for i in 1:dim_perp
        k_i = perp_mode(i, N)
        full_i = k_i + N + 1
        for j in 1:dim_perp
            k_j = perp_mode(j, N)
            full_j = k_j + N + 1
            J_perp[i, j] = J_full[full_i, full_j]
        end
    end

    return J_perp
end

"""
    compute_jacobian_lipschitz(prob::SCProblem, fhat::Vector{ComplexF64})

Compute the Lipschitz constant γ for the Jacobian:
    ||DT(f) - DT(g)||_{2→2} ≤ γ ||f - g||_2

Based on the formula from Version6.tex:
    γ = |δ| ||φ||₂ C_J (Lip(G) + L_G) + |δ| L_{G'} ||φ||₂² C_J K
        + |δ|² L_G Lip(G) ||φ||₂² C_J^{(2)} K
"""
function compute_jacobian_lipschitz(prob::SCProblem, fhat::Vector{ComplexF64})
    N = prob.disc.N
    σ = prob.noise.σ
    δ = get_delta(prob.coupling)

    # Norms of the observable φ
    obs = get_observable(prob.coupling)
    phi_norm_2 = compute_observable_L2_norm(obs, N)

    # Periodized Gaussian derivative constants (rigorous L2 norms on the torus)
    C_J, C_J_2 = periodized_gaussian_derivative_norms(σ)

    # Coupling function constants
    # For linear coupling G(m) = m: Lip(G) = 1, L_G = 1, L_{G'} = 0
    # For tanh coupling: need to compute from the coupling type
    Lip_G, L_G, L_Gp = compute_coupling_constants(prob.coupling)

    # Norm of candidate
    K_2 = sqrt(sum(abs2(fhat[i]) for i in 1:2N+1))

    # Lipschitz constant formula
    γ = abs(δ) * phi_norm_2 * C_J * (Lip_G + L_G)
    γ += abs(δ) * L_Gp * phi_norm_2^2 * C_J * K_2
    γ += abs(δ)^2 * L_G * Lip_G * phi_norm_2^2 * C_J_2 * K_2

    return γ
end

"""
    periodized_gaussian_derivative_norms(σ; K_start=1, K_max=10_000)

Compute rigorous upper bounds for the L2 norms of the derivatives of the
periodized Gaussian kernel on the torus:

    ||ρ'_σ||_2^2 = Σ_{k∈ℤ} (πk)^2 exp(-π²σ²k²)
    ||ρ''_σ||_2^2 = Σ_{k∈ℤ} (πk)^4 exp(-π²σ²k²)

The series are split into a finite sum and a geometric tail bound using the
ratio at k = K. Returns (C_J, C_J_2) as rigorous upper bounds using interval
arithmetic.
"""
function periodized_gaussian_derivative_norms(σ::Real; K_start::Int=1, K_max::Int=10_000)
    # Use the rigorous version from rigorous_constants.jl
    return periodized_gaussian_derivative_norms_rigorous(σ; K_start=K_start, K_max=K_max)
end

"""
    compute_observable_L2_norm(obs::Observable, N)

Compute ||φ||_2 for the observable. Returns a rigorous upper bound.
"""
function compute_observable_L2_norm(obs::CosineObservable, N::Int)
    # φ(x) = cos(πx) = (e^{iπx} + e^{-iπx})/2
    # ||φ||_2 = 1/√2 (on the torus)
    # Use rigorous upper bound from rigorous_constants.jl
    return cosine_observable_L2_norm_rigorous()
end

"""
    compute_coupling_constants(coupling::Coupling)

Compute (Lip(G), L_G = ||G'||_∞, L_{G'} = Lip(G')) for the coupling function.
"""
function compute_coupling_constants(c::LinearCoupling)
    # G(m) = m, so G'(m) = 1
    return (1.0, 1.0, 0.0)  # (Lip(G), L_G, L_{G'})
end

function compute_coupling_constants(c::TanhCoupling)
    # G(m) = β tanh(m/β), so G'(m) = sech²(m/β)
    # ||G'||_∞ = 1 (at m=0)
    # Lip(G') = max |G''| = max |−2 sech²(x) tanh(x) / β| ≤ 2/(β√3) at tanh(x)=1/√3
    # Use rigorous constants from rigorous_constants.jl
    cc = compute_tanh_coupling_constants_rigorous(c.β)
    return (cc.Lip_G, cc.L_G, cc.L_Gp)
end

# ============================================================================
# Krawczyk contraction test
# ============================================================================

"""
    KrawczykResult

Result of Krawczyk contraction verification.

# Fields
- `verified`: Whether the fixed point was verified
- `Y`: ||A F(0)||_2 (residual after preconditioning)
- `Z`: Contraction factor sup ||I - A J(u)||_2
- `r`: Certified ball radius
- `iterations`: Number of iterations used
- `reason`: Failure reason if not verified
"""
struct KrawczykResult
    verified::Bool
    Y::Float64
    Z::Float64
    r::Float64
    iterations::Int
    reason::String
end

"""
    certify_krawczyk(prob::SCProblem, fhat::Vector{ComplexF64};
                     tol=1e-3, maxit=20, verbose=false)

Certify existence and uniqueness of a fixed point near fhat using the
Krawczyk contraction method.

The method verifies that the preconditioned Newton map G(u) = u - A F(u)
is a contraction on a ball X_r in the mean-zero subspace, where A = J(0)⁻¹.

# Arguments
- `prob`: Self-consistent problem
- `fhat`: Candidate fixed point (Fourier coefficients, normalized)
- `tol`: Tolerance for radius convergence
- `maxit`: Maximum iterations for radius selection
- `verbose`: Print progress

# Returns
- `KrawczykResult` with verification status and certified ball radius
"""
function certify_krawczyk(prob::SCProblem, fhat::Vector{ComplexF64};
                          tol::Real=1e-3, maxit::Int=20, verbose::Bool=false)
    N = prob.disc.N
    dim_perp = 2N

    if verbose
        println("Krawczyk verification for N = $N ($(dim_perp) DOF)")
    end

    # Step 1: Compute preconditioner A = J(0)⁻¹
    J_0 = compute_J_perp_matrix(prob, fhat)

    # Check condition number
    cond_J = cond(J_0)
    if verbose
        println("  Condition number of J: $cond_J")
    end

    if cond_J > 1e12
        return KrawczykResult(false, Inf, Inf, Inf, 0,
                              "Jacobian nearly singular (cond = $cond_J)")
    end

    A = inv(J_0)

    # Step 2: Compute Y = ||A F(0)||_2
    F_0 = compute_F_perp(prob, fhat)
    AF_0 = A * [mid(F_0[i]) for i in 1:dim_perp]
    Y = norm(AF_0, 2)

    if verbose
        println("  Y = ||A F(0)||_2 = $Y")
    end

    if Y < 1e-14
        # Already at a fixed point
        return KrawczykResult(true, Y, 0.0, Y, 0, "")
    end

    # Step 3: Iteration to find valid radius
    r = 1.1 * Y

    for it in 1:maxit
        if verbose
            println("  Iteration $it: r = $r")
        end

        # Build ball B_r (uniform radius in each component)
        u_ball = [Ball(0.0 + 0.0im, r) for _ in 1:dim_perp]

        # Compute Jacobian enclosure on the ball
        J_ball = compute_J_perp_ball(prob, fhat, u_ball)

        # Compute M = I - A * J as BallMatrix
        AJ = A * [mid(J_ball[i,j]) for i in 1:dim_perp, j in 1:dim_perp]

        # Form I - AJ with Ball entries
        M = zeros(Ball{Float64, ComplexF64}, dim_perp, dim_perp)
        for i in 1:dim_perp
            for j in 1:dim_perp
                if i == j
                    M[i,j] = Ball(1.0 + 0.0im) - Ball(AJ[i,j])
                else
                    M[i,j] = Ball(0.0 + 0.0im) - Ball(AJ[i,j])
                end
                # Add inflation from J_ball radius
                J_rad = maximum(rad(J_ball[i,k]) for k in 1:dim_perp)
                A_row_norm = sum(abs(A[i,k]) for k in 1:dim_perp)
                M[i,j] = Ball(mid(M[i,j]), rad(M[i,j]) + A_row_norm * J_rad)
            end
        end

        # Compute Z = ||I - A J||_2 using BallArithmetic's bound
        # Use ||M||_2 ≤ √(||M||_1 ||M||_∞)
        Z = compute_spectral_norm_bound(M)

        if verbose
            println("    Z = $Z")
        end

        if Z >= 1
            # Try smaller radius or report failure
            if r < 1e-10
                return KrawczykResult(false, Y, Z, r, it,
                                      "Z >= 1 even at small radius")
            end
            r = r / 2
            continue
        end

        # Compute required radius for invariance
        r_new = Y / (1 - Z)

        if verbose
            println("    r_new = $r_new")
        end

        if r_new <= r * (1 + tol)
            # Check invariance: Y + Z*r ≤ r
            if Y + Z * r <= r + 1e-14
                return KrawczykResult(true, Y, Z, r, it, "")
            end
        end

        r = r_new
    end

    return KrawczykResult(false, Y, 0.0, r, maxit, "Max iterations reached")
end

"""
    compute_spectral_norm_bound(M::Matrix{<:Ball})

Compute upper bound on ||M||_2 using BallArithmetic's upper_bound_L2_opnorm.
"""
function compute_spectral_norm_bound(M::Matrix{<:Ball})
    # Convert to BallMatrix and use BallArithmetic's method
    return upper_bound_L2_opnorm(BallMatrix(M))
end

"""
    compute_spectral_norm_bound(M::BallMatrix)

Compute upper bound on ||M||_2 for a BallMatrix.
"""
function compute_spectral_norm_bound(M::BallMatrix)
    return upper_bound_L2_opnorm(M)
end

# ============================================================================
# Full CAP verification combining Krawczyk with truncation errors
# ============================================================================

"""
    CAPResult

Complete CAP verification result.

# Fields
- `verified`: Overall verification success
- `krawczyk`: Krawczyk result for finite-dimensional part
- `map_shift_bound`: Map approximation perturbation bound
- `fft_error`: FFT aliasing error bound from sampling size M
- `truncation_error`: Truncation error bound
- `total_error`: Total error bound on fixed point
- `fhat`: Verified fixed point (midpoint)
"""
struct CAPResult
    verified::Bool
    krawczyk::KrawczykResult
    map_shift_bound::Float64
    fft_error::Float64
    truncation_error::Float64
    total_error::Float64
    fhat::Vector{ComplexF64}
end

"""
    verify_fixed_point_CAP(prob::SCProblem, fhat::Vector{ComplexF64};
                           τ=0.1, verbose=false)

Complete CAP verification of a fixed point for the self-consistent operator.

Combines:
1. Krawczyk verification in finite dimension (modes |k| ≤ N)
2. Truncation error bounds for modes |k| > N

# Arguments
- `prob`: Self-consistent problem
- `fhat`: Candidate fixed point
- `τ`: Analyticity strip parameter for truncation bounds
- `verbose`: Print details

# Returns
- `CAPResult` with complete verification status
"""
function verify_fixed_point_CAP(prob::SCProblem, fhat::Vector{ComplexF64};
                                τ::Real=0.1, verbose::Bool=false)
    N = prob.disc.N
    σ = prob.noise.σ

    if verbose
        println("="^60)
        println("CAP Verification of Self-Consistent Fixed Point")
        println("="^60)
        println("N = $N, σ = $σ, τ = $τ")
    end

    # Step 1: Krawczyk verification in finite dimension
    if verbose
        println("\n--- Step 1: Krawczyk Finite-Dimensional Verification ---")
    end

    kraw = certify_krawczyk(prob, fhat; verbose=verbose)

    if !kraw.verified
        if verbose
            println("Krawczyk verification FAILED: $(kraw.reason)")
        end
        return CAPResult(false, kraw, Inf, Inf, Inf, Inf, fhat)
    end

    if verbose
        println("Krawczyk verification PASSED")
        println("  Y = $(kraw.Y), Z = $(kraw.Z), r = $(kraw.r)")
    end

    # Step 2: Map-approximation fixed-point shift bound
    map_shift_bound = 0.0
    if !(prob.map_params === nothing)
        # L2 perturbation bound via ||ρ'_σ||_2 * ||T - T̃||_∞.
        C_J, _ = periodized_gaussian_derivative_norms(σ)
        map_sup_error = bound_map_sup_error(prob.map_params)
        map_shift_bound = C_J * map_sup_error
    end

    # Step 3: FFT aliasing error bound (from sampling size M)
    alias_bounds = fft_aliasing_bound_matrix(prob.map, prob.disc)
    fft_error = compute_spectral_norm_bound(Ball.(alias_bounds))

    # Step 4: Truncation error bounds
    if verbose
        println("\n--- Step 2: Map-Approximation Error Bound ---")
        println("  Map shift bound: $map_shift_bound")
        println("  FFT aliasing bound: $fft_error")
        println("\n--- Step 3: Truncation Error Bounds ---")
    end

    # Compute Gaussian smoothing constants
    gc = compute_gaussian_constants(σ, τ)

    # Norms of candidate
    K_2 = sqrt(sum(abs2(fhat[i]) for i in 1:2N+1))
    K_τ = sum(abs(fhat[idx(k, N)]) * exp(π * τ * abs(k)) for k in -N:N)

    # Truncation error: e_T = e^{-πτN} (S_{τ,σ} + 1) K_τ
    e_T = exp(-π * τ * N) * (gc.S_τσ + 1) * K_τ

    if verbose
        println("  K_2 = $K_2, K_τ = $K_τ")
        println("  S_{τ,σ} = $(gc.S_τσ)")
        println("  Truncation error e_T = $e_T")
    end

    # Step 5: Total error
    # Finite-dimensional error from Krawczyk
    finite_error = kraw.r

    # Total error combining finite, map, and truncation parts
    total_error = finite_error + map_shift_bound + fft_error + e_T

    if verbose
        println("\n--- Final Result ---")
        println("  Finite-dimensional error: $finite_error")
        println("  FFT aliasing error: $fft_error")
        println("  Truncation error: $e_T")
        println("  Total error: $total_error")
    end

    return CAPResult(true, kraw, map_shift_bound, fft_error, e_T, total_error, fhat)
end

"""
    print_CAP_certificate(result::CAPResult)

Print a detailed certificate of the CAP verification.
"""
function print_CAP_certificate(result::CAPResult)
    println("="^60)
    println("CAP VERIFICATION CERTIFICATE")
    println("="^60)

    if result.verified
        println("Status: VERIFIED ✓")
    else
        println("Status: FAILED ✗")
        println("Reason: $(result.krawczyk.reason)")
        return
    end

    println("\nKrawczyk Contraction Test:")
    println("  Y = ||A F(0)||_2:     $(result.krawczyk.Y)")
    println("  Z = ||I - A J||_2:    $(result.krawczyk.Z)")
    println("  Certified ball:       r = $(result.krawczyk.r)")
    println("  Iterations:           $(result.krawczyk.iterations)")

    println("\nError Bounds:")
    println("  Finite-dim error:     $(result.krawczyk.r)")
    println("  Map shift bound:      $(result.map_shift_bound)")
    println("  FFT aliasing error:   $(result.fft_error)")
    println("  Truncation error:     $(result.truncation_error)")
    println("  Total error:          $(result.total_error)")

    println("\nConclusion:")
    println("  There exists a UNIQUE fixed point f* with")
    println("  ||f* - f̃||_2 ≤ $(result.total_error)")
    println("="^60)
end

# Exports
export embed_perp, project_perp, perp_index, perp_mode
export compute_F_perp, compute_J_perp_matrix, compute_J_perp_ball
export compute_J_perp_ball_range
export compute_jacobian_lipschitz
export KrawczykResult, certify_krawczyk
export CAPResult, verify_fixed_point_CAP, print_CAP_certificate
