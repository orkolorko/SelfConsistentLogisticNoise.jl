# Newton solver for fixed points
#
# Newton's method on ker(ℓ) where ℓ(f) = f̂₀ = 1:
#   (I - DT(fₖ)) hₖ = T(fₖ) - fₖ   in ker(ℓ)
#   fₖ₊₁ = fₖ + hₖ

export solve_newton

"""
    compute_DT_matrix(prob::SCProblem, fhat::Vector{ComplexF64})

Compute the full Jacobian matrix DT(f) in Fourier coordinates.

The Jacobian has the form (Lemma 14):
    DT(f) = A(c) + b ⊗ a*

where:
- A(c)_{k,m} = ρ̂_σ(k) e^{-iπkc} B_{k,m}  (linearized transfer operator)
- a = δ G'(m(f)) Φ̂  (derivative of shift w.r.t. f)
- b = B(c) f̂  where B(c)_{k,m} = (-iπk) ρ̂_σ(k) e^{-iπkc} B_{k,m}
"""
function compute_DT_matrix(prob::SCProblem, fhat::Vector{ComplexF64})
    N = prob.disc.N
    c = compute_shift(prob.coupling, fhat, N)

    # A(c)_{k,m} = ρ̂_σ(k) e^{-iπkc} B_{k,m}
    A = zeros(ComplexF64, 2N + 1, 2N + 1)
    for k in modes(N)
        k_idx = idx(k, N)
        factor = rhohat(prob.noise, k) * exp(-π * im * k * c)
        for m in modes(N)
            m_idx = idx(m, N)
            A[k_idx, m_idx] = factor * prob.B[k_idx, m_idx]
        end
    end

    # Rank-one correction: b ⊗ a*
    # a = δ G'(m(f)) Φ̂
    obs = get_observable(prob.coupling)
    m_val = compute_m(obs, fhat, N)
    Gp = derivative(prob.coupling, m_val)

    a = zeros(ComplexF64, 2N + 1)
    if obs isa CosineObservable
        # Φ(x) = cos(πx) => Φ̂_1 = Φ̂_{-1} = 1/2
        a[idx(1, N)] = 0.5 * Gp
        a[idx(-1, N)] = 0.5 * Gp
    elseif obs isa SineObservable
        # Φ(x) = sin(πx) => Φ̂_1 = -i/2, Φ̂_{-1} = i/2
        a[idx(1, N)] = -0.5im * Gp
        a[idx(-1, N)] = 0.5im * Gp
    else
        error("Observable type not supported for Newton solver")
    end

    # b = B(c) f̂  where B(c)_{k,m} = (-iπk) ρ̂_σ(k) e^{-iπkc} B_{k,m}
    Bf = prob.B * fhat
    b = zeros(ComplexF64, 2N + 1)
    for k in modes(N)
        k_idx = idx(k, N)
        factor = (-π * im * k) * rhohat(prob.noise, k) * exp(-π * im * k * c)
        b[k_idx] = factor * Bf[k_idx]
    end

    # DT = A + b ⊗ a*  (outer product)
    DT = A + b * a'

    return DT
end

"""
    solve_newton_step(prob::SCProblem, fhat::Vector{ComplexF64})

Perform one Newton step: solve (I - DT(f)) h = T(f) - f on ker(ℓ).

Returns the Newton correction h (with h₀ = 0 to stay in ker(ℓ)).
"""
function solve_newton_step(prob::SCProblem, fhat::Vector{ComplexF64})
    N = prob.disc.N

    # Compute residual: r = T(f) - f
    Tf = apply_self_consistent(prob, fhat)
    r = Tf - fhat

    # Compute Jacobian: J = I - DT(f)
    DT = compute_DT_matrix(prob, fhat)
    J = I - DT

    # Project onto ker(ℓ): remove mode k=0
    # We solve J_reduced * h_reduced = r_reduced
    k0_idx = idx(0, N)
    indices = [1:k0_idx-1; k0_idx+1:2N+1]

    J_reduced = J[indices, indices]
    r_reduced = r[indices]

    # Solve the linear system
    h_reduced = J_reduced \ r_reduced

    # Reconstruct full h with h₀ = 0
    h = zeros(ComplexF64, 2N + 1)
    h[indices] = h_reduced

    return h, norm(r)
end

"""
    solve_newton(prob::SCProblem; kwargs...)

Solve for fixed point f* such that 𝒯(f*) = f* using Newton's method.

Newton's method converges quadratically near the fixed point, but requires
a good initial guess. Consider using Picard iteration first to get close.

# Keyword Arguments
- `tol=1e-12`: convergence tolerance for residual norm.
- `maxit=50`: maximum number of Newton iterations.
- `init=:uniform`: initial condition (:uniform, :bump, :random, or a Vector{ComplexF64})
- `verbose=false`: print progress information.
- `damping=1.0`: damping factor for Newton step (1.0 = full Newton, <1 = damped).

# Returns
- `FixedPointResult` containing converged solution and diagnostics.
"""
function solve_newton(
    prob::SCProblem;
    tol::Float64=1e-12,
    maxit::Int=50,
    init::Union{Symbol, Vector{ComplexF64}}=:uniform,
    verbose::Bool=false,
    damping::Float64=1.0
)
    N = prob.disc.N

    # Initialize
    fhat = if init isa Symbol
        if init == :uniform
            uniform_initial(N)
        elseif init == :bump
            bump_initial(N)
        elseif init == :random
            random_initial(N)
        else
            error("Unknown initial condition: $init")
        end
    else
        copy(init)
    end

    # Newton iteration
    converged = false
    residual = Inf
    iter = 0

    for i in 1:maxit
        iter = i

        # Newton step
        h, res_norm = solve_newton_step(prob, fhat)
        residual = res_norm

        if verbose
            m_val = compute_m(get_observable(prob.coupling), fhat, N)
            c_val = compute_shift(prob.coupling, fhat, N)
            @info "Newton iteration $i: residual = $residual, m = $m_val, c = $c_val"
        end

        # Check convergence
        if residual < tol
            converged = true
            break
        end

        # Update with optional damping
        fhat = fhat + damping * h

        # Enforce conjugate symmetry for real density
        enforce_conjugate_symmetry!(fhat, N)
    end

    # Final values
    m_final = compute_m(get_observable(prob.coupling), fhat, N)
    c_final = compute_shift(prob.coupling, fhat, N)

    # Store parameters
    params = Dict{Symbol, Any}(
        :a => prob.map.a,
        :σ => prob.noise.σ,
        :δ => get_delta(prob.coupling),
        :N => N,
        :M => prob.disc.M,
        :method => :newton,
        :damping => damping,
        :tol => tol
    )

    if verbose
        status = converged ? "converged" : "did not converge"
        @info "Newton $status after $iter iterations (residual = $residual)"
    end

    return FixedPointResult(converged, fhat, m_final, c_final, residual, iter, params)
end

"""
    solve_hybrid(prob::SCProblem; kwargs...)

Hybrid solver: use Picard iteration to get close, then switch to Newton.

# Keyword Arguments
- `picard_tol=1e-6`: tolerance for Picard phase (switch to Newton after).
- `newton_tol=1e-12`: final tolerance for Newton phase.
- `picard_maxit=1000`: max Picard iterations.
- `newton_maxit=20`: max Newton iterations.
- `α=0.3`: Picard damping parameter.
- `init=:uniform`: initial condition.
- `verbose=false`: print progress.

# Returns
- `FixedPointResult` containing converged solution.
"""
function solve_hybrid(
    prob::SCProblem;
    picard_tol::Float64=1e-6,
    newton_tol::Float64=1e-12,
    picard_maxit::Int=1000,
    newton_maxit::Int=20,
    α::Float64=0.3,
    init::Union{Symbol, Vector{ComplexF64}}=:uniform,
    verbose::Bool=false
)
    # Phase 1: Picard iteration to get close
    if verbose
        @info "Phase 1: Picard iteration (tol = $picard_tol)"
    end

    result_picard = solve_fixed_point(
        prob;
        α=α,
        tol=picard_tol,
        maxit=picard_maxit,
        init=init,
        verbose=verbose
    )

    picard_iters = result_picard.iterations

    if !result_picard.converged && result_picard.residual > 1e-3
        @warn "Picard phase did not converge well, Newton may fail"
    end

    # Phase 2: Newton iteration for fast convergence
    if verbose
        @info "Phase 2: Newton iteration (tol = $newton_tol)"
    end

    result_newton = solve_newton(
        prob;
        tol=newton_tol,
        maxit=newton_maxit,
        init=result_picard.fhat,
        verbose=verbose
    )

    # Combine iteration counts
    total_iters = picard_iters + result_newton.iterations

    # Update params
    params = Dict{Symbol, Any}(
        :a => prob.map.a,
        :σ => prob.noise.σ,
        :δ => get_delta(prob.coupling),
        :N => prob.disc.N,
        :M => prob.disc.M,
        :method => :hybrid,
        :picard_iters => picard_iters,
        :newton_iters => result_newton.iterations,
        :tol => newton_tol
    )

    return FixedPointResult(
        result_newton.converged,
        result_newton.fhat,
        result_newton.m,
        result_newton.c,
        result_newton.residual,
        total_iters,
        params
    )
end

export solve_hybrid
