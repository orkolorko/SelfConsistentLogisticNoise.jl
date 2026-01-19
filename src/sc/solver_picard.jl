# Picard iteration solver for fixed points

"""
    FixedPointResult

Result of a fixed-point computation.
"""
struct FixedPointResult
    converged::Bool
    fhat::Vector{ComplexF64}
    m::Float64          # Observable value m(f*)
    c::Float64          # Shift c(f*)
    residual::Float64   # Final residual norm
    iterations::Int     # Number of iterations
    params::Dict{Symbol, Any}  # Problem parameters
end

"""
    uniform_initial(N)

Create uniform density initial condition: f̂_0 = 1, f̂_k = 0 for k ≠ 0.
"""
function uniform_initial(N::Int)
    fhat = zeros(ComplexF64, 2N + 1)
    fhat[idx(0, N)] = 1.0 + 0.0im
    return fhat
end

"""
    bump_initial(N; amplitude=0.1)

Create low-mode bump initial condition.
"""
function bump_initial(N::Int; amplitude::Float64=0.1)
    fhat = zeros(ComplexF64, 2N + 1)
    fhat[idx(0, N)] = 1.0 + 0.0im
    # Add small perturbation to mode 1 (and -1 for conjugate symmetry)
    fhat[idx(1, N)] = amplitude + 0.0im
    fhat[idx(-1, N)] = amplitude + 0.0im
    return fhat
end

"""
    random_initial(N; amplitude=0.1, seed=nothing)

Create random low-mode initial condition with conjugate symmetry.
"""
function random_initial(N::Int; amplitude::Float64=0.1, seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    fhat = zeros(ComplexF64, 2N + 1)
    fhat[idx(0, N)] = 1.0 + 0.0im

    # Add random perturbations to low modes (up to mode 10)
    max_mode = min(10, N)
    for k in 1:max_mode
        z = amplitude * (randn() + im * randn()) / sqrt(2)
        fhat[idx(k, N)] = z
        fhat[idx(-k, N)] = conj(z)  # Conjugate symmetry
    end

    return fhat
end

"""
    enforce_normalization!(fhat, N)

Enforce that f̂_0 = 1 (density integrates to 1).
"""
function enforce_normalization!(fhat::Vector{ComplexF64}, N::Int)
    fhat[idx(0, N)] = 1.0 + 0.0im
    return fhat
end

"""
    enforce_conjugate_symmetry!(fhat, N)

Enforce conjugate symmetry: f̂_{-k} = conj(f̂_k) for real-valued densities.
"""
function enforce_conjugate_symmetry!(fhat::Vector{ComplexF64}, N::Int)
    fhat[idx(0, N)] = real(fhat[idx(0, N)]) + 0.0im
    for k in 1:N
        avg = (fhat[idx(k, N)] + conj(fhat[idx(-k, N)])) / 2
        fhat[idx(k, N)] = avg
        fhat[idx(-k, N)] = conj(avg)
    end
    return fhat
end

"""
    solve_fixed_point(prob::SCProblem; kwargs...)

Solve for fixed point f* such that 𝒯(f*) = f* using damped Picard iteration.

# Keyword Arguments
- `α=0.2`: damping parameter (0 < α ≤ 1). Smaller values give more stability.
- `tol=1e-12`: convergence tolerance for relative residual.
- `maxit=5000`: maximum number of iterations.
- `init=:uniform`: initial condition (:uniform, :bump, :random, or a Vector{ComplexF64})
- `verbose=false`: print progress information.
- `enforce_symmetry=true`: enforce conjugate symmetry at each step.

# Returns
- `FixedPointResult` containing converged solution and diagnostics.
"""
function solve_fixed_point(
    prob::SCProblem;
    α::Float64=0.2,
    tol::Float64=1e-12,
    maxit::Int=5000,
    init::Union{Symbol, Vector{ComplexF64}}=:uniform,
    verbose::Bool=false,
    enforce_symmetry::Bool=true
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

    fhat_new = similar(fhat)

    # Iteration
    converged = false
    residual = Inf
    iter = 0

    for i in 1:maxit
        iter = i

        # Compute shift
        c = compute_shift(prob.coupling, fhat, N)

        # Apply operator
        fhat_new = apply_Pc(prob, fhat, c)

        # Enforce normalization
        enforce_normalization!(fhat_new, N)

        # Enforce conjugate symmetry if requested
        if enforce_symmetry
            enforce_conjugate_symmetry!(fhat_new, N)
        end

        # Compute residual
        diff = fhat_new - fhat
        residual = norm(diff) / norm(fhat)

        if verbose && (i % 100 == 0 || i == 1)
            m_val = compute_m(get_observable(prob.coupling), fhat, N)
            @info "Iteration $i: residual = $residual, m = $m_val, c = $c"
        end

        # Check convergence
        if residual < tol
            converged = true
            fhat = fhat_new
            break
        end

        # Damped update
        fhat = (1 - α) * fhat + α * fhat_new
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
        :α => α,
        :tol => tol
    )

    if verbose
        status = converged ? "converged" : "did not converge"
        @info "Solver $status after $iter iterations (residual = $residual)"
    end

    return FixedPointResult(converged, fhat, m_final, c_final, residual, iter, params)
end
