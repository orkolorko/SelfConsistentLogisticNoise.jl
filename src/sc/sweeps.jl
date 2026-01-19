# Parameter sweep utilities

"""
    sweep_delta(prob::SCProblem, δ_range; warm_start=true, kwargs...)

Sweep over δ values, optionally using warm-start continuation.

Returns a DataFrame with columns: δ, m, c, residual, iterations, converged
"""
function sweep_delta(
    prob::SCProblem,
    δ_range::AbstractVector{Float64};
    warm_start::Bool=true,
    verbose::Bool=false,
    kwargs...
)
    N = prob.disc.N

    results = DataFrame(
        δ = Float64[],
        m = Float64[],
        c = Float64[],
        residual = Float64[],
        iterations = Int[],
        converged = Bool[]
    )

    # Initial condition
    init = :uniform

    for (i, δ) in enumerate(δ_range)
        # Update coupling strength
        prob.coupling = update_delta(prob.coupling, δ)

        # Solve
        result = solve_fixed_point(prob; init=init, verbose=verbose, kwargs...)

        # Store results
        push!(results, (
            δ = δ,
            m = result.m,
            c = result.c,
            residual = result.residual,
            iterations = result.iterations,
            converged = result.converged
        ))

        # Warm start: use converged solution as next initial condition
        if warm_start && result.converged
            init = result.fhat
        end

        if verbose
            @info "δ = $δ: m = $(result.m), converged = $(result.converged)"
        end
    end

    return results
end

"""
    sweep_delta_a(; a_range, δ_range, σ=0.02, N=256, kwargs...)

Sweep over both a and δ values.

Returns a DataFrame with columns: a, δ, m, c, residual, iterations, converged
"""
function sweep_delta_a(;
    a_range::AbstractVector{Float64},
    δ_range::AbstractVector{Float64},
    σ::Float64=0.02,
    N::Int=256,
    warm_start::Bool=true,
    verbose::Bool=false,
    kwargs...
)
    results = DataFrame(
        a = Float64[],
        δ = Float64[],
        m = Float64[],
        c = Float64[],
        residual = Float64[],
        iterations = Int[],
        converged = Bool[]
    )

    for a in a_range
        if verbose
            @info "Processing a = $a"
        end

        # Build problem for this a
        prob = build_problem(a=a, σ=σ, N=N, δ=δ_range[1]; kwargs...)

        # Initial condition
        init = :uniform

        for δ in δ_range
            # Update coupling
            prob.coupling = update_delta(prob.coupling, δ)

            # Solve
            result = solve_fixed_point(prob; init=init, verbose=false)

            # Store
            push!(results, (
                a = a,
                δ = δ,
                m = result.m,
                c = result.c,
                residual = result.residual,
                iterations = result.iterations,
                converged = result.converged
            ))

            # Warm start
            if warm_start && result.converged
                init = result.fhat
            end
        end
    end

    return results
end

"""
    multistart_solve(prob::SCProblem; n_starts=5, kwargs...)

Run solver from multiple initial conditions to detect multiple fixed points.

Returns a vector of FixedPointResults, clustered by uniqueness.
"""
function multistart_solve(
    prob::SCProblem;
    n_starts::Int=5,
    cluster_tol::Float64=1e-6,
    kwargs...
)
    N = prob.disc.N
    results = FixedPointResult[]

    # Try uniform
    push!(results, solve_fixed_point(prob; init=:uniform, kwargs...))

    # Try bump
    push!(results, solve_fixed_point(prob; init=:bump, kwargs...))

    # Try random starts
    for i in 1:(n_starts - 2)
        init = random_initial(N; seed=i)
        push!(results, solve_fixed_point(prob; init=init, kwargs...))
    end

    # Filter converged
    converged = filter(r -> r.converged, results)

    # Cluster by m value (simple clustering)
    unique_results = FixedPointResult[]
    for r in converged
        is_new = true
        for u in unique_results
            if abs(r.m - u.m) < cluster_tol
                is_new = false
                break
            end
        end
        if is_new
            push!(unique_results, r)
        end
    end

    return unique_results
end

# Helper to update δ in coupling
function update_delta(c::LinearCoupling, δ::Float64)
    return LinearCoupling(δ, c.observable)
end

function update_delta(c::TanhCoupling, δ::Float64)
    return TanhCoupling(δ, c.β, c.observable)
end

export multistart_solve
