# # Rigorous Delta Sweep (Self-Consistent Logistic Noise)
#
# This tutorial computes verified fixed points for a sweep of coupling strengths
# and plots only those with a rigorous L2 error bound below a target threshold.
#
# We then compare selected deltas on a single plot.
#
# ## Mathematical Setup
#
# We study the self-consistent fixed point problem
# ```math
# T(f) = P_{c(f)} f, \qquad c(f) = \delta G(m(f)).
# ```
# The observable $m(f) = \langle \phi, f \rangle$ is a scalar functional. In the
# default cosine case, $\phi(x) = \cos(\pi x)$ so $m(f)$ is the first cosine
# Fourier coefficient. The coupling nonlinearity $G$ is model-dependent; for the
# linear coupling used here, $G(m) = m$ so $c(f) = \delta m(f)$.
# The underlying dynamics is the noisy quadratic map on the period-2 torus,
# $x \mapsto T_a(x) + c(f)$ with $T_a(x) = a - (a+1)x^2$, followed by Gaussian
# noise of width $\sigma$. Note that this $a$ is the parameter for the
# $[-1, 1]$ quadratic map, not the logistic map $a x(1-x)$ on $[0,1]$.

using SelfConsistentLogisticNoise
using Plots
using Printf
using Serialization

include(joinpath(@__DIR__, "cert_log.jl"))

# ## Configuration

a = 0.915         # Quadratic map parameter on [-1, 1]
σ = 0.02          # Noise width
N_candidates = [128, 192, 256, 384, 512, 768, 1024]
η = 0.0025
σ_sm = 0.0075
use_taper = false
M_override = 1048576
τ_CAP = 0.1       # Analyticity strip for truncation bounds
cache_dir = ".cache"
precompute_B = true
use_snapshots = false
write_markdown_summary = true
write_latex_summary = true

δ_range = 0.0:0.01:0.5
error_threshold = 0.1

npts = 1000
output_dir = joinpath("docs", "literate", "rigorous_delta_sweep_png")
mkpath(output_dir)
snapshot_a = joinpath(output_dir, "sweep_snapshot_A.bin")
snapshot_b = joinpath(output_dir, "sweep_snapshot_B.bin")
log_path = joinpath(output_dir, "certification_log.md")
jld2_path = joinpath(output_dir, "certification_output_delta_sweep.jld2")
latex_path = joinpath(output_dir, "certification_summary_delta_sweep.tex")

config = Dict(
    :script => "rigorous_delta_sweep",
    :a => a,
    :σ => σ,
    :N_candidates => N_candidates,
    :η => η,
    :σ_sm => σ_sm,
    :use_taper => use_taper,
    :M_override => M_override,
    :τ_CAP => τ_CAP,
    :δ_range => collect(δ_range),
    :error_threshold => error_threshold,
    :npts => npts,
    :log_path => log_path,
    :jld2_path => jld2_path,
    :latex_path => latex_path,
)

function load_snapshot(config, path_a, path_b)
    paths = filter(isfile, (path_a, path_b))
    if isempty(paths)
        return Dict{Float64, NamedTuple}(), Dict{Float64, NamedTuple}()
    end
    mtimes = map(p -> stat(p).mtime, paths)
    latest_path = paths[argmax(mtimes)]
    state = try
        deserialize(latest_path)
    catch err
        println("Snapshot load failed ($(err)); ignoring saved state.")
        return Dict{Float64, NamedTuple}(), Dict{Float64, NamedTuple}()
    end
    if get(state, :config, nothing) != config
        println("Snapshot config mismatch; ignoring saved state.")
        return Dict{Float64, NamedTuple}(), Dict{Float64, NamedTuple}()
    end
    return state[:verified], state[:records]
end

function save_snapshot!(config, verified, records, path_a, path_b, toggle_ref)
    state = Dict(
        :config => config,
        :verified => verified,
        :records => records,
        :timestamp => time(),
    )
    path = toggle_ref[] ? path_a : path_b
    serialize(path, state)
    toggle_ref[] = !toggle_ref[]
end

println("="^70)
println("RIGOROUS DELTA SWEEP")
println("="^70)
println("a = $a, σ = $σ, N candidates = $(N_candidates), τ = $τ_CAP")
println("η = $η (taper width)")
println("σ_sm = $σ_sm (smoothing width)")
println("taper = $use_taper, M = $M_override")
println("δ range = $(first(δ_range)) to $(last(δ_range)) (step = $(step(δ_range)))")
println("L2 error threshold = $error_threshold")

# ## Helper

function compute_verified_density(prob::SCProblem, fhat::Vector{ComplexF64})
    cap = verify_fixed_point_CAP(prob, fhat; τ=τ_CAP, verbose=false)
    if cap.verified && cap.total_error <= error_threshold
        xgrid, fvals = reconstruct_density(fhat; npts=npts)
        return (xgrid=xgrid, fvals=fvals, cap=cap), cap
    end
    return nothing, cap
end

# ## Sweep and Plot (separate)

verified, records = use_snapshots ? load_snapshot(config, snapshot_a, snapshot_b) :
    (Dict{Float64, NamedTuple}(), Dict{Float64, NamedTuple}())
if !isempty(verified) || !isempty(records)
    println("Loaded snapshot with $(length(records)) recorded points.")
end
snapshot_toggle = Ref(true)
verified = copy(verified)
records = copy(records)

if precompute_B
    println("\nPrecomputing B matrices for all N (cached)...")
    pre_start = time()
    map = QuadraticMap(a)
    for N in N_candidates
        disc = isnothing(M_override) ? FourierDisc(N) : FourierDisc(N, M_override)
        build_B(map, disc; cache=true, cache_dir=cache_dir, η=η, σ_sm=σ_sm, taper=use_taper)
    end
    @printf("Precompute time: %.2f s\n", time() - pre_start)
end

println("\nRunning sweep...")
total_start = time()
for δ in δ_range
    if use_snapshots && haskey(records, δ)
        @printf("\nδ = %.2f (skipping; already recorded)\n", δ)
        continue
    end
    @printf("\nδ = %.2f\n", δ)
    delta_start = time()
    verified_entry = nothing
    cap = nothing
    used_N = nothing
    for N in N_candidates
        @printf("  trying N = %d\n", N)
        prob = build_problem(
            a=a, σ=σ, N=N, δ=δ, coupling_type=:linear,
            η=η, σ_sm=σ_sm, taper=use_taper, M_override=M_override, cache=true
        )
        result = solve_hybrid(prob; α=0.3, picard_tol=1e-6, newton_tol=1e-14, verbose=false)

        if !result.converged
            println("    solver did not converge; continuing")
            continue
        end

        verified_entry, cap = compute_verified_density(prob, result.fhat)
        if verified_entry !== nothing
            used_N = N
            break
        end
        println("    not verified or error too large: total_error = $(cap.total_error)")
    end

    if verified_entry === nothing
        println("  no certified result for this δ")
        records[δ] = (
            verified=false,
            used_N=used_N,
            total_error=cap === nothing ? nothing : cap.total_error,
            cap=cap,
            xgrid=nothing,
            fvals=nothing,
        )
        save_snapshot!(config, verified, records, snapshot_a, snapshot_b, snapshot_toggle)
        @printf("  elapsed: %.2f s\n", time() - delta_start)
        continue
    end

    verified[δ] = verified_entry
    records[δ] = (
        verified=true,
        used_N=used_N,
        total_error=cap === nothing ? nothing : cap.total_error,
        cap=cap,
        xgrid=verified_entry.xgrid,
        fvals=verified_entry.fvals,
    )
    save_snapshot!(config, verified, records, snapshot_a, snapshot_b, snapshot_toggle)
    cap = verified_entry.cap
    @printf("  verified at N = %d: total_error = %.6e\n", used_N, cap.total_error)
    @printf("  elapsed: %.2f s\n", time() - delta_start)

    p = plot(
        verified_entry.xgrid,
        verified_entry.fvals;
        label=false,
        lw=2,
        title=@sprintf("δ = %.2f (L2 ≤ %.2e, N = %d)", δ, cap.total_error, used_N),
        xlabel="x",
        ylabel="f(x)",
        size=(800, 400),
    )
    savefig(p, joinpath(output_dir, @sprintf("density_delta_%0.2f.png", δ)))
    display(p)
end
@printf("\nTotal sweep time: %.2f s\n", time() - total_start)

# ## Comparison Plot (selected deltas)

selected = 0.0:0.1:0.5
p_compare = plot(
    title="Verified stationary densities (selected δ)",
    xlabel="x",
    ylabel="f(x)",
    size=(900, 500),
)

for δ in selected
    entry = get(verified, δ, nothing)
    if entry === nothing
        @printf("  skipping δ = %.2f (not verified or not computed)\n", δ)
        continue
    end
    cap = entry.cap
    plot!(
        p_compare,
        entry.xgrid,
        entry.fvals;
        label=@sprintf("δ = %.2f (≤ %.1e)", δ, cap.total_error),
        lw=2,
    )
end

savefig(p_compare, joinpath(output_dir, "density_comparison_selected.png"))
display(p_compare)

if write_markdown_summary
    write_cert_log(
        log_path;
        title="Delta sweep certification log",
        key_label="δ",
        key_format=δ -> @sprintf("%.2f", δ),
        config=config,
        records=records,
    )
end
save_state_jld2(jld2_path, config, verified, records)

if write_latex_summary
    write_latex_summary(
        latex_path;
        title="Rigorous delta sweep summary",
        key_label="\\\$\\delta\\\$",
        key_values=collect(0.0:0.05:0.5),
        key_format=δ -> @sprintf("%.2f", δ),
        records=records,
        fig_path=joinpath("docs", "literate", "rigorous_delta_sweep_png", "density_comparison_selected.png"),
        fig_caption="Verified stationary densities for selected \\\$\\delta\\\$ values.",
        fig_label="delta_sweep",
    )
end
