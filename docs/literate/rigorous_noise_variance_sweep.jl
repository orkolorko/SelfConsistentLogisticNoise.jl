# # Rigorous Noise-Variance Sweep (δ = 0)
#
# This tutorial performs a sanity-check sweep over the noise variance
# (σ²) while keeping the coupling fixed at δ = 0. We only retain
# fixed points whose rigorous L2 error bound is below a target threshold.

using SelfConsistentLogisticNoise
using Plots
using Printf
using Serialization

include(joinpath(@__DIR__, "cert_log.jl"))

# ## Configuration

a = 0.915         # Quadratic map parameter on [-1, 1]
δ = 0.0           # No coupling for this sweep
N_candidates = [128, 192, 256, 384, 512, 768, 1024]
η = 0.0025
σ_sm = 0.0075
use_taper = false
M_override = 1048576
τ_CAP = 0.1       # Analyticity strip for truncation bounds
cache_dir = ".cache"
precompute_B = true

# Sweep over variance: σ² in [0.0001, 0.0025] corresponds to σ in [0.01, 0.05]
var_range = 0.0001:0.0001:0.0025
error_threshold = 0.1

npts = 1000
output_dir = joinpath("docs", "literate", "rigorous_noise_variance_sweep_png")
mkpath(output_dir)
snapshot_a = joinpath(output_dir, "sweep_snapshot_A.bin")
snapshot_b = joinpath(output_dir, "sweep_snapshot_B.bin")
log_path = joinpath(output_dir, "certification_log.md")
jld2_path = joinpath(output_dir, "certification_output.jld2")

config = Dict(
    :script => "rigorous_noise_variance_sweep",
    :a => a,
    :δ => δ,
    :N_candidates => N_candidates,
    :η => η,
    :σ_sm => σ_sm,
    :use_taper => use_taper,
    :M_override => M_override,
    :τ_CAP => τ_CAP,
    :var_range => collect(var_range),
    :error_threshold => error_threshold,
    :npts => npts,
    :log_path => log_path,
    :jld2_path => jld2_path,
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
println("RIGOROUS NOISE-VARIANCE SWEEP (δ = 0)")
println("="^70)
println("a = $a, N candidates = $(N_candidates), τ = $τ_CAP, δ = $δ")
println("η = $η (taper width)")
println("σ_sm = $σ_sm (smoothing width)")
println("taper = $use_taper, M = $M_override")
println("variance range = $(first(var_range)) to $(last(var_range)) (step = $(step(var_range)))")
println("L2 error threshold = $error_threshold")

# ## Helper

function compute_verified_density(prob::SCProblem, fhat::Vector{ComplexF64}, σ::Float64)
    cap = verify_fixed_point_CAP(prob, fhat; τ=τ_CAP, verbose=false)
    if cap.verified && cap.total_error <= error_threshold
        xgrid, fvals = reconstruct_density(fhat; npts=npts)
        return (xgrid=xgrid, fvals=fvals, cap=cap, σ=σ), cap
    end
    return nothing, cap
end

# ## Sweep and Plot

verified, records = load_snapshot(config, snapshot_a, snapshot_b)
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
for var in var_range
    if haskey(records, var)
        @printf("\nσ² = %.4e (skipping; already recorded)\n", var)
        continue
    end
    σ = sqrt(var)
    @printf("\nσ² = %.4e (σ = %.4e)\n", var, σ)

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

        verified_entry, cap = compute_verified_density(prob, result.fhat, σ)
        if verified_entry !== nothing
            used_N = N
            break
        end
        println("    not verified or error too large: total_error = $(cap.total_error)")
    end

    if verified_entry === nothing
        println("  no certified result for this variance")
        records[var] = (
            verified=false,
            used_N=used_N,
            total_error=cap === nothing ? nothing : cap.total_error,
            cap=cap,
            xgrid=nothing,
            fvals=nothing,
            σ=σ,
        )
        save_snapshot!(config, verified, records, snapshot_a, snapshot_b, snapshot_toggle)
        continue
    end

    verified[var] = verified_entry
    records[var] = (
        verified=true,
        used_N=used_N,
        total_error=cap === nothing ? nothing : cap.total_error,
        cap=cap,
        xgrid=verified_entry.xgrid,
        fvals=verified_entry.fvals,
        σ=σ,
    )
    save_snapshot!(config, verified, records, snapshot_a, snapshot_b, snapshot_toggle)
    @printf("  verified at N = %d: total_error = %.6e\n", used_N, cap.total_error)

    p = plot(
        verified_entry.xgrid,
        verified_entry.fvals;
        label=false,
        lw=2,
        title=@sprintf("σ² = %.4e (σ = %.3e, N = %d)", var, σ, used_N),
        xlabel="x",
        ylabel="f(x)",
        size=(800, 400),
    )
    savefig(p, joinpath(output_dir, @sprintf("density_sigma2_%0.4e_N_%d.png", var, used_N)))
    display(p)
end

write_cert_log(
    log_path;
    title="Noise-variance sweep certification log",
    key_label="σ²",
    key_format=σ2 -> @sprintf("%.4e", σ2),
    config=config,
    records=records,
)
save_state_jld2(jld2_path, config, verified, records)
