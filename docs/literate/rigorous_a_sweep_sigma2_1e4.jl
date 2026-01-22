# # Rigorous a-Sweep at Fixed Noise Variance (σ² = 1e-4)
#
# This tutorial sweeps the quadratic map parameter a in [0.9, 1.0]
# at fixed noise variance σ² = 1e-4 and coupling δ = 0. We only retain
# fixed points whose rigorous L2 error bound is below a target threshold.
# If certification fails, we retry with larger N.

using SelfConsistentLogisticNoise
using Plots
using Printf

include(joinpath(@__DIR__, "cert_log.jl"))

# ## Configuration

σ2 = 1e-4
σ = sqrt(σ2)
δ = 0.0
N_candidates = [128, 192, 256, 384, 512, 768, 1024]
η = 0.0025
σ_sm = 0.0075
use_taper = false
M_override = 1024
τ_CAP = 0.1

a_range = 0.9:0.001:1.0
error_threshold = 0.1

npts = 1000
output_dir = joinpath("docs", "literate", "rigorous_a_sweep_sigma2_1e4_png")
mkpath(output_dir)
log_path = joinpath(output_dir, "certification_log.md")
jld2_path = joinpath(output_dir, "certification_output.jld2")

config = Dict(
    :script => "rigorous_a_sweep_sigma2_1e4",
    :σ2 => σ2,
    :σ => σ,
    :δ => δ,
    :N_candidates => N_candidates,
    :η => η,
    :σ_sm => σ_sm,
    :use_taper => use_taper,
    :M_override => M_override,
    :τ_CAP => τ_CAP,
    :a_range => collect(a_range),
    :error_threshold => error_threshold,
    :npts => npts,
    :log_path => log_path,
    :jld2_path => jld2_path,
)

println("="^70)
println("RIGOROUS a-SWEEP (σ² = 1e-4, δ = 0)")
println("="^70)
println("σ² = $σ2, σ = $σ, δ = $δ, τ = $τ_CAP")
println("η = $η (taper width)")
println("σ_sm = $σ_sm (smoothing width)")
println("taper = $use_taper, M = $M_override")
println("a range = $(first(a_range)) to $(last(a_range)) (step = $(step(a_range)))")
println("N candidates = $(N_candidates)")
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

# ## Sweep and Plot

println("\nRunning sweep...")
verified = Dict{Float64, NamedTuple}()
records = Dict{Float64, NamedTuple}()
for a in a_range
    @printf("\na = %.2f\n", a)
    verified_entry = nothing
    cap = nothing
    used_N = nothing
    for N in N_candidates
        @printf("  trying N = %d\n", N)
        prob = build_problem(
            a=a, σ=σ, N=N, δ=δ, coupling_type=:linear,
            η=η, σ_sm=σ_sm, taper=use_taper, M_override=M_override, cache=false
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
        println("  no certified result for this a")
        records[a] = (
            verified=false,
            used_N=used_N,
            total_error=cap === nothing ? nothing : cap.total_error,
            cap=cap,
            xgrid=nothing,
            fvals=nothing,
        )
        continue
    end

    @printf("  verified at N = %d: total_error = %.6e\n", used_N, cap.total_error)
    verified[a] = verified_entry
    records[a] = (
        verified=true,
        used_N=used_N,
        total_error=cap === nothing ? nothing : cap.total_error,
        cap=cap,
        xgrid=verified_entry.xgrid,
        fvals=verified_entry.fvals,
    )

    p = plot(
        verified_entry.xgrid,
        verified_entry.fvals;
        label=false,
        lw=2,
        title=@sprintf("a = %.2f (σ² = %.1e, N = %d)", a, σ2, used_N),
        xlabel="x",
        ylabel="f(x)",
        size=(800, 400),
    )
    savefig(p, joinpath(output_dir, @sprintf("density_a_%0.2f_N_%d.png", a, used_N)))
    display(p)
end

# ## Comparison Plot (selected a)

selected = collect(range(0.9, 0.99; length=10))
p_compare = plot(
    title="Verified stationary densities (selected a)",
    xlabel="x",
    ylabel="f(x)",
    size=(900, 500),
)

for a in selected
    entry = get(verified, a, nothing)
    if entry === nothing
        @printf("  skipping a = %.2f (not verified or not computed)\n", a)
        continue
    end
    cap = entry.cap
    plot!(
        p_compare,
        entry.xgrid,
        entry.fvals;
        label=@sprintf("a = %.2f (≤ %.1e)", a, cap.total_error),
        lw=2,
    )
end

savefig(p_compare, joinpath(output_dir, "density_comparison_selected.png"))
display(p_compare)

write_cert_log(
    log_path;
    title="a-sweep certification log (σ² = 1e-4)",
    key_label="a",
    key_format=a -> @sprintf("%.3f", a),
    config=config,
    records=records,
)
save_state_jld2(jld2_path, config, verified, records)
