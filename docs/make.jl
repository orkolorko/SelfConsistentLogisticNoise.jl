using SelfConsistentLogisticNoise
using Documenter
using Literate

DocMeta.setdocmeta!(SelfConsistentLogisticNoise, :DocTestSetup, :(using SelfConsistentLogisticNoise); recursive=true)

examples = [
    "rigorous_delta_sweep.jl",
    "rigorous_noise_variance_sweep.jl",
    "rigorous_a_sweep_sigma2_1e4.jl",
]

lit_in_dir = joinpath(@__DIR__, "literate")
lit_out_dir = joinpath(@__DIR__, "src", "generated")
mkpath(lit_out_dir)

for file in examples
    input = joinpath(lit_in_dir, file)
    name = splitext(file)[1]
    # Render as plain markdown (no @example blocks) to avoid executing heavy runs.
    Literate.markdown(input, lit_out_dir; name=name, documenter=false, execute=false)
end

makedocs(;
    modules=[SelfConsistentLogisticNoise],
    authors="Isaia Nisoli",
    sitename="SelfConsistentLogisticNoise.jl",
    format=Documenter.HTML(;
        canonical="https://orkolorko.github.io/SelfConsistentLogisticNoise.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Mathematical Background" => "theory.md",
        "API Reference" => "api.md",
        "Examples" => [
            "Delta Sweep" => "generated/rigorous_delta_sweep.md",
            "Noise Variance Sweep" => "generated/rigorous_noise_variance_sweep.md",
            "a Sweep" => "generated/rigorous_a_sweep_sigma2_1e4.md",
        ],
    ],
    warnonly = [:missing_docs],
)

deploydocs(;
    repo="github.com/orkolorko/SelfConsistentLogisticNoise.jl",
    devbranch="main",
)
