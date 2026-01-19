using SelfConsistentLogisticNoise
using Documenter
using Literate

# Generate notebooks as markdown for documentation
TUTORIALS_INPUT = joinpath(@__DIR__, "..", "notebooks")
TUTORIALS_OUTPUT = joinpath(@__DIR__, "src", "tutorials")

# Create output directory
mkpath(TUTORIALS_OUTPUT)

# Convert notebooks to markdown
for file in readdir(TUTORIALS_INPUT)
    if endswith(file, ".jl")
        Literate.markdown(
            joinpath(TUTORIALS_INPUT, file),
            TUTORIALS_OUTPUT;
            documenter=true
        )
    end
end

DocMeta.setdocmeta!(SelfConsistentLogisticNoise, :DocTestSetup, :(using SelfConsistentLogisticNoise); recursive=true)

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
        "Tutorials" => [
            "Computing a Fixed Point" => "tutorials/fixed_point_tutorial.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/orkolorko/SelfConsistentLogisticNoise.jl",
    devbranch="main",
)
