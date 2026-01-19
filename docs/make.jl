using SelfConsistentLogisticNoise
using Documenter

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
        "API Reference" => "api.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(;
    repo="github.com/orkolorko/SelfConsistentLogisticNoise.jl",
    devbranch="main",
)
