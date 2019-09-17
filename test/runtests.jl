using ITensors, Test

@testset "ITensors.jl" begin
    @testset "$filename" for filename in (
        "test_tagset.jl",
        "test_smallstring.jl",
        "test_index.jl",
        "test_indexset.jl",
        "test_itensor.jl",
        "test_contract.jl",
        "test_diagITensor.jl",
        "test_combiner.jl",
        "test_trg.jl",
        #"test_ctmrg.jl",
        "test_dmrg.jl",
        "test_siteset.jl",
        "test_phys_sitesets.jl",
        "test_decomp.jl",
        "test_lattices.jl",
        "test_mps.jl",
        "test_mpo.jl",
        "test_autompo.jl",
        "test_svd.jl",
        "test_cucontract.jl",
        "test_cuitensor.jl",
        "test_cumps.jl",
        "test_cumpo.jl",
    )
      println("Running $filename")
      include(filename)
    end
end
