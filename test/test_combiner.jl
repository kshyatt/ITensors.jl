using ITensors, Test
using Combinatorics: permutations

i = Index(2,"i")
j = Index(2,"j")
k = Index(2,"k")
l = Index(2,"l")

A = randomITensor(i, j, k, l)

@testset "Two index combiner" begin
    for inds_ij ∈ permutations([i,j])
        C = combiner(inds_ij...)
        B = A*C
        @test hasinds(B, l, k)
        D = permute(B*C, i, j, k, l)
        @test hasinds(D, i, j, k, l)
        for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k), ll in 1:dim(l)
            @test D[i(ii), j(jj), k(kk), l(ll)] == A[i(ii), j(jj), k(kk), l(ll)]
        end
    end
    for inds_il ∈ permutations([i,l])
        C = combiner(inds_il...)
        B = A*C
        @test hasinds(B, j, k)
        D = permute(B*C, i, j, k, l)
        for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k), ll in 1:dim(l)
            @test D[i(ii), j(jj), k(kk), l(ll)] == A[i(ii), j(jj), k(kk), l(ll)]
        end
    end
    #=for inds_ik ∈ permutations([i,k])
        C = combiner(inds_ik...)
        B = A*C
        @test hasinds(B, j, l)
        D = B*C
        @test hasinds(D, i, j, k, l)
        for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k), ll in 1:dim(l)
            @test D[i(ii), j(jj), k(kk), l(ll)] == A[i(ii), j(jj), k(kk), l(ll)]
        end
    end
    for inds_jk ∈ permutations([j,k])
        C = combiner(inds_jk...)
        B = A*C
        @test hasindex(B, i, l)
        D = B*C
        @test hasinds(D, i, j, k, l)
        for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k), ll in 1:dim(l)
            @test D[i(ii), j(jj), k(kk), l(ll)] == A[i(ii), j(jj), k(kk), l(ll)]
        end
    end
    for inds_jl ∈ permutations([j,l])
        C = combiner(inds_jl...)
        B = A*C
        @test hasinds(B, i, k)
        D = B*C
        @test hasinds(D, i, j, k, l)
        for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k), ll in 1:dim(l)
            @test D[i(ii), j(jj), k(kk), l(ll)] == A[i(ii), j(jj), k(kk), l(ll)]
        end
    end
    for inds_kl ∈ permutations([k,l])
        C = combiner(inds_kl...)
        B = A*C
        @test hasinds(B, i, j)
        D = B*C
        @test hasinds(D, i, j, k, l)
        for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k), ll in 1:dim(l)
            @test D[i(ii), j(jj), k(kk), l(ll)] == A[i(ii), j(jj), k(kk), l(ll)]
        end
    end=#
end
#=
@testset "Three index combiner" begin
    for inds_ijl ∈ permutations([i,j,l])
        C = combiner(inds_ijl...)
        B = A*C
        D = B*C
        @test hasinds(D, i, j, k, l)
        for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k), ll in 1:dim(l)
            @test D[i(ii), j(jj), k(kk), l(ll)] == A[i(ii), j(jj), k(kk), l(ll)]
        end
    end
    for inds_ijk ∈ permutations([i,j,k])
        C = combiner(inds_ijk...)
        B = A*C
        @test hasindex(B, l)
        D = B*C
        @test hasinds(D, i, j, k, l)
        for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k), ll in 1:dim(l)
            @test D[i(ii), j(jj), k(kk), l(ll)] == A[i(ii), j(jj), k(kk), l(ll)]
        end
    end
    for inds_jkl ∈ permutations([j,k,l])
        C = combiner(inds_jkl...)
        B = A*C
        @test hasindex(B, i)
        D = B*C
        @test hasinds(D, i, j, k, l)
        for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k), ll in 1:dim(l)
            @test D[i(ii), j(jj), k(kk), l(ll)] == A[i(ii), j(jj), k(kk), l(ll)]
        end
    end
end
=#
