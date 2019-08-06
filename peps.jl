using ITensors, ITensors.CuITensors, Random, Logging

import ITensors: tensors

mutable struct PEPS
    Nx::Int
    Ny::Int
    A_::AbstractMatrix{ITensor}

    PEPS() = new(0, 0, Matrix{ITensor}(),0,0)

    PEPS(Nx::Int, Ny::Int, A::Matrix{ITensor}) = new(Nx, Ny, A)
    function PEPS(sites::SiteSet, lattice::Lattice, Nx::Int, Ny::Int; mindim::Int=1)
        p  = Matrix{ITensor}(undef, Nx, Ny)
        @show mindim
        right_links = [ Index(mindim, "Link,c$i,r$j,r") for i in 1:Nx, j in 1:Ny ]
        up_links    = [ Index(mindim, "Link,c$i,r$j,u") for i in 1:Nx, j in 1:Ny ]
        @inbounds for ii in eachindex(sites)
            row = div(ii-1, Nx) + 1
            col = mod(ii-1, Nx) + 1
            s = sites[ii]
            if 1 < row < Ny && 1 < col < Nx 
                p[row, col] = ITensor(right_links[row, col], up_links[row, col], right_links[row, col-1], up_links[row-1, col], s)
            elseif row == 1 && 1 < col < Nx
                p[row, col] = ITensor(right_links[row, col], up_links[row, col], right_links[row, col-1], s)
            elseif 1 < row < Ny && col == 1
                p[row, col] = ITensor(right_links[row, col], up_links[row, col], up_links[row-1, col], s)
            elseif row == Ny && 1 < col < Nx 
                p[row, col] = ITensor(right_links[row, col], right_links[row, col-1], up_links[row-1, col], s)
            elseif 1 < row < Ny && col == Nx 
                p[row, col] = ITensor(up_links[row, col], right_links[row, col-1], up_links[row-1, col], s)
            elseif row == Ny && col == 1 
                p[row, col] = ITensor(right_links[row, col], up_links[row-1, col], s)
            elseif row == Ny && col == Nx 
                p[row, col] = ITensor(right_links[row, col-1], up_links[row-1, col], s)
            elseif row == 1 && col == 1 
                p[row, col] = ITensor(right_links[row, col], up_links[row, col], s)
            elseif row == 1 && col == Nx
                p[row, col] = ITensor(up_links[row, col], right_links[row, col-1], s)
            end
        end
        new(Nx, Ny, p)
    end
end

function randomPEPS(sites, Nx::Int, Ny::Int; mindim::Int=1)
    lattice = squareLattice(Nx,Ny,yperiodic=false)
    A = PEPS(sites, lattice, Nx, Ny, mindim=mindim)
    @inbounds for ii ∈ eachindex(sites)
        randn!(A[ii])
        normalize!(A[ii])
    end
    return A
end

function cuPEPS(A::PEPS)
    cA = copy(A)
    for i in 1:Nx, j in 1:Ny
        cA[i, j] = cuITensor(A[i, j])
    end
    return cA
end

tensors(A::PEPS) = A.A_
Base.size(A::PEPS) = (A.Ny, A.Nx)
Base.getindex(A::PEPS, i::Integer, j::Integer) = getindex(tensors(A), i, j)::ITensor
Base.getindex(A::PEPS, i::Integer) = getindex(tensors(A), i)::ITensor
Base.setindex!(A::PEPS, val::ITensor, i::Integer, j::Integer) = setindex!(tensors(A), val, i, j)

Base.copy(A::PEPS)    = PEPS(A.Nx, A.Ny, copy(tensors(A)))
Base.similar(A::PEPS) = PEPS(A.Nx, A.Ny, similar(tensors(A)))

function Base.show(io::IO, A::PEPS)
  print(io,"PEPS")
  (size(A)[1] > 0 && size(A)[2] > 0) && print(io,"\n")
  @inbounds for i in 1:A.Nx, j in 1:A.Ny
      println(io,"$i $j $(inds(A[i,j]))")
  end
end

@enum Op_Type Field=0 Vertical=1 Horizontal=2
struct Operator
    sites::Vector{Pair{Int,Int}}
    ops::Vector{ITensor}
    site_ind::Index
    dir::Op_Type
end

getDirectional(ops::Vector{Operator}, dir::Op_Type) = collect(filter(x->x.dir==dir, ops))

function spinI(s::Index)::ITensor
    I = ITensor(s, s')
    I[s(1), s'(1)] = 1.0
    I[s(2), s'(2)] = 1.0
    return I
end

function makeH_XXZ(Nx::Int, Ny::Int, J::Real; pinning::Bool=false)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    Ident = spinI(s)
    P = ITensor(s, s')
    M = ITensor(s, s')
    P[s(1), s'(2)] = 1.0
    M[s(2), s'(1)] = 1.0
    H = Matrix{Vector{Operator}}(undef, Ny, Nx)
    for col in 1:Nx, row in 1:Ny
        H[row, col] = Vector{Operator}()
        if row < Ny
            op_a  = 0.5 * P
            op_b  = copy(M)
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Vertical))

            op_a  = 0.5 * M
            op_b  = copy(P)
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Vertical))

            if J != 0.0
                op_a  = J * Z
                op_b  = copy(Z)
                sites = [row=>col, row+1=>col]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Vertical))
            end
        end
        if col < Nx
            op_a  = 0.5 * P
            op_b  = copy(M)
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Horizontal))

            op_a  = 0.5 * M
            op_b  = copy(P)
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Horizontal))

            if J != 0.0
                op_a  = J * Z
                op_b  = copy(Z)
                sites = [row=>col, row=>col+1]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Horizontal))
            end
        end
    end
    return H
end

function makeCuH_XXZ(Nx::Int, Ny::Int, J::Real; pinning::Bool=false)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    Ident = spinI(s)
    P = ITensor(s, s')
    M = ITensor(s, s')
    P[s(1), s'(2)] = 1.0
    M[s(2), s'(1)] = 1.0
    H = Matrix{Vector{Operator}}(undef, Ny, Nx)
    for col in 1:Nx, row in 1:Ny
        H[row, col] = Vector{Operator}()
        if row < Ny
            op_a  = 0.5 * P
            op_b  = copy(M)
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Vertical))

            op_a  = 0.5 * M
            op_b  = copy(P)
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Vertical))

            if J != 0.0
                op_a  = J * Z
                op_b  = copy(Z)
                sites = [row=>col, row+1=>col]
                push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Vertical))
            end
        end
        if col < Nx
            op_a  = 0.5 * P
            op_b  = copy(M)
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Horizontal))

            op_a  = 0.5 * M
            op_b  = copy(P)
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Horizontal))

            if J != 0.0
                op_a  = J * Z
                op_b  = copy(Z)
                sites = [row=>col, row=>col+1]
                push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Horizontal))
            end
        end
    end
    return H
end


function combine(AA::ITensor, Aorig::ITensor, Anext::ITensor, tags::String)
    ci   = commonindex(Aorig, Anext)
    cmb  = combiner(IndexSet(ci, prime(ci)), tags=tags)
    AA  *= cmb
    return cmb, AA
end

function reconnect(combiner_ind::Index, environment::ITensor)
    environment_combiner       = findIndex(environment, "Site")
    new_combiner               = combiner(IndexSet(combiner_ind, prime(combiner_ind)), tags="Site")
    combined_ind               = findindex(combiner, "Site")
    combiner_transfer          = δ(combined_ind, environment_combiner)
    return new_combiner*combiner_transfer
end

include("environments.jl")
include("ancillaries.jl")
io = open("log.txt", "w+")
logger = SimpleLogger(io)
global_logger(logger)
Nx = 4
Ny = 4
J  = 1.0
sites = spinHalfSites(Nx*Ny)
println("Beginning A")
A  = randomPEPS(sites, Nx, Ny, mindim=6)
cA = cuPEPS(A)
H  = makeCuH_XXZ(Nx, Ny, J)
@info "Built cA and H"
Ls = buildLs(cA, H; maxdim=12)
@info "Built first Ls"
Rs = buildRs(cA, H; maxdim=12)
@info "Built first Rs"
Ls, tL, bytes, gctime, memallocs = @timed buildLs(cA, H; maxdim=12)
Rs, tR, bytes, gctime, memallocs = @timed buildRs(cA, H; maxdim=12)
println("Done building GPU $tL $tR")
A  = randomPEPS(sites, Nx, Ny, mindim=6)
H  = makeH_XXZ(Nx, Ny, J)
@info "Built A and H"
Ls = buildLs(A, H; maxdim=12)
@info "Built first Ls"
Rs = buildRs(A, H; maxdim=12)
@info "Built first Rs"
Ls, tL, bytes, gctime, memallocs = @timed buildLs(A, H; maxdim=12)
Rs, tR, bytes, gctime, memallocs = @timed buildRs(A, H; maxdim=12)
println("Done building CPU $tL $tR")
