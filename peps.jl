using ITensors, Random

import ITensors: tensors

mutable struct PEPS
    Nx::Int
    Ny::Int
    A_::Matrix{ITensor}

    PEPS() = new(0, 0, Matrix{ITensor}(),0,0)

    PEPS(Nx::Int, Ny::Int, A::Matrix{ITensor}) = new(Nx, Ny, A)
    function PEPS(sites::SiteSet, lattice::Lattice, Nx::Int, Ny::Int)
        p  = Matrix{ITensor}(undef, Nx, Ny)
        right_links = [ Index(1, "Link,c$i,r$j,r") for i in 1:Nx, j in 1:Ny ]
        up_links    = [ Index(1, "Link,c$i,r$j,u") for i in 1:Nx, j in 1:Ny ]
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

function randomPEPS(sites, Nx::Int, Ny::Int)
    lattice = squareLattice(Nx,Ny,yperiodic=false)
    A = PEPS(sites, lattice, Nx, Ny)
    @inbounds for ii ∈ eachindex(sites)
        randn!(A[ii])
        normalize!(A[ii])
    end
    return A
end

tensors(A::PEPS) = A.A_
Base.size(A::PEPS) = (A.Ny, A.Nx)
Base.getindex(A::PEPS, i::Integer, j::Integer) = getindex(tensors(A), i, j)
Base.getindex(A::PEPS, i::Integer) = getindex(tensors(A), i)
Base.setindex!(A::PEPS, val::ITensor, i::Integer, j::Integer) = setindex!(tensors(A), T, i, j)

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

Nx = 4
Ny = 4
J  = 1.0
sites = spinHalfSites(Nx*Ny)
A = randomPEPS(sites, Nx, Ny)
H = makeH_XXZ(Nx, Ny, J)
Ls = buildLs(A, H)
Rs = buildRs(A, H)
