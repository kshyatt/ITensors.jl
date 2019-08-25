using ITensors, ITensors.CuITensors
using CuArrays
using Random, Logging, LinearAlgebra

import ITensors: tensors

mutable struct PEPS
    Nx::Int
    Ny::Int
    A_::AbstractMatrix{ITensor}

    PEPS() = new(0, 0, Matrix{ITensor}(),0,0)

    PEPS(Nx::Int, Ny::Int, A::Matrix{ITensor}) = new(Nx, Ny, A)
    function PEPS(sites::SiteSet, lattice::Lattice, Nx::Int, Ny::Int; mindim::Int=1, is_gpu::Bool=false)
        p  = Matrix{ITensor}(undef, Nx, Ny)
        right_links = [ Index(mindim, "Link,c$j,r$i,r") for i in 1:Nx, j in 1:Ny ]
        up_links    = [ Index(mindim, "Link,c$j,r$i,u") for i in 1:Nx, j in 1:Ny ]
        T           = is_gpu ? cuITensor : ITensor
        @inbounds for ii in eachindex(sites)
            row = div(ii-1, Nx) + 1
            col = mod(ii-1, Nx) + 1
            s = sites[ii]
            if 1 < row < Ny && 1 < col < Nx 
                p[row, col] = T(right_links[row, col], up_links[row, col], right_links[row, col-1], up_links[row-1, col], s)
            elseif row == 1 && 1 < col < Nx
                p[row, col] = T(right_links[row, col], up_links[row, col], right_links[row, col-1], s)
            elseif 1 < row < Ny && col == 1
                p[row, col] = T(right_links[row, col], up_links[row, col], up_links[row-1, col], s)
            elseif row == Ny && 1 < col < Nx 
                p[row, col] = T(right_links[row, col], right_links[row, col-1], up_links[row-1, col], s)
            elseif 1 < row < Ny && col == Nx 
                p[row, col] = T(up_links[row, col], right_links[row, col-1], up_links[row-1, col], s)
            elseif row == Ny && col == 1 
                p[row, col] = T(right_links[row, col], up_links[row-1, col], s)
            elseif row == Ny && col == Nx 
                p[row, col] = T(right_links[row, col-1], up_links[row-1, col], s)
            elseif row == 1 && col == 1 
                p[row, col] = T(right_links[row, col], up_links[row, col], s)
            elseif row == 1 && col == Nx
                p[row, col] = T(up_links[row, col], right_links[row, col-1], s)
            end
        end
        new(Nx, Ny, p)
    end
end

include("environments.jl")
include("ancillaries.jl")
include("gauge.jl")

function cudelt(left::Index, right::Index)
    d_data   = CuArrays.zeros(Float64, dim(left), dim(right))
    ddi = diagind(d_data, 0)
    d_data[ddi] = 1.0
    delt = cuITensor(vec(d_data), left, right)
    return delt
end

function checkerboardPEPS(sites, Nx::Int, Ny::Int; mindim::Int=1)
    lattice = squareLattice(Nx,Ny,yperiodic=false)
    A = PEPS(sites, lattice, Nx, Ny, mindim=mindim)
    @inbounds for ii ∈ eachindex(sites)
        row = div(ii-1, Nx) + 1
        col = mod(ii-1, Nx) + 1
        spin_side = isodd(row - 1) ⊻ isodd(col - 1) ? 1 : 2
        si  = findindex(A[ii], "Site") 
        lis = findinds(A[ii], "Link") 
        ivs = [li(1) for li in lis]
        ivs = vcat(ivs, si(spin_side))
        A[ii][ivs...] = 1.0
    end
    return A
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

function randomCuPEPS(sites, Nx::Int, Ny::Int; mindim::Int=1)
    lattice = squareLattice(Nx,Ny,yperiodic=false)
    A = PEPS(sites, lattice, Nx, Ny; mindim=mindim, is_gpu=true)
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

tensors(A::PEPS)   = A.A_
Base.size(A::PEPS) = (A.Ny, A.Nx)

Base.getindex(A::PEPS, i::Integer, j::Integer) = getindex(tensors(A), i, j)::ITensor
Base.getindex(A::PEPS, ::Colon,    j::Integer) = getindex(tensors(A), :, j)::Vector{ITensor}
Base.getindex(A::PEPS, i::Integer, ::Colon)    = getindex(tensors(A), i, :)::Vector{ITensor}
Base.getindex(A::PEPS, i::Integer)             = getindex(tensors(A), i)::ITensor

Base.setindex!(A::PEPS, val::ITensor, i::Integer, j::Integer)       = setindex!(tensors(A), val, i, j)
Base.setindex!(A::PEPS, vals::Vector{ITensor}, ::Colon, j::Integer) = setindex!(tensors(A), vals, :, j)
Base.setindex!(A::PEPS, vals::Vector{ITensor}, i::Integer, ::Colon) = setindex!(tensors(A), vals, i, :)

Base.copy(A::PEPS)    = PEPS(A.Nx, A.Ny, copy(tensors(A)))
Base.similar(A::PEPS) = PEPS(A.Nx, A.Ny, similar(tensors(A)))

function Base.show(io::IO, A::PEPS)
  print(io,"PEPS")
  (size(A)[1] > 0 && size(A)[2] > 0) && print(io,"\n")
  @inbounds for i in 1:A.Nx, j in 1:A.Ny
      println(io,"$i $j $(A[i,j])")
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

function spinI(s::Index; is_gpu::Bool=false)::ITensor
    I_data      = is_gpu ? CuArrays.zeros(Float64, dim(s), dim(s)) : zeros(Float64, dim(s), dim(s))
    idi         = diagind(I_data, 0)
    I_data[idi] = CuArrays.ones(Float64, dim(s))
    I           = is_gpu ? cuITensor( vec(I_data), IndexSet(s, s') ) : ITensor(vec(I_data), IndexSet(s, s'))
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
    # pinning fields
    J_ = 1.;
    for row in 1:Ny
        op = isodd(row) ? (J_/2.0) * Z : (-J_/2.0) * Z   
        push!(H[row, 1], Operator([row=>1], [op], s, Field))    
        op = isodd(row) ? (-J_/2.0) * Z : (J_/2.0) * Z   
        push!(H[row, Nx], Operator([row=>Nx], [op], s, Field))    
    end
    for col in 2:Nx-1
        op = isodd(col) ? (-J_/2.0) * Z : (J_/2.0) * Z   
        push!(H[Ny, col], Operator([Ny=>col], [op], s, Field))    
        op = isodd(col) ? (J_/2.0) * Z : (-J_/2.0) * Z   
        push!(H[1, col], Operator([1=>col], [op], s, Field))    
    end
    return H
end

function makeCuH_XXZ(Nx::Int, Ny::Int, J::Real; pinning::Bool=false)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    #Ident = spinI(s)
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
    # pinning fields
    J_ = 1.;
    for row in 1:Ny
        op = isodd(row) ? (-J_/2.0) * Z : (J_/2.0) * Z   
        push!(H[row, 1], Operator([row=>1], [cuITensor(op)], s, Field))    
        op = isodd(row) ? (J_/2.0) * Z : (-J_/2.0) * Z   
        push!(H[row, Nx], Operator([row=>Nx], [cuITensor(op)], s, Field))    
    end
    for col in 2:Nx-1
        op = isodd(col) ? (J_/2.0) * Z : (-J_/2.0) * Z   
        push!(H[Ny, col], Operator([Ny=>col], [cuITensor(op)], s, Field))    
        op = isodd(col) ? (-J_/2.0) * Z : (J_/2.0) * Z   
        push!(H[1, col], Operator([1=>col], [cuITensor(op)], s, Field))    
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

function buildN(A::PEPS, L::Environments, R::Environments, IEnvs, row::Int, col::Int)::ITensor
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    N      = spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
    workingN = N
    if row > 1
        workingN *= IEnvs[:below][row - 1]
    end
    if row < Ny
        workingN *= IEnvs[:above][end - row]
    end
    if col > 1
        ci = commonindex(A[row, col], A[row, col-1])
        workingN *= multiply_side_ident(A[row, col], ci, copy(L.I[row])) 
    end
    if col < Nx
        ci = commonindex(A[row, col], A[row, col+1])
        workingN *= multiply_side_ident(A[row, col], ci, copy(R.I[row])) 
    end
    return workingN
end

function multiply_side_ident(A::ITensor, ci::Index, side_I::ITensor)
    is_gpu  = !(data(store(A)) isa Array)
    scmb = findindex(side_I, "Site")
    acmb = combiner(IndexSet(ci, ci'), tags="Site")
    delt = is_gpu ? cudelt(combinedindex(acmb), scmb) : δ(combinedindex(acmb), scmb)
    msi = side_I * delt * acmb
    return msi
end

function nonWorkRow(A::PEPS, L::Environments, R::Environments, H::Operator, row::Int, col::Int)::ITensor
    Ny, Nx  = size(A)
    op_rows = H.sites
    is_gpu  = !(data(store(A[1,1])) isa Array)
    ops     = deepcopy(H.ops)
    for op_ind in 1:length(ops)
        as = findindex(A[op_rows[op_ind][1][1], col], "Site")
        ops[op_ind] = replaceindex!(ops[op_ind], H.site_ind, as)
        ops[op_ind] = replaceindex!(ops[op_ind], H.site_ind', as')
    end
    op = spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
    op_ind = findfirst( x -> x == row, op_rows)
    AA = A[row, col] * op * dag(A[row, col])'
    if col > 1
        ci = commonindex(A[row, col], A[row, col-1])
        msi = multiply_side_ident(A[row, col], ci, L.I[row])
        AA *= msi
    end
    if col < Nx
        ci = commonindex(A[row, col], A[row, col+1])
        msi = multiply_side_ident(A[row, col], ci, R.I[row])
        AA *= msi 
    end
    return AA
end

function sum_rows_in_col(A::PEPS, L::Environments, R::Environments, H::Operator, row::Int, col::Int, low_row::Int, high_row::Int, above::Bool, IA::ITensor, IB::ITensor)::ITensor
    Ny, Nx = size(A)
    op_rows = H.sites
    is_gpu = !(data(store(A[1,1])) isa Array)
    ops = deepcopy(H.ops)
    start_row_ = row == op_rows[1][1] ? low_row + 1 : high_row
    stop_row_ = row == op_rows[1][1] ? high_row : low_row + 1
    start_row_ = min(start_row_, Ny)
    stop_row_ = min(stop_row_, Ny)
    step_row = row == op_rows[1][1] ? 1 : -1;
    start_row, stop_row = minmax(start_row_, stop_row_)
    op_row_a = H.sites[1][1]
    op_row_b = H.sites[2][1]
    for op_ind in 1:length(ops)
        this_A = A[op_rows[op_ind][1][1], col]
        as = findindex(this_A, "Site")
        ops[op_ind] = replaceindex!(ops[op_ind], H.site_ind, as)
        ops[op_ind] = replaceindex!(ops[op_ind], H.site_ind', as')
    end
    nwrs  = ITensor(1.0)
    nwrs_ = ITensor(1.0)
    Hterm = ITensor()
    if row == op_row_a
        Hterm = IA
        Hterm *= nonWorkRow(A, L, R, H, op_row_b, col)
        if col > 1
            ci  = commonindex(A[row, col], A[row, col-1])
            msi = multiply_side_ident(A[row, col], ci, copy(L.I[row]))
            Hterm *= msi 
        end
        if col < Nx
            ci  = commonindex(A[row, col], A[row, col+1])
            msi = multiply_side_ident(A[row, col], ci, copy(R.I[row]))
            Hterm *= msi
        end
        Hterm *= IB
    else
        Hterm = IB
        Hterm *= nonWorkRow(A, L, R, H, op_row_a, col)
        if col > 1
            ci  = commonindex(A[row, col], A[row, col-1])
            msi = multiply_side_ident(A[row, col], ci, copy(L.I[row]))
            Hterm *= msi 
        end
        if col < Nx
            ci  = commonindex(A[row, col], A[row, col+1])
            msi = multiply_side_ident(A[row, col], ci, copy(R.I[row]))
            Hterm *= msi
        end
        Hterm *= IA
    end
    op = spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
    op_ind = findfirst( x -> x[1] == row, op_rows)
    if op_ind > 0 
        op = ops[op_ind]
    end
    Hterm = Hterm*op
    return Hterm
end

function buildHIedge( A::PEPS, E::Environments, row::Int, col::Int, side )
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    HI = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    IH = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    next_col = side == :left ? 2 : Nx - 1
    for work_row in 1:row-1
        AA = A[work_row, col] * prime(A[work_row, col], "Link")
        ci = commonindex(A[work_row, col], A[work_row, next_col])
        cmb = findindex(E.I[work_row], "Site")
        acmb = combiner(IndexSet(ci, ci'), tags="Site")
        #delt = is_gpu ? cudelt(combinedindex(acmb), cmb) : δ(combinedindex(acmb), cmb)
        replaceindex!(acmb, combinedindex(acmb), cmb)
        AA *= acmb
        HI *= AA * E.I[work_row]
        IH *= E.H[work_row] * AA
    end
    ci = commonindex(A[row, col], A[row, next_col])
    cmb = findindex(E.I[row], "Site")
    acmb = combiner(IndexSet(ci, ci'), tags="Site")
    
    op = spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
    op = is_gpu ? cuITensor(op) : op
    HI *= op
    IH *= op
    #delt = is_gpu ? cudelt(combinedindex(acmb), cmb) : δ(combinedindex(acmb), cmb)
    replaceindex!(acmb, combinedindex(acmb), cmb)
    HI *= E.I[row] * acmb
    IH *= E.H[row] * acmb
    for work_row in row+1:Ny
        AA = A[work_row, col] * prime(A[work_row, col], "Link")
        ci = commonindex(A[work_row, col], A[work_row, next_col])
        cmb = findindex(E.I[work_row], "Site")
        acmb = combiner(IndexSet(ci, ci'), tags="Site")
        #delt = is_gpu ? cudelt(combinedindex(acmb), cmb) : δ(combinedindex(acmb), cmb)
        AA *= replaceindex!(acmb, combinedindex(acmb), cmb)
        HI *= AA * E.I[work_row]
        IH *= E.H[work_row] * AA
    end
    AAinds = IndexSet(inds(A[row, col]), inds(A[row, col]'))
    @assert hasinds(inds(IH), AAinds)
    @assert hasinds(AAinds, inds(IH))
    return (IH,)
end

function buildHIs(A::PEPS, L::Environments, R::Environments, row::Int, col::Int)
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    col == 1  && return buildHIedge(A, R, row, col, :left)
    col == Nx && return buildHIedge(A, L, row, col, :right)
    HLI_a = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    HLI_b = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    IHR_a = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    IHR_b = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    HLI   = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    IHR   = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    for work_row in 1:row-1
        AA = A[work_row, col] * prime(A[work_row, col], "Link")
        lci = commonindex(A[work_row, col], A[work_row, col-1])
        rci = commonindex(A[work_row, col], A[work_row, col+1])
        lcmb = findindex(L.I[work_row], "Site")
        rcmb = findindex(R.I[work_row], "Site")
        lacmb = combiner(IndexSet(lci, lci'), tags="Site")
        racmb = combiner(IndexSet(rci, rci'), tags="Site")
        replaceindex!(lacmb, combinedindex(lacmb), lcmb)
        replaceindex!(racmb, combinedindex(racmb), rcmb)
        AA *= lacmb
        AA *= racmb
        HLI_b *= L.H[work_row] * AA * R.I[work_row]
        IHR_b *= L.I[work_row] * AA * R.H[work_row]
    end
    lci = commonindex(A[row, col], A[row, col-1])
    rci = commonindex(A[row, col], A[row, col+1])
    lcmb = findindex(L.I[row], "Site")
    rcmb = findindex(R.I[row], "Site")
    lacmb = combiner(IndexSet(lci, lci'), tags="Site")
    racmb = combiner(IndexSet(rci, rci'), tags="Site")
    replaceindex!(lacmb, combinedindex(lacmb), lcmb)
    replaceindex!(racmb, combinedindex(racmb), rcmb)
    HLI  *= L.H[row] * lacmb 
    HLI  *= R.I[row] * racmb 
    IHR  *= L.I[row] * lacmb 
    IHR  *= R.H[row] * racmb 
    op = spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
    HLI *= op
    IHR *= op
    for work_row in reverse(row+1:Ny)
        AA = A[work_row, col] * prime(A[work_row, col], "Link")
        lci = commonindex(A[work_row, col], A[work_row, col-1])
        rci = commonindex(A[work_row, col], A[work_row, col+1])
        lcmb = findindex(L.I[work_row], "Site")
        rcmb = findindex(R.I[work_row], "Site")
        lacmb = combiner(IndexSet(lci, lci'), tags="Site")
        racmb = combiner(IndexSet(rci, rci'), tags="Site")
        replaceindex!(lacmb, combinedindex(lacmb), lcmb)
        replaceindex!(racmb, combinedindex(racmb), rcmb)
        AA *= lacmb
        AA *= racmb
        HLI_a *= L.H[work_row] * AA * R.I[work_row]
        IHR_a *= L.I[work_row] * AA * R.H[work_row]
    end
    HLI *= HLI_a
    HLI *= HLI_b
    IHR *= IHR_a
    IHR *= IHR_b
    AAinds = IndexSet(inds(A[row, col]), inds(A[row, col]'))
    @assert hasinds(inds(IHR), AAinds)
    @assert hasinds(inds(HLI), AAinds)
    @assert hasinds(AAinds, inds(IHR))
    @assert hasinds(AAinds, inds(HLI))
    return HLI, IHR 
end

function verticalTerms(A::PEPS, L::Environments, R::Environments, AI, AV, H, row::Int, col::Int)::Vector{ITensor} 
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    vTerms = ITensor[]#fill(ITensor(), length(H))
    AAinds = IndexSet(inds(A[row, col]), inds(A[row, col]'))
    for opcode in 1:length(H)
        thisVert = is_gpu ? cuITensor(1.0) : ITensor(1.0)
        op_row_a = H[opcode].sites[1][1]
        op_row_b = H[opcode].sites[2][1]
        if op_row_b < row || row < op_row_a
            local V, I
            if op_row_a > row
                V = AV[:above][opcode][end - row]
                I = row > 1 ? AI[:below][row - 1] : ITensor(1.0)
            elseif op_row_b < row
                V = AV[:below][opcode][row - op_row_b]
                I = row < Ny ? AI[:above][end - row] : ITensor(1.0)
            end
            thisVert *= V
            if col > 1
                ci = commonindex(A[row, col], A[row, col-1])
                msi = multiply_side_ident(A[row, col], ci, copy(L.I[row]))
                thisVert *= msi
            end
            if col < Nx
                ci = commonindex(A[row, col], A[row, col+1])
                msi = multiply_side_ident(A[row, col], ci, copy(R.I[row]))
                thisVert *= msi 
            end
            thisVert *= I
            thisVert *= spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
        elseif row == op_row_a
            low_row  = op_row_a - 1
            high_row = op_row_b
            AIL = low_row > 0 ? AI[:below][low_row] : ITensor(1)
            AIH = high_row < Ny ? AI[:above][end - high_row] : ITensor(1)
            thisVert = AIH
            if col > 1
                ci  = commonindex(A[op_row_b, col], A[op_row_b, col-1])
                msi = multiply_side_ident(A[op_row_b, col], ci, copy(L.I[op_row_b]))
                thisVert *= msi 
            end
            if col < Nx
                ci  = commonindex(A[op_row_b, col], A[op_row_b, col+1])
                msi = multiply_side_ident(A[op_row_b, col], ci, copy(R.I[op_row_b]))
                thisVert *= msi
            end
            sA = findindex(A[op_row_a, col], "Site")
            op_a = replaceindex!(copy(H[opcode].ops[1]), H[opcode].site_ind, sA)
            op_a = replaceindex!(op_a, H[opcode].site_ind', sA')
            sB = findindex(A[op_row_b, col], "Site")
            op_b = replaceindex!(copy(H[opcode].ops[2]), H[opcode].site_ind, sB)
            op_b = replaceindex!(op_b, H[opcode].site_ind', sB')
            thisVert *= A[op_row_b, col] * op_b * dag(A[op_row_b, col])'
            if col > 1
                ci  = commonindex(A[op_row_a, col], A[op_row_a, col-1])
                msi = multiply_side_ident(A[op_row_a, col], ci, copy(L.I[op_row_a]))
                thisVert *= msi 
            end
            if col < Nx
                ci  = commonindex(A[op_row_a, col], A[op_row_a, col+1])
                msi = multiply_side_ident(A[op_row_a, col], ci, copy(R.I[op_row_a]))
                thisVert *= msi
            end
            thisVert *= AIL 
            thisVert *= op_a
        elseif row == op_row_b
            low_row  = op_row_a - 1
            high_row = op_row_b
            AIL = low_row > 0 ? AI[:below][low_row] : ITensor(1)
            AIH = high_row < Ny ? AI[:above][end - high_row] : ITensor(1)
            thisVert = AIL
            if col > 1
                ci  = commonindex(A[op_row_a, col], A[op_row_a, col-1])
                msi = multiply_side_ident(A[op_row_a, col], ci, copy(L.I[op_row_a]))
                thisVert *= msi 
            end
            if col < Nx
                ci  = commonindex(A[op_row_a, col], A[op_row_a, col+1])
                msi = multiply_side_ident(A[op_row_a, col], ci, copy(R.I[op_row_a]))
                thisVert *= msi
            end
            sA = findindex(A[op_row_a, col], "Site")
            op_a = replaceindex!(copy(H[opcode].ops[1]), H[opcode].site_ind, sA)
            op_a = replaceindex!(op_a, H[opcode].site_ind', sA')
            sB = findindex(A[op_row_b, col], "Site")
            op_b = replaceindex!(copy(H[opcode].ops[2]), H[opcode].site_ind, sB)
            op_b = replaceindex!(op_b, H[opcode].site_ind', sB')
            thisVert *= A[op_row_a, col] * op_a * dag(A[op_row_a, col])'
            if col > 1
                ci  = commonindex(A[op_row_b, col], A[op_row_b, col-1])
                msi = multiply_side_ident(A[op_row_b, col], ci, copy(L.I[op_row_b]))
                thisVert *= msi 
            end
            if col < Nx
                ci  = commonindex(A[op_row_b, col], A[op_row_b, col+1])
                msi = multiply_side_ident(A[op_row_b, col], ci, copy(R.I[op_row_b]))
                thisVert *= msi
            end
            thisVert *= AIH 
            thisVert *= op_b
        end
        @assert hasinds(inds(thisVert), AAinds) "inds of thisVert and AAinds differ!\n$(inds(thisVert))\n$AAinds\n"
        @assert hasinds(AAinds, inds(thisVert)) "inds of thisVert and AAinds differ!\n$(inds(thisVert))\n$AAinds\n"
        if hasinds(AAinds, inds(thisVert)) && hasinds(inds(thisVert), AAinds)
            push!(vTerms, thisVert)
        end
    end
    return vTerms
end

function fieldTerms(A::PEPS, L::Environments, R::Environments, AI, AF, H, row::Int, col::Int)::Vector{ITensor} 
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    fTerms = fill(ITensor(), length(H))
    AAinds = IndexSet(inds(A[row, col]), inds(A[row, col]'))
    for opcode in 1:length(H)
        thisField = ITensor(1);
        op_row = H[opcode].sites[1][1];
        if op_row != row
            local F, I
            if op_row > row
                F = AF[:above][opcode][end - row]
                I = row > 1 ? AI[:below][row - 1] : ITensor(1.0)
            else
                F = AF[:below][opcode][row - 1]
                I = row < Ny ? AI[:above][end - row] : ITensor(1.0)
            end
            thisField *= F
            if col > 1
                ci = commonindex(A[row, col], A[row, col-1])
                thisField *= multiply_side_ident(A[row, col], ci, copy(L.I[row]))
            end
            if col < Nx
                ci = commonindex(A[row, col], A[row, col+1])
                thisField *= multiply_side_ident(A[row, col], ci, copy(R.I[row]))
            end
            thisField *= I
            thisField *= spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
        else
            low_row = op_row - 1
            high_row = op_row
            AIL = low_row > 0 ? AI[:below][low_row] : ITensor(1.0)
            AIH = high_row < Ny ? AI[:above][end - high_row] : ITensor(1.0)
            thisField = AIL
            if col > 1
                ci  = commonindex(A[row, col], A[row, col-1])
                msi = multiply_side_ident(A[row, col], ci, copy(L.I[row]))
                thisField *= msi 
            end
            if col < Nx
                ci  = commonindex(A[row, col], A[row, col+1])
                msi = multiply_side_ident(A[row, col], ci, copy(R.I[row]))
                thisField *= msi
            end
            thisField *= AIH
            sA = findindex(A[row, col], "Site")
            op = copy(H[opcode].ops[1])
            op = replaceindex!(op, H[opcode].site_ind, sA) 
            op = replaceindex!(op, H[opcode].site_ind', sA') 
            thisField *= op
        end
        @assert hasinds(inds(thisField), AAinds)
        @assert hasinds(AAinds, inds(thisField))
        fTerms[opcode] = thisField;
    end
    return fTerms
end

function connectLeftTerms(A::PEPS, L::Environments, R::Environments, AI, AL, H, row::Int, col::Int)::Vector{ITensor} 
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    lTerms = fill(ITensor(), length(H))
    AAinds = IndexSet(inds(A[row, col]), inds(A[row, col]'))
    for opcode in 1:length(H)
        op_row_b = H[opcode].sites[2][1]
        op_b = copy(H[opcode].ops[2])
        as   = findindex(A[op_row_b, col], "Site")
        op_b = replaceindex!(op_b, H[opcode].site_ind, as)
        op_b = replaceindex!(op_b, H[opcode].site_ind', as')
        thisHori = ITensor(1)
        if op_row_b != row
            local ancL, I
            if op_row_b > row
                ancL = AL[:above][opcode][end - row]
                I = row > 1 ? AL[:below][opcode][row - 1] : ITensor(1)
            else
                ancL = AL[:below][opcode][row - 1]
                I = row < Ny ? AL[:above][opcode][end - row] : ITensor(1)
            end
            thisHori = ancL
            thisHori *= L.InProgress[row, opcode]
            if col < Nx
                ci = commonindex(A[row, col], A[row, col+1])
                thisHori *= multiply_side_ident(A[row, col], ci, copy(R.I[row]))
            end
            thisHori *= I
            thisHori *= spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
        else
            low_row = (op_row_b <= row) ? op_row_b - 1 : row - 1;
            high_row = (op_row_b >= row) ? op_row_b + 1 : row + 1;
            if low_row >= 1
                thisHori *= AL[:below][opcode][low_row]
            end
            if high_row <= Ny
                thisHori *= AL[:above][opcode][end - high_row + 1]
            end
            if col < Nx
                ci = commonindex(A[row, col], A[row, col+1])
                thisHori *= multiply_side_ident(A[row, col], ci, R.I[row])
            end
            uih = uniqueinds(thisHori, L.InProgress[row, opcode])
            uil = uniqueinds(L.InProgress[row, opcode], thisHori)
            thisHori = thisHori * L.InProgress[row, opcode]
            thisHori = thisHori * op_b
        end
        @assert hasinds(inds(thisHori), AAinds)
        @assert hasinds(AAinds, inds(thisHori))
        lTerms[opcode] = thisHori
    end
    return lTerms
end

function connectRightTerms(A::PEPS, L::Environments, R::Environments, AI, AR, H, row::Int, col::Int)::Vector{ITensor} 
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    rTerms = fill(ITensor(), length(H))
    AAinds = IndexSet(inds(A[row, col]), inds(A[row, col]'))
    for opcode in 1:length(H)
        op_row_a = H[opcode].sites[1][1]
        op_a = copy(H[opcode].ops[1])
        as   = findindex(A[op_row_a, col], "Site")
        op_a = replaceindex!(op_a, H[opcode].site_ind, as)
        op_a = replaceindex!(op_a, H[opcode].site_ind', as')
        thisHori = ITensor(1)
        if op_row_a != row
            local ancR, I
            if op_row_a > row
                ancR = AR[:above][opcode][end - row]
                I = row > 1 ? AR[:below][opcode][row - 1] : ITensor(1)
            else
                ancR = AR[:below][opcode][row - 1]
                I = row < Ny ? AR[:above][opcode][end - row] : ITensor(1)
            end
            thisHori = ancR
            thisHori *= R.InProgress[row, opcode]
            if col > 1
                ci = commonindex(A[row, col], A[row, col-1])
                thisHori *= multiply_side_ident(A[row, col], ci, copy(L.I[row]))
            end
            thisHori *= I
            thisHori *= spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
        else
            low_row = (op_row_a <= row) ? op_row_a - 1 : row - 1;
            high_row = (op_row_a >= row) ? op_row_a + 1 : row + 1;
            if low_row >= 1
                thisHori *= AR[:below][opcode][low_row]
            end
            if high_row <= Ny
                thisHori *= AR[:above][opcode][end - high_row + 1]
            end
            if col > 1
                ci = commonindex(A[row, col], A[row, col-1])
                thisHori *= multiply_side_ident(A[row, col], ci, L.I[row])
            end
            thisHori *= R.InProgress[row, opcode]
            thisHori *= op_a 
        end
        @assert hasinds(inds(thisHori), AAinds)
        @assert hasinds(AAinds, inds(thisHori))
        rTerms[opcode] = thisHori
    end
    return rTerms
end

function buildLocalH(A::PEPS, L::Environments, R::Environments, AncEnvs, H, row::Int, col::Int )::Vector{ITensor}
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    field_H_terms = getDirectional(vcat(H[:, col]...), Field)
    vert_H_terms  = getDirectional(vcat(H[:, col]...), Vertical)

    N   = buildN(A, L, R, AncEnvs[:I], row, col)
    @debug "\t\tBuilding I*H and H*I row $row col $col"
    HIs = buildHIs(A, L, R, row, col)
    @debug "\t\tBuilding vertical H terms row $row col $col"
    vTs = verticalTerms(A, L, R, AncEnvs[:I], AncEnvs[:V], vert_H_terms, row, col) 
    @debug "\t\tBuilding field H terms row $row col $col"
    fTs = fieldTerms(A, L, R, AncEnvs[:I], AncEnvs[:F], field_H_terms, row, col)
    hTs = vcat(HIs..., vTs, fTs)
    Ny, Nx = size(A)
    if col > 1
        @debug "\t\tBuilding left H terms row $row col $col"
        left_H_terms = getDirectional(vcat(H[:, col - 1]...), Horizontal)
        lTs = connectLeftTerms(A, L, R, AncEnvs[:I], AncEnvs[:L], left_H_terms, row, col)
        @debug "\t\tBuilt left terms"
        hTs = vcat(hTs, lTs)
    end
    if col < Nx
        @debug "\t\tBuilding right H terms row $row col $col"
        right_H_terms = getDirectional(vcat(H[:, col]...), Horizontal)
        rTs = connectRightTerms(A, L, R, AncEnvs[:I], AncEnvs[:R], right_H_terms, row, col)
        @debug "\t\tBuilt right terms"
        hTs = vcat(hTs, rTs)
    end
    return vcat(hTs, N)
end

function intraColumnGauge(A::PEPS, col::Int; kwargs...)::PEPS
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    for row in reverse(2:Ny)
        @debug "\tBeginning intraColumnGauge for col $col row $row"
        Lis   = IndexSet(findindex(A[row, col], "Site"))
        if col > 1
            push!(Lis, commonindex(A[row, col], A[row, col - 1]))
        end
        if col < Nx
            push!(Lis, commonindex(A[row, col], A[row, col + 1]))
        end
        if row < Ny 
            push!(Lis, commonindex(A[row, col], A[row + 1, col]))
        end
        #U, S, V  = svd(A[row, col], Lis; tags="Link,u,c$col,r$(row-1)", kwargs...)
        cmb  = combiner(Lis, tags="CMB")
        ci = findindex(cmb, "CMB")
        U, S, V  = svd(A[row, col]*cmb, ci; kwargs...)
        A[row, col] = U*cmb
        A[row-1, col] *= S*V
    end
    return A
end

function buildAncs(A::PEPS, L::Environments, R::Environments, H, col::Int)
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    
    @debug "\tMaking ancillary identity terms for col $col"
    Ia = makeAncillaryIs(A, L, R, col)
    Ib = fill(ITensor(1.), Ny)
    Is = (above=Ia, below=Ib)

    @debug "\tMaking ancillary vertical terms for col $col"
    vH  = getDirectional(vcat(H[:, col]...), Vertical)
    Va = makeAncillaryVs(A, L, R, vH, col)
    Vb = [Vector{ITensor}() for ii in 1:length(Va)]
    Vs = (above=Va, below=Vb)

    @debug "\tMaking ancillary field terms for col $col"
    fH = getDirectional(vcat(H[:, col]...), Field)
    Fa = makeAncillaryFs(A, L, R, fH, col)
    Fb = [Vector{ITensor}() for ii in 1:length(Fa)]
    Fs = (above=Fa, below=Fb)

    Ls = (above=Vector{ITensor}(), below=Vector{ITensor}()) 
    Rs = (above=Vector{ITensor}(), below=Vector{ITensor}()) 
    if col > 1
        @debug "\tMaking ancillary left terms for col $col"
        lH = getDirectional(vcat(H[:, col-1]...), Horizontal)
        La = makeAncillarySide(A, L, R, lH, col, :left)
        Lb = [Vector{ITensor}() for ii in  1:length(La)]
        Ls = (above=La, below=Lb)
    end
    if col < Nx
        @debug "\tMaking ancillary right terms for col $col"
        rH = getDirectional(vcat(H[:, col]...), Horizontal)
        Ra = makeAncillarySide(A, R, L, rH, col, :right)
        Rb = [Vector{ITensor}() for ii in  1:length(Ra)]
        Rs = (above=Ra, below=Rb)
    end
    Ancs = (I=Is, V=Vs, F=Fs, L=Ls, R=Rs)
    return Ancs
end

function updateAncs(A::PEPS, L::Environments, R::Environments, AncEnvs, H, row::Int, col::Int)
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
   
    Is, Vs, Fs, Ls, Rs = AncEnvs
    @debug "\tUpdating ancillary identity terms for col $col row $row"
    Ib = updateAncillaryIs(A, Is[:below], L, R, row, col)
    Is = (above=Is[:above], below=Ib)

    @debug "\tUpdating ancillary vertical terms for col $col row $row"
    vH = getDirectional(vcat(H[:, col]...), Vertical)
    Vb = updateAncillaryVs(A, Vs[:below], Ib, L, R, vH, row, col)  
    Vs = (above=Vs[:above], below=Vb)

    @debug "\tUpdating ancillary field terms for col $col row $row"
    fH = getDirectional(vcat(H[:, col]...), Field)
    Fb = updateAncillaryFs(A, Fs[:below], Ib, L, R, fH, row, col)  
    Fs = (above=Fs[:above], below=Fb)

    if col > 1
        @debug "\tUpdating ancillary left terms for col $col row $row"
        lH = getDirectional(vcat(H[:, col-1]...), Horizontal)
        Lb = updateAncillarySide(A, Ls[:below], Ib, L, R, lH, row, col, :left)
        Ls = (above=Ls[:above], below=Lb)
    end
    if col < Nx
        @debug "\tUpdating ancillary right terms for col $col row $row"
        rH = getDirectional(vcat(H[:, col]...), Horizontal)
        Rb = updateAncillarySide(A, Rs[:below], Ib, R, L, rH, row, col, :right)
        Rs = (above=Rs[:above], below=Rb)
    end
    Ancs = (I=Is, V=Vs, F=Fs, L=Ls, R=Rs)
    return Ancs
end

function optimizeLocalH(A::PEPS, L::Environments, R::Environments, AncEnvs, H, row::Int, col::Int; kwargs...)
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    field_H_terms = getDirectional(vcat(H[:, col]...), Field)
    vert_H_terms  = getDirectional(vcat(H[:, col]...), Vertical)
    @debug "\tBuilding H for col $col row $row"
    Hs = buildLocalH(A, L, R, AncEnvs, H, row, col)
    N = Hs[end]
    initial_N = (A[row, col] * N * dag(A[row, col])')
    localH = sum(Hs[1:end-1])
    initial_E = A[row, col] * localH * dag(A[row, col])'
    println()
    println("Initial energy at row $row col $col : $(scalar(initial_E)/scalar(initial_N))")
    println("Initial norm at row $row col $col : $(scalar(initial_N))")
    @debug "\tBeginning davidson for col $col row $row"
    λ, new_A = davidson(localH, A[row, col]; kwargs...)
    new_E = new_A * localH * dag(new_A)'
    new_N = new_A * N * dag(new_A)'
    println("Optimized energy at row $row col $col : $(scalar(new_E)/(scalar(new_N)))")
    println("Optimized norm at row $row col $col : $(scalar(new_N))")
    if row < Ny
        @debug "\tRestoring intraColumnGauge for col $col row $row"
        Lis   = IndexSet(findindex(new_A, "Site"))
        if col > 1
            push!(Lis, commonindex(A[row, col], A[row, col - 1]))
        end
        if col < Nx
            push!(Lis, commonindex(A[row, col], A[row, col + 1]))
        end
        if row > 1
            push!(Lis,  commonindex(A[row, col], A[row - 1, col]))
        end
        old_ci = commonindex(A[row, col], A[row+1, col])
        U, V  = factorize(new_A, Lis; dir="fromleft", which_factorization="svd", tags="Link,u,c$col,r$row", kwargs...)
        new_ci = commonindex(U, V)
        A[row, col] = replaceindex!(U, new_ci, old_ci)
        A[row+1, col] *= V
        A[row+1, col] = replaceindex!(A[row+1, col], new_ci, old_ci)
        if row < Ny - 1
            nI = spinI(findindex(A[row+1, col], "Site"); is_gpu=is_gpu)
            newAA = A[row+1,col] * nI * dag(A[row+1,col])'
            if col > 1
                ci     = commonindex(A[row+1, col], A[row+1, col-1])
                newAA *= multiply_side_ident(A[row+1, col], ci, L.I[row+1])
            end
            if col < Nx
                ci     = commonindex(A[row+1, col], A[row+1, col+1])
                newAA *= multiply_side_ident(A[row+1, col], ci, R.I[row+1])
            end
            AncEnvs[:I][:above][end - row] = newAA * AncEnvs[:I][:above][end - row - 1]
        end
    else
        A[row, col] = (scalar(initial_N) > 0) ? new_A/√scalar(initial_N) : new_A
    end
    return A, AncEnvs
end

function measureEnergy(A::PEPS, L::Environments, R::Environments, AncEnvs, H, row::Int, col::Int)::Float64
    Ny, Nx = size(A)
    Hs = buildLocalH(A, L, R, AncEnvs, H, row, col)
    N = Hs[end]
    initial_N = (A[row, col] * N * dag(A[row, col])')
    localH    = sum(Hs[1:end-1])
    initial_E = A[row, col] * localH * dag(A[row, col])'
    return scalar(initial_E)/scalar(initial_N)
end

function sweepColumn(A::PEPS, L::Environments, R::Environments, H, col::Int; kwargs...)
    Ny, Nx = size(A)
    @debug "Beginning intraColumnGauge for col $col" 
    A = intraColumnGauge(A, col; kwargs...)
    @debug "Beginning buildAncs for col $col" 
    AncEnvs = buildAncs(A, L, R, H, col)
    for row in 1:Ny
        if row > 1
            @debug "Beginning updateAncs for col $col" 
            AncEnvs = updateAncs(A, L, R, AncEnvs, H, row-1, col) 
        end
        @debug "Beginning optimizing H for col $col" 
        A, AncEnvs = optimizeLocalH(A, L, R, AncEnvs, H, row, col; kwargs...)
    end
    return A
end

function rightwardSweep(A::PEPS, Ls::Vector{Environments}, Rs::Vector{Environments}, H; kwargs...)
    Ny, Nx = size(A)
    dummyI = MPS(Ny, fill(ITensor(1.0), Ny), 0, Ny+1)
    dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny)) 
    prev_cmb_r = fill(ITensor(1.0), Ny)
    next_cmb_r = fill(ITensor(1.0), Ny)
    for col in 1:Nx-1
        L = col == 1 ? dummyEnv : Ls[col - 1]
        @debug "Sweeping col $col"
        A, tg, bytes, gctime, memallocs = @timed sweepColumn(A, L, Rs[col+1], H, col; kwargs...)
        @debug "Time to sweep column $col $tg"
        # Simple update...
        # ....
        # Gauge
        @debug "Gauging col $col"
        #A, tg, bytes, gctime, memallocs = @timed gaugeColumn(A, col, :right; kwargs...)
        @debug "Time to gauge $tg"
        if col == 1
            left_H_terms = getDirectional(H[1], Horizontal)
            Ls[col], tg, bytes, gctime, memallocs = @timed buildEdgeEnvironment(A, H, left_H_terms, prev_cmb_r, :left, 1; kwargs...)
            @debug "Time to build edge $tg"
        else
            Ls[col], tg, bytes, gctime, memallocs = @timed buildNextEnvironment(A, Ls[col-1], H, prev_cmb_r, next_cmb_r, :left, col; kwargs...)
            @debug "Time to build next edge at col $col $tg"
            prev_cmb_r = deepcopy(next_cmb_r)
        end
    end
    return A, Ls, Rs
end

function leftwardSweep(A::PEPS, Ls::Vector{Environments}, Rs::Vector{Environments}, H; kwargs...)
    Ny, Nx = size(A)
    dummyI = MPS(Ny, fill(ITensor(1.0), Ny), 0, Ny+1)
    dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny)) 
    prev_cmb_l = fill(ITensor(1.0), Ny)
    next_cmb_l = fill(ITensor(1.0), Ny)
    for col in reverse(2:Nx)
        R = col == Nx ? dummyEnv : Rs[col + 1]
        @debug "Sweeping col $col"
        A, tg, bytes, gctime, memallocs = @timed sweepColumn(A, Ls[col - 1], R, H, col; kwargs...)
        @debug "Time to sweep column $col $tg"
        # Simple update...
        # ....
        # Gauge
        @debug "Gauging col $col"
        #A, tg, bytes, gctime, memallocs = @timed gaugeColumn(A, col, :left; kwargs...) 
        @debug "Time to gauge $tg"
        if col == Nx
            right_H_terms = getDirectional(H[Nx - 1], Horizontal)
            Rs[col], tg, bytes, gctime, memallocs = @timed buildEdgeEnvironment(A, H, right_H_terms, prev_cmb_l, :right, Nx; kwargs...)
            @debug "Time to build edge $tg"
        else
            Rs[col], tg, bytes, gctime, memallocs = @timed buildNextEnvironment(A, Rs[col+1], H, prev_cmb_l, next_cmb_l, :right, col; kwargs...)
            @debug "Time to build next edge at col $col $tg"
            prev_cmb_l = deepcopy(next_cmb_l)
        end
    end
    return A, Ls, Rs
end
