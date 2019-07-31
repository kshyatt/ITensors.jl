using ITensors

mutable struct PEPS
    Nx::Int
    Ny::Int
    A_::Matrix{ITensor}

    PEPS() = new(0, 0, Matrix{ITensor}(),0,0)

    function PEPS(Nx::Int, Ny::Int 
                  A::Matrix{ITensor}) 
        new(Nx, Ny, A)
    end
    function PEPS(sites::SiteSet, lattice::Lattice)
        Nx = count(unique(x->x.x1, lattice)) 
        Ny = count(unique(x->x.x2, lattice))
        p  = Matrix{ITensor}(undef, Nx, Ny)
        right_links = [ Index(1, "Link,c$i,r$j,r") for i in 1:Nx, j in 1:Ny ]
        up_links    = [ Index(1, "Link,c$i,r$j,u") for i in 1:Nx, j in 1:Ny ]
        @inbounds for ii in eachindex(sites)
            row = div(ii, Nx) + 1
            col = mod(ii, Nx) + 1
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

function randomPEPS(sites, lattice)
    A = PEPS(sites)
    @inbounds for i ∈ eachindex(sites)
        row = div(ii, Nx) + 1
        col = mod(ii, Nx) + 1
        randn!(A[row, col])
        normalize!(A[row, col])
    end
    return A
end

tensors(A::PEPS) = A.A_
size(A::PEPS) = (A.Nx, A.Ny)
getindex(A::PEPS, i::Integer, j::Integer) = getindex(tensors(A), i, j)
setindex!(A::PEPS, val::ITensor, i::Integer, j::Integer) = setindex!(tensors(A), T, i, j)

copy(A::PEPS)    = PEPS(A.Nx, A.Ny, copy(tensors(A)))
similar(A::PEPS) = PEPS(A.Nx, A.Ny, similar(tensors(A)))

function show(io::IO, A::PEPS)
  print(io,"PEPS")
  (size(A)[1] > 0 && size(A)[2] > 0) && print(io,"\n")
  @inbounds for i in 1:A.Nx, j in 1:A.Ny
      println(io,"$i $j $(inds(A[i,j]))")
  end
end

struct Environments
    I::MPO
    H::MPO
    InProgress::Matrix{ITensor}
end

function combine(AA::ITensor, Aorig::ITensor, Anext::ITensor, tags::String)
    ci = commonIndex(Aorig, Anext)
    cmb = combiner(IndexSet(ci, prime(ci)), tags=tags)
    AA *= cmb
    return cmb, AA
end

function buildEdgeEnvironment(A::PEPS, H, left_H_terms, next_combiners::Vector{ITensor}, side::Symbol, col::Int; kwargs...)::Payload

end

function buildNextEnvironment(A::PEPS, prev_Env::Payload, H, previous_combiners::Vector{ITensor}, next_combiners::Vector{ITensor}, side::Symbol, col::Int; kwargs...)::Payload

end

function buildNewVerticals(A::PEPS, previous_combiners::Vector{ITensor}, next_combiners::Vector{ITensor}, up_combiners::Vector{ITensor}, H)

end

function buildNewFields(A::PEPS, previous_combiners::Vector{ITensor}, next_combiners::Vector{ITensor}, up_combiners::Vector{ITensor}, H)

end

function buildNewI(A::PEPS, col::Int, previous_combiners::Vector{ITensor}, next_combiners::Vector{ITensor}, up_combiners::Vector{ITensor}, side::Symbol)::MPO
    Nx, Ny = size(A)
    MPO Iapp(Ny)
    next_col = side == :left ? col + 1 : col - 1 # side is handedness of environment
    @inbounds for row in 1:Ny
        AA = A[row, col] * prime(dag(A[row, col]), "Link")
        next_combiners[row], AA = combine(AA, A[row, col], A[row, next_col], "Site,r$row,c$col")
        AA *= previous_combiners[row]
        if row > 0
            AA *= up_combiners[row - 1]
        end
        if row < Ny
            up_combiners[row], AA = combine(AA, A[row, col], A[row+1, col], "Link,CMB,c$col,r$row")
        end
        Iapp[row] = AA
    end
    return Iapp
end

function generateEdgeDanglingBonds()

end

function generateNextDanglingBonds()

end

function connectDanglingBonds()

end

function buildLs(A::PEPS, H; kwargs...)
    Nx, Ny = size(A)
    previous_combiners = Vector{ITensor}(undef, Ny)
    next_combiners = Vector{ITensor}(undef, Ny)
    Ls = Vector{Payload}(undef, Nx)
    start_col::Int = get(kwargs, :start_col, 1)
    if start_col == 1
        left_H_terms = directionalH(H[1], Horizontal)
        Ls[1] = buildEdgeEnvironment(A, H, left_H_terms, previous_combiners, :left, 1, kwargs...)
    else
        if start_col - 1 > 1
            for row in 1:Ny
                previous_combiners[row] = reconnect( )
                #prev_cmb_r[row] = reconnect(commonIndex(As[start_col][row], As[prev_col][row]), Ls[prev_col].I(row+1));
            end
        end
    end
    loop_col = start_col == 1 ? 2 : start_col
    for col in loop_col:Nx-1
        Ls[col] = buildNextEnvironment(A, Ls[col-1], H, previous_combiners, next_combiners, :left, col, kwargs...)
        previous_combiners = deepcopy(next_combiners)
    end
    return Ls
end

function buildRs(A::PEPS, H; kwargs...)
    Nx, Ny = size(A)
    previous_combiners = Vector{ITensor}(undef, Ny)
    next_combiners = Vector{ITensor}(undef, Ny)
    start_col::Int = get(kwargs, :start_col, Nx)
    Rs = Vector{Payload}(undef, Nx)
    if start_col == Nx
        right_H_terms = directionalH(H[Nx-1], Horizontal)
        Rs[Nx] = buildEdgeEnvironment(A, H, right_H_terms, previous_combiners, :right, Nx, kwargs...)
    else
        if start_col + 1 < Nx
            for row in 1:Ny
                previous_combiners[row] = reconnect( )
                #prev_cmb_l[row] = reconnect(commonIndex(As[prev_col][row], As[start_col][row]), Rs[prev_col].I(row+1));
            end
        end
    end
    loop_col = start_col == Nx ? Nx-1 : start_col
    for col in reverse(2:loop_col)
        Rs[col] = buildNextEnvironment(A, Rs[col+1], H, previous_combiners, next_combiners, :right, col, kwargs...)
        previous_combiners = deepcopy(next_combiners)
    end
    return Rs
end
