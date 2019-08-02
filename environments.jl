struct Environments
    I::MPS
    H::MPS
    InProgress::Matrix{ITensor}
end

function buildEdgeEnvironment(A::PEPS, H, left_H_terms, next_combiners::Vector{ITensor}, side::Symbol, col::Int; kwargs...)::Environments
    Ny, Nx = size(A)
    up_combiners = Vector{ITensor}(undef, Ny-1)
    fake_next_combiners = Vector{ITensor}(undef, Ny)
    fake_prev_combiners = fill(ITensor(1), Ny)
    I_mpo  = buildNewI(A, col, fake_prev_combiners, fake_next_combiners, up_combiners, side)
    copyto!(next_combiners, fake_next_combiners)
    I_mps  = MPS(Ny, tensors(I_mpo), 0, Ny+1)
    field_H_terms = getDirectional(vcat(H[:, col]...), Field)
    vert_H_terms  = getDirectional(vcat(H[:, col]...), Vertical)
    vHs           = [buildNewVerticals(A, fake_prev_combiners, fake_next_combiners, up_combiners, vert_H_terms[vert_op], col, kwargs...) for vert_op in 1:length(vert_H_terms)]
    fHs           = [buildNewFields(A, fake_prev_combiners, fake_next_combiners, up_combiners, field_H_terms[field_op], col, kwargs...) for field_op in 1:length(field_H_terms)]
    Hs            = MPS[MPS(Ny, tensors(H_term), 0, Ny+1) for H_term in vcat(vHs, fHs)]
    for row in 1:Ny-1
        ci = linkindex(I_mps, row)
        ni = Index(dim(ci), "u,Link,c$col,r$row")
        replaceindex!(I_mps[row], ci, ni) 
        replaceindex!(I_mps[row+1], ci, ni)
    end
    # kwargs...
    H_overall    = reduce(sum, Hs)
    side_H       = side == :left ? H[:, col] : H[:, col - 1]
    side_H_terms = getDirectional(vcat(side_H...), Horizontal)
    in_progress  = Matrix{ITensor}(undef, Ny, length(side_H_terms))
    for side_term in 1:length(side_H_terms)
        in_progress[1:Ny, side_term] = generateEdgeDanglingBonds(A, up_combiners, side_H_terms[side_term], side, col)
    end
    return Environments(I_mps, H_overall, in_progress)
end

function buildNextEnvironment(A::PEPS, prev_Env::Environments, H, previous_combiners::Vector{ITensor}, next_combiners::Vector{ITensor}, side::Symbol, col::Int; kwargs...)::Environments
    Ny, Nx = size(A)
    working_combiner = Vector{ITensor}(undef, Ny)
    up_combiners     = Vector{ITensor}(undef, Ny-1)
    I_mpo            = buildNewI(A, col, previous_combiners, working_combiner, up_combiners, side)
    copyto!(next_combiners, working_combiner)
    for row in 1:Ny-1
        ci = linkindex(I_mpo, row)
        ni = Index(dim(ci), "u,Link,c$col,r$row")
        replaceindex!(I_mpo[row], ci, ni) 
        replaceindex!(I_mpo[row+1], ci, ni)
    end
    new_I         = applyMPO(I_mpo, prev_Env.I, kwargs...)
    new_H         = applyMPO(I_mpo, prev_Env.H, kwargs...)
    field_H_terms = getDirectional(vcat(H[:, col]...), Field)
    vert_H_terms  = getDirectional(vcat(H[:, col]...), Vertical)
    hori_H_terms  = getDirectional(vcat(H[:, col]...), Horizontal)
    side_H        = side == :left ? H[:, col] : H[:, col - 1]
    side_H_terms  = getDirectional(vcat(side_H...), Horizontal)
    H_term_count  = 1 + length(field_H_terms) + length(vert_H_terms) + (side == :left ? length(side_H_terms) : length(hori_H_terms))

    new_H_mps     = Vector{MPS}(undef, H_term_count)
    new_H_mps[1]  = deepcopy(new_H)
    vHs = [buildNewVerticals(A, previous_combiners, next_combiners, up_combiners, vert_H_terms[vert_op], col, kwargs...) for vert_op in 1:length(vert_H_terms)]
    fHs = [buildNewFields(A, previous_combiners, next_combiners, up_combiners, field_H_terms[field_op], col, kwargs...) for field_op in 1:length(field_H_terms)]
    new_H_mps[2:length(vert_H_terms) + length(field_H_terms) + 1] = [applyMPO(H_term, prev_Env.I, kwargs...) for H_term in vcat(vHs, fHs)]

    connect_H    = side == :left ? side_H_terms : hori_H_terms
    new_Hs = [connectDanglingBonds(A, next_combiners, up_combiners, connect_H_term, prev_Env.InProgress[:, term_ind], side, -1, col, kwargs...) for (term_ind, connect_H_term) in enumerate(connect_H)]
    new_H_mps[length(vert_H_terms) + length(field_H_terms) + 2:end] = [MPS(Ny, new_H, 0, Ny+1) for new_H in new_Hs]

    H_overall    = reduce(sum, new_H_mps)
    #H_overall    = sum(new_H_mps, kwargs...)
    gen_H_terms  = side == :left ? hori_H_terms : side_H_terms
    in_progress  = Matrix{ITensor}(undef, Ny, length(side_H_terms))
    for side_term in 1:length(gen_H_terms)
        in_progress[1:Ny, side_term] = generateNextDanglingBonds(A, previous_combiners, next_combiners, up_combiners, gen_H_terms[side_term], prev_Env.I, side, col, kwargs...)
    end
    return Environments(new_I, H_overall, in_progress)
end

function buildNewVerticals(A::PEPS, previous_combiners::Vector{ITensor}, next_combiners::Vector{ITensor}, up_combiners::Vector{ITensor}, H, col::Int)::MPO
    Ny, Nx = size(A)
    col_site_inds       = [findindex(A[row, col], "Site") for row in 1:Ny]
    ops                 = ITensor[spinI(spin_ind) for spin_ind in col_site_inds] 
    vertical_row_a      = H.sites[1][1]
    vertical_row_b      = H.sites[2][1]
    ops[vertical_row_a] = replaceindex!(copy(H.ops[1]), H.site_ind, col_site_inds[vertical_row_a]) 
    ops[vertical_row_a] = replaceindex!(ops[vertical_row_a], H.site_ind', col_site_inds[vertical_row_a]') 
    ops[vertical_row_b] = replaceindex!(copy(H.ops[2]), H.site_ind, col_site_inds[vertical_row_b])
    ops[vertical_row_b] = replaceindex!(ops[vertical_row_b], H.site_ind', col_site_inds[vertical_row_b]') 
    internal_cmb_u      = vcat(ITensor(1.0), up_combiners, ITensor(1.0))
    AAs                 = [ A[row, col] * ops[row] * prime(dag(A[row, col])) * next_combiners[row] * previous_combiners[row] * internal_cmb_u[row] * internal_cmb_u[row+1] for row in 1:Ny ]
    return MPO(Ny, AAs, 0, Ny+1)
end

function buildNewFields(A::PEPS, previous_combiners::Vector{ITensor}, next_combiners::Vector{ITensor}, up_combiners::Vector{ITensor}, H)::MPO
    Ny, Nx = size(A)
    col_site_inds       = [findindex(A[row, col], "Site") for row in 1:Ny]
    ops                 = ITensor[spinI(spin_ind) for spin_ind in col_site_inds] 
    field_row      = H.sites[1][1]
    ops[field_row] = replaceindex!(copy(H.ops[1]), H.site_ind, col_site_inds[field_row]) 
    ops[field_row] = replaceindex!(ops[field_row], H.site_ind', col_site_inds[field_row]') 
    internal_cmb_u = vcat(ITensor(1.0), up_combiners, ITensor(1.0))
    AAs            = [ A[row, col] * ops[row] * prime(dag(A[row, col])) * next_combiners[row] * previous_combiners[row] * internal_cmb_u[row] * internal_cmb_u[row+1] for row in 1:Ny ]
    return MPO(Ny, AAs, 0, Ny+1)
end

function buildNewI(A::PEPS, col::Int, previous_combiners::Vector{ITensor}, next_combiners::Vector{ITensor}, up_combiners::Vector{ITensor}, side::Symbol)::MPO
    Ny, Nx = size(A)
    Iapp = MPO(Ny)
    next_col = side == :left ? col + 1 : col - 1 # side is handedness of environment
    @inbounds for row in 1:Ny
        AA = A[row, col] * prime(dag(A[row, col]), "Link")
        next_combiners[row], AA = combine(AA, A[row, col], A[row, next_col], "Site,r$row,c$col")
        AA *= previous_combiners[row]
        if row > 1
            AA *= up_combiners[row-1]
        end
        if row < Ny
            up_combiners[row], AA = combine(AA, A[row, col], A[row+1, col], "Link,CMB,c$col,r$row")
        end
        Iapp[row] = AA
    end
    return Iapp
end

function generateEdgeDanglingBonds(A::PEPS, up_combiners::Vector{ITensor}, H, side::Symbol, col::Int)::Vector{ITensor}
    Ny, Nx = size(A)
    op_row         = side == :left ? H.sites[1][1] : H.sites[2][1];
    H_op           = side == :left ? H.ops[1] : H.ops[2];
    col_site_inds       = [findindex(A[row, col], "Site") for row in 1:Ny]
    ops                 = ITensor[spinI(spin_ind) for spin_ind in col_site_inds] 
    ops[op_row]    = replaceindex!(copy(H_op), H.site_ind, col_site_inds[op_row]) 
    ops[op_row]    = replaceindex!(ops[op_row], H.site_ind', col_site_inds[op_row]') 
    internal_cmb_u = vcat(ITensor(1.0), up_combiners, ITensor(1.0))
    this_IP        = [A[row, col] * ops[row] * prime(dag(A[row, col])) * internal_cmb_u[row] * internal_cmb_u[row+1] for row in 1:Ny]
    return this_IP;
end

function generateNextDanglingBonds(A::PEPS, previous_combiners::Vector{ITensor}, next_combiners::Vector{ITensor}, up_combiners::Vector{ITensor}, H, Ident::MPS, side::Symbol, col::Int; kwargs...)::Vector{ITensor}
    Ny, Nx = size(A)
    op_row = side == :left ? H.sites[1][1] : H.sites[2][1]
    H_op   = side == :left ? H.ops[1] : H.ops[2]
    col_site_inds       = [findindex(A[row, col], "Site") for row in 1:Ny]
    ops                 = ITensor[spinI(spin_ind) for spin_ind in col_site_inds] 
    ops[op_row]     = replaceindex!(copy(H_op), H.site_ind, col_site_inds[op_row]) 
    ops[op_row]     = replaceindex!(ops[op_row], H.site_ind', col_site_inds[op_row]') 
    internal_cmb_u  = vcat(ITensor(1.0), up_combiners, ITensor(1.0))
    this_IP         = [A[row, col] * ops[row] * prime(dag(A[row, col])) * previous_combiners[row] * next_combiners[row] * internal_cmb_u[row] * internal_cmb_u[row+1] for row in 1:Ny]
    in_progress_MPO = MPO(Ny, this_IP, 0, Ny+1)
    result          = applyMPO(in_progress_MPO, Ident, kwargs...)
    return ITensor[result[row]*next_combiners[row] for row in 1:Ny]
end

function connectDanglingBonds(A::PEPS, next_combiners::Vector{ITensor}, up_combiners::Vector{ITensor}, oldH, in_progress::Vector{ITensor}, side::Symbol, work_row::Int, col::Int; kwargs...)::Vector{ITensor}
    Ny, Nx = size(A)
    op_row_a = oldH.sites[1][1]
    op_row_b = oldH.sites[2][1]
    op       = side == :left ? oldH.ops[2] : oldH.ops[1]
    application_row = side == :left ? op_row_b : op_row_a
    col_site_inds  = [findindex(A[row, col], "Site") for row in 1:Ny]
    ops            = ITensor[spinI(spin_ind) for spin_ind in col_site_inds] 
    ops[application_row]   = replaceindex!(copy(op), oldH.site_ind, col_site_inds[application_row])
    ops[application_row]   = replaceindex!(ops[application_row], oldH.site_ind', col_site_inds[application_row]')
    internal_cmb_u = vcat(ITensor(1.0), up_combiners, ITensor(1.0))
    in_prog_mps    = MPS(Ny, in_progress, 0, Ny + 1)
    this_IP        = [A[row, col] * ops[row] * prime(dag(A[row, col])) * next_combiners[row] * internal_cmb_u[row] * internal_cmb_u[row+1] for row in 1:Ny]
    if 0 < work_row < Ny + 1
        this_IP[work_row] = ops[work_row]
    end
    completed_H = MPO(Ny, this_IP, 0, Ny+1)
    if work_row == -1
        dummy_cmbs     = [combiner(commoninds(completed_H[row], in_prog_mps[row]), tags="Site,r$row") for row in 1:Ny]
        completed_H.A_ = dummy_cmbs .* tensors(completed_H)
        ci = Vector{Index}(undef, Ny)
        si = Vector{Index}(undef, Ny)
        for row in 1:Ny
            ci[row] = commonindex(dummy_cmbs[row], completed_H[row])
            si[row] = findindex(completed_H[row], "Site,c$col")
            completed_H[row] = replaceindex!(completed_H[row], si[row], ci[row]')
        end
        in_prog_mps.A_ = dummy_cmbs .* tensors(in_prog_mps)
        result         = applyMPO(completed_H, in_prog_mps, kwargs...)
        for row in 1:Ny
            result[row] = replaceindex!(result[row], ci[row], si[row])
        end 
        return tensors(result)
    else
        for row in 1:Ny-1
            ci = linkIndex(completed_H, row)
            ni = Index(dim(ci), "u,Link,c$col,r$row")
            replaceIndex!(completed_H[row], ci, ni) 
            replaceIndex!(completed_H[row+1], ci, ni) 
        end
        return tensors(completed_H) .* tensors(in_prog_mps)
    end
end

function buildLs(A::PEPS, H; kwargs...)
    Ny, Nx = size(A)
    previous_combiners = Vector{ITensor}(undef, Ny)
    next_combiners     = Vector{ITensor}(undef, Ny)
    Ls                 = Vector{Environments}(undef, Nx)
    start_col::Int     = get(kwargs, :start_col, 1)
    if start_col == 1
        left_H_terms = getDirectional(H[1], Horizontal)
        Ls[1] = buildEdgeEnvironment(A, H, left_H_terms, previous_combiners, :left, 1, kwargs...)
    elseif start_col - 1 > 1
        previous_combiners = [reconnect(commonindex(A[row, start_col], A[row, start_col - 1]), Ls[start_col-1].I[row]) for row in 1:Ny]
    end
    loop_col = start_col == 1 ? 2 : start_col
    for col in loop_col:Nx-1
        Ls[col] = buildNextEnvironment(A, Ls[col-1], H, previous_combiners, next_combiners, :left, col, kwargs...)
        previous_combiners = deepcopy(next_combiners)
    end
    return Ls
end

function buildRs(A::PEPS, H; kwargs...)
    Ny, Nx = size(A)
    previous_combiners = Vector{ITensor}(undef, Ny)
    next_combiners     = Vector{ITensor}(undef, Ny)
    start_col::Int     = get(kwargs, :start_col, Nx)
    Rs                 = Vector{Environments}(undef, Nx)
    if start_col == Nx
        right_H_terms = getDirectional(H[Nx-1], Horizontal)
        Rs[Nx] = buildEdgeEnvironment(A, H, right_H_terms, previous_combiners, :right, Nx, kwargs...)
    elseif start_col + 1 < Nx
        previous_combiners = [reconnect(commonindex(A[row, start_col], A[row, start_col + 1]), Rs[start_col+1].I[row]) for row in 1:Ny]
    end
    loop_col = start_col == Nx ? Nx-1 : start_col
    for col in reverse(2:loop_col)
        Rs[col] = buildNextEnvironment(A, Rs[col+1], H, previous_combiners, next_combiners, :right, col, kwargs...)
        previous_combiners = deepcopy(next_combiners)
    end
    return Rs
end