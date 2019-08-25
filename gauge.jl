using ITensors, ITensors.CuITensors

function initQs( A::PEPS, col::Int, next_col::Int; kwargs...)
    Ny, Nx = size(A)
    maxdim::Int = get(kwargs, :maxdim, 1)
    Q         = MPO(Ny, deepcopy(A[:, col]), 0, Ny+1)
    prev_col  = next_col > col ? col - 1 : col + 1
    QR_inds   = [Index(maxdim, "Site,QR,c$col,r$row") for row in 1:Ny]
    A_up_inds = [commonindex(A[row, col], A[row+1, col]) for row in 1:Ny-1]
    Q_up_inds = [Index(dim(A_up_inds[row]), "Link,u,Qup$row") for row in 1:Ny-1]
    next_col_inds = [commonindex(A[row, col], A[row, next_col]) for row in 1:Ny]
    prev_col_inds = 0 < prev_col < Nx ? [commonindex(A[row, col], A[row, prev_col]) for row in 1:Ny] : Vector{Index}(undef, Ny)
    for row in 1:Ny
        row < Ny && replaceindex!(Q[row], A_up_inds[row], Q_up_inds[row])
        row > 1  && replaceindex!(Q[row], A_up_inds[row-1], Q_up_inds[row-1])
        replaceindex!(Q[row], next_col_inds[row], QR_inds[row])
    end
    return Q, QR_inds, next_col_inds
end

function gaugeQR(A::PEPS, col::Int, side::Symbol; kwargs...)
    overlap_cutoff::Real = get(kwargs, :overlap_cutoff, 1e-4)
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    prev_col_inds = Vector{Index}(undef, Ny)

    next_col = side == :left ? col - 1 : col + 1
    prev_col = side == :left ? col + 1 : col - 1
    Q, QR_inds, next_col_inds = initQs(A, col, next_col; kwargs...)
    left_edge  = col == 1
    right_edge = col == Nx
    ratio_history = Vector{Float64}() 
    ratio         = 0.0
    best_overlap  = 0.0
    Ampo   = MPO(Ny)
    best_Q = MPO(Ny)
    best_R = MPO(Ny)
    iter   = 1
    dummy_nexts = [Index(dim(next_col_inds[row]), "DM,Site,r$row") for row in 1:Ny]
    cmb_l = Vector{ITensor}(undef, Ny)
    for iter in 1:50
        thisTerm  = Vector{ITensor}(undef, Ny)
        thisfTerm = Vector{ITensor}(undef, Ny)
        for row in 1:Ny
            Ap = dag(copy(A[row, col]))'
            Ap = setprime(Ap, 0, next_col_inds[row]')
            Qp = dag(copy(Q[row]))'
            Qp = setprime(Qp, 0, QR_inds[row]')
            thisTerm[row] = A[row, col] * Ap
            thisTerm[row] = Qp*thisTerm[row]
            thisfTerm[row] = thisTerm[row] * Q[row]
        end
        fF = cumprod(thisfTerm)
        rF = reverse(cumprod(reverse(thisfTerm)))
        Envs = is_gpu ? [cuITensor(1.0) for row in 1:Ny] : [ITensor(1.0) for row in 1:Ny]
        for row in 1:Ny
            if row > 1
                Envs[row] *= fF[row - 1]
            end
            Envs[row] *= thisTerm[row]
            if row < Ny
                Envs[row] *= rF[row + 1]
            end
        end
        for row in 1:Ny
            #=println()
            @show row
            @show Envs[row]
            println()=#
            if row < Ny
                Q_, P_ = polar(Envs[row], QR_inds[row], commonindex(Q[row], Q[row+1]), findindex(Envs[row], "Site"))
                Q[row] = noprime(Q_) 
            else
                Q_, P_ = polar(Envs[row], QR_inds[row], findindex(Envs[row], "Site"))
                Q[row] = noprime(Q_)
            end
            AQinds = IndexSet(findindex(A[row, col], "Site")) 
            if (side == :left && !right_edge) || (side == :right && !left_edge)
                push!(AQinds, commonindex(findinds(A[row, col], "Link"), findinds(Q[row], "Link"))) # prev_col_ind
            end
            cmb_l[row] = combiner(AQinds, tags="Site,AQ,r$row")
            Ampo[row]  = A[row, col] * cmb_l[row]
            Ampo[row]  = replaceindex!(Ampo[row], next_col_inds[row], dummy_nexts[row])
            Q[row]    *= cmb_l[row]
        end
        R           = nmultMPO(dag(Q), Ampo; kwargs...)
        aqr_overlap = is_gpu ? cuITensor(1.0) : ITensor(1.0)
        a_norm      = is_gpu ? cuITensor(1.0) : ITensor(1.0)
        for row in 1:Ny
            aqr_overlap *= Ampo[row] * Q[row] * R[row]
            Q[row]      *= cmb_l[row]
            a_norm      *= A[row, col] * A[row, col]
        end
        ratio = abs(scalar(aqr_overlap))/abs(scalar(a_norm))
        push!(ratio_history, ratio)
        if ratio > best_overlap || iter == 1
            best_Q = deepcopy(Q)
            best_R = deepcopy(R)
            best_overlap = ratio
        end
        ratio > overlap_cutoff && break
        iter += 1
        #=if (iter > 10 && ratio < 0.5) || (iter > 20 && mod(iter, 20) == 0)
            for row in 1:Ny
                salt = randomITensor(inds(Q[row]))
                salt = is_gpu ? cuITensor(salt) : salt 
                salt /= 10.0 * norm(salt)
                Q[row] += salt
                Q[row] /= √norm(Q[row])
            end
        end=#
    end
    #@show best_overlap
    return best_Q, best_R, next_col_inds, QR_inds, dummy_nexts
end

function gaugeColumn( A::PEPS, col::Int, side::Symbol; kwargs...)
    Ny, Nx = size(A)

    prev_col_inds = Vector{Index}(undef, Ny)
    next_col_inds = Vector{Index}(undef, Ny)

    next_col   = side == :left ? col - 1 : col + 1
    prev_col   = side == :left ? col + 1 : col - 1
    left_edge  = col == 1
    right_edge = col == Nx
    is_gpu = !(data(store(A[1,1])) isa Array)
    println()
    a_norm        = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    na_norm       = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    for row in reverse(1:Ny)
        a_norm *= dag(A[row, col]) * A[row, col]
        na_norm *= dag(A[row, next_col]) * A[row, next_col]
    end
    
    #@show scalar(a_norm)
    #@show scalar(na_norm)

    Q, R, next_col_inds, QR_inds, dummy_next_inds = gaugeQR(A, col, side; kwargs...)
    cmb_r = Vector{ITensor}(undef, Ny)
    cmb_u = Vector{ITensor}(undef, Ny - 1)
    if (side == :left && col > 1 ) || (side == :right && col < Nx)
        next_col_As = MPO(Ny, A[:, next_col], 0, Ny+1)
        nn_col = side == :left ? next_col - 1 : next_col + 1
        cmb_inds = [IndexSet(findindex(A[row, next_col], "Site")) for row in 1:Ny]
        for row in 1:Ny
            0 < nn_col < Nx + 1 && push!(cmb_inds[row], commonindex(A[row, next_col], A[row, nn_col]))
            cmb_r[row] = combiner(cmb_inds[row], tags="Site,CMB,c$col,r$row")
            next_col_As[row] *= cmb_r[row]
            if (side == :left && !left_edge) || (side == :right && !right_edge)
                next_col_As[row] = replaceindex!(next_col_As[row], next_col_inds[row], dummy_next_inds[row])
            end
        end
    end
    maxdim::Int = get(kwargs, :maxdim, 1)
    result = nmultMPO(R, next_col_As; maxdim=maxdim)
    r_norm       = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    for row in reverse(1:Ny)
        r_norm     *= dag(result[row]) * result[row]
    end
    #=for row in 1:Ny
        println()
        @show row
        @show A[row, col]
        @show Q[row]
        @show R[row]
        @show result[row]
        println()
    end=#
    #orthogonalize!(result, 1; kwargs...)
    #result[1] /= √scalar(r_norm)
    true_QR_inds = [Index(dim(QR_inds[row]), "Link,r,r$row" * (side == :left ? ",c$(col-1)" : ",c$col")) for row in 1:Ny]

    A[:, col] = tensors(Q)
    cUs = [commonindex(A[row, col], A[row+1, col]) for row in 1:Ny-1]
    true_U_inds = [Index(dim(cUs[row]), "Link,u,r$row,c$col") for row in 1:Ny-1]
    A[:, col] = [replaceindex!(A[row, col], QR_inds[row], true_QR_inds[row]) for row in 1:Ny]
    A[:, col] = vcat([replaceindex!(A[row, col], cUs[row], true_U_inds[row]) for row in 1:Ny-1], A[Ny, col])
    A[:, col] = vcat(A[1, col], [replaceindex!(A[row, col], cUs[row-1], true_U_inds[row-1]) for row in 2:Ny])

    A[:, next_col] = tensors(result)
    A[:, next_col] = [replaceindex!(A[row, next_col], QR_inds[row], true_QR_inds[row]) for row in 1:Ny]
    cUs = [commonindex(A[row, next_col], A[row+1, next_col]) for row in 1:Ny-1]
    true_nU_inds = [Index(dim(cUs[row]), "Link,u,r$row,c" * string(next_col)) for row in 1:Ny-1]
    A[:, next_col] = vcat([replaceindex!(A[row, next_col], cUs[row], true_nU_inds[row]) for row in 1:Ny-1], A[Ny, next_col])
    A[:, next_col] = vcat(A[1, next_col], [replaceindex!(A[row, next_col], cUs[row-1], true_nU_inds[row-1]) for row in 2:Ny])
    A[:, next_col] = [A[row, next_col] * cmb_r[row] for row in 1:Ny]
    a_norm        = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    na_norm       = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    for row in reverse(1:Ny)
        a_norm *= dag(A[row, col]) * A[row, col]
        na_norm *= dag(A[row, next_col]) * A[row, next_col]
    end
    
    @show scalar(a_norm)
    @show scalar(na_norm)
    println()
    return A
end