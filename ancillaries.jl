function makeAncillaryIs(A::PEPS, L::Environments, R::Environments, col::Int)
    Ny, Nx  = size(A)
    left_As  = [col > 1 ? A[row, col - 1] : ITensor(1) for row in 1:Ny] 
    right_As = [col < Nx ? A[row, col + 1] :ITensor(1) for row in 1:Ny]
    col_site_inds = [findIndex(x, "Site") for x in A[:, col]]
    AAs = [prepareRow(A[row, col], spinI(col_site_inds[row]), left_As[row], right_As[row], L.I[row], R.I[row], col, Nx) for row in 1:Nx]
    return cumprod(reverse(AA)), fill(ITensor(1), Ny)
end

function updateAncillaryIs(A::PEPS, Ibelow::Vector{ITensor}, L::Environments, R::Environments, col::Int, row::Int )
    Ny, Nx  = size(A)
    left_A  = col > 1 ? A[row, col - 1] : ITensor(1) 
    right_A = col < Nx ? A[row, col + 1] :ITensor(1)
    AA      = prepareRow(A[row, col], spinI(col_site_inds[row]), left_As[row], right_As[row], L.I[row], R.I[row], col, Nx)
    AA     *= row > 0 ? Ibelow[row - 1] : ITensor(1)
    push!(Ibelow, AA)
    return Ibelow
end

function makeAncillaryFs(A::PEPS, L::Environments, R::Environments, H, col::Int)
    Ny, Nx   = size(A)
    left_As  = [col > 1 ? A[row, col - 1] : ITensor(1) for row in 1:Ny] 
    right_As = [col < Nx ? A[row, col + 1] :ITensor(1) for row in 1:Ny]
    col_site_inds = [findIndex(x, "Site") for x in A[:, col]]
    Fabove   = fill(Vector{ITensor}(), length(H))
    for opcode in 1:length(H)
        op_row      = H[opcode].sites[1][1]
        ops         = ITensor[spinI(spin_ind) for spin_inds in col_site_inds] 
        ops[op_row] = replaceindex!(copy(H[opcode].ops[1]), H.site_ind, col_site_inds[op_row])
        ancFs = [prepareRow(A[row, col], ops[row], left_As[row], right_As[row], L.I[row], R.I[row], col, Nx) for row in 1:Ny]
        Fabove[opcode] = cumprod(reverse(ancFs)) 
    end
    return Fabove
end

function updateAncillaryFs(A::PEPS, Fbelow::Vector{ITensor}, Ibelow::Vector{ITensor}, L::Environments, R::Environments, H, col::Int, row::Int)
    Ny, Nx   = size(A)
    left_As  = [col > 1 ? A[row, col - 1] : ITensor(1) for row in 1:Ny] 
    right_As = [col < Nx ? A[row, col + 1] :ITensor(1) for row in 1:Ny]
    col_site_inds = [findIndex(x, "Site") for x in A[:, col]]
    for opcode in 1:length(H)
        op_row      = H[opcode].sites[1][1]
        ops         = ITensor[spinI(spin_ind) for spin_inds in col_site_inds] 
        ops[op_row] = op_row == row ? replaceindex!(copy(H[opcode].ops[1]), H.site_ind, col_site_inds[op_row]) : spinI(spin_ind)
        ancF = prepareRow(A[row, col], ops[row], left_As[row], right_As[row], L.I[row], R.I[row], col, Nx)
        push!(Fbelow[opcode], Fbelow[opcode][end]*ancF) 
    end
    return Fbelow
end

function makeAncillaryVs(A::PEPS, L::Environments, R::Environments, H, col::Int)
    Ny, Nx   = size(A)
    left_As  = [col > 1 ? A[row, col - 1] : ITensor(1) for row in 1:Ny] 
    right_As = [col < Nx ? A[row, col + 1] :ITensor(1) for row in 1:Ny]
    col_site_inds = [findIndex(x, "Site") for x in A[:, col]]
    Vabove   = fill(Vector{ITensor}(), length(H))
    for opcode in 1:length(H)
        op_row_a      = H[opcode].sites[1][1]
        op_row_b      = H[opcode].sites[2][1]
        ops           = ITensor[spinI(spin_ind) for spin_inds in col_site_inds] 
        ops[op_row_a] = replaceindex!(copy(H[opcode].ops[1]), H.site_ind, col_site_inds[op_row_a])
        ops[op_row_b] = replaceindex!(copy(H[opcode].ops[2]), H.site_ind, col_site_inds[op_row_b])
        ancVs = [prepareRow(A[row, col], ops[row], left_As[row], right_As[row], L.I[row], R.I[row], col, Nx) for row in 1:Ny]
        Vabove[opcode] = cumprod(reverse(ancFs))[1:op_row_a+1] 
    end
    return Vabove
end

function updateAncillaryVs(A::PEPS, Vbelow::Vector{ITensor}, Ibelow::Vector{ITensor}, L::Environments, R::Environments, H, col::Int, row::Int)
    Ny, Nx   = size(A)
    left_As  = [col > 1 ? A[row, col - 1] : ITensor(1) for row in 1:Ny] 
    right_As = [col < Nx ? A[row, col + 1] :ITensor(1) for row in 1:Ny]
    col_site_inds = [findIndex(x, "Site") for x in A[:, col]]
    for opcode in 1:length(H)
        op_row_a      = H[opcode].sites[1][1]
        op_row_b      = H[opcode].sites[2][1]
        ops           = ITensor[spinI(spin_ind) for spin_inds in col_site_inds] 
        ops[op_row_a] = replaceindex!(copy(H[opcode].ops[1]), H.site_ind, col_site_inds[op_row_a])
        ops[op_row_b] = replaceindex!(copy(H[opcode].ops[2]), H.site_ind, col_site_inds[op_row_b])
        if op_row_b < row
            AA  = prepareRow(A[row, col], ops[row], left_As[row], right_As[row], L.I[row], R.I[row], col, Nx)
            push!(Vbelow[opcode], Vbelow[opcode][end] * AA)
        elseif op_row_b == row
            Ib  = op_row_a > 1 ? Ibelow[op_row_a - 1] : ITensor(1)
            AA  = prepareRow(A[op_row_a, col], ops[op_row_a], left_As[op_row_a], right_As[op_row_a], L.I[op_row_a], R.I[op_row_a], col, Nx)
            AA *= prepareRow(A[op_row_b, col], ops[op_row_b], left_As[op_row_b], right_As[op_row_b], L.I[op_row_b], R.I[op_row_b], col, Nx)
            push!(Vbelow[opcode], AA*Ib)
        end
    end
    return Vbelow
end

function makeAncillarySide(A::PEPS, EnvIP::Environments, EnvIdent::Environments, H, col::Int, side::Symbol)
    Ny, Nx   = size(A)
    col_site_inds = [findIndex(x, "Site") for x in A[:, col]]
    Sabove   = fill(Vector{ITensor}(), length(H))
    next_col = side == :left ? col + 1 : col - 1
    for opcode in 1:length(H)
        op_row        = side == :left ? H[opcode].sites[2][1] : H[opcode].sites[1][1] 
        ops           = ITensor[spinI(spin_ind) for spin_inds in col_site_inds] 
        this_op       = side == :left ? H[opcode].ops[2] : H[opcode].ops[1]
        ops[op_row] = replaceindex!(copy(this_op), H.site_ind, col_site_inds[op_row])
        AAs     = [A[row, col] * ops[row] * prime(dag(A[row, col])) * EnvIP.InProgress[opcode, row] for row in 1:Ny]
        if (col > 1 && side == :right) || (col <  Nx && side == :left)
            cmb, dlt, AAs = [combineAndConnect(AAs[row], A[row, col], A[row, next_col], EnvIDent.I[row]) for row in 1:Ny]
            AAs = [AAs[row] * EnvIdent.I[row] for row in 1:Ny]
        end
        Sabove[opcode] = cumprod(reverse(As))
    end
    return Sabove
end

function updateAncillarySide(A::PEPS, Sbelow::Vector{ITensor}, Ibelow::Vector{ITensor}, EnvIP::Environments, EnvIdent::Environments, H, col::Int, row::Int, side::Symbol)
    Ny, Nx   = size(A)
    col_site_inds = [findIndex(x, "Site") for x in A[:, col]]
    next_col = side == :left ? col + 1 : col - 1
    for opcode in 1:length(H)
        op_row        = side == :left ? H[opcode].sites[2][1] : H[opcode].sites[1][1] 
        ops           = ITensor[spinI(spin_ind) for spin_inds in col_site_inds] 
        this_op       = side == :left ? H[opcode].ops[2] : H[opcode].ops[1]
        ops[op_row]   = replaceindex!(copy(this_op), H.site_ind, col_site_inds[op_row])
        AA            = A[row, col] * ops[row] * prime(dag(A[row, col])) * EnvIP.InProgress[opcode, row]
        next_col = side == :left ? col + 1 : col - 1
        if (col > 1 && side == :right) || (col <  Nx && side == :left)
            cmb, dlt, AA = combineAndConnect(AA, A[row, col], A[row, next_col], EnvIDent.I[row])
            AA = AA * EnvIdent.I[row]
        end
        Sbelow[opcode][row] = (row > 1 ? Sbelow[opcode][row-1] : ITensor(1)) * AA
    end
    return Sabove
end
