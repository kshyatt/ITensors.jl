function cuMPO(O::MPO)
    P = copy(O)
    for site in 1:length(O)
        P.A_[site] = cuITensor(O.A_[site])
    end
    return P 
end
  
cuMPO(N::Int, A::Vector{ITensor}) = cuMPO(MPO(N, A))
cuMPO(sites::SiteSet) = cuMPO(MPO(sites))

function cuMPO(::Type{T}, 
               sites::SiteSet, 
               ops::Vector{String}) where {T}
    return cuMPO(MPO(T, sites, ops))
end

function cuMPO(::Type{T}, 
               sites::SiteSet, 
               ops::String;
               store_type::DataType = Float64) where {T}
    return cuMPO(MPO(T, sites, fill(ops, length(sites)), store_type=store_type))
end

function plussers(::Type{T}, left_ind::Index, right_ind::Index, sum_ind::Index) where T <: CuArray
    #if dir(left_ind) == dir(right_ind) == Neither
        total_dim    = dim(left_ind) + dim(right_ind)
        total_dim    = max(total_dim, 1)
        left_data   = CuArrays.zeros(Float64, dim(left_ind), dim(sum_ind))
        ldi = diagind(left_data, 0)
        left_data[ldi] = 1.0
        left_tensor = cuITensor(vec(left_data), left_ind, sum_ind)
        right_data   = CuArrays.zeros(Float64, dim(right_ind), dim(sum_ind))
        rdi = diagind(right_data, dim(left_ind))
        right_data[rdi] = 1.0
        right_tensor = cuITensor(vec(right_data), right_ind, sum_ind)
        return left_tensor, right_tensor
    #else # tensors have QNs
    #    throw(ArgumentError("support for adding MPOs with defined quantum numbers not implemented yet."))
    #end
end
