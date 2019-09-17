Dense{T, SA}(x::Dense{T, SB}) where {T<:Number, SA<:CuArray, SB<:Array} = Dense{T, S}(CuArray(x))
Dense{T, SA}(x::Dense{T, SB}) where {T<:Number, SA<:Array, SB<:CuArray} = Dense{T, S}(collect(x.data))
Base.collect(x::Dense{T, S}) where {T<:Number, S<:CuArray} = Dense{T, Vector{T}}(collect(x.data))

*(D::Dense{T, AT},x::S) where {T,AT<:CuArray,S<:Number} = Dense{promote_type(T,S), CuVector{promote_type(T,S)}}(x .* data(D))

function truncate!(P::CuVector{Float64};
                   kwargs...)::Tuple{Float64,Float64,CuVector{Float64}}
  maxdim::Int = min(get(kwargs,:maxdim,length(P)), length(P))
  mindim::Int = min(get(kwargs,:mindim,1), maxdim)
  cutoff::Float64 = get(kwargs,:cutoff,0.0)
  absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
  doRelCutoff::Bool = get(kwargs,:doRelCutoff,true)
  origm = length(P)
  docut = 0.0
  maxP  = maximum(P)
  if maxP == 0.0
    P = CuArrays.zeros(Float64, 1)
    return 0.,0.,P
  end
  if origm==1
    docut = maxP/2
    return 0., docut, P[1:1]
  end

  @timeit "setup rP" begin
      #Zero out any negative weight
      #neg_z_f = (!signbit(x) ? x : 0.0)
      rP = map(x -> !signbit(x) ? x : 0.0, P)
      #rP = reverse(P)
      n = origm
      truncerr = 0.0
      if n >= maxdim
          truncerr = sum(rP[1:n-maxdim])
          n = maxdim
      end
  end
  @timeit "handle cutoff" begin
      if absoluteCutoff
        #Test if individual prob. weights fall below cutoff
        #rather than using *sum* of discarded weights
        err_rP = inv.(rP .+ truncerr .- cutoff*scale)
        cut_ind = CuArrays.CUBLAS.iamax(err_rP) - 1
        n = min(maxdim, length(P) - cut_ind)
        n = max(n, mindim)
        truncerr += sum(rP[1:cut_ind])
      else
        scale = 1.0
        @timeit "find scale" begin 
            if doRelCutoff
              scale = sum(P)
              scale = scale > 0.0 ? scale : 1.0
            end
        end

        #Continue truncating until *sum* of discarded probability 
        #weight reaches cutoff reached (or m==mindim)
        sub_arr = rP .+ truncerr .- cutoff*scale
        err_rP  = sub_arr ./ abs.(sub_arr)
        flags   = reinterpret(Float64, (signbit.(err_rP) .<< 1 .& 2) .<< 61)
        cut_ind = CuArrays.CUBLAS.iamax(err_rP .* flags) - 1
        truncerr += sum(rP[1:cut_ind])
        n = min(maxdim, length(P) - cut_ind)
        n = max(n, mindim)
        if scale==0.0
          truncerr = 0.0
        else
          truncerr /= scale
        end
      end
  end
  if n < 1
    n = 1
  end
  if n < origm
    hP = collect(P)
    docut = (hP[n]+hP[n+1])/2
    if abs(hP[n]-hP[n+1]) < 1E-3*hP[n]
      docut += 1E-3*hP[n]
    end
  end
  @timeit "setup return" begin
      rinds = Iterators.reverse(1:n)
      rrP = P[rinds]
  end
  return truncerr,docut,rrP
end

function storage_scalar(D::Dense{AT, A}) where {AT, A<:CuArray}
    length(D)==1 && return collect(D)[1]
  throw(ErrorException("Cannot convert Dense -> Number for length of data greater than 1"))
end

function storage_contract(Astore::Dense{T, SA},
                          Ais::IndexSet,
                          Bstore::Dense{T, SB},
                          Bis::IndexSet) where {T, SA<:Array, SB<:CuArray}
    cAstore = Dense{T, SB}(CuArray(data(Astore)))
    return storage_contract(cAstore, Ais, Bstore, Bis)
end

function storage_contract(Astore::Dense{T, SA},
                          Ais::IndexSet,
                          Bstore::Dense{T, SB},
                          Bis::IndexSet) where {T, SA<:CuArray, SB<:Array}
    cBstore = Dense{T, SA}(CuArray(data(Bstore)))
    return storage_contract(Astore, Ais, cBstore, Bis)
end

function storage_contract(Astore::Dense{T, S},
                          Ais::IndexSet,
                          Bstore::Dense{T, S},
                          Bis::IndexSet) where {T, S<:CuArray}
  
  if length(Ais)==0
    Cis = Bis
    Cs = similar(data(Bstore))
    Cstore = Dense{T, S}(mul!(Cs, data(Bstore), data(Astore)))
  elseif length(Bis)==0
    Cis = Ais
    Cs = similar(data(Astore))
    Cstore = Dense{T, S}(mul!(Cs, data(Astore), data(Bstore)))
  else
    #TODO: check for special case when Ais and Bis are disjoint sets
    #I think we should do this analysis outside of storage_contract, at the ITensor level
    #(since it is universal for any storage type and just analyzes in indices)
    (Alabels,Blabels) = compute_contraction_labels(Ais,Bis)
    if is_outer(Alabels,Blabels)
      Cis = IndexSet(Ais,Bis)
      Cstore = outer(Astore,Bstore)
    else
      (Cis,Clabels) = contract_inds(Ais,Alabels,Bis,Blabels)
      Cstore = contract(Cis,Clabels,Astore,Ais,Alabels,Bstore,Bis,Blabels)
    end
  end
  return (Cis,Cstore)
end

function storage_svd(Astore::Dense{T, S},
                     Lis::IndexSet,
                     Ris::IndexSet;
                     kwargs...
                    ) where {T, S<:CuArray}
  maxdim::Int = get(kwargs,:maxdim,min(dim(Lis),dim(Ris)))
  mindim::Int = get(kwargs,:mindim,1)
  cutoff::Float64 = get(kwargs,:cutoff,0.0)
  absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
  doRelCutoff::Bool = get(kwargs,:doRelCutoff,true)
  utags::String     = get(kwargs,:utags,"Link,u")
  vtags::String     = get(kwargs,:vtags,"Link,v")
  rsd = reshape(data(Astore),dim(Lis),dim(Ris))
  @timeit "cusolver" begin
      MU,MS,MV = CUSOLVER.svd(rsd)
  end
  sqr(x) = x^2
  P = sqr.(MS)
  @timeit "truncate!" begin
      err, cut, P = truncate!(P; mindim=mindim, maxdim=maxdim, cutoff=cutoff, absoluteCutoff=absoluteCutoff, doRelCutoff=doRelCutoff)
  end
  dS = length(P)
  if dS < length(MS)
    MU = MU[:,1:dS]
    MS = MS[1:dS]
    MV = MV[:,1:dS]
  end

  u = Index(dS,utags)
  v = settags(u,vtags)
  Uis,Ustore = IndexSet(Lis...,u),Dense{T, CuVector{T}}(vec(MU))
  #TODO: make a diag storage
  Sdata      = CuArrays.zeros(T, dS, dS)
  dsi        = diagind(Sdata, 0)
  Sdata[dsi] = MS
  Sis,Sstore = IndexSet(u,v),Dense{T, CuVector{T}}(vec(Sdata))
  Vis,Vstore = IndexSet(Ris...,v),Dense{T, CuVector{T}}(CuVector{T}(vec(MV)))

  return (Uis,Ustore,Sis,Sstore,Vis,Vstore)
end

function storage_eigen(Astore::Dense{S, T}, Lis::IndexSet,Ris::IndexSet;kwargs...) where {T<:CuArray,S<:Number}
  maxdim::Int          = get(kwargs,:maxdim,min(dim(Lis),dim(Ris)))
  mindim::Int          = get(kwargs,:mindim,1)
  cutoff::Float64      = get(kwargs,:cutoff,0.0)
  absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
  doRelCutoff::Bool    = get(kwargs,:doRelCutoff,true)
  tags::TagSet         = get(kwargs,:lefttags,"Link,u")
  lefttags::TagSet     = get(kwargs,:lefttags,tags)
  righttags::TagSet    = get(kwargs,:righttags,prime(lefttags))
  
  local d_W, d_V
  dim_left  = dim(Lis)
  dim_right = dim(Ris)
  d_A       = reshape(data(Astore),dim_left,dim_right)
  if S <: Complex
    d_W, d_V = CUSOLVER.heevd!('V', 'U', d_A)
  else
    d_W, d_V = CUSOLVER.syevd!('V', 'U', d_A)
  end
  #TODO: include truncation parameters as keyword arguments
  #dW = reverse(d_W)
  err, cut, dW = truncate!(d_W; maxdim=maxdim, mindim=mindim,
                           cutoff=cutoff,
                           absoluteCutoff=absoluteCutoff,
                           doRelCutoff=doRelCutoff)
  dD = length(dW)
  u  = Index(dD,tags)
  v  = prime(u)
  dV = reverse(d_V, dims=2)
  if dD < size(dV,2)
    dV = CuMatrix(dV[:,1:dD])
  end
  Uis,Ustore = IndexSet(Lis...,u),Dense{S, T}(vec(dV))
  #TODO: make a diag storage
  Ddata = CuArrays.zeros(S, dD, dD)
  ddi = diagind(Ddata, 0)
  Ddata[ddi] = dW
  Dis,Dstore = IndexSet(u,v),Dense{S, T}(vec(Ddata))
  return (Uis,Ustore,Dis,Dstore)
end

function storage_qr(Astore::Dense{S, T},Lis::IndexSet,Ris::IndexSet; kwargs...) where {T<:CuArray, S<:Number}
  tags::TagSet = get(kwargs,:tags,"Link,u")
  dim_left = dim(Lis)
  dim_right = dim(Ris)
  dQR = qr!(reshape(data(Astore),dim_left,dim_right))
  MQ = dQR.Q
  MP = dQR.R
  dim_middle = min(dim_left,dim_right)
  u = Index(dim_middle,tags)
  #Must call Matrix() on MQ since the QR decomposition outputs a sparse
  #form of the decomposition
  Qis,Qstore = IndexSet(Lis...,u),Dense{S, T}(vec(CuArray(MQ)))
  Pis,Pstore = IndexSet(u,Ris...),Dense{S, T}(vec(CuArray(MP)))
  return (Qis,Qstore,Pis,Pstore)
end

function storage_polar(Astore::Dense{S, T},Lis::IndexSet,Ris::IndexSet) where {T<:CuArray, S<:Number}
  dim_left   = dim(Lis)
  dim_right  = dim(Ris)
  MQ,MP      = polar(reshape(data(Astore),dim_left,dim_right))
  dim_middle = min(dim_left,dim_right)
  Uis        = prime(Ris)
  Qis,Qstore = IndexSet(Lis...,Uis...),Dense{S, T}(vec(MQ))
  Pis,Pstore = IndexSet(Uis...,Ris...),Dense{S, T}(vec(MP))
  return (Qis,Qstore,Pis,Pstore)
end

function outer(D1::Dense{T, S}, D2::Dense{T, S}) where {T, S<:CuArray}
    D1_dat = reshape(data(D1), length(data(D1)))
    D2_dat = transpose(reshape(data(D2), length(data(D2))))
    D3_dat = CuMatrix{T}(undef, length(D1_dat), length(D2_dat))
    mul!(D3_dat, D1_dat, D2_dat)
    return Dense{T, S}(reshape(D3_dat, length(D3_dat)))
end

function storage_add!(Bstore::Dense{SB, T},Bis::IndexSet,Astore::Dense{SA, T},Ais::IndexSet, x::Number=1.) where {T<:CuArray, SA<:Number, SB<:Number}
  Adata = x*reshape(data(Astore), dims(Ais))
  p = ITensors.calculate_permutation(Bis,Ais)
  #permAdata = permutedims(reshape(Adata,dims(Ais)),p)
  Bdata = reshape(data(Bstore), dims(Bis))
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(Bis)
      push!(ind_dict, i)
  end
  id_op = CuArrays.CUTENSOR.CUTENSOR_OP_IDENTITY
  ctainds = zeros(Int, length(Ais))
  ctbinds = zeros(Int, length(Bis))
  for (ii, ia) in enumerate(Ais)
      ctainds[ii] = findfirst(x->x==ia, ind_dict)
  end
  for (ii, ib) in enumerate(Bis)
      ctbinds[ii] = findfirst(x->x==ib, ind_dict)
  end
  ctcinds = copy(ctbinds)
  C = CuArrays.zeros(SB, size(Bdata))
  CuArrays.CUTENSOR.elementwiseBinary!(one(SA), Adata, Vector{Char}(ctainds), id_op, one(SB), Bdata, Vector{Char}(ctbinds), id_op, C, Vector{Char}(ctcinds), CUTENSOR.CUTENSOR_OP_ADD)
  copyto!(Bstore.data, reshape(C, length(Bdata)))
  return Bstore
end

function storage_permute!(Bstore::Dense{SB, T},Bis::IndexSet,Astore::Dense{SB, T},Ais::IndexSet) where {T<:CuArray, SA<:Number, SB<:Number}
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(Ais)
      push!(ind_dict, i)
  end
  Adata = data(Astore)
  Bdata = data(Bstore)
  reshapeBdata = reshape(Bdata,dims(Bis))
  reshapeAdata = reshape(Adata,dims(Ais))
  ctainds = zeros(Int, length(Ais))
  ctbinds = zeros(Int, length(Bis))
  for (ii, ia) in enumerate(Ais)
      ctainds[ii] = findfirst(x->x==ia, ind_dict)
  end
  for (ii, ib) in enumerate(Bis)
      ctbinds[ii] = findfirst(x->x==ib, ind_dict)
  end
  CuArrays.CUTENSOR.permutation!(one(eltype(Adata)), reshapeAdata, Vector{Char}(ctainds), reshapeBdata, Vector{Char}(ctbinds)) 
  copyto!(Bstore.data, reshape(reshapeBdata, length(Bstore.data)))
end

storage_convert(::Type{CuArray},D::Dense,is::IndexSet) = reshape(data(D),dims(is))

function storage_convert(::Type{CuArray},
                         D::Dense,
                         ois::IndexSet,
                         nis::IndexSet)
  P = calculate_permutation(nis,ois)
  A = reshape(data(D),dims(ois))
  return permutedims(A,P)
end
