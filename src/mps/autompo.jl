#
# Optimizations:
#  - replace leftmap, rightmap with sorted vectors
# 

export SiteOp,
       MPOTerm,
       AutoMPO,
       terms,
       add!,
       toMPO,
       MPOTerm,
       MatElem,
       SiteOp

import LinearAlgebra.svd

###########################
# SiteOp                  # 
###########################

struct SiteOp
  name::String
  site::Int
end
name(s::SiteOp) = s.name
site(s::SiteOp) = s.site
Base.show(io::IO,s::SiteOp) = print(io,"\"$(name(s))\"($(site(s)))")

#function Base.isless(s1::SiteOp,s2::SiteOp)::Bool
#  if site(s1) < site(s2)
#    return true
#  end
#  return name(s1) < name(s2)
#end

###########################
# OpTerm                  # 
###########################

const OpTerm = Vector{SiteOp}
mult(t1::OpTerm,t2::OpTerm) = isempty(t2) ? t1 : vcat(t1,t2)

###########################
# MPOTerm                 # 
###########################

struct MPOTerm
  coef::ComplexF64
  ops::OpTerm
end
coef(op::MPOTerm) = op.coef
ops(op::MPOTerm) = op.ops

function ==(t1::MPOTerm,t2::MPOTerm)
  return (t1.ops==t2.ops && isapprox(t1.coef,t2.coef))
end

function Base.isless(t1::MPOTerm,t2::MPOTerm)::Bool
  if !isapprox(coef(t1),coef(t2))
    return coef(t1) < coef(t2)
  end
  return ops(t1) < ops(t2)
end

function MPOTerm(c::Number,
                 op1::String,
                 i1::Int) 
  return MPOTerm(convert(ComplexF64,c),[SiteOp(op1,i1)])
end

function MPOTerm(c::Number,
                 op1::String,i1::Int,
                 op2::String,i2::Int)
  return MPOTerm(convert(ComplexF64,c),[SiteOp(op1,i1),SiteOp(op2,i2)])
end

function MPOTerm(c::Number,
                 op1::String,i1::Int,
                 op2::String,i2::Int,
                 ops...)
  vop = OpTerm(undef,2+div(length(ops),2))
  vop[1] = SiteOp(op1,i1)
  vop[2] = SiteOp(op2,i2)
  for n=1:2:length(ops)
    vop[2+n] = SiteOp(ops[n],ops[n+1])
  end
  return MPOTerm(convert(ComplexF64,c),vop)
end

#function MPOTerm(c::Number,
#                 ops::OpTerm)
#  return MPOTerm(convert(ComplexF64,c),ops)
#end

function Base.show(io::IO,
              op::MPOTerm) 
  c = coef(op)
  if c != 1.0+0.0im
    if imag(c) == 0.0
      print(io,"$(real(c)) ")
    elseif real(c) == 0.0
      print(io,"$(imag(c))im ")
    else
      print(io,"($c) ")
    end
  end
  for o in ops(op)
    print(io,"\"$(name(o))\"($(site(o)))")
  end
end

############################
## AutoMPO                 #
############################

struct AutoMPO
  sites::SiteSet
  terms::Vector{MPOTerm}
  AutoMPO(s::SiteSet) = new(s,Vector{MPOTerm}())
end
sites(ampo::AutoMPO) = ampo.sites
terms(ampo::AutoMPO) = ampo.terms

function add!(ampo::AutoMPO,
              op::String, i::Int)
  push!(terms(ampo),MPOTerm(1.0,op,i))
end

function add!(ampo::AutoMPO,
              coef::Number,
              op::String, i::Int)
  push!(terms(ampo),MPOTerm(coef,op,i))
end

function add!(ampo::AutoMPO,
              op1::String, i1::Int,
              op2::String, i2::Int)
  push!(terms(ampo),MPOTerm(1.0,op1,i1,op2,i2))
end

function add!(ampo::AutoMPO,
              coef::Number,
              op1::String, i1::Int,
              op2::String, i2::Int)
  push!(terms(ampo),MPOTerm(coef,op1,i1,op2,i2))
end

function add!(ampo::AutoMPO,
              op1::String, i1::Int,
              op2::String, i2::Int,
              ops...)
  push!(terms(ampo),MPOTerm(1.0,op1,i1,op2,i2,ops...))
end

function add!(ampo::AutoMPO,
              coef::Number,
              op1::String, i1::Int,
              op2::String, i2::Int,
              ops...)
  push!(terms(ampo),MPOTerm(coef,op1,i1,op2,i2,ops...))
end


function show(io::IO,
              ampo::AutoMPO) 
  println(io,"AutoMPO:")
  for term in terms(ampo)
    println(io,"  $term")
  end
end

##################################
# MatElem (simple sparse matrix) #
##################################

struct MatElem{T}
  row::Int
  col::Int
  val::T
end

#function Base.show(io::IO,m::MatElem)
#  print(io,"($(m.row),$(m.col),$(m.val))")
#end

function toMatrix(els::Vector{MatElem{T}})::Matrix{T} where {T}
  nr = 0
  nc = 0
  for el in els
    nr = max(nr,el.row)
    nc = max(nr,el.col)
  end
  M = zeros(T,nr,nc)
  for el in els
    M[el.row,el.col] = el.val
  end
  return M
end

function ==(m1::MatElem{T},m2::MatElem{T})::Bool where {T}
  return (m1.row==m2.row && m1.col==m2.col && m1.val==m2.val)
end

function Base.isless(m1::MatElem{T},m2::MatElem{T})::Bool where {T}
  if m1.row != m2.row
    return m1.row < m2.row
  elseif m1.col != m2.col
    return m1.col < m2.col
  end
  return m1.val < m2.val
end

function posInLink!(linkmap::Vector{OpTerm},
                    op::OpTerm)::Int
  isempty(op) && return -1
  for n=1:length(linkmap)
    (linkmap[n]==op) && return n
  end
  push!(linkmap,op)
  return length(linkmap)
end

function determineValType(terms::Vector{MPOTerm})
  for t in terms
    (!isreal(coef(t))) && return ComplexF64
  end
  return Float64
end

function computeSiteProd(sites::SiteSet,
                         ops::OpTerm)::ITensor
  i = ops[1].site
  T = op(sites,ops[1].name,i)
  for j=2:length(ops)
    (ops[j].site != i) && error("Mismatch of site number in computeSiteProd")
    opj = op(sites,ops[j].name,i)
    T = multSiteOps(T,opj)
  end
  return T
end


function remove_dups!(v::Vector{T}) where {T}
  N = length(v)
  (N==0) && return
  sort!(v)
  n = 1
  u = 2
  while u <= N
    while u < N && v[u]==v[n] 
      u += 1
    end
    if v[u] != v[n]
      v[n+1] = v[u]
      n += 1
    end
    u += 1
  end
  resize!(v,n)
  return
end

function svdMPO(ampo::AutoMPO; 
                kwargs...)::MPO

  mindim::Int = get(kwargs,:mindim,1)
  maxdim::Int = get(kwargs,:maxdim,10000)
  cutoff::Float64 = get(kwargs,:cutoff,1E-13)

  N = length(sites(ampo))

  ValType = determineValType(terms(ampo))

  Vs = [Matrix{ValType}(undef,1,1) for n=1:N]
  tempMPO = [MatElem{MPOTerm}[] for n=1:N]

  crosses_bond(t::MPOTerm,n::Int) = (ops(t)[1].site <= n <= ops(t)[end].site)

  rightmap = OpTerm[]
  next_rightmap = OpTerm[]
  
  for n=1:N

    leftbond_coefs = MatElem{ValType}[]

    leftmap = OpTerm[]
    for term in terms(ampo)
      crosses_bond(term,n) || continue

      left::OpTerm   = filter(t->(t.site < n),ops(term))
      onsite::OpTerm = filter(t->(t.site == n),ops(term))
      right::OpTerm  = filter(t->(t.site > n),ops(term))

      bond_row = -1
      bond_col = -1
      if !isempty(left)
        bond_row = posInLink!(leftmap,left)
        bond_col = posInLink!(rightmap,mult(onsite,right))
        bond_coef = convert(ValType,coef(term))
        push!(leftbond_coefs,MatElem(bond_row,bond_col,bond_coef))
      end

      A_row = bond_col
      A_col = posInLink!(next_rightmap,right)
      site_coef = 1.0+0.0im
      if A_row == -1
        site_coef = coef(term)
      end
      isempty(onsite) && push!(onsite,SiteOp("Id",n))
      el = MatElem(A_row,A_col,MPOTerm(site_coef,onsite))
      push!(tempMPO[n],el)
    end
    rightmap = next_rightmap
    next_rightmap = OpTerm[]

    remove_dups!(tempMPO[n])

    if n > 1 && !isempty(leftbond_coefs)
      M = toMatrix(leftbond_coefs)
      U,S,V = svd(M)
      P = S.^2
      truncate!(P;maxdim=maxdim,cutoff=cutoff,mindim=mindim)
      tdim = length(P)
      nc = size(M,2)
      Vs[n-1] = Matrix{ValType}(V[1:nc,1:tdim])
    end

  end

  llinks = [Index() for n=1:N+1]
  llinks[1] = Index(2,"Link,n=0")

  H = MPO(sites(ampo))

  # Constants which define MPO start/end scheme
  rowShift = 2
  colShift = 2
  startState = 2
  endState = 1

  for n=1:N
    VL = Matrix{ValType}(undef,1,1)
    if n > 1
      VL = Vs[n-1]
    end
    VR = Vs[n]
    tdim = size(VR,2)

    llinks[n+1] = Index(2+tdim,"Link,n=$n")

    finalMPO = Dict{OpTerm,Matrix{ValType}}()

    ll = llinks[n]
    rl = llinks[n+1]

    idTerm = [SiteOp("Id",n)]
    finalMPO[idTerm] = zeros(ValType,dim(ll),dim(rl))
    idM = finalMPO[idTerm]
    idM[1,1] = 1.0
    idM[2,2] = 1.0

    defaultMat() = zeros(ValType,dim(ll),dim(rl))

    for el in tempMPO[n]
      A_row = el.row
      A_col = el.col
      t = el.val
      (abs(coef(t)) > eps()) || continue

      M = get!(finalMPO,ops(t),defaultMat())

      ct = convert(ValType,coef(t))
      if A_row==-1 && A_col==-1 #onsite term
        M[startState,endState] += ct
      elseif A_row==-1 #term starting on site n
        for c=1:size(VR,2)
          z = ct*VR[A_col,c]
          M[startState,colShift+c] += z
        end
      elseif A_col==-1 #term ending on site n
        for r=1:size(VL,2)
          z = ct*conj(VL[A_row,r])
          M[rowShift+r,endState] += z
        end
      else
        for r=1:size(VL,2),c=1:size(VR,2)
          z = ct*conj(VL[A_row,r])*VR[A_col,c]
          M[rowShift+r,colShift+c] += z
        end
      end
    end

    s = sites(ampo)[n]
    H[n] = ITensor(dag(s),s',ll,rl)
    for (op,M) in finalMPO
      T = ITensor(M,ll,rl)
      csp = computeSiteProd(sites(ampo),op)
      H[n] += T*csp
    end

  end

  L = ITensor(llinks[1])
  L[startState] = 1.0

  R = ITensor(llinks[N+1])
  R[endState] = 1.0

  H[1] *= L
  H[N] *= R

  return H
end

function sortEachTerm!(ampo::AutoMPO)
  isless_site(o1::SiteOp,o2::SiteOp) = (site(o1)<site(o2))
  for t in terms(ampo)
    sort!(ops(t),alg=InsertionSort,lt=isless_site)
  end
end

function toMPO(ampo::AutoMPO; 
               kwargs...)::MPO
  sortEachTerm!(ampo)
  return svdMPO(ampo;kwargs...)
end
