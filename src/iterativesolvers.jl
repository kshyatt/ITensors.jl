export davidson

function get_vecs!((phi,q),M,V,AV,ni)
  F = eigen(Hermitian(M))
  lambda = F.values[1]
  u = F.vectors[:,1]
  #phi = u[1]*V[1]
  mul!(phi,u[1],V[1])
  #q = u[1]*AV[1]
  mul!(q,u[1],AV[1])
  for n=2:ni
    #phi += u[n]*V[n]
    add!(phi,u[n],V[n])
    #q += u[n]*AV[n]
    add!(q,u[n],AV[n])
  end
  #q -= lambda*phi
  add!(q,-lambda,phi)
  #Fix sign
  if real(u[1]) < 0.0
    #phi *= -1
    scale!(phi,-1)
    #q *= -1
    scale!(q,-1)
  end
  return lambda, phi
end

function orthogonalize!(q::ITensor,V,ni)
  q0 = copy(q)
  for k=1:ni
    Vq0k = dot(V[k],q0)
    #q += -Vq0k*V[k]
    if( abs(Vq0k) < 1e-12 )
        add!(q,-Vq0k,V[k])
    end
  end
  qnrm = norm(q)
  if qnrm < 1E-10 #orthog failure, try randomizing
    randn!(q)
    qnrm = norm(q)
  end
  scale!(q,1.0/qnrm)
  return q
end

function davidson(A, phi0::ITensor; kwargs...)::Tuple{Float64,ITensor}
  phi         = deepcopy(phi0)
  maxiter     = get(kwargs,:maxiter,2)
  miniter     = get(kwargs,:maxiter,1)
  errgoal     = get(kwargs,:errgoal,1E-14)
  Northo_pass = get(kwargs,:Northo_pass,1)

  approx0     = 1E-12

  nrm = norm(phi)
  if nrm < 1E-18 
    phi = randomITensor(inds(phi))
    nrm = norm(phi)
  end
  scale!(phi, 1.0/nrm)

  maxsize        = size(A)[1]
  actual_maxiter = min(maxiter, maxsize-1)

  #if dim(inds(phi)) != maxsize
  #  error("linear size of A and dimension of phi should match in davidson")
  #end

  V  = ITensor[copy(phi)]
  AV = ITensor[noprime(A*phi)]

  last_lambda = NaN
  lambda      = dot(V[1], AV[1])
  q           = AV[1] - lambda*V[1]
  M           = fill(lambda,(1,1))
  #@show lambda
  for ni=1:actual_maxiter

    qnorm = norm(q)

    errgoal_reached = (qnorm < errgoal && abs(lambda-last_lambda) < errgoal)
    small_qnorm     = (qnorm < max(approx0,errgoal*1E-3))
    converged       = errgoal_reached || small_qnorm

    if (qnorm < 1E-20) || (converged && ni > miniter) #|| (ni >= actual_maxiter)
      #@printf "  done with davidson, ni=%d, qnorm=%.3E\n" ni qnorm
      break
    end

    last_lambda = lambda

    for pass = 1:Northo_pass
      q = orthogonalize!(q, V, ni)
    end

    push!(V,  copy(q))
    push!(AV, noprime(A*q))
    newM = fill(0.0,(ni+1,ni+1))
    newM[1:ni,1:ni] = M
    for k=1:ni+1
      newM[k,ni+1] = dot(V[k],AV[ni+1])
      newM[ni+1,k] = conj(newM[k,ni+1])
    end
    M = newM

    lambda, phi = get_vecs!((phi,q),M,V,AV,ni+1)
  end #for ni=1:actual_maxiter+1
  #nrm = norm(phi)
  #scale!(phi, 1.0/nrm)
  return lambda, phi
end

