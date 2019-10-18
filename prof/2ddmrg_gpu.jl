using ITensors, ITensors.CuITensors, LinearAlgebra, TimerOutputs

Ny = 6
Nx = 12

N = Nx*Ny

sites = spinHalfSites(N;conserveQNs=false)

lattice = squareLattice(Nx,Ny,yperiodic=false)

ampo = AutoMPO(sites)
for b in lattice
    add!(ampo,0.5,"S+",b.s1,"S-",b.s2)
    add!(ampo,0.5,"S-",b.s1,"S+",b.s2)
    add!(ampo,    "Sz",b.s1,"Sz",b.s2)
end
H = toMPO(ampo)

state = [isodd(n) ? "Up" : "Dn" for n=1:N]
psi0  = productMPS(sites,state)
cH    = cuMPO(H)
cpsi0 = cuMPS(psi0)

sweeps = Sweeps(6)
maxdim!(sweeps,10,100,200,300,400,500)
cutoff!(sweeps,1E-8)
(energy, psi), tcpu, bytes, gctime, memallocs = @timed dmrg(H,psi0,sweeps) 
println("Time to do DMRG on CPU: $tcpu")
(cenergy, cpsi), tgpu, bytes, gctime, memallocs = @timed dmrg(cH,cpsi0,sweeps)
println("Time to do DMRG on GPU: $tgpu")
