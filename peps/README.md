## Getting CuITensors and PEPS up and running

What you'll need:
  - Julia 1.x -- I use 1.4-dev but anything 1.1 or after should work
  - CUDA 10.1
  - CUTENSOR v0.2.2 -- `libcutensor.so` needs to be on your `LD_LIBRARY_PATH` so that `CuArrays.jl` will be able to find it.
  - A copy of this repo that is available to Julia. `ITensors.jl` is presently *not* registered in the main Julia package registry. The easiest way to make Julia aware of it is to do a `git` clone of the `ksh/gpu` branch of `https://github.com/kshyatt/ITensors.jl`, and then execute the following in `julia`:
    ```
    julia> cd("YOUR_PATH_HERE/ITensors.jl")

    julia> ]

    pkg> activate .
    ...

    ITensors> 
    ```
    For a bit more explanation of what's going on here, check out the [Julia Pkg docs](https://docs.julialang.org/en/v1/stdlib/Pkg/).
  - You might also need to do a checkout of the `master` branches of some dependencies:
    ```
    julia> cd("YOUR_PATH_HERE/ITensors.jl")

    julia> ]

    pkg> activate .
    ...

    ITensors> dev CuArrays GPUArrays CUDAnative CUDAapi

    ITensors> build
    ```

To check if this has all worked, you can run one of the profiling scripts in `prof/` using something like `julia-1.4 prof_run.jl`.

Scripts included:
- `prof/`: Really basic time profiling of algorithms that exist in "basic" `ITensors.jl`: 1-D and 2-D DMRG 
- `peps/`: Implementation of the PEPS algorithm in [this article](https://arxiv.org/abs/1908.08833), along with profiling and timing info.

Probably the most interesting file in `peps/` is `full_peps_run.jl`. This runs identical (or, extremely similar) simulations on the CPU (using BLAS) and GPU (using CUTENSOR) and does time profiling of them.
Because of the way Julia compilation works, only the "main" simulation is timed (this is the vast majority of the walltime) -- for more information on this, see [here](https://docs.julialang.org/en/v1/manual/profile/).
There are two command line arguments you need to provide: the system size `L`, which creates an `L x L` lattice, and `chi`, which controls the internal matrix size. `chi` has the name "bond dimension" in tensor network
algorithms, but essentially it sets the size of the `chi x chi x chi x chi x chi` tensors that make up the PEPS. Right now the C++ code handles `chi` in the range `[3 .. 7]`. You might see OOM issues if you try to run
a big lattice with big `chi` on the GPU, even one with a lot of memory -- `CuArrays.jl` and the Julia GC are working "as intended" but maybe not as we might like, see [here](https://github.com/JuliaGPU/CuArrays.jl/issues/323) for more.

An example of running this file:

`julia-1.4 full_peps_run.jl 4 3`  

Here, `L=4` and `chi=3`. If you find this annoying it's easy to hardcode these variables in the script itself, or load them from JSON files.

Right now, the code is single GPU and single node only.
I've checked that the CPU run (the first one in the file) involves no GPU activity (besides phoning it up to say "hi, I'm here!") using `nvprof`. There is a bit of transfer from the host to device
in the GPU code, but last time I profiled the code with `nvprof` and Julia's inbuilt profiler it was not contributing significantly to runtime -- if you see something different, let me know, as I'm
pretty sure I know where most of it's coming from and it could be fixed.

The code outputs to `STDOUT` the energy at each sweep and, at the end of run, the total walltime needed to do ten sweeps. We found in the Julia and C++ simulations that 10 sweeps was realistic to get the
energy to converge, and for larger systems with larger `chi` you will be waiting a *long* time for the CPU to finish (it's a big reason we're so excited about the GPU results!).

All the simulations I've conducted were on the rusty cluster at the Simons Foundation. The nodes I ran the code on have 32GB V100 GPUs with 36 core Skylake CPUs (I can get more detailed information if it would be helpful).
