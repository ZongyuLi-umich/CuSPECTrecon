"""
    SPECTrecon_gpu
GPU version of system matrix (forward and back-projector) for SPECT image reconstruction.
"""
module SPECTrecon_gpu

    include("helper_gpu.jl")
    include("plan-rotate_gpu.jl")
    include("rotate_gpu.jl")
    include("plan-fft_gpu.jl")
    include("psf-gauss_gpu.jl")
    include("fft_convolve_gpu.jl")
    include("SPECTplan_gpu.jl")
    include("project_gpu.jl")
    include("backproject_gpu.jl")

end # module
