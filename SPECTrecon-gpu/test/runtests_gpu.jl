# runtests_gpu.jl

include("../src/SPECTrecon_gpu.jl")
using Main.SPECTrecon_gpu
using Test: @test, @testset, @test_throws, @inferred, detect_ambiguities
using CUDA
CUDA.allowscalar(false)

include("helper_gpu.jl")
include("rotate_gpu.jl")
include("fftconv_gpu.jl")
include("psf-gauss_gpu.jl")
include("project_gpu.jl")

@testset "SPECTrecon_gpu" begin
    @test isempty(detect_ambiguities(SPECTrecon_gpu))
end
