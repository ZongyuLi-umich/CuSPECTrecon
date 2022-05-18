# fftconv_gpu.jl
# test FFT-based convolution and
# adjoint consistency for FFT convolution methods on very small case

using SPECTrecon: fft_conv!, fft_conv_adj!, plan_psf
using LinearAlgebra: dot


@testset "fftconv!-gpu" begin
    T = Float32
    nx = 12
    nz = 8
    px = 7
    pz = 7
    for i = 1:4
        img = randn(T, nx, nz)
        ker = rand(T, px, pz)
        ker_sym = ker .+ reverse(ker, dims=:)
        ker_sym /= sum(ker_sym)
        out = similar(img)
        out_gpu = CuArray(out)
        img_gpu = CuArray(img)
        ker_sym_gpu = CuArray(ker_sym)

        plan = plan_psf( ; nx, nz, px, pz, T, nthread = 1)[1]
        fft_conv!(out, img, ker_sym, plan)
        plan_gpu = PlanFFT_gpu( ; nx, nz, px, pz, T)
        fft_conv_gpu!(out_gpu, img_gpu, ker_sym_gpu, plan_gpu)
        @test isapprox(out, Array(out_gpu))
    end
end


@testset "adjoint-fftconv!-gpu" begin
    nx = 20
    nz = 14
    px = 5
    pz = 5
    T = Float32
    for i = 1:4 # test with different kernels
        x = cu(randn(T, nx, nz))
        out_x = similar(x)
        y = cu(randn(T, nx, nz))
        out_y = similar(y)
        ker = cu(rand(T, px, pz))
        ker = ker .+ reverse(reverse(ker, dims=1), dims=2)
        ker /= sum(ker)
        plan = PlanFFT_gpu( ; nx, nz, px, pz, T)
        fft_conv_gpu!(out_x, x, ker, plan)
        fft_conv_adj_gpu!(out_y, y, ker, plan)

        @test dot(Array(out_x), Array(y)) ≈ dot(Array(out_y), Array(x))
    end
end
