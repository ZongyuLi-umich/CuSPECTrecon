# helper_gpu.jl

using SPECTrecon: rot180!, rot_f90!, rotl90!, rotr90!
using OffsetArrays
using ImageFiltering: BorderArray, Fill, Pad


@testset "padzero-gpu!" begin
    T = Float32
    x = randn(T, 7, 5)
    x_gpu = CuArray(x)
    y = randn(T, 3, 3)
    y_gpu = CuArray(y)
    padzero_gpu!(x_gpu, y_gpu, (2, 2, 1, 1))
    z = OffsetArrays.no_offset_view(BorderArray(y, Fill(0, (2, 1), (2, 1))))
    x_cpu = Array(x_gpu)
    @test x_cpu == z
end


@testset "padrepl-gpu!" begin
    T = Float32
    x = randn(T, 10, 9)
    x_gpu = CuArray(x)
    y = randn(T, 5, 4)
    y_gpu = CuArray(y)
    padrepl_gpu!(x_gpu, y_gpu, (1, 4, 3, 2)) # up, down, left, right
    z = OffsetArrays.no_offset_view(BorderArray(y, Pad(:replicate, (1, 3), (4, 2)))) # up, left, down, right
    x_cpu = Array(x_gpu)
    @test x_cpu == z
end


@testset "pad2sizezero-gpu!" begin
    T = Float32
    x = reshape(Int16(1):Int16(15), 5, 3)
    x_gpu = CuArray(x)
    padsize = (8, 6)
    z = CuArray(randn(T, padsize))
    pad2sizezero_gpu!(z, x, padsize)
    tmp = OffsetArrays.no_offset_view(BorderArray(x, Fill(0, (2, 2), (1, 1))))
    z_cpu = Array(z)
    @test tmp == z_cpu
end


@testset "plus1di-gpu" begin
    T = Float32
    x = randn(T, 4, 9)
    x_gpu = CuArray(x)
    v = randn(T, 9)
    v_gpu = CuArray(v)
    y = x[2, :] .+ v
    plus1di_gpu!(x_gpu, v_gpu, 2)
    x_cpu = Array(x_gpu)
    @test x_cpu[2, :] == y
end


@testset "plus1dj-gpu!" begin
    T = Float32
    x = randn(T, 9, 4)
    x_gpu = CuArray(x)
    v = randn(T, 9)
    v_gpu = CuArray(v)
    y = x[:, 2] .+ v
    plus1dj_gpu!(x_gpu, v_gpu, 2)
    x_cpu = Array(x_gpu)
    @test x_cpu[:, 2] == y
end


@testset "plus2di-gpu!" begin
    T = Float32
    x = randn(9)
    x_gpu = CuArray(x)
    v = randn(4, 9)
    v_gpu = CuArray(v)
    y = x .+ v[2, :]
    plus2di_gpu!(x_gpu, v_gpu, 2)
    x_cpu = Array(x_gpu)
    @test x_cpu == y
end


@testset "plus2dj-gpu!" begin
    T = Float32
    x = randn(T, 9)
    x_gpu = CuArray(x)
    v = randn(T, 9, 4)
    v_gpu = CuArray(v)
    y = x .+ v[:, 2]
    plus2dj_gpu!(x_gpu, v_gpu, 2)
    x_cpu = Array(x_gpu)
    @test x_cpu == y
end


@testset "plus3di-gpu!" begin
    T = Float32
    x = randn(T, 9, 7)
    x_gpu = CuArray(x)
    v = randn(T, 4, 9, 7)
    v_gpu = CuArray(v)
    y = x .+ v[2, :, :]
    plus3di_gpu!(x_gpu, v_gpu, 2)
    x_cpu = Array(x_gpu)
    @test x_cpu == y
end


@testset "plus3dj-gpu!" begin
    T = Float32
    x = randn(T, 9, 7)
    x_gpu = CuArray(x)
    v = randn(T, 9, 4, 7)
    v_gpu = CuArray(v)
    y = x .+ v[:, 2, :]
    plus3dj_gpu!(x_gpu, v_gpu, 2)
    x_cpu = Array(x_gpu)
    @test x_cpu == y
end


@testset "plus3dk-gpu!" begin
    T = Float32
    x = randn(T, 9, 7)
    x_gpu = CuArray(x)
    v = randn(T, 9, 7, 4)
    v_gpu = CuArray(v)
    y = x .+ v[:, :, 2]
    plus3dk_gpu!(x_gpu, v_gpu, 2)
    x_cpu = Array(x_gpu)
    @test x_cpu == y
end


@testset "scale3dj-gpu!" begin
    T = Float32
    x = randn(T, 9, 7)
    x_gpu = CuArray(x)
    v = randn(T, 9, 4, 7)
    v_gpu = CuArray(v)
    s = -0.5
    y = s * v[:, 2, :]
    scale3dj_gpu!(x_gpu, v_gpu, 2, s)
    x_cpu = Array(x_gpu)
    @test x_cpu == y
end


@testset "mul3dj-gpu!" begin
    T = Float32
    x = randn(T, 9, 4, 7)
    x_gpu = CuArray(x)
    v = randn(T, 9, 7)
    v_gpu = CuArray(v)
    y = x[:,2,:] .* v
    mul3dj_gpu!(x_gpu, v_gpu, 2)
    x_cpu = Array(x_gpu[:,2,:])
    @test x_cpu == y
end


@testset "copy3dj-gpu!" begin
    T = Float32
    x = randn(T, 9, 7)
    x_gpu = CuArray(x)
    v = randn(T, 9, 4, 7)
    v_gpu = CuArray(v)
    y = v[:,2,:]
    copy3dj_gpu!(x_gpu, v_gpu, 2)
    x_cpu = Array(x_gpu)
    @test x_cpu == y
end


@testset "rotl90-gpu!" begin
    T = Float32
    N = 20
    A = rand(T, N, N)
    B = similar(A, N, N)
    A_gpu = CuArray(A)
    B_gpu = CuArray(B)
    rotl90_gpu!(B_gpu, A_gpu)
    rotl90!(B, A)
    @test isequal(B, Array(B_gpu))
end


@testset "rotr90-gpu!" begin
    T = Float32
    N = 20
    A = rand(T, N, N)
    B = similar(A)
    A_gpu = CuArray(A)
    B_gpu = CuArray(B)
    rotr90_gpu!(B_gpu, A_gpu)
    rotr90!(B, A)
    @test isequal(B, Array(B_gpu))
end


@testset "rot180-gpu!" begin
    T = Float32
    N = 20
    A = rand(T, N, N)
    B = similar(A)
    A_gpu = CuArray(A)
    B_gpu = CuArray(B)
    rot180_gpu!(B_gpu, A_gpu)
    rot180!(B, A)
    @test isequal(B, Array(B_gpu))
end


@testset "rot_f90-gpu!" begin
    T = Float32
    N = 20
    A = CuArray(rand(T, N, N))
    B = CuArray(rand(T, N, N))
    @test_throws String rot_f90_gpu!(A, B, -1)
    @test_throws String rot_f90_gpu!(A, B, 4)
end
