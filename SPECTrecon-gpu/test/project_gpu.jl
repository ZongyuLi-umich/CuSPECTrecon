# project_gpu.jl

using SPECTrecon: SPECTplan, project!


@testset "project-gpu" begin
    T = Float32
    nx = 8; ny = nx
    nz = 6
    nview = 7

    mumap = rand(T, nx, ny, nz)

    px = 5
    pz = 3
    psfs = rand(T, px, pz, ny, nview)
    psfs = psfs .+ mapslices(reverse, psfs, dims = [1, 2]) # symmetrize
    psfs = psfs ./ mapslices(sum, psfs, dims = [1, 2])

    dy = T(4.7952)
    plan = SPECTplan(mumap, psfs, dy; T, interpmeth = :two, mode = :fast)
    x = randn(T, nx, ny, nz)
    views = zeros(T, nx, nz, nview)

    mumap_gpu = CuArray(mumap)
    psfs_gpu = CuArray(psfs)
    plan_gpu = SPECTplan_gpu(mumap_gpu, psfs_gpu, dy; T)
    x_gpu = CuArray(x)
    views_gpu = CuArray(zeros(T, nx, nz, nview))
    project!(views, x, plan)
    project_gpu!(views_gpu, x_gpu, plan_gpu)
    @test isapprox(Array(views_gpu), views; rtol = 1e-2)
end
