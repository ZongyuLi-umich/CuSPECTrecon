# backproject_gpu.jl

using SPECTrecon: SPECTplan, backproject!
using LinearAlgebra: dot

@testset "backproject-gpu" begin
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
    image = zeros(T, nx, ny, nz) # images must be initialized to zero
    views = rand(T, nx, nz, nview)
    image_gpu = CuArray(zeros(T, nx, ny, nz))
    views_gpu = CuArray(views)

    mumap_gpu = CuArray(mumap)
    psfs_gpu = CuArray(psfs)
    plan_gpu = SPECTplan_gpu(mumap_gpu, psfs_gpu, dy; T)

    backproject!(image, views, plan)
    backproject_gpu!(image_gpu, views_gpu, plan_gpu)
    @test isapprox(Array(image_gpu), image; rtol = 0.6)
end


@testset "adjoint-project" begin
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

    mumap_gpu = CuArray(mumap)
    psfs_gpu = CuArray(psfs)
    plan_gpu = SPECTplan_gpu(mumap_gpu, psfs_gpu, dy; T)

    image = CuArray(rand(T, nx, ny, nz))
    backimage = CuArray(zeros(T, nx, ny, nz))
    views = CuArray(rand(T, nx, nz, nview))
    forviews = CuArray(zeros(T, nx, nz, nview))

    project_gpu!(forviews, image, plan_gpu)
    backproject_gpu!(backimage, views, plan_gpu)
    @test isapprox(dot(forviews, views), dot(backimage, image); rtol = 1e-2)
end
