# rotatez.jl

using SPECTrecon: imrotate!, plan_rotate

@testset "rotate-gpu" begin
    nx = 20
    θ_list = (1:23) / 12 * π
    T = Float32
    image2 = rand(T, nx, nx)
    plan = plan_rotate(nx; T, method = :two)[1]
    result = similar(image2)
    image2_gpu = CuArray(image2)
    result_gpu = similar(image2_gpu)
    plan_gpu = PlanRotate_gpu(nx; T)
    for θ in θ_list
        imrotate!(result, image2, θ, plan)
        imrotate_gpu!(result_gpu, image2_gpu, θ, plan_gpu)
        @test isapprox(result, Array(result_gpu); rtol = 1e-2)
    end
end
