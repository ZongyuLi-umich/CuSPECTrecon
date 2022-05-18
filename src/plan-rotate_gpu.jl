# plan-rotate-gpu.jl

using CUDA
export PlanRotate_gpu

"""
    PlanRotate_gpu{T, G2}
Struct for storing GPU work arrays and factors for 2D square image rotation
- `T` datatype of work arrays (default `Float32`)
- `nx::Int` image size
- `padsize::Int` : pad each side of image by this much
- `center::Int`: center of the (padded) image
- `xaxis [nx + 2 * padsize, nx + 2 * padsize]` the index array for x axis
- `yaxis [nx + 2 * padsize, nx + 2 * padsize]` the index array for y axis
- `workmat1 [nx + 2 * padsize, nx + 2 * padsize]` padded work matrix
- `workmat2 [nx + 2 * padsize, nx + 2 * padsize]` padded work matrix
- `workmat3 [nx + 2 * padsize, nx + 2 * padsize]` padded work matrix
- `padded_src`: matrix storing the padded input image
- `query!`: function that returns the value of query indices 
"""
struct PlanRotate_gpu{T, G2}
    nx::Int
    padsize::Int
    center::T
    xaxis::G2
    yaxis::G2
    workmat1::G2
    workmat2::G2
    workmat3::G2
    padded_src::G2
    query!::Function

    function PlanRotate_gpu(
        nx::Int ;
        T::DataType = Float32,
    )
        # only support the case that the image is square
        padsize = ceil(Int, 1 + nx * sqrt(2)/2 - nx / 2)

        px = nx + 2 * padsize
        center = (1 + px) / 2
        axis = T.((1:px) .- center)
        idxarray = CuArray([(i, j) for i in axis, j in axis])
        xaxis = first.(idxarray)
        yaxis = last.(idxarray)
        padded_src = CuArray{T, 2}(undef, px, px)
        workmat1 = CuArray{T, 2}(undef, px, px)
        workmat2 = CuArray{T, 2}(undef, px, px)
        workmat3 = CuArray{T, 2}(undef, px, px)
        G2 = typeof(workmat1)

        function query!(dst::CuArray, tex::CuTexture, queryidx::CuArray)
            broadcast!(dst, queryidx, Ref(tex)) do idx, tex
                tex[idx...]
            end
        end
        new{T, G2}(nx, padsize, center, xaxis, yaxis, workmat1,
                    workmat2, workmat3, padded_src, query!)
    end
end



"""
    show(io::IO, ::MIME"text/plain", plan::PlanRotate_gpu)
"""
function Base.show(io::IO, ::MIME"text/plain", plan::PlanRotate_gpu{T, G2}) where {T, G2}
    t = typeof(plan)
    println(io, t)
    for f in (:nx, :padsize, :center)
        p = getproperty(plan, f)
        t = typeof(p)
        println(io, " ", f, "::", t, " ", p)
    end
    for f in (:xaxis, :yaxis, :workmat1, :workmat2, :workmat3)
        p = getproperty(plan, f)
        println(io, " ", f, ":", " ", summary(p))
    end
    println(io, " (", sizeof(plan), " bytes)")
end


"""
    sizeof(::PlanRotate_gpu)
Show size in bytes of `PlanRotate_gpu` object.
"""
function Base.sizeof(ob::T) where {T <: PlanRotate_gpu}
    sum(f -> sizeof(getfield(ob, f)), fieldnames(typeof(ob)))
end
