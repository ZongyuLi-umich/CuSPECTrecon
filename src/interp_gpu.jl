# interp_gpu.jl
# Perform linear interpolation for GPU arrays
using CUDA
export interpolator

struct interp_gpu{T}
    src::CuArray{T}
    tex::CuTexture{T}
    query::Function
    query!::Function
    function interp_gpu(src::AbstractArray;
                          interpmeth::Symbol = :linear,
                          extrapolation::Symbol = :zero)
        interp_dict = [:nearest, :linear, :cubic]
        interp_func = [CUDA.NearestNeighbour(), CUDA.LinearInterpolation(), CUDA.CubicInterpolation()]
        extra_dict = [:zero, :replicate]
        extra_func = [CUDA.ADDRESS_MODE_BORDER, CUDA.ADDRESS_MODE_CLAMP]
        @assert interpmeth in interp_dict || throw("invalid interp method!")
        @assert extrapolation in extra_dict || throw("invalid extrapolation!")
        T = eltype(src)
        src = typeof(src) <: CuArray ? src : CuArray(src)
        tex = CuTexture(CuTextureArray(src);
                        interpolation = interp_func[findall(x->x==interpmeth, interp_dict)[1]],
                        address_mode = extra_func[findall(x->x==extrapolation, extra_dict)[1]])
        function query(queryidx::Array)
            queryidx = CuArray(queryidx)
            dst = CuArray{T}(undef, size(queryidx))
            broadcast!(dst, queryidx, Ref(tex)) do idx, tex
                tex[idx...]
            end
            return dst
        end
        function query!(dst::CuArray, queryidx::CuArray)
            broadcast!(dst, queryidx, Ref(tex)) do idx, tex
                tex[idx...]
            end
        end
        new{T}(src, tex, query, query!)
    end
end
