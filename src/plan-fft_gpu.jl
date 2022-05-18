# plan-fft-gpu.jl

export PlanFFT_gpu
import AbstractFFTs
import FFTW
using CUDA

"""
    PlanFFT_gpu{T,Tf,Ti}( ; nx::Int, nz::Int, px::Int, pz::Int, T::DataType)
Struct for storing work arrays and factors for 2D convolution for one thread.
Each PSF is `px × pz`
- `T` datatype of work arrays (subtype of `AbstractFloat`)
- `nx::Int = 128` (`ny` implicitly the same)
- `nz::Int = nx` image size is `[nx,nx,nz]`
- `px::Int = 1`
- `pz::Int = px` (PSF size)
- `padsize::Tuple` : `(padup, paddown, padleft, padright)`
- `workmat [nx+padup+paddown, nz+padleft+padright]` 2D padded image for FFT convolution
- `workvecx [nx+padup+paddown,]`: 1D work vector
- `workvecz [nz+padleft+padright,]`: 1D work vector
- `img_compl [nx+padup+paddown, nz+padleft+padright]`: 2D [complex] padded image for FFT
- `ker_compl [nx+padup+paddown, nz+padleft+padright]`: 2D [complex] padded image for FFT
- `fft_plan::Tf` plan for doing FFT; see `plan_fft!`
- `ifft_plan::Ti` plan for doing IFFT; see `plan_ifft!`
"""
struct PlanFFT_gpu{T, A1, A2, C2, Tf, Ti}
    nx::Int
    nz::Int
    px::Int
    pz::Int
    padsize::NTuple{4, Int}
    workmat::A2
    workvecx::A1
    workvecz::A1
    img_compl::C2
    ker_compl::C2
    fft_plan::Tf # Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}
    ifft_plan::Ti # Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}

    function PlanFFT_gpu( ;
        nx::Int = 128,
        nz::Int = nx,
        px::Int = 1,
        pz::Int = px,
        T::DataType = Float32)

        T <: AbstractFloat || throw("invalid T=$T")
        padup = _padup(nx, px)
        paddown = _paddown(nx, px)
        padleft = _padleft(nz, pz)
        padright = _padright(nz, pz)
        padsize = (padup, paddown, padleft, padright)

        workmat = CuArray(Matrix{T}(undef, nx+padup+paddown, nz+padleft+padright))
        workvecx = CuArray(Vector{T}(undef, nx+padup+paddown))
        workvecz = CuArray(Vector{T}(undef, nz+padleft+padright))
        vectype = typeof(workvecz)
        mattype = typeof(workmat)
        # complex padimg
        img_compl = CuArray(Matrix{Complex{T}}(undef, nx+padup+paddown, nz+padleft+padright))
        # complex kernel
        ker_compl = CuArray(Matrix{Complex{T}}(undef, nx+padup+paddown, nz+padleft+padright))
        compmattype = typeof(img_compl)
        fft_plan = plan_fft!(convert(compmattype, ker_compl))
        ifft_plan = plan_ifft!(convert(compmattype, ker_compl))
        Tf = typeof(fft_plan)
        Ti = typeof(ifft_plan)

        new{T, vectype, mattype, compmattype, Tf, Ti}(
            nx,
            nz,
            px,
            pz,
            padsize,
            workmat,
            workvecx,
            workvecz,
            img_compl,
            ker_compl,
            fft_plan,
            ifft_plan,
        )
    end
end


"""
    show(io::IO, ::MIME"text/plain", plan::PlanFFT_gpu)
"""
function Base.show(io::IO, ::MIME"text/plain", plan::PlanFFT_gpu{T, A1, A2, C2, Tf, Ti}) where {T, A1, A2, C2, Tf, Ti}
    t = typeof(plan)
    println(io, t)
    for f in (:nx, :nz, :px, :pz, :padsize)
        p = getproperty(plan, f)
        t = typeof(p)
        println(io, " ", f, "::", t, " ", p)
    end
    for f in (:workmat, :workvecx, :workvecz, :img_compl, :ker_compl, :fft_plan, :ifft_plan)
        p = getproperty(plan, f)
        println(io, " ", f, ":", " ", summary(p))
    end
    println(io, " (", sizeof(plan), " bytes)")
end


"""
    show(io::IO, mime::MIME"text/plain", vp::Vector{<:PlanFFT_gpu})
"""
function Base.show(io::IO, mime::MIME"text/plain", vp::Vector{<: PlanFFT_gpu})
    t = typeof(vp)
    println(io, length(vp), "-element ", t, " with N=", vp[1].nx)
#   show(io, mime, vp[1])
end


"""
    sizeof(::PlanFFT_gpu)
Show size in bytes of `PlanFFT_gpu` object.
"""
function Base.sizeof(ob::T) where {T <: PlanFFT_gpu}
    sum(f -> sizeof(getfield(ob, f)), fieldnames(typeof(ob)))
end
