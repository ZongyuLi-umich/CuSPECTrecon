# spectplan_gpu.jl

export SPECTplan_gpu

"""
    SPECTplan_gpu
Struct for storing key factors for a SPECT system model
- `T` datatype of work arrays
- `imgsize` size of image
- `px,pz` psf dimension
- `imgr [nx, ny, nz]` 3D rotated version of image
- `add_img [nx, ny, nz]` 3D image for adding views and backprojection
- `mumap [nx,ny,nz]` attenuation map, must be 3D, possibly zeros()
- `mumapr [nx, ny, nz]` 3D rotated mumap
- `exp_mumapr [nx, nz]` 2D exponential rotated mumap
- `psfs [px,pz,ny,nview]` point spread function, must be 4D, with `px and `pz` odd, and symmetric for each slice
- `nview` number of views, must be integer
- `viewangle` set of view angles, must be from 0 to 2π
- `dy` voxel size in y direction (dx is the same value)
- `planrot` Vector of struct `PlanRotate_gpu`
- `planpsf` Vector of struct `PlanFFT_gpu`
Currently code assumes the following:
* each of the `nview` projection views is `[nx,nz]`
* `nx = ny`
* uniform angular sampling
* `psf` is symmetric
"""
struct SPECTplan_gpu{T,A2,A3,A4}
    T::DataType # default type for work arrays etc.
    imgsize::NTuple{3, Int}
    px::Int
    pz::Int
    imgr::A3 # 3D rotated image, (nx, ny, nz)
    add_img::A3
    add_view::A2
    mumap::A3 # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    mumapr::A3 # 3D rotated mumap, (nx, ny, nz)
    exp_mumapr::A2 # 2D exponential rotated mumap, (nx, ny)
    psfs::A4 # PSFs must be 4D, [px, pz, ny, nview], finally be centered psf
    nview::Int # number of views
    viewangle::StepRangeLen{T}
    dy::T
    planrot::PlanRotate_gpu # todo: make it concrete?
    planpsf::PlanFFT_gpu

    """
        SPECTplan_gpu(mumap, psfs, dy; T, viewangle)
    """
    function SPECTplan_gpu(
        mumap::AbstractArray{<:RealU,3},
        psfs::AbstractArray{<:RealU,4},
        dy::RealU;
        T::DataType = promote_type(eltype(mumap), eltype(psfs), Float32),
        viewangle::StepRangeLen{<:RealU} = (0:size(psfs, 4) - 1) / size(psfs, 4) * T(2π), # set of view angles
        )

        # convert to the same type
        dy = convert(T, dy)
        mumap .= T.(mumap)
        psfs .= T.(psfs)

        # convert to gpu if not
        mumap = typeof(mumap) <: CuArray ? mumap : CuArray(mumap)
        psfs = typeof(psfs) <: CuArray ? psfs : CuArray(psfs)

        (nx, ny, nz) = size(mumap) # typically 128 x 128 x 81

        isequal(nx, ny) || throw("nx != ny")
        (iseven(nx) && iseven(ny)) || throw("nx odd")

        imgsize = (nx, ny, nz)
        # check psf
        px, pz, _, nview = size(psfs)
        (isodd(px) && isodd(pz)) || throw("non-odd size psfs")
        psfs == reverse(reverse(psfs, dims = 1), dims = 2) ||
            throw("asym. psf")
        # all(mapslices(x -> x == reverse(x, dims=:), psfs, dims = [1, 2])) ||
        #     throw("asym. psf")
        # imgr stores 3D image in different view angles
        imgr = CuArray{T, 3}(undef, nx, ny, nz)
        # add_img stores 3d image for backprojection
        add_img = CuArray{T, 3}(undef, nx, ny, nz)
        # mumapr stores 3D mumap in different view angles
        mumapr = CuArray{T, 3}(undef, nx, ny, nz)

        A2 = typeof(mumap[:,:,1])
        A3 = typeof(mumap)
        A4 = typeof(psfs)

        exp_mumapr = CuArray{T, 2}(undef, nx, nz)
        add_view = CuArray{T, 2}(undef, nx, nz)

        planrot = PlanRotate_gpu(nx; T)

        planpsf = PlanFFT_gpu(; nx, nz, px, pz, T)

        new{T, A2, A3, A4}(
            T, # default type for work arrays etc.
            imgsize,
            px,
            pz,
            imgr, # 3D rotated image, (nx, ny, nz)
            add_img,
            add_view,
            mumap, # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
            mumapr, # 3D rotated mumap, (nx, ny, nz)
            exp_mumapr,
            psfs, # PSFs must be 4D, [px, pz, ny, nview], finally be centered psf
            nview, # number of views
            viewangle,
            dy,
            planrot,
            planpsf,
        )
    end
end


"""
    gen_attenuation_gpu!(plan, y)
Generate depth-dependent attenuation map
"""
function gen_attenuation_gpu!(plan::SPECTplan_gpu, y::Int)
    scale3dj_gpu!(plan.exp_mumapr, plan.mumapr, y, -0.5)
    for j = 1:y
        plus3dj_gpu!(plan.exp_mumapr, plan.mumapr, j)
    end

    broadcast!(*, plan.exp_mumapr, plan.exp_mumapr, - plan.dy)
    broadcast!(exp, plan.exp_mumapr, plan.exp_mumapr)
    return plan.exp_mumapr
end


"""
    show(io::IO, ::MIME"text/plain", plan::SPECTplan_gpu)
"""
function Base.show(io::IO, ::MIME"text/plain", plan::SPECTplan_gpu{T,A2,A3,A4}) where {T,A2,A3,A4}
    t = typeof(plan)
    println(io, t)
    for f in (:imgsize, :px, :pz, :nview, :viewangle, :dy)
        p = getproperty(plan, f)
        t = typeof(p)
        println(io, " ", f, "::", t, " ", p)
    end
    for f in (:mumap, )
        p = getproperty(plan, f)
        println(io, " ", f, ":", " ", summary(p))
    end
    println(io, " (", sizeof(plan), " bytes)")
end


"""
    sizeof(::SPECTplan_gpu)
Show size in bytes of `SPECTplan_gpu` object.
"""
function Base.sizeof(ob::T) where {T <: Union{SPECTplan_gpu}}
    sum(f -> sizeof(getfield(ob, f)), fieldnames(typeof(ob)))
end
