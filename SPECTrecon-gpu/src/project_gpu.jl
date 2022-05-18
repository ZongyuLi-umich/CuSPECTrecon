# project_gpu.jl

export project_gpu, project_gpu!

"""
    project_gpu!(view, plan, image, viewidx)
GPU version of SPECT projection of `image` into a single `view` with index `viewidx`.
The `view` must be pre-allocated but need not be initialized to zero.
"""
function project_gpu!(
    view::AbstractMatrix{<:RealU},
    image::AbstractArray{<:RealU, 3},
    plan::SPECTplan_gpu,
    viewidx::Int,
)

    for z = 1:plan.imgsize[3]
        # rotate images
        imrotate_gpu!(
            (@view plan.imgr[:, :, z]),
            (@view image[:, :, z]),
            plan.viewangle[viewidx],
            plan.planrot)
        # rotate mumap
        imrotate_gpu!(
            (@view plan.mumapr[:, :, z]),
            (@view plan.mumap[:, :, z]),
            plan.viewangle[viewidx],
            plan.planrot)
    end

    for y = 1:plan.imgsize[2] # 1:ny

        scale3dj_gpu!(plan.exp_mumapr, plan.mumapr, y, -0.5)
        for j = 1:y
            plus3dj_gpu!(plan.exp_mumapr, plan.mumapr, j)
        end

        broadcast!(*, plan.exp_mumapr, plan.exp_mumapr, - plan.dy)
        broadcast!(exp, plan.exp_mumapr, plan.exp_mumapr)
        # apply depth-dependent attenuation
        mul3dj_gpu!(plan.imgr, plan.exp_mumapr, y)

        fft_conv_gpu!(plan.add_view,
                     (@view plan.imgr[:, y, :]),
                     (@view plan.psfs[:, :, y, viewidx]),
                     plan.planpsf)

        view .+= plan.add_view
    end

    return view
end


"""
    project_gpu!(views, image, plan; viewlist)
Project `image` into multiple `views` with indexes `index` (defaults to `1:nview`).
The 3D `views` array must be pre-allocated, but need not be initialized.
"""
function project_gpu!(
    views::AbstractArray{<:RealU,3},
    image::AbstractArray{<:RealU,3},
    plan::SPECTplan_gpu;
    viewlist::AbstractVector{<:Int} = 1:plan.nview, # all views
)

    # loop over each view index
    for viewidx in viewlist
        project_gpu!((@view views[:,:,viewidx]), image, plan, viewidx)
    end

    return views
end


"""
    views = project_gpu(image, plan ; kwargs...)
GPU version of a convenience method for SPECT forward projector that allocates and returns views.
"""
function project_gpu(
    image::AbstractArray{<:RealU,3},
    plan::SPECTplan_gpu;
    kwargs...,
)
    views = CuArray{plan.T}(undef, plan.imgsize[1], plan.imgsize[3], plan.nview)
    project_gpu!(views, image, plan; kwargs...)
    return views
end


"""
    views = project_gpu(image, mumap, psfs, dy; kwargs...)
GPU version of a convenience method for SPECT forward projector that does all allocation
including initializing `plan`.

In
* `image` : 3D array `(nx,ny,nz)`
* `mumap` : `(nx,ny,nz)` 3D attenuation map, possibly zeros()
* `psfs` : 4D PSF array
* `dy::RealU` : pixel size
"""
function project_gpu(
    image::AbstractArray{<:RealU, 3},
    mumap::AbstractArray{<:RealU, 3}, # (nx,ny,nz) 3D attenuation map
    psfs::AbstractArray{<:RealU, 4}, # (px,pz,ny,nview)
    dy::RealU;
    kwargs...,
)
    size(mumap) == size(image) || throw(DimensionMismatch("image/mumap size"))
    size(image,2) == size(psfs,3) || throw(DimensionMismatch("image/psfs size"))
    plan = SPECTplan_gpu(mumap, psfs, dy; kwargs...)
    return project_gpu(image, plan; kwargs...)
end
