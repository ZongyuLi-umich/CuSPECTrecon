# backproject_gpu.jl

export backproject_gpu, backproject_gpu!


"""
    backproject_gpu!(image, view, plan, viewidx)
Backproject a single view.
"""
function backproject_gpu!(
    image::AbstractArray{<:RealU, 3},
    view::AbstractMatrix{<:RealU},
    plan::SPECTplan_gpu,
    viewidx::Int,
)

    for z = 1:plan.imgsize[3] # 1:nz
        # rotate mumap by "-angle"
        imrotate_gpu!((@view plan.mumapr[:, :, z]),
                      (@view plan.mumap[:, :, z]),
                      -plan.viewangle[viewidx],
                      plan.planrot)
    end

    # adjoint of convolving img with psf and applying attenuation map
    for y = 1:plan.imgsize[2] # 1:ny

        gen_attenuation_gpu!(plan::SPECTplan_gpu, y::Int)

        fft_conv_adj_gpu!((@view plan.imgr[:, y, :]),
                          view,
                          (@view plan.psfs[:, :, y, viewidx]),
                          plan.planpsf)

        mul3dj_gpu!(plan.imgr, plan.exp_mumapr, y)
    end

    # adjoint of rotating image, again, rotating by "-angle"
    for z = 1:plan.imgsize[3] # 1:nz

        imrotate_gpu!((@view image[:, :, z]),
                      (@view plan.imgr[:, :, z]),
                      -plan.viewangle[viewidx],
                      plan.planrot)
    end

    return image
end


"""
    backproject_gpu!(image, views, plan ; viewlist)
Backproject multiple views into `image`.
Users must initialize `image` to zero.
"""
function backproject_gpu!(
    image::AbstractArray{<:RealU, 3},
    views::AbstractArray{<:RealU, 3},
    plan::SPECTplan_gpu;
    viewlist::AbstractVector{<:Int} = 1:plan.nview, # all views
)

    # loop over each view index
    for viewidx in viewlist
        backproject_gpu!(plan.add_img, (@view views[:, :, viewidx]), plan, viewidx)
        broadcast!(+, image, image, plan.add_img)
    end
    return image
end


"""
    image = backproject_gpu(views, plan ; kwargs...)
SPECT backproject `views`; this allocates the returned 3D array.
"""
function backproject_gpu(
    views::AbstractArray{<:RealU, 3},
    plan::SPECTplan_gpu;
    kwargs...,
)
    image = CuArray(zeros(plan.T, plan.imgsize))
    backproject_gpu!(image, views, plan; kwargs...)
    return image
end


"""
    image = backproject_gpu(views, mumap, psfs, dy; kwargs...)
SPECT backproject `views` using attenuation map `mumap` and PSF array `psfs` for pixel size `dy`.
This method initializes the `plan` as a convenience.
Most users should use `backproject_gpu!` instead after initializing those, for better efficiency.
"""
function backproject_gpu(
    views::AbstractArray{<:RealU, 3}, # [nx,nz,nview]
    mumap::AbstractArray{<:RealU, 3}, # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:RealU, 4},
    dy::RealU;
    kwargs...,
)

    size(mumap,1) == size(mumap,1) == size(views,1) ||
        throw(DimensionMismatch("nx"))
    size(mumap,3) == size(views,2) || throw(DimensionMismatch("nz"))
    plan = SPECTplan_gpu(mumap, psfs, dy; kwargs...)
    return backproject_gpu(views, plan; kwargs...)
end
