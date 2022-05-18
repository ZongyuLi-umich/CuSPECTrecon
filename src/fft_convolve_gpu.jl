# fft_convolve_gpu.jl

export fft_conv_gpu!, fft_conv_adj_gpu!
export fft_conv_gpu, fft_conv_adj_gpu

"""
    imfilterz_gpu!(plan)
GPU version of FFT-based convolution of `plan.img_compl`
and kernel `plan.ker_compl` (not centered),
storing result in `plan.workmat`.
"""
function imfilterz_gpu!(plan::PlanFFT_gpu)
    mul!(plan.img_compl, plan.fft_plan, plan.img_compl)
    mul!(plan.ker_compl, plan.fft_plan, plan.ker_compl)
    broadcast!(*, plan.img_compl, plan.img_compl, plan.ker_compl)
    mul!(plan.img_compl, plan.ifft_plan, plan.img_compl)
    fftshift!(plan.ker_compl, plan.img_compl)
    plan.workmat .= real.(plan.ker_compl)
    return plan.workmat
end


"""
    fft_conv_gpu!(output, img, ker, plan)
Convolve 2D image `img` with 2D (symmetric!) kernel `ker` using FFT,
storing the result in `output`.
"""
function fft_conv_gpu!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    ker::AbstractMatrix{<:RealU},
    plan::PlanFFT_gpu,
)
    @boundscheck size(output) == size(img) || throw("size output")
    @boundscheck size(img) == (plan.nx, plan.nz) || throw("size img")
    @boundscheck size(ker) == (plan.px, plan.pz) ||
        throw("size ker $(size(ker)) $(plan.px) $(plan.pz)")

    # filter image with a kernel, using replicate padding and fft convolution
    padrepl_gpu!(plan.img_compl, img, plan.padsize)

    pad2sizezero_gpu!(plan.ker_compl, ker, size(plan.ker_compl)) # zero pad kernel

    imfilterz_gpu!(plan)

    (M, N) = size(img)
    copyto!(output, (@view plan.workmat[plan.padsize[1] .+ (1:M),
                                        plan.padsize[3] .+ (1:N)]))
    return output
end


"""
    fft_conv_gpu(img, ker; T)
GPU version of convolving 2D image `img` with 2D (symmetric!) kernel `ker` using FFT.
"""
function fft_conv_gpu(
    img::AbstractMatrix{I},
    ker::AbstractMatrix{K};
    T::DataType = promote_type(I, K, Float32),
) where {I <: Number, K <: Number}

    ker ≈ reverse(ker, dims=:) || throw("asymmetric kernel")
    nx, nz = size(img)
    px, pz = size(ker)
    plan = PlanFFT_gpu( ; nx, nz, px, pz, T)
    output = similar(img)
    fft_conv_gpu!(output, img, ker, plan)
    return output
end


"""
    fft_conv_adj_gpu!(output, img, ker, plan)
GPU version of the adjoint of `fft_conv!`.
"""
function fft_conv_adj_gpu!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    ker::AbstractMatrix{<:RealU},
    plan::PlanFFT_gpu{T},
) where {T}

    @boundscheck size(output) == size(img) || throw("size output")
    @boundscheck size(img) == (plan.nx, plan.nz) || throw("size img")
    @boundscheck size(ker) == (plan.px, plan.pz) ||
        throw("size ker $(size(ker)) $(plan.px) $(plan.pz)")

    padzero_gpu!(plan.img_compl, img, plan.padsize) # pad the image with zeros
    pad2sizezero_gpu!(plan.ker_compl, ker, size(plan.ker_compl)) # pad the kernel with zeros

    imfilterz_gpu!(plan)
    (M, N) = size(img)
    # adjoint of replicate padding
    plan.workvecz .= zero(T)
    for i = 1:plan.padsize[1]
        plus2di_gpu!(plan.workvecz, plan.workmat, i)
    end
    plus1di_gpu!(plan.workmat, plan.workvecz, 1+plan.padsize[1])

    plan.workvecz .= zero(T)
    for i = (plan.padsize[1]+M+1):size(plan.workmat, 1)
        plus2di_gpu!(plan.workvecz, plan.workmat, i)
    end
    plus1di_gpu!(plan.workmat, plan.workvecz, M+plan.padsize[1])

    plan.workvecx .= zero(T)
    for j = 1:plan.padsize[3]
        plus2dj_gpu!(plan.workvecx, plan.workmat, j)
    end
    plus1dj_gpu!(plan.workmat, plan.workvecx, 1+plan.padsize[3])

    plan.workvecx .= zero(T)
    for j = (plan.padsize[3]+N+1):size(plan.workmat, 2)
        plus2dj_gpu!(plan.workvecx, plan.workmat, j)
    end
    plus1dj_gpu!(plan.workmat, plan.workvecx, N+plan.padsize[3])

    copyto!(output,
        (@view plan.workmat[(plan.padsize[1]+1):(plan.padsize[1]+M),
                            (plan.padsize[3]+1):(plan.padsize[3]+N)]),
    )

    return output
end


"""
    fft_conv_adj_gpu(img, ker; T)
GPU version of the adjoint of `fft_conv`.
"""
function fft_conv_adj_gpu(
    img::AbstractMatrix{I},
    ker::AbstractMatrix{K};
    T::DataType = promote_type(I, K, Float32),
) where {I <: Number, K <: Number}

    ker ≈ reverse(ker, dims=:) || throw("asymmetric kernel")
    nx, nz = size(img)
    px, pz = size(ker)
    plan = PlanFFT_gpu( ; nx, nz, px, pz, T)
    output = similar(Matrix{T}, size(img))
    fft_conv_adj_gpu!(output, img, ker, plan)
    return output
end
