"A funtion to create an `OMPVariables` object from a `BEDFile` object `x`, a `SharedVector` `y`, and a desired sparsity level `k`."
function OMPVariables{T <: Float}(
    x :: BEDFile{T},
    y :: SharedVector{T},
    k :: Int;
    pids :: Vector{Int} = procs(x)
)
    n,p  = size(x)
    r    = SharedArray(T, n, pids=pids) :: SharedVector{T}
    copy!(r, y)
    idxs = ones(Int, k) 
    a    = zeros(T, n)
    b    = SharedArray(T, p, pids=pids) :: SharedVector{T}
    dots = SharedArray(T, p, pids=pids) :: SharedVector{T}

    OMPVariables{T}(r, idxs, a, b, dots)
end

"""
    omp!(w::OMPVariables, x::BEDFile, y, k) 

Greedily computes the vector `b` of sparsity level `k` that minimizes `norm(y - x*b), where `x` is a `BEDFile` object`.
This is the kernel of the function `omp`.
"""
function omp!{T <: Float}(
    w    :: OMPVariables{T},
    x    :: BEDFile{T},
    y    :: SharedVector{T},
    k    :: Int;
    m    :: Vector{Int} = ones(Int, length(y)),
    pids :: Vector{Int} = procs(x)
)
    # dot_p = x' * r
    # need this for basis pursuit, since OMP selects dot product of largest magnitude
    At_mul_B!(w.dots, x, w.r, m, pids=pids)

    # compute index of next index to add to support 
    λ = indmax_abs(w.dots)

    # expand active set indices Λ = [Λ i]
    w.idxs[k] = λ 
    idx       = w.idxs[1:k]

    # get subset of x corresponding to active set
    temp = zeros(T, (n,k)) 
    decompress_genotypes!(temp, x, idx)

    # z = argmin_{b} ( norm(y - temp*b) )
    z, = lsqr(temp, y)       

    # r = y - temp*z
    #BLAS.gemv!('N', -one(T), temp, z, zero(T), w.r)
    #BLAS.axpy!(one(T), y, w.r)
    A_mul_B!(w.r, temp, z)
    BLAS.axpy!(-one(T), y, w.r)

    # save current model to b
    # no need to erase w.b beforehand since OMP builds models stepwise,
    # and previous model (sparsity level k-1) is subset of current model of size k
    w.b[idx] = z

    return nothing
end


"""
    omp(x::BEDFile, y, k) -> SparseMatrixCSC 

Perform *o*rthogonal *m*atching *p*ursuit using a BEDFile object `x`, a response vector `y`, and a sparsity level `k`.
This function will compute in greedy fashion all sparsity levels `1, 2, ..., k`.
The result is a matrix of type `SparseMatrixCSC` containing the sparse vector that minimizes `sumsq(y - x*b)` for each model size.

Arguments:
- `x` is the design matrix (dictionary)
- `y` is the response vector (signal)
- `k` is the desired sparsity level

Output:
- `B` is the sparse matrix whose columns provide estimates of `y` with sparisty level `1, 2, ..., k`.
"""
function omp{T <: Float}(
    x :: BEDFile{T},
    y :: DenseVector{T},
    k :: Int;
    m :: Vector{Int} = ones(Int, length(y)),
    quiet :: Bool = true,
)
    # size of problem?
    n,p = size(x)

    # initialize all temporary arrays
    w = OMPVariables(x, y, k)

    # initialize sparse matrix of models
    # will fill this iteratively as we grow our beta
    # result is a "path" of betas
    B = spzeros(T, p, k)

    # compute models 1, 2, ..., k
    for i = 1:k

        # this computes model size i
        omp!(w, x, y, i)

        # output progress if desired
        quiet || @printf("Sparsity level: %d, sum(residuals): %f\n", i, norm(w.r));

        # save model for sparsity level i
        B[:,i] = sparsevec(w.b)
    end

    # return path
    return B
end

### NOT DONE: CROSSVALIDATION ROUTINES
