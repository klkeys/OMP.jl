###################################
### orthogonal matching pursuit ###
###################################

"A container object for all temporary arrays used in `omp`."
type OMPVariables{T <: Float}
    r    :: DenseVector{T}   # vector of residuals
    idxs :: DenseVector{Int} # vector of active indices
    a    :: DenseVector{T}   # temp vector x*b
    b    :: DenseVector{T}   # (sparse) model b
    dots :: DenseVector{T}   # dot products x'*r
end

"A funtion to create an `OMPVariables` object from a matrix `x`, a vector `y`, and a desired sparsity level `k`."
function OMPVariables{T <: Float}(
    x :: DenseMatrix{T},
    y :: DenseVector{T},
    k :: Int
)
    n,p  = size(x)
    r    = copy(y)
    idxs = ones(Int, k) 
    a    = zeros(T, n)
    b    = zeros(T, p)
    dots = zeros(T, p)
    OMPVariables{T}(r, idxs, a, b, dots)
end

"""
    omp!(w::OMPVariables, x, y, k [, quiet::Bool = true])

This function computes **one step** of OMP; that is, it greedily computes the vector `b` of sparsity level `k` that minimizes `norm(y - x*b)`.
As a result, `omp!` **requires** computation of previous 1, 2, ..., k-1 indices in order to work correctly!
Thus, `omp!` is best used within a loop or a function such as `omp`.
"""
function omp!{T <: Float}(
    w :: OMPVariables{T},
    x :: DenseMatrix{T},
    y :: DenseVector{T},
    k :: Int;
)
    # dot_p = x' * r
    # need this for basis pursuit, since OMP selects dot product of largest magnitude
    At_mul_B!(w.dots, x, w.r)

    # compute index of next index to add to support 
    λ = indmax_abs(w.dots)

    # expand active set indices Λ = [Λ i]
    w.idxs[k] = λ 
    idx       = w.idxs[1:k]

    # get subset of x corresponding to active set
    temp = x[:, idx]
    #temp = view(x, :, idx) ### this variant burns CPU to avoid allocating memory; only good for biiiiig matrices

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
    omp(x,y,k) -> SparseMatrixCSC 

Perform *o*rthogonal *m*atching *p*ursuit using a matrix `x`, a response vector `y`, and a sparsity level `k`.
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
    x :: DenseMatrix{T},
    y :: DenseVector{T},
    k :: Int;
    quiet::Bool = true
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
