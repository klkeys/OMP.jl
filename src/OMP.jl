"""
A Julia module to perform *o*rthogonal *m*atching *p*ursuit.
OMP.jl includes some simple crossvalidation facilities to find the best predictive model of a given sparsity level. """
module OMP

using IterativeSolvers # need LSQR
using RegressionTools

export omp
export cv_omp

# typealias to only allow single or double precision OMP
typealias Float Union{Float32, Float64}

# subroutine to compute a default number of folds
@inline cv_get_num_folds(nmin::Int, nmax::Int) = max(nmin, min(Sys.CPU_CORES::Int, nmax))

###################################
### orthogonal matching pursuit ###
###################################

"A subroutine to efficiently compute i = indmax(abs(x))."
function indmax_abs{T <: Float}(x::DenseVector{T})
    a = abs(x[1])
    b = 1
    for i in eachindex(x)
        c = abs(x[i])
        if c > a
            a = c
            b = i
        end
    end
    return b
end


"A container object for all temporary arrays used in `omp`."
type OMPVariables{T <: Float}
    r    :: DenseVector{T}   # vector of residuals
    idxs :: DenseVector{Int} # vector of active indices
    a    :: DenseVector{T}   # temp vector x*b
    b    :: DenseVector{T}   # (sparse) model b
    dots :: DenseVector{T}   # dot products x'*r
end

"A funtion to create an `OMPVariables` object from a matrix `x` and a vector `y`."
function OMPVariables{T <: Float}(
    x :: DenseMatrix{T},
    y :: DenseVector{T}
)
    n,p  = size(x)
    r    = copy(y)
    idxs = Int[]
    a    = zeros(T, n)
    b    = zeros(T, p)
    dots = zeros(T, p)
    OMPVariables{T}(r, idxs, a, b, dots)
end

"""
    omp!(w::OMPVariables, x, y, k [, quiet::Bool = true]) -> Vector

This function computes **one step** of OMP; that is, it greedily computes the vector `b` of sparsity level `k` that minimizes `norm(y - x*b)`.
As a result, `omp!` **requires** computation of previous 1, 2, ..., k-1 indices in order to work correctly!
Thus, `omp!` is best used within a loop or a function such as `omp`.
"""
function omp!{T <: Float}(
    w :: OMPVariables{T},
    x :: DenseMatrix{T},
    y :: DenseVector{T},
    k :: Int;
    quiet::Bool = true
)
    # dot_p = x' * r
    # need this for basis pursuit, since OMP selects dot product of largest magnitude
    At_mul_B!(w.dots, x, w.r)

    # λ = indmax(abs(dots))
    λ = indmax_abs(w.dots)

    # expand active set indices Λ = [Λ i]
    push!(w.idxs, λ)

    # get subset of x corresponding to active set
    temp = x[:, w.idxs]

    # z = argmin_{b} ( norm(y - temp*b) )
    z, = lsqr(temp, y)       

    # r = y - temp*z
    A_mul_B!(w.r, temp, z)
    BLAS.axpy!(one(T), y, w.r)

    # output progress if desired
    quiet || @printf("Iter %d, residual: %f\n", iter, norm(w.r));

    # save current model to b
    # no need to erase w.b beforehand since OMP builds models stepwise,
    # and previous model (sparsity level k-1) is subset of current model of size k
    w.b[w.idxs] = z

    return nothing
end

"""
    omp(x,y,k) -> Vector

Perform *o*rthogonal *m*atching *p*ursuit using a matrix `A`, a response vector `y`, and a sparsity level `k`.
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
    w = OMPVariables(x, y)

    # initialize sparse matrix of models
    # will fill this iteratively as we grow our beta
    # result is a "path" of betas
    B = spzeros(T, p, k)

    # compute models 1, 2, ..., k
    for i = 1:k

        # this computes model size i
        omp!(w, x, y, i, quiet=quiet)

        # save model for sparsity level i
        B[:,i] = sparsevec(w.b)
    end

    # return path
    return B
end


################################
### crossvalidation routines ###
################################

# container object for XV results
immutable OMPCrossvalidationResults{T <: Float}
    mses :: Vector{T}
    path :: Vector{Int}
    b    :: Vector{T}
    bidx :: Vector{Int}
    k    :: Int
end

# subroutine to refit preditors after crossvalidation
function refit_omp{T <: Float}(
    x     :: DenseMatrix{T},
    y     :: DenseVector{T},
    k     :: Int;
    quiet :: Bool = true,
)
    # first use OMP to extract model
    betas = omp(x, y, k, quiet=quiet)

    # which components of β are nonzero?
    # cannot use binary indices here since we need to return Int indices
    bidx = find(betas[:,end])

    # allocate the submatrix of x corresponding to the inferred model
    x_inferred = x[:,bidx]

    # now estimate β with the ordinary least squares estimator β = inv(x'x)x'y
    # return it with the vector of MSEs
    xty = BLAS.gemv('T', x_inferred, sdata(y)) :: Vector{T}
    xtx = At_mul_B(x_inferred, x_inferred) :: Matrix{T}
    b   = zeros(T, length(bidx))
    try
        b = (xtx \ xty) :: Vector{T}
    catch e
        warn("caught error: ", e, "\nSetting returned values of b to -Inf")
        fill!(b, -Inf)
    end
  
    return b, bidx
end

"""
    one_fold(x, y, k, folds, fold) -> Vector

For a regularization path given by the integer `k`,
this function performs orthogonal matching pursuit on `x` and `y` and computes an out-of-sample error based on the indices given in `folds`.

Arguments:
- `x` is the `n` x `p` design matrix.
- `y` is the `n`-vector of responses.
- `k` is an integer that specifies the maximum desired sparsity level to test.
- `folds` is an `Int` vector indicating which component of `y` goes to which fold, e.g. `folds = IHT.cv_get_folds(n,nfolds)`
- `fold` is the current fold to compute.

Optional Arguments:
- `quiet` is a Boolean to activate output. Defaults to `true` (no output).

Output:
- `errors` is a vector of out-of-sample errors (MSEs) for the current fold.
"""
function one_fold{T <: Float}(
    x     :: DenseMatrix{T},
    y     :: DenseVector{T},
    k     :: Int,
    folds :: DenseVector{Int},
    fold  :: Int;
    quiet :: Bool  = true,
)

    # make vector of indices for folds
    test_idx = folds .== fold
    test_size = sum(test_idx)

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx

    # allocate the arrays for the training set
    x_train = x[train_idx,:]
    y_train = y[train_idx]

    # compute the regularization path on the training set
    betas = omp(x_train, y_train, k, quiet=quiet)

    # compute the mean out-of-sample error for the TEST set
    #errors = vec(sumabs2(broadcast(-, y[test_idx], x[test_idx,:] * betas), 1)) ./ (2*test_size)
    xb = view(x, test_idx, :) * betas
    r  = broadcast(-, y[test_idx], xb)
    errors = vec(sumabs2(r, 1)) ./ (2*test_size)

    return errors :: Vector{T}
end

"""
    pfold(x, y, k, folds, q [, pids=procs()]) -> Vector

This function is the parallel execution kernel in `cv_omp`. It is not meant to be called outside of `cv_omp`.
For floating point data `x` and `y` and an integer `k`, `pfold` will distribute `q` crossvalidation folds across the processes supplied by the optional argument `pids`.
It calls `one_fold` for each fold, then collects the vectors of MSEs each process, applys a reduction, and finally returns the average MSE vector across all folds.
"""
function pfold{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    k        :: Int,
    folds    :: DenseVector{Int},
    q        :: Int;
    pids     :: Vector{Int} = procs(),
    quiet    :: Bool  = true,
)
    # how many CPU processes can pfold use?
    np = length(pids)

    # report on CPU processes
    quiet || println("pfold: np = ", np)
    quiet || println("pids = ", pids)

    # set up function to share state (indices of folds)
    i = 1
    nextidx() = (idx=i; i+=1; idx)

    # preallocate cell array for results
    results = SharedArray(T, (k,q), pids=pids) :: SharedMatrix{T}

    # master process will distribute tasks to workers
    # master synchronizes results at end before returning
    @sync begin

        # loop over all workers
        for worker in pids

            # exclude process that launched pfold, unless only one process is available
            if worker != myid() || np == 1

                # asynchronously distribute tasks
                @async begin
                    while true

                        # grab next fold
                        current_fold = nextidx()

                        # if current fold exceeds total number of folds then exit loop
                        current_fold > q && break

                        # report distribution of fold to worker and device
                        quiet || print_with_color(:blue, "Computing fold $current_fold on worker $worker.\n\n")

                        # launch job on worker
                        # worker loads data from file paths and then computes the errors in one fold
                        r = remotecall_fetch(worker) do
                                one_fold(x, y, k, folds, i, quiet=quiet)
                        end # end remotecall_fetch()
                        setindex!(results, r, :, current_fold)
                    end # end while
                end # end @async
            end # end if
        end # end for
    end # end @sync

    # return reduction (row-wise sum) over results
    return (vec(sum(results, 2) ./ q)) :: Vector{T}
end


"""
    cv_omp(x,y) -> Vector

This function will perform `q`-fold cross validation for the ideal model size in OMP least squares regression using the `n` x `p` design matrix `x` and the response vector `y`.
Each path is asynchronously spawned using any available processor.
For each path, one fold is held out of the analysis for testing, while the rest of the data are used for training.
The function to compute each path, `one_fold()`, will return a vector of out-of-sample errors (MSEs).
Arguments:
- `x` is the `n` x `p` design matrix.
- `y` is the `n`-vector of responses.
Optional Arguments:
- `q` is the number of folds to compute. Defaults to `max(3, min(CPU_CORES, 5))`, where `CPU_CORES`is the Julia variable to query the number of available CPU cores.
- `path` is an `Int` vector that specifies which model sizes to include in the path. Defaults to `path = collect(1:min(p,20))`.
- `folds` is the partition of the data. Defaults to `IHT.cv_get_folds(n,q)`.
- `pids`, a vector of process IDs. Defaults to `procs()`, which recruits all available processes.
- `quiet` is a Boolean to activate output. Defaults to `true` (no output).
   *NOTA BENE*: each processor outputs feed to the console without regard to the others,
   so setting `quiet = false` can yield very messy output!
Output:
An `OMPCrossvalidationResults` object with the following fields:
- `mses` is the averaged MSE over all folds.
- `k` is the best crossvalidated model size.
- `path` is the regularization path used in the crossvalidation.
- `b`, a vector of `k` floats
- `bidx`, a vector of `k` indices indicating the support of the best model.
"""
function cv_omp{T <: Float}(
    x     :: DenseMatrix{T},
    y     :: DenseVector{T};
    q     :: Int  = cv_get_num_folds(3,5),
    k     :: Int  = min(size(x,2), 20),
    folds :: DenseVector{Int} = cv_get_folds(sdata(y),q),
    pids  :: Vector{Int} = procs(),
    quiet :: Bool = true,
)
    # do not allow crossvalidation with fewer than 3 folds
    q > 2 || throw(ArgumentError("Number of folds q = $q must be at least 3."))

    # problem dimensions?
    n,p = size(x)

    # check for conformable arrays
    n == length(y) || throw(DimensionMismatch("Row dimension of x ($n) must match number of rows in y ($(length(y)))"))

    # want to compute a path for each fold
    # the folds are computed asynchronously over processes enumerated by pids
    # master process then reduces errors across folds and returns MSEs
    mses = pfold(x, y, k, folds, q, pids=pids, quiet=quiet)

    # what is the best model size?
    path = collect(1:k)
    kbest = convert(Int, floor(mean(path[mses .== minimum(mses)])))

    # print results
    !quiet && print_cv_results(mses, path, k)

    # refit best model
    b, bidx = refit_omp(x, y, kbest, quiet=quiet)

    return OMPCrossvalidationResults{T}(mses, path, b, bidx, kbest)
end

################
### old code ###
################

# this works but we will not use it
# instead use omp! as OMP function
# then omp() will compute entire path of betas
function omp_working_old{T <: Float}(
    Φ :: DenseMatrix{T},
    v :: DenseVector{T},
    m :: Int;
    quiet::Bool = true
)
    n,p = size(Φ);
    β = zeros(T, p);
    r = copy(v);
    Λ = Int[];
    a = zeros(T, n)
    x = [zero(T)]
    dot_p = zeros(T, p)

    for iter = 1:m
        # dot_p = Φ' * r;
        BLAS.gemv!('T', one(T), Φ, r, zero(T), dot_p)

#        λ = indmax(abs(dot_p));
        λ = indmax_abs(dot_p);
#        Λ = [Λ i];
        push!(Λ, λ)
#        sort!(Λ)
        temp = Φ[:, Λ];

        # x = argmin_{y} ( norm(v - temp*y) )
#        x = pinv(temp) * v;
#        x = temp \ v
        x, = lsqr(temp, v)       

        # a = temp*x
        BLAS.gemv!('N', 1.0, temp, x, 0.0, a)

        # r = v - a;
        copy!(r, v)
        BLAS.axpy!(-1.0, a, r)

        # output if desired
        quiet || @printf("Iter %d, residual: %f\n", iter, norm(r));
    end
    β[Λ] = x
    return β
end

end
