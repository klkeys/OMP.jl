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
                                one_fold(x, y, k, folds, current_fold, quiet=quiet)
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
    kbest = path[indmin(mses)] :: Int 

    # print results
    !quiet && print_cv_results(mses, path, kbest)

    # refit best model
    #b, bidx = refit_omp(x, y, kbest, quiet=quiet)
    betas = omp(x, y, kbest, quiet=quiet)
    bidx  = find(betas[:,end])
    b     = full(betas[bidx,end])

    return OMPCrossvalidationResults{T}(mses, path, b, bidx, kbest)
end

