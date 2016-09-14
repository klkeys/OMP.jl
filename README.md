# OMP.jl
A Julia module to perform [orthogonal matching pursuit](https://en.wikipedia.org/wiki/Matching_pursuit).

## Installation

From the Julia REPL, type

    Pkg.clone("https://github.com/klkeys/OMP.jl")

## Usage

OMP.jl exports two functions, `omp` and `cv_omp`.
To construct a `k`-sparse approximation to a response `y` from a design matrix `x`, the call is

    output = omp(x, y, k)

The result `output` is a `SparseMatrixCSC` with one column for each sparsity level `1, 2, ..., k`.

To compute the sparsity level of the best predictive model, the call is

    output = cv_omp(x, y)

By default `cv_omp` tests sparsity levels `1, 2, ..., 20` in a `q = 5`-fold crossvalidation scheme.
Use the keyword argument `k = MODEL_SIZE` to change the maximum sparsity level to `MODEL_SIZE`. 
The keyword argument `q = NUM_FOLDS` sets the number of folds to `NUM_FOLDS`.
Also by default, `cv_omp` spawns crossvalidation folds on all available processors.
The processes involved in the crossvalidation are controled by the keyword argument `pids`.
For example, `pids = [1,2,3]` will only spawn folds on process IDs `1`, `2`, and `3`.

## References

[Signal Recovery From Random Measurements Via Orthogonal Matching Pursuit](http://users.cms.caltech.edu/~jtropp/papers/TG07-Signal-Recovery.pdf), Joel A. Tropp and Anna C. Gilbert, _IEEE Transactions on Information Theory_ **53**:12, December 2007.
