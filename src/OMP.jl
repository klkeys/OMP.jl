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


include("common.jl")
include("cv.jl")
include("gwas.jl")
include("mp.jl")

end
