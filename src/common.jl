# subroutine to compute a default number of folds
@inline cv_get_num_folds(nmin::Int, nmax::Int) = max(nmin, min(Sys.CPU_CORES::Int, nmax))

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


# verbose printing of cv results
function print_cv_results{T <: Float}(io::IO, errors::Vector{T}, path::DenseVector{Int}, k::Int)
    println(io, "\n\nCrossvalidation Results:")
    println(io, "k\tMSE")
    for i = 1:length(errors)
        println(io, path[i], "\t", errors[i])
    end
    println(io, "\nThe lowest MSE is achieved at k = ", k)
end

# default IO for print_cv_results is STDOUT
print_cv_results{T <: Float}(errors::Vector{T}, path::DenseVector{Int}, k::Int) = print_cv_results(STDOUT, errors, path, k)
