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
