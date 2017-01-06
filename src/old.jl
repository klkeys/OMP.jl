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
