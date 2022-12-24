function SLPAM_1D(z::Vector{Float64}, beta::Float64, lambda::Float64, maxiter::Int64, order::Int64, prox_e::String, kernel::Matrix{Float64}, dk::Float64)
    x, e = z, zeros(Float64, size(z))
    crit_table = zeros(Float64, maxiter)
    it = 1
    err = 10.0
    crit_table[1] = crit_func(x, e, z, beta, lambda, kernel, prox_e)
    while (it < maxiter) && (err > 1e-6)
        ck = cal_ck(x, e, z, beta, kernel)
        x = prox_L2(x .- (1 / ck) .* grad_C_x(x, e, beta, kernel), 1 / ck, z)

        if prox_e == "L1"
            over = beta .* optD(x, kernel) .^ 2 .+ e .* dk / 2.0
            lower = beta .* optD(x, kernel) .^ 2 .+ dk / 2.0
            e = prox_L1(over ./ lower, lambda ./ (2 .* lower))
        end
        crit_table[it+1] = crit_func(x, e, z, beta, lambda, kernel, prox_e)
        err = abs(crit_table[it+1] - crit_table[it]) / abs(crit_table[it+1] + 1e-10)
        # print(it,"---",crit_table[it+1],"\n")
        it += 1
    end
    return x, e, crit_table[1:it-1]
end

