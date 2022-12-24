function PALM_1D(z::Vector{Float64},beta::Float64,lambda::Float64,maxiter::Int64,order::Int64,prox_e::String,kernel::Matrix{Float64})
	x, e = z, zeros(Float64,size(z))
	crit_table = zeros(Float64,maxiter)
	it = 1
	err= 10.
	crit_table[1] = crit_func(x,e,z,beta,lambda,kernel,prox_e)
	while (it<maxiter) &&(err>1e-4) 
		ck = cal_ck(x,e,z,beta,kernel)
		dk = cal_dk(x,e,z,beta,kernel)

		x = prox_L2(x .- (1/ck).*grad_C_x(x,e,beta,kernel),1/ck,z)
		
		if prox_e =="L1"
			e = prox_L1(e .- (1/dk).*grad_C_e(x,e,beta,kernel),lambda/dk)
		elseif prox_e == "L0"
			e = prox_L0(e .- (1/dk).*grad_C_e(x,e,beta,kernel),lambda/dk)			
		end 
		crit_table[it+1] = crit_func(x,e,z,beta,lambda,kernel,prox_e)
		err = abs(crit_table[it+1]-crit_table[it])/abs(crit_table[it+1]+1e-10)
		# print(it,"---",crit_table[it+1],"\n")
		it+=1
	end
	return x,e, crit_table[1:it-1]
end 

	