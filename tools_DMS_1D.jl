
# function psf2otf(psf::AbstractArray{T,N}, outsz::NTuple{N,Int}=size(psf))
#   psfsz = size(psf)
#   if psfsz != outsz
#     pad = map(-,outsz,psfsz)
#     all(x -> x >= 0, pad) || throw(DimensionMismatch("psf too large for outsz."))
#     psf = padarray(psf,tuple(zeros(Int,N)...),pad,"value",T(0))
#   end
#   shift = map(x -> -floor(Int,x/2), psfsz)
#   fft(circshift(psf,shift))
# end


# function optD(x,kernel)
# 	return real(ifft(fft(x)*fft(kernel)))
# end
function optD(x::Vector{Float64},kernel::Matrix{Float64})
	y = circshift(x,-1)-x
	y[end] = 0
	return y
end
# function optDadj(x,kernel)
# 	return real(ifft(fft(x)*conj(fft(kernel))))[:,2]
# end
function optDadj(x::Vector{Float64},kernel::Matrix{Float64})
	y = circshift(x,1)-x
	y[1] = - x[1]
	y[end] = x[end-1]
	return y
end
function fidelity_term(x::Vector{Float64},z::Vector{Float64})
	return 0.5*sum((x-z).^2)
end 
function coupling_term(e::Vector{Float64},x::Vector{Float64},beta::Float64,kernel::Matrix{Float64})
	return  beta*sum((e.-1).^2 .*optD(x,kernel).^2)
end

function e_term(e::Vector{Float64},lambd::Float64,prox_e::String)
	if prox_e=="L1"
		return  lambd.*sum(abs.(e))
	elseif prox_e == "L0"
		return lambd.*sum(e .!= 0)
	end
end

function crit_func(x::Vector{Float64},e::Vector{Float64},z::Vector{Float64},beta::Float64,lambda::Float64,kernel::Matrix{Float64},prox_e::String)
	return fidelity_term(x,z)+coupling_term(e,x,beta,kernel)+e_term(e,lambda,prox_e)
end

function grad_C_e(x::Vector{Float64},e::Vector{Float64},beta::Float64,k::Matrix{Float64})
	return 2*beta*(e.-1.) .*optD(x,k).^2
end

function grad_C_x(x::Vector{Float64},e::Vector{Float64},beta::Float64,k::Matrix{Float64})
	return  2*beta*optDadj(optD(x,k).*(e.-1.).^2,k)
end

function cal_ck(xlocal::Vector{Float64},elocal::Vector{Float64},z::Vector{Float64},beta::Float64,k::Matrix{Float64})
	bk = xlocal
	# rhon= 1. + 1e-2
	rhokx = 1.
	it = 1
	# while abs(rhokx-rhon)/abs(rhokx)>1e-5 && it < 5000
	while it < 5000
		bk1= grad_C_x(bk,elocal,beta,k)
		bk = bk1./sqrt(sum(bk1.^2)) 
		rhokx= (grad_C_x(bk,elocal,beta,k)'*bk)/(bk'*bk)
		it+= 1
	end
	return 1.01*rhokx
end


function cal_dk(xlocal::Vector{Float64},elocal::Vector{Float64},z::Vector{Float64},beta::Float64,k::Matrix{Float64})
	bk = elocal
	# rhon= 1. + 1e-2
	rhokx = 1.
	it = 0
	# while abs(rhokx-rhon)/abs(rhokx)>1e-5 && it < 5000
	while it < 5000
		bk1= grad_C_e(xlocal,bk,beta,k)
		bk = bk1./sqrt(sum(bk1.^2))  
		rhokx= (grad_C_e(xlocal,bk,beta,k)'*bk)/(bk'*bk)
		it+= 1
	end
	return 1.01*rhokx
end



function prox_L2(x::Vector{Float64},tau::Float64,z::Vector{Float64})
	return (x+tau*z)/(1+tau)
end

# function norml1(tau::Float64)
# 	return NormL1(tau)
# end

# function prox_L1(x::Vector{Float64},tau::Float64)
# 	y,fy= prox(norml1(tau),x,1)
# 	return y
# end

function prox_L1(x::Vector{Float64},tau)
	return sign.(x).*max.(abs.(x).-tau,0)
end


function normalize(x::Vector{Float64})
	return (x.-minimum(x))/(maximum(x)-minimum(x))
end

function  prox_L0(x::Vector{Float64},tau)
	return x .* (abs.(x) .> sqrt.(2. .* tau))
end