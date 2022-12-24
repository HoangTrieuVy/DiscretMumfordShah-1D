include("DMS_1D.jl")
using Plots: plot, plot!, scatter, scatter!
using Distributions: Normal
import .DMS_1D
using NPZ: npzread

# signal_block =  npzread("Blocks.npy");
piece_block = npzread("Piece.npy");

n_s = DMS_1D.normalize(piece_block) + 0.05 * rand(Normal(0, 1), size(piece_block)[1]);

noisy_signal = n_s
maxiter = 500
order = 1
kernel = [-1.0 1.0]
dk = 1e-3
# d_s_p, e_s_p, crit_s_p = @time DMS_1D.PALM_1D(noisy_signal, 50.0, 1e-1, maxiter, order, "L1", kernel);
d_s, e_s, crit_s = @time DMS_1D.SLPAM_1D(noisy_signal, 50.0, 1e-1, maxiter, order, "L1", kernel, dk);

plot(piece_block, label="original")
plot!(d_s, label="denoised SLPAM")
# plot!(d_s_p, label="denoised PALM")