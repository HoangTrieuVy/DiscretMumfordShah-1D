module DMS_1D

import Pkg
Pkg.activate(mktempdir());
# Pkg.add(["Images", "ImageIO", "ImageMagick"])
Pkg.add(["NPZ"])
Pkg.add("Plots")
Pkg.add("FFTW")
# Pkg.add(["ProximalOperators"])
Pkg.add("Distributions")
# Pkg.add("DSP")
# Pkg.add("ImageFiltering")
# using DSP
# using ProximalOperators
# using ImageFiltering
# using FFTW
# using Images

include("tools_DMS_1D.jl")
include("PALM_DMS_1D.jl")
include("SLPAM_DMS_1D.jl")

end