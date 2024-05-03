module BayesOptim
using Pkg
# Pkg.build("PyCall") 
using PyCall

include("SafetyChecks.jl")
include("Fit.jl")

export Fit

end
