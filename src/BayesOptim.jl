module BayesOptim
using Pkg
# Pkg.build("PyCall") 
using PyCall

DIR = @__DIR__
@pyinclude(DIR*"/Bopt.py")

include("SafetyChecks.jl")
include("Fit.jl")

export Fit

end
