function SafetyChecks(bounds)
    if length(bounds) > 10
        @warn "Using Bayesian Optimization for more than 10 variables can be slow. "
    end 

    for b in bounds
        @assert b[2][1] < b[2][2] "Set ordered bounds for the parameters."
    end 
end
