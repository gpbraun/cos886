module Cos886

using Random
using LinearAlgebra
using JuMP

import MathOptInterface
const MOI = MathOptInterface

import Gurobi
import MosekTools

include("core/instance.jl")
include("core/utils.jl")
include("core/eval.jl")

include("models/subproblem.jl")
include("models/master.jl")

include("algorithms/ecp.jl")
include("algorithms/oa.jl")
include("algorithms/bb.jl")

function solve(inst::Instance; method::Symbol = :ecp, params = nothing)
    if method === :ecp
        p = params === nothing ? ECPParams() : params::ECPParams
        return solve_ecp(inst, p)
    elseif method === :oa
        p = params === nothing ? OAParams() : params::OAParams
        return solve_oa(inst, p)
    elseif method === :bb
        p = params === nothing ? BBParams() : params::BBParams
        return solve_bb(inst, p)
    else
        error("methodo inválido: $method (use :ecp, :oa ou :bb)")
    end
end

export Instance,
    make_poly1d_instance,
    make_gaussian_instance,
    Results,
    ECPParams,
    OAParams,
    BBParams,
    solve

end # module
