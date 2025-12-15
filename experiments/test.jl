# test_quick.jl
# Script mínimo (hardcoded) para rodar uma instância pequena rapidamente.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Cos886
using Gurobi

const GRB_ENV = Gurobi.Env()
const GRB_OPT = () -> Gurobi.Optimizer(GRB_ENV)

# Instância pequena e rápida
inst = make_gaussian_instance(5, 40, 10; seed = 123, normalize = true, delta = 1e-5)

time_limit = 200.0
max_gap = 1e-3
silent = true

println("\n=== ECP ===")
res_ecp = Cos886.solve(
    inst;
    method = :ecp,
    params = ECPParams(
        time_limit = time_limit,
        silent = silent,
        master_optimizer = GRB_OPT,
        max_gap = max_gap,
    ),
)
println(
    "lb=$(res_ecp.lb)  ub=$(res_ecp.ub)  gap=$(res_ecp.gap)  iters=$(res_ecp.iters)  time=$(res_ecp.time)",
)

println("\n=== OA ===")
res_oa = Cos886.solve(
    inst;
    method = :oa,
    params = OAParams(
        time_limit = time_limit,
        silent = silent,
        master_optimizer = GRB_OPT,
        max_gap = max_gap,
    ),
)
println(
    "lb=$(res_oa.lb)  ub=$(res_oa.ub)  gap=$(res_oa.gap)  iters=$(res_oa.iters)  time=$(res_oa.time)",
)

println("\n=== BB ===")
res_bb = Cos886.solve(
    inst;
    method = :bb,
    params = BBParams(
        time_limit = time_limit,
        master_optimizer = GRB_OPT,
        silent = silent,
        max_gap = max_gap,
    ),
)
println(
    "lb=$(res_bb.lb)  ub=$(res_bb.ub)  gap=$(res_bb.gap)  nodes=$(res_bb.nodes)  time=$(res_bb.time)",
)
