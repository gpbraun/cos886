# test_quick.jl
# Roda várias instâncias e salva resultados em experiments.txt

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Cos886
using Gurobi
using Printf
using Dates

const GRB_ENV = Gurobi.Env()
const GRB_OPT = () -> Gurobi.Optimizer(GRB_ENV)

# -----------------------
# Config geral do teste
# -----------------------
const TIME_LIMIT = 200.0
const MAX_GAP = 1e-3
const SILENT = true

const NORMALIZE = true
const DELTA = 1e-5

const SPECS = [
    (n = 20, p = 5, k = 5),
    # (n = 50, p = 5, k = 10),
    # (n = 50, p = 5, k = 15),
    # (n = 50, p = 10, k = 5),
    # (n = 50, p = 10, k = 10),
    # (n = 100, p = 5, k = 22),
    # (n = 100, p = 5, k = 25),
    # (n = 100, p = 5, k = 28),
    # (n = 100, p = 10, k = 30),
    # (n = 100, p = 10, k = 35),
]

# -----------------------
# Helpers
# -----------------------
const SEP = repeat("=", 62)

function fmt_val(x)
    return @sprintf("%.4g", x)
end

# -----------------------
# Execução
# -----------------------
out_path = joinpath(@__DIR__, "experiments.txt")

open(out_path, "w") do io
    println(io, "# COS886")
    println(io, "# time_limit=", TIME_LIMIT, "  max_gap=", MAX_GAP, "  silent=", SILENT)
    println(io)

    for (idx, spec) in enumerate(SPECS)
        n, p, k = spec.n, spec.p, spec.k
        seed = 123 + idx

        inst = make_gaussian_instance(
            p,
            n,
            k;
            seed = seed,
            normalize = NORMALIZE,
            delta = DELTA,
        )

        println(io, SEP)
        @printf(io, "#%d. n = %d  p = %d  k = %d  seed = %d\n\n", idx, n, p, k, seed)

        # ---------- ECP ----------
        res_ecp = Cos886.solve(
            inst;
            method = :ecp,
            params = ECPParams(
                time_limit = TIME_LIMIT,
                silent = SILENT,
                master_optimizer = GRB_OPT,
                max_gap = MAX_GAP,
            ),
        )
        @printf(
            io,
            "[ ECP ] lb = %-10s  ub = %-10s  iters = %-6d  time = %6.2f\n",
            fmt_val(res_ecp.lb),
            fmt_val(res_ecp.ub),
            res_ecp.iters,
            res_ecp.time
        )

        # ---------- OA ----------
        res_oa = Cos886.solve(
            inst;
            method = :oa,
            params = OAParams(
                time_limit = TIME_LIMIT,
                silent = SILENT,
                master_optimizer = GRB_OPT,
                max_gap = MAX_GAP,
            ),
        )
        @printf(
            io,
            "[ OA  ] lb = %-10s  ub = %-10s  iters = %-6d  time = %6.2f\n",
            fmt_val(res_oa.lb),
            fmt_val(res_oa.ub),
            res_oa.iters,
            res_oa.time
        )

        # ---------- BB1 (sem usercuts) ----------
        res_bb1 = Cos886.solve(
            inst;
            method = :bb,
            params = BBParams(
                time_limit = TIME_LIMIT,
                master_optimizer = GRB_OPT,
                silent = SILENT,
                max_gap = MAX_GAP,
                user_cuts = false,
            ),
        )
        @printf(
            io,
            "[ BB1 ] lb = %-10s  ub = %-10s  nodes = %-6d  time = %6.2f\n",
            fmt_val(res_bb1.lb),
            fmt_val(res_bb1.ub),
            res_bb1.nodes,
            res_bb1.time
        )

        # ---------- BB2 (com usercuts) ----------
        res_bb2 = Cos886.solve(
            inst;
            method = :bb,
            params = BBParams(
                time_limit = TIME_LIMIT,
                master_optimizer = GRB_OPT,
                silent = SILENT,
                max_gap = MAX_GAP,
                user_cuts = true,
            ),
        )
        @printf(
            io,
            "[ BB2 ] lb = %-10s  ub = %-10s  nodes = %-6d  time = %6.2f\n",
            fmt_val(res_bb2.lb),
            fmt_val(res_bb2.ub),
            res_bb2.nodes,
            res_bb2.time
        )

        println(io)
        flush(io)
        GC.gc()
    end

    return println(io, SEP)
end

println("OK: resultados em $(out_path)")
