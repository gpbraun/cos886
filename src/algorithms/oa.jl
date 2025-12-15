# algorithms/oa.jl

"""
Parâmetros do OA.
"""
Base.@kwdef struct OAParams
    silent::Bool = false
    time_limit::Float64 = 60.0

    nL0::Int = 10
    max_gap::Float64 = 1e-4
    master_optimizer = Gurobi.Optimizer
    subproblem_optimizer = MosekTools.Optimizer
end

"""
Resolve com o método: Outer Approximation (OA).
"""
function solve_oa(inst::Instance, params::OAParams = OAParams())::Results
    t0 = time()
    n = inst.n

    # ESTADO
    mp = Master(inst, optimizer = params.master_optimizer, silent = true)
    ew = EvalWorker(inst)
    sp = Subproblem(inst, optimizer = params.subproblem_optimizer, silent = true)

    x_val = zeros(Float64, n)
    y_val = zeros(Int, n)
    g_val = zeros(Float64, n)
    x_best = zeros(Float64, n)
    y_best = zeros(Int, n)

    LB = -Inf
    UB = Inf

    # PARÂMETROS DO MESTRE
    MOI.set(JuMP.backend(mp.model), MOI.RelativeGapTolerance(), 0.1 * params.max_gap)

    # L0
    UB = add_L0!(inst, mp, ew, sp, x_val, y_val, g_val, x_best, y_best, params.nL0)

    # ============================================================
    # LOOP PRINCIPAL DO OA
    # ============================================================
    iters = 0
    while true
        iters += 1

        time_rem = time_remaining(t0, params.time_limit)
        time_rem <= 0 && break

        JuMP.set_time_limit_sec(mp.model, time_rem)
        optimize!(mp.model)

        mp_status = termination_status(mp.model)
        if !(mp_status in (MOI.OPTIMAL, MOI.TIME_LIMIT))
            error("Mestre falhou: $(mp_status)")
        end
        if mp_status == MOI.TIME_LIMIT || primal_status(mp.model) != MOI.FEASIBLE_POINT
            break
        end

        LB = max(LB, objective_bound(mp.model))

        get_value!(y_val, mp.y)

        # resolve P_y e lineariza em x*
        time_rem = time_remaining(t0, params.time_limit)
        time_rem <= 0 && break

        JuMP.set_time_limit_sec(sp.model, time_rem)
        solve!(sp, y_val, x_val)

        sp_status = termination_status(sp.model)
        if sp_status == MOI.TIME_LIMIT || primal_status(sp.model) != MOI.FEASIBLE_POINT
            break
        end

        f_val = f_and_grad!(ew, x_val, inst, g_val)

        if f_val < UB
            UB = f_val
            copyto!(x_best, x_val)
            copyto!(y_best, y_val)
        end

        gap = relative_gap(LB, UB)

        if !params.silent
            println("OA  iter=$(iters)  LB=$(LB)  UB=$(UB)  gap=$(gap)")
        end

        (gap <= params.max_gap) && break

        # adiciona corte e warm-start factível ao novo corte (em x_val exige t >= f_val)
        add_cut!(mp, x_val, f_val, g_val)

        set_start_value(mp.t, f_val)
        set_start_value(mp.x, x_val)
        set_start_value(mp.y, y_val)
    end

    return Results(
        x = copy(x_best),
        y = copy(y_best),
        lb = LB,
        ub = UB,
        gap = relative_gap(LB, UB),
        time = time_elapsed(t0),
        iters = iters,
    )
end
