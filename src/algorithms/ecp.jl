# algorithms/ecp.jl

"""
Parâmetros do ECP.
"""
Base.@kwdef struct ECPParams
    silent::Bool = false
    time_limit::Float64 = 60.0

    nL0::Int = 20
    max_gap::Float64 = 1e-4
    master_optimizer = Gurobi.Optimizer
end

"""
Resolve com o método: Extended Cutting Plane (ECP).
"""
function solve_ecp(inst::Instance, params::ECPParams = ECPParams())::Results
    t0 = time()
    n, k = inst.n, inst.k

    # ESTADO
    mp = Master(inst, optimizer = params.master_optimizer, silent = true)
    ew = EvalWorker(inst)

    x_val = zeros(Float64, n)
    y_val = zeros(Float64, n)
    g_val = zeros(Float64, n)
    x_best = zeros(Float64, n)
    y_best = zeros(Int, n)

    LB = -Inf
    UB = Inf

    # PARÂMETROS DO MESTRE
    MOI.set(JuMP.backend(mp.model), MOI.RelativeGapTolerance(), 0.1 * params.max_gap)

    # L0
    UB = add_L0!(inst, mp, ew, nothing, x_val, y_val, g_val, x_best, y_best, params.nL0)

    # ============================================================
    # LOOP PRINCIPAL DO ECP
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

        get_value!(x_val, mp.x)
        get_value!(y_val, mp.y)
        f_val = f_and_grad!(ew, x_val, inst, g_val)
        y_is_int = is_integral(y_val)

        if y_is_int && f_val < UB
            UB = f_val
            copyto!(x_best, x_val)
            copyto!(y_best, round01(y_val))
        end

        gap = relative_gap(LB, UB)

        if !params.silent
            println("ECP iter=$(iters)  LB=$(LB)  UB=$(UB)  gap=$(gap)")
        end
        (gap <= params.max_gap) && break

        # separa corte e faz warm-start factível ao novo corte em x_val: t := f_val
        add_cut!(mp, x_val, f_val, g_val)

        set_start_value(mp.t, f_val)
        set_start_value(mp.x, x_val)
        y_is_int && set_start_value(mp.y, y_val)
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
