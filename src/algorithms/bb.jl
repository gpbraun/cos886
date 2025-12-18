# algorithms/bb.jl

"""
Política de cortes fracionários (UserCuts).
"""
Base.@kwdef struct CutPolicy
    min_violation_abs::Float64 = 1e-2
    min_violation_rel::Float64 = 1e-3
    min_efficacy::Float64 = 1e-3
    frac_max::Float64 = 0.30
    max_cuts_per_node::Int = 1
end

"""
Parâmetros do BB.
"""
Base.@kwdef struct BBParams
    silent::Bool = false
    time_limit::Float64 = 60.0

    nL0::Int = 0
    max_gap::Float64 = 1e-4
    master_optimizer = Gurobi.Optimizer
    subproblem_optimizer = MosekTools.Optimizer

    # Parâmetros para cortes fracionários
    user_cuts::Bool = false
    cut_policy::CutPolicy = CutPolicy()
end

"""
Estado do BB para callbacks.
"""
Base.@kwdef mutable struct BBState
    UB::Float64 = Inf
    n_lazy::Int = 0
    n_user::Int = 0
    hs_update::Bool = false

    _uc_node::Int = -1
    _uc_in_node::Int = 0
end

"""
Resolve com o método: Branch-and-Bound baseado em PL/PNL (via callbacks).
"""
function solve_bb(inst::Instance, params::BBParams = BBParams())::Results
    t0 = time()
    n = inst.n
    cp = params.cut_policy

    # ESTADO
    mp = Master(
        inst,
        optimizer = params.master_optimizer,
        silent = params.silent,
        relax_y = false,
    )
    ew = EvalWorker(inst)
    sp = Subproblem(inst; optimizer = params.subproblem_optimizer, silent = true)

    x_val = zeros(Float64, n)
    y_val = zeros(Int, n)
    g_val = zeros(Float64, n)
    x_best = zeros(Float64, n)
    y_best = zeros(Int, n)

    # armazena a solução heurística
    hs_vars = Vector{MOI.VariableIndex}(undef, 1 + 2n)
    hs_vars[1] = JuMP.index(mp.t)
    @inbounds for i in 1:n
        hs_vars[1+i] = JuMP.index(mp.x[i])
        hs_vars[1+n+i] = JuMP.index(mp.y[i])
    end
    hs_vals = zeros(Float64, length(hs_vars))

    # PARÂMETROS DO MESTRE
    backend = JuMP.backend(mp.model)
    JuMP.set_time_limit_sec(mp.model, time_remaining(t0, params.time_limit))
    JuMP.set_optimizer_attribute(mp.model, "LazyConstraints", 1)
    JuMP.set_optimizer_attribute(mp.model, "PreCrush", 1)
    JuMP.set_optimizer_attribute(mp.model, "Threads", 1)
    MOI.set(JuMP.backend(mp.model), MOI.RelativeGapTolerance(), params.max_gap)

    # L0
    UB0 = add_L0!(inst, mp, ew, sp, x_val, y_val, g_val, x_best, y_best, params.nL0)

    state = BBState(UB = UB0)

    # ============================================================
    # CALLBACK: LAZY
    # ============================================================
    function lazy_cb(cb_data)
        state.n_lazy += 1
        t_val = JuMP.callback_value(cb_data, mp.t)

        eps = 1e-6

        # tempo global para o subproblema
        tr = time_remaining(t0, params.time_limit)
        tr <= 0 && return
        JuMP.set_time_limit_sec(sp.model, tr)

        # Recupera os valores
        @inbounds for i in 1:n
            x_val[i] = JuMP.callback_value(cb_data, mp.x[i])
            y_val[i] = round01(JuMP.callback_value(cb_data, mp.y[i]))
        end

        # Corte de viabilidade no próprio incumbente: lineariza em x_val
        f_val = f_and_grad!(ew, x_val, inst, g_val)
        if t_val + eps < f_val
            alpha = f_val - dot(g_val, x_val)
            con = @build_constraint(mp.t >= dot(g_val, mp.x) + alpha)
            MOI.submit(mp.model, MOI.LazyConstraint(cb_data), con)
        end

        # UB verdadeiro via P_y (x_val <- x_star)
        _ = solve!(sp, y_val, x_val)

        f_star = f_and_grad!(ew, x_val, inst, g_val)

        if f_star < state.UB
            state.UB = f_star
            copyto!(x_best, x_val)
            copyto!(y_best, y_val)

            # monta solução heurística pendente
            hs_vals[1] = f_star
            @inbounds for i in 1:n
                hs_vals[1+i] = x_val[i]
                hs_vals[1+n+i] = Float64(y_val[i])
            end
            state.hs_update = true
        end

        # Corte extra só se separar o incumbente atual.
        alpha_star = f_star - dot(g_val, x_val)

        rhs_val_star = alpha_star
        @inbounds for i in 1:n
            rhs_val_star += g_val[i] * JuMP.callback_value(cb_data, mp.x[i])
        end

        if t_val + eps < rhs_val_star
            con = @build_constraint(mp.t >= dot(g_val, mp.x) + alpha_star)
            MOI.submit(mp.model, MOI.LazyConstraint(cb_data), con)
        end

        return
    end

    # ============================================================
    # CALLBACK: USERCUTS
    # ============================================================
    function usercut_cb(cb_data)
        state.n_user += 1

        # 1) orçamento por nó
        node_id = try
            MOI.get(backend, MOI.NodeCount())
        catch
            -1
        end
        if node_id != state._uc_node
            state._uc_node = node_id
            state._uc_in_node = 0
        end
        (state._uc_in_node < cp.max_cuts_per_node) || return

        # 2) filtro por fracionariedade (em y)
        frac = 0.0
        @inbounds for i in 1:n
            yi = JuMP.callback_value(cb_data, mp.y[i])
            frac += min(yi, 1.0 - yi)
        end
        frac /= n
        (frac <= cp.frac_max) || return

        # 3) valores de x e violação
        t_val = JuMP.callback_value(cb_data, mp.t)
        @inbounds for i in 1:n
            x_val[i] = JuMP.callback_value(cb_data, mp.x[i])
        end

        f_val = f_and_grad!(ew, x_val, inst, g_val)
        v = f_val - t_val
        (v >= cp.min_violation_abs) || return

        v_rel = v / max(1.0, abs(f_val))
        (v_rel >= cp.min_violation_rel) || return

        # 4) eficácia (normalizada)
        denom = sqrt(1.0 + dot(g_val, g_val))
        eff = v / denom
        (eff >= cp.min_efficacy) || return

        # 5) adiciona o corte
        alpha = f_val - dot(g_val, x_val)
        con = @build_constraint(mp.t >= dot(g_val, mp.x) + alpha)
        MOI.submit(mp.model, MOI.UserCut(cb_data), con)

        state._uc_in_node += 1

        return
    end

    # ============================================================
    # CALLBACK: HEURÍSTICAS
    # ============================================================
    function heuristic_cb(cb_data)
        state.hs_update || return

        MOI.submit(backend, MOI.HeuristicSolution(cb_data), hs_vars, hs_vals)
        state.hs_update = false

        return
    end

    # ============================================================
    # SOLVE
    # ============================================================
    MOI.set(backend, MOI.LazyConstraintCallback(), lazy_cb)
    MOI.set(backend, MOI.HeuristicCallback(), heuristic_cb)
    params.user_cuts && MOI.set(backend, MOI.UserCutCallback(), usercut_cb)

    optimize!(mp.model)

    mp_status = termination_status(mp.model)
    if !(mp_status in (MOI.OPTIMAL, MOI.TIME_LIMIT, MOI.INTERRUPTED))
        error("Mestre falhou: $(mp_status)")
    end

    LB = objective_bound(mp.model)
    UB = state.UB

    return Results(
        x = copy(x_best),
        y = copy(y_best),
        lb = LB,
        ub = UB,
        gap = relative_gap(LB, UB),
        time = time_elapsed(t0),
        nodes = MOI.get(backend, MOI.NodeCount()),
    )
end
