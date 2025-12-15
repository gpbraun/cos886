# models/master.jl

struct Master
    model::Model
    t::VariableRef
    x::Vector{VariableRef}
    y::Vector{VariableRef}
end

"""
Retorna: problema mestre.
"""
function Master(
    inst::Instance;
    optimizer = Gurobi.Optimizer,
    silent::Bool = true,
    relax_y::Bool = false,
)
    n = inst.n
    m = Model(optimizer)
    silent && set_silent(m)

    @variable(m, t)
    @variable(m, 0.0 <= x[1:n] <= 1.0)
    if relax_y
        @variable(m, 0.0 <= y[1:n] <= 1.0)
    else
        @variable(m, y[1:n], Bin)
    end

    @constraint(m, x .<= y)
    @constraint(m, sum(y) == inst.k)
    @constraint(m, sum(x) == 1.0)

    @objective(m, Min, t)

    return Master(m, t, Vector{JuMP.VariableRef}(x), Vector{JuMP.VariableRef}(y))
end

"""
Adiciona: corte de suporte.
"""
function add_cut!(
    pm::Master,
    x_bar::AbstractVector{<:Real},
    f_bar::Real,
    g_bar::AbstractVector{<:Real},
)
    @assert length(x_bar) == length(pm.x)
    @assert length(g_bar) == length(pm.x)

    alpha = Float64(f_bar) - dot(g_bar, x_bar)
    @constraint(pm.model, pm.t >= dot(g_bar, pm.x) + alpha)
    return nothing
end

"""
Calcula: f(x) e ∇f(x) via f_and_grad!
Adiciona: corte de suporte no mestre.
Atualiza: g
Retorna: f
"""
function eval_and_cut!(
    pm::Master,
    ws::EvalWorker,
    inst::Instance,
    x::Vector{<:Real},
    g::Vector{<:Real},
)::Float64
    f = f_and_grad!(ws, x, inst, g)
    add_cut!(pm, x, f, g)
    return f
end

# ============================================================
# BUILD POINTS (ECP vs OA/BB)
# ============================================================

"""
Escolhe y0: QR pivotado em V'.
Retorna idx (tamanho k), ordenado.
"""
function initial_idx_qr(inst::Instance)::Vector{Int}
    perm = Vector{Int}(qr(transpose(inst.V), ColumnNorm()).p)
    idx = perm[1:inst.k]
    sort!(idx)
    return idx
end

# ============================================================
# BUILD POINTS (ECP vs OA/BB) via `sp`
# ============================================================

"""
Monta ponto a partir do suporte idx: x uniforme 1/k no suporte e y binário no suporte.
"""
function build_L0_point_uniform_x!(
    inst::Instance,
    idx::Vector{Int},
    x_val::Vector{<:Real},
    y_val::Vector{<:Real},
)::Bool
    fill!(x_val, 0)
    fill!(y_val, 0)
    w = 1.0 / inst.k
    @inbounds for j in idx
        x_val[j] = w
        y_val[j] = 1
    end
    return true
end

"""
Atualiza ponto por swap (j_out -> 0, j_in -> 1) sem zerar tudo.
"""
function build_L0_point_uniform_x!(
    inst::Instance,
    x_val::Vector{<:Real},
    y_val::Vector{<:Real},
    j_out::Int,
    j_in::Int,
)::Bool
    w = 1.0 / inst.k
    @inbounds begin
        x_val[j_out] = 0
        y_val[j_out] = 0
        x_val[j_in] = w
        y_val[j_in] = 1
    end
    return true
end

"""
Monta y binário no suporte idx e obtém x resolvendo P_y.
"""
function build_L0_point_via_subproblem!(
    sp::Subproblem,
    idx::Vector{Int},
    x_val::Vector{<:Real},
    y_val::Vector{<:Real},
)::Bool
    fill!(y_val, 0)
    @inbounds for j in idx
        y_val[j] = 1
    end
    try
        solve!(sp, y_val, x_val)
        return true
    catch
        return false
    end
end

"""
Atualiza y por swap e re-resolve P_y para obter x. Reverte y se o subproblema falhar.
"""
function build_L0_point_via_subproblem!(
    sp::Subproblem,
    x_val::Vector{<:Real},
    y_val::Vector{<:Real},
    j_out::Int,
    j_in::Int,
)::Bool
    @inbounds begin
        y_val[j_out] = 0
        y_val[j_in] = 1
    end
    try
        solve!(sp, y_val, x_val)
        return true
    catch
        @inbounds begin
            y_val[j_out] = 1
            y_val[j_in] = 0
        end
        return false
    end
end

# ============================================================
# L0
# ============================================================

"""
Constrói cortes iniciais (L0) no mestre.
"""
function add_L0!(
    inst::Instance,
    pm::Master,
    ws::EvalWorker,
    sp::Union{Nothing,Subproblem},
    x_val::Vector{<:Real},
    y_val::Vector{<:Real},
    g_val::Vector{<:Real},
    x_best::Vector{<:Real},
    y_best::Vector{<:Real},
    nL0::Int,
)::Float64
    n, k = inst.n, inst.k
    UB = Inf

    # hash do conjunto (idx ordenado) sem alocar
    function hash_idx(idx::Vector{Int})::UInt
        h = UInt(0)
        @inbounds for s in 1:k
            h = hash(idx[s], h)
        end
        return h
    end

    # substitui old->new mantendo ordenação via "bubble" (O(k))
    function replace_sorted!(idx::Vector{Int}, old::Int, new::Int)::Bool
        pos = 0
        @inbounds for t in 1:k
            if idx[t] == old
                pos = t
                break
            end
        end
        pos == 0 && return false
        idx[pos] = new
        @inbounds begin
            while pos > 1 && idx[pos] < idx[pos-1]
                idx[pos], idx[pos-1] = idx[pos-1], idx[pos]
                pos -= 1
            end
            while pos < k && idx[pos] > idx[pos+1]
                idx[pos], idx[pos+1] = idx[pos+1], idx[pos]
                pos += 1
            end
        end
        return true
    end

    # (a) init por QR
    idx_curr = initial_idx_qr(inst)
    @assert length(idx_curr) == k

    seen = Set{UInt}()
    push!(seen, hash_idx(idx_curr))

    ok =
        sp === nothing ? build_L0_point_uniform_x!(inst, idx_curr, x_val, y_val) :
        build_L0_point_via_subproblem!(sp, idx_curr, x_val, y_val)

    ok || return UB

    UB = eval_and_cut!(pm, ws, inst, x_val, g_val)
    copyto!(x_best, x_val)
    copyto!(y_best, y_val)

    # casos degenerados
    (nL0 <= 0 || k == 0 || k == n) && begin
        set_start_value(pm.t, UB)
        set_start_value(pm.x, x_best)
        set_start_value(pm.y, y_best)
        return UB
    end

    # suporte atual
    inS = falses(n)
    @inbounds for j in idx_curr
        inS[j] = true
    end

    # buffers candidatos
    m_in = min(10, n - k)
    m_out = min(10, k)

    cand_in = Vector{Int}(undef, m_in)
    cand_out = Vector{Int}(undef, m_out)
    vin = Vector{Float64}(undef, m_in)
    vout = Vector{Float64}(undef, m_out)

    idx_new = similar(idx_curr)

    added = 0
    tries = 0
    max_tries = max(100, 50 * nL0)

    while added < nL0 && tries < max_tries
        tries += 1

        # candidatos guiados por g_val (do último ponto avaliado)
        fill!(vin, Inf)
        fill!(cand_in, 0)
        fill!(vout, -Inf)
        fill!(cand_out, 0)

        @inbounds for j in 1:n
            gj = Float64(g_val[j])
            if inS[j]
                if gj > vout[end]
                    t = m_out
                    while t > 1 && gj > vout[t-1]
                        vout[t] = vout[t-1]
                        cand_out[t] = cand_out[t-1]
                        t -= 1
                    end
                    vout[t] = gj
                    cand_out[t] = j
                end
            else
                if gj < vin[end]
                    t = m_in
                    while t > 1 && gj < vin[t-1]
                        vin[t] = vin[t-1]
                        cand_in[t] = cand_in[t-1]
                        t -= 1
                    end
                    vin[t] = gj
                    cand_in[t] = j
                end
            end
        end

        nin = 0
        @inbounds for t in 1:m_in
            (cand_in[t] == 0) && break
            nin += 1
        end
        nout = 0
        @inbounds for t in 1:m_out
            (cand_out[t] == 0) && break
            nout += 1
        end
        (nin == 0 || nout == 0) && break

        # tenta achar um swap novo
        found = false
        j_in = 0
        j_out = 0

        @inbounds for a in 1:nin
            ja = cand_in[a]
            for b in 1:nout
                jb = cand_out[b]
                (!inS[ja] && inS[jb]) || continue

                copyto!(idx_new, idx_curr)
                replace_sorted!(idx_new, jb, ja) || continue

                h = hash_idx(idx_new)
                (h in seen) && continue

                push!(seen, h)
                found = true
                j_in = ja
                j_out = jb
                break
            end
            found && break
        end

        found || break

        # aplica swap em idx_curr + suporte
        replace_sorted!(idx_curr, j_out, j_in) || break
        inS[j_out] = false
        inS[j_in] = true

        # monta ponto de forma incremental
        ok = if sp === nothing
            build_L0_point_uniform_x!(inst, x_val, y_val, j_out, j_in)
        else
            build_L0_point_via_subproblem!(sp, x_val, y_val, j_out, j_in)
        end

        if !ok
            # reverte swap e tenta próxima iteração
            inS[j_out] = true
            inS[j_in] = false
            replace_sorted!(idx_curr, j_in, j_out)
            sp === nothing && build_L0_point_uniform_x!(inst, x_val, y_val, j_in, j_out)
            continue
        end

        f = eval_and_cut!(pm, ws, inst, x_val, g_val)

        if f < UB
            UB = f
            copyto!(x_best, x_val)
            copyto!(y_best, y_val)
        end

        added += 1
    end

    # warm-start após L0
    set_start_value(pm.t, UB)
    set_start_value(pm.x, x_best)
    set_start_value(pm.y, y_best)

    return UB
end
