# models/subproblem.jl

"Vetoriza triângulo superior (coluna a coluna, i<=j) no padrão do LogDetConeTriangle."
function triangle_vec(M::AbstractMatrix)
    p1, p2 = size(M)
    @assert p1 == p2
    v = Vector{eltype(M)}(undef, p1 * (p1 + 1) ÷ 2)
    idx = 1
    @inbounds for j in 1:p1
        @views col = M[1:j, j]
        copyto!(v, idx, col, 1, j)
        idx += j
    end
    return v
end

"""
Subproblema convexo (cone logdet), cacheado, com optimizer parametrizável.
"""
mutable struct Subproblem
    model::Model
    x::Vector{VariableRef}
    n::Int
end

function Subproblem(
    inst::Instance;
    optimizer = MosekTools.Optimizer,
    silent::Bool = true,
)
    n, p = inst.n, inst.p

    model = Model(optimizer)
    silent && set_silent(model)

    @variable(model, 0.0 <= x[1:n] <= 1.0)
    @constraint(model, sum(x) == 1.0)

    @expression(
        model,
        Σ[i = 1:p, j = 1:p],
        sum(inst.V[r, i] * x[r] * inst.V[r, j] for r in 1:n) +
        (i == j ? inst.delta : 0.0)
    )

    @variable(model, t)
    @objective(model, Min, -t)

    @constraint(model, [t; 1.0; triangle_vec(Σ)] in MOI.LogDetConeTriangle(p))

    return Subproblem(model, Vector{JuMP.VariableRef}(x), n)
end

"""
    solve!(sp, y, x_out) -> f_opt

Resolve:
  min  -logdet(Σ(x))
  s.a. sum(x)=1, 0<=x_i<=y_i

Escreve x* em x_out e retorna f*.

Robustez:
- Aceita MOI.SLOW_PROGRESS ou MOI.TIME_LIMIT se existir ponto primal viável.
- Faz warm-start do subproblema com x_out (clamp em [0, y_i]).
"""
function solve!(sp::Subproblem, y::AbstractVector{<:Real}, x_out::Vector{Float64})
    @assert length(y) == sp.n
    @assert length(x_out) == sp.n

    # Atualiza bounds e warm-start coerente (x <= y)
    @inbounds for i in 1:sp.n
        ub = Float64(y[i])
        ub = ub < 0.0 ? 0.0 : (ub > 1.0 ? 1.0 : ub)
        set_upper_bound(sp.x[i], ub)

        xi = x_out[i]
        xi = xi < 0.0 ? 0.0 : xi
        xi = xi > ub ? ub : xi
        set_start_value(sp.x[i], xi)
    end

    optimize!(sp.model)

    st = termination_status(sp.model)
    ps = primal_status(sp.model)

    ok =
        (st == MOI.OPTIMAL) ||
        (st == MOI.LOCALLY_SOLVED) ||
        ((st == MOI.SLOW_PROGRESS || st == MOI.TIME_LIMIT) && ps == MOI.FEASIBLE_POINT)

    ok || error("Subproblema não ótimo: status = $st")

    @inbounds for i in 1:sp.n
        x_out[i] = value(sp.x[i])
    end

    return objective_value(sp.model)
end

"Wrapper que aloca x."
function solve!(sp::Subproblem, y::AbstractVector{<:Real})
    x = zeros(Float64, sp.n)
    f = solve!(sp, y, x)
    return x, f
end

"Wrapper simples (sem cache)."
function solve_subproblem(
    inst::Instance,
    y::AbstractVector{<:Real};
    optimizer = MosekTools.Optimizer,
    silent::Bool = true,
)
    sp = Subproblem(inst; optimizer = optimizer, silent = silent)
    return solve!(sp, y)
end
