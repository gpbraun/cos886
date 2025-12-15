# core/eval.jl

"""
Worker para acelerar f(x) e ∇f(x) evitando alocações repetidas.
"""
mutable struct EvalWorker
    s::Vector{Float64}
    XV::Matrix{Float64}
    M::Matrix{Float64}
    Z::Matrix{Float64}
end

function EvalWorker(inst::Instance)
    return EvalWorker(
        zeros(inst.n),
        zeros(inst.n, inst.p),
        zeros(inst.p, inst.p),
        zeros(inst.p, inst.n),
    )
end

"""
Constrói: Σ(x)
"""
@inline function calculate_sigma!(
    ws::EvalWorker,
    x::AbstractVector{<:Real},
    inst::Instance,
)::Nothing
    n, p = inst.n, inst.p

    # s[i] = sqrt(max(x[i],0))
    @inbounds for i in 1:n
        xi = Float64(x[i])
        ws.s[i] = xi > 0.0 ? sqrt(xi) : 0.0
    end

    # W = diag(s) * V
    @inbounds for j in 1:p
        for i in 1:n
            ws.XV[i, j] = inst.V[i, j] * ws.s[i]
        end
    end

    # Σ = W'W  (triângulo superior)
    BLAS.syrk!('U', 'T', 1.0, ws.XV, 0.0, ws.M)

    # Σ += delta*I
    @inbounds for d in 1:p
        ws.M[d, d] += inst.delta
    end

    return
end

"""
Calcula: f(x) = -logdet(Σ(x)).
Deixa o fator U (triângulo superior) armazenado em ws.M após a Cholesky.
"""
function calculate_f!(
    ws::EvalWorker,
    x::AbstractVector{<:Real},
    inst::Instance,
)::Float64
    n, p = inst.n, inst.p
    @assert length(x) == n

    calculate_sigma!(ws, x, inst)

    # Cholesky no triângulo superior (ws.M será sobrescrita por U)
    F = cholesky!(Symmetric(ws.M, :U); check = false)

    # Se (raro) falhar numericamente, reconstrói Σ e tenta com jitter crescente.
    if !issuccess(F)
        jitter = max(1e-12, 1e-9 * inst.delta)
        ok = false
        for _ in 1:6
            calculate_sigma!(ws, x, inst)
            @inbounds for d in 1:p
                ws.M[d, d] += jitter
            end
            F = cholesky!(Symmetric(ws.M, :U); check = false)
            if issuccess(F)
                ok = true
                break
            end
            jitter *= 10.0
        end
        ok || return Inf
    end

    # f = -logdet(Σ) = -2*sum(log(diag(U)))
    sld = 0.0
    @inbounds for j in 1:p
        sld += log(F.U[j, j])
    end

    return -2.0 * sld
end

"""
Calcula: gradiente.
"""
function calculate_grad!(ws::EvalWorker, inst::Instance, g::Vector{<:Real})::Nothing
    n, p = inst.n, inst.p
    @assert length(g) == n

    # ws.Z := V' (p×n)
    copyto!(ws.Z, transpose(inst.V))

    # resolve U' * T = V'  => T = U' \ V'
    U = UpperTriangular(ws.M)
    ldiv!(LowerTriangular(U'), ws.Z)

    # g_i = -||T[:,i]||^2 
    @inbounds for i in 1:n
        s = 0.0
        for j in 1:p
            v = ws.Z[j, i]
            s += v * v
        end
        g[i] = -s
    end

    return
end

"""
Calcula: f e o gradiente.
"""
function f_and_grad!(
    ws::EvalWorker,
    x::AbstractVector{<:Real},
    inst::Instance,
    g::Vector{Float64},
)::Float64
    f = calculate_f!(ws, x, inst)
    calculate_grad!(ws, inst, g)
    return f
end
