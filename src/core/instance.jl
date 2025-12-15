# core/instance.jl

"""
Instância do problema D-ótimo discreto (MINLP/MICP convexo).
"""
struct Instance
    V::Matrix{Float64}
    n::Int
    p::Int
    k::Int
    delta::Float64

    function Instance(V::Matrix{Float64}, k::Int; delta::Float64 = 1e-8)
        n, p = size(V)
        @assert p >= 1 "p deve ser ≥ 1"
        @assert n >= 1 "n deve ser ≥ 1"
        @assert 1 <= k <= n "k deve estar entre 1 e n"
        @assert delta >= 0.0 "delta deve ser ≥ 0"
        return new(V, n, p, k, delta)
    end
end

function Instance(V::AbstractMatrix{<:Real}, k::Integer; delta::Real = 1e-8)
    return Instance(Matrix{Float64}(V), Int(k); delta = Float64(delta))
end

# ------------------------------------------------------------------
# Geradores de instâncias
# ------------------------------------------------------------------

"""
Regressão polinomial 1D em t ∈ [-1, 1]: v_i = (1, t_i, t_i^2, ..., t_i^(p-1))'.
"""
function make_poly1d_instance(p::Int, n::Int, k::Int; delta::Float64 = 1e-8)::Instance
    @assert n >= p "para ter posto completo tipicamente use n ≥ p"
    t = range(-1.0, 1.0; length = n)

    V = Matrix{Float64}(undef, n, p)
    @inbounds for i in 1:n
        V[i, 1] = 1.0
        ti = t[i]
        pow = 1.0
        for j in 2:p
            pow *= ti
            V[i, j] = pow
        end
    end

    return Instance(V, k; delta = delta)
end

"""
Gera v_i ~ N(0, I_p), opcionalmente normalizados.
"""
function make_gaussian_instance(
    p::Int,
    n::Int,
    k::Int;
    seed::Union{Nothing,Int} = nothing,
    normalize::Bool = true,
    delta::Float64 = 1e-8,
)::Instance
    @assert n >= p "para ter posto completo tipicamente use n ≥ p"

    if seed !== nothing
        Random.seed!(seed)
    end

    V = randn(n, p)

    if normalize
        @inbounds for i in 1:n
            vi = @view V[i, :]
            nrm = norm(vi)
            if nrm > 0
                vi ./= nrm
            end
        end
    end

    return Instance(V, k; delta = delta)
end
