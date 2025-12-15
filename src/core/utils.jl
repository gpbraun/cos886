# core/utils.jl

"""
Tipo abstrado para estado do algorítimo.
"""
abstract type AbstractState end

"""
Resultados do algorítimo.
"""
Base.@kwdef struct Results
    x::Vector{Float64} = Float64[]
    y::Vector{Int} = Int[]
    lb::Float64 = -Inf
    ub::Float64 = Inf
    gap::Float64 = Inf
    time::Float64 = 0.0
    iters::Int = 0
    nodes::Int = 0
end

"""
Retorna: tempo decorrido.
"""
@inline time_elapsed(t0::Float64) = (time() - t0)

"""
Retorna: tempo remanescente.
"""
@inline time_remaining(t0::Float64, limit::Float64) = (limit - time_elapsed(t0))

"""
Retorna: gap relativo
"""
function relative_gap(lb::Real, ub::Real)::Float64
    (isfinite(lb) && isfinite(ub)) || return Inf

    return max(0.0, (Float64(ub) - Float64(lb)) / abs(Float64(ub)))
end

"""
Arredonda para 0/1.
"""
round01(v::Real) = v >= 0.5 ? 1 : 0

"""
Arredonda para 0/1 (Vector).
"""
round01(y::AbstractVector{<:Real}) = map(v -> round01(v), y)

"""
Checa se vetor é inteiro (0/1) dentro de tolerância.
"""
function is_integral(y::AbstractVector{<:Real}; tol::Float64 = 1e-6)
    return all(abs.(y .- round.(y)) .<= tol)
end

"""
Estende JuMP.set_start_value para vetores/matrizes de variáveis
"""
function JuMP.set_start_value(
    vars::AbstractArray{<:JuMP.VariableRef},
    vals::AbstractArray{<:Real},
)
    @assert axes(vars) == axes(vals)
    @inbounds for i in eachindex(vars, vals)
        JuMP.set_start_value(vars[i], Float64(vals[i]))
    end
    return nothing
end

"""
Estende value para vetores/matrizes de variáveis
"""
function get_value!(out::AbstractArray{<:Real}, vars::AbstractArray{<:JuMP.VariableRef})
    @assert axes(out) == axes(vars)
    @inbounds for i in eachindex(out, vars)
        out[i] = JuMP.value(vars[i])
    end
    return out
end
