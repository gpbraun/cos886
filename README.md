# COS886

Otimização Não-Linear Inteira Mista

## O projeto

Resolução do problema de D-design ótimo discreto (MINLP/MICP convexo), que consiste em selecionar $k$ dentre $n$ candidatos de forma a maximizar o critério D-otimalidade (log-det da matriz de informação de Fisher).

Os métodos implementados são:

1. **ECP** — Extended Cutting Planes
2. **OA** — Outer Approximation
3. **BB** — Branch-and-Bound com callbacks (sem e com User Cuts)

## Estrutura

```
src/
  Cos886.jl          # módulo principal
  core/
    instance.jl      # struct Instance e geradores de instâncias
    eval.jl          # avaliação do critério D-ótimo
    utils.jl         # utilitários
  models/
    master.jl        # problema mestre (MILP)
    subproblem.jl    # subproblema (SDP/SOCP)
  algorithms/
    ecp.jl           # Extended Cutting Planes
    oa.jl            # Outer Approximation
    bb.jl            # Branch-and-Bound com callbacks
experiments/
  experiments.jl     # script de experimentos
```

## Instalação

Ative o ambiente Julia e instancie as dependências:

```shell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

As licenças do Gurobi e do Mosek devem estar configuradas no ambiente.

## Experimentos

Para rodar os experimentos:

```shell
julia experiments/experiments.jl
```

Os resultados são registrados em `experiments/experiments.txt`.
