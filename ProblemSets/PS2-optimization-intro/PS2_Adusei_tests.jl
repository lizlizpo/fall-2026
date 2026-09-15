#################################################
# PS2_Adusei_tests.jl
#################################################

using Test
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

include("PS2_Adusei_source.jl")#################################################
# OLS Function Tests
#################################################

@testset "OLS Tests" begin

    X = [1.0 1.0;
         1.0 2.0;
         1.0 3.0]

    beta = [1.0, 2.0]

    y = X * beta

    @test ols(beta, X, y) ≈ 0.0
    @test ols(beta, X, y) >= 0.0

end

#################################################
# Binary Logit Likelihood Tests
#################################################

@testset "Logit Likelihood Tests" begin

    X = [1.0 0.0;
         1.0 1.0;
         1.0 2.0;
         1.0 3.0]

    y = [0.0, 1.0, 0.0, 1.0]

    beta = zeros(2)

    expected_nll = length(y) * log(2)

    @test logit_like(beta, X, y) ≈ expected_nll
    @test logit_like(beta, X, y) >= 0.0

end

#################################################
# Multinomial Logit Likelihood Tests
#################################################

@testset "Multinomial Logit Tests" begin

    X = [1.0 0.0;
         1.0 1.0;
         1.0 2.0]

    y = [1, 2, 3]

    J = maximum(y)
    K = size(X, 2)

    alpha = zeros(K * (J - 1))

    expected_nll = length(y) * log(J)

    @test mlogit(alpha, X, y) ≈ expected_nll
    @test mlogit(alpha, X, y) >= 0.0

end