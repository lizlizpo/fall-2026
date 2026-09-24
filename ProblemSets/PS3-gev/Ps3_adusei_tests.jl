#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 3
# Elizabeth Adusei
#
# Unit tests
#---------------------------------------------------

using Test

cd(@__DIR__)

# Read in the functions
include("PS3_Adusei_source.jl")
#---------------------------------------------------
# Test 1: Data loading
#---------------------------------------------------

@testset "Data loading" begin

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS3-gev/nlsw88w.csv"

    df, X, Z, y = load_data(url)

    @test size(X,2) == 3
    @test size(Z,2) == 8
    @test size(X,1) == size(Z,1)
    @test size(X,1) == length(y)

end
#---------------------------------------------------
# Test 2: Multinomial logit likelihood
#---------------------------------------------------

@testset "Multinomial logit likelihood" begin

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS3-gev/nlsw88w.csv"

    df, X, Z, y = load_data(url)

    theta = zeros(22)

    loglike = mlogit_with_Z(theta, X, Z, y)

    @test isfinite(loglike)
    @test loglike >= 0

end
#---------------------------------------------------
# Test 3: Nested logit likelihood
#---------------------------------------------------

@testset "Nested logit likelihood" begin

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS3-gev/nlsw88w.csv"

    df, X, Z, y = load_data(url)

    nesting_structure = [
        [1,2,3],
        [4,5,6,7]
    ]

    theta = [
        zeros(6);
        1.0;
        1.0;
        0.0
    ]

    loglike = nested_logit_with_Z(
        theta,
        X,
        Z,
        y,
        nesting_structure
    )

    @test isfinite(loglike)
    @test loglike >= 0

end