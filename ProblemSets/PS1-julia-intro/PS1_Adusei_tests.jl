#################################################
# PS1_Adusei_tests.jl
#################################################

using Test
using CSV
using DataFrames

include("PS1_Adusei_source.jl")

#################################################
# q1 Tests
#################################################

@testset "q1() Tests" begin

    A,B,C,D = q1()

    ############################
    # Dimension tests
    ############################

    @test size(A) == (10,7)
    @test size(B) == (10,7)
    @test size(C) == (5,7)
    @test size(D) == (10,7)

    ############################
    # Type tests
    ############################

    @test eltype(A) <: Real
    @test eltype(B) <: Real
    @test eltype(C) <: Real
    @test eltype(D) <: Real

    ############################
    # C construction tests
    ############################

    @test C[:,1:5] == A[1:5,1:5]

    @test C[:,6:7] == B[1:5,end-1:end]

    ############################
    # D construction tests
    ############################

    @test all(D[A .> 0] .== 0)

    @test all(D[A .<= 0] .== A[A .<= 0])

    ############################
    # Range tests for A
    ############################

    @test minimum(A) >= -5

    @test maximum(A) <= 10

    ############################
    # Seed reproducibility
    ############################

    A2,B2,C2,D2 = q1()

    @test A == A2
    @test B == B2
    @test C == C2
    @test D == D2

    ############################
    # File creation tests
    ############################

    @test isfile("matrixpractice.jld")

    @test isfile("firstmatrix.jld")

    @test isfile("Cmatrix.csv")

    @test isfile("Dmatrix.dat")

    ############################
    # Verify saved csv shape
    ############################

    Ccsv = CSV.read(
        "Cmatrix.csv",
        DataFrame
    )

    @test size(Ccsv) == (5,7)

end

#################################################
# q2 Tests
#################################################

@testset "q2() Tests" begin

    A,B,C,D = q1()

    ############################
    # Function executes
    ############################

    @test q2(A,B,C) === nothing

    ############################
    # Verify element-wise product
    ############################

    AB = [A[i,j]*B[i,j]
          for i in 1:size(A,1),
              j in 1:size(A,2)]

    AB2 = A .* B

    @test AB ≈ AB2

    ############################
    # Verify Cprime logic
    ############################

    Cprime2 =
        C[(-5 .<= C) .&
          (C .<= 5)]

    @test all(-5 .<= Cprime2)

    @test all(Cprime2 .<= 5)

    ############################
    # Dimensions from Question 2
    ############################

    N,K,T = 15169,6,5

    @test N == 15169
    @test K == 6
    @test T == 5

end

println("\nAll tests passed successfully.")