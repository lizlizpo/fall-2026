#################################################
# PS1_Adusei_source.jl
#################################################

using JLD
using Random
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using FreqTables
using Distributions

#################################################
# Question 1
#################################################

function q1()

    Random.seed!(1234)

    # (a)
    A = rand(Uniform(-5,10), 10,7)
    B = rand(Normal(-2,15), 10,7)

    C = [A[1:5,1:5] B[1:5,end-1:end]]

    D = A .* (A .<= 0)

    # (b)
    println("Number of elements in A: ", length(A))

    # (c)
    println("Number of unique elements in D: ",
            length(unique(D)))

    # (d)
    E = vec(B)

    # (e)
    F = cat(A,B; dims=3)

    # (f)
    F = permutedims(F,(3,1,2))

    # (g)
    G = kron(B,C)

    # (h)
    save("matrixpractice.jld",
         "A",A,
         "B",B,
         "C",C,
         "D",D,
         "E",E,
         "F",F,
         "G",G)

    # (i)
    save("firstmatrix.jld",
         "A",A,
         "B",B,
         "C",C,
         "D",D)

    # (j)
    CSV.write("Cmatrix.csv",
              DataFrame(C,:auto))

    # (k)
    CSV.write("Dmatrix.dat",
              DataFrame(D,:auto);
              delim='\t')

    return A,B,C,D

end

#################################################
# Question 2
#################################################

function q2(A,B,C)

    # (a)
    AB = [A[i,j] * B[i,j]
          for i in 1:size(A,1),
              j in 1:size(A,2)]

    AB2 = A .* B

    # (b)
    Cprime = Float64[]

    for j in 1:size(C,2),
        i in 1:size(C,1)

        if -5 <= C[i,j] <= 5
            push!(Cprime,C[i,j])
        end
    end

    Cprime2 =
        C[(-5 .<= C) .&
          (C .<= 5)]

    # (c)
    N,K,T = 15169,6,5

    X = zeros(N,K,T)

    col5 = rand(Binomial(20,0.6),N)
    col6 = rand(Binomial(20,0.5),N)

    for t in 1:T

        X[:,1,t] .= 1

        X[:,2,t] =
            rand(N) .<
            (0.75*(6-t)/5)

        sd3 = t == 1 ?
              0.01 :
              5*(t-1)

        X[:,3,t] =
            rand(
                Normal(
                    15+t-1,
                    sd3
                ),
                N
            )

        X[:,4,t] =
            rand(
                Normal(
                    pi*(6-t)/3,
                    1/exp(1)
                ),
                N
            )

        X[:,5,t] = col5
        X[:,6,t] = col6

    end

    # (d)
    beta = [
        i == 1 ? 1 + 0.25*(t-1) :
        i == 2 ? log(t) :
        i == 3 ? -sqrt(t) :
        i == 4 ? exp(t)-exp(t+1) :
        i == 5 ? t :
                 t/3
        for i in 1:K,
            t in 1:T
    ]

    # (e)
    Y = zeros(N,T)

    for t in 1:T
        Y[:,t] =
            X[:,:,t] *
            beta[:,t] .+
            rand(
                Normal(0,0.36),
                N
            )
    end

    return nothing

end