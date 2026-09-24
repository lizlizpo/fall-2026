using DataFrames
using CSV
using HTTP
using Optim
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using HTTP
using Optim
using LinearAlgebra
using Random
using Statistics

#---------------------------------------------------
# Data Loading Function
#---------------------------------------------------

function load_data(url)

    df = CSV.read(HTTP.get(url).body, DataFrame)

    X = [df.age df.white df.collgrad]

    Z = hcat(
        df.elnwage1,
        df.elnwage2,
        df.elnwage3,
        df.elnwage4,
        df.elnwage5,
        df.elnwage6,
        df.elnwage7,
        df.elnwage8
    )

    y = df.occupation

    return df, X, Z, y

end

#---------------------------------------------------
# Question 1
# Multinomial Logit
#---------------------------------------------------

function mlogit_with_Z(θ, X, Z, y)

    α = θ[1:end-1]
    γ = θ[end]

    K = size(X,2)
    J = length(unique(y))
    N = length(y)

    # Choice indicator matrix
    bigY = zeros(N,J)

    for j = 1:J
        bigY[:,j] = y .== j
    end

    # Normalize alternative J
    bigα = [reshape(α,K,J-1) zeros(K)]

    T = promote_type(eltype(X), eltype(θ))

    num = zeros(T,N,J)

    # Numerators
    for j = 1:J

        num[:,j] =
            exp.(
                X * bigα[:,j] .+
                γ .* (Z[:,j] .- Z[:,J])
            )

    end

    # Denominator
    dem = sum(num,dims=2)

    # Choice probabilities
    P = num ./ dem

    # Negative log likelihood
    loglike = -sum(bigY .* log.(P .+ 1e-10))

    return loglike

end

#---------------------------------------------------
# Question 3
# Nested Logit
#---------------------------------------------------

function nested_logit_with_Z(θ, X, Z, y, nesting_structure)

    α = θ[1:end-3]
    λ = θ[end-2:end-1]
    γ = θ[end]

    K = size(X,2)
    J = length(unique(y))
    N = length(y)

    bigY = zeros(N,J)

    for j = 1:J
        bigY[:,j] = y .== j
    end

    bigAlpha = cat(
        repeat(α[1:K],1,length(nesting_structure[1])),
        repeat(α[K+1:2K],1,length(nesting_structure[2])),
        zeros(K,1),
        dims=2
    )

    T = promote_type(eltype(X), eltype(θ))

    lidx = zeros(T,N,J)
    num = zeros(T,N,J)
    dem = zeros(T,N)

    for j = 1:J

        if j in nesting_structure[1]

            lidx[:,j] = exp.(
                (
                    X * bigAlpha[:,j] .+
                    γ .* (Z[:,j] .- Z[:,J])
                ) ./ λ[1]
            )

        elseif j in nesting_structure[2]

            lidx[:,j] = exp.(
                (
                    X * bigAlpha[:,j] .+
                    γ .* (Z[:,j] .- Z[:,J])
                ) ./ λ[2]
            )

        else

            lidx[:,j] .= 1.0

        end

    end

    WCsum = sum(lidx[:,nesting_structure[1]],dims=2)
    BCsum = sum(lidx[:,nesting_structure[2]],dims=2)

    for j = 1:J

        if j in nesting_structure[1]

            num[:,j] =
                lidx[:,j] .* WCsum.^(λ[1]-1)

        elseif j in nesting_structure[2]

            num[:,j] =
                lidx[:,j] .* BCsum.^(λ[2]-1)

        else

            num[:,j] = lidx[:,j]

        end

        dem .+= num[:,j]

    end

    P = num ./ dem

    loglike = -sum(bigY .* log.(P .+ 1e-10))

    return loglike

end

#---------------------------------------------------
# Optimization Functions
#---------------------------------------------------

function optimize_mlogit(X,Z,y)

    startvals = [2 .* rand(7*size(X,2)) .- 1; 0.1]

    result = optimize(
        theta -> mlogit_with_Z(theta,X,Z,y),
        startvals,
        LBFGS(),
        Optim.Options(
            g_tol = 1e-5,
            iterations = 100_000,
            show_trace = true
        )
    )

    return result.minimizer

end

function optimize_nested_logit(X,Z,y,nesting_structure)

    startvals = [
        2 .* rand(2*size(X,2)) .- 1;
        1.0;
        1.0;
        0.1
    ]

    result = optimize(
        theta -> nested_logit_with_Z(
            theta,
            X,
            Z,
            y,
            nesting_structure
        ),
        startvals,
        LBFGS(),
        Optim.Options(
            g_tol = 1e-5,
            iterations = 100_000,
            show_trace = true
        )
    )

    return result.minimizer

end

#---------------------------------------------------
# Main Wrapper
#---------------------------------------------------

function allwrap()

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS3-gev/nlsw88w.csv"

    df, X, Z, y = load_data(url)

    println("Data loaded successfully!")
    println("Sample size: ", size(X,1))
    println("Number of covariates: ", size(X,2))
    println("Number of alternatives: ", length(unique(y)))

    println("\n=== MULTINOMIAL LOGIT RESULTS ===")

    theta_hat_mle = optimize_mlogit(X,Z,y)

    println(theta_hat_mle)

    println("\n=== NESTED LOGIT RESULTS ===")

    nesting_structure = [
        [1,2,3],
        [4,5,6,7]
    ]

    theta_hat_nlogit =
        optimize_nested_logit(
            X,
            Z,
            y,
            nesting_structure
        )

    println(theta_hat_nlogit)

    return nothing

end