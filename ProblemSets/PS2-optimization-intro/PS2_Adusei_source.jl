####################################################
#question 1
####################################################

function q1()
    f(x) =  -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2 # Define the function f(x) already
    minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2 # The package Optim does only minimization so we multiply by minus to be able to use Optim
    startval = rand(1)   # random starting value
    result = optimize(minusf, startval, LBFGS())
    println("optimization summary:", result) # for extra details
    println("argmin (minimizer) is ",Optim.minimizer(result)[1])
    println("min is ",Optim.minimum(result))
    println("max is ",-Optim.minimum(result))
    return nothing
end


####################################################
#question 2
####################################################

# Non-linear Optimization
# 1- Objective function (min, max) 
# 2- Starting value
# 3- Algorithm to get from x0 to x*


function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

function q2()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(beta_hat_ols.minimizer)

    bols = inv(X'*X)*X'*y
    @show bols
    df.white = df.race.==1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)
    @show bols_lm


    #Standard errors
      
    σ²  = sum((y .- X*bols).^2)/(size(X,1)-size(X,2))
    vcov_bols = σ²*inv(X'*X)

    @show [bols sqrt.(diag(vcov_bols))]
    @show vcov_bols
    ## print it out the vcov matrix prettily
    #for i in 1:size(vcov_bols, 1)
    #for j in 1:size(vcov_bols, 2)
    # print(vcov_bols[j,j], "\t")
    # end
    # println()
    return nothing 
end


####################################################
#question 3
####################################################

function logit_like(beta, X, y)

    p = exp.(X*beta)./(1 .+ exp.(X*beta))

    neg_log_like = -sum(y.*log.(p) .+ (1 .- y).*log.(1 .- p))

    return neg_log_like
end

function q3_q4()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    @show describe(df)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    beta_hat_logit = optimize(b -> logit_like(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(beta_hat_logit.minimizer)

   #bols = inv(X'*X)*X'*y
    #@show bols
    df.white = df.race.==1
    blogit_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
    #@show blogit_glm
    println(coeftable(blogit_glm))

    return nothing 
end


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function q5()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    @show describe(df)
    freqtable(df, :occupation) # note small number of obs in some occupations
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    freqtable(df, :occupation) # problem solved

    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation

    alpha_hat_logit = optimize(b -> mlogit(b, X, y), rand((maximum(y)-1)*size(X,2)), LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))
    println(alpha_hat_logit.minimizer)

    return nothing

end

function mlogit(alpha, X, y)
    N = size(X, 1) # no of obs
    K = size(X, 2) # no of covariables
    J = maximum(y) # number of choice alternatives

    # unslice the parameter
    alpha_mat = [reshape(alpha, K, J - 1) zeros(K, 1)]

    # make d matrix
    d = zeros(N, J)
    for j = 1:J
        d[:, j] .= y .== j
    end

    # make p matrix
    p = zeros(N, J)
    denominator = sum([exp.(X * alpha_mat[:, j]) for j in 1:J])
    for j = 1:J
        p[:, j] .= exp.(X * alpha_mat[:, j]) ./ denominator
    end

    # log-likelihood expression
    loglike = -sum(d .* log.(p))


    return loglike
end