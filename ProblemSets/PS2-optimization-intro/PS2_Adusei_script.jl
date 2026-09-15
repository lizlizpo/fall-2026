# script file

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

include("PS2_Adusei_source.jl")

function main()

    q1()
    q2()
    q3_q4()
    q5()

    println("Ran successfully")

    return nothing
end

main()
