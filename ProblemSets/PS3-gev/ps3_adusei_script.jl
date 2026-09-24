#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 3
# Elizabeth Adusei
#
# Script file: load packages and source file, then run everything.
#---------------------------------------------------

using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

# Read in the functions
include("PS3_Adusei_source.jl")

#---------------------------------------------------
# Question 2: Interpretation of γ̂
#---------------------------------------------------
# In the multinomial logit model, γ measures the effect of the relative
# expected log wage, (Z_ij - Z_i8), on the utility of choosing occupation j
# relative to the normalized occupation 8.
#
# The estimate is γ̂ ≈ -0.0942. The negative sign means that, holding the
# other variables constant, an increase in occupation j's expected log wage
# relative to occupation 8 decreases the utility, and therefore the choice
# probability, of occupation j relative to occupation 8.
#
# The negative sign is somewhat counterintuitive because we would normally
# expect a higher expected wage to make an occupation more attractive.

#---------------------------------------------------
# Question 4: Call the main function
#---------------------------------------------------

allwrap()