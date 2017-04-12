using POMDPs
importall POMDPModels
using GraphViz
using MXNet
using DiscreteValueIteration
using POMDPToolbox
using StatsBase
typealias RealVector Union{Vector{Real}, Vector{Int}, Vector{Float64}, Vector{Float32}}
import Base: size, push!, peek, copy!, convert

include("ExperienceReplay.jl")


# stuff to make things work
importall POMDPs
iterator(ipa::POMDPModels.InvertedPendulumActions) = ipa.actions

ip = InvertedPendulum()

mem = PrioritizedMemory(ip, capacity=8)

push!(mem, [1,1], 1, 2, [1,2], false, 1.)
push!(mem, [1,1], 2, 2, [1,2], false, 2.)
push!(mem, [1,1], 3, 2, [1,2], false, 3.)
push!(mem, [1,1], 4, 2, [1,2], false, 2.)

rng = RandomDevice()
a_idxs = zeros(Int, 1000)
for i = 1:1000
    _, a_idxs[i], _, _, _ = peek(mem, rng=rng)
end

display(counts(a_idxs, 1:4))