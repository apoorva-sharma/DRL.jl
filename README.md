# DRL.jl

## Deep Reinforcement Learning for Julia

This is a fork of [\@cho3's DRL.jl](https://github.com/cho3/DRL.jl), a package that tries to provide implementations Deep Reinforcement Learning algorithms for solving any problem that can be expressed in the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) framework. It leverages [MXNet.jl](https://github.com/dmlc/MXNet.jl), a neural network framework with Julia libraries. I have found MXNet (at least when working with the Julia library) to be cumbersome at times, and considering that there is now a TensorFlow julia library, I would consider porting this codebase to use that.

There are a lot of solvers that have been partially implemented in this repository, but currently only DQN is working. Additionally, I have added a "Guided DQN" implementation which attempts to choose the initial state for each simulation in a way that selects "surprising" trajectories, in an effort to use importance sampling over trajectories to obtain better training performance. It is not fully tested, and it is unclear at this point whether this approach is actually useful.


NOTE: the signature for solve doesn't exactly match `POMDPs.solve`. It is `solve(::Solver, ::MDP, ::Policy, ::rng)`

NOTE: try to define your vectors as Float32 if possible (which is what mxnet uses)
