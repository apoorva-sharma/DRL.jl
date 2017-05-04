# ExperienceReplay.jl
# stuff.

abstract ReplayMemory

# I hate julia Vectors sometimes
typealias IntRealVector Union{Int, Vector{Real}, Vector{Int}, Vector{Float64}, Vector{Float32}}

size(::ReplayMemory) = error("Unimplemented")
#push!(::ReplayMemory, ::RealVector, ::Int, ::Real, ::RealVector, td::Real=1.;
#        rng::Union{Void,AbstractRNG}=nothing) = error("Unimplemented")
peek(::ReplayMemory; rng::Union{Void,AbstractRNG}=nothing) = error("Unimplemented")
state(::ReplayMemory, idx::Int) = error("Unimplemented")

# TODO ref paper
type UniformMemory <: ReplayMemory
    states::mx.NDArray # giant NDArray for speed--probably not too much fatter in memory
    actions::Union{Vector{Int},mx.NDArray} # which action was taken
    rewards::RealVector
    terminals::Vector{Bool}
    weights::RealVector
    mem_size::Int
    vectorized_actions::Bool
    rng::Nullable{AbstractRNG}
end
function UniformMemory(mdp::MDP; 
                        vectorized_actions::Bool=false,
                        mem_size::Int=256, 
                        rng::Nullable{AbstractRNG}=Nullable{AbstractRNG}())
    s = initial_state(mdp, RandomDevice())
    s_vec = vec(mdp, s)
    # if length(s_vec) == 2
    #     if size(s_vec)[2] == 1
    #         s_vec = vec(s_vec)
    #     end
    # end

    # TODO is there any case in which actions might have a higher dimensional representation?
    if vectorized_actions
        acts = mx.zeros( dimensions( POMDPs.actions(mdp) ), mem_size * 2 )
    else
        acts = zeros(Int, mem_size)
    end

    # currently pushes to cpu context (by default...)
    return UniformMemory(
                        mx.zeros(size(s_vec)..., mem_size * 2),
                        acts,
                        zeros(mem_size),
                        falses(mem_size),
                        ones(mem_size),
                        0,
                        vectorized_actions,
                        rng
                        )
end
size(mem::UniformMemory) = mem.mem_size # ??size(mem.states, 2) / 2
function push!(mem::UniformMemory, 
                s_vec::RealVector,
                a::IntRealVector,
                r::Real,
                sp_vec::RealVector,
                terminalp::Bool=false,
                td::Real=1.;
                weight=1.,
                rng::Union{Void,AbstractRNG}=nothing )
    if mem.mem_size * 2 > size(mem.states, 2)
        error("Oh shoot something messed up here")
    end

    # if memory is full
    if mem.mem_size * 2 == size(mem.states, 2)
        replace_idx = 0
        if rng == nothing
            replace_idx = rand(mem.rng, 1:mem.mem_size)
        else
            replace_idx = rand(rng, 1:mem.mem_size)
        end


        if mem.vectorized_actions
            mem.actions[replace_idx:replace_idx] = a
        else
            mem.actions[replace_idx] = a
        end
        mem.rewards[replace_idx] = r
        mem.terminals[replace_idx] = terminalp
        mem.weights[replace_idx] = weight

        mem.states[replace_idx:replace_idx] = reshape(s_vec, length(s_vec), 1)
        idx2 = replace_idx + mem.mem_size
        mem.states[idx2:idx2] = reshape(sp_vec, length(sp_vec), 1)

        return
    end


    mem.mem_size += 1

    if mem.vectorized_actions
        mem.actions[mem.mem_size:mem.mem_size] = a
    else
        mem.actions[mem.mem_size] = a
    end
    mem.rewards[mem.mem_size] = r
    mem.terminals[mem.mem_size] = terminalp
    mem.weights[mem.mem_size] = weight

    mem.states[mem.mem_size:mem.mem_size] = reshape(s_vec, length(s_vec), 1)
    idx2 = mem.mem_size + convert(Int, size(mem.states, 2)/2)
    mem.states[idx2:idx2] = reshape(sp_vec, length(sp_vec), 1)

end

function peek(mem::UniformMemory; rng::Union{Void,AbstractRNG}=nothing )

    idx = rand( rng==nothing ? mem.rng : rng, 1:mem.mem_size)

    return idx, 
            mem.vectorized_actions ? idx : mem.actions[idx], 
            mem.rewards[idx], 
            idx + convert(Int,(size(mem.states, 2) / 2)), 
            mem.terminals[idx]
end

state(mem::UniformMemory, idx::Int) = mem.states[idx:idx]
action(mem::UniformMemory, idx::Int) = mem.actions[idx:idx]
weight(mem::UniformMemory, idx::Int) = mem.weights[idx]







type PrioritizedMemory <: ReplayMemory
    capacity::Int
    states::mx.NDArray # giant NDArray for speed--probably not too much fatter in memory
    actions::Vector{Int} # which action was taken
    rewards::RealVector
    terminals::Vector{Bool}
    weights::RealVector

    mem_size::Int
    priority_tree::RealVector
    eps::Float64
    Beta::Float64

    write_idx::Int64

    rng::Nullable{AbstractRNG}
end

function PrioritizedMemory(mdp::MDP; 
                        capacity::Int=256, 
                        rng::Nullable{AbstractRNG}=Nullable{AbstractRNG}())
    s = initial_state(mdp, RandomDevice())
    s_vec = vec(mdp, s)

    acts = zeros(Int, capacity)

    return PrioritizedMemory(
                        capacity,
                        mx.zeros(size(s_vec)..., 2*capacity),
                        zeros(Int, capacity),
                        zeros(capacity),
                        falses(capacity),
                        ones(capacity),
                        0,
                        zeros(Float64, capacity*2 - 1),
                        0.01,
                        0.9,
                        1,
                        rng
                        )
end

size(mem::PrioritizedMemory) = mem.mem_size

# priority_tree functions
# propagate change to a tree leaf to the root
function propagate!(tree::RealVector, idx::Int64, change::Float64)
    parent = convert(Int, floor( ((idx-1) - 1) / 2 ) + 1)
    tree[parent] += change

    if parent != 1 # i.e. we are not at the root
        propagate!(tree, parent, change)
    end
end

# retrieve the first element from the tree such that the elements considered so far sum up to s
function retrieve(tree::RealVector, idx::Int64, s::Float64)
    left = 2*(idx-1) + 2
    right = left + 1

    if left > length(tree)
        return idx
    end

    if s <= tree[left]
        return retrieve(tree, left, s)
    else
        return retrieve(tree, right, s-tree[left])
    end
end

# update the priority at idx to have the given value
function update!(tree::RealVector, idx::Int64, priority::Float64)
    change = priority - tree[idx]
    tree[idx] = priority
    propagate!(tree, idx, change)
end

# return the total prioirty in the tree
total(tree::RealVector) = tree[1]

# PrioritizedMemory functions
function push!(mem::PrioritizedMemory, 
               s_vec::RealVector,
               a::Int,
               r::Real,
               sp_vec::RealVector,
               terminalp::Bool=false,
               td::Real=1.;
               weight::Real=1.,
               rng::Union{Void,AbstractRNG}=nothing)
    tree_idx = mem.write_idx + (mem.capacity - 1)

    #write data at write_idx
    mem.actions[mem.write_idx] = a
    mem.rewards[mem.write_idx] = r
    mem.terminals[mem.write_idx] = terminalp
    mem.weights[mem.write_idx] = weight

    mem.states[mem.write_idx:mem.write_idx] = reshape(s_vec, length(s_vec), 1)
    
    idx2 = mem.write_idx + mem.capacity

    mem.states[idx2:idx2] = reshape(sp_vec, length(sp_vec), 1)

    # increment write position
    mem.write_idx += 1
    if mem.write_idx > mem.capacity
        mem.write_idx = 1
    end

    # increment memory size
    if mem.mem_size < mem.capacity
        mem.mem_size += 1
    end

    # update tree with the new priority element
    update!(mem.priority_tree, tree_idx, 1.)#abs(td)+mem.eps)
end

function peek(mem::PrioritizedMemory; rng::Union{Void,AbstractRNG}=nothing )

    s = total(mem.priority_tree)*rand( rng==nothing ? mem.rng : rng )

    tree_idx = retrieve(mem.priority_tree, 1, s)

    idx = tree_idx - (mem.capacity - 1)

    return idx, 
            mem.actions[idx], 
            mem.rewards[idx], 
            idx + mem.capacity,
            mem.terminals[idx]
end

state(mem::PrioritizedMemory, idx::Int) = mem.states[idx:idx]
action(mem::PrioritizedMemory, idx::Int) = mem.actions[idx:idx]
weight(mem::PrioritizedMemory, idx::Int) = mem.weights[idx]*(mem.mem_size*mem.priority_tree[idx+mem.capacity-1])^(-mem.Beta)

