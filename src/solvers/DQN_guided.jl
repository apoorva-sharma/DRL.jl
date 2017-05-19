# DQN.jl
# built off @zsunberg's HistoryRecorder.jl
# making stuff up as I'm going along
# uses MxNet as backend because native julia etc etc
# TODO modify solve signature (add policy=create_policy)

using Distributions
using POMDPToolbox

type GDQN <: POMDPs.Solver
    nn::NeuralNetwork
    target_nn::Nullable{mx.Executor}
    exp_pol::ExplorationPolicy
    max_steps::Int
    num_epochs::Int
    checkpoint_interval::Int
    verbose::Bool
    stats::Dict{AbstractString,Vector{Real}}
    replay_mem::Nullable{ReplayMemory}
    target_refresh_interval::Int

    # exception stuff from History Recorder -- couldn't hurt
    capture_exception::Bool
    exception::Nullable{Exception}
    backtrace::Nullable{Any}
end
function GDQN(;
            nn::NeuralNetwork=build_partial_mlp(),
            target_nn::Nullable{mx.Executor}=Nullable{mx.Executor}(),
            exp_pol::ExplorationPolicy=EpsilonGreedy(),
            max_steps::Int=100,
            num_epochs::Int=100,
            checkpoint_interval::Int=5,
            verbose::Bool=true,
            stats::Dict{AbstractString,Vector{Real}}=
                    Dict{AbstractString,Vector{Real}}(
                            "r_total"=>zeros(num_epochs),
                            "td"=>zeros(num_epochs),
                            "r_test"=>zeros(num_epochs)),
            replay_mem::Nullable{ReplayMemory}=Nullable{ReplayMemory}(),
            capture_exception::Bool=false,
            target_refresh_interval::Int=10000
            )


    # TODO check stuff or something--leave replay memory null?
    return GDQN(
                nn,
                target_nn,
                exp_pol,
                max_steps,
                num_epochs,
                checkpoint_interval,
                verbose,
                stats,
                replay_mem,
                target_refresh_interval,
                capture_exception,
                nothing,
                nothing
                )

end

type GDQNPolicy{S,A} <: POMDPs.Policy
    exec::Union{mx.Executor,Void}
    input_name::Symbol
    q_values::Vector{Float32} # julia side output - for memory efficiency
    actions::Vector{A}
    mdp::MDP{S,A}
end

function create_policy(sol::GDQN, mdp::MDP)
    A = iterator(actions(mdp))
    return GDQNPolicy(
                    isnull(sol.nn.exec) ? nothing : sol.nn.exec,
                    sol.nn.input_name,
                    zeros(Float32, length(A)),
                    A,
                    mdp
                    )
end
# TODO constructor


function util{S,A}(p::GDQNPolicy{S,A}, s::S)
    # move to computational graph -- potential bottleneck?
    s_vec = vec(p.mdp, s)
    mx.copy!(p.exec.arg_dict[p.input_name], convert(Array{Float32,2}, reshape(s_vec, length(s_vec), 1) ) )

    mx.forward( p.exec )

    # possible bottleneck: copy output, get maximum element
    # TODO this won't work--change shape of one of the two
    #copy!( p.q_values, p.exec.outputs[1] )
    q_values = vec( mx.copy!( zeros(Float32, size(p.exec.outputs[1])), p.exec.outputs[1] ) )

    p_desc = sortperm( q_values, rev=true)

    # return the highest value legal action
    As = POMDPs.actions( p.mdp, s ) # TODO action space arg to keep things memory efficient
    for idx in p_desc
        a = p.actions[idx]
        if a in As
            return q_values[idx]
        end
    end
end

function POMDPs.action{S,A}(p::GDQNPolicy{S,A}, s::S)
    # TODO figure out if its better to have a reference to the mdp

    # assuming that s is of the right type and stuff, means one less handle

    # move to computational graph -- potential bottleneck?
    s_vec = vec(p.mdp, s)
    mx.copy!(p.exec.arg_dict[p.input_name], convert(Array{Float32,2}, reshape(s_vec, length(s_vec), 1) ) )

    mx.forward( p.exec )

    # possible bottleneck: copy output, get maximum element
    # TODO this won't work--change shape of one of the two
    #copy!( p.q_values, p.exec.outputs[1] )
    q_values = vec( mx.copy!( zeros(Float32, size(p.exec.outputs[1])), p.exec.outputs[1] ) )

    p_desc = sortperm( q_values, rev=true)

    # return the highest value legal action
    As = iterator(POMDPs.actions( p.mdp, s )) # TODO action space arg to keep things memory efficient
    for idx in p_desc
        a = p.actions[idx]
        if a in As
            return a
        end
    end

    error("Check your actions(mdp, s) function; no legal actions available from state $s")

end



function action{S,A}(p::EpsilonGreedy, solver::GDQN, mdp::MDP{S,A}, s::S, rng::AbstractRNG, As_all::Vector{A}, a::A=As_all[1])

    # move to computational graph -- potential bottleneck?
    s_vec = convert(Vector{Float32}, vec(mdp, s) )
    mx.copy!(get(solver.nn.exec).arg_dict[solver.nn.input_name], reshape(s_vec, length(s_vec), 1) )

    mx.forward( get(solver.nn.exec) )

    # possible bottleneck: copy output, get maximum element ( argmax over online network )
    q_values = vec( mx.copy!( zeros(Float32, size(get(solver.nn.exec).outputs[1])), get(solver.nn.exec).outputs[1] ) )

    As = iterator(POMDPs.actions( mdp, s ) ) # TODO action space arg to keep things memory efficient

    r = rand(rng)
    # explore, it's here because we need all that extra spaghetti
    if r > p.eps
        rdx = rand(rng, 1:length(As))
        return (As[rdx], q_values[rdx], rdx, s_vec,)
    end

    p_desc = sortperm( q_values, rev=true)
    q = q_values[p_desc[1]] # highest value regardless of noise or legality

    # return the highest value legal action
    for idx in p_desc
        a = As_all[idx]
        if a in As
            return (a, q, idx, s_vec,)
        end
    end

    error("Check your actions(mdp, s) function; no legal actions available from state $s")

end


function gdqn_update!( nn::NeuralNetwork, target_nn::mx.Executor, mem::ReplayMemory, refresh_target::Bool, disc::Float64, rng::AbstractRNG )

    # NOTE its probably more efficient to have a network setup for batch passes, and one for the individual passes (e.g. action(...)), depends on memory, I guess

    # TODO preallocate s, a, r, sp
    td_avg = 0.

    s_batch = []
    a_batch = []
    r_batch = []
    sp_batch = []
    terminalp_batch = []
    weights = []

    # collect batch of samples
    for idx = 1:nn.batch_size
        s_idx, a_idx, r, sp_idx, terminalp = peek(mem, rng=rng)
        append!(s_batch, s_idx)
        append!(a_batch, a_idx)
        append!(r_batch, r)
        append!(sp_batch, sp_idx)
        append!(terminalp_batch, terminalp)
        append!(weights, weight(mem, s_idx))
    end

    # normalize weights
    weights = weights/maximum(weights)

    # println("s\ta\tq\tqp")
    for idx = 1:nn.batch_size
        s_idx = s_batch[idx]
        a_idx = a_batch[idx]
        r = r_batch[idx]
        sp_idx = sp_batch[idx]
        terminalp = terminalp_batch[idx]
        weight = weights[idx]


        # setup input data accordingly
        # TODO abstract out to kDim input
        mx.copy!( target_nn.arg_dict[nn.input_name], state(mem, sp_idx) )
        mx.copy!( get(nn.exec).arg_dict[nn.input_name], state(mem, sp_idx) )

        # get target q values
        # TODO need discount/mdp
        mx.forward( target_nn )
        mx.forward( get(nn.exec) )
        qps = vec(copy!(zeros(Float32,size( get(nn.exec).outputs[1] ) ), get(nn.exec).outputs[1]))
        qps_target = vec(copy!(zeros(Float32,size( target_nn.outputs[1] ) ), target_nn.outputs[1]))

        # using the Double DQN algorithm:
        _ , ap_idx = findmax(qps)
        if terminalp
            qp = r
        else
            qp = r + disc * qps_target[ap_idx]
        end

        # setup target, do forward, backward pass to get gradient
        mx.copy!( get(nn.exec).arg_dict[nn.input_name], state(mem, s_idx) )
        mx.forward( get(nn.exec), is_train=true )
        qs = copy!( zeros(Float32, size(get(nn.exec).outputs[1])), get(nn.exec).outputs[1])

        # s_vec = mx.try_get_shared(state(mem, s_idx))
        # println("$(s_vec)\t$(a_idx)\t$(qs[a_idx])\t$(qp)")

        td_avg += (qp - qs[a_idx])^2

        qp_vec = copy(qs)
        qp_vec[a_idx] = qp
        lossGrad = copy(qs - qp_vec, mx.cpu())

        mx.backward( get(nn.exec), weight*lossGrad )
    end

    update!(nn)

    # update target network
    if refresh_target
        for (param, param_target) in zip( get(nn.exec).arg_arrays, target_nn.arg_arrays )
            mx.copy!(param_target, param)
        end
    end


    return sqrt(td_avg/nn.batch_size)

end

function addGDQNstats(solver::GDQN, s0_dist::GMM)
  n = s0_dist.n
  d = s0_dist.d
  for i in 1:n
    solver.stats["alpha$(i)"] = zeros(solver.num_epochs)
    for j in 1:d
      solver.stats["mu$(i)_$(j)"] = zeros(solver.num_epochs)
      solver.stats["sigma$(i)_$(j)$(j)"] = zeros(solver.num_epochs)
    end
  end
end

function logGDQNstats(solver::GDQN, s0_dist::GMM, ep)
  n = s0_dist.n
  d = s0_dist.d
  for i in 1:n
    solver.stats["alpha$(i)"][ep] = s0_dist.α[i]
    for j in 1:d
      solver.stats["mu$(i)_$(j)"][ep] = s0_dist.μ[i][j]
      solver.stats["sigma$(i)_$(j)$(j)"][ep] = s0_dist.Σ[i].mat[j,j]
    end
  end
end

function solve{S,A}(solver::GDQN, mdp::MDP{S,A}; policy::GDQNPolicy=create_policy(solver, mdp), s0_dist::Nullable{GMM}=nothing, rng::AbstractRNG=RandomDevice())

    sim = RolloutSimulator(max_steps=solver.max_steps)

    if !isnull(s0_dist)
        addGDQNstats(solver, get(s0_dist))
    end

    # setup experience replay; initialized here because of the whole solve paradigm (decouple solver, problem)
    if isnull(solver.replay_mem)
        # TODO add option to choose what kind of replayer to use
        solver.replay_mem = PrioritizedMemory(mdp,capacity=2048) #UniformMemory(mdp, mem_size=2048) #
    end



    # get all actions: this is for my/computational convenience
    As = POMDPs.iterator(actions(mdp))

    # complete setup for neural network if necessary (This is most often the case)
    if !solver.nn.valid
        warn("You didn't specify a neural network or your number of output units didn't match the number of actions. Either way, not recommended")
        solver.nn.arch = mx.FullyConnected(mx.SymbolicNode, name=:output, num_hidden=length(As), data=solver.nn.arch)
        solver.nn.valid = true
    end

    # setup policy if neccessary
    if isnull(solver.nn.exec)
        if isnull(solver.target_nn)
            solver.target_nn = initialize!(solver.nn, mdp, copy=true)
        else
            initialize!(solver.nn, mdp)
        end
    end

    # Bias to a particular action
    output_bias = get(solver.nn.exec).arg_dict[:output_bias]
    @mx.nd_as_jl rw=output_bias begin
      output_bias[:] = -1
      output_bias[1] = 0
      println("output_bias is $(output_bias)")
    end

    # set up initial_state score tables
    s0_sample_size = 100
    s_dim = size(vec(mdp,initial_state(mdp, rng)))
    s0_set = Vector{Vector{Float64}}(s0_sample_size)
    s0_weight_set = zeros(Float64, s0_sample_size)

    # CSLV SPECIFIC TODO REMOVE
    solver.stats["s0x"] = zeros(solver.num_epochs)
    solver.stats["s0y"] = zeros(solver.num_epochs)
    solver.stats["s0h"] = zeros(solver.num_epochs)

    # set up sampling distribution
    function sample_s0()
        if isnull(s0_dist)
            s0 = initial_state(mdp, rng)
            w_s0 =  1.0
        else
            if rand(rng) > 0.2
                s0 = initial_state(mdp, MixtureModel(get(s0_dist)), rng)
                s0_vec = vec(mdp, s0)
                w_s0 = 1.0 #( Distributions.pdf(MixtureModel(get(s0_dist)), s0_vec) + 0.001 )^(-1)
            else
                 s0 = initial_state(mdp, rng)
                 w_s0 = 1.0
            end
        end
        (s0,w_s0)
    end

    terminalp = false
    max_steps = solver.max_steps
    ctr = 1

    for ep = 1:solver.num_epochs

        s0, w_s0 = sample_s0()
        s = s0
        td_s0 = 1.0 # count total td from s0

        (a, q, a_idx, s_vec,) = action(solver.exp_pol, solver, mdp, s, rng, As)
        terminal = isterminal(mdp, s)

        disc = 1.0
        r_total = 0.0


        step = 1

        td_avg = 0.

        try
            while !isterminal(mdp, s) && step <= max_steps

                sp, r = generate_sr(mdp, s, a, rng)

                (ap, qp, ap_idx, sp_vec,) = action(solver.exp_pol, solver, mdp, sp, rng, As) # convenience, maybe remove ap_idx, s_vec

                # 1-step TD error just in case you care (e.g. prioritized experience replay)
                _td = r + discount(mdp) * qp - q
                td_s0 += _td

                # terminality condition for easy access later (possibly expensive fn)
                terminalp = isterminal(mdp, sp)

                # update replay memory
                push!( get(solver.replay_mem), s_vec, a_idx, r, sp_vec, terminalp, _td, weight=w_s0, rng=rng)

                td = 0
                if size( get(solver.replay_mem) ) > solver.nn.batch_size
                # only update every batch_size steps? or what?
                    refresh_target = mod(ctr, solver.target_refresh_interval) == 0
                    td = gdqn_update!( solver.nn, get(solver.target_nn), get(solver.replay_mem), refresh_target, discount(mdp), rng )

                    td_avg += td
                end

                r_total += disc*r

                disc *= discount(mdp)
                step += 1
                ctr += 1

                s = sp
                a = ap
                q = qp
                terminal = terminalp

                # possibly remove
                a_idx = ap_idx
                s_vec = sp_vec
            end
        catch ex
            if solver.capture_exception
                solver.exception = ex
                solver.backtrace = catch_backtrace()
            else
            rethrow(ex)
            end
        end

        idx = mod(ep-1, s0_sample_size) + 1
        s0_set[idx] = copy(vec(mdp,s0))
        s0_weight_set[idx] = abs(td_s0) / step #(mean td over episode)

        if !isnull(s0_dist)
            # fit initial state distribution to the new stats
            if mod(ep, s0_sample_size) == 0
                println("Refitting initial state distribution:")
                fit_em!(get(s0_dist), s0_set, s0_weight_set, verbose=false)
                for j in 1:get(s0_dist).n
                    println("   $(j):\tα: $(get(s0_dist).α[j])")
                    println("    \tμ: $(get(s0_dist).μ[j])")
                    println("    \tΣ: $(get(s0_dist).Σ[j].mat)")
                end
            end
        end

        # update metrics
        solver.stats["td"][ep] = td_avg
        solver.stats["r_total"][ep] = r_total

        #CSLV SPECIFIC (TODO REMOVE)
        solver.stats["s0x"][ep] = s0_set[idx][1]
        solver.stats["s0y"][ep] = s0_set[idx][2]
        solver.stats["s0h"][ep] = s0_set[idx][3]

        # checkpoint stuff
        if mod(ep, solver.checkpoint_interval) == 0
            policy.exec = get(solver.nn.exec)

            # save model
            # TODO

            # print relevant metrics
            print("Epoch ", ep,
                "\n\tTD: ", mean(solver.stats["td"][ep-solver.checkpoint_interval+1:ep]),
                "\n\tTotal Reward: ", mean(solver.stats["r_total"][ep-solver.checkpoint_interval+1:ep]),"\n")

            # run learned policy for feedback
            r_total = 0
            N_sim = 100
            for i in 1:N_sim
                r_total += simulate(sim, mdp, policy, initial_state(mdp, rng))
            end
            println("\tAvg total reward: $(r_total/N_sim)")
            ep_range = ep-solver.checkpoint_interval+1:ep
            solver.stats["r_test"][ep_range] = r_total/N_sim;
            if !isnull(s0_dist)
                logGDQNstats(solver, get(s0_dist), ep_range)
            end
        end
        #return r_total

    end

    # return policy
    # TODO update policy.exec more frequently
    # TODO make new exec that doesn't need to train
    policy.exec = get(solver.nn.exec)
    return policy

end
