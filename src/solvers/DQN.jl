# DQN.jl
# built off @zsunberg's HistoryRecorder.jl
# making stuff up as I'm going along
# uses MxNet as backend because native julia etc etc
# TODO modify solve signature (add policy=create_policy)


type DQN <: POMDPs.Solver
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

    # stuff for if we have an estimate of the true q function
    q_hat::Nullable{Function} # should take state, output vector of q values of each action
    q_hat_bias::Real
    pretrain_batch_size::Int
    pretrain_num_batches::Int

    # exception stuff from History Recorder -- couldn't hurt
    capture_exception::Bool
    exception::Nullable{Exception}
    backtrace::Nullable{Any}
end
function DQN(;
            nn::NeuralNetwork=build_partial_mlp(ctx=mx.cpu(),hidden_sizes=[128,64,32]),
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
            q_hat::Nullable{Function}=Nullable{Function}(),
            q_hat_bias::Real=1.,
            pretrain_batch_size::Int=128,
            pretrain_num_batches::Int=1000,
            capture_exception::Bool=false,
            target_refresh_interval::Int=10000
            )


    # TODO check stuff or something--leave replay memory null?
    return DQN(
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
                q_hat,
                q_hat_bias,
                pretrain_batch_size,
                pretrain_num_batches,
                capture_exception,
                nothing,
                nothing
                )

end

type DQNPolicy{S,A} <: POMDPs.Policy
    exec::Union{mx.Executor,Void}
    input_name::Symbol
    q_values::Vector{Float32} # julia side output - for memory efficiency
    actions::Vector{A}
    mdp::MDP{S,A}
end

function create_policy(sol::DQN, mdp::MDP)
    A = iterator(actions(mdp))
    return DQNPolicy(
                    isnull(sol.nn.exec) ? nothing : sol.nn.exec,
                    sol.nn.input_name,
                    zeros(Float32, length(A)),
                    A,
                    mdp
                    )
end
# TODO constructor


# TODO make this more standard
# function scale(s_vec)
#     return s_vec./[10000,10000,100,1,1]
# end

function util{S,A}(p::DQNPolicy{S,A}, s::S)
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

function POMDPs.action{S,A}(p::DQNPolicy{S,A}, s::S)
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



function action{S,A}(p::EpsilonGreedy, solver::DQN, mdp::MDP{S,A}, s::S, rng::AbstractRNG, As_all::Vector{A}, a::A=As_all[1])

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

function referenceQLoss(q, q_ref)
  _, correct_i = findmax(q_ref)
  n_actions = length(q_ref)
  lossGrad = q - reshape(q_ref, size(q))
  # bias toward correct relative actions
  imbalance_factor = 5.
  biasedLossGrad = max(lossGrad, imbalance_factor*lossGrad) # bigger loss if overestimating q for incorrect action
  biasedLossGrad[correct_i] = min(lossGrad[correct_i], (n_actions - 1)*imbalance_factor*lossGrad[correct_i]) # bigger loss for underestimating correct q

  loss = sum(biasedLossGrad.^2)
  (loss, biasedLossGrad)
end

function pretrain( nn::NeuralNetwork, mdp, q_hat, rng, num_batches, batch_size)
  N = n_states(mdp)
  println("N: $N. num_batches: $num_batches, batch_size: $batch_size")
  batch_size = min(N, batch_size)
  ss = ordered_states(mdp)
  sequence = randperm(N)
  batches_per_epoch = Int64(floor(N/batch_size)+1)
  n_epochs = Int64(floor((num_batches-1)/batches_per_epoch) + 1)
  steps_per_epoch = min(N, num_batches*batch_size)
  checkpoint_interval = 25
  n_checkpoints = Int64(floor((steps_per_epoch-1)/batch_size/checkpoint_interval + 1)*n_epochs)
  println("Starting pretraining, using $(n_epochs) epoch(s), with $(steps_per_epoch) steps per epoch, and a batch size of $(batch_size). There will be $(n_checkpoints) checkpoints.")

  loss_hist = zeros(n_checkpoints)
  k = 1
  for i = 1:n_epochs
    l2_error = 0.
    for j = 1:steps_per_epoch
      s_vec = vec(mdp, ss[sequence[j]])
      # setup target, do forward, backward pass to get gradient
      mx.copy!( get(nn.exec).arg_dict[nn.input_name], s_vec )
      mx.forward( get(nn.exec), is_train=true )
      qs = copy!( zeros(Float32, size(get(nn.exec).outputs[1])), get(nn.exec).outputs[1])
      q_ref = q_hat(s_vec)

      loss, grad = referenceQLoss(qs, q_ref)
      l2_error += loss

      mx.backward( get(nn.exec), copy(grad, nn.ctx) )

      if mod(j-1, batch_size) == 0
        batch_i = floor(j/batch_size)
        if mod(batch_i, checkpoint_interval) == 0
          println(" epoch $(i), batch $(batch_i):\tavg loss: $(l2_error)")
          loss_hist[k] = l2_error
          k += 1
          l2_error = 0.
        end
        update!(nn)
      end
    end
  end

  writedlm(open("pretraining_loss.txt", "w"), loss_hist)
end

function dqn_update!( nn::NeuralNetwork, target_nn::mx.Executor, mem::ReplayMemory, refresh_target::Bool, disc::Float64, q_hat, q_hat_bias, rng::AbstractRNG )

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

        td_avg += (qp - qs[a_idx])^2

        # compute loss gradient
        # grad_vec = zeros(qs)
        # grad_vec[a_idx] = qs[a_idx]-qp
        # lossGrad = copy(grad_vec, mx.cpu())

        qp_vec = copy(qs)
        qp_vec[a_idx] = qp


        # compute weighted loss gradient
        lossGrad = weight*(qs - qp_vec)

        if !isnull(q_hat)
          # add weighted regression loss against q_hat
          s_vec = state(mem, s_idx)
          @mx.nd_as_jl rw=s_vec begin
            qhat_vec = get(q_hat)(squeeze(s_vec,2))
            _, grad = referenceQLoss(qs, qhat_vec)
            lossGrad += q_hat_bias*grad
          end
        end

        lossGradArray = copy(lossGrad, nn.ctx)
        mx.backward( get(nn.exec), lossGradArray )
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

function solve{S,A}(solver::DQN, mdp::MDP{S,A}, policy::DQNPolicy=create_policy(solver, mdp), rng::AbstractRNG=RandomDevice())

    sim = RolloutSimulator(max_steps=solver.max_steps)

    # setup experience replay; initialized here because of the whole solve paradigm (decouple solver, problem)
    if isnull(solver.replay_mem)
        # TODO add option to choose what kind of replayer to use
        solver.replay_mem = PrioritizedMemory(mdp,capacity=2048) #UniformMemory(mdp, mem_size=100000) #
    end

    # get all actions: this is for my/computational convenience
    As = POMDPs.iterator(actions(mdp))

    # complete setup for neural network if necessary (This is most often the case)
    if !solver.nn.valid
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

    # output_bias = get(solver.nn.exec).arg_dict[:output_bias]
    # @mx.nd_as_jl rw=output_bias begin
    #   output_bias[:] = -1
    #   output_bias[1] = 0
    #   println("output_bias is $(output_bias)")
    # end

    # pretraining
    if !isnull(solver.q_hat)
      println("Pretraining neural network")
      pretrain(solver.nn, mdp, get(solver.q_hat), rng, solver.pretrain_num_batches, solver.pretrain_batch_size)
      for (param, param_target) in zip( get(solver.nn.exec).arg_arrays, get(solver.target_nn).arg_arrays )
          mx.copy!(param_target, param)
      end

      policy.exec = get(solver.nn.exec)

      # run learned policy for feedback
      r_total = 0
      N_sim = 100
      for i in 1:N_sim
          r_total += simulate(sim, mdp, policy, initial_state(mdp, rng))
      end
      println("\tAvg total reward: $(r_total/N_sim)")
    end

    terminalp = false
    max_steps = solver.max_steps
    ctr = 1

    for ep = 1:solver.num_epochs

        s = initial_state(mdp, rng)

        (a, q, a_idx, s_vec,) = action(solver.exp_pol, solver, mdp, s, rng, As)
        terminal = isterminal(mdp, s)

        disc = 1.0 # what discount to use for the total reward statistics
        r_total = 0.0


        step = 1

        td_avg = 0.

        try
            while !isterminal(mdp, s) && step <= max_steps

                sp, r = generate_sr(mdp, s, a, rng)

                (ap, qp, ap_idx, sp_vec,) = action(solver.exp_pol, solver, mdp, sp, rng, As) # convenience, maybe remove ap_idx, s_vec

                # 1-step TD error just in case you care (e.g. prioritized experience replay)
                _td = r + discount(mdp) * qp - q

                # terminality condition for easy access later (possibly expensive fn)
                terminalp = isterminal(mdp, sp)

                # update replay memory
                push!( get(solver.replay_mem), s_vec, a_idx, r, sp_vec, terminalp, _td, rng=rng)

                td = 0
                if size( get(solver.replay_mem) ) > solver.nn.batch_size
                # only update every batch_size steps? or what?
                    refresh_target = mod(ctr, solver.target_refresh_interval) == 0
                    td = dqn_update!( solver.nn, get(solver.target_nn), get(solver.replay_mem), refresh_target, discount(mdp), solver.q_hat, solver.q_hat_bias, rng )

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


        # update metrics
        solver.stats["td"][ep] = td_avg
        solver.stats["r_total"][ep] = r_total

        # print metrics
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
            solver.stats["r_test"][ep-solver.checkpoint_interval+1:ep] = r_total/N_sim;
        end

    end

    policy.exec = get(solver.nn.exec)
    return policy

end
