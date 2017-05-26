push!(LOAD_PATH, ".")
include(joinpath("..", "src", "DRL.jl"))
using DRL
importall POMDPModels
using MXNet
using DiscreteValueIteration
using POMDPToolbox
using DataFrames

# stuff to make things work
importall POMDPs

gw = GridWorld()#GridWorld(sx=20,sy=1,rs=[GridWorldState(3,1)],rv=[5.],penalty=-10.0, tp=1.0)
sim = RolloutSimulator(max_steps=100)

println("Solving with DVI")
dvi = ValueIterationSolver()
dvipol = solve(dvi,gw,verbose=true)

function vec(s::GridWorldState)
    if s.done
        return [0,0]
    else
        return POMDPs.vec(gw, s)
    end
end

function unvec(mdp::GridWorld, svec::Vector)
    x = ceil(svec[1]*mdp.size_x)
    y = ceil(svec[1]*mdp.size_y)
    done = (x == 0 && y == 0)
    GridWorldState(x, y, done)
end


function qhat(s_vec)
    s = unvec(gw, s_vec)
    s_idx = state_index(gw, s)
    q_vec = dvipol.qmat[s_idx,:]
end

#DQN
println("Solving with DQN, seeded with DVI Policy")
dqn = rl.DQN(max_steps=50, checkpoint_interval=25, num_epochs=0, target_refresh_interval=100, q_hat=Nullable{Function}(qhat), q_hat_bias=0.05)
dqnpol = rl.solve(dqn, gw)
for s in iterator(states(gw))
    a = action(dqnpol, s)
    println("state $(s.x), action $(a)")
end
r_total = 0
N_sim = 50
for i in 1:N_sim
    r_total += simulate(sim, gw, dqnpol, initial_state(gw, RandomDevice()))
end
println("Avg total reward $(r_total/N_sim)")

df = DataFrame(dqn.stats);
writetable("testDQNadvised.csv", df);

r_total = 0
N_sim = 50
for i in 1:N_sim
    r_total += simulate(sim, gw, dvipol, initial_state(gw, RandomDevice()))
end
println("Avg total reward $(r_total/N_sim)")


for s in iterator(states(gw))
    a = action(dvipol, s)
    a1 = action(dqnpol, s)

    if isterminal(gw, s)
        continue
    end

    if a1 != a
        println("actions differ at state $(s), dqn $(a1) vs dvi $(a)")
    end
end
