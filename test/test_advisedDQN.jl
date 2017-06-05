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

gw = GridWorld(rv = [-1, -0.5, 1, 0.3])#GridWorld(sx=20,sy=1,rs=[GridWorldState(3,1)],rv=[5.],penalty=-10.0, tp=1.0)
rs_small=[GridWorldState(2,1), GridWorldState(2,2), GridWorldState(3,1)]
rs_big=collect(GridWorldState(x,y) for (x,y) in zip([3,4,6], [4,1,1]))
small_gw = GridWorld(sx=3,sy=3,
                     rs=rs_small,
                     rv=[-1.,-1.,1.], tp=1.0,
                     terminals=Set{GridWorldState}(rs_small))
big_gw = GridWorld(sx=6,sy=6,
                   rs=rs_big,
                   rv=[-1.,-1.,1.], tp=1.0,
                   terminals=Set{GridWorldState}(rs_big))

sim = RolloutSimulator(max_steps=100)

println("Solving small problem with DVI")
dvi = ValueIterationSolver()
dvipol = solve(dvi,small_gw,verbose=true)

println("Solving big problem with DVI")
dvi2 = ValueIterationSolver()
dvibigpol = solve(dvi2,big_gw,verbose=true)

function vec(s::GridWorldState)
    if s.done
        return [0,0]
    else
        return POMDPs.vec(small_gw, s)
    end
end

function unvec(mdp::GridWorld, svec::Vector)
    x = clamp(round(svec[1]*mdp.size_x),1,mdp.size_x+1)
    y = clamp(round(svec[2]*mdp.size_y),1,mdp.size_y+1)
    done = (svec[1] == 0 && svec[2] == 0)
    GridWorldState(x, y, done)
end


function qhat(s_vec)
    s = unvec(small_gw, s_vec)
    s_idx = state_index(small_gw, s)
    q_vec = dvipol.qmat[s_idx,:]
end

#DQN
println("Solving full problem with DQN, seeded with DVI Policy for discretized problem")
dqn = rl.DQN(max_steps=50, checkpoint_interval=25, num_epochs=0, target_refresh_interval=25, q_hat=Nullable{Function}(qhat), q_hat_bias=0.)
dqnpol = rl.solve(dqn, big_gw)
for s in iterator(states(big_gw))
    a = action(dqnpol, s)
    println("state $(s.x), action $(a)")
end
r_total = 0
N_sim = 50
for i in 1:N_sim
    r_total += simulate(sim, big_gw, dqnpol, initial_state(big_gw, RandomDevice()))
end
println("Avg total reward DQN on big gridworld: $(r_total/N_sim)")

df = DataFrame(dqn.stats);
writetable("testDQNadvised.csv", df);

r_total = 0
N_sim = 50
for i in 1:N_sim
    r_total += simulate(sim, small_gw, dvipol, initial_state(small_gw, RandomDevice()))
end
println("Avg total reward DVI on small $(r_total/N_sim)")


r_total = 0
N_sim = 50
for i in 1:N_sim
    r_total += simulate(sim, big_gw, dvibigpol, initial_state(small_gw, RandomDevice()))
end
println("Avg total reward DVI on big $(r_total/N_sim)")


for s in iterator(states(big_gw))
    a = action(dvipol, unvec(small_gw, vec(big_gw, s)))
    a1 = action(dqnpol, s)

    if isterminal(gw, s)
        continue
    end

    if a1 != a
        println("actions differ at state $(s), dqn $(a1) vs dvi $(a)")
    end
end
