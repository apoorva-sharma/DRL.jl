include(joinpath("..", "src", "DRL.jl"))
using DRL
importall POMDPModels
using GraphViz
using MXNet
using DiscreteValueIteration
using POMDPToolbox

# stuff to make things work
importall POMDPs
iterator(ipa::POMDPModels.InvertedPendulumActions) = ipa.actions

ip = InvertedPendulum()
gw = GridWorld(sx=20,sy=1,rs=[GridWorldState(3,1)],rv=[5.],penalty=-10.0, tp=1.0)
mc = MountainCar()

iterator(mca::POMDPModels.MountainCarActions) = mca.actions


sim = RolloutSimulator(max_steps=100)

#DQN
println("Testing with DQN")
dqn = rl.DQN(max_steps=50, checkpoint_interval=25, num_epochs=500, target_refresh_interval=250)
dqnpol = rl.solve(dqn, ip)
# for s in iterator(states(gw))
#     a = action(pol, s)
#     println("state $(s.x), action $(a)")
# end
r_total = 0
N_sim = 50
for i in 1:N_sim
    r_total += simulate(sim, ip, dqnpol, initial_state(ip, RandomDevice()))
end
println("Avg total reward $(r_total/N_sim)")

# println("Testing with DVI")
# dvi = ValueIterationSolver()
# dvipol = solve(dvi,gw,verbose=true)
# # for s in iterator(states(gw))
# #     a = action(pol, s)
# #     println("state $(s.x), action $(a)")
# # end

# r_total = 0
# N_sim = 50
# for i in 1:N_sim
#     r_total += simulate(sim, gw, dvipol, initial_state(gw, RandomDevice()))
# end
# println("Avg total reward $(r_total/N_sim)")


# for s in iterator(states(gw))
#     a = action(dvipol, s)
#     a1 = action(dqnpol, s)

#     if isterminal(gw, s)
#         continue
#     end

#     if a1 != a
#         println("actions differ at state $(s), dqn $(a1) vs dvi $(a)")
#     end
# end