# Runs the lid-driven cavity at Re = 7500 on 512² cells until the flow is steady, for schematic.jl, and saves the
# velocity. The flow is steady below the Hopf bifurcation near Re = 8000, but it settles slowly so close to it.
#
# RE (default 7500), RESOLUTION (default 512) and OUTPUT_DIRECTORY (default output) can be set.
#
#     julia --startup-file=no --project simulate_steady.jl

include("lid_driven_cavity.jl")

Re = parse(Int, get(ENV, "RE", "7500"))
N = parse(Int, get(ENV, "RESOLUTION", "512"))
maximum_time = 3000

# The schematic was drawn from second-order centered advection. Only the final state matters, not the path to it, so
# the time step is as long as stability allows.
simulation = lid_driven_cavity_simulation(; Re, N, stop_time = maximum_time, advection = Centered(order = 2), cfl = 0.9)

stop_when_steady!(simulation; Re, check_interval = 50)

run!(simulation)
