# Simulates one run of animate_comparison.jl: the lid-driven cavity at the Reynolds number RE, from rest until
# t = 300 L/U, or until the flow is steady if that comes first. It saves the same output as simulate_evolution.jl, whose
# run at Re = 5000 is one of the six panels. Run it once for each of the other five, from the lowest Reynolds number to
# the highest:
#
#     RE=100 julia --startup-file=no --project simulate_comparison.jl
#
# RESOLUTION (default the grid in `comparison_runs`), STOP_TIME (default 300) and OUTPUT_DIRECTORY (default output) can
# be set for quick tests.

include("lid_driven_cavity.jl")

Re = parse(Int, ENV["RE"])
Re == 5000 && error("simulate_evolution.jl makes the run at Re = 5000")

grids = Dict(Re => N for (; Re, N) in comparison_runs)
N = parse(Int, get(ENV, "RESOLUTION", string(grids[Re])))
stop_time = parse(Float64, get(ENV, "STOP_TIME", "300"))
directory = mkpath(run_directory(Re))

# Every 0.025 L/U while the flow forms and every 0.1 L/U after that. That is at least twice as often as the video shows,
# so its pace can change without a new run.
output_times = filter(≤(stop_time), vcat(0.025:0.025:50, 50.1:0.1:300))

simulation = lid_driven_cavity_simulation(; Re, N, stop_time)

save_evolution!(simulation, output_times; directory)
add_callback!(simulation, progress, TimeInterval(10))
stop_when_steady!(simulation; Re, check_interval = 10, directory)

run!(simulation)
