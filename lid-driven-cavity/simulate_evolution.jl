# Simulates the lid-driven cavity at Re = 5000 on 1024² cells, from rest until the flow is steady, for
# animate_evolution.jl and animate_comparison.jl. Saves the vorticity, the velocity and the mean squares of the velocity
# and vorticity, at times that are closely spaced while the flow forms and further apart as it settles, and the final
# state, for profiles.jl and check_steady.jl.
#
# RESOLUTION (default 1024) and OUTPUT_DIRECTORY (default output) can be set for quick tests.
#
# Skipping the startup file keeps Revise out of the run, which would otherwise patch it if packages are updated
# meanwhile.
#
#     julia --startup-file=no --project simulate_evolution.jl

include("lid_driven_cavity.jl")

Re = 5000
N = parse(Int, get(ENV, "RESOLUTION", "1024"))
maximum_time = 3000

"""
    output_times(; spacing = 0.025, constant_until = 40, growth = 1 / 200)

The times to save the output: `spacing` apart until `constant_until`, and then further apart in proportion to the time
since then. `animate_evolution.jl` shows some of these, so it can change the pace of the video without a new run.
"""
function output_times(; spacing = 0.025, constant_until = 40, growth = 1 / 200)
    times = Float64[]
    t = 0.0

    while t < maximum_time
        t += spacing + growth * max(0, t - constant_until)
        push!(times, t)
    end

    return times
end

simulation = lid_driven_cavity_simulation(; Re, N, stop_time = maximum_time)

save_evolution!(simulation, output_times())
add_callback!(simulation, progress, TimeInterval(10))
stop_when_steady!(simulation; Re, check_interval = 50)

run!(simulation)
