# Simulates the Taylor–Green vortex at Re = 1600 in three dimensions on 512³ cells and in two dimensions on 512² cells
# until t = 60 L/V₀, and saves everything that the animation scripts read, every 0.05 L/V₀.
#
# RESOLUTION (default 512), STOP_TIME (default 60) and OUTPUT_DIRECTORY (default output) can be set for quick tests.
# VELOCITY_STOP_TIME (default 20) is when the velocity output stops; 0 saves no velocity at all.
#
#     julia --project simulate.jl

include("taylor_green.jl")

Re = 1600
N = parse(Int, get(ENV, "RESOLUTION", "512"))
stop_time = parse(Float64, get(ENV, "STOP_TIME", "60"))
velocity_stop_time = min(stop_time, parse(Float64, get(ENV, "VELOCITY_STOP_TIME", "20")))

function save_output!(simulation, name, outputs, filename; writer_options...)
    simulation.output_writers[name] = JLD2Writer(simulation.model, outputs;
            schedule = TimeInterval(0.05),
            filename = output_file(filename),
            overwrite_files = true,
            writer_options...
    )

    return nothing
end

function progress(sim)
    @info @sprintf("t = %5.2f, iteration %6d, Δt = %.2e, wall time %s",
            time(sim),
            iteration(sim),
            sim.Δt,
            prettytime(sim.run_wall_time)
    )

    return nothing
end

mean_square(field) = Field(Average(field^2))

#####
##### Three dimensions
#####

simulation = taylor_green_simulation(; Re, N, stop_time)
u, v, w = simulation.model.velocities
ωx, ωy, ωz = vorticity(u, v, w)

# The plane z = πL/4. Unlike the plane z = 0, which the flow's symmetry keeps calm, it cuts through the turbulence.
slice(ω) = Field(@at((Center, Center, Face), ω); indices = (:, :, N ÷ 2 + N ÷ 8 + 1))
slices = (ωx = slice(ωx), ωy = slice(ωy), ωz = slice(ωz))
save_output!(simulation, :slice, slices, "taylor_green_3d_slice.jld2")

averages = (
        u² = mean_square(u),
        v² = mean_square(v),
        w² = mean_square(w),
        ωx² = mean_square(ωx),
        ωy² = mean_square(ωy),
        ωz² = mean_square(ωz)
)

save_output!(simulation, :averages, averages, "taylor_green_3d_averages.jld2")

add_callback!(simulation, progress, TimeInterval(1))

# The simulation runs until velocity_stop_time with the velocity writer, then continues without it.
if velocity_stop_time > 0
    save_output!(simulation, :velocities, (; u, v, w), "taylor_green_3d.jld2";
            file_splitting = TimeInterval(1),
            compress = false
    )

    simulation.stop_time = velocity_stop_time
    run!(simulation)

    delete!(simulation.output_writers, :velocities)
    simulation.stop_time = stop_time
end

run!(simulation)

#####
##### Two dimensions
#####

simulation = taylor_green_simulation(; Re, N, dimensions = 2, stop_time)
u, v, _ = simulation.model.velocities
ωz = ∂x(v) - ∂y(u)

outputs = (ωz = Field(@at((Center, Center, Center), ωz)), ωz² = mean_square(ωz))
save_output!(simulation, :vorticity, outputs, "taylor_green_2d.jld2")

run!(simulation)
