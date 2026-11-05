# The two-dimensional Taylor–Green vortex at Re = 10, 100, 1000 and 10,000, animated side by side. Each simulation
# starts from the vortex plus the same random perturbation, 0.1% of its root-mean-square speed. At low Re the vortices
# decay before the perturbation can grow. At high Re the perturbation grows until the vortices break up into turbulence.
#
#     julia --project taylor_green_2d_reynolds.jl

include("taylor_green.jl")

# The grid is refined with Re to resolve the thinner vorticity filaments.
runs = [
        (Re = 10, N = 256, label = "10"),
        (Re = 100, N = 256, label = "100"),
        (Re = 1000, N = 512, label = "1000"),
        (Re = 10_000, N = 1024, label = "10,000")
]

run_output(Re) = output_file("taylor_green_2d_Re$Re.jld2")
velocity_series(Re, component) = FieldTimeSeries(run_output(Re), component; backend = OnDisk())

for (; Re, N) in runs
    simulation = taylor_green_simulation(;
            Re,
            N,
            stop_time = 200,
            perturbation = 1e-3
    )

    model = simulation.model
    u, v, _ = model.velocities

    simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, model.particles);
            filename = run_output(Re),
            schedule = TimeInterval(0.25),
            overwrite_files = true
    )

    run!(simulation)
end

times = velocity_series(first(runs).Re, "u").times

fig = Figure()
frame = Observable(1)
animation_title = @lift @sprintf("2D Taylor–Green vortex:   t = %5.1f", times[$frame])
Label(fig[0, 1:3], animation_title)

heatmaps = map(enumerate(runs)) do (j, (; Re, label))
    ut = velocity_series(Re, "u")
    vt = velocity_series(Re, "v")

    row, column = fldmod1(j, 2)
    ax = domain_axis(fig[row, column], ut.grid; size = 640, title = "Re = $label")
    row == 1 && hidexdecorations!(ax; ticks = false)
    column == 2 && hideydecorations!(ax; ticks = false)

    vorticity_panel!(ax, ut, vt, particle_trajectories(run_output(Re)), frame)
end
Colorbar(fig[1:2, 3], first(heatmaps); label = "ω / kU₀")

resize_to_layout!(fig)
save_animation(fig, frame, times, "taylor_green_2d_reynolds"; still_time = 100)
