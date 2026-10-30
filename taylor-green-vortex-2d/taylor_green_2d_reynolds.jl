# The two-dimensional Taylor–Green vortex at four Reynolds numbers, Re = 10, 100, 1000 and 10,000, animated side by
# side: the vorticity with streamlines and fluid particles. Each simulation starts from the vortex plus the same small
# random perturbation, 0.1% of its root-mean-square speed, like the disturbances any real flow has. At low Re the
# vortices decay before the perturbation can grow. At high Re the array of vortices is unstable: the perturbation
# grows until the vortices break up into two-dimensional turbulence.
#
#     julia --project taylor_green_2d_reynolds.jl

include("taylor_green.jl")
include("video_encoding.jl")

U₀ = 1
k = 1
stop_time = 200
perturbation = 1e-3
output_directory = joinpath(@__DIR__, "output")
mkpath(output_directory)

# The grid is refined with Re to resolve the thinner vorticity filaments.
runs = [(Re = 10, N = 256, label = "10"),
        (Re = 100, N = 256, label = "100"),
        (Re = 1000, N = 512, label = "1000"),
        (Re = 10_000, N = 1024, label = "10,000")]

filenames = map(runs) do (; Re, N)
    simulation = taylor_green_simulation(; Re, N, stop_time, U₀, k, perturbation)
    model = simulation.model
    u, v, w = model.velocities
    filename = joinpath(output_directory, "taylor_green_2d_Re$Re.jld2")
    simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, model.particles); filename,
                                                     schedule = TimeInterval(0.25), overwrite_files = true)
    run!(simulation)
    filename
end

set_theme!(Theme(fontsize = 22))
times = FieldTimeSeries(filenames[1], "u"; backend = OnDisk()).times

fig = Figure()
frame = Observable(1)
Label(fig[0, 1:3], @lift(@sprintf("2D Taylor–Green vortex:   t = %5.1f", times[$frame])),
      fontsize = 28, tellwidth = false)

ticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"])
heatmaps = map(enumerate(zip(runs, filenames))) do (j, ((; label), filename))
    row, column = fldmod1(j, 2)
    ax = Axis(fig[row, column]; title = "Re = $label", xlabel = "x", ylabel = "y", width = 640, height = 640,
              limits = ((0, 2π / k), (0, 2π / k)), xticks = ticks, yticks = ticks)
    row == 1 && hidexdecorations!(ax; ticks = false)
    column == 2 && hideydecorations!(ax; ticks = false)
    ut = FieldTimeSeries(filename, "u"; backend = OnDisk())
    vt = FieldTimeSeries(filename, "v"; backend = OnDisk())
    # The single-panel animation's axis is 840 pixels wide.
    vorticity_panel!(ax, ut, vt, particle_positions(filename), frame; U₀, k, scale = 640 / 840)
end
Colorbar(fig[1:2, 3], heatmaps[1]; label = "ω / kU₀")
resize_to_layout!(fig)
# The video encoder needs even dimensions.
figure_width, figure_height = size(fig.scene)
resize!(fig, figure_width + isodd(figure_width), figure_height + isodd(figure_height))

master = record_lossless(fig, joinpath(output_directory, "taylor_green_2d_reynolds_master.mkv"), eachindex(times)) do i
    frame[] = i
end
encode_webm(master, joinpath(output_directory, "taylor_green_2d_reynolds.webm"))

frame[] = findfirst(≥(100), times)
save(joinpath(output_directory, "taylor_green_2d_reynolds.png"), fig; px_per_unit = 2)
