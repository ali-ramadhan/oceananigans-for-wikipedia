# The two-dimensional Taylor–Green vortex at four Reynolds numbers, Re = 10, 100, 1000 and 10,000, animated
# side by side: the vorticity with streamlines and fluid particles. Each simulation starts from the vortex plus
# the same small random perturbation, 0.1% of its root-mean-square speed, like the disturbances any real flow
# has. At low Re the vortices decay before the perturbation can grow. At high Re the array of vortices is
# unstable: the perturbation grows until the vortices break up into two-dimensional turbulence.
#
#     julia --project taylor_green_2d_reynolds.jl

using JLD2

include("taylor_green.jl")
include("video_encoding.jl")

# The grid is refined with Re to resolve the thinner vorticity filaments.
runs = [(Re = 10, N = 256, label = "10"),
        (Re = 100, N = 256, label = "100"),
        (Re = 1000, N = 512, label = "1000"),
        (Re = 10_000, N = 1024, label = "10,000")]
stop_time = 200
frame_interval = 0.25
perturbation = 1e-3

output_directory = joinpath(@__DIR__, "output")
mkpath(output_directory)

all_frames = map(runs) do (; Re, N)
    simulation = taylor_green_simulation(Re; N, stop_time, perturbation)
    frames = record_frames!(simulation, frame_interval; resolution = 512, T = Float32)
    run!(simulation)
    jldsave(joinpath(output_directory, "taylor_green_2d_Re$Re.jld2"); frames...)
    frames
end

set_theme!(Theme(fontsize = 22))
times = all_frames[1].t

fig = Figure()
frame = Observable(1)
Label(fig[0, 1:3], @lift(@sprintf("2D Taylor–Green vortex:   t = %5.1f", times[$frame])),
      fontsize = 28, tellwidth = false)

ticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"])
heatmaps = map(enumerate(zip(runs, all_frames))) do (j, ((; label), frames))
    row, column = fldmod1(j, 2)
    ax = Axis(fig[row, column]; title = "Re = $label", xlabel = "x", ylabel = "y", width = 640, height = 640,
              limits = ((0, L), (0, L)), xticks = ticks, yticks = ticks)
    row == 1 && hidexdecorations!(ax; ticks = false)
    column == 2 && hideydecorations!(ax; ticks = false)
    vorticity_panel!(ax, frames, frame; scale = 640 / 840)  # the single-panel animation's axis is 840 pixels
end
Colorbar(fig[1:2, 3], heatmaps[1]; label = "ω / kU₀")
resize_to_layout!(fig)
w, h = size(fig.scene)
resize!(fig, w + isodd(w), h + isodd(h))  # the video encoder needs even dimensions

master = record_lossless(fig, joinpath(output_directory, "taylor_green_2d_reynolds_master.mkv"), eachindex(times)) do i
    frame[] = i
end
encode_webm(master, joinpath(output_directory, "taylor_green_2d_reynolds.webm"))

frame[] = findfirst(≥(100), times)
save(joinpath(output_directory, "taylor_green_2d_reynolds.png"), fig; px_per_unit = 2)
