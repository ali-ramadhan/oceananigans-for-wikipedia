# The lid-driven cavity at Re = 5000 from rest until the flow is steady: streamlines over the vorticity. Run
# simulate_evolution.jl first.
#
# TIMES, a list of times such as "1,5,30", renders stills at those times instead of the video.
#
#     julia --project animate_evolution.jl

include("lid_driven_cavity.jl")
include("plotting.jl")

set_theme!(fontsize = 22)

Re = 5000

ω = saved_series(run_directory(Re), "vorticity", "ω")
u = saved_series(run_directory(Re), "velocity", "u")
times = ω.times

fig = Figure()
frame = Observable(1)

title = @lift @sprintf("Lid-driven cavity, Re = %d:   t = %.1f L/U", Re, times[$frame])
Label(fig[0, 1:2], title; fontsize = 28, tellwidth = false)

# One pixel for each of the 1024² cells.
ax = Axis(fig[1, 1];
        xlabel = L"x / L",
        ylabel = L"y / L",
        xticks = 0:0.25:1,
        yticks = 0:0.25:1,
        width = 1024,
        height = 1024,
        limits = ((0, 1), (0, 1))
)

vorticity_heatmap = vorticity_panel!(ax, ω, u, frame)

Colorbar(fig[1, 2], vorticity_heatmap;
        label = L"\omega L / U",
        ticks = vorticity_ticks([-30, -10, -3, -1, 0, 1, 3, 10, 30])
)

resize_to_layout!(fig)

function show_time!(t)
    n = nearest(times, t)
    frame[] == n || (frame[] = n)

    return nothing
end

# The video shows each output at most once. It speeds up after t = 25 L/U, and no faster than this: the center of the
# primary vortex circles its final position every 7.5 L/U or so until about t = 120 L/U, and a faster speed-up would
# turn that into a shake.
shown = unique(nearest(times, t) for t in video_times(last(times); pace = 0.025, constant_until = 25))

if isempty(still_times)
    save_animation(show_time!, fig, times[shown], "lid_driven_cavity_evolution"; px_per_unit = 2)
else
    save_stills(show_time!, fig, still_times, "lid_driven_cavity_evolution")
end
