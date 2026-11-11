# Six animations in one, in 4K: the lid-driven cavity from rest until t = 300 L/U at six Reynolds numbers, on one
# clock, as streamlines over the vorticity. Run simulate_evolution.jl first, for Re = 5000, and simulate_comparison.jl
# for each of the other five.
#
# TIMES, a list of times such as "1,5,30", renders stills at those times instead of the video.
#
#     julia --project animate_comparison.jl

include("lid_driven_cavity.jl")
include("plotting.jl")

set_theme!(fontsize = 56, Axis = (; titlefont = :regular))

panel_title(Re) = Re < 10_000 ? "Re = $Re" : "Re = $(Re ÷ 1000),$(lpad(Re % 1000, 3, '0'))"

fig = Figure(size = (3840, 2160), figure_padding = (20, 30, 8, 12))
grid = fig[1, 1] = GridLayout()
side = fig[1, 2] = GridLayout()

clock = Observable(0.0)
Label(side[1, 1], "Lid-driven cavity"; fontsize = 92)
Label(side[2, 1], @lift(@sprintf("t = %.1f L/U", $clock)); fontsize = 92)

ticks = ([0, 0.5, 1], [L"0", L"L/2", L"L"])

panels = map(enumerate(comparison_runs)) do (n, (; Re))
    row, column = fldmod1(n, 3)

    ax = Axis(grid[row, column];
            title = panel_title(Re),
            titlesize = 76,
            titlegap = 10,
            xticks = ticks,
            yticks = ticks,
            xticklabelsize = 60,
            yticklabelsize = 60,
            xticklabelsvisible = row == 2,
            yticklabelsvisible = column == 1,
            yticklabelspace = column == 1 ? 110.0 : 0.0,
            width = 930,
            height = 930,
            limits = ((0, 1), (0, 1))
    )

    ω = saved_series(run_directory(Re), "vorticity", "ω")
    u = saved_series(run_directory(Re), "velocity", "u")
    frame = Observable(1)

    # The corner eddies at Re = 100 are too weak for the levels near zero, so that panel has one more.
    levels = Re == 100 ? sort([streamline_levels; 2e-7]) : streamline_levels

    vorticity_heatmap = vorticity_panel!(ax, ω, u, frame; levels, linewidth = 2, interpolate = true)

    (; frame, times = ω.times, vorticity_heatmap)
end

Colorbar(side[3, 1], first(panels).vorticity_heatmap;
        label = L"\omega L / U",
        labelsize = 88,
        ticklabelsize = 68,
        ticks = vorticity_ticks([-30, -10, -3, -1, 0, 1, 3, 10, 30]),
        width = 80,
        height = 1500,
        alignmode = Outside()
)

rowgap!(side, 1, 0)
rowgap!(side, 2, 50)
colgap!(grid, 40)
rowgap!(grid, 4)
colgap!(fig.layout, 30)

"""
    show_time!(t)

Show every panel's output nearest to the time `t`. A run that stopped once its flow was steady stays on its last output.
"""
function show_time!(t)
    clock[] = t

    for panel in panels
        n = nearest(panel.times, t)
        panel.frame[] == n || (panel.frame[] = n)
    end

    return nothing
end

# The video speeds up after t = 20 L/U, to at most 0.25 L/U per frame. Beyond that, the vortices in the chaotic panels
# would jump from frame to frame.
times = video_times(300; pace = 0.05, constant_until = 20, maximum_pace = 0.25)

if isempty(still_times)
    save_animation(show_time!, fig, times, "lid_driven_cavity_comparison")
else
    save_stills(show_time!, fig, still_times, "lid_driven_cavity_comparison")
end
