# Draws `Lid-driven cavity velocity and vorticity profiles.png`: heatmaps of u, v and the vorticity in the steady flow
# at Re = 100, 1000 and 5000, with translucent lines where the velocity profiles beside them are taken, and the
# vorticity along the lid, against the tables of Ghia, Ghia and Shin (1982), Botella and Peyret (1998) and Erturk, Corke
# and Gökçöl (2005). It also prints how far apart they are. It reads the final states of simulate_comparison.jl
# (Re = 100 and 1000) and simulate_evolution.jl (Re = 5000), so run those first.
#
# FIGURE_DIRECTORY (default this folder) sets where the figure goes.
#
#     julia --project profiles.jl

include("lid_driven_cavity.jl")
include("plotting.jl")
include("published_results.jl")

figure_directory = get(ENV, "FIGURE_DIRECTORY", @__DIR__)

# The three steady flows of the comparison, and their colors in the figure.
run_colors = Dict(100 => "#2a78d6", 1000 => "#eb6834", 5000 => "#1baf7a")
runs = [(; Re, N, color = run_colors[Re]) for (; Re, N) in comparison_runs if haskey(run_colors, Re)]

#####
##### The simulated flows
#####

"""
    steady_flow(Re, N)

The fields for the heatmaps and the profiles along the centerlines and the lid, from the steady state of the run at
Reynolds number `Re` on `N²` cells. For even `N`, the faces and corners where the fields live fall exactly on the
centerlines. The heatmaps don't need more than 512² values, whatever the grid.
"""
function steady_flow(Re, N)
    (; u, v) = steady_state(run_directory(Re), Re, N)
    ω = vorticity_on_corners(u, v)

    middle = N ÷ 2 + 1
    centers = (0.5:N-0.5) ./ N
    corners = (0:N) ./ N

    return (;
            N,
            u_field = at_most((u[1:end-1, :] .+ u[2:end, :]) ./ 2, 512),
            v_field = at_most((v[:, 1:end-1] .+ v[:, 2:end]) ./ 2, 512),
            ω_field = at_most(cell_average(ω), 512),
            y_centers = [0; centers; 1],
            u_profile = [0; u[middle, :]; 1],
            x_centers = [0; centers; 1],
            v_profile = [0; v[:, middle]; 0],
            corners,
            ω_vertical = ω[middle, :],
            ω_horizontal = ω[:, middle],
            lid_x = corners[2:N],
            ω_lid = ω[2:N, N+1]
    )
end

flows = Dict(Re => steady_flow(Re, N) for (; Re, N) in runs)

#####
##### How far apart they are
#####

"""
    interpolated(nodes, values, at)

The piecewise-linear interpolant through `values` at the increasing `nodes`, evaluated at each point of `at`.
"""
function interpolated(nodes, values, at)
    return map(at) do point
        i = clamp(searchsortedlast(nodes, point), 1, length(nodes) - 1)
        weight = (point - nodes[i]) / (nodes[i+1] - nodes[i])
        (1 - weight) * values[i] + weight * values[i+1]
    end
end

"""
    extremum(nodes, values, pick)

The extreme value that `pick` (`findmin` or `findmax`) finds in `values`, and where it is, refined by a parabola through
the three nodes around it.
"""
function extremum(nodes, values, pick)
    _, i = pick(values)
    a, b, c = nodes[i-1], nodes[i], nodes[i+1]
    fa, fb, fc = values[i-1], values[i], values[i+1]
    denominator = (a - b) * (a - c) * (b - c)
    α = (c * (fb - fa) + b * (fa - fc) + a * (fc - fb)) / denominator
    β = (c^2 * (fa - fb) + b^2 * (fc - fa) + a^2 * (fb - fc)) / denominator
    γ = fb - α * b^2 - β * b
    position = -β / 2α

    return α * position^2 + β * position + γ, position
end

"""
    report(name, nodes, simulated, table_nodes, table_values; relative = false)

Print the largest and the root-mean-square difference between the profile `simulated` at `nodes` and a published table.
"""
function report(name, nodes, simulated, table_nodes, table_values; relative = false)
    difference = interpolated(nodes, simulated, table_nodes) .- table_values
    difference = relative ? difference ./ abs.(table_values) : difference
    largest, i = findmax(abs, difference)
    units = relative ? 100 : 1
    rms = sqrt(sum(abs2, difference) / length(difference))

    @printf("  %-34s largest %8.4f%s at %.4f, rms %8.4f%s\n",
            name,
            units * largest,
            relative ? "%" : " ",
            table_nodes[i],
            units * rms,
            relative ? "%" : " "
    )

    return nothing
end

println("Simulated profiles against the tables, in units of U and U/L, or relative where marked with %:")

for (; Re) in runs
    flow = flows[Re]
    println("Re = $Re, $(flow.N)² cells:")
    report("u, Ghia et al.", flow.y_centers, flow.u_profile, ghia_y, ghia_u[Re])
    report("v, Ghia et al.", flow.x_centers, flow.v_profile, ghia_x, ghia_v[Re])
    report("ω on the lid, Ghia et al.", flow.lid_x, flow.ω_lid, ghia_lid_x, ghia_lid[Re]; relative = true)

    if Re == 100
        @printf("  ω at the center: %.4f, Botella and Peyret %.4f\n",
                interpolated(flow.corners, flow.ω_vertical, [0.5])[1],
                botella_center_ω_at_100
        )

        @printf("  ω on the lid at x = L/2: %.4f, Botella and Peyret %.4f\n",
                interpolated(flow.lid_x, flow.ω_lid, [0.5])[1],
                botella_lid_ω_at_100
        )
    end

    if Re == 1000
        report("u, Botella and Peyret", flow.y_centers, flow.u_profile, botella_y, botella_u)
        report("v, Botella and Peyret", flow.x_centers, flow.v_profile, botella_x, botella_v)
        report("ω vertical, Botella and Peyret", flow.corners, flow.ω_vertical, botella_y, botella_ω_vertical)
        report("ω horizontal, Botella and Peyret", flow.corners, flow.ω_horizontal, botella_x, botella_ω_horizontal)
        report("ω on the lid, Botella and Peyret", flow.lid_x, flow.ω_lid, botella_lid_x, botella_lid_ω;
                relative = true
        )
    end

    if haskey(erturk_u, Re)
        report("u, Erturk et al.", flow.y_centers, flow.u_profile, erturk_y, erturk_u[Re])
        report("v, Erturk et al.", flow.x_centers, flow.v_profile, erturk_x, erturk_v[Re])
    end

    @printf("  extrema: u = %.5f at y = %.4f, v = %.5f at x = %.4f, v = %.5f at x = %.4f\n",
            extremum(flow.y_centers, flow.u_profile, findmin)...,
            extremum(flow.x_centers, flow.v_profile, findmin)...,
            extremum(flow.x_centers, flow.v_profile, findmax)...
    )

    @printf("  ω ranges: vertical %.2f to %.2f, horizontal %.2f to %.2f\n",
            extrema(flow.ω_vertical)...,
            extrema(flow.ω_horizontal)...
    )
end

println("Botella and Peyret's extrema at Re = 1000, mirrored: u = -0.38857 at y = 0.1717, v = -0.52708 at x = 0.9092, ",
        "v = 0.37694 at x = 0.1578")

#####
##### The figure
#####

ink = "#0b0b0b"
secondary_ink = "#52514e"
axis_color = "#c3c2b7"

set_theme!(
        fontsize = 46,
        Axis = (;
                titlefont = :regular,
                titlesize = 46,
                titlegap = 12,
                xlabelsize = 52,
                ylabelsize = 52,
                xticklabelsize = 44,
                yticklabelsize = 44,
                ylabelpadding = 16,
                xgridvisible = false,
                ygridvisible = false,
                leftspinecolor = axis_color,
                rightspinecolor = axis_color,
                topspinecolor = axis_color,
                bottomspinecolor = axis_color,
                xtickcolor = axis_color,
                ytickcolor = axis_color
        )
)

fig = Figure(size = (3560, 1840), figure_padding = (20, 44, 16, 16), backgroundcolor = :white)

heatmap_layout = fig[1, 1] = GridLayout()
profile_layout = fig[1, 2] = GridLayout()

#####
##### The heatmaps
#####

panel_pixels = 460

velocity_colors = (colormap = :balance, colorrange = (-1, 1))

centerline = (color = (:black, 0.5), linewidth = 7)

vertical_centerline!(ax) = lines!(ax, [0.5, 0.5], [0, 1]; centerline...)
horizontal_centerline!(ax) = lines!(ax, [0, 1], [0.5, 0.5]; centerline...)

for (column, title) in enumerate((L"u / U", L"v / U", L"\omega L / U"))
    Label(heatmap_layout[1, column+2], title; fontsize = 56)
end

position_ticks = ([0, 0.5, 1], [L"0", L"L/2", L"L"])

# Only the bottom row labels its 0, which would otherwise run into the L of the row below.
upper_row_ticks = ([0.5, 1], [L"L/2", L"L"])

heatmaps = map(enumerate(runs)) do (row, (; Re, color))
    flow = flows[Re]

    Label(heatmap_layout[row+1, 1], "Re = $Re"; rotation = π / 2, fontsize = 50, tellheight = false)
    Box(heatmap_layout[row+1, 2]; color, strokevisible = false, width = 14, height = panel_pixels)

    # The vorticity's profile is along the lid, the top edge of its panel, so it needs no line.
    columns = (
            (flow.u_field, velocity_colors, identity, vertical_centerline!),
            (flow.v_field, velocity_colors, identity, horizontal_centerline!),
            (flow.ω_field, vorticity_colors, vorticity_scale, identity)
    )

    map(enumerate(columns)) do (column, (field, colors, scale, mark_profile!))
        ax = Axis(heatmap_layout[row+1, column+2];
                xticks = position_ticks,
                yticks = row == length(runs) ? position_ticks : upper_row_ticks,
                xticklabelsvisible = row == length(runs),
                yticklabelsvisible = column == 1,
                yticklabelspace = column == 1 ? 90.0 : 0.0,
                width = panel_pixels,
                height = panel_pixels,
                limits = ((0, 1), (0, 1))
        )

        cells = size(field, 1)
        centers = range(0.5 / cells, 1 - 0.5 / cells, length = cells)

        field_heatmap = heatmap!(ax, centers, centers, scale.(field); colors..., interpolate = true)
        mark_profile!(ax)

        field_heatmap
    end
end

velocity_heatmap, _, vorticity_heatmap = last(heatmaps)

Colorbar(heatmap_layout[5, 3:4], velocity_heatmap;
        vertical = false,
        flipaxis = false,
        ticks = labelled_ticks([-1.0, -0.5, 0.0, 0.5, 1.0]),
        ticklabelsize = 44,
        height = 34
)

# The asinh scale crowds the values near zero, so only some of the ticks carry a label.
Colorbar(heatmap_layout[5, 5], vorticity_heatmap;
        vertical = false,
        flipaxis = false,
        ticks = vorticity_ticks([-30, -3, 0, 3, 30]),
        minorticks = vorticity_scale.([-10, -1, 1, 10]),
        minorticksvisible = true,
        minorticksize = 12,
        ticklabelsize = 44,
        height = 34
)

colgap!(heatmap_layout, 1, 4)
colgap!(heatmap_layout, 2, 12)
colgap!(heatmap_layout, 3, 34)
colgap!(heatmap_layout, 4, 76)
rowgap!(heatmap_layout, 1, 8)
rowgap!(heatmap_layout, 2, 14)
rowgap!(heatmap_layout, 3, 14)
rowgap!(heatmap_layout, 4, 26)

#####
##### The profiles
#####

unit_ticks = labelled_ticks([0, 0.5, 1.0])

# The vorticity along the lid is on the same scale as the colors of the vorticity.
vorticity_axis_scale = Makie.ReversibleScale(vorticity_scale, x -> 2sinh(x);
        limits = (-30.0, 30.0),
        interval = (-Inf, Inf)
)

# Thick enough to still read as black when Wikipedia shows the figure at a quarter of this size.
black_frame = (;
        leftspinecolor = :black,
        rightspinecolor = :black,
        topspinecolor = :black,
        bottomspinecolor = :black,
        spinewidth = 3,
        xtickcolor = :black,
        ytickcolor = :black,
        xtickwidth = 3,
        ytickwidth = 3,
        xticksize = 14,
        yticksize = 14
)

u_axis = Axis(profile_layout[1, 1];
        black_frame...,
        title = "Along the vertical centerline",
        xlabel = L"u / U",
        ylabel = L"y / L",
        xticks = labelled_ticks([-0.5, 0.0, 0.5, 1.0]),
        yticks = unit_ticks,
        limits = ((-0.6, 1.05), (-0.03, 1.03))
)

v_axis = Axis(profile_layout[1, 2];
        black_frame...,
        title = "Along the horizontal centerline",
        xlabel = L"x / L",
        ylabel = L"v / U",
        xticks = unit_ticks,
        yticks = labelled_ticks([-0.5, 0.0, 0.5]),
        limits = ((-0.03, 1.03), (-0.68, 0.56))
)

ω_lid_axis = Axis(profile_layout[2, 1];
        black_frame...,
        title = "Along the lid",
        xlabel = L"x / L",
        ylabel = L"\omega L / U",
        xticks = unit_ticks,
        yscale = vorticity_axis_scale,
        yticks = labelled_ticks([-300, -100, -30, -10]),
        limits = ((-0.03, 1.03), (-400, -3))
)

for (; Re, color) in runs
    flow = flows[Re]
    style = (; color, linewidth = 5)
    lines!(u_axis, flow.u_profile, flow.y_centers; style...)
    lines!(v_axis, flow.x_centers, flow.v_profile; style...)
    lines!(ω_lid_axis, flow.lid_x, flow.ω_lid; style...)
end

# The squares sit on the curves at Re = 1000 and 5000, so their color has to stand out from both, and from the black
# diamonds. This one passes the palette's checks against all three curves.
square = (marker = :rect, markersize = 24, color = "#c44fa0", strokecolor = :white, strokewidth = 2.5)
circle(color) = (marker = :circle, markersize = 22, color = color, strokecolor = :white, strokewidth = 2.5)
diamond = (marker = :diamond, markersize = 34, color = :transparent, strokecolor = ink, strokewidth = 3)

for (; Re) in runs
    if haskey(erturk_u, Re)
        scatter!(u_axis, erturk_u[Re], erturk_y; square...)
        scatter!(v_axis, erturk_x, erturk_v[Re]; square...)
    end
end

for (; Re, color) in runs
    scatter!(u_axis, ghia_u[Re], ghia_y; circle(color)...)
    scatter!(v_axis, ghia_x, ghia_v[Re]; circle(color)...)
    scatter!(ω_lid_axis, ghia_lid_x, ghia_lid[Re]; circle(color)...)
end

scatter!(u_axis, botella_u, botella_y; diamond...)
scatter!(v_axis, botella_x, botella_v; diamond...)
scatter!(ω_lid_axis, [botella_lid_x; 0.5], [botella_lid_ω; botella_lid_ω_at_100]; diamond...)

Legend(profile_layout[2, 2],
        [
                [LineElement(; color, linewidth = 6) for (; color) in runs],
                [
                        MarkerElement(color = secondary_ink, marker = :circle, markersize = 22),
                        MarkerElement(color = :transparent, marker = :diamond, markersize = 34,
                                strokecolor = ink, strokewidth = 3),
                        MarkerElement(; square...)
                ]
        ],
        [
                ["Re = $Re" for (; Re) in runs],
                ["Ghia et al. (1982)", "Botella & Peyret (1998)", "Erturk et al. (2005)"]
        ],
        ["Simulated", "Published"];
        framevisible = false,
        titlefont = :regular,
        titlesize = 48,
        labelsize = 46,
        patchsize = (64, 36),
        rowgap = 10,
        groupgap = 34,
        titlehalign = :left,
        gridshalign = :left,
        halign = :left,
        tellwidth = false,
        tellheight = false
)

colgap!(profile_layout, 44)
rowgap!(profile_layout, 1, 22)
colgap!(fig.layout, 56)

# A PNG, because the heatmaps are raster images, which Commons discourages embedding in an SVG. At this size the text
# and lines stay sharp at any size Wikipedia shows them.
path = joinpath(figure_directory, "Lid-driven cavity velocity and vorticity profiles.png")
save(path, fig; px_per_unit = 1)

# oxipng makes Cairo's PNG about a third smaller, without changing a pixel.
isnothing(Sys.which("oxipng")) || run(`oxipng --quiet --opt 4 --strip safe $path`)

@info @sprintf("Wrote %s, %.0f kB", basename(path), filesize(path) / 1000)
