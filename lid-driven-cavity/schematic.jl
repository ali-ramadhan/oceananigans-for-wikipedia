# Draws the schematic of the lid-driven cavity, `Lid-driven cavity schematic.svg`: the geometry, the boundary
# conditions, the singular corners, and the eddies in the notation of Ghia, Ghia and Shin (1982). The streamlines are
# those of the steady flow at Re = 7500, which shows all of those eddies. Run simulate_steady.jl first.
#
# FIGURE_DIRECTORY (default this folder) sets where the figure goes.
#
#     julia --project schematic.jl

include("lid_driven_cavity.jl")
include("svg_drawing.jl")

using Contour

Re = 7500
N = 512

ψ = stream_function(steady_state(output_directory, Re, N).u)
nodes = range(0, 1, length = N + 1)

#####
##### The layout, in pixels
#####

width, height = 1000, 1000
left, right, top, bottom = 190, 810, 190, 810
middle = (left + right) / 2
hatch_depth = 18

pixel_x(x) = left + (right - left) * x
pixel_y(y) = bottom - (bottom - top) * y

#####
##### Streamlines
#####

"""
    streamlines(level, region)

The lines of constant ψ = `level` that lie inside `region`, a named tuple of ranges of `x` and `y`, as vectors of points
in pixels.
"""
function streamlines(level, region = (x = (0, 1), y = (0, 1)))
    curves = Vector{Vector{NTuple{2, Float64}}}()

    for streamline in lines(contour(nodes, nodes, ψ, level))
        xs, ys = coordinates(streamline)

        if all(region.x[1] .≤ xs .≤ region.x[2]) && all(region.y[1] .≤ ys .≤ region.y[2])
            push!(curves, collect(zip(pixel_x.(xs), pixel_y.(ys))))
        end
    end

    return curves
end

longest(curves) = argmax(length, curves)
curve_at(level, region) = longest(streamlines(level, region))

"""
    flow_arrow(points, x, y, color; size = 19)

An arrowhead on the streamline `points` where it passes closest to the position (`x`, `y`) in the cavity, pointing the
way the fluid moves there.
"""
function flow_arrow(points, x, y, color; size = 19)
    i = closest(points, (pixel_x(x), pixel_y(y)))
    p = points[i]

    # The velocity is (∂ψ/∂y, -∂ψ/∂x), and pixels count downwards.
    a = clamp(round(Int, (p[1] - left) / (right - left) * N) + 1, 2, N)
    b = clamp(round(Int, (bottom - p[2]) / (bottom - top) * N) + 1, 2, N)
    velocity = (ψ[a, b+1] - ψ[a, b-1], ψ[a+1, b] - ψ[a-1, b])

    return arrowhead(p..., downstream(points, i, velocity), color; size)
end

"""
    centroid(curve)

The centroid of the area inside a closed streamline, where the label of its eddy goes.
"""
function centroid(curve)
    twice_area = sum(p[1] * q[2] - q[1] * p[2] for (p, q) in zip(curve, curve[2:end]))
    x = sum((p[1] + q[1]) * (p[1] * q[2] - q[1] * p[2]) for (p, q) in zip(curve, curve[2:end]))
    y = sum((p[2] + q[2]) * (p[1] * q[2] - q[1] * p[2]) for (p, q) in zip(curve, curve[2:end]))

    return x / 3twice_area, y / 3twice_area
end

#####
##### The flow
#####

primary = [longest(streamlines(level)) for level in (-0.01, -0.03, -0.06, -0.09)]

# Each eddy is a box that holds it, and its levels. The first level is so close to zero that the streamline marks the
# edge of the eddy and runs under the lines of the walls. The primary vortex turns clockwise, like the second eddy in
# each corner, and the first eddy in each corner turns the other way.
BL₁ = (x = (0, 0.5), y = (0, 0.45))
BR₁ = (x = (0.5, 1), y = (0, 0.6))
TL₁ = (x = (0, 0.3), y = (0.5, 1))
BL₂ = (x = (0, 0.08), y = (0, 0.08))
BR₂ = (x = (0.75, 1), y = (0, 0.25))

eddies = [
        (region = BL₁, levels = [1e-7, 5e-4], color = counterclockwise_color),
        (region = BR₁, levels = [1e-7, 1.2e-3], color = counterclockwise_color),
        (region = TL₁, levels = [1e-7, 5e-4], color = counterclockwise_color),
        (region = BL₂, levels = [-1e-8], color = clockwise_color),
        (region = BR₂, levels = [-1e-7], color = clockwise_color)
]

eddy_curves = [(; eddy.color, curve)
               for eddy in eddies for level in eddy.levels for curve in streamlines(level, eddy.region)]

arrows = [
        flow_arrow(primary[2], 0.50, 0.02, clockwise_color),
        flow_arrow(primary[2], 0.97, 0.75, clockwise_color),
        flow_arrow(primary[2], 0.04, 0.55, clockwise_color),
        flow_arrow(primary[2], 0.50, 0.97, clockwise_color),
        flow_arrow(primary[3], 0.50, 0.25, clockwise_color),
        flow_arrow(primary[3], 0.85, 0.60, clockwise_color),
        flow_arrow(primary[3], 0.18, 0.55, clockwise_color),
        flow_arrow(primary[3], 0.50, 0.85, clockwise_color),
        flow_arrow(curve_at(1e-7, BL₁), 0.18, 0.13, counterclockwise_color),
        flow_arrow(curve_at(1e-7, BR₁), 0.80, 0.22, counterclockwise_color),
        flow_arrow(curve_at(1e-7, TL₁), 0.11, 0.80, counterclockwise_color),
        flow_arrow(curve_at(-1e-7, BR₂), 0.975, 0.09, clockwise_color; size = 15)
]

#####
##### The drawing
#####

start_drawing!(;
        width,
        height,
        title = "Lid-driven cavity",
        description = "Schematic of the two-dimensional lid-driven cavity. A lid moves at speed U over three " *
                      "stationary walls of a square of side L, driving a clockwise primary vortex and " *
                      "counter-rotating eddies in the corners, labelled in the notation of Ghia, Ghia and Shin " *
                      "(1982). The two upper corners are singular. The streamlines are those of the steady flow at " *
                      "Re = $Re, computed with Oceananigans.jl on $N × $N cells.",
        font_size = 28,
        definitions = [clip_path("cavity", left, top, right - left, bottom - top)]
)

# Streamlines, clipped to the cavity.
draw!("""<g clip-path="url(#cavity)" fill="none" stroke-width="2.4" stroke-linejoin="round">""")
foreach(curve -> draw!("""<path d="$(smooth_path(curve))" stroke="$clockwise_color"/>"""), primary)
foreach(eddy -> draw!("""<path d="$(smooth_path(eddy.curve))" stroke="$(eddy.color)"/>"""), eddy_curves)
draw!("""</g>""")
foreach(draw!, arrows)

# The stationary walls and the lid.
draw!(hatching(left - hatch_depth, top, hatch_depth, bottom - top + hatch_depth))
draw!(hatching(right, top, hatch_depth, bottom - top + hatch_depth))
draw!(hatching(left, bottom, right - left, hatch_depth))
draw!("""<path d="M$left $top V$bottom H$right V$top" fill="none" stroke="$ink" stroke-width="4"/>""")
draw!(sliding_wall(left, top, right - left))

# The motion of the lid.
draw!(text(middle, top - 76, "Lid: $(variable("u")) = $(variable("U")), $(variable("v")) = 0"))
draw!(arrow(middle - 130, top - 40, middle + 130, top - 40; thickness = 5, head = 26))

# The boundary conditions on the stationary walls.
no_slip = "$(variable("u")) = $(variable("v")) = 0"
draw!(text(left - hatch_depth - 14, (top + bottom) / 2, no_slip; rotate = -90))
draw!(text(right + hatch_depth + 36, (top + bottom) / 2, no_slip; rotate = -90))
draw!(text(middle, bottom + hatch_depth + 36, no_slip))

# Coordinates.
y_axis = left - 96
x_axis = bottom + 84
tick = 8
draw!(arrow(y_axis, bottom, y_axis, top - 58))
draw!(line(y_axis - tick, bottom, y_axis + tick, bottom))
draw!(line(y_axis - tick, top, y_axis + tick, top))
draw!(text(y_axis - 18, bottom + 10, "0"; anchor = "end"))
draw!(text(y_axis - 18, top + 10, variable("L"); anchor = "end"))
draw!(text(y_axis - 16, top - 42, variable("y"); anchor = "end"))
draw!(arrow(left, x_axis, right + 58, x_axis))
draw!(line(left, x_axis - tick, left, x_axis + tick))
draw!(line(right, x_axis - tick, right, x_axis + tick))
draw!(text(left, x_axis + 42, "0"))
draw!(text(right, x_axis + 42, variable("L")))
draw!(text(right + 58, x_axis + 34, variable("x"); anchor = "end"))

# The singular corners.
draw!(line(left - 66, top - 84, left - 8, top - 16; thickness = 2, color = corner_color))
draw!(line(right + 66, top - 84, right + 8, top - 16; thickness = 2, color = corner_color))
draw!(corner_marker(left, top - 6))
draw!(corner_marker(right, top - 6))
draw!(text(left - 96, top - 118, "Singular"; anchor = "start"))
draw!(text(left - 96, top - 88, "corner"; anchor = "start"))
draw!(text(right + 96, top - 118, "Singular"; anchor = "end"))
draw!(text(right + 96, top - 88, "corner"; anchor = "end"))

# The labels of the flow.
draw!(text(pixel_x(0.515), pixel_y(0.55), "Primary"))
draw!(text(pixel_x(0.515), pixel_y(0.50), "vortex"))

eddy_label(name, index) = name * subscript(index)

function eddy_name!(curve, name, index; size = 26)
    x, y = centroid(curve)
    draw!(text(x, y + 0.36size, eddy_label(name, index); size))

    return nothing
end

eddy_name!(curve_at(5e-4, BL₁), "BL", 1)
eddy_name!(curve_at(1.2e-3, BR₁), "BR", 1)
eddy_name!(curve_at(5e-4, TL₁), "TL", 1)
eddy_name!(curve_at(-1e-7, BR₂), "BR", 2; size = 22)

# The second eddy in the bottom left corner is too small for its label, which goes outside with a line to it.
draw!(text(left - hatch_depth - 44, bottom + hatch_depth + 44, eddy_label("BL", 2); size = 24))
draw!(line(left - hatch_depth - 26, bottom + hatch_depth + 22, pixel_x(0.012), pixel_y(0.012); thickness = 1.5))

save_drawing("Lid-driven cavity schematic")
