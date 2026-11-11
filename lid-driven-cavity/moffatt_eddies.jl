# Draws `Moffatt eddies in a right-angled corner.svg`: the sequence of eddies in a corner between two stationary walls,
# from the closed-form solution of Moffatt (1964), magnified twice to show the next ones in the sequence. Nothing here
# is simulated.
#
# FIGURE_DIRECTORY (default this folder) sets where the figure goes.
#
#     julia --project moffatt_eddies.jl

include("svg_drawing.jl")

using Contour

title = "Moffatt eddies in a right-angled corner"

#####
##### The flow
#####

# Moffatt's half-angle of the corner, and the root of his eq. (3.5) that governs the flow closest to the corner, found
# by Newton's method from the value in his Table 1.
α = π / 4

function exponent(; start = (4.30 + 1.77im) / 2α)
    μ = start

    for _ in 1:60
        μ -= (sin(2α * μ) + μ * sin(2α)) / (2α * cos(2α * μ) + sin(2α))
    end

    return 1 + μ
end

λ = exponent()
p, q = real(λ), imag(λ)

# Each eddy is this many times smaller than the last.
ρ = exp(π / q)

# Moffatt's eq. (3.9), with ξ = ln r and θ the angle from the bisector of the corner. The factor sets the sizes and
# signs, so that the eddy with ψ > 0 reaches r = 1 along the bisector and turns counterclockwise.
shape(θ) = cos(λ * θ) * cos((λ - 2) * α) - cos((λ - 2) * θ) * cos(λ * α)
factor = 1im / shape(0)
ψ(ξ, θ) = exp(p * ξ) * real(cis(q * ξ) * factor * shape(θ))

# The center of that eddy, on the bisector, and the value of ψ there.
center = exp(-atan(q / p) / q)
peak = ψ(log(center), 0.0)

# The eddies are first drawn with the corner at the origin, a panel 1000 units wide and y downwards. The corner is at
# the bottom right, and the panel reaches this far from it, in the units where the eddy reaches r = 1 along the
# bisector.
panel_units = 1000
panel_reach = 1.6
point(ξ, θ) = (panel_units * exp(ξ) / panel_reach) .* (cos(3π / 4 + θ), -sin(3π / 4 + θ))

"""
    eddy_edge()

The outer edge of the eddy, where ψ = 0, as points in the panel. It meets each wall where the shear stress on the wall
changes sign.
"""
function eddy_edge()
    half = Vector{NTuple{2, Float64}}()
    phase = angle(factor * shape(0.0))
    previous = factor * shape(0.0)

    for θ in range(0, α - 1e-4, length = 1500)
        current = factor * shape(θ)
        phase += angle(current / previous)
        previous = current
        push!(half, ((π / 2 - phase) / q, θ))
    end

    return vcat([point(ξ, -θ) for (ξ, θ) in reverse(half)], [point(ξ, θ) for (ξ, θ) in half[2:end]])
end

ξ_nodes = range(-π / q - 0.4, log(3), length = 1500)
θ_nodes = range(-α, α, length = 900)
ψ_grid = [ψ(ξ, θ) for ξ in ξ_nodes, θ in θ_nodes]

function streamlines(level)
    curves = Vector{Vector{NTuple{2, Float64}}}()

    for curve in lines(contour(ξ_nodes, θ_nodes, ψ_grid, level))
        ξs, θs = coordinates(curve)
        push!(curves, [point(ξ, θ) for (ξ, θ) in zip(ξs, θs)])
    end

    return curves
end

edge = eddy_edge()
loops = [only(streamlines(fraction * peak)) for fraction in (0.06, 0.3, 0.65)]
surrounding_lines = reduce(vcat, [streamlines(-multiple * peak) for multiple in (1, 4, 10, 22, 45)])

"""
    velocity(point)

The direction the fluid moves in at `point` in the panel, whose y runs downwards, around the eddy that turns
counterclockwise.
"""
function velocity(point)
    x, y = point[1] * panel_reach / panel_units, -point[2] * panel_reach / panel_units
    stream(a, b) = ψ(log(hypot(a, b)), atan(b, a) - 3π / 4)
    h = 1e-5

    return ((stream(x, y + h) - stream(x, y - h)) / 2h, (stream(x + h, y) - stream(x - h, y)) / 2h)
end

#####
##### The layout, in pixels
#####

width, height = 962, 692
panel_top, panel_bottom = 56, 616

main_left, main_side = 20, panel_bottom - panel_top
main_right = main_left + main_side

small_left, small_side = 680, 250
small_right = small_left + small_side
first_small_bottom = panel_top + small_side
second_small_top = panel_bottom - small_side

# The three views. Each shows one eddy, which turns counterclockwise if `turns` is 1 and clockwise if it is -1, and is
# drawn with lines `thickness` pixels wide, arrowheads `arrow_size` pixels long and walls `depth` pixels deep.
views = [
        (name = "main", left = main_left, top = panel_top, side = main_side,
         turns = 1, thickness = 2.4, arrow_size = 16, depth = 18),

        (name = "first", left = small_left, top = panel_top, side = small_side,
         turns = -1, thickness = 1.7, arrow_size = 11, depth = 12),

        (name = "second", left = small_left, top = second_small_top, side = small_side,
         turns = 1, thickness = 1.7, arrow_size = 11, depth = 12)
]

#####
##### The drawing
#####

# A group of curves that the views copy at different scales, so that they are exact copies of one another.
function shared_curves(id, curves)
    paths = ["""<path d="$(smooth_path(curve))"/>""" for curve in curves]

    return ["""<g id="$id" fill="none" stroke-linejoin="round">"""; paths; """</g>"""]
end

function scaled_copy!(id, corner_x, corner_y, scale, color, thickness)
    draw!(@sprintf(
            """<use xlink:href="#%s" transform="translate(%.1f %.1f) scale(%.6f)" stroke="%s" stroke-width="%.4f"/>""",
            id,
            corner_x,
            corner_y,
            scale,
            color,
            thickness / scale
    ))

    return nothing
end

"""
    draw_view!(view)

Draw one of the `views`: its eddy, which fills most of the panel, the fluid around it, the next two eddies of the
sequence towards the corner, arrowheads for the direction of the flow, and the two walls.
"""
function draw_view!((; name, left, top, side, turns, thickness, arrow_size, depth))
    corner_x, corner_y = left + side, top + side
    scale = side / panel_units

    # Each eddy turns the other way from the fluid around it and from the next eddy.
    own_color = turns == 1 ? counterclockwise_color : clockwise_color
    other_color = turns == 1 ? clockwise_color : counterclockwise_color

    draw!("""<g clip-path="url(#$name-panel)">""")
    scaled_copy!("surrounding", corner_x, corner_y, scale, other_color, thickness)
    scaled_copy!("eddy", corner_x, corner_y, scale, own_color, thickness)
    scaled_copy!("distant-eddy", corner_x, corner_y, scale / ρ, other_color, 0.65thickness)
    scaled_copy!("distant-eddy", corner_x, corner_y, scale / ρ^2, own_color, 0.4thickness)
    draw!("""</g>""")

    function flow_arrow!(curve, target, color)
        i = closest(curve, target)
        along = downstream(curve, i, turns .* velocity(curve[i]))
        x, y = (corner_x, corner_y) .+ scale .* curve[i]
        draw!(arrowhead(x, y, along, color; size = arrow_size))

        return nothing
    end

    # On the bisector of the corner, on either side of the center of the eddy, and in the fluid around it.
    flow_arrow!(loops[2], point(log(1.05), 0.0), own_color)
    flow_arrow!(loops[2], point(log(0.35), 0.0), own_color)
    flow_arrow!(surrounding_lines[2], point(log(1.2), 0.0), other_color)
    flow_arrow!(surrounding_lines[4], point(log(1.55), 0.0), other_color)

    draw!(panel_edge(left, top, side))
    draw!(hatching(corner_x, top, depth, side + depth))
    draw!(hatching(left, corner_y, side, depth))
    draw!("""<path d="M$corner_x $top V$corner_y H$left" fill="none" stroke="$ink" """ *
          """stroke-width="$(0.8 + thickness)"/>""")

    return nothing
end

# The box around the corner of a view, which the next view magnifies.
function magnified_corner(right, bottom, side)
    return """<rect x="$(right - side)" y="$(bottom - side)" width="$side" height="$side" """ *
           """fill="none" stroke="$ink" stroke-width="2"/>"""
end

start_drawing!(;
        width,
        height,
        title,
        description = "Streamlines of Moffatt's sequence of counter-rotating eddies, the Stokes flow in a " *
                      "right-angled corner between two stationary walls, shown at three magnifications, each about " *
                      "16 times greater than the last. Blue streamlines turn clockwise and red ones counterclockwise.",
        font_size = 30,
        links = true,
        definitions = [
                [clip_path("$name-panel", left, top, side, side) for (; name, left, top, side) in views];
                shared_curves("surrounding", surrounding_lines);
                shared_curves("eddy", [[edge]; loops]);
                shared_curves("distant-eddy", [edge, loops[2]])
        ]
)

foreach(draw_view!, views)

main_box = main_side / ρ
small_box = small_side / ρ

draw!(magnified_corner(main_right, panel_bottom, main_box))
draw!(magnified_corner(small_right, first_small_bottom, small_box))

# From each box to the view that magnifies it.
draw!(arrow(main_right + 3, panel_bottom - main_box - 3, small_left - 6, first_small_bottom + 6;
        thickness = 2,
        head = 16
))

draw!(arrow(small_right - small_box / 2, first_small_bottom + 14, small_right - small_box / 2, second_small_top - 4;
        thickness = 2,
        head = 14
))

draw!(text(small_left, panel_top - 12, "≈ 16× closer"; anchor = "start", size = 28))
draw!(text(small_left, second_small_top - 12, "≈ 270× closer"; anchor = "start", size = 28))
draw!(text(width / 2, height - 18, "Each eddy turns the other way and is about 2000 times weaker"; size = 26))

save_drawing(title)

@info @sprintf("λ = %.4f + %.4fi, so each eddy is %.2f times smaller than the last", p, q, ρ)
