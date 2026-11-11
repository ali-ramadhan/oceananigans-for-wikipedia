# Draws `Taylor scraping flow in a right-angled corner.svg`: the flow near a corner where one wall slides into another,
# from the closed-form solution of Taylor, as given by Batchelor (1967, pp. 224–226). Nothing here is simulated.
#
# FIGURE_DIRECTORY (default this folder) sets where the figure goes.
#
#     julia --project scraping_flow.jl

include("svg_drawing.jl")

title = "Taylor scraping flow in a right-angled corner"

#####
##### The flow
#####

# ψ = U r f(φ), where r is the distance from the corner and φ is the angle from the sliding wall, with the stationary
# wall at φ = π/2. This is Batchelor's eq. (4.8.27), on p. 226, in units of U.
f(φ) = (-(π^2 / 4) * sin(φ) + (π / 2) * φ * sin(φ) + φ * cos(φ)) / (π^2 / 4 - 1)
df(φ) = (-(π^2 / 4) * cos(φ) + (π / 2) * (sin(φ) + φ * cos(φ)) + cos(φ) - φ * sin(φ)) / (π^2 / 4 - 1)

"""
    velocity(φ)

The velocity (u, v) in units of U, with v positive upwards. It depends only on the angle from the sliding wall, not on
the distance from the corner.
"""
function velocity(φ)
    radial = (-cos(φ), -sin(φ))
    azimuthal = (sin(φ), -cos(φ))

    return df(φ) .* radial .- f(φ) .* azimuthal
end

#####
##### The layout, in pixels
#####

width, height = 720, 724
left, top, side = 40, 104, 560
right, bottom = left + side, top + side
middle = left + side / 2
hatch_depth = 18

# The distance r is in units of the width of the panel.
point(r, φ) = (right - side * r * cos(φ), top + side * r * sin(φ))

"""
    streamline(level)

The streamline ψ = `level`, in units of U times the width of the panel, as points in pixels.
"""
function streamline(level)
    points = [point(level / f(φ), φ) for φ in range(0.002, π / 2 - 0.002, length = 4000)]

    return filter(p -> p[1] > right - 1.3side && p[2] < top + 1.3side, points)
end

levels = -0.04 .* (1:7)

# Lines through the corner, along which the velocity doesn't change.
rays = deg2rad.((12, 35, 65))

#####
##### The drawing
#####

start_drawing!(;
        width,
        height,
        title,
        description = "Streamlines of Taylor's scraping flow, the Stokes flow in a right-angled corner where one " *
                      "wall slides into a stationary one. The streamlines are scaled copies of one another, and the " *
                      "velocity is the same at every distance along a line through the corner, so it has no single " *
                      "value at the corner itself.",
        font_size = 30,
        definitions = [clip_path("panel", left, top, side, side)]
)

draw!("""<g clip-path="url(#panel)">""")

draw!("""<g fill="none" stroke="$clockwise_color" stroke-width="2.4" stroke-linejoin="round">""")
foreach(level -> draw!("""<path d="$(smooth_path(streamline(level)))"/>"""), levels)
draw!("""</g>""")

for φ in rays
    draw!(line(point(0, φ)..., point(1.42, φ)...; thickness = 1.5, color = "#8f8e85", dashes = "7 6"))
end

draw!("""</g>""")

# The velocity at two distances on each line, with lengths proportional to the speed.
arrow_length = 0.46side

for φ in rays, r in (0.36, 0.76)
    x, y = point(r, φ)
    u, v = velocity(φ) .* (arrow_length / 2)

    draw!(arrow(x - u, y + v, x + u, y - v; thickness = 3.5, head = 18))
end

# The walls.
draw!(panel_edge(left, top, side))
draw!(hatching(right, top, hatch_depth, side))
draw!(line(right, top, right, bottom; thickness = 4))
draw!(sliding_wall(left, top, side))

sliding = "Sliding wall: $(variable("u")) = $(variable("U")), $(variable("v")) = 0"
stationary = "Stationary wall: $(variable("u")) = $(variable("v")) = 0"
draw!(text(left, top - 34, sliding; anchor = "start"))
draw!(arrow(left + 430, top - 42, left + 540, top - 42; thickness = 4, head = 22))
draw!(text(right + hatch_depth + 34, (top + bottom) / 2, stationary; rotate = -90))

# The singular corner.
draw!(text(right + 66, top - 78, "Singular corner"; anchor = "end", size = 26))
draw!(line(right, top - 70, right, top - 16; thickness = 2, color = corner_color))
draw!(corner_marker(right, top - 6))

draw!(text(middle, bottom + 44, "Same velocity along each dashed line"; size = 28))

save_drawing(title)
