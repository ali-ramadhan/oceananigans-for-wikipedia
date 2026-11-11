# Shared code for the scripts that write SVG files directly: the colors, the document and the parts of the drawings
# that repeat, curves as smooth paths, and text, lines and arrows. Positions are in pixels, with y downwards.

using Printf

figure_directory = get(ENV, "FIGURE_DIRECTORY", @__DIR__)

clockwise_color = "#1f5f99"
counterclockwise_color = "#b5452c"
ink = "#1a1a1a"
corner_color = "#e08214"

#####
##### The document
#####

# The elements of the drawing, in the order they are painted.
drawing = String[]
draw!(element) = push!(drawing, element)

"""
    start_drawing!(; width, height, title, description, font_size, definitions = String[], links = false)

Start the drawing: its size, its `title` and `description` for screen readers and search engines, its font, the
hatching of the walls along with any other `definitions`, and a white background. `links` declares the namespace that
`<use>` elements need.
"""
function start_drawing!(; width, height, title, description, font_size, definitions = String[], links = false)
    link_namespace = links ? " xmlns:xlink=\"http://www.w3.org/1999/xlink\"" : ""

    draw!("""<?xml version="1.0" encoding="UTF-8"?>""")

    draw!("""<svg xmlns="http://www.w3.org/2000/svg"$link_namespace width="$width" height="$height" """ *
          """viewBox="0 0 $width $height" font-family="DejaVu Sans, Verdana, sans-serif" font-size="$font_size" """ *
          """fill="$ink">""")

    draw!("""<title>$title</title>""")
    draw!("""<desc>$description</desc>""")
    draw!("""<defs>""")

    draw!("""<pattern id="hatch" width="12" height="12" patternUnits="userSpaceOnUse" """ *
          """patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="12" stroke="#777777" stroke-width="2.5"/>""" *
          """</pattern>""")

    foreach(draw!, definitions)
    draw!("""</defs>""")
    draw!("""<rect width="$width" height="$height" fill="#ffffff"/>""")

    return nothing
end

"""
    save_drawing(name)

Finish the drawing and write it to `<name>.svg` in the figure directory.
"""
function save_drawing(name)
    draw!("""</svg>""")

    path = joinpath(figure_directory, "$name.svg")
    write(path, join(drawing, "\n") * "\n")

    @info @sprintf("Wrote %s, %.0f kB", basename(path), filesize(path) / 1000)

    return path
end

function clip_path(id, x, y, width, height)
    return """<clipPath id="$id"><rect x="$x" y="$y" width="$width" height="$height"/></clipPath>"""
end

# The solid behind a wall.
hatching(x, y, width, height) = """<rect x="$x" y="$y" width="$width" height="$height" fill="url(#hatch)"/>"""

# A sliding wall, drawn as a bar above the line that starts at (`x`, `y`).
sliding_wall(x, y, width) = """<rect x="$x" y="$(y - 12)" width="$width" height="12" fill="$ink"/>"""

# The edge of a square panel, where it is not a wall.
function panel_edge(x, y, side)
    return """<rect x="$x" y="$y" width="$side" height="$side" fill="none" stroke="#b8b7ad" stroke-width="1.5"/>"""
end

# The mark on a singular corner.
corner_marker(x, y) = """<circle cx="$x" cy="$y" r="11" fill="$corner_color" stroke="#ffffff" stroke-width="2"/>"""

#####
##### Curves as smooth paths
#####

distance(p, q) = hypot(p[1] - q[1], p[2] - q[2])

closest(points, target) = argmin(distance.(points, Ref(target)))

function distance_to_chord(p, a, b)
    chord = distance(a, b)
    chord == 0 && return distance(p, a)

    return abs((b[1] - a[1]) * (a[2] - p[2]) - (a[1] - p[1]) * (b[2] - a[2])) / chord
end

"""
    simplified(points, tolerance)

The indices of the points that the Ramer–Douglas–Peucker algorithm keeps, so that the curve through them stays within
`tolerance` pixels of the original.
"""
function simplified(points, tolerance)
    keep = falses(length(points))
    keep[1] = keep[end] = true
    stack = [(1, length(points))]

    while !isempty(stack)
        a, b = pop!(stack)
        b - a < 2 && continue

        deviation, offset = findmax(i -> distance_to_chord(points[i], points[a], points[b]), a+1:b-1)

        if deviation > tolerance
            index = a + offset
            keep[index] = true
            push!(stack, (a, index), (index, b))
        end
    end

    return findall(keep)
end

is_closed(points) = distance(points[1], points[end]) < 1e-6

"""
    unit_tangent(points, i; window = 3)

The direction of the curve at point `i`, from the points `window` steps on either side of it.
"""
function unit_tangent(points, i; window = 3)
    n = length(points)

    if is_closed(points)
        before = points[mod1(i - window, n - 1)]
        after = points[mod1(i + window, n - 1)]
    else
        before = points[max(1, i - window)]
        after = points[min(n, i + window)]
    end

    length_between = distance(before, after)

    return ((after[1] - before[1]) / length_between, (after[2] - before[2]) / length_between)
end

"""
    downstream(points, i, velocity)

The direction of the curve at point `i`, with the sense of the `velocity` there.
"""
function downstream(points, i, velocity)
    tangent = unit_tangent(points, i)
    sense = tangent[1] * velocity[1] + tangent[2] * velocity[2] ≥ 0 ? 1 : -1

    return sense .* tangent
end

"""
    smooth_path(points; tolerance = 0.2)

The curve through `points` as the data of an SVG path: a few cubic Bézier segments that follow it to about `tolerance`
pixels.
"""
function smooth_path(points; tolerance = 0.2)
    kept = simplified(points, tolerance)
    path = @sprintf("M%.1f %.1f", points[kept[1]]...)

    for (a, b) in zip(kept, kept[2:end])
        p, q = points[a], points[b]
        reach = distance(p, q) / 3
        tp, tq = unit_tangent(points, a), unit_tangent(points, b)

        path *= @sprintf("C%.1f %.1f %.1f %.1f %.1f %.1f",
                p[1] + reach * tp[1], p[2] + reach * tp[2],
                q[1] - reach * tq[1], q[2] - reach * tq[2],
                q[1], q[2]
        )
    end

    return is_closed(points) ? path * "Z" : path
end

#####
##### Text, lines and arrows
#####

variable(name) = """<tspan font-style="italic">$name</tspan>"""
subscript(index) = """<tspan font-size="70%" dy="0.3em">$index</tspan>"""

function text(x, y, content; anchor = "middle", size = nothing, rotate = nothing, color = nothing)
    attributes = @sprintf("""x="%.1f" y="%.1f" text-anchor="%s" """, x, y, anchor)
    isnothing(size) || (attributes *= """font-size="$size" """)
    isnothing(color) || (attributes *= """fill="$color" """)
    isnothing(rotate) || (attributes *= @sprintf("""transform="rotate(%d %.1f %.1f)" """, rotate, x, y))

    return "<text $(rstrip(attributes))>$content</text>"
end

function line(x₁, y₁, x₂, y₂; thickness = 2.5, color = ink, dashes = nothing)
    dash_pattern = isnothing(dashes) ? "" : " stroke-dasharray=\"$dashes\""

    return @sprintf("""<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="%.1f"%s/>""",
            x₁, y₁, x₂, y₂,
            color,
            thickness,
            dash_pattern
    )
end

# A filled triangle, for the heads of the arrows.
function triangle(a, b, c, color)
    return @sprintf("""<path d="M%.1f %.1fL%.1f %.1fL%.1f %.1fZ" fill="%s"/>""", a..., b..., c..., color)
end

"""
    arrow(x₁, y₁, x₂, y₂; thickness = 2.5, head = 18, color = ink)

An arrow from (`x₁`, `y₁`) to its tip at (`x₂`, `y₂`), with a head `head` pixels long.
"""
function arrow(x₁, y₁, x₂, y₂; thickness = 2.5, head = 18, color = ink)
    along = (x₂ - x₁, y₂ - y₁) ./ hypot(x₂ - x₁, y₂ - y₁)
    across = (-along[2], along[1])
    base = (x₂ - head * along[1], y₂ - head * along[2])
    half_width = 0.4head

    left = (base[1] + half_width * across[1], base[2] + half_width * across[2])
    right = (base[1] - half_width * across[1], base[2] - half_width * across[2])

    return line(x₁, y₁, base...; thickness, color) * triangle((x₂, y₂), left, right, color)
end

"""
    arrowhead(x, y, along, color; size)

An arrowhead `size` pixels long, centered on (`x`, `y`) and pointing in the direction `along`, for a streamline.
"""
function arrowhead(x, y, along, color; size)
    across = (-along[2], along[1])
    half_width = 0.36size

    tip = (x + size / 2 * along[1], y + size / 2 * along[2])
    back = (x - size / 2 * along[1], y - size / 2 * along[2])
    left = (back[1] + half_width * across[1], back[2] + half_width * across[2])
    right = (back[1] - half_width * across[1], back[2] - half_width * across[2])

    return triangle(tip, left, right, color)
end
