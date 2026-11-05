# The three-dimensional Taylor–Green vortex at Re = 1600, animated from the velocity fields that simulate.jl saves, in
# several styles, each into its own video.
#
# STYLES picks some of the styles (default all of them). TIMES, a list of times such as "0,4,9", renders stills at those
# times instead of videos. GLMakie renders with OpenGL, so it needs a display. In a terminal without one, point DISPLAY
# (and XAUTHORITY) at an X server; its windows stay hidden.
#
#     julia --project --threads=auto animate_3d.jl

include("taylor_green.jl")
include("video_encoding.jl")

using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using GLMakie
using GLMakie: GeometryBasics
using MarchingCubes
using Statistics

Makie.MathTeXEngine.set_texfont_family!(Makie.MathTeXEngine.FontFamily("TeXGyreHeros"))
GLMakie.activate!(visible = false)

Re = 1600
V₀ = 1
L = 1
extent = (-π * L, π * L)

styles = Symbol.(split(get(ENV, "STYLES", "strongest,level,q,volume,annotated"), ","))
still_times = parse.(Float64, filter(!isempty, split(get(ENV, "TIMES", ""), ",")))

# Levels of the surfaces, and ranges of the colors, in units of V₀/L (V₀²/L² for Q).
vorticity_level = 7
q_level = 10
magnitude_range = (0, 20)
volume_range = (0.5, 30)

# The camera is fixed for the whole video.
azimuth = 0.3π
elevation = π / 9
toward_camera = Vec3f(cos(elevation) * cos(azimuth), cos(elevation) * sin(azimuth), sin(elevation))

#####
##### Fields
#####

velocity_series(name) = FieldTimeSeries(output_file("taylor_green_3d.jld2"), name; backend = OnDisk())

ut = velocity_series("u")
vt = velocity_series("v")
wt = velocity_series("w")
times = ut.times

grid = on_architecture(CUDA.functional() ? GPU() : CPU(), ut.grid)
N = size(grid, 1)

u = XFaceField(grid)
v = YFaceField(grid)
w = ZFaceField(grid)
ωx, ωy, ωz = vorticity(u, v, w)

vorticity_magnitude = Field(@at (Center, Center, Center) sqrt(ωx^2 + ωy^2 + ωz^2))
vertical_vorticity = Field(@at (Center, Center, Center) ωz)

# Q = -½ ∂ⱼuᵢ ∂ᵢuⱼ, half the difference between the squared rotation and strain rates.
q_criterion = Field(@at (Center, Center, Center) -(
        ∂x(u)^2 + ∂y(v)^2 + ∂z(w)^2 +
        2 * ∂y(u) * ∂x(v) + 2 * ∂z(u) * ∂x(w) + 2 * ∂z(v) * ∂y(w)
) / 2)

# A 1-2-1 filter in each direction, which removes the grid-scale facets that marching cubes leaves on thin sheets.
function smoothed(A)
    for dimension in 1:3
        shift = ntuple(i -> Int(i == dimension), 3)
        A = (circshift(A, shift) .+ 2 .* A .+ circshift(A, .-shift)) ./ 4
    end

    return A
end

"""
    load_frame!(n)

Load output `n` and return the vorticity magnitude, the vertical component of the vorticity direction, and the
Q-criterion at the cell centers, smoothed and in single precision on the CPU.
"""
function load_frame!(n)
    set!(u, ut[n])
    set!(v, vt[n])
    set!(w, wt[n])
    fill_halo_regions!((u, v, w))

    compute!(vorticity_magnitude)
    compute!(vertical_vorticity)
    compute!(q_criterion)

    magnitude = copy(interior(vorticity_magnitude))
    direction = copy(interior(vertical_vorticity)) ./ max.(magnitude, floatmin(Float64))
    q = copy(interior(q_criterion))

    return map(field -> Array{Float32}(smoothed(field)), (; magnitude, direction, q))
end

#####
##### Isosurfaces
#####

# The fields are periodic, so the first plane in each direction is repeated at the end to close the surface.
periodic_extension(A) = A[[1:end; 1], [1:end; 1], [1:end; 1]]

lerp(a, b, t) = (1 - t) * a + t * b

function trilinear_interpolation(A, p)
    i, j, k = clamp.(floor.(Int, p), 1, size(A) .- 1)
    tx, ty, tz = p .- (i, j, k)

    @inbounds begin
        near_bottom = lerp(A[i, j, k], A[i+1, j, k], tx)
        far_bottom = lerp(A[i, j+1, k], A[i+1, j+1, k], tx)
        near_top = lerp(A[i, j, k+1], A[i+1, j, k+1], tx)
        far_top = lerp(A[i, j+1, k+1], A[i+1, j+1, k+1], tx)
    end

    return lerp(lerp(near_bottom, far_bottom, ty), lerp(near_top, far_top, ty), tz)
end

marching_cubes = MC(zeros(Float32, N + 1, N + 1, N + 1), Int32; normal_sign = -1)
Δx = 2π * L / N
x₁ = -π * L + Δx / 2

"""
    isosurface(field, isovalue, color)

The surface where `field` equals `isovalue`, as a mesh, and `color` interpolated to its vertices, or `nothing` if
there is no such surface.
"""
function isosurface(field, isovalue, color)
    marching_cubes.vol[] = periodic_extension(field)

    # MarchingCubes.jl only overwrites the vertex indices of edges the surface crosses, so clear the last surface's.
    fill!(marching_cubes.vert_indices, 0)
    march(marching_cubes, isovalue)
    isempty(marching_cubes.triangles) && return nothing

    extended_color = periodic_extension(color)
    colors = Vector{Float32}(undef, length(marching_cubes.vertices))

    Threads.@threads for n in eachindex(colors)
        colors[n] = trilinear_interpolation(extended_color, marching_cubes.vertices[n] .+ 1)
    end

    # Marching cubes returns vertices in index space, starting from zero at the first cell center.
    points = Point3f[x₁ .+ Δx .* vertex for vertex in marching_cubes.vertices]
    faces = [GeometryBasics.TriangleFace(triangle...) for triangle in marching_cubes.triangles]
    normals = Vec3f.(marching_cubes.normals)

    return GeometryBasics.Mesh(points, faces; normal = normals), colors
end

#####
##### Styles
#####

direction_colorbar = (
        label = L"Direction of the vortex lines, $\omega_z / |\omega|$",
        colormap = :balance,
        limits = (-1, 1),
        ticks = ([-1, 0, 1], ["−1 (down)", "0", "1 (up)"])
)

magnitude_label = L"Vorticity $|\omega| L / V_0$"

"""
    style_figure(subtitle; colorbar, background = :white)

A figure with a title, a `subtitle`, a 3D axis that the cube fills, and a horizontal colorbar. Returns the figure, an
observable time for the title, and the axis.
"""
function style_figure(subtitle; colorbar, background = :white)
    foreground = background == :white ? :black : :white
    time_shown = Observable(0.0)
    title = @lift @sprintf("3D Taylor–Green vortex, Re = %d:   t = %5.2f L/V₀", Re, $time_shown)

    fig = Figure(size = (1080, 1080), backgroundcolor = background, fontsize = 22)
    Label(fig[1, 1], title; fontsize = 28, color = foreground, tellwidth = false)
    Label(fig[2, 1], subtitle; color = foreground, tellwidth = false)

    ax = Axis3(fig[3, 1];
            aspect = :data,
            viewmode = :fitzoom,
            azimuth,
            elevation,
            perspectiveness = 0.3,
            protrusions = 0,
            backgroundcolor = background
    )

    hidedecorations!(ax)
    hidespines!(ax)

    corners = [Point3f(x, y, z) for x in extent, y in extent, z in extent]
    edges = [(corners[i], corners[j]) for i in eachindex(corners), j in eachindex(corners)
             if i < j && count(corners[i] .!= corners[j]) == 1]
    linesegments!(ax, edges; color = (foreground, 0.35), linewidth = 1.5)

    Colorbar(fig[4, 1];
            vertical = false,
            flipaxis = false,
            width = Relative(0.5),
            height = 16,
            labelcolor = foreground,
            tickcolor = foreground,
            ticklabelcolor = foreground,
            colorbar...
    )

    rowgap!(fig.layout, 6)

    return fig, time_shown, ax
end

# Each style returns its figure and two functions: `update!(fields, t)` draws the frame at time `t`, and
# `pause!(t, write_frame)`, called after that frame is written, may write more frames before the video goes on.
no_pause(t, write_frame) = nothing

"""
    surface_style(find_surface, subtitle; colorbar, colorrange = colorbar.limits)

A style that draws the surface and vertex colors returned by `find_surface(fields)`, in the colormap of `colorbar`.
"""
function surface_style(find_surface, subtitle; colorbar, colorrange = colorbar.limits)
    fig, time_shown, ax = style_figure(subtitle; colorbar)
    plot = nothing

    function update!(fields, t)
        time_shown[] = t
        found = find_surface(fields)

        if isnothing(found)
            isnothing(plot) || (plot.visible = false)
            return nothing
        end

        surface, colors = found

        if isnothing(plot)
            plot = mesh!(ax, surface;
                    color = colors,
                    colormap = colorbar.colormap,
                    colorrange,
                    specular = 0.3,
                    shininess = 24
            )
        else
            Makie.update!(plot, surface; color = colors, visible = true)
        end

        return nothing
    end

    return fig, update!, no_pause
end

const strongest_cache = Ref{Any}((; fields = nothing))

"""
    strongest_surface(fields)

The 99th percentile of |ω|, estimated from every eighth cell, and the surface where |ω| equals it, as `isosurface`
returns it. The surface encloses the 1% of the volume where |ω| is largest. It is extracted once per frame and shared by
the styles that draw it.
"""
function strongest_surface(fields)
    if strongest_cache[].fields !== fields
        isovalue = quantile(vec(fields.magnitude[1:2:end, 1:2:end, 1:2:end]), 0.99)
        surface = isosurface(fields.magnitude, isovalue, fields.direction)
        strongest_cache[] = (; fields, isovalue, surface)
    end

    return strongest_cache[]
end

function strongest_subtitle(isovalue)
    threshold = @sprintf("%.1f", isovalue)
    return L"Surface enclosing the 1% of the volume where $|\omega|$ is largest ($|\omega| > %$(threshold)\,V_0/L$)"
end

function strongest_style()
    subtitle = Observable(strongest_subtitle(0))

    return surface_style(subtitle; colorbar = direction_colorbar) do fields
        (; isovalue, surface) = strongest_surface(fields)
        subtitle[] = strongest_subtitle(isovalue)
        surface
    end
end

function level_style()
    subtitle = L"Surface where $|\omega| = %$(vorticity_level)\,V_0/L$"

    return surface_style(subtitle; colorbar = direction_colorbar) do fields
        isosurface(fields.magnitude, vorticity_level * V₀ / L, fields.direction)
    end
end

function q_style()
    subtitle = L"Vortex cores, the surface where $Q = %$(q_level)\,V_0^2/L^2$"
    colorbar = (label = magnitude_label, colormap = :viridis, limits = magnitude_range)

    return surface_style(subtitle; colorbar, colorrange = magnitude_range .* (V₀ / L)) do fields
        isosurface(fields.q, q_level * V₀^2 / L^2, fields.magnitude)
    end
end

function volume_style()
    subtitle = L"Volume rendering of the vorticity magnitude $|\omega|$"

    colorbar = (
            label = magnitude_label,
            colormap = :inferno,
            limits = volume_range,
            scale = log10,
            ticks = [1, 3, 10, 30]
    )

    fig, time_shown, ax = style_figure(subtitle; colorbar, background = :black)
    data = Observable(zeros(Float32, N, N, N))

    volume!(ax, extent, extent, extent, data;
            algorithm = :absorption,
            absorption = 4f0,
            colormap = :inferno,
            colorrange = log10.(volume_range .* (V₀ / L))
    )

    function update!(fields, t)
        data[] = log10.(max.(fields.magnitude, 1f-6))
        time_shown[] = t
        return nothing
    end

    return fig, update!, no_pause
end

#####
##### The annotated video
#####

# The 1% surface pauses at each stage of the transition to turbulence while a label fades in, with an arrow to each
# feature it describes. Each target is a region of the box, as ranges of x, y or z in units of πL, and its arrow points
# at the vertex of the surface in that region nearest the camera, so that it is never hidden. The camera looks from the
# corner x = y = πL, which is in front; the corner x = πL, y = -πL is on the left of the picture.
region(; x = (-Inf, Inf), y = (-Inf, Inf), z = (-Inf, Inf)) = (; x, y, z)

stages = [
        (t = 0, side = :left, targets = [region(z = (-0.2, 0.2))],
         text = "Cores of the initial vortices.\nTheir vortex lines point up (red) or down (blue)."),

        (t = 3, side = :right, targets = [region(z = (0.2, 0.8))],
         text = "Thin vortex sheets form.\nTheir vortex lines are horizontal (white)."),

        (t = 5, side = :left, targets = [region(z = (0.2, 0.8))],
         text = "The sheets roll up into vortex tubes."),

        (t = 7, side = :left, targets = [region(x = (0, 1), y = (-1, 0), z = (0.2, 0.8)), region(z = (-0.8, -0.2))],
         text = "The tubes break down into small eddies,\nin two layers."),

        (t = 9, side = :right, targets = [region(z = (0.2, 0.8))],
         text = "The dissipation of kinetic energy peaks.\nThe strongest vorticity gathers in clusters."),

        (t = 20, side = :left, targets = [region(z = (-0.25, 0.25))],
         text = "The turbulence has spread through the box and decays.\nIts vortex lines point in all directions.")
]

fade_frames = 10

# A label stays on screen long enough to read once it has faded in, and longer at the end.
function hold_frames(stage)
    seconds = 1 + 0.22 * length(split(stage.text))
    stage === last(stages) && (seconds += 1.5)
    return round(Int, 30 * seconds)
end

in_range(x, (low, high)) = low * π * L ≤ x ≤ high * π * L
within(p, target) = in_range(p[1], target.x) && in_range(p[2], target.y) && in_range(p[3], target.z)

# Annotations are drawn with `overdraw`, or the depth test would hide them wherever they cross the surface, and in pixel
# space, since plain scenes fail to convert data coordinates.
overlay_attributes = (space = :pixel, overdraw = true)

"""
    arrow!(scene, from, to; color, halo, linewidth = 3, head = 20, gap = 4)

Draw an arrow from `from` to `gap` pixels short of `to`, over a white `halo` that keeps it visible on any background.
`from` and `to` are in pixels. Returns the plots.
"""
function arrow!(scene, from, to; color, halo, linewidth = 3, head = 20, gap = 4)
    from, to = Point2f(from), Point2f(to)
    along = (to - from) / sqrt(sum(abs2, to - from))
    across = Point2f(-along[2], along[1])
    tip = to - gap * along
    shaft_end = tip - (head - 2) * along

    function triangle(tip, height, width)
        base = tip - height * along
        return Point2f[tip, base + width / 2 * across, base - width / 2 * across]
    end

    plots = []

    for (paint, margin) in ((halo, 3), (color, 0))
        shaft = lines!(scene, Point2f[from, shaft_end];
                color = paint,
                linewidth = linewidth + 2margin,
                overlay_attributes...
        )

        arrowhead = poly!(scene, triangle(tip + margin * along, head + 2margin, 0.8head + 2margin);
                color = paint,
                overlay_attributes...
        )

        push!(plots, shaft, arrowhead)
    end

    return plots
end

function annotated_style()
    fig, update_surface!, _ = strongest_style()
    ax = content(fig[3, 1])

    # Points of the flow are projected to the screen through an invisible plot in the axis, and the labels are drawn in
    # front of the axis, in pixels of the figure.
    reference = scatter!(ax, [Point3f(0, 0, 0)]; visible = false)
    overlay = Scene(fig.scene; camera = campixel!, clear = false)

    opacity = Observable(0.0)
    ink = @lift RGBAf(0.1, 0.1, 0.1, $opacity)
    halo = @lift RGBAf(1, 1, 1, 0.8 * $opacity)

    annotation = []
    frames_to_fade_out = 0
    surface = nothing

    function update!(fields, t)
        update_surface!(fields, t)
        surface = strongest_surface(fields).surface

        if frames_to_fade_out > 0
            frames_to_fade_out -= 1
            opacity[] = frames_to_fade_out / fade_frames
        end

        return nothing
    end

    function pause!(t, write_frame)
        stage_index = findfirst(stage -> stage.t ≈ t, stages)
        (isnothing(stage_index) || isnothing(surface)) && return nothing
        stage = stages[stage_index]

        foreach(plot -> delete!(overlay, plot), annotation)
        empty!(annotation)

        label = text!(overlay, Point2f(stage.side == :left ? 40 : 1040, 905);
                text = stage.text,
                align = (stage.side, :center),
                justification = stage.side,
                fontsize = 24,
                lineheight = 1.15,
                color = ink,
                overlay_attributes...
        )

        push!(annotation, label)

        label_box = Rect2f(Makie.boundingbox(label, :pixel))
        vertices = GeometryBasics.coordinates(first(surface))
        origin = minimum(viewport(ax.scene)[])

        for target in stage.targets
            candidates = filter(p -> within(p, target), vertices)
            nearest = argmax(p -> sum(p .* toward_camera), candidates)
            tip = Makie.plot_to_screen(reference, nearest) .+ origin

            # Arrows leave from just under the label.
            start = Point2f(clamp(tip[1], minimum(label_box)[1], maximum(label_box)[1]), minimum(label_box)[2] - 8)
            append!(annotation, arrow!(overlay, start, tip; color = ink, halo))
        end

        for frame in 1:fade_frames
            opacity[] = frame / fade_frames
            write_frame()
        end

        foreach(_ -> write_frame(), 1:hold_frames(stage))
        frames_to_fade_out = fade_frames

        return nothing
    end

    return fig, update!, pause!
end

#####
##### Stills or videos
#####

constructors = (;
        strongest = strongest_style,
        level = level_style,
        q = q_style,
        volume = volume_style,
        annotated = annotated_style
)

figures = [(style, constructors[style]()...) for style in styles]

function render_still(t)
    n = argmin(abs.(times .- t))
    fields = load_frame!(n)

    for (style, fig, update!, pause!) in figures
        update!(fields, times[n])
        colorbuffer(fig)  # lays out the figure, which a pause needs to place its labels
        pause!(times[n], () -> nothing)
        save(output_file(@sprintf("taylor_green_3d_%s_t%04.1f.png", style, times[n])), fig; px_per_unit = 1)
    end

    return nothing
end

"""
    open_masters(record, figures, paths)

Open a lossless master at each of `paths` for the corresponding figure, as `open_lossless` does for one, and call
`record(write_frames)`, where `write_frames[i]()` appends figure `i` as it is now.
"""
function open_masters(record, figures, paths)
    isempty(figures) && return record(Function[])

    return open_lossless(first(figures), first(paths); supersampling = 2) do write_frame
        open_masters(figures[2:end], paths[2:end]) do write_frames
            record([write_frame; write_frames])
        end
    end
end

function render_videos()
    masters = [output_file("taylor_green_3d_$(style)_master.mkv") for (style, _, _) in figures]
    still = argmin(abs.(times .- 9))

    open_masters([fig for (_, fig, _) in figures], masters) do write_frames
        for n in eachindex(times)
            fields = load_frame!(n)

            for ((style, fig, update!, pause!), write_frame) in zip(figures, write_frames)
                update!(fields, times[n])
                write_frame()
                pause!(times[n], write_frame)
                n == still && save(output_file("taylor_green_3d_$style.png"), fig; px_per_unit = 2)
            end

            n % 20 == 1 && @info @sprintf("Frame %d of %d (t = %.2f)", n, length(times), times[n])
        end
    end

    for ((style, _, _), master) in zip(figures, masters)
        encode_webm(master, output_file("taylor_green_3d_$style.webm"))
    end

    return nothing
end

isempty(still_times) ? render_videos() : foreach(render_still, still_times)
