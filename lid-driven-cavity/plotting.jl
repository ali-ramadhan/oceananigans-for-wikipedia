# Shared code for the animations and the figure drawn with Makie: the colors of the vorticity, a panel of the vorticity
# under the streamlines, and the videos and stills. It builds on lid_driven_cavity.jl, so include that first.

using CairoMakie
using Printf

include("video_encoding.jl")

Makie.MathTeXEngine.set_texfont_family!(Makie.MathTeXEngine.FontFamily("TeXGyreHeros"))

# TIMES, a list of times such as "1,5,30", makes the animation scripts render stills at those times instead of a video.
still_times = parse.(Float64, filter(!isempty, split(get(ENV, "TIMES", ""), ",")))

#####
##### Colors and ticks
#####

# The vorticity is about 2 U/L in the primary vortex and in the corner eddies and much larger in the thin layers around
# them, so its colors follow asinh(ω / 2): roughly linear below 2 U/L, logarithmic above it, and with both signs. They
# saturate at 30 U/L, on a range a hair wider than that, since a colorbar drops ticks that fall exactly on its ends.
vorticity_scale(ω) = asinh(ω / 2)
vorticity_limit = vorticity_scale(30)
vorticity_colors = (colormap = :curl, colorrange = (-vorticity_limit, vorticity_limit) .* (1 + 1e-6))

# Numbers as tick labels, with true minus signs and no trailing zeros.
tick_label(value) = replace(string(value), r"\.0$" => "", "-" => "−")
labelled_ticks(values) = (values, tick_label.(values))
vorticity_ticks(values) = (vorticity_scale.(values), tick_label.(values))

#####
##### Vorticity and streamlines
#####

# The primary vortex is 40 to 90 times stronger than the corner eddies, and the strongest of those about 2000 times
# stronger than the second eddy in the bottom right corner. The levels near zero mark the edges of the eddies.
streamline_levels = [-0.1, -0.07, -0.04, -0.01, -1e-4, -2e-7, 1e-5, 5e-4, 2e-3]

"""
    vorticity_panel!(ax, ω, u, frame; levels = streamline_levels, linewidth = 1.5, heatmap_attributes...)

Plot output `frame[]` of the time series of the vorticity `ω` and the velocity `u` on the axis `ax`: the vorticity, and
the streamlines at `levels` over it. A grid finer than 1024² cells is drawn at 1024², so that its thin vortex sheets
don't flicker from frame to frame. Returns the heatmap, for a colorbar.
"""
function vorticity_panel!(ax, ω, u, frame; levels = streamline_levels, linewidth = 1.5, heatmap_attributes...)
    # The vorticity is saved in single precision. Its colors are computed in double precision, as they were for the
    # videos on Wikimedia Commons: in single precision, a few pixels of each frame come out one level of 255 away.
    colors = @lift vorticity_scale.(Float64.(at_most(cell_average(interior(ω[$frame], :, :, 1)), 1024)))

    ψ = @lift begin
        ψ = stream_function(interior(u[$frame], :, :, 1))
        stride = max(1, (size(ψ, 1) - 1) ÷ 1024)
        ψ[1:stride:end, 1:stride:end]
    end

    cells = size(colors[], 1)
    centers = range(1 / 2cells, 1 - 1 / 2cells, length = cells)
    corners = range(0, 1, length = size(ψ[], 1))

    vorticity_heatmap = heatmap!(ax, centers, centers, colors; vorticity_colors..., heatmap_attributes...)
    contour!(ax, corners, corners, ψ; levels, color = :black, linewidth)

    return vorticity_heatmap
end

#####
##### Videos and stills
#####

"""
    video_times(stop; pace, constant_until, speed_up = 150, maximum_pace = Inf)

The times to show in a video that ends at `stop`. It runs at `pace` L/U per frame until `constant_until`. The pace then
grows in proportion to the time since then, exponentially with an e-folding of `speed_up` frames, up to `maximum_pace`.
"""
function video_times(stop; pace, constant_until, speed_up = 150, maximum_pace = Inf)
    times = [0.0]

    while last(times) < stop
        t = last(times)
        push!(times, min(stop, t + min(maximum_pace, pace + max(0, t - constant_until) / speed_up)))
    end

    return times
end

"""
    save_animation(show_time!, figure, times, name; framerate = 30, hold_seconds = 3, px_per_unit = 1)

Call `show_time!(t)` for each of `times` to write the lossless master `<name>_master.mkv` and the video `<name>.webm`
to the output directory. The video ends by holding the last of them for `hold_seconds`. That frame is also saved as
`<name>.png`, and it serves as the opening frame on Wikipedia.
"""
function save_animation(show_time!, figure, times, name; framerate = 30, hold_seconds = 3, px_per_unit = 1)
    shown = vcat(times, fill(last(times), hold_seconds * framerate))

    master = record_lossless(figure, output_file("$(name)_master.mkv"), eachindex(shown); framerate) do n
        show_time!(shown[n])
        n % 100 == 1 && @info @sprintf("Frame %d of %d (t = %.2f)", n, length(shown), shown[n])
    end

    encode_webm(master, output_file("$name.webm"))
    save(output_file("$name.png"), figure; px_per_unit)

    return nothing
end

"""
    save_stills(show_time!, figure, times, name)

Call `show_time!(t)` for each of `times` and save the figure as `<name>_t<t>.png` in the output directory.
"""
function save_stills(show_time!, figure, times, name)
    for t in times
        show_time!(t)
        save(output_file(@sprintf("%s_t%07.2f.png", name, t)), figure; px_per_unit = 1)
    end

    return nothing
end
