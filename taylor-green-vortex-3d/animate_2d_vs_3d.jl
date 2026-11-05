# The Taylor–Green vortex at Re = 1600 in two and three dimensions, side by side, until t = 60 L/V₀: the two-dimensional
# vorticity beside the vertical vorticity in the plane z = πL/4 of the three-dimensional flow. Run simulate.jl first.
#
#     julia --project animate_2d_vs_3d.jl

include("taylor_green.jl")
include("video_encoding.jl")

using CairoMakie

Makie.MathTeXEngine.set_texfont_family!(Makie.MathTeXEngine.FontFamily("TeXGyreHeros"))
set_theme!(fontsize = 22, Axis = (; titlefont = :regular))

Re = 1600
V₀ = 1
L = 1

planar_vorticity = FieldTimeSeries(output_file("taylor_green_2d.jld2"), "ωz")
vertical_vorticity = FieldTimeSeries(output_file("taylor_green_3d_slice.jld2"), "ωz")
times = planar_vorticity.times
@assert vertical_vorticity.times ≈ times "the two simulations must have the same output times"

function vorticity_panel(position, title)
    ticks = (-π:π/2:π, ["−π", "−π/2", "0", "π/2", "π"])

    return Axis(position;
            title,
            xlabel = L"x / L",
            ylabel = L"y / L",
            xticks = ticks,
            yticks = ticks,
            width = 600,
            height = 600,
            limits = ((-π * L, π * L), (-π * L, π * L))
    )
end

fig = Figure()
frame = Observable(1)

animation_title = @lift @sprintf("Taylor–Green vortex, Re = %d:   t = %5.2f L/V₀", Re, times[$frame])
Label(fig[0, 1:3], animation_title; fontsize = 28, tellwidth = false)

planar_panel = vorticity_panel(fig[1, 1], L"Two dimensions, vorticity $\omega$")
vertical_panel = vorticity_panel(fig[1, 2], L"Three dimensions, vertical vorticity $\omega_z$ at $z = \pi L/4$")
hideydecorations!(vertical_panel; ticks = false)

# Room for the tick labels π and −π where the two panels meet.
colgap!(fig.layout, 1, 48)

initial_maximum = 2V₀ / L
colorrange = (-initial_maximum, initial_maximum)

heatmap!(planar_panel, @lift(planar_vorticity[$frame]); colormap = :balance, colorrange)
vorticity_heatmap = heatmap!(vertical_panel, @lift(vertical_vorticity[$frame]); colormap = :balance, colorrange)
Colorbar(fig[1, 3], vorticity_heatmap; label = L"\omega L / V_0")

resize_to_layout!(fig)

master = record_lossless(fig, output_file("taylor_green_2d_vs_3d_master.mkv"), eachindex(times)) do n
    frame[] = n
end

encode_webm(master, output_file("taylor_green_2d_vs_3d.webm"))

frame[] = argmin(abs.(times .- 9))
save(output_file("taylor_green_2d_vs_3d.png"), fig; px_per_unit = 2)
