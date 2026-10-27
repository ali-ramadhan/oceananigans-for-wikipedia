# The two-dimensional Taylor–Green vortex at Re = 100, an exact solution of the Navier–Stokes equations
# (see taylor_green.jl). The script simulates the flow with Oceananigans, checks the numerical solution against
# the exact one, and makes two animations: the vorticity with streamlines and fluid particles, and all five
# scalar fields (the velocity components u and v, the pressure p, the streamfunction ψ and the vorticity ω).
#
#     julia --project taylor_green_2d.jl

using JLD2

include("taylor_green.jl")
include("video_encoding.jl")

Re = 100
N = 256
stop_time = 100
frame_interval = 0.25

output_directory = joinpath(@__DIR__, "output")
mkpath(output_directory)

simulation = taylor_green_simulation(Re; N, stop_time)
frames = record_frames!(simulation, frame_interval; primitive = true)
run!(simulation)
jldsave(joinpath(output_directory, "taylor_green_2d.jld2"); frames...)

#####
##### Check the simulation against the exact solution
#####

ν = U₀ / (k * Re)
F(t) = exp(-2ν * k^2 * t)
xᶠ = frames.x                   # cell faces and corners
xᶜ = xᶠ .+ (xᶠ[2] - xᶠ[1]) / 2  # cell centres

t = frames.t[end]
ω_exact = [2k * U₀ * sin(k * x) * sin(k * y) * F(t) for x in xᶠ, y in xᶠ]
p_exact = [U₀^2 / 4 * (cos(2k * x) + cos(2k * y)) * F(t)^2 for x in xᶜ, y in xᶜ]
ω_error = maximum(abs, frames.ω[end] .- ω_exact) / (2k * U₀ * F(t))
p_error = maximum(abs, frames.p[end] .- p_exact) / (U₀^2 / 2 * F(t)^2)
kinetic_energy = [(sum(abs2, u) + sum(abs2, v)) / (2N^2) for (u, v) in zip(frames.u, frames.v)]
E_error = maximum(abs, kinetic_energy ./ (U₀^2 / 4 .* F.(frames.t) .^ 2) .- 1)
@info @sprintf("Relative errors at t = %d: vorticity %.1e, pressure %.1e; maximum kinetic energy error %.1e",
               t, ω_error, p_error, E_error)

#####
##### Animation of the vorticity, streamlines and fluid particles
#####

set_theme!(Theme(fontsize = 22))
animation_title(n) = @lift @sprintf("2D Taylor–Green vortex, Re = %d:   t = %5.1f", Re, frames.t[$n])

fig = Figure(size = (1080, 1080))
frame = Observable(1)
Label(fig[0, 1:2], animation_title(frame), fontsize = 28, tellwidth = false)

ticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"])
ax = Axis(fig[1, 1]; title = "Vorticity, streamlines and fluid particles", xlabel = "x", ylabel = "y",
          limits = ((0, L), (0, L)), xticks = ticks, yticks = ticks)
hm = vorticity_panel!(ax, frames, frame)
Colorbar(fig[1, 2], hm; label = "ω / kU₀")
colsize!(fig.layout, 1, Fixed(840))
rowsize!(fig.layout, 1, Fixed(840))

master = record_lossless(fig, joinpath(output_directory, "taylor_green_2d_master.mkv"), eachindex(frames.t)) do i
    frame[] = i
end
encode_webm(master, joinpath(output_directory, "taylor_green_2d.webm"))

frame[] = findfirst(≥(20), frames.t)
save(joinpath(output_directory, "taylor_green_2d.png"), fig; px_per_unit = 2)

#####
##### Animation of all five fields
#####

# Each field has a fixed colour range, its initial maximum, so the panels fade as the flow decays. The pressure
# fades fastest, as F(t)².
panels = [("Velocity u", frames.u, xᶠ, xᶜ, U₀, "u / U₀"),
          ("Velocity v", frames.v, xᶜ, xᶠ, U₀, "v / U₀"),
          ("Pressure p", frames.p, xᶜ, xᶜ, U₀^2 / 2, "p / ρU₀²"),
          ("Streamfunction ψ", frames.ψ, xᶠ, xᶠ, U₀ / k, "kψ / U₀"),
          ("Vorticity ω", frames.ω, xᶠ, xᶠ, 2k * U₀, "ω / kU₀")]

fig = Figure()
frame = Observable(1)
Label(fig[0, 1], animation_title(frame), fontsize = 28, tellwidth = false)
top = fig[1, 1] = GridLayout()
bottom = fig[2, 1] = GridLayout(tellwidth = false)  # centred under the top row

ticks = (0:π:2π, ["0", "π", "2π"])
for (j, (name, data, x, y, extent, label)) in enumerate(panels)
    position = j ≤ 3 ? top[1, j] : bottom[1, j - 3]
    panel = Axis(position[1, 1]; title = name, xlabel = "x", ylabel = "y", width = 440, height = 440,
                 limits = ((0, L), (0, L)), xticks = ticks, yticks = ticks)
    field = heatmap!(panel, periodic(x), periodic(y), @lift(periodic(data[$frame])); colormap = :balance,
                     colorrange = (-extent, extent))
    Colorbar(position[1, 2], field; label)
end
resize_to_layout!(fig)
w, h = size(fig.scene)
resize!(fig, w + isodd(w), h + isodd(h))  # the video encoder needs even dimensions

master = record_lossless(fig, joinpath(output_directory, "taylor_green_2d_fields_master.mkv"), eachindex(frames.t)) do i
    frame[] = i
end
encode_webm(master, joinpath(output_directory, "taylor_green_2d_fields.webm"))

frame[] = findfirst(≥(20), frames.t)
save(joinpath(output_directory, "taylor_green_2d_fields.png"), fig; px_per_unit = 2)
