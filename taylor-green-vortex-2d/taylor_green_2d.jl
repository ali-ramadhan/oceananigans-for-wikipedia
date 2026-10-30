# The two-dimensional Taylor–Green vortex at Re = 100, an exact solution of the Navier–Stokes equations
# (see taylor_green.jl). The script simulates the flow with Oceananigans, checks the numerical solution against the
# exact one, and makes two animations: the vorticity with streamlines and fluid particles, and all five scalar fields
# (the velocity components u and v, the pressure p, the streamfunction ψ and the vorticity ω).
#
#     julia --project taylor_green_2d.jl

include("taylor_green.jl")
include("video_encoding.jl")

Re = 100
U₀ = 1
k = 1
output_directory = joinpath(@__DIR__, "output")
mkpath(output_directory)
filename = joinpath(output_directory, "taylor_green_2d.jld2")

simulation = taylor_green_simulation(; Re, U₀, k, N = 256, stop_time = 100)
model = simulation.model
u, v, w = model.velocities
p = model.pressures.pNHS
simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, p, model.particles); filename,
                                                 schedule = TimeInterval(0.25), overwrite_files = true)
run!(simulation)

ut = FieldTimeSeries(filename, "u")
vt = FieldTimeSeries(filename, "v")
pt = FieldTimeSeries(filename, "p")
times = ut.times
particles = particle_positions(filename)

#####
##### Check the simulation against the exact solution
#####

ν = U₀ / (k * Re)
F(t) = exp(-2ν * k^2 * t)
n = length(times)
t = times[n]

ω = vorticity(ut[n], vt[n])
xω, yω, _ = nodes(ω)
exact_vorticity = [2k * U₀ * sin(k * x) * sin(k * y) * F(t) for x in xω, y in yω]
vorticity_error = maximum(abs, interior(ω, :, :, 1) .- exact_vorticity) / (2k * U₀ * F(t))

xp, yp, _ = nodes(pt[n])
pressure = interior(pt[n], :, :, 1)
exact_pressure = [U₀^2 / 4 * (cos(2k * x) + cos(2k * y)) * F(t)^2 for x in xp, y in yp]
pressure_error = maximum(abs, pressure .- sum(pressure) / length(pressure) .- exact_pressure) / (U₀^2 / 2 * F(t)^2)

kinetic_energy = [(sum(abs2, interior(ut[i])) + sum(abs2, interior(vt[i]))) / 2length(interior(ut[i]))
                  for i in eachindex(times)]
kinetic_energy_error = maximum(abs, kinetic_energy ./ (U₀^2 / 4 .* F.(times) .^ 2) .- 1)

@info @sprintf("Relative errors at t = %d: vorticity %.1e, pressure %.1e; maximum kinetic energy error %.1e",
               t, vorticity_error, pressure_error, kinetic_energy_error)

#####
##### Animation of the vorticity, streamlines and fluid particles
#####

set_theme!(Theme(fontsize = 22))
animation_title(frame) = @lift @sprintf("2D Taylor–Green vortex, Re = %d:   t = %5.1f", Re, times[$frame])

fig = Figure(size = (1080, 1080))
frame = Observable(1)
Label(fig[0, 1:2], animation_title(frame), fontsize = 28, tellwidth = false)

ticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"])
ax = Axis(fig[1, 1]; title = "Vorticity, streamlines and fluid particles", xlabel = "x", ylabel = "y",
          limits = ((0, 2π / k), (0, 2π / k)), xticks = ticks, yticks = ticks)
vorticity_heatmap = vorticity_panel!(ax, ut, vt, particles, frame; U₀, k)
Colorbar(fig[1, 2], vorticity_heatmap; label = "ω / kU₀")
colsize!(fig.layout, 1, Fixed(840))
rowsize!(fig.layout, 1, Fixed(840))

master = record_lossless(fig, joinpath(output_directory, "taylor_green_2d_master.mkv"), eachindex(times)) do i
    frame[] = i
end
encode_webm(master, joinpath(output_directory, "taylor_green_2d.webm"))

frame[] = findfirst(≥(20), times)
save(joinpath(output_directory, "taylor_green_2d.png"), fig; px_per_unit = 2)

#####
##### Animation of all five fields
#####

# Each field has a fixed colour range, its initial maximum, so the panels fade as the flow decays. The pressure fades
# fastest, as F(t)².
panels = [("Velocity u", i -> ut[i], U₀, "u / U₀"),
          ("Velocity v", i -> vt[i], U₀, "v / U₀"),
          ("Pressure p", i -> pt[i], U₀^2 / 2, "p / ρU₀²"),
          ("Streamfunction ψ", i -> streamfunction(ut[i], vt[i]), U₀ / k, "kψ / U₀"),
          ("Vorticity ω", i -> vorticity(ut[i], vt[i]), 2k * U₀, "ω / kU₀")]

fig = Figure()
frame = Observable(1)
Label(fig[0, 1], animation_title(frame), fontsize = 28, tellwidth = false)
top = fig[1, 1] = GridLayout()
bottom = fig[2, 1] = GridLayout(tellwidth = false)  # centred under the top row

ticks = (0:π:2π, ["0", "π", "2π"])
for (j, (name, field, extent, label)) in enumerate(panels)
    position = j ≤ 3 ? top[1, j] : bottom[1, j - 3]
    panel = Axis(position[1, 1]; title = name, xlabel = "x", ylabel = "y", width = 440, height = 440,
                 limits = ((0, 2π / k), (0, 2π / k)), xticks = ticks, yticks = ticks)
    field_heatmap = heatmap!(panel, @lift(cell_centered(field($frame))); colormap = :balance,
                             colorrange = (-extent, extent))
    Colorbar(position[1, 2], field_heatmap; label)
end
resize_to_layout!(fig)
# The video encoder needs even dimensions.
figure_width, figure_height = size(fig.scene)
resize!(fig, figure_width + isodd(figure_width), figure_height + isodd(figure_height))

master = record_lossless(fig, joinpath(output_directory, "taylor_green_2d_fields_master.mkv"), eachindex(times)) do i
    frame[] = i
end
encode_webm(master, joinpath(output_directory, "taylor_green_2d_fields.webm"))

frame[] = findfirst(≥(20), times)
save(joinpath(output_directory, "taylor_green_2d_fields.png"), fig; px_per_unit = 2)
