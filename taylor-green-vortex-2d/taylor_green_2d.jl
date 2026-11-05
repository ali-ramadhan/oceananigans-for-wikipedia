# The two-dimensional Taylor–Green vortex at Re = 100, an exact solution of the Navier–Stokes equations. This script
# simulates the flow, checks it against the exact solution, and makes two animations: the vorticity with streamlines and
# fluid particles, and all five scalar fields.
#
#     julia --project taylor_green_2d.jl

include("taylor_green.jl")

Re = 100
U₀ = 1
k = 1
filename = output_file("taylor_green_2d.jld2")

simulation = taylor_green_simulation(; Re, U₀, k, N = 256, stop_time = 100)

model = simulation.model
u, v, _ = model.velocities
p = model.pressures.pNHS

simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, p, model.particles);
        filename,
        schedule = TimeInterval(0.25),
        overwrite_files = true
)

run!(simulation)

ut = FieldTimeSeries(filename, "u")
vt = FieldTimeSeries(filename, "v")
pt = FieldTimeSeries(filename, "p")
times = ut.times
trajectories = particle_trajectories(filename)

#####
##### Check the simulation against the exact solution
#####

ν = U₀ / (k * Re)
F(t) = exp(-2ν * k^2 * t)
t = times[end]

exact_vorticity(x, y) = 2k * U₀ * sin(k * x) * sin(k * y) * F(t)
exact_pressure(x, y) = U₀^2 / 4 * (cos(2k * x) + cos(2k * y)) * F(t)^2

# The exact fields have zero mean, and the simulated pressure is only defined up to a constant.
function maximum_error(field, exact)
    xs, ys, _ = nodes(field)
    simulated = interior(field, :, :, 1)
    return maximum(abs, simulated .- mean(simulated) .- [exact(x, y) for x in xs, y in ys])
end

vorticity_error = maximum_error(vorticity(ut[end], vt[end]), exact_vorticity) / (2k * U₀ * F(t))
pressure_error = maximum_error(pt[end], exact_pressure) / (U₀^2 / 2 * F(t)^2)

mean_square(field) = mean(abs2, interior(field))
kinetic_energy = [(mean_square(ut[n]) + mean_square(vt[n])) / 2 for n in eachindex(times)]
exact_kinetic_energy = @. U₀^2 / 4 * F(times)^2
kinetic_energy_error = maximum(abs, kinetic_energy ./ exact_kinetic_energy .- 1)

@info @sprintf("Relative errors at t = %d: vorticity %.1e, pressure %.1e; maximum kinetic energy error %.1e",
        t,
        vorticity_error,
        pressure_error,
        kinetic_energy_error
)

#####
##### Animation of the vorticity, streamlines and fluid particles
#####

animation_title(frame) = @lift @sprintf("2D Taylor–Green vortex, Re = %d:   t = %5.1f", Re, times[$frame])

fig = Figure(size = (1080, 1080))
frame = Observable(1)
Label(fig[0, 1:2], animation_title(frame))

ax = domain_axis(fig[1, 1], ut.grid;
        size = 840,
        title = "Vorticity, streamlines and fluid particles"
)

vorticity_heatmap = vorticity_panel!(ax, ut, vt, trajectories, frame; U₀, k)
Colorbar(fig[1, 2], vorticity_heatmap; label = "ω / kU₀")

save_animation(fig, frame, times, "taylor_green_2d"; still_time = 20)

#####
##### Animation of all five fields
#####

fig = Figure()
frame = Observable(1)
Label(fig[0, 1], animation_title(frame))
top = fig[1, 1] = GridLayout()
bottom = fig[2, 1] = GridLayout(tellwidth = false)  # centered under the top row

function field_panel!(position, title, field, initial_maximum, label)
    ax = domain_axis(position[1, 1], ut.grid;
            title,
            size = 440,
            ticks = (0:π:2π, ["0", "π", "2π"])
    )

    field_heatmap = heatmap!(ax, @lift(cell_centered(field($frame)));
            colormap = :balance,
            colorrange = (-initial_maximum, initial_maximum)
    )

    Colorbar(position[1, 2], field_heatmap; label)
end

field_panel!(top[1, 1], "Velocity u", n -> ut[n], U₀, "u / U₀")
field_panel!(top[1, 2], "Velocity v", n -> vt[n], U₀, "v / U₀")
field_panel!(top[1, 3], "Pressure p", n -> pt[n], U₀^2 / 2, "p / ρU₀²")
field_panel!(bottom[1, 1], "Streamfunction ψ", n -> streamfunction(ut[n], vt[n]), U₀ / k, "kψ / U₀")
field_panel!(bottom[1, 2], "Vorticity ω", n -> vorticity(ut[n], vt[n]), 2k * U₀, "ω / kU₀")

resize_to_layout!(fig)
save_animation(fig, frame, times, "taylor_green_2d_fields"; still_time = 20)
